#!/usr/bin/env python3
"""
Mini-EDR entry point.

Two modes:

  train   Collect a baseline of clean snapshots, fit the IsolationForest,
          and save the model to disk.

  run     Load a saved model and continuously collect telemetry, score
          every process, and emit alerts for anomalies.

Examples:
  python main.py train --snapshots 30 --interval 2 --model models/detector.joblib
  python main.py run   --model models/detector.joblib --log alerts.jsonl
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from collectors import ProcessCollector, NetworkCollector, ResourceCollector
from features import FeatureExtractor
from detector import IsolationForestDetector
from alerter import Alerter


# ---------------------------------------------------------------------------
# Shared collection helper
# ---------------------------------------------------------------------------

def collect_snapshot(
    proc_col: ProcessCollector,
    net_col: NetworkCollector,
    res_col: ResourceCollector,
    extractor: FeatureExtractor,
):
    """Run all three collectors and return (vectors, matrix)."""
    processes  = proc_col.collect()
    conns      = net_col.collect()
    resource   = res_col.collect()
    vectors    = extractor.extract(processes, conns, resource)
    matrix     = extractor.to_matrix(vectors)
    return vectors, matrix


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    print(f"[train] Collecting {args.snapshots} snapshots "
          f"every {args.interval}s — keep the system in a clean state.")

    proc_col  = ProcessCollector()
    net_col   = NetworkCollector(kind="inet")
    res_col   = ResourceCollector()
    extractor = FeatureExtractor()

    chunks: list[np.ndarray] = []

    for i in range(1, args.snapshots + 1):
        _, matrix = collect_snapshot(proc_col, net_col, res_col, extractor)
        if matrix.shape[0] > 0:
            chunks.append(matrix)
        print(f"  snapshot {i}/{args.snapshots}  ({matrix.shape[0]} processes)",
              end="\r", flush=True)
        if i < args.snapshots:
            time.sleep(args.interval)

    print()  # newline after \r progress

    if not chunks:
        print("[train] No data collected — aborting.", file=sys.stderr)
        sys.exit(1)

    baseline = np.vstack(chunks)
    print(f"[train] Baseline: {baseline.shape[0]} samples, "
          f"{baseline.shape[1]} features.")

    detector = IsolationForestDetector(
        contamination=args.contamination,
        n_estimators=args.estimators,
        threshold=args.threshold,
    )
    detector.fit(baseline)
    detector.save(args.model)
    print(f"[train] Model saved → {args.model}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    print(f"[run] Loading model from {args.model}")
    detector  = IsolationForestDetector.load(args.model)

    if args.threshold is not None:
        detector.threshold = args.threshold
        print(f"[run] Threshold overridden → {detector.threshold}")

    proc_col  = ProcessCollector()
    net_col   = NetworkCollector(kind="inet")
    res_col   = ResourceCollector()
    extractor = FeatureExtractor()
    alerter   = Alerter(
        log_path=args.log,
        print_stdout=not args.quiet,
        top_n=args.top_features,
    )

    print(f"[run] Polling every {args.interval}s  "
          f"(threshold={detector.threshold}, log={args.log or 'stdout only'})")
    print("[run] Press Ctrl-C to stop.\n")

    total_alerts = 0
    try:
        while True:
            vectors, _ = collect_snapshot(proc_col, net_col, res_col, extractor)
            results    = detector.predict(vectors)
            alerts     = alerter.process(results)
            total_alerts += len(alerts)

            if args.verbose and not alerts:
                flagged = sum(1 for r in results if r.is_anomaly)
                scores  = [r.anomaly_score for r in results]
                print(
                    f"  [{_now()}] {len(results)} processes scored — "
                    f"max={max(scores, default=0):.3f}  alerts={flagged}",
                    file=sys.stderr,
                )

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\n[run] Stopped. Total alerts emitted: {total_alerts}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mini-EDR: process anomaly detection agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- train ---------------------------------------------------------------
    tr = sub.add_parser("train", help="Collect baseline and fit the model.",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    tr.add_argument("--snapshots",     type=int,   default=30,
                    help="Number of collection snapshots to build the baseline.")
    tr.add_argument("--interval",      type=float, default=2.0,
                    help="Seconds between snapshots.")
    tr.add_argument("--contamination", type=float, default=0.01,
                    help="Expected fraction of anomalies in baseline (IsolationForest).")
    tr.add_argument("--estimators",    type=int,   default=200,
                    help="Number of trees in the IsolationForest.")
    tr.add_argument("--threshold",     type=float, default=0.8,
                    help="Anomaly score threshold for flagging (0-1).")
    tr.add_argument("--model",         type=str,   default="models/detector.joblib",
                    help="Path to save the fitted model.")

    # -- run -----------------------------------------------------------------
    ru = sub.add_parser("run", help="Load model and start monitoring.",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ru.add_argument("--model",        type=str,   default="models/detector.joblib",
                    help="Path to a saved model produced by 'train'.")
    ru.add_argument("--interval",     type=float, default=5.0,
                    help="Seconds between collection cycles.")
    ru.add_argument("--threshold",    type=float, default=None,
                    help="Override the model's saved threshold (0-1).")
    ru.add_argument("--log",          type=str,   default="alerts.jsonl",
                    help="Path to JSONL alert log file. Use '' to disable.")
    ru.add_argument("--top-features", type=int,   default=5,
                    help="Max contributing features shown per alert.")
    ru.add_argument("--quiet",        action="store_true",
                    help="Suppress human-readable stderr output.")
    ru.add_argument("--verbose",      action="store_true",
                    help="Print a status line every cycle even when no alerts fire.")

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    # Normalise empty string log path
    if hasattr(args, "log") and args.log == "":
        args.log = None

    if args.command == "train":
        cmd_train(args)
    elif args.command == "run":
        cmd_run(args)


if __name__ == "__main__":
    main()
