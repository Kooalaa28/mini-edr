"""
Alerter: consumes DetectionResult records, formats structured alerts,
and writes them as JSON lines to stdout and/or a log file.

Each alert includes the top contributing features so the analyst
has immediate context for investigation without digging into raw vectors.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import IO, List, Optional

import numpy as np

from detector.isolation_forest import DetectionResult
from features.feature_extractor import FEATURE_NAMES

# Features treated as binary flags — if set to 1.0 they are always reported.
_BINARY_FEATURES = {
    "is_unknown_name",
    "exe_missing",
    "cmdline_has_encoded",
    "running_from_suspicious_path",
    "has_external_connection",
    "is_listening",
    "rare_port",
}

# Continuous features and the threshold above which they are noteworthy.
# Values are in the same (possibly log-transformed) units as the vector.
_CONTINUOUS_THRESHOLDS = {
    "cpu_percent_log":         2.0,   # log1p(~6.4 %) — elevated but not extreme
    "memory_rss_log":          5.5,   # log1p(~244 MB)
    "num_threads":             50.0,
    "open_files_count":        100.0,
    "num_fds":                 200.0,
    "process_age_seconds_log": 0.0,   # always shown if process is very new (age ≈ 0)
    "num_connections":         20.0,
    "cpu_ratio":               0.5,
    "mem_ratio":               0.3,
}


@dataclass
class Alert:
    alert_id: str               # ISO-8601 timestamp + pid, unique enough for a log
    timestamp_utc: str
    pid: int
    name: str
    exe: str
    cmdline: List[str]
    anomaly_score: float
    top_features: List[dict]    # [{"feature": str, "value": float, "reason": str}]


class Alerter:
    """
    Formats and emits alerts for anomalous processes.

    Parameters
    ----------
    log_path:
        Path to a JSONL file where alerts are appended.
        Pass None to suppress file output.
    print_stdout:
        If True, pretty-print each alert to stdout as well.
    top_n:
        Maximum number of contributing features to include per alert.
    """

    def __init__(
        self,
        log_path: Optional[str] = None,
        print_stdout: bool = True,
        top_n: int = 5,
    ) -> None:
        self._log_path    = log_path
        self._print_stdout = print_stdout
        self._top_n       = top_n
        self._feat_index  = {name: i for i, name in enumerate(FEATURE_NAMES)}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, results: List[DetectionResult]) -> List[Alert]:
        """
        Filter flagged results, build Alert objects, and emit them.
        Returns the list of emitted alerts (empty if none were flagged).
        """
        alerts = [
            self._build_alert(r)
            for r in results
            if r.is_anomaly
        ]

        if not alerts:
            return []

        for alert in alerts:
            line = json.dumps(asdict(alert), ensure_ascii=False)
            if self._print_stdout:
                self._print(alert, line)
            if self._log_path:
                self._append_to_file(line)

        return alerts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_alert(self, result: DetectionResult) -> Alert:
        now    = datetime.now(timezone.utc)
        ts_str = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        return Alert(
            alert_id=f"{ts_str}_{result.pid}",
            timestamp_utc=ts_str,
            pid=result.pid,
            name=result.name,
            exe=result.exe,
            cmdline=result.cmdline,
            anomaly_score=round(float(result.anomaly_score), 4),
            top_features=self._explain(result.feature_vector),
        )

    def _explain(self, vector: np.ndarray) -> List[dict]:
        """
        Return up to top_n features that best explain why this process
        was flagged, ordered by severity.
        """
        hits: List[dict] = []

        # 1. Binary flags that are set
        for name in _BINARY_FEATURES:
            idx = self._feat_index.get(name)
            if idx is not None and vector[idx] == 1.0:
                hits.append({
                    "feature": name,
                    "value":   1.0,
                    "reason":  _BINARY_REASONS[name],
                })

        # 2. Continuous features above their noteworthy threshold
        for name, threshold in _CONTINUOUS_THRESHOLDS.items():
            idx = self._feat_index.get(name)
            if idx is None:
                continue
            val = float(vector[idx])
            if val > threshold:
                hits.append({
                    "feature": name,
                    "value":   round(val, 4),
                    "reason":  _continuous_reason(name, val),
                })

        # Sort binary flags first, then by value descending for continuous
        hits.sort(key=lambda h: (h["value"] != 1.0, -h["value"]))
        return hits[: self._top_n]

    def _print(self, alert: Alert, line: str) -> None:
        score_bar = "█" * int(alert.anomaly_score * 10)
        print(
            f"\n[ALERT] {alert.timestamp_utc}  pid={alert.pid}  "
            f"name={alert.name!r}  score={alert.anomaly_score:.3f} {score_bar}",
            file=sys.stderr,
        )
        if alert.exe:
            print(f"        exe  : {alert.exe}", file=sys.stderr)
        if alert.cmdline:
            print(f"        cmd  : {' '.join(alert.cmdline)}", file=sys.stderr)
        for feat in alert.top_features:
            print(
                f"        ├── {feat['feature']} = {feat['value']}  ({feat['reason']})",
                file=sys.stderr,
            )
        print(f"        json : {line}", file=sys.stderr)

    def _append_to_file(self, line: str) -> None:
        with open(self._log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Human-readable reasons
# ---------------------------------------------------------------------------

_BINARY_REASONS = {
    "is_unknown_name":              "process name not in known-good whitelist",
    "exe_missing":                  "executable no longer exists on disk (deleted binary)",
    "cmdline_has_encoded":          "command line contains base64 or hex blob",
    "running_from_suspicious_path": "executable running from /tmp, /dev/shm, or similar",
    "has_external_connection":      "active connection to a public IP",
    "is_listening":                 "process has an open listening socket",
    "rare_port":                    "remote port outside common port set",
}


def _continuous_reason(name: str, value: float) -> str:
    descriptions = {
        "cpu_percent_log":         f"high CPU (log-scaled {value:.2f})",
        "memory_rss_log":          f"high RSS memory (log-scaled {value:.2f})",
        "num_threads":             f"{int(value)} threads",
        "open_files_count":        f"{int(value)} open files",
        "num_fds":                 f"{int(value)} file descriptors",
        "process_age_seconds_log": f"very new process (log-age {value:.2f})",
        "num_connections":         f"{int(value)} network connections",
        "cpu_ratio":               f"using {value*100:.1f}% of host CPU",
        "mem_ratio":               f"using {value*100:.1f}% of host memory",
    }
    return descriptions.get(name, f"value={value:.4f}")
