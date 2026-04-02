"""
Background monitoring loop that runs in a daemon thread.
Reads config from AppState on every cycle so whitelist changes take effect
without restarting.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from collectors import ProcessCollector, NetworkCollector, ResourceCollector
from detector import IsolationForestDetector
from features import FeatureExtractor
from dashboard.state import AlertEntry, AppState


def monitor_loop(state: AppState) -> None:
    """Entry point for the monitor thread."""
    # Snapshot config at start (model path / threshold don't change mid-run)
    with state.lock:
        model_path = state.model_path
        threshold  = state.threshold
        interval   = state.interval
        log_path   = state.log_path or None

    # Load detector
    try:
        detector = IsolationForestDetector.load(model_path)
    except Exception as exc:
        with state.lock:
            state.status = "error"
            state.error_message = f"Failed to load model '{model_path}': {exc}"
        return

    if threshold is not None:
        detector.threshold = threshold

    # Optional file logger (independent of the dashboard's in-memory store)
    file_logger = None
    if log_path:
        try:
            from alerter import Alerter
            file_logger = Alerter(log_path=log_path, print_stdout=False)
        except Exception:
            pass

    # Initialise collectors once (ProcessCollector primes cpu_percent on init)
    proc_col  = ProcessCollector()
    net_col   = NetworkCollector(kind="inet")
    res_col   = ResourceCollector()

    with state.lock:
        state.status = "running"
        state.error_message = ""

    while not state.stop_event.is_set():
        # Snapshot mutable whitelists inside lock, then release before heavy work
        with state.lock:
            known_names  = set(state.known_names)
            common_ports = set(state.common_ports)
            interval     = state.interval

        try:
            extractor = FeatureExtractor(
                known_names=known_names,
                common_ports=common_ports,
            )

            processes = proc_col.collect()
            conns     = net_col.collect()
            resource  = res_col.collect()
            vectors   = extractor.extract(processes, conns, resource)
            results   = detector.predict(vectors)

            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            flagged = [r for r in results if r.is_anomaly]

            # Build AlertEntry objects for the dashboard
            new_entries = []
            for r in flagged:
                # Reuse the Alerter's explain logic by constructing an Alert
                from alerter.alerter import Alerter as _Alerter
                _tmp = _Alerter(print_stdout=False)
                alert = _tmp._build_alert(r)
                new_entries.append(AlertEntry(
                    alert_id=alert.alert_id,
                    timestamp_utc=alert.timestamp_utc,
                    pid=alert.pid,
                    name=alert.name,
                    exe=alert.exe,
                    cmdline=alert.cmdline,
                    anomaly_score=alert.anomaly_score,
                    top_features=alert.top_features,
                ))

            # Optionally write to file
            if file_logger and flagged:
                file_logger.process(results)

            with state.lock:
                for entry in new_entries:
                    state.alerts.appendleft(entry)
                state.stats.cycles += 1
                state.stats.total_alerts += len(new_entries)
                state.stats.processes_last_cycle = len(results)
                state.stats.last_cycle_utc = now_str

        except Exception as exc:
            with state.lock:
                state.error_message = str(exc)

        state.stop_event.wait(timeout=interval)

    with state.lock:
        state.status = "stopped"
        state.error_message = ""
