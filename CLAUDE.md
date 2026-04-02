# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Mini-EDR: a Python-based Endpoint Detection and Response agent for Linux. It collects telemetry from running processes, network connections, and CPU/memory usage, then applies anomaly detection to flag suspicious activity such as malware, rare scripts, and anomalous processes.

## Environment

Always use the project venv:
```bash
source venv/bin/activate
```

To install a new dependency: add it to `requirements.txt` first, then run:
```bash
pip install -r requirements.txt
```

## Running the tools

Train a baseline model:
```bash
python main.py train --snapshots 30
```

Run the CLI monitor:
```bash
python main.py run --verbose
```

Launch the web dashboard (http://127.0.0.1:5000):
```bash
python dashboard.py
```

## Running tests

All tests:
```bash
python -m pytest
```

Single module:
```bash
python -m pytest collectors/test_collectors.py -v
python -m pytest features/test_features.py -v
python -m pytest detector/test_detector.py -v
python -m pytest alerter/test_alerter.py -v
```

## Architecture

Data flows: **Collectors → Feature Engineering → Detector → Alerter**.

### `collectors/`
Three independent collectors, each importable and testable standalone:

- `ProcessCollector` — snapshots all running processes via `psutil.process_iter`. Returns `List[ProcessRecord]` with pid, ppid, name, exe, cmdline, user, status, cpu%, rss/vms, open files, fds. Primes `cpu_percent` on `__init__` so subsequent calls return real values.
- `NetworkCollector(kind="all")` — snapshots connections via `psutil.net_connections`. Falls back to per-process iteration on `AccessDenied`. Returns `List[ConnectionRecord]`.
- `ResourceCollector` — host-level CPU and memory via `psutil`. Returns a single `ResourceRecord`.

### `features/`
`FeatureExtractor` joins all three collector outputs into a fixed 17-feature float32 vector per process. The feature schema is defined in `FEATURE_NAMES` (and `N_FEATURES`).

Key features: `cpu_percent_log`, `memory_rss_log`, `num_threads`, `open_files_count`, `num_fds`, `process_age_seconds_log`, `cmdline_length`, `is_unknown_name`, `exe_missing`, `cmdline_has_encoded`, `running_from_suspicious_path`, `num_connections`, `has_external_connection`, `is_listening`, `rare_port`, `cpu_ratio`, `mem_ratio`.

Connections are indexed by PID internally so the join is O(1) per process.

```python
extractor = FeatureExtractor()
vectors = extractor.extract(processes, connections, resource)  # List[ProcessFeatureVector]
matrix  = extractor.to_matrix(vectors)                         # np.ndarray (n, 17)
```

`KNOWN_PROCESS_NAMES` and `COMMON_PORTS` are constants in `feature_extractor.py` — extend them to reduce false positives for a specific environment.

### `detector/`
`IsolationForestDetector` wraps `StandardScaler → IsolationForest` (scikit-learn). Scores are normalised to [0, 1] (higher = more anomalous) using training min/max, then flipped. Default `threshold=0.8`.

Training is done once on a clean baseline snapshot; the fitted model is persisted with `joblib`.

```python
det = IsolationForestDetector(contamination=0.01, threshold=0.8)
det.fit(matrix)
det.save("models/detector.joblib")

det = IsolationForestDetector.load("models/detector.joblib")
results = det.predict(vectors)   # List[DetectionResult]
```

Each `DetectionResult` carries `pid`, `name`, `exe`, `cmdline`, `timestamp`, `anomaly_score`, `is_anomaly`, and `feature_vector`.

**Threshold tuning**: 0.8 is a starting point. Override at runtime with `--threshold` (CLI) or the dashboard's settings panel; the fitted value is also stored in `config.json`.

### `alerter/`
`Alerter(log_path, print_stdout, top_n)` filters flagged `DetectionResult` objects and emits structured `Alert` objects. Binary flags (e.g. `exe_missing`) are sorted first in `top_features`. Output goes to stderr (human-readable) and/or a JSONL file.

### `dashboard/`
Flask web dashboard at `http://127.0.0.1:5000`. Entry point: `dashboard.py`.

- `state.py` — `AppState` holds all shared state (status, alerts deque, stats, mutable whitelists). All mutations are protected by `AppState.lock`. Config is persisted to `config.json` on every whitelist change.
- `monitor.py` — `monitor_loop(state)` runs in a daemon thread. It snapshots `known_names` and `common_ports` from `AppState` at the start of each cycle, so whitelist edits take effect immediately without restarting.
- `app.py` — Flask routes. Key endpoints: `POST /api/start`, `POST /api/stop`, `GET /api/alerts`, `POST /api/config/names/add|remove`, `POST /api/config/ports/add|remove`.
- `templates/index.html` — single-page UI (Tailwind CDN, vanilla JS). Polls `/api/status` and `/api/alerts` every 2 seconds.

## Key design decisions

- Collectors are usable independently (no dependency on the rest of the pipeline).
- `StandardScaler` is applied before `IsolationForest` because features span very different ranges.
- Dashboard whitelist changes are applied on the next monitor cycle — no restart needed. `FeatureExtractor` is recreated each cycle (cheap) with the latest sets from `AppState`.
- `config.json` persists whitelist + runtime settings across dashboard restarts.
- Privilege: `psutil.net_connections` and reading some `/proc` entries may require root or `CAP_NET_ADMIN`.
