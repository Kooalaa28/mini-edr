"""
Shared application state for the dashboard.
All mutations must acquire `AppState.lock`.
"""
from __future__ import annotations

import json
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Set

from features.feature_extractor import KNOWN_PROCESS_NAMES, COMMON_PORTS

CONFIG_PATH = "config.json"


@dataclass
class AlertEntry:
    alert_id: str
    timestamp_utc: str
    pid: int
    name: str
    exe: str
    cmdline: list
    anomaly_score: float
    top_features: list


@dataclass
class Stats:
    cycles: int = 0
    total_alerts: int = 0
    processes_last_cycle: int = 0
    last_cycle_utc: str = ""


class AppState:
    """
    Central state object shared between the Flask routes and the monitor thread.
    All public attributes are protected by `lock`.
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()

        # Monitor thread control
        self.status: str = "stopped"          # "stopped" | "running" | "error"
        self.error_message: str = ""
        self._monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Live data
        self.alerts: Deque[AlertEntry] = deque(maxlen=200)
        self.stats = Stats()

        # Runtime config (editable from UI)
        self.model_path: str = "models/detector.joblib"
        self.threshold: Optional[float] = None   # None → use saved model value
        self.interval: float = 5.0
        self.log_path: str = "alerts.jsonl"

        # Mutable whitelists (start from built-in defaults)
        self.known_names: Set[str] = set(KNOWN_PROCESS_NAMES)
        self.common_ports: Set[int] = set(COMMON_PORTS)

        self._load_config()

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def save_config(self) -> None:
        data = {
            "model_path":   self.model_path,
            "threshold":    self.threshold,
            "interval":     self.interval,
            "log_path":     self.log_path,
            "known_names":  sorted(self.known_names),
            "common_ports": sorted(self.common_ports),
        }
        with open(CONFIG_PATH, "w") as fh:
            json.dump(data, fh, indent=2)

    def _load_config(self) -> None:
        if not os.path.exists(CONFIG_PATH):
            return
        try:
            with open(CONFIG_PATH) as fh:
                data = json.load(fh)
            self.model_path   = data.get("model_path",   self.model_path)
            self.threshold    = data.get("threshold",    self.threshold)
            self.interval     = data.get("interval",     self.interval)
            self.log_path     = data.get("log_path",     self.log_path)
            self.known_names  = set(data.get("known_names",  list(self.known_names)))
            self.common_ports = set(data.get("common_ports", list(self.common_ports)))
        except Exception:
            pass   # corrupt config — keep defaults

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------

    def set_thread(self, t: threading.Thread) -> None:
        self._monitor_thread = t

    def thread_alive(self) -> bool:
        return self._monitor_thread is not None and self._monitor_thread.is_alive()

    # ------------------------------------------------------------------
    # Serialisation helpers (called from Flask routes, lock held by caller)
    # ------------------------------------------------------------------

    def to_status_dict(self) -> Dict:
        return {
            "status":        self.status,
            "error_message": self.error_message,
            "cycles":        self.stats.cycles,
            "total_alerts":  self.stats.total_alerts,
            "processes_last_cycle": self.stats.processes_last_cycle,
            "last_cycle_utc": self.stats.last_cycle_utc,
            "model_path":    self.model_path,
            "threshold":     self.threshold,
            "interval":      self.interval,
            "log_path":      self.log_path,
        }

    def to_config_dict(self) -> Dict:
        return {
            "known_names":  sorted(self.known_names),
            "common_ports": sorted(self.common_ports),
        }
