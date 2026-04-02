"""
Transforms raw collector records into numeric feature vectors, one per process.
Each vector has a fixed schema defined by FEATURE_NAMES.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np

from collectors.process_collector import ProcessRecord
from collectors.network_collector import ConnectionRecord
from collectors.resource_collector import ResourceRecord

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Common process names seen on a healthy Linux desktop/server.
# Extend this set to reduce false positives in your environment.
KNOWN_PROCESS_NAMES: Set[str] = {
    "systemd", "kthreadd", "bash", "zsh", "sh", "python", "python3",
    "node", "npm", "cargo", "rustc", "gcc", "make", "cmake",
    "ssh", "sshd", "gpg-agent", "dbus-daemon", "NetworkManager",
    "dockerd", "containerd", "kubelet",
    "Xorg", "gnome-shell", "kwin_wayland", "plasmashell",
    "firefox", "chromium", "chrome",
    "vim", "nvim", "code", "cursor",
    "git", "grep", "find", "ls", "cat", "ps", "top", "htop",
    "journald", "rsyslogd", "cron", "atd",
    "postgres", "mysql", "redis-server", "nginx", "apache2",
    "pulseaudio", "pipewire", "wireplumber",
    "sudo", "su", "login", "getty",
}

# Ports considered "common" — connections to others are flagged as rare.
COMMON_PORTS: Set[int] = {
    20, 21, 22, 25, 53, 67, 68, 80, 110, 143,
    443, 465, 587, 993, 995, 3306, 5432, 6379,
    8080, 8443, 8888, 9200, 27017,
}

# Filesystem prefixes that legitimate executables rarely live in.
SUSPICIOUS_PATH_PREFIXES = (
    "/tmp/", "/dev/shm/", "/var/tmp/",
    "/run/user/",
)

# Regex to detect base64 blobs or hex strings in command lines.
_RE_BASE64 = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")
_RE_HEX    = re.compile(r"[0-9a-fA-F]{40,}")

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    # --- process (continuous) ---
    "cpu_percent_log",          # log1p(cpu_percent)
    "memory_rss_log",           # log1p(rss in MB)
    "num_threads",
    "open_files_count",
    "num_fds",
    "process_age_seconds_log",  # log1p(age in seconds)
    "cmdline_length",
    # --- process (binary) ---
    "is_unknown_name",          # 1 if name NOT in KNOWN_PROCESS_NAMES
    "exe_missing",              # 1 if exe path does not exist on disk
    "cmdline_has_encoded",      # 1 if cmdline contains b64/hex blob
    "running_from_suspicious_path",
    # --- network (per-pid aggregates) ---
    "num_connections",
    "has_external_connection",  # 1 if any raddr is non-private/non-loopback
    "is_listening",             # 1 if any connection in LISTEN state
    "rare_port",                # 1 if any remote port outside COMMON_PORTS
    # --- host-level ratios ---
    "cpu_ratio",                # proc_cpu / host_cpu  (clamped 0-1)
    "mem_ratio",                # proc_rss / host_mem  (clamped 0-1)
]

N_FEATURES = len(FEATURE_NAMES)


@dataclass
class ProcessFeatureVector:
    pid: int
    name: str
    exe: str
    cmdline: List[str]
    timestamp: float
    vector: np.ndarray      # shape (N_FEATURES,), dtype float32


# ---------------------------------------------------------------------------
# Helper: private IP detection
# ---------------------------------------------------------------------------

def _is_private_or_loopback(ip: str) -> bool:
    """Return True for loopback, link-local, and RFC-1918 addresses."""
    if not ip:
        return True
    if ip.startswith("127.") or ip == "::1":
        return True
    if ip.startswith("10."):
        return True
    if ip.startswith("192.168."):
        return True
    if ip.startswith("169.254."):
        return True
    if ip.startswith("172."):
        try:
            second = int(ip.split(".")[1])
            if 16 <= second <= 31:
                return True
        except (IndexError, ValueError):
            pass
    return False


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Combines records from all three collectors into a feature matrix.

    Usage:
        extractor = FeatureExtractor()
        vectors = extractor.extract(process_records, connection_records, resource_record)
        matrix  = extractor.to_matrix(vectors)   # np.ndarray (n_procs, N_FEATURES)
    """

    def __init__(
        self,
        known_names: Optional[Set[str]] = None,
        common_ports: Optional[Set[int]] = None,
    ) -> None:
        self._known_names  = known_names  if known_names  is not None else KNOWN_PROCESS_NAMES
        self._common_ports = common_ports if common_ports is not None else COMMON_PORTS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        processes: List[ProcessRecord],
        connections: List[ConnectionRecord],
        resource: ResourceRecord,
    ) -> List[ProcessFeatureVector]:
        """Return one ProcessFeatureVector per process record."""
        conn_index = self._index_connections(connections)
        return [
            self._extract_one(proc, conn_index, resource)
            for proc in processes
        ]

    def to_matrix(self, vectors: List[ProcessFeatureVector]) -> np.ndarray:
        """Stack vectors into a float32 matrix of shape (n, N_FEATURES)."""
        if not vectors:
            return np.empty((0, N_FEATURES), dtype=np.float32)
        return np.vstack([v.vector for v in vectors]).astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_connections(
        self, connections: List[ConnectionRecord]
    ) -> Dict[int, List[ConnectionRecord]]:
        """Group connections by PID for O(1) lookup during extraction."""
        index: Dict[int, List[ConnectionRecord]] = {}
        for c in connections:
            index.setdefault(c.pid, []).append(c)
        return index

    def _extract_one(
        self,
        proc: ProcessRecord,
        conn_index: Dict[int, List[ConnectionRecord]],
        resource: ResourceRecord,
    ) -> ProcessFeatureVector:
        conns = conn_index.get(proc.pid, [])

        # --- continuous process features ---
        cpu_percent_log        = math.log1p(max(proc.cpu_percent, 0.0))
        memory_rss_log         = math.log1p(proc.memory_rss / (1024 * 1024))  # MB
        num_threads            = float(proc.num_threads)
        open_files_count       = float(proc.open_files_count)
        num_fds                = float(proc.num_fds)
        age                    = max(proc.timestamp - proc.create_time, 0.0)
        process_age_log        = math.log1p(age)
        cmdline_str            = " ".join(proc.cmdline)
        cmdline_length         = float(len(cmdline_str))

        # --- binary process features ---
        is_unknown_name        = float(proc.name not in self._known_names)
        exe_missing            = float(bool(proc.exe) and not os.path.exists(proc.exe))
        cmdline_has_encoded    = float(
            bool(_RE_BASE64.search(cmdline_str) or _RE_HEX.search(cmdline_str))
        )
        running_from_suspicious = float(
            any(proc.exe.startswith(p) for p in SUSPICIOUS_PATH_PREFIXES)
            if proc.exe else False
        )

        # --- network features ---
        num_connections        = float(len(conns))
        has_external           = float(
            any(not _is_private_or_loopback(c.raddr_ip) for c in conns if c.raddr_ip)
        )
        is_listening           = float(any(c.status == "LISTEN" for c in conns))
        rare_port              = float(
            any(
                c.raddr_port not in self._common_ports and c.raddr_port > 0
                for c in conns
            )
        )

        # --- host-level ratio features ---
        host_cpu = resource.cpu_percent_total or 1e-6   # avoid /0
        host_mem = resource.mem_total or 1
        cpu_ratio = min(proc.cpu_percent / host_cpu, 1.0)
        mem_ratio = min(proc.memory_rss  / host_mem,  1.0)

        vector = np.array([
            cpu_percent_log,
            memory_rss_log,
            num_threads,
            open_files_count,
            num_fds,
            process_age_log,
            cmdline_length,
            is_unknown_name,
            exe_missing,
            cmdline_has_encoded,
            running_from_suspicious,
            num_connections,
            has_external,
            is_listening,
            rare_port,
            cpu_ratio,
            mem_ratio,
        ], dtype=np.float32)

        assert len(vector) == N_FEATURES, "Vector length mismatch — update FEATURE_NAMES"

        return ProcessFeatureVector(
            pid=proc.pid,
            name=proc.name,
            exe=proc.exe,
            cmdline=proc.cmdline,
            timestamp=proc.timestamp,
            vector=vector,
        )
