"""
Collects a snapshot of all running processes with key attributes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import psutil


@dataclass
class ProcessRecord:
    timestamp: float
    pid: int
    ppid: int
    name: str
    exe: str
    cmdline: List[str]
    username: str
    status: str
    create_time: float
    num_threads: int
    cpu_percent: float       # % since last call (or since process start on first call)
    memory_rss: int          # bytes
    memory_vms: int          # bytes
    open_files_count: int
    num_fds: int


_ATTRS = [
    "pid", "ppid", "name", "exe", "cmdline", "username",
    "status", "create_time", "num_threads",
    "memory_info", "open_files", "num_fds",
]


class ProcessCollector:
    """
    Snapshots all running processes.

    Usage:
        collector = ProcessCollector()
        records = collector.collect()
    """

    def __init__(self) -> None:
        # First cpu_percent call returns 0.0 per psutil docs; call once to prime.
        for proc in psutil.process_iter(["pid"]):
            try:
                proc.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self._primed = True

    def collect(self) -> List[ProcessRecord]:
        """Return a list of ProcessRecord, one per live process."""
        now = time.time()
        records: List[ProcessRecord] = []

        for proc in psutil.process_iter(_ATTRS):
            try:
                info = proc.info
                mem = info.get("memory_info") or psutil.pmem(0, 0)
                records.append(ProcessRecord(
                    timestamp=now,
                    pid=info["pid"],
                    ppid=info.get("ppid") or 0,
                    name=info.get("name") or "",
                    exe=info.get("exe") or "",
                    cmdline=info.get("cmdline") or [],
                    username=info.get("username") or "",
                    status=info.get("status") or "",
                    create_time=info.get("create_time") or 0.0,
                    num_threads=info.get("num_threads") or 0,
                    cpu_percent=proc.cpu_percent(interval=None),
                    memory_rss=mem.rss,
                    memory_vms=mem.vms,
                    open_files_count=len(info.get("open_files") or []),
                    num_fds=info.get("num_fds") or 0,
                ))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        return records
