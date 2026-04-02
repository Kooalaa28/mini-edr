"""
Collects host-level CPU and memory snapshots.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

import psutil


@dataclass
class ResourceRecord:
    timestamp: float
    cpu_percent_total: float        # overall CPU usage %
    cpu_percent_per_core: List[float]
    cpu_count_logical: int
    cpu_freq_current: float         # MHz; 0.0 if unavailable
    mem_total: int                  # bytes
    mem_available: int              # bytes
    mem_used: int                   # bytes
    mem_percent: float
    swap_total: int                 # bytes
    swap_used: int                  # bytes
    swap_percent: float


class ResourceCollector:
    """
    Snapshots host-wide CPU and memory utilisation.

    The first collect() call may return cpu_percent_total = 0.0 because
    psutil needs an interval to compute the rate.  Subsequent calls
    (interval >= 0.1 s apart) return accurate values.

    Usage:
        collector = ResourceCollector()
        record = collector.collect()
    """

    def __init__(self) -> None:
        # Prime the CPU measurement
        psutil.cpu_percent(interval=None, percpu=False)
        psutil.cpu_percent(interval=None, percpu=True)

    def collect(self) -> ResourceRecord:
        """Return a single ResourceRecord for the current moment."""
        now = time.time()

        cpu_total = psutil.cpu_percent(interval=None, percpu=False)
        cpu_per   = psutil.cpu_percent(interval=None, percpu=True)

        freq = psutil.cpu_freq()
        cpu_freq_current = freq.current if freq else 0.0

        mem  = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return ResourceRecord(
            timestamp=now,
            cpu_percent_total=cpu_total,
            cpu_percent_per_core=cpu_per,
            cpu_count_logical=psutil.cpu_count(logical=True),
            cpu_freq_current=cpu_freq_current,
            mem_total=mem.total,
            mem_available=mem.available,
            mem_used=mem.used,
            mem_percent=mem.percent,
            swap_total=swap.total,
            swap_used=swap.used,
            swap_percent=swap.percent,
        )
