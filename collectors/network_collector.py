"""
Collects a snapshot of all active network connections via psutil.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import psutil


@dataclass
class ConnectionRecord:
    timestamp: float
    pid: int                    # 0 when the OS does not expose it
    fd: int
    family: str                 # e.g. "AF_INET", "AF_INET6", "AF_UNIX"
    type: str                   # e.g. "SOCK_STREAM", "SOCK_DGRAM"
    laddr_ip: str
    laddr_port: int
    raddr_ip: str
    raddr_port: int
    status: str                 # "ESTABLISHED", "LISTEN", "" for UDP/Unix


_FAMILY_MAP = {2: "AF_INET", 10: "AF_INET6", 1: "AF_UNIX"}
_TYPE_MAP   = {1: "SOCK_STREAM", 2: "SOCK_DGRAM", 3: "SOCK_RAW"}


class NetworkCollector:
    """
    Snapshots all network connections on the host.

    Requires elevated privileges on some systems to see connections
    belonging to other users.

    Usage:
        collector = NetworkCollector()
        records = collector.collect()
    """

    def __init__(self, kind: str = "all") -> None:
        """
        kind: passed to psutil.net_connections — "inet", "inet4", "inet6",
              "tcp", "tcp4", "tcp6", "udp", "udp4", "udp6", "unix", "all".
        """
        self._kind = kind

    def collect(self) -> List[ConnectionRecord]:
        """Return a list of ConnectionRecord for every active connection."""
        now = time.time()
        records: List[ConnectionRecord] = []

        try:
            conns = psutil.net_connections(kind=self._kind)
        except psutil.AccessDenied:
            # Fallback: iterate per-process to get what we can
            conns = []
            for proc in psutil.process_iter(["pid"]):
                try:
                    conns.extend(proc.net_connections())
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        for c in conns:
            laddr_ip   = c.laddr.ip   if c.laddr else ""
            laddr_port = c.laddr.port if c.laddr else 0
            raddr_ip   = c.raddr.ip   if c.raddr else ""
            raddr_port = c.raddr.port if c.raddr else 0

            records.append(ConnectionRecord(
                timestamp=now,
                pid=c.pid or 0,
                fd=c.fd if c.fd is not None else -1,
                family=_FAMILY_MAP.get(c.family, str(c.family)),
                type=_TYPE_MAP.get(c.type, str(c.type)),
                laddr_ip=laddr_ip,
                laddr_port=laddr_port,
                raddr_ip=raddr_ip,
                raddr_port=raddr_port,
                status=c.status or "",
            ))

        return records
