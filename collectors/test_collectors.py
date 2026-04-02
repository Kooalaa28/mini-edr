"""
Smoke tests for the three collectors.
Run with:  python -m pytest collectors/test_collectors.py -v
"""
import time
from collectors import ProcessCollector, NetworkCollector, ResourceCollector
from collectors.process_collector import ProcessRecord
from collectors.network_collector import ConnectionRecord
from collectors.resource_collector import ResourceRecord


def test_process_collector_returns_records():
    col = ProcessCollector()
    records = col.collect()
    assert len(records) > 0, "Expected at least one running process"
    assert all(isinstance(r, ProcessRecord) for r in records)


def test_process_record_has_self():
    """The collector's own process must appear."""
    import os
    col = ProcessCollector()
    pids = {r.pid for r in col.collect()}
    assert os.getpid() in pids


def test_process_record_fields():
    col = ProcessCollector()
    r = col.collect()[0]
    assert isinstance(r.timestamp, float)
    assert isinstance(r.pid, int) and r.pid > 0
    assert isinstance(r.cmdline, list)
    assert isinstance(r.memory_rss, int)


def test_network_collector_returns_list():
    col = NetworkCollector(kind="inet")
    records = col.collect()
    assert isinstance(records, list)
    assert all(isinstance(r, ConnectionRecord) for r in records)


def test_network_record_fields():
    col = NetworkCollector(kind="inet")
    records = col.collect()
    if records:
        r = records[0]
        assert isinstance(r.timestamp, float)
        assert r.family in ("AF_INET", "AF_INET6", "AF_UNIX") or r.family.isdigit() is False


def test_resource_collector_returns_record():
    col = ResourceCollector()
    time.sleep(0.2)          # let cpu_percent accumulate
    r = col.collect()
    assert isinstance(r, ResourceRecord)
    assert 0.0 <= r.cpu_percent_total <= 100.0
    assert r.mem_total > 0
    assert len(r.cpu_percent_per_core) == r.cpu_count_logical
