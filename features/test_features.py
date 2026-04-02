"""
Tests for the feature extractor.
Run with:  python -m pytest features/test_features.py -v
"""
import math
import os
import time

import numpy as np
import pytest

from collectors import ProcessCollector, NetworkCollector, ResourceCollector
from collectors.process_collector import ProcessRecord
from collectors.network_collector import ConnectionRecord
from collectors.resource_collector import ResourceRecord
from features import FeatureExtractor, ProcessFeatureVector, FEATURE_NAMES, N_FEATURES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_process(**kwargs) -> ProcessRecord:
    defaults = dict(
        timestamp=time.time(),
        pid=1234,
        ppid=1,
        name="pytest",
        exe="/usr/bin/python3",
        cmdline=["python3", "-m", "pytest"],
        username="user",
        status="running",
        create_time=time.time() - 60,
        num_threads=2,
        cpu_percent=5.0,
        memory_rss=50 * 1024 * 1024,   # 50 MB
        memory_vms=200 * 1024 * 1024,
        open_files_count=10,
        num_fds=20,
    )
    defaults.update(kwargs)
    return ProcessRecord(**defaults)


def _make_connection(**kwargs) -> ConnectionRecord:
    defaults = dict(
        timestamp=time.time(),
        pid=1234,
        fd=5,
        family="AF_INET",
        type="SOCK_STREAM",
        laddr_ip="192.168.1.10",
        laddr_port=54321,
        raddr_ip="8.8.8.8",
        raddr_port=443,
        status="ESTABLISHED",
    )
    defaults.update(kwargs)
    return ConnectionRecord(**defaults)


def _make_resource(**kwargs) -> ResourceRecord:
    defaults = dict(
        timestamp=time.time(),
        cpu_percent_total=20.0,
        cpu_percent_per_core=[20.0, 20.0],
        cpu_count_logical=2,
        cpu_freq_current=2400.0,
        mem_total=8 * 1024 ** 3,    # 8 GB
        mem_available=4 * 1024 ** 3,
        mem_used=4 * 1024 ** 3,
        mem_percent=50.0,
        swap_total=2 * 1024 ** 3,
        swap_used=0,
        swap_percent=0.0,
    )
    defaults.update(kwargs)
    return ResourceRecord(**defaults)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

def test_feature_names_length():
    from features.feature_extractor import N_FEATURES
    assert len(FEATURE_NAMES) == N_FEATURES


def test_vector_shape():
    ex = FeatureExtractor()
    fv = ex.extract([_make_process()], [], _make_resource())
    assert len(fv) == 1
    assert fv[0].vector.shape == (N_FEATURES,)
    assert fv[0].vector.dtype == np.float32


def test_to_matrix_shape():
    ex = FeatureExtractor()
    procs = [_make_process(pid=i) for i in range(1, 6)]
    fvs = ex.extract(procs, [], _make_resource())
    mat = ex.to_matrix(fvs)
    assert mat.shape == (5, N_FEATURES)


def test_to_matrix_empty():
    ex = FeatureExtractor()
    mat = ex.to_matrix([])
    assert mat.shape == (0, N_FEATURES)


# ---------------------------------------------------------------------------
# Feature value tests
# ---------------------------------------------------------------------------

def test_exe_missing_flag():
    ex = FeatureExtractor()
    proc = _make_process(exe="/tmp/definitely_not_a_real_binary_xyz")
    fv = ex.extract([proc], [], _make_resource())[0]
    idx = FEATURE_NAMES.index("exe_missing")
    assert fv.vector[idx] == 1.0


def test_exe_present_flag():
    ex = FeatureExtractor()
    proc = _make_process(exe="/usr/bin/python3")
    fv = ex.extract([proc], [], _make_resource())[0]
    idx = FEATURE_NAMES.index("exe_missing")
    assert fv.vector[idx] == 0.0


def test_cmdline_has_encoded_b64():
    ex = FeatureExtractor()
    blob = "A" * 50  # 50-char string triggers the base64 regex
    proc = _make_process(cmdline=["python3", "-c", blob])
    fv = ex.extract([proc], [], _make_resource())[0]
    idx = FEATURE_NAMES.index("cmdline_has_encoded")
    assert fv.vector[idx] == 1.0


def test_cmdline_has_encoded_clean():
    ex = FeatureExtractor()
    proc = _make_process(cmdline=["python3", "-m", "pytest"])
    fv = ex.extract([proc], [], _make_resource())[0]
    idx = FEATURE_NAMES.index("cmdline_has_encoded")
    assert fv.vector[idx] == 0.0


def test_running_from_suspicious_path():
    ex = FeatureExtractor()
    proc = _make_process(exe="/tmp/evil")
    fv = ex.extract([proc], [], _make_resource())[0]
    idx = FEATURE_NAMES.index("running_from_suspicious_path")
    assert fv.vector[idx] == 1.0


def test_has_external_connection():
    ex = FeatureExtractor()
    proc = _make_process()
    conn = _make_connection(raddr_ip="8.8.8.8", raddr_port=443)
    fv = ex.extract([proc], [conn], _make_resource())[0]
    idx = FEATURE_NAMES.index("has_external_connection")
    assert fv.vector[idx] == 1.0


def test_no_external_connection_private_ip():
    ex = FeatureExtractor()
    proc = _make_process()
    conn = _make_connection(raddr_ip="192.168.1.1", raddr_port=80)
    fv = ex.extract([proc], [conn], _make_resource())[0]
    idx = FEATURE_NAMES.index("has_external_connection")
    assert fv.vector[idx] == 0.0


def test_rare_port():
    ex = FeatureExtractor()
    proc = _make_process()
    conn = _make_connection(raddr_ip="8.8.8.8", raddr_port=31337)
    fv = ex.extract([proc], [conn], _make_resource())[0]
    idx = FEATURE_NAMES.index("rare_port")
    assert fv.vector[idx] == 1.0


def test_common_port_not_rare():
    ex = FeatureExtractor()
    proc = _make_process()
    conn = _make_connection(raddr_ip="8.8.8.8", raddr_port=443)
    fv = ex.extract([proc], [conn], _make_resource())[0]
    idx = FEATURE_NAMES.index("rare_port")
    assert fv.vector[idx] == 0.0


def test_is_listening():
    ex = FeatureExtractor()
    proc = _make_process()
    conn = _make_connection(raddr_ip="", raddr_port=0, status="LISTEN")
    fv = ex.extract([proc], [conn], _make_resource())[0]
    idx = FEATURE_NAMES.index("is_listening")
    assert fv.vector[idx] == 1.0


def test_cpu_ratio_clamped():
    ex = FeatureExtractor()
    # Process using 200% CPU on a host reporting 10% — ratio must clamp to 1.0
    proc = _make_process(cpu_percent=200.0)
    res  = _make_resource(cpu_percent_total=10.0)
    fv   = ex.extract([proc], [], res)[0]
    idx  = FEATURE_NAMES.index("cpu_ratio")
    assert fv.vector[idx] == pytest.approx(1.0)


def test_no_nan_or_inf_on_live_data():
    """End-to-end: run on real system data and check there are no NaN/Inf."""
    proc_col = ProcessCollector()
    net_col  = NetworkCollector(kind="inet")
    res_col  = ResourceCollector()

    time.sleep(0.2)
    procs   = proc_col.collect()
    conns   = net_col.collect()
    resource = res_col.collect()

    ex  = FeatureExtractor()
    fvs = ex.extract(procs, conns, resource)
    mat = ex.to_matrix(fvs)

    assert not np.any(np.isnan(mat)), "NaN found in feature matrix"
    assert not np.any(np.isinf(mat)), "Inf found in feature matrix"
