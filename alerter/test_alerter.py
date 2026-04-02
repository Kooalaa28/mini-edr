"""
Tests for the Alerter.
Run with:  python -m pytest alerter/test_alerter.py -v
"""
import json
import time

import numpy as np
import pytest

from detector.isolation_forest import DetectionResult
from features.feature_extractor import FEATURE_NAMES, N_FEATURES
from alerter import Alerter, Alert


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    pid: int = 1234,
    name: str = "evil",
    exe: str = "/tmp/evil",
    cmdline: list | None = None,
    score: float = 0.9,
    is_anomaly: bool = True,
    vector: np.ndarray | None = None,
) -> DetectionResult:
    if vector is None:
        vector = np.zeros(N_FEATURES, dtype=np.float32)
    return DetectionResult(
        pid=pid,
        name=name,
        exe=exe,
        cmdline=cmdline or [exe],
        timestamp=time.time(),
        anomaly_score=score,
        is_anomaly=is_anomaly,
        feature_vector=vector,
    )


def _anomalous_vector() -> np.ndarray:
    vec = np.zeros(N_FEATURES, dtype=np.float32)
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    vec[idx["exe_missing"]]               = 1.0
    vec[idx["running_from_suspicious_path"]] = 1.0
    vec[idx["has_external_connection"]]   = 1.0
    vec[idx["cpu_percent_log"]]           = 4.5
    vec[idx["num_connections"]]           = 30.0
    return vec


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_no_alerts_when_nothing_flagged():
    alerter = Alerter(print_stdout=False)
    results = [_make_result(is_anomaly=False)]
    assert alerter.process(results) == []


def test_returns_alert_for_flagged_result():
    alerter = Alerter(print_stdout=False)
    results = [_make_result(is_anomaly=True)]
    alerts  = alerter.process(results)
    assert len(alerts) == 1
    assert isinstance(alerts[0], Alert)


def test_alert_fields():
    alerter = Alerter(print_stdout=False)
    result  = _make_result(pid=42, name="bad", exe="/tmp/bad", score=0.95)
    alert   = alerter.process([result])[0]

    assert alert.pid == 42
    assert alert.name == "bad"
    assert alert.exe == "/tmp/bad"
    assert alert.anomaly_score == pytest.approx(0.95, abs=1e-3)
    assert "42" in alert.alert_id
    assert alert.timestamp_utc.endswith("Z")


def test_binary_flags_appear_in_top_features():
    alerter = Alerter(print_stdout=False, top_n=10)
    vec     = _anomalous_vector()
    result  = _make_result(vector=vec)
    alert   = alerter.process([result])[0]

    feature_names = {f["feature"] for f in alert.top_features}
    assert "exe_missing"               in feature_names
    assert "running_from_suspicious_path" in feature_names
    assert "has_external_connection"   in feature_names


def test_continuous_features_above_threshold_appear():
    alerter = Alerter(print_stdout=False, top_n=10)
    vec     = _anomalous_vector()
    result  = _make_result(vector=vec)
    alert   = alerter.process([result])[0]

    feature_names = {f["feature"] for f in alert.top_features}
    assert "cpu_percent_log"  in feature_names
    assert "num_connections"  in feature_names


def test_top_n_limit_respected():
    alerter = Alerter(print_stdout=False, top_n=3)
    vec     = _anomalous_vector()
    result  = _make_result(vector=vec)
    alert   = alerter.process([result])[0]

    assert len(alert.top_features) <= 3


def test_binary_features_sorted_first():
    alerter = Alerter(print_stdout=False, top_n=10)
    vec     = _anomalous_vector()
    result  = _make_result(vector=vec)
    alert   = alerter.process([result])[0]

    binary  = {f["feature"] for f in alert.top_features if f["value"] == 1.0}
    # First entry must be a binary flag (value == 1.0)
    assert alert.top_features[0]["value"] == 1.0
    assert len(binary) >= 1


def test_jsonl_file_written(tmp_path):
    log_file = str(tmp_path / "alerts.jsonl")
    alerter  = Alerter(log_path=log_file, print_stdout=False)
    alerter.process([_make_result(pid=1), _make_result(pid=2)])

    lines = open(log_file).read().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert "pid" in obj
        assert "anomaly_score" in obj
        assert "top_features" in obj


def test_jsonl_appends_across_calls(tmp_path):
    log_file = str(tmp_path / "alerts.jsonl")
    alerter  = Alerter(log_path=log_file, print_stdout=False)
    alerter.process([_make_result(pid=1)])
    alerter.process([_make_result(pid=2)])

    lines = open(log_file).read().strip().split("\n")
    assert len(lines) == 2


def test_empty_results_no_file_created(tmp_path):
    log_file = str(tmp_path / "alerts.jsonl")
    alerter  = Alerter(log_path=log_file, print_stdout=False)
    alerter.process([_make_result(is_anomaly=False)])
    assert not (tmp_path / "alerts.jsonl").exists()
