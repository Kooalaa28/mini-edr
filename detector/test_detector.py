"""
Tests for IsolationForestDetector.
Run with:  python -m pytest detector/test_detector.py -v
"""
import os
import time
import tempfile

import numpy as np
import pytest

from features.feature_extractor import (
    FeatureExtractor, ProcessFeatureVector, N_FEATURES, FEATURE_NAMES,
)
from detector import IsolationForestDetector, DetectionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_normal_matrix(n: int = 200, seed: int = 0) -> np.ndarray:
    """Simulate a 'clean' baseline: low CPU, moderate memory, no suspicious flags."""
    rng = np.random.default_rng(seed)
    mat = np.zeros((n, N_FEATURES), dtype=np.float32)
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    mat[:, idx["cpu_percent_log"]]          = rng.normal(0.5,  0.2, n).clip(0)
    mat[:, idx["memory_rss_log"]]           = rng.normal(3.0,  0.5, n).clip(0)
    mat[:, idx["num_threads"]]              = rng.integers(1, 10, n)
    mat[:, idx["open_files_count"]]         = rng.integers(0, 20, n)
    mat[:, idx["num_fds"]]                  = rng.integers(0, 30, n)
    mat[:, idx["process_age_seconds_log"]]  = rng.normal(6.0,  1.0, n).clip(0)
    mat[:, idx["cmdline_length"]]           = rng.integers(5, 50, n)
    mat[:, idx["num_connections"]]          = rng.integers(0, 5, n)
    mat[:, idx["cpu_ratio"]]                = rng.uniform(0, 0.1, n)
    mat[:, idx["mem_ratio"]]                = rng.uniform(0, 0.05, n)
    # binary flags all 0 for clean baseline
    return mat


def _make_anomalous_vector() -> np.ndarray:
    """A single clearly anomalous process: high CPU, running from /tmp, encoded cmdline."""
    vec = np.zeros(N_FEATURES, dtype=np.float32)
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    vec[idx["cpu_percent_log"]]           = 5.0    # e^5 ≈ 148 % CPU
    vec[idx["memory_rss_log"]]            = 7.0    # ~1 GB RSS
    vec[idx["exe_missing"]]               = 1.0
    vec[idx["running_from_suspicious_path"]] = 1.0
    vec[idx["cmdline_has_encoded"]]       = 1.0
    vec[idx["has_external_connection"]]   = 1.0
    vec[idx["rare_port"]]                 = 1.0
    vec[idx["cpu_ratio"]]                 = 1.0
    vec[idx["mem_ratio"]]                 = 0.9
    return vec


def _make_fv(vector: np.ndarray, pid: int = 9999) -> ProcessFeatureVector:
    return ProcessFeatureVector(
        pid=pid, name="evil", exe="/tmp/evil",
        cmdline=["/tmp/evil"], timestamp=time.time(), vector=vector,
    )


# ---------------------------------------------------------------------------
# Fit / predict tests
# ---------------------------------------------------------------------------

def test_fit_and_predict_returns_results():
    det = IsolationForestDetector()
    mat = _make_normal_matrix()
    det.fit(mat)

    fvs = [_make_fv(mat[0])]
    results = det.predict(fvs)
    assert len(results) == 1
    assert isinstance(results[0], DetectionResult)


def test_scores_in_range():
    det = IsolationForestDetector()
    mat = _make_normal_matrix()
    det.fit(mat)

    scores = det.score_matrix(mat)
    assert scores.shape == (len(mat),)
    assert float(scores.min()) >= 0.0
    assert float(scores.max()) <= 1.0


def test_anomalous_scores_higher_than_normal():
    det = IsolationForestDetector(contamination=0.01, threshold=0.6)
    mat = _make_normal_matrix(n=300)
    det.fit(mat)

    normal_scores    = det.score_matrix(mat)
    anomalous_vector = _make_anomalous_vector().reshape(1, -1)
    anomalous_score  = det.score_matrix(anomalous_vector)[0]

    assert anomalous_score > normal_scores.mean(), (
        f"Anomalous score {anomalous_score:.3f} should exceed mean normal "
        f"score {normal_scores.mean():.3f}"
    )


def test_anomaly_is_flagged():
    det = IsolationForestDetector(contamination=0.01, threshold=0.6)
    det.fit(_make_normal_matrix(n=300))
    result = det.predict([_make_fv(_make_anomalous_vector())])[0]
    assert result.is_anomaly, f"Expected anomaly flag, score was {result.anomaly_score:.3f}"


def test_normal_process_not_flagged():
    # threshold=0.85 → top 15% of the normalised range; on a clean baseline
    # the vast majority of training samples should score well below that.
    det = IsolationForestDetector(contamination=0.01, threshold=0.85)
    mat = _make_normal_matrix(n=300)
    det.fit(mat)
    results = det.predict([_make_fv(mat[i], pid=i) for i in range(100)])
    flagged = sum(r.is_anomaly for r in results)
    assert flagged < 10, f"{flagged}/100 normal processes were flagged"


def test_predict_empty_list():
    det = IsolationForestDetector()
    det.fit(_make_normal_matrix())
    assert det.predict([]) == []


def test_unfitted_raises():
    det = IsolationForestDetector()
    with pytest.raises(RuntimeError, match="fit"):
        det.predict([_make_fv(_make_anomalous_vector())])


def test_fit_wrong_feature_count_raises():
    det = IsolationForestDetector()
    with pytest.raises(ValueError):
        det.fit(np.zeros((10, N_FEATURES + 1)))


# ---------------------------------------------------------------------------
# Persistence tests
# ---------------------------------------------------------------------------

def test_save_and_load(tmp_path):
    det = IsolationForestDetector(threshold=0.55)
    mat = _make_normal_matrix()
    det.fit(mat)

    path = str(tmp_path / "model.joblib")
    det.save(path)
    assert os.path.exists(path)

    loaded = IsolationForestDetector.load(path)
    assert loaded.threshold == 0.55

    # Scores must be identical after round-trip
    original_scores = det.score_matrix(mat)
    loaded_scores   = loaded.score_matrix(mat)
    np.testing.assert_allclose(original_scores, loaded_scores, rtol=1e-5)
