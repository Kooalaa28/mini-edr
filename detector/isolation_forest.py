"""
Anomaly detector built on IsolationForest.

Pipeline: StandardScaler → IsolationForest.

Training (fit) is done once on a "clean" baseline snapshot.
Inference (predict) scores new snapshots and flags outliers.

Anomaly scores are normalised to [0, 1] where 1 = most anomalous,
using the min/max of the training distribution.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from features.feature_extractor import FEATURE_NAMES, N_FEATURES, ProcessFeatureVector


@dataclass
class DetectionResult:
    pid: int
    name: str
    exe: str
    cmdline: List[str]
    timestamp: float
    anomaly_score: float    # [0, 1] — higher means more anomalous
    is_anomaly: bool        # True when score >= threshold
    feature_vector: np.ndarray


class IsolationForestDetector:
    """
    Wraps a StandardScaler + IsolationForest pipeline.

    Parameters
    ----------
    contamination:
        Expected fraction of anomalies in training data (passed to
        IsolationForest). Use "auto" to let sklearn decide the threshold,
        or a float like 0.01 for 1 % contamination.
    n_estimators:
        Number of trees in the forest.
    threshold:
        Normalised score [0, 1] above which a process is flagged.
        Defaults to 0.6 — tune after observing your baseline distribution.
    random_state:
        Seed for reproducibility.
    """

    def __init__(
        self,
        contamination: float | str = 0.01,
        n_estimators: int = 200,
        threshold: float = 0.8,
        random_state: int = 42,
    ) -> None:
        self.threshold = threshold
        self._scaler = StandardScaler()
        self._forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self._score_min: float = 0.0
        self._score_max: float = 1.0
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, matrix: np.ndarray) -> "IsolationForestDetector":
        """
        Train on a baseline matrix of shape (n_samples, N_FEATURES).
        Should be collected from a known-clean system state.
        """
        if matrix.shape[0] == 0:
            raise ValueError("Cannot fit on an empty matrix.")
        if matrix.shape[1] != N_FEATURES:
            raise ValueError(
                f"Expected {N_FEATURES} features, got {matrix.shape[1]}."
            )

        scaled = self._scaler.fit_transform(matrix)
        self._forest.fit(scaled)

        # Compute normalisation bounds from training scores so that
        # inference scores map consistently to [0, 1].
        raw = self._forest.score_samples(scaled)   # more negative = more anomalous
        self._score_min = float(raw.min())
        self._score_max = float(raw.max())
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self, vectors: List[ProcessFeatureVector]
    ) -> List[DetectionResult]:
        """Score a list of feature vectors and return DetectionResult per process."""
        self._check_fitted()
        if not vectors:
            return []

        matrix = np.vstack([v.vector for v in vectors]).astype(np.float32)
        scores = self._normalised_scores(matrix)
        flags  = scores >= self.threshold

        return [
            DetectionResult(
                pid=v.pid,
                name=v.name,
                exe=v.exe,
                cmdline=v.cmdline,
                timestamp=v.timestamp,
                anomaly_score=float(scores[i]),
                is_anomaly=bool(flags[i]),
                feature_vector=v.vector,
            )
            for i, v in enumerate(vectors)
        ]

    def score_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Score a raw feature matrix directly.
        Returns a float32 array of shape (n,) with values in [0, 1].
        """
        self._check_fitted()
        return self._normalised_scores(matrix)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the fitted detector to disk (joblib format)."""
        self._check_fitted()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump({
            "scaler":     self._scaler,
            "forest":     self._forest,
            "score_min":  self._score_min,
            "score_max":  self._score_max,
            "threshold":  self.threshold,
        }, path)

    @classmethod
    def load(cls, path: str) -> "IsolationForestDetector":
        """Load a previously saved detector from disk."""
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj._scaler    = data["scaler"]
        obj._forest    = data["forest"]
        obj._score_min = data["score_min"]
        obj._score_max = data["score_max"]
        obj.threshold  = data["threshold"]
        obj._fitted    = True
        return obj

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalised_scores(self, matrix: np.ndarray) -> np.ndarray:
        """
        Scale features → raw IF score → normalise to [0, 1].
        Higher value means more anomalous.
        """
        scaled = self._scaler.transform(matrix)
        raw    = self._forest.score_samples(scaled)   # lower = more anomalous

        span = self._score_max - self._score_min
        if span < 1e-9:
            return np.zeros(len(raw), dtype=np.float32)

        # Flip so that high score = anomalous, then normalise.
        normalised = (self._score_max - raw) / span
        return np.clip(normalised, 0.0, 1.0).astype(np.float32)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "Detector has not been fitted yet. Call fit() first."
            )
