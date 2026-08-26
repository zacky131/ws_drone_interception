"""
Passthrough (no-op) estimator for MPC variants that perform their own
internal prediction (e.g. mpc_cv which assumes constant velocity directly).

Returns the raw measurement as-is, padded with zeros for higher derivatives.
"""

from __future__ import annotations
from typing import Optional
import numpy as np
from .estimator_base import EstimatorBase


class NoEstimator(EstimatorBase):
    """Passthrough stub — returns the sensor measurement directly."""

    def __init__(self) -> None:
        self._est = np.zeros(12)
        self._initialized = False

    def initialize(self, measurement: np.ndarray) -> None:
        z = np.asarray(measurement, dtype=float).ravel()
        self._est = np.zeros(12)
        self._est[: len(z)] = z[: 12]
        self._initialized = True

    def predict(self, dt: float) -> None:
        pass  # no propagation model

    def update(self, measurement: np.ndarray) -> None:
        if not self._initialized:
            self.initialize(measurement)
            return
        z = np.asarray(measurement, dtype=float).ravel()
        self._est[: len(z)] = z[: 12]

    def get_estimate(self) -> np.ndarray:
        return self._est.copy()

    def get_covariance(self) -> None:
        return None

    def reset(self) -> None:
        self._est = np.zeros(12)
        self._initialized = False
