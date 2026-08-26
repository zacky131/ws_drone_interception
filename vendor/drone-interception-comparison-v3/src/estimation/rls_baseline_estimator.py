"""
Recursive Least-Squares baseline estimator.

Estimates target position via a sliding-window polynomial fit, then obtains
velocity, acceleration, and jerk by finite differencing.  This is the
estimation method from the prior submission, retained **only** as a baseline
for ablation comparison.  Its known weaknesses:
    - No explicit noise model or covariance output
    - Finite differences amplify noise
    - No Bayesian data fusion
"""

from __future__ import annotations

from collections import deque

import numpy as np

from src.utils.config_schema import EstimatorConfig
from .estimator_base import EstimatorBase


class RLSBaselineEstimator(EstimatorBase):
    """RLS + finite-difference estimator (prior-submission baseline)."""

    def __init__(self, config: EstimatorConfig) -> None:
        self._cfg = config
        self._lambda = config.rls_forgetting_factor
        self._window = config.rls_window_size

        # Sliding window of (time, position) pairs
        self._history: deque = deque(maxlen=self._window)
        self._t_accum: float = 0.0  # accumulated time

        # Latest estimates
        self._est = np.zeros(12)
        self._initialized = False

    # ── public interface ──────────────────────────────────────────────────

    def initialize(self, measurement: np.ndarray) -> None:
        z = np.asarray(measurement, dtype=float).ravel()
        self._est = np.zeros(12)
        self._est[0:3] = z[0:3]
        self._est[3:6] = z[3:6]
        self._history.clear()
        self._history.append((0.0, z[0:3].copy()))
        self._t_accum = 0.0
        self._initialized = True

    def predict(self, dt: float) -> None:
        if not self._initialized:
            return
        self._t_accum += dt
        # Extrapolate position forward using current derivatives
        p = self._est[0:3]
        v = self._est[3:6]
        a = self._est[6:9]
        j = self._est[9:12]
        self._est[0:3] = p + v * dt + 0.5 * a * dt ** 2 + (1 / 6) * j * dt ** 3
        self._est[3:6] = v + a * dt + 0.5 * j * dt ** 2
        self._est[6:9] = a + j * dt
        # jerk stays constant

    def update(self, measurement: np.ndarray) -> None:
        if not self._initialized:
            self.initialize(measurement)
            return

        z = np.asarray(measurement, dtype=float).ravel()
        pos_meas = z[0:3]
        vel_meas = z[3:6]

        self._history.append((self._t_accum, pos_meas.copy()))
        self._est[0:3] = pos_meas
        self._est[3:6] = vel_meas

        # Finite-difference acceleration and jerk from history
        if len(self._history) >= 3:
            self._estimate_higher_derivatives()

    def get_estimate(self) -> np.ndarray:
        return self._est.copy()

    def get_covariance(self) -> np.ndarray:
        # RLS baseline does not produce a meaningful covariance.
        return np.eye(12) * 1e6

    def reset(self) -> None:
        self._history.clear()
        self._est = np.zeros(12)
        self._t_accum = 0.0
        self._initialized = False

    # ── internal ──────────────────────────────────────────────────────────

    def _estimate_higher_derivatives(self) -> None:
        """Weighted least-squares polynomial fit over the position history,
        followed by analytic differentiation for vel / acc / jerk."""
        times = np.array([h[0] for h in self._history])
        positions = np.array([h[1] for h in self._history])  # (N, 3)
        n = len(times)

        # Normalise time
        t0 = times[-1]
        dt_vec = times - t0  # ≤ 0

        # RLS weights (exponential forgetting)
        w = np.array([self._lambda ** (n - 1 - i) for i in range(n)])
        W = np.diag(w)

        # Fit cubic polynomial per axis:  p(t) = c0 + c1*t + c2*t² + c3*t³
        order = min(3, n - 1)
        T = np.vander(dt_vec, N=order + 1, increasing=True)  # (n, order+1)

        TW = T.T @ W  # (order+1, n)
        A = TW @ T     # (order+1, order+1)
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            return  # singular – skip update

        for ax in range(3):
            b = TW @ positions[:, ax]
            c = A_inv @ b  # polynomial coefficients

            # At t=0 (latest time): derivatives of c0 + c1*t + c2*t² + c3*t³
            # vel  = c1
            # acc  = 2*c2
            # jerk = 6*c3
            if order >= 1:
                self._est[3 + ax] = c[1]
            if order >= 2:
                self._est[6 + ax] = 2.0 * c[2]
            if order >= 3:
                self._est[9 + ax] = 6.0 * c[3]
