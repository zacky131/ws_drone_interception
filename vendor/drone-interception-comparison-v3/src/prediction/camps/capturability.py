"""
src/prediction/camps/capturability.py

Phase 4: Kinematic Capturability Proxy & Margin Calculation.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Tuple, Dict, Any
from src.prediction.camps.protocol import PredictionHorizon

@dataclass
class CapturabilityResult:
    is_capturable: bool
    capturability_margin_s: float
    best_intercept_step: int
    estimated_time_to_intercept_s: float
    required_acceleration_peak_m_s2: float
    acceleration_headroom_m_s2: float
    jerk_headroom_m_s3: float
    metadata: Dict[str, Any]

class KinematicCapturabilityProxy:
    """Evaluates whether a predicted target trajectory is physically capturable by pursuer."""

    def __init__(
        self,
        max_velocity: float = 15.0,
        max_acceleration: float = 20.0,
        max_jerk: float = 30.0,
        actuator_tau: float = 0.05,
    ):
        self.v_max = max_velocity
        self.a_max = max_acceleration
        self.j_max = max_jerk
        self.tau = actuator_tau

    def estimate_time_to_reach(
        self,
        p_p: np.ndarray,
        v_p: np.ndarray,
        p_target: np.ndarray,
    ) -> float:
        """Estimates minimum pursuer time to reach p_target considering v_max and a_max."""
        dp = p_target - p_p
        dist = float(np.linalg.norm(dp))
        if dist < 1e-3:
            return 0.0

        dir_vec = dp / dist
        v_proj = float(np.dot(v_p, dir_vec))

        # Kinematic time lower bound under acceleration constraint:
        # s = v_0 * t + 0.5 * a * t^2 -> 0.5 * a_max * t^2 + v_proj * t - dist = 0
        a = 0.5 * self.a_max
        b = max(0.0, v_proj)
        c = -dist

        disc = b**2 - 4 * a * c
        if disc >= 0:
            t_acc = (-b + np.sqrt(disc)) / (2 * a)
        else:
            t_acc = dist / self.v_max

        # Cruise time bound
        t_cruise = dist / self.v_max
        t_reach = max(t_acc, t_cruise)
        return float(t_reach)

    def evaluate(
        self,
        pursuer_state: np.ndarray,
        horizon: PredictionHorizon,
        dt: float = 0.02,
    ) -> CapturabilityResult:
        """Evaluates capturability margin for a predicted target horizon."""
        p_p = pursuer_state[0:3]
        v_p = pursuer_state[3:6]
        a_p = pursuer_state[6:9] if len(pursuer_state) >= 9 else np.zeros(3)

        pos_t = horizon.position
        vel_t = horizon.velocity
        N = pos_t.shape[0]

        margins = []
        t_reaches = []
        req_accs = []

        for k in range(N):
            t_k = (k + 1) * dt
            p_tk = pos_t[k]
            v_tk = vel_t[k]

            t_reach = self.estimate_time_to_reach(p_p, v_p, p_tk)
            margin = t_k - t_reach
            margins.append(margin)
            t_reaches.append(t_reach)

            # Ideal constant-acceleration engagement required to reach p_tk, v_tk at t_k
            # p_tk = p_p + v_p * t_k + 0.5 * a_req * t_k^2
            a_req = 2.0 * (p_tk - p_p - v_p * t_k) / (t_k**2 + 1e-6)
            req_accs.append(float(np.linalg.norm(a_req)))

        best_step = int(np.argmax(margins))
        best_margin = float(margins[best_step])
        best_t_reach = float(t_reaches[best_step])
        peak_req_acc = float(req_accs[best_step])

        a_headroom = self.a_max - peak_req_acc
        j_headroom = self.j_max - (peak_req_acc / max(1e-3, self.tau))

        is_capturable = best_margin >= 0.0 and a_headroom >= -2.0

        return CapturabilityResult(
            is_capturable=is_capturable,
            capturability_margin_s=best_margin,
            best_intercept_step=best_step + 1,
            estimated_time_to_intercept_s=best_t_reach,
            required_acceleration_peak_m_s2=peak_req_acc,
            acceleration_headroom_m_s2=a_headroom,
            jerk_headroom_m_s3=j_headroom,
            metadata={"all_margins": margins, "all_req_accs": req_accs}
        )
