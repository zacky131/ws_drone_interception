"""
Three-dimensional True Proportional Navigation (TPN) baseline.

a_cmd = N · V_c · (ω × r̂)

where N is the navigation constant, V_c is the scalar closing velocity,
ω is the LOS rate vector, and r̂ is the unit LOS vector.

This baseline uses raw sensor measurements (no estimator) and assumes
a point-mass pursuer model for fairness in comparison.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.utils.config_schema import ControllerConfig, PursuerConfig
from src.utils.math_helpers import clip_norm, safe_normalize
from src.control.controller_base import ControllerBase


class ProportionalNavigation(ControllerBase):
    """True Proportional Navigation in 3-D."""

    def __init__(self, ctrl_cfg: ControllerConfig, pursuer_cfg: PursuerConfig) -> None:
        self.N = ctrl_cfg.fallback_gain  # navigation constant
        self.a_max = pursuer_cfg.max_acceleration

    def compute_control(
        self,
        pursuer_state: np.ndarray,
        target_measurement: Optional[np.ndarray],
        target_estimate: np.ndarray,
        wind_estimate: np.ndarray,
        t: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        p_p = pursuer_state[0:3]
        v_p = pursuer_state[3:6]

        # PN uses direct measurement when available; fall back to estimate
        if target_measurement is not None:
            p_t = target_measurement[0:3]
            v_t = target_measurement[3:6]
        else:
            p_t = target_estimate[0:3]
            v_t = target_estimate[3:6]

        r = p_t - p_p
        v_rel = v_t - v_p
        r_norm = np.linalg.norm(r)

        if r_norm < 1e-6:
            return np.zeros(3), {"method": "pn", "degenerate": True}

        r_hat = r / r_norm
        Vc = -np.dot(v_rel, r_hat)                 # closing velocity (positive when closing)
        omega = np.cross(r, v_rel) / (r_norm ** 2)  # LOS rate

        a_cmd = self.N * Vc * np.cross(omega, r_hat)
        a_cmd = clip_norm(a_cmd, self.a_max)

        return a_cmd, {"method": "pn", "closing_velocity": Vc}

    def reset(self) -> None:
        pass
