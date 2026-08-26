"""
Sliding Mode Guidance (SMG) baseline.

Combines a PN-like closing-velocity term with a sliding-mode correction that
drives the zero-effort miss (ZEM) to zero.  A boundary layer smooths the
signum function to reduce chattering.

    a_cmd = a_closing + a_smc

    a_closing = N · V_c · (ω × r̂)
    a_smc     = η · ZEM / ‖ZEM‖                     outside boundary layer
              = η · ZEM / boundary_thickness          inside boundary layer

The ZEM is approximated as:  ZEM ≈ r + v_rel · t_go
where t_go = ‖r‖ / max(V_c, ε).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.utils.config_schema import ControllerConfig, PursuerConfig
from src.utils.math_helpers import clip_norm
from src.control.controller_base import ControllerBase


class SlidingModeGuidance(ControllerBase):
    """Sliding mode guidance law baseline."""

    def __init__(
        self,
        ctrl_cfg: ControllerConfig,
        pursuer_cfg: PursuerConfig,
        *,
        N: float = 4.0,
        eta: float = 3.0,
        boundary_layer: float = 0.5,
    ) -> None:
        self.N = N
        self.eta = eta
        self.boundary = boundary_layer
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
            return np.zeros(3), {"method": "smc", "degenerate": True}

        r_hat = r / r_norm
        Vc = max(-np.dot(v_rel, r_hat), 1e-3)
        omega = np.cross(r, v_rel) / (r_norm ** 2)

        # PN component
        a_pn = self.N * Vc * np.cross(omega, r_hat)

        # ZEM-based sliding mode correction
        t_go = r_norm / Vc
        zem = r + v_rel * t_go
        zem_norm = np.linalg.norm(zem)

        if zem_norm > self.boundary:
            a_smc = self.eta * zem / zem_norm
        else:
            a_smc = self.eta * zem / self.boundary

        a_cmd = clip_norm(a_pn + a_smc, self.a_max)
        return a_cmd, {"method": "smc", "zem_norm": zem_norm, "t_go": t_go}

    def reset(self) -> None:
        pass
