"""
RLS-based Adaptive MPC baseline.

This is a thin composition wrapper that pairs the
:class:`~src.estimation.rls_baseline_estimator.RLSBaselineEstimator` with the
:class:`~src.control.adaptive_interception_mpc.AdaptiveInterceptionMPC`.

It implements :class:`ControllerBase` so the simulation engine treats it as a
single guidance law, but internally it runs the RLS estimator before each MPC
solve.  This mirrors the architecture of the prior submission.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.utils.config_schema import ControllerConfig, EstimatorConfig, PursuerConfig, SimulationConfig
from src.estimation.rls_baseline_estimator import RLSBaselineEstimator
from src.control.adaptive_interception_mpc import AdaptiveInterceptionMPC
from src.control.controller_base import ControllerBase


class RLSAdaptiveMPC(ControllerBase):
    """Adaptive MPC guided by the RLS baseline estimator (prior submission)."""

    def __init__(
        self,
        ctrl_cfg: ControllerConfig,
        pursuer_cfg: PursuerConfig,
        sim_cfg: SimulationConfig,
        estimator_cfg: EstimatorConfig,
    ) -> None:
        self._estimator = RLSBaselineEstimator(estimator_cfg)
        self._mpc = AdaptiveInterceptionMPC(ctrl_cfg, pursuer_cfg, sim_cfg)
        self._dt: float = sim_cfg.dt          # use config dt — not hardcoded 0.02
        self._initialized = False

    def compute_control(
        self,
        pursuer_state: np.ndarray,
        target_measurement: Optional[np.ndarray],
        target_estimate: np.ndarray,
        wind_estimate: np.ndarray,
        t: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Run internal RLS estimator (overrides the externally-provided estimate)
        if not self._initialized and target_measurement is not None:
            self._estimator.initialize(target_measurement)
            self._initialized = True
        else:
            self._estimator.predict(self._dt)   # use sim dt from config
            if target_measurement is not None:
                self._estimator.update(target_measurement)

        rls_estimate = self._estimator.get_estimate()

        cmd, info = self._mpc.compute_control(
            pursuer_state, target_measurement, rls_estimate, wind_estimate, t
        )
        info["estimator"] = "rls"
        return cmd, info

    def reset(self) -> None:
        self._estimator.reset()
        self._mpc.reset()
        self._initialized = False
