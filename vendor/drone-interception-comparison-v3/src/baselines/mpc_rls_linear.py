from __future__ import annotations

import time as _time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.utils.config_schema import ControllerConfig, EstimatorConfig, PursuerConfig, SimulationConfig
from src.estimation.rls_baseline_estimator import RLSBaselineEstimator
from src.baselines.mpc_ca import MPCConstantAcceleration


class MPCRLSLinearPrediction:
    """MPCRLSLinearPrediction runs RLS to estimate pos/vel/acc/jerk, but drops jerk from the MPC NLP."""

    def __init__(
        self,
        ctrl_cfg: ControllerConfig,
        pursuer_cfg: PursuerConfig,
        sim_cfg: SimulationConfig,
        estimator_cfg: Optional[Any] = None,
    ) -> None:
        if isinstance(estimator_cfg, EstimatorConfig):
            self._estimator = RLSBaselineEstimator(estimator_cfg)
        elif estimator_cfg is not None and hasattr(estimator_cfg, "get_estimate"):
            self._estimator = estimator_cfg
        else:
            self._estimator = RLSBaselineEstimator(EstimatorConfig())

        self._mpc_ca = MPCConstantAcceleration(ctrl_cfg, pursuer_cfg, sim_cfg)
        self._solver = getattr(self._mpc_ca, "_solver", None)
        self._n_u = getattr(self._mpc_ca, "_n_u", 3 * ctrl_cfg.horizon)
        self._lbg = getattr(self._mpc_ca, "_lbg", np.zeros(3 * ctrl_cfg.horizon))
        self.a_max = getattr(self._mpc_ca, "a_max", pursuer_cfg.max_acceleration)

    def compute_control(
        self,
        pursuer_state: np.ndarray,
        target_measurement: Optional[np.ndarray],
        target_estimate: np.ndarray,
        wind_estimate: np.ndarray,
        t: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if target_measurement is not None:
            self._estimator.update(target_measurement)

        rls_est = self._estimator.get_estimate()
        jerk_norm = float(np.linalg.norm(rls_est[9:12]))

        # We pass rls_est to MPCConstantAcceleration, which uses pos, vel, acc and drops jerk
        cmd, info = self._mpc_ca.compute_control(
            pursuer_state, target_measurement, rls_est, wind_estimate, t
        )
        info["estimator"] = "rls"
        info["rls_jerk_est_norm"] = jerk_norm
        return cmd, info

    def reset(self) -> None:
        self._estimator.reset()
        self._mpc_ca.reset()

    def _fallback(self, ps: np.ndarray, te: np.ndarray) -> np.ndarray:
        return self._mpc_ca._fallback(ps, te)