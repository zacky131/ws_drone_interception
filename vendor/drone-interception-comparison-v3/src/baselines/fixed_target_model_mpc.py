from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np

from src.utils.config_schema import ControllerConfig, PursuerConfig, SimulationConfig
from src.baselines.constant_velocity_mpc import ConstantVelocityMPC
from src.baselines.mpc_ca import MPCConstantAcceleration

class FixedTargetModelMPC:
    """Fixed target model MPC wrapper supporting constant_acceleration and constant_velocity."""

    def __init__(
        self,
        ctrl_cfg: ControllerConfig,
        pursuer_cfg: PursuerConfig,
        sim_cfg: SimulationConfig,
        target_prediction_model: Optional[str] = "constant_acceleration",
    ) -> None:
        if target_prediction_model not in ("constant_acceleration", "constant_velocity"):
            raise ValueError(
                f"Invalid target_prediction_model '{target_prediction_model}'. "
                f"Must be 'constant_acceleration' or 'constant_velocity'."
            )
        self._prediction_mode = target_prediction_model
        if self._prediction_mode == "constant_acceleration":
            self._impl = MPCConstantAcceleration(ctrl_cfg, pursuer_cfg, sim_cfg)
        else:
            self._impl = ConstantVelocityMPC(ctrl_cfg, pursuer_cfg, sim_cfg)
        self._solver = getattr(self._impl, "_solver", None)
        self.a_max = getattr(self._impl, "a_max", pursuer_cfg.max_acceleration)
        self.a_max_axis = getattr(self._impl, "a_max_axis", pursuer_cfg.max_acceleration_per_axis)

    def compute_control(
        self,
        pursuer_state: np.ndarray,
        target_measurement: Optional[np.ndarray],
        target_estimate: np.ndarray,
        wind_estimate: np.ndarray,
        t: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        return self._impl.compute_control(
            pursuer_state, target_measurement, target_estimate, wind_estimate, t
        )

    def reset(self) -> None:
        self._impl.reset()

    def _fallback(self, ps: np.ndarray, te: np.ndarray) -> np.ndarray:
        return self._impl._fallback(ps, te)