"""
src/prediction/camps/exact_state_ca_mpc.py

Phase 1.2: Exact Current State CA MPC Baseline (mpc_exact_state_ca).

Uses exact current target state (zero measurement & estimation error) at time t
but retains constant-acceleration (CA) future extrapolation over the MPC horizon.
"""

from __future__ import annotations
import time as _time
from typing import Any, Dict, Optional, Tuple
import numpy as np

from src.utils.config_schema import ControllerConfig, PursuerConfig, SimulationConfig
from src.control.controller_base import ControllerBase
from src.prediction.camps.predictors import ExactStateCAPredictor
from src.control.acados_wrapper import build_acados_mpc

class MPCExactStateCA(ControllerBase):
    """MPC controller provided with exact current target state and CA extrapolation."""

    def __init__(
        self,
        ctrl_cfg: ControllerConfig,
        pursuer_cfg: PursuerConfig,
        sim_cfg: SimulationConfig,
    ) -> None:
        self.N: int = ctrl_cfg.horizon
        self.dt: float = sim_cfg.dt
        self.Q_pos: float = ctrl_cfg.Q_pos
        self.Q_T_pos: float = ctrl_cfg.Q_terminal_pos
        self._solver_type: str = getattr(ctrl_cfg, "solver", "casadi")

        self._u_prev: np.ndarray = np.zeros(3)
        self.scenario: Any = None
        self.exact_ca_predictor = ExactStateCAPredictor()

        self._acados_solver, self._acados_export_dir = build_acados_mpc(
            ctrl_cfg, pursuer_cfg, sim_cfg, self.Q_pos, self.Q_T_pos
        )

    def set_scenario(self, scenario: Any) -> None:
        self.scenario = scenario

    def reset(self) -> None:
        self._u_prev = np.zeros(3)

    def compute_control(
        self,
        pursuer_state: np.ndarray,
        target_measurement: Optional[np.ndarray],
        target_estimate: np.ndarray,
        wind_estimate: np.ndarray,
        t: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        t0 = _time.perf_counter()
        
        horizon = self.exact_ca_predictor.predict(
            target_estimate=target_estimate,
            horizon_steps=self.N,
            dt=self.dt,
            t_curr=t,
            scenario=self.scenario
        )
        waypoints = horizon.waypoints() # (N, 6)

        x0 = np.concatenate([pursuer_state[0:9], self._u_prev])
        self._acados_solver.set(0, "lbx", x0)
        self._acados_solver.set(0, "ubx", x0)

        wind = wind_estimate
        for k in range(self.N):
            self._acados_solver.set(
                k, "p", np.concatenate([waypoints[k, 0:3], waypoints[k, 3:6], wind])
            )
        self._acados_solver.set(
            self.N, "p", np.concatenate([waypoints[-1, 0:3], waypoints[-1, 3:6], wind])
        )

        status_code = self._acados_solver.solve()
        solve_time = _time.perf_counter() - t0

        if status_code in (0, 2):
            cmd = self._acados_solver.get(0, "u")[0:3]
            ok = True
            status = f"acados_{status_code}"
        else:
            cmd = np.zeros(3)
            ok = False
            status = f"acados_fail_{status_code}"

        self._u_prev = np.asarray(cmd, dtype=float).copy()
        return cmd, {
            "solver_status": status,
            "solver_success": ok,
            "solve_time_s": solve_time,
            "estimator": "exact_current_state",
            "predictor": "exact_state_ca"
        }
