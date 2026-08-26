"""
src/prediction/camps/camps_mpc.py

Phase 6: Closed-loop CAMPS Controller with full telemetry and logging.
"""

from __future__ import annotations
import time as _time
from typing import Any, Dict, Optional, Tuple
import numpy as np

from src.utils.config_schema import ControllerConfig, PursuerConfig, SimulationConfig
from src.control.controller_base import ControllerBase
from src.prediction.camps.predictor_bank import CAMPSPredictorBank
from src.control.acados_wrapper import build_acados_mpc

class MPCCAMPSController(ControllerBase):
    """Closed-loop MPC controller driven by CAMPS Multimodel Predictor Selector."""

    def __init__(
        self,
        ctrl_cfg: ControllerConfig,
        pursuer_cfg: PursuerConfig,
        sim_cfg: SimulationConfig,
        selector_type: str = "camps_rule",
    ) -> None:
        self.N: int = ctrl_cfg.horizon
        self.dt: float = sim_cfg.dt
        self.Q_pos: float = ctrl_cfg.Q_pos
        self.Q_T_pos: float = ctrl_cfg.Q_terminal_pos
        self.selector_type = selector_type

        self._u_prev: np.ndarray = np.zeros(3)
        self.predictor_bank = CAMPSPredictorBank(selector_type=selector_type)

        self._acados_solver, self._acados_export_dir = build_acados_mpc(
            ctrl_cfg, pursuer_cfg, sim_cfg, self.Q_pos, self.Q_T_pos
        )

    def reset(self, seed: int = 42) -> None:
        self._u_prev = np.zeros(3)
        self.predictor_bank.reset(seed)

    def compute_control(
        self,
        pursuer_state: np.ndarray,
        target_measurement: Optional[np.ndarray],
        target_estimate: np.ndarray,
        wind_estimate: np.ndarray,
        t: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        t0 = _time.perf_counter()

        selected_horiz, selected_name, sel_info = self.predictor_bank.predict_selected(
            pursuer_state=pursuer_state,
            target_estimate=target_estimate,
            horizon_steps=self.N,
            dt=self.dt,
            t_curr=t,
        )

        waypoints = selected_horiz.waypoints() # (N, 6)

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

        deadline_missed = solve_time > self.dt

        return cmd, {
            "solver_status": status,
            "solver_success": ok,
            "solve_time_s": solve_time,
            "deadline_missed": deadline_missed,
            "selected_predictor": selected_name,
            "selector_type": self.selector_type,
            "selection_reason": sel_info.get("selection_reason", "unknown"),
            "capturability_margin_s": sel_info.get("selected_capturability_margin", 0.0),
            "disagreement_max": sel_info.get("disagreement", {}).get("max_pairwise_position_disagreement", 0.0),
            "narx_ca_disagreement": sel_info.get("disagreement", {}).get("narx_minus_ca_disagreement", 0.0),
        }
