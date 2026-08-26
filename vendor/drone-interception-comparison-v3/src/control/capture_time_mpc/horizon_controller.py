"""
src/control/capture_time_mpc/horizon_controller.py

Phase 2 & Phase 4: Unified Horizon & Capture-Time MPC Controller.
"""

from __future__ import annotations
import time
import numpy as np
from typing import Dict, Any, Tuple, Optional, List

from src.utils.config_schema import ControllerConfig, PursuerConfig, SimulationConfig
from src.control.capture_time_mpc.horizon_config import get_horizon_spec, HorizonSpecification
from src.control.capture_time_mpc.acados_horizon_wrapper import build_acados_horizon_mpc

CANDIDATE_CAPTURE_TIMES = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]

class HorizonMPCController:
    def __init__(
        self,
        ctrl_cfg: ControllerConfig,
        pursuer_cfg: PursuerConfig,
        sim_cfg: SimulationConfig,
        variant_name: str,
        enable_capture_time_opt: bool = False,
    ):
        self.ctrl_cfg = ctrl_cfg
        self.pursuer_cfg = pursuer_cfg
        self.sim_cfg = sim_cfg
        self.variant_name = variant_name
        self.enable_capture_time_opt = enable_capture_time_opt
        self.scenario = None

        # Parse prediction mode
        if "oracle" in variant_name:
            self.prediction_mode = "oracle"
        elif "exact_state_ca" in variant_name:
            self.prediction_mode = "exact_state_ca"
        elif "ekf_ca" in variant_name:
            self.prediction_mode = "ekf_ca"
        else:
            raise ValueError(f"Unknown prediction mode in variant: {variant_name}")

        self.horizon_spec = get_horizon_spec(variant_name)
        self.N = self.horizon_spec.N
        self.node_dts = self.horizon_spec.node_dts
        self.cum_times = self.horizon_spec.cumulative_times

        self.Q_pos = getattr(ctrl_cfg, 'Q_pos', 50.0)
        self.Q_T_pos = getattr(ctrl_cfg, 'Q_terminal_pos', 500.0)

        # Build solver
        self.solver, self.export_dir = build_acados_horizon_mpc(
            ctrl_cfg=self.ctrl_cfg,
            pursuer_cfg=self.pursuer_cfg,
            sim_cfg=self.sim_cfg,
            horizon_spec=self.horizon_spec,
            Q_pos=self.Q_pos,
            Q_T_pos=self.Q_T_pos,
        )

        self.u_prev = np.zeros(3)
        self.a_app = np.zeros(3)

    def reset(self):
        self.u_prev = np.zeros(3)
        self.a_app = np.zeros(3)
        self.scenario = None

    def set_scenario(self, scenario: Any):
        self.scenario = scenario

    def _predict_target_horizon(
        self,
        t: float,
        target_est: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates target position and velocity predictions for N shooting nodes."""
        p_t_seq = np.zeros((self.N + 1, 3))
        v_t_seq = np.zeros((self.N + 1, 3))

        if self.prediction_mode == "oracle":
            if self.scenario is None:
                if target_est is not None:
                    p_curr = target_est[0:3]
                    v_curr = target_est[3:6]
                    a_curr = target_est[6:9] if len(target_est) >= 9 else np.zeros(3)
                else:
                    p_curr, v_curr, a_curr = np.zeros(3), np.zeros(3), np.zeros(3)
                for k in range(self.N + 1):
                    dt_k = 0.0 if k == 0 else self.cum_times[k - 1]
                    p_t_seq[k] = p_curr + v_curr * dt_k + 0.5 * a_curr * dt_k**2
                    v_t_seq[k] = v_curr + a_curr * dt_k
            else:
                for k in range(self.N + 1):
                    tk = t + (0.0 if k == 0 else self.cum_times[k - 1])
                    p_t, v_t, _ = self.scenario.get_target_state(tk)
                    p_t_seq[k] = p_t
                    v_t_seq[k] = v_t
        else:
            # CA prediction (exact state or EKF estimate)
            if self.prediction_mode == "exact_state_ca" and self.scenario is not None:
                p_curr, v_curr, a_curr = self.scenario.get_target_state(t)
            else:
                if target_est is not None:
                    p_curr = target_est[0:3]
                    v_curr = target_est[3:6]
                    a_curr = target_est[6:9] if len(target_est) >= 9 else np.zeros(3)
                else:
                    p_curr, v_curr, a_curr = np.zeros(3), np.zeros(3), np.zeros(3)

            for k in range(self.N + 1):
                dt_k = 0.0 if k == 0 else self.cum_times[k - 1]
                p_t_seq[k] = p_curr + v_curr * dt_k + 0.5 * a_curr * dt_k**2
                v_t_seq[k] = v_curr + a_curr * dt_k

        return p_t_seq, v_t_seq

    def compute_control(
        self,
        pursuer_state: np.ndarray,
        target_meas: Optional[np.ndarray],
        target_est: Optional[np.ndarray],
        wind: np.ndarray,
        t: float = 0.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        t_start = time.perf_counter()

        p_p = pursuer_state[0:3]
        v_p = pursuer_state[3:6]
        a_app = pursuer_state[6:9] if len(pursuer_state) >= 9 else self.a_app

        x0 = np.concatenate([p_p, v_p, a_app, self.u_prev])

        # Get target horizon
        p_t_seq, v_t_seq = self._predict_target_horizon(t, target_est)

        # Dynamic Capture-Time Node Selection (Phase 4)
        target_capture_k = self.N - 1
        selected_tau_c = None
        if self.enable_capture_time_opt:
            v_max = self.pursuer_cfg.max_velocity
            best_tau = self.horizon_spec.total_duration
            best_k = self.N - 1

            for candidate_tau in CANDIDATE_CAPTURE_TIMES:
                if candidate_tau <= self.horizon_spec.total_duration:
                    k_idx = int(np.argmin(np.abs(self.cum_times - candidate_tau)))
                    p_t_cand = p_t_seq[k_idx + 1]
                    dist_to_cand = np.linalg.norm(p_p - p_t_cand)
                    v_req = dist_to_cand / max(candidate_tau, 0.1)
                    if v_req <= v_max * 1.1:
                        best_tau = candidate_tau
                        best_k = k_idx
                        break
            target_capture_k = best_k
            selected_tau_c = best_tau

        # Update solver initial condition
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

        # Update stage parameters (p_t, v_t, wind, dt_k, w_pos)
        for k in range(self.N):
            dt_k = self.node_dts[k]
            w_pos_k = self.Q_T_pos if k == target_capture_k else self.Q_pos
            p_k = np.concatenate([p_t_seq[k+1], v_t_seq[k+1], wind, [dt_k, w_pos_k]])
            self.solver.set(k, "p", p_k)

        # Solve OCP
        status = self.solver.solve()
        t_solve = time.perf_counter() - t_start

        if status in (0, 2):
            u_opt = self.solver.get(0, "u")
            a_cmd = np.clip(u_opt, -self.pursuer_cfg.max_acceleration_per_axis, self.pursuer_cfg.max_acceleration_per_axis)
        else:
            a_cmd = np.zeros(3)

        self.u_prev = a_cmd.copy()

        info = {
            "solver_status": status,
            "solve_time_s": t_solve,
            "horizon_variant": self.variant_name,
            "selected_tau_c": selected_tau_c,
            "target_capture_k": target_capture_k,
            "N_nodes": self.N,
        }

        return a_cmd, info
