"""
Standard (non-adaptive) MPC baseline.

Uses the same CasADi NLP structure as :class:`AdaptiveInterceptionMPC` but
assumes a **constant-velocity** target model (no acceleration or jerk
prediction).  This isolates the benefit of the EKF-based target prediction.
"""

from __future__ import annotations

import time as _time
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import casadi as ca
    _HAS_CASADI = True
except ImportError:
    _HAS_CASADI = False

from src.utils.config_schema import ControllerConfig, PursuerConfig, SimulationConfig
from src.utils.math_helpers import clip_norm
from src.control.controller_base import ControllerBase

try:
    from src.control.acados_wrapper import build_acados_mpc
    _HAS_ACADOS = True
except ImportError:
    _HAS_ACADOS = False


class StandardMPC(ControllerBase):
    """MPC with constant-velocity target assumption (no estimator needed)."""

    def __init__(
        self,
        ctrl_cfg: ControllerConfig,
        pursuer_cfg: PursuerConfig,
        sim_cfg: SimulationConfig,
    ) -> None:
        if not _HAS_CASADI:
            raise ImportError("CasADi is required for StandardMPC.")

        self.N = ctrl_cfg.horizon
        self.dt = sim_cfg.dt
        self.tau = pursuer_cfg.actuator_time_constant
        self.a_max = pursuer_cfg.max_acceleration
        self.a_max_axis = pursuer_cfg.max_acceleration_per_axis
        self.jerk_max = pursuer_cfg.max_jerk
        self.v_max = pursuer_cfg.max_velocity
        self.Q_pos = ctrl_cfg.Q_pos
        self.Q_vel = ctrl_cfg.Q_vel
        self.R = ctrl_cfg.R_control
        self.R_rate = ctrl_cfg.R_rate
        self.Q_T_pos = ctrl_cfg.Q_terminal_pos
        self.Q_T_vel = ctrl_cfg.Q_terminal_vel
        self.fallback_gain = ctrl_cfg.fallback_gain
        self._solver_type = getattr(ctrl_cfg, "solver", "casadi")

        self._u_prev = np.zeros(3)
        self._warm_x0: np.ndarray | None = None

        if self._solver_type == "acados":
            if not _HAS_ACADOS:
                raise ImportError("acados_template is required for solver='acados'.")
            self._acados_solver, self._acados_export_dir = build_acados_mpc(
                ctrl_cfg, pursuer_cfg, sim_cfg, self.Q_pos, self.Q_T_pos
            )
        else:
            self._build_nlp()

    def _build_nlp(self) -> None:
        N = self.N
        dt = self.dt
        alpha = min(dt / self.tau, 1.0) if self.tau > 1e-9 else 1.0
        n_u = 3 * N

        U = ca.MX.sym("U", n_u)
        # Parameters: [x0_pursuer(9), target_pos(3), target_vel(3), u_prev(3), wind(3)] = 21
        P = ca.MX.sym("P", 21)

        x0 = P[0:9]
        pt0 = P[9:12]
        vt0 = P[12:15]
        u_prev_sym = P[15:18]
        wind = P[18:21]

        cost = 0.0
        g_list = []
        x = x0

        for k in range(N):
            u_k = U[3 * k : 3 * (k + 1)]
            p_p, v_p, a_app = x[0:3], x[3:6], x[6:9]

            a_app_new = a_app + alpha * (u_k - a_app)
            total_acc = a_app_new + wind
            v_new = v_p + total_acc * dt
            p_new = p_p + v_p * dt + 0.5 * total_acc * dt ** 2
            x = ca.vertcat(p_new, v_new, a_app_new)

            # Constant-velocity target prediction
            t_pred = (k + 1) * dt
            p_t = pt0 + vt0 * t_pred
            v_t = vt0

            dp = p_new - p_t
            dv = v_new - v_t

            if k < N - 1:
                cost += self.Q_pos * ca.dot(dp, dp) + self.Q_vel * ca.dot(dv, dv)
            else:
                cost += self.Q_T_pos * ca.dot(dp, dp) + self.Q_T_vel * ca.dot(dv, dv)
            cost += self.R * ca.dot(u_k, u_k)

            if k == 0:
                du = (u_k - u_prev_sym) / dt
            else:
                du = (u_k - U[3 * (k - 1) : 3 * k]) / dt
            cost += self.R_rate * ca.dot(du, du)

            g_list.append(ca.dot(u_k, u_k))
            g_list.append(ca.dot(v_new, v_new))
            g_list.append(ca.dot(du, du))

        nlp = {"f": cost, "x": U, "g": ca.vertcat(*g_list), "p": P}
        opts = {
            "ipopt.max_iter": 100,
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.sb": "yes",
            "ipopt.warm_start_init_point": "yes",
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
        }
        self._solver = ca.nlpsol("stdmpc", "ipopt", nlp, opts)
        self._n_u = n_u
        self._lbx = np.full(n_u, -self.a_max_axis)
        self._ubx = np.full(n_u, self.a_max_axis)
        self._lbg = np.zeros(3 * N)
        self._ubg = np.tile([self.a_max ** 2, self.v_max ** 2, self.jerk_max ** 2], N)

    def compute_control(
        self,
        pursuer_state: np.ndarray,
        target_measurement: Optional[np.ndarray],
        target_estimate: np.ndarray,
        wind_estimate: np.ndarray,
        t: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Use measurement for pos/vel (no acc/jerk needed)
        if target_measurement is not None:
            pt = target_measurement[0:3]
            vt = target_measurement[3:6]
        else:
            pt = target_estimate[0:3]
            vt = target_estimate[3:6]

        p_param = np.zeros(21)
        p_param[0:9] = pursuer_state
        p_param[9:12] = pt
        p_param[12:15] = vt
        p_param[15:18] = self._u_prev
        p_param[18:21] = wind_estimate

        t0 = _time.perf_counter()
        
        if self._solver_type == "acados":
            ok = False
            try:
                # ── Set Initial State ─────────────────────────────
                x0 = np.concatenate([pursuer_state[0:9], self._u_prev])
                self._acados_solver.set(0, "lbx", x0)
                self._acados_solver.set(0, "ubx", x0)
                
                wind = wind_estimate
                
                # ── Set Parameters for all stages ──────────────────
                for k in range(self.N):
                    t_pred = (k + 1) * self.dt
                    p_t = pt + vt * t_pred
                    v_t = vt
                    self._acados_solver.set(k, "p", np.concatenate([p_t, v_t, wind]))
                
                # Terminal node (k=N) evaluates parameter at t_pred = N * dt
                t_pred = self.N * self.dt
                p_t = pt + vt * t_pred
                v_t = vt
                self._acados_solver.set(self.N, "p", np.concatenate([p_t, v_t, wind]))
                
                status_code = self._acados_solver.solve()
                solve_time = _time.perf_counter() - t0
                
                if status_code in (0, 2):  # 0: success, 2: max iter
                    u_opt = self._acados_solver.get(0, "u")
                    cmd = u_opt[0:3]
                    ok = True
                    status = f"acados_{status_code}"
                else:
                    cmd = self._fallback(pursuer_state, target_estimate)
                    status = f"acados_fail_{status_code}"
                    
            except Exception:
                solve_time = _time.perf_counter() - t0
                cmd = self._fallback(pursuer_state, target_estimate)
                status = "exception"
                ok = False
        else:
            x0_guess = self._warm_x0 if self._warm_x0 is not None else np.zeros(self._n_u)

            try:
                sol = self._solver(
                    x0=x0_guess, p=p_param,
                    lbx=self._lbx, ubx=self._ubx,
                    lbg=self._lbg, ubg=self._ubg,
                )
                solve_time = _time.perf_counter() - t0
                u_opt = np.array(sol["x"]).ravel()
                stats = self._solver.stats()
                status = stats.get("return_status", "unknown")
                ok = status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")

                if ok:
                    cmd = u_opt[0:3]
                    self._warm_x0 = np.concatenate([u_opt[3:], u_opt[-3:]])
                else:
                    cmd = self._fallback(pursuer_state, target_estimate)
            except Exception:
                solve_time = _time.perf_counter() - t0
                cmd = self._fallback(pursuer_state, target_estimate)
                status = "exception"
                ok = False

        self._u_prev = cmd.copy()
        return cmd, {"solver_status": status, "solver_success": ok, "solve_time_s": solve_time}

    def reset(self) -> None:
        self._u_prev = np.zeros(3)
        self._warm_x0 = None

    def _fallback(self, ps: np.ndarray, te: np.ndarray) -> np.ndarray:
        r = te[0:3] - ps[0:3]
        v_rel = te[3:6] - ps[3:6]
        rn = np.linalg.norm(r)
        if rn < 1e-6:
            return np.zeros(3)
        rh = r / rn
        Vc = -np.dot(v_rel, rh)
        omega = np.cross(r, v_rel) / rn ** 2
        return clip_norm(self.fallback_gain * Vc * np.cross(omega, rh), self.a_max)
