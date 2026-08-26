"""
Adaptive Interception MPC with CasADi / IPOPT.

Receding-horizon nonlinear MPC that:
  - tracks a predicted target trajectory derived from the EKF state estimate,
  - includes first-order actuator lag in the prediction model,
  - imposes acceleration norm, per-axis, rate, and velocity constraints,
  - supports warm-starting and solver-failure fallback (PN).

The NLP is built once (parametrically) in ``__init__`` and re-solved at each
control step by updating the parameter vector only.  This avoids CasADi
re-compilation overhead and enables real-time-oriented evaluation.

Formulation
-----------
Decision variable:  U ∈ ℝ^{3N}   (commanded accelerations over horizon N)

Parameters (passed each call):
    P = [x₀_pursuer(9), target_est(12), u_prev(3), wind(3)]   (27-dim)

Pursuer prediction model (per step k):
    a_app(k+1) = a_app(k) + α·(u(k) − a_app(k))        actuator lag
    v(k+1)     = v(k) + (a_app(k+1) + wind)·dt           translation
    p(k+1)     = p(k) + v(k)·dt + ½(a_app(k+1)+wind)·dt²

Target prediction (polynomial from estimator):
    p_t(k) = p₀ + v₀·t_k + ½a₀·t_k² + ⅙j₀·t_k³
    v_t(k) = v₀ + a₀·t_k + ½j₀·t_k²

Cost:
    Σ  Q_pos‖Δp‖² + Q_vel‖Δv‖²  (stage)   +   Q_T·(...)  (terminal)
    + R‖u‖²  +  R_rate‖Δu/dt‖²

Constraints:
    ‖u(k)‖² ≤ a_max²       (acceleration norm)
    ‖v(k+1)‖² ≤ v_max²     (velocity bound)
    ‖Δu(k)/dt‖² ≤ j_max²   (rate / jerk bound)
    u_i ∈ [−a_max_axis, a_max_axis]  (box)
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
from src.utils.math_helpers import clip_norm, closing_velocity, los_rate, safe_normalize
from .controller_base import ControllerBase

try:
    from .acados_wrapper import build_acados_mpc
    _HAS_ACADOS = True
except ImportError:
    _HAS_ACADOS = False


class AdaptiveInterceptionMPC(ControllerBase):
    """CasADi-based adaptive interception MPC (proposed method)."""

    def __init__(
        self,
        ctrl_cfg: ControllerConfig,
        pursuer_cfg: PursuerConfig,
        sim_cfg: SimulationConfig,
    ) -> None:
        if not _HAS_CASADI:
            raise ImportError(
                "CasADi is required for AdaptiveInterceptionMPC. "
                "Install with: pip install casadi"
            )

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
        self._solver_max_iter = ctrl_cfg.solver_max_iter
        self._solver_print = ctrl_cfg.solver_print_level
        self._warm_start = ctrl_cfg.warm_start
        self._solver_type = getattr(ctrl_cfg, "solver", "casadi")

        # Internal buffers
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

    # ── NLP construction (called once) ────────────────────────────────────

    def _build_nlp(self) -> None:
        N = self.N
        dt = self.dt
        alpha = min(dt / self.tau, 1.0) if self.tau > 1e-9 else 1.0

        # ── Symbolic variables ────────────────────────────────────────────
        n_u = 3 * N
        U = ca.MX.sym("U", n_u)
        P = ca.MX.sym("P", 27)

        # Unpack parameters
        x0 = P[0:9]
        target_est = P[9:21]
        u_prev_sym = P[21:24]
        wind = P[24:27]

        pt0 = target_est[0:3]
        vt0 = target_est[3:6]
        at0 = target_est[6:9]
        jt0 = target_est[9:12]

        # ── Build NLP ─────────────────────────────────────────────────────
        cost = 0.0
        g_list = []

        x = x0  # pursuer state: [pos(3), vel(3), a_app(3)]
        for k in range(N):
            u_k = U[3 * k : 3 * (k + 1)]
            p_p = x[0:3]
            v_p = x[3:6]
            a_app = x[6:9]

            # Actuator dynamics
            a_app_new = a_app + alpha * (u_k - a_app)

            # Translational dynamics
            total_acc = a_app_new + wind
            v_new = v_p + total_acc * dt
            p_new = p_p + v_p * dt + 0.5 * total_acc * dt ** 2

            x = ca.vertcat(p_new, v_new, a_app_new)

            # Target prediction
            t_pred = (k + 1) * dt
            p_t = pt0 + vt0 * t_pred + 0.5 * at0 * t_pred ** 2 + (1.0 / 6.0) * jt0 * t_pred ** 3
            v_t = vt0 + at0 * t_pred + 0.5 * jt0 * t_pred ** 2

            dp = p_new - p_t
            dv = v_new - v_t

            # Stage / terminal cost
            if k < N - 1:
                cost += self.Q_pos * ca.dot(dp, dp) + self.Q_vel * ca.dot(dv, dv)
            else:
                cost += self.Q_T_pos * ca.dot(dp, dp) + self.Q_T_vel * ca.dot(dv, dv)

            cost += self.R * ca.dot(u_k, u_k)

            # Rate cost
            if k == 0:
                du = (u_k - u_prev_sym) / dt
            else:
                du = (u_k - U[3 * (k - 1) : 3 * k]) / dt
            cost += self.R_rate * ca.dot(du, du)

            # ── Constraints ───────────────────────────────────────────────
            g_list.append(ca.dot(u_k, u_k))         # accel norm²
            g_list.append(ca.dot(v_new, v_new))      # velocity norm²
            g_list.append(ca.dot(du, du))             # jerk norm²

        g = ca.vertcat(*g_list)

        nlp = {"f": cost, "x": U, "g": g, "p": P}

        opts = {
            "ipopt.max_iter": self._solver_max_iter,
            "ipopt.print_level": self._solver_print,
            "print_time": 0,
            "ipopt.sb": "yes",
            "ipopt.warm_start_init_point": "yes" if self._warm_start else "no",
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.acceptable_iter": 5,
        }

        self._solver = ca.nlpsol("aimpc", "ipopt", nlp, opts)
        self._n_u = n_u

        # Bounds
        self._lbx = np.full(n_u, -self.a_max_axis)
        self._ubx = np.full(n_u, self.a_max_axis)

        self._lbg = np.zeros(3 * N)
        self._ubg = np.tile(
            [self.a_max ** 2, self.v_max ** 2, self.jerk_max ** 2], N
        )

    # ── ControllerBase implementation ─────────────────────────────────────

    def compute_control(
        self,
        pursuer_state: np.ndarray,
        target_measurement: Optional[np.ndarray],
        target_estimate: np.ndarray,
        wind_estimate: np.ndarray,
        t: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:

        p_param = np.zeros(27)
        p_param[0:9] = pursuer_state
        p_param[9:21] = target_estimate
        p_param[21:24] = self._u_prev
        p_param[24:27] = wind_estimate

        t_start = _time.perf_counter()
        
        if self._solver_type == "acados":
            success = False
            try:
                # ── Set Initial State ─────────────────────────────
                x0 = np.concatenate([pursuer_state[0:9], self._u_prev])
                self._acados_solver.set(0, "lbx", x0)
                self._acados_solver.set(0, "ubx", x0)
                
                pt0 = target_estimate[0:3]
                vt0 = target_estimate[3:6]
                at0 = target_estimate[6:9]
                jt0 = target_estimate[9:12]
                wind = wind_estimate
                
                # ── Set Parameters for all stages ──────────────────
                for k in range(self.N):
                    t_pred = (k + 1) * self.dt
                    p_t = pt0 + vt0 * t_pred + 0.5 * at0 * t_pred**2 + (1.0/6.0) * jt0 * t_pred**3
                    v_t = vt0 + at0 * t_pred + 0.5 * jt0 * t_pred**2
                    self._acados_solver.set(k, "p", np.concatenate([p_t, v_t, wind]))
                
                # Terminal node (k=N) evaluates parameter at t_pred = N * dt
                t_pred = self.N * self.dt
                p_t = pt0 + vt0 * t_pred + 0.5 * at0 * t_pred**2 + (1.0/6.0) * jt0 * t_pred**3
                v_t = vt0 + at0 * t_pred + 0.5 * jt0 * t_pred**2
                self._acados_solver.set(self.N, "p", np.concatenate([p_t, v_t, wind]))
                
                status = self._acados_solver.solve()
                solve_time = _time.perf_counter() - t_start
                
                if status in (0, 2):  # 0: success, 2: max iter (sometimes acceptable)
                    u_opt = self._acados_solver.get(0, "u")
                    cmd = u_opt[0:3]
                    success = True
                    solver_status = f"acados_{status}"
                else:
                    cmd = self._fallback(pursuer_state, target_estimate)
                    solver_status = f"acados_fail_{status}"
                    
            except Exception:
                solve_time = _time.perf_counter() - t_start
                cmd = self._fallback(pursuer_state, target_estimate)
                solver_status = "exception"
                success = False
        else:
            # Initial guess (warm start or zero)
            x0_guess = self._warm_x0 if self._warm_x0 is not None else np.zeros(self._n_u)
            
            try:
                sol = self._solver(
                    x0=x0_guess,
                    p=p_param,
                    lbx=self._lbx,
                    ubx=self._ubx,
                    lbg=self._lbg,
                    ubg=self._ubg,
                )
                solve_time = _time.perf_counter() - t_start

                u_opt = np.array(sol["x"]).ravel()
                stats = self._solver.stats()
                solver_status = stats.get("return_status", "unknown")
                success = solver_status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")

                if success:
                    cmd = u_opt[0:3]
                    # Shift warm start
                    if self._warm_start:
                        self._warm_x0 = np.concatenate([u_opt[3:], u_opt[-3:]])
                else:
                    cmd = self._fallback(pursuer_state, target_estimate)

            except Exception:
                solve_time = _time.perf_counter() - t_start
                cmd = self._fallback(pursuer_state, target_estimate)
                solver_status = "exception"
                success = False

        self._u_prev = cmd.copy()

        info: Dict[str, Any] = {
            "solver_status": solver_status,
            "solver_success": success,
            "solve_time_s": solve_time,
        }
        return cmd, info

    def reset(self) -> None:
        self._u_prev = np.zeros(3)
        self._warm_x0 = None

    # ── fallback (proportional navigation) ────────────────────────────────

    def _fallback(self, pursuer_state: np.ndarray, target_estimate: np.ndarray) -> np.ndarray:
        """Pure PN fallback when the NLP solver fails."""
        p_p = pursuer_state[0:3]
        v_p = pursuer_state[3:6]
        p_t = target_estimate[0:3]
        v_t = target_estimate[3:6]

        r = p_t - p_p
        v_rel = v_t - v_p
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-6:
            return np.zeros(3)

        r_hat = r / r_norm
        Vc = -np.dot(v_rel, r_hat)
        omega = np.cross(r, v_rel) / (r_norm ** 2)
        a_cmd = self.fallback_gain * Vc * np.cross(omega, r_hat)

        return clip_norm(a_cmd, self.a_max)
