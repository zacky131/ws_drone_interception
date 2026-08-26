"""
MPC with constant-acceleration target prediction (ablation variant ``mpc_ca``).

This module provides :class:`MPCConstantAcceleration`, which pairs the
:class:`~src.estimation.rls_baseline_estimator.RLSBaselineEstimator` with an
MPC whose target prediction model includes acceleration but **omits jerk**.
This isolates the contribution of jerk estimation relative to ``mpc_cv``.

Design choice isolated
    ``mpc_ca`` isolates the value of **jerk estimation** by comparing with
    ``mpc_cv`` (where no estimator runs at all).  It differs from DTAMPC-RLS
    in that (a) the jerk estimate from RLS is forced to zero in the target
    prediction, and (b) only a quadratic (constant-acceleration) polynomial is
    used:

        p̂_T(n|k) = p_T(k) + v_T(k) · τ_n + ½ â_T · τ_n²
        v̂_T(n|k) = v_T(k) + â_T · τ_n

    where â_T is the RLS acceleration estimate and the jerk term is
    intentionally omitted.  All other MPC parameters (horizon N, weights,
    constraints) are identical to :class:`~src.baselines.standard_mpc.StandardMPC`
    and therefore to DTAMPC-RLS.  It corresponds to the ``mpc_ca`` row in
    Table II ("RLS accel only, CA prediction").
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


class MPCConstantAcceleration(ControllerBase):
    """MPC with constant-acceleration prediction; RLS provides â_T; jerk forced to zero.

    This class isolates the design choice of **including acceleration
    estimation but excluding jerk** from the target prediction.  It differs
    from DTAMPC-RLS in that the cubic jerk term is intentionally suppressed
    even though the RLS estimator does compute a jerk estimate internally
    (that estimate is ignored).  Comparing ``mpc_ca`` with ``mpc_rls_linear``
    therefore shows only the value of running the full RLS window rather than
    just the acceleration portion.  This is the ``mpc_ca`` ablation row in
    Table II.

    All MPC hyperparameters are loaded exclusively from the config objects
    passed to the constructor — no values are hardcoded.

    Parameters
    ----------
    ctrl_cfg : ControllerConfig
    pursuer_cfg : PursuerConfig
    sim_cfg : SimulationConfig
    estimator_cfg : EstimatorConfig
    """

    def __init__(
        self,
        ctrl_cfg: ControllerConfig,
        pursuer_cfg: PursuerConfig,
        sim_cfg: SimulationConfig,
    ) -> None:
        self._solver_type = getattr(ctrl_cfg, "solver", "casadi")
        if self._solver_type == "acados":
            if not _HAS_ACADOS:
                raise ImportError("acados_template is required for solver='acados'.")
        elif not _HAS_CASADI:
            raise ImportError("CasADi is required for MPCConstantAcceleration.")

        self.N: int = ctrl_cfg.horizon
        self.dt: float = sim_cfg.dt
        self.tau: float = pursuer_cfg.actuator_time_constant
        self.a_max: float = pursuer_cfg.max_acceleration
        self.a_max_axis: float = pursuer_cfg.max_acceleration_per_axis
        self.jerk_max: float = pursuer_cfg.max_jerk
        self.v_max: float = pursuer_cfg.max_velocity
        self.Q_pos: float = ctrl_cfg.Q_pos
        self.Q_vel: float = ctrl_cfg.Q_vel
        self.R: float = ctrl_cfg.R_control
        self.R_rate: float = ctrl_cfg.R_rate
        self.Q_T_pos: float = ctrl_cfg.Q_terminal_pos
        self.Q_T_vel: float = ctrl_cfg.Q_terminal_vel
        self.fallback_gain: float = ctrl_cfg.fallback_gain
        self._solver_max_iter: int = ctrl_cfg.solver_max_iter
        self._solver_print: int = ctrl_cfg.solver_print_level

        # No internal estimator — acceleration comes directly from the external
        # EKF estimate supplied by the simulation engine.
        self._u_prev: np.ndarray = np.zeros(3)
        self._warm_x0: Optional[np.ndarray] = None

        if self._solver_type == "acados":
            self._acados_solver, self._acados_export_dir = build_acados_mpc(
                ctrl_cfg, pursuer_cfg, sim_cfg, self.Q_pos, self.Q_T_pos
            )
        else:
            self._build_nlp()

    # ── NLP construction ──────────────────────────────────────────────────

    def _build_nlp(self) -> None:
        N = self.N
        dt = self.dt
        alpha = min(dt / self.tau, 1.0) if self.tau > 1e-9 else 1.0
        n_u = 3 * N

        U = ca.MX.sym("U", n_u)
        # P = [x0_pursuer(9), pt(3), vt(3), at(3), u_prev(3), wind(3)] = 24
        P = ca.MX.sym("P", 24)

        x0 = P[0:9]
        pt0 = P[9:12]
        vt0 = P[12:15]
        at0 = P[15:18]       # RLS acceleration estimate (jerk NOT included)
        u_prev_sym = P[18:21]
        wind = P[21:24]

        cost = 0.0
        g_list = []
        x = x0

        for k in range(N):
            u_k = U[3 * k : 3 * (k + 1)]
            p_p, v_p, a_app = x[0:3], x[3:6], x[6:9]

            # Actuator lag
            a_app_new = a_app + alpha * (u_k - a_app)
            total_acc = a_app_new + wind
            v_new = v_p + total_acc * dt
            p_new = p_p + v_p * dt + 0.5 * total_acc * dt ** 2
            x = ca.vertcat(p_new, v_new, a_app_new)

            # Constant-acceleration target prediction (jerk term omitted)
            t_pred = (k + 1) * dt
            p_t = pt0 + vt0 * t_pred + 0.5 * at0 * t_pred ** 2
            v_t = vt0 + at0 * t_pred

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
            "ipopt.max_iter": getattr(self, "_solver_max_iter", 100),
            "ipopt.print_level": getattr(self, "_solver_print", 0),
            "print_time": 0,
            "ipopt.sb": "yes",
            "ipopt.warm_start_init_point": "yes",
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
        }
        self._solver = ca.nlpsol("mpc_ca", "ipopt", nlp, opts)
        self._n_u = n_u
        self._lbx = np.full(n_u, -self.a_max_axis)
        self._ubx = np.full(n_u, self.a_max_axis)
        self._lbg = np.zeros(3 * N)
        self._ubg = np.tile([self.a_max ** 2, self.v_max ** 2, self.jerk_max ** 2], N)

    # ── ControllerBase implementation ─────────────────────────────────────

    def compute_control(
        self,
        pursuer_state: np.ndarray,
        target_measurement: Optional[np.ndarray],
        target_estimate: np.ndarray,
        wind_estimate: np.ndarray,
        t: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute commanded acceleration using constant-acceleration target prediction.

        Runs the internal RLS estimator to obtain â_T; forces ĵ_T to zero
        before populating the NLP parameter vector.

        Parameters
        ----------
        pursuer_state : (9,) array
        target_measurement : (6,) array or None
        target_estimate : (12,) array
        wind_estimate : (3,) array
        t : float

        Returns
        -------
        cmd : (3,) array
        info : dict
        """
        # Use the external EKF estimate directly — no internal estimator.
        # pos/vel/acc all come from the sim-engine EKF; jerk is NOT passed to the NLP.
        # This cleanly isolates the value of EKF acceleration vs. no acceleration (mpc_cv).
        pt = target_estimate[0:3]
        vt = target_estimate[3:6]
        at = target_estimate[6:9]   # EKF acceleration — jerk intentionally omitted

        if self._solver_type == "acados":
            return self._compute_control_acados(
                pursuer_state, target_estimate, pt, vt, at, wind_estimate
            )

        p_param = np.zeros(24)
        p_param[0:9] = pursuer_state
        p_param[9:12] = pt
        p_param[12:15] = vt
        p_param[15:18] = at          # acceleration only
        p_param[18:21] = self._u_prev
        p_param[21:24] = wind_estimate

        x0_guess = self._warm_x0 if self._warm_x0 is not None else np.zeros(self._n_u)

        t0 = _time.perf_counter()
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
        return cmd, {
            "solver_status": status,
            "solver_success": ok,
            "solve_time_s": solve_time,
            "estimator": "ekf",
        }

    def _compute_control_acados(
        self,
        pursuer_state: np.ndarray,
        target_estimate: np.ndarray,
        pt: np.ndarray,
        vt: np.ndarray,
        at: np.ndarray,
        wind_estimate: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        t0 = _time.perf_counter()
        ok = False
        try:
            x0 = np.concatenate([pursuer_state[0:9], self._u_prev])
            self._acados_solver.set(0, "lbx", x0)
            self._acados_solver.set(0, "ubx", x0)

            wind = wind_estimate
            for k in range(self.N):
                t_pred = (k + 1) * self.dt
                p_t = pt + vt * t_pred + 0.5 * at * t_pred ** 2
                v_t = vt + at * t_pred
                self._acados_solver.set(k, "p", np.concatenate([p_t, v_t, wind]))

            t_pred = self.N * self.dt
            p_t = pt + vt * t_pred + 0.5 * at * t_pred ** 2
            v_t = vt + at * t_pred
            self._acados_solver.set(self.N, "p", np.concatenate([p_t, v_t, wind]))

            status_code = self._acados_solver.solve()
            solve_time = _time.perf_counter() - t0

            if status_code in (0, 2):
                cmd = self._acados_solver.get(0, "u")[0:3]
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

        self._u_prev = np.asarray(cmd, dtype=float).copy()
        return cmd, {
            "solver_status": status,
            "solver_success": ok,
            "solve_time_s": solve_time,
            "estimator": "ekf",
        }

    def reset(self) -> None:
        """Reset warm-start buffer and previous control."""
        self._u_prev = np.zeros(3)
        self._warm_x0 = None

    def _fallback(self, pursuer_state: np.ndarray, target_est: np.ndarray) -> np.ndarray:
        r = target_est[0:3] - pursuer_state[0:3]
        v_rel = target_est[3:6] - pursuer_state[3:6]
        rn = np.linalg.norm(r)
        if rn < 1e-6:
            return np.zeros(3)
        rh = r / rn
        Vc = -np.dot(v_rel, rh)
        omega = np.cross(r, v_rel) / rn ** 2
        return clip_norm(self.fallback_gain * Vc * np.cross(omega, rh), self.a_max)