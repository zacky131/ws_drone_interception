"""Velocity/yaw-rate Srivastava interception MPC port."""

from __future__ import annotations
import math
import time
import casadi as ca
import numpy as np

N_HORIZON = 20
DT_S = 0.1
HORIZON_S = 2.0
MPC_PERIOD_S = 0.1
V_MAX_MPS = 15.0
YAW_RATE_MAX_RADPS = math.radians(45.0)
Q_DIAGONAL = np.array([0.5, 0.5, 1.0, 180.0 / math.pi])
R_DIAGONAL = np.array([0.8, 0.8, 0.8, 0.5])
CLOSING_OFFSET_MPS = 0.5


def causal_ca_reference(state, count=N_HORIZON + 1):
    state = np.asarray(state, float).reshape(9)
    times = np.arange(count, dtype=float) * DT_S
    return state[:3] + times[:, None] * state[3:6] + 0.5 * times[:, None] ** 2 * state[6:9]


def propagate_follower(state, control):
    return np.asarray(state, float) + DT_S * np.asarray(control, float)


def stage_cost(error, control):
    e, u = np.asarray(error, float), np.asarray(control, float)
    return float(np.sum(Q_DIAGONAL * e**2) + np.sum(R_DIAGONAL * u**2))


def closing_heuristic(inertial_velocity, yaw):
    v = np.asarray(inertial_velocity, float)
    c, s = math.cos(yaw), math.sin(yaw)
    body = np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]])
    forward = min(V_MAX_MPS, CLOSING_OFFSET_MPS + math.hypot(body[0], body[1]))
    body_command = np.array([forward, 0.0, body[2]])
    inertial = np.array([c * forward, s * forward, body[2]])
    return inertial, body_command


class SrivastavaMPC:
    def __init__(self):
        # The published sum is k=0,...,N, so retain N+1 control variables.
        # u_N has no successor dynamics and is driven to zero by its R term.
        u = ca.SX.sym("u", 4, N_HORIZON + 1)
        x0 = ca.SX.sym("x0", 4)
        target = ca.SX.sym("target", 3, N_HORIZON + 1)
        x = x0
        objective = 0
        for k in range(N_HORIZON + 1):
            # The LOS reference is frozen causally at the MPC update from the
            # current follower position to each CA target-reference point.
            delta = target[:, k] - x0[:3]
            psi_ref = ca.atan2(delta[1], delta[0])
            psi_error = x[3] - psi_ref
            error = ca.vertcat(x[:3] - target[:, k], psi_error)
            uk = u[:, k]
            objective += ca.dot(ca.DM(Q_DIAGONAL), error**2) + ca.dot(ca.DM(R_DIAGONAL), uk**2)
            if k < N_HORIZON:
                x = x + DT_S * uk
        nlp = {"x": ca.reshape(u, -1, 1), "p": ca.vertcat(x0, ca.reshape(target, -1, 1)), "f": objective}
        options = {
            "qpsol": "qrqp", "print_header": False, "print_iteration": False,
            "print_status": False, "max_iter": 100,
            "hessian_approximation": "limited-memory",
            "qpsol_options": {"print_iter": False, "print_header": False},
        }
        self.solver = ca.nlpsol("srivastava_sqp", "sqpmethod", nlp, options)
        self.lower = np.tile([-V_MAX_MPS, -V_MAX_MPS, -V_MAX_MPS, -YAW_RATE_MAX_RADPS], N_HORIZON + 1)
        self.upper = -self.lower
        self.reset()

    def reset(self):
        self.last_solve_s = -math.inf
        self.active_control = np.zeros(4)
        self.guess = np.zeros(4 * (N_HORIZON + 1))
        self.attempts = self.successes = self.failures = 0

    def command(self, now_s, follower_state, target_state):
        attempted = not np.isfinite(self.last_solve_s) or now_s - self.last_solve_s >= MPC_PERIOD_S - 1e-9
        solve_time = 0.0
        status = "held_active_velocity"
        solve_succeeded = True
        if attempted:
            self.last_solve_s = float(now_s); self.attempts += 1
            reference = causal_ca_reference(target_state)
            follower = np.asarray(follower_state, float).reshape(4)
            if self.successes == 0:
                seed_velocity = (reference[-1] - follower[:3]) / HORIZON_S
                seed_norm = np.linalg.norm(seed_velocity)
                if seed_norm > V_MAX_MPS: seed_velocity *= V_MAX_MPS / seed_norm
                seed_yaw = math.atan2(reference[-1, 1] - follower[1], reference[-1, 0] - follower[0])
                seed_rate = np.clip((seed_yaw - follower[3]) / HORIZON_S, -YAW_RATE_MAX_RADPS, YAW_RATE_MAX_RADPS)
                self.guess = np.tile(np.r_[seed_velocity, seed_rate], N_HORIZON + 1)
            parameters = np.concatenate([follower, reference.reshape(-1, order="F")])
            start = time.perf_counter_ns()
            try:
                result = self.solver(x0=self.guess, p=parameters, lbx=self.lower, ubx=self.upper)
                solve_time = (time.perf_counter_ns() - start) * 1e-9
                solution = np.asarray(result["x"], float).reshape(4, N_HORIZON + 1, order="F")
                if not np.all(np.isfinite(solution)):
                    raise FloatingPointError("nonfinite SQP solution")
                solver_status = str(self.solver.stats().get("return_status", ""))
                if solver_status not in {"Solve_Succeeded", "Search_Direction_Becomes_Too_Small"}:
                    raise RuntimeError(f"SQP status {solver_status}")
                self.guess = solution.reshape(-1, order="F")
                self.active_control = solution[:, 0]
                self.successes += 1; status = solver_status
            except Exception as exc:
                solve_time = (time.perf_counter_ns() - start) * 1e-9
                self.failures += 1; solve_succeeded = False
                status = f"{type(exc).__name__}: {str(exc).splitlines()[-1]}"
        velocity, body = closing_heuristic(self.active_control[:3], float(follower_state[3]))
        yaw_rate = float(np.clip(self.active_control[3], -YAW_RATE_MAX_RADPS, YAW_RATE_MAX_RADPS))
        return velocity, yaw_rate, {
            "solver_success": solve_succeeded,
            "solver_status": status, "solve_time_s": solve_time,
            "srivastava_replan_attempted": int(attempted), "srivastava_mpc_rate_hz": 10.0,
            "srivastava_body_forward_mps": float(body[0]), "srivastava_body_lateral_mps": 0.0,
            "srivastava_body_vertical_mps": float(body[2]), "srivastava_yaw_rate_radps": yaw_rate,
            "srivastava_solve_attempts_total": self.attempts,
            "srivastava_solve_successes_total": self.successes,
            "srivastava_solve_failures_total": self.failures,
            "target_rollout_used": 1,
        }
