"""Translational single-target port of published variable-time-step MPC."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
import time
from typing import Any

import casadi as ca
import numpy as np


N_HORIZON = 12
C_MAX_S = 4.0
T_MIN_S = 0.1
T_MAX_S = 0.9
REPLAN_PERIOD_S = 0.9
V_MAX_MPS = 15.0
A_MAX_MPS2 = 20.0
J_MAX_MPS3 = 30.0
CAPTURE_SCALE_M = 1.0
BUTTERWORTH_ORDER = 2
ALPHA_G = 0.0
INITIAL_REWARD = 1.0
INPUT_CHANGE_WEIGHT = 0.0001
TERMINAL_DISTANCE_WEIGHT = 1000.0
IPOPT_MAX_ITERATIONS = 100_000
IPOPT_MAX_CPU_TIME_S = 5.0


def variable_step_propagate(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    jerk: np.ndarray,
    timestep_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Published jerk-input p/v/a transition."""

    p = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)
    a = np.asarray(acceleration, dtype=float)
    j = np.asarray(jerk, dtype=float)
    dt = float(timestep_s)
    return (
        p + v * dt + 0.5 * a * dt**2 + (1.0 / 6.0) * j * dt**3,
        v + a * dt + 0.5 * j * dt**2,
        a + j * dt,
    )


def causal_ca_rollout(target_state: np.ndarray, times_s: np.ndarray) -> np.ndarray:
    state = np.asarray(target_state, dtype=float).reshape(9)
    times = np.asarray(times_s, dtype=float)
    if np.any(times < 0.0):
        raise ValueError("target rollout cannot use negative/future-truth indices")
    return (
        state[None, :3]
        + times[:, None] * state[None, 3:6]
        + 0.5 * times[:, None] ** 2 * state[None, 6:9]
    )


@dataclass(frozen=True)
class VariableTimePlan:
    created_at_s: float
    timesteps_s: np.ndarray
    node_times_s: np.ndarray
    positions_m: np.ndarray
    velocities_mps: np.ndarray
    accelerations_mps2: np.ndarray
    jerks_mps3: np.ndarray
    target_positions_m: np.ndarray
    objective: float
    solve_time_s: float
    solver_status: str


class VariableTimeStepMPC:
    """IPOPT port preserving variable timesteps and the 4-s horizon."""

    def __init__(self) -> None:
        self._build_problem()
        self.reset()

    def _build_problem(self) -> None:
        opti = ca.Opti()
        jerks = opti.variable(3, N_HORIZON)
        timesteps = opti.variable(N_HORIZON)
        initial_state = opti.parameter(9)
        target_state = opti.parameter(9)
        previous_jerk = opti.parameter(3)

        opti.subject_to(opti.bounded(T_MIN_S, timesteps, T_MAX_S))
        opti.subject_to(timesteps[0] == T_MIN_S)
        opti.subject_to(ca.sum1(timesteps) == C_MAX_S)

        position = initial_state[:3]
        velocity = initial_state[3:6]
        acceleration = initial_state[6:9]
        cumulative_time = 0
        reward = INITIAL_REWARD
        objective = 0
        previous = previous_jerk
        positions = [position]
        velocities = [velocity]
        accelerations = [acceleration]
        node_times = [cumulative_time]

        for index in range(N_HORIZON):
            dt = timesteps[index]
            jerk = jerks[:, index]
            target_position = (
                target_state[:3]
                + target_state[3:6] * cumulative_time
                + 0.5 * target_state[6:9] * cumulative_time**2
            )
            distance_squared = ca.sumsqr(position - target_position)
            collection = 1.0 / (1.0 + distance_squared / CAPTURE_SCALE_M**2)
            objective += reward + INPUT_CHANGE_WEIGHT * ca.sumsqr(jerk - previous)
            reward = reward * (1.0 - collection)
            position = (
                position + velocity * dt + 0.5 * acceleration * dt**2
                + (1.0 / 6.0) * jerk * dt**3
            )
            velocity = velocity + acceleration * dt + 0.5 * jerk * dt**2
            acceleration = acceleration + jerk * dt
            cumulative_time = cumulative_time + dt
            previous = jerk
            opti.subject_to(ca.sumsqr(jerk) <= J_MAX_MPS3**2)
            opti.subject_to(ca.sumsqr(velocity) <= V_MAX_MPS**2)
            opti.subject_to(ca.sumsqr(acceleration) <= A_MAX_MPS2**2)
            positions.append(position)
            velocities.append(velocity)
            accelerations.append(acceleration)
            node_times.append(cumulative_time)

        terminal_target = (
            target_state[:3]
            + target_state[3:6] * cumulative_time
            + 0.5 * target_state[6:9] * cumulative_time**2
        )
        objective += reward + TERMINAL_DISTANCE_WEIGHT * ca.sumsqr(position - terminal_target)
        opti.minimize(objective)

        plugin_options = {"expand": True, "print_time": False}
        solver_options = {
            "linear_solver": "mumps",
            "print_level": 0,
            "sb": "yes",
            "max_iter": IPOPT_MAX_ITERATIONS,
            "max_cpu_time": IPOPT_MAX_CPU_TIME_S,
            "tol": 1e-6,
            "acceptable_tol": 1e-5,
            "acceptable_iter": 5,
            "print_timing_statistics": "no",
        }
        opti.solver("ipopt", plugin_options, solver_options)
        self.opti = opti
        self.jerks_var = jerks
        self.timesteps_var = timesteps
        self.initial_state_param = initial_state
        self.target_state_param = target_state
        self.previous_jerk_param = previous_jerk
        self.objective_expr = objective

    @staticmethod
    def feasible_initial_timesteps() -> np.ndarray:
        values = np.full(N_HORIZON, (C_MAX_S - T_MIN_S) / (N_HORIZON - 1))
        values[0] = T_MIN_S
        return values

    def reset(self) -> None:
        old_executor = getattr(self, "_executor", None)
        if old_executor is not None:
            old_executor.shutdown(wait=False, cancel_futures=True)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vtmpc_ipopt")
        self._pending: Future | None = None
        self.active_plan: VariableTimePlan | None = None
        self.last_replan_attempt_s = -math.inf
        self.last_successful_replan_s = -math.inf
        self.last_jerk = np.zeros(3)
        self.solve_attempts = 0
        self.solve_successes = 0
        self.solve_failures = 0
        self._warm_jerks = np.zeros((3, N_HORIZON))
        self._warm_timesteps = self.feasible_initial_timesteps()
        self._last_async_solver_info: dict[str, Any] = {
            "solver_success": True,
            "solver_status": "not_started",
            "solve_time_s": 0.0,
        }

    @staticmethod
    def causal_initial_jerk_guess(
        interceptor_state: np.ndarray, target_state: np.ndarray
    ) -> np.ndarray:
        """Constraint-clipped numerical seed aimed at the causal terminal target."""

        state = np.asarray(interceptor_state, dtype=float).reshape(9)
        target = np.asarray(target_state, dtype=float).reshape(9)
        terminal_target = causal_ca_rollout(target, np.array([C_MAX_S]))[0]
        desired_acceleration = (
            2.0 * (terminal_target - state[:3] - state[3:6] * C_MAX_S)
            / C_MAX_S**2
        )
        acceleration_norm = float(np.linalg.norm(desired_acceleration))
        if acceleration_norm > A_MAX_MPS2:
            desired_acceleration *= A_MAX_MPS2 / acceleration_norm
        first_jerk = (desired_acceleration - state[6:9]) / T_MIN_S
        jerk_norm = float(np.linalg.norm(first_jerk))
        if jerk_norm > J_MAX_MPS3:
            first_jerk *= J_MAX_MPS3 / jerk_norm
        guess = np.zeros((3, N_HORIZON))
        guess[:, 0] = first_jerk
        return guess

    def _validate_solution(self, timesteps: np.ndarray, jerks: np.ndarray) -> None:
        if not np.all(np.isfinite(timesteps)) or not np.all(np.isfinite(jerks)):
            raise FloatingPointError("VT-MPC solution is nonfinite")
        if abs(float(np.sum(timesteps)) - C_MAX_S) > 1e-5:
            raise ValueError("VT-MPC horizon-time equality failed")
        if abs(float(timesteps[0]) - T_MIN_S) > 1e-6:
            raise ValueError("VT-MPC first timestep equality failed")
        if np.min(timesteps) < T_MIN_S - 1e-6 or np.max(timesteps) > T_MAX_S + 1e-6:
            raise ValueError("VT-MPC timestep bounds failed")
        if np.max(np.linalg.norm(jerks, axis=0)) > J_MAX_MPS3 + 1e-4:
            raise ValueError("VT-MPC jerk constraint failed")

    def solve(
        self,
        now_s: float,
        interceptor_state: np.ndarray,
        target_state: np.ndarray,
        previous_jerk_override: np.ndarray | None = None,
    ) -> tuple[VariableTimePlan | None, dict[str, Any]]:
        state = np.asarray(interceptor_state, dtype=float).reshape(9)
        target = np.asarray(target_state, dtype=float).reshape(9)
        self.solve_attempts += 1
        if np.linalg.norm(state[3:6]) > V_MAX_MPS + 1e-9:
            self.solve_failures += 1
            return None, {
                "solver_success": False,
                "solver_status": "initial_velocity_outside_frozen_bound",
                "solve_time_s": 0.0,
            }
        if np.linalg.norm(state[6:9]) > A_MAX_MPS2 + 1e-9:
            self.solve_failures += 1
            return None, {
                "solver_success": False,
                "solver_status": "initial_acceleration_outside_frozen_bound",
                "solve_time_s": 0.0,
            }
        self.opti.set_value(self.initial_state_param, state)
        self.opti.set_value(self.target_state_param, target)
        previous_jerk = (
            self.last_jerk.copy()
            if previous_jerk_override is None
            else np.asarray(previous_jerk_override, dtype=float).reshape(3)
        )
        self.opti.set_value(self.previous_jerk_param, previous_jerk)
        jerk_guess = (
            self._warm_jerks if self.solve_successes
            else self.causal_initial_jerk_guess(state, target)
        )
        self.opti.set_initial(self.jerks_var, jerk_guess)
        self.opti.set_initial(self.timesteps_var, self._warm_timesteps)
        start = time.perf_counter_ns()
        try:
            solution = self.opti.solve()
            solve_time_s = (time.perf_counter_ns() - start) * 1e-9
            timesteps = np.asarray(solution.value(self.timesteps_var), dtype=float).reshape(N_HORIZON)
            jerks = np.asarray(solution.value(self.jerks_var), dtype=float).reshape(3, N_HORIZON)
            self._validate_solution(timesteps, jerks)
            positions = [state[:3].copy()]
            velocities = [state[3:6].copy()]
            accelerations = [state[6:9].copy()]
            for index in range(N_HORIZON):
                p, v, a = variable_step_propagate(
                    positions[-1], velocities[-1], accelerations[-1],
                    jerks[:, index], timesteps[index],
                )
                positions.append(p)
                velocities.append(v)
                accelerations.append(a)
            positions_array = np.asarray(positions)
            velocities_array = np.asarray(velocities)
            accelerations_array = np.asarray(accelerations)
            if np.max(np.linalg.norm(velocities_array, axis=1)) > V_MAX_MPS + 1e-3:
                raise ValueError("VT-MPC velocity constraint failed")
            if np.max(np.linalg.norm(accelerations_array, axis=1)) > A_MAX_MPS2 + 1e-3:
                raise ValueError("VT-MPC acceleration constraint failed")
            node_times = np.concatenate(([0.0], np.cumsum(timesteps)))
            target_positions = causal_ca_rollout(target, node_times)
            status = str(self.opti.stats().get("return_status", "Solve_Succeeded"))
            plan = VariableTimePlan(
                created_at_s=float(now_s),
                timesteps_s=timesteps,
                node_times_s=node_times,
                positions_m=positions_array,
                velocities_mps=velocities_array,
                accelerations_mps2=accelerations_array,
                jerks_mps3=jerks.T,
                target_positions_m=target_positions,
                objective=float(solution.value(self.objective_expr)),
                solve_time_s=solve_time_s,
                solver_status=status,
            )
            self.solve_successes += 1
            self._warm_jerks = jerks.copy()
            self._warm_timesteps = timesteps.copy()
            self.last_jerk = jerks[:, 0].copy()
            return plan, {"solver_success": True, "solver_status": status, "solve_time_s": solve_time_s}
        except Exception as exc:
            solve_time_s = (time.perf_counter_ns() - start) * 1e-9
            self.solve_failures += 1
            return None, {
                "solver_success": False,
                "solver_status": f"{type(exc).__name__}: {str(exc).splitlines()[-1]}",
                "solve_time_s": solve_time_s,
            }

    def _sample_plan(self, now_s: float) -> tuple[np.ndarray, int, float]:
        if self.active_plan is None:
            return np.zeros(3), -1, math.nan
        elapsed = max(0.0, float(now_s) - self.active_plan.created_at_s)
        if elapsed >= self.active_plan.node_times_s[-1]:
            return self.active_plan.accelerations_mps2[-1].copy(), N_HORIZON, elapsed
        index = int(np.searchsorted(self.active_plan.node_times_s, elapsed, side="right") - 1)
        index = min(max(index, 0), N_HORIZON - 1)
        local_time = elapsed - self.active_plan.node_times_s[index]
        command = (
            self.active_plan.accelerations_mps2[index]
            + self.active_plan.jerks_mps3[index] * local_time
        )
        self.last_jerk = self.active_plan.jerks_mps3[index].copy()
        return command, index, elapsed

    def command(
        self,
        now_s: float,
        interceptor_state: np.ndarray,
        target_state: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        replan_attempted = bool(
            not np.isfinite(self.last_replan_attempt_s)
            or float(now_s) - self.last_replan_attempt_s >= REPLAN_PERIOD_S - 1e-9
        )
        replan_interval = (
            math.nan if not np.isfinite(self.last_replan_attempt_s)
            else float(now_s) - self.last_replan_attempt_s
        )
        solver = {"solver_success": True, "solver_status": "not_replanned", "solve_time_s": 0.0}
        reused_previous = False
        if replan_attempted:
            self.last_replan_attempt_s = float(now_s)
            plan, solver = self.solve(now_s, interceptor_state, target_state)
            if plan is not None:
                self.active_plan = plan
                self.last_successful_replan_s = float(now_s)
            else:
                reused_previous = self.active_plan is not None
        command, active_segment, plan_elapsed = self._sample_plan(now_s)
        plan = self.active_plan
        diagnostics: dict[str, Any] = {
            **solver,
            "vtmpc_replan_attempted": int(replan_attempted),
            "vtmpc_replan_interval_s": replan_interval,
            "vtmpc_reused_previous_plan": int(reused_previous),
            "vtmpc_has_valid_plan": int(plan is not None),
            "vtmpc_active_segment": active_segment,
            "vtmpc_plan_elapsed_s": plan_elapsed,
            "vtmpc_solve_attempts_total": self.solve_attempts,
            "vtmpc_solve_successes_total": self.solve_successes,
            "vtmpc_solve_failures_total": self.solve_failures,
            "vtmpc_target_prediction_source": "causal_current_posterior_CA_rollout",
            "target_rollout_used": 1,
        }
        if plan is not None:
            diagnostics.update(
                {
                    "vtmpc_timestep_sum_s": float(np.sum(plan.timesteps_s)),
                    "vtmpc_first_timestep_s": float(plan.timesteps_s[0]),
                    "vtmpc_min_timestep_s": float(np.min(plan.timesteps_s)),
                    "vtmpc_max_timestep_s": float(np.max(plan.timesteps_s)),
                    "vtmpc_max_velocity_mps": float(np.max(np.linalg.norm(plan.velocities_mps, axis=1))),
                    "vtmpc_max_acceleration_mps2": float(np.max(np.linalg.norm(plan.accelerations_mps2, axis=1))),
                    "vtmpc_max_jerk_mps3": float(np.max(np.linalg.norm(plan.jerks_mps3, axis=1))),
                    "vtmpc_objective": plan.objective,
                    "vtmpc_plan_position_horizon": plan.positions_m[1:],
                    "vtmpc_target_position_horizon": plan.target_positions_m[1:],
                    "vtmpc_timesteps_s": plan.timesteps_s,
                }
            )
        return command, diagnostics

    def command_async(
        self,
        now_s: float,
        interceptor_state: np.ndarray,
        target_state: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Keep the 50-Hz command stream alive while IPOPT replans in one worker."""

        replan_completed = False
        reused_previous = False
        completion_info = self._last_async_solver_info
        if self._pending is not None and self._pending.done():
            replan_completed = True
            try:
                plan, completion_info = self._pending.result()
            except Exception as exc:  # defensive boundary around worker execution
                plan = None
                completion_info = {
                    "solver_success": False,
                    "solver_status": f"worker_{type(exc).__name__}: {exc}",
                    "solve_time_s": 0.0,
                }
            self._pending = None
            self._last_async_solver_info = dict(completion_info)
            if plan is not None:
                self.active_plan = plan
                self.last_successful_replan_s = plan.created_at_s
            else:
                reused_previous = self.active_plan is not None

        replan_attempted = bool(
            self._pending is None
            and (
                not np.isfinite(self.last_replan_attempt_s)
                or float(now_s) - self.last_replan_attempt_s >= REPLAN_PERIOD_S - 1e-9
            )
        )
        replan_interval = (
            math.nan if not np.isfinite(self.last_replan_attempt_s)
            else float(now_s) - self.last_replan_attempt_s
        )
        if replan_attempted:
            self.last_replan_attempt_s = float(now_s)
            self._pending = self._executor.submit(
                self.solve,
                float(now_s),
                np.asarray(interceptor_state, dtype=float).copy(),
                np.asarray(target_state, dtype=float).copy(),
                self.last_jerk.copy(),
            )

        command, active_segment, plan_elapsed = self._sample_plan(now_s)
        plan = self.active_plan
        diagnostics: dict[str, Any] = {
            **completion_info,
            "solver_status": (
                "ipopt_pending" if self._pending is not None and not replan_completed
                else completion_info.get("solver_status", "not_started")
            ),
            "solve_time_s": (
                float(completion_info.get("solve_time_s", 0.0)) if replan_completed else 0.0
            ),
            "vtmpc_replan_attempted": int(replan_attempted),
            "vtmpc_replan_completed": int(replan_completed),
            "vtmpc_replan_interval_s": replan_interval,
            "vtmpc_reused_previous_plan": int(reused_previous),
            "vtmpc_has_valid_plan": int(plan is not None),
            "vtmpc_solver_pending": int(self._pending is not None),
            "vtmpc_active_segment": active_segment,
            "vtmpc_plan_elapsed_s": plan_elapsed,
            "vtmpc_solve_attempts_total": self.solve_attempts,
            "vtmpc_solve_successes_total": self.solve_successes,
            "vtmpc_solve_failures_total": self.solve_failures,
            "vtmpc_target_prediction_source": "causal_current_posterior_CA_rollout",
            "target_rollout_used": 1,
        }
        if plan is not None:
            diagnostics.update(
                {
                    "vtmpc_timestep_sum_s": float(np.sum(plan.timesteps_s)),
                    "vtmpc_first_timestep_s": float(plan.timesteps_s[0]),
                    "vtmpc_min_timestep_s": float(np.min(plan.timesteps_s)),
                    "vtmpc_max_timestep_s": float(np.max(plan.timesteps_s)),
                    "vtmpc_max_velocity_mps": float(np.max(np.linalg.norm(plan.velocities_mps, axis=1))),
                    "vtmpc_max_acceleration_mps2": float(np.max(np.linalg.norm(plan.accelerations_mps2, axis=1))),
                    "vtmpc_max_jerk_mps3": float(np.max(np.linalg.norm(plan.jerks_mps3, axis=1))),
                    "vtmpc_objective": plan.objective,
                    "vtmpc_plan_position_horizon": plan.positions_m[1:],
                    "vtmpc_target_position_horizon": plan.target_positions_m[1:],
                    "vtmpc_timesteps_s": plan.timesteps_s,
                }
            )
        return command, diagnostics
