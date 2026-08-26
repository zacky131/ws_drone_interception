"""Validated 3.0 s nonuniform-grid capture-point ACADOS MPC."""

from __future__ import annotations

import atexit
from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
import uuid

import numpy as np
import yaml


@dataclass(frozen=True)
class NonuniformGrid:
    dts_s: np.ndarray
    times_s: np.ndarray

    @property
    def stages(self) -> int:
        return len(self.dts_s)

    @property
    def duration_s(self) -> float:
        return float(self.dts_s.sum())

    def nearest_node(self, candidate_time_s: float) -> tuple[int, float]:
        index = int(np.argmin(np.abs(self.times_s - float(candidate_time_s))))
        return index + 1, float(self.times_s[index])


def make_grid() -> NonuniformGrid:
    dts = np.asarray([.02] * 20 + [.05] * 16 + [.10] * 18, dtype=float)
    times = np.cumsum(dts)
    grid = NonuniformGrid(dts, times)
    if grid.stages != 54 or not math.isclose(grid.duration_s, 3.0, abs_tol=1e-12):
        raise AssertionError("invalid nonuniform horizon")
    return grid


def map_candidate_times(grid: NonuniformGrid, candidates_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mapped = [grid.nearest_node(value) for value in np.asarray(candidates_s, dtype=float)]
    nodes = np.asarray([item[0] for item in mapped], dtype=int)
    times = np.asarray([item[1] for item in mapped], dtype=float)
    if len(np.unique(nodes)) != len(nodes):
        raise ValueError("candidate times must map to unique grid nodes")
    return nodes, times


def discrete_pursuer_step(
    state: np.ndarray,
    command: np.ndarray,
    dt_s: float,
    actuator_time_constant_s: float,
) -> np.ndarray:
    state = np.asarray(state, dtype=float)
    command = np.asarray(command, dtype=float)
    p, v, acceleration, _ = np.split(state, 4)
    alpha = min(float(dt_s) / max(float(actuator_time_constant_s), 1e-9), 1.0)
    acceleration_next = acceleration + alpha * (command - acceleration)
    velocity_next = v + acceleration_next * dt_s
    position_next = p + v * dt_s + .5 * acceleration_next * dt_s * dt_s
    return np.concatenate([position_next, velocity_next, acceleration_next, command])


def scaled_stage_cost(
    position_error: np.ndarray,
    velocity_error: np.ndarray,
    command: np.ndarray,
    command_rate: np.ndarray,
    dt_s: float,
    weights: tuple[float, float, float, float],
) -> float:
    qp, qv, control, rate = weights
    return float(dt_s) * (
        qp * float(np.dot(position_error, position_error))
        + qv * float(np.dot(velocity_error, velocity_error))
        + control * float(np.dot(command, command))
        + rate * float(np.dot(command_rate, command_rate))
    )


class NonuniformCaptureMPC:
    """One-solve capture-point OCP shared identically by M2 and M3."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text())
        self.grid = make_grid()
        constraints = self.config["physical_constraints"]
        self.vmax = float(constraints["max_velocity_mps"])
        self.amax = float(constraints["max_acceleration_mps2"])
        self.amax_axis = float(constraints["max_acceleration_per_axis_mps2"])
        self.jerkmax = float(constraints["max_jerk_mps3"])
        self.tau = float(constraints["actuator_time_constant_s"])
        self.weights = {key: float(value) for key, value in self.config["mpc_weights"].items()}
        self.previous_command = np.zeros(3)
        self._build_solver()

    def reset(self) -> None:
        self.previous_command = np.zeros(3)

    def _build_solver(self) -> None:
        import casadi as ca
        from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

        ocp = AcadosOcp()
        model = AcadosModel()
        model.name = "dapcs_nonuniform_capture_mpc"
        position = ca.SX.sym("position", 3)
        velocity = ca.SX.sym("velocity", 3)
        acceleration = ca.SX.sym("acceleration", 3)
        previous = ca.SX.sym("previous_command", 3)
        state = ca.vertcat(position, velocity, acceleration, previous)
        command = ca.SX.sym("command", 3)
        weak_position = ca.SX.sym("weak_position", 3)
        weak_velocity = ca.SX.sym("weak_velocity", 3)
        capture_position = ca.SX.sym("capture_position", 3)
        capture_velocity = ca.SX.sym("capture_velocity", 3)
        dt = ca.SX.sym("dt", 1)
        capture_weight = ca.SX.sym("capture_weight", 1)
        parameters = ca.vertcat(
            weak_position, weak_velocity, capture_position, capture_velocity, dt, capture_weight
        )
        alpha = ca.fmin(dt / self.tau, 1.0)
        acceleration_next = acceleration + alpha * (command - acceleration)
        velocity_next = velocity + acceleration_next * dt
        position_next = position + velocity * dt + .5 * acceleration_next * dt * dt
        model.x = state
        model.u = command
        model.p = parameters
        model.disc_dyn_expr = ca.vertcat(position_next, velocity_next, acceleration_next, command)
        command_rate = (command - previous) / ca.fmax(dt, 1e-6)
        weak_cost = (
            self.weights["weak_position"] * ca.sumsqr(position_next - weak_position)
            + self.weights["weak_velocity"] * ca.sumsqr(velocity_next - weak_velocity)
        )
        capture_cost = capture_weight * (
            self.weights["capture_position"] * ca.sumsqr(position_next - capture_position)
            + self.weights["capture_velocity"] * ca.sumsqr(velocity_next - capture_velocity)
        )
        effort_cost = (
            self.weights["control"] * ca.sumsqr(command)
            + self.weights["command_rate"] * ca.sumsqr(command_rate)
        )
        model.cost_expr_ext_cost = dt * (weak_cost + capture_cost + effort_cost)
        model.cost_expr_ext_cost_e = self.weights["terminal_weak_position"] * ca.sumsqr(
            position - weak_position
        )
        model.con_h_expr = ca.vertcat(
            ca.sumsqr(command), ca.sumsqr(velocity_next), ca.sumsqr(command_rate)
        )
        model.con_h_expr_e = ca.vertcat(ca.sumsqr(velocity))
        ocp.model = model
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"
        ocp.constraints.lbu = np.full(3, -self.amax_axis)
        ocp.constraints.ubu = np.full(3, self.amax_axis)
        ocp.constraints.idxbu = np.arange(3)
        ocp.constraints.lh = np.zeros(3)
        ocp.constraints.uh = np.asarray([self.amax**2, self.vmax**2, self.jerkmax**2])
        ocp.constraints.lh_e = np.zeros(1)
        ocp.constraints.uh_e = np.asarray([self.vmax**2])
        ocp.constraints.lbx_0 = np.zeros(12)
        ocp.constraints.ubx_0 = np.zeros(12)
        ocp.constraints.idxbx_0 = np.arange(12)
        ocp.parameter_values = np.zeros(14)
        ocp.parameter_values[12] = self.grid.dts_s[0]
        ocp.dims.N = self.grid.stages
        solver_config = self.config["solver"]
        ocp.solver_options.tf = self.grid.duration_s
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = str(solver_config["nlp_solver_type"])
        ocp.solver_options.qp_solver = str(solver_config["qp_solver"])
        ocp.solver_options.nlp_solver_max_iter = int(solver_config["max_iterations"])
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.print_level = int(solver_config["print_level"])
        unique = uuid.uuid4().hex[:10]
        export = Path(tempfile.gettempdir()) / "drone_interception_acados" / f"dapcs_{unique}"
        export.mkdir(parents=True, exist_ok=True)
        ocp.code_export_directory = str(export)
        self.export_directory = export
        self.solver = AcadosOcpSolver(
            ocp, json_file=str(export / f"dapcs_{unique}.json"), build=True, generate=True
        )
        atexit.register(shutil.rmtree, export, ignore_errors=True)

    def compute_command(
        self,
        pursuer_state: np.ndarray,
        weak_positions: np.ndarray,
        weak_velocities: np.ndarray,
        selected_stage_index: int,
        capture_position: np.ndarray,
        capture_velocity: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float | int | bool]]:
        start = time.perf_counter_ns()
        state = np.concatenate([np.asarray(pursuer_state, dtype=float)[:9], self.previous_command])
        self.solver.set(0, "lbx", state)
        self.solver.set(0, "ubx", state)
        for stage in range(self.grid.stages):
            parameters = np.concatenate([
                weak_positions[stage], weak_velocities[stage], capture_position, capture_velocity,
                [self.grid.dts_s[stage], float(stage == selected_stage_index)],
            ])
            self.solver.set(stage, "p", parameters)
        terminal_parameters = np.concatenate([
            weak_positions[-1], weak_velocities[-1], capture_position, capture_velocity,
            [self.grid.dts_s[-1], 0.0],
        ])
        self.solver.set(self.grid.stages, "p", terminal_parameters)
        status = int(self.solver.solve())
        solve_time = (time.perf_counter_ns() - start) * 1e-9
        command = np.asarray(self.solver.get(0, "u"), dtype=float) if status in (0, 2) else np.zeros(3)
        self.previous_command = command.copy()
        return command, {
            "solver_status": status,
            "solver_success": status in (0, 2),
            "solve_time_s": solve_time,
        }
