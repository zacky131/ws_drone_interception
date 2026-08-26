"""Finite two-X500 PX4/Gazebo interception trial supervisor.

The supervisor owns the complete trial state machine so that target commands,
telemetry impairment, controller execution, virtual capture, and logging share
one monotonic 50 Hz clock.  Existing interception algorithms are used only via
``ExistingControllerAdapter``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand
from px4_msgs.msg import VehicleLocalPosition, VehicleStatus

from .capture import CAPTURE_RADIUS_M, interpolated_crossing_time
from .controller_adapter import ExistingControllerAdapter, METHODS
from .frames import (
    enu_acceleration_to_ned,
    enu_position_to_ned,
    enu_velocity_to_ned,
    ned_acceleration_to_enu,
    ned_position_to_enu,
    ned_velocity_to_enu,
)
from .px4_contract import INTERCEPTOR, TARGET
from .single_vehicle_smoke import PX4_QOS
from .telemetry import (
    CONDITIONS,
    TargetStateSample,
    apply_schedule_row,
    generate_schedule,
    save_schedule,
)
from .trajectory import Trajectory


DT_S = 0.02
SPAWN_ENU = {
    INTERCEPTOR.role: np.array([0.0, 0.0, 0.0]),
    TARGET.role: np.array([30.0, 0.0, 0.0]),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


class ExperimentSupervisor(Node):
    def __init__(
        self,
        trajectory_path: Path,
        trajectory_id: str,
        family: str,
        condition_name: str,
        method: str,
        trial_seed: int,
        config_path: Path,
        output_dir: Path,
        timeout_s: float,
    ) -> None:
        super().__init__("experiment_supervisor")
        if condition_name not in CONDITIONS:
            raise ValueError(f"unknown condition {condition_name}")
        if method not in METHODS:
            raise ValueError(f"unknown method {method}")

        self.trajectory_path = trajectory_path.resolve()
        self.trajectory_id = trajectory_id
        self.family = family
        self.condition = CONDITIONS[condition_name]
        self.method = method
        self.trial_seed = int(trial_seed)
        self.config_path = config_path.resolve()
        self.output_dir = output_dir.resolve()
        self.timeout_s = float(timeout_s)
        self.trial_id = f"{trajectory_id}__{condition_name}__{method}"
        self.source_trajectory = Trajectory.from_csv(self.trajectory_path)
        self.trajectory: Trajectory | None = None
        self.schedule = generate_schedule(
            self.condition, self.source_trajectory.duration_s, DT_S, self.trial_seed
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.initial_px4_parameter_dump_hashes = {}
        for role in ("px4_1", "px4_2"):
            parameter_dump = self.output_dir / "simulator" / role / "parameters.bson"
            if parameter_dump.is_file():
                self.initial_px4_parameter_dump_hashes[role] = _sha256(parameter_dump)
        combined_parameters = hashlib.sha256()
        for role, digest in sorted(self.initial_px4_parameter_dump_hashes.items()):
            combined_parameters.update(role.encode())
            combined_parameters.update(digest.encode())
        self.initial_px4_parameter_dump_sha256 = combined_parameters.hexdigest()
        self.schedule_hash = save_schedule(self.schedule, self.output_dir / "telemetry_schedule.csv")

        self.controller = ExistingControllerAdapter(method, self.config_path, self.trial_seed)
        self.start_iso = datetime.now(timezone.utc).isoformat()
        self.start_ns = time.perf_counter_ns()
        self.state_enter_ns = self.start_ns
        self.last_tick_ns = self.start_ns
        self.last_state_rx_ns = {INTERCEPTOR.role: 0, TARGET.role: 0}
        self.run_start_ns: int | None = None
        self.state = "RESET"
        self.done = False
        self.success = False
        self.infrastructure_invalid = False
        self.reason = ""
        self.prestream_steps = 0
        self.last_arm_command_s = {INTERCEPTOR.role: -math.inf, TARGET.role: -math.inf}
        self.last_land_command_s = -math.inf
        self.landing_completed = False
        self.controller_exception = ""
        self.outcome_state = ""
        self.capture_time_s: float | None = None
        self.interpolated_capture_time_s: float | None = None
        self.capture_index: int | None = None
        self.relative_speed_at_capture_mps: float | None = None
        self.minimum_separation_m = math.inf
        self.previous_separation: tuple[float, float] | None = None
        self.initial_achieved: dict[str, Any] = {}
        self.transitions: list[dict[str, Any]] = []
        self.rows: list[dict[str, Any]] = []
        self.horizon_ca: list[np.ndarray] = []
        self.horizon_raw_narx: list[np.ndarray] = []
        self.horizon_used: list[np.ndarray] = []
        self.horizon_ca_velocity: list[np.ndarray] = []
        self.horizon_raw_narx_velocity: list[np.ndarray] = []
        self.horizon_used_velocity: list[np.ndarray] = []
        self.true_history: list[TargetStateSample] = []
        self.target_ground_enu: np.ndarray | None = None
        self.interceptor_ground_enu: np.ndarray | None = None
        self.target_takeoff_local_enu: np.ndarray | None = None
        self.interceptor_takeoff_local_enu: np.ndarray | None = None
        self.statuses: dict[str, VehicleStatus | None] = {
            INTERCEPTOR.role: None,
            TARGET.role: None,
        }
        self.positions: dict[str, VehicleLocalPosition | None] = {
            INTERCEPTOR.role: None,
            TARGET.role: None,
        }

        self.mode_pubs = {}
        self.setpoint_pubs = {}
        self.command_pubs = {}
        for identity in (INTERCEPTOR, TARGET):
            prefix = f"/{identity.namespace}/fmu"
            self.mode_pubs[identity.role] = self.create_publisher(
                OffboardControlMode, f"{prefix}/in/offboard_control_mode", PX4_QOS
            )
            self.setpoint_pubs[identity.role] = self.create_publisher(
                TrajectorySetpoint, f"{prefix}/in/trajectory_setpoint", PX4_QOS
            )
            self.command_pubs[identity.role] = self.create_publisher(
                VehicleCommand, f"{prefix}/in/vehicle_command", PX4_QOS
            )
            self.create_subscription(
                VehicleStatus,
                f"{prefix}/out/vehicle_status",
                lambda msg, role=identity.role: self._status(role, msg),
                PX4_QOS,
            )
            self.create_subscription(
                VehicleLocalPosition,
                f"{prefix}/out/vehicle_local_position",
                lambda msg, role=identity.role: self._position(role, msg),
                PX4_QOS,
            )
        self.timer = self.create_timer(DT_S, self._tick)
        self._transition("WAIT_FOR_PX4")

    def _elapsed(self) -> float:
        return (time.perf_counter_ns() - self.start_ns) * 1e-9

    def _state_elapsed(self) -> float:
        return (time.perf_counter_ns() - self.state_enter_ns) * 1e-9

    def _sim_time(self) -> float:
        if self.run_start_ns is None:
            return 0.0
        return (time.perf_counter_ns() - self.run_start_ns) * 1e-9

    def _status(self, role: str, msg: VehicleStatus) -> None:
        self.statuses[role] = msg
        self.last_state_rx_ns[role] = time.perf_counter_ns()

    def _position(self, role: str, msg: VehicleLocalPosition) -> None:
        self.positions[role] = msg
        self.last_state_rx_ns[role] = time.perf_counter_ns()

    def _transition(self, state: str) -> None:
        self.state = state
        self.state_enter_ns = time.perf_counter_ns()
        self.transitions.append({"wall_elapsed_s": self._elapsed(), "state": state})
        self.get_logger().info(f"trial transition: {state}")

    def _local_state(self, role: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        msg = self.positions[role]
        assert msg is not None
        pos = ned_position_to_enu([msg.x, msg.y, msg.z])
        vel = ned_velocity_to_enu([msg.vx, msg.vy, msg.vz])
        acc_ned = np.array([msg.ax, msg.ay, msg.az], dtype=float)
        acc = ned_acceleration_to_enu(np.nan_to_num(acc_ned, nan=0.0))
        return pos, vel, acc

    def _global_state(self, role: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos, vel, acc = self._local_state(role)
        return pos + SPAWN_ENU[role], vel, acc

    def _command(self, identity, command: int, param1: float = 0.0, param2: float = 0.0) -> None:
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds // 1000)
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = int(command)
        msg.target_system = identity.expected_system_id
        msg.target_component = 1
        msg.source_system = 255
        msg.source_component = 1
        msg.confirmation = 0
        msg.from_external = True
        self.command_pubs[identity.role].publish(msg)

    def _publish_position(
        self, identity, position_local_enu: np.ndarray, velocity_enu: np.ndarray | None = None,
        acceleration_enu: np.ndarray | None = None,
    ) -> None:
        timestamp = int(self.get_clock().now().nanoseconds // 1000)
        mode = OffboardControlMode()
        mode.timestamp = timestamp
        mode.position = True
        mode.velocity = False
        mode.acceleration = False
        self.mode_pubs[identity.role].publish(mode)
        point = TrajectorySetpoint()
        point.timestamp = timestamp
        point.position = enu_position_to_ned(position_local_enu).tolist()
        point.velocity = (
            [math.nan] * 3 if velocity_enu is None else enu_velocity_to_ned(velocity_enu).tolist()
        )
        point.acceleration = (
            [math.nan] * 3
            if acceleration_enu is None
            else enu_acceleration_to_ned(acceleration_enu).tolist()
        )
        point.jerk = [math.nan] * 3
        point.yaw = math.nan
        point.yawspeed = math.nan
        self.setpoint_pubs[identity.role].publish(point)

    def _publish_acceleration(self, acceleration_enu: np.ndarray) -> np.ndarray:
        timestamp = int(self.get_clock().now().nanoseconds // 1000)
        mode = OffboardControlMode()
        mode.timestamp = timestamp
        mode.position = False
        mode.velocity = False
        mode.acceleration = True
        mode.attitude = False
        mode.body_rate = False
        mode.thrust_and_torque = False
        mode.direct_actuator = False
        self.mode_pubs[INTERCEPTOR.role].publish(mode)
        acceleration_ned = enu_acceleration_to_ned(acceleration_enu)
        point = TrajectorySetpoint()
        point.timestamp = timestamp
        point.position = [math.nan] * 3
        point.velocity = [math.nan] * 3
        point.acceleration = acceleration_ned.tolist()
        point.jerk = [math.nan] * 3
        point.yaw = math.nan
        point.yawspeed = math.nan
        self.setpoint_pubs[INTERCEPTOR.role].publish(point)
        return acceleration_ned

    def _ready(self) -> bool:
        for identity in (INTERCEPTOR, TARGET):
            status = self.statuses[identity.role]
            position = self.positions[identity.role]
            if status is None or position is None:
                return False
            if int(status.system_id) != identity.expected_system_id:
                self._finish(True, f"{identity.role} system ID {status.system_id} != {identity.expected_system_id}")
                return False
            if not status.pre_flight_checks_pass or not position.xy_valid or not position.z_valid:
                return False
        return True

    def _publish_takeoff(self) -> None:
        assert self.interceptor_takeoff_local_enu is not None
        assert self.target_takeoff_local_enu is not None
        self._publish_position(INTERCEPTOR, self.interceptor_takeoff_local_enu)
        self._publish_position(TARGET, self.target_takeoff_local_enu)

    @staticmethod
    def _flatten(prefix: str, values: np.ndarray, suffix: str) -> dict[str, float]:
        labels = ("e", "n", "u")
        return {f"{prefix}_{axis}_{suffix}": float(values[i]) for i, axis in enumerate(labels)}

    def _append_horizon(self, storage: list[np.ndarray], info: dict[str, Any], key: str) -> None:
        horizon = np.asarray(info.get(key, np.full((20, 3), np.nan)), dtype=float)
        if horizon.shape != (20, 3):
            fixed = np.full((20, 3), np.nan)
            if horizon.ndim == 2:
                fixed[: min(20, len(horizon)), : min(3, horizon.shape[1])] = horizon[:20, :3]
            horizon = fixed
        storage.append(horizon)

    def _run_step(self, tick_start_ns: int) -> None:
        assert self.trajectory is not None
        sim_time_s = self._sim_time()
        schedule_index = min(int(round(sim_time_s / DT_S)), len(self.schedule) - 1)
        desired = self.trajectory.sample(sim_time_s)
        self._publish_position(TARGET, desired.position, desired.velocity, desired.acceleration)

        target_pos, target_vel, target_acc = self._global_state(TARGET.role)
        interceptor_pos, interceptor_vel, interceptor_acc = self._global_state(INTERCEPTOR.role)
        schedule_row = self.schedule.iloc[schedule_index]
        # The schedule timestamp is the campaign's monotonic control timestamp:
        # it is deterministic across paired arms and names the executed state
        # sampled for this control event. Repeated callbacks at one schedule
        # tick replace that tick's latest state instead of creating ambiguous
        # duplicate timestamps.
        arrival_timestamp_s = float(schedule_row["timestamp"])
        target_sample = TargetStateSample(arrival_timestamp_s, target_pos, target_vel)
        if self.true_history and arrival_timestamp_s == self.true_history[-1].timestamp_s:
            self.true_history[-1] = target_sample
        elif self.true_history and arrival_timestamp_s < self.true_history[-1].timestamp_s:
            raise RuntimeError("non-monotonic telemetry control timestamp")
        else:
            self.true_history.append(target_sample)
        telemetry = apply_schedule_row(self.true_history, schedule_row, arrival_timestamp_s)

        controller_start_ns = time.perf_counter_ns()
        pursuer_state = np.concatenate([interceptor_pos, interceptor_vel, interceptor_acc])
        command_raw, info = self.controller.step(pursuer_state, telemetry, DT_S, sim_time_s)
        raw_norm = float(np.linalg.norm(command_raw))
        command = np.clip(command_raw, -20.0, 20.0)
        command_norm = float(np.linalg.norm(command))
        if command_norm > 20.0:
            command *= 20.0 / command_norm
        adapter_clipped = not np.allclose(command, command_raw, rtol=0.0, atol=1e-12)
        acceleration_ned = self._publish_acceleration(command)
        complete_fast_loop_s = (time.perf_counter_ns() - tick_start_ns) * 1e-9

        separation = float(np.linalg.norm(target_pos - interceptor_pos))
        relative_speed = float(np.linalg.norm(target_vel - interceptor_vel))
        captured = separation <= CAPTURE_RADIUS_M
        self.minimum_separation_m = min(self.minimum_separation_m, separation)
        if captured and self.capture_time_s is None:
            self.capture_time_s = sim_time_s
            self.capture_index = len(self.rows)
            self.relative_speed_at_capture_mps = relative_speed
            if self.previous_separation is not None:
                self.interpolated_capture_time_s = interpolated_crossing_time(
                    self.previous_separation[0], self.previous_separation[1], sim_time_s, separation
                )
            if self.interpolated_capture_time_s is None:
                self.interpolated_capture_time_s = sim_time_s
        self.previous_separation = (sim_time_s, separation)

        estimate = np.asarray(info.get("target_estimate", np.full(12, np.nan)), dtype=float)
        desired_global = desired.position + SPAWN_ENU[TARGET.role]
        telemetry_age = telemetry.configured_delay_s if telemetry.valid else math.nan
        row: dict[str, Any] = {
            "trial_id": self.trial_id,
            "condition": self.condition.name,
            "trajectory_id": self.trajectory_id,
            "family": self.family,
            "method": self.method,
            "trial_seed": self.trial_seed,
            "sim_time_s": sim_time_s,
            "wall_time_s": self._elapsed(),
            "telemetry_age_s": telemetry_age,
            "drop_flag": int(telemetry.drop),
            "configured_delay_s": telemetry.configured_delay_s,
            "arrival_timestamp_s": telemetry.arrival_timestamp_s,
            "requested_source_timestamp_s": telemetry.requested_source_timestamp_s,
            "actual_measurement_source_timestamp_s": telemetry.actual_source_timestamp_s,
            "measurement_history_left_timestamp_s": telemetry.history_left_timestamp_s,
            "measurement_history_right_timestamp_s": telemetry.history_right_timestamp_s,
            "measurement_interpolation_alpha": telemetry.interpolation_alpha,
            "startup_clamped": int(telemetry.startup_clamped),
            "physical_measurement_age_s": telemetry.physical_measurement_age_s,
            "separation_m": separation,
            "relative_speed_mps": relative_speed,
            "capture_flag": int(captured),
            "interceptor_nav_state": int(self.statuses[INTERCEPTOR.role].nav_state),
            "target_nav_state": int(self.statuses[TARGET.role].nav_state),
            "interceptor_arming_state": int(self.statuses[INTERCEPTOR.role].arming_state),
            "target_arming_state": int(self.statuses[TARGET.role].arming_state),
            "interceptor_offboard": int(
                self.statuses[INTERCEPTOR.role].nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
            ),
            "target_offboard": int(
                self.statuses[TARGET.role].nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
            ),
            "acados_status": str(info.get("solver_status", "not_applicable")),
            "acados_success": int(bool(info.get("solver_success", True))),
            "acados_solve_time_s": _float(info.get("solve_time_s")),
            "estimator_time_s": _float(info.get("estimator_time_s")),
            "narx_inference_time_s": _float(info.get("narx_infer_time_s")),
            "gate_time_s": math.nan,
            "controller_total_time_s": _float(info.get("controller_total_time_s")),
            "fast_loop_wall_time_s": complete_fast_loop_s,
            "narx_training_event_time_s": _float(info.get("narx_train_time_s"), 0.0),
            "narx_training_executed": int(bool(info.get("narx_training_executed", False))),
            "narx_training_failure": int(bool(info.get("narx_training_skipped_deadline", False))),
            "trust": _float(info.get("narx_trust"), 0.0),
            "prequential_error": _float(info.get("narx_prequential_loss")),
            "raw_command_norm_mps2": raw_norm,
            "adapter_clipped": int(adapter_clipped),
        }
        row.update(self._flatten("target_desired_pos", desired_global, "m"))
        row.update(self._flatten("target_desired_vel", desired.velocity, "mps"))
        row.update(self._flatten("target_desired_acc", desired.acceleration, "mps2"))
        row.update(self._flatten("target_desired_jerk", desired.jerk, "mps3"))
        row.update(self._flatten("target_actual_pos", target_pos, "m"))
        row.update(self._flatten("target_actual_vel", target_vel, "mps"))
        telemetry_log = np.full(6, np.nan) if telemetry.measurement is None else telemetry.measurement
        row.update(self._flatten("telemetry_pos", telemetry_log[:3], "m"))
        row.update(self._flatten("telemetry_vel", telemetry_log[3:6], "mps"))
        row.update(
            self._flatten(
                "measurement_truth_position", telemetry.source_truth_position_enu, "m"
            )
        )
        row.update(
            self._flatten(
                "measurement_truth_velocity", telemetry.source_truth_velocity_enu, "mps"
            )
        )
        row.update(self._flatten("ekf_pos", estimate[:3], "m"))
        row.update(self._flatten("ekf_vel", estimate[3:6], "mps"))
        row.update(self._flatten("ekf_acc", estimate[6:9], "mps2"))
        row.update(self._flatten("interceptor_actual_pos", interceptor_pos, "m"))
        row.update(self._flatten("interceptor_actual_vel", interceptor_vel, "mps"))
        row.update(self._flatten("interceptor_actual_acc", interceptor_acc, "mps2"))
        row.update(self._flatten("acceleration_command_enu", command, "mps2"))
        row.update({
            "acceleration_command_ned_n_mps2": float(acceleration_ned[0]),
            "acceleration_command_ned_e_mps2": float(acceleration_ned[1]),
            "acceleration_command_ned_d_mps2": float(acceleration_ned[2]),
        })
        self.rows.append(row)
        self._append_horizon(self.horizon_ca, info, "ca_predicted_position_horizon")
        self._append_horizon(self.horizon_raw_narx, info, "raw_narx_position_horizon")
        self._append_horizon(self.horizon_used, info, "used_position_horizon")
        self._append_horizon(self.horizon_ca_velocity, info, "ca_predicted_velocity_horizon")
        self._append_horizon(self.horizon_raw_narx_velocity, info, "raw_narx_velocity_horizon")
        self._append_horizon(self.horizon_used_velocity, info, "used_velocity_horizon")

        if captured:
            self.outcome_state = "CAPTURE"
            self._transition("HOLD")
        elif sim_time_s >= self.source_trajectory.duration_s:
            self.outcome_state = "TIMEOUT"
            self._transition("HOLD")

    def _write_outputs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        steps_path = self.output_dir / "steps.csv"
        if self.rows:
            with steps_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
        np.savez_compressed(
            self.output_dir / "horizons.npz",
            ca=np.asarray(self.horizon_ca, dtype=float),
            raw_narx=np.asarray(self.horizon_raw_narx, dtype=float),
            used=np.asarray(self.horizon_used, dtype=float),
            ca_velocity=np.asarray(self.horizon_ca_velocity, dtype=float),
            raw_narx_velocity=np.asarray(self.horizon_raw_narx_velocity, dtype=float),
            used_velocity=np.asarray(self.horizon_used_velocity, dtype=float),
        )
        target_position_errors = []
        target_velocity_errors = []
        for row in self.rows:
            desired_p = np.array([row[f"target_desired_pos_{a}_m"] for a in "enu"])
            actual_p = np.array([row[f"target_actual_pos_{a}_m"] for a in "enu"])
            desired_v = np.array([row[f"target_desired_vel_{a}_mps"] for a in "enu"])
            actual_v = np.array([row[f"target_actual_vel_{a}_mps"] for a in "enu"])
            target_position_errors.append(np.linalg.norm(desired_p - actual_p))
            target_velocity_errors.append(np.linalg.norm(desired_v - actual_v))
        p_err = np.asarray(target_position_errors)
        v_err = np.asarray(target_velocity_errors)
        summary = {
            "trial_id": self.trial_id,
            "condition": self.condition.name,
            "world": self.condition.world,
            "trajectory_id": self.trajectory_id,
            "family": self.family,
            "method": self.method,
            "planned_valid_trial": True,
            "infrastructure_invalid": self.infrastructure_invalid,
            "exit_status": "infrastructure_invalid" if self.infrastructure_invalid else "complete",
            "reason": self.reason,
            "outcome": self.outcome_state,
            "capture": self.capture_time_s is not None,
            "capture_time_s": self.capture_time_s,
            "interpolated_capture_time_s": self.interpolated_capture_time_s,
            "capture_index": self.capture_index,
            "minimum_separation_m": None if math.isinf(self.minimum_separation_m) else self.minimum_separation_m,
            "relative_speed_at_capture_mps": self.relative_speed_at_capture_mps,
            "control_steps": len(self.rows),
            "target_position_rmse_m": None if not len(p_err) else float(np.sqrt(np.mean(p_err**2))),
            "target_velocity_rmse_mps": None if not len(v_err) else float(np.sqrt(np.mean(v_err**2))),
            "target_maximum_tracking_error_m": None if not len(p_err) else float(np.max(p_err)),
            "landing_completed": self.landing_completed,
            "controller_exception": self.controller_exception,
            "telemetry_schedule_sha256": self.schedule_hash,
            "initial_achieved_states": self.initial_achieved,
            "transitions": self.transitions,
            "steps_file": str(steps_path),
            "horizons_file": str(self.output_dir / "horizons.npz"),
        }
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

        px4_param_files = sorted(
            Path(os.environ.get("PX4_DIR", "/nonexistent")).glob("build/px4_sitl_default/etc/init.d-posix/airframes/*")
        )
        param_digest = hashlib.sha256()
        for path in px4_param_files:
            if path.is_file():
                param_digest.update(path.name.encode())
                param_digest.update(path.read_bytes())
        metadata = {
            "px4_commit": "4817c0618a1286846116e90c6eb8919efaa013cf",
            "existing_repo_commit": "unavailable_not_a_git_worktree",
            "workspace_commit": "unavailable_not_a_git_worktree",
            "ros_version": "ROS 2 Humble",
            "gazebo_version": "Gazebo Sim 8.12.0",
            "acados_version": "acados_template 0.5.1",
            "method_config_sha256": _sha256(self.config_path),
            "trajectory_sha256": _sha256(self.trajectory_path),
            "telemetry_schedule_sha256": self.schedule_hash,
            "px4_parameter_source_hash": param_digest.hexdigest(),
            "initial_px4_parameter_dump_sha256": self.initial_px4_parameter_dump_sha256,
            "initial_px4_parameter_dump_hashes": self.initial_px4_parameter_dump_hashes,
            "command_line": " ".join(sys.argv),
            "start_time_utc": self.start_iso,
            "end_time_utc": datetime.now(timezone.utc).isoformat(),
            "exit_status": summary["exit_status"],
        }
        (self.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def _finish(self, infrastructure_invalid: bool, reason: str) -> None:
        if self.done:
            return
        self.infrastructure_invalid = bool(infrastructure_invalid)
        self.reason = reason
        if infrastructure_invalid:
            self.outcome_state = "INFRASTRUCTURE_FAILURE"
        self.success = not infrastructure_invalid
        self.done = True
        self._write_outputs()
        self.get_logger().info(f"trial result: {self.outcome_state}: {reason}")

    def _tick(self) -> None:
        if self.done:
            return
        tick_start_ns = time.perf_counter_ns()
        if self._elapsed() > self.timeout_s:
            if self.rows and self.state in {"HOLD", "LAND", "DISARM", "SAVE"}:
                self._finish(False, f"valid outcome saved; cleanup timeout in {self.state}")
            else:
                self._finish(True, f"process timeout in {self.state}")
            return
        if self.state not in {"WAIT_FOR_PX4", "DONE", "SAVE"}:
            for role, last_ns in self.last_state_rx_ns.items():
                if last_ns and (tick_start_ns - last_ns) * 1e-9 > 2.0:
                    if self.rows and self.state in {"HOLD", "LAND", "DISARM"}:
                        # Gazebo / ODE can abort when an experimentally unstable
                        # vehicle contacts the ground.  Once RUN has ended and a
                        # complete result log exists, that is a bounded-cleanup
                        # failure rather than a reason to discard the outcome.
                        self._finish(
                            False,
                            f"valid outcome saved; {role} PX4 state topics disappeared during {self.state}",
                        )
                    else:
                        self._finish(True, f"required {role} PX4 state topics disappeared")
                    return

        if self.state == "WAIT_FOR_PX4":
            if self._ready():
                interceptor_pos, _, _ = self._local_state(INTERCEPTOR.role)
                target_pos, _, _ = self._local_state(TARGET.role)
                self.interceptor_ground_enu = interceptor_pos.copy()
                self.target_ground_enu = target_pos.copy()
                self.interceptor_takeoff_local_enu = np.array([0.0, 0.0, interceptor_pos[2] + 10.0])
                self.target_takeoff_local_enu = np.array([0.0, 0.0, target_pos[2] + 10.0])
                source_start = self.source_trajectory.sample(0.0).position
                self.trajectory = self.source_trajectory.adapted(
                    self.target_takeoff_local_enu - source_start
                )
                self._transition("ARM_TARGET")
            return

        if self.state in {"ARM_TARGET", "ARM_INTERCEPTOR", "TAKEOFF_BOTH", "STABILIZE"}:
            self._publish_takeoff()

        if self.state == "ARM_TARGET":
            self.prestream_steps += 1
            target_status = self.statuses[TARGET.role]
            if self.prestream_steps >= 100 and target_status.arming_state == VehicleStatus.ARMING_STATE_ARMED:
                self._transition("ARM_INTERCEPTOR")
            elif self.prestream_steps >= 100 and self._elapsed() - self.last_arm_command_s[TARGET.role] >= 1.0:
                self._command(TARGET, VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self._command(TARGET, VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.last_arm_command_s[TARGET.role] = self._elapsed()
        elif self.state == "ARM_INTERCEPTOR":
            interceptor_status = self.statuses[INTERCEPTOR.role]
            if interceptor_status.arming_state == VehicleStatus.ARMING_STATE_ARMED:
                self._transition("TAKEOFF_BOTH")
            elif self._elapsed() - self.last_arm_command_s[INTERCEPTOR.role] >= 1.0:
                self._command(INTERCEPTOR, VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self._command(INTERCEPTOR, VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.last_arm_command_s[INTERCEPTOR.role] = self._elapsed()
        elif self.state == "TAKEOFF_BOTH":
            interceptor, _, _ = self._local_state(INTERCEPTOR.role)
            target, _, _ = self._local_state(TARGET.role)
            statuses = [self.statuses[role] for role in (INTERCEPTOR.role, TARGET.role)]
            if (
                all(s.arming_state == VehicleStatus.ARMING_STATE_ARMED for s in statuses)
                and np.linalg.norm(interceptor - self.interceptor_takeoff_local_enu) <= 0.75
                and np.linalg.norm(target - self.target_takeoff_local_enu) <= 0.75
            ):
                self._transition("STABILIZE")
        elif self.state == "STABILIZE" and self._state_elapsed() >= 2.0:
            self.initial_achieved = {
                "interceptor_global_enu": self._global_state(INTERCEPTOR.role)[0].tolist(),
                "target_global_enu": self._global_state(TARGET.role)[0].tolist(),
                "interceptor_velocity_enu": self._global_state(INTERCEPTOR.role)[1].tolist(),
                "target_velocity_enu": self._global_state(TARGET.role)[1].tolist(),
            }
            self._transition("START_TARGET_TRAJECTORY")
        elif self.state == "START_TARGET_TRAJECTORY":
            self._transition("START_TELEMETRY")
        elif self.state == "START_TELEMETRY":
            self._transition("START_CONTROLLER")
        elif self.state == "START_CONTROLLER":
            self.run_start_ns = time.perf_counter_ns()
            self._transition("RUN")
        elif self.state == "RUN":
            try:
                self._run_step(tick_start_ns)
            except Exception as exc:
                self.get_logger().error(f"controller/run exception: {type(exc).__name__}: {exc}")
                # Controller exceptions are experimental outcomes, except when no
                # valid log can be produced.  Preserve the partial trial log.
                self.controller_exception = f"{type(exc).__name__}: {exc}"
                self.outcome_state = "CONTROLLER_EXCEPTION"
                self._transition("HOLD")
        elif self.state == "HOLD":
            interceptor, _, _ = self._local_state(INTERCEPTOR.role)
            target, _, _ = self._local_state(TARGET.role)
            self._publish_position(INTERCEPTOR, interceptor)
            self._publish_position(TARGET, target)
            if self._state_elapsed() >= 1.0:
                self._command(TARGET, VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self._command(INTERCEPTOR, VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.last_land_command_s = self._elapsed()
                self._transition("LAND")
        elif self.state == "LAND":
            statuses = [self.statuses[role] for role in (INTERCEPTOR.role, TARGET.role)]
            if all(s.arming_state == VehicleStatus.ARMING_STATE_DISARMED for s in statuses):
                self.landing_completed = True
                self._transition("DISARM")
            elif self._state_elapsed() >= 35.0:
                # A controller can leave the vehicle far above the world after
                # a failed interception.  That is an experimental outcome, not
                # broken simulator infrastructure.  Bound cleanup time and let
                # exclusive process supervision terminate this disposable SITL.
                self._command(TARGET, VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
                self._command(INTERCEPTOR, VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
                self._transition("DISARM")
            elif self._elapsed() - self.last_land_command_s >= 2.0:
                self._command(TARGET, VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self._command(INTERCEPTOR, VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.last_land_command_s = self._elapsed()
        elif self.state == "DISARM":
            self._transition("SAVE")
        elif self.state == "SAVE":
            self._transition("DONE")
            self._finish(False, f"valid experimental outcome: {self.outcome_state}")


def main(argv: list[str] | None = None) -> int:
    source_root = Path(os.environ.get("DRONE_INTERCEPTION_V3", "."))
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--trajectory-id", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--trial-seed", type=int, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=source_root / "configs/q2_revision_pilot.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    args, ros_args = parser.parse_known_args(argv)
    rclpy.init(args=ros_args)
    node: ExperimentSupervisor | None = None
    try:
        node = ExperimentSupervisor(
            args.trajectory,
            args.trajectory_id,
            args.family,
            args.condition,
            args.method,
            args.trial_seed,
            args.config,
            args.output_dir,
            args.timeout_s,
        )
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        if node is not None:
            node._finish(True, "interrupted")
    finally:
        success = bool(node is not None and node.success)
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
