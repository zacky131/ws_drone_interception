"""Standalone target-X500 trajectory follower and tracking-fidelity recorder."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.node import Node
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand
from px4_msgs.msg import VehicleLocalPosition, VehicleStatus

from .frames import enu_acceleration_to_ned, enu_position_to_ned, enu_velocity_to_ned
from .frames import ned_position_to_enu, ned_velocity_to_enu
from .single_vehicle_smoke import PX4_QOS
from .trajectory import Trajectory


class TargetTrajectoryPlayer(Node):
    NAMESPACE = "px4_2"
    SYSTEM_ID = 3

    def __init__(self, source: Path, output_dir: Path, timeout_s: float) -> None:
        super().__init__("target_trajectory_player")
        self.source = source.resolve()
        self.output_dir = output_dir.resolve()
        self.timeout_s = timeout_s
        self.source_trajectory = Trajectory.from_csv(self.source)
        self.trajectory: Trajectory | None = None
        self.translation_enu: np.ndarray | None = None

        self.start_ns = time.perf_counter_ns()
        self.state_enter_ns = self.start_ns
        self.trajectory_start_ns: int | None = None
        self.state = "WAIT_READY"
        self.done = False
        self.success = False
        self.prestream_count = 0
        self.status: VehicleStatus | None = None
        self.position: VehicleLocalPosition | None = None
        self.takeoff_position_enu: np.ndarray | None = None
        self.last_land_command_s = -math.inf
        self.transitions: list[dict[str, object]] = []
        self.rows: list[dict[str, object]] = []
        self.heartbeat_ns: list[int] = []

        prefix = f"/{self.NAMESPACE}/fmu"
        self.mode_pub = self.create_publisher(
            OffboardControlMode, f"{prefix}/in/offboard_control_mode", PX4_QOS
        )
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, f"{prefix}/in/trajectory_setpoint", PX4_QOS
        )
        self.command_pub = self.create_publisher(
            VehicleCommand, f"{prefix}/in/vehicle_command", PX4_QOS
        )
        self.create_subscription(
            VehicleStatus, f"{prefix}/out/vehicle_status", self._status, PX4_QOS
        )
        self.create_subscription(
            VehicleLocalPosition,
            f"{prefix}/out/vehicle_local_position",
            self._position,
            PX4_QOS,
        )
        self.timer = self.create_timer(0.02, self._tick)

    def _elapsed(self) -> float:
        return (time.perf_counter_ns() - self.start_ns) / 1e9

    def _state_elapsed(self) -> float:
        return (time.perf_counter_ns() - self.state_enter_ns) / 1e9

    def _status(self, msg: VehicleStatus) -> None:
        self.status = msg

    def _position(self, msg: VehicleLocalPosition) -> None:
        self.position = msg

    def _transition(self, state: str) -> None:
        self.state = state
        self.state_enter_ns = time.perf_counter_ns()
        self.transitions.append(
            {
                "elapsed_s": self._elapsed(),
                "state": state,
                "actual_local_position_enu_m": None
                if self.position is None
                else ned_position_to_enu(
                    [self.position.x, self.position.y, self.position.z]
                ).tolist(),
                "arming_state": None
                if self.status is None
                else int(self.status.arming_state),
                "nav_state": None if self.status is None else int(self.status.nav_state),
            }
        )
        self.get_logger().info(f"target transition: {state}")

    def _command(self, command: int, param1: float = 0.0, param2: float = 0.0) -> None:
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds // 1000)
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = int(command)
        msg.target_system = self.SYSTEM_ID
        msg.target_component = 1
        msg.source_system = 255
        msg.source_component = 1
        msg.confirmation = 0
        msg.from_external = True
        self.command_pub.publish(msg)

    def _publish(self, position_enu: np.ndarray, velocity_enu: np.ndarray, acceleration_enu: np.ndarray) -> None:
        timestamp_us = int(self.get_clock().now().nanoseconds // 1000)
        mode = OffboardControlMode()
        mode.timestamp = timestamp_us
        mode.position = True
        mode.velocity = False
        mode.acceleration = False
        mode.attitude = False
        mode.body_rate = False
        mode.thrust_and_torque = False
        mode.direct_actuator = False
        self.mode_pub.publish(mode)

        point = TrajectorySetpoint()
        point.timestamp = timestamp_us
        point.position = enu_position_to_ned(position_enu).astype(float).tolist()
        point.velocity = enu_velocity_to_ned(velocity_enu).astype(float).tolist()
        point.acceleration = enu_acceleration_to_ned(acceleration_enu).astype(float).tolist()
        point.jerk = [math.nan, math.nan, math.nan]
        point.yaw = math.nan
        point.yawspeed = math.nan
        self.setpoint_pub.publish(point)
        self.heartbeat_ns.append(time.perf_counter_ns())

    def _actual_state_enu(self) -> tuple[np.ndarray, np.ndarray]:
        assert self.position is not None
        position = ned_position_to_enu([self.position.x, self.position.y, self.position.z])
        velocity = ned_velocity_to_enu([self.position.vx, self.position.vy, self.position.vz])
        return position, velocity

    def _record(self, trajectory_time_s: float, desired) -> None:
        actual_position, actual_velocity = self._actual_state_enu()
        desired_world = desired.position + np.array([30.0, 0.0, 0.0])
        actual_world = actual_position + np.array([30.0, 0.0, 0.0])
        self.rows.append(
            {
                "wall_elapsed_s": self._elapsed(),
                "trajectory_time_s": trajectory_time_s,
                "desired_world_e_m": desired_world[0],
                "desired_world_n_m": desired_world[1],
                "desired_world_u_m": desired_world[2],
                "desired_vel_e_mps": desired.velocity[0],
                "desired_vel_n_mps": desired.velocity[1],
                "desired_vel_u_mps": desired.velocity[2],
                "actual_world_e_m": actual_world[0],
                "actual_world_n_m": actual_world[1],
                "actual_world_u_m": actual_world[2],
                "actual_vel_e_mps": actual_velocity[0],
                "actual_vel_n_mps": actual_velocity[1],
                "actual_vel_u_mps": actual_velocity[2],
                "position_error_m": float(np.linalg.norm(desired.position - actual_position)),
                "velocity_error_mps": float(np.linalg.norm(desired.velocity - actual_velocity)),
                "nav_state": int(self.status.nav_state) if self.status else -1,
                "arming_state": int(self.status.arming_state) if self.status else -1,
                "offboard": bool(self.status and self.status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD),
            }
        )

    def _finish(self, success: bool, reason: str) -> None:
        if self.done:
            return
        self.done = True
        self.success = success
        self.output_dir.mkdir(parents=True, exist_ok=True)
        samples_path = self.output_dir / "target_tracking.csv"
        if self.rows:
            with samples_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
                writer.writeheader()
                writer.writerows(self.rows)
        position_errors = np.array([row["position_error_m"] for row in self.rows], dtype=float)
        velocity_errors = np.array([row["velocity_error_mps"] for row in self.rows], dtype=float)
        intervals = [(b - a) / 1e9 for a, b in zip(self.heartbeat_ns, self.heartbeat_ns[1:])]
        summary = {
            "success": success,
            "reason": reason,
            "namespace": self.NAMESPACE,
            "expected_system_id": self.SYSTEM_ID,
            "reported_system_id": None if self.status is None else int(self.status.system_id),
            "source_file": str(self.source),
            "source_duration_s": self.source_trajectory.duration_s,
            "source_time_scaled": False,
            "spawn_world_enu_m": [30.0, 0.0, 0.0],
            "translation_enu_m": None if self.translation_enu is None else self.translation_enu.tolist(),
            "position_rmse_m": None if not len(position_errors) else float(np.sqrt(np.mean(position_errors**2))),
            "velocity_rmse_mps": None if not len(velocity_errors) else float(np.sqrt(np.mean(velocity_errors**2))),
            "maximum_tracking_error_m": None if not len(position_errors) else float(position_errors.max()),
            "tracking_samples": len(self.rows),
            "heartbeat_worst_rate_hz": None if not intervals else 1.0 / max(intervals),
            "transitions": self.transitions,
            "samples_file": str(samples_path),
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        self.get_logger().info(f"target result: {success}: {reason}")

    def _tick(self) -> None:
        if self.done:
            return
        if self._elapsed() > self.timeout_s:
            self._finish(False, f"timeout in {self.state}")
            return

        if self.state == "WAIT_READY":
            if (
                self.status is not None
                and self.position is not None
                and self.status.pre_flight_checks_pass
                and self.position.xy_valid
                and self.position.z_valid
            ):
                if int(self.status.system_id) != self.SYSTEM_ID:
                    self._finish(False, f"reported system ID {self.status.system_id}")
                    return
                actual_position, _ = self._actual_state_enu()
                self.takeoff_position_enu = np.array(
                    [0.0, 0.0, actual_position[2] + 10.0], dtype=float
                )
                source_start = self.source_trajectory.sample(0.0).position
                self.translation_enu = self.takeoff_position_enu - source_start
                self.trajectory = self.source_trajectory.adapted(self.translation_enu)
                self._transition("PRESTREAM")
            return

        assert self.takeoff_position_enu is not None
        assert self.trajectory is not None

        if self.state in {"PRESTREAM", "TAKEOFF", "STABILIZE"}:
            self._publish(self.takeoff_position_enu, np.zeros(3), np.zeros(3))
        elif self.state in {"TRACK", "FINAL_HOLD"}:
            assert self.trajectory_start_ns is not None
            trajectory_time = min(
                (time.perf_counter_ns() - self.trajectory_start_ns) / 1e9,
                self.trajectory.duration_s,
            )
            desired = self.trajectory.sample(trajectory_time)
            self._publish(desired.position, desired.velocity, desired.acceleration)
            self._record(trajectory_time, desired)

        if self.state == "PRESTREAM":
            self.prestream_count += 1
            if self.prestream_count >= 100:
                self._command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self._command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self._transition("TAKEOFF")
        elif self.state == "TAKEOFF":
            actual, _ = self._actual_state_enu()
            if (
                self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED
                and np.linalg.norm(actual - self.takeoff_position_enu) <= 0.75
            ):
                self._transition("STABILIZE")
        elif self.state == "STABILIZE" and self._state_elapsed() >= 2.0:
            self.trajectory_start_ns = time.perf_counter_ns()
            self._transition("TRACK")
        elif self.state == "TRACK" and self.trajectory_start_ns is not None:
            if (time.perf_counter_ns() - self.trajectory_start_ns) / 1e9 >= self.trajectory.duration_s:
                self._transition("FINAL_HOLD")
        elif self.state == "FINAL_HOLD" and self._state_elapsed() >= 1.0:
            self._command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.last_land_command_s = self._elapsed()
            self._transition("LAND")
        elif self.state == "LAND":
            if self.status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                self._transition("DONE")
                self._finish(True, "target trajectory completed, landed, and disarmed")
            elif self._elapsed() - self.last_land_command_s >= 1.0:
                self._command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.last_land_command_s = self._elapsed()


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=root / "results/ral_gazebo_v1/smoke/s1_target"
    )
    parser.add_argument("--timeout-s", type=float, default=80.0)
    args, ros_args = parser.parse_known_args(argv)
    rclpy.init(args=ros_args)
    node = TargetTrajectoryPlayer(args.trajectory, args.output_dir, args.timeout_s)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node._finish(False, "interrupted")
    finally:
        success = node.success
        node.destroy_node()
        rclpy.shutdown()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
