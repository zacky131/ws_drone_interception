"""Standalone interceptor acceleration-Offboard hold acceptance runner."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import rclpy
from rclpy.node import Node
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand
from px4_msgs.msg import VehicleLocalPosition, VehicleStatus

from .frames import enu_acceleration_to_ned, ned_position_to_enu, ned_velocity_to_enu
from .single_vehicle_smoke import PX4_QOS


class AccelerationHoldSmoke(Node):
    def __init__(self, output: Path, timeout_s: float) -> None:
        super().__init__("interceptor_acceleration_hold")
        self.output = output.resolve()
        self.timeout_s = timeout_s
        self.start_ns = time.perf_counter_ns()
        self.state_enter_ns = self.start_ns
        self.state = "WAIT_READY"
        self.done = False
        self.success = False
        self.status: VehicleStatus | None = None
        self.position: VehicleLocalPosition | None = None
        self.takeoff_z_enu: float | None = None
        self.prestream_count = 0
        self.heartbeat_ns: list[int] = []
        self.hold_samples: list[dict[str, float]] = []
        self.transitions: list[dict[str, object]] = []
        self.last_land_command_s = -math.inf

        prefix = "/px4_1/fmu"
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
                "arming_state": None if self.status is None else int(self.status.arming_state),
                "nav_state": None if self.status is None else int(self.status.nav_state),
            }
        )
        self.get_logger().info(f"interceptor transition: {state}")

    def _command(self, command: int, param1: float = 0.0, param2: float = 0.0) -> None:
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds // 1000)
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = int(command)
        msg.target_system = 2
        msg.target_component = 1
        msg.source_system = 255
        msg.source_component = 1
        msg.confirmation = 0
        msg.from_external = True
        self.command_pub.publish(msg)

    def _publish_position(self) -> None:
        assert self.takeoff_z_enu is not None
        timestamp = int(self.get_clock().now().nanoseconds // 1000)
        mode = OffboardControlMode()
        mode.timestamp = timestamp
        mode.position = True
        self.mode_pub.publish(mode)
        point = TrajectorySetpoint()
        point.timestamp = timestamp
        point.position = [0.0, 0.0, -self.takeoff_z_enu]
        point.velocity = [math.nan] * 3
        point.acceleration = [math.nan] * 3
        point.jerk = [math.nan] * 3
        point.yaw = 0.0
        point.yawspeed = math.nan
        self.setpoint_pub.publish(point)
        self.heartbeat_ns.append(time.perf_counter_ns())

    def _publish_acceleration_hold(self) -> None:
        assert self.position is not None and self.takeoff_z_enu is not None
        position = ned_position_to_enu([self.position.x, self.position.y, self.position.z])
        velocity = ned_velocity_to_enu([self.position.vx, self.position.vy, self.position.vz])
        vertical_acceleration = float(
            np.clip(1.5 * (self.takeoff_z_enu - position[2]) - 1.2 * velocity[2], -3.0, 3.0)
        )
        acceleration_enu = np.array([0.0, 0.0, vertical_acceleration])

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
        self.mode_pub.publish(mode)
        point = TrajectorySetpoint()
        point.timestamp = timestamp
        point.position = [math.nan] * 3
        point.velocity = [math.nan] * 3
        point.acceleration = enu_acceleration_to_ned(acceleration_enu).tolist()
        point.jerk = [math.nan] * 3
        point.yaw = math.nan
        point.yawspeed = math.nan
        self.setpoint_pub.publish(point)
        self.heartbeat_ns.append(time.perf_counter_ns())
        self.hold_samples.append(
            {
                "elapsed_s": self._elapsed(),
                "altitude_enu_m": float(position[2]),
                "vertical_velocity_enu_mps": float(velocity[2]),
                "command_e_mps2": 0.0,
                "command_n_mps2": 0.0,
                "command_u_mps2": vertical_acceleration,
            }
        )

    def _finish(self, success: bool, reason: str) -> None:
        if self.done:
            return
        self.done = True
        self.success = success
        altitudes = [row["altitude_enu_m"] for row in self.hold_samples]
        intervals = [(b - a) / 1e9 for a, b in zip(self.heartbeat_ns, self.heartbeat_ns[1:])]
        evidence = {
            "success": success,
            "reason": reason,
            "expected_system_id": 2,
            "reported_system_id": None if self.status is None else int(self.status.system_id),
            "hold_duration_s": 5.0,
            "hold_sample_count": len(self.hold_samples),
            "altitude_mean_m": None if not altitudes else float(np.mean(altitudes)),
            "altitude_peak_to_peak_m": None if not altitudes else float(np.ptp(altitudes)),
            "horizontal_acceleration_command_max_mps2": 0.0,
            "heartbeat_worst_rate_hz": None if not intervals else 1.0 / max(intervals),
            "transitions": self.transitions,
            "samples": self.hold_samples,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        self.get_logger().info(f"interceptor result: {success}: {reason}")

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
                if int(self.status.system_id) != 2:
                    self._finish(False, f"reported system ID {self.status.system_id}")
                    return
                altitude = ned_position_to_enu(
                    [self.position.x, self.position.y, self.position.z]
                )[2]
                self.takeoff_z_enu = float(altitude + 10.0)
                self._transition("PRESTREAM")
            return
        if self.state in {"PRESTREAM", "TAKEOFF", "STABILIZE"}:
            self._publish_position()
        elif self.state == "ACCELERATION_HOLD":
            self._publish_acceleration_hold()

        if self.state == "PRESTREAM":
            self.prestream_count += 1
            if self.prestream_count >= 100:
                self._command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self._command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self._transition("TAKEOFF")
        elif self.state == "TAKEOFF":
            altitude = ned_position_to_enu(
                [self.position.x, self.position.y, self.position.z]
            )[2]
            if (
                self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED
                and abs(altitude - self.takeoff_z_enu) <= 0.75
            ):
                self._transition("STABILIZE")
        elif self.state == "STABILIZE" and self._state_elapsed() >= 2.0:
            self._transition("ACCELERATION_HOLD")
        elif self.state == "ACCELERATION_HOLD" and self._state_elapsed() >= 5.0:
            self._command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.last_land_command_s = self._elapsed()
            self._transition("LAND")
        elif self.state == "LAND":
            if self.status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                altitude_range = np.ptp([row["altitude_enu_m"] for row in self.hold_samples])
                passed = bool(altitude_range <= 1.0)
                self._transition("DONE")
                self._finish(passed, f"acceleration hold altitude peak-to-peak {altitude_range:.3f} m")
            elif self._elapsed() - self.last_land_command_s >= 1.0:
                self._command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.last_land_command_s = self._elapsed()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ral_gazebo_v1/smoke/s2_interceptor_hold.json"),
    )
    parser.add_argument("--timeout-s", type=float, default=70.0)
    args, ros_args = parser.parse_known_args(argv)
    rclpy.init(args=ros_args)
    node = AccelerationHoldSmoke(args.output, args.timeout_s)
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
