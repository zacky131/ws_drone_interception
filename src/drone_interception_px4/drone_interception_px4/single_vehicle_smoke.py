"""Recorded PX4 single-vehicle arm/offboard/takeoff/hold/land acceptance run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand
from px4_msgs.msg import VehicleLocalPosition, VehicleStatus


PX4_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class SingleVehicleSmoke(Node):
    """Fly one X500 to 5 m in position offboard mode and land it safely."""

    def __init__(self, output: Path, timeout_s: float = 60.0) -> None:
        super().__init__("single_vehicle_smoke")
        self.output = output
        self.timeout_s = timeout_s
        self.start_ns = time.perf_counter_ns()
        self.state = "WAIT_READY"
        self.state_enter_ns = self.start_ns
        self.prestream_count = 0
        self.status: VehicleStatus | None = None
        self.position: VehicleLocalPosition | None = None
        self.done = False
        self.success = False
        self.samples: list[dict[str, object]] = []
        self.transitions: list[dict[str, object]] = []
        self.heartbeat_ns: list[int] = []
        self.system_id: int | None = None
        self.ground_z_ned_m: float | None = None
        self.desired_z_ned_m: float | None = None

        self.mode_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", PX4_QOS
        )
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", PX4_QOS
        )
        self.command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", PX4_QOS
        )
        self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status", self._status_callback, PX4_QOS
        )
        self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position",
            self._position_callback,
            PX4_QOS,
        )
        self.timer = self.create_timer(0.02, self._tick)

    def _elapsed(self) -> float:
        return (time.perf_counter_ns() - self.start_ns) / 1e9

    def _state_elapsed(self) -> float:
        return (time.perf_counter_ns() - self.state_enter_ns) / 1e9

    def _transition(self, state: str) -> None:
        self.state = state
        self.state_enter_ns = time.perf_counter_ns()
        self.transitions.append(
            {
                "elapsed_s": self._elapsed(),
                "state": state,
                "z_ned_m": None if self.position is None else float(self.position.z),
                "arming_state": None
                if self.status is None
                else int(self.status.arming_state),
                "nav_state": None if self.status is None else int(self.status.nav_state),
            }
        )
        self.get_logger().info(f"acceptance transition: {state}")

    def _status_callback(self, msg: VehicleStatus) -> None:
        self.status = msg
        if self.system_id is None:
            self.system_id = int(msg.system_id)

    def _position_callback(self, msg: VehicleLocalPosition) -> None:
        self.position = msg

    def _publish_offboard_position(self) -> None:
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
        point.position = [0.0, 0.0, float(self.desired_z_ned_m or -5.0)]
        point.velocity = [math.nan, math.nan, math.nan]
        point.acceleration = [math.nan, math.nan, math.nan]
        point.jerk = [math.nan, math.nan, math.nan]
        point.yaw = 0.0
        point.yawspeed = math.nan
        self.setpoint_pub.publish(point)
        self.heartbeat_ns.append(time.perf_counter_ns())

    def _command(self, command: int, param1: float = 0.0, param2: float = 0.0) -> None:
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds // 1000)
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = int(command)
        msg.target_system = int(self.system_id or 1)
        msg.target_component = 1
        msg.source_system = 255
        msg.source_component = 1
        msg.confirmation = 0
        msg.from_external = True
        self.command_pub.publish(msg)

    def _finish(self, success: bool, reason: str) -> None:
        if self.done:
            return
        self.success = success
        self.done = True
        intervals = [
            (b - a) / 1e9 for a, b in zip(self.heartbeat_ns, self.heartbeat_ns[1:])
        ]
        evidence = {
            "success": success,
            "reason": reason,
            "system_id": self.system_id,
            "ground_z_ned_m": self.ground_z_ned_m,
            "desired_z_ned_m": self.desired_z_ned_m,
            "elapsed_s": self._elapsed(),
            "heartbeat_count": len(self.heartbeat_ns),
            "heartbeat_worst_interval_s": max(intervals, default=None),
            "heartbeat_worst_rate_hz":
                None if not intervals else 1.0 / max(intervals),
            "transitions": self.transitions,
            "samples": self.samples,
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        self.get_logger().info(f"acceptance result: {success}: {reason}")

    def _tick(self) -> None:
        if self.done:
            return
        if self._elapsed() > self.timeout_s:
            self._finish(False, f"timeout in {self.state}")
            return

        if self.position is not None and self.status is not None:
            self.samples.append(
                {
                    "elapsed_s": self._elapsed(),
                    "state": self.state,
                    "z_ned_m": float(self.position.z),
                    "vz_ned_mps": float(self.position.vz),
                    "arming_state": int(self.status.arming_state),
                    "nav_state": int(self.status.nav_state),
                    "preflight_checks_pass": bool(self.status.pre_flight_checks_pass),
                }
            )

        if self.state == "WAIT_READY":
            if (
                self.position is not None
                and self.status is not None
                and self.position.xy_valid
                and self.position.z_valid
                and self.status.pre_flight_checks_pass
            ):
                self.ground_z_ned_m = float(self.position.z)
                self.desired_z_ned_m = self.ground_z_ned_m - 5.0
                self._transition("PRESTREAM")
            return

        if self.state in {"PRESTREAM", "TAKEOFF", "HOLD"}:
            self._publish_offboard_position()

        if self.state == "PRESTREAM":
            self.prestream_count += 1
            if self.prestream_count >= 100:
                self._command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self._command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self._transition("TAKEOFF")
        elif self.state == "TAKEOFF":
            if (
                self.position is not None
                and self.status is not None
                and self.status.arming_state == VehicleStatus.ARMING_STATE_ARMED
                and self.desired_z_ned_m is not None
                and self.position.z <= self.desired_z_ned_m + 0.5
            ):
                self._transition("HOLD")
        elif self.state == "HOLD" and self._state_elapsed() >= 4.0:
            self._command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self._transition("LAND")
        elif self.state == "LAND" and self.status is not None:
            # PX4's automatic land detector commonly disarms before the local
            # estimator returns exactly to its takeoff z reference.  The
            # transition to DISARMED is the authoritative touchdown evidence.
            if self.status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                self._transition("DISARM")
            elif (
                self.position is not None
                and self.ground_z_ned_m is not None
                and self.position.z >= self.ground_z_ned_m - 0.15
                and abs(self.position.vz) <= 0.3
            ):
                self._command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
                self._transition("DISARM")
        elif self.state == "DISARM" and self.status is not None:
            if self.status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                self._transition("DONE")
                self._finish(True, "arm/offboard/takeoff/hold/land/disarm completed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ral_gazebo_v1/single_vehicle_smoke.json"),
    )
    parser.add_argument("--timeout-s", type=float, default=60.0)
    args, ros_args = parser.parse_known_args(argv)
    rclpy.init(args=ros_args)
    node = SingleVehicleSmoke(args.output.resolve(), args.timeout_s)
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
