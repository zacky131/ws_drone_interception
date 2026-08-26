"""Recorded two-X500 namespace, identity, and independent-control acceptance run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand
from px4_msgs.msg import VehicleLocalPosition, VehicleStatus

from .single_vehicle_smoke import PX4_QOS


class TwoVehicleSmoke(Node):
    IDENTITIES = {
        "px4_1": {"system_id": 2, "independent_x_ned_m": 3.0},
        "px4_2": {"system_id": 3, "independent_x_ned_m": -3.0},
    }

    def __init__(self, output: Path, timeout_s: float = 90.0) -> None:
        super().__init__("two_vehicle_smoke")
        self.output = output
        self.timeout_s = timeout_s
        self.start_ns = time.perf_counter_ns()
        self.state_enter_ns = self.start_ns
        self.state = "WAIT_READY"
        self.done = False
        self.success = False
        self.prestream_count = 0
        self.status: dict[str, VehicleStatus] = {}
        self.position: dict[str, VehicleLocalPosition] = {}
        self.ground_z: dict[str, float] = {}
        self.desired_z: dict[str, float] = {}
        self.mode_pubs = {}
        self.setpoint_pubs = {}
        self.command_pubs = {}
        self.heartbeat_ns: dict[str, list[int]] = {name: [] for name in self.IDENTITIES}
        self.transitions: list[dict[str, object]] = []
        self.samples: list[dict[str, object]] = []

        for namespace in self.IDENTITIES:
            prefix = f"/{namespace}/fmu"
            self.mode_pubs[namespace] = self.create_publisher(
                OffboardControlMode, f"{prefix}/in/offboard_control_mode", PX4_QOS
            )
            self.setpoint_pubs[namespace] = self.create_publisher(
                TrajectorySetpoint, f"{prefix}/in/trajectory_setpoint", PX4_QOS
            )
            self.command_pubs[namespace] = self.create_publisher(
                VehicleCommand, f"{prefix}/in/vehicle_command", PX4_QOS
            )
            self.create_subscription(
                VehicleStatus,
                f"{prefix}/out/vehicle_status",
                lambda msg, ns=namespace: self._status(ns, msg),
                PX4_QOS,
            )
            self.create_subscription(
                VehicleLocalPosition,
                f"{prefix}/out/vehicle_local_position",
                lambda msg, ns=namespace: self._position(ns, msg),
                PX4_QOS,
            )
        self.timer = self.create_timer(0.02, self._tick)

    def _elapsed(self) -> float:
        return (time.perf_counter_ns() - self.start_ns) / 1e9

    def _state_elapsed(self) -> float:
        return (time.perf_counter_ns() - self.state_enter_ns) / 1e9

    def _status(self, namespace: str, msg: VehicleStatus) -> None:
        self.status[namespace] = msg

    def _position(self, namespace: str, msg: VehicleLocalPosition) -> None:
        self.position[namespace] = msg

    def _transition(self, state: str) -> None:
        self.state = state
        self.state_enter_ns = time.perf_counter_ns()
        self.transitions.append(
            {
                "elapsed_s": self._elapsed(),
                "state": state,
                "vehicles": {
                    ns: {
                        "position_ned_m": None
                        if ns not in self.position
                        else [
                            float(self.position[ns].x),
                            float(self.position[ns].y),
                            float(self.position[ns].z),
                        ],
                        "arming_state": None
                        if ns not in self.status
                        else int(self.status[ns].arming_state),
                        "nav_state": None
                        if ns not in self.status
                        else int(self.status[ns].nav_state),
                    }
                    for ns in self.IDENTITIES
                },
            }
        )
        self.get_logger().info(f"acceptance transition: {state}")

    def _publish_position(self, namespace: str, x_ned_m: float) -> None:
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
        self.mode_pubs[namespace].publish(mode)

        point = TrajectorySetpoint()
        point.timestamp = timestamp_us
        point.position = [float(x_ned_m), 0.0, self.desired_z[namespace]]
        point.velocity = [math.nan, math.nan, math.nan]
        point.acceleration = [math.nan, math.nan, math.nan]
        point.jerk = [math.nan, math.nan, math.nan]
        point.yaw = 0.0
        point.yawspeed = math.nan
        self.setpoint_pubs[namespace].publish(point)
        self.heartbeat_ns[namespace].append(time.perf_counter_ns())

    def _command(
        self, namespace: str, command: int, param1: float = 0.0, param2: float = 0.0
    ) -> None:
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds // 1000)
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = int(command)
        msg.target_system = self.IDENTITIES[namespace]["system_id"]
        msg.target_component = 1
        msg.source_system = 255
        msg.source_component = 1
        msg.confirmation = 0
        msg.from_external = True
        self.command_pubs[namespace].publish(msg)

    def _finish(self, success: bool, reason: str) -> None:
        if self.done:
            return
        self.done = True
        self.success = success
        heartbeat = {}
        for namespace, timestamps in self.heartbeat_ns.items():
            intervals = [(b - a) / 1e9 for a, b in zip(timestamps, timestamps[1:])]
            heartbeat[namespace] = {
                "count": len(timestamps),
                "worst_interval_s": max(intervals, default=None),
                "worst_rate_hz": None if not intervals else 1.0 / max(intervals),
            }
        evidence = {
            "success": success,
            "reason": reason,
            "elapsed_s": self._elapsed(),
            "identities": {
                ns: {
                    "expected_system_id": contract["system_id"],
                    "reported_system_id": None
                    if ns not in self.status
                    else int(self.status[ns].system_id),
                }
                for ns, contract in self.IDENTITIES.items()
            },
            "ground_z_ned_m": self.ground_z,
            "desired_z_ned_m": self.desired_z,
            "heartbeat": heartbeat,
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
        if all(ns in self.status and ns in self.position for ns in self.IDENTITIES):
            self.samples.append(
                {
                    "elapsed_s": self._elapsed(),
                    "state": self.state,
                    "vehicles": {
                        ns: {
                            "position_ned_m": [
                                float(self.position[ns].x),
                                float(self.position[ns].y),
                                float(self.position[ns].z),
                            ],
                            "velocity_ned_mps": [
                                float(self.position[ns].vx),
                                float(self.position[ns].vy),
                                float(self.position[ns].vz),
                            ],
                            "arming_state": int(self.status[ns].arming_state),
                            "nav_state": int(self.status[ns].nav_state),
                        }
                        for ns in self.IDENTITIES
                    },
                }
            )

        if self.state == "WAIT_READY":
            if all(
                ns in self.status
                and ns in self.position
                and self.status[ns].pre_flight_checks_pass
                and self.position[ns].xy_valid
                and self.position[ns].z_valid
                for ns in self.IDENTITIES
            ):
                for ns, contract in self.IDENTITIES.items():
                    reported = int(self.status[ns].system_id)
                    if reported != contract["system_id"]:
                        self._finish(
                            False,
                            f"{ns} reported system ID {reported}, expected {contract['system_id']}",
                        )
                        return
                    self.ground_z[ns] = float(self.position[ns].z)
                    self.desired_z[ns] = self.ground_z[ns] - 10.0
                self._transition("PRESTREAM")
            return

        if self.state in {"PRESTREAM", "TAKEOFF", "HOLD", "INDEPENDENT"}:
            for ns, contract in self.IDENTITIES.items():
                x = contract["independent_x_ned_m"] if self.state == "INDEPENDENT" else 0.0
                self._publish_position(ns, x)

        if self.state == "PRESTREAM":
            self.prestream_count += 1
            if self.prestream_count >= 100:
                for ns in self.IDENTITIES:
                    self._command(ns, VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                    self._command(ns, VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self._transition("TAKEOFF")
        elif self.state == "TAKEOFF":
            if all(
                self.status[ns].arming_state == VehicleStatus.ARMING_STATE_ARMED
                and self.position[ns].z <= self.desired_z[ns] + 0.75
                for ns in self.IDENTITIES
            ):
                self._transition("HOLD")
        elif self.state == "HOLD" and self._state_elapsed() >= 4.0:
            self._transition("INDEPENDENT")
        elif self.state == "INDEPENDENT":
            independent_reached = (
                self.position["px4_1"].x >= 2.0
                and self.position["px4_2"].x <= -2.0
            )
            if independent_reached and self._state_elapsed() >= 4.0:
                for ns in self.IDENTITIES:
                    self._command(ns, VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self._transition("LAND")
            elif self._state_elapsed() >= 10.0:
                self._finish(False, "independent lateral positions were not reached")
        elif self.state == "LAND":
            for ns in self.IDENTITIES:
                if (
                    self.status[ns].arming_state == VehicleStatus.ARMING_STATE_ARMED
                    and self.position[ns].z >= self.ground_z[ns] - 0.15
                    and abs(self.position[ns].vz) <= 0.3
                ):
                    self._command(ns, VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 0.0)
            if all(
                self.status[ns].arming_state == VehicleStatus.ARMING_STATE_DISARMED
                for ns in self.IDENTITIES
            ):
                self._transition("DONE")
                self._finish(
                    True,
                    "both vehicles took off to 10 m, accepted distinct commands, landed, and disarmed",
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ral_gazebo_v1/two_vehicle_smoke.json"),
    )
    parser.add_argument("--timeout-s", type=float, default=90.0)
    args, ros_args = parser.parse_known_args(argv)
    rclpy.init(args=ros_args)
    node = TwoVehicleSmoke(args.output.resolve(), args.timeout_s)
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
