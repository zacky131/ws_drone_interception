"""Read-only ROS diagnostic for the verified PX4 identity/state boundary."""

from __future__ import annotations

import argparse
import json

import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleLocalPosition, VehicleStatus

from .frames import ned_position_to_enu, ned_velocity_to_enu
from .px4_contract import INTERCEPTOR, TARGET
from .single_vehicle_smoke import PX4_QOS


class StateBridgeDiagnostic(Node):
    def __init__(self) -> None:
        super().__init__("px4_state_bridge")
        self.states = {}
        for identity in (INTERCEPTOR, TARGET):
            prefix = f"/{identity.namespace}/fmu/out"
            self.create_subscription(
                VehicleLocalPosition, f"{prefix}/vehicle_local_position",
                lambda msg, role=identity.role: self._position(role, msg), PX4_QOS,
            )
            self.create_subscription(
                VehicleStatus, f"{prefix}/vehicle_status",
                lambda msg, role=identity.role: self._status(role, msg), PX4_QOS,
            )

    def _position(self, role, msg):
        state = self.states.setdefault(role, {})
        state["position_enu_m"] = ned_position_to_enu([msg.x, msg.y, msg.z]).tolist()
        state["velocity_enu_mps"] = ned_velocity_to_enu([msg.vx, msg.vy, msg.vz]).tolist()

    def _status(self, role, msg):
        state = self.states.setdefault(role, {})
        state["system_id"] = int(msg.system_id)
        state["nav_state"] = int(msg.nav_state)
        state["arming_state"] = int(msg.arming_state)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=float, default=2.0)
    args, ros_args = parser.parse_known_args(argv)
    rclpy.init(args=ros_args)
    node = StateBridgeDiagnostic()
    end_ns = node.get_clock().now().nanoseconds + int(args.duration_s * 1e9)
    while rclpy.ok() and node.get_clock().now().nanoseconds < end_ns:
        rclpy.spin_once(node, timeout_sec=0.1)
    print(json.dumps(node.states, indent=2))
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

