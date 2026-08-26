"""Source-stamped target telemetry with independently recorded arrival time."""

from __future__ import annotations

import json
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String

from .config_utils import condition_config, default_config, load_mirror_config
from .delay_emulator import DelayQueue
from .ros_utils import diagnostic, odom_vectors, odometry, stamp_seconds


INTERNAL_MEASUREMENT_TOPIC = "/ras_hw_mirror/internal/target/measurement_state"


class TelemetryEmulatorNode(Node):
    def __init__(self) -> None:
        super().__init__("telemetry_emulator")
        self.declare_parameter("config", str(default_config()))
        self.declare_parameter("condition", "HC1")
        self.declare_parameter("seed", 1)
        self.config = load_mirror_config(self.get_parameter("config").value)
        self.condition_name = str(self.get_parameter("condition").value)
        condition = condition_config(self.config, self.condition_name)
        self.delay = DelayQueue(
            float(condition["delay_ms"]) * 1e-3,
            float(condition["position_noise_std_m"]),
            float(condition["velocity_noise_std_mps"]),
            float(condition["dropout_probability"]),
            int(self.get_parameter("seed").value),
        )
        self.active = False
        self.public_pub = self.create_publisher(PoseWithCovarianceStamped, "/ras_hw_mirror/target/measurement", 10)
        self.state_pub = self.create_publisher(Odometry, INTERNAL_MEASUREMENT_TOPIC, 10)
        self.status_pub = self.create_publisher(DiagnosticArray, "/ras_hw_mirror/telemetry/status", 10)
        self.ready_pub = self.create_publisher(Bool, "/ras_hw_mirror/ready/telemetry", 1)
        self.create_subscription(Odometry, "/ras_hw_mirror/target/truth", self._truth, 50)
        self.create_subscription(String, "/ras_hw_mirror/experiment/phase", self._phase, 10)
        self.create_subscription(String, "/ras_hw_mirror/scenario/selection", self._scenario, 10)
        self.timer = self.create_timer(0.002, self._release)
        self.heartbeat = self.create_timer(0.1, self._ready)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _ready(self) -> None:
        value = Bool()
        value.data = True
        self.ready_pub.publish(value)

    def _phase(self, msg: String) -> None:
        if msg.data == "RUN" and not self.active:
            self.active = True
            self.delay.reset()
        elif msg.data in {"CAPTURE", "HOLD", "ABORT", "LAND", "DONE"}:
            self.active = False

    def _scenario(self, msg: String) -> None:
        try:
            request = json.loads(msg.data)
            condition_name = str(request["condition"])
            seed = int(request["seed"])
            condition = condition_config(self.config, condition_name)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.get_logger().warning("ignored malformed scenario selection")
            return
        self.condition_name = condition_name
        self.delay = DelayQueue(
            float(condition["delay_ms"]) * 1e-3,
            float(condition["position_noise_std_m"]),
            float(condition["velocity_noise_std_mps"]),
            float(condition["dropout_probability"]),
            seed,
        )
        self.active = False

    def _truth(self, msg: Odometry) -> None:
        if not self.active:
            return
        position, velocity = odom_vectors(msg)
        self.delay.enqueue(stamp_seconds(msg.header.stamp), __import__("numpy").concatenate((position, velocity)))

    def _release(self) -> None:
        now = self._now()
        for sample in self.delay.pop_ready(now):
            age_ms = 1000.0 * (now - sample.source_time_s)
            status = diagnostic(
                "ras_hw_mirror/telemetry",
                DiagnosticStatus.WARN if sample.dropped else DiagnosticStatus.OK,
                "DROPPED" if sample.dropped else "DELIVERED",
                packet_id=sample.packet_id,
                condition=self.condition_name,
                source_time_s=f"{sample.source_time_s:.9f}",
                arrival_time_s=f"{now:.9f}",
                actual_age_ms=f"{age_ms:.6f}",
                requested_age_ms=f"{1000.0 * self.delay.delay_s:.6f}",
                dropped=int(sample.dropped),
            )
            status.header.stamp = self.get_clock().now().to_msg()
            self.status_pub.publish(status)
            if sample.dropped:
                continue
            state = odometry("map", "target_measurement", sample.source_time_s, sample.measurement[:3], sample.measurement[3:])
            self.state_pub.publish(state)
            public = PoseWithCovarianceStamped()
            public.header = state.header
            public.pose = state.pose
            self.public_pub.publish(public)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TelemetryEmulatorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
