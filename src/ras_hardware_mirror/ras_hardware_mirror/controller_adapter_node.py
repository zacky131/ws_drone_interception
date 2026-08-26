"""Thin common-topic adapter for the frozen M0-prime and M1 controller paths."""

from __future__ import annotations

from collections import deque
import json
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import AccelStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Bool, Float64, String

from drone_interception_px4.telemetry import TelemetryEvent
from m0prime_confirmatory.controller import ConfirmatoryControllerAdapter

from .config_utils import default_config, load_mirror_config, method_code
from .ros_utils import diagnostic, odom_vectors, odometry, stamp_seconds


class ControllerAdapterNode(Node):
    def __init__(self) -> None:
        super().__init__("controller_adapter")
        self.declare_parameter("config", str(default_config()))
        self.declare_parameter("method", "M1")
        self.declare_parameter("condition", "HC1")
        self.declare_parameter("trajectory", "HT1")
        self.declare_parameter("seed", 1)
        self.config = load_mirror_config(self.get_parameter("config").value)
        self.method_public = str(self.get_parameter("method").value)
        self.method_internal = method_code(self.method_public)
        frozen = self.config["controller"]["frozen_config_path"]
        self.adapter = ConfirmatoryControllerAdapter(self.method_internal, frozen, int(self.get_parameter("seed").value))
        self.condition = str(self.get_parameter("condition").value)
        self.trajectory = str(self.get_parameter("trajectory").value)
        self.seed = int(self.get_parameter("seed").value)
        self.epoch_s: float | None = None
        self.active = False
        self.interceptor: Odometry | None = None
        self.truth_cache: deque[tuple[float, np.ndarray]] = deque(maxlen=250)
        self.command_pub = self.create_publisher(AccelStamped, "/ras_hw_mirror/controller/command_raw", 10)
        self.estimate_pub = self.create_publisher(Odometry, "/ras_hw_mirror/target/estimate", 10)
        self.prediction_pub = self.create_publisher(Path, "/ras_hw_mirror/target/prediction_path", 10)
        self.status_pub = self.create_publisher(DiagnosticArray, "/ras_hw_mirror/controller/status", 10)
        self.ready_pub = self.create_publisher(Bool, "/ras_hw_mirror/ready/controller", 1)
        self.scenario_ready_pub = self.create_publisher(String, "/ras_hw_mirror/scenario/ready/controller", 10)
        self.create_subscription(Odometry, "/ras_hw_mirror/interceptor/state/px4", self._interceptor, 10)
        self.create_subscription(Odometry, "/ras_hw_mirror/target/truth", self._truth, 50)
        self.create_subscription(Odometry, "/ras_hw_mirror/internal/target/measurement_state", self._measurement, 50)
        self.create_subscription(Float64, "/ras_hw_mirror/experiment/epoch", self._epoch, 10)
        self.create_subscription(String, "/ras_hw_mirror/experiment/phase", self._phase, 10)
        self.create_subscription(String, "/ras_hw_mirror/scenario/selection", self._scenario, 10)
        self.ready_timer = self.create_timer(0.1, self._ready)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _ready(self) -> None:
        value = Bool()
        value.data = True
        self.ready_pub.publish(value)

    def _epoch(self, msg: Float64) -> None:
        self.epoch_s = float(msg.data)

    def _phase(self, msg: String) -> None:
        if msg.data == "RUN" and not self.active:
            condition = {"HC0": "C0", "HC1": "C2", "DEV0": "C0"}.get(self.condition)
            if condition is None:
                raise ValueError(f"unsupported condition: {self.condition}")
            self.adapter.reset(self.seed, self.trajectory, condition)
            self.active = True
        elif msg.data in {"CAPTURE", "HOLD", "ABORT", "LAND", "DONE"}:
            self.active = False

    def _scenario(self, msg: String) -> None:
        try:
            request = json.loads(msg.data)
            method = str(request["method"])
            condition = str(request["condition"])
            trajectory = str(request["trajectory"])
            seed = int(request["seed"])
            internal = method_code(method)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self.get_logger().warning("ignored malformed scenario selection")
            return
        frozen = self.config["controller"]["frozen_config_path"]
        self.adapter = ConfirmatoryControllerAdapter(internal, frozen, seed)
        self.method_public = method
        self.method_internal = internal
        self.condition = condition
        self.trajectory = trajectory
        self.seed = seed
        self.active = False
        self.epoch_s = None
        self.truth_cache.clear()
        ready = String()
        ready.data = str(request.get("stage", ""))
        self.scenario_ready_pub.publish(ready)

    def _interceptor(self, msg: Odometry) -> None:
        self.interceptor = msg

    def _truth(self, msg: Odometry) -> None:
        p, v = odom_vectors(msg)
        self.truth_cache.append((stamp_seconds(msg.header.stamp), np.concatenate((p, v))))

    def _source_truth(self, stamp_s: float, measured: np.ndarray) -> np.ndarray:
        if not self.truth_cache:
            return measured.copy()
        _, state = min(self.truth_cache, key=lambda item: abs(item[0] - stamp_s))
        return state.copy()

    def _measurement(self, msg: Odometry) -> None:
        if not self.active or self.interceptor is None or self.epoch_s is None:
            return
        started = time.perf_counter_ns()
        now = self._now()
        source_absolute = stamp_seconds(msg.header.stamp)
        source = max(0.0, source_absolute - self.epoch_s)
        arrival = max(source, now - self.epoch_s)
        p, v = odom_vectors(msg)
        measurement = np.concatenate((p, v))
        truth = self._source_truth(source_absolute, measurement)
        event = TelemetryEvent(
            measurement=measurement,
            noise=measurement - truth,
            configured_delay_s=float(arrival - source),
            arrival_timestamp_s=arrival,
            requested_source_timestamp_s=source,
            actual_source_timestamp_s=source,
            history_left_timestamp_s=source,
            history_right_timestamp_s=source,
            interpolation_alpha=0.0,
            startup_clamped=source_absolute < self.epoch_s,
            source_truth_position_enu=truth[:3],
            source_truth_velocity_enu=truth[3:],
            drop=False,
        )
        interceptor_position, interceptor_velocity = odom_vectors(self.interceptor)
        interceptor_acceleration = np.array([
            self.interceptor.twist.twist.angular.x,
            self.interceptor.twist.twist.angular.y,
            self.interceptor.twist.twist.angular.z,
        ], dtype=float)
        try:
            command, info = self.adapter.step(np.concatenate((interceptor_position, interceptor_velocity, interceptor_acceleration)), event, 1.0 / float(self.config["experiment"]["control_rate_hz"]), arrival)
        except Exception as exc:
            status = diagnostic("ras_hw_mirror/controller", DiagnosticStatus.ERROR, "CONTROLLER_ERROR", method=self.method_public, error=repr(exc))
            status.header.stamp = self.get_clock().now().to_msg()
            self.status_pub.publish(status)
            self.get_logger().error(f"controller step failed: {exc!r}")
            return
        command = np.asarray(command, dtype=float).reshape(3)
        out = AccelStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = "map"
        out.accel.linear.x, out.accel.linear.y, out.accel.linear.z = map(float, command)
        self.command_pub.publish(out)
        estimate = np.asarray(info.get("target_estimate", np.concatenate((measurement, np.zeros(3)))), dtype=float)
        estimate_msg = odometry("map", "target_estimate", now, estimate[:3], estimate[3:6])
        if len(estimate) >= 9:
            estimate_msg.twist.twist.angular.x, estimate_msg.twist.twist.angular.y, estimate_msg.twist.twist.angular.z = map(float, estimate[6:9])
        self.estimate_pub.publish(estimate_msg)
        predictions = np.asarray(info.get("ca_predicted_position_horizon", []), dtype=float)
        path = Path()
        path.header = estimate_msg.header
        for row in predictions:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, row[:3])
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.prediction_pub.publish(path)
        status = diagnostic(
            "ras_hw_mirror/controller",
            DiagnosticStatus.OK,
            "READY",
            method=self.method_public,
            method_code=self.method_internal,
            runtime_timing_semantics=info.get("runtime_timing_semantics", "source_time_then_repropagate"),
            source_time_s=f"{source:.9f}",
            arrival_time_s=f"{arrival:.9f}",
            packet_age_ms=f"{1000.0 * (arrival - source):.6f}",
            estimator_time_ms=f"{1000.0 * float(info.get('estimator_time_s', math.nan)):.6f}",
            solve_time_ms=f"{1000.0 * float(info.get('solve_time_s', math.nan)):.6f}",
            complete_loop_ms=f"{(time.perf_counter_ns() - started) * 1e-6:.6f}",
            repropagation_steps=info.get("last_repropagation_steps", 0),
        )
        status.header.stamp = self.get_clock().now().to_msg()
        self.status_pub.publish(status)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControllerAdapterNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
