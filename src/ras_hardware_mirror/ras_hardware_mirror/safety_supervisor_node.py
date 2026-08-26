"""Method-independent safety monitor for simulation and later hardware reuse."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import AccelStamped
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleStatus
from std_msgs.msg import Bool, String

from .config_utils import default_config, default_field, load_mirror_config, load_yaml
from .geometry_utils import inside_altitude, inside_horizontal_box
from .ros_utils import diagnostic, odom_vectors
from .state_types import NavigationState, SafetyInputs
from .gazebo_state_provider_node import PX4_QOS
from .manual_control import px4_fmu_prefix


@dataclass(frozen=True)
class SafetyDecision:
    abort: bool
    reason: str = ""


class SafetyMonitor:
    def __init__(self, field: dict, timeout_s: float, heartbeat_timeout_s: float, startup_grace_s: float = 0.5) -> None:
        self.field = field
        self.timeout_s = float(timeout_s)
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        self.startup_grace_s = float(startup_grace_s)

    def evaluate(self, values: SafetyInputs) -> SafetyDecision:
        if values.manual_abort:
            return SafetyDecision(True, "manual abort")
        if values.interceptor is None or not values.interceptor.valid:
            return SafetyDecision(True, "invalid interceptor state")
        if values.target is None or not values.target.valid:
            return SafetyDecision(True, "invalid target state")
        if not values.px4_healthy:
            return SafetyDecision(True, "PX4/offboard unhealthy")
        run_age = math.inf if values.run_start_s is None else values.now_s - values.run_start_s
        if run_age > self.startup_grace_s:
            if not values.command_finite:
                return SafetyDecision(True, "non-finite controller command")
            if values.controller_age_s > self.heartbeat_timeout_s:
                return SafetyDecision(True, "controller heartbeat timeout")
        if not inside_horizontal_box(values.interceptor.position_enu, self.field["hard_geofence"]):
            return SafetyDecision(True, "interceptor hard-geofence violation")
        if not inside_altitude(values.interceptor.position_enu, self.field["altitude"]):
            return SafetyDecision(True, "interceptor altitude violation")
        if not inside_horizontal_box(values.target.position_enu, self.field["target_region"]):
            return SafetyDecision(True, "target-region violation")
        if values.run_start_s is not None and values.now_s - values.run_start_s > self.timeout_s:
            return SafetyDecision(True, "experiment timeout")
        return SafetyDecision(False)


def run_gs6_gate() -> tuple[bool, str]:
    """Run the deterministic, non-flight safety cases used by keyboard key 7."""
    field = load_yaml(default_field())
    monitor = SafetyMonitor(field, 30.0, 0.25)

    def state(position=(0.0, 0.0, 5.0), valid=True):
        return NavigationState(0.0, np.asarray(position, dtype=float), np.zeros(3), valid=valid)

    def inputs(**changes):
        values = dict(
            now_s=3.0,
            run_start_s=0.0,
            interceptor=state(),
            target=state(),
            controller_age_s=0.01,
            command_finite=True,
            px4_healthy=True,
            manual_abort=False,
        )
        values.update(changes)
        return SafetyInputs(**values)

    cases = [
        (inputs(interceptor=state((41.0, 0.0, 5.0))), "interceptor hard-geofence violation"),
        (inputs(target=state((31.0, 0.0, 5.0))), "target-region violation"),
        (inputs(interceptor=state(valid=False)), "invalid interceptor state"),
        (inputs(controller_age_s=0.3), "controller heartbeat timeout"),
        (inputs(command_finite=False), "non-finite controller command"),
        (inputs(manual_abort=True), "manual abort"),
        (inputs(now_s=31.0), "experiment timeout"),
    ]
    failures = [expected for values, expected in cases if monitor.evaluate(values).reason != expected]
    if monitor.evaluate(inputs()).abort:
        failures.append("nominal state unexpectedly aborted")
    return not failures, "8 deterministic cases" if not failures else "; ".join(failures)


class SafetySupervisorNode(Node):
    def __init__(self) -> None:
        super().__init__("safety_supervisor")
        self.declare_parameter("config", str(default_config()))
        self.declare_parameter("field", str(default_field()))
        self.config = load_mirror_config(self.get_parameter("config").value)
        self.field = load_yaml(self.get_parameter("field").value)
        self.monitor = SafetyMonitor(self.field, self.config["experiment"]["trial_timeout_s"], self.config["controller"]["heartbeat_timeout_s"], self.config["controller"]["startup_grace_s"])
        self.phase = "PRECHECK"
        self.run_start_s: float | None = None
        self.interceptor: NavigationState | None = None
        self.target: NavigationState | None = None
        self.controller_stamp_s = -math.inf
        self.command_finite = False
        self.px4_healthy = False
        self.manual_abort = False
        self.latched_reason = ""
        self.abort_pub = self.create_publisher(Bool, "/ras_hw_mirror/safety/abort", 10)
        self.status_pub = self.create_publisher(DiagnosticArray, "/ras_hw_mirror/safety/status", 10)
        self.create_subscription(String, "/ras_hw_mirror/experiment/phase", self._phase, 10)
        self.create_subscription(Odometry, "/ras_hw_mirror/interceptor/state/px4", self._interceptor, 10)
        self.create_subscription(Odometry, "/ras_hw_mirror/target/truth", self._target, 10)
        self.create_subscription(AccelStamped, "/ras_hw_mirror/controller/command_raw", self._command, 10)
        self.create_subscription(Bool, "/ras_hw_mirror/safety/manual_abort", self._manual, 10)
        prefix = px4_fmu_prefix(self.config["interceptor"]["px4_namespace"])
        self.create_subscription(VehicleStatus, f"{prefix}/out/vehicle_status", self._px4, PX4_QOS)
        self.timer = self.create_timer(0.02, self._tick)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _state(msg: Odometry, source: str) -> NavigationState:
        p, v = odom_vectors(msg)
        finite = bool(np.all(np.isfinite(p)) and np.all(np.isfinite(v)) and all(np.isfinite(msg.pose.covariance)))
        return NavigationState(0.0, p, v, valid=finite, quality="valid" if finite else "invalid", source=source)

    def _phase(self, msg: String) -> None:
        self.phase = msg.data
        if msg.data == "RUN":
            self.run_start_s = self._now()
            self.latched_reason = ""
            self.manual_abort = False

    def _interceptor(self, msg: Odometry) -> None:
        self.interceptor = self._state(msg, "px4")

    def _target(self, msg: Odometry) -> None:
        self.target = self._state(msg, "virtual_target")

    def _command(self, msg: AccelStamped) -> None:
        values = [msg.accel.linear.x, msg.accel.linear.y, msg.accel.linear.z]
        self.command_finite = bool(np.all(np.isfinite(values)))
        self.controller_stamp_s = self._now()

    def _manual(self, msg: Bool) -> None:
        self.manual_abort = self.manual_abort or bool(msg.data)

    def _px4(self, msg: VehicleStatus) -> None:
        self.px4_healthy = bool(int(msg.system_id) == int(self.config["interceptor"]["expected_system_id"]) and int(msg.arming_state) == VehicleStatus.ARMING_STATE_ARMED)

    def _tick(self) -> None:
        now = self._now()
        decision = SafetyDecision(False)
        if self.phase == "RUN" and not self.latched_reason:
            decision = self.monitor.evaluate(SafetyInputs(now, self.run_start_s, self.interceptor, self.target, now - self.controller_stamp_s, self.command_finite, self.px4_healthy, self.manual_abort))
            if decision.abort:
                self.latched_reason = decision.reason
        abort = Bool()
        abort.data = bool(self.latched_reason)
        self.abort_pub.publish(abort)
        level = DiagnosticStatus.ERROR if self.latched_reason else DiagnosticStatus.OK
        status = diagnostic("ras_hw_mirror/safety", level, "ABORT" if self.latched_reason else "OK", phase=self.phase, abort=int(bool(self.latched_reason)), reason=self.latched_reason or "none")
        status.header.stamp = self.get_clock().now().to_msg()
        self.status_pub.publish(status)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafetySupervisorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
