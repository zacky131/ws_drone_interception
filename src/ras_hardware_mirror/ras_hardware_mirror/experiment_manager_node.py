"""Hardware-like PX4 state machine; scientific methods cannot bypass it."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import AccelStamped, TwistStamped
from nav_msgs.msg import Odometry
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus
from std_msgs.msg import Bool, Float64, String

from .config_utils import default_config, load_mirror_config
from .gazebo_state_provider_node import PX4_QOS
from .geometry_utils import distance, enu_to_ned
from .manifest_utils import default_manifest, load_campaign_scenarios
from .manual_control import SCENARIOS, px4_fmu_prefix
from .ros_utils import diagnostic, odom_vectors
from .state_types import ExperimentPhase


@dataclass
class ExperimentStateMachine:
    phase: ExperimentPhase = ExperimentPhase.PRECHECK

    def event(self, name: str) -> ExperimentPhase:
        transitions = {
            (ExperimentPhase.PRECHECK, "ready"): ExperimentPhase.TAKEOFF,
            (ExperimentPhase.TAKEOFF, "altitude_reached"): ExperimentPhase.STABILIZE,
            (ExperimentPhase.STABILIZE, "stable"): ExperimentPhase.RUN,
            (ExperimentPhase.RUN, "capture"): ExperimentPhase.CAPTURE,
            (ExperimentPhase.RUN, "abort"): ExperimentPhase.ABORT,
            (ExperimentPhase.CAPTURE, "settle"): ExperimentPhase.HOLD,
            (ExperimentPhase.ABORT, "settle"): ExperimentPhase.HOLD,
            (ExperimentPhase.HOLD, "land"): ExperimentPhase.LAND,
            (ExperimentPhase.HOLD, "finish"): ExperimentPhase.DONE,
            (ExperimentPhase.LAND, "disarmed"): ExperimentPhase.DONE,
        }
        key = (self.phase, name)
        if key not in transitions:
            raise ValueError(f"invalid experiment transition {self.phase.value} + {name}")
        self.phase = transitions[key]
        return self.phase


NAV_STATE_NAMES = {
    0: "MANUAL",
    1: "ALTCTL",
    2: "POSCTL",
    3: "AUTO_MISSION",
    4: "AUTO_LOITER",
    5: "AUTO_RTL",
    10: "ACRO",
    14: "OFFBOARD",
    15: "STABILIZED",
    18: "AUTO_LAND",
}


class ExperimentManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("experiment_manager")
        self.declare_parameter("config", str(default_config()))
        self.declare_parameter("method", "M1")
        self.declare_parameter("trajectory", "HT1")
        self.declare_parameter("condition", "HC1")
        self.declare_parameter("seed", 1)
        self.declare_parameter("visualization_only", False)
        self.declare_parameter("manifest", str(default_manifest()))
        self.config = load_mirror_config(self.get_parameter("config").value)
        self.campaign_scenarios = load_campaign_scenarios(self.get_parameter("manifest").value)
        self.allowed_scenarios = {
            **{value["stage"]: value for value in SCENARIOS.values()},
            **{value["stage"]: value for value in self.campaign_scenarios.values()},
        }
        self.method = str(self.get_parameter("method").value)
        self.trajectory = str(self.get_parameter("trajectory").value)
        self.condition = str(self.get_parameter("condition").value)
        self.seed = int(self.get_parameter("seed").value)
        self.machine = ExperimentStateMachine()
        self.phase_enter_s = self._now()
        self.epoch_s: float | None = None
        self.interceptor: Odometry | None = None
        self.interceptor_ground_truth: Odometry | None = None
        self.target: Odometry | None = None
        self.status: VehicleStatus | None = None
        self.local_position: VehicleLocalPosition | None = None
        self.px4_reference_ned: np.ndarray | None = None
        self.command: np.ndarray | None = None
        self.command_stamp_s = -math.inf
        self.ready = {key: False for key in ("virtual_target", "telemetry", "state_provider", "controller")}
        self.safety_abort = False
        self.capture_time_s: float | None = None
        self.minimum_separation_m = math.inf
        self.hold_position_map: np.ndarray | None = None
        self.prestream_count = 0
        self.last_vehicle_command_s = -math.inf
        self.manual_command = np.zeros(4, dtype=float)
        self.manual_command_stamp_s = -math.inf
        self.manual_override_latched = False
        self.rc_takeover_active = False
        self.offboard_requested_s: float | None = None
        self.last_offboard_command_s = -math.inf
        self.takeoff_active = False
        self.takeoff_position_map: np.ndarray | None = None
        self.land_requested = False
        self.pending_scenario: dict | None = None
        self.pending_scenario_s = -math.inf
        self.controller_scenario_ready_stage = ""
        interceptor = self.config["interceptor"]
        prefix = px4_fmu_prefix(interceptor["px4_namespace"])
        self.mode_pub = self.create_publisher(OffboardControlMode, f"{prefix}/in/offboard_control_mode", PX4_QOS)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, f"{prefix}/in/trajectory_setpoint", PX4_QOS)
        self.vehicle_command_pub = self.create_publisher(VehicleCommand, f"{prefix}/in/vehicle_command", PX4_QOS)
        self.phase_pub = self.create_publisher(String, "/ras_hw_mirror/experiment/phase", 10)
        self.epoch_pub = self.create_publisher(Float64, "/ras_hw_mirror/experiment/epoch", 10)
        self.status_pub = self.create_publisher(DiagnosticArray, "/ras_hw_mirror/experiment/status", 10)
        self.scenario_pub = self.create_publisher(String, "/ras_hw_mirror/scenario/selection", 10)
        self.applied_pub = self.create_publisher(AccelStamped, "/ras_hw_mirror/controller/command_applied", 10)
        self.create_subscription(VehicleStatus, f"{prefix}/out/vehicle_status", self._px4_status, PX4_QOS)
        self.create_subscription(VehicleLocalPosition, f"{prefix}/out/vehicle_local_position", self._local_position, PX4_QOS)
        self.create_subscription(Odometry, "/ras_hw_mirror/interceptor/state/px4", self._interceptor, 10)
        self.create_subscription(Odometry, "/ras_hw_mirror/interceptor/state/ground_truth", self._interceptor_ground_truth, 10)
        self.create_subscription(Odometry, "/ras_hw_mirror/target/truth", self._target, 10)
        self.create_subscription(AccelStamped, "/ras_hw_mirror/controller/command_raw", self._controller, 10)
        self.create_subscription(Bool, "/ras_hw_mirror/safety/abort", self._abort, 10)
        self.create_subscription(TwistStamped, "/ras_hw_mirror/manual/velocity", self._manual_velocity, 10)
        self.create_subscription(String, "/ras_hw_mirror/manual/action", self._manual_action, 10)
        self.create_subscription(String, "/ras_hw_mirror/scenario/request", self._scenario_request, 10)
        self.create_subscription(String, "/ras_hw_mirror/scenario/ready/controller", self._controller_scenario_ready, 10)
        for key in self.ready:
            self.create_subscription(Bool, f"/ras_hw_mirror/ready/{key}", lambda msg, name=key: self._ready(name, msg), 10)
        rate = float(self.config["experiment"]["control_rate_hz"])
        self.timer = self.create_timer(1.0 / rate, self._tick)
        self._publish_phase()

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _ready(self, key: str, msg: Bool) -> None:
        self.ready[key] = bool(msg.data)

    def _px4_status(self, msg: VehicleStatus) -> None:
        prev_status = self.status
        self.status = msg
        nav_state = int(msg.nav_state)
        is_offboard = (nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD)

        if prev_status is not None:
            prev_nav_state = int(prev_status.nav_state)
            was_offboard = (prev_nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD)
            # Detect RC / QGC mode switch away from OFFBOARD while active
            if was_offboard and not is_offboard:
                mode_name = NAV_STATE_NAMES.get(nav_state, f"MODE_{nav_state}")
                self.get_logger().warn(
                    f"⚠️ MANUAL TAKEOVER / MODE SWITCH DETECTED: PX4 transitioned from OFFBOARD -> {mode_name} (nav_state={nav_state}). "
                    f"Yielding control immediately and halting ROS Offboard setpoints."
                )
                self.rc_takeover_active = True
                self.manual_override_latched = True
                self.offboard_requested_s = None
                self.pending_scenario = None
                if self.machine.phase in {ExperimentPhase.TAKEOFF, ExperimentPhase.STABILIZE, ExperimentPhase.RUN}:
                    self._transition("abort")

    def _local_position(self, msg: VehicleLocalPosition) -> None:
        self.local_position = msg
        if self.px4_reference_ned is None and bool(msg.xy_valid and msg.z_valid):
            self.px4_reference_ned = np.array([msg.x, msg.y, msg.z], dtype=float)

    def _interceptor(self, msg: Odometry) -> None:
        self.interceptor = msg

    def _interceptor_ground_truth(self, msg: Odometry) -> None:
        self.interceptor_ground_truth = msg

    def _target(self, msg: Odometry) -> None:
        self.target = msg

    def _controller(self, msg: AccelStamped) -> None:
        self.command = np.array([msg.accel.linear.x, msg.accel.linear.y, msg.accel.linear.z], dtype=float)
        self.command_stamp_s = self._now()

    def _abort(self, msg: Bool) -> None:
        self.safety_abort = self.safety_abort or bool(msg.data)

    def _controller_scenario_ready(self, msg: String) -> None:
        self.controller_scenario_ready_stage = msg.data

    def _is_armed(self) -> bool:
        return self.status is not None and int(self.status.arming_state) == VehicleStatus.ARMING_STATE_ARMED

    def _is_offboard(self) -> bool:
        return self.status is not None and int(self.status.nav_state) == VehicleStatus.NAVIGATION_STATE_OFFBOARD

    def _manual_velocity(self, msg: TwistStamped) -> None:
        self.manual_command = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z, msg.twist.angular.z], dtype=float)
        self.manual_command_stamp_s = self._now()
        if np.linalg.norm(self.manual_command) > 1e-6 and self.machine.phase == ExperimentPhase.RUN:
            self.manual_override_latched = True

    def _manual_action(self, msg: String) -> None:
        action = msg.data.strip().upper()
        now = self._now()
        if action == "ARM":
            if not self._is_armed():
                self._vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
                self.get_logger().info("keyboard ARM requested; sent ARM command to PX4")
            else:
                self.get_logger().info("PX4 is already ARMED")
        elif action == "OFFBOARD":
            self.rc_takeover_active = False
            self.manual_override_latched = False
            if self.machine.phase in {ExperimentPhase.DONE, ExperimentPhase.ABORT, ExperimentPhase.HOLD}:
                self.machine.phase = ExperimentPhase.PRECHECK
                self.phase_enter_s = now
                self.safety_abort = False
                self.capture_time_s = None
                self.minimum_separation_m = math.inf
                self.land_requested = False
                self._publish_phase()
            self.offboard_requested_s = now
            self.last_offboard_command_s = -math.inf
            self.get_logger().info("keyboard OFFBOARD requested; prestreaming zero-velocity setpoints and switching to OFFBOARD")
        elif action == "TAKEOFF":
            takeoff_state = self.interceptor_ground_truth if self.interceptor_ground_truth is not None else self.interceptor
            if not self._is_armed() or takeoff_state is None:
                self.get_logger().warning("keyboard TAKEOFF rejected: PX4 is not armed or position is unavailable")
                return
            position, _ = odom_vectors(takeoff_state)
            self.takeoff_position_map = position.copy()
            self.takeoff_position_map[2] = (
                float(self.config["interceptor"]["initial_position_enu_m"][2])
                + float(self.config["manual_control"]["takeoff_altitude_m"])
            )
            self.takeoff_active = True
        elif action == "HOLD":
            self.pending_scenario = None
            self.takeoff_active = False
            self.rc_takeover_active = True
            self.manual_override_latched = True
            self.offboard_requested_s = None
            if self.machine.phase in {ExperimentPhase.TAKEOFF, ExperimentPhase.STABILIZE, ExperimentPhase.RUN}:
                self._transition("abort")
            self._vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0)
            self.get_logger().warning("keyboard HOLD requested: commanded PX4 AUTO_LOITER (Hold) mode and halted Offboard streaming")
        elif action == "LAND":
            self.pending_scenario = None
            self.takeoff_active = False
            self.land_requested = True
            self.manual_override_latched = False
            self.offboard_requested_s = None
            if self.machine.phase != ExperimentPhase.DONE:
                self.machine.phase = ExperimentPhase.LAND
                self.phase_enter_s = now
                self._publish_phase()
            self.get_logger().warning("keyboard LAND requested")

    def _scenario_request(self, msg: String) -> None:
        try:
            request = json.loads(msg.data)
        except (TypeError, json.JSONDecodeError):
            self.get_logger().warning("rejected malformed scenario request")
            return
        expected = self.allowed_scenarios.get(request.get("stage"))
        if expected is None or any(request.get(key) != expected[key] for key in expected):
            self.get_logger().warning(f"rejected unknown or altered scenario request: {request!r}")
            return
        if not self._is_armed():
            self.get_logger().warning(f"rejected {expected['stage']}: PX4 must report ARMED")
            return
        if self.machine.phase != ExperimentPhase.PRECHECK:
            self.get_logger().warning(
                f"rejected {expected['stage']}: experiment state must be PRECHECK, "
                f"not {self.machine.phase.value}"
            )
            return
        selection = String()
        selection.data = json.dumps(expected, separators=(",", ":"))
        self.scenario_pub.publish(selection)
        if expected["stage"] == "GS6":
            self.get_logger().info("GS6 selected: run the deterministic in-process safety gate")
            self._run_gs6_gate()
            return
        self.pending_scenario = dict(expected)
        self.pending_scenario_s = self._now()
        self.controller_scenario_ready_stage = ""
        self.get_logger().info(f"accepted armed scenario request {expected['stage']}; synchronizing nodes")

    def _run_gs6_gate(self) -> None:
        from .safety_supervisor_node import run_gs6_gate
        passed, detail = run_gs6_gate()
        log = self.get_logger().info if passed else self.get_logger().error
        log(f"GS6 {'PASS' if passed else 'FAIL'}: {detail}")

    def _prepare_scenario(self) -> None:
        request = self.pending_scenario
        if request is None:
            return
        self.method = request["method"]
        self.trajectory = request["trajectory"]
        self.condition = request["condition"]
        self.seed = int(request["seed"])
        self.pending_scenario = None
        self.takeoff_active = False
        self.land_requested = False
        self.manual_override_latched = False
        self.capture_time_s = None
        self.minimum_separation_m = math.inf
        self.hold_position_map = None
        self.prestream_count = 0
        self.epoch_s = None
        self.safety_abort = False
        self.command = None
        self.command_stamp_s = -math.inf
        # A selected scenario must use the complete flight state machine.  The
        # manual `t` action is only a 2 m safety takeoff; scientific RUN begins
        # after climbing to the configured nominal altitude and stabilizing.
        self._transition("ready")
        self.get_logger().info(
            f"prepared {request['stage']}: TAKEOFF to "
            f"{float(self.config['interceptor']['nominal_altitude_m']):.1f} m, "
            "then STABILIZE before RUN"
        )

    def _publish_phase(self) -> None:
        msg = String()
        msg.data = self.machine.phase.value
        self.phase_pub.publish(msg)

    def _transition(self, event: str) -> None:
        previous = self.machine.phase
        phase = self.machine.event(event)
        self.phase_enter_s = self._now()
        if phase == ExperimentPhase.RUN:
            self.epoch_s = self.phase_enter_s
            epoch = Float64()
            epoch.data = self.epoch_s
            self.epoch_pub.publish(epoch)
            self.safety_abort = False
            self.command = None
            self.command_stamp_s = -math.inf
        if phase in {ExperimentPhase.CAPTURE, ExperimentPhase.ABORT} and self.interceptor is not None:
            self.hold_position_map = odom_vectors(self.interceptor)[0]
        self._publish_phase()
        self.get_logger().info(f"state: {previous.value} -> {phase.value} ({event})")

    def _vehicle_command(self, command: int, param1: float = 0.0, param2: float = 0.0) -> None:
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds // 1000)
        msg.param1, msg.param2, msg.command = float(param1), float(param2), int(command)
        msg.target_system = int(self.config["interceptor"]["expected_system_id"])
        msg.target_component, msg.source_system, msg.source_component = 1, 255, 1
        msg.confirmation, msg.from_external = 0, True
        self.vehicle_command_pub.publish(msg)

    def _position_setpoint(self, map_position: np.ndarray) -> None:
        if self.rc_takeover_active or (not self._is_offboard() and self.machine.phase != ExperimentPhase.TAKEOFF):
            return
        local_enu = np.asarray(map_position) - np.asarray(self.config["interceptor"]["initial_position_enu_m"], dtype=float)
        reference = np.zeros(3) if self.px4_reference_ned is None else self.px4_reference_ned
        position_ned = reference + enu_to_ned(local_enu)
        now_us = int(self.get_clock().now().nanoseconds // 1000)
        mode = OffboardControlMode()
        mode.timestamp, mode.position = now_us, True
        self.mode_pub.publish(mode)
        point = TrajectorySetpoint()
        point.timestamp = now_us
        point.position = position_ned.tolist()
        point.velocity = [math.nan] * 3
        point.acceleration = [math.nan] * 3
        point.jerk = [math.nan] * 3
        point.yaw, point.yawspeed = 0.0, math.nan
        self.setpoint_pub.publish(point)

    def _velocity_setpoint(self, override: np.ndarray | None = None) -> None:
        if self.rc_takeover_active:
            return
        if not self._is_offboard() and self.offboard_requested_s is None and not self.takeoff_active:
            return
        command = self.manual_command.copy() if override is None else np.asarray(override, dtype=float).copy()
        if override is None and self._now() - self.manual_command_stamp_s > float(self.config["manual_control"]["input_timeout_s"]):
            command[:] = 0.0
        forward, left, up, yaw_left = command
        heading = 0.0
        if self.local_position is not None and math.isfinite(float(self.local_position.heading)):
            heading = float(self.local_position.heading)
        # Body FLU -> map ENU. PX4 heading is NED yaw (clockwise from north).
        north = forward * math.cos(heading) + left * math.sin(heading)
        east = forward * math.sin(heading) - left * math.cos(heading)
        velocity_enu = np.array([east, north, up], dtype=float)
        now_us = int(self.get_clock().now().nanoseconds // 1000)
        mode = OffboardControlMode()
        mode.timestamp, mode.velocity = now_us, True
        self.mode_pub.publish(mode)
        point = TrajectorySetpoint()
        point.timestamp = now_us
        point.position = [math.nan] * 3
        point.velocity = enu_to_ned(velocity_enu).tolist()
        point.acceleration = [math.nan] * 3
        point.jerk = [math.nan] * 3
        point.yaw = math.nan
        point.yawspeed = -float(yaw_left)
        self.setpoint_pub.publish(point)

    def _acceleration_setpoint(self, raw: np.ndarray) -> None:
        if self.rc_takeover_active or not self._is_offboard():
            return
        command = np.asarray(raw, dtype=float)
        maximum = float(self.config["interceptor"]["max_acceleration_mps2"])
        norm = np.linalg.norm(command)
        if norm > maximum:
            command = command * maximum / norm
        if self.interceptor is not None:
            _, velocity = odom_vectors(self.interceptor)
            max_speed = float(self.config["interceptor"]["max_speed_mps"])
            if np.linalg.norm(velocity) > max_speed and np.dot(command, velocity) > 0.0:
                command = command - np.dot(command, velocity) * velocity / max(np.dot(velocity, velocity), 1e-9)
        now_us = int(self.get_clock().now().nanoseconds // 1000)
        mode = OffboardControlMode()
        mode.timestamp, mode.acceleration = now_us, True
        self.mode_pub.publish(mode)
        point = TrajectorySetpoint()
        point.timestamp = now_us
        point.position = [math.nan] * 3
        point.velocity = [math.nan] * 3
        point.acceleration = enu_to_ned(command).tolist()
        point.jerk = [math.nan] * 3
        point.yaw = point.yawspeed = math.nan
        self.setpoint_pub.publish(point)
        applied = AccelStamped()
        applied.header.stamp = self.get_clock().now().to_msg()
        applied.header.frame_id = "map"
        applied.accel.linear.x, applied.accel.linear.y, applied.accel.linear.z = map(float, command)
        self.applied_pub.publish(applied)

    def _takeoff_point(self) -> np.ndarray:
        point = np.asarray(self.config["interceptor"]["initial_position_enu_m"], dtype=float).copy()
        point[2] = float(self.config["interceptor"]["nominal_altitude_m"])
        return point

    def _tick(self) -> None:
        now = self._now()
        phase = self.machine.phase
        visual_only = bool(self.get_parameter("visualization_only").value)
        if phase == ExperimentPhase.PRECHECK:
            if visual_only:
                self.ready = {key: True for key in self.ready}
                self.machine.phase = ExperimentPhase.STABILIZE
                self.phase_enter_s = now - float(self.config["experiment"]["stabilize_s"])
            else:
                if self.takeoff_active and self.takeoff_position_map is not None:
                    takeoff_state = self.interceptor_ground_truth if self.interceptor_ground_truth is not None else self.interceptor
                    if takeoff_state is not None:
                        position, _ = odom_vectors(takeoff_state)
                        error = self.takeoff_position_map[2] - position[2]
                        if abs(error) <= 0.08:
                            self.takeoff_active = False
                            self.get_logger().info("keyboard takeoff altitude reached; zero-velocity hover active")
                            self._velocity_setpoint(np.zeros(4))
                        else:
                            limit = float(self.config["manual_control"]["vertical_speed_mps"])
                            up = float(np.clip(0.8 * error, -limit, limit))
                            self._velocity_setpoint(np.array([0.0, 0.0, up, 0.0]))
                    else:
                        self._velocity_setpoint(np.zeros(4))
                else:
                    self._velocity_setpoint()
                if self.offboard_requested_s is not None:
                    if self._is_offboard():
                        self.get_logger().info("PX4 entered OFFBOARD mode")
                        self.offboard_requested_s = None
                    elif not self.rc_takeover_active and now - self.offboard_requested_s >= float(self.config["manual_control"]["offboard_prestream_s"]) and now - self.last_offboard_command_s >= 1.0:
                        self._vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                        self.last_offboard_command_s = now
                if (
                    self.pending_scenario is not None
                    and self._is_armed()
                    and self._is_offboard()
                    and not self.rc_takeover_active
                    and all(self.ready.values())
                    and self.controller_scenario_ready_stage == self.pending_scenario["stage"]
                    and now - self.pending_scenario_s >= 0.25
                ):
                    self._prepare_scenario()
        elif phase == ExperimentPhase.TAKEOFF:
            self._position_setpoint(self._takeoff_point())
            self.prestream_count += 1
            if self.prestream_count == 100 and not self.rc_takeover_active:
                self._vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self._vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            if self.interceptor is not None and self.status is not None:
                position, _ = odom_vectors(self.interceptor)
                if int(self.status.arming_state) == VehicleStatus.ARMING_STATE_ARMED and abs(position[2] - self._takeoff_point()[2]) <= 0.5:
                    self._transition("altitude_reached")
        elif phase == ExperimentPhase.STABILIZE:
            if not visual_only:
                self._position_setpoint(self._takeoff_point())
            if now - self.phase_enter_s >= float(self.config["experiment"]["stabilize_s"]):
                self._transition("stable")
        elif phase == ExperimentPhase.RUN:
            if self.manual_override_latched or self.rc_takeover_active:
                self._transition("abort")
                if not self.rc_takeover_active:
                    self._velocity_setpoint()
            elif self.safety_abort:
                self._transition("abort")
            elif self.interceptor is not None and self.target is not None:
                separation = distance(odom_vectors(self.interceptor)[0], odom_vectors(self.target)[0])
                self.minimum_separation_m = min(self.minimum_separation_m, separation)
                if separation <= float(self.config["capture"]["radius_m"]):
                    self.capture_time_s = now - float(self.epoch_s or now)
                    self._transition("capture")
                elif not visual_only and self.command is not None:
                    self._acceleration_setpoint(self.command)
        elif phase in {ExperimentPhase.CAPTURE, ExperimentPhase.ABORT}:
            self._transition("settle")
        elif phase == ExperimentPhase.HOLD:
            if self.manual_override_latched or self.rc_takeover_active:
                if not self.rc_takeover_active:
                    self._velocity_setpoint()
            elif not visual_only:
                self._position_setpoint(self.hold_position_map if self.hold_position_map is not None else self._takeoff_point())
            if not self.manual_override_latched and not self.rc_takeover_active and now - self.phase_enter_s >= float(self.config["experiment"]["hold_after_capture_s"]):
                self._transition("land" if self.config["experiment"]["auto_land"] and not visual_only else "finish")
        elif phase == ExperimentPhase.LAND:
            if not self.rc_takeover_active and now - self.last_vehicle_command_s >= 1.0:
                self._vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.last_vehicle_command_s = now
            if self.status is not None and int(self.status.arming_state) == VehicleStatus.ARMING_STATE_DISARMED:
                self._transition("disarmed")
        self._publish_status()

    def _publish_status(self) -> None:
        sep = math.nan
        if self.interceptor is not None and self.target is not None:
            sep = distance(odom_vectors(self.interceptor)[0], odom_vectors(self.target)[0])
        if self.status is None:
            px4_state = "UNAVAILABLE"
        else:
            nav_state = int(self.status.nav_state)
            px4_state = NAV_STATE_NAMES.get(nav_state, str(nav_state))
        status = diagnostic(
            "ras_hw_mirror/experiment",
            DiagnosticStatus.ERROR if self.machine.phase == ExperimentPhase.ABORT else DiagnosticStatus.OK,
            self.machine.phase.value,
            phase=self.machine.phase.value,
            method=self.method,
            trajectory=self.trajectory,
            condition=self.condition,
            seed=self.seed,
            px4=px4_state,
            armed=int(self._is_armed()),
            manual_override=int(self.manual_override_latched),
            takeoff_active=int(self.takeoff_active),
            pending_stage="none" if self.pending_scenario is None else self.pending_scenario["stage"],
            separation_m=f"{sep:.6f}",
            minimum_separation_m=f"{self.minimum_separation_m:.6f}",
            captured=int(self.capture_time_s is not None),
            capture_time_s="" if self.capture_time_s is None else f"{self.capture_time_s:.6f}",
            safety_abort=int(self.safety_abort),
        )
        status.header.stamp = self.get_clock().now().to_msg()
        self.status_pub.publish(status)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ExperimentManagerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
