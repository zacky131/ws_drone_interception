"""Foreground SSH-terminal keyboard control for the hardware mirror."""

from __future__ import annotations

import errno
import json
import os
import termios
import time
import tty

import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import TwistStamped
from px4_msgs.msg import VehicleStatus
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from .config_utils import default_config, load_mirror_config
from .gazebo_state_provider_node import PX4_QOS
from .manifest_utils import default_manifest, load_campaign_scenarios
from .manual_control import MOTION_KEYS, SCENARIOS, decode_terminal_input, held_key_command, px4_fmu_prefix
from .ros_utils import diagnostic_values


HELP = """RAS X500 SSH terminal control (this terminal is now in cbreak mode)

q  arm only                   o  enter offboard mode
t  take off to 2 m            h  hold (loiter)       g  land
w / s                         up / down
a / d                         yaw left / yaw right
arrow up / down               forward / back
arrow left / right            move left / right

1  GS1                         2  GS2                    3  GS3
4  GS4                         5  GS5 M0'                6  GS5 M1
7  GS6 safety checks
8  selected hardware_mirror_24 manifest row (set manifest_row:=1..24)

Hold a movement key so the SSH terminal sends key-repeat characters.
When repeat input stops, the velocity watchdog commands zero automatically.
Scenario keys are accepted only while PX4 reports ARMED.
Any movement key during autonomous RUN cancels it and takes manual control.
Press Ctrl-C to command zero, restore the terminal, and exit.
"""


class TerminalKeyboard:
    """Nonblocking reader for the process's controlling SSH/local terminal."""

    def __init__(self, path: str = "/dev/tty") -> None:
        try:
            self.fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
        except OSError as exc:
            raise RuntimeError(
                "keyboard_control needs an interactive terminal; connect with "
                "`ssh -t ...` and run it in its own foreground terminal"
            ) from exc
        try:
            self.original_attributes = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd, termios.TCSANOW)
        except Exception:
            os.close(self.fd)
            raise
        self.pending = b""
        self.closed = False

    def write(self, text: str) -> None:
        if not self.closed:
            os.write(self.fd, text.encode("utf-8", errors="replace"))

    def read_keys(self) -> list[str]:
        chunks: list[bytes] = []
        while True:
            try:
                chunk = os.read(self.fd, 256)
            except BlockingIOError:
                break
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    break
                raise
            if not chunk:
                break
            chunks.append(chunk)
        if not chunks:
            return []
        keys, self.pending = decode_terminal_input(b"".join(chunks), self.pending)
        return keys

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original_attributes)
        finally:
            os.close(self.fd)


class KeyboardControlNode(Node):
    def __init__(self, terminal: TerminalKeyboard) -> None:
        super().__init__("keyboard_control")
        self.declare_parameter("config", str(default_config()))
        self.declare_parameter("manifest", str(default_manifest()))
        self.declare_parameter("manifest_row", 0)
        self.config = load_mirror_config(self.get_parameter("config").value)
        self.manifest_path = str(self.get_parameter("manifest").value)
        self.manifest_row = int(self.get_parameter("manifest_row").value)
        self.manifest_scenario = None
        if self.manifest_row:
            scenarios = load_campaign_scenarios(self.manifest_path)
            if self.manifest_row not in scenarios:
                raise ValueError(f"manifest_row must be 1..24, got {self.manifest_row}")
            self.manifest_scenario = scenarios[self.manifest_row]
        manual = self.config["manual_control"]
        self.horizontal = float(manual["horizontal_speed_mps"])
        self.vertical = float(manual["vertical_speed_mps"])
        self.yaw_rate = float(manual["yaw_rate_rad_s"])
        self.repeat_timeout_s = float(manual.get("terminal_repeat_timeout_s", 0.30))
        self.action_debounce_s = float(manual.get("terminal_action_debounce_s", 0.70))
        self.terminal = terminal
        self.motion_seen_s: dict[str, float] = {}
        self.action_seen_s: dict[str, float] = {}
        self.held: set[str] = set()
        self.armed = False
        self.phase = "PRECHECK"
        prefix = px4_fmu_prefix(self.config["interceptor"]["px4_namespace"])
        self.velocity_pub = self.create_publisher(TwistStamped, "/ras_hw_mirror/manual/velocity", 10)
        self.action_pub = self.create_publisher(String, "/ras_hw_mirror/manual/action", 10)
        self.scenario_pub = self.create_publisher(String, "/ras_hw_mirror/scenario/request", 10)
        self.create_subscription(VehicleStatus, f"{prefix}/out/vehicle_status", self._vehicle_status, PX4_QOS)
        self.create_subscription(DiagnosticArray, "/ras_hw_mirror/experiment/status", self._experiment_status, 10)
        self.create_timer(0.01, self._poll_terminal)
        self.create_timer(0.05, self._publish_velocity)
        self.terminal.write(HELP + "\n")
        if self.manifest_scenario is None:
            self.terminal.write("Manifest key 8 disabled: relaunch with manifest_row:=1..24.\n\n")
        else:
            selected = self.manifest_scenario
            self.terminal.write(
                f"SELECTED ROW {self.manifest_row:02d}/24: {selected['run_id']} "
                f"seed={selected['seed']}. After ARMED, press 8 exactly once.\n\n"
            )
        self.get_logger().info("terminal keyboard ready; waiting for PX4 status")

    def _report(self, text: str, warning: bool = False) -> None:
        (self.get_logger().warning if warning else self.get_logger().info)(text)

    def _vehicle_status(self, msg: VehicleStatus) -> None:
        previous = self.armed
        self.armed = int(msg.arming_state) == VehicleStatus.ARMING_STATE_ARMED
        if self.armed and not previous:
            self._report("PX4 ARMED — manual controls and stage keys are enabled")
        elif previous and not self.armed:
            self._report("PX4 DISARMED — press q to arm")

    def _experiment_status(self, msg: DiagnosticArray) -> None:
        values = diagnostic_values(msg)
        self.phase = values.get("phase", self.phase)

    def _poll_terminal(self) -> None:
        now = time.monotonic()
        for key in self.terminal.read_keys():
            self._handle_key(key, now)

        previous = self.held
        self.held = {
            key for key, seen_s in self.motion_seen_s.items()
            if now - seen_s <= self.repeat_timeout_s
        }
        self.motion_seen_s = {key: seen_s for key, seen_s in self.motion_seen_s.items() if key in self.held}
        if previous and not self.held:
            self._publish_velocity()
            self._report("manual velocity zero — key repeat stopped")

    def _handle_key(self, key: str, now: float) -> None:
        if key in MOTION_KEYS:
            newly_active = key not in self.held
            self.motion_seen_s[key] = now
            self.held.add(key)
            if newly_active:
                self._report(f"MANUAL velocity active: {key}")
            return

        if key not in {"q", "o", "t", "h", "g", "8"} and key not in SCENARIOS:
            return
        previous_s = self.action_seen_s.get(key)
        self.action_seen_s[key] = now
        if previous_s is not None and now - previous_s < self.action_debounce_s:
            return

        if key in {"q", "o", "t", "h", "g"}:
            action = {"q": "ARM", "o": "OFFBOARD", "t": "TAKEOFF", "h": "HOLD", "g": "LAND"}[key]
            msg = String()
            msg.data = action
            self.action_pub.publish(msg)
            self._report(f"requested {action}")
            return

        if key == "8":
            if self.manifest_scenario is None:
                self._report("REJECTED manifest key 8: relaunch keyboard_control with manifest_row:=1..24", warning=True)
                return
            stage = self.manifest_scenario
        else:
            stage = SCENARIOS[key]
        if not self.armed:
            self._report(f"REJECTED {stage['stage']}: press q and wait for ARMED first", warning=True)
            return
        msg = String()
        msg.data = json.dumps(stage, separators=(",", ":"))
        self.scenario_pub.publish(msg)
        self._report(
            f"requested {stage['stage']} ({stage['method']} / {stage['trajectory']} / "
            f"{stage['condition']} / seed {stage['seed']})"
        )

    def command_zero(self) -> None:
        self.motion_seen_s.clear()
        self.held.clear()
        self._publish_velocity()

    def _publish_velocity(self) -> None:
        command = held_key_command(self.held, self.horizontal, self.vertical, self.yaw_rate)
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "interceptor_body_flu"
        msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z, msg.twist.angular.z = map(float, command)
        self.velocity_pub.publish(msg)


def main(args=None) -> None:
    terminal: TerminalKeyboard | None = None
    node: KeyboardControlNode | None = None
    terminal_error = False
    try:
        terminal = TerminalKeyboard()
        rclpy.init(args=args)
        node = KeyboardControlNode(terminal)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as exc:
        print(f"keyboard_control: {exc}", flush=True)
        terminal_error = True
    finally:
        try:
            if node is not None:
                if rclpy.ok():
                    node.command_zero()
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        finally:
            # Restoring the SSH terminal must not depend on ROS cleanup.
            if terminal is not None:
                terminal.close()
    if terminal_error:
        raise SystemExit(2)
