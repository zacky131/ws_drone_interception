"""Non-CSV, four-panel live ROS dashboard with bounded histories."""

from __future__ import annotations

from collections import deque
import math
import signal
import time
import numpy as np
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from diagnostic_msgs.msg import DiagnosticArray
from nav_msgs.msg import Odometry, Path

from .config_utils import default_config, default_field, load_mirror_config, load_yaml, package_file
from .geometry_utils import rectangle_points
from .ros_utils import diagnostic_values, odom_vectors


class DashboardNode(Node):
    def __init__(self) -> None:
        super().__init__("live_dashboard")
        self.declare_parameter("config", str(default_config()))
        self.declare_parameter("field", str(default_field()))
        self.config = load_mirror_config(self.get_parameter("config").value)
        self.field = load_yaml(self.get_parameter("field").value)
        dash = load_yaml(package_file("config/dashboard.yaml"))
        size = int(dash["history_samples"])
        self.colors = dash["colors"]
        self.target_xy = deque(maxlen=size)
        self.interceptor_xy = deque(maxlen=size)
        self.time_s = deque(maxlen=size)
        self.separation = deque(maxlen=size)
        self.error = deque(maxlen=size)
        self.age_ms = deque(maxlen=size)
        self.requested_ms = deque(maxlen=size)
        self.age_time_s = deque(maxlen=size)
        self.start_s = self._now()
        self.target = self.interceptor = self.estimate = None
        self.prediction = Path()
        self.experiment = {"method": "?", "trajectory": "?", "condition": "?", "phase": "PRECHECK", "px4": "?", "captured": "0", "minimum_separation_m": "nan", "capture_time_s": ""}
        self.safety = {"message": "OK", "reason": "none"}
        self.telemetry = {"actual_age_ms": "nan", "requested_age_ms": "nan", "dropped": "0"}
        self.create_subscription(Odometry, "/ras_hw_mirror/target/truth", self._target, 10)
        self.create_subscription(Odometry, "/ras_hw_mirror/interceptor/state/px4", self._interceptor, 10)
        self.create_subscription(Odometry, "/ras_hw_mirror/target/estimate", lambda msg: setattr(self, "estimate", msg), 10)
        self.create_subscription(Path, "/ras_hw_mirror/target/prediction_path", lambda msg: setattr(self, "prediction", msg), 10)
        self.create_subscription(DiagnosticArray, "/ras_hw_mirror/experiment/status", self._experiment, 10)
        self.create_subscription(DiagnosticArray, "/ras_hw_mirror/safety/status", self._safety, 10)
        self.create_subscription(DiagnosticArray, "/ras_hw_mirror/telemetry/status", self._telemetry, 50)
        # Constrained layout is intentionally avoided here.  Recomputing it while
        # four axes are cleared on every live frame can saturate the Qt/X11 event
        # loop and make unrelated desktop windows feel unresponsive.
        self.fig, axes = plt.subplots(2, 2, figsize=tuple(dash["figure_size_inches"]))
        self.ax_track, self.ax_sep, self.ax_err, self.ax_age = axes.ravel()
        self.fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.86, wspace=0.27, hspace=0.34)
        self.fig.canvas.manager.set_window_title("RAS one-X500 hardware mirror")

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _target(self, msg: Odometry) -> None:
        self.target = msg
        p, _ = odom_vectors(msg)
        self.target_xy.append(p[:2])
        self._sample()

    def _interceptor(self, msg: Odometry) -> None:
        self.interceptor = msg
        self.interceptor_xy.append(odom_vectors(msg)[0][:2])

    def _sample(self) -> None:
        if self.target is None or self.interceptor is None:
            return
        target = odom_vectors(self.target)[0]
        interceptor = odom_vectors(self.interceptor)[0]
        self.time_s.append(self._now() - self.start_s)
        self.separation.append(float(np.linalg.norm(target - interceptor)))
        if self.estimate is None:
            self.error.append(math.nan)
        else:
            self.error.append(float(np.linalg.norm(odom_vectors(self.estimate)[0] - target)))

    def _experiment(self, msg: DiagnosticArray) -> None:
        self.experiment.update(diagnostic_values(msg))

    def _safety(self, msg: DiagnosticArray) -> None:
        self.safety.update(diagnostic_values(msg))
        if msg.status:
            self.safety["message"] = msg.status[0].message

    def _telemetry(self, msg: DiagnosticArray) -> None:
        self.telemetry.update(diagnostic_values(msg))
        try:
            self.age_time_s.append(self._now() - self.start_s)
            self.age_ms.append(float(self.telemetry["actual_age_ms"]))
            self.requested_ms.append(float(self.telemetry["requested_age_ms"]))
        except (KeyError, ValueError):
            pass

    def redraw(self) -> None:
        for axis in (self.ax_track, self.ax_sep, self.ax_err, self.ax_age):
            axis.clear()
            axis.grid(alpha=0.22)
        if self.target_xy:
            xy = np.asarray(self.target_xy)
            self.ax_track.plot(xy[:, 0], xy[:, 1], ":", color=self.colors["truth"], label="target truth")
            self.ax_track.scatter(*xy[-1], marker="o", color=self.colors["truth"], s=45)
        if self.interceptor_xy:
            xy = np.asarray(self.interceptor_xy)
            self.ax_track.plot(xy[:, 0], xy[:, 1], "-", color=self.colors["interceptor"], label="X500")
            self.ax_track.scatter(*xy[-1], marker="^", color=self.colors["interceptor"], s=55)
        if self.estimate is not None:
            p = odom_vectors(self.estimate)[0]
            self.ax_track.scatter(p[0], p[1], marker="s", color=self.colors["estimate"], label="estimate")
        if self.prediction.poses:
            prediction = np.array([[p.pose.position.x, p.pose.position.y] for p in self.prediction.poses])
            self.ax_track.plot(prediction[:, 0], prediction[:, 1], ".-", color=self.colors["prediction"], label="prediction")
        for bounds, style, label in ((self.field["hard_geofence"], "r-", "hard geofence"), (self.field["target_region"], "--", "target region")):
            box = rectangle_points(bounds)
            self.ax_track.plot(box[:, 0], box[:, 1], style, lw=1.0, label=label)
        if self.target is not None:
            p = odom_vectors(self.target)[0]
            circle = plt.Circle(p[:2], float(self.config["capture"]["radius_m"]), fill=False, color=self.colors["capture"], ls="-.")
            self.ax_track.add_patch(circle)
        self.ax_track.set(title="A  Top-down ENU tracking", xlabel="East (m)", ylabel="North (m)", aspect="equal")
        self.ax_track.legend(fontsize=7, frameon=False, ncol=2)
        self.ax_sep.plot(self.time_s, self.separation, color=self.colors["interceptor"])
        self.ax_sep.axhline(float(self.config["capture"]["radius_m"]), color=self.colors["capture"], ls="-.", label="capture radius")
        self.ax_sep.set(title="B  Virtual separation", xlabel="Time (s)", ylabel="Separation (m)", ylim=(0, None))
        self.ax_sep.legend(frameon=False)
        self.ax_err.plot(self.time_s, self.error, color=self.colors["estimate"])
        self.ax_err.set(title="C  Current-time target error", xlabel="Time (s)", ylabel="Position error (m)", ylim=(0, None))
        self.ax_age.plot(self.age_time_s, self.requested_ms, "--", color="#666666", label="requested")
        self.ax_age.plot(self.age_time_s, self.age_ms, color=self.colors["prediction"], label="actual")
        self.ax_age.set(title="D  Telemetry age", xlabel="Time (s)", ylabel="Age (ms)", ylim=(0, None))
        self.ax_age.legend(frameon=False)
        sep = self.separation[-1] if self.separation else math.nan
        err = self.error[-1] if self.error else math.nan
        age = self.age_ms[-1] if self.age_ms else math.nan
        capture = "YES" if self.experiment.get("captured") == "1" else "NO"
        safety = self.safety.get("message", "?")
        header = f"Method: {self.experiment.get('method')}   Trajectory: {self.experiment.get('trajectory')}   Condition: {self.experiment.get('condition')}   State: {self.experiment.get('phase')}   PX4: {self.experiment.get('px4')}\nPacket age: {age:.1f} ms   Virtual separation: {sep:.2f} m   Target error: {err:.2f} m   Capture: {capture}   Safety: {safety}"
        if capture == "YES":
            header += f"   CAPTURED t={self.experiment.get('capture_time_s')} s, min={self.experiment.get('minimum_separation_m')} m"
        if safety == "ABORT":
            header += f"   ABORT — {self.safety.get('reason')}"
        self.fig.suptitle(header, fontsize=10)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


def main(args=None) -> None:
    rclpy.init(args=args)
    # Interactive mode must be enabled before the figure is constructed.  The
    # dashboard remains a foreground process, but its Qt window is non-modal and
    # does not own the terminal/desktop input loop.
    plt.rcParams["figure.raise_window"] = False
    plt.ion()
    node = DashboardNode()
    plt.show(block=False)
    period = 1.0 / float(node.config["visualization"]["dashboard_rate_hz"])
    next_draw = time.monotonic()
    stop_requested = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        while rclpy.ok() and not stop_requested and plt.fignum_exists(node.fig.number):
            rclpy.spin_once(node, timeout_sec=0.01)
            if time.monotonic() >= next_draw:
                node.redraw()
                next_draw = time.monotonic() + period
            # plt.pause() calls pyplot.show() every pass; with Qt that may raise
            # the window continuously and steal clicks from Gazebo/RViz/terminal
            # windows.  Processing canvas events directly avoids that behavior.
            node.fig.canvas.flush_events()
    except KeyboardInterrupt:
        pass
    finally:
        plt.close(node.fig)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
