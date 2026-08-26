"""Bounded RViz markers and minimal map-relative TF tree."""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import TransformStamped, Point, Pose
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry, Path
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from .config_utils import default_config, default_field, load_mirror_config, load_yaml
from .geometry_utils import rectangle_points
from .ros_utils import diagnostic_values, odom_vectors, point


class VisualizationNode(Node):
    def __init__(self) -> None:
        super().__init__("visualization")
        self.declare_parameter("config", str(default_config()))
        self.declare_parameter("field", str(default_field()))
        self.config = load_mirror_config(self.get_parameter("config").value)
        self.field = load_yaml(self.get_parameter("field").value)
        self.interceptor = self.target = self.estimate = None
        self.prediction = Path()
        self.status = {"phase": "PRECHECK", "method": "?", "condition": "?", "separation_m": "nan"}
        self.publisher = self.create_publisher(MarkerArray, "/ras_hw_mirror/visualization/markers", 10)
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Odometry, "/ras_hw_mirror/interceptor/state/px4", lambda msg: setattr(self, "interceptor", msg), 10)
        self.create_subscription(Odometry, "/ras_hw_mirror/target/truth", lambda msg: setattr(self, "target", msg), 10)
        self.create_subscription(Odometry, "/ras_hw_mirror/target/estimate", lambda msg: setattr(self, "estimate", msg), 10)
        self.create_subscription(Path, "/ras_hw_mirror/target/prediction_path", lambda msg: setattr(self, "prediction", msg), 10)
        self.create_subscription(DiagnosticArray, "/ras_hw_mirror/experiment/status", self._status, 10)
        self.create_timer(1.0 / float(self.config["visualization"]["marker_rate_hz"]), self._tick)

    def _status(self, msg: DiagnosticArray) -> None:
        self.status.update(diagnostic_values(msg))

    def _base(self, index: int, marker_type: int, namespace: str) -> Marker:
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns, marker.id, marker.type, marker.action = namespace, index, marker_type, Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    @staticmethod
    def _color(marker: Marker, rgba) -> None:
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = map(float, rgba)

    def _quadrotor_markers(self, start_id: int, pose: Pose, namespace: str, color_body: tuple, color_rotors: tuple) -> list[Marker]:
        markers = []
        p = np.array([pose.position.x, pose.position.y, pose.position.z])
        q_raw = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        if np.isclose(np.linalg.norm(q_raw), 0):
            q_raw = [0.0, 0.0, 0.0, 1.0]
        rot = R.from_quat(q_raw)

        # 1. Central Body Hub
        hub = self._base(start_id, Marker.CUBE, namespace)
        hub.pose = pose
        hub.scale.x, hub.scale.y, hub.scale.z = 0.28, 0.28, 0.12
        self._color(hub, color_body)
        markers.append(hub)

        # 2 & 3. Cross Arms (X-Frame)
        q_arm1 = (rot * R.from_euler('z', 45, degrees=True)).as_quat()
        arm1 = self._base(start_id + 1, Marker.CUBE, namespace)
        arm1.pose.position = pose.position
        arm1.pose.orientation.x, arm1.pose.orientation.y, arm1.pose.orientation.z, arm1.pose.orientation.w = q_arm1
        arm1.scale.x, arm1.scale.y, arm1.scale.z = 0.85, 0.05, 0.04
        self._color(arm1, (0.12, 0.12, 0.15, color_body[3]))
        markers.append(arm1)

        q_arm2 = (rot * R.from_euler('z', -45, degrees=True)).as_quat()
        arm2 = self._base(start_id + 2, Marker.CUBE, namespace)
        arm2.pose.position = pose.position
        arm2.pose.orientation.x, arm2.pose.orientation.y, arm2.pose.orientation.z, arm2.pose.orientation.w = q_arm2
        arm2.scale.x, arm2.scale.y, arm2.scale.z = 0.85, 0.05, 0.04
        self._color(arm2, (0.12, 0.12, 0.15, color_body[3]))
        markers.append(arm2)

        # 4-7. Four Rotor Disks
        local_rotors = [
            np.array([+0.30, -0.30, +0.04]),
            np.array([+0.30, +0.30, +0.04]),
            np.array([-0.30, -0.30, +0.04]),
            np.array([-0.30, +0.30, +0.04]),
        ]
        for i, loc in enumerate(local_rotors):
            world_pos = p + rot.apply(loc)
            rotor = self._base(start_id + 3 + i, Marker.CYLINDER, namespace)
            rotor.pose.position.x, rotor.pose.position.y, rotor.pose.position.z = world_pos
            rotor.pose.orientation = pose.orientation
            rotor.scale.x = rotor.scale.y = 0.32
            rotor.scale.z = 0.025
            self._color(rotor, color_rotors)
            markers.append(rotor)

        # 8. Front Heading Nose / Arrow Pointer (Bright Yellow)
        nose_loc = np.array([+0.38, 0.0, 0.0])
        nose_pos = p + rot.apply(nose_loc)
        nose = self._base(start_id + 7, Marker.SPHERE, namespace)
        nose.pose.position.x, nose.pose.position.y, nose.pose.position.z = nose_pos
        nose.pose.orientation = pose.orientation
        nose.scale.x = nose.scale.y = nose.scale.z = 0.18
        self._color(nose, (1.0, 0.85, 0.0, color_body[3]))
        markers.append(nose)

        # 9. Altitude stem / Ground drop line
        line = self._base(start_id + 8, Marker.LINE_STRIP, namespace)
        line.scale.x = 0.04
        self._color(line, (color_body[0], color_body[1], color_body[2], 0.35))
        line.points = [Point(x=p[0], y=p[1], z=0.0), Point(x=p[0], y=p[1], z=p[2])]
        markers.append(line)

        return markers

    def _rectangle(self, index: int, bounds: dict, name: str, color, width: float) -> Marker:
        marker = self._base(index, Marker.LINE_STRIP, name)
        marker.scale.x = width
        self._color(marker, color)
        marker.points = [point(row) for row in rectangle_points(bounds)]
        return marker

    def _broadcast(self, child: str, msg: Odometry | None) -> None:
        if msg is None:
            return
        transform = TransformStamped()
        transform.header.frame_id = "map"
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.child_frame_id = child
        transform.transform.translation.x = msg.pose.pose.position.x
        transform.transform.translation.y = msg.pose.pose.position.y
        transform.transform.translation.z = msg.pose.pose.position.z
        transform.transform.rotation = msg.pose.pose.orientation
        self.tf.sendTransform(transform)

    def _tick(self) -> None:
        markers = MarkerArray()

        # 1. Pursuer / Interceptor UAV Quadrotor (Bright Electric Blue & Cyan Rotors)
        if self.interceptor is not None:
            quad_markers = self._quadrotor_markers(
                start_id=100,
                pose=self.interceptor.pose.pose,
                namespace="interceptor_uav",
                color_body=(0.0, 0.45, 0.95, 1.0),
                color_rotors=(0.0, 0.85, 1.0, 0.75)
            )
            markers.markers.extend(quad_markers)

        # 2. Target Truth UAV Quadrotor (Vivid Red & Orange Rotors) + Capture Sphere
        if self.target is not None:
            target_markers = self._quadrotor_markers(
                start_id=200,
                pose=self.target.pose.pose,
                namespace="target_uav",
                color_body=(1.0, 0.20, 0.0, 1.0),
                color_rotors=(1.0, 0.55, 0.0, 0.75)
            )
            markers.markers.extend(target_markers)

            capture = self._base(2, Marker.SPHERE, "capture_radius")
            capture.pose = self.target.pose.pose
            diameter = 2.0 * float(self.config["capture"]["radius_m"])
            capture.scale.x = capture.scale.y = capture.scale.z = diameter
            self._color(capture, (1.0, 0.7, 0.0, 0.16))
            markers.markers.append(capture)

        # 3. Target Estimate UAV Quadrotor (Translucent Violet/Purple)
        if self.estimate is not None:
            est_markers = self._quadrotor_markers(
                start_id=300,
                pose=self.estimate.pose.pose,
                namespace="estimate_uav",
                color_body=(0.65, 0.15, 0.95, 0.8),
                color_rotors=(0.85, 0.35, 0.95, 0.55)
            )
            markers.markers.extend(est_markers)

        prediction = self._base(4, Marker.LINE_STRIP, "prediction")
        prediction.scale.x = 0.09
        self._color(prediction, (0.0, 0.62, 0.38, 1.0))
        prediction.points = [pose.pose.position for pose in self.prediction.poses]
        markers.markers.append(prediction)
        points = self._base(5, Marker.POINTS, "prediction_points")
        points.scale.x = points.scale.y = 0.16
        self._color(points, (0.0, 0.62, 0.38, 1.0))
        points.points = prediction.points
        markers.markers.append(points)
        markers.markers.append(self._rectangle(6, self.field["hard_geofence"], "hard_geofence", (0.85, 0.0, 0.0, 1.0), 0.16))
        markers.markers.append(self._rectangle(7, self.field["target_region"], "target_region", (0.95, 0.65, 0.0, 1.0), 0.10))
        initial = self._base(8, Marker.CYLINDER, "initial_interceptor")
        initial.pose.position = point(self.config["interceptor"]["initial_position_enu_m"])
        initial.scale.x = initial.scale.y = 0.8
        initial.scale.z = 0.08
        self._color(initial, (0.0, 0.45, 0.75, 0.7))
        markers.markers.append(initial)
        text = self._base(9, Marker.TEXT_VIEW_FACING, "status")
        text.pose.position.x, text.pose.position.y, text.pose.position.z = -38.0, 20.0, 8.5
        text.scale.z = 1.3
        self._color(text, (1.0, 1.0, 1.0, 1.0))
        text.text = f"{self.status.get('phase')} | {self.status.get('method')} | {self.status.get('condition')} | r={self.status.get('separation_m')} m"
        markers.markers.append(text)
        self.publisher.publish(markers)
        self._broadcast("interceptor_base", self.interceptor)
        self._broadcast("virtual_target", self.target)
        self._broadcast("target_estimate", self.estimate)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisualizationNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
