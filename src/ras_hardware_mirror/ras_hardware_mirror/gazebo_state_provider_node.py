"""Gazebo/PX4 transport adapter publishing the common map/ENU interceptor state."""

from __future__ import annotations

from collections import deque
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy._rclpy_pybind11 import RCLError
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from px4_msgs.msg import VehicleLocalPosition
from std_msgs.msg import Bool
from gz.transport13 import Node as GazeboTransportNode
from gz.msgs10.pose_v_pb2 import Pose_V

from .config_utils import default_config, load_mirror_config
from .geometry_utils import ned_to_enu
from .manual_control import px4_fmu_prefix
from .ros_utils import odometry


PX4_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST, depth=1)


class GazeboStateProviderNode(Node):
    def __init__(self) -> None:
        super().__init__("gazebo_state_provider")
        self.declare_parameter("config", str(default_config()))
        self.declare_parameter("gazebo_pose_topic", "/world/default/dynamic_pose/info")
        self.config = load_mirror_config(self.get_parameter("config").value)
        interceptor = self.config["interceptor"]
        self.map_origin = np.asarray(interceptor["initial_position_enu_m"], dtype=float)
        prefix = f"{px4_fmu_prefix(interceptor['px4_namespace'])}/out"
        self.px4_pub = self.create_publisher(Odometry, "/ras_hw_mirror/interceptor/state/px4", 10)
        self.truth_pub = self.create_publisher(Odometry, "/ras_hw_mirror/interceptor/state/ground_truth", 10)
        self.path_pub = self.create_publisher(Path, "/ras_hw_mirror/interceptor/path", 10)
        self.ready_pub = self.create_publisher(Bool, "/ras_hw_mirror/ready/state_provider", 1)
        self.path = deque(maxlen=int(self.config["experiment"]["path_history"]))
        self.last_gazebo_position: np.ndarray | None = None
        self.last_gazebo_time_s: float | None = None
        self.px4_reference_ned: np.ndarray | None = None
        self.create_subscription(VehicleLocalPosition, f"{prefix}/vehicle_local_position", self._px4, PX4_QOS)
        self.gz_topic = str(self.get_parameter("gazebo_pose_topic").value)
        self.gz_transport = GazeboTransportNode()
        self.gz_transport.subscribe(Pose_V, self.gz_topic, self._gazebo_pose)

    def _px4(self, msg: VehicleLocalPosition) -> None:
        if not rclpy.ok():
            return
        raw_position_ned = np.array([msg.x, msg.y, msg.z], dtype=float)
        if self.px4_reference_ned is None and bool(msg.xy_valid and msg.z_valid):
            self.px4_reference_ned = raw_position_ned.copy()
        reference = np.zeros(3) if self.px4_reference_ned is None else self.px4_reference_ned
        position = self.map_origin + ned_to_enu(raw_position_ned - reference)
        velocity = ned_to_enu([msg.vx, msg.vy, msg.vz])
        valid = bool(msg.xy_valid and msg.z_valid and msg.v_xy_valid and msg.v_z_valid)
        out = odometry("map", "interceptor_base", self.get_clock().now().nanoseconds * 1e-9, position, velocity)
        acceleration = ned_to_enu([msg.ax, msg.ay, msg.az])
        out.twist.twist.angular.x, out.twist.twist.angular.y, out.twist.twist.angular.z = map(float, acceleration)
        if not valid:
            out.pose.covariance[0] = float("inf")
        try:
            self.px4_pub.publish(out)
            pose = PoseStamped()
            pose.header = out.header
            pose.pose = out.pose.pose
            self.path.append(pose)
            path = Path()
            path.header = out.header
            path.poses = list(self.path)
            self.path_pub.publish(path)
            ready = Bool()
            ready.data = valid
            self.ready_pub.publish(ready)
        except RCLError:
            if rclpy.ok():
                raise

    def _gazebo_pose(self, msg: Pose_V) -> None:
        if not rclpy.ok():
            return
        model = str(self.config["interceptor"]["gazebo_model_name"])
        selected = next((value for value in msg.pose if value.name == model), None)
        if selected is None:
            return
        translation = selected.position
        # The standard PX4 world spawns at Gazebo (0, 0, 0). Translate that
        # local world pose into the configured common map/ENU origin.
        position = self.map_origin + np.array([translation.x, translation.y, translation.z], dtype=float)
        now = self.get_clock().now().nanoseconds * 1e-9
        velocity = np.zeros(3)
        if self.last_gazebo_position is not None and self.last_gazebo_time_s is not None:
            dt = now - self.last_gazebo_time_s
            if dt > 1e-4:
                velocity = (position - self.last_gazebo_position) / dt
        self.last_gazebo_position, self.last_gazebo_time_s = position, now
        out = odometry("map", "interceptor_base_ground_truth", now, position, velocity)
        out.pose.pose.orientation.x = selected.orientation.x
        out.pose.pose.orientation.y = selected.orientation.y
        out.pose.pose.orientation.z = selected.orientation.z
        out.pose.pose.orientation.w = selected.orientation.w
        try:
            self.truth_pub.publish(out)
        except RCLError:
            if rclpy.ok():
                raise

    def destroy_node(self):
        self.gz_transport.unsubscribe(self.gz_topic)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GazeboStateProviderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
