"""Deterministic software target and passive Gazebo visual-marker updater."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from gz.transport13 import Node as GazeboTransportNode
from gz.msgs10.pose_pb2 import Pose as GazeboPose
from gz.msgs10.boolean_pb2 import Boolean as GazeboBoolean
from gz.msgs10.entity_factory_pb2 import EntityFactory
from std_msgs.msg import Bool, String
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path

from .config_utils import default_config, load_mirror_config
from .ros_utils import odometry
from .trajectory_library import TRAJECTORIES, evaluate_trajectory


TRUTH_TOPIC = "/ras_hw_mirror/target/truth"


class VirtualTargetNode(Node):
    def __init__(self) -> None:
        super().__init__("virtual_target")
        self.declare_parameter("config", str(default_config()))
        self.declare_parameter("trajectory", "HT1")
        self.declare_parameter("autostart", False)
        self.declare_parameter("gazebo_world", "default")
        self.declare_parameter("gazebo_entity", "virtual_target_marker")
        self.config = load_mirror_config(self.get_parameter("config").value)
        self.trajectory = str(self.get_parameter("trajectory").value)
        if self.trajectory not in TRAJECTORIES:
            raise ValueError(f"invalid trajectory {self.trajectory}")
        self.running = bool(self.get_parameter("autostart").value)
        self.epoch_s = self._now()
        self.paths = deque(maxlen=int(self.config["experiment"]["path_history"]))
        self.truth_pub = self.create_publisher(Odometry, TRUTH_TOPIC, 10)
        self.path_pub = self.create_publisher(Path, "/ras_hw_mirror/target/truth_path", 10)
        self.ready_pub = self.create_publisher(Bool, "/ras_hw_mirror/ready/virtual_target", 1)
        self.create_subscription(String, "/ras_hw_mirror/experiment/phase", self._phase, 10)
        self.create_subscription(String, "/ras_hw_mirror/scenario/selection", self._scenario, 10)
        self.pose_service = f"/world/{self.get_parameter('gazebo_world').value}/set_pose"
        self.create_service_name = f"/world/{self.get_parameter('gazebo_world').value}/create"
        self.gz_transport = GazeboTransportNode()
        self.pose_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="virtual-target-gz-pose")
        self.pose_pending = None
        self.marker_spawned = False
        self.map_origin = __import__("numpy").asarray(self.config["interceptor"]["initial_position_enu_m"], dtype=float)
        rate = float(self.config["experiment"]["truth_rate_hz"])
        self.timer = self.create_timer(1.0 / rate, self._tick)
        self.pose_divisor = max(1, round(rate / float(self.config["visualization"]["gazebo_pose_rate_hz"])))
        self.tick_count = 0

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _phase(self, msg: String) -> None:
        if msg.data == "RUN" and not self.running:
            self.running = True
            self.epoch_s = self._now()
            self.paths.clear()
        elif msg.data in {"CAPTURE", "HOLD", "ABORT", "LAND", "DONE"}:
            self.running = False

    def _scenario(self, msg: String) -> None:
        try:
            request = json.loads(msg.data)
            trajectory = str(request["trajectory"])
        except (KeyError, TypeError, json.JSONDecodeError):
            self.get_logger().warning("ignored malformed scenario selection")
            return
        if trajectory not in TRAJECTORIES:
            self.get_logger().warning(f"ignored unsupported trajectory {trajectory!r}")
            return
        self.trajectory = trajectory
        self.running = False
        self.paths.clear()

    @staticmethod
    def _marker_sdf() -> str:
        return """<sdf version='1.9'><model name='virtual_target_marker'><static>true</static>
<link name='marker_link'><visual name='marker_visual'><geometry><sphere><radius>0.45</radius></sphere></geometry>
<material><ambient>1 0.35 0 1</ambient><diffuse>1 0.35 0 1</diffuse><emissive>0.35 0.08 0 1</emissive></material>
</visual></link></model></sdf>"""

    def _gazebo_update(self, pose: GazeboPose):
        if not self.marker_spawned:
            factory = EntityFactory()
            factory.name = str(self.get_parameter("gazebo_entity").value)
            factory.sdf = self._marker_sdf()
            factory.pose.CopyFrom(pose)
            executed, response = self.gz_transport.request(
                self.create_service_name, factory, EntityFactory, GazeboBoolean, 500
            )
            self.marker_spawned = bool(executed and response.data)
            if self.marker_spawned:
                return executed, response
        return self.gz_transport.request(
            self.pose_service, pose, GazeboPose, GazeboBoolean, 50
        )

    def _move_gazebo_marker(self, truth: Odometry) -> None:
        if self.pose_pending is not None and not self.pose_pending.done():
            return
        pose = GazeboPose()
        pose.name = str(self.get_parameter("gazebo_entity").value)
        pose.position.x = truth.pose.pose.position.x - self.map_origin[0]
        pose.position.y = truth.pose.pose.position.y - self.map_origin[1]
        pose.position.z = truth.pose.pose.position.z - self.map_origin[2]
        pose.orientation.x = truth.pose.pose.orientation.x
        pose.orientation.y = truth.pose.pose.orientation.y
        pose.orientation.z = truth.pose.pose.orientation.z
        pose.orientation.w = truth.pose.pose.orientation.w
        self.pose_pending = self.pose_executor.submit(self._gazebo_update, pose)

    def destroy_node(self):
        self.pose_executor.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()

    def _tick(self) -> None:
        now = self._now()
        elapsed = now - self.epoch_s if self.running else 0.0
        state = evaluate_trajectory(self.trajectory, elapsed, self.config)
        msg = odometry("map", "virtual_target", now, state.position_enu, state.velocity_enu)
        # Encode truth acceleration in the otherwise unused angular velocity fields for
        # logging only; the scientific adapter uses the standard linear state.
        msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z = map(float, state.acceleration_enu)
        self.truth_pub.publish(msg)
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose
        self.paths.append(pose)
        path = Path()
        path.header = msg.header
        path.poses = list(self.paths)
        self.path_pub.publish(path)
        ready = Bool()
        ready.data = True
        self.ready_pub.publish(ready)
        if self.tick_count % self.pose_divisor == 0:
            self._move_gazebo_marker(msg)
        self.tick_count += 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VirtualTargetNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
