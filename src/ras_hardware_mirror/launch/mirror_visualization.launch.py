from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    share = Path(get_package_share_directory("ras_hardware_mirror"))
    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("dashboard", default_value="true"),
        Node(package="ras_hardware_mirror", executable="visualization", parameters=[{"config": str(share / "config/hardware_mirror_dev.yaml"), "field": str(share / "config/field.yaml")}], output="screen"),
        Node(package="rviz2", executable="rviz2", arguments=["-d", str(share / "config/rviz_hardware_mirror.rviz")], condition=IfCondition(LaunchConfiguration("rviz")), output="screen"),
        Node(package="ras_hardware_mirror", executable="live_dashboard", parameters=[{"config": str(share / "config/hardware_mirror_dev.yaml"), "field": str(share / "config/field.yaml")}], condition=IfCondition(LaunchConfiguration("dashboard")), output="screen"),
    ])
