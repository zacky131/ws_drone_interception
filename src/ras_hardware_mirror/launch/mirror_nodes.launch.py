from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    share = Path(get_package_share_directory("ras_hardware_mirror"))
    args = [
        DeclareLaunchArgument("method", default_value="M1", choices=["M0prime", "M1"]),
        DeclareLaunchArgument("trajectory", default_value="HT1", choices=["STATIC", "HT1", "HT2"]),
        DeclareLaunchArgument("condition", default_value="HC1", choices=["DEV0", "HC0", "HC1"]),
        DeclareLaunchArgument("seed", default_value="1"),
        DeclareLaunchArgument("repetition", default_value="1"),
        DeclareLaunchArgument("visualization_only", default_value="false", choices=["true", "false"]),
        DeclareLaunchArgument("config", default_value=str(share / "config/hardware_mirror_dev.yaml")),
        DeclareLaunchArgument("field", default_value=str(share / "config/field.yaml")),
        DeclareLaunchArgument("manifest", default_value=str(share / "manifests/hardware_mirror_24.csv")),
        DeclareLaunchArgument("output_root", default_value=""),
    ]
    common = {"config": LaunchConfiguration("config"), "method": LaunchConfiguration("method"), "trajectory": LaunchConfiguration("trajectory"), "condition": LaunchConfiguration("condition"), "seed": ParameterValue(LaunchConfiguration("seed"), value_type=int)}
    active = UnlessCondition(LaunchConfiguration("visualization_only"))
    nodes = [
        Node(package="ras_hardware_mirror", executable="virtual_target", parameters=[{"config": LaunchConfiguration("config"), "trajectory": LaunchConfiguration("trajectory"), "autostart": ParameterValue(LaunchConfiguration("visualization_only"), value_type=bool)}], output="screen"),
        Node(package="ras_hardware_mirror", executable="gazebo_state_provider", parameters=[{"config": LaunchConfiguration("config")}], output="screen"),
        Node(package="ras_hardware_mirror", executable="telemetry_emulator", parameters=[{"config": LaunchConfiguration("config"), "condition": LaunchConfiguration("condition"), "seed": ParameterValue(LaunchConfiguration("seed"), value_type=int)}], condition=active, output="screen"),
        Node(package="ras_hardware_mirror", executable="controller_adapter", parameters=[common], condition=active, output="screen"),
        Node(package="ras_hardware_mirror", executable="safety_supervisor", parameters=[{"config": LaunchConfiguration("config"), "field": LaunchConfiguration("field")}], condition=active, output="screen"),
        Node(package="ras_hardware_mirror", executable="experiment_manager", parameters=[{**common, "manifest": LaunchConfiguration("manifest")}], condition=active, output="screen"),
        Node(package="ras_hardware_mirror", executable="mirror_logger", parameters=[{**common, "field": LaunchConfiguration("field"), "repetition": ParameterValue(LaunchConfiguration("repetition"), value_type=int), "output_root": LaunchConfiguration("output_root")}], condition=active, output="screen"),
    ]
    return LaunchDescription(args + nodes)
