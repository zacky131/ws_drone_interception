from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    share = Path(get_package_share_directory("ras_hardware_mirror"))
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=str(share / "config/hardware_mirror_dev.yaml")),
        DeclareLaunchArgument("manifest", default_value=str(share / "manifests/hardware_mirror_24.csv")),
        DeclareLaunchArgument("manifest_row", default_value="0"),
        Node(
            package="ras_hardware_mirror",
            executable="keyboard_control",
            parameters=[{
                "config": LaunchConfiguration("config"),
                "manifest": LaunchConfiguration("manifest"),
                "manifest_row": ParameterValue(LaunchConfiguration("manifest_row"), value_type=int),
            }],
            output="screen",
        ),
    ])
