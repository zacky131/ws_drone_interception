from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    share = Path(get_package_share_directory("ras_hardware_mirror"))
    arguments = [DeclareLaunchArgument("method", default_value="M1"), DeclareLaunchArgument("trajectory", default_value="HT1"), DeclareLaunchArgument("condition", default_value="HC1"), DeclareLaunchArgument("seed", default_value="1"), DeclareLaunchArgument("repetition", default_value="1"), DeclareLaunchArgument("visualization_only", default_value="false"), DeclareLaunchArgument("output_root", default_value=""), DeclareLaunchArgument("gui", default_value="--gui"), DeclareLaunchArgument("manifest_row", default_value="0")]
    gazebo = IncludeLaunchDescription(PythonLaunchDescriptionSource(str(share / "launch/mirror_gazebo.launch.py")), launch_arguments={"gui": LaunchConfiguration("gui")}.items())
    nodes = IncludeLaunchDescription(PythonLaunchDescriptionSource(str(share / "launch/mirror_nodes.launch.py")), launch_arguments={key: LaunchConfiguration(key) for key in ("method", "trajectory", "condition", "seed", "repetition", "visualization_only", "output_root")}.items())
    visual = IncludeLaunchDescription(PythonLaunchDescriptionSource(str(share / "launch/mirror_visualization.launch.py")))
    keyboard = IncludeLaunchDescription(PythonLaunchDescriptionSource(str(share / "launch/mirror_keyboard.launch.py")), launch_arguments={"manifest_row": LaunchConfiguration("manifest_row")}.items())
    return LaunchDescription(arguments + [gazebo, nodes, visual, keyboard])
