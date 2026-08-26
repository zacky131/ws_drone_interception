from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    share = Path(get_package_share_directory("ras_hardware_mirror"))
    gui = LaunchConfiguration("gui")
    launcher = ExecuteProcess(cmd=[str(share / "scripts/launch_single_x500.sh"), gui], output="screen")
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="--gui", choices=["--gui", "--headless"]),
        launcher,
    ])
