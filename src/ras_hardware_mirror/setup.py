from glob import glob
from pathlib import Path

from setuptools import find_packages, setup


PACKAGE = "ras_hardware_mirror"


def data_files():
    entries = [
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE}"]),
        (f"share/{PACKAGE}", ["package.xml", "README.md"]),
    ]
    for folder in ("launch", "config", "worlds", "scripts", "tools", "manifests"):
        for path in glob(f"{folder}/**/*", recursive=True):
            if Path(path).is_file():
                entries.append((f"share/{PACKAGE}/{Path(path).parent}", [path]))
    return entries


setup(
    name=PACKAGE,
    version="0.1.0",
    packages=find_packages(),
    data_files=data_files(),
    install_requires=["setuptools", "numpy", "pyyaml", "matplotlib"],
    zip_safe=True,
    maintainer="zacky",
    maintainer_email="zacky@example.com",
    description="One-X500 Gazebo/RTK hardware-mirror interception rehearsal",
    license="MIT",
    entry_points={
        "console_scripts": [
            "virtual_target = ras_hardware_mirror.virtual_target_node:main",
            "telemetry_emulator = ras_hardware_mirror.telemetry_emulator_node:main",
            "gazebo_state_provider = ras_hardware_mirror.gazebo_state_provider_node:main",
            "controller_adapter = ras_hardware_mirror.controller_adapter_node:main",
            "experiment_manager = ras_hardware_mirror.experiment_manager_node:main",
            "safety_supervisor = ras_hardware_mirror.safety_supervisor_node:main",
            "visualization = ras_hardware_mirror.visualization_node:main",
            "live_dashboard = ras_hardware_mirror.live_dashboard:main",
            "mirror_logger = ras_hardware_mirror.logger_node:main",
            "keyboard_control = ras_hardware_mirror.keyboard_control_node:main",
        ]
    },
)
