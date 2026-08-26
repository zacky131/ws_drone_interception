from setuptools import find_packages, setup

package_name = "drone_interception_px4"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "pandas", "pyyaml"],
    zip_safe=True,
    maintainer="zacky",
    maintainer_email="zacky@localhost",
    description="PX4 v1.15.2 adapters for drone interception validation",
    license="MIT",
    entry_points={
        "console_scripts": [
            "target_trajectory_player = drone_interception_px4.target_trajectory_player:main",
            "target_telemetry_emulator = drone_interception_px4.target_telemetry_emulator:main",
            "interceptor_controller = drone_interception_px4.interceptor_controller:main",
            "experiment_supervisor = drone_interception_px4.experiment_supervisor:main",
            "px4_state_bridge = drone_interception_px4.px4_state_bridge:main",
            "single_vehicle_smoke = drone_interception_px4.single_vehicle_smoke:main",
            "two_vehicle_smoke = drone_interception_px4.two_vehicle_smoke:main",
        ]
    },
)
