"""Interactive and ROS 2-based Field Orientation & Heading Calibrator.

Usage:
  # 1. Live measurement from PX4 drone heading (requires MicroXRCEAgent):
  ros2 run ras_hardware_mirror calibrate_field

  # 2. Manual heading entry (e.g., if you measured 45° with a compass):
  ros2 run ras_hardware_mirror calibrate_field --ros-args -p heading:=45.0
  # Or via python directly:
  python3 src/ras_hardware_mirror/ras_hardware_mirror/calibrate_field_node.py --heading 45.0
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from px4_msgs.msg import VehicleLocalPosition
    HAVE_ROS = True
except ImportError:
    HAVE_ROS = False


PX4_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
) if HAVE_ROS else None


def compass_name(deg: float) -> str:
    """Return 8-wind compass direction name."""
    deg = deg % 360.0
    sectors = [
        "North (N)", "North-East (NE)", "East (E)", "South-East (SE)",
        "South (S)", "South-West (SW)", "West (W)", "North-West (NW)",
    ]
    idx = int((deg + 22.5) // 45) % 8
    return sectors[idx]


def compute_field_calibration(
    heading_deg_ned: float,
    pursuer_dist_m: float = 29.0,
    static_target_dist_m: float = 11.0,
    altitude_m: float = 5.0,
) -> dict[str, Any]:
    """Compute ENU orientation angle and rotated pursuer/target coordinates.

    Args:
        heading_deg_ned: Compass yaw (clockwise from North, in degrees [0..360]).
        pursuer_dist_m: Distance of pursuer takeoff pad back from field center.
        static_target_dist_m: Distance of static target back from field center.
        altitude_m: Nominal flight altitude.

    Returns:
        Dictionary of calibrated parameters.
    """
    heading_rad_ned = math.radians(heading_deg_ned % 360.0)

    # Conversion from NED compass heading (clockwise from North) to ENU angle (counter-clockwise from East):
    # theta_enu = 90 - heading_ned
    theta_enu_deg = (90.0 - heading_deg_ned) % 360.0
    if theta_enu_deg > 180.0:
        theta_enu_deg -= 360.0
    theta_enu_rad = math.radians(theta_enu_deg)

    cos_t = math.cos(theta_enu_rad)
    sin_t = math.sin(theta_enu_rad)

    # Pursuer starting position (D meters backwards along the field axis)
    pursuer_x = round(-pursuer_dist_m * cos_t, 3)
    pursuer_y = round(-pursuer_dist_m * sin_t, 3)
    pursuer_enu = [pursuer_x, pursuer_y, 0.0]

    # Static target position (D meters backwards along field axis, 18m ahead of pursuer)
    static_x = round(-static_target_dist_m * cos_t, 3)
    static_y = round(-static_target_dist_m * sin_t, 3)
    static_enu = [static_x, static_y, float(altitude_m)]

    return {
        "heading_deg_ned": round(heading_deg_ned % 360.0, 2),
        "compass_dir": compass_name(heading_deg_ned),
        "orientation_deg_enu": round(theta_enu_deg, 2),
        "pursuer_enu_m": pursuer_enu,
        "static_target_enu_m": static_enu,
        "altitude_m": float(altitude_m),
    }


from .config_utils import default_config, default_field, package_file, WORKSPACE_ROOT


def update_single_config_file(calib: dict[str, Any], path: Path, dry_run: bool = False) -> None:
    """Update a specific YAML configuration file if it exists."""
    path = Path(path)
    if not path.is_file():
        return

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "field" in cfg and "hard_geofence" in cfg:
        # field.yaml
        cfg["field"]["orientation_deg"] = calib["orientation_deg_enu"]
        cfg["hard_geofence"]["orientation_deg"] = calib["orientation_deg_enu"]
        cfg["target_region"]["orientation_deg"] = calib["orientation_deg_enu"]
    elif "interceptor" in cfg and "virtual_target" in cfg:
        # hardware_mirror_dev.yaml
        cfg["interceptor"]["initial_position_enu_m"] = calib["pursuer_enu_m"]
        cfg["virtual_target"]["orientation_deg"] = calib["orientation_deg_enu"]
        if "STATIC" in cfg["virtual_target"]:
            cfg["virtual_target"]["STATIC"]["position_enu_m"] = calib["static_target_enu_m"]

    if dry_run:
        return

    # Backup and write
    bak = path.with_suffix(".yaml.bak")
    with open(bak, "w", encoding="utf-8") as f:
        f.write(path.read_text(encoding="utf-8"))

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def update_config_files(
    calib: dict[str, Any],
    field_path: Path,
    dev_config_path: Path,
    dry_run: bool = False,
) -> list[Path]:
    """Update field.yaml and hardware_mirror_dev.yaml in workspace and share directory."""
    targets = [Path(field_path), Path(dev_config_path)]

    # Also locate source workspace files if different
    src_field = WORKSPACE_ROOT / "src" / "ras_hardware_mirror" / "config" / "field.yaml"
    src_dev = WORKSPACE_ROOT / "src" / "ras_hardware_mirror" / "config" / "hardware_mirror_dev.yaml"
    if src_field.is_file() and src_field not in targets:
        targets.append(src_field)
    if src_dev.is_file() and src_dev not in targets:
        targets.append(src_dev)

    updated = []
    for t in targets:
        if t.is_file():
            update_single_config_file(calib, t, dry_run=dry_run)
            updated.append(t)

    return updated



def print_calibration_summary(calib: dict[str, Any], field_path: Path, dev_config_path: Path) -> None:
    """Print a clean visual summary of calibration results."""
    banner = "=" * 65
    print("\n" + banner)
    print("      FIELD ORIENTATION & HEADING CALIBRATION COMPLETE")
    print(banner)
    print(f"  Drone Compass Heading (NED) : {calib['heading_deg_ned']}°  [{calib['compass_dir']}]")
    print(f"  Field Orientation Angle (θ) : {calib['orientation_deg_enu']}° (ENU)")
    print(f"  Pursuer Starting Position   : {calib['pursuer_enu_m']} m")
    print(f"  Static Target Position      : {calib['static_target_enu_m']} m")
    print(banner)
    print("  Updated Configuration Files:")
    print(f"    • {field_path}")
    print(f"    • {dev_config_path}")
    print(banner)
    print("  [✓] System is calibrated! You can now run hardware trials.")
    print(banner + "\n")


if HAVE_ROS:
    class HeadingSamplerNode(Node):
        """Samples PX4 heading from /fmu/out/vehicle_local_position."""

        def __init__(self, target_samples: int = 40) -> None:
            super().__init__("field_heading_calibrator")
            self.target_samples = target_samples
            self.samples_sin: list[float] = []
            self.samples_cos: list[float] = []
            self.done = False
            self.create_subscription(
                VehicleLocalPosition,
                "/fmu/out/vehicle_local_position",
                self._cb_local_pos,
                PX4_QOS,
            )

        def _cb_local_pos(self, msg: VehicleLocalPosition) -> None:
            if not math.isfinite(msg.heading):
                return
            h = float(msg.heading)
            self.samples_sin.append(math.sin(h))
            self.samples_cos.append(math.cos(h))
            sys.stdout.write(f"\r  Sampling drone heading: {len(self.samples_sin)}/{self.target_samples} samples...")
            sys.stdout.flush()
            if len(self.samples_sin) >= self.target_samples:
                self.done = True


def sample_px4_heading(target_samples: int = 40, timeout_s: float = 10.0) -> float:
    """Sample live heading from PX4 via ROS 2."""
    if not HAVE_ROS:
        raise RuntimeError("ROS 2 or px4_msgs is not available in current python environment")

    node = HeadingSamplerNode(target_samples=target_samples)
    start_t = time.monotonic()

    print(f"\n[1/2] Waiting for PX4 heading on /fmu/out/vehicle_local_position (timeout: {timeout_s}s)...")
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.05)
            if time.monotonic() - start_t > timeout_s:
                break
    finally:
        node.destroy_node()

    if not node.samples_sin:
        raise TimeoutError(
            "Timed out waiting for PX4 heading on /fmu/out/vehicle_local_position.\n"
            "Ensure MicroXRCEAgent is running and PX4 is powered on.\n"
            "Tip: You can also specify manual heading via: --heading <degrees>"
        )

    # Circular mean of angles
    mean_sin = np.mean(node.samples_sin)
    mean_cos = np.mean(node.samples_cos)
    mean_heading_rad = math.atan2(mean_sin, mean_cos)
    mean_heading_deg = (math.degrees(mean_heading_rad) + 360.0) % 360.0
    print(f"\n  Average Heading: {mean_heading_deg:.2f}° [{compass_name(mean_heading_deg)}]")
    return mean_heading_deg


def find_default_paths() -> tuple[Path, Path]:
    """Locate field.yaml and hardware_mirror_dev.yaml."""
    base = Path(__file__).resolve().parents[1]
    field = base / "config" / "field.yaml"
    dev = base / "config" / "hardware_mirror_dev.yaml"
    return field, dev


def main(args=None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate field orientation and pursuer/target layout.")
    parser.add_argument("--heading", type=float, default=None, help="Manual heading in degrees [0..360] (bypasses PX4 ROS sampling)")
    parser.add_argument("--samples", type=int, default=40, help="Number of heading samples to collect from PX4")
    parser.add_argument("--timeout", type=float, default=10.0, help="Timeout in seconds waiting for PX4")
    parser.add_argument("--dry-run", action="store_true", help="Calculate and display values without writing files")
    parser.add_argument("--field-path", type=str, default="", help="Path to field.yaml")
    parser.add_argument("--config-path", type=str, default="", help="Path to hardware_mirror_dev.yaml")

    # Filter out ROS args if launched via ros2 run
    clean_argv = []
    i = 0
    raw_argv = sys.argv[1:] if args is None else list(args)
    while i < len(raw_argv):
        arg = raw_argv[i]
        if arg == "--ros-args":
            break
        clean_argv.append(arg)
        i += 1

    parsed = parser.parse_args(clean_argv)

    field_path = Path(parsed.field_path) if parsed.field_path else default_field()
    dev_path = Path(parsed.config_path) if parsed.config_path else default_config()

    heading_deg = parsed.heading
    if heading_deg is None:
        if HAVE_ROS:
            rclpy.init(args=args)
            try:
                heading_deg = sample_px4_heading(target_samples=parsed.samples, timeout_s=parsed.timeout)
            finally:
                if rclpy.ok():
                    rclpy.shutdown()
        else:
            print("Error: ROS 2 not available. Please pass --heading <degrees> manually.")
            sys.exit(1)

    print("\n[2/2] Calculating rotated geometry and updating configuration files...")
    calib = compute_field_calibration(heading_deg_ned=heading_deg)
    update_config_files(calib, field_path, dev_path, dry_run=parsed.dry_run)
    print_calibration_summary(calib, field_path, dev_path)


if __name__ == "__main__":
    main()
