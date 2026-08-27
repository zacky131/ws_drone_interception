import math
from pathlib import Path
import numpy as np
import yaml

from ras_hardware_mirror.calibrate_field_node import compute_field_calibration, compass_name, update_config_files


def test_compass_naming():
    assert compass_name(0.0) == "North (N)"
    assert compass_name(45.0) == "North-East (NE)"
    assert compass_name(90.0) == "East (E)"
    assert compass_name(180.0) == "South (S)"
    assert compass_name(270.0) == "West (W)"


def test_compute_field_calibration_angles_and_positions():
    # Facing North-East (45 deg NED) -> ENU angle is 45 deg
    calib_ne = compute_field_calibration(heading_deg_ned=45.0, pursuer_dist_m=29.0, static_target_dist_m=11.0)
    assert np.isclose(calib_ne["orientation_deg_enu"], 45.0)
    assert np.isclose(calib_ne["pursuer_enu_m"][0], -20.506, atol=1e-2)
    assert np.isclose(calib_ne["pursuer_enu_m"][1], -20.506, atol=1e-2)
    assert np.isclose(calib_ne["static_target_enu_m"][0], -7.778, atol=1e-2)
    assert np.isclose(calib_ne["static_target_enu_m"][1], -7.778, atol=1e-2)

    # Facing North (0 deg NED) -> ENU angle is 90 deg (pointing +Y)
    calib_n = compute_field_calibration(heading_deg_ned=0.0, pursuer_dist_m=29.0)
    assert np.isclose(calib_n["orientation_deg_enu"], 90.0)
    assert np.isclose(calib_n["pursuer_enu_m"][0], 0.0, atol=1e-3)
    assert np.isclose(calib_n["pursuer_enu_m"][1], -29.0, atol=1e-3)

    # Facing East (90 deg NED) -> ENU angle is 0 deg (pointing +X)
    calib_e = compute_field_calibration(heading_deg_ned=90.0, pursuer_dist_m=29.0)
    assert np.isclose(calib_e["orientation_deg_enu"], 0.0)
    assert np.isclose(calib_e["pursuer_enu_m"][0], -29.0, atol=1e-3)
    assert np.isclose(calib_e["pursuer_enu_m"][1], 0.0, atol=1e-3)


def test_update_config_files_dry_run(tmp_path):
    field_file = tmp_path / "field.yaml"
    dev_file = tmp_path / "hardware_mirror_dev.yaml"

    field_file.write_text(yaml.dump({
        "field": {"orientation_deg": 0.0},
        "hard_geofence": {"orientation_deg": 0.0},
        "target_region": {"orientation_deg": 0.0},
    }))
    dev_file.write_text(yaml.dump({
        "interceptor": {"initial_position_enu_m": [-29.0, 0.0, 0.0]},
        "virtual_target": {"orientation_deg": 0.0, "STATIC": {"position_enu_m": [-11.0, 0.0, 5.0]}},
    }))

    calib = compute_field_calibration(heading_deg_ned=45.0)
    update_config_files(calib, field_file, dev_file, dry_run=False)

    updated_field = yaml.safe_load(field_file.read_text())
    updated_dev = yaml.safe_load(dev_file.read_text())

    assert np.isclose(updated_field["field"]["orientation_deg"], 45.0)
    assert np.isclose(updated_field["hard_geofence"]["orientation_deg"], 45.0)
    assert np.isclose(updated_dev["virtual_target"]["orientation_deg"], 45.0)
    assert np.isclose(updated_dev["interceptor"]["initial_position_enu_m"][0], -20.506, atol=1e-2)
