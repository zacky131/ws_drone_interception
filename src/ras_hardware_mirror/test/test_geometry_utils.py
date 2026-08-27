import numpy as np

from ras_hardware_mirror.geometry_utils import distance, enu_to_ned, inside_altitude, inside_horizontal_box, ned_to_enu
from ras_hardware_mirror.ros_utils import diagnostic
from diagnostic_msgs.msg import DiagnosticStatus


def test_ned_enu_roundtrip_and_axes():
    ned = np.array([2.0, 3.0, -4.0])
    assert np.array_equal(ned_to_enu(ned), [3.0, 2.0, 4.0])
    assert np.array_equal(enu_to_ned(ned_to_enu(ned)), ned)


def test_distance_and_box_inclusion_are_boundary_inclusive():
    bounds = {"east_min_m": -1.0, "east_max_m": 2.0, "north_min_m": -3.0, "north_max_m": 4.0}
    assert distance([0, 0, 0], [3, 4, 0]) == 5.0
    assert inside_horizontal_box([-1, 4, 100], bounds)
    assert not inside_horizontal_box([-1.01, 0, 0], bounds)
    assert inside_altitude([0, 0, 2], {"min_m": 2, "max_m": 8})
    assert not inside_altitude([0, 0, 8.01], {"min_m": 2, "max_m": 8})


def test_humble_byte_diagnostic_level_is_normalized():
    assert diagnostic("test", DiagnosticStatus.OK).status[0].level == DiagnosticStatus.OK


def test_rotated_box_inclusion_and_rectangle_points():
    # Rotated by 45 deg (North-East)
    bounds = {
        "east_min_m": -10.0,
        "east_max_m": 10.0,
        "north_min_m": -5.0,
        "north_max_m": 5.0,
        "orientation_deg": 45.0,
    }
    # Point along 45 deg line at distance 8 (8*cos45, 8*sin45) is inside [-10, 10]
    p_inside = [8.0 * np.cos(np.pi / 4), 8.0 * np.sin(np.pi / 4), 0.0]
    assert inside_horizontal_box(p_inside, bounds)

    # Point along 45 deg line at distance 12 is outside
    p_outside = [12.0 * np.cos(np.pi / 4), 12.0 * np.sin(np.pi / 4), 0.0]
    assert not inside_horizontal_box(p_outside, bounds)

