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
