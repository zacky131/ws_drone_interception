from pathlib import Path
import numpy as np
import yaml

from ras_hardware_mirror.trajectory_library import evaluate_trajectory


CONFIG = yaml.safe_load((Path(__file__).parents[1] / "config/hardware_mirror_dev.yaml").read_text())
FIELD = yaml.safe_load((Path(__file__).parents[1] / "config/field.yaml").read_text())


def test_trajectories_are_finite_repeatable_and_at_configured_altitude():
    for name in ("HT1", "HT2"):
        for t in np.linspace(0, 30, 601):
            first = evaluate_trajectory(name, t, CONFIG)
            second = evaluate_trajectory(name, t, CONFIG)
            for value in (first.position_enu, first.velocity_enu, first.acceleration_enu):
                assert value.shape == (3,)
                assert np.all(np.isfinite(value))
            assert np.array_equal(first.position_enu, second.position_enu)
            assert first.position_enu[2] == CONFIG["virtual_target"]["nominal_altitude_m"]


def test_trajectories_stay_in_development_target_region():
    bounds = FIELD["target_region"]
    for name in ("HT1", "HT2"):
        positions = np.array([evaluate_trajectory(name, t, CONFIG).position_enu for t in np.linspace(0, 30, 3001)])
        assert positions[:, 0].min() >= bounds["east_min_m"]
        assert positions[:, 0].max() <= bounds["east_max_m"]
        assert positions[:, 1].min() >= bounds["north_min_m"]
        assert positions[:, 1].max() <= bounds["north_max_m"]


def test_ht1_state_is_continuous_through_terminal_hold():
    terminal = CONFIG["virtual_target"]["HT1"]["duration_s"]
    left = evaluate_trajectory("HT1", terminal - 1e-5, CONFIG)
    at = evaluate_trajectory("HT1", terminal, CONFIG)
    right = evaluate_trajectory("HT1", terminal + 1e-5, CONFIG)
    assert np.linalg.norm(left.position_enu - at.position_enu) < 1e-5
    assert np.linalg.norm(left.velocity_enu - at.velocity_enu) < 1e-4
    assert np.linalg.norm(left.acceleration_enu - at.acceleration_enu) < 1e-3
    assert np.array_equal(at.position_enu, right.position_enu)
    assert np.array_equal(at.velocity_enu, right.velocity_enu)
    assert np.array_equal(at.acceleration_enu, right.acceleration_enu)
