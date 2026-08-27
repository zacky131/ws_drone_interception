import numpy as np

from ras_hardware_mirror.manual_control import SCENARIOS, decode_terminal_input, held_key_command, px4_fmu_prefix
from ras_hardware_mirror.manifest_utils import FROZEN_MANIFEST_SHA256, load_campaign_scenarios
from ras_hardware_mirror.safety_supervisor_node import run_gs6_gate


def test_default_px4_instance_has_un_namespaced_fmu_topics():
    assert px4_fmu_prefix("") == "/fmu"
    assert px4_fmu_prefix("px4_1") == "/px4_1/fmu"


def test_key_holds_compose_and_release_to_zero():
    command = held_key_command({"Up", "Left", "w", "d"}, 1.0, 0.6, 0.7)
    np.testing.assert_allclose(command, [1.0, 1.0, 0.6, -0.7])
    np.testing.assert_allclose(held_key_command(set(), 1.0, 0.6, 0.7), np.zeros(4))


def test_terminal_decoder_handles_ascii_and_both_arrow_forms():
    keys, pending = decode_terminal_input(b"qwt\x1b[A\x1bOD6")
    assert keys == ["q", "w", "t", "Up", "Left", "6"]
    assert pending == b""


def test_terminal_decoder_preserves_split_arrow_sequence():
    keys, pending = decode_terminal_input(b"a\x1b[")
    assert keys == ["a"]
    assert pending == b"\x1b["
    keys, pending = decode_terminal_input(b"C5", pending)
    assert keys == ["Right", "5"]
    assert pending == b""


def test_scenario_keys_are_complete_and_gs5_is_paired():
    assert set(SCENARIOS) == set("1234567")
    assert SCENARIOS["5"]["method"] == "M0prime"
    assert SCENARIOS["6"]["method"] == "M1"
    for key in ("trajectory", "condition", "seed"):
        assert SCENARIOS["5"][key] == SCENARIOS["6"][key]


def test_keyboard_gs6_gate():
    passed, detail = run_gs6_gate()
    assert passed, detail


def test_frozen_24_run_manifest_is_keyboard_selectable_and_paired():
    scenarios = load_campaign_scenarios()
    assert FROZEN_MANIFEST_SHA256 == "e32cc93e8a9a900dd9aeb5c7634894a823d72eb429a3bf48100b0ce5728543bb"
    assert set(scenarios) == set(range(1, 25))
    assert scenarios[1]["run_id"] == "HT1_HC0_M0prime_rep01"
    assert scenarios[24]["run_id"] == "HT2_HC1_M1_rep03"
    for index in range(1, 25, 2):
        left, right = scenarios[index], scenarios[index + 1]
        assert left["method"] == "M0prime"
        assert right["method"] == "M1"
        for key in ("trajectory", "condition", "seed", "repetition"):
            assert left[key] == right[key]


def test_rc_takeover_mode_names():
    from ras_hardware_mirror.experiment_manager_node import NAV_STATE_NAMES
    assert NAV_STATE_NAMES[0] == "MANUAL"
    assert NAV_STATE_NAMES[1] == "ALTCTL"
    assert NAV_STATE_NAMES[2] == "POSCTL"
    assert NAV_STATE_NAMES[5] == "AUTO_RTL"
    assert NAV_STATE_NAMES[14] == "OFFBOARD"
    assert NAV_STATE_NAMES[18] == "AUTO_LAND"

