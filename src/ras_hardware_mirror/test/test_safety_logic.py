from pathlib import Path
import numpy as np
import yaml

from ras_hardware_mirror.experiment_manager_node import ExperimentStateMachine
from ras_hardware_mirror.safety_supervisor_node import SafetyMonitor
from ras_hardware_mirror.state_types import ExperimentPhase, NavigationState, SafetyInputs


FIELD = yaml.safe_load((Path(__file__).parents[1] / "config/field.yaml").read_text())


def state(position=(0, 0, 5), valid=True):
    return NavigationState(0, np.asarray(position), np.zeros(3), valid=valid)


def values(**changes):
    base = dict(now_s=3.0, run_start_s=0.0, interceptor=state(), target=state(), controller_age_s=0.01, command_finite=True, px4_healthy=True, manual_abort=False)
    base.update(changes)
    return SafetyInputs(**base)


def test_successful_capture_transitions_to_hold():
    machine = ExperimentStateMachine(ExperimentPhase.RUN)
    assert machine.event("capture") == ExperimentPhase.CAPTURE
    assert machine.event("settle") == ExperimentPhase.HOLD


def test_scenario_uses_takeoff_and_stabilize_before_run():
    machine = ExperimentStateMachine(ExperimentPhase.PRECHECK)
    assert machine.event("ready") == ExperimentPhase.TAKEOFF
    assert machine.event("altitude_reached") == ExperimentPhase.STABILIZE
    assert machine.event("stable") == ExperimentPhase.RUN


def test_manual_abort_transitions_run_abort_hold():
    monitor = SafetyMonitor(FIELD, 30.0, 0.25)
    assert monitor.evaluate(values(manual_abort=True)).reason == "manual abort"
    machine = ExperimentStateMachine(ExperimentPhase.RUN)
    assert machine.event("abort") == ExperimentPhase.ABORT
    assert machine.event("settle") == ExperimentPhase.HOLD


def test_all_required_faults_abort_deterministically():
    monitor = SafetyMonitor(FIELD, 30.0, 0.25)
    cases = [
        (values(interceptor=state((41, 0, 5))), "interceptor hard-geofence violation"),
        (values(target=state((31, 0, 5))), "target-region violation"),
        (values(interceptor=state(valid=False)), "invalid interceptor state"),
        (values(controller_age_s=0.3), "controller heartbeat timeout"),
        (values(command_finite=False), "non-finite controller command"),
        (values(now_s=31.0), "experiment timeout"),
    ]
    for inputs, expected in cases:
        decision = monitor.evaluate(inputs)
        assert decision.abort
        assert decision.reason == expected


def test_nominal_safety_state_is_ok():
    assert SafetyMonitor(FIELD, 30.0, 0.25).evaluate(values()).abort is False
