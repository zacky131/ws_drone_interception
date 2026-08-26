"""Five-method adapter using one authoritative corrected telemetry event."""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from dapcs_mpc.controller import ExtensionControllerAdapter
from dapcs_mpc.delay_aware_imm import TelemetryPacket
from drone_interception_px4.controller_adapter import ExistingControllerAdapter
from drone_interception_px4.telemetry import TelemetryEvent
from final_closure.controller import ClosureControllerAdapter
from m0prime_confirmatory.controller import ConfirmatoryControllerAdapter


M0 = "mpc_ekf_ca"
M0PRIME = "A0prime_CA_arrival"
M1 = "mpc_dca_tracking"
M2 = "mpc_dimm_tracking"
M3 = "mpc_dca_capture"
METHODS = (M0, M0PRIME, M1, M2, M3)
MANUSCRIPT_NAMES = {M0: "M0", M0PRIME: "M0prime", M1: "M1", M2: "M2", M3: "M3"}


def packet_from_event(event: TelemetryEvent) -> TelemetryPacket:
    """Convert once, preserving the generator's authoritative timestamps."""

    if not isinstance(event, TelemetryEvent):
        raise TypeError("unified corrected control requires a timestamped TelemetryEvent")
    measurement = np.zeros(6) if event.measurement is None else event.measurement
    return TelemetryPacket(
        source_timestamp_s=event.actual_source_timestamp_s,
        arrival_timestamp_s=event.arrival_timestamp_s,
        position=measurement[:3],
        velocity=measurement[3:6],
        valid=event.valid,
        drop=event.drop,
    )


def event_diagnostics(event: TelemetryEvent) -> dict[str, Any]:
    return {
        "packet_source_timestamp_s": event.actual_source_timestamp_s,
        "packet_arrival_timestamp_s": event.arrival_timestamp_s,
        "configured_delay_s": event.configured_delay_s,
        "requested_source_timestamp_s": event.requested_source_timestamp_s,
        "actual_measurement_source_timestamp_s": event.actual_source_timestamp_s,
        "measurement_history_left_timestamp_s": event.history_left_timestamp_s,
        "measurement_history_right_timestamp_s": event.history_right_timestamp_s,
        "measurement_interpolation_alpha": event.interpolation_alpha,
        "startup_clamped": int(event.startup_clamped),
        "physical_measurement_age_s": event.physical_measurement_age_s,
    }


class UnifiedControllerAdapter:
    """Route five frozen architectures through corrected packet timestamp ingress."""

    def __init__(self, method: str, config_path: str | Path, trial_seed: int) -> None:
        if method not in METHODS:
            raise ValueError(f"unsupported unified method: {method}")
        self.method = method
        self.seed = int(trial_seed)
        if method == M0:
            self.controller = ExistingControllerAdapter(M0, config_path, trial_seed)
        elif method in (M0PRIME, M1):
            self.controller = ConfirmatoryControllerAdapter(method, config_path, trial_seed)
        elif method == M2:
            self.controller = ExtensionControllerAdapter(M2, config_path, trial_seed)
        else:
            self.controller = ClosureControllerAdapter(M3, config_path, trial_seed)
        self.last_diagnostics: dict[str, Any] = {}

    def reset(
        self, seed: int | None = None, trajectory_id: str = "", condition: str = "C0"
    ) -> None:
        self.seed = self.seed if seed is None else int(seed)
        if self.method == M0:
            self.controller.reset()
        else:
            self.controller.reset(self.seed, trajectory_id, condition)
        self.last_diagnostics = {}

    def _legacy_arrival_step(
        self,
        pursuer_state: np.ndarray,
        event: TelemetryEvent,
        dt_s: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        command, info = self.controller.step(
            pursuer_state, event.measurement, dt_s, event.arrival_timestamp_s
        )
        diagnostics = dict(info)
        diagnostics.update(event_diagnostics(event))
        diagnostics.update(
            {
                "packet_accepted": int(event.valid),
                "measurement_update_timestamp_s": (
                    event.arrival_timestamp_s if event.valid else math.nan
                ),
                "posterior_timestamp_s": event.arrival_timestamp_s,
                "runtime_timing_semantics": "legacy_arrival_time",
                "estimator_architecture": "legacy_EKF_CA",
            }
        )
        return command, diagnostics

    def _m2_step(
        self, pursuer_state: np.ndarray, event: TelemetryEvent
    ) -> tuple[np.ndarray, dict[str, Any]]:
        packet = packet_from_event(event)
        estimator = self.controller.estimator
        assert estimator is not None
        accepted_before = estimator.accepted_updates
        update_start = time.perf_counter_ns()
        self.controller.update_telemetry(packet)
        self.controller._last_update_time_s = (time.perf_counter_ns() - update_start) * 1e-9
        command = self.controller.compute_command(
            event.arrival_timestamp_s, np.asarray(pursuer_state, dtype=float)
        )
        diagnostics = self.controller.get_diagnostics()
        accepted = estimator.accepted_updates > accepted_before
        diagnostics.update(event_diagnostics(event))
        diagnostics.update(
            {
                "packet_accepted": int(accepted),
                "measurement_update_timestamp_s": (
                    float(estimator.last_update_source_timestamp_s) if accepted else math.nan
                ),
                "posterior_timestamp_s": float(estimator.current_time_s),
                "runtime_timing_semantics": "exact_source_time_then_repropagate",
                "estimator_architecture": "CV_CA_Singer_IMM",
            }
        )
        return command, diagnostics

    def _m3_step(
        self, pursuer_state: np.ndarray, event: TelemetryEvent
    ) -> tuple[np.ndarray, dict[str, Any]]:
        packet = packet_from_event(event)
        estimator = self.controller.estimator
        accepted_before = estimator.accepted_updates
        self.controller._step_start_ns = time.perf_counter_ns()
        self.controller.update_telemetry(packet)
        command = self.controller.compute_command(
            event.arrival_timestamp_s, np.asarray(pursuer_state, dtype=float)
        )
        diagnostics = self.controller.get_diagnostics()
        accepted = estimator.accepted_updates > accepted_before
        diagnostics.update(event_diagnostics(event))
        diagnostics.update(
            {
                "packet_accepted": int(accepted),
                "measurement_update_timestamp_s": (
                    float(estimator.last_update_source_timestamp_s) if accepted else math.nan
                ),
                "posterior_timestamp_s": float(estimator.current_time_s),
                "runtime_timing_semantics": "exact_source_time_then_repropagate",
                "estimator_architecture": "single_CA",
            }
        )
        return command, diagnostics

    def step(
        self,
        interceptor_state_enu: np.ndarray,
        target_measurement_enu: TelemetryEvent,
        dt_s: float,
        sim_time_s: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not isinstance(target_measurement_enu, TelemetryEvent):
            raise TypeError("unified corrected control requires a timestamped TelemetryEvent")
        if self.method == M0:
            command, diagnostics = self._legacy_arrival_step(
                interceptor_state_enu, target_measurement_enu, dt_s
            )
        elif self.method in (M0PRIME, M1):
            command, diagnostics = self.controller.step(
                interceptor_state_enu, target_measurement_enu, dt_s, sim_time_s
            )
            diagnostics = dict(diagnostics)
            diagnostics["estimator_architecture"] = "single_CA"
        elif self.method == M2:
            command, diagnostics = self._m2_step(
                interceptor_state_enu, target_measurement_enu
            )
        else:
            command, diagnostics = self._m3_step(
                interceptor_state_enu, target_measurement_enu
            )
        self.last_diagnostics = dict(diagnostics)
        return np.asarray(command, dtype=float), dict(diagnostics)

    def get_diagnostics(self) -> dict[str, Any]:
        return dict(self.last_diagnostics)
