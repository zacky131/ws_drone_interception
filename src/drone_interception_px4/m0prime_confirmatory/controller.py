"""Two-arm controller interface for the M0-prime/M1 confirmatory study."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from drone_interception_px4.telemetry import TelemetryEvent
from paper_completion.controller import AttributionControllerAdapter

from .arrival_time_ca import ArrivalTimeCA


M0PRIME = "A0prime_CA_arrival"
M1 = "mpc_dca_tracking"
METHODS = (M0PRIME, M1)
MANUSCRIPT_LABELS = {M0PRIME: "M0'", M1: "M1"}


class ConfirmatoryControllerAdapter(AttributionControllerAdapter):
    """Reuse M1's complete controller path and vary only update placement."""

    def __init__(self, method: str, config_path: str | Path, trial_seed: int) -> None:
        if method not in METHODS:
            raise ValueError(f"unsupported confirmatory method: {method}")
        # Construct the frozen M1 path in both arms.  M0-prime then swaps only
        # the DelayAwareCA instance for its subclass with arrival-time update.
        super().__init__(M1, config_path, trial_seed)
        if method == M0PRIME:
            self.estimator = ArrivalTimeCA(self.imm_config_path)
            self.method = M0PRIME
            self.reset(trial_seed, "", "C0")

    def _delayed_ca_step(
        self,
        interceptor_state_enu: np.ndarray,
        target_measurement_enu: TelemetryEvent,
        dt_s: float,
        sim_time_s: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        assert self.estimator is not None
        if not isinstance(target_measurement_enu, TelemetryEvent):
            raise TypeError("confirmatory control requires a timestamped TelemetryEvent")
        accepted_before = self.estimator.accepted_updates
        command, diagnostics = super()._delayed_ca_step(
            interceptor_state_enu,
            target_measurement_enu,
            dt_s,
            sim_time_s,
        )
        arrival = target_measurement_enu.arrival_timestamp_s
        source = target_measurement_enu.actual_source_timestamp_s
        accepted = self.estimator.accepted_updates > accepted_before
        update_timestamp = (
            float(self.estimator.last_update_source_timestamp_s)
            if accepted
            else math.nan
        )
        diagnostics.update(
            {
                "packet_source_timestamp_s": source,
                "packet_arrival_timestamp_s": arrival,
                "packet_delay_s": target_measurement_enu.physical_measurement_age_s,
                "packet_accepted": int(accepted),
                "measurement_update_timestamp_s": update_timestamp,
                "posterior_timestamp_s": float(self.estimator.current_time_s),
                "configured_delay_s": target_measurement_enu.configured_delay_s,
                "requested_source_timestamp_s": (
                    target_measurement_enu.requested_source_timestamp_s
                ),
                "actual_measurement_source_timestamp_s": source,
                "measurement_history_left_timestamp_s": (
                    target_measurement_enu.history_left_timestamp_s
                ),
                "measurement_history_right_timestamp_s": (
                    target_measurement_enu.history_right_timestamp_s
                ),
                "measurement_interpolation_alpha": (
                    target_measurement_enu.interpolation_alpha
                ),
                "startup_clamped": int(target_measurement_enu.startup_clamped),
                "physical_measurement_age_s": (
                    target_measurement_enu.physical_measurement_age_s
                ),
                "runtime_timing_semantics": (
                    "arrival_time"
                    if self.method == M0PRIME
                    else "source_time_then_repropagate"
                ),
            }
        )
        self.last_diagnostics = diagnostics
        return command, dict(diagnostics)
