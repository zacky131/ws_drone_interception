"""Common exact-source-time estimator and command envelope for external baselines."""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from dapcs_mpc.delay_aware_imm import TelemetryPacket
from drone_interception_px4.telemetry import TelemetryEvent
from paper_completion.delay_aware_ca import DelayAwareCA


CONTROL_DT_S = 0.02


class SourceTimeCAInterface:
    """Expose exactly M1's causal current-time single-CA posterior."""

    def __init__(self, config_path: str | Path) -> None:
        self.estimator = DelayAwareCA(config_path)
        self.last_diagnostics: dict[str, Any] = {}

    def reset(self) -> None:
        self.estimator.reset(0.0)
        self.last_diagnostics = {}

    def process(self, event: TelemetryEvent) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        if not isinstance(event, TelemetryEvent):
            raise TypeError("external baselines require a timestamped TelemetryEvent")
        measurement = np.zeros(6) if event.measurement is None else event.measurement
        packet = TelemetryPacket(
            source_timestamp_s=event.actual_source_timestamp_s,
            arrival_timestamp_s=event.arrival_timestamp_s,
            position=measurement[:3],
            velocity=measurement[3:6],
            valid=event.valid,
            drop=event.drop,
        )
        accepted_before = self.estimator.accepted_updates
        start = time.perf_counter_ns()
        self.estimator.process_packet(packet)
        elapsed_s = (time.perf_counter_ns() - start) * 1e-9
        accepted = self.estimator.accepted_updates > accepted_before
        mean, covariance = self.estimator.state()
        base = self.estimator.diagnostics()
        self.last_diagnostics = {
            **base,
            "target_estimate": mean,
            "current_covariance": covariance,
            "mode_probabilities": np.array([0.0, 1.0, 0.0]),
            "estimator_time_s": elapsed_s,
            "packet_source_timestamp_s": event.actual_source_timestamp_s,
            "packet_arrival_timestamp_s": event.arrival_timestamp_s,
            "packet_accepted": int(accepted),
            "measurement_update_timestamp_s": (
                float(self.estimator.last_update_source_timestamp_s)
                if accepted else math.nan
            ),
            "posterior_timestamp_s": float(self.estimator.current_time_s),
            "configured_delay_s": event.configured_delay_s,
            "requested_source_timestamp_s": event.requested_source_timestamp_s,
            "actual_measurement_source_timestamp_s": event.actual_source_timestamp_s,
            "measurement_history_left_timestamp_s": event.history_left_timestamp_s,
            "measurement_history_right_timestamp_s": event.history_right_timestamp_s,
            "measurement_interpolation_alpha": event.interpolation_alpha,
            "startup_clamped": int(event.startup_clamped),
            "physical_measurement_age_s": event.physical_measurement_age_s,
            "runtime_timing_semantics": "exact_source_time_then_repropagate",
            "estimator_architecture": "single_CA_identical_to_M1",
            "controller_information_source": "corrected_CA_posterior_only",
        }
        return mean, covariance, dict(self.last_diagnostics)

    def forecast(self, times_s: np.ndarray) -> np.ndarray:
        times = np.asarray(times_s, dtype=float)
        if np.any(times < 0.0):
            raise ValueError("forecast times must be causal nonnegative horizons")
        return np.asarray([self.estimator.forecast_mean(float(t)) for t in times])


class CommandSafetyEnvelope:
    """Frozen Rev6 acceleration and command-rate envelope."""

    def __init__(self, config_path: str | Path) -> None:
        config = yaml.safe_load(Path(config_path).read_text())
        physical = config["physical_constraints"]
        self.max_acceleration = float(physical["max_acceleration_mps2"])
        self.max_acceleration_axis = float(physical["max_acceleration_per_axis_mps2"])
        self.max_jerk = float(physical["max_jerk_mps3"])
        self.previous_command = np.zeros(3)

    def reset(self) -> None:
        self.previous_command = np.zeros(3)

    def apply(self, command: np.ndarray, dt_s: float = CONTROL_DT_S) -> tuple[np.ndarray, dict[str, Any]]:
        raw = np.asarray(command, dtype=float).reshape(3)
        if not np.all(np.isfinite(raw)):
            raise FloatingPointError("external baseline produced a nonfinite command")
        limited = np.clip(raw, -self.max_acceleration_axis, self.max_acceleration_axis)
        norm = float(np.linalg.norm(limited))
        if norm > self.max_acceleration:
            limited *= self.max_acceleration / norm
        delta = limited - self.previous_command
        delta_norm = float(np.linalg.norm(delta))
        maximum_delta = self.max_jerk * float(dt_s)
        rate_limited = delta_norm > maximum_delta + 1e-12
        if rate_limited:
            limited = self.previous_command + delta * (maximum_delta / delta_norm)
        limited = np.clip(limited, -self.max_acceleration_axis, self.max_acceleration_axis)
        norm = float(np.linalg.norm(limited))
        if norm > self.max_acceleration:
            limited *= self.max_acceleration / norm
        diagnostics = {
            "external_raw_command": raw.copy(),
            "external_envelope_clipped": int(not np.allclose(raw, limited, rtol=0.0, atol=1e-12)),
            "external_rate_limited": int(rate_limited),
            "external_command_delta_norm_mps2": float(np.linalg.norm(limited - self.previous_command)),
            "external_max_command_delta_mps2": maximum_delta,
        }
        self.previous_command = limited.copy()
        return limited, diagnostics
