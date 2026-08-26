"""Single-model CA estimator with the frozen IMM's delayed-update semantics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np

from dapcs_mpc.delay_aware_imm import (
    DelayAwareIMM,
    IMMConfig,
    MEASUREMENT_DIM,
    STATE_DIM,
    TelemetryPacket,
)


def _symmetrize_psd(matrix: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    return (vectors * np.maximum(values, floor)) @ vectors.T


@dataclass
class _Snapshot:
    timestamp_s: float
    mean: np.ndarray
    covariance: np.ndarray

    def clone(self) -> "_Snapshot":
        return _Snapshot(self.timestamp_s, self.mean.copy(), self.covariance.copy())


class DelayAwareCA:
    """One CA model with A2-identical timestamp rollback and repropagation.

    The state, CA transition, CA process noise, measurement model/noise,
    initial distribution, delay acceptance, integration step and history
    handling are taken directly from the frozen :class:`DelayAwareIMM`.
    There is no arm-specific tuning.
    """

    def __init__(self, config: IMMConfig | str | Path) -> None:
        self.config = config if isinstance(config, IMMConfig) else IMMConfig.from_yaml(config)
        self.H = np.zeros((MEASUREMENT_DIM, STATE_DIM))
        self.H[:6, :6] = np.eye(6)
        self.R = np.diag(
            [self.config.position_measurement_std_m**2] * 3
            + [self.config.velocity_measurement_std_mps**2] * 3
        )
        self.reset()

    def reset(self, timestamp_s: float = 0.0) -> None:
        self.current_time_s = float(timestamp_s)
        self.mean = np.zeros(STATE_DIM, dtype=float)
        diagonal = np.array(
            [self.config.initial_position_std_m**2] * 3
            + [self.config.initial_velocity_std_mps**2] * 3
            + [self.config.initial_acceleration_std_mps2**2] * 3,
            dtype=float,
        )
        self.covariance = np.diag(diagonal)
        self.history: list[_Snapshot] = [self._snapshot()]
        self.last_arrival_timestamp_s = float(timestamp_s)
        self.accepted_updates = 0
        self.dropped_packets = 0
        self.rejected_packets = 0
        self.last_update_source_timestamp_s = math.nan
        self.last_repropagation_steps = 0
        self.last_innovation = np.full(MEASUREMENT_DIM, np.nan)
        self.last_innovation_covariance = np.full(
            (MEASUREMENT_DIM, MEASUREMENT_DIM), np.nan
        )
        self.last_nis = math.nan

    def _snapshot(self) -> _Snapshot:
        return _Snapshot(self.current_time_s, self.mean.copy(), self.covariance.copy())

    @staticmethod
    def state_transition(dt_s: float, config: IMMConfig) -> np.ndarray:
        return DelayAwareIMM.state_transition("ca", dt_s, config)

    def process_covariance(self, dt_s: float) -> np.ndarray:
        qp, qv, qa = self.config.process_noise["ca"]
        dt = max(float(dt_s), 0.0)
        return np.diag([qp * dt] * 3 + [qv * dt] * 3 + [qa * dt] * 3)

    def _predict_once(self, dt_s: float) -> None:
        if dt_s <= 0.0:
            return
        F = self.state_transition(dt_s, self.config)
        self.mean = F @ self.mean
        self.covariance = _symmetrize_psd(
            F @ self.covariance @ F.T + self.process_covariance(dt_s)
        )
        self.current_time_s += dt_s

    def _trim_history(self) -> None:
        cutoff = self.current_time_s - self.config.history_buffer_duration_s
        while len(self.history) > 1 and self.history[1].timestamp_s < cutoff - 1e-12:
            self.history.pop(0)

    def predict_to(self, timestamp_s: float, record_history: bool = True) -> None:
        target = float(timestamp_s)
        if target < self.current_time_s - 1e-12:
            raise ValueError("filter time cannot move backward")
        while self.current_time_s < target - 1e-12:
            dt = min(self.config.nominal_dt_s, target - self.current_time_s)
            self._predict_once(dt)
            if record_history:
                self.history.append(self._snapshot())
        if record_history:
            self._trim_history()

    def _measurement_update(self, measurement: np.ndarray) -> None:
        innovation = np.asarray(measurement, dtype=float) - self.H @ self.mean
        innovation_covariance = _symmetrize_psd(
            self.H @ self.covariance @ self.H.T + self.R
        )
        gain = np.linalg.solve(innovation_covariance, self.H @ self.covariance).T
        identity = np.eye(STATE_DIM)
        self.mean = self.mean + gain @ innovation
        joseph_left = identity - gain @ self.H
        self.covariance = _symmetrize_psd(
            joseph_left @ self.covariance @ joseph_left.T + gain @ self.R @ gain.T
        )
        self.last_innovation = innovation.copy()
        self.last_innovation_covariance = innovation_covariance.copy()
        self.last_nis = float(
            innovation @ np.linalg.solve(innovation_covariance, innovation)
        )

    def process_packet(self, packet: TelemetryPacket) -> None:
        arrival = float(packet.arrival_timestamp_s)
        source = float(packet.source_timestamp_s)
        if arrival < self.last_arrival_timestamp_s - 1e-12:
            raise ValueError("arrival timestamps cannot move backward")
        if source > arrival + 1e-12:
            raise ValueError("source timestamp cannot be after arrival")
        self.predict_to(arrival)
        self.last_arrival_timestamp_s = arrival
        if packet.drop or not packet.valid:
            self.dropped_packets += 1
            return
        if arrival - source > self.config.maximum_accepted_delay_s + 1e-12:
            self.rejected_packets += 1
            return
        if source < self.history[0].timestamp_s - 1e-12:
            self.rejected_packets += 1
            return

        current_time = self.current_time_s
        future_times = [
            snapshot.timestamp_s
            for snapshot in self.history
            if snapshot.timestamp_s > source + 1e-12
        ]
        base_index = max(
            index
            for index, snapshot in enumerate(self.history)
            if snapshot.timestamp_s <= source + 1e-12
        )
        base = self.history[base_index].clone()
        self.current_time_s = base.timestamp_s
        self.mean = base.mean
        self.covariance = base.covariance
        if self.current_time_s < source - 1e-12:
            self._predict_once(source - self.current_time_s)
        self._measurement_update(packet.measurement)
        replacement = [snapshot.clone() for snapshot in self.history[:base_index]]
        replacement.append(self._snapshot())
        self.last_repropagation_steps = 0
        for timestamp in future_times:
            if timestamp <= self.current_time_s + 1e-12:
                continue
            self._predict_once(timestamp - self.current_time_s)
            replacement.append(self._snapshot())
            self.last_repropagation_steps += 1
        if self.current_time_s < current_time - 1e-12:
            self._predict_once(current_time - self.current_time_s)
            replacement.append(self._snapshot())
            self.last_repropagation_steps += 1
        self.history = replacement
        self._trim_history()
        self.accepted_updates += 1
        self.last_update_source_timestamp_s = source

    def state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.mean.copy(), self.covariance.copy()

    def mixture_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Compatibility alias for common A1/A2 replay and logging code."""
        return self.state()

    def forecast_mean(self, horizon_s: float) -> np.ndarray:
        horizon = float(horizon_s)
        if horizon < 0.0:
            raise ValueError("forecast horizon cannot be negative")
        F = self.state_transition(horizon, self.config)
        return F @ self.mean

    def diagnostics(self) -> dict[str, Any]:
        return {
            "mode_probabilities": np.array([0.0, 1.0, 0.0]),
            "position_covariance_trace": float(np.trace(self.covariance[:3, :3])),
            "velocity_covariance_trace": float(np.trace(self.covariance[3:6, 3:6])),
            "last_update_source_timestamp_s": self.last_update_source_timestamp_s,
            "last_repropagation_steps": self.last_repropagation_steps,
            "history_size": len(self.history),
            "last_nis": self.last_nis,
        }
