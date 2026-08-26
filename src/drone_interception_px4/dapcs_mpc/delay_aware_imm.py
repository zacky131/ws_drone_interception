"""Timestamp-aware out-of-sequence-measurement IMM target estimator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


MODE_NAMES = ("cv", "ca", "singer")
STATE_DIM = 9
MEASUREMENT_DIM = 6


@dataclass(frozen=True)
class TelemetryPacket:
    source_timestamp_s: float
    arrival_timestamp_s: float
    position: np.ndarray
    velocity: np.ndarray
    valid: bool = True
    drop: bool = False

    @property
    def measurement(self) -> np.ndarray:
        return np.concatenate([
            np.asarray(self.position, dtype=float),
            np.asarray(self.velocity, dtype=float),
        ])


@dataclass(frozen=True)
class IMMConfig:
    initial_mode_probabilities: np.ndarray
    transition_matrix: np.ndarray
    process_noise: dict[str, tuple[float, float, float]]
    singer_time_constant_s: float
    cv_acceleration_time_constant_s: float
    position_measurement_std_m: float
    velocity_measurement_std_mps: float
    initial_position_std_m: float
    initial_velocity_std_mps: float
    initial_acceleration_std_mps2: float
    history_buffer_duration_s: float
    maximum_accepted_delay_s: float
    nominal_dt_s: float
    dropout_propagation_rule: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> "IMMConfig":
        data: dict[str, Any] = yaml.safe_load(Path(path).read_text())
        probs = np.asarray(data["initial_mode_probabilities"], dtype=float)
        transition = np.asarray(data["mode_transition_matrix"], dtype=float)
        process = {
            name: (
                float(data["process_noise"][name]["position"]),
                float(data["process_noise"][name]["velocity"]),
                float(data["process_noise"][name]["acceleration"]),
            )
            for name in MODE_NAMES
        }
        measurement = data["measurement_noise"]
        initial = data["initial_uncertainty"]
        config = cls(
            initial_mode_probabilities=probs,
            transition_matrix=transition,
            process_noise=process,
            singer_time_constant_s=float(data["singer_time_constant_s"]),
            cv_acceleration_time_constant_s=float(data["cv_acceleration_time_constant_s"]),
            position_measurement_std_m=float(measurement["position_std_m"]),
            velocity_measurement_std_mps=float(measurement["velocity_std_mps"]),
            initial_position_std_m=float(initial["position_std_m"]),
            initial_velocity_std_mps=float(initial["velocity_std_mps"]),
            initial_acceleration_std_mps2=float(initial["acceleration_std_mps2"]),
            history_buffer_duration_s=float(data["history_buffer_duration_s"]),
            maximum_accepted_delay_s=float(data["maximum_accepted_delay_s"]),
            nominal_dt_s=float(data["nominal_dt_s"]),
            dropout_propagation_rule=str(data["dropout_propagation_rule"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.initial_mode_probabilities.shape != (3,):
            raise ValueError("three initial mode probabilities are required")
        if self.transition_matrix.shape != (3, 3):
            raise ValueError("mode transition matrix must be 3x3")
        if not np.isclose(self.initial_mode_probabilities.sum(), 1.0):
            raise ValueError("initial probabilities must sum to one")
        if not np.allclose(self.transition_matrix.sum(axis=1), 1.0):
            raise ValueError("each transition-matrix row must sum to one")
        if np.any(self.initial_mode_probabilities < 0.0) or np.any(self.transition_matrix < 0.0):
            raise ValueError("IMM probabilities must be nonnegative")
        if self.dropout_propagation_rule != "prediction_only":
            raise ValueError("only causal prediction-only dropout handling is supported")


@dataclass
class _Snapshot:
    timestamp_s: float
    means: np.ndarray
    covariances: np.ndarray
    probabilities: np.ndarray

    def clone(self) -> "_Snapshot":
        return _Snapshot(
            self.timestamp_s,
            self.means.copy(),
            self.covariances.copy(),
            self.probabilities.copy(),
        )


def _symmetrize_psd(matrix: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, floor)
    return (vectors * values) @ vectors.T


class DelayAwareIMM:
    """Three-model CV/CA/Singer IMM with buffered delayed updates.

    Every accepted packet is updated at its source timestamp. Buffered mode
    states are then deterministically repropagated to the latest arrival time.
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
        self.means = np.zeros((3, STATE_DIM), dtype=float)
        diagonal = np.array(
            [self.config.initial_position_std_m**2] * 3
            + [self.config.initial_velocity_std_mps**2] * 3
            + [self.config.initial_acceleration_std_mps2**2] * 3,
            dtype=float,
        )
        self.covariances = np.repeat(np.diag(diagonal)[None, :, :], 3, axis=0)
        self.probabilities = self.config.initial_mode_probabilities.copy()
        self.history: list[_Snapshot] = [self._snapshot()]
        self.last_arrival_timestamp_s = float(timestamp_s)
        self.accepted_updates = 0
        self.dropped_packets = 0
        self.rejected_packets = 0
        self.last_update_source_timestamp_s = math.nan
        self.last_repropagation_steps = 0

    def _snapshot(self) -> _Snapshot:
        return _Snapshot(
            self.current_time_s,
            self.means.copy(),
            self.covariances.copy(),
            self.probabilities.copy(),
        )

    @staticmethod
    def state_transition(mode: str, dt_s: float, config: IMMConfig) -> np.ndarray:
        dt = float(dt_s)
        if dt < 0.0:
            raise ValueError("prediction dt cannot be negative")
        F = np.eye(STATE_DIM)
        for axis in range(3):
            p, v, a = axis, axis + 3, axis + 6
            if mode == "ca":
                phi = 1.0
                velocity_gain = dt
                position_gain = 0.5 * dt * dt
            else:
                tau = (
                    config.cv_acceleration_time_constant_s
                    if mode == "cv"
                    else config.singer_time_constant_s
                )
                phi = math.exp(-dt / tau) if dt > 0.0 else 1.0
                velocity_gain = tau * (1.0 - phi)
                position_gain = tau * dt - tau * tau * (1.0 - phi)
            F[p, v] = dt
            F[p, a] = position_gain
            F[v, a] = velocity_gain
            F[a, a] = phi
        return F

    def process_covariance(self, mode: str, dt_s: float) -> np.ndarray:
        qp, qv, qa = self.config.process_noise[mode]
        dt = max(float(dt_s), 0.0)
        return np.diag([qp * dt] * 3 + [qv * dt] * 3 + [qa * dt] * 3)

    def _mix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        transition = self.config.transition_matrix
        predicted_probabilities = self.probabilities @ transition
        predicted_probabilities = np.maximum(predicted_probabilities, 1e-15)
        mixing = self.probabilities[:, None] * transition / predicted_probabilities[None, :]
        mixed_means = np.zeros_like(self.means)
        mixed_covariances = np.zeros_like(self.covariances)
        for destination in range(3):
            weights = mixing[:, destination]
            mean = np.sum(weights[:, None] * self.means, axis=0)
            covariance = np.zeros((STATE_DIM, STATE_DIM))
            for source in range(3):
                delta = self.means[source] - mean
                covariance += weights[source] * (
                    self.covariances[source] + np.outer(delta, delta)
                )
            mixed_means[destination] = mean
            mixed_covariances[destination] = _symmetrize_psd(covariance)
        return mixed_means, mixed_covariances, predicted_probabilities

    def _predict_once(self, dt_s: float) -> None:
        if dt_s <= 0.0:
            return
        mixed_means, mixed_covariances, probabilities = self._mix()
        for index, mode in enumerate(MODE_NAMES):
            F = self.state_transition(mode, dt_s, self.config)
            self.means[index] = F @ mixed_means[index]
            self.covariances[index] = _symmetrize_psd(
                F @ mixed_covariances[index] @ F.T + self.process_covariance(mode, dt_s)
            )
        self.probabilities = probabilities / probabilities.sum()
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
        log_likelihoods = np.zeros(3)
        identity = np.eye(STATE_DIM)
        for index in range(3):
            mean = self.means[index]
            covariance = self.covariances[index]
            innovation = measurement - self.H @ mean
            innovation_covariance = _symmetrize_psd(self.H @ covariance @ self.H.T + self.R)
            gain = np.linalg.solve(innovation_covariance, self.H @ covariance).T
            self.means[index] = mean + gain @ innovation
            joseph_left = identity - gain @ self.H
            self.covariances[index] = _symmetrize_psd(
                joseph_left @ covariance @ joseph_left.T + gain @ self.R @ gain.T
            )
            sign, logdet = np.linalg.slogdet(innovation_covariance)
            quadratic = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
            log_likelihoods[index] = (
                -0.5 * (MEASUREMENT_DIM * math.log(2.0 * math.pi) + logdet + quadratic)
                if sign > 0
                else -1e12
            )
        shifted = log_likelihoods - np.max(log_likelihoods)
        weights = self.probabilities * np.exp(np.clip(shifted, -700.0, 0.0))
        total = float(weights.sum())
        self.probabilities = (
            weights / total if total > 1e-300 else np.full(3, 1.0 / 3.0)
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
            snapshot.timestamp_s for snapshot in self.history if snapshot.timestamp_s > source + 1e-12
        ]
        base_index = max(
            index for index, snapshot in enumerate(self.history)
            if snapshot.timestamp_s <= source + 1e-12
        )
        base = self.history[base_index].clone()
        self.current_time_s = base.timestamp_s
        self.means = base.means
        self.covariances = base.covariances
        self.probabilities = base.probabilities
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

    def mixture_state(self) -> tuple[np.ndarray, np.ndarray]:
        mean = np.sum(self.probabilities[:, None] * self.means, axis=0)
        covariance = np.zeros((STATE_DIM, STATE_DIM))
        for index in range(3):
            delta = self.means[index] - mean
            covariance += self.probabilities[index] * (
                self.covariances[index] + np.outer(delta, delta)
            )
        return mean, _symmetrize_psd(covariance)

    def forecast_mixture_mean(self, horizon_s: float) -> np.ndarray:
        """Return a causal future mixture mean without mutating the filter."""
        remaining = float(horizon_s)
        if remaining < 0.0:
            raise ValueError("forecast horizon cannot be negative")
        means = self.means.copy()
        covariances = self.covariances.copy()
        probabilities = self.probabilities.copy()
        while remaining > 1e-12:
            dt = min(self.config.nominal_dt_s, remaining)
            predicted_probabilities = probabilities @ self.config.transition_matrix
            predicted_probabilities = np.maximum(predicted_probabilities, 1e-15)
            mixing = probabilities[:, None] * self.config.transition_matrix / predicted_probabilities[None, :]
            new_means = np.zeros_like(means)
            new_covariances = np.zeros_like(covariances)
            for destination, mode in enumerate(MODE_NAMES):
                weights = mixing[:, destination]
                mixed_mean = np.sum(weights[:, None] * means, axis=0)
                mixed_covariance = np.zeros((STATE_DIM, STATE_DIM))
                for source in range(3):
                    delta = means[source] - mixed_mean
                    mixed_covariance += weights[source] * (
                        covariances[source] + np.outer(delta, delta)
                    )
                F = self.state_transition(mode, dt, self.config)
                new_means[destination] = F @ mixed_mean
                new_covariances[destination] = _symmetrize_psd(
                    F @ mixed_covariance @ F.T + self.process_covariance(mode, dt)
                )
            means = new_means
            covariances = new_covariances
            probabilities = predicted_probabilities / predicted_probabilities.sum()
            remaining -= dt
        return np.sum(probabilities[:, None] * means, axis=0)

    def diagnostics(self) -> dict[str, Any]:
        _, covariance = self.mixture_state()
        return {
            "mode_probabilities": self.probabilities.copy(),
            "position_covariance_trace": float(np.trace(covariance[:3, :3])),
            "velocity_covariance_trace": float(np.trace(covariance[3:6, 3:6])),
            "last_update_source_timestamp_s": self.last_update_source_timestamp_s,
            "last_repropagation_steps": self.last_repropagation_steps,
            "history_size": len(self.history),
        }
