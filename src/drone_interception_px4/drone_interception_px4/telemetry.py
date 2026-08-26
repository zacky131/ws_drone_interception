"""Pre-generated deterministic and pairable target telemetry impairment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


SCHEDULE_COLUMNS = [
    "timestamp", "delay_s", "drop",
    "position_noise_x", "position_noise_y", "position_noise_z",
    "velocity_noise_x", "velocity_noise_y", "velocity_noise_z",
]


@dataclass(frozen=True)
class ImpairmentCondition:
    name: str
    delay_s: float
    position_sigma_m: float
    velocity_sigma_mps: float
    dropout_probability: float
    world: str


@dataclass(frozen=True)
class TargetStateSample:
    """Executed target state associated with an explicit control timestamp."""

    timestamp_s: float
    position_enu: np.ndarray
    velocity_enu: np.ndarray

    def __post_init__(self) -> None:
        timestamp = float(self.timestamp_s)
        position = np.asarray(self.position_enu, dtype=float).copy()
        velocity = np.asarray(self.velocity_enu, dtype=float).copy()
        if not np.isfinite(timestamp):
            raise ValueError("target-state timestamp must be finite")
        if position.shape != (3,) or velocity.shape != (3,):
            raise ValueError("target-state position and velocity must have shape (3,)")
        position.setflags(write=False)
        velocity.setflags(write=False)
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "position_enu", position)
        object.__setattr__(self, "velocity_enu", velocity)

    @property
    def state(self) -> np.ndarray:
        return np.concatenate([self.position_enu, self.velocity_enu])


@dataclass(frozen=True)
class TelemetryEvent:
    """One impaired packet and the physical timestamps that define it."""

    measurement: np.ndarray | None
    noise: np.ndarray
    configured_delay_s: float
    arrival_timestamp_s: float
    requested_source_timestamp_s: float
    actual_source_timestamp_s: float
    history_left_timestamp_s: float
    history_right_timestamp_s: float
    interpolation_alpha: float
    startup_clamped: bool
    source_truth_position_enu: np.ndarray
    source_truth_velocity_enu: np.ndarray
    drop: bool

    def __post_init__(self) -> None:
        measurement = None if self.measurement is None else np.asarray(self.measurement, dtype=float).copy()
        noise = np.asarray(self.noise, dtype=float).copy()
        position = np.asarray(self.source_truth_position_enu, dtype=float).copy()
        velocity = np.asarray(self.source_truth_velocity_enu, dtype=float).copy()
        if measurement is not None and measurement.shape != (6,):
            raise ValueError("telemetry measurement must have shape (6,)")
        if noise.shape != (6,) or position.shape != (3,) or velocity.shape != (3,):
            raise ValueError("invalid telemetry event vector shape")
        for value in (measurement, noise, position, velocity):
            if value is not None:
                value.setflags(write=False)
        object.__setattr__(self, "measurement", measurement)
        object.__setattr__(self, "noise", noise)
        object.__setattr__(self, "source_truth_position_enu", position)
        object.__setattr__(self, "source_truth_velocity_enu", velocity)

    @property
    def physical_measurement_age_s(self) -> float:
        return float(self.arrival_timestamp_s - self.actual_source_timestamp_s)

    @property
    def valid(self) -> bool:
        return not self.drop and self.measurement is not None


CONDITIONS = {
    "C0": ImpairmentCondition("C0", 0.050, 0.05, 0.10, 0.00, "default"),
    "C1": ImpairmentCondition("C1", 0.080, 0.10, 0.20, 0.05, "default"),
    "C2": ImpairmentCondition("C2", 0.120, 0.15, 0.30, 0.10, "windy"),
}


def generate_schedule(condition: ImpairmentCondition, duration_s: float, dt_s: float, seed: int) -> pd.DataFrame:
    if dt_s <= 0.0 or duration_s < 0.0:
        raise ValueError("invalid schedule timing")
    timestamps = np.arange(0.0, duration_s + 0.5 * dt_s, dt_s)
    rng = np.random.default_rng(int(seed))
    position_noise = rng.normal(0.0, condition.position_sigma_m, (len(timestamps), 3))
    velocity_noise = rng.normal(0.0, condition.velocity_sigma_mps, (len(timestamps), 3))
    drop = rng.random(len(timestamps)) < condition.dropout_probability
    return pd.DataFrame({
        "timestamp": timestamps,
        "delay_s": np.full(len(timestamps), condition.delay_s),
        "drop": drop.astype(np.uint8),
        "position_noise_x": position_noise[:, 0],
        "position_noise_y": position_noise[:, 1],
        "position_noise_z": position_noise[:, 2],
        "velocity_noise_x": velocity_noise[:, 0],
        "velocity_noise_y": velocity_noise[:, 1],
        "velocity_noise_z": velocity_noise[:, 2],
    }, columns=SCHEDULE_COLUMNS)


def canonical_csv_bytes(schedule: pd.DataFrame) -> bytes:
    return schedule[SCHEDULE_COLUMNS].to_csv(index=False, float_format="%.17g", lineterminator="\n").encode()


def schedule_sha256(schedule: pd.DataFrame) -> str:
    return hashlib.sha256(canonical_csv_bytes(schedule)).hexdigest()


def save_schedule(schedule: pd.DataFrame, path: str | Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_csv_bytes(schedule)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _source_state(
    true_history: Sequence[TargetStateSample], requested_source_timestamp_s: float
) -> tuple[np.ndarray, float, float, float, float, bool]:
    """Return source state and interpolation metadata without extrapolation."""

    if not true_history:
        raise ValueError("true_history must contain at least one timestamped sample")
    timestamps = np.asarray([sample.timestamp_s for sample in true_history], dtype=float)
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("true_history timestamps must be strictly increasing")
    requested = float(requested_source_timestamp_s)
    tolerance = 1e-12
    if requested < timestamps[0] - tolerance:
        oldest = true_history[0]
        return oldest.state, oldest.timestamp_s, oldest.timestamp_s, oldest.timestamp_s, 0.0, True
    if requested > timestamps[-1] + tolerance:
        raise ValueError("requested source time is newer than available target history")

    right_index = int(np.searchsorted(timestamps, requested, side="left"))
    if right_index < len(timestamps) and abs(timestamps[right_index] - requested) <= tolerance:
        sample = true_history[right_index]
        return sample.state, requested, sample.timestamp_s, sample.timestamp_s, 0.0, False
    if right_index == 0:
        sample = true_history[0]
        return sample.state, requested, sample.timestamp_s, sample.timestamp_s, 0.0, False
    if right_index >= len(true_history):
        sample = true_history[-1]
        return sample.state, requested, sample.timestamp_s, sample.timestamp_s, 0.0, False

    left = true_history[right_index - 1]
    right = true_history[right_index]
    alpha = (requested - left.timestamp_s) / (right.timestamp_s - left.timestamp_s)
    state = (1.0 - alpha) * left.state + alpha * right.state
    return state, requested, left.timestamp_s, right.timestamp_s, float(alpha), False


def apply_schedule_row(
    true_history: Sequence[TargetStateSample],
    row: pd.Series,
    arrival_timestamp_s: float,
) -> TelemetryEvent:
    """Generate telemetry at its exact represented physical source time."""

    arrival = float(arrival_timestamp_s)
    configured_delay = float(row["delay_s"])
    requested_source = arrival - configured_delay
    state, actual_source, left_time, right_time, alpha, startup_clamped = _source_state(
        true_history, requested_source
    )
    noise = row[[
        "position_noise_x", "position_noise_y", "position_noise_z",
        "velocity_noise_x", "velocity_noise_y", "velocity_noise_z",
    ]].to_numpy(dtype=float)
    dropped = bool(row["drop"])
    measurement = None if dropped else state + noise
    return TelemetryEvent(
        measurement=measurement,
        noise=noise,
        configured_delay_s=configured_delay,
        arrival_timestamp_s=arrival,
        requested_source_timestamp_s=requested_source,
        actual_source_timestamp_s=actual_source,
        history_left_timestamp_s=left_time,
        history_right_timestamp_s=right_time,
        interpolation_alpha=alpha,
        startup_clamped=startup_clamped,
        source_truth_position_enu=state[:3],
        source_truth_velocity_enu=state[3:6],
        drop=dropped,
    )


def measurement_from_event(value: TelemetryEvent | np.ndarray | None) -> np.ndarray | None:
    """Unwrap timestamped telemetry for legacy controllers that do not use time."""

    if isinstance(value, TelemetryEvent):
        return value.measurement
    return None if value is None else np.asarray(value, dtype=float)
