"""CSV trajectory loading, rigid adaptation, and control-rate interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


_VECTORS = {
    "position": ("pos_x", "pos_y", "pos_z"),
    "velocity": ("vel_x", "vel_y", "vel_z"),
    "acceleration": ("acc_x", "acc_y", "acc_z"),
    "jerk": ("jerk_x", "jerk_y", "jerk_z"),
}


@dataclass(frozen=True)
class TrajectorySample:
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray


class Trajectory:
    def __init__(self, times: np.ndarray, vectors: dict[str, np.ndarray]) -> None:
        self.times = np.asarray(times, dtype=float)
        if self.times.ndim != 1 or len(self.times) < 2 or np.any(np.diff(self.times) <= 0.0):
            raise ValueError("trajectory time must be a strictly increasing 1-D array")
        self.times = self.times - self.times[0]
        self.vectors = {name: np.asarray(value, dtype=float) for name, value in vectors.items()}
        for name, value in self.vectors.items():
            if value.shape != (len(self.times), 3):
                raise ValueError(f"{name} has invalid shape {value.shape}")

    @classmethod
    def from_csv(cls, path: str | Path) -> "Trajectory":
        frame = pd.read_csv(path, comment="#")
        if "time" not in frame:
            raise ValueError("trajectory CSV is missing time")
        vectors: dict[str, np.ndarray] = {}
        for name, columns in _VECTORS.items():
            if all(column in frame for column in columns):
                vectors[name] = frame[list(columns)].to_numpy(dtype=float)
        if "position" not in vectors or "velocity" not in vectors:
            raise ValueError("trajectory CSV requires position and velocity")
        times = frame["time"].to_numpy(dtype=float)
        if "acceleration" not in vectors:
            vectors["acceleration"] = np.gradient(vectors["velocity"], times, axis=0)
        if "jerk" not in vectors:
            vectors["jerk"] = np.gradient(vectors["acceleration"], times, axis=0)
        return cls(times, vectors)

    @property
    def duration_s(self) -> float:
        return float(self.times[-1])

    def adapted(
        self,
        translation_enu: Iterable[float] = (0.0, 0.0, 0.0),
        horizontal_rotation_rad: float = 0.0,
    ) -> "Trajectory":
        translation = np.asarray(translation_enu, dtype=float)
        if translation.shape != (3,):
            raise ValueError("translation must be a 3-vector")
        c, s = np.cos(horizontal_rotation_rad), np.sin(horizontal_rotation_rad)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        vectors = {name: values @ rotation.T for name, values in self.vectors.items()}
        vectors["position"] = vectors["position"] + translation
        return Trajectory(self.times.copy(), vectors)

    def sample(self, time_s: float) -> TrajectorySample:
        t = float(np.clip(time_s, 0.0, self.times[-1]))

        def interpolate(name: str) -> np.ndarray:
            values = self.vectors[name]
            return np.array([np.interp(t, self.times, values[:, axis]) for axis in range(3)])

        return TrajectorySample(
            position=interpolate("position"),
            velocity=interpolate("velocity"),
            acceleration=interpolate("acceleration"),
            jerk=interpolate("jerk"),
        )

