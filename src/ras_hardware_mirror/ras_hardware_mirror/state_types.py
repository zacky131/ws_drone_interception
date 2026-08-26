"""Common ENU scientific state, independent of Gazebo or future RTK transport."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


def _v3(value) -> np.ndarray:
    vector = np.asarray(value, dtype=float).copy()
    if vector.shape != (3,):
        raise ValueError("expected a finite three-vector")
    return vector


@dataclass
class NavigationState:
    timestamp_s: float
    position_enu: np.ndarray
    velocity_enu: np.ndarray
    covariance: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    quality: str = "unknown"
    valid: bool = True
    source: str = "unknown"

    def __post_init__(self) -> None:
        self.timestamp_s = float(self.timestamp_s)
        self.position_enu = _v3(self.position_enu)
        self.velocity_enu = _v3(self.velocity_enu)
        self.covariance = np.asarray(self.covariance, dtype=float).reshape(6, 6)
        self.valid = bool(self.valid and np.isfinite(self.timestamp_s) and np.all(np.isfinite(self.position_enu)) and np.all(np.isfinite(self.velocity_enu)))

    @property
    def state6(self) -> np.ndarray:
        return np.concatenate((self.position_enu, self.velocity_enu))


@dataclass(frozen=True)
class TargetKinematics:
    position_enu: np.ndarray
    velocity_enu: np.ndarray
    acceleration_enu: np.ndarray


class ExperimentPhase(str, Enum):
    PRECHECK = "PRECHECK"
    TAKEOFF = "TAKEOFF"
    STABILIZE = "STABILIZE"
    RUN = "RUN"
    CAPTURE = "CAPTURE"
    HOLD = "HOLD"
    ABORT = "ABORT"
    LAND = "LAND"
    DONE = "DONE"


@dataclass(frozen=True)
class SafetyInputs:
    now_s: float
    run_start_s: float | None
    interceptor: NavigationState | None
    target: NavigationState | None
    controller_age_s: float
    command_finite: bool
    px4_healthy: bool
    manual_abort: bool = False
