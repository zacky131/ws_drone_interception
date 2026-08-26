"""Single authoritative conversion between PX4 NED and evaluation ENU."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _vector3(value: Iterable[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"expected a 3-vector, got shape {vector.shape}")
    return vector


def _swap_ned_enu(value: Iterable[float]) -> np.ndarray:
    x, y, z = _vector3(value)
    return np.array([y, x, -z], dtype=float)


def ned_position_to_enu(value: Iterable[float]) -> np.ndarray:
    return _swap_ned_enu(value)


def enu_position_to_ned(value: Iterable[float]) -> np.ndarray:
    return _swap_ned_enu(value)


def ned_velocity_to_enu(value: Iterable[float]) -> np.ndarray:
    return _swap_ned_enu(value)


def enu_velocity_to_ned(value: Iterable[float]) -> np.ndarray:
    return _swap_ned_enu(value)


def ned_acceleration_to_enu(value: Iterable[float]) -> np.ndarray:
    return _swap_ned_enu(value)


def enu_acceleration_to_ned(value: Iterable[float]) -> np.ndarray:
    return _swap_ned_enu(value)

