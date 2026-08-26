"""Experiment ENU geometry and PX4 NED boundary conversions."""

from __future__ import annotations

from collections.abc import Iterable
import numpy as np


def vector3(value: Iterable[float]) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (3,):
        raise ValueError(f"expected shape (3,), got {result.shape}")
    return result


def ned_to_enu(value: Iterable[float]) -> np.ndarray:
    north, east, down = vector3(value)
    return np.array([east, north, -down], dtype=float)


def enu_to_ned(value: Iterable[float]) -> np.ndarray:
    east, north, up = vector3(value)
    return np.array([north, east, -up], dtype=float)


def distance(a: Iterable[float], b: Iterable[float]) -> float:
    return float(np.linalg.norm(vector3(a) - vector3(b)))


def inside_horizontal_box(position: Iterable[float], bounds: dict[str, float]) -> bool:
    east, north, _ = vector3(position)
    return bool(bounds["east_min_m"] <= east <= bounds["east_max_m"] and bounds["north_min_m"] <= north <= bounds["north_max_m"])


def inside_altitude(position: Iterable[float], bounds: dict[str, float]) -> bool:
    return bool(bounds["min_m"] <= vector3(position)[2] <= bounds["max_m"])


def rectangle_points(bounds: dict[str, float], altitude_m: float = 0.05) -> np.ndarray:
    e0, e1 = bounds["east_min_m"], bounds["east_max_m"]
    n0, n1 = bounds["north_min_m"], bounds["north_max_m"]
    return np.array([[e0, n0, altitude_m], [e1, n0, altitude_m], [e1, n1, altitude_m], [e0, n1, altitude_m], [e0, n0, altitude_m]], dtype=float)
