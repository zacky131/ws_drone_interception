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
    theta_deg = float(bounds.get("orientation_deg", bounds.get("heading_deg", 0.0)))
    if abs(theta_deg) > 1e-4:
        theta = np.radians(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        xf = east * cos_t + north * sin_t
        yf = -east * sin_t + north * cos_t
    else:
        xf, yf = east, north
    e_min = bounds.get("length_min_m", bounds.get("east_min_m", -40.0))
    e_max = bounds.get("length_max_m", bounds.get("east_max_m", 40.0))
    n_min = bounds.get("width_min_m", bounds.get("north_min_m", -22.0))
    n_max = bounds.get("width_max_m", bounds.get("north_max_m", 22.0))
    return bool(e_min <= xf <= e_max and n_min <= yf <= n_max)


def inside_altitude(position: Iterable[float], bounds: dict[str, float]) -> bool:
    return bool(bounds["min_m"] <= vector3(position)[2] <= bounds["max_m"])


def rectangle_points(bounds: dict[str, float], altitude_m: float = 0.05) -> np.ndarray:
    theta_deg = float(bounds.get("orientation_deg", bounds.get("heading_deg", 0.0)))
    e0 = bounds.get("length_min_m", bounds.get("east_min_m", -40.0))
    e1 = bounds.get("length_max_m", bounds.get("east_max_m", 40.0))
    n0 = bounds.get("width_min_m", bounds.get("north_min_m", -22.0))
    n1 = bounds.get("width_max_m", bounds.get("north_max_m", 22.0))
    corners = np.array([
        [e0, n0],
        [e1, n0],
        [e1, n1],
        [e0, n1],
        [e0, n0],
    ], dtype=float)
    if abs(theta_deg) > 1e-4:
        theta = np.radians(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        corners = (rot @ corners.T).T
    return np.column_stack([corners, np.full(len(corners), altitude_m)])

