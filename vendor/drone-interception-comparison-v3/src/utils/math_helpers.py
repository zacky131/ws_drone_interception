"""
Lightweight math utilities used across modules.
"""

from __future__ import annotations

import numpy as np


def safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return unit vector; returns zeros if *v* has near-zero norm."""
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def clip_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    """Clip the Euclidean norm of *v* to *max_norm*."""
    n = np.linalg.norm(v)
    if n > max_norm and n > 1e-12:
        return v * (max_norm / n)
    return v.copy()


def clip_components(v: np.ndarray, max_val: float) -> np.ndarray:
    """Clip each component of *v* to [-max_val, max_val]."""
    return np.clip(v, -max_val, max_val)


def skew(v: np.ndarray) -> np.ndarray:
    """Return the 3×3 skew-symmetric matrix of *v*."""
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])


def closing_velocity(r: np.ndarray, v_rel: np.ndarray) -> float:
    """Scalar closing velocity (positive when range is decreasing)."""
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-12:
        return 0.0
    return -np.dot(v_rel, r / r_norm)


def los_rate(r: np.ndarray, v_rel: np.ndarray) -> np.ndarray:
    """Line-of-sight angular rate vector ω = (r × v_rel) / |r|²."""
    r_sq = np.dot(r, r)
    if r_sq < 1e-12:
        return np.zeros(3)
    return np.cross(r, v_rel) / r_sq
