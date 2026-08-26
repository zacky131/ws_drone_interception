"""Virtual-capture logic using actual PX4 states."""

from __future__ import annotations

import numpy as np


CAPTURE_RADIUS_M = 1.0


def actual_separation(interceptor_actual_enu: np.ndarray, target_actual_enu: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(interceptor_actual_enu) - np.asarray(target_actual_enu)))


def is_capture(interceptor_actual_enu: np.ndarray, target_actual_enu: np.ndarray) -> bool:
    return actual_separation(interceptor_actual_enu, target_actual_enu) <= CAPTURE_RADIUS_M


def interpolated_crossing_time(t0: float, d0: float, t1: float, d1: float) -> float | None:
    if d0 <= CAPTURE_RADIUS_M:
        return float(t0)
    if d1 > CAPTURE_RADIUS_M or d1 == d0:
        return None
    fraction = (d0 - CAPTURE_RADIUS_M) / (d0 - d1)
    return float(t0 + fraction * (t1 - t0))

