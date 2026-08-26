"""Kinematic capture-opportunity proxy (not a formal reachable set)."""

from __future__ import annotations

import numpy as np


def kinematic_reachable_distance(
    pursuer_position: np.ndarray,
    pursuer_velocity: np.ndarray,
    target_position: np.ndarray,
    vmax: float,
    amax: float,
    tau_s: float,
) -> float:
    """Conservative radial distance using only nonnegative velocity toward target.

    Lateral or adverse velocity is not credited. Acceleration is assumed usable
    along the target ray until the physical speed bound is reached.
    """
    delta = np.asarray(target_position, dtype=float) - np.asarray(pursuer_position, dtype=float)
    distance = float(np.linalg.norm(delta))
    direction = delta / distance if distance > 1e-12 else np.zeros(3)
    toward_speed = max(0.0, float(np.dot(np.asarray(pursuer_velocity, dtype=float), direction)))
    toward_speed = min(toward_speed, float(vmax))
    duration = max(0.0, float(tau_s))
    if amax <= 0.0 or toward_speed >= vmax:
        return toward_speed * duration
    acceleration_time = (vmax - toward_speed) / float(amax)
    if duration <= acceleration_time:
        return toward_speed * duration + .5 * float(amax) * duration * duration
    accelerated = toward_speed * acceleration_time + .5 * float(amax) * acceleration_time**2
    return accelerated + float(vmax) * (duration - acceleration_time)
