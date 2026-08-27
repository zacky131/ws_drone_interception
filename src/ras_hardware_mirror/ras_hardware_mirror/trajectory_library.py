"""Deterministic, analytic hardware-mirror target trajectories."""

from __future__ import annotations

import math
from typing import Any
import numpy as np

from .state_types import TargetKinematics


TRAJECTORIES = ("STATIC", "HT1", "HT2")


def evaluate_trajectory(name: str, t_s: float, config: dict[str, Any]) -> TargetKinematics:
    if name not in TRAJECTORIES:
        raise ValueError(f"trajectory must be one of {TRAJECTORIES}")
    t = max(0.0, float(t_s))
    target = config["virtual_target"]
    altitude = float(target["nominal_altitude_m"])
    origin = np.asarray(target.get("origin_enu_m", [0.0, 0.0, altitude]), dtype=float)
    origin[2] = altitude
    theta_deg = float(target.get("orientation_deg", target.get("heading_deg", 0.0)))
    if abs(theta_deg) > 1e-4:
        theta = math.radians(theta_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
    else:
        cos_t, sin_t = 1.0, 0.0

    if name == "STATIC":
        position = np.asarray(target["STATIC"]["position_enu_m"], dtype=float)
        return TargetKinematics(position, np.zeros(3), np.zeros(3))
    if name == "HT1":
        p = target["HT1"]
        speed = float(target["nominal_speed_mps"])
        amplitude = float(p["lateral_amplitude_m"])
        wave = float(p["wavelength_m"])
        duration = float(p["duration_s"])
        # Quintic progress gives zero velocity and acceleration at both ends.
        q = min(t / duration, 1.0)
        progress = 10.0 * q**3 - 15.0 * q**4 + 6.0 * q**5
        progress_rate = (30.0 * q**2 - 60.0 * q**3 + 30.0 * q**4) / duration if t < duration else 0.0
        progress_acceleration = (60.0 * q - 180.0 * q**2 + 120.0 * q**3) / duration**2 if t < duration else 0.0
        track_length = speed * duration / 1.875
        phase_scale = 2.0 * math.pi * track_length / wave
        phase = phase_scale * progress
        phase_rate = phase_scale * progress_rate
        phase_acceleration = phase_scale * progress_acceleration

        x_local = -0.5 * track_length + track_length * progress
        y_local = amplitude * math.sin(phase)
        vx_local = track_length * progress_rate
        vy_local = amplitude * math.cos(phase) * phase_rate
        ax_local = track_length * progress_acceleration
        ay_local = amplitude * (-math.sin(phase) * phase_rate**2 + math.cos(phase) * phase_acceleration)

        east = origin[0] + x_local * cos_t - y_local * sin_t
        north = origin[1] + x_local * sin_t + y_local * cos_t
        vx = vx_local * cos_t - vy_local * sin_t
        vy = vx_local * sin_t + vy_local * cos_t
        ax = ax_local * cos_t - ay_local * sin_t
        ay = ax_local * sin_t + ay_local * cos_t

        position = np.array([east, north, altitude])
        velocity = np.array([vx, vy, 0.0])
        acceleration = np.array([ax, ay, 0.0])
        return TargetKinematics(position, velocity, acceleration)
    p = target["HT2"]
    major, minor = float(p["major_radius_m"]), float(p["minor_radius_m"])
    omega = float(p["angular_rate_rad_s"])
    phase = omega * t

    x_local = major * math.cos(phase)
    y_local = minor * math.sin(phase)
    vx_local = -major * omega * math.sin(phase)
    vy_local = minor * omega * math.cos(phase)
    ax_local = -major * omega**2 * math.cos(phase)
    ay_local = -minor * omega**2 * math.sin(phase)

    east = origin[0] + x_local * cos_t - y_local * sin_t
    north = origin[1] + x_local * sin_t + y_local * cos_t
    vx = vx_local * cos_t - vy_local * sin_t
    vy = vx_local * sin_t + vy_local * cos_t
    ax = ax_local * cos_t - ay_local * sin_t
    ay = ax_local * sin_t + ay_local * cos_t

    position = np.array([east, north, altitude])
    velocity = np.array([vx, vy, 0.0])
    acceleration = np.array([ax, ay, 0.0])
    return TargetKinematics(position, velocity, acceleration)


def sample_trajectory(name: str, duration_s: float, dt_s: float, config: dict[str, Any]) -> list[TargetKinematics]:
    return [evaluate_trajectory(name, t, config) for t in np.arange(0.0, duration_s + 0.5 * dt_s, dt_s)]
