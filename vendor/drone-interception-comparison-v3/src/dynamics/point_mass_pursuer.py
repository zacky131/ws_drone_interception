"""
Simple point-mass pursuer model.

Direct acceleration control without actuator dynamics – retained for backward
compatibility and ablation only.  Applied acceleration equals the commanded
acceleration (after saturation).
"""

from __future__ import annotations

import numpy as np

from src.utils.config_schema import PursuerConfig
from src.utils.math_helpers import clip_norm, clip_components
from .pursuer_base import PursuerBase, PursuerState


class PointMassPursuer(PursuerBase):
    """Point-mass pursuer with instantaneous acceleration application."""

    def __init__(self, config: PursuerConfig) -> None:
        self._cfg = config
        self._state = PursuerState()
        self.reset(
            np.asarray(config.initial_position, dtype=float),
            np.asarray(config.initial_velocity, dtype=float),
        )

    # ── interface ──────────────────────────────────────────────────────────

    def reset(self, position: np.ndarray, velocity: np.ndarray) -> None:
        self._state = PursuerState(
            position=position.copy(),
            velocity=velocity.copy(),
            applied_acceleration=np.zeros(3),
        )

    def step(
        self,
        commanded_acceleration: np.ndarray,
        wind_disturbance: np.ndarray,
        dt: float,
    ) -> PursuerState:
        # Saturate command
        a_cmd = clip_components(commanded_acceleration, self._cfg.max_acceleration_per_axis)
        a_cmd = clip_norm(a_cmd, self._cfg.max_acceleration)

        # For point mass, applied = commanded (no actuator dynamics)
        a_applied = a_cmd

        # Translational dynamics with wind
        new_vel = self._state.velocity + (a_applied + wind_disturbance) * dt

        # Velocity saturation
        new_vel = clip_norm(new_vel, self._cfg.max_velocity)

        new_pos = self._state.position + self._state.velocity * dt + 0.5 * (a_applied + wind_disturbance) * dt ** 2

        # Ground constraint
        new_pos[2] = max(0.0, new_pos[2])

        self._state = PursuerState(
            position=new_pos,
            velocity=new_vel,
            applied_acceleration=a_applied,
        )
        return self._state

    @property
    def state(self) -> PursuerState:
        return self._state
