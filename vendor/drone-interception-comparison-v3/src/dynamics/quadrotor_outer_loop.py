"""
Quadrotor outer-loop pursuer model with first-order actuator lag.

This model approximates the cascaded inner-loop attitude controller as a
first-order transfer function on each acceleration axis:

    τ · ȧ_applied + a_applied = a_commanded

where τ is the actuator time constant.  The outer-loop guidance perceives this
as a lag between the commanded and realised acceleration, which is a standard
engineering approximation for outer-loop guidance studies (cf. Zarchan 2012).

State vector  x = [p_x, p_y, p_z, v_x, v_y, v_z, a_x, a_y, a_z]  (9D)

Constraints enforced at each step:
    - Per-axis acceleration bound
    - Acceleration norm bound
    - Acceleration rate (jerk) bound
    - Velocity norm bound
    - Ground constraint  z ≥ 0
"""

from __future__ import annotations

import numpy as np

from src.utils.config_schema import PursuerConfig
from src.utils.math_helpers import clip_norm, clip_components
from .pursuer_base import PursuerBase, PursuerState


class QuadrotorOuterLoopPursuer(PursuerBase):
    """Outer-loop pursuer model with actuator lag and realistic constraints."""

    def __init__(self, config: PursuerConfig) -> None:
        self._cfg = config
        self._tau = config.actuator_time_constant
        self._state = PursuerState()
        self._prev_applied = np.zeros(3)  # for jerk limiting
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
        self._prev_applied = np.zeros(3)

    def step(
        self,
        commanded_acceleration: np.ndarray,
        wind_disturbance: np.ndarray,
        dt: float,
    ) -> PursuerState:
        # ── 1. Saturate raw command ───────────────────────────────────────
        a_cmd = clip_components(commanded_acceleration, self._cfg.max_acceleration_per_axis)
        a_cmd = clip_norm(a_cmd, self._cfg.max_acceleration)

        # ── 2. First-order actuator dynamics ──────────────────────────────
        #  a_applied(k+1) = a_applied(k) + α · (a_cmd - a_applied(k))
        #  where α = dt / τ  (clamped to [0, 1] for stability)
        alpha = min(dt / self._tau, 1.0) if self._tau > 1e-9 else 1.0
        a_new = self._state.applied_acceleration + alpha * (a_cmd - self._state.applied_acceleration)

        # ── 3. Jerk (rate) limiting ───────────────────────────────────────
        jerk = (a_new - self._state.applied_acceleration) / dt if dt > 1e-12 else np.zeros(3)
        jerk_norm = np.linalg.norm(jerk)
        if jerk_norm > self._cfg.max_jerk:
            a_new = self._state.applied_acceleration + clip_norm(jerk, self._cfg.max_jerk) * dt

        # ── 4. Re-saturate after rate limiting ────────────────────────────
        a_new = clip_components(a_new, self._cfg.max_acceleration_per_axis)
        a_new = clip_norm(a_new, self._cfg.max_acceleration)

        # ── 5. Translational dynamics ─────────────────────────────────────
        total_acc = a_new + wind_disturbance
        new_vel = self._state.velocity + total_acc * dt
        new_vel = clip_norm(new_vel, self._cfg.max_velocity)

        new_pos = (
            self._state.position
            + self._state.velocity * dt
            + 0.5 * total_acc * dt ** 2
        )

        # Ground constraint
        new_pos[2] = max(0.0, new_pos[2])

        # ── 6. Store ─────────────────────────────────────────────────────
        self._prev_applied = self._state.applied_acceleration.copy()
        self._state = PursuerState(
            position=new_pos,
            velocity=new_vel,
            applied_acceleration=a_new,
        )
        return self._state

    @property
    def state(self) -> PursuerState:
        return self._state
