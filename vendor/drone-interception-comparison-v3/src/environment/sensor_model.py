"""
Sensor model with additive Gaussian noise, latency buffer, and packet dropout.

Provides separate processing for target and pursuer measurements.

Delay model:
    Measurements are buffered in a FIFO queue of length ``delay_steps``.
    The sensor returns the oldest buffered measurement, simulating a
    fixed-latency transport delay.

Dropout model:
    With probability ``dropout_probability`` the sensor returns ``None``,
    indicating that no measurement is available at this timestep.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from src.utils.config_schema import SensorConfig


class SensorModel:
    """Noisy sensor with latency and dropout."""

    def __init__(self, config: SensorConfig, rng: np.random.Generator | None = None) -> None:
        self._cfg = config
        self._rng = rng if rng is not None else np.random.default_rng()

        # Delay buffers (one per sensor channel)
        delay = max(0, config.delay_steps)
        self._target_buffer: deque[np.ndarray] = deque(maxlen=delay + 1)
        self._pursuer_buffer: deque[np.ndarray] = deque(maxlen=delay + 1)
        self._delay = delay

    # ── public ────────────────────────────────────────────────────────────

    def process_target(
        self, true_state: np.ndarray, t: float
    ) -> Optional[np.ndarray]:
        """Return a noisy, delayed measurement of the target, or ``None`` if dropped.

        Parameters
        ----------
        true_state : (6,) array
            True [position(3), velocity(3)].
        t : float
            Current simulation time (unused but available for time-varying noise).

        Returns
        -------
        measurement : (6,) ndarray or None
        """
        noisy = self._add_noise_target(true_state)
        self._target_buffer.append(noisy)

        # Dropout check
        if self._rng.random() < self._cfg.dropout_probability:
            return None

        # Delay: return oldest if buffer is full
        if len(self._target_buffer) > self._delay:
            return self._target_buffer[0]
        return self._target_buffer[0]  # not enough history yet; return what we have

    def process_pursuer(
        self, true_state_9d: np.ndarray, t: float
    ) -> Optional[np.ndarray]:
        """Return a noisy pursuer position + velocity measurement.

        Parameters
        ----------
        true_state_9d : (9,) array
            [position(3), velocity(3), applied_acc(3)].

        Returns
        -------
        measurement : (6,) ndarray or None
            Noisy [position(3), velocity(3)].
        """
        pos = true_state_9d[0:3]
        vel = true_state_9d[3:6]
        noisy = np.concatenate([
            pos + self._rng.normal(0, self._cfg.pursuer_position_noise_std, size=3),
            vel + self._rng.normal(0, self._cfg.pursuer_velocity_noise_std, size=3),
        ])
        self._pursuer_buffer.append(noisy)

        if len(self._pursuer_buffer) > self._delay:
            return self._pursuer_buffer[0]
        return self._pursuer_buffer[0]

    def reset(self) -> None:
        self._target_buffer.clear()
        self._pursuer_buffer.clear()

    # ── internal ──────────────────────────────────────────────────────────

    def _add_noise_target(self, true_state: np.ndarray) -> np.ndarray:
        pos = true_state[0:3] + self._rng.normal(0, self._cfg.position_noise_std, size=3)
        vel = true_state[3:6] + self._rng.normal(0, self._cfg.velocity_noise_std, size=3)
        return np.concatenate([pos, vel])
