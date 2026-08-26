"""
Wind disturbance model.

Supports a constant (steady) component and an optional low-frequency sinusoidal
gust.  Both can be toggled independently for ablation studies.

    d_wind(t) = w_steady + w_gust(t)
    w_gust(t) = A · [sin(2π f t),  sin(2π f t + φ_y),  sin(2π f t + φ_z)]

Phase offsets φ_y, φ_z are fixed to avoid purely correlated axes.
"""

from __future__ import annotations

import numpy as np

from src.utils.config_schema import WindConfig


class WindModel:
    """Configurable wind disturbance (steady + gust)."""

    def __init__(self, config: WindConfig) -> None:
        self._cfg = config
        self._steady = np.asarray(config.steady_wind, dtype=float) if config.enabled else np.zeros(3)
        self._gust_amp = config.gust_amplitude if config.gust_enabled and config.enabled else 0.0
        self._gust_freq = config.gust_frequency
        # Fixed phase offsets for y and z gust components
        self._phase_y = np.pi / 3.0
        self._phase_z = 2.0 * np.pi / 3.0

    def get_wind(self, t: float) -> np.ndarray:
        """Return the wind disturbance vector at time *t* [m/s].

        This is an *additive velocity disturbance* applied directly to the
        pursuer translational dynamics (equivalent to a force disturbance
        d_wind / dt in acceleration form when divided by dt, but typically
        kept in acceleration space as d_wind · g for unit consistency).
        """
        w = self._steady.copy()
        if self._gust_amp > 0.0:
            omega = 2.0 * np.pi * self._gust_freq
            w[0] += self._gust_amp * np.sin(omega * t)
            w[1] += self._gust_amp * np.sin(omega * t + self._phase_y)
            w[2] += self._gust_amp * np.sin(omega * t + self._phase_z)
        return w

    def reset(self) -> None:
        """No internal state to reset; included for interface consistency."""
