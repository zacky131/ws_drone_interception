"""
Abstract base class for pursuer dynamics models.

Every concrete pursuer must implement :meth:`step` and :meth:`reset`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PursuerState:
    """Observable pursuer state vector."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    applied_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def to_array(self) -> np.ndarray:
        """Flatten to a 9-element vector [pos, vel, a_applied]."""
        return np.concatenate([self.position, self.velocity, self.applied_acceleration])

    @staticmethod
    def from_array(x: np.ndarray) -> "PursuerState":
        return PursuerState(
            position=x[0:3].copy(),
            velocity=x[3:6].copy(),
            applied_acceleration=x[6:9].copy(),
        )


class PursuerBase(ABC):
    """Interface for all pursuer dynamics models."""

    @abstractmethod
    def reset(self, position: np.ndarray, velocity: np.ndarray) -> None:
        """Re-initialise the pursuer to the given state."""

    @abstractmethod
    def step(
        self,
        commanded_acceleration: np.ndarray,
        wind_disturbance: np.ndarray,
        dt: float,
    ) -> PursuerState:
        """Advance the pursuer by one timestep and return the new state.

        Parameters
        ----------
        commanded_acceleration : (3,) array
            Virtual control input (commanded acceleration).
        wind_disturbance : (3,) array
            Additive wind disturbance acceleration.
        dt : float
            Integration timestep [s].
        """

    @property
    @abstractmethod
    def state(self) -> PursuerState:
        """Current pursuer state."""

    @property
    def position(self) -> np.ndarray:
        return self.state.position

    @property
    def velocity(self) -> np.ndarray:
        return self.state.velocity
