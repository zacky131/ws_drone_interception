"""
Abstract base class for target-state estimators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EstimatorBase(ABC):
    """Interface for all target-state estimators."""

    @abstractmethod
    def initialize(self, measurement: np.ndarray) -> None:
        """Set the initial state from the first measurement.

        Parameters
        ----------
        measurement : (6,) array
            [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
        """

    @abstractmethod
    def predict(self, dt: float) -> None:
        """Time-update (propagation) step."""

    @abstractmethod
    def update(self, measurement: np.ndarray) -> None:
        """Measurement-update step.

        Parameters
        ----------
        measurement : (6,) array
            Noisy [position, velocity].
        """

    @abstractmethod
    def get_estimate(self) -> np.ndarray:
        """Return the current state estimate.

        Returns
        -------
        x_hat : (12,) array
            [pos(3), vel(3), acc(3), jerk(3)]
        """

    @abstractmethod
    def get_covariance(self) -> np.ndarray:
        """Return the current estimation-error covariance P (12×12)."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state to uninitialised."""
