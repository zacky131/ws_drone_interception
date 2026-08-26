"""
Abstract base class for guidance controllers.

All guidance laws – MPC variants, PN, SMC – implement this interface so
the simulation engine can treat them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np


class ControllerBase(ABC):
    """Unified interface for guidance controllers."""

    @abstractmethod
    def compute_control(
        self,
        pursuer_state: np.ndarray,
        target_measurement: Optional[np.ndarray],
        target_estimate: np.ndarray,
        wind_estimate: np.ndarray,
        t: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute the commanded acceleration.

        Parameters
        ----------
        pursuer_state : (9,) array
            [position(3), velocity(3), applied_acceleration(3)].
        target_measurement : (6,) array or None
            Raw (noisy, delayed) sensor measurement [pos(3), vel(3)].
            May be ``None`` if the sensor dropped the packet.
        target_estimate : (12,) array
            Filtered estimate [pos(3), vel(3), acc(3), jerk(3)].
        wind_estimate : (3,) array
            Current wind disturbance estimate (may be zero if unknown).
        t : float
            Current simulation time [s].

        Returns
        -------
        commanded_acc : (3,) array
            Commanded acceleration vector.
        info : dict
            Auxiliary information (solver status, solve time, etc.).
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset controller internal state (warm-start buffers, etc.)."""
