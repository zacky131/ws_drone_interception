from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np

from src.baselines.standard_mpc import StandardMPC

class ConstantVelocityMPC(StandardMPC):
    """ConstantVelocityMPC always uses the EKF target estimate (pos and vel), ignoring raw measurement."""

    def compute_control(
        self,
        pursuer_state: np.ndarray,
        target_measurement: Optional[np.ndarray],
        target_estimate: np.ndarray,
        wind_estimate: np.ndarray,
        t: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Force passing target_measurement=None so StandardMPC uses target_estimate
        return super().compute_control(
            pursuer_state,
            None,
            target_estimate,
            wind_estimate,
            t,
        )