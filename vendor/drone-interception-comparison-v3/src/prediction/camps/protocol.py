"""
src/prediction/camps/protocol.py

Defines common data classes and protocols for CAMPS target predictors.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Protocol, runtime_checkable
import numpy as np

@dataclass
class PredictionHorizon:
    """Holds predicted target trajectory over horizon N."""
    position: np.ndarray        # (N, 3)
    velocity: np.ndarray        # (N, 3)
    covariance: Optional[np.ndarray] = None # (N, 3, 3) optional
    predictor_name: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def waypoints(self) -> np.ndarray:
        """Returns concatenated (N, 6) position and velocity waypoints."""
        return np.hstack([self.position, self.velocity])

@runtime_checkable
class TargetHorizonPredictor(Protocol):
    """Protocol interface for CAMPS candidate target predictors."""
    def reset(self, seed: int = 42) -> None:
        ...

    def predict(
        self,
        target_estimate: np.ndarray, # (12,) or (6,) state
        target_history: Optional[np.ndarray], # recent positions/velocities
        horizon_steps: int,
        dt: float,
        **kwargs: Any
    ) -> PredictionHorizon:
        ...
