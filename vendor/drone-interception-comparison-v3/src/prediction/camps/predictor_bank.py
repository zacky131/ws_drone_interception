"""
src/prediction/camps/predictor_bank.py

Phase 2 & Phase 5: Candidate Predictor Bank orchestrator.
Manages candidate predictors (CV, CA, CT, Helical, NARX) and integrates
with reliability tracker, capturability proxy, and selector.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, Optional
from src.prediction.camps.protocol import PredictionHorizon
from src.prediction.camps.predictors import (
    ConstantVelocityPredictor,
    ConstantAccelerationPredictor,
    CoordinatedTurnPredictor,
    HelicalManeuverPredictor,
    NARXResidualPredictor,
)
from src.prediction.camps.reliability import PredictorReliabilityTracker
from src.prediction.camps.capturability import KinematicCapturabilityProxy
from src.prediction.camps.selector import CAMPSRuleSelector, CAMPSLearnedSelector, CAMPSFusionSelector

class CAMPSPredictorBank:
    """Manages bank of target predictors and returns selected forecast for MPC."""

    def __init__(self, selector_type: str = "camps_rule"):
        self.selector_type = selector_type
        
        self.predictors = {
            "predictor_cv": ConstantVelocityPredictor(),
            "predictor_ca": ConstantAccelerationPredictor(),
            "predictor_ct": CoordinatedTurnPredictor(),
            "predictor_helical": HelicalManeuverPredictor(),
            "predictor_narx": NARXResidualPredictor(),
        }

        self.reliability_tracker = PredictorReliabilityTracker()
        self.capturability_proxy = KinematicCapturabilityProxy()

        if selector_type == "camps_learned":
            self.selector = CAMPSLearnedSelector()
        elif selector_type == "camps_fusion":
            self.selector = CAMPSFusionSelector()
        else:
            self.selector = CAMPSRuleSelector()

    def reset(self, seed: int = 42) -> None:
        for p in self.predictors.values():
            p.reset(seed)
        self.reliability_tracker.reset(seed)
        self.selector.reset(seed)

    def predict_selected(
        self,
        pursuer_state: np.ndarray,
        target_estimate: np.ndarray,
        target_history: Optional[np.ndarray] = None,
        horizon_steps: int = 20,
        dt: float = 0.02,
        t_curr: float = 0.0,
        **kwargs: Any
    ) -> Tuple[PredictionHorizon, str, Dict[str, Any]]:
        """Run candidate predictors, update reliability, evaluate capturability, select horizon."""
        
        # 1. Update reliability tracker with actual ground truth / updated estimate
        pt_curr = target_estimate[0:3]
        vt_curr = target_estimate[3:6]
        self.reliability_tracker.update_actual(t_curr, pt_curr, vt_curr, dt=dt)

        # 2. Query all candidate predictors
        candidate_horizons: Dict[str, PredictionHorizon] = {}
        for name, pred in self.predictors.items():
            horiz = pred.predict(
                target_estimate=target_estimate,
                target_history=target_history,
                horizon_steps=horizon_steps,
                dt=dt,
                **kwargs
            )
            candidate_horizons[name] = horiz

        # 3. Log issued predictions for future prequential evaluation
        self.reliability_tracker.log_issued_predictions(t_curr, candidate_horizons)

        # 4. Select optimal forecast
        selected_horiz, selected_name, info = self.selector.select(
            horizons=candidate_horizons,
            reliability_tracker=self.reliability_tracker,
            capturability_proxy=self.capturability_proxy,
            pursuer_state=pursuer_state,
            dt=dt
        )

        return selected_horiz, selected_name, info
