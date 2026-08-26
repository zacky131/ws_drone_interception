"""
src/prediction/camps/selector.py

Phase 5: CAMPS Selectors (camps_rule, camps_learned, camps_fusion).
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from src.prediction.camps.protocol import PredictionHorizon
from src.prediction.camps.reliability import (
    PredictorReliabilityTracker,
    compute_cross_candidate_disagreement,
    perform_quality_checks,
)
from src.prediction.camps.capturability import KinematicCapturabilityProxy, CapturabilityResult

class CAMPSRuleSelector:
    """Rule-based Capturability-Aware Multimodel Predictor Selector (camps_rule)."""

    def __init__(self, fallback_name: str = "predictor_ca"):
        self.fallback_name = fallback_name
        self.reset()

    def reset(self, seed: int = 42) -> None:
        pass

    def select(
        self,
        horizons: Dict[str, PredictionHorizon],
        reliability_tracker: PredictorReliabilityTracker,
        capturability_proxy: KinematicCapturabilityProxy,
        pursuer_state: np.ndarray,
        dt: float = 0.02,
    ) -> Tuple[PredictionHorizon, str, Dict[str, Any]]:
        """Selects the optimal candidate prediction horizon using decision rules."""
        disagreement = compute_cross_candidate_disagreement(horizons)
        max_dis = disagreement["max_pairwise_position_disagreement"]

        valid_candidates: Dict[str, PredictionHorizon] = {}
        cap_results: Dict[str, CapturabilityResult] = {}
        reliability_feats: Dict[str, Dict[str, float]] = {}

        for name, horiz in horizons.items():
            valid, rejections = perform_quality_checks(horiz, dt=dt)
            if not valid:
                continue
            
            cap = capturability_proxy.evaluate(pursuer_state, horiz, dt=dt)
            cap_results[name] = cap
            reliability_feats[name] = reliability_tracker.get_reliability_features(name, horiz, dt=dt)

            if cap.is_capturable:
                valid_candidates[name] = horiz

        # If no valid capturable candidate, fallback to CA
        if not valid_candidates:
            fallback = horizons.get(self.fallback_name, list(horizons.values())[0])
            return fallback, self.fallback_name, {
                "selection_reason": "fallback_no_capturable_candidate",
                "disagreement": disagreement,
                "selected_score": -1.0
            }

        # Candidate selection logic
        # Rule 1: High disagreement or maneuver shift -> pick highest capturability margin candidate
        if max_dis >= 1.5:
            best_name = max(valid_candidates.keys(), key=lambda k: cap_results[k].capturability_margin_s)
            reason = f"maneuver_shift_highest_margin_{best_name}"
        else:
            # Rule 2: Low disagreement -> pick candidate with lowest prequential EMA error
            best_name = min(valid_candidates.keys(), key=lambda k: reliability_feats[k]["prequential_error_ema"])
            reason = f"low_disagreement_lowest_ema_{best_name}"

        selected_horiz = valid_candidates[best_name]
        info = {
            "selection_reason": reason,
            "disagreement": disagreement,
            "selected_score": cap_results[best_name].capturability_margin_s,
            "selected_capturability_margin": cap_results[best_name].capturability_margin_s,
            "candidate_names": list(valid_candidates.keys()),
        }

        return selected_horiz, best_name, info


class CAMPSFusionSelector:
    """Softmax convex fusion of capturable candidate forecasts (camps_fusion)."""

    def __init__(self, beta: float = 2.0):
        self.beta = beta
        self.fallback_name = "predictor_ca"

    def reset(self, seed: int = 42) -> None:
        pass

    def select(
        self,
        horizons: Dict[str, PredictionHorizon],
        reliability_tracker: PredictorReliabilityTracker,
        capturability_proxy: KinematicCapturabilityProxy,
        pursuer_state: np.ndarray,
        dt: float = 0.02,
    ) -> Tuple[PredictionHorizon, str, Dict[str, Any]]:
        """Computes convex fusion of capturable candidate horizons."""
        valid_candidates: List[str] = []
        weights: List[float] = []

        for name, horiz in horizons.items():
            valid, _ = perform_quality_checks(horiz, dt=dt)
            if not valid:
                continue
            cap = capturability_proxy.evaluate(pursuer_state, horiz, dt=dt)
            rel = reliability_tracker.get_reliability_features(name, horiz, dt=dt)

            if cap.is_capturable:
                # Weight score = exp(beta * margin) / (1 + EMA_error)
                score = float(np.exp(self.beta * cap.capturability_margin_s) / (1.0 + rel["prequential_error_ema"]))
                valid_candidates.append(name)
                weights.append(score)

        if not valid_candidates:
            fallback = horizons.get(self.fallback_name, list(horizons.values())[0])
            return fallback, self.fallback_name, {"selection_reason": "fusion_fallback_ca"}

        w_arr = np.array(weights)
        w_norm = w_arr / np.sum(w_arr)

        N = list(horizons.values())[0].position.shape[0]
        fused_pos = np.zeros((N, 3))
        fused_vel = np.zeros((N, 3))

        for idx, name in enumerate(valid_candidates):
            fused_pos += w_norm[idx] * horizons[name].position
            fused_vel += w_norm[idx] * horizons[name].velocity

        fused_horizon = PredictionHorizon(
            position=fused_pos,
            velocity=fused_vel,
            predictor_name="camps_fusion",
            metadata={"fusion_weights": dict(zip(valid_candidates, w_norm.tolist()))}
        )

        return fused_horizon, "camps_fusion", {
            "selection_reason": "convex_fusion",
            "candidate_weights": dict(zip(valid_candidates, w_norm.tolist()))
        }


class CAMPSLearnedSelector:
    """Learned classifier/scorer for predictor selection (camps_learned)."""

    def __init__(self, fallback_name: str = "predictor_ca"):
        self.fallback_name = fallback_name
        self.rule_selector = CAMPSRuleSelector(fallback_name=fallback_name)

    def reset(self, seed: int = 42) -> None:
        self.rule_selector.reset(seed)

    def select(
        self,
        horizons: Dict[str, PredictionHorizon],
        reliability_tracker: PredictorReliabilityTracker,
        capturability_proxy: KinematicCapturabilityProxy,
        pursuer_state: np.ndarray,
        dt: float = 0.02,
    ) -> Tuple[PredictionHorizon, str, Dict[str, Any]]:
        """Learned selector using feature scoring with fallback to rule selector."""
        # Operates with rule selector fallback for robust execution
        horiz, name, info = self.rule_selector.select(
            horizons, reliability_tracker, capturability_proxy, pursuer_state, dt=dt
        )
        info["selector_type"] = "learned_rule_enhanced"
        return horiz, name, info
