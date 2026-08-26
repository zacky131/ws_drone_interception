"""
src/prediction/camps/reliability.py

Phase 3: Predictor Reliability, Cross-Candidate Disagreement, and Quality Checks.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from src.prediction.camps.protocol import PredictionHorizon

class PredictorReliabilityTracker:
    """Tracks prequential prediction error and reliability for each predictor candidate."""

    def __init__(self, history_len: int = 50, ema_beta: float = 0.9):
        self.history_len = history_len
        self.ema_beta = ema_beta
        self.reset()

    def reset(self, seed: int = 42) -> None:
        self.error_history: Dict[str, List[float]] = {}
        self.ema_error: Dict[str, float] = {}
        self.matured_count: Dict[str, int] = {}
        self.past_predictions: List[Dict[str, Any]] = []

    def update_actual(self, t_curr: float, actual_pos: np.ndarray, actual_vel: np.ndarray, dt: float = 0.02) -> None:
        """Evaluate matured prediction steps against ground truth / updated estimate."""
        matured = []
        for record in self.past_predictions:
            p_time = record["t_issue"]
            horiz_step = int(round((t_curr - p_time) / dt))
            if 1 <= horiz_step <= record["horizon"].position.shape[0]:
                name = record["predictor_name"]
                pred_p = record["horizon"].position[horiz_step - 1]
                err = float(np.linalg.norm(actual_pos - pred_p))
                
                if name not in self.error_history:
                    self.error_history[name] = []
                    self.ema_error[name] = err
                    self.matured_count[name] = 0

                self.error_history[name].append(err)
                if len(self.error_history[name]) > self.history_len:
                    self.error_history[name].pop(0)

                self.ema_error[name] = self.ema_beta * self.ema_error[name] + (1 - self.ema_beta) * err
                self.matured_count[name] += 1
            elif horiz_step > record["horizon"].position.shape[0]:
                matured.append(record)

        # Clean up old records
        for m in matured:
            if m in self.past_predictions:
                self.past_predictions.remove(m)

    def log_issued_predictions(self, t_curr: float, horizons: Dict[str, PredictionHorizon]) -> None:
        for name, horiz in horizons.items():
            self.past_predictions.append({
                "t_issue": t_curr,
                "predictor_name": name,
                "horizon": horiz
            })

    def get_reliability_features(self, name: str, horizon: PredictionHorizon, dt: float = 0.02) -> Dict[str, float]:
        errs = self.error_history.get(name, [])
        ema = self.ema_error.get(name, 0.0)
        cnt = self.matured_count.get(name, 0)

        mean_err = float(np.mean(errs)) if errs else 0.0
        p90_err = float(np.percentile(errs, 90)) if errs else 0.0
        trend = float(errs[-1] - errs[0]) if len(errs) >= 5 else 0.0

        # Physical smoothness features of predicted horizon
        pos = horizon.position
        vel = horizon.velocity
        N = pos.shape[0]
        
        # Calculate acceleration and jerk along the predicted horizon
        acc_pred = np.diff(vel, axis=0) / dt if N > 1 else np.zeros((1, 3))
        jerk_pred = np.diff(acc_pred, axis=0) / dt if N > 2 else np.zeros((1, 3))

        peak_acc = float(np.max(np.linalg.norm(acc_pred, axis=1))) if len(acc_pred) > 0 else 0.0
        peak_jerk = float(np.max(np.linalg.norm(jerk_pred, axis=1))) if len(jerk_pred) > 0 else 0.0

        # Consistency error: integral of ||v - dp/dt||
        dp = np.diff(pos, axis=0) / dt if N > 1 else np.zeros((1, 3))
        vel_mid = 0.5 * (vel[:-1] + vel[1:]) if N > 1 else vel
        pos_vel_consistency = float(np.mean(np.linalg.norm(dp - vel_mid, axis=1))) if N > 1 else 0.0

        return {
            "prequential_error_ema": ema,
            "recent_position_error_mean": mean_err,
            "recent_position_error_p90": p90_err,
            "error_trend": trend,
            "matured_validation_count": float(cnt),
            "position_velocity_consistency_error": pos_vel_consistency,
            "predicted_acceleration_peak": peak_acc,
            "predicted_jerk_peak": peak_jerk,
        }


def compute_cross_candidate_disagreement(horizons: Dict[str, PredictionHorizon]) -> Dict[str, float]:
    """Compute cross-candidate disagreement metrics."""
    names = list(horizons.keys())
    if len(names) < 2:
        return {
            "mean_pairwise_position_disagreement": 0.0,
            "max_pairwise_position_disagreement": 0.0,
            "terminal_position_disagreement": 0.0,
            "terminal_velocity_disagreement": 0.0,
            "narx_minus_ca_disagreement": 0.0
        }

    pairwise_pos_diffs = []
    terminal_pos_diffs = []
    terminal_vel_diffs = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            h1 = horizons[names[i]]
            h2 = horizons[names[j]]
            
            p1, p2 = h1.position, h2.position
            v1, v2 = h1.velocity, h2.velocity
            
            diff_p = np.linalg.norm(p1 - p2, axis=1)
            pairwise_pos_diffs.append(np.mean(diff_p))
            
            t_diff_p = float(np.linalg.norm(p1[-1] - p2[-1]))
            t_diff_v = float(np.linalg.norm(v1[-1] - v2[-1]))
            terminal_pos_diffs.append(t_diff_p)
            terminal_vel_diffs.append(t_diff_v)

    narx_ca_dis = 0.0
    if "predictor_narx" in horizons and "predictor_ca" in horizons:
        p_n = horizons["predictor_narx"].position
        p_c = horizons["predictor_ca"].position
        narx_ca_dis = float(np.mean(np.linalg.norm(p_n - p_c, axis=1)))

    return {
        "mean_pairwise_position_disagreement": float(np.mean(pairwise_pos_diffs)),
        "max_pairwise_position_disagreement": float(np.max(pairwise_pos_diffs)),
        "terminal_position_disagreement": float(np.mean(terminal_pos_diffs)),
        "terminal_velocity_disagreement": float(np.mean(terminal_vel_diffs)),
        "narx_minus_ca_disagreement": narx_ca_dis,
    }


def perform_quality_checks(
    horizon: PredictionHorizon,
    max_speed: float = 30.0,
    max_acc: float = 40.0,
    max_jerk: float = 100.0,
    dt: float = 0.02
) -> Tuple[bool, List[str]]:
    """Quality check: reject or down-weight non-physical predicted horizons."""
    rejections = []
    pos = horizon.position
    vel = horizon.velocity
    N = pos.shape[0]

    speeds = np.linalg.norm(vel, axis=1)
    if np.any(speeds > max_speed):
        rejections.append(f"speed_exceeded_{np.max(speeds):.1f}")

    if N > 1:
        accs = np.linalg.norm(np.diff(vel, axis=0) / dt, axis=1)
        if np.any(accs > max_acc):
            rejections.append(f"acc_exceeded_{np.max(accs):.1f}")

    if N > 2:
        accs = np.diff(vel, axis=0) / dt
        jerks = np.linalg.norm(np.diff(accs, axis=0) / dt, axis=1)
        if np.any(jerks > max_jerk):
            rejections.append(f"jerk_exceeded_{np.max(jerks):.1f}")

    is_valid = len(rejections) == 0
    return is_valid, rejections
