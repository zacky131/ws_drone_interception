"""
src/prediction/camps/predictors.py

Implements candidate target predictors for the CAMPS framework:
1. ConstantVelocityPredictor (predictor_cv)
2. ConstantAccelerationPredictor (predictor_ca)
3. CoordinatedTurnPredictor (predictor_ct)
4. HelicalManeuverPredictor (predictor_helical)
5. NARXResidualPredictor (predictor_narx)
6. OracleTargetPredictor (predictor_oracle)
7. ExactStateCAPredictor (predictor_exact_ca)
"""

from __future__ import annotations
import numpy as np
from typing import Optional, Dict, Any
from src.prediction.camps.protocol import PredictionHorizon

class ConstantVelocityPredictor:
    """Constant velocity target predictor (predictor_cv)."""
    def __init__(self, name: str = "predictor_cv"):
        self.name = name

    def reset(self, seed: int = 42) -> None:
        pass

    def predict(
        self,
        target_estimate: np.ndarray,
        target_history: Optional[np.ndarray] = None,
        horizon_steps: int = 20,
        dt: float = 0.02,
        **kwargs: Any
    ) -> PredictionHorizon:
        pt = target_estimate[0:3]
        vt = target_estimate[3:6]
        
        pos = np.zeros((horizon_steps, 3))
        vel = np.zeros((horizon_steps, 3))
        for k in range(horizon_steps):
            t_pred = (k + 1) * dt
            pos[k] = pt + vt * t_pred
            vel[k] = vt.copy()
            
        return PredictionHorizon(position=pos, velocity=vel, predictor_name=self.name)


class ConstantAccelerationPredictor:
    """Constant acceleration target predictor (predictor_ca)."""
    def __init__(self, name: str = "predictor_ca"):
        self.name = name

    def reset(self, seed: int = 42) -> None:
        pass

    def predict(
        self,
        target_estimate: np.ndarray,
        target_history: Optional[np.ndarray] = None,
        horizon_steps: int = 20,
        dt: float = 0.02,
        **kwargs: Any
    ) -> PredictionHorizon:
        pt = target_estimate[0:3]
        vt = target_estimate[3:6]
        at = target_estimate[6:9] if len(target_estimate) >= 9 else np.zeros(3)
        
        pos = np.zeros((horizon_steps, 3))
        vel = np.zeros((horizon_steps, 3))
        for k in range(horizon_steps):
            t_pred = (k + 1) * dt
            pos[k] = pt + vt * t_pred + 0.5 * at * t_pred**2
            vel[k] = vt + at * t_pred
            
        return PredictionHorizon(position=pos, velocity=vel, predictor_name=self.name)


class CoordinatedTurnPredictor:
    """Coordinated turn target predictor (predictor_ct)."""
    def __init__(self, max_turn_rate: float = 2.0, name: str = "predictor_ct"):
        self.max_turn_rate = max_turn_rate
        self.name = name

    def reset(self, seed: int = 42) -> None:
        pass

    def predict(
        self,
        target_estimate: np.ndarray,
        target_history: Optional[np.ndarray] = None,
        horizon_steps: int = 20,
        dt: float = 0.02,
        **kwargs: Any
    ) -> PredictionHorizon:
        pt = target_estimate[0:3]
        vt = target_estimate[3:6]
        at = target_estimate[6:9] if len(target_estimate) >= 9 else np.zeros(3)
        
        speed_xy = float(np.linalg.norm(vt[0:2]))
        if speed_xy < 0.1:
            # Near zero horizontal speed -> fallback to CA
            ca = ConstantAccelerationPredictor(name=self.name)
            return ca.predict(target_estimate, target_history, horizon_steps, dt)
            
        heading = np.arctan2(vt[1], vt[0])
        # Turn rate from v_x * a_y - v_y * a_x
        raw_omega = (vt[0] * at[1] - vt[1] * at[0]) / (speed_xy**2 + 1e-6)
        omega = float(np.clip(raw_omega, -self.max_turn_rate, self.max_turn_rate))
        
        pos = np.zeros((horizon_steps, 3))
        vel = np.zeros((horizon_steps, 3))
        
        curr_p = pt.copy()
        for k in range(horizon_steps):
            t_step = (k + 1) * dt
            if abs(omega) > 1e-4:
                d_head = omega * t_step
                new_head = heading + d_head
                vx = speed_xy * np.cos(new_head)
                vy = speed_xy * np.sin(new_head)
                # Integrated displacement for circular arc
                px = pt[0] + (speed_xy / omega) * (np.sin(new_head) - np.sin(heading))
                py = pt[1] - (speed_xy / omega) * (np.cos(new_head) - np.cos(heading))
            else:
                vx = vt[0]
                vy = vt[1]
                px = pt[0] + vt[0] * t_step
                py = pt[1] + vt[1] * t_step
                
            # Vertical motion: constant acceleration
            vz = vt[2] + at[2] * t_step
            pz = pt[2] + vt[2] * t_step + 0.5 * at[2] * t_step**2
            
            pos[k] = np.array([px, py, pz])
            vel[k] = np.array([vx, vy, vz])
            
        return PredictionHorizon(position=pos, velocity=vel, predictor_name=self.name, metadata={"omega": omega})


class HelicalManeuverPredictor:
    """Vertical / Helical maneuver target predictor (predictor_helical)."""
    def __init__(self, max_turn_rate: float = 3.0, name: str = "predictor_helical"):
        self.max_turn_rate = max_turn_rate
        self.name = name

    def reset(self, seed: int = 42) -> None:
        pass

    def predict(
        self,
        target_estimate: np.ndarray,
        target_history: Optional[np.ndarray] = None,
        horizon_steps: int = 20,
        dt: float = 0.02,
        **kwargs: Any
    ) -> PredictionHorizon:
        pt = target_estimate[0:3]
        vt = target_estimate[3:6]
        at = target_estimate[6:9] if len(target_estimate) >= 9 else np.zeros(3)
        
        speed_xy = float(np.linalg.norm(vt[0:2]))
        heading = np.arctan2(vt[1], vt[0])
        raw_omega = (vt[0] * at[1] - vt[1] * at[0]) / (speed_xy**2 + 1e-6) if speed_xy >= 0.1 else 0.0
        omega = float(np.clip(raw_omega, -self.max_turn_rate, self.max_turn_rate))
        
        pos = np.zeros((horizon_steps, 3))
        vel = np.zeros((horizon_steps, 3))
        
        for k in range(horizon_steps):
            t_step = (k + 1) * dt
            if abs(omega) > 1e-4 and speed_xy >= 0.1:
                d_head = omega * t_step
                new_head = heading + d_head
                vx = speed_xy * np.cos(new_head)
                vy = speed_xy * np.sin(new_head)
                px = pt[0] + (speed_xy / omega) * (np.sin(new_head) - np.sin(heading))
                py = pt[1] - (speed_xy / omega) * (np.cos(new_head) - np.cos(heading))
            else:
                vx = vt[0] + at[0] * t_step
                vy = vt[1] + at[1] * t_step
                px = pt[0] + vt[0] * t_step + 0.5 * at[0] * t_step**2
                py = pt[1] + vt[1] * t_step + 0.5 * at[1] * t_step**2
                
            # Vertical helical acceleration
            vz = vt[2] + at[2] * t_step
            pz = pt[2] + vt[2] * t_step + 0.5 * at[2] * t_step**2
            pos[k] = np.array([px, py, pz])
            vel[k] = np.array([vx, vy, vz])
            
        return PredictionHorizon(position=pos, velocity=vel, predictor_name=self.name)


class NARXResidualPredictor:
    """NARX residual target predictor (predictor_narx)."""
    def __init__(self, narx_model: Any = None, name: str = "predictor_narx"):
        self.narx_model = narx_model
        self.name = name

    def reset(self, seed: int = 42) -> None:
        if self.narx_model is not None and hasattr(self.narx_model, "reset"):
            self.narx_model.reset(seed)

    def predict(
        self,
        target_estimate: np.ndarray,
        target_history: Optional[np.ndarray] = None,
        horizon_steps: int = 20,
        dt: float = 0.02,
        **kwargs: Any
    ) -> PredictionHorizon:
        # Base CA prediction
        ca = ConstantAccelerationPredictor()
        ca_horiz = ca.predict(target_estimate, target_history, horizon_steps, dt)
        
        pos = ca_horiz.position.copy()
        vel = ca_horiz.velocity.copy()
        
        if self.narx_model is not None and kwargs.get("narx_waypoints") is not None:
            narx_wp = kwargs["narx_waypoints"]
            if len(narx_wp) >= horizon_steps:
                pos = narx_wp[:horizon_steps, 0:3]
                vel = narx_wp[:horizon_steps, 3:6]
                
        return PredictionHorizon(position=pos, velocity=vel, predictor_name=self.name)


class OracleTargetPredictor:
    """Oracle target predictor with perfect future ground truth knowledge."""
    def __init__(self, scenario: Any = None, name: str = "predictor_oracle"):
        self.scenario = scenario
        self.name = name

    def set_scenario(self, scenario: Any) -> None:
        self.scenario = scenario

    def reset(self, seed: int = 42) -> None:
        pass

    def predict(
        self,
        target_estimate: np.ndarray,
        target_history: Optional[np.ndarray] = None,
        horizon_steps: int = 20,
        dt: float = 0.02,
        t_curr: float = 0.0,
        **kwargs: Any
    ) -> PredictionHorizon:
        pos = np.zeros((horizon_steps, 3))
        vel = np.zeros((horizon_steps, 3))
        
        scenario = kwargs.get("scenario", self.scenario)
        if scenario is not None:
            for k in range(horizon_steps):
                t_fut = t_curr + (k + 1) * dt
                p_t, v_t, _ = scenario.get_target_state(t_fut)
                pos[k] = p_t
                vel[k] = v_t
        else:
            # Fallback if scenario missing
            ca = ConstantAccelerationPredictor()
            return ca.predict(target_estimate, target_history, horizon_steps, dt)
            
        return PredictionHorizon(position=pos, velocity=vel, predictor_name=self.name)


class ExactStateCAPredictor:
    """Exact current state CA target predictor (zero measurement/estimation noise)."""
    def __init__(self, name: str = "predictor_exact_ca"):
        self.name = name

    def reset(self, seed: int = 42) -> None:
        pass

    def predict(
        self,
        target_estimate: np.ndarray,
        target_history: Optional[np.ndarray] = None,
        horizon_steps: int = 20,
        dt: float = 0.02,
        t_curr: float = 0.0,
        **kwargs: Any
    ) -> PredictionHorizon:
        scenario = kwargs.get("scenario")
        if scenario is not None:
            pt, vt, at = scenario.get_target_state(t_curr)
            true_state = np.concatenate([pt, vt, at])
            ca = ConstantAccelerationPredictor(name=self.name)
            return ca.predict(true_state, target_history, horizon_steps, dt)
        else:
            ca = ConstantAccelerationPredictor(name=self.name)
            return ca.predict(target_estimate, target_history, horizon_steps, dt)
