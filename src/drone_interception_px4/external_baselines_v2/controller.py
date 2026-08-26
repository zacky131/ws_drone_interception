"""Fair common-estimator adapters for GPN and Srivastava-MPC."""

from __future__ import annotations
import math
import os
from pathlib import Path
import time
import numpy as np

from external_baselines.common import CommandSafetyEnvelope, SourceTimeCAInterface
from .gpn import gpn_command
from .srivastava_mpc import SrivastavaMPC

GPN = "GPN"
SRIVASTAVA = "SRIVASTAVA_MPC"
METHODS = (GPN, SRIVASTAVA)


class V2ControllerAdapter:
    def __init__(self, method, config_path, trial_seed):
        if method not in METHODS:
            raise ValueError(method)
        root = Path(os.environ.get("WS_DRONE_INTERCEPTION", Path(__file__).resolve().parents[3]))
        self.method = method
        self.estimator_interface = SourceTimeCAInterface(root / "configs/dapcs_mpc_v1/imm.yaml")
        self.safety = CommandSafetyEnvelope(root / "configs/dapcs_mpc_v1/controller.yaml")
        self.mpc = SrivastavaMPC() if method == SRIVASTAVA else None
        self.interceptor_yaw_enu = 0.0
        self.last_diagnostics = {}
        self.reset(trial_seed, "", "C0")

    @property
    def estimator(self):
        return self.estimator_interface.estimator

    def reset(self, seed=None, trajectory_id="", condition="C0"):
        self.estimator_interface.reset(); self.safety.reset()
        if self.mpc is not None: self.mpc.reset()
        self.last_diagnostics = {}

    def _acceleration_stages(self, raw, dt):
        raw = np.asarray(raw, float)
        acceleration = np.clip(raw, -self.safety.max_acceleration_axis, self.safety.max_acceleration_axis)
        norm = np.linalg.norm(acceleration)
        if norm > self.safety.max_acceleration: acceleration *= self.safety.max_acceleration / norm
        previous = self.safety.previous_command.copy()
        delta = acceleration - previous; rate = np.linalg.norm(delta) > self.safety.max_jerk * dt + 1e-12
        applied, info = self.safety.apply(raw, dt)
        return acceleration, applied, previous, {**info, "external_rate_limited": int(rate)}

    def step(self, interceptor_state_enu, target_measurement_enu, dt_s, sim_time_s):
        start = time.perf_counter_ns()
        interceptor = np.asarray(interceptor_state_enu, float).reshape(9)
        mean, covariance, estimator = self.estimator_interface.process(target_measurement_enu)
        if self.method == GPN:
            raw, method = gpn_command(mean[:3], mean[3:6], interceptor[:3], interceptor[3:6])
            accel_stage, command, previous, limiter = self._acceleration_stages(raw, dt_s)
            method.update({
                "solver_success": True, "solver_status": "closed_form_GPN", "solve_time_s": 0.0,
                "raw_command": raw, "post_acceleration_limit": accel_stage,
                "post_rate_limit": command, "previous_applied_command": previous,
            })
        else:
            assert self.mpc is not None
            follower = np.r_[interceptor[:3], self.interceptor_yaw_enu]
            command, yaw_rate, method = self.mpc.command(
                target_measurement_enu.arrival_timestamp_s, follower, mean
            )
            limiter = {"external_envelope_clipped": 0, "external_rate_limited": 0}
            method.update({"native_velocity_command_enu": command, "native_yaw_rate_radps": yaw_rate})
        forecast = self.estimator_interface.forecast(np.arange(1, 21) * 0.02)
        info = {
            **estimator, **method, **limiter, "target_estimate": mean,
            "current_covariance": covariance, "ca_predicted_position_horizon": forecast[:, :3],
            "ca_predicted_velocity_horizon": forecast[:, 3:6],
            "future_truth_access": 0, "executed_truth_controller_access": 0,
            "commanded_target_reference_access": 0,
        }
        info["controller_total_time_s"] = (time.perf_counter_ns() - start) * 1e-9
        info.setdefault("belief_rollout_time_s", math.nan); info.setdefault("capture_selector_time_s", math.nan)
        self.last_diagnostics = info
        return command, dict(info)

    def get_diagnostics(self):
        return dict(self.last_diagnostics)
