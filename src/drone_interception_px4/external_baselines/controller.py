"""Controller adapter for isolated FRPN and VT-MPC Rev6 baselines."""

from __future__ import annotations

import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from drone_interception_px4.telemetry import TelemetryEvent

from .common import CommandSafetyEnvelope, SourceTimeCAInterface
from .frpn import FRPNGuidance, PUBLISHED_G, PUBLISHED_W
from .vtmpc import VariableTimeStepMPC


FRPN = "FRPN"
VTMPC = "VTMPC"
METHODS = (FRPN, VTMPC)
INTERNAL_NAMES = {FRPN: "FRPN", VTMPC: "VTMPC_translational_port"}
MANUSCRIPT_NAMES = {FRPN: "FRPN", VTMPC: "VTMPC"}


class ExternalBaselineControllerAdapter:
    """Vary only guidance/planning behind a shared corrected CA posterior."""

    def __init__(self, method: str, config_path: str | Path, trial_seed: int) -> None:
        if method not in METHODS:
            raise ValueError(f"unsupported external baseline: {method}")
        self.method = method
        self.seed = int(trial_seed)
        root = Path(os.environ.get("WS_DRONE_INTERCEPTION", Path(__file__).resolve().parents[3]))
        self.estimator_config_path = root / "configs/dapcs_mpc_v1/imm.yaml"
        self.controller_config_path = root / "configs/dapcs_mpc_v1/controller.yaml"
        self.common_estimator = SourceTimeCAInterface(self.estimator_config_path)
        self.safety = CommandSafetyEnvelope(self.controller_config_path)
        self.guidance = FRPNGuidance() if method == FRPN else None
        self.planner = VariableTimeStepMPC() if method == VTMPC else None
        if self.guidance is not None:
            assert self.guidance.gain == PUBLISHED_G and self.guidance.blend == PUBLISHED_W
        self.last_diagnostics: dict[str, Any] = {}
        self.reset(self.seed, "", "C0")

    @property
    def estimator(self):
        return self.common_estimator.estimator

    def reset(
        self, seed: int | None = None, trajectory_id: str = "", condition: str = "C0"
    ) -> None:
        self.seed = self.seed if seed is None else int(seed)
        self.trajectory_id = str(trajectory_id)
        self.condition = str(condition)
        self.common_estimator.reset()
        self.safety.reset()
        if self.guidance is not None:
            self.guidance.reset()
        if self.planner is not None:
            self.planner.reset()
        self.last_diagnostics = {}

    def _ca_logging_horizon(self) -> np.ndarray:
        return self.common_estimator.forecast(np.arange(1, 21, dtype=float) * 0.02)

    def step(
        self,
        interceptor_state_enu: np.ndarray,
        target_measurement_enu: TelemetryEvent,
        dt_s: float,
        sim_time_s: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if not isinstance(target_measurement_enu, TelemetryEvent):
            raise TypeError("external baseline control requires a timestamped TelemetryEvent")
        total_start = time.perf_counter_ns()
        interceptor = np.asarray(interceptor_state_enu, dtype=float).reshape(9)
        mean, covariance, estimator_info = self.common_estimator.process(target_measurement_enu)
        guidance_start = time.perf_counter_ns()
        if self.method == FRPN:
            assert self.guidance is not None
            raw, method_info = self.guidance.compute(
                mean[:3], mean[3:6], interceptor[:3], interceptor[3:6]
            )
            method_info.update(
                {
                    "solver_success": True,
                    "solver_status": "closed_form_FRPN",
                    "solve_time_s": (time.perf_counter_ns() - guidance_start) * 1e-9,
                }
            )
        else:
            assert self.planner is not None
            raw, method_info = self.planner.command_async(
                target_measurement_enu.arrival_timestamp_s, interceptor, mean
            )
        command, safety_info = self.safety.apply(raw, dt_s)
        forecasts = self._ca_logging_horizon()
        diagnostics: dict[str, Any] = {
            **estimator_info,
            **method_info,
            **safety_info,
            "target_estimate": mean,
            "current_covariance": covariance,
            "ca_predicted_position_horizon": forecasts[:, :3],
            "ca_predicted_velocity_horizon": forecasts[:, 3:6],
            "external_method_internal": INTERNAL_NAMES[self.method],
            "external_information_fairness": "same_corrected_M1_CA_posterior",
            "future_truth_access": 0,
            "executed_truth_controller_access": 0,
            "commanded_target_reference_access": 0,
        }
        diagnostics["controller_total_time_s"] = (
            time.perf_counter_ns() - total_start
        ) * 1e-9
        diagnostics.setdefault("belief_rollout_time_s", math.nan)
        diagnostics.setdefault("capture_selector_time_s", math.nan)
        self.last_diagnostics = diagnostics
        return command, dict(diagnostics)

    def get_diagnostics(self) -> dict[str, Any]:
        return dict(self.last_diagnostics)
