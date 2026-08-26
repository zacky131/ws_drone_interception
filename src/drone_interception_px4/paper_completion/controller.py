"""Three-arm interface for the frozen paper estimator-attribution campaign."""

from __future__ import annotations

import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from dapcs_mpc.controller import ExtensionControllerAdapter
from dapcs_mpc.delay_aware_imm import TelemetryPacket
from drone_interception_px4.controller_adapter import ExistingControllerAdapter
from drone_interception_px4.telemetry import TelemetryEvent

from .delay_aware_ca import DelayAwareCA


METHODS = ("mpc_ekf_ca", "mpc_dca_tracking", "mpc_dimm_tracking")


class AttributionControllerAdapter:
    """Keep tracking MPC fixed while varying only estimator temporal/model logic."""

    def __init__(self, method: str, config_path: str | Path, trial_seed: int) -> None:
        if method not in METHODS:
            raise ValueError(f"unsupported attribution method: {method}")
        self.method = method
        self.config_path = Path(config_path)
        self.seed = int(trial_seed)
        root = Path(os.environ.get("WS_DRONE_INTERCEPTION", Path(__file__).resolve().parents[3]))
        self.imm_config_path = root / "configs/dapcs_mpc_v1/imm.yaml"
        self.last_diagnostics: dict[str, Any] = {}
        if method == "mpc_ekf_ca":
            self.existing = ExistingControllerAdapter(method, config_path, trial_seed)
            self.extension = None
            self.estimator = None
            self.short_controller = None
        elif method == "mpc_dimm_tracking":
            self.existing = None
            self.extension = ExtensionControllerAdapter(method, config_path, trial_seed)
            self.estimator = None
            self.short_controller = None
        else:
            self.existing = None
            self.extension = None
            self.estimator = DelayAwareCA(self.imm_config_path)
            anchor = ExistingControllerAdapter("mpc_ekf_ca", config_path, trial_seed)
            self.short_controller = anchor.controller
        self.reset(trial_seed, "", "C0")

    def reset(
        self, seed: int | None = None, trajectory_id: str = "", condition: str = "C0"
    ) -> None:
        self.seed = self.seed if seed is None else int(seed)
        self.trajectory_id = str(trajectory_id)
        self.condition = str(condition)
        self.delay_s = {"C0": 0.05, "C1": 0.08, "C2": 0.12}[self.condition]
        if self.existing is not None:
            self.existing.reset()
        if self.extension is not None:
            self.extension.reset(self.seed, trajectory_id, condition)
        if self.estimator is not None:
            self.estimator.reset(0.0)
        if self.short_controller is not None:
            self.short_controller.reset()
        self.last_diagnostics = {}

    def _delayed_ca_step(
        self,
        interceptor_state_enu: np.ndarray,
        target_measurement_enu: TelemetryEvent,
        dt_s: float,
        sim_time_s: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        assert self.estimator is not None and self.short_controller is not None
        if not isinstance(target_measurement_enu, TelemetryEvent):
            raise TypeError("delay-aware control requires a timestamped TelemetryEvent")
        total_start = time.perf_counter_ns()
        arrival = target_measurement_enu.arrival_timestamp_s
        source = target_measurement_enu.actual_source_timestamp_s
        measurement = (
            np.zeros(6)
            if target_measurement_enu.measurement is None
            else target_measurement_enu.measurement
        )
        packet = TelemetryPacket(
            source_timestamp_s=source,
            arrival_timestamp_s=arrival,
            position=measurement[:3],
            velocity=measurement[3:6],
            valid=target_measurement_enu.valid,
            drop=target_measurement_enu.drop,
        )
        update_start = time.perf_counter_ns()
        self.estimator.process_packet(packet)
        update_time_s = (time.perf_counter_ns() - update_start) * 1e-9
        mean, covariance = self.estimator.state()
        solve_start = time.perf_counter_ns()
        command, info = self.short_controller.compute_control(
            np.asarray(interceptor_state_enu, dtype=float),
            None if packet.drop else packet.measurement,
            mean,
            np.zeros(3),
            float(arrival),
        )
        solve_elapsed_s = (time.perf_counter_ns() - solve_start) * 1e-9
        diagnostics = dict(info or {})
        estimator_diagnostics = self.estimator.diagnostics()
        times = np.arange(1, 21, dtype=float) * 0.02
        predictions = np.asarray([self.estimator.forecast_mean(t) for t in times])
        diagnostics.update(
            {
                "target_estimate": mean,
                "current_covariance": covariance,
                "mode_probabilities": estimator_diagnostics["mode_probabilities"],
                "position_covariance_trace": estimator_diagnostics[
                    "position_covariance_trace"
                ],
                "velocity_covariance_trace": estimator_diagnostics[
                    "velocity_covariance_trace"
                ],
                "last_update_source_timestamp_s": estimator_diagnostics[
                    "last_update_source_timestamp_s"
                ],
                "last_repropagation_steps": estimator_diagnostics[
                    "last_repropagation_steps"
                ],
                "last_nis": estimator_diagnostics["last_nis"],
                "estimator_time_s": update_time_s,
                "imm_update_time_s": update_time_s,
                "solve_time_s": diagnostics.get("solve_time_s", solve_elapsed_s),
                "ca_predicted_position_horizon": predictions[:, :3],
                "ca_predicted_velocity_horizon": predictions[:, 3:6],
                "belief_rollout_time_s": math.nan,
                "capture_selector_time_s": math.nan,
                "packet_source_timestamp_s": source,
                "packet_arrival_timestamp_s": arrival,
                "packet_delay_s": target_measurement_enu.physical_measurement_age_s,
                "configured_delay_s": target_measurement_enu.configured_delay_s,
                "requested_source_timestamp_s": (
                    target_measurement_enu.requested_source_timestamp_s
                ),
                "actual_measurement_source_timestamp_s": source,
                "measurement_history_left_timestamp_s": (
                    target_measurement_enu.history_left_timestamp_s
                ),
                "measurement_history_right_timestamp_s": (
                    target_measurement_enu.history_right_timestamp_s
                ),
                "measurement_interpolation_alpha": (
                    target_measurement_enu.interpolation_alpha
                ),
                "startup_clamped": int(target_measurement_enu.startup_clamped),
                "physical_measurement_age_s": (
                    target_measurement_enu.physical_measurement_age_s
                ),
            }
        )
        diagnostics["controller_total_time_s"] = (
            time.perf_counter_ns() - total_start
        ) * 1e-9
        self.last_diagnostics = diagnostics
        return np.asarray(command, dtype=float), dict(diagnostics)

    def step(
        self,
        interceptor_state_enu: np.ndarray,
        target_measurement_enu: TelemetryEvent | np.ndarray | None,
        dt_s: float,
        sim_time_s: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.existing is not None:
            command, info = self.existing.step(
                interceptor_state_enu, target_measurement_enu, dt_s, sim_time_s
            )
            self.last_diagnostics = dict(info)
            return command, info
        if self.extension is not None:
            command, info = self.extension.step(
                interceptor_state_enu, target_measurement_enu, dt_s, sim_time_s
            )
            self.last_diagnostics = dict(info)
            return command, info
        return self._delayed_ca_step(
            interceptor_state_enu, target_measurement_enu, dt_s, sim_time_s
        )

    def get_diagnostics(self) -> dict[str, Any]:
        return dict(self.last_diagnostics)
