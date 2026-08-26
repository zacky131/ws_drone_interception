"""Clean B2 controller: validated delayed CA plus frozen deterministic capture MPC."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from dapcs_mpc.delay_aware_imm import TelemetryPacket
from dapcs_mpc.deterministic_capture_selector import select_deterministic_capture
from dapcs_mpc.nonuniform_mpc import NonuniformCaptureMPC, make_grid, map_candidate_times
from paper_completion.delay_aware_ca import DelayAwareCA


METHODS = ("mpc_dca_capture",)


@dataclass(frozen=True)
class DelayedCAMeanHorizon:
    """Minimal mean-only interface consumed by the frozen deterministic selector."""

    times_s: np.ndarray
    means: np.ndarray

    def mixture_means(self) -> np.ndarray:
        # The deterministic M2 selector consumes only this method.  There is
        # exactly one CA mean and no IMM probability or covariance operation.
        return self.means


class ClosureControllerAdapter:
    """Combine A1's estimator unchanged with M2's deterministic long MPC."""

    def __init__(self, method: str, config_path: str | Path, trial_seed: int) -> None:
        if method not in METHODS:
            raise ValueError(f"unsupported closure method: {method}")
        self.method = method
        self.old_config_path = Path(config_path)
        self.seed = int(trial_seed)
        root = Path(os.environ.get("WS_DRONE_INTERCEPTION", Path(__file__).resolve().parents[3]))
        self.estimator_config_path = root / "configs/dapcs_mpc_v1/imm.yaml"
        self.controller_config_path = root / "configs/dapcs_mpc_v1/controller.yaml"
        self.controller_config = yaml.safe_load(self.controller_config_path.read_text())
        self.estimator = DelayAwareCA(self.estimator_config_path)
        self.long_controller = NonuniformCaptureMPC(self.controller_config_path)
        self.grid = make_grid()
        candidates = np.asarray(self.controller_config["candidate_capture_times_s"], dtype=float)
        self.candidate_nodes, self.candidate_times = map_candidate_times(self.grid, candidates)
        self.last_diagnostics: dict[str, Any] = {}
        self.reset(trial_seed, "", "C0")

    def reset(
        self, seed: int | None = None, trajectory_id: str = "", condition: str = "C0"
    ) -> None:
        self.seed = self.seed if seed is None else int(seed)
        self.trajectory_id = str(trajectory_id)
        self.condition = str(condition)
        self.delay_s = {"C0": 0.05, "C1": 0.08, "C2": 0.12}[self.condition]
        self.estimator.reset(0.0)
        self.long_controller.reset()
        self.last_diagnostics = {}
        self._packet: TelemetryPacket | None = None
        self._last_update_time_s = 0.0
        self._step_start_ns = 0

    def make_packet(
        self,
        target_measurement_enu: np.ndarray | None,
        dt_s: float,
        sim_time_s: float,
    ) -> TelemetryPacket:
        """Use the exact A1 source/arrival timestamp quantization semantics."""
        arrival = max(0.0, round(float(sim_time_s) / dt_s) * dt_s)
        source = max(
            0.0, math.floor((arrival - self.delay_s + 1e-12) / dt_s) * dt_s
        )
        measurement = (
            np.zeros(6)
            if target_measurement_enu is None
            else np.asarray(target_measurement_enu, dtype=float)
        )
        return TelemetryPacket(
            source_timestamp_s=source,
            arrival_timestamp_s=arrival,
            position=measurement[:3],
            velocity=measurement[3:6],
            valid=target_measurement_enu is not None,
            drop=target_measurement_enu is None,
        )

    def update_telemetry(self, packet: TelemetryPacket) -> None:
        self._packet = packet
        start = time.perf_counter_ns()
        self.estimator.process_packet(packet)
        self._last_update_time_s = (time.perf_counter_ns() - start) * 1e-9

    def rollout_delayed_ca(self) -> DelayedCAMeanHorizon:
        start = time.perf_counter_ns()
        means = np.asarray(
            [self.estimator.forecast_mean(float(t)) for t in self.grid.times_s],
            dtype=float,
        )
        self._last_prediction_time_s = (time.perf_counter_ns() - start) * 1e-9
        return DelayedCAMeanHorizon(self.grid.times_s.copy(), means)

    def compute_command(self, now: float, pursuer_state: np.ndarray) -> np.ndarray:
        mean, covariance = self.estimator.state()
        estimator_diagnostics = self.estimator.diagnostics()
        horizon = self.rollout_delayed_ca()
        constraints = self.controller_config["physical_constraints"]
        selector_start = time.perf_counter_ns()
        selection = select_deterministic_capture(
            horizon,
            self.candidate_nodes,
            self.candidate_times,
            np.asarray(pursuer_state[:3], dtype=float),
            np.asarray(pursuer_state[3:6], dtype=float),
            float(constraints["max_velocity_mps"]),
            float(constraints["max_acceleration_mps2"]),
            float(self.controller_config["capture_radius_m"]),
        )
        selector_time_s = (time.perf_counter_ns() - selector_start) * 1e-9
        selected_stage = int(selection.selected_node) - 1
        command, solver_info = self.long_controller.compute_command(
            np.asarray(pursuer_state, dtype=float),
            horizon.means[:, :3],
            horizon.means[:, 3:6],
            selected_stage,
            selection.target_position,
            selection.target_velocity,
        )
        diagnostics: dict[str, Any] = {
            "target_estimate": mean,
            "current_covariance": covariance,
            "position_covariance_trace": estimator_diagnostics["position_covariance_trace"],
            "velocity_covariance_trace": estimator_diagnostics["velocity_covariance_trace"],
            "last_update_source_timestamp_s": estimator_diagnostics[
                "last_update_source_timestamp_s"
            ],
            "last_repropagation_steps": estimator_diagnostics["last_repropagation_steps"],
            "last_nis": estimator_diagnostics["last_nis"],
            "estimator_time_s": self._last_update_time_s,
            "future_target_prediction_time_s": self._last_prediction_time_s,
            "belief_rollout_time_s": self._last_prediction_time_s,
            "capture_selector_time_s": selector_time_s,
            "selected_capture_time_s": selection.selected_time_s,
            "selected_capture_node": selection.selected_node,
            "selected_capture_target": selection.target_position,
            "selected_capture_velocity": selection.target_velocity,
            "capture_candidate_times": self.candidate_times.copy(),
            "capture_candidate_deterministic_margins": selection.margins_m.copy(),
            "capture_candidate_reachable_distances": selection.reachable_distances_m.copy(),
            "capture_target_rule": "delayed_ca_mean",
            "ca_predicted_position_horizon": horizon.means[:20, :3],
            "ca_predicted_velocity_horizon": horizon.means[:20, 3:6],
            "delayed_ca_predicted_state_horizon": horizon.means,
        }
        diagnostics.update(solver_info)
        start_ns = self._step_start_ns or time.perf_counter_ns()
        diagnostics["controller_total_time_s"] = (time.perf_counter_ns() - start_ns) * 1e-9
        self.last_diagnostics = diagnostics
        return np.asarray(command, dtype=float)

    def step(
        self,
        interceptor_state_enu: np.ndarray,
        target_measurement_enu: np.ndarray | None,
        dt_s: float,
        sim_time_s: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._step_start_ns = time.perf_counter_ns()
        packet = self.make_packet(target_measurement_enu, dt_s, sim_time_s)
        self.update_telemetry(packet)
        command = self.compute_command(
            packet.arrival_timestamp_s, np.asarray(interceptor_state_enu, dtype=float)
        )
        return command, self.get_diagnostics()

    def get_diagnostics(self) -> dict[str, Any]:
        return dict(self.last_diagnostics)
