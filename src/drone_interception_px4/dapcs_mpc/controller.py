"""Common four-method interface for the frozen DAPCS-MPC extension."""

from __future__ import annotations

import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import yaml

from drone_interception_px4.controller_adapter import ExistingControllerAdapter

from .belief_rollout import rollout_belief
from .delay_aware_imm import DelayAwareIMM, TelemetryPacket
from .deterministic_capture_selector import select_deterministic_capture
from .nonuniform_mpc import NonuniformCaptureMPC, make_grid, map_candidate_times
from .probabilistic_capture_selector import select_probabilistic_capture


METHODS = ("mpc_ekf_ca", "mpc_dimm_tracking", "mpc_dimm_capture", "mpc_dapcs")


class ExtensionControllerAdapter:
    """Expose reset/update/compute/diagnostics while retaining the old step API.

    M0 delegates every numerical operation to the existing adapter. M1 changes
    only the estimator supplied to the existing short-horizon controller. M2
    and M3 share one identical long-horizon OCP and differ only in selection.
    """

    def __init__(self, method: str, config_path: str | Path, trial_seed: int) -> None:
        if method not in METHODS:
            raise ValueError(f"unsupported extension method: {method}")
        self.method = method
        self.old_config_path = Path(config_path)
        self.seed = int(trial_seed)
        root = Path(os.environ.get("WS_DRONE_INTERCEPTION", Path(__file__).resolve().parents[3]))
        self.imm_config_path = root / "configs/dapcs_mpc_v1/imm.yaml"
        self.controller_config_path = root / "configs/dapcs_mpc_v1/controller.yaml"
        self.controller_config = yaml.safe_load(self.controller_config_path.read_text())
        self.condition = "C0"
        self.delay_s = 0.05
        self.last_diagnostics: dict[str, Any] = {}

        if method == "mpc_ekf_ca":
            self.existing = ExistingControllerAdapter(method, config_path, trial_seed)
            self.estimator = None
            self.short_controller = None
            self.long_controller = None
        else:
            self.existing = None
            self.estimator = DelayAwareIMM(self.imm_config_path)
            if method == "mpc_dimm_tracking":
                anchor = ExistingControllerAdapter("mpc_ekf_ca", config_path, trial_seed)
                self.short_controller = anchor.controller
                self.long_controller = None
            else:
                self.short_controller = None
                self.long_controller = NonuniformCaptureMPC(self.controller_config_path)
        self.grid = make_grid()
        candidates = np.asarray(self.controller_config["candidate_capture_times_s"], dtype=float)
        self.candidate_nodes, self.candidate_times = map_candidate_times(self.grid, candidates)
        self.reset(trial_seed, "", "C0")

    def reset(
        self, seed: int | None = None, trajectory_id: str = "", condition: str = "C0"
    ) -> None:
        self.seed = self.seed if seed is None else int(seed)
        self.trajectory_id = str(trajectory_id)
        self.condition = str(condition)
        self.delay_s = {"C0": .05, "C1": .08, "C2": .12}[self.condition]
        if self.existing is not None:
            self.existing.reset()
        if self.estimator is not None:
            self.estimator.reset(0.0)
        if self.short_controller is not None:
            self.short_controller.reset()
        if self.long_controller is not None:
            self.long_controller.reset()
        self.last_diagnostics = {}
        self._packet: TelemetryPacket | None = None

    def update_telemetry(self, packet: TelemetryPacket) -> None:
        self._packet = packet
        if self.estimator is not None:
            self.estimator.process_packet(packet)

    def _base_diagnostics(self, update_time_s: float) -> dict[str, Any]:
        assert self.estimator is not None
        mean, covariance = self.estimator.mixture_state()
        diagnostics = self.estimator.diagnostics()
        return {
            "target_estimate": mean,
            "mode_probabilities": diagnostics["mode_probabilities"],
            "position_covariance_trace": diagnostics["position_covariance_trace"],
            "velocity_covariance_trace": diagnostics["velocity_covariance_trace"],
            "last_update_source_timestamp_s": diagnostics["last_update_source_timestamp_s"],
            "last_repropagation_steps": diagnostics["last_repropagation_steps"],
            "imm_update_time_s": update_time_s,
            "estimator_time_s": update_time_s,
            "current_covariance": covariance,
        }

    def compute_command(self, now: float, pursuer_state: np.ndarray) -> np.ndarray:
        if self.method == "mpc_ekf_ca":
            raise RuntimeError("M0 uses the unchanged existing step path")
        assert self.estimator is not None
        total_start = time.perf_counter_ns()
        diagnostics = self._base_diagnostics(self._last_update_time_s)
        target_estimate = np.asarray(diagnostics["target_estimate"], dtype=float)

        if self.method == "mpc_dimm_tracking":
            solve_start = time.perf_counter_ns()
            command, info = self.short_controller.compute_control(
                np.asarray(pursuer_state, dtype=float),
                None if self._packet is None or self._packet.drop else self._packet.measurement,
                target_estimate,
                np.zeros(3),
                float(now),
            )
            solve_elapsed = (time.perf_counter_ns() - solve_start) * 1e-9
            diagnostics.update(dict(info or {}))
            diagnostics.setdefault("solve_time_s", solve_elapsed)
            horizon_times = np.arange(1, 21, dtype=float) * .02
            forecast_start = time.perf_counter_ns()
            belief = rollout_belief(self.estimator, horizon_times)
            diagnostics["belief_rollout_time_s"] = (time.perf_counter_ns() - forecast_start) * 1e-9
            mixture = belief.mixture_means()
            diagnostics["ca_predicted_position_horizon"] = mixture[:, :3]
            diagnostics["ca_predicted_velocity_horizon"] = mixture[:, 3:6]
            diagnostics["belief_mode_means"] = belief.means
            diagnostics["belief_covariance_position_trace"] = np.trace(
                belief.covariances[:, :, :3, :3], axis1=2, axis2=3
            )
            diagnostics["capture_selector_time_s"] = math.nan
        else:
            rollout_start = time.perf_counter_ns()
            belief = rollout_belief(self.estimator, self.grid.times_s)
            diagnostics["belief_rollout_time_s"] = (time.perf_counter_ns() - rollout_start) * 1e-9
            mixture = belief.mixture_means()
            constraints = self.controller_config["physical_constraints"]
            selector_start = time.perf_counter_ns()
            selector_args = (
                belief, self.candidate_nodes, self.candidate_times,
                np.asarray(pursuer_state[:3], dtype=float),
                np.asarray(pursuer_state[3:6], dtype=float),
                float(constraints["max_velocity_mps"]),
                float(constraints["max_acceleration_mps2"]),
                float(self.controller_config["capture_radius_m"]),
            )
            if self.method == "mpc_dimm_capture":
                selection = select_deterministic_capture(*selector_args)
                coverages = np.full(len(self.candidate_times), np.nan)
                weighted_margins = np.full(len(self.candidate_times), np.nan)
                radii = np.full((3, len(self.candidate_times)), np.nan)
                selected_radius = math.nan
                selected_coverage = math.nan
                target_rule = "mixture mean"
                margins = selection.margins_m
            else:
                selection_config = self.controller_config["probabilistic_selector"]
                selection = select_probabilistic_capture(
                    *selector_args,
                    epsilon=float(selection_config["epsilon"]),
                    required_coverage=float(selection_config["required_coverage"]),
                    beta_margin=float(selection_config["beta_margin"]),
                    beta_time=float(selection_config["beta_time"]),
                )
                coverages = selection.coverages
                weighted_margins = selection.weighted_margins_m
                radii = selection.confidence_radii_m
                selected_radius = selection.selected_confidence_radius_m
                selected_coverage = selection.selected_coverage
                target_rule = selection.target_rule
                margins = selection.mode_margins_m
            diagnostics["capture_selector_time_s"] = (
                time.perf_counter_ns() - selector_start
            ) * 1e-9
            selected_stage = int(selection.selected_node) - 1
            command, solver_info = self.long_controller.compute_command(
                pursuer_state, mixture[:, :3], mixture[:, 3:6], selected_stage,
                selection.target_position, selection.target_velocity,
            )
            diagnostics.update(solver_info)
            diagnostics.update({
                "selected_capture_time_s": selection.selected_time_s,
                "selected_capture_node": selection.selected_node,
                "selected_capture_target": selection.target_position,
                "selected_capture_velocity": selection.target_velocity,
                "capture_candidate_times": self.candidate_times.copy(),
                "capture_candidate_deterministic_margins": margins,
                "capture_candidate_coverages": coverages,
                "capture_candidate_weighted_margins": weighted_margins,
                "confidence_radii": radii,
                "selected_confidence_radius_m": selected_radius,
                "selected_coverage": selected_coverage,
                "capture_target_rule": target_rule,
                "ca_predicted_position_horizon": mixture[:20, :3],
                "ca_predicted_velocity_horizon": mixture[:20, 3:6],
                "belief_mode_means": belief.means,
                "belief_covariance_position_trace": np.trace(
                    belief.covariances[:, :, :3, :3], axis1=2, axis2=3
                ),
            })
        diagnostics["controller_total_time_s"] = (
            time.perf_counter_ns() - total_start
        ) * 1e-9 + self._last_update_time_s
        self.last_diagnostics = diagnostics
        return np.asarray(command, dtype=float)

    def get_diagnostics(self) -> dict[str, Any]:
        return dict(self.last_diagnostics)

    def step(
        self,
        interceptor_state_enu: np.ndarray,
        target_measurement_enu: np.ndarray | None,
        dt_s: float,
        sim_time_s: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self.existing is not None:
            command, info = self.existing.step(
                interceptor_state_enu, target_measurement_enu, dt_s, sim_time_s
            )
            self.last_diagnostics = dict(info)
            return command, info
        arrival = max(0.0, round(float(sim_time_s) / dt_s) * dt_s)
        source = max(0.0, math.floor((arrival - self.delay_s + 1e-12) / dt_s) * dt_s)
        measurement = (
            np.zeros(6) if target_measurement_enu is None
            else np.asarray(target_measurement_enu, dtype=float)
        )
        packet = TelemetryPacket(
            source_timestamp_s=source,
            arrival_timestamp_s=arrival,
            position=measurement[:3],
            velocity=measurement[3:6],
            valid=target_measurement_enu is not None,
            drop=target_measurement_enu is None,
        )
        update_start = time.perf_counter_ns()
        self.update_telemetry(packet)
        self._last_update_time_s = (time.perf_counter_ns() - update_start) * 1e-9
        command = self.compute_command(arrival, np.asarray(interceptor_state_enu, dtype=float))
        return command, self.get_diagnostics()
