"""PX4/Gazebo supervisor for the isolated Rev6 external baselines."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import rclpy

from drone_interception_px4 import experiment_supervisor as base

from .controller import (
    ExternalBaselineControllerAdapter, INTERNAL_NAMES, MANUSCRIPT_NAMES, METHODS,
)


base.METHODS = METHODS
base.ExistingControllerAdapter = ExternalBaselineControllerAdapter


def scalar(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class ExternalBaselineSupervisor(base.ExperimentSupervisor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.controller.reset(self.trial_seed, self.trajectory_id, self.condition.name)

    def _run_step(self, tick_start_ns: int) -> None:
        previous = len(self.rows)
        super()._run_step(tick_start_ns)
        if len(self.rows) == previous:
            return
        info = self.controller.get_diagnostics()
        row = self.rows[-1]
        row.update(
            {
                "method_manuscript": MANUSCRIPT_NAMES[self.method],
                "external_method_internal": INTERNAL_NAMES[self.method],
                "external_packet_source_timestamp_s": scalar(info.get("packet_source_timestamp_s")),
                "external_packet_arrival_timestamp_s": scalar(info.get("packet_arrival_timestamp_s")),
                "external_packet_accepted": int(info.get("packet_accepted", 0)),
                "external_measurement_update_timestamp_s": scalar(info.get("measurement_update_timestamp_s")),
                "external_posterior_timestamp_s": scalar(info.get("posterior_timestamp_s")),
                "external_last_update_source_timestamp_s": scalar(info.get("last_update_source_timestamp_s")),
                "external_last_repropagation_steps": scalar(info.get("last_repropagation_steps")),
                "external_position_covariance_trace": scalar(info.get("position_covariance_trace")),
                "external_velocity_covariance_trace": scalar(info.get("velocity_covariance_trace")),
                "external_envelope_clipped": int(info.get("external_envelope_clipped", 0)),
                "external_rate_limited": int(info.get("external_rate_limited", 0)),
                "external_command_delta_norm_mps2": scalar(info.get("external_command_delta_norm_mps2")),
                "frpn_time_to_go_s": scalar(info.get("frpn_time_to_go_s")),
                "frpn_speed_guard_activated": int(info.get("frpn_speed_guard_activated", 0)),
                "frpn_time_guard_activated": int(info.get("frpn_time_guard_activated", 0)),
                "frpn_speed_guard_activations_total": int(info.get("frpn_speed_guard_activations_total", 0)),
                "frpn_time_guard_activations_total": int(info.get("frpn_time_guard_activations_total", 0)),
                "vtmpc_replan_attempted": int(info.get("vtmpc_replan_attempted", 0)),
                "vtmpc_replan_completed": int(info.get("vtmpc_replan_completed", 0)),
                "vtmpc_replan_interval_s": scalar(info.get("vtmpc_replan_interval_s")),
                "vtmpc_reused_previous_plan": int(info.get("vtmpc_reused_previous_plan", 0)),
                "vtmpc_has_valid_plan": int(info.get("vtmpc_has_valid_plan", 0)),
                "vtmpc_solver_pending": int(info.get("vtmpc_solver_pending", 0)),
                "vtmpc_active_segment": int(info.get("vtmpc_active_segment", -1)),
                "vtmpc_timestep_sum_s": scalar(info.get("vtmpc_timestep_sum_s")),
                "vtmpc_first_timestep_s": scalar(info.get("vtmpc_first_timestep_s")),
                "vtmpc_min_timestep_s": scalar(info.get("vtmpc_min_timestep_s")),
                "vtmpc_max_timestep_s": scalar(info.get("vtmpc_max_timestep_s")),
                "vtmpc_max_velocity_mps": scalar(info.get("vtmpc_max_velocity_mps")),
                "vtmpc_max_acceleration_mps2": scalar(info.get("vtmpc_max_acceleration_mps2")),
                "vtmpc_max_jerk_mps3": scalar(info.get("vtmpc_max_jerk_mps3")),
                "vtmpc_solve_attempts_total": int(info.get("vtmpc_solve_attempts_total", 0)),
                "vtmpc_solve_successes_total": int(info.get("vtmpc_solve_successes_total", 0)),
                "vtmpc_solve_failures_total": int(info.get("vtmpc_solve_failures_total", 0)),
                "external_future_truth_access": int(info.get("future_truth_access", 0)),
                "external_executed_truth_controller_access": int(info.get("executed_truth_controller_access", 0)),
                "external_commanded_reference_access": int(info.get("commanded_target_reference_access", 0)),
            }
        )

    def _write_outputs(self) -> None:
        super()._write_outputs()
        path = self.output_dir / "summary.json"
        summary = json.loads(path.read_text())
        summary.update(
            {
                "method_manuscript": MANUSCRIPT_NAMES[self.method],
                "external_method_internal": INTERNAL_NAMES[self.method],
                "capture_distance_m": 1.0,
                "ground_truth_for_estimation": "executed_px4_target_state_metrics_only",
                "controller_information_source": "corrected_M1_exact_source_time_CA_posterior",
                "telemetry_timestamp_semantics": "exact_physical_source_time",
                "external_method_set": list(METHODS),
                "future_truth_controller_access": False,
                "executed_truth_controller_access": False,
            }
        )
        path.write_text(json.dumps(summary, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    source_root = Path(os.environ.get("DRONE_INTERCEPTION_V3", "."))
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--trajectory-id", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--condition", choices=sorted(base.CONDITIONS), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--trial-seed", type=int, required=True)
    parser.add_argument("--config", type=Path, default=source_root / "configs/q2_revision_pilot.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    args, ros_args = parser.parse_known_args(argv)
    rclpy.init(args=ros_args)
    node: ExternalBaselineSupervisor | None = None
    try:
        node = ExternalBaselineSupervisor(
            args.trajectory, args.trajectory_id, args.family, args.condition,
            args.method, args.trial_seed, args.config, args.output_dir, args.timeout_s,
        )
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        if node is not None:
            node._finish(True, "interrupted")
    finally:
        success = bool(node is not None and node.success)
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
