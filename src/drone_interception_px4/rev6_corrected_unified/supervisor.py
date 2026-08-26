"""PX4/Gazebo supervisor for the corrected five-method Rev6 campaign."""

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

from .controller import MANUSCRIPT_NAMES, METHODS, UnifiedControllerAdapter


base.METHODS = METHODS
base.ExistingControllerAdapter = UnifiedControllerAdapter


def scalar(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class UnifiedSupervisor(base.ExperimentSupervisor):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.controller.reset(self.trial_seed, self.trajectory_id, self.condition.name)

    def _run_step(self, tick_start_ns: int) -> None:
        previous = len(self.rows)
        super()._run_step(tick_start_ns)
        if len(self.rows) == previous:
            return
        info = self.controller.get_diagnostics()
        probabilities = np.asarray(info.get("mode_probabilities", np.full(3, np.nan)), float)
        row = self.rows[-1]
        row.update(
            {
                "method_manuscript": MANUSCRIPT_NAMES[self.method],
                "unified_packet_source_timestamp_s": scalar(
                    info.get("packet_source_timestamp_s")
                ),
                "unified_packet_arrival_timestamp_s": scalar(
                    info.get("packet_arrival_timestamp_s")
                ),
                "unified_packet_accepted": int(info.get("packet_accepted", 0)),
                "unified_measurement_update_timestamp_s": scalar(
                    info.get("measurement_update_timestamp_s")
                ),
                "unified_posterior_timestamp_s": scalar(
                    info.get("posterior_timestamp_s")
                ),
                "unified_last_update_source_timestamp_s": scalar(
                    info.get("last_update_source_timestamp_s")
                ),
                "unified_last_repropagation_steps": scalar(
                    info.get("last_repropagation_steps")
                ),
                "unified_position_covariance_trace": scalar(
                    info.get("position_covariance_trace")
                ),
                "unified_velocity_covariance_trace": scalar(
                    info.get("velocity_covariance_trace")
                ),
                "unified_pi_cv": float(probabilities[0]),
                "unified_pi_ca": float(probabilities[1]),
                "unified_pi_singer": float(probabilities[2]),
                "future_target_prediction_time_s": scalar(
                    info.get("future_target_prediction_time_s")
                ),
                "belief_rollout_time_s": scalar(info.get("belief_rollout_time_s")),
                "capture_selector_time_s": scalar(info.get("capture_selector_time_s")),
            }
        )

    def _write_outputs(self) -> None:
        super()._write_outputs()
        path = self.output_dir / "summary.json"
        summary = json.loads(path.read_text())
        summary.update(
            {
                "method_manuscript": MANUSCRIPT_NAMES[self.method],
                "capture_distance_m": 1.0,
                "ground_truth_for_estimation": "executed_px4_target_state",
                "telemetry_timestamp_semantics": "exact_physical_source_time",
                "unified_method_set": list(METHODS),
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
    node: UnifiedSupervisor | None = None
    try:
        node = UnifiedSupervisor(
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
