"""PX4/Gazebo supervisor for the isolated two-arm confirmatory experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import rclpy

from drone_interception_px4 import experiment_supervisor as base
from paper_completion.attribution_supervisor import AttributionSupervisor

from .controller import ConfirmatoryControllerAdapter, METHODS


# The base supervisor deliberately exposes these two campaign hooks.  Limit
# this process to exactly the two confirmatory methods.
base.METHODS = METHODS
base.ExistingControllerAdapter = ConfirmatoryControllerAdapter


def _scalar(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class ConfirmatorySupervisor(AttributionSupervisor):
    def _run_step(self, tick_start_ns: int) -> None:
        previous_rows = len(self.rows)
        super()._run_step(tick_start_ns)
        if len(self.rows) == previous_rows:
            return
        info = self.controller.get_diagnostics()
        self.rows[-1].update(
            {
                "confirm_packet_source_timestamp_s": _scalar(
                    info.get("packet_source_timestamp_s")
                ),
                "confirm_packet_arrival_timestamp_s": _scalar(
                    info.get("packet_arrival_timestamp_s")
                ),
                "confirm_packet_delay_s": _scalar(info.get("packet_delay_s")),
                "confirm_configured_delay_s": _scalar(
                    info.get("configured_delay_s")
                ),
                "confirm_requested_source_timestamp_s": _scalar(
                    info.get("requested_source_timestamp_s")
                ),
                "confirm_actual_measurement_source_timestamp_s": _scalar(
                    info.get("actual_measurement_source_timestamp_s")
                ),
                "confirm_measurement_history_left_timestamp_s": _scalar(
                    info.get("measurement_history_left_timestamp_s")
                ),
                "confirm_measurement_history_right_timestamp_s": _scalar(
                    info.get("measurement_history_right_timestamp_s")
                ),
                "confirm_measurement_interpolation_alpha": _scalar(
                    info.get("measurement_interpolation_alpha")
                ),
                "confirm_startup_clamped": int(info.get("startup_clamped", 0)),
                "confirm_physical_measurement_age_s": _scalar(
                    info.get("physical_measurement_age_s")
                ),
                "confirm_packet_accepted": int(info.get("packet_accepted", 0)),
                "confirm_measurement_update_timestamp_s": _scalar(
                    info.get("measurement_update_timestamp_s")
                ),
                "confirm_posterior_timestamp_s": _scalar(
                    info.get("posterior_timestamp_s")
                ),
            }
        )

    def _write_outputs(self) -> None:
        super()._write_outputs()
        summary_path = self.output_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["capture_distance_m"] = 1.0
        summary["ground_truth_for_estimation"] = "executed_px4_target_state"
        summary["confirmatory_method_set"] = list(METHODS)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    source_root = Path(os.environ.get("DRONE_INTERCEPTION_V3", "."))
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--trajectory-id", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--condition", choices=sorted(base.CONDITIONS), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--trial-seed", type=int, required=True)
    parser.add_argument(
        "--config", type=Path, default=source_root / "configs/q2_revision_pilot.yaml"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    args, ros_args = parser.parse_known_args(argv)
    rclpy.init(args=ros_args)
    node: ConfirmatorySupervisor | None = None
    try:
        node = ConfirmatorySupervisor(
            args.trajectory,
            args.trajectory_id,
            args.family,
            args.condition,
            args.method,
            args.trial_seed,
            args.config,
            args.output_dir,
            args.timeout_s,
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
