"""Paper attribution supervisor built on the frozen PX4/Gazebo state machine."""

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

from .controller import AttributionControllerAdapter, METHODS


base.METHODS = METHODS
base.ExistingControllerAdapter = AttributionControllerAdapter


def _scalar(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class AttributionSupervisor(base.ExperimentSupervisor):
    def __init__(self, *args, **kwargs) -> None:
        self.attribution_diagnostics: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)
        self.controller.reset(self.trial_seed, self.trajectory_id, self.condition.name)

    def _run_step(self, tick_start_ns: int) -> None:
        previous_rows = len(self.rows)
        super()._run_step(tick_start_ns)
        if len(self.rows) == previous_rows:
            return
        info = self.controller.get_diagnostics()
        row = self.rows[-1]
        probabilities = np.asarray(
            info.get("mode_probabilities", np.full(3, np.nan)), dtype=float
        )
        row.update(
            {
                "attr_pi_cv": float(probabilities[0]),
                "attr_pi_ca": float(probabilities[1]),
                "attr_pi_singer": float(probabilities[2]),
                "attr_position_cov_trace": _scalar(
                    info.get("position_covariance_trace")
                ),
                "attr_velocity_cov_trace": _scalar(
                    info.get("velocity_covariance_trace")
                ),
                "attr_last_update_source_timestamp_s": _scalar(
                    info.get("last_update_source_timestamp_s")
                ),
                "attr_last_repropagation_steps": _scalar(
                    info.get("last_repropagation_steps")
                ),
                "attr_nis": _scalar(info.get("last_nis")),
                "belief_rollout_time_s": _scalar(info.get("belief_rollout_time_s")),
                "capture_selector_time_s": _scalar(
                    info.get("capture_selector_time_s")
                ),
            }
        )
        self.attribution_diagnostics.append(info)

    def _write_outputs(self) -> None:
        super()._write_outputs()
        covariances = []
        for info in self.attribution_diagnostics:
            covariance = np.asarray(
                info.get("current_covariance", np.full((9, 9), np.nan)), dtype=float
            )
            if covariance.shape != (9, 9):
                covariance = np.full((9, 9), np.nan)
            covariances.append(covariance.astype(np.float32))
        sidecar = self.output_dir / "attribution_diagnostics.npz"
        np.savez_compressed(sidecar, current_covariance=np.asarray(covariances))
        summary_path = self.output_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["attribution_diagnostics_file"] = str(sidecar)
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
    node: AttributionSupervisor | None = None
    try:
        node = AttributionSupervisor(
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
