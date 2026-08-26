"""PX4/Gazebo supervisor for the single final-closure B2 method."""

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

from .controller import ClosureControllerAdapter, METHODS


base.METHODS = METHODS
base.ExistingControllerAdapter = ClosureControllerAdapter


def _scalar(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class ClosureSupervisor(base.ExperimentSupervisor):
    def __init__(self, *args, **kwargs) -> None:
        self.closure_diagnostics: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)
        self.controller.reset(self.trial_seed, self.trajectory_id, self.condition.name)

    def _run_step(self, tick_start_ns: int) -> None:
        previous_rows = len(self.rows)
        super()._run_step(tick_start_ns)
        if len(self.rows) == previous_rows:
            return
        info = self.controller.get_diagnostics()
        row = self.rows[-1]
        row.update(
            {
                "dca_position_cov_trace": _scalar(info.get("position_covariance_trace")),
                "dca_velocity_cov_trace": _scalar(info.get("velocity_covariance_trace")),
                "dca_last_update_source_timestamp_s": _scalar(
                    info.get("last_update_source_timestamp_s")
                ),
                "dca_last_repropagation_steps": _scalar(
                    info.get("last_repropagation_steps")
                ),
                "dca_nis": _scalar(info.get("last_nis")),
                "future_target_prediction_time_s": _scalar(
                    info.get("future_target_prediction_time_s")
                ),
                "capture_selector_time_s": _scalar(info.get("capture_selector_time_s")),
                "selected_capture_time_s": _scalar(info.get("selected_capture_time_s")),
                "selected_capture_node": _scalar(info.get("selected_capture_node")),
                "finite_capture_margin_count": int(
                    np.isfinite(
                        np.asarray(
                            info.get("capture_candidate_deterministic_margins", []),
                            dtype=float,
                        )
                    ).sum()
                ),
                "capture_target_rule": str(
                    info.get("capture_target_rule", "not_applicable")
                ),
            }
        )
        target = np.asarray(info.get("selected_capture_target", np.full(3, np.nan)))
        velocity = np.asarray(info.get("selected_capture_velocity", np.full(3, np.nan)))
        for index, axis in enumerate("xyz"):
            row[f"selected_capture_target_{axis}"] = float(target[index])
            row[f"selected_capture_velocity_{axis}"] = float(velocity[index])
        self.closure_diagnostics.append(info)

    @staticmethod
    def _stack(items: list[dict[str, Any]], key: str, shape: tuple[int, ...]) -> np.ndarray:
        values = []
        for item in items:
            value = np.asarray(item.get(key, np.full(shape, np.nan)), dtype=np.float32)
            if value.shape != shape:
                value = np.full(shape, np.nan, dtype=np.float32)
            values.append(value)
        return np.asarray(values, dtype=np.float32)

    def _write_outputs(self) -> None:
        super()._write_outputs()
        sidecar = self.output_dir / "closure_diagnostics.npz"
        np.savez_compressed(
            sidecar,
            current_covariance=self._stack(
                self.closure_diagnostics, "current_covariance", (9, 9)
            ),
            candidate_times=self._stack(
                self.closure_diagnostics, "capture_candidate_times", (8,)
            ),
            deterministic_margins=self._stack(
                self.closure_diagnostics,
                "capture_candidate_deterministic_margins",
                (8,),
            ),
            reachable_distances=self._stack(
                self.closure_diagnostics,
                "capture_candidate_reachable_distances",
                (8,),
            ),
            predicted_ca_state=self._stack(
                self.closure_diagnostics, "delayed_ca_predicted_state_horizon", (54, 9)
            ),
        )
        summary_path = self.output_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["closure_diagnostics_file"] = str(sidecar)
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
    node: ClosureSupervisor | None = None
    try:
        node = ClosureSupervisor(
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
