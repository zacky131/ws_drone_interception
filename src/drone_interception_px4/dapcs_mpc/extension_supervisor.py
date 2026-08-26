"""Extension-only supervisor that leaves the frozen RA-L supervisor untouched."""

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

from .controller import ExtensionControllerAdapter, METHODS


# The base class owns the validated PX4/Gazebo state machine. These two module
# bindings alter only its controller factory and accepted extension method set.
base.METHODS = METHODS
base.ExistingControllerAdapter = ExtensionControllerAdapter


def _scalar(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class ExtensionSupervisor(base.ExperimentSupervisor):
    def __init__(self, *args, **kwargs) -> None:
        self.extension_diagnostics: list[dict[str, Any]] = []
        super().__init__(*args, **kwargs)
        self.controller.reset(self.trial_seed, self.trajectory_id, self.condition.name)

    def _run_step(self, tick_start_ns: int) -> None:
        previous_rows = len(self.rows)
        super()._run_step(tick_start_ns)
        if len(self.rows) == previous_rows:
            return
        info = self.controller.get_diagnostics()
        row = self.rows[-1]
        probabilities = np.asarray(info.get("mode_probabilities", np.full(3, np.nan)), dtype=float)
        row.update({
            "imm_pi_cv": float(probabilities[0]),
            "imm_pi_ca": float(probabilities[1]),
            "imm_pi_singer": float(probabilities[2]),
            "imm_current_position_cov_trace": _scalar(info.get("position_covariance_trace")),
            "imm_current_velocity_cov_trace": _scalar(info.get("velocity_covariance_trace")),
            "selected_capture_time_s": _scalar(info.get("selected_capture_time_s")),
            "selected_capture_node": _scalar(info.get("selected_capture_node")),
            "selected_confidence_radius_m": _scalar(info.get("selected_confidence_radius_m")),
            "selected_coverage": _scalar(info.get("selected_coverage")),
            "imm_update_time_s": _scalar(info.get("imm_update_time_s")),
            "belief_rollout_time_s": _scalar(info.get("belief_rollout_time_s")),
            "capture_selector_time_s": _scalar(info.get("capture_selector_time_s")),
            "acados_solve_time_s": _scalar(info.get("solve_time_s")),
            "imm_last_update_source_timestamp_s": _scalar(
                info.get("last_update_source_timestamp_s")
            ),
            "imm_last_repropagation_steps": _scalar(info.get("last_repropagation_steps")),
            "capture_target_rule": str(info.get("capture_target_rule", "not_applicable")),
        })
        target = np.asarray(info.get("selected_capture_target", np.full(3, np.nan)), dtype=float)
        velocity = np.asarray(info.get("selected_capture_velocity", np.full(3, np.nan)), dtype=float)
        for index, axis in enumerate("xyz"):
            row[f"selected_capture_target_{axis}"] = float(target[index])
            row[f"selected_capture_velocity_{axis}"] = float(velocity[index])
        self.extension_diagnostics.append(info)

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
        sidecar = self.output_dir / "dapcs_diagnostics.npz"
        n_candidates = 8
        n_nodes = 54 if self.method in {"mpc_dimm_capture", "mpc_dapcs"} else 20
        np.savez_compressed(
            sidecar,
            candidate_times=self._stack(
                self.extension_diagnostics, "capture_candidate_times", (n_candidates,)
            ),
            deterministic_margins=self._stack(
                self.extension_diagnostics, "capture_candidate_deterministic_margins",
                (3, n_candidates) if self.method == "mpc_dapcs" else (n_candidates,),
            ),
            coverages=self._stack(
                self.extension_diagnostics, "capture_candidate_coverages", (n_candidates,)
            ),
            weighted_margins=self._stack(
                self.extension_diagnostics, "capture_candidate_weighted_margins", (n_candidates,)
            ),
            confidence_radii=self._stack(
                self.extension_diagnostics, "confidence_radii", (3, n_candidates)
            ),
            belief_mode_means=self._stack(
                self.extension_diagnostics, "belief_mode_means", (3, n_nodes, 9)
            ),
            belief_position_covariance_trace=self._stack(
                self.extension_diagnostics, "belief_covariance_position_trace", (3, n_nodes)
            ),
        )
        summary_path = self.output_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        summary["dapcs_diagnostics_file"] = str(sidecar)
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
    node: ExtensionSupervisor | None = None
    try:
        node = ExtensionSupervisor(
            args.trajectory, args.trajectory_id, args.family, args.condition,
            args.method, args.trial_seed, args.config, args.output_dir, args.timeout_s,
        )
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=.1)
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
