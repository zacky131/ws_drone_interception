"""Geometry-only extension of the frozen corrected M0prime--M1 supervisor."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import rclpy

from drone_interception_px4 import experiment_supervisor as experiment_base
from drone_interception_px4.px4_contract import INTERCEPTOR, TARGET
from rev6_corrected_unified import supervisor as unified


TARGET_INITIAL_ENU_M = np.array([30.0, 0.0, 10.0])
R0_ENU_M = np.array([-30.0, 0.0, 0.0])


def rotate_z(vector: np.ndarray, degrees: float) -> np.ndarray:
    angle = math.radians(float(degrees))
    return np.array(
        [[math.cos(angle), -math.sin(angle), 0.0],
         [math.sin(angle), math.cos(angle), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=float,
    ) @ np.asarray(vector, dtype=float)


e_h = R0_ENU_M / np.linalg.norm(R0_ENU_M[:2])
GEOMETRY_RELATIVE_ENU_M = {
    "G1": 30.0 * rotate_z(e_h, 90.0),
    "G2": (
        30.0 * math.cos(math.radians(15.0)) * rotate_z(e_h, 45.0)
        + np.array([0.0, 0.0, 30.0 * math.sin(math.radians(15.0))])
    ),
}


class GeometrySupervisor(unified.UnifiedSupervisor):
    def __init__(self, geometry: str, *args: Any, **kwargs: Any) -> None:
        if geometry not in GEOMETRY_RELATIVE_ENU_M:
            raise ValueError(f"unknown geometry: {geometry}")
        self.geometry = geometry
        self.expected_relative_enu_m = GEOMETRY_RELATIVE_ENU_M[geometry].copy()
        self.expected_interceptor_initial_enu_m = (
            TARGET_INITIAL_ENU_M + self.expected_relative_enu_m
        )
        # Gazebo models spawn on the common ground plane at the prescribed
        # horizontal coordinates.  The vertical engagement offset is realized
        # by the interceptor takeoff setpoint, preserving the target spawn/state.
        experiment_base.SPAWN_ENU = {
            INTERCEPTOR.role: np.array(
                [self.expected_interceptor_initial_enu_m[0],
                 self.expected_interceptor_initial_enu_m[1], 0.0]
            ),
            TARGET.role: np.array([30.0, 0.0, 0.0]),
        }
        super().__init__(*args, **kwargs)
        self.trial_id = f"{geometry}__{self.trial_id}"

    def _tick(self) -> None:
        previous_state = self.state
        super()._tick()
        if previous_state == "WAIT_FOR_PX4" and self.state == "ARM_TARGET":
            assert self.interceptor_takeoff_local_enu is not None
            vertical_offset = float(self.expected_relative_enu_m[2])
            self.interceptor_takeoff_local_enu[2] += vertical_offset

    def _run_step(self, tick_start_ns: int) -> None:
        previous = len(self.rows)
        super()._run_step(tick_start_ns)
        if len(self.rows) > previous:
            self.rows[-1]["geometry"] = self.geometry

    def _write_outputs(self) -> None:
        super()._write_outputs()
        summary_path = self.output_dir / "summary.json"
        summary = json.loads(summary_path.read_text())
        achieved = summary.get("initial_achieved_states", {})
        interceptor = np.asarray(achieved.get("interceptor_global_enu", [math.nan] * 3), float)
        target = np.asarray(achieved.get("target_global_enu", [math.nan] * 3), float)
        relative = interceptor - target
        summary.update(
            {
                "geometry": self.geometry,
                "geometry_change": "interceptor initial position only",
                "target_initial_expected_enu_m": TARGET_INITIAL_ENU_M.tolist(),
                "interceptor_initial_expected_enu_m": self.expected_interceptor_initial_enu_m.tolist(),
                "relative_initial_expected_enu_m": self.expected_relative_enu_m.tolist(),
                "initial_separation_expected_m": 30.0,
                "relative_initial_achieved_enu_m": relative.tolist(),
                "initial_separation_achieved_m": float(np.linalg.norm(relative)),
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        metadata_path = self.output_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.update(
            {
                "geometry": self.geometry,
                "target_initial_expected_enu_m": TARGET_INITIAL_ENU_M.tolist(),
                "interceptor_initial_expected_enu_m": self.expected_interceptor_initial_enu_m.tolist(),
                "relative_initial_expected_enu_m": self.expected_relative_enu_m.tolist(),
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def main(argv: list[str] | None = None) -> int:
    source_root = Path(os.environ.get("DRONE_INTERCEPTION_V3", "."))
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", choices=sorted(GEOMETRY_RELATIVE_ENU_M), required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--trajectory-id", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--condition", choices=sorted(experiment_base.CONDITIONS), required=True)
    parser.add_argument("--method", choices=("A0prime_CA_arrival", "mpc_dca_tracking"), required=True)
    parser.add_argument("--trial-seed", type=int, required=True)
    parser.add_argument(
        "--config", type=Path,
        default=source_root / "configs/q2_revision_pilot.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=90.0)
    args, ros_args = parser.parse_known_args(argv)
    rclpy.init(args=ros_args)
    node: GeometrySupervisor | None = None
    try:
        node = GeometrySupervisor(
            args.geometry, args.trajectory, args.trajectory_id, args.family,
            args.condition, args.method, args.trial_seed, args.config,
            args.output_dir, args.timeout_s,
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
