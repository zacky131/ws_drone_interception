"""Thin imports and state translation for the frozen existing controllers."""

from __future__ import annotations

import copy
from dataclasses import replace
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


METHODS = (
    "baseline_pn_6dof",
    "mpc_ekf_ca",
    "narx_ca_ungated_5hz",
    "narx_ca_gated_5hz",
)


class ExistingControllerAdapter:
    def __init__(self, method: str, config_path: str | Path, trial_seed: int) -> None:
        if method not in METHODS:
            raise ValueError(f"unsupported primary method: {method}")
        source_root = os.environ.get("DRONE_INTERCEPTION_V3", "")
        if not source_root:
            raise RuntimeError("DRONE_INTERCEPTION_V3 is not set")
        if source_root not in sys.path:
            sys.path.insert(0, source_root)

        from src.utils.config_schema import load_config
        from scripts.run_single_case import apply_ablation_overrides, build_controller, build_estimator

        cfg = load_config(str(config_path))
        cfg.simulation.dt = 0.02
        cfg.simulation.success_distance = 1.0
        cfg.controller.solver = "acados"
        cfg.controller.narx_seed = int(trial_seed)
        # Fairness gate: all MPC methods use identical cost weights.
        cfg.controller.narx_Q_pos = 0.0
        cfg.controller.narx_Q_terminal_pos = 0.0
        cfg.controller.mpc_ca_Q_pos = 0.0
        cfg.controller.mpc_ca_Q_terminal_pos = 0.0

        if method.startswith("narx_"):
            selected = None
            for variant in cfg.controller.narx_training_variants:
                if variant.get("label") == method:
                    selected = variant
                    break
            if selected is None:
                raise ValueError(f"missing frozen NARX variant {method}")
            for key, value in selected.items():
                if key != "label":
                    setattr(cfg.controller, key, value)

        # The existing runner stores the gated/ungated NARX distinction in
        # ``narx_training_variants`` while its ablation registry uses the
        # common implementation key ``mpc_ekf_narx``.
        registry_method = "mpc_ekf_narx" if method.startswith("narx_") else method
        apply_ablation_overrides(cfg, registry_method)
        self.method = method
        self.config = cfg
        self.estimator = build_estimator(cfg)
        self.controller = build_controller(cfg)
        self.reset()

    def reset(self) -> None:
        self.estimator.reset()
        self.controller.reset()

    def step(
        self,
        interceptor_state_enu: np.ndarray,
        target_measurement_enu: np.ndarray | None,
        dt_s: float,
        sim_time_s: float,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        start_ns = time.perf_counter_ns()
        estimate_start_ns = time.perf_counter_ns()
        self.estimator.predict(dt_s)
        if target_measurement_enu is not None:
            self.estimator.update(target_measurement_enu)
        target_estimate = self.estimator.get_estimate()
        estimator_time_s = (time.perf_counter_ns() - estimate_start_ns) * 1e-9
        command, info = self.controller.compute_control(
            np.asarray(interceptor_state_enu, dtype=float),
            target_measurement_enu,
            target_estimate,
            np.zeros(3),
            sim_time_s,
        )
        info = dict(info or {})
        horizon_steps = int(getattr(self.controller, "N", 0))
        horizon_dt = float(getattr(self.controller, "dt", dt_s))
        target_state = np.asarray(target_estimate, dtype=float)
        if horizon_steps > 0 and len(target_state) >= 9:
            times = np.arange(1, horizon_steps + 1, dtype=float) * horizon_dt
            ca_horizon = np.column_stack(
                [
                    target_state[:3]
                    + target_state[3:6] * t
                    + 0.5 * target_state[6:9] * t * t
                    for t in times
                ]
            ).T
            info["ca_predicted_position_horizon"] = ca_horizon
            info["ca_predicted_velocity_horizon"] = np.column_stack(
                [target_state[3:6] + target_state[6:9] * t for t in times]
            ).T
        issued = getattr(self.controller, "_issued_predictions", None)
        if issued:
            latest = issued[-1]
            baseline = np.asarray(latest["baseline_wp"], dtype=float)
            raw_narx = np.asarray(latest["narx_wp"], dtype=float)
            trust = float(info.get("narx_trust", 0.0))
            info["raw_narx_position_horizon"] = raw_narx[:, :3]
            info["raw_narx_velocity_horizon"] = raw_narx[:, 3:6]
            info["used_position_horizon"] = (
                trust * raw_narx + (1.0 - trust) * baseline
            )[:, :3]
            info["used_velocity_horizon"] = (
                trust * raw_narx + (1.0 - trust) * baseline
            )[:, 3:6]
        info["estimator_time_s"] = estimator_time_s
        info["controller_total_time_s"] = (time.perf_counter_ns() - start_ns) * 1e-9
        info["target_estimate"] = target_estimate
        return np.asarray(command, dtype=float), info
