#!/usr/bin/env python
"""
Estimate-only benchmark: run target trajectories through the EKF and RLS
estimators without any pursuer dynamics and compare estimation accuracy.

Usage:
    python scripts/benchmark_estimators.py --config configs/default_config.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from src.utils.config_schema import load_config
from src.estimation.ekf_target_estimator import EKFTargetEstimator
from src.estimation.rls_baseline_estimator import RLSBaselineEstimator
from src.environment.sensor_model import SensorModel
from src.simulation.scenario import TargetScenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimator-only benchmark")
    parser.add_argument("--config", type=str,
                        default=os.path.join(_PROJECT_ROOT, "configs", "default_config.yaml"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = np.random.default_rng(args.seed)

    scenario = TargetScenario(cfg.scenario, cfg.simulation)
    sensor = SensorModel(cfg.sensor, rng=rng)

    estimators = {
        "EKF": EKFTargetEstimator(cfg.estimator),
        "RLS": RLSBaselineEstimator(cfg.estimator),
    }

    dt = cfg.simulation.dt
    max_steps = int(cfg.simulation.max_time / dt) + 1

    results = {name: {"err_pos": [], "err_vel": [], "err_acc": []} for name in estimators}

    # Initialize estimators with first measurement
    pos0, vel0, _ = scenario.get_target_state(0.0)
    z0 = np.concatenate([pos0, vel0])
    meas0 = sensor.process_target(z0, 0.0)
    if meas0 is None:
        meas0 = z0.copy()
    for est in estimators.values():
        est.initialize(meas0)

    for step in range(1, max_steps):
        t = step * dt
        pos_true, vel_true, acc_true = scenario.get_target_state(t)
        z_true = np.concatenate([pos_true, vel_true])
        meas = sensor.process_target(z_true, t)

        for name, est in estimators.items():
            est.predict(dt)
            if meas is not None:
                est.update(meas)
            x_hat = est.get_estimate()

            err_p = np.linalg.norm(pos_true - x_hat[0:3])
            err_v = np.linalg.norm(vel_true - x_hat[3:6])
            err_a = np.linalg.norm(acc_true - x_hat[6:9])

            results[name]["err_pos"].append(err_p)
            results[name]["err_vel"].append(err_v)
            results[name]["err_acc"].append(err_a)

    # ── Print summary ─────────────────────────────────────────────────────
    print("=" * 60)
    print("  ESTIMATOR BENCHMARK")
    print("=" * 60)
    print(f"  Scenario: {cfg.scenario.scenario_type}")
    print(f"  Duration: {cfg.simulation.max_time} s  |  Steps: {max_steps}")
    print(f"  Sensor noise  pos σ={cfg.sensor.position_noise_std} m  "
          f"vel σ={cfg.sensor.velocity_noise_std} m/s")
    print(f"  Delay: {cfg.sensor.delay_steps} steps  "
          f"Dropout: {cfg.sensor.dropout_probability}")
    print("-" * 60)

    rows = []
    for name in estimators:
        rmse_p = np.sqrt(np.mean(np.array(results[name]["err_pos"]) ** 2))
        rmse_v = np.sqrt(np.mean(np.array(results[name]["err_vel"]) ** 2))
        rmse_a = np.sqrt(np.mean(np.array(results[name]["err_acc"]) ** 2))
        print(f"  {name:6s}  RMSE pos={rmse_p:.4f} m   vel={rmse_v:.4f} m/s   acc={rmse_a:.4f} m/s²")
        rows.append({"estimator": name, "rmse_pos": rmse_p, "rmse_vel": rmse_v, "rmse_acc": rmse_a})

    print("=" * 60)

    out_dir = cfg.output.output_dir
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "estimator_benchmark.csv")
    df.to_csv(path, index=False)
    print(f"Results saved to {path}")

    # ── Plot ──────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        t_arr = np.arange(1, max_steps) * dt
        labels = {"err_pos": "Position [m]", "err_vel": "Velocity [m/s]", "err_acc": "Acceleration [m/s²]"}
        for ax, key in zip(axes, ["err_pos", "err_vel", "err_acc"]):
            for name in estimators:
                ax.plot(t_arr, results[name][key], label=name, alpha=0.8)
            ax.set_xlabel("Time [s]")
            ax.set_ylabel(labels[key])
            ax.set_title(f"Estimation Error – {labels[key].split('[')[0].strip()}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(out_dir, "estimator_benchmark.png")
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {fig_path}")
    except ImportError:
        pass

    print("Done.")


if __name__ == "__main__":
    main()
