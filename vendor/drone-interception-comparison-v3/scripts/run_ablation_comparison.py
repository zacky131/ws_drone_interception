#!/usr/bin/env python
"""
Ablation comparison script — runs every named method once (or N times) and
produces a side-by-side CSV table and comparison figures.

Usage
-----
# Single run per method (fast, qualitative):
python scripts/run_ablation_comparison.py --config configs/default_config.yaml

# Average over N runs per method (more reliable):
python scripts/run_ablation_comparison.py --config configs/default_config.yaml --n-runs 10

# Compare only a subset of methods:
python scripts/run_ablation_comparison.py --methods proposed_full ablation_no_ekf baseline_pn

Outputs (saved to the ``output_dir`` from the config, default: results/ablation/):
    ablation_comparison.csv         — per-run metrics for every method
    ablation_summary.csv            — mean ± std per method (when n-runs > 1)
    {method}_trajectory.csv         — per-timestep log for the last run of each method
    ablation_trajectories.png       — XY trajectory overlay for each method
    ablation_metrics.png            — bar charts: success rate, intercept time,
                                      control effort, estimator RMSE
    ablation_distance.png           — pursuer–target distance over time per method

Also saved to the parent output_dir (default: results/):
    estimator_benchmark.csv         — RMSE summary: EKF vs RLS
    estimator_benchmark_timeseries.csv — per-timestep estimation errors: EKF vs RLS
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from src.utils.config_schema import load_config
from src.environment.wind_model import WindModel
from src.environment.sensor_model import SensorModel
from src.simulation.scenario import TargetScenario
from src.simulation.sim_engine import SimulationEngine
from src.simulation.logger import SimulationLogger
from src.evaluation.metrics import compute_metrics
from run_single_case import (
    _METHOD_ABLATIONS,
    apply_ablation_overrides,
    build_pursuer,
    build_estimator,
    build_controller,
)

# Default method order for plots — proposed first, then ablations, then baselines
_DEFAULT_METHODS = [
    "proposed_full",
    "ablation_no_ekf",
    "ablation_ideal_pursuer",
    "ablation_no_disturbance",
    "ablation_fixed_target_model",
    "baseline_standard_mpc",
    "baseline_pn",
    "baseline_smc",
]

# Short labels for plots
_LABELS = {
    "proposed_full":               "Proposed\n(Full)",
    "ablation_no_ekf":             "No EKF\n(RLS)",
    "ablation_ideal_pursuer":      "Ideal\nPursuer",
    "ablation_no_disturbance":     "No\nDisturbance",
    "ablation_fixed_target_model": "Fixed\nTarget Model",
    "baseline_standard_mpc":       "Std-MPC",
    "baseline_pn":                 "PN",
    "baseline_smc":                "SMC",
    "baseline_rls_adaptive_mpc":   "RLS-AMPC",
}


# ── colours ───────────────────────────────────────────────────────────────────

def _palette(n: int):
    try:
        import matplotlib.pyplot as plt
        return [plt.cm.tab10(i / max(n - 1, 1)) for i in range(n)]
    except ImportError:
        return [(0.2 + 0.7 * i / max(n - 1, 1),) * 3 for i in range(n)]


# ── single run ────────────────────────────────────────────────────────────────

def _run_one(method: str, base_cfg, seed: int):
    """Run one simulation for *method* and return (metrics_dict, logger)."""
    cfg = apply_ablation_overrides(copy.deepcopy(base_cfg), method)
    rng = np.random.default_rng(seed)

    scenario  = TargetScenario(cfg.scenario, cfg.simulation)
    pursuer   = build_pursuer(cfg)
    estimator = build_estimator(cfg)
    controller = build_controller(cfg)
    wind      = WindModel(cfg.wind)
    sensor    = SensorModel(cfg.sensor, rng=rng)
    engine    = SimulationEngine(cfg)
    logger    = SimulationLogger()

    logger = engine.run(scenario, pursuer, estimator, controller, wind, sensor, logger)
    m = compute_metrics(logger)
    m["method"] = method
    m["seed"]   = seed
    return m, logger


# ── comparison table ──────────────────────────────────────────────────────────

def _relative_gain(ref: float, other: float, higher_is_better: bool) -> str:
    if np.isnan(ref) or np.isnan(other) or other == 0:
        return "—"
    gain = (ref - other) / abs(other) * 100 if higher_is_better else (other - ref) / abs(other) * 100
    sign = "+" if gain >= 0 else ""
    return f"{sign}{gain:.1f}%"


def _build_comparison_table(records: List[dict], reference: str = "proposed_full") -> pd.DataFrame:
    df = pd.DataFrame(records)
    # Average over runs
    num_cols = ["success", "intercept_time", "terminal_distance", "min_distance",
                "control_effort", "max_cmd_acc", "control_smoothness",
                "mean_solve_time_s", "solver_feasibility_rate",
                "terminal_speed", "rmse_pos", "rmse_vel", "rmse_acc"]
    num_cols = [c for c in num_cols if c in df.columns]

    summary = df.groupby("method")[num_cols].mean().reset_index()
    summary["success_rate_pct"] = summary["success"] * 100

    # Reference row
    ref_row = summary[summary["method"] == reference]

    rows = []
    for _, row in summary.iterrows():
        is_ref = row["method"] == reference
        ref_sr = ref_row["success_rate_pct"].values[0] if not ref_row.empty else float("nan")
        ref_td = ref_row["terminal_distance"].values[0] if not ref_row.empty else float("nan")
        ref_ce = ref_row["control_effort"].values[0] if not ref_row.empty else float("nan")
        ref_rp = ref_row["rmse_pos"].values[0] if "rmse_pos" in ref_row.columns and not ref_row.empty else float("nan")

        rows.append({
            "method":               row["method"],
            "success_rate_%":       f"{row['success_rate_pct']:.1f}",
            "mean_terminal_dist_m": f"{row['terminal_distance']:.3f}",
            "mean_intercept_s":     f"{row['intercept_time']:.3f}" if not np.isnan(row["intercept_time"]) else "—",
            "control_effort":       f"{row['control_effort']:.1f}",
            "rmse_pos_m":           f"{row['rmse_pos']:.4f}" if "rmse_pos" in row.index and not np.isnan(row["rmse_pos"]) else "—",
            "rmse_vel_m/s":         f"{row['rmse_vel']:.4f}" if "rmse_vel" in row.index and not np.isnan(row["rmse_vel"]) else "—",
            "vs_ref_success":       "REF" if is_ref else _relative_gain(ref_sr, row["success_rate_pct"], True),
            "vs_ref_distance":      "REF" if is_ref else _relative_gain(ref_td, row["terminal_distance"], False),
            "vs_ref_effort":        "REF" if is_ref else _relative_gain(ref_ce, row["control_effort"], False),
        })
    return pd.DataFrame(rows)


# ── figures ───────────────────────────────────────────────────────────────────

def _plot_trajectories(method_logs: Dict[str, SimulationLogger], out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    methods = list(method_logs.keys())
    colours = _palette(len(methods))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for i, method in enumerate(methods):
        df = method_logs[method].to_dataframe()
        if df.empty:
            continue
        c = colours[i]
        label = _LABELS.get(method, method)
        success = method_logs[method].success
        ls = "-" if success else "--"
        marker = "o" if success else "x"

        # XY projection
        axes[0].plot(df["p_px"], df["p_py"], color=c, ls=ls, lw=1.5,
                     label=f"{label} ({'✓' if success else '✗'})")
        axes[0].plot(df["p_px"].iloc[-1], df["p_py"].iloc[-1],
                     marker=marker, color=c, markersize=8)

        # XZ projection
        axes[1].plot(df["p_px"], df["p_pz"], color=c, ls=ls, lw=1.5, label=label)
        axes[1].plot(df["p_px"].iloc[-1], df["p_pz"].iloc[-1],
                     marker=marker, color=c, markersize=8)

    # Target trajectory (from first valid log)
    for log in method_logs.values():
        df = log.to_dataframe()
        if not df.empty:
            axes[0].plot(df["t_px"], df["t_py"], "k--", lw=2, alpha=0.6, label="Target")
            axes[0].plot(df["t_px"].iloc[0], df["t_py"].iloc[0],
                         "kD", markersize=10, label="Target start")
            axes[1].plot(df["t_px"], df["t_pz"], "k--", lw=2, alpha=0.6)
            break

    for ax, xlabel, ylabel, title in [
        (axes[0], "X [m]", "Y [m]", "XY Pursuit Trajectories"),
        (axes[1], "X [m]", "Z [m]", "XZ Pursuit Trajectories"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    plt.suptitle("Ablation Comparison — Pursuer Trajectories", fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_trajectories.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _plot_distance(method_logs: Dict[str, SimulationLogger], success_dist: float,
                   out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    methods = list(method_logs.keys())
    colours = _palette(len(methods))

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, method in enumerate(methods):
        df = method_logs[method].to_dataframe()
        if df.empty:
            continue
        success = method_logs[method].success
        label = _LABELS.get(method, method) + (" ✓" if success else " ✗")
        ax.plot(df["time"], df["distance"], color=colours[i], lw=1.5, label=label)

    ax.axhline(success_dist, color="green", ls="--", lw=1.5, label="Success threshold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Pursuer–Target Distance [m]")
    ax.set_title("Ablation Comparison — Pursuer–Target Distance over Time")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_distance.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _plot_metrics(records: List[dict], out_dir: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    df = pd.DataFrame(records)
    methods = df["method"].unique().tolist()
    colours = _palette(len(methods))
    labels  = [_LABELS.get(m, m).replace("\n", " ") for m in methods]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    def _bar(ax, values, ylabel, title, fmt=".2f", highlight_min=False):
        bars = ax.bar(range(len(methods)), values, color=colours, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        best_idx = int(np.nanargmin(values)) if highlight_min else int(np.nanargmax(values))
        bars[best_idx].set_edgecolor("red")
        bars[best_idx].set_linewidth(2.5)
        for b, v in zip(bars, values):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.01,
                        f"{v:{fmt}}", ha="center", fontsize=8)

    # 1. Success rate
    sr = [df[df["method"] == m]["success"].mean() * 100 for m in methods]
    _bar(axes[0], sr, "Success Rate (%)", "Success Rate", fmt=".1f", highlight_min=False)

    # 2. Mean intercept time (successful only)
    it = [df[(df["method"] == m) & (df["success"] == True)]["intercept_time"].mean()
          for m in methods]
    _bar(axes[1], it, "Intercept Time [s]", "Mean Intercept Time (success only)",
         fmt=".2f", highlight_min=True)

    # 3. Mean terminal distance
    td = [df[df["method"] == m]["terminal_distance"].mean() for m in methods]
    _bar(axes[2], td, "Terminal Distance [m]", "Mean Terminal Miss Distance",
         fmt=".3f", highlight_min=True)

    # 4. Control effort
    ce = [df[df["method"] == m]["control_effort"].mean() for m in methods]
    _bar(axes[3], ce, "Control Effort", "Mean Control Effort", fmt=".1f", highlight_min=True)

    # 5. Estimator RMSE position
    if "rmse_pos" in df.columns:
        rp = [df[df["method"] == m]["rmse_pos"].mean() for m in methods]
        _bar(axes[4], rp, "RMSE pos [m]", "Estimator Position RMSE",
             fmt=".4f", highlight_min=True)
    else:
        axes[4].set_visible(False)

    # 6. Control smoothness
    if "control_smoothness" in df.columns:
        cs = [df[df["method"] == m]["control_smoothness"].mean() for m in methods]
        _bar(axes[5], cs, "Smoothness Σ‖Δu‖²·dt", "Control Smoothness (lower = smoother)",
             fmt=".1f", highlight_min=True)
    else:
        axes[5].set_visible(False)

    plt.suptitle("Ablation Comparison — Performance Metrics\n"
                 "(red border = best value)", fontweight="bold", fontsize=13)
    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_metrics.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _plot_estimation_error(method_logs: Dict[str, SimulationLogger], out_dir: str) -> None:
    """Plot target position estimation error over time per method."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    methods = list(method_logs.keys())
    colours = _palette(len(methods))

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, method in enumerate(methods):
        df = method_logs[method].to_dataframe()
        if df.empty or not all(c in df.columns for c in ["t_px", "te_px"]):
            continue
        err = np.sqrt(
            (df["t_px"] - df["te_px"]) ** 2 +
            (df["t_py"] - df["te_py"]) ** 2 +
            (df["t_pz"] - df["te_pz"]) ** 2
        )
        label = _LABELS.get(method, method).replace("\n", " ")
        ax.plot(df["time"], err, color=colours[i], lw=1.5, label=label)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position Estimation Error [m]")
    ax.set_title("Ablation Comparison — Target Position Estimation Error")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_estimation_error.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── estimator benchmark ──────────────────────────────────────────────────────

def _run_estimator_benchmark(base_cfg, out_parent: str) -> None:
    """Run estimator-only benchmark (EKF vs RLS) and save CSV results.

    Saves two files to *out_parent*:
      estimator_benchmark.csv            — one-row-per-estimator RMSE summary
      estimator_benchmark_timeseries.csv — per-timestep errors for both estimators
    """
    try:
        from src.estimation.ekf_target_estimator import EKFTargetEstimator
        from src.estimation.rls_baseline_estimator import RLSBaselineEstimator
    except ImportError as exc:
        print(f"  [estimator benchmark] import error: {exc}")
        return

    rng      = np.random.default_rng(42)
    scenario = TargetScenario(base_cfg.scenario, base_cfg.simulation)
    sensor   = SensorModel(base_cfg.sensor, rng=rng)

    estimators = {
        "EKF": EKFTargetEstimator(base_cfg.estimator),
        "RLS": RLSBaselineEstimator(base_cfg.estimator),
    }

    dt        = base_cfg.simulation.dt
    max_steps = int(base_cfg.simulation.max_time / dt) + 1

    time_series: dict = {name: {"time": [], "err_pos": [], "err_vel": [], "err_acc": []}
                         for name in estimators}

    pos0, vel0, _ = scenario.get_target_state(0.0)
    z0   = np.concatenate([pos0, vel0])
    meas0 = sensor.process_target(z0, 0.0)
    if meas0 is None:
        meas0 = z0.copy()
    for est in estimators.values():
        est.initialize(meas0)

    for step in range(1, max_steps):
        t = step * dt
        pos_true, vel_true, acc_true = scenario.get_target_state(t)
        z_true = np.concatenate([pos_true, vel_true])
        meas   = sensor.process_target(z_true, t)
        for name, est in estimators.items():
            est.predict(dt)
            if meas is not None:
                est.update(meas)
            x_hat = est.get_estimate()
            time_series[name]["time"].append(t)
            time_series[name]["err_pos"].append(float(np.linalg.norm(pos_true - x_hat[0:3])))
            time_series[name]["err_vel"].append(float(np.linalg.norm(vel_true - x_hat[3:6])))
            time_series[name]["err_acc"].append(float(np.linalg.norm(acc_true - x_hat[6:9])))

    os.makedirs(out_parent, exist_ok=True)

    # Summary CSV
    summary_rows = [
        {
            "estimator": name,
            "rmse_pos": float(np.sqrt(np.mean(np.array(time_series[name]["err_pos"]) ** 2))),
            "rmse_vel": float(np.sqrt(np.mean(np.array(time_series[name]["err_vel"]) ** 2))),
            "rmse_acc": float(np.sqrt(np.mean(np.array(time_series[name]["err_acc"]) ** 2))),
        }
        for name in estimators
    ]
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(out_parent, "estimator_benchmark.csv"), index=False
    )
    print(f"  Saved: {os.path.join(out_parent, 'estimator_benchmark.csv')}")

    # Time-series CSV
    ts_rows = [
        {"estimator": name, "time": t,
         "err_pos": time_series[name]["err_pos"][i],
         "err_vel": time_series[name]["err_vel"][i],
         "err_acc": time_series[name]["err_acc"][i]}
        for name in estimators
        for i, t in enumerate(time_series[name]["time"])
    ]
    pd.DataFrame(ts_rows).to_csv(
        os.path.join(out_parent, "estimator_benchmark_timeseries.csv"), index=False
    )
    print(f"  Saved: {os.path.join(out_parent, 'estimator_benchmark_timeseries.csv')}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all ablation methods and produce comparison tables and figures"
    )
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(_PROJECT_ROOT, "configs", "default_config.yaml"),
        help="Base YAML config file",
    )
    parser.add_argument(
        "--methods", nargs="+", default=None,
        help="Subset of method names to compare (default: all standard ablations). "
             "Available: " + " ".join(sorted(_METHOD_ABLATIONS)),
    )
    parser.add_argument(
        "--n-runs", type=int, default=1,
        help="Number of independent runs per method (averaged in the summary)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (incremented per run)",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory (default: <config output_dir>/ablation/)",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)

    methods = args.methods if args.methods is not None else _DEFAULT_METHODS
    # Filter to only those registered
    unknown = [m for m in methods if m not in _METHOD_ABLATIONS]
    if unknown:
        parser.error(f"Unknown methods: {unknown}\nAvailable: {sorted(_METHOD_ABLATIONS)}")

    out_dir = args.out_dir or os.path.join(base_cfg.output.output_dir, "ablation")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("  ABLATION COMPARISON")
    print("=" * 72)
    print(f"  Config   : {args.config}")
    print(f"  Methods  : {methods}")
    print(f"  Runs/method: {args.n_runs}")
    print(f"  Scenario : {base_cfg.scenario.scenario_type}")
    print(f"  Out dir  : {out_dir}")
    print("=" * 72)

    all_records: List[dict] = []
    # Keep only the LAST run's logger per method for trajectory plots
    last_logs: Dict[str, SimulationLogger] = {}

    for method in methods:
        for run_i in range(args.n_runs):
            seed = args.seed + run_i
            try:
                m, logger = _run_one(method, base_cfg, seed)
                all_records.append(m)
                last_logs[method] = logger
                status_char = "✓" if m["success"] else "✗"
                print(f"  [{status_char}] {method:40s}  run {run_i + 1}/{args.n_runs}"
                      f"  dist={m['terminal_distance']:.3f} m"
                      f"  t={m['intercept_time']:.2f} s")
            except Exception as exc:
                print(f"  [!] {method} run {run_i + 1} FAILED: {exc}")

    if not all_records:
        print("No results to save.")
        return

    # ── Save per-run CSV ──────────────────────────────────────────────────
    detail_df = pd.DataFrame(all_records)
    detail_path = os.path.join(out_dir, "ablation_comparison.csv")
    detail_df.to_csv(detail_path, index=False)
    print(f"\n  Per-run CSV  → {detail_path}")

    # ── Comparison summary table ──────────────────────────────────────────
    ref = "proposed_full" if "proposed_full" in methods else methods[0]
    comparison = _build_comparison_table(all_records, reference=ref)
    summary_path = os.path.join(out_dir, "ablation_summary.csv")
    comparison.to_csv(summary_path, index=False)
    print(f"  Summary CSV  → {summary_path}")

    print("\n" + comparison.to_string(index=False))

    # ── Figures ───────────────────────────────────────────────────────────
    print("\n  Generating figures...")
    _plot_trajectories(last_logs, out_dir)
    _plot_distance(last_logs, base_cfg.simulation.success_distance, out_dir)
    _plot_metrics(all_records, out_dir)
    _plot_estimation_error(last_logs, out_dir)

    # ── Save per-method trajectory CSVs ──────────────────────────────────
    print("\n  Saving per-method trajectory CSVs...")
    for method, log in last_logs.items():
        traj_df = log.to_dataframe()
        traj_df.insert(0, "method", method)
        traj_path = os.path.join(out_dir, f"{method}_trajectory.csv")
        traj_df.to_csv(traj_path, index=False)
        print(f"  Saved: {traj_path}")

    # ── Estimator benchmark ───────────────────────────────────────────────
    parent_dir = base_cfg.output.output_dir
    print("\n  Running estimator benchmark...")
    _run_estimator_benchmark(base_cfg, parent_dir)

    print("\n" + "=" * 72)
    print(f"  All outputs saved to: {out_dir}/")
    print("=" * 72)
    print("Done.")


if __name__ == "__main__":
    main()
