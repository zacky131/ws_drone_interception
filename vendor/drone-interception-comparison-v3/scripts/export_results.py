#!/usr/bin/env python
"""
Export tables and plots from Monte Carlo results.

Reads the ``monte_carlo_detailed.csv`` produced by ``run_monte_carlo.py`` and
generates publication-quality summary tables and figures.

Usage:
    python scripts/export_results.py --results-dir monte_carlo_results/
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export tables and plots from MC results")
    parser.add_argument("--results-dir", type=str, default="monte_carlo_results")
    args = parser.parse_args()

    detail_path = os.path.join(args.results_dir, "monte_carlo_detailed.csv")
    if not os.path.isfile(detail_path):
        print(f"File not found: {detail_path}")
        print("Run 'scripts/run_monte_carlo.py' first.")
        sys.exit(1)

    df = pd.read_csv(detail_path)
    out = args.results_dir

    # ── Per-algorithm summary table ───────────────────────────────────────
    _RT_BUDGET_MS = 20.0   # 20 ms realtime budget (50 Hz control loop, dt=0.02 s)
    rows = []
    for algo in df["algorithm"].unique():
        sub = df[df["algorithm"] == algo]
        n = len(sub)
        ns = sub["success"].sum()
        sr = ns / n * 100 if n > 0 else 0

        # ── Computation time ──────────────────────────────────────────────
        if "mean_solve_time_s" in sub.columns and sub["mean_solve_time_s"].notna().any():
            mean_solve_ms = f"{sub['mean_solve_time_s'].mean() * 1000:.2f}"
        else:
            mean_solve_ms = "—"

        if "max_solve_time_s" in sub.columns and sub["max_solve_time_s"].notna().any():
            p95_max_ms     = f"{sub['max_solve_time_s'].quantile(0.95) * 1000:.2f}"
            worst_max_ms   = f"{sub['max_solve_time_s'].max() * 1000:.2f}"
            rt_feasible    = f"{(sub['max_solve_time_s'] <= _RT_BUDGET_MS / 1000).mean() * 100:.1f}"
        else:
            p95_max_ms = worst_max_ms = rt_feasible = "—"

        if "solver_feasibility_rate" in sub.columns and sub["solver_feasibility_rate"].notna().any():
            solver_feas = f"{sub['solver_feasibility_rate'].mean() * 100:.1f}"
        else:
            solver_feas = "—"

        rows.append({
            "Algorithm": algo,
            "Trials": n,
            "Success": int(ns),
            "Rate (%)": f"{sr:.1f}",
            "Mean Intercept (s)": f"{sub.loc[sub['success'] == True, 'intercept_time'].mean():.3f}" if ns > 0 else "—",
            "Mean Effort": f"{sub['control_effort'].mean():.1f}",
            "RMSE pos (m)": f"{sub['rmse_pos'].mean():.3f}",
            "RMSE vel (m/s)": f"{sub['rmse_vel'].mean():.3f}",
            "Mean Solve (ms)": mean_solve_ms,
            "P95 Max Solve (ms)": p95_max_ms,
            "Worst Max Solve (ms)": worst_max_ms,
            f"RT Feasible (≤{_RT_BUDGET_MS:.0f}ms) (%)": rt_feasible,
            "Solver Feasibility (%)": solver_feas,
        })

    summary = pd.DataFrame(rows)
    table_path = os.path.join(out, "summary_table.csv")
    summary.to_csv(table_path, index=False)
    print("Summary table:")
    print(summary.to_string(index=False))
    print(f"\n→ {table_path}")

    # ── LaTeX table ───────────────────────────────────────────────────────
    latex_path = os.path.join(out, "summary_table.tex")
    with open(latex_path, "w") as fh:
        fh.write(summary.to_latex(index=False, escape=True))
    print(f"LaTeX table → {latex_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        algos = df["algorithm"].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(algos)))

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Success rate bar
        ax = axes[0]
        rates = [df[df["algorithm"] == a]["success"].mean() * 100 for a in algos]
        bars = ax.bar(range(len(algos)), rates, color=colors)
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(algos, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("Success Rate")
        ax.grid(axis="y", alpha=0.3)
        for b, r in zip(bars, rates):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                    f"{r:.1f}%", ha="center", fontsize=9, fontweight="bold")

        # Intercept time violin
        ax = axes[1]
        data = [df[(df["algorithm"] == a) & (df["success"] == True)]["intercept_time"].dropna().values
                for a in algos]
        parts = ax.violinplot([d for d in data if len(d) > 0],
                              positions=[i for i, d in enumerate(data) if len(d) > 0],
                              showmeans=True, showmedians=True)
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(algos, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Intercept Time (s)")
        ax.set_title("Intercept Time (successes)")
        ax.grid(axis="y", alpha=0.3)

        # Control effort box
        ax = axes[2]
        bp_data = [df[df["algorithm"] == a]["control_effort"].values for a in algos]
        ax.boxplot(bp_data, labels=algos, patch_artist=True,
                   boxprops=dict(alpha=0.6))
        ax.set_xticklabels(algos, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Control Effort")
        ax.set_title("Control Effort Distribution")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(out, "monte_carlo_export.png")
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Figure → {fig_path}")

        # ── Additional publication-quality figures ────────────────────────
        _plot_cdf(df, "intercept_time", "Intercept Time [s]",
                  "CDF of Intercept Time", "fig_cdf_intercept_time.png", out, success_only=True)
        _plot_cdf(df, "terminal_distance", "Terminal Miss Distance [m]",
                  "CDF of Terminal Miss Distance", "fig_cdf_terminal_distance.png", out)
        _plot_estimator_rmse(df, out)
        _plot_robustness_metrics(df, out)
        _plot_effort_vs_time_scatter(df, out)
        _plot_success_vs_range(df, out)
        _plot_failure_modes(df, out)
        _plot_radar(df, out)
        _plot_compute_time(df, out)

    except ImportError:
        print("matplotlib not available; skipping plots.")

    # ── Ablation delta plot (from ablation_summary.csv if present) ───────
    ablation_path = os.path.join(out, "ablation_summary.csv")
    if os.path.isfile(ablation_path):
        _plot_ablation_deltas(ablation_path, out)
    else:
        print("\nNo ablation_summary.csv found — skipping ablation delta plot.")
        print("  Run with multiple ablation algorithms to generate it.")

    print("Done.")


def _plot_cdf(df: pd.DataFrame, col: str, xlabel: str, title: str, fname: str,
              out_dir: str, success_only: bool = False) -> None:
    """Empirical CDF per algorithm — standard in aerospace/GNC literature."""
    import matplotlib.pyplot as plt

    algos = df["algorithm"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(algos)))
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, algo in enumerate(algos):
        sub = df[df["algorithm"] == algo]
        if success_only:
            sub = sub[sub["success"] == True]
        vals = sub[col].dropna().sort_values().values
        if len(vals) == 0:
            continue
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.step(vals, cdf, where="post", color=colors[i], linewidth=2, label=algo)
        ax.plot(vals[-1], cdf[-1], "o", color=colors[i], markersize=5)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Cumulative Probability", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(title + (" (successful trials only)" if success_only else ""),
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure → {path}")


def _plot_estimator_rmse(df: pd.DataFrame, out_dir: str) -> None:
    """Grouped bar chart of estimator RMSE components per algorithm."""
    import matplotlib.pyplot as plt

    rmse_cols = [c for c in ["rmse_pos", "rmse_vel", "rmse_acc", "rmse_jerk"] if c in df.columns]
    if not rmse_cols:
        return

    labels_map = {"rmse_pos": "Position (m)", "rmse_vel": "Velocity (m/s)",
                  "rmse_acc": "Acceleration (m/s²)", "rmse_jerk": "Jerk (m/s³)"}
    algos = df["algorithm"].unique().tolist()
    n_algos = len(algos)
    n_cols = len(rmse_cols)
    x = np.arange(n_algos)
    width = 0.8 / n_cols
    colors = plt.cm.Set1(np.linspace(0, 1, n_cols))

    fig, ax = plt.subplots(figsize=(max(10, n_algos * 1.3), 5))
    for j, col in enumerate(rmse_cols):
        means = [df[df["algorithm"] == a][col].mean() for a in algos]
        stds  = [df[df["algorithm"] == a][col].std()  for a in algos]
        offsets = x + (j - n_cols / 2 + 0.5) * width
        ax.bar(offsets, means, width=width * 0.9, yerr=stds, capsize=4,
               color=colors[j], edgecolor="black", linewidth=0.5,
               label=labels_map.get(col, col), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("RMSE (mean ± std across trials)", fontsize=11)
    ax.set_title("Target State Estimation RMSE by Component and Algorithm",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_estimator_rmse.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure → {path}")


def _plot_robustness_metrics(df: pd.DataFrame, out_dir: str) -> None:
    """Solver feasibility rate and control saturation rate — proves MPC robustness."""
    import matplotlib.pyplot as plt

    cols = [c for c in ["solver_feasibility_rate", "saturation_rate", "control_smoothness"]
            if c in df.columns]
    if not cols:
        return

    titles_map = {
        "solver_feasibility_rate": "Solver Feasibility Rate",
        "saturation_rate":         "Actuator Saturation Rate",
        "control_smoothness":      "Control Smoothness (lower = smoother)",
    }
    algos = df["algorithm"].unique().tolist()
    colors = plt.cm.Set2(np.linspace(0, 1, len(algos)))

    fig, axes = plt.subplots(1, len(cols), figsize=(len(cols) * 5, 5))
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        means = [df[df["algorithm"] == a][col].mean() for a in algos]
        stds  = [df[df["algorithm"] == a][col].std()  for a in algos]
        bars = ax.bar(range(len(algos)), means, yerr=stds, capsize=5,
                      color=colors, edgecolor="black", linewidth=0.7, alpha=0.85)
        ax.set_xticks(range(len(algos)))
        ax.set_xticklabels(algos, rotation=35, ha="right", fontsize=9)
        ax.set_title(titles_map.get(col, col), fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                    f"{v:.3f}", ha="center", fontsize=8)

    plt.suptitle("MPC Robustness Indicators", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_robustness_metrics.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure → {path}")


def _plot_effort_vs_time_scatter(df: pd.DataFrame, out_dir: str) -> None:
    """Control effort vs. intercept time Pareto scatter — tradeoff visualization."""
    import matplotlib.pyplot as plt

    if "intercept_time" not in df.columns or "control_effort" not in df.columns:
        return

    algos = df["algorithm"].unique().tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(algos)))
    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, algo in enumerate(algos):
        sub = df[(df["algorithm"] == algo) & (df["success"] == True)]
        if sub.empty:
            continue
        ax.scatter(sub["intercept_time"], sub["control_effort"],
                   color=colors[i], marker=markers[i % len(markers)],
                   s=60, alpha=0.7, label=algo, edgecolors="black", linewidths=0.4)
        # Mean marker
        ax.scatter(sub["intercept_time"].mean(), sub["control_effort"].mean(),
                   color=colors[i], marker=markers[i % len(markers)],
                   s=200, edgecolors="black", linewidths=1.5, zorder=5)

    ax.set_xlabel("Intercept Time [s]", fontsize=11)
    ax.set_ylabel("Control Effort  Σ‖u‖²·dt", fontsize=11)
    ax.set_title("Control Effort vs. Intercept Time Tradeoff\n"
                 "(successful trials; large marker = mean)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_effort_vs_time.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure → {path}")


def _plot_success_vs_range(df: pd.DataFrame, out_dir: str, n_bins: int = 5) -> None:
    """Success rate vs. initial pursuer–target separation — shows engagement envelope."""
    import matplotlib.pyplot as plt

    if not all(c in df.columns for c in ["p0_x", "p0_y", "p0_z"]):
        return

    # Compute initial separation (assuming target starts at scenario origin;
    # use magnitude of pursuer initial position as proxy for range)
    df = df.copy()
    df["_range"] = np.sqrt(df["p0_x"] ** 2 + df["p0_y"] ** 2 + df["p0_z"] ** 2)

    bin_edges = np.linspace(df["_range"].min(), df["_range"].max(), n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    algos = df["algorithm"].unique().tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(algos)))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, algo in enumerate(algos):
        sub = df[df["algorithm"] == algo]
        srs, n_trials = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            bucket = sub[(sub["_range"] >= lo) & (sub["_range"] < hi)]
            n_trials.append(len(bucket))
            srs.append(bucket["success"].mean() * 100 if len(bucket) > 0 else np.nan)
        ax.plot(bin_centers, srs, "o-", color=colors[i], linewidth=2,
                markersize=7, label=algo, alpha=0.85)

    ax.set_xlabel("Initial Pursuer–Target Range [m]", fontsize=11)
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_ylim(-5, 105)
    ax.set_title("Engagement Envelope: Success Rate vs. Initial Separation Range",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_success_vs_range.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure → {path}")


def _plot_failure_modes(df: pd.DataFrame, out_dir: str) -> None:
    """Stacked bar of failure categories per algorithm."""
    import matplotlib.pyplot as plt

    if "failure_category" not in df.columns:
        return

    algos = df["algorithm"].unique().tolist()
    categories = sorted(df["failure_category"].dropna().unique().tolist())
    cat_colors = dict(zip(categories, plt.cm.Set3(np.linspace(0, 1, len(categories)))))

    fig, ax = plt.subplots(figsize=(max(10, len(algos) * 1.3), 5))
    bottoms = np.zeros(len(algos))
    x = np.arange(len(algos))

    for cat in categories:
        counts = []
        for algo in algos:
            sub = df[df["algorithm"] == algo]
            n = len(sub)
            counts.append(sub[sub["failure_category"] == cat].shape[0] / n * 100 if n > 0 else 0.0)
        ax.bar(x, counts, bottom=bottoms, label=cat,
               color=cat_colors[cat], edgecolor="white", linewidth=0.5)
        bottoms += np.array(counts)

    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Percentage of Trials (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title("Trial Outcome Distribution by Algorithm", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, title="Outcome")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_failure_modes.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure → {path}")


def _plot_radar(df: pd.DataFrame, out_dir: str) -> None:
    """Radar/spider chart — normalised multi-metric comparison for journal figures."""
    import matplotlib.pyplot as plt

    metrics_cfg = [
        # (column, label, higher_is_better)
        ("success",                "Success\nRate",    True),
        ("terminal_distance",      "Miss\nDist.",      False),
        ("control_effort",         "Control\nEffort",  False),
        ("control_smoothness",     "Smoothness",       False),
        ("solver_feasibility_rate","Solver\nFeas.",    True),
        ("rmse_pos",               "RMSE\nPos",        False),
        ("rmse_vel",               "RMSE\nVel",        False),
    ]
    metrics_cfg = [(c, l, h) for c, l, h in metrics_cfg if c in df.columns]
    if len(metrics_cfg) < 3:
        return

    algos = df["algorithm"].unique().tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(algos)))
    N = len(metrics_cfg)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    # Compute per-algorithm means, then normalise 0–1 (1 = best)
    raw = {}
    for algo in algos:
        sub = df[df["algorithm"] == algo]
        raw[algo] = [sub[col].mean() for col, _, _ in metrics_cfg]

    norm = {}
    for j, (col, _, higher_is_better) in enumerate(metrics_cfg):
        col_vals = np.array([raw[a][j] for a in algos])
        lo, hi = np.nanmin(col_vals), np.nanmax(col_vals)
        for algo in algos:
            v = raw[algo][j]
            if hi == lo:
                n = 0.5
            else:
                n = (v - lo) / (hi - lo)
                if not higher_is_better:
                    n = 1.0 - n
            norm.setdefault(algo, []).append(n)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), [l for _, l, _ in metrics_cfg], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7, alpha=0.6)
    ax.grid(True, alpha=0.3)

    for i, algo in enumerate(algos):
        values = norm[algo] + norm[algo][:1]
        ax.plot(angles, values, "o-", linewidth=2, color=colors[i], label=algo)
        ax.fill(angles, values, alpha=0.08, color=colors[i])

    ax.set_title("Multi-Metric Normalised Performance Radar\n(outer = better)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_radar.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure → {path}")


def _plot_ablation_deltas(ablation_path: str, out_dir: str) -> None:
    """Grouped bar chart showing relative gain of proposed_full vs each ablation."""
    import matplotlib.pyplot as plt

    ab = pd.read_csv(ablation_path)
    if ab.empty:
        return

    # Normalise column name: run_monte_carlo writes "algorithm", older versions wrote "method"
    if "algorithm" in ab.columns and "method" not in ab.columns:
        ab = ab.rename(columns={"algorithm": "method"})

    # Expected columns from run_monte_carlo._ablation_summary():
    #   method, relative_success_gain_pct, relative_distance_reduction_pct,
    #   relative_time_reduction_pct, relative_effort_reduction_pct,
    #   relative_rmse_pos_reduction_pct
    delta_cols = {
        "relative_success_gain_pct":        "Success Rate\nGain (%)",
        "relative_distance_reduction_pct":  "Terminal Dist.\nReduction (%)",
        "relative_time_reduction_pct":      "Intercept Time\nReduction (%)",
        "relative_effort_reduction_pct":    "Control Effort\nReduction (%)",
        "relative_rmse_pos_reduction_pct":  "RMSE-pos\nReduction (%)",
    }
    available = {k: v for k, v in delta_cols.items() if k in ab.columns}
    if not available:
        print("  ablation_summary.csv lacks expected delta columns; skipping delta plot.")
        return

    methods = ab["method"].tolist()
    n_methods = len(methods)
    n_metrics = len(available)
    colors = plt.cm.Set3(np.linspace(0, 1, n_metrics))

    fig, ax = plt.subplots(figsize=(max(10, n_methods * 1.5), 6))
    x = np.arange(n_methods)
    width = 0.8 / n_metrics

    for i, (col, label) in enumerate(available.items()):
        vals = ab[col].values
        offsets = x + (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(offsets, vals, width=width * 0.9, color=colors[i],
                      edgecolor="black", linewidth=0.5, label=label)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2,
                        b.get_height() + (0.5 if v >= 0 else -2.5),
                        f"{v:.1f}%", ha="center", fontsize=7)

    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Relative Gain of Proposed-Full (%, positive = proposed better)")
    ax.set_title("Ablation Study — Relative Improvement of Proposed Method\n"
                 "vs Each Ablation/Baseline (from Monte Carlo)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_delta.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Ablation delta plot → {path}")


# ── Task 6: New export functions ──────────────────────────────────────────────

def export_cdf_plot(
    results_df: pd.DataFrame,
    output_path: str,
    rho_th: float = 0.5,
    dist_col: str = "terminal_distance",
) -> None:
    """Export CDF of terminal miss distance — one curve per guidance method.

    A vertical reference line is drawn at *rho_th* (the success threshold)
    so readers can read off the success probability directly.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain an ``algorithm`` (or ``method``) column and
        *dist_col* (float terminal miss distance in metres).
    output_path : str
        Path prefix for output files (without extension).  Both a ``.pdf``
        and a ``.png`` are written.
    rho_th : float
        Success threshold [m] (vertical reference line).  Default 0.5 m.
    dist_col : str
        Name of the miss-distance column.  Default ``"terminal_distance"``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.evaluation.metrics import compute_miss_distance_cdf

    df = results_df.copy()
    # Normalise method column
    if "algorithm" in df.columns and "method" not in df.columns:
        df["method"] = df["algorithm"]
    elif "method" not in df.columns:
        raise KeyError("DataFrame must have an 'algorithm' or 'method' column.")

    if dist_col not in df.columns:
        raise KeyError(f"DataFrame must have a '{dist_col}' column.")

    methods = df["method"].unique().tolist()
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    thresholds = np.linspace(0.0, df[dist_col].quantile(0.99) * 1.05, 500)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, method in enumerate(methods):
        sub = df[df["method"] == method]
        distances = sub[dist_col].dropna().values
        if len(distances) == 0:
            continue
        cdf_vals = compute_miss_distance_cdf(distances, thresholds)
        ax.plot(thresholds, cdf_vals * 100, linewidth=2,
                color=colors[i], label=method)

    ax.axvline(x=rho_th, color="black", linestyle="--", linewidth=1.5,
               label=f"Success threshold ρ={rho_th} m")
    ax.set_xlabel("Terminal Miss Distance [m]", fontsize=12)
    ax.set_ylabel("Cumulative Probability (%)", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title("CDF of Terminal Miss Distance", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    for ext in (".pdf", ".png"):
        path = output_path + ext
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  CDF plot → {path}")
    plt.close()


def export_robustness_curves(
    robustness_csv_dir: str,
    output_path: str,
    success_threshold_m: float = 0.5,
) -> None:
    """Export robustness sweep success-rate curves — one subplot per parameter.

    Reads ``<robustness_csv_dir>/<param>_sweep.csv`` for every CSV present.
    Each subplot shows success rate (%) vs. parameter value.

    Parameters
    ----------
    robustness_csv_dir : str
        Directory containing ``*_sweep.csv`` files produced by
        ``run_robustness_sweep.py``.
    output_path : str
        Path prefix (no extension).  Both ``.pdf`` and ``.png`` are written.
    success_threshold_m : float
        Success distance threshold used when computing success rate from
        a ``terminal_distance`` column.  Ignored if a ``success`` column
        is present.  Default 0.5 m.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.evaluation.metrics import compute_success_rate_vs_parameter

    # Discover sweep CSVs
    sweep_csvs = sorted([
        f for f in os.listdir(robustness_csv_dir)
        if f.endswith("_sweep.csv")
    ])
    if not sweep_csvs:
        raise FileNotFoundError(
            f"No *_sweep.csv files found in '{robustness_csv_dir}'."
        )

    n_plots = len(sweep_csvs)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5),
                             squeeze=False)

    PARAM_LABELS: Dict[str, str] = {
        "wind_magnitude":      "Wind Magnitude [m/s]",
        "sensor_noise_std":    "Sensor Position Noise σ [m]",
        "sensor_delay_steps":  "Sensor Delay [timesteps]",
        "target_scenario":     "Target Scenario",
    }

    for col_idx, fname in enumerate(sweep_csvs):
        param = fname.replace("_sweep.csv", "")
        df = pd.read_csv(os.path.join(robustness_csv_dir, fname))
        ax = axes[0][col_idx]

        if "sweep_value" not in df.columns:
            ax.set_title(param)
            ax.text(0.5, 0.5, "No sweep_value column",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        try:
            summary = compute_success_rate_vs_parameter(
                df, "sweep_value", threshold=success_threshold_m
            )
        except Exception as exc:
            ax.set_title(param)
            ax.text(0.5, 0.5, str(exc), ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            continue

        x = list(range(len(summary)))
        vals = summary["success_rate_pct"].values
        lo = summary["sr_ci_lower"].values
        hi = summary["sr_ci_upper"].values
        labels = summary["sweep_value"].astype(str).tolist()

        # Numeric parameters → line+fill; categorical → bar
        is_numeric = pd.to_numeric(summary["sweep_value"], errors="coerce").notna().all()
        if is_numeric:
            xn = pd.to_numeric(summary["sweep_value"]).values
            ax.plot(xn, vals, "o-", linewidth=2, markersize=7, color="steelblue")
            ax.fill_between(xn, lo, hi, alpha=0.2, color="steelblue",
                            label="95% CI")
            ax.set_xlabel(PARAM_LABELS.get(param, param), fontsize=12)
        else:
            bars = ax.bar(x, vals, color="steelblue", edgecolor="black",
                          linewidth=0.7, alpha=0.85)
            ax.errorbar(x,
                        vals,
                        yerr=[vals - lo, hi - vals],
                        fmt="none", color="black", capsize=5, linewidth=1.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=11)
            ax.set_xlabel(PARAM_LABELS.get(param, param), fontsize=12)

        ax.set_ylabel("Success Rate (%)", fontsize=12)
        ax.set_ylim(-3, 103)
        ax.set_title(PARAM_LABELS.get(param, param), fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if is_numeric:
            ax.legend(fontsize=10)

    fig.suptitle("Robustness Sweep: Success Rate vs. Swept Parameter",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    for ext in (".pdf", ".png"):
        path = output_path + ext
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Robustness curves → {path}")
    plt.close()


def export_ablation_bar_chart(
    results_df: pd.DataFrame,
    output_path: str,
    highlight_method: str = "proposed_full",
    success_threshold_m: float = 0.5,
) -> None:
    """Export a horizontal bar chart of success rate for all ablation variants.

    The *highlight_method* bar is rendered in a distinct accent colour so it
    stands out as the reference.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain an ``algorithm`` (or ``method``) column and either a
        boolean ``success`` column or a ``terminal_distance`` column.
    output_path : str
        Path prefix (no extension).  Both ``.pdf`` and ``.png`` are written.
    highlight_method : str
        Method name to render in accent colour.  Default ``"proposed_full"``.
    success_threshold_m : float
        Used to binarise ``terminal_distance`` when ``success`` is absent.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _wilson_ci(k: int, n: int):
        """Wilson score 95 % CI; returns (lo, hi) in [0, 1]."""
        if n == 0:
            return (0.0, 0.0)
        p_hat = k / n
        z = 1.959964
        denom = 1.0 + z ** 2 / n
        centre = (p_hat + z ** 2 / (2 * n)) / denom
        half = z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
        return (max(0.0, centre - half), min(1.0, centre + half))

    df = results_df.copy()
    if "algorithm" in df.columns and "method" not in df.columns:
        df["method"] = df["algorithm"]
    elif "method" not in df.columns:
        raise KeyError("DataFrame must have an 'algorithm' or 'method' column.")

    if "success" not in df.columns:
        if "terminal_distance" not in df.columns:
            raise KeyError("DataFrame must have 'success' or 'terminal_distance'.")
        df["success"] = df["terminal_distance"] <= success_threshold_m

    methods = df["method"].unique().tolist()
    rows = []
    for m in methods:
        sub = df[df["method"] == m]
        n = len(sub)
        k = int(sub["success"].sum())
        sr = k / n * 100.0 if n > 0 else float("nan")
        ci = _wilson_ci(k, n)
        rows.append({"method": m, "sr": sr,
                     "ci_lo": ci[0] * 100, "ci_hi": ci[1] * 100})
    rows.sort(key=lambda r: r["sr"])
    sorted_methods = [r["method"] for r in rows]
    success_rates = [r["sr"] for r in rows]
    xerr_lo = [r["sr"] - r["ci_lo"] for r in rows]
    xerr_hi = [r["ci_hi"] - r["sr"] for r in rows]

    accent = "#E05C2E"   # orange-red for the highlighted method
    default_color = "#4C72B0"

    colors = [
        accent if m == highlight_method else default_color
        for m in sorted_methods
    ]

    fig, ax = plt.subplots(figsize=(9, max(4, len(methods) * 0.55 + 1)))
    y = np.arange(len(sorted_methods))
    bars = ax.barh(y, success_rates, color=colors, edgecolor="white",
                   linewidth=0.7, height=0.65)
    ax.errorbar(success_rates, y,
                xerr=[xerr_lo, xerr_hi],
                fmt="none", color="black", capsize=4, linewidth=1.5)

    for bar, sr in zip(bars, success_rates):
        ax.text(sr + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{sr:.1f}%", va="center", ha="left", fontsize=11, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_methods, fontsize=11)
    ax.set_xlabel("Success Rate (%)", fontsize=12)
    ax.set_xlim(0, 110)
    ax.set_title("Ablation Study: Success Rate by Variant\n"
                 f"(orange = {highlight_method})", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    for ext in (".pdf", ".png"):
        path = output_path + ext
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Ablation bar chart → {path}")
    plt.close()


def export_solver_time_histogram(
    solve_times: np.ndarray,
    output_path: str,
    realtime_budget_s: float = 0.05,
    method_label: str = "",
) -> None:
    """Export a histogram of MPC solver wall-clock times.

    A vertical dashed line marks the real-time budget.  The fraction of
    steps violating the budget is annotated on the plot.

    Parameters
    ----------
    solve_times : array-like, shape (N,)
        Solver times in seconds.
    output_path : str
        Path prefix (no extension).  Both ``.pdf`` and ``.png`` are written.
    realtime_budget_s : float
        Real-time budget [s].  Default 50 ms.
    method_label : str
        Optional method name for the plot title.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.evaluation.metrics import compute_solver_stats

    times = np.asarray(solve_times, dtype=float)
    times = times[~np.isnan(times)]
    stats = compute_solver_stats(times, realtime_budget_s=realtime_budget_s)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(times * 1e3, bins=50, color="steelblue", edgecolor="white",
            linewidth=0.5, alpha=0.85)
    ax.axvline(x=realtime_budget_s * 1e3, color="crimson",
               linestyle="--", linewidth=2.0,
               label=f"Budget = {realtime_budget_s * 1e3:.0f} ms")

    budget_ms = realtime_budget_s * 1e3
    viol_pct = stats["budget_violation_rate"] * 100
    ax.text(
        0.98, 0.96,
        f"Mean: {stats['mean_s']*1e3:.2f} ms\n"
        f"P99:  {stats['p99_s']*1e3:.2f} ms\n"
        f"Max:  {stats['max_s']*1e3:.2f} ms\n"
        f"Violations: {viol_pct:.2f}%",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=11, family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    title = "MPC Solver Time Distribution"
    if method_label:
        title += f" — {method_label}"
    ax.set_xlabel("Solver Wall-Clock Time [ms]", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    for ext in (".pdf", ".png"):
        path = output_path + ext
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Solver time histogram → {path}")
    plt.close()


def _plot_compute_time(df: pd.DataFrame, out_dir: str) -> None:
    """Computation time comparison across algorithms for real-time feasibility analysis.

    Produces two sub-figures:
    (a) Box plot of per-trial mean solver time per algorithm with a 20 ms RT budget line.
    (b) Grouped bar chart of mean / P95 / worst-case max solver time per algorithm,
        with a horizontal reference at the 50 Hz control-loop budget (20 ms).

    Only algorithms that have a ``mean_solve_time_s`` column (i.e. MPC variants
    with IPOPT) are plotted; PN/SMC entries with NaN are silently skipped.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if "mean_solve_time_s" not in df.columns:
        print("No solve-time data found — skipping compute-time plot.")
        return

    _RT_BUDGET_MS = 20.0   # 50 Hz → 20 ms budget

    algos_all = df["algorithm"].unique().tolist()
    # Only include algorithms that actually have solver timing data
    algos = [
        a for a in algos_all
        if df[df["algorithm"] == a]["mean_solve_time_s"].notna().any()
    ]
    if not algos:
        print("No solver timing data found — skipping compute-time plot.")
        return

    colors = plt.cm.Set2(np.linspace(0, 1, len(algos)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── (a) Box plot of per-trial mean solve time ─────────────────────────
    ax = axes[0]
    box_data = [
        df[df["algorithm"] == a]["mean_solve_time_s"].dropna().values * 1000.0
        for a in algos
    ]
    bp = ax.boxplot(
        box_data,
        labels=algos,
        patch_artist=True,
        notch=False,
        widths=0.5,
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(_RT_BUDGET_MS, color="red", linestyle="--", linewidth=1.5,
               label=f"RT budget ({_RT_BUDGET_MS:.0f} ms, 50 Hz)")
    ax.set_ylabel("Mean Solver Time per Trial [ms]", fontsize=11)
    ax.set_title("Per-Trial Mean Solve Time Distribution", fontsize=12, fontweight="bold")
    ax.set_xticklabels(algos, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # ── (b) Grouped bar: mean / P95 / worst-case max solve time ──────────
    ax = axes[1]
    x = np.arange(len(algos))
    width = 0.25

    mean_vals, p95_vals, worst_vals = [], [], []
    for a in algos:
        sub = df[df["algorithm"] == a]
        max_col = sub["max_solve_time_s"].dropna() if "max_solve_time_s" in sub.columns else pd.Series(dtype=float)
        mean_col = sub["mean_solve_time_s"].dropna() if "mean_solve_time_s" in sub.columns else pd.Series(dtype=float)
        mean_vals.append(mean_col.mean() * 1000.0 if len(mean_col) > 0 else 0.0)
        p95_vals.append(max_col.quantile(0.95) * 1000.0 if len(max_col) > 0 else 0.0)
        worst_vals.append(max_col.max() * 1000.0 if len(max_col) > 0 else 0.0)

    b1 = ax.bar(x - width, mean_vals,   width, label="Mean solve time",       color="#4e9af1", alpha=0.85)
    b2 = ax.bar(x,          p95_vals,   width, label="P95 max solve time",    color="#f4a261", alpha=0.85)
    b3 = ax.bar(x + width,  worst_vals, width, label="Worst max solve time",  color="#e76f51", alpha=0.85)

    ax.axhline(_RT_BUDGET_MS, color="red", linestyle="--", linewidth=1.5,
               label=f"RT budget ({_RT_BUDGET_MS:.0f} ms)")

    # Annotate bars
    for bar_grp in (b1, b2, b3):
        for bar in bar_grp:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.2,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7, rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Solver Time [ms]", fontsize=11)
    ax.set_title("Solver Time: Mean / P95 / Worst-Case\nvs. 50 Hz Realtime Budget",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(
        "Computation Time Analysis — Realtime Feasibility for Drone Interception\n"
        f"(Red dashed line = {_RT_BUDGET_MS:.0f} ms / 50 Hz control-loop budget)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_compute_time.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Compute-time figure → {path}")

    # ── CDF of max solve time per algorithm ───────────────────────────────
    if "max_solve_time_s" in df.columns:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        colors2 = plt.cm.tab10(np.linspace(0, 1, len(algos)))
        for i, a in enumerate(algos):
            vals = df[df["algorithm"] == a]["max_solve_time_s"].dropna().sort_values().values * 1000.0
            if len(vals) == 0:
                continue
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax2.step(vals, cdf, where="post", color=colors2[i], linewidth=2, label=a)
        ax2.axvline(_RT_BUDGET_MS, color="red", linestyle="--", linewidth=1.5,
                    label=f"RT budget ({_RT_BUDGET_MS:.0f} ms)")
        ax2.set_xlabel("Worst-Case (Max) Solver Time per Trial [ms]", fontsize=11)
        ax2.set_ylabel("Cumulative Probability", fontsize=11)
        ax2.set_ylim(0, 1.05)
        ax2.set_title("CDF of Worst-Case Solver Time\n"
                      "(x < RT budget ⟹ real-time feasible for that trial)",
                      fontsize=12, fontweight="bold")
        ax2.legend(fontsize=9, loc="lower right")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        cdf_path = os.path.join(out_dir, "fig_compute_time_cdf.png")
        plt.savefig(cdf_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Compute-time CDF → {cdf_path}")


if __name__ == "__main__":
    main()
