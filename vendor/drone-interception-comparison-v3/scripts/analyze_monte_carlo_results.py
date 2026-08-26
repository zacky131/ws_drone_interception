#!/usr/bin/env python
"""
Post-hoc analysis of Monte Carlo results.

Reads the ``monte_carlo_detailed.csv`` produced by ``run_monte_carlo.py`` and
generates publication-quality figures, a summary CSV, and LaTeX tables —
without re-running any simulations.

Usage
-----
python scripts/analyze_monte_carlo_results.py --results-dir monte_carlo_results/

# Specify reference algorithm explicitly:
python scripts/analyze_monte_carlo_results.py --results-dir monte_carlo_results/ --reference proposed_full

# Save outputs to a different directory:
python scripts/analyze_monte_carlo_results.py --results-dir monte_carlo_results/ --out-dir paper_figures/

Outputs
-------
    summary_stats.csv           — mean ± std per algorithm per scenario, with bootstrap 95 % CI
    ablation_deltas.csv         — relative improvement of reference vs every other algorithm
    fig_success_rate.png        — success rate bar chart with 95 % CI error bars
    fig_box_intercept_time.png  — box plot of intercept time (successful trials only)
    fig_box_terminal_distance.png — box plot of terminal miss distance
    fig_box_control_effort.png  — box plot of control effort
    fig_box_rmse_pos.png        — box plot of estimator position RMSE
    fig_failure_modes.png       — stacked bar chart of failure categories
    fig_heatmap.png             — algorithm × metric heatmap (normalised)
    ablation_table.tex          — LaTeX table for paper
    comparison_table.tex        — extended LaTeX comparison table
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

# ── Display labels ────────────────────────────────────────────────────────────

_LABELS: Dict[str, str] = {
    "proposed_full":               "Proposed (Full)",
    "ablation_no_ekf":             "Ablation: No EKF (RLS)",
    "ablation_ideal_pursuer":      "Ablation: Ideal Pursuer",
    "ablation_no_disturbance":     "Ablation: No Disturbance",
    "ablation_fixed_target_model": "Ablation: Fixed Target Model",
    "baseline_pn":                 "Baseline: PN",
    "baseline_smc":                "Baseline: SMC",
    "baseline_standard_mpc":       "Baseline: Standard MPC",
    "baseline_rls_adaptive_mpc":   "Baseline: RLS-AMPC",
    # Legacy
    "ekf_adaptive_mpc":            "EKF-AMPC (legacy)",
    "rls_adaptive_mpc":            "RLS-AMPC (legacy)",
    "standard_mpc":                "Standard MPC (legacy)",
    "pn":                          "PN (legacy)",
    "smc":                         "SMC (legacy)",
}

_SHORT_LABELS: Dict[str, str] = {
    "proposed_full":               "Proposed",
    "ablation_no_ekf":             "No EKF",
    "ablation_ideal_pursuer":      "Ideal Pursuer",
    "ablation_no_disturbance":     "No Disturb.",
    "ablation_fixed_target_model": "Fixed Target",
    "baseline_pn":                 "PN",
    "baseline_smc":                "SMC",
    "baseline_standard_mpc":       "Std-MPC",
    "baseline_rls_adaptive_mpc":   "RLS-AMPC",
    "ekf_adaptive_mpc":            "EKF-AMPC",
    "rls_adaptive_mpc":            "RLS-AMPC",
    "standard_mpc":                "Std-MPC",
    "pn":                          "PN",
    "smc":                         "SMC",
}

# Preferred display order (any extra algorithms are appended alphabetically)
_ORDER = [
    "proposed_full",
    "ablation_no_ekf",
    "ablation_ideal_pursuer",
    "ablation_no_disturbance",
    "ablation_fixed_target_model",
    "baseline_standard_mpc",
    "baseline_rls_adaptive_mpc",
    "baseline_pn",
    "baseline_smc",
]


def _label(algo: str, short: bool = False) -> str:
    d = _SHORT_LABELS if short else _LABELS
    return d.get(algo, algo)


def _sort_algorithms(algos: List[str]) -> List[str]:
    ordered = [a for a in _ORDER if a in algos]
    rest = sorted(a for a in algos if a not in _ORDER)
    return ordered + rest


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def _bootstrap_ci(
    values: np.ndarray,
    stat_fn=np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Return (lower, upper) bootstrap confidence interval."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(values)
    if n < 2:
        v = stat_fn(values)
        return float(v), float(v)
    boot = np.array([stat_fn(rng.choice(values, size=n, replace=True)) for _ in range(n_boot)])
    return float(np.percentile(boot, alpha / 2 * 100)), float(np.percentile(boot, (1 - alpha / 2) * 100))


# ── Summary statistics ────────────────────────────────────────────────────────

def _compute_summary(df: pd.DataFrame, algos: List[str]) -> pd.DataFrame:
    """Per-algorithm summary statistics with bootstrap 95 % CI on success rate."""
    rng = np.random.default_rng(42)
    rows = []
    for algo in algos:
        for scen in df["scenario"].unique():
            sub = df[(df["algorithm"] == algo) & (df["scenario"] == scen)]
            if sub.empty:
                continue
            n = len(sub)
            n_ok = int(sub["success"].sum())
            sr = n_ok / n * 100
            ci_lo, ci_hi = _bootstrap_ci(sub["success"].values * 100.0, rng=rng)

            ok = sub[sub["success"] == True]

            def _mean(col: str) -> float:
                return float(sub[col].mean()) if col in sub.columns else float("nan")

            def _std(col: str) -> float:
                return float(sub[col].std()) if col in sub.columns else float("nan")

            rows.append({
                "algorithm":               algo,
                "label":                   _label(algo, short=True),
                "scenario":                scen,
                "n_trials":                n,
                "n_success":               n_ok,
                "success_rate_pct":        round(sr, 1),
                "sr_ci_lower":             round(ci_lo, 1),
                "sr_ci_upper":             round(ci_hi, 1),
                "mean_intercept_time_s":   round(ok["intercept_time"].mean(), 3) if len(ok) else float("nan"),
                "std_intercept_time_s":    round(ok["intercept_time"].std(), 3) if len(ok) > 1 else float("nan"),
                "mean_terminal_dist_m":    round(_mean("terminal_distance"), 4),
                "std_terminal_dist_m":     round(_std("terminal_distance"), 4),
                "mean_control_effort":     round(_mean("control_effort"), 1),
                "std_control_effort":      round(_std("control_effort"), 1),
                "mean_rmse_pos_m":         round(_mean("rmse_pos"), 4),
                "std_rmse_pos_m":          round(_std("rmse_pos"), 4),
                "mean_rmse_vel":           round(_mean("rmse_vel"), 4),
                "mean_control_smoothness": round(_mean("control_smoothness"), 2),
                "mean_solver_feas_rate":   round(_mean("solver_feasibility_rate"), 4),
                "mean_saturation_rate":    round(_mean("saturation_rate"), 4),
            })
    return pd.DataFrame(rows)


# ── Ablation deltas ───────────────────────────────────────────────────────────

def _compute_deltas(summary: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Relative improvement (%) of *reference* over every other algorithm."""
    ref_rows = summary[summary["algorithm"] == reference]
    if ref_rows.empty:
        print(f"  WARNING: reference algorithm '{reference}' not found in results.")
        return pd.DataFrame()

    rows = []
    for _, other in summary[summary["algorithm"] != reference].iterrows():
        scen = other["scenario"]
        ref = ref_rows[ref_rows["scenario"] == scen]
        if ref.empty:
            continue
        r = ref.iloc[0]

        def _rel_gain(ref_val, other_val, higher_is_better: bool) -> Optional[float]:
            if np.isnan(ref_val) or np.isnan(other_val) or other_val == 0:
                return None
            if higher_is_better:
                return round((ref_val - other_val) / abs(other_val) * 100, 1)
            else:
                return round((other_val - ref_val) / abs(other_val) * 100, 1)

        rows.append({
            "algorithm":       other["algorithm"],
            "label":           _label(other["algorithm"], short=True),
            "scenario":        scen,
            "sr_gain_pct":     _rel_gain(r["success_rate_pct"], other["success_rate_pct"], True),
            "dist_reduct_pct": _rel_gain(r["mean_terminal_dist_m"], other["mean_terminal_dist_m"], False),
            "time_reduct_pct": _rel_gain(r["mean_intercept_time_s"], other["mean_intercept_time_s"], False),
            "effort_reduct_pct":    _rel_gain(r["mean_control_effort"], other["mean_control_effort"], False),
            "rmse_pos_reduct_pct":  _rel_gain(r["mean_rmse_pos_m"], other["mean_rmse_pos_m"], False),
        })
    return pd.DataFrame(rows)


# ── Figures ───────────────────────────────────────────────────────────────────

def _make_palette(n: int):
    import matplotlib.pyplot as plt
    if n <= 10:
        return [plt.cm.tab10(i / 10) for i in range(n)]
    return [plt.cm.tab20(i / 20) for i in range(n)]


def _fig_success_rate(summary: pd.DataFrame, algos: List[str], out_dir: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(10, len(algos) * 1.3), 5))
    colours = _make_palette(len(algos))
    x = np.arange(len(algos))
    width = 0.6

    for i, algo in enumerate(algos):
        row = summary[summary["algorithm"] == algo]
        if row.empty:
            continue
        row = row.iloc[0]
        sr = row["success_rate_pct"]
        ci_lo = sr - row["sr_ci_lower"]
        ci_hi = row["sr_ci_upper"] - sr
        bar = ax.bar(i, sr, width=width, color=colours[i], edgecolor="black", linewidth=0.7,
                     label=_label(algo, short=True))
        ax.errorbar(i, sr, yerr=[[ci_lo], [ci_hi]], fmt="none",
                    color="black", capsize=5, linewidth=1.5)
        ax.text(i, sr + ci_hi + 2, f"{sr:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([_label(a, short=True) for a in algos], rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Success Rate (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title("Monte Carlo Success Rate with 95% Bootstrap CI", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(100, color="green", ls="--", lw=1, alpha=0.5)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_success_rate.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  fig_success_rate.png")


def _fig_boxplot(df: pd.DataFrame, algos: List[str], col: str,
                 ylabel: str, title: str, fname: str, out_dir: str,
                 success_only: bool = False) -> None:
    import matplotlib.pyplot as plt

    data, labels, colours = [], [], _make_palette(len(algos))
    for algo in algos:
        sub = df[df["algorithm"] == algo]
        if success_only:
            sub = sub[sub["success"] == True]
        vals = sub[col].dropna().values if col in sub.columns else np.array([])
        data.append(vals)
        labels.append(_label(algo, short=True))

    fig, ax = plt.subplots(figsize=(max(10, len(algos) * 1.3), 5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2),
                    flierprops=dict(marker=".", markersize=4, alpha=0.5))
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.75)

    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title + (" (successful trials only)" if success_only else ""),
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  {fname}")


def _fig_failure_modes(df: pd.DataFrame, algos: List[str], out_dir: str) -> None:
    import matplotlib.pyplot as plt

    if "failure_category" not in df.columns:
        return

    categories = sorted(df["failure_category"].dropna().unique().tolist())
    cat_colours = dict(zip(categories, _make_palette(len(categories))))

    fig, ax = plt.subplots(figsize=(max(10, len(algos) * 1.3), 5))
    bottoms = np.zeros(len(algos))
    x = np.arange(len(algos))

    for cat in categories:
        counts = []
        for algo in algos:
            sub = df[df["algorithm"] == algo]
            n = len(sub)
            if n == 0:
                counts.append(0.0)
            else:
                counts.append(sub[sub["failure_category"] == cat].shape[0] / n * 100)
        ax.bar(x, counts, bottom=bottoms, label=cat, color=cat_colours[cat], edgecolor="white", linewidth=0.5)
        bottoms += np.array(counts)

    ax.set_xticks(x)
    ax.set_xticklabels([_label(a, short=True) for a in algos], rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Percentage of trials (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_title("Trial Outcome Distribution by Algorithm", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_failure_modes.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  fig_failure_modes.png")


def _fig_heatmap(summary: pd.DataFrame, algos: List[str], out_dir: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import RdYlGn

    metrics = {
        "success_rate_pct":     ("Success Rate (%)", True),
        "mean_intercept_time_s":("Intercept Time (s)", False),
        "mean_terminal_dist_m": ("Terminal Dist. (m)", False),
        "mean_control_effort":  ("Control Effort", False),
        "mean_rmse_pos_m":      ("RMSE pos (m)", False),
    }
    # Drop metrics not present
    metrics = {k: v for k, v in metrics.items() if k in summary.columns}
    if not metrics:
        return

    mat = np.full((len(algos), len(metrics)), np.nan)
    for i, algo in enumerate(algos):
        row = summary[summary["algorithm"] == algo]
        if row.empty:
            continue
        for j, col in enumerate(metrics.keys()):
            mat[i, j] = row.iloc[0][col]

    # Normalise each column 0–1 (direction-aware)
    norm_mat = np.full_like(mat, np.nan)
    for j, (col, (_, higher_is_better)) in enumerate(metrics.items()):
        col_vals = mat[:, j]
        valid = col_vals[~np.isnan(col_vals)]
        if len(valid) < 2:
            norm_mat[:, j] = col_vals
            continue
        lo, hi = np.nanmin(col_vals), np.nanmax(col_vals)
        if hi == lo:
            norm_mat[:, j] = 0.5
            continue
        normalized = (col_vals - lo) / (hi - lo)
        norm_mat[:, j] = normalized if higher_is_better else (1 - normalized)

    fig, ax = plt.subplots(figsize=(len(metrics) * 2, max(4, len(algos) * 0.7)))
    im = ax.imshow(norm_mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([v[0] for v in metrics.values()], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels([_label(a, short=True) for a in algos], fontsize=9)

    # Annotate with raw values
    for i in range(len(algos)):
        for j, col in enumerate(metrics.keys()):
            v = mat[i, j]
            if not np.isnan(v):
                text = f"{v:.1f}" if abs(v) >= 0.1 else f"{v:.3f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8,
                        color="black" if 0.3 < norm_mat[i, j] < 0.75 else "white")

    ax.set_title("Algorithm Performance Heatmap\n(green = better)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Normalised score (1 = best)")
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_heatmap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  fig_heatmap.png")


def _fig_ablation_deltas(deltas: pd.DataFrame, out_dir: str) -> None:
    """Grouped bar chart: relative gain of reference over each ablation/baseline."""
    import matplotlib.pyplot as plt

    if deltas.empty:
        return

    delta_cols = {
        "sr_gain_pct":          "Success Rate\nGain (%)",
        "dist_reduct_pct":      "Miss Dist.\nReduction (%)",
        "time_reduct_pct":      "Intercept Time\nReduction (%)",
        "effort_reduct_pct":    "Control Effort\nReduction (%)",
        "rmse_pos_reduct_pct":  "RMSE-pos\nReduction (%)",
    }
    available = {k: v for k, v in delta_cols.items() if k in deltas.columns}
    if not available:
        return

    algos = deltas["algorithm"].tolist()
    n_algos = len(algos)
    n_metrics = len(available)
    colours = _make_palette(n_metrics)

    fig, ax = plt.subplots(figsize=(max(10, n_algos * 1.5), 6))
    x = np.arange(n_algos)
    width = 0.8 / n_metrics

    for i, (col, label) in enumerate(available.items()):
        vals = [deltas[deltas["algorithm"] == a][col].values[0]
                if not deltas[deltas["algorithm"] == a].empty else np.nan
                for a in algos]
        vals = [v if v is not None else np.nan for v in vals]
        offsets = x + (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(offsets, vals, width=width * 0.9, color=colours[i],
                      edgecolor="black", linewidth=0.5, label=label)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2,
                        b.get_height() + (1 if v >= 0 else -3),
                        f"{v:+.1f}", ha="center", va="bottom", fontsize=7)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([_label(a, short=True) for a in algos], rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Relative improvement over comparison (%)", fontsize=10)
    ax.set_title("Relative Gain of Proposed Method vs Ablations & Baselines", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "fig_ablation_deltas.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  fig_ablation_deltas.png")


# ── LaTeX tables ──────────────────────────────────────────────────────────────

def _latex_main_table(summary: pd.DataFrame, algos: List[str], reference: str) -> str:
    """Generate the main comparison LaTeX table."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Monte Carlo Performance Comparison (" + str(summary["n_trials"].max()) + r" trials/algorithm)}",
        r"\label{tab:monte_carlo_comparison}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Algorithm & SR (\%) & SR 95\% CI & Intercept (s) & Miss Dist. (m) & Control Effort & RMSE pos (m) \\",
        r"\midrule",
    ]

    for algo in algos:
        row = summary[summary["algorithm"] == algo]
        if row.empty:
            continue
        r = row.iloc[0]
        label = _label(algo, short=True).replace("_", r"\_")
        sr = f"{r['success_rate_pct']:.1f}"
        ci = f"[{r['sr_ci_lower']:.1f}, {r['sr_ci_upper']:.1f}]"
        it = f"{r['mean_intercept_time_s']:.2f}" if not np.isnan(r["mean_intercept_time_s"]) else "---"
        td = f"{r['mean_terminal_dist_m']:.3f}" if not np.isnan(r["mean_terminal_dist_m"]) else "---"
        ce = f"{r['mean_control_effort']:.1f}" if not np.isnan(r["mean_control_effort"]) else "---"
        rp = f"{r['mean_rmse_pos_m']:.4f}" if not np.isnan(r["mean_rmse_pos_m"]) else "---"

        bold_open  = r"\textbf{" if algo == reference else ""
        bold_close = r"}"        if algo == reference else ""
        lines.append(
            f"{bold_open}{label}{bold_close} & {sr} & {ci} & {it} & {td} & {ce} & {rp} \\\\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\footnotesize SR = success rate; CI = 95\% bootstrap confidence interval; "
        r"proposed method shown in \textbf{bold}.",
        r"\end{tablenotes}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def _latex_ablation_table(deltas: pd.DataFrame, reference: str) -> str:
    """Generate the ablation delta LaTeX table."""
    if deltas.empty:
        return ""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Relative Improvement of Proposed Method over Ablations and Baselines}",
        r"\label{tab:ablation_deltas}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Comparison & $\Delta$SR (\%) & $\Delta$Dist (\%) & $\Delta$Time (\%) & $\Delta$Effort (\%) & $\Delta$RMSE-pos (\%) \\",
        r"\midrule",
    ]
    for _, row in deltas.iterrows():
        label = _label(row["algorithm"], short=True).replace("_", r"\_")

        def _fmt(v) -> str:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "---"
            return f"{v:+.1f}"

        lines.append(
            f"{label} & {_fmt(row.get('sr_gain_pct'))} & "
            f"{_fmt(row.get('dist_reduct_pct'))} & "
            f"{_fmt(row.get('time_reduct_pct'))} & "
            f"{_fmt(row.get('effort_reduct_pct'))} & "
            f"{_fmt(row.get('rmse_pos_reduct_pct'))} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\footnotesize Positive values indicate improvement of the proposed method over the comparison. "
        r"$\Delta$SR = success-rate gain; $\Delta$Dist = miss-distance reduction; "
        r"$\Delta$Time = intercept-time reduction; $\Delta$Effort = control-effort reduction.",
        r"\end{tablenotes}",
        r"\end{table}",
    ]
    return "\n".join(lines)


# ── Console summary ───────────────────────────────────────────────────────────

def _print_summary(summary: pd.DataFrame, algos: List[str]) -> None:
    print("\n" + "=" * 90)
    print("  MONTE CARLO RESULTS SUMMARY")
    print("=" * 90)
    header = f"{'Algorithm':<35} {'N':>5} {'SR%':>6} {'95% CI':>15} {'Intercept(s)':>13} {'MissDist(m)':>12} {'RMSE pos':>10}"
    print(header)
    print("-" * 90)
    for algo in algos:
        row = summary[summary["algorithm"] == algo]
        if row.empty:
            continue
        r = row.iloc[0]
        ci = f"[{r['sr_ci_lower']:.0f}, {r['sr_ci_upper']:.0f}]"
        it = f"{r['mean_intercept_time_s']:.2f}" if not np.isnan(r["mean_intercept_time_s"]) else "    —   "
        td = f"{r['mean_terminal_dist_m']:.3f}" if not np.isnan(r["mean_terminal_dist_m"]) else "    —   "
        rp = f"{r['mean_rmse_pos_m']:.4f}" if not np.isnan(r["mean_rmse_pos_m"]) else "   —    "
        print(f"  {_label(algo, short=True):<33} {int(r['n_trials']):>5} {r['success_rate_pct']:>6.1f} {ci:>15} {it:>13} {td:>12} {rp:>10}")
    print("=" * 90)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc analysis of Monte Carlo results (no re-simulation)"
    )
    parser.add_argument(
        "--results-dir", type=str,
        default=os.path.join(_PROJECT_ROOT, "monte_carlo_results"),
        help="Directory containing monte_carlo_detailed.csv",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory for figures and tables (default: same as --results-dir)",
    )
    parser.add_argument(
        "--reference", type=str, default="proposed_full",
        help="Algorithm used as the reference for ablation delta computation",
    )
    parser.add_argument(
        "--scenario", type=str, default=None,
        help="Filter to a single scenario name (default: use first scenario found)",
    )
    args = parser.parse_args()

    detailed_csv = os.path.join(args.results_dir, "monte_carlo_detailed.csv")
    if not os.path.isfile(detailed_csv):
        sys.exit(f"ERROR: {detailed_csv} not found. Run run_monte_carlo.py first.")

    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(detailed_csv)
    print(f"Loaded {len(df)} trial records from {detailed_csv}")
    print(f"Algorithms: {sorted(df['algorithm'].unique())}")
    print(f"Scenarios:  {sorted(df['scenario'].unique())}")

    # Scenario filter
    if args.scenario:
        df = df[df["scenario"] == args.scenario]
        if df.empty:
            sys.exit(f"ERROR: No data for scenario '{args.scenario}'.")
        print(f"Filtered to scenario: {args.scenario}")
    else:
        scen = df["scenario"].unique()
        if len(scen) > 1:
            chosen = scen[0]
            print(f"  Multiple scenarios found, using '{chosen}'. Use --scenario to select.")
            df = df[df["scenario"] == chosen]

    algos = _sort_algorithms(df["algorithm"].unique().tolist())

    # Reference fallback
    reference = args.reference
    if reference not in algos:
        reference = algos[0]
        print(f"  WARNING: '{args.reference}' not in results; using '{reference}' as reference.")

    # ── Compute statistics ────────────────────────────────────────────────
    summary = _compute_summary(df, algos)
    deltas  = _compute_deltas(summary, reference)

    # ── Print to console ──────────────────────────────────────────────────
    _print_summary(summary, algos)

    # ── Save CSVs ─────────────────────────────────────────────────────────
    summary_path = os.path.join(out_dir, "summary_stats.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\n  summary_stats.csv")

    if not deltas.empty:
        deltas_path = os.path.join(out_dir, "ablation_deltas.csv")
        deltas.to_csv(deltas_path, index=False)
        print(f"  ablation_deltas.csv")

    # ── Figures ───────────────────────────────────────────────────────────
    print("\nGenerating figures:")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401

        _fig_success_rate(summary, algos, out_dir)
        _fig_boxplot(df, algos, "intercept_time", "Intercept Time [s]",
                     "Intercept Time Distribution", "fig_box_intercept_time.png", out_dir, success_only=True)
        _fig_boxplot(df, algos, "terminal_distance", "Terminal Miss Distance [m]",
                     "Terminal Miss Distance Distribution", "fig_box_terminal_distance.png", out_dir)
        _fig_boxplot(df, algos, "control_effort", "Control Effort",
                     "Control Effort Distribution", "fig_box_control_effort.png", out_dir)
        if "rmse_pos" in df.columns:
            _fig_boxplot(df, algos, "rmse_pos", "Position RMSE [m]",
                         "Estimator Position RMSE Distribution", "fig_box_rmse_pos.png", out_dir)
        _fig_failure_modes(df, algos, out_dir)
        _fig_heatmap(summary, algos, out_dir)
        if not deltas.empty:
            _fig_ablation_deltas(deltas, out_dir)

    except ImportError:
        print("  matplotlib not available — skipping figures.")

    # ── LaTeX tables ──────────────────────────────────────────────────────
    print("\nGenerating LaTeX tables:")
    tex_main = _latex_main_table(summary, algos, reference)
    tex_main_path = os.path.join(out_dir, "comparison_table.tex")
    with open(tex_main_path, "w") as f:
        f.write(tex_main)
    print(f"  comparison_table.tex")

    if not deltas.empty:
        tex_abl = _latex_ablation_table(deltas, reference)
        tex_abl_path = os.path.join(out_dir, "ablation_table.tex")
        with open(tex_abl_path, "w") as f:
            f.write(tex_abl)
        print(f"  ablation_table.tex")

    print(f"\nAll outputs saved to: {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
