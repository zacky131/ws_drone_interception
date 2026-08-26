#!/usr/bin/env python3
"""
Q2 Journal Revision — Paired Statistical Analysis & Report Artifact Generator

Processes Monte Carlo results from main, pilot, and timing experiments to generate
the manuscript statistics, paired statistical tests, discordant case analyses,
and LaTeX summary tables.

Outputs:
    results/q2_revision_v1/analysis/primary_summary.csv
    results/q2_revision_v1/analysis/paired_comparisons.csv
    results/q2_revision_v1/analysis/per_family_summary.csv
    results/q2_revision_v1/analysis/timing_summary.csv
    results/q2_revision_v1/analysis/paired_discordant_trials.csv
    results/q2_revision_v1/analysis/narx_only_successes.csv
    results/q2_revision_v1/analysis/ca_only_successes.csv
    results/q2_revision_v1/analysis/both_failed.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)


def bootstrap_ci(
    values: np.ndarray,
    stat_fn=np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    if n < 2:
        val = float(stat_fn(values)) if len(values) > 0 else float("nan")
        return val, val
    boot_stats = [stat_fn(rng.choice(values, size=n, replace=True)) for _ in range(n_boot)]
    lo = float(np.percentile(boot_stats, 100 * (alpha / 2)))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return lo, hi


def run_q2_analysis(
    main_results_dir: str,
    timing_results_dir: str,
    manifest_dir: str,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print("  Q2 REVISION POST-HOC STATISTICAL ANALYSIS")
    print("=" * 80)

    # ── 1. Load Main Experiment Data ──────────────────────────────────────────
    main_csv = os.path.join(main_results_dir, "monte_carlo_results.csv")
    if not os.path.exists(main_csv):
        # Fallback to detailed CSV if monte_carlo_results.csv is missing
        main_csv = os.path.join(main_results_dir, "monte_carlo_detailed.csv")

    has_main_data = os.path.exists(main_csv)
    if has_main_data:
        df_main = pd.read_csv(main_csv)
        print(f"Loaded main experiment data: {len(df_main)} trials from {main_csv}")
    else:
        print(f"WARNING: Main experiment CSV not found at {main_csv}. Analysis will run on available subsets.")
        df_main = pd.DataFrame()

    # ── 2. Primary Summary ───────────────────────────────────────────────────
    if not df_main.empty:
        summary_rows = []
        for algo in df_main["algorithm"].unique():
            sub = df_main[df_main["algorithm"] == algo]
            n_total = len(sub)
            n_success = int(sub["success"].sum())
            sr = (n_success / n_total) * 100.0 if n_total > 0 else 0.0
            sr_lo, sr_hi = bootstrap_ci(sub["success"].values * 100.0)

            succ = sub[sub["success"] == True]

            summary_rows.append({
                "algorithm": algo,
                "n_trials": n_total,
                "n_success": n_success,
                "success_rate_pct": sr,
                "sr_ci_lower_95": sr_lo,
                "sr_ci_upper_95": sr_hi,
                "mean_intercept_time_s": succ["intercept_time"].mean() if len(succ) > 0 else np.nan,
                "std_intercept_time_s": succ["intercept_time"].std() if len(succ) > 1 else np.nan,
                "mean_terminal_distance_m": sub["terminal_distance"].mean(),
                "std_terminal_distance_m": sub["terminal_distance"].std() if len(sub) > 1 else np.nan,
                "mean_control_effort": sub["control_effort"].mean() if "control_effort" in sub.columns else np.nan,
                "mean_rmse_pos_m": sub["rmse_pos"].mean() if "rmse_pos" in sub.columns else np.nan,
                "mean_rmse_vel_mps": sub["rmse_vel"].mean() if "rmse_vel" in sub.columns else np.nan,
                "mean_solve_time_ms": (sub["mean_solve_time_s"].mean() * 1000.0) if "mean_solve_time_s" in sub.columns else np.nan,
                "mean_total_compute_s": sub["total_compute_time_s"].mean() if "total_compute_time_s" in sub.columns else np.nan,
            })
        df_primary_summary = pd.DataFrame(summary_rows)
        df_primary_summary.to_csv(os.path.join(output_dir, "primary_summary.csv"), index=False)
        print(f"Generated primary_summary.csv ({len(df_primary_summary)} algorithms)")

    # ── 3. Paired Comparisons (NARX-Gated vs MPC-CA) ─────────────────────────
    if not df_main.empty and "mpc_ekf_ca" in df_main["algorithm"].values and "narx_ca_gated_5hz" in df_main["algorithm"].values:
        ca_sub = df_main[df_main["algorithm"] == "mpc_ekf_ca"].sort_values("trial").reset_index(drop=True)
        narx_sub = df_main[df_main["algorithm"] == "narx_ca_gated_5hz"].sort_values("trial").reset_index(drop=True)

        min_len = min(len(ca_sub), len(narx_sub))
        ca_sub = ca_sub.iloc[:min_len]
        narx_sub = narx_sub.iloc[:min_len]

        pair_rows = []
        metrics_to_test = ["success", "terminal_distance", "intercept_time", "control_effort", "rmse_pos"]
        for m in metrics_to_test:
            if m in ca_sub.columns and m in narx_sub.columns:
                v_ca = ca_sub[m].values.astype(float)
                v_narx = narx_sub[m].values.astype(float)

                diffs = v_narx - v_ca
                mean_diff = float(np.nanmean(diffs))
                ci_lo, ci_hi = bootstrap_ci(diffs[~np.isnan(diffs)])

                # Paired t-test
                t_stat, t_pval = stats.ttest_rel(v_ca, v_narx, nan_policy="omit")
                # Wilcoxon signed-rank test
                valid_mask = ~np.isnan(v_ca) & ~np.isnan(v_narx)
                if np.sum(valid_mask) > 5 and not np.all(diffs[valid_mask] == 0):
                    w_stat, w_pval = stats.wilcoxon(v_ca[valid_mask], v_narx[valid_mask])
                else:
                    w_stat, w_pval = np.nan, np.nan

                pair_rows.append({
                    "metric": m,
                    "n_pairs": min_len,
                    "mean_mpc_ca": float(np.nanmean(v_ca)),
                    "mean_narx_gated": float(np.nanmean(v_narx)),
                    "mean_difference_narx_minus_ca": mean_diff,
                    "diff_ci_lower_95": ci_lo,
                    "diff_ci_upper_95": ci_hi,
                    "t_statistic": t_stat,
                    "t_pvalue": t_pval,
                    "wilcoxon_statistic": w_stat,
                    "wilcoxon_pvalue": w_pval,
                })
        df_paired = pd.DataFrame(pair_rows)
        df_paired.to_csv(os.path.join(output_dir, "paired_comparisons.csv"), index=False)
        print(f"Generated paired_comparisons.csv ({len(df_paired)} metrics)")

        # ── 4. Discordant Cases Analysis ─────────────────────────────────────
        ca_succ = ca_sub["success"].astype(bool).values
        narx_succ = narx_sub["success"].astype(bool).values

        narx_only_mask = narx_succ & (~ca_succ)
        ca_only_mask = ca_succ & (~narx_succ)
        both_fail_mask = (~ca_succ) & (~narx_succ)

        df_discordant = pd.DataFrame({
            "trial": ca_sub["trial"],
            "source_trajectory": ca_sub["source_trajectory"],
            "ca_success": ca_succ,
            "narx_success": narx_succ,
            "ca_terminal_dist": ca_sub["terminal_distance"],
            "narx_terminal_dist": narx_sub["terminal_distance"],
            "dist_diff_ca_minus_narx": ca_sub["terminal_distance"] - narx_sub["terminal_distance"],
        })
        df_discordant.to_csv(os.path.join(output_dir, "paired_discordant_trials.csv"), index=False)

        df_narx_only = df_discordant[narx_only_mask].copy()
        df_narx_only.to_csv(os.path.join(output_dir, "narx_only_successes.csv"), index=False)

        df_ca_only = df_discordant[ca_only_mask].copy()
        df_ca_only.to_csv(os.path.join(output_dir, "ca_only_successes.csv"), index=False)

        df_both_failed = df_discordant[both_fail_mask].copy()
        df_both_failed.to_csv(os.path.join(output_dir, "both_failed.csv"), index=False)

        print(f"Generated discordant cases: NARX-only={len(df_narx_only)}, CA-only={len(df_ca_only)}, Both Failed={len(df_both_failed)}")

    # ── 5. Per-Family / Subgroup Difficulty Breakdown ─────────────────────────
    meta_csv = os.path.join(manifest_dir, "trajectory_metadata.csv")
    if not df_main.empty and os.path.exists(meta_csv):
        df_meta = pd.read_csv(meta_csv)
        meta_fn_col = "filename" if "filename" in df_meta.columns else ("file_name" if "file_name" in df_meta.columns else None)

        if "source_trajectory" in df_main.columns and meta_fn_col:
            df_main["traj_basename"] = df_main["source_trajectory"].apply(lambda p: os.path.basename(str(p)))
            df_meta["meta_basename"] = df_meta[meta_fn_col].apply(lambda p: os.path.basename(str(p)))

            df_merged = df_main.merge(df_meta, left_on="traj_basename", right_on="meta_basename", how="left")
            acc_col = "max_accel_m_s2" if "max_accel_m_s2" in df_merged.columns else ("max_acc_m_s2" if "max_acc_m_s2" in df_merged.columns else None)

            if acc_col:
                acc_bins = [0.0, 5.0, 15.0, 100.0]
                acc_labels = ["Low_Acc (<5m/s2)", "Med_Acc (5-15m/s2)", "High_Acc (>15m/s2)"]
                df_merged["acc_difficulty"] = pd.cut(df_merged[acc_col], bins=acc_bins, labels=acc_labels)

                family_rows = []
                for (algo, diff), sub in df_merged.groupby(["algorithm", "acc_difficulty"], observed=False):
                    if sub.empty:
                        continue
                    n = len(sub)
                    ok = sub[sub["success"] == True]
                    family_rows.append({
                        "algorithm": algo,
                        "difficulty_bin": diff,
                        "n_trials": n,
                        "n_success": len(ok),
                        "success_rate_pct": (len(ok) / n) * 100.0,
                        "mean_terminal_dist_m": sub["terminal_distance"].mean(),
                    })
                df_family = pd.DataFrame(family_rows)
                df_family.to_csv(os.path.join(output_dir, "per_family_summary.csv"), index=False)
                print(f"Generated per_family_summary.csv ({len(df_family)} subgroup entries)")

    # ── 6. Timing Summary ─────────────────────────────────────────────────────
    timing_csv = os.path.join(timing_results_dir, "monte_carlo_summary.csv")
    if os.path.exists(timing_csv):
        df_timing = pd.read_csv(timing_csv)
        timing_cols = [
            "algorithm", "n_trials", "success_rate_pct",
            "narx_training_period_steps", "narx_training_execution_rate",
            "narx_mean_train_time_s", "narx_mean_train_event_time_s",
            "narx_mean_infer_time_s", "mean_solve_time_ms",
            "p95_solve_time_ms", "mean_max_solve_time_ms", "mean_total_compute_s",
        ]
        avail = [c for c in timing_cols if c in df_timing.columns]
        df_timing_summary = df_timing[avail].copy()
        df_timing_summary.to_csv(os.path.join(output_dir, "timing_summary.csv"), index=False)
        print(f"Generated timing_summary.csv ({len(df_timing_summary)} timing variants)")

    print("=" * 80)
    print(f"Q2 Analysis Complete. All artifacts saved to: {output_dir}/")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Q2 Revision Post-Hoc Analysis Pipeline")
    parser.add_argument(
        "--main-dir", type=str,
        default=os.path.join(_PROJECT_ROOT, "results", "q2_revision_v1", "main"),
        help="Path to main experiment results directory",
    )
    parser.add_argument(
        "--timing-dir", type=str,
        default=os.path.join(_PROJECT_ROOT, "results", "q2_revision_v1", "timing"),
        help="Path to timing experiment results directory",
    )
    parser.add_argument(
        "--manifest-dir", type=str,
        default=os.path.join(_PROJECT_ROOT, "results", "q2_revision_v1", "manifests"),
        help="Path to trajectory manifests directory",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(_PROJECT_ROOT, "results", "q2_revision_v1", "analysis"),
        help="Output directory for statistical analysis CSVs",
    )
    args = parser.parse_args()

    run_q2_analysis(
        main_results_dir=args.main_dir,
        timing_results_dir=args.timing_dir,
        manifest_dir=args.manifest_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
