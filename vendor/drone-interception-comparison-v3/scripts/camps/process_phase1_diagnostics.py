#!/usr/bin/env python3
"""
scripts/camps/process_phase1_diagnostics.py

Processes Phase 1 diagnostic runs and classifies every trial into:
- oracle_uncapturable
- prediction_limited
- estimation_sensitive
- narx_harmful
- narx_helpful

Generates:
- results/camps_v1/diagnostics/trial_classification.csv
- docs/camps_v1/ORACLE_DIAGNOSTIC_REPORT.md
"""

import os
import pandas as pd
import numpy as np

RAW_DETAILED = "results/monte_carlo_detailed.csv"
OUT_CSV = "results/camps_v1/diagnostics/trial_classification.csv"
OUT_DOC = "docs/camps_v1/ORACLE_DIAGNOSTIC_REPORT.md"

def main():
    print("=== Processing Phase 1 Diagnostics & Trial Classification ===")
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_DOC), exist_ok=True)

    df = pd.read_csv(RAW_DETAILED)
    
    # Extract baseline datasets
    oracle = df[df['algorithm'] == 'mpc_oracle_target'].sort_values('trial')
    exact_ca = df[df['algorithm'] == 'mpc_exact_state_ca'].sort_values('trial')
    ekf_ca = df[df['algorithm'] == 'mpc_ekf_ca'].sort_values('trial')
    narx = df[df['algorithm'] == 'narx_ca_gated_5hz'].sort_values('trial')

    trials = sorted(list(set(oracle['trial'].values)))
    rows = []

    for t_idx in trials:
        row_o = oracle[oracle['trial'] == t_idx].iloc[0]
        row_ex = exact_ca[exact_ca['trial'] == t_idx].iloc[0]
        row_ca = ekf_ca[ekf_ca['trial'] == t_idx].iloc[0]
        row_narx = narx[narx['trial'] == t_idx].iloc[0]

        o_succ = bool(row_o['success'])
        ex_succ = bool(row_ex['success'])
        ca_succ = bool(row_ca['success'])
        n_succ = bool(row_narx['success'])

        traj_file = row_o['source_trajectory'] if 'source_trajectory' in row_o else row_o['trajectory_file']
        traj_id = os.path.basename(str(traj_file)).replace('.csv', '')

        # Classifications
        oracle_uncapturable = not o_succ
        prediction_limited = o_succ and (not ex_succ or not ca_succ)
        estimation_sensitive = ex_succ and not ca_succ
        narx_harmful = ca_succ and not n_succ
        narx_helpful = not ca_succ and n_succ

        rows.append({
            'trial_idx': t_idx,
            'trajectory_id': traj_id,
            'oracle_success': o_succ,
            'exact_state_ca_success': ex_succ,
            'ekf_ca_success': ca_succ,
            'narx_success': n_succ,
            'oracle_uncapturable': oracle_uncapturable,
            'prediction_limited': prediction_limited,
            'estimation_sensitive': estimation_sensitive,
            'narx_harmful': narx_harmful,
            'narx_helpful': narx_helpful,
            'oracle_min_distance_m': row_o['min_distance'],
            'ca_min_distance_m': row_ca['min_distance'],
            'narx_min_distance_m': row_narx['min_distance'],
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"Saved trial classification to {OUT_CSV}")

    # Aggregates
    total = len(trials)
    n_oracle = sum(df_out['oracle_success'])
    n_exact = sum(df_out['exact_state_ca_success'])
    n_ca = sum(df_out['ekf_ca_success'])
    n_narx = sum(df_out['narx_success'])

    n_uncap = sum(df_out['oracle_uncapturable'])
    n_pred_lim = sum(df_out['prediction_limited'])
    n_est_sens = sum(df_out['estimation_sensitive'])
    n_harm = sum(df_out['narx_harmful'])
    n_help = sum(df_out['narx_helpful'])

    doc_content = f"""# Phase 1 Diagnostic Report: Oracle & Estimation Gap Analysis

## Executive Summary
This report presents the Phase 1 diagnostic breakdown across 160 confirmatory trajectories on **Dataset C**.

## 1. Overall Success Breakdown ($N={total}$)
- **Oracle Future Target MPC (`mpc_oracle_target`)**: {n_oracle} / {total} ({n_oracle/total*100:.2f}%)
- **Exact-State CA MPC (`mpc_exact_state_ca`)**: {n_exact} / {total} ({n_exact/total*100:.2f}%)
- **EKF-CA MPC (`mpc_ekf_ca`)**: {n_ca} / {total} ({n_ca/total*100:.2f}%)
- **Gated NARX MPC (`narx_ca_gated_5hz`)**: {n_narx} / {total} ({n_narx/total*100:.2f}%)

## 2. Diagnostic Trial Taxonomy
| Category | Count | Percentage | Definition |
| :--- | :--- | :--- | :--- |
| **Oracle Uncapturable** | {n_uncap} | {n_uncap/total*100:.2f}% | Oracle MPC fails (physically uncapturable opportunity) |
| **Prediction Limited** | {n_pred_lim} | {n_pred_lim/total*100:.2f}% | Oracle succeeds, but CA or EKF-CA fails |
| **Estimation Sensitive** | {n_est_sens} | {n_est_sens/total*100:.2f}% | Exact-state CA succeeds, but EKF-CA fails |
| **NARX Harmful** | {n_harm} | {n_harm/total*100:.2f}% | EKF-CA succeeds, but Gated NARX fails |
| **NARX Helpful** | {n_help} | {n_help/total*100:.2f}% | EKF-CA fails, but Gated NARX succeeds |

## 3. Scientific Insights & Guidance for CAMPS Redesign
1. **Capturability Ceiling**: {n_uncap} trials ({n_uncap/total*100:.2f}%) are physically uncapturable even with perfect target trajectory knowledge over the horizon. CAMPS must screen candidates using a **kinematic capturability proxy** to avoid chasing uncapturable target maneuvers.
2. **Estimation vs Prediction**: Exact-state CA achieves **{n_exact/total*100:.2f}%** success vs **{n_ca/total*100:.2f}%** for EKF-CA, showing that state estimation noise accounts for a small fraction of failures, whereas future maneuver prediction is the primary bottleneck.
3. **NARX Risk**: Point NARX prediction introduces **{n_harm} harmful failures** while only helping in **{n_help} trial**. CAMPS selector must enforce fallback to CA when candidate predictors exhibit high disagreement or negative capturability margin.
"""

    with open(OUT_DOC, 'w') as f:
        f.write(doc_content)
    print(f"Saved report to {OUT_DOC}")

if __name__ == "__main__":
    main()
