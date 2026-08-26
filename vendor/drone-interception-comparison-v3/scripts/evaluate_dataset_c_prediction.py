#!/usr/bin/env python3
"""
scripts/evaluate_dataset_c_prediction.py

Phase 3: Causal prediction-only evaluation of Dataset C.
Evaluates 3 predictors causally:
1. ca_predictor (Constant Acceleration)
2. narx_raw_predictor (Raw NARX Network)
3. narx_gated_predictor (Trust-gated NARX Prediction)

Horizons evaluated: 0.10s, 0.20s, 0.30s, 0.40s.
Evaluates distribution shifts (pre_shift, immediate_post_shift, recovery).
Saves raw records, summaries, and 5 required publication-grade plots.
"""

import os
import sys
import argparse
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.baselines.mpc_narx import _NARXNet, MPCNARXPredictor
from src.estimation import EKFTargetEstimator
from src.utils.config_schema import ControllerConfig, PursuerConfig, SimulationConfig, EstimatorConfig

PREDICTION_DIR = "/home/t462/Documents/Zacks_research/drone-interception-comparison-v3/results/q2_revision_v1/dataset_c/prediction"
DATASET_CSV_DIR = "/home/t462/Documents/Zacks_research/drone-interception-comparison-v3/data/q2_challenge_v1/generated_6dof/csv"

HORIZON_TIMES = [0.10, 0.20, 0.30, 0.40]
DT = 0.0200  # 50 Hz
HORIZON_STEPS = [int(round(h / DT)) for h in HORIZON_TIMES]  # [5, 10, 15, 20]

FAMILIES = [
    "abrupt_axis_switch",
    "helical_reversal",
    "minimum_jerk_waypoints",
    "mixed_mode_shift",
    "pop_up_dive",
    "rotating_acceleration",
    "s_turn_chicane",
    "variable_radius_turn"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Dataset C Causal Prediction")
    parser.add_argument("--manifest", type=str, default="/home/t462/Documents/Zacks_research/drone-interception-comparison-v3/results/q2_revision_v1/dataset_c/manifests/dataset_c_pilot.txt",
                        help="Manifest file containing list of relative trajectory paths")
    parser.add_argument("--engagement_seeds", type=int, nargs="+", default=[42, 101],
                        help="Engagement seeds per trajectory")
    parser.add_argument("--out_dir", type=str, default=PREDICTION_DIR,
                        help="Output directory for prediction results")
    return parser.parse_args()

def predict_ca(pos, vel, acc, horizon_steps, dt):
    """Compute Constant Acceleration predicted waypoints over horizon_steps."""
    preds_pos = []
    preds_vel = []
    for k in range(1, horizon_steps + 1):
        t_h = k * dt
        p_h = pos + vel * t_h + 0.5 * acc * (t_h ** 2)
        v_h = vel + acc * t_h
        preds_pos.append(p_h)
        preds_vel.append(v_h)
    return np.array(preds_pos), np.array(preds_vel)

def get_shift_window(t_curr, t_shifts):
    """Classify current time into pre_shift, immediate_post_shift, or recovery window."""
    if not t_shifts:
        return "nominal"
    for t_s in t_shifts:
        if (t_s - 1.0) <= t_curr < t_s:
            return "pre_shift"
        elif t_s <= t_curr < (t_s + 0.5):
            return "immediate_post_shift"
        elif (t_s + 0.5) <= t_curr <= (t_s + 2.0):
            return "recovery"
    return "nominal"

def run_trajectory_prediction(filepath, engagement_seed):
    filename = os.path.basename(filepath)
    trajectory_id = filename.replace(".csv", "")
    family = "unknown"
    for fam in FAMILIES:
        if f"q2c_{fam}_" in filename:
            family = fam
            break

    df = pd.read_csv(filepath)
    t_arr = df['time'].values
    n_frames = len(t_arr)

    # Detect shift times if phase changes
    t_shifts = []
    if 'phase' in df.columns:
        phases = df['phase'].values
        shift_indices = np.where(np.diff(phases) != 0)[0]
        t_shifts = [t_arr[idx + 1] for idx in shift_indices]

    # Initialize EKF
    ekf_cfg = EstimatorConfig()
    ekf = EKFTargetEstimator(ekf_cfg)

    # Setup initial EKF state from first ground truth
    p0 = df[['pos_x', 'pos_y', 'pos_z']].iloc[0].values
    v0 = df[['vel_x', 'vel_y', 'vel_z']].iloc[0].values
    ekf.initialize(np.concatenate([p0, v0]))

    # Seed RNG for noise / dropout
    rng = np.random.RandomState(engagement_seed)

    # Initialize NARX Net (W=10, H=20 steps)
    W = 10
    N = 20
    d_in = W * 12
    d_out = N * 6
    narx_net = _NARXNet(d_in, 64, 64, d_out, lr=1e-3, seed=engagement_seed)

    history = []
    issued_records = []
    
    # Validation queue for prequential loss / trust calculation
    pending_predictions = []
    
    ema_narx_loss = None
    ema_ca_loss = None
    ema_beta = 0.9
    trust = 0.0

    for step in range(n_frames):
        t_curr = t_arr[step]
        true_pos = df[['pos_x', 'pos_y', 'pos_z']].iloc[step].values
        true_vel = df[['vel_x', 'vel_y', 'vel_z']].iloc[step].values
        true_acc = df[['acc_x', 'acc_y', 'acc_z']].iloc[step].values
        true_jerk = df[['jerk_x', 'jerk_y', 'jerk_z']].iloc[step].values if 'jerk_x' in df.columns else np.zeros(3)
        phase_val = int(df['phase'].iloc[step]) if 'phase' in df.columns else 0
        shift_win = get_shift_window(t_curr, t_shifts)

        # Synthetic noisy measurement with dropout
        meas_pos = true_pos + rng.normal(0.0, 0.05, size=3)
        meas_vel = true_vel + rng.normal(0.0, 0.02, size=3)
        has_dropout = rng.rand() < 0.02

        # EKF predict + update
        ekf.predict(DT)
        if not has_dropout:
            ekf.update(np.concatenate([meas_pos, meas_vel]))
        
        ekf_state = ekf.get_estimate()
        est_pos = ekf_state[0:3]
        est_vel = ekf_state[3:6]
        est_acc = ekf_state[6:9]
        est_jerk = ekf_state[9:12]

        state_12d = np.concatenate([est_pos, est_vel, est_acc, est_jerk])
        history.append(state_12d)

        # Evaluate pending predictions whose target_step == step
        matured = [p for p in pending_predictions if p['target_step'] == step]
        pending_predictions = [p for p in pending_predictions if p['target_step'] > step]

        for p in matured:
            err_pos = np.linalg.norm(p['pred_pos'] - true_pos)
            err_vel = np.linalg.norm(p['pred_vel'] - true_vel)

            # Record final record
            rec = {
                'trajectory_id': trajectory_id,
                'family': family,
                'engagement_seed': engagement_seed,
                'issue_step': p['issue_step'],
                'issue_time_s': round(p['issue_time_s'], 4),
                'target_step': step,
                'target_time_s': round(t_curr, 4),
                'horizon_step': p['horizon_step'],
                'horizon_s': round(p['horizon_s'], 2),
                'predictor': p['predictor'],
                'pred_x': p['pred_pos'][0],
                'pred_y': p['pred_pos'][1],
                'pred_z': p['pred_pos'][2],
                'true_x': true_pos[0],
                'true_y': true_pos[1],
                'true_z': true_pos[2],
                'position_error_m': round(err_pos, 4),
                'velocity_error_mps': round(err_vel, 4),
                'trust': round(trust, 4),
                'prequential_loss': round(ema_narx_loss if ema_narx_loss else 0.0, 5),
                'model_checksum_at_issue': p['checksum'],
                'phase': phase_val,
                'shift_window': shift_win
            }
            issued_records.append(rec)

            # Update prequential loss for 10-step horizon (0.2s) predictions
            if p['horizon_step'] == 10 and p['predictor'] == 'narx_raw_predictor':
                loss_narx = (err_pos / 5.0) ** 2
                ca_err_pos = np.linalg.norm(p['ca_pos'] - true_pos)
                loss_ca = (ca_err_pos / 5.0) ** 2

                if ema_narx_loss is None:
                    ema_narx_loss = loss_narx
                    ema_ca_loss = loss_ca
                else:
                    ema_narx_loss = ema_beta * ema_narx_loss + (1.0 - ema_beta) * loss_narx
                    ema_ca_loss = ema_beta * ema_ca_loss + (1.0 - ema_beta) * loss_ca

                # Prequential trust ratio
                if ema_ca_loss > 1e-6:
                    raw_trust = np.clip(1.0 - (ema_narx_loss / (ema_ca_loss + 1e-6)), 0.0, 1.0)
                    trust = float(raw_trust)
                else:
                    trust = 0.0

        # Online training at 5 Hz (every 10 steps) if history is sufficient
        if len(history) >= (W + N) and (step % 10 == 0):
            # Encode input W steps
            ref_pos = history[step - N][0:3]
            in_states = history[step - N - W : step - N]
            
            # Target N steps
            out_target = []
            for k_tgt in range(step - N, step):
                st = history[k_tgt]
                dp = (st[0:3] - ref_pos) / 5.0  # residual scale
                dv = st[3:6] / 2.0
                out_target.append(np.concatenate([dp, dv]))
            y_target = np.concatenate(out_target)

            # Prepare input
            parts = []
            for st in in_states:
                dp = (st[0:3] - ref_pos) / 100.0
                dv = st[3:6] / 20.0
                da = st[6:9] / 15.0
                dj = st[9:12] / 50.0
                parts.append(np.concatenate([dp, dv, da, dj]))
            x_in = np.concatenate(parts)

            # Single gradient step
            narx_net.train_step(x_in, y_target)

        # Issue predictions if history >= W
        if len(history) >= W:
            ref_pos = est_pos
            in_states = history[-W:]
            parts = []
            for st in in_states:
                dp = (st[0:3] - ref_pos) / 100.0
                dv = st[3:6] / 20.0
                da = st[6:9] / 15.0
                dj = st[9:12] / 50.0
                parts.append(np.concatenate([dp, dv, da, dj]))
            x_in = np.concatenate(parts)

            # Model checksum
            checksum = str(hash(narx_net.W1.tobytes()[:16]))

            # Raw NARX output (N x 6)
            narx_out = narx_net.forward(x_in).reshape((N, 6))

            # CA rollout (N x 3 pos, N x 3 vel)
            ca_pos_all, ca_vel_all = predict_ca(est_pos, est_vel, est_acc, N, DT)

            # Unscale NARX predictions
            narx_pos_all = ca_pos_all + narx_out[:, 0:3] * 5.0
            narx_vel_all = ca_vel_all + narx_out[:, 3:6] * 2.0

            # Gated predictions
            gated_pos_all = (1.0 - trust) * ca_pos_all + trust * narx_pos_all
            gated_vel_all = (1.0 - trust) * ca_vel_all + trust * narx_vel_all

            # Save prediction targets for required horizons
            for h_s, h_time in zip(HORIZON_STEPS, HORIZON_TIMES):
                tgt_step = step + h_s
                if tgt_step < n_frames:
                    h_idx = h_s - 1
                    # CA
                    pending_predictions.append({
                        'issue_step': step, 'issue_time_s': t_curr, 'target_step': tgt_step,
                        'horizon_step': h_s, 'horizon_s': h_time, 'predictor': 'ca_predictor',
                        'pred_pos': ca_pos_all[h_idx], 'pred_vel': ca_vel_all[h_idx],
                        'ca_pos': ca_pos_all[h_idx], 'ca_vel': ca_vel_all[h_idx],
                        'checksum': checksum
                    })
                    # Raw NARX
                    pending_predictions.append({
                        'issue_step': step, 'issue_time_s': t_curr, 'target_step': tgt_step,
                        'horizon_step': h_s, 'horizon_s': h_time, 'predictor': 'narx_raw_predictor',
                        'pred_pos': narx_pos_all[h_idx], 'pred_vel': narx_vel_all[h_idx],
                        'ca_pos': ca_pos_all[h_idx], 'ca_vel': ca_vel_all[h_idx],
                        'checksum': checksum
                    })
                    # Gated NARX
                    pending_predictions.append({
                        'issue_step': step, 'issue_time_s': t_curr, 'target_step': tgt_step,
                        'horizon_step': h_s, 'horizon_s': h_time, 'predictor': 'narx_gated_predictor',
                        'pred_pos': gated_pos_all[h_idx], 'pred_vel': gated_vel_all[h_idx],
                        'ca_pos': ca_pos_all[h_idx], 'ca_vel': ca_vel_all[h_idx],
                        'checksum': checksum
                    })

    return issued_records

def main():
    args = parse_args()
    print(f"=== Phase 3: Prediction-Only Benchmark ===")
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.manifest, 'r') as f:
        rel_paths = [line.strip() for line in f if line.strip()]

    csv_files = [os.path.join(os.path.dirname(os.path.dirname(args.manifest)), os.path.basename(r)) if not os.path.exists(r)
                 else r for r in rel_paths]
    # Correct resolution of full CSV paths
    csv_files = [os.path.join(DATASET_CSV_DIR, os.path.basename(r)) for r in rel_paths]

    print(f"Loaded {len(csv_files)} trajectories from manifest {args.manifest}")

    all_records = []
    for filepath in csv_files:
        for seed in args.engagement_seeds:
            recs = run_trajectory_prediction(filepath, seed)
            all_records.extend(recs)

    df_rec = pd.DataFrame(all_records)
    records_csv = os.path.join(args.out_dir, "prediction_records.csv")
    df_rec.to_csv(records_csv, index=False)
    print(f"Saved {len(df_rec)} prediction records to {records_csv}")

    # Generate summary CSVs
    # 1. Summary by predictor & horizon
    df_summary = df_rec.groupby(['predictor', 'horizon_s']).agg(
        pos_rmse=('position_error_m', lambda x: np.sqrt(np.mean(x**2))),
        pos_mae=('position_error_m', 'mean'),
        pos_median=('position_error_m', 'median'),
        pos_p90=('position_error_m', lambda x: np.percentile(x, 90)),
        pos_p95=('position_error_m', lambda x: np.percentile(x, 95)),
        pos_max=('position_error_m', 'max'),
        vel_rmse=('velocity_error_mps', lambda x: np.sqrt(np.mean(x**2)))
    ).reset_index()
    summary_csv = os.path.join(args.out_dir, "prediction_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved summary to {summary_csv}")

    # 2. By Family
    df_fam = df_rec.groupby(['family', 'predictor', 'horizon_s']).agg(
        pos_rmse=('position_error_m', lambda x: np.sqrt(np.mean(x**2))),
        pos_median=('position_error_m', 'median'),
        vel_rmse=('velocity_error_mps', lambda x: np.sqrt(np.mean(x**2)))
    ).reset_index()
    df_fam.to_csv(os.path.join(args.out_dir, "prediction_by_family.csv"), index=False)

    # 3. By Shift Window
    df_shift = df_rec.groupby(['shift_window', 'predictor', 'horizon_s']).agg(
        pos_rmse=('position_error_m', lambda x: np.sqrt(np.mean(x**2))),
        pos_median=('position_error_m', 'median')
    ).reset_index()
    df_shift.to_csv(os.path.join(args.out_dir, "prediction_by_shift_window.csv"), index=False)

    # 4. By Trust Bin
    df_rec['trust_bin'] = pd.cut(df_rec['trust'], bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    df_trust = df_rec.groupby(['trust_bin', 'predictor']).agg(
        pos_rmse=('position_error_m', lambda x: np.sqrt(np.mean(x**2))),
        sample_count=('position_error_m', 'count')
    ).reset_index()
    df_trust.to_csv(os.path.join(args.out_dir, "prediction_by_trust_bin.csv"), index=False)

    # Generate 5 Publication Figures
    print("Generating prediction benchmark plots...")

    # Fig 1: Prediction Error by Horizon
    fig, ax = plt.subplots(figsize=(8, 5))
    for pred in ['ca_predictor', 'narx_raw_predictor', 'narx_gated_predictor']:
        sub = df_summary[df_summary['predictor'] == pred]
        ax.plot(sub['horizon_s'], sub['pos_rmse'], marker='o', linewidth=2, label=pred)
    ax.set_title("Position RMSE vs Prediction Horizon")
    ax.set_xlabel("Horizon (s)")
    ax.set_ylabel("Position RMSE (m)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "prediction_error_by_horizon.png"), dpi=150)
    plt.close()

    # Fig 2: Prediction Error by Family at H=0.2s
    fig, ax = plt.subplots(figsize=(10, 5))
    sub_fam = df_fam[df_fam['horizon_s'] == 0.20]
    pivot_fam = sub_fam.pivot(index='family', columns='predictor', values='pos_rmse')
    pivot_fam.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title("Position RMSE by Trajectory Family (Horizon = 0.20s)")
    ax.set_ylabel("Position RMSE (m)")
    ax.grid(True, axis='y')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "prediction_error_by_family.png"), dpi=150)
    plt.close()

    # Fig 3: Error Around Shifts
    fig, ax = plt.subplots(figsize=(8, 5))
    sub_shift = df_shift[df_shift['horizon_s'] == 0.20]
    pivot_shift = sub_shift.pivot(index='shift_window', columns='predictor', values='pos_rmse')
    pivot_shift.plot(kind='bar', ax=ax, width=0.8)
    ax.set_title("Prediction Error Around Maneuver Shifts (Horizon = 0.20s)")
    ax.set_ylabel("Position RMSE (m)")
    ax.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "prediction_error_around_shifts.png"), dpi=150)
    plt.close()

    # Fig 4: Trust vs Realized Error
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df_rec['trust'], df_rec['position_error_m'], alpha=0.1, s=5, color='purple')
    ax.set_title("Prequential Trust vs Realized Position Error")
    ax.set_xlabel("Prequential Trust T_k")
    ax.set_ylabel("Position Error (m)")
    ax.set_ylim(0, 5)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "trust_vs_realized_error.png"), dpi=150)
    plt.close()

    # Fig 5: Risk Coverage Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    trust_thresholds = np.linspace(0, 1, 50)
    coverages = []
    gated_errors = []
    sub_h2 = df_rec[df_rec['horizon_s'] == 0.20]

    for th in trust_thresholds:
        accepted = sub_h2[sub_h2['trust'] >= th]
        cov = len(accepted) / len(sub_h2) if len(sub_h2) > 0 else 0
        err = accepted['position_error_m'].mean() if len(accepted) > 0 else 0
        coverages.append(cov)
        gated_errors.append(err)

    ax.plot(coverages, gated_errors, color='darkgreen', linewidth=2, marker='s')
    ax.set_title("Risk-Coverage Curve (Horizon = 0.20s)")
    ax.set_xlabel("Coverage (Fraction of Accepted Predictions)")
    ax.set_ylabel("Mean Position Error (m)")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "risk_coverage_curve.png"), dpi=150)
    plt.close()

    print("=== Phase 3 Evaluation Complete! ===")

if __name__ == "__main__":
    main()
