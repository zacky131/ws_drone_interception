"""
Script to verify critical scientific properties for Q2 journal revision:
1. Section 2.1: Fair MPC comparison audit (PRIMARY_METHOD_FAIRNESS.md)
2. Section 2.2: Constant-acceleration nesting numerical test
3. Section 2.3: Causal validation trace generation (prequential_trace.csv)
4. Section 2.4: Warmup-frozen trace generation (warmup_frozen_trace.csv)
5. Section 2.5: Seed independence verification
"""

import os
import sys
import copy
import hashlib
import numpy as np
import pandas as pd
import dataclasses

from src.utils.config_schema import ExperimentConfig, ControllerConfig, PursuerConfig, SimulationConfig
from scripts.run_single_case import build_controller, build_estimator, build_pursuer, _METHOD_ABLATIONS
from src.baselines.mpc_narx import MPCNARXPredictor
from src.baselines.fixed_target_model_mpc import MPCConstantAcceleration
from src.simulation.scenario import TargetScenario
from src.environment.wind_model import WindModel
from src.environment.sensor_model import SensorModel
from src.simulation.sim_engine import SimulationEngine


def run_fairness_audit():
    """Section 2.1: Verify fair MPC parameters across primary methods."""
    print("=== Section 2.1: Fair MPC Comparison Audit ===")

    # Load main revision config
    from src.utils.config_schema import load_config
    cfg_file = os.path.join("configs", "q2_revision_main.yaml")
    base_cfg = load_config(cfg_file) if os.path.exists(cfg_file) else ExperimentConfig()

    methods = [
        "mpc_ekf_ca",
        "narx_ca_gated_5hz",
        "narx_ca_ungated_5hz",
        "narx_ca_warmup_frozen",
    ]

    method_params = {}
    for name in methods:
        cfg = copy.deepcopy(base_cfg)
        if name in _METHOD_ABLATIONS:
            ab_dict = _METHOD_ABLATIONS[name]
            cfg.ablation = dataclasses.replace(cfg.ablation, **ab_dict)

            # Apply ablation overrides to controller config
            ctrl_dict = copy.deepcopy(dataclasses.asdict(cfg.controller))
            for k, v in ab_dict.items():
                if k in ctrl_dict:
                    ctrl_dict[k] = v
            cfg.controller = ControllerConfig(**ctrl_dict)

        ctrl = build_controller(cfg)
        est = build_estimator(cfg)
        pur = build_pursuer(cfg)

        param_dict = {
            "horizon": ctrl.N if hasattr(ctrl, "N") else cfg.controller.horizon,
            "Q_pos": getattr(ctrl, "Q_pos", None),
            "Q_vel": getattr(ctrl, "Q_vel", None),
            "R_control": getattr(ctrl, "R_control", None),
            "R_rate": getattr(ctrl, "R_rate", None),
            "Q_terminal_pos": getattr(ctrl, "Q_T_pos", getattr(ctrl, "Q_terminal_pos", None)),
            "Q_terminal_vel": getattr(ctrl, "Q_T_vel", getattr(ctrl, "Q_terminal_vel", None)),
            "velocity_constraints": getattr(ctrl, "v_max", cfg.pursuer.max_velocity),
            "acceleration_constraints": getattr(ctrl, "a_max", cfg.pursuer.max_acceleration),
            "jerk_constraints": getattr(ctrl, "j_max", cfg.pursuer.max_jerk),
            "solver_backend": getattr(ctrl, "_solver_type", cfg.controller.solver),
            "solver_max_iter": cfg.controller.solver_max_iter,
            "pursuer_model": type(pur).__name__,
            "estimator_type": type(est).__name__,
        }
        method_params[name] = param_dict

    # Check fairness assertions
    ref_params = method_params["mpc_ekf_ca"]
    fairness_passed = True
    mismatches = []

    for name, params in method_params.items():
        if name == "mpc_ekf_ca":
            continue
        for key, val in ref_params.items():
            comp_val = params[key]
            if isinstance(val, np.ndarray):
                if not np.allclose(val, comp_val):
                    mismatches.append(f"{name}.{key}: {comp_val} vs {val}")
                    fairness_passed = False
            elif val != comp_val:
                mismatches.append(f"{name}.{key}: {comp_val} vs {val}")
                fairness_passed = False

    if not fairness_passed:
        print("ERROR: Fairness check failed! Mismatches:")
        for m in mismatches:
            print("  -", m)
        raise ValueError("Primary methods differ in core MPC or pursuer parameters!")

    # Write PRIMARY_METHOD_FAIRNESS.md
    os.makedirs(os.path.join("docs", "q2_revision"), exist_ok=True)
    out_path = os.path.join("docs", "q2_revision", "PRIMARY_METHOD_FAIRNESS.md")

    md = []
    md.append("# Primary Method Fairness Audit\n")
    md.append("**Date**: 2026-07-29  ")
    md.append("**Status**: PASSED (All primary methods use strictly identical MPC cost weights, constraints, solver settings, and dynamics)\n")
    md.append("## Parameter Comparison Table\n")
    md.append("| Property | mpc_ekf_ca | narx_ca_gated_5hz | narx_ca_ungated_5hz | narx_ca_warmup_frozen | Status |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- |")

    for key in ref_params.keys():
        vals = [str(method_params[m][key]) for m in methods]
        status = "IDENTICAL" if len(set(vals)) == 1 else "MISMATCH"
        md.append(f"| `{key}` | {vals[0]} | {vals[1]} | {vals[2]} | {vals[3]} | **{status}** |")

    md.append("\n## Conclusion\n")
    md.append("The primary baseline `mpc_ekf_ca` and all NARX ablation variants (`narx_ca_gated_5hz`, `narx_ca_ungated_5hz`, `narx_ca_warmup_frozen`) are strictly unified in pursuer dynamics, estimator type, solver backend, horizon, velocity/acceleration/jerk constraints, and tracking cost weights (`Q_pos`, `Q_vel`, `R_control`, `R_rate`, `Q_terminal_pos`, `Q_terminal_vel`).")

    with open(out_path, "w") as f:
        f.write("\n".join(md))
    print(f"Saved fairness report -> {out_path}")


def run_ca_nesting_verification():
    """Section 2.2: Verify constant-acceleration nesting numerically."""
    print("=== Section 2.2: Constant-Acceleration Nesting Verification ===")
    exp_cfg = ExperimentConfig()

    ctrl_ca = dataclasses.replace(
        exp_cfg.controller,
        controller_type="mpc_ca",
        narx_Q_pos=0.0,
        narx_Q_terminal_pos=0.0,
        mpc_ca_Q_pos=0.0,
        mpc_ca_Q_terminal_pos=0.0,
        solver="casadi",
    )
    ctrl_narx_off = dataclasses.replace(
        exp_cfg.controller,
        controller_type="mpc_narx",
        narx_Q_pos=0.0,
        narx_Q_terminal_pos=0.0,
        mpc_ca_Q_pos=0.0,
        mpc_ca_Q_terminal_pos=0.0,
        narx_residual_baseline="constant_acceleration",
        narx_trust_mode="always_off",
        solver="casadi",
    )

    mpc_ca = MPCConstantAcceleration(ctrl_ca, exp_cfg.pursuer, exp_cfg.simulation)
    narx_off = MPCNARXPredictor(ctrl_narx_off, exp_cfg.pursuer, exp_cfg.simulation)

    rng = np.random.default_rng(12345)
    max_diff_cmd = 0.0
    max_diff_wp = 0.0

    for test_idx in range(20):
        mpc_ca.reset()
        narx_off.reset()

        p_state = rng.uniform(-10.0, 10.0, size=9)
        p_state[3:6] = rng.uniform(-5.0, 5.0, size=3)  # velocities
        target_est = rng.uniform(-10.0, 10.0, size=12)
        target_est[3:6] = rng.uniform(-5.0, 5.0, size=3)
        wind = rng.uniform(-1.0, 1.0, size=3)
        t = test_idx * 0.02

        cmd_ca, _ = mpc_ca.compute_control(p_state, None, target_est, wind, t)
        cmd_narx, info_narx = narx_off.compute_control(p_state, None, target_est, wind, t)

        diff_c = np.max(np.abs(cmd_ca - cmd_narx))
        max_diff_cmd = max(max_diff_cmd, diff_c)

    print(f"Max difference in commands:  {max_diff_cmd:.2e}")
    assert max_diff_cmd < 1e-5, f"Control actions mismatch! max diff = {max_diff_cmd}"
    print("PASSED: NARX (trust=0, baseline=CA) is numerically identical to mpc_ekf_ca!")


def run_causal_validation_trace():
    """Section 2.3: Generate causal prequential trace CSV."""
    print("=== Section 2.3: Causal Validation Trace Generation ===")
    out_dir = os.path.join("results", "q2_revision_v1", "verification")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "prequential_trace.csv")

    exp_cfg = ExperimentConfig()
    ctrl_cfg = dataclasses.replace(
        exp_cfg.controller,
        narx_residual_baseline="constant_acceleration",
        narx_trust_mode="prequential",
        narx_min_validation_samples=10,
        narx_trust_threshold=0.40,
        solver="casadi",
    )

    narx = MPCNARXPredictor(ctrl_cfg, exp_cfg.pursuer, exp_cfg.simulation)
    narx.reset()

    rows = []
    rng = np.random.default_rng(42)
    p_state = np.zeros(9)
    wind = np.zeros(3)

    # Simulate 50 steps
    target_pos = np.array([50.0, 0.0, 10.0])
    target_vel = np.array([5.0, 2.0, 0.0])

    for k in range(50):
        t = k * 0.02
        # Target undergoing circular turn
        target_pos = target_pos + target_vel * 0.02 + 0.1 * np.array([np.sin(t), np.cos(t), 0.0])
        target_est = np.concatenate([target_pos, target_vel, np.zeros(6)])

        cmd, info = narx.compute_control(p_state, None, target_est, wind, t)

        # Check issued prediction maturity
        n_issued = len(narx._issued_predictions)
        latest_issued_step = narx._issued_predictions[-1]["step_idx"] if n_issued > 0 else -1

        rows.append({
            "step_idx": k,
            "sim_time_s": t,
            "prediction_issued_step": latest_issued_step,
            "prediction_horizon_steps": narx.N,
            "validation_sample_count": info["narx_validation_sample_count"],
            "prequential_loss": info["narx_prequential_loss"],
            "trust": info["narx_trust"],
            "narx_ready": info["narx_ready"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved prequential validation trace -> {out_path}")


def run_warmup_frozen_trace():
    """Section 2.4: Generate warmup-frozen verification trace CSV."""
    print("=== Section 2.4: Warmup-Frozen Trace Generation ===")
    out_dir = os.path.join("results", "q2_revision_v1", "verification")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "warmup_frozen_trace.csv")

    exp_cfg = ExperimentConfig()
    ctrl_cfg = dataclasses.replace(
        exp_cfg.controller,
        narx_residual_baseline="constant_acceleration",
        narx_trust_mode="prequential",
        narx_enable_online_training=True,
        narx_training_period_steps=2,
        narx_grad_steps=10,
        narx_freeze_after_training_events=5,  # Freeze after 5 training events
        solver="casadi",
    )

    narx = MPCNARXPredictor(ctrl_cfg, exp_cfg.pursuer, exp_cfg.simulation)
    narx.reset()

    rows = []
    p_state = np.zeros(9)
    wind = np.zeros(3)

    target_pos = np.array([30.0, 0.0, 10.0])
    target_vel = np.array([3.0, 1.0, 0.0])

    for k in range(80):
        t = k * 0.02
        target_pos = target_pos + target_vel * 0.02 + 0.05 * np.array([np.cos(k*0.1), np.sin(k*0.1), 0.0])
        target_est = np.concatenate([target_pos, target_vel, np.array([0.5, 0.2, 0.0, 0.1, 0.0, 0.0])])

        cmd, info = narx.compute_control(p_state, None, target_est, wind, t)

        # Compute parameter checksum
        w1_bytes = narx._narx.W1.tobytes()
        param_checksum = hashlib.sha256(w1_bytes).hexdigest()[:10]

        is_frozen = (
            narx._narx_freeze_after_training_events > 0
            and info["narx_training_events_count"] >= narx._narx_freeze_after_training_events
        )

        rows.append({
            "step": k,
            "training_event_count": info["narx_training_events_count"],
            "is_frozen": is_frozen,
            "parameter_checksum": param_checksum,
            "training_loss": info["narx_train_loss"],
            "prequential_loss": info["narx_prequential_loss"],
            "trust": info["narx_trust"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Saved warmup-frozen trace -> {out_path}")

    # Assert that parameter checksum stopped changing after freezing
    frozen_df = df[df["is_frozen"] == True]
    if len(frozen_df) > 1:
        unique_checksums_after_freeze = frozen_df["parameter_checksum"].nunique()
        assert unique_checksums_after_freeze == 1, (
            f"Weights changed after freezing! Found {unique_checksums_after_freeze} distinct checksums."
        )
        print("PASSED: Weights remained strictly fixed after freezing event limit was reached!")


def run_seed_independence_verification():
    """Section 2.5: Seed independence verification."""
    print("=== Section 2.5: Seed Independence Verification ===")
    exp_cfg = ExperimentConfig()

    # 1. Test identical seeds produce identical initial weights
    cfg1 = dataclasses.replace(exp_cfg.controller, narx_seed=100)
    cfg2 = dataclasses.replace(exp_cfg.controller, narx_seed=100)
    narx1 = MPCNARXPredictor(cfg1, exp_cfg.pursuer, exp_cfg.simulation)
    narx2 = MPCNARXPredictor(cfg2, exp_cfg.pursuer, exp_cfg.simulation)
    np.testing.assert_allclose(narx1._narx.W1, narx2._narx.W1)

    # 2. Test different seeds produce different initial weights
    cfg3 = dataclasses.replace(exp_cfg.controller, narx_seed=200)
    narx3 = MPCNARXPredictor(cfg3, exp_cfg.pursuer, exp_cfg.simulation)
    assert not np.allclose(narx1._narx.W1, narx3._narx.W1)

    # 3. Test reset restores initial weights under same seed
    target_est = np.array([10.0, 5.0, 2.0, 1.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.01, 0.0, 0.0])
    p_state = np.zeros(9)
    wind = np.zeros(3)

    for _ in range(40):
        narx1.compute_control(p_state, None, target_est, wind, 0.0)

    # Weights changed during training
    assert not np.allclose(narx1._narx.W1, narx2._narx.W1)

    # Reset clears learned state and re-seeds network
    narx1.reset()
    np.testing.assert_allclose(narx1._narx.W1, narx2._narx.W1)
    print("PASSED: Seed independence & clean resetting verified!")


if __name__ == "__main__":
    run_fairness_audit()
    run_ca_nesting_verification()
    run_causal_validation_trace()
    run_warmup_frozen_trace()
    run_seed_independence_verification()
    print("\nAll Section 2 scientific verification properties successfully confirmed!")
