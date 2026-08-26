#!/usr/bin/env python
"""
Run a single interception simulation, optionally comparing multiple methods.

Usage (single method):
    python scripts/run_single_case.py --method baseline_pn baseline_smc mpc_cv mpc_ca mpc_ekf_narx  

Usage (multi-method comparison):
    python3 scripts/run_single_case.py --method baseline_pn mpc_cv mpc_ca mpc_ekf_narx

All methods run against the same random scenario (same --seed) so their
trajectories can be meaningfully overlaid on a single comparison plot.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time as _time

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from src.utils.config_schema import ExperimentConfig, load_config
from src.dynamics.point_mass_pursuer import PointMassPursuer
from src.dynamics.quadrotor_outer_loop import QuadrotorOuterLoopPursuer
try:
    from src.dynamics.quadrotor_6dof import Quadrotor6DOFPursuer
except ImportError:
    pass
from src.estimation.ekf_target_estimator import EKFTargetEstimator
from src.estimation.rls_baseline_estimator import RLSBaselineEstimator
from src.control.adaptive_interception_mpc import AdaptiveInterceptionMPC
from src.baselines.proportional_navigation import ProportionalNavigation
from src.baselines.sliding_mode_guidance import SlidingModeGuidance
from src.baselines.standard_mpc import StandardMPC
from src.baselines.rls_adaptive_mpc import RLSAdaptiveMPC
try:
    from src.baselines.fixed_target_model_mpc import FixedTargetModelMPC
except ImportError: pass
try:
    from src.baselines.constant_velocity_mpc import ConstantVelocityMPC
except ImportError: pass
try:
    from src.baselines.mpc_cv import MPCConstantVelocity
except ImportError: pass
try:
    from src.baselines.mpc_ca import MPCConstantAcceleration
except ImportError: pass
try:
    from src.baselines.mpc_rls_linear import MPCRLSLinearPrediction
except ImportError: pass
try:
    from src.baselines.mpc_narx import MPCNARXPredictor
except ImportError: pass
try:
    from src.estimation.no_estimator import NoEstimator
except ImportError:
    pass
from src.environment.wind_model import WindModel
from src.environment.sensor_model import SensorModel
from src.simulation.scenario import TargetScenario
from src.simulation.scenarios import create_scenario
from src.simulation.sim_engine import SimulationEngine
from src.simulation.logger import SimulationLogger
from src.evaluation.metrics import compute_metrics


# ── Named-method / ablation registry ──────────────────────────────────────────

#: Maps method name → (ablation overrides dict).  Keys match AblationConfig fields.
_METHOD_ABLATIONS: dict = {
    "proposed_full": {},   # use all defaults (full method)
    "ablation_no_ekf": {
        "ablation_name": "ablation_no_ekf",
        "use_ekf_estimator": False,
    },
    "ablation_ideal_pursuer": {
        "ablation_name": "ablation_ideal_pursuer",
        "use_realistic_pursuer_model": False,
    },
    "ablation_no_disturbance": {
        "ablation_name": "ablation_no_disturbance",
        "use_disturbances": False,
    },
    "ablation_fixed_target_model": {
        "ablation_name": "ablation_fixed_target_model",
        "target_prediction_model": "constant_acceleration",
    },
    # Legacy / baseline names (kept for backward compat)
    "ekf_adaptive_mpc": {},
    "rls_adaptive_mpc": {
        "ablation_name": "rls_adaptive_mpc",
        "use_ekf_estimator": False,
    },
    "standard_mpc": {
        "ablation_name": "standard_mpc",
        "target_prediction_model": "constant_velocity",
    },
    "baseline_pn": {
        "ablation_name": "baseline_pn",
        "use_realistic_pursuer_model": False,
    },
    "baseline_smc": {
        "ablation_name": "baseline_smc",
        "use_realistic_pursuer_model": False,
    },
    "baseline_standard_mpc": {
        "ablation_name": "baseline_standard_mpc",
        "target_prediction_model": "constant_velocity",
    },
    "baseline_rls_adaptive_mpc": {
        "ablation_name": "baseline_rls_adaptive_mpc",
        # NOTE: use_ekf_estimator is intentionally NOT set to False here.
        # RLSAdaptiveMPC runs its own internal RLS and ignores the external
        # estimator output.  Keeping EKF as external estimator means the
        # logged rmse_pos/rmse_vel metrics reflect EKF quality (consistent
        # with proposed_full), not the internal RLS noise.
    },
    "pn": {"ablation_name": "pn", "use_realistic_pursuer_model": False},
    "smc": {"ablation_name": "smc", "use_realistic_pursuer_model": False},
    # ── New baselines ──────────────────────────────────────────────────────
    "constant_velocity_mpc": {
        "ablation_name": "constant_velocity_mpc",
    },
    "baseline_constant_velocity_mpc": {
        "ablation_name": "baseline_constant_velocity_mpc",
    },
    # ── New ablation ───────────────────────────────────────────────────────
    "ablation_no_rate_constraint": {
        "ablation_name": "ablation_no_rate_constraint",
        "use_rate_constraints": False,
    },
    # ── Controlled ablation variants for Table II ──────────────────────────
    "mpc_cv": {
        "ablation_name": "mpc_cv",
    },
    "mpc_ca": {
        "ablation_name": "mpc_ca",
    },
    "mpc_rls_linear": {
        "ablation_name": "mpc_rls_linear",
    },
    # ── Hybrid learned-prediction variant ─────────────────────────────────
    "mpc_ekf_narx": {
        "ablation_name": "mpc_ekf_narx",
    },
    # ── Q2 revision fair comparison baseline names ────────────────────────
    "baseline_pn_6dof": {
        "ablation_name": "baseline_pn_6dof",
        "use_realistic_pursuer_model": True,
    },
    "baseline_smc_6dof": {
        "ablation_name": "baseline_smc_6dof",
        "use_realistic_pursuer_model": True,
    },
    "mpc_ekf_cv": {
        "ablation_name": "mpc_ekf_cv",
        "target_prediction_model": "constant_velocity",
        "use_realistic_pursuer_model": True,
    },
    "mpc_ekf_ca": {
        "ablation_name": "mpc_ekf_ca",
        "target_prediction_model": "constant_acceleration",
        "use_realistic_pursuer_model": True,
    },
    "mpc_oracle_target": {
        "ablation_name": "mpc_oracle_target",
        "use_realistic_pursuer_model": True,
    },
    "mpc_exact_state_ca": {
        "ablation_name": "mpc_exact_state_ca",
        "use_realistic_pursuer_model": True,
    },
    "camps_rule": {
        "ablation_name": "camps_rule",
        "use_realistic_pursuer_model": True,
    },
    "camps_learned": {
        "ablation_name": "camps_learned",
        "use_realistic_pursuer_model": True,
    },
    "camps_fusion": {
        "ablation_name": "camps_fusion",
        "use_realistic_pursuer_model": True,
    },
    "oracle_fixed_0p4s": {"ablation_name": "oracle_fixed_0p4s", "use_realistic_pursuer_model": True},
    "oracle_fixed_1p0s": {"ablation_name": "oracle_fixed_1p0s", "use_realistic_pursuer_model": True},
    "oracle_fixed_2p0s": {"ablation_name": "oracle_fixed_2p0s", "use_realistic_pursuer_model": True},
    "oracle_nonuniform_3p0s": {"ablation_name": "oracle_nonuniform_3p0s", "use_realistic_pursuer_model": True},
    "exact_state_ca_fixed_0p4s": {"ablation_name": "exact_state_ca_fixed_0p4s", "use_realistic_pursuer_model": True},
    "exact_state_ca_fixed_1p0s": {"ablation_name": "exact_state_ca_fixed_1p0s", "use_realistic_pursuer_model": True},
    "exact_state_ca_fixed_2p0s": {"ablation_name": "exact_state_ca_fixed_2p0s", "use_realistic_pursuer_model": True},
    "exact_state_ca_nonuniform_3p0s": {"ablation_name": "exact_state_ca_nonuniform_3p0s", "use_realistic_pursuer_model": True},
    "ekf_ca_fixed_0p4s": {"ablation_name": "ekf_ca_fixed_0p4s", "use_realistic_pursuer_model": True},
    "ekf_ca_nonuniform_3p0s": {"ablation_name": "ekf_ca_nonuniform_3p0s", "use_realistic_pursuer_model": True},
    "oracle_capture_time_nonuniform_3p0s": {"ablation_name": "oracle_capture_time_nonuniform_3p0s", "use_realistic_pursuer_model": True},
    "exact_state_ca_capture_time_nonuniform_3p0s": {"ablation_name": "exact_state_ca_capture_time_nonuniform_3p0s", "use_realistic_pursuer_model": True},
    "ekf_ca_capture_time_nonuniform_3p0s": {"ablation_name": "ekf_ca_capture_time_nonuniform_3p0s", "use_realistic_pursuer_model": True},
}

# Colour palette for multi-method overlay plots (avoids target-red #d62728)
_METHOD_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#17becf",  # cyan
    "#bcbd22",  # yellow-green
]
_TARGET_COLOR = "#d62728"  # red — reserved for the target trajectory


# ── Pursuer initial-condition helpers (mirrors run_monte_carlo logic) ──────────

def _random_pursuer_position(
    target_pos: np.ndarray,
    mode: str,
    radius_range: list,
    rng: np.random.Generator,
    min_altitude: float = 0.0,
) -> np.ndarray:
    """Place pursuer randomly relative to the target.

    Modes
    -----
    spherical_shell  : full sphere, any elevation (default)
    hemisphere       : upper hemisphere — pursuer ABOVE target
    lower_hemisphere : lower hemisphere — pursuer BELOW target  ← realistic defender
    uniform_box      : random offset in a cube of half-width radius_range[1],
                       rejection-sampled to keep distance >= radius_range[0]
    """
    if mode == "uniform_box":
        half = radius_range[1]
        for _ in range(50):
            offset = rng.uniform(-half, half, size=3)
            if np.linalg.norm(offset) >= radius_range[0]:
                break
        pos = target_pos + offset
        pos[2] = max(min_altitude, pos[2])
        return pos

    r = rng.uniform(radius_range[0], radius_range[1])
    az = rng.uniform(0, 2 * np.pi)
    if mode == "hemisphere":
        el = rng.uniform(0, np.pi / 2)       # upper half: pursuer ABOVE target
    elif mode == "lower_hemisphere":
        el = rng.uniform(-np.pi / 2, 0)     # lower half: pursuer BELOW target
    else:  # spherical_shell (default)
        el = rng.uniform(-np.pi / 2, np.pi / 2)
    offset = np.array([
        r * np.cos(el) * np.cos(az),
        r * np.cos(el) * np.sin(az),
        r * np.sin(el),
    ])
    pos = target_pos + offset
    pos[2] = max(min_altitude, pos[2])
    return pos


def _random_pursuer_velocity(
    mode: str,
    speed_range: list,
    rng: np.random.Generator,
    target_pos: np.ndarray,
    pursuer_pos: np.ndarray,
) -> np.ndarray:
    """Generate a pursuer initial velocity (toward target by default)."""
    speed = rng.uniform(speed_range[0], speed_range[1])
    if mode == "toward_target":
        d = target_pos - pursuer_pos
        dn = np.linalg.norm(d)
        if dn > 1e-6:
            return (d / dn) * speed
    az = rng.uniform(0, 2 * np.pi)
    el = rng.uniform(-np.pi / 6, np.pi / 6)
    return np.array([
        speed * np.cos(el) * np.cos(az),
        speed * np.cos(el) * np.sin(az),
        speed * np.sin(el),
    ])


def apply_ablation_overrides(cfg: ExperimentConfig, method_name: str) -> ExperimentConfig:
    """Apply ablation-flag overrides for *method_name* onto *cfg* in-place.

    Raises ``ValueError`` if *method_name* is not in the registry.
    When ``use_disturbances`` is False the function also zeros wind and noise.
    """
    if method_name not in _METHOD_ABLATIONS:
        raise ValueError(
            f"Unknown method name '{method_name}'. "
            f"Registered names: {sorted(_METHOD_ABLATIONS)}"
        )
    overrides = _METHOD_ABLATIONS[method_name]
    ab = cfg.ablation
    for k, v in overrides.items():
        setattr(ab, k, v)

    # Cascade: master disturbance switch disables sub-features
    if not ab.use_disturbances:
        cfg.wind.enabled = False
        cfg.sensor.position_noise_std = 0.0
        cfg.sensor.velocity_noise_std = 0.0
        cfg.sensor.pursuer_position_noise_std = 0.0
        cfg.sensor.pursuer_velocity_noise_std = 0.0
        cfg.sensor.delay_steps = 0
        cfg.sensor.dropout_probability = 0.0
    else:
        if not ab.use_sensor_delay:
            cfg.sensor.delay_steps = 0
        if not ab.use_sensor_dropout:
            cfg.sensor.dropout_probability = 0.0

    # Rate-constraint ablation: set max_jerk to a very large value so the
    # hard constraint in the MPC NLP is effectively inactive.
    if not ab.use_rate_constraints:
        cfg.pursuer.max_jerk = 1_000_000.0

    return cfg


def build_pursuer(cfg: ExperimentConfig):
    """Return a pursuer model honoring the ablation config.

    When ``use_realistic_pursuer_model`` is False (ablation override) the
    pursuer is forced to point-mass regardless of ``model_type``.
    Otherwise ``model_type`` from the YAML is respected.
    """
    ab = cfg.ablation
    if not ab.use_realistic_pursuer_model:
        return PointMassPursuer(cfg.pursuer)
    # Normal path: use whatever model_type the YAML specifies
    mt = cfg.pursuer.model_type
    if mt == "point_mass":
        return PointMassPursuer(cfg.pursuer)
    if mt == "quadrotor_6dof":
        try:
            return Quadrotor6DOFPursuer(cfg.pursuer)
        except NameError:
            print("Warning: Quadrotor6DOFPursuer not implemented, falling back to outer_loop")
            return QuadrotorOuterLoopPursuer(cfg.pursuer)
    return QuadrotorOuterLoopPursuer(cfg.pursuer)  # "quadrotor_outer_loop" or default


def build_estimator(cfg: ExperimentConfig):
    """Return an estimator honoring the ablation config and YAML estimator_type.

    Priority:
      1. If ``use_ekf_estimator=False`` (ablation override) → force RLS.
      2. Otherwise respect ``estimator_type`` from the YAML config.
    """
    ab = cfg.ablation
    # mpc_cv needs no real estimator — use passthrough stub
    if ab.ablation_name == "mpc_cv":
        return NoEstimator()
    # Ablation override: force RLS when flag is explicitly set to False
    if not ab.use_ekf_estimator:
        return RLSBaselineEstimator(cfg.estimator)
    # Normal path: respect YAML estimator_type
    if cfg.estimator.estimator_type == "rls":
        return RLSBaselineEstimator(cfg.estimator)
    return EKFTargetEstimator(cfg.estimator)


def build_controller(cfg: ExperimentConfig):
    """Return a controller honoring both the ablation config and controller_type."""
    ab = cfg.ablation
    ct = cfg.controller.controller_type

    # Ablation-name overrides: legacy/baseline PN and SMC names must map to
    # their respective controllers regardless of controller_type in the YAML.
    if ab.ablation_name in ("pn", "baseline_pn", "baseline_pn_6dof"):
        return ProportionalNavigation(cfg.controller, cfg.pursuer)
    if ab.ablation_name in ("smc", "baseline_smc", "baseline_smc_6dof"):
        return SlidingModeGuidance(cfg.controller, cfg.pursuer)

    # New baseline: constant_velocity_mpc (always uses EKF, constant-vel prediction)
    if ab.ablation_name in ("constant_velocity_mpc", "baseline_constant_velocity_mpc", "mpc_ekf_cv"):
        return ConstantVelocityMPC(cfg.controller, cfg.pursuer, cfg.simulation)
    if ab.ablation_name == "mpc_ekf_ca":
        return MPCConstantAcceleration(cfg.controller, cfg.pursuer, cfg.simulation)

    # ── Controlled ablation variants (Table II) ───────────────────────────
    # mpc_cv: no estimator, constant-velocity prediction
    if ab.ablation_name == "mpc_cv":
        return MPCConstantVelocity(cfg.controller, cfg.pursuer, cfg.simulation)
    # mpc_ca: EKF acceleration used directly (no internal RLS; jerk dropped from prediction)
    # Apply per-method Q weights so CA baseline is less aggressive than NARX.
    if ab.ablation_name == "mpc_ca":
        q_pos = cfg.controller.mpc_ca_Q_pos or cfg.controller.Q_pos
        q_term = cfg.controller.mpc_ca_Q_terminal_pos or cfg.controller.Q_terminal_pos
        ctrl_ca = dataclasses.replace(cfg.controller, Q_pos=q_pos, Q_terminal_pos=q_term)
        return MPCConstantAcceleration(ctrl_ca, cfg.pursuer, cfg.simulation)
    # mpc_rls_linear: full RLS (acc+jerk estimated) but jerk dropped from NLP
    if ab.ablation_name == "mpc_rls_linear":
        return MPCRLSLinearPrediction(cfg.controller, cfg.pursuer, cfg.simulation, cfg.estimator)
    # baseline_rls_adaptive_mpc: composition wrapper — RLS internal + cubic AdaptiveInterceptionMPC.
    # RLSAdaptiveMPC runs its own RLS and ignores the external estimator output,
    # reproducing the original submission architecture (ISR ≈ 99.57 %).
    if ab.ablation_name == "baseline_rls_adaptive_mpc":
        return RLSAdaptiveMPC(cfg.controller, cfg.pursuer, cfg.simulation, cfg.estimator)
    # mpc_ekf_narx or narx_ variant: EKF-filtered state + online NARX trajectory prediction + tracking MPC.
    if ab.ablation_name in ("mpc_ekf_narx", "mpc_narx") or ab.ablation_name.startswith("narx_"):
        return MPCNARXPredictor(cfg.controller, cfg.pursuer, cfg.simulation)

    # Ablation: fixed target model variant
    if ab.target_prediction_model in ("constant_velocity", "constant_acceleration"):
        # If the user also set controller_type to something other than adaptive
        # that takes precedence; otherwise use FixedTargetModelMPC.
        if ct in ("adaptive_mpc",):
            return FixedTargetModelMPC(
                cfg.controller, cfg.pursuer, cfg.simulation,
                target_prediction_model=ab.target_prediction_model,
            )

    if ab.ablation_name == "mpc_oracle_target" or ct == "mpc_oracle_target":
        from src.prediction.camps.oracle_target_mpc import MPCOracleTarget
        return MPCOracleTarget(cfg.controller, cfg.pursuer, cfg.simulation)
    if ab.ablation_name == "mpc_exact_state_ca" or ct == "mpc_exact_state_ca":
        from src.prediction.camps.exact_state_ca_mpc import MPCExactStateCA
        return MPCExactStateCA(cfg.controller, cfg.pursuer, cfg.simulation)
    if ab.ablation_name in ("camps_rule", "camps_learned", "camps_fusion") or ct.startswith("camps_"):
        from src.prediction.camps.camps_mpc import MPCCAMPSController
        st = ab.ablation_name if ab.ablation_name in ("camps_rule", "camps_learned", "camps_fusion") else ct
        return MPCCAMPSController(cfg.controller, cfg.pursuer, cfg.simulation, selector_type=st)

    HORIZON_VARIANTS = (
        "oracle_fixed_0p4s", "oracle_fixed_1p0s", "oracle_fixed_2p0s", "oracle_nonuniform_3p0s",
        "exact_state_ca_fixed_0p4s", "exact_state_ca_fixed_1p0s", "exact_state_ca_fixed_2p0s", "exact_state_ca_nonuniform_3p0s",
        "ekf_ca_fixed_0p4s", "ekf_ca_nonuniform_3p0s",
        "oracle_capture_time_nonuniform_3p0s", "exact_state_ca_capture_time_nonuniform_3p0s", "ekf_ca_capture_time_nonuniform_3p0s"
    )
    if ab.ablation_name in HORIZON_VARIANTS or ct in HORIZON_VARIANTS:
        from src.control.capture_time_mpc.horizon_controller import HorizonMPCController
        v_name = ab.ablation_name if ab.ablation_name in HORIZON_VARIANTS else ct
        is_cap_opt = "capture_time" in v_name
        return HorizonMPCController(cfg.controller, cfg.pursuer, cfg.simulation, variant_name=v_name, enable_capture_time_opt=is_cap_opt)

    # Standard dispatch by controller_type
    if ct == "adaptive_mpc":
        return AdaptiveInterceptionMPC(cfg.controller, cfg.pursuer, cfg.simulation)
    elif ct in ("mpc_narx", "mpc_ekf_narx"):
        return MPCNARXPredictor(cfg.controller, cfg.pursuer, cfg.simulation)
    elif ct in ("mpc_ca", "mpc_ekf_ca"):
        return MPCConstantAcceleration(cfg.controller, cfg.pursuer, cfg.simulation)
    elif ct == "standard_mpc":
        return StandardMPC(cfg.controller, cfg.pursuer, cfg.simulation)
    elif ct == "constant_velocity_mpc":
        return ConstantVelocityMPC(cfg.controller, cfg.pursuer, cfg.simulation)
    elif ct == "pn":
        return ProportionalNavigation(cfg.controller, cfg.pursuer)
    elif ct == "smc":
        return SlidingModeGuidance(cfg.controller, cfg.pursuer)
    elif ct == "rls_adaptive_mpc":
        return RLSAdaptiveMPC(cfg.controller, cfg.pursuer, cfg.simulation, cfg.estimator)
    elif ct == "fixed_target_model_mpc":
        mode = ab.target_prediction_model
        if mode not in ("constant_velocity", "constant_acceleration"):
            mode = "constant_acceleration"
        return FixedTargetModelMPC(cfg.controller, cfg.pursuer, cfg.simulation,
                                   target_prediction_model=mode)
    else:
        raise ValueError(f"Unknown controller type: {ct}")


def _plot_results(
    results: list,
    cfg_base: "ExperimentConfig",
    out_dir: str,
) -> None:
    """Save a 6-panel comparison figure for one or more methods.

    Panels
    ------
    [0,0] 3D trajectory — pursuer paths + target path; ▲ initial, ■ final
    [0,1] XY top-view trajectory
    [0,2] Altitude (Z) profile vs time
    [1,0] Pursuer–target distance vs time
    [1,1] Control acceleration magnitude vs time
    [1,2] Target position estimation error vs time
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

    fig = plt.figure(figsize=(20, 12))
    ax3d     = fig.add_subplot(2, 3, 1, projection="3d")
    ax_xy    = fig.add_subplot(2, 3, 2)
    ax_z     = fig.add_subplot(2, 3, 3)
    ax_dist  = fig.add_subplot(2, 3, 4)
    ax_ctrl  = fig.add_subplot(2, 3, 5)
    ax_est   = fig.add_subplot(2, 3, 6)

    # ── Target trajectory (once, same for all methods) ──────────────────
    df0 = results[0][1]
    ax3d.plot(df0["t_px"], df0["t_py"], df0["t_pz"],
              color=_TARGET_COLOR, ls="--", lw=1.5, alpha=0.8, label="Target")
    ax3d.scatter(*df0[["t_px", "t_py", "t_pz"]].iloc[0].values,
                 s=200, color=_TARGET_COLOR, marker="^", zorder=6, edgecolors="k", lw=0.5)
    ax3d.scatter(*df0[["t_px", "t_py", "t_pz"]].iloc[-1].values,
                 s=200, color=_TARGET_COLOR, marker="s", zorder=6, edgecolors="k", lw=0.5)

    ax_xy.plot(df0["t_px"], df0["t_py"],
               color=_TARGET_COLOR, ls="--", lw=1.5, label="Target")
    ax_xy.scatter(df0["t_px"].iloc[0], df0["t_py"].iloc[0],
                  s=120, color=_TARGET_COLOR, marker="^", zorder=6, edgecolors="k", lw=0.5)
    ax_xy.scatter(df0["t_px"].iloc[-1], df0["t_py"].iloc[-1],
                  s=120, color=_TARGET_COLOR, marker="s", zorder=6, edgecolors="k", lw=0.5)

    ax_z.plot(df0["time"], df0["t_pz"],
              color=_TARGET_COLOR, ls="--", lw=1.5, label="Target")

    # ── Per-method pursuer trajectories ─────────────────────────────────
    for i, (method, df, metrics, _cfg) in enumerate(results):
        color = _METHOD_COLORS[i % len(_METHOD_COLORS)]
        ok = metrics["success"]
        label = f"{method}  ({'✓' if ok else '✗'} {metrics['intercept_time']:.1f} s)"

        # --- 3D ---
        ax3d.plot(df["p_px"], df["p_py"], df["p_pz"],
                  color=color, lw=2, label=label)
        ax3d.scatter(*df[["p_px", "p_py", "p_pz"]].iloc[0].values,
                     s=150, color=color, marker="^", zorder=6, edgecolors="k", lw=0.5)
        ax3d.scatter(*df[["p_px", "p_py", "p_pz"]].iloc[-1].values,
                     s=150, color=color, marker="s", zorder=6, edgecolors="k", lw=0.5)

        # --- XY ---
        ax_xy.plot(df["p_px"], df["p_py"], color=color, lw=2, label=label)
        ax_xy.scatter(df["p_px"].iloc[0], df["p_py"].iloc[0],
                      s=100, color=color, marker="^", zorder=6, edgecolors="k", lw=0.5)
        ax_xy.scatter(df["p_px"].iloc[-1], df["p_py"].iloc[-1],
                      s=100, color=color, marker="s", zorder=6, edgecolors="k", lw=0.5)

        # --- Z altitude ---
        ax_z.plot(df["time"], df["p_pz"], color=color, lw=2, label=method)

        # --- Distance ---
        ax_dist.plot(df["time"], df["distance"], color=color, lw=2, label=label)

        # --- Control magnitude ---
        cmd_norm = np.sqrt(df["cmd_ax"] ** 2 + df["cmd_ay"] ** 2 + df["cmd_az"] ** 2)
        ax_ctrl.plot(df["time"], cmd_norm, color=color, lw=1.5, label=method)

        # --- Estimation error ---
        if all(c in df.columns for c in ("te_px", "te_py", "te_pz")):
            err_p = np.sqrt(
                (df["t_px"] - df["te_px"]) ** 2
                + (df["t_py"] - df["te_py"]) ** 2
                + (df["t_pz"] - df["te_pz"]) ** 2
            )
            ax_est.plot(df["time"], err_p, color=color, lw=1.5, label=method)

    # ── Decoration ───────────────────────────────────────────────────────
    ax_dist.axhline(
        cfg_base.simulation.success_distance,
        color="green", ls="--", lw=1.0, label="Threshold",
    )

    ax3d.set_xlabel("X [m]"); ax3d.set_ylabel("Y [m]"); ax3d.set_zlabel("Z [m]")
    ax3d.set_title("3D Trajectory  (▲ = initial,  ■ = final)")
    ax3d.legend(fontsize=7, loc="upper right")
    ax3d.grid(True, alpha=0.3)

    ax_xy.set_xlabel("X [m]"); ax_xy.set_ylabel("Y [m]")
    ax_xy.set_title("XY Trajectory (top view)  (▲ initial, ■ final)")
    ax_xy.legend(fontsize=7); ax_xy.grid(True, alpha=0.3); ax_xy.set_aspect("equal")

    ax_z.set_xlabel("Time [s]"); ax_z.set_ylabel("Altitude Z [m]")
    ax_z.set_title("Altitude Profile")
    ax_z.legend(fontsize=7); ax_z.grid(True, alpha=0.3)

    ax_dist.set_xlabel("Time [s]"); ax_dist.set_ylabel("Distance [m]")
    ax_dist.set_title("Pursuer–Target Distance")
    ax_dist.legend(fontsize=7); ax_dist.grid(True, alpha=0.3)

    ax_ctrl.set_xlabel("Time [s]"); ax_ctrl.set_ylabel("‖a_cmd‖ [m/s²]")
    ax_ctrl.set_title("Control Acceleration Magnitude")
    ax_ctrl.legend(fontsize=7); ax_ctrl.grid(True, alpha=0.3)

    ax_est.set_xlabel("Time [s]"); ax_est.set_ylabel("Position error [m]")
    ax_est.set_title("Target Position Estimation Error")
    ax_est.legend(fontsize=7); ax_est.grid(True, alpha=0.3)

    n = len(results)
    suptitle = (
        f"Interception Method Comparison  ({n} methods, seed={results[0][3].ablation.ablation_name})"
        if n > 1
        else f"Single-Run Simulation:  {results[0][0]}"
    )
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.005, "▲ = initial position     ■ = final position",
             ha="center", fontsize=9, style="italic", color="#555555")

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig_path = os.path.join(out_dir, "single_run_plots.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nPlots saved to {fig_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single interception simulation (one or more methods)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "configs", "default_config.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--pursuer-init",
        choices=["random", "fixed"],
        default="random",
        help=(
            "How to set the pursuer initial position/velocity.  "
            "'random' (default) places the pursuer on a random spherical shell "
            "around the target using the same MC parameters "
            "(pursuer_radius_range, pursuer_velocity_mode from the config) — "
            "this matches Monte Carlo behaviour and gives consistent results.  "
            "'fixed' uses the initial_position/initial_velocity from the YAML."
        ),
    )
    parser.add_argument(
        "--from-trial",
        type=int,
        default=None,
        dest="from_trial",
        metavar="TRIAL_IDX",
        help=(
            "Reproduce a specific Monte Carlo trial.  Reads pursuer initial "
            "conditions and trajectory file from monte_carlo_detailed.csv for each "
            "method, then runs the simulation with those exact parameters.  "
            "Automatically switches to 'experiment_prediction_ablation.yaml' unless "
            "--config is also given.  Example: --from-trial 763"
        ),
    )
    parser.add_argument(
        "--mc-results",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "monte_carlo_results", "monte_carlo_detailed.csv"),
        dest="mc_results",
        metavar="PATH",
        help="Path to monte_carlo_detailed.csv used by --from-trial.",
    )
    parser.add_argument(
        "--method",
        type=str,
        nargs="+",
        default=None,
        help=(
            "One or more named method / ablation variants to run.  When multiple "
            "names are given, all methods run against the same random scenario "
            "(same --seed) and their trajectories are overlaid on one comparison "
            "plot.  Example:  --method mpc_cv mpc_ca mpc_ekf_narx proposed_full\n"
            "Available: " + ", ".join(sorted(_METHOD_ABLATIONS))
        ),
    )
    parser.add_argument(
        "--solver", type=str, choices=["casadi", "acados"], default=None,
        help="Override the optimization solver for MPC methods."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output.output_dir from the config file.",
    )
    args = parser.parse_args()

    # ── Load MC trial data if reproducing a specific trial ────────────────────
    mc_trial_rows: dict = {}   # algorithm_name -> CSV row dict
    if args.from_trial is not None:
        import csv as _csv_mod
        _mc_path = args.mc_results
        # Accept either an absolute/path-as-given or a path relative to the
        # project root for convenience (users often supply workspace-relative
        # paths). Try several fallbacks before failing.
        if not os.path.isfile(_mc_path):
            # 1) try joining the project root
            _mc_try = os.path.join(_PROJECT_ROOT, _mc_path)
            if os.path.isfile(_mc_try):
                _mc_path = _mc_try
            else:
                # 2) if the user accidentally passed a path that includes the
                # project folder name (e.g. 'PNG_Hybrid/drone-interception-comparison-v3/...'),
                # strip the leading path up to the project folder and retry.
                proj_name = os.path.basename(_PROJECT_ROOT)
                if proj_name in _mc_path:
                    idx = _mc_path.find(proj_name)
                    rel = _mc_path[idx + len(proj_name) :].lstrip("/\\")
                    _mc_try2 = os.path.join(_PROJECT_ROOT, rel)
                    if os.path.isfile(_mc_try2):
                        _mc_path = _mc_try2
                    else:
                        _mc_try2 = None
                else:
                    _mc_try2 = None

                # 3) as a last resort, search for a file with the same basename
                # under the project tree and pick the first match.
                if not os.path.isfile(_mc_path):
                    base = os.path.basename(args.mc_results)
                    matches = []
                    for root, _, files in os.walk(_PROJECT_ROOT):
                        if base in files:
                            matches.append(os.path.join(root, base))
                    if matches:
                        _mc_path = matches[0]
                    else:
                        raise FileNotFoundError(
                            f"Monte Carlo detailed CSV not found: {args.mc_results}\n"
                            "Run the Monte Carlo first, or provide --mc-results <path>."
                        )
        with open(_mc_path, newline="") as _mc_f:
            for _mc_row in _csv_mod.DictReader(_mc_f):
                if int(_mc_row["trial"]) == args.from_trial:
                    mc_trial_rows[_mc_row["algorithm"]] = _mc_row
        if not mc_trial_rows:
            raise ValueError(f"Trial {args.from_trial} not found in {_mc_path}.")
        print(f"\n[--from-trial {args.from_trial}]  MC data found for: {sorted(mc_trial_rows)}")
        # Auto-use the MC experiment config unless the user explicitly specified one
        _default_cfg_path = os.path.join(_PROJECT_ROOT, "configs", "default_config.yaml")
        if args.config == _default_cfg_path:
            args.config = os.path.join(
                _PROJECT_ROOT, "configs", "experiment_prediction_ablation.yaml"
            )
            print(f"[--from-trial] Auto-using MC config: {args.config}")

    cfg_base = load_config(args.config)
    if args.solver:
        cfg_base.controller.solver = args.solver
    if args.output_dir:
        cfg_base.output.output_dir = args.output_dir

    # ── Resolve method list ───────────────────────────────────────────────
    if args.method is not None:
        invalid = [m for m in args.method if m not in _METHOD_ABLATIONS]
        if invalid:
            raise ValueError(
                f"Unknown method(s): {invalid}.  "
                f"Available: {sorted(_METHOD_ABLATIONS)}"
            )
        methods = args.method
    else:
        methods = [cfg_base.ablation.ablation_name or "proposed_full"]

    # ── Run each method ───────────────────────────────────────────────────
    results = []  # list of (method_name, df, metrics, cfg_m)

    for method in methods:
        print("=" * 72)
        print(f"  RUNNING:  {method}")
        print("=" * 72)

        # Reload + override config so each method gets a clean copy.
        # Re-seeding before each method ensures every method sees the same
        # target trajectory, wind, and sensor noise for a fair comparison.
        cfg_m = load_config(args.config)
        if args.solver:
            cfg_m.controller.solver = args.solver
        if args.output_dir:
            cfg_m.output.output_dir = args.output_dir
        cfg_m = apply_ablation_overrides(cfg_m, method)

        ab = cfg_m.ablation
        print(f"  Controller     : {cfg_m.controller.controller_type}")
        print(f"  Solver         : {getattr(cfg_m.controller, 'solver', 'casadi')}")
        print(f"  Disturbances   : {'ON' if ab.use_disturbances else 'OFF'}")
        print(f"  Target pred    : {ab.target_prediction_model}")
        print(f"  Scenario       : {cfg_m.scenario.scenario_type}")

        np.random.seed(args.seed)
        rng = np.random.default_rng(args.seed)

        # ── Trajectory override (--from-trial) — must happen before create_scenario
        if args.from_trial is not None:
            _mc_row_traj = mc_trial_rows.get(method) or next(iter(mc_trial_rows.values()))
            # source_trajectory stores only the bare filename; resolve it against
            # the trajectory_csv_dir from the config (same logic as the MC runner).
            _src_fname = (
                _mc_row_traj.get("source_trajectory", "")
                or _mc_row_traj.get("trajectory_file", "")
            )
            if _src_fname:
                # Build candidate paths in order of priority:
                # 1. as-is (already absolute)
                # 2. joined with trajectory_csv_dir from config
                # 3. joined with project root directly
                _csv_dir = getattr(cfg_m.monte_carlo, "trajectory_csv_dir", "") or ""
                if _csv_dir and not os.path.isabs(_csv_dir):
                    _csv_dir = os.path.normpath(os.path.join(_PROJECT_ROOT, _csv_dir))
                _candidates = [
                    _src_fname,
                    os.path.join(_csv_dir, os.path.basename(_src_fname)) if _csv_dir else "",
                    os.path.join(_PROJECT_ROOT, _src_fname),
                ]
                _resolved = next(
                    (p for p in _candidates if p and os.path.isfile(p)), None
                )
                if _resolved:
                    cfg_m.scenario.trajectory_csv_path = _resolved
                    print(f"  Trajectory     : {os.path.basename(_resolved)}")
                else:
                    raise FileNotFoundError(
                        f"Cannot resolve trajectory file '{_src_fname}'.\n"
                        f"Tried: {[c for c in _candidates if c]}"
                    )

        # Use create_scenario (same factory as Monte Carlo) so CSV and all
        # extended scenario types are handled identically.
        scenario   = create_scenario(cfg_m.scenario, cfg_m.simulation, seed=args.seed)
        pursuer    = build_pursuer(cfg_m)
        estimator  = build_estimator(cfg_m)
        controller = build_controller(cfg_m)
        wind_model = WindModel(cfg_m.wind)
        sensor     = SensorModel(cfg_m.sensor, rng=rng)

        # ── Pursuer initial conditions ────────────────────────────────────
        mc = cfg_m.monte_carlo
        if args.from_trial is not None:
            # Use the exact IC recorded in the MC run for this method so the
            # visual result matches the saved statistics precisely.
            _mc_row_ic = mc_trial_rows.get(method)
            if _mc_row_ic is None:
                _fallback_algo = next(iter(mc_trial_rows))
                _mc_row_ic = mc_trial_rows[_fallback_algo]
                print(
                    f"  [!] Method '{method}' not in MC data — "
                    f"using IC from '{_fallback_algo}'"
                )
            p0 = np.array([
                float(_mc_row_ic["p0_x"]),
                float(_mc_row_ic["p0_y"]),
                float(_mc_row_ic["p0_z"]),
            ])
            v0 = np.array([
                float(_mc_row_ic["v0_x"]),
                float(_mc_row_ic["v0_y"]),
                float(_mc_row_ic["v0_z"]),
            ])
            pursuer.reset(p0, v0)
            print(
                f"  Pursuer init   : from-trial {args.from_trial}  "
                f"p0={p0.tolist()}  v0={v0.tolist()}"
            )
        elif args.pursuer_init == "random":
            # Mirror Monte Carlo: random spherical-shell placement near target
            t0_pos, _, _ = scenario.get_target_state(0.0)
            p0 = _random_pursuer_position(
                t0_pos,
                mc.pursuer_position_mode,
                mc.pursuer_radius_range,
                rng,
                min_altitude=getattr(mc, "pursuer_min_altitude", 0.0),
            )
            v0 = _random_pursuer_velocity(
                mc.pursuer_velocity_mode,
                mc.pursuer_speed_range,
                rng,
                t0_pos,
                p0,
            )
            pursuer.reset(p0, v0)
            print(
                f"  Pursuer init   : random (r={np.linalg.norm(p0 - t0_pos):.1f} m "
                f"from target,  |v|={np.linalg.norm(v0):.1f} m/s)"
            )
        else:
            # fixed: use whatever initial_position/velocity is in the YAML
            p0 = np.array(cfg_m.pursuer.initial_position, dtype=float)
            v0 = np.array(cfg_m.pursuer.initial_velocity, dtype=float)
            pursuer.reset(p0, v0)
            print(f"  Pursuer init   : fixed  p0={p0.tolist()}  v0={v0.tolist()}")

        engine = SimulationEngine(cfg_m)
        logger = SimulationLogger()

        t_wall_start = _time.perf_counter()
        logger = engine.run(
            scenario, pursuer, estimator, controller, wind_model, sensor, logger
        )
        wall_time = _time.perf_counter() - t_wall_start

        metrics = compute_metrics(logger)
        df      = logger.to_dataframe()
        df.insert(0, "solver", getattr(cfg_m.controller, "solver", "casadi"))
        results.append((method, df, metrics, cfg_m))

        status = "SUCCESS" if metrics["success"] else "FAILURE"
        total_comp = metrics.get("total_compute_time_s", float("nan"))
        print(
            f"  → {status}  |  t_intercept = {metrics['intercept_time']:.3f} s  |  "
            f"min_dist = {metrics['min_distance']:.4f} m\n"
            f"     ctrl_compute = {total_comp:.4f} s (cumulative solver)  |  "
            f"wall_time = {wall_time:.2f} s"
        )

    # ── Print comparison table ────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  RESULTS SUMMARY")
    print("=" * 90)
    hdr = (
        f"  {'Method':<26} {'Solver':<7} {'Result':<8} {'T_int[s]':>9} "
        f"{'MinDist[m]':>11} {'Effort':>8} {'TotalComp[s]':>13} "
        f"{'MeanSolve[ms]':>14} {'RMSE_pos[m]':>12}"
    )
    print(hdr)
    print("  " + "-" * 96)
    for method, df, metrics, _cfg in results:
        status = "SUCCESS" if metrics["success"] else "FAILURE"
        solver_name  = df["solver"].iloc[0] if "solver" in df.columns and not df.empty else "casadi"
        total_comp   = metrics.get("total_compute_time_s", float("nan"))
        mean_solve   = metrics.get("mean_solve_time_s", float("nan")) * 1e3
        rmse_pos     = metrics.get("rmse_pos", float("nan"))
        print(
            f"  {method:<26} {solver_name:<7} {status:<8} {metrics['intercept_time']:>9.3f} "
            f"{metrics['min_distance']:>11.4f} {metrics['control_effort']:>8.2f} "
            f"{total_comp:>13.4f} {mean_solve:>14.3f} {rmse_pos:>12.4f}"
        )
    print("=" * 90)

    # ── Save CSV logs ─────────────────────────────────────────────────────
    out_dir = cfg_base.output.output_dir
    os.makedirs(out_dir, exist_ok=True)

    if cfg_base.output.save_logs:
        for method, df, metrics, _cfg in results:
            if len(methods) == 1:
                csv_path = os.path.join(out_dir, "single_run_log.csv")
            else:
                safe = method.replace("/", "_")
                csv_path = os.path.join(out_dir, f"single_run_log_{safe}.csv")
            df.to_csv(csv_path, index=False)
            print(f"  Log saved: {csv_path}")

    # ── Save plots ────────────────────────────────────────────────────────
    if cfg_base.output.save_plots:
        try:
            _plot_results(results, cfg_base, out_dir)
        except ImportError:
            print("matplotlib not available; skipping plots.")

    print("\nDone.")


if __name__ == "__main__":
    main()
