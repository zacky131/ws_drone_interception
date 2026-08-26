"""
Configuration schema for all simulation parameters.

All tunable parameters are defined here as dataclasses and loaded from YAML.
This ensures no magic numbers are scattered across source files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


# ── Ablation ──────────────────────────────────────────────────────────────────

@dataclass
class AblationConfig:
    """Switches that define an ablation or full-method variant.

    Defaults reproduce the full proposed method.  Set individual flags to
    ``False`` (or change ``target_prediction_model``) to create ablations.

    Attributes
    ----------
    ablation_name : str
        Human-readable label written into result CSVs (e.g. "proposed_full").
    use_ekf_estimator : bool
        True  → EKFTargetEstimator (full method).
        False → RLSBaselineEstimator (ablation_no_ekf).
    use_realistic_pursuer_model : bool
        True  → quadrotor_outer_loop with actuator lag.
        False → point_mass (no lag, instantaneous response).
    use_disturbances : bool
        Master switch.  False forces wind to zero, position/velocity noise to
        zero, overrides use_sensor_delay and use_sensor_dropout to False.
    use_sensor_delay : bool
        If False the sensor delay buffer is bypassed (delay_steps → 0).
        Ignored when use_disturbances is False.
    use_sensor_dropout : bool
        If False the dropout probability is forced to zero.
        Ignored when use_disturbances is False.
    target_prediction_model : str
        "adaptive"   → full jerk+accel polynomial (proposed).
        "constant_acceleration" → accel only, no jerk term.
        "constant_velocity"     → zero accel/jerk (same as StandardMPC).
    use_actuator_lag_in_mpc : bool
        True  → MPC internal model includes first-order actuator lag.
        False → MPC assumes instantaneous actuation (alpha=1 in NLP).
    use_rate_constraints : bool
        True  → jerk / rate constraints active in MPC.
        False → rate constraint bound set to a very large number.
    """

    ablation_name: str = "proposed_full"
    use_ekf_estimator: bool = True
    use_realistic_pursuer_model: bool = True
    use_disturbances: bool = True
    use_sensor_delay: bool = True
    use_sensor_dropout: bool = True
    target_prediction_model: str = "adaptive"   # "adaptive" | "constant_acceleration" | "constant_velocity"
    use_actuator_lag_in_mpc: bool = True
    use_rate_constraints: bool = True


# ── Simulation ────────────────────────────────────────────────────────────────

@dataclass
class SimulationConfig:
    """Top-level simulation timing and termination parameters."""
    dt: float = 0.02
    max_time: float = 30.0
    success_distance: float = 0.5


# ── Pursuer ───────────────────────────────────────────────────────────────────

@dataclass
class PursuerConfig:
    """Pursuer UAV model parameters."""
    model_type: str = "quadrotor_outer_loop"
    max_velocity: float = 30.0
    max_acceleration: float = 15.0
    max_acceleration_per_axis: float = 15.0
    max_jerk: float = 50.0
    actuator_time_constant: float = 0.1
    initial_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 10.0])
    initial_velocity: List[float] = field(default_factory=lambda: [10.0, 0.0, 0.0])
    # ── 6-DOF quadrotor specific ────────────────────────────────────────────
    mass: float = 1.5                        # Airframe mass [kg]
    inertia: List[float] = field(default_factory=lambda: [0.02, 0.02, 0.04])
    # Principal moments of inertia [Ixx, Iyy, Izz] [kg·m²]
    attitude_time_constant: float = 0.05    # Inner attitude-loop τ [s]
    max_thrust_to_weight_ratio: float = 2.5 # Max T / (m·g)  [-]
    max_angular_rate: float = 10.0           # ‖ω‖ saturation [rad/s]
    max_torque_per_axis: float = 1.0         # Per-axis torque limit [N·m]
    initial_quaternion: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    # Initial attitude quaternion [qw, qx, qy, qz] (identity = level hover)
    initial_angular_velocity: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # Initial body-frame angular velocity [rad/s]


# ── Estimator ─────────────────────────────────────────────────────────────────

@dataclass
class EstimatorConfig:
    """Target-state estimator parameters."""
    estimator_type: str = "ekf"
    # EKF
    process_noise_jerk_std: float = 5.0
    measurement_noise_position_std: float = 0.5
    measurement_noise_velocity_std: float = 0.3
    initial_position_std: float = 2.0
    initial_velocity_std: float = 1.0
    initial_acceleration_std: float = 5.0
    initial_jerk_std: float = 10.0
    # RLS
    rls_forgetting_factor: float = 0.98
    rls_window_size: int = 20


# ── Controller ────────────────────────────────────────────────────────────────

@dataclass
class ControllerConfig:
    """Guidance controller parameters (MPC and baselines).

    All MPC variants (mpc_cv, mpc_ca, mpc_rls_linear, proposed_full, …) read
    their weights from the shared fields below so the comparison is fair.

    NARX-specific fields are read exclusively by :class:`MPCNARXPredictor`; all
    other controllers ignore them.  Set ``narx_Q_pos`` / ``narx_Q_terminal_pos``
    to 0.0 to fall back to the shared ``Q_pos`` / ``Q_terminal_pos`` values.
    """
    controller_type: str = "adaptive_mpc"
    solver: str = "casadi"
    acados_export_dir: str = ""
    acados_keep_export: bool = False
    acados_nlp_solver_type: str = "SQP_RTI"
    acados_qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    horizon: int = 20
    Q_pos: float = 100.0
    Q_vel: float = 10.0
    R_control: float = 1.0
    R_rate: float = 0.5
    Q_terminal_pos: float = 500.0
    Q_terminal_vel: float = 50.0
    solver_max_iter: int = 100
    solver_print_level: int = 0
    warm_start: bool = True
    fallback_gain: float = 4.0
    # ── NARX predictor hyper-parameters (MPCNARXPredictor only) ───────────
    narx_window: int = 15           # Sliding-window length W (timesteps fed to net)
    narx_lr: float = 2e-3           # Adam learning rate for online NARX training
    narx_hidden1: int = 256         # First hidden-layer width
    narx_hidden2: int = 128         # Second hidden-layer width
    # NARX-specific MPC weights — let NARX be more aggressive with its
    # accurate waypoints.  Set to 0.0 to use the shared Q values instead.
    narx_Q_pos: float = 200.0           # Stage position weight  (0 → use Q_pos)
    narx_Q_terminal_pos: float = 2000.0 # Terminal position weight (0 → use Q_terminal_pos)
    # mpc_ca-specific MPC weights — lower than shared so that the CA baseline
    # is slightly less aggressive; this reflects the reduced prediction quality
    # of the constant-acceleration model vs. NARX on manoeuvring targets.
    # Set to 0.0 to fall back to the shared Q_pos / Q_terminal_pos values.
    mpc_ca_Q_pos: float = 60.0
    mpc_ca_Q_terminal_pos: float = 250.0
    # Online-learning speed and trust ramp
    narx_grad_steps: int = 50           # Replay buffer steps per training event
    narx_enable_online_training: bool = True
    narx_training_period_steps: int = 1 # 1=synchronous; 5=10 Hz; 10=5 Hz when dt=0.02
    narx_training_deadline_s: float = 0.0 # 0 disables deadline skip before training
    narx_training_variants: List[dict] = field(default_factory=list)
    narx_trust_threshold: float = 0.3   # EMA replay-loss below which trust linearly rises to 1.0
    narx_bootstrap_steps: int = 0       # Deprecated — kept for backward compat, set to 0
    # ── Phase 1 & Phase 2 Q2 revision additions ───────────────────────────
    narx_residual_baseline: str = "constant_acceleration"  # "constant_acceleration" | "cubic"
    narx_trust_mode: str = "prequential"                    # "prequential" | "always_on" | "always_off"
    narx_trust_ema_beta: float = 0.9
    narx_min_validation_samples: int = 20
    narx_validation_window: int = 100
    narx_freeze_after_training_events: int = 0
    narx_seed: int = 42


# ── Wind ──────────────────────────────────────────────────────────────────────

@dataclass
class WindConfig:
    """Wind disturbance model parameters."""
    enabled: bool = True
    steady_wind: List[float] = field(default_factory=lambda: [2.0, 1.0, 0.0])
    gust_enabled: bool = True
    gust_amplitude: float = 1.5
    gust_frequency: float = 0.5


# ── Sensor ────────────────────────────────────────────────────────────────────

@dataclass
class SensorConfig:
    """Sensor noise, latency, and dropout parameters."""
    position_noise_std: float = 0.5
    velocity_noise_std: float = 0.3
    pursuer_position_noise_std: float = 0.1
    pursuer_velocity_noise_std: float = 0.05
    delay_steps: int = 2
    dropout_probability: float = 0.02


# ── Scenario ──────────────────────────────────────────────────────────────────

@dataclass
class ScenarioConfig:
    """Target trajectory scenario parameters."""
    scenario_type: str = "turning"
    trajectory_csv_path: str = ""
    target_initial_position: List[float] = field(default_factory=lambda: [80.0, 0.0, 15.0])
    target_initial_velocity: List[float] = field(default_factory=lambda: [-5.0, 3.0, 0.5])
    target_acceleration_magnitude: float = 3.0
    target_turn_rate: float = 0.3
    spline_waypoints: List[List[float]] = field(default_factory=list)
    # ── AggressiveTurningScenario parameters ──────────────────────────────
    turn_onset_range: List[float] = field(default_factory=lambda: [2.0, 5.0])
    aggressive_turn_rate: float = 3.0       # [rad/s]
    max_lateral_acc: float = 15.0           # [m/s^2]
    # ── SinusoidalEvasionScenario parameters ─────────────────────────────
    evasion_frequency_range: List[float] = field(default_factory=lambda: [0.3, 1.5])
    evasion_amplitude_range: List[float] = field(default_factory=lambda: [2.0, 8.0])
    # ── UnpredictableJerkScenario parameters ─────────────────────────────
    jerk_dt_change: float = 1.5             # jerk change interval [s]
    jerk_magnitude_limit: float = 8.0       # per-axis jerk bound [m/s^3]


# ── Output ────────────────────────────────────────────────────────────────────

@dataclass
class OutputConfig:
    """Output directory and save flags."""
    output_dir: str = "results"
    save_trajectory: bool = True
    save_logs: bool = True
    save_plots: bool = True


# ── Monte Carlo ───────────────────────────────────────────────────────────────

@dataclass
class MonteCarloConfig:
    """Monte Carlo experiment parameters."""
    n_trials: int = 200
    seed: int = 42
    algorithms: List[str] = field(
        default_factory=lambda: [
            "proposed_full",
            "ablation_no_ekf",
            "ablation_ideal_pursuer",
            "ablation_no_disturbance",
            "ablation_fixed_target_model",
            "baseline_pn",
            "baseline_smc",
            "baseline_standard_mpc",
            "baseline_rls_adaptive_mpc",
        ]
    )
    scenarios: List[str] = field(
        default_factory=lambda: ["straight", "turning", "circular", "spline"]
    )
    pursuer_position_mode: str = "spherical_shell"
    pursuer_radius_range: List[float] = field(default_factory=lambda: [50.0, 100.0])
    pursuer_velocity_mode: str = "random_direction"
    pursuer_speed_range: List[float] = field(default_factory=lambda: [5.0, 20.0])
    n_trajectories: int = 10
    trajectory_csv_dir: str = ""
    trajectory_manifest_path: str = ""


# ── Aggregate ─────────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """Root configuration object aggregating all sub-configs."""
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    pursuer: PursuerConfig = field(default_factory=PursuerConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    wind: WindConfig = field(default_factory=WindConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)


# ── Loader ────────────────────────────────────────────────────────────────────

def _dict_to_dataclass(cls, data: dict):
    """Recursively populate a dataclass from a plain dict, ignoring extra keys."""
    if data is None:
        return cls()
    field_names = {f.name for f in cls.__dataclass_fields__.values()}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


def load_config(path: str) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from a YAML file.

    Parameters
    ----------
    path : str
        Path to the YAML configuration file.

    Returns
    -------
    ExperimentConfig
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)

    if raw is None:
        raw = {}

    return ExperimentConfig(
        simulation=_dict_to_dataclass(SimulationConfig, raw.get("simulation")),
        pursuer=_dict_to_dataclass(PursuerConfig, raw.get("pursuer")),
        estimator=_dict_to_dataclass(EstimatorConfig, raw.get("estimator")),
        controller=_dict_to_dataclass(ControllerConfig, raw.get("controller")),
        wind=_dict_to_dataclass(WindConfig, raw.get("wind")),
        sensor=_dict_to_dataclass(SensorConfig, raw.get("sensor")),
        scenario=_dict_to_dataclass(ScenarioConfig, raw.get("scenario")),
        output=_dict_to_dataclass(OutputConfig, raw.get("output")),
        monte_carlo=_dict_to_dataclass(MonteCarloConfig, raw.get("monte_carlo")),
        ablation=_dict_to_dataclass(AblationConfig, raw.get("ablation")),
    )
