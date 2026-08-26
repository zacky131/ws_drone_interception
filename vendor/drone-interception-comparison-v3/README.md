# Autonomous Drone Interception: Adaptive MPC with EKF-Based Target Estimation

## Overview

This repository implements a modular simulation and evaluation framework for
real-time autonomous drone interception in 3D space. It is designed to support
aerospace-oriented validation of guidance, navigation, and control (GNC)
architectures for pursuer–evader engagements.

### Revision Context

This codebase directly addresses reviewer feedback on a prior submission by:

1. **Replacing simplistic estimation**: The primary estimator is now an Extended
   Kalman Filter (EKF) with a 12-dimensional nearly-constant-jerk target model,
   providing filtered estimates of position, velocity, acceleration, and jerk
   with covariance diagnostics. The prior RLS + finite-difference estimator is
   retained only as a baseline for ablation.

2. **More realistic pursuer dynamics**: The default pursuer model includes
   first-order actuator lag (inner-loop approximation), acceleration magnitude
   and per-axis limits, acceleration rate limits, and velocity saturation.
   A simple point-mass model is retained for backward compatibility and ablation.

3. **Disturbance and sensor imperfection modeling**: The framework includes
   configurable wind disturbance (steady + gust), Gaussian measurement noise on
   position and velocity, sensor latency (delay buffer), and packet dropout
   simulation. All disturbance sources can be toggled independently for ablation.

4. **Constrained adaptive MPC**: The guidance controller is a receding-horizon
   MPC formulated with CasADi/IPOPT, incorporating actuator dynamics in the
   prediction model, disturbance-aware state propagation, warm-starting, and
   solver fallback logic.

5. **Rigorous ablation evaluation**: Monte Carlo experiments with deterministic
   seeding, YAML-driven ablation configuration, comprehensive per-timestep
   logging, extended metrics, and automated relative-improvement tables comparing
   the proposed method against every ablation and baseline.

### What This Is Not

This is a simulation-level validation framework. It does not claim hardware
readiness or flight-test validation. The pursuer model approximates inner-loop
attitude dynamics via a first-order lag, which is standard practice for
outer-loop guidance studies but does not replace full 6-DOF simulation or
hardware-in-the-loop testing.

## Repository Structure

```
project_root/
├── configs/                             # YAML configuration files
│   ├── default_config.yaml              # Single-run default parameters
│   ├── monte_carlo_config.yaml          # Full Monte Carlo experiment settings
│   ├── ablation_no_rate_constraint.yaml # Preset: MPC rate constraint removed
│   ├── experiment_tier1_comparison.yaml # Tier-1: fair baseline comparison
│   ├── experiment_tier2_ablation.yaml   # Tier-2: full ablation study
│   └── experiment_tier3_robustness.yaml # Tier-3: robustness sweep
├── src/                            # Core source modules
│   ├── dynamics/                   # Pursuer UAV models
│   │   ├── pursuer_base.py
│   │   ├── point_mass_pursuer.py
│   │   ├── quadrotor_outer_loop.py
│   │   └── quadrotor_6dof.py
│   ├── estimation/                 # Target state estimators
│   │   ├── estimator_base.py
│   │   ├── ekf_target_estimator.py
│   │   └── rls_baseline_estimator.py
│   ├── control/                    # MPC guidance controllers
│   │   ├── controller_base.py
│   │   └── adaptive_interception_mpc.py
│   ├── environment/                # Disturbance and sensor models
│   │   ├── wind_model.py
│   │   └── sensor_model.py
│   ├── baselines/                  # Baseline and ablation guidance laws
│   │   ├── proportional_navigation.py
│   │   ├── sliding_mode_guidance.py
│   │   ├── standard_mpc.py
│   │   ├── rls_adaptive_mpc.py
│   │   ├── fixed_target_model_mpc.py   # Ablation: constant-accel target prediction
│   │   ├── constant_velocity_mpc.py    # Fair baseline: EKF + const-vel MPC
│   │   ├── mpc_cv.py                   # Table-II baseline: no estimator, CV prediction
│   │   ├── mpc_ca.py                   # Table-II baseline: EKF acc, CA prediction
│   │   ├── mpc_rls_linear.py           # Table-II baseline: RLS acc, linear prediction
│   │   └── mpc_narx.py                 # Proposed: EKF + online NARX NN + tracking MPC
│   ├── simulation/                 # Simulation engine and logging
│   │   ├── scenario.py
│   │   ├── scenarios.py            # Aggressive/sinusoidal/unpredictable-jerk scenarios
│   │   ├── sim_engine.py
│   │   └── logger.py
│   ├── evaluation/                 # Metrics computation
│   │   └── metrics.py
│   └── utils/                      # Configuration and math utilities
│       ├── config_schema.py
│       └── math_helpers.py
├── scripts/                        # Executable experiment scripts
│   ├── run_single_case.py          # Single run with --method flag
│   ├── run_monte_carlo.py          # Parallel Monte Carlo evaluation
│   ├── run_robustness_sweep.py     # Parameter sweep (one-at-a-time)
│   ├── run_ablation_comparison.py  # Quick ablation comparison
│   ├── benchmark_estimators.py     # EKF vs RLS benchmark
│   ├── analyze_monte_carlo_results.py
│   └── export_results.py           # Publication-quality figures (PDF + PNG)
├── tests/                          # Unit tests (230 passing)
│   ├── test_ekf.py
│   ├── test_pursuer_dynamics.py
│   ├── test_mpc_constraints.py
│   ├── test_sensor_delay.py
│   ├── test_ablation.py
│   ├── test_ablation_variants.py
│   ├── test_constant_velocity_mpc.py
│   ├── test_new_scenarios.py
│   ├── test_extended_metrics.py
│   ├── test_robustness_sweep.py
│   └── test_export_functions.py
├── README.md
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies

| Package    | Purpose                          | Required |
|------------|----------------------------------|----------|
| numpy      | Numerical computation            | Yes      |
| scipy      | Linear algebra, optimization     | Yes      |
| casadi     | NLP solver for MPC               | Yes      |
| pandas     | Results I/O                      | Yes      |
| matplotlib | Plotting                         | Yes      |
| pyyaml     | Configuration loading            | Yes      |
| tqdm       | Progress bars                    | Yes      |
| pytest     | Unit testing                     | Dev      |

## Quick Start

### Single Simulation Run

```bash
python3 scripts/run_single_case.py --config configs/default_config.yaml
```

Run a specific named ablation variant with the `--method` flag:

```bash
python3 scripts/run_single_case.py --config configs/default_config.yaml --method proposed_full
python3 scripts/run_single_case.py --config configs/default_config.yaml --method ablation_no_ekf
python3 scripts/run_single_case.py --config configs/default_config.yaml --method ablation_no_disturbance
```

### Monte Carlo Evaluation

```bash
python3 scripts/run_monte_carlo.py --config configs/monte_carlo_config.yaml
```

### Robustness Sweep

```bash
python3 scripts/run_robustness_sweep.py \
    --config configs/experiment_tier3_robustness.yaml \
    --output-dir results/tier3_robustness/ \
    --trials 500 --method proposed_full --workers 4
```

### Estimator Benchmark

```bash
python3 scripts/benchmark_estimators.py --config configs/default_config.yaml
```

### Export Results

```bash
python3 scripts/export_results.py --results-dir monte_carlo_results/
```

## Configuration

All simulation parameters are specified in YAML configuration files. Key groups:

- **simulation**: timestep, duration, success threshold
- **pursuer**: model type, actuator lag, acceleration/velocity limits
- **estimator**: EKF noise parameters, initial covariance; RLS forgetting factor
- **controller**: MPC horizon, weights, solver settings, NARX hyper-parameters
- **wind / sensor**: wind model, sensor noise, delay, dropout
- **monte_carlo**: trial count, seed, algorithm list, scenario definitions
- **ablation**: ablation flags (see below)

See `configs/default_config.yaml` for the full parameter schema with comments.

### NARX Hyper-Parameters (`controller:` block)

These fields are read exclusively by `mpc_ekf_narx`; all other algorithms ignore
them, so changing them does not affect the fairness of comparisons.

```yaml
controller:
  # Shared MPC weights (all algorithms)
  horizon: 20
  Q_pos: 100.0
  Q_vel: 10.0
  R_control: 1.0
  R_rate: 0.5
  Q_terminal_pos: 500.0
  Q_terminal_vel: 50.0
  solver_max_iter: 100

  # NARX hyper-parameters (mpc_ekf_narx only)
  narx_window: 15           # Sliding-window length W (steps fed to the network)
  narx_lr: 0.002            # Adam learning rate for online adaptation
  narx_hidden1: 256         # First hidden-layer width
  narx_hidden2: 128         # Second hidden-layer width
  # NARX-specific MPC weights (set to 0.0 to fall back to shared Q_pos / Q_terminal_pos)
  narx_Q_pos: 200.0         # Stage position weight  (NARX has accurate waypoints → higher weight OK)
  narx_Q_terminal_pos: 2000.0  # Terminal position weight
```

### RLS Parameters (`estimator:` block)

```yaml
estimator:
  estimator_type: ekf         # "ekf" | "rls"
  rls_forgetting_factor: 0.98 # λ for exponentially weighted least squares
  rls_window_size: 20         # Sliding window for RLS polynomial fit
  # EKF process / measurement noise ...
```

## Ablation Study Support

### Named Methods

Every experiment is identified by a **method name** that maps to a set of
ablation flag overrides. The full set of supported names:

| Method name | Estimator | Pursuer | Disturbances | Target prediction |
|---|---|---|---|---|
| `proposed_full` | EKF | Quadrotor outer-loop | On | Adaptive (accel + jerk) |
| `ablation_no_ekf` | RLS | Quadrotor outer-loop | On | Adaptive |
| `ablation_ideal_pursuer` | EKF | Point-mass (no lag) | On | Adaptive |
| `ablation_no_disturbance` | EKF | Quadrotor outer-loop | Off | Adaptive |
| `ablation_fixed_target_model` | EKF | Quadrotor outer-loop | On | Fixed (const-accel, no jerk) |
| `ablation_no_rate_constraint` | EKF | Quadrotor outer-loop | On | Adaptive (no jerk hard constraint) |
| `mpc_cv` | None | Quadrotor outer-loop | On | Constant-velocity (no estimator) |
| `mpc_ca` | EKF | Quadrotor outer-loop | On | Constant-acceleration (EKF acc) |
| `mpc_rls_linear` | RLS | Quadrotor outer-loop | On | Linear (RLS acc, jerk dropped from NLP) |
| `mpc_ekf_narx` | EKF + NARX | Quadrotor outer-loop | On | Online NARX neural-network prediction |
| `constant_velocity_mpc` | EKF | Quadrotor outer-loop | On | Constant-velocity (fair baseline) |
| `baseline_pn` | EKF | Point-mass | On | — |
| `baseline_smc` | EKF | Point-mass | On | — |
| `baseline_standard_mpc` | EKF | Quadrotor outer-loop | On | Constant-velocity |
| `baseline_rls_adaptive_mpc` | RLS | Quadrotor outer-loop | On | Adaptive |

Legacy names (`ekf_adaptive_mpc`, `pn`, `smc`, `standard_mpc`, `rls_adaptive_mpc`)
are still accepted for backward compatibility.

### Using `--method` in a Single Run

```bash
# Full proposed method
python3 scripts/run_single_case.py --method proposed_full

# EKF ablation: replace EKF with RLS estimator
python3 scripts/run_single_case.py --method ablation_no_ekf

# Pursuer ablation: ideal point-mass (no actuator lag)
python3 scripts/run_single_case.py --method ablation_ideal_pursuer

# Disturbance ablation: wind off, noise zeroed, delay and dropout disabled
python3 scripts/run_single_case.py --method ablation_no_disturbance

# Target-prediction ablation: MPC uses constant-acceleration instead of adaptive jerk
python3 scripts/run_single_case.py --method ablation_fixed_target_model
```

### Multi-Method Comparison in a Single Run

Pass **multiple method names** to `--method` to run all of them against the same
random scenario (same `--seed`) and overlay their trajectories on one comparison
plot:

```bash
# Compare all Table-II prediction variants on one scenario
python3 scripts/run_single_case.py \
    --method mpc_cv mpc_ca mpc_rls_linear mpc_ekf_narx proposed_full

# Compare NARX against the EKF-adaptive baseline and a classical law
python3 scripts/run_single_case.py \
    --method mpc_ekf_narx proposed_full baseline_pn

# Reproduce ablation behaviour side-by-side
python3 scripts/run_single_case.py \
    --method proposed_full ablation_no_ekf ablation_no_disturbance
```

The output includes:
* A **6-panel figure** saved to `results/single_run_plots.png`:
  * 3-D trajectory with ▲ initial / ■ final position markers
  * XY top-view trajectory
  * Altitude profile Z vs time
  * Pursuer–target distance vs time
  * Control acceleration magnitude vs time
  * Target position estimation error vs time
* A **comparison table** in stdout with intercept time, miss distance, control
  effort, cumulative solver compute time, mean solve time per step, and
  estimator RMSE.
* One CSV log per method in `results/`.

The console output prints all active flags so you can confirm the ablation
configuration at a glance:

```
========================================================================
  SINGLE-RUN INTERCEPTION SIMULATION
========================================================================
  Method         : ablation_no_ekf
  Controller     : adaptive_mpc
  Estimator      : RLS
  Pursuer model  : realistic (quad ol)
  Disturbances   : ON
  Sensor delay   : ON
  Sensor dropout : ON
  Target pred    : adaptive
  ...
```

### Defining Ablations in YAML

Add an `ablation:` section to any config file to set flags directly without
using `--method`. Any key not specified inherits the full-method default.

```yaml
ablation:
  ablation_name: my_custom_variant
  use_ekf_estimator: true
  use_realistic_pursuer_model: true
  use_disturbances: false          # disables wind + zeroes noise/delay/dropout
  use_sensor_delay: true           # ignored when use_disturbances: false
  use_sensor_dropout: true         # ignored when use_disturbances: false
  target_prediction_model: "adaptive"   # "adaptive" | "constant_acceleration" | "constant_velocity"
  use_actuator_lag_in_mpc: true
  use_rate_constraints: true
```

Existing configs without an `ablation:` section load cleanly with all
full-method defaults.

### Running Ablations in Monte Carlo

Set the `algorithms` list in `monte_carlo_config.yaml` to the method names you
want to compare:

```yaml
monte_carlo:
  n_trials: 200
  algorithms:
    - "proposed_full"
    - "ablation_no_ekf"
    - "ablation_ideal_pursuer"
    - "ablation_no_disturbance"
    - "ablation_fixed_target_model"
    - "baseline_pn"
    - "baseline_smc"
    - "baseline_standard_mpc"
```

After the run, three CSV files are saved to the output directory:

| File | Contents |
|---|---|
| `monte_carlo_detailed.csv` | One row per trial with all metrics |
| `monte_carlo_summary.csv` | Grouped by `(algorithm, scenario)` with success rate, CI, mean metrics |
| `ablation_summary.csv` | Relative improvement (%) of `proposed_full` over each variant |

`ablation_summary.csv` columns:
- `relative_success_gain_pct` — success rate improvement over the ablation
- `relative_distance_reduction_pct` — miss distance reduction
- `relative_time_reduction_pct` — intercept time reduction
- `relative_effort_reduction_pct` — control effort reduction
- `relative_rmse_pos_reduction_pct` — estimator position RMSE reduction

## Algorithms

| Method | Estimator | Pursuer Model | Description |
|---|---|---|---|
| `proposed_full` | EKF | 6-DOF | Full method: adaptive MPC with EKF + actuator lag + disturbances |
| `ablation_no_ekf` | RLS | 6-DOF | EKF replaced by RLS estimator |
| `ablation_ideal_pursuer` | EKF | Point-mass | Actuator lag removed (instantaneous response) |
| `ablation_no_disturbance` | EKF | 6-DOF | Wind, noise, delay, and dropout all disabled |
| `ablation_fixed_target_model` | EKF | 6-DOF | MPC uses constant-acceleration prediction (no jerk term) |
| `ablation_no_rate_constraint` | EKF | 6-DOF | MPC jerk hard-constraint removed (unlimited acceleration rate) |
| `mpc_cv` | None | 6-DOF | No estimator; raw sensor pos/vel → constant-velocity MPC |
| `mpc_ca` | EKF | 6-DOF | EKF acc used directly; constant-acceleration MPC (jerk dropped) |
| `mpc_rls_linear` | RLS | 6-DOF | RLS-estimated acc/jerk; jerk dropped from MPC NLP (linear prediction) |
| `mpc_ekf_narx` | EKF + NARX NN | 6-DOF | Online NARX neural network learns evasion pattern; tracking MPC follows predicted waypoints |
| `constant_velocity_mpc` | EKF | 6-DOF | Fair baseline: same EKF/dynamics as proposed; const-vel target prediction |
| `baseline_pn` | EKF | Point-mass | Proportional navigation |
| `baseline_smc` | EKF | Point-mass | Sliding mode guidance |
| `baseline_standard_mpc` | EKF | 6-DOF | MPC with constant-velocity target assumption |
| `baseline_rls_adaptive_mpc` | RLS | 6-DOF | Prior submission: RLS-guided adaptive MPC |

### mpc_ekf_narx — EKF + Online NARX Hybrid

`mpc_ekf_narx` combines three components:

1. **EKF** (shared with `proposed_full`): Filters noisy target measurements into
   smooth 12-state estimates [pos, vel, acc, jerk].
2. **Online NARX neural network** (`_NARXNet`): A two-hidden-layer MLP (default
   256 → 128) that maps a sliding window of *W* past EKF states (each 9-D:
   [pos_dev, vel, acc]) to the next *N* predicted target waypoints [pos, vel].
   The network is updated online at every timestep via one-step Adam gradient
   descent, allowing it to learn the current target's evasion pattern within the
   episode.
3. **Tracking MPC**: A CasADi/IPOPT receding-horizon NLP that minimises
   weighted deviation from the NARX waypoints subject to the pursuer actuator
   model and safety constraints.

During the initial warmup period (*W* + *N* steps ≈ 0.7 s at dt = 0.02 s), the
controller falls back to constant-acceleration prediction (identical to `mpc_ca`).
Once warmed up, NARX predictions replace the polynomial prediction.

All hyper-parameters are exposed in the YAML `controller:` section — nothing is
hardcoded (see **NARX Hyper-Parameters** below).

## Experiment Tiers

Three pre-configured experiment tiers are provided as YAML presets:

### Tier 1 — Fair Baseline Comparison (`experiment_tier1_comparison.yaml`)

Compares methods that all use the EKF and the realistic 6-DOF pursuer, isolating
the value of the adaptive jerk-aware target prediction model.

| Axis of comparison | Value |
|---|---|
| Methods | `proposed_full`, `baseline_standard_mpc`, `constant_velocity_mpc` |
| Scenarios | `straight`, `turning`, `aggressive_turning` |
| Trials | 1 000 per (method × scenario) |
| Estimator | EKF (identical for all) |
| Pursuer | 6-DOF quadrotor outer-loop (identical for all) |
| Disturbances | On (identical for all) |

```bash
python3 scripts/run_monte_carlo.py --config configs/experiment_tier1_comparison.yaml --workers 8
```

### Tier 2 — Full Ablation Study (`experiment_tier2_ablation.yaml`)

Systematically disables one component at a time on the hardest scenario
(`aggressive_turning`) to quantify each component's contribution.

| Ablation variant | Component disabled |
|---|---|
| `ablation_no_ekf` | EKF → RLS estimator |
| `ablation_no_disturbance` | Wind + noise + delay + dropout |
| `ablation_fixed_target_model` | Jerk term in MPC prediction |
| `ablation_ideal_pursuer` | Actuator lag (point-mass) |
| `ablation_no_rate_constraint` | MPC jerk hard-constraint |

```bash
python3 scripts/run_monte_carlo.py \
    --config configs/experiment_tier2_ablation.yaml
python3 scripts/analyze_monte_carlo_results.py \
    --results-dir results/tier2_ablation/ --reference proposed_full
```

### Tier 3 — Robustness Sweep (`experiment_tier3_robustness.yaml`)

Sweeps one disturbance/scenario parameter at a time and reports success rate
with 95 % Wilson confidence intervals.

| Swept parameter | Values |
|---|---|
| `wind_magnitude` | 0, 2, 5, 10 m/s |
| `sensor_noise_std` | 0, 0.1, 0.3, 0.5 m |
| `sensor_delay_steps` | 0, 1, 2, 4 timesteps |
| `target_scenario` | straight, turning, aggressive\_turning, sinusoidal\_evasion, unpredictable\_jerk |

```bash
python3 scripts/run_robustness_sweep.py \
    --config configs/experiment_tier3_robustness.yaml \
    --output-dir results/tier3_robustness/ \
    --trials 500 --method proposed_full --workers 4
# Generate robustness-curve figures:
python3 -c "
import sys; sys.path.insert(0,'scripts')
from export_results import export_robustness_curves
export_robustness_curves('results/tier3_robustness', 'results/tier3_robustness/robustness_curves')
"
```

## Metrics

`compute_metrics()` in `src/evaluation/metrics.py` returns:

| Metric | Description |
|---|---|
| `success` | Whether interception succeeded |
| `intercept_time` | Time to interception [s] |
| `terminal_distance` | Miss distance at end of simulation [m] |
| `min_distance` | Minimum pursuer–target distance [m] |
| `control_effort` | Σ ‖u‖² · dt |
| `max_cmd_acc` | Peak commanded acceleration [m/s²] |
| `max_applied_acc` | Peak applied acceleration [m/s²] |
| `control_smoothness` | Σ ‖Δu‖² · dt (lower = smoother) |
| `mean_solve_time_s` | Mean MPC NLP solver time per step [s] |
| `max_solve_time_s` | Maximum single-step NLP solver time [s] |
| `total_compute_time_s` | Cumulative NLP solver time for the episode [s] |
| `solver_feasibility_rate` | Fraction of steps with successful solve |
| `saturation_rate` | Fraction of steps near acceleration limit |
| `terminal_speed` | Pursuer–target relative speed at end [m/s] |
| `rmse_pos` | Estimator position RMSE [m] |
| `rmse_vel` | Estimator velocity RMSE [m/s] |
| `rmse_acc` | Estimator acceleration RMSE [m/s²] |
| `rmse_jerk` | Estimator jerk RMSE [m/s³] (when available) |
| `failure_reason` | Free-form failure description |
| `failure_category` | `success` / `timeout` / `altitude_violation` / `divergence` / `solver_failure` / `other` |

### Extended Metrics

Additional functions in `src/evaluation/metrics.py`:

| Function | Description |
|---|---|
| `compute_miss_distance_cdf(distances, thresholds)` | Fraction of trials at or below each threshold |
| `compute_success_rate_vs_parameter(df, param_col, threshold)` | Success rate + Wilson 95% CI grouped by a sweep parameter |
| `compute_solver_stats(solve_times, realtime_budget_s)` | Mean/std/max/P99 solve time; budget violation rate |
| `compute_estimation_convergence(ekf_errors, convergence_threshold_m)` | Mean±std error curve; first convergence timestep |

### Export Functions

All functions in `scripts/export_results.py` save both `.pdf` and `.png` at 300 DPI:

| Function | Output |
|---|---|
| `export_cdf_plot(results_df, path)` | CDF of terminal miss distance, one curve per method |
| `export_robustness_curves(sweep_dir, path)` | Success rate vs. each swept parameter |
| `export_ablation_bar_chart(results_df, path)` | Horizontal bar chart of success rate by variant |
| `export_solver_time_histogram(solve_times, path)` | Histogram of MPC solver wall-clock time |

## Scenarios

In addition to the original scenario types (`straight`, `turning`, `circular`,
`spline`, `csv`), three harder scenarios are available via `src/simulation/scenarios.py`:

| Scenario type | Description | Key parameters |
|---|---|---|
| `aggressive_turning` | High-g banked turn with randomised onset time | `turn_onset_range`, `aggressive_turn_rate`, `max_lateral_acc` |
| `sinusoidal_evasion` | Lateral sinusoidal evasion with random frequency/amplitude | `evasion_frequency_range`, `evasion_amplitude_range` |
| `unpredictable_jerk` | Random jerk impulses at fixed time intervals | `jerk_dt_change`, `jerk_magnitude_limit` |

All scenario parameters are configurable in the `scenario:` block of any YAML config.

## License

Academic use.
