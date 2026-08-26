#!/usr/bin/env python
"""
Robustness sweep runner.

Sweeps one simulation parameter at a time while keeping all others at their
default values and runs N Monte Carlo trials for the specified guidance method.

Parameters swept
----------------
* ``wind_magnitude``    — scales the steady-wind vector norm [m/s]
* ``sensor_noise_std``  — target position sensor noise std [m]
* ``sensor_delay_steps``— measurement latency [timesteps]
* ``target_scenario``   — scenario type string

For each (parameter, value) pair a CSV is written to
``<output_dir>/<param>_sweep.csv``.  A final summary table is printed at
the end.

Usage
-----
python scripts/run_robustness_sweep.py \\
    --config configs/monte_carlo_config.yaml \\
    --output-dir results/robustness/ \\
    --trials 500 \\
    --method proposed_full \\
    --workers 4
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

from src.utils.config_schema import ExperimentConfig, load_config
from run_single_case import (  # noqa: E402
    _METHOD_ABLATIONS,
    apply_ablation_overrides,
    build_estimator,
    build_controller,
)


# ── Sweep parameter definitions ───────────────────────────────────────────────

#: Default sweep grid: maps parameter name → list of values to sweep.
DEFAULT_SWEEP_GRID: Dict[str, List[Any]] = {
    "wind_magnitude":      [0.0, 2.0, 5.0, 10.0],
    "sensor_noise_std":    [0.0, 0.1, 0.3, 0.5],
    "sensor_delay_steps":  [0, 1, 2, 4],
    "target_scenario":     [
        "straight",
        "turning",
        "aggressive_turning",
        "sinusoidal_evasion",
        "unpredictable_jerk",
    ],
}


# ── Picklable trial spec ──────────────────────────────────────────────────────

@dataclass
class SweepTrialSpec:
    """Fully serialisable specification for one robustness sweep trial."""

    sweep_param: str          # parameter name being swept
    sweep_value: Any          # value of the swept parameter
    trial_idx: int
    trial_seed: int
    method: str               # guidance method name
    p0: List[float]           # pursuer initial position
    v0: List[float]           # pursuer initial velocity
    cfg: ExperimentConfig


# ── Apply sweep parameter to config ──────────────────────────────────────────

def _apply_sweep(cfg: ExperimentConfig, param: str, value: Any) -> ExperimentConfig:
    """Modify *cfg* in-place according to the sweep parameter.

    Parameters
    ----------
    cfg : ExperimentConfig
        Config to mutate.
    param : str
        One of the keys in :data:`DEFAULT_SWEEP_GRID`.
    value : Any
        The parameter value for this sweep point.

    Returns
    -------
    cfg : ExperimentConfig
        The same (mutated) object for chaining.
    """
    if param == "wind_magnitude":
        if cfg.wind.enabled:
            sw = np.asarray(cfg.wind.steady_wind, dtype=float)
            norm = float(np.linalg.norm(sw))
            if norm > 1e-9:
                cfg.wind.steady_wind = (sw / norm * float(value)).tolist()
            else:
                # Default direction when steady wind is zero
                cfg.wind.steady_wind = [float(value), 0.0, 0.0]
        else:
            cfg.wind.steady_wind = [float(value), 0.0, 0.0]
            cfg.wind.enabled = float(value) > 0.0
    elif param == "sensor_noise_std":
        cfg.sensor.position_noise_std = float(value)
        cfg.sensor.velocity_noise_std = float(value) * 0.6  # keep ~same ratio
    elif param == "sensor_delay_steps":
        cfg.sensor.delay_steps = int(value)
    elif param == "target_scenario":
        cfg.scenario.scenario_type = str(value)
    else:
        raise ValueError(f"Unknown sweep parameter: '{param}'")
    return cfg


# ── Worker function ───────────────────────────────────────────────────────────

def _run_sweep_trial(spec: SweepTrialSpec) -> Dict[str, Any]:
    """Worker executed in a subprocess; builds all components from scratch."""
    from src.dynamics.point_mass_pursuer import PointMassPursuer
    from src.dynamics.quadrotor_outer_loop import QuadrotorOuterLoopPursuer
    from src.dynamics.quadrotor_6dof import Quadrotor6DOFPursuer
    from src.environment.wind_model import WindModel
    from src.environment.sensor_model import SensorModel
    from src.simulation.scenarios import create_scenario
    from src.simulation.sim_engine import SimulationEngine
    from src.simulation.logger import SimulationLogger
    from src.evaluation.metrics import compute_metrics

    cfg = copy.deepcopy(spec.cfg)
    rng = np.random.default_rng(spec.trial_seed)

    # Apply sweep override, then method ablation
    _apply_sweep(cfg, spec.sweep_param, spec.sweep_value)
    cfg = apply_ablation_overrides(cfg, spec.method)

    # Build scenario (supports all existing + new types via create_scenario)
    scenario = create_scenario(cfg.scenario, cfg.simulation, seed=spec.trial_seed)

    estimator = build_estimator(cfg)
    controller = build_controller(cfg)

    # Pursuer model
    if not cfg.ablation.use_realistic_pursuer_model:
        pursuer = PointMassPursuer(cfg.pursuer)
    elif cfg.pursuer.model_type == "quadrotor_6dof":
        pursuer = Quadrotor6DOFPursuer(cfg.pursuer)
    else:
        pursuer = QuadrotorOuterLoopPursuer(cfg.pursuer)

    p0 = np.array(spec.p0, dtype=float)
    v0 = np.array(spec.v0, dtype=float)
    pursuer.reset(p0, v0)

    wind_model = WindModel(cfg.wind)
    sensor = SensorModel(cfg.sensor, rng=rng)
    engine = SimulationEngine(cfg)
    logger = SimulationLogger()

    logger = engine.run(scenario, pursuer, estimator, controller, wind_model, sensor, logger)
    m = compute_metrics(logger)

    return {
        "sweep_param": spec.sweep_param,
        "sweep_value": spec.sweep_value,
        "method": spec.method,
        "trial": spec.trial_idx,
        **m,
    }


# ── Initial-condition helpers (duplicated from run_monte_carlo for isolation) ─

def _random_pursuer_ic(
    target_pos: np.ndarray,
    radius_range: List[float],
    speed_range: List[float],
    rng: np.random.Generator,
) -> tuple:
    """Return (p0, v0) for a pursuer positioned on a spherical shell."""
    r = rng.uniform(radius_range[0], radius_range[1])
    az = rng.uniform(0, 2 * np.pi)
    el = rng.uniform(-np.pi / 2, np.pi / 2)
    offset = np.array([
        r * np.cos(el) * np.cos(az),
        r * np.cos(el) * np.sin(az),
        r * np.sin(el),
    ])
    p0 = target_pos + offset
    p0[2] = max(0.0, p0[2])

    speed = rng.uniform(speed_range[0], speed_range[1])
    az2 = rng.uniform(0, 2 * np.pi)
    el2 = rng.uniform(-np.pi / 6, np.pi / 6)
    v0 = np.array([
        speed * np.cos(el2) * np.cos(az2),
        speed * np.cos(el2) * np.sin(az2),
        speed * np.sin(el2),
    ])
    return p0.tolist(), v0.tolist()


# ── Main sweep logic ──────────────────────────────────────────────────────────

def run_sweep(
    cfg: ExperimentConfig,
    method: str,
    n_trials: int,
    sweep_grid: Dict[str, List[Any]],
    output_dir: str,
    n_workers: int,
    base_seed: int = 0,
) -> pd.DataFrame:
    """Run the full robustness sweep and write per-parameter CSVs.

    Parameters
    ----------
    cfg : ExperimentConfig
        Base configuration (unmodified; each trial gets its own copy).
    method : str
        Registered guidance method name (must be in ``_METHOD_ABLATIONS``).
    n_trials : int
        Number of trials per (parameter, value) pair.
    sweep_grid : dict
        Maps parameter name → list of sweep values.
    output_dir : str
        Directory in which to save ``<param>_sweep.csv`` files.
    n_workers : int
        Number of parallel worker processes.
    base_seed : int
        Base random seed; each trial gets ``base_seed + global_trial_index``.

    Returns
    -------
    all_results : pd.DataFrame
        Combined results from all sweep configurations.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Pre-compute target initial position for IC sampling
    target_pos = np.asarray(cfg.scenario.target_initial_position, dtype=float)
    radius_range = cfg.monte_carlo.pursuer_radius_range
    speed_range = cfg.monte_carlo.pursuer_speed_range

    # Build all trial specs
    all_specs: List[SweepTrialSpec] = []
    seed_counter = base_seed
    for param, values in sweep_grid.items():
        for val in values:
            rng_ic = np.random.default_rng(seed_counter)
            for i in range(n_trials):
                p0, v0 = _random_pursuer_ic(
                    target_pos, radius_range, speed_range, rng_ic
                )
                all_specs.append(SweepTrialSpec(
                    sweep_param=param,
                    sweep_value=val,
                    trial_idx=i,
                    trial_seed=seed_counter + i,
                    method=method,
                    p0=p0,
                    v0=v0,
                    cfg=copy.deepcopy(cfg),
                ))
            seed_counter += n_trials

    print(f"\nTotal sweep trials: {len(all_specs)}")
    print(f"Workers: {n_workers}")

    all_rows: List[Dict[str, Any]] = []
    t_start = _time.perf_counter()

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_run_sweep_trial, s): s for s in all_specs}
        with tqdm(total=len(all_specs), desc="Sweep", unit="trial") as pbar:
            for fut in as_completed(futures):
                try:
                    row = fut.result()
                    all_rows.append(row)
                except Exception as exc:
                    spec = futures[fut]
                    all_rows.append({
                        "sweep_param": spec.sweep_param,
                        "sweep_value": spec.sweep_value,
                        "method": spec.method,
                        "trial": spec.trial_idx,
                        "success": False,
                        "failure_reason": f"worker_exception: {exc}",
                    })
                finally:
                    pbar.update(1)

    elapsed = _time.perf_counter() - t_start
    print(f"\nSweep complete in {elapsed:.1f}s")

    all_df = pd.DataFrame(all_rows)

    # Save per-parameter CSVs
    for param in sweep_grid:
        sub = all_df[all_df["sweep_param"] == param]
        if sub.empty:
            continue
        csv_path = os.path.join(output_dir, f"{param}_sweep.csv")
        sub.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    # Combined CSV
    combined_path = os.path.join(output_dir, "robustness_combined.csv")
    all_df.to_csv(combined_path, index=False)
    print(f"  Combined: {combined_path}")

    return all_df


def _print_summary_table(df: pd.DataFrame, sweep_grid: Dict[str, List[Any]]) -> None:
    """Print a formatted summary of success rates across all sweep conditions."""
    from src.evaluation.metrics import compute_success_rate_vs_parameter

    print("\n" + "=" * 72)
    print("  ROBUSTNESS SWEEP SUMMARY")
    print("=" * 72)

    for param in sweep_grid:
        sub = df[df["sweep_param"] == param]
        if sub.empty:
            continue
        try:
            summary = compute_success_rate_vs_parameter(sub, "sweep_value", threshold=0.5)
        except Exception:
            summary = None

        print(f"\n  Parameter: {param}")
        print(f"  {'Value':>20}  {'Trials':>7}  {'Success':>7}  {'Rate (%)':>9}  "
              f"{'CI_lo':>7}  {'CI_hi':>7}")
        print("  " + "-" * 64)
        if summary is not None:
            for _, row in summary.iterrows():
                print(f"  {str(row['sweep_value']):>20}  {row['n_trials']:>7}  "
                      f"{row['n_success']:>7}  {row['success_rate_pct']:>9.1f}  "
                      f"{row['sr_ci_lower']:>7.1f}  {row['sr_ci_upper']:>7.1f}")
        else:
            for val in df[df["sweep_param"] == param]["sweep_value"].unique():
                bucket = sub[sub["sweep_value"] == val]
                n = len(bucket)
                k = int(bucket["success"].sum()) if "success" in bucket.columns else 0
                print(f"  {str(val):>20}  {n:>7}  {k:>7}  {k/n*100:>9.1f}")

    print("=" * 72)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Robustness sweep: vary one parameter at a time and measure success rate."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "configs", "monte_carlo_config.yaml"),
        help="Path to base YAML config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "results", "robustness"),
        help="Directory to write sweep CSVs.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=500,
        help="Number of trials per (parameter, value) pair. Default 500.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="proposed_full",
        choices=sorted(_METHOD_ABLATIONS.keys()),
        help="Guidance method to evaluate. Default: proposed_full.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cpu_count() - 1),
        help="Parallel worker processes. Default: cpu_count − 1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed. Default 0.",
    )
    # Allow selective parameter sweeping
    parser.add_argument(
        "--params",
        nargs="+",
        choices=list(DEFAULT_SWEEP_GRID.keys()),
        default=None,
        help="Restrict sweep to specific parameters. Default: all.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    cfg = load_config(args.config)

    sweep_grid = DEFAULT_SWEEP_GRID
    if args.params is not None:
        sweep_grid = {k: v for k, v in DEFAULT_SWEEP_GRID.items() if k in args.params}

    print("=" * 72)
    print("  ROBUSTNESS SWEEP RUNNER")
    print("=" * 72)
    print(f"  Config  : {args.config}")
    print(f"  Method  : {args.method}")
    print(f"  Trials  : {args.trials} per (param, value)")
    print(f"  Workers : {args.workers}")
    print(f"  Output  : {args.output_dir}")
    print(f"  Params  : {list(sweep_grid.keys())}")
    print("=" * 72)

    all_df = run_sweep(
        cfg=cfg,
        method=args.method,
        n_trials=args.trials,
        sweep_grid=sweep_grid,
        output_dir=args.output_dir,
        n_workers=args.workers,
        base_seed=args.seed,
    )

    _print_summary_table(all_df, sweep_grid)
    print("\nDone.")


if __name__ == "__main__":
    main()
