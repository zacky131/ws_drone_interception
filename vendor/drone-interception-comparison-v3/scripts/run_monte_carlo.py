#!/usr/bin/env python
"""
Monte Carlo robustness evaluation — parallel multiprocessing version.

Runs N trials per (algorithm, scenario) pair with randomised pursuer initial
conditions using a ``ProcessPoolExecutor`` for multi-core parallelism.

Each worker process builds its own algorithm components from scratch so that
CasADi-compiled NLP solvers (which are not picklable) are safely created inside
the subprocess, avoiding inter-process serialisation problems.

Usage:
    python3 scripts/run_monte_carlo.py --config configs/monte_carlo_config.yaml --workers 8
    python3 scripts/run_monte_carlo.py --workers 8   # explicit core count
    python3 scripts/run_monte_carlo.py --workers 1   # sequential (for debugging)
"""

from __future__ import annotations

import argparse
import os
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import dataclasses
from multiprocessing import cpu_count
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from src.utils.config_schema import ExperimentConfig, load_config


def _open_progress_stream():
    """Return a stream that keeps tqdm live even under `conda run` capture."""
    if os.name != "nt":
        try:
            return open("/dev/tty", "w", buffering=1)
        except OSError:
            pass
    return sys.stderr


# Import the shared ablation helpers from run_single_case so the builder logic
# lives in one place only.
sys.path.insert(0, _SCRIPT_DIR)
from run_single_case import (  # noqa: E402  (local import after path fix)
    _METHOD_ABLATIONS,
    apply_ablation_overrides,
    build_estimator,
    build_controller,
)


def _effective_solver_name(controller) -> str:
    """Return the solver actually used by a controller instance."""
    if hasattr(controller, "_solver_type"):
        return str(controller._solver_type)
    inner = getattr(controller, "_mpc", None)
    if inner is not None and hasattr(inner, "_solver_type"):
        return str(inner._solver_type)
    if hasattr(controller, "_solver"):
        return "casadi"
    return "none"


# ── Trial specification (must be picklable for multiprocessing) ───────────────

@dataclass
class TrialSpec:
    """Fully serialisable description of one Monte Carlo trial."""
    algo_name: str
    scen_name: str
    trial_idx: int
    trial_seed: int
    p0: List[float]      # pursuer initial position
    v0: List[float]      # pursuer initial velocity
    cfg: ExperimentConfig
    base_algo_name: str = ""


# ── Per-worker builder (top-level so it is picklable) ─────────────────────────

def _base_algo_name(algo_name: str, cfg: ExperimentConfig) -> str:
    aliases = getattr(cfg.controller, "narx_variant_aliases", {}) or {}
    if algo_name in aliases:
        return aliases[algo_name]
    variants = _narx_training_variants(cfg)
    for variant in variants:
        if variant.get("label") == algo_name:
            return "mpc_ekf_narx"
    return algo_name


def _narx_training_variants(cfg: ExperimentConfig) -> List[Dict]:
    variants = getattr(cfg.controller, "narx_training_variants", None)
    if not variants:
        return []
    if not isinstance(variants, list):
        raise TypeError("controller.narx_training_variants must be a list of dictionaries")
    cleaned: List[Dict] = []
    for idx, raw in enumerate(variants):
        if not isinstance(raw, dict):
            raise TypeError("Each narx_training_variants entry must be a dictionary")
        label = str(raw.get("label") or raw.get("name") or f"narx_variant_{idx}")
        if not label:
            raise ValueError("NARX training variant label cannot be empty")
        item = dict(raw)
        item["label"] = label
        cleaned.append(item)
    return cleaned


def _apply_narx_training_variant(cfg: ExperimentConfig, variant: Dict) -> ExperimentConfig:
    updates = {k: v for k, v in variant.items() if k not in {"label", "name", "description"}}
    ctrl = cfg.controller
    for key, value in updates.items():
        if not hasattr(ctrl, key):
            raise AttributeError(f"Unknown controller field in NARX variant {variant['label']!r}: {key}")
        setattr(ctrl, key, value)
    return cfg


def _expanded_algorithms(cfg: ExperimentConfig, algorithms: List[str]) -> List[str]:
    variants = _narx_training_variants(cfg)
    aliases: Dict[str, str] = {}
    for variant in variants:
        aliases[str(variant["label"])] = "mpc_ekf_narx"
    cfg.controller.narx_variant_aliases = aliases

    if not variants:
        return algorithms
    expanded: List[str] = []
    for algo_name in algorithms:
        if algo_name == "mpc_ekf_narx":
            for variant in variants:
                expanded.append(str(variant["label"]))
        else:
            expanded.append(algo_name)
    return expanded


def _build_components(algo_name: str, cfg: ExperimentConfig):
    """Instantiate estimator, controller, and pursuer model type string.

    Supports both the new ablation-aware names (``proposed_full``,
    ``ablation_*``, ``baseline_*``) and the legacy names (``ekf_adaptive_mpc``,
    ``pn``, etc.) for backward compatibility.
    """
    import copy
    # Apply ablation overrides to a fresh copy of cfg so worker state is isolated
    cfg_w = apply_ablation_overrides(copy.deepcopy(cfg), _base_algo_name(algo_name, cfg))

    estimator = build_estimator(cfg_w)
    controller = build_controller(cfg_w)

    # Pursuer model string — determined by ablation flag
    if not cfg_w.ablation.use_realistic_pursuer_model:
        pursuer_type = "point_mass"
    elif cfg_w.pursuer.model_type == "quadrotor_6dof":
        pursuer_type = "quadrotor_6dof"
    else:
        pursuer_type = "quadrotor_outer_loop"

    return estimator, controller, pursuer_type, cfg_w


def _run_trial(spec: TrialSpec) -> Dict:
    """Worker function executed in a subprocess.

    All imports and component construction happen here so that CasADi NLP
    solvers are compiled inside the subprocess and never transmitted across
    process boundaries.
    """
    # Deferred imports — each subprocess loads only what it needs
    from src.dynamics.point_mass_pursuer import PointMassPursuer
    from src.dynamics.quadrotor_outer_loop import QuadrotorOuterLoopPursuer
    try:
        from src.dynamics.quadrotor_6dof import Quadrotor6DOFPursuer
    except ImportError:
        pass
    from src.environment.wind_model import WindModel
    from src.environment.sensor_model import SensorModel
    from src.simulation.scenarios import create_scenario
    from src.simulation.sim_engine import SimulationEngine
    from src.simulation.logger import SimulationLogger
    from src.evaluation.metrics import compute_metrics

    cfg = spec.cfg
    rng = np.random.default_rng(spec.trial_seed)

    # Rebuild scenario inside worker (supports all scenario types via create_scenario)
    cfg.scenario.scenario_type = spec.scen_name
    scenario = create_scenario(cfg.scenario, cfg.simulation, seed=spec.trial_seed)

    estimator, controller, pursuer_type, cfg_w = _build_components(
        spec.base_algo_name or spec.algo_name, cfg
    )

    p0 = np.array(spec.p0, dtype=float)
    v0 = np.array(spec.v0, dtype=float)

    if pursuer_type == "point_mass":
        pursuer = PointMassPursuer(cfg_w.pursuer)
    elif pursuer_type == "quadrotor_6dof":
        try:
            pursuer = Quadrotor6DOFPursuer(cfg_w.pursuer)
        except NameError:
            print("Warning: Quadrotor6DOFPursuer not implemented, falling back to outer_loop")
            pursuer = QuadrotorOuterLoopPursuer(cfg_w.pursuer)
    else:
        pursuer = QuadrotorOuterLoopPursuer(cfg_w.pursuer)
    pursuer.reset(p0, v0)

    wind_model = WindModel(cfg_w.wind)
    sensor = SensorModel(cfg_w.sensor, rng=rng)
    engine = SimulationEngine(cfg_w)
    logger = SimulationLogger()

    logger = engine.run(scenario, pursuer, estimator, controller, wind_model, sensor, logger)
    m = compute_metrics(logger)

    # ── Save per-trial trajectory CSV (pursuer + target) if requested ─────
    traj_file: str = ""
    # Capture the source trajectory filename for traceability
    source_csv = os.path.basename(cfg.scenario.trajectory_csv_path or "")
    if cfg.output.save_trajectory:
        out_dir = cfg.output.output_dir
        traj_dir = os.path.join(out_dir, "trajectory", spec.algo_name, spec.scen_name)
        os.makedirs(traj_dir, exist_ok=True)
        df_traj = logger.to_dataframe()
        # Keep only trajectory-relevant columns (position/velocity of pursuer & target)
        traj_cols = [
            "time", "step",
            "p_px", "p_py", "p_pz",
            "p_vx", "p_vy", "p_vz",
            "p_ax", "p_ay", "p_az",
            "t_px", "t_py", "t_pz",
            "t_vx", "t_vy", "t_vz",
            "t_ax", "t_ay", "t_az",
            "distance",
        ]
        available = [c for c in traj_cols if c in df_traj.columns]
        traj_path = os.path.join(traj_dir, f"trial_{spec.trial_idx:04d}.csv")
        # Embed source trajectory filename as a comment in the first row so it
        # is visible when the CSV is opened without the xlsx index.
        with open(traj_path, "w") as _fh:
            _fh.write(f"# source_trajectory: {source_csv}\n")
            _fh.write(f"# algorithm: {spec.algo_name}\n")
            _fh.write(f"# scenario: {spec.scen_name}\n")
            _fh.write(f"# trial: {spec.trial_idx}\n")
        df_traj[available].to_csv(traj_path, index=False, mode="a")
        # Store relative path (relative to output_dir) so it is portable
        traj_file = os.path.relpath(traj_path, out_dir)

    return {
        "algorithm": spec.algo_name,
        "base_algorithm": spec.base_algo_name or spec.algo_name,
        "solver": _effective_solver_name(controller),
        "scenario": spec.scen_name,
        "trial": spec.trial_idx,
        "source_trajectory": source_csv,
        "p0_x": p0[0], "p0_y": p0[1], "p0_z": p0[2],
        "v0_x": v0[0], "v0_y": v0[1], "v0_z": v0[2],
        "trajectory_file": traj_file,
        **m,
    }


# ── Initial-condition generators ─────────────────────────────────────────────

def _random_pursuer_position(
    target_pos: np.ndarray,
    mode: str,
    radius_range: List[float],
    rng: np.random.Generator,
    min_altitude: float = 0.0,
) -> np.ndarray:
    if mode == "uniform_box":
        # Uniform sample in a cube of half-width = radius_range[1].
        # Rejection-sample until Euclidean distance >= radius_range[0] so the
        # lower-bound constraint is respected.  Acceptance rate > 90 % for
        # typical ratio radius_range[0]/radius_range[1] < 0.9.
        half = radius_range[1]
        for _ in range(50):
            offset = rng.uniform(-half, half, size=3)
            if np.linalg.norm(offset) >= radius_range[0]:
                break
        pos = target_pos + offset
        pos[2] = max(min_altitude, pos[2])
        return pos

    # Spherical-shell placement shared by spherical_shell / hemisphere / lower_hemisphere
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
    speed_range: List[float],
    rng: np.random.Generator,
    target_pos: Optional[np.ndarray] = None,
    pursuer_pos: Optional[np.ndarray] = None,
) -> np.ndarray:
    speed = rng.uniform(speed_range[0], speed_range[1])
    if mode == "toward_target" and target_pos is not None and pursuer_pos is not None:
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


# ── Summary statistics ────────────────────────────────────────────────────────

def _summarise(df: pd.DataFrame, algorithms: List[str]) -> pd.DataFrame:
    rows = []
    for algo_name in algorithms:
        for scen_name in df["scenario"].unique():
            sub = df[(df["algorithm"] == algo_name) & (df["scenario"] == scen_name)]
            if sub.empty:
                continue
            n_total = len(sub)
            n_success = int(sub["success"].sum())
            sr = n_success / n_total * 100

            # Bootstrap 95 % CI
            boot = np.array([
                sub["success"].sample(n=n_total, replace=True, random_state=i).mean() * 100
                for i in range(2000)
            ])
            ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

            successful = sub.loc[sub["success"] == True]
            solver_values = sub["solver"].dropna().astype(str).unique() if "solver" in sub.columns else []
            row = {
                "algorithm": algo_name,
                "solver": solver_values[0] if len(solver_values) == 1 else ",".join(solver_values),
                "scenario": scen_name,
                "n_trials": n_total,
                "n_success": n_success,
                "success_rate_pct": sr,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "mean_intercept_time": (
                    successful["intercept_time"].mean() if len(successful) > 0 else float("nan")
                ),
                # Distance at the moment of termination (intercept or timeout) — ALL trials
                "mean_terminal_distance": sub["terminal_distance"].mean(),
                # Closest approach over the full trajectory — success-only and all-trials
                "mean_min_dist_success": (
                    successful["min_distance"].mean()
                    if "min_distance" in sub.columns and len(successful) > 0
                    else float("nan")
                ),
                "mean_min_dist_all": (
                    sub["min_distance"].mean()
                    if "min_distance" in sub.columns
                    else float("nan")
                ),
                "mean_control_effort": sub["control_effort"].mean(),
                "mean_rmse_pos": sub["rmse_pos"].mean() if "rmse_pos" in sub.columns else float("nan"),
                "mean_rmse_vel": sub["rmse_vel"].mean() if "rmse_vel" in sub.columns else float("nan"),
            }
            # Extended metrics if present
            for col in ("max_cmd_acc", "control_smoothness", "solver_feasibility_rate",
                        "saturation_rate", "terminal_speed"):
                if col in sub.columns:
                    row[f"mean_{col}"] = sub[col].mean()
            for col in (
                "narx_ready_rate",
                "narx_mean_trust",
                "narx_max_trust",
                "narx_final_trust",
                "narx_mean_loss",
                "narx_final_loss",
                "narx_mean_ema_loss",
                "narx_final_ema_loss",
                "narx_training_period_steps",
                "narx_training_execution_rate",
                "narx_training_deadline_skip_rate",
                "narx_mean_train_time_s",
                "narx_mean_train_event_time_s",
                "narx_max_train_time_s",
                "narx_mean_infer_time_s",
                "narx_max_infer_time_s",
            ):
                if col in sub.columns:
                    row[col] = sub[col].mean()
            # ── Computation time metrics (realtime feasibility assessment) ─────
            # dt = 0.02 s → 50 Hz control loop → 20 ms budget per solve
            _RT_BUDGET_S = 0.020   # 20 ms realtime budget
            if "mean_solve_time_s" in sub.columns:
                row["mean_solve_time_ms"] = sub["mean_solve_time_s"].mean() * 1000.0
                row["p95_solve_time_ms"]  = sub["mean_solve_time_s"].quantile(0.95) * 1000.0
            else:
                row["mean_solve_time_ms"] = float("nan")
                row["p95_solve_time_ms"]  = float("nan")
            if "max_solve_time_s" in sub.columns:
                row["mean_max_solve_time_ms"] = sub["max_solve_time_s"].mean() * 1000.0
                row["worst_max_solve_time_ms"] = sub["max_solve_time_s"].max() * 1000.0
                # Fraction of trials where every step solved within the RT budget
                row["realtime_feasible_pct"] = float(
                    (sub["max_solve_time_s"] <= _RT_BUDGET_S).mean() * 100.0
                )
            else:
                row["mean_max_solve_time_ms"]  = float("nan")
                row["worst_max_solve_time_ms"] = float("nan")
                row["realtime_feasible_pct"]   = float("nan")
            if "total_compute_time_s" in sub.columns:
                row["mean_total_compute_s"] = sub["total_compute_time_s"].mean()
            else:
                row["mean_total_compute_s"] = float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def _ablation_summary(summary_df: pd.DataFrame, reference: str = "proposed_full") -> pd.DataFrame:
    """Compute relative improvement of *reference* over each other algorithm.

    Returns a DataFrame with columns:
        algorithm, scenario, relative_success_gain,
        relative_distance_reduction, relative_time_reduction,
        relative_effort_reduction, relative_rmse_pos_reduction.
    """
    rows = []
    ref_rows = summary_df[summary_df["algorithm"] == reference]
    if ref_rows.empty:
        return pd.DataFrame()

    for _, other_row in summary_df[summary_df["algorithm"] != reference].iterrows():
        # Match on scenario
        scen = other_row["scenario"]
        ref_match = ref_rows[ref_rows["scenario"] == scen]
        if ref_match.empty:
            continue
        ref = ref_match.iloc[0]

        def _rel(ref_val, other_val, higher_is_better: bool = True) -> float:
            if other_val == 0 or np.isnan(other_val) or np.isnan(ref_val):
                return float("nan")
            if higher_is_better:
                return (ref_val - other_val) / abs(other_val) * 100
            else:
                return (other_val - ref_val) / abs(other_val) * 100

        rows.append({
            "algorithm": other_row["algorithm"],
            "scenario": scen,
            "relative_success_gain_pct": _rel(
                ref["success_rate_pct"], other_row["success_rate_pct"], True),
            "relative_distance_reduction_pct": _rel(
                ref["mean_terminal_distance"], other_row["mean_terminal_distance"], False),
            "relative_time_reduction_pct": _rel(
                ref.get("mean_intercept_time", float("nan")),
                other_row.get("mean_intercept_time", float("nan")), False),
            "relative_effort_reduction_pct": _rel(
                ref["mean_control_effort"], other_row["mean_control_effort"], False),
            "relative_rmse_pos_reduction_pct": _rel(
                ref.get("mean_rmse_pos", float("nan")),
                other_row.get("mean_rmse_pos", float("nan")), False),
        })
    return pd.DataFrame(rows)


# ── Trajectory summary xlsx ───────────────────────────────────────────────────

def _save_trajectory_summary_xlsx(df: pd.DataFrame, out_dir: str) -> None:
    """Write a multi-sheet Excel workbook summarising saved trajectory files.

    Sheets
    ------
    All Trials
        Every trial row: algorithm, scenario, trial index, success flag,
        intercept time, min/terminal distance, control effort, and a
        clickable hyperlink to the trajectory CSV file.
    Successful
        Subset of rows where success == True.
    Failed
        Subset of rows where success == False.
    Per-Algorithm Summary
        Success rate, mean intercept time, mean min distance per algorithm.
    """
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils.dataframe import dataframe_to_rows
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("openpyxl not installed — skipping trajectory_summary.xlsx")
        return

    # ── Columns to include ────────────────────────────────────────────────
    base_cols = [
        "algorithm", "solver", "scenario", "trial", "source_trajectory", "success",
        "intercept_time", "min_distance", "terminal_distance",
        "control_effort", "mean_solve_time_s", "total_compute_time_s",
    ]
    # Include trajectory_file and rmse columns if present
    for extra in ("trajectory_file", "rmse_pos", "rmse_vel", "failure_reason"):
        if extra in df.columns:
            base_cols.append(extra)

    available_cols = [c for c in base_cols if c in df.columns]
    df_out = df[available_cols].copy()

    # ── Per-algorithm summary ─────────────────────────────────────────────
    grp = df.groupby("algorithm")
    summary_rows = []
    for algo, sub in grp:
        n = len(sub)
        n_ok = int(sub["success"].sum())
        ok_sub = sub[sub["success"] == True]
        summary_rows.append({
            "algorithm": algo,
            "n_trials": n,
            "n_success": n_ok,
            "n_failed": n - n_ok,
            "success_rate_%": round(n_ok / n * 100, 2),
            "mean_intercept_time_s": round(ok_sub["intercept_time"].mean(), 3)
                if len(ok_sub) else float("nan"),
            "mean_min_distance_m": round(sub["min_distance"].mean(), 4),
            "mean_control_effort": round(sub["control_effort"].mean(), 3),
            "mean_total_compute_time_s": round(sub["total_compute_time_s"].mean(), 4)
                if "total_compute_time_s" in sub.columns else float("nan"),
        })
    df_summary = pd.DataFrame(summary_rows)

    # ── Build workbook ────────────────────────────────────────────────────
    wb = openpyxl.Workbook()

    HEADER_FILL = PatternFill("solid", fgColor="1F4E79")   # dark blue
    HEADER_FONT = Font(color="FFFFFF", bold=True)
    SUCCESS_FILL = PatternFill("solid", fgColor="C6EFCE")  # light green
    FAIL_FILL    = PatternFill("solid", fgColor="FFC7CE")  # light red
    SUMMARY_FILL = PatternFill("solid", fgColor="DDEBF7")  # light blue

    def _write_sheet(ws, data: pd.DataFrame, sheet_title: str, row_color_col: str = "success") -> None:
        """Write a DataFrame to a worksheet with styled headers and row colours."""
        headers = list(data.columns)
        ws.append(headers)
        for cell in ws[1]:
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
            cell.alignment = Alignment(horizontal="center")

        success_col_idx = headers.index(row_color_col) if row_color_col in headers else -1
        traj_col_idx    = headers.index("trajectory_file") if "trajectory_file" in headers else -1

        for row_data in dataframe_to_rows(data, index=False, header=False):
            ws.append(row_data)
            row_num = ws.max_row
            # Row colour based on success/fail
            if success_col_idx >= 0:
                cell_val = ws.cell(row=row_num, column=success_col_idx + 1).value
                fill = SUCCESS_FILL if cell_val else FAIL_FILL
                for c in range(1, len(headers) + 1):
                    ws.cell(row=row_num, column=c).fill = fill
            # Make trajectory_file column a hyperlink
            if traj_col_idx >= 0:
                cell = ws.cell(row=row_num, column=traj_col_idx + 1)
                traj_rel = str(cell.value or "")
                if traj_rel:
                    abs_path = os.path.join(out_dir, traj_rel)
                    cell.hyperlink = abs_path
                    cell.font = Font(color="0563C1", underline="single")

        # Auto-width
        for col_idx, col_name in enumerate(headers, start=1):
            lengths = [len(str(col_name))]
            lengths.extend(
                len(str(ws.cell(row=r, column=col_idx).value or ""))
                for r in range(2, ws.max_row + 1)
            )
            max_len = max(lengths)
            ws.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 2, 60)

        ws.freeze_panes = "A2"
        ws.title = sheet_title

    # Sheet 1 — All Trials
    ws_all = wb.active
    _write_sheet(ws_all, df_out, "All Trials")

    # Sheet 2 — Successful
    df_ok = df_out[df_out["success"] == True].copy() if "success" in df_out.columns else df_out.iloc[0:0]
    ws_ok = wb.create_sheet("Successful")
    _write_sheet(ws_ok, df_ok, "Successful")

    # Sheet 3 — Failed
    df_fail = df_out[df_out["success"] == False].copy() if "success" in df_out.columns else df_out.iloc[0:0]
    ws_fail = wb.create_sheet("Failed")
    _write_sheet(ws_fail, df_fail, "Failed")

    # Sheet 4 — Per-algorithm summary
    ws_sum = wb.create_sheet("Per-Algorithm Summary")
    for row_data in dataframe_to_rows(df_summary, index=False, header=True):
        ws_sum.append(row_data)
    for cell in ws_sum[1]:
        cell.fill = PatternFill("solid", fgColor="2E75B6")
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center")
    for col_idx in range(1, len(df_summary.columns) + 1):
        max_len = max(
            len(str(ws_sum.cell(row=r, column=col_idx).value or ""))
            for r in range(1, ws_sum.max_row + 1)
        )
        ws_sum.column_dimensions[get_column_letter(col_idx)].width = min(max_len + 2, 40)
    for row_num in range(2, ws_sum.max_row + 1):
        for c in range(1, len(df_summary.columns) + 1):
            ws_sum.cell(row=row_num, column=c).fill = SUMMARY_FILL
    ws_sum.freeze_panes = "A2"

    xlsx_path = os.path.join(out_dir, "trajectory_summary.xlsx")
    wb.save(xlsx_path)
    print(f"Trajectory summary → {xlsx_path}  "
          f"({len(df_out)} trials, {int(df_out['success'].sum()) if 'success' in df_out.columns else '?'} successful)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo evaluation (parallel)")
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(_PROJECT_ROOT, "configs", "monte_carlo_config.yaml"),
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of worker processes. Defaults to os.cpu_count(). Use 1 to run sequentially.",
    )
    parser.add_argument(
        "--method", "--methods", dest="methods", nargs="+", default=None,
        metavar="METHOD",
        help="Run only these ablation/algorithm names instead of all algorithms in the config. "
             "Available: " + " ".join(sorted(_METHOD_ABLATIONS)),
    )
    parser.add_argument(
        "--solver", type=str, choices=["casadi", "acados"], default=None,
        help="Override the optimization solver for MPC methods."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override monte_carlo.n_trials from the config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output.output_dir from the config file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    mc = cfg.monte_carlo

    if args.solver:
        cfg.controller.solver = args.solver
    if args.trials is not None:
        if args.trials <= 0:
            parser.error("--trials must be positive")
        mc.n_trials = args.trials
    if args.output_dir:
        cfg.output.output_dir = args.output_dir
    cfg.controller.acados_export_dir = os.path.join(
        cfg.output.output_dir, "acados_exports"
    )

    # Override algorithm list when --method is given
    if args.methods:
        unknown = [m for m in args.methods if m not in _METHOD_ABLATIONS]
        if unknown:
            parser.error(f"Unknown method(s): {unknown}\nAvailable: {sorted(_METHOD_ABLATIONS)}")
        mc.algorithms = args.methods
    mc.algorithms = _expanded_algorithms(cfg, list(mc.algorithms))
    master_rng = np.random.default_rng(mc.seed)

    n_workers = args.workers if args.workers is not None else cpu_count()
    n_workers = max(1, n_workers)

    out_dir = cfg.output.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # ── Pre-generate all trial specs in the main process ─────────────────
    # This keeps random seed generation deterministic regardless of worker count.
    import copy
    import glob as _glob

    # Pre-resolve CSV file list once (used for every algo when scen == "csv")
    def _resolve_csv_files(mc_cfg) -> List[str]:
        manifest_path = getattr(mc_cfg, "trajectory_manifest_path", None)
        if manifest_path:
            if not os.path.isabs(manifest_path):
                manifest_path = os.path.normpath(os.path.join(_PROJECT_ROOT, manifest_path))
            if os.path.exists(manifest_path):
                with open(manifest_path, "r") as f:
                    lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
                if lines:
                    limit = getattr(mc_cfg, "n_trajectories", 0)
                    return lines[:limit] if limit > 0 else lines
        csv_dir = mc_cfg.trajectory_csv_dir or ""
        if not csv_dir:
            return []
        if not os.path.isabs(csv_dir):
            csv_dir = os.path.normpath(os.path.join(_PROJECT_ROOT, csv_dir))
        files = sorted(_glob.glob(os.path.join(csv_dir, "*.csv")))
        if not files:
            raise FileNotFoundError(
                f"No CSV files found in trajectory_csv_dir: {csv_dir}\n"
                "Check the 'trajectory_csv_dir' setting in your Monte Carlo config."
            )
        limit = getattr(mc_cfg, "n_trajectories", 0)
        files = files[:limit] if limit > 0 else files

        # ── Filter out corrupt / non-trajectory CSV files ─────────────────────
        import pandas as _pd
        _REQUIRED = {"time", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"}
        valid, skipped = [], []
        for _f in files:
            try:
                _hdr = _pd.read_csv(_f, nrows=0, comment="#")
                if _REQUIRED.issubset(set(_hdr.columns)):
                    valid.append(_f)
                else:
                    skipped.append(_f)
            except Exception:
                skipped.append(_f)
        if skipped:
            print(
                f"[run_monte_carlo] WARNING: Skipped {len(skipped)} invalid/corrupt "
                f"CSV file(s) in {csv_dir}."
            )
        if not valid:
            raise FileNotFoundError(
                f"No valid trajectory CSV files found in {csv_dir} "
                f"(all {len(skipped)} file(s) were missing required columns)."
            )
        return valid

    if getattr(mc, "trajectory_manifest_path", None):
        mc.scenarios = ["csv"]

    _csv_files: List[str] = []
    if "csv" in mc.scenarios:
        _csv_files = _resolve_csv_files(mc)

    # ── Pre-generate per-scenario per-trial ICs (shared across algorithms) ──────
    # FAIRNESS FIX: all algorithms for the same trial_idx now face the SAME
    # pursuer initial condition.  Previously the master_rng advanced once per
    # (algo × scenario × trial), so each algorithm received a different seed and
    # therefore a different IC for the same trial index — making the comparison
    # unfair.  Now ICs are generated once per (scenario × trial) and reused for
    # every algorithm, so performance differences reflect the algorithm alone.
    from src.simulation.scenarios import create_scenario as _create_scenario
    trial_ics_per_scenario: Dict[str, List] = {}
    for scen_name in mc.scenarios:
        cfg.scenario.scenario_type = scen_name
        _min_alt = getattr(mc, "pursuer_min_altitude", 0.0)

        # ── Per-trial target starting positions ───────────────────────────────
        # IMPORTANT: place the pursuer relative to each trial's SPECIFIC
        # trajectory start, not the first CSV file's start.  Using only the
        # first file caused the distance constraint to be measured from the
        # wrong reference, so pursuers could end up hundreds of metres from
        # the target they are actually chasing.
        if scen_name == "csv":
            if not _csv_files:
                raise FileNotFoundError(
                    "Scenario type is 'csv' but trajectory_csv_dir is empty or not set."
                )
            _t0_cache: dict = {}
            _trial_t0s: list = []
            for _ti in range(mc.n_trials):
                _tc = _csv_files[_ti % len(_csv_files)]
                if _tc not in _t0_cache:
                    cfg.scenario.trajectory_csv_path = _tc
                    _s = _create_scenario(cfg.scenario, cfg.simulation, seed=0)
                    _t0_cache[_tc] = _s.get_target_state(0.0)[0].copy()
                _trial_t0s.append(_t0_cache[_tc])
        else:
            scenario_tmp = _create_scenario(cfg.scenario, cfg.simulation, seed=0)
            _ref_t0 = scenario_tmp.get_target_state(0.0)[0]
            _trial_t0s = [_ref_t0] * mc.n_trials

        ics: List = []
        for trial_idx in range(mc.n_trials):
            trial_seed = int(master_rng.integers(0, 2 ** 31))
            trial_rng = np.random.default_rng(trial_seed)
            t0_pos = _trial_t0s[trial_idx]
            p0 = _random_pursuer_position(
                t0_pos, mc.pursuer_position_mode, mc.pursuer_radius_range, trial_rng,
                min_altitude=_min_alt,
            )
            v0 = _random_pursuer_velocity(
                mc.pursuer_velocity_mode, mc.pursuer_speed_range, trial_rng, t0_pos, p0
            )
            ics.append((trial_seed, p0.tolist(), v0.tolist()))
        trial_ics_per_scenario[scen_name] = ics

    specs: List[TrialSpec] = []
    for algo_name in mc.algorithms:
        for scen_name in mc.scenarios:
            ics = trial_ics_per_scenario[scen_name]
            for trial_idx in range(mc.n_trials):
                trial_seed, p0, v0 = ics[trial_idx]

                # Clone config so each spec carries its own scenario_type
                cfg_copy = copy.deepcopy(cfg)
                base_algo_name = _base_algo_name(algo_name, cfg_copy)
                if base_algo_name == "mpc_ekf_narx":
                    for variant in _narx_training_variants(cfg_copy):
                        if str(variant["label"]) == algo_name:
                            cfg_copy = _apply_narx_training_variant(cfg_copy, variant)
                            break
                cfg_copy.scenario.scenario_type = scen_name

                # Assign a rotating CSV file per trial so diversity is maintained
                if scen_name == "csv" and _csv_files:
                    cfg_copy.scenario.trajectory_csv_path = _csv_files[trial_idx % len(_csv_files)]

                specs.append(TrialSpec(
                    algo_name=algo_name,
                    scen_name=scen_name,
                    trial_idx=trial_idx,
                    trial_seed=trial_seed,
                    p0=p0,
                    v0=v0,
                    cfg=cfg_copy,
                    base_algo_name=base_algo_name,
                ))

    total = len(specs)
    print("=" * 72)
    print("  MONTE CARLO ROBUSTNESS EVALUATION  — PARALLEL")
    print("=" * 72)
    print(f"  Algorithms    : {mc.algorithms}")
    print(f"  Solver        : {getattr(cfg.controller, 'solver', 'casadi')}")
    print(f"  Scenarios     : {mc.scenarios}")
    print(f"  Trials        : {mc.n_trials} per (algo, scenario)")
    print(f"  Total runs    : {total}")
    print(f"  Worker procs  : {n_workers}  (logical cores available: {cpu_count()})")
    print("=" * 72)

    all_records: List[Dict] = []
    t_start = _time.perf_counter()

    progress_file = _open_progress_stream()
    progress_kwargs = dict(
        desc="Running trials",
        unit="trial",
        file=progress_file,
        dynamic_ncols=True,
        mininterval=0.2,
        smoothing=0.0,
    )
    try:
        if n_workers == 1:
            # Sequential path — useful for debugging / profiling
            with tqdm(total=total, **progress_kwargs) as pbar:
                for spec in specs:
                    all_records.append(_run_trial(spec))
                    pbar.set_postfix_str(
                        f"{spec.algo_name}/{spec.scen_name}#{spec.trial_idx}",
                        refresh=False,
                    )
                    pbar.update(1)
        else:
            # Parallel path
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_run_trial, spec): spec for spec in specs}
                with tqdm(total=total, **progress_kwargs) as pbar:
                    for future in as_completed(futures):
                        spec = futures[future]
                        try:
                            all_records.append(future.result())
                        except Exception as exc:
                            print(f"\n  [ERROR] {spec.algo_name}/{spec.scen_name} "
                                  f"trial {spec.trial_idx}: {exc}", flush=True)
                        finally:
                            pbar.set_postfix_str(
                                f"{spec.algo_name}/{spec.scen_name}#{spec.trial_idx}",
                                refresh=False,
                            )
                            pbar.update(1)
    finally:
        if progress_file not in (sys.stdout, sys.stderr):
            progress_file.close()

    elapsed = _time.perf_counter() - t_start
    print(f"\n  Completed {total} trials in {elapsed:.1f} s  "
          f"({elapsed / total * 1000:.1f} ms/trial avg)")

    # ── Save detailed results ─────────────────────────────────────────────
    df = pd.DataFrame(all_records)
    df.sort_values(["algorithm", "scenario", "trial"], inplace=True)
    detail_path = os.path.join(out_dir, "monte_carlo_detailed.csv")
    df.to_csv(detail_path, index=False)
    print(f"Detailed results → {detail_path}")

    # ── Summary statistics ────────────────────────────────────────────────
    summary_df = _summarise(df, mc.algorithms)
    summary_path = os.path.join(out_dir, "monte_carlo_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 72)
    print("  MONTE CARLO SUMMARY")
    print("=" * 72)
    print(summary_df.to_string(index=False))
    print("=" * 72)
    print(f"Summary → {summary_path}")

    # ── Ablation relative-improvement table ───────────────────────────────
    # Determine which name to treat as the reference (proposed_full or first listed)
    ref_name = "proposed_full" if "proposed_full" in mc.algorithms else mc.algorithms[0]
    ablation_df = _ablation_summary(summary_df, reference=ref_name)
    if not ablation_df.empty:
        ablation_path = os.path.join(out_dir, "ablation_summary.csv")
        ablation_df.to_csv(ablation_path, index=False)
        print(f"\n  Ablation relative-improvement table (ref={ref_name}):")
        print(ablation_df.to_string(index=False))
        print(f"Ablation summary → {ablation_path}")

    # ── Trajectory summary Excel workbook ─────────────────────────────────
    if cfg.output.save_trajectory:
        _save_trajectory_summary_xlsx(df, out_dir)

    print("Done.")


if __name__ == "__main__":
    main()
