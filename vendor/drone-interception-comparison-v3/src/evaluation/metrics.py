"""
Performance metrics computation from simulation logs.

Metrics include:
    - Interception success (bool)
    - Terminal miss distance (m)
    - Time to intercept (s)
    - Total control effort  Σ ‖u‖² · dt
    - Mean / max solver time (s)
    - Solver feasibility rate (fraction of steps with successful solve)
    - Maximum commanded acceleration magnitude (m/s²)
    - Maximum applied acceleration magnitude (m/s²)
    - Control smoothness Σ ‖Δu‖² · dt
    - Actuator saturation rate (fraction of steps at or near a_max)
    - Terminal relative speed (m/s)
    - Estimator RMSE for position, velocity, acceleration, jerk
    - Failure reason
    - Failure category (for Monte Carlo summary)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.simulation.logger import SimulationLogger


def compute_metrics(logger: SimulationLogger) -> Dict[str, Any]:
    """Derive scalar performance metrics from a completed simulation log.

    Parameters
    ----------
    logger : SimulationLogger
        A logger whose simulation has already finished.

    Returns
    -------
    metrics : dict
        All existing keys are preserved; new keys are added only when the
        relevant data columns exist in the log.
    """
    df = logger.to_dataframe()
    if df.empty:
        return {"success": False, "failure_reason": "empty_log", "failure_category": "empty_log"}

    success = logger.success
    intercept_time = logger.intercept_time

    # Terminal miss distance
    terminal_distance = df["distance"].iloc[-1]
    min_distance = df["distance"].min()

    # Control effort:  Σ ‖u_cmd‖² · dt
    dt = df["time"].diff().median() if len(df) > 1 else 0.02
    cmd = df[["cmd_ax", "cmd_ay", "cmd_az"]].values
    cmd_norms = np.linalg.norm(cmd, axis=1)
    control_effort = float(np.sum(cmd_norms ** 2) * dt)

    # Maximum commanded acceleration
    max_cmd_acc = float(cmd_norms.max()) if len(cmd_norms) > 0 else float("nan")

    # Maximum applied acceleration
    if all(c in df.columns for c in ["p_ax", "p_ay", "p_az"]):
        applied_acc = df[["p_ax", "p_ay", "p_az"]].values
        max_applied_acc = float(np.linalg.norm(applied_acc, axis=1).max())
    else:
        max_applied_acc = float("nan")

    # Control smoothness:  Σ ‖Δu‖² · dt
    if len(cmd) > 1:
        dcmd = np.diff(cmd, axis=0)
        control_smoothness = float(np.sum(np.sum(dcmd ** 2, axis=1)) * dt)
    else:
        control_smoothness = 0.0

    # Solver & Pipeline timing metrics
    if "ctrl_solve_time_s" in df.columns:
        stimes = df["ctrl_solve_time_s"].dropna().values
        mean_solve_time = float(np.mean(stimes)) if len(stimes) > 0 else float("nan")
        max_solve_time = float(np.max(stimes)) if len(stimes) > 0 else float("nan")
        total_compute_time = float(np.sum(stimes)) if len(stimes) > 0 else float("nan")
        p50_solve_time = float(np.percentile(stimes, 50)) if len(stimes) > 0 else float("nan")
        p95_solve_time = float(np.percentile(stimes, 95)) if len(stimes) > 0 else float("nan")
        p99_solve_time = float(np.percentile(stimes, 99)) if len(stimes) > 0 else float("nan")
    else:
        mean_solve_time = max_solve_time = total_compute_time = float("nan")
        p50_solve_time = p95_solve_time = p99_solve_time = float("nan")

    if "ctrl_control_pipeline_time_s" in df.columns:
        pipe_times = df["ctrl_control_pipeline_time_s"].dropna().values
        if len(pipe_times) > 0:
            p50_control_pipeline_time_s = float(np.percentile(pipe_times, 50))
            p95_control_pipeline_time_s = float(np.percentile(pipe_times, 95))
            p99_control_pipeline_time_s = float(np.percentile(pipe_times, 99))
            max_control_pipeline_time_s = float(np.max(pipe_times))
            control_pipeline_deadline_miss_rate = float(np.mean(pipe_times > 0.02))
        else:
            p50_control_pipeline_time_s = max_control_pipeline_time_s = float("nan")
            p95_control_pipeline_time_s = p99_control_pipeline_time_s = float("nan")
            control_pipeline_deadline_miss_rate = float("nan")
    else:
        p50_control_pipeline_time_s = max_control_pipeline_time_s = float("nan")
        p95_control_pipeline_time_s = p99_control_pipeline_time_s = float("nan")
        control_pipeline_deadline_miss_rate = float("nan")

    if "ctrl_solver_success" in df.columns:
        solver_feasibility_rate = float(df["ctrl_solver_success"].mean())
    else:
        solver_feasibility_rate = float("nan")

    narx_metrics: Dict[str, float] = {}
    if "ctrl_narx_ready" in df.columns:
        ready = pd.to_numeric(df["ctrl_narx_ready"], errors="coerce").dropna()
        narx_metrics["narx_ready_rate"] = float(ready.mean()) if not ready.empty else float("nan")
    if "ctrl_narx_trust" in df.columns:
        trust = pd.to_numeric(df["ctrl_narx_trust"], errors="coerce").dropna()
        narx_metrics["narx_mean_trust"] = float(trust.mean()) if not trust.empty else float("nan")
        narx_metrics["narx_max_trust"] = float(trust.max()) if not trust.empty else float("nan")
        narx_metrics["narx_final_trust"] = float(trust.iloc[-1]) if not trust.empty else float("nan")
    if "ctrl_narx_loss" in df.columns:
        loss = pd.to_numeric(df["ctrl_narx_loss"], errors="coerce").dropna()
        narx_metrics["narx_mean_loss"] = float(loss.mean()) if not loss.empty else float("nan")
        narx_metrics["narx_final_loss"] = float(loss.iloc[-1]) if not loss.empty else float("nan")
    if "ctrl_narx_ema_loss" in df.columns:
        ema_loss = pd.to_numeric(df["ctrl_narx_ema_loss"], errors="coerce").dropna()
        narx_metrics["narx_mean_ema_loss"] = float(ema_loss.mean()) if not ema_loss.empty else float("nan")
        narx_metrics["narx_final_ema_loss"] = float(ema_loss.iloc[-1]) if not ema_loss.empty else float("nan")
    executed = pd.Series(dtype=float)
    if "ctrl_narx_training_period_steps" in df.columns:
        period = pd.to_numeric(df["ctrl_narx_training_period_steps"], errors="coerce").dropna()
        narx_metrics["narx_training_period_steps"] = (
            float(period.mode().iloc[0]) if not period.empty else float("nan")
        )
    if "ctrl_narx_training_executed" in df.columns:
        executed = pd.to_numeric(df["ctrl_narx_training_executed"], errors="coerce").dropna()
        narx_metrics["narx_training_execution_rate"] = (
            float(executed.mean()) if not executed.empty else float("nan")
        )
    if "ctrl_narx_training_skipped_deadline" in df.columns:
        skipped = pd.to_numeric(df["ctrl_narx_training_skipped_deadline"], errors="coerce").dropna()
        narx_metrics["narx_training_deadline_skip_rate"] = (
            float(skipped.mean()) if not skipped.empty else float("nan")
        )
    if "ctrl_narx_train_time_s" in df.columns:
        train_time = pd.to_numeric(df["ctrl_narx_train_time_s"], errors="coerce").dropna()
        narx_metrics["narx_mean_train_time_s"] = float(train_time.mean()) if not train_time.empty else float("nan")
        narx_metrics["narx_max_train_time_s"] = float(train_time.max()) if not train_time.empty else float("nan")
        if not train_time.empty and not executed.empty:
            train_events = train_time.loc[executed.reindex(train_time.index, fill_value=0).astype(bool)]
            narx_metrics["narx_mean_train_event_time_s"] = (
                float(train_events.mean()) if not train_events.empty else 0.0
            )
    if "ctrl_narx_infer_time_s" in df.columns:
        infer_time = pd.to_numeric(df["ctrl_narx_infer_time_s"], errors="coerce").dropna()
        narx_metrics["narx_mean_infer_time_s"] = float(infer_time.mean()) if not infer_time.empty else float("nan")
        narx_metrics["narx_max_infer_time_s"] = float(infer_time.max()) if not infer_time.empty else float("nan")

    # Actuator saturation rate — fraction of steps where ‖u_cmd‖ ≥ 0.99·a_max
    # We cannot directly access a_max here, so we use the empirical maximum as
    # an approximation; the exact bound is not stored in the log.
    if len(cmd_norms) > 0 and max_cmd_acc > 0:
        sat_threshold = 0.99 * max_cmd_acc
        saturation_rate = float(np.mean(cmd_norms >= sat_threshold))
    else:
        saturation_rate = float("nan")

    # Terminal relative speed
    if all(c in df.columns for c in ["p_vx", "t_vx"]):
        last = df.iloc[-1]
        v_rel = np.array([
            last["p_vx"] - last["t_vx"],
            last["p_vy"] - last["t_vy"],
            last["p_vz"] - last["t_vz"],
        ])
        terminal_speed = float(np.linalg.norm(v_rel))
    else:
        terminal_speed = float("nan")

    # Failure category
    failure_category = _categorise_failure(logger.failure_reason, success)

    # Estimator RMSE
    est_rmse = compute_estimator_rmse(df)

    return {
        "success": success,
        "intercept_time": intercept_time,
        "terminal_distance": terminal_distance,
        "min_distance": min_distance,
        "control_effort": control_effort,
        "max_cmd_acc": max_cmd_acc,
        "max_applied_acc": max_applied_acc,
        "control_smoothness": control_smoothness,
        "mean_solve_time_s": mean_solve_time,
        "max_solve_time_s": max_solve_time,
        "total_compute_time_s": total_compute_time,
        "p50_solve_time_s": p50_solve_time,
        "p95_solve_time_s": p95_solve_time,
        "p99_solve_time_s": p99_solve_time,
        "p50_control_pipeline_time_s": p50_control_pipeline_time_s,
        "p95_control_pipeline_time_s": p95_control_pipeline_time_s,
        "p99_control_pipeline_time_s": p99_control_pipeline_time_s,
        "max_control_pipeline_time_s": max_control_pipeline_time_s,
        "control_pipeline_deadline_miss_rate": control_pipeline_deadline_miss_rate,
        "solver_feasibility_rate": solver_feasibility_rate,
        "saturation_rate": saturation_rate,
        "terminal_speed": terminal_speed,
        "failure_reason": logger.failure_reason,
        "failure_category": failure_category,
        **narx_metrics,
        **est_rmse,
    }


def _categorise_failure(reason: str, success: bool) -> str:
    """Map a free-form failure reason string to a short category label."""
    if success:
        return "success"
    if not reason:
        return "timeout"
    r = reason.lower()
    if "timeout" in r or "max_time" in r:
        return "timeout"
    if "altitude" in r or "ground" in r or "crash" in r:
        return "altitude_violation"
    if "diverge" in r or "unstable" in r:
        return "divergence"
    if "solver" in r or "infeasible" in r:
        return "solver_failure"
    return "other"


def compute_estimator_rmse(df: pd.DataFrame) -> Dict[str, float]:
    """Compute RMSE of the estimator vs. ground truth.

    Returns dict with keys ``rmse_pos``, ``rmse_vel``, ``rmse_acc``, ``rmse_jerk``.
    All values fall back to NaN when the required columns are absent.
    """
    result: Dict[str, float] = {}

    # Position RMSE
    if all(c in df.columns for c in ["t_px", "te_px"]):
        err_pos = np.sqrt(
            (df["t_px"] - df["te_px"]) ** 2
            + (df["t_py"] - df["te_py"]) ** 2
            + (df["t_pz"] - df["te_pz"]) ** 2
        )
        result["rmse_pos"] = float(err_pos.mean())
    else:
        result["rmse_pos"] = float("nan")

    # Velocity RMSE
    if all(c in df.columns for c in ["t_vx", "te_vx"]):
        err_vel = np.sqrt(
            (df["t_vx"] - df["te_vx"]) ** 2
            + (df["t_vy"] - df["te_vy"]) ** 2
            + (df["t_vz"] - df["te_vz"]) ** 2
        )
        result["rmse_vel"] = float(err_vel.mean())
    else:
        result["rmse_vel"] = float("nan")

    # Acceleration RMSE
    if all(c in df.columns for c in ["t_ax", "te_ax"]):
        err_acc = np.sqrt(
            (df["t_ax"] - df["te_ax"]) ** 2
            + (df["t_ay"] - df["te_ay"]) ** 2
            + (df["t_az"] - df["te_az"]) ** 2
        )
        result["rmse_acc"] = float(err_acc.mean())
    else:
        result["rmse_acc"] = float("nan")

    # Jerk RMSE — available only when ground-truth jerk columns are present
    if all(c in df.columns for c in ["t_jx", "te_jx"]):
        err_jerk = np.sqrt(
            (df["t_jx"] - df["te_jx"]) ** 2
            + (df["t_jy"] - df["te_jy"]) ** 2
            + (df["t_jz"] - df["te_jz"]) ** 2
        )
        result["rmse_jerk"] = float(err_jerk.mean())
    else:
        result["rmse_jerk"] = float("nan")

    return result


# ── Extended metric functions (Task 5) ───────────────────────────────────────

def compute_miss_distance_cdf(
    miss_distances: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Compute the empirical CDF of miss distances at specified thresholds.

    Parameters
    ----------
    miss_distances : array-like, shape (N,)
        Terminal miss distances from N trials [m].
    thresholds : array-like, shape (K,)
        Distance thresholds at which to evaluate the CDF [m].

    Returns
    -------
    cdf_values : np.ndarray, shape (K,)
        ``cdf_values[k]`` is the fraction of trials with miss distance
        ≤ ``thresholds[k]``.

    Examples
    --------
    >>> d = np.array([0.2, 0.4, 0.6, 0.8, 1.2])
    >>> compute_miss_distance_cdf(d, np.array([0.5, 1.0]))
    array([0.4, 0.8])
    """
    miss_distances = np.asarray(miss_distances, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    cdf_values = np.array(
        [float(np.mean(miss_distances <= t)) for t in thresholds]
    )
    return cdf_values


def compute_success_rate_vs_parameter(
    results_df: "pd.DataFrame",
    param_col: str,
    threshold: float,
    dist_col: str = "terminal_distance",
) -> "pd.DataFrame":
    """Group results by a swept parameter and compute success rate per level.

    A trial is counted as successful when its terminal miss distance is
    ≤ *threshold* metres.  If the results already contain a boolean
    ``success`` column it is used directly (overrides *threshold*).

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain *param_col* and either ``success`` (bool) or
        *dist_col* (float).
    param_col : str
        Column name of the parameter being swept (e.g. ``"wind_magnitude"``).
    threshold : float
        Success distance threshold [m].  Ignored when ``success`` column
        is present.
    dist_col : str
        Column holding the terminal miss distance.  Default
        ``"terminal_distance"``.

    Returns
    -------
    summary : pd.DataFrame
        Columns: ``param_col``, ``n_trials``, ``n_success``,
        ``success_rate_pct``, ``sr_ci_lower``, ``sr_ci_upper``.
        The 95 % Wilson confidence interval bounds are included.
    """
    import pandas as pd

    df = results_df.copy()
    if "success" not in df.columns:
        if dist_col not in df.columns:
            raise KeyError(
                f"DataFrame must have a 'success' column or '{dist_col}' column."
            )
        df["success"] = df[dist_col] <= threshold

    rows = []
    for param_val, group in df.groupby(param_col, sort=True):
        n = len(group)
        k = int(group["success"].sum())
        sr = k / n * 100.0 if n > 0 else float("nan")
        # Wilson 95 % CI
        if n > 0:
            # Wilson score 95 % confidence interval (no external dependency)
            p_hat = k / n
            z = 1.959964  # z_{0.975}
            denom = 1.0 + z ** 2 / n
            centre = (p_hat + z ** 2 / (2 * n)) / denom
            half = z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
            lo, hi = (centre - half) * 100.0, (centre + half) * 100.0
        else:
            lo = hi = float("nan")
        rows.append({
            param_col: param_val,
            "n_trials": n,
            "n_success": k,
            "success_rate_pct": sr,
            "sr_ci_lower": lo,
            "sr_ci_upper": hi,
        })
    return pd.DataFrame(rows)


def compute_solver_stats(
    solve_times: np.ndarray,
    realtime_budget_s: float = 0.05,
) -> Dict[str, Any]:
    """Compute summary statistics for MPC solver wall-clock times.

    Parameters
    ----------
    solve_times : array-like, shape (N,)
        Solver wall-clock times in seconds from N control steps.
    realtime_budget_s : float
        Real-time budget in seconds.  Default 50 ms.  Any solve exceeding
        this value is flagged in ``budget_violations``.

    Returns
    -------
    stats : dict
        Keys:
        * ``mean_s`` — mean solve time [s]
        * ``std_s``  — standard deviation [s]
        * ``max_s``  — worst-case solve time [s]
        * ``p99_s``  — 99th-percentile solve time [s]
        * ``realtime_budget_s`` — the budget used for violation check [s]
        * ``budget_violations`` — number of steps exceeding budget
        * ``budget_violation_rate`` — fraction of steps exceeding budget
        * ``exceeds_budget`` — ``True`` if *any* step exceeded the budget
    """
    solve_times = np.asarray(solve_times, dtype=float)
    solve_times = solve_times[~np.isnan(solve_times)]
    if len(solve_times) == 0:
        return {
            "mean_s": float("nan"),
            "std_s": float("nan"),
            "max_s": float("nan"),
            "p99_s": float("nan"),
            "realtime_budget_s": realtime_budget_s,
            "budget_violations": 0,
            "budget_violation_rate": float("nan"),
            "exceeds_budget": False,
        }
    violations = int(np.sum(solve_times > realtime_budget_s))
    return {
        "mean_s": float(np.mean(solve_times)),
        "std_s": float(np.std(solve_times)),
        "max_s": float(np.max(solve_times)),
        "p99_s": float(np.percentile(solve_times, 99)),
        "realtime_budget_s": realtime_budget_s,
        "budget_violations": violations,
        "budget_violation_rate": float(violations / len(solve_times)),
        "exceeds_budget": violations > 0,
    }


def compute_estimation_convergence(
    ekf_errors_over_time: np.ndarray,
    convergence_threshold_m: float = 0.5,
) -> Dict[str, Any]:
    """Compute EKF position RMSE convergence curve across trials.

    Parameters
    ----------
    ekf_errors_over_time : np.ndarray, shape (T, n_trials)
        Per-step EKF position RMSE values.  Rows are time steps, columns are
        trials.  NaN values (e.g. from different trajectory lengths) are
        excluded from the mean/std at each step.
    convergence_threshold_m : float
        Convergence is declared when the mean error curve first drops below
        this value [m].  Default 0.5 m.

    Returns
    -------
    result : dict
        Keys:
        * ``mean_curve``  — shape (T,) mean RMSE across trials at each step
        * ``std_curve``   — shape (T,) std of RMSE across trials at each step
        * ``convergence_step``  — first step index where mean < threshold
          (``None`` if never reached)
        * ``convergence_threshold_m``  — the threshold used
    """
    arr = np.asarray(ekf_errors_over_time, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    mean_curve = np.nanmean(arr, axis=1)
    std_curve = np.nanstd(arr, axis=1)

    below = np.where(mean_curve < convergence_threshold_m)[0]
    convergence_step: Optional[int] = int(below[0]) if len(below) > 0 else None

    return {
        "mean_curve": mean_curve,
        "std_curve": std_curve,
        "convergence_step": convergence_step,
        "convergence_threshold_m": convergence_threshold_m,
    }
