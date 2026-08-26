"""
Per-timestep simulation logger.

Records all relevant quantities at each integration step into a list of dicts,
then converts to a :class:`pandas.DataFrame` for analysis and CSV export.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class SimulationLogger:
    """Accumulates per-step records during a simulation run."""

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []
        self._success: bool = False
        self._intercept_time: float = float("nan")
        self._failure_reason: str = ""

    def log_step(
        self,
        t: float,
        step: int,
        pursuer_pos: np.ndarray,
        pursuer_vel: np.ndarray,
        pursuer_applied_acc: np.ndarray,
        target_true_pos: np.ndarray,
        target_true_vel: np.ndarray,
        target_true_acc: np.ndarray,
        target_est: np.ndarray,
        target_meas: Optional[np.ndarray],
        commanded_acc: np.ndarray,
        wind: np.ndarray,
        distance: float,
        ctrl_info: Dict[str, Any],
    ) -> None:
        rec: Dict[str, Any] = {
            "time": t,
            "step": step,
            # Pursuer true state
            "p_px": pursuer_pos[0], "p_py": pursuer_pos[1], "p_pz": pursuer_pos[2],
            "p_vx": pursuer_vel[0], "p_vy": pursuer_vel[1], "p_vz": pursuer_vel[2],
            "p_ax": pursuer_applied_acc[0], "p_ay": pursuer_applied_acc[1], "p_az": pursuer_applied_acc[2],
            # Target true state
            "t_px": target_true_pos[0], "t_py": target_true_pos[1], "t_pz": target_true_pos[2],
            "t_vx": target_true_vel[0], "t_vy": target_true_vel[1], "t_vz": target_true_vel[2],
            "t_ax": target_true_acc[0], "t_ay": target_true_acc[1], "t_az": target_true_acc[2],
            # Target estimated state
            "te_px": target_est[0], "te_py": target_est[1], "te_pz": target_est[2],
            "te_vx": target_est[3], "te_vy": target_est[4], "te_vz": target_est[5],
            "te_ax": target_est[6], "te_ay": target_est[7], "te_az": target_est[8],
            "te_jx": target_est[9], "te_jy": target_est[10], "te_jz": target_est[11],
            # Measurement available
            "meas_available": target_meas is not None,
            # Control
            "cmd_ax": commanded_acc[0], "cmd_ay": commanded_acc[1], "cmd_az": commanded_acc[2],
            # Wind
            "wind_x": wind[0], "wind_y": wind[1], "wind_z": wind[2],
            # Miss distance
            "distance": distance,
        }
        # Solver info
        for k, v in ctrl_info.items():
            rec[f"ctrl_{k}"] = v

        self._records.append(rec)

    def set_outcome(self, success: bool, intercept_time: float = float("nan"),
                    failure_reason: str = "") -> None:
        self._success = success
        self._intercept_time = intercept_time
        self._failure_reason = failure_reason

    @property
    def success(self) -> bool:
        return self._success

    @property
    def intercept_time(self) -> float:
        return self._intercept_time

    @property
    def failure_reason(self) -> str:
        return self._failure_reason

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)

    def reset(self) -> None:
        self._records.clear()
        self._success = False
        self._intercept_time = float("nan")
        self._failure_reason = ""
