"""
Target trajectory scenario generators and CSV loader.

Scenario types:
    - ``straight``  : constant-velocity straight-line motion
    - ``turning``   : horizontal coordinated turn with constant speed
    - ``circular``  : 3-D helical / circular trajectory
    - ``spline``    : cubic spline through user-defined waypoints
    - ``csv``       : load a pre-recorded trajectory from a CSV file

All scenarios expose the same interface:
    get_target_state(t) -> (position, velocity, acceleration)  each (3,)

CSV schema (required columns):
    time, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, acc_x, acc_y, acc_z
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from src.utils.config_schema import ScenarioConfig, SimulationConfig


class TargetScenario:
    """Generate or load a target trajectory and query it at arbitrary time."""

    def __init__(self, scenario_cfg: ScenarioConfig, sim_cfg: SimulationConfig) -> None:
        self._cfg = scenario_cfg
        self._dt = sim_cfg.dt
        self._max_time = sim_cfg.max_time

        self._times: np.ndarray | None = None
        self._positions: np.ndarray | None = None
        self._velocities: np.ndarray | None = None
        self._accelerations: np.ndarray | None = None

        self._build()

    # ── public ────────────────────────────────────────────────────────────

    def get_target_state(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Interpolated (pos, vel, acc) at time *t*."""
        idx = t / self._dt
        i0 = int(np.clip(np.floor(idx), 0, len(self._times) - 2))
        i1 = i0 + 1
        alpha = np.clip(idx - i0, 0.0, 1.0)

        pos = (1 - alpha) * self._positions[i0] + alpha * self._positions[i1]
        vel = (1 - alpha) * self._velocities[i0] + alpha * self._velocities[i1]
        acc = (1 - alpha) * self._accelerations[i0] + alpha * self._accelerations[i1]
        return pos, vel, acc

    @property
    def duration(self) -> float:
        return self._times[-1] if self._times is not None else 0.0

    @property
    def dataframe(self) -> pd.DataFrame:
        """Full trajectory as a DataFrame for logging / export."""
        return pd.DataFrame({
            "time": self._times,
            "pos_x": self._positions[:, 0], "pos_y": self._positions[:, 1], "pos_z": self._positions[:, 2],
            "vel_x": self._velocities[:, 0], "vel_y": self._velocities[:, 1], "vel_z": self._velocities[:, 2],
            "acc_x": self._accelerations[:, 0], "acc_y": self._accelerations[:, 1], "acc_z": self._accelerations[:, 2],
        })

    # ── generators ────────────────────────────────────────────────────────

    def _build(self) -> None:
        kind = self._cfg.scenario_type
        if kind == "straight":
            self._generate_straight()
        elif kind == "turning":
            self._generate_turning()
        elif kind == "circular":
            self._generate_circular()
        elif kind == "spline":
            self._generate_spline()
        elif kind == "csv":
            self._load_csv()
        else:
            raise ValueError(f"Unknown scenario type: {kind}")

    def _time_vector(self) -> np.ndarray:
        return np.arange(0, self._max_time + self._dt, self._dt)

    def _generate_straight(self) -> None:
        t = self._time_vector()
        p0 = np.asarray(self._cfg.target_initial_position)
        v0 = np.asarray(self._cfg.target_initial_velocity)
        N = len(t)
        self._times = t
        self._positions = np.outer(np.ones(N), p0) + np.outer(t, v0)
        self._velocities = np.tile(v0, (N, 1))
        self._accelerations = np.zeros((N, 3))

    def _generate_turning(self) -> None:
        t = self._time_vector()
        p0 = np.asarray(self._cfg.target_initial_position)
        v0 = np.asarray(self._cfg.target_initial_velocity)
        omega = self._cfg.target_turn_rate
        speed_h = np.linalg.norm(v0[:2])  # horizontal speed
        heading0 = np.arctan2(v0[1], v0[0])

        N = len(t)
        pos = np.zeros((N, 3))
        vel = np.zeros((N, 3))
        acc = np.zeros((N, 3))

        for i, ti in enumerate(t):
            theta = heading0 + omega * ti
            vx = speed_h * np.cos(theta)
            vy = speed_h * np.sin(theta)
            vz = v0[2]

            if i == 0:
                pos[i] = p0
            else:
                pos[i] = pos[i - 1] + vel[i - 1] * self._dt

            vel[i] = [vx, vy, vz]
            acc[i] = [-speed_h * omega * np.sin(theta),
                       speed_h * omega * np.cos(theta),
                       0.0]

        self._times = t
        self._positions = pos
        self._velocities = vel
        self._accelerations = acc

    def _generate_circular(self) -> None:
        t = self._time_vector()
        p0 = np.asarray(self._cfg.target_initial_position)
        omega = self._cfg.target_turn_rate
        a_mag = self._cfg.target_acceleration_magnitude
        R = a_mag / (omega ** 2) if omega > 1e-6 else 50.0
        vz = self._cfg.target_initial_velocity[2] if len(self._cfg.target_initial_velocity) > 2 else 0.0

        N = len(t)
        # Circle centre offset
        cx = p0[0]
        cy = p0[1] - R

        pos = np.zeros((N, 3))
        vel = np.zeros((N, 3))
        acc = np.zeros((N, 3))

        for i, ti in enumerate(t):
            angle = omega * ti
            pos[i] = [cx + R * np.sin(angle),
                      cy + R * np.cos(angle),
                      p0[2] + vz * ti]
            vel[i] = [R * omega * np.cos(angle),
                      -R * omega * np.sin(angle),
                      vz]
            acc[i] = [-R * omega ** 2 * np.sin(angle),
                      -R * omega ** 2 * np.cos(angle),
                      0.0]

        self._times = t
        self._positions = pos
        self._velocities = vel
        self._accelerations = acc

    def _generate_spline(self) -> None:
        wps = self._cfg.spline_waypoints
        if len(wps) < 2:
            # Fallback: generate straight trajectory
            self._generate_straight()
            return

        wps = np.asarray(wps, dtype=float)
        n_wp = len(wps)
        # Assign times equally spaced
        t_wp = np.linspace(0, self._max_time, n_wp)

        cs_x = CubicSpline(t_wp, wps[:, 0], bc_type="natural")
        cs_y = CubicSpline(t_wp, wps[:, 1], bc_type="natural")
        cs_z = CubicSpline(t_wp, wps[:, 2], bc_type="natural")

        t = self._time_vector()
        self._times = t
        self._positions = np.column_stack([cs_x(t), cs_y(t), cs_z(t)])
        self._velocities = np.column_stack([cs_x(t, 1), cs_y(t, 1), cs_z(t, 1)])
        self._accelerations = np.column_stack([cs_x(t, 2), cs_y(t, 2), cs_z(t, 2)])

    def _load_csv(self) -> None:
        path = self._cfg.trajectory_csv_path
        if not path:
            raise FileNotFoundError("No trajectory_csv_path specified in config.")
        df = pd.read_csv(path, comment='#')
        # Support both canonical column names (pos_x/vel_x) and
        # simulation-log column names (t_px/t_vx used in trajectory run-logs).
        _col_map = {}
        if "pos_x" not in df.columns and "t_px" in df.columns:
            _col_map = {"t_px": "pos_x", "t_py": "pos_y", "t_pz": "pos_z",
                        "t_vx": "vel_x", "t_vy": "vel_y", "t_vz": "vel_z",
                        "t_ax": "acc_x", "t_ay": "acc_y", "t_az": "acc_z"}
            df = df.rename(columns=_col_map)
        required = ["time", "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"CSV missing required column: {col}")

        self._times = df["time"].values
        self._positions = df[["pos_x", "pos_y", "pos_z"]].values
        self._velocities = df[["vel_x", "vel_y", "vel_z"]].values
        if all(c in df.columns for c in ["acc_x", "acc_y", "acc_z"]):
            self._accelerations = df[["acc_x", "acc_y", "acc_z"]].values
        else:
            # Finite-difference fallback
            self._accelerations = np.zeros_like(self._positions)
            dt_arr = np.diff(self._times)
            for i in range(1, len(self._times)):
                self._accelerations[i] = (self._velocities[i] - self._velocities[i - 1]) / max(dt_arr[i - 1], 1e-9)