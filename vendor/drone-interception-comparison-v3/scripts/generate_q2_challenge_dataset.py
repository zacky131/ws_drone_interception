#!/usr/bin/env python3
"""Generate a physics-driven quadrotor trajectory challenge dataset.

The generator reuses the repository's ``Quadrotor6DOFPursuer`` as the target
plant.  Each trajectory family produces a jerk-limited world-frame
acceleration command.  The 13-state rigid-body plant then applies thrust,
attitude, angular-rate, torque, speed, and ground constraints before the
result is written in the CSV schema consumed by ``TargetScenario``:

    time, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z,
    acc_x, acc_y, acc_z

Extra diagnostic columns (commanded acceleration, jerk, quaternion, angular
rate, and maneuver phase) are retained but ignored by the existing loader.

Typical use from the repository root:

    python scripts/generate_q2_challenge_dataset.py \
        --output data/q2_challenge_v1/generated_6dof \
        --per-family 25 --seed 20260729 --dt 0.02 --duration 12 --plots

This creates 8 families x 25 = 200 trajectories, plus metadata and SHA-256
manifests.  Dataset selection is independent of controller performance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Repository import setup
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
_CANDIDATE_ROOTS = [_THIS_FILE.parent, _THIS_FILE.parent.parent, Path.cwd()]
_PROJECT_ROOT: Path | None = None
for _candidate in _CANDIDATE_ROOTS:
    if (_candidate / "src" / "dynamics" / "quadrotor_6dof.py").exists():
        _PROJECT_ROOT = _candidate
        break
if _PROJECT_ROOT is None:
    raise RuntimeError(
        "Could not find repository root containing src/dynamics/quadrotor_6dof.py. "
        "Place this script in <repo>/scripts/ or run it from the repository root."
    )
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dynamics.quadrotor_6dof import Quadrotor6DOFPursuer  # noqa: E402
from src.utils.config_schema import PursuerConfig  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - simple fallback
    def tqdm(iterable: Iterable, **_: object) -> Iterable:
        return iterable


G = 9.81
FAMILIES = (
    "variable_radius_turn",
    "s_turn_chicane",
    "helical_reversal",
    "pop_up_dive",
    "abrupt_axis_switch",
    "minimum_jerk_waypoints",
    "rotating_acceleration",
    "mixed_mode_shift",
)


@dataclass(frozen=True)
class GenerationConfig:
    dt: float = 0.02
    duration: float = 12.0
    per_family: int = 25
    seed: int = 20260729
    max_speed: float = 20.0
    max_acceleration: float = 14.0
    max_acceleration_per_axis: float = 12.0
    max_jerk: float = 30.0
    min_altitude: float = 5.0
    max_altitude: float = 80.0
    mass: float = 1.5
    inertia_x: float = 0.02
    inertia_y: float = 0.02
    inertia_z: float = 0.04
    attitude_time_constant: float = 0.08
    max_thrust_to_weight_ratio: float = 2.5
    max_angular_rate: float = 9.0
    max_torque_per_axis: float = 1.0
    disturbance_std: float = 0.0
    save_plots: bool = False


@dataclass
class CommandContext:
    """State exposed to maneuver command generators."""

    t: float
    dt: float
    position: np.ndarray
    velocity: np.ndarray
    applied_acceleration: np.ndarray
    rng: np.random.Generator


@dataclass
class Profile:
    command: Callable[[CommandContext], tuple[np.ndarray, int]]
    parameters: dict[str, float | int | list[float] | list[list[float]]]
    shift_times: list[float]


class JerkLimiter:
    """Component-wise and norm acceleration limiter with jerk constraints."""

    def __init__(self, max_jerk: float, max_accel_axis: float, max_accel_norm: float):
        self.max_jerk = float(max_jerk)
        self.max_accel_axis = float(max_accel_axis)
        self.max_accel_norm = float(max_accel_norm)
        self.previous = np.zeros(3, dtype=float)

    def reset(self, initial: np.ndarray | None = None) -> None:
        self.previous = np.zeros(3) if initial is None else np.asarray(initial, dtype=float).copy()

    def step(self, desired: np.ndarray, dt: float) -> np.ndarray:
        desired = np.asarray(desired, dtype=float)
        desired = np.clip(desired, -self.max_accel_axis, self.max_accel_axis)
        norm = np.linalg.norm(desired)
        if norm > self.max_accel_norm:
            desired = desired * (self.max_accel_norm / max(norm, 1e-12))

        max_delta = self.max_jerk * dt
        delta = np.clip(desired - self.previous, -max_delta, max_delta)
        limited = self.previous + delta
        norm = np.linalg.norm(limited)
        if norm > self.max_accel_norm:
            limited = limited * (self.max_accel_norm / max(norm, 1e-12))
        self.previous = limited
        return limited.copy()


def _unit_xy(v: np.ndarray, fallback_heading: float = 0.0) -> np.ndarray:
    vec = np.asarray(v[:2], dtype=float)
    n = np.linalg.norm(vec)
    if n < 1e-6:
        return np.array([math.cos(fallback_heading), math.sin(fallback_heading)])
    return vec / n


def _left_normal_xy(v: np.ndarray, fallback_heading: float = 0.0) -> np.ndarray:
    u = _unit_xy(v, fallback_heading)
    return np.array([-u[1], u[0]])


def _smooth_pulse(t: float, start: float, end: float, edge: float) -> float:
    """Smooth unit pulse using tanh transitions."""
    edge = max(edge, 1e-3)
    return 0.5 * (math.tanh((t - start) / edge) - math.tanh((t - end) / edge))


def _minimum_jerk_blend(s: float) -> tuple[float, float, float]:
    """Return position, first derivative, second derivative blend on s in [0,1]."""
    s = float(np.clip(s, 0.0, 1.0))
    h = 10*s**3 - 15*s**4 + 6*s**5
    dh = 30*s**2 - 60*s**3 + 30*s**4
    ddh = 60*s - 180*s**2 + 120*s**3
    return h, dh, ddh


def build_profile(family: str, rng: np.random.Generator, duration: float) -> Profile:
    """Create one randomly parameterized maneuver profile."""
    if family == "variable_radius_turn":
        base_lat = rng.uniform(3.0, 7.5)
        modulation = rng.uniform(0.25, 0.65)
        freq = rng.uniform(0.12, 0.35)
        direction = int(rng.choice([-1, 1]))
        shift = rng.uniform(0.42, 0.62) * duration

        def command(ctx: CommandContext) -> tuple[np.ndarray, int]:
            normal = _left_normal_xy(ctx.velocity)
            sign = direction if ctx.t < shift else -direction
            lat = base_lat * (1.0 + modulation * math.sin(2*math.pi*freq*ctx.t))
            vertical = 1.2 * math.sin(2*math.pi*0.16*ctx.t)
            return np.array([sign*lat*normal[0], sign*lat*normal[1], vertical]), int(ctx.t >= shift)

        return Profile(command, {"base_lat": base_lat, "modulation": modulation,
                                 "frequency_hz": freq, "initial_direction": direction}, [shift])

    if family == "s_turn_chicane":
        amp = rng.uniform(5.0, 9.0)
        f0 = rng.uniform(0.18, 0.32)
        f1 = rng.uniform(0.45, 0.75)
        shift = rng.uniform(0.45, 0.60) * duration
        vertical_amp = rng.uniform(0.5, 2.0)

        def command(ctx: CommandContext) -> tuple[np.ndarray, int]:
            normal = _left_normal_xy(ctx.velocity)
            f = f0 if ctx.t < shift else f1
            phase_t = ctx.t if ctx.t < shift else shift + (ctx.t-shift) * (f1/f0)
            lateral = amp * math.sin(2*math.pi*f*phase_t)
            vertical = vertical_amp * math.sin(2*math.pi*0.22*ctx.t + 0.5)
            return np.array([lateral*normal[0], lateral*normal[1], vertical]), int(ctx.t >= shift)

        return Profile(command, {"amplitude": amp, "frequency_before_hz": f0,
                                 "frequency_after_hz": f1, "vertical_amplitude": vertical_amp}, [shift])

    if family == "helical_reversal":
        lateral = rng.uniform(4.5, 8.0)
        climb = rng.uniform(1.5, 3.5)
        omega = rng.uniform(0.45, 0.9)
        shift = rng.uniform(0.45, 0.58) * duration
        direction = int(rng.choice([-1, 1]))

        def command(ctx: CommandContext) -> tuple[np.ndarray, int]:
            normal = _left_normal_xy(ctx.velocity)
            sign = direction if ctx.t < shift else -direction
            z_cmd = climb if ctx.t < shift else -1.2*climb
            z_cmd += 0.8*math.sin(omega*ctx.t)
            return np.array([sign*lateral*normal[0], sign*lateral*normal[1], z_cmd]), int(ctx.t >= shift)

        return Profile(command, {"lateral_acceleration": lateral, "vertical_acceleration": climb,
                                 "modulation_rad_s": omega, "initial_direction": direction}, [shift])

    if family == "pop_up_dive":
        up_start = rng.uniform(0.18, 0.28) * duration
        up_end = up_start + rng.uniform(0.12, 0.18) * duration
        down_start = up_end + rng.uniform(0.08, 0.14) * duration
        down_end = down_start + rng.uniform(0.14, 0.20) * duration
        vertical = rng.uniform(6.0, 10.5)
        lateral = rng.uniform(2.5, 6.0)
        direction = int(rng.choice([-1, 1]))

        def command(ctx: CommandContext) -> tuple[np.ndarray, int]:
            normal = _left_normal_xy(ctx.velocity)
            up = vertical * _smooth_pulse(ctx.t, up_start, up_end, 0.12)
            down = -1.15*vertical * _smooth_pulse(ctx.t, down_start, down_end, 0.12)
            lat = direction*lateral * _smooth_pulse(ctx.t, up_end, down_end, 0.18)
            phase = 0 if ctx.t < up_start else 1 if ctx.t < down_start else 2
            return np.array([lat*normal[0], lat*normal[1], up+down]), phase

        return Profile(command, {"vertical_acceleration": vertical, "lateral_acceleration": lateral},
                       [up_start, up_end, down_start, down_end])

    if family == "abrupt_axis_switch":
        interval = rng.uniform(0.75, 1.35)
        magnitude = rng.uniform(5.0, 10.0)
        n_segments = int(math.ceil(duration / interval)) + 1
        candidates = np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
            [0.7, 0.7, 0], [-0.7, 0.7, 0], [0.7, -0.7, 0],
            [0, 0, 0.75], [0, 0, -0.75],
        ], dtype=float)
        sequence = []
        previous = None
        for _ in range(n_segments):
            choices = [c for c in candidates if previous is None or not np.allclose(c, -previous)]
            vec = choices[int(rng.integers(0, len(choices)))]
            sequence.append(vec)
            previous = vec

        def command(ctx: CommandContext) -> tuple[np.ndarray, int]:
            idx = min(int(ctx.t // interval), len(sequence)-1)
            vec = sequence[idx]
            return magnitude * vec, idx

        return Profile(command, {"interval_s": interval, "magnitude": magnitude,
                                 "axis_sequence": [v.tolist() for v in sequence]},
                       [interval*i for i in range(1, n_segments) if interval*i < duration])

    if family == "minimum_jerk_waypoints":
        n_wp = int(rng.integers(5, 8))
        t_wp = np.linspace(0.0, duration, n_wp)
        # Relative waypoint pattern; actual initial position is inserted at runtime.
        increments = np.column_stack([
            rng.uniform(10.0, 22.0, n_wp-1),
            rng.uniform(-18.0, 18.0, n_wp-1),
            rng.uniform(-7.0, 7.0, n_wp-1),
        ])
        rel_wp = np.vstack([np.zeros(3), np.cumsum(increments, axis=0)])
        rel_wp[:, 2] -= rel_wp[0, 2]
        kp = rng.uniform(1.1, 1.8)
        kd = rng.uniform(1.5, 2.5)
        origin_holder: list[np.ndarray] = []

        def reference(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
            seg = min(np.searchsorted(t_wp, t, side="right") - 1, n_wp-2)
            seg = max(seg, 0)
            t0, t1 = t_wp[seg], t_wp[seg+1]
            T = max(t1-t0, 1e-6)
            s = (t-t0)/T
            h, dh, ddh = _minimum_jerk_blend(s)
            delta = rel_wp[seg+1] - rel_wp[seg]
            p = rel_wp[seg] + h*delta
            v = (dh/T)*delta
            a = (ddh/T**2)*delta
            return p, v, a, seg

        def command(ctx: CommandContext) -> tuple[np.ndarray, int]:
            if not origin_holder:
                origin_holder.append(ctx.position.copy())
            p_ref, v_ref, a_ff, seg = reference(ctx.t)
            p_ref = origin_holder[0] + p_ref
            a_cmd = a_ff + kp*(p_ref-ctx.position) + kd*(v_ref-ctx.velocity)
            return a_cmd, seg

        return Profile(command, {"relative_waypoints": rel_wp.tolist(), "times": t_wp.tolist(),
                                 "tracking_kp": kp, "tracking_kd": kd}, t_wp[1:-1].tolist())

    if family == "rotating_acceleration":
        amp = rng.uniform(4.0, 8.5)
        f0 = rng.uniform(0.16, 0.32)
        f1 = rng.uniform(0.38, 0.70)
        shift = rng.uniform(0.45, 0.62)*duration
        z_amp = rng.uniform(1.0, 3.0)
        phase0 = rng.uniform(-math.pi, math.pi)

        def command(ctx: CommandContext) -> tuple[np.ndarray, int]:
            if ctx.t < shift:
                theta = 2*math.pi*f0*ctx.t + phase0
                phase = 0
            else:
                theta_shift = 2*math.pi*f0*shift + phase0
                theta = theta_shift + 2*math.pi*f1*(ctx.t-shift)
                phase = 1
            return np.array([amp*math.cos(theta), amp*math.sin(theta),
                             z_amp*math.sin(0.5*theta)]), phase

        return Profile(command, {"amplitude": amp, "frequency_before_hz": f0,
                                 "frequency_after_hz": f1, "vertical_amplitude": z_amp}, [shift])

    if family == "mixed_mode_shift":
        shift1 = rng.uniform(0.28, 0.36)*duration
        shift2 = rng.uniform(0.62, 0.72)*duration
        lat1 = rng.uniform(3.0, 6.0)
        lat2 = rng.uniform(6.0, 10.0)
        z_amp = rng.uniform(3.0, 7.0)
        f = rng.uniform(0.35, 0.65)
        direction = int(rng.choice([-1, 1]))

        def command(ctx: CommandContext) -> tuple[np.ndarray, int]:
            normal = _left_normal_xy(ctx.velocity)
            if ctx.t < shift1:
                return np.array([direction*lat1*normal[0], direction*lat1*normal[1], 0.0]), 0
            if ctx.t < shift2:
                lateral = lat2*math.sin(2*math.pi*f*(ctx.t-shift1))
                return np.array([lateral*normal[0], lateral*normal[1], 1.0]), 1
            theta = 2*math.pi*(1.35*f)*(ctx.t-shift2)
            return np.array([lat2*math.cos(theta), lat2*math.sin(theta),
                             -z_amp*_smooth_pulse(ctx.t, shift2+0.4, duration-0.3, 0.15)]), 2

        return Profile(command, {"first_lateral": lat1, "second_lateral": lat2,
                                 "vertical_dive": z_amp, "s_turn_frequency_hz": f}, [shift1, shift2])

    raise ValueError(f"Unsupported family: {family}")


def make_target_config(cfg: GenerationConfig, position: np.ndarray, velocity: np.ndarray) -> PursuerConfig:
    return PursuerConfig(
        model_type="quadrotor_6dof",
        max_velocity=cfg.max_speed,
        max_acceleration=cfg.max_acceleration,
        max_acceleration_per_axis=cfg.max_acceleration_per_axis,
        max_jerk=cfg.max_jerk,
        actuator_time_constant=0.1,
        initial_position=position.tolist(),
        initial_velocity=velocity.tolist(),
        mass=cfg.mass,
        inertia=[cfg.inertia_x, cfg.inertia_y, cfg.inertia_z],
        attitude_time_constant=cfg.attitude_time_constant,
        max_thrust_to_weight_ratio=cfg.max_thrust_to_weight_ratio,
        max_angular_rate=cfg.max_angular_rate,
        max_torque_per_axis=cfg.max_torque_per_axis,
        initial_quaternion=[1.0, 0.0, 0.0, 0.0],
        initial_angular_velocity=[0.0, 0.0, 0.0],
    )


def initial_state(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    position = np.array([
        rng.uniform(65.0, 95.0),
        rng.uniform(-25.0, 25.0),
        rng.uniform(18.0, 40.0),
    ])
    speed = rng.uniform(7.0, 15.0)
    heading = rng.uniform(-math.pi, math.pi)
    climb = rng.uniform(-1.5, 1.5)
    velocity = np.array([speed*math.cos(heading), speed*math.sin(heading), climb])
    return position, velocity


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values)
    return np.gradient(values, dt, axis=0, edge_order=2 if len(values) >= 3 else 1)


def simulate_trajectory(
    family: str,
    trajectory_seed: int,
    cfg: GenerationConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rng = np.random.default_rng(trajectory_seed)
    p0, v0 = initial_state(rng)
    target_cfg = make_target_config(cfg, p0, v0)
    target = Quadrotor6DOFPursuer(target_cfg)
    target.reset(p0, v0)
    profile = build_profile(family, rng, cfg.duration)
    limiter = JerkLimiter(cfg.max_jerk, cfg.max_acceleration_per_axis, cfg.max_acceleration)

    times = np.arange(0.0, cfg.duration + 0.5*cfg.dt, cfg.dt)
    rows: list[dict[str, float | int]] = []
    disturbance_rng = np.random.default_rng(trajectory_seed ^ 0xA5A5A5A5)

    for step, t in enumerate(times):
        state = target.state
        ctx = CommandContext(
            t=float(t), dt=cfg.dt,
            position=state.position.copy(),
            velocity=state.velocity.copy(),
            applied_acceleration=state.applied_acceleration.copy(),
            rng=rng,
        )
        desired_acc, phase = profile.command(ctx)

        # Soft altitude envelope: this is a plant-safe correction, not a controller-outcome filter.
        if state.position[2] < cfg.min_altitude + 3.0:
            desired_acc[2] += 4.0*(cfg.min_altitude + 3.0 - state.position[2])
        elif state.position[2] > cfg.max_altitude - 5.0:
            desired_acc[2] -= 3.0*(state.position[2] - (cfg.max_altitude - 5.0))

        cmd_acc = limiter.step(desired_acc, cfg.dt)
        disturbance = disturbance_rng.normal(0.0, cfg.disturbance_std, size=3)

        q = target.quaternion
        omega = target.angular_velocity
        rows.append({
            "time": float(t),
            "pos_x": float(state.position[0]), "pos_y": float(state.position[1]), "pos_z": float(state.position[2]),
            "vel_x": float(state.velocity[0]), "vel_y": float(state.velocity[1]), "vel_z": float(state.velocity[2]),
            "acc_x": float(state.applied_acceleration[0]), "acc_y": float(state.applied_acceleration[1]),
            "acc_z": float(state.applied_acceleration[2]),
            "cmd_acc_x": float(cmd_acc[0]), "cmd_acc_y": float(cmd_acc[1]), "cmd_acc_z": float(cmd_acc[2]),
            "quat_w": float(q[0]), "quat_x": float(q[1]), "quat_y": float(q[2]), "quat_z": float(q[3]),
            "omega_x": float(omega[0]), "omega_y": float(omega[1]), "omega_z": float(omega[2]),
            "phase": int(phase),
        })
        if step < len(times)-1:
            target.step(cmd_acc, disturbance, cfg.dt)

    df = pd.DataFrame(rows)
    acc = df[["acc_x", "acc_y", "acc_z"]].to_numpy()
    jerk = _finite_difference(acc, cfg.dt)
    df[["jerk_x", "jerk_y", "jerk_z"]] = jerk

    vel = df[["vel_x", "vel_y", "vel_z"]].to_numpy()
    speed = np.linalg.norm(vel, axis=1)
    acc_norm = np.linalg.norm(acc, axis=1)
    jerk_norm = np.linalg.norm(jerk, axis=1)
    cross_va = np.cross(vel, acc)
    curvature = np.linalg.norm(cross_va, axis=1) / np.maximum(speed**3, 1e-6)
    cmd = df[["cmd_acc_x", "cmd_acc_y", "cmd_acc_z"]].to_numpy()
    command_error = np.linalg.norm(cmd - acc, axis=1)

    metadata: dict[str, object] = {
        "family": family,
        "trajectory_seed": int(trajectory_seed),
        "duration_s": float(cfg.duration),
        "dt_s": float(cfg.dt),
        "n_samples": int(len(df)),
        "initial_position": p0.tolist(),
        "initial_velocity": v0.tolist(),
        "shift_times_s": profile.shift_times,
        "profile_parameters": profile.parameters,
        "mean_speed_mps": float(np.mean(speed)),
        "max_speed_mps": float(np.max(speed)),
        "p95_acceleration_mps2": float(np.percentile(acc_norm, 95)),
        "max_acceleration_mps2": float(np.max(acc_norm)),
        "p95_jerk_mps3": float(np.percentile(jerk_norm, 95)),
        "max_jerk_mps3": float(np.max(jerk_norm)),
        "p95_curvature_inv_m": float(np.percentile(curvature, 95)),
        "max_curvature_inv_m": float(np.max(curvature)),
        "max_altitude_m": float(df["pos_z"].max()),
        "min_altitude_m": float(df["pos_z"].min()),
        "mean_command_tracking_error_mps2": float(np.mean(command_error)),
        "max_command_tracking_error_mps2": float(np.max(command_error)),
        "n_phase_transitions": int(np.count_nonzero(np.diff(df["phase"].to_numpy()))),
    }
    return df, metadata


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_plots(df: pd.DataFrame, base_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots", file=sys.stderr)
        return

    pos = df[["pos_x", "pos_y", "pos_z"]].to_numpy()
    plots = base_path.parent
    plots.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(base_path.with_name(base_path.name + "_3d.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(pos[:, 0], pos[:, 1])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title + " — XY")
    fig.tight_layout()
    fig.savefig(base_path.with_name(base_path.name + "_xy.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(pos[:, 1], pos[:, 2])
    ax.set_xlabel("y [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(title + " — YZ")
    fig.tight_layout()
    fig.savefig(base_path.with_name(base_path.name + "_yz.png"), dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/q2_challenge_v1/generated_6dof"))
    parser.add_argument("--per-family", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--max-speed", type=float, default=20.0)
    parser.add_argument("--max-acceleration", type=float, default=14.0)
    parser.add_argument("--max-jerk", type=float, default=30.0)
    parser.add_argument("--disturbance-std", type=float, default=0.0)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--plots", action="store_true", help="Save 3D, XY, and YZ PNGs for each trajectory")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.per_family <= 0:
        raise ValueError("--per-family must be positive")
    if args.dt <= 0 or args.duration <= args.dt:
        raise ValueError("Require 0 < dt < duration")

    cfg = GenerationConfig(
        dt=args.dt,
        duration=args.duration,
        per_family=args.per_family,
        seed=args.seed,
        max_speed=args.max_speed,
        max_acceleration=args.max_acceleration,
        max_acceleration_per_axis=min(args.max_acceleration, 12.0),
        max_jerk=args.max_jerk,
        disturbance_std=args.disturbance_std,
        save_plots=args.plots,
    )

    output = args.output.resolve()
    csv_dir = output / "csv"
    plot_dir = output / "plots"
    manifest_dir = output / "manifests"
    csv_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    if args.plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    master_rng = np.random.default_rng(cfg.seed)
    jobs: list[tuple[str, int, int]] = []
    for family in args.families:
        for index in range(cfg.per_family):
            trajectory_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
            jobs.append((family, index, trajectory_seed))

    metadata_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []

    for family, index, trajectory_seed in tqdm(jobs, desc="Generating trajectories", unit="traj"):
        filename = f"q2c_{family}_{index:04d}_seed{trajectory_seed}.csv"
        csv_path = csv_dir / filename
        if csv_path.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {csv_path}. Use --overwrite to replace it.")

        df, metadata = simulate_trajectory(family, trajectory_seed, cfg)
        df.to_csv(csv_path, index=False, float_format="%.9f")
        file_hash = sha256_file(csv_path)

        metadata_row = {
            "trajectory_id": csv_path.stem,
            "relative_path": str(csv_path.relative_to(output)),
            "sha256": file_hash,
            **{k: v for k, v in metadata.items() if k not in {"profile_parameters", "shift_times_s"}},
            "shift_times_s": json.dumps(metadata["shift_times_s"]),
            "profile_parameters": json.dumps(metadata["profile_parameters"], sort_keys=True),
        }
        metadata_rows.append(metadata_row)
        manifest_rows.append({
            "trajectory_id": csv_path.stem,
            "family": family,
            "relative_path": str(csv_path.relative_to(output)),
            "sha256": file_hash,
            "trajectory_seed": trajectory_seed,
        })

        if args.plots:
            save_plots(df, plot_dir / csv_path.stem, f"{family} #{index:04d}")

    metadata_df = pd.DataFrame(metadata_rows).sort_values(["family", "trajectory_id"])
    metadata_path = output / "trajectory_metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)

    manifest_txt = manifest_dir / "generated_test_manifest.txt"
    manifest_txt.write_text(
        "\n".join(str((output / row["relative_path"]).resolve()) for row in manifest_rows) + "\n",
        encoding="utf-8",
    )

    family_counts = metadata_df.groupby("family").size().to_dict()
    dataset_manifest = {
        "dataset_name": "Q2 Quadrotor 6-DOF Challenge Dataset C",
        "schema_version": 1,
        "generator": str(_THIS_FILE),
        "generator_sha256": sha256_file(_THIS_FILE),
        "project_root": str(_PROJECT_ROOT),
        "generation_config": asdict(cfg),
        "families": list(args.families),
        "family_counts": {str(k): int(v) for k, v in family_counts.items()},
        "n_trajectories": int(len(metadata_df)),
        "metadata_file": metadata_path.name,
        "metadata_sha256": sha256_file(metadata_path),
        "trajectory_manifest": str(manifest_txt.relative_to(output)),
        "trajectory_manifest_sha256": sha256_file(manifest_txt),
        "files": manifest_rows,
    }
    manifest_json = manifest_dir / "manifest.json"
    manifest_json.write_text(json.dumps(dataset_manifest, indent=2, sort_keys=True), encoding="utf-8")

    # Dataset card with exact scope and limitations.
    card = output / "DATASET_CARD.md"
    card.write_text(
        "# Q2 Quadrotor 6-DOF Challenge Dataset C\n\n"
        f"- Generated trajectories: **{len(metadata_df)}**\n"
        f"- Families: **{len(args.families)}**\n"
        f"- Time step: **{cfg.dt:.4f} s**\n"
        f"- Duration per trajectory: **{cfg.duration:.2f} s**\n"
        "- Plant: repository `Quadrotor6DOFPursuer` (13-state rigid body with cascaded attitude loop)\n"
        "- Commands: deterministic family-specific world-frame acceleration profiles passed through a jerk limiter\n"
        "- Selection: generated independently of interception-controller outcomes\n\n"
        "## Intended use\n\n"
        "Challenge evaluation of causal target prediction and learning-assisted MPC under maneuver families and "
        "predeclared distribution shifts. This is a physics-generated benchmark, not real-flight data.\n\n"
        "## Important limitations\n\n"
        "The target and pursuer may share the same plant class if the interception simulator also uses "
        "`Quadrotor6DOFPursuer`. This reduces structural model mismatch. Report that limitation and use external "
        "real-flight trajectories as a separate benchmark when available.\n",
        encoding="utf-8",
    )

    print(f"Generated {len(metadata_df)} trajectories in: {output}")
    print(f"Metadata: {metadata_path}")
    print(f"Manifest: {manifest_json}")
    print(f"CSV list: {manifest_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
