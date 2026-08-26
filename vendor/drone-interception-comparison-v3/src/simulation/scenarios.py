"""
Extended target trajectory scenario classes for harder maneuver profiles.

This module adds three new scenario types on top of the existing
:class:`~src.simulation.scenario.TargetScenario` base without modifying
any existing scenario generators:

* :class:`AggressiveTurningScenario` — high-g banked turn with randomised onset.
* :class:`SinusoidalEvasionScenario` — lateral sinusoidal evasion.
* :class:`UnpredictableJerkScenario` — random jerk changes at fixed intervals.

Factory
-------
Use :func:`create_scenario` to obtain the correct object for any scenario
type string, including both existing and new types::

    scenario = create_scenario(cfg.scenario, cfg.simulation, seed=42)

All new classes expose the **same public interface** as
:class:`~src.simulation.scenario.TargetScenario`:

* ``get_target_state(t)`` → ``(pos, vel, acc)`` each ``(3,)``
* ``duration`` property
* ``dataframe`` property

New scenario type strings
-------------------------
* ``"aggressive_turning"``
* ``"sinusoidal_evasion"``
* ``"unpredictable_jerk"``
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.config_schema import ScenarioConfig, SimulationConfig
from src.simulation.scenario import TargetScenario


# ── AggressiveTurningScenario ─────────────────────────────────────────────────

class AggressiveTurningScenario(TargetScenario):
    """Target performs a high-g banked turn that starts at a random onset time.

    The trajectory has two phases:

    1. **Pre-onset** (``t < t_onset``): straight constant-velocity flight.
    2. **Turn phase** (``t ≥ t_onset``): horizontal coordinated turn at
       ``aggressive_turn_rate`` rad/s, clipped so the centripetal acceleration
       does not exceed ``max_lateral_acc`` m/s².

    Parameters
    ----------
    scenario_cfg : ScenarioConfig
        Scenario configuration.  Relevant new fields:

        * ``turn_onset_range`` — ``[t_min, t_max]`` for the randomised onset [s].
        * ``aggressive_turn_rate`` — desired turn rate [rad/s].
        * ``max_lateral_acc`` — centripetal acceleration cap [m/s²].
    sim_cfg : SimulationConfig
        Simulation timing.
    seed : int or None
        Random seed for onset-time sampling.  ``None`` → non-deterministic.
    """

    def __init__(
        self,
        scenario_cfg: ScenarioConfig,
        sim_cfg: SimulationConfig,
        seed: Optional[int] = None,
    ) -> None:
        # Store seed before super().__init__ calls _build() via polymorphism.
        self._init_seed = seed
        super().__init__(scenario_cfg, sim_cfg)

    def _build(self) -> None:  # type: ignore[override]
        """Generate the aggressive-turning trajectory (overrides parent)."""
        self._generate_aggressive_turning()

    def _generate_aggressive_turning(self) -> None:
        rng = np.random.default_rng(self._init_seed)
        cfg = self._cfg
        t = self._time_vector()
        N = len(t)

        p0 = np.asarray(cfg.target_initial_position, dtype=float)
        v0 = np.asarray(cfg.target_initial_velocity, dtype=float)

        t_min, t_max = cfg.turn_onset_range
        t_onset = float(rng.uniform(t_min, t_max))

        omega_raw = cfg.aggressive_turn_rate
        speed_h = np.linalg.norm(v0[:2])

        # Cap turn rate so centripetal acc ≤ max_lateral_acc
        if speed_h > 1e-6:
            omega_max = cfg.max_lateral_acc / speed_h
            omega = min(omega_raw, omega_max)
        else:
            omega = omega_raw

        heading0 = np.arctan2(v0[1], v0[0])

        pos = np.zeros((N, 3))
        vel = np.zeros((N, 3))
        acc = np.zeros((N, 3))

        for i, ti in enumerate(t):
            if i == 0:
                pos[i] = p0
                vel[i] = v0
                acc[i] = np.zeros(3)
                continue

            if ti < t_onset:
                # Straight flight
                vel[i] = v0.copy()
                acc[i] = np.zeros(3)
            else:
                # Coordinated horizontal turn
                tau = ti - t_onset
                theta = heading0 + omega * tau
                vx = speed_h * np.cos(theta)
                vy = speed_h * np.sin(theta)
                vel[i] = np.array([vx, vy, v0[2]])
                acc[i] = np.array([
                    -speed_h * omega * np.sin(theta),
                     speed_h * omega * np.cos(theta),
                    0.0,
                ])

            pos[i] = pos[i - 1] + vel[i - 1] * self._dt

        self._times = t
        self._positions = pos
        self._velocities = vel
        self._accelerations = acc


# ── SinusoidalEvasionScenario ─────────────────────────────────────────────────

class SinusoidalEvasionScenario(TargetScenario):
    """Target oscillates laterally with randomised frequency and amplitude.

    The target moves forward at its initial speed while adding a sinusoidal
    lateral velocity component:

    .. math::

        v_\\perp(t) = A \\sin(2\\pi f t)

    where *A* (amplitude) and *f* (frequency) are drawn uniformly from the
    configured ranges at initialisation time.

    Parameters
    ----------
    scenario_cfg : ScenarioConfig
        Relevant new fields:

        * ``evasion_frequency_range`` — ``[f_min, f_max]`` [Hz].
        * ``evasion_amplitude_range`` — ``[A_min, A_max]`` [m/s].
    sim_cfg : SimulationConfig
        Simulation timing.
    seed : int or None
        Random seed for parameter sampling.
    """

    def __init__(
        self,
        scenario_cfg: ScenarioConfig,
        sim_cfg: SimulationConfig,
        seed: Optional[int] = None,
    ) -> None:
        self._init_seed = seed
        super().__init__(scenario_cfg, sim_cfg)

    def _build(self) -> None:  # type: ignore[override]
        self._generate_sinusoidal_evasion()

    def _generate_sinusoidal_evasion(self) -> None:
        rng = np.random.default_rng(self._init_seed)
        cfg = self._cfg
        t = self._time_vector()
        N = len(t)

        p0 = np.asarray(cfg.target_initial_position, dtype=float)
        v0 = np.asarray(cfg.target_initial_velocity, dtype=float)

        f_min, f_max = cfg.evasion_frequency_range
        A_min, A_max = cfg.evasion_amplitude_range
        freq = float(rng.uniform(f_min, f_max))  # [Hz]
        amp = float(rng.uniform(A_min, A_max))   # [m/s]

        # Lateral direction: perpendicular to initial horizontal velocity
        v_h = np.array([v0[0], v0[1], 0.0])
        v_h_norm = np.linalg.norm(v_h)
        if v_h_norm > 1e-6:
            # Right-hand perpendicular in the horizontal plane
            lat_dir = np.array([-v_h[1], v_h[0], 0.0]) / v_h_norm
        else:
            lat_dir = np.array([0.0, 1.0, 0.0])

        omega_ev = 2.0 * np.pi * freq

        pos = np.zeros((N, 3))
        vel = np.zeros((N, 3))
        acc = np.zeros((N, 3))
        pos[0] = p0

        for i, ti in enumerate(t):
            lat_v = amp * np.sin(omega_ev * ti)
            lat_a = amp * omega_ev * np.cos(omega_ev * ti)
            vel[i] = v0 + lat_dir * lat_v
            acc[i] = lat_dir * lat_a
            if i > 0:
                pos[i] = pos[i - 1] + vel[i - 1] * self._dt

        self._times = t
        self._positions = pos
        self._velocities = vel
        self._accelerations = acc


# ── UnpredictableJerkScenario ─────────────────────────────────────────────────

class UnpredictableJerkScenario(TargetScenario):
    """Target applies random jerk impulses at fixed time intervals.

    Every ``jerk_dt_change`` seconds the jerk vector is redrawn uniformly
    from ``[-jerk_magnitude_limit, jerk_magnitude_limit]^3``.  Velocity and
    position are integrated forward from these jerk segments.  The target
    speed is **not** constrained; for realistic scenarios keep
    ``jerk_magnitude_limit`` moderate (≤ 10 m/s³).

    Parameters
    ----------
    scenario_cfg : ScenarioConfig
        Relevant new fields:

        * ``jerk_dt_change`` — interval between jerk redraws [s].
        * ``jerk_magnitude_limit`` — per-axis jerk uniform bound [m/s³].
    sim_cfg : SimulationConfig
        Simulation timing.
    seed : int or None
        Random seed for reproducible jerk sequences.
    """

    def __init__(
        self,
        scenario_cfg: ScenarioConfig,
        sim_cfg: SimulationConfig,
        seed: Optional[int] = None,
    ) -> None:
        self._init_seed = seed
        super().__init__(scenario_cfg, sim_cfg)

    def _build(self) -> None:  # type: ignore[override]
        self._generate_unpredictable_jerk()

    def _generate_unpredictable_jerk(self) -> None:
        rng = np.random.default_rng(self._init_seed)
        cfg = self._cfg
        t = self._time_vector()
        N = len(t)
        dt = self._dt

        p0 = np.asarray(cfg.target_initial_position, dtype=float)
        v0 = np.asarray(cfg.target_initial_velocity, dtype=float)
        j_limit = cfg.jerk_magnitude_limit
        dt_change = max(cfg.jerk_dt_change, dt)  # at least one step

        pos = np.zeros((N, 3))
        vel = np.zeros((N, 3))
        acc = np.zeros((N, 3))

        pos[0] = p0
        vel[0] = v0
        acc[0] = np.zeros(3)

        # Initial jerk segment
        current_jerk = rng.uniform(-j_limit, j_limit, size=3)
        next_change_t = dt_change

        for i in range(1, N):
            ti = t[i]
            if ti >= next_change_t:
                current_jerk = rng.uniform(-j_limit, j_limit, size=3)
                next_change_t += dt_change

            # Euler integration of jerk → acceleration → velocity → position
            acc[i] = acc[i - 1] + current_jerk * dt
            vel[i] = vel[i - 1] + acc[i - 1] * dt
            pos[i] = pos[i - 1] + vel[i - 1] * dt

        self._times = t
        self._positions = pos
        self._velocities = vel
        self._accelerations = acc


# ── Factory ───────────────────────────────────────────────────────────────────

#: Registry mapping scenario type strings to their classes.
_SCENARIO_REGISTRY: dict = {
    "aggressive_turning": AggressiveTurningScenario,
    "sinusoidal_evasion": SinusoidalEvasionScenario,
    "unpredictable_jerk": UnpredictableJerkScenario,
}


def create_scenario(
    scenario_cfg: ScenarioConfig,
    sim_cfg: SimulationConfig,
    seed: Optional[int] = None,
) -> TargetScenario:
    """Return the appropriate scenario object for *scenario_cfg.scenario_type*.

    Handles both existing types (delegated to :class:`TargetScenario`) and the
    three new extended types registered in :data:`_SCENARIO_REGISTRY`.

    Parameters
    ----------
    scenario_cfg : ScenarioConfig
        Scenario configuration including ``scenario_type``.
    sim_cfg : SimulationConfig
        Simulation timing parameters.
    seed : int or None
        Random seed passed to new scenario classes for reproducibility.
        Ignored for existing scenario types (they are deterministic).

    Returns
    -------
    TargetScenario
        Scenario object with a ``get_target_state(t)`` interface.

    Raises
    ------
    ValueError
        If ``scenario_type`` is not recognised by either registry.
    """
    stype = scenario_cfg.scenario_type
    if stype in _SCENARIO_REGISTRY:
        cls = _SCENARIO_REGISTRY[stype]
        return cls(scenario_cfg, sim_cfg, seed=seed)
    # Delegate to the original factory for existing types
    return TargetScenario(scenario_cfg, sim_cfg)