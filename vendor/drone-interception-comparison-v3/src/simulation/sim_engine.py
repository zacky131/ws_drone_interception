"""
Simulation engine – orchestrates one complete interception run.

Responsibilities:
    1. Query the target scenario for ground-truth state at each timestep.
    2. Pass truth through the sensor model to obtain noisy measurements.
    3. Feed measurements to the estimator (predict + update cycle).
    4. Query the controller for commanded acceleration.
    5. Step the pursuer dynamics (with wind disturbance).
    6. Log all quantities.
    7. Detect termination (success, timeout, divergence).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.utils.config_schema import ExperimentConfig
from src.dynamics.pursuer_base import PursuerBase
from src.estimation.estimator_base import EstimatorBase
from src.control.controller_base import ControllerBase
from src.environment.wind_model import WindModel
from src.environment.sensor_model import SensorModel
from src.simulation.scenario import TargetScenario
from src.simulation.logger import SimulationLogger


class SimulationEngine:
    """Run a single interception simulation from initial conditions to termination."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._cfg = config
        self.dt = config.simulation.dt
        self.max_time = config.simulation.max_time
        self.success_distance = config.simulation.success_distance

    def run(
        self,
        scenario: TargetScenario,
        pursuer: PursuerBase,
        estimator: EstimatorBase,
        controller: ControllerBase,
        wind_model: WindModel,
        sensor: SensorModel,
        logger: Optional[SimulationLogger] = None,
    ) -> SimulationLogger:
        """Execute the simulation loop and return a populated logger.

        Parameters
        ----------
        scenario : TargetScenario
        pursuer : PursuerBase
        estimator : EstimatorBase
        controller : ControllerBase
        wind_model : WindModel
        sensor : SensorModel
        logger : SimulationLogger, optional
            If ``None``, a fresh logger is created.

        Returns
        -------
        logger : SimulationLogger
            Contains per-step records and outcome flag.
        """
        if logger is None:
            logger = SimulationLogger()
        else:
            logger.reset()

        # Reset components
        estimator.reset()
        controller.reset()
        if hasattr(controller, "set_scenario"):
            controller.set_scenario(scenario)
        sensor.reset()

        max_steps = int(self.max_time / self.dt) + 1
        initial_distance: float | None = None
        import time as _time

        for step in range(max_steps):
            t = step * self.dt

            t_step_start = _time.perf_counter()

            # ── 1. Target kinematics ──────────────────────────────────────
            target_pos, target_vel, target_acc = scenario.get_target_state(t)
            target_true_6d = np.concatenate([target_pos, target_vel])

            # ── 2. Sensor model ───────────────────────────────────────────
            target_meas = sensor.process_target(target_true_6d, t)

            # ── 3. Estimator predict + update ─────────────────────────────
            t_est_start = _time.perf_counter()
            if step > 0:
                estimator.predict(self.dt)
            if target_meas is not None:
                estimator.update(target_meas)
            elif step == 0:
                # First step with no measurement (dropout) – use truth as init
                estimator.initialize(target_true_6d)

            target_est = estimator.get_estimate()  # 12-D
            estimator_time_s = _time.perf_counter() - t_est_start

            # ── 4. Wind ───────────────────────────────────────────────────
            wind = wind_model.get_wind(t)

            # ── 5. Distance check ─────────────────────────────────────────
            distance = float(np.linalg.norm(pursuer.position - target_pos))
            if initial_distance is None:
                initial_distance = distance

            # ── 6. Compute control ────────────────────────────────────────
            t_ctrl_start = _time.perf_counter()
            cmd_acc, ctrl_info = controller.compute_control(
                pursuer.state.to_array(), target_meas, target_est, wind, t
            )
            controller_total_time_s = _time.perf_counter() - t_ctrl_start

            step_wall_time_s = _time.perf_counter() - t_step_start

            if ctrl_info is None:
                ctrl_info = {}
            ctrl_info["estimator_time_s"] = estimator_time_s
            ctrl_info["controller_total_time_s"] = controller_total_time_s
            ctrl_info["control_pipeline_time_s"] = estimator_time_s + controller_total_time_s
            ctrl_info["simulation_step_wall_time_s"] = step_wall_time_s

            # ── 7. Log ────────────────────────────────────────────────────
            logger.log_step(
                t=t,
                step=step,
                pursuer_pos=pursuer.position.copy(),
                pursuer_vel=pursuer.velocity.copy(),
                pursuer_applied_acc=pursuer.state.applied_acceleration.copy(),
                target_true_pos=target_pos,
                target_true_vel=target_vel,
                target_true_acc=target_acc,
                target_est=target_est,
                target_meas=target_meas,
                commanded_acc=cmd_acc,
                wind=wind,
                distance=distance,
                ctrl_info=ctrl_info,
            )

            # ── 8. Termination checks ────────────────────────────────────
            if distance <= self.success_distance:
                logger.set_outcome(True, intercept_time=t)
                return logger

            if initial_distance is not None and distance > 2.5 * initial_distance:
                logger.set_outcome(False, failure_reason="divergence")
                return logger

            if t >= self.max_time:
                logger.set_outcome(False, failure_reason="timeout")
                return logger

            # ── 9. Step pursuer dynamics ──────────────────────────────────
            pursuer.step(cmd_acc, wind, self.dt)

        logger.set_outcome(False, failure_reason="timeout")
        return logger
