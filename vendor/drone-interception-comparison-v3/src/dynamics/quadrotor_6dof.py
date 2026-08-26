"""
Full 6-DOF rigid-body quadrotor pursuer with cascaded attitude controller.

Equations of motion
-------------------
Translational (world frame, z-up):
    ṗ = v
    v̇ = (T / m) · R · e₃  −  g · e₃  +  a_wind

Rotational (body frame):
    q̇ = ½ · q ⊗ [0, ω]        (unit-quaternion kinematics)
    ω̇ = J⁻¹ · (τ − ω × J·ω)   (Euler's rigid-body equation)

where R ∈ SO(3) is the rotation from body to world derived from the
unit quaternion q = [qw, qx, qy, qz], and [T, τ] are the total thrust
(scalar, along body z) and body-frame torque vector (3D) respectively.

Cascaded inner-loop attitude controller
----------------------------------------
The outer-loop guidance layer provides `commanded_acceleration` in world
frame.  The inner loops convert this to [T, τ] as follows:

  1. Gravity-compensating desired force:
         f_des = m · (a_cmd + g · e₃)

  2. Desired body-z (thrust direction):
         z_b_des = f_des / ‖f_des‖    (with T capped at m·TWR·g)

  3. Axis-angle attitude error → desired angular-velocity setpoint:
         ω_att_des  =  (θ_err / τ_att) · axis_err    [world frame]
         θ_err = arccos( z_b · z_b_des )
         axis_err = (z_b × z_b_des) / ‖z_b × z_b_des‖

     where τ_att is the attitude-loop time constant.

  4. Rate controller torque (body frame):
         τ = J · k_r · (R^T · ω_att_des − ω)  +  ω × J·ω
     k_r = 2 / τ_att  (critically-damped second-order response)

Integration
-----------
RK4 is used for the coupled 13-state ODE.  The quaternion is re-normalised
after each sub-step to prevent drift.

External interface
------------------
Accepts `commanded_acceleration` (3D, world frame) and returns PursuerState
(position, velocity, applied_acceleration) – identical to the other pursuer
models so that guidance controllers are model-agnostic.

Additional state exposed via properties:
    .quaternion         – current attitude q = [qw, qx, qy, qz]
    .angular_velocity   – body-frame angular velocity ω [rad/s]
    .euler_angles_deg   – roll, pitch, yaw in degrees (ZYX convention)

State vector  x = [p(3), v(3), q(4), ω(3)]  ∈  R¹³

Physical constraints enforced:
    - Thrust bounded to  [0, m · TWR · g]
    - Per-axis torque bounded to  ±max_torque_per_axis
    - Angular rate bounded to  ‖ω‖ ≤ max_angular_rate
    - Velocity bounded to  ‖v‖ ≤ max_velocity
    - Ground constraint  z ≥ 0
"""

from __future__ import annotations

import numpy as np

from src.utils.config_schema import PursuerConfig
from src.utils.math_helpers import clip_norm, clip_components, safe_normalize
from .pursuer_base import PursuerBase, PursuerState

# Physical constants
_G: float = 9.81          # gravitational acceleration [m/s²]
_E3: np.ndarray = np.array([0.0, 0.0, 1.0])   # world z-up unit vector


# ─────────────────────────────────────────────────────────────────────────────
# Quaternion / rotation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix from unit quaternion q = [qw, qx, qy, qz].

    Maps body-frame vectors to world frame:  p_w = R · p_b
    """
    qw, qx, qy, qz = q
    return np.array([
        [1.0 - 2.0*(qy*qy + qz*qz),  2.0*(qx*qy - qz*qw),  2.0*(qx*qz + qy*qw)],
        [2.0*(qx*qy + qz*qw),  1.0 - 2.0*(qx*qx + qz*qz),  2.0*(qy*qz - qx*qw)],
        [2.0*(qx*qz - qy*qw),  2.0*(qy*qz + qx*qw),  1.0 - 2.0*(qx*qx + qy*qy)],
    ])


def _quat_product(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product q1 ⊗ q2.  Both arrays shaped (4,) as [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def _quat_derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """Quaternion kinematic rate: q̇ = ½ · q ⊗ [0, ω_body]."""
    omega_pure = np.concatenate([[0.0], omega_body])
    return 0.5 * _quat_product(q, omega_pure)


def _euler_from_quat(q: np.ndarray) -> np.ndarray:
    """ZYX Euler angles (roll φ, pitch θ, yaw ψ) in degrees from [qw,qx,qy,qz]."""
    qw, qx, qy, qz = q
    roll  = np.degrees(np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy)))
    sinp  = 2*(qw*qy - qz*qx)
    pitch = np.degrees(np.arcsin(np.clip(sinp, -1.0, 1.0)))
    yaw   = np.degrees(np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz)))
    return np.array([roll, pitch, yaw])


# ─────────────────────────────────────────────────────────────────────────────
# 6-DOF pursuer class
# ─────────────────────────────────────────────────────────────────────────────

class Quadrotor6DOFPursuer(PursuerBase):
    """Full 6-DOF quadrotor pursuer with cascaded attitude controller.

    Parameters
    ----------
    config : PursuerConfig
        Must include the 6DOF-specific fields:
        ``mass``, ``inertia``, ``attitude_time_constant``,
        ``max_thrust_to_weight_ratio``, ``max_angular_rate``,
        ``max_torque_per_axis``, ``initial_quaternion``,
        ``initial_angular_velocity``.
    """

    def __init__(self, config: PursuerConfig) -> None:
        self._cfg = config

        # Physical properties
        self._m: float = config.mass
        inertia = np.asarray(config.inertia, dtype=float)
        self._J: np.ndarray = np.diag(inertia)
        self._J_inv: np.ndarray = np.diag(1.0 / inertia)

        # Attitude controller gains
        self._tau_att: float = config.attitude_time_constant     # [s]
        self._k_rate: float = 2.0 / config.attitude_time_constant  # critical damping

        # Actuator limits
        self._max_thrust: float = config.mass * config.max_thrust_to_weight_ratio * _G
        self._max_omega: float = config.max_angular_rate
        self._max_torque: float = config.max_torque_per_axis

        # Rotational state
        self._q: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])
        self._omega: np.ndarray = np.zeros(3)

        # Translational state (PursuerState)
        self._state = PursuerState()

        self.reset(
            np.asarray(config.initial_position, dtype=float),
            np.asarray(config.initial_velocity, dtype=float),
        )

    # ── interface ─────────────────────────────────────────────────────────────

    def reset(self, position: np.ndarray, velocity: np.ndarray) -> None:
        """Reset translational state; attitude initialised from config."""
        q0 = np.asarray(self._cfg.initial_quaternion, dtype=float)
        norm_q0 = np.linalg.norm(q0)
        self._q = q0 / norm_q0 if norm_q0 > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0])
        self._omega = np.asarray(self._cfg.initial_angular_velocity, dtype=float).copy()

        self._state = PursuerState(
            position=position.copy(),
            velocity=velocity.copy(),
            applied_acceleration=np.zeros(3),
        )

    def step(
        self,
        commanded_acceleration: np.ndarray,
        wind_disturbance: np.ndarray,
        dt: float,
    ) -> PursuerState:
        """Advance the 6DOF model by one timestep.

        Parameters
        ----------
        commanded_acceleration : (3,) array
            Outer-loop guidance command [m/s²] in world frame.
        wind_disturbance : (3,) array
            Wind acceleration disturbance [m/s²] in world frame.
        dt : float
            Integration timestep [s].

        Returns
        -------
        PursuerState
            Updated translational state (position, velocity,
            applied_acceleration in world frame).
        """
        # ── 1. Saturate outer-loop command ────────────────────────────────
        a_cmd = clip_components(commanded_acceleration, self._cfg.max_acceleration_per_axis)
        a_cmd = clip_norm(a_cmd, self._cfg.max_acceleration)

        # ── 2. Inner-loop: compute T and τ ────────────────────────────────
        R = _quat_to_rotmat(self._q)
        T, tau = self._compute_inner_loop(a_cmd, R)

        # ── 3. RK4 integration ────────────────────────────────────────────
        p_new, v_new, q_new, omega_new = self._rk4_step(
            self._state.position.copy(),
            self._state.velocity.copy(),
            self._q.copy(),
            self._omega.copy(),
            T, tau, wind_disturbance, dt,
        )

        # ── 4. Post-integration constraints ──────────────────────────────
        v_new = clip_norm(v_new, self._cfg.max_velocity)
        omega_new = clip_norm(omega_new, self._max_omega)
        p_new[2] = max(0.0, p_new[2])

        # ── 5. Compute reported applied acceleration ── ───────────────────
        # a_applied is the net non-gravitational + non-wind acceleration
        # that the pursuer body actually exerts, expressed in world frame.
        R_new = _quat_to_rotmat(q_new)
        a_applied = (T / self._m) * R_new[:, 2] - _G * _E3

        # ── 6. Store ──────────────────────────────────────────────────────
        self._q = q_new
        self._omega = omega_new
        self._state = PursuerState(
            position=p_new,
            velocity=v_new,
            applied_acceleration=a_applied,
        )
        return self._state

    @property
    def state(self) -> PursuerState:
        return self._state

    @property
    def quaternion(self) -> np.ndarray:
        """Current attitude quaternion [qw, qx, qy, qz]."""
        return self._q.copy()

    @property
    def angular_velocity(self) -> np.ndarray:
        """Body-frame angular velocity [rad/s]."""
        return self._omega.copy()

    @property
    def euler_angles_deg(self) -> np.ndarray:
        """ZYX Euler angles (roll, pitch, yaw) in degrees."""
        return _euler_from_quat(self._q)

    # ── private helpers ───────────────────────────────────────────────────────

    def _compute_inner_loop(
        self,
        a_cmd_world: np.ndarray,
        R: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Convert outer-loop commanded acceleration to (thrust T, torque τ).

        Returns
        -------
        T : float
            Thrust magnitude [N].
        tau_body : (3,) ndarray
            Body-frame torque [N·m].
        """
        # ── Thrust channel ────────────────────────────────────────────────
        # Desired force in world frame including gravity compensation
        f_des = self._m * (a_cmd_world + _G * _E3)
        T_des = float(np.linalg.norm(f_des))
        T = float(np.clip(T_des, 0.0, self._max_thrust))

        # Desired thrust direction (body-z target in world frame)
        z_b_des = f_des / T_des if T_des > 1e-6 else _E3.copy()

        # ── Attitude error → desired angular-velocity setpoint ────────────
        z_b = R[:, 2]   # current body-z expressed in world frame

        cross = np.cross(z_b, z_b_des)
        cross_norm = float(np.linalg.norm(cross))
        dot_val = float(np.clip(np.dot(z_b, z_b_des), -1.0, 1.0))
        theta_err = np.arccos(dot_val)

        if cross_norm > 1e-9:
            axis_w = cross / cross_norm
            omega_att_des_world = (theta_err / self._tau_att) * axis_w
        else:
            omega_att_des_world = np.zeros(3)

        # Clip desired angular rate
        omega_att_des_world = clip_norm(omega_att_des_world, self._max_omega)

        # ── Rate controller (body frame) ──────────────────────────────────
        omega_des_body = R.T @ omega_att_des_world
        Jomega = self._J @ self._omega
        tau = (
            self._J @ (self._k_rate * (omega_des_body - self._omega))
            + np.cross(self._omega, Jomega)
        )

        # Saturate torque per axis
        tau = clip_components(tau, self._max_torque)

        return T, tau

    def _derivatives(
        self,
        p: np.ndarray,
        v: np.ndarray,
        q: np.ndarray,
        omega: np.ndarray,
        T: float,
        tau: np.ndarray,
        a_wind: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate the 6DOF ODE right-hand side."""
        R = _quat_to_rotmat(q)
        # Translational
        dp = v.copy()
        dv = (T / self._m) * R[:, 2] - _G * _E3 + a_wind
        # Rotational
        dq = _quat_derivative(q, omega)
        domega = self._J_inv @ (tau - np.cross(omega, self._J @ omega))
        return dp, dv, dq, domega

    def _rk4_step(
        self,
        p: np.ndarray,
        v: np.ndarray,
        q: np.ndarray,
        omega: np.ndarray,
        T: float,
        tau: np.ndarray,
        a_wind: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fourth-order Runge-Kutta step for the 13-state ODE."""

        def f(p_, v_, q_, o_):
            return self._derivatives(p_, v_, q_, o_, T, tau, a_wind)

        # k1
        dp1, dv1, dq1, do1 = f(p, v, q, omega)

        # k2
        q2 = q + 0.5 * dt * dq1
        q2 = q2 / np.linalg.norm(q2)
        dp2, dv2, dq2, do2 = f(
            p + 0.5*dt*dp1, v + 0.5*dt*dv1, q2, omega + 0.5*dt*do1
        )

        # k3
        q3 = q + 0.5 * dt * dq2
        q3 = q3 / np.linalg.norm(q3)
        dp3, dv3, dq3, do3 = f(
            p + 0.5*dt*dp2, v + 0.5*dt*dv2, q3, omega + 0.5*dt*do2
        )

        # k4
        q4 = q + dt * dq3
        q4 = q4 / np.linalg.norm(q4)
        dp4, dv4, dq4, do4 = f(
            p + dt*dp3, v + dt*dv3, q4, omega + dt*do3
        )

        # Combine
        new_p = p + (dt / 6.0) * (dp1 + 2.0*dp2 + 2.0*dp3 + dp4)
        new_v = v + (dt / 6.0) * (dv1 + 2.0*dv2 + 2.0*dv3 + dv4)
        new_q = q + (dt / 6.0) * (dq1 + 2.0*dq2 + 2.0*dq3 + dq4)
        new_q = new_q / np.linalg.norm(new_q)   # re-normalise
        new_omega = omega + (dt / 6.0) * (do1 + 2.0*do2 + 2.0*do3 + do4)

        return new_p, new_v, new_q, new_omega