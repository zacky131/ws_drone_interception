"""
Extended Kalman Filter for 3-D target state estimation.

State vector (12 × 1):
    x = [p_x, p_y, p_z,  v_x, v_y, v_z,  a_x, a_y, a_z,  j_x, j_y, j_z]^T

Process model – *nearly constant jerk*:
    ṗ = v,   v̇ = a,   ȧ = j,   j̇ = w   (w ~ N(0, Q_c))

The state transition matrix F and discrete process-noise matrix Q are derived
analytically from the continuous-time model for a given dt.

Measurement model:
    z = H x + v,   v ~ N(0, R)
    where H picks out position and velocity (6 measurements).
"""

from __future__ import annotations

import numpy as np

from src.utils.config_schema import EstimatorConfig
from .estimator_base import EstimatorBase


class EKFTargetEstimator(EstimatorBase):
    """EKF with a 12-D nearly-constant-jerk target model.

    Addresses the reviewer concern about using only finite-difference / RLS
    estimation by providing a Bayesian filter with explicit process and
    measurement noise handling and covariance diagnostics.
    """

    def __init__(self, config: EstimatorConfig) -> None:
        self._cfg = config
        self._dim_x = 12
        self._dim_z = 6

        # Noise parameters (per-axis, isotropic)
        self._sigma_j = config.process_noise_jerk_std
        self._sigma_p = config.measurement_noise_position_std
        self._sigma_v = config.measurement_noise_velocity_std

        # Measurement matrix  H (6 × 12): picks position and velocity
        self._H = np.zeros((self._dim_z, self._dim_x))
        self._H[0:3, 0:3] = np.eye(3)   # position
        self._H[3:6, 3:6] = np.eye(3)   # velocity

        # Measurement noise covariance R (6 × 6)
        self._R = np.diag([
            self._sigma_p ** 2, self._sigma_p ** 2, self._sigma_p ** 2,
            self._sigma_v ** 2, self._sigma_v ** 2, self._sigma_v ** 2,
        ])

        # State + covariance (uninitialised)
        self._x: np.ndarray | None = None
        self._P: np.ndarray | None = None
        self._initialized = False

        # Innovation logging (latest values)
        self.innovation: np.ndarray = np.zeros(self._dim_z)
        self.innovation_covariance: np.ndarray = np.eye(self._dim_z)

    # ── public interface ──────────────────────────────────────────────────

    def initialize(self, measurement: np.ndarray) -> None:
        z = np.asarray(measurement, dtype=float).ravel()
        self._x = np.zeros(self._dim_x)
        self._x[0:3] = z[0:3]   # position
        self._x[3:6] = z[3:6]   # velocity
        # acceleration and jerk initialised to zero

        diag = np.array([
            self._cfg.initial_position_std ** 2,
            self._cfg.initial_position_std ** 2,
            self._cfg.initial_position_std ** 2,
            self._cfg.initial_velocity_std ** 2,
            self._cfg.initial_velocity_std ** 2,
            self._cfg.initial_velocity_std ** 2,
            self._cfg.initial_acceleration_std ** 2,
            self._cfg.initial_acceleration_std ** 2,
            self._cfg.initial_acceleration_std ** 2,
            self._cfg.initial_jerk_std ** 2,
            self._cfg.initial_jerk_std ** 2,
            self._cfg.initial_jerk_std ** 2,
        ])
        self._P = np.diag(diag)
        self._initialized = True

    def predict(self, dt: float) -> None:
        if not self._initialized:
            return
        F = self._build_F(dt)
        Q = self._build_Q(dt)
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q

    def update(self, measurement: np.ndarray) -> None:
        if not self._initialized:
            self.initialize(measurement)
            return

        z = np.asarray(measurement, dtype=float).ravel()
        H = self._H
        y = z - H @ self._x                   # innovation
        S = H @ self._P @ H.T + self._R       # innovation covariance
        S_inv = np.linalg.inv(S)
        K = self._P @ H.T @ S_inv             # Kalman gain

        self._x = self._x + K @ y
        I_KH = np.eye(self._dim_x) - K @ H
        # Joseph form for numerical stability
        self._P = I_KH @ self._P @ I_KH.T + K @ self._R @ K.T

        # Store diagnostics
        self.innovation = y
        self.innovation_covariance = S

    def get_estimate(self) -> np.ndarray:
        if self._x is None:
            return np.zeros(self._dim_x)
        return self._x.copy()

    def get_covariance(self) -> np.ndarray:
        if self._P is None:
            return np.eye(self._dim_x) * 1e6
        return self._P.copy()

    def reset(self) -> None:
        self._x = None
        self._P = None
        self._initialized = False
        self.innovation = np.zeros(self._dim_z)
        self.innovation_covariance = np.eye(self._dim_z)

    # ── internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _build_F(dt: float) -> np.ndarray:
        """State transition matrix for the nearly-constant-jerk model.

        For each decoupled axis the 4×4 block is:
            | 1   dt   dt²/2   dt³/6 |
            | 0    1   dt      dt²/2 |
            | 0    0    1      dt    |
            | 0    0    0       1    |

        Because the state is ordered [p(3), v(3), a(3), j(3)],
        the full 12×12 F consists of I₃-scaled blocks.
        """
        dt2 = dt * dt
        dt3 = dt2 * dt
        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))

        F = np.block([
            [I3,       dt * I3,  (dt2 / 2) * I3,  (dt3 / 6) * I3],
            [Z3,       I3,       dt * I3,          (dt2 / 2) * I3],
            [Z3,       Z3,       I3,               dt * I3       ],
            [Z3,       Z3,       Z3,               I3            ],
        ])
        return F

    def _build_Q(self, dt: float) -> np.ndarray:
        """Discrete process-noise matrix for the nearly-constant-jerk model.

        Derived from the continuous-time spectral density q_j = σ_j².
        For each axis the 4×4 covariance block is:

            q_j * | dt⁷/252   dt⁶/72   dt⁵/30   dt⁴/24 |
                  | dt⁶/72    dt⁵/20   dt⁴/8    dt³/6  |
                  | dt⁵/30    dt⁴/8    dt³/3    dt²/2  |
                  | dt⁴/24    dt³/6    dt²/2    dt     |

        Assembled into 12×12 with the same block structure as F.
        """
        q = self._sigma_j ** 2
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4
        dt5 = dt ** 5
        dt6 = dt ** 6
        dt7 = dt ** 7

        # Per-axis 4×4 block
        Q1 = q * np.array([
            [dt7 / 252, dt6 / 72, dt5 / 30, dt4 / 24],
            [dt6 / 72,  dt5 / 20, dt4 / 8,  dt3 / 6 ],
            [dt5 / 30,  dt4 / 8,  dt3 / 3,  dt2 / 2 ],
            [dt4 / 24,  dt3 / 6,  dt2 / 2,  dt      ],
        ])

        # Map per-axis block into the full 12×12 matrix
        Q = np.zeros((12, 12))
        # Indices: pos={0,1,2}, vel={3,4,5}, acc={6,7,8}, jrk={9,10,11}
        idx_map = [0, 3, 6, 9]  # block-row starting indices for each axis group
        for ax in range(3):
            for bi in range(4):
                for bj in range(4):
                    Q[idx_map[bi] + ax, idx_map[bj] + ax] = Q1[bi, bj]

        return Q
