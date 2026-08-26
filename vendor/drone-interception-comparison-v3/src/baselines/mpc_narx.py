"""
EKF-NARX Hybrid MPC (``mpc_ekf_narx``).

Architecture
------------
1. **External EKF** (shared sim-engine estimator) filters raw sensor noise and
   provides a clean 12-state estimate ``[pos, vel, acc, jerk]``.

2. **Internal NARX neural network** receives a sliding window of *W* past EKF
   position / velocity / acceleration estimates (9-D per timestep) and outputs
   *N* future ``[pos, vel]`` predictions (6-D per step) covering the full MPC
   prediction horizon.  The network is trained *online* via a single Adam
   gradient-descent step per simulation timestep, using the EKF state history
   as the supervised signal with a look-ahead of *N* steps.

3. **Tracking MPC** minimises the sum of squared distances between the
   pursuer's predicted trajectory and the NARX-predicted target waypoints.
   Because the target trajectory is precomputed *outside* the NLP, the NLP is
   a pure reference-tracking problem — simpler and faster to solve than
   polynomial-based variants that embed target dynamics inside the cost.

Why NARX?
---------
A Nonlinear AutoRegressive network with eXogenous inputs (NARX) captures
temporal dependencies that polynomial approximations (CV, CA, cubic) miss,
especially during sharp evasive manoeuvres.  Online adaptation means the
network continuously personalises its prediction to the current target's
behaviour within each episode, without requiring pre-training data.

Ablation role
-------------
Comparing ``mpc_ekf_narx`` with ``proposed_full`` (cubic polynomial + EKF)
directly quantifies the gain from replacing the analytical jerk-based
prediction with a learned, data-driven trajectory prediction.

MPC parameter vector
--------------------
::

    P = [x0_pursuer(9), wp_0(6), wp_1(6), ..., wp_{N-1}(6), u_prev(3), wind(3)]
      size = 15 + 6·N    (N=20 → 135)

Input normalisation
-------------------
All position features are expressed *relative to the most recent EKF
position*, making the encoding translation-invariant.  Fixed scales:
``POS_SCALE=100 m``, ``VEL_SCALE=20 m/s``, ``ACC_SCALE=15 m/s²``.
"""

from __future__ import annotations

from collections import deque
import time as _time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import casadi as ca
    _HAS_CASADI = True
except ImportError:
    _HAS_CASADI = False

from src.utils.config_schema import ControllerConfig, PursuerConfig, SimulationConfig
from src.utils.math_helpers import clip_norm
from src.control.controller_base import ControllerBase

try:
    from src.control.acados_wrapper import build_acados_mpc
    _HAS_ACADOS = True
except ImportError:
    _HAS_ACADOS = False


# ── Online NARX neural network (pure numpy, Adam optimiser) ───────────────────

class _NARXNet:
    """Two-hidden-layer MLP trained online via Adam (MSE loss).

    All weight operations are in-place numpy; no external ML framework
    required.

    Parameters
    ----------
    d_in, d_h1, d_h2, d_out : int
        Layer widths.
    lr : float
        Adam learning rate.
    seed : int
        RNG seed for He initialisation.
    """

    def __init__(
        self,
        d_in: int,
        d_h1: int,
        d_h2: int,
        d_out: int,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)

        def _he(fan_in: int, fan_out: int) -> np.ndarray:
            return rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)

        self.W1 = _he(d_in, d_h1)
        self.b1 = np.zeros(d_h1)
        self.W2 = _he(d_h1, d_h2)
        self.b2 = np.zeros(d_h2)
        self.W3 = _he(d_h2, d_out)
        self.b3 = np.zeros(d_out)

        # Flat list of all trainable parameters (views into the arrays above)
        self._params: List[np.ndarray] = [
            self.W1, self.b1, self.W2, self.b2, self.W3, self.b3
        ]
        self._m = [np.zeros_like(p) for p in self._params]
        self._v = [np.zeros_like(p) for p in self._params]

        self._lr = lr
        self._beta1 = 0.9
        self._beta2 = 0.999
        self._eps = 1e-8
        self._t = 0

        # Cached activations for backprop (populated by forward())
        self._x: np.ndarray = np.empty(0)
        self._z1: np.ndarray = np.empty(0)
        self._a1: np.ndarray = np.empty(0)
        self._z2: np.ndarray = np.empty(0)
        self._a2: np.ndarray = np.empty(0)

    # ── forward / backward ────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass; caches intermediates for a subsequent ``train_step``."""
        self._x = x
        self._z1 = x @ self.W1 + self.b1
        self._a1 = np.maximum(0.0, self._z1)        # ReLU
        self._z2 = self._a1 @ self.W2 + self.b2
        self._a2 = np.maximum(0.0, self._z2)        # ReLU
        return self._a2 @ self.W3 + self.b3

    def train_step(self, x: np.ndarray, y_true: np.ndarray) -> float:
        """Forward pass + backprop + Adam weight update.

        Parameters
        ----------
        x : (d_in,) normalised input
        y_true : (d_out,) normalised target

        Returns
        -------
        loss : float
            Scalar MSE loss before the update.
        """
        y_pred = self.forward(x)
        residual = y_pred - y_true
        n = float(y_true.shape[0])
        loss = float(np.dot(residual, residual) / n)

        # ── backprop ──────────────────────────────────────────────────────
        g3 = 2.0 * residual / n
        dW3 = np.outer(self._a2, g3)
        db3 = g3

        g2 = (g3 @ self.W3.T) * (self._z2 > 0.0)
        dW2 = np.outer(self._a1, g2)
        db2 = g2

        g1 = (g2 @ self.W2.T) * (self._z1 > 0.0)
        dW1 = np.outer(self._x, g1)
        db1 = g1

        grads = [dW1, db1, dW2, db2, dW3, db3]

        # ── Adam update ───────────────────────────────────────────────────
        self._t += 1
        bc1 = 1.0 - self._beta1 ** self._t
        bc2 = 1.0 - self._beta2 ** self._t

        for i, (p, g) in enumerate(zip(self._params, grads)):
            self._m[i] = self._beta1 * self._m[i] + (1.0 - self._beta1) * g
            self._v[i] = self._beta2 * self._v[i] + (1.0 - self._beta2) * g * g
            m_hat = self._m[i] / bc1
            v_hat = self._v[i] / bc2
            p -= self._lr * m_hat / (np.sqrt(v_hat) + self._eps)   # in-place

        return loss

    def reset_optimizer(self) -> None:
        """Reset Adam moment estimates (keep learned weights)."""
        self._m = [np.zeros_like(p) for p in self._params]
        self._v = [np.zeros_like(p) for p in self._params]
        self._t = 0


# ── Main controller class ──────────────────────────────────────────────────────

class MPCNARXPredictor(ControllerBase):
    """EKF-NARX hybrid MPC: learned trajectory prediction inside a tracking MPC.

    The external EKF (provided by the simulation engine) supplies filtered
    target-state estimates.  An internal NARX neural network predicts the
    target trajectory over the full MPC horizon using a sliding window of
    recent EKF states.  The MPC then minimises distance to the
    NARX-predicted waypoints via a simple tracking formulation.

    Before the NARX has accumulated enough history (*warmup_steps* =
    ``narx_window + horizon``), the controller falls back to a
    constant-acceleration prediction using the EKF acceleration estimate,
    identical in behaviour to ``mpc_ca``.

    All hyper-parameters are read from *ctrl_cfg* (fields ``narx_window``,
    ``narx_lr``, ``narx_hidden1``, ``narx_hidden2``, ``narx_Q_pos``,
    ``narx_Q_terminal_pos``).  Nothing is hardcoded.

    Parameters
    ----------
    ctrl_cfg : ControllerConfig
    pursuer_cfg : PursuerConfig
    sim_cfg : SimulationConfig
    """

    # Fixed normalisation scales (based on scenario statistics)
    _POS_SCALE: float = 100.0   # m     — typical range of position deviations
    _VEL_SCALE: float = 20.0    # m/s   — typical target speed
    _ACC_SCALE: float = 15.0    # m/s²  — typical target acceleration
    _JRK_SCALE: float = 50.0    # m/s³  — typical target jerk (EKF jerk estimate)
    # Residual prediction scales — NARX predicts (actual − cubic) not absolute pos.
    # Cubic error on aggressive manoeuvres is ~1–5 m; using a 5 m scale keeps
    # normalised targets in [-1, 1] range for reliable gradient flow.
    _RESIDUAL_POS_SCALE: float = 5.0  # m    — expected cubic position residual magnitude
    _RESIDUAL_VEL_SCALE: float = 2.0  # m/s  — expected cubic velocity residual magnitude

    def __init__(
        self,
        ctrl_cfg: ControllerConfig,
        pursuer_cfg: PursuerConfig,
        sim_cfg: SimulationConfig,
    ) -> None:
        self._solver_type = getattr(ctrl_cfg, "solver", "casadi")
        if self._solver_type == "acados":
            if not _HAS_ACADOS:
                raise ImportError("acados_template is required for solver='acados'.")
        elif not _HAS_CASADI:
            raise ImportError("CasADi is required for MPCNARXPredictor.")

        self.N: int = ctrl_cfg.horizon
        self.dt: float = sim_cfg.dt
        self.tau: float = pursuer_cfg.actuator_time_constant
        self.a_max: float = pursuer_cfg.max_acceleration
        self.a_max_axis: float = pursuer_cfg.max_acceleration_per_axis
        self.jerk_max: float = pursuer_cfg.max_jerk
        self.v_max: float = pursuer_cfg.max_velocity
        self.Q_vel: float = ctrl_cfg.Q_vel
        self.R: float = ctrl_cfg.R_control
        self.R_rate: float = ctrl_cfg.R_rate
        self.Q_T_vel: float = ctrl_cfg.Q_terminal_vel
        self.fallback_gain: float = ctrl_cfg.fallback_gain
        self._solver_max_iter: int = ctrl_cfg.solver_max_iter
        self._solver_print: int = ctrl_cfg.solver_print_level

        # NARX-specific Q weights (fall back to shared values when not set)
        self.Q_pos: float = (
            ctrl_cfg.narx_Q_pos if ctrl_cfg.narx_Q_pos > 0 else ctrl_cfg.Q_pos
        )
        self.Q_T_pos: float = (
            ctrl_cfg.narx_Q_terminal_pos
            if ctrl_cfg.narx_Q_terminal_pos > 0
            else ctrl_cfg.Q_terminal_pos
        )

        # NARX network hyper-parameters — all from config, nothing hardcoded
        self._W: int = ctrl_cfg.narx_window
        self._narx_lr: float = ctrl_cfg.narx_lr
        self._d_h1: int = ctrl_cfg.narx_hidden1
        self._d_h2: int = ctrl_cfg.narx_hidden2
        self._narx_grad_steps: int = getattr(ctrl_cfg, "narx_grad_steps", 50)
        self._narx_enable_online_training: bool = bool(
            getattr(ctrl_cfg, "narx_enable_online_training", True)
        )
        self._narx_training_period_steps: int = max(
            1, int(getattr(ctrl_cfg, "narx_training_period_steps", 1))
        )
        self._narx_training_deadline_s: float = max(
            0.0, float(getattr(ctrl_cfg, "narx_training_deadline_s", 0.0))
        )
        self._narx_trust_threshold: float = getattr(ctrl_cfg, "narx_trust_threshold", 0.3)
        self._control_step_index: int = 0

        # Phase 2 Q2 revision parameters
        self._narx_residual_baseline: str = getattr(ctrl_cfg, "narx_residual_baseline", "constant_acceleration")
        self._narx_trust_mode: str = getattr(ctrl_cfg, "narx_trust_mode", "prequential")
        self._narx_trust_ema_beta: float = float(getattr(ctrl_cfg, "narx_trust_ema_beta", 0.9))
        self._narx_min_validation_samples: int = int(getattr(ctrl_cfg, "narx_min_validation_samples", 20))
        self._narx_validation_window: int = int(getattr(ctrl_cfg, "narx_validation_window", 100))
        self._narx_freeze_after_training_events: int = int(getattr(ctrl_cfg, "narx_freeze_after_training_events", 0))
        self._narx_seed: int = int(getattr(ctrl_cfg, "narx_seed", 42))

        # Minimum history length before NARX is used:
        # needs W steps as input + N steps as training target
        self._warmup_steps: int = self._W + self.N

        # NARX network —
        #   input : W steps × 12 features [pos_dev(3), vel(3), acc(3), jrk(3)]
        #   output: N steps × 6  features [pos_dev(3), vel(3)]
        d_in = self._W * 12
        d_out = self.N * 6
        self._narx = _NARXNet(d_in, self._d_h1, self._d_h2, d_out, lr=self._narx_lr, seed=self._narx_seed)

        # RNG for replay-buffer random sampling (seeded for reproducibility)
        self._rng = np.random.default_rng(self._narx_seed)

        self._narx_ready: bool = False   # True once warmup complete

        # Causal prequential validation structures
        self._issued_predictions: deque = deque(maxlen=self._narx_validation_window)
        self._narx_prequential_loss: Optional[float] = None
        self._narx_validation_sample_count: int = 0
        self._training_events_count: int = 0
        self._narx_ema_loss: Optional[float] = None

        # History buffer: list of 12-D arrays [pos, vel, acc, jrk] from EKF.
        self._history: List[np.ndarray] = []

        self._u_prev: np.ndarray = np.zeros(3)
        self._warm_x0: Optional[np.ndarray] = None

        if self._solver_type == "acados":
            self._acados_solver, self._acados_export_dir = build_acados_mpc(
                ctrl_cfg, pursuer_cfg, sim_cfg, self.Q_pos, self.Q_T_pos
            )
        else:
            self._build_nlp()

    # ── NLP construction ──────────────────────────────────────────────────

    def _build_nlp(self) -> None:
        """Build the CasADi NLP (tracking MPC with pre-computed NARX waypoints)."""
        N = self.N
        dt = self.dt
        alpha = min(dt / self.tau, 1.0) if self.tau > 1e-9 else 1.0
        n_u = 3 * N

        U = ca.MX.sym("U", n_u)

        # Parameter layout:
        #   P[0:9]           = x0_pursuer  (pos, vel, applied_acc)
        #   P[9 : 9+6N]      = NARX waypoints, row-major  (N × [pos(3), vel(3)])
        #   P[9+6N : 9+6N+3] = u_prev
        #   P[9+6N+3 : 15+6N]= wind
        p_size = 15 + 6 * N
        P = ca.MX.sym("P", p_size)

        x0 = P[0:9]
        u_prev_sym = P[9 + 6 * N : 9 + 6 * N + 3]
        wind = P[9 + 6 * N + 3 : 9 + 6 * N + 6]

        cost = 0.0
        g_list = []
        x = x0

        for k in range(N):
            u_k = U[3 * k : 3 * (k + 1)]
            p_p, v_p, a_app = x[0:3], x[3:6], x[6:9]

            # First-order actuator lag
            a_app_new = a_app + alpha * (u_k - a_app)
            total_acc = a_app_new + wind
            v_new = v_p + total_acc * dt
            p_new = p_p + v_p * dt + 0.5 * total_acc * dt ** 2
            x = ca.vertcat(p_new, v_new, a_app_new)

            # NARX-predicted target waypoint at step k+1
            wp_k = P[9 + 6 * k : 9 + 6 * (k + 1)]   # [pos(3), vel(3)]
            p_t = wp_k[0:3]
            v_t = wp_k[3:6]

            dp = p_new - p_t
            dv = v_new - v_t

            if k < N - 1:
                cost += self.Q_pos * ca.dot(dp, dp) + self.Q_vel * ca.dot(dv, dv)
            else:
                cost += self.Q_T_pos * ca.dot(dp, dp) + self.Q_T_vel * ca.dot(dv, dv)

            cost += self.R * ca.dot(u_k, u_k)

            if k == 0:
                du = (u_k - u_prev_sym) / dt
            else:
                du = (u_k - U[3 * (k - 1) : 3 * k]) / dt
            cost += self.R_rate * ca.dot(du, du)

            g_list.append(ca.dot(u_k, u_k))       # acceleration norm²
            g_list.append(ca.dot(v_new, v_new))    # velocity norm²
            g_list.append(ca.dot(du, du))           # jerk norm²

        nlp = {"f": cost, "x": U, "g": ca.vertcat(*g_list), "p": P}
        opts = {
            "ipopt.max_iter": self._solver_max_iter,
            "ipopt.print_level": self._solver_print,
            "print_time": 0,
            "ipopt.sb": "yes",
            "ipopt.warm_start_init_point": "yes",
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
        }
        self._solver = ca.nlpsol("mpc_ekf_narx", "ipopt", nlp, opts)
        self._n_u = n_u
        self._p_size = p_size
        self._lbx = np.full(n_u, -self.a_max_axis)
        self._ubx = np.full(n_u, self.a_max_axis)
        self._lbg = np.zeros(3 * N)
        self._ubg = np.tile([self.a_max ** 2, self.v_max ** 2, self.jerk_max ** 2], N)

    # ── NARX helpers ──────────────────────────────────────────────────────

    def _encode_window(
        self,
        states: List[np.ndarray],
        ref_pos: np.ndarray,
    ) -> np.ndarray:
        """Normalise *W* EKF states into a flat NARX input vector.

        Positions are expressed relative to *ref_pos* (translation-invariant).
        Velocities, accelerations, and jerks are divided by fixed scales.

        Parameters
        ----------
        states : list of W 12-D arrays  [pos(3), vel(3), acc(3), jrk(3)]
        ref_pos : (3,) reference position (last known target position)

        Returns
        -------
        x : (W × 12,) normalised input
        """
        parts: List[np.ndarray] = []
        for s in states:
            pos_dev = (s[0:3] - ref_pos) / self._POS_SCALE
            vel_n   = s[3:6] / self._VEL_SCALE
            acc_n   = s[6:9] / self._ACC_SCALE
            jrk_n   = s[9:12] / self._JRK_SCALE if len(s) >= 12 else np.zeros(3)
            parts.append(np.concatenate([pos_dev, vel_n, acc_n, jrk_n]))
        return np.concatenate(parts)

    def _baseline_pred_from_state(
        self, s: np.ndarray, n_steps: int
    ) -> np.ndarray:
        """Compute baseline Taylor prediction from a 12-D EKF state.

        Returns an ``(n_steps, 6)`` array of ``[pos(3), vel(3)]`` waypoints.
        Used both in the fallback and in the replay training target computation.
        """
        pt, vt, at = s[0:3], s[3:6], s[6:9]
        if self._narx_residual_baseline == "cubic":
            jt = s[9:12] if len(s) >= 12 else np.zeros(3)
        else:
            jt = np.zeros(3)
        wp = np.empty((n_steps, 6))
        for k in range(n_steps):
            tau = (k + 1) * self.dt
            wp[k, 0:3] = pt + vt * tau + 0.5 * at * tau ** 2 + (1.0 / 6.0) * jt * tau ** 3
            wp[k, 3:6] = vt + at * tau + 0.5 * jt * tau ** 2
        return wp

    def _decode_waypoints(
        self,
        pred_norm: np.ndarray,
        ref_pos: np.ndarray,
    ) -> np.ndarray:
        pred = pred_norm.reshape(self.N, 6)
        res = np.empty_like(pred)
        res[:, 0:3] = pred[:, 0:3] * self._RESIDUAL_POS_SCALE
        res[:, 3:6] = pred[:, 3:6] * self._RESIDUAL_VEL_SCALE
        return res

    def _narx_predict(self) -> np.ndarray:
        W = self._W
        ref_pos = self._history[-1][0:3].copy()
        x_in = self._encode_window(self._history[-W:], ref_pos)
        pred_norm = self._narx.forward(x_in)          # (N*6,)
        residuals = self._decode_waypoints(pred_norm, ref_pos)  # (N, 6)
        residuals[:, 0:3] = np.clip(
            residuals[:, 0:3], -self._RESIDUAL_POS_SCALE, self._RESIDUAL_POS_SCALE
        )
        residuals[:, 3:6] = np.clip(
            residuals[:, 3:6], -self._RESIDUAL_VEL_SCALE, self._RESIDUAL_VEL_SCALE
        )
        baseline_wp = self._baseline_pred_from_state(self._history[-1], self.N)
        return baseline_wp + residuals

    def _narx_train_with_replay(self, n_steps: int) -> float:
        W, N = self._W, self.N
        n_available = len(self._history) - W - N
        if n_available < 1:
            return float('inf')

        recent_start = max(0, n_available - 50)
        total_loss = 0.0
        for _ in range(n_steps):
            j = int(self._rng.integers(recent_start, n_available))
            input_states  = self._history[j : j + W]
            target_states = self._history[j + W : j + W + N]

            ref_pos = input_states[-1][0:3].copy()
            x_in    = self._encode_window(input_states, ref_pos)

            baseline_wp = self._baseline_pred_from_state(input_states[-1], N)

            y_parts: List[np.ndarray] = []
            for k, s in enumerate(target_states):
                res_pos = (s[0:3] - baseline_wp[k, 0:3]) / self._RESIDUAL_POS_SCALE
                res_vel = (s[3:6] - baseline_wp[k, 3:6]) / self._RESIDUAL_VEL_SCALE
                y_parts.append(np.concatenate([res_pos, res_vel]))
            y_true = np.concatenate(y_parts)

            total_loss += self._narx.train_step(x_in, y_true)

        val_n = min(5, n_available)
        val_loss = 0.0
        for j_v in range(n_available - val_n, n_available):
            input_states  = self._history[j_v : j_v + W]
            target_states = self._history[j_v + W : j_v + W + N]
            ref_pos = input_states[-1][0:3].copy()
            x_in    = self._encode_window(input_states, ref_pos)
            baseline_wp = self._baseline_pred_from_state(input_states[-1], N)
            y_parts_v: List[np.ndarray] = []
            for k, s in enumerate(target_states):
                res_pos = (s[0:3] - baseline_wp[k, 0:3]) / self._RESIDUAL_POS_SCALE
                res_vel = (s[3:6] - baseline_wp[k, 3:6]) / self._RESIDUAL_VEL_SCALE
                y_parts_v.append(np.concatenate([res_pos, res_vel]))
            y_true_v = np.concatenate(y_parts_v)
            y_pred_v = self._narx.forward(x_in)
            diff_v   = y_pred_v - y_true_v
            val_loss += float(np.dot(diff_v, diff_v) / len(y_true_v))

        return val_loss / val_n

    def _cubic_fallback_waypoints(self, ekf_state: np.ndarray) -> np.ndarray:
        return self._baseline_pred_from_state(ekf_state, self.N)

    # ── ControllerBase implementation ─────────────────────────────────────

    def compute_control(
        self,
        pursuer_state: np.ndarray,
        target_measurement: Optional[np.ndarray],
        target_estimate: np.ndarray,
        wind_estimate: np.ndarray,
        t: float,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._history.append(target_estimate[0:12].copy())

        max_buf = max(self._warmup_steps + 1, 200)
        if len(self._history) > max_buf:
            self._history = self._history[-max_buf:]

        step_idx = self._control_step_index
        self._control_step_index += 1
        step_t0 = _time.perf_counter()

        # Causal prequential validation of previously issued predictions
        for item in list(self._issued_predictions):
            k_issued = item["step_idx"]
            m = step_idx - k_issued
            if 1 <= m <= self.N and m not in item["validated_steps"]:
                item["validated_steps"].add(m)
                issued_narx = item["narx_wp"][m - 1]
                actual_p = target_estimate[0:3]
                actual_v = target_estimate[3:6]
                res_pos = (actual_p - issued_narx[0:3]) / self._RESIDUAL_POS_SCALE
                res_vel = (actual_v - issued_narx[3:6]) / self._RESIDUAL_VEL_SCALE
                sample_loss = float((np.dot(res_pos, res_pos) + np.dot(res_vel, res_vel)) / 6.0)

                if self._narx_prequential_loss is None:
                    self._narx_prequential_loss = sample_loss
                else:
                    b = self._narx_trust_ema_beta
                    self._narx_prequential_loss = b * self._narx_prequential_loss + (1.0 - b) * sample_loss
                self._narx_validation_sample_count += 1

        # Online training execution check
        narx_loss: Optional[float] = None
        narx_train_time_s = 0.0
        narx_training_due = False
        narx_training_executed = False
        narx_training_skipped_deadline = False

        if len(self._history) >= self._warmup_steps + 1:
            self._narx_ready = True
            is_frozen = (
                self._narx_freeze_after_training_events > 0
                and self._training_events_count >= self._narx_freeze_after_training_events
            )
            narx_training_due = (
                self._narx_enable_online_training
                and not is_frozen
                and self._narx_grad_steps > 0
                and step_idx % self._narx_training_period_steps == 0
            )
            if narx_training_due:
                elapsed_before_train = _time.perf_counter() - step_t0
                if (
                    self._narx_training_deadline_s > 0.0
                    and elapsed_before_train >= self._narx_training_deadline_s
                ):
                    narx_training_skipped_deadline = True
                else:
                    train_t0 = _time.perf_counter()
                    narx_loss = self._narx_train_with_replay(self._narx_grad_steps)
                    narx_train_time_s = _time.perf_counter() - train_t0
                    narx_training_executed = True
                    self._training_events_count += 1
                    if self._narx_ema_loss is None:
                        self._narx_ema_loss = narx_loss
                    else:
                        self._narx_ema_loss = 0.9 * self._narx_ema_loss + 0.1 * narx_loss

        # Target waypoints calculation & trust factor evaluation
        baseline_wp = self._baseline_pred_from_state(target_estimate, self.N)
        narx_infer_time_s = 0.0
        if self._narx_ready:
            infer_t0 = _time.perf_counter()
            narx_wp = self._narx_predict()
            narx_infer_time_s = _time.perf_counter() - infer_t0

            if self._narx_trust_mode == "always_on":
                trust = 1.0 if (self._narx_ready and self._training_events_count >= 1) else 0.0
            elif self._narx_trust_mode == "always_off":
                trust = 0.0
            else:  # "prequential"
                if (
                    self._narx_validation_sample_count < self._narx_min_validation_samples
                    or self._narx_prequential_loss is None
                ):
                    trust = 0.0
                else:
                    trust = float(np.clip(
                        1.0 - self._narx_prequential_loss / self._narx_trust_threshold, 0.0, 1.0
                    ))
            waypoints = trust * narx_wp + (1.0 - trust) * baseline_wp
        else:
            narx_wp = baseline_wp
            waypoints = baseline_wp
            trust = 0.0

        # Save issued prediction for causal validation
        self._issued_predictions.append({
            "step_idx": step_idx,
            "baseline_wp": baseline_wp.copy(),
            "narx_wp": narx_wp.copy(),
            "validated_steps": set(),
        })

        narx_timing_info = {
            "narx_training_period_steps": self._narx_training_period_steps,
            "narx_training_due": narx_training_due,
            "narx_training_executed": narx_training_executed,
            "narx_training_skipped_deadline": narx_training_skipped_deadline,
            "narx_train_time_s": narx_train_time_s,
            "narx_infer_time_s": narx_infer_time_s,
        }

        if self._solver_type == "acados":
            return self._compute_control_acados(
                pursuer_state=pursuer_state,
                target_estimate=target_estimate,
                wind_estimate=wind_estimate,
                waypoints=waypoints,
                narx_loss=narx_loss,
                trust=trust,
                narx_timing_info=narx_timing_info,
            )

        # Build NLP parameter vector
        p_param = np.zeros(self._p_size)
        p_param[0:9] = pursuer_state
        p_param[9 : 9 + 6 * self.N] = waypoints.ravel()
        p_param[9 + 6 * self.N : 9 + 6 * self.N + 3] = self._u_prev
        p_param[9 + 6 * self.N + 3 : 9 + 6 * self.N + 6] = wind_estimate

        x0_guess = self._warm_x0 if self._warm_x0 is not None else np.zeros(self._n_u)

        # Solve MPC NLP
        t0 = _time.perf_counter()
        try:
            sol = self._solver(
                x0=x0_guess, p=p_param,
                lbx=self._lbx, ubx=self._ubx,
                lbg=self._lbg, ubg=self._ubg,
            )
            solve_time = _time.perf_counter() - t0
            u_opt = np.array(sol["x"]).ravel()
            stats = self._solver.stats()
            status = stats.get("return_status", "unknown")
            ok = status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")

            if ok:
                cmd = u_opt[0:3]
                self._warm_x0 = np.concatenate([u_opt[3:], u_opt[-3:]])
            else:
                cmd = self._fallback_pn(pursuer_state, target_estimate)
        except Exception:
            solve_time = _time.perf_counter() - t0
            cmd = self._fallback_pn(pursuer_state, target_estimate)
            status = "exception"
            ok = False

        self._u_prev = cmd.copy()
        return cmd, {
            "solver_status": status,
            "solver_success": ok,
            "solve_time_s": solve_time,
            "estimator": "ekf",
            "narx_ready": self._narx_ready,
            "narx_train_loss": float(narx_loss) if narx_loss is not None else float("nan"),
            "narx_prequential_loss": float(self._narx_prequential_loss) if self._narx_prequential_loss is not None else float("nan"),
            "narx_validation_sample_count": int(self._narx_validation_sample_count),
            "narx_ema_loss": self._narx_ema_loss,
            "narx_trust": trust if self._narx_ready else 0.0,
            "narx_training_events_count": int(self._training_events_count),
            **narx_timing_info,
        }

    def _compute_control_acados(
        self,
        pursuer_state: np.ndarray,
        target_estimate: np.ndarray,
        wind_estimate: np.ndarray,
        waypoints: np.ndarray,
        narx_loss: Optional[float],
        trust: float,
        narx_timing_info: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        t0 = _time.perf_counter()
        ok = False
        try:
            x0 = np.concatenate([pursuer_state[0:9], self._u_prev])
            self._acados_solver.set(0, "lbx", x0)
            self._acados_solver.set(0, "ubx", x0)

            wind = wind_estimate
            for k in range(self.N):
                self._acados_solver.set(
                    k, "p", np.concatenate([waypoints[k, 0:3], waypoints[k, 3:6], wind])
                )
            self._acados_solver.set(
                self.N, "p", np.concatenate([waypoints[-1, 0:3], waypoints[-1, 3:6], wind])
            )

            status_code = self._acados_solver.solve()
            solve_time = _time.perf_counter() - t0

            if status_code in (0, 2):
                cmd = self._acados_solver.get(0, "u")[0:3]
                ok = True
                status = f"acados_{status_code}"
            else:
                cmd = self._fallback_pn(pursuer_state, target_estimate)
                status = f"acados_fail_{status_code}"
        except Exception:
            solve_time = _time.perf_counter() - t0
            cmd = self._fallback_pn(pursuer_state, target_estimate)
            status = "exception"
            ok = False

        self._u_prev = np.asarray(cmd, dtype=float).copy()
        return cmd, {
            "solver_status": status,
            "solver_success": ok,
            "solve_time_s": solve_time,
            "estimator": "ekf",
            "narx_ready": self._narx_ready,
            "narx_train_loss": float(narx_loss) if narx_loss is not None else float("nan"),
            "narx_prequential_loss": float(self._narx_prequential_loss) if self._narx_prequential_loss is not None else float("nan"),
            "narx_validation_sample_count": int(self._narx_validation_sample_count),
            "narx_ema_loss": self._narx_ema_loss,
            "narx_trust": trust if self._narx_ready else 0.0,
            "narx_training_events_count": int(self._training_events_count),
            **narx_timing_info,
        }

    def reset(self) -> None:
        """Reset controller state for a new episode."""
        self._history.clear()
        self._issued_predictions.clear()
        self._narx_ready = False
        self._narx_prequential_loss = None
        self._narx_validation_sample_count = 0
        self._training_events_count = 0
        self._narx_ema_loss = None
        self._control_step_index = 0
        self._u_prev = np.zeros(3)
        self._warm_x0 = None
        # Re-initialize _NARXNet and RNG with deterministic seed
        d_in = self._W * 12
        d_out = self.N * 6
        self._narx = _NARXNet(d_in, self._d_h1, self._d_h2, d_out, lr=self._narx_lr, seed=self._narx_seed)
        self._rng = np.random.default_rng(self._narx_seed)

    def _fallback_pn(
        self, pursuer_state: np.ndarray, target_est: np.ndarray
    ) -> np.ndarray:
        """PN guidance fallback when IPOPT fails."""
        r = target_est[0:3] - pursuer_state[0:3]
        v_rel = target_est[3:6] - pursuer_state[3:6]
        rn = np.linalg.norm(r)
        if rn < 1e-6:
            return np.zeros(3)
        rh = r / rn
        Vc = -np.dot(v_rel, rh)
        omega = np.cross(r, v_rel) / rn ** 2
        return clip_norm(self.fallback_gain * Vc * np.cross(omega, rh), self.a_max)