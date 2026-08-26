"""Single-interceptor Global Integrated Proportional Navigation."""

from __future__ import annotations
import numpy as np

K1 = 40.0
K2 = 1.0
V_R_MPS = -5.0
EPS_RANGE_M = 1e-9


def gpn_command(target_position, target_velocity, interceptor_position, interceptor_velocity):
    r_vec = np.asarray(target_position, float) - np.asarray(interceptor_position, float)
    v_rel = np.asarray(target_velocity, float) - np.asarray(interceptor_velocity, float)
    range_raw = float(np.linalg.norm(r_vec))
    guarded = range_raw < EPS_RANGE_M
    r = max(range_raw, EPS_RANGE_M)
    line_of_sight = r_vec / r
    range_rate = float(line_of_sight @ v_rel)
    los_rate = (v_rel - range_rate * line_of_sight) / r
    command = (
        (-2.0 * range_rate + K1) * los_rate
        + (K2 * (range_rate - V_R_MPS) + np.asarray(target_velocity, float) @ los_rate)
        * line_of_sight
    )
    if not np.all(np.isfinite(command)):
        raise FloatingPointError("GPN produced nonfinite acceleration")
    return command, {
        "gpn_range_m": range_raw, "gpn_range_rate_mps": range_rate,
        "gpn_los_rate_norm_s_inv": float(np.linalg.norm(los_rate)),
        "gpn_range_guard_activated": int(guarded), "gpn_k1": K1,
        "gpn_k2": K2, "gpn_vr_mps": V_R_MPS, "target_rollout_used": 0,
    }
