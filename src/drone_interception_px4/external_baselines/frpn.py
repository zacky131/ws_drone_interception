"""Fast Response Proportional Navigation from Pliska et al. (RA-L 2024)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


PUBLISHED_G = 19.7
PUBLISHED_W = 0.051
EPS_RELATIVE_SPEED_MPS = 1e-9
EPS_TIME_TO_GO_S = 1e-9


def frpn_command(
    delta_p: np.ndarray,
    delta_v: np.ndarray,
    gain: float = PUBLISHED_G,
    blend: float = PUBLISHED_W,
    eps_v: float = EPS_RELATIVE_SPEED_MPS,
    eps_t: float = EPS_TIME_TO_GO_S,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Evaluate the published law with finite-arithmetic guards only."""

    position = np.asarray(delta_p, dtype=float).reshape(3)
    velocity = np.asarray(delta_v, dtype=float).reshape(3)
    distance = float(np.linalg.norm(position))
    relative_speed_raw = float(np.linalg.norm(velocity))
    speed_guard = relative_speed_raw < eps_v
    relative_speed = max(relative_speed_raw, eps_v)
    time_to_go_raw = distance / relative_speed
    time_guard = time_to_go_raw < eps_t
    time_to_go = max(time_to_go_raw, eps_t)
    lpn = (position + velocity * time_to_go) / (time_to_go * time_to_go)
    command = float(gain) * ((1.0 - float(blend)) * lpn + float(blend) * position)
    if not np.all(np.isfinite(command)):
        raise FloatingPointError("FRPN equation produced a nonfinite command")
    return command, {
        "frpn_distance_m": distance,
        "frpn_relative_speed_mps": relative_speed_raw,
        "frpn_time_to_go_s": time_to_go,
        "frpn_speed_guard_activated": int(speed_guard),
        "frpn_time_guard_activated": int(time_guard),
    }


@dataclass
class FRPNGuidance:
    """Frozen published FRPN guidance parameters."""

    gain: float = PUBLISHED_G
    blend: float = PUBLISHED_W

    def __post_init__(self) -> None:
        if self.gain != PUBLISHED_G or self.blend != PUBLISHED_W:
            raise ValueError("experimental FRPN parameters are frozen at G=19.7 and W=0.051")
        self.reset()

    def reset(self) -> None:
        self.speed_guard_activations = 0
        self.time_guard_activations = 0

    def compute(
        self,
        target_position: np.ndarray,
        target_velocity: np.ndarray,
        interceptor_position: np.ndarray,
        interceptor_velocity: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        command, diagnostics = frpn_command(
            np.asarray(target_position) - np.asarray(interceptor_position),
            np.asarray(target_velocity) - np.asarray(interceptor_velocity),
            self.gain,
            self.blend,
        )
        self.speed_guard_activations += diagnostics["frpn_speed_guard_activated"]
        self.time_guard_activations += diagnostics["frpn_time_guard_activated"]
        diagnostics.update(
            {
                "frpn_G": self.gain,
                "frpn_W": self.blend,
                "frpn_speed_guard_activations_total": self.speed_guard_activations,
                "frpn_time_guard_activations_total": self.time_guard_activations,
                "target_rollout_used": 0,
            }
        )
        return command, diagnostics
