"""Verified PX4 v1.15.2 namespace, identity, heartbeat, and message contract."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np

from .frames import enu_acceleration_to_ned


@dataclass(frozen=True)
class VehicleIdentity:
    role: str
    instance: int
    namespace: str
    expected_system_id: int

    def verify(self, reported_system_id: int) -> None:
        if int(reported_system_id) != self.expected_system_id:
            raise RuntimeError(
                f"{self.role} {self.namespace} reported MAV system "
                f"{reported_system_id}, expected {self.expected_system_id}"
            )


INTERCEPTOR = VehicleIdentity("interceptor", 1, "px4_1", 2)
TARGET = VehicleIdentity("target", 2, "px4_2", 3)


def px4_topic(identity: VehicleIdentity, direction: str, name: str) -> str:
    if direction not in {"in", "out"}:
        raise ValueError("direction must be 'in' or 'out'")
    return f"/{identity.namespace}/fmu/{direction}/{name}"


def command_fields(identity: VehicleIdentity) -> dict[str, int | bool]:
    return {
        "target_system": identity.expected_system_id,
        "target_component": 1,
        "source_system": 255,
        "source_component": 1,
        "confirmation": 0,
        "from_external": True,
    }


def acceleration_setpoint_fields(acceleration_enu: Iterable[float]) -> dict[str, object]:
    """Return v1.15.2 TrajectorySetpoint fields for acceleration control."""
    acceleration_ned = enu_acceleration_to_ned(acceleration_enu).astype(float)
    nan3 = [math.nan, math.nan, math.nan]
    return {
        "position": nan3.copy(),
        "velocity": nan3.copy(),
        "acceleration": acceleration_ned.tolist(),
        "jerk": nan3.copy(),
        "yaw": math.nan,
        "yawspeed": math.nan,
    }


def offboard_acceleration_mode_fields() -> dict[str, bool]:
    return {
        "position": False,
        "velocity": False,
        "acceleration": True,
        "attitude": False,
        "body_rate": False,
        "thrust_and_torque": False,
        "direct_actuator": False,
    }


def heartbeat_rate_hz(timestamps_ns: Sequence[int]) -> float:
    if len(timestamps_ns) < 2:
        return 0.0
    intervals = np.diff(np.asarray(timestamps_ns, dtype=np.int64)) / 1e9
    if np.any(intervals <= 0.0):
        raise ValueError("heartbeat timestamps must be strictly monotonic")
    return float(1.0 / np.max(intervals))


def assert_offboard_heartbeat(timestamps_ns: Sequence[int], minimum_hz: float = 2.0) -> None:
    rate = heartbeat_rate_hz(timestamps_ns)
    if rate < minimum_hz:
        raise RuntimeError(f"Offboard heartbeat worst-case rate {rate:.3f} Hz < {minimum_hz:.3f} Hz")

