"""Small standard-message helpers shared by mirror nodes."""

from __future__ import annotations

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
import numpy as np


def stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def set_stamp(stamp, seconds: float) -> None:
    whole = int(seconds)
    stamp.sec = whole
    stamp.nanosec = int(round((float(seconds) - whole) * 1e9))
    if stamp.nanosec >= 1_000_000_000:
        stamp.sec += 1
        stamp.nanosec -= 1_000_000_000


def odometry(frame: str, child: str, stamp_s: float, position, velocity, covariance=None) -> Odometry:
    msg = Odometry()
    msg.header.frame_id = frame
    msg.child_frame_id = child
    set_stamp(msg.header.stamp, stamp_s)
    p, v = np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)
    msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = map(float, p)
    msg.pose.pose.orientation.w = 1.0
    msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z = map(float, v)
    if covariance is not None:
        flat = np.asarray(covariance, dtype=float).reshape(6, 6)
        msg.pose.covariance = flat.reshape(-1).tolist()
    return msg


def odom_vectors(msg: Odometry) -> tuple[np.ndarray, np.ndarray]:
    p, v = msg.pose.pose.position, msg.twist.twist.linear
    return np.array([p.x, p.y, p.z], dtype=float), np.array([v.x, v.y, v.z], dtype=float)


def point(value) -> Point:
    p = Point()
    p.x, p.y, p.z = map(float, value)
    return p


def diagnostic(name: str, level: int = DiagnosticStatus.OK, message: str = "OK", **values) -> DiagnosticArray:
    array = DiagnosticArray()
    status = DiagnosticStatus()
    status.name = name
    status.hardware_id = "ras_hardware_mirror"
    # This Humble installation exposes uint8 message fields as one-byte values.
    status.level = bytes(level) if isinstance(level, (bytes, bytearray)) else bytes([int(level)])
    status.message = str(message)
    status.values = [KeyValue(key=str(key), value=str(value)) for key, value in values.items()]
    array.status = [status]
    return array


def diagnostic_values(msg: DiagnosticArray) -> dict[str, str]:
    if not msg.status:
        return {}
    return {item.key: item.value for item in msg.status[0].values}
