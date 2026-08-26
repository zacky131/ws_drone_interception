"""Pure terminal-keyboard mappings shared by the controller and tests."""

from __future__ import annotations

from typing import Iterable
import numpy as np


SCENARIOS = {
    "1": {"stage": "GS1", "method": "M1", "trajectory": "STATIC", "condition": "DEV0", "seed": 6101},
    "2": {"stage": "GS2", "method": "M1", "trajectory": "HT1", "condition": "DEV0", "seed": 6202},
    "3": {"stage": "GS3", "method": "M1", "trajectory": "HT1", "condition": "HC0", "seed": 6303},
    "4": {"stage": "GS4", "method": "M1", "trajectory": "HT1", "condition": "HC1", "seed": 6404},
    "5": {"stage": "GS5_M0prime", "method": "M0prime", "trajectory": "HT1", "condition": "HC1", "seed": 6505},
    "6": {"stage": "GS5_M1", "method": "M1", "trajectory": "HT1", "condition": "HC1", "seed": 6505},
    "7": {"stage": "GS6", "method": "M1", "trajectory": "HT1", "condition": "HC1", "seed": 6606},
}

MOTION_KEYS = frozenset(("w", "s", "a", "d", "Up", "Down", "Left", "Right"))

ARROW_KEYS = {"A": "Up", "B": "Down", "C": "Right", "D": "Left"}


def decode_terminal_input(data: bytes, pending: bytes = b"") -> tuple[list[str], bytes]:
    """Decode ASCII keys and common SSH/xterm arrow sequences.

    A trailing partial escape sequence is returned to be prepended to the next
    read. Both CSI (ESC [ A) and application-cursor (ESC O A) forms work.
    """
    buffer = pending + data
    keys: list[str] = []
    index = 0
    while index < len(buffer):
        value = buffer[index]
        if value == 0x1B:
            remaining = len(buffer) - index
            if remaining < 2:
                break
            if buffer[index + 1] in (ord("["), ord("O")):
                if remaining < 3:
                    break
                arrow = ARROW_KEYS.get(chr(buffer[index + 2]))
                if arrow is not None:
                    keys.append(arrow)
                    index += 3
                    continue
            index += 1
            continue
        if 0x20 <= value <= 0x7E:
            keys.append(chr(value).lower())
        index += 1
    return keys, bytes(buffer[index:])


def held_key_command(held: Iterable[str], horizontal: float, vertical: float, yaw_rate: float) -> np.ndarray:
    """Return body-FLU [forward, left, up, yaw-left] velocity command."""
    keys = set(held)
    return np.array([
        horizontal * (int("Up" in keys) - int("Down" in keys)),
        horizontal * (int("Left" in keys) - int("Right" in keys)),
        vertical * (int("w" in keys) - int("s" in keys)),
        yaw_rate * (int("a" in keys) - int("d" in keys)),
    ], dtype=float)


def px4_fmu_prefix(namespace: str) -> str:
    clean = str(namespace).strip("/")
    return f"/{clean}/fmu" if clean else "/fmu"
