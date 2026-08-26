"""Configuration and workspace-path helpers."""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(os.environ.get("WS_DRONE_INTERCEPTION", PACKAGE_ROOT.parents[1]))


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return value


def package_file(relative: str) -> Path:
    source = PACKAGE_ROOT / relative
    if source.exists():
        return source
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("ras_hardware_mirror")) / relative
    except Exception:
        return source


def default_config() -> Path:
    return package_file("config/hardware_mirror_dev.yaml")


def default_field() -> Path:
    return package_file("config/field.yaml")


def load_mirror_config(path: str | Path | None = None) -> dict[str, Any]:
    config = load_yaml(default_config() if path is None or not str(path) else path)
    required = ("experiment", "interceptor", "virtual_target", "capture", "telemetry_conditions", "controller")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"hardware-mirror config missing keys: {missing}")
    frozen_config = Path(str(config["controller"]["frozen_config_path"])).expanduser()
    if not frozen_config.is_absolute():
        frozen_config = WORKSPACE_ROOT / frozen_config
    frozen_config = frozen_config.resolve()
    if not frozen_config.is_file():
        raise FileNotFoundError(f"frozen controller config not found: {frozen_config}")
    config["controller"]["frozen_config_path"] = str(frozen_config)
    return deepcopy(config)


def method_code(public_name: str) -> str:
    mapping = {"M0prime": "A0prime_CA_arrival", "M1": "mpc_dca_tracking"}
    try:
        return mapping[public_name]
    except KeyError as exc:
        raise ValueError(f"method must be one of {tuple(mapping)}, got {public_name!r}") from exc


def condition_config(config: dict[str, Any], condition: str) -> dict[str, Any]:
    try:
        return dict(config["telemetry_conditions"][condition])
    except KeyError as exc:
        raise ValueError(f"unknown telemetry condition: {condition}") from exc
