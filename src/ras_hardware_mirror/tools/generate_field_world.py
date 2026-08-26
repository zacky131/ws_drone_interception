#!/usr/bin/env python3
"""Generate the Gazebo football-field SDF from the measured/development field YAML."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml


ROOT = Path(__file__).resolve().parents[1]


def box_visual(name: str, x: float, y: float, z: float, sx: float, sy: float, sz: float, color: str) -> str:
    return f'''<model name="{name}"><static>true</static><pose>{x:.6g} {y:.6g} {z:.6g} 0 0 0</pose><link name="link"><visual name="visual"><geometry><box><size>{sx:.6g} {sy:.6g} {sz:.6g}</size></box></geometry><material><ambient>{color}</ambient><diffuse>{color}</diffuse></material></visual></link></model>'''


def rectangle(name: str, bounds: dict, z: float, color: str, thickness: float = 0.10) -> list[str]:
    e0, e1 = float(bounds["east_min_m"]), float(bounds["east_max_m"])
    n0, n1 = float(bounds["north_min_m"]), float(bounds["north_max_m"])
    return [
        box_visual(f"{name}_south", (e0 + e1) / 2, n0, z, e1 - e0, thickness, 0.03, color),
        box_visual(f"{name}_north", (e0 + e1) / 2, n1, z, e1 - e0, thickness, 0.03, color),
        box_visual(f"{name}_west", e0, (n0 + n1) / 2, z, thickness, n1 - n0, 0.03, color),
        box_visual(f"{name}_east", e1, (n0 + n1) / 2, z, thickness, n1 - n0, 0.03, color),
    ]


def generate(config_path: Path, output_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    field = config["field"]
    length, width = float(field["length_m"]), float(field["width_m"])
    if length <= 0 or width <= 0:
        raise ValueError("field dimensions must be positive")
    markings = rectangle("field_boundary", {"east_min_m": -length / 2, "east_max_m": length / 2, "north_min_m": -width / 2, "north_max_m": width / 2}, 0.025, "1 1 1 1", 0.12)
    markings.append(box_visual("center_line", 0, 0, 0.025, 0.12, width, 0.03, "1 1 1 1"))
    # Twelve short boxes approximate a center circle without adding dynamic geometry.
    import math
    for index in range(12):
        angle = 2 * math.pi * index / 12
        markings.append(box_visual(f"center_circle_{index:02d}", 9.15 * math.cos(angle), 9.15 * math.sin(angle), 0.026, 0.8, 0.12, 0.03, "1 1 1 1"))
    markings.extend(rectangle("hard_geofence", config["hard_geofence"], 0.09, "0.9 0.05 0.05 1", 0.16))
    markings.extend(rectangle("target_region", config["target_region"], 0.07, "1 0.75 0.05 1", 0.12))
    template = (ROOT / "worlds/football_field_template.sdf").read_text(encoding="utf-8")
    rendered = template.replace("{{FIELD_LENGTH}}", f"{length:.9g}").replace("{{FIELD_WIDTH}}", f"{width:.9g}").replace("{{FIELD_VISUALS}}", "\n    ".join(markings))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config/field.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "worlds/football_field.sdf")
    args = parser.parse_args()
    print(generate(args.config.resolve(), args.output.resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
