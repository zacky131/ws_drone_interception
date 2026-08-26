#!/usr/bin/env python3
"""Create the frozen-shape (but not yet physically frozen) 24-run mirror manifest."""

from __future__ import annotations
import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["run_id", "trajectory", "condition", "delay_ms", "method", "method_code", "repetition", "seed", "nominal_initial_range_m", "trial_timeout_s", "status"]


def rows():
    codes = {"M0prime": "A0prime_CA_arrival", "M1": "mpc_dca_tracking"}
    delays = {"HC0": 50, "HC1": 120}
    for trajectory_index, trajectory in enumerate(("HT1", "HT2"), 1):
        for condition_index, condition in enumerate(("HC0", "HC1"), 1):
            for repetition in range(1, 4):
                seed = 10_000 * trajectory_index + 1_000 * condition_index + repetition
                for method in ("M0prime", "M1"):
                    yield {"run_id": f"{trajectory}_{condition}_{method}_rep{repetition:02d}", "trajectory": trajectory, "condition": condition, "delay_ms": delays[condition], "method": method, "method_code": codes[method], "repetition": repetition, "seed": seed, "nominal_initial_range_m": 18.0, "trial_timeout_s": 30.0, "status": "pending"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "manifests/hardware_mirror_24.csv")
    args = parser.parse_args()
    data = list(rows())
    if len(data) != 24:
        raise RuntimeError("manifest must have 24 rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(data)
    print(f"wrote {len(data)} unexecuted runs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
