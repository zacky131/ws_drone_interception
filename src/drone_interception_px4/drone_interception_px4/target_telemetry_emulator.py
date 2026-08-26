"""CLI for materializing a frozen deterministic telemetry schedule."""

from __future__ import annotations

import argparse
from pathlib import Path

from .telemetry import CONDITIONS, generate_schedule, save_schedule


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    schedule = generate_schedule(CONDITIONS[args.condition], args.duration_s, 0.02, args.seed)
    print(save_schedule(schedule, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

