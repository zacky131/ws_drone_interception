#!/usr/bin/env python3
"""Explicitly guarded sequential campaign runner; never auto-tunes or auto-reruns misses."""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import signal
import os
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PACKAGE / "manifests/hardware_mirror_24.csv")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        "automatic campaign disabled: the default-world workflow requires visible foreground processes, "
        "operator arming, and one explicit keyboard manifest selection; use "
        "mirror_keyboard.launch.py manifest_row:=N, then press q, wait for ARMED, and press 8"
    )
    if not args.execute or args.confirm != "GS0-GS6-PASS":
        raise SystemExit("campaign refused: require --execute --confirm GS0-GS6-PASS after visual approval")
    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    if len(rows) != 24:
        raise SystemExit(f"campaign refused: expected 24 manifest rows, found {len(rows)}")
    results = ROOT / "results/ras_hardware_mirror"
    events = results / "campaign_events.csv"
    events.parent.mkdir(parents=True, exist_ok=True)
    with events.open("a", encoding="utf-8", buffering=1) as log:
        for index, row in enumerate(rows, 1):
            output = results / row["run_id"]
            if args.resume and (output / "metadata.json").exists():
                try:
                    if json.loads((output / "metadata.json").read_text()).get("complete"):
                        log.write(f"{time.time()},{row['run_id']},resume_skip,0\n")
                        continue
                except json.JSONDecodeError:
                    pass
            command = ["ros2", "launch", "ras_hardware_mirror", "mirror_demo.launch.py", f"method:={row['method']}", f"trajectory:={row['trajectory']}", f"condition:={row['condition']}", f"seed:={row['seed']}", f"repetition:={row['repetition']}", "gui:=--headless"]
            log.write(f"{time.time()},{row['run_id']},start,{index}/24\n")
            process = subprocess.Popen(command, start_new_session=True)
            deadline = time.monotonic() + 105.0
            complete = False
            try:
                while process.poll() is None and time.monotonic() < deadline:
                    if (output / "metadata.json").exists():
                        try:
                            if json.loads((output / "metadata.json").read_text()).get("complete"):
                                complete = True
                                break
                        except json.JSONDecodeError:
                            pass
                    time.sleep(0.5)
            finally:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    try:
                        process.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGTERM)
                subprocess.run([str(PACKAGE / "scripts/stop_mirror_sim.sh")], check=False, timeout=10)
            log.write(f"{time.time()},{row['run_id']},{'complete' if complete else 'infrastructure_invalid'},{process.returncode}\n")
            if not complete:
                raise SystemExit(f"stopped after infrastructure-invalid attempt: {row['run_id']}; no scientific rerun was made")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
