#!/usr/bin/env python3
"""Generate a paired M0-prime/M1 summary without modifying run data."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("m0prime", type=Path)
    parser.add_argument("m1", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metadata = [json.loads((path / "metadata.json").read_text()) for path in (args.m0prime, args.m1)]
    paired = (metadata[0]["trajectory"], metadata[0]["condition"], metadata[0]["seed"])
    if paired != (metadata[1]["trajectory"], metadata[1]["condition"], metadata[1]["seed"]):
        raise SystemExit("refusing unpaired analysis: trajectory/condition/seed differ")
    frames = {label: pd.read_csv(path / "steps.csv") for label, path in (("M0prime", args.m0prime), ("M1", args.m1))}
    packets = {label: pd.read_csv(path / "telemetry_packets.csv") for label, path in (("M0prime", args.m0prime), ("M1", args.m1))}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    colors = {"M0prime": "#D55E00", "M1": "#0072B2"}
    for label, frame in frames.items():
        axes[0, 0].plot(frame.target_truth_e_m, frame.target_truth_n_m, ":", color="black", alpha=0.55)
        axes[0, 0].plot(frame.interceptor_px4_e_m, frame.interceptor_px4_n_m, label=label, color=colors[label])
        error = np.sqrt(sum((frame[f"target_estimate_{a}_m"] - frame[f"target_truth_{a}_m"])**2 for a in "enu"))
        t = frame.ros_time_s - frame.ros_time_s.iloc[0]
        axes[0, 1].plot(t, error, label=label, color=colors[label])
        axes[1, 0].plot(t, frame.separation_m, label=label, color=colors[label])
        packet = packets[label]
        axes[1, 1].plot(packet.arrival_time_s - packet.arrival_time_s.iloc[0], packet.actual_age_ms, label=label, color=colors[label])
    axes[0, 0].set(title="Top-down trajectories", xlabel="East (m)", ylabel="North (m)", aspect="equal")
    axes[0, 1].set(title="Target-state position error", xlabel="Time (s)", ylabel="Error (m)")
    axes[1, 0].axhline(1.0, color="#A51C30", ls="-.")
    axes[1, 0].set(title="Virtual separation", xlabel="Time (s)", ylabel="Separation (m)", ylim=(0, None))
    axes[1, 1].set(title="Packet age", xlabel="Time (s)", ylabel="Age (ms)")
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    summary = " | ".join(f"{m['method']}: capture={m.get('captured')}, min={m.get('minimum_separation_m')} m" for m in metadata)
    fig.suptitle(f"{paired[0]} / {paired[1]} / seed {paired[2]}\n{summary}")
    output = args.output or args.m1.parent / f"paired_{paired[0]}_{paired[1]}_seed{paired[2]}"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    print(output.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
