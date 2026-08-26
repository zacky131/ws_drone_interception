#!/usr/bin/env python3
"""
scripts/split_dataset_c.py

Phase 2: Deterministic Stratified Split for Dataset C.
Uses seed 20260729 to split 200 trajectories into:
- Pilot (20% = 5 per family = 40 trajectories)
- Confirmatory (80% = 20 per family = 160 trajectories)

Outputs:
- results/q2_revision_v1/dataset_c/manifests/dataset_c_pilot.txt
- results/q2_revision_v1/dataset_c/manifests/dataset_c_confirmatory.txt
- results/q2_revision_v1/dataset_c/manifests/dataset_c_split_manifest.json
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd

META_CSV = "/home/t462/Documents/Zacks_research/drone-interception-comparison-v3/results/q2_revision_v1/dataset_c/trajectory_metadata.csv"
OUTPUT_MANIFEST_DIR = "/home/t462/Documents/Zacks_research/drone-interception-comparison-v3/results/q2_revision_v1/dataset_c/manifests"
DATASET_CSV_DIR = "/home/t462/Documents/Zacks_research/drone-interception-comparison-v3/data/q2_challenge_v1/generated_6dof/csv"

SPLIT_SEED = 20260729

def main():
    print(f"=== Phase 2: Stratified Pilot/Confirmatory Split (Seed {SPLIT_SEED}) ===")
    os.makedirs(OUTPUT_MANIFEST_DIR, exist_ok=True)

    df_meta = pd.read_csv(META_CSV)
    valid_meta = df_meta[df_meta['is_valid'] == True].copy()

    pilot_files = []
    confirmatory_files = []
    split_info = {}

    rng = np.random.RandomState(SPLIT_SEED)

    families = sorted(valid_meta['family'].unique())
    for fam in families:
        fam_df = valid_meta[valid_meta['family'] == fam].sort_values(by='filename').reset_index(drop=True)
        n = len(fam_df)
        n_pilot = max(1, int(round(n * 0.20)))  # 5 for 25
        
        # Shuffle deterministically
        indices = np.arange(n)
        rng.shuffle(indices)

        pilot_idx = set(indices[:n_pilot])
        conf_idx = set(indices[n_pilot:])

        fam_pilot = fam_df.iloc[sorted(list(pilot_idx))]['filename'].tolist()
        fam_conf = fam_df.iloc[sorted(list(conf_idx))]['filename'].tolist()

        pilot_files.extend(fam_pilot)
        confirmatory_files.extend(fam_conf)

        split_info[fam] = {
            "total": n,
            "pilot_count": len(fam_pilot),
            "confirmatory_count": len(fam_conf)
        }

    pilot_files = sorted(pilot_files)
    confirmatory_files = sorted(confirmatory_files)

    # Verify non-overlap
    overlap = set(pilot_files).intersection(set(confirmatory_files))
    assert len(overlap) == 0, f"Error: Pilot and confirmatory sets overlap! {overlap}"
    assert len(pilot_files) + len(confirmatory_files) == len(valid_meta), "Error: Total file count mismatch!"

    pilot_txt = os.path.join(OUTPUT_MANIFEST_DIR, "dataset_c_pilot.txt")
    with open(pilot_txt, 'w') as f:
        for fn in pilot_files:
            f.write(f"csv/{fn}\n")
    print(f"Saved pilot manifest ({len(pilot_files)} files) to {pilot_txt}")

    conf_txt = os.path.join(OUTPUT_MANIFEST_DIR, "dataset_c_confirmatory.txt")
    with open(conf_txt, 'w') as f:
        for fn in confirmatory_files:
            f.write(f"csv/{fn}\n")
    print(f"Saved confirmatory manifest ({len(confirmatory_files)} files) to {conf_txt}")

    # Split manifest JSON
    split_manifest = {
        "dataset_name": "Q2 Quadrotor 6-DOF Challenge Dataset C Split",
        "split_seed": SPLIT_SEED,
        "total_valid_trajectories": len(valid_meta),
        "pilot_count": len(pilot_files),
        "confirmatory_count": len(confirmatory_files),
        "non_overlap_assertion": True,
        "family_split_summary": split_info,
        "pilot_files": pilot_files,
        "confirmatory_files": confirmatory_files
    }
    split_json = os.path.join(OUTPUT_MANIFEST_DIR, "dataset_c_split_manifest.json")
    with open(split_json, 'w') as f:
        json.dump(split_manifest, f, indent=2)
    print(f"Saved split manifest JSON to {split_json}")
    print("=== Phase 2 Complete! ===")

if __name__ == "__main__":
    main()
