"""Load and validate the frozen 24-run operator-gated manifest."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

from .config_utils import package_file


FROZEN_MANIFEST_SHA256 = "e32cc93e8a9a900dd9aeb5c7634894a823d72eb429a3bf48100b0ce5728543bb"
EXPECTED_ROWS = 24
METHOD_CODES = {
    "M0prime": "A0prime_CA_arrival",
    "M1": "mpc_dca_tracking",
}
CONDITION_DELAYS_MS = {"HC0": 50, "HC1": 120}


def default_manifest() -> Path:
    return package_file("manifests/hardware_mirror_24.csv")


def load_campaign_scenarios(path: str | Path | None = None) -> dict[int, dict[str, Any]]:
    """Return immutable-by-validation keyboard requests indexed from 1 to 24."""
    manifest = Path(path).expanduser() if path else default_manifest()
    payload = manifest.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != FROZEN_MANIFEST_SHA256:
        raise ValueError(
            f"campaign manifest checksum mismatch: {manifest} has {digest}, "
            f"expected {FROZEN_MANIFEST_SHA256}"
        )
    raw_rows = list(csv.DictReader(payload.decode("utf-8").splitlines()))
    if len(raw_rows) != EXPECTED_ROWS:
        raise ValueError(f"campaign manifest must contain {EXPECTED_ROWS} rows, found {len(raw_rows)}")

    scenarios: dict[int, dict[str, Any]] = {}
    seen_run_ids: set[str] = set()
    for index, row in enumerate(raw_rows, 1):
        method = str(row["method"])
        trajectory = str(row["trajectory"])
        condition = str(row["condition"])
        repetition = int(row["repetition"])
        seed = int(row["seed"])
        delay_ms = int(row["delay_ms"])
        run_id = str(row["run_id"])
        expected_run_id = f"{trajectory}_{condition}_{method}_rep{repetition:02d}"
        if method not in METHOD_CODES or row["method_code"] != METHOD_CODES[method]:
            raise ValueError(f"invalid method mapping in manifest row {index}")
        if trajectory not in {"HT1", "HT2"}:
            raise ValueError(f"invalid trajectory in manifest row {index}: {trajectory}")
        if condition not in CONDITION_DELAYS_MS or delay_ms != CONDITION_DELAYS_MS[condition]:
            raise ValueError(f"invalid condition/delay in manifest row {index}")
        if repetition not in {1, 2, 3}:
            raise ValueError(f"invalid repetition in manifest row {index}: {repetition}")
        if run_id != expected_run_id or run_id in seen_run_ids:
            raise ValueError(f"invalid or duplicate run_id in manifest row {index}: {run_id}")
        if row.get("status") != "pending":
            raise ValueError(f"manifest row {index} is not frozen as pending")
        seen_run_ids.add(run_id)
        scenarios[index] = {
            "stage": f"HM24_{index:02d}",
            "campaign": "hardware_mirror_24",
            "manifest_row": index,
            "run_id": run_id,
            "method": method,
            "trajectory": trajectory,
            "condition": condition,
            "seed": seed,
            "repetition": repetition,
        }

    for pair_start in range(1, EXPECTED_ROWS + 1, 2):
        left, right = scenarios[pair_start], scenarios[pair_start + 1]
        for key in ("trajectory", "condition", "seed", "repetition"):
            if left[key] != right[key]:
                raise ValueError(f"paired rows {pair_start}/{pair_start + 1} differ in {key}")
        if (left["method"], right["method"]) != ("M0prime", "M1"):
            raise ValueError(f"paired rows {pair_start}/{pair_start + 1} have invalid method order")
    return scenarios

