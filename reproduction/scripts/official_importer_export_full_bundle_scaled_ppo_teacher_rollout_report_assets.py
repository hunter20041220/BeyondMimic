#!/usr/bin/env python3
"""Create report assets for the scaled official-importer-export PPO teacher rollout dataset."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/official_importer_export_full_bundle_teacher_rollout_report_assets.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("bm_importer_teacher_rollout_assets", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base report asset script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROLLOUT_JSON = (
        ROOT
        / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset/"
        "tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset.json"
    )
    module.OUT = (
        ROOT / "res/report_assets/official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset"
    )
    module.main()


if __name__ == "__main__":
    main()
