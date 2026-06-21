#!/usr/bin/env python3
"""Create report assets for robot-order FK-repaired full-bundle PPO eval."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/official_importer_export_full_bundle_ppo_eval_report_assets.py"


def main() -> None:
    os.environ["BM_IMPORTER_PPO_REPORT_EVAL_AUDIT"] = str(
        ROOT
        / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval/"
        "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval.json"
    )
    os.environ["BM_IMPORTER_PPO_REPORT_TRAINING_AUDIT"] = str(
        ROOT
        / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run/"
        "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run.json"
    )
    os.environ["BM_IMPORTER_PPO_REPORT_OUT"] = str(
        ROOT
        / "res/report_assets/"
        "official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval"
    )
    os.environ[
        "BM_IMPORTER_PPO_REPORT_ASSET_JSON_NAME"
    ] = "official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_assets.json"
    os.environ["BM_IMPORTER_PPO_REPORT_TITLE"] = (
        "Robot-order FK-repaired official-importer-export PPO checkpoint tracking errors"
    )
    os.environ["BM_IMPORTER_PPO_REPORT_TRAINING_TITLE"] = (
        "Robot-order FK-repaired official-importer-export PPO training curve"
    )
    os.environ["BM_IMPORTER_PPO_REPORT_CLAIM_LEVEL"] = (
        "local_virtual_official_importer_export_robot_order_fk_repaired_ppo_report_asset"
    )
    spec = importlib.util.spec_from_file_location("bm_importer_ppo_report_assets", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load report asset script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
