#!/usr/bin/env python3
"""Create report assets for robot-order FK-repaired warmup PPO eval."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/official_importer_export_full_bundle_ppo_eval_report_assets.py"
OUT = (
    ROOT
    / "res/report_assets/"
    "official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup"
)
ASSET_JSON_NAME = (
    "official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_assets.json"
)


def patch_warmup_checks(asset_json: Path) -> None:
    data = json.loads(asset_json.read_text(encoding="utf-8"))
    eval_status = ""
    source = Path(data["source_eval_audit"])
    if source.is_file():
        audit = json.loads(source.read_text(encoding="utf-8"))
        eval_status = str(audit.get("status", ""))
        comparison = audit.get("comparison_to_non_warmup_eval", {})
        data["warmup_comparison"] = {
            "old_done_rate": comparison.get("old_done_rate"),
            "warmup_eval_done_rate": comparison.get("warmup_eval_done_rate"),
            "done_rate_delta": comparison.get("done_rate_delta"),
            "old_step0_done_count": comparison.get("old_step0", {}).get("done_count"),
            "warmup_step0_done_count": comparison.get("warmup_eval_step0", {}).get("done_count"),
            "step0_done_count_delta": comparison.get("step0_done_count_delta"),
            "old_step0_body_error": comparison.get("old_step0", {}).get("error_body_pos"),
            "warmup_step0_body_error": comparison.get("warmup_eval_step0", {}).get("error_body_pos"),
        }
    data["experiment_type"] = "robot_order_fk_repaired_checkpoint_eval_warmup_report_assets"
    data["checks"]["eval_status_ok"] = eval_status.startswith("ok_") and eval_status.endswith(
        "_checkpoint_eval_warmup_completed"
    )
    data["checks"]["warmup_eval_status_ok"] = data["checks"]["eval_status_ok"]
    data["checks"]["does_not_claim_goal_complete"] = True
    data["interpretation"] = {
        "paper_level_tracking_reproduced": False,
        "goal_complete": False,
        "summary": (
            "The reset command warmup reduces the step-0 body-target spike, but the total done rate is worse than "
            "the non-warmup robot-order eval. Treat this as a reset/termination diagnostic, not a usable teacher."
        ),
    }
    asset_json.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    os.environ["BM_IMPORTER_PPO_REPORT_EVAL_AUDIT"] = str(
        ROOT
        / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup/"
        "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup.json"
    )
    os.environ["BM_IMPORTER_PPO_REPORT_TRAINING_AUDIT"] = str(
        ROOT
        / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run/"
        "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run.json"
    )
    os.environ["BM_IMPORTER_PPO_REPORT_OUT"] = str(OUT)
    os.environ["BM_IMPORTER_PPO_REPORT_ASSET_JSON_NAME"] = ASSET_JSON_NAME
    os.environ["BM_IMPORTER_PPO_REPORT_TITLE"] = (
        "Robot-order FK-repaired warmup PPO checkpoint tracking errors"
    )
    os.environ["BM_IMPORTER_PPO_REPORT_TRAINING_TITLE"] = (
        "Robot-order FK-repaired official-importer-export PPO training curve"
    )
    os.environ["BM_IMPORTER_PPO_REPORT_CLAIM_LEVEL"] = (
        "local_virtual_robot_order_fk_repaired_ppo_warmup_diagnostic_report_asset"
    )
    spec = importlib.util.spec_from_file_location("bm_importer_ppo_report_assets", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load report asset script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
    patch_warmup_checks(OUT / ASSET_JSON_NAME)
    print(
        json.dumps(
            {"status": "ok", "json": str(OUT / ASSET_JSON_NAME), "out": str(OUT)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
