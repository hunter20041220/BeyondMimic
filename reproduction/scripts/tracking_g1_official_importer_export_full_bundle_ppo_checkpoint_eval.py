#!/usr/bin/env python3
"""Evaluate the official-importer-export full-bundle PPO checkpoint."""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py"
OUT = ROOT / "res/tracking/g1_official_importer_export_full_bundle_ppo_checkpoint_eval"
LOG_DIR = ROOT / "logs/tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval"
OFFICIAL_IMPORTER_USD = (
    ROOT
    / "res/tracking/g1_urdf_in_memory_gpu4_probe/"
    "g1_official_importer_in_memory_gpu4_export.usda"
)
FULL_BUNDLE_MOTION_NPZ = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_motion_npz/"
    "official_csv_loop_full_public_motion_bundle.npz"
)
FULL_BUNDLE_AUDIT = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_motion_npz/"
    "tracking_g1_official_csv_loop_full_bundle_motion_npz.json"
)
TRAINING_RUN_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_full_bundle_ppo_training_run/"
    "tracking_g1_official_importer_export_full_bundle_ppo_training_run.json"
)
TARGET_GPUS = [4, 7]
DEFAULT_SEED = 20260681


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_resource_adjusted_ppo_eval_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base PPO eval script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_base_compatible_training_summary() -> Path:
    training = load_json(TRAINING_RUN_JSON)
    compatible = dict(training)
    compatible["status"] = "ok_resource_adjusted_ppo_training_completed"
    compatible.setdefault("inputs", {})
    compatible["inputs"]["original_official_importer_export_training_run_json"] = str(TRAINING_RUN_JSON)
    compatible.setdefault("interpretation", {})
    compatible["interpretation"]["base_compatibility_shim"] = (
        "This copy only adapts the status enum for the shared checkpoint-eval harness. "
        "The authoritative training audit remains the original official-importer-export training JSON."
    )
    shim_path = OUT / "base_compatible_official_importer_export_training_run_for_eval.json"
    shim_path.write_text(json.dumps(compatible, indent=2, sort_keys=True), encoding="utf-8")
    return shim_path


def patch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    bundle = load_json(FULL_BUNDLE_AUDIT)
    training = load_json(TRAINING_RUN_JSON)
    bundle_info = bundle.get("bundle", {})
    eval_ok = summary.get("status") == "ok_resource_adjusted_ppo_checkpoint_eval_completed"
    summary["status"] = (
        "ok_official_importer_export_full_bundle_ppo_checkpoint_eval_completed"
        if eval_ok
        else summary.get("status", "failed_official_importer_export_full_bundle_ppo_checkpoint_eval")
    )
    summary["experiment_type"] = "tracking_official_importer_export_full_bundle_ppo_checkpoint_eval"
    summary["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    summary["scope"] = (
        "Evaluates the PPO checkpoint trained with the official-importer GPU4 G1 USDA export and the concatenated "
        "40-motion public official csv-loop bundle inside Tracking-Flat-G1-v0. This is a local virtual policy "
        "evaluation, not the official paper teacher-policy result."
    )
    summary.setdefault("inputs", {})
    summary["inputs"].update(
        {
            "training_run_json": str(TRAINING_RUN_JSON),
            "base_compatible_training_run_json": str(OUT / "base_compatible_official_importer_export_training_run_for_eval.json"),
            "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
            "asset_note": (
                "The inherited base-harness key `enriched_usd` is also present in this JSON, but its value has been "
                "overridden to the official-importer-export USDA path for this evaluation."
            ),
            "full_bundle_motion_npz": str(FULL_BUNDLE_MOTION_NPZ),
            "full_bundle_audit": str(FULL_BUNDLE_AUDIT),
        }
    )
    summary.setdefault("input_checks", {})
    summary["input_checks"].update(
        {
            "official_importer_usd_exists": OFFICIAL_IMPORTER_USD.is_file(),
            "full_bundle_motion_npz_exists": FULL_BUNDLE_MOTION_NPZ.is_file(),
            "full_bundle_audit_passed": bundle.get("status")
            == "ok_official_csv_loop_full_bundle_motion_npz",
            "full_bundle_motion_count_40": bundle_info.get("motion_count") == 40,
            "full_bundle_total_frames_11960": bundle_info.get("total_frames") == 11960,
            "training_run_completed": training.get("status")
            == "ok_official_importer_export_full_bundle_ppo_training_completed",
        }
    )
    if "run" in summary and isinstance(summary["run"].get("metrics"), dict):
        metrics = summary["run"]["metrics"]
        metrics["uses_official_importer_export_usd"] = True
        metrics["uses_resource_adjusted_usd"] = False
        metrics["official_csv_loop_full_public_bundle"] = True
        metrics["official_csv_to_npz_unpatched_output"] = False
        metrics["paper_level_tracking_eval"] = False
        metrics["motion_count"] = bundle_info.get("motion_count")
        metrics["total_motion_frames"] = bundle_info.get("total_frames")
    final_json = OUT / "tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval.json"
    summary.setdefault("outputs", {})
    summary["outputs"]["json"] = str(final_json)
    summary["outputs"]["worker_script"] = str(OUT / "tracking_g1_resource_adjusted_ppo_checkpoint_eval_worker.py")
    summary["interpretation"] = {
        "goal_complete": False,
        "official_importer_export_full_bundle_checkpoint_eval_complete": bool(eval_ok),
        "official_tracking_eval_complete": False,
        "paper_level_tracking_eval_complete": False,
        "why_not_paper_level": (
            "The evaluated checkpoint was trained locally on public motion data with a local official-importer USDA "
            "asset. It is useful simulation evidence, but not an official BeyondMimic teacher checkpoint, not the "
            "paper's full training protocol, and not real-robot validation."
        ),
    }
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    module = load_base_module()
    compatible_training_summary = make_base_compatible_training_summary()
    module.OUT = OUT
    module.LOG_DIR = LOG_DIR
    module.RUN_ROOT = RUN_ROOT
    module.ENRICHED_USD = OFFICIAL_IMPORTER_USD
    module.CSV_MOTION_NPZ = FULL_BUNDLE_MOTION_NPZ
    module.TRAINING_RUN_JSON = compatible_training_summary
    module.CANDIDATE_GPUS = TARGET_GPUS
    module.SEED = int(os.environ.get("BM_OFFICIAL_IMPORTER_EXPORT_PPO_EVAL_SEED", str(DEFAULT_SEED)))
    module.main()

    output_json = OUT / "tracking_g1_resource_adjusted_ppo_checkpoint_eval.json"
    final_json = OUT / "tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval.json"
    summary = patch_summary(load_json(output_json))
    output_json.unlink(missing_ok=True)
    final_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(final_json),
                "attempted_eval": summary.get("run", {}).get("attempted_eval"),
                "metrics_exists": summary.get("run", {}).get("metrics_exists"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
