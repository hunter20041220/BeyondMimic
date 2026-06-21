#!/usr/bin/env python3
"""Evaluate the FK-repaired full-bundle PPO checkpoint."""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py"
OUT = ROOT / "res/tracking/g1_official_importer_export_fk_repaired_full_bundle_ppo_checkpoint_eval"
LOG_DIR = ROOT / "logs/tracking_g1_official_importer_export_fk_repaired_full_bundle_ppo_checkpoint_eval"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_importer_export_fk_repaired_full_bundle_ppo_checkpoint_eval"
TRAINING_RUN_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_full_bundle_ppo_training_run/"
    "tracking_g1_official_importer_export_fk_repaired_full_bundle_ppo_training_run.json"
)
OFFICIAL_IMPORTER_USD = (
    ROOT
    / "res/tracking/g1_urdf_in_memory_gpu4_probe/"
    "g1_official_importer_in_memory_gpu4_export.usda"
)
FK_BUNDLE_AUDIT = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_motion_npz/"
    "tracking_g1_official_csv_loop_full_bundle_fk_repaired_motion_npz.json"
)
FK_BUNDLE_NPZ = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_motion_npz/"
    "official_csv_loop_full_public_motion_bundle_fk_repaired.npz"
)
TARGET_GPUS = [4, 7]
DEFAULT_SEED = 20260702
DEFAULT_NUM_ENVS = 2048


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


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
    compatible["inputs"]["original_fk_repaired_full_bundle_training_run_json"] = str(TRAINING_RUN_JSON)
    compatible.setdefault("interpretation", {})
    compatible["interpretation"]["base_compatibility_shim"] = (
        "This copy adapts the FK-repaired training status enum for the shared checkpoint-eval harness. "
        "The authoritative training audit remains the original FK-repaired training JSON."
    )
    shim_path = OUT / "base_compatible_fk_repaired_full_bundle_training_run_for_eval.json"
    shim_path.write_text(json.dumps(compatible, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return shim_path


def patch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    training = load_json(TRAINING_RUN_JSON)
    bundle = load_json(FK_BUNDLE_AUDIT)
    bundle_info = bundle.get("bundle", {})
    eval_ok = summary.get("status") == "ok_resource_adjusted_ppo_checkpoint_eval_completed"
    final_json = OUT / "tracking_g1_official_importer_export_fk_repaired_full_bundle_ppo_checkpoint_eval.json"
    summary["status"] = (
        "ok_official_importer_export_fk_repaired_full_bundle_ppo_checkpoint_eval_completed"
        if eval_ok
        else summary.get("status", "failed_official_importer_export_fk_repaired_full_bundle_ppo_checkpoint_eval")
    )
    summary["experiment_type"] = "tracking_official_importer_export_fk_repaired_full_bundle_ppo_checkpoint_eval"
    summary["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    summary["scope"] = (
        "Evaluates the local PPO checkpoint trained on the official-importer-export G1 USDA and the FK-repaired "
        "40-motion public bundle. This is the tracking-repair checkpoint evaluation path, not an official "
        "BeyondMimic paper teacher evaluation."
    )
    summary.setdefault("inputs", {})
    summary["inputs"].update(
        {
            "training_run_json": str(TRAINING_RUN_JSON),
            "base_compatible_training_run_json": str(
                OUT / "base_compatible_fk_repaired_full_bundle_training_run_for_eval.json"
            ),
            "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
            "fk_repaired_motion_npz": str(FK_BUNDLE_NPZ),
            "fk_repaired_bundle_audit": str(FK_BUNDLE_AUDIT),
        }
    )
    summary.setdefault("input_checks", {})
    summary["input_checks"].update(
        {
            "training_run_completed": training.get("status")
            == "ok_official_importer_export_fk_repaired_full_bundle_ppo_training_completed",
            "official_importer_usd_exists": OFFICIAL_IMPORTER_USD.is_file(),
            "fk_repaired_motion_npz_exists": FK_BUNDLE_NPZ.is_file(),
            "fk_repaired_bundle_audit_passed": bundle.get("status")
            == "ok_official_csv_loop_full_bundle_fk_repaired_motion_npz",
            "fk_repaired_motion_count_40": bundle_info.get("motion_count") == 40,
            "fk_repaired_total_frames_11960": bundle_info.get("total_frames") == 11960,
        }
    )
    summary.setdefault("config", {})
    summary["config"]["seed"] = int(os.environ.get("BM_FK_REPAIRED_FULL_BUNDLE_PPO_EVAL_SEED", str(DEFAULT_SEED)))
    summary["config"]["num_envs"] = int(
        os.environ.get("BM_FK_REPAIRED_FULL_BUNDLE_PPO_EVAL_NUM_ENVS", str(DEFAULT_NUM_ENVS))
    )
    if "run" in summary and isinstance(summary["run"].get("metrics"), dict):
        metrics = summary["run"]["metrics"]
        metrics["uses_official_importer_export_usd"] = True
        metrics["uses_resource_adjusted_usd"] = False
        metrics["uses_fk_repaired_full_public_motion_bundle"] = True
        metrics["uses_old_degenerate_full_public_motion_bundle"] = False
        metrics["official_csv_to_npz_unpatched_output"] = False
        metrics["paper_level_tracking_eval"] = False
        metrics["motion_count"] = bundle_info.get("motion_count")
        metrics["total_motion_frames"] = bundle_info.get("total_frames")
    summary.setdefault("outputs", {})
    summary["outputs"]["json"] = str(final_json)
    summary["outputs"]["worker_script"] = str(OUT / "tracking_g1_resource_adjusted_ppo_checkpoint_eval_worker.py")
    summary["interpretation"] = {
        "goal_complete": False,
        "fk_repaired_full_bundle_checkpoint_eval_complete": bool(eval_ok),
        "official_tracking_eval_complete": False,
        "paper_level_tracking_eval_complete": False,
        "why_not_paper_level": (
            "The evaluated checkpoint is locally trained on a non-Kit FK-repaired public-motion bundle with a local "
            "official-importer USDA. It is useful virtual tracking evidence, but not an official BeyondMimic "
            "teacher checkpoint, not DAgger, not the paper full protocol, and not real-robot validation."
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
    module.CSV_MOTION_NPZ = FK_BUNDLE_NPZ
    module.TRAINING_RUN_JSON = compatible_training_summary
    module.CANDIDATE_GPUS = TARGET_GPUS
    module.NUM_ENVS = int(os.environ.get("BM_FK_REPAIRED_FULL_BUNDLE_PPO_EVAL_NUM_ENVS", str(DEFAULT_NUM_ENVS)))
    module.SEED = int(os.environ.get("BM_FK_REPAIRED_FULL_BUNDLE_PPO_EVAL_SEED", str(DEFAULT_SEED)))
    module.main()

    base_json = OUT / "tracking_g1_resource_adjusted_ppo_checkpoint_eval.json"
    final_json = OUT / "tracking_g1_official_importer_export_fk_repaired_full_bundle_ppo_checkpoint_eval.json"
    summary = patch_summary(load_json(base_json))
    base_json.unlink(missing_ok=True)
    final_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(final_json),
                "attempted_eval": summary.get("run", {}).get("attempted_eval"),
                "metrics_exists": summary.get("run", {}).get("metrics_exists"),
                "num_envs": summary.get("config", {}).get("num_envs"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
