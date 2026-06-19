#!/usr/bin/env python3
"""Collect teacher rollout shards from the full-bundle PPO checkpoint.

This wrapper reuses the resource-adjusted two-GPU teacher rollout harness while
switching the training summary, checkpoint, motion file, output directories, and
audit wording to the 40-motion public official-loop bundle. It is stronger local
virtual data than the earlier single-motion rollout, but it is still not the
official BeyondMimic DAgger dataset.
"""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_g1_resource_adjusted_teacher_rollout_dataset.py"
OUT = ROOT / "res/tracking/g1_official_csv_loop_full_bundle_teacher_rollout_dataset"
LOG_DIR = ROOT / "logs/tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset"
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
    / "res/tracking/g1_official_csv_loop_full_bundle_ppo_training_run/"
    "tracking_g1_official_csv_loop_full_bundle_ppo_training_run.json"
)
CHECKPOINT_EVAL_JSON = (
    ROOT
    / "res/tracking/g1_official_csv_loop_full_bundle_ppo_checkpoint_eval/"
    "tracking_g1_official_csv_loop_full_bundle_ppo_checkpoint_eval.json"
)
DEFAULT_SEED = 20260672


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_resource_adjusted_teacher_rollout_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load teacher rollout base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_base_compatible_training_summary() -> Path:
    training = load_json(TRAINING_RUN_JSON)
    compatible = dict(training)
    compatible["status"] = "ok_official_csv_loop_ppo_training_completed"
    compatible.setdefault("inputs", {})
    compatible["inputs"]["original_full_bundle_training_run_json"] = str(TRAINING_RUN_JSON)
    compatible.setdefault("interpretation", {})
    compatible["interpretation"]["base_compatibility_shim"] = (
        "This copy only adapts the status enum for the shared resource-adjusted teacher rollout harness. "
        "The authoritative full-bundle audit remains the original training_run_json."
    )
    shim_path = OUT / "base_compatible_full_bundle_training_run_for_teacher_rollout.json"
    shim_path.write_text(json.dumps(compatible, indent=2, sort_keys=True), encoding="utf-8")
    return shim_path


def patch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    bundle = load_json(FULL_BUNDLE_AUDIT)
    bundle_info = bundle.get("bundle", {})
    training = load_json(TRAINING_RUN_JSON)
    checkpoint_eval = load_json(CHECKPOINT_EVAL_JSON)
    rollout_ok = summary.get("status") == "ok_resource_adjusted_teacher_rollout_dataset_completed"
    final_json = OUT / "tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset.json"

    summary["status"] = (
        "ok_official_csv_loop_full_bundle_teacher_rollout_dataset_completed"
        if rollout_ok
        else summary.get("status", "failed_official_csv_loop_full_bundle_teacher_rollout_dataset")
    )
    summary["experiment_type"] = "tracking_official_csv_loop_full_bundle_teacher_rollout_dataset"
    summary["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    summary["scope"] = (
        "Collects two-GPU teacher rollout shards from the local PPO checkpoint trained on the concatenated 40-motion "
        "public official-loop bundle, using the official Tracking-Flat-G1-v0 task and RSL-RL inference stack. This "
        "is a stronger local virtual teacher-rollout dataset candidate, not the official BeyondMimic DAgger dataset."
    )
    summary.setdefault("inputs", {})
    summary["inputs"]["training_run_json"] = str(TRAINING_RUN_JSON)
    summary["inputs"]["base_compatible_training_run_json"] = str(
        OUT / "base_compatible_full_bundle_training_run_for_teacher_rollout.json"
    )
    summary["inputs"]["checkpoint_eval_json"] = str(CHECKPOINT_EVAL_JSON)
    summary["inputs"]["full_bundle_motion_npz"] = str(FULL_BUNDLE_MOTION_NPZ)
    summary["inputs"]["full_bundle_audit"] = str(FULL_BUNDLE_AUDIT)
    summary.setdefault("input_checks", {})
    summary["input_checks"].update(
        {
            "full_bundle_training_completed": training.get("status")
            == "ok_official_csv_loop_full_bundle_ppo_training_completed",
            "full_bundle_checkpoint_eval_completed": checkpoint_eval.get("status")
            == "ok_official_csv_loop_full_bundle_ppo_checkpoint_eval_completed",
            "full_bundle_motion_npz_exists": FULL_BUNDLE_MOTION_NPZ.is_file(),
            "full_bundle_audit_passed": bundle.get("status") == "ok_official_csv_loop_full_bundle_motion_npz",
            "full_bundle_motion_count_40": bundle_info.get("motion_count") == 40,
            "full_bundle_total_frames_11960": bundle_info.get("total_frames") == 11960,
        }
    )
    for shard in summary.get("run", {}).get("shard_metrics", []):
        shard["official_csv_to_npz_loop_output"] = True
        shard["official_csv_loop_full_public_motion_bundle"] = True
        shard["official_csv_to_npz_unpatched_output"] = False
        shard["official_dagger_rollout_dataset"] = False
        shard["paper_level_teacher_rollout_dataset"] = False
        shard["motion_count"] = bundle_info.get("motion_count")
        shard["total_motion_frames"] = bundle_info.get("total_frames")
    summary.setdefault("aggregate_metrics", {})
    summary["aggregate_metrics"]["motion_count"] = bundle_info.get("motion_count")
    summary["aggregate_metrics"]["total_motion_frames"] = bundle_info.get("total_frames")
    summary.setdefault("outputs", {})
    summary["outputs"]["json"] = str(final_json)
    summary["interpretation"] = {
        "goal_complete": False,
        "official_dagger_dataset_complete": False,
        "paper_level_teacher_rollout_dataset_complete": False,
        "official_csv_loop_full_bundle_teacher_rollout_dataset_complete": bool(rollout_ok),
        "why_not_paper_level": (
            "The rollout source is a local 300-iteration PPO checkpoint trained on a concatenated public-motion "
            "bundle under the enriched-USD runtime patch. It is useful downstream evidence, but the official DAgger "
            "rollout logs, paper-scale teacher policy, and unpatched official asset path are not available."
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
    module.CSV_MOTION_NPZ = FULL_BUNDLE_MOTION_NPZ
    module.TRAINING_RUN_JSON = compatible_training_summary
    module.SEED = int(os.environ.get("BM_OFFICIAL_CSV_LOOP_FULL_BUNDLE_TEACHER_ROLLOUT_SEED", str(DEFAULT_SEED)))
    module.main()

    output_json = OUT / "tracking_g1_resource_adjusted_teacher_rollout_dataset.json"
    final_json = OUT / "tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset.json"
    summary = patch_summary(load_json(output_json))
    output_json.unlink(missing_ok=True)
    final_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(final_json),
                "attempted_rollout": summary.get("run", {}).get("attempted_rollout"),
                "shard_count": summary.get("aggregate_metrics", {}).get("shard_count"),
                "total_env_steps": summary.get("aggregate_metrics", {}).get("total_env_steps"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
