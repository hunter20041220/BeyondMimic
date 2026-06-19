#!/usr/bin/env python3
"""Collect teacher rollout shards from the official csv-loop PPO checkpoint.

This wraps the existing resource-adjusted rollout harness while switching the
checkpoint, motion artifact, output paths, and audit wording to the stronger
official-csv-loop motion chain. It is still not the official BeyondMimic DAgger
dataset because the run uses the enriched-USD runtime patch and a local
300-iteration PPO checkpoint.
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
OUT = ROOT / "res/tracking/g1_official_csv_loop_teacher_rollout_dataset"
LOG_DIR = ROOT / "logs/tracking_g1_official_csv_loop_teacher_rollout_dataset"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_csv_loop_teacher_rollout_dataset"
OFFICIAL_LOOP_MOTION_NPZ = (
    ROOT
    / "res/tracking/official_csv_to_npz_loop_with_enriched_usd/"
    "walk1_subject1_frames_1_180_official_loop_enriched_usd_motion.npz"
)
TRAINING_RUN_JSON = (
    ROOT
    / "res/tracking/g1_official_csv_loop_ppo_training_run/"
    "tracking_g1_official_csv_loop_ppo_training_run.json"
)
CHECKPOINT_EVAL_JSON = (
    ROOT
    / "res/tracking/g1_official_csv_loop_ppo_checkpoint_eval/"
    "tracking_g1_official_csv_loop_ppo_checkpoint_eval.json"
)
OFFICIAL_CSV_LOOP_AUDIT = (
    ROOT
    / "res/tracking/official_csv_to_npz_loop_with_enriched_usd/"
    "tracking_official_csv_to_npz_loop_with_enriched_usd_audit.json"
)
OFFICIAL_REPLAY_LOOP_AUDIT = (
    ROOT
    / "res/tracking/official_replay_npz_loop_with_enriched_usd/"
    "tracking_official_replay_npz_loop_with_enriched_usd_audit.json"
)
DEFAULT_SEED = 20260631


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


def patch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    official_csv_loop = load_json(OFFICIAL_CSV_LOOP_AUDIT)
    official_replay_loop = load_json(OFFICIAL_REPLAY_LOOP_AUDIT)
    checkpoint_eval = load_json(CHECKPOINT_EVAL_JSON)
    rollout_ok = summary.get("status") == "ok_resource_adjusted_teacher_rollout_dataset_completed"
    final_json = OUT / "tracking_g1_official_csv_loop_teacher_rollout_dataset.json"

    summary["status"] = (
        "ok_official_csv_loop_teacher_rollout_dataset_completed"
        if rollout_ok
        else summary.get("status", "failed_official_csv_loop_teacher_rollout_dataset")
    )
    summary["experiment_type"] = "tracking_official_csv_loop_teacher_rollout_dataset"
    summary["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    summary["scope"] = (
        "Collects two-GPU teacher rollout shards from the local PPO checkpoint trained on the official csv_to_npz.py "
        "loop motion artifact, using the official Tracking-Flat-G1-v0 task and RSL-RL inference stack. This is a "
        "local virtual teacher-rollout dataset candidate, not the official BeyondMimic DAgger dataset."
    )
    summary.setdefault("inputs", {})
    summary["inputs"]["training_run_json"] = str(TRAINING_RUN_JSON)
    summary["inputs"]["checkpoint_eval_json"] = str(CHECKPOINT_EVAL_JSON)
    summary["inputs"]["motion_npz"] = str(OFFICIAL_LOOP_MOTION_NPZ)
    summary["inputs"]["official_csv_to_npz_loop_audit"] = str(OFFICIAL_CSV_LOOP_AUDIT)
    summary["inputs"]["official_replay_loop_audit"] = str(OFFICIAL_REPLAY_LOOP_AUDIT)
    summary.setdefault("input_checks", {})
    summary["input_checks"]["official_csv_loop_training_completed"] = (
        load_json(TRAINING_RUN_JSON).get("status") == "ok_official_csv_loop_ppo_training_completed"
    )
    summary["input_checks"]["official_csv_loop_checkpoint_eval_completed"] = (
        checkpoint_eval.get("status") == "ok_official_csv_loop_ppo_checkpoint_eval_completed"
    )
    summary["input_checks"]["official_csv_loop_motion_npz_exists"] = OFFICIAL_LOOP_MOTION_NPZ.is_file()
    summary["input_checks"]["official_csv_loop_audit_passed"] = (
        official_csv_loop.get("status") == "ok_official_csv_to_npz_loop_with_enriched_usd_patch"
    )
    summary["input_checks"]["official_replay_loop_audit_passed"] = (
        official_replay_loop.get("status") == "ok_official_replay_loop_with_enriched_usd_patch"
    )
    for shard in summary.get("run", {}).get("shard_metrics", []):
        shard["official_csv_to_npz_loop_output"] = True
        shard["official_csv_to_npz_unpatched_output"] = False
        shard["official_dagger_rollout_dataset"] = False
        shard["paper_level_teacher_rollout_dataset"] = False
    summary.setdefault("outputs", {})
    summary["outputs"]["json"] = str(final_json)
    summary["interpretation"] = {
        "goal_complete": False,
        "official_dagger_dataset_complete": False,
        "paper_level_teacher_rollout_dataset_complete": False,
        "official_csv_loop_teacher_rollout_dataset_complete": bool(rollout_ok),
        "why_not_paper_level": (
            "The rollout source is a local 300-iteration PPO checkpoint trained on official-loop motion under the "
            "enriched-USD runtime patch. It is useful downstream evidence but not the paper's official DAgger rollout "
            "logs or a paper-scale teacher policy."
        ),
    }
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    module = load_base_module()
    module.OUT = OUT
    module.LOG_DIR = LOG_DIR
    module.RUN_ROOT = RUN_ROOT
    module.CSV_MOTION_NPZ = OFFICIAL_LOOP_MOTION_NPZ
    module.TRAINING_RUN_JSON = TRAINING_RUN_JSON
    module.SEED = int(os.environ.get("BM_OFFICIAL_CSV_LOOP_TEACHER_ROLLOUT_SEED", str(DEFAULT_SEED)))
    module.main()

    output_json = OUT / "tracking_g1_resource_adjusted_teacher_rollout_dataset.json"
    final_json = OUT / "tracking_g1_official_csv_loop_teacher_rollout_dataset.json"
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
