#!/usr/bin/env python3
"""Train the local action VAE on full-bundle official-csv-loop teacher rollout shards."""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py"
OUT = ROOT / "res/level_c/official_csv_loop_full_bundle_teacher_rollout_vae_training"
RUN_ROOT = ROOT / "res/runs/level_c_official_csv_loop_full_bundle_teacher_rollout_vae_training"
LOG_DIR = ROOT / "logs/level_c_official_csv_loop_full_bundle_teacher_rollout_vae_training"
FAILED_DIR = ROOT / "res/failed_runs/level_c_official_csv_loop_full_bundle_teacher_rollout_vae_training"
TEACHER_ROLLOUT_JSON = (
    ROOT
    / "res/tracking/g1_official_csv_loop_full_bundle_teacher_rollout_dataset/"
    "tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset.json"
)
DEFAULT_SEED = 20260673


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_resource_adjusted_vae_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load VAE base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    teacher = load_json(TEACHER_ROLLOUT_JSON)
    worker = summary.get("worker_summary", {})
    source = worker.get("source_teacher_rollout", {})
    status_ok = summary.get("status") == "ok"
    final_json = OUT / "level_c_official_csv_loop_full_bundle_teacher_rollout_vae_training.json"
    final_tsv = OUT / "level_c_official_csv_loop_full_bundle_teacher_rollout_vae_training.tsv"

    summary["status"] = (
        "ok_official_csv_loop_full_bundle_teacher_rollout_vae_training"
        if status_ok
        else summary.get("status", "failed")
    )
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["experiment_type"] = "official_csv_loop_full_bundle_teacher_rollout_conditional_action_vae_training"
    summary["scope"] = (
        "Trains the local conditional action VAE on teacher rollout shards from the 40-motion public official-loop "
        "bundle. This is a stronger local virtual downstream artifact than the single-motion official-loop VAE, but "
        "it is not an official BeyondMimic DAgger/VAE checkpoint or paper-level closed-loop evaluation."
    )
    if source:
        source["json"] = str(TEACHER_ROLLOUT_JSON)
        source["status"] = teacher.get("status")
        source["official_csv_loop_full_bundle_teacher_rollout_dataset"] = True
        source["motion_count"] = teacher.get("aggregate_metrics", {}).get("motion_count")
        source["total_motion_frames"] = teacher.get("aggregate_metrics", {}).get("total_motion_frames")
        source["official_dagger_rollout_dataset"] = False
        source["paper_level_teacher_rollout_dataset"] = False
    summary.setdefault("checks", {})
    summary["checks"]["official_csv_loop_full_bundle_teacher_rollout_source"] = (
        teacher.get("status") == "ok_official_csv_loop_full_bundle_teacher_rollout_dataset_completed"
    )
    summary["checks"]["full_bundle_motion_count_40"] = teacher.get("aggregate_metrics", {}).get("motion_count") == 40
    summary["checks"]["full_bundle_total_motion_frames_11960"] = (
        teacher.get("aggregate_metrics", {}).get("total_motion_frames") == 11960
    )
    summary["checks"]["does_not_claim_official_beyondmimic_vae"] = True
    summary["checks"]["does_not_claim_closed_loop_eval"] = True
    summary["interpretation"] = {
        "goal_complete": False,
        "paper_level_status": "full_bundle_teacher_rollout_training_only",
        "why_not_complete": (
            "The VAE is trained on a local full-bundle teacher rollout dataset from the enriched-USD official-loop "
            "motion chain. It broadens local data coverage but still is not trained from official DAgger logs and "
            "has not been validated as paper-level closed-loop VAE control."
        ),
    }
    summary.setdefault("outputs", {})
    summary["outputs"]["json"] = str(final_json)
    summary["outputs"]["tsv"] = str(final_tsv)
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)

    module = load_base_module()
    module.OUT = OUT
    module.RUN_ROOT = RUN_ROOT
    module.LOG_DIR = LOG_DIR
    module.FAILED_DIR = FAILED_DIR
    module.TEACHER_ROLLOUT_JSON = TEACHER_ROLLOUT_JSON
    module.SEED = int(os.environ.get("BM_OFFICIAL_CSV_LOOP_FULL_BUNDLE_VAE_SEED", str(DEFAULT_SEED)))
    module.EPOCHS = int(os.environ.get("BM_OFFICIAL_CSV_LOOP_FULL_BUNDLE_VAE_EPOCHS", str(module.EPOCHS)))
    module.main()

    base_json = OUT / "level_c_resource_adjusted_teacher_rollout_vae_training.json"
    base_tsv = OUT / "level_c_resource_adjusted_teacher_rollout_vae_training.tsv"
    final_json = OUT / "level_c_official_csv_loop_full_bundle_teacher_rollout_vae_training.json"
    final_tsv = OUT / "level_c_official_csv_loop_full_bundle_teacher_rollout_vae_training.tsv"
    summary = patch_summary(load_json(base_json))
    if base_tsv.is_file():
        base_tsv.replace(final_tsv)
    base_json.unlink(missing_ok=True)
    final_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(final_json),
                "test_action_mse": summary.get("worker_summary", {})
                .get("evaluation", {})
                .get("test", {})
                .get("action_mse"),
                "sample_count": summary.get("worker_summary", {}).get("dataset", {}).get("sample_count"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
