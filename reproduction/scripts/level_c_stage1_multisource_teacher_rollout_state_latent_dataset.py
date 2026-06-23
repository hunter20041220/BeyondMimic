#!/usr/bin/env python3
"""Build state-latent windows from Stage-1 multi-source teacher rollout + VAE."""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py"
OUT = ROOT / "res/level_c/stage1_multisource_teacher_rollout_state_latent_dataset"
RUN_ROOT = ROOT / "res/runs/level_c_stage1_multisource_teacher_rollout_state_latent_dataset"
LOG_DIR = ROOT / "logs/level_c_stage1_multisource_teacher_rollout_state_latent_dataset"
FAILED_DIR = ROOT / "res/failed_runs/level_c_stage1_multisource_teacher_rollout_state_latent_dataset"
TEACHER_ROLLOUT_JSON = (
    ROOT
    / "res/tracking/stage1_multisource_best_teacher_rollout_dataset/"
    "tracking_stage1_multisource_best_teacher_rollout_dataset.json"
)
VAE_TRAINING_JSON = (
    ROOT
    / "res/level_c/stage1_multisource_teacher_rollout_vae_training/"
    "level_c_stage1_multisource_teacher_rollout_vae_training.json"
)
DEFAULT_SEED = 20260856
TARGET_GPUS = [5, 6]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_stage1_multisource_state_latent_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load state-latent base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    teacher = load_json(TEACHER_ROLLOUT_JSON)
    vae = load_json(VAE_TRAINING_JSON)
    worker = summary.get("worker_summary", {})
    status_ok = summary.get("status") == "ok"
    final_json = OUT / "level_c_stage1_multisource_teacher_rollout_state_latent_dataset.json"
    final_tsv = OUT / "level_c_stage1_multisource_teacher_rollout_state_latent_dataset.tsv"
    summary["status"] = "ok_stage1_multisource_teacher_rollout_state_latent_dataset" if status_ok else summary.get("status", "failed")
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["experiment_type"] = "stage1_multisource_teacher_rollout_state_latent_dataset"
    summary["scope"] = "Builds local state/action-latent windows from the selected multi-source teacher rollout and local VAE."
    source_teacher = worker.get("source_teacher_rollout", {})
    if source_teacher:
        source_teacher.update(
            {
                "json": str(TEACHER_ROLLOUT_JSON),
                "status": teacher.get("status"),
                "stage1_multisource_teacher_rollout_dataset": True,
                "uses_official_importer_export_usd": True,
                "official_dagger_rollout_dataset": False,
                "paper_level_teacher_rollout_dataset": False,
            }
        )
    source_vae = worker.get("source_vae", {})
    if source_vae:
        source_vae.update(
            {
                "json": str(VAE_TRAINING_JSON),
                "status": vae.get("status"),
                "stage1_multisource_teacher_rollout_vae": True,
                "official_beyondmimic_vae_checkpoint": False,
                "paper_level_vae_checkpoint": False,
            }
        )
    dataset = worker.get("dataset", {})
    if dataset:
        dataset["state_source"] = "policy_obs in local multi-source best-teacher rollout shards"
        dataset["latent_source"] = "posterior mean/logvar from local multi-source conditional action VAE"
    summary.setdefault("checks", {})
    summary["checks"].update(
        {
            "stage1_multisource_teacher_rollout_source": teacher.get("status")
            == "ok_stage1_multisource_best_teacher_rollout_dataset_completed",
            "stage1_multisource_vae_source": vae.get("status") == "ok_stage1_multisource_teacher_rollout_vae_training",
            "uses_official_importer_export_usd": True,
            "does_not_claim_official_dagger": True,
            "does_not_claim_paper_level_state_latent_dataset": True,
        }
    )
    summary["interpretation"] = {
        "goal_complete": False,
        "paper_level_status": "local_multisource_state_latent_dataset_only",
        "why_not_complete": "The dataset is derived from a local multi-source teacher candidate and local VAE, not official DAgger data.",
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
    module.VAE_TRAINING_JSON = VAE_TRAINING_JSON
    module.CANDIDATE_GPUS = TARGET_GPUS
    module.SEED = int(os.environ.get("BM_STAGE1_MULTISOURCE_STATE_LATENT_SEED", str(DEFAULT_SEED)))
    module.SEQUENCE_LENGTH = int(os.environ.get("BM_STAGE1_MULTISOURCE_STATE_LATENT_SEQ_LEN", str(module.SEQUENCE_LENGTH)))
    module.BATCH_SIZE = int(os.environ.get("BM_STAGE1_MULTISOURCE_STATE_LATENT_BATCH_SIZE", str(module.BATCH_SIZE)))
    module.main()
    base_json = OUT / "level_c_resource_adjusted_teacher_rollout_state_latent_dataset.json"
    base_tsv = OUT / "level_c_resource_adjusted_teacher_rollout_state_latent_dataset.tsv"
    final_json = OUT / "level_c_stage1_multisource_teacher_rollout_state_latent_dataset.json"
    final_tsv = OUT / "level_c_stage1_multisource_teacher_rollout_state_latent_dataset.tsv"
    summary = patch_summary(load_json(base_json))
    if base_tsv.is_file():
        base_tsv.replace(final_tsv)
    base_json.unlink(missing_ok=True)
    final_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    dataset = summary.get("worker_summary", {}).get("dataset", {})
    print(json.dumps({"status": summary["status"], "json": str(final_json), "window_count": dataset.get("window_count")}, sort_keys=True))
    if summary["status"].startswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
