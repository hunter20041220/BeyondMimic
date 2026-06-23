#!/usr/bin/env python3
"""Offline task-cost guidance over Stage-1 multi-source denoiser outputs."""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/level_c_resource_adjusted_state_latent_guidance_eval.py"
OUT = ROOT / "res/level_c/stage1_multisource_state_latent_guidance_eval"
RUN_ROOT = ROOT / "res/runs/level_c_stage1_multisource_state_latent_guidance_eval"
LOG_DIR = ROOT / "logs/level_c_stage1_multisource_state_latent_guidance_eval"
FAILED_DIR = ROOT / "res/failed_runs/level_c_stage1_multisource_state_latent_guidance_eval"
DIFFUSION_JSON = (
    ROOT
    / "res/level_c/stage1_multisource_state_latent_diffusion_training/"
    "level_c_stage1_multisource_state_latent_diffusion_training.json"
)
STATE_LATENT_JSON = (
    ROOT
    / "res/level_c/stage1_multisource_teacher_rollout_state_latent_dataset/"
    "level_c_stage1_multisource_teacher_rollout_state_latent_dataset.json"
)
DEFAULT_SEED = 20260858
TARGET_GPUS = [5, 6]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_stage1_multisource_guidance_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load guidance base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    worker = summary.get("worker_summary", {})
    diffusion = load_json(DIFFUSION_JSON)
    dataset = load_json(STATE_LATENT_JSON)
    final_json = OUT / "level_c_stage1_multisource_state_latent_guidance_eval.json"
    final_tsv = OUT / "level_c_stage1_multisource_state_latent_guidance_eval.tsv"
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["experiment_type"] = "stage1_multisource_state_latent_guidance_eval"
    summary["scope"] = "Offline task-cost guidance over the local multi-source denoiser. This is not closed-loop control."
    source_diffusion = worker.get("source_diffusion", {})
    if source_diffusion:
        source_diffusion.update(
            {
                "json": str(DIFFUSION_JSON),
                "status": diffusion.get("status"),
                "stage1_multisource_state_latent_diffusion": True,
                "official_beyondmimic_diffusion_checkpoint": False,
                "paper_level_diffusion_checkpoint": False,
            }
        )
    source_dataset = worker.get("source_dataset", {})
    if source_dataset:
        source_dataset.update(
            {
                "json": str(STATE_LATENT_JSON),
                "status": dataset.get("status"),
                "stage1_multisource_state_latent_dataset": True,
                "official_dagger_rollout_dataset": False,
                "paper_level_state_latent_dataset": False,
            }
        )
    summary.setdefault("checks", {})
    summary["checks"].update(
        {
            "stage1_multisource_diffusion_source": diffusion.get("status")
            == "ok_stage1_multisource_state_latent_diffusion_training",
            "stage1_multisource_state_latent_dataset_source": dataset.get("status")
            == "ok_stage1_multisource_teacher_rollout_state_latent_dataset",
            "does_not_claim_closed_loop_guidance": True,
            "does_not_claim_fig5_fig6": True,
            "does_not_claim_real_robot": True,
        }
    )
    status_ok = summary.get("returncode") == 0 and worker.get("status") == "ok" and all(summary["checks"].values())
    summary["status"] = "ok_stage1_multisource_state_latent_guidance_eval" if status_ok else "failed"
    summary["interpretation"] = {
        "goal_complete": False,
        "paper_level_status": "local_multisource_offline_guidance_only",
        "why_not_complete": "The guidance calculation is offline over local denoiser outputs and proxy task costs.",
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
    module.DIFFUSION_JSON = DIFFUSION_JSON
    module.STATE_LATENT_JSON = STATE_LATENT_JSON
    module.CANDIDATE_GPUS = TARGET_GPUS
    module.SEED = int(os.environ.get("BM_STAGE1_MULTISOURCE_GUIDANCE_SEED", str(DEFAULT_SEED)))
    module.MAX_WINDOWS_PER_SPLIT = int(os.environ.get("BM_STAGE1_MULTISOURCE_GUIDANCE_MAX_WINDOWS_PER_SPLIT", str(module.MAX_WINDOWS_PER_SPLIT)))
    module.BATCH_WINDOWS = int(os.environ.get("BM_STAGE1_MULTISOURCE_GUIDANCE_BATCH_WINDOWS", str(module.BATCH_WINDOWS)))
    module.SCALES = os.environ.get("BM_STAGE1_MULTISOURCE_GUIDANCE_SCALES", module.SCALES)
    module.main()
    base_json = OUT / "level_c_resource_adjusted_state_latent_guidance_eval.json"
    base_tsv = OUT / "level_c_resource_adjusted_state_latent_guidance_eval.tsv"
    final_json = OUT / "level_c_stage1_multisource_state_latent_guidance_eval.json"
    final_tsv = OUT / "level_c_stage1_multisource_state_latent_guidance_eval.tsv"
    summary = patch_summary(load_json(base_json))
    if base_tsv.is_file():
        base_tsv.replace(final_tsv)
    base_json.unlink(missing_ok=True)
    final_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics = summary.get("worker_summary", {}).get("metrics", {})
    print(json.dumps({"status": summary["status"], "json": str(final_json), "total_selected_windows": metrics.get("total_selected_windows")}, sort_keys=True))
    if summary["status"] != "ok_stage1_multisource_state_latent_guidance_eval":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
