#!/usr/bin/env python3
"""Train a denoiser on the official-csv-loop state-latent dataset.

This wrapper reuses the full resource-adjusted state-latent diffusion trainer
and switches the input dataset to the local official-csv-loop state-latent
windows. It is a local virtual downstream experiment, not an official
BeyondMimic diffusion checkpoint or closed-loop guidance result.
"""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py"
OUT = ROOT / "res/level_c/official_csv_loop_state_latent_diffusion_training"
RUN_ROOT = ROOT / "res/runs/level_c_official_csv_loop_state_latent_diffusion_training"
LOG_DIR = ROOT / "logs/level_c_official_csv_loop_state_latent_diffusion_training"
FAILED_DIR = ROOT / "res/failed_runs/level_c_official_csv_loop_state_latent_diffusion_training"
STATE_LATENT_JSON = (
    ROOT
    / "res/level_c/official_csv_loop_teacher_rollout_state_latent_dataset/"
    "level_c_official_csv_loop_teacher_rollout_state_latent_dataset.json"
)
DEFAULT_SEED = 20260634


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_resource_adjusted_diffusion_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load diffusion base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    worker = summary.get("worker_summary", {})
    source_dataset = worker.get("source_dataset", {})
    dataset_summary = load_json(STATE_LATENT_JSON)
    status_ok = summary.get("status") == "ok"
    final_json = OUT / "level_c_official_csv_loop_state_latent_diffusion_training.json"
    final_tsv = OUT / "level_c_official_csv_loop_state_latent_diffusion_training.tsv"

    summary["status"] = "ok_official_csv_loop_state_latent_diffusion_training" if status_ok else summary.get(
        "status", "failed"
    )
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["experiment_type"] = "official_csv_loop_state_latent_diffusion_training"
    summary["scope"] = (
        "Trains a local state-latent denoiser over all windows derived from the official-csv-loop teacher rollout "
        "chain. It advances the virtual BeyondMimic pipeline, but it is not an official diffusion checkpoint and "
        "does not run closed-loop guided control."
    )
    if source_dataset:
        source_dataset["json"] = str(STATE_LATENT_JSON)
        source_dataset["status"] = dataset_summary.get("status")
        source_dataset["official_csv_loop_state_latent_dataset"] = True
        source_dataset["official_dagger_rollout_dataset"] = False
        source_dataset["paper_level_state_latent_dataset"] = False
    summary.setdefault("checks", {})
    summary["checks"]["official_csv_loop_state_latent_dataset_source"] = (
        dataset_summary.get("status") == "ok_official_csv_loop_teacher_rollout_state_latent_dataset"
    )
    summary["checks"]["does_not_claim_official_diffusion_checkpoint"] = True
    summary["checks"]["does_not_claim_closed_loop_guidance"] = True
    summary["checks"]["does_not_claim_fig5_fig6"] = True
    summary["interpretation"] = {
        "goal_complete": False,
        "paper_level_status": "official_csv_loop_state_latent_diffusion_training_only",
        "why_not_complete": (
            "The denoiser is trained on local virtual state-latent windows. It is not the unreleased official "
            "BeyondMimic diffusion checkpoint and has not been evaluated through IsaacLab closed-loop guidance, "
            "TensorRT deployment, or paper Fig. 5/Fig. 6 task metrics."
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
    module.STATE_LATENT_JSON = STATE_LATENT_JSON
    module.SEED = int(os.environ.get("BM_OFFICIAL_CSV_LOOP_DIFFUSION_SEED", str(DEFAULT_SEED)))
    module.EPOCHS = int(os.environ.get("BM_OFFICIAL_CSV_LOOP_DIFFUSION_EPOCHS", str(module.EPOCHS)))
    module.BATCH_WINDOWS = int(os.environ.get("BM_OFFICIAL_CSV_LOOP_DIFFUSION_BATCH_WINDOWS", str(module.BATCH_WINDOWS)))
    module.HIDDEN_DIM = int(os.environ.get("BM_OFFICIAL_CSV_LOOP_DIFFUSION_HIDDEN_DIM", str(module.HIDDEN_DIM)))
    module.LEARNING_RATE = float(os.environ.get("BM_OFFICIAL_CSV_LOOP_DIFFUSION_LR", str(module.LEARNING_RATE)))
    module.DENOISING_STEPS = int(os.environ.get("BM_OFFICIAL_CSV_LOOP_DIFFUSION_STEPS", str(module.DENOISING_STEPS)))
    module.main()

    base_json = OUT / "level_c_resource_adjusted_state_latent_diffusion_training.json"
    base_tsv = OUT / "level_c_resource_adjusted_state_latent_diffusion_training.tsv"
    final_json = OUT / "level_c_official_csv_loop_state_latent_diffusion_training.json"
    final_tsv = OUT / "level_c_official_csv_loop_state_latent_diffusion_training.tsv"
    summary = patch_summary(load_json(base_json))
    if base_tsv.is_file():
        base_tsv.replace(final_tsv)
    base_json.unlink(missing_ok=True)
    final_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    evaluation = summary.get("worker_summary", {}).get("evaluation", {})
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(final_json),
                "test_pred_token_mse": evaluation.get("test", {}).get("pred_token_mse"),
                "test_denoising_improvement_ratio": evaluation.get("test", {}).get(
                    "denoising_improvement_ratio"
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
