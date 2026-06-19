#!/usr/bin/env python3
"""Build state-latent windows from the official-csv-loop teacher rollout VAE.

This wrapper reuses the full resource-adjusted state-latent dataset builder and
switches both inputs to the stronger local official-csv-loop teacher rollout
chain. It remains a local virtual reproduction artifact: the teacher rollout is
not official DAgger data and the VAE is not an official BeyondMimic checkpoint.
"""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py"
OUT = ROOT / "res/level_c/official_csv_loop_teacher_rollout_state_latent_dataset"
RUN_ROOT = ROOT / "res/runs/level_c_official_csv_loop_teacher_rollout_state_latent_dataset"
LOG_DIR = ROOT / "logs/level_c_official_csv_loop_teacher_rollout_state_latent_dataset"
FAILED_DIR = ROOT / "res/failed_runs/level_c_official_csv_loop_teacher_rollout_state_latent_dataset"
TEACHER_ROLLOUT_JSON = (
    ROOT
    / "res/tracking/g1_official_csv_loop_teacher_rollout_dataset/"
    "tracking_g1_official_csv_loop_teacher_rollout_dataset.json"
)
VAE_TRAINING_JSON = (
    ROOT
    / "res/level_c/official_csv_loop_teacher_rollout_vae_training/"
    "level_c_official_csv_loop_teacher_rollout_vae_training.json"
)
DEFAULT_SEED = 20260633


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_resource_adjusted_state_latent_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load state-latent base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    worker = summary.get("worker_summary", {})
    status_ok = summary.get("status") == "ok"
    final_json = OUT / "level_c_official_csv_loop_teacher_rollout_state_latent_dataset.json"
    final_tsv = OUT / "level_c_official_csv_loop_teacher_rollout_state_latent_dataset.tsv"
    teacher = load_json(TEACHER_ROLLOUT_JSON)
    vae = load_json(VAE_TRAINING_JSON)

    summary["status"] = "ok_official_csv_loop_teacher_rollout_state_latent_dataset" if status_ok else summary.get(
        "status", "failed"
    )
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["experiment_type"] = "official_csv_loop_teacher_rollout_state_latent_dataset"
    summary["scope"] = (
        "Builds full state/action-latent windows from the locally collected official-csv-loop teacher rollout dataset "
        "and the local official-csv-loop conditional action VAE. This advances the virtual BeyondMimic pipeline, but "
        "it is not official DAgger data and not paper-level closed-loop guidance evidence."
    )

    source_teacher = worker.get("source_teacher_rollout", {})
    if source_teacher:
        source_teacher["json"] = str(TEACHER_ROLLOUT_JSON)
        source_teacher["status"] = teacher.get("status")
        source_teacher["official_csv_loop_teacher_rollout_dataset"] = True
        source_teacher["official_dagger_rollout_dataset"] = False
        source_teacher["paper_level_teacher_rollout_dataset"] = False
    source_vae = worker.get("source_vae", {})
    if source_vae:
        source_vae["json"] = str(VAE_TRAINING_JSON)
        source_vae["status"] = vae.get("status")
        source_vae["official_csv_loop_teacher_rollout_vae"] = True
        source_vae["official_beyondmimic_vae_checkpoint"] = False
        source_vae["paper_level_vae_checkpoint"] = False

    dataset = worker.get("dataset", {})
    if dataset:
        dataset["state_source"] = "policy_obs in local official-csv-loop teacher rollout shards"
        dataset["latent_source"] = "posterior mean/logvar from local official-csv-loop conditional action VAE"

    summary.setdefault("checks", {})
    summary["checks"]["official_csv_loop_teacher_rollout_source"] = (
        teacher.get("status") == "ok_official_csv_loop_teacher_rollout_dataset_completed"
    )
    summary["checks"]["official_csv_loop_vae_source"] = (
        vae.get("status") == "ok_official_csv_loop_teacher_rollout_vae_training"
    )
    summary["checks"]["does_not_claim_official_dagger"] = True
    summary["checks"]["does_not_claim_paper_level_state_latent_dataset"] = True
    summary["checks"]["does_not_claim_closed_loop_guidance"] = True
    summary["interpretation"] = {
        "goal_complete": False,
        "paper_level_status": "official_csv_loop_teacher_rollout_state_latent_dataset_only",
        "why_not_complete": (
            "The windows are derived from local virtual teacher rollout shards and a local VAE checkpoint. They are "
            "suitable for downstream local diffusion experiments, but they are not the unreleased official DAgger "
            "trajectory dataset and do not evaluate guided control in IsaacLab closed loop."
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
    module.VAE_TRAINING_JSON = VAE_TRAINING_JSON
    module.SEED = int(os.environ.get("BM_OFFICIAL_CSV_LOOP_STATE_LATENT_SEED", str(DEFAULT_SEED)))
    module.SEQUENCE_LENGTH = int(os.environ.get("BM_OFFICIAL_CSV_LOOP_STATE_LATENT_SEQ_LEN", str(module.SEQUENCE_LENGTH)))
    module.BATCH_SIZE = int(os.environ.get("BM_OFFICIAL_CSV_LOOP_STATE_LATENT_BATCH_SIZE", str(module.BATCH_SIZE)))
    module.main()

    base_json = OUT / "level_c_resource_adjusted_teacher_rollout_state_latent_dataset.json"
    base_tsv = OUT / "level_c_resource_adjusted_teacher_rollout_state_latent_dataset.tsv"
    final_json = OUT / "level_c_official_csv_loop_teacher_rollout_state_latent_dataset.json"
    final_tsv = OUT / "level_c_official_csv_loop_teacher_rollout_state_latent_dataset.tsv"
    summary = patch_summary(load_json(base_json))
    if base_tsv.is_file():
        base_tsv.replace(final_tsv)
    base_json.unlink(missing_ok=True)
    final_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    dataset = summary.get("worker_summary", {}).get("dataset", {})
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(final_json),
                "sample_count": dataset.get("sample_count"),
                "window_count": dataset.get("window_count"),
                "weighted_posterior_reconstruction_mse": dataset.get("weighted_posterior_reconstruction_mse"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
