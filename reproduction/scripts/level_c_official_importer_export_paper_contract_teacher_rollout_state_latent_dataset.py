#!/usr/bin/env python3
"""Build state-latent windows from paper-contract teacher rollout + VAE."""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py"
OUT = ROOT / "res/level_c/official_importer_export_paper_contract_teacher_rollout_state_latent_dataset"
RUN_ROOT = ROOT / "res/runs/level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset"
LOG_DIR = ROOT / "logs/level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset"
FAILED_DIR = ROOT / "res/failed_runs/level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset"
TEACHER_ROLLOUT_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_paper_contract_best_teacher_rollout_dataset/"
    "tracking_g1_official_importer_export_paper_contract_best_teacher_rollout_dataset.json"
)
VAE_TRAINING_JSON = (
    ROOT
    / "res/level_c/official_importer_export_paper_contract_teacher_rollout_vae_training/"
    "level_c_official_importer_export_paper_contract_teacher_rollout_vae_training.json"
)
DEFAULT_SEED = 20260806
DEFAULT_GPUS = [4, 7]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_paper_contract_state_latent_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load state-latent base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_gpus() -> list[int]:
    raw = os.environ.get("BM_PAPER_CONTRACT_LEVEL_C_GPUS", "")
    return [int(part.strip()) for part in raw.split(",") if part.strip()] if raw.strip() else list(DEFAULT_GPUS)


def patch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    teacher = load_json(TEACHER_ROLLOUT_JSON)
    vae = load_json(VAE_TRAINING_JSON)
    worker = summary.get("worker_summary", {})
    base_status = summary.get("status", "failed")
    base_status_ok = base_status == "ok"
    final_json = OUT / "level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.json"
    final_tsv = OUT / "level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.tsv"
    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["experiment_type"] = "official_importer_export_paper_contract_teacher_rollout_state_latent_dataset"
    summary["scope"] = (
        "Builds local state/action-latent windows from the selected paper-contract teacher rollout and local VAE."
    )
    source_teacher = worker.get("source_teacher_rollout", {})
    if source_teacher:
        source_teacher["json"] = str(TEACHER_ROLLOUT_JSON)
        source_teacher["status"] = teacher.get("status")
        source_teacher["paper_contract_best_teacher_rollout_dataset"] = True
        source_teacher["uses_official_importer_export_usd"] = True
        source_teacher["official_dagger_rollout_dataset"] = False
        source_teacher["paper_level_teacher_rollout_dataset"] = False
    source_vae = worker.get("source_vae", {})
    if source_vae:
        source_vae["json"] = str(VAE_TRAINING_JSON)
        source_vae["status"] = vae.get("status")
        source_vae["paper_contract_teacher_rollout_vae"] = True
        source_vae["official_beyondmimic_vae_checkpoint"] = False
        source_vae["paper_level_vae_checkpoint"] = False
    dataset = worker.get("dataset", {})
    if dataset:
        dataset["latent_source"] = "posterior mean/logvar from local paper-contract conditional action VAE"
    summary.setdefault("checks", {})
    paper_checks = {
        "paper_contract_best_teacher_rollout_source": teacher.get("status")
        == "ok_official_importer_export_paper_contract_best_teacher_rollout_dataset_completed",
        "paper_contract_vae_source": vae.get("status")
        == "ok_official_importer_export_paper_contract_teacher_rollout_vae_training",
        "state_source_is_raw_hybrid_or_projected": dataset.get("state_source")
        in {
            "paper_99d_hybrid_state_from_raw_rollout_world_state",
            "paper_163d_projected_hybrid_state_from_raw_rollout_world_state",
        },
        "state_dim_matches_paper_contract": dataset.get("state_dim") in {99, 163},
        "window_filter_rejects_discontinuities": worker.get("checks", {}).get("window_index_respects_rejection_filter")
        is True,
        "uses_official_importer_export_usd": True,
        "does_not_claim_official_dagger": True,
        "does_not_claim_paper_level_state_latent_dataset": True,
        "does_not_claim_closed_loop_guidance": True,
    }
    summary["checks"].update(paper_checks)
    blocking_reasons: list[str] = []
    if not base_status_ok:
        blocking_reasons.append(f"base_state_latent_builder_status={base_status}")
    for key in [
        "paper_contract_best_teacher_rollout_source",
        "paper_contract_vae_source",
        "state_source_is_raw_hybrid_or_projected",
        "state_dim_matches_paper_contract",
        "window_filter_rejects_discontinuities",
    ]:
        if not paper_checks.get(key):
            blocking_reasons.append(key)
    if base_status_ok and not blocking_reasons:
        summary["status"] = "ok_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset"
        paper_level_status = "paper_contract_local_state_latent_dataset_only"
        why_not_complete = (
            "The dataset is derived from a local teacher candidate and local VAE. It is not the unreleased official "
            "BeyondMimic DAgger/state-latent dataset."
        )
    elif base_status_ok:
        summary["status"] = (
            "blocked_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset_requires_hybrid_state"
        )
        paper_level_status = "blocked_local_state_latent_dataset_not_paper_contract"
        why_not_complete = (
            "The base builder returned ok, but the wrapper-level paper-contract checks failed. A paper-contract "
            "dataset must use raw rollout hybrid/projected state tokens, paper-consistent dimensions, and contiguous "
            "accepted windows. The current source is not allowed for downstream VAE/diffusion/guidance training."
        )
    else:
        summary["status"] = base_status
        paper_level_status = "failed_or_incomplete_state_latent_dataset_build"
        why_not_complete = "The base state-latent builder did not complete successfully."
    summary["blocking_reasons"] = blocking_reasons
    summary["interpretation"] = {
        "goal_complete": False,
        "paper_level_status": paper_level_status,
        "why_not_complete": why_not_complete,
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
    module.CANDIDATE_GPUS = parse_gpus()
    module.SEED = int(os.environ.get("BM_PAPER_CONTRACT_STATE_LATENT_SEED", str(DEFAULT_SEED)))
    module.SEQUENCE_LENGTH = int(os.environ.get("BM_PAPER_CONTRACT_STATE_LATENT_SEQ_LEN", str(module.SEQUENCE_LENGTH)))
    module.BATCH_SIZE = int(os.environ.get("BM_PAPER_CONTRACT_STATE_LATENT_BATCH_SIZE", str(module.BATCH_SIZE)))
    old_env = {
        key: os.environ.get(key)
        for key in [
            "BM_STATE_LATENT_STATE_MODE",
            "BM_STATE_LATENT_REQUIRE_RAW_STATE",
            "BM_STATE_LATENT_REQUIRE_PAPER_CONTRACT_VAE",
            "BM_STATE_LATENT_REJECT_DONES",
            "BM_STATE_LATENT_QUAT_FORMAT",
        ]
    }
    os.environ.update(
        {
            "BM_STATE_LATENT_STATE_MODE": "paper_hybrid",
            "BM_STATE_LATENT_REQUIRE_RAW_STATE": "1",
            "BM_STATE_LATENT_REQUIRE_PAPER_CONTRACT_VAE": "1",
            "BM_STATE_LATENT_REJECT_DONES": "1",
            "BM_STATE_LATENT_QUAT_FORMAT": "wxyz",
        }
    )
    try:
        module.main()
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    base_json = OUT / "level_c_resource_adjusted_teacher_rollout_state_latent_dataset.json"
    base_tsv = OUT / "level_c_resource_adjusted_teacher_rollout_state_latent_dataset.tsv"
    final_json = OUT / "level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.json"
    final_tsv = OUT / "level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.tsv"
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
