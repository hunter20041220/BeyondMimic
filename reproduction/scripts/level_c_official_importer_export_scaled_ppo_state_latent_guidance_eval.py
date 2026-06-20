#!/usr/bin/env python3
"""Full-split offline guidance for the scaled PPO official-importer-export denoiser.

This wrapper reuses the resource-adjusted state-latent guidance evaluator while
switching inputs to the local official-importer-export scaled PPO downstream
chain. It is offline task-cost guidance over local denoiser outputs, not
closed-loop IsaacLab control, not official BeyondMimic Fig. 5/Fig. 6, and not
real robot evidence.
"""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/level_c_resource_adjusted_state_latent_guidance_eval.py"
OUT = ROOT / "res/level_c/official_importer_export_scaled_ppo_state_latent_guidance_eval"
RUN_ROOT = ROOT / "res/runs/level_c_official_importer_export_scaled_ppo_state_latent_guidance_eval"
LOG_DIR = ROOT / "logs/level_c_official_importer_export_scaled_ppo_state_latent_guidance_eval"
FAILED_DIR = ROOT / "res/failed_runs/level_c_official_importer_export_scaled_ppo_state_latent_guidance_eval"
DIFFUSION_JSON = (
    ROOT
    / "res/level_c/official_importer_export_scaled_ppo_state_latent_diffusion_training/"
    "level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training.json"
)
STATE_LATENT_JSON = (
    ROOT
    / "res/level_c/official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset/"
    "level_c_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset.json"
)
DEFAULT_SEED = 20260704
DEFAULT_MAX_WINDOWS_PER_SPLIT = 0


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_scaled_ppo_guidance_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load guidance base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    worker = summary.get("worker_summary", {})
    diffusion = load_json(DIFFUSION_JSON)
    dataset = load_json(STATE_LATENT_JSON)
    final_json = OUT / "level_c_official_importer_export_scaled_ppo_state_latent_guidance_eval.json"
    final_tsv = OUT / "level_c_official_importer_export_scaled_ppo_state_latent_guidance_eval.tsv"

    summary["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary["experiment_type"] = "official_importer_export_scaled_ppo_state_latent_guidance_eval"
    summary["scope"] = (
        "Full-split offline task-cost guidance over local official-importer-export scaled PPO denoiser outputs. "
        "This updates the virtual guidance evidence to the larger iteration-999 scaled PPO teacher rollout chain, "
        "but it is not closed-loop IsaacLab guided control and not paper Fig. 5/Fig. 6 evidence."
    )
    source_diffusion = worker.get("source_diffusion", {})
    if source_diffusion:
        source_diffusion["json"] = str(DIFFUSION_JSON)
        source_diffusion["status"] = diffusion.get("status")
        source_diffusion["official_importer_export_scaled_ppo_state_latent_diffusion"] = True
        source_diffusion["official_beyondmimic_diffusion_checkpoint"] = False
        source_diffusion["paper_level_diffusion_checkpoint"] = False
    source_dataset = worker.get("source_dataset", {})
    if source_dataset:
        source_dataset["json"] = str(STATE_LATENT_JSON)
        source_dataset["status"] = dataset.get("status")
        source_dataset["official_importer_export_scaled_ppo_state_latent_dataset"] = True
        source_dataset["official_dagger_rollout_dataset"] = False
        source_dataset["paper_level_state_latent_dataset"] = False

    selected_counts = worker.get("settings", {}).get("selected_split_counts", {})
    dataset_splits = dataset.get("worker_summary", {}).get("dataset", {}).get("split_counts", {})
    summary.setdefault("checks", {})
    summary["checks"]["official_importer_export_scaled_ppo_diffusion_source"] = (
        diffusion.get("status") == "ok_official_importer_export_scaled_ppo_state_latent_diffusion_training"
    )
    summary["checks"]["official_importer_export_scaled_ppo_state_latent_dataset_source"] = (
        dataset.get("status") == "ok_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset"
    )
    summary["checks"]["evaluates_full_validation_test_splits"] = (
        selected_counts.get("validation") == dataset_splits.get("validation")
        and selected_counts.get("test") == dataset_splits.get("test")
    )
    summary["checks"]["uses_official_importer_export_usd_chain"] = True
    summary["checks"]["uses_scaled_ppo_teacher_rollout_chain"] = True
    summary["checks"]["does_not_claim_closed_loop_guidance"] = True
    summary["checks"]["does_not_claim_fig5_fig6"] = True
    summary["checks"]["does_not_claim_real_robot"] = True
    status_ok = (
        summary.get("returncode") == 0
        and worker.get("status") == "ok"
        and all(summary["checks"].values())
    )
    summary["status"] = (
        "ok_official_importer_export_scaled_ppo_state_latent_guidance_eval" if status_ok else "failed"
    )
    summary["interpretation"] = {
        "goal_complete": False,
        "paper_level_status": "official_importer_export_scaled_ppo_offline_guidance_only",
        "why_not_complete": (
            "The guidance calculation is offline over local denoiser outputs and proxy task costs. It does not roll "
            "policies out in IsaacLab, does not measure paper success/failure metrics, does not generate Fig. 5/"
            "Fig. 6 videos, and is not a real-robot result."
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
    module.DIFFUSION_JSON = DIFFUSION_JSON
    module.STATE_LATENT_JSON = STATE_LATENT_JSON
    module.SEED = int(os.environ.get("BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_GUIDANCE_SEED", str(DEFAULT_SEED)))
    module.MAX_WINDOWS_PER_SPLIT = int(
        os.environ.get(
            "BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_GUIDANCE_MAX_WINDOWS_PER_SPLIT",
            str(DEFAULT_MAX_WINDOWS_PER_SPLIT),
        )
    )
    module.BATCH_WINDOWS = int(
        os.environ.get("BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_GUIDANCE_BATCH_WINDOWS", str(module.BATCH_WINDOWS))
    )
    module.SCALES = os.environ.get("BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_GUIDANCE_SCALES", module.SCALES)
    module.main()

    base_json = OUT / "level_c_resource_adjusted_state_latent_guidance_eval.json"
    base_tsv = OUT / "level_c_resource_adjusted_state_latent_guidance_eval.tsv"
    final_json = OUT / "level_c_official_importer_export_scaled_ppo_state_latent_guidance_eval.json"
    final_tsv = OUT / "level_c_official_importer_export_scaled_ppo_state_latent_guidance_eval.tsv"
    summary = patch_summary(load_json(base_json))
    if base_tsv.is_file():
        base_tsv.replace(final_tsv)
    base_json.unlink(missing_ok=True)
    final_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metrics = summary.get("worker_summary", {}).get("metrics", {})
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(final_json),
                "total_selected_windows": metrics.get("total_selected_windows"),
                "tasks_with_all_best_costs_improve": metrics.get("tasks_with_all_best_costs_improve"),
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok_official_importer_export_scaled_ppo_state_latent_guidance_eval":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
