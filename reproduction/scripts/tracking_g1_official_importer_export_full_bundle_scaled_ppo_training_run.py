#!/usr/bin/env python3
"""Run a larger official-importer-export full-bundle PPO training stage.

This wrapper keeps the validated official-importer-export PPO harness but writes
to a separate scaled-run directory and raises the default environment count so
GPU memory telemetry can distinguish this from the earlier short 300-iteration
engineering run. It is still local virtual training, not an official
BeyondMimic teacher checkpoint.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_g1_official_importer_export_full_bundle_ppo_training_run.py"
OUT = ROOT / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_training_run"
LOG_DIR = ROOT / "logs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_training"
TARGET_GPUS = [4, 7]
DEFAULT_MAX_ITERATIONS = 1000
DEFAULT_NUM_ENVS_PER_RANK = 2048
DEFAULT_SEED = 20260693


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_official_importer_ppo_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base PPO script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    trained_ok = summary.get("status") == "ok_official_importer_export_full_bundle_ppo_training_completed"
    final_json = OUT / "tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run.json"
    summary["status"] = (
        "ok_official_importer_export_full_bundle_scaled_ppo_training_completed"
        if trained_ok
        else summary.get("status", "failed_official_importer_export_full_bundle_scaled_ppo_training")
    )
    summary["experiment_type"] = "tracking_official_importer_export_full_bundle_scaled_ppo_training_run"
    summary["scope"] = (
        "Larger local PPO training in Tracking-Flat-G1-v0 using the official-importer GPU4 G1 USDA export and "
        "the 40-motion public official csv-loop bundle. The default configuration uses GPUs 4/7, 2048 envs per "
        "rank, and 1000 PPO iterations to move beyond smoke-scale execution. It remains local virtual evidence, "
        "not the paper's official teacher-policy training run."
    )
    summary.setdefault("config", {})
    summary["config"]["max_iterations"] = int(
        os.environ.get("BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_MAX_ITERATIONS", str(DEFAULT_MAX_ITERATIONS))
    )
    summary["config"]["num_envs_per_rank"] = int(
        os.environ.get("BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_NUM_ENVS_PER_RANK", str(DEFAULT_NUM_ENVS_PER_RANK))
    )
    summary["config"]["seed"] = int(os.environ.get("BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_SEED", str(DEFAULT_SEED)))
    summary["config"]["formal_gpu_memory_target_mb_per_card"] = 10_000
    summary.setdefault("outputs", {})
    summary["outputs"]["json"] = str(final_json)
    summary["outputs"]["worker_script"] = str(OUT / "tracking_g1_resource_adjusted_ppo_worker.py")
    summary["interpretation"] = {
        "goal_complete": False,
        "official_importer_export_full_bundle_scaled_ppo_training_complete": bool(trained_ok),
        "official_ppo_training_complete": False,
        "paper_level_tracking_training_complete": False,
        "why_not_paper_level": (
            "This is a larger local virtual training run using a USDA produced by the official Isaac Sim importer "
            "and a public-motion bundle. It still does not use an official released BeyondMimic teacher checkpoint, "
            "does not reproduce the paper's full 30000-iteration teacher-training protocol, and does not prove real "
            "robot or Fig. 5/Fig. 6 behavior."
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
    module.TARGET_GPUS = TARGET_GPUS
    module.DEFAULT_MAX_ITERATIONS = int(
        os.environ.get("BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_MAX_ITERATIONS", str(DEFAULT_MAX_ITERATIONS))
    )
    module.DEFAULT_SEED = int(os.environ.get("BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_SEED", str(DEFAULT_SEED)))
    module.os.environ["BM_PPO_NUM_ENVS_PER_RANK"] = os.environ.get(
        "BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_NUM_ENVS_PER_RANK", str(DEFAULT_NUM_ENVS_PER_RANK)
    )
    module.os.environ["BM_OFFICIAL_IMPORTER_EXPORT_PPO_MAX_ITERATIONS"] = str(module.DEFAULT_MAX_ITERATIONS)
    module.os.environ["BM_OFFICIAL_IMPORTER_EXPORT_PPO_SEED"] = str(module.DEFAULT_SEED)
    module.main()

    base_json = OUT / "tracking_g1_official_importer_export_full_bundle_ppo_training_run.json"
    final_json = OUT / "tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run.json"
    summary = patch_summary(load_json(base_json))
    base_json.unlink(missing_ok=True)
    final_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(final_json),
                "checkpoint_count": summary.get("run", {}).get("checkpoint_count"),
                "max_iterations": summary.get("config", {}).get("max_iterations"),
                "num_envs_per_rank": summary.get("config", {}).get("num_envs_per_rank"),
                "selected_physical_gpus": summary.get("config", {}).get("selected_physical_gpus"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
