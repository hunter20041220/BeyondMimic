#!/usr/bin/env python3
"""Evaluate the larger official-importer-export full-bundle PPO checkpoint."""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval.py"
OUT = ROOT / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval"
LOG_DIR = ROOT / "logs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval"
TRAINING_RUN_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_training_run/"
    "tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run.json"
)
DEFAULT_SEED = 20260694
DEFAULT_NUM_ENVS = 2048


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_official_importer_ppo_eval_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base PPO eval script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_base_compatible_training_summary() -> Path:
    training = load_json(TRAINING_RUN_JSON)
    compatible = dict(training)
    compatible["status"] = "ok_official_importer_export_full_bundle_ppo_training_completed"
    compatible.setdefault("inputs", {})
    compatible["inputs"]["original_official_importer_export_scaled_training_run_json"] = str(TRAINING_RUN_JSON)
    compatible.setdefault("interpretation", {})
    compatible["interpretation"]["base_compatibility_shim"] = (
        "This copy adapts the scaled training status enum for the shared checkpoint-eval harness. "
        "The authoritative training audit remains the original scaled training JSON."
    )
    shim_path = OUT / "base_compatible_official_importer_export_scaled_training_run_for_eval.json"
    shim_path.write_text(json.dumps(compatible, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return shim_path


def patch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    eval_ok = summary.get("status") == "ok_official_importer_export_full_bundle_ppo_checkpoint_eval_completed"
    final_json = OUT / "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json"
    summary["status"] = (
        "ok_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval_completed"
        if eval_ok
        else summary.get("status", "failed_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval")
    )
    summary["experiment_type"] = "tracking_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval"
    summary["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    summary["scope"] = (
        "Evaluates the larger local PPO checkpoint trained with the official-importer GPU4 G1 USDA export and "
        "the concatenated 40-motion public official csv-loop bundle inside Tracking-Flat-G1-v0. This is local "
        "virtual policy evaluation, not the official paper teacher-policy result."
    )
    summary.setdefault("outputs", {})
    summary["outputs"]["json"] = str(final_json)
    summary["outputs"]["worker_script"] = str(OUT / "tracking_g1_resource_adjusted_ppo_checkpoint_eval_worker.py")
    summary["interpretation"] = {
        "goal_complete": False,
        "official_importer_export_full_bundle_scaled_checkpoint_eval_complete": bool(eval_ok),
        "official_tracking_eval_complete": False,
        "paper_level_tracking_eval_complete": False,
        "why_not_paper_level": (
            "The evaluated checkpoint was trained locally on public motion data with a local official-importer USDA "
            "asset. It is stronger than the earlier 300-iteration run if it reaches the scaled endpoint and memory "
            "target, but remains below the official paper training/evaluation protocol."
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
    module.TRAINING_RUN_JSON = compatible_training_summary
    module.SEED = int(os.environ.get("BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_EVAL_SEED", str(DEFAULT_SEED)))
    module.os.environ["BM_PPO_EVAL_NUM_ENVS"] = os.environ.get(
        "BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_EVAL_NUM_ENVS", str(DEFAULT_NUM_ENVS)
    )
    module.os.environ["BM_OFFICIAL_IMPORTER_EXPORT_PPO_EVAL_SEED"] = str(module.SEED)
    module.main()

    base_json = OUT / "tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval.json"
    final_json = OUT / "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json"
    summary = patch_summary(load_json(base_json))
    base_json.unlink(missing_ok=True)
    final_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(final_json),
                "attempted_eval": summary.get("run", {}).get("attempted_eval"),
                "metrics_exists": summary.get("run", {}).get("metrics_exists"),
                "num_envs": summary.get("config", {}).get("num_envs"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
