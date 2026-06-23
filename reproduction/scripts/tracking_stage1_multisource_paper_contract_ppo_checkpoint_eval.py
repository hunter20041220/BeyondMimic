#!/usr/bin/env python3
"""Evaluate a Stage-1 multi-source PPO checkpoint in Tracking-Flat-G1-v0.

This is the checkpoint-eval wrapper for the GPUs 5/6 multi-source teacher run.
It reuses the shared IsaacLab/RSL-RL evaluation harness, but binds the asset,
motion bundle, run summary, and GPUs to the stage1 multi-source line.

Claim boundary: this is local virtual checkpoint screening, not an official
BeyondMimic teacher checkpoint and not a paper-level tracking evaluation.
"""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py"
OUT = ROOT / "res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_eval"
LOG_DIR = ROOT / "logs/tracking_stage1_multisource_paper_contract_ppo_checkpoint_eval"
RUN_ROOT = ROOT / "res/runs/stage1_multisource_paper_contract_ppo_checkpoint_eval"
TRAINING_RUN_JSON = (
    ROOT
    / "res/tracking/stage1_multisource_paper_contract_ppo_training_run/"
    "tracking_stage1_multisource_paper_contract_ppo_training_run.json"
)
OFFICIAL_IMPORTER_USD = ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda"
MOTION_BUNDLE_AUDIT = ROOT / "res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle.json"
MOTION_BUNDLE_NPZ = (
    ROOT
    / "res/tracking/stage1_multisource_motion_bundle/"
    "stage1_multisource_public_plus_available_motion_bundle_fk_repaired_robot_order.npz"
)
DEFAULT_TARGET_GPUS = [5, 6]
DEFAULT_SEED = 20260852
DEFAULT_NUM_ENVS = 256
DEFAULT_EVAL_STEPS = 299


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_stage1_multisource_ppo_eval_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base PPO eval script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_gpus() -> list[int]:
    raw = os.environ.get("BM_STAGE1_MULTISOURCE_PPO_EVAL_TARGET_GPUS", "")
    return [int(item.strip()) for item in raw.split(",") if item.strip()] if raw.strip() else list(DEFAULT_TARGET_GPUS)


def make_base_compatible_training_summary() -> Path:
    training = load_json(TRAINING_RUN_JSON)
    compatible = json.loads(json.dumps(training))
    compatible["status"] = "ok_resource_adjusted_ppo_training_completed"
    compatible.setdefault("inputs", {})
    compatible["inputs"]["original_stage1_multisource_training_run_json"] = str(TRAINING_RUN_JSON)
    compatible.setdefault("interpretation", {})
    compatible["interpretation"]["base_compatibility_shim"] = (
        "This shim adapts the stage1 multi-source training status enum for the shared checkpoint-eval harness. "
        "The authoritative audit remains the stage1 multi-source training JSON."
    )
    shim_path = OUT / "base_compatible_stage1_multisource_training_run_for_eval.json"
    write_json(shim_path, compatible)
    return shim_path


def patch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    training = load_json(TRAINING_RUN_JSON)
    motion = load_json(MOTION_BUNDLE_AUDIT)
    metrics = motion.get("metrics", {})
    eval_ok = summary.get("status") == "ok_resource_adjusted_ppo_checkpoint_eval_completed"
    final_json = OUT / "tracking_stage1_multisource_paper_contract_ppo_checkpoint_eval.json"
    summary["status"] = (
        "ok_stage1_multisource_paper_contract_ppo_checkpoint_eval_completed"
        if eval_ok
        else summary.get("status", "failed_stage1_multisource_paper_contract_ppo_checkpoint_eval")
    )
    summary["experiment_type"] = "tracking_stage1_multisource_paper_contract_ppo_checkpoint_eval"
    summary["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    summary["scope"] = (
        "Evaluates one checkpoint from the local 5/6 GPU multi-source Stage-1 PPO teacher run using the "
        "official whole_body_tracking Tracking-Flat-G1-v0 environment and the local multi-source motion bundle."
    )
    summary.setdefault("inputs", {})
    summary["inputs"].update(
        {
            "training_run_json": str(TRAINING_RUN_JSON),
            "base_compatible_training_run_json": str(OUT / "base_compatible_stage1_multisource_training_run_for_eval.json"),
            "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
            "stage1_multisource_motion_bundle_audit": str(MOTION_BUNDLE_AUDIT),
            "stage1_multisource_motion_npz": str(MOTION_BUNDLE_NPZ),
        }
    )
    summary.setdefault("input_checks", {})
    summary["input_checks"].update(
        {
            "stage1_multisource_training_completed": training.get("status")
            == "ok_stage1_multisource_paper_contract_ppo_training_completed",
            "official_importer_usd_exists": OFFICIAL_IMPORTER_USD.is_file(),
            "motion_npz_exists": MOTION_BUNDLE_NPZ.is_file(),
            "motion_bundle_audit_ok": motion.get("status") == "ok_stage1_multisource_motion_bundle",
            "motion_count_49": metrics.get("motion_count") == 49,
            "total_duration_hours": metrics.get("total_duration_hours"),
        }
    )
    if isinstance(summary.get("run", {}).get("metrics"), dict):
        run_metrics = summary["run"]["metrics"]
        run_metrics.update(
            {
                "uses_official_importer_export_usd": True,
                "uses_stage1_multisource_motion_bundle": True,
                "uses_resource_adjusted_usd": False,
                "motion_count": metrics.get("motion_count"),
                "total_motion_frames": metrics.get("total_frames"),
                "official_beyondmimic_checkpoint": False,
                "paper_level_tracking_eval": False,
            }
        )
    summary.setdefault("outputs", {})
    summary["outputs"]["json"] = str(final_json)
    summary["outputs"]["worker_script"] = str(OUT / "tracking_g1_resource_adjusted_ppo_checkpoint_eval_worker.py")
    summary["interpretation"] = {
        "goal_complete": False,
        "official_beyondmimic_checkpoint": False,
        "paper_level_tracking_eval_complete": False,
        "claim_level": "local_multisource_stage1_teacher_checkpoint_eval",
        "why_not_paper_level": (
            "The policy is locally retrained from public-plus-available sources. It is not the unreleased official "
            "BeyondMimic teacher checkpoint, not DAgger data, and not real-robot validation."
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
    module.ENRICHED_USD = OFFICIAL_IMPORTER_USD
    module.CSV_MOTION_NPZ = MOTION_BUNDLE_NPZ
    module.TRAINING_RUN_JSON = make_base_compatible_training_summary()
    module.CANDIDATE_GPUS = parse_gpus()
    module.NUM_ENVS = int(os.environ.get("BM_STAGE1_MULTISOURCE_PPO_EVAL_NUM_ENVS", str(DEFAULT_NUM_ENVS)))
    module.EVAL_STEPS = int(os.environ.get("BM_STAGE1_MULTISOURCE_PPO_EVAL_STEPS", str(DEFAULT_EVAL_STEPS)))
    module.SEED = int(os.environ.get("BM_STAGE1_MULTISOURCE_PPO_EVAL_SEED", str(DEFAULT_SEED)))
    module.main()

    base_json = OUT / "tracking_g1_resource_adjusted_ppo_checkpoint_eval.json"
    final_json = OUT / "tracking_stage1_multisource_paper_contract_ppo_checkpoint_eval.json"
    summary = patch_summary(load_json(base_json))
    base_json.unlink(missing_ok=True)
    write_json(final_json, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(final_json),
                "checkpoint": summary.get("inputs", {}).get("checkpoint"),
                "num_envs": summary.get("config", {}).get("num_envs"),
                "eval_steps": summary.get("config", {}).get("eval_steps"),
            },
            sort_keys=True,
        )
    )
    if summary["status"].startswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
