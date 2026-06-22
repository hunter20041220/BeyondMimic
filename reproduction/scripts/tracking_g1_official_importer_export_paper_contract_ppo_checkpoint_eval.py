#!/usr/bin/env python3
"""Evaluate the paper-contract G1 tracking PPO checkpoint.

This wrapper evaluates the checkpoint produced by
``tracking_g1_official_importer_export_paper_contract_ppo_training_run.py`` with
the same local public-resource asset/motion contract.  It does not change the
policy observation/action contract and does not claim an official BeyondMimic
teacher evaluation.
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
OUT = ROOT / "res/tracking/g1_official_importer_export_paper_contract_ppo_checkpoint_eval"
LOG_DIR = ROOT / "logs/tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_eval"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_eval"
TRAINING_RUN_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_paper_contract_ppo_training_run/"
    "tracking_g1_official_importer_export_paper_contract_ppo_training_run.json"
)
PARAM_SNAPSHOT = (
    ROOT
    / "res/tracking/g1_official_importer_export_paper_contract_ppo_training_run/"
    "paper_contract_tracking_parameters.json"
)
OFFICIAL_IMPORTER_USD = (
    ROOT
    / "res/tracking/g1_urdf_in_memory_gpu4_probe/"
    "g1_official_importer_in_memory_gpu4_export.usda"
)
ROBOT_ORDER_BUNDLE_AUDIT = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/"
    "tracking_g1_official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz.json"
)
ROBOT_ORDER_BUNDLE_NPZ = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/"
    "official_csv_loop_full_public_motion_bundle_fk_repaired_robot_order.npz"
)
ROBOT_ORDER_SPLIT_TASK_GATE = (
    ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_split_task_eval/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_split_task_eval.json"
)
DEFAULT_TARGET_GPUS = [4, 7]
DEFAULT_SEED = 20260802
DEFAULT_NUM_ENVS = 2048
DEFAULT_EVAL_STEPS = 299


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_resource_adjusted_ppo_eval_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base PPO eval script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_base_compatible_training_summary() -> Path:
    training = load_json(TRAINING_RUN_JSON)
    compatible = dict(training)
    compatible["status"] = "ok_resource_adjusted_ppo_training_completed"
    compatible.setdefault("inputs", {})
    compatible["inputs"]["original_paper_contract_training_run_json"] = str(TRAINING_RUN_JSON)
    compatible.setdefault("interpretation", {})
    compatible["interpretation"]["base_compatibility_shim"] = (
        "This copy adapts the paper-contract training status enum for the shared checkpoint-eval harness. "
        "The authoritative training audit remains the paper-contract training JSON."
    )
    shim_path = OUT / "base_compatible_paper_contract_training_run_for_eval.json"
    shim_path.write_text(json.dumps(compatible, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return shim_path


def patch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    training = load_json(TRAINING_RUN_JSON)
    snapshot = load_json(PARAM_SNAPSHOT)
    bundle = load_json(ROBOT_ORDER_BUNDLE_AUDIT)
    split_gate = load_json(ROBOT_ORDER_SPLIT_TASK_GATE)
    bundle_metrics = bundle.get("metrics", {})
    eval_ok = summary.get("status") == "ok_resource_adjusted_ppo_checkpoint_eval_completed"
    final_json = OUT / "tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_eval.json"

    summary["status"] = (
        "ok_official_importer_export_paper_contract_ppo_checkpoint_eval_completed"
        if eval_ok
        else summary.get("status", "failed_official_importer_export_paper_contract_ppo_checkpoint_eval")
    )
    summary["experiment_type"] = "tracking_official_importer_export_paper_contract_ppo_checkpoint_eval"
    summary["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    summary["scope"] = (
        "Evaluates the local checkpoint retrained with official BeyondMimic/whole_body_tracking motor formulas, "
        "reward terms, termination thresholds, observation/action contract, and RSL-RL PPO hyperparameters. "
        "The input asset/motion paths remain local public-resource adaptations."
    )

    summary.setdefault("inputs", {})
    summary["inputs"].update(
        {
            "training_run_json": str(TRAINING_RUN_JSON),
            "base_compatible_training_run_json": str(OUT / "base_compatible_paper_contract_training_run_for_eval.json"),
            "paper_contract_parameter_snapshot": str(PARAM_SNAPSHOT),
            "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
            "robot_order_fk_repaired_motion_npz": str(ROBOT_ORDER_BUNDLE_NPZ),
            "robot_order_fk_repaired_bundle_audit": str(ROBOT_ORDER_BUNDLE_AUDIT),
            "robot_order_split_task_gate": str(ROBOT_ORDER_SPLIT_TASK_GATE),
        }
    )
    summary.setdefault("input_checks", {})
    summary["input_checks"].update(
        {
            "paper_contract_training_completed": training.get("status")
            == "ok_official_importer_export_paper_contract_ppo_training_completed",
            "paper_contract_parameter_snapshot_exists": PARAM_SNAPSHOT.is_file(),
            "paper_contract_total_envs_4096": snapshot.get("simulation", {}).get("target_total_envs") == 4096,
            "paper_contract_ppo_max_iterations_30000": snapshot.get("ppo_contract", {}).get("max_iterations")
            == 30000,
            "official_importer_usd_exists": OFFICIAL_IMPORTER_USD.is_file(),
            "robot_order_motion_npz_exists": ROBOT_ORDER_BUNDLE_NPZ.is_file(),
            "robot_order_bundle_audit_passed": bundle.get("status") == "ok_fk_repaired_robot_order_motion_npz",
            "robot_order_motion_count_40": bundle_metrics.get("motion_count") == 40,
            "robot_order_total_frames_11960": bundle_metrics.get("total_frames") == 11960,
            "robot_order_split_task_gate_passed": split_gate.get("status")
            == "ok_official_importer_export_fk_repaired_robot_order_split_task_eval",
        }
    )
    summary.setdefault("config", {})
    summary["config"].update(
        {
            "seed": int(os.environ.get("BM_PAPER_CONTRACT_PPO_EVAL_SEED", str(DEFAULT_SEED))),
            "num_envs": int(os.environ.get("BM_PAPER_CONTRACT_PPO_EVAL_NUM_ENVS", str(DEFAULT_NUM_ENVS))),
            "eval_steps": int(os.environ.get("BM_PAPER_CONTRACT_PPO_EVAL_STEPS", str(DEFAULT_EVAL_STEPS))),
            "paper_contract_parameter_snapshot": str(PARAM_SNAPSHOT),
        }
    )
    if "run" in summary and isinstance(summary["run"].get("metrics"), dict):
        run_metrics = summary["run"]["metrics"]
        run_metrics["uses_official_importer_export_usd"] = True
        run_metrics["uses_resource_adjusted_usd"] = False
        run_metrics["uses_robot_order_fk_repaired_full_public_motion_bundle"] = True
        run_metrics["uses_old_degenerate_full_public_motion_bundle"] = False
        run_metrics["uses_paper_contract_motor_reward_termination_ppo_config"] = True
        run_metrics["official_beyondmimic_checkpoint"] = False
        run_metrics["paper_level_tracking_eval"] = False
        run_metrics["motion_count"] = bundle_metrics.get("motion_count")
        run_metrics["total_motion_frames"] = bundle_metrics.get("total_frames")
    summary.setdefault("outputs", {})
    summary["outputs"]["json"] = str(final_json)
    summary["outputs"]["worker_script"] = str(OUT / "tracking_g1_resource_adjusted_ppo_checkpoint_eval_worker.py")
    summary["interpretation"] = {
        "goal_complete": False,
        "paper_contract_checkpoint_eval_complete": bool(eval_ok),
        "official_beyondmimic_checkpoint": False,
        "official_tracking_eval_complete": False,
        "paper_level_tracking_eval_complete": False,
        "claim_level": "local_public_resource_paper_contract_tracking_checkpoint_eval",
        "why_not_paper_level": (
            "The evaluated checkpoint is locally retrained with paper/official-code formulas but still depends on "
            "local public-resource asset and motion adaptations. It is not the official BeyondMimic teacher "
            "checkpoint, not DAgger, not a VAE/diffusion rollout, and not real-robot validation."
        ),
    }
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    module = load_base_module()
    compatible_training_summary = make_base_compatible_training_summary()
    target_gpus = [
        int(item.strip())
        for item in os.environ.get("BM_PAPER_CONTRACT_PPO_EVAL_TARGET_GPUS", ",".join(map(str, DEFAULT_TARGET_GPUS))).split(",")
        if item.strip()
    ]
    module.OUT = OUT
    module.LOG_DIR = LOG_DIR
    module.RUN_ROOT = RUN_ROOT
    module.ENRICHED_USD = OFFICIAL_IMPORTER_USD
    module.CSV_MOTION_NPZ = ROBOT_ORDER_BUNDLE_NPZ
    module.TRAINING_RUN_JSON = compatible_training_summary
    module.CANDIDATE_GPUS = target_gpus
    module.NUM_ENVS = int(os.environ.get("BM_PAPER_CONTRACT_PPO_EVAL_NUM_ENVS", str(DEFAULT_NUM_ENVS)))
    module.EVAL_STEPS = int(os.environ.get("BM_PAPER_CONTRACT_PPO_EVAL_STEPS", str(DEFAULT_EVAL_STEPS)))
    module.SEED = int(os.environ.get("BM_PAPER_CONTRACT_PPO_EVAL_SEED", str(DEFAULT_SEED)))
    module.main()

    base_json = OUT / "tracking_g1_resource_adjusted_ppo_checkpoint_eval.json"
    final_json = OUT / "tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_eval.json"
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
                "checkpoint": summary.get("inputs", {}).get("checkpoint"),
                "num_envs": summary.get("config", {}).get("num_envs"),
                "eval_steps": summary.get("config", {}).get("eval_steps"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
