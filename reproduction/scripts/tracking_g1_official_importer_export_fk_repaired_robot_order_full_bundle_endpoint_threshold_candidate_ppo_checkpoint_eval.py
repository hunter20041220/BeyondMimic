#!/usr/bin/env python3
"""Evaluate the endpoint-threshold-candidate robot-order PPO checkpoint."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = (
    ROOT
    / "reproduction/scripts/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_group_ablation.py"
)
OUT = (
    ROOT
    / "res/tracking/"
    "g1_official_importer_export_fk_repaired_robot_order_full_bundle_endpoint_threshold_candidate_ppo_checkpoint_eval"
)
LOG_DIR = (
    ROOT
    / "logs/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_endpoint_threshold_candidate_ppo_checkpoint_eval"
)
RUN_ROOT = (
    ROOT
    / "res/runs/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_endpoint_threshold_candidate_ppo_checkpoint_eval"
)
TRAINING_RUN_JSON = (
    ROOT
    / "res/tracking/"
    "g1_official_importer_export_fk_repaired_robot_order_full_bundle_endpoint_threshold_candidate_ppo_training_run/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_"
    "endpoint_threshold_candidate_ppo_training_run.json"
)
THRESHOLD_SWEEP_JSON = (
    ROOT
    / "res/tracking/"
    "g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_threshold_sweep/"
    "endpoint_threshold_sweep.json"
)
DEFAULT_SEED = 20260761


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_robot_order_threshold_candidate_eval_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base endpoint eval script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_base_compatible_training_summary() -> Path:
    training = load_json(TRAINING_RUN_JSON)
    compatible = dict(training)
    compatible["status"] = "ok_resource_adjusted_ppo_training_completed"
    compatible.setdefault("inputs", {})
    compatible["inputs"]["original_endpoint_threshold_candidate_training_run_json"] = str(TRAINING_RUN_JSON)
    compatible.setdefault("interpretation", {})
    compatible["interpretation"]["base_compatibility_shim"] = (
        "This copy adapts the endpoint-threshold-candidate training status enum for the shared endpoint eval harness."
    )
    shim_path = OUT / "base_compatible_endpoint_threshold_candidate_training_run_for_eval.json"
    shim_path.write_text(json.dumps(compatible, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return shim_path


def patch_summary(summary: dict[str, Any], threshold: float) -> dict[str, Any]:
    training = load_json(TRAINING_RUN_JSON)
    sweep = load_json(THRESHOLD_SWEEP_JSON)
    eval_ok = summary.get("status") == "ok_resource_adjusted_ppo_checkpoint_eval_completed"
    final_json = OUT / (
        "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_"
        "endpoint_threshold_candidate_ppo_checkpoint_eval.json"
    )
    summary["status"] = (
        "ok_official_importer_export_fk_repaired_robot_order_full_bundle_endpoint_threshold_candidate_ppo_checkpoint_eval_completed"
        if eval_ok
        else summary.get(
            "status",
            "failed_official_importer_export_fk_repaired_robot_order_full_bundle_endpoint_threshold_candidate_ppo_checkpoint_eval",
        )
    )
    summary["experiment_type"] = (
        "tracking_official_importer_export_fk_repaired_robot_order_full_bundle_endpoint_threshold_candidate_ppo_checkpoint_eval"
    )
    summary["scope"] = (
        "Evaluates the PPO checkpoint trained with the endpoint-threshold candidate under the same active endpoint "
        "body set and threshold. This is a local tracking-quality candidate eval, not a paper metric."
    )
    summary.setdefault("config", {})
    summary["config"]["seed"] = int(
        os.environ.get("BM_ROBOT_ORDER_ENDPOINT_THRESHOLD_CANDIDATE_PPO_EVAL_SEED", str(DEFAULT_SEED))
    )
    summary["config"]["endpoint_threshold_candidate"] = threshold
    summary["config"]["endpoint_threshold_source_json"] = str(THRESHOLD_SWEEP_JSON)
    summary.setdefault("inputs", {})
    summary["inputs"]["training_run_json"] = str(TRAINING_RUN_JSON)
    summary["inputs"]["endpoint_threshold_sweep_json"] = str(THRESHOLD_SWEEP_JSON)
    summary["inputs"]["training_status"] = training.get("status")
    summary["inputs"]["endpoint_threshold_sweep_best_done_rate"] = sweep.get("comparison_to_baselines", {}).get(
        "best_done_rate"
    )
    if "run" in summary and isinstance(summary["run"].get("metrics"), dict):
        run_metrics = summary["run"]["metrics"]
        run_metrics["ee_body_pos_threshold_candidate_eval"] = True
        run_metrics["ee_body_pos_threshold_candidate"] = threshold
        run_metrics["paper_level_tracking_eval"] = False
    summary.setdefault("outputs", {})
    summary["outputs"]["json"] = str(final_json)
    summary["outputs"]["worker_script"] = str(OUT / "tracking_g1_resource_adjusted_ppo_checkpoint_eval_worker.py")
    summary["interpretation"] = {
        "goal_complete": False,
        "endpoint_threshold_candidate_eval_complete": bool(eval_ok),
        "official_tracking_eval_complete": False,
        "paper_level_tracking_eval_complete": False,
        "claim_level": "tracking_endpoint_threshold_candidate_full_ppo_eval",
        "why_mainline": (
            "This eval checks whether the threshold-candidate training run improves the local tracking teacher before "
            "recollecting teacher rollout and rerunning downstream VAE/diffusion/guidance."
        ),
        "why_not_paper_level": (
            "The evaluator threshold is locally calibrated. The result is a candidate teacher-quality signal, not the "
            "official BeyondMimic paper tracking metric."
        ),
    }
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    training = load_json(TRAINING_RUN_JSON)
    threshold = float(
        os.environ.get(
            "BM_ROBOT_ORDER_ENDPOINT_THRESHOLD_CANDIDATE",
            str(training.get("config", {}).get("endpoint_threshold_candidate", 0.5)),
        )
    )
    sweep = load_json(THRESHOLD_SWEEP_JSON)
    active_names = sweep.get("config", {}).get(
        "active_ee_body_names",
        [
            "left_ankle_roll_link",
            "right_ankle_roll_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ],
    )

    module = load_base_module()
    module.OUT = OUT
    module.LOG_DIR = LOG_DIR
    module.RUN_ROOT = RUN_ROOT
    module.TRAINING_RUN_JSON = make_base_compatible_training_summary()
    module.DEFAULT_SEED = int(
        os.environ.get("BM_ROBOT_ORDER_ENDPOINT_THRESHOLD_CANDIDATE_PPO_EVAL_SEED", str(DEFAULT_SEED))
    )
    module.VARIANTS = [
        {
            "name": "endpoint_threshold_candidate_eval",
            "active_ee_body_names": active_names,
            "threshold": threshold,
            "description": "Evaluate the threshold-candidate PPO checkpoint with all official endpoint bodies active.",
        }
    ]
    variant = {
        "name": "endpoint_threshold_candidate_eval",
        "active_ee_body_names": active_names,
        "threshold": threshold,
        "description": "Evaluate the threshold-candidate PPO checkpoint with all official endpoint bodies active.",
    }
    variant_summary = module.run_variant(variant, module.TRAINING_RUN_JSON)
    base_json = Path(variant_summary["summary_json"])
    final_json = OUT / (
        "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_"
        "endpoint_threshold_candidate_ppo_checkpoint_eval.json"
    )
    summary = patch_summary(load_json(base_json), threshold)
    summary["variant_summary"] = {key: value for key, value in variant_summary.items() if key != "base_summary"}
    final_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(final_json),
                "endpoint_threshold_candidate": threshold,
                "attempted_eval": summary.get("run", {}).get("attempted_eval"),
                "metrics_exists": summary.get("run", {}).get("metrics_exists"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
