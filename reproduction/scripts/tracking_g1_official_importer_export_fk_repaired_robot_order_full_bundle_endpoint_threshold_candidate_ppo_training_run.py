#!/usr/bin/env python3
"""Run full PPO with the robot-order FK bundle and endpoint-threshold candidate."""

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
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run.py"
)
OUT = (
    ROOT
    / "res/tracking/"
    "g1_official_importer_export_fk_repaired_robot_order_full_bundle_endpoint_threshold_candidate_ppo_training_run"
)
LOG_DIR = (
    ROOT
    / "logs/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_endpoint_threshold_candidate_ppo_training_run"
)
RUN_ROOT = (
    ROOT
    / "res/runs/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_endpoint_threshold_candidate_ppo_training"
)
THRESHOLD_SWEEP_JSON = (
    ROOT
    / "res/tracking/"
    "g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_threshold_sweep/"
    "endpoint_threshold_sweep.json"
)
DEFAULT_THRESHOLD = 0.5
DEFAULT_SEED = 20260760


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_robot_order_threshold_candidate_training_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base robot-order PPO training script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_summary(summary: dict[str, Any], threshold: float) -> dict[str, Any]:
    sweep = load_json(THRESHOLD_SWEEP_JSON)
    trained_ok = summary.get("status") == "ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_completed"
    final_json = OUT / (
        "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_"
        "endpoint_threshold_candidate_ppo_training_run.json"
    )
    summary["status"] = (
        "ok_official_importer_export_fk_repaired_robot_order_full_bundle_endpoint_threshold_candidate_ppo_training_completed"
        if trained_ok
        else summary.get(
            "status",
            "failed_official_importer_export_fk_repaired_robot_order_full_bundle_endpoint_threshold_candidate_ppo_training",
        )
    )
    summary["experiment_type"] = (
        "tracking_official_importer_export_fk_repaired_robot_order_full_bundle_endpoint_threshold_candidate_ppo_training"
    )
    summary["scope"] = (
        "Full robot-order FK-repaired public-bundle PPO training with all official endpoint bodies retained and the "
        "candidate ee_body_pos z-threshold selected by the endpoint-threshold sweep. This is a tracking-quality "
        "candidate run, not a paper-level score."
    )
    summary.setdefault("config", {})
    summary["config"]["endpoint_threshold_candidate"] = threshold
    summary["config"]["endpoint_threshold_source_json"] = str(THRESHOLD_SWEEP_JSON)
    summary["config"]["seed"] = int(
        os.environ.get("BM_ROBOT_ORDER_ENDPOINT_THRESHOLD_CANDIDATE_PPO_SEED", str(DEFAULT_SEED))
    )
    summary.setdefault("inputs", {})
    summary["inputs"]["endpoint_threshold_sweep_json"] = str(THRESHOLD_SWEEP_JSON)
    summary["inputs"]["endpoint_threshold_sweep_recommended_next_action"] = sweep.get("interpretation", {}).get(
        "recommended_next_action"
    )
    summary["inputs"]["endpoint_threshold_sweep_best_done_rate"] = sweep.get("comparison_to_baselines", {}).get(
        "best_done_rate"
    )
    for rank_metric in summary.get("run", {}).get("rank_metrics", []):
        rank_metric["ee_body_pos_threshold_candidate_training"] = True
        rank_metric["ee_body_pos_threshold_candidate"] = threshold
        rank_metric["paper_level_training"] = False
    summary.setdefault("outputs", {})
    summary["outputs"]["json"] = str(final_json)
    summary["outputs"]["worker_script"] = str(OUT / "tracking_g1_resource_adjusted_ppo_worker.py")
    summary["interpretation"] = {
        "goal_complete": False,
        "endpoint_threshold_candidate_training_complete": bool(trained_ok),
        "official_ppo_training_complete": False,
        "paper_level_tracking_training_complete": False,
        "claim_level": "tracking_endpoint_threshold_candidate_full_ppo_training",
        "why_mainline": (
            "The previous full-size threshold sweep showed that the endpoint threshold candidate lowers termination "
            "while keeping all official endpoint bodies. This run tests whether training under that candidate yields a "
            "more useful local teacher before regenerating downstream VAE/diffusion/guidance artifacts."
        ),
        "why_not_paper_level": (
            "The termination threshold is calibrated locally, so this cannot be reported as the official paper "
            "tracking metric or official BeyondMimic teacher training."
        ),
    }
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    sweep = load_json(THRESHOLD_SWEEP_JSON)
    threshold = float(
        os.environ.get(
            "BM_ROBOT_ORDER_ENDPOINT_THRESHOLD_CANDIDATE",
            str(sweep.get("comparison_to_baselines", {}).get("best_threshold", DEFAULT_THRESHOLD)),
        )
    )

    os.environ["BM_EE_BODY_POS_TRAIN_THRESHOLD"] = str(threshold)
    os.environ["BM_EE_BODY_POS_TRAIN_BODY_NAMES"] = ",".join(
        sweep.get("config", {}).get(
            "active_ee_body_names",
            [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        )
    )
    os.environ["BM_ROBOT_ORDER_FK_REPAIRED_FULL_BUNDLE_PPO_SEED"] = os.environ.get(
        "BM_ROBOT_ORDER_ENDPOINT_THRESHOLD_CANDIDATE_PPO_SEED", str(DEFAULT_SEED)
    )

    module = load_base_module()
    module.OUT = OUT
    module.LOG_DIR = LOG_DIR
    module.RUN_ROOT = RUN_ROOT
    module.DEFAULT_SEED = DEFAULT_SEED
    module.main()

    base_json = OUT / "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run.json"
    final_json = OUT / (
        "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_"
        "endpoint_threshold_candidate_ppo_training_run.json"
    )
    summary = patch_summary(load_json(base_json), threshold)
    base_json.unlink(missing_ok=True)
    final_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(final_json),
                "endpoint_threshold_candidate": threshold,
                "attempted_training": summary.get("run", {}).get("attempted_training"),
                "checkpoint_count": summary.get("run", {}).get("checkpoint_count"),
                "selected_physical_gpus": summary.get("config", {}).get("selected_physical_gpus"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
