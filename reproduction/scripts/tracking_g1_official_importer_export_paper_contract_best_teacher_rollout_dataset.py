#!/usr/bin/env python3
"""Collect teacher rollout shards from the paper-contract best checkpoint."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_g1_resource_adjusted_teacher_rollout_dataset.py"
OUT = ROOT / "res/tracking/g1_official_importer_export_paper_contract_best_teacher_rollout_dataset"
LOG_DIR = ROOT / "logs/tracking_g1_official_importer_export_paper_contract_best_teacher_rollout_dataset"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_importer_export_paper_contract_best_teacher_rollout_dataset"
BEST_TEACHER_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_paper_contract_ppo_checkpoint_sweep/"
    "paper_contract_best_teacher.json"
)
TRAINING_RUN_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_paper_contract_ppo_training_run/"
    "tracking_g1_official_importer_export_paper_contract_ppo_training_run.json"
)
OFFICIAL_IMPORTER_USD = ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda"
ROBOT_ORDER_BUNDLE_NPZ = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/"
    "official_csv_loop_full_public_motion_bundle_fk_repaired_robot_order.npz"
)
ROBOT_ORDER_BUNDLE_AUDIT = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/"
    "tracking_g1_official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz.json"
)
DEFAULT_SEED = 20260804
DEFAULT_NUM_ENVS_PER_RANK = 2048
TARGET_GPUS = [4, 7]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_paper_contract_teacher_rollout_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load teacher rollout base: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def best_checkpoint() -> Path:
    payload = load_json(BEST_TEACHER_JSON)
    path = Path(payload.get("best_checkpoint", {}).get("checkpoint", ""))
    if not path.is_file():
        raise FileNotFoundError(f"Best checkpoint not available. Run checkpoint sweep first: {BEST_TEACHER_JSON}")
    return path


def checkpoint_iteration(path: Path) -> int:
    try:
        return int(path.stem.split("_", maxsplit=1)[1])
    except Exception:
        return -1


def make_training_shim(checkpoint: Path) -> Path:
    training = load_json(TRAINING_RUN_JSON)
    iteration = checkpoint_iteration(checkpoint)
    shim_root = OUT / "best_teacher_training_shim"
    rank0 = shim_root / "run_dir/rank_0"
    rank0.mkdir(parents=True, exist_ok=True)
    link = rank0 / checkpoint.name
    if link.exists() or link.is_symlink():
        link.unlink()
    try:
        link.symlink_to(checkpoint)
    except OSError:
        shutil.copy2(checkpoint, link)
    shim = json.loads(json.dumps(training))
    shim["status"] = "ok_official_csv_loop_ppo_training_completed"
    shim.setdefault("outputs", {})
    shim["outputs"]["run_dir"] = str(shim_root / "run_dir")
    shim.setdefault("inputs", {})
    shim["inputs"].update(
        {
            "source_training_run_json": str(TRAINING_RUN_JSON),
            "source_best_teacher_json": str(BEST_TEACHER_JSON),
            "selected_best_checkpoint": str(checkpoint),
            "selected_best_iteration": iteration,
        }
    )
    shim.setdefault("interpretation", {})
    shim["interpretation"]["best_teacher_shim"] = (
        "Status and run_dir are adapted for the shared teacher-rollout harness. "
        "The selected checkpoint is the best local candidate from the paper-contract sweep."
    )
    path = shim_root / "paper_contract_best_teacher_training_summary.json"
    write_json(path, shim)
    return path


def patch_summary(summary: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    bundle = load_json(ROBOT_ORDER_BUNDLE_AUDIT)
    bundle_metrics = bundle.get("metrics", {})
    best = load_json(BEST_TEACHER_JSON)
    rollout_ok = summary.get("status") == "ok_resource_adjusted_teacher_rollout_dataset_completed"
    final_json = OUT / "tracking_g1_official_importer_export_paper_contract_best_teacher_rollout_dataset.json"
    summary["status"] = (
        "ok_official_importer_export_paper_contract_best_teacher_rollout_dataset_completed"
        if rollout_ok
        else summary.get("status", "failed_official_importer_export_paper_contract_best_teacher_rollout_dataset")
    )
    summary["experiment_type"] = "tracking_official_importer_export_paper_contract_best_teacher_rollout_dataset"
    summary["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    summary["scope"] = (
        "Collects state/action rollout shards from the best local paper-contract PPO teacher candidate selected by "
        "checkpoint sweep. This is the LAFAN1 downstream dataset source for local VAE/diffusion experiments."
    )
    summary.setdefault("inputs", {})
    summary["inputs"].update(
        {
            "best_teacher_json": str(BEST_TEACHER_JSON),
            "best_checkpoint": str(checkpoint),
            "best_iteration": checkpoint_iteration(checkpoint),
            "training_run_json": str(TRAINING_RUN_JSON),
            "base_compatible_training_run_json": str(
                OUT / "best_teacher_training_shim/paper_contract_best_teacher_training_summary.json"
            ),
            "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
            "robot_order_fk_repaired_motion_npz": str(ROBOT_ORDER_BUNDLE_NPZ),
            "robot_order_bundle_audit": str(ROBOT_ORDER_BUNDLE_AUDIT),
        }
    )
    summary.setdefault("input_checks", {})
    summary["input_checks"].update(
        {
            "best_teacher_json_exists": BEST_TEACHER_JSON.is_file(),
            "best_checkpoint_exists": checkpoint.is_file(),
            "best_teacher_status_ok": best.get("status") == "ok",
            "official_importer_usd_exists": OFFICIAL_IMPORTER_USD.is_file(),
            "robot_order_motion_npz_exists": ROBOT_ORDER_BUNDLE_NPZ.is_file(),
            "robot_order_bundle_audit_passed": bundle.get("status") == "ok_fk_repaired_robot_order_motion_npz",
            "robot_order_motion_count_40": bundle_metrics.get("motion_count") == 40,
            "robot_order_total_frames_11960": bundle_metrics.get("total_frames") == 11960,
        }
    )
    for shard in summary.get("run", {}).get("shard_metrics", []):
        shard["uses_official_importer_export_usd"] = True
        shard["uses_resource_adjusted_usd"] = False
        shard["uses_robot_order_fk_repaired_full_public_motion_bundle"] = True
        shard["source_paper_contract_best_checkpoint"] = True
        shard["official_dagger_rollout_dataset"] = False
        shard["paper_level_teacher_rollout_dataset"] = False
        shard["motion_count"] = bundle_metrics.get("motion_count")
        shard["total_motion_frames"] = bundle_metrics.get("total_frames")
    summary.setdefault("aggregate_metrics", {})
    summary["aggregate_metrics"].update(
        {
            "motion_count": bundle_metrics.get("motion_count"),
            "total_motion_frames": bundle_metrics.get("total_frames"),
            "source_paper_contract_best_checkpoint": True,
            "best_iteration": checkpoint_iteration(checkpoint),
        }
    )
    summary.setdefault("outputs", {})
    summary["outputs"]["json"] = str(final_json)
    summary["outputs"]["worker_script"] = str(OUT / "tracking_g1_resource_adjusted_teacher_rollout_worker.py")
    summary["interpretation"] = {
        "goal_complete": False,
        "official_dagger_dataset_complete": False,
        "paper_level_teacher_rollout_dataset_complete": False,
        "paper_contract_best_teacher_rollout_dataset_complete": bool(rollout_ok),
        "claim_level": "local_virtual_paper_contract_best_teacher_rollout_dataset",
        "why_not_paper_level": (
            "The source policy is locally retrained and locally selected. The official BeyondMimic teacher checkpoint "
            "and true DAgger rollout logs are not public in this workspace."
        ),
    }
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = best_checkpoint()
    shim = make_training_shim(checkpoint)
    module = load_base_module()
    module.OUT = OUT
    module.LOG_DIR = LOG_DIR
    module.RUN_ROOT = RUN_ROOT
    module.ENRICHED_USD = OFFICIAL_IMPORTER_USD
    module.CSV_MOTION_NPZ = ROBOT_ORDER_BUNDLE_NPZ
    module.TRAINING_RUN_JSON = shim
    module.CANDIDATE_GPUS = TARGET_GPUS
    module.NUM_ENVS_PER_RANK = int(
        os.environ.get("BM_PAPER_CONTRACT_TEACHER_ROLLOUT_NUM_ENVS_PER_RANK", str(DEFAULT_NUM_ENVS_PER_RANK))
    )
    module.SEED = int(os.environ.get("BM_PAPER_CONTRACT_TEACHER_ROLLOUT_SEED", str(DEFAULT_SEED)))
    module.main()

    base_json = OUT / "tracking_g1_resource_adjusted_teacher_rollout_dataset.json"
    final_json = OUT / "tracking_g1_official_importer_export_paper_contract_best_teacher_rollout_dataset.json"
    summary = patch_summary(load_json(base_json), checkpoint)
    base_json.unlink(missing_ok=True)
    write_json(final_json, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(final_json),
                "best_iteration": checkpoint_iteration(checkpoint),
                "attempted_rollout": summary.get("run", {}).get("attempted_rollout"),
                "shard_count": summary.get("aggregate_metrics", {}).get("shard_count"),
                "total_env_steps": summary.get("aggregate_metrics", {}).get("total_env_steps"),
            },
            sort_keys=True,
        )
    )
    if summary["status"].startswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
