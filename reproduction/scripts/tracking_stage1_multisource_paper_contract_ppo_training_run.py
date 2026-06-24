#!/usr/bin/env python3
"""Start a parallel Stage-1 multi-source PPO teacher-training run on GPUs 5/6.

This wrapper intentionally does not touch the existing paper-contract run on
GPUs 4/7.  It reuses the audited official whole_body_tracking/RSL-RL training
harness, but replaces the motion file with the multi-source bundle built by
``tracking_stage1_multisource_motion_bundle.py``.
"""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_g1_resource_adjusted_ppo_training_run.py"
OUT = ROOT / "res/tracking/stage1_multisource_paper_contract_ppo_training_run"
LOG_DIR = ROOT / "logs/tracking_stage1_multisource_paper_contract_ppo_training_run"
RUN_ROOT = ROOT / "res/runs/stage1_multisource_paper_contract_ppo_training"
OFFICIAL_IMPORTER_USD = (
    ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda"
)
MOTION_BUNDLE_AUDIT = (
    ROOT / "res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle.json"
)
MOTION_BUNDLE_NPZ = (
    ROOT
    / "res/tracking/stage1_multisource_motion_bundle/"
    "stage1_multisource_public_plus_available_motion_bundle_fk_repaired_robot_order.npz"
)
TRAIN_ENTRY = (
    ROOT
    / "res/tracking/g1_official_importer_export_paper_contract_ppo_training_run/"
    "base_compatible_robot_order_fk_repaired_split_task_gate_for_training.json"
)
FINAL_JSON = OUT / "tracking_stage1_multisource_paper_contract_ppo_training_run.json"

TARGET_GPUS = [5, 6]
DEFAULT_MAX_ITERATIONS = 30000
DEFAULT_NUM_ENVS_PER_RANK = 2048
DEFAULT_SEED = 20260851
DEFAULT_SAVE_INTERVAL = 500
DEFAULT_ADAPTIVE_KERNEL_SIZE = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_multisource_ppo_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base PPO script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_worker_save_interval(module: Any) -> None:
    module.WORKER_CODE = module.WORKER_CODE.replace(
        'agent_cfg.save_interval = max(1, min(50, agent_cfg.max_iterations))',
        'agent_cfg.save_interval = int(os.environ.get("BM_PPO_SAVE_INTERVAL", "500"))',
    )
    module.WORKER_CODE = module.WORKER_CODE.replace(
        "env_cfg.commands.motion.debug_vis = False",
        (
            "env_cfg.commands.motion.debug_vis = False\n"
            "    env_cfg.commands.motion.adaptive_kernel_size = int(os.environ.get(\"BM_ADAPTIVE_KERNEL_SIZE\", \"3\"))"
        ),
    )


def patch_summary(summary: dict[str, Any], motion_audit: dict[str, Any]) -> dict[str, Any]:
    trained_ok = summary.get("status") == "ok_resource_adjusted_ppo_training_completed"
    summary["status"] = (
        "ok_stage1_multisource_paper_contract_ppo_training_completed"
        if trained_ok
        else summary.get("status", "started_or_blocked_stage1_multisource_paper_contract_ppo_training")
    )
    summary["experiment_type"] = "tracking_stage1_multisource_paper_contract_ppo_training_run"
    summary["scope"] = (
        "Parallel Stage-1 PPO teacher-policy training candidate on GPUs 5/6 using the official "
        "whole_body_tracking Tracking-Flat-G1-v0/RSL-RL harness and the local multi-source motion bundle. "
        "This is a local public-plus-available-data run, not an official BeyondMimic teacher checkpoint."
    )
    summary.setdefault("config", {})
    summary["config"].update(
        {
            "target_physical_gpus": TARGET_GPUS,
            "num_envs_per_rank": int(os.environ.get("BM_STAGE1_MULTISOURCE_NUM_ENVS_PER_RANK", DEFAULT_NUM_ENVS_PER_RANK)),
            "max_iterations": int(os.environ.get("BM_STAGE1_MULTISOURCE_MAX_ITERATIONS", DEFAULT_MAX_ITERATIONS)),
            "save_interval": int(os.environ.get("BM_STAGE1_MULTISOURCE_SAVE_INTERVAL", DEFAULT_SAVE_INTERVAL)),
            "seed": int(os.environ.get("BM_STAGE1_MULTISOURCE_SEED", DEFAULT_SEED)),
            "adaptive_kernel_size": int(
                os.environ.get("BM_STAGE1_MULTISOURCE_ADAPTIVE_KERNEL_SIZE", DEFAULT_ADAPTIVE_KERNEL_SIZE)
            ),
            "adaptive_kernel_source": (
                "Explicitly set for paper-contract runs because the paper supplement uses u={0,1,2}; "
                "the upstream public code default is adaptive_kernel_size=1."
            ),
            "motion_count": motion_audit.get("metrics", {}).get("motion_count"),
            "motion_duration_hours": motion_audit.get("metrics", {}).get("total_duration_hours"),
        }
    )
    summary.setdefault("inputs", {})
    summary["inputs"].update(
        {
            "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
            "stage1_multisource_motion_bundle_audit": str(MOTION_BUNDLE_AUDIT),
            "stage1_multisource_motion_npz": str(MOTION_BUNDLE_NPZ),
            "train_entry": str(TRAIN_ENTRY),
        }
    )
    summary.setdefault("outputs", {})
    summary["outputs"].update(
        {
            "json": str(FINAL_JSON),
            "worker_script": str(OUT / "tracking_g1_resource_adjusted_ppo_worker.py"),
        }
    )
    summary["interpretation"] = {
        "goal_complete": False,
        "paper_level_tracking_training_complete": False,
        "official_beyondmimic_checkpoint": False,
        "does_not_touch_current_4_7_training": True,
        "claim_level": "local_multisource_stage1_teacher_training_candidate",
        "why_this_run": (
            "The existing 4/7 run uses the 40-motion LAFAN public bundle. This parallel line adds all currently "
            "directly trainable local sources: full LAFAN1 CSVs, the Zenodo tkd_skill generalized-coordinate CSV, "
            "and HuB 29-DoF pkl motions converted to the official 36-column CSV contract."
        ),
        "why_not_paper_level": (
            "The paper's full prior-work plus online-animation motion set is not completely present as train-ready "
            "G1 generalized-coordinate data. PBHC sidekick and ASAP Ronaldo are present but only as 23-DoF pkl files, "
            "so they are excluded until an audited 23-to-29 mapping is implemented. This run must still be evaluated "
            "for tracking quality before any downstream VAE/diffusion data collection."
        ),
    }
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    motion_audit = load_json(MOTION_BUNDLE_AUDIT)
    input_checks = {
        "motion_bundle_audit_ok": motion_audit.get("status") == "ok_stage1_multisource_motion_bundle",
        "motion_npz_exists": MOTION_BUNDLE_NPZ.is_file(),
        "official_importer_usd_exists": OFFICIAL_IMPORTER_USD.is_file(),
        "train_entry_exists": TRAIN_ENTRY.is_file(),
        "train_entry_status_ok": load_json(TRAIN_ENTRY).get("status") == "ok_resource_adjusted_train_entry_diagnostic",
    }
    if not all(input_checks.values()):
        summary = {
            "status": "blocked_stage1_multisource_paper_contract_ppo_training_preflight",
            "experiment_type": "tracking_stage1_multisource_paper_contract_ppo_training_run",
            "timestamp_utc": utc_now(),
            "input_checks": input_checks,
            "inputs": {
                "motion_bundle_audit": str(MOTION_BUNDLE_AUDIT),
                "motion_npz": str(MOTION_BUNDLE_NPZ),
                "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
                "train_entry": str(TRAIN_ENTRY),
            },
            "interpretation": {"goal_complete": False},
        }
        write_json(FINAL_JSON, summary)
        print(json.dumps({"status": summary["status"], "json": str(FINAL_JSON)}, sort_keys=True))
        raise SystemExit(1)

    module = load_base_module()
    patch_worker_save_interval(module)
    module.OUT = OUT
    module.LOG_DIR = LOG_DIR
    module.RUN_ROOT = RUN_ROOT
    module.ENRICHED_USD = OFFICIAL_IMPORTER_USD
    module.CSV_MOTION_NPZ = MOTION_BUNDLE_NPZ
    module.TRAIN_ENTRY = TRAIN_ENTRY
    module.CANDIDATE_GPUS = TARGET_GPUS
    module.MIN_FREE_MB = int(os.environ.get("BM_STAGE1_MULTISOURCE_MIN_FREE_MB", "20000"))
    module.MAX_BUSY_UTIL = int(os.environ.get("BM_STAGE1_MULTISOURCE_MAX_BUSY_UTIL", "50"))
    module.NUM_ENVS_PER_RANK = int(os.environ.get("BM_STAGE1_MULTISOURCE_NUM_ENVS_PER_RANK", str(DEFAULT_NUM_ENVS_PER_RANK)))
    module.MAX_ITERATIONS = int(os.environ.get("BM_STAGE1_MULTISOURCE_MAX_ITERATIONS", str(DEFAULT_MAX_ITERATIONS)))
    module.SEED = int(os.environ.get("BM_STAGE1_MULTISOURCE_SEED", str(DEFAULT_SEED)))
    os.environ["BM_PPO_SAVE_INTERVAL"] = os.environ.get("BM_STAGE1_MULTISOURCE_SAVE_INTERVAL", str(DEFAULT_SAVE_INTERVAL))
    os.environ["BM_ADAPTIVE_KERNEL_SIZE"] = os.environ.get(
        "BM_STAGE1_MULTISOURCE_ADAPTIVE_KERNEL_SIZE", str(DEFAULT_ADAPTIVE_KERNEL_SIZE)
    )

    module.main()

    base_json = OUT / "tracking_g1_resource_adjusted_ppo_training_run.json"
    summary = patch_summary(load_json(base_json), motion_audit)
    base_json.unlink(missing_ok=True)
    write_json(FINAL_JSON, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(FINAL_JSON),
                "attempted_training": summary.get("run", {}).get("attempted_training"),
                "checkpoint_count": summary.get("run", {}).get("checkpoint_count"),
                "target_gpus": TARGET_GPUS,
                "motion_count": motion_audit.get("metrics", {}).get("motion_count"),
                "motion_duration_hours": motion_audit.get("metrics", {}).get("total_duration_hours"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
