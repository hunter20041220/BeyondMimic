#!/usr/bin/env python3
"""Train a focused Stage-1 teacher on the Hub single-leg reference motion.

This is a repair line for the current failure mode where downstream
teacher/VAE/diffusion videos do not learn the lifted-leg standing posture.
It intentionally uses one continuous Hub single-leg motion so the Stage-1
teacher quality can be debugged before reusing the larger weak multi-source
teacher for downstream VAE/diffusion work.
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
OUT = ROOT / "res/tracking/hub_singleleg_paper_contract_ppo_training_run"
LOG_DIR = ROOT / "logs/tracking_hub_singleleg_paper_contract_ppo_training_run"
RUN_ROOT = ROOT / "res/runs/hub_singleleg_paper_contract_ppo_training"
OFFICIAL_IMPORTER_USD = (
    ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda"
)
SINGLELEG_MOTION_NPZ = (
    ROOT
    / "res/tracking/stage1_multisource_motion_bundle/motions/"
    "hub_singleleg_video_single_leg_stand_1/motion.npz"
)
STAGE1_MOTION_BUNDLE_AUDIT = (
    ROOT / "res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle.json"
)
TRAIN_ENTRY = (
    ROOT
    / "res/tracking/g1_official_importer_export_paper_contract_ppo_training_run/"
    "base_compatible_robot_order_fk_repaired_split_task_gate_for_training.json"
)
RUN_TAG = os.environ.get("BM_HUB_SINGLELEG_RUN_TAG", "").strip()
if RUN_TAG:
    OUT = ROOT / f"res/tracking/hub_singleleg_paper_contract_ppo_training_run_{RUN_TAG}"
    LOG_DIR = ROOT / f"logs/tracking_hub_singleleg_paper_contract_ppo_training_run_{RUN_TAG}"
    RUN_ROOT = ROOT / f"res/runs/hub_singleleg_paper_contract_ppo_training_{RUN_TAG}"
FINAL_JSON = OUT / "tracking_hub_singleleg_paper_contract_ppo_training_run.json"

DEFAULT_TARGET_GPUS = [5, 6]
DEFAULT_MAX_ITERATIONS = 5000
DEFAULT_NUM_ENVS_PER_RANK = 4096
DEFAULT_SAVE_INTERVAL = 250
DEFAULT_SEED = 20260911
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


def parse_gpus() -> list[int]:
    raw = os.environ.get("BM_HUB_SINGLELEG_TARGET_GPUS", "")
    if raw.strip():
        return [int(item.strip()) for item in raw.split(",") if item.strip()]
    return list(DEFAULT_TARGET_GPUS)


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("bm_hub_singleleg_ppo_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base PPO script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_worker_save_interval(module: Any) -> None:
    module.WORKER_CODE = module.WORKER_CODE.replace(
        'agent_cfg.save_interval = max(1, min(50, agent_cfg.max_iterations))',
        'agent_cfg.save_interval = int(os.environ.get("BM_PPO_SAVE_INTERVAL", "250"))',
    )
    module.WORKER_CODE = module.WORKER_CODE.replace(
        "env_cfg.commands.motion.debug_vis = False",
        (
            "env_cfg.commands.motion.debug_vis = False\n"
            "    env_cfg.commands.motion.adaptive_kernel_size = int(os.environ.get(\"BM_ADAPTIVE_KERNEL_SIZE\", \"3\"))"
        ),
    )


def patch_summary(summary: dict[str, Any], motion_audit: dict[str, Any], target_gpus: list[int]) -> dict[str, Any]:
    trained_ok = summary.get("status") == "ok_resource_adjusted_ppo_training_completed"
    summary["status"] = (
        "ok_hub_singleleg_paper_contract_ppo_training_completed"
        if trained_ok
        else summary.get("status", "started_or_blocked_hub_singleleg_paper_contract_ppo_training")
    )
    summary["experiment_type"] = "tracking_hub_singleleg_paper_contract_ppo_training_run"
    summary["timestamp_utc"] = utc_now()
    summary["scope"] = (
        "Focused Stage-1 PPO teacher repair run on the continuous Hub single-leg standing motion using the "
        "official whole_body_tracking Tracking-Flat-G1-v0/RSL-RL harness and the local official-importer-export "
        "G1 USDA. This is a single-motion teacher-quality recovery experiment, not the full BeyondMimic "
        "multi-motion teacher and not a paper-level result."
    )
    summary.setdefault("config", {})
    summary["config"].update(
        {
            "target_physical_gpus": target_gpus,
            "num_envs_per_rank": int(
                os.environ.get("BM_HUB_SINGLELEG_NUM_ENVS_PER_RANK", DEFAULT_NUM_ENVS_PER_RANK)
            ),
            "max_iterations": int(os.environ.get("BM_HUB_SINGLELEG_MAX_ITERATIONS", DEFAULT_MAX_ITERATIONS)),
            "save_interval": int(os.environ.get("BM_HUB_SINGLELEG_SAVE_INTERVAL", DEFAULT_SAVE_INTERVAL)),
            "seed": int(os.environ.get("BM_HUB_SINGLELEG_SEED", DEFAULT_SEED)),
            "adaptive_kernel_size": int(
                os.environ.get("BM_HUB_SINGLELEG_ADAPTIVE_KERNEL_SIZE", DEFAULT_ADAPTIVE_KERNEL_SIZE)
            ),
            "adaptive_kernel_source": (
                "Explicitly set for paper-contract repair runs because the paper supplement uses u={0,1,2}; "
                "the upstream public code default is adaptive_kernel_size=1."
            ),
            "motion_count": 1,
            "motion_name": "hub_singleleg_video_single_leg_stand_1",
            "source_motion_duration_seconds": 15.98,
            "formal_gpu_experiment_target": True,
            "run_tag": RUN_TAG,
            "expected_per_gpu_memory_rule": (
                "Use GPUs 5 and 6. For formal retraining, increase num_envs_per_rank until memory is near the "
                "requested high-throughput target without OOM; actual peak is recorded in gpu_metrics.csv."
            ),
            "ee_body_pos_train_threshold": os.environ.get("BM_EE_BODY_POS_TRAIN_THRESHOLD"),
            "ee_body_pos_train_body_names": os.environ.get("BM_EE_BODY_POS_TRAIN_BODY_NAMES"),
        }
    )
    summary.setdefault("inputs", {})
    summary["inputs"].update(
        {
            "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
            "singleleg_motion_npz": str(SINGLELEG_MOTION_NPZ),
            "stage1_multisource_motion_bundle_audit": str(STAGE1_MOTION_BUNDLE_AUDIT),
            "train_entry": str(TRAIN_ENTRY),
        }
    )
    summary.setdefault("input_checks", {})
    stage1_rows = motion_audit.get("rows", [])
    summary["input_checks"].update(
        {
            "singleleg_motion_npz_exists": SINGLELEG_MOTION_NPZ.is_file(),
            "singleleg_motion_in_stage1_bundle": any(
                row.get("motion") == "hub_singleleg_video_single_leg_stand_1"
                for row in stage1_rows
                if isinstance(row, dict)
            ),
            "official_importer_usd_exists": OFFICIAL_IMPORTER_USD.is_file(),
            "train_entry_exists": TRAIN_ENTRY.is_file(),
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
        "single_motion_teacher_repair_line": True,
        "why_this_run": (
            "The current multi-source and scaled-PPO teachers do not reproduce the single-leg posture; this run "
            "isolates the same motion so Stage-1 teacher quality can be verified before downstream VAE/diffusion."
        ),
        "why_not_paper_level": (
            "This is a local single-motion repair experiment. The paper's full 2.5h diverse teacher set, official "
            "teacher checkpoints, true DAgger rollout logs, Fig.5/Fig.6 metrics, TensorRT deployment, and real robot "
            "execution remain absent."
        ),
    }
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    motion_audit = load_json(STAGE1_MOTION_BUNDLE_AUDIT)
    input_checks = {
        "base_script_exists": BASE_SCRIPT.is_file(),
        "singleleg_motion_npz_exists": SINGLELEG_MOTION_NPZ.is_file(),
        "official_importer_usd_exists": OFFICIAL_IMPORTER_USD.is_file(),
        "stage1_motion_bundle_audit_ok": motion_audit.get("status") == "ok_stage1_multisource_motion_bundle",
        "train_entry_status_ok": load_json(TRAIN_ENTRY).get("status") == "ok_resource_adjusted_train_entry_diagnostic",
    }
    if not all(input_checks.values()):
        summary = {
            "status": "blocked_hub_singleleg_paper_contract_ppo_training_preflight",
            "experiment_type": "tracking_hub_singleleg_paper_contract_ppo_training_run",
            "timestamp_utc": utc_now(),
            "input_checks": input_checks,
            "inputs": {
                "base_script": str(BASE_SCRIPT),
                "singleleg_motion_npz": str(SINGLELEG_MOTION_NPZ),
                "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
                "stage1_motion_bundle_audit": str(STAGE1_MOTION_BUNDLE_AUDIT),
                "train_entry": str(TRAIN_ENTRY),
            },
            "interpretation": {"goal_complete": False},
        }
        write_json(FINAL_JSON, summary)
        print(json.dumps({"status": summary["status"], "json": str(FINAL_JSON)}, sort_keys=True))
        raise SystemExit(1)

    target_gpus = parse_gpus()
    module = load_base_module()
    patch_worker_save_interval(module)
    module.OUT = OUT
    module.LOG_DIR = LOG_DIR
    module.RUN_ROOT = RUN_ROOT
    module.ENRICHED_USD = OFFICIAL_IMPORTER_USD
    module.CSV_MOTION_NPZ = SINGLELEG_MOTION_NPZ
    module.TRAIN_ENTRY = TRAIN_ENTRY
    module.CANDIDATE_GPUS = target_gpus
    module.MIN_FREE_MB = int(os.environ.get("BM_HUB_SINGLELEG_MIN_FREE_MB", "20000"))
    module.MAX_BUSY_UTIL = int(os.environ.get("BM_HUB_SINGLELEG_MAX_BUSY_UTIL", "50"))
    module.NUM_ENVS_PER_RANK = int(
        os.environ.get("BM_HUB_SINGLELEG_NUM_ENVS_PER_RANK", str(DEFAULT_NUM_ENVS_PER_RANK))
    )
    module.MAX_ITERATIONS = int(os.environ.get("BM_HUB_SINGLELEG_MAX_ITERATIONS", str(DEFAULT_MAX_ITERATIONS)))
    module.SEED = int(os.environ.get("BM_HUB_SINGLELEG_SEED", str(DEFAULT_SEED)))
    os.environ["BM_PPO_SAVE_INTERVAL"] = os.environ.get(
        "BM_HUB_SINGLELEG_SAVE_INTERVAL", str(DEFAULT_SAVE_INTERVAL)
    )
    os.environ["BM_ADAPTIVE_KERNEL_SIZE"] = os.environ.get(
        "BM_HUB_SINGLELEG_ADAPTIVE_KERNEL_SIZE", str(DEFAULT_ADAPTIVE_KERNEL_SIZE)
    )

    module.main()

    base_json = OUT / "tracking_g1_resource_adjusted_ppo_training_run.json"
    summary = patch_summary(load_json(base_json), motion_audit, target_gpus)
    base_json.unlink(missing_ok=True)
    write_json(FINAL_JSON, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(FINAL_JSON),
                "attempted_training": summary.get("run", {}).get("attempted_training"),
                "checkpoint_count": summary.get("run", {}).get("checkpoint_count"),
                "target_gpus": target_gpus,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
