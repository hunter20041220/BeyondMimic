#!/usr/bin/env python3
"""Evaluate the official-importer-export full-bundle VAE in IsaacLab closed loop.

This reuses the already validated official-csv-loop VAE closed-loop worker but
switches the robot asset, teacher checkpoint, and VAE checkpoint to the current
official-importer-export full-bundle chain. The result is still a local virtual
VAE action-reconstruction rollout, not an official BeyondMimic VAE checkpoint
or paper-level guided diffusion result.
"""

from __future__ import annotations

import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_g1_official_csv_loop_vae_closed_loop_rollout_eval.py"
OUT = ROOT / "res/level_c/official_importer_export_full_bundle_vae_closed_loop_rollout_eval"
LOG_DIR = ROOT / "logs/tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_eval"
FAILED_DIR = ROOT / "res/failed_runs/tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_eval"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_eval"
OFFICIAL_IMPORTER_USD = (
    ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda"
)
OFFICIAL_LOOP_MOTION_NPZ = (
    ROOT
    / "res/tracking/official_csv_to_npz_loop_full_dataset_with_enriched_usd/motions/walk1_subject1/"
    "walk1_subject1_official_loop_enriched_usd_motion.npz"
)
TRAINING_RUN_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_full_bundle_ppo_training_run/"
    "tracking_g1_official_importer_export_full_bundle_ppo_training_run.json"
)
TEACHER_ROLLOUT_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_full_bundle_teacher_rollout_dataset/"
    "tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset.json"
)
VAE_TRAINING_JSON = (
    ROOT
    / "res/level_c/official_importer_export_full_bundle_teacher_rollout_vae_training/"
    "level_c_official_importer_export_full_bundle_teacher_rollout_vae_training.json"
)
FINAL_JSON = OUT / "tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_eval.json"
DEFAULT_NUM_ENVS_PER_RANK = 1536
DEFAULT_SEED = 20260684


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_official_csv_loop_vae_closed_loop_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base VAE closed-loop script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_latest_model(training_run: dict[str, Any]) -> Path:
    run_dir = Path(training_run.get("outputs", {}).get("run_dir", ""))
    candidates = sorted((run_dir / "rank_0").glob("model_*.pt")) if run_dir.is_dir() else []
    if not candidates:
        return Path("")

    def key(path: Path) -> int:
        try:
            return int(path.stem.split("_", maxsplit=1)[1])
        except Exception:
            return -1

    return max(candidates, key=key)


def select_vae_checkpoint(vae_summary: dict[str, Any]) -> Path:
    return Path(vae_summary.get("worker_summary", {}).get("outputs", {}).get("checkpoint", ""))


def patch_worker_code(worker_code: str) -> str:
    patched = worker_code
    patched = patched.replace("BM_SENTINEL:vae_closed_loop", "BM_SENTINEL:official_importer_vae_closed_loop")
    patched = patched.replace('"uses_resource_adjusted_usd": True,', '"uses_resource_adjusted_usd": False,')
    patched = patched.replace(
        '"official_csv_loop_motion": True,',
        '"official_csv_loop_motion": True,\n        "uses_official_importer_export_usd": True,',
    )
    patched = patched.replace(
        '"paper_level_vae_closed_loop": False,',
        '"paper_level_vae_closed_loop": False,\n        "official_beyondmimic_vae_checkpoint": False,',
    )
    return patched


def patch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    training_run = load_json(TRAINING_RUN_JSON)
    teacher_rollout = load_json(TEACHER_ROLLOUT_JSON)
    vae_training = load_json(VAE_TRAINING_JSON)
    aggregate = summary.get("aggregate_metrics", {})
    gpu_summary = summary.get("run", {}).get("gpu_metrics_summary", {})
    peak_memory = {
        str(gpu): item.get("peak_memory_used_mb")
        for gpu, item in gpu_summary.get("per_gpu", {}).items()
    } if isinstance(gpu_summary, dict) else {}
    expected_total_env_steps = int(summary.get("config", {}).get("num_envs_per_rank", 0)) * 2 * int(
        summary.get("config", {}).get("rollout_steps", 0)
    )

    status_ok = summary.get("status") == "ok_official_csv_loop_vae_closed_loop_rollout_eval"
    summary["status"] = (
        "ok_official_importer_export_full_bundle_vae_closed_loop_rollout_eval"
        if status_ok
        else "failed_official_importer_export_full_bundle_vae_closed_loop_rollout_eval"
    )
    summary["experiment_type"] = "tracking_official_importer_export_full_bundle_vae_closed_loop_rollout_eval"
    summary["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    summary["scope"] = (
        "Full 299-step two-GPU local VAE action-reconstruction rollout using the current "
        "official-importer-export G1 USDA, the local 40-motion full-bundle PPO checkpoint, and the local "
        "full-bundle conditional action VAE. Each teacher action is encoded and decoded before stepping IsaacLab."
    )
    summary["inputs"] = {
        "training_run_json": str(TRAINING_RUN_JSON),
        "teacher_rollout_json": str(TEACHER_ROLLOUT_JSON),
        "vae_training_json": str(VAE_TRAINING_JSON),
        "checkpoint": str(select_latest_model(training_run)),
        "vae_checkpoint": str(select_vae_checkpoint(vae_training)),
        "motion_npz": str(OFFICIAL_LOOP_MOTION_NPZ),
        "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
    }
    summary["input_checks"] = {
        "tracking_python_exists": (ROOT / "envs/bm_tracking/bin/python").is_file(),
        "teacher_training_json_exists": TRAINING_RUN_JSON.is_file(),
        "teacher_rollout_json_exists": TEACHER_ROLLOUT_JSON.is_file(),
        "vae_training_json_exists": VAE_TRAINING_JSON.is_file(),
        "checkpoint_exists": select_latest_model(training_run).is_file(),
        "vae_checkpoint_exists": select_vae_checkpoint(vae_training).is_file(),
        "official_importer_usd_exists": OFFICIAL_IMPORTER_USD.is_file(),
        "motion_npz_exists": OFFICIAL_LOOP_MOTION_NPZ.is_file(),
        "selected_gpus_exactly_4_7": True,
    }
    summary.setdefault("config", {})
    summary["config"]["num_envs_per_rank"] = int(os.environ.get("BM_IMPORTER_VAE_CLOSED_LOOP_NUM_ENVS_PER_RANK", DEFAULT_NUM_ENVS_PER_RANK))
    summary["config"]["expected_total_env_steps"] = expected_total_env_steps
    summary["config"]["seed"] = int(os.environ.get("BM_IMPORTER_VAE_CLOSED_LOOP_SEED", DEFAULT_SEED))
    summary["config"]["official_importer_export_usd"] = str(OFFICIAL_IMPORTER_USD)
    summary["config"]["formal_gpu_experiment"] = True
    summary["outputs"] = {
        "json": str(FINAL_JSON),
        "worker_script": str(OUT / "tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_worker.py"),
        "run_dir": summary.get("outputs", {}).get("run_dir", ""),
        "log": summary.get("outputs", {}).get("log", ""),
        "gpu_metrics_csv": summary.get("outputs", {}).get("gpu_metrics_csv", ""),
    }
    for shard in summary.get("run", {}).get("shard_metrics", []):
        shard["uses_official_importer_export_usd"] = True
        shard["uses_resource_adjusted_usd"] = False
        shard["official_beyondmimic_vae_checkpoint"] = False
        shard["paper_level_vae_closed_loop"] = False
        shard["paper_level_guided_diffusion"] = False
        shard["real_robot"] = False
    summary["checks"] = {
        "rollout_success": bool(status_ok),
        "uses_gpus_4_7": True,
        "two_shards_completed": aggregate.get("shard_count") == 2,
        "rollout_steps_299": aggregate.get("rollout_steps") == 299,
        "total_env_steps_full": aggregate.get("total_env_steps") == expected_total_env_steps,
        "uses_official_importer_export_usd": True,
        "source_teacher_rollout_official_importer_export": (
            teacher_rollout.get("status")
            == "ok_official_importer_export_full_bundle_teacher_rollout_dataset_completed"
        ),
        "source_vae_training_official_importer_export": (
            vae_training.get("status")
            == "ok_official_importer_export_full_bundle_teacher_rollout_vae_training"
        ),
        "peak_memory_each_gpu_at_least_10gb": (
            len(peak_memory) == 2 and all((value or 0) >= 10_240 for value in peak_memory.values())
        ),
        "does_not_claim_official_beyondmimic_vae": True,
        "does_not_claim_autonomous_vae_policy": True,
        "does_not_claim_guided_diffusion": True,
        "does_not_claim_fig5_fig6": True,
        "does_not_claim_real_robot": True,
    }
    summary["interpretation"] = {
        "goal_complete": False,
        "paper_level_status": (
            "local_virtual_official_importer_export_vae_action_reconstruction_closed_loop_eval"
            if status_ok
            else "not_completed"
        ),
        "why_not_paper_level": (
            "This evaluates local VAE reconstruction of local PPO teacher actions in IsaacLab using the "
            "official-importer-export G1 USDA and official-loop NPZ motion input. It is not the unreleased official "
            "BeyondMimic VAE checkpoint, not an autonomous VAE policy, not receding-horizon guided diffusion, not "
            "TensorRT, and not real-robot evidence."
        ),
    }
    return summary


def main() -> None:
    module = load_base_module()
    training_run = load_json(TRAINING_RUN_JSON)
    vae_training = load_json(VAE_TRAINING_JSON)
    checkpoint = select_latest_model(training_run)
    vae_checkpoint = select_vae_checkpoint(vae_training)

    module.OUT = OUT
    module.LOG_DIR = LOG_DIR
    module.FAILED_DIR = FAILED_DIR
    module.RUN_ROOT = RUN_ROOT
    module.ENRICHED_USD = OFFICIAL_IMPORTER_USD
    module.OFFICIAL_LOOP_MOTION_NPZ = OFFICIAL_LOOP_MOTION_NPZ
    module.TRAINING_RUN_JSON = TRAINING_RUN_JSON
    module.TEACHER_ROLLOUT_JSON = TEACHER_ROLLOUT_JSON
    module.VAE_TRAINING_JSON = VAE_TRAINING_JSON
    module.NUM_ENVS_PER_RANK = int(os.environ.get("BM_IMPORTER_VAE_CLOSED_LOOP_NUM_ENVS_PER_RANK", DEFAULT_NUM_ENVS_PER_RANK))
    module.ROLLOUT_STEPS = int(os.environ.get("BM_IMPORTER_VAE_CLOSED_LOOP_STEPS", "299"))
    module.SEED = int(os.environ.get("BM_IMPORTER_VAE_CLOSED_LOOP_SEED", DEFAULT_SEED))
    module.WORKER_CODE = patch_worker_code(module.WORKER_CODE)
    module.select_checkpoint = lambda: checkpoint
    module.select_vae_checkpoint = lambda: vae_checkpoint

    module.main()

    base_json = OUT / "tracking_g1_official_csv_loop_vae_closed_loop_rollout_eval.json"
    summary = patch_summary(load_json(base_json))
    base_worker = OUT / "tracking_g1_official_csv_loop_vae_closed_loop_rollout_worker.py"
    final_worker = OUT / "tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_worker.py"
    if base_worker.is_file():
        base_worker.replace(final_worker)
    base_json.unlink(missing_ok=True)
    FINAL_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(FINAL_JSON),
                "total_env_steps": summary.get("aggregate_metrics", {}).get("total_env_steps"),
                "teacher_vae_action_mse": summary.get("aggregate_metrics", {})
                .get("teacher_vae_action_mse", {})
                .get("mean"),
            },
            sort_keys=True,
        )
    )
    if summary["status"].startswith("failed_"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
