#!/usr/bin/env python3
"""Run official-importer-export task-conditioned latent-guidance rollouts.

This wrapper reuses the validated official-csv-loop task-conditioned runner and
redirects the underlying receding-latent guidance path to the current local
official-importer-export G1 USDA, PPO teacher, VAE, denoiser, and offline
guidance summary. It remains local virtual evidence, not official BeyondMimic
Fig. 5/Fig. 6 or real-robot evidence.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tracking_g1_official_csv_loop_receding_latent_guidance_rollout_eval as base_receding
import tracking_g1_official_csv_loop_task_conditioned_latent_guidance_rollout_eval as base_task


ROOT = Path("/mnt/infini-data/test/BeyondMimic")


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default)))


OUT_ROOT = env_path(
    "BM_TASK_CONDITIONED_OUT_ROOT",
    ROOT / "res/visualization/official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout",
)
SUMMARY_ROOT = env_path(
    "BM_TASK_CONDITIONED_SUMMARY_ROOT",
    ROOT / "res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval",
)
SUMMARY_JSON = env_path(
    "BM_TASK_CONDITIONED_SUMMARY_JSON",
    SUMMARY_ROOT / "level_c_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval.json",
)
SUMMARY_TSV = env_path(
    "BM_TASK_CONDITIONED_SUMMARY_TSV",
    SUMMARY_ROOT / "level_c_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval.tsv",
)
LOG_ROOT = env_path(
    "BM_TASK_CONDITIONED_LOG_ROOT",
    ROOT / "logs/tracking_g1_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval",
)
FAILED_ROOT = env_path(
    "BM_TASK_CONDITIONED_FAILED_ROOT",
    ROOT / "res/failed_runs/tracking_g1_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval",
)
RUN_ROOT = env_path(
    "BM_TASK_CONDITIONED_RUN_ROOT",
    ROOT / "res/runs/tracking_g1_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval",
)
OFFICIAL_IMPORTER_USD = (
    ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda"
)
FULL_BUNDLE_MOTION_NPZ = (
    ROOT / "res/tracking/official_csv_loop_full_bundle_motion_npz/official_csv_loop_full_public_motion_bundle.npz"
)
FULL_BUNDLE_AUDIT = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_motion_npz/"
    "tracking_g1_official_csv_loop_full_bundle_motion_npz.json"
)
TRAINING_RUN_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_full_bundle_ppo_training_run/"
    "tracking_g1_official_importer_export_full_bundle_ppo_training_run.json"
)
CHECKPOINT_EVAL_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_full_bundle_ppo_checkpoint_eval/"
    "tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval.json"
)
VAE_TRAINING_JSON = (
    ROOT
    / "res/level_c/official_importer_export_full_bundle_teacher_rollout_vae_training/"
    "level_c_official_importer_export_full_bundle_teacher_rollout_vae_training.json"
)
DIFFUSION_JSON = (
    ROOT
    / "res/level_c/official_importer_export_full_bundle_state_latent_diffusion_training/"
    "level_c_official_importer_export_full_bundle_state_latent_diffusion_training.json"
)
GUIDANCE_JSON = (
    ROOT
    / "res/level_c/official_importer_export_full_bundle_state_latent_guidance_eval/"
    "level_c_official_importer_export_full_bundle_state_latent_guidance_eval.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def patch_receding_worker_code(code: str) -> str:
    patched = code
    patched = patched.replace("BM_SENTINEL:receding_latent_guidance", "BM_SENTINEL:official_importer_receding_latent_guidance")
    patched = patched.replace("official_csv_loop_receding_latent_guidance_rollout", "official_importer_export_receding_latent_guidance_rollout")
    patched = patched.replace('"uses_resource_adjusted_usd": True,', '"uses_resource_adjusted_usd": False,')
    patched = patched.replace(
        '"official_csv_loop_motion": True,',
        '"official_csv_loop_motion": True,\n        "uses_official_importer_export_usd": True,',
    )
    patched = patched.replace(
        '"paper_level_guidance_rollout": False,',
        '"paper_level_guidance_rollout": False,\n        "official_beyondmimic_diffusion_checkpoint": False,',
    )
    return patched


def patch_modules() -> None:
    base_receding.OUT = OUT_ROOT
    base_receding.ENRICHED_USD = OFFICIAL_IMPORTER_USD
    base_receding.OFFICIAL_LOOP_MOTION_NPZ = FULL_BUNDLE_MOTION_NPZ
    base_receding.TRAINING_RUN_JSON = TRAINING_RUN_JSON
    base_receding.CHECKPOINT_EVAL_JSON = CHECKPOINT_EVAL_JSON
    base_receding.VAE_TRAINING_JSON = VAE_TRAINING_JSON
    base_receding.DIFFUSION_JSON = DIFFUSION_JSON
    base_receding.GUIDANCE_JSON = GUIDANCE_JSON
    base_receding.WORKER_CODE = patch_receding_worker_code(base_receding.WORKER_CODE)

    def patched_import_base_module():
        return base_receding

    base_task.import_base_module = patched_import_base_module
    base_task.OUT_ROOT = OUT_ROOT
    base_task.SUMMARY_ROOT = SUMMARY_ROOT
    base_task.SUMMARY_JSON = SUMMARY_JSON
    base_task.SUMMARY_TSV = SUMMARY_TSV
    base_task.LOG_ROOT = LOG_ROOT
    base_task.FAILED_ROOT = FAILED_ROOT
    base_task.RUN_ROOT = RUN_ROOT
    task_seeds = {
        "joystick": 20260689,
        "waypoint": 20260690,
        "obstacle_avoidance": 20260691,
        "composed": 20260692,
    }
    if os.environ.get("BM_TASK_CONDITIONED_TASK_SEEDS_JSON"):
        task_seeds.update(
            {
                key: int(value)
                for key, value in json.loads(os.environ["BM_TASK_CONDITIONED_TASK_SEEDS_JSON"]).items()
            }
        )
    base_task.DEFAULT_TASK_SEEDS.update(task_seeds)
    base_task.TASK_SEEDS.update(task_seeds)


def patch_summary_boundaries() -> None:
    summary = load_json(SUMMARY_JSON)
    if not summary:
        raise FileNotFoundError(SUMMARY_JSON)
    bundle = load_json(FULL_BUNDLE_AUDIT)
    rows = summary.get("rows", [])
    ok = summary.get("status") == "ok_official_csv_loop_task_conditioned_latent_guidance_rollout_eval"
    summary["status"] = (
        "ok_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval"
        if ok
        else f"failed_official_importer_export_task_conditioned_from_{summary.get('status')}"
    )
    summary["experiment_type"] = "tracking_g1_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval"
    summary["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    summary["scope"] = (
        "Aggregates local closed-loop task-conditioned latent-guidance rollouts for joystick, waypoint, "
        "obstacle-avoidance, and composed proxy objectives in IsaacLab using the official-importer-export G1 USDA "
        "and the local official-importer-export PPO/VAE/denoiser chain."
    )
    summary["bundle"] = {
        "motion_count": bundle.get("bundle", {}).get("motion_count"),
        "total_frames": bundle.get("bundle", {}).get("total_frames"),
        "fps": bundle.get("bundle", {}).get("fps"),
        "clip_boundary_count": len(bundle.get("boundary_jumps", [])),
    }
    summary["inputs"] = {
        "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
        "full_bundle_motion_audit": str(FULL_BUNDLE_AUDIT),
        "motion_npz": str(FULL_BUNDLE_MOTION_NPZ),
        "training_run_json": str(TRAINING_RUN_JSON),
        "checkpoint_eval_json": str(CHECKPOINT_EVAL_JSON),
        "vae_training_json": str(VAE_TRAINING_JSON),
        "diffusion_json": str(DIFFUSION_JSON),
        "guidance_json": str(GUIDANCE_JSON),
    }
    summary["checks"]["uses_full_public_motion_bundle"] = bundle.get("status") == "ok_official_csv_loop_full_bundle_motion_npz"
    summary["checks"]["full_bundle_motion_count_40"] = summary["bundle"]["motion_count"] == 40
    summary["checks"]["uses_official_importer_export_usd"] = OFFICIAL_IMPORTER_USD.is_file()
    summary["checks"]["all_tasks_have_mp4_paths"] = all(bool(row.get("mp4")) for row in rows)
    summary["checks"]["does_not_claim_official_checkpoint"] = True
    summary["checks"]["does_not_claim_fig5_fig6"] = True
    summary["checks"]["does_not_claim_real_robot"] = True
    summary["interpretation"] = {
        "goal_complete": False,
        "paper_level_status": "local_virtual_official_importer_export_task_conditioned_latent_guidance_rollout",
        "why_not_complete": (
            "These are local virtual task-conditioned guidance rollouts over the public 40-motion bundle. They use "
            "local PPO/VAE/denoiser checkpoints and proxy costs. They are not official BeyondMimic checkpoints, not "
            "unpatched official rollouts, not paper Fig. 5/Fig. 6 success-rate reproduction, not TensorRT deployment, "
            "and not real robot evidence."
        ),
    }
    write_json(SUMMARY_JSON, summary)

    for row in rows:
        row_summary_path = Path(row.get("summary_json", ""))
        row_summary = load_json(row_summary_path)
        if row_summary:
            row_summary["status"] = (
                "ok_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval"
                if row_summary.get("status") == "ok_official_csv_loop_task_conditioned_latent_guidance_rollout_eval"
                else row_summary.get("status")
            )
            row_summary["experiment_type"] = (
                "tracking_g1_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval"
            )
            row_summary["bundle"] = summary["bundle"]
            row_summary.setdefault("checks", {})["uses_official_importer_export_usd"] = True
            row_summary.setdefault("checks", {})["uses_resource_adjusted_usd"] = False
            row_summary.setdefault("checks", {})["does_not_claim_official_checkpoint"] = True
            row_summary["interpretation"] = summary["interpretation"]
            write_json(row_summary_path, row_summary)

        asset_path = Path(row.get("asset_json", ""))
        asset = load_json(asset_path)
        if asset:
            asset["claim_level"] = "local_virtual_official_importer_export_task_conditioned_latent_guidance_rollout"
            asset["bundle"] = summary["bundle"]
            asset.setdefault("checks", {})["uses_official_importer_export_usd"] = True
            asset.setdefault("checks", {})["does_not_claim_official_checkpoint"] = True
            asset["interpretation"] = summary["interpretation"]
            write_json(asset_path, asset)


def main() -> None:
    if not GUIDANCE_JSON.is_file():
        raise FileNotFoundError(f"Run importer-export offline guidance first: {GUIDANCE_JSON}")
    patch_modules()
    base_task.main()
    patch_summary_boundaries()
    final = load_json(SUMMARY_JSON)
    print(json.dumps({"status": final.get("status"), "json": str(SUMMARY_JSON)}, sort_keys=True))
    if final.get("status") != "ok_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
