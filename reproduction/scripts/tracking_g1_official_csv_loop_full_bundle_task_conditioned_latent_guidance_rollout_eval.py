#!/usr/bin/env python3
"""Run full-bundle task-conditioned receding latent-guidance rollouts.

This wrapper keeps the validated task-conditioned runner but redirects the
underlying receding-latent guidance path to the 40-motion official-csv-loop
public bundle and corresponding local full-bundle PPO/VAE/denoiser artifacts.
It remains local virtual evidence, not official BeyondMimic Fig. 5/Fig. 6.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tracking_g1_official_csv_loop_receding_latent_guidance_rollout_eval as base_receding
import tracking_g1_official_csv_loop_task_conditioned_latent_guidance_rollout_eval as base_task


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT_ROOT = ROOT / "res/visualization/official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout"
SUMMARY_ROOT = ROOT / "res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval"
SUMMARY_JSON = SUMMARY_ROOT / "level_c_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval.json"
SUMMARY_TSV = SUMMARY_ROOT / "level_c_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval.tsv"
LOG_ROOT = ROOT / "logs/tracking_g1_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval"
FAILED_ROOT = ROOT / "res/failed_runs/tracking_g1_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval"
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
    / "res/tracking/g1_official_csv_loop_full_bundle_ppo_training_run/"
    "tracking_g1_official_csv_loop_full_bundle_ppo_training_run.json"
)
CHECKPOINT_EVAL_JSON = (
    ROOT
    / "res/tracking/g1_official_csv_loop_full_bundle_ppo_checkpoint_eval/"
    "tracking_g1_official_csv_loop_full_bundle_ppo_checkpoint_eval.json"
)
VAE_TRAINING_JSON = (
    ROOT
    / "res/level_c/official_csv_loop_full_bundle_teacher_rollout_vae_training/"
    "level_c_official_csv_loop_full_bundle_teacher_rollout_vae_training.json"
)
DIFFUSION_JSON = (
    ROOT
    / "res/level_c/official_csv_loop_full_bundle_state_latent_diffusion_training/"
    "level_c_official_csv_loop_full_bundle_state_latent_diffusion_training.json"
)
GUIDANCE_JSON = (
    ROOT
    / "res/level_c/official_csv_loop_full_bundle_state_latent_guidance_eval/"
    "level_c_official_csv_loop_full_bundle_state_latent_guidance_eval.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def patch_modules() -> None:
    base_receding.OFFICIAL_LOOP_MOTION_NPZ = FULL_BUNDLE_MOTION_NPZ
    base_receding.TRAINING_RUN_JSON = TRAINING_RUN_JSON
    base_receding.CHECKPOINT_EVAL_JSON = CHECKPOINT_EVAL_JSON
    base_receding.VAE_TRAINING_JSON = VAE_TRAINING_JSON
    base_receding.DIFFUSION_JSON = DIFFUSION_JSON
    base_receding.GUIDANCE_JSON = GUIDANCE_JSON

    # The task runner imports the receding runner through importlib. Repoint it
    # to the already patched module by replacing the importer.
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
    base_task.DEFAULT_TASK_SEEDS.update(
        {
            "joystick": 20260681,
            "waypoint": 20260682,
            "obstacle_avoidance": 20260683,
            "composed": 20260684,
        }
    )
    base_task.TASK_SEEDS.update(base_task.DEFAULT_TASK_SEEDS)


def patch_summary_boundaries() -> None:
    summary = load_json(SUMMARY_JSON)
    if not summary:
        raise FileNotFoundError(SUMMARY_JSON)
    bundle = load_json(FULL_BUNDLE_AUDIT)
    rows = summary.get("rows", [])
    ok = summary.get("status") == "ok_official_csv_loop_task_conditioned_latent_guidance_rollout_eval"
    summary["status"] = (
        "ok_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval"
        if ok
        else f"failed_full_bundle_task_conditioned_from_{summary.get('status')}"
    )
    summary["experiment_type"] = "tracking_g1_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval"
    summary["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    summary["scope"] = (
        "Aggregates full-bundle local closed-loop task-conditioned latent-guidance rollouts for joystick, waypoint, "
        "obstacle-avoidance, and composed proxy objectives in IsaacLab. The runs use the 40-motion official-csv-loop "
        "public bundle and local full-bundle PPO/VAE/denoiser artifacts."
    )
    summary["bundle"] = {
        "motion_count": bundle.get("bundle", {}).get("motion_count"),
        "total_frames": bundle.get("bundle", {}).get("total_frames"),
        "fps": bundle.get("bundle", {}).get("fps"),
        "clip_boundary_count": len(bundle.get("boundary_jumps", [])),
    }
    summary["inputs"] = {
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
    summary["checks"]["all_tasks_have_mp4_paths"] = all(bool(row.get("mp4")) for row in rows)
    summary["checks"]["does_not_claim_official_checkpoint"] = True
    summary["interpretation"] = {
        "goal_complete": False,
        "paper_level_status": "local_virtual_full_bundle_task_conditioned_latent_guidance_rollout",
        "why_not_complete": (
            "These are closed-loop local virtual task-conditioned guidance rollouts over the 40-motion public "
            "full-bundle chain. They are not official BeyondMimic checkpoints, not unpatched official rollouts, "
            "not paper Fig. 5/Fig. 6 success-rate reproduction, not TensorRT deployment, and not real robot evidence."
        ),
    }
    write_json(SUMMARY_JSON, summary)

    for row in rows:
        row_summary_path = Path(row.get("summary_json", ""))
        row_summary = load_json(row_summary_path)
        if not row_summary:
            continue
        row_summary["status"] = (
            "ok_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval"
            if row_summary.get("status") == "ok_official_csv_loop_task_conditioned_latent_guidance_rollout_eval"
            else row_summary.get("status")
        )
        row_summary["experiment_type"] = (
            "tracking_g1_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval"
        )
        row_summary["bundle"] = summary["bundle"]
        row_summary.setdefault("checks", {})["uses_full_public_motion_bundle"] = True
        row_summary.setdefault("checks", {})["does_not_claim_official_checkpoint"] = True
        row_summary["interpretation"] = summary["interpretation"]
        write_json(row_summary_path, row_summary)

        asset_path = Path(row.get("asset_json", ""))
        asset = load_json(asset_path)
        if asset:
            asset["claim_level"] = "local_virtual_full_bundle_task_conditioned_latent_guidance_rollout"
            asset["bundle"] = summary["bundle"]
            asset.setdefault("checks", {})["uses_full_public_motion_bundle"] = True
            asset.setdefault("checks", {})["does_not_claim_official_checkpoint"] = True
            asset["interpretation"] = summary["interpretation"]
            write_json(asset_path, asset)


def main() -> None:
    patch_modules()
    base_task.main()
    patch_summary_boundaries()
    final = load_json(SUMMARY_JSON)
    print(json.dumps({"status": final.get("status"), "json": str(SUMMARY_JSON)}, sort_keys=True))
    if not str(final.get("status", "")).startswith("ok_"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
