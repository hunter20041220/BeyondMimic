#!/usr/bin/env python3
"""Run full-bundle local receding-latent guidance through the proven rollout path.

The single-motion receding-latent rollout script already contains the heavy
IsaacLab worker and renderer. This wrapper redirects that audited path to the
40-motion official-csv-loop public bundle and the corresponding local
full-bundle PPO/VAE/denoiser artifacts, then rewrites the summary boundary so
the result is tracked as a distinct full-bundle virtual evidence artifact.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tracking_g1_official_csv_loop_receding_latent_guidance_rollout_eval as base


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/visualization/official_csv_loop_full_bundle_receding_latent_guidance_rollout"
SUMMARY_JSON = (
    ROOT
    / "res/level_c/official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval/"
    "level_c_official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval.json"
)
SUMMARY_TSV = (
    ROOT
    / "res/level_c/official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval/"
    "level_c_official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval.tsv"
)
LOG_DIR = ROOT / "logs/tracking_g1_official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval"
FAILED_DIR = ROOT / "res/failed_runs/tracking_g1_official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval"
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
SEED = 20260680


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def patch_base_module() -> None:
    base.OUT = OUT
    base.SUMMARY_JSON = SUMMARY_JSON
    base.SUMMARY_TSV = SUMMARY_TSV
    base.LOG_DIR = LOG_DIR
    base.FAILED_DIR = FAILED_DIR
    base.RUN_ROOT = RUN_ROOT
    base.OFFICIAL_LOOP_MOTION_NPZ = FULL_BUNDLE_MOTION_NPZ
    base.TRAINING_RUN_JSON = TRAINING_RUN_JSON
    base.CHECKPOINT_EVAL_JSON = CHECKPOINT_EVAL_JSON
    base.VAE_TRAINING_JSON = VAE_TRAINING_JSON
    base.DIFFUSION_JSON = DIFFUSION_JSON
    base.GUIDANCE_JSON = GUIDANCE_JSON
    base.SEED = SEED


def patch_outputs() -> None:
    summary = load_json(SUMMARY_JSON)
    if not summary:
        raise FileNotFoundError(SUMMARY_JSON)
    asset_json = Path(summary.get("outputs", {}).get("asset_json", ""))
    asset = load_json(asset_json)
    bundle = load_json(FULL_BUNDLE_AUDIT)
    capture_ok = bool(summary.get("checks", {}).get("capture_ok"))
    render_ok = bool(summary.get("checks", {}).get("render_ok"))
    full_status = (
        "ok_official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval"
        if capture_ok and render_ok
        else f"failed_full_bundle_receding_latent_guidance_rollout_from_{summary.get('status')}"
    )

    summary["status"] = full_status
    summary["experiment_type"] = "tracking_g1_official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval"
    summary["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    summary["scope"] = (
        "Runs teacher, VAE-base, denoised-latent, and receding-horizon guided-latent variants in the local "
        "resource-adjusted IsaacLab task using the 40-motion official-csv-loop public bundle and the corresponding "
        "local full-bundle PPO/VAE/denoiser artifacts. This is stronger full-bundle virtual evidence, not official "
        "BeyondMimic Fig. 5/Fig. 6 evidence."
    )
    summary["inputs"]["full_bundle_motion_audit"] = str(FULL_BUNDLE_AUDIT)
    summary["inputs"]["motion_npz"] = str(FULL_BUNDLE_MOTION_NPZ)
    summary["inputs"]["training_run_json"] = str(TRAINING_RUN_JSON)
    summary["inputs"]["checkpoint_eval_json"] = str(CHECKPOINT_EVAL_JSON)
    summary["inputs"]["vae_training_json"] = str(VAE_TRAINING_JSON)
    summary["inputs"]["diffusion_json"] = str(DIFFUSION_JSON)
    summary["inputs"]["guidance_json"] = str(GUIDANCE_JSON)
    summary["bundle"] = {
        "motion_count": bundle.get("bundle", {}).get("motion_count"),
        "total_frames": bundle.get("bundle", {}).get("total_frames"),
        "fps": bundle.get("bundle", {}).get("fps"),
        "clip_boundary_count": len(bundle.get("boundary_jumps", [])),
    }
    summary["input_checks"]["full_bundle_motion_npz_exists"] = FULL_BUNDLE_MOTION_NPZ.is_file()
    summary["input_checks"]["full_bundle_audit_ok"] = bundle.get("status") == "ok_official_csv_loop_full_bundle_motion_npz"
    summary["checks"]["uses_full_public_motion_bundle"] = summary["input_checks"]["full_bundle_audit_ok"]
    summary["checks"]["full_bundle_motion_count_40"] = summary["bundle"]["motion_count"] == 40
    summary["checks"]["does_not_claim_official_checkpoint"] = True
    summary["interpretation"] = {
        "goal_complete": False,
        "paper_level_status": (
            "local_virtual_full_bundle_receding_horizon_latent_guidance_rollout"
            if capture_ok and render_ok
            else "not_completed"
        ),
        "why_not_complete": (
            "This is a local virtual closed-loop guidance rollout using full-bundle public motions and local "
            "resource-adjusted PPO/VAE/denoiser checkpoints. It is not the official BeyondMimic diffusion checkpoint, "
            "not unpatched official replay, not paper Fig. 5/Fig. 6 task reproduction, not TensorRT deployment, and "
            "not real-robot evidence."
        ),
    }
    write_json(SUMMARY_JSON, summary)

    if asset:
        asset["experiment_type"] = (
            "tracking_g1_official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval_assets"
        )
        asset["claim_level"] = "local_virtual_full_bundle_receding_horizon_latent_guidance_rollout"
        asset["source_metrics"] = summary.get("run", {}).get("metrics_json", asset.get("source_metrics", ""))
        asset["bundle"] = summary["bundle"]
        asset["checks"]["does_not_claim_official_checkpoint"] = True
        asset["interpretation"] = summary["interpretation"]
        write_json(asset_json, asset)


def main() -> None:
    patch_base_module()
    base.main()
    patch_outputs()
    final = load_json(SUMMARY_JSON)
    print(json.dumps({"status": final.get("status"), "json": str(SUMMARY_JSON)}, sort_keys=True))
    if not str(final.get("status", "")).startswith("ok_"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
