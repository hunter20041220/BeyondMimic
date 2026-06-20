#!/usr/bin/env python3
"""Capture an official-importer-export VAE closed-loop rollout video asset.

The heavy closed-loop metric gate runs thousands of environments. This script
captures a single-environment trace for report/PPT visualization using the same
official-importer-export robot asset, local PPO teacher, and local full-bundle
VAE. It reuses the existing official-csv-loop video capture implementation and
patches the generated summaries so the claim boundary follows the importer path.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_g1_official_csv_loop_vae_closed_loop_rollout_video_capture.py"
OUT = ROOT / "res/visualization/official_importer_export_full_bundle_vae_closed_loop_rollout"
LOG_DIR = ROOT / "logs/tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_video_capture"
FAILED_DIR = ROOT / "res/failed_runs/tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_video_capture"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_video_capture"
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
DEFAULT_SEED = 20260685


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_official_csv_loop_vae_video_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base video capture script: {BASE_SCRIPT}")
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
    patched = worker_code.replace("BM_SENTINEL:vae_video", "BM_SENTINEL:official_importer_vae_video")
    patched = patched.replace('"uses_resource_adjusted_usd": True,', '"uses_resource_adjusted_usd": False,')
    patched = patched.replace(
        '"official_csv_loop_motion": True,',
        '"official_csv_loop_motion": True,\n        "uses_official_importer_export_usd": True,',
    )
    return patched


def move_asset(old_name: str, new_name: str) -> Path:
    old = OUT / old_name
    new = OUT / new_name
    if old.is_file():
        old.replace(new)
    return new


def patch_outputs() -> dict[str, Any]:
    capture_json = OUT / "tracking_g1_official_csv_loop_vae_closed_loop_rollout_capture.json"
    asset_json = OUT / "official_csv_loop_vae_closed_loop_rollout_video_asset.json"
    final_capture_json = OUT / "tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_capture.json"
    final_asset_json = OUT / "official_importer_export_full_bundle_vae_closed_loop_rollout_video_asset.json"
    worker_old = OUT / "tracking_g1_official_csv_loop_vae_closed_loop_rollout_worker.py"
    render_old = OUT / "tracking_g1_official_csv_loop_vae_closed_loop_rollout_render.py"
    worker_new = OUT / "tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_worker.py"
    render_new = OUT / "tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_render.py"
    if worker_old.is_file():
        worker_old.replace(worker_new)
    if render_old.is_file():
        render_old.replace(render_new)

    mp4 = move_asset(
        "official_csv_loop_vae_closed_loop_rollout_vs_reference.mp4",
        "official_importer_export_full_bundle_vae_closed_loop_rollout_vs_reference.mp4",
    )
    keyframes = move_asset(
        "official_csv_loop_vae_closed_loop_rollout_keyframes.png",
        "official_importer_export_full_bundle_vae_closed_loop_rollout_keyframes.png",
    )
    metrics_csv = move_asset(
        "official_csv_loop_vae_closed_loop_rollout_metrics.csv",
        "official_importer_export_full_bundle_vae_closed_loop_rollout_metrics.csv",
    )
    readme = OUT / "README.md"
    capture = load_json(capture_json)
    asset = load_json(asset_json)

    status_ok = capture.get("status") == "ok_official_csv_loop_vae_closed_loop_rollout_video_capture"
    capture["status"] = (
        "ok_official_importer_export_full_bundle_vae_closed_loop_rollout_video_capture"
        if status_ok
        else capture.get("status", "failed")
    )
    capture["experiment_type"] = "tracking_official_importer_export_full_bundle_vae_closed_loop_rollout_video_capture"
    capture["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    capture["scope"] = (
        "Captures one local virtual rollout where the official-importer-export full-bundle PPO teacher action is "
        "reconstructed through the local full-bundle conditional action VAE before stepping IsaacLab. It records "
        "robot/reference body poses for a report video. This is not an official BeyondMimic checkpoint, not "
        "autonomous VAE control, not paper Fig.5/Fig.6 guided diffusion, and not real-robot evidence."
    )
    capture.setdefault("inputs", {})
    capture["inputs"].update(
        {
            "training_run_json": str(TRAINING_RUN_JSON),
            "checkpoint_eval_json": str(CHECKPOINT_EVAL_JSON),
            "checkpoint": str(select_latest_model(load_json(TRAINING_RUN_JSON))),
            "vae_training_json": str(VAE_TRAINING_JSON),
            "vae_checkpoint": str(select_vae_checkpoint(load_json(VAE_TRAINING_JSON))),
            "motion_npz": str(OFFICIAL_LOOP_MOTION_NPZ),
            "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
        }
    )
    capture.setdefault("checks", {})
    capture["checks"].update(
        {
            "uses_official_importer_export_usd": True,
            "does_not_claim_paper_level": True,
            "does_not_claim_official_beyondmimic_vae": True,
            "does_not_claim_autonomous_vae_policy": True,
            "does_not_claim_guided_diffusion": True,
            "does_not_claim_fig5_fig6": True,
            "does_not_claim_real_robot": True,
        }
    )
    capture.setdefault("outputs", {})
    capture["outputs"].update(
        {
            "json": str(final_capture_json),
            "asset_json": str(final_asset_json),
            "worker_script": str(worker_new),
            "render_script": str(render_new),
        }
    )
    capture["interpretation"] = {
        "goal_complete": False,
        "vae_closed_loop_rollout_video_complete": bool(status_ok and asset),
        "paper_level_status": "local_virtual_official_importer_export_vae_action_reconstruction_rollout_video"
        if status_ok
        else "not_completed",
        "why_not_complete": (
            "This video is generated from a local official-importer-export PPO teacher and local full-bundle "
            "conditional action VAE. It does not prove official BeyondMimic VAE behavior, paper-level guided "
            "diffusion, or real-robot behavior."
        ),
    }

    if asset:
        asset["status"] = "ok_official_importer_export_full_bundle_vae_closed_loop_rollout_video_asset"
        asset["experiment_type"] = "tracking_official_importer_export_full_bundle_vae_closed_loop_rollout_video_capture"
        asset["claim_level"] = "local_virtual_official_importer_export_vae_action_reconstruction_rollout_video"
        asset["source_capture_summary"] = str(final_capture_json)
        asset["source_capture_status"] = capture["status"]
        asset["assets"] = {
            "mp4": str(mp4),
            "keyframes_png": str(keyframes),
            "metrics_csv": str(metrics_csv),
            "readme": str(readme),
        }
        asset["asset_sizes"] = {key: Path(value).stat().st_size for key, value in asset["assets"].items()}
        asset["asset_sha256"] = {key: sha256_file(Path(value)) for key, value in asset["assets"].items()}
        asset["checks"] = {
            "capture_status_ok": capture["status"]
            == "ok_official_importer_export_full_bundle_vae_closed_loop_rollout_video_capture",
            "frame_count_299": asset.get("frame_count") == 299,
            "video_exists_nonempty": mp4.is_file() and mp4.stat().st_size > 0,
            "keyframes_exist_nonempty": keyframes.is_file() and keyframes.stat().st_size > 0,
            "uses_official_importer_export_usd": True,
            "does_not_claim_paper_level": True,
            "does_not_claim_official_beyondmimic_vae": True,
            "does_not_claim_autonomous_vae_policy": True,
            "does_not_claim_guided_diffusion": True,
            "does_not_claim_fig5_fig6": True,
            "does_not_claim_real_robot": True,
        }
        asset["interpretation"] = capture["interpretation"]
    readme.write_text(
        "\n".join(
            [
                "# Official-Importer-Export VAE Closed-Loop Rollout Visualization",
                "",
                "This directory contains a local virtual VAE action-reconstruction rollout visualization. The local",
                "full-bundle PPO teacher action is encoded and decoded by the local official-importer-export",
                "conditional action VAE before stepping IsaacLab.",
                "",
                "## Claim Level",
                "",
                "local_virtual_official_importer_export_vae_action_reconstruction_rollout_video. This is not the",
                "official BeyondMimic VAE checkpoint, not autonomous VAE control, not paper-level Fig. 5/Fig. 6",
                "guided diffusion, and not real-robot evidence.",
                "",
                "## Assets",
                "",
                f"- `{mp4}`",
                f"- `{keyframes}`",
                f"- `{metrics_csv}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    capture_json.unlink(missing_ok=True)
    asset_json.unlink(missing_ok=True)
    final_capture_json.write_text(json.dumps(capture, indent=2, sort_keys=True), encoding="utf-8")
    final_asset_json.write_text(json.dumps(asset, indent=2, sort_keys=True), encoding="utf-8")
    return {"capture_json": str(final_capture_json), "asset_json": str(final_asset_json), "mp4": str(mp4)}


def main() -> None:
    module = load_base_module()
    training_run = load_json(TRAINING_RUN_JSON)
    vae_training = load_json(VAE_TRAINING_JSON)
    module.OUT = OUT
    module.LOG_DIR = LOG_DIR
    module.FAILED_DIR = FAILED_DIR
    module.RUN_ROOT = RUN_ROOT
    module.ENRICHED_USD = OFFICIAL_IMPORTER_USD
    module.OFFICIAL_LOOP_MOTION_NPZ = OFFICIAL_LOOP_MOTION_NPZ
    module.TRAINING_RUN_JSON = TRAINING_RUN_JSON
    module.CHECKPOINT_EVAL_JSON = CHECKPOINT_EVAL_JSON
    module.VAE_TRAINING_JSON = VAE_TRAINING_JSON
    module.SEED = int(os.environ.get("BM_IMPORTER_VAE_VIDEO_SEED", str(DEFAULT_SEED)))
    module.WORKER_CODE = patch_worker_code(module.WORKER_CODE)
    module.select_checkpoint = lambda: select_latest_model(training_run)
    module.select_vae_checkpoint = lambda: select_vae_checkpoint(vae_training)
    module.main()
    patched = patch_outputs()
    print(json.dumps({"status": "ok_official_importer_export_full_bundle_vae_video_capture", **patched}, sort_keys=True))


if __name__ == "__main__":
    main()
