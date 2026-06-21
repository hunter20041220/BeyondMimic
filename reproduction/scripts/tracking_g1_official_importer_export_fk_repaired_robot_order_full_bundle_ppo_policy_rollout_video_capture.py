#!/usr/bin/env python3
"""Capture a robot-order FK-repaired PPO policy rollout video.

This wrapper reuses the one-environment policy rollout capture/render path and
points it at the current strongest local virtual tracking baseline: the
official-importer-export G1 USDA, the robot-order FK-repaired full public motion
bundle, and the robot-order FK-repaired PPO checkpoint. The video is report/PPT
media only, not a paper-level BeyondMimic rollout.
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
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_g1_official_csv_loop_policy_rollout_video_capture.py"
OUT = ROOT / "res/visualization/official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout"
LOG_DIR = (
    ROOT
    / "logs/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout_video_capture"
)
FAILED_DIR = (
    ROOT
    / "res/failed_runs/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout_video_capture"
)
RUN_ROOT = (
    ROOT
    / "res/runs/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout_video_capture"
)
OFFICIAL_IMPORTER_USD = (
    ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda"
)
ROBOT_ORDER_MOTION_NPZ = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/"
    "official_csv_loop_full_public_motion_bundle_fk_repaired_robot_order.npz"
)
ROBOT_ORDER_BUNDLE_AUDIT = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/"
    "tracking_g1_official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz.json"
)
TRAINING_RUN_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run.json"
)
CHECKPOINT_EVAL_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval.json"
)
DEFAULT_SEED = 20260722


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_policy_video_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base policy video script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_worker_code(worker_code: str) -> str:
    return (
        worker_code.replace("BM_SENTINEL:policy_video", "BM_SENTINEL:robot_order_fk_policy_video")
        .replace('"uses_resource_adjusted_usd": True,', '"uses_resource_adjusted_usd": False,')
        .replace(
            '"official_csv_loop_motion": True,',
            '"official_csv_loop_motion": True,\n        "uses_official_importer_export_usd": True,\n        '
            '"uses_robot_order_fk_repaired_full_public_motion_bundle": True,\n        '
            '"uses_full_public_motion_bundle": True,',
        )
    )


def patch_render_code(render_code: str) -> str:
    return (
        render_code.replace(
            "Official-Loop Policy Rollout Visualization",
            "Robot-Order FK-Repaired PPO Policy Rollout Visualization",
        )
        .replace(
            "official csv-loop PPO checkpoint",
            "robot-order FK-repaired official-importer-export PPO checkpoint",
        )
        .replace(
            "local_virtual_resource_adjusted_policy_rollout_video",
            "local_virtual_official_importer_export_robot_order_fk_repaired_ppo_policy_rollout_video",
        )
        .replace(
            "resource-adjusted official csv-loop PPO checkpoint",
            "robot-order FK-repaired official-importer-export PPO checkpoint",
        )
    )


def move_if_exists(old_name: str, new_name: str) -> Path:
    old = OUT / old_name
    new = OUT / new_name
    if old.is_file():
        old.replace(new)
    return new


def patch_outputs() -> dict[str, str]:
    base_capture_json = OUT / "tracking_g1_official_csv_loop_policy_rollout_capture.json"
    base_asset_json = OUT / "official_csv_loop_policy_rollout_video_asset.json"
    final_capture_json = (
        OUT / "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout_capture.json"
    )
    final_asset_json = (
        OUT / "official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout_video_asset.json"
    )
    worker_old = OUT / "tracking_g1_official_csv_loop_policy_rollout_worker.py"
    render_old = OUT / "tracking_g1_official_csv_loop_policy_rollout_render.py"
    worker_new = (
        OUT
        / "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout_worker.py"
    )
    render_new = (
        OUT
        / "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout_render.py"
    )
    if worker_old.is_file():
        worker_old.replace(worker_new)
    if render_old.is_file():
        render_old.replace(render_new)

    mp4 = move_if_exists(
        "official_csv_loop_policy_rollout_vs_reference.mp4",
        "official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout_vs_reference.mp4",
    )
    keyframes = move_if_exists(
        "official_csv_loop_policy_rollout_keyframes.png",
        "official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout_keyframes.png",
    )
    metrics_csv = move_if_exists(
        "official_csv_loop_policy_rollout_metrics.csv",
        "official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout_metrics.csv",
    )
    readme = OUT / "README.md"

    capture = load_json(base_capture_json)
    asset = load_json(base_asset_json)
    training = load_json(TRAINING_RUN_JSON)
    checkpoint_eval = load_json(CHECKPOINT_EVAL_JSON)
    bundle = load_json(ROBOT_ORDER_BUNDLE_AUDIT)
    bundle_info = bundle.get("metrics", {})
    base_capture_ok = capture.get("status") == "ok_official_csv_loop_policy_rollout_video_capture"
    base_asset_ok = asset.get("status") == "ok"
    final_status = (
        "ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout_video_capture"
        if base_capture_ok and base_asset_ok
        else f"failed_robot_order_fk_policy_rollout_from_{capture.get('status')}"
    )

    capture["status"] = final_status
    capture["experiment_type"] = (
        "tracking_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout_video_capture"
    )
    capture["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    capture["scope"] = (
        "Captures one local virtual policy-vs-reference rollout from the current robot-order FK-repaired PPO "
        "baseline. This is report media and not official paper-level tracking, Fig. 5/Fig. 6 guided diffusion, "
        "TensorRT deployment, or real-robot evidence."
    )
    capture.setdefault("inputs", {})
    capture["inputs"].update(
        {
            "training_run_json": str(TRAINING_RUN_JSON),
            "checkpoint_eval_json": str(CHECKPOINT_EVAL_JSON),
            "checkpoint": checkpoint_eval.get("inputs", {}).get("checkpoint"),
            "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
            "motion_npz": str(ROBOT_ORDER_MOTION_NPZ),
            "robot_order_bundle_audit": str(ROBOT_ORDER_BUNDLE_AUDIT),
        }
    )
    capture.setdefault("input_checks", {})
    capture["input_checks"].update(
        {
            "official_importer_usd_exists": OFFICIAL_IMPORTER_USD.is_file(),
            "robot_order_motion_npz_exists": ROBOT_ORDER_MOTION_NPZ.is_file(),
            "robot_order_bundle_audit_ok": bundle.get("status") == "ok_fk_repaired_robot_order_motion_npz",
            "robot_order_motion_count_40": bundle_info.get("motion_count") == 40,
            "robot_order_training_completed": training.get("status")
            == "ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_completed",
            "robot_order_checkpoint_eval_completed": checkpoint_eval.get("status")
            == "ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_completed",
        }
    )
    capture["bundle"] = {
        "motion_count": bundle_info.get("motion_count"),
        "total_frames": bundle_info.get("total_frames"),
        "fps": bundle_info.get("fps"),
        "npz_sha256": bundle_info.get("npz_sha256"),
    }
    capture.setdefault("checks", {})
    capture["checks"].update(
        {
            "capture_status_ok": final_status
            == "ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout_video_capture",
            "uses_official_importer_export_usd": True,
            "uses_robot_order_fk_repaired_full_public_motion_bundle": True,
            "uses_iteration_999_checkpoint": "model_999.pt"
            in str(checkpoint_eval.get("inputs", {}).get("checkpoint", "")),
            "does_not_claim_official_checkpoint": True,
            "does_not_claim_paper_level": True,
            "does_not_claim_fig5_fig6": True,
            "does_not_claim_tensorrt": True,
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
        "policy_rollout_video_complete": capture["checks"]["capture_status_ok"],
        "paper_level_status": (
            "local_virtual_official_importer_export_robot_order_fk_repaired_ppo_policy_rollout_video"
            if base_capture_ok and base_asset_ok
            else "not_completed"
        ),
        "why_not_complete": (
            "The video uses a local robot-order FK-repaired PPO checkpoint and local official-importer-export USDA. "
            "It is useful report evidence for the current best virtual tracking baseline, but it is not an official "
            "BeyondMimic teacher checkpoint, not paper-scale tracking performance, not guided diffusion, and not real robot."
        ),
    }

    if asset:
        asset["status"] = (
            "ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout_video_asset"
        )
        asset["experiment_type"] = capture["experiment_type"]
        asset["claim_level"] = (
            "local_virtual_official_importer_export_robot_order_fk_repaired_ppo_policy_rollout_video"
        )
        asset["source_capture_summary"] = str(final_capture_json)
        asset["source_capture_status"] = final_status
        asset["bundle"] = capture["bundle"]
        asset["assets"] = {
            "mp4": str(mp4),
            "keyframes_png": str(keyframes),
            "metrics_csv": str(metrics_csv),
            "readme": str(readme),
        }
        asset["asset_sizes"] = {key: Path(value).stat().st_size for key, value in asset["assets"].items()}
        asset["asset_sha256"] = {key: sha256_file(Path(value)) for key, value in asset["assets"].items()}
        asset["checks"] = {
            **asset.get("checks", {}),
            "capture_status_ok": capture["checks"]["capture_status_ok"],
            "uses_official_importer_export_usd": True,
            "uses_robot_order_fk_repaired_full_public_motion_bundle": True,
            "uses_iteration_999_checkpoint": capture["checks"]["uses_iteration_999_checkpoint"],
            "does_not_claim_official_checkpoint": True,
            "does_not_claim_paper_level": True,
            "does_not_claim_fig5_fig6": True,
            "does_not_claim_tensorrt": True,
            "does_not_claim_real_robot": True,
        }
        asset["interpretation"] = capture["interpretation"]

    readme.write_text(
        "\n".join(
            [
                "# Robot-Order FK-Repaired PPO Policy Rollout Visualization",
                "",
                "This directory contains a local virtual policy rollout visualization captured from the current",
                "robot-order FK-repaired official-importer-export PPO checkpoint.",
                "",
                "## Claim Level",
                "",
                "local_virtual_official_importer_export_robot_order_fk_repaired_ppo_policy_rollout_video.",
                "This is not the official BeyondMimic tracking teacher checkpoint, not paper-level Fig. 5/Fig. 6",
                "guided diffusion, not TensorRT deployment, and not real-robot evidence.",
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

    base_capture_json.unlink(missing_ok=True)
    base_asset_json.unlink(missing_ok=True)
    write_json(final_capture_json, capture)
    write_json(final_asset_json, asset)
    return {"capture_json": str(final_capture_json), "asset_json": str(final_asset_json), "mp4": str(mp4)}


def main() -> None:
    module = load_base_module()
    module.OUT = OUT
    module.LOG_DIR = LOG_DIR
    module.FAILED_DIR = FAILED_DIR
    module.RUN_ROOT = RUN_ROOT
    module.ENRICHED_USD = OFFICIAL_IMPORTER_USD
    module.OFFICIAL_LOOP_MOTION_NPZ = ROBOT_ORDER_MOTION_NPZ
    module.TRAINING_RUN_JSON = TRAINING_RUN_JSON
    module.CHECKPOINT_EVAL_JSON = CHECKPOINT_EVAL_JSON
    module.SEED = int(os.environ.get("BM_ROBOT_ORDER_FK_PPO_POLICY_VIDEO_SEED", str(DEFAULT_SEED)))
    module.WORKER_CODE = patch_worker_code(module.WORKER_CODE)
    module.RENDER_CODE = patch_render_code(module.RENDER_CODE)
    module.main()
    patched = patch_outputs()
    print(
        json.dumps(
            {
                "status": (
                    "ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout_video_capture"
                ),
                **patched,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
