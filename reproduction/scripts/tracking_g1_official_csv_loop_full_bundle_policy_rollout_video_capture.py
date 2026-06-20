#!/usr/bin/env python3
"""Capture a policy rollout video from the full public official-loop bundle.

This wrapper reuses the audited single-motion policy rollout capture/render
path and redirects it to the 40-motion full-bundle PPO checkpoint and motion
artifact. The resulting video is local virtual evidence only: it uses the
enriched USD scaffold and local checkpoint, not unpatched official replay or a
paper-level BeyondMimic teacher result.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tracking_g1_official_csv_loop_policy_rollout_video_capture as base


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/visualization/official_csv_loop_full_bundle_policy_rollout"
LOG_DIR = ROOT / "logs/tracking_g1_official_csv_loop_full_bundle_policy_rollout_video_capture"
FAILED_DIR = ROOT / "res/failed_runs/tracking_g1_official_csv_loop_full_bundle_policy_rollout_video_capture"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_csv_loop_full_bundle_policy_rollout_video_capture"
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
SEED = 20260682


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def patch_base_module() -> None:
    base.OUT = OUT
    base.LOG_DIR = LOG_DIR
    base.FAILED_DIR = FAILED_DIR
    base.RUN_ROOT = RUN_ROOT
    base.OFFICIAL_LOOP_MOTION_NPZ = FULL_BUNDLE_MOTION_NPZ
    base.TRAINING_RUN_JSON = TRAINING_RUN_JSON
    base.CHECKPOINT_EVAL_JSON = CHECKPOINT_EVAL_JSON
    base.SEED = SEED


def patch_render_code() -> None:
    base.RENDER_CODE = base.RENDER_CODE.replace(
        "Official-Loop Policy Rollout Visualization",
        "Official-Loop Full-Bundle Policy Rollout Visualization",
    ).replace(
        "local_virtual_resource_adjusted_policy_rollout_video",
        "local_virtual_full_bundle_resource_adjusted_policy_rollout_video",
    ).replace(
        "official csv-loop PPO checkpoint",
        "official csv-loop full-bundle PPO checkpoint",
    )


def patch_outputs() -> None:
    capture_json = OUT / "tracking_g1_official_csv_loop_policy_rollout_capture.json"
    asset_json = OUT / "official_csv_loop_policy_rollout_video_asset.json"
    capture = load_json(capture_json)
    asset = load_json(asset_json)
    bundle = load_json(FULL_BUNDLE_AUDIT)
    bundle_info = bundle.get("bundle", {})
    capture_ok = capture.get("status") == "ok_official_csv_loop_policy_rollout_video_capture"
    asset_ok = asset.get("status") == "ok"

    capture["status"] = (
        "ok_official_csv_loop_full_bundle_policy_rollout_video_capture"
        if capture_ok and asset_ok
        else f"failed_full_bundle_policy_rollout_from_{capture.get('status')}"
    )
    capture["experiment_type"] = "tracking_official_csv_loop_full_bundle_policy_rollout_video_capture"
    capture["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    capture["scope"] = (
        "Captures one local virtual policy-vs-reference rollout video from the 40-motion public official-csv-loop "
        "bundle PPO checkpoint. This is stronger report media than the earlier single-motion policy video, but it "
        "still uses the enriched USD scaffold and local checkpoint."
    )
    capture.setdefault("inputs", {})
    capture["inputs"].update(
        {
            "training_run_json": str(TRAINING_RUN_JSON),
            "checkpoint_eval_json": str(CHECKPOINT_EVAL_JSON),
            "full_bundle_motion_audit": str(FULL_BUNDLE_AUDIT),
            "motion_npz": str(FULL_BUNDLE_MOTION_NPZ),
        }
    )
    capture.setdefault("input_checks", {})
    capture["input_checks"].update(
        {
            "full_bundle_motion_npz_exists": FULL_BUNDLE_MOTION_NPZ.is_file(),
            "full_bundle_audit_ok": bundle.get("status") == "ok_official_csv_loop_full_bundle_motion_npz",
            "full_bundle_motion_count_40": bundle_info.get("motion_count") == 40,
            "full_bundle_total_frames_11960": bundle_info.get("total_frames") == 11960,
        }
    )
    capture["bundle"] = {
        "motion_count": bundle_info.get("motion_count"),
        "total_frames": bundle_info.get("total_frames"),
        "fps": bundle_info.get("fps"),
        "clip_boundary_count": bundle_info.get("boundary_count"),
    }
    capture.setdefault("checks", {})
    capture["checks"].update(
        {
            "uses_full_public_motion_bundle": capture["input_checks"]["full_bundle_audit_ok"],
            "full_bundle_motion_count_40": capture["input_checks"]["full_bundle_motion_count_40"],
            "does_not_claim_official_checkpoint": True,
        }
    )
    capture["interpretation"] = {
        "goal_complete": False,
        "policy_rollout_video_complete": bool(capture_ok and asset_ok),
        "paper_level_status": (
            "local_virtual_full_bundle_resource_adjusted_policy_rollout_video"
            if capture_ok and asset_ok
            else "not_completed"
        ),
        "why_not_complete": (
            "This is a local virtual full-bundle policy rollout video from a resource-adjusted official-csv-loop "
            "PPO checkpoint. It is not unpatched official replay, not paper-level Fig. 5/Fig. 6 guided diffusion, "
            "not the paper teacher-policy result, and not real-robot evidence."
        ),
    }
    write_json(capture_json, capture)

    if asset:
        asset["experiment_type"] = "tracking_official_csv_loop_full_bundle_policy_rollout_video_capture"
        asset["claim_level"] = "local_virtual_full_bundle_resource_adjusted_policy_rollout_video"
        asset["source_capture_summary"] = str(capture_json)
        asset["source_capture_status"] = capture["status"]
        asset["bundle"] = capture["bundle"]
        asset.setdefault("checks", {})
        asset["checks"]["capture_status_ok"] = capture["status"] == (
            "ok_official_csv_loop_full_bundle_policy_rollout_video_capture"
        )
        asset["checks"]["uses_full_public_motion_bundle"] = capture["checks"]["uses_full_public_motion_bundle"]
        asset["checks"]["full_bundle_motion_count_40"] = capture["checks"]["full_bundle_motion_count_40"]
        asset["checks"]["does_not_claim_official_checkpoint"] = True
        asset["interpretation"] = capture["interpretation"]
        write_json(asset_json, asset)


def main() -> None:
    patch_base_module()
    patch_render_code()
    base.main()
    patch_outputs()
    summary = load_json(OUT / "tracking_g1_official_csv_loop_policy_rollout_capture.json")
    print(
        json.dumps(
            {
                "status": summary.get("status"),
                "json": str(OUT / "tracking_g1_official_csv_loop_policy_rollout_capture.json"),
                "asset_json": str(OUT / "official_csv_loop_policy_rollout_video_asset.json"),
            },
            sort_keys=True,
        )
    )
    if not str(summary.get("status", "")).startswith("ok_"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
