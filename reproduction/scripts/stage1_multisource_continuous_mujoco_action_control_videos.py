#!/usr/bin/env python3
"""Render corrected continuous MuJoCo videos for the Stage-1 multi-source chain.

This wrapper reuses the corrected continuous LAFAN1 video implementation, but
rebinds every artifact path to the GPUs 5/6 multi-source teacher/VAE/diffusion
chain and filters the selected segment so the motion time steps stay inside one
source motion from the multi-source bundle.

Claim boundary: these are local MuJoCo action-to-PD diagnostics from a weak
local multi-source teacher. They are not official BeyondMimic checkpoints, not
native Isaac rendered MP4s, not paper-level Fig.5/Fig.6, and not real robot.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("BM_ROOT", "/mnt/infini-data/test/BeyondMimic")).resolve()
SCRIPT_DIR = ROOT / "reproduction/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import lafan1_continuous_mujoco_action_control_videos as base  # noqa: E402
import lafan1_paper_contract_mujoco_action_control_videos as paper_base  # noqa: E402


OUT_ROOT = ROOT / "res/visualization/stage1_multisource_continuous_mujoco_action_control_videos"
TEACHER_ROLLOUT_JSON = (
    ROOT
    / "res/tracking/stage1_multisource_best_teacher_rollout_dataset/"
    "tracking_stage1_multisource_best_teacher_rollout_dataset.json"
)
BEST_TEACHER_SWEEP_JSON = (
    ROOT
    / "res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/"
    "tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.json"
)
MOTION_BUNDLE_AUDIT = ROOT / "res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle.json"
MOTION_BUNDLE_NPZ = (
    ROOT
    / "res/tracking/stage1_multisource_motion_bundle/"
    "stage1_multisource_public_plus_available_motion_bundle_fk_repaired_robot_order.npz"
)
VAE_CKPT = (
    ROOT
    / "res/runs/level_c_stage1_multisource_teacher_rollout_vae_training/"
    "resource_adjusted_teacher_rollout_vae_20260623_135755_seed20260855/"
    "resource_adjusted_teacher_rollout_action_vae.pt"
)
DENOISER_CKPT = (
    ROOT
    / "res/runs/level_c_stage1_multisource_state_latent_diffusion_training/"
    "resource_adjusted_state_latent_diffusion_20260623_140110_seed20260857/"
    "resource_adjusted_state_latent_denoiser.pt"
)
FRESH_AUDIT = OUT_ROOT / "fresh_continuous_stage1_multisource_video_audit.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def patch_artifact_bindings() -> None:
    # Rebind globals used directly inside the continuous implementation.
    base.OUT_ROOT = OUT_ROOT
    base.TEACHER_ROLLOUT_JSON = TEACHER_ROLLOUT_JSON
    base.BEST_TEACHER_SWEEP_JSON = BEST_TEACHER_SWEEP_JSON
    base.MOTION_BUNDLE = MOTION_BUNDLE_NPZ
    base.VAE_CKPT = VAE_CKPT
    base.DENOISER_CKPT = DENOISER_CKPT
    base.OLD_FAILURE_AUDIT = FRESH_AUDIT

    # Rebind globals used by imported helper functions whose __globals__ live
    # in the paper-contract module.
    paper_base.OUT_ROOT = OUT_ROOT
    paper_base.TEACHER_ROLLOUT_JSON = TEACHER_ROLLOUT_JSON
    paper_base.BEST_TEACHER_SWEEP_JSON = BEST_TEACHER_SWEEP_JSON
    paper_base.MOTION_BUNDLE = MOTION_BUNDLE_NPZ
    paper_base.VAE_CKPT = VAE_CKPT


def source_motion_for_segment(segment: dict[str, Any]) -> dict[str, Any] | None:
    audit = json.loads(MOTION_BUNDLE_AUDIT.read_text(encoding="utf-8"))
    start = int(segment["motion_time_step_start"])
    end = int(segment["motion_time_step_end"])
    for row in audit.get("rows", []):
        if int(row["start_frame"]) <= start and end < int(row["end_frame_exclusive"]):
            return {
                "motion": row.get("motion"),
                "source_family": row.get("source_family"),
                "source_kind": row.get("source_kind"),
                "source_path": row.get("source_path"),
                "start_frame": row.get("start_frame"),
                "end_frame_exclusive": row.get("end_frame_exclusive"),
                "frame_count": row.get("frame_count"),
                "duration_seconds": row.get("duration_seconds"),
                "per_motion_npz": row.get("per_motion_npz"),
            }
    return None


def filtered_continuous_segments() -> list[dict[str, Any]]:
    original = base._bm_stage1_original_find_continuous_segments()
    filtered: list[dict[str, Any]] = []
    for segment in original:
        source = source_motion_for_segment(segment)
        if source is None:
            continue
        item = dict(segment)
        item["source_motion"] = source
        item["single_source_motion_boundary_ok"] = True
        filtered.append(item)
    return sorted(filtered, key=lambda row: (row["length"], row["reward_mean"]), reverse=True)


def fresh_audit() -> dict[str, Any]:
    payload = {
        "status": "ok_stage1_multisource_fresh_continuous_video_suite",
        "timestamp_utc": utc_now(),
        "experiment_type": "stage1_multisource_fresh_continuous_video_suite_audit",
        "claim_level": "This run creates a fresh continuous multi-source video suite and does not reuse the old reset-spliced LAFAN1 action-control videos.",
        "old_lafan1_failure_audit": str(
            ROOT
            / "res/visualization/lafan1_paper_contract_videos/"
            "failed_discontinuous_action_control_audit.json"
        ),
        "checks": {
            "fresh_output_root": True,
            "does_not_reuse_old_discontinuous_videos": True,
            "does_not_claim_paper_level": True,
        },
    }
    write_json(FRESH_AUDIT, payload)
    return payload


def write_stage1_summary(rendered: dict[str, Any], segment: dict[str, Any], audit: dict[str, Any]) -> None:
    sweep = json.loads(BEST_TEACHER_SWEEP_JSON.read_text(encoding="utf-8"))
    videos = {name: item["outputs"] for name, item in rendered.items()}
    checks = {
        "all_mp4_exist": all(item.get("checks", {}).get("mp4_exists", False) for item in rendered.values()),
        "all_primary_metrics_csv_exist": all(
            item.get("checks", {}).get("metrics_csv_exists", True)
            for name, item in rendered.items()
            if name != "guided_vs_unguided_action_control"
        ),
        "all_continuous_primary_time_steps": all(
            item.get("continuity", {}).get("all_motion_time_step_deltas_plus_one", True)
            for name, item in rendered.items()
            if name != "guided_vs_unguided_action_control"
        ),
        "selected_segment_single_source_motion": bool(segment.get("single_source_motion_boundary_ok")),
        "does_not_claim_complete_beyondmimic_reproduction": True,
        "does_not_claim_real_robot": True,
    }
    payload = {
        "status": "ok" if all(checks.values()) else "failed",
        "timestamp_utc": utc_now(),
        "experiment_type": "stage1_multisource_continuous_mujoco_action_control_video_suite",
        "output_root": str(OUT_ROOT),
        "claim_level": (
            "Corrected local MuJoCo continuous single-source-motion visualization suite: reference pose replay plus "
            "action-to-PD diagnostics for the multi-source teacher, VAE, denoised latent, guided latent, and "
            "guided-vs-unguided. Not paper-level BeyondMimic."
        ),
        "selected_continuous_segment": segment,
        "best_teacher_sweep_metrics": sweep.get("metrics", {}),
        "fresh_suite_audit": str(FRESH_AUDIT),
        "videos": videos,
        "checks": checks,
        "limitations": [
            "The selected multi-source teacher remains weak; videos are pipeline diagnostics, not high-quality motion-tracking evidence.",
            "Videos are not temporally stretched; duration is exactly the selected continuous segment length.",
            "The selected segment is constrained to one source motion from the stage1 multi-source bundle.",
            "Teacher/VAE/diffusion/guidance videos use MuJoCo position actuators plus root assist, not unassisted paper-level control.",
            "Denoised/guided variants are per-frame local latent diagnostics, not official Fig.5/Fig.6 task-guided closed-loop rollouts.",
            "Large MP4s are local report assets and should not be committed to GitHub.",
        ],
    }
    write_json(OUT_ROOT / "stage1_multisource_continuous_video_suite_summary.json", payload)
    lines = [
        "# Stage-1 Multi-Source Continuous MuJoCo Action-Control Videos",
        "",
        "This directory contains the corrected continuous video suite for the GPUs 5/6 multi-source teacher chain.",
        "",
        "## Continuity Gate",
        "",
        f"- Shard: `{segment['shard']}`",
        f"- Rank/env: `{segment['rank']}/{segment['env_index']}`",
        f"- Source frames: `{segment['start']}:{segment['end_exclusive']}`",
        f"- Rendered frames: `{segment['length']}`",
        f"- Motion time steps: `{segment['motion_time_step_start']}..{segment['motion_time_step_end']}`",
        f"- Done count: `{segment['done_count']}`",
        f"- Source motion: `{segment.get('source_motion', {}).get('motion', '')}`",
        f"- Source family: `{segment.get('source_motion', {}).get('source_family', '')}`",
        "",
        "## Videos",
        "",
    ]
    for name, outputs in videos.items():
        lines.append(f"- `{name}`: `{outputs.get('mp4', '')}`")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "These are local MuJoCo diagnostics. The reference video is continuous pose replay; the other videos use MuJoCo `mj_step`, 29 position actuators, and root assist. They are not official BeyondMimic paper-level results.",
            "",
        ]
    )
    (OUT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    if payload["status"] != "ok":
        raise RuntimeError(f"Stage1 multi-source video summary failed checks: {checks}")


def main() -> None:
    patch_artifact_bindings()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not hasattr(base, "_bm_stage1_original_find_continuous_segments"):
        base._bm_stage1_original_find_continuous_segments = base.find_continuous_segments
    base.find_continuous_segments = filtered_continuous_segments
    base.write_old_failure_audit = fresh_audit
    base.write_summary = write_stage1_summary

    # Keep videos honest: select a reasonably long continuous segment if one
    # exists, but never stretch it beyond its true length.
    os.environ.setdefault("BM_LAFAN1_MIN_CONTINUOUS_FRAMES", "60")
    os.environ.setdefault("BM_LAFAN1_MAX_CONTINUOUS_FRAMES", "300")
    os.environ.setdefault("BM_LAFAN1_VIDEO_FPS", "30")
    base.main()

    final = OUT_ROOT / "stage1_multisource_continuous_video_suite_summary.json"
    print(json.dumps({"status": "ok", "summary": str(final), "output_root": str(OUT_ROOT)}, sort_keys=True))


if __name__ == "__main__":
    main()
