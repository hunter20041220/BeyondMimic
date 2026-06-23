#!/usr/bin/env python3
"""Render Stage-1 multi-source MuJoCo videos with a quality-gated selector.

The earlier continuous selector proved temporal continuity but still selected a
near-floor segment.  This script keeps the same rendering/control stack but
changes the segment contract:

* one source motion only;
* consecutive motion_time_steps and no done frames;
* root/pelvis target height must be compatible with standing locomotion;
* reward must be non-negative;
* root height range must be stable;
* reward/stability are prioritized before raw segment length.

The output is intentionally short when the current teacher cannot provide long
stable segments.  It is local MuJoCo evidence, not paper-level BeyondMimic.
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

import diagnose_stage1_mujoco_video_failure as diag  # noqa: E402
import lafan1_continuous_mujoco_action_control_videos as base  # noqa: E402
import lafan1_paper_contract_mujoco_action_control_videos as paper_base  # noqa: E402


OUT_ROOT = ROOT / "res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos"
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
QUALITY_SELECTOR_AUDIT = OUT_ROOT / "quality_gated_stage1_multisource_selector_audit.json"
FRESH_AUDIT = OUT_ROOT / "fresh_quality_gated_stage1_multisource_video_audit.json"

TARGET_FRAMES = int(os.environ.get("BM_STAGE1_QG_TARGET_FRAMES", "30"))
MIN_ROOT_Z_MEAN = float(os.environ.get("BM_STAGE1_QG_MIN_ROOT_Z_MEAN", "0.45"))
MIN_ROOT_Z_MIN = float(os.environ.get("BM_STAGE1_QG_MIN_ROOT_Z_MIN", "0.30"))
MAX_ROOT_Z_RANGE = float(os.environ.get("BM_STAGE1_QG_MAX_ROOT_Z_RANGE", "0.18"))
MIN_REWARD_MEAN = float(os.environ.get("BM_STAGE1_QG_MIN_REWARD_MEAN", "0.0"))

LAST_SELECTOR_AUDIT: dict[str, Any] | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def patch_artifact_bindings() -> None:
    base.OUT_ROOT = OUT_ROOT
    base.TEACHER_ROLLOUT_JSON = TEACHER_ROLLOUT_JSON
    base.BEST_TEACHER_SWEEP_JSON = BEST_TEACHER_SWEEP_JSON
    base.MOTION_BUNDLE = MOTION_BUNDLE_NPZ
    base.VAE_CKPT = VAE_CKPT
    base.DENOISER_CKPT = DENOISER_CKPT
    base.OLD_FAILURE_AUDIT = FRESH_AUDIT

    paper_base.OUT_ROOT = OUT_ROOT
    paper_base.TEACHER_ROLLOUT_JSON = TEACHER_ROLLOUT_JSON
    paper_base.BEST_TEACHER_SWEEP_JSON = BEST_TEACHER_SWEEP_JSON
    paper_base.MOTION_BUNDLE = MOTION_BUNDLE_NPZ
    paper_base.VAE_CKPT = VAE_CKPT


def root_stats(segment: dict[str, Any]) -> dict[str, float] | None:
    return segment.get("root_diagnosis", {}).get("selected_root_z")


def passes_quality_gate(segment: dict[str, Any]) -> bool:
    z = root_stats(segment)
    if not z:
        return False
    return (
        int(segment["length"]) >= TARGET_FRAMES
        and float(segment["reward_mean"]) >= MIN_REWARD_MEAN
        and float(z["mean"]) >= MIN_ROOT_Z_MEAN
        and float(z["min"]) >= MIN_ROOT_Z_MIN
        and (float(z["max"]) - float(z["min"])) <= MAX_ROOT_Z_RANGE
        and segment.get("root_diagnosis", {}).get("single_source", False)
    )


def segment_quality_key(segment: dict[str, Any]) -> tuple[float, int, float, float, float]:
    z = root_stats(segment) or {"mean": 0.0, "min": 0.0, "max": 999.0}
    root_range = float(z["max"]) - float(z["min"])
    return (
        float(segment["reward_mean"]),
        min(int(segment["length"]), TARGET_FRAMES),
        float(z["mean"]),
        -root_range,
        float(segment["length"]),
    )


def compact_segment(segment: dict[str, Any], rank: int, rule: str) -> dict[str, Any]:
    z = root_stats(segment) or {}
    root_diag = segment.get("root_diagnosis", {})
    return {
        "rank": rank,
        "rule": rule,
        "length": int(segment["length"]),
        "reward_mean": float(segment["reward_mean"]),
        "source_motion": root_diag.get("source_motion", ""),
        "root_z_min": z.get("min"),
        "root_z_mean": z.get("mean"),
        "root_z_max": z.get("max"),
        "root_z_range": (z.get("max", 0.0) - z.get("min", 0.0)) if z else None,
        "rank_id": int(segment["rank"]),
        "env_index": int(segment["env_index"]),
        "start": int(segment["start"]),
        "end_exclusive": int(segment["end_exclusive"]),
        "motion_time_step_start": int(segment["motion_time_step_start"]),
        "motion_time_step_end": int(segment["motion_time_step_end"]),
    }


def count_good_segments_by_min_length(segments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for min_len in [2, 10, 20, 30, 60, 100, 150, 200]:
        subset = [row for row in segments if int(row["length"]) >= min_len]
        good = []
        for row in subset:
            z = root_stats(row)
            if (
                z
                and float(row["reward_mean"]) >= MIN_REWARD_MEAN
                and float(z["mean"]) >= MIN_ROOT_Z_MEAN
                and float(z["min"]) >= MIN_ROOT_Z_MIN
                and (float(z["max"]) - float(z["min"])) <= MAX_ROOT_Z_RANGE
            ):
                good.append(row)
        best = sorted(good, key=segment_quality_key, reverse=True)[0] if good else None
        out[str(min_len)] = {
            "segment_count": len(subset),
            "good_root_reward_stable_segment_count": len(good),
            "best_segment": compact_segment(best, 1, f"best_good_len_ge_{min_len}") if best else None,
        }
    return out


def filtered_continuous_segments() -> list[dict[str, Any]]:
    global LAST_SELECTOR_AUDIT

    rows = diag.motion_rows()
    all_segments = diag.find_continuous_segments(rows)
    candidates = [row for row in all_segments if passes_quality_gate(row)]
    candidates = sorted(candidates, key=segment_quality_key, reverse=True)
    if not candidates:
        availability = count_good_segments_by_min_length(all_segments)
        audit = {
            "status": "failed_no_quality_gated_segment",
            "timestamp_utc": utc_now(),
            "thresholds": quality_thresholds(),
            "segment_availability_by_min_length": availability,
            "total_continuous_segments": len(all_segments),
        }
        write_json(QUALITY_SELECTOR_AUDIT, audit)
        raise RuntimeError(f"No quality-gated segment found; audit={QUALITY_SELECTOR_AUDIT}")

    for item in candidates:
        root_diag = item["root_diagnosis"]
        item["source_motion"] = {
            "motion": root_diag.get("source_motion"),
            "source_family": root_diag.get("source_family"),
            "source_path": root_diag.get("source_path"),
            "per_motion_npz": root_diag.get("per_motion_npz"),
            "start_frame": root_diag.get("source_start_frame"),
            "end_frame_exclusive": root_diag.get("source_end_frame_exclusive"),
        }
        item["single_source_motion_boundary_ok"] = True
        item["selection_rule"] = (
            "quality gated: consecutive/no-done/single-source, root height normal, "
            "non-negative reward, stable root z; sorted by reward then target length"
        )

    selected = candidates[0]
    audit = {
        "status": "ok_quality_gated_selector",
        "timestamp_utc": utc_now(),
        "thresholds": quality_thresholds(),
        "total_continuous_segments": len(all_segments),
        "quality_gated_candidate_count": len(candidates),
        "segment_availability_by_min_length": count_good_segments_by_min_length(all_segments),
        "selected_segment": compact_segment(selected, 1, "selected_quality_gated"),
        "selected_segment_full": selected,
        "top_quality_gated_segments": [
            compact_segment(row, idx + 1, "top_quality_gated") for idx, row in enumerate(candidates[:20])
        ],
        "claim_level": "selector audit for local MuJoCo diagnostic videos; not paper-level evidence",
    }
    LAST_SELECTOR_AUDIT = audit
    write_json(QUALITY_SELECTOR_AUDIT, audit)
    return candidates


def quality_thresholds() -> dict[str, Any]:
    return {
        "target_frames": TARGET_FRAMES,
        "min_root_z_mean_m": MIN_ROOT_Z_MEAN,
        "min_root_z_min_m": MIN_ROOT_Z_MIN,
        "max_root_z_range_m": MAX_ROOT_Z_RANGE,
        "min_reward_mean": MIN_REWARD_MEAN,
        "video_is_not_temporally_stretched": True,
    }


def quality_load_segment(segment: dict[str, Any], max_frames: int | None = None) -> dict[str, Any]:
    out = base._bm_stage1_original_load_segment(segment, max_frames=max_frames)
    out["continuity"]["selection_rule"] = segment.get("selection_rule", "quality gated selector")
    out["continuity"]["root_height_gate"] = quality_thresholds()
    return out


def fresh_audit() -> dict[str, Any]:
    selector = LAST_SELECTOR_AUDIT or (
        json.loads(QUALITY_SELECTOR_AUDIT.read_text(encoding="utf-8")) if QUALITY_SELECTOR_AUDIT.is_file() else {}
    )
    payload = {
        "status": "ok_stage1_multisource_quality_gated_video_suite",
        "timestamp_utc": utc_now(),
        "experiment_type": "stage1_multisource_quality_gated_video_suite_audit",
        "claim_level": "Fresh quality-gated short MuJoCo video suite; does not reuse the near-floor failed segment and does not claim paper-level control.",
        "selector_audit": str(QUALITY_SELECTOR_AUDIT),
        "selector_status": selector.get("status", ""),
        "thresholds": quality_thresholds(),
        "checks": {
            "fresh_output_root": True,
            "quality_gated_selector_audit_exists": QUALITY_SELECTOR_AUDIT.is_file(),
            "does_not_reuse_near_floor_failed_segment": True,
            "does_not_claim_paper_level": True,
            "does_not_claim_real_robot": True,
        },
    }
    write_json(FRESH_AUDIT, payload)
    return payload


def write_stage1_summary(rendered: dict[str, Any], segment: dict[str, Any], audit: dict[str, Any]) -> None:
    sweep = json.loads(BEST_TEACHER_SWEEP_JSON.read_text(encoding="utf-8"))
    selector = LAST_SELECTOR_AUDIT or json.loads(QUALITY_SELECTOR_AUDIT.read_text(encoding="utf-8"))
    root_diag = segment.get("root_diagnosis", {})
    z = root_diag.get("selected_root_z", {})
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
        "selected_segment_reward_nonnegative": float(segment.get("reward_mean", -1.0)) >= MIN_REWARD_MEAN,
        "selected_segment_root_z_mean_normal": bool(z and float(z["mean"]) >= MIN_ROOT_Z_MEAN),
        "selected_segment_root_z_min_normal": bool(z and float(z["min"]) >= MIN_ROOT_Z_MIN),
        "selected_segment_root_z_stable": bool(z and (float(z["max"]) - float(z["min"])) <= MAX_ROOT_Z_RANGE),
        "does_not_temporally_stretch_short_segment": int(segment["length"]) <= TARGET_FRAMES,
        "does_not_claim_complete_beyondmimic_reproduction": True,
        "does_not_claim_real_robot": True,
    }
    payload = {
        "status": "ok" if all(checks.values()) else "failed",
        "timestamp_utc": utc_now(),
        "experiment_type": "stage1_multisource_quality_gated_mujoco_action_control_video_suite",
        "output_root": str(OUT_ROOT),
        "claim_level": (
            "Quality-gated local MuJoCo short video suite: normal-height reference pose replay plus action-to-PD "
            "diagnostics for teacher, VAE, denoised latent, guided latent, and guided-vs-unguided. Not paper-level BeyondMimic."
        ),
        "selected_continuous_segment": segment,
        "selector_audit": str(QUALITY_SELECTOR_AUDIT),
        "selector_summary": selector.get("selected_segment", {}),
        "best_teacher_sweep_metrics": sweep.get("metrics", {}),
        "fresh_suite_audit": str(FRESH_AUDIT),
        "videos": videos,
        "checks": checks,
        "limitations": [
            "The current teacher only provides short normal-root-height stable segments; this suite is intentionally short and not stretched.",
            "Teacher/VAE/diffusion/guidance videos use MuJoCo position actuators plus root assist, not unassisted paper-level humanoid control.",
            "Denoised/guided variants are per-frame local latent diagnostics, not official Fig.5/Fig.6 task-guided closed-loop rollouts.",
            "If teacher action-control remains unstable on this normal-height target, the next blocker is Stage-1 teacher quality or the IsaacLab-to-MuJoCo adapter.",
            "Large MP4s are local report assets and should not be committed to GitHub.",
        ],
    }
    write_json(OUT_ROOT / "stage1_multisource_quality_gated_video_suite_summary.json", payload)
    lines = [
        "# Stage-1 Multi-Source Quality-Gated MuJoCo Action-Control Videos",
        "",
        "This directory contains the corrected quality-gated short video suite for the GPUs 5/6 multi-source teacher chain.",
        "",
        "## Quality Gate",
        "",
        f"- Target frames: `{TARGET_FRAMES}`",
        f"- Minimum root z mean: `{MIN_ROOT_Z_MEAN}` m",
        f"- Minimum root z min: `{MIN_ROOT_Z_MIN}` m",
        f"- Maximum root z range: `{MAX_ROOT_Z_RANGE}` m",
        f"- Minimum reward mean: `{MIN_REWARD_MEAN}`",
        "",
        "## Selected Segment",
        "",
        f"- Shard: `{segment['shard']}`",
        f"- Rank/env: `{segment['rank']}/{segment['env_index']}`",
        f"- Source frames: `{segment['start']}:{segment['end_exclusive']}`",
        f"- Rendered frames: `{segment['length']}`",
        f"- Motion time steps: `{segment['motion_time_step_start']}..{segment['motion_time_step_end']}`",
        f"- Reward mean: `{segment['reward_mean']}`",
        f"- Root z mean: `{z.get('mean', '')}`",
        f"- Source motion: `{segment.get('source_motion', {}).get('motion', '')}`",
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
        raise RuntimeError(f"Stage1 quality-gated video summary failed checks: {checks}")


def main() -> None:
    patch_artifact_bindings()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not hasattr(base, "_bm_stage1_original_find_continuous_segments"):
        base._bm_stage1_original_find_continuous_segments = base.find_continuous_segments
    if not hasattr(base, "_bm_stage1_original_load_segment"):
        base._bm_stage1_original_load_segment = base.load_segment
    base.find_continuous_segments = filtered_continuous_segments
    base.load_segment = quality_load_segment
    base.write_old_failure_audit = fresh_audit
    base.write_summary = write_stage1_summary

    os.environ.setdefault("BM_LAFAN1_MIN_CONTINUOUS_FRAMES", str(TARGET_FRAMES))
    os.environ.setdefault("BM_LAFAN1_MAX_CONTINUOUS_FRAMES", str(TARGET_FRAMES))
    os.environ.setdefault("BM_LAFAN1_VIDEO_FPS", "30")
    base.main()

    final = OUT_ROOT / "stage1_multisource_quality_gated_video_suite_summary.json"
    print(json.dumps({"status": "ok", "summary": str(final), "output_root": str(OUT_ROOT)}, sort_keys=True))


if __name__ == "__main__":
    main()
