#!/usr/bin/env python3
"""Diagnose the next blocker after the quality-gated MuJoCo video fix.

The quality gate fixes the near-floor root-target bug.  This script asks the
next question on the same selected segment:

* can the MuJoCo PD/root-assist stack track the reference joint targets?
* how far are the teacher action-derived PD targets from the reference joints?

It renders one additional short diagnostic MP4:
``reference_joint_pd_control``.  This is not a policy rollout; it is a control
adapter baseline for the same segment.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SCRIPT_DIR = ROOT / "reproduction/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import lafan1_continuous_mujoco_action_control_videos as base  # noqa: E402
import stage1_multisource_quality_gated_mujoco_action_control_videos as qg  # noqa: E402


OUT_ROOT = ROOT / "res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos"
SUITE_JSON = OUT_ROOT / "stage1_multisource_quality_gated_video_suite_summary.json"
SELECTOR_JSON = OUT_ROOT / "quality_gated_stage1_multisource_selector_audit.json"
OUT_JSON = OUT_ROOT / "quality_gated_stage1_multisource_adapter_diagnostic.json"
OUT_MD = OUT_ROOT / "quality_gated_stage1_multisource_adapter_diagnostic.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
        "last": float(values.reshape(-1)[-1]),
    }


def metrics_stats(metrics_path: Path, columns: list[str]) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = {column: [] for column in columns}
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for column in columns:
                raw = row.get(column)
                if raw in (None, ""):
                    continue
                try:
                    values[column].append(float(raw))
                except ValueError:
                    pass
    return {column: stats(np.asarray(vals)) for column, vals in values.items() if vals}


def load_quality_segment() -> tuple[dict[str, Any], dict[str, Any]]:
    selector = read_json(SELECTOR_JSON)
    segment = selector.get("selected_segment_full")
    if not segment:
        raise RuntimeError(f"{SELECTOR_JSON} does not contain selected_segment_full")
    qg.patch_artifact_bindings()
    base.OUT_ROOT = OUT_ROOT
    base.TEACHER_ROLLOUT_JSON = qg.TEACHER_ROLLOUT_JSON
    base.BEST_TEACHER_SWEEP_JSON = qg.BEST_TEACHER_SWEEP_JSON
    base.MOTION_BUNDLE = qg.MOTION_BUNDLE_NPZ
    data = base.load_segment(segment, max_frames=int(selector["selected_segment"]["length"]))
    return segment, data


def mujoco_ctrlrange_and_action_meta(frames: int, actions: np.ndarray, reference_joint: np.ndarray) -> dict[str, Any]:
    import mujoco

    action_rows = base.load_action_rows()
    model_path = Path(os.environ.get("BM_MUJOCO_G1_XML", str(base.DEFAULT_MODEL))).expanduser()
    patched_xml = model_path.parent / "g1_mocap_29dof_quality_gated_adapter_diagnostic_pd.xml"
    base.patch_joints_and_actuators(model_path, patched_xml, action_rows)
    model = mujoco.MjModel.from_xml_path(str(patched_xml))
    ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
    action_scale = np.asarray([float(row["action_scale"]) for row in action_rows], dtype=np.float64)
    default_joint_pos, default_source, default_note = base.parse_default_joint_position(action_rows)
    teacher_targets, teacher_meta = base.action_to_joint_targets(
        actions,
        default_joint_pos,
        action_scale,
        ctrlrange,
        frames,
        float(os.environ.get("BM_LAFAN1_ACTION_CLIP", "3.0")),
    )
    reference_targets = np.clip(reference_joint[:frames], ctrlrange[:, 0], ctrlrange[:, 1])
    gap = teacher_targets - reference_targets
    return {
        "patched_xml": str(patched_xml),
        "default_joint_pos_source": default_source,
        "default_joint_pos_note": default_note,
        "teacher_target_meta": teacher_meta,
        "teacher_vs_reference_target_abs_error": {
            "overall": stats(np.abs(gap)),
            "per_frame_mean": stats(np.mean(np.abs(gap), axis=1)),
            "per_joint_mean": stats(np.mean(np.abs(gap), axis=0)),
        },
        "teacher_target_abs": stats(np.abs(teacher_targets)),
        "reference_target_abs": stats(np.abs(reference_targets)),
        "reference_targets": reference_targets,
        "teacher_targets": teacher_targets,
    }


def render_reference_joint_pd(data: dict[str, Any], root_pos: np.ndarray, root_quat: np.ndarray, ref_joint: np.ndarray) -> dict[str, Any]:
    base.render_action_control_video.__globals__["OUT_ROOT"] = OUT_ROOT
    source_meta = {
        "experiment_type": "stage1_multisource_quality_gated_reference_joint_pd_adapter_diagnostic",
        "claim_level": (
            "MuJoCo PD/root-assist adapter diagnostic using reference joint qpos targets on the same quality-gated "
            "segment; not a policy rollout and not paper-level control."
        ),
        "target_source": "reference_joint_qpos_pd_targets",
        "theta_sp_formula": "theta_sp = reference_joint_qpos clipped to MuJoCo actuator ctrlrange",
        "continuity": data["continuity"],
        "limitations": [
            "This is not policy output; it tests the MuJoCo PD/root-assist adapter on reference joint targets.",
            "It is a short 30-frame diagnostic because the current teacher lacks longer stable normal-height segments.",
            "It is not official IsaacLab, not real robot, and not paper-level BeyondMimic Fig.5/Fig.6.",
        ],
    }
    common_meta = {
        "quality_gated_suite_summary": str(SUITE_JSON),
        "selector_audit": str(SELECTOR_JSON),
        "video_frames": int(ref_joint.shape[0]),
        "video_duration_seconds": int(ref_joint.shape[0]) / int(os.environ.get("BM_LAFAN1_VIDEO_FPS", "30")),
    }
    return base.render_action_control_video(
        "reference_joint_pd_control",
        ref_joint,
        root_pos,
        root_quat,
        source_meta,
        common_meta,
    )


def main() -> None:
    if not SUITE_JSON.is_file():
        raise FileNotFoundError(SUITE_JSON)
    if not SELECTOR_JSON.is_file():
        raise FileNotFoundError(SELECTOR_JSON)

    segment, data = load_quality_segment()
    ref_joint, root_pos, root_quat, ref_meta = base.load_continuous_reference_for_steps(data["motion_time_steps"])
    frames = int(ref_joint.shape[0])
    action_meta = mujoco_ctrlrange_and_action_meta(frames, data["actions"], ref_joint)
    ref_targets = action_meta.pop("reference_targets")
    action_meta.pop("teacher_targets")
    rendered = render_reference_joint_pd(data, root_pos, root_quat, ref_targets)

    teacher_summary = read_json(OUT_ROOT / "teacher_policy_action_control/teacher_policy_action_control_summary.json")
    reference_pd_summary = read_json(OUT_ROOT / "reference_joint_pd_control/reference_joint_pd_control_summary.json")
    teacher_metrics = teacher_summary["metrics"]
    reference_pd_metrics = reference_pd_summary["metrics"]
    reference_pd_stats = metrics_stats(
        OUT_ROOT / "reference_joint_pd_control/reference_joint_pd_control_metrics.csv",
        ["root_z", "root_target_z", "root_position_error_m", "joint_error_abs_mean", "fall_proxy"],
    )
    teacher_stats = metrics_stats(
        OUT_ROOT / "teacher_policy_action_control/teacher_policy_action_control_metrics.csv",
        ["root_z", "root_target_z", "root_position_error_m", "joint_error_abs_mean", "fall_proxy"],
    )

    ref_root_min = float(reference_pd_metrics["root_height_min"])
    teacher_root_min = float(teacher_metrics["root_height_min"])
    ref_root_error = float(reference_pd_metrics["root_position_error_mean_m"])
    teacher_root_error = float(teacher_metrics["root_position_error_mean_m"])
    target_gap = action_meta["teacher_vs_reference_target_abs_error"]["per_frame_mean"]["mean"]
    checks = {
        "reference_joint_pd_mp4_exists": bool(rendered.get("checks", {}).get("mp4_exists")),
        "reference_joint_pd_fall_proxy_zero": int(reference_pd_metrics["fall_proxy_count"]) == 0,
        "reference_joint_pd_root_height_above_teacher": ref_root_min >= teacher_root_min,
        "reference_joint_pd_root_error_below_teacher": ref_root_error <= teacher_root_error,
        "teacher_targets_differ_from_reference": target_gap > 0.05,
        "does_not_claim_paper_level": True,
        "does_not_claim_real_robot": True,
    }
    if checks["reference_joint_pd_root_height_above_teacher"] and checks["reference_joint_pd_root_error_below_teacher"]:
        primary_blocker = "teacher_action_targets_or_isaac_to_mujoco_action_mapping"
    else:
        primary_blocker = "mujoco_pd_root_assist_adapter_also_limited"

    payload = {
        "status": "ok_quality_gated_adapter_diagnostic",
        "timestamp_utc": utc_now(),
        "experiment_type": "stage1_multisource_quality_gated_adapter_diagnostic",
        "claim_level": "Adapter/target-gap diagnostic for the short quality-gated MuJoCo suite; not paper-level evidence.",
        "selected_segment": {
            "motion": segment.get("root_diagnosis", {}).get("source_motion"),
            "motion_time_step_start": int(segment["motion_time_step_start"]),
            "motion_time_step_end": int(segment["motion_time_step_end"]),
            "frames": frames,
            "reward_mean": float(segment["reward_mean"]),
        },
        "reference_metadata": ref_meta,
        "action_target_gap": action_meta,
        "reference_joint_pd_control": {
            "summary_json": str(OUT_ROOT / "reference_joint_pd_control/reference_joint_pd_control_summary.json"),
            "metrics_csv": str(OUT_ROOT / "reference_joint_pd_control/reference_joint_pd_control_metrics.csv"),
            "outputs": rendered.get("outputs", {}),
            "metrics": reference_pd_metrics,
            "stats": reference_pd_stats,
        },
        "teacher_policy_action_control": {
            "summary_json": str(OUT_ROOT / "teacher_policy_action_control/teacher_policy_action_control_summary.json"),
            "metrics": teacher_metrics,
            "stats": teacher_stats,
        },
        "checks": checks,
        "primary_blocker_inference": primary_blocker,
        "interpretation": {
            "reference_pd_control_baseline": (
                "If this baseline is much more stable than teacher action-control, then the MuJoCo PD/root-assist "
                "stack can handle the selected reference and the teacher action/action-scale bridge is the main issue."
            ),
            "teacher_target_gap": (
                "The teacher action-derived setpoints are compared against the same segment's reference joint qpos. "
                "Large gaps mean the action stream is not simply replaying the reference pose and can destabilize MuJoCo."
            ),
            "claim_boundary": "This diagnostic still uses root assist and short horizon; it is not paper-level closed-loop control.",
        },
    }
    write_json(OUT_JSON, payload)
    write_markdown(payload)
    print(json.dumps({"status": payload["status"], "json": str(OUT_JSON), "md": str(OUT_MD)}, sort_keys=True))


def write_markdown(payload: dict[str, Any]) -> None:
    ref = payload["reference_joint_pd_control"]["metrics"]
    teacher = payload["teacher_policy_action_control"]["metrics"]
    gap = payload["action_target_gap"]["teacher_vs_reference_target_abs_error"]["per_frame_mean"]
    lines = [
        "# Quality-Gated MuJoCo Adapter Diagnostic",
        "",
        "## 结论",
        "",
        f"Primary blocker inference: `{payload['primary_blocker_inference']}`.",
        "",
        "在同一个 normal-root quality-gated 片段上，新增 `reference_joint_pd_control` 作为 MuJoCo PD/root-assist baseline。"
        "它使用 reference joint qpos 作为 PD target，不是 policy 输出。",
        "",
        "## 指标对比",
        "",
        f"- Reference-PD fall proxy count: `{ref['fall_proxy_count']}`",
        f"- Reference-PD root height min/max: `{ref['root_height_min']}` / `{ref['root_height_max']}` m",
        f"- Reference-PD root position error mean/max: `{ref['root_position_error_mean_m']}` / `{ref['root_position_error_max_m']}` m",
        f"- Teacher-action fall proxy count: `{teacher['fall_proxy_count']}`",
        f"- Teacher-action root height min/max: `{teacher['root_height_min']}` / `{teacher['root_height_max']}` m",
        f"- Teacher-action root position error mean/max: `{teacher['root_position_error_mean_m']}` / `{teacher['root_position_error_max_m']}` m",
        f"- Teacher target vs reference target per-frame mean abs gap: mean `{gap['mean']}`, max `{gap['max']}` rad",
        "",
        "## 解释",
        "",
        "- 如果 Reference-PD 明显更稳，下一步优先查 teacher action、action scale、obs/action adapter。",
        "- 如果 Reference-PD 也明显下滑，下一步优先查 MuJoCo PD/root-assist/asset dynamics 适配。",
        "- 该诊断仍是短时 root-assist MuJoCo local diagnostic，不是 paper-level BeyondMimic 控制结果。",
        "",
        f"JSON: `{OUT_JSON}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
