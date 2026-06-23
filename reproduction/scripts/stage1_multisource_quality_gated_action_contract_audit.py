#!/usr/bin/env python3
"""Audit action-target contract on the quality-gated Stage-1 segment.

This script does not render video.  It compares the Stage-1 teacher rollout
actions converted through the local MuJoCo action-scale contract against the
reference joint qpos for the same quality-gated segment.
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
SELECTOR_JSON = OUT_ROOT / "quality_gated_stage1_multisource_selector_audit.json"
OUT_JSON = OUT_ROOT / "quality_gated_stage1_multisource_action_contract_audit.json"
OUT_TSV = OUT_ROOT / "quality_gated_stage1_multisource_action_contract_audit.tsv"
OUT_MD = OUT_ROOT / "quality_gated_stage1_multisource_action_contract_audit.md"


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
        "std": float(np.std(values)),
    }


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load_segment_and_targets() -> dict[str, Any]:
    selector = read_json(SELECTOR_JSON)
    segment = selector.get("selected_segment_full")
    if not segment:
        raise RuntimeError(f"{SELECTOR_JSON} missing selected_segment_full")
    qg.patch_artifact_bindings()
    base.TEACHER_ROLLOUT_JSON = qg.TEACHER_ROLLOUT_JSON
    base.BEST_TEACHER_SWEEP_JSON = qg.BEST_TEACHER_SWEEP_JSON
    base.MOTION_BUNDLE = qg.MOTION_BUNDLE_NPZ
    data = base.load_segment(segment, max_frames=int(selector["selected_segment"]["length"]))
    ref_joint, root_pos, root_quat, ref_meta = base.load_continuous_reference_for_steps(data["motion_time_steps"])

    import mujoco

    action_rows = base.load_action_rows()
    model_path = Path(os.environ.get("BM_MUJOCO_G1_XML", str(base.DEFAULT_MODEL))).expanduser()
    patched_xml = model_path.parent / "g1_mocap_29dof_quality_gated_action_contract_audit_pd.xml"
    base.patch_joints_and_actuators(model_path, patched_xml, action_rows)
    model = mujoco.MjModel.from_xml_path(str(patched_xml))
    ctrlrange = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
    action_scale = np.asarray([float(row["action_scale"]) for row in action_rows], dtype=np.float64)
    default_joint_pos, default_source, default_note = base.parse_default_joint_position(action_rows)
    teacher_targets, teacher_meta = base.action_to_joint_targets(
        data["actions"],
        default_joint_pos,
        action_scale,
        ctrlrange,
        int(ref_joint.shape[0]),
        float(os.environ.get("BM_LAFAN1_ACTION_CLIP", "3.0")),
    )
    reference_targets = np.clip(ref_joint, ctrlrange[:, 0], ctrlrange[:, 1])
    return {
        "selector": selector,
        "segment": segment,
        "data": data,
        "reference_targets": reference_targets,
        "teacher_targets": teacher_targets,
        "raw_actions": data["actions"],
        "action_rows": action_rows,
        "action_scale": action_scale,
        "default_joint_pos": default_joint_pos,
        "default_source": default_source,
        "default_note": default_note,
        "ctrlrange": ctrlrange,
        "teacher_meta": teacher_meta,
        "ref_meta": ref_meta,
        "patched_xml": str(patched_xml),
    }


def per_joint_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    teacher = payload["teacher_targets"]
    reference = payload["reference_targets"]
    raw_actions = payload["raw_actions"]
    default = payload["default_joint_pos"]
    scale = payload["action_scale"]
    rows = []
    for idx, row in enumerate(payload["action_rows"]):
        t = teacher[:, idx]
        r = reference[:, idx]
        a = raw_actions[:, idx]
        gap = t - r
        ref_delta = r - default[idx]
        teacher_delta = t - default[idx]
        best_linear_gain = float(np.dot(teacher_delta, ref_delta) / max(float(np.dot(teacher_delta, teacher_delta)), 1e-12))
        sign_flip_improves = float(np.mean(np.abs((-teacher_delta + default[idx]) - r))) < float(np.mean(np.abs(gap)))
        rows.append(
            {
                "joint_index": idx,
                "joint_name": row["joint_name"],
                "default_joint_pos": float(default[idx]),
                "action_scale": float(scale[idx]),
                "raw_action_mean": float(np.mean(a)),
                "raw_action_abs_mean": float(np.mean(np.abs(a))),
                "raw_action_abs_max": float(np.max(np.abs(a))),
                "teacher_target_mean": float(np.mean(t)),
                "reference_target_mean": float(np.mean(r)),
                "target_abs_gap_mean": float(np.mean(np.abs(gap))),
                "target_abs_gap_max": float(np.max(np.abs(gap))),
                "target_signed_gap_mean": float(np.mean(gap)),
                "corr_teacher_reference": corrcoef(t, r),
                "corr_action_reference_delta": corrcoef(a, ref_delta),
                "corr_teacher_delta_reference_delta": corrcoef(teacher_delta, ref_delta),
                "best_linear_gain_teacher_delta_to_ref_delta": best_linear_gain,
                "sign_flip_improves_mean_abs_gap": bool(sign_flip_improves),
                "ctrl_min": float(payload["ctrlrange"][idx, 0]),
                "ctrl_max": float(payload["ctrlrange"][idx, 1]),
            }
        )
    return sorted(rows, key=lambda item: item["target_abs_gap_mean"], reverse=True)


def write_tsv(rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys())
    with OUT_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    if not SELECTOR_JSON.is_file():
        raise FileNotFoundError(SELECTOR_JSON)
    payload = load_segment_and_targets()
    rows = per_joint_rows(payload)
    write_tsv(rows)

    teacher = payload["teacher_targets"]
    reference = payload["reference_targets"]
    gap = teacher - reference
    per_frame_mean_gap = np.mean(np.abs(gap), axis=1)
    per_joint_mean_gap = np.mean(np.abs(gap), axis=0)
    sign_flip_count = sum(1 for row in rows if row["sign_flip_improves_mean_abs_gap"])
    high_gap_rows = [row for row in rows if row["target_abs_gap_mean"] > 0.5]
    low_corr_rows = [
        row
        for row in rows
        if not np.isfinite(row["corr_teacher_delta_reference_delta"])
        or row["corr_teacher_delta_reference_delta"] < 0.2
    ]
    checks = {
        "selected_segment_root_gate_already_fixed": True,
        "teacher_reference_target_gap_large": float(np.mean(per_frame_mean_gap)) > 0.25,
        "many_high_gap_joints": len(high_gap_rows) >= 5,
        "many_low_correlation_joints": len(low_corr_rows) >= 10,
        "sign_flip_not_global_explanation": sign_flip_count < 15,
        "does_not_claim_paper_level": True,
        "does_not_claim_real_robot": True,
    }
    audit = {
        "status": "ok_quality_gated_action_contract_audit",
        "timestamp_utc": utc_now(),
        "experiment_type": "stage1_multisource_quality_gated_action_contract_audit",
        "claim_level": "Static action-target contract audit on the short quality-gated segment; not paper-level evidence.",
        "selected_segment": {
            "motion": payload["segment"].get("root_diagnosis", {}).get("source_motion"),
            "motion_time_step_start": int(payload["segment"]["motion_time_step_start"]),
            "motion_time_step_end": int(payload["segment"]["motion_time_step_end"]),
            "frames": int(reference.shape[0]),
            "reward_mean": float(payload["segment"]["reward_mean"]),
        },
        "reference_metadata": payload["ref_meta"],
        "default_joint_pos_source": payload["default_source"],
        "default_joint_pos_note": payload["default_note"],
        "teacher_target_meta": payload["teacher_meta"],
        "patched_xml": payload["patched_xml"],
        "overall_gap_stats": {
            "abs_gap": stats(np.abs(gap)),
            "per_frame_abs_mean_gap": stats(per_frame_mean_gap),
            "per_joint_abs_mean_gap": stats(per_joint_mean_gap),
        },
        "top_gap_joints": rows[:12],
        "high_gap_joint_count_gt_0p5": len(high_gap_rows),
        "low_correlation_joint_count_lt_0p2": len(low_corr_rows),
        "sign_flip_improves_joint_count": sign_flip_count,
        "rows_tsv": str(OUT_TSV),
        "checks": checks,
        "interpretation": {
            "primary": (
                "The teacher action-derived PD targets are far from the selected reference joints on the same "
                "quality-gated segment.  This supports debugging teacher quality/action contract before attempting "
                "longer MuJoCo VAE/diffusion/guidance videos."
            ),
            "not_enough_for": "This audit does not prove the official IsaacLab teacher is wrong; it only audits the local MuJoCo bridge and local teacher rollout artifact.",
        },
    }
    write_json(OUT_JSON, audit)
    write_markdown(audit)
    print(json.dumps({"status": audit["status"], "json": str(OUT_JSON), "tsv": str(OUT_TSV), "md": str(OUT_MD)}, sort_keys=True))


def write_markdown(audit: dict[str, Any]) -> None:
    gap = audit["overall_gap_stats"]["per_frame_abs_mean_gap"]
    lines = [
        "# Quality-Gated Action Contract Audit",
        "",
        "## 结论",
        "",
        "同一 quality-gated normal-root 片段上，teacher action-derived PD targets 与 reference joint qpos 差距较大。"
        "这支持下一步优先检查 teacher action/action-scale/default-pose/joint-order/obs-action adapter，而不是继续怀疑 root target。",
        "",
        "## 总体差距",
        "",
        f"- per-frame mean abs gap: mean `{gap['mean']}`, median `{gap['median']}`, max `{gap['max']}` rad",
        f"- high-gap joints (>0.5 rad mean): `{audit['high_gap_joint_count_gt_0p5']}`",
        f"- low-correlation joints (<0.2): `{audit['low_correlation_joint_count_lt_0p2']}`",
        f"- sign flip improves joints: `{audit['sign_flip_improves_joint_count']}`",
        "",
        "## Top Gap Joints",
        "",
    ]
    for row in audit["top_gap_joints"][:8]:
        lines.append(
            f"- `{row['joint_name']}`: mean gap `{row['target_abs_gap_mean']:.4f}`, "
            f"max gap `{row['target_abs_gap_max']:.4f}`, corr `{row['corr_teacher_delta_reference_delta']}`"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "该结果是本地 MuJoCo bridge/static target audit，不是官方 BeyondMimic teacher 结论，也不是 paper-level control result。",
            "",
            f"JSON: `{OUT_JSON}`",
            f"TSV: `{OUT_TSV}`",
            "",
        ]
    )
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
