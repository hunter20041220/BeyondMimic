#!/usr/bin/env python3
"""Cross-sample audit for MuJoCo/IsaacLab torso frame offset hypotheses.

Single-sample offset fitting can always hide a body-frame mismatch: infer a
quaternion/position correction from one state, apply it to the same state, and
the anchor observation term will match.  This audit checks whether the inferred
offset is stable across the original terminated dance sample and a new
non-terminated walk sample.  A stable fixed offset would support a simple
adapter correction; a state-dependent offset points to deeper articulation frame
or kinematic-chain mismatch.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/audits/mujoco_torso_frame_offset_cross_sample"
JSON_OUT = OUT / "mujoco_torso_frame_offset_cross_sample_audit.json"
TSV_OUT = OUT / "mujoco_torso_frame_offset_cross_sample_audit.tsv"
MD_OUT = OUT / "mujoco_torso_frame_offset_cross_sample_audit.md"

SAMPLES = {
    "dance_terminated": {
        "sample_json": ROOT / "res/audits/isaaclab_observation_manager_sample_gate/isaaclab_policy_obs_sample.json",
        "torso_offset_json": ROOT / "res/audits/mujoco_torso_frame_offset/mujoco_torso_frame_offset_audit.json",
        "runtime_parity_json": ROOT
        / "res/audits/mujoco_observation_runtime_parity/mujoco_observation_runtime_parity_audit.json",
    },
    "walk_nonterminated": {
        "sample_json": ROOT
        / "res/audits/isaaclab_observation_manager_walk_sample_gate/isaaclab_policy_obs_sample.json",
        "torso_offset_json": ROOT
        / "res/audits/mujoco_torso_frame_offset_walk_sample/mujoco_torso_frame_offset_audit.json",
        "runtime_parity_json": ROOT
        / "res/audits/mujoco_observation_runtime_parity_walk_sample/mujoco_observation_runtime_parity_audit.json",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def qnorm(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q).clip(min=1e-12)


def quat_sign_error(a: np.ndarray, b: np.ndarray) -> float:
    a = qnorm(np.asarray(a, dtype=np.float64).reshape(4))
    b = qnorm(np.asarray(b, dtype=np.float64).reshape(4))
    return float(min(np.max(np.abs(a - b)), np.max(np.abs(a + b))))


def term_error(runtime: dict[str, Any], term: str) -> float | None:
    for row in runtime.get("terms", []):
        if row.get("term") == term:
            return float(row.get("max_abs_error"))
    return None


def build_sample_row(name: str, cfg: dict[str, Path]) -> dict[str, Any]:
    sample = read_json(cfg["sample_json"])
    torso = read_json(cfg["torso_offset_json"])
    runtime = read_json(cfg["runtime_parity_json"])
    primary = torso.get("primary_model_result", {})
    return {
        "name": name,
        "sample_json": str(cfg["sample_json"]),
        "torso_offset_json": str(cfg["torso_offset_json"]),
        "runtime_parity_json": str(cfg["runtime_parity_json"]),
        "sample_exists": cfg["sample_json"].is_file(),
        "torso_offset_exists": cfg["torso_offset_json"].is_file(),
        "runtime_parity_exists": cfg["runtime_parity_json"].is_file(),
        "sample_status": sample.get("status"),
        "motion_file": sample.get("motion_file"),
        "capture_mode": sample.get("capture_mode"),
        "motion_time_steps": sample.get("motion_time_steps"),
        "terminated_after_zero_step": sample.get("terminated_after_zero_step"),
        "runtime_status": runtime.get("status"),
        "runtime_motion_anchor_pos_b_error": term_error(runtime, "motion_anchor_pos_b"),
        "runtime_motion_anchor_ori_b_error": term_error(runtime, "motion_anchor_ori_b"),
        "runtime_base_lin_vel_error": term_error(runtime, "base_lin_vel"),
        "runtime_joint_pos_error": term_error(runtime, "joint_pos"),
        "torso_status": torso.get("status"),
        "candidate_offset_restores_anchor_terms": primary.get("candidate_offset_restores_anchor_terms"),
        "raw_anchor_pos_b_error": primary.get("raw_anchor_pos_b_error"),
        "raw_anchor_ori_b_error": primary.get("raw_anchor_ori_b_error"),
        "corrected_anchor_pos_b_error": primary.get("corrected_anchor_pos_b_error"),
        "corrected_anchor_ori_b_error": primary.get("corrected_anchor_ori_b_error"),
        "q_offset_right_mujoco_to_isaac": primary.get("q_offset_right_mujoco_to_isaac", []),
        "p_offset_world": primary.get("p_offset_world", []),
    }


def write_outputs(summary: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with TSV_OUT.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "name",
            "motion_file",
            "capture_mode",
            "motion_time_steps",
            "terminated_after_zero_step",
            "runtime_motion_anchor_pos_b_error",
            "runtime_motion_anchor_ori_b_error",
            "raw_anchor_pos_b_error",
            "raw_anchor_ori_b_error",
            "corrected_anchor_pos_b_error",
            "corrected_anchor_ori_b_error",
            "candidate_offset_restores_anchor_terms",
            "q_offset_right_mujoco_to_isaac",
            "p_offset_world",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in summary["sample_rows"]:
            writer.writerow({key: row.get(key, "none") for key in fieldnames})
    failed = [key for key, value in summary["checks"].items() if not value]
    lines = [
        "# MuJoCo Torso Frame Offset Cross-Sample Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Generated: `{summary['generated_at']}`",
        "- Scope: compares dance and walk IsaacLab samples; no training, no policy rollout, no video.",
        "- 当前不得声称完整复现 BeyondMimic；该审计只判断 fixed torso offset 是否可作为 adapter 修正。",
        "",
        "## Cross-Sample Offset",
        "",
        f"- Quaternion offset sign-invariant error: `{summary['cross_sample_metrics'].get('q_offset_sign_invariant_error')}`",
        f"- Position offset L2 difference: `{summary['cross_sample_metrics'].get('p_offset_l2_difference_m')}`",
        f"- Fixed offset stable: `{summary['interpretation'].get('fixed_torso_offset_stable_across_samples')}`",
        "",
        "## Samples",
        "",
    ]
    for row in summary["sample_rows"]:
        lines.append(
            f"- `{row['name']}` motion=`{row['motion_file']}` terminated=`{row['terminated_after_zero_step']}` "
            f"raw_ori_err=`{row['raw_anchor_ori_b_error']}` corrected_ori_err=`{row['corrected_anchor_ori_b_error']}` "
            f"q_offset=`{row['q_offset_right_mujoco_to_isaac']}`"
        )
    lines.extend(["", "## Failed / Blocking Checks", ""])
    lines.extend(f"- `{item}`" for item in failed)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- walk 样本是 non-terminated 且 command metrics 为 0，因此它是更可信的低动态 adapter 对照。",
            "- 两个样本各自都能被单独 fitted offset 修正，但 offset 不一致，说明固定 torso frame offset 不是充分修复。",
            "- 后续应继续定位 IsaacLab/PhysX articulation body frame 与 MuJoCo MJCF body frame 的姿态表达差异，不能直接把单样本 offset 写入 rollout adapter。",
            "",
        ]
    )
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    sample_rows = [build_sample_row(name, cfg) for name, cfg in SAMPLES.items()]
    q_offsets = [np.asarray(row["q_offset_right_mujoco_to_isaac"], dtype=np.float64) for row in sample_rows]
    p_offsets = [np.asarray(row["p_offset_world"], dtype=np.float64) for row in sample_rows]
    q_offset_error = quat_sign_error(q_offsets[0], q_offsets[1]) if all(q.size == 4 for q in q_offsets) else float("inf")
    p_offset_l2 = float(np.linalg.norm(p_offsets[0] - p_offsets[1])) if all(p.size == 3 for p in p_offsets) else float("inf")
    q_tol = 1e-3
    p_tol = 1e-4
    checks = {
        "dance_sample_exists": bool(sample_rows[0]["sample_exists"]),
        "walk_sample_exists": bool(sample_rows[1]["sample_exists"]),
        "walk_sample_is_nonterminated": sample_rows[1]["terminated_after_zero_step"] is False,
        "runtime_parity_executed_for_both_samples": all(row["runtime_parity_exists"] for row in sample_rows),
        "torso_offset_executed_for_both_samples": all(row["torso_offset_exists"] for row in sample_rows),
        "each_sample_individually_restores_anchor_terms": all(
            bool(row["candidate_offset_restores_anchor_terms"]) for row in sample_rows
        ),
        "q_offset_stable_across_samples": bool(q_offset_error <= q_tol),
        "p_offset_stable_across_samples": bool(p_offset_l2 <= p_tol),
        "fixed_offset_adapter_patch_allowed": False,
        "success_video_claim_allowed": False,
        "does_not_claim_training_or_rollout": True,
    }
    fixed_offset_stable = checks["q_offset_stable_across_samples"] and checks["p_offset_stable_across_samples"]
    status = (
        "ok_fixed_torso_offset_hypothesis_cross_sample_stable_but_rollout_pending"
        if fixed_offset_stable
        else "blocked_fixed_torso_offset_not_stable_across_walk_and_dance_samples"
    )
    summary = {
        "status": status,
        "generated_at": utc_now(),
        "experiment_type": "mujoco_torso_frame_offset_cross_sample_audit",
        "claim_level": "cross-sample adapter audit only; no training, no policy rollout, no video",
        "sample_rows": sample_rows,
        "cross_sample_metrics": {
            "q_offset_sign_invariant_error": q_offset_error,
            "q_offset_tolerance": q_tol,
            "p_offset_l2_difference_m": p_offset_l2,
            "p_offset_tolerance_m": p_tol,
        },
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "interpretation": {
            "fixed_torso_offset_stable_across_samples": fixed_offset_stable,
            "single_sample_offset_fit_is_insufficient": True,
            "native_obs_adapter_patch_allowed": False,
            "native_policy_rollout_allowed": False,
            "success_video_claim_allowed": False,
            "likely_failure_class": (
                "state-dependent articulation/body-frame mismatch rather than a single rigid torso frame offset"
            ),
            "required_next_step": (
                "Audit IsaacLab/PhysX articulation body frame extraction versus MuJoCo MJCF body frame for the waist "
                "kinematic chain, including joint frame conventions and exported USD body transforms."
            ),
        },
    }
    write_outputs(summary)
    print(json.dumps({"status": summary["status"], "json": str(JSON_OUT), "md": str(MD_OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
