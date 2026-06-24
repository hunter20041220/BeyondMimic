#!/usr/bin/env python3
"""Audit a MuJoCo-to-IsaacLab torso frame offset hypothesis.

The injected-state runtime parity audit showed that root pose, base velocity,
joint order, joint position, joint velocity, command and last-action slices can
match the captured IsaacLab sample, while `torso_link` / motion-anchor pose does
not.  This script quantifies whether a rigid local offset between the MuJoCo
`torso_link` frame and the IsaacLab importer/exported `torso_link` frame can
explain the mismatch.

This is deliberately not a rollout, not training, and not a rollout-video
success gate.  The current IsaacLab sample was captured from a terminated
dance-state sample, so any offset inferred here is a hypothesis that must be
revalidated on a non-terminated walk/single-leg sample before being used by a
native MuJoCo policy adapter.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduction.scripts.mujoco_observation_runtime_parity_audit import (
    ACTION_ADAPTER_JSON,
    ANCHOR_BODY_NAME,
    CANDIDATE_MODEL_XMLS,
    MODEL_XML,
    ROOT,
    SAMPLE_JSON,
    VENV_PYTHON,
    as_env0,
    as_flat,
    local_pos,
    local_quat,
    max_error,
    norm_quat,
    qmul,
    quat_sign_error,
    read_json,
    rot6,
)


OUT = ROOT / "res/audits/mujoco_torso_frame_offset"
JSON_OUT = OUT / "mujoco_torso_frame_offset_audit.json"
TSV_OUT = OUT / "mujoco_torso_frame_offset_audit.tsv"
MD_OUT = OUT / "mujoco_torso_frame_offset_audit.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_action_joint_names() -> list[str]:
    payload = read_json(ACTION_ADAPTER_JSON)
    return [str(name) for name in payload.get("joint_order", {}).get("action_joint_names", [])]


def fail_summary(status: str, error: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = {
        "status": status,
        "generated_at": utc_now(),
        "experiment_type": "mujoco_torso_frame_offset_audit",
        "claim_level": "single-sample frame-offset hypothesis only; no rollout, no training, no success-video claim",
        "files": {
            "venv_python": str(VENV_PYTHON),
            "sample_json": str(SAMPLE_JSON),
            "primary_model_xml": str(MODEL_XML),
            "action_adapter_json": str(ACTION_ADAPTER_JSON),
        },
        "error": error,
        "checks": {
            "venv_python_exists": VENV_PYTHON.is_file(),
            "sample_json_available": SAMPLE_JSON.is_file(),
            "primary_model_xml_available": MODEL_XML.is_file(),
            "mujoco_worker_executed": False,
            "does_not_claim_rollout_or_training": True,
        },
        "interpretation": {
            "torso_frame_offset_hypothesis_supported": False,
            "native_obs_adapter_patch_allowed": False,
            "success_video_claim_allowed": False,
        },
    }
    if extra:
        summary.update(extra)
    return summary


def build_model_row(
    *,
    mujoco: Any,
    model_xml: Path,
    sample: dict[str, Any],
    action_joint_names: list[str],
    tol: float,
) -> dict[str, Any]:
    raw = sample["raw_state"]
    critic_terms = sample["critic_terms"]
    model_row: dict[str, Any] = {
        "model_xml": str(model_xml),
        "exists": model_xml.is_file(),
        "loaded": False,
        "joint_order_matches_action_order": False,
        "raw_anchor_pos_b_error": None,
        "raw_anchor_ori_b_error": None,
        "corrected_anchor_pos_b_error": None,
        "corrected_anchor_ori_b_error": None,
        "candidate_offset_restores_anchor_terms": False,
        "error": "",
    }
    if not model_xml.is_file():
        return model_row

    try:
        model = mujoco.MjModel.from_xml_path(str(model_xml))
        data = mujoco.MjData(model)
        hinge_joint_ids = [jid for jid in range(model.njnt) if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE]
        joint_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or "" for jid in hinge_joint_ids[:29]
        ]
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ANCHOR_BODY_NAME)
        if body_id < 0:
            model_row["error"] = f"missing body {ANCHOR_BODY_NAME}"
            return model_row

        robot_root_pos_w = as_env0(raw["robot_root_pos_w"])
        robot_root_quat_w = norm_quat(as_env0(raw["robot_root_quat_w"]))
        robot_root_lin_vel_w = as_env0(raw["robot_root_lin_vel_w"])
        robot_root_ang_vel_w = as_env0(raw["robot_root_ang_vel_w"])
        robot_joint_pos = as_env0(raw["robot_joint_pos"])
        robot_joint_vel = as_env0(raw["robot_joint_vel"])

        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[0:3] = robot_root_pos_w
        data.qpos[3:7] = robot_root_quat_w
        data.qvel[0:3] = robot_root_lin_vel_w
        data.qvel[3:6] = robot_root_ang_vel_w
        for idx, jid in enumerate(hinge_joint_ids[:29]):
            data.qpos[model.jnt_qposadr[jid]] = robot_joint_pos[idx]
            data.qvel[model.jnt_dofadr[jid]] = robot_joint_vel[idx]
        mujoco.mj_forward(model, data)

        mujoco_anchor_pos_w = data.xpos[body_id].copy()
        mujoco_anchor_quat_w = norm_quat(data.xquat[body_id].copy())
        isaac_anchor_pos_w = as_env0(raw["robot_anchor_pos_w"])
        isaac_anchor_quat_w = norm_quat(as_env0(raw["robot_anchor_quat_w"]))
        motion_anchor_pos_w = as_env0(raw["command_anchor_pos_w"])
        motion_anchor_quat_w = norm_quat(as_env0(raw["command_anchor_quat_w"]))

        q_offset_right = local_quat(mujoco_anchor_quat_w, isaac_anchor_quat_w)
        p_offset_world = isaac_anchor_pos_w - mujoco_anchor_pos_w
        corrected_anchor_pos_w = mujoco_anchor_pos_w + p_offset_world
        corrected_anchor_quat_w = norm_quat(qmul(mujoco_anchor_quat_w, q_offset_right))

        raw_pos_b = local_pos(mujoco_anchor_pos_w, mujoco_anchor_quat_w, motion_anchor_pos_w)
        raw_ori_b = rot6(local_quat(mujoco_anchor_quat_w, motion_anchor_quat_w))
        corrected_pos_b = local_pos(corrected_anchor_pos_w, corrected_anchor_quat_w, motion_anchor_pos_w)
        corrected_ori_b = rot6(local_quat(corrected_anchor_quat_w, motion_anchor_quat_w))
        official_pos_b = as_flat(critic_terms["motion_anchor_pos_b"])
        official_ori_b = as_flat(critic_terms["motion_anchor_ori_b"])

        raw_pos_error = max_error(raw_pos_b, official_pos_b)
        raw_ori_error = max_error(raw_ori_b, official_ori_b)
        corrected_pos_error = max_error(corrected_pos_b, official_pos_b)
        corrected_ori_error = max_error(corrected_ori_b, official_ori_b)

        model_row.update(
            {
                "loaded": True,
                "nq": int(model.nq),
                "nv": int(model.nv),
                "nu": int(model.nu),
                "nbody": int(model.nbody),
                "joint_order_matches_action_order": joint_names == action_joint_names,
                "first_29_hinge_joint_names": joint_names,
                "mujoco_anchor_pos_w": mujoco_anchor_pos_w.tolist(),
                "mujoco_anchor_quat_w": mujoco_anchor_quat_w.tolist(),
                "isaac_anchor_pos_w": isaac_anchor_pos_w.tolist(),
                "isaac_anchor_quat_w": isaac_anchor_quat_w.tolist(),
                "p_offset_world": p_offset_world.tolist(),
                "q_offset_right_mujoco_to_isaac": q_offset_right.tolist(),
                "mujoco_to_isaac_anchor_position_error_m": max_error(mujoco_anchor_pos_w, isaac_anchor_pos_w),
                "mujoco_to_isaac_anchor_quat_sign_invariant_error": quat_sign_error(
                    mujoco_anchor_quat_w, isaac_anchor_quat_w
                ),
                "raw_anchor_pos_b": raw_pos_b.tolist(),
                "raw_anchor_ori_b": raw_ori_b.tolist(),
                "corrected_anchor_pos_b": corrected_pos_b.tolist(),
                "corrected_anchor_ori_b": corrected_ori_b.tolist(),
                "official_anchor_pos_b": official_pos_b.tolist(),
                "official_anchor_ori_b": official_ori_b.tolist(),
                "raw_anchor_pos_b_error": raw_pos_error,
                "raw_anchor_ori_b_error": raw_ori_error,
                "corrected_anchor_pos_b_error": corrected_pos_error,
                "corrected_anchor_ori_b_error": corrected_ori_error,
                "candidate_offset_restores_anchor_terms": bool(
                    corrected_pos_error <= tol and corrected_ori_error <= tol
                ),
            }
        )
    except Exception as exc:  # pragma: no cover - audit keeps all failures.
        model_row["error"] = repr(exc)
    return model_row


def run_worker() -> dict[str, Any]:
    import mujoco

    sample = read_json(SAMPLE_JSON)
    if not sample:
        return fail_summary("failed_mujoco_torso_frame_offset_no_sample", f"missing {SAMPLE_JSON}")
    action_joint_names = load_action_joint_names()
    tol = 1e-5
    model_rows = [
        build_model_row(
            mujoco=mujoco,
            model_xml=path,
            sample=sample,
            action_joint_names=action_joint_names,
            tol=tol,
        )
        for path in CANDIDATE_MODEL_XMLS
    ]
    primary = next((row for row in model_rows if row["model_xml"] == str(MODEL_XML)), model_rows[0])
    any_model_restores = any(row.get("candidate_offset_restores_anchor_terms") for row in model_rows)
    all_loaded_joint_order = all(
        row.get("joint_order_matches_action_order") for row in model_rows if row.get("loaded") and row.get("nu", 0)
    )
    sample_motion_file = str(sample.get("motion_file", ""))
    sample_terminated = bool(sample.get("terminated_after_zero_step"))
    sample_is_walk = "walk" in sample_motion_file.lower()
    checks = {
        "venv_python_exists": VENV_PYTHON.is_file(),
        "sample_json_available": SAMPLE_JSON.is_file(),
        "primary_model_xml_available": MODEL_XML.is_file(),
        "mujoco_worker_executed": True,
        "primary_model_loaded": bool(primary.get("loaded")),
        "primary_joint_order_matches_action_order": bool(primary.get("joint_order_matches_action_order")),
        "candidate_offset_restores_primary_anchor_terms": bool(
            primary.get("candidate_offset_restores_anchor_terms")
        ),
        "candidate_offset_restores_any_model_anchor_terms": bool(any_model_restores),
        "loaded_actuated_models_have_matching_joint_order": bool(all_loaded_joint_order),
        "sample_is_nonterminated": not sample_terminated,
        "sample_motion_is_walk_or_low_dynamic": sample_is_walk,
        "independent_nonterminated_walk_sample_available": False,
        "offset_validated_across_independent_walk_sample": False,
        "does_not_claim_rollout_or_training": True,
        "does_not_patch_adapter_from_single_sample": True,
    }
    status = (
        "blocked_torso_frame_offset_hypothesis_single_terminated_sample_requires_walk_validation"
        if any_model_restores
        else "failed_torso_frame_offset_hypothesis_not_supported"
    )
    return {
        "status": status,
        "generated_at": utc_now(),
        "experiment_type": "mujoco_torso_frame_offset_audit",
        "claim_level": "single-sample MuJoCo/IsaacLab torso frame offset hypothesis only; no rollout, no training",
        "files": {
            "venv_python": str(VENV_PYTHON),
            "sample_json": str(SAMPLE_JSON),
            "primary_model_xml": str(MODEL_XML),
            "action_adapter_json": str(ACTION_ADAPTER_JSON),
        },
        "tolerance": tol,
        "sample": {
            "status": sample.get("status"),
            "motion_file": sample.get("motion_file"),
            "motion_time_steps": sample.get("motion_time_steps"),
            "reward_after_zero_step": sample.get("reward_after_zero_step"),
            "terminated_after_zero_step": sample.get("terminated_after_zero_step"),
            "truncated_after_zero_step": sample.get("truncated_after_zero_step"),
            "policy_obs_dim": sample.get("policy_obs_dim"),
            "policy_term_names": sample.get("policy_term_names"),
            "robot_anchor_body_index": sample.get("robot_anchor_body_index"),
            "motion_anchor_body_index": sample.get("motion_anchor_body_index"),
        },
        "primary_model_result": primary,
        "model_results": model_rows,
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "interpretation": {
            "torso_frame_offset_hypothesis_supported": bool(any_model_restores),
            "primary_offset_restores_anchor_observation_terms": bool(
                primary.get("candidate_offset_restores_anchor_terms")
            ),
            "sample_quality_blocks_patch_claim": sample_terminated or not sample_is_walk,
            "native_obs_adapter_patch_allowed": False,
            "native_policy_rollout_allowed": False,
            "success_video_claim_allowed": False,
            "why_this_matters": (
                "The current MuJoCo policy/video failures can be caused before learning quality is tested: "
                "a dimension-correct actor input can encode the wrong anchor pose if the MuJoCo torso frame is "
                "not converted to the IsaacLab importer/exported frame used during PPO training."
            ),
            "required_next_step": (
                "Capture a non-terminated IsaacLab observation_manager sample on a low-dynamic walk motion, then "
                "verify the same MuJoCo-to-IsaacLab torso frame offset across that independent sample before using "
                "the correction inside any native MuJoCo PPO/VAE/diffusion rollout."
            ),
        },
    }


def write_outputs(summary: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(JSON_OUT, summary)
    with TSV_OUT.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "model_xml",
            "loaded",
            "joint_order_matches_action_order",
            "raw_anchor_pos_b_error",
            "raw_anchor_ori_b_error",
            "corrected_anchor_pos_b_error",
            "corrected_anchor_ori_b_error",
            "candidate_offset_restores_anchor_terms",
            "q_offset_right_mujoco_to_isaac",
            "p_offset_world",
            "error",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in summary.get("model_results", []):
            writer.writerow(
                {
                    "model_xml": row.get("model_xml"),
                    "loaded": row.get("loaded"),
                    "joint_order_matches_action_order": row.get("joint_order_matches_action_order"),
                    "raw_anchor_pos_b_error": row.get("raw_anchor_pos_b_error"),
                    "raw_anchor_ori_b_error": row.get("raw_anchor_ori_b_error"),
                    "corrected_anchor_pos_b_error": row.get("corrected_anchor_pos_b_error"),
                    "corrected_anchor_ori_b_error": row.get("corrected_anchor_ori_b_error"),
                    "candidate_offset_restores_anchor_terms": row.get("candidate_offset_restores_anchor_terms"),
                    "q_offset_right_mujoco_to_isaac": json.dumps(
                        row.get("q_offset_right_mujoco_to_isaac", []), separators=(",", ":")
                    ),
                    "p_offset_world": json.dumps(row.get("p_offset_world", []), separators=(",", ":")),
                    "error": row.get("error") or "none",
                }
            )
    failed = [key for key, value in summary.get("checks", {}).items() if not value]
    primary = summary.get("primary_model_result", {})
    lines = [
        "# MuJoCo Torso Frame Offset Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Generated: `{summary['generated_at']}`",
        "- Scope: single-sample MuJoCo/IsaacLab torso frame offset hypothesis; no training, no rollout, no video.",
        "- 当前不得声称完整复现 BeyondMimic；本审计不能放行 native MuJoCo PPO/VAE/diffusion 视频。",
        "",
        "## Primary Result",
        "",
        f"- Primary model: `{primary.get('model_xml')}`",
        f"- Raw `motion_anchor_pos_b` error: `{primary.get('raw_anchor_pos_b_error')}`",
        f"- Raw `motion_anchor_ori_b` error: `{primary.get('raw_anchor_ori_b_error')}`",
        f"- Corrected `motion_anchor_pos_b` error: `{primary.get('corrected_anchor_pos_b_error')}`",
        f"- Corrected `motion_anchor_ori_b` error: `{primary.get('corrected_anchor_ori_b_error')}`",
        f"- Candidate right-multiplied quaternion offset: `{primary.get('q_offset_right_mujoco_to_isaac')}`",
        f"- Candidate world position offset: `{primary.get('p_offset_world')}`",
        "",
        "## Sample Quality",
        "",
        f"- Motion file: `{summary.get('sample', {}).get('motion_file')}`",
        f"- Motion time steps: `{summary.get('sample', {}).get('motion_time_steps')}`",
        f"- Reward after zero step: `{summary.get('sample', {}).get('reward_after_zero_step')}`",
        f"- Terminated after zero step: `{summary.get('sample', {}).get('terminated_after_zero_step')}`",
        "",
        "## Failed / Blocking Checks",
        "",
    ]
    if failed:
        lines.extend(f"- `{item}`" for item in failed)
    else:
        lines.append("- None for this single sample; independent rollout validation is still pending.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- 该审计支持一个很具体的失败假设：MuJoCo 的 `torso_link` frame 与 IsaacLab importer/exported `torso_link` frame 不一致。",
            "- 在当前样本上，右乘候选四元数 offset 后 anchor orientation term 从大误差恢复到数值一致。",
            "- 但当前样本来自 terminated dance state，因此该 offset 不能直接写入最终 adapter。",
            "- 下一步必须抓取 non-terminated walk/single-leg IsaacLab observation sample，并验证同一个 offset 是否仍成立。",
            "",
        ]
    )
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if "--worker" not in sys.argv:
        if not VENV_PYTHON.is_file():
            summary = fail_summary("failed_mujoco_torso_frame_offset_missing_venv", f"missing {VENV_PYTHON}")
            write_outputs(summary)
            print(json.dumps({"status": summary["status"], "json": str(JSON_OUT)}, sort_keys=True))
            return
        env = os.environ.copy()
        env.setdefault("MUJOCO_GL", "egl")
        env["PYTHONNOUSERSITE"] = "1"
        proc = subprocess.run(
            [str(VENV_PYTHON), str(Path(__file__).resolve()), "--worker"],
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
            check=False,
        )
        if proc.returncode != 0:
            summary = fail_summary(
                "failed_mujoco_torso_frame_offset_worker_error",
                proc.stdout[-4000:],
                {"worker_returncode": proc.returncode},
            )
            write_outputs(summary)
        print(proc.stdout, end="")
        if proc.returncode != 0:
            print(json.dumps({"status": "failed_mujoco_torso_frame_offset_worker_error", "json": str(JSON_OUT)}, sort_keys=True))
        return

    summary = run_worker()
    write_outputs(summary)
    print(json.dumps({"status": summary["status"], "json": str(JSON_OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
