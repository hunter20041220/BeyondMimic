#!/usr/bin/env python3
"""Audit MuJoCo runtime observation slices against an IsaacLab sample.

The previous same-state parity audit proved that local NumPy formulas can
recompute the official IsaacLab observation slices from captured raw tensors.
This audit goes one layer closer to deployment: it loads the local MuJoCo G1
model, injects the captured IsaacLab root/joint state into MuJoCo qpos/qvel,
calls `mj_forward`, then builds the 160-D observation from MuJoCo runtime
body/joint state and compares every actor slice against the official
noise-free IsaacLab critic terms.

It still does not train, step a policy, render video, or claim a successful
MuJoCo rollout.  It is a pre-training/pre-video adapter gate.
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


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = Path(os.environ.get("BM_MUJOCO_OBS_RUNTIME_PARITY_OUT", ROOT / "res/audits/mujoco_observation_runtime_parity"))
JSON_OUT = OUT / "mujoco_observation_runtime_parity_audit.json"
TSV_OUT = OUT / "mujoco_observation_runtime_parity_audit.tsv"
MD_OUT = OUT / "mujoco_observation_runtime_parity_audit.md"

VENV_PYTHON = ROOT / "mujoco_mp4/.venv/bin/python"
MODEL_XML = ROOT / "mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_mocap_29dof.xml"
SAMPLE_JSON = Path(
    os.environ.get(
        "BM_MUJOCO_OBS_RUNTIME_PARITY_SAMPLE",
        ROOT / "res/audits/isaaclab_observation_manager_sample_gate/isaaclab_policy_obs_sample.json",
    )
)
ACTION_ADAPTER_JSON = (
    ROOT / "res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json"
)
CANDIDATE_MODEL_XMLS = [
    MODEL_XML,
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/assets/unitree_description/mjcf/g1.xml",
    ROOT / "mujoco_mp4/assets/work_g1/pbhc_g1/g1_29dof_rev_1_0.xml",
    ROOT / "mujoco_mp4/assets/work_g1/pbhc_g1/g1_29dof_rev_1_0_with_toe.xml",
    ROOT / "download/reference_code/mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml",
    ROOT / "download/reference_code/unitree_rl_mjlab/src/assets/robots/unitree_g1/xmls/g1.xml",
    ROOT / "download/reference_code/ASAP/humanoidverse/data/robots/g1/g1_29dof_old.xml",
]

BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
ANCHOR_BODY_NAME = "torso_link"
EXPECTED_TERMS = [
    "command",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def as_env0(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim >= 2 and arr.shape[0] == 1:
        return arr[0].copy()
    return arr.copy()


def as_flat(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def norm_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-12)


def qconj(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).copy()
    out[..., 1:] *= -1.0
    return out


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def qapply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = norm_quat(q)
    v = np.asarray(v, dtype=np.float64)
    vq = np.concatenate([np.zeros(v.shape[:-1] + (1,), dtype=np.float64), v], axis=-1)
    return qmul(qmul(q, vq), qconj(q))[..., 1:4]


def qmat(q: np.ndarray) -> np.ndarray:
    q = norm_quat(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def rot6(q: np.ndarray) -> np.ndarray:
    return qmat(q)[..., :2].reshape(-1)


def local_pos(parent_pos: np.ndarray, parent_quat: np.ndarray, child_pos: np.ndarray) -> np.ndarray:
    return qapply(qconj(parent_quat), child_pos - parent_pos).reshape(-1)


def local_quat(parent_quat: np.ndarray, child_quat: np.ndarray) -> np.ndarray:
    return norm_quat(qmul(qconj(parent_quat), child_quat))


def max_error(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def quat_sign_error(a: np.ndarray, b: np.ndarray) -> float:
    a = norm_quat(np.asarray(a, dtype=np.float64))
    b = norm_quat(np.asarray(b, dtype=np.float64))
    return float(min(np.max(np.abs(a - b)), np.max(np.abs(a + b))))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_outputs(summary: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(JSON_OUT, summary)
    with TSV_OUT.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["term", "dimension", "max_abs_error", "passed", "source", "notes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary.get("terms", []))
    failed = [key for key, value in summary.get("checks", {}).items() if not value]
    lines = [
        "# MuJoCo Observation Runtime Parity Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Generated: `{summary['generated_at']}`",
        "- Scope: load MuJoCo G1, inject captured IsaacLab state, call `mj_forward`, compare 160-D observation slices.",
        "- 当前不得声称完整复现 BeyondMimic；本审计不是训练、不是 rollout、不是视频成功。",
        "",
        "## Term Errors",
        "",
    ]
    for row in summary.get("terms", []):
        lines.append(
            f"- `{row['term']}` dim={row['dimension']} max_abs_error={float(row['max_abs_error']):.6e} "
            f"passed=`{row['passed']}` source=`{row['source']}`"
        )
    lines.extend(["", "## Failed / Blocking Checks", ""])
    if failed:
        lines.extend(f"- `{item}`" for item in failed)
    else:
        lines.append("- None for injected-state MuJoCo observation runtime parity. Policy rollout is still a separate gate.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- 这个 gate 比 pure formula parity 更强，因为它真的加载 MuJoCo model 并读取 `data.xpos/xquat/qpos/qvel`。",
            "- 但它仍然只是 injected-state adapter parity，没有执行 policy closed-loop、没有 `mj_step` 物理稳定性结论。",
            "- 如果该 gate 失败，当前前倾/不抬腿视频很可能仍由 MuJoCo body frame/default pose/joint order/velocity frame mismatch 导致。",
            "",
        ]
    )
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def fail_summary(status: str, error: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = {
        "status": status,
        "generated_at": utc_now(),
        "experiment_type": "mujoco_observation_runtime_parity_audit",
        "claim_level": "runtime adapter audit only; no training, no video, no rollout success claim",
        "files": {
            "venv_python": str(VENV_PYTHON),
            "model_xml": str(MODEL_XML),
            "sample_json": str(SAMPLE_JSON),
            "action_adapter_json": str(ACTION_ADAPTER_JSON),
        },
        "error": error,
        "terms": [],
        "checks": {
            "venv_python_exists": VENV_PYTHON.is_file(),
            "sample_json_available": SAMPLE_JSON.is_file(),
            "model_xml_available": MODEL_XML.is_file(),
            "mujoco_worker_executed": False,
            "does_not_claim_rollout_or_success": True,
        },
        "interpretation": {
            "mujoco_runtime_observation_parity_ready": False,
            "native_mujoco_policy_rollout_allowed": False,
        },
    }
    if extra:
        summary.update(extra)
    return summary


def load_action_joint_names() -> list[str]:
    payload = read_json(ACTION_ADAPTER_JSON)
    names = payload.get("joint_order", {}).get("action_joint_names", [])
    return [str(name) for name in names]


def candidate_model_frame_errors(
    *,
    mujoco: Any,
    raw: dict[str, Any],
    action_joint_names: list[str],
) -> list[dict[str, Any]]:
    robot_root_pos_w = as_env0(raw["robot_root_pos_w"])
    robot_root_quat_w = norm_quat(as_env0(raw["robot_root_quat_w"]))
    robot_root_lin_vel_w = as_env0(raw["robot_root_lin_vel_w"])
    robot_root_ang_vel_w = as_env0(raw["robot_root_ang_vel_w"])
    robot_joint_pos = as_env0(raw["robot_joint_pos"])
    robot_joint_vel = as_env0(raw["robot_joint_vel"])
    raw_anchor_pos = as_env0(raw["robot_anchor_pos_w"])
    raw_anchor_quat = norm_quat(as_env0(raw["robot_anchor_quat_w"]))
    rows: list[dict[str, Any]] = []
    for path in CANDIDATE_MODEL_XMLS:
        row: dict[str, Any] = {
            "model_xml": str(path),
            "exists": path.is_file(),
            "loaded": False,
            "joint_order_matches_action_order": False,
            "pelvis_position_error_m": None,
            "pelvis_quat_sign_invariant_error": None,
            "torso_position_error_m": None,
            "torso_quat_sign_invariant_error": None,
            "error": "",
        }
        if not path.is_file():
            rows.append(row)
            continue
        try:
            model = mujoco.MjModel.from_xml_path(str(path))
            data = mujoco.MjData(model)
            hinge_joint_ids = [
                jid for jid in range(model.njnt) if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
            ]
            joint_names = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or "" for jid in hinge_joint_ids[:29]
            ]
            data.qpos[:] = 0.0
            data.qvel[:] = 0.0
            data.qpos[0:3] = robot_root_pos_w
            data.qpos[3:7] = robot_root_quat_w
            for idx, jid in enumerate(hinge_joint_ids[:29]):
                data.qpos[model.jnt_qposadr[jid]] = robot_joint_pos[idx]
                data.qvel[model.jnt_dofadr[jid]] = robot_joint_vel[idx]
            data.qvel[0:3] = robot_root_lin_vel_w
            data.qvel[3:6] = robot_root_ang_vel_w
            mujoco.mj_forward(model, data)
            pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
            torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ANCHOR_BODY_NAME)
            row.update(
                {
                    "loaded": True,
                    "nbody": int(model.nbody),
                    "nq": int(model.nq),
                    "nv": int(model.nv),
                    "nu": int(model.nu),
                    "joint_order_matches_action_order": joint_names == action_joint_names,
                    "pelvis_position_error_m": max_error(data.xpos[pelvis_id], robot_root_pos_w)
                    if pelvis_id >= 0
                    else float("inf"),
                    "pelvis_quat_sign_invariant_error": quat_sign_error(data.xquat[pelvis_id], robot_root_quat_w)
                    if pelvis_id >= 0
                    else float("inf"),
                    "torso_position_error_m": max_error(data.xpos[torso_id], raw_anchor_pos)
                    if torso_id >= 0
                    else float("inf"),
                    "torso_quat_sign_invariant_error": quat_sign_error(data.xquat[torso_id], raw_anchor_quat)
                    if torso_id >= 0
                    else float("inf"),
                    "first_29_hinge_joint_names": joint_names,
                }
            )
        except Exception as exc:  # pragma: no cover - audit records all candidate failures.
            row["error"] = repr(exc)
        rows.append(row)
    return rows


def run_worker() -> dict[str, Any]:
    import mujoco

    sample = read_json(SAMPLE_JSON)
    action_joint_names = load_action_joint_names()
    if not sample:
        return fail_summary("failed_mujoco_runtime_observation_parity_no_sample", "missing sample json")
    if not MODEL_XML.is_file():
        return fail_summary("failed_mujoco_runtime_observation_parity_no_model_xml", "missing model xml")

    raw = sample.get("raw_state", {})
    critic_terms = sample.get("critic_terms", {})
    motion_file = Path(sample.get("motion_file", ""))
    motion_steps = sample.get("motion_time_steps") or []
    time_step = int(motion_steps[0]) if motion_steps else -1
    body_indexes = [int(x) for x in sample.get("body_indexes", [])]
    motion_anchor_body_index = int(sample.get("motion_anchor_body_index", -1))
    motion_raw_anchor_index = body_indexes[motion_anchor_body_index] if 0 <= motion_anchor_body_index < len(body_indexes) else -1

    model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
    data = mujoco.MjData(model)
    mujoco_joint_names: list[str] = []
    hinge_joint_ids: list[int] = []
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
            hinge_joint_ids.append(jid)
            mujoco_joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or "")
    body_ids: dict[str, int] = {}
    missing_bodies: list[str] = []
    for name in BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            missing_bodies.append(name)
        else:
            body_ids[name] = int(bid)

    robot_root_pos_w = as_env0(raw["robot_root_pos_w"])
    robot_root_quat_w = norm_quat(as_env0(raw["robot_root_quat_w"]))
    robot_root_lin_vel_w = as_env0(raw["robot_root_lin_vel_w"])
    robot_root_ang_vel_w = as_env0(raw["robot_root_ang_vel_w"])
    robot_joint_pos = as_env0(raw["robot_joint_pos"])
    robot_default_joint_pos = as_env0(raw["robot_default_joint_pos"])
    robot_joint_vel = as_env0(raw["robot_joint_vel"])
    zero_action = as_env0(raw["zero_action"])

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = robot_root_pos_w
    data.qpos[3:7] = robot_root_quat_w
    for idx, jid in enumerate(hinge_joint_ids[:29]):
        data.qpos[model.jnt_qposadr[jid]] = robot_joint_pos[idx]
        data.qvel[model.jnt_dofadr[jid]] = robot_joint_vel[idx]
    data.qvel[0:3] = robot_root_lin_vel_w
    data.qvel[3:6] = robot_root_ang_vel_w
    mujoco.mj_forward(model, data)

    if not motion_file.is_file():
        return fail_summary("failed_mujoco_runtime_observation_parity_no_motion_file", f"missing {motion_file}")
    motion = np.load(motion_file)
    motion_in_range = 0 <= time_step < int(motion["joint_pos"].shape[0])
    if not motion_in_range:
        return fail_summary("failed_mujoco_runtime_observation_parity_bad_time_step", f"bad time_step={time_step}")
    motion_joint_pos = np.asarray(motion["joint_pos"][time_step], dtype=np.float64)
    motion_joint_vel = np.asarray(motion["joint_vel"][time_step], dtype=np.float64)
    motion_anchor_pos_w = np.asarray(motion["body_pos_w"][time_step, motion_raw_anchor_index], dtype=np.float64)
    motion_anchor_quat_w = norm_quat(np.asarray(motion["body_quat_w"][time_step, motion_raw_anchor_index], dtype=np.float64))

    anchor_id = body_ids.get(ANCHOR_BODY_NAME, -1)
    robot_anchor_pos_mj = data.xpos[anchor_id].copy() if anchor_id >= 0 else np.full(3, np.nan)
    robot_anchor_quat_mj = norm_quat(data.xquat[anchor_id].copy()) if anchor_id >= 0 else np.full(4, np.nan)
    root_quat_mj = norm_quat(data.xquat[body_ids["pelvis"]].copy()) if "pelvis" in body_ids else robot_root_quat_w

    runtime_terms = {
        "command": np.concatenate([motion_joint_pos, motion_joint_vel]),
        "motion_anchor_pos_b": local_pos(robot_anchor_pos_mj, robot_anchor_quat_mj, motion_anchor_pos_w),
        "motion_anchor_ori_b": rot6(local_quat(robot_anchor_quat_mj, motion_anchor_quat_w)),
        "base_lin_vel": qapply(qconj(root_quat_mj), data.qvel[0:3].copy()).reshape(-1),
        "base_ang_vel": qapply(qconj(root_quat_mj), data.qvel[3:6].copy()).reshape(-1),
        "joint_pos": data.qpos[7 : 7 + 29].copy() - robot_default_joint_pos,
        "joint_vel": data.qvel[6 : 6 + 29].copy(),
        "actions": zero_action.reshape(-1),
    }

    terms: list[dict[str, Any]] = []
    tol = 1e-5
    for name in EXPECTED_TERMS:
        official = as_flat(critic_terms.get(name, []))
        local = as_flat(runtime_terms.get(name, []))
        err = max_error(local, official)
        terms.append(
            {
                "term": name,
                "dimension": int(local.size) if local.size else int(official.size),
                "max_abs_error": err,
                "passed": bool(np.isfinite(err) and err <= tol),
                "source": "mujoco_runtime_injected_state" if name not in {"command", "actions"} else "motion_file_or_last_action",
                "notes": "Compared to official IsaacLab critic/noise-free shared term.",
            }
        )

    raw_robot_anchor_pos = as_env0(raw["robot_anchor_pos_w"])
    raw_robot_anchor_quat = norm_quat(as_env0(raw["robot_anchor_quat_w"]))
    raw_command_anchor_pos = as_env0(raw["command_anchor_pos_w"])
    raw_command_anchor_quat = norm_quat(as_env0(raw["command_anchor_quat_w"]))
    diagnostics = {
        "mujoco_version": mujoco.__version__,
        "model_dims": {"nq": int(model.nq), "nv": int(model.nv), "nbody": int(model.nbody), "njnt": int(model.njnt), "nu": int(model.nu)},
        "time_step": time_step,
        "motion_raw_anchor_index": motion_raw_anchor_index,
        "mujoco_joint_names": mujoco_joint_names,
        "action_joint_names": action_joint_names,
        "missing_required_bodies": missing_bodies,
        "anchor_world_pose_error": {
            "position_m": max_error(robot_anchor_pos_mj, raw_robot_anchor_pos),
            "quat_sign_invariant": quat_sign_error(robot_anchor_quat_mj, raw_robot_anchor_quat),
        },
        "root_world_pose_error": {
            "position_m": max_error(data.xpos[body_ids["pelvis"]], robot_root_pos_w) if "pelvis" in body_ids else float("inf"),
            "quat_sign_invariant": quat_sign_error(root_quat_mj, robot_root_quat_w),
        },
        "motion_anchor_from_file_vs_sample": {
            "position_m": max_error(motion_anchor_pos_w, raw_command_anchor_pos),
            "quat_sign_invariant": quat_sign_error(motion_anchor_quat_w, raw_command_anchor_quat),
        },
        "command_from_file_vs_sample": {
            "joint_pos": max_error(motion_joint_pos, as_env0(raw["command_joint_pos"])),
            "joint_vel": max_error(motion_joint_vel, as_env0(raw["command_joint_vel"])),
        },
        "base_velocity_from_qvel_vs_raw_body_frame": {
            "lin": max_error(runtime_terms["base_lin_vel"], as_env0(raw["robot_root_lin_vel_b"])),
            "ang": max_error(runtime_terms["base_ang_vel"], as_env0(raw["robot_root_ang_vel_b"])),
        },
        "candidate_model_frame_errors": candidate_model_frame_errors(
            mujoco=mujoco, raw=raw, action_joint_names=action_joint_names
        ),
    }

    all_terms_pass = bool(terms) and all(row["passed"] for row in terms)
    best_torso_quat_error = min(
        [
            float(row["torso_quat_sign_invariant_error"])
            for row in diagnostics["candidate_model_frame_errors"]
            if row.get("loaded") and row.get("torso_quat_sign_invariant_error") is not None
        ]
        or [float("inf")]
    )
    checks = {
        "venv_python_exists": VENV_PYTHON.is_file(),
        "sample_json_available": SAMPLE_JSON.is_file(),
        "model_xml_available": MODEL_XML.is_file(),
        "mujoco_import_ok": True,
        "mujoco_model_loaded": True,
        "mujoco_worker_executed": True,
        "mujoco_joint_order_matches_action_order": mujoco_joint_names == action_joint_names,
        "required_body_names_present": not missing_bodies,
        "motion_file_available": motion_file.is_file(),
        "motion_time_step_in_range": motion_in_range,
        "motion_anchor_from_file_matches_sample": bool(
            diagnostics["motion_anchor_from_file_vs_sample"]["position_m"] <= tol
            and diagnostics["motion_anchor_from_file_vs_sample"]["quat_sign_invariant"] <= tol
        ),
        "command_from_motion_file_matches_sample": bool(
            diagnostics["command_from_file_vs_sample"]["joint_pos"] <= tol
            and diagnostics["command_from_file_vs_sample"]["joint_vel"] <= tol
        ),
        "mujoco_root_pose_matches_injected_sample": bool(
            diagnostics["root_world_pose_error"]["position_m"] <= tol
            and diagnostics["root_world_pose_error"]["quat_sign_invariant"] <= tol
        ),
        "mujoco_anchor_pose_matches_isaaclab_sample": bool(
            diagnostics["anchor_world_pose_error"]["position_m"] <= tol
            and diagnostics["anchor_world_pose_error"]["quat_sign_invariant"] <= tol
        ),
        "candidate_mujoco_models_any_anchor_orientation_matches_isaaclab": bool(best_torso_quat_error <= tol),
        "all_runtime_observation_slices_pass": all_terms_pass,
        "mujoco_runtime_builder_executed": True,
        "does_not_step_policy_or_train": True,
        "does_not_claim_rollout_or_success": True,
    }
    status = (
        "ok_mujoco_injected_state_observation_runtime_parity_passed_but_rollout_pending"
        if all_terms_pass
        else "blocked_mujoco_injected_state_observation_runtime_parity_mismatch"
    )
    return {
        "status": status,
        "generated_at": utc_now(),
        "experiment_type": "mujoco_observation_runtime_parity_audit",
        "claim_level": "MuJoCo injected-state observation adapter audit only; no policy rollout, no training, no video",
        "files": {
            "venv_python": str(VENV_PYTHON),
            "model_xml": str(MODEL_XML),
            "sample_json": str(SAMPLE_JSON),
            "motion_file": str(motion_file),
            "action_adapter_json": str(ACTION_ADAPTER_JSON),
        },
        "tolerance": tol,
        "terms": terms,
        "diagnostics": diagnostics,
        "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value],
        "interpretation": {
            "mujoco_runtime_observation_parity_ready": all_terms_pass,
            "native_mujoco_policy_rollout_allowed": False,
            "policy_rollout_still_requires": [
                "resolve MuJoCo MJCF/IsaacLab USD anchor body frame mismatch",
                "normalizer-loaded actor integration",
                "deployment controller frame parity",
                "mj_step closed-loop no-root-assist stability gate",
                "teacher/VAE/diffusion quality gates",
            ],
            "why_this_matters": (
                "If MuJoCo body kinematics or velocity-frame extraction disagrees with the official IsaacLab sample, "
                "the actor can see a false target error and converge to a leaning stabilizing pose rather than lifting "
                "legs or walking."
            ),
        },
    }


def main() -> None:
    if "--worker" not in sys.argv:
        if not VENV_PYTHON.is_file():
            summary = fail_summary("failed_mujoco_runtime_observation_parity_missing_venv", f"missing {VENV_PYTHON}")
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
                "failed_mujoco_runtime_observation_parity_worker_error",
                proc.stdout[-4000:],
                {"worker_returncode": proc.returncode},
            )
            write_outputs(summary)
        print(proc.stdout, end="")
        if proc.returncode != 0:
            print(json.dumps({"status": "failed_mujoco_runtime_observation_parity_worker_error", "json": str(JSON_OUT)}, sort_keys=True))
        return

    summary = run_worker()
    write_outputs(summary)
    print(json.dumps({"status": summary["status"], "json": str(JSON_OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
