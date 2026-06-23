#!/usr/bin/env python3
"""Probe a MuJoCo-side PPO observation/action adapter on the quality-gated segment.

The previous quality-gated videos fixed the bad near-floor reference target,
but the teacher video still replayed stored IsaacLab actions open-loop.  That
is not the same as closed-loop motion control.  This script builds an
approximate 160-D observation from the current MuJoCo state and the same
reference motion, loads the local Stage-1 PPO actor and empirical normalizer,
and runs a short receding closed-loop rollout:

    MuJoCo state -> approximate IsaacLab policy obs -> PPO actor -> action
    -> theta_default + action_scale * action -> MuJoCo PD position actuators.

Claim boundary: this is a local adapter probe.  It is not official
BeyondMimic, not IsaacLab physics, not real robot, and not paper-level Fig. 5/6.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from torch import nn

ROOT = Path(os.environ.get("BM_ROOT", "/mnt/infini-data/test/BeyondMimic")).resolve()
SCRIPT_DIR = ROOT / "reproduction/scripts"
MUJOCO_SCRIPT_DIR = ROOT / "mujoco_mp4/scripts"
for path in [SCRIPT_DIR, MUJOCO_SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import stage1_multisource_quality_gated_mujoco_action_control_videos as qg  # noqa: E402
from lafan1_paper_contract_mujoco_action_control_videos import (  # noqa: E402
    action_to_joint_targets,
    make_keyframe_strip,
    parse_default_joint_position,
    sha256,
    utc_now,
    write_json,
)
from mujoco_common import render_frame  # noqa: E402
from mujoco_pd_control_video import (  # noqa: E402
    PD_CAMERA,
    actuator_joint_order,
    apply_root_assist,
    load_action_rows,
    normalize_quat_wxyz,
    patch_joints_and_actuators,
    quat_conj,
    quat_mul,
    quat_to_roll_pitch_yaw,
)
from mujoco_trace_mesh_video import DEFAULT_MODEL  # noqa: E402


OUT_ROOT = ROOT / "res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos"
OUT_DIR = OUT_ROOT / "native_ppo_obs_adapter_probe"
OUT_SUMMARY = OUT_DIR / "native_ppo_obs_adapter_probe_summary.json"
OUT_METRICS = OUT_DIR / "native_ppo_obs_adapter_probe_metrics.csv"
OUT_MP4 = OUT_DIR / "native_ppo_obs_adapter_probe.mp4"
OUT_KEYFRAME = OUT_DIR / "native_ppo_obs_adapter_probe_keyframe.png"
OUT_KEYFRAMES = OUT_DIR / "native_ppo_obs_adapter_probe_keyframes.png"
OUT_COMPARE = OUT_ROOT / "quality_gated_stage1_multisource_native_adapter_comparison.json"
OUT_COMPARE_MD = OUT_ROOT / "quality_gated_stage1_multisource_native_adapter_comparison.md"

BEST_TEACHER_JSON = (
    ROOT
    / "res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/"
    "stage1_multisource_best_teacher.json"
)
MOTION_BUNDLE_NPZ = (
    ROOT
    / "res/tracking/stage1_multisource_motion_bundle/"
    "stage1_multisource_public_plus_available_motion_bundle_fk_repaired_robot_order.npz"
)

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


class Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(160, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 29),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


@dataclass
class LoadedPolicy:
    checkpoint: Path
    actor: Actor
    obs_mean: torch.Tensor
    obs_std: torch.Tensor
    obs_eps: float = 1e-2

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
        norm = (obs_t - self.obs_mean) / (self.obs_std + self.obs_eps)
        with torch.inference_mode():
            action = self.actor(norm)
        return action.squeeze(0).cpu().numpy().astype(np.float64)


def load_policy() -> LoadedPolicy:
    best = json.loads(BEST_TEACHER_JSON.read_text(encoding="utf-8"))
    checkpoint = Path(best["best_checkpoint"]["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu")
    actor = Actor()
    actor_state = {key: value for key, value in payload["model_state_dict"].items() if key.startswith("actor.")}
    actor.load_state_dict(actor_state)
    actor.eval()
    obs_norm = payload["obs_norm_state_dict"]
    return LoadedPolicy(
        checkpoint=checkpoint,
        actor=actor,
        obs_mean=obs_norm["_mean"],
        obs_std=obs_norm["_std"],
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = normalize_quat_wxyz(np.asarray(q, dtype=np.float64))
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rot6(q: np.ndarray) -> np.ndarray:
    mat = quat_to_matrix(q)
    return mat[:, :2].reshape(-1)


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.concatenate([[0.0], np.asarray(v, dtype=np.float64)])
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:4]


def local_pos(anchor_pos: np.ndarray, anchor_quat: np.ndarray, pos: np.ndarray) -> np.ndarray:
    return quat_apply(quat_conj(anchor_quat), pos - anchor_pos)


def local_quat(anchor_quat: np.ndarray, quat: np.ndarray) -> np.ndarray:
    q = quat_mul(quat_conj(anchor_quat), quat)
    if q[0] < 0:
        q = -q
    return normalize_quat_wxyz(q)


def body_id_map(model) -> dict[str, int]:
    import mujoco

    out = {}
    for name in BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"MuJoCo body missing: {name}")
        out[name] = int(bid)
    return out


def ref_arrays(time_steps: np.ndarray) -> dict[str, np.ndarray]:
    motion = np.load(MOTION_BUNDLE_NPZ, allow_pickle=True)
    total = int(motion["joint_pos"].shape[0])
    steps = np.clip(np.asarray(time_steps, dtype=np.int64), 0, total - 1)
    out = {
        "time_steps": steps,
        "joint_pos": np.asarray(motion["joint_pos"][steps], dtype=np.float64),
        "joint_vel": np.asarray(motion["joint_vel"][steps], dtype=np.float64),
        "body_pos_w": np.asarray(motion["body_pos_w"][steps], dtype=np.float64),
        "body_quat_w": np.asarray(motion["body_quat_w"][steps], dtype=np.float64),
        "body_lin_vel_w": np.asarray(motion["body_lin_vel_w"][steps], dtype=np.float64),
        "body_ang_vel_w": np.asarray(motion["body_ang_vel_w"][steps], dtype=np.float64),
    }
    return out


def infer_body_indices_from_reference(ref_body_pos: np.ndarray, model_body_pos: np.ndarray) -> list[int]:
    # The motion bundle stores 40 bodies.  Match each of the 14 MuJoCo named
    # bodies to the nearest reference body at the initial standing frame.  This
    # avoids hard-coding a body order that may differ across converted assets.
    used: set[int] = set()
    indices: list[int] = []
    for target in model_body_pos:
        distances = np.linalg.norm(ref_body_pos - target[None, :], axis=1)
        order = np.argsort(distances)
        chosen = int(next(idx for idx in order if int(idx) not in used))
        indices.append(chosen)
        used.add(chosen)
    return indices


def build_obs(
    *,
    model,
    data,
    ids: dict[str, int],
    ref: dict[str, np.ndarray],
    frame_idx: int,
    default_joint_pos: np.ndarray,
    last_action: np.ndarray,
    ref_body_indices: list[int],
    world_to_init_yaw: float,
    world_to_init_translation: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    anchor_id = ids[ANCHOR_BODY_NAME]
    robot_anchor_pos = data.xpos[anchor_id].copy()
    robot_anchor_quat = normalize_quat_wxyz(data.xquat[anchor_id].copy())

    ref_body_pos = ref["body_pos_w"][frame_idx, ref_body_indices].copy()
    ref_body_quat = ref["body_quat_w"][frame_idx, ref_body_indices].copy()
    ref_joint_pos = ref["joint_pos"][frame_idx].copy()
    ref_joint_vel = ref["joint_vel"][frame_idx].copy()
    motion_anchor_index = BODY_NAMES.index(ANCHOR_BODY_NAME)

    yaw_quat = np.array([math.cos(world_to_init_yaw / 2.0), 0.0, 0.0, math.sin(world_to_init_yaw / 2.0)])
    ref_anchor_pos = world_to_init_translation + quat_apply(yaw_quat, ref_body_pos[motion_anchor_index])
    ref_anchor_quat = quat_mul(yaw_quat, normalize_quat_wxyz(ref_body_quat[motion_anchor_index]))

    command = np.concatenate([ref_joint_pos, ref_joint_vel])
    motion_anchor_pos_b = local_pos(robot_anchor_pos, robot_anchor_quat, ref_anchor_pos)
    motion_anchor_ori_b = rot6(local_quat(robot_anchor_quat, ref_anchor_quat))

    root_quat = normalize_quat_wxyz(data.qpos[3:7].copy())
    base_lin_vel_b = quat_apply(quat_conj(root_quat), data.qvel[0:3].copy())
    base_ang_vel_b = quat_apply(quat_conj(root_quat), data.qvel[3:6].copy())
    joint_pos_rel = data.qpos[7 : 7 + 29].copy() - default_joint_pos
    joint_vel_rel = data.qvel[6 : 6 + 29].copy()
    obs = np.concatenate(
        [
            command,
            motion_anchor_pos_b,
            motion_anchor_ori_b,
            base_lin_vel_b,
            base_ang_vel_b,
            joint_pos_rel,
            joint_vel_rel,
            last_action,
        ]
    ).astype(np.float32)
    if obs.shape != (160,):
        raise RuntimeError(f"Expected 160-D obs, got {obs.shape}")
    debug = {
        "motion_anchor_pos_b_norm": float(np.linalg.norm(motion_anchor_pos_b)),
        "base_lin_vel_b_norm": float(np.linalg.norm(base_lin_vel_b)),
        "base_ang_vel_b_norm": float(np.linalg.norm(base_ang_vel_b)),
        "joint_pos_rel_abs_mean": float(np.mean(np.abs(joint_pos_rel))),
        "joint_vel_rel_abs_mean": float(np.mean(np.abs(joint_vel_rel))),
    }
    return obs, debug


def choose_segment() -> dict[str, Any]:
    qg.patch_artifact_bindings()
    if not hasattr(qg.base, "_bm_stage1_original_find_continuous_segments"):
        qg.base._bm_stage1_original_find_continuous_segments = qg.base.find_continuous_segments
    if not hasattr(qg.base, "_bm_stage1_original_load_segment"):
        qg.base._bm_stage1_original_load_segment = qg.base.load_segment
    segments = qg.filtered_continuous_segments()
    segment = dict(segments[0])
    segment["end_exclusive"] = int(segment["start"] + min(int(segment["length"]), qg.TARGET_FRAMES))
    segment["length"] = int(segment["end_exclusive"] - segment["start"])
    return segment


def load_segment_motion_steps(segment: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    data = qg.quality_load_segment(segment, max_frames=int(segment["length"]))
    return np.asarray(data["motion_time_steps"], dtype=np.int64), data["continuity"]


def compare_with_open_loop(native_metrics: dict[str, Any]) -> dict[str, Any]:
    open_loop_summary_path = OUT_ROOT / "teacher_policy_action_control/teacher_policy_action_control_summary.json"
    open_loop = json.loads(open_loop_summary_path.read_text(encoding="utf-8")) if open_loop_summary_path.is_file() else {}
    open_metrics = open_loop.get("metrics", {})
    comparison = {
        "status": "ok_native_adapter_comparison",
        "timestamp_utc": utc_now(),
        "experiment_type": "stage1_multisource_quality_gated_native_adapter_comparison",
        "claim_level": "Compares approximate native MuJoCo obs->PPO action probe against old open-loop stored-action replay; not paper-level evidence.",
        "native_probe_summary": str(OUT_SUMMARY),
        "open_loop_teacher_summary": str(open_loop_summary_path),
        "native_metrics": native_metrics,
        "open_loop_teacher_metrics": open_metrics,
        "improvement": {
            "root_height_min_delta_native_minus_open_loop": (
                native_metrics.get("root_height_min", 0.0) - open_metrics.get("root_height_min", 0.0)
                if open_metrics
                else None
            ),
            "fall_proxy_count_delta_native_minus_open_loop": (
                native_metrics.get("fall_proxy_count", 0) - open_metrics.get("fall_proxy_count", 0)
                if open_metrics
                else None
            ),
            "root_position_error_mean_delta_native_minus_open_loop": (
                native_metrics.get("root_position_error_mean_m", 0.0)
                - open_metrics.get("root_position_error_mean_m", 0.0)
                if open_metrics
                else None
            ),
        },
        "checks": {
            "native_summary_exists": OUT_SUMMARY.is_file(),
            "open_loop_summary_exists": open_loop_summary_path.is_file(),
            "does_not_claim_paper_level": True,
            "does_not_claim_real_robot": True,
        },
    }
    write_json(OUT_COMPARE, comparison)
    lines = [
        "# Native PPO MuJoCo Adapter Comparison",
        "",
        "## 结论",
        "",
        "该结果比较 approximate native MuJoCo obs->PPO->action adapter 与旧的 open-loop stored-action replay。它用于排查 adapter，不是 paper-level 结果。",
        "",
        "## Metrics",
        "",
        f"- Native root height min/max: `{native_metrics.get('root_height_min')}` / `{native_metrics.get('root_height_max')}`",
        f"- Native fall proxy count: `{native_metrics.get('fall_proxy_count')}`",
        f"- Native root position error mean/max: `{native_metrics.get('root_position_error_mean_m')}` / `{native_metrics.get('root_position_error_max_m')}`",
        f"- Open-loop root height min/max: `{open_metrics.get('root_height_min')}` / `{open_metrics.get('root_height_max')}`",
        f"- Open-loop fall proxy count: `{open_metrics.get('fall_proxy_count')}`",
        f"- Open-loop root position error mean/max: `{open_metrics.get('root_position_error_mean_m')}` / `{open_metrics.get('root_position_error_max_m')}`",
        "",
        "## Claim Boundary",
        "",
        "这是本地 MuJoCo adapter probe，不是官方 BeyondMimic IsaacLab rollout，不是真实机器人结果。",
        "",
    ]
    OUT_COMPARE_MD.write_text("\n".join(lines), encoding="utf-8")
    return comparison


def main() -> None:
    import mujoco

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["BM_MUJOCO_ROOT_ASSIST"] = os.environ.get("BM_MUJOCO_ROOT_ASSIST", "1")
    os.environ["BM_STAGE1_QG_TARGET_FRAMES"] = os.environ.get("BM_STAGE1_QG_TARGET_FRAMES", "30")

    segment = choose_segment()
    time_steps, continuity = load_segment_motion_steps(segment)
    frames = int(len(time_steps))
    fps = int(os.environ.get("BM_STAGE1_NATIVE_ADAPTER_FPS", "30"))
    substeps = int(os.environ.get("BM_STAGE1_NATIVE_ADAPTER_SUBSTEPS", "4"))
    settle_steps = int(os.environ.get("BM_STAGE1_NATIVE_ADAPTER_SETTLE_STEPS", "20"))
    action_clip = float(os.environ.get("BM_STAGE1_NATIVE_ADAPTER_ACTION_CLIP", "3.0"))
    width = int(os.environ.get("BM_STAGE1_NATIVE_ADAPTER_WIDTH", "960"))
    height = int(os.environ.get("BM_STAGE1_NATIVE_ADAPTER_HEIGHT", "540"))

    policy = load_policy()
    action_rows = load_action_rows()
    model_path = Path(os.environ.get("BM_MUJOCO_G1_XML", str(DEFAULT_MODEL))).expanduser()
    patched_xml = model_path.parent / "g1_mocap_29dof_quality_gated_native_ppo_probe.xml"
    patch_joints_and_actuators(model_path, patched_xml, action_rows)
    model = mujoco.MjModel.from_xml_path(str(patched_xml))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    if actuator_joint_order(model) != [row["joint_name"] for row in action_rows]:
        raise RuntimeError("MuJoCo actuator order does not match action rows")
    ids = body_id_map(model)
    pelvis_body = ids["pelvis"]

    action_scale = np.asarray([float(row["action_scale"]) for row in action_rows], dtype=np.float64)
    default_joint_pos, default_source, default_note = parse_default_joint_position(action_rows)
    ref = ref_arrays(time_steps)

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = ref["body_pos_w"][0, 0, :].copy()
    data.qpos[0:2] = 0.0
    data.qpos[3:7] = normalize_quat_wxyz(ref["body_quat_w"][0, 0, :])
    data.qpos[7 : 7 + 29] = ref["joint_pos"][0]
    data.ctrl[:] = np.clip(ref["joint_pos"][0], model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    mujoco.mj_forward(model, data)
    model_body_initial = np.stack([data.xpos[ids[name]].copy() for name in BODY_NAMES], axis=0)
    ref_body_indices = infer_body_indices_from_reference(ref["body_pos_w"][0], model_body_initial)
    anchor_idx = BODY_NAMES.index(ANCHOR_BODY_NAME)
    ref_anchor_initial = ref["body_pos_w"][0, ref_body_indices[anchor_idx]]
    robot_anchor_initial = data.xpos[ids[ANCHOR_BODY_NAME]].copy()
    robot_anchor_yaw = quat_to_roll_pitch_yaw(data.xquat[ids[ANCHOR_BODY_NAME]].copy())[2]
    ref_anchor_yaw = quat_to_roll_pitch_yaw(ref["body_quat_w"][0, ref_body_indices[anchor_idx]])[2]
    world_to_init_yaw = robot_anchor_yaw - ref_anchor_yaw
    yaw_quat0 = np.array([math.cos(world_to_init_yaw / 2.0), 0.0, 0.0, math.sin(world_to_init_yaw / 2.0)])
    world_to_init_translation = robot_anchor_initial - quat_apply(yaw_quat0, ref_anchor_initial)

    for _ in range(settle_steps):
        data.xfrc_applied[:] = 0.0
        apply_root_assist(model, data, pelvis_body, data.xpos[pelvis_body].copy(), data.xquat[pelvis_body].copy())
        mujoco.mj_step(model, data)

    last_action = np.zeros(29, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    strips: list[np.ndarray] = []
    strip_indices = {0, frames // 2, frames - 1}
    with imageio.get_writer(OUT_MP4, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame_idx in range(frames):
            mujoco.mj_forward(model, data)
            obs, obs_debug = build_obs(
                model=model,
                data=data,
                ids=ids,
                ref=ref,
                frame_idx=frame_idx,
                default_joint_pos=default_joint_pos,
                last_action=last_action,
                ref_body_indices=ref_body_indices,
                world_to_init_yaw=world_to_init_yaw,
                world_to_init_translation=world_to_init_translation,
            )
            action = policy.act(obs)
            target, target_meta = action_to_joint_targets(
                action[None, :],
                default_joint_pos,
                action_scale,
                np.asarray(model.actuator_ctrlrange, dtype=np.float64),
                frames=1,
                clip_actions=action_clip,
            )
            target = target[0]
            data.ctrl[:] = target
            ref_root_pos = ref["body_pos_w"][frame_idx, 0, :].copy()
            ref_root_pos[0:2] = data.xpos[pelvis_body, 0:2]
            ref_root_quat = normalize_quat_wxyz(ref["body_quat_w"][frame_idx, 0, :])
            for _ in range(substeps):
                data.xfrc_applied[:] = 0.0
                apply_root_assist(model, data, pelvis_body, ref_root_pos, ref_root_quat)
                mujoco.mj_step(model, data)
            frame = render_frame(model, data, renderer, camera=PD_CAMERA)
            if frame_idx == 0:
                imageio.imwrite(OUT_KEYFRAME, frame)
            if frame_idx in strip_indices:
                strips.append(frame)
            writer.append_data(frame)
            q = data.qpos[7 : 7 + 29].copy()
            qd = data.qvel[6 : 6 + 29].copy()
            ref_joint = ref["joint_pos"][frame_idx]
            roll, pitch, yaw = quat_to_roll_pitch_yaw(data.qpos[3:7])
            row = {
                "frame": frame_idx,
                "motion_time_step": int(time_steps[frame_idx]),
                "sim_time_s": float(data.time),
                "root_x": float(data.qpos[0]),
                "root_y": float(data.qpos[1]),
                "root_z": float(data.qpos[2]),
                "root_roll": roll,
                "root_pitch": pitch,
                "root_yaw": yaw,
                "action_abs_mean": float(np.mean(np.abs(action))),
                "action_abs_max": float(np.max(np.abs(action))),
                "target_abs_mean": float(np.mean(np.abs(target))),
                "joint_error_to_policy_target_abs_mean": float(np.mean(np.abs(q - target))),
                "joint_error_to_reference_abs_mean": float(np.mean(np.abs(q - ref_joint))),
                "joint_velocity_abs_mean": float(np.mean(np.abs(qd))),
                "root_position_error_m": float(np.linalg.norm(data.xpos[pelvis_body] - ref_root_pos)),
                "root_target_z": float(ref["body_pos_w"][frame_idx, 0, 2]),
                "contact_count": int(data.ncon),
                "fall_proxy": bool(data.qpos[2] < 0.45 or abs(roll) > 1.2 or abs(pitch) > 1.2),
                **obs_debug,
            }
            rows.append(row)
            last_action = action

    renderer.close()
    make_keyframe_strip(strips, OUT_KEYFRAMES)
    with OUT_METRICS.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "fall_proxy_count": int(sum(1 for row in rows if row["fall_proxy"])),
        "root_height_min": float(min(row["root_z"] for row in rows)),
        "root_height_max": float(max(row["root_z"] for row in rows)),
        "root_position_error_mean_m": float(np.mean([row["root_position_error_m"] for row in rows])),
        "root_position_error_max_m": float(np.max([row["root_position_error_m"] for row in rows])),
        "joint_error_to_reference_abs_mean": float(np.mean([row["joint_error_to_reference_abs_mean"] for row in rows])),
        "joint_error_to_policy_target_abs_mean": float(
            np.mean([row["joint_error_to_policy_target_abs_mean"] for row in rows])
        ),
        "action_abs_mean": float(np.mean([row["action_abs_mean"] for row in rows])),
        "action_abs_max": float(np.max([row["action_abs_max"] for row in rows])),
        "contact_count_mean": float(np.mean([row["contact_count"] for row in rows])),
    }
    summary = {
        "status": "ok_native_ppo_obs_adapter_probe",
        "timestamp_utc": utc_now(),
        "experiment_type": "stage1_multisource_quality_gated_native_ppo_mujoco_probe",
        "claim_level": "Approximate local MuJoCo obs->PPO->action adapter probe; not official IsaacLab rollout, not paper-level result.",
        "selected_segment": {
            "source_motion": segment.get("source_motion", {}),
            "rank": int(segment["rank"]),
            "env_index": int(segment["env_index"]),
            "start": int(segment["start"]),
            "end_exclusive": int(segment["end_exclusive"]),
            "frames": frames,
            "motion_time_step_start": int(time_steps[0]),
            "motion_time_step_end": int(time_steps[-1]),
        },
        "checkpoint": {
            "path": str(policy.checkpoint),
            "sha256": sha256(policy.checkpoint),
            "actor_architecture": "160-512-256-128-29 ELU MLP",
            "uses_obs_normalizer": True,
        },
        "motion_bundle": {"path": str(MOTION_BUNDLE_NPZ), "sha256": sha256(MOTION_BUNDLE_NPZ)},
        "simulation": {
            "uses_mj_step": True,
            "writes_qpos_each_frame": False,
            "uses_root_assist_controller": os.environ.get("BM_MUJOCO_ROOT_ASSIST", "1") == "1",
            "substeps_per_frame": substeps,
            "settle_steps": settle_steps,
            "patched_xml": str(patched_xml),
            "actuator_count": int(model.nu),
        },
        "observation_adapter": {
            "obs_dim": 160,
            "layout": [
                "command: reference joint_pos + joint_vel (58)",
                "motion_anchor_pos_b (3)",
                "motion_anchor_ori_b rot6 (6)",
                "base_lin_vel_b (3)",
                "base_ang_vel_b (3)",
                "joint_pos_rel (29)",
                "joint_vel_rel (29)",
                "last_action (29)",
            ],
            "approximation_notes": [
                "Reference command is read from local motion bundle, not from an official deployment ONNX time_step output.",
                "Reference body indices are inferred by nearest initial body positions between MuJoCo model and 40-body motion bundle.",
                "Motion frame yaw alignment is approximated for this short probe.",
                "This does not reconstruct IsaacLab contact sensors, terrain, randomization, or exact reset manager.",
            ],
        },
        "default_joint_pos_source": default_source,
        "default_joint_pos_note": default_note,
        "ref_body_indices": {name: int(idx) for name, idx in zip(BODY_NAMES, ref_body_indices)},
        "continuity": continuity,
        "outputs": {
            "mp4": str(OUT_MP4),
            "keyframe_png": str(OUT_KEYFRAME),
            "keyframes_png": str(OUT_KEYFRAMES),
            "metrics_csv": str(OUT_METRICS),
            "summary_json": str(OUT_SUMMARY),
        },
        "metrics": metrics,
        "checks": {
            "mp4_exists": OUT_MP4.is_file() and OUT_MP4.stat().st_size > 0,
            "metrics_csv_exists": OUT_METRICS.is_file() and OUT_METRICS.stat().st_size > 0,
            "uses_mj_step": True,
            "does_not_write_qpos_each_frame": True,
            "uses_actor_checkpoint": policy.checkpoint.is_file(),
            "uses_obs_normalizer": True,
            "obs_dim_160": True,
            "action_dim_29": True,
            "does_not_claim_paper_level": True,
            "does_not_claim_real_robot": True,
        },
        "limitations": [
            "The local Stage-1 teacher is still weak in IsaacLab evaluation, so good long-horizon motion is not expected yet.",
            "This probe verifies adapter direction; it is not a validated replacement for official IsaacLab or motion_tracking_controller deployment.",
            "Root assist remains enabled to keep the short diagnostic centered and comparable to earlier MuJoCo videos.",
        ],
    }
    write_json(OUT_SUMMARY, summary)
    comparison = compare_with_open_loop(metrics)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "mp4": str(OUT_MP4),
                "summary": str(OUT_SUMMARY),
                "comparison": str(OUT_COMPARE),
                "fall_proxy_count": metrics["fall_proxy_count"],
                "root_height_min": metrics["root_height_min"],
                "root_position_error_mean_m": metrics["root_position_error_mean_m"],
                "open_loop_delta_root_height_min": comparison["improvement"][
                    "root_height_min_delta_native_minus_open_loop"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
