#!/usr/bin/env python3
"""Render LAFAN1 paper-contract Stage-1/Level-C artifacts as MuJoCo control videos.

This script consumes the newly trained paper-contract teacher sweep, teacher
rollout dataset, local VAE checkpoint, and offline guided state-latent samples.
It then drives the MuJoCo G1 with 29 position actuators and ``mj_step`` to
produce report-ready MP4s.

Claim boundary: these videos are local MuJoCo action-to-PD control
visualizations from the current weak paper-contract chain.  They are not
native IsaacLab rendered MP4s, not official BeyondMimic checkpoints, not
unassisted paper-level closed-loop results, and not real-robot results.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from torch import nn

ROOT = Path(os.environ.get("BM_ROOT", "/mnt/infini-data/test/BeyondMimic")).resolve()
MUJOCO_SCRIPT_DIR = ROOT / "mujoco_mp4/scripts"
if str(MUJOCO_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(MUJOCO_SCRIPT_DIR))

from mujoco_common import render_frame, sha256, traceback_payload, utc_now, write_json
from mujoco_pd_control_video import (
    ACTION_SCALE_AUDIT,
    PD_CAMERA,
    actuator_joint_order,
    add_fixed_camera,
    apply_root_assist,
    load_action_rows,
    normalize_quat_wxyz,
    patch_joints_and_actuators,
    quat_error_rotvec,
    quat_to_roll_pitch_yaw,
)
from mujoco_trace_mesh_video import DEFAULT_MODEL, resample_array


OUT_ROOT = ROOT / "res/visualization/lafan1_paper_contract_videos"
TEACHER_ROLLOUT_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_paper_contract_best_teacher_rollout_dataset/"
    "tracking_g1_official_importer_export_paper_contract_best_teacher_rollout_dataset.json"
)
BEST_TEACHER_SWEEP_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_paper_contract_ppo_checkpoint_sweep/"
    "tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_sweep.json"
)
VAE_CKPT = (
    ROOT
    / "res/runs/level_c_official_importer_export_paper_contract_teacher_rollout_vae_training/"
    "resource_adjusted_teacher_rollout_vae_20260623_062423_seed20260805/"
    "resource_adjusted_teacher_rollout_action_vae.pt"
)
GUIDANCE_SAMPLES = (
    ROOT
    / "res/runs/level_c_official_importer_export_paper_contract_state_latent_guidance_eval/"
    "resource_adjusted_state_latent_guidance_20260623_062841_seed20260808/"
    "resource_adjusted_state_latent_guidance_samples.npz"
)
MOTION_BUNDLE = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/"
    "official_csv_loop_full_public_motion_bundle_fk_repaired_robot_order.npz"
)
DEFAULT_CONTINUOUS_REFERENCE_MOTION = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/"
    "motions/walk1_subject1/motion.npz"
)
MOTION_BUNDLE_AUDIT = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/"
    "tracking_g1_official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz.json"
)
CONTROLLER_YAML = ROOT / "download/official/motion_tracking_controller/config/g1/controllers.yaml"
REFERENCE_POSE_CAMERA = "bm_reference_pose_replay"


class ConditionalActionVAE(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(obs_dim + latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def posterior_mean(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        stats = self.encoder(torch.cat([obs, action], dim=-1))
        mu, _ = stats.chunk(2, dim=-1)
        return mu

    def decode(self, obs: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([obs, latent], dim=-1))


def load_vae() -> tuple[ConditionalActionVAE, dict[str, Any]]:
    payload = torch.load(VAE_CKPT, map_location="cpu")
    cfg = dict(payload["config"])
    model = ConditionalActionVAE(
        int(cfg["obs_dim"]),
        int(cfg["action_dim"]),
        int(cfg["latent_dim"]),
        int(cfg["hidden_dim"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, cfg


def parse_default_joint_position(action_rows: list[dict[str, Any]]) -> tuple[np.ndarray, str, str]:
    if CONTROLLER_YAML.is_file():
        text = CONTROLLER_YAML.read_text(encoding="utf-8")
        match = re.search(r"default_position:\s*\[(.*?)\]", text, flags=re.S)
        if match:
            values = [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", match.group(1))]
            if len(values) == 29:
                return (
                    np.asarray(values, dtype=np.float64),
                    str(CONTROLLER_YAML),
                    "official motion_tracking_controller standby_controller.default_position",
                )
    return np.zeros(len(action_rows), dtype=np.float64), "mujoco_qpos0_zero_joint_fallback", "fallback"


def select_teacher_sequence() -> dict[str, Any]:
    summary = json.loads(TEACHER_ROLLOUT_JSON.read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = []
    for shard_path_str in summary["run"]["shard_npz_paths"]:
        shard_path = Path(shard_path_str)
        data = np.load(shard_path)
        rewards = np.asarray(data["rewards"], dtype=np.float64)
        dones = np.asarray(data["dones"], dtype=np.bool_)
        time_steps = np.asarray(data["motion_time_steps"], dtype=np.int64)
        first_done = np.where(dones.any(axis=0), np.argmax(dones, axis=0), dones.shape[0])
        reward_sum = rewards.sum(axis=0)
        score = first_done.astype(np.float64) * 1000.0 + reward_sum
        env_idx = int(np.argmax(score))
        candidates.append(
            {
                "path": shard_path,
                "env_index": env_idx,
                "first_done": int(first_done[env_idx]),
                "reward_sum": float(reward_sum[env_idx]),
                "mean_reward": float(rewards[:, env_idx].mean()),
                "time_step_first": int(time_steps[0, env_idx]),
                "time_step_last": int(time_steps[-1, env_idx]),
                "rank": int(np.asarray(data["rank"])[0]),
                "world_size": int(np.asarray(data["world_size"])[0]),
            }
        )
        data.close()
    selected = max(candidates, key=lambda row: (row["first_done"], row["reward_sum"]))
    data = np.load(selected["path"])
    env = int(selected["env_index"])
    selected["policy_obs"] = np.asarray(data["policy_obs"][:, env, :], dtype=np.float32)
    selected["actions"] = np.asarray(data["actions"][:, env, :], dtype=np.float32)
    selected["rewards"] = np.asarray(data["rewards"][:, env], dtype=np.float32)
    selected["dones"] = np.asarray(data["dones"][:, env], dtype=np.bool_)
    selected["timeouts"] = np.asarray(data["timeouts"][:, env], dtype=np.bool_)
    selected["motion_time_steps"] = np.asarray(data["motion_time_steps"][:, env], dtype=np.int64)
    deltas = np.diff(selected["motion_time_steps"].astype(np.int64))
    selected["motion_time_step_discontinuity"] = {
        "non_plus_one_count": int(np.sum(deltas != 1)),
        "negative_jump_count": int(np.sum(deltas < 0)),
        "large_abs_jump_gt_10_count": int(np.sum(np.abs(deltas) > 10)),
        "min_delta": int(deltas.min()) if deltas.size else 0,
        "max_delta": int(deltas.max()) if deltas.size else 0,
        "interpretation": "Teacher rollout command sampling/resets make this time-step sequence discontinuous; it is not a clean single-motion reference replay.",
    }
    selected["summary_json"] = str(TEACHER_ROLLOUT_JSON)
    data.close()
    return selected


def resample_time_steps(time_steps: np.ndarray, frames: int, total_frames: int) -> np.ndarray:
    idx = np.linspace(0, len(time_steps) - 1, frames)
    out = np.rint(np.interp(idx, np.arange(len(time_steps)), time_steps.astype(np.float64))).astype(np.int64)
    return np.clip(out, 0, total_frames - 1)


def load_reference_for_time_steps(time_steps: np.ndarray, frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    motion = np.load(MOTION_BUNDLE, allow_pickle=True)
    total = int(motion["joint_pos"].shape[0])
    frame_steps = resample_time_steps(time_steps, frames, total)
    joint_targets = np.asarray(motion["joint_pos"][frame_steps], dtype=np.float64)
    root_pos = np.asarray(motion["body_pos_w"][frame_steps, 0, :], dtype=np.float64)
    root_pos[:, 0:2] = 0.0
    root_quat = np.asarray(motion["body_quat_w"][frame_steps, 0, :], dtype=np.float64)
    root_quat = np.stack([normalize_quat_wxyz(q) for q in root_quat], axis=0)
    audit = json.loads(MOTION_BUNDLE_AUDIT.read_text(encoding="utf-8")) if MOTION_BUNDLE_AUDIT.is_file() else {}
    return joint_targets, root_pos, root_quat, {
        "motion_bundle": str(MOTION_BUNDLE),
        "motion_bundle_sha256": sha256(MOTION_BUNDLE),
        "motion_bundle_frames": total,
        "selected_time_step_first": int(frame_steps[0]),
        "selected_time_step_last": int(frame_steps[-1]),
        "selected_time_step_min": int(frame_steps.min()),
        "selected_time_step_max": int(frame_steps.max()),
        "motion_bundle_status": audit.get("status", ""),
        "root_xy_recentered_targets": True,
        "source_time_steps_from_teacher_rollout": True,
        "not_clean_continuous_reference_replay": True,
        "teacher_time_step_non_plus_one_count": int(np.sum(np.diff(time_steps.astype(np.int64)) != 1)),
        "teacher_time_step_negative_jump_count": int(np.sum(np.diff(time_steps.astype(np.int64)) < 0)),
        "teacher_time_step_large_abs_jump_gt_10_count": int(np.sum(np.abs(np.diff(time_steps.astype(np.int64))) > 10)),
    }


def add_reference_pose_camera(model_xml: Path, out_xml: Path, center_xy: np.ndarray) -> Path:
    text = model_xml.read_text(encoding="utf-8")
    cx, cy = float(center_xy[0]), float(center_xy[1])
    camera = (
        f'<camera name="{REFERENCE_POSE_CAMERA}" mode="fixed" '
        f'pos="{cx - 0.35:.4f} {cy - 4.80:.4f} 1.75" '
        'xyaxes="1 0 0 0 0.32 0.947" fovy="48"/>'
    )
    if f'name="{REFERENCE_POSE_CAMERA}"' in text:
        text = re.sub(rf'<camera name="{REFERENCE_POSE_CAMERA}"[^>]*/>', camera, text)
    else:
        text = text.replace("</worldbody>", camera + "\n  </worldbody>", 1)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    out_xml.write_text(text, encoding="utf-8")
    return out_xml


def load_continuous_reference_motion(frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    motion_path = Path(os.environ.get("BM_LAFAN1_REFERENCE_MOTION_NPZ", str(DEFAULT_CONTINUOUS_REFERENCE_MOTION)))
    motion = np.load(motion_path, allow_pickle=True)
    source_frames = int(motion["joint_pos"].shape[0])
    joint_pos = resample_array(np.asarray(motion["joint_pos"], dtype=np.float64), frames)
    joint_vel = resample_array(np.asarray(motion["joint_vel"], dtype=np.float64), frames)
    root_pos = resample_array(np.asarray(motion["body_pos_w"][:, 0, :], dtype=np.float64), frames)
    root_quat = resample_array(np.asarray(motion["body_quat_w"][:, 0, :], dtype=np.float64), frames)
    root_quat = np.stack([normalize_quat_wxyz(q) for q in root_quat], axis=0)
    root_xy_offset = root_pos[0, :2].copy()
    root_pos[:, :2] -= root_xy_offset[None, :]
    fps = int(np.asarray(motion["fps"]).reshape(-1)[0])
    return joint_pos, joint_vel, root_pos, root_quat, {
        "motion_npz": str(motion_path),
        "motion_sha256": sha256(motion_path),
        "motion_name": motion_path.parent.name,
        "source_frames": source_frames,
        "source_fps": fps,
        "frames_rendered": frames,
        "root_xy_offset_subtracted": [float(root_xy_offset[0]), float(root_xy_offset[1])],
        "root_xy_displacement_m": float(np.linalg.norm(root_pos[-1, :2] - root_pos[0, :2])),
        "target_source": "continuous_single_motion_root_pose_and_joint_qpos",
        "claim_level": "MuJoCo kinematic pose replay of one continuous FK-repaired LAFAN1 motion; not policy control and not PD tracking",
    }


def render_reference_pose_replay_video(frames: int) -> dict[str, Any]:
    import mujoco

    backend = os.environ.get("MUJOCO_GL", "egl")
    fps = int(os.environ.get("BM_LAFAN1_VIDEO_FPS", "30"))
    width = int(os.environ.get("BM_LAFAN1_VIDEO_WIDTH", "960"))
    height = int(os.environ.get("BM_LAFAN1_VIDEO_HEIGHT", "540"))
    model_path = Path(os.environ.get("BM_MUJOCO_G1_XML", str(DEFAULT_MODEL))).expanduser()
    joint_pos, joint_vel, root_pos, root_quat, source_meta = load_continuous_reference_motion(frames)
    patched_xml = model_path.parent / f"{model_path.stem}_lafan1_reference_pose_replay_camera.xml"
    add_reference_pose_camera(model_path, patched_xml, np.mean(root_pos[:, :2], axis=0))

    model = mujoco.MjModel.from_xml_path(str(patched_xml))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    out_dir = OUT_ROOT / "reference_pose_replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / "reference_pose_replay.mp4"
    keyframe_path = out_dir / "reference_pose_replay_keyframe.png"
    strip_path = out_dir / "reference_pose_replay_keyframes.png"
    metrics_path = out_dir / "reference_pose_replay_metrics.csv"
    summary_path = out_dir / "reference_pose_replay_summary.json"

    rows_out: list[dict[str, Any]] = []
    strip_frames: list[np.ndarray] = []
    strip_indices = {0, frames // 2, frames - 1}
    with imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame_idx in range(frames):
            data.qpos[:] = 0.0
            data.qvel[:] = 0.0
            data.qpos[0:3] = root_pos[frame_idx]
            data.qpos[3:7] = root_quat[frame_idx]
            data.qpos[7 : 7 + 29] = joint_pos[frame_idx]
            data.qvel[6 : 6 + 29] = joint_vel[frame_idx]
            mujoco.mj_forward(model, data)
            frame = render_frame(model, data, renderer, camera=REFERENCE_POSE_CAMERA)
            if frame_idx == 0:
                imageio.imwrite(keyframe_path, frame)
            if frame_idx in strip_indices:
                strip_frames.append(frame)
            writer.append_data(frame)
            rows_out.append(
                {
                    "frame": frame_idx,
                    "video_time_s": frame_idx / fps,
                    "root_x": float(root_pos[frame_idx, 0]),
                    "root_y": float(root_pos[frame_idx, 1]),
                    "root_z": float(root_pos[frame_idx, 2]),
                    "joint_abs_mean": float(np.mean(np.abs(joint_pos[frame_idx]))),
                    "joint_vel_abs_mean": float(np.mean(np.abs(joint_vel[frame_idx]))),
                    "contact_count_after_forward": int(data.ncon),
                }
            )
    renderer.close()
    make_keyframe_strip(strip_frames, strip_path)

    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        fields = list(rows_out[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)

    payload = {
        "status": "ok",
        "timestamp_utc": utc_now(),
        "experiment_type": "lafan1_continuous_reference_pose_replay",
        "spec_name": "reference_pose_replay",
        "backend": backend,
        "claim_level": source_meta["claim_level"],
        "source_metadata": source_meta,
        "source_model_xml": str(model_path),
        "patched_camera_model_xml": str(patched_xml),
        "frames_rendered": frames,
        "video_fps": fps,
        "duration_seconds": frames / fps,
        "simulation": {
            "uses_mj_forward": True,
            "uses_mj_step": False,
            "writes_qpos_each_frame": True,
            "actuator_control_used": False,
        },
        "outputs": {
            "mp4": str(mp4_path),
            "keyframe_png": str(keyframe_path),
            "keyframes_png": str(strip_path),
            "metrics_csv": str(metrics_path),
            "summary_json": str(summary_path),
        },
        "file_sizes": {
            "mp4": mp4_path.stat().st_size if mp4_path.exists() else 0,
            "keyframe_png": keyframe_path.stat().st_size if keyframe_path.exists() else 0,
            "keyframes_png": strip_path.stat().st_size if strip_path.exists() else 0,
            "metrics_csv": metrics_path.stat().st_size if metrics_path.exists() else 0,
        },
        "metrics": {
            "root_xy_displacement_m": source_meta["root_xy_displacement_m"],
            "root_height_min": float(np.min(root_pos[:, 2])),
            "root_height_max": float(np.max(root_pos[:, 2])),
            "joint_abs_mean": float(np.mean(np.abs(joint_pos))),
            "joint_vel_abs_mean": float(np.mean(np.abs(joint_vel))),
        },
        "checks": {
            "mp4_exists": mp4_path.is_file() and mp4_path.stat().st_size > 0,
            "keyframe_exists": keyframe_path.is_file() and keyframe_path.stat().st_size > 0,
            "metrics_csv_exists": metrics_path.is_file() and metrics_path.stat().st_size > 0,
            "uses_mujoco_g1_mesh": True,
            "uses_mj_forward": True,
            "does_not_use_mj_step": True,
            "writes_qpos_each_frame": True,
            "does_not_claim_policy_rollout": True,
            "does_not_claim_pd_control": True,
            "does_not_claim_real_robot": True,
        },
        "limitations": [
            "This is the correct visualization for the continuous reference motion itself.",
            "It intentionally imposes qpos frame-by-frame, so it does not evaluate control stability, contact realism, or policy quality.",
        ],
    }
    write_json(summary_path, payload)
    print(json.dumps({"status": "ok", "spec": "reference_pose_replay", "mp4": str(mp4_path)}))
    return payload


def action_to_joint_targets(
    actions: np.ndarray,
    default_joint_pos: np.ndarray,
    action_scale: np.ndarray,
    ctrlrange: np.ndarray,
    frames: int,
    clip_actions: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    seq = resample_array(np.asarray(actions, dtype=np.float64), frames)
    clipped = np.clip(seq, -clip_actions, clip_actions)
    targets = default_joint_pos[None, :] + clipped * action_scale[None, :]
    targets = np.clip(targets, ctrlrange[:, 0], ctrlrange[:, 1])
    return targets, {
        "source_action_shape": list(actions.shape),
        "resampled_action_shape": list(seq.shape),
        "raw_action_abs_mean": float(np.mean(np.abs(seq))),
        "raw_action_abs_max": float(np.max(np.abs(seq))),
        "clip_actions_abs": float(clip_actions),
        "clipped_fraction": float(np.mean(np.abs(seq) > clip_actions)),
        "target_min": float(np.min(targets)),
        "target_max": float(np.max(targets)),
    }


def vae_reconstruct_actions(model: ConditionalActionVAE, obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        obs_t = torch.from_numpy(obs.astype(np.float32))
        act_t = torch.from_numpy(actions.astype(np.float32))
        z = model.posterior_mean(obs_t, act_t)
        pred = model.decode(obs_t, z)
    return pred.cpu().numpy().astype(np.float32)


def decode_guidance_actions(model: ConditionalActionVAE, key: str, sample_index: int = 0) -> np.ndarray:
    samples = np.load(GUIDANCE_SAMPLES)
    tokens = np.asarray(samples[key][sample_index], dtype=np.float32)
    obs = tokens[:, :160]
    latent = tokens[:, 160:]
    with torch.inference_mode():
        pred = model.decode(torch.from_numpy(obs), torch.from_numpy(latent))
    return pred.cpu().numpy().astype(np.float32)


def make_keyframe_strip(frames: list[np.ndarray], path: Path) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.concatenate(frames, axis=1))


def render_action_control_video(
    spec_name: str,
    joint_targets: np.ndarray,
    root_pos_targets: np.ndarray,
    root_quat_targets: np.ndarray,
    source_meta: dict[str, Any],
    common_meta: dict[str, Any],
) -> dict[str, Any]:
    import mujoco

    backend = os.environ.get("MUJOCO_GL", "egl")
    frames = int(joint_targets.shape[0])
    fps = int(os.environ.get("BM_LAFAN1_VIDEO_FPS", "30"))
    width = int(os.environ.get("BM_LAFAN1_VIDEO_WIDTH", "960"))
    height = int(os.environ.get("BM_LAFAN1_VIDEO_HEIGHT", "540"))
    substeps = int(os.environ.get("BM_LAFAN1_CONTROL_SUBSTEPS", "4"))
    settle_steps = int(os.environ.get("BM_LAFAN1_SETTLE_STEPS", "40"))
    model_path = Path(os.environ.get("BM_MUJOCO_G1_XML", str(DEFAULT_MODEL))).expanduser()
    action_rows = load_action_rows()
    patched_xml = model_path.parent / "g1_mocap_29dof_lafan1_paper_contract_pd.xml"
    patch_joints_and_actuators(model_path, patched_xml, action_rows)

    model = mujoco.MjModel.from_xml_path(str(patched_xml))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    expected_order = [row["joint_name"] for row in action_rows]
    actual_order = actuator_joint_order(model)
    if actual_order != expected_order:
        raise RuntimeError(f"Actuator order mismatch: {actual_order} != {expected_order}")
    pelvis_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_body < 0:
        raise RuntimeError("MuJoCo body 'pelvis' not found")

    out_dir = OUT_ROOT / spec_name
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / f"{spec_name}.mp4"
    keyframe_path = out_dir / f"{spec_name}_keyframe.png"
    strip_path = out_dir / f"{spec_name}_keyframes.png"
    metrics_path = out_dir / f"{spec_name}_metrics.csv"
    summary_path = out_dir / f"{spec_name}_summary.json"

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = root_pos_targets[0]
    data.qpos[3:7] = root_quat_targets[0]
    data.qpos[7 : 7 + 29] = joint_targets[0]
    data.ctrl[:] = np.clip(joint_targets[0], model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    mujoco.mj_forward(model, data)
    for _ in range(settle_steps):
        data.xfrc_applied[:] = 0.0
        apply_root_assist(model, data, pelvis_body, root_pos_targets[0], root_quat_targets[0])
        mujoco.mj_step(model, data)

    rows_out: list[dict[str, Any]] = []
    strip_frames: list[np.ndarray] = []
    strip_indices = {0, frames // 2, frames - 1}
    with imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame_idx in range(frames):
            target = np.clip(joint_targets[frame_idx], model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            data.ctrl[:] = target
            for _ in range(substeps):
                data.xfrc_applied[:] = 0.0
                apply_root_assist(model, data, pelvis_body, root_pos_targets[frame_idx], root_quat_targets[frame_idx])
                mujoco.mj_step(model, data)
            frame = render_frame(model, data, renderer, camera=PD_CAMERA)
            if frame_idx == 0:
                imageio.imwrite(keyframe_path, frame)
            if frame_idx in strip_indices:
                strip_frames.append(frame)
            writer.append_data(frame)

            q = data.qpos[7 : 7 + 29].copy()
            qd = data.qvel[6 : 6 + 29].copy()
            err = q - target
            roll, pitch, yaw = quat_to_roll_pitch_yaw(data.qpos[3:7])
            rows_out.append(
                {
                    "frame": frame_idx,
                    "video_time_s": frame_idx / fps,
                    "sim_time_s": float(data.time),
                    "root_x": float(data.qpos[0]),
                    "root_y": float(data.qpos[1]),
                    "root_z": float(data.qpos[2]),
                    "root_roll": roll,
                    "root_pitch": pitch,
                    "root_yaw": yaw,
                    "joint_target_abs_mean": float(np.mean(np.abs(target))),
                    "joint_error_abs_mean": float(np.mean(np.abs(err))),
                    "joint_error_abs_max": float(np.max(np.abs(err))),
                    "joint_velocity_abs_mean": float(np.mean(np.abs(qd))),
                    "ctrl_abs_mean": float(np.mean(np.abs(data.ctrl))),
                    "root_target_z": float(root_pos_targets[frame_idx, 2]),
                    "root_position_error_m": float(np.linalg.norm(data.xpos[pelvis_body] - root_pos_targets[frame_idx])),
                    "root_orientation_error_rad": float(
                        np.linalg.norm(quat_error_rotvec(root_quat_targets[frame_idx], data.xquat[pelvis_body]))
                    ),
                    "contact_count": int(data.ncon),
                    "fall_proxy": bool(data.qpos[2] < 0.45 or abs(roll) > 1.2 or abs(pitch) > 1.2),
                }
            )
    renderer.close()
    make_keyframe_strip(strip_frames, strip_path)

    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        fields = list(rows_out[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)

    fall_count = sum(1 for row in rows_out if row["fall_proxy"])
    payload = {
        "status": "ok",
        "timestamp_utc": utc_now(),
        "experiment_type": source_meta.get(
            "experiment_type",
            "lafan1_paper_contract_mujoco_action_control_video",
        ),
        "spec_name": spec_name,
        "backend": backend,
        "claim_level": source_meta["claim_level"],
        "common_chain": common_meta,
        "source_metadata": source_meta,
        "continuity": source_meta.get("continuity", {}),
        "source_model_xml": str(model_path),
        "patched_pd_model_xml": str(patched_xml),
        "action_scale_audit": str(ACTION_SCALE_AUDIT),
        "frames_rendered": frames,
        "video_fps": fps,
        "duration_seconds": frames / fps,
        "simulation": {
            "uses_mj_step": True,
            "writes_qpos_each_frame": False,
            "actuator_type": "position",
            "actuator_count": int(model.nu),
            "control_substeps_per_frame": substeps,
            "settle_steps": settle_steps,
            "timestep": float(model.opt.timestep),
            "sim_time_s": float(data.time),
            "root_assist_enabled": os.environ.get("BM_MUJOCO_ROOT_ASSIST", "1") == "1",
            "root_assist_type": "external pelvis force/torque stabilizer applied before mj_step",
        },
        "camera": {
            "name": PD_CAMERA,
            "mode": "fixed",
            "position": os.environ.get("BM_MUJOCO_PD_CAMERA_POS", "-0.35 -4.80 1.75"),
            "fovy": os.environ.get("BM_MUJOCO_PD_CAMERA_FOVY", "48"),
            "robot_centering": "root XY targets are recentered to zero and a fixed camera is used",
        },
        "outputs": {
            "mp4": str(mp4_path),
            "keyframe_png": str(keyframe_path),
            "keyframes_png": str(strip_path),
            "metrics_csv": str(metrics_path),
            "summary_json": str(summary_path),
        },
        "file_sizes": {
            "mp4": mp4_path.stat().st_size if mp4_path.exists() else 0,
            "keyframe_png": keyframe_path.stat().st_size if keyframe_path.exists() else 0,
            "keyframes_png": strip_path.stat().st_size if strip_path.exists() else 0,
            "metrics_csv": metrics_path.stat().st_size if metrics_path.exists() else 0,
        },
        "metrics": {
            "joint_error_abs_mean": float(np.mean([row["joint_error_abs_mean"] for row in rows_out])),
            "joint_error_abs_max": float(np.max([row["joint_error_abs_max"] for row in rows_out])),
            "root_position_error_mean_m": float(np.mean([row["root_position_error_m"] for row in rows_out])),
            "root_position_error_max_m": float(np.max([row["root_position_error_m"] for row in rows_out])),
            "root_orientation_error_mean_rad": float(np.mean([row["root_orientation_error_rad"] for row in rows_out])),
            "root_height_min": float(np.min([row["root_z"] for row in rows_out])),
            "root_height_max": float(np.max([row["root_z"] for row in rows_out])),
            "root_xy_abs_max": float(np.max([max(abs(row["root_x"]), abs(row["root_y"])) for row in rows_out])),
            "contact_count_mean": float(np.mean([row["contact_count"] for row in rows_out])),
            "fall_proxy_count": int(fall_count),
        },
        "checks": {
            "mp4_exists": mp4_path.is_file() and mp4_path.stat().st_size > 0,
            "keyframe_exists": keyframe_path.is_file() and keyframe_path.stat().st_size > 0,
            "metrics_csv_exists": metrics_path.is_file() and metrics_path.stat().st_size > 0,
            "uses_mujoco_g1_mesh": True,
            "uses_mj_step": True,
            "does_not_write_qpos_each_frame": True,
            "uses_root_assist_controller": os.environ.get("BM_MUJOCO_ROOT_ASSIST", "1") == "1",
            "uses_29_position_actuators": int(model.nu) == 29,
            "does_not_claim_isaaclab_render": True,
            "does_not_claim_official_beyondmimic_checkpoint": True,
            "does_not_claim_real_robot": True,
        },
        "limitations": source_meta.get(
            "limitations",
            [
                "The selected Stage-1 teacher is weak; its rollout shard marks done at frame 0 for all sampled envs.",
                "This is a MuJoCo action-to-PD visualization using a root-assist stabilizer, not an unassisted paper-level humanoid controller.",
                "VAE/diffusion/guidance videos decode local surrogate latent/action artifacts; they are not official BeyondMimic checkpoints or closed-loop IsaacLab Fig.5/Fig.6 results.",
                "The MuJoCo observation/action adapter for native PPO deployment is not claimed here.",
            ],
        ),
    }
    write_json(summary_path, payload)
    print(json.dumps({"status": "ok", "spec": spec_name, "mp4": str(mp4_path), "fall_proxy_count": fall_count}))
    return payload


def render_side_by_side(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    out_dir = OUT_ROOT / "guided_vs_unguided_action_control"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / "guided_vs_unguided_action_control.mp4"
    keyframe = out_dir / "guided_vs_unguided_action_control_keyframe.png"
    summary = out_dir / "guided_vs_unguided_action_control_summary.json"
    fps = int(os.environ.get("BM_LAFAN1_VIDEO_FPS", "30"))
    reader_l = imageio.get_reader(left["outputs"]["mp4"])
    reader_r = imageio.get_reader(right["outputs"]["mp4"])
    n = min(int(left["frames_rendered"]), int(right["frames_rendered"]))
    with imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for i in range(n):
            frame = np.concatenate([reader_l.get_data(i), reader_r.get_data(i)], axis=1)
            if i == 0:
                imageio.imwrite(keyframe, frame)
            writer.append_data(frame)
    reader_l.close()
    reader_r.close()
    payload = {
        "status": "ok",
        "timestamp_utc": utc_now(),
        "experiment_type": "lafan1_paper_contract_guided_vs_unguided_mujoco_action_control_side_by_side",
        "claim_level": "Side-by-side local MuJoCo action-to-PD visualization: denoised latent action targets on the left, guided latent action targets on the right; not native paper-level guidance rollout",
        "left_label": "unguided / denoised latent",
        "right_label": "guided latent",
        "left": left["outputs"],
        "right": right["outputs"],
        "frames_rendered": n,
        "video_fps": fps,
        "duration_seconds": n / fps,
        "outputs": {"mp4": str(out_mp4), "keyframe_png": str(keyframe), "summary_json": str(summary)},
        "checks": {
            "mp4_exists": out_mp4.is_file() and out_mp4.stat().st_size > 0,
            "keyframe_exists": keyframe.is_file() and keyframe.stat().st_size > 0,
            "uses_mj_step_source_videos": left["checks"]["uses_mj_step"] and right["checks"]["uses_mj_step"],
            "does_not_claim_native_mujoco_guidance": True,
            "does_not_claim_real_robot": True,
        },
    }
    write_json(summary, payload)
    print(json.dumps({"status": "ok", "spec": "guided_vs_unguided_action_control", "mp4": str(out_mp4)}))
    return payload


def write_top_level_summary(rendered: dict[str, Any], selected: dict[str, Any]) -> None:
    sweep = json.loads(BEST_TEACHER_SWEEP_JSON.read_text(encoding="utf-8"))
    summary = {
        "status": "ok",
        "timestamp_utc": utc_now(),
        "experiment_type": "lafan1_paper_contract_mujoco_action_control_video_suite",
        "output_root": str(OUT_ROOT),
        "claim_level": "Local MuJoCo visualization suite from the current LAFAN1 paper-contract chain: reference_pose_replay is kinematic reference visualization; action_control videos are PD/controller diagnostics; not official paper-level closed-loop result",
        "selected_teacher_rollout": {
            "shard": str(selected["path"]),
            "rank": selected["rank"],
            "env_index": selected["env_index"],
            "first_done": selected["first_done"],
            "reward_sum": selected["reward_sum"],
            "mean_reward": selected["mean_reward"],
            "time_step_first": selected["time_step_first"],
            "time_step_last": selected["time_step_last"],
            "motion_time_step_discontinuity": selected["motion_time_step_discontinuity"],
        },
        "best_teacher_sweep_metrics": sweep.get("metrics", {}),
        "videos": {name: item["outputs"] for name, item in rendered.items()},
        "checks": {
            "all_mp4_exist": all(item.get("checks", {}).get("mp4_exists", False) for item in rendered.values()),
            "all_metrics_csv_exist_for_primary_videos": all(
                item.get("checks", {}).get("metrics_csv_exists", True)
                for name, item in rendered.items()
                if name != "guided_vs_unguided_action_control"
            ),
            "does_not_claim_complete_beyondmimic_reproduction": True,
            "does_not_claim_real_robot": True,
        },
        "limitations": [
            "The best local Stage-1 teacher remains weak; downstream videos are evidence of the local control pipeline, not proof of paper-level motion quality.",
            "reference_action_control uses teacher-rollout motion time steps, which are discontinuous because the weak teacher resets/re-samples commands; it is a PD control diagnostic, not a clean original-dataset replay.",
            "reference_pose_replay is the clean continuous-reference visualization; it writes qpos frame-by-frame and therefore is not control evidence.",
            "Root assist is enabled to keep the robot centered for visualization.",
            "Large MP4s are local report assets and should not be committed to GitHub.",
        ],
    }
    write_json(OUT_ROOT / "lafan1_paper_contract_video_suite_summary.json", summary)
    lines = [
        "# LAFAN1 Paper-Contract MuJoCo Action-Control Videos",
        "",
        "These MP4s separate continuous reference visualization from MuJoCo action-to-PD control diagnostics generated from the current paper-contract Stage-1 teacher, VAE, diffusion, and offline guidance artifacts.",
        "",
        "They are not Isaac rendered MP4s, not official BeyondMimic checkpoints, not real robot results, and not paper-level Fig.5/Fig.6 closed-loop reproduction.",
        "",
        "## Reference Semantics",
        "",
        "- `reference_pose_replay` is the clean continuous LAFAN1 reference visualization. It writes root pose and 29 joint positions frame-by-frame with `mj_forward`; use this to show what the source motion looks like.",
        "- `reference_action_control` is a PD tracking diagnostic. It uses discontinuous teacher-rollout time steps and MuJoCo `mj_step`; do not treat it as the original dataset motion replay.",
        "",
        "## Selected Teacher",
        "",
        f"- Shard: `{selected['path']}`",
        f"- Rank/env: `{selected['rank']}/{selected['env_index']}`",
        f"- First done frame: `{selected['first_done']}`",
        f"- Mean reward: `{selected['mean_reward']:.6f}`",
        f"- Teacher motion-time-step non-+1 jumps: `{selected['motion_time_step_discontinuity']['non_plus_one_count']}`",
        "",
        "## Videos",
        "",
    ]
    for name, item in rendered.items():
        outputs = item.get("outputs", {})
        lines.append(f"- `{name}`: `{outputs.get('mp4', '')}`")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "The action-control videos use MuJoCo `mj_step` and 29 position actuators, but also use a root-assist stabilizer for report-ready visualization. The reference-pose replay writes qpos frame-by-frame. The suite should be described as local virtual simulation evidence, not full BeyondMimic reproduction.",
            "",
        ]
    )
    (OUT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    frames = int(os.environ.get("BM_LAFAN1_VIDEO_FRAMES", "450"))
    clip_actions = float(os.environ.get("BM_LAFAN1_ACTION_CLIP", "3.0"))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    model, vae_cfg = load_vae()
    selected = select_teacher_sequence()
    ref_joint, root_pos, root_quat, ref_meta = load_reference_for_time_steps(selected["motion_time_steps"], frames)

    import mujoco

    action_rows = load_action_rows()
    model_path = Path(os.environ.get("BM_MUJOCO_G1_XML", str(DEFAULT_MODEL))).expanduser()
    patched_xml = model_path.parent / "g1_mocap_29dof_lafan1_paper_contract_pd.xml"
    patch_joints_and_actuators(model_path, patched_xml, action_rows)
    mujoco_model = mujoco.MjModel.from_xml_path(str(patched_xml))
    action_scale = np.asarray([float(row["action_scale"]) for row in action_rows], dtype=np.float64)
    default_joint_pos, default_source, default_note = parse_default_joint_position(action_rows)
    ctrlrange = np.asarray(mujoco_model.actuator_ctrlrange, dtype=np.float64)

    teacher_actions = selected["actions"]
    vae_actions = vae_reconstruct_actions(model, selected["policy_obs"], teacher_actions)
    denoised_actions = decode_guidance_actions(model, "pred_velocity_command_batch0")
    guided_actions = decode_guidance_actions(model, "guided_velocity_command_max_scale_batch0")

    common_meta = {
        "best_teacher_sweep_json": str(BEST_TEACHER_SWEEP_JSON),
        "teacher_rollout_json": str(TEACHER_ROLLOUT_JSON),
        "vae_checkpoint": str(VAE_CKPT),
        "vae_config": vae_cfg,
        "guidance_samples": str(GUIDANCE_SAMPLES),
        "default_joint_pos_source": default_source,
        "default_joint_pos_note": default_note,
        "action_scale_formula_source": str(ACTION_SCALE_AUDIT),
        "video_frames": frames,
        "video_duration_seconds": frames / int(os.environ.get("BM_LAFAN1_VIDEO_FPS", "30")),
        "weak_teacher_warning": "The teacher rollout selected here has first_done == 0, so videos are control-pipeline evidence rather than high-quality motion tracking evidence.",
    }

    specs: list[tuple[str, np.ndarray, dict[str, Any]]] = [
        (
            "reference_action_control",
            ref_joint,
            {
                "claim_level": "MuJoCo PD control tracking of FK-repaired LAFAN1 reference joint targets from the selected teacher rollout time steps",
                "target_source": "reference_joint_targets_from_motion_bundle",
                **ref_meta,
            },
        ),
    ]
    for name, actions, claim in [
        (
            "teacher_policy_action_control",
            teacher_actions,
            "MuJoCo action-to-PD visualization of current best local Stage-1 PPO teacher actions; weak teacher, not paper-level tracking",
        ),
        (
            "vae_reconstructed_action_control",
            vae_actions,
            "MuJoCo action-to-PD visualization of VAE reconstructed actions from the local teacher rollout",
        ),
        (
            "diffusion_denoised_latent_action_control",
            denoised_actions,
            "MuJoCo action-to-PD visualization of actions decoded from local denoised state-latent diffusion samples",
        ),
        (
            "guided_latent_action_control",
            guided_actions,
            "MuJoCo action-to-PD visualization of actions decoded from local classifier-guided latent samples",
        ),
    ]:
        targets, meta = action_to_joint_targets(actions, default_joint_pos, action_scale, ctrlrange, frames, clip_actions)
        specs.append(
            (
                name,
                targets,
                {
                    "claim_level": claim,
                    "target_source": "default_joint_pos_plus_action_scale_times_action",
                    "theta_sp_formula": "theta_sp = theta_default + action_scale * clip(action, -clip, clip)",
                    **meta,
                },
            )
        )

    rendered: dict[str, Any] = {}
    try:
        rendered["reference_pose_replay"] = render_reference_pose_replay_video(frames)
        for name, joint_targets, source_meta in specs:
            rendered[name] = render_action_control_video(name, joint_targets, root_pos, root_quat, source_meta, common_meta)
        rendered["guided_vs_unguided_action_control"] = render_side_by_side(
            rendered["diffusion_denoised_latent_action_control"], rendered["guided_latent_action_control"]
        )
        write_top_level_summary(rendered, selected)
    except Exception as exc:  # noqa: BLE001
        write_json(
            OUT_ROOT / "failed_lafan1_paper_contract_video_suite_summary.json",
            {
                "status": "failed",
                "timestamp_utc": utc_now(),
                "experiment_type": "lafan1_paper_contract_mujoco_action_control_video_suite",
                "error": traceback_payload(exc),
            },
        )
        raise


if __name__ == "__main__":
    main()
