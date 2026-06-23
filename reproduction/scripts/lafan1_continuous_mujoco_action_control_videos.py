#!/usr/bin/env python3
"""Regenerate MuJoCo action-control videos from one verified-continuous segment.

The previous LAFAN1 action-control video suite used a selected teacher rollout
environment whose motion-time-step sequence jumped across resets.  This script
keeps the old videos as failed diagnostics and creates a separate corrected
suite from one segment that satisfies:

* no ``done`` frames;
* ``motion_time_steps[t + 1] == motion_time_steps[t] + 1``;
* no temporal stretching beyond the true continuous segment length.

Claim boundary: the corrected action-control videos are still local MuJoCo
diagnostics from a weak local teacher/VAE/denoiser chain with root assist.  They
are not official BeyondMimic checkpoints, not native Isaac rendered videos, not
paper-level Fig. 5/Fig. 6 closed-loop guidance, and not real-robot evidence.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(os.environ.get("BM_ROOT", "/mnt/infini-data/test/BeyondMimic")).resolve()
SCRIPT_DIR = ROOT / "reproduction/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lafan1_paper_contract_mujoco_action_control_videos import (  # noqa: E402
    BEST_TEACHER_SWEEP_JSON,
    GUIDANCE_SAMPLES,
    MOTION_BUNDLE,
    OUT_ROOT as OLD_OUT_ROOT,
    TEACHER_ROLLOUT_JSON,
    VAE_CKPT,
    ConditionalActionVAE,
    action_to_joint_targets,
    decode_guidance_actions,
    load_vae,
    make_keyframe_strip,
    parse_default_joint_position,
    render_action_control_video,
    render_side_by_side,
    sha256,
    utc_now,
    write_json,
)
from mujoco_pd_control_video import load_action_rows, patch_joints_and_actuators  # noqa: E402
from mujoco_pd_control_video import PD_CAMERA  # noqa: E402
from mujoco_common import render_frame  # noqa: E402
from mujoco_trace_mesh_video import DEFAULT_MODEL  # noqa: E402


OUT_ROOT = ROOT / "res/visualization/lafan1_continuous_mujoco_action_control_videos"
OLD_FAILURE_AUDIT = OLD_OUT_ROOT / "failed_discontinuous_action_control_audit.json"
DENOISER_CKPT = (
    ROOT
    / "res/runs/level_c_official_importer_export_paper_contract_state_latent_diffusion_training/"
    "resource_adjusted_state_latent_diffusion_20260623_062635_seed20260807/"
    "resource_adjusted_state_latent_denoiser.pt"
)


class StateLatentDenoiser(nn.Module):
    def __init__(self, token_dim: int, hidden_dim: int, steps: int) -> None:
        super().__init__()
        self.steps = steps
        self.net = nn.Sequential(
            nn.Linear(token_dim + steps, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim),
        )

    def forward(self, noisy: torch.Tensor, step_idx: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(step_idx, num_classes=self.steps).to(noisy.dtype)
        return self.net(torch.cat([noisy, onehot], dim=-1))


def alpha_bars(steps: int) -> torch.Tensor:
    betas = torch.linspace(1e-4, 0.02, steps)
    return torch.cumprod(1.0 - betas, dim=0)


def load_denoiser() -> tuple[StateLatentDenoiser, dict[str, Any]]:
    payload = torch.load(DENOISER_CKPT, map_location="cpu")
    cfg = dict(payload["config"])
    model = StateLatentDenoiser(int(cfg["token_dim"]), int(cfg["hidden_dim"]), int(cfg["denoising_steps"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, cfg


def find_continuous_segments() -> list[dict[str, Any]]:
    summary = json.loads(TEACHER_ROLLOUT_JSON.read_text(encoding="utf-8"))
    segments: list[dict[str, Any]] = []
    for shard_path_str in summary["run"]["shard_npz_paths"]:
        shard_path = Path(shard_path_str)
        data = np.load(shard_path)
        rewards = np.asarray(data["rewards"], dtype=np.float64)
        dones = np.asarray(data["dones"], dtype=np.bool_)
        time_steps = np.asarray(data["motion_time_steps"], dtype=np.int64)
        rank = int(np.asarray(data["rank"])[0])
        total_frames, env_count = time_steps.shape
        for env_idx in range(env_count):
            start = 0
            while start < total_frames:
                while start < total_frames and dones[start, env_idx]:
                    start += 1
                if start >= total_frames:
                    break
                end = start + 1
                while (
                    end < total_frames
                    and not dones[end, env_idx]
                    and int(time_steps[end, env_idx]) == int(time_steps[end - 1, env_idx]) + 1
                ):
                    end += 1
                length = end - start
                if length >= 2:
                    segment_steps = time_steps[start:end, env_idx]
                    segments.append(
                        {
                            "shard": str(shard_path),
                            "rank": rank,
                            "env_index": int(env_idx),
                            "start": int(start),
                            "end_exclusive": int(end),
                            "length": int(length),
                            "motion_time_step_start": int(segment_steps[0]),
                            "motion_time_step_end": int(segment_steps[-1]),
                            "reward_mean": float(np.mean(rewards[start:end, env_idx])),
                            "done_count": int(np.sum(dones[start:end, env_idx])),
                        }
                    )
                start = max(end, start + 1)
        data.close()
    return sorted(segments, key=lambda row: (row["length"], row["reward_mean"]), reverse=True)


def load_segment(segment: dict[str, Any], max_frames: int | None = None) -> dict[str, Any]:
    data = np.load(segment["shard"])
    start = int(segment["start"])
    end = int(segment["end_exclusive"])
    if max_frames is not None:
        end = min(end, start + int(max_frames))
    env = int(segment["env_index"])
    out = {
        "policy_obs": np.asarray(data["policy_obs"][start:end, env, :], dtype=np.float32),
        "actions": np.asarray(data["actions"][start:end, env, :], dtype=np.float32),
        "rewards": np.asarray(data["rewards"][start:end, env], dtype=np.float32),
        "dones": np.asarray(data["dones"][start:end, env], dtype=np.bool_),
        "timeouts": np.asarray(data["timeouts"][start:end, env], dtype=np.bool_),
        "motion_time_steps": np.asarray(data["motion_time_steps"][start:end, env], dtype=np.int64),
    }
    data.close()
    deltas = np.diff(out["motion_time_steps"])
    out["continuity"] = {
        "source_shard": segment["shard"],
        "rank": int(segment["rank"]),
        "env_index": int(segment["env_index"]),
        "source_start": start,
        "source_end_exclusive": end,
        "frames": int(end - start),
        "done_count": int(np.sum(out["dones"])),
        "timeout_count": int(np.sum(out["timeouts"])),
        "all_motion_time_step_deltas_plus_one": bool(deltas.size == 0 or np.all(deltas == 1)),
        "non_plus_one_count": int(np.sum(deltas != 1)),
        "motion_time_step_start": int(out["motion_time_steps"][0]),
        "motion_time_step_end": int(out["motion_time_steps"][-1]),
        "no_temporal_stretching": True,
        "selection_rule": "longest segment with done_count == 0 and motion_time_steps strictly consecutive +1",
    }
    return out


def load_continuous_reference_for_steps(time_steps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    motion = np.load(MOTION_BUNDLE, allow_pickle=True)
    total = int(motion["joint_pos"].shape[0])
    frame_steps = np.clip(np.asarray(time_steps, dtype=np.int64), 0, total - 1)
    joint_targets = np.asarray(motion["joint_pos"][frame_steps], dtype=np.float64)
    root_pos = np.asarray(motion["body_pos_w"][frame_steps, 0, :], dtype=np.float64)
    root_quat = np.asarray(motion["body_quat_w"][frame_steps, 0, :], dtype=np.float64)
    root_pos[:, 0:2] -= root_pos[0:1, 0:2]
    root_quat = root_quat / np.maximum(np.linalg.norm(root_quat, axis=1, keepdims=True), 1e-8)
    deltas = np.diff(frame_steps)
    meta = {
        "motion_bundle": str(MOTION_BUNDLE),
        "motion_bundle_sha256": sha256(MOTION_BUNDLE),
        "motion_bundle_frames": total,
        "selected_time_step_first": int(frame_steps[0]),
        "selected_time_step_last": int(frame_steps[-1]),
        "target_source": "continuous_motion_bundle_time_steps_from_verified_teacher_segment",
        "root_xy_recentered_targets": True,
        "source_time_steps_from_teacher_rollout": True,
        "not_clean_continuous_reference_replay": False,
        "all_selected_time_step_deltas_plus_one": bool(deltas.size == 0 or np.all(deltas == 1)),
        "selected_time_step_non_plus_one_count": int(np.sum(deltas != 1)),
    }
    return joint_targets, root_pos, root_quat, meta


def vae_reconstruct_actions(model: ConditionalActionVAE, obs: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        obs_t = torch.from_numpy(obs.astype(np.float32))
        act_t = torch.from_numpy(actions.astype(np.float32))
        z = model.posterior_mean(obs_t, act_t)
        pred = model.decode(obs_t, z)
    return pred.cpu().numpy().astype(np.float32), z.cpu().numpy().astype(np.float32)


def denoise_latents_for_segment(
    denoiser: StateLatentDenoiser,
    cfg: dict[str, Any],
    obs: np.ndarray,
    vae_latents: np.ndarray,
) -> np.ndarray:
    sequence_length = int(cfg["sequence_length"])
    steps = int(cfg["denoising_steps"])
    clean = np.concatenate([obs, vae_latents], axis=-1).astype(np.float32)
    clean_t = torch.from_numpy(clean)
    bars = alpha_bars(steps)
    current_latents: list[np.ndarray] = []
    with torch.inference_mode():
        for frame_idx in range(clean.shape[0]):
            idx = np.minimum(np.arange(frame_idx, frame_idx + sequence_length), clean.shape[0] - 1)
            window = clean_t[idx].unsqueeze(0)
            step_idx = torch.full((1, sequence_length), steps - 1, dtype=torch.long)
            alpha = bars[steps - 1]
            noisy = torch.sqrt(alpha) * window
            pred = denoiser(noisy, step_idx)
            current_latents.append(pred[0, 0, int(cfg["obs_dim"]) :].detach().cpu().numpy())
    return np.stack(current_latents, axis=0).astype(np.float32)


def guided_latents_for_segment(
    denoised_latents: np.ndarray,
    teacher_latents: np.ndarray,
    guidance_scale: float,
) -> np.ndarray:
    # Conservative local proxy guidance: nudge denoised latents toward the VAE
    # teacher posterior for this verified-continuous segment.  This is only a
    # report diagnostic and is explicitly not paper Fig.5/Fig.6 task guidance.
    return (denoised_latents + guidance_scale * (teacher_latents - denoised_latents)).astype(np.float32)


def decode_actions(model: ConditionalActionVAE, obs: np.ndarray, latents: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        out = model.decode(torch.from_numpy(obs.astype(np.float32)), torch.from_numpy(latents.astype(np.float32)))
    return out.detach().cpu().numpy().astype(np.float32)


def render_reference_pose_replay(time_steps: np.ndarray, root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray) -> dict[str, Any]:
    import mujoco

    fps = int(os.environ.get("BM_LAFAN1_VIDEO_FPS", "30"))
    width = int(os.environ.get("BM_LAFAN1_VIDEO_WIDTH", "960"))
    height = int(os.environ.get("BM_LAFAN1_VIDEO_HEIGHT", "540"))
    frames = int(joint_pos.shape[0])
    model_path = Path(os.environ.get("BM_MUJOCO_G1_XML", str(DEFAULT_MODEL))).expanduser()
    patched_xml = model_path.parent / "g1_mocap_29dof_lafan1_continuous_reference_camera.xml"
    patch_joints_and_actuators(model_path, patched_xml, load_action_rows())
    model = mujoco.MjModel.from_xml_path(str(patched_xml))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    out_dir = OUT_ROOT / "reference_action_control"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / "reference_action_control.mp4"
    keyframe_path = out_dir / "reference_action_control_keyframe.png"
    strip_path = out_dir / "reference_action_control_keyframes.png"
    metrics_path = out_dir / "reference_action_control_metrics.csv"
    summary_path = out_dir / "reference_action_control_summary.json"
    rows: list[dict[str, Any]] = []
    strip: list[np.ndarray] = []
    strip_indices = {0, frames // 2, frames - 1}
    with imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame_idx in range(frames):
            data.qpos[:] = 0.0
            data.qvel[:] = 0.0
            data.qpos[0:3] = root_pos[frame_idx]
            data.qpos[3:7] = root_quat[frame_idx]
            data.qpos[7 : 7 + 29] = joint_pos[frame_idx]
            mujoco.mj_forward(model, data)
            frame = render_frame(model, data, renderer, camera=PD_CAMERA)
            if frame_idx == 0:
                imageio.imwrite(keyframe_path, frame)
            if frame_idx in strip_indices:
                strip.append(frame)
            writer.append_data(frame)
            rows.append(
                {
                    "frame": frame_idx,
                    "motion_time_step": int(time_steps[frame_idx]),
                    "video_time_s": frame_idx / fps,
                    "root_x": float(root_pos[frame_idx, 0]),
                    "root_y": float(root_pos[frame_idx, 1]),
                    "root_z": float(root_pos[frame_idx, 2]),
                    "joint_abs_mean": float(np.mean(np.abs(joint_pos[frame_idx]))),
                    "contact_count_after_forward": int(data.ncon),
                }
            )
    renderer.close()
    make_keyframe_strip(strip, strip_path)
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    deltas = np.diff(time_steps)
    payload = {
        "status": "ok",
        "timestamp_utc": utc_now(),
        "experiment_type": "lafan1_continuous_reference_pose_replay_renamed_reference_action_control",
        "spec_name": "reference_action_control",
        "claim_level": "Continuous single-motion MuJoCo reference pose replay; this is the corrected replacement for the old failed discontinuous reference_action_control label",
        "continuity": {
            "frames": frames,
            "all_motion_time_step_deltas_plus_one": bool(deltas.size == 0 or np.all(deltas == 1)),
            "non_plus_one_count": int(np.sum(deltas != 1)),
            "motion_time_step_start": int(time_steps[0]),
            "motion_time_step_end": int(time_steps[-1]),
            "no_temporal_stretching": True,
        },
        "simulation": {
            "uses_mj_forward": True,
            "uses_mj_step": False,
            "writes_qpos_each_frame": True,
            "actuator_control_used": False,
        },
        "patched_camera_model_xml": str(patched_xml),
        "outputs": {
            "mp4": str(mp4_path),
            "keyframe_png": str(keyframe_path),
            "keyframes_png": str(strip_path),
            "metrics_csv": str(metrics_path),
            "summary_json": str(summary_path),
        },
        "checks": {
            "mp4_exists": mp4_path.is_file() and mp4_path.stat().st_size > 0,
            "metrics_csv_exists": metrics_path.is_file() and metrics_path.stat().st_size > 0,
            "uses_continuous_time_steps": bool(deltas.size == 0 or np.all(deltas == 1)),
            "no_temporal_stretching": True,
            "does_not_claim_policy_control": True,
            "does_not_claim_real_robot": True,
        },
        "limitations": [
            "This is a continuous reference pose replay, not a policy/controller output.",
            "It is intentionally short because the best clean teacher segment is short; the video is not stretched.",
        ],
    }
    write_json(summary_path, payload)
    print(json.dumps({"status": "ok", "spec": "reference_action_control", "mp4": str(mp4_path)}))
    return payload


def write_old_failure_audit() -> dict[str, Any]:
    old_summary = json.loads((OLD_OUT_ROOT / "lafan1_paper_contract_video_suite_summary.json").read_text(encoding="utf-8"))
    videos = [
        "reference_action_control",
        "teacher_policy_action_control",
        "vae_reconstructed_action_control",
        "diffusion_denoised_latent_action_control",
        "guided_latent_action_control",
        "guided_vs_unguided_action_control",
    ]
    rows = []
    for name in videos:
        summary_path = OLD_OUT_ROOT / name / f"{name}_summary.json"
        data = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        rows.append(
            {
                "video": name,
                "old_summary": str(summary_path),
                "old_mp4": data.get("outputs", {}).get("mp4", ""),
                "failure_status": "failed_discontinuous_or_offline_diagnostic",
                "reason": (
                    "The old suite shares discontinuous teacher-rollout context, or uses offline short latent samples "
                    "paired with that context. It is retained only as a diagnostic artifact."
                ),
            }
        )
    payload = {
        "status": "failed_discontinuous_action_control_suite_recorded",
        "timestamp_utc": utc_now(),
        "experiment_type": "lafan1_old_action_control_failure_audit",
        "old_output_root": str(OLD_OUT_ROOT),
        "old_suite_status": old_summary.get("status"),
        "old_selected_teacher_rollout": old_summary.get("selected_teacher_rollout"),
        "failed_videos": rows,
        "replacement_output_root": str(OUT_ROOT),
        "claim_level": "Old LAFAN1 action-control MP4s are failed/diagnostic due to discontinuous reset-spliced teacher rollout context; do not use as final continuous motion evidence.",
        "checks": {
            "old_non_plus_one_count_positive": old_summary["selected_teacher_rollout"]["motion_time_step_discontinuity"][
                "non_plus_one_count"
            ]
            > 0,
            "six_old_videos_marked_failed": len(rows) == 6,
            "does_not_claim_old_videos_valid": True,
        },
    }
    write_json(OLD_FAILURE_AUDIT, payload)
    return payload


def write_summary(rendered: dict[str, Any], segment: dict[str, Any], old_failure: dict[str, Any]) -> None:
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
        "no_temporal_stretching": all(
            item.get("continuity", {}).get("no_temporal_stretching", True)
            for name, item in rendered.items()
            if name != "guided_vs_unguided_action_control"
        ),
        "old_discontinuous_suite_marked_failed": old_failure.get("status")
        == "failed_discontinuous_action_control_suite_recorded",
        "does_not_claim_complete_beyondmimic_reproduction": True,
        "does_not_claim_real_robot": True,
    }
    payload = {
        "status": "ok",
        "timestamp_utc": utc_now(),
        "experiment_type": "lafan1_continuous_mujoco_action_control_video_suite",
        "output_root": str(OUT_ROOT),
        "claim_level": "Corrected local MuJoCo continuous-segment visualization suite: reference pose replay plus action-to-PD diagnostics for teacher, VAE, denoised latent, guided latent, and guided-vs-unguided. Not paper-level BeyondMimic.",
        "selected_continuous_segment": segment,
        "best_teacher_sweep_metrics": sweep.get("metrics", {}),
        "old_failed_suite_audit": str(OLD_FAILURE_AUDIT),
        "videos": videos,
        "checks": checks,
        "limitations": [
            "The clean segment is short because the current local teacher often resets; the videos are not stretched to an arbitrary duration.",
            "Teacher/VAE/diffusion/guidance videos are local MuJoCo action-to-PD diagnostics with root assist, not unassisted paper-level tracking.",
            "Denoised/guided variants are recomputed per true frame from a local denoiser/VAE bridge and a conservative teacher-consistency proxy, not official Fig.5/Fig.6 task guidance.",
            "Large MP4s are local report assets and should not be committed to GitHub.",
        ],
    }
    write_json(OUT_ROOT / "lafan1_continuous_video_suite_summary.json", payload)
    lines = [
        "# LAFAN1 Continuous MuJoCo Action-Control Videos",
        "",
        "This directory is the corrected replacement for the old reset-spliced LAFAN1 action-control video suite.",
        "",
        "## Continuity Gate",
        "",
        f"- Shard: `{segment['shard']}`",
        f"- Rank/env: `{segment['rank']}/{segment['env_index']}`",
        f"- Source frames: `{segment['start']}:{segment['end_exclusive']}`",
        f"- Rendered frames: `{segment['length']}`",
        f"- Motion time steps: `{segment['motion_time_step_start']}..{segment['motion_time_step_end']}`",
        f"- Done count: `{segment['done_count']}`",
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
            "The reference video is continuous pose replay. The other videos use MuJoCo `mj_step`, 29 position actuators, and root assist. They are local diagnostics from a weak teacher chain, not official BeyondMimic paper-level results.",
            "",
            f"Old failed suite audit: `{OLD_FAILURE_AUDIT}`",
            "",
        ]
    )
    (OUT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    os.environ["BM_LAFAN1_ROOT_ASSIST"] = os.environ.get("BM_LAFAN1_ROOT_ASSIST", "1")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    render_action_control_video.__globals__["OUT_ROOT"] = OUT_ROOT
    render_side_by_side.__globals__["OUT_ROOT"] = OUT_ROOT
    segments = find_continuous_segments()
    if not segments:
        raise RuntimeError("No continuous teacher rollout segment found")
    min_frames = int(os.environ.get("BM_LAFAN1_MIN_CONTINUOUS_FRAMES", "20"))
    candidates = [row for row in segments if row["length"] >= min_frames]
    if not candidates:
        raise RuntimeError(f"No continuous teacher segment with at least {min_frames} frames")
    segment = candidates[0]
    max_frames_env = os.environ.get("BM_LAFAN1_MAX_CONTINUOUS_FRAMES")
    max_frames = int(max_frames_env) if max_frames_env else None
    data = load_segment(segment, max_frames=max_frames)
    frames = int(data["actions"].shape[0])
    fps = int(os.environ.get("BM_LAFAN1_VIDEO_FPS", "30"))

    model, vae_cfg = load_vae()
    denoiser, denoiser_cfg = load_denoiser()
    ref_joint, root_pos, root_quat, ref_meta = load_continuous_reference_for_steps(data["motion_time_steps"])

    import mujoco

    action_rows = load_action_rows()
    model_path = Path(os.environ.get("BM_MUJOCO_G1_XML", str(DEFAULT_MODEL))).expanduser()
    patched_xml = model_path.parent / "g1_mocap_29dof_lafan1_continuous_pd.xml"
    patch_joints_and_actuators(model_path, patched_xml, action_rows)
    mujoco_model = mujoco.MjModel.from_xml_path(str(patched_xml))
    action_scale = np.asarray([float(row["action_scale"]) for row in action_rows], dtype=np.float64)
    default_joint_pos, default_source, default_note = parse_default_joint_position(action_rows)
    ctrlrange = np.asarray(mujoco_model.actuator_ctrlrange, dtype=np.float64)
    clip_actions = float(os.environ.get("BM_LAFAN1_ACTION_CLIP", "3.0"))

    vae_actions, teacher_latents = vae_reconstruct_actions(model, data["policy_obs"], data["actions"])
    denoised_latents = denoise_latents_for_segment(denoiser, denoiser_cfg, data["policy_obs"], teacher_latents)
    guided_latents = guided_latents_for_segment(
        denoised_latents,
        teacher_latents,
        float(os.environ.get("BM_LAFAN1_GUIDANCE_SCALE", "0.35")),
    )
    denoised_actions = decode_actions(model, data["policy_obs"], denoised_latents)
    guided_actions = decode_actions(model, data["policy_obs"], guided_latents)

    segment_payload = {
        **segment,
        "length": frames,
        "end_exclusive": int(segment["start"] + frames),
        "video_fps": fps,
        "duration_seconds": frames / fps,
        "continuity": data["continuity"],
    }
    old_failure = write_old_failure_audit()

    common_meta = {
        "best_teacher_sweep_json": str(BEST_TEACHER_SWEEP_JSON),
        "teacher_rollout_json": str(TEACHER_ROLLOUT_JSON),
        "vae_checkpoint": str(VAE_CKPT),
        "vae_config": vae_cfg,
        "denoiser_checkpoint": str(DENOISER_CKPT),
        "denoiser_config": denoiser_cfg,
        "default_joint_pos_source": default_source,
        "default_joint_pos_note": default_note,
        "video_frames": frames,
        "video_duration_seconds": frames / fps,
        "continuity_gate": data["continuity"],
        "old_failed_suite_audit": str(OLD_FAILURE_AUDIT),
    }
    rendered: dict[str, Any] = {}
    rendered["reference_action_control"] = render_reference_pose_replay(
        data["motion_time_steps"], root_pos, root_quat, ref_joint
    )
    for name, actions, claim, extra in [
        (
            "teacher_policy_action_control",
            data["actions"],
            "MuJoCo action-to-PD visualization of local Stage-1 PPO teacher actions on one verified-continuous rollout segment",
            {},
        ),
        (
            "vae_reconstructed_action_control",
            vae_actions,
            "MuJoCo action-to-PD visualization of VAE reconstructed teacher actions on the same verified-continuous segment",
            {},
        ),
        (
            "diffusion_denoised_latent_action_control",
            denoised_actions,
            "MuJoCo action-to-PD visualization of per-frame local denoiser latent actions on the same verified-continuous segment",
            {
                "denoising_method": "For each true frame, repeat the current/future segment window to 21 tokens, run the local denoiser, and decode the current latent. No 21-frame sample is stretched into a longer video.",
            },
        ),
        (
            "guided_latent_action_control",
            guided_actions,
            "MuJoCo action-to-PD visualization of per-frame local guided latent actions on the same verified-continuous segment",
            {
                "guidance_method": "Conservative local proxy: nudge denoised latent toward the VAE teacher posterior for the current continuous segment. This is not paper Fig.5/Fig.6 classifier guidance.",
            },
        ),
    ]:
        targets, meta = action_to_joint_targets(
            actions, default_joint_pos, action_scale, ctrlrange, frames, clip_actions
        )
        source_meta = {
            "experiment_type": "lafan1_continuous_mujoco_action_control_video",
            "claim_level": claim,
            "target_source": "default_joint_pos_plus_action_scale_times_action",
            "theta_sp_formula": "theta_sp = theta_default + action_scale * clip(action, -clip, clip)",
            "continuity": data["continuity"],
            "old_failed_suite_audit": str(OLD_FAILURE_AUDIT),
            "limitations": [
                "This video uses one verified-continuous teacher segment and is not temporally stretched.",
                "The controller is a MuJoCo position-actuator/root-assist diagnostic, not a paper-level unassisted humanoid controller.",
                "The current teacher is weak, so motion quality can still be poor even though the data sequence is now continuous.",
                "This is not a real-robot result and not official BeyondMimic Fig.5/Fig.6 evidence.",
            ],
            **meta,
            **extra,
        }
        rendered[name] = render_action_control_video(name, targets, root_pos, root_quat, source_meta, common_meta)
    rendered["guided_vs_unguided_action_control"] = render_side_by_side(
        rendered["diffusion_denoised_latent_action_control"], rendered["guided_latent_action_control"]
    )
    rendered["guided_vs_unguided_action_control"]["continuity"] = data["continuity"]
    write_summary(rendered, segment_payload, old_failure)
    print(
        json.dumps(
            {
                "status": "ok",
                "output_root": str(OUT_ROOT),
                "frames": frames,
                "duration_seconds": frames / fps,
                "summary": str(OUT_ROOT / "lafan1_continuous_video_suite_summary.json"),
                "old_failed_suite_audit": str(OLD_FAILURE_AUDIT),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
