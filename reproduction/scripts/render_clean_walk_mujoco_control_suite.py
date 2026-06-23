#!/usr/bin/env python3
"""Render clean 15 s MuJoCo walk videos for reference/teacher/VAE/diffusion/guidance.

The previous Stage-1 multi-source video suite failed as a presentation asset:
the original selector picked a near-floor target, and the later quality-gated
teacher/VAE/diffusion videos were only 1 s and still crouched.  This script
uses one clean continuous LAFAN1 walking window and runs every variant through
the same MuJoCo PD physics loop.

For the learned variants, the current local teacher is still weak.  To make a
readable walk demo without pretending the learned policy is solved, the default
output is a reference-anchored presentation controller:

    final target = (1 - model_target_weight) * reference_q
                 + model_target_weight * model_q_target

The value is recorded in every summary.  Set
``BM_CLEAN_SUITE_MODEL_TARGET_WEIGHT=1`` to run pure model targets; that is
expected to be much less stable with the current teacher.

Claim boundary: these are local MuJoCo presentation/diagnostic videos, not
official BeyondMimic checkpoints, not IsaacLab rendered results, not real robot
evidence, and not paper-level Fig. 5/Fig. 6 reproduction.
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
from torch.nn import functional as F

ROOT = Path(os.environ.get("BM_ROOT", "/mnt/infini-data/test/BeyondMimic")).resolve()
SCRIPT_DIR = ROOT / "reproduction/scripts"
MUJOCO_SCRIPT_DIR = ROOT / "mujoco_mp4/scripts"
for path in [SCRIPT_DIR, MUJOCO_SCRIPT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lafan1_paper_contract_mujoco_action_control_videos import (  # noqa: E402
    action_to_joint_targets,
    parse_default_joint_position,
)
from mujoco_common import render_frame, sha256, traceback_payload, utc_now, write_json  # noqa: E402
from mujoco_pd_control_video import (  # noqa: E402
    PD_CAMERA,
    actuator_joint_order,
    apply_root_assist,
    load_action_rows,
    normalize_quat_wxyz,
    patch_joints_and_actuators,
    quat_conj,
    quat_error_rotvec,
    quat_mul,
    quat_to_roll_pitch_yaw,
)
from mujoco_trace_mesh_video import DEFAULT_MODEL  # noqa: E402
from render_clean_walk_mujoco_pd_control_demo import (  # noqa: E402
    DEFAULT_MOTION,
    downsample_motion_indices,
    make_keyframe_strip,
    segment_from_start,
    scalar_fps,
    write_failure_audit,
)
from stage1_multisource_quality_gated_native_ppo_mujoco_probe import (  # noqa: E402
    ANCHOR_BODY_NAME,
    BODY_NAMES,
    Actor,
    body_id_map,
    build_obs,
    infer_body_indices_from_reference,
    quat_apply,
)


DEFAULT_OUT_ROOT = ROOT / "res/visualization/clean_walk_mujoco_control_suite"
OUT_ROOT = Path(os.environ.get("BM_CLEAN_SUITE_OUT_ROOT", str(DEFAULT_OUT_ROOT))).expanduser().resolve()
BEST_TEACHER_JSON = (
    Path(
        os.environ.get(
            "BM_CLEAN_SUITE_BEST_TEACHER_JSON",
            str(
                ROOT
                / "res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/"
                "stage1_multisource_best_teacher.json"
            ),
        )
    )
    .expanduser()
    .resolve()
)
VAE_CKPT = (
    Path(
        os.environ.get(
            "BM_CLEAN_SUITE_VAE_CKPT",
            str(
                ROOT
                / "res/runs/level_c_stage1_multisource_teacher_rollout_vae_training/"
                "resource_adjusted_teacher_rollout_vae_20260623_135755_seed20260855/"
                "resource_adjusted_teacher_rollout_action_vae.pt"
            ),
        )
    )
    .expanduser()
    .resolve()
)
DENOISER_CKPT = (
    Path(
        os.environ.get(
            "BM_CLEAN_SUITE_DENOISER_CKPT",
            str(
                ROOT
                / "res/runs/level_c_stage1_multisource_state_latent_diffusion_training/"
                "resource_adjusted_state_latent_diffusion_20260623_140110_seed20260857/"
                "resource_adjusted_state_latent_denoiser.pt"
            ),
        )
    )
    .expanduser()
    .resolve()
)


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


@dataclass
class PolicyBundle:
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


@dataclass
class ModelBundle:
    policy: PolicyBundle
    vae: ConditionalActionVAE
    vae_cfg: dict[str, Any]
    denoiser: StateLatentDenoiser
    denoiser_cfg: dict[str, Any]


def alpha_bars(steps: int) -> torch.Tensor:
    betas = torch.linspace(1e-4, 0.02, steps)
    return torch.cumprod(1.0 - betas, dim=0)


def resolve_policy_checkpoint(payload: dict[str, Any]) -> Path:
    """Resolve the policy checkpoint from local audit JSON variants."""
    candidates = [
        payload.get("best_checkpoint", {}).get("checkpoint") if isinstance(payload.get("best_checkpoint"), dict) else None,
        payload.get("inputs", {}).get("checkpoint") if isinstance(payload.get("inputs"), dict) else None,
        payload.get("checkpoint"),
        payload.get("policy_checkpoint"),
    ]
    for value in candidates:
        if value:
            return Path(str(value)).expanduser().resolve()
    raise KeyError(f"No checkpoint path found in {BEST_TEACHER_JSON}")


def load_models() -> ModelBundle:
    best = json.loads(BEST_TEACHER_JSON.read_text(encoding="utf-8"))
    checkpoint = resolve_policy_checkpoint(best)
    policy_payload = torch.load(checkpoint, map_location="cpu")
    actor = Actor()
    actor_state = {key: value for key, value in policy_payload["model_state_dict"].items() if key.startswith("actor.")}
    actor.load_state_dict(actor_state)
    actor.eval()
    policy = PolicyBundle(
        checkpoint=checkpoint,
        actor=actor,
        obs_mean=policy_payload["obs_norm_state_dict"]["_mean"],
        obs_std=policy_payload["obs_norm_state_dict"]["_std"],
    )

    vae_payload = torch.load(VAE_CKPT, map_location="cpu")
    vae_cfg = dict(vae_payload["config"])
    vae = ConditionalActionVAE(
        int(vae_cfg["obs_dim"]),
        int(vae_cfg["action_dim"]),
        int(vae_cfg["latent_dim"]),
        int(vae_cfg["hidden_dim"]),
    )
    vae.load_state_dict(vae_payload["model_state_dict"])
    vae.eval()

    denoiser_payload = torch.load(DENOISER_CKPT, map_location="cpu")
    denoiser_cfg = dict(denoiser_payload["config"])
    denoiser = StateLatentDenoiser(
        int(denoiser_cfg["token_dim"]),
        int(denoiser_cfg["hidden_dim"]),
        int(denoiser_cfg["denoising_steps"]),
    )
    denoiser.load_state_dict(denoiser_payload["model_state_dict"])
    denoiser.eval()
    return ModelBundle(policy=policy, vae=vae, vae_cfg=vae_cfg, denoiser=denoiser, denoiser_cfg=denoiser_cfg)


def prepare_clean_walk() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    motion_path = Path(os.environ.get("BM_CLEAN_WALK_MOTION_NPZ", str(DEFAULT_MOTION))).expanduser()
    duration_s = float(os.environ.get("BM_CLEAN_WALK_SECONDS", "15.0"))
    video_fps = int(os.environ.get("BM_CLEAN_WALK_VIDEO_FPS", "30"))
    start_index = int(os.environ.get("BM_CLEAN_WALK_START_INDEX", "0"))
    motion = np.load(motion_path, allow_pickle=True)
    motion_fps = scalar_fps(motion["fps"]) if "fps" in motion else 50.0
    root_pos_all = np.asarray(motion["body_pos_w"][:, 0, :], dtype=np.float64)
    segment = segment_from_start(root_pos_all, start_index, motion_fps, duration_s)
    indices = downsample_motion_indices(segment, motion_fps, video_fps)
    arrays = {
        "joint_pos": np.asarray(motion["joint_pos"][indices], dtype=np.float64),
        "joint_vel": np.asarray(motion["joint_vel"][indices], dtype=np.float64),
        "body_pos_w": np.asarray(motion["body_pos_w"][indices], dtype=np.float64),
        "body_quat_w": np.asarray(motion["body_quat_w"][indices], dtype=np.float64),
        "body_lin_vel_w": np.asarray(motion["body_lin_vel_w"][indices], dtype=np.float64),
        "body_ang_vel_w": np.asarray(motion["body_ang_vel_w"][indices], dtype=np.float64),
    }
    root_pos_targets = arrays["body_pos_w"][:, 0, :].copy()
    root_pos_targets[:, 0:2] = 0.0
    arrays["root_pos_targets"] = root_pos_targets
    arrays["root_quat_targets"] = np.stack(
        [normalize_quat_wxyz(q) for q in arrays["body_quat_w"][:, 0, :]], axis=0
    )
    meta = {
        "motion_path": str(motion_path),
        "motion_sha256": sha256(motion_path),
        "motion_fps": motion_fps,
        "video_fps": video_fps,
        "duration_seconds": len(indices) / float(video_fps),
        "frames_rendered": int(len(indices)),
        "selected_start_index": segment.start,
        "selected_end_index_exclusive": segment.end,
        "selected_root_z_min": segment.root_z_min,
        "selected_root_z_mean": segment.root_z_mean,
        "selected_root_z_max": segment.root_z_max,
        "selected_root_z_range": segment.root_z_range,
        "selected_root_xy_displacement_m": segment.root_xy_displacement_m,
        "temporal_stretching": False,
        "motion_downsampled_from_50hz_to_video_fps": True,
    }
    return arrays, meta


def denoise_latent(models: ModelBundle, obs: np.ndarray, latent: np.ndarray) -> np.ndarray:
    obs_dim = int(models.denoiser_cfg["obs_dim"])
    steps = int(models.denoiser_cfg["denoising_steps"])
    token = np.concatenate([obs.astype(np.float32), latent.astype(np.float32)], axis=-1)
    bars = alpha_bars(steps)
    noisy = torch.sqrt(bars[-1]) * torch.from_numpy(token).unsqueeze(0)
    step_idx = torch.full((1,), steps - 1, dtype=torch.long)
    with torch.inference_mode():
        pred = models.denoiser(noisy, step_idx).squeeze(0)
    return pred[obs_dim:].cpu().numpy().astype(np.float64)


def action_for_variant(
    variant: str,
    obs: np.ndarray,
    teacher_action: np.ndarray,
    models: ModelBundle,
    guidance_scale: float,
) -> tuple[np.ndarray, dict[str, float]]:
    obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
    act_t = torch.from_numpy(teacher_action.astype(np.float32)).unsqueeze(0)
    if variant == "teacher_policy_action_control":
        return teacher_action, {}
    with torch.inference_mode():
        latent = models.vae.posterior_mean(obs_t, act_t).squeeze(0)
    if variant == "vae_reconstructed_action_control":
        used_latent = latent
    elif variant in {"diffusion_denoised_latent_action_control", "guided_latent_action_control"}:
        denoised = denoise_latent(models, obs, latent.cpu().numpy())
        if variant == "guided_latent_action_control":
            used_latent = torch.from_numpy((denoised + guidance_scale * (latent.cpu().numpy() - denoised)).astype(np.float32))
        else:
            used_latent = torch.from_numpy(denoised.astype(np.float32))
    else:
        raise KeyError(f"Unknown variant: {variant}")
    with torch.inference_mode():
        decoded = models.vae.decode(obs_t, used_latent.unsqueeze(0)).squeeze(0)
    return decoded.cpu().numpy().astype(np.float64), {"latent_norm": float(torch.linalg.vector_norm(used_latent))}


def setup_simulation(action_rows: list[dict[str, Any]], width: int, height: int):
    import mujoco

    os.environ.setdefault("BM_MUJOCO_PD_CAMERA_POS", "-0.15 -3.95 1.50")
    os.environ.setdefault("BM_MUJOCO_PD_CAMERA_XYAXES", "1 0 0 0 0.30 0.954")
    os.environ.setdefault("BM_MUJOCO_PD_CAMERA_FOVY", "38")
    model_path = Path(os.environ.get("BM_MUJOCO_G1_XML", str(DEFAULT_MODEL))).expanduser()
    patched_xml = model_path.parent / "g1_clean_walk_control_suite_pd.xml"
    patch_joints_and_actuators(model_path, patched_xml, action_rows)
    model = mujoco.MjModel.from_xml_path(str(patched_xml))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    if actuator_joint_order(model) != [row["joint_name"] for row in action_rows]:
        raise RuntimeError("Actuator order does not match action scale audit")
    return model, data, renderer, patched_xml


def initialize_state(model, data, ref: dict[str, np.ndarray], settle_steps: int) -> tuple[dict[str, int], list[int], float, np.ndarray]:
    import mujoco

    ids = body_id_map(model)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = ref["root_pos_targets"][0]
    data.qpos[3:7] = ref["root_quat_targets"][0]
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
    pelvis_body = ids["pelvis"]
    for _ in range(settle_steps):
        data.xfrc_applied[:] = 0.0
        apply_root_assist(model, data, pelvis_body, ref["root_pos_targets"][0], ref["root_quat_targets"][0])
        mujoco.mj_step(model, data)
    return ids, ref_body_indices, world_to_init_yaw, world_to_init_translation


def render_variant(
    *,
    variant: str,
    ref: dict[str, np.ndarray],
    clean_meta: dict[str, Any],
    models: ModelBundle,
    action_rows: list[dict[str, Any]],
    default_joint_pos: np.ndarray,
    action_scale: np.ndarray,
    default_source: str,
    default_note: str,
    model_target_weight: float,
    guidance_scale: float,
) -> dict[str, Any]:
    import mujoco

    video_fps = int(clean_meta["video_fps"])
    width = int(os.environ.get("BM_CLEAN_WALK_WIDTH", "960"))
    height = int(os.environ.get("BM_CLEAN_WALK_HEIGHT", "540"))
    substeps = int(os.environ.get("BM_CLEAN_WALK_SUBSTEPS", "4"))
    settle_steps = int(os.environ.get("BM_CLEAN_WALK_SETTLE_STEPS", "50"))
    action_clip = float(os.environ.get("BM_CLEAN_SUITE_ACTION_CLIP", "3.0"))
    out_dir = OUT_ROOT / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / f"{variant}.mp4"
    keyframe_path = out_dir / f"{variant}_keyframe.png"
    keyframes_path = out_dir / f"{variant}_keyframes.png"
    metrics_path = out_dir / f"{variant}_metrics.csv"
    summary_path = out_dir / f"{variant}_summary.json"

    model, data, renderer, patched_xml = setup_simulation(action_rows, width, height)
    ids, ref_body_indices, world_to_init_yaw, world_to_init_translation = initialize_state(
        model, data, ref, settle_steps
    )
    pelvis_body = ids["pelvis"]
    rows: list[dict[str, Any]] = []
    strip_frames: list[np.ndarray] = []
    frames = int(ref["joint_pos"].shape[0])
    strip_indices = set(int(v) for v in np.linspace(0, frames - 1, 5))
    last_action = np.zeros(29, dtype=np.float64)
    with imageio.get_writer(mp4_path, fps=video_fps, codec="libx264", quality=8, macro_block_size=1) as writer:
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
            teacher_action = models.policy.act(obs)
            if variant == "reference_action_control":
                model_action = np.zeros(29, dtype=np.float64)
                model_target = ref["joint_pos"][frame_idx]
                target = model_target
                latent_meta: dict[str, float] = {}
            else:
                model_action, latent_meta = action_for_variant(variant, obs, teacher_action, models, guidance_scale)
                model_target = action_to_joint_targets(
                    model_action[None, :],
                    default_joint_pos,
                    action_scale,
                    np.asarray(model.actuator_ctrlrange, dtype=np.float64),
                    frames=1,
                    clip_actions=action_clip,
                )[0][0]
                target = (1.0 - model_target_weight) * ref["joint_pos"][frame_idx] + model_target_weight * model_target
            target = np.clip(target, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            data.ctrl[:] = target
            for _ in range(substeps):
                data.xfrc_applied[:] = 0.0
                apply_root_assist(model, data, pelvis_body, ref["root_pos_targets"][frame_idx], ref["root_quat_targets"][frame_idx])
                mujoco.mj_step(model, data)
            frame = render_frame(model, data, renderer, camera=PD_CAMERA)
            if frame_idx == 0:
                imageio.imwrite(keyframe_path, frame)
            if frame_idx in strip_indices:
                strip_frames.append(frame)
            writer.append_data(frame)
            q = data.qpos[7 : 7 + 29].copy()
            qd = data.qvel[6 : 6 + 29].copy()
            roll, pitch, yaw = quat_to_roll_pitch_yaw(data.qpos[3:7])
            rows.append(
                {
                    "frame": frame_idx,
                    "video_time_s": frame_idx / float(video_fps),
                    "sim_time_s": float(data.time),
                    "root_x": float(data.qpos[0]),
                    "root_y": float(data.qpos[1]),
                    "root_z": float(data.qpos[2]),
                    "root_roll": roll,
                    "root_pitch": pitch,
                    "root_yaw": yaw,
                    "teacher_action_abs_mean": float(np.mean(np.abs(teacher_action))),
                    "model_action_abs_mean": float(np.mean(np.abs(model_action))),
                    "model_action_abs_max": float(np.max(np.abs(model_action))),
                    "reference_anchor_weight": float(1.0 - model_target_weight),
                    "model_target_weight": float(model_target_weight if variant != "reference_action_control" else 0.0),
                    "model_target_gap_to_reference_abs_mean": float(np.mean(np.abs(model_target - ref["joint_pos"][frame_idx]))),
                    "final_target_gap_to_reference_abs_mean": float(np.mean(np.abs(target - ref["joint_pos"][frame_idx]))),
                    "joint_error_abs_mean": float(np.mean(np.abs(q - target))),
                    "joint_error_to_reference_abs_mean": float(np.mean(np.abs(q - ref["joint_pos"][frame_idx]))),
                    "joint_velocity_abs_mean": float(np.mean(np.abs(qd))),
                    "root_position_error_m": float(np.linalg.norm(data.xpos[pelvis_body] - ref["root_pos_targets"][frame_idx])),
                    "root_orientation_error_rad": float(
                        np.linalg.norm(quat_error_rotvec(ref["root_quat_targets"][frame_idx], data.xquat[pelvis_body]))
                    ),
                    "contact_count": int(data.ncon),
                    "fall_proxy": bool(data.qpos[2] < 0.45 or abs(roll) > 1.2 or abs(pitch) > 1.2),
                    **obs_debug,
                    **latent_meta,
                }
            )
            # The policy observation contract contains the previous action.
            # For VAE/diffusion/guided rollouts this must be the action that
            # was actually decoded/applied by that controller, not always the
            # teacher action, otherwise the closed-loop observation is
            # internally inconsistent after the first frame.
            last_action = model_action
    renderer.close()
    make_keyframe_strip(strip_frames, keyframes_path)
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    fall_count = sum(1 for row in rows if row["fall_proxy"])
    summary = {
        "status": "ok" if fall_count == 0 else "failed_unstable",
        "timestamp_utc": utc_now(),
        "experiment_type": "clean_walk_mujoco_reference_anchored_control_suite",
        "spec_name": variant,
        "claim_level": (
            "Local MuJoCo clean-walk presentation/diagnostic video. Learned variants use a recorded "
            "reference-anchor blend and are not pure policy/diffusion control unless model_target_weight=1."
        ),
        "clean_walk_source": clean_meta,
        "model_target_weight": float(model_target_weight if variant != "reference_action_control" else 0.0),
        "reference_anchor_weight": float(1.0 - model_target_weight if variant != "reference_action_control" else 1.0),
        "guidance_scale": float(guidance_scale),
        "default_joint_pos_source": default_source,
        "default_joint_pos_note": default_note,
        "policy_checkpoint": str(models.policy.checkpoint),
        "vae_checkpoint": str(VAE_CKPT),
        "denoiser_checkpoint": str(DENOISER_CKPT),
        "simulation": {
            "uses_mj_step": True,
            "writes_qpos_each_frame": False,
            "actuator_type": "position",
            "actuator_count": int(model.nu),
            "root_assist_enabled": os.environ.get("BM_MUJOCO_ROOT_ASSIST", "1") == "1",
            "patched_xml": str(patched_xml),
            "substeps_per_frame": substeps,
            "settle_steps": settle_steps,
        },
        "camera": {
            "name": PD_CAMERA,
            "position": os.environ.get("BM_MUJOCO_PD_CAMERA_POS", "-0.15 -3.95 1.50"),
            "fovy": os.environ.get("BM_MUJOCO_PD_CAMERA_FOVY", "38"),
        },
        "outputs": {
            "mp4": str(mp4_path),
            "keyframe_png": str(keyframe_path),
            "keyframes_png": str(keyframes_path),
            "metrics_csv": str(metrics_path),
            "summary_json": str(summary_path),
        },
        "metrics": {
            "fall_proxy_count": int(fall_count),
            "root_height_min": float(min(row["root_z"] for row in rows)),
            "root_height_mean": float(np.mean([row["root_z"] for row in rows])),
            "root_height_max": float(max(row["root_z"] for row in rows)),
            "root_xy_abs_max": float(max(max(abs(row["root_x"]), abs(row["root_y"])) for row in rows)),
            "joint_error_abs_mean": float(np.mean([row["joint_error_abs_mean"] for row in rows])),
            "joint_error_to_reference_abs_mean": float(np.mean([row["joint_error_to_reference_abs_mean"] for row in rows])),
            "model_target_gap_to_reference_abs_mean": float(
                np.mean([row["model_target_gap_to_reference_abs_mean"] for row in rows])
            ),
            "final_target_gap_to_reference_abs_mean": float(
                np.mean([row["final_target_gap_to_reference_abs_mean"] for row in rows])
            ),
            "root_position_error_mean_m": float(np.mean([row["root_position_error_m"] for row in rows])),
            "root_position_error_max_m": float(np.max([row["root_position_error_m"] for row in rows])),
            "contact_count_mean": float(np.mean([row["contact_count"] for row in rows])),
        },
        "checks": {
            "mp4_exists": mp4_path.is_file() and mp4_path.stat().st_size > 0,
            "keyframes_exists": keyframes_path.is_file() and keyframes_path.stat().st_size > 0,
            "metrics_csv_exists": metrics_path.is_file() and metrics_path.stat().st_size > 0,
            "uses_mj_step": True,
            "does_not_write_qpos_each_frame": True,
            "fall_proxy_zero": fall_count == 0,
            "video_duration_at_least_10s": clean_meta["duration_seconds"] >= 10.0,
            "does_not_claim_paper_level": True,
            "does_not_claim_real_robot": True,
        },
        "limitations": [
            "The current Stage-1 teacher is weak; pure model-target control is not claimed stable.",
            "Learned variants use a reference-anchor blend by default for readable presentation.",
            "This is not classifier-guided paper Fig.5/Fig.6 closed-loop task control.",
        ],
    }
    write_json(summary_path, summary)
    print(json.dumps({"status": summary["status"], "variant": variant, "mp4": str(mp4_path)}))
    return summary


def render_side_by_side(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    out_dir = OUT_ROOT / "guided_vs_unguided_action_control"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / "guided_vs_unguided_action_control.mp4"
    keyframe = out_dir / "guided_vs_unguided_action_control_keyframe.png"
    summary_path = out_dir / "guided_vs_unguided_action_control_summary.json"
    fps = int(left["clean_walk_source"]["video_fps"])
    reader_l = imageio.get_reader(left["outputs"]["mp4"])
    reader_r = imageio.get_reader(right["outputs"]["mp4"])
    n = min(reader_l.count_frames(), reader_r.count_frames())
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
        "experiment_type": "clean_walk_guided_vs_unguided_reference_anchored_side_by_side",
        "claim_level": "Side-by-side clean-walk local MuJoCo presentation video: denoised latent left, guided latent right; not paper-level guidance.",
        "left": left["outputs"],
        "right": right["outputs"],
        "frames_rendered": int(n),
        "video_fps": fps,
        "duration_seconds": n / float(fps),
        "outputs": {"mp4": str(out_mp4), "keyframe_png": str(keyframe), "summary_json": str(summary_path)},
        "checks": {
            "mp4_exists": out_mp4.is_file() and out_mp4.stat().st_size > 0,
            "does_not_claim_paper_level": True,
            "does_not_claim_real_robot": True,
        },
    }
    write_json(summary_path, payload)
    return payload


def main() -> None:
    try:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        ref, clean_meta = prepare_clean_walk()
        models = load_models()
        action_rows = load_action_rows()
        action_scale = np.asarray([float(row["action_scale"]) for row in action_rows], dtype=np.float64)
        default_joint_pos, default_source, default_note = parse_default_joint_position(action_rows)
        model_target_weight = float(os.environ.get("BM_CLEAN_SUITE_MODEL_TARGET_WEIGHT", "0.20"))
        guidance_scale = float(os.environ.get("BM_CLEAN_SUITE_GUIDANCE_SCALE", "0.35"))
        variants = [
            "reference_action_control",
            "teacher_policy_action_control",
            "vae_reconstructed_action_control",
            "diffusion_denoised_latent_action_control",
            "guided_latent_action_control",
        ]
        rendered: dict[str, Any] = {}
        for variant in variants:
            rendered[variant] = render_variant(
                variant=variant,
                ref=ref,
                clean_meta=clean_meta,
                models=models,
                action_rows=action_rows,
                default_joint_pos=default_joint_pos,
                action_scale=action_scale,
                default_source=default_source,
                default_note=default_note,
                model_target_weight=model_target_weight,
                guidance_scale=guidance_scale,
            )
        rendered["guided_vs_unguided_action_control"] = render_side_by_side(
            rendered["diffusion_denoised_latent_action_control"], rendered["guided_latent_action_control"]
        )
        failure_audit = write_failure_audit()
        if model_target_weight >= 0.999:
            suite_claim = (
                "Clean 15 s local MuJoCo walk suite with pure local model targets. "
                "Root assist is still enabled, so this is not paper-level BeyondMimic evidence."
            )
        else:
            suite_claim = (
                "Clean 15 s local MuJoCo walk suite with reference-anchored learned variants. "
                "Not paper-level BeyondMimic."
            )
        suite = {
            "status": "ok" if all(v.get("status") == "ok" for k, v in rendered.items() if k != "guided_vs_unguided_action_control") else "failed_unstable_variant",
            "timestamp_utc": utc_now(),
            "experiment_type": "clean_walk_mujoco_control_suite",
            "claim_level": suite_claim,
            "clean_walk_source": clean_meta,
            "model_target_weight": model_target_weight,
            "reference_anchor_weight": 1.0 - model_target_weight,
            "guidance_scale": guidance_scale,
            "output_root": str(OUT_ROOT),
            "default_output_root": str(DEFAULT_OUT_ROOT),
            "previous_stage1_video_failure_audit": str(OUT_ROOT / "../clean_walk_mujoco_pd_control_demo/why_previous_stage1_six_videos_failed.json"),
            "variants": {name: payload["outputs"] for name, payload in rendered.items()},
            "variant_metrics": {name: payload.get("metrics", {}) for name, payload in rendered.items()},
            "checks": {
                "all_mp4_exist": all(Path(payload["outputs"]["mp4"]).is_file() for payload in rendered.values()),
                "all_primary_variants_fall_proxy_zero": all(
                    payload.get("metrics", {}).get("fall_proxy_count", 0) == 0
                    for name, payload in rendered.items()
                    if name != "guided_vs_unguided_action_control"
                ),
                "video_duration_at_least_10s": clean_meta["duration_seconds"] >= 10.0,
                "uses_reference_anchor_blend_for_learned_variants": model_target_weight < 1.0,
                "does_not_claim_paper_level_pure_policy_success": True,
                "does_not_claim_paper_level": True,
            },
            "limitations": [
                "The learned variants are stabilized presentation diagnostics, not pure policy/diffusion controllers.",
                "The reference-action control video is the only unambiguous normal walk control baseline in this suite.",
                "Pure Stage-1 teacher/diffusion control remains blocked by teacher quality and MuJoCo adapter fidelity.",
            ],
            "failure_context": failure_audit,
        }
        write_json(OUT_ROOT / "clean_walk_mujoco_control_suite_summary.json", suite)
        lines = [
            "# Clean Walk MuJoCo Control Suite",
            "",
            "同一段 15 秒 LAFAN1 walk，在 MuJoCo 中用 29 个 position actuator + `mj_step` 生成展示视频。",
            "",
            "## 重要边界",
            "",
            f"- Learned variants 使用 reference-anchor blend：`model_target_weight={model_target_weight}`，`reference_anchor_weight={1.0 - model_target_weight}`。",
            "- 因此 teacher/VAE/denoised/guided 视频是展示/诊断，不是纯 policy 或 paper-level guidance。",
            "- 不是真实机器人，不是 IsaacLab rendered MP4，不是 BeyondMimic Fig.5/Fig.6 完整复现。",
            "",
            "## Videos",
            "",
        ]
        for name, payload in rendered.items():
            lines.append(f"- `{name}`: `{payload['outputs']['mp4']}`")
        lines.extend(
            [
                "",
                "## Summary",
                "",
                f"- Summary JSON: `{OUT_ROOT / 'clean_walk_mujoco_control_suite_summary.json'}`",
                f"- Source motion: `{clean_meta['motion_path']}`",
                f"- Frames/duration: `{clean_meta['frames_rendered']}` / `{clean_meta['duration_seconds']}` s",
                "",
            ]
        )
        (OUT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")
        print(json.dumps({"status": suite["status"], "summary": str(OUT_ROOT / "clean_walk_mujoco_control_suite_summary.json")}))
    except Exception as exc:  # noqa: BLE001
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        write_json(
            OUT_ROOT / "clean_walk_mujoco_control_suite_failed_summary.json",
            {
                "status": "failed",
                "timestamp_utc": utc_now(),
                "experiment_type": "clean_walk_mujoco_control_suite",
                "error": traceback_payload(exc),
            },
        )
        raise


if __name__ == "__main__":
    main()
