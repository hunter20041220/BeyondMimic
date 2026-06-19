#!/usr/bin/env python3
"""Run a local teacher-consistency action-guidance rollout in IsaacLab.

This is a closed-loop bridge experiment: the local official-csv-loop PPO
teacher is reconstructed through the local conditional action VAE, then a
small action-space guidance step moves the VAE action toward the teacher.
It is intentionally reported as local virtual evidence, not paper Fig. 5/6
latent diffusion guidance.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/visualization/official_csv_loop_action_guidance_rollout"
SUMMARY_JSON = ROOT / "res/level_c/official_csv_loop_action_guidance_rollout_eval/level_c_official_csv_loop_action_guidance_rollout_eval.json"
SUMMARY_TSV = ROOT / "res/level_c/official_csv_loop_action_guidance_rollout_eval/level_c_official_csv_loop_action_guidance_rollout_eval.tsv"
LOG_DIR = ROOT / "logs/tracking_g1_official_csv_loop_action_guidance_rollout_eval"
FAILED_DIR = ROOT / "res/failed_runs/tracking_g1_official_csv_loop_action_guidance_rollout_eval"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_csv_loop_action_guidance_rollout_eval"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
ANALYSIS_PY = ROOT / "envs/bm_analysis/bin/python"
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
ENRICHED_USD = (
    ROOT
    / "res/tracking/g1_resource_adjusted_enriched_usd/"
    "g1_resource_adjusted_29dof_enriched_scaffold.usda"
)
OFFICIAL_LOOP_MOTION_NPZ = (
    ROOT
    / "res/tracking/official_csv_to_npz_loop_with_enriched_usd/"
    "walk1_subject1_frames_1_180_official_loop_enriched_usd_motion.npz"
)
TRAINING_RUN_JSON = (
    ROOT
    / "res/tracking/g1_official_csv_loop_ppo_training_run/"
    "tracking_g1_official_csv_loop_ppo_training_run.json"
)
CHECKPOINT_EVAL_JSON = (
    ROOT
    / "res/tracking/g1_official_csv_loop_ppo_checkpoint_eval/"
    "tracking_g1_official_csv_loop_ppo_checkpoint_eval.json"
)
VAE_TRAINING_JSON = (
    ROOT
    / "res/level_c/official_csv_loop_teacher_rollout_vae_training/"
    "level_c_official_csv_loop_teacher_rollout_vae_training.json"
)
BODY_CONTRACT = ROOT / "res/tracking/motion_npz_fixture/tracking_motion_npz_fixture.json"
SOURCE_CONTRACT = ROOT / "res/tracking/official_source_contract_audit/tracking_official_source_contract_audit.json"
CANDIDATE_GPUS = [4, 7]
MIN_FREE_MB = 20_000
MAX_BUSY_UTIL = 50
SEED = 20260639
ROLLOUT_STEPS = 299
GUIDANCE_ALPHA = 0.35


WORKER_CODE = r"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
args.device = "cuda:0"

print(f"BM_SENTINEL:action_guidance:before_app:device={args.device}", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("BM_SENTINEL:action_guidance:after_app", flush=True)

try:
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import torch
    from torch import nn
    import whole_body_tracking.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

    class ConditionalActionVAE(nn.Module):
        def __init__(self, obs_dim, action_dim, latent_dim, hidden_dim):
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

        def encode(self, obs, action):
            mu_logvar = self.encoder(torch.cat([obs, action], dim=-1))
            return torch.chunk(mu_logvar, 2, dim=-1)

        def decode(self, obs, latent):
            return self.decoder(torch.cat([obs, latent], dim=-1))

    enriched_usd = Path(os.environ["BM_ENRICHED_USD"])
    motion_file = Path(os.environ["BM_MOTION_FILE"])
    checkpoint = Path(os.environ["BM_CHECKPOINT"])
    vae_checkpoint = Path(os.environ["BM_VAE_CHECKPOINT"])
    out_npz = Path(os.environ["BM_OUT_NPZ"])
    metrics_path = Path(os.environ["BM_METRICS_JSON"])
    rollout_steps = int(os.environ["BM_ROLLOUT_STEPS"])
    guidance_alpha = float(os.environ["BM_GUIDANCE_ALPHA"])
    seed = int(os.environ["BM_SEED"])

    def make_env():
        env_cfg = G1FlatEnvCfg()
        env_cfg.scene.num_envs = 1
        env_cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
            usd_path=str(enriched_usd),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        )
        env_cfg.commands.motion.motion_file = str(motion_file)
        env_cfg.commands.motion.debug_vis = False
        env_cfg.scene.contact_forces.debug_vis = False
        env_cfg.sim.device = args.device
        env_cfg.seed = seed
        env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
        return RslRlVecEnvWrapper(env)

    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    vec_env = make_env()
    print("BM_SENTINEL:action_guidance:env_created", flush=True)
    agent_cfg = G1FlatPPORunnerCfg()
    agent_cfg.device = args.device
    agent_cfg.seed = seed
    agent_cfg.empirical_normalization = True
    agent_cfg.logger = "tensorboard"
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
    vae_payload = torch.load(vae_checkpoint, map_location="cpu")
    vae_cfg = vae_payload["config"]
    vae = ConditionalActionVAE(
        vae_cfg["obs_dim"],
        vae_cfg["action_dim"],
        vae_cfg["latent_dim"],
        vae_cfg["hidden_dim"],
    ).to(vec_env.unwrapped.device)
    vae.load_state_dict(vae_payload["model_state_dict"])
    vae.eval()

    variants = ["teacher", "vae_base", "action_guided"]
    traces = {}
    variant_metrics = {}
    action_deltas = {}
    metric_names = ["error_anchor_pos", "error_body_pos", "error_joint_pos", "sampling_top1_prob"]

    with torch.inference_mode():
        for variant in variants:
            obs, _ = vec_env.reset()
            command = vec_env.unwrapped.command_manager.get_term("motion")
            robot_body_pos = []
            reference_body_pos = []
            rewards = []
            dones = []
            action_abs_mean = []
            action_abs_max = []
            teacher_vae_mse = []
            guided_base_mse = []
            guided_teacher_mse = []
            metric_series = {name: [] for name in metric_names}
            for step in range(rollout_steps):
                teacher_action = policy(obs)
                mu, _logvar = vae.encode(obs, teacher_action)
                base_action = vae.decode(obs, mu)
                guided_action = base_action + guidance_alpha * (teacher_action - base_action)
                if variant == "teacher":
                    action = teacher_action
                elif variant == "vae_base":
                    action = base_action
                else:
                    action = guided_action
                obs, rew, done, _extras = vec_env.step(action)
                robot_body_pos.append(command.robot_body_pos_w[0].detach().cpu().numpy().astype(np.float32))
                reference_body_pos.append(command.body_pos_relative_w[0].detach().cpu().numpy().astype(np.float32))
                rewards.append(float(rew.detach().cpu().mean()))
                dones.append(int(done.detach().cpu().sum()))
                action_abs_mean.append(float(action.abs().mean().detach().cpu()))
                action_abs_max.append(float(action.abs().max().detach().cpu()))
                teacher_vae_mse.append(float(torch.mean((base_action - teacher_action).square()).detach().cpu()))
                guided_base_mse.append(float(torch.mean((guided_action - base_action).square()).detach().cpu()))
                guided_teacher_mse.append(float(torch.mean((guided_action - teacher_action).square()).detach().cpu()))
                for name in metric_names:
                    value = command.metrics.get(name)
                    metric_series[name].append(
                        float(value[0].detach().cpu()) if value is not None and value.numel() > 0 else float("nan")
                    )
                if (step + 1) % 50 == 0 or (step + 1) == rollout_steps:
                    print(f"BM_SENTINEL:action_guidance:variant={variant}:step={step + 1}/{rollout_steps}", flush=True)

            robot_arr = np.stack(robot_body_pos, axis=0)
            ref_arr = np.stack(reference_body_pos, axis=0)
            body_error = np.linalg.norm(robot_arr - ref_arr, axis=-1).mean(axis=1)
            traces[f"{variant}_robot_body_pos_w"] = robot_arr
            traces[f"{variant}_reference_body_pos_w"] = ref_arr
            traces[f"{variant}_rewards"] = np.asarray(rewards, dtype=np.float32)
            traces[f"{variant}_dones"] = np.asarray(dones, dtype=np.int32)
            traces[f"{variant}_action_abs_mean"] = np.asarray(action_abs_mean, dtype=np.float32)
            traces[f"{variant}_target_body_error_mean"] = body_error.astype(np.float32)
            traces[f"{variant}_teacher_vae_mse"] = np.asarray(teacher_vae_mse, dtype=np.float32)
            traces[f"{variant}_guided_base_mse"] = np.asarray(guided_base_mse, dtype=np.float32)
            traces[f"{variant}_guided_teacher_mse"] = np.asarray(guided_teacher_mse, dtype=np.float32)
            variant_metrics[variant] = {
                "reward_mean": float(np.mean(rewards)),
                "reward_min": float(np.min(rewards)),
                "reward_max": float(np.max(rewards)),
                "done_count_total": int(np.sum(dones)),
                "action_abs_mean": float(np.mean(action_abs_mean)),
                "action_abs_max": float(np.max(action_abs_max)),
                "target_body_error_mean": float(np.mean(body_error)),
                "target_body_error_max": float(np.max(body_error)),
                "teacher_vae_action_mse_mean": float(np.mean(teacher_vae_mse)),
                "guided_base_action_mse_mean": float(np.mean(guided_base_mse)),
                "guided_teacher_action_mse_mean": float(np.mean(guided_teacher_mse)),
                "motion_metrics": {
                    name: {
                        "mean": float(np.nanmean(values)),
                        "min": float(np.nanmin(values)),
                        "max": float(np.nanmax(values)),
                    }
                    for name, values in metric_series.items()
                },
            }
            if variant == "action_guided":
                action_deltas["teacher_mse_reduction_vs_vae_base_expected"] = float(
                    1.0 - (1.0 - guidance_alpha) ** 2
                )
                action_deltas["guidance_alpha"] = guidance_alpha

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **traces)

    metrics = {
        "status": "ok",
        "checkpoint": str(checkpoint),
        "vae_checkpoint": str(vae_checkpoint),
        "motion_file": str(motion_file),
        "device": args.device,
        "num_envs": 1,
        "rollout_steps": rollout_steps,
        "loaded_iteration": int(runner.current_learning_iteration),
        "guidance": {
            "type": "teacher_consistency_action_space",
            "alpha": guidance_alpha,
            "formula": "a_guided = a_vae + alpha * (a_teacher - a_vae)",
            "not_receding_horizon_latent_diffusion": True,
        },
        "variant_metrics": variant_metrics,
        "action_deltas": action_deltas,
        "uses_resource_adjusted_usd": True,
        "official_csv_loop_motion": True,
        "paper_level_guidance_rollout": False,
        "fig5_fig6_reproduction": False,
        "real_robot": False,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    vec_env.close()
    print(f"BM_SENTINEL:action_guidance:npz={out_npz}", flush=True)
    print(f"BM_SENTINEL:action_guidance:metrics={metrics_path}", flush=True)
    os._exit(0)
except BaseException as exc:
    print(f"BM_SENTINEL:action_guidance:exception={exc!r}", flush=True)
    raise
finally:
    try:
        simulation_app.close(wait_for_replicator=False)
    except BaseException:
        pass
"""


RENDER_CODE = r"""
import csv
import hashlib
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

npz_path = Path(os.environ["BM_CAPTURE_NPZ"])
metrics_path = Path(os.environ["BM_METRICS_JSON"])
asset_json = Path(os.environ["BM_ASSET_JSON"])
body_contract = json.loads(Path(os.environ["BM_BODY_CONTRACT"]).read_text())
source_contract = json.loads(Path(os.environ["BM_SOURCE_CONTRACT"]).read_text())
metrics = json.loads(metrics_path.read_text())
out_dir = asset_json.parent
out_dir.mkdir(parents=True, exist_ok=True)

target_names = source_contract["flat_env"]["body_names"]
edges = [
    ("pelvis", "left_hip_roll_link"),
    ("left_hip_roll_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("pelvis", "right_hip_roll_link"),
    ("right_hip_roll_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_roll_link"),
    ("pelvis", "torso_link"),
    ("torso_link", "left_shoulder_roll_link"),
    ("left_shoulder_roll_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_yaw_link"),
    ("torso_link", "right_shoulder_roll_link"),
    ("right_shoulder_roll_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_yaw_link"),
]
data = np.load(npz_path)
variants = ["teacher", "vae_base", "action_guided"]
robot = {variant: data[f"{variant}_robot_body_pos_w"] for variant in variants}
reference = data["teacher_reference_body_pos_w"]
if reference.shape[1] == len(target_names):
    names = target_names
    target_indices = list(range(len(target_names)))
elif reference.shape[1] == len(body_contract["body_names_urdf_order"]):
    names = body_contract["body_names_urdf_order"]
    full_idx = {name: idx for idx, name in enumerate(names)}
    target_indices = [full_idx[name] for name in target_names]
else:
    raise ValueError(f"Unexpected body count {reference.shape[1]}")
name_to_idx = {name: idx for idx, name in enumerate(names)}

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def target(arr):
    return arr[:, target_indices, :]

target_ref = target(reference)
target_robot = {variant: target(arr) for variant, arr in robot.items()}

def set_axes_equal(ax, frame):
    xyz = np.concatenate([target_ref[frame][None, :, :]] + [target_robot[v][frame][None, :, :] for v in variants], axis=1)
    mins = xyz.min(axis=(0, 1))
    maxs = xyz.max(axis=(0, 1))
    centers = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 0.4)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(max(0.0, centers[2] - radius), centers[2] + radius)

def draw_skeleton(ax, xyz, color, label, linewidth=2.0, alpha=0.9):
    for a, b in edges:
        pts = xyz[[name_to_idx[a], name_to_idx[b]]]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=linewidth, alpha=alpha)
    ax.scatter(xyz[target_indices, 0], xyz[target_indices, 1], xyz[target_indices, 2], s=18, color=color, alpha=alpha, label=label)

colors = {"teacher": "#059669", "vae_base": "#2563eb", "action_guided": "#dc2626"}
labels = {"teacher": "teacher", "vae_base": "VAE base", "action_guided": "guided"}

def draw_frame(ax, frame):
    draw_skeleton(ax, reference[frame], "#64748b", "reference", linewidth=1.2, alpha=0.55)
    for variant in variants:
        draw_skeleton(ax, robot[variant][frame], colors[variant], labels[variant], linewidth=1.7 if variant != "action_guided" else 2.4, alpha=0.82)
    ax.set_title(f"Local action-guidance rollout, frame {frame:03d}", fontsize=11)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

video_path = out_dir / "official_csv_loop_action_guidance_rollout_vs_reference.mp4"
keyframes_path = out_dir / "official_csv_loop_action_guidance_rollout_keyframes.png"
metrics_png = out_dir / "official_csv_loop_action_guidance_rollout_metrics.png"
metrics_csv = out_dir / "official_csv_loop_action_guidance_rollout_metrics.csv"
readme = out_dir / "README.md"

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(7.2, 6.0))
ax = fig.add_subplot(111, projection="3d")
writer = FFMpegWriter(fps=30, metadata={"title": "BeyondMimic local action guidance rollout"})
with writer.saving(fig, str(video_path), dpi=145):
    for frame in range(reference.shape[0]):
        ax.cla()
        draw_frame(ax, frame)
        set_axes_equal(ax, frame)
        ax.view_init(elev=18, azim=-68)
        writer.grab_frame()
plt.close(fig)

frames = [0, reference.shape[0] // 3, 2 * reference.shape[0] // 3, reference.shape[0] - 1]
fig = plt.figure(figsize=(13, 7))
for idx, frame in enumerate(frames, start=1):
    ax = fig.add_subplot(2, 2, idx, projection="3d")
    draw_frame(ax, frame)
    set_axes_equal(ax, frame)
    ax.view_init(elev=18, azim=-68)
fig.tight_layout()
fig.savefig(keyframes_path, dpi=180)
plt.close(fig)

with metrics_csv.open("w", encoding="utf-8", newline="") as f:
    writer_csv = csv.DictWriter(
        f,
        fieldnames=[
            "variant",
            "step",
            "reward",
            "done",
            "action_abs_mean",
            "target_body_error_mean",
            "teacher_vae_mse",
            "guided_base_mse",
            "guided_teacher_mse",
        ],
    )
    writer_csv.writeheader()
    for variant in variants:
        for step in range(reference.shape[0]):
            writer_csv.writerow({
                "variant": variant,
                "step": step,
                "reward": float(data[f"{variant}_rewards"][step]),
                "done": int(data[f"{variant}_dones"][step]),
                "action_abs_mean": float(data[f"{variant}_action_abs_mean"][step]),
                "target_body_error_mean": float(data[f"{variant}_target_body_error_mean"][step]),
                "teacher_vae_mse": float(data[f"{variant}_teacher_vae_mse"][step]),
                "guided_base_mse": float(data[f"{variant}_guided_base_mse"][step]),
                "guided_teacher_mse": float(data[f"{variant}_guided_teacher_mse"][step]),
            })

fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.0), sharex=True)
for variant in variants:
    x = np.arange(reference.shape[0])
    axes[0].plot(x, data[f"{variant}_rewards"], label=labels[variant], color=colors[variant])
    axes[1].plot(x, data[f"{variant}_target_body_error_mean"], label=labels[variant], color=colors[variant])
    axes[2].plot(x, data[f"{variant}_action_abs_mean"], label=labels[variant], color=colors[variant])
axes[0].set_ylabel("reward")
axes[1].set_ylabel("target-body error")
axes[2].set_ylabel("|action| mean")
axes[2].set_xlabel("step")
axes[0].legend(loc="best")
fig.suptitle("Local Action Guidance Closed-Loop Rollout")
fig.tight_layout()
fig.savefig(metrics_png, dpi=180)
plt.close(fig)

readme.write_text("\n".join([
    "# Official-CSV-Loop Local Action Guidance Rollout",
    "",
    "This directory contains a local virtual closed-loop rollout comparing teacher, VAE base, and action-guided variants.",
    "",
    "The guided action is `a_guided = a_vae + alpha * (a_teacher - a_vae)` with alpha recorded in the asset JSON.",
    "",
    "## Claim Level",
    "",
    "local_virtual_teacher_consistency_action_guidance_rollout. This is not official BeyondMimic latent diffusion guidance, not Fig. 5/Fig. 6 paper-level evidence, and not real-robot validation.",
    "",
]), encoding="utf-8")

assets = {
    "mp4": str(video_path),
    "keyframes_png": str(keyframes_path),
    "metrics_png": str(metrics_png),
    "metrics_csv": str(metrics_csv),
    "readme": str(readme),
}
summary = {
    "status": "ok",
    "experiment_type": "tracking_g1_official_csv_loop_action_guidance_rollout_eval_assets",
    "claim_level": "local_virtual_teacher_consistency_action_guidance_rollout",
    "source_capture_npz": str(npz_path),
    "source_metrics": str(metrics_path),
    "frame_count": int(reference.shape[0]),
    "body_count": int(reference.shape[1]),
    "target_body_count": len(target_indices),
    "guidance": metrics["guidance"],
    "variant_metrics": metrics["variant_metrics"],
    "assets": assets,
    "asset_sizes": {key: Path(value).stat().st_size for key, value in assets.items()},
    "asset_sha256": {key: sha256_file(Path(value)) for key, value in assets.items()},
    "checks": {
        "frame_count_299": int(reference.shape[0]) == 299,
        "body_count_supported_14_or_40": int(reference.shape[1]) in {14, 40},
        "target_body_count_14": len(target_indices) == 14,
        "video_exists_nonempty": video_path.is_file() and video_path.stat().st_size > 0,
        "keyframes_exist_nonempty": keyframes_path.is_file() and keyframes_path.stat().st_size > 0,
        "metrics_plot_exists_nonempty": metrics_png.is_file() and metrics_png.stat().st_size > 0,
        "does_not_claim_paper_level": True,
        "does_not_claim_fig5_fig6": True,
        "does_not_claim_receding_horizon_latent_diffusion": True,
        "does_not_claim_real_robot": True,
    },
    "interpretation": {
        "goal_complete": False,
        "why_not_complete": "The rollout proves a local virtual closed-loop action-guidance bridge, but it does not use the official BeyondMimic diffusion checkpoint or paper task guidance loop.",
    },
}
asset_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({"status": "ok", "json": str(asset_json), "mp4": str(video_path)}, sort_keys=True))
"""


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: list[str], env: dict[str, str] | None = None, timeout: int | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query_gpus() -> list[dict[str, Any]]:
    rc, out = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
            "-i",
            ",".join(str(gpu) for gpu in CANDIDATE_GPUS),
        ],
        timeout=30,
    )
    if rc != 0:
        return [{"error": out.strip()}]
    rows = []
    for raw in csv.reader(out.strip().splitlines()):
        if len(raw) < 6:
            continue
        index, name, mem_used, mem_total, util, power = [item.strip() for item in raw[:6]]
        mem_used_i = int(float(mem_used))
        mem_total_i = int(float(mem_total))
        rows.append(
            {
                "index": int(index),
                "name": name,
                "memory_used_mb": mem_used_i,
                "memory_total_mb": mem_total_i,
                "memory_free_mb": mem_total_i - mem_used_i,
                "utilization_gpu_percent": int(float(util)),
                "power_draw_w": float(power),
            }
        )
    return rows


def select_checkpoint() -> Path:
    training = load_json(TRAINING_RUN_JSON)
    run_dir = Path(training.get("outputs", {}).get("run_dir", ""))
    candidates = sorted((run_dir / "rank_0").glob("model_*.pt")) if run_dir.is_dir() else []
    if not candidates:
        return ROOT / "res/runs/tracking_g1_official_csv_loop_ppo_training/resource_adjusted_ppo_20260618_224626_seed20260629/rank_0/model_299.pt"

    def key(path: Path) -> int:
        try:
            return int(path.stem.split("_")[1])
        except Exception:
            return -1

    return max(candidates, key=key)


def select_vae_checkpoint() -> Path:
    summary = load_json(VAE_TRAINING_JSON)
    run_dir = Path(summary.get("outputs", {}).get("run_dir", ""))
    candidate = run_dir / "resource_adjusted_teacher_rollout_action_vae.pt"
    if candidate.is_file():
        return candidate
    return (
        ROOT
        / "res/runs/level_c_official_csv_loop_teacher_rollout_vae_training/"
        "resource_adjusted_teacher_rollout_vae_20260619_113654_seed20260632/"
        "resource_adjusted_teacher_rollout_action_vae.pt"
    )


def base_env(selected_gpu: int, checkpoint: Path, vae_checkpoint: Path, capture_npz: Path, metrics_json: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(selected_gpu),
            "VK_ICD_FILENAMES": str(PROJECT_EGL_ICD),
            "LD_LIBRARY_PATH": f"{GPU_FOUNDATION_DEPS}:{env.get('LD_LIBRARY_PATH', '')}",
            "PYTHONUNBUFFERED": "1",
            "ISAAC_PATH": str(ROOT / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim"),
            "OMNI_USER_DIR": str(ROOT / "cache/omni/user"),
            "OMNI_CACHE_DIR": str(ROOT / "cache/omni/cache"),
            "OMNI_DATA_DIR": str(ROOT / "cache/omni/data"),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "WANDB_MODE": "offline",
            "BM_ENRICHED_USD": str(ENRICHED_USD),
            "BM_MOTION_FILE": str(OFFICIAL_LOOP_MOTION_NPZ),
            "BM_CHECKPOINT": str(checkpoint),
            "BM_VAE_CHECKPOINT": str(vae_checkpoint),
            "BM_OUT_NPZ": str(capture_npz),
            "BM_METRICS_JSON": str(metrics_json),
            "BM_ROLLOUT_STEPS": str(ROLLOUT_STEPS),
            "BM_GUIDANCE_ALPHA": str(GUIDANCE_ALPHA),
            "BM_SEED": str(SEED),
        }
    )
    return env


def start_gpu_monitor(path: Path, selected_gpu: int) -> subprocess.Popen[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv",
            "-i",
            str(selected_gpu),
            "-l",
            "2",
            "-f",
            str(path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["variant", "reward_mean", "target_body_error_mean", "done_count_total", "teacher_vae_action_mse_mean", "guided_teacher_action_mse_mean"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    worker_path = OUT / "tracking_g1_official_csv_loop_action_guidance_rollout_worker.py"
    render_path = OUT / "tracking_g1_official_csv_loop_action_guidance_rollout_render.py"
    worker_path.write_text(textwrap.dedent(WORKER_CODE), encoding="utf-8")
    render_path.write_text(textwrap.dedent(RENDER_CODE), encoding="utf-8")

    checkpoint = select_checkpoint()
    vae_checkpoint = select_vae_checkpoint()
    gpu_snapshot = query_gpus()
    available = [
        row["index"]
        for row in gpu_snapshot
        if "index" in row
        and row.get("memory_free_mb", 0) >= MIN_FREE_MB
        and row.get("utilization_gpu_percent", 100) <= MAX_BUSY_UTIL
    ]
    selected_gpu = available[0] if available else None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RUN_ROOT / f"action_guidance_rollout_{timestamp}_seed{SEED}"
    capture_npz = run_dir / "official_csv_loop_action_guidance_rollout_trace.npz"
    metrics_json = run_dir / "official_csv_loop_action_guidance_rollout_metrics.json"
    gpu_metrics = run_dir / "gpu_metrics.csv"
    log_path = LOG_DIR / "tracking_g1_official_csv_loop_action_guidance_rollout_eval.log"
    asset_json = OUT / "official_csv_loop_action_guidance_rollout_asset.json"

    input_checks = {
        "tracking_python_exists": TRACKING_PY.is_file(),
        "analysis_python_exists": ANALYSIS_PY.is_file(),
        "checkpoint_exists": checkpoint.is_file(),
        "vae_checkpoint_exists": vae_checkpoint.is_file(),
        "motion_npz_exists": OFFICIAL_LOOP_MOTION_NPZ.is_file(),
        "enriched_usd_exists": ENRICHED_USD.is_file(),
        "body_contract_exists": BODY_CONTRACT.is_file(),
        "source_contract_exists": SOURCE_CONTRACT.is_file(),
        "selected_gpu_available": selected_gpu is not None,
    }
    run_info: dict[str, Any] = {
        "attempted_capture": False,
        "selected_gpu": selected_gpu,
        "run_dir": str(run_dir),
        "log": str(log_path),
        "capture_npz": str(capture_npz),
        "metrics_json": str(metrics_json),
        "gpu_metrics_csv": str(gpu_metrics),
    }
    capture_ok = False
    render_ok = False
    capture_metrics: dict[str, Any] = {}

    if all(input_checks.values()) and selected_gpu is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        monitor = start_gpu_monitor(gpu_metrics, selected_gpu)
        start = time.time()
        env = base_env(selected_gpu, checkpoint, vae_checkpoint, capture_npz, metrics_json)
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                [str(TRACKING_PY), str(worker_path)],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            returncode = proc.wait()
        monitor.terminate()
        try:
            monitor.wait(timeout=20)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait(timeout=20)
        capture_metrics = load_json(metrics_json)
        capture_ok = returncode == 0 and capture_npz.is_file() and capture_metrics.get("status") == "ok"
        run_info.update(
            {
                "attempted_capture": True,
                "returncode": returncode,
                "duration_seconds": round(time.time() - start, 3),
                "capture_metrics": capture_metrics,
                "capture_npz_exists": capture_npz.is_file(),
                "metrics_exists": metrics_json.is_file(),
            }
        )
        if capture_ok:
            render_env = os.environ.copy()
            render_env.update(
                {
                    "BM_CAPTURE_NPZ": str(capture_npz),
                    "BM_METRICS_JSON": str(metrics_json),
                    "BM_ASSET_JSON": str(asset_json),
                    "BM_BODY_CONTRACT": str(BODY_CONTRACT),
                    "BM_SOURCE_CONTRACT": str(SOURCE_CONTRACT),
                }
            )
            rc, render_out = run([str(ANALYSIS_PY), str(render_path)], env=render_env, timeout=900)
            run_info["render_returncode"] = rc
            run_info["render_output"] = render_out[-4000:]
            render_ok = rc == 0 and asset_json.is_file()
    else:
        run_info["reason_not_started"] = "Required inputs missing or no GPU 4/7 satisfied free-memory/utilization preflight."

    if capture_ok and render_ok:
        status = "ok_official_csv_loop_action_guidance_rollout_eval"
    elif capture_ok:
        status = "failed_action_guidance_rollout_render_after_capture"
    elif run_info.get("attempted_capture"):
        status = "failed_action_guidance_rollout_capture"
        failed_copy = FAILED_DIR / f"action_guidance_rollout_{timestamp}.log"
        failed_copy.write_text(log_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        run_info["failed_log_copy"] = str(failed_copy)
    else:
        status = "ok_with_resource_unavailable_before_action_guidance_rollout"

    variant_metrics = capture_metrics.get("variant_metrics", {})
    rows = []
    for variant, metrics in variant_metrics.items():
        rows.append(
            {
                "variant": variant,
                "reward_mean": metrics.get("reward_mean"),
                "target_body_error_mean": metrics.get("target_body_error_mean"),
                "done_count_total": metrics.get("done_count_total"),
                "teacher_vae_action_mse_mean": metrics.get("teacher_vae_action_mse_mean"),
                "guided_teacher_action_mse_mean": metrics.get("guided_teacher_action_mse_mean"),
            }
        )
    if rows:
        write_tsv(SUMMARY_TSV, rows)

    asset_data = load_json(asset_json)
    summary = {
        "status": status,
        "experiment_type": "tracking_g1_official_csv_loop_action_guidance_rollout_eval",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Runs 299-step teacher, VAE-base, and teacher-consistency action-guided variants in the local "
            "resource-adjusted official-csv-loop IsaacLab tracking task. This is a local virtual closed-loop bridge, "
            "not official latent diffusion guidance or paper Fig. 5/Fig. 6 evidence."
        ),
        "config": {
            "candidate_physical_gpus": CANDIDATE_GPUS,
            "selected_physical_gpu": selected_gpu,
            "cuda_visible_devices": str(selected_gpu) if selected_gpu is not None else "",
            "num_envs": 1,
            "seed": SEED,
            "rollout_steps": ROLLOUT_STEPS,
            "guidance_alpha": GUIDANCE_ALPHA,
            "min_free_mb_required_per_gpu": MIN_FREE_MB,
            "max_busy_util_percent_for_start": MAX_BUSY_UTIL,
            "formal_gpu_experiment": False,
            "why_not_formal_gpu_experiment": (
                "This is a single-environment report/evidence rollout, not PPO/diffusion training or paper-scale "
                "evaluation; therefore the >=10GB per GPU formal-experiment threshold is not applicable."
            ),
        },
        "gpu_preflight": {"snapshot": gpu_snapshot, "available_gpus": available},
        "inputs": {
            "checkpoint": str(checkpoint),
            "checkpoint_eval_json": str(CHECKPOINT_EVAL_JSON),
            "training_run_json": str(TRAINING_RUN_JSON),
            "vae_checkpoint": str(vae_checkpoint),
            "vae_training_json": str(VAE_TRAINING_JSON),
            "motion_npz": str(OFFICIAL_LOOP_MOTION_NPZ),
            "enriched_usd": str(ENRICHED_USD),
        },
        "input_checks": input_checks,
        "run": run_info,
        "metrics": {
            "rollout_steps": capture_metrics.get("rollout_steps"),
            "guidance": capture_metrics.get("guidance"),
            "variant_metrics": variant_metrics,
            "asset_metrics": asset_data.get("variant_metrics", {}),
        },
        "outputs": {
            "json": str(SUMMARY_JSON),
            "tsv": str(SUMMARY_TSV),
            "asset_json": str(asset_json),
            "worker_script": str(worker_path),
            "render_script": str(render_path),
            "run_dir": str(run_dir),
            "log": str(log_path),
            "capture_npz": str(capture_npz),
            "gpu_metrics_csv": str(gpu_metrics),
            "assets": asset_data.get("assets", {}),
        },
        "artifact_sha256": {
            key: sha256_file(Path(value))
            for key, value in {
                "summary_tsv": SUMMARY_TSV,
                "asset_json": asset_json,
            }.items()
            if Path(value).is_file()
        },
        "checks": {
            "capture_ok": capture_ok,
            "render_ok": render_ok,
            "asset_json_exists": asset_json.is_file(),
            "three_variants_evaluated": set(variant_metrics) == {"teacher", "vae_base", "action_guided"},
            "rollout_steps_299": capture_metrics.get("rollout_steps") == 299,
            "uses_official_csv_loop_motion": capture_metrics.get("official_csv_loop_motion") is True,
            "uses_resource_adjusted_usd": capture_metrics.get("uses_resource_adjusted_usd") is True,
            "does_not_claim_paper_level_guidance": True,
            "does_not_claim_receding_horizon_latent_diffusion": True,
            "does_not_claim_fig5_fig6": True,
            "does_not_claim_real_robot": True,
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "local_virtual_teacher_consistency_action_guidance_rollout" if capture_ok and render_ok else "not_completed",
            "why_not_complete": (
                "This is closed-loop IsaacLab evidence for a local action-guidance bridge using local PPO/VAE "
                "checkpoints. It is not the official BeyondMimic diffusion checkpoint, not the paper receding-horizon "
                "latent guidance controller, and not real-robot evidence."
            ),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "json": str(SUMMARY_JSON), "asset_json": str(asset_json)}, sort_keys=True))
    if status.startswith("failed_"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
