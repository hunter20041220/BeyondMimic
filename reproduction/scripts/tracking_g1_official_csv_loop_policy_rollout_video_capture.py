#!/usr/bin/env python3
"""Capture a local policy rollout body-pose trace and visualization video."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/visualization/official_csv_loop_policy_rollout"
LOG_DIR = ROOT / "logs/tracking_g1_official_csv_loop_policy_rollout_video_capture"
FAILED_DIR = ROOT / "res/failed_runs/tracking_g1_official_csv_loop_policy_rollout_video_capture"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_csv_loop_policy_rollout_video_capture"
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
BODY_CONTRACT = ROOT / "res/tracking/motion_npz_fixture/tracking_motion_npz_fixture.json"
SOURCE_CONTRACT = ROOT / "res/tracking/official_source_contract_audit/tracking_official_source_contract_audit.json"
CANDIDATE_GPUS = [4, 7]
MIN_FREE_MB = 20_000
MAX_BUSY_UTIL = 50
SEED = 20260637
ROLLOUT_STEPS = 299


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

print(f"BM_SENTINEL:policy_video:before_app:device={args.device}", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("BM_SENTINEL:policy_video:after_app", flush=True)

try:
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import torch
    import whole_body_tracking.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

    enriched_usd = Path(os.environ["BM_ENRICHED_USD"])
    motion_file = Path(os.environ["BM_MOTION_FILE"])
    checkpoint = Path(os.environ["BM_CHECKPOINT"])
    out_npz = Path(os.environ["BM_OUT_NPZ"])
    metrics_path = Path(os.environ["BM_METRICS_JSON"])
    rollout_steps = int(os.environ["BM_ROLLOUT_STEPS"])

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
    env_cfg.seed = int(os.environ["BM_SEED"])

    agent_cfg = G1FlatPPORunnerCfg()
    agent_cfg.device = args.device
    agent_cfg.seed = int(os.environ["BM_SEED"])
    agent_cfg.empirical_normalization = True
    agent_cfg.logger = "tensorboard"

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print("BM_SENTINEL:policy_video:env_created", flush=True)
    vec_env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
    obs, extras = vec_env.get_observations()
    command = vec_env.unwrapped.command_manager.get_term("motion")

    robot_body_pos = []
    reference_body_pos = []
    robot_anchor_pos = []
    reference_anchor_pos = []
    rewards = []
    dones = []
    action_abs_mean = []
    action_abs_max = []
    motion_time_steps = []
    metric_series = {}

    metric_names = ["error_anchor_pos", "error_body_pos", "error_joint_pos", "sampling_top1_prob"]
    with torch.inference_mode():
        for step in range(rollout_steps):
            actions = policy(obs)
            obs, rew, done, step_extras = vec_env.step(actions)
            # MotionCommand updates these tensors during the manager step.
            robot_body_pos.append(command.robot_body_pos_w[0].detach().cpu().numpy().astype(np.float32))
            reference_body_pos.append(command.body_pos_relative_w[0].detach().cpu().numpy().astype(np.float32))
            robot_anchor_pos.append(command.robot_anchor_pos_w[0].detach().cpu().numpy().astype(np.float32))
            reference_anchor_pos.append(command.anchor_pos_w[0].detach().cpu().numpy().astype(np.float32))
            rewards.append(float(rew.detach().cpu().mean()))
            dones.append(int(done.detach().cpu().sum()))
            action_abs_mean.append(float(actions.abs().mean().detach().cpu()))
            action_abs_max.append(float(actions.abs().max().detach().cpu()))
            if hasattr(command, "time_steps"):
                motion_time_steps.append(int(command.time_steps[0].detach().cpu()))
            else:
                motion_time_steps.append(step)
            for name in metric_names:
                value = command.metrics.get(name)
                metric_series.setdefault(name, []).append(
                    float(value[0].detach().cpu()) if value is not None and value.numel() > 0 else float("nan")
                )
            if (step + 1) % 50 == 0 or (step + 1) == rollout_steps:
                print(f"BM_SENTINEL:policy_video:step={step + 1}/{rollout_steps}", flush=True)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        robot_body_pos_w=np.stack(robot_body_pos, axis=0),
        reference_body_pos_w=np.stack(reference_body_pos, axis=0),
        robot_anchor_pos_w=np.stack(robot_anchor_pos, axis=0),
        reference_anchor_pos_w=np.stack(reference_anchor_pos, axis=0),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.int32),
        action_abs_mean=np.asarray(action_abs_mean, dtype=np.float32),
        action_abs_max=np.asarray(action_abs_max, dtype=np.float32),
        motion_time_steps=np.asarray(motion_time_steps, dtype=np.int32),
    )

    def summarize(values):
        values = list(values)
        return {
            "count": len(values),
            "mean": float(np.nanmean(values)),
            "min": float(np.nanmin(values)),
            "max": float(np.nanmax(values)),
        }

    metrics = {
        "status": "ok",
        "checkpoint": str(checkpoint),
        "motion_file": str(motion_file),
        "device": args.device,
        "num_envs": 1,
        "rollout_steps": rollout_steps,
        "loaded_iteration": int(runner.current_learning_iteration),
        "robot_body_pos_shape": list(np.stack(robot_body_pos, axis=0).shape),
        "reference_body_pos_shape": list(np.stack(reference_body_pos, axis=0).shape),
        "reward": summarize(rewards),
        "done_count_total": int(np.sum(dones)),
        "action_abs_mean": summarize(action_abs_mean),
        "action_abs_max": summarize(action_abs_max),
        "motion_time_step_min": int(np.min(motion_time_steps)),
        "motion_time_step_max": int(np.max(motion_time_steps)),
        "motion_metrics": {name: summarize(values) for name, values in metric_series.items()},
        "uses_resource_adjusted_usd": True,
        "official_csv_loop_motion": True,
        "paper_level_tracking_eval": False,
        "real_robot": False,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    vec_env.close()
    print(f"BM_SENTINEL:policy_video:npz={out_npz}", flush=True)
    print(f"BM_SENTINEL:policy_video:metrics={metrics_path}", flush=True)
    os._exit(0)
except BaseException as exc:
    print(f"BM_SENTINEL:policy_video:exception={exc!r}", flush=True)
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

root = Path(os.environ["BM_ROOT"])
npz_path = Path(os.environ["BM_CAPTURE_NPZ"])
asset_json = Path(os.environ["BM_ASSET_JSON"])
out_dir = asset_json.parent
body_contract = json.loads(Path(os.environ["BM_BODY_CONTRACT"]).read_text())
source_contract = json.loads(Path(os.environ["BM_SOURCE_CONTRACT"]).read_text())
capture_summary = json.loads(Path(os.environ["BM_CAPTURE_SUMMARY"]).read_text())
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
robot = data["robot_body_pos_w"]
reference = data["reference_body_pos_w"]
rewards = data["rewards"]
actions = data["action_abs_mean"]
if robot.shape[1] == len(target_names):
    names = target_names
    target_indices = list(range(len(target_names)))
elif robot.shape[1] == len(body_contract["body_names_urdf_order"]):
    names = body_contract["body_names_urdf_order"]
    name_to_idx_full = {name: idx for idx, name in enumerate(names)}
    target_indices = [name_to_idx_full[name] for name in target_names]
else:
    raise ValueError(f"Unexpected body count {robot.shape[1]} for policy rollout trace")
name_to_idx = {name: idx for idx, name in enumerate(names)}
target_robot = robot[:, target_indices, :]
target_reference = reference[:, target_indices, :]

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def set_axes_equal(ax):
    xyz = np.concatenate([target_robot, target_reference], axis=1)
    mins = xyz.min(axis=(0, 1))
    maxs = xyz.max(axis=(0, 1))
    centers = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 0.4)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(max(0.0, centers[2] - radius), centers[2] + radius)

def draw_skeleton(ax, xyz, color, label, linewidth=2.0):
    for a, b in edges:
        pts = xyz[[name_to_idx[a], name_to_idx[b]]]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=linewidth, alpha=0.9)
    ax.scatter(xyz[target_indices, 0], xyz[target_indices, 1], xyz[target_indices, 2], s=22, color=color, alpha=0.9, label=label)

def draw_frame(ax, frame):
    draw_skeleton(ax, reference[frame], "#64748b", "reference", linewidth=1.5)
    draw_skeleton(ax, robot[frame], "#2563eb", "policy robot", linewidth=2.2)
    ax.set_title(f"Policy rollout vs reference, frame {frame:03d}", fontsize=11)
    ax.legend(loc="upper right")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

out_dir.mkdir(parents=True, exist_ok=True)
video_path = out_dir / "official_csv_loop_policy_rollout_vs_reference.mp4"
keyframes_path = out_dir / "official_csv_loop_policy_rollout_keyframes.png"
metrics_csv = out_dir / "official_csv_loop_policy_rollout_metrics.csv"
readme = out_dir / "README.md"

plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(7.2, 6.0))
ax = fig.add_subplot(111, projection="3d")
writer = FFMpegWriter(fps=30, metadata={"title": "BeyondMimic policy rollout vs reference"})
with writer.saving(fig, str(video_path), dpi=150):
    for frame in range(robot.shape[0]):
        ax.cla()
        draw_frame(ax, frame)
        set_axes_equal(ax)
        ax.view_init(elev=18, azim=-68)
        writer.grab_frame()
plt.close(fig)

frames = [0, robot.shape[0] // 3, 2 * robot.shape[0] // 3, robot.shape[0] - 1]
fig = plt.figure(figsize=(13, 7))
for idx, frame in enumerate(frames, start=1):
    ax = fig.add_subplot(2, 2, idx, projection="3d")
    draw_frame(ax, frame)
    set_axes_equal(ax)
    ax.view_init(elev=18, azim=-68)
fig.tight_layout()
fig.savefig(keyframes_path, dpi=180)
plt.close(fig)

tracking_error = np.linalg.norm(target_robot - target_reference, axis=-1).mean(axis=1)
with metrics_csv.open("w", encoding="utf-8", newline="") as f:
    writer_csv = csv.DictWriter(f, fieldnames=["step", "reward", "action_abs_mean", "target_body_error_mean"])
    writer_csv.writeheader()
    for step in range(robot.shape[0]):
        writer_csv.writerow({
            "step": step,
            "reward": float(rewards[step]),
            "action_abs_mean": float(actions[step]),
            "target_body_error_mean": float(tracking_error[step]),
        })

readme.write_text("\n".join([
    "# Official-Loop Policy Rollout Visualization",
    "",
    "This directory contains a local virtual policy rollout visualization captured from the official csv-loop PPO checkpoint.",
    "",
    "## Claim Level",
    "",
    "local_virtual_resource_adjusted_policy_rollout_video. This is not unpatched official BeyondMimic replay, not paper-level Fig. 5/Fig. 6 guided diffusion, and not real-robot evidence.",
    "",
]), encoding="utf-8")

assets = {
    "mp4": str(video_path),
    "keyframes_png": str(keyframes_path),
    "metrics_csv": str(metrics_csv),
    "readme": str(readme),
}
summary = {
    "status": "ok",
    "experiment_type": "tracking_official_csv_loop_policy_rollout_video_capture",
    "claim_level": "local_virtual_resource_adjusted_policy_rollout_video",
    "source_capture_npz": str(npz_path),
    "source_capture_summary": os.environ["BM_CAPTURE_SUMMARY"],
    "source_capture_status": capture_summary["status"],
    "frame_count": int(robot.shape[0]),
    "body_count": int(robot.shape[1]),
    "target_body_count": len(target_indices),
    "metrics": {
        "reward_mean": float(np.mean(rewards)),
        "done_count_total": int(np.sum(data["dones"])),
        "action_abs_mean": float(np.mean(actions)),
        "target_body_error_mean": float(np.mean(tracking_error)),
        "target_body_error_max": float(np.max(tracking_error)),
    },
    "assets": assets,
    "asset_sizes": {key: Path(value).stat().st_size for key, value in assets.items()},
    "asset_sha256": {key: sha256_file(Path(value)) for key, value in assets.items()},
    "checks": {
        "capture_status_ok": capture_summary["status"] == "ok_official_csv_loop_policy_rollout_video_capture",
        "frame_count_299": int(robot.shape[0]) == 299,
        "body_count_supported_14_or_40": int(robot.shape[1]) in {14, 40},
        "target_body_count_14": len(target_indices) == 14,
        "video_exists_nonempty": video_path.is_file() and video_path.stat().st_size > 0,
        "keyframes_exist_nonempty": keyframes_path.is_file() and keyframes_path.stat().st_size > 0,
        "does_not_claim_paper_level": True,
        "does_not_claim_fig5_fig6": True,
        "does_not_claim_real_robot": True,
    },
    "interpretation": {
        "goal_complete": False,
        "why_not_complete": "This is a local virtual policy rollout video from the resource-adjusted official csv-loop PPO checkpoint, not paper-level guided diffusion or real-robot validation.",
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


def base_env(run_dir: Path, selected_gpu: int, checkpoint: Path, capture_npz: Path, metrics_json: Path) -> dict[str, str]:
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
            "BM_OUT_NPZ": str(capture_npz),
            "BM_METRICS_JSON": str(metrics_json),
            "BM_ROLLOUT_STEPS": str(ROLLOUT_STEPS),
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
            "5",
            "-f",
            str(path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    worker_path = OUT / "tracking_g1_official_csv_loop_policy_rollout_worker.py"
    render_path = OUT / "tracking_g1_official_csv_loop_policy_rollout_render.py"
    worker_path.write_text(textwrap.dedent(WORKER_CODE), encoding="utf-8")
    render_path.write_text(textwrap.dedent(RENDER_CODE), encoding="utf-8")

    checkpoint = select_checkpoint()
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
    run_dir = RUN_ROOT / f"policy_rollout_capture_{timestamp}_seed{SEED}"
    capture_npz = run_dir / "official_csv_loop_policy_rollout_body_pose_trace.npz"
    metrics_json = run_dir / "official_csv_loop_policy_rollout_capture_metrics.json"
    gpu_metrics = run_dir / "gpu_metrics.csv"
    log_path = LOG_DIR / "tracking_g1_official_csv_loop_policy_rollout_video_capture.log"
    asset_json = OUT / "official_csv_loop_policy_rollout_video_asset.json"

    input_checks = {
        "tracking_python_exists": TRACKING_PY.is_file(),
        "analysis_python_exists": ANALYSIS_PY.is_file(),
        "checkpoint_exists": checkpoint.is_file(),
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

    if all(input_checks.values()) and selected_gpu is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        monitor = start_gpu_monitor(gpu_metrics, selected_gpu)
        start = time.time()
        env = base_env(run_dir, selected_gpu, checkpoint, capture_npz, metrics_json)
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
                    "BM_ROOT": str(ROOT),
                    "BM_CAPTURE_NPZ": str(capture_npz),
                    "BM_ASSET_JSON": str(asset_json),
                    "BM_CAPTURE_SUMMARY": str(OUT / "tracking_g1_official_csv_loop_policy_rollout_capture.json"),
                    "BM_BODY_CONTRACT": str(BODY_CONTRACT),
                    "BM_SOURCE_CONTRACT": str(SOURCE_CONTRACT),
                }
            )
            # Write the capture summary before rendering because the render summary references it.
            interim = {
                "status": "ok_official_csv_loop_policy_rollout_video_capture",
                "run": run_info,
                "inputs": {
                    "checkpoint": str(checkpoint),
                    "motion_npz": str(OFFICIAL_LOOP_MOTION_NPZ),
                    "enriched_usd": str(ENRICHED_USD),
                },
                "input_checks": input_checks,
            }
            (OUT / "tracking_g1_official_csv_loop_policy_rollout_capture.json").write_text(
                json.dumps(interim, indent=2, sort_keys=True), encoding="utf-8"
            )
            rc, render_out = run([str(ANALYSIS_PY), str(render_path)], env=render_env, timeout=600)
            run_info["render_returncode"] = rc
            run_info["render_output"] = render_out[-4000:]
            render_ok = rc == 0 and asset_json.is_file()
    else:
        run_info["reason_not_started"] = "Required inputs missing or no GPU 4/7 satisfied free-memory/utilization preflight."

    if capture_ok and render_ok:
        status = "ok_official_csv_loop_policy_rollout_video_capture"
    elif capture_ok:
        status = "failed_policy_rollout_render_after_capture"
    elif run_info.get("attempted_capture"):
        status = "failed_policy_rollout_capture"
        failed_copy = FAILED_DIR / f"policy_rollout_capture_{timestamp}.log"
        failed_copy.write_text(log_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        run_info["failed_log_copy"] = str(failed_copy)
    else:
        status = "ok_with_resource_unavailable_before_policy_rollout_capture"

    summary = {
        "status": status,
        "experiment_type": "tracking_official_csv_loop_policy_rollout_video_capture",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Captures one local virtual rollout from the official csv-loop PPO checkpoint and records robot/reference "
            "body poses for a report video. This is not unpatched official replay, not paper Fig.5/Fig.6 guided "
            "diffusion, and not real-robot evidence."
        ),
        "config": {
            "candidate_physical_gpus": CANDIDATE_GPUS,
            "selected_physical_gpu": selected_gpu,
            "cuda_visible_devices": str(selected_gpu) if selected_gpu is not None else "",
            "num_envs": 1,
            "rollout_steps": ROLLOUT_STEPS,
            "seed": SEED,
            "min_free_mb_required_per_gpu": MIN_FREE_MB,
            "max_busy_util_percent_for_start": MAX_BUSY_UTIL,
        },
        "gpu_preflight": {"snapshot": gpu_snapshot, "available_gpus": available},
        "inputs": {
            "training_run_json": str(TRAINING_RUN_JSON),
            "checkpoint_eval_json": str(CHECKPOINT_EVAL_JSON),
            "checkpoint": str(checkpoint),
            "motion_npz": str(OFFICIAL_LOOP_MOTION_NPZ),
            "enriched_usd": str(ENRICHED_USD),
        },
        "input_checks": input_checks,
        "run": run_info,
        "outputs": {
            "json": str(OUT / "tracking_g1_official_csv_loop_policy_rollout_capture.json"),
            "asset_json": str(asset_json),
            "worker_script": str(worker_path),
            "render_script": str(render_path),
            "run_dir": str(run_dir),
            "log": str(log_path),
            "capture_npz": str(capture_npz),
            "gpu_metrics_csv": str(gpu_metrics),
        },
        "checks": {
            "capture_ok": capture_ok,
            "render_ok": render_ok,
            "asset_json_exists": asset_json.is_file(),
            "does_not_claim_paper_level": True,
            "does_not_claim_fig5_fig6": True,
            "does_not_claim_real_robot": True,
        },
        "interpretation": {
            "goal_complete": False,
            "policy_rollout_video_complete": capture_ok and render_ok,
            "paper_level_status": "local_virtual_resource_adjusted_policy_rollout_video" if capture_ok and render_ok else "not_completed",
            "why_not_complete": (
                "Even when successful, this video is generated from a local resource-adjusted official csv-loop PPO "
                "checkpoint and does not prove paper-level guided diffusion or real-robot behavior."
            ),
        },
    }
    (OUT / "tracking_g1_official_csv_loop_policy_rollout_capture.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": status, "json": summary["outputs"]["json"], "asset_json": str(asset_json)}, sort_keys=True))
    if status.startswith("failed_"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
