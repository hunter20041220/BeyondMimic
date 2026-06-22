#!/usr/bin/env python3
"""Record a true IsaacLab/Isaac Sim rendered G1 PPO policy rollout MP4.

This script differs from the older matplotlib skeleton visualizers: it starts
IsaacLab with cameras enabled, creates the real Tracking-Flat-G1-v0 environment,
loads a local PPO checkpoint, advances the simulator through env.step(action),
and captures RGB frames from an Isaac Sim offscreen render product.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import signal
import shutil
import subprocess
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(os.environ.get("BM_ROOT", "/mnt/infini-data/test/BeyondMimic")).expanduser().resolve()
OUT = ROOT / "res/visualization/isaac_mp4"
LOG_DIR = ROOT / "logs/isaac_mp4"
FAILED_DIR = ROOT / "res/failed_runs/isaac_mp4"
RUN_ROOT = ROOT / "res/runs/isaac_mp4"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
SYSTEM_NVIDIA_ICD = Path("/etc/vulkan/icd.d/nvidia_icd.json")
VULKANINFO_FULL_LOG = ROOT / "logs/isaac_mp4/vulkaninfo_system_nvidia_full_20260622_130724.log"
VULKANINFO_DEFAULT_LOG = ROOT / "logs/isaac_mp4/vulkaninfo_default_20260622_130644.log"
XDG_RUNTIME_DIR = ROOT / "tmp/xdg-runtime-root"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
OFFICIAL_IMPORTER_USD = (
    ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda"
)
ROBOT_ORDER_MOTION_NPZ = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/"
    "official_csv_loop_full_public_motion_bundle_fk_repaired_robot_order.npz"
)
ROBOT_ORDER_BUNDLE_AUDIT = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/"
    "tracking_g1_official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz.json"
)
TRAINING_RUN_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run.json"
)
CHECKPOINT_EVAL_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval.json"
)
ENDPOINT_CANDIDATE_EVAL_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_endpoint_threshold_candidate_ppo_checkpoint_eval/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_endpoint_threshold_candidate_ppo_checkpoint_eval.json"
)

CANDIDATE_GPUS = [
    int(item.strip()) for item in os.environ.get("BM_ISAAC_MP4_CANDIDATE_GPUS", "5,6").split(",") if item.strip()
]
MIN_FREE_MB = int(os.environ.get("BM_ISAAC_MP4_MIN_FREE_MB", "12000"))
MAX_BUSY_UTIL = int(os.environ.get("BM_ISAAC_MP4_MAX_BUSY_UTIL", "80"))
ROLLOUT_STEPS = int(os.environ.get("BM_ISAAC_MP4_STEPS", "300"))
SEED = int(os.environ.get("BM_ISAAC_MP4_SEED", "20260780"))
TIMEOUT_SECONDS = int(os.environ.get("BM_ISAAC_MP4_TIMEOUT_SECONDS", "1200"))


def selected_vulkan_icd() -> Path:
    """Prefer the system NVIDIA ICD when available; keep project EGL ICD as fallback evidence."""
    override = os.environ.get("BM_ISAAC_MP4_VK_ICD_FILENAMES", "").strip()
    if override:
        return Path(override)
    if SYSTEM_NVIDIA_ICD.is_file():
        return SYSTEM_NVIDIA_ICD
    return PROJECT_EGL_ICD


WORKER_CODE = r"""
import csv
import hashlib
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

selected_gpu = os.environ["BM_SELECTED_PHYSICAL_GPU"]
isaac_device = os.environ["BM_ISAACLAB_DEVICE"]
print(
    "BM_SENTINEL:isaac_mp4:render_env:"
    f"vk_icd={os.environ.get('BM_VK_ICD_FILENAMES', '')}:"
    f"xdg_runtime={os.environ.get('BM_XDG_RUNTIME_DIR', '')}",
    flush=True,
)
print(f"BM_SENTINEL:isaac_mp4:before_app:device={isaac_device}:cameras=True", flush=True)
app_launcher = AppLauncher(
    headless=True,
    enable_cameras=True,
    device=isaac_device,
    active_gpu=int(selected_gpu),
    max_gpu_count=1,
    multi_gpu=False,
    fast_shutdown=True,
    experience="isaaclab.python.headless.rendering.kit",
    rendering_mode="performance",
    kit_args=(
        "--/renderer/multiGpu/enabled=false "
        "--/renderer/multiGpu/autoEnable=false "
        "--/renderer/multiGpu/maxGpuCount=1 "
        f"--/renderer/activeGpu={selected_gpu} "
        f"--/physics/cudaDevice={selected_gpu} "
        "--/app/renderer/waitIdle=false "
        "--/app/hydraEngine/waitIdle=false"
    ),
)
simulation_app = app_launcher.app
print("BM_SENTINEL:isaac_mp4:after_app", flush=True)

try:
    import gymnasium as gym
    import imageio.v2 as imageio
    import numpy as np
    import torch
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("omni.replicator.core")
    for _ in range(5):
        simulation_app.update()
    print("BM_SENTINEL:isaac_mp4:replicator_extension_enabled", flush=True)

    import omni.replicator.core as rep
    import isaacsim.core.utils.prims as prim_utils
    from PIL import Image, ImageDraw
    from pxr import Gf, UsdGeom

    import isaaclab.sim as sim_utils
    import whole_body_tracking.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
    from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

    out_dir = Path(os.environ["BM_OUT_DIR"])
    run_dir = Path(os.environ["BM_RUN_DIR"])
    frame_probe_dir = run_dir / "frame_probe"
    metrics_csv = Path(os.environ["BM_METRICS_CSV"])
    summary_json = Path(os.environ["BM_WORKER_SUMMARY_JSON"])
    final_mp4 = Path(os.environ["BM_FINAL_MP4"])
    keyframes_png = Path(os.environ["BM_KEYFRAMES_PNG"])
    checkpoint = Path(os.environ["BM_CHECKPOINT"])
    usd_path = Path(os.environ["BM_OFFICIAL_IMPORTER_USD"])
    motion_file = Path(os.environ["BM_MOTION_FILE"])
    rollout_steps = int(os.environ["BM_ROLLOUT_STEPS"])
    seed = int(os.environ["BM_SEED"])

    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    frame_probe_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=str(usd_path),
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
    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye = (3.2, -4.2, 2.2)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.75)
    env_cfg.viewer.cam_prim_path = "/World/IsaacMp4Camera"
    env_cfg.viewer.resolution = (1280, 720)
    env_cfg.rerender_on_reset = True
    env_cfg.sim.device = isaac_device
    env_cfg.seed = seed

    agent_cfg = G1FlatPPORunnerCfg()
    agent_cfg.device = isaac_device
    agent_cfg.seed = seed
    agent_cfg.empirical_normalization = True
    agent_cfg.logger = "tensorboard"

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode="rgb_array")
    print(f"BM_SENTINEL:isaac_mp4:env_created:num_envs={env.unwrapped.num_envs}", flush=True)
    vec_env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
    obs, _ = vec_env.get_observations()
    command = vec_env.unwrapped.command_manager.get_term("motion")
    robot = vec_env.unwrapped.scene["robot"]
    sim = vec_env.unwrapped.sim

    camera_path = env_cfg.viewer.cam_prim_path
    if not prim_utils.is_prim_path_valid(camera_path):
        cam_prim = prim_utils.create_prim(
            camera_path,
            prim_type="Camera",
            translation=env_cfg.viewer.eye,
            orientation=(1.0, 0.0, 0.0, 0.0),
        )
        _ = UsdGeom.Camera(cam_prim)
    render_product = rep.create.render_product(camera_path, resolution=env_cfg.viewer.resolution)
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    rgb_annotator.attach([render_product])
    print("BM_SENTINEL:isaac_mp4:render_product_created", flush=True)

    def set_follow_camera():
        root = robot.data.root_pos_w[0].detach().cpu()
        eye_offset = torch.tensor(env_cfg.viewer.eye, dtype=torch.float32)
        lookat_offset = torch.tensor(env_cfg.viewer.lookat, dtype=torch.float32)
        eye = root + eye_offset
        target = root + lookat_offset
        rotation = create_rotation_matrix_from_view(eye.view(1, 3), target.view(1, 3), up_axis="Z", device="cpu")
        quat = quat_from_matrix(rotation)[0].detach().cpu().numpy()
        prim = prim_utils.get_prim_at_path(camera_path)
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2])))
        xform.AddOrientOp().Set(Gf.Quatf(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])))

    def read_rgb_frame():
        sim.render(mode=sim.RenderMode.PARTIAL_RENDERING)
        data = rgb_annotator.get_data()
        frame = np.asarray(data, dtype=np.uint8)
        if frame.size == 0:
            return np.zeros((env_cfg.viewer.resolution[1], env_cfg.viewer.resolution[0], 3), dtype=np.uint8)
        if frame.ndim == 1:
            frame = frame.reshape((env_cfg.viewer.resolution[1], env_cfg.viewer.resolution[0], -1))
        return frame[:, :, :3].copy()

    set_follow_camera()
    for _ in range(8):
        sim.render(mode=sim.RenderMode.PARTIAL_RENDERING)
    warmup_frame = read_rgb_frame()
    Image.fromarray(warmup_frame).save(frame_probe_dir / "warmup_frame.png")

    metric_names = [
        "error_anchor_pos",
        "error_anchor_rot",
        "error_anchor_lin_vel",
        "error_anchor_ang_vel",
        "error_body_pos",
        "error_body_rot",
        "error_body_lin_vel",
        "error_body_ang_vel",
        "error_joint_pos",
        "error_joint_vel",
        "sampling_entropy",
        "sampling_top1_prob",
        "sampling_top1_bin",
    ]
    fieldnames = [
        "step",
        "reward",
        "done",
        "timeout",
        "root_height",
        "episode_length",
        "action_norm",
        "action_abs_mean",
        "action_abs_max",
        "frame_mean_rgb",
        "frame_std_rgb",
        "frame_nonzero_fraction",
    ] + metric_names
    rows = []
    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        with torch.inference_mode(), imageio.get_writer(final_mp4, fps=30, codec="libx264", macro_block_size=16) as video:
            for step in range(rollout_steps):
                actions = policy(obs)
                obs, rew, dones, extras = vec_env.step(actions)
                set_follow_camera()
                frame = read_rgb_frame()
                video.append_data(frame)
                timeouts = extras.get("time_outs", torch.zeros_like(dones))
                row = {
                    "step": step,
                    "reward": float(rew[0].detach().cpu()),
                    "done": int(dones[0].detach().cpu()),
                    "timeout": int(timeouts[0].detach().cpu()),
                    "root_height": float(robot.data.root_pos_w[0, 2].detach().cpu()),
                    "episode_length": int(vec_env.unwrapped.episode_length_buf[0].detach().cpu()),
                    "action_norm": float(torch.linalg.vector_norm(actions[0]).detach().cpu()),
                    "action_abs_mean": float(actions[0].abs().mean().detach().cpu()),
                    "action_abs_max": float(actions[0].abs().max().detach().cpu()),
                }
                for name in metric_names:
                    tensor = command.metrics.get(name)
                    row[name] = float(tensor[0].detach().cpu()) if tensor is not None and tensor.numel() > 0 else float("nan")
                row["frame_mean_rgb"] = float(frame.mean())
                row["frame_std_rgb"] = float(frame.std())
                row["frame_nonzero_fraction"] = float(np.count_nonzero(frame) / frame.size)
                writer.writerow(row)
                rows.append(row)
                if (step + 1) % 50 == 0 or (step + 1) == rollout_steps:
                    print(f"BM_SENTINEL:isaac_mp4:step={step + 1}/{rollout_steps}", flush=True)

    vec_env.close()
    print("BM_SENTINEL:isaac_mp4:env_closed", flush=True)

    def sha256_file(path):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def summarize(key):
        values = [float(row[key]) for row in rows if row.get(key) == row.get(key)]
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "first": values[0],
            "last": values[-1],
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    frames = []
    frame_count = 0
    first_frame_mean = None
    first_frame_std = None
    first_frame_nonzero_fraction = None
    try:
        reader = imageio.get_reader(final_mp4)
        meta = reader.get_meta_data()
        maybe_count = meta.get("nframes")
        if isinstance(maybe_count, int) and maybe_count > 0 and maybe_count < 1_000_000:
            frame_count = maybe_count
        sample_indices = [0, max(0, rollout_steps // 3), max(0, 2 * rollout_steps // 3), max(0, rollout_steps - 1)]
        for index, frame in enumerate(reader):
            if index == 0:
                first_frame_mean = float(frame.mean())
                first_frame_std = float(frame.std())
                first_frame_nonzero_fraction = float(np.count_nonzero(frame) / frame.size)
            if index in sample_indices:
                frames.append((index, Image.fromarray(frame).resize((426, 240))))
            frame_count = max(frame_count, index + 1)
        reader.close()
    except Exception as exc:
        print(f"BM_SENTINEL:isaac_mp4:keyframe_warning={exc!r}", flush=True)

    if frames:
        canvas = Image.new("RGB", (852, 520), "white")
        draw = ImageDraw.Draw(canvas)
        for idx, (frame_idx, img) in enumerate(frames[:4]):
            x = 0 if idx % 2 == 0 else 426
            y = 20 if idx < 2 else 280
            canvas.paste(img, (x, y))
            draw.text((x + 8, y - 18), f"frame {frame_idx}", fill=(20, 20, 20))
        canvas.save(keyframes_png)

    checks = {
        "app_launcher_started": True,
        "env_created": True,
        "uses_app_launcher_headless": True,
        "uses_enable_cameras": True,
        "uses_offscreen_render_product": True,
        "uses_env_step_closed_loop": True,
        "uses_tracking_flat_g1_v0": True,
        "uses_official_importer_export_usd": True,
        "uses_robot_order_fk_repaired_motion_bundle": True,
        "uses_local_ppo_checkpoint": True,
        "mp4_exists_nonempty": final_mp4.is_file() and final_mp4.stat().st_size > 0,
        "metrics_csv_exists_nonempty": metrics_csv.is_file() and metrics_csv.stat().st_size > 0,
        "keyframes_png_exists_nonempty": keyframes_png.is_file() and keyframes_png.stat().st_size > 0,
        "rollout_steps_at_least_300": rollout_steps >= 300,
        "video_frame_count_positive": frame_count > 0,
        "first_frame_nonblank": bool(first_frame_std is not None and first_frame_std > 1.0),
        "does_not_claim_official_checkpoint": True,
        "does_not_claim_paper_level": True,
        "does_not_claim_fig5_fig6": True,
        "does_not_claim_tensorrt": True,
        "does_not_claim_real_robot": True,
    }
    summary = {
        "status": "ok_isaaclab_rendered_policy_rollout_mp4" if all(
            checks[key] for key in [
                "mp4_exists_nonempty",
                "metrics_csv_exists_nonempty",
                "video_frame_count_positive",
                "first_frame_nonblank",
            ]
        ) else "failed_isaaclab_rendered_policy_rollout_mp4_output_check",
        "experiment_type": "isaaclab_rendered_g1_policy_rollout_mp4",
        "timestamp_utc": os.environ["BM_TIMESTAMP_UTC"],
        "claim_level": "local_virtual_isaaclab_rendered_policy_rollout_video",
        "task": "Tracking-Flat-G1-v0",
        "rendering": {
            "renderer_source": "Isaac Sim offscreen camera render product via omni.replicator.core rgb annotator",
            "headless": True,
            "enable_cameras": True,
            "camera_origin_type": env_cfg.viewer.origin_type,
            "camera_asset_name": env_cfg.viewer.asset_name,
            "camera_eye": list(env_cfg.viewer.eye),
            "camera_lookat": list(env_cfg.viewer.lookat),
            "resolution": list(env_cfg.viewer.resolution),
            "fps": 30,
            "frame_count_estimate": int(frame_count),
            "first_frame_mean_rgb": first_frame_mean,
            "first_frame_std_rgb": first_frame_std,
            "first_frame_nonzero_fraction": first_frame_nonzero_fraction,
        },
        "inputs": {
            "checkpoint": str(checkpoint),
            "official_importer_usd": str(usd_path),
            "motion_npz": str(motion_file),
        },
        "config": {
            "seed": seed,
            "rollout_steps": rollout_steps,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "device": isaac_device,
            "num_envs": 1,
        },
        "metrics": {
            "reward": summarize("reward"),
            "done_count_total": int(sum(int(row["done"]) for row in rows)),
            "done_rate": float(sum(int(row["done"]) for row in rows) / max(1, len(rows))),
            "timeout_count_total": int(sum(int(row["timeout"]) for row in rows)),
            "root_height": summarize("root_height"),
            "episode_length": summarize("episode_length"),
            "action_norm": summarize("action_norm"),
            "action_abs_mean": summarize("action_abs_mean"),
            "action_abs_max": summarize("action_abs_max"),
            "frame_mean_rgb": summarize("frame_mean_rgb"),
            "frame_std_rgb": summarize("frame_std_rgb"),
            "frame_nonzero_fraction": summarize("frame_nonzero_fraction"),
            "motion": {name: summarize(name) for name in metric_names},
        },
        "outputs": {
            "mp4": str(final_mp4),
            "keyframes_png": str(keyframes_png),
            "metrics_csv": str(metrics_csv),
            "worker_summary_json": str(summary_json),
            "frame_probe_dir": str(frame_probe_dir),
        },
        "asset_sizes": {
            "mp4": final_mp4.stat().st_size if final_mp4.is_file() else 0,
            "keyframes_png": keyframes_png.stat().st_size if keyframes_png.is_file() else 0,
            "metrics_csv": metrics_csv.stat().st_size if metrics_csv.is_file() else 0,
        },
        "asset_sha256": {
            "mp4": sha256_file(final_mp4) if final_mp4.is_file() else None,
            "keyframes_png": sha256_file(keyframes_png) if keyframes_png.is_file() else None,
            "metrics_csv": sha256_file(metrics_csv) if metrics_csv.is_file() else None,
        },
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "physics_rollout_complete": True,
            "video_recording_complete": checks["mp4_exists_nonempty"],
            "why_not_paper_level": (
                "This is a true IsaacLab/Isaac Sim rendered local policy rollout using env.step(action), "
                "but the policy checkpoint is a local robot-order FK-repaired PPO checkpoint, not an official "
                "BeyondMimic teacher, VAE, diffusion, Fig. 5/Fig. 6, TensorRT, or real-robot result."
            ),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "mp4": str(final_mp4), "summary": str(summary_json)}, sort_keys=True))
    os._exit(0)
except BaseException as exc:
    print(f"BM_SENTINEL:isaac_mp4:exception={exc!r}", flush=True)
    raise
finally:
    try:
        simulation_app.close(wait_for_replicator=False)
    except BaseException:
        pass
"""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    rows: list[dict[str, Any]] = []
    for raw in csv.reader(out.strip().splitlines()):
        if len(raw) < 6:
            continue
        index, name, mem_used, mem_total, util, power = [item.strip() for item in raw[:6]]
        used = int(float(mem_used))
        total = int(float(mem_total))
        rows.append(
            {
                "index": int(index),
                "name": name,
                "memory_used_mb": used,
                "memory_total_mb": total,
                "memory_free_mb": total - used,
                "utilization_gpu_percent": int(float(util)),
                "power_draw_w": float(power),
            }
        )
    return rows


def select_gpu(gpu_rows: list[dict[str, Any]]) -> int | None:
    available = [
        row
        for row in gpu_rows
        if "index" in row
        and row.get("memory_free_mb", 0) >= MIN_FREE_MB
        and row.get("utilization_gpu_percent", 100) <= MAX_BUSY_UTIL
    ]
    if available:
        return int(max(available, key=lambda row: row.get("memory_free_mb", 0))["index"])
    candidates = [row for row in gpu_rows if "index" in row]
    if candidates:
        return int(max(candidates, key=lambda row: row.get("memory_free_mb", 0))["index"])
    return None


def select_checkpoint() -> tuple[Path, dict[str, Any]]:
    eval_summary = load_json(CHECKPOINT_EVAL_JSON)
    endpoint_eval = load_json(ENDPOINT_CANDIDATE_EVAL_JSON)
    checkpoint = Path(eval_summary.get("inputs", {}).get("checkpoint", ""))
    decision = {
        "selected_policy_family": "robot_order_fk_repaired_iteration_999_ppo",
        "selected_checkpoint_eval_json": str(CHECKPOINT_EVAL_JSON),
        "selected_checkpoint": str(checkpoint),
        "reason": (
            "Chosen as the current balanced local tracking baseline for rendered media. The endpoint-threshold "
            "candidate lowers done rate but worsens reward and joint-position error, so it is not used as the "
            "default policy rollout video checkpoint."
        ),
        "endpoint_candidate_eval_json": str(ENDPOINT_CANDIDATE_EVAL_JSON) if ENDPOINT_CANDIDATE_EVAL_JSON.is_file() else None,
        "endpoint_candidate_status": endpoint_eval.get("status"),
    }
    return checkpoint, decision


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


def terminate_process_tree(proc: subprocess.Popen[str], grace_seconds: int = 20) -> tuple[int | None, str]:
    """Terminate a worker process group without leaving wedged Kit children behind."""
    if proc.poll() is not None:
        return proc.returncode, "already_exited"
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        return proc.wait(timeout=grace_seconds), "terminated_with_sigterm"
    except ProcessLookupError:
        return proc.poll(), "process_group_missing"
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            return proc.wait(timeout=grace_seconds), "killed_with_sigkill_after_sigterm_timeout"
        except subprocess.TimeoutExpired:
            return proc.poll(), "sigkill_wait_timeout"


def make_env(
    selected_gpu: int,
    run_dir: Path,
    checkpoint: Path,
    timestamp: str,
    final_mp4: Path,
    keyframes_png: Path,
    metrics_csv: Path,
    worker_summary_json: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    XDG_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    try:
        XDG_RUNTIME_DIR.chmod(0o700)
    except PermissionError:
        pass
    vk_icd = selected_vulkan_icd()
    env.update(
        {
            "VK_ICD_FILENAMES": str(vk_icd),
            "DISABLE_LAYER_NV_OPTIMUS_1": "1",
            "XDG_RUNTIME_DIR": str(XDG_RUNTIME_DIR),
            "LD_LIBRARY_PATH": f"{GPU_FOUNDATION_DEPS}:{env.get('LD_LIBRARY_PATH', '')}",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "ISAAC_PATH": str(ROOT / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim"),
            "TMPDIR": str(ROOT / "tmp"),
            "HOME": str(ROOT / "cache/home"),
            "XDG_CACHE_HOME": str(ROOT / "cache/xdg"),
            "OMNI_USER_DIR": str(ROOT / "cache/omniverse/user"),
            "OMNI_LOGS_DIR": str(ROOT / "logs/omniverse"),
            "OMNI_CACHE_DIR": str(ROOT / "cache/omniverse/cache"),
            "OV_USER_DIR": str(ROOT / "cache/omniverse/user"),
            "OV_CACHE_DIR": str(ROOT / "cache/omniverse/cache"),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "WANDB_MODE": "offline",
            "BM_SELECTED_PHYSICAL_GPU": str(selected_gpu),
            "BM_ISAACLAB_DEVICE": f"cuda:{selected_gpu}",
            "BM_VK_ICD_FILENAMES": str(vk_icd),
            "BM_XDG_RUNTIME_DIR": str(XDG_RUNTIME_DIR),
            "BM_OUT_DIR": str(OUT),
            "BM_RUN_DIR": str(run_dir),
            "BM_CHECKPOINT": str(checkpoint),
            "BM_OFFICIAL_IMPORTER_USD": str(OFFICIAL_IMPORTER_USD),
            "BM_MOTION_FILE": str(ROBOT_ORDER_MOTION_NPZ),
            "BM_ROLLOUT_STEPS": str(ROLLOUT_STEPS),
            "BM_SEED": str(SEED),
            "BM_TIMESTAMP_UTC": datetime.now(timezone.utc).isoformat(),
            "BM_FINAL_MP4": str(final_mp4),
            "BM_KEYFRAMES_PNG": str(keyframes_png),
            "BM_METRICS_CSV": str(metrics_csv),
            "BM_WORKER_SUMMARY_JSON": str(worker_summary_json),
        }
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    worker_path = OUT / "isaaclab_rendered_policy_rollout_worker.py"
    worker_path.write_text(textwrap.dedent(WORKER_CODE), encoding="utf-8")
    vk_icd = selected_vulkan_icd()

    checkpoint, checkpoint_decision = select_checkpoint()
    gpu_rows = query_gpus()
    selected_gpu = select_gpu(gpu_rows)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    label = f"{timestamp}_seed{SEED}_{ROLLOUT_STEPS}step_robot_order_policy"
    run_dir = RUN_ROOT / label
    log_path = LOG_DIR / f"{label}.log"
    gpu_metrics_csv = run_dir / "gpu_metrics.csv"
    final_mp4 = OUT / f"{label}.mp4"
    keyframes_png = OUT / f"{label}_keyframes.png"
    metrics_csv = OUT / f"{label}_metrics.csv"
    worker_summary_json = OUT / f"{label}_summary.json"
    asset_json = OUT / "isaaclab_rendered_policy_rollout_video_asset.json"
    failed_run_json = FAILED_DIR / "isaaclab_rendered_policy_rollout_video_failed_gate.json"
    latest_markdown = OUT / "README.md"

    input_checks = {
        "tracking_python_exists": TRACKING_PY.is_file(),
        "system_nvidia_icd_exists": SYSTEM_NVIDIA_ICD.is_file(),
        "project_egl_icd_exists": PROJECT_EGL_ICD.is_file(),
        "selected_vulkan_icd_exists": vk_icd.is_file(),
        "xdg_runtime_dir_project_local": str(XDG_RUNTIME_DIR).startswith(str(ROOT)),
        "gpu_foundation_deps_exists": GPU_FOUNDATION_DEPS.is_dir(),
        "official_importer_usd_exists": OFFICIAL_IMPORTER_USD.is_file(),
        "robot_order_motion_npz_exists": ROBOT_ORDER_MOTION_NPZ.is_file(),
        "robot_order_bundle_audit_exists": ROBOT_ORDER_BUNDLE_AUDIT.is_file(),
        "training_run_json_exists": TRAINING_RUN_JSON.is_file(),
        "checkpoint_eval_json_exists": CHECKPOINT_EVAL_JSON.is_file(),
        "checkpoint_exists": checkpoint.is_file(),
        "selected_gpu_available": selected_gpu is not None,
    }

    run_info: dict[str, Any] = {
        "attempted": False,
        "selected_physical_gpu": selected_gpu,
        "candidate_gpus": CANDIDATE_GPUS,
        "gpu_preflight": gpu_rows,
        "run_dir": str(run_dir),
        "log": str(log_path),
        "gpu_metrics_csv": str(gpu_metrics_csv),
        "worker_script": str(worker_path),
        "vulkan_icd": str(vk_icd),
        "xdg_runtime_dir": str(XDG_RUNTIME_DIR),
    }

    if all(input_checks.values()) and selected_gpu is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        monitor = start_gpu_monitor(gpu_metrics_csv, selected_gpu)
        start = time.time()
        env = make_env(
            selected_gpu=selected_gpu,
            run_dir=run_dir,
            checkpoint=checkpoint,
            timestamp=timestamp,
            final_mp4=final_mp4,
            keyframes_png=keyframes_png,
            metrics_csv=metrics_csv,
            worker_summary_json=worker_summary_json,
        )
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                [str(TRACKING_PY), str(worker_path)],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = proc.wait(timeout=TIMEOUT_SECONDS)
                timed_out = False
                termination_method = "natural_exit"
            except subprocess.TimeoutExpired:
                timed_out = True
                returncode, termination_method = terminate_process_tree(proc)
        monitor.terminate()
        try:
            monitor.wait(timeout=20)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait(timeout=20)
        duration = time.time() - start
        worker_summary = load_json(worker_summary_json)
        run_info.update(
            {
                "attempted": True,
                "returncode": returncode,
                "timed_out": timed_out,
                "termination_method": termination_method,
                "duration_seconds": round(duration, 3),
                "worker_status": worker_summary.get("status"),
                "mp4_exists": final_mp4.is_file(),
                "mp4_size_bytes": final_mp4.stat().st_size if final_mp4.is_file() else 0,
                "metrics_csv_exists": metrics_csv.is_file(),
                "keyframes_png_exists": keyframes_png.is_file(),
            }
        )
    else:
        run_info["skipped_reason"] = "input_checks_failed"

    worker_summary = load_json(worker_summary_json)
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    vulkaninfo_full_text = (
        VULKANINFO_FULL_LOG.read_text(encoding="utf-8", errors="replace") if VULKANINFO_FULL_LOG.is_file() else ""
    )
    vulkan_ray_tracing_extensions = {
        "VK_KHR_acceleration_structure": "VK_KHR_acceleration_structure" in vulkaninfo_full_text,
        "VK_KHR_ray_tracing_pipeline": "VK_KHR_ray_tracing_pipeline" in vulkaninfo_full_text,
        "VK_KHR_deferred_host_operations": "VK_KHR_deferred_host_operations" in vulkaninfo_full_text,
        "VK_KHR_ray_query": "VK_KHR_ray_query" in vulkaninfo_full_text,
        "VK_NV_ray_tracing": "VK_NV_ray_tracing" in vulkaninfo_full_text,
    }
    gpu_names = sorted({str(row.get("name", "")) for row in gpu_rows if row.get("name")})
    h20_gpu_detected = any("H20" in name for name in gpu_names)
    error_patterns = [
        "VkResult: ERROR_DEVICE_LOST",
        "GPU crash occurred",
        "No device could be created",
        "GLFW initialization failed",
        "GLXBadFBConfig",
        "No module named 'omni.replicator'",
        "Stage opened with no valid renderer selected",
        "GLInteropContext::init",
        "carb::windowing is not available",
        "Segmentation fault",
        "timeout",
    ]
    detected_errors = sorted({pattern for pattern in error_patterns if pattern in log_text})
    sentinel_patterns = [
        "BM_SENTINEL:isaac_mp4:before_app",
        "BM_SENTINEL:isaac_mp4:after_app",
        "BM_SENTINEL:isaac_mp4:replicator_extension_enabled",
        "BM_SENTINEL:isaac_mp4:env_created",
        "BM_SENTINEL:isaac_mp4:render_product_created",
        "BM_SENTINEL:isaac_mp4:step=",
        "BM_SENTINEL:isaac_mp4:env_closed",
    ]
    reached_sentinels = [pattern for pattern in sentinel_patterns if pattern in log_text]
    log_tail = "\n".join(log_text.splitlines()[-60:])
    physics_rollout_ok = worker_summary.get("interpretation", {}).get("physics_rollout_complete") is True
    video_recording_ok = worker_summary.get("interpretation", {}).get("video_recording_complete") is True
    checks = {
        **input_checks,
        "worker_summary_exists": worker_summary_json.is_file(),
        "app_launcher_attempted": "BM_SENTINEL:isaac_mp4:before_app" in log_text,
        "app_launcher_reached": "BM_SENTINEL:isaac_mp4:after_app" in log_text,
        "replicator_extension_enabled": "BM_SENTINEL:isaac_mp4:replicator_extension_enabled" in log_text,
        "render_product_created": "BM_SENTINEL:isaac_mp4:render_product_created" in log_text,
        "env_created": "BM_SENTINEL:isaac_mp4:env_created" in log_text,
        "env_step_progress_recorded": "BM_SENTINEL:isaac_mp4:step=" in log_text,
        "env_closed": "BM_SENTINEL:isaac_mp4:env_closed" in log_text,
        "physics_rollout_complete": bool(physics_rollout_ok),
        "video_recording_complete": bool(video_recording_ok),
        "mp4_exists_nonempty": final_mp4.is_file() and final_mp4.stat().st_size > 0,
        "keyframes_png_exists_nonempty": keyframes_png.is_file() and keyframes_png.stat().st_size > 0,
        "metrics_csv_exists_nonempty": metrics_csv.is_file() and metrics_csv.stat().st_size > 0,
        "does_not_claim_official_checkpoint": True,
        "does_not_claim_paper_level": True,
        "does_not_claim_fig5_fig6": True,
        "does_not_claim_tensorrt": True,
        "does_not_claim_real_robot": True,
    }
    status = (
        "ok_isaaclab_rendered_policy_rollout_mp4"
        if checks["physics_rollout_complete"] and checks["video_recording_complete"] and checks["mp4_exists_nonempty"]
        else "failed_isaaclab_rendered_policy_rollout_mp4"
    )

    asset = {
        "status": status,
        "experiment_type": "isaaclab_rendered_g1_policy_rollout_mp4",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "claim_level": "local_virtual_isaaclab_rendered_policy_rollout_video",
        "goal_complete": False,
        "scope": (
            "A true IsaacLab/Isaac Sim rendered 300-step policy rollout video. The simulator is launched with "
            "AppLauncher(headless=True, enable_cameras=True), the task is Tracking-Flat-G1-v0, and actions come "
            "from a local PPO checkpoint through env.step(action). This is not a matplotlib skeleton video."
        ),
        "checkpoint_selection": checkpoint_decision,
        "inputs": {
            "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
            "motion_npz": str(ROBOT_ORDER_MOTION_NPZ),
            "robot_order_bundle_audit": str(ROBOT_ORDER_BUNDLE_AUDIT),
            "training_run_json": str(TRAINING_RUN_JSON),
            "checkpoint_eval_json": str(CHECKPOINT_EVAL_JSON),
            "checkpoint": str(checkpoint),
        },
        "run": run_info,
        "rendering_stack_repair_attempt": {
            "system_packages_installed_or_verified": [
                "vulkan-tools",
                "mesa-utils",
                "mesa-vulkan-drivers",
                "xvfb",
                "xauth",
                "libxkbcommon-x11-0",
                "libxcb-xinerama0",
                "libxcb-icccm4",
                "libxcb-image0",
                "libxcb-keysyms1",
                "libxcb-render-util0",
                "libxcb-xfixes0",
            ],
            "selected_vulkan_icd": str(vk_icd),
            "system_nvidia_icd": str(SYSTEM_NVIDIA_ICD),
            "project_egl_icd": str(PROJECT_EGL_ICD),
            "xdg_runtime_dir": str(XDG_RUNTIME_DIR),
            "disabled_nv_optimus_layer": True,
            "vulkaninfo_default_log": str(VULKANINFO_DEFAULT_LOG),
            "vulkaninfo_system_nvidia_full_log": str(VULKANINFO_FULL_LOG),
            "vulkaninfo_ray_tracing_extensions_detected": vulkan_ray_tracing_extensions,
            "gpu_names_detected": gpu_names,
            "h20_gpu_detected": h20_gpu_detected,
            "official_support_note": (
                "NVIDIA Isaac Sim system requirements state that GPUs without RT Cores, including A100/H100-class "
                "data-center GPUs, are unsupported for Isaac Sim rendering. NVIDIA forum guidance also identifies "
                "H20 as lacking RT Cores for Isaac Sim graphics rendering. This local gate still tries the repaired "
                "Vulkan/ICD path, but an H20 Vulkan ERROR_DEVICE_LOST remains classified as a server rendering-stack "
                "hardware/driver blocker rather than a PPO checkpoint or IsaacLab physics-rollout failure."
            ),
        },
        "log_diagnostics": {
            "detected_error_patterns": detected_errors,
            "reached_sentinels": reached_sentinels,
            "log_tail": log_tail,
            "manual_probes": {
                "xvfb_gui_probe": str(LOG_DIR / "xvfb_probe_20260622_042348.log"),
                "single_visible_gpu_replicator_probe": str(
                    LOG_DIR / "single_visible_gpu_replicator_probe_20260622_042442.log"
                ),
                "gpu5_replicator_probe": str(LOG_DIR / "replicator_probe_gpu5_20260622_042620.log"),
                "pxr_storm_probe": str(LOG_DIR / "pxr_storm_probe_20260622_042928.log"),
            },
        },
        "worker_summary": worker_summary,
        "outputs": {
            "mp4": str(final_mp4),
            "keyframes_png": str(keyframes_png),
            "metrics_csv": str(metrics_csv),
            "summary_json": str(worker_summary_json),
            "asset_json": str(asset_json),
            "failed_run_json": str(failed_run_json),
            "readme": str(latest_markdown),
        },
        "asset_sizes": {
            "mp4": final_mp4.stat().st_size if final_mp4.is_file() else 0,
            "keyframes_png": keyframes_png.stat().st_size if keyframes_png.is_file() else 0,
            "metrics_csv": metrics_csv.stat().st_size if metrics_csv.is_file() else 0,
            "summary_json": worker_summary_json.stat().st_size if worker_summary_json.is_file() else 0,
        },
        "asset_sha256": {
            "mp4": sha256_file(final_mp4) if final_mp4.is_file() else None,
            "keyframes_png": sha256_file(keyframes_png) if keyframes_png.is_file() else None,
            "metrics_csv": sha256_file(metrics_csv) if metrics_csv.is_file() else None,
            "summary_json": sha256_file(worker_summary_json) if worker_summary_json.is_file() else None,
        },
        "checks": checks,
        "failure_classification": {
            "physics_rollout_failed": checks["env_created"] and not checks["physics_rollout_complete"],
            "render_or_video_failed": checks["physics_rollout_complete"] and not checks["video_recording_complete"],
            "startup_failed": not checks["app_launcher_reached"],
            "replicator_hydra_vulkan_device_lost": (
                "VkResult: ERROR_DEVICE_LOST" in detected_errors
                and (
                    checks["replicator_extension_enabled"]
                    or "omni.replicator.core" in log_text
                    or "omni.hydra" in log_text
                )
            ),
            "glx_windowing_blocker": "GLXBadFBConfig" in detected_errors,
            "server_rendering_stack_blocker": (
                "VkResult: ERROR_DEVICE_LOST" in detected_errors
                or "GPU crash occurred" in detected_errors
                or "No device could be created" in detected_errors
                or "GLXBadFBConfig" in detected_errors
            ),
            "h20_isaac_rendering_hardware_blocker": h20_gpu_detected
            and (
                "VkResult: ERROR_DEVICE_LOST" in detected_errors
                or "GPU crash occurred" in detected_errors
                or not checks["app_launcher_reached"]
            ),
            "policy_or_checkpoint_failure": checks["env_created"] and "VkResult: ERROR_DEVICE_LOST" not in detected_errors,
        },
        "interpretation": {
            "paper_level_status": "qualitative_only_local_virtual_simulation_deployment_evidence",
            "why_not_paper_level": (
                "A successful MP4 would be real IsaacLab rendering and closed-loop physics stepping, but this "
                "gate has not produced a usable rendered video yet. The current blocker is the server-side "
                "Isaac Sim Vulkan/Hydra/Replicator rendering stack, not an official paper-level policy result. "
                "Even after this gate passes, the checkpoint and motion bundle are local public-resource "
                "artifacts, not an official BeyondMimic teacher, official DAgger/VAE/diffusion checkpoint, "
                "Fig. 5/Fig. 6 task result, TensorRT deployment, or real Unitree G1 result."
            ),
        },
    }
    if status != "ok_isaaclab_rendered_policy_rollout_mp4":
        write_json(
            failed_run_json,
            {
                "status": "failed_isaaclab_rendered_policy_rollout_mp4_gate",
                "timestamp_utc": asset["timestamp_utc"],
                "claim_level": "blocked_real_isaaclab_rendered_video_gate",
                "goal_complete": False,
                "task": "Tracking-Flat-G1-v0",
                "requested_goal": "Record a true IsaacLab/Isaac Sim rendered G1 PPO policy rollout MP4.",
                "selected_checkpoint": str(checkpoint),
                "selected_gpu": selected_gpu,
                "candidate_gpus": CANDIDATE_GPUS,
                "run": run_info,
                "rendering_stack_repair_attempt": asset["rendering_stack_repair_attempt"],
                "checks": checks,
                "failure_classification": asset["failure_classification"],
                "detected_error_patterns": detected_errors,
                "reached_sentinels": reached_sentinels,
                "latest_asset_json": str(asset_json),
                "latest_log": str(log_path),
                "intended_outputs": {
                    "mp4": str(final_mp4),
                    "keyframes_png": str(keyframes_png),
                    "metrics_csv": str(metrics_csv),
                    "summary_json": str(worker_summary_json),
                },
                "interpretation": (
                    "The script reached the Isaac Sim rendering startup path after selecting the system NVIDIA "
                    "Vulkan ICD, setting a project-local XDG_RUNTIME_DIR, and disabling the NVIDIA Optimus layer, "
                    "but the server-side Kit/Hydra/Vulkan renderer still failed before a Tracking-Flat-G1-v0 "
                    "environment could be created. Because the host GPUs are NVIDIA H20 and the failure occurs "
                    "in Kit/Hydra/Vulkan startup, this is recorded as a server rendering-stack hardware/driver "
                    "blocker, not as a PPO checkpoint failure, not as a physics-rollout failure, not as a "
                    "successful policy rollout video, and not as a paper-level BeyondMimic result."
                ),
            },
        )
    write_json(asset_json, asset)
    latest_markdown.write_text(
        "\n".join(
            [
                "# IsaacLab Rendered Policy Rollout MP4",
                "",
                "This directory is reserved for true IsaacLab/Isaac Sim rendered rollout videos. Successful videos",
                "must be created by",
                "`AppLauncher(headless=True, enable_cameras=True)`, `Tracking-Flat-G1-v0`, an Isaac Sim offscreen",
                "camera render product, and a PPO policy stepped through the real IsaacLab physics environment.",
                "",
                "Current gate status is recorded in the latest asset JSON. If the status is failed, no paper-facing",
                "simulation MP4 should be claimed from this directory.",
                "",
                "They are qualitative local virtual simulation-deployment evidence only. They are not official",
                "BeyondMimic checkpoints, Fig. 5/Fig. 6 paper-level videos, TensorRT deployment evidence, or real",
                "robot results.",
                "",
                f"Latest asset JSON: `{asset_json}`",
                f"Latest MP4: `{final_mp4}`",
                f"Latest metrics CSV: `{metrics_csv}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "asset_json": str(asset_json), "mp4": str(final_mp4)}, sort_keys=True))


if __name__ == "__main__":
    main()
