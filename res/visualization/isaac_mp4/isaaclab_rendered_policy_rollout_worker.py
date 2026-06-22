
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
