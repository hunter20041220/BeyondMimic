
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

print(f"BM_SENTINEL:guided_action_probe:before_app:device={args.device}", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("BM_SENTINEL:guided_action_probe:after_app", flush=True)

try:
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import torch
    import whole_body_tracking.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

    enriched_usd = Path(os.environ["BM_ENRICHED_USD"])
    motion_file = Path(os.environ["BM_MOTION_FILE"])
    samples_npz = Path(os.environ["BM_DECODE_SAMPLES_NPZ"])
    out_npz = Path(os.environ["BM_OUT_NPZ"])
    metrics_path = Path(os.environ["BM_METRICS_JSON"])
    sample_index = int(os.environ["BM_SAMPLE_INDEX"])
    task = os.environ["BM_TASK"]
    seed = int(os.environ["BM_SEED"])

    data = np.load(samples_npz)
    variants = {
        "base": data[f"base_action_{task}"][sample_index].astype(np.float32),
        "guided": data[f"guided_action_{task}"][sample_index].astype(np.float32),
        "teacher": data[f"teacher_action_{task}"][sample_index].astype(np.float32),
    }
    rollout_steps = int(variants["base"].shape[0])

    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))

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
    print("BM_SENTINEL:guided_action_probe:env_created", flush=True)
    vec_env = RslRlVecEnvWrapper(env)
    command = vec_env.unwrapped.command_manager.get_term("motion")
    device = vec_env.unwrapped.device

    traces = {}
    rollout_metrics = {}
    metric_names = ["error_anchor_pos", "error_body_pos", "error_joint_pos", "sampling_top1_prob"]
    action_delta = {
        "base_guided_max_abs": float(np.max(np.abs(variants["base"] - variants["guided"]))),
        "base_guided_l2_mean": float(np.linalg.norm((variants["base"] - variants["guided"]).reshape(-1, 29), axis=-1).mean()),
        "base_teacher_mse": float(np.mean((variants["base"] - variants["teacher"]) ** 2)),
        "guided_teacher_mse": float(np.mean((variants["guided"] - variants["teacher"]) ** 2)),
    }

    for variant_name, seq_np in variants.items():
        obs, _ = vec_env.reset()
        robot_body_pos = []
        reference_body_pos = []
        rewards = []
        dones = []
        action_abs_mean = []
        action_abs_max = []
        metric_series = {name: [] for name in metric_names}
        with torch.no_grad():
            for step in range(rollout_steps):
                action = torch.as_tensor(seq_np[step], dtype=torch.float32, device=device).reshape(1, -1)
                obs, rew, done, extras = vec_env.step(action)
                robot_body_pos.append(command.robot_body_pos_w[0].detach().cpu().numpy().astype(np.float32))
                reference_body_pos.append(command.body_pos_relative_w[0].detach().cpu().numpy().astype(np.float32))
                rewards.append(float(rew.detach().cpu().mean()))
                dones.append(int(done.detach().cpu().sum()))
                action_abs_mean.append(float(action.abs().mean().detach().cpu()))
                action_abs_max.append(float(action.abs().max().detach().cpu()))
                for name in metric_names:
                    value = command.metrics.get(name)
                    metric_series[name].append(
                        float(value[0].detach().cpu()) if value is not None and value.numel() > 0 else float("nan")
                    )
        robot_arr = np.stack(robot_body_pos, axis=0)
        ref_arr = np.stack(reference_body_pos, axis=0)
        body_error = np.linalg.norm(robot_arr - ref_arr, axis=-1).mean(axis=1)
        traces[f"{variant_name}_robot_body_pos_w"] = robot_arr
        traces[f"{variant_name}_reference_body_pos_w"] = ref_arr
        traces[f"{variant_name}_rewards"] = np.asarray(rewards, dtype=np.float32)
        traces[f"{variant_name}_dones"] = np.asarray(dones, dtype=np.int32)
        traces[f"{variant_name}_action_abs_mean"] = np.asarray(action_abs_mean, dtype=np.float32)
        traces[f"{variant_name}_target_body_error_mean"] = body_error.astype(np.float32)
        rollout_metrics[variant_name] = {
            "reward_mean": float(np.mean(rewards)),
            "reward_min": float(np.min(rewards)),
            "reward_max": float(np.max(rewards)),
            "done_count_total": int(np.sum(dones)),
            "action_abs_mean": float(np.mean(action_abs_mean)),
            "action_abs_max": float(np.max(action_abs_max)),
            "target_body_error_mean": float(np.mean(body_error)),
            "target_body_error_max": float(np.max(body_error)),
            "motion_metrics": {
                name: {
                    "mean": float(np.nanmean(values)),
                    "min": float(np.nanmin(values)),
                    "max": float(np.nanmax(values)),
                }
                for name, values in metric_series.items()
            },
        }
        print(f"BM_SENTINEL:guided_action_probe:variant={variant_name}:steps={rollout_steps}", flush=True)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        **traces,
        base_actions=variants["base"],
        guided_actions=variants["guided"],
        teacher_actions=variants["teacher"],
    )

    metrics = {
        "status": "ok",
        "task": task,
        "sample_index": sample_index,
        "device": args.device,
        "num_envs": 1,
        "rollout_steps": rollout_steps,
        "variant_count": len(variants),
        "action_delta": action_delta,
        "variant_metrics": rollout_metrics,
        "uses_resource_adjusted_usd": True,
        "official_csv_loop_motion": True,
        "decoded_action_source": str(samples_npz),
        "paper_level_guidance_rollout": False,
        "fig5_fig6_reproduction": False,
        "real_robot": False,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    vec_env.close()
    print(f"BM_SENTINEL:guided_action_probe:npz={out_npz}", flush=True)
    print(f"BM_SENTINEL:guided_action_probe:metrics={metrics_path}", flush=True)
    os._exit(0)
except BaseException as exc:
    print(f"BM_SENTINEL:guided_action_probe:exception={exc!r}", flush=True)
    raise
finally:
    try:
        simulation_app.close(wait_for_replicator=False)
    except BaseException:
        pass
