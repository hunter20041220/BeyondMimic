
import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
rank = int(os.environ.get("RANK", str(local_rank)))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
device = f"cuda:{local_rank}"

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
args.device = device

print(f"BM_SENTINEL:teacher_rollout:before_app:rank={rank}:device={args.device}", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"BM_SENTINEL:teacher_rollout:after_app:rank={rank}", flush=True)

try:
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import numpy as np
    import torch
    import whole_body_tracking.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

    enriched_usd = Path(os.environ["BM_ENRICHED_USD"])
    motion_file = Path(os.environ["BM_MOTION_FILE"])
    checkpoint = Path(os.environ["BM_CHECKPOINT"])
    run_dir = Path(os.environ["BM_RUN_DIR"])
    shard_dir = run_dir / f"rank_{rank}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_npz = shard_dir / "teacher_rollout_shard.npz"
    shard_metrics_path = shard_dir / "teacher_rollout_shard_metrics.json"

    env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = int(os.environ["BM_NUM_ENVS_PER_RANK"])
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
    env_cfg.seed = int(os.environ["BM_TEACHER_ROLLOUT_SEED"]) + rank

    agent_cfg = G1FlatPPORunnerCfg()
    agent_cfg.device = args.device
    agent_cfg.seed = int(os.environ["BM_TEACHER_ROLLOUT_SEED"]) + rank
    agent_cfg.empirical_normalization = True
    agent_cfg.logger = "tensorboard"

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print(f"BM_SENTINEL:teacher_rollout:env_created:rank={rank}:num_envs={env.unwrapped.num_envs}", flush=True)
    vec_env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    loaded_infos = runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
    obs, extras = vec_env.get_observations()
    command = vec_env.unwrapped.command_manager.get_term("motion")
    rollout_steps = int(os.environ["BM_TEACHER_ROLLOUT_STEPS"])

    policy_obs = []
    critic_obs = []
    actions_out = []
    rewards_out = []
    dones_out = []
    timeouts_out = []
    motion_time_steps = []
    metric_series = {}
    episode_log_accum = {}
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

    with torch.inference_mode():
        for step in range(rollout_steps):
            policy_obs.append(obs.detach().cpu().numpy().astype(np.float32))
            critic = extras["observations"]["critic"]
            critic_obs.append(critic.detach().cpu().numpy().astype(np.float32))
            if hasattr(command, "time_steps"):
                motion_time_steps.append(command.time_steps.detach().cpu().numpy().astype(np.int32))
            else:
                motion_time_steps.append(np.zeros((vec_env.num_envs,), dtype=np.int32))

            actions = policy(obs)
            actions_out.append(actions.detach().cpu().numpy().astype(np.float32))
            obs, rew, dones, step_extras = vec_env.step(actions)
            extras = step_extras
            rewards_out.append(rew.detach().cpu().numpy().astype(np.float32))
            dones_out.append(dones.detach().cpu().numpy().astype(np.bool_))
            timeout_tensor = step_extras.get("time_outs", torch.zeros_like(dones))
            timeouts_out.append(timeout_tensor.detach().cpu().numpy().astype(np.bool_))
            for name in metric_names:
                tensor = command.metrics.get(name)
                if tensor is None:
                    value = float("nan")
                else:
                    value = float(tensor.mean().detach().cpu())
                metric_series.setdefault(name, []).append(value)
            for key, value in step_extras.get("log", {}).items():
                try:
                    scalar = float(value.detach().mean().cpu()) if hasattr(value, "detach") else float(value)
                except Exception:
                    continue
                episode_log_accum.setdefault(key, []).append(scalar)

    policy_obs_arr = np.stack(policy_obs, axis=0)
    critic_obs_arr = np.stack(critic_obs, axis=0)
    actions_arr = np.stack(actions_out, axis=0)
    rewards_arr = np.stack(rewards_out, axis=0)
    dones_arr = np.stack(dones_out, axis=0)
    timeouts_arr = np.stack(timeouts_out, axis=0)
    motion_time_steps_arr = np.stack(motion_time_steps, axis=0)
    final_obs, final_extras = vec_env.get_observations()
    np.savez_compressed(
        shard_npz,
        policy_obs=policy_obs_arr,
        critic_obs=critic_obs_arr,
        actions=actions_arr,
        rewards=rewards_arr,
        dones=dones_arr,
        timeouts=timeouts_arr,
        motion_time_steps=motion_time_steps_arr,
        final_policy_obs=final_obs.detach().cpu().numpy().astype(np.float32),
        final_critic_obs=final_extras["observations"]["critic"].detach().cpu().numpy().astype(np.float32),
        rank=np.array([rank], dtype=np.int32),
        world_size=np.array([world_size], dtype=np.int32),
        seed=np.array([int(os.environ["BM_TEACHER_ROLLOUT_SEED"]) + rank], dtype=np.int32),
    )

    def summarize(values):
        values = list(values)
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "first": values[0],
            "last": values[-1],
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    metrics = {
        "status": "ok",
        "rank": rank,
        "world_size": world_size,
        "task": "Tracking-Flat-G1-v0",
        "checkpoint": str(checkpoint),
        "loaded_iteration": int(runner.current_learning_iteration),
        "loaded_infos_type": type(loaded_infos).__name__,
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "num_envs": int(vec_env.num_envs),
        "rollout_steps": rollout_steps,
        "total_env_steps": int(vec_env.num_envs) * rollout_steps,
        "num_actions": int(vec_env.num_actions),
        "num_obs": int(vec_env.num_obs),
        "num_privileged_obs": int(vec_env.num_privileged_obs),
        "policy_obs_shape": list(policy_obs_arr.shape),
        "critic_obs_shape": list(critic_obs_arr.shape),
        "actions_shape": list(actions_arr.shape),
        "rewards_shape": list(rewards_arr.shape),
        "dones_shape": list(dones_arr.shape),
        "motion_time_steps_shape": list(motion_time_steps_arr.shape),
        "reward_mean": float(rewards_arr.mean()),
        "reward_min": float(rewards_arr.min()),
        "reward_max": float(rewards_arr.max()),
        "done_count_total": int(dones_arr.sum()),
        "timeout_count_total": int(timeouts_arr.sum()),
        "action_abs_mean": float(np.abs(actions_arr).mean()),
        "action_abs_max": float(np.abs(actions_arr).max()),
        "motion_metrics": {name: summarize(values) for name, values in metric_series.items()},
        "episode_log_metrics": {name: summarize(values) for name, values in episode_log_accum.items()},
        "dataset_npz": str(shard_npz),
        "dataset_npz_size_bytes": shard_npz.stat().st_size,
        "uses_resource_adjusted_usd": True,
        "official_csv_source": True,
        "official_dagger_rollout_dataset": False,
        "paper_level_teacher_rollout_dataset": False,
    }
    shard_metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    vec_env.close()
    print(f"BM_SENTINEL:teacher_rollout:metrics_written:rank={rank}:{shard_metrics_path}", flush=True)
    os._exit(0)
except BaseException as exc:
    print(f"BM_SENTINEL:teacher_rollout:exception:rank={rank}:{exc!r}", flush=True)
    raise
finally:
    try:
        simulation_app.close(wait_for_replicator=False)
    except BaseException:
        pass
