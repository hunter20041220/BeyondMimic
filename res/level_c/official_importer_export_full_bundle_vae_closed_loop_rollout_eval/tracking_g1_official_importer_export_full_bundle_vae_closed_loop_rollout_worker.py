
import argparse
import csv
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

print(f"BM_SENTINEL:official_importer_vae_closed_loop:before_app:rank={rank}:device={args.device}", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"BM_SENTINEL:official_importer_vae_closed_loop:after_app:rank={rank}", flush=True)

try:
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import numpy as np
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
    run_dir = Path(os.environ["BM_RUN_DIR"])
    rollout_steps = int(os.environ["BM_ROLLOUT_STEPS"])
    shard_dir = run_dir / f"rank_{rank}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_metrics_path = shard_dir / "vae_closed_loop_rollout_metrics.json"
    shard_timeseries_path = shard_dir / "vae_closed_loop_rollout_timeseries.csv"
    shard_npz = shard_dir / "vae_closed_loop_rollout_summary_arrays.npz"

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
    env_cfg.seed = int(os.environ["BM_SEED"]) + rank

    agent_cfg = G1FlatPPORunnerCfg()
    agent_cfg.device = args.device
    agent_cfg.seed = int(os.environ["BM_SEED"]) + rank
    agent_cfg.empirical_normalization = True
    agent_cfg.logger = "tensorboard"

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print(f"BM_SENTINEL:official_importer_vae_closed_loop:env_created:rank={rank}:num_envs={env.unwrapped.num_envs}", flush=True)
    vec_env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    loaded_infos = runner.load(str(checkpoint))
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

    obs, extras = vec_env.get_observations()
    command = vec_env.unwrapped.command_manager.get_term("motion")
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
        "reward_mean",
        "reward_min",
        "reward_max",
        "done_count",
        "timeout_count",
        "teacher_action_abs_mean",
        "vae_action_abs_mean",
        "teacher_vae_action_mse",
        "teacher_vae_action_abs_error_mean",
        "latent_mu_abs_mean",
    ] + metric_names

    rows = []
    reward_means = []
    done_counts = []
    timeout_counts = []
    teacher_vae_mse = []
    teacher_vae_abs = []
    latent_abs = []
    teacher_action_abs = []
    vae_action_abs = []
    metric_series = {name: [] for name in metric_names}

    with shard_timeseries_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        with torch.no_grad():
            for step in range(rollout_steps):
                teacher_action = policy(obs)
                mu, logvar = vae.encode(obs, teacher_action)
                vae_action = vae.decode(obs, mu)
                action_delta = vae_action - teacher_action
                obs, rew, dones, step_extras = vec_env.step(vae_action)
                row = {
                    "step": step,
                    "reward_mean": float(rew.mean().detach().cpu()),
                    "reward_min": float(rew.min().detach().cpu()),
                    "reward_max": float(rew.max().detach().cpu()),
                    "done_count": int(dones.sum().detach().cpu()),
                    "timeout_count": int(step_extras.get("time_outs", torch.zeros_like(dones)).sum().detach().cpu()),
                    "teacher_action_abs_mean": float(teacher_action.abs().mean().detach().cpu()),
                    "vae_action_abs_mean": float(vae_action.abs().mean().detach().cpu()),
                    "teacher_vae_action_mse": float(torch.mean(action_delta.square()).detach().cpu()),
                    "teacher_vae_action_abs_error_mean": float(torch.mean(action_delta.abs()).detach().cpu()),
                    "latent_mu_abs_mean": float(mu.abs().mean().detach().cpu()),
                }
                for name in metric_names:
                    tensor = command.metrics.get(name)
                    value = float(tensor.mean().detach().cpu()) if tensor is not None else float("nan")
                    row[name] = value
                    metric_series[name].append(value)
                writer.writerow(row)
                rows.append(row)
                reward_means.append(row["reward_mean"])
                done_counts.append(row["done_count"])
                timeout_counts.append(row["timeout_count"])
                teacher_vae_mse.append(row["teacher_vae_action_mse"])
                teacher_vae_abs.append(row["teacher_vae_action_abs_error_mean"])
                latent_abs.append(row["latent_mu_abs_mean"])
                teacher_action_abs.append(row["teacher_action_abs_mean"])
                vae_action_abs.append(row["vae_action_abs_mean"])
                if (step + 1) % 50 == 0 or (step + 1) == rollout_steps:
                    print(f"BM_SENTINEL:official_importer_vae_closed_loop:rank={rank}:step={step + 1}/{rollout_steps}", flush=True)

    def summarize(values):
        values = list(values)
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "first": float(values[0]),
            "last": float(values[-1]),
            "mean": float(sum(values) / len(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }

    np.savez_compressed(
        shard_npz,
        reward_mean=np.asarray(reward_means, dtype=np.float32),
        done_count=np.asarray(done_counts, dtype=np.int32),
        timeout_count=np.asarray(timeout_counts, dtype=np.int32),
        teacher_vae_action_mse=np.asarray(teacher_vae_mse, dtype=np.float32),
        teacher_vae_action_abs_error_mean=np.asarray(teacher_vae_abs, dtype=np.float32),
        latent_mu_abs_mean=np.asarray(latent_abs, dtype=np.float32),
        teacher_action_abs_mean=np.asarray(teacher_action_abs, dtype=np.float32),
        vae_action_abs_mean=np.asarray(vae_action_abs, dtype=np.float32),
    )
    metrics = {
        "status": "ok",
        "rank": rank,
        "world_size": world_size,
        "task": "Tracking-Flat-G1-v0",
        "checkpoint": str(checkpoint),
        "vae_checkpoint": str(vae_checkpoint),
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
        "robot_num_joints": int(vec_env.unwrapped.scene["robot"].num_joints),
        "robot_num_bodies": int(vec_env.unwrapped.scene["robot"].num_bodies),
        "vae_config": vae_cfg,
        "reward_mean": summarize(reward_means),
        "done_count_total": int(sum(done_counts)),
        "timeout_count_total": int(sum(timeout_counts)),
        "teacher_vae_action_mse": summarize(teacher_vae_mse),
        "teacher_vae_action_abs_error_mean": summarize(teacher_vae_abs),
        "teacher_action_abs_mean": summarize(teacher_action_abs),
        "vae_action_abs_mean": summarize(vae_action_abs),
        "latent_mu_abs_mean": summarize(latent_abs),
        "motion_metrics": {name: summarize(values) for name, values in metric_series.items()},
        "timeseries_csv": str(shard_timeseries_path),
        "summary_npz": str(shard_npz),
        "summary_npz_size_bytes": shard_npz.stat().st_size,
        "uses_resource_adjusted_usd": False,
        "official_csv_loop_motion": True,
        "uses_official_importer_export_usd": True,
        "official_beyondmimic_vae_checkpoint": False,
        "paper_level_vae_closed_loop": False,
        "official_beyondmimic_vae_checkpoint": False,
        "paper_level_guided_diffusion": False,
        "real_robot": False,
    }
    shard_metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    vec_env.close()
    print(f"BM_SENTINEL:official_importer_vae_closed_loop:metrics_written:rank={rank}:{shard_metrics_path}", flush=True)
    os._exit(0)
except BaseException as exc:
    print(f"BM_SENTINEL:official_importer_vae_closed_loop:exception:rank={rank}:{exc!r}", flush=True)
    raise
finally:
    try:
        simulation_app.close(wait_for_replicator=False)
    except BaseException:
        pass
