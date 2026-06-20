
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

print(f"BM_SENTINEL:official_importer_vae_video:before_app:device={args.device}", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("BM_SENTINEL:official_importer_vae_video:after_app", flush=True)

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
    print("BM_SENTINEL:official_importer_vae_video:env_created", flush=True)
    vec_env = RslRlVecEnvWrapper(env)
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
    teacher_action_abs_mean = []
    teacher_vae_action_mse = []
    teacher_vae_action_abs_error_mean = []
    latent_mu_abs_mean = []
    motion_time_steps = []
    metric_series = {}

    metric_names = ["error_anchor_pos", "error_body_pos", "error_joint_pos", "sampling_top1_prob"]
    with torch.inference_mode():
        for step in range(rollout_steps):
            teacher_actions = policy(obs)
            mu, _logvar = vae.encode(obs, teacher_actions)
            actions = vae.decode(obs, mu)
            action_delta = actions - teacher_actions
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
            teacher_action_abs_mean.append(float(teacher_actions.abs().mean().detach().cpu()))
            teacher_vae_action_mse.append(float(torch.mean(action_delta.square()).detach().cpu()))
            teacher_vae_action_abs_error_mean.append(float(torch.mean(action_delta.abs()).detach().cpu()))
            latent_mu_abs_mean.append(float(mu.abs().mean().detach().cpu()))
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
                print(f"BM_SENTINEL:official_importer_vae_video:step={step + 1}/{rollout_steps}", flush=True)

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
        teacher_action_abs_mean=np.asarray(teacher_action_abs_mean, dtype=np.float32),
        teacher_vae_action_mse=np.asarray(teacher_vae_action_mse, dtype=np.float32),
        teacher_vae_action_abs_error_mean=np.asarray(teacher_vae_action_abs_error_mean, dtype=np.float32),
        latent_mu_abs_mean=np.asarray(latent_mu_abs_mean, dtype=np.float32),
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
        "vae_checkpoint": str(vae_checkpoint),
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
        "teacher_action_abs_mean": summarize(teacher_action_abs_mean),
        "teacher_vae_action_mse": summarize(teacher_vae_action_mse),
        "teacher_vae_action_abs_error_mean": summarize(teacher_vae_action_abs_error_mean),
        "latent_mu_abs_mean": summarize(latent_mu_abs_mean),
        "motion_time_step_min": int(np.min(motion_time_steps)),
        "motion_time_step_max": int(np.max(motion_time_steps)),
        "motion_metrics": {name: summarize(values) for name, values in metric_series.items()},
        "uses_resource_adjusted_usd": False,
        "official_csv_loop_motion": True,
        "uses_official_importer_export_usd": True,
        "official_beyondmimic_vae_checkpoint": False,
        "paper_level_vae_closed_loop": False,
        "paper_level_guided_diffusion": False,
        "real_robot": False,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    vec_env.close()
    print(f"BM_SENTINEL:official_importer_vae_video:npz={out_npz}", flush=True)
    print(f"BM_SENTINEL:official_importer_vae_video:metrics={metrics_path}", flush=True)
    os._exit(0)
except BaseException as exc:
    print(f"BM_SENTINEL:official_importer_vae_video:exception={exc!r}", flush=True)
    raise
finally:
    try:
        simulation_app.close(wait_for_replicator=False)
    except BaseException:
        pass
