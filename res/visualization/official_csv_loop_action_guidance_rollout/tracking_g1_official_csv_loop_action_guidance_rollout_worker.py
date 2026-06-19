
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
