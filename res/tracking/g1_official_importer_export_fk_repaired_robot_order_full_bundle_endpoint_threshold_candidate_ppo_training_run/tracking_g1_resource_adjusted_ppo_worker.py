
import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
global_rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
args.device = f"cuda:{local_rank}"

print(f"BM_SENTINEL:rank={global_rank}:before_app:device={args.device}", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"BM_SENTINEL:rank={global_rank}:after_app", flush=True)

try:
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import torch
    import whole_body_tracking.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
    from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner

    enriched_usd = Path(os.environ["BM_ENRICHED_USD"])
    motion_file = Path(os.environ["BM_MOTION_FILE"])
    run_dir = Path(os.environ["BM_RUN_DIR"])
    rank_dir = run_dir / f"rank_{global_rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)

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
    env_cfg.seed = int(os.environ["BM_PPO_SEED"]) + global_rank

    agent_cfg = G1FlatPPORunnerCfg()
    agent_cfg.device = args.device
    agent_cfg.seed = int(os.environ["BM_PPO_SEED"]) + global_rank
    agent_cfg.max_iterations = int(os.environ["BM_MAX_ITERATIONS"])
    agent_cfg.num_steps_per_env = int(os.environ["BM_NUM_STEPS_PER_ENV"])
    agent_cfg.save_interval = max(1, min(50, agent_cfg.max_iterations))
    agent_cfg.empirical_normalization = True
    agent_cfg.logger = "tensorboard"
    agent_cfg.run_name = f"resource_adjusted_rank{global_rank}"

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print(f"BM_SENTINEL:rank={global_rank}:env_created:num_envs={env.unwrapped.num_envs}", flush=True)
    ee_body_pos_threshold_patch = None
    ee_body_pos_body_names_patch = None
    ee_cfg_original = None
    if os.environ.get("BM_EE_BODY_POS_TRAIN_THRESHOLD"):
        ee_cfg = env.unwrapped.termination_manager.get_term_cfg("ee_body_pos")
        ee_cfg_original = {
            "threshold": float(ee_cfg.params.get("threshold", 0.25)),
            "body_names": list(ee_cfg.params.get("body_names", [])),
        }
        ee_cfg.params = dict(ee_cfg.params)
        ee_cfg.params["threshold"] = float(os.environ["BM_EE_BODY_POS_TRAIN_THRESHOLD"])
        if os.environ.get("BM_EE_BODY_POS_TRAIN_BODY_NAMES"):
            ee_cfg.params["body_names"] = [
                name.strip()
                for name in os.environ["BM_EE_BODY_POS_TRAIN_BODY_NAMES"].split(",")
                if name.strip()
            ]
        env.unwrapped.termination_manager.set_term_cfg("ee_body_pos", ee_cfg)
        ee_cfg_after = env.unwrapped.termination_manager.get_term_cfg("ee_body_pos")
        ee_body_pos_threshold_patch = float(ee_cfg_after.params.get("threshold"))
        ee_body_pos_body_names_patch = list(ee_cfg_after.params.get("body_names", []))
        print(
            f"BM_SENTINEL:rank={global_rank}:ee_body_pos_threshold_patch={ee_body_pos_threshold_patch}",
            flush=True,
        )
    vec_env = RslRlVecEnvWrapper(env)
    runner = MotionOnPolicyRunner(
        vec_env,
        agent_cfg.to_dict(),
        log_dir=str(rank_dir),
        device=agent_cfg.device,
        registry_name=f"local:{motion_file}",
    )
    print(f"BM_SENTINEL:rank={global_rank}:runner_created", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"BM_SENTINEL:rank={global_rank}:learn_completed", flush=True)

    obs, extras = vec_env.get_observations()
    checkpoints = sorted(str(path) for path in rank_dir.glob("model_*.pt"))
    metrics = {
        "rank": global_rank,
        "world_size": world_size,
        "device": args.device,
        "num_envs": int(vec_env.num_envs),
        "num_actions": int(vec_env.num_actions),
        "num_obs": int(vec_env.num_obs),
        "num_privileged_obs": int(vec_env.num_privileged_obs),
        "num_steps_per_env": int(agent_cfg.num_steps_per_env),
        "max_iterations": int(agent_cfg.max_iterations),
        "current_learning_iteration": int(runner.current_learning_iteration),
        "tot_timesteps": int(runner.tot_timesteps),
        "tot_time": float(runner.tot_time),
        "robot_num_joints": int(vec_env.unwrapped.scene["robot"].num_joints),
        "robot_num_bodies": int(vec_env.unwrapped.scene["robot"].num_bodies),
        "post_train_policy_obs_shape": list(obs.shape),
        "post_train_critic_obs_shape": list(extras["observations"]["critic"].shape),
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "uses_resource_adjusted_usd": True,
        "official_csv_source": True,
        "official_csv_to_npz_output": False,
        "paper_level_training": False,
        "ee_body_pos_threshold_patch_applied": ee_body_pos_threshold_patch is not None,
        "ee_body_pos_original_cfg": ee_cfg_original,
        "ee_body_pos_train_threshold_after": ee_body_pos_threshold_patch,
        "ee_body_pos_train_body_names_after": ee_body_pos_body_names_patch,
    }
    (rank_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    vec_env.close()
    print(f"BM_SENTINEL:rank={global_rank}:metrics_written={rank_dir / 'training_metrics.json'}", flush=True)
    os._exit(0)
except BaseException as exc:
    print(f"BM_SENTINEL:rank={global_rank}:exception={exc!r}", flush=True)
    raise
finally:
    try:
        simulation_app.close(wait_for_replicator=False)
    except BaseException:
        pass
