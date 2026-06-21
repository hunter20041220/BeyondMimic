
import json
import math
import os
from pathlib import Path

OUT = Path(os.environ["BM_TASK_METRICS_JSON"])
ROBOT_USD = Path(os.environ["BM_ROBOT_USD"])
MOTION_FILE = Path(os.environ["BM_MOTION_FILE"])
MAX_STEPS = int(os.environ["BM_MAX_STEPS"])

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
target_gpu = os.environ.get("BM_TARGET_GPU", "4")
args.headless = True
args.enable_cameras = False
args.device = os.environ.get("BM_DEVICE", f"cuda:{target_gpu}")
args.multi_gpu = False
args.kit_args = (
    "--/renderer/multiGpu/enabled=false "
    "--/renderer/multiGpu/autoEnable=false "
    "--/renderer/multiGpu/maxGpuCount=1 "
    f"--/renderer/activeGpu={target_gpu} "
    f"--/physics/cudaDevice={target_gpu}"
)

print("BM_SENTINEL:before_app", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("BM_SENTINEL:after_app", flush=True)

try:
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import torch
    import whole_body_tracking.tasks  # noqa: F401
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

    def tensor_shape(x):
        if hasattr(x, "shape"):
            return list(x.shape)
        if isinstance(x, dict):
            return {k: tensor_shape(v) for k, v in x.items()}
        return str(type(x))

    def summarize(values):
        finite = [float(v) for v in values if math.isfinite(float(v))]
        if not finite:
            return {"count": 0, "mean": None, "min": None, "max": None}
        return {"count": len(finite), "mean": sum(finite) / len(finite), "min": min(finite), "max": max(finite)}

    env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=str(ROBOT_USD),
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
    env_cfg.commands.motion.motion_file = str(MOTION_FILE)
    env_cfg.commands.motion.debug_vis = False
    env_cfg.scene.contact_forces.debug_vis = False
    env_cfg.sim.device = args.device
    env_cfg.episode_length_s = 0.24
    env_cfg.seed = int(os.environ.get("BM_SEED", "20260750"))

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print("BM_SENTINEL:env_created", flush=True)
    obs, extras = env.reset()
    print("BM_SENTINEL:env_reset", flush=True)

    action_dim = int(env.unwrapped.action_manager.total_action_dim)
    action = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
    command = env.unwrapped.command_manager.get_term("motion")
    step_count = min(int(command.motion.time_step_total), MAX_STEPS)
    rewards = []
    terminated_counts = []
    truncated_counts = []
    metric_series = {}
    for i in range(step_count):
        obs, reward, terminated, truncated, extras = env.step(action)
        rewards.append(float(reward.detach().cpu().mean().item()))
        terminated_counts.append(int(terminated.detach().cpu().sum().item()))
        truncated_counts.append(int(truncated.detach().cpu().sum().item()))
        for name, value in command.metrics.items():
            if hasattr(value, "detach") and value.numel() > 0:
                metric_series.setdefault(name, []).append(float(value.detach().cpu().mean().item()))
        if (i + 1) % 50 == 0 or (i + 1) == step_count:
            print(f"BM_SENTINEL:env_step={i + 1}/{step_count}", flush=True)

    command_metrics_final = {
        k: float(v.detach().cpu().mean().item())
        for k, v in command.metrics.items()
        if hasattr(v, "detach") and v.numel() > 0
    }
    metrics = {
        "status": "ok",
        "task": "Tracking-Flat-G1-v0",
        "motion_file": str(MOTION_FILE),
        "num_envs": int(env.unwrapped.num_envs),
        "device": str(env.unwrapped.device),
        "action_dim": action_dim,
        "policy_observation_dim": int(env.unwrapped.observation_manager.group_obs_dim["policy"][0]),
        "critic_observation_dim": int(env.unwrapped.observation_manager.group_obs_dim["critic"][0]),
        "observation_shapes": tensor_shape(obs),
        "action_terms": list(env.unwrapped.action_manager.active_terms),
        "reward_terms": list(env.unwrapped.reward_manager.active_terms),
        "termination_terms": list(env.unwrapped.termination_manager.active_terms),
        "command_terms": list(env.unwrapped.command_manager.active_terms),
        "event_modes": list(env.unwrapped.event_manager.available_modes),
        "step_count": step_count,
        "reward": summarize(rewards),
        "terminated_total": int(sum(terminated_counts)),
        "truncated_total": int(sum(truncated_counts)),
        "done_total": int(sum(terminated_counts) + sum(truncated_counts)),
        "command_metrics_final": command_metrics_final,
        "command_metrics_timeseries": {name: summarize(values) for name, values in metric_series.items()},
        "robot_num_joints": int(env.unwrapped.scene["robot"].num_joints),
        "robot_num_bodies": int(env.unwrapped.scene["robot"].num_bodies),
        "usd_path": str(ROBOT_USD),
        "uses_official_importer_export_usd": True,
        "uses_fk_repaired_robot_order_motion_npz": True,
        "uses_fk_repaired_motion_npz": True,
        "uses_resource_adjusted_usd": False,
        "official_csv_to_npz_unpatched_output": False,
        "paper_level_rollout": False,
        "ppo_training": False,
        "real_robot": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"BM_SENTINEL:metrics_file={OUT}", flush=True)
    print("BM_SENTINEL:task_eval_success", flush=True)
    os._exit(0)
except BaseException as exc:
    print("BM_SENTINEL:exception=" + repr(exc), flush=True)
    raise
finally:
    try:
        simulation_app.close(wait_for_replicator=False)
        print("BM_SENTINEL:after_close", flush=True)
    except BaseException:
        pass
