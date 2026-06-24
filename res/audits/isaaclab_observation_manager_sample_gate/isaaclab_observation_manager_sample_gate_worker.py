
import faulthandler
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

faulthandler.enable(file=sys.stdout, all_threads=True)

OUT_JSON = Path(os.environ["BM_SAMPLE_JSON"])
OUT_NPZ = Path(os.environ["BM_SAMPLE_NPZ"])
ROBOT_USD = Path(os.environ["BM_ROBOT_USD"])
MOTION_FILE = Path(os.environ["BM_MOTION_FILE"])
TARGET_GPU = os.environ.get("BM_TARGET_GPU", "4")


def to_list(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().tolist()
    return value


def write_stage(stage, **extra):
    payload = {
        "stage": stage,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "robot_usd": str(ROBOT_USD),
        "motion_file": str(MOTION_FILE),
    }
    payload.update(extra)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("BM_SENTINEL:obs_sample:state:" + json.dumps(payload, sort_keys=True), flush=True)


write_stage("before_import")
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
args.device = os.environ.get("BM_DEVICE", f"cuda:{TARGET_GPU}")
args.multi_gpu = False
args.fast_shutdown = True
args.kit_args = (
    "--/renderer/multiGpu/enabled=false "
    "--/renderer/multiGpu/autoEnable=false "
    "--/renderer/multiGpu/maxGpuCount=1 "
    f"--/renderer/activeGpu={TARGET_GPU} "
    f"--/physics/cudaDevice={TARGET_GPU}"
)

write_stage("before_app", target_gpu=TARGET_GPU, device=args.device)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
write_stage("after_app", app_running=bool(simulation_app.is_running()))

try:
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import numpy as np
    import torch
    import whole_body_tracking.tasks  # noqa: F401
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

    write_stage("before_cfg")
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
    env_cfg.seed = int(os.environ.get("BM_SEED", "20260770"))
    write_stage("cfg_ready")

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    unwrapped = env.unwrapped
    write_stage(
        "env_created",
        num_envs=int(unwrapped.num_envs),
        device=str(unwrapped.device),
        action_dim=int(unwrapped.action_manager.total_action_dim),
        robot_num_joints=int(unwrapped.scene["robot"].num_joints),
        robot_num_bodies=int(unwrapped.scene["robot"].num_bodies),
    )
    obs, extras = env.reset()
    write_stage("env_reset")

    # Force a deterministic no-op step with zero action so last_action and
    # command buffers are populated through the official runtime path.
    action_dim = int(unwrapped.action_manager.total_action_dim)
    zero_action = torch.zeros((unwrapped.num_envs, action_dim), device=unwrapped.device)
    obs, reward, terminated, truncated, extras = env.step(zero_action)

    obs_dict = unwrapped.observation_manager.compute()
    policy_obs = obs_dict["policy"].detach().cpu()
    critic_obs = obs_dict["critic"].detach().cpu()
    active_terms = unwrapped.observation_manager.active_terms
    term_dims = unwrapped.observation_manager.group_obs_term_dim
    iterable_terms = unwrapped.observation_manager.get_active_iterable_terms(0)
    policy_terms = {
        name.split("-", 1)[1]: values
        for name, values in iterable_terms
        if name.startswith("policy-")
    }
    command = unwrapped.command_manager.get_term("motion")
    motion_time_steps = getattr(command, "time_steps", None)
    motion_time_step_total = getattr(command.motion, "time_step_total", None)
    motion_time_steps_list = to_list(motion_time_steps) if motion_time_steps is not None else None

    sample = {
        "status": "ok_isaaclab_observation_manager_sample_captured",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "Tracking-Flat-G1-v0",
        "claim_level": "official_isaaclab_observation_sample_only; no_mujoco_parity_claim",
        "robot_usd": str(ROBOT_USD),
        "motion_file": str(MOTION_FILE),
        "device": str(unwrapped.device),
        "num_envs": int(unwrapped.num_envs),
        "policy_obs_shape": list(policy_obs.shape),
        "critic_obs_shape": list(critic_obs.shape),
        "policy_obs_dim": int(unwrapped.observation_manager.group_obs_dim["policy"][0]),
        "critic_obs_dim": int(unwrapped.observation_manager.group_obs_dim["critic"][0]),
        "policy_term_names": list(active_terms["policy"]),
        "critic_term_names": list(active_terms["critic"]),
        "policy_term_dims": [list(dim) for dim in term_dims["policy"]],
        "critic_term_dims": [list(dim) for dim in term_dims["critic"]],
        "policy_terms": policy_terms,
        "policy_obs": policy_obs[0].tolist(),
        "critic_obs_head": critic_obs[0, : min(64, critic_obs.shape[-1])].tolist(),
        "last_action": policy_terms.get("actions", []),
        "zero_action_applied_before_capture": True,
        "motion_time_steps": motion_time_steps_list,
        "motion_time_step_total": int(motion_time_step_total) if motion_time_step_total is not None else None,
        "robot_anchor_body_index": int(command.robot_anchor_body_index),
        "motion_anchor_body_index": int(command.motion_anchor_body_index),
        "body_indexes": to_list(command.body_indexes),
        "reward_after_zero_step": float(reward.detach().cpu().mean().item()),
        "terminated_after_zero_step": bool(terminated.detach().cpu().any().item()),
        "truncated_after_zero_step": bool(truncated.detach().cpu().any().item()),
        "command_metrics": {
            name: float(value.detach().cpu().mean().item())
            for name, value in command.metrics.items()
            if hasattr(value, "detach") and value.numel() > 0
        },
        "checks": {
            "policy_obs_dim_160": int(unwrapped.observation_manager.group_obs_dim["policy"][0]) == 160,
            "policy_term_count_8": len(active_terms["policy"]) == 8,
            "policy_terms_expected_order": list(active_terms["policy"])
            == [
                "command",
                "motion_anchor_pos_b",
                "motion_anchor_ori_b",
                "base_lin_vel",
                "base_ang_vel",
                "joint_pos",
                "joint_vel",
                "actions",
            ],
            "zero_action_applied_before_capture": True,
            "does_not_claim_mujoco_parity_or_rollout": True,
        },
        "interpretation": {
            "official_observation_manager_sample_available": True,
            "mujoco_native_observation_parity_ready": False,
            "next_step": "Run a same-state MuJoCo adapter comparison against this sample; this file alone is not parity.",
        },
    }
    OUT_JSON.write_text(json.dumps(sample, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(
        OUT_NPZ,
        policy_obs=policy_obs.numpy(),
        critic_obs=critic_obs.numpy(),
        **{f"policy_{name}": np.asarray(values, dtype=np.float64) for name, values in policy_terms.items()},
    )
    print("BM_SENTINEL:obs_sample:sample_written=" + str(OUT_JSON), flush=True)
    print("BM_SENTINEL:obs_sample:npz_written=" + str(OUT_NPZ), flush=True)
    env.close()
    simulation_app.close(wait_for_replicator=False)
    print("BM_SENTINEL:obs_sample:success", flush=True)
    os._exit(0)
except BaseException as exc:
    write_stage("exception", exception=repr(exc), traceback=traceback.format_exc())
    raise
finally:
    try:
        simulation_app.close(wait_for_replicator=False)
    except BaseException:
        pass
