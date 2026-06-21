
import json
import math
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(os.environ["BM_WORKER_METRICS_JSON"])
ROBOT_USD = Path(os.environ["BM_ROBOT_USD"])
MOTION_FILE = Path(os.environ["BM_MOTION_FILE"])
NUM_ENVS = int(os.environ["BM_NUM_ENVS"])
ENDPOINT_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]


def write_payload(payload):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("BM_SENTINEL:worker_metrics_written=" + str(OUT), flush=True)


def stats_tensor(tensor):
    if tensor.numel() == 0:
        return {"count": 0, "mean": None, "min": None, "max": None, "std": None}
    t = tensor.detach().float().reshape(-1).cpu()
    finite = t[torch.isfinite(t)]
    if finite.numel() == 0:
        return {"count": 0, "mean": None, "min": None, "max": None, "std": None}
    return {
        "count": int(finite.numel()),
        "mean": float(finite.mean().item()),
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
        "std": float(finite.std(unbiased=False).item()) if finite.numel() > 1 else 0.0,
    }


def tensor_int(value):
    if hasattr(value, "detach"):
        return int(value.detach().cpu().sum().item())
    return int(value)


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
args.fast_shutdown = True
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
    from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

    def snapshot(label, command):
        body_names = list(command.cfg.body_names)
        endpoint_indexes = [body_names.index(name) for name in ENDPOINT_NAMES if name in body_names]
        body_target = command.body_pos_relative_w.detach()
        robot_body = command.robot_body_pos_w.detach()
        body_error = torch.linalg.norm(body_target - robot_body, dim=-1)
        endpoint_z_error = torch.abs(body_target[:, endpoint_indexes, 2] - robot_body[:, endpoint_indexes, 2])
        endpoint_manual_done = torch.any(endpoint_z_error > 0.25, dim=-1)
        anchor_error = torch.linalg.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=-1)
        time_steps = command.time_steps.detach().float()
        return {
            "label": label,
            "body_name_count": len(body_names),
            "endpoint_names_present": [name for name in ENDPOINT_NAMES if name in body_names],
            "endpoint_indexes": endpoint_indexes,
            "time_steps": stats_tensor(time_steps),
            "body_pos_relative_abs_max": float(body_target.abs().max().detach().cpu().item()),
            "robot_body_pos_abs_max": float(robot_body.abs().max().detach().cpu().item()),
            "body_error_m": stats_tensor(body_error),
            "anchor_error_m": stats_tensor(anchor_error),
            "endpoint_z_error_m": stats_tensor(endpoint_z_error),
            "manual_endpoint_z_done_count": int(endpoint_manual_done.detach().cpu().sum().item()),
            "manual_endpoint_z_done_rate": float(endpoint_manual_done.float().mean().detach().cpu().item()),
            "command_metric_error_body_pos_mean": (
                float(command.metrics["error_body_pos"].detach().float().mean().cpu().item())
                if "error_body_pos" in command.metrics
                else None
            ),
        }

    env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = NUM_ENVS
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
    env_cfg.seed = int(os.environ.get("BM_SEED", "20260760"))

    print("BM_SENTINEL:before_gym_make", flush=True)
    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print("BM_SENTINEL:env_created", flush=True)
    obs, extras = env.reset()
    print("BM_SENTINEL:env_reset", flush=True)
    command = env.unwrapped.command_manager.get_term("motion")

    def refresh_motion_targets_no_advance():
        anchor_pos_w_repeat = command.anchor_pos_w[:, None, :].repeat(1, len(command.cfg.body_names), 1)
        anchor_quat_w_repeat = command.anchor_quat_w[:, None, :].repeat(1, len(command.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = command.robot_anchor_pos_w[:, None, :].repeat(1, len(command.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = command.robot_anchor_quat_w[:, None, :].repeat(1, len(command.cfg.body_names), 1)
        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))
        command.body_quat_relative_w = quat_mul(delta_ori_w, command.body_quat_w)
        command.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, command.body_pos_w - anchor_pos_w_repeat)
        command._update_metrics()

    before = snapshot("after_reset_before_target_refresh", command)
    time_steps_before_refresh = command.time_steps.detach().clone()
    refresh_motion_targets_no_advance()
    time_steps_after_refresh = command.time_steps.detach().clone()
    obs = env.unwrapped.observation_manager.compute()
    after_warmup = snapshot("after_no_advance_target_refresh", command)
    after_warmup["time_steps_unchanged_by_refresh"] = bool(torch.equal(time_steps_before_refresh, time_steps_after_refresh))

    action_dim = int(env.unwrapped.action_manager.total_action_dim)
    zero_action = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
    obs, reward, terminated, truncated, extras = env.step(zero_action)
    after_step = snapshot("after_zero_action_step_following_warmup", command)
    terminated_count = tensor_int(terminated)
    truncated_count = tensor_int(truncated)
    done_count = terminated_count + truncated_count
    step_summary = {
        "terminated_count": terminated_count,
        "truncated_count": truncated_count,
        "done_count": done_count,
        "done_rate": done_count / float(env.unwrapped.num_envs),
        "reward_mean": float(reward.detach().float().mean().cpu().item()),
    }

    pre_mean = before["endpoint_z_error_m"]["mean"]
    post_mean = after_warmup["endpoint_z_error_m"]["mean"]
    pre_done = before["manual_endpoint_z_done_rate"]
    post_done = after_warmup["manual_endpoint_z_done_rate"]
    checks = {
        "uses_official_importer_export_usd": True,
        "uses_robot_order_fk_repaired_bundle": True,
        "num_envs_positive": int(env.unwrapped.num_envs) == NUM_ENVS,
        "endpoint_names_found": len(after_warmup["endpoint_names_present"]) == len(ENDPOINT_NAMES),
        "pre_warmup_manual_endpoint_done_rate_high": pre_done > 0.90,
        "warmup_reduces_endpoint_z_error_mean": post_mean is not None and pre_mean is not None and post_mean < pre_mean,
        "warmup_reduces_manual_endpoint_done_rate": post_done < pre_done,
        "post_warmup_manual_endpoint_done_rate_low": post_done < 0.05,
        "zero_action_step_after_warmup_not_all_done": step_summary["done_rate"] < 0.95,
        "does_not_claim_paper_level_tracking": True,
        "does_not_claim_goal_complete": True,
        "does_not_claim_real_robot": True,
    }
    if checks["post_warmup_manual_endpoint_done_rate_low"] and checks["zero_action_step_after_warmup_not_all_done"]:
        diagnosis = "no_advance_target_refresh_clears_reset_endpoint_z_spike"
    elif checks["warmup_reduces_endpoint_z_error_mean"]:
        diagnosis = "no_advance_target_refresh_partially_reduces_reset_endpoint_z_spike"
    else:
        diagnosis = "no_advance_target_refresh_does_not_clear_reset_endpoint_z_spike"

    payload = {
        "status": "ok_robot_order_fk_reset_target_refresh_no_advance_live_probe",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "robot_order_fk_reset_target_refresh_no_advance_live_probe_worker",
        "scope": "Live IsaacLab reset/no-advance target refresh diagnostic for the robot-order FK repaired tracking pipeline.",
        "device": str(env.unwrapped.device),
        "target_gpu": int(target_gpu),
        "num_envs": int(env.unwrapped.num_envs),
        "step_dt": float(env.unwrapped.step_dt),
        "physics_dt": float(env.unwrapped.physics_dt),
        "action_dim": action_dim,
        "policy_observation_dim": int(env.unwrapped.observation_manager.group_obs_dim["policy"][0]),
        "critic_observation_dim": int(env.unwrapped.observation_manager.group_obs_dim["critic"][0]),
        "robot_num_joints": int(env.unwrapped.scene["robot"].num_joints),
        "robot_num_bodies": int(env.unwrapped.scene["robot"].num_bodies),
        "robot_usd": str(ROBOT_USD),
        "motion_file": str(MOTION_FILE),
        "snapshots": [before, after_warmup, after_step],
        "step_summary": step_summary,
        "checks": checks,
        "diagnosis": diagnosis,
        "interpretation": {
            "claim_level": "live_tracking_reset_diagnostic",
            "goal_complete": False,
            "paper_level_tracking_reproduced": False,
            "real_robot": False,
            "next_step_if_cleared": (
                "Patch local tracking train/eval wrappers to refresh reset targets without advancing motion time, then "
                "rerun full robot-order FK tracking eval/PPO."
            ),
            "next_step_if_not_cleared": (
                "Inspect motion target frame, endpoint body mapping, and termination thresholds before another "
                "full PPO run."
            ),
        },
    }
    write_payload(payload)
    print("BM_SENTINEL:live_probe_success", flush=True)
    os._exit(0)
except BaseException as exc:
    payload = {
        "status": "failed_robot_order_fk_reset_target_refresh_no_advance_live_probe",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exception": repr(exc),
        "traceback": traceback.format_exc(),
        "robot_usd": str(ROBOT_USD),
        "motion_file": str(MOTION_FILE),
    }
    write_payload(payload)
    print("BM_SENTINEL:exception=" + repr(exc), flush=True)
    raise
finally:
    try:
        simulation_app.close(wait_for_replicator=False)
        print("BM_SENTINEL:after_close", flush=True)
    except BaseException:
        pass
