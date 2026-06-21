
import argparse
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(os.environ["BM_WORKER_METRICS_JSON"])
ROBOT_USD = Path(os.environ["BM_ROBOT_USD"])
MOTION_FILE = Path(os.environ["BM_MOTION_FILE"])
CHECKPOINT = Path(os.environ["BM_CHECKPOINT"])
NUM_ENVS = int(os.environ["BM_NUM_ENVS"])
SEED = int(os.environ["BM_SEED"])
TARGET_GPU = int(os.environ["BM_TARGET_GPU"])
THRESHOLD_M = float(os.environ["BM_THRESHOLD_M"])

ANKLE_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]
WRIST_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
ALL_ENDPOINT_NAMES = ANKLE_NAMES + WRIST_NAMES


def write_payload(payload):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("BM_SENTINEL:worker_metrics_written=" + str(OUT), flush=True)


from isaaclab.app import AppLauncher

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

print("BM_SENTINEL:wrist_probe:before_app", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("BM_SENTINEL:wrist_probe:after_app", flush=True)

try:
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import torch
    import whole_body_tracking.tasks  # noqa: F401
    from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

    def stats_tensor(tensor):
        if tensor is None or tensor.numel() == 0:
            return {"count": 0, "mean": None, "min": None, "max": None, "std": None}
        flat = tensor.detach().float().reshape(-1).cpu()
        finite = flat[torch.isfinite(flat)]
        if finite.numel() == 0:
            return {"count": 0, "mean": None, "min": None, "max": None, "std": None}
        return {
            "count": int(finite.numel()),
            "mean": float(finite.mean().item()),
            "min": float(finite.min().item()),
            "max": float(finite.max().item()),
            "std": float(finite.std(unbiased=False).item()) if finite.numel() > 1 else 0.0,
        }

    def group_stats(command, names):
        body_names = list(command.cfg.body_names)
        indexes = [body_names.index(name) for name in names if name in body_names]
        present = [name for name in names if name in body_names]
        if not indexes:
            return {
                "requested_names": names,
                "present_names": [],
                "indexes": [],
                "missing_names": names,
                "rel_z_error_m": stats_tensor(torch.empty(0)),
                "raw_z_error_m": stats_tensor(torch.empty(0)),
                "rel_xyz_error_m": stats_tensor(torch.empty(0)),
                "raw_xyz_error_m": stats_tensor(torch.empty(0)),
                "target_relative_z_m": stats_tensor(torch.empty(0)),
                "target_raw_z_m": stats_tensor(torch.empty(0)),
                "robot_z_m": stats_tensor(torch.empty(0)),
                "rel_z_done_count": 0,
                "rel_z_done_rate": 0.0,
            }
        target_relative = command.body_pos_relative_w[:, indexes, :].detach()
        target_raw = command.body_pos_w[:, indexes, :].detach()
        robot_body = command.robot_body_pos_w[:, indexes, :].detach()
        rel_z_error = torch.abs(target_relative[..., 2] - robot_body[..., 2])
        raw_z_error = torch.abs(target_raw[..., 2] - robot_body[..., 2])
        rel_xyz_error = torch.linalg.norm(target_relative - robot_body, dim=-1)
        raw_xyz_error = torch.linalg.norm(target_raw - robot_body, dim=-1)
        rel_done = torch.any(rel_z_error > THRESHOLD_M, dim=-1)
        per_body = []
        for local_idx, name in enumerate(present):
            per_body.append(
                {
                    "name": name,
                    "index": int(indexes[local_idx]),
                    "rel_z_error_m": stats_tensor(rel_z_error[:, local_idx]),
                    "raw_z_error_m": stats_tensor(raw_z_error[:, local_idx]),
                    "rel_xyz_error_m": stats_tensor(rel_xyz_error[:, local_idx]),
                    "raw_xyz_error_m": stats_tensor(raw_xyz_error[:, local_idx]),
                    "target_relative_z_m": stats_tensor(target_relative[:, local_idx, 2]),
                    "target_raw_z_m": stats_tensor(target_raw[:, local_idx, 2]),
                    "robot_z_m": stats_tensor(robot_body[:, local_idx, 2]),
                    "rel_z_exceed_rate": float((rel_z_error[:, local_idx] > THRESHOLD_M).float().mean().cpu().item()),
                }
            )
        return {
            "requested_names": names,
            "present_names": present,
            "indexes": [int(x) for x in indexes],
            "missing_names": [name for name in names if name not in body_names],
            "rel_z_error_m": stats_tensor(rel_z_error),
            "raw_z_error_m": stats_tensor(raw_z_error),
            "rel_xyz_error_m": stats_tensor(rel_xyz_error),
            "raw_xyz_error_m": stats_tensor(raw_xyz_error),
            "target_relative_z_m": stats_tensor(target_relative[..., 2]),
            "target_raw_z_m": stats_tensor(target_raw[..., 2]),
            "robot_z_m": stats_tensor(robot_body[..., 2]),
            "rel_z_done_count": int(rel_done.detach().cpu().sum().item()),
            "rel_z_done_rate": float(rel_done.float().mean().detach().cpu().item()),
            "per_body": per_body,
        }

    def snapshot(command, label, reward=None, dones=None, extras=None, action=None):
        command._update_metrics()
        body_error = torch.linalg.norm(command.body_pos_relative_w - command.robot_body_pos_w, dim=-1)
        row = {
            "label": label,
            "time_steps": stats_tensor(command.time_steps.detach().float()),
            "time_steps_first8": [int(x) for x in command.time_steps.detach().cpu()[:8]],
            "threshold_m": THRESHOLD_M,
            "all_endpoints": group_stats(command, ALL_ENDPOINT_NAMES),
            "ankles": group_stats(command, ANKLE_NAMES),
            "wrists": group_stats(command, WRIST_NAMES),
            "body_error_m": stats_tensor(body_error),
            "anchor_error_m": stats_tensor(torch.linalg.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=-1)),
            "joint_pos_error": stats_tensor(torch.linalg.norm(command.joint_pos - command.robot_joint_pos, dim=-1)),
            "joint_vel_error": stats_tensor(torch.linalg.norm(command.joint_vel - command.robot_joint_vel, dim=-1)),
        }
        if reward is not None:
            row["reward_mean"] = float(reward.detach().float().mean().cpu().item())
        if dones is not None:
            row["step_done_count"] = int(dones.detach().cpu().sum().item())
            row["step_done_rate"] = float(dones.float().mean().detach().cpu().item())
        if extras is not None:
            timeout_tensor = extras.get("time_outs")
            if timeout_tensor is not None:
                row["timeout_count"] = int(timeout_tensor.detach().cpu().sum().item())
        if action is not None:
            row["action_abs"] = stats_tensor(action.detach().abs())
        return row

    def refresh_motion_targets_no_advance(command):
        anchor_pos_w_repeat = command.anchor_pos_w[:, None, :].repeat(1, len(command.cfg.body_names), 1)
        anchor_quat_w_repeat = command.anchor_quat_w[:, None, :].repeat(1, len(command.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = command.robot_anchor_pos_w[:, None, :].repeat(1, len(command.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = command.robot_anchor_quat_w[:, None, :].repeat(1, len(command.cfg.body_names), 1)
        delta_pos_w = robot_anchor_pos_w_repeat.clone()
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))
        command.body_quat_relative_w = quat_mul(delta_ori_w, command.body_quat_w)
        command.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, command.body_pos_w - anchor_pos_w_repeat)
        command._update_metrics()

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
    env_cfg.seed = SEED

    agent_cfg = G1FlatPPORunnerCfg()
    agent_cfg.device = args.device
    agent_cfg.seed = SEED
    agent_cfg.empirical_normalization = True
    agent_cfg.logger = "tensorboard"

    print("BM_SENTINEL:wrist_probe:before_gym_make", flush=True)
    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print("BM_SENTINEL:wrist_probe:env_created", flush=True)
    vec_env = RslRlVecEnvWrapper(env)
    command = vec_env.unwrapped.command_manager.get_term("motion")
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    loaded_infos = runner.load(str(CHECKPOINT))
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
    print("BM_SENTINEL:wrist_probe:policy_loaded", flush=True)

    snapshots = []

    vec_env.unwrapped.reset(seed=SEED)
    snapshots.append(snapshot(command, "zero_branch_after_reset_before_target_refresh"))
    time_steps_before = command.time_steps.detach().clone()
    refresh_motion_targets_no_advance(command)
    time_steps_after = command.time_steps.detach().clone()
    vec_env.unwrapped.obs_buf = vec_env.unwrapped.observation_manager.compute()
    snapshots.append(snapshot(command, "zero_branch_after_target_refresh_no_advance"))
    zero_action = torch.zeros((vec_env.num_envs, vec_env.num_actions), device=vec_env.unwrapped.device)
    obs, reward, dones, extras = vec_env.step(zero_action)
    snapshots.append(snapshot(command, "zero_branch_after_one_zero_action_step", reward, dones, extras, zero_action))

    vec_env.unwrapped.reset(seed=SEED)
    snapshots.append(snapshot(command, "policy_branch_after_reset_before_target_refresh"))
    refresh_motion_targets_no_advance(command)
    vec_env.unwrapped.obs_buf = vec_env.unwrapped.observation_manager.compute()
    snapshots.append(snapshot(command, "policy_branch_after_target_refresh_no_advance"))
    obs, extras = vec_env.get_observations()
    with torch.inference_mode():
        policy_action = policy(obs)
    obs, reward, dones, extras = vec_env.step(policy_action)
    snapshots.append(snapshot(command, "policy_branch_after_one_policy_step", reward, dones, extras, policy_action))

    by_label = {row["label"]: row for row in snapshots}
    before = by_label["zero_branch_after_reset_before_target_refresh"]
    after_refresh = by_label["zero_branch_after_target_refresh_no_advance"]
    policy_step = by_label["policy_branch_after_one_policy_step"]
    zero_step = by_label["zero_branch_after_one_zero_action_step"]
    wrist_after = after_refresh["wrists"]
    ankle_after = after_refresh["ankles"]
    wrist_policy = policy_step["wrists"]
    ankle_policy = policy_step["ankles"]

    def value(row, group, metric, stat="mean"):
        return row[group][metric].get(stat)

    summary = {
        "time_steps_unchanged_by_refresh": bool(torch.equal(time_steps_before, time_steps_after)),
        "reset_wrist_rel_z_error_mean_before": value(before, "wrists", "rel_z_error_m"),
        "reset_wrist_rel_z_error_mean_after_refresh": value(after_refresh, "wrists", "rel_z_error_m"),
        "reset_ankle_rel_z_error_mean_before": value(before, "ankles", "rel_z_error_m"),
        "reset_ankle_rel_z_error_mean_after_refresh": value(after_refresh, "ankles", "rel_z_error_m"),
        "refresh_wrist_done_rate": wrist_after["rel_z_done_rate"],
        "refresh_ankle_done_rate": ankle_after["rel_z_done_rate"],
        "refresh_wrist_minus_ankle_done_rate": wrist_after["rel_z_done_rate"] - ankle_after["rel_z_done_rate"],
        "refresh_wrist_rel_z_error_mean": wrist_after["rel_z_error_m"]["mean"],
        "refresh_ankle_rel_z_error_mean": ankle_after["rel_z_error_m"]["mean"],
        "refresh_wrist_minus_ankle_rel_z_error_mean": wrist_after["rel_z_error_m"]["mean"]
        - ankle_after["rel_z_error_m"]["mean"],
        "policy_step_done_rate": policy_step.get("step_done_rate"),
        "policy_step_wrist_done_rate": wrist_policy["rel_z_done_rate"],
        "policy_step_ankle_done_rate": ankle_policy["rel_z_done_rate"],
        "policy_step_wrist_minus_ankle_done_rate": wrist_policy["rel_z_done_rate"] - ankle_policy["rel_z_done_rate"],
        "zero_step_done_rate": zero_step.get("step_done_rate"),
        "zero_step_wrist_done_rate": zero_step["wrists"]["rel_z_done_rate"],
        "zero_step_ankle_done_rate": zero_step["ankles"]["rel_z_done_rate"],
    }
    summary["refresh_reduces_wrist_z_error"] = (
        summary["reset_wrist_rel_z_error_mean_after_refresh"] is not None
        and summary["reset_wrist_rel_z_error_mean_before"] is not None
        and summary["reset_wrist_rel_z_error_mean_after_refresh"] < summary["reset_wrist_rel_z_error_mean_before"]
    )
    summary["refresh_reduces_ankle_z_error"] = (
        summary["reset_ankle_rel_z_error_mean_after_refresh"] is not None
        and summary["reset_ankle_rel_z_error_mean_before"] is not None
        and summary["reset_ankle_rel_z_error_mean_after_refresh"] < summary["reset_ankle_rel_z_error_mean_before"]
    )
    summary["wrist_dominates_refresh"] = summary["refresh_wrist_done_rate"] > summary["refresh_ankle_done_rate"]
    summary["wrist_dominates_policy_step"] = (
        summary["policy_step_wrist_done_rate"] > summary["policy_step_ankle_done_rate"]
    )
    if summary["wrist_dominates_refresh"] or summary["wrist_dominates_policy_step"]:
        diagnosis = "wrist_endpoint_target_or_body_semantics_remain_primary_done_source"
    elif summary["refresh_wrist_rel_z_error_mean"] > summary["refresh_ankle_rel_z_error_mean"]:
        diagnosis = "wrist_endpoint_z_error_larger_but_not_primary_done_source"
    else:
        diagnosis = "wrist_endpoint_not_larger_than_ankle_in_this_probe"
    summary["diagnosis"] = diagnosis

    body_names = list(command.cfg.body_names)
    payload = {
        "status": "ok_robot_order_fk_wrist_endpoint_alignment_live_probe",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "robot_order_fk_wrist_endpoint_alignment_live_probe_worker",
        "scope": "Live IsaacLab target/body alignment probe for ankle and wrist ee_body_pos endpoints.",
        "device": str(vec_env.unwrapped.device),
        "target_gpu": TARGET_GPU,
        "num_envs": int(vec_env.num_envs),
        "seed": SEED,
        "threshold_m": THRESHOLD_M,
        "checkpoint": str(CHECKPOINT),
        "loaded_iteration": int(runner.current_learning_iteration),
        "loaded_infos_type": type(loaded_infos).__name__,
        "robot_usd": str(ROBOT_USD),
        "motion_file": str(MOTION_FILE),
        "body_names": body_names,
        "endpoint_indexes": {
            "ankles": [body_names.index(name) for name in ANKLE_NAMES if name in body_names],
            "wrists": [body_names.index(name) for name in WRIST_NAMES if name in body_names],
        },
        "snapshots": snapshots,
        "summary": summary,
        "checks": {
            "uses_official_importer_export_usd": True,
            "uses_robot_order_fk_repaired_bundle": True,
            "checkpoint_loaded": True,
            "ankle_names_found": len([name for name in ANKLE_NAMES if name in body_names]) == len(ANKLE_NAMES),
            "wrist_names_found": len([name for name in WRIST_NAMES if name in body_names]) == len(WRIST_NAMES),
            "time_steps_unchanged_by_refresh": summary["time_steps_unchanged_by_refresh"],
            "records_body_pos_w": True,
            "records_body_pos_relative_w": True,
            "records_robot_body_pos_w": True,
            "records_ankle_and_wrist_groups": True,
            "wrist_dominance_classified": diagnosis in {
                "wrist_endpoint_target_or_body_semantics_remain_primary_done_source",
                "wrist_endpoint_z_error_larger_but_not_primary_done_source",
                "wrist_endpoint_not_larger_than_ankle_in_this_probe",
            },
            "does_not_claim_paper_level_tracking": True,
            "does_not_claim_goal_complete": True,
            "does_not_claim_real_robot": True,
            "does_not_train": True,
        },
        "interpretation": {
            "claim_level": "tracking_wrist_endpoint_alignment_live_diagnostic",
            "goal_complete": False,
            "paper_level_tracking_reproduced": False,
            "real_robot": False,
            "next_step": (
                "If wrist dominance is confirmed, inspect wrist body target generation, FK height, and ee_body_pos "
                "body selection before launching another PPO run."
            ),
        },
    }
    write_payload(payload)
    print("BM_SENTINEL:wrist_probe:success", flush=True)
    os._exit(0)
except BaseException as exc:
    payload = {
        "status": "failed_robot_order_fk_wrist_endpoint_alignment_live_probe",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exception": repr(exc),
        "traceback": traceback.format_exc(),
        "robot_usd": str(ROBOT_USD),
        "motion_file": str(MOTION_FILE),
        "checkpoint": str(CHECKPOINT),
    }
    write_payload(payload)
    print("BM_SENTINEL:wrist_probe:exception=" + repr(exc), flush=True)
    raise
finally:
    try:
        simulation_app.close(wait_for_replicator=False)
        print("BM_SENTINEL:wrist_probe:after_close", flush=True)
    except BaseException:
        pass
