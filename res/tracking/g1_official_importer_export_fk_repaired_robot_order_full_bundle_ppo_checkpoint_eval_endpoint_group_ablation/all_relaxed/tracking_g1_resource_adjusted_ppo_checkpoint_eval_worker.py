
import argparse
import csv
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
args.device = "cuda:0"

print(f"BM_SENTINEL:eval:before_app:device={args.device}", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("BM_SENTINEL:eval:after_app", flush=True)

try:
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import torch
    import whole_body_tracking.tasks  # noqa: F401
    from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

    enriched_usd = Path(os.environ["BM_ENRICHED_USD"])
    motion_file = Path(os.environ["BM_MOTION_FILE"])
    checkpoint = Path(os.environ["BM_CHECKPOINT"])
    run_dir = Path(os.environ["BM_RUN_DIR"])
    metrics_path = run_dir / "eval_metrics.json"
    timeseries_path = run_dir / "eval_timeseries.csv"
    run_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = int(os.environ["BM_NUM_ENVS"])
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
    env_cfg.seed = int(os.environ["BM_EVAL_SEED"])

    agent_cfg = G1FlatPPORunnerCfg()
    agent_cfg.device = args.device
    agent_cfg.seed = int(os.environ["BM_EVAL_SEED"])
    agent_cfg.empirical_normalization = True
    agent_cfg.logger = "tensorboard"

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print(f"BM_SENTINEL:eval:env_created:num_envs={env.unwrapped.num_envs}", flush=True)
    vec_env = RslRlVecEnvWrapper(env)
    command = vec_env.unwrapped.command_manager.get_term("motion")
    endpoint_names = [
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]
    ankle_endpoint_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    wrist_endpoint_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]

    def stats_tensor(tensor):
        if tensor.numel() == 0:
            return {"count": 0, "mean": None, "min": None, "max": None}
        flat = tensor.detach().float().reshape(-1).cpu()
        finite = flat[torch.isfinite(flat)]
        if finite.numel() == 0:
            return {"count": 0, "mean": None, "min": None, "max": None}
        return {
            "count": int(finite.numel()),
            "mean": float(finite.mean().item()),
            "min": float(finite.min().item()),
            "max": float(finite.max().item()),
        }

    def endpoint_group_stats(names, threshold=0.25):
        body_names = list(command.cfg.body_names)
        indexes = [body_names.index(name) for name in names if name in body_names]
        if not indexes:
            zeros = torch.zeros(vec_env.unwrapped.num_envs, dtype=torch.bool, device=vec_env.unwrapped.device)
            return {
                "names": names,
                "names_present": [],
                "indexes": [],
                "done_count": 0,
                "done_rate": 0.0,
                "z_error_m": {"count": 0, "mean": None, "min": None, "max": None},
            }
        body_target = command.body_pos_relative_w.detach()
        robot_body = command.robot_body_pos_w.detach()
        z_error = torch.abs(body_target[:, indexes, 2] - robot_body[:, indexes, 2])
        done = torch.any(z_error > threshold, dim=-1)
        return {
            "names": names,
            "names_present": [name for name in names if name in body_names],
            "indexes": indexes,
            "done_count": int(done.detach().cpu().sum().item()),
            "done_rate": float(done.float().mean().detach().cpu().item()),
            "z_error_m": stats_tensor(z_error),
        }

    def endpoint_snapshot(label):
        body_error = torch.linalg.norm(command.body_pos_relative_w.detach() - command.robot_body_pos_w.detach(), dim=-1)
        return {
            "label": label,
            "all": endpoint_group_stats(endpoint_names, threshold=0.25),
            "ankles": endpoint_group_stats(ankle_endpoint_names, threshold=0.25),
            "wrists": endpoint_group_stats(wrist_endpoint_names, threshold=0.25),
            "body_error_m": stats_tensor(body_error),
            "time_steps": stats_tensor(command.time_steps.detach().float()),
        }

    def refresh_motion_targets_no_advance():
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

    reset_target_refresh_before = endpoint_snapshot("after_wrapper_reset_before_target_refresh")
    time_steps_before_refresh = command.time_steps.detach().clone()
    refresh_motion_targets_no_advance()
    time_steps_after_refresh = command.time_steps.detach().clone()
    vec_env.unwrapped.obs_buf = vec_env.unwrapped.observation_manager.compute()
    reset_target_refresh_after = endpoint_snapshot("after_wrapper_reset_target_refresh_no_advance")
    reset_target_refresh_after["time_steps_unchanged_by_refresh"] = bool(
        torch.equal(time_steps_before_refresh, time_steps_after_refresh)
    )

    active_ee_body_names = [name for name in os.environ.get("BM_ENDPOINT_GROUP_ACTIVE_EE_BODY_NAMES", "").split(",") if name]
    ee_threshold = float(os.environ.get("BM_ENDPOINT_GROUP_EE_BODY_POS_THRESHOLD", "0.25"))
    ee_cfg = vec_env.unwrapped.termination_manager.get_term_cfg("ee_body_pos")
    original_ee_body_pos_threshold = float(ee_cfg.params.get("threshold", 0.25))
    original_ee_body_pos_body_names = list(ee_cfg.params.get("body_names", []))
    ee_cfg.params = dict(ee_cfg.params)
    ee_cfg.params["body_names"] = active_ee_body_names
    ee_cfg.params["threshold"] = ee_threshold
    vec_env.unwrapped.termination_manager.set_term_cfg("ee_body_pos", ee_cfg)
    ee_cfg_after = vec_env.unwrapped.termination_manager.get_term_cfg("ee_body_pos")
    ee_body_pos_threshold_after = float(ee_cfg_after.params.get("threshold"))
    ee_body_pos_body_names_after = list(ee_cfg_after.params.get("body_names", []))

    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    loaded_infos = runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
    obs, extras = vec_env.get_observations()

    reward_means = []
    reward_mins = []
    reward_maxs = []
    done_counts = []
    timeout_counts = []
    action_abs_means = []
    action_abs_maxs = []
    term_anchor_pos_counts = []
    term_anchor_ori_counts = []
    term_ee_body_pos_counts = []
    manual_all_endpoint_counts = []
    manual_ankle_endpoint_counts = []
    manual_wrist_endpoint_counts = []
    manual_all_endpoint_rates = []
    manual_ankle_endpoint_rates = []
    manual_wrist_endpoint_rates = []
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
    fieldnames = [
        "step",
        "reward_mean",
        "reward_min",
        "reward_max",
        "done_count",
        "timeout_count",
        "action_abs_mean",
        "action_abs_max",
        "term_anchor_pos_count",
        "term_anchor_ori_count",
        "term_ee_body_pos_count",
        "manual_all_endpoint_done_count_0p25",
        "manual_ankle_endpoint_done_count_0p25",
        "manual_wrist_endpoint_done_count_0p25",
        "manual_all_endpoint_done_rate_0p25",
        "manual_ankle_endpoint_done_rate_0p25",
        "manual_wrist_endpoint_done_rate_0p25",
    ] + metric_names

    with timeseries_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        with torch.inference_mode():
            for step in range(int(os.environ["BM_EVAL_STEPS"])):
                actions = policy(obs)
                obs, rew, dones, step_extras = vec_env.step(actions)
                endpoint_after_step = endpoint_snapshot(f"after_step_{step}")
                row = {
                    "step": step,
                    "reward_mean": float(rew.mean().detach().cpu()),
                    "reward_min": float(rew.min().detach().cpu()),
                    "reward_max": float(rew.max().detach().cpu()),
                    "done_count": int(dones.sum().detach().cpu()),
                    "timeout_count": int(step_extras.get("time_outs", torch.zeros_like(dones)).sum().detach().cpu()),
                    "action_abs_mean": float(actions.abs().mean().detach().cpu()),
                    "action_abs_max": float(actions.abs().max().detach().cpu()),
                    "term_anchor_pos_count": int(vec_env.unwrapped.termination_manager.get_term("anchor_pos").sum().detach().cpu()),
                    "term_anchor_ori_count": int(vec_env.unwrapped.termination_manager.get_term("anchor_ori").sum().detach().cpu()),
                    "term_ee_body_pos_count": int(vec_env.unwrapped.termination_manager.get_term("ee_body_pos").sum().detach().cpu()),
                    "manual_all_endpoint_done_count_0p25": endpoint_after_step["all"]["done_count"],
                    "manual_ankle_endpoint_done_count_0p25": endpoint_after_step["ankles"]["done_count"],
                    "manual_wrist_endpoint_done_count_0p25": endpoint_after_step["wrists"]["done_count"],
                    "manual_all_endpoint_done_rate_0p25": endpoint_after_step["all"]["done_rate"],
                    "manual_ankle_endpoint_done_rate_0p25": endpoint_after_step["ankles"]["done_rate"],
                    "manual_wrist_endpoint_done_rate_0p25": endpoint_after_step["wrists"]["done_rate"],
                }
                for name in metric_names:
                    tensor = command.metrics.get(name)
                    value = float(tensor.mean().detach().cpu()) if tensor is not None else float("nan")
                    row[name] = value
                    metric_series.setdefault(name, []).append(value)
                writer.writerow(row)

                reward_means.append(row["reward_mean"])
                reward_mins.append(row["reward_min"])
                reward_maxs.append(row["reward_max"])
                done_counts.append(row["done_count"])
                timeout_counts.append(row["timeout_count"])
                action_abs_means.append(row["action_abs_mean"])
                action_abs_maxs.append(row["action_abs_max"])
                term_anchor_pos_counts.append(row["term_anchor_pos_count"])
                term_anchor_ori_counts.append(row["term_anchor_ori_count"])
                term_ee_body_pos_counts.append(row["term_ee_body_pos_count"])
                manual_all_endpoint_counts.append(row["manual_all_endpoint_done_count_0p25"])
                manual_ankle_endpoint_counts.append(row["manual_ankle_endpoint_done_count_0p25"])
                manual_wrist_endpoint_counts.append(row["manual_wrist_endpoint_done_count_0p25"])
                manual_all_endpoint_rates.append(row["manual_all_endpoint_done_rate_0p25"])
                manual_ankle_endpoint_rates.append(row["manual_ankle_endpoint_done_rate_0p25"])
                manual_wrist_endpoint_rates.append(row["manual_wrist_endpoint_done_rate_0p25"])
                for key, value in step_extras.get("log", {}).items():
                    try:
                        scalar = float(value.detach().mean().cpu()) if hasattr(value, "detach") else float(value)
                    except Exception:
                        continue
                    episode_log_accum.setdefault(key, []).append(scalar)

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

    obs, extras = vec_env.get_observations()
    metrics = {
        "status": "ok",
        "task": "Tracking-Flat-G1-v0",
        "checkpoint": str(checkpoint),
        "loaded_iteration": int(runner.current_learning_iteration),
        "loaded_infos_type": type(loaded_infos).__name__,
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "num_envs": int(vec_env.num_envs),
        "eval_steps": int(os.environ["BM_EVAL_STEPS"]),
        "total_env_steps": int(vec_env.num_envs) * int(os.environ["BM_EVAL_STEPS"]),
        "num_actions": int(vec_env.num_actions),
        "num_obs": int(vec_env.num_obs),
        "num_privileged_obs": int(vec_env.num_privileged_obs),
        "policy_obs_shape": list(obs.shape),
        "critic_obs_shape": list(extras["observations"]["critic"].shape),
        "robot_num_joints": int(vec_env.unwrapped.scene["robot"].num_joints),
        "robot_num_bodies": int(vec_env.unwrapped.scene["robot"].num_bodies),
        "reward": {
            "mean_over_steps": summarize(reward_means),
            "min_over_steps": summarize(reward_mins),
            "max_over_steps": summarize(reward_maxs),
        },
        "done_count_total": int(sum(done_counts)),
        "timeout_count_total": int(sum(timeout_counts)),
        "action_abs_mean_over_steps": summarize(action_abs_means),
        "action_abs_max_over_steps": summarize(action_abs_maxs),
        "termination_term_counts": {
            "anchor_pos": summarize(term_anchor_pos_counts),
            "anchor_ori": summarize(term_anchor_ori_counts),
            "ee_body_pos_active_group": summarize(term_ee_body_pos_counts),
            "manual_all_endpoint_original_0p25": summarize(manual_all_endpoint_counts),
            "manual_ankle_endpoint_original_0p25": summarize(manual_ankle_endpoint_counts),
            "manual_wrist_endpoint_original_0p25": summarize(manual_wrist_endpoint_counts),
            "manual_all_endpoint_original_0p25_rate": summarize(manual_all_endpoint_rates),
            "manual_ankle_endpoint_original_0p25_rate": summarize(manual_ankle_endpoint_rates),
            "manual_wrist_endpoint_original_0p25_rate": summarize(manual_wrist_endpoint_rates),
        },
        "reset_target_refresh_no_advance": {
            "applied": True,
            "before": reset_target_refresh_before,
            "after": reset_target_refresh_after,
        },
        "ee_body_pos_endpoint_group_ablation": {
            "applied": True,
            "original_threshold": original_ee_body_pos_threshold,
            "original_body_names": original_ee_body_pos_body_names,
            "active_body_names": ee_body_pos_body_names_after,
            "active_threshold": ee_body_pos_threshold_after,
        },
        "motion_metrics": {name: summarize(values) for name, values in metric_series.items()},
        "episode_log_metrics": {name: summarize(values) for name, values in episode_log_accum.items()},
        "uses_resource_adjusted_usd": True,
        "official_csv_source": True,
        "official_csv_to_npz_output": False,
        "official_replay_output": False,
        "paper_level_tracking_eval": False,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    vec_env.close()
    print(f"BM_SENTINEL:eval:metrics_written={metrics_path}", flush=True)
    os._exit(0)
except BaseException as exc:
    print(f"BM_SENTINEL:eval:exception={exc!r}", flush=True)
    raise
finally:
    try:
        simulation_app.close(wait_for_replicator=False)
    except BaseException:
        pass
