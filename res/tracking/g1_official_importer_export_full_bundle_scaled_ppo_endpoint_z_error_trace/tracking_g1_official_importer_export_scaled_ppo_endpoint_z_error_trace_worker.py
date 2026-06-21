
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

print(f"BM_SENTINEL:endpoint_trace:before_app:device={args.device}", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("BM_SENTINEL:endpoint_trace:after_app", flush=True)

try:
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import torch
    import whole_body_tracking.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

    usd = Path(os.environ["BM_OFFICIAL_IMPORTER_USD"])
    motion_file = Path(os.environ["BM_MOTION_FILE"])
    checkpoint = Path(os.environ["BM_CHECKPOINT"])
    run_dir = Path(os.environ["BM_RUN_DIR"])
    num_envs = int(os.environ["BM_NUM_ENVS"])
    eval_steps = int(os.environ["BM_EVAL_STEPS"])
    seed = int(os.environ["BM_EVAL_SEED"])
    threshold = float(os.environ["BM_EE_BODY_POS_THRESHOLD"])
    termination_bodies = os.environ["BM_TERMINATION_BODIES"].split(",")
    run_dir.mkdir(parents=True, exist_ok=True)
    step_csv = run_dir / "endpoint_z_error_timeseries.csv"
    body_csv = run_dir / "endpoint_z_error_by_body.csv"
    metrics_json = run_dir / "endpoint_z_error_metrics.json"

    env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=str(usd),
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

    agent_cfg = G1FlatPPORunnerCfg()
    agent_cfg.device = args.device
    agent_cfg.seed = seed
    agent_cfg.empirical_normalization = True
    agent_cfg.logger = "tensorboard"

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    vec_env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    loaded_infos = runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
    obs, extras = vec_env.get_observations()
    command = vec_env.unwrapped.command_manager.get_term("motion")

    body_indexes = [command.cfg.body_names.index(name) for name in termination_bodies]
    body_accum = {
        name: {
            "mean_abs_z": [],
            "p95_abs_z": [],
            "max_abs_z": [],
            "exceed_rate": [],
            "mean_signed_z": [],
        }
        for name in termination_bodies
    }
    aggregate_rows = []
    step_fields = [
        "step",
        "done_count",
        "timeout_count",
        "ee_body_pos_count",
        "aggregate_mean_abs_z",
        "aggregate_p95_abs_z",
        "aggregate_max_abs_z",
        "aggregate_exceed_rate",
    ]
    for name in termination_bodies:
        step_fields.extend(
            [
                f"{name}_mean_abs_z",
                f"{name}_p95_abs_z",
                f"{name}_max_abs_z",
                f"{name}_exceed_rate",
                f"{name}_mean_signed_z",
            ]
        )

    with step_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=step_fields, lineterminator="\n")
        writer.writeheader()
        with torch.inference_mode():
            for step in range(eval_steps):
                actions = policy(obs)
                obs, rew, dones, step_extras = vec_env.step(actions)
                target_z = command.body_pos_relative_w[:, body_indexes, 2]
                robot_z = command.robot_body_pos_w[:, body_indexes, 2]
                signed_z = robot_z - target_z
                abs_z = signed_z.abs()
                aggregate_abs = abs_z.reshape(-1)
                exceed_any = torch.any(abs_z > threshold, dim=1)
                log = step_extras.get("log", {})
                ee_count = log.get("Episode_Termination/ee_body_pos")
                if hasattr(ee_count, "detach"):
                    ee_value = int(float(ee_count.detach().mean().cpu()))
                else:
                    ee_value = int(float(ee_count)) if ee_count is not None else int(exceed_any.sum().detach().cpu())
                row = {
                    "step": step,
                    "done_count": int(dones.sum().detach().cpu()),
                    "timeout_count": int(step_extras.get("time_outs", torch.zeros_like(dones)).sum().detach().cpu()),
                    "ee_body_pos_count": ee_value,
                    "aggregate_mean_abs_z": float(aggregate_abs.mean().detach().cpu()),
                    "aggregate_p95_abs_z": float(torch.quantile(aggregate_abs, 0.95).detach().cpu()),
                    "aggregate_max_abs_z": float(aggregate_abs.max().detach().cpu()),
                    "aggregate_exceed_rate": float(exceed_any.float().mean().detach().cpu()),
                }
                for idx, name in enumerate(termination_bodies):
                    values = abs_z[:, idx]
                    signed_values = signed_z[:, idx]
                    body_exceed = (values > threshold).float()
                    stats = {
                        "mean_abs_z": float(values.mean().detach().cpu()),
                        "p95_abs_z": float(torch.quantile(values, 0.95).detach().cpu()),
                        "max_abs_z": float(values.max().detach().cpu()),
                        "exceed_rate": float(body_exceed.mean().detach().cpu()),
                        "mean_signed_z": float(signed_values.mean().detach().cpu()),
                    }
                    for key, value in stats.items():
                        body_accum[name][key].append(value)
                        row[f"{name}_{key}"] = value
                aggregate_rows.append(row)
                writer.writerow(row)

    body_rows = []
    for name, stats in body_accum.items():
        row = {"body_name": name, "threshold_m": threshold}
        for key, values in stats.items():
            row[f"{key}_mean_over_steps"] = sum(values) / len(values)
            row[f"{key}_max_over_steps"] = max(values)
            row[f"{key}_first"] = values[0]
            row[f"{key}_last"] = values[-1]
        body_rows.append(row)
    with body_csv.open("w", encoding="utf-8", newline="") as f:
        fields = list(body_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(body_rows)

    def series_summary(key):
        values = [float(row[key]) for row in aggregate_rows]
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
        "task": "Tracking-Flat-G1-v0",
        "checkpoint": str(checkpoint),
        "loaded_iteration": int(runner.current_learning_iteration),
        "loaded_infos_type": type(loaded_infos).__name__,
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "num_envs": int(vec_env.num_envs),
        "eval_steps": eval_steps,
        "total_env_steps": int(vec_env.num_envs) * eval_steps,
        "num_actions": int(vec_env.num_actions),
        "num_obs": int(vec_env.num_obs),
        "num_privileged_obs": int(vec_env.num_privileged_obs),
        "robot_num_joints": int(vec_env.unwrapped.scene["robot"].num_joints),
        "robot_num_bodies": int(vec_env.unwrapped.scene["robot"].num_bodies),
        "threshold_m": threshold,
        "termination_bodies": termination_bodies,
        "body_indexes": body_indexes,
        "aggregate": {
            "mean_abs_z": series_summary("aggregate_mean_abs_z"),
            "p95_abs_z": series_summary("aggregate_p95_abs_z"),
            "max_abs_z": series_summary("aggregate_max_abs_z"),
            "exceed_rate": series_summary("aggregate_exceed_rate"),
            "done_count": series_summary("done_count"),
            "ee_body_pos_count": series_summary("ee_body_pos_count"),
        },
        "body_rows": body_rows,
        "outputs": {
            "step_csv": str(step_csv),
            "body_csv": str(body_csv),
            "metrics_json": str(metrics_json),
        },
        "paper_level_tracking_eval": False,
        "uses_official_importer_export_usd": True,
        "official_csv_loop_full_public_bundle": True,
    }
    metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    vec_env.close()
    print(f"BM_SENTINEL:endpoint_trace:metrics_written={metrics_json}", flush=True)
    os._exit(0)
except BaseException as exc:
    print(f"BM_SENTINEL:endpoint_trace:exception={exc!r}", flush=True)
    raise
finally:
    try:
        simulation_app.close(wait_for_replicator=False)
    except BaseException:
        pass
