
import argparse
import csv
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OUT = Path(os.environ["BM_WORKER_METRICS_JSON"])
RUN_DIR = Path(os.environ["BM_RUN_DIR"])
ROBOT_USD = Path(os.environ["BM_ROBOT_USD"])
MOTION_FILE = Path(os.environ["BM_MOTION_FILE"])
MOTION_AUDIT = Path(os.environ["BM_MOTION_AUDIT"])
CHECKPOINT = Path(os.environ["BM_CHECKPOINT"])
NUM_ENVS = int(os.environ["BM_NUM_ENVS"])
EVAL_STEPS = int(os.environ["BM_EVAL_STEPS"])
SEED = int(os.environ["BM_SEED"])
TARGET_GPU = int(os.environ["BM_TARGET_GPU"])
THRESHOLD_M = float(os.environ["BM_THRESHOLD_M"])
PHASE_BINS = int(os.environ["BM_PHASE_BINS"])
ENDPOINT_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]
ANKLE_NAMES = ENDPOINT_NAMES[:2]
WRIST_NAMES = ENDPOINT_NAMES[2:]


def write_payload(payload):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("BM_SENTINEL:wrist_source:metrics_written=" + str(OUT), flush=True)


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

print("BM_SENTINEL:wrist_source:before_app", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("BM_SENTINEL:wrist_source:after_app", flush=True)

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

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    step_csv = RUN_DIR / "wrist_endpoint_source_step_timeseries.csv"
    motion_csv = RUN_DIR / "wrist_endpoint_source_by_motion.csv"
    phase_csv = RUN_DIR / "wrist_endpoint_source_by_phase_bin.csv"
    body_csv = RUN_DIR / "wrist_endpoint_source_by_body.csv"

    with MOTION_AUDIT.open("r", encoding="utf-8") as f:
        motion_audit = json.load(f)
    motion_rows = sorted(motion_audit["rows"], key=lambda row: int(row["start_frame"]))
    motion_names = [row["motion"] for row in motion_rows]
    frames_per_motion = int(motion_rows[0]["frame_count"])
    motion_count = len(motion_names)

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

    print("BM_SENTINEL:wrist_source:before_gym_make", flush=True)
    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print("BM_SENTINEL:wrist_source:env_created", flush=True)
    vec_env = RslRlVecEnvWrapper(env)
    command = vec_env.unwrapped.command_manager.get_term("motion")
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    loaded_infos = runner.load(str(CHECKPOINT))
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
    print("BM_SENTINEL:wrist_source:policy_loaded", flush=True)

    endpoint_indexes = [list(command.cfg.body_names).index(name) for name in ENDPOINT_NAMES]
    ankle_local = np.array([0, 1], dtype=np.int64)
    wrist_local = np.array([2, 3], dtype=np.int64)

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

    def endpoint_z_errors():
        target = command.body_pos_relative_w[:, endpoint_indexes, 2].detach()
        robot = command.robot_body_pos_w[:, endpoint_indexes, 2].detach()
        return torch.abs(target - robot)

    def get_term(name):
        try:
            return vec_env.unwrapped.termination_manager.get_term(name).detach()
        except BaseException:
            return torch.zeros(vec_env.num_envs, dtype=torch.bool, device=vec_env.unwrapped.device)

    vec_env.unwrapped.reset(seed=SEED)
    time_before = command.time_steps.detach().clone()
    refresh_motion_targets_no_advance()
    time_after = command.time_steps.detach().clone()
    vec_env.unwrapped.obs_buf = vec_env.unwrapped.observation_manager.compute()
    obs, extras = vec_env.get_observations()

    motion_count_shape = (motion_count, len(ENDPOINT_NAMES))
    phase_shape = (PHASE_BINS, len(ENDPOINT_NAMES))
    motion_count_arr = np.zeros(motion_count_shape, dtype=np.int64)
    motion_sum_pre = np.zeros(motion_count_shape, dtype=np.float64)
    motion_sum_post = np.zeros(motion_count_shape, dtype=np.float64)
    motion_max_pre = np.zeros(motion_count_shape, dtype=np.float64)
    motion_max_post = np.zeros(motion_count_shape, dtype=np.float64)
    motion_exceed_pre = np.zeros(motion_count_shape, dtype=np.int64)
    motion_exceed_post = np.zeros(motion_count_shape, dtype=np.int64)
    motion_done = np.zeros(motion_count, dtype=np.int64)
    motion_ee_term = np.zeros(motion_count, dtype=np.int64)

    phase_count_arr = np.zeros(phase_shape, dtype=np.int64)
    phase_sum_pre = np.zeros(phase_shape, dtype=np.float64)
    phase_sum_post = np.zeros(phase_shape, dtype=np.float64)
    phase_exceed_pre = np.zeros(phase_shape, dtype=np.int64)
    phase_exceed_post = np.zeros(phase_shape, dtype=np.int64)

    step_rows = []
    with step_csv.open("w", encoding="utf-8", newline="") as f:
        step_fields = [
            "step",
            "done_count",
            "ee_body_pos_count",
            "anchor_pos_count",
            "anchor_ori_count",
            "reward_mean",
            "pre_ankle_done_rate",
            "pre_wrist_done_rate",
            "post_ankle_done_rate",
            "post_wrist_done_rate",
            "pre_ankle_z_mean",
            "pre_wrist_z_mean",
            "post_ankle_z_mean",
            "post_wrist_z_mean",
            "left_wrist_pre_z_mean",
            "right_wrist_pre_z_mean",
            "left_wrist_post_z_mean",
            "right_wrist_post_z_mean",
        ]
        writer = csv.DictWriter(f, fieldnames=step_fields, lineterminator="\n")
        writer.writeheader()
        with torch.inference_mode():
            for step in range(EVAL_STEPS):
                pre_time = command.time_steps.detach().clone()
                pre_motion = torch.clamp(pre_time // frames_per_motion, 0, motion_count - 1).cpu().numpy()
                pre_phase = torch.clamp((pre_time % frames_per_motion) * PHASE_BINS // frames_per_motion, 0, PHASE_BINS - 1).cpu().numpy()
                pre_errors_t = endpoint_z_errors()
                pre_errors = pre_errors_t.cpu().numpy()
                actions = policy(obs)
                obs, reward, dones, step_extras = vec_env.step(actions)
                command._update_metrics()
                post_errors = endpoint_z_errors().cpu().numpy()
                pre_exceed = pre_errors > THRESHOLD_M
                post_exceed = post_errors > THRESHOLD_M
                done_np = dones.detach().cpu().numpy().astype(bool)
                ee_np = get_term("ee_body_pos").cpu().numpy().astype(bool)

                for body_i in range(len(ENDPOINT_NAMES)):
                    np.add.at(motion_count_arr[:, body_i], pre_motion, 1)
                    np.add.at(motion_sum_pre[:, body_i], pre_motion, pre_errors[:, body_i])
                    np.add.at(motion_sum_post[:, body_i], pre_motion, post_errors[:, body_i])
                    np.maximum.at(motion_max_pre[:, body_i], pre_motion, pre_errors[:, body_i])
                    np.maximum.at(motion_max_post[:, body_i], pre_motion, post_errors[:, body_i])
                    np.add.at(motion_exceed_pre[:, body_i], pre_motion, pre_exceed[:, body_i].astype(np.int64))
                    np.add.at(motion_exceed_post[:, body_i], pre_motion, post_exceed[:, body_i].astype(np.int64))
                    np.add.at(phase_count_arr[:, body_i], pre_phase, 1)
                    np.add.at(phase_sum_pre[:, body_i], pre_phase, pre_errors[:, body_i])
                    np.add.at(phase_sum_post[:, body_i], pre_phase, post_errors[:, body_i])
                    np.add.at(phase_exceed_pre[:, body_i], pre_phase, pre_exceed[:, body_i].astype(np.int64))
                    np.add.at(phase_exceed_post[:, body_i], pre_phase, post_exceed[:, body_i].astype(np.int64))
                np.add.at(motion_done, pre_motion, done_np.astype(np.int64))
                np.add.at(motion_ee_term, pre_motion, ee_np.astype(np.int64))

                row = {
                    "step": step,
                    "done_count": int(done_np.sum()),
                    "ee_body_pos_count": int(ee_np.sum()),
                    "anchor_pos_count": int(get_term("anchor_pos").sum().detach().cpu().item()),
                    "anchor_ori_count": int(get_term("anchor_ori").sum().detach().cpu().item()),
                    "reward_mean": float(reward.detach().float().mean().cpu().item()),
                    "pre_ankle_done_rate": float(np.any(pre_exceed[:, ankle_local], axis=1).mean()),
                    "pre_wrist_done_rate": float(np.any(pre_exceed[:, wrist_local], axis=1).mean()),
                    "post_ankle_done_rate": float(np.any(post_exceed[:, ankle_local], axis=1).mean()),
                    "post_wrist_done_rate": float(np.any(post_exceed[:, wrist_local], axis=1).mean()),
                    "pre_ankle_z_mean": float(pre_errors[:, ankle_local].mean()),
                    "pre_wrist_z_mean": float(pre_errors[:, wrist_local].mean()),
                    "post_ankle_z_mean": float(post_errors[:, ankle_local].mean()),
                    "post_wrist_z_mean": float(post_errors[:, wrist_local].mean()),
                    "left_wrist_pre_z_mean": float(pre_errors[:, 2].mean()),
                    "right_wrist_pre_z_mean": float(pre_errors[:, 3].mean()),
                    "left_wrist_post_z_mean": float(post_errors[:, 2].mean()),
                    "right_wrist_post_z_mean": float(post_errors[:, 3].mean()),
                }
                step_rows.append(row)
                writer.writerow(row)
                if step in {0, 1, 2, 9, 49, 99, 199, 298}:
                    print(
                        "BM_SENTINEL:wrist_source:step="
                        + str(step)
                        + f":done={row['done_count']}:pre_wrist={row['pre_wrist_done_rate']:.6f}:post_wrist={row['post_wrist_done_rate']:.6f}",
                        flush=True,
                    )

    def safe_rate(num, den):
        return float(num / den) if den else 0.0

    motion_rows_out = []
    for motion_i, name in enumerate(motion_names):
        group_count = int(motion_count_arr[motion_i, 0])
        row = {
            "motion_id": motion_i,
            "motion": name,
            "sample_count": group_count,
            "done_rate": safe_rate(motion_done[motion_i], group_count),
            "ee_body_pos_rate": safe_rate(motion_ee_term[motion_i], group_count),
        }
        for body_i, body_name in enumerate(ENDPOINT_NAMES):
            count = int(motion_count_arr[motion_i, body_i])
            row[f"{body_name}_pre_z_mean"] = safe_rate(motion_sum_pre[motion_i, body_i], count)
            row[f"{body_name}_post_z_mean"] = safe_rate(motion_sum_post[motion_i, body_i], count)
            row[f"{body_name}_pre_exceed_rate"] = safe_rate(motion_exceed_pre[motion_i, body_i], count)
            row[f"{body_name}_post_exceed_rate"] = safe_rate(motion_exceed_post[motion_i, body_i], count)
            row[f"{body_name}_pre_z_max"] = float(motion_max_pre[motion_i, body_i])
            row[f"{body_name}_post_z_max"] = float(motion_max_post[motion_i, body_i])
        row["wrist_pre_exceed_rate"] = max(
            row["left_wrist_yaw_link_pre_exceed_rate"], row["right_wrist_yaw_link_pre_exceed_rate"]
        )
        row["ankle_pre_exceed_rate"] = max(
            row["left_ankle_roll_link_pre_exceed_rate"], row["right_ankle_roll_link_pre_exceed_rate"]
        )
        row["wrist_minus_ankle_pre_exceed_rate"] = row["wrist_pre_exceed_rate"] - row["ankle_pre_exceed_rate"]
        motion_rows_out.append(row)

    motion_fields = list(motion_rows_out[0].keys())
    with motion_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=motion_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(motion_rows_out)

    phase_rows_out = []
    for phase_i in range(PHASE_BINS):
        for body_i, body_name in enumerate(ENDPOINT_NAMES):
            count = int(phase_count_arr[phase_i, body_i])
            phase_rows_out.append(
                {
                    "phase_bin": phase_i,
                    "body_name": body_name,
                    "sample_count": count,
                    "pre_z_mean": safe_rate(phase_sum_pre[phase_i, body_i], count),
                    "post_z_mean": safe_rate(phase_sum_post[phase_i, body_i], count),
                    "pre_exceed_rate": safe_rate(phase_exceed_pre[phase_i, body_i], count),
                    "post_exceed_rate": safe_rate(phase_exceed_post[phase_i, body_i], count),
                }
            )
    with phase_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(phase_rows_out[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(phase_rows_out)

    body_rows_out = []
    for body_i, body_name in enumerate(ENDPOINT_NAMES):
        total = int(motion_count_arr[:, body_i].sum())
        body_rows_out.append(
            {
                "body_name": body_name,
                "sample_count": total,
                "pre_z_mean": safe_rate(motion_sum_pre[:, body_i].sum(), total),
                "post_z_mean": safe_rate(motion_sum_post[:, body_i].sum(), total),
                "pre_exceed_rate": safe_rate(motion_exceed_pre[:, body_i].sum(), total),
                "post_exceed_rate": safe_rate(motion_exceed_post[:, body_i].sum(), total),
                "pre_z_max": float(motion_max_pre[:, body_i].max()),
                "post_z_max": float(motion_max_post[:, body_i].max()),
            }
        )
    with body_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(body_rows_out[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(body_rows_out)

    def series_summary(key):
        vals = [float(row[key]) for row in step_rows]
        return {
            "count": len(vals),
            "first": vals[0],
            "last": vals[-1],
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
        }

    top_wrist_motions = sorted(
        motion_rows_out,
        key=lambda row: (row["wrist_pre_exceed_rate"], row["ee_body_pos_rate"], row["wrist_minus_ankle_pre_exceed_rate"]),
        reverse=True,
    )[:8]
    summary = {
        "status": "ok_robot_order_fk_wrist_endpoint_source_full_diagnostic_worker",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "robot_order_fk_wrist_endpoint_source_full_diagnostic_worker",
        "scope": "2048-env x 299-step wrist/ankle endpoint source attribution for the robot-order FK PPO checkpoint.",
        "device": str(vec_env.unwrapped.device),
        "target_gpu": TARGET_GPU,
        "num_envs": int(vec_env.num_envs),
        "eval_steps": EVAL_STEPS,
        "total_env_steps": int(vec_env.num_envs) * EVAL_STEPS,
        "seed": SEED,
        "threshold_m": THRESHOLD_M,
        "checkpoint": str(CHECKPOINT),
        "loaded_iteration": int(runner.current_learning_iteration),
        "loaded_infos_type": type(loaded_infos).__name__,
        "robot_usd": str(ROBOT_USD),
        "motion_file": str(MOTION_FILE),
        "motion_count": motion_count,
        "frames_per_motion": frames_per_motion,
        "time_steps_unchanged_by_initial_refresh": bool(torch.equal(time_before, time_after)),
        "metrics": {
            "done_count_total": int(sum(row["done_count"] for row in step_rows)),
            "ee_body_pos_count_total": int(sum(row["ee_body_pos_count"] for row in step_rows)),
            "done_rate": float(sum(row["done_count"] for row in step_rows) / float(NUM_ENVS * EVAL_STEPS)),
            "ee_body_pos_rate": float(sum(row["ee_body_pos_count"] for row in step_rows) / float(NUM_ENVS * EVAL_STEPS)),
            "pre_wrist_done_rate": series_summary("pre_wrist_done_rate"),
            "pre_ankle_done_rate": series_summary("pre_ankle_done_rate"),
            "post_wrist_done_rate": series_summary("post_wrist_done_rate"),
            "post_ankle_done_rate": series_summary("post_ankle_done_rate"),
            "pre_wrist_z_mean": series_summary("pre_wrist_z_mean"),
            "pre_ankle_z_mean": series_summary("pre_ankle_z_mean"),
            "post_wrist_z_mean": series_summary("post_wrist_z_mean"),
            "post_ankle_z_mean": series_summary("post_ankle_z_mean"),
            "top_wrist_motion_count": len(top_wrist_motions),
        },
        "body_rows": body_rows_out,
        "top_wrist_motions": top_wrist_motions,
        "checks": {
            "uses_official_importer_export_usd": True,
            "uses_robot_order_fk_repaired_bundle": True,
            "checkpoint_loaded": True,
            "same_full_eval_scope_2048x299": int(vec_env.num_envs) == 2048 and EVAL_STEPS == 299,
            "motion_count_40": motion_count == 40,
            "records_step_motion_phase_and_body_sources": True,
            "time_steps_unchanged_by_initial_refresh": bool(torch.equal(time_before, time_after)),
            "wrist_pre_exceed_rate_exceeds_ankle": series_summary("pre_wrist_done_rate")["mean"]
            > series_summary("pre_ankle_done_rate")["mean"],
            "wrist_post_exceed_rate_exceeds_ankle": series_summary("post_wrist_done_rate")["mean"]
            > series_summary("post_ankle_done_rate")["mean"],
            "does_not_claim_paper_level_tracking": True,
            "does_not_claim_goal_complete": True,
            "does_not_claim_real_robot": True,
            "does_not_train": True,
        },
        "interpretation": {
            "claim_level": "tracking_wrist_endpoint_source_full_diagnostic",
            "goal_complete": False,
            "paper_level_tracking_reproduced": False,
            "real_robot": False,
            "next_step": (
                "Use the by-motion and by-phase rows to decide whether to repair selected wrist-heavy motions, "
                "change wrist endpoint body selection, or relax/replace the wrist component of ee_body_pos before "
                "launching another full PPO run."
            ),
        },
        "outputs": {
            "worker_metrics": str(OUT),
            "step_csv": str(step_csv),
            "motion_csv": str(motion_csv),
            "phase_csv": str(phase_csv),
            "body_csv": str(body_csv),
        },
    }
    write_payload(summary)
    print("BM_SENTINEL:wrist_source:success", flush=True)
    os._exit(0)
except BaseException as exc:
    payload = {
        "status": "failed_robot_order_fk_wrist_endpoint_source_full_diagnostic_worker",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exception": repr(exc),
        "traceback": traceback.format_exc(),
        "robot_usd": str(ROBOT_USD),
        "motion_file": str(MOTION_FILE),
        "checkpoint": str(CHECKPOINT),
    }
    write_payload(payload)
    print("BM_SENTINEL:wrist_source:exception=" + repr(exc), flush=True)
    raise
finally:
    try:
        simulation_app.close(wait_for_replicator=False)
    except BaseException:
        pass
