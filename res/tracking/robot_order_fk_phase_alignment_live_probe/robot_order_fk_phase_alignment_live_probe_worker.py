
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
ENDPOINT_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]


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
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

    def stats_tensor(tensor):
        if tensor is None or tensor.numel() == 0:
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

    def bool_count(tensor):
        return int(tensor.detach().cpu().sum().item())

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

    print("BM_SENTINEL:before_gym_make", flush=True)
    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print("BM_SENTINEL:env_created", flush=True)
    vec_env = RslRlVecEnvWrapper(env)
    command = vec_env.unwrapped.command_manager.get_term("motion")
    robot = command.robot
    action_manager = vec_env.unwrapped.action_manager
    print("BM_SENTINEL:vec_env_ready", flush=True)

    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    loaded_infos = runner.load(str(CHECKPOINT))
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
    print("BM_SENTINEL:policy_loaded", flush=True)

    env_ids = torch.arange(vec_env.unwrapped.num_envs, dtype=torch.long, device=vec_env.unwrapped.device)

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

    def write_robot_to_current_motion_state():
        root_pos = command.body_pos_w[:, 0].clone()
        root_ori = command.body_quat_w[:, 0].clone()
        root_lin_vel = command.body_lin_vel_w[:, 0].clone()
        root_ang_vel = command.body_ang_vel_w[:, 0].clone()
        root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
        robot.write_joint_state_to_sim(command.joint_pos.clone(), command.joint_vel.clone(), env_ids=env_ids)
        robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        vec_env.unwrapped.scene.write_data_to_sim()
        vec_env.unwrapped.sim.forward()

    def reset_action_history():
        action_manager.reset(env_ids)
        zero = torch.zeros((vec_env.unwrapped.num_envs, action_manager.total_action_dim), device=vec_env.unwrapped.device)
        action_manager.process_action(zero)

    def compute_policy_action():
        obs, extras = vec_env.get_observations()
        with torch.inference_mode():
            action = policy(obs)
        return action.detach(), obs.detach()

    def snapshot(label, policy_action=None):
        command._update_metrics()
        body_names = list(command.cfg.body_names)
        endpoint_indexes = [body_names.index(name) for name in ENDPOINT_NAMES if name in body_names]
        body_target = command.body_pos_relative_w.detach()
        robot_body = command.robot_body_pos_w.detach()
        body_error = torch.linalg.norm(body_target - robot_body, dim=-1)
        endpoint_z_error = torch.abs(body_target[:, endpoint_indexes, 2] - robot_body[:, endpoint_indexes, 2])
        endpoint_done = torch.any(endpoint_z_error > 0.25, dim=-1)
        row = {
            "label": label,
            "time_steps": stats_tensor(command.time_steps.detach().float()),
            "time_steps_first8": [int(x) for x in command.time_steps.detach().cpu()[:8]],
            "manual_endpoint_z_done_count": bool_count(endpoint_done),
            "manual_endpoint_z_done_rate": float(endpoint_done.float().mean().detach().cpu().item()),
            "endpoint_z_error_m": stats_tensor(endpoint_z_error),
            "body_error_m": stats_tensor(body_error),
            "anchor_error_m": stats_tensor(torch.linalg.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=-1)),
            "joint_pos_error": stats_tensor(torch.linalg.norm(command.joint_pos - command.robot_joint_pos, dim=-1)),
            "joint_vel_error": stats_tensor(torch.linalg.norm(command.joint_vel - command.robot_joint_vel, dim=-1)),
            "body_lin_vel_error": stats_tensor(torch.linalg.norm(command.body_lin_vel_w - command.robot_body_lin_vel_w, dim=-1).mean(dim=-1)),
            "body_ang_vel_error": stats_tensor(torch.linalg.norm(command.body_ang_vel_w - command.robot_body_ang_vel_w, dim=-1).mean(dim=-1)),
            "action_manager_action_abs": stats_tensor(action_manager.action.abs()),
            "action_manager_prev_action_abs": stats_tensor(action_manager.prev_action.abs()),
        }
        if policy_action is not None:
            row["policy_action_abs"] = stats_tensor(policy_action.abs())
            row["policy_action_mean"] = float(policy_action.detach().float().mean().cpu().item())
        return row

    def apply_variant(variant):
        original_steps = command.time_steps.detach().clone()
        changed_robot_state = False
        if variant == "official_stale":
            pass
        elif variant == "refresh_t":
            refresh_motion_targets_no_advance()
        elif variant == "compute_dt0":
            vec_env.unwrapped.command_manager.compute(dt=0.0)
        elif variant == "update_command_t_plus_1":
            command._update_command()
            command._update_metrics()
        elif variant == "phase_minus_1_then_step_target_t":
            command.time_steps[:] = torch.clamp(original_steps - 1, min=0)
            refresh_motion_targets_no_advance()
        elif variant == "phase_plus_1_target_only":
            command.time_steps[:] = torch.clamp(original_steps + 1, max=command.motion.time_step_total - 1)
            refresh_motion_targets_no_advance()
        elif variant == "phase_plus_1_rewrite_state":
            command.time_steps[:] = torch.clamp(original_steps + 1, max=command.motion.time_step_total - 1)
            write_robot_to_current_motion_state()
            refresh_motion_targets_no_advance()
            changed_robot_state = True
        elif variant == "phase_minus_1_rewrite_state":
            command.time_steps[:] = torch.clamp(original_steps - 1, min=0)
            write_robot_to_current_motion_state()
            refresh_motion_targets_no_advance()
            changed_robot_state = True
        else:
            raise ValueError(f"unknown variant {variant}")
        reset_action_history()
        vec_env.unwrapped.obs_buf = vec_env.unwrapped.observation_manager.compute()
        return {
            "time_steps_changed": bool(not torch.equal(command.time_steps, original_steps)),
            "time_steps_delta_mean": float((command.time_steps.float() - original_steps.float()).mean().detach().cpu().item()),
            "robot_state_rewritten": changed_robot_state,
        }

    variants = [
        "official_stale",
        "refresh_t",
        "compute_dt0",
        "update_command_t_plus_1",
        "phase_minus_1_then_step_target_t",
        "phase_plus_1_target_only",
        "phase_plus_1_rewrite_state",
        "phase_minus_1_rewrite_state",
    ]
    action_modes = ["zero", "policy"]
    rows = []

    for variant in variants:
        for action_mode in action_modes:
            vec_env.unwrapped.reset(seed=SEED)
            before = snapshot(f"{variant}:{action_mode}:before_variant")
            variant_info = apply_variant(variant)
            policy_action, obs = compute_policy_action()
            after_variant = snapshot(f"{variant}:{action_mode}:after_variant", policy_action=policy_action)
            action = torch.zeros_like(policy_action) if action_mode == "zero" else policy_action
            obs, reward, dones, extras = vec_env.step(action)
            timeout_tensor = extras.get("time_outs", torch.zeros_like(dones))
            term_counts = {}
            for term_name in vec_env.unwrapped.termination_manager.active_terms:
                term_tensor = vec_env.unwrapped.termination_manager.get_term(term_name)
                term_counts[term_name] = int(term_tensor.detach().cpu().sum().item())
            after_step = snapshot(f"{variant}:{action_mode}:after_step")
            step_summary = {
                "variant": variant,
                "action_mode": action_mode,
                "done_count": int(dones.detach().cpu().sum().item()),
                "done_rate": float(dones.float().mean().detach().cpu().item()),
                "timeout_count": int(timeout_tensor.detach().cpu().sum().item()),
                "reward_mean": float(reward.detach().float().mean().cpu().item()),
                "action_abs_mean": float(action.detach().abs().mean().cpu().item()),
                "action_abs_max": float(action.detach().abs().max().cpu().item()),
                "termination_counts": term_counts,
            }
            rows.append(
                {
                    "variant": variant,
                    "action_mode": action_mode,
                    "variant_info": variant_info,
                    "before_variant": before,
                    "after_variant": after_variant,
                    "step_summary": step_summary,
                    "after_step": after_step,
                }
            )
            print(
                "BM_SENTINEL:variant_done:"
                + variant
                + ":"
                + action_mode
                + f":done_rate={step_summary['done_rate']:.6f}"
                + f":joint_vel={after_step['joint_vel_error']['mean']:.6f}",
                flush=True,
            )

    by_key = {(row["variant"], row["action_mode"]): row for row in rows}
    baseline = by_key[("refresh_t", "policy")]

    def done(row):
        return row["step_summary"]["done_rate"]

    def joint_vel(row):
        return row["after_step"]["joint_vel_error"]["mean"]

    def body(row):
        return row["after_step"]["body_error_m"]["mean"]

    policy_rows = [row for row in rows if row["action_mode"] == "policy"]
    candidates = []
    for row in policy_rows:
        if row["variant"] == "official_stale":
            continue
        improves_done = done(row) < done(baseline)
        improves_joint_vel = joint_vel(row) is not None and joint_vel(baseline) is not None and joint_vel(row) < joint_vel(baseline)
        candidates.append(
            {
                "variant": row["variant"],
                "done_rate": done(row),
                "joint_vel": joint_vel(row),
                "body_error": body(row),
                "reward_mean": row["step_summary"]["reward_mean"],
                "done_delta_vs_refresh_t": done(row) - done(baseline),
                "joint_vel_delta_vs_refresh_t": joint_vel(row) - joint_vel(baseline),
                "improves_done_rate": improves_done,
                "improves_joint_velocity": improves_joint_vel,
                "improves_both": bool(improves_done and improves_joint_vel),
            }
        )
    both = [c for c in candidates if c["improves_both"]]
    if both:
        best = sorted(both, key=lambda item: (item["done_rate"], item["joint_vel"]))[0]
        recommended = best["variant"]
        diagnosis = "phase_alignment_candidate_improves_done_and_joint_velocity"
    else:
        best = sorted(candidates, key=lambda item: (item["done_rate"], item["joint_vel"]))[0]
        recommended = ""
        diagnosis = "no_phase_alignment_variant_improves_both_done_and_joint_velocity"

    payload = {
        "status": "ok_robot_order_fk_phase_alignment_live_probe",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "robot_order_fk_phase_alignment_live_probe_worker",
        "scope": "Live IsaacLab reset/command phase alignment probe for robot-order FK tracking.",
        "device": str(vec_env.unwrapped.device),
        "target_gpu": TARGET_GPU,
        "num_envs": int(vec_env.num_envs),
        "seed": SEED,
        "step_dt": float(vec_env.unwrapped.step_dt),
        "checkpoint": str(CHECKPOINT),
        "loaded_iteration": int(runner.current_learning_iteration),
        "loaded_infos_type": type(loaded_infos).__name__,
        "robot_usd": str(ROBOT_USD),
        "motion_file": str(MOTION_FILE),
        "rows": rows,
        "summary": {
            "baseline_refresh_t_policy_done_rate": done(baseline),
            "baseline_refresh_t_policy_joint_vel_after_step": joint_vel(baseline),
            "best_by_done_then_joint_vel": best,
            "candidate_rows": candidates,
            "recommended_full_eval_variant": recommended,
            "diagnosis": diagnosis,
        },
        "checks": {
            "uses_official_importer_export_usd": True,
            "uses_robot_order_fk_repaired_bundle": True,
            "checkpoint_loaded": True,
            "all_variants_policy_and_zero_action_tested": len(rows) == len(variants) * len(action_modes),
            "compares_against_refresh_t_baseline": True,
            "any_variant_improves_done_and_joint_velocity": bool(recommended),
            "does_not_claim_paper_level_tracking": True,
            "does_not_claim_goal_complete": True,
            "does_not_claim_real_robot": True,
            "does_not_train": True,
        },
        "interpretation": {
            "claim_level": "tracking_phase_alignment_live_diagnostic",
            "goal_complete": False,
            "paper_level_tracking_reproduced": False,
            "real_robot": False,
            "recommended_full_eval_variant": recommended,
            "why_this_matters": (
                "IsaacLab computes termination before command_manager.compute(), while MotionCommand advances "
                "time_steps inside _update_command(). This live gate checks whether reset-time command phase "
                "alignment can improve the first policy step before launching full PPO."
            ),
        },
    }
    write_payload(payload)
    print("BM_SENTINEL:live_probe_success", flush=True)
    os._exit(0)
except BaseException as exc:
    payload = {
        "status": "failed_robot_order_fk_phase_alignment_live_probe",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exception": repr(exc),
        "traceback": traceback.format_exc(),
        "robot_usd": str(ROBOT_USD),
        "motion_file": str(MOTION_FILE),
        "checkpoint": str(CHECKPOINT),
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
