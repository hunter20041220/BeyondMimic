#!/usr/bin/env python3
"""Live reset/command phase-alignment probe for robot-order FK tracking.

The current robot-order FK chain has two coupled symptoms:

* after reset, official ``body_pos_relative_w`` starts at zero and the first
  termination check sees a stale target;
* refreshing the target removes the huge body-position spike but exposes a
  first-step joint-velocity/termination transient.

This probe compares phase-alignment variants in one IsaacLab process before
launching another expensive full PPO run. It does not modify the official
worktree, does not train, and does not claim paper-level tracking.
"""

from __future__ import annotations

import csv
import json
import os
import select
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/robot_order_fk_phase_alignment_live_probe"
LOG_DIR = ROOT / "logs/tracking_robot_order_fk_phase_alignment_live_probe"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
OFFICIAL_IMPORTER_USD = ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda"
ROBOT_ORDER_MOTION_NPZ = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/"
    "official_csv_loop_full_public_motion_bundle_fk_repaired_robot_order.npz"
)
TRAINING_RUN_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run.json"
)
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)

TARGET_GPU = int(os.environ.get("BM_PHASE_ALIGNMENT_GPU", "4"))
NUM_ENVS = int(os.environ.get("BM_PHASE_ALIGNMENT_NUM_ENVS", "256"))
SEED = int(os.environ.get("BM_PHASE_ALIGNMENT_SEED", "20260762"))
STALL_SECONDS = int(os.environ.get("BM_PHASE_ALIGNMENT_STALL_SECONDS", "900"))


WORKER_CODE = r"""
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
"""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    tmp.replace(path)


def latest_checkpoint() -> Path:
    training = load_json(TRAINING_RUN_JSON)
    rank_metrics = training.get("run", {}).get("rank_metrics", [])
    if rank_metrics:
        checkpoints = rank_metrics[0].get("checkpoints", [])
        if checkpoints:
            return Path(checkpoints[-1])
    candidates = sorted(
        (
            ROOT
            / "res/runs/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training"
        ).glob("*/rank_0/model_*.pt")
    )
    if not candidates:
        raise FileNotFoundError("No robot-order FK PPO checkpoint found.")
    return candidates[-1]


def env_for(worker_metrics: Path, checkpoint: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "VK_ICD_FILENAMES": str(PROJECT_EGL_ICD),
            "LD_LIBRARY_PATH": f"{GPU_FOUNDATION_DEPS}:{env.get('LD_LIBRARY_PATH', '')}",
            "PYTHONUNBUFFERED": "1",
            "ISAAC_PATH": str(ROOT / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim"),
            "OMNI_USER_DIR": str(ROOT / "cache/omni/user"),
            "OMNI_CACHE_DIR": str(ROOT / "cache/omni/cache"),
            "OMNI_DATA_DIR": str(ROOT / "cache/omni/data"),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "BM_WORKER_METRICS_JSON": str(worker_metrics),
            "BM_ROBOT_USD": str(OFFICIAL_IMPORTER_USD),
            "BM_MOTION_FILE": str(ROBOT_ORDER_MOTION_NPZ),
            "BM_CHECKPOINT": str(checkpoint),
            "BM_NUM_ENVS": str(NUM_ENVS),
            "BM_TARGET_GPU": str(TARGET_GPU),
            "BM_DEVICE": f"cuda:{TARGET_GPU}",
            "BM_SEED": str(SEED),
        }
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def run_worker(worker: Path, worker_metrics: Path, checkpoint: Path) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "robot_order_fk_phase_alignment_live_probe.log"
    cmd = [str(TRACKING_PY), str(worker), "--headless", "--device", f"cuda:{TARGET_GPU}"]
    start = time.time()
    last_change = time.time()
    last_size = -1
    stdout_tail: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env_for(worker_metrics, checkpoint),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert proc.stdout is not None
        while proc.poll() is None:
            ready, _, _ = select.select([proc.stdout], [], [], 5)
            if ready:
                line = proc.stdout.readline()
                if line:
                    stdout_tail.append(line.rstrip())
                    stdout_tail = stdout_tail[-120:]
                    log_file.write(line)
                    log_file.flush()
                    if line.startswith("BM_SENTINEL") or "Traceback" in line or "Error" in line:
                        print(line.rstrip(), flush=True)
            size = log_path.stat().st_size if log_path.is_file() else 0
            if size != last_size:
                last_size = size
                last_change = time.time()
            elif time.time() - last_change > STALL_SECONDS:
                log_file.write(f"\nBM_STALL_TIMEOUT:no_log_progress_for_{STALL_SECONDS}s\n")
                log_file.flush()
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=60)
                break
        for line in proc.stdout:
            stdout_tail.append(line.rstrip())
            stdout_tail = stdout_tail[-120:]
            log_file.write(line)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "duration_seconds": round(time.time() - start, 3),
        "log_path": str(log_path),
        "stdout_tail": stdout_tail,
        "worker_metrics_exists": worker_metrics.is_file(),
    }


def flat_rows(worker: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in worker.get("rows", []):
        step = item.get("step_summary", {})
        after = item.get("after_variant", {})
        after_step = item.get("after_step", {})
        term_counts = step.get("termination_counts", {})
        rows.append(
            {
                "variant": step.get("variant"),
                "action_mode": step.get("action_mode"),
                "time_steps_changed": item.get("variant_info", {}).get("time_steps_changed"),
                "time_steps_delta_mean": item.get("variant_info", {}).get("time_steps_delta_mean"),
                "robot_state_rewritten": item.get("variant_info", {}).get("robot_state_rewritten"),
                "pre_endpoint_done_rate": after.get("manual_endpoint_z_done_rate"),
                "pre_endpoint_z_mean": (after.get("endpoint_z_error_m") or {}).get("mean"),
                "pre_body_error_mean": (after.get("body_error_m") or {}).get("mean"),
                "pre_joint_pos_error_mean": (after.get("joint_pos_error") or {}).get("mean"),
                "pre_joint_vel_error_mean": (after.get("joint_vel_error") or {}).get("mean"),
                "policy_action_abs_mean": (after.get("policy_action_abs") or {}).get("mean"),
                "step_done_rate": step.get("done_rate"),
                "step_done_count": step.get("done_count"),
                "step_reward_mean": step.get("reward_mean"),
                "step_action_abs_mean": step.get("action_abs_mean"),
                "term_anchor_pos": term_counts.get("anchor_pos"),
                "term_anchor_ori": term_counts.get("anchor_ori"),
                "term_ee_body_pos": term_counts.get("ee_body_pos"),
                "post_endpoint_done_rate": after_step.get("manual_endpoint_z_done_rate"),
                "post_body_error_mean": (after_step.get("body_error_m") or {}).get("mean"),
                "post_joint_pos_error_mean": (after_step.get("joint_pos_error") or {}).get("mean"),
                "post_joint_vel_error_mean": (after_step.get("joint_vel_error") or {}).get("mean"),
            }
        )
    return rows


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary.get("metrics", {})
    rows = summary.get("rows", [])
    lines = [
        "# Robot-Order FK Phase-Alignment Live Probe",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- Worker status: `{summary.get('worker_metrics', {}).get('status')}`",
        f"- Diagnosis: `{metrics.get('diagnosis')}`",
        f"- Recommended full-eval variant: `{metrics.get('recommended_full_eval_variant')}`",
        f"- Baseline refresh-t policy done rate: `{metrics.get('baseline_refresh_t_policy_done_rate')}`",
        f"- Baseline refresh-t policy joint velocity: `{metrics.get('baseline_refresh_t_policy_joint_vel_after_step')}`",
        "",
        "## Policy-Step Comparison",
        "",
        "| variant | dt | rewritten | pre endpoint done | step done | ee term | body error | joint vel | reward |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["action_mode"] != "policy":
            continue
        lines.append(
            "| {variant} | {time_steps_delta_mean} | {robot_state_rewritten} | {pre_endpoint_done_rate} | "
            "{step_done_rate} | {term_ee_body_pos} | {post_body_error_mean} | {post_joint_vel_error_mean} | "
            "{step_reward_mean} |".format(**row)
        )
    lines.extend(
        [
            "",
            "This is a local virtual live diagnostic. It does not train PPO, does not claim paper-level tracking, "
            "does not claim DAgger/VAE/diffusion success, and does not use real hardware.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = latest_checkpoint()
    worker = OUT / "robot_order_fk_phase_alignment_live_probe_worker.py"
    worker_metrics_path = OUT / "robot_order_fk_phase_alignment_live_probe_worker_metrics.json"
    worker.write_text(WORKER_CODE, encoding="utf-8")
    run = run_worker(worker, worker_metrics_path, checkpoint)
    worker_metrics = load_json(worker_metrics_path)
    rows = flat_rows(worker_metrics)
    tsv_path = OUT / "robot_order_fk_phase_alignment_live_probe.tsv"
    write_tsv(
        tsv_path,
        rows,
        [
            "variant",
            "action_mode",
            "time_steps_changed",
            "time_steps_delta_mean",
            "robot_state_rewritten",
            "pre_endpoint_done_rate",
            "pre_endpoint_z_mean",
            "pre_body_error_mean",
            "pre_joint_pos_error_mean",
            "pre_joint_vel_error_mean",
            "policy_action_abs_mean",
            "step_done_rate",
            "step_done_count",
            "step_reward_mean",
            "step_action_abs_mean",
            "term_anchor_pos",
            "term_anchor_ori",
            "term_ee_body_pos",
            "post_endpoint_done_rate",
            "post_body_error_mean",
            "post_joint_pos_error_mean",
            "post_joint_vel_error_mean",
        ],
    )
    status_ok = run["returncode"] == 0 and worker_metrics.get("status") == "ok_robot_order_fk_phase_alignment_live_probe"
    metrics = dict(worker_metrics.get("summary", {}))
    summary = {
        "status": "ok_robot_order_fk_phase_alignment_live_probe" if status_ok else "failed_robot_order_fk_phase_alignment_live_probe",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "robot_order_fk_phase_alignment_live_probe",
        "scope": "Live IsaacLab gate comparing reset-time command phase alignment variants for robot-order FK tracking.",
        "config": {
            "target_gpu": TARGET_GPU,
            "num_envs": NUM_ENVS,
            "seed": SEED,
            "robot_usd": str(OFFICIAL_IMPORTER_USD),
            "motion_file": str(ROBOT_ORDER_MOTION_NPZ),
            "checkpoint": str(checkpoint),
        },
        "run": run,
        "worker_metrics": worker_metrics,
        "rows": rows,
        "metrics": metrics,
        "checks": {
            "worker_returned_zero": run["returncode"] == 0,
            "worker_status_ok": status_ok,
            "uses_official_importer_export_usd": worker_metrics.get("checks", {}).get("uses_official_importer_export_usd") is True,
            "uses_robot_order_fk_repaired_bundle": worker_metrics.get("checks", {}).get("uses_robot_order_fk_repaired_bundle") is True,
            "checkpoint_loaded": worker_metrics.get("checks", {}).get("checkpoint_loaded") is True,
            "all_variants_policy_and_zero_action_tested": worker_metrics.get("checks", {}).get(
                "all_variants_policy_and_zero_action_tested"
            )
            is True,
            "any_variant_improves_done_and_joint_velocity": worker_metrics.get("checks", {}).get(
                "any_variant_improves_done_and_joint_velocity"
            )
            is True,
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
            "recommended_full_eval_variant": metrics.get("recommended_full_eval_variant", ""),
            "next_step": (
                "If a phase variant improves both first-step done rate and joint velocity, run a full same-seed "
                "checkpoint eval for that variant before PPO. Otherwise return to motion/body target or "
                "termination semantics before launching another full PPO training run."
            ),
        },
        "outputs": {
            "json": str(OUT / "robot_order_fk_phase_alignment_live_probe.json"),
            "tsv": str(tsv_path),
            "md": str(OUT / "robot_order_fk_phase_alignment_live_probe.md"),
            "worker": str(worker),
            "worker_metrics": str(worker_metrics_path),
            "log": run["log_path"],
        },
    }
    json_path = OUT / "robot_order_fk_phase_alignment_live_probe.json"
    md_path = OUT / "robot_order_fk_phase_alignment_live_probe.md"
    write_json(json_path, summary)
    write_markdown(md_path, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(json_path),
                "metrics": metrics,
                "checks": summary["checks"],
            },
            sort_keys=True,
        )
    )
    if not status_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
