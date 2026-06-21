#!/usr/bin/env python3
"""Live ankle/wrist endpoint target-alignment probe for robot-order FK tracking.

The latest endpoint-group ablation showed that wrist endpoint termination is a
dominant source of done events.  This probe inspects the actual target tensors
used by ``ee_body_pos`` inside a live IsaacLab ``Tracking-Flat-G1-v0`` process:
``body_pos_w``, ``body_pos_relative_w``, and ``robot_body_pos_w`` for ankles and
wrists, before/after no-advance target refresh and after one zero/policy step.

It is a data-quality diagnostic only.  It does not train PPO and it does not
claim paper-level tracking performance.
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
OUT = ROOT / "res/tracking/robot_order_fk_wrist_endpoint_alignment_live_probe"
LOG_DIR = ROOT / "logs/tracking_robot_order_fk_wrist_endpoint_alignment_live_probe"
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

TARGET_GPU = int(os.environ.get("BM_WRIST_ENDPOINT_GPU", "4"))
NUM_ENVS = int(os.environ.get("BM_WRIST_ENDPOINT_NUM_ENVS", "256"))
SEED = int(os.environ.get("BM_WRIST_ENDPOINT_SEED", "20260770"))
STALL_SECONDS = int(os.environ.get("BM_WRIST_ENDPOINT_STALL_SECONDS", "900"))
THRESHOLD_M = 0.25


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
            checkpoint = Path(checkpoints[-1])
            if checkpoint.is_file():
                return checkpoint
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
            "BM_THRESHOLD_M": str(THRESHOLD_M),
        }
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def run_worker(worker: Path, worker_metrics: Path, checkpoint: Path) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "robot_order_fk_wrist_endpoint_alignment_live_probe.log"
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
                    proc.wait(timeout=30)
                break
        for line in proc.stdout:
            stdout_tail.append(line.rstrip())
            stdout_tail = stdout_tail[-120:]
            log_file.write(line)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "duration_seconds": time.time() - start,
        "log_path": str(log_path),
        "stdout_tail": stdout_tail,
    }


def flat_rows(worker: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snap in worker.get("snapshots", []):
        for group in ["ankles", "wrists", "all_endpoints"]:
            stats = snap.get(group, {})
            rows.append(
                {
                    "label": snap.get("label"),
                    "group": group,
                    "present_names": ",".join(stats.get("present_names", [])),
                    "rel_z_error_mean": (stats.get("rel_z_error_m") or {}).get("mean"),
                    "rel_z_error_max": (stats.get("rel_z_error_m") or {}).get("max"),
                    "raw_z_error_mean": (stats.get("raw_z_error_m") or {}).get("mean"),
                    "raw_z_error_max": (stats.get("raw_z_error_m") or {}).get("max"),
                    "rel_xyz_error_mean": (stats.get("rel_xyz_error_m") or {}).get("mean"),
                    "raw_xyz_error_mean": (stats.get("raw_xyz_error_m") or {}).get("mean"),
                    "target_relative_z_mean": (stats.get("target_relative_z_m") or {}).get("mean"),
                    "target_raw_z_mean": (stats.get("target_raw_z_m") or {}).get("mean"),
                    "robot_z_mean": (stats.get("robot_z_m") or {}).get("mean"),
                    "rel_z_done_count": stats.get("rel_z_done_count"),
                    "rel_z_done_rate": stats.get("rel_z_done_rate"),
                    "step_done_rate": snap.get("step_done_rate"),
                    "reward_mean": snap.get("reward_mean"),
                    "joint_vel_error_mean": (snap.get("joint_vel_error") or {}).get("mean"),
                }
            )
    return rows


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    metrics = summary.get("metrics", {})
    rows = summary.get("rows", [])
    lines = [
        "# Robot-Order FK Wrist Endpoint Alignment Live Probe",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- Worker status: `{summary.get('worker_metrics', {}).get('status')}`",
        f"- Diagnosis: `{metrics.get('diagnosis')}`",
        f"- GPU: `{summary['config']['target_gpu']}`",
        f"- Num envs: `{summary['config']['num_envs']}`",
        f"- Threshold: `{THRESHOLD_M}` m",
        "",
        "## Key Metrics",
        "",
        f"- Refresh wrist done rate: `{metrics.get('refresh_wrist_done_rate')}`",
        f"- Refresh ankle done rate: `{metrics.get('refresh_ankle_done_rate')}`",
        f"- Refresh wrist minus ankle done rate: `{metrics.get('refresh_wrist_minus_ankle_done_rate')}`",
        f"- Policy-step wrist done rate: `{metrics.get('policy_step_wrist_done_rate')}`",
        f"- Policy-step ankle done rate: `{metrics.get('policy_step_ankle_done_rate')}`",
        f"- Policy-step done rate: `{metrics.get('policy_step_done_rate')}`",
        "",
        "## Snapshot Table",
        "",
        "| label | group | rel z mean | rel z max | raw z mean | target rel z mean | robot z mean | rel done rate | step done |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {label} | {group} | {rel_z_error_mean} | {rel_z_error_max} | {raw_z_error_mean} | "
            "{target_relative_z_mean} | {robot_z_mean} | {rel_z_done_rate} | {step_done_rate} |".format(**row)
        )
    lines.extend(["", "## Checks", ""])
    for key, value in summary.get("checks", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "This is a live local tracking data-quality diagnostic. It does not train PPO, does not claim "
            "paper-level tracking, and does not use real robot hardware.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = latest_checkpoint()
    worker = OUT / "robot_order_fk_wrist_endpoint_alignment_live_probe_worker.py"
    worker_metrics_path = OUT / "robot_order_fk_wrist_endpoint_alignment_live_probe_worker_metrics.json"
    worker.write_text(WORKER_CODE, encoding="utf-8")
    run = run_worker(worker, worker_metrics_path, checkpoint)
    worker_metrics = load_json(worker_metrics_path)
    status_ok = run["returncode"] == 0 and worker_metrics.get("status") == "ok_robot_order_fk_wrist_endpoint_alignment_live_probe"
    rows = flat_rows(worker_metrics)
    tsv_path = OUT / "robot_order_fk_wrist_endpoint_alignment_live_probe.tsv"
    write_tsv(
        tsv_path,
        rows,
        [
            "label",
            "group",
            "present_names",
            "rel_z_error_mean",
            "rel_z_error_max",
            "raw_z_error_mean",
            "raw_z_error_max",
            "rel_xyz_error_mean",
            "raw_xyz_error_mean",
            "target_relative_z_mean",
            "target_raw_z_mean",
            "robot_z_mean",
            "rel_z_done_count",
            "rel_z_done_rate",
            "step_done_rate",
            "reward_mean",
            "joint_vel_error_mean",
        ],
    )
    metrics = worker_metrics.get("summary", {})
    summary = {
        "status": (
            "ok_robot_order_fk_wrist_endpoint_alignment_live_probe"
            if status_ok
            else "failed_robot_order_fk_wrist_endpoint_alignment_live_probe"
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "robot_order_fk_wrist_endpoint_alignment_live_probe",
        "scope": (
            "Live IsaacLab data-quality probe separating ankle and wrist endpoint target tensors for the robot-order "
            "FK-repaired tracking chain."
        ),
        "config": {
            "target_gpu": TARGET_GPU,
            "num_envs": NUM_ENVS,
            "seed": SEED,
            "threshold_m": THRESHOLD_M,
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
            "uses_official_importer_export_usd": worker_metrics.get("checks", {}).get("uses_official_importer_export_usd")
            is True,
            "uses_robot_order_fk_repaired_bundle": worker_metrics.get("checks", {}).get(
                "uses_robot_order_fk_repaired_bundle"
            )
            is True,
            "checkpoint_loaded": worker_metrics.get("checks", {}).get("checkpoint_loaded") is True,
            "ankle_names_found": worker_metrics.get("checks", {}).get("ankle_names_found") is True,
            "wrist_names_found": worker_metrics.get("checks", {}).get("wrist_names_found") is True,
            "time_steps_unchanged_by_refresh": worker_metrics.get("checks", {}).get(
                "time_steps_unchanged_by_refresh"
            )
            is True,
            "records_body_pos_w": worker_metrics.get("checks", {}).get("records_body_pos_w") is True,
            "records_body_pos_relative_w": worker_metrics.get("checks", {}).get("records_body_pos_relative_w") is True,
            "records_robot_body_pos_w": worker_metrics.get("checks", {}).get("records_robot_body_pos_w") is True,
            "records_ankle_and_wrist_groups": worker_metrics.get("checks", {}).get(
                "records_ankle_and_wrist_groups"
            )
            is True,
            "wrist_dominance_classified": worker_metrics.get("checks", {}).get("wrist_dominance_classified") is True,
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
            "diagnosis": metrics.get("diagnosis"),
            "next_step": (
                "Use this probe to decide whether to repair wrist body target generation, FK height/body order, or "
                "ee_body_pos termination before launching another full PPO run."
            ),
        },
        "outputs": {
            "json": str(OUT / "robot_order_fk_wrist_endpoint_alignment_live_probe.json"),
            "tsv": str(tsv_path),
            "md": str(OUT / "robot_order_fk_wrist_endpoint_alignment_live_probe.md"),
            "worker": str(worker),
            "worker_metrics": str(worker_metrics_path),
            "log": run["log_path"],
        },
    }
    json_path = OUT / "robot_order_fk_wrist_endpoint_alignment_live_probe.json"
    md_path = OUT / "robot_order_fk_wrist_endpoint_alignment_live_probe.md"
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
