#!/usr/bin/env python3
"""Run paper-contract G1 tracking PPO training.

This wrapper keeps the BeyondMimic/whole_body_tracking formulas and parameters
from the paper-facing official code path:

- G1 armature/stiffness/damping/action-scale formulas from ``robots/g1.py``.
- Tracking reward weights/stds and termination thresholds from
  ``tracking_env_cfg.py``.
- RSL-RL PPO hyperparameters from ``G1FlatPPORunnerCfg``.

The only local adaptation is the already-audited public-resource input path:
the local official-importer-export G1 USD and the robot-order FK-repaired
public motion bundle. This is not an official BeyondMimic checkpoint.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = (
    ROOT
    / "reproduction/scripts/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run.py"
)
OUT = ROOT / "res/tracking/g1_official_importer_export_paper_contract_ppo_training_run"
LOG_DIR = ROOT / "logs/tracking_g1_official_importer_export_paper_contract_ppo_training_run"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_importer_export_paper_contract_ppo_training"
PARAM_SNAPSHOT = OUT / "paper_contract_tracking_parameters.json"
FINAL_JSON = OUT / "tracking_g1_official_importer_export_paper_contract_ppo_training_run.json"

DEFAULT_TARGET_GPUS = [4, 7]
DEFAULT_TOTAL_ENVS = 4096
DEFAULT_MAX_ITERATIONS = 30000
DEFAULT_NUM_STEPS_PER_ENV = 24
DEFAULT_SAVE_INTERVAL = 500
DEFAULT_SEED = 20260801
DEFAULT_ADAPTIVE_KERNEL_SIZE = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_robot_order_ppo_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base robot-order PPO script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_num_envs_per_rank(target_gpus: list[int], total_envs: int) -> int:
    world_size = max(1, len(target_gpus))
    if total_envs % world_size != 0:
        raise ValueError(f"BM_PAPER_CONTRACT_TOTAL_ENVS={total_envs} must divide world_size={world_size}")
    return total_envs // world_size


def paper_contract_snapshot(target_gpus: list[int], total_envs: int, num_envs_per_rank: int) -> dict[str, Any]:
    natural_freq = 10.0 * 2.0 * 3.1415926535
    damping_ratio = 2.0
    armature = {
        "ARMATURE_5020": 0.003609725,
        "ARMATURE_7520_14": 0.010177520,
        "ARMATURE_7520_22": 0.025101925,
        "ARMATURE_4010": 0.00425,
    }
    stiffness = {name.replace("ARMATURE", "STIFFNESS"): value * natural_freq**2 for name, value in armature.items()}
    damping = {
        name.replace("ARMATURE", "DAMPING"): 2.0 * damping_ratio * value * natural_freq
        for name, value in armature.items()
    }
    return {
        "status": "paper_contract_tracking_parameter_snapshot",
        "timestamp_utc": utc_now(),
        "claim_level": "paper_official_code_contract_snapshot_for_local_public_resource_training",
        "source_files": {
            "paper_method": str(ROOT / "reproduction/paper/source/tex/method.tex"),
            "paper_supplement": str(ROOT / "reproduction/paper/source/root.tex"),
            "official_g1_robot_config": str(
                ROOT
                / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
                "whole_body_tracking/robots/g1.py"
            ),
            "official_tracking_env_cfg": str(
                ROOT
                / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
                "whole_body_tracking/tasks/tracking/tracking_env_cfg.py"
            ),
            "official_ppo_cfg": str(
                ROOT
                / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
                "whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py"
            ),
        },
        "simulation": {
            "sim_dt": 0.005,
            "decimation": 4,
            "policy_control_frequency_hz": 50.0,
            "episode_length_s": 10.0,
            "target_total_envs": total_envs,
            "target_gpus": target_gpus,
            "num_envs_per_rank": num_envs_per_rank,
        },
        "motor_and_pd": {
            "natural_frequency_hz": 10.0,
            "natural_frequency_rad_s": natural_freq,
            "damping_ratio": damping_ratio,
            "armature": armature,
            "stiffness": stiffness,
            "damping": damping,
            "action_scale_formula": "G1_ACTION_SCALE[joint] = 0.25 * effort_limit_sim[joint] / stiffness[joint]",
            "action_semantics": "theta_sp = theta_default + action_scale * normalized_action",
        },
        "tracking_task": {
            "adaptive_sampling": {
                "kernel_size": DEFAULT_ADAPTIVE_KERNEL_SIZE,
                "lambda": 0.8,
                "uniform_ratio": 0.1,
                "alpha": 0.001,
                "reason": "Paper supplement describes a non-causal look-back kernel over u={0,1,2}; official source default is 1, so paper-contract runs set 3 explicitly.",
            },
            "anchor_body": "torso_link",
            "target_body_names": [
                "pelvis",
                "left_hip_roll_link",
                "left_knee_link",
                "left_ankle_roll_link",
                "right_hip_roll_link",
                "right_knee_link",
                "right_ankle_roll_link",
                "torso_link",
                "left_shoulder_roll_link",
                "left_elbow_link",
                "left_wrist_yaw_link",
                "right_shoulder_roll_link",
                "right_elbow_link",
                "right_wrist_yaw_link",
            ],
            "policy_observation_terms_order": [
                "generated_commands",
                "motion_anchor_pos_b",
                "motion_anchor_ori_b",
                "base_lin_vel",
                "base_ang_vel",
                "joint_pos_rel",
                "joint_vel_rel",
                "last_action",
            ],
            "policy_obs_dim": 160,
            "critic_obs_dim": 286,
            "action_dim": 29,
        },
        "reward_contract": {
            "motion_global_anchor_pos": {"weight": 0.5, "std": 0.3},
            "motion_global_anchor_ori": {"weight": 0.5, "std": 0.4},
            "motion_body_pos": {"weight": 1.0, "std": 0.3},
            "motion_body_ori": {"weight": 1.0, "std": 0.4},
            "motion_body_lin_vel": {"weight": 1.0, "std": 1.0},
            "motion_body_ang_vel": {"weight": 1.0, "std": 3.14},
            "action_rate_l2": {"weight": -0.1},
            "joint_limit": {"weight": -10.0, "soft_limit_factor": 0.9},
            "undesired_contacts": {"weight": -0.1, "threshold": 1.0},
        },
        "termination_contract": {
            "anchor_pos_z_only_threshold_m": 0.25,
            "anchor_ori_projected_gravity_threshold": 0.8,
            "ee_body_pos_z_only_threshold_m": 0.25,
            "ee_body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
        "ppo_contract": {
            "num_steps_per_env": DEFAULT_NUM_STEPS_PER_ENV,
            "max_iterations": DEFAULT_MAX_ITERATIONS,
            "save_interval": DEFAULT_SAVE_INTERVAL,
            "empirical_normalization": True,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "init_noise_std": 1.0,
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 0.001,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "local_public_resource_adaptation": {
            "uses_official_importer_export_usd": True,
            "uses_robot_order_fk_repaired_public_motion_bundle": True,
            "uses_endpoint_threshold_relaxation": False,
            "uses_mujoco_adapter": False,
            "why_not_official_paper_checkpoint": (
                "The official paper checkpoints and full DAgger/diffusion artifacts are not public. This run retrains "
                "locally using paper/official-code formulas and public-resource inputs."
            ),
        },
    }


def summarize_gpu_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "index": int(row.get("index", -1)),
                        "memory_used_mb": float(row.get("memory.used [MiB]", 0)),
                        "utilization_gpu_percent": float(row.get("utilization.gpu [%]", 0)),
                        "power_draw_w": float(row.get("power.draw [W]", 0)),
                    }
                )
            except (TypeError, ValueError):
                continue
    out: dict[str, Any] = {"exists": True, "row_count": len(rows), "by_gpu": {}}
    for gpu in sorted({row["index"] for row in rows}):
        group = [row for row in rows if row["index"] == gpu]
        out["by_gpu"][str(gpu)] = {
            "memory_used_mb_max": max(row["memory_used_mb"] for row in group),
            "utilization_gpu_percent_mean": sum(row["utilization_gpu_percent"] for row in group) / len(group),
            "power_draw_w_mean": sum(row["power_draw_w"] for row in group) / len(group),
        }
    return out


def patch_summary(summary: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    trained_ok = summary.get("status") == "ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_completed"
    summary["status"] = (
        "ok_official_importer_export_paper_contract_ppo_training_completed"
        if trained_ok
        else summary.get("status", "failed_official_importer_export_paper_contract_ppo_training")
    )
    summary["experiment_type"] = "tracking_official_importer_export_paper_contract_ppo_training_run"
    summary["scope"] = (
        "Paper-contract PPO tracking retraining with official BeyondMimic/whole_body_tracking motor formulas, "
        "reward terms, termination thresholds, observation/action contract, and RSL-RL PPO hyperparameters. "
        "The input asset/motion paths are local public-resource adaptations."
    )
    summary.setdefault("config", {})
    summary["config"].update(
        {
            "paper_contract_total_envs": snapshot["simulation"]["target_total_envs"],
            "paper_contract_num_envs_per_rank": snapshot["simulation"]["num_envs_per_rank"],
            "paper_contract_max_iterations": snapshot["ppo_contract"]["max_iterations"],
            "paper_contract_save_interval": snapshot["ppo_contract"]["save_interval"],
            "paper_contract_num_steps_per_env": snapshot["ppo_contract"]["num_steps_per_env"],
            "paper_contract_seed": int(os.environ.get("BM_PAPER_CONTRACT_SEED", str(DEFAULT_SEED))),
            "paper_contract_parameter_snapshot": str(PARAM_SNAPSHOT),
        }
    )
    summary.setdefault("inputs", {})
    summary["inputs"]["paper_contract_parameter_snapshot"] = str(PARAM_SNAPSHOT)
    summary.setdefault("outputs", {})
    summary["outputs"]["json"] = str(FINAL_JSON)
    summary["outputs"]["paper_contract_parameter_snapshot"] = str(PARAM_SNAPSHOT)
    telemetry = Path(summary.get("outputs", {}).get("gpu_metrics_csv", ""))
    summary["gpu_metrics_summary"] = summarize_gpu_metrics(telemetry)
    summary["interpretation"] = {
        "goal_complete": False,
        "paper_contract_tracking_training_complete": bool(trained_ok),
        "official_beyondmimic_checkpoint": False,
        "paper_level_tracking_eval_complete": False,
        "claim_level": "local_public_resource_paper_contract_tracking_training",
        "why_this_run": (
            "This is the requested clean rerun using paper/official-code motor, reward, action, termination, and PPO "
            "settings instead of ad-hoc MuJoCo video fixes."
        ),
        "why_not_paper_level": (
            "It uses local public-resource motion/asset adaptations and must be evaluated after training. It is not "
            "an official BeyondMimic checkpoint and does not by itself prove Fig.5/Fig.6 or real-robot behavior."
        ),
    }
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    target_gpus = [
        int(x)
        for x in os.environ.get("BM_PAPER_CONTRACT_TARGET_GPUS", ",".join(map(str, DEFAULT_TARGET_GPUS))).split(",")
        if x.strip()
    ]
    total_envs = int(os.environ.get("BM_PAPER_CONTRACT_TOTAL_ENVS", str(DEFAULT_TOTAL_ENVS)))
    num_envs_per_rank = compute_num_envs_per_rank(target_gpus, total_envs)
    max_iterations = int(os.environ.get("BM_PAPER_CONTRACT_MAX_ITERATIONS", str(DEFAULT_MAX_ITERATIONS)))
    save_interval = int(os.environ.get("BM_PAPER_CONTRACT_SAVE_INTERVAL", str(DEFAULT_SAVE_INTERVAL)))
    seed = int(os.environ.get("BM_PAPER_CONTRACT_SEED", str(DEFAULT_SEED)))
    adaptive_kernel_size = int(
        os.environ.get("BM_PAPER_CONTRACT_ADAPTIVE_KERNEL_SIZE", str(DEFAULT_ADAPTIVE_KERNEL_SIZE))
    )

    snapshot = paper_contract_snapshot(target_gpus, total_envs, num_envs_per_rank)
    snapshot["ppo_contract"]["max_iterations"] = max_iterations
    snapshot["ppo_contract"]["save_interval"] = save_interval
    snapshot["ppo_contract"]["seed"] = seed
    snapshot["tracking_task"]["adaptive_sampling"]["kernel_size"] = adaptive_kernel_size
    write_json(PARAM_SNAPSHOT, snapshot)

    module = load_base_module()
    module.OUT = OUT
    module.LOG_DIR = LOG_DIR
    module.RUN_ROOT = RUN_ROOT
    module.TARGET_GPUS = target_gpus
    module.DEFAULT_MAX_ITERATIONS = max_iterations
    module.DEFAULT_NUM_ENVS_PER_RANK = num_envs_per_rank
    module.DEFAULT_SEED = seed

    # Keep official save interval instead of the shorter report-oriented interval
    # used by earlier local resource-adjusted runs.
    module_base = module.load_base_module()
    worker = module_base.WORKER_CODE.replace(
        'agent_cfg.save_interval = max(1, min(50, agent_cfg.max_iterations))',
        'agent_cfg.save_interval = int(os.environ.get("BM_PPO_SAVE_INTERVAL", "500"))',
    )
    worker = worker.replace(
        "env_cfg.commands.motion.debug_vis = False",
        (
            "env_cfg.commands.motion.debug_vis = False\n"
            "    env_cfg.commands.motion.adaptive_kernel_size = int(os.environ.get(\"BM_ADAPTIVE_KERNEL_SIZE\", \"3\"))"
        ),
    )
    module_base.WORKER_CODE = worker
    old_load_base = module.load_base_module
    module.load_base_module = lambda: module_base

    os.environ["BM_ROBOT_ORDER_FK_REPAIRED_FULL_BUNDLE_PPO_MAX_ITERATIONS"] = str(max_iterations)
    os.environ["BM_ROBOT_ORDER_FK_REPAIRED_FULL_BUNDLE_PPO_NUM_ENVS_PER_RANK"] = str(num_envs_per_rank)
    os.environ["BM_ROBOT_ORDER_FK_REPAIRED_FULL_BUNDLE_PPO_SEED"] = str(seed)
    os.environ["BM_PPO_SAVE_INTERVAL"] = str(save_interval)
    os.environ["BM_ADAPTIVE_KERNEL_SIZE"] = str(adaptive_kernel_size)

    try:
        module.main()
    finally:
        module.load_base_module = old_load_base

    base_json = OUT / "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run.json"
    summary = patch_summary(load_json(base_json), snapshot)
    base_json.unlink(missing_ok=True)
    write_json(FINAL_JSON, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(FINAL_JSON),
                "parameter_snapshot": str(PARAM_SNAPSHOT),
                "attempted_training": summary.get("run", {}).get("attempted_training"),
                "checkpoint_count": summary.get("run", {}).get("checkpoint_count"),
                "target_gpus": target_gpus,
                "total_envs": total_envs,
                "max_iterations": max_iterations,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
