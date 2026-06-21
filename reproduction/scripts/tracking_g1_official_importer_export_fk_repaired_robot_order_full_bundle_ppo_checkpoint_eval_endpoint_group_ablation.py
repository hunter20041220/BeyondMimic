#!/usr/bin/env python3
"""Run endpoint-group termination ablations for the robot-order FK PPO eval.

The previous same-seed full eval relaxed the whole ``ee_body_pos`` termination
term and showed that endpoint termination dominates the done count. This script
keeps the same checkpoint, seed, reset-target refresh, robot asset, and public
motion bundle, but edits the active ``ee_body_pos`` body list per variant:

- ankles_only: keep ankle endpoint termination active, remove wrists.
- wrists_only: keep wrist endpoint termination active, remove ankles.
- all_relaxed: keep all endpoint names but relax the threshold.

This is diagnostic evidence only. Any variant that removes or relaxes official
termination bodies is not a paper-level tracking metric.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py"
OUT = (
    ROOT
    / "res/tracking/"
    "g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_group_ablation"
)
LOG_DIR = (
    ROOT
    / "logs/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_group_ablation"
)
RUN_ROOT = (
    ROOT
    / "res/runs/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_group_ablation"
)
TRAINING_RUN_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_run.json"
)
OLD_EVAL_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval.json"
)
TARGET_REFRESH_EVAL_JSON = (
    ROOT
    / "res/tracking/"
    "g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance.json"
)
ALL_RELAXED_EVAL_JSON = (
    ROOT
    / "res/tracking/"
    "g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_ee_termination_ablation/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_ee_termination_ablation.json"
)
OFFICIAL_IMPORTER_USD = ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda"
ROBOT_ORDER_BUNDLE_AUDIT = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/"
    "tracking_g1_official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz.json"
)
ROBOT_ORDER_BUNDLE_NPZ = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/"
    "official_csv_loop_full_public_motion_bundle_fk_repaired_robot_order.npz"
)

TARGET_GPUS = [4, 7]
DEFAULT_SEED = 20260721
DEFAULT_NUM_ENVS = 2048
DEFAULT_EVAL_STEPS = 299
ORIGINAL_ENDPOINT_THRESHOLD = 0.25
RELAXED_ENDPOINT_THRESHOLD = float(os.environ.get("BM_ENDPOINT_GROUP_ABLATION_RELAXED_THRESHOLD", "1000.0"))

ANKLE_ENDPOINTS = ["left_ankle_roll_link", "right_ankle_roll_link"]
WRIST_ENDPOINTS = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
ALL_ENDPOINTS = [*ANKLE_ENDPOINTS, *WRIST_ENDPOINTS]

VARIANTS = [
    {
        "name": "ankles_only",
        "active_ee_body_names": ANKLE_ENDPOINTS,
        "threshold": ORIGINAL_ENDPOINT_THRESHOLD,
        "description": "Only ankle endpoint termination is active at the original threshold.",
    },
    {
        "name": "wrists_only",
        "active_ee_body_names": WRIST_ENDPOINTS,
        "threshold": ORIGINAL_ENDPOINT_THRESHOLD,
        "description": "Only wrist endpoint termination is active at the original threshold.",
    },
    {
        "name": "all_relaxed",
        "active_ee_body_names": ALL_ENDPOINTS,
        "threshold": RELAXED_ENDPOINT_THRESHOLD,
        "description": "All endpoint bodies remain active but the threshold is relaxed.",
    },
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_base_module(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base PPO eval script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_base_compatible_training_summary() -> Path:
    training = load_json(TRAINING_RUN_JSON)
    compatible = dict(training)
    compatible["status"] = "ok_resource_adjusted_ppo_training_completed"
    compatible.setdefault("inputs", {})
    compatible["inputs"]["original_robot_order_fk_repaired_full_bundle_training_run_json"] = str(TRAINING_RUN_JSON)
    compatible.setdefault("interpretation", {})
    compatible["interpretation"]["base_compatibility_shim"] = (
        "This copy adapts the robot-order FK-repaired training status enum for the shared checkpoint-eval harness. "
        "The authoritative training audit remains the robot-order FK-repaired training JSON."
    )
    shim_path = OUT / "base_compatible_robot_order_fk_repaired_full_bundle_training_run_for_endpoint_group_ablation.json"
    shim_path.write_text(json.dumps(compatible, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return shim_path


def patch_worker_code(worker_code: str) -> str:
    worker_code = worker_code.replace(
        "import whole_body_tracking.tasks  # noqa: F401\n",
        "import whole_body_tracking.tasks  # noqa: F401\n"
        "    from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat\n",
    )
    old = """    env = gym.make(\"Tracking-Flat-G1-v0\", cfg=env_cfg, render_mode=None)
    print(f\"BM_SENTINEL:eval:env_created:num_envs={env.unwrapped.num_envs}\", flush=True)
    vec_env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    loaded_infos = runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
    obs, extras = vec_env.get_observations()
    command = vec_env.unwrapped.command_manager.get_term(\"motion\")
"""
    new = """    env = gym.make(\"Tracking-Flat-G1-v0\", cfg=env_cfg, render_mode=None)
    print(f\"BM_SENTINEL:eval:env_created:num_envs={env.unwrapped.num_envs}\", flush=True)
    vec_env = RslRlVecEnvWrapper(env)
    command = vec_env.unwrapped.command_manager.get_term(\"motion\")
    endpoint_names = [
        \"left_ankle_roll_link\",
        \"right_ankle_roll_link\",
        \"left_wrist_yaw_link\",
        \"right_wrist_yaw_link\",
    ]
    ankle_endpoint_names = [\"left_ankle_roll_link\", \"right_ankle_roll_link\"]
    wrist_endpoint_names = [\"left_wrist_yaw_link\", \"right_wrist_yaw_link\"]

    def stats_tensor(tensor):
        if tensor.numel() == 0:
            return {\"count\": 0, \"mean\": None, \"min\": None, \"max\": None}
        flat = tensor.detach().float().reshape(-1).cpu()
        finite = flat[torch.isfinite(flat)]
        if finite.numel() == 0:
            return {\"count\": 0, \"mean\": None, \"min\": None, \"max\": None}
        return {
            \"count\": int(finite.numel()),
            \"mean\": float(finite.mean().item()),
            \"min\": float(finite.min().item()),
            \"max\": float(finite.max().item()),
        }

    def endpoint_group_stats(names, threshold=0.25):
        body_names = list(command.cfg.body_names)
        indexes = [body_names.index(name) for name in names if name in body_names]
        if not indexes:
            zeros = torch.zeros(vec_env.unwrapped.num_envs, dtype=torch.bool, device=vec_env.unwrapped.device)
            return {
                \"names\": names,
                \"names_present\": [],
                \"indexes\": [],
                \"done_count\": 0,
                \"done_rate\": 0.0,
                \"z_error_m\": {\"count\": 0, \"mean\": None, \"min\": None, \"max\": None},
            }
        body_target = command.body_pos_relative_w.detach()
        robot_body = command.robot_body_pos_w.detach()
        z_error = torch.abs(body_target[:, indexes, 2] - robot_body[:, indexes, 2])
        done = torch.any(z_error > threshold, dim=-1)
        return {
            \"names\": names,
            \"names_present\": [name for name in names if name in body_names],
            \"indexes\": indexes,
            \"done_count\": int(done.detach().cpu().sum().item()),
            \"done_rate\": float(done.float().mean().detach().cpu().item()),
            \"z_error_m\": stats_tensor(z_error),
        }

    def endpoint_snapshot(label):
        body_error = torch.linalg.norm(command.body_pos_relative_w.detach() - command.robot_body_pos_w.detach(), dim=-1)
        return {
            \"label\": label,
            \"all\": endpoint_group_stats(endpoint_names, threshold=0.25),
            \"ankles\": endpoint_group_stats(ankle_endpoint_names, threshold=0.25),
            \"wrists\": endpoint_group_stats(wrist_endpoint_names, threshold=0.25),
            \"body_error_m\": stats_tensor(body_error),
            \"time_steps\": stats_tensor(command.time_steps.detach().float()),
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

    reset_target_refresh_before = endpoint_snapshot(\"after_wrapper_reset_before_target_refresh\")
    time_steps_before_refresh = command.time_steps.detach().clone()
    refresh_motion_targets_no_advance()
    time_steps_after_refresh = command.time_steps.detach().clone()
    vec_env.unwrapped.obs_buf = vec_env.unwrapped.observation_manager.compute()
    reset_target_refresh_after = endpoint_snapshot(\"after_wrapper_reset_target_refresh_no_advance\")
    reset_target_refresh_after[\"time_steps_unchanged_by_refresh\"] = bool(
        torch.equal(time_steps_before_refresh, time_steps_after_refresh)
    )

    active_ee_body_names = [name for name in os.environ.get(\"BM_ENDPOINT_GROUP_ACTIVE_EE_BODY_NAMES\", \"\").split(\",\") if name]
    ee_threshold = float(os.environ.get(\"BM_ENDPOINT_GROUP_EE_BODY_POS_THRESHOLD\", \"0.25\"))
    ee_cfg = vec_env.unwrapped.termination_manager.get_term_cfg(\"ee_body_pos\")
    original_ee_body_pos_threshold = float(ee_cfg.params.get(\"threshold\", 0.25))
    original_ee_body_pos_body_names = list(ee_cfg.params.get(\"body_names\", []))
    ee_cfg.params = dict(ee_cfg.params)
    ee_cfg.params[\"body_names\"] = active_ee_body_names
    ee_cfg.params[\"threshold\"] = ee_threshold
    vec_env.unwrapped.termination_manager.set_term_cfg(\"ee_body_pos\", ee_cfg)
    ee_cfg_after = vec_env.unwrapped.termination_manager.get_term_cfg(\"ee_body_pos\")
    ee_body_pos_threshold_after = float(ee_cfg_after.params.get(\"threshold\"))
    ee_body_pos_body_names_after = list(ee_cfg_after.params.get(\"body_names\", []))

    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    loaded_infos = runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
    obs, extras = vec_env.get_observations()
"""
    if old not in worker_code:
        raise RuntimeError("Base worker code shape changed; cannot inject endpoint group ablation.")
    worker_code = worker_code.replace(old, new)
    worker_code = worker_code.replace(
        """    action_abs_means = []
    action_abs_maxs = []
    metric_series = {}
""",
        """    action_abs_means = []
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
""",
    )
    worker_code = worker_code.replace(
        """        "action_abs_max",
    ] + metric_names
""",
        """        "action_abs_max",
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
""",
    )
    worker_code = worker_code.replace(
        """                row = {
                    "step": step,
                    "reward_mean": float(rew.mean().detach().cpu()),
                    "reward_min": float(rew.min().detach().cpu()),
                    "reward_max": float(rew.max().detach().cpu()),
                    "done_count": int(dones.sum().detach().cpu()),
                    "timeout_count": int(step_extras.get("time_outs", torch.zeros_like(dones)).sum().detach().cpu()),
                    "action_abs_mean": float(actions.abs().mean().detach().cpu()),
                    "action_abs_max": float(actions.abs().max().detach().cpu()),
                }
""",
        """                endpoint_after_step = endpoint_snapshot(f"after_step_{step}")
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
""",
    )
    worker_code = worker_code.replace(
        """                action_abs_means.append(row["action_abs_mean"])
                action_abs_maxs.append(row["action_abs_max"])
                for key, value in step_extras.get("log", {}).items():
""",
        """                action_abs_means.append(row["action_abs_mean"])
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
""",
    )
    worker_code = worker_code.replace(
        """        "action_abs_mean_over_steps": summarize(action_abs_means),
        "action_abs_max_over_steps": summarize(action_abs_maxs),
        "motion_metrics": {name: summarize(values) for name, values in metric_series.items()},
""",
        """        "action_abs_mean_over_steps": summarize(action_abs_means),
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
""",
    )
    return worker_code


def first_timeseries_row(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        try:
            row = next(reader)
        except StopIteration:
            return {}
    return convert_row(row)


def convert_row(row: dict[str, str]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in row.items():
        try:
            converted[key] = float(value)
        except (TypeError, ValueError):
            converted[key] = value
    return converted


def post_step0_rate(timeseries: Path, num_envs: int, column: str) -> float | None:
    if not timeseries.is_file():
        return None
    total = 0.0
    steps = 0
    with timeseries.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if int(float(row["step"])) == 0:
                continue
            total += float(row[column])
            steps += 1
    return total / float(max(steps * num_envs, 1)) if steps else None


def run_variant(variant: dict[str, Any], compatible_training_summary: Path) -> dict[str, Any]:
    variant_name = variant["name"]
    variant_out = OUT / variant_name
    variant_log = LOG_DIR / variant_name
    variant_runs = RUN_ROOT / variant_name
    variant_out.mkdir(parents=True, exist_ok=True)
    variant_log.mkdir(parents=True, exist_ok=True)
    variant_runs.mkdir(parents=True, exist_ok=True)

    module = load_base_module(f"bm_endpoint_group_ablation_{variant_name}")
    module.OUT = variant_out
    module.LOG_DIR = variant_log
    module.RUN_ROOT = variant_runs
    module.ENRICHED_USD = OFFICIAL_IMPORTER_USD
    module.CSV_MOTION_NPZ = ROBOT_ORDER_BUNDLE_NPZ
    module.TRAINING_RUN_JSON = compatible_training_summary
    module.CANDIDATE_GPUS = TARGET_GPUS
    module.NUM_ENVS = int(os.environ.get("BM_ENDPOINT_GROUP_ABLATION_NUM_ENVS", str(DEFAULT_NUM_ENVS)))
    module.SEED = int(os.environ.get("BM_ENDPOINT_GROUP_ABLATION_SEED", str(DEFAULT_SEED)))
    module.WORKER_CODE = patch_worker_code(module.WORKER_CODE)

    old_env = {
        key: os.environ.get(key)
        for key in ["BM_ENDPOINT_GROUP_ACTIVE_EE_BODY_NAMES", "BM_ENDPOINT_GROUP_EE_BODY_POS_THRESHOLD"]
    }
    os.environ["BM_ENDPOINT_GROUP_ACTIVE_EE_BODY_NAMES"] = ",".join(variant["active_ee_body_names"])
    os.environ["BM_ENDPOINT_GROUP_EE_BODY_POS_THRESHOLD"] = str(variant["threshold"])
    try:
        module.main()
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    base_json = variant_out / "tracking_g1_resource_adjusted_ppo_checkpoint_eval.json"
    base_summary = load_json(base_json)
    eval_ok = base_summary.get("status") == "ok_resource_adjusted_ppo_checkpoint_eval_completed"
    metrics = base_summary.get("run", {}).get("metrics", {})
    timeseries = Path(base_summary.get("outputs", {}).get("timeseries_csv", ""))
    num_envs = int(module.NUM_ENVS)
    variant_summary = {
        "variant": variant_name,
        "description": variant["description"],
        "status": "ok" if eval_ok else "failed",
        "active_ee_body_names": variant["active_ee_body_names"],
        "active_threshold": variant["threshold"],
        "summary_json": str(base_json),
        "worker_script": str(variant_out / "tracking_g1_resource_adjusted_ppo_checkpoint_eval_worker.py"),
        "timeseries_csv": str(timeseries),
        "metrics_json": base_summary.get("outputs", {}).get("metrics_json", ""),
        "gpu_metrics_csv": base_summary.get("outputs", {}).get("gpu_metrics_csv", ""),
        "done_count_total": metrics.get("done_count_total"),
        "total_env_steps": metrics.get("total_env_steps"),
        "done_rate": (
            metrics.get("done_count_total") / metrics.get("total_env_steps")
            if metrics.get("done_count_total") is not None and metrics.get("total_env_steps")
            else None
        ),
        "post_step0_done_rate": post_step0_rate(timeseries, num_envs, "done_count"),
        "post_step0_active_ee_body_pos_rate": post_step0_rate(timeseries, num_envs, "term_ee_body_pos_count"),
        "post_step0_manual_all_endpoint_rate_0p25": post_step0_rate(
            timeseries, num_envs, "manual_all_endpoint_done_count_0p25"
        ),
        "post_step0_manual_ankle_endpoint_rate_0p25": post_step0_rate(
            timeseries, num_envs, "manual_ankle_endpoint_done_count_0p25"
        ),
        "post_step0_manual_wrist_endpoint_rate_0p25": post_step0_rate(
            timeseries, num_envs, "manual_wrist_endpoint_done_count_0p25"
        ),
        "step0": first_timeseries_row(timeseries),
        "base_summary": base_summary,
    }
    final_variant_json = variant_out / f"{variant_name}_endpoint_group_ablation.json"
    final_variant_json.write_text(json.dumps(variant_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    variant_summary["variant_json"] = str(final_variant_json)
    return variant_summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    compatible_training_summary = make_base_compatible_training_summary()
    training = load_json(TRAINING_RUN_JSON)
    bundle = load_json(ROBOT_ORDER_BUNDLE_AUDIT)
    target_refresh = load_json(TARGET_REFRESH_EVAL_JSON)
    all_relaxed = load_json(ALL_RELAXED_EVAL_JSON)

    variant_rows = []
    for variant in VARIANTS:
        variant_rows.append(run_variant(variant, compatible_training_summary))

    target_refresh_metrics = target_refresh.get("run", {}).get("metrics", {})
    target_refresh_done_rate = (
        target_refresh_metrics.get("done_count_total") / target_refresh_metrics.get("total_env_steps")
        if target_refresh_metrics.get("done_count_total") is not None
        and target_refresh_metrics.get("total_env_steps")
        else None
    )
    all_relaxed_done_rate = all_relaxed.get("comparison_to_baselines", {}).get("ee_ablation_done_rate")
    best_by_done = sorted(
        [row for row in variant_rows if row["done_rate"] is not None],
        key=lambda row: row["done_rate"],
    )
    ankle = next(row for row in variant_rows if row["variant"] == "ankles_only")
    wrist = next(row for row in variant_rows if row["variant"] == "wrists_only")
    dominant_group = "undetermined"
    if ankle["post_step0_active_ee_body_pos_rate"] is not None and wrist["post_step0_active_ee_body_pos_rate"] is not None:
        if ankle["post_step0_active_ee_body_pos_rate"] > wrist["post_step0_active_ee_body_pos_rate"] * 1.2:
            dominant_group = "ankles"
        elif wrist["post_step0_active_ee_body_pos_rate"] > ankle["post_step0_active_ee_body_pos_rate"] * 1.2:
            dominant_group = "wrists"
        else:
            dominant_group = "both_or_coupled"

    final_json = OUT / "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_group_ablation.json"
    rows_csv = OUT / "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_group_ablation_rows.csv"
    with rows_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "variant",
            "active_ee_body_names",
            "active_threshold",
            "done_rate",
            "post_step0_done_rate",
            "post_step0_active_ee_body_pos_rate",
            "post_step0_manual_all_endpoint_rate_0p25",
            "post_step0_manual_ankle_endpoint_rate_0p25",
            "post_step0_manual_wrist_endpoint_rate_0p25",
            "variant_json",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in variant_rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    compact_rows = [
        {key: value for key, value in row.items() if key != "base_summary"}
        for row in variant_rows
    ]
    summary = {
        "status": "ok_endpoint_group_ablation_completed"
        if all(row["status"] == "ok" for row in variant_rows)
        else "failed_endpoint_group_ablation",
        "experiment_type": "tracking_endpoint_group_ablation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Same-seed 2048-env x 299-step robot-order FK PPO eval variants that isolate ankle vs wrist "
            "membership in the ee_body_pos termination term after reset-target refresh."
        ),
        "config": {
            "seed": int(os.environ.get("BM_ENDPOINT_GROUP_ABLATION_SEED", str(DEFAULT_SEED))),
            "num_envs": int(os.environ.get("BM_ENDPOINT_GROUP_ABLATION_NUM_ENVS", str(DEFAULT_NUM_ENVS))),
            "eval_steps": DEFAULT_EVAL_STEPS,
            "target_gpus": TARGET_GPUS,
            "original_endpoint_threshold": ORIGINAL_ENDPOINT_THRESHOLD,
            "relaxed_endpoint_threshold": RELAXED_ENDPOINT_THRESHOLD,
            "variants": deepcopy(VARIANTS),
        },
        "inputs": {
            "training_run_json": str(TRAINING_RUN_JSON),
            "target_refresh_eval_json": str(TARGET_REFRESH_EVAL_JSON),
            "all_endpoint_relaxed_eval_json": str(ALL_RELAXED_EVAL_JSON),
            "robot_order_bundle_audit": str(ROBOT_ORDER_BUNDLE_AUDIT),
            "robot_order_motion_npz": str(ROBOT_ORDER_BUNDLE_NPZ),
            "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
        },
        "input_checks": {
            "training_run_completed": training.get("status")
            == "ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_completed",
            "target_refresh_eval_completed": target_refresh.get("status")
            == "ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance_completed",
            "all_endpoint_relaxed_eval_completed": all_relaxed.get("status")
            == "ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_ee_termination_ablation_completed",
            "bundle_status_ok": bundle.get("status") == "ok_fk_repaired_robot_order_motion_npz",
            "motion_count_40": bundle.get("metrics", {}).get("motion_count") == 40,
            "total_frames_11960": bundle.get("metrics", {}).get("total_frames") == 11960,
        },
        "comparison_to_baselines": {
            "target_refresh_done_rate": target_refresh_done_rate,
            "all_endpoint_relaxed_done_rate": all_relaxed_done_rate,
            "best_variant_by_done_rate": best_by_done[0]["variant"] if best_by_done else "",
            "dominant_endpoint_group": dominant_group,
            "ankles_only_done_rate": ankle["done_rate"],
            "wrists_only_done_rate": wrist["done_rate"],
            "ankles_only_active_ee_body_pos_post_step0_rate": ankle["post_step0_active_ee_body_pos_rate"],
            "wrists_only_active_ee_body_pos_post_step0_rate": wrist["post_step0_active_ee_body_pos_rate"],
        },
        "variant_rows": compact_rows,
        "checks": {
            "all_variants_completed": all(row["status"] == "ok" for row in variant_rows),
            "same_seed_scope": all(
                row["base_summary"].get("config", {}).get("seed") == DEFAULT_SEED
                and row["base_summary"].get("config", {}).get("num_envs") == DEFAULT_NUM_ENVS
                and row["base_summary"].get("config", {}).get("eval_steps") == DEFAULT_EVAL_STEPS
                for row in variant_rows
            ),
            "ankle_and_wrist_variants_present": {row["variant"] for row in variant_rows}
            == {"ankles_only", "wrists_only", "all_relaxed"},
            "records_manual_endpoint_group_rates": all(
                row["post_step0_manual_ankle_endpoint_rate_0p25"] is not None
                and row["post_step0_manual_wrist_endpoint_rate_0p25"] is not None
                for row in variant_rows
            ),
            "does_not_claim_paper_level_tracking": True,
            "does_not_claim_goal_complete": True,
            "does_not_claim_real_robot": True,
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_tracking_eval_complete": False,
            "dominant_endpoint_group": dominant_group,
            "why_this_is_mainline": (
                "This separates ankle and wrist endpoint contributions to the current tracking done count, so the "
                "next fix can target motion/termination semantics instead of blindly rerunning PPO."
            ),
            "why_not_paper_level": (
                "These variants remove or relax official endpoint termination bodies. They are diagnostics only, not "
                "valid paper tracking metrics, not DAgger/VAE/diffusion evidence, and not real-robot evidence."
            ),
        },
        "outputs": {
            "json": str(final_json),
            "rows_csv": str(rows_csv),
            "base_compatible_training_json": str(compatible_training_summary),
        },
    }
    final_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(final_json),
                "dominant_endpoint_group": dominant_group,
                "comparison_to_baselines": summary["comparison_to_baselines"],
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok_endpoint_group_ablation_completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
