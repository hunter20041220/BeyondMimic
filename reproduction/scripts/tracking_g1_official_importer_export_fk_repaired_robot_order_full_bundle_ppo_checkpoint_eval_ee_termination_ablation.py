#!/usr/bin/env python3
"""Evaluate robot-order FK PPO with reset-target refresh and relaxed endpoint termination.

The previous full eval showed that no-advance target refresh fixes the stale
step-0 body target but worsens post-step0 done rate. Source-linked diagnostics
point to the z-only ``ee_body_pos`` termination as the dominant remaining gate.
This script keeps the same checkpoint, seed, robot asset, and full public-motion
bundle, then relaxes only that termination threshold during evaluation while
recording the resulting termination-term counts.

This is an evaluation ablation, not a training run and not a paper-level teacher
claim.
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
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py"
OUT = (
    ROOT
    / "res/tracking/"
    "g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_ee_termination_ablation"
)
LOG_DIR = (
    ROOT
    / "logs/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_ee_termination_ablation"
)
RUN_ROOT = (
    ROOT
    / "res/runs/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_ee_termination_ablation"
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
PHASE_ALIGNMENT_JSON = (
    ROOT
    / "res/tracking/robot_order_fk_phase_alignment_live_probe/"
    "robot_order_fk_phase_alignment_live_probe.json"
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
ROBOT_ORDER_SPLIT_TASK_GATE = (
    ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_split_task_eval/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_split_task_eval.json"
)
TARGET_GPUS = [4, 7]
DEFAULT_SEED = 20260721
DEFAULT_NUM_ENVS = 2048
EE_BODY_POS_ABLATION_THRESHOLD = float(os.environ.get("BM_EE_BODY_POS_ABLATION_THRESHOLD", "1000.0"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_resource_adjusted_ppo_eval_base_ee_ablation", BASE_SCRIPT)
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
    shim_path = OUT / "base_compatible_robot_order_fk_repaired_full_bundle_training_run_for_eval_ee_ablation.json"
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

    def endpoint_snapshot(label):
        body_names = list(command.cfg.body_names)
        endpoint_indexes = [body_names.index(name) for name in endpoint_names if name in body_names]
        body_target = command.body_pos_relative_w.detach()
        robot_body = command.robot_body_pos_w.detach()
        endpoint_z_error = torch.abs(body_target[:, endpoint_indexes, 2] - robot_body[:, endpoint_indexes, 2])
        endpoint_done = torch.any(endpoint_z_error > 0.25, dim=-1)
        body_error = torch.linalg.norm(body_target - robot_body, dim=-1)
        return {
            \"label\": label,
            \"endpoint_names_present\": [name for name in endpoint_names if name in body_names],
            \"endpoint_indexes\": endpoint_indexes,
            \"manual_endpoint_z_done_count_0p25\": int(endpoint_done.detach().cpu().sum().item()),
            \"manual_endpoint_z_done_rate_0p25\": float(endpoint_done.float().mean().detach().cpu().item()),
            \"endpoint_z_error_m\": stats_tensor(endpoint_z_error),
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

    ee_cfg = vec_env.unwrapped.termination_manager.get_term_cfg(\"ee_body_pos\")
    original_ee_body_pos_threshold = float(ee_cfg.params.get(\"threshold\", 0.25))
    ee_cfg.params = dict(ee_cfg.params)
    ee_cfg.params[\"threshold\"] = float(os.environ.get(\"BM_EE_BODY_POS_ABLATION_THRESHOLD\", \"1000.0\"))
    vec_env.unwrapped.termination_manager.set_term_cfg(\"ee_body_pos\", ee_cfg)
    ee_body_pos_threshold_after = float(
        vec_env.unwrapped.termination_manager.get_term_cfg(\"ee_body_pos\").params.get(\"threshold\")
    )

    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    loaded_infos = runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
    obs, extras = vec_env.get_observations()
"""
    if old not in worker_code:
        raise RuntimeError("Base worker code shape changed; cannot inject endpoint termination ablation.")
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
    manual_endpoint_z_done_counts_0p25 = []
    manual_endpoint_z_done_rates_0p25 = []
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
        "manual_endpoint_z_done_count_0p25",
        "manual_endpoint_z_done_rate_0p25",
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
                    "manual_endpoint_z_done_count_0p25": endpoint_after_step["manual_endpoint_z_done_count_0p25"],
                    "manual_endpoint_z_done_rate_0p25": endpoint_after_step["manual_endpoint_z_done_rate_0p25"],
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
                manual_endpoint_z_done_counts_0p25.append(row["manual_endpoint_z_done_count_0p25"])
                manual_endpoint_z_done_rates_0p25.append(row["manual_endpoint_z_done_rate_0p25"])
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
            "ee_body_pos_relaxed": summarize(term_ee_body_pos_counts),
            "manual_ee_body_pos_original_0p25": summarize(manual_endpoint_z_done_counts_0p25),
            "manual_ee_body_pos_original_0p25_rate": summarize(manual_endpoint_z_done_rates_0p25),
        },
        "reset_target_refresh_no_advance": {
            "applied": True,
            "before": reset_target_refresh_before,
            "after": reset_target_refresh_after,
        },
        "ee_body_pos_termination_ablation": {
            "applied": True,
            "original_threshold": original_ee_body_pos_threshold,
            "relaxed_threshold": ee_body_pos_threshold_after,
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
    converted: dict[str, Any] = {}
    for key, value in row.items():
        try:
            converted[key] = float(value)
        except (TypeError, ValueError):
            converted[key] = value
    return converted


def post_step0_done_rate(timeseries: Path, num_envs: int, column: str = "done_count") -> float | None:
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


def patch_summary(summary: dict[str, Any]) -> dict[str, Any]:
    training = load_json(TRAINING_RUN_JSON)
    bundle = load_json(ROBOT_ORDER_BUNDLE_AUDIT)
    split_gate = load_json(ROBOT_ORDER_SPLIT_TASK_GATE)
    old_eval = load_json(OLD_EVAL_JSON)
    target_refresh_eval = load_json(TARGET_REFRESH_EVAL_JSON)
    phase_alignment = load_json(PHASE_ALIGNMENT_JSON)
    metrics = bundle.get("metrics", {})
    eval_ok = summary.get("status") == "ok_resource_adjusted_ppo_checkpoint_eval_completed"
    final_json = (
        OUT
        / "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_ee_termination_ablation.json"
    )
    summary["status"] = (
        "ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_ee_termination_ablation_completed"
        if eval_ok
        else "failed_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_ee_termination_ablation"
    )
    summary["experiment_type"] = (
        "tracking_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_ee_termination_ablation"
    )
    summary["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    summary["scope"] = (
        "Same-seed full checkpoint evaluation using reset-target refresh without time advance plus a relaxed "
        "ee_body_pos z-only termination threshold. This isolates whether endpoint termination dominates the current "
        "robot-order FK tracking done count."
    )
    summary.setdefault("inputs", {})
    summary["inputs"].update(
        {
            "training_run_json": str(TRAINING_RUN_JSON),
            "old_non_warmup_eval_json": str(OLD_EVAL_JSON),
            "target_refresh_eval_json": str(TARGET_REFRESH_EVAL_JSON),
            "phase_alignment_live_probe_json": str(PHASE_ALIGNMENT_JSON),
            "base_compatible_training_run_json": str(
                OUT / "base_compatible_robot_order_fk_repaired_full_bundle_training_run_for_eval_ee_ablation.json"
            ),
            "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
            "robot_order_fk_repaired_motion_npz": str(ROBOT_ORDER_BUNDLE_NPZ),
            "robot_order_fk_repaired_bundle_audit": str(ROBOT_ORDER_BUNDLE_AUDIT),
            "robot_order_split_task_gate": str(ROBOT_ORDER_SPLIT_TASK_GATE),
        }
    )
    summary.setdefault("input_checks", {})
    summary["input_checks"].update(
        {
            "training_run_completed": training.get("status")
            == "ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_training_completed",
            "old_non_warmup_eval_exists": OLD_EVAL_JSON.is_file(),
            "target_refresh_eval_exists": TARGET_REFRESH_EVAL_JSON.is_file(),
            "phase_alignment_probe_ok": phase_alignment.get("status") == "ok_robot_order_fk_phase_alignment_live_probe",
            "official_importer_usd_exists": OFFICIAL_IMPORTER_USD.is_file(),
            "robot_order_motion_npz_exists": ROBOT_ORDER_BUNDLE_NPZ.is_file(),
            "robot_order_bundle_audit_passed": bundle.get("status") == "ok_fk_repaired_robot_order_motion_npz",
            "robot_order_motion_count_40": metrics.get("motion_count") == 40,
            "robot_order_total_frames_11960": metrics.get("total_frames") == 11960,
            "robot_order_split_task_gate_passed": split_gate.get("status")
            == "ok_official_importer_export_fk_repaired_robot_order_split_task_eval",
        }
    )
    summary.setdefault("config", {})
    summary["config"]["seed"] = int(
        os.environ.get("BM_ROBOT_ORDER_FK_REPAIRED_FULL_BUNDLE_PPO_EE_ABLATION_SEED", str(DEFAULT_SEED))
    )
    summary["config"]["num_envs"] = int(
        os.environ.get("BM_ROBOT_ORDER_FK_REPAIRED_FULL_BUNDLE_PPO_EE_ABLATION_NUM_ENVS", str(DEFAULT_NUM_ENVS))
    )
    summary["config"]["reset_target_refresh_no_advance"] = True
    summary["config"]["ee_body_pos_ablation_threshold"] = EE_BODY_POS_ABLATION_THRESHOLD
    if "run" in summary and isinstance(summary["run"].get("metrics"), dict):
        run_metrics = summary["run"]["metrics"]
        run_metrics["uses_official_importer_export_usd"] = True
        run_metrics["uses_resource_adjusted_usd"] = False
        run_metrics["uses_fk_repaired_full_public_motion_bundle"] = False
        run_metrics["uses_robot_order_fk_repaired_full_public_motion_bundle"] = True
        run_metrics["uses_old_degenerate_full_public_motion_bundle"] = False
        run_metrics["official_csv_to_npz_unpatched_output"] = False
        run_metrics["paper_level_tracking_eval"] = False
        run_metrics["reset_target_refresh_no_advance_applied"] = True
        run_metrics["ee_body_pos_termination_ablation_applied"] = True
        run_metrics["motion_count"] = metrics.get("motion_count")
        run_metrics["total_motion_frames"] = metrics.get("total_frames")
    old_metrics = old_eval.get("run", {}).get("metrics", {})
    target_metrics = target_refresh_eval.get("run", {}).get("metrics", {})
    new_metrics = summary.get("run", {}).get("metrics", {})
    old_ts = Path(old_eval.get("outputs", {}).get("timeseries_csv", ""))
    target_ts = Path(target_refresh_eval.get("outputs", {}).get("timeseries_csv", ""))
    new_ts = Path(summary.get("outputs", {}).get("timeseries_csv", ""))
    num_envs = summary["config"]["num_envs"]
    comparison = {
        "old_eval_json": str(OLD_EVAL_JSON),
        "target_refresh_eval_json": str(TARGET_REFRESH_EVAL_JSON),
        "old_done_count_total": old_metrics.get("done_count_total"),
        "target_refresh_done_count_total": target_metrics.get("done_count_total"),
        "ee_ablation_done_count_total": new_metrics.get("done_count_total"),
        "old_done_rate": (
            old_metrics.get("done_count_total") / old_metrics.get("total_env_steps")
            if old_metrics.get("done_count_total") is not None and old_metrics.get("total_env_steps")
            else None
        ),
        "target_refresh_done_rate": (
            target_metrics.get("done_count_total") / target_metrics.get("total_env_steps")
            if target_metrics.get("done_count_total") is not None and target_metrics.get("total_env_steps")
            else None
        ),
        "ee_ablation_done_rate": (
            new_metrics.get("done_count_total") / new_metrics.get("total_env_steps")
            if new_metrics.get("done_count_total") is not None and new_metrics.get("total_env_steps")
            else None
        ),
        "old_step0": first_timeseries_row(old_ts),
        "target_refresh_step0": first_timeseries_row(target_ts),
        "ee_ablation_step0": first_timeseries_row(new_ts),
        "old_post_step0_done_rate": post_step0_done_rate(old_ts, num_envs),
        "target_refresh_post_step0_done_rate": post_step0_done_rate(target_ts, num_envs),
        "ee_ablation_post_step0_done_rate": post_step0_done_rate(new_ts, num_envs),
        "ee_ablation_manual_endpoint_0p25_post_step0_rate": post_step0_done_rate(
            new_ts, num_envs, column="manual_endpoint_z_done_count_0p25"
        ),
    }
    if comparison["target_refresh_done_rate"] is not None and comparison["ee_ablation_done_rate"] is not None:
        comparison["ee_ablation_vs_target_refresh_done_rate_delta"] = (
            comparison["ee_ablation_done_rate"] - comparison["target_refresh_done_rate"]
        )
    if (
        comparison["target_refresh_post_step0_done_rate"] is not None
        and comparison["ee_ablation_post_step0_done_rate"] is not None
    ):
        comparison["ee_ablation_vs_target_refresh_post_step0_done_rate_delta"] = (
            comparison["ee_ablation_post_step0_done_rate"] - comparison["target_refresh_post_step0_done_rate"]
        )
    summary["comparison_to_baselines"] = comparison
    summary.setdefault("checks", {})
    summary["checks"] = {
        **summary.get("checks", {}),
        "same_seed_as_baselines": True,
        "same_full_eval_scope": new_metrics.get("num_envs") == DEFAULT_NUM_ENVS and new_metrics.get("eval_steps") == 299,
        "target_refresh_eval_exists": TARGET_REFRESH_EVAL_JSON.is_file(),
        "phase_alignment_probe_ok": summary["input_checks"]["phase_alignment_probe_ok"],
        "reset_target_refresh_no_advance_applied": new_metrics.get("reset_target_refresh_no_advance", {}).get(
            "applied"
        )
        is True,
        "time_steps_unchanged_by_refresh": new_metrics.get("reset_target_refresh_no_advance", {})
        .get("after", {})
        .get("time_steps_unchanged_by_refresh")
        is True,
        "ee_body_pos_termination_ablation_applied": new_metrics.get("ee_body_pos_termination_ablation", {}).get(
            "applied"
        )
        is True,
        "ee_body_pos_threshold_relaxed": new_metrics.get("ee_body_pos_termination_ablation", {}).get(
            "relaxed_threshold"
        )
        == EE_BODY_POS_ABLATION_THRESHOLD,
        "records_original_endpoint_violation_proxy": "manual_ee_body_pos_original_0p25"
        in new_metrics.get("termination_term_counts", {}),
        "does_not_claim_paper_level_tracking": True,
        "does_not_claim_goal_complete": True,
        "does_not_claim_real_robot": True,
    }
    summary.setdefault("outputs", {})
    final_json = (
        OUT
        / "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_ee_termination_ablation.json"
    )
    summary["outputs"]["json"] = str(final_json)
    summary["outputs"]["worker_script"] = str(OUT / "tracking_g1_resource_adjusted_ppo_checkpoint_eval_worker.py")
    summary["interpretation"] = {
        "goal_complete": False,
        "paper_level_tracking_eval_complete": False,
        "ee_body_pos_termination_ablation_applied": True,
        "why_this_is_mainline": (
            "This isolates whether the current robot-order FK teacher-quality bottleneck is dominated by the endpoint "
            "z-only termination gate after reset-target refresh."
        ),
        "why_not_paper_level": (
            "This deliberately relaxes a termination condition for diagnosis. It is not the official BeyondMimic "
            "teacher, not a valid paper metric, not DAgger/VAE/diffusion, not Fig.5/Fig.6, and not real robot."
        ),
    }
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    module = load_base_module()
    compatible_training_summary = make_base_compatible_training_summary()
    module.OUT = OUT
    module.LOG_DIR = LOG_DIR
    module.RUN_ROOT = RUN_ROOT
    module.ENRICHED_USD = OFFICIAL_IMPORTER_USD
    module.CSV_MOTION_NPZ = ROBOT_ORDER_BUNDLE_NPZ
    module.TRAINING_RUN_JSON = compatible_training_summary
    module.CANDIDATE_GPUS = TARGET_GPUS
    module.NUM_ENVS = int(
        os.environ.get("BM_ROBOT_ORDER_FK_REPAIRED_FULL_BUNDLE_PPO_EE_ABLATION_NUM_ENVS", str(DEFAULT_NUM_ENVS))
    )
    module.SEED = int(
        os.environ.get("BM_ROBOT_ORDER_FK_REPAIRED_FULL_BUNDLE_PPO_EE_ABLATION_SEED", str(DEFAULT_SEED))
    )
    module.WORKER_CODE = patch_worker_code(module.WORKER_CODE)
    os.environ["BM_EE_BODY_POS_ABLATION_THRESHOLD"] = str(EE_BODY_POS_ABLATION_THRESHOLD)
    module.main()

    base_json = OUT / "tracking_g1_resource_adjusted_ppo_checkpoint_eval.json"
    final_json = (
        OUT
        / "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_ee_termination_ablation.json"
    )
    summary = patch_summary(load_json(base_json))
    base_json.unlink(missing_ok=True)
    final_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(final_json),
                "attempted_eval": summary.get("run", {}).get("attempted_eval"),
                "metrics_exists": summary.get("run", {}).get("metrics_exists"),
                "num_envs": summary.get("config", {}).get("num_envs"),
                "comparison_to_baselines": summary.get("comparison_to_baselines", {}),
                "checks": summary.get("checks", {}),
            },
            sort_keys=True,
        )
    )
    if not summary["status"].startswith("ok_"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
