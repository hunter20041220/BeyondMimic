#!/usr/bin/env python3
"""Audit the robot-order FK reset/termination alignment bottleneck.

This script does not launch IsaacLab, train PPO, or mutate raw data. It joins
the current motion-bundle, split-task, PPO-eval, and official-source evidence
to decide whether the next mainline tracking step should be another PPO run or
a reset/command/termination live probe.
"""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/robot_order_fk_reset_termination_alignment_audit"

MOTION_BUNDLE = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/"
    "tracking_g1_official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz.json"
)
SPLIT_EVAL = (
    ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_split_task_eval/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_split_task_eval.json"
)
QUALITY = (
    ROOT
    / "res/tracking/robot_order_fk_ppo_tracking_quality_diagnostic/"
    "robot_order_fk_ppo_tracking_quality_diagnostic.json"
)
SINGLE_EVAL = (
    ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval.json"
)
MULTISEED_EVAL = (
    ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_multiseed_eval/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_multiseed_eval.json"
)
COMMANDS_SRC = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/commands.py"
)
TERMINATIONS_SRC = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/terminations.py"
)
TRACKING_ENV_CFG_SRC = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/tracking_env_cfg.py"
)
RL_ENV_SRC = (
    ROOT
    / "reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab/isaaclab/envs/"
    "manager_based_rl_env.py"
)
BASE_ENV_SRC = (
    ROOT
    / "reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab/isaaclab/envs/"
    "manager_based_env.py"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(value: Any) -> float:
    try:
        if value in {"", None}:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def summarize(values: list[float]) -> dict[str, Any]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = sum(finite) / len(finite)
    var = sum((value - mean) ** 2 for value in finite) / len(finite)
    return {
        "count": len(finite),
        "mean": mean,
        "std": math.sqrt(var),
        "min": min(finite),
        "max": max(finite),
    }


def first_metric(rows: list[dict[str, str]], column: str) -> float | None:
    if not rows:
        return None
    value = fnum(rows[0].get(column))
    return value if math.isfinite(value) else None


def metric_at_step(rows: list[dict[str, str]], column: str, step: int) -> float | None:
    for row in rows:
        try:
            row_step = int(float(row.get("step", "")))
        except ValueError:
            continue
        if row_step == step:
            value = fnum(row.get(column))
            return value if math.isfinite(value) else None
    return None


def mean_metric_from_step(rows: list[dict[str, str]], column: str, start_step: int) -> float | None:
    values: list[float] = []
    for row in rows:
        try:
            row_step = int(float(row.get("step", "")))
        except ValueError:
            continue
        if row_step >= start_step:
            value = fnum(row.get(column))
            if math.isfinite(value):
                values.append(value)
    return sum(values) / len(values) if values else None


def sum_metric_from_step(rows: list[dict[str, str]], column: str, start_step: int) -> float:
    total = 0.0
    for row in rows:
        try:
            row_step = int(float(row.get("step", "")))
        except ValueError:
            continue
        if row_step >= start_step:
            value = fnum(row.get(column))
            if math.isfinite(value):
                total += value
    return total


def line_no(path: Path, pattern: str) -> int | None:
    if not path.is_file():
        return None
    regex = re.compile(pattern)
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if regex.search(line):
            return idx
    return None


def contains_in_order(path: Path, patterns: list[str]) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    pos = -1
    for pattern in patterns:
        match = re.search(pattern, text[pos + 1 :], flags=re.DOTALL)
        if not match:
            return False
        pos += match.end()
    return True


def write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    tmp.replace(path)


def build_source_contract() -> dict[str, Any]:
    tracking_cfg = TRACKING_ENV_CFG_SRC.read_text(encoding="utf-8") if TRACKING_ENV_CFG_SRC.is_file() else ""
    return {
        "commands_py": str(COMMANDS_SRC),
        "terminations_py": str(TERMINATIONS_SRC),
        "tracking_env_cfg_py": str(TRACKING_ENV_CFG_SRC),
        "manager_based_rl_env_py": str(RL_ENV_SRC),
        "manager_based_env_py": str(BASE_ENV_SRC),
        "line_numbers": {
            "motion_loader_body_pos_w_indexes_runtime_body_indexes": line_no(COMMANDS_SRC, r"return self\._body_pos_w\[:, self\._body_indexes\]"),
            "body_pos_relative_w_zero_init": line_no(COMMANDS_SRC, r"self\.body_pos_relative_w = torch\.zeros"),
            "motion_command_update_metrics": line_no(COMMANDS_SRC, r"def _update_metrics"),
            "motion_command_update_command": line_no(COMMANDS_SRC, r"def _update_command"),
            "motion_command_time_steps_increment": line_no(COMMANDS_SRC, r"self\.time_steps \+= 1"),
            "ee_body_pos_z_only_termination_function": line_no(TERMINATIONS_SRC, r"def bad_motion_body_pos_z_only"),
            "ee_body_pos_z_error_expr": line_no(TERMINATIONS_SRC, r"body_pos_relative_w.*robot_body_pos_w"),
            "rl_step_termination_compute": line_no(RL_ENV_SRC, r"self\.reset_buf = self\.termination_manager\.compute"),
            "rl_step_command_compute": line_no(RL_ENV_SRC, r"self\.command_manager\.compute"),
            "base_reset_observation_compute": line_no(BASE_ENV_SRC, r"self\.obs_buf = self\.observation_manager\.compute"),
        },
        "facts": {
            "motion_loader_applies_runtime_body_indexes": "return self._body_pos_w[:, self._body_indexes]" in COMMANDS_SRC.read_text(encoding="utf-8"),
            "body_pos_relative_w_zero_initialized": "self.body_pos_relative_w = torch.zeros" in COMMANDS_SRC.read_text(encoding="utf-8"),
            "update_command_populates_body_pos_relative_w": "self.body_pos_relative_w = delta_pos_w + quat_apply" in COMMANDS_SRC.read_text(encoding="utf-8"),
            "update_command_increments_time_steps_before_targets": "self.time_steps += 1" in COMMANDS_SRC.read_text(encoding="utf-8"),
            "termination_uses_body_pos_relative_w_z_only": "body_pos_relative_w[:, body_indexes, -1]" in TERMINATIONS_SRC.read_text(encoding="utf-8"),
            "ee_body_pos_threshold_0_25m": '"threshold": 0.25' in tracking_cfg,
            "ee_body_pos_ankles_and_wrists": all(
                name in tracking_cfg
                for name in [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ]
            ),
            "rl_step_terminates_before_command_compute": contains_in_order(
                RL_ENV_SRC,
                [
                    r"self\.reset_buf = self\.termination_manager\.compute",
                    r"self\.reward_buf = self\.reward_manager\.compute",
                    r"self\.command_manager\.compute",
                ],
            ),
            "reset_computes_observations_without_command_compute": contains_in_order(
                BASE_ENV_SRC,
                [
                    r"self\._reset_idx\(env_ids\)",
                    r"self\.scene\.write_data_to_sim\(\)",
                    r"self\.sim\.forward\(\)",
                    r"self\.obs_buf = self\.observation_manager\.compute",
                ],
            ),
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = load_json(MOTION_BUNDLE)
    split_eval = load_json(SPLIT_EVAL)
    quality = load_json(QUALITY)
    single_eval = load_json(SINGLE_EVAL)
    multiseed_eval = load_json(MULTISEED_EVAL)
    source_contract = build_source_contract()

    split_rows_path = Path(split_eval.get("outputs", {}).get("rows_csv", ""))
    split_rows = read_csv(split_rows_path)
    top_done_motions = sorted(
        split_rows,
        key=lambda row: fnum(row.get("done_total")),
        reverse=True,
    )[:10]
    top_body_error_motions = sorted(
        split_rows,
        key=lambda row: fnum(row.get("error_body_pos")),
        reverse=True,
    )[:10]

    split_total_steps = split_eval.get("aggregate", {}).get("total_steps") or 0
    split_done_total = split_eval.get("aggregate", {}).get("total_done_count") or 0
    split_done_rate = split_done_total / split_total_steps if split_total_steps else None

    eval_rows: list[dict[str, Any]] = []
    single_outputs = single_eval.get("outputs", {})
    if single_outputs.get("timeseries_csv"):
        rows = read_csv(Path(single_outputs["timeseries_csv"]))
        num_envs = int(single_eval.get("config", {}).get("num_envs") or 0)
        total_env_steps = int(single_eval.get("config", {}).get("total_env_steps") or 0)
        eval_rows.append(
            {
                "label": "single_seed_reference",
                "seed": single_eval.get("config", {}).get("seed"),
                "timeseries_csv": single_outputs["timeseries_csv"],
                "num_envs": num_envs,
                "total_env_steps": total_env_steps,
                "step0_done_count": first_metric(rows, "done_count"),
                "step0_done_rate": first_metric(rows, "done_count") / num_envs if num_envs else None,
                "step0_error_body_pos": first_metric(rows, "error_body_pos"),
                "step0_error_anchor_pos": first_metric(rows, "error_anchor_pos"),
                "step0_error_joint_pos": first_metric(rows, "error_joint_pos"),
                "step1_error_body_pos": metric_at_step(rows, "error_body_pos", 1),
                "post_step0_error_body_pos_mean": mean_metric_from_step(rows, "error_body_pos", 1),
                "post_step0_done_rate": (
                    sum_metric_from_step(rows, "done_count", 1) / max(total_env_steps - num_envs, 1)
                    if total_env_steps and num_envs
                    else None
                ),
            }
        )

    for row in multiseed_eval.get("rows", []):
        path = row.get("timeseries_csv")
        if not path:
            continue
        rows = read_csv(Path(path))
        num_envs = int(row.get("num_envs") or multiseed_eval.get("config", {}).get("num_envs") or 0)
        total_env_steps = int(row.get("total_env_steps") or (num_envs * multiseed_eval.get("config", {}).get("eval_steps", 0)))
        eval_rows.append(
            {
                "label": "multiseed_eval",
                "seed": row.get("seed"),
                "timeseries_csv": path,
                "num_envs": num_envs,
                "total_env_steps": total_env_steps,
                "step0_done_count": first_metric(rows, "done_count"),
                "step0_done_rate": first_metric(rows, "done_count") / num_envs if num_envs else None,
                "step0_error_body_pos": first_metric(rows, "error_body_pos"),
                "step0_error_anchor_pos": first_metric(rows, "error_anchor_pos"),
                "step0_error_joint_pos": first_metric(rows, "error_joint_pos"),
                "step1_error_body_pos": metric_at_step(rows, "error_body_pos", 1),
                "post_step0_error_body_pos_mean": mean_metric_from_step(rows, "error_body_pos", 1),
                "post_step0_done_rate": (
                    sum_metric_from_step(rows, "done_count", 1) / max(total_env_steps - num_envs, 1)
                    if total_env_steps and num_envs
                    else None
                ),
            }
        )

    multiseed_rows = [row for row in eval_rows if row["label"] == "multiseed_eval"]
    multiseed_step0_done_rates = [fnum(row["step0_done_rate"]) for row in multiseed_rows]
    multiseed_step0_body_errors = [fnum(row["step0_error_body_pos"]) for row in multiseed_rows]
    multiseed_step0_anchor_errors = [fnum(row["step0_error_anchor_pos"]) for row in multiseed_rows]
    multiseed_step1_body_errors = [fnum(row["step1_error_body_pos"]) for row in multiseed_rows]
    multiseed_post_step0_body_errors = [fnum(row["post_step0_error_body_pos_mean"]) for row in multiseed_rows]
    multiseed_post_step0_done_rates = [fnum(row["post_step0_done_rate"]) for row in multiseed_rows]

    source_facts = source_contract["facts"]
    checks = {
        "motion_bundle_status_ok": bundle.get("status") == "ok_fk_repaired_robot_order_motion_npz",
        "motion_bundle_robot_order_probe_ok": bool(bundle.get("checks", {}).get("body_order_probe_ok")),
        "motion_bundle_named_target_z_preserved": bool(bundle.get("checks", {}).get("named_target_z_preserved_after_reorder")),
        "motion_bundle_ankle_z_plausible": bool(bundle.get("checks", {}).get("left_right_ankle_mean_z_below_0_25m")),
        "split_eval_40_motions_ok": split_eval.get("aggregate", {}).get("ok_count") == 40,
        "split_eval_not_ready_done_rate_still_high": split_done_rate is not None and split_done_rate > 0.1,
        "quality_diag_step0_all_done": quality.get("checks", {}).get("all_multiseed_step0_done_rate_one") is True,
        "quality_diag_step0_body_spike_gt_40m": quality.get("checks", {}).get("step0_body_spike_gt_40m") is True,
        "quality_diag_post_step0_body_error_below_0_25m": (
            (quality.get("aggregate", {}).get("mean_error_body_pos_post_step0", {}).get("mean") or 999.0) < 0.25
        ),
        "eval_step0_anchor_error_not_spiking": summarize(multiseed_step0_anchor_errors)["mean"] is not None
        and summarize(multiseed_step0_anchor_errors)["mean"] < 0.1,
        "eval_step1_body_error_drops_below_0_3m": summarize(multiseed_step1_body_errors)["mean"] is not None
        and summarize(multiseed_step1_body_errors)["mean"] < 0.3,
        "source_motion_loader_runtime_body_indexes_recorded": bool(source_facts["motion_loader_applies_runtime_body_indexes"]),
        "source_body_pos_relative_w_zero_init_recorded": bool(source_facts["body_pos_relative_w_zero_initialized"]),
        "source_update_command_populates_body_targets_recorded": bool(source_facts["update_command_populates_body_pos_relative_w"]),
        "source_ee_body_pos_z_only_threshold_recorded": bool(
            source_facts["termination_uses_body_pos_relative_w_z_only"]
            and source_facts["ee_body_pos_threshold_0_25m"]
            and source_facts["ee_body_pos_ankles_and_wrists"]
        ),
        "source_rl_step_termination_before_command_compute_recorded": bool(
            source_facts["rl_step_terminates_before_command_compute"]
        ),
        "source_reset_observation_without_command_compute_recorded": bool(
            source_facts["reset_computes_observations_without_command_compute"]
        ),
        "recommends_live_probe_before_more_training": True,
        "does_not_claim_paper_level_tracking": True,
        "does_not_claim_real_robot": True,
        "does_not_claim_goal_complete": True,
    }

    evidence_rows = [
        {
            "evidence": "motion_bundle",
            "status": bundle.get("status"),
            "key_metric": "left/right ankle mean z; target-z reorder preservation",
            "value": json.dumps(
                {
                    "left_ankle_mean_z_m": bundle.get("metrics", {}).get("left_ankle_mean_z_m"),
                    "right_ankle_mean_z_m": bundle.get("metrics", {}).get("right_ankle_mean_z_m"),
                    "max_named_target_z_delta_after_reorder_m": bundle.get("metrics", {}).get(
                        "max_named_target_z_delta_after_reorder_m"
                    ),
                    "motion_count": bundle.get("metrics", {}).get("motion_count"),
                    "total_frames": bundle.get("metrics", {}).get("total_frames"),
                },
                sort_keys=True,
            ),
            "interpretation": "Robot-order FK bundle is plausible and preserves named target heights.",
        },
        {
            "evidence": "split_task_eval",
            "status": split_eval.get("status"),
            "key_metric": "zero-action full split done rate",
            "value": json.dumps(
                {
                    "done_total": split_done_total,
                    "total_steps": split_total_steps,
                    "done_rate": split_done_rate,
                    "error_body_pos_mean": split_eval.get("aggregate", {}).get("error_body_pos", {}).get("mean"),
                    "error_anchor_pos_mean": split_eval.get("aggregate", {}).get("error_anchor_pos", {}).get("mean"),
                },
                sort_keys=True,
            ),
            "interpretation": "Robot-order data repair helped substantially, but the split task remains termination-heavy.",
        },
        {
            "evidence": "ppo_eval_step0",
            "status": quality.get("status"),
            "key_metric": "step0 all-env done and body error spike",
            "value": json.dumps(
                {
                    "step0_done_rate_mean": summarize(multiseed_step0_done_rates)["mean"],
                    "step0_body_error_mean": summarize(multiseed_step0_body_errors)["mean"],
                    "step0_anchor_error_mean": summarize(multiseed_step0_anchor_errors)["mean"],
                    "step1_body_error_mean": summarize(multiseed_step1_body_errors)["mean"],
                    "post_step0_body_error_mean": summarize(multiseed_post_step0_body_errors)["mean"],
                    "post_step0_done_rate_mean": summarize(multiseed_post_step0_done_rates)["mean"],
                },
                sort_keys=True,
            ),
            "interpretation": (
                "The huge step0 body-target spike with normal anchor error and immediate step1 recovery points to "
                "reset/command/termination ordering, not a wholesale broken motion bundle."
            ),
        },
        {
            "evidence": "official_source_order",
            "status": "source_indexed",
            "key_metric": "termination is computed before command_manager.compute in ManagerBasedRLEnv.step",
            "value": json.dumps(source_contract["line_numbers"], sort_keys=True),
            "interpretation": (
                "After reset, observations can be computed before MotionCommand._update_command populates "
                "body_pos_relative_w; the next live probe should force or inspect command update before first "
                "termination."
            ),
        },
    ]

    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "robot_order_fk_reset_termination_alignment_audit",
        "scope": (
            "Source-linked audit for the current robot-order FK tracking bottleneck: body_pos_w, reset bootstrap, "
            "endpoint-z termination, and done-count quality."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "motion_bundle": str(MOTION_BUNDLE),
            "split_eval": str(SPLIT_EVAL),
            "quality_diagnostic": str(QUALITY),
            "single_eval": str(SINGLE_EVAL),
            "multiseed_eval": str(MULTISEED_EVAL),
            "official_sources": source_contract,
        },
        "motion_bundle": {
            "status": bundle.get("status"),
            "metrics": bundle.get("metrics", {}),
            "checks": bundle.get("checks", {}),
        },
        "split_task_eval": {
            "status": split_eval.get("status"),
            "aggregate": split_eval.get("aggregate", {}),
            "done_rate": split_done_rate,
            "top_done_motions": [
                {
                    "motion": row.get("motion"),
                    "done_total": fnum(row.get("done_total")),
                    "terminated_total": fnum(row.get("terminated_total")),
                    "truncated_total": fnum(row.get("truncated_total")),
                    "error_body_pos": fnum(row.get("error_body_pos")),
                    "error_anchor_pos": fnum(row.get("error_anchor_pos")),
                    "reward_mean": fnum(row.get("reward_mean")),
                }
                for row in top_done_motions
            ],
            "top_body_error_motions": [
                {
                    "motion": row.get("motion"),
                    "done_total": fnum(row.get("done_total")),
                    "error_body_pos": fnum(row.get("error_body_pos")),
                    "error_anchor_pos": fnum(row.get("error_anchor_pos")),
                    "reward_mean": fnum(row.get("reward_mean")),
                }
                for row in top_body_error_motions
            ],
        },
        "ppo_step_alignment": {
            "eval_rows": eval_rows,
            "multiseed_step0_done_rate": summarize(multiseed_step0_done_rates),
            "multiseed_step0_error_body_pos": summarize(multiseed_step0_body_errors),
            "multiseed_step0_error_anchor_pos": summarize(multiseed_step0_anchor_errors),
            "multiseed_step1_error_body_pos": summarize(multiseed_step1_body_errors),
            "multiseed_post_step0_error_body_pos": summarize(multiseed_post_step0_body_errors),
            "multiseed_post_step0_done_rate": summarize(multiseed_post_step0_done_rates),
            "quality_diagnostic_status": quality.get("status"),
            "quality_diagnostic_checks": quality.get("checks", {}),
        },
        "source_contract": source_contract,
        "hypothesis": {
            "primary": (
                "The robot-order FK bundle is plausible, but the current first returned eval step is contaminated by "
                "a reset/bootstrap target-alignment issue: body_pos_relative_w is zero-initialized and official "
                "ManagerBasedRLEnv computes termination before command_manager.compute updates the command target."
            ),
            "why_not_just_train_more": (
                "The same checkpoint has a reproducible step0 2048/2048 done spike and the split task still has "
                "2166/11960 zero-action done signals. More PPO before a reset/termination live probe risks training "
                "against a corrupted first-step termination signal and then propagating a weak teacher into VAE and "
                "diffusion."
            ),
        },
        "recommended_next_live_probe": {
            "goal": "Prove whether first-step ee_body_pos termination disappears when MotionCommand targets are populated before termination.",
            "actions": [
                "Create a small live IsaacLab probe on GPU 4 with the robot-order FK bundle and fixed motion IDs.",
                "After env.reset(), record time_steps, body_pos_relative_w endpoint z, robot endpoint z, and termination terms before any policy action.",
                "Call command_manager.compute(dt=env.step_dt) or MotionCommand._update_command() once, then record the same endpoint-z and termination terms.",
                "Run one zero-action step with and without the explicit pre-step command update and compare step0 done_count, ee_body_pos, anchor_pos, and body_pos error.",
                "If the explicit pre-step command update clears step0, patch the local eval/train wrapper to warm up commands after reset without changing official source semantics.",
            ],
            "success_criteria_for_next_fix": {
                "step0_done_rate_after_command_warmup_max": 0.05,
                "step0_error_body_pos_after_command_warmup_max_m": 0.3,
                "no_named_endpoint_z_error_above_threshold_after_command_warmup": True,
                "no_claim_paper_level_until_multiseed_tracking_eval_improves": True,
            },
        },
        "checks": checks,
        "interpretation": {
            "claim_level": "source_linked_tracking_reset_termination_diagnostic",
            "mainline_effect": (
                "This narrows the current tracking bottleneck to a testable reset/command/termination ordering fix. "
                "It should be resolved before rerunning full PPO or rebuilding teacher rollout, VAE, state-latent, "
                "denoiser, and guidance artifacts."
            ),
            "goal_complete": False,
            "paper_level_tracking_reproduced": False,
            "real_robot_evidence": False,
        },
        "evidence_rows": evidence_rows,
        "outputs": {
            "json": str(OUT / "robot_order_fk_reset_termination_alignment_audit.json"),
            "evidence_csv": str(OUT / "robot_order_fk_reset_termination_alignment_evidence.csv"),
            "markdown": str(OUT / "robot_order_fk_reset_termination_alignment_audit.md"),
        },
    }

    write_json(OUT / "robot_order_fk_reset_termination_alignment_audit.json", summary)
    write_csv(
        OUT / "robot_order_fk_reset_termination_alignment_evidence.csv",
        evidence_rows,
        ["evidence", "status", "key_metric", "value", "interpretation"],
    )
    md = [
        "# Robot-Order FK Reset / Termination Alignment Audit",
        "",
        "This audit does not launch simulation or claim paper-level tracking. It joins current robot-order FK data,",
        "existing eval traces, and official source order to define the next live tracking fix.",
        "",
        "## Findings",
        "",
        f"- Motion bundle status: `{bundle.get('status')}`; motions `{bundle.get('metrics', {}).get('motion_count')}`, frames `{bundle.get('metrics', {}).get('total_frames')}`.",
        f"- Split zero-action done rate: `{split_done_rate}` from `{split_done_total}/{split_total_steps}` done signals.",
        f"- Multi-seed step0 done-rate mean: `{summary['ppo_step_alignment']['multiseed_step0_done_rate']['mean']}`.",
        f"- Multi-seed step0 body-error mean: `{summary['ppo_step_alignment']['multiseed_step0_error_body_pos']['mean']}`.",
        f"- Multi-seed step0 anchor-error mean: `{summary['ppo_step_alignment']['multiseed_step0_error_anchor_pos']['mean']}`.",
        f"- Multi-seed step1 body-error mean: `{summary['ppo_step_alignment']['multiseed_step1_error_body_pos']['mean']}`.",
        "",
        "## Source-Linked Interpretation",
        "",
        "- `MotionCommand` zero-initializes `body_pos_relative_w` and later populates it in `_update_command()`.",
        "- `ee_body_pos` uses z-only ankle/wrist body-position termination with a 0.25 m threshold.",
        "- `ManagerBasedRLEnv.step()` computes termination before `command_manager.compute()`.",
        "- Therefore the next live probe should test command warmup immediately after reset before doing more PPO.",
        "",
        "## Next Mainline Probe",
        "",
        "Run a small GPU-4 IsaacLab probe that records endpoint z-errors before and after an explicit command update",
        "after reset. If the first-step all-done spike disappears, patch the local train/eval wrappers to warm up",
        "commands after reset, then rerun full tracking eval before rebuilding downstream VAE/diffusion artifacts.",
        "",
        "## Claim Boundary",
        "",
        "This is tracking-quality diagnosis only. It is not paper-level tracking, not DAgger, not Fig. 5/Fig. 6,",
        "not TensorRT deployment, and not real-robot evidence.",
        "",
    ]
    (OUT / "robot_order_fk_reset_termination_alignment_audit.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"]}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
