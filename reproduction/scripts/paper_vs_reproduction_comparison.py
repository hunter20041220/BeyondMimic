#!/usr/bin/env python3
"""Build the paper-vs-reproduction comparison table requested by goal.md."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/comparison"

FIELDS = [
    "experiment",
    "paper_value",
    "reproduction_value",
    "absolute_difference",
    "relative_difference",
    "paper_figure_or_table",
    "paper_source",
    "run_id",
    "reproduction_level",
    "comparison_type",
    "difference_explanation",
]

ALLOWED_COMPARISON_TYPES = {
    "exactly_comparable",
    "approximately_comparable",
    "qualitative_only",
    "not_publicly_reproducible",
    "requires_real_robot",
}


def load_json(rel_path: str) -> dict[str, Any]:
    return json.loads((ROOT / rel_path).read_text(encoding="utf-8"))


def stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def numeric_difference(expected: Any, observed: Any) -> tuple[str, str]:
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        abs_diff = abs(float(expected) - float(observed))
        if float(expected) == 0.0:
            rel_diff = 0.0 if abs_diff == 0.0 else math.inf
        else:
            rel_diff = abs_diff / abs(float(expected))
        return f"{abs_diff:.12g}", f"{rel_diff:.12g}"
    if (
        isinstance(expected, list)
        and isinstance(observed, list)
        and len(expected) == len(observed)
        and all(isinstance(x, (int, float)) for x in expected)
        and all(isinstance(x, (int, float)) for x in observed)
    ):
        diffs = [abs(float(a) - float(b)) for a, b in zip(expected, observed)]
        max_abs = max(diffs, default=0.0)
        denom = max((abs(float(a)) for a in expected), default=0.0)
        max_rel = 0.0 if denom == 0.0 and max_abs == 0.0 else (math.inf if denom == 0.0 else max_abs / denom)
        return f"{max_abs:.12g}", f"{max_rel:.12g}"
    return "", ""


def comparison_type_for_panel(status: str) -> str:
    if status in {"released-data reproduced", "released-data processed"}:
        return "approximately_comparable"
    if status == "official-code audited":
        return "qualitative_only"
    if "blocked" in status:
        return "not_publicly_reproducible"
    if status == "paper-only evidence":
        return "not_publicly_reproducible"
    return "qualitative_only"


def comparison_type_for_coverage(row: dict[str, Any]) -> str:
    notes = f"{row.get('notes', '')} {row.get('paper_item', '')} {row.get('label', '')}".lower()
    bucket = row.get("status_bucket", "")
    if bucket == "blocked_or_unreproduced":
        if "real" in notes or "hardware" in notes or "robot" in notes or "fig:more_motion" in notes:
            return "requires_real_robot"
        return "not_publicly_reproducible"
    if bucket == "partial":
        return "approximately_comparable"
    return "qualitative_only"


def reproduction_level_for_table(row: dict[str, Any]) -> str:
    status = row.get("status")
    table = row.get("table")
    if status == "match" and table == "tab:ppo_hyperparameters":
        return "official-code reproduction"
    if status == "source_value_present":
        return "official-code audit"
    if status == "debug_match":
        return "paper-faithful reimplementation debug-only"
    return "value audit"


def add_table_value_rows(rows: list[dict[str, str]]) -> None:
    data = load_json("res/paper_table_values/paper_table_value_audit.json")
    for item in data["rows"]:
        abs_diff, rel_diff = numeric_difference(item.get("paper_expected"), item.get("observed"))
        status = item.get("status", "")
        if status == "debug_match":
            explanation = (
                "Paper table value matches the current debug-only Level C probe/config evidence. "
                "This is value-level comparability only; no trained VAE/diffusion checkpoint or paper metric is claimed."
            )
        elif status == "source_value_present":
            explanation = (
                "Paper value is present in the audited official/source configuration. "
                "This is a source-level comparison, not an executed training result."
            )
        else:
            explanation = "Paper table value matches the audited reproduction evidence."
        rows.append(
            {
                "experiment": f"table:{item.get('table')}:{item.get('parameter')}",
                "paper_value": stringify(item.get("paper_expected")),
                "reproduction_value": stringify(item.get("observed")),
                "absolute_difference": abs_diff,
                "relative_difference": rel_diff,
                "paper_figure_or_table": item.get("table", ""),
                "paper_source": item.get("source", ""),
                "run_id": item.get("evidence", ""),
                "reproduction_level": reproduction_level_for_table(item),
                "comparison_type": "exactly_comparable",
                "difference_explanation": explanation,
            }
        )


def add_panel_map_rows(rows: list[dict[str, str]]) -> None:
    path = ROOT / "reproduction/docs/paper_panel_map.tsv"
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for item in reader:
            status = item["status"]
            ctype = comparison_type_for_panel(status)
            if ctype == "approximately_comparable":
                level = "released-data reproduction"
            elif ctype == "qualitative_only":
                level = "official-code/source audit"
            else:
                level = "not reproducible from current public/local artifacts"
            rows.append(
                {
                    "experiment": f"panel:{item['paper_item']}:{item['label_or_panel']}",
                    "paper_value": "paper panel/claim",
                    "reproduction_value": item["local_reproduction_artifact"],
                    "absolute_difference": "",
                    "relative_difference": "",
                    "paper_figure_or_table": item["paper_item"],
                    "paper_source": item["source_evidence"],
                    "run_id": item["local_reproduction_artifact"],
                    "reproduction_level": level,
                    "comparison_type": ctype,
                    "difference_explanation": item["notes"],
                }
            )


def add_source_coverage_rows(rows: list[dict[str, str]]) -> None:
    data = load_json("res/paper_source_coverage/paper_source_coverage_audit.json")
    for item in data["rows"]:
        ctype = comparison_type_for_coverage(item)
        bucket = item.get("status_bucket", "")
        if ctype == "requires_real_robot":
            level = "requires real Unitree G1/hardware evidence"
        elif ctype == "not_publicly_reproducible":
            level = "not reproducible from current public/local artifacts"
        elif ctype == "approximately_comparable":
            level = "partial released-data or local reproduction"
        elif bucket == "debug_only":
            level = "debug-only mechanism evidence"
        else:
            level = "paper/source indexed"
        rows.append(
            {
                "experiment": f"coverage:{item.get('kind')}:{item.get('paper_item')}:{item.get('label')}",
                "paper_value": item.get("caption_excerpt", "") or item.get("coverage_status", ""),
                "reproduction_value": item.get("coverage_status", ""),
                "absolute_difference": "",
                "relative_difference": "",
                "paper_figure_or_table": item.get("paper_item", ""),
                "paper_source": item.get("source", ""),
                "run_id": item.get("evidence", ""),
                "reproduction_level": level,
                "comparison_type": ctype,
                "difference_explanation": item.get("notes", ""),
            }
        )


def add_goal_checkpoint_rows(rows: list[dict[str, str]]) -> None:
    checkpoints = [
        {
            "experiment": "goal_checkpoint:walking_velocity_tracking_error",
            "paper_value": "12.14%",
            "paper_figure_or_table": "goal.md comparison checkpoint",
            "paper_source": "goal.md:1635-1701",
            "comparison_type": "not_publicly_reproducible",
            "explanation": (
                "No paper-level trained motion-tracking evaluation run is available in the current evidence set. "
                "The velocity-tracking metric formula/API is covered by local tests, but live IsaacLab/Kit evaluation "
                "remains blocked by the inotify gate."
            ),
            "formula_evidence": "res/evaluation_metrics_coverage/evaluation_metrics_coverage_audit.json;res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json;res/tests/core_math_unit_tests/core_math_unit_tests.json",
        },
        {
            "experiment": "goal_checkpoint:running_velocity_tracking_error",
            "paper_value": "13.65%",
            "paper_figure_or_table": "goal.md comparison checkpoint",
            "paper_source": "goal.md:1635-1701",
            "comparison_type": "not_publicly_reproducible",
            "explanation": (
                "No paper-level trained motion-tracking evaluation run is available in the current evidence set. "
                "The velocity-tracking metric formula/API is covered by local tests, but live IsaacLab/Kit evaluation "
                "remains blocked by the inotify gate."
            ),
            "formula_evidence": "res/evaluation_metrics_coverage/evaluation_metrics_coverage_audit.json;res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json;res/tests/core_math_unit_tests/core_math_unit_tests.json",
        },
        {
            "experiment": "goal_checkpoint:direct_diffusion_cartwheel_success",
            "paper_value": "5%",
            "paper_figure_or_table": "Figure 6 / goal.md comparison checkpoint",
            "paper_source": "goal.md:1635-1701",
            "comparison_type": "not_publicly_reproducible",
            "explanation": (
                "Official BeyondMimic VAE/diffusion code, trained checkpoints, and Fig.6 rollout logs are absent; "
                "current Level C evidence is debug-only. Success/fall-rate metric formulas are covered by local tests, "
                "but the paper cartwheel rollout result is not reproduced."
            ),
            "formula_evidence": "res/evaluation_metrics_coverage/evaluation_metrics_coverage_audit.json;res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json;res/tests/core_math_unit_tests/core_math_unit_tests.json;res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json",
        },
        {
            "experiment": "goal_checkpoint:latent_diffusion_cartwheel_success",
            "paper_value": "95%",
            "paper_figure_or_table": "Figure 6 / goal.md comparison checkpoint",
            "paper_source": "goal.md:1635-1701",
            "comparison_type": "not_publicly_reproducible",
            "explanation": (
                "Official BeyondMimic VAE/diffusion code, trained checkpoints, and Fig.6 rollout logs are absent; "
                "current Level C evidence is debug-only. Success/fall-rate metric formulas are covered by local tests, "
                "but the paper cartwheel rollout result is not reproduced."
            ),
            "formula_evidence": "res/evaluation_metrics_coverage/evaluation_metrics_coverage_audit.json;res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json;res/tests/core_math_unit_tests/core_math_unit_tests.json;res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json",
        },
    ]
    for item in checkpoints:
        rows.append(
            {
                "experiment": item["experiment"],
                "paper_value": item["paper_value"],
                "reproduction_value": "not available in current reproduction evidence",
                "absolute_difference": "",
                "relative_difference": "",
                "paper_figure_or_table": item["paper_figure_or_table"],
                "paper_source": item["paper_source"],
                "run_id": item["formula_evidence"],
                "reproduction_level": "not reproduced",
                "comparison_type": item["comparison_type"],
                "difference_explanation": item["explanation"],
            }
        )


def add_guidance_full_split_rows(rows: list[dict[str, str]]) -> None:
    result_table = load_json(
        "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.json"
    )
    visual = load_json("res/level_c/guidance_checkpoint_visualization/level_c_guidance_checkpoint_visualization.json")
    offline_json = (
        "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/"
        "level_c_lafan1_paper_arch_guidance_eval.json"
    )
    reverse_json = (
        "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/"
        "level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.json"
    )
    table_json = "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.json"
    visual_json = "res/level_c/guidance_checkpoint_visualization/level_c_guidance_checkpoint_visualization.json"
    source_by_task = {
        "joystick": "reproduction/paper/source/root.tex:549-555",
        "waypoint": "reproduction/paper/source/root.tex:556-563",
        "obstacle_avoidance": "reproduction/paper/source/root.tex:564-586",
        "inpainting": "reproduction/paper/source/root.tex:241-242",
        "composed_objectives": "reproduction/paper/source/root.tex:241-242;reproduction/paper/source/root.tex:593-594",
    }
    paper_item_by_task = {
        "joystick": "Figure 5 / joystick guidance cost",
        "waypoint": "Figure 6 / waypoint guidance cost",
        "obstacle_avoidance": "Figure 6 / obstacle-avoidance guidance cost",
        "inpainting": "Figure 6 / motion inpainting",
        "composed_objectives": "Figure 6 / composed waypoint and obstacle guidance",
    }
    label_by_task = {
        "joystick": "joystick",
        "waypoint": "waypoint",
        "obstacle_avoidance": "obstacle avoidance",
        "inpainting": "motion inpainting",
        "composed_objectives": "composed objectives",
    }

    for item in result_table["rows"]:
        mode = item["mode"]
        task = item["task"]
        source_json = offline_json if mode == "offline" else reverse_json
        if mode == "offline":
            mode_phrase = "single-step offline classifier-gradient guidance"
        else:
            mode_phrase = "batched reverse-process classifier guidance"
        reproduction_value = {
            "mode": mode,
            "mean_best_cost_delta": item["mean_best_cost_delta"],
            "positive_best_cost_delta_fraction": item["positive_best_cost_delta_fraction"],
            "primary_improved_count": item["primary_improved_count"],
            "window_count": item["window_count"],
            "scale_count": item["scale_count"],
            "source_rows": item["row_count"],
        }
        rows.append(
            {
                "experiment": f"guidance_full_split:{mode}:{task}",
                "paper_value": (
                    f"{paper_item_by_task[task]} qualitative task behavior; no public paper numeric "
                    "cost-delta table."
                ),
                "reproduction_value": stringify(reproduction_value),
                "absolute_difference": "",
                "relative_difference": "",
                "paper_figure_or_table": paper_item_by_task[task],
                "paper_source": source_by_task[task],
                "run_id": f"{source_json};{table_json}",
                "reproduction_level": "public LAFAN1 paper-architecture guidance surrogate",
                "comparison_type": "qualitative_only",
                "difference_explanation": (
                    f"Local full-split {mode_phrase} implements the paper guidance objective family for "
                    f"{label_by_task[task]} on public LAFAN1 windows and records task cost changes. "
                    "This is not a paper Fig.5/Fig.6 closed-loop robot rollout and is not compared against "
                    "a published paper numeric value."
                ),
            }
        )

    rows.append(
        {
            "experiment": "guidance_full_split:summary_table_and_plot",
            "paper_value": "Figure 5/Figure 6 qualitative downstream guidance demonstrations",
            "reproduction_value": (
                f"offline_rows={result_table['metrics']['offline_source_rows']}; "
                f"reverse_rows={result_table['metrics']['reverse_source_rows']}; "
                f"table_rows={result_table['metrics']['row_count']}; "
                f"figure_count={len(result_table['outputs']['figures'])}"
            ),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Figure 5 / Figure 6",
            "paper_source": "reproduction/paper/source/root.tex:223-243;reproduction/paper/source/root.tex:549-586",
            "run_id": table_json,
            "reproduction_level": "public LAFAN1 paper-architecture guidance surrogate",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The local result table consolidates full-split offline and reverse guidance metrics and "
                "plots cost-delta summaries, but it deliberately remains a public-data surrogate rather than "
                "a reproduction of the paper's closed-loop videos or robot success rates."
            ),
        }
    )
    rows.append(
        {
            "experiment": "guidance_full_split:checkpoint_visualization",
            "paper_value": "Figure 5/Figure 6 qualitative denoising and downstream-task visualization",
            "reproduction_value": (
                f"visual_file_count={visual['metrics']['visual_file_count']}; "
                f"representative_window={visual['metrics']['representative_window_index']}; "
                f"modes={visual['metrics']['mode_count']}; tasks={visual['metrics']['task_count']}"
            ),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Figure 5 / Figure 6",
            "paper_source": "reproduction/paper/source/root.tex:223-243",
            "run_id": visual_json,
            "reproduction_level": "debug checkpoint guidance visualization",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The checkpoint visualization shows clean, unguided, offline-guided, and reverse-guided "
                "public-data trajectories for one representative window. It is classified as debug visual "
                "evidence, not as a paper video or closed-loop evaluation."
            ),
        }
    )


def add_tracking_train_entry_diagnostic_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/tracking/g1_resource_adjusted_train_entry_diagnostic/"
        "tracking_g1_resource_adjusted_train_entry_diagnostic_audit.json"
    )
    metrics = audit["metrics"]
    markers = audit.get("run", {}).get("markers", {})
    runtime_warning = audit.get("interpretation", {}).get("runtime_warning")
    if runtime_warning is None:
        if markers.get("physx_gpu_kernel_error"):
            runtime_warning = "The probe log contains PhysX GPU kernel warnings before the success sentinel."
        else:
            runtime_warning = "Runtime warning field is absent in this audit JSON; see the raw probe log if needed."
    reproduction_value = {
        "status": audit["status"],
        "runner_class": metrics["runner_class"],
        "runner_training_type": metrics["runner_training_type"],
        "requested_learning_iterations": metrics["requested_learning_iterations"],
        "configured_num_steps_per_env": metrics["configured_num_steps_per_env"],
        "num_envs": metrics["num_envs"],
        "num_actions": metrics["num_actions"],
        "num_obs": metrics["num_obs"],
        "num_privileged_obs": metrics["num_privileged_obs"],
        "checkpoint_written": metrics["checkpoint_written"],
        "runtime_warning": runtime_warning,
    }
    rows.append(
        {
            "experiment": "tracking:resource_adjusted_train_entry_diagnostic",
            "paper_value": (
                "BeyondMimic trains a motion-tracking teacher with PPO before DAgger/VAE/diffusion stages; "
                "the paper does not publish a tiny train-entry diagnostic metric."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion tracking teacher / PPO pipeline",
            "paper_source": "reproduction/paper/source/root.tex;official whole_body_tracking source",
            "run_id": (
                "res/tracking/g1_resource_adjusted_train_entry_diagnostic/"
                "tracking_g1_resource_adjusted_train_entry_diagnostic_audit.json"
            ),
            "reproduction_level": "resource-adjusted train-entry diagnostic",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The local probe constructs the official Tracking-Flat-G1-v0 env, RSL-RL wrapper, and custom "
                "MotionOnPolicyRunner, then executes one four-step PPO update without logging a checkpoint. "
                "It verifies wiring after IsaacLab recovery but uses generated resource-adjusted USD and an "
                "official-CSV-derived resource-adjusted motion file; it is not formal PPO training, not a trained "
                "teacher checkpoint, and not a paper-level tracking metric."
            ),
        }
    )


def add_tracking_resource_adjusted_ppo_training_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/tracking/g1_resource_adjusted_ppo_training_run/"
        "tracking_g1_resource_adjusted_ppo_training_run.json"
    )
    reproduction_value = {
        "status": audit["status"],
        "attempted_training": audit["run"]["attempted_training"],
        "reason_not_started": audit["run"].get("reason_not_started", ""),
        "candidate_physical_gpus": audit["config"]["candidate_physical_gpus"],
        "selected_physical_gpus": audit["config"]["selected_physical_gpus"],
        "world_size": audit["config"]["world_size"],
        "total_num_envs": audit["config"]["total_num_envs"],
        "num_steps_per_env": audit["config"]["num_steps_per_env"],
        "max_iterations": audit["config"]["max_iterations"],
        "resource_ready": audit["gpu_preflight"]["resource_ready"],
        "checkpoint_count": audit["run"].get("checkpoint_count", 0),
        "duration_seconds": audit["run"].get("duration_seconds"),
        "rank_metric_count": len(audit["run"].get("rank_metrics", [])),
    }
    rows.append(
        {
            "experiment": "tracking:resource_adjusted_ppo_training_run",
            "paper_value": (
                "BeyondMimic trains the motion-tracking teacher with PPO at large IsaacLab scale; the paper does not "
                "publish a directly comparable resource-adjusted training-run metric."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion tracking teacher / PPO pipeline",
            "paper_source": "reproduction/paper/source/root.tex;official whole_body_tracking source",
            "run_id": (
                "res/tracking/g1_resource_adjusted_ppo_training_run/"
                "tracking_g1_resource_adjusted_ppo_training_run.json"
            ),
            "reproduction_level": "resource-adjusted PPO training run",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The run launches the official Tracking-Flat-G1-v0 manager stack and RSL-RL PPO through "
                "torch.distributed on available GPUs selected from physical GPUs 4-7, using the official PPO rollout "
                "length, checkpoint writing, and GPU telemetry. The current artifact completed 100 resource-adjusted "
                "iterations on the generated G1 USD and official-CSV-derived motion, producing checkpoints and rank "
                "metrics. Because it does not use the official converted replay asset/motion pipeline and is far "
                "below paper training scale, it is not an official paper-level tracking teacher or paper metric."
            ),
        }
    )


def add_tracking_resource_adjusted_ppo_checkpoint_eval_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/tracking/g1_resource_adjusted_ppo_checkpoint_eval/"
        "tracking_g1_resource_adjusted_ppo_checkpoint_eval.json"
    )
    metrics = audit["run"].get("metrics", {})
    motion_metrics = metrics.get("motion_metrics", {})
    reproduction_value = {
        "status": audit["status"],
        "checkpoint": audit["inputs"]["checkpoint"],
        "selected_physical_gpus": audit["config"]["selected_physical_gpus"],
        "cuda_visible_devices": audit["config"]["cuda_visible_devices"],
        "num_envs": audit["config"]["num_envs"],
        "eval_steps": audit["config"]["eval_steps"],
        "total_env_steps": audit["config"]["total_env_steps"],
        "duration_seconds": audit["run"].get("duration_seconds"),
        "loaded_iteration": metrics.get("loaded_iteration"),
        "reward_mean": metrics.get("reward", {}).get("mean_over_steps", {}).get("mean"),
        "done_count_total": metrics.get("done_count_total"),
        "error_anchor_pos_mean": motion_metrics.get("error_anchor_pos", {}).get("mean"),
        "error_body_pos_mean": motion_metrics.get("error_body_pos", {}).get("mean"),
        "error_joint_pos_mean": motion_metrics.get("error_joint_pos", {}).get("mean"),
    }
    rows.append(
        {
            "experiment": "tracking:resource_adjusted_ppo_checkpoint_eval",
            "paper_value": (
                "BeyondMimic uses a trained motion-tracking teacher in the official PPO pipeline; the paper does not "
                "publish a directly comparable local resource-adjusted checkpoint-evaluation metric."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion tracking teacher / PPO pipeline",
            "paper_source": "reproduction/paper/source/root.tex;official whole_body_tracking play.py source",
            "run_id": (
                "res/tracking/g1_resource_adjusted_ppo_checkpoint_eval/"
                "tracking_g1_resource_adjusted_ppo_checkpoint_eval.json"
            ),
            "reproduction_level": "resource-adjusted PPO checkpoint evaluation",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The evaluator loads the locally trained resource-adjusted `model_99.pt` checkpoint with the official "
                "RSL-RL `OnPolicyRunner` inference API and runs `Tracking-Flat-G1-v0` for 512 environments x 299 "
                "steps. It records reward, termination, action, GPU, and motion-command tracking metrics. Because "
                "the checkpoint and motion come from the generated resource-adjusted USD/official-CSV-derived path, "
                "this is virtual checkpoint-evaluation evidence, not official paper-level tracking evaluation."
            ),
        }
    )


def add_tracking_resource_adjusted_teacher_rollout_dataset_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/tracking/g1_resource_adjusted_teacher_rollout_dataset/"
        "tracking_g1_resource_adjusted_teacher_rollout_dataset.json"
    )
    aggregate = audit["aggregate_metrics"]
    gpu_summary = audit["run"].get("gpu_metrics_summary", {})
    reproduction_value = {
        "status": audit["status"],
        "checkpoint": audit["inputs"]["checkpoint"],
        "selected_physical_gpus": audit["config"]["selected_physical_gpus"],
        "cuda_visible_devices": audit["config"]["cuda_visible_devices"],
        "world_size": audit["config"]["world_size"],
        "num_envs_per_rank": audit["config"]["num_envs_per_rank"],
        "rollout_steps": audit["config"]["rollout_steps"],
        "total_env_steps": aggregate["total_env_steps"],
        "shard_count": aggregate["shard_count"],
        "dataset_npz_total_size_bytes": aggregate["dataset_npz_total_size_bytes"],
        "reward_mean_by_rank": aggregate["reward_mean_by_rank"],
        "done_count_total": aggregate["done_count_total"],
        "duration_seconds": audit["run"].get("duration_seconds"),
        "gpu_metrics_summary": gpu_summary,
    }
    rows.append(
        {
            "experiment": "tracking:resource_adjusted_teacher_rollout_dataset",
            "paper_value": (
                "BeyondMimic trains VAE/diffusion on teacher/Dagger-style state-latent trajectories, but the paper "
                "does not release the official teacher checkpoint or rollout logs for a direct public comparison."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Teacher rollout / DAgger trajectory data prerequisite",
            "paper_source": "reproduction/paper/source/root.tex;official whole_body_tracking task sources",
            "run_id": (
                "res/tracking/g1_resource_adjusted_teacher_rollout_dataset/"
                "tracking_g1_resource_adjusted_teacher_rollout_dataset.json"
            ),
            "reproduction_level": "resource-adjusted teacher rollout dataset gate",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The run collected two GPU shards of policy observations, critic observations, actions, rewards, "
                "done flags, timeouts, and motion timesteps from the local resource-adjusted `model_99.pt` teacher "
                "inside `Tracking-Flat-G1-v0`. It is useful downstream data for local VAE/state-latent experiments, "
                "but it is not the paper's official DAgger rollout dataset and cannot validate paper-level Fig. 5/"
                "Fig. 6 closed-loop diffusion results."
            ),
        }
    )


def add_tracking_urdf_source_equivalence_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json("res/tracking/g1_urdf_source_equivalence_audit/tracking_g1_urdf_source_equivalence_audit.json")
    comparison = audit["comparisons"]["download_vs_whole_body_tracking"]
    reproduction_value = {
        "status": audit["status"],
        "download_vs_reproduction_data_identical": audit["checks"][
            "download_and_reproduction_data_structurally_identical"
        ],
        "same_29_nonfixed_action_joints": audit["checks"]["whole_body_tracking_has_same_29_nonfixed_action_joints"],
        "download_vs_wbt_link_diff": comparison["link_set_diff"],
        "download_vs_wbt_joint_diff": comparison["joint_set_diff"],
    }
    rows.append(
        {
            "experiment": "tracking:g1_urdf_source_equivalence",
            "paper_value": (
                "BeyondMimic uses a Unitree G1 tracking setup; the paper does not publish a numeric URDF "
                "source-equivalence metric."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion tracking G1 asset/source setup",
            "paper_source": "official downloaded G1 LAFAN robot_description; official whole_body_tracking source",
            "run_id": "res/tracking/g1_urdf_source_equivalence_audit/tracking_g1_urdf_source_equivalence_audit.json",
            "reproduction_level": "official-source asset audit",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The audit shows that the downloaded official G1 URDF and reproduction-data copy are byte-identical, "
                "while the official whole_body_tracking URDF preserves the same 29 non-fixed/action joints but differs "
                "in support links/joints and physical bookkeeping. This narrows the official replay blocker to the "
                "conversion/scaffold path; it is not official converter output, motion.npz, replay, PPO, or a paper "
                "tracking metric."
            ),
        }
    )


def add_tracking_official_replay_entry_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/tracking/official_replay_npz_entry_diagnostic/"
        "tracking_official_replay_npz_entry_diagnostic_audit.json"
    )
    reproduction_value = {
        "status": audit["status"],
        "latest_blocker": audit["latest_blocker"],
        "app_launcher_constructed": audit["checks"]["app_launcher_constructed"],
        "blocked_before_artifact_download": audit["checks"]["fake_wandb_download_seen"] is False,
        "layer_save_blocker": audit["run"]["markers"]["failed_to_save_layer"],
        "empty_robot_after_converter": audit["run"]["markers"]["empty_robot_after_converter"],
    }
    rows.append(
        {
            "experiment": "tracking:official_replay_npz_entry_diagnostic",
            "paper_value": (
                "BeyondMimic uses official IsaacLab replay/evaluation for motion tracking; the paper does not "
                "publish a numeric replay-entry diagnostic metric."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion tracking replay / evaluation pipeline",
            "paper_source": "official whole_body_tracking scripts/replay_npz.py",
            "run_id": (
                "res/tracking/official_replay_npz_entry_diagnostic/"
                "tracking_official_replay_npz_entry_diagnostic_audit.json"
            ),
            "reproduction_level": "official-entry blocked diagnostic",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The diagnostic runs the official replay_npz.py entrypoint with a local fake-WandB artifact and a "
                "bounded AppLauncher wrapper. The entry reaches AppLauncher but blocks in the official URDF converter "
                "layer-save path before artifact download or replay-loop execution, leaving an empty robot prim. This "
                "is useful failure evidence, not official replay success, not csv_to_npz output, and not a paper-level "
                "tracking metric."
            ),
        }
    )


def add_tracking_g1_import_config_variant_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/tracking/g1_urdf_import_config_variant_probe/"
        "tracking_g1_urdf_import_config_variant_probe.json"
    )
    method_payload = audit["method_probe"]["payload"]
    baseline_usd = audit["variant_summary"]["variant_baseline_make_instanceable_false"]["usd"]
    reproduction_value = {
        "status": audit["status"],
        "current_blocker": audit["current_blocker"],
        "has_set_make_instanceable": method_payload["has_set_make_instanceable"],
        "has_set_instanceable_usd_path": method_payload["has_set_instanceable_usd_path"],
        "baseline_stage_open_ok": baseline_usd["stage_open_ok"],
        "baseline_prim_count": baseline_usd["prim_count"],
        "baseline_joint_count": baseline_usd["joint_count"],
        "baseline_rigid_body_like_count": baseline_usd["rigid_body_like_count"],
    }
    rows.append(
        {
            "experiment": "tracking:g1_urdf_import_config_variant_probe",
            "paper_value": (
                "BeyondMimic uses a valid IsaacLab G1 asset for motion tracking/replay; the paper does not publish "
                "a numeric URDF ImportConfig diagnostic."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion tracking replay / G1 asset conversion",
            "paper_source": "official whole_body_tracking assets and IsaacLab URDF importer",
            "run_id": (
                "res/tracking/g1_urdf_import_config_variant_probe/"
                "tracking_g1_urdf_import_config_variant_probe.json"
            ),
            "reproduction_level": "official-asset converter diagnostic",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The probe records the Isaac Sim 4.5 URDF ImportConfig surface and a baseline official G1 URDF "
                "conversion attempt. The config exposes no instanceable-related setters, and the baseline conversion "
                "produces an openable but empty USD with zero prims, joints, or rigid bodies. This narrows the replay "
                "blocker but is not official replay, not motion.npz generation, not PPO, and not a paper-level metric."
            ),
        }
    )


def add_tracking_g1_in_memory_gpu4_probe_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/tracking/g1_urdf_in_memory_gpu4_probe/"
        "tracking_g1_urdf_in_memory_gpu4_probe.json"
    )
    checks = audit["checks"]
    reproduction_value = {
        "status": audit["status"],
        "returncode": audit["returncode"],
        "duration_seconds": audit["duration_seconds"],
        "app_launcher_reached": checks["app_launcher_reached"],
        "in_memory_import_returned": checks["in_memory_import_returned"],
        "export_exists": checks["export_exists"],
        "export_has_joints": checks["export_has_joints"],
        "export_has_rigid_bodies": checks["export_has_rigid_bodies"],
    }
    rows.append(
        {
            "experiment": "tracking:g1_urdf_in_memory_gpu4_probe",
            "paper_value": (
                "BeyondMimic relies on a valid IsaacLab G1 robot stage for official replay/tracking; the paper does "
                "not publish a numeric in-memory URDF-import diagnostic."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion tracking replay / G1 asset conversion",
            "paper_source": "official whole_body_tracking assets and Isaac Sim URDF importer",
            "run_id": (
                "res/tracking/g1_urdf_in_memory_gpu4_probe/"
                "tracking_g1_urdf_in_memory_gpu4_probe.json"
            ),
            "reproduction_level": "official-asset converter diagnostic",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The probe runs the official Isaac Sim URDF importer on current GPU4 with `dest_path=\"\"` so the "
                "importer avoids the file-layer save branch. It reaches AppLauncher and begins parsing the official "
                "G1 URDF, but Vulkan `ERROR_DEVICE_LOST` kills the process before import return or stage export. This "
                "records a narrower official replay blocker; it is not official motion.npz, replay, PPO, or a "
                "paper-level tracking result."
            ),
        }
    )


def add_resource_adjusted_teacher_rollout_vae_training_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/level_c/resource_adjusted_teacher_rollout_vae_training/"
        "level_c_resource_adjusted_teacher_rollout_vae_training.json"
    )
    worker = audit["worker_summary"]
    dataset = worker["dataset"]
    training = worker["training"]
    evaluation = worker["evaluation"]
    gpu_summary = audit.get("gpu_metrics_summary", {})
    reproduction_value = {
        "status": audit["status"],
        "source_teacher_status": worker["source_teacher_rollout"]["status"],
        "sample_count": dataset["sample_count"],
        "obs_dim": dataset["obs_dim"],
        "action_dim": dataset["action_dim"],
        "splits": worker["splits"],
        "latent_dim": training["latent_dim"],
        "hidden_dim": training["hidden_dim"],
        "epochs": training["epochs"],
        "cuda_visible_devices": worker["cuda_visible_devices"],
        "torch_cuda_device_count": worker["torch_cuda_device_count"],
        "data_parallel_used": worker["data_parallel_used"],
        "validation_action_mse": evaluation["validation"]["action_mse"],
        "test_action_mse": evaluation["test"]["action_mse"],
        "test_action_abs_error_mean": evaluation["test"]["action_abs_error_mean"],
        "gpu_metrics_summary": gpu_summary,
    }
    rows.append(
        {
            "experiment": "level_c:resource_adjusted_teacher_rollout_vae_training",
            "paper_value": (
                "BeyondMimic trains a conditional VAE on teacher/Dagger-style trajectory data before state-latent "
                "diffusion; the paper does not release the official teacher rollout dataset or VAE checkpoint."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Conditional VAE / DAgger trajectory prerequisite",
            "paper_source": "BeyondMimic method sections and local resource-adjusted teacher-rollout evidence",
            "run_id": (
                "res/level_c/resource_adjusted_teacher_rollout_vae_training/"
                "level_c_resource_adjusted_teacher_rollout_vae_training.json"
            ),
            "reproduction_level": "resource-adjusted full teacher-rollout VAE training",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This is a full local training run over all currently collected resource-adjusted teacher-rollout "
                "shards, using two visible GPUs and a conditional action VAE. It is stronger than a smoke test and "
                "useful for the reading-report reproduction narrative, but it is not the official BeyondMimic DAgger "
                "dataset, not an official VAE checkpoint, and not a closed-loop Fig. 5/Fig. 6 paper result."
            ),
        }
    )


def add_resource_adjusted_state_latent_dataset_and_diffusion_rows(rows: list[dict[str, str]]) -> None:
    dataset_audit = load_json(
        "res/level_c/resource_adjusted_teacher_rollout_state_latent_dataset/"
        "level_c_resource_adjusted_teacher_rollout_state_latent_dataset.json"
    )
    diffusion_audit = load_json(
        "res/level_c/resource_adjusted_state_latent_diffusion_training/"
        "level_c_resource_adjusted_state_latent_diffusion_training.json"
    )
    dataset_worker = dataset_audit["worker_summary"]
    diffusion_worker = diffusion_audit["worker_summary"]
    dataset_value = {
        "status": dataset_audit["status"],
        "sample_count": dataset_worker["dataset"]["sample_count"],
        "window_count": dataset_worker["dataset"]["window_count"],
        "split_counts": dataset_worker["dataset"]["split_counts"],
        "sequence_length": dataset_worker["dataset"]["sequence_length"],
        "obs_dim": dataset_worker["dataset"]["obs_dim"],
        "latent_dim": dataset_worker["dataset"]["latent_dim"],
        "token_dim": dataset_worker["dataset"]["token_dim"],
        "weighted_posterior_reconstruction_mse": dataset_worker["dataset"][
            "weighted_posterior_reconstruction_mse"
        ],
        "latent_shard_count": len(dataset_worker["outputs"]["latent_shards"]),
    }
    rows.append(
        {
            "experiment": "level_c:resource_adjusted_teacher_rollout_state_latent_dataset",
            "paper_value": (
                "BeyondMimic trains diffusion on state-latent trajectories from teacher/DAgger rollouts and a "
                "trained conditional VAE; the official dataset is not released."
            ),
            "reproduction_value": stringify(dataset_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "State-latent trajectory dataset prerequisite",
            "paper_source": "BeyondMimic method sections and local resource-adjusted rollout/VAE evidence",
            "run_id": (
                "res/level_c/resource_adjusted_teacher_rollout_state_latent_dataset/"
                "level_c_resource_adjusted_teacher_rollout_state_latent_dataset.json"
            ),
            "reproduction_level": "resource-adjusted full state-latent dataset",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This builds full local windows from all currently collected resource-adjusted teacher rollout samples "
                "and the local resource-adjusted action VAE posterior. It is a meaningful downstream dataset gate, but "
                "not the official DAgger/state-latent dataset used for the paper's closed-loop diffusion results."
            ),
        }
    )
    diffusion_value = {
        "status": diffusion_audit["status"],
        "window_count": diffusion_worker["dataset"]["window_count"],
        "split_counts": diffusion_worker["dataset"]["split_counts"],
        "epochs": diffusion_worker["training"]["epochs"],
        "batch_windows": diffusion_worker["training"]["batch_windows"],
        "cuda_visible_devices": diffusion_worker["cuda_visible_devices"],
        "torch_cuda_device_count": diffusion_worker["torch_cuda_device_count"],
        "data_parallel_used": diffusion_worker["data_parallel_used"],
        "validation_pred_token_mse": diffusion_worker["evaluation"]["validation"]["pred_token_mse"],
        "test_pred_token_mse": diffusion_worker["evaluation"]["test"]["pred_token_mse"],
        "test_noisy_token_mse": diffusion_worker["evaluation"]["test"]["noisy_token_mse"],
        "test_denoising_improvement_ratio": diffusion_worker["evaluation"]["test"][
            "denoising_improvement_ratio"
        ],
        "gpu_metrics_summary": diffusion_audit.get("gpu_metrics_summary", {}),
    }
    rows.append(
        {
            "experiment": "level_c:resource_adjusted_state_latent_diffusion_training",
            "paper_value": (
                "BeyondMimic trains a state-latent diffusion model and evaluates it with guided closed-loop humanoid "
                "control; the official training data and checkpoint are not released."
            ),
            "reproduction_value": stringify(diffusion_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "State-latent diffusion training / Fig. 5-6 prerequisite",
            "paper_source": "BeyondMimic method sections and local resource-adjusted state-latent dataset evidence",
            "run_id": (
                "res/level_c/resource_adjusted_state_latent_diffusion_training/"
                "level_c_resource_adjusted_state_latent_diffusion_training.json"
            ),
            "reproduction_level": "resource-adjusted full state-latent denoiser training",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This trains a local denoising model on all generated resource-adjusted state-latent windows and "
                "reports held-out denoising improvement. It is not the official BeyondMimic diffusion checkpoint, "
                "not TensorRT deployment, and not closed-loop Fig. 5/Fig. 6 guidance evaluation."
            ),
        }
    )


def validate_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    missing_required_field_rows: list[dict[str, Any]] = []
    invalid_comparison_type_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        missing = [field for field in FIELDS if field not in row]
        if missing:
            missing_required_field_rows.append({"row_index": idx, "missing": missing, "experiment": row.get("experiment")})
        if row.get("comparison_type") not in ALLOWED_COMPARISON_TYPES:
            invalid_comparison_type_rows.append(
                {"row_index": idx, "comparison_type": row.get("comparison_type"), "experiment": row.get("experiment")}
            )
    required_experiments = {
        "goal_checkpoint:walking_velocity_tracking_error",
        "goal_checkpoint:running_velocity_tracking_error",
        "goal_checkpoint:direct_diffusion_cartwheel_success",
        "goal_checkpoint:latent_diffusion_cartwheel_success",
    }
    required_guidance_experiments = {
        "guidance_full_split:offline:joystick",
        "guidance_full_split:offline:waypoint",
        "guidance_full_split:offline:obstacle_avoidance",
        "guidance_full_split:offline:inpainting",
        "guidance_full_split:offline:composed_objectives",
        "guidance_full_split:reverse:joystick",
        "guidance_full_split:reverse:waypoint",
        "guidance_full_split:reverse:obstacle_avoidance",
        "guidance_full_split:reverse:inpainting",
        "guidance_full_split:reverse:composed_objectives",
        "guidance_full_split:summary_table_and_plot",
        "guidance_full_split:checkpoint_visualization",
    }
    present = {row["experiment"] for row in rows}
    goal_checkpoint_rows = [row for row in rows if row["experiment"] in required_experiments]
    guidance_rows = [row for row in rows if row["experiment"] in required_guidance_experiments]
    return {
        "missing_required_field_rows": missing_required_field_rows,
        "invalid_comparison_type_rows": invalid_comparison_type_rows,
        "required_goal_checkpoint_rows_present": required_experiments <= present,
        "missing_goal_checkpoint_rows": sorted(required_experiments - present),
        "required_guidance_full_split_rows_present": required_guidance_experiments <= present,
        "missing_guidance_full_split_rows": sorted(required_guidance_experiments - present),
        "guidance_full_split_rows_remain_qualitative_only": all(
            row.get("comparison_type") == "qualitative_only" for row in guidance_rows
        ),
        "guidance_full_split_rows_do_not_claim_paper_rollout": all(
            "closed-loop" in row.get("difference_explanation", "")
            or "public-data surrogate" in row.get("difference_explanation", "")
            or "debug visual" in row.get("difference_explanation", "")
            for row in guidance_rows
        ),
        "goal_checkpoint_rows_have_formula_evidence": all(
            "res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json" in row.get("run_id", "")
            and "res/tests/core_math_unit_tests/core_math_unit_tests.json" in row.get("run_id", "")
            for row in goal_checkpoint_rows
        ),
        "goal_checkpoint_rows_remain_not_reproduced": all(
            row.get("reproduction_level") == "not reproduced" for row in goal_checkpoint_rows
        ),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, str]]) -> None:
    lines: list[str] = []
    lines.append("# Paper vs Reproduction Comparison")
    lines.append("")
    lines.append("This table is generated from current local evidence. It separates exact value checks, released-data re-rendering, qualitative source/debug audits, public-artifact blockers, and real-hardware requirements.")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Total rows: `{summary['total_rows']}`.")
    lines.append(f"- Comparison types: `{json.dumps(summary['comparison_type_counts'], sort_keys=True)}`.")
    lines.append(f"- Reproduction levels: `{json.dumps(summary['reproduction_level_counts'], sort_keys=True)}`.")
    lines.append(f"- Required goal checkpoint rows present: `{not summary['missing_goal_checkpoint_rows']}`.")
    lines.append(f"- CSV: `{summary['outputs']['csv']}`.")
    lines.append("")
    lines.append("## Boundary")
    lines.append("Rows marked `not_publicly_reproducible` or `requires_real_robot` are deliberate non-claims. They identify paper results that cannot be reproduced from the current public/local artifact set or this hardware environment.")
    lines.append("")
    lines.append("## Rows")
    lines.append("| experiment | paper_value | reproduction_value | comparison_type | reproduction_level | explanation |")
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        paper_value = row["paper_value"].replace("|", "\\|")[:160]
        reproduction_value = row["reproduction_value"].replace("|", "\\|")[:160]
        explanation = row["difference_explanation"].replace("|", "\\|")[:220]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["experiment"].replace("|", "\\|"),
                    paper_value,
                    reproduction_value,
                    row["comparison_type"],
                    row["reproduction_level"].replace("|", "\\|"),
                    explanation,
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    add_table_value_rows(rows)
    add_panel_map_rows(rows)
    add_source_coverage_rows(rows)
    add_guidance_full_split_rows(rows)
    add_tracking_train_entry_diagnostic_rows(rows)
    add_tracking_resource_adjusted_ppo_training_rows(rows)
    add_tracking_resource_adjusted_ppo_checkpoint_eval_rows(rows)
    add_tracking_resource_adjusted_teacher_rollout_dataset_rows(rows)
    add_tracking_urdf_source_equivalence_rows(rows)
    add_tracking_official_replay_entry_rows(rows)
    add_tracking_g1_import_config_variant_rows(rows)
    add_tracking_g1_in_memory_gpu4_probe_rows(rows)
    add_resource_adjusted_teacher_rollout_vae_training_rows(rows)
    add_resource_adjusted_state_latent_dataset_and_diffusion_rows(rows)
    add_goal_checkpoint_rows(rows)

    validation = validate_rows(rows)
    comparison_type_counts = Counter(row["comparison_type"] for row in rows)
    reproduction_level_counts = Counter(row["reproduction_level"] for row in rows)
    csv_path = OUT / "paper_vs_reproduction.csv"
    md_path = OUT / "paper_vs_reproduction.md"
    json_path = OUT / "paper_vs_reproduction.json"

    summary: dict[str, Any] = {
        "status": "ok"
        if not validation["missing_required_field_rows"] and not validation["invalid_comparison_type_rows"]
        else "failed",
        "experiment_type": "paper_vs_reproduction_comparison",
        "scope": (
            "comparison rows for paper table values, paper panels/source coverage, full-split guidance "
            "surrogate metrics, and goal.md checkpoint metrics"
        ),
        "allowed_comparison_types": sorted(ALLOWED_COMPARISON_TYPES),
        "required_fields": FIELDS,
        "total_rows": len(rows),
        "comparison_type_counts": dict(sorted(comparison_type_counts.items())),
        "reproduction_level_counts": dict(sorted(reproduction_level_counts.items())),
        "missing_required_field_rows": validation["missing_required_field_rows"],
        "invalid_comparison_type_rows": validation["invalid_comparison_type_rows"],
        "missing_goal_checkpoint_rows": validation["missing_goal_checkpoint_rows"],
        "missing_guidance_full_split_rows": validation["missing_guidance_full_split_rows"],
        "checks": {
            "required_goal_checkpoint_rows_present": validation["required_goal_checkpoint_rows_present"],
            "required_guidance_full_split_rows_present": validation["required_guidance_full_split_rows_present"],
            "guidance_full_split_rows_remain_qualitative_only": validation[
                "guidance_full_split_rows_remain_qualitative_only"
            ],
            "guidance_full_split_rows_do_not_claim_paper_rollout": validation[
                "guidance_full_split_rows_do_not_claim_paper_rollout"
            ],
            "goal_checkpoint_rows_have_formula_evidence": validation["goal_checkpoint_rows_have_formula_evidence"],
            "goal_checkpoint_rows_remain_not_reproduced": validation["goal_checkpoint_rows_remain_not_reproduced"],
            "does_not_claim_goal_complete": True,
        },
        "rows": rows,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The comparison table is complete for current evidence, but several paper results remain marked "
                "not_publicly_reproducible or requires_real_robot because trained Level B/C runs, Fig.5/Fig.6 logs, "
                "official Level C artifacts, and real hardware are unavailable."
            ),
        },
        "outputs": {"csv": str(csv_path), "markdown": str(md_path), "json": str(json_path)},
    }

    write_csv(csv_path, rows)
    write_markdown(md_path, summary, rows)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(json_path), "csv": str(csv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
