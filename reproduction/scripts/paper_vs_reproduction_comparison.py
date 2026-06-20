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


def add_tracking_official_csv_loop_ppo_training_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/tracking/g1_official_csv_loop_ppo_training_run/"
        "tracking_g1_official_csv_loop_ppo_training_run.json"
    )
    rank_metrics = audit["run"].get("rank_metrics", [])
    rank0 = next((item for item in rank_metrics if item.get("rank") == 0), {})
    reproduction_value = {
        "status": audit["status"],
        "selected_physical_gpus": audit["config"]["selected_physical_gpus"],
        "cuda_visible_devices": audit["config"]["cuda_visible_devices"],
        "world_size": audit["config"]["world_size"],
        "total_num_envs": audit["config"]["total_num_envs"],
        "num_steps_per_env": audit["config"]["num_steps_per_env"],
        "max_iterations": audit["config"]["max_iterations"],
        "tot_timesteps_rank0": rank0.get("tot_timesteps"),
        "duration_seconds": audit["run"].get("duration_seconds"),
        "checkpoint_count": audit["run"].get("checkpoint_count", 0),
        "latest_learning_iteration_rank0": rank0.get("current_learning_iteration"),
        "official_csv_to_npz_loop_output": rank0.get("official_csv_to_npz_loop_output"),
        "paper_level_training": rank0.get("paper_level_training"),
    }
    rows.append(
        {
            "experiment": "tracking:official_csv_loop_ppo_training_run",
            "paper_value": (
                "BeyondMimic trains the motion-tracking teacher with large-scale PPO before downstream DAgger/VAE/"
                "diffusion stages; the paper does not publish a directly comparable 300-iteration local training "
                "metric."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion tracking teacher / PPO pipeline",
            "paper_source": "reproduction/paper/source/root.tex;official whole_body_tracking source",
            "run_id": (
                "res/tracking/g1_official_csv_loop_ppo_training_run/"
                "tracking_g1_official_csv_loop_ppo_training_run.json"
            ),
            "reproduction_level": "official csv-loop motion PPO training run",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The run launches official `Tracking-Flat-G1-v0` and RSL-RL PPO through `torch.distributed` on "
                "physical GPUs 4 and 7, using the motion artifact generated by the official `csv_to_npz.py` loop "
                "under the enriched-USD runtime patch. It completed 300 iterations and wrote checkpoints/telemetry, "
                "but the robot asset path is still resource-adjusted, the official paper training scale is much "
                "larger, and no paper-level teacher metric is claimed."
            ),
        }
    )


def add_tracking_official_csv_loop_ppo_checkpoint_eval_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/tracking/g1_official_csv_loop_ppo_checkpoint_eval/"
        "tracking_g1_official_csv_loop_ppo_checkpoint_eval.json"
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
        "sampling_top1_prob_mean": motion_metrics.get("sampling_top1_prob", {}).get("mean"),
    }
    rows.append(
        {
            "experiment": "tracking:official_csv_loop_ppo_checkpoint_eval",
            "paper_value": (
                "BeyondMimic evaluates trained motion-tracking teachers before using them for downstream trajectory "
                "data; the paper does not publish a directly comparable local official-loop checkpoint metric."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion tracking teacher / PPO pipeline",
            "paper_source": "reproduction/paper/source/root.tex;official whole_body_tracking play.py source",
            "run_id": (
                "res/tracking/g1_official_csv_loop_ppo_checkpoint_eval/"
                "tracking_g1_official_csv_loop_ppo_checkpoint_eval.json"
            ),
            "reproduction_level": "official csv-loop motion PPO checkpoint evaluation",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The evaluator loads the 300-iteration checkpoint trained on official-loop motion with the official "
                "RSL-RL `OnPolicyRunner` inference API and runs `Tracking-Flat-G1-v0` for 512 environments x 299 "
                "steps. It records rollout metrics, but it is still an enriched-USD, resource-adjusted virtual "
                "evaluation and not a paper-level teacher-policy result."
            ),
        }
    )


def add_tracking_official_csv_loop_ppo_multiseed_eval_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/tracking/g1_official_csv_loop_ppo_checkpoint_multiseed_eval/"
        "tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval.json"
    )
    assets = load_json(
        "res/report_assets/official_csv_loop_ppo_checkpoint_multiseed_eval/"
        "official_csv_loop_ppo_checkpoint_multiseed_eval_assets.json"
    )
    aggregate = audit["aggregate"]
    reproduction_value = {
        "status": audit["status"],
        "seeds": audit["config"]["seeds"],
        "gpu_assignment": audit["config"]["gpu_assignment"],
        "num_envs": audit["config"]["num_envs"],
        "eval_steps": audit["config"]["eval_steps"],
        "total_env_steps": audit["metrics"]["total_env_steps"],
        "reward_mean_mean": aggregate["reward_mean"]["mean"],
        "reward_mean_std": aggregate["reward_mean"]["std"],
        "error_body_pos_mean": aggregate["error_body_pos_mean"]["mean"],
        "error_body_pos_std": aggregate["error_body_pos_mean"]["std"],
        "error_joint_pos_mean": aggregate["error_joint_pos_mean"]["mean"],
        "error_joint_pos_std": aggregate["error_joint_pos_mean"]["std"],
        "report_assets": assets["assets"],
    }
    rows.append(
        {
            "experiment": "tracking:official_csv_loop_ppo_checkpoint_multiseed_eval",
            "paper_value": (
                "BeyondMimic reports paper-level tracking teacher performance before downstream DAgger/VAE/"
                "diffusion stages, but it does not publish a directly comparable 3-seed local checkpoint eval table."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion tracking teacher / PPO pipeline",
            "paper_source": "reproduction/paper/source/root.tex;official whole_body_tracking play.py source",
            "run_id": (
                "res/tracking/g1_official_csv_loop_ppo_checkpoint_multiseed_eval/"
                "tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval.json"
            ),
            "reproduction_level": "official csv-loop motion PPO checkpoint multi-seed local virtual evaluation",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This run repeats the local official-csv-loop checkpoint evaluator for three seeds over 512 "
                "environments x 299 steps, producing mean/std tracking statistics and report assets. It strengthens "
                "stability evidence for the local virtual tracking chain, but it still uses the enriched-USD runtime "
                "patch and a reduced 300-iteration checkpoint, so it is not the unpatched official paper-level "
                "tracking teacher evaluation."
            ),
        }
    )


def add_tracking_official_csv_loop_full_bundle_rows(rows: list[dict[str, str]]) -> None:
    bundle = load_json(
        "res/tracking/official_csv_loop_full_bundle_motion_npz/"
        "tracking_g1_official_csv_loop_full_bundle_motion_npz.json"
    )
    training = load_json(
        "res/tracking/g1_official_csv_loop_full_bundle_ppo_training_run/"
        "tracking_g1_official_csv_loop_full_bundle_ppo_training_run.json"
    )
    eval_audit = load_json(
        "res/tracking/g1_official_csv_loop_full_bundle_ppo_checkpoint_eval/"
        "tracking_g1_official_csv_loop_full_bundle_ppo_checkpoint_eval.json"
    )
    assets = load_json(
        "res/report_assets/official_csv_loop_full_bundle_ppo_checkpoint_eval/"
        "official_csv_loop_full_bundle_ppo_checkpoint_eval_assets.json"
    )
    policy_video = load_json(
        "res/visualization/official_csv_loop_full_bundle_policy_rollout/"
        "official_csv_loop_policy_rollout_video_asset.json"
    )
    bundle_info = bundle["bundle"]
    rank0 = next((item for item in training["run"].get("rank_metrics", []) if item.get("rank") == 0), {})
    metrics = eval_audit["run"].get("metrics", {})
    motion_metrics = metrics.get("motion_metrics", {})
    rows.append(
        {
            "experiment": "tracking:official_csv_loop_full_public_motion_bundle",
            "paper_value": (
                "BeyondMimic trains a tracking teacher over the available motion corpus, but the paper does not "
                "publish a one-file concatenated public-motion MotionLoader artifact."
            ),
            "reproduction_value": stringify(
                {
                    "status": bundle["status"],
                    "motion_count": bundle_info["motion_count"],
                    "total_frames": bundle_info["total_frames"],
                    "fps": bundle_info["fps"],
                    "boundary_count": bundle_info["boundary_count"],
                    "npz_sha256": bundle_info["npz_sha256"],
                    "clip_manifest": bundle["outputs"]["clips_csv"],
                }
            ),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion tracking teacher / public motion input",
            "paper_source": "reproduction/paper/source/root.tex;official whole_body_tracking MotionLoader source",
            "run_id": (
                "res/tracking/official_csv_loop_full_bundle_motion_npz/"
                "tracking_g1_official_csv_loop_full_bundle_motion_npz.json"
            ),
            "reproduction_level": "full-public-motion local bundle for official MotionLoader",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The local official MotionLoader accepts one NPZ path, so all 40 public official-loop motion NPZs "
                "were concatenated into one audited bundle without patching official loader code. This improves "
                "motion coverage for local virtual PPO, but the boundaries are artificial and it is not the paper's "
                "original teacher motion sampler or official DAgger dataset."
            ),
        }
    )
    rows.append(
        {
            "experiment": "tracking:official_csv_loop_full_bundle_policy_rollout_video",
            "paper_value": (
                "BeyondMimic uses a trained motion-tracking teacher and later reports qualitative robot behavior, "
                "but it does not publish a directly comparable local full-public-bundle PPO policy-vs-reference MP4."
            ),
            "reproduction_value": stringify(
                {
                    "status": policy_video["status"],
                    "claim_level": policy_video["claim_level"],
                    "frame_count": policy_video["frame_count"],
                    "target_body_count": policy_video["target_body_count"],
                    "bundle": policy_video.get("bundle", {}),
                    "metrics": policy_video["metrics"],
                    "assets": policy_video["assets"],
                }
            ),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion tracking teacher / local report video",
            "paper_source": "reproduction/paper/source/root.tex;official whole_body_tracking source",
            "run_id": (
                "res/visualization/official_csv_loop_full_bundle_policy_rollout/"
                "official_csv_loop_policy_rollout_video_asset.json"
            ),
            "reproduction_level": "full-public-motion local virtual PPO policy rollout video",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The new video records a 299-frame local policy-vs-reference rollout from the iteration-299 "
                "full-bundle PPO checkpoint trained over the 40-motion public official-csv-loop bundle. It is "
                "useful evidence for the English report/PPT because it shows the robot policy moving in IsaacLab, "
                "but it still uses the enriched-USD scaffold and a local checkpoint, and it is not an official "
                "BeyondMimic checkpoint, Fig. 5/Fig. 6 guided diffusion rollout, TensorRT deployment, or real-robot "
                "result."
            ),
        }
    )
    rows.append(
        {
            "experiment": "tracking:official_csv_loop_full_bundle_ppo_training_run",
            "paper_value": (
                "BeyondMimic trains the motion-tracking teacher at paper scale before downstream DAgger/VAE/"
                "diffusion stages; no directly comparable 300-iteration full-public-bundle metric is published."
            ),
            "reproduction_value": stringify(
                {
                    "status": training["status"],
                    "selected_physical_gpus": training["config"]["selected_physical_gpus"],
                    "world_size": training["config"]["world_size"],
                    "total_num_envs": training["config"]["total_num_envs"],
                    "num_steps_per_env": training["config"]["num_steps_per_env"],
                    "max_iterations": training["config"]["max_iterations"],
                    "duration_seconds": training["run"].get("duration_seconds"),
                    "checkpoint_count": training["run"].get("checkpoint_count"),
                    "rank0_learning_iteration": rank0.get("current_learning_iteration"),
                    "rank0_timesteps": rank0.get("tot_timesteps"),
                    "motion_count": bundle_info["motion_count"],
                    "total_motion_frames": bundle_info["total_frames"],
                }
            ),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion tracking teacher / PPO pipeline",
            "paper_source": "reproduction/paper/source/root.tex;official whole_body_tracking source",
            "run_id": (
                "res/tracking/g1_official_csv_loop_full_bundle_ppo_training_run/"
                "tracking_g1_official_csv_loop_full_bundle_ppo_training_run.json"
            ),
            "reproduction_level": "full-public-motion local virtual PPO training run",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The run launches official `Tracking-Flat-G1-v0` and RSL-RL PPO on GPUs 4 and 7 for 300 iterations "
                "using the concatenated 40-motion public bundle. It is stronger than the earlier single-motion PPO "
                "run, but still uses the enriched-USD runtime patch, artificial clip boundaries, and a reduced "
                "training budget; it is not the paper's official tracking teacher."
            ),
        }
    )
    rows.append(
        {
            "experiment": "tracking:official_csv_loop_full_bundle_ppo_checkpoint_eval",
            "paper_value": (
                "BeyondMimic evaluates the trained tracking teacher before using teacher rollouts downstream, but "
                "the paper does not publish a directly comparable full-public-bundle local checkpoint metric."
            ),
            "reproduction_value": stringify(
                {
                    "status": eval_audit["status"],
                    "checkpoint": eval_audit["inputs"]["checkpoint"],
                    "num_envs": eval_audit["config"]["num_envs"],
                    "eval_steps": eval_audit["config"]["eval_steps"],
                    "total_env_steps": eval_audit["config"]["total_env_steps"],
                    "loaded_iteration": metrics.get("loaded_iteration"),
                    "duration_seconds": eval_audit["run"].get("duration_seconds"),
                    "done_count_total": metrics.get("done_count_total"),
                    "reward_mean": metrics.get("reward", {}).get("mean_over_steps", {}).get("mean"),
                    "error_anchor_pos_mean": motion_metrics.get("error_anchor_pos", {}).get("mean"),
                    "error_body_pos_mean": motion_metrics.get("error_body_pos", {}).get("mean"),
                    "error_joint_pos_mean": motion_metrics.get("error_joint_pos", {}).get("mean"),
                    "motion_count": metrics.get("motion_count"),
                    "total_motion_frames": metrics.get("total_motion_frames"),
                    "report_assets": assets["assets"],
                }
            ),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion tracking teacher / PPO pipeline",
            "paper_source": "reproduction/paper/source/root.tex;official whole_body_tracking play.py source",
            "run_id": (
                "res/tracking/g1_official_csv_loop_full_bundle_ppo_checkpoint_eval/"
                "tracking_g1_official_csv_loop_full_bundle_ppo_checkpoint_eval.json"
            ),
            "reproduction_level": "full-public-motion local virtual PPO checkpoint evaluation",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The evaluator loads the iteration-299 full-bundle PPO checkpoint through the official RSL-RL "
                "`OnPolicyRunner` inference API and runs `Tracking-Flat-G1-v0` for 512 environments x 299 steps. "
                "It records local virtual tracking metrics and report plots, but still depends on the enriched-USD "
                "patch and artificial bundle boundaries, so it is not paper-level teacher evaluation."
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


def add_tracking_official_csv_loop_teacher_rollout_dataset_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/tracking/g1_official_csv_loop_teacher_rollout_dataset/"
        "tracking_g1_official_csv_loop_teacher_rollout_dataset.json"
    )
    aggregate = audit["aggregate_metrics"]
    gpu_summary = audit["run"].get("gpu_metrics_summary", {})
    shard_metrics = audit["run"].get("shard_metrics", [])
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
        "loaded_iteration_by_rank": [row.get("loaded_iteration") for row in shard_metrics],
        "duration_seconds": audit["run"].get("duration_seconds"),
        "gpu_metrics_summary": gpu_summary,
    }
    rows.append(
        {
            "experiment": "tracking:official_csv_loop_teacher_rollout_dataset",
            "paper_value": (
                "BeyondMimic trains downstream VAE/diffusion components on teacher/Dagger-style state-latent "
                "trajectories; the paper does not release the official teacher checkpoint or DAgger rollout logs."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Teacher rollout / DAgger trajectory data prerequisite",
            "paper_source": "reproduction/paper/source/root.tex;official whole_body_tracking task sources",
            "run_id": (
                "res/tracking/g1_official_csv_loop_teacher_rollout_dataset/"
                "tracking_g1_official_csv_loop_teacher_rollout_dataset.json"
            ),
            "reproduction_level": "official csv-loop motion teacher rollout dataset gate",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The run collected two GPU shards from the local iteration-299 checkpoint trained on official-loop "
                "motion, recording policy observations, critic observations, actions, rewards, done flags, timeouts, "
                "and motion timesteps. It is stronger local trajectory evidence for downstream VAE/state-latent work, "
                "but it is still produced under the enriched-USD runtime patch and is not the paper's official DAgger "
                "rollout dataset."
            ),
        }
    )


def add_tracking_official_csv_loop_full_bundle_teacher_rollout_dataset_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/tracking/g1_official_csv_loop_full_bundle_teacher_rollout_dataset/"
        "tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset.json"
    )
    assets = load_json(
        "res/report_assets/official_csv_loop_full_bundle_teacher_rollout_dataset/"
        "official_csv_loop_full_bundle_teacher_rollout_report_assets.json"
    )
    aggregate = audit["aggregate_metrics"]
    gpu_summary = audit["run"].get("gpu_metrics_summary", {})
    shard_metrics = audit["run"].get("shard_metrics", [])
    reproduction_value = {
        "status": audit["status"],
        "checkpoint": audit["inputs"]["checkpoint"],
        "selected_physical_gpus": audit["config"]["selected_physical_gpus"],
        "cuda_visible_devices": audit["config"]["cuda_visible_devices"],
        "world_size": audit["config"]["world_size"],
        "num_envs_per_rank": audit["config"]["num_envs_per_rank"],
        "rollout_steps": audit["config"]["rollout_steps"],
        "total_env_steps": aggregate["total_env_steps"],
        "motion_count": aggregate["motion_count"],
        "total_motion_frames": aggregate["total_motion_frames"],
        "shard_count": aggregate["shard_count"],
        "dataset_npz_total_size_bytes": aggregate["dataset_npz_total_size_bytes"],
        "reward_mean_by_rank": aggregate["reward_mean_by_rank"],
        "done_count_total": aggregate["done_count_total"],
        "loaded_iteration_by_rank": [row.get("loaded_iteration") for row in shard_metrics],
        "duration_seconds": audit["run"].get("duration_seconds"),
        "gpu_metrics_summary": gpu_summary,
        "report_assets": assets["assets"],
    }
    rows.append(
        {
            "experiment": "tracking:official_csv_loop_full_bundle_teacher_rollout_dataset",
            "paper_value": (
                "BeyondMimic trains downstream VAE/diffusion components from teacher/Dagger-style state-latent "
                "trajectories. The official paper does not publish the teacher checkpoint, official DAgger rollout "
                "logs, or a directly comparable full-public-motion local rollout dataset."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Teacher rollout / DAgger trajectory data prerequisite",
            "paper_source": "reproduction/paper/source/root.tex;official whole_body_tracking task sources",
            "run_id": (
                "res/tracking/g1_official_csv_loop_full_bundle_teacher_rollout_dataset/"
                "tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset.json"
            ),
            "reproduction_level": "full-public-motion local virtual teacher rollout dataset gate",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The run collected two GPU shards from the local iteration-299 PPO checkpoint trained on the "
                "40-motion public official-loop bundle, recording observations, actions, rewards, done flags, "
                "timeouts, and motion timesteps for 306,176 virtual env steps. It is the strongest current local "
                "teacher-rollout dataset candidate, but it still uses the enriched-USD runtime patch and artificial "
                "bundle boundaries, so it is not the official BeyondMimic DAgger dataset or paper-level Fig. 5/"
                "Fig. 6 closed-loop evidence."
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


def add_tracking_official_replay_loop_patch_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/tracking/official_replay_npz_loop_with_enriched_usd/"
        "tracking_official_replay_npz_loop_with_enriched_usd_audit.json"
    )
    markers = audit["run"]["markers"]
    reproduction_value = {
        "status": audit["status"],
        "latest_blocker": audit["latest_blocker"],
        "app_launcher_constructed": audit["checks"]["app_launcher_constructed"],
        "g1_cfg_patched_to_enriched_usd": audit["checks"]["g1_cfg_patched_to_enriched_usd"],
        "fake_wandb_download_seen": audit["checks"]["fake_wandb_download_seen"],
        "official_loop_call_299_seen": audit["checks"]["official_loop_call_299_seen"],
        "official_loop_complete_seen": audit["checks"]["official_loop_complete_seen"],
        "simulation_app_close_called": markers["simulation_app_close_called"],
        "process_returned_zero_or_forced_after_success_sentinel": audit["checks"].get(
            "process_returned_zero_or_forced_after_success_sentinel",
            audit["checks"].get("process_returned_zero"),
        ),
    }
    rows.append(
        {
            "experiment": "tracking:official_replay_npz_loop_with_enriched_usd_patch",
            "paper_value": (
                "BeyondMimic replays/evaluates motion tracking in IsaacLab using the official G1 asset and motion "
                "pipeline; the paper does not publish a numeric replay-loop diagnostic metric."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion tracking replay / evaluation pipeline",
            "paper_source": "official whole_body_tracking scripts/replay_npz.py",
            "run_id": (
                "res/tracking/official_replay_npz_loop_with_enriched_usd/"
                "tracking_official_replay_npz_loop_with_enriched_usd_audit.json"
            ),
            "reproduction_level": "official replay loop with resource-adjusted runtime asset patch",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The audit executes the official replay_npz.py loop to the 299-step bound after runtime patching only "
                "the G1 robot config to use the validated resource-adjusted enriched USD and a local fake-WandB "
                "artifact for the official-CSV-derived motion. This is stronger than the copied local replay script, "
                "but it is still not official csv_to_npz.py output, not official URDF-converter output, not PPO, and "
                "not a paper-level tracking metric."
            ),
        }
    )


def add_tracking_official_replay_loop_full_dataset_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/tracking/official_replay_npz_loop_full_dataset_with_enriched_usd/"
        "tracking_official_replay_npz_loop_full_dataset_with_enriched_usd_audit.json"
    )
    aggregate = audit["aggregate"]
    reproduction_value = {
        "status": audit["status"],
        "row_count": aggregate["row_count"],
        "ok_count": aggregate["ok_count"],
        "failed_count": aggregate["failed_count"],
        "total_replayed_steps": aggregate["total_replayed_steps"],
        "all_40_csv_loop_outputs_selected": audit["checks"]["all_40_csv_loop_outputs_selected"],
        "all_rows_reached_official_loop_299": audit["checks"]["all_rows_reached_official_loop_299"],
        "all_rows_used_enriched_usd_patch": audit["checks"]["all_rows_used_enriched_usd_patch"],
        "uses_official_replay_npz_loop": audit["checks"]["uses_official_replay_npz_loop"],
        "uses_official_csv_loop_npz_inputs": audit["checks"]["uses_official_csv_loop_npz_inputs"],
        "uses_resource_adjusted_usd": audit["checks"]["uses_resource_adjusted_usd"],
    }
    rows.append(
        {
            "experiment": "tracking:official_replay_npz_loop_full_dataset_with_enriched_usd",
            "paper_value": (
                "BeyondMimic uses reference motion replay/evaluation inside the official tracking stack, but the "
                "paper does not publish a numeric full-dataset replay-loop diagnostic."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion preprocessing / tracking replay pipeline",
            "paper_source": "official whole_body_tracking scripts/replay_npz.py plus local G1 LAFAN motion bundle",
            "run_id": (
                "res/tracking/official_replay_npz_loop_full_dataset_with_enriched_usd/"
                "tracking_official_replay_npz_loop_full_dataset_with_enriched_usd_audit.json"
            ),
            "reproduction_level": "full public-motion official replay loop with resource-adjusted runtime asset patch",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The audit runs the official replay_npz.py loop body over all 40 NPZ files generated by the full "
                "official csv_to_npz.py loop audit, reaching the 299-step bound for every motion. This is strong "
                "full public-motion reference replay evidence for the loop body, but it still uses the enriched USD "
                "runtime patch and enriched-USD csv-loop inputs, so it is not unpatched official converter output, "
                "not trained-policy evaluation, and not a paper-level tracking metric."
            ),
        }
    )


def add_tracking_official_replay_loop_full_dataset_official_importer_export_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/tracking/official_replay_npz_loop_full_dataset_with_official_importer_export/"
        "tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_audit.json"
    )
    aggregate = audit["aggregate"]
    reproduction_value = {
        "status": audit["status"],
        "row_count": aggregate["row_count"],
        "ok_count": aggregate["ok_count"],
        "failed_count": aggregate["failed_count"],
        "total_replayed_steps": aggregate["total_replayed_steps"],
        "shutdown_warning_count": aggregate["shutdown_warning_count"],
        "all_40_csv_loop_outputs_selected": audit["checks"]["all_40_csv_loop_outputs_selected"],
        "all_rows_reached_official_loop_299": audit["checks"]["all_rows_reached_official_loop_299"],
        "all_rows_used_official_importer_export_usd": audit["checks"][
            "all_rows_used_official_importer_export_usd"
        ],
        "uses_official_replay_npz_loop": audit["checks"]["uses_official_replay_npz_loop"],
        "uses_official_csv_loop_npz_inputs": audit["checks"]["uses_official_csv_loop_npz_inputs"],
        "uses_official_importer_export_usd": audit["checks"]["uses_official_importer_export_usd"],
        "does_not_use_resource_adjusted_enriched_usd": audit["checks"][
            "does_not_use_resource_adjusted_enriched_usd"
        ],
    }
    rows.append(
        {
            "experiment": "tracking:official_replay_npz_loop_full_dataset_with_official_importer_export",
            "paper_value": (
                "BeyondMimic uses reference motion replay/evaluation inside the official tracking stack, but the "
                "paper does not publish a numeric full-dataset replay-loop diagnostic."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion preprocessing / tracking replay pipeline",
            "paper_source": "official whole_body_tracking scripts/replay_npz.py plus local G1 LAFAN motion bundle",
            "run_id": (
                "res/tracking/official_replay_npz_loop_full_dataset_with_official_importer_export/"
                "tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_audit.json"
            ),
            "reproduction_level": (
                "full public-motion official replay loop with captured official-importer-export G1 USDA"
            ),
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The audit runs the official replay_npz.py loop body over all 40 NPZ files generated by the matching "
                "official csv_to_npz.py loop audit, reaches the 299-step bound for every motion, and selects the "
                "G1 USDA captured from the official Isaac Sim URDF importer instead of the generated enriched-USD "
                "scaffold. It still bypasses the live unmodified official converter entry and is not trained-policy "
                "evaluation, DAgger, Fig. 5/Fig. 6, or a paper-level tracking metric."
            ),
        }
    )


def add_official_importer_export_replay_full_dataset_report_asset_rows(rows: list[dict[str, str]]) -> None:
    assets = load_json(
        "res/report_assets/official_importer_export_replay_full_dataset/"
        "official_importer_export_replay_full_dataset_report_assets.json"
    )
    reproduction_value = {
        "status": assets["status"],
        "source_status": assets["source_status"],
        "aggregate": assets["aggregate"],
        "family_summary": assets["family_summary"],
        "assets": assets["assets"],
        "checks": assets["checks"],
        "claim_level": assets["interpretation"]["claim_level"],
    }
    rows.append(
        {
            "experiment": "report_assets:official_importer_export_replay_full_dataset",
            "paper_value": (
                "BeyondMimic relies on reference motion replay and tracking infrastructure, but the paper does not "
                "publish a report-facing full-public-motion replay completion/duration table."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion preprocessing / reference replay report asset",
            "paper_source": "official whole_body_tracking scripts/replay_npz.py plus local G1 LAFAN motion bundle",
            "run_id": (
                "res/report_assets/official_importer_export_replay_full_dataset/"
                "official_importer_export_replay_full_dataset_report_assets.json"
            ),
            "reproduction_level": (
                "report assets for full public-motion official replay loop with captured official-importer-export G1 USDA"
            ),
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The report asset converts the 40/40 local official replay-loop audit into completion/failure-rate, "
                "duration, and family-summary plots and CSVs, and links the representative reference replay video. It "
                "is useful for the English report/PPT, but it remains local virtual reference-replay evidence: no "
                "trained policy evaluation, no unmodified live converter-entry success, no paper tracking metric, no "
                "Fig. 5/Fig. 6 guidance, and no real-robot result."
            ),
        }
    )


def add_tracking_official_csv_to_npz_loop_patch_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/tracking/official_csv_to_npz_loop_with_enriched_usd/"
        "tracking_official_csv_to_npz_loop_with_enriched_usd_audit.json"
    )
    metrics = audit["metrics"]
    reproduction_value = {
        "status": audit["status"],
        "latest_blocker": audit["latest_blocker"],
        "app_launcher_constructed": audit["checks"]["app_launcher_constructed"],
        "g1_cfg_patched_to_enriched_usd": audit["checks"]["g1_cfg_patched_to_enriched_usd"],
        "motion_loaded": audit["checks"]["motion_loaded"],
        "motion_interpolated": audit["checks"]["motion_interpolated"],
        "official_loop_call_299_seen": audit["checks"]["official_loop_call_299_seen"],
        "official_loop_complete_seen": audit["checks"]["official_loop_complete_seen"],
        "np_savez_redirect_seen": audit["checks"]["np_savez_redirect_seen"],
        "fake_wandb_log_artifact_seen": audit["checks"]["fake_wandb_log_artifact_seen"],
        "joint_pos_shape": metrics.get("joint_pos_shape"),
        "body_pos_w_shape": metrics.get("body_pos_w_shape"),
        "npz_size_bytes": metrics.get("npz_size_bytes"),
    }
    rows.append(
        {
            "experiment": "tracking:official_csv_to_npz_loop_with_enriched_usd_patch",
            "paper_value": (
                "BeyondMimic converts/replays G1 motion data through the official tracking preprocessing path; "
                "the paper does not publish a numeric csv_to_npz loop diagnostic metric."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion preprocessing / tracking replay pipeline",
            "paper_source": "official whole_body_tracking scripts/csv_to_npz.py",
            "run_id": (
                "res/tracking/official_csv_to_npz_loop_with_enriched_usd/"
                "tracking_official_csv_to_npz_loop_with_enriched_usd_audit.json"
            ),
            "reproduction_level": "official csv_to_npz loop with resource-adjusted runtime asset patch",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The audit executes the official csv_to_npz.py loop body to the 299-step bound, redirects the "
                "script's hard-coded /tmp motion output into the project result directory, and replaces wandb with "
                "a local fake registry. The robot config is patched in memory to use the validated resource-adjusted "
                "enriched USD, so the generated NPZ is not unpatched official converter output and is not a "
                "paper-level tracking replay metric."
            ),
        }
    )


def add_tracking_official_csv_to_npz_loop_full_dataset_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/tracking/official_csv_to_npz_loop_full_dataset_with_enriched_usd/"
        "tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_audit.json"
    )
    aggregate = audit["aggregate"]
    reproduction_value = {
        "status": audit["status"],
        "row_count": aggregate["row_count"],
        "ok_count": aggregate["ok_count"],
        "failed_count": aggregate["failed_count"],
        "total_frames": aggregate["total_frames"],
        "total_joint_values": aggregate["total_joint_values"],
        "all_40_csvs_selected": audit["checks"]["all_40_csvs_selected"],
        "all_joint_shapes_299_29": audit["checks"]["all_joint_shapes_299_29"],
        "all_body_shapes_299_40": audit["checks"]["all_body_shapes_299_40"],
        "uses_official_csv_to_npz_loop": audit["checks"]["uses_official_csv_to_npz_loop"],
        "uses_resource_adjusted_usd": audit["checks"]["uses_resource_adjusted_usd"],
    }
    rows.append(
        {
            "experiment": "tracking:official_csv_to_npz_loop_full_dataset_with_enriched_usd",
            "paper_value": (
                "BeyondMimic uses retargeted motion data for the tracking teacher, but the paper does not publish "
                "a numeric full-dataset csv_to_npz conversion diagnostic."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion preprocessing / tracking replay pipeline",
            "paper_source": "official whole_body_tracking scripts/csv_to_npz.py plus local G1 LAFAN CSV bundle",
            "run_id": (
                "res/tracking/official_csv_to_npz_loop_full_dataset_with_enriched_usd/"
                "tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_audit.json"
            ),
            "reproduction_level": "full public-motion official csv_to_npz loop with resource-adjusted runtime asset patch",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The audit runs the official csv_to_npz.py loop body over all 40 local G1 LAFAN CSV files, producing "
                "40 resource-adjusted project-local NPZ outputs with 299 frames, 29 joints, and 40 bodies per motion. "
                "This strengthens public-motion coverage beyond the previous single-motion gate, but it still patches "
                "the G1 config in memory to use the validated enriched USD, so it is not unpatched official converter "
                "output, not policy evaluation, and not a paper-level tracking result."
            ),
        }
    )


def add_tracking_official_csv_to_npz_loop_full_dataset_official_importer_export_rows(
    rows: list[dict[str, str]],
) -> None:
    audit = load_json(
        "res/tracking/official_csv_to_npz_loop_full_dataset_with_official_importer_export/"
        "tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_audit.json"
    )
    aggregate = audit["aggregate"]
    reproduction_value = {
        "status": audit["status"],
        "row_count": aggregate["row_count"],
        "ok_count": aggregate["ok_count"],
        "failed_count": aggregate["failed_count"],
        "total_frames": aggregate["total_frames"],
        "total_joint_values": aggregate["total_joint_values"],
        "total_npz_bytes": aggregate["total_npz_bytes"],
        "all_40_csvs_selected": audit["checks"]["all_40_csvs_selected"],
        "all_joint_shapes_299_29": audit["checks"]["all_joint_shapes_299_29"],
        "all_body_shapes_299_40": audit["checks"]["all_body_shapes_299_40"],
        "uses_official_csv_to_npz_loop": audit["checks"]["uses_official_csv_to_npz_loop"],
        "uses_official_importer_export_usd": audit["checks"]["uses_official_importer_export_usd"],
        "does_not_use_resource_adjusted_enriched_usd": audit["checks"][
            "does_not_use_resource_adjusted_enriched_usd"
        ],
    }
    rows.append(
        {
            "experiment": "tracking:official_csv_to_npz_loop_full_dataset_with_official_importer_export",
            "paper_value": (
                "BeyondMimic uses retargeted motion data for the tracking teacher, but the paper does not publish "
                "a numeric full-dataset csv_to_npz conversion diagnostic."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion preprocessing / tracking replay pipeline",
            "paper_source": "official whole_body_tracking scripts/csv_to_npz.py plus local G1 LAFAN CSV bundle",
            "run_id": (
                "res/tracking/official_csv_to_npz_loop_full_dataset_with_official_importer_export/"
                "tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_audit.json"
            ),
            "reproduction_level": (
                "full public-motion official csv_to_npz loop with captured official-importer-export G1 USDA"
            ),
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The audit runs the official csv_to_npz.py loop body over all 40 local G1 LAFAN CSV files and "
                "produces project-local NPZ outputs with the expected 299-frame, 29-joint, and 40-body shapes while "
                "selecting the G1 USDA captured from the official Isaac Sim URDF importer. This removes the generated "
                "enriched-USD scaffold from the full-loop test, but it still uses a captured importer export rather "
                "than the live unmodified converter entry, so it is not official paper-level preprocessing evidence "
                "or a tracking-policy metric."
            ),
        }
    )


def add_tracking_official_csv_loop_full_dataset_task_eval_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/tracking/g1_official_csv_loop_full_dataset_task_eval/"
        "tracking_g1_official_csv_loop_full_dataset_task_eval.json"
    )
    aggregate = audit["aggregate"]
    checks = audit["checks"]
    reproduction_value = {
        "status": audit["status"],
        "row_count": aggregate["row_count"],
        "ok_count": aggregate["ok_count"],
        "failed_count": aggregate["failed_count"],
        "total_steps": aggregate["total_steps"],
        "reward_mean": aggregate["reward_mean"]["mean"],
        "error_anchor_pos_mean": aggregate["error_anchor_pos"]["mean"],
        "error_body_pos_mean": aggregate["error_body_pos"]["mean"],
        "error_joint_pos_mean": aggregate["error_joint_pos"]["mean"],
        "all_rows_step_299": checks["all_rows_step_299"],
        "all_rows_action_dim_29": checks["all_rows_action_dim_29"],
        "all_rows_policy_obs_dim_160": checks["all_rows_policy_obs_dim_160"],
        "all_rows_critic_obs_dim_286": checks["all_rows_critic_obs_dim_286"],
        "all_rows_reward_terms_9": checks["all_rows_reward_terms_9"],
        "all_rows_termination_terms_4": checks["all_rows_termination_terms_4"],
        "all_rows_robot_contract_29j_40b": checks["all_rows_robot_contract_29j_40b"],
        "uses_official_csv_loop_npz_inputs": checks["uses_official_csv_loop_npz_inputs"],
        "uses_resource_adjusted_usd": checks["uses_resource_adjusted_usd"],
    }
    rows.append(
        {
            "experiment": "tracking:official_csv_loop_full_dataset_task_eval",
            "paper_value": (
                "BeyondMimic evaluates a trained motion-tracking teacher in IsaacLab, but the paper does not publish "
                "a numeric zero-action full-public-motion task diagnostic."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion tracking teacher / IsaacLab tracking task",
            "paper_source": "official whole_body_tracking Tracking-Flat-G1-v0 task plus local G1 LAFAN motion bundle",
            "run_id": (
                "res/tracking/g1_official_csv_loop_full_dataset_task_eval/"
                "tracking_g1_official_csv_loop_full_dataset_task_eval.json"
            ),
            "reproduction_level": "full public-motion resource-adjusted official-loop task diagnostic",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The audit feeds all 40 official csv-loop NPZ motions into Tracking-Flat-G1-v0 and reaches the "
                "299-step bound for every motion while validating policy/critic observation dimensions, action "
                "dimension, reward terms, termination terms, and robot joint/body counts. It uses zero diagnostic "
                "actions and the enriched-USD runtime patch, so it is not a trained-policy PPO evaluation, not "
                "unpatched official asset execution, not Fig. 5/Fig. 6, and not a paper-level tracking result."
            ),
        }
    )


def add_tracking_official_importer_export_full_dataset_task_eval_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/tracking/g1_official_importer_export_full_dataset_task_eval/"
        "tracking_g1_official_importer_export_full_dataset_task_eval.json"
    )
    aggregate = audit["aggregate"]
    checks = audit["checks"]
    reproduction_value = {
        "status": audit["status"],
        "row_count": aggregate["row_count"],
        "ok_count": aggregate["ok_count"],
        "failed_count": aggregate["failed_count"],
        "total_steps": aggregate["total_steps"],
        "reward_mean": aggregate["reward_mean"]["mean"],
        "error_anchor_pos_mean": aggregate["error_anchor_pos"]["mean"],
        "error_body_pos_mean": aggregate["error_body_pos"]["mean"],
        "error_joint_pos_mean": aggregate["error_joint_pos"]["mean"],
        "all_rows_step_299": checks["all_rows_step_299"],
        "all_rows_action_dim_29": checks["all_rows_action_dim_29"],
        "all_rows_policy_obs_dim_160": checks["all_rows_policy_obs_dim_160"],
        "all_rows_critic_obs_dim_286": checks["all_rows_critic_obs_dim_286"],
        "all_rows_reward_terms_9": checks["all_rows_reward_terms_9"],
        "all_rows_termination_terms_4": checks["all_rows_termination_terms_4"],
        "all_rows_robot_contract_29j_40b": checks["all_rows_robot_contract_29j_40b"],
        "all_rows_use_official_importer_export_usd": checks["all_rows_use_official_importer_export_usd"],
        "no_rows_use_resource_adjusted_enriched_usd": checks["no_rows_use_resource_adjusted_enriched_usd"],
    }
    rows.append(
        {
            "experiment": "tracking:official_importer_export_full_dataset_task_eval",
            "paper_value": (
                "BeyondMimic evaluates a trained motion-tracking teacher in IsaacLab, but the paper does not publish "
                "a numeric zero-action full-public-motion task diagnostic."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion tracking teacher / IsaacLab tracking task",
            "paper_source": "official whole_body_tracking Tracking-Flat-G1-v0 task plus local G1 LAFAN motion bundle",
            "run_id": (
                "res/tracking/g1_official_importer_export_full_dataset_task_eval/"
                "tracking_g1_official_importer_export_full_dataset_task_eval.json"
            ),
            "reproduction_level": "full public-motion official-importer-export task diagnostic",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The audit feeds all 40 official csv-loop NPZ motions into Tracking-Flat-G1-v0 and reaches the "
                "299-step bound for every motion while using the large GPU4 USDA exported by the official Isaac Sim "
                "URDF importer, not the resource-adjusted enriched scaffold. It validates policy/critic observation "
                "dimensions, action dimension, reward terms, termination terms, and robot joint/body counts. It still "
                "uses zero diagnostic actions and official-loop NPZs generated under the enriched-USD runtime patch, "
                "so it is not a trained-policy PPO evaluation, not unpatched official replay entry success, not "
                "Fig. 5/Fig. 6, and not a paper-level tracking result."
            ),
        }
    )


def add_tracking_official_importer_export_full_dataset_reference_replay_video_rows(
    rows: list[dict[str, str]]
) -> None:
    asset = load_json(
        "res/visualization/official_importer_export_full_dataset_reference_replay/"
        "official_importer_export_full_dataset_reference_replay_video_asset.json"
    )
    reproduction_value = {
        "status": asset["status"],
        "claim_level": asset["claim_level"],
        "selected_motion": asset["selected_motion"],
        "frame_count": asset["frame_count"],
        "body_count": asset["body_count"],
        "target_body_count": asset["target_body_count"],
        "source_dataset_aggregate": asset["source_dataset_aggregate"],
        "metrics": asset["metrics"],
        "assets": asset["assets"],
    }
    rows.append(
        {
            "experiment": "tracking:official_importer_export_full_dataset_reference_replay_video",
            "paper_value": (
                "BeyondMimic uses reference motion replay inside the tracking pipeline, but the paper does not "
                "publish a directly comparable kinematic reference-only visualization for the public motion bundle."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion preprocessing / reference replay visualization",
            "paper_source": "official whole_body_tracking scripts/csv_to_npz.py; scripts/replay_npz.py",
            "run_id": (
                "res/visualization/official_importer_export_full_dataset_reference_replay/"
                "official_importer_export_full_dataset_reference_replay_video_asset.json"
            ),
            "reproduction_level": "official-importer-export full-dataset local kinematic reference visualization",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The asset renders a representative saved reference body trajectory selected from the 40/40 "
                "official csv_to_npz full-dataset audit using the G1 USDA captured from the official Isaac Sim "
                "URDF importer. It is useful report/PPT media and visually documents the recovered reference "
                "trajectory path, but it is not an IsaacLab closed-loop policy rollout, not unmodified live official "
                "converter-entry output, not Fig. 5/Fig. 6 guided-diffusion evidence, and not real-robot validation."
            ),
        }
    )


def add_tracking_official_importer_export_full_bundle_ppo_rows(rows: list[dict[str, str]]) -> None:
    training = load_json(
        "res/tracking/g1_official_importer_export_full_bundle_ppo_training_run/"
        "tracking_g1_official_importer_export_full_bundle_ppo_training_run.json"
    )
    eval_audit = load_json(
        "res/tracking/g1_official_importer_export_full_bundle_ppo_checkpoint_eval/"
        "tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval.json"
    )
    assets = load_json(
        "res/report_assets/official_importer_export_full_bundle_ppo_checkpoint_eval/"
        "official_importer_export_full_bundle_ppo_checkpoint_eval_assets.json"
    )
    rank0 = next((item for item in training["run"].get("rank_metrics", []) if item.get("rank") == 0), {})
    metrics = eval_audit["run"].get("metrics", {})
    motion_metrics = metrics.get("motion_metrics", {})
    rows.append(
        {
            "experiment": "tracking:official_importer_export_full_bundle_ppo_training_run",
            "paper_value": (
                "BeyondMimic trains a motion-tracking teacher before DAgger/VAE/diffusion stages, but it does not "
                "publish a directly comparable 300-iteration public-bundle PPO metric or local USDA-export result."
            ),
            "reproduction_value": stringify(
                {
                    "status": training["status"],
                    "selected_physical_gpus": training["config"]["selected_physical_gpus"],
                    "world_size": training["config"]["world_size"],
                    "total_num_envs": training["config"]["total_num_envs"],
                    "num_steps_per_env": training["config"]["num_steps_per_env"],
                    "max_iterations": training["config"]["max_iterations"],
                    "duration_seconds": training["run"].get("duration_seconds"),
                    "checkpoint_count": training["run"].get("checkpoint_count"),
                    "rank0_learning_iteration": rank0.get("current_learning_iteration"),
                    "rank0_timesteps": rank0.get("tot_timesteps"),
                    "uses_official_importer_export_usd": rank0.get("uses_official_importer_export_usd"),
                    "motion_count": training["input_checks"].get("full_bundle_has_40_motions"),
                    "total_frames_11960": training["input_checks"].get("full_bundle_total_frames_11960"),
                }
            ),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion tracking teacher / PPO pipeline",
            "paper_source": "reproduction/paper/source/root.tex;official whole_body_tracking source",
            "run_id": (
                "res/tracking/g1_official_importer_export_full_bundle_ppo_training_run/"
                "tracking_g1_official_importer_export_full_bundle_ppo_training_run.json"
            ),
            "reproduction_level": "official-importer-export local virtual PPO training run",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This run uses GPUs 4 and 7 for a 300-iteration RSL-RL PPO job in `Tracking-Flat-G1-v0`, using the "
                "local USDA exported by the official Isaac Sim importer and the 40-motion public official-loop "
                "bundle. It removes the earlier resource-adjusted/enriched robot asset from the PPO path, but it is "
                "still a local exported asset with artificial bundle boundaries and no official BeyondMimic teacher "
                "checkpoint, so it remains below paper-level tracking training."
            ),
        }
    )
    rows.append(
        {
            "experiment": "tracking:official_importer_export_full_bundle_ppo_checkpoint_eval",
            "paper_value": (
                "BeyondMimic evaluates trained tracking policies before downstream data collection, but it does not "
                "publish this local public-bundle checkpoint-eval metric."
            ),
            "reproduction_value": stringify(
                {
                    "status": eval_audit["status"],
                    "checkpoint": eval_audit["inputs"]["checkpoint"],
                    "num_envs": eval_audit["config"]["num_envs"],
                    "eval_steps": eval_audit["config"]["eval_steps"],
                    "total_env_steps": eval_audit["config"]["total_env_steps"],
                    "loaded_iteration": metrics.get("loaded_iteration"),
                    "duration_seconds": eval_audit["run"].get("duration_seconds"),
                    "done_count_total": metrics.get("done_count_total"),
                    "reward_mean": metrics.get("reward", {}).get("mean_over_steps", {}).get("mean"),
                    "error_anchor_pos_mean": motion_metrics.get("error_anchor_pos", {}).get("mean"),
                    "error_body_pos_mean": motion_metrics.get("error_body_pos", {}).get("mean"),
                    "error_joint_pos_mean": motion_metrics.get("error_joint_pos", {}).get("mean"),
                    "motion_count": metrics.get("motion_count"),
                    "total_motion_frames": metrics.get("total_motion_frames"),
                    "report_assets": assets["assets"],
                }
            ),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion tracking teacher / PPO pipeline",
            "paper_source": "reproduction/paper/source/root.tex;official whole_body_tracking play.py source",
            "run_id": (
                "res/tracking/g1_official_importer_export_full_bundle_ppo_checkpoint_eval/"
                "tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval.json"
            ),
            "reproduction_level": "official-importer-export local virtual PPO checkpoint evaluation",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The evaluator loads the iteration-299 local PPO checkpoint and rolls out 512 environments x 299 "
                "steps with the official-importer-export USDA and full public motion bundle. It records report-ready "
                "training/eval curves, but it is not an official BeyondMimic checkpoint, not DAgger data quality "
                "evidence, not Fig. 5/Fig. 6, and not real-robot validation."
            ),
        }
    )


def add_tracking_official_importer_export_full_bundle_scaled_ppo_rows(rows: list[dict[str, str]]) -> None:
    training = load_json(
        "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_training_run/"
        "tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run.json"
    )
    eval_audit = load_json(
        "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/"
        "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json"
    )
    assets = load_json(
        "res/report_assets/official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/"
        "official_importer_export_full_bundle_scaled_ppo_checkpoint_eval_assets.json"
    )
    policy_video = load_json(
        "res/visualization/official_importer_export_full_bundle_scaled_ppo_policy_rollout/"
        "official_importer_export_full_bundle_scaled_ppo_policy_rollout_video_asset.json"
    )
    rank0 = next((item for item in training["run"].get("rank_metrics", []) if item.get("rank") == 0), {})
    metrics = eval_audit["run"].get("metrics", {})
    motion_metrics = metrics.get("motion_metrics", {})
    rows.append(
        {
            "experiment": "tracking:official_importer_export_full_bundle_scaled_ppo_training_run",
            "paper_value": (
                "BeyondMimic reports a motion-tracking teacher pipeline trained at paper scale, but does not publish "
                "a directly comparable scaled local public-bundle PPO metric for the official-importer-export USDA."
            ),
            "reproduction_value": stringify(
                {
                    "status": training["status"],
                    "selected_physical_gpus": training["config"]["selected_physical_gpus"],
                    "world_size": training["config"]["world_size"],
                    "total_num_envs": training["config"]["total_num_envs"],
                    "num_steps_per_env": training["config"]["num_steps_per_env"],
                    "max_iterations": training["config"]["max_iterations"],
                    "duration_seconds": training["run"].get("duration_seconds"),
                    "checkpoint_count": training["run"].get("checkpoint_count"),
                    "rank0_learning_iteration": rank0.get("current_learning_iteration"),
                    "rank0_timesteps": rank0.get("tot_timesteps"),
                    "formal_gpu_memory_target_mb_per_card": training["config"].get(
                        "formal_gpu_memory_target_mb_per_card"
                    ),
                    "uses_official_importer_export_usd": rank0.get("uses_official_importer_export_usd"),
                }
            ),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion tracking teacher / PPO pipeline",
            "paper_source": "reproduction/paper/source/root.tex;official whole_body_tracking source",
            "run_id": (
                "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_training_run/"
                "tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run.json"
            ),
            "reproduction_level": "official-importer-export larger local virtual PPO training run",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This larger run is intended to move beyond the earlier 300-iteration engineering run by using more "
                "envs and more PPO iterations on GPUs 4/7. It remains a local public-bundle training job with a local "
                "official-importer USDA and no official BeyondMimic teacher checkpoint, so it is not paper-level "
                "tracking reproduction."
            ),
        }
    )
    rows.append(
        {
            "experiment": "tracking:official_importer_export_full_bundle_scaled_ppo_policy_rollout_video",
            "paper_value": (
                "BeyondMimic uses a trained motion-tracking teacher and reports qualitative robot behavior, but it "
                "does not publish a directly comparable single-env local scaled-PPO policy-vs-reference video."
            ),
            "reproduction_value": stringify(
                {
                    "status": policy_video["status"],
                    "claim_level": policy_video["claim_level"],
                    "frame_count": policy_video["frame_count"],
                    "target_body_count": policy_video["target_body_count"],
                    "bundle": policy_video.get("bundle", {}),
                    "metrics": policy_video["metrics"],
                    "assets": policy_video["assets"],
                }
            ),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion tracking teacher / local report video",
            "paper_source": "reproduction/paper/source/root.tex;official whole_body_tracking source",
            "run_id": (
                "res/visualization/official_importer_export_full_bundle_scaled_ppo_policy_rollout/"
                "official_importer_export_full_bundle_scaled_ppo_policy_rollout_video_asset.json"
            ),
            "reproduction_level": "official-importer-export scaled local virtual PPO policy rollout video",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The video captures a 299-frame local policy-vs-reference rollout from the iteration-999 scaled PPO "
                "checkpoint on the official-importer-export G1 USDA and the 40-motion public bundle. It is useful "
                "report/PPT media for the recovered tracking pipeline, but it is not an official BeyondMimic "
                "tracking teacher checkpoint, not a published Fig. 5/Fig. 6 guided-diffusion rollout, not TensorRT "
                "deployment, and not real-robot evidence."
            ),
        }
    )
    rows.append(
        {
            "experiment": "tracking:official_importer_export_full_bundle_scaled_ppo_checkpoint_eval",
            "paper_value": (
                "BeyondMimic requires tracking-teacher evaluation before downstream DAgger/VAE/diffusion, but this "
                "scaled local checkpoint-eval metric is not a paper-published value."
            ),
            "reproduction_value": stringify(
                {
                    "status": eval_audit["status"],
                    "checkpoint": eval_audit["inputs"]["checkpoint"],
                    "num_envs": eval_audit["config"]["num_envs"],
                    "eval_steps": eval_audit["config"]["eval_steps"],
                    "total_env_steps": eval_audit["config"]["total_env_steps"],
                    "loaded_iteration": metrics.get("loaded_iteration"),
                    "duration_seconds": eval_audit["run"].get("duration_seconds"),
                    "done_count_total": metrics.get("done_count_total"),
                    "reward_mean": metrics.get("reward", {}).get("mean_over_steps", {}).get("mean"),
                    "error_anchor_pos_mean": motion_metrics.get("error_anchor_pos", {}).get("mean"),
                    "error_body_pos_mean": motion_metrics.get("error_body_pos", {}).get("mean"),
                    "error_joint_pos_mean": motion_metrics.get("error_joint_pos", {}).get("mean"),
                    "motion_count": metrics.get("motion_count"),
                    "total_motion_frames": metrics.get("total_motion_frames"),
                    "report_assets": assets["assets"],
                }
            ),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion tracking teacher / PPO pipeline",
            "paper_source": "reproduction/paper/source/root.tex;official whole_body_tracking play.py source",
            "run_id": (
                "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/"
                "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json"
            ),
            "reproduction_level": "official-importer-export larger local virtual PPO checkpoint evaluation",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The evaluation loads the scaled local PPO checkpoint and rolls it out in Tracking-Flat-G1-v0 with "
                "the official-importer-export USDA and full public motion bundle. It gives stronger local virtual "
                "tracking evidence than a smoke test, but remains below official paper-level tracking evaluation."
            ),
        }
    )


def add_official_importer_export_tracking_eval_summary_asset_rows(rows: list[dict[str, str]]) -> None:
    summary = load_json(
        "res/report_assets/official_importer_export_tracking_eval_summary/"
        "official_importer_export_tracking_eval_summary_assets.json"
    )
    metrics = summary["metrics"]
    rows.append(
        {
            "experiment": "report_assets:official_importer_export_tracking_eval_summary",
            "paper_value": (
                "BeyondMimic requires a strong motion-tracking teacher before DAgger/VAE/diffusion, but the paper does "
                "not publish a directly comparable local official-importer-export public-bundle tracking-summary metric."
            ),
            "reproduction_value": stringify(
                {
                    "status": summary["status"],
                    "claim_level": summary["interpretation"]["claim_level"],
                    "task_diagnostic": metrics["full_dataset_task_diagnostic"],
                    "scaled_ppo_checkpoint_eval": metrics["scaled_ppo_checkpoint_eval"],
                    "scaled_ppo_policy_video": metrics["scaled_ppo_policy_video"],
                    "checks": summary["checks"],
                    "assets": summary["assets"],
                }
            ),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Motion tracking teacher / local virtual tracking evidence",
            "paper_source": "reproduction/paper/source/root.tex;official whole_body_tracking source",
            "run_id": (
                "res/report_assets/official_importer_export_tracking_eval_summary/"
                "official_importer_export_tracking_eval_summary_assets.json"
            ),
            "reproduction_level": "official-importer-export tracking evaluation summary report asset",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This report asset links the 40/40 official-importer-export task diagnostic, the scaled PPO checkpoint "
                "evaluation, and the scaled single-env policy video into one reading-report evidence bundle. It is useful "
                "for explaining the recovered virtual tracking pipeline, but it is not an official BeyondMimic teacher, "
                "not a paper metric, not DAgger data, not Fig. 5/Fig. 6 guided-diffusion evidence, not TensorRT "
                "deployment, and not real-robot validation."
            ),
        }
    )


def add_tracking_official_importer_export_full_bundle_teacher_rollout_dataset_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/tracking/g1_official_importer_export_full_bundle_teacher_rollout_dataset/"
        "tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset.json"
    )
    assets = load_json(
        "res/report_assets/official_importer_export_full_bundle_teacher_rollout_dataset/"
        "official_importer_export_full_bundle_teacher_rollout_report_assets.json"
    )
    aggregate = audit["aggregate_metrics"]
    shard_metrics = audit["run"].get("shard_metrics", [])
    reproduction_value = {
        "status": audit["status"],
        "training_run_json": audit["inputs"]["training_run_json"],
        "checkpoint_eval_json": audit["inputs"]["checkpoint_eval_json"],
        "official_importer_usd": audit["inputs"]["official_importer_usd"],
        "selected_physical_gpus": audit["config"]["selected_physical_gpus"],
        "cuda_visible_devices": audit["config"]["cuda_visible_devices"],
        "world_size": audit["config"]["world_size"],
        "num_envs_per_rank": audit["config"]["num_envs_per_rank"],
        "rollout_steps": audit["config"]["rollout_steps"],
        "total_env_steps": aggregate["total_env_steps"],
        "motion_count": aggregate["motion_count"],
        "total_motion_frames": aggregate["total_motion_frames"],
        "shard_count": aggregate["shard_count"],
        "dataset_npz_total_size_bytes": aggregate["dataset_npz_total_size_bytes"],
        "reward_mean_by_rank": aggregate["reward_mean_by_rank"],
        "done_count_total": aggregate["done_count_total"],
        "loaded_iteration_by_rank": [row.get("loaded_iteration") for row in shard_metrics],
        "uses_official_importer_export_usd_by_rank": [
            row.get("uses_official_importer_export_usd") for row in shard_metrics
        ],
        "duration_seconds": audit["run"].get("duration_seconds"),
        "gpu_metrics_summary": audit["run"].get("gpu_metrics_summary", {}),
        "report_assets": assets["assets"],
    }
    rows.append(
        {
            "experiment": "tracking:official_importer_export_full_bundle_teacher_rollout_dataset",
            "paper_value": (
                "BeyondMimic trains downstream VAE/diffusion components from teacher/Dagger-style state-latent "
                "trajectories. The paper does not publish the official teacher checkpoint, official DAgger rollout "
                "logs, or a directly comparable public-bundle rollout dataset."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Teacher rollout / DAgger trajectory data prerequisite",
            "paper_source": "reproduction/paper/source/root.tex;official whole_body_tracking task sources",
            "run_id": (
                "res/tracking/g1_official_importer_export_full_bundle_teacher_rollout_dataset/"
                "tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset.json"
            ),
            "reproduction_level": "official-importer-export local virtual teacher rollout dataset gate",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "The run collected two GPU shards from the local iteration-299 PPO checkpoint trained with the "
                "official-importer-export G1 USDA and the 40-motion public bundle, recording observations, actions, "
                "rewards, done flags, timeouts, and motion timesteps for 306,176 virtual env steps. This is the "
                "strongest current local teacher-data candidate on the more official robot-asset path, but it is "
                "still a local virtual dataset from a short PPO run, not the official BeyondMimic DAgger dataset, "
                "not Fig. 5/Fig. 6 closed-loop guidance, and not real-robot evidence."
            ),
        }
    )


def add_tracking_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset_rows(
    rows: list[dict[str, str]],
) -> None:
    audit = load_json(
        "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset/"
        "tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset.json"
    )
    assets = load_json(
        "res/report_assets/official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset/"
        "official_importer_export_full_bundle_teacher_rollout_report_assets.json"
    )
    aggregate = audit["aggregate_metrics"]
    shard_metrics = audit["run"].get("shard_metrics", [])
    reproduction_value = {
        "status": audit["status"],
        "training_run_json": audit["inputs"]["training_run_json"],
        "checkpoint_eval_json": audit["inputs"]["checkpoint_eval_json"],
        "checkpoint": audit["inputs"].get("checkpoint"),
        "official_importer_usd": audit["inputs"]["official_importer_usd"],
        "selected_physical_gpus": audit["config"]["selected_physical_gpus"],
        "cuda_visible_devices": audit["config"]["cuda_visible_devices"],
        "world_size": audit["config"]["world_size"],
        "num_envs_per_rank": audit["config"]["num_envs_per_rank"],
        "rollout_steps": audit["config"]["rollout_steps"],
        "total_env_steps": aggregate["total_env_steps"],
        "motion_count": aggregate["motion_count"],
        "total_motion_frames": aggregate["total_motion_frames"],
        "shard_count": aggregate["shard_count"],
        "dataset_npz_total_size_bytes": aggregate["dataset_npz_total_size_bytes"],
        "reward_mean_by_rank": aggregate["reward_mean_by_rank"],
        "done_count_total": aggregate["done_count_total"],
        "loaded_iteration_by_rank": [row.get("loaded_iteration") for row in shard_metrics],
        "uses_official_importer_export_usd_by_rank": [
            row.get("uses_official_importer_export_usd") for row in shard_metrics
        ],
        "duration_seconds": audit["run"].get("duration_seconds"),
        "gpu_metrics_summary": audit["run"].get("gpu_metrics_summary", {}),
        "report_assets": assets["assets"],
    }
    rows.append(
        {
            "experiment": "tracking:official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset",
            "paper_value": (
                "BeyondMimic's downstream VAE/diffusion stages depend on teacher/DAgger-style trajectories. "
                "The official teacher checkpoint and DAgger rollout logs are not publicly released."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Teacher rollout / DAgger trajectory data prerequisite",
            "paper_source": "reproduction/paper/source/root.tex;official whole_body_tracking task sources",
            "run_id": (
                "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset/"
                "tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset.json"
            ),
            "reproduction_level": "official-importer-export scaled local virtual teacher rollout dataset gate",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This run collected two full GPU shards from the local iteration-999 scaled PPO checkpoint on the "
                "official-importer-export G1 USDA and 40-motion public bundle, yielding 1,224,704 virtual env steps. "
                "It supersedes the older iteration-299 importer-export teacher-rollout candidate for future local "
                "downstream experiments, but it remains a local virtual dataset, not the official BeyondMimic DAgger "
                "dataset, not paper Fig. 5/Fig. 6 guidance, and not real-robot evidence."
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


def add_official_csv_loop_teacher_rollout_vae_training_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/level_c/official_csv_loop_teacher_rollout_vae_training/"
        "level_c_official_csv_loop_teacher_rollout_vae_training.json"
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
            "experiment": "level_c:official_csv_loop_teacher_rollout_vae_training",
            "paper_value": (
                "BeyondMimic trains a conditional VAE on DAgger/teacher trajectory data before state-latent "
                "diffusion; the paper does not release the official teacher rollout dataset or VAE checkpoint."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Conditional VAE / DAgger trajectory prerequisite",
            "paper_source": "BeyondMimic method sections and local official-loop teacher-rollout evidence",
            "run_id": (
                "res/level_c/official_csv_loop_teacher_rollout_vae_training/"
                "level_c_official_csv_loop_teacher_rollout_vae_training.json"
            ),
            "reproduction_level": "official csv-loop teacher-rollout VAE training",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This full local VAE training run uses the teacher rollout shards collected from the iteration-299 "
                "official-loop-motion PPO checkpoint. It is stronger downstream evidence than the earlier model-99 "
                "resource-adjusted VAE, but it is still not the paper's official DAgger dataset, not an official VAE "
                "checkpoint, and not closed-loop VAE/diffusion evaluation."
            ),
        }
    )


def add_official_csv_loop_state_latent_dataset_and_diffusion_rows(rows: list[dict[str, str]]) -> None:
    dataset_audit = load_json(
        "res/level_c/official_csv_loop_teacher_rollout_state_latent_dataset/"
        "level_c_official_csv_loop_teacher_rollout_state_latent_dataset.json"
    )
    diffusion_audit = load_json(
        "res/level_c/official_csv_loop_state_latent_diffusion_training/"
        "level_c_official_csv_loop_state_latent_diffusion_training.json"
    )
    dataset_worker = dataset_audit["worker_summary"]
    diffusion_worker = diffusion_audit["worker_summary"]
    dataset_value = {
        "status": dataset_audit["status"],
        "source_teacher_status": dataset_worker["source_teacher_rollout"]["status"],
        "source_vae_status": dataset_worker["source_vae"]["status"],
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
        "gpu_metrics_summary": dataset_audit.get("gpu_metrics_summary", {}),
    }
    rows.append(
        {
            "experiment": "level_c:official_csv_loop_teacher_rollout_state_latent_dataset",
            "paper_value": (
                "BeyondMimic trains diffusion on state-latent trajectories from teacher/DAgger rollouts and a "
                "trained conditional VAE; the official DAgger/state-latent dataset is not released."
            ),
            "reproduction_value": stringify(dataset_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "State-latent trajectory dataset prerequisite",
            "paper_source": "BeyondMimic method sections and local official-loop rollout/VAE evidence",
            "run_id": (
                "res/level_c/official_csv_loop_teacher_rollout_state_latent_dataset/"
                "level_c_official_csv_loop_teacher_rollout_state_latent_dataset.json"
            ),
            "reproduction_level": "official csv-loop teacher-rollout state-latent dataset",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This builds full local windows from the official-csv-loop teacher rollout shards and the local "
                "official-csv-loop VAE posterior. It is closer to the paper pipeline than the earlier model-99 "
                "resource-adjusted chain, but it is still not the unreleased official DAgger/state-latent dataset "
                "and does not validate closed-loop guidance."
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
            "experiment": "level_c:official_csv_loop_state_latent_diffusion_training",
            "paper_value": (
                "BeyondMimic trains a state-latent diffusion model and evaluates it with guided closed-loop humanoid "
                "control; the official training data and checkpoint are not released."
            ),
            "reproduction_value": stringify(diffusion_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "State-latent diffusion training / Fig. 5-6 prerequisite",
            "paper_source": "BeyondMimic method sections and local official-loop state-latent dataset evidence",
            "run_id": (
                "res/level_c/official_csv_loop_state_latent_diffusion_training/"
                "level_c_official_csv_loop_state_latent_diffusion_training.json"
            ),
            "reproduction_level": "official csv-loop full state-latent denoiser training",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This trains a local denoising model on all official-csv-loop state-latent windows and reports "
                "held-out denoising improvement. It is not the official BeyondMimic diffusion checkpoint, not "
                "TensorRT/asynchronous deployment, and not closed-loop Fig. 5/Fig. 6 guidance evaluation."
            ),
        }
    )


def add_official_csv_loop_full_bundle_downstream_rows(rows: list[dict[str, str]]) -> None:
    vae_audit = load_json(
        "res/level_c/official_csv_loop_full_bundle_teacher_rollout_vae_training/"
        "level_c_official_csv_loop_full_bundle_teacher_rollout_vae_training.json"
    )
    dataset_audit = load_json(
        "res/level_c/official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset/"
        "level_c_official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset.json"
    )
    diffusion_audit = load_json(
        "res/level_c/official_csv_loop_full_bundle_state_latent_diffusion_training/"
        "level_c_official_csv_loop_full_bundle_state_latent_diffusion_training.json"
    )
    assets = load_json(
        "res/report_assets/official_csv_loop_full_bundle_downstream/"
        "official_csv_loop_full_bundle_downstream_report_assets.json"
    )
    vae_worker = vae_audit["worker_summary"]
    dataset_worker = dataset_audit["worker_summary"]
    diffusion_worker = diffusion_audit["worker_summary"]
    rows.append(
        {
            "experiment": "level_c:official_csv_loop_full_bundle_teacher_rollout_vae_training",
            "paper_value": (
                "BeyondMimic trains a conditional VAE on teacher/DAGGER trajectory data. The paper does not release "
                "the official full teacher rollout dataset or VAE checkpoint."
            ),
            "reproduction_value": stringify(
                {
                    "status": vae_audit["status"],
                    "source_teacher_status": vae_worker["source_teacher_rollout"]["status"],
                    "sample_count": vae_worker["dataset"]["sample_count"],
                    "motion_time_step_max": vae_worker["dataset"]["motion_time_step_max"],
                    "splits": vae_worker["splits"],
                    "epochs": vae_worker["training"]["epochs"],
                    "test_action_mse": vae_worker["evaluation"]["test"]["action_mse"],
                    "test_action_abs_error_mean": vae_worker["evaluation"]["test"]["action_abs_error_mean"],
                    "report_assets": assets["assets"],
                }
            ),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Conditional VAE / DAgger trajectory prerequisite",
            "paper_source": "BeyondMimic method sections and local full-bundle teacher-rollout evidence",
            "run_id": (
                "res/level_c/official_csv_loop_full_bundle_teacher_rollout_vae_training/"
                "level_c_official_csv_loop_full_bundle_teacher_rollout_vae_training.json"
            ),
            "reproduction_level": "full-public-motion teacher-rollout VAE training",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This VAE trains on local teacher rollout shards collected from the 40-motion public official-loop "
                "bundle. It broadens data coverage beyond the single-motion official-loop chain, but it remains a "
                "local virtual artifact rather than the paper's official DAgger/VAE checkpoint or closed-loop VAE "
                "evaluation."
            ),
        }
    )
    rows.append(
        {
            "experiment": "level_c:official_csv_loop_full_bundle_state_latent_dataset_and_diffusion_training",
            "paper_value": (
                "BeyondMimic trains a state-latent diffusion model from teacher rollouts and evaluates it through "
                "guided closed-loop humanoid control; official training data/checkpoints are not released."
            ),
            "reproduction_value": stringify(
                {
                    "state_latent_status": dataset_audit["status"],
                    "diffusion_status": diffusion_audit["status"],
                    "sample_count": dataset_worker["dataset"]["sample_count"],
                    "window_count": dataset_worker["dataset"]["window_count"],
                    "split_counts": dataset_worker["dataset"]["split_counts"],
                    "weighted_posterior_reconstruction_mse": dataset_worker["dataset"][
                        "weighted_posterior_reconstruction_mse"
                    ],
                    "diffusion_epochs": diffusion_worker["training"]["epochs"],
                    "test_pred_token_mse": diffusion_worker["evaluation"]["test"]["pred_token_mse"],
                    "test_noisy_token_mse": diffusion_worker["evaluation"]["test"]["noisy_token_mse"],
                    "test_denoising_improvement_ratio": diffusion_worker["evaluation"]["test"][
                        "denoising_improvement_ratio"
                    ],
                    "report_assets": assets["assets"],
                }
            ),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "State-latent diffusion training / Fig. 5-6 prerequisite",
            "paper_source": "BeyondMimic method sections and local full-bundle state-latent evidence",
            "run_id": (
                "res/level_c/official_csv_loop_full_bundle_state_latent_diffusion_training/"
                "level_c_official_csv_loop_full_bundle_state_latent_diffusion_training.json"
            ),
            "reproduction_level": "full-public-motion state-latent denoiser training",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This trains a local denoiser on all full-bundle state-latent windows and records held-out "
                "denoising improvement plus report-ready curves. It is not the official BeyondMimic diffusion "
                "checkpoint, not TensorRT/asynchronous deployment, and not closed-loop Fig. 5/Fig. 6 guidance "
                "evaluation."
            ),
        }
    )


def add_official_importer_export_full_bundle_vae_rows(rows: list[dict[str, str]]) -> None:
    vae_audit = load_json(
        "res/level_c/official_importer_export_full_bundle_teacher_rollout_vae_training/"
        "level_c_official_importer_export_full_bundle_teacher_rollout_vae_training.json"
    )
    assets = load_json(
        "res/report_assets/official_importer_export_full_bundle_vae_training/"
        "official_importer_export_full_bundle_vae_training_assets.json"
    )
    worker = vae_audit["worker_summary"]
    rows.append(
        {
            "experiment": "level_c:official_importer_export_full_bundle_teacher_rollout_vae_training",
            "paper_value": (
                "BeyondMimic trains a conditional VAE from teacher/DAgger trajectory data, but the paper does not "
                "release the official DAgger logs or VAE checkpoint."
            ),
            "reproduction_value": stringify(
                {
                    "status": vae_audit["status"],
                    "source_teacher_status": worker["source_teacher_rollout"]["status"],
                    "sample_count": worker["dataset"]["sample_count"],
                    "motion_time_step_max": worker["dataset"]["motion_time_step_max"],
                    "splits": worker["splits"],
                    "epochs": worker["training"]["epochs"],
                    "test_action_mse": worker["evaluation"]["test"]["action_mse"],
                    "test_action_abs_error_mean": worker["evaluation"]["test"]["action_abs_error_mean"],
                    "uses_official_importer_export_usd": vae_audit["checks"]["uses_official_importer_export_usd"],
                    "report_assets": assets["assets"],
                }
            ),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Conditional VAE / DAgger trajectory prerequisite",
            "paper_source": "BeyondMimic method sections and local official-importer teacher-rollout evidence",
            "run_id": (
                "res/level_c/official_importer_export_full_bundle_teacher_rollout_vae_training/"
                "level_c_official_importer_export_full_bundle_teacher_rollout_vae_training.json"
            ),
            "reproduction_level": "official-importer-export local full-bundle VAE training",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This trains the local conditional action VAE on the two-shard teacher rollout dataset collected "
                "from the official-importer-export G1 USDA PPO checkpoint and 40-motion public bundle. It is the "
                "strongest local VAE training source on the more official robot-asset path, but the source policy is "
                "a short local PPO checkpoint and the result is not the official BeyondMimic DAgger/VAE checkpoint "
                "or a closed-loop Fig. 5/Fig. 6 result."
            ),
        }
    )


def add_official_importer_export_full_bundle_vae_closed_loop_rows(rows: list[dict[str, str]]) -> None:
    rollout = load_json(
        "res/level_c/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/"
        "tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_eval.json"
    )
    assets = load_json(
        "res/report_assets/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/"
        "official_importer_export_full_bundle_vae_closed_loop_rollout_assets.json"
    )
    video_asset = load_json(
        "res/visualization/official_importer_export_full_bundle_vae_closed_loop_rollout/"
        "official_importer_export_full_bundle_vae_closed_loop_rollout_video_asset.json"
    )
    aggregate = rollout["run"]["aggregate_metrics"]
    reproduction_value = {
        "status": rollout["status"],
        "total_num_envs": aggregate["total_num_envs"],
        "rollout_steps": aggregate["rollout_steps"],
        "total_env_steps": aggregate["total_env_steps"],
        "teacher_vae_action_mse_mean": aggregate["teacher_vae_action_mse"]["mean"],
        "teacher_vae_action_abs_error_mean": aggregate["teacher_vae_action_abs_error"]["mean"],
        "reward_mean": aggregate["reward_mean"]["mean"],
        "done_count_total": aggregate["done_count_total"],
        "timeout_count_total": aggregate["timeout_count_total"],
        "gpu_metrics_summary": rollout["run"]["gpu_metrics_summary"],
        "peak_memory_each_gpu_at_least_10gb": rollout["checks"]["peak_memory_each_gpu_at_least_10gb"],
        "uses_official_importer_export_usd": rollout["checks"]["uses_official_importer_export_usd"],
        "assets": assets["assets"],
        "video_asset": video_asset.get("assets", {}),
        "video_metrics": video_asset.get("metrics", {}),
        "video_claim_level": video_asset.get("claim_level", ""),
    }
    rows.append(
        {
            "experiment": "level_c:official_importer_export_full_bundle_vae_closed_loop_rollout_eval",
            "paper_value": (
                "BeyondMimic uses a conditional VAE inside a closed-loop latent-control stack, but the official "
                "VAE checkpoint, DAgger logs, and Fig. 5/Fig. 6 rollout logs are not public in this artifact set."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Conditional VAE / guided-control pipeline",
            "paper_source": "BeyondMimic VAE and guided diffusion method sections",
            "run_id": (
                "res/level_c/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/"
                "tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_eval.json"
            ),
            "reproduction_level": "official-importer-export local VAE action-reconstruction closed-loop rollout",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This full 299-step, two-rank IsaacLab run uses the official-importer-export G1 USDA, local "
                "40-motion PPO teacher checkpoint, and local full-bundle conditional action VAE. It covers "
                "918528 simulated env steps and shows low teacher/VAE action reconstruction error, but all env-step "
                "done counts are still high, the source teacher is a short local PPO run, per-GPU memory remains "
                "below the requested 10GB/card threshold, and the result is not the official BeyondMimic VAE "
                "checkpoint, autonomous VAE policy, receding-horizon guided diffusion, Fig. 5/Fig. 6 reproduction, "
                "TensorRT deployment, or real-robot evidence. A companion single-env MP4/keyframe asset is useful "
                "for the English report and slides, but it is also local qualitative evidence only."
            ),
        }
    )


def add_official_importer_export_full_bundle_downstream_rows(rows: list[dict[str, str]]) -> None:
    dataset_audit = load_json(
        "res/level_c/official_importer_export_full_bundle_teacher_rollout_state_latent_dataset/"
        "level_c_official_importer_export_full_bundle_teacher_rollout_state_latent_dataset.json"
    )
    diffusion_audit = load_json(
        "res/level_c/official_importer_export_full_bundle_state_latent_diffusion_training/"
        "level_c_official_importer_export_full_bundle_state_latent_diffusion_training.json"
    )
    assets = load_json(
        "res/report_assets/official_importer_export_full_bundle_downstream/"
        "official_importer_export_full_bundle_downstream_report_assets.json"
    )
    dataset_worker = dataset_audit["worker_summary"]
    diffusion_worker = diffusion_audit["worker_summary"]
    rows.append(
        {
            "experiment": "level_c:official_importer_export_full_bundle_state_latent_dataset_and_diffusion_training",
            "paper_value": (
                "BeyondMimic trains a state-latent diffusion model from teacher/DAgger rollouts and evaluates it "
                "through guided closed-loop humanoid control. The official state-latent dataset and diffusion "
                "checkpoint are not released."
            ),
            "reproduction_value": stringify(
                {
                    "state_latent_status": dataset_audit["status"],
                    "diffusion_status": diffusion_audit["status"],
                    "sample_count": dataset_worker["dataset"]["sample_count"],
                    "window_count": dataset_worker["dataset"]["window_count"],
                    "split_counts": dataset_worker["dataset"]["split_counts"],
                    "weighted_posterior_reconstruction_mse": dataset_worker["dataset"][
                        "weighted_posterior_reconstruction_mse"
                    ],
                    "uses_official_importer_export_usd": diffusion_audit["checks"][
                        "uses_official_importer_export_usd"
                    ],
                    "cuda_visible_devices": diffusion_audit["settings"]["cuda_visible_devices"],
                    "diffusion_epochs": diffusion_worker["training"]["epochs"],
                    "test_pred_token_mse": diffusion_worker["evaluation"]["test"]["pred_token_mse"],
                    "test_noisy_token_mse": diffusion_worker["evaluation"]["test"]["noisy_token_mse"],
                    "test_denoising_improvement_ratio": diffusion_worker["evaluation"]["test"][
                        "denoising_improvement_ratio"
                    ],
                    "report_assets": assets["assets"],
                }
            ),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "State-latent diffusion training / Fig. 5-6 prerequisite",
            "paper_source": "BeyondMimic method sections and local official-importer-export downstream evidence",
            "run_id": (
                "res/level_c/official_importer_export_full_bundle_state_latent_diffusion_training/"
                "level_c_official_importer_export_full_bundle_state_latent_diffusion_training.json"
            ),
            "reproduction_level": "official-importer-export local state-latent denoiser training",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This trains a local denoiser on all official-importer-export state-latent windows and records "
                "held-out denoising improvement plus report-ready curves. It uses the more official G1 USDA chain "
                "for the source teacher data, but it is still a local virtual model trained from a short local PPO "
                "teacher and local VAE. It is not the official BeyondMimic diffusion checkpoint, not TensorRT or "
                "asynchronous deployment, not closed-loop Fig. 5/Fig. 6 guidance evaluation, and not real-robot "
                "evidence."
            ),
        }
    )


def add_official_importer_export_scaled_ppo_downstream_rows(rows: list[dict[str, str]]) -> None:
    vae_audit = load_json(
        "res/level_c/official_importer_export_scaled_ppo_teacher_rollout_vae_training/"
        "level_c_official_importer_export_scaled_ppo_teacher_rollout_vae_training.json"
    )
    dataset_audit = load_json(
        "res/level_c/official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset/"
        "level_c_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset.json"
    )
    diffusion_audit = load_json(
        "res/level_c/official_importer_export_scaled_ppo_state_latent_diffusion_training/"
        "level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training.json"
    )
    assets = load_json(
        "res/report_assets/official_importer_export_scaled_ppo_downstream/"
        "official_importer_export_full_bundle_downstream_report_assets.json"
    )
    vae_worker = vae_audit["worker_summary"]
    dataset_worker = dataset_audit["worker_summary"]
    diffusion_worker = diffusion_audit["worker_summary"]
    rows.append(
        {
            "experiment": "level_c:official_importer_export_scaled_ppo_state_latent_dataset_and_diffusion_training",
            "paper_value": (
                "BeyondMimic trains a state-latent diffusion model from teacher/DAgger rollouts and evaluates it "
                "through guided closed-loop humanoid control. The official state-latent dataset and diffusion "
                "checkpoint are not released."
            ),
            "reproduction_value": stringify(
                {
                    "vae_status": vae_audit["status"],
                    "state_latent_status": dataset_audit["status"],
                    "diffusion_status": diffusion_audit["status"],
                    "vae_sample_count": vae_worker["dataset"]["sample_count"],
                    "vae_test_action_mse": vae_worker["evaluation"]["test"]["action_mse"],
                    "sample_count": dataset_worker["dataset"]["sample_count"],
                    "window_count": dataset_worker["dataset"]["window_count"],
                    "split_counts": dataset_worker["dataset"]["split_counts"],
                    "weighted_posterior_reconstruction_mse": dataset_worker["dataset"][
                        "weighted_posterior_reconstruction_mse"
                    ],
                    "uses_official_importer_export_usd": diffusion_audit["checks"][
                        "uses_official_importer_export_usd"
                    ],
                    "cuda_visible_devices": diffusion_audit["settings"]["cuda_visible_devices"],
                    "diffusion_epochs": diffusion_worker["training"]["epochs"],
                    "test_pred_token_mse": diffusion_worker["evaluation"]["test"]["pred_token_mse"],
                    "test_noisy_token_mse": diffusion_worker["evaluation"]["test"]["noisy_token_mse"],
                    "test_denoising_improvement_ratio": diffusion_worker["evaluation"]["test"][
                        "denoising_improvement_ratio"
                    ],
                    "report_assets": assets["assets"],
                }
            ),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "State-latent diffusion training / Fig. 5-6 prerequisite",
            "paper_source": "BeyondMimic method sections and local scaled official-importer-export downstream evidence",
            "run_id": (
                "res/level_c/official_importer_export_scaled_ppo_state_latent_diffusion_training/"
                "level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training.json"
            ),
            "reproduction_level": "official-importer-export scaled PPO local state-latent denoiser training",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This retrains the local downstream VAE/state-latent/diffusion chain from the larger iteration-999 "
                "scaled PPO teacher rollout dataset, covering 1,224,704 action-VAE samples and 1,142,784 "
                "state-latent windows. It is the strongest current local downstream training evidence on the "
                "official-importer-export asset path, but it is still a local virtual model trained from a weak local "
                "PPO teacher and local VAE. It is not the official BeyondMimic diffusion checkpoint, not TensorRT or "
                "asynchronous deployment, not closed-loop Fig. 5/Fig. 6 guidance evaluation, and not real-robot "
                "evidence."
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


def add_official_csv_loop_state_latent_guidance_rows(rows: list[dict[str, str]]) -> None:
    guidance_audit = load_json(
        "res/level_c/official_csv_loop_state_latent_guidance_eval/"
        "level_c_official_csv_loop_state_latent_guidance_eval.json"
    )
    worker = guidance_audit["worker_summary"]
    task_value = {
        task: {
            "mean_best_cost_delta": summary["mean_best_cost_delta"],
            "mean_positive_delta_fraction": summary["mean_positive_delta_fraction"],
            "all_best_costs_improve": summary["all_best_costs_improve"],
            "all_best_gradients_nonzero": summary["all_best_gradients_nonzero"],
        }
        for task, summary in worker["task_summaries"].items()
    }
    reproduction_value = {
        "status": guidance_audit["status"],
        "total_selected_windows": worker["metrics"]["total_selected_windows"],
        "selected_split_counts": worker["settings"]["selected_split_counts"],
        "row_count": worker["metrics"]["row_count"],
        "tasks": worker["settings"]["tasks"],
        "scales": worker["settings"]["scales"],
        "tasks_with_all_best_costs_improve": worker["metrics"]["tasks_with_all_best_costs_improve"],
        "tasks_with_nonzero_best_gradients": worker["metrics"]["tasks_with_nonzero_best_gradients"],
        "task_summaries": task_value,
        "gpu_metrics_summary": guidance_audit.get("gpu_metrics_summary", {}),
    }
    rows.append(
        {
            "experiment": "level_c:official_csv_loop_state_latent_guidance_eval",
            "paper_value": (
                "BeyondMimic applies guided diffusion to produce closed-loop humanoid skills in simulation and on a "
                "real Unitree G1; the official rollout logs/checkpoints and Fig. 5/Fig. 6 trajectories are not "
                "publicly available here."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Guided diffusion / Fig. 5-6 prerequisite",
            "paper_source": "BeyondMimic guided diffusion sections and local official-loop denoiser evidence",
            "run_id": (
                "res/level_c/official_csv_loop_state_latent_guidance_eval/"
                "level_c_official_csv_loop_state_latent_guidance_eval.json"
            ),
            "reproduction_level": "official csv-loop full-split offline state-latent guidance surrogate",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This evaluates task-cost guidance over all validation/test windows from the local official-loop "
                "state-latent denoiser. It strengthens the local virtual pipeline beyond denoiser training, but it is "
                "still offline cost guidance rather than an IsaacLab closed-loop rollout, TensorRT deployment, or "
                "paper Fig. 5/Fig. 6 reproduction."
            ),
        }
    )


def add_official_csv_loop_full_bundle_guidance_rows(rows: list[dict[str, str]]) -> None:
    guidance_audit = load_json(
        "res/level_c/official_csv_loop_full_bundle_state_latent_guidance_eval/"
        "level_c_official_csv_loop_full_bundle_state_latent_guidance_eval.json"
    )
    assets = load_json(
        "res/report_assets/official_csv_loop_full_bundle_guidance/"
        "official_csv_loop_full_bundle_guidance_report_assets.json"
    )
    worker = guidance_audit["worker_summary"]
    task_value = {
        task: {
            "mean_best_cost_delta": summary["mean_best_cost_delta"],
            "mean_positive_delta_fraction": summary["mean_positive_delta_fraction"],
            "all_best_costs_improve": summary["all_best_costs_improve"],
            "all_best_gradients_nonzero": summary["all_best_gradients_nonzero"],
        }
        for task, summary in worker["task_summaries"].items()
    }
    rows.append(
        {
            "experiment": "level_c:official_csv_loop_full_bundle_state_latent_guidance_eval",
            "paper_value": (
                "BeyondMimic uses guided diffusion for closed-loop humanoid skills and reports Fig. 5/Fig. 6 task "
                "results; official rollout logs/checkpoints and trajectories are not public."
            ),
            "reproduction_value": stringify(
                {
                    "status": guidance_audit["status"],
                    "total_selected_windows": worker["metrics"]["total_selected_windows"],
                    "selected_split_counts": worker["settings"]["selected_split_counts"],
                    "row_count": worker["metrics"]["row_count"],
                    "tasks": worker["settings"]["tasks"],
                    "scales": worker["settings"]["scales"],
                    "tasks_with_all_best_costs_improve": worker["metrics"]["tasks_with_all_best_costs_improve"],
                    "tasks_with_nonzero_best_gradients": worker["metrics"]["tasks_with_nonzero_best_gradients"],
                    "task_summaries": task_value,
                    "report_assets": assets["assets"],
                    "gpu_metrics_summary": guidance_audit.get("gpu_metrics_summary", {}),
                }
            ),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Guided diffusion / Fig. 5-6 prerequisite",
            "paper_source": "BeyondMimic guided diffusion sections and local full-bundle denoiser evidence",
            "run_id": (
                "res/level_c/official_csv_loop_full_bundle_state_latent_guidance_eval/"
                "level_c_official_csv_loop_full_bundle_state_latent_guidance_eval.json"
            ),
            "reproduction_level": "full-public-motion full-split offline state-latent guidance surrogate",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This evaluates task-cost guidance over all validation/test windows from the local 40-motion "
                "full-bundle state-latent denoiser and adds report-ready guidance plots. It is still offline cost "
                "guidance, not an IsaacLab closed-loop rollout, TensorRT deployment, real robot run, or paper "
                "Fig. 5/Fig. 6 reproduction."
            ),
        }
    )


def add_official_importer_export_full_bundle_guidance_rows(rows: list[dict[str, str]]) -> None:
    guidance_audit = load_json(
        "res/level_c/official_importer_export_full_bundle_state_latent_guidance_eval/"
        "level_c_official_importer_export_full_bundle_state_latent_guidance_eval.json"
    )
    worker = guidance_audit["worker_summary"]
    task_value = {
        task: {
            "mean_best_cost_delta": summary["mean_best_cost_delta"],
            "mean_positive_delta_fraction": summary["mean_positive_delta_fraction"],
            "all_best_costs_improve": summary["all_best_costs_improve"],
            "all_best_gradients_nonzero": summary["all_best_gradients_nonzero"],
        }
        for task, summary in worker["task_summaries"].items()
    }
    rows.append(
        {
            "experiment": "level_c:official_importer_export_full_bundle_state_latent_guidance_eval",
            "paper_value": (
                "BeyondMimic uses guided diffusion for closed-loop humanoid skills and reports Fig. 5/Fig. 6 task "
                "results; official rollout logs/checkpoints and trajectories are not public."
            ),
            "reproduction_value": stringify(
                {
                    "status": guidance_audit["status"],
                    "total_selected_windows": worker["metrics"]["total_selected_windows"],
                    "selected_split_counts": worker["settings"]["selected_split_counts"],
                    "row_count": worker["metrics"]["row_count"],
                    "tasks": worker["settings"]["tasks"],
                    "scales": worker["settings"]["scales"],
                    "tasks_with_all_best_costs_improve": worker["metrics"][
                        "tasks_with_all_best_costs_improve"
                    ],
                    "tasks_with_nonzero_best_gradients": worker["metrics"]["tasks_with_nonzero_best_gradients"],
                    "task_summaries": task_value,
                    "checks": guidance_audit["checks"],
                    "gpu_metrics_summary": guidance_audit.get("gpu_metrics_summary", {}),
                }
            ),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Guided diffusion / Fig. 5-6 prerequisite",
            "paper_source": "BeyondMimic guided diffusion sections and local official-importer-export denoiser evidence",
            "run_id": (
                "res/level_c/official_importer_export_full_bundle_state_latent_guidance_eval/"
                "level_c_official_importer_export_full_bundle_state_latent_guidance_eval.json"
            ),
            "reproduction_level": "official-importer-export full-split offline state-latent guidance surrogate",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This evaluates task-cost guidance over every validation/test window from the local "
                "official-importer-export 40-motion state-latent denoiser. It moves the more official G1 USDA "
                "downstream chain beyond denoiser training and confirms positive best-scale cost deltas for the "
                "four proxy tasks, but it is still offline guidance over local checkpoints, not IsaacLab closed-loop "
                "guided control, not TensorRT/asynchronous deployment, not paper Fig. 5/Fig. 6, and not real robot "
                "evidence."
            ),
        }
    )

def add_official_importer_export_scaled_ppo_guidance_rows(rows: list[dict[str, str]]) -> None:
    guidance_audit = load_json(
        "res/level_c/official_importer_export_scaled_ppo_state_latent_guidance_eval/"
        "level_c_official_importer_export_scaled_ppo_state_latent_guidance_eval.json"
    )
    assets = load_json(
        "res/report_assets/official_importer_export_scaled_ppo_guidance/"
        "official_importer_export_scaled_ppo_guidance_report_assets.json"
    )
    worker = guidance_audit["worker_summary"]
    task_value = {
        task: {
            "mean_best_cost_delta": summary["mean_best_cost_delta"],
            "mean_positive_delta_fraction": summary["mean_positive_delta_fraction"],
            "all_best_costs_improve": summary["all_best_costs_improve"],
            "all_best_gradients_nonzero": summary["all_best_gradients_nonzero"],
        }
        for task, summary in worker["task_summaries"].items()
    }
    rows.append(
        {
            "experiment": "level_c:official_importer_export_scaled_ppo_state_latent_guidance_eval",
            "paper_value": (
                "BeyondMimic uses guided diffusion for closed-loop humanoid skills and reports Fig. 5/Fig. 6 task "
                "results; official rollout logs/checkpoints and trajectories are not public."
            ),
            "reproduction_value": stringify(
                {
                    "status": guidance_audit["status"],
                    "total_selected_windows": worker["metrics"]["total_selected_windows"],
                    "selected_split_counts": worker["settings"]["selected_split_counts"],
                    "row_count": worker["metrics"]["row_count"],
                    "tasks": worker["settings"]["tasks"],
                    "scales": worker["settings"]["scales"],
                    "tasks_with_all_best_costs_improve": worker["metrics"][
                        "tasks_with_all_best_costs_improve"
                    ],
                    "tasks_with_nonzero_best_gradients": worker["metrics"]["tasks_with_nonzero_best_gradients"],
                    "task_summaries": task_value,
                    "asset_status": assets["status"],
                    "asset_paths": assets["assets"],
                    "checks": guidance_audit["checks"],
                    "gpu_metrics_summary": guidance_audit.get("gpu_metrics_summary", {}),
                }
            ),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Guided diffusion / Fig. 5-6 prerequisite",
            "paper_source": "BeyondMimic guided diffusion sections and local scaled PPO denoiser evidence",
            "run_id": (
                "res/level_c/official_importer_export_scaled_ppo_state_latent_guidance_eval/"
                "level_c_official_importer_export_scaled_ppo_state_latent_guidance_eval.json"
            ),
            "reproduction_level": "official-importer-export scaled PPO full-split offline state-latent guidance surrogate",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This evaluates task-cost guidance over every validation/test window from the local scaled PPO "
                "official-importer-export state-latent denoiser: 228557 windows, 48 task/split/scale rows, and "
                "positive best-scale cost deltas for all four proxy tasks. It updates the offline guidance evidence "
                "to the larger iteration-999 teacher-rollout downstream chain, but it remains offline guidance over "
                "local checkpoints, not IsaacLab closed-loop guided control, not TensorRT/asynchronous deployment, "
                "not paper Fig. 5/Fig. 6, and not real robot evidence."
            ),
        }
    )


def add_official_csv_loop_guidance_vae_action_decode_rows(rows: list[dict[str, str]]) -> None:
    decode_audit = load_json(
        "res/level_c/official_csv_loop_guidance_vae_action_decode_eval/"
        "level_c_official_csv_loop_guidance_vae_action_decode_eval.json"
    )
    assets = load_json(
        "res/report_assets/official_csv_loop_guidance_vae_action_decode/"
        "official_csv_loop_guidance_vae_action_decode_assets.json"
    )
    worker = decode_audit["worker_summary"]
    task_value = {
        task: {
            "mean_guided_base_action_l2": summary["mean_guided_base_action_l2"],
            "mean_guided_minus_base_teacher_mse": summary["mean_guided_minus_base_teacher_mse"],
            "all_actions_finite": summary["all_actions_finite"],
        }
        for task, summary in worker["task_summaries"].items()
    }
    reproduction_value = {
        "status": decode_audit["status"],
        "total_windows": worker["metrics"]["total_windows"],
        "total_action_steps_per_task": worker["metrics"]["total_action_steps_per_task"],
        "tasks_with_finite_actions": worker["metrics"]["tasks_with_finite_actions"],
        "decoded_action_dim_29": decode_audit["checks"]["decoded_action_dim_29"],
        "task_summaries": task_value,
        "report_assets": assets["assets"],
        "gpu_metrics_summary": decode_audit.get("gpu_metrics_summary", {}),
    }
    rows.append(
        {
            "experiment": "level_c:official_csv_loop_guidance_vae_action_decode_eval",
            "paper_value": (
                "BeyondMimic decodes generated latent actions through the learned action model during control. "
                "The official closed-loop VAE/diffusion rollout logs and deployment artifacts are not public here."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "VAE-decoded guided action prerequisite",
            "paper_source": "BeyondMimic guided diffusion and latent action sections",
            "run_id": (
                "res/level_c/official_csv_loop_guidance_vae_action_decode_eval/"
                "level_c_official_csv_loop_guidance_vae_action_decode_eval.json"
            ),
            "reproduction_level": "official csv-loop offline guided latent to VAE action decode",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This decodes guided local state-latent trajectories into 29D VAE actions over all validation/test "
                "windows and creates report-ready plots. It is an important bridge toward closed-loop rollout, but "
                "it still does not execute the actions in IsaacLab and does not reproduce Fig. 5/Fig. 6."
            ),
        }
    )


def add_official_csv_loop_guided_action_rollout_probe_rows(rows: list[dict[str, str]]) -> None:
    probe = load_json(
        "res/level_c/official_csv_loop_guided_action_rollout_probe/"
        "tracking_g1_official_csv_loop_guided_action_rollout_probe.json"
    )
    assets = load_json(
        "res/level_c/official_csv_loop_guided_action_rollout_probe/"
        "official_csv_loop_guided_action_rollout_probe_assets.json"
    )
    reproduction_value = {
        "status": probe["status"],
        "rollout_steps": probe["metrics"]["rollout_steps"],
        "task": probe["config"]["task"],
        "sample_index": probe["config"]["sample_index"],
        "selected_physical_gpu": probe["config"]["selected_physical_gpu"],
        "base_guided_max_abs_action_delta": probe["metrics"]["base_guided_max_abs_action_delta"],
        "base_guided_l2_mean": probe["metrics"]["base_guided_l2_mean"],
        "base_teacher_mse": probe["metrics"]["base_teacher_mse"],
        "guided_teacher_mse": probe["metrics"]["guided_teacher_mse"],
        "variant_metrics": probe["metrics"]["variant_metrics"],
        "assets": assets["assets"],
    }
    rows.append(
        {
            "experiment": "level_c:official_csv_loop_guided_action_rollout_probe",
            "paper_value": (
                "BeyondMimic executes generated latent actions in closed-loop control for guided tasks. The paper "
                "does not release official closed-loop guided rollout logs/checkpoints for this local environment."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Guided diffusion closed-loop rollout bridge",
            "paper_source": "BeyondMimic guided diffusion / Fig. 5-6 task sections",
            "run_id": (
                "res/level_c/official_csv_loop_guided_action_rollout_probe/"
                "tracking_g1_official_csv_loop_guided_action_rollout_probe.json"
            ),
            "reproduction_level": "local virtual decoded-action IsaacLab rollout probe",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This executes one 21-step decoded local VAE action sample for base/guided/teacher variants inside "
                "the resource-adjusted official-csv-loop Tracking-Flat-G1-v0 task. It validates the action-to-sim "
                "bridge but is not receding-horizon diffusion guidance, not paper task success evaluation, not "
                "Fig. 5/Fig. 6 reproduction, and not real-robot evidence. The sampled base and guided actions are "
                "numerically identical in this probe, so it is also a negative result for behavior change."
            ),
        }
    )


def add_official_csv_loop_action_guidance_rollout_rows(rows: list[dict[str, str]]) -> None:
    rollout = load_json(
        "res/level_c/official_csv_loop_action_guidance_rollout_eval/"
        "level_c_official_csv_loop_action_guidance_rollout_eval.json"
    )
    asset = load_json(
        "res/visualization/official_csv_loop_action_guidance_rollout/"
        "official_csv_loop_action_guidance_rollout_asset.json"
    )
    reproduction_value = {
        "status": rollout["status"],
        "rollout_steps": rollout["metrics"]["rollout_steps"],
        "selected_physical_gpu": rollout["config"]["selected_physical_gpu"],
        "guidance": rollout["metrics"]["guidance"],
        "variant_metrics": rollout["metrics"]["variant_metrics"],
        "assets": asset["assets"],
        "claim_level": asset["claim_level"],
    }
    rows.append(
        {
            "experiment": "level_c:official_csv_loop_action_guidance_rollout_eval",
            "paper_value": (
                "BeyondMimic evaluates guided latent diffusion in closed-loop tasks such as joystick, waypoint, "
                "inpainting, and obstacle avoidance. Official Fig. 5/Fig. 6 rollout logs and checkpoints are not "
                "public in this local artifact set."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Guided diffusion closed-loop rollout bridge",
            "paper_source": "BeyondMimic guided diffusion / Fig. 5-6 task sections",
            "run_id": (
                "res/level_c/official_csv_loop_action_guidance_rollout_eval/"
                "level_c_official_csv_loop_action_guidance_rollout_eval.json"
            ),
            "reproduction_level": "local virtual teacher-consistency action-guidance rollout",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This executes 299-step teacher, VAE-base, and action-guided variants in the local "
                "resource-adjusted official-csv-loop Tracking-Flat-G1-v0 task and produces MP4/keyframe/metric "
                "assets. The guidance is an action-space teacher-consistency bridge, not the paper's "
                "receding-horizon latent diffusion controller, not an official BeyondMimic checkpoint, not "
                "Fig. 5/Fig. 6 paper-level evaluation, and not real-robot evidence."
            ),
        }
    )


def add_official_csv_loop_receding_latent_guidance_rollout_rows(rows: list[dict[str, str]]) -> None:
    rollout = load_json(
        "res/level_c/official_csv_loop_receding_latent_guidance_rollout_eval/"
        "level_c_official_csv_loop_receding_latent_guidance_rollout_eval.json"
    )
    asset = load_json(
        "res/visualization/official_csv_loop_receding_latent_guidance_rollout/"
        "official_csv_loop_receding_latent_guidance_rollout_asset.json"
    )
    reproduction_value = {
        "status": rollout["status"],
        "rollout_steps": rollout["metrics"]["rollout_steps"],
        "selected_physical_gpu": rollout["config"]["selected_physical_gpu"],
        "guidance": rollout["metrics"]["guidance"],
        "variant_metrics": rollout["metrics"]["variant_metrics"],
        "assets": asset["assets"],
        "claim_level": asset["claim_level"],
    }
    rows.append(
        {
            "experiment": "level_c:official_csv_loop_receding_latent_guidance_rollout_eval",
            "paper_value": (
                "BeyondMimic evaluates receding-horizon guided latent diffusion in closed-loop humanoid tasks "
                "such as joystick, waypoint, inpainting, and obstacle avoidance. Official Fig. 5/Fig. 6 rollout "
                "logs and checkpoints are not public in this local artifact set."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Guided diffusion closed-loop rollout bridge",
            "paper_source": "BeyondMimic guided diffusion / Fig. 5-6 task sections",
            "run_id": (
                "res/level_c/official_csv_loop_receding_latent_guidance_rollout_eval/"
                "level_c_official_csv_loop_receding_latent_guidance_rollout_eval.json"
            ),
            "reproduction_level": "local virtual receding-horizon latent-guidance rollout",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This executes 299-step teacher, VAE-base, denoised-latent, and receding-horizon guided-latent "
                "variants in the local resource-adjusted official-csv-loop Tracking-Flat-G1-v0 task and produces "
                "MP4/keyframe/metric assets. The guided variant recomputes a local state-latent horizon every "
                "control step, applies the local denoiser and one composed-cost guidance step, and decodes the "
                "current latent through the local VAE. It is the strongest current local closed-loop bridge toward "
                "paper guidance, but it still uses local resource-adjusted PPO/VAE/denoiser checkpoints and "
                "enriched USD, not the official BeyondMimic checkpoint, not paper Fig. 5/Fig. 6 task setup, not "
                "TensorRT/asynchronous deployment, and not real-robot evidence."
            ),
        }
    )


def add_official_csv_loop_task_conditioned_latent_guidance_rollout_rows(rows: list[dict[str, str]]) -> None:
    rollout = load_json(
        "res/level_c/official_csv_loop_task_conditioned_latent_guidance_rollout_eval/"
        "level_c_official_csv_loop_task_conditioned_latent_guidance_rollout_eval.json"
    )
    reproduction_value = {
        "status": rollout["status"],
        "tasks": rollout["tasks"],
        "rows": rollout["rows"],
        "claim_level": rollout["interpretation"]["paper_level_status"],
    }
    rows.append(
        {
            "experiment": "level_c:official_csv_loop_task_conditioned_latent_guidance_rollout_eval",
            "paper_value": (
                "BeyondMimic reports guided latent diffusion for joystick, waypoint, obstacle-avoidance, "
                "inpainting, and composed-objective humanoid tasks, with qualitative Fig. 5/Fig. 6 rollouts. "
                "The paper-level checkpoints, rollout logs, and task success-rate traces are not public here."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Guided diffusion task-conditioned closed-loop rollout bridge",
            "paper_source": "BeyondMimic guided diffusion / Fig. 5-6 task sections",
            "run_id": (
                "res/level_c/official_csv_loop_task_conditioned_latent_guidance_rollout_eval/"
                "level_c_official_csv_loop_task_conditioned_latent_guidance_rollout_eval.json"
            ),
            "reproduction_level": "local virtual task-conditioned receding-horizon latent-guidance rollout",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This executes four local IsaacLab closed-loop task-conditioned proxy rollouts: joystick, waypoint, "
                "obstacle_avoidance, and composed. Each task runs 299 steps with teacher/VAE/denoised/guided variants "
                "and records MP4, keyframes, plots, CSV metrics, reward, target-body error, action MSE, and guidance "
                "cost deltas. It advances from a single composed-cost bridge toward paper-style guided tasks, but it "
                "uses local resource-adjusted PPO/VAE/denoiser checkpoints, an enriched USD scaffold, and proxy "
                "costs rather than the official BeyondMimic checkpoint, paper Fig. 5/Fig. 6 task setup, TensorRT/"
                "asynchronous deployment, or real robot evidence."
            ),
        }
    )


def add_official_csv_loop_task_conditioned_latent_guidance_multiseed_rows(rows: list[dict[str, str]]) -> None:
    rollout = load_json(
        "res/level_c/official_csv_loop_task_conditioned_latent_guidance_multiseed_eval/"
        "official_csv_loop_task_conditioned_latent_guidance_multiseed_eval.json"
    )
    assets = load_json(
        "res/report_assets/official_csv_loop_task_conditioned_guidance_multiseed/"
        "official_csv_loop_task_conditioned_guidance_multiseed_assets.json"
    )
    reproduction_value = {
        "status": rollout["status"],
        "tasks": rollout["tasks"],
        "seed_groups": rollout["seed_groups"],
        "metrics": rollout["metrics"],
        "aggregate": rollout["aggregate"],
        "asset_paths": assets["assets"],
        "claim_level": rollout["interpretation"]["paper_level_status"],
    }
    rows.append(
        {
            "experiment": "level_c:official_csv_loop_task_conditioned_latent_guidance_multiseed_eval",
            "paper_value": (
                "BeyondMimic reports guided latent diffusion on joystick, waypoint, obstacle-avoidance, "
                "inpainting, and composed-objective humanoid tasks with qualitative Fig. 5/Fig. 6 rollouts. "
                "The paper-level checkpoints, rollout logs, task definitions, and success-rate traces are not "
                "public in this local artifact set."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Guided diffusion task-conditioned multi-seed closed-loop rollout bridge",
            "paper_source": "BeyondMimic guided diffusion / Fig. 5-6 task sections",
            "run_id": (
                "res/level_c/official_csv_loop_task_conditioned_latent_guidance_multiseed_eval/"
                "official_csv_loop_task_conditioned_latent_guidance_multiseed_eval.json"
            ),
            "reproduction_level": "local virtual task-conditioned receding-horizon latent-guidance multiseed rollout",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This aggregates three local seed groups over joystick, waypoint, obstacle_avoidance, and composed "
                "proxy tasks, for 12 closed-loop IsaacLab rollouts and 14352 variant control steps. Each new task "
                "run records JSON/TSV metrics plus MP4/keyframes/plots under res/visualization, and the report "
                "assets summarize guided reward, target-body error, done counts, action MSE, and guidance cost "
                "delta across seeds. It improves robustness over the prior single-seed task-conditioned bridge, "
                "but it still uses local resource-adjusted PPO/VAE/denoiser checkpoints, local proxy costs, and "
                "an enriched USD scaffold rather than official BeyondMimic Fig. 5/Fig. 6 evaluation, official "
                "checkpoints, TensorRT/asynchronous deployment, or real-robot evidence."
            ),
        }
    )


def add_guided_vs_unguided_closed_loop_matrix_rows(rows: list[dict[str, str]]) -> None:
    matrix = load_json(
        "res/report_assets/guided_vs_unguided_closed_loop_matrix/"
        "guided_vs_unguided_closed_loop_matrix.json"
    )
    reproduction_value = {
        "status": matrix["status"],
        "claim_level": matrix["claim_level"],
        "metrics": matrix["metrics"],
        "aggregate": matrix["aggregate"],
        "outputs": matrix["outputs"],
    }
    rows.append(
        {
            "experiment": "report_assets:guided_vs_unguided_closed_loop_matrix",
            "paper_value": (
                "BeyondMimic qualitatively and quantitatively evaluates guided latent diffusion against task "
                "objectives in closed-loop humanoid settings, but the public/local artifact set does not include "
                "official Fig. 5/Fig. 6 rollouts, official task success logs, or official VAE/diffusion checkpoints."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Guided diffusion closed-loop evidence summary",
            "paper_source": "BeyondMimic guided diffusion / Fig. 5-6 task sections",
            "run_id": (
                "res/report_assets/guided_vs_unguided_closed_loop_matrix/"
                "guided_vs_unguided_closed_loop_matrix.json"
            ),
            "reproduction_level": "local virtual guided-vs-unguided closed-loop report matrix",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This is a report-facing matrix over existing local action-guidance, receding-latent, "
                "task-conditioned single-seed, and task-conditioned multi-seed closed-loop rollouts. It records "
                "guided-vs-baseline reward, tracking-error, done-count, action-MSE, guidance-cost, video-path, "
                "claim-level, and comparison-type fields. It is useful reading-report evidence, but not a new "
                "paper metric and not official Fig. 5/Fig. 6 reproduction."
            ),
        }
    )


def add_official_csv_loop_full_bundle_receding_latent_guidance_rollout_rows(rows: list[dict[str, str]]) -> None:
    rollout = load_json(
        "res/level_c/official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval/"
        "level_c_official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval.json"
    )
    reproduction_value = {
        "status": rollout["status"],
        "bundle": rollout["bundle"],
        "config": rollout["config"],
        "guidance": rollout["metrics"]["guidance"],
        "variant_metrics": rollout["metrics"]["variant_metrics"],
        "assets": rollout["outputs"]["assets"],
        "claim_level": rollout["interpretation"]["paper_level_status"],
    }
    rows.append(
        {
            "experiment": "level_c:official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval",
            "paper_value": (
                "BeyondMimic reports guided latent diffusion rollouts over versatile humanoid control tasks. "
                "Official Fig. 5/Fig. 6 videos, task definitions, success traces, and VAE/diffusion checkpoints "
                "are not public in this artifact set."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Guided diffusion full-bundle closed-loop rollout bridge",
            "paper_source": "BeyondMimic guided diffusion / Fig. 5-6 task sections",
            "run_id": (
                "res/level_c/official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval/"
                "level_c_official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval.json"
            ),
            "reproduction_level": "local virtual full-bundle receding-horizon latent-guidance rollout",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This executes teacher, VAE-base, denoised-latent, and receding-horizon guided-latent variants in "
                "the local IsaacLab task using the 40-motion official-csv-loop public bundle and matching local "
                "full-bundle PPO/VAE/denoiser artifacts. It produces MP4, keyframes, metric plots, CSV, GPU metrics, "
                "and JSON evidence. It is stronger than the single-motion bridge because it uses the full public "
                "motion bundle, but it still depends on enriched USD and local checkpoints, not official "
                "BeyondMimic Fig. 5/Fig. 6 rollouts, TensorRT deployment, or real-robot validation."
            ),
        }
    )


def add_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_rows(
    rows: list[dict[str, str]],
) -> None:
    rollout = load_json(
        "res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval/"
        "level_c_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval.json"
    )
    reproduction_value = {
        "status": rollout["status"],
        "bundle": rollout["bundle"],
        "tasks": rollout["tasks"],
        "rows": [
            {
                "task": row["task"],
                "rollout_steps": row["rollout_steps"],
                "guided_reward_mean": row["guided_reward_mean"],
                "guided_target_body_error_mean": row["guided_target_body_error_mean"],
                "guidance_cost_delta_mean": row["guidance_cost_delta_mean"],
                "mp4": row["mp4"],
            }
            for row in rollout["rows"]
        ],
        "checks": rollout["checks"],
        "claim_level": rollout["interpretation"]["paper_level_status"],
    }
    rows.append(
        {
            "experiment": "level_c:official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval",
            "paper_value": (
                "BeyondMimic evaluates guided latent diffusion on joystick, waypoint, obstacle/inpainting, and "
                "composed humanoid control tasks. Official Fig. 5/Fig. 6 task rollouts, success logs, and "
                "VAE/diffusion checkpoints are not public in this local artifact set."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Guided diffusion full-bundle task-conditioned closed-loop rollout bridge",
            "paper_source": "BeyondMimic guided diffusion / Fig. 5-6 task sections",
            "run_id": (
                "res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval/"
                "level_c_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval.json"
            ),
            "reproduction_level": "local virtual full-bundle task-conditioned receding-horizon latent-guidance rollout",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This executes four closed-loop local task-conditioned receding-latent guidance rollouts over the "
                "40-motion public official-csv-loop bundle, using matching local full-bundle PPO/VAE/denoiser "
                "artifacts. It adds joystick, waypoint, obstacle_avoidance, and composed proxy rollouts plus "
                "MP4/keyframe/plot/CSV evidence to the prior full-bundle composed rollout. It is still "
                "qualitative-only local virtual evidence: the robot asset path is resource-adjusted, tasks are "
                "local proxy objectives, checkpoints are local rather than official BeyondMimic checkpoints, and "
                "no paper Fig. 5/Fig. 6 success metric, TensorRT deployment, or real-robot result is claimed."
            ),
        }
    )


def add_official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_rows(
    rows: list[dict[str, str]],
) -> None:
    rollout = load_json(
        "res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval/"
        "official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json"
    )
    assets = load_json(
        "res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_multiseed/"
        "official_csv_loop_full_bundle_task_conditioned_guidance_multiseed_assets.json"
    )
    reproduction_value = {
        "status": rollout["status"],
        "bundle": rollout["bundle"],
        "tasks": rollout["tasks"],
        "seed_groups": rollout["seed_groups"],
        "metrics": rollout["metrics"],
        "aggregate": rollout["aggregate"],
        "asset_paths": assets["assets"],
        "claim_level": rollout["interpretation"]["paper_level_status"],
    }
    rows.append(
        {
            "experiment": "level_c:official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval",
            "paper_value": (
                "BeyondMimic evaluates guided latent diffusion on joystick, waypoint, obstacle/inpainting, and "
                "composed humanoid tasks. Official Fig. 5/Fig. 6 task rollouts, success logs, and VAE/diffusion "
                "checkpoints are not public in this local artifact set."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Guided diffusion full-bundle task-conditioned multi-seed closed-loop bridge",
            "paper_source": "BeyondMimic guided diffusion / Fig. 5-6 task sections",
            "run_id": (
                "res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval/"
                "official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json"
            ),
            "reproduction_level": (
                "local virtual full-bundle task-conditioned receding-horizon latent-guidance multiseed rollout"
            ),
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This aggregates three seed groups over joystick, waypoint, obstacle_avoidance, and composed proxy "
                "tasks for 12 local closed-loop IsaacLab rollouts over the 40-motion public official-csv-loop "
                "bundle. It adds robustness evidence beyond the single full-bundle task-conditioned run and records "
                "JSON/CSV/plot/keyframe/video paths, but remains qualitative-only local virtual evidence: local "
                "checkpoints, local proxy costs, enriched USD, no official BeyondMimic checkpoint, no paper Fig. 5/"
                "Fig. 6 success metric, no TensorRT deployment claim, and no real-robot result."
            ),
        }
    )


def add_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_rows(
    rows: list[dict[str, str]],
) -> None:
    rollout = load_json(
        "res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval/"
        "level_c_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval.json"
    )
    reproduction_value = {
        "status": rollout["status"],
        "bundle": rollout["bundle"],
        "tasks": rollout["tasks"],
        "rows": [
            {
                "task": row["task"],
                "rollout_steps": row["rollout_steps"],
                "guided_reward_mean": row["guided_reward_mean"],
                "guided_target_body_error_mean": row["guided_target_body_error_mean"],
                "guidance_cost_delta_mean": row["guidance_cost_delta_mean"],
                "guided_teacher_action_mse_mean": row["guided_teacher_action_mse_mean"],
                "mp4": row["mp4"],
            }
            for row in rollout["rows"]
        ],
        "checks": rollout["checks"],
        "claim_level": rollout["interpretation"]["paper_level_status"],
    }
    rows.append(
        {
            "experiment": (
                "level_c:official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval"
            ),
            "paper_value": (
                "BeyondMimic evaluates guided latent diffusion on joystick, waypoint, obstacle/inpainting, and "
                "composed humanoid control tasks. Official Fig. 5/Fig. 6 task rollouts, success logs, TensorRT "
                "deployment traces, and VAE/diffusion checkpoints are not public in this local artifact set."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Guided diffusion official-importer-export task-conditioned closed-loop bridge",
            "paper_source": "BeyondMimic guided diffusion / Fig. 5-6 task sections",
            "run_id": (
                "res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval/"
                "level_c_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval.json"
            ),
            "reproduction_level": (
                "local virtual official-importer-export task-conditioned receding-horizon latent-guidance rollout"
            ),
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This executes four closed-loop local task-conditioned receding-latent guidance rollouts on the "
                "official-importer-export G1 USDA path over the public 40-motion bundle. It compares teacher, "
                "VAE-base, denoised-latent, and guided-latent variants for joystick, waypoint, obstacle_avoidance, "
                "and composed proxy tasks, and records MP4/keyframes/plots/CSV/JSON evidence. It is the strongest "
                "current local virtual guidance bridge on the recovered official-importer-export asset path, but "
                "it still uses local PPO/VAE/denoiser checkpoints and proxy costs, not official BeyondMimic "
                "checkpoints, not the paper Fig. 5/Fig. 6 success protocol, not TensorRT deployment, and not "
                "real-robot evidence."
            ),
        }
    )


def add_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_rows(
    rows: list[dict[str, str]],
) -> None:
    rollout = load_json(
        "res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval/"
        "level_c_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval.json"
    )
    assets = load_json(
        "res/report_assets/official_importer_export_scaled_ppo_task_conditioned_guidance_summary/"
        "official_csv_loop_task_conditioned_guidance_summary_assets.json"
    )
    reproduction_value = {
        "status": rollout["status"],
        "bundle": rollout["bundle"],
        "tasks": rollout["tasks"],
        "input_statuses": {
            "training": rollout["inputs"]["training_run_status"],
            "checkpoint_eval": rollout["inputs"]["checkpoint_eval_status"],
            "vae": rollout["inputs"]["vae_training_status"],
            "diffusion": rollout["inputs"]["diffusion_status"],
            "offline_guidance": rollout["inputs"]["guidance_status"],
        },
        "rows": [
            {
                "task": row["task"],
                "rollout_steps": row["rollout_steps"],
                "guided_reward_mean": row["guided_reward_mean"],
                "guided_target_body_error_mean": row["guided_target_body_error_mean"],
                "guidance_cost_delta_mean": row["guidance_cost_delta_mean"],
                "guided_teacher_action_mse_mean": row["guided_teacher_action_mse_mean"],
                "mp4": row["mp4"],
            }
            for row in rollout["rows"]
        ],
        "checks": rollout["checks"],
        "report_assets": assets["assets"],
        "claim_level": rollout["interpretation"]["paper_level_status"],
    }
    rows.append(
        {
            "experiment": (
                "level_c:official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval"
            ),
            "paper_value": (
                "BeyondMimic evaluates guided latent diffusion on joystick, waypoint, obstacle/inpainting, and "
                "composed humanoid-control tasks. The public/local artifact set still lacks official BeyondMimic "
                "VAE/diffusion checkpoints, Fig. 5/Fig. 6 rollout logs, TensorRT traces, and real-robot data."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Guided diffusion scaled-PPO official-importer-export closed-loop bridge",
            "paper_source": "BeyondMimic guided diffusion / Fig. 5-6 task sections",
            "run_id": (
                "res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval/"
                "level_c_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval.json"
            ),
            "reproduction_level": (
                "local virtual official-importer-export scaled-PPO task-conditioned receding-horizon "
                "latent-guidance rollout"
            ),
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This reruns the local official-importer-export task-conditioned closed-loop bridge with the newer "
                "iteration-999 scaled PPO teacher data, scaled VAE, scaled denoiser, and scaled offline guidance "
                "summary. It records four 299-step IsaacLab proxy rollouts plus report CSV/PNG assets. It is better "
                "local virtual evidence for the paper's guided-control mechanism than offline guidance alone, but "
                "it remains qualitative-only: local proxy costs, local checkpoints, no official BeyondMimic "
                "diffusion/VAE checkpoint, no paper Fig. 5/Fig. 6 success/failure protocol, no TensorRT deployment, "
                "and no real robot evidence."
            ),
        }
    )


def add_official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_rows(
    rows: list[dict[str, str]],
) -> None:
    rollout = load_json(
        "res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval/"
        "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json"
    )
    assets = load_json(
        "res/report_assets/official_importer_export_full_bundle_task_conditioned_guidance_multiseed/"
        "official_importer_export_full_bundle_task_conditioned_guidance_multiseed_assets.json"
    )
    reproduction_value = {
        "status": rollout["status"],
        "bundle": rollout["bundle"],
        "tasks": rollout["tasks"],
        "seed_groups": rollout["seed_groups"],
        "metrics": rollout["metrics"],
        "aggregate": rollout["aggregate"],
        "asset_paths": assets["assets"],
        "checks": rollout["checks"],
        "claim_level": rollout["interpretation"]["paper_level_status"],
    }
    metrics = rollout["metrics"]
    rows.append(
        {
            "experiment": (
                "level_c:official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval"
            ),
            "paper_value": (
                "BeyondMimic evaluates guided latent diffusion on joystick, waypoint, obstacle/inpainting, and "
                "composed humanoid tasks. Official Fig. 5/Fig. 6 task rollouts, success logs, TensorRT traces, and "
                "VAE/diffusion checkpoints are not public in this local artifact set."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": (
                "Guided diffusion official-importer-export task-conditioned multi-seed closed-loop bridge"
            ),
            "paper_source": "BeyondMimic guided diffusion / Fig. 5-6 task sections",
            "run_id": (
                "res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval/"
                "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json"
            ),
            "reproduction_level": (
                "local virtual official-importer-export task-conditioned receding-horizon latent-guidance "
                "multiseed rollout"
            ),
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                f"This aggregates {metrics['seed_group_count']} seed groups over joystick, waypoint, "
                f"obstacle_avoidance, and composed proxy tasks for {metrics['row_count']} local closed-loop "
                "IsaacLab rollouts on the official-importer-export G1 USDA path over the public 40-motion bundle, "
                f"covering {metrics['total_rollout_variant_steps']} recorded rollout-variant steps. It records "
                "JSON/CSV/plot/keyframe/video paths and improves robustness over the prior single-seed "
                "importer-export task-conditioned bridge. It remains qualitative-only local virtual evidence: local "
                "PPO/VAE/denoiser checkpoints, local proxy costs, no official BeyondMimic checkpoint, no paper Fig. "
                "5/Fig. 6 success metric, no TensorRT deployment claim, and no real-robot result."
            ),
        }
    )


def add_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_rows(
    rows: list[dict[str, str]],
) -> None:
    rollout = load_json(
        "res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval/"
        "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval.json"
    )
    assets = load_json(
        "res/report_assets/official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed/"
        "official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed_assets.json"
    )
    reproduction_value = {
        "status": rollout["status"],
        "bundle": rollout["bundle"],
        "tasks": rollout["tasks"],
        "seed_groups": rollout["seed_groups"],
        "inputs": rollout["inputs"],
        "input_checks": {
            "training": rollout["checks"]["uses_scaled_ppo_training_run"],
            "checkpoint_eval": rollout["checks"]["uses_scaled_ppo_checkpoint_eval"],
            "vae": rollout["checks"]["uses_scaled_ppo_vae"],
            "diffusion": rollout["checks"]["uses_scaled_ppo_denoiser"],
            "offline_guidance": rollout["checks"]["uses_scaled_ppo_offline_guidance"],
        },
        "metrics": rollout["metrics"],
        "aggregate": rollout["aggregate"],
        "asset_paths": assets["assets"],
        "checks": rollout["checks"],
        "claim_level": rollout["interpretation"]["paper_level_status"],
    }
    metrics = rollout["metrics"]
    rows.append(
        {
            "experiment": (
                "level_c:official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval"
            ),
            "paper_value": (
                "BeyondMimic evaluates guided latent diffusion on joystick, waypoint, obstacle/inpainting, and "
                "composed humanoid-control tasks. The public/local artifact set still lacks official BeyondMimic "
                "VAE/diffusion checkpoints, Fig. 5/Fig. 6 task logs, TensorRT traces, and real-robot data."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": (
                "Guided diffusion scaled-PPO official-importer-export task-conditioned multi-seed closed-loop bridge"
            ),
            "paper_source": "BeyondMimic guided diffusion / Fig. 5-6 task sections",
            "run_id": (
                "res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval/"
                "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval.json"
            ),
            "reproduction_level": (
                "local virtual official-importer-export scaled-PPO task-conditioned receding-horizon "
                "latent-guidance multiseed rollout"
            ),
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                f"This aggregates {metrics['seed_group_count']} seed groups over joystick, waypoint, "
                f"obstacle_avoidance, and composed proxy tasks for {metrics['row_count']} local closed-loop "
                "IsaacLab rollouts on the official-importer-export G1 USDA path over the public 40-motion bundle, "
                f"covering {metrics['total_rollout_variant_steps']} recorded rollout-variant steps. It extends the "
                "single-seed scaled-PPO bridge with robustness evidence and report-ready aggregate CSV/PNG assets. "
                "It remains qualitative-only local virtual evidence: local scaled PPO/VAE/denoiser checkpoints, "
                "local proxy costs, no official BeyondMimic checkpoint, no paper Fig. 5/Fig. 6 success metric, no "
                "TensorRT deployment claim, and no real-robot result."
            ),
        }
    )


def add_official_importer_export_full_bundle_task_conditioned_guidance_success_boundary_rows(
    rows: list[dict[str, str]],
) -> None:
    boundary = load_json(
        "res/report_assets/official_importer_export_full_bundle_task_conditioned_guidance_success_boundary/"
        "local_proxy_success_boundary.json"
    )
    reproduction_value = {
        "status": boundary["status"],
        "metrics": boundary["metrics"],
        "aggregate": boundary["aggregate"],
        "asset_paths": boundary["assets"],
        "checks": boundary["checks"],
        "claim_level": boundary["interpretation"]["claim_level"],
    }
    metrics = boundary["metrics"]
    rows.append(
        {
            "experiment": (
                "report_assets:official_importer_export_full_bundle_task_conditioned_guidance_success_boundary"
            ),
            "paper_value": (
                "BeyondMimic reports qualitative and task-success evidence for guided diffusion tasks in Fig. 5/"
                "Fig. 6, but the official task rollout logs, checkpoints, TensorRT traces, and exact success/"
                "failure protocol are not public in this local artifact set."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": (
                "Guided diffusion official-importer-export local proxy success-boundary summary"
            ),
            "paper_source": "BeyondMimic guided diffusion / Fig. 5-6 task sections",
            "run_id": (
                "res/report_assets/official_importer_export_full_bundle_task_conditioned_guidance_success_boundary/"
                "local_proxy_success_boundary.json"
            ),
            "reproduction_level": (
                "local virtual official-importer-export task-conditioned guidance success-boundary proxy"
            ),
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                f"This converts {metrics['row_count']} importer-export task-conditioned guidance rollouts across "
                f"{metrics['seed_group_count']} seed groups into an explicit local proxy boundary table: 299-step "
                "completion, positive guidance signal, action change, reward improvement over the denoised baseline, "
                "tracking-error non-worsening, and a conservative local proxy pass flag. It is useful for the "
                "English reading report because it makes the local guided-control evidence easier to interpret, but "
                "it is not an official BeyondMimic success rate, not the paper Fig. 5/Fig. 6 protocol, not TensorRT "
                "deployment, and not real-robot evidence."
            ),
        }
    )


def add_official_importer_export_fig5_fig6_proxy_protocol_matrix_rows(rows: list[dict[str, str]]) -> None:
    matrix = load_json(
        "res/report_assets/official_importer_export_fig5_fig6_proxy_protocol_matrix/"
        "fig5_fig6_proxy_protocol_matrix.json"
    )
    reproduction_value = {
        "status": matrix["status"],
        "metrics": matrix["metrics"],
        "checks": matrix["checks"],
        "assets": matrix["assets"],
        "claim_level": matrix["interpretation"]["claim_level"],
        "no_hardware_next_steps": matrix["interpretation"]["no_hardware_next_steps"],
    }
    rows.append(
        {
            "experiment": "report_assets:official_importer_export_fig5_fig6_proxy_protocol_matrix",
            "paper_value": (
                "BeyondMimic Fig. 5/Fig. 6 cover joystick denoising and teleoperation, latent transition "
                "visualization, motion inpainting with keyframes, and waypoint plus SDF obstacle navigation. The "
                "official task protocol, success/fall/collision logs, TensorRT traces, and real/mocap evidence are "
                "not public in this artifact set."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Figure 5 / Figure 6 proxy protocol matrix",
            "paper_source": "reproduction/paper/source/root.tex:223-243; root.tex:549-593",
            "run_id": (
                "res/report_assets/official_importer_export_fig5_fig6_proxy_protocol_matrix/"
                "fig5_fig6_proxy_protocol_matrix.json"
            ),
            "reproduction_level": "local virtual official-importer-export Fig.5/Fig.6 proxy protocol matrix",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This report asset maps current local importer-export guidance evidence onto the six paper Fig. 5/"
                "Fig. 6 panels. It records which panels have local closed-loop proxy support, which have only "
                "offline/debug evidence, and which next virtual validations remain possible without real hardware. "
                "It is deliberately qualitative-only: no official BeyondMimic checkpoint, exact paper task protocol, "
                "TensorRT deployment, mocap/real-world context, or real robot result is claimed."
            ),
        }
    )


def add_official_importer_export_fig5_fig6_task_protocol_proxy_rows(rows: list[dict[str, str]]) -> None:
    proxy = load_json(
        "res/report_assets/official_importer_export_fig5_fig6_task_protocol_proxy/"
        "fig5_fig6_task_protocol_proxy.json"
    )
    reproduction_value = {
        "status": proxy["status"],
        "metrics": proxy["metrics"],
        "aggregate": proxy["aggregate"],
        "checks": proxy["checks"],
        "thresholds": proxy["thresholds"],
        "assets": proxy["assets"],
        "claim_level": proxy["interpretation"]["claim_level"],
    }
    metrics = proxy["metrics"]
    rows.append(
        {
            "experiment": "report_assets:official_importer_export_fig5_fig6_task_protocol_proxy",
            "paper_value": (
                "BeyondMimic Fig. 5/Fig. 6 report guided diffusion task behavior with paper-specific task "
                "protocols, success/fall/collision criteria, and real-world or deployment evidence. The public "
                "artifact set used here does not include the official closed-loop task logs, exact thresholds, "
                "VAE/diffusion checkpoints, TensorRT traces, or real-robot records."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Figure 5 / Figure 6 local task-protocol proxy metrics",
            "paper_source": "BeyondMimic guided diffusion task sections; reproduction/paper/source/root.tex:223-243; root.tex:549-593",
            "run_id": (
                "res/report_assets/official_importer_export_fig5_fig6_task_protocol_proxy/"
                "fig5_fig6_task_protocol_proxy.json"
            ),
            "reproduction_level": (
                "local virtual official-importer-export Fig.5/Fig.6 task-protocol proxy metrics"
            ),
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                f"This converts {metrics['row_count']} local importer-export closed-loop traces across "
                f"{metrics['seed_group_count']} seed groups and {metrics['task_count']} proxy tasks into explicit "
                "task-protocol proxy metrics: 299-step trace completion, local endpoint/root-reference error, "
                "target-body tracking error, guidance-cost decrease, reward delta vs the denoised baseline, and "
                "tracking-error delta vs the denoised baseline. It improves the reading report's evidence for what "
                "was actually run in simulation, but it remains qualitative-only: the thresholds are local proxy "
                "thresholds, not paper thresholds, and the artifact does not claim official BeyondMimic Fig. 5/"
                "Fig. 6 success, fall, collision, TensorRT, mocap, or real-robot results."
            ),
        }
    )


def add_official_importer_export_full_bundle_latent_projection_rows(rows: list[dict[str, str]]) -> None:
    assets = load_json(
        "res/report_assets/official_importer_export_full_bundle_latent_projection/"
        "official_importer_export_full_bundle_latent_projection_assets.json"
    )
    reproduction_value = {
        "status": assets["status"],
        "metrics": assets["metrics"],
        "checks": assets["checks"],
        "assets": assets["assets"],
        "claim_level": assets["interpretation"]["claim_level"],
    }
    rows.append(
        {
            "experiment": "report_assets:official_importer_export_full_bundle_latent_projection",
            "paper_value": (
                "BeyondMimic Fig. 5D shows latent-space visualization for walking/running transition behavior. "
                "The official t-SNE protocol, exact transition clips, and official VAE/diffusion checkpoints are "
                "not public in this local artifact set."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Figure 5D / latent visualization",
            "paper_source": "reproduction/paper/source/root.tex:223-243; root.tex:549-593",
            "run_id": (
                "res/report_assets/official_importer_export_full_bundle_latent_projection/"
                "official_importer_export_full_bundle_latent_projection_assets.json"
            ),
            "reproduction_level": "local virtual official-importer-export VAE latent PCA projection proxy",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This report asset projects local official-importer-export full-bundle VAE posterior means with PCA, "
                "labels 40 public-motion families, and records walk/run traces from 306176 local latent samples. It "
                "is useful Fig. 5D-adjacent reading-report evidence, but it is not the paper t-SNE panel, not an "
                "official BeyondMimic checkpoint result, not a closed-loop walk-to-run transition protocol, not "
                "TensorRT deployment, and not real-robot evidence."
            ),
        }
    )


def add_official_importer_export_full_bundle_transition_guidance_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/level_c/official_importer_export_full_bundle_transition_guidance_rollout_eval/"
        "level_c_official_importer_export_full_bundle_transition_guidance_rollout_eval.json"
    )
    row = audit["rows"][0]
    reproduction_value = {
        "status": audit["status"],
        "row": row,
        "transition_metrics": {
            key: {
                "late_minus_early_speed_mps": value.get("late_minus_early_speed_mps"),
                "target_speed_rmse_mps": value.get("target_speed_rmse_mps"),
                "speed_target_corr": value.get("speed_target_corr"),
                "x_progress_m": value.get("x_progress_m"),
                "lateral_abs_mean_m": value.get("lateral_abs_mean_m"),
            }
            for key, value in audit.get("transition_metrics", {}).items()
        },
        "checks": audit["checks"],
        "assets": audit["outputs"]["assets"],
        "claim_level": audit["interpretation"]["claim_level"],
    }
    rows.append(
        {
            "experiment": "level_c:official_importer_export_full_bundle_transition_guidance_rollout_eval",
            "paper_value": (
                "BeyondMimic Fig. 5B/Fig. 5D discuss latent transition behavior such as walking-to-running. "
                "The official transition protocol, official checkpoints, paper t-SNE data, and success/failure "
                "rollout traces are not public in this local artifact set."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Figure 5B / Figure 5D transition-related behavior",
            "paper_source": "reproduction/paper/source/root.tex:223-243; root.tex:549-593",
            "run_id": (
                "res/level_c/official_importer_export_full_bundle_transition_guidance_rollout_eval/"
                "level_c_official_importer_export_full_bundle_transition_guidance_rollout_eval.json"
            ),
            "reproduction_level": "local virtual official-importer-export walk-to-run transition proxy",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This runs one 299-step local IsaacLab closed-loop transition proxy on GPU4 using the recovered "
                "official-importer-export G1 USDA path and local PPO/VAE/denoiser checkpoints. It records a local "
                "velocity-ramp diagnostic, MP4 path, and report plots. The guided variant has positive late-vs-early "
                "speed delta but weak speed-target correlation and high target-speed RMSE, so it should be treated "
                "as diagnostic evidence, not a success-rate claim. It is not the paper transition protocol, not "
                "paper Fig. 5D t-SNE, not an official checkpoint result, not TensorRT deployment, and not real-robot "
                "evidence."
            ),
        }
    )


def add_official_importer_export_full_bundle_inpainting_guidance_rollout_rows(rows: list[dict[str, str]]) -> None:
    audit = load_json(
        "res/level_c/official_importer_export_full_bundle_inpainting_guidance_rollout_eval/"
        "level_c_official_importer_export_full_bundle_inpainting_guidance_rollout_eval.json"
    )
    row = audit["rows"][0]
    reproduction_value = {
        "status": audit["status"],
        "task": row["task"],
        "rollout_steps": row["rollout_steps"],
        "selected_physical_gpu": row["selected_physical_gpu"],
        "guided_keyframe_error_mean": row["guided_keyframe_error_mean"],
        "denoised_keyframe_error_mean": row["denoised_keyframe_error_mean"],
        "guided_keyframe_error_delta_vs_denoised": row["guided_keyframe_error_delta_vs_denoised"],
        "guidance_cost_delta_mean": row["guidance_cost_delta_mean"],
        "checks": audit["checks"],
        "outputs": audit["outputs"],
        "claim_level": audit["interpretation"]["claim_level"],
    }
    rows.append(
        {
            "experiment": "level_c:official_importer_export_full_bundle_inpainting_guidance_rollout_eval",
            "paper_value": (
                "BeyondMimic Fig. 6A shows motion inpainting with keyframes, including a cartwheel-style qualitative "
                "rollout. The official Fig. 6A rollout logs, exact keyframe protocol, VAE/diffusion checkpoints, "
                "and success/failure traces are not public in this local artifact set."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Figure 6A / motion inpainting with keyframes",
            "paper_source": "reproduction/paper/source/root.tex:241-243; root.tex:549-593",
            "run_id": (
                "res/level_c/official_importer_export_full_bundle_inpainting_guidance_rollout_eval/"
                "level_c_official_importer_export_full_bundle_inpainting_guidance_rollout_eval.json"
            ),
            "reproduction_level": (
                "local virtual official-importer-export future-keyframe inpainting diagnostic proxy"
            ),
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This adds one 299-step local IsaacLab closed-loop diagnostic for a synthetic future-keyframe/"
                "root-path inpainting cost on the official-importer-export G1 USDA path. It generated capture and "
                "video assets, but the guided keyframe proxy error is larger than the denoised baseline on this "
                "seed, so it is not a success claim. It remains qualitative-only virtual evidence: not the paper "
                "cartwheel keyframe protocol, not an official BeyondMimic checkpoint, not TensorRT deployment, and "
                "not real-robot validation."
            ),
        }
    )


def add_official_csv_loop_vae_closed_loop_rollout_rows(rows: list[dict[str, str]]) -> None:
    rollout = load_json(
        "res/level_c/official_csv_loop_vae_closed_loop_rollout_eval/"
        "tracking_g1_official_csv_loop_vae_closed_loop_rollout_eval.json"
    )
    assets = load_json(
        "res/report_assets/official_csv_loop_vae_closed_loop_rollout_eval/"
        "official_csv_loop_vae_closed_loop_rollout_assets.json"
    )
    video_asset = load_json(
        "res/visualization/official_csv_loop_vae_closed_loop_rollout/"
        "official_csv_loop_vae_closed_loop_rollout_video_asset.json"
    )
    aggregate = rollout["run"]["aggregate_metrics"]
    gpu_summary = rollout["run"]["gpu_metrics_summary"]
    reproduction_value = {
        "status": rollout["status"],
        "total_num_envs": aggregate["total_num_envs"],
        "rollout_steps": aggregate["rollout_steps"],
        "total_env_steps": aggregate["total_env_steps"],
        "teacher_vae_action_mse_mean": aggregate["teacher_vae_action_mse"]["mean"],
        "teacher_vae_action_abs_error_mean": aggregate["teacher_vae_action_abs_error"]["mean"],
        "reward_mean": aggregate["reward_mean"]["mean"],
        "done_count_total": aggregate["done_count_total"],
        "gpu_metrics_summary": gpu_summary,
        "peak_memory_each_gpu_at_least_10gb": rollout["checks"]["peak_memory_each_gpu_at_least_10gb"],
        "assets": assets["assets"],
        "video_asset": video_asset.get("assets", {}),
        "video_metrics": video_asset.get("metrics", {}),
        "video_claim_level": video_asset.get("claim_level", ""),
    }
    rows.append(
        {
            "experiment": "level_c:official_csv_loop_vae_closed_loop_rollout_eval",
            "paper_value": (
                "BeyondMimic uses a conditional VAE as part of a guided latent-control stack, but the paper's "
                "official VAE checkpoint and closed-loop VAE/diffusion rollout logs are not public in this local "
                "artifact set."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Conditional VAE / guided-control pipeline",
            "paper_source": "BeyondMimic VAE and guided diffusion method sections",
            "run_id": (
                "res/level_c/official_csv_loop_vae_closed_loop_rollout_eval/"
                "tracking_g1_official_csv_loop_vae_closed_loop_rollout_eval.json"
            ),
            "reproduction_level": "local virtual VAE action-reconstruction closed-loop rollout",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This full 299-step, two-rank IsaacLab run executes actions reconstructed by the local "
                "official-csv-loop conditional action VAE from local PPO teacher actions. It is stronger than the "
                "short decoded-action bridge because it runs 612352 simulated env steps, but it is not the official "
                "BeyondMimic VAE checkpoint, not an autonomous VAE policy, not receding-horizon diffusion guidance, "
                "not Fig. 5/Fig. 6 reproduction, and not real-robot evidence. GPU4 exceeded 10GB peak memory while "
                "GPU7 did not, so the resource usage is recorded rather than inflated. A separate single-env MP4 "
                "visualizes the same local VAE action-reconstruction mechanism for the reading report, but it is "
                "also local qualitative evidence rather than a paper-level video."
            ),
        }
    )


def add_official_csv_loop_vae_denoiser_onnx_async_rows(rows: list[dict[str, str]]) -> None:
    for label, audit_path in [
        (
            "official_csv_loop",
            "res/level_c/official_csv_loop_vae_denoiser_onnx_async/"
            "level_c_official_csv_loop_vae_denoiser_onnx_async_audit.json",
        ),
        (
            "official_csv_loop_full_bundle",
            "res/level_c/official_csv_loop_full_bundle_vae_denoiser_onnx_async/"
            "level_c_official_csv_loop_full_bundle_vae_denoiser_onnx_async_audit.json",
        ),
        (
            "official_importer_export_full_bundle",
            "res/level_c/official_importer_export_full_bundle_vae_denoiser_onnx_async/"
            "level_c_official_importer_export_full_bundle_vae_denoiser_onnx_async_audit.json",
        ),
    ]:
        audit = load_json(audit_path)
        reproduction_value = {
            "status": audit["status"],
            "providers": audit["settings"]["onnxruntime_available_providers"],
            "providers_used": audit["settings"]["onnxruntime_execution_providers_used"],
            "onnx_exports": audit["onnx_exports"],
            "consistency": audit["consistency"],
            "async_summary": audit["async_summary"],
            "claim_level": audit["interpretation"]["paper_level_status"],
        }
        rows.append(
            {
                "experiment": f"level_c:{label}_vae_denoiser_onnx_async_audit",
                "paper_value": (
                    "BeyondMimic reports deployment-oriented execution with a 25 Hz control loop, a 20 ms "
                    "diffusion budget, asynchronous diffusion, TensorRT, and an RTX 4060 Mobile Mini PC. Public "
                    "artifacts here do not include the paper deployment engine, official VAE/diffusion checkpoints, "
                    "or Mini-PC logs."
                ),
                "reproduction_value": stringify(reproduction_value),
                "absolute_difference": "",
                "relative_difference": "",
                "paper_figure_or_table": "Deployment protocol / runtime system",
                "paper_source": "BeyondMimic deployment discussion and method sections",
                "run_id": audit_path,
                "reproduction_level": "local CPU ONNXRuntime export and async proxy audit",
                "comparison_type": "qualitative_only",
                "difference_explanation": (
                    f"This exports the local {label.replace('_', '-')} VAE encoder/decoder and state-latent "
                    "denoiser to ONNX, checks ONNXRuntime CPU outputs against PyTorch, and measures local "
                    "sequential/thread-pool async microbenchmarks. It records that CUDAExecutionProvider and "
                    "TensorRT provider are unavailable in the local ORT build. Therefore it is useful deployment-path "
                    "evidence for the reading report, but it is not TensorRT, not the paper RTX 4060 Mini-PC latency "
                    "result, not CppAD guidance, not live IsaacLab deployment, not an official BeyondMimic "
                    "checkpoint, and not real-robot evidence."
                ),
            }
        )


def add_resource_adjusted_state_latent_guidance_rows(rows: list[dict[str, str]]) -> None:
    guidance_audit = load_json(
        "res/level_c/resource_adjusted_state_latent_guidance_eval/"
        "level_c_resource_adjusted_state_latent_guidance_eval.json"
    )
    worker = guidance_audit["worker_summary"]
    task_value = {
        task: {
            "mean_best_cost_delta": summary["mean_best_cost_delta"],
            "mean_positive_delta_fraction": summary["mean_positive_delta_fraction"],
            "all_best_costs_improve": summary["all_best_costs_improve"],
            "all_best_gradients_nonzero": summary["all_best_gradients_nonzero"],
        }
        for task, summary in worker["task_summaries"].items()
    }
    reproduction_value = {
        "status": guidance_audit["status"],
        "total_selected_windows": worker["metrics"]["total_selected_windows"],
        "row_count": worker["metrics"]["row_count"],
        "tasks": worker["settings"]["tasks"],
        "scales": worker["settings"]["scales"],
        "task_summaries": task_value,
        "gpu_metrics_summary": guidance_audit.get("gpu_metrics_summary", {}),
    }
    rows.append(
        {
            "experiment": "level_c:resource_adjusted_state_latent_guidance_eval",
            "paper_value": (
                "BeyondMimic applies guided diffusion to produce closed-loop humanoid skills in simulation and on a "
                "real Unitree G1; the paper-level rollout logs/checkpoints are not publicly available here."
            ),
            "reproduction_value": stringify(reproduction_value),
            "absolute_difference": "",
            "relative_difference": "",
            "paper_figure_or_table": "Guided diffusion / Fig. 5-6 prerequisite",
            "paper_source": "BeyondMimic guided diffusion sections and local resource-adjusted denoiser evidence",
            "run_id": (
                "res/level_c/resource_adjusted_state_latent_guidance_eval/"
                "level_c_resource_adjusted_state_latent_guidance_eval.json"
            ),
            "reproduction_level": "resource-adjusted offline state-latent guidance surrogate",
            "comparison_type": "qualitative_only",
            "difference_explanation": (
                "This evaluates task-cost gradients over local 192-D policy-observation plus VAE-latent denoiser "
                "outputs on validation/test windows. It demonstrates that the local denoiser can be connected to "
                "offline guidance objectives, but it is not an IsaacLab closed-loop rollout, not official Fig. 5/Fig. 6 "
                "evidence, and not a paper-level guidance reproduction."
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
    add_tracking_official_csv_loop_ppo_training_rows(rows)
    add_tracking_official_csv_loop_ppo_checkpoint_eval_rows(rows)
    add_tracking_official_csv_loop_ppo_multiseed_eval_rows(rows)
    add_tracking_official_csv_loop_full_bundle_rows(rows)
    add_tracking_resource_adjusted_teacher_rollout_dataset_rows(rows)
    add_tracking_official_csv_loop_teacher_rollout_dataset_rows(rows)
    add_tracking_official_csv_loop_full_bundle_teacher_rollout_dataset_rows(rows)
    add_tracking_urdf_source_equivalence_rows(rows)
    add_tracking_official_replay_entry_rows(rows)
    add_tracking_official_csv_to_npz_loop_patch_rows(rows)
    add_tracking_official_csv_to_npz_loop_full_dataset_rows(rows)
    add_tracking_official_csv_to_npz_loop_full_dataset_official_importer_export_rows(rows)
    add_tracking_official_csv_loop_full_dataset_task_eval_rows(rows)
    add_tracking_official_importer_export_full_dataset_task_eval_rows(rows)
    add_tracking_official_importer_export_full_bundle_ppo_rows(rows)
    add_tracking_official_importer_export_full_bundle_scaled_ppo_rows(rows)
    add_official_importer_export_tracking_eval_summary_asset_rows(rows)
    add_tracking_official_importer_export_full_bundle_teacher_rollout_dataset_rows(rows)
    add_tracking_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset_rows(rows)
    add_tracking_official_replay_loop_patch_rows(rows)
    add_tracking_official_replay_loop_full_dataset_rows(rows)
    add_tracking_official_replay_loop_full_dataset_official_importer_export_rows(rows)
    add_official_importer_export_replay_full_dataset_report_asset_rows(rows)
    add_tracking_official_importer_export_full_dataset_reference_replay_video_rows(rows)
    add_tracking_g1_import_config_variant_rows(rows)
    add_tracking_g1_in_memory_gpu4_probe_rows(rows)
    add_resource_adjusted_teacher_rollout_vae_training_rows(rows)
    add_official_csv_loop_teacher_rollout_vae_training_rows(rows)
    add_official_csv_loop_state_latent_dataset_and_diffusion_rows(rows)
    add_official_csv_loop_full_bundle_downstream_rows(rows)
    add_official_importer_export_full_bundle_vae_rows(rows)
    add_official_importer_export_full_bundle_vae_closed_loop_rows(rows)
    add_official_importer_export_full_bundle_downstream_rows(rows)
    add_official_importer_export_scaled_ppo_downstream_rows(rows)
    add_resource_adjusted_state_latent_dataset_and_diffusion_rows(rows)
    add_official_csv_loop_state_latent_guidance_rows(rows)
    add_official_csv_loop_full_bundle_guidance_rows(rows)
    add_official_importer_export_full_bundle_guidance_rows(rows)
    add_official_csv_loop_guidance_vae_action_decode_rows(rows)
    add_official_csv_loop_guided_action_rollout_probe_rows(rows)
    add_official_csv_loop_action_guidance_rollout_rows(rows)
    add_official_csv_loop_receding_latent_guidance_rollout_rows(rows)
    add_official_csv_loop_task_conditioned_latent_guidance_rollout_rows(rows)
    add_official_csv_loop_task_conditioned_latent_guidance_multiseed_rows(rows)
    add_guided_vs_unguided_closed_loop_matrix_rows(rows)
    add_official_csv_loop_full_bundle_receding_latent_guidance_rollout_rows(rows)
    add_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_rows(rows)
    add_official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_rows(rows)
    add_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_rows(rows)
    add_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_rows(rows)
    add_official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_rows(rows)
    add_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_rows(rows)
    add_official_importer_export_full_bundle_task_conditioned_guidance_success_boundary_rows(rows)
    add_official_importer_export_full_bundle_transition_guidance_rows(rows)
    add_official_importer_export_full_bundle_latent_projection_rows(rows)
    add_official_importer_export_full_bundle_inpainting_guidance_rollout_rows(rows)
    add_official_importer_export_fig5_fig6_proxy_protocol_matrix_rows(rows)
    add_official_importer_export_fig5_fig6_task_protocol_proxy_rows(rows)
    add_official_csv_loop_vae_closed_loop_rollout_rows(rows)
    add_official_csv_loop_vae_denoiser_onnx_async_rows(rows)
    add_resource_adjusted_state_latent_guidance_rows(rows)
    add_official_importer_export_scaled_ppo_guidance_rows(rows)
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
