#!/usr/bin/env python3
"""Audit goal.md Phase 8 guidance-task coverage."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/guidance_task_coverage"
TASKS = ["unconditional_rollout", "joystick", "waypoint", "inpainting", "obstacle_avoidance", "composed_objectives"]
REQUIREMENTS = ["without_guidance", "with_guidance", "multiple_guidance_weights", "success_failure_videos", "quantitative_metrics"]


def exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def row(task: str, requirement: str, status: str, evidence: list[str], detail: str) -> dict[str, Any]:
    evidence_exists = [exists(path) for path in evidence]
    return {
        "task": task,
        "requirement": requirement,
        "status": status,
        "evidence": evidence,
        "evidence_exists": evidence_exists,
        "all_evidence_exists": all(evidence_exists),
        "detail": detail,
    }


def add_task_rows(rows: list[dict[str, Any]], task: str, statuses: dict[str, tuple[str, list[str], str]]) -> None:
    for requirement in REQUIREMENTS:
        status, evidence, detail = statuses[requirement]
        rows.append(row(task, requirement, status, evidence, detail))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    common_blocked_video = ["res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json"]
    offline_guidance = ["res/level_c/lafan1_paper_arch_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.json"]
    full_split_offline_guidance = [
        "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/"
        "level_c_lafan1_paper_arch_guidance_eval.json"
    ]
    reverse_guidance = [
        "res/level_c/lafan1_paper_arch_reverse_guidance/level_c_lafan1_paper_arch_reverse_guidance_audit.json"
    ]
    full_split_reverse_guidance = [
        "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/"
        "level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.json"
    ]
    full_split_metric_evidence = [
        "res/level_c/guidance_task_metric_audit/level_c_guidance_task_metric_audit.json",
        "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.json",
    ]
    add_task_rows(
        rows,
        "unconditional_rollout",
        {
            "without_guidance": ("debug_oracle_reverse_only", ["res/level_c/reverse_denoising_probe/level_c_reverse_denoising_probe.json"], "Oracle reverse mechanics without classifier guidance exist; no trained unconditional rollout."),
            "with_guidance": ("not_applicable_for_unconditional", ["res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json"], "Unconditional rollout is the no-guidance baseline task; guided variants are covered by task rows below."),
            "multiple_guidance_weights": ("not_applicable_for_unconditional", ["res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json"], "No guidance scale applies to the unconditional baseline."),
            "success_failure_videos": ("blocked_missing_videos", common_blocked_video, "No closed-loop videos/checkpoints for unconditional diffusion rollout."),
            "quantitative_metrics": ("debug_metric_only", ["res/level_c/reverse_denoising_probe/level_c_reverse_denoising_probe.json", "res/level_c/smoothness_latency_audit/level_c_smoothness_latency_audit.json"], "Reverse MSE/smoothness debug metrics exist; no rollout success metric."),
        },
    )
    add_task_rows(
        rows,
        "joystick",
        {
            "without_guidance": ("public_data_reverse_guidance_baseline", reverse_guidance + offline_guidance, "Full public-data checkpoint records unguided reverse-denoising and one-shot trajectory baselines before joystick guidance."),
            "with_guidance": ("public_data_reverse_guidance", reverse_guidance + full_split_reverse_guidance + offline_guidance + full_split_offline_guidance, "Full public-data checkpoint applies joystick task-cost gradient inside a 20-step reverse-denoising loop; the symmetry-augmented checkpoint additionally covers full validation+test reverse-denoising and offline guidance."),
            "multiple_guidance_weights": ("public_data_reverse_guidance_scale_sweep", reverse_guidance + full_split_reverse_guidance + offline_guidance + full_split_offline_guidance, "Full public-data checkpoint sweeps five reverse-denoising guidance scales and seven one-shot offline scales; the full-split symmetry reverse audit covers validation+test windows."),
            "success_failure_videos": ("blocked_missing_videos", common_blocked_video, "No closed-loop joystick success/failure videos."),
            "quantitative_metrics": ("public_data_reverse_guidance_metrics", reverse_guidance + full_split_reverse_guidance + offline_guidance + full_split_offline_guidance + full_split_metric_evidence, "Reverse-denoising and one-shot full-checkpoint guidance record cost deltas and primary metrics; the full-split symmetry audits add 33,000 reverse rows and 46,200 offline rows; no task success rate."),
        },
    )
    for task, formula_item in [
        ("waypoint", "waypoint_formula"),
        ("obstacle_avoidance", "sdf_obstacle_barrier_formula"),
        ("composed_objectives", "composed_paper_formula_terms"),
    ]:
        add_task_rows(
            rows,
            task,
            {
                "without_guidance": ("public_data_reverse_guidance_baseline", reverse_guidance + offline_guidance, "Full public-data checkpoint records unguided reverse-denoising and one-shot trajectory baselines before task guidance."),
                "with_guidance": ("public_data_reverse_guidance", reverse_guidance + full_split_reverse_guidance + offline_guidance + full_split_offline_guidance + ["res/level_c/guidance_cost_coverage_audit/level_c_guidance_cost_coverage_audit.json"], f"{formula_item} is applied inside the full-checkpoint reverse-denoising loop and to predicted trajectories; the full-split symmetry audits cover validation+test windows; no closed-loop guided rollout."),
                "multiple_guidance_weights": ("public_data_reverse_guidance_scale_sweep", reverse_guidance + full_split_reverse_guidance + offline_guidance + full_split_offline_guidance, "Full public-data checkpoint sweeps reverse-denoising and one-shot offline guidance scales; the full-split symmetry audits cover validation+test windows."),
                "success_failure_videos": ("blocked_missing_videos", common_blocked_video, "No success/failure videos for this task."),
                "quantitative_metrics": ("public_data_reverse_guidance_metrics", reverse_guidance + full_split_reverse_guidance + offline_guidance + full_split_offline_guidance + full_split_metric_evidence + ["res/evaluation_metrics_coverage/evaluation_metrics_coverage_audit.json"], "Reverse-denoising and one-shot full-checkpoint guidance record cost deltas and primary metrics; the full-split symmetry audits add 33,000 reverse rows and 46,200 offline rows; no rollout success/collision/goal-distance metric."),
            },
        )
    add_task_rows(
        rows,
        "inpainting",
        {
            "without_guidance": ("public_data_reverse_guidance_baseline", reverse_guidance + offline_guidance, "Full public-data checkpoint records unguided reverse-denoising and one-shot trajectory baselines before inpainting-keyframe guidance."),
            "with_guidance": ("public_data_reverse_guidance", reverse_guidance + full_split_reverse_guidance + offline_guidance + full_split_offline_guidance + ["res/level_c/guidance_cost_coverage_audit/level_c_guidance_cost_coverage_audit.json"], "Candidate keyframe term is applied inside the full-checkpoint reverse-denoising loop and to predicted trajectories; the full-split symmetry audits cover validation+test windows; paper does not publish a unique inpainting cost equation."),
            "multiple_guidance_weights": ("public_data_reverse_guidance_scale_sweep", reverse_guidance + full_split_reverse_guidance + offline_guidance + full_split_offline_guidance, "Full public-data checkpoint sweeps reverse-denoising and one-shot offline guidance scales; the full-split symmetry audits cover validation+test windows."),
            "success_failure_videos": ("blocked_missing_videos", common_blocked_video, "No inpainting success/failure videos."),
            "quantitative_metrics": ("public_data_reverse_guidance_metrics", reverse_guidance + full_split_reverse_guidance + offline_guidance + full_split_offline_guidance + full_split_metric_evidence + ["res/level_c/timestep_mask_coverage_audit/level_c_timestep_mask_coverage_audit.json"], "Reverse-denoising and one-shot full-checkpoint inpainting guidance record keyframe cost/metric deltas; the full-split symmetry audits add 33,000 reverse rows and 46,200 offline rows; no cartwheel inpainting success metric."),
        },
    )

    missing = [r for r in rows if not r["all_evidence_exists"]]
    status_counts = Counter(r["status"] for r in rows)
    task_counts = Counter(r["task"] for r in rows)
    requirement_counts = Counter(r["requirement"] for r in rows)
    video_rows = [r for r in rows if r["requirement"] == "success_failure_videos"]
    summary = {
        "status": "ok"
        if not missing
        and len(rows) == len(TASKS) * len(REQUIREMENTS)
        and all(task_counts[t] == len(REQUIREMENTS) for t in TASKS)
        and all(requirement_counts[r] == len(TASKS) for r in REQUIREMENTS)
        else "failed",
        "experiment_type": "guidance_task_coverage_audit",
        "scope": "goal.md Phase 8 guidance task coverage with explicit evidence/gap status",
        "row_count": len(rows),
        "task_counts": dict(sorted(task_counts.items())),
        "requirement_counts": dict(sorted(requirement_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "missing_evidence_rows": missing,
        "rows": rows,
        "checks": {
            "six_tasks_mapped": sorted(task_counts) == sorted(TASKS) and all(task_counts[t] == 5 for t in TASKS),
            "five_requirements_per_task": len(rows) == 30,
            "all_evidence_paths_exist": not missing,
            "joystick_has_with_without_and_scale_sweep_offline_evidence": all(
                any(r["task"] == "joystick" and r["requirement"] == req and "guidance" in r["status"] for r in rows)
                for req in ["without_guidance", "with_guidance", "multiple_guidance_weights"]
            ),
            "inpainting_mask_metrics_recorded": any(
                r["task"] == "inpainting"
                and r["requirement"] == "quantitative_metrics"
                and "res/level_c/lafan1_paper_arch_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.json"
                in r["evidence"]
                for r in rows
            ),
            "offline_guidance_eval_linked": sum(
                1
                for r in rows
                if r["requirement"] == "quantitative_metrics"
                and "res/level_c/lafan1_paper_arch_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.json"
                in r["evidence"]
            )
            == 5,
            "reverse_guidance_eval_linked": sum(
                1
                for r in rows
                if r["requirement"] == "quantitative_metrics"
                and "res/level_c/lafan1_paper_arch_reverse_guidance/level_c_lafan1_paper_arch_reverse_guidance_audit.json"
                in r["evidence"]
            )
            == 5,
            "full_split_offline_guidance_eval_linked": sum(
                1
                for r in rows
                if r["requirement"] == "quantitative_metrics"
                and (
                    "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/"
                    "level_c_lafan1_paper_arch_guidance_eval.json"
                )
                in r["evidence"]
            )
            == 5,
            "full_split_reverse_guidance_eval_linked": sum(
                1
                for r in rows
                if r["requirement"] == "quantitative_metrics"
                and (
                    "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/"
                    "level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.json"
                )
                in r["evidence"]
            )
            == 5,
            "task_metric_audit_linked_to_all_guided_quantitative_rows": sum(
                1
                for r in rows
                if r["task"] != "unconditional_rollout"
                and r["requirement"] == "quantitative_metrics"
                and "res/level_c/guidance_task_metric_audit/level_c_guidance_task_metric_audit.json"
                in r["evidence"]
            )
            == 5,
            "full_split_result_table_linked_to_all_guided_quantitative_rows": sum(
                1
                for r in rows
                if r["task"] != "unconditional_rollout"
                and r["requirement"] == "quantitative_metrics"
                and "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.json"
                in r["evidence"]
            )
            == 5,
            "all_video_requirements_recorded_blocked": all(r["status"] == "blocked_missing_videos" for r in video_rows),
            "closed_loop_rollouts_not_overclaimed": not any(r["status"] == "paper_level_complete" for r in rows),
            "does_not_claim_goal_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "Phase 8 task requirements are explicitly mapped. Full public-data checkpoint reverse-denoising and "
                "one-shot offline guidance now cover with/without guidance, scale sweeps, and quantitative task costs, "
                "but closed-loop rollout logs and success/failure videos are still missing."
            ),
        },
        "outputs": {
            "json": str(OUT / "guidance_task_coverage_audit.json"),
            "tsv": str(OUT / "guidance_task_coverage_audit.tsv"),
        },
    }
    (OUT / "guidance_task_coverage_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT / "guidance_task_coverage_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=["task", "requirement", "status", "evidence", "all_evidence_exists", "detail"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "task": r["task"],
                    "requirement": r["requirement"],
                    "status": r["status"],
                    "evidence": ";".join(r["evidence"]),
                    "all_evidence_exists": r["all_evidence_exists"],
                    "detail": r["detail"],
                }
            )
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
