#!/usr/bin/env python3
"""Audit goal.md Section 12 evaluation-metric coverage."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/evaluation_metrics_coverage"


def exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def row(section: str, metric: str, status: str, evidence: list[str], detail: str) -> dict[str, Any]:
    evidence_exists = [exists(path) for path in evidence]
    return {
        "section": section,
        "metric": metric,
        "status": status,
        "evidence": evidence,
        "evidence_exists": evidence_exists,
        "all_evidence_exists": all(evidence_exists),
        "detail": detail,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        row("motion_tracking", "local position error", "released_data", ["res/tables/released_data_metrics_summary/released_tracking_ablation_metrics.csv", "res/tables/released_data_metrics_summary/released_data_metrics_summary.json", "res/tables/released_data_statistical_audit/released_ablation_effect_sizes.csv"], "Released-data numeric/statistical summaries report local tracking position-error rows and effect-size style comparisons; no live tracking eval run."),
        row("motion_tracking", "local orientation error", "released_data", ["res/tables/released_data_metrics_summary/released_tracking_ablation_metrics.csv", "res/tables/released_data_metrics_summary/released_data_metrics_summary.json", "res/tables/released_data_statistical_audit/released_ablation_effect_sizes.csv"], "Released-data numeric/statistical summaries report local tracking orientation-error rows and effect-size style comparisons; no live rollout metric."),
        row("motion_tracking", "local linear velocity error", "released_data", ["res/tables/released_data_metrics_summary/released_tracking_ablation_metrics.csv", "res/tables/released_data_metrics_summary/released_data_metrics_summary.json", "res/tables/released_data_statistical_audit/released_ablation_effect_sizes.csv"], "Released-data numeric/statistical summaries report local linear-velocity tracking-error rows and effect-size style comparisons; no live rollout metric."),
        row("motion_tracking", "local angular velocity error", "released_data", ["res/tables/released_data_metrics_summary/released_tracking_ablation_metrics.csv", "res/tables/released_data_metrics_summary/released_data_metrics_summary.json", "res/tables/released_data_statistical_audit/released_ablation_effect_sizes.csv"], "Released-data numeric/statistical summaries report local angular-velocity tracking-error rows and effect-size style comparisons; no live rollout metric."),
        row("motion_tracking", "global position error", "released_data", ["res/tables/released_data_metrics_summary/released_tracking_ablation_metrics.csv", "res/tables/released_data_metrics_summary/released_data_metrics_summary.json", "res/tables/released_data_statistical_audit/released_ablation_effect_sizes.csv"], "Released-data numeric/statistical summaries report global position-error rows and effect-size style comparisons."),
        row("motion_tracking", "global yaw error", "released_data", ["res/tables/released_data_metrics_summary/released_tracking_ablation_metrics.csv", "res/tables/released_data_statistical_audit/released_ablation_effect_sizes.csv", "res/released_figures/ablation_orientation_representation/ablation_orientation_representation_global.pdf"], "Released-data summary/statistical tables and global-error panels cover global orientation/yaw-related error; no live rollout metric."),
        row("motion_tracking", "success rate", "formula_api_only", ["res/paper_skill_success_table_audit/skill_success_table_data_audit.json", "res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json", "res/tests/core_math_unit_tests/core_math_unit_tests.json", "res/blocked_gates/blocked_gate_audit.json"], "Skill-success table is source/data-audited and the success-rate formula API is tested, but local sim/real success execution is blocked."),
        row("motion_tracking", "fall rate", "formula_api_only", ["res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json", "res/tests/core_math_unit_tests/core_math_unit_tests.json", "res/blocked_gates/blocked_gate_audit.json"], "Fall-rate formula API is tested, but live Isaac/robot rollout termination statistics are missing."),
        row("motion_tracking", "episode length", "blocked_or_missing", ["res/run_management_audit/run_management_audit.json", "res/blocked_gates/blocked_gate_audit.json"], "Run schema exists, but no completed tracking rollout episodes."),
        row("motion_tracking", "iterations to convergence", "blocked_or_missing", ["res/run_management_audit/run_management_audit.json", "res/blocked_gates/blocked_gate_audit.json"], "No long PPO training run reached convergence."),
        row("adaptive_sampling", "per-bin failure rate", "debug_or_released", ["res/tracking/adaptive_sampling_discrepancy_audit/adaptive_sampling_discrepancy_audit.json", "res/released_figures/adaptive_sampling/adaptive_sampling_probability_evolution_heatmap.pdf"], "Paper/code discrepancy and released heatmap evidence exist; no live rerun failure table."),
        row("adaptive_sampling", "per-bin sampling probability", "released_data", ["res/released_figures/adaptive_sampling/adaptive_sampling_probability_evolution_heatmap.pdf", "res/comparison/paper_vs_reproduction.json"], "Released sampling probability evolution panel reproduced."),
        row("adaptive_sampling", "per-bin resets", "debug_or_released", ["res/tracking/adaptive_sampling_discrepancy_audit/adaptive_sampling_discrepancy_audit.json"], "Adaptive sampling audit covers bin distribution mechanics, not full live reset logs."),
        row("adaptive_sampling", "iterations until success", "blocked_or_missing", ["res/blocked_gates/blocked_gate_audit.json"], "Requires live training/evaluation progression."),
        row("adaptive_sampling", "failed segments", "debug_or_released", ["res/tracking/adaptive_sampling_discrepancy_audit/adaptive_sampling_discrepancy_audit.json", "res/released_figures/adaptive_sampling/adaptive_sampling_w_iter_matrix.pdf"], "Released/adaptive failure-map evidence exists; no live rerun segment log."),
        row("adaptive_sampling", "distribution evolution", "released_data", ["res/released_figures/adaptive_sampling/adaptive_sampling_probability_evolution_heatmap.pdf"], "Released distribution-evolution panel reproduced."),
        row("vae", "action MSE", "public_data_checkpoint", ["res/level_c/lafan1_paper_arch_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_audit.json", "res/level_c/lafan1_paper_arch_vae_diffusion_training/lafan1_paper_arch_vae_diffusion_training.json"], "Full paper-architecture public-LAFAN1 checkpoint reports split-wise decoded-action MSE; no true DAgger rollout."),
        row("vae", "KL divergence", "public_data_checkpoint", ["res/level_c/lafan1_paper_arch_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_audit.json"], "Full paper-architecture public-LAFAN1 checkpoint reports split-wise KL means."),
        row("vae", "teacher-student discrepancy", "public_data_checkpoint", ["res/level_c/lafan1_paper_arch_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_audit.json", "res/level_c/dagger_schema_audit/dagger_schema_audit.json"], "Full public-data checkpoint reports split-wise teacher-student MSE, but teacher is the local supervised teacher head rather than official true DAgger teacher rollouts."),
        row("vae", "tracking error", "formula_api_only", ["res/level_c/dataset_collection_protocol_audit/level_c_dataset_collection_protocol_audit.json", "res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json", "res/tests/core_math_unit_tests/core_math_unit_tests.json"], "Tracking-error formula API is tested, but trained VAE closed-loop tracking rollout is missing."),
        row("vae", "closed-loop survival", "blocked_or_missing", ["res/level_c/dataset_collection_protocol_audit/level_c_dataset_collection_protocol_audit.json", "res/blocked_gates/blocked_gate_audit.json"], "Needs live VAE policy rollouts."),
        row("vae", "fall rate", "formula_api_only", ["res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json", "res/tests/core_math_unit_tests/core_math_unit_tests.json", "res/blocked_gates/blocked_gate_audit.json"], "Fall-rate formula API is tested, but closed-loop VAE termination statistics are missing."),
        row("vae", "latent smoothness", "public_data_checkpoint", ["res/level_c/lafan1_paper_arch_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_audit.json", "res/level_c/vae_latent_probe/level_c_vae_latent_probe.json"], "Full public-data checkpoint reports latent first/second-difference smoothness; interpolation remains debug-only."),
        row("vae", "latent interpolation", "debug_only", ["res/level_c/vae_latent_probe/level_c_vae_latent_probe.json"], "Three-seed latent interpolation endpoint and neighbor metrics exist."),
        row("diffusion_guidance", "state reconstruction error", "public_data_checkpoint", ["res/level_c/lafan1_paper_arch_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_audit.json"], "Full public-data checkpoint reports split-wise projected-state reconstruction MSE; no closed-loop rollout metric."),
        row("diffusion_guidance", "latent reconstruction error", "public_data_checkpoint", ["res/level_c/lafan1_paper_arch_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_audit.json"], "Full public-data checkpoint reports split-wise latent reconstruction MSE; no official VAE rollout latent dataset."),
        row("diffusion_guidance", "trajectory reconstruction error", "public_data_checkpoint", ["res/level_c/lafan1_paper_arch_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_audit.json", "res/level_c/lafan1_paper_arch_vae_diffusion_training/lafan1_paper_arch_vae_diffusion_training.json"], "Full public-data checkpoint reports split-wise tau trajectory reconstruction MSE; no closed-loop rollout metric."),
        row("diffusion_guidance", "unconditional success", "blocked_or_missing", ["res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json"], "Needs trained unguided policy rollout success logs."),
        row("diffusion_guidance", "guided success", "blocked_or_missing", ["res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json"], "Needs closed-loop guided task rollout success logs."),
        row("diffusion_guidance", "velocity error", "public_data_checkpoint", ["res/comparison/paper_vs_reproduction.json", "res/tables/released_data_metrics_summary/released_tracking_ablation_metrics.csv", "res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json", "res/tests/core_math_unit_tests/core_math_unit_tests.json", "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/level_c_lafan1_paper_arch_guidance_eval.json", "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.json", "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.json"], "Paper 12.14/13.65 closed-loop velocity errors remain comparison-only/unreproduced, but the public LAFAN1 paper-architecture checkpoint now records joystick velocity-command primary metrics over validation+test windows in offline and reverse guidance sweeps."),
        row("diffusion_guidance", "goal distance", "public_data_checkpoint", ["res/level_c/guidance_cost_coverage_audit/level_c_guidance_cost_coverage_audit.json", "res/level_c/guidance_task_metric_audit/level_c_guidance_task_metric_audit.json", "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/level_c_lafan1_paper_arch_guidance_eval.json", "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.json", "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.json"], "Waypoint guidance formula/debug gradients are covered, and full-split public-data offline/reverse guidance records waypoint primary-metric and cost improvements over validation+test windows; no closed-loop navigation success rate is claimed."),
        row("diffusion_guidance", "collision rate", "blocked_or_missing", ["res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json"], "Needs SDF obstacle closed-loop rollout logs."),
        row("diffusion_guidance", "fall rate", "formula_api_only", ["res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json", "res/tests/core_math_unit_tests/core_math_unit_tests.json", "res/blocked_gates/blocked_gate_audit.json"], "Fall-rate formula API is tested, but live guided rollout termination statistics are missing."),
        row("diffusion_guidance", "action smoothness", "public_data_checkpoint", ["res/level_c/lafan1_paper_arch_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_audit.json", "res/level_c/diffusion_to_vae_action_smoothness_audit/level_c_diffusion_to_vae_action_smoothness_audit.json"], "Full public-data checkpoint reports decoded predicted action first/second-difference smoothness; not closed-loop action smoothness."),
        row("diffusion_guidance", "trajectory smoothness", "public_data_checkpoint", ["res/level_c/lafan1_paper_arch_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_audit.json", "res/level_c/smoothness_latency_audit/level_c_smoothness_latency_audit.json"], "Full public-data checkpoint reports predicted tau and latent first/second-difference smoothness; not rollout smoothness."),
        row("diffusion_guidance", "inference latency", "public_data_checkpoint", ["res/level_c/lafan1_paper_arch_onnx_latency/level_c_lafan1_paper_arch_onnx_latency_audit.json", "res/level_c/deployment_protocol_audit/level_c_deployment_protocol_audit.json"], "Full public-data checkpoint has PyTorch CPU and ONNX ReferenceEvaluator latency; no TensorRT/asynchronous deployment benchmark."),
        row("diffusion_guidance", "denoising latency", "public_data_checkpoint", ["res/level_c/lafan1_paper_arch_onnx_latency/level_c_lafan1_paper_arch_onnx_latency_audit.json", "res/level_c/deployment_protocol_audit/level_c_deployment_protocol_audit.json"], "Full public-data diffusion denoiser latency is measured on host CPU; no TensorRT or paper Mini PC deployment measurement."),
        row("diffusion_guidance", "guidance cost", "public_data_checkpoint", ["res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.json", "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/level_c_lafan1_paper_arch_guidance_eval.json", "res/level_c/guidance_cost_coverage_audit/level_c_guidance_cost_coverage_audit.json", "res/level_c/smoothness_latency_audit/level_c_smoothness_latency_audit.json"], "Full-split symmetry-augmented public-data checkpoint guidance evaluates joystick, waypoint, obstacle, inpainting, and composed costs over validation+test windows in both reverse-denoising and offline one-shot modes; no closed-loop success metric."),
        row("statistics", "3 seeds", "public_data_checkpoint", ["res/level_c/lafan1_paper_arch_multiseed_audit/level_c_lafan1_paper_arch_multiseed_audit.json", "res/level_c/lafan1_paper_arch_symmetry_augmented_multiseed_audit/level_c_lafan1_paper_arch_symmetry_augmented_multiseed_audit.json"], "Full paper-architecture public-LAFAN1 VAE/diffusion training and symmetry-augmented public-LAFAN1 training now both have 3 seed summaries; still not official DAgger/rollout paper data."),
        row("statistics", "mean", "public_data_checkpoint", ["res/level_c/lafan1_paper_arch_multiseed_audit/level_c_lafan1_paper_arch_multiseed_audit.json", "res/level_c/lafan1_paper_arch_symmetry_augmented_multiseed_audit/level_c_lafan1_paper_arch_symmetry_augmented_multiseed_audit.json", "res/tables/released_data_statistical_audit/released_data_statistical_audit.json"], "Full public-data paper-architecture and symmetry-augmented 3-seed audits report means for core VAE/diffusion metrics; released-data audit reports means for extracted paper/released tables."),
        row("statistics", "standard deviation", "public_data_checkpoint", ["res/level_c/lafan1_paper_arch_multiseed_audit/level_c_lafan1_paper_arch_multiseed_audit.json", "res/level_c/lafan1_paper_arch_symmetry_augmented_multiseed_audit/level_c_lafan1_paper_arch_symmetry_augmented_multiseed_audit.json", "res/tables/released_data_statistical_audit/released_data_statistical_audit.json"], "Full public-data paper-architecture and symmetry-augmented 3-seed audits report sample standard deviations for core VAE/diffusion metrics; released-data audit reports standard deviations where source rows expose them."),
        row("statistics", "individual results", "public_data_checkpoint", ["res/level_c/lafan1_paper_arch_multiseed_audit/level_c_lafan1_paper_arch_multiseed_audit.json", "res/level_c/lafan1_paper_arch_symmetry_augmented_multiseed_audit/level_c_lafan1_paper_arch_symmetry_augmented_multiseed_audit.json"], "The full public-data paper-architecture and symmetry-augmented 3-seed audits write individual seed rows, checkpoint hashes, and per-seed metrics."),
        row("statistics", "trial count", "partial", ["res/evaluation/trial_failure_accounting_audit/trial_failure_accounting_audit.json", "res/results_claims_audit/results_claims_audit.json", "res/paper_skill_success_table_audit/skill_success_table_data_audit.json", "res/tables/released_data_statistical_audit/released_data_statistical_audit.json"], "Paper/source trial counts, released-data metric rows, debug seed runs, and run-catalog counts are audited where available; local full rollout trial counts missing."),
        row("statistics", "failure count", "partial", ["res/evaluation/trial_failure_accounting_audit/trial_failure_accounting_audit.json", "res/results_claims_audit/results_claims_audit.json", "res/failed_runs/failed_run_audit/failed_run_audit.json"], "Retained failed-run evidence and claim-accounting gaps are audited; full rollout failure counts remain missing."),
    ]

    missing_evidence = [r for r in rows if not r["all_evidence_exists"]]
    status_counts = Counter(r["status"] for r in rows)
    section_counts = Counter(r["section"] for r in rows)
    explicit_goal_metrics = {
        "motion_tracking": 10,
        "adaptive_sampling": 6,
        "vae": 8,
        "diffusion_guidance": 14,
        "statistics": 6,
    }
    summary = {
        "status": "ok" if not missing_evidence and len(rows) == 44 and all(section_counts[k] == v for k, v in explicit_goal_metrics.items()) else "failed",
        "experiment_type": "evaluation_metric_coverage_audit",
        "scope": "goal.md Section 12 evaluation metric coverage with explicit evidence/gap status",
        "row_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "section_counts": dict(sorted(section_counts.items())),
        "missing_evidence_rows": missing_evidence,
        "rows": rows,
        "checks": {
            "all_44_goal_metrics_mapped": len(rows) == 44,
            "section_counts_match_goal_md": all(section_counts[k] == v for k, v in explicit_goal_metrics.items()),
            "all_evidence_paths_exist": not missing_evidence,
            "blocked_metrics_not_claimed_complete": all(r["status"] != "reported_exact_or_released" for r in rows if "blocked" in r["status"]),
            "released_tracking_metrics_use_numeric_summary": all(
                "res/tables/released_data_metrics_summary/released_tracking_ablation_metrics.csv" in r["evidence"]
                for r in rows
                if r["section"] == "motion_tracking"
                and r["metric"]
                in {
                    "local position error",
                    "local orientation error",
                    "local linear velocity error",
                    "local angular velocity error",
                    "global position error",
                    "global yaw error",
                }
            ),
            "statistics_three_seed_boundary_recorded": any(
                r["metric"] == "3 seeds"
                and r["status"] == "public_data_checkpoint"
                and "res/level_c/lafan1_paper_arch_multiseed_audit/level_c_lafan1_paper_arch_multiseed_audit.json"
                in r["evidence"]
                and "res/level_c/lafan1_paper_arch_symmetry_augmented_multiseed_audit/level_c_lafan1_paper_arch_symmetry_augmented_multiseed_audit.json"
                in r["evidence"]
                for r in rows
            ),
            "trial_failure_accounting_linked": all(
                "res/evaluation/trial_failure_accounting_audit/trial_failure_accounting_audit.json" in r["evidence"]
                for r in rows
                if r["section"] == "statistics" and r["metric"] in {"trial count", "failure count"}
            ),
            "goal_metric_formula_api_rows_linked": all(
                "res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json" in r["evidence"]
                and "res/tests/core_math_unit_tests/core_math_unit_tests.json" in r["evidence"]
                for r in rows
                if r["status"] == "formula_api_only"
            ),
            "formula_api_metrics_not_claimed_rollout_results": all(
                "missing" in r["detail"] or "blocked" in r["detail"]
                for r in rows
                if r["status"] == "formula_api_only"
            ),
            "public_data_checkpoint_rows_link_offline_metrics": all(
                "res/level_c/lafan1_paper_arch_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_audit.json"
                in r["evidence"]
                or "res/level_c/lafan1_paper_arch_onnx_latency/level_c_lafan1_paper_arch_onnx_latency_audit.json"
                in r["evidence"]
                or "res/level_c/lafan1_paper_arch_multiseed_audit/level_c_lafan1_paper_arch_multiseed_audit.json"
                in r["evidence"]
                or "res/level_c/lafan1_paper_arch_symmetry_augmented_multiseed_audit/level_c_lafan1_paper_arch_symmetry_augmented_multiseed_audit.json"
                in r["evidence"]
                or "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/level_c_lafan1_paper_arch_guidance_eval.json"
                in r["evidence"]
                or "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.json"
                in r["evidence"]
                for r in rows
                if r["status"] == "public_data_checkpoint"
            ),
            "diffusion_velocity_error_links_guidance_full_split_metrics": any(
                r["section"] == "diffusion_guidance"
                and r["metric"] == "velocity error"
                and r["status"] == "public_data_checkpoint"
                and "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/level_c_lafan1_paper_arch_guidance_eval.json"
                in r["evidence"]
                and "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.json"
                in r["evidence"]
                and "closed-loop" in r["detail"]
                and "unreproduced" in r["detail"]
                for r in rows
            ),
            "diffusion_goal_distance_links_guidance_full_split_metrics": any(
                r["section"] == "diffusion_guidance"
                and r["metric"] == "goal distance"
                and r["status"] == "public_data_checkpoint"
                and "res/level_c/guidance_task_metric_audit/level_c_guidance_task_metric_audit.json"
                in r["evidence"]
                and "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/level_c_lafan1_paper_arch_guidance_eval.json"
                in r["evidence"]
                and "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.json"
                in r["evidence"]
                and "no closed-loop" in r["detail"]
                for r in rows
            ),
            "does_not_claim_goal_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "All Section 12 metric names are explicitly mapped. Several VAE/diffusion metrics now have full "
                "paper-architecture public-data checkpoint evidence, including a 3-seed full-architecture public-data "
                "statistics audit, but closed-loop success/fall/collision, Fig.5/Fig.6 logs, TensorRT benchmarks, "
                "and hardware execution are still absent."
            ),
        },
        "outputs": {
            "json": str(OUT / "evaluation_metrics_coverage_audit.json"),
            "tsv": str(OUT / "evaluation_metrics_coverage_audit.tsv"),
        },
    }
    (OUT / "evaluation_metrics_coverage_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT / "evaluation_metrics_coverage_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=["section", "metric", "status", "evidence", "all_evidence_exists", "detail"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "section": r["section"],
                    "metric": r["metric"],
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
