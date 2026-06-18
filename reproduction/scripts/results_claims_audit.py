#!/usr/bin/env python3
"""Audit paper Results claims against current reproduction evidence."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/results_claims_audit"


def load_json(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def file_exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def released_imu_metrics() -> dict[str, float]:
    path = ROOT / "res/released_figures/imu_orientation_accel_angular_velocity/imu_orientation_accel_angular_velocity_processed.csv"
    max_acc = 0.0
    max_ang = 0.0
    sum_ang = 0.0
    n_ang = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            def val(key: str) -> float:
                try:
                    return float(row[key])
                except Exception:
                    return float("nan")

            acc = [val(key) for key in ["acc_x", "acc_y", "acc_z"]]
            ang = [val(key) for key in ["ang_x", "ang_y", "ang_z"]]
            if all(math.isfinite(x) for x in acc):
                max_acc = max(max_acc, math.sqrt(sum(x * x for x in acc)))
            if all(math.isfinite(x) for x in ang):
                norm = math.sqrt(sum(x * x for x in ang))
                max_ang = max(max_ang, norm)
                sum_ang += norm
                n_ang += 1
    return {
        "released_fig3b_max_linear_acceleration_norm": max_acc,
        "released_fig3b_max_angular_velocity_norm": max_ang,
        "released_fig3b_mean_angular_velocity_norm": sum_ang / n_ang,
        "released_fig3b_valid_angular_velocity_samples": float(n_ang),
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "claim_id",
        "paper_source",
        "paper_claim",
        "paper_value",
        "local_status",
        "evidence",
        "evidence_strength",
        "missing_for_paper_reproduction",
        "passed",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = row.copy()
            for key in ["evidence", "missing_for_paper_reproduction"]:
                if isinstance(out.get(key), list):
                    out[key] = "; ".join(out[key])
            writer.writerow({field: out.get(field, "") for field in fields})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig56 = load_json("res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json")
    official = load_json("res/level_c/official_artifact_audit/level_c_official_artifact_audit.json")
    skill = load_json("res/paper_skill_success_table_audit/skill_success_table_data_audit.json")
    released_panel = load_json("res/released_panel_mapping_audit/released_panel_mapping_audit.json")
    guidance = load_json("res/level_c/guidance_cost_coverage_audit/level_c_guidance_cost_coverage_audit.json")
    timestep = load_json("res/level_c/timestep_mask_coverage_audit/level_c_timestep_mask_coverage_audit.json")
    deployment = load_json("res/level_c/deployment_protocol_audit/level_c_deployment_protocol_audit.json")
    comparison = load_json("res/comparison/paper_vs_reproduction.json")
    metrics_coverage = load_json("res/evaluation_metrics_coverage/evaluation_metrics_coverage_audit.json")
    package_api_tests = load_json("res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json")
    core_math_tests = load_json("res/tests/core_math_unit_tests/core_math_unit_tests.json")
    released_stats = load_json("res/tables/released_data_statistical_audit/released_data_statistical_audit.json")
    imu_metrics = released_imu_metrics()
    imu_claim_metrics = released_stats["metrics"]["imu_paper_claim_comparison"]

    official_level_c_missing = not official["conclusion"]["official_beyondmimic_checkpoint_or_engine_found"]
    released_panel_ok = released_panel["checks"]["all_released_panel_rows_pass"]
    fig56_blocked = not fig56["conclusion"]["fig5_fig6_paper_reproduction_possible_from_current_local_artifacts"]

    common_level_c_missing = [
        "trained VAE/diffusion checkpoint",
        "closed-loop Level C rollout log",
        "official Fig.5/Fig.6 data or videos",
    ]
    hardware_missing = ["Unitree G1 hardware execution log/video", "real robot state/action telemetry"]
    formula_api_evidence = [
        "res/evaluation_metrics_coverage/evaluation_metrics_coverage_audit.json",
        "res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json",
        "res/tests/core_math_unit_tests/core_math_unit_tests.json",
    ]
    formula_api_checks_ok = (
        metrics_coverage["checks"]["goal_metric_formula_api_rows_linked"]
        and metrics_coverage["checks"]["formula_api_metrics_not_claimed_rollout_results"]
        and package_api_tests["checks"]["all_package_api_tests_pass"]
        and package_api_tests["checks"]["does_not_claim_training_or_deployment"]
        and core_math_tests["checks"]["all_core_math_tests_pass"]
        and core_math_tests["checks"]["does_not_claim_training_or_deployment"]
    )

    rows: list[dict[str, Any]] = [
        {
            "claim_id": "scalable_2_5h_all_sim_30_real_clips",
            "paper_source": "reproduction/paper/source/tex/results.tex:8-11; reproduction/paper/source/root.tex:472-864",
            "paper_claim": "About 2.5 hours of motions, all validated in simulation, 30 representative clips deployed on hardware.",
            "paper_value": "2.5h; 30 clips; 15 minutes",
            "local_status": "source_table_data_audit_only",
            "evidence": ["res/paper_skill_success_table_audit/skill_success_table_data_audit.json"],
            "evidence_strength": "table/source/data availability audit",
            "missing_for_paper_reproduction": ["sim execution success logs", *hardware_missing],
            "passed": skill["checks"]["paper_table_source_found"] and skill["checks"]["sim_real_success_not_claimed_reproduced"],
        },
        {
            "claim_id": "fig3b_imu_dynamics_released",
            "paper_source": "reproduction/paper/source/tex/results.tex:13-18; reproduction/paper/source/root.tex:206-212",
            "paper_claim": "Agile real-world motion has high acceleration and angular velocity; Fig.3B IMU curves are released.",
            "paper_value": "31 m/s^2 peak acceleration; 20 rad/s peak angular velocity; 7.01 rad/s mean",
            "local_status": "partial_released_data_reproduced",
            "evidence": [
                "res/released_figures/imu_orientation_accel_angular_velocity/imu_orientation_accel_angular_velocity_processed.csv",
                "res/released_panel_mapping_audit/released_panel_mapping_audit.json",
                "res/tables/released_data_statistical_audit/released_data_statistical_audit.json",
            ],
            "evidence_strength": "released Fig.3B curve reproduction plus statistical audit and paper-claim norm checks",
            "local_metrics": {**imu_metrics, "statistical_audit_claim_metrics": imu_claim_metrics},
            "missing_for_paper_reproduction": ["paper's exact highlighted interval mask for the reported scalar values"],
            "passed": released_panel_ok and file_exists(
                "res/released_figures/imu_orientation_accel_angular_velocity/imu_orientation_accel_angular_velocity_processed.csv"
            )
            and released_stats["checks"]["imu_norm_claim_metrics_present"],
        },
        {
            "claim_id": "fig4c_grf_released",
            "paper_source": "reproduction/paper/source/tex/results.tex:23-32; reproduction/paper/source/root.tex:215-219",
            "paper_claim": "Walking/running GRF profiles are human-like and released for Fig.4C.",
            "paper_value": "GRF shape comparison",
            "local_status": "partial_released_data_reproduced",
            "evidence": [
                "res/released_figures/grf_walk_human_reference/grf_walk_human_reference.pdf",
                "res/released_figures/grf_run_human_reference/grf_run_human_reference.pdf",
                "res/released_figures/grf_walk_robot_real/grf_walk_robot_real.pdf",
                "res/released_figures/grf_run_robot_real/grf_run_robot_real.pdf",
                "res/tables/released_data_statistical_audit/released_grf_confidence_intervals.csv",
            ],
            "evidence_strength": "released Fig.4C GRF component reproduction plus confidence intervals",
            "missing_for_paper_reproduction": ["video/user-study/recovery panels outside released GRF data"],
            "passed": released_panel_ok and released_stats["checks"]["grf_ci_rows_12"],
        },
        {
            "claim_id": "user_study_n77_preference",
            "paper_source": "reproduction/paper/source/tex/results.tex:28-29",
            "paper_claim": "User study prefers BeyondMimic over Unitree native controller.",
            "paper_value": "N=77; 70.8/29.2 overall; 57.0/43.0 walking; 84.7/15.3 running",
            "local_status": "paper_only_unreproduced",
            "evidence": ["reproduction/paper/source/tex/results.tex:28-29"],
            "evidence_strength": "paper source only",
            "missing_for_paper_reproduction": ["participant responses", "clip set", "statistical analysis notebook"],
            "passed": True,
        },
        {
            "claim_id": "fig5a_joystick_diffusion_process",
            "paper_source": "reproduction/paper/source/root.tex:227-229; reproduction/paper/source/tex/results.tex:34-38",
            "paper_claim": "Diffusion process under joystick guidance converges toward a right-turn command.",
            "paper_value": "Figure 5A qualitative process",
            "local_status": "debug_mechanics_only",
            "evidence": [
                "res/level_c/guidance_cost_coverage_audit/level_c_guidance_cost_coverage_audit.json",
                "res/level_c/guided_reverse_loop_probe/level_c_guided_reverse_loop_probe.json",
                "res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json",
            ],
            "evidence_strength": "formula/source plus debug reverse-loop mechanics",
            "missing_for_paper_reproduction": common_level_c_missing,
            "passed": guidance["checks"]["guided_reverse_loop_valid"] and fig56_blocked,
        },
        {
            "claim_id": "fig5b_waypoint_navigation",
            "paper_source": "reproduction/paper/source/root.tex:229; reproduction/paper/source/tex/results.tex:40-42",
            "paper_claim": "Robot reaches waypoint goals from multiple starts using forward/backward walking.",
            "paper_value": "Figure 5B qualitative result",
            "local_status": "formula_debug_only",
            "evidence": ["res/level_c/guidance_cost_coverage_audit/level_c_guidance_cost_coverage_audit.json"],
            "evidence_strength": "waypoint cost formula and debug gradients",
            "missing_for_paper_reproduction": [*common_level_c_missing, "waypoint rollout logs"],
            "passed": guidance["checks"]["all_paper_explicit_costs_have_source_and_gradients"],
        },
        {
            "claim_id": "fig5c_joystick_disturbance_recovery",
            "paper_source": "reproduction/paper/source/root.tex:230; reproduction/paper/source/tex/results.tex:42",
            "paper_claim": "Joystick teleoperation tracks commands and recovers from impulsive disturbance.",
            "paper_value": "Figure 5C qualitative result",
            "local_status": "blocked_closed_loop_required",
            "evidence": ["res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json"],
            "evidence_strength": "blocked-feasibility audit",
            "missing_for_paper_reproduction": [*common_level_c_missing, "disturbance/recovery execution log"],
            "passed": fig56_blocked,
        },
        {
            "claim_id": "velocity_tracking_errors",
            "paper_source": "reproduction/paper/source/tex/results.tex:42",
            "paper_claim": "Simulation walking/running command tracking errors.",
            "paper_value": "12.14% walking; 13.65% running",
            "local_status": "not_publicly_reproducible_currently",
            "evidence": ["res/comparison/paper_vs_reproduction.json", *formula_api_evidence],
            "evidence_strength": "comparison marks paper metric unavailable; formula/API metric contracts are tested",
            "missing_for_paper_reproduction": ["trained controller simulation evaluation logs"],
            "passed": not comparison["missing_goal_checkpoint_rows"] and formula_api_checks_ok,
        },
        {
            "claim_id": "fig5d_tsne_latent_transition",
            "paper_source": "reproduction/paper/source/root.tex:231; reproduction/paper/source/tex/results.tex:44",
            "paper_claim": "t-SNE latent space illustrates walking-to-running transition.",
            "paper_value": "Figure 5D t-SNE",
            "local_status": "debug_latent_probe_only",
            "evidence": ["res/level_c/vae_latent_probe/level_c_vae_latent_probe.json"],
            "evidence_strength": "VAE latent math/probe only, no trained embedding",
            "missing_for_paper_reproduction": ["trained VAE encoder", "latent samples", "t-SNE embedding data"],
            "passed": official_level_c_missing,
        },
        {
            "claim_id": "fig5e_real_walk_to_run_transition",
            "paper_source": "reproduction/paper/source/root.tex:232; reproduction/paper/source/tex/results.tex:44",
            "paper_claim": "Real-world walking-to-running transition conditioned on velocity command.",
            "paper_value": "Figure 5E real-world transition",
            "local_status": "requires_real_robot",
            "evidence": ["res/blocked_gates/blocked_gate_audit.json"],
            "evidence_strength": "hardware gate",
            "missing_for_paper_reproduction": hardware_missing,
            "passed": True,
        },
        {
            "claim_id": "fig6a_keyframe_cartwheel_inpainting",
            "paper_source": "reproduction/paper/source/root.tex:240-242; reproduction/paper/source/tex/results.tex:46-54",
            "paper_claim": "Future keyframes at 0.2 s intervals drive cartwheel inpainting and multi-round transitions.",
            "paper_value": "0.2 s keyframe interval; long-horizon cartwheel/task switching",
            "local_status": "mask_reverse_debug_only",
            "evidence": [
                "res/level_c/timestep_mask_coverage_audit/level_c_timestep_mask_coverage_audit.json",
                "res/level_c/paper_state_mask_reverse_probe/level_c_paper_state_mask_reverse_probe.json",
            ],
            "evidence_strength": "mask/reverse mechanics only",
            "missing_for_paper_reproduction": [*common_level_c_missing, "cartwheel keyframe rollout logs"],
            "passed": timestep["checks"]["paper_state_mask_reverse_debug_artifact_present"],
        },
        {
            "claim_id": "fig6b_obstacle_avoidance",
            "paper_source": "reproduction/paper/source/root.tex:242; reproduction/paper/source/tex/results.tex:56-59",
            "paper_claim": "Waypoint and SDF costs steer real-world obstacle avoidance to the target.",
            "paper_value": "Figure 6B qualitative result",
            "local_status": "formula_debug_only_requires_mocap",
            "evidence": [
                "res/level_c/guidance_cost_coverage_audit/level_c_guidance_cost_coverage_audit.json",
                "res/level_c/deployment_protocol_audit/level_c_deployment_protocol_audit.json",
            ],
            "evidence_strength": "waypoint/SDF formulas plus deployment boundary audit",
            "missing_for_paper_reproduction": [*common_level_c_missing, "mocap/environment context", *hardware_missing],
            "passed": guidance["checks"]["all_paper_explicit_costs_have_source_and_gradients"]
            and deployment["checks"]["does_not_claim_deployment_reproduction"],
        },
        {
            "claim_id": "latent_diffusion_cartwheel_ablation",
            "paper_source": "reproduction/paper/source/tex/method.tex:249-252",
            "paper_claim": "Latent diffusion improves cartwheel sim-to-sim success over direct diffusion.",
            "paper_value": "5% direct diffusion; 95% latent diffusion",
            "local_status": "not_publicly_reproducible_currently",
            "evidence": [
                "res/comparison/paper_vs_reproduction.json",
                *formula_api_evidence,
                "res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json",
            ],
            "evidence_strength": "comparison marks paper ablation unavailable; success/fall metric APIs are tested",
            "missing_for_paper_reproduction": ["MuJoCo sim-to-sim setup", "trained direct/latent diffusion checkpoints"],
            "passed": not comparison["missing_goal_checkpoint_rows"] and formula_api_checks_ok and fig56_blocked,
        },
        {
            "claim_id": "official_level_c_artifacts_absent",
            "paper_source": "all Level C results",
            "paper_claim": "Paper Level C result reproduction requires official or reproduced trained VAE/diffusion artifacts.",
            "paper_value": "trained code/checkpoint/engine required",
            "local_status": "blocking_boundary_recorded",
            "evidence": ["res/level_c/official_artifact_audit/level_c_official_artifact_audit.json"],
            "evidence_strength": "artifact absence audit",
            "missing_for_paper_reproduction": common_level_c_missing,
            "passed": official_level_c_missing,
        },
    ]

    failed = [row for row in rows if not row["passed"]]
    status_counts = Counter(row["local_status"] for row in rows)
    evidence_counts = Counter(row["evidence_strength"] for row in rows)
    paper_metric_claim_ids = {"velocity_tracking_errors", "latent_diffusion_cartwheel_ablation"}
    paper_metric_claim_rows = [row for row in rows if row["claim_id"] in paper_metric_claim_ids]
    formula_api_paths = set(formula_api_evidence)
    formula_api_linked_claim_rows = [
        row for row in paper_metric_claim_rows if formula_api_paths.issubset(set(row["evidence"]))
    ]
    formula_api_not_overclaimed = all(
        row["local_status"] == "not_publicly_reproducible_currently"
        and any(
            term in " ".join(row["missing_for_paper_reproduction"]).lower()
            for term in ["evaluation logs", "rollout", "mujoco", "trained direct/latent diffusion checkpoints"]
        )
        for row in paper_metric_claim_rows
    )

    summary: dict[str, Any] = {
        "status": "ok" if not failed else "failed",
        "experiment_type": "results_claims_audit",
        "scope": "source-indexed audit of paper Results claims against released data, debug probes, and blocked gates",
        "metrics": {
            "row_count": len(rows),
            "failed_row_count": len(failed),
            "released_or_partial_reproduced_rows": sum(
                1 for row in rows if "released" in row["local_status"] or row["local_status"].startswith("partial")
            ),
            "debug_only_rows": sum(1 for row in rows if "debug" in row["local_status"]),
            "blocked_or_unreproduced_rows": sum(
                1
                for row in rows
                if any(term in row["local_status"] for term in ["blocked", "requires_real_robot", "not_publicly"])
            ),
            "formula_api_linked_paper_metric_claim_rows": len(formula_api_linked_claim_rows),
            "paper_metric_claim_rows_still_unreproduced": sum(
                1 for row in paper_metric_claim_rows if row["local_status"] == "not_publicly_reproducible_currently"
            ),
            **imu_metrics,
        },
        "local_status_counts": dict(sorted(status_counts.items())),
        "evidence_strength_counts": dict(sorted(evidence_counts.items())),
        "checks": {
            "all_rows_pass": not failed,
            "fig5_fig6_blocked_boundary_reused": fig56_blocked,
            "released_panel_mapping_passes": released_panel_ok,
            "released_statistical_audit_passes": released_stats["status"] == "ok",
            "skill_success_execution_not_overclaimed": skill["checks"]["sim_real_success_not_claimed_reproduced"],
            "official_level_c_artifacts_absent_recorded": official_level_c_missing,
            "goal_checkpoint_rows_present_in_comparison": not comparison["missing_goal_checkpoint_rows"],
            "paper_metric_claims_have_formula_api_evidence": len(formula_api_linked_claim_rows) == len(paper_metric_claim_ids),
            "formula_api_evidence_not_overclaimed_as_paper_results": formula_api_not_overclaimed,
            "does_not_claim_fig5_fig6_reproduction": True,
        },
        "rows": rows,
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "partial",
            "why_not_complete": (
                "Results claims are now source-indexed against released-data reproductions and debug probes, but most "
                "Level C/Fig.5/Fig.6 claims still require trained VAE/diffusion checkpoints, closed-loop logs, MuJoCo "
                "or hardware execution, and/or released task data."
            ),
        },
        "outputs": {
            "json": str(OUT / "results_claims_audit.json"),
            "tsv": str(OUT / "results_claims_audit.tsv"),
        },
    }
    (OUT / "results_claims_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(OUT / "results_claims_audit.tsv", rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"]}, sort_keys=True))


if __name__ == "__main__":
    main()
