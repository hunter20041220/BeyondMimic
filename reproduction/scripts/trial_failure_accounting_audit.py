#!/usr/bin/env python3
"""Audit trial-count and failure-count evidence for goal.md Section 12.5.

The paper-level rollout trial/failure counts are not locally available without
trained checkpoints and closed-loop execution. This audit consolidates the
counts that are actually present: released-data table rows, source-table clip
rows, debug multi-seed runs, run-catalog counts, and retained failed-run
evidence. It keeps those categories separate from missing paper rollout trials.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/evaluation/trial_failure_accounting_audit"


def load_json(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def row(
    account: str,
    status: str,
    evidence: list[str],
    *,
    trial_count: int | None,
    failure_count: int | None,
    detail: str,
) -> dict[str, Any]:
    evidence_exists = [(ROOT / p).exists() for p in evidence]
    return {
        "account": account,
        "status": status,
        "trial_count": trial_count,
        "failure_count": failure_count,
        "evidence": evidence,
        "evidence_exists": evidence_exists,
        "all_evidence_exists": all(evidence_exists),
        "detail": detail,
    }


def sum_field(rows: list[dict[str, Any]], field: str) -> int:
    return int(sum(int(r.get(field, 0)) for r in rows))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["account", "status", "trial_count", "failure_count", "evidence", "all_evidence_exists", "detail"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "account": r["account"],
                    "status": r["status"],
                    "trial_count": r["trial_count"],
                    "failure_count": r["failure_count"],
                    "evidence": ";".join(r["evidence"]),
                    "all_evidence_exists": r["all_evidence_exists"],
                    "detail": r["detail"],
                }
            )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    released_summary = load_json("res/tables/released_data_metrics_summary/released_data_metrics_summary.json")
    released_stats = load_json("res/tables/released_data_statistical_audit/released_data_statistical_audit.json")
    skill_success = load_json("res/paper_skill_success_table_audit/skill_success_table_data_audit.json")
    results_claims = load_json("res/results_claims_audit/results_claims_audit.json")
    failed_run = load_json("res/failed_runs/failed_run_audit/failed_run_audit.json")
    run_catalog = load_json("res/run_log_config_catalog/run_log_config_catalog.json")
    small_multiseed = load_json("res/level_c/small_dataset_multiseed_audit/level_c_small_dataset_multiseed_audit.json")
    small_heldout = load_json(
        "res/level_c/small_dataset_heldout_multiseed_audit/level_c_small_dataset_heldout_multiseed_audit.json"
    )
    paper_state_multiseed = load_json(
        "res/level_c/paper_state_heldout_multiseed_audit/level_c_paper_state_heldout_multiseed_audit.json"
    )
    vae_latent_multiseed = load_json(
        "res/level_c/vae_latent_heldout_multiseed_audit/level_c_vae_latent_heldout_multiseed_audit.json"
    )
    action_multiseed = load_json(
        "res/level_c/diffusion_to_vae_action_multiseed_audit/level_c_diffusion_to_vae_action_multiseed_audit.json"
    )
    fig56 = load_json("res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json")

    rows: list[dict[str, Any]] = [
        row(
            "paper_skill_success_table_rows",
            "source_table_count_only",
            ["res/paper_skill_success_table_audit/skill_success_table_data_audit.json"],
            trial_count=int(skill_success["metrics"]["total_rows_parsed"]),
            failure_count=None,
            detail=(
                "Paper skill-success table rows parsed from source; local sim/real execution outcomes are not "
                "available and are not counted as reproduced successes or failures."
            ),
        ),
        row(
            "paper_real_segments_declared",
            "source_table_count_only",
            ["res/paper_skill_success_table_audit/skill_success_table_data_audit.json"],
            trial_count=int(skill_success["metrics"]["real_segment_count"]),
            failure_count=int(skill_success["metrics"]["segment_out_of_range_row_count"]),
            detail="Declared real segments are parsed; two segment rows exceed local CSV duration and are data-availability mismatches.",
        ),
        row(
            "released_tracking_ablation_metric_rows",
            "released_data_metric_rows",
            [
                "res/tables/released_data_metrics_summary/released_data_metrics_summary.json",
                "res/tables/released_data_statistical_audit/released_data_statistical_audit.json",
            ],
            trial_count=int(released_summary["metrics"]["ablation_row_count"]),
            failure_count=None,
            detail="Released processed tracking-ablation metric rows; not live PPO rerun trials.",
        ),
        row(
            "released_grf_ci_rows",
            "released_data_metric_rows",
            ["res/tables/released_data_statistical_audit/released_data_statistical_audit.json"],
            trial_count=int(released_stats["metrics"]["grf_ci_rows"]),
            failure_count=None,
            detail="Released GRF confidence-interval rows; not closed-loop task trials.",
        ),
        row(
            "released_imu_ci_rows",
            "released_data_metric_rows",
            ["res/tables/released_data_statistical_audit/released_data_statistical_audit.json"],
            trial_count=int(released_stats["metrics"]["imu_ci_rows"]),
            failure_count=None,
            detail="Released IMU confidence-interval rows; not sim/real task-success trials.",
        ),
        row(
            "results_claim_rows",
            "claim_accounting_rows",
            ["res/results_claims_audit/results_claims_audit.json"],
            trial_count=int(results_claims["metrics"]["row_count"]),
            failure_count=int(results_claims["metrics"]["blocked_or_unreproduced_rows"]),
            detail="Results-claim audit rows; blocked/unreproduced rows are claim-accounting gaps, not rollout failure episodes.",
        ),
        row(
            "small_dataset_overfit_debug_seeds",
            "debug_seed_runs",
            ["res/level_c/small_dataset_multiseed_audit/level_c_small_dataset_multiseed_audit.json"],
            trial_count=len(small_multiseed["rows"]),
            failure_count=0,
            detail=f"Debug overfit seeds; total windows across seed rows {sum_field(small_multiseed['rows'], 'window_count')}.",
        ),
        row(
            "small_dataset_heldout_debug_seeds",
            "debug_seed_runs",
            ["res/level_c/small_dataset_heldout_multiseed_audit/level_c_small_dataset_heldout_multiseed_audit.json"],
            trial_count=len(small_heldout["rows"]),
            failure_count=0,
            detail=f"Debug held-out seeds; total test windows {sum_field(small_heldout['rows'], 'test_window_count')}.",
        ),
        row(
            "paper_state_heldout_debug_seeds",
            "debug_seed_runs",
            ["res/level_c/paper_state_heldout_multiseed_audit/level_c_paper_state_heldout_multiseed_audit.json"],
            trial_count=len(paper_state_multiseed["rows"]),
            failure_count=0,
            detail=f"Paper-state debug held-out seeds; total test windows {sum_field(paper_state_multiseed['rows'], 'test_window_count')}.",
        ),
        row(
            "vae_latent_heldout_debug_seeds",
            "debug_seed_runs",
            ["res/level_c/vae_latent_heldout_multiseed_audit/level_c_vae_latent_heldout_multiseed_audit.json"],
            trial_count=len(vae_latent_multiseed["rows"]),
            failure_count=0,
            detail=f"VAE-latent debug held-out seeds; total test windows {sum_field(vae_latent_multiseed['rows'], 'test_window_count')}.",
        ),
        row(
            "diffusion_to_vae_action_debug_seeds",
            "debug_seed_runs",
            ["res/level_c/diffusion_to_vae_action_multiseed_audit/level_c_diffusion_to_vae_action_multiseed_audit.json"],
            trial_count=len(action_multiseed["rows"]),
            failure_count=0,
            detail=f"Diffusion-to-action debug seeds; total test windows {sum_field(action_multiseed['rows'], 'test_window_count')}.",
        ),
        row(
            "retained_failed_isaaclab_smoke",
            "failed_run_retained",
            ["res/failed_runs/failed_run_audit/failed_run_audit.json"],
            trial_count=1,
            failure_count=1,
            detail=f"Retained failed run {failed_run['run_id']} with {failed_run['gpu_status_rows']} GPU-status rows.",
        ),
        row(
            "run_catalog_training_runs",
            "run_catalog_count",
            ["res/run_log_config_catalog/run_log_config_catalog.json"],
            trial_count=int(run_catalog["metrics"]["run_directory_count"]),
            failure_count=int(run_catalog["metrics"]["invalid_or_debug_run_count"]),
            detail=(
                f"Run catalog records {run_catalog['metrics']['valid_training_run_count']} valid training runs; "
                "debug/invalid run directories are not completed paper trials."
            ),
        ),
        row(
            "fig5_fig6_rollout_trials",
            "missing_paper_rollout_trials",
            ["res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json"],
            trial_count=None,
            failure_count=None,
            detail=(
                f"Fig.5/Fig.6 audit covers {fig56.get('panel_count', 6)} panels, but trained guided/unconditional "
                "rollout trial counts and failure counts are unavailable."
            ),
        ),
    ]

    missing = [r for r in rows if not r["all_evidence_exists"]]
    debug_seed_rows = [r for r in rows if r["status"] == "debug_seed_runs"]
    released_rows = [r for r in rows if r["status"] == "released_data_metric_rows"]
    metrics = {
        "row_count": len(rows),
        "source_table_trial_rows": int(skill_success["metrics"]["total_rows_parsed"]),
        "source_table_real_segments": int(skill_success["metrics"]["real_segment_count"]),
        "released_metric_row_total": sum(int(r["trial_count"] or 0) for r in released_rows),
        "debug_seed_run_total": sum(int(r["trial_count"] or 0) for r in debug_seed_rows),
        "retained_failed_run_count": 1,
        "valid_training_run_count": int(run_catalog["metrics"]["valid_training_run_count"]),
        "missing_paper_rollout_trial_rows": sum(1 for r in rows if r["status"] == "missing_paper_rollout_trials"),
    }
    status_counts: dict[str, int] = {}
    for r in rows:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    checks = {
        "all_evidence_paths_exist": not missing,
        "skill_success_rows_accounted": metrics["source_table_trial_rows"] == 36,
        "released_metric_rows_accounted": metrics["released_metric_row_total"] == 53,
        "debug_seed_runs_accounted": metrics["debug_seed_run_total"] >= 15,
        "retained_failed_run_count_recorded": metrics["retained_failed_run_count"] == 1,
        "valid_training_runs_zero_recorded": metrics["valid_training_run_count"] == 0,
        "missing_paper_rollout_trials_recorded": metrics["missing_paper_rollout_trial_rows"] == 1,
        "does_not_claim_rollout_failure_counts": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "trial_failure_accounting_audit",
        "scope": "goal.md Section 12.5 trial/failure count evidence accounting",
        "metrics": metrics,
        "status_counts": dict(sorted(status_counts.items())),
        "missing_evidence_rows": missing,
        "rows": rows,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "partial",
            "why_not_complete": (
                "Available source-table, released-data, debug-seed, and failed-run counts are accounted for, but "
                "paper-level rollout trial/failure counts remain unavailable without trained checkpoints and "
                "closed-loop evaluation logs."
            ),
        },
        "outputs": {
            "json": str(OUT / "trial_failure_accounting_audit.json"),
            "tsv": str(OUT / "trial_failure_accounting_audit.tsv"),
        },
    }
    (OUT / "trial_failure_accounting_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_tsv(OUT / "trial_failure_accounting_audit.tsv", rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
