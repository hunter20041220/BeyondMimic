#!/usr/bin/env python3
"""Catalog current metric artifacts by evidence level and paper-reproduction boundary."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/metrics/metrics_catalog"


SOURCES: list[dict[str, str]] = [
    {
        "metric_group": "released_tracking_ablation",
        "evidence_level": "released_data",
        "path": "res/tables/released_data_metrics_summary/released_tracking_ablation_metrics.csv",
        "description": "Numeric local/global tracking-error summary from released ablation processed CSVs.",
    },
    {
        "metric_group": "released_grf",
        "evidence_level": "released_data",
        "path": "res/tables/released_data_metrics_summary/released_grf_metrics.csv",
        "description": "Walking/running GRF axis summary from released Fig.4C processed CSVs.",
    },
    {
        "metric_group": "released_imu",
        "evidence_level": "released_data",
        "path": "res/tables/released_data_metrics_summary/released_imu_metrics.csv",
        "description": "IMU orientation, acceleration, angular-velocity summary from released Fig.3B processed CSV.",
    },
    {
        "metric_group": "released_data_statistical_audit",
        "evidence_level": "released_data",
        "path": "res/tables/released_data_statistical_audit/released_data_statistical_audit.json",
        "description": "Confidence intervals, effect-size summaries, and paper-claim checks for released-data figures.",
    },
    {
        "metric_group": "paper_comparison",
        "evidence_level": "comparison",
        "path": "res/comparison/paper_vs_reproduction.csv",
        "description": "Paper-vs-current reproduction comparison table with exact/approximate/blocked classifications.",
    },
    {
        "metric_group": "section12_coverage",
        "evidence_level": "coverage_audit",
        "path": "res/evaluation_metrics_coverage/evaluation_metrics_coverage_audit.json",
        "description": "Goal Section 12 metric coverage and explicit missing/debug-only boundary status.",
    },
    {
        "metric_group": "results_claims",
        "evidence_level": "coverage_audit",
        "path": "res/results_claims_audit/results_claims_audit.json",
        "description": "Results-claim source index separating released data, debug mechanisms, paper-only claims, and blockers.",
    },
    {
        "metric_group": "trial_failure_accounting",
        "evidence_level": "coverage_audit",
        "path": "res/evaluation/trial_failure_accounting_audit/trial_failure_accounting_audit.json",
        "description": "Section 12.5 trial/failure-count accounting across source tables, released data, debug seeds, and failed runs.",
    },
    {
        "metric_group": "goal_metric_api_contracts",
        "evidence_level": "formula_api",
        "path": "res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json",
        "description": "Package API tests for goal-level success, fall, velocity-tracking, and split metric formula contracts.",
    },
    {
        "metric_group": "goal_metric_core_math",
        "evidence_level": "formula_api",
        "path": "res/tests/core_math_unit_tests/core_math_unit_tests.json",
        "description": "Pure-NumPy core math tests for success/fall/velocity-tracking metric formulas and related metric helpers.",
    },
    {
        "metric_group": "small_dataset_multiseed",
        "evidence_level": "debug_only",
        "path": "res/level_c/small_dataset_multiseed_audit/level_c_small_dataset_multiseed_audit.json",
        "description": "Three-seed debug small-dataset memorization statistics.",
    },
    {
        "metric_group": "small_dataset_heldout_multiseed",
        "evidence_level": "debug_only",
        "path": "res/level_c/small_dataset_heldout_multiseed_audit/level_c_small_dataset_heldout_multiseed_audit.json",
        "description": "Three-seed debug small-dataset held-out baseline statistics.",
    },
    {
        "metric_group": "paper_state_heldout_multiseed",
        "evidence_level": "debug_only",
        "path": "res/level_c/paper_state_heldout_multiseed_audit/level_c_paper_state_heldout_multiseed_audit.json",
        "description": "Three-seed paper-state debug held-out baseline statistics.",
    },
    {
        "metric_group": "vae_latent_heldout_multiseed",
        "evidence_level": "debug_only",
        "path": "res/level_c/vae_latent_heldout_multiseed_audit/level_c_vae_latent_heldout_multiseed_audit.json",
        "description": "Three-seed debug held-out baseline using nonzero tiny-VAE latents.",
    },
    {
        "metric_group": "guidance_task_scale_sweep",
        "evidence_level": "debug_only",
        "path": "res/level_c/guidance_task_scale_sweep/level_c_guidance_task_scale_sweep.json",
        "description": "Formula-level guidance task scale sweep metrics over five task costs.",
    },
    {
        "metric_group": "guidance_task_metric_audit",
        "evidence_level": "debug_only",
        "path": "res/level_c/guidance_task_metric_audit/level_c_guidance_task_metric_audit.json",
        "description": "Debug task-level guidance primary metrics linked to formula scale-sweep summaries.",
    },
    {
        "metric_group": "smoothness_latency",
        "evidence_level": "debug_only",
        "path": "res/level_c/smoothness_latency_audit/level_c_smoothness_latency_audit.json",
        "description": "Debug smoothness and latency-budget metrics.",
    },
    {
        "metric_group": "diffusion_to_vae_action_smoothness",
        "evidence_level": "debug_only",
        "path": "res/level_c/diffusion_to_vae_action_smoothness_audit/level_c_diffusion_to_vae_action_smoothness_audit.json",
        "description": "Debug downstream diffusion-to-VAE action-sequence smoothness metrics at 25 Hz.",
    },
    {
        "metric_group": "direct_vs_latent_action_ablation",
        "evidence_level": "debug_only",
        "path": "res/level_c/direct_vs_latent_action_ablation_audit/level_c_direct_vs_latent_action_ablation_audit.json",
        "description": "Offline debug direct-state vs state-latent action-pipe ablation metrics.",
    },
    {
        "metric_group": "state_latent_dataset_consistency",
        "evidence_level": "debug_only",
        "path": "res/level_c/state_latent_dataset_consistency_audit/level_c_state_latent_dataset_consistency_audit.json",
        "description": "Cross-artifact consistency metrics for paper-state windows, debug VAE latents, and action NPZ ordering.",
    },
    {
        "metric_group": "reimpl_runtime_integration",
        "evidence_level": "debug_only",
        "path": "res/code/reimpl_runtime_integration_audit/reimpl_runtime_integration_audit.json",
        "description": "Runtime integration metrics for package APIs over local debug fixtures and Level-C outputs.",
    },
    {
        "metric_group": "fig5_fig6_boundary",
        "evidence_level": "blocked_boundary",
        "path": "res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json",
        "description": "Panel-level feasibility audit for missing Fig.5/Fig.6 rollout artifacts.",
    },
    {
        "metric_group": "required_artifact_absence",
        "evidence_level": "blocked_boundary",
        "path": "res/required_artifact_absence/required_artifact_absence_audit.json",
        "description": "Absence audit for trained checkpoints, TensorRT engines, videos, and deployment outputs.",
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows(path: Path) -> int | None:
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return max(sum(1 for _ in f) - 1, 0)
    if path.suffix == ".tsv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return max(sum(1 for _ in f) - 1, 0)
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in ["row_count", "artifact_count"]:
            if isinstance(data.get(key), int):
                return int(data[key])
        metrics = data.get("metrics")
        if isinstance(metrics, dict) and isinstance(metrics.get("row_count"), int):
            return int(metrics["row_count"])
        rows = data.get("rows")
        if isinstance(rows, list):
            return len(rows)
    return None


def compact_metric_preview(path: Path) -> dict[str, Any]:
    if path.suffix != ".json":
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    preview: dict[str, Any] = {}
    for key in ["status", "row_count", "status_counts", "section_counts", "metrics", "checks"]:
        if key in data:
            value = data[key]
            if key == "checks" and isinstance(value, dict):
                preview[key] = {k: value[k] for k in sorted(value)[:8]}
            elif key == "metrics" and isinstance(value, dict):
                preview[key] = {k: value[k] for k in sorted(value)[:10]}
            else:
                preview[key] = value
    return preview


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "metric_group",
        "evidence_level",
        "relative_path",
        "absolute_path",
        "size_bytes",
        "sha256",
        "row_count",
        "description",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Metrics catalog",
        "",
        "This catalog indexes current metric-bearing artifacts by evidence level.",
        "It does not upgrade debug or blocked evidence into paper-level reproduction.",
        "",
        f"- Status: `{summary['status']}`",
        f"- Source count: `{summary['metrics']['source_count']}`",
        f"- Released-data source count: `{summary['metrics']['released_data_source_count']}`",
        f"- Formula-API source count: `{summary['metrics']['formula_api_source_count']}`",
        f"- Debug-only source count: `{summary['metrics']['debug_only_source_count']}`",
        f"- Blocked-boundary source count: `{summary['metrics']['blocked_boundary_source_count']}`",
        f"- Total indexed rows: `{summary['metrics']['total_indexed_rows']}`",
        "",
        "| Metric group | Evidence level | Rows | Source |",
        "|---|---:|---:|---|",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| `{row['metric_group']}` | `{row['evidence_level']}` | `{row['row_count']}` | `{row['relative_path']}` |"
        )
    lines.extend(
        [
            "",
            "Outputs:",
            f"- `{summary['outputs']['json']}`",
            f"- `{summary['outputs']['csv']}`",
            f"- `{summary['outputs']['markdown']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    previews: dict[str, Any] = {}
    for source in SOURCES:
        path = ROOT / source["path"]
        rows.append(
            {
                "metric_group": source["metric_group"],
                "evidence_level": source["evidence_level"],
                "relative_path": source["path"],
                "absolute_path": str(path),
                "size_bytes": path.stat().st_size if path.is_file() else 0,
                "sha256": sha256_file(path) if path.is_file() else "",
                "row_count": count_rows(path) if path.is_file() else None,
                "description": source["description"],
            }
        )
        if path.is_file():
            previews[source["metric_group"]] = compact_metric_preview(path)

    missing = [row for row in rows if not Path(row["absolute_path"]).is_file()]
    level_counts: dict[str, int] = {}
    for row in rows:
        level_counts[row["evidence_level"]] = level_counts.get(row["evidence_level"], 0) + 1
    total_rows = sum(int(row["row_count"] or 0) for row in rows)
    checks = {
        "all_metric_sources_exist": not missing,
        "all_metric_sources_hashed": all(bool(row["sha256"]) for row in rows),
        "released_data_sources_present": level_counts.get("released_data", 0) >= 3,
        "debug_only_sources_present": level_counts.get("debug_only", 0) >= 5,
        "formula_api_sources_present": level_counts.get("formula_api", 0) >= 2,
        "blocked_boundary_sources_present": level_counts.get("blocked_boundary", 0) >= 2,
        "comparison_source_present": level_counts.get("comparison", 0) >= 1,
        "coverage_sources_present": level_counts.get("coverage_audit", 0) >= 2,
        "does_not_claim_training": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "metrics_catalog",
        "scope": "current metric-bearing artifacts indexed by reproduction evidence level",
        "metrics": {
            "source_count": len(rows),
            "released_data_source_count": level_counts.get("released_data", 0),
            "formula_api_source_count": level_counts.get("formula_api", 0),
            "debug_only_source_count": level_counts.get("debug_only", 0),
            "blocked_boundary_source_count": level_counts.get("blocked_boundary", 0),
            "comparison_source_count": level_counts.get("comparison", 0),
            "coverage_audit_source_count": level_counts.get("coverage_audit", 0),
            "total_indexed_rows": total_rows,
        },
        "level_counts": dict(sorted(level_counts.items())),
        "missing_sources": missing,
        "rows": rows,
        "previews": previews,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The catalog makes current metrics easier to audit, but it still indexes many debug-only or blocked "
                "boundary artifacts. It does not create trained checkpoints, rollout videos, or paper-level results."
            ),
        },
        "outputs": {
            "json": str(OUT / "metrics_catalog.json"),
            "csv": str(OUT / "metrics_catalog.csv"),
            "markdown": str(OUT / "metrics_catalog.md"),
        },
    }
    (OUT / "metrics_catalog.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(OUT / "metrics_catalog.csv", rows)
    write_markdown(OUT / "metrics_catalog.md", summary)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "sources": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
