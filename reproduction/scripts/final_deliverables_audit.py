#!/usr/bin/env python3
"""Audit goal.md section 16 final deliverables across environment, code, experiment, and docs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/final_deliverables_audit"


def exists(rel: str) -> bool:
    path = ROOT / rel
    return path.exists() and (path.is_dir() or path.stat().st_size > 0)


def load_json(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def count_glob(patterns: list[str]) -> int:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(p for p in ROOT.glob(pattern) if p.is_file())
    return len(paths)


def sample_glob(patterns: list[str], limit: int = 8) -> list[str]:
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(p for p in ROOT.glob(pattern) if p.is_file())
    return [str(p.relative_to(ROOT)) for p in sorted(paths)[:limit]]


def row(
    category: str,
    item: str,
    status: str,
    evidence: list[str],
    remaining_gap: str,
    counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    evidence_exists = [exists(path) for path in evidence]
    return {
        "category": category,
        "item": item,
        "status": status,
        "evidence": evidence,
        "evidence_exists": evidence_exists,
        "all_listed_evidence_exists": all(evidence_exists) if evidence else False,
        "remaining_gap": remaining_gap,
        "counts": counts or {},
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    visual_inventory = load_json("res/visual_media_inventory/visual_media_inventory_audit.json")
    visual_kind_counts = visual_inventory.get("kind_counts", {})
    visual_category_counts = visual_inventory.get("category_counts", {})
    visual_checks = visual_inventory.get("checks", {})
    pdf_count = int(visual_kind_counts.get("pdf", count_glob(["res/**/*.pdf"])))
    svg_count = int(visual_kind_counts.get("svg", count_glob(["res/**/*.svg"])))
    png_count = int(visual_kind_counts.get("png", count_glob(["res/**/*.png"])))
    gif_count = int(visual_kind_counts.get("gif", count_glob(["res/**/*.gif"])))
    mp4_count = count_glob(["res/**/*.mp4", "res/**/*.mov", "res/**/*.mkv"])
    checkpoint_paths = list(ROOT.glob("res/**/*.pt")) + list(ROOT.glob("res/**/*.pth")) + list(ROOT.glob("res/**/*.ckpt")) + list(ROOT.glob("res/**/*.onnx"))
    debug_checkpoint_paths = [
        path
        for path in checkpoint_paths
        if "vae_checkpoint_smoke" in str(path)
        or "diffusion_checkpoint_smoke" in str(path)
        or "resource_adjusted_tiny_diffusion" in str(path)
        or "debug_motion_policy_onnx_export" in str(path)
    ]
    checkpoint_count = len([path for path in checkpoint_paths if path not in debug_checkpoint_paths])
    metric_count = count_glob(["res/**/*.json", "res/**/*.csv", "res/**/*.tsv"])
    failed_run_count = len([p for p in (ROOT / "res/failed_runs").glob("*") if p.is_dir() and p.name != "failed_run_audit"])

    rows = [
        row(
            "environment",
            "environment.yml",
            "complete",
            [
                "envs/bm_analysis/environment.yml",
                "envs/bm_tracking/environment.yml",
                "envs/bm_diffusion/environment.yml",
            ],
            "IsaacLab/Kit live smoke remains blocked separately.",
        ),
        row(
            "environment",
            "requirements-lock.txt",
            "complete",
            [
                "envs/bm_analysis/requirements-lock.txt",
                "envs/bm_tracking/requirements-lock.txt",
                "envs/bm_diffusion/requirements-lock.txt",
            ],
            "Locks prove environment capture, not successful long training.",
        ),
        row(
            "environment",
            "pip-freeze.txt",
            "complete",
            ["envs/bm_analysis/pip-freeze.txt", "envs/bm_tracking/pip-freeze.txt", "envs/bm_diffusion/pip-freeze.txt"],
            "Full deployment stack remains blocked.",
        ),
        row(
            "environment",
            "conda-list-explicit.txt",
            "complete",
            [
                "envs/bm_analysis/conda-list-explicit.txt",
                "envs/bm_tracking/conda-list-explicit.txt",
                "envs/bm_diffusion/conda-list-explicit.txt",
            ],
            "Environment export does not prove Isaac/ROS runtime gates.",
        ),
        row("environment", "docs/environment.md", "complete", ["reproduction/docs/environment.md"], "None for document scope."),
        row(
            "code",
            "official worktrees",
            "complete_for_local_copies",
            [
                "reproduction/third_party/official/whole_body_tracking",
                "reproduction/third_party/official/motion_tracking_controller",
                "reproduction/third_party/official/IsaacLab-v2.1.0",
            ],
            "Live execution of official worktrees remains blocked by Kit/inotify and ROS/OS gates.",
        ),
        row(
            "code",
            "patches",
            "partial",
            [
                "reproduction/patches",
                "res/code/patch_inventory_audit/patch_inventory_audit.json",
                "res/code/patch_snapshot_audit/patch_snapshot_audit.json",
                "res/tracking/motion_preprocessing_contract_audit/motion_preprocessing_contract_audit.json",
            ],
            "Patch inventory and tracked-diff snapshots are audited, but no explicit full training/deployment patch series exists.",
        ),
        row("code", "VAE", "partial", ["reproduction/src/beyondmimic_reimpl/vae/latent.py", "res/level_c/vae_contract_audit/level_c_vae_contract_audit.json"], "No trained VAE checkpoint or true DAgger rollout."),
        row("code", "DAgger", "partial", ["reproduction/src/beyondmimic_reimpl/dagger", "res/level_c/dataset_collection_protocol_audit/level_c_dataset_collection_protocol_audit.json"], "Protocol/debug evidence only; true teacher rollouts missing."),
        row("code", "trajectory dataset", "partial", ["reproduction/src/beyondmimic_reimpl/trajectory", "res/level_c/paper_state_windows/level_c_paper_state_windows.json"], "Paper-scale state-latent rollout dataset with learned VAE latents missing."),
        row("code", "diffusion", "partial", ["reproduction/src/beyondmimic_reimpl/diffusion/schedules.py", "res/level_c/diffusion_equation_audit/level_c_diffusion_equation_audit.json"], "No trained diffusion Transformer checkpoint."),
        row("code", "guidance", "partial", ["reproduction/src/beyondmimic_reimpl/guidance/costs.py", "res/level_c/guidance_cost_coverage_audit/level_c_guidance_cost_coverage_audit.json", "res/level_c/guidance_debug_visualization/level_c_guidance_debug_visualization.json"], "Guided rollout and required paper success/failure videos remain missing; debug formula visualization exists."),
        row("code", "evaluation", "partial", ["reproduction/src/beyondmimic_reimpl/evaluation", "res/comparison/paper_vs_reproduction.json"], "Paper-level closed-loop evaluation not complete."),
        row("code", "plotting", "complete_for_released_data", ["reproduction/scripts/reproduce_released_figures.py", "res/released_figures/released_figure_summary.tsv"], "Plotting covers released-data figures, not Fig. 5/Fig. 6."),
        row(
            "code",
            "tests",
            "complete_for_core_math",
            [
                "reproduction/tests/test_core_math.py",
                "res/tests/core_math_unit_tests/core_math_unit_tests.json",
                "res/tests/core_test_coverage_audit/core_test_coverage_audit.json",
                "reproduction/tests/test_reimpl_package_api.py",
                "res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json",
            ],
            "Formula, package API, and core checklist coverage tests exist; no end-to-end Isaac/ROS/TensorRT test suite.",
        ),
        row(
            "experiment",
            "raw logs",
            "partial",
            [
                "res/run_log_config_catalog/run_log_config_catalog.json",
                "logs/gpu/gpu_metrics.csv",
                "logs/data/plot_adaptive_sampling_released.log",
            ],
            "Run/log/config catalog indexes setup/data/debug/failure logs, but long training stdout/stderr logs are absent.",
        ),
        row(
            "experiment",
            "resolved configs",
            "partial",
            [
                "res/run_log_config_catalog/run_log_config_catalog.json",
                "res/config/resolved_reproduction_config.yaml",
                "res/runs/setup_run_management_diagnostic_static_000_20260617_050000/resolved_config.yaml",
            ],
            "Run/log/config catalog indexes resolved configs, but no completed training run config exists.",
        ),
        row(
            "experiment",
            "checkpoints",
            "blocked_or_missing",
            [
                "res/required_artifact_absence/required_artifact_absence_audit.json",
                "res/failed_runs/phase1_isaaclab_headless_smoke_g1_inotify_0_20260616_200654/checkpoint.txt",
                "res/tracking/debug_motion_policy_onnx_export/tracking_debug_motion_policy_onnx_export.json",
            ],
            "No real trained checkpoint/ONNX/TensorRT artifact is available; debug smoke checkpoints and the debug contract ONNX are counted separately.",
            {"model_file_count": checkpoint_count, "debug_checkpoint_file_count": len(debug_checkpoint_paths)},
        ),
        row(
            "experiment",
            "metrics",
            "partial",
            [
                "res/metrics/metrics_catalog/metrics_catalog.json",
                "res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json",
                "res/tests/core_math_unit_tests/core_math_unit_tests.json",
                "res/tests/core_test_coverage_audit/core_test_coverage_audit.json",
                "res/paper_formula_code_trace/paper_formula_code_trace_audit.json",
                "res/comparison/paper_vs_reproduction.csv",
                "res/results_claims_audit/results_claims_audit.json",
            ],
            "Metrics catalog now indexes released-data, formula-API, debug-only, comparison, coverage, and blocked-boundary metric artifacts; paper-scale training/evaluation metrics remain missing.",
            {"metric_file_count": metric_count},
        ),
        row(
            "experiment",
            "videos",
            "blocked_or_missing",
            [
                "res/videos",
                "res/runs/setup_run_management_diagnostic_static_000_20260617_050000/videos",
                "res/visual_media_inventory/visual_media_inventory_audit.json",
            ],
            "Visual media inventory records missing tracking rollout/replay, Fig. 5, Fig. 6, and real-robot videos; no reproduced simulation or real robot mp4/mov/mkv exists.",
            {"video_file_count": mp4_count, "debug_gif_preview_count": gif_count},
        ),
        row(
            "experiment",
            "PDF figures",
            "complete_for_released_and_debug",
            ["res/visual_media_inventory/visual_media_inventory_audit.json"] + sample_glob(["res/**/*.pdf"], 4),
            "Paper Fig. 5/Fig. 6 reproduced figures missing.",
            {"pdf_count": pdf_count, "released_data_figure_count": int(visual_category_counts.get("released_data_figure", 0))},
        ),
        row(
            "experiment",
            "SVG figures",
            "complete_for_released_and_debug",
            ["res/visual_media_inventory/visual_media_inventory_audit.json"] + sample_glob(["res/**/*.svg"], 4),
            "Paper Fig. 5/Fig. 6 reproduced figures missing.",
            {"svg_count": svg_count, "released_data_figure_count": int(visual_category_counts.get("released_data_figure", 0))},
        ),
        row(
            "experiment",
            "PNG previews",
            "complete_for_released_and_debug",
            ["res/visual_media_inventory/visual_media_inventory_audit.json"] + sample_glob(["res/**/*.png"], 4),
            "Paper Fig. 5/Fig. 6 reproduced previews missing.",
            {
                "png_count": png_count,
                "debug_visual_count": int(visual_inventory.get("row_count", 0))
                - int(visual_category_counts.get("released_data_figure", 0)),
            },
        ),
        row("experiment", "multi-seed statistics", "partial", ["res/level_c/small_dataset_multiseed_audit/level_c_small_dataset_multiseed_audit.json", "res/level_c/paper_state_heldout_multiseed_audit/level_c_paper_state_heldout_multiseed_audit.json"], "Smoke/debug multi-seed statistics only; no paper multi-seed training."),
        row("experiment", "failed runs", "complete_for_current_failures", ["res/failed_runs/failed_run_audit/failed_run_audit.json"], "Records current known failed run, but does not resolve it.", {"failed_run_count": failed_run_count}),
        row("documentation", "README.md", "complete", ["README.md", "reproduction/README.md"], "None for document scope."),
        row("documentation", "RUNBOOK.md", "complete", ["reproduction/RUNBOOK.md"], "None for document scope."),
        row("documentation", "PROGRESS.md", "complete", ["reproduction/PROGRESS.md"], "None for document scope."),
        row("documentation", "docs/local_inventory.tsv", "complete", ["reproduction/docs/local_inventory.tsv"], "None for document scope."),
        row("documentation", "docs/source_ledger.md", "complete", ["reproduction/docs/source_ledger.md"], "None for document scope."),
        row("documentation", "docs/paper_parameter_map.md", "complete", ["reproduction/docs/paper_parameter_map.md"], "None for document scope."),
        row("documentation", "docs/discrepancy_report.md", "complete", ["reproduction/docs/discrepancy_report.md"], "None for document scope."),
        row("documentation", "docs/unresolved_details.md", "complete", ["reproduction/docs/unresolved_details.md"], "None for document scope."),
        row("documentation", "docs/environment.md", "complete", ["reproduction/docs/environment.md"], "None for document scope."),
        row("documentation", "docs/experiment_protocol.md", "complete", ["reproduction/docs/experiment_protocol.md"], "None for document scope."),
        row("documentation", "docs/known_limitations.md", "complete", ["reproduction/docs/known_limitations.md"], "None for document scope."),
        row("documentation", "res/final_report/reproduction_report.md", "complete", ["res/final_report/reproduction_report.md"], "None for document scope."),
        row(
            "documentation",
            "verification command coverage audit",
            "complete",
            ["res/verification_command_coverage/verification_command_coverage_audit.json"],
            "Smoke subset verifies command hygiene only; it does not execute heavy training/deployment gates.",
        ),
    ]

    status_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    for item in rows:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
        category_counts[item["category"]] = category_counts.get(item["category"], 0) + 1
    missing_evidence_rows = [item for item in rows if item["evidence"] and not item["all_listed_evidence_exists"]]
    blocked_or_missing_rows = [item for item in rows if item["status"] == "blocked_or_missing"]
    summary = {
        "status": "ok" if not missing_evidence_rows else "failed",
        "experiment_type": "deliverable_audit",
        "scope": "goal.md section 16 final deliverables across environment, code, experiment, and documentation",
        "row_count": len(rows),
        "category_counts": dict(sorted(category_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "missing_evidence_rows": missing_evidence_rows,
        "blocked_or_missing_rows": blocked_or_missing_rows,
        "rows": rows,
        "checks": {
            "all_listed_evidence_exists": not missing_evidence_rows,
            "environment_deliverables_present": all(
                item["status"] == "complete" for item in rows if item["category"] == "environment"
            ),
            "documentation_deliverables_present": all(
                item["status"] == "complete" for item in rows if item["category"] == "documentation"
            ),
            "code_deliverables_are_not_all_complete": any(
                item["status"] == "partial" for item in rows if item["category"] == "code"
            ),
            "patch_inventory_recorded": any(
                item["item"] == "patches"
                and "res/code/patch_inventory_audit/patch_inventory_audit.json" in item["evidence"]
                and "res/code/patch_snapshot_audit/patch_snapshot_audit.json" in item["evidence"]
                for item in rows
            ),
            "experiment_deliverables_record_missing_checkpoints_and_videos": any(
                item["item"] == "checkpoints" and item["status"] == "blocked_or_missing" for item in rows
            )
            and any(item["item"] == "videos" and item["status"] == "blocked_or_missing" for item in rows),
            "visual_media_inventory_status_ok": visual_inventory.get("status") == "ok",
            "visual_media_inventory_hashes_recorded": visual_checks.get("all_hashes_recorded") is True,
            "visual_media_inventory_video_gaps_recorded": visual_checks.get("paper_required_video_gaps_recorded") is True,
            "visual_media_inventory_no_reproduction_videos": visual_checks.get("no_mp4_mov_mkv_reproduction_videos") is True,
            "visual_media_inventory_counts_match_deliverable_rows": pdf_count >= 20
            and svg_count >= 20
            and png_count >= 20
            and gif_count >= 3
            and mp4_count == 0,
            "metrics_deliverable_records_formula_api_evidence": any(
                item["category"] == "experiment"
                and item["item"] == "metrics"
                and "res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json" in item["evidence"]
                and "res/tests/core_math_unit_tests/core_math_unit_tests.json" in item["evidence"]
                and "res/tests/core_test_coverage_audit/core_test_coverage_audit.json" in item["evidence"]
                and "res/paper_formula_code_trace/paper_formula_code_trace_audit.json" in item["evidence"]
                and "formula-API" in item["remaining_gap"]
                for item in rows
            ),
            "tests_deliverable_records_core_coverage": any(
                item["category"] == "code"
                and item["item"] == "tests"
                and "res/tests/core_test_coverage_audit/core_test_coverage_audit.json" in item["evidence"]
                for item in rows
            ),
            "does_not_claim_goal_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "Environment and documentation deliverables are present, and many code/experiment artifacts exist, "
                "but checkpoints, videos, full training outputs, and paper-level closed-loop evaluation remain missing."
            ),
        },
        "outputs": {
            "json": str(OUT / "final_deliverables_audit.json"),
            "tsv": str(OUT / "final_deliverables_audit.tsv"),
        },
    }
    (OUT / "final_deliverables_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT / "final_deliverables_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "category",
            "item",
            "status",
            "all_listed_evidence_exists",
            "evidence",
            "remaining_gap",
            "counts",
        ]
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for item in rows:
            writer.writerow(
                {
                    "category": item["category"],
                    "item": item["item"],
                    "status": item["status"],
                    "all_listed_evidence_exists": item["all_listed_evidence_exists"],
                    "evidence": "; ".join(item["evidence"]),
                    "remaining_gap": item["remaining_gap"],
                    "counts": json.dumps(item["counts"], sort_keys=True),
                }
            )
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
