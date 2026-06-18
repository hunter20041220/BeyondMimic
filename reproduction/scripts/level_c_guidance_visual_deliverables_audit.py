#!/usr/bin/env python3
"""Audit debug guidance visual deliverables against Fig. 5/Fig. 6 needs.

The project has debug guidance plots/GIFs and task metrics, but no trained
closed-loop rollout videos. This audit makes that distinction machine-readable.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/guidance_visual_deliverables_audit"
VIS_JSON = ROOT / "res/level_c/guidance_debug_visualization/level_c_guidance_debug_visualization.json"
TASK_METRIC_JSON = ROOT / "res/level_c/guidance_task_metric_audit/level_c_guidance_task_metric_audit.json"
TASK_SWEEP_JSON = ROOT / "res/level_c/guidance_task_scale_sweep/level_c_guidance_task_scale_sweep.json"
FIG_AUDIT_JSON = ROOT / "res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json"
TASK_COVERAGE_JSON = ROOT / "res/guidance_task_coverage/guidance_task_coverage_audit.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_row(kind: str, path: Path) -> dict[str, Any]:
    return {
        "kind": kind,
        "path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else "",
    }


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["kind", "path", "exists", "size_bytes", "sha256", "task", "metric", "before", "after", "delta"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vis = load_json(VIS_JSON)
    task_metric = load_json(TASK_METRIC_JSON)
    task_sweep = load_json(TASK_SWEEP_JSON)
    fig_audit = load_json(FIG_AUDIT_JSON)
    task_coverage = load_json(TASK_COVERAGE_JSON)

    output_files = [
        file_row(kind, Path(path))
        for kind, path in vis["outputs"].items()
        if kind in {"png", "svg", "pdf", "gif", "npz", "tsv"}
    ]
    task_rows = [
        {
            "kind": "task_metric",
            "path": str(TASK_METRIC_JSON),
            "exists": True,
            "size_bytes": TASK_METRIC_JSON.stat().st_size,
            "sha256": sha256_file(TASK_METRIC_JSON),
            "task": row["task"],
            "metric": row["primary_metric"],
            "before": row["before"],
            "after": row["after"],
            "delta": row["delta"],
        }
        for row in task_metric["rows"]
    ]
    figure_rows = fig_audit["rows"]
    blocked_panels = [
        row
        for row in figure_rows
        if row["status"] == "blocked_for_paper_reproduction_debug_mechanics_only"
    ]
    video_blocked_rows = [
        row for row in task_coverage["rows"] if row["requirement"] == "success_failure_videos"
    ]

    tasks = sorted(row["task"] for row in task_metric["rows"])
    expected_tasks = ["composed", "inpainting", "joystick", "obstacle_avoidance", "waypoint"]
    checks = {
        "visualization_status_ok": vis["status"] == "ok",
        "task_metric_status_ok": task_metric["status"] == "ok",
        "task_scale_sweep_status_ok": task_sweep["status"] == "ok",
        "fig_feasibility_status_ok": fig_audit["status"] == "ok",
        "task_coverage_status_ok": task_coverage["status"] == "ok",
        "png_svg_pdf_gif_npz_tsv_exist": all(row["exists"] and row["size_bytes"] > 0 for row in output_files),
        "file_hashes_recorded": all(len(row["sha256"]) == 64 for row in output_files),
        "five_debug_tasks_visualized": vis["checks"]["five_debug_tasks_visualized"],
        "five_task_metrics_improve": task_metric["checks"]["five_tasks_recorded"]
        and task_metric["checks"]["all_primary_metrics_improve"]
        and task_metric["metrics"]["improved_task_count"] == 5,
        "task_set_matches_expected_guidance_tasks": tasks == expected_tasks,
        "scale_sweep_covers_five_tasks_and_40_rows": task_sweep["checks"]["five_tasks_swept"]
        and task_sweep["row_count"] == 40,
        "fig5_fig6_six_panels_recorded_blocked": fig_audit["counts"]["panels_audited"] == 6
        and len(blocked_panels) == 6,
        "success_failure_video_requirements_recorded_blocked": task_coverage["checks"][
            "all_video_requirements_recorded_blocked"
        ]
        and len(video_blocked_rows) == 6,
        "does_not_claim_paper_video_or_rollout": vis["checks"]["does_not_claim_paper_video"]
        and task_metric["checks"]["does_not_claim_success_failure_videos"]
        and task_metric["checks"]["does_not_claim_closed_loop_rollout"],
        "does_not_claim_fig5_fig6_reproduction": task_metric["checks"]["does_not_claim_fig5_fig6_reproduction"],
        "does_not_claim_goal_complete": True,
        "atomic_write_used": True,
    }
    metrics = {
        "visual_file_count": len(output_files),
        "visual_total_size_bytes": sum(row["size_bytes"] for row in output_files),
        "guidance_task_count": len(tasks),
        "improved_task_count": task_metric["metrics"]["improved_task_count"],
        "mean_primary_delta": task_metric["metrics"]["mean_primary_delta"],
        "scale_sweep_row_count": task_sweep["row_count"],
        "fig5_fig6_panel_count": fig_audit["counts"]["panels_audited"],
        "blocked_fig_panel_count": len(blocked_panels),
        "blocked_video_requirement_count": len(video_blocked_rows),
        "failed_check_count": sum(1 for value in checks.values() if not value),
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "level_c_guidance_visual_deliverables_audit",
        "scope": (
            "Audit of debug guidance visualization deliverables and task metrics against paper Fig. 5/Fig. 6 and "
            "success/failure video requirements."
        ),
        "source_artifacts": {
            "guidance_debug_visualization": str(VIS_JSON),
            "guidance_task_metric_audit": str(TASK_METRIC_JSON),
            "guidance_task_scale_sweep": str(TASK_SWEEP_JSON),
            "fig5_fig6_feasibility_audit": str(FIG_AUDIT_JSON),
            "guidance_task_coverage": str(TASK_COVERAGE_JSON),
        },
        "visual_files": output_files,
        "task_primary_metrics": task_metric["rows"],
        "figure_panel_statuses": figure_rows,
        "blocked_video_rows": video_blocked_rows,
        "metrics": metrics,
        "checks": checks,
        "interpretation": {
            "paper_level_status": "debug_visuals_only",
            "goal_complete": False,
            "why_not_complete": (
                "The debug guidance deliverables provide static plots, a GIF, task metrics, and scale sweeps for local "
                "formula effects. They are not success/failure videos, trained diffusion rollouts, closed-loop "
                "sim/hardware logs, or paper Fig. 5/Fig. 6 reproduction data."
            ),
        },
        "outputs": {
            "json": str(OUT / "level_c_guidance_visual_deliverables_audit.json"),
            "tsv": str(OUT / "level_c_guidance_visual_deliverables_audit.tsv"),
        },
    }
    rows = output_files + task_rows
    atomic_write_text(
        OUT / "level_c_guidance_visual_deliverables_audit.json",
        json.dumps(summary, indent=2, sort_keys=True),
    )
    atomic_write_tsv(OUT / "level_c_guidance_visual_deliverables_audit.tsv", rows)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "visual_files": metrics["visual_file_count"],
                "tasks": metrics["guidance_task_count"],
                "failed": metrics["failed_check_count"],
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
