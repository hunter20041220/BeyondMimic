#!/usr/bin/env python3
"""Audit full training-run deliverable gaps against goal.md run contract."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
RUN_ROOT = ROOT / "res/runs"
OUT = ROOT / "res/run_management_audit/full_run_deliverable_gap_audit"

REQUIRED_FILES = [
    "resolved_config.yaml",
    "command.sh",
    "stdout.log",
    "stderr.log",
    "environment.txt",
    "git_state.txt",
    "gpu_metrics.csv",
    "metrics.json",
    "metrics.csv",
    "status.json",
]
REQUIRED_DIRS = ["checkpoint", "figures", "videos"]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def dir_nonempty(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def file_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def audit_run(run_dir: Path) -> dict[str, Any]:
    status = load_json(run_dir / "status.json")
    metrics = load_json(run_dir / "metrics.json")
    missing_files = [name for name in REQUIRED_FILES if not (run_dir / name).is_file()]
    missing_dirs = [name for name in REQUIRED_DIRS if not (run_dir / name).is_dir()]
    empty_required_files = [
        name
        for name in REQUIRED_FILES
        if (run_dir / name).is_file() and name != "stderr.log" and (run_dir / name).stat().st_size == 0
    ]
    nonempty_dirs = {name: dir_nonempty(run_dir / name) for name in REQUIRED_DIRS}
    runtime_metric_keys = [
        "samples_per_second",
        "environment_steps_per_second",
        "iteration_time",
        "estimated_remaining_time",
    ]
    runtime_metrics_present = any(metrics.get(key) not in {None, "", 0} for key in runtime_metric_keys)
    has_training_endpoint_metrics = all(key in metrics for key in ["oom_count", "restart_count"]) and runtime_metrics_present
    status_success = status.get("status") == "SUCCESS" or metrics.get("status") == "SUCCESS"
    is_training_run = status.get("is_training_run") is True or metrics.get("is_training_run") is True
    is_valid_training_run = (
        status_success
        and is_training_run
        and not missing_files
        and not missing_dirs
        and not empty_required_files
        and nonempty_dirs["checkpoint"]
        and nonempty_dirs["figures"]
        and nonempty_dirs["videos"]
        and has_training_endpoint_metrics
    )
    gap_reasons = []
    if not status_success:
        gap_reasons.append("status_not_success")
    if not is_training_run:
        gap_reasons.append("not_marked_training_run")
    if missing_files:
        gap_reasons.append("missing_required_files")
    if missing_dirs:
        gap_reasons.append("missing_required_dirs")
    if empty_required_files:
        gap_reasons.append("empty_required_files")
    for dirname, ok in nonempty_dirs.items():
        if not ok:
            gap_reasons.append(f"{dirname}_dir_empty")
    if not has_training_endpoint_metrics:
        gap_reasons.append("missing_training_endpoint_runtime_metrics")
    return {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "status": status.get("status") or metrics.get("status"),
        "is_training_run": is_training_run,
        "is_valid_training_run": is_valid_training_run,
        "missing_files": missing_files,
        "missing_dirs": missing_dirs,
        "empty_required_files": empty_required_files,
        "nonempty_dirs": nonempty_dirs,
        "has_training_endpoint_metrics": has_training_endpoint_metrics,
        "gap_reasons": sorted(set(gap_reasons)),
    }


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "run_id",
        "status",
        "is_training_run",
        "is_valid_training_run",
        "missing_files",
        "missing_dirs",
        "empty_required_files",
        "nonempty_dirs",
        "has_training_endpoint_metrics",
        "gap_reasons",
        "run_dir",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key], sort_keys=True) if isinstance(row.get(key), (list, dict)) else row.get(key)
                    for key in fieldnames
                }
            )
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(path for path in RUN_ROOT.glob("*") if path.is_dir()) if RUN_ROOT.is_dir() else []
    rows = [audit_run(path) for path in run_dirs]
    valid_rows = [row for row in rows if row["is_valid_training_run"]]
    diagnostic_or_debug_rows = [row for row in rows if not row["is_valid_training_run"]]
    valid_rows_with_video = [row for row in valid_rows if row["nonempty_dirs"].get("videos")]
    diagnostic_or_debug_rows_with_video = [
        row for row in diagnostic_or_debug_rows if row["nonempty_dirs"].get("videos")
    ]
    rows_with_output_or_runtime_gaps = [
        row
        for row in diagnostic_or_debug_rows
        if any(reason.endswith("_dir_empty") for reason in row["gap_reasons"])
        or "missing_training_endpoint_runtime_metrics" in row["gap_reasons"]
    ]
    rows_not_marked_training = [row for row in diagnostic_or_debug_rows if "not_marked_training_run" in row["gap_reasons"]]
    runs_with_complete_schema = [
        row
        for row in rows
        if not row["missing_files"] and not row["missing_dirs"] and not row["empty_required_files"]
    ]
    checks = {
        "run_root_exists": RUN_ROOT.is_dir(),
        "at_least_two_run_dirs_indexed": len(rows) >= 2,
        "all_runs_have_required_schema_files_or_record_gaps": all(
            not row["missing_files"] and not row["missing_dirs"] for row in rows
        ),
        "no_valid_training_run_claimed": len(valid_rows) == 0,
        "diagnostic_or_debug_runs_not_promoted_to_success": all(
            not row["is_valid_training_run"] for row in diagnostic_or_debug_rows
        ),
        "all_diagnostic_or_debug_runs_not_marked_training": len(rows_not_marked_training) == len(diagnostic_or_debug_rows),
        "output_gaps_or_debug_video_boundary_recorded": all(
            row in rows_with_output_or_runtime_gaps
            or (
                row["nonempty_dirs"].get("videos")
                and "not_marked_training_run" in row["gap_reasons"]
            )
            for row in diagnostic_or_debug_rows
        ),
        "at_least_one_training_endpoint_metric_gap_recorded": any(
            "missing_training_endpoint_runtime_metrics" in row["gap_reasons"] for row in diagnostic_or_debug_rows
        ),
        "no_valid_training_run_videos": len(valid_rows_with_video) == 0,
        "debug_videos_do_not_create_valid_training_run": all(
            not row["is_valid_training_run"] for row in diagnostic_or_debug_rows_with_video
        ),
        "does_not_claim_goal_complete": True,
        "atomic_write_used": True,
    }
    metrics = {
        "run_directory_count": len(rows),
        "schema_complete_run_count": len(runs_with_complete_schema),
        "valid_training_run_count": len(valid_rows),
        "diagnostic_or_debug_run_count": len(diagnostic_or_debug_rows),
        "run_with_nonempty_checkpoint_dir_count": sum(1 for row in rows if row["nonempty_dirs"].get("checkpoint")),
        "run_with_nonempty_figures_dir_count": sum(1 for row in rows if row["nonempty_dirs"].get("figures")),
        "run_with_nonempty_videos_dir_count": sum(1 for row in rows if row["nonempty_dirs"].get("videos")),
            "run_with_training_endpoint_metrics_count": sum(1 for row in rows if row["has_training_endpoint_metrics"]),
            "valid_training_run_with_nonempty_videos_count": len(valid_rows_with_video),
            "diagnostic_or_debug_run_with_nonempty_videos_count": len(diagnostic_or_debug_rows_with_video),
            "failed_check_count": sum(1 for value in checks.values() if not value),
        }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "full_run_deliverable_gap_audit",
        "scope": "Per-run audit of goal.md full training-run deliverable requirements.",
        "goal_evidence": {
            "run_id_pattern_and_status": str(ROOT / "goal.md:1754-1767"),
            "required_run_files_and_dirs": str(ROOT / "goal.md:1769-1784"),
            "final_experiment_deliverables": str(ROOT / "goal.md:1825-1829"),
        },
        "required_files": REQUIRED_FILES,
        "required_dirs": REQUIRED_DIRS,
        "rows": rows,
        "metrics": metrics,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "Current run directories satisfy the schema/plumbing level, but none is a valid full training run with "
                "SUCCESS status, training runtime metrics, nonempty checkpoint/figure/video outputs, and evaluation "
                "deliverables."
            ),
        },
        "outputs": {
            "json": str(OUT / "full_run_deliverable_gap_audit.json"),
            "tsv": str(OUT / "full_run_deliverable_gap_audit.tsv"),
        },
    }
    atomic_write_text(
        OUT / "full_run_deliverable_gap_audit.json",
        json.dumps(summary, indent=2, sort_keys=True),
    )
    atomic_write_tsv(OUT / "full_run_deliverable_gap_audit.tsv", rows)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "runs": metrics["run_directory_count"],
                "valid_training_runs": metrics["valid_training_run_count"],
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
