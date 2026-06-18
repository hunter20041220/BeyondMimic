#!/usr/bin/env python3
"""Catalog run logs, commands, configs, metrics, and failure/status files."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/run_log_config_catalog"


PATTERNS = [
    ("setup_log", "logs/setup/*"),
    ("data_log", "logs/data/*"),
    ("level_c_log", "logs/level_c/*"),
    ("gpu_log", "logs/gpu/*"),
    ("run_command", "res/runs/*/command.sh"),
    ("run_environment", "res/runs/*/environment.txt"),
    ("run_git_state", "res/runs/*/git_state.txt"),
    ("run_gpu_metrics", "res/runs/*/gpu_metrics.csv"),
    ("run_metrics", "res/runs/*/metrics.*"),
    ("run_config", "res/runs/*/resolved_config.yaml"),
    ("run_status", "res/runs/*/status.json"),
    ("run_stdout", "res/runs/*/stdout.log"),
    ("run_stderr", "res/runs/*/stderr.log"),
    ("global_config", "res/config/resolved_reproduction_config.*"),
    ("failed_run_status", "res/failed_runs/*/*"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def line_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def classify_run(path: Path) -> str:
    parts = path.relative_to(ROOT).parts
    if len(parts) >= 3 and parts[0] == "res" and parts[1] == "runs":
        return parts[2]
    if len(parts) >= 3 and parts[0] == "res" and parts[1] == "failed_runs":
        return parts[2]
    return ""


def collect_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for category, pattern in PATTERNS:
        for path in sorted(ROOT.glob(pattern)):
            if not path.is_file() or path in seen:
                continue
            seen.add(path)
            rel = path.relative_to(ROOT)
            rows.append(
                {
                    "category": category,
                    "run_id": classify_run(path),
                    "relative_path": str(rel),
                    "absolute_path": str(path),
                    "size_bytes": path.stat().st_size,
                    "line_count": line_count(path),
                    "sha256": sha256_file(path),
                }
            )
    return sorted(rows, key=lambda row: (row["category"], row["relative_path"]))


def summarize_status_files(rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["relative_path"].endswith("status.json"):
            try:
                data = json.loads((ROOT / row["relative_path"]).read_text(encoding="utf-8"))
            except Exception:
                data = {}
            status_rows.append(
                {
                    "run_id": row["run_id"],
                    "path": row["relative_path"],
                    "status": data.get("status"),
                    "experiment_type": data.get("experiment_type"),
                    "is_valid_training_run": data.get("is_valid_training_run"),
                }
            )
    return {
        "status_rows": status_rows,
        "valid_training_run_count": sum(1 for row in status_rows if row.get("is_valid_training_run") is True),
        "invalid_or_debug_run_count": sum(1 for row in status_rows if row.get("is_valid_training_run") is not True),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["category", "run_id", "relative_path", "absolute_path", "size_bytes", "line_count", "sha256"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Run/log/config catalog",
        "",
        "This catalog hashes current setup, data, debug, failure, and diagnostic run logs/configs.",
        "It does not claim any paper-scale training run is complete.",
        "",
        f"- Status: `{summary['status']}`",
        f"- File count: `{summary['metrics']['file_count']}`",
        f"- Log file count: `{summary['metrics']['log_file_count']}`",
        f"- Run directory count: `{summary['metrics']['run_directory_count']}`",
        f"- Config file count: `{summary['metrics']['config_file_count']}`",
        f"- Valid training run count: `{summary['metrics']['valid_training_run_count']}`",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for category, count in sorted(summary["category_counts"].items()):
        lines.append(f"| `{category}` | `{count}` |")
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
    rows = collect_rows()
    category_counts: dict[str, int] = {}
    run_ids = {row["run_id"] for row in rows if row["run_id"]}
    for row in rows:
        category_counts[row["category"]] = category_counts.get(row["category"], 0) + 1
    status_summary = summarize_status_files(rows)
    config_count = sum(1 for row in rows if row["category"] in {"run_config", "global_config"})
    log_count = sum(1 for row in rows if row["relative_path"].endswith((".log", ".txt")))
    checks = {
        "logs_are_indexed": log_count >= 20,
        "run_configs_are_indexed": config_count >= 3,
        "run_status_files_are_indexed": len(status_summary["status_rows"]) >= 2,
        "hashes_recorded_for_all_files": all(bool(row["sha256"]) for row in rows),
        "no_valid_training_run_claimed": status_summary["valid_training_run_count"] == 0,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "run_log_config_catalog",
        "scope": "setup/data/debug/failure logs, run configs, commands, metrics, and status files",
        "metrics": {
            "file_count": len(rows),
            "log_file_count": log_count,
            "run_directory_count": len(run_ids),
            "config_file_count": config_count,
            "valid_training_run_count": status_summary["valid_training_run_count"],
            "invalid_or_debug_run_count": status_summary["invalid_or_debug_run_count"],
        },
        "category_counts": dict(sorted(category_counts.items())),
        "status_rows": status_summary["status_rows"],
        "rows": rows,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The catalog improves traceability for logs and configs, but all indexed run statuses are diagnostic, "
                "debug, failed, or setup evidence. It does not provide completed paper-scale training logs."
            ),
        },
        "outputs": {
            "json": str(OUT / "run_log_config_catalog.json"),
            "csv": str(OUT / "run_log_config_catalog.csv"),
            "markdown": str(OUT / "run_log_config_catalog.md"),
        },
    }
    (OUT / "run_log_config_catalog.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(OUT / "run_log_config_catalog.csv", rows)
    write_markdown(OUT / "run_log_config_catalog.md", summary)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "files": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
