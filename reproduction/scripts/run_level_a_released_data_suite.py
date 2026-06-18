#!/usr/bin/env python3
"""Run Level-A paper table and released-data reproduction audits."""

from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_a/released_data_suite"
LOG_DIR = ROOT / "logs/level_a_released_data_suite"
PYTHON = "python3"
ANALYSIS_PYTHON = str(ROOT / "envs/bm_analysis/bin/python")


COMMANDS = [
    (
        "paper_table_value_audit",
        [PYTHON, str(ROOT / "reproduction/scripts/paper_table_value_audit.py")],
        "res/paper_table_values/paper_table_value_audit.json",
    ),
    (
        "skill_success_table_data_audit",
        [PYTHON, str(ROOT / "reproduction/scripts/skill_success_table_data_audit.py")],
        "res/paper_skill_success_table_audit/skill_success_table_data_audit.json",
    ),
    (
        "released_panel_mapping_audit",
        [PYTHON, str(ROOT / "reproduction/scripts/released_panel_mapping_audit.py")],
        "res/released_panel_mapping_audit/released_panel_mapping_audit.json",
    ),
    (
        "released_data_metrics_summary",
        [ANALYSIS_PYTHON, str(ROOT / "reproduction/scripts/released_data_metrics_summary.py")],
        "res/tables/released_data_metrics_summary/released_data_metrics_summary.json",
    ),
    (
        "released_data_statistical_audit",
        [ANALYSIS_PYTHON, str(ROOT / "reproduction/scripts/released_data_statistical_audit.py")],
        "res/tables/released_data_statistical_audit/released_data_statistical_audit.json",
    ),
    (
        "paper_vs_reproduction_comparison",
        [PYTHON, str(ROOT / "reproduction/scripts/paper_vs_reproduction_comparison.py")],
        "res/comparison/paper_vs_reproduction.json",
    ),
]


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_json(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def run_step(name: str, command: list[str], output_rel: str) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=240,
        check=False,
    )
    duration = time.perf_counter() - start
    log_path = LOG_DIR / f"{name}.log"
    atomic_write_text(log_path, proc.stdout)
    output_path = ROOT / output_rel
    output = load_json(output_rel)
    return {
        "name": name,
        "command": " ".join(command),
        "returncode": proc.returncode,
        "duration_sec": duration,
        "passed": proc.returncode == 0 and output.get("status") == "ok",
        "output": str(output_path),
        "output_exists": output_path.is_file() and output_path.stat().st_size > 0,
        "output_status": output.get("status"),
        "log": str(log_path),
        "stdout_tail": proc.stdout[-2000:],
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["name", "command", "returncode", "duration_sec", "passed", "output", "output_exists", "output_status", "log"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    steps = [run_step(name, command, output_rel) for name, command, output_rel in COMMANDS]

    table = load_json("res/paper_table_values/paper_table_value_audit.json")
    skill = load_json("res/paper_skill_success_table_audit/skill_success_table_data_audit.json")
    panels = load_json("res/released_panel_mapping_audit/released_panel_mapping_audit.json")
    metrics = load_json("res/tables/released_data_metrics_summary/released_data_metrics_summary.json")
    stats = load_json("res/tables/released_data_statistical_audit/released_data_statistical_audit.json")
    comparison = load_json("res/comparison/paper_vs_reproduction.json")

    checks = {
        "all_steps_pass": all(step["passed"] for step in steps),
        "paper_table_rows_58": table.get("counts", {}).get("total_rows") == 58,
        "paper_table_mismatch_zero": table.get("counts", {}).get("mismatch_rows") == 0,
        "skill_table_lafan_rows_parsed": skill.get("checks", {}).get("lafan_rows_parsed") is True,
        "released_panel_rows_15": panels.get("metrics", {}).get("released_panel_rows") == 15,
        "released_panel_failures_zero": panels.get("metrics", {}).get("released_panel_fail_count") == 0,
        "released_metrics_source_csv_count_10": metrics.get("metrics", {}).get("source_csv_count") == 10,
        "released_metrics_finite": metrics.get("checks", {}).get("metrics_are_finite") is True,
        "released_stats_ci_rows_present": (
            stats.get("checks", {}).get("grf_ci_rows_12") is True
            and stats.get("checks", {}).get("imu_ci_rows_11") is True
        ),
        "comparison_goal_rows_present": comparison.get("checks", {}).get("required_goal_checkpoint_rows_present") is True,
        "comparison_does_not_claim_goal_complete": comparison.get("checks", {}).get("does_not_claim_goal_complete") is True,
        "atomic_write_used": True,
        "does_not_claim_training_or_deployment": True,
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "level_a_released_data_suite",
        "scope": "Unified execution of paper table, released-data mapping/statistics, and paper-vs-reproduction audits.",
        "step_count": len(steps),
        "pass_count": sum(1 for step in steps if step["passed"]),
        "steps": steps,
        "checks": checks,
        "metrics": {
            "paper_table_rows": table.get("counts", {}).get("total_rows"),
            "paper_table_mismatch_rows": table.get("counts", {}).get("mismatch_rows"),
            "lafan_rows": skill.get("metrics", {}).get("lafan_rows"),
            "released_panel_rows": panels.get("metrics", {}).get("released_panel_rows"),
            "released_panel_fail_count": panels.get("metrics", {}).get("released_panel_fail_count"),
            "released_source_csv_count": metrics.get("metrics", {}).get("source_csv_count"),
            "ablation_row_count": metrics.get("metrics", {}).get("ablation_row_count"),
            "grf_ci_rows": stats.get("metrics", {}).get("grf_ci_rows"),
            "imu_ci_rows": stats.get("metrics", {}).get("imu_ci_rows"),
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This suite reruns Level-A released-data and table audits. It does not train policies, run closed-loop "
                "IsaacLab/ROS/TensorRT evaluation, reproduce Fig. 5/Fig. 6, or execute real Unitree G1 hardware."
            ),
        },
        "outputs": {
            "json": str(OUT / "level_a_released_data_suite.json"),
            "tsv": str(OUT / "level_a_released_data_suite.tsv"),
            "log_dir": str(LOG_DIR),
        },
    }
    atomic_write_text(OUT / "level_a_released_data_suite.json", json.dumps(summary, indent=2, sort_keys=True))
    write_tsv(OUT / "level_a_released_data_suite.tsv", steps)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "steps": summary["step_count"],
                "pass": summary["pass_count"],
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
