#!/usr/bin/env python3
"""Audit final report coverage of the 12 explicit goal.md final-report requirements."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DOCS_REPORT = ROOT / "reproduction/docs/final_reproduction_report.md"
GOAL_REPORT = ROOT / "res/final_report/reproduction_report.md"
SUMMARY_JSON = ROOT / "res/final_report/final_reproduction_report.json"
OUT = ROOT / "res/final_report/final_report_requirement_audit"

REQUIREMENTS = [
    ("official_code_used", "Official code used:", ["whole_body_tracking", "motion_tracking_controller"]),
    ("paper_reimplementation", "Paper-faithful reimplementation:", ["VAE", "diffusion", "guidance"]),
    ("released_data_reproduction", "Released-data reproduction:", ["released data", "released-figure"]),
    ("retrained_results", "Retrained results:", ["no paper-scale", "debug overfit"]),
    ("qualitative_only", "Qualitative-only comparison:", ["qualitative", "paper_vs_reproduction.csv"]),
    ("not_publicly_reproducible", "Not publicly reproducible:", ["missing official Level C", "Fig. 5/Fig. 6"]),
    ("result_differences", "Result differences:", ["exact", "approximate", "not-publicly-reproducible"]),
    ("difference_sources", "Difference sources:", ["blocked gates", "adaptive-sampling discrepancy"]),
    ("credibility", "Current reproduction credibility:", ["strong", "partial", "blocked"]),
    ("complete_incomplete_scope", "Completed and incomplete scope:", ["completion matrix", "blocked-gate"]),
    ("hardware_cost_training_time", "Hardware cost and training time:", ["GPU resource", "wall-clock time"]),
    ("one_command_rerun", "One-command rerun path:", ["Verification Commands", "RUNBOOK.md"]),
]


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    docs_text = DOCS_REPORT.read_text(encoding="utf-8") if DOCS_REPORT.is_file() else ""
    goal_text = GOAL_REPORT.read_text(encoding="utf-8") if GOAL_REPORT.is_file() else ""
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8")) if SUMMARY_JSON.is_file() else {}
    texts_identical = docs_text == goal_text and bool(docs_text)
    rows = []
    for name, anchor, patterns in REQUIREMENTS:
        rows.append(
            {
                "requirement": name,
                "anchor": anchor,
                "patterns": "; ".join(patterns),
                "anchor_present": anchor in goal_text,
                "patterns_present": all(pattern in goal_text for pattern in patterns),
                "passed": anchor in goal_text and all(pattern in goal_text for pattern in patterns),
            }
        )
    failed = [row for row in rows if not row["passed"]]
    report_paths = summary.get("outputs", {})
    summary_atomic_write_used = summary.get("checks", {}).get("atomic_write_used") is True
    audit = {
        "status": (
            "ok"
            if texts_identical
            and not failed
            and summary.get("goal_complete") is False
            and summary_atomic_write_used
            else "failed"
        ),
        "experiment_type": "doc_audit",
        "scope": "coverage of the 12 explicit final-report requirements in goal.md section 16",
        "documents": {
            "docs_report": str(DOCS_REPORT),
            "goal_report": str(GOAL_REPORT),
            "summary_json": str(SUMMARY_JSON),
        },
        "row_count": len(rows),
        "missing_count": len(failed),
        "rows": rows,
        "missing_rows": failed,
        "checks": {
            "docs_report_exists": DOCS_REPORT.is_file(),
            "goal_report_exists": GOAL_REPORT.is_file(),
            "summary_json_exists": SUMMARY_JSON.is_file(),
            "markdown_reports_identical": texts_identical,
            "summary_points_to_goal_report": report_paths.get("goal_markdown") == str(GOAL_REPORT),
            "summary_points_to_docs_report": report_paths.get("markdown") == str(DOCS_REPORT),
            "summary_atomic_write_used": summary_atomic_write_used,
            "atomic_write_used": True,
            "all_12_goal_report_requirements_present": not failed,
            "verification_commands_present": "## Verification Commands" in goal_text,
            "blocked_gates_present": "## Blocked Gates" in goal_text,
            "key_evidence_present": "## Key Evidence" in goal_text,
            "does_not_claim_goal_complete": summary.get("goal_complete") is False and "does not mark the full goal complete" in goal_text,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The final report now explicitly covers the 12 required reporting topics, but that is reporting "
                "coverage only; it does not provide missing training checkpoints, paper-level figures, videos, "
                "TensorRT deployment, or hardware execution."
            ),
        },
        "outputs": {
            "json": str(OUT / "final_report_requirement_audit.json"),
            "tsv": str(OUT / "final_report_requirement_audit.tsv"),
        },
    }
    atomic_write_text(OUT / "final_report_requirement_audit.json", json.dumps(audit, indent=2, sort_keys=True))
    tsv_path = OUT / "final_report_requirement_audit.tsv"
    tmp_tsv_path = tsv_path.with_suffix(tsv_path.suffix + ".tmp")
    with tmp_tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=["requirement", "anchor", "patterns", "anchor_present", "patterns_present", "passed"],
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp_tsv_path.replace(tsv_path)
    print(json.dumps({"status": audit["status"], "json": audit["outputs"]["json"], "rows": len(rows)}))
    if audit["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
