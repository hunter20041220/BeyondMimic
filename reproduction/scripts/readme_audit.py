#!/usr/bin/env python3
"""Audit the top-level README coverage for reproduction entry-point requirements."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DOC = ROOT / "README.md"
OUT = ROOT / "res/docs/readme_audit"

REQUIRED_PATTERNS = {
    "goal_reference": "goal.md",
    "download_read_only": "read-only",
    "goal_final_report": "res/final_report/reproduction_report.md",
    "docs_final_report": "reproduction/docs/final_reproduction_report.md",
    "experiment_protocol": "reproduction/docs/experiment_protocol.md",
    "master_audit": "res/master_audit/reproduction_master_audit.json",
    "artifact_manifest": "res/artifact_manifest/artifact_manifest.json",
    "goal_not_complete": "not complete",
    "isaaclab_kit_blocker": "IsaacLab/Kit",
    "dagger_gap": "DAgger",
    "checkpoint_gap": "checkpoints",
    "fig5_fig6_gap": "Fig. 5/Fig. 6",
    "unitree_g1_gap": "Unitree G1",
    "master_audit_command": "reproduction/scripts/reproduction_master_audit.py",
    "final_report_command": "reproduction/scripts/final_reproduction_report.py",
    "resolved_config_command": "reproduction/scripts/resolved_reproduction_config.py",
    "artifact_manifest_command": "reproduction/scripts/artifact_manifest.py",
    "protocol_audit_command": "reproduction/scripts/experiment_protocol_audit.py",
    "no_fabrication_rule": "Do not fabricate",
    "failed_runs_rule": "Preserve failed runs",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    text = DOC.read_text(encoding="utf-8") if DOC.is_file() else ""
    rows = [
        {
            "requirement": name,
            "pattern": pattern,
            "present": pattern in text,
        }
        for name, pattern in REQUIRED_PATTERNS.items()
    ]
    missing = [row for row in rows if not row["present"]]
    summary = {
        "status": "ok" if DOC.is_file() and not missing else "failed",
        "experiment_type": "doc_audit",
        "scope": "top-level README coverage for current BeyondMimic reproduction entry point",
        "document": str(DOC),
        "row_count": len(rows),
        "missing_count": len(missing),
        "missing_rows": missing,
        "checks": {
            "document_exists": DOC.is_file(),
            "all_required_patterns_present": not missing,
            "points_to_final_report": (
                "res/final_report/reproduction_report.md" in text
                and "reproduction/docs/final_reproduction_report.md" in text
            ),
            "points_to_goal_final_report": "res/final_report/reproduction_report.md" in text,
            "points_to_docs_final_report": "reproduction/docs/final_reproduction_report.md" in text,
            "points_to_experiment_protocol": "reproduction/docs/experiment_protocol.md" in text,
            "points_to_master_audit": "res/master_audit/reproduction_master_audit.json" in text,
            "documents_raw_download_readonly": "download/" in text and "read-only" in text,
            "documents_current_incomplete_boundary": "not complete" in text and "remain blocked or missing" in text,
            "documents_major_blockers": all(
                pattern in text
                for pattern in ["IsaacLab/Kit", "DAgger", "checkpoints", "Fig. 5/Fig. 6", "Unitree G1"]
            ),
            "documents_no_fabrication_rule": "Do not fabricate" in text,
            "documents_failed_run_retention_rule": "Preserve failed runs" in text,
            "does_not_claim_goal_complete": "goal_complete=true" not in text and "full paper reproduction is complete" not in text,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The README is a verified entry point for the current evidence set, but it does not create missing "
                "trained checkpoints, videos, Fig. 5/Fig. 6 paper-level results, TensorRT deployment, or G1 hardware "
                "execution evidence."
            ),
        },
        "outputs": {
            "json": str(OUT / "readme_audit.json"),
            "tsv": str(OUT / "readme_audit.tsv"),
        },
    }
    (OUT / "readme_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT / "readme_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["requirement", "pattern", "present"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
