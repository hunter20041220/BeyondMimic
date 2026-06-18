#!/usr/bin/env python3
"""Audit reproduction/docs/experiment_protocol.md coverage."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DOC = ROOT / "reproduction/docs/experiment_protocol.md"
OUT = ROOT / "res/docs/experiment_protocol_audit"

REQUIRED_PATTERNS = {
    "phase_0": "Phase 0",
    "phase_1": "Phase 1",
    "phase_2": "Phase 2",
    "phase_3": "Phase 3",
    "phase_4": "Phase 4",
    "phase_5": "Phase 5",
    "phase_6": "Phase 6",
    "phase_7": "Phase 7",
    "phase_8": "Phase 8",
    "phase_9": "Phase 9",
    "phase_10": "Phase 10",
    "no_fabrication": "Do not fabricate",
    "download_read_only": "download` is read-only",
    "run_directory_contract": "Run Directory Contract",
    "failed_runs": "res/failed_runs",
    "gpu_metrics": "gpu_metrics.csv",
    "kit_inotify_boundary": "isaaclab_kit_inotify",
    "final_report": "final_reproduction_report.md",
    "goal_incomplete_boundary": "full goal is incomplete",
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    text = DOC.read_text(encoding="utf-8")
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
        "status": "ok" if not missing else "failed",
        "experiment_type": "doc_audit",
        "scope": "experiment protocol coverage against goal.md execution phases and gates",
        "document": str(DOC),
        "row_count": len(rows),
        "missing_count": len(missing),
        "missing_rows": missing,
        "checks": {
            "document_exists": DOC.is_file(),
            "all_required_patterns_present": not missing,
            "all_phases_present": all(row["present"] for row in rows if row["requirement"].startswith("phase_")),
            "run_contract_present": any(row["requirement"] == "run_directory_contract" and row["present"] for row in rows),
            "failure_handling_present": any(row["requirement"] == "failed_runs" and row["present"] for row in rows),
            "does_not_claim_goal_complete": "full goal is incomplete" in text,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The protocol documents the required execution gates, but it does not execute blocked training, "
                "generate checkpoints, or reproduce paper-level deployment results."
            ),
        },
        "outputs": {
            "json": str(OUT / "experiment_protocol_audit.json"),
            "tsv": str(OUT / "experiment_protocol_audit.tsv"),
        },
    }
    (OUT / "experiment_protocol_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT / "experiment_protocol_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["requirement", "pattern", "present"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
