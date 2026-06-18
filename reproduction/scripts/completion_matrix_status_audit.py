#!/usr/bin/env python3
"""Audit completion_matrix.md row hygiene and status vocabulary."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DOC = ROOT / "reproduction/docs/completion_matrix.md"
OUT = ROOT / "res/docs/completion_matrix_status_audit"
ALLOWED_STATUSES = {"complete", "partial", "blocked", "pending", "out_of_scope"}


def parse_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(DOC.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.startswith("|") or line.startswith("|---"):
            continue
        columns = [column.strip() for column in line.strip("|").split("|")]
        if len(columns) < 3:
            rows.append(
                {
                    "line": line_no,
                    "kind": "malformed",
                    "requirement": "",
                    "status": "",
                    "evidence": line,
                    "valid": False,
                    "issue": "fewer_than_three_columns",
                }
            )
            continue
        requirement, status, evidence = columns[:3]
        if requirement == "Requirement" and status == "Status" and evidence == "Evidence":
            rows.append(
                {
                    "line": line_no,
                    "kind": "header",
                    "requirement": requirement,
                    "status": status,
                    "evidence": evidence,
                    "valid": True,
                    "issue": "",
                }
            )
            continue
        valid = status in ALLOWED_STATUSES and bool(requirement) and bool(evidence)
        issue = ""
        if status not in ALLOWED_STATUSES:
            issue = f"invalid_status={status}"
        elif not requirement:
            issue = "empty_requirement"
        elif not evidence:
            issue = "empty_evidence"
        rows.append(
            {
                "line": line_no,
                "kind": "requirement",
                "requirement": requirement,
                "status": status,
                "evidence": evidence,
                "valid": valid,
                "issue": issue,
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = parse_rows()
    requirement_rows = [row for row in rows if row["kind"] == "requirement"]
    header_rows = [row for row in rows if row["kind"] == "header"]
    invalid_rows = [row for row in requirement_rows if not row["valid"]]
    status_counts = Counter(row["status"] for row in requirement_rows if row["status"] in ALLOWED_STATUSES)
    duplicate_requirements = [
        requirement
        for requirement, count in Counter(row["requirement"] for row in requirement_rows).items()
        if count > 1
    ]
    summary = {
        "status": "ok" if DOC.is_file() and not invalid_rows and len(header_rows) >= 1 else "failed",
        "experiment_type": "completion_matrix_status_audit",
        "scope": "Validate completion_matrix.md table rows, status enum usage, and countability.",
        "document": str(DOC),
        "allowed_statuses": sorted(ALLOWED_STATUSES),
        "row_count": len(requirement_rows),
        "header_count": len(header_rows),
        "invalid_status_count": sum(
            1 for row in requirement_rows if row["status"] not in ALLOWED_STATUSES
        ),
        "invalid_row_count": len(invalid_rows),
        "duplicate_requirement_count": len(duplicate_requirements),
        "status_counts": dict(sorted(status_counts.items())),
        "invalid_rows": invalid_rows,
        "duplicate_requirements": duplicate_requirements,
        "rows": rows,
        "checks": {
            "document_exists": DOC.is_file(),
            "has_table_headers": len(header_rows) >= 1,
            "all_requirement_rows_have_three_columns": all(row["kind"] != "malformed" for row in rows),
            "all_statuses_are_allowed_enum": not any(
                row["status"] not in ALLOWED_STATUSES for row in requirement_rows
            ),
            "all_requirement_rows_have_evidence": all(bool(row["evidence"]) for row in requirement_rows),
            "row_count_matches_status_counts": sum(status_counts.values()) == len(requirement_rows),
            "no_duplicate_requirement_names": not duplicate_requirements,
            "does_not_claim_goal_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The matrix status audit makes completion accounting stricter, but it does not resolve the "
                "remaining partial, blocked, or out-of-scope reproduction requirements."
            ),
        },
        "outputs": {
            "json": str(OUT / "completion_matrix_status_audit.json"),
            "tsv": str(OUT / "completion_matrix_status_audit.tsv"),
        },
    }
    (OUT / "completion_matrix_status_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (OUT / "completion_matrix_status_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=["line", "kind", "requirement", "status", "valid", "issue", "evidence"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": summary["outputs"]["json"],
                "rows": summary["row_count"],
                "invalid_status_count": summary["invalid_status_count"],
            }
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
