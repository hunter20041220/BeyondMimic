#!/usr/bin/env python3
"""Index directive-bearing lines from goal.md for traceable reproduction work."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
GOAL = ROOT / "goal.md"
OUT = ROOT / "res/goal_directive_index"

DIRECTIVE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("mandatory", re.compile(r"必须|需要实际|首要职责|首先|至少|务必|should|must", re.I)),
    ("prohibition", re.compile(r"不得|禁止|不能|不允许|不要|未经|avoid|do not|must not", re.I)),
    ("deliverable", re.compile(r"交付|输出|保存|记录|生成|报告|图表|视频|checkpoint|模型|日志|指标|final report", re.I)),
    ("execution", re.compile(r"运行|训练|评测|复现|smoke|执行|启动|deploy|rollout|PPO|VAE|diffusion|DAgger", re.I)),
    ("boundary", re.compile(r"区分|标记|明确|无法|缺失|blocked|out.of.scope|resource_adjusted|paper_exact", re.I)),
]


def section_id(title: str) -> str:
    text = title.strip("# ").strip()
    return text.split("｜", 1)[0].strip()


def line_class(text: str) -> list[str]:
    tags = [name for name, pattern in DIRECTIVE_PATTERNS if pattern.search(text)]
    return tags or ["context"]


def existing_json(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key, "")
                out[key] = ",".join(value) if isinstance(value, list) else value
            writer.writerow(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines = GOAL.read_text(encoding="utf-8").splitlines()

    heading_rows: list[dict[str, Any]] = []
    directive_rows: list[dict[str, Any]] = []
    current_heading = ""
    current_section = ""
    tag_counts: Counter[str] = Counter()
    section_directive_counts: Counter[str] = Counter()

    for idx, raw_line in enumerate(lines, start=1):
        text = raw_line.strip()
        if text.startswith("#"):
            current_heading = text
            current_section = section_id(text)
            heading_rows.append({"line": idx, "heading": text, "section_id": current_section})
            continue
        if not text or text.startswith("```") or set(text) <= {"-"}:
            continue
        tags = line_class(text)
        if tags == ["context"]:
            continue
        row = {
            "line": idx,
            "section_id": current_section,
            "heading": current_heading,
            "tags": tags,
            "text": text,
        }
        directive_rows.append(row)
        section_directive_counts[current_section] += 1
        for tag in tags:
            tag_counts[tag] += 1

    traceability = existing_json("res/goal_traceability/goal_traceability_audit.json")
    requirement_matrix = existing_json("res/goal_requirement_matrix/goal_requirement_matrix_audit.json")

    top_sections = sorted(
        [
            {"section_id": section, "directive_count": count}
            for section, count in section_directive_counts.items()
        ],
        key=lambda row: (-row["directive_count"], row["section_id"]),
    )

    summary: dict[str, Any] = {
        "status": "ok" if directive_rows and heading_rows else "failed",
        "experiment_type": "goal_directive_index_audit",
        "scope": "line-level directive index for goal.md",
        "goal_path": str(GOAL),
        "line_count": len(lines),
        "heading_count": len(heading_rows),
        "directive_row_count": len(directive_rows),
        "tag_counts": dict(sorted(tag_counts.items())),
        "top_section_directive_counts": top_sections[:20],
        "traceability_summary": {
            "status": traceability.get("status"),
            "goal_line_count": traceability.get("goal_line_count"),
            "heading_count": traceability.get("heading_count"),
            "trace_row_count": traceability.get("trace_row_count"),
            "status_counts": traceability.get("status_counts"),
            "missing_evidence_rows": len(traceability.get("missing_evidence_rows", [])),
        },
        "requirement_matrix_summary": {
            "status": requirement_matrix.get("status"),
            "goal_line_count": requirement_matrix.get("goal_line_count"),
            "requirement_row_count": requirement_matrix.get("requirement_row_count"),
            "status_counts": requirement_matrix.get("status_counts"),
            "missing_evidence_rows": len(requirement_matrix.get("missing_evidence_rows", [])),
        },
        "checks": {
            "goal_md_exists": GOAL.is_file(),
            "line_count_matches_goal_traceability": len(lines) == traceability.get("goal_line_count"),
            "heading_count_matches_goal_traceability": len(heading_rows) == traceability.get("heading_count"),
            "has_many_directive_rows": len(directive_rows) >= 250,
            "has_mandatory_rows": tag_counts["mandatory"] >= 40,
            "has_prohibition_rows": tag_counts["prohibition"] >= 40,
            "has_deliverable_rows": tag_counts["deliverable"] >= 50,
            "has_execution_rows": tag_counts["execution"] >= 80,
            "traceability_has_no_missing_evidence": len(traceability.get("missing_evidence_rows", [])) == 0,
            "requirement_matrix_has_no_missing_evidence": len(requirement_matrix.get("missing_evidence_rows", [])) == 0,
            "does_not_claim_goal_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This index proves the current audit chain has read and categorized directive-bearing goal.md lines. "
                "It does not satisfy incomplete paper-level requirements such as live Isaac/ROS execution, true DAgger, "
                "trained checkpoints, Fig. 5/6 reproduction, videos, or hardware deployment."
            ),
        },
        "outputs": {
            "json": str(OUT / "goal_directive_index_audit.json"),
            "directives_tsv": str(OUT / "goal_directive_rows.tsv"),
            "headings_tsv": str(OUT / "goal_heading_rows.tsv"),
            "markdown": str(OUT / "goal_directive_index_audit.md"),
        },
    }

    (OUT / "goal_directive_index_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_tsv(
        OUT / "goal_directive_rows.tsv",
        directive_rows,
        ["line", "section_id", "heading", "tags", "text"],
    )
    write_tsv(OUT / "goal_heading_rows.tsv", heading_rows, ["line", "heading", "section_id"])

    md_lines = [
        "# goal.md Directive Index Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Lines: `{summary['line_count']}`",
        f"- Headings: `{summary['heading_count']}`",
        f"- Directive rows: `{summary['directive_row_count']}`",
        f"- Tag counts: `{json.dumps(summary['tag_counts'], sort_keys=True)}`",
        "",
        "## Top Sections",
    ]
    for row in top_sections[:20]:
        md_lines.append(f"- `{row['section_id']}`: `{row['directive_count']}`")
    md_lines.extend(["", "## Outputs"])
    for key, value in summary["outputs"].items():
        md_lines.append(f"- `{key}`: `{value}`")
    (OUT / "goal_directive_index_audit.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": summary["status"],
                "lines": summary["line_count"],
                "headings": summary["heading_count"],
                "directives": summary["directive_row_count"],
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
