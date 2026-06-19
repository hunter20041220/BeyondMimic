#!/usr/bin/env python3
"""Index report-ready visual evidence without promoting it to paper-level results."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/report_assets/visual_evidence_index"
VIS_ROOT = ROOT / "res/visualization"
REPORT_ROOT = ROOT / "res/report_assets"

VISUAL_SUFFIXES = {".mp4", ".png", ".gif"}
TABLE_SUFFIXES = {".csv", ".tsv", ".md"}
ASSET_JSON_PATTERNS = [
    "res/visualization/*/*asset*.json",
    "res/visualization/*/*/*asset*.json",
    "res/report_assets/*/*assets*.json",
    "res/report_assets/*/*report_assets*.json",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_info(value: str) -> dict[str, Any]:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    exists = path.exists()
    suffix = path.suffix.lower()
    if suffix in VISUAL_SUFFIXES:
        kind = "visual"
    elif suffix in TABLE_SUFFIXES:
        kind = "table_or_readme"
    elif suffix == ".json":
        kind = "json"
    else:
        kind = "other"
    return {
        "path": str(path),
        "relative_path": rel(path) if path.is_absolute() and path.is_relative_to(ROOT) else str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists and path.is_file() else 0,
        "suffix": suffix,
        "kind": kind,
        "github_commit_policy": "do_not_commit_large_video" if suffix == ".mp4" else "small_asset_ok_if_tracked",
    }


def iter_asset_jsons() -> list[Path]:
    paths: set[Path] = set()
    for pattern in ASSET_JSON_PATTERNS:
        paths.update(ROOT.glob(pattern))
    return sorted(p for p in paths if p.is_file())


def collect_asset_paths(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for container_name in ("assets", "outputs", "artifacts"):
        container = payload.get(container_name)
        if not isinstance(container, dict):
            continue
        for key, value in sorted(container.items()):
            if isinstance(value, str):
                info = file_info(value)
                if info["kind"] in {"visual", "table_or_readme", "json"}:
                    info["asset_key"] = key
                    info["asset_container"] = container_name
                    rows.append(info)
            elif isinstance(value, dict):
                for subkey, subvalue in sorted(value.items()):
                    if isinstance(subvalue, str):
                        info = file_info(subvalue)
                        if info["kind"] in {"visual", "table_or_readme", "json"}:
                            info["asset_key"] = f"{key}.{subkey}"
                            info["asset_container"] = container_name
                            rows.append(info)
    return rows


def no_overclaim_checks(payload: dict[str, Any]) -> dict[str, bool]:
    checks = payload.get("checks", {})
    interpretation = payload.get("interpretation", {})
    claim_level = str(payload.get("claim_level") or interpretation.get("claim_level") or payload.get("experiment_type", ""))
    text = json.dumps({"checks": checks, "interpretation": interpretation, "claim": claim_level})
    lowered = text.lower()
    claim_lower = claim_level.lower()
    local_or_limited_claim = any(
        marker in claim_lower
        for marker in [
            "local_virtual",
            "resource_adjusted",
            "qualitative_only",
            "report_asset",
            "report_assets",
            "visualization",
        ]
    )
    return {
        "does_not_claim_paper_level": (
            checks.get("does_not_claim_paper_level") is True
            or checks.get("does_not_claim_paper_level_eval") is True
            or checks.get("does_not_claim_fig5_fig6") is True
            or checks.get("does_not_claim_paper_fig5_fig6") is True
            or checks.get("does_not_claim_official_unpatched_output") is True
            or checks.get("does_not_claim_official_checkpoint") is True
            or checks.get("does_not_claim_closed_loop_guidance") is True
            or checks.get("does_not_claim_closed_loop") is True
            or checks.get("does_not_claim_official_dagger") is True
            or local_or_limited_claim
            or "not paper" in lowered
            or "not official" in lowered
        ),
        "does_not_claim_real_robot": checks.get("does_not_claim_real_robot") is True
        or local_or_limited_claim
        or "not real-robot" in lowered
        or "real robot" not in lowered,
        "goal_complete_false": interpretation.get("goal_complete") is False
        or checks.get("does_not_claim_goal_complete") is True
        or local_or_limited_claim,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    asset_json_paths = iter_asset_jsons()
    for json_path in asset_json_paths:
        payload = load_json(json_path)
        asset_rows = collect_asset_paths(payload)
        status = str(payload.get("status", ""))
        claim_level = (
            payload.get("claim_level")
            or payload.get("interpretation", {}).get("claim_level")
            or payload.get("experiment_type", "")
        )
        overclaim = no_overclaim_checks(payload)
        visual_count = sum(1 for row in asset_rows if row["kind"] == "visual")
        video_count = sum(1 for row in asset_rows if row["suffix"] == ".mp4")
        png_count = sum(1 for row in asset_rows if row["suffix"] == ".png")
        table_count = sum(1 for row in asset_rows if row["kind"] == "table_or_readme")
        all_assets_exist = all(row["exists"] and row["size_bytes"] > 0 for row in asset_rows) if asset_rows else False
        row = {
            "asset_json": str(json_path),
            "relative_asset_json": rel(json_path),
            "status": status,
            "claim_level": claim_level,
            "visual_count": visual_count,
            "video_count": video_count,
            "png_count": png_count,
            "table_count": table_count,
            "all_assets_exist": all_assets_exist,
            "does_not_claim_paper_level": overclaim["does_not_claim_paper_level"],
            "does_not_claim_real_robot": overclaim["does_not_claim_real_robot"],
            "goal_complete_false": overclaim["goal_complete_false"],
            "scope": payload.get("scope") or payload.get("interpretation", {}).get("why_not_complete", ""),
            "assets": asset_rows,
        }
        rows.append(row)

    mp4_rows = [
        asset
        | {
            "source_asset_json": row["relative_asset_json"],
            "claim_level": row["claim_level"],
            "does_not_claim_paper_level": row["does_not_claim_paper_level"],
            "does_not_claim_real_robot": row["does_not_claim_real_robot"],
        }
        for row in rows
        for asset in row["assets"]
        if asset["suffix"] == ".mp4"
    ]
    png_rows = [
        asset
        | {
            "source_asset_json": row["relative_asset_json"],
            "claim_level": row["claim_level"],
        }
        for row in rows
        for asset in row["assets"]
        if asset["suffix"] == ".png"
    ]
    table_rows = [
        asset
        | {
            "source_asset_json": row["relative_asset_json"],
            "claim_level": row["claim_level"],
        }
        for row in rows
        for asset in row["assets"]
        if asset["kind"] == "table_or_readme"
    ]

    checks = {
        "asset_json_count_positive": len(rows) > 0,
        "all_indexed_asset_jsons_exist": all(Path(row["asset_json"]).is_file() for row in rows),
        "all_indexed_assets_exist": all(row["all_assets_exist"] for row in rows),
        "has_report_ready_videos": len(mp4_rows) > 0,
        "has_report_ready_pngs": len(png_rows) > 0,
        "has_metric_tables_or_readmes": len(table_rows) > 0,
        "all_video_rows_marked_do_not_commit_large_video": all(
            row["github_commit_policy"] == "do_not_commit_large_video" for row in mp4_rows
        ),
        "all_rows_avoid_paper_level_overclaim": all(row["does_not_claim_paper_level"] for row in rows),
        "all_rows_avoid_real_robot_overclaim": all(row["does_not_claim_real_robot"] for row in rows),
        "all_rows_keep_goal_incomplete": all(row["goal_complete_false"] for row in rows),
        "does_not_claim_goal_complete": True,
    }
    status = "ok" if all(checks.values()) else "needs_review"
    summary = {
        "status": status,
        "experiment_type": "visual_evidence_index",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_level": "report_visual_index_only",
        "metrics": {
            "asset_json_count": len(rows),
            "report_ready_video_count": len(mp4_rows),
            "report_ready_png_count": len(png_rows),
            "table_or_readme_asset_count": len(table_rows),
            "total_video_size_bytes": sum(row["size_bytes"] for row in mp4_rows),
            "total_png_size_bytes": sum(row["size_bytes"] for row in png_rows),
        },
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This is an index of local report/PPT visual evidence. It records videos, plots, tables, claim levels, "
                "and limitations, but it does not create new paper-level closed-loop results or real-robot evidence."
            ),
        },
        "rows": rows,
        "mp4_rows": mp4_rows,
        "outputs": {
            "json": str(OUT / "visual_evidence_index.json"),
            "csv": str(OUT / "visual_evidence_index.csv"),
            "md": str(OUT / "visual_evidence_index.md"),
        },
    }

    json_path = OUT / "visual_evidence_index.json"
    csv_path = OUT / "visual_evidence_index.csv"
    md_path = OUT / "visual_evidence_index.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "relative_asset_json",
            "status",
            "claim_level",
            "visual_count",
            "video_count",
            "png_count",
            "table_count",
            "all_assets_exist",
            "does_not_claim_paper_level",
            "does_not_claim_real_robot",
            "goal_complete_false",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})
    lines = [
        "# Visual Evidence Index",
        "",
        "This index lists local report/PPT visual evidence and preserves claim boundaries.",
        "",
        f"- Status: `{status}`",
        f"- Asset JSON files indexed: `{len(rows)}`",
        f"- Report-ready MP4 files: `{len(mp4_rows)}`",
        f"- Report-ready PNG files: `{len(png_rows)}`",
        f"- Table/README assets: `{len(table_rows)}`",
        "",
        "Large videos are intentionally not committed to GitHub; their paths and claim levels are recorded here.",
        "",
        "## Videos",
        "",
        "| Video | Claim Level | Size Bytes | Source Asset |",
        "|---|---:|---:|---|",
    ]
    for row in mp4_rows:
        lines.append(
            f"| `{row['relative_path']}` | `{row['claim_level']}` | `{row['size_bytes']}` | "
            f"`{row['source_asset_json']}` |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "This project still does not fully reproduce BeyondMimic at paper level. These videos and plots are local "
            "virtual/resource-adjusted evidence, not official Fig. 5/Fig. 6 results and not real-robot evidence.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": status, "json": str(json_path), "videos": len(mp4_rows)}, sort_keys=True))
    if status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
