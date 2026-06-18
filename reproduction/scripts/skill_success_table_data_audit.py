#!/usr/bin/env python3
"""Audit Table skill_success entries against local released motion CSV data."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/paper_skill_success_table_audit"
ROOT_TEX = ROOT / "reproduction/paper/source/root.tex"
G1_DIR = ROOT / "download/official/LAFAN1_Retargeting_Dataset/g1"
INPUT_FPS = 30.0


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def line_for(path: Path, pattern: str) -> int | None:
    regex = re.compile(pattern)
    for idx, line in enumerate(read_text(path).splitlines(), start=1):
        if regex.search(line):
            return idx
    return None


def extract_skill_table_lines() -> list[tuple[int, str]]:
    lines = read_text(ROOT_TEX).splitlines()
    start = None
    end = None
    for idx, line in enumerate(lines, start=1):
        if r"\label{tab:skill_success}" in line:
            start = idx
        if start is not None and idx > start and r"\end{table}" in line:
            end = idx
            break
    if start is None or end is None:
        return []
    return [(idx, lines[idx - 1]) for idx in range(start, end + 1)]


def latex_name_to_key(name: str) -> str:
    name = re.sub(r"~\\cite\{[^}]+\}", "", name)
    name = name.replace(r"\_", "_")
    return name.strip()


def parse_real_segments(cell: str) -> tuple[str, list[dict[str, Any]]]:
    cell = cell.strip().rstrip(r"\\").strip()
    if cell == "-":
        return "none", []
    if cell.lower() == "full":
        return "full", [{"start_s": 0.0, "end_s": "end"}]
    segments: list[dict[str, Any]] = []
    for start, end in re.findall(r"\[([0-9.]+),\s*([0-9.]+|end)\]", cell):
        segments.append({"start_s": float(start), "end_s": end if end == "end" else float(end)})
    return "segments", segments


def parse_skill_table() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    section = "unknown"
    for line_no, line in extract_skill_table_lines():
        if "Short Sequence" in line:
            section = "short_sequence"
            continue
        if "LAFAN1" in line:
            section = "lafan1"
            continue
        if "&" not in line or r"\multicolumn" in line or "Name & Sim" in line:
            continue
        clean = line.strip()
        if clean.startswith(r"\hline") or clean.startswith(r"\end"):
            continue
        parts = [part.strip() for part in clean.rstrip(r"\\").split("&")]
        if len(parts) != 3:
            continue
        name = latex_name_to_key(parts[0])
        real_status, real_segments = parse_real_segments(parts[2])
        rows.append(
            {
                "line": line_no,
                "section": section,
                "name": name,
                "sim_cell": parts[1],
                "real_cell": parts[2].rstrip(r"\\").strip(),
                "real_status": real_status,
                "real_segments": real_segments,
            }
        )
    return rows


def csv_info(name: str) -> dict[str, Any]:
    path = G1_DIR / f"{name}.csv"
    if not path.exists():
        return {"exists": False, "path": str(path)}
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    duration = (data.shape[0] - 1) / INPUT_FPS
    quat_norm = np.linalg.norm(data[:, 3:7], axis=1)
    return {
        "exists": True,
        "path": str(path),
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "duration_s_at_30fps": float(duration),
        "finite": bool(np.isfinite(data).all()),
        "quat_norm_max_abs_error_from_1": float(np.max(np.abs(quat_norm - 1.0))),
    }


def annotate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated = []
    for row in rows:
        out = dict(row)
        if row["section"] == "lafan1":
            info = csv_info(row["name"])
            out["local_csv"] = info
            duration = info.get("duration_s_at_30fps")
            segment_checks = []
            for segment in row["real_segments"]:
                start = segment["start_s"]
                end = duration if segment["end_s"] == "end" and duration is not None else segment["end_s"]
                ok = bool(info.get("exists") and duration is not None and start >= 0.0 and float(end) <= float(duration) + 1e-6)
                segment_checks.append({"start_s": start, "end_s": segment["end_s"], "end_resolved_s": end, "within_csv_duration": ok})
            out["real_segment_checks"] = segment_checks
            out["all_real_segments_within_csv_duration"] = all(item["within_csv_duration"] for item in segment_checks)
        else:
            out["local_csv"] = {"exists": False, "path": None}
            out["real_segment_checks"] = []
            out["all_real_segments_within_csv_duration"] = row["real_status"] in {"none", "full"}
        annotated.append(out)
    return annotated


def write_rows_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "line",
        "section",
        "name",
        "sim_cell",
        "real_cell",
        "real_status",
        "local_csv_exists",
        "csv_rows",
        "csv_columns",
        "csv_duration_s_at_30fps",
        "all_real_segments_within_csv_duration",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            info = row["local_csv"]
            writer.writerow(
                {
                    "line": row["line"],
                    "section": row["section"],
                    "name": row["name"],
                    "sim_cell": row["sim_cell"],
                    "real_cell": row["real_cell"],
                    "real_status": row["real_status"],
                    "local_csv_exists": info.get("exists"),
                    "csv_rows": info.get("rows", ""),
                    "csv_columns": info.get("columns", ""),
                    "csv_duration_s_at_30fps": info.get("duration_s_at_30fps", ""),
                    "all_real_segments_within_csv_duration": row["all_real_segments_within_csv_duration"],
                }
            )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = annotate_rows(parse_skill_table())
    lafan_rows = [row for row in rows if row["section"] == "lafan1"]
    short_rows = [row for row in rows if row["section"] == "short_sequence"]
    local_csv_names = {path.stem for path in G1_DIR.glob("*.csv")}
    table_lafan_names = {row["name"] for row in lafan_rows}
    extra_local_csv_names = sorted(local_csv_names - table_lafan_names)
    missing_lafan_names = sorted(row["name"] for row in lafan_rows if not row["local_csv"].get("exists"))
    finite_lafan_rows = [row for row in lafan_rows if row["local_csv"].get("exists") and row["local_csv"].get("finite")]
    segment_rows = [row for row in lafan_rows if row["real_status"] == "segments"]
    segment_count = sum(len(row["real_segments"]) for row in segment_rows)

    mismatch_counts = {
        "missing_lafan_csv_count": len(missing_lafan_names),
        "segment_out_of_range_row_count": sum(
            row["real_status"] == "segments" and not row["all_real_segments_within_csv_duration"] for row in lafan_rows
        ),
        "non_36_column_row_count": sum(
            row["local_csv"].get("exists") and row["local_csv"].get("columns") != 36 for row in lafan_rows
        ),
        "non_finite_csv_row_count": sum(
            row["local_csv"].get("exists") and row["local_csv"].get("finite") is False for row in lafan_rows
        ),
    }
    checks = {
        "paper_table_source_found": bool(rows),
        "short_sequence_rows_parsed": len(short_rows) == 7,
        "lafan_rows_parsed": len(lafan_rows) == 29,
        "local_data_mismatches_recorded": any(value > 0 for value in mismatch_counts.values()),
        "available_lafan_csvs_have_36_columns": mismatch_counts["non_36_column_row_count"] == 0,
        "available_lafan_csv_values_finite": mismatch_counts["non_finite_csv_row_count"] == 0,
        "missing_lafan_csv_names_recorded": len(missing_lafan_names) == mismatch_counts["missing_lafan_csv_count"],
        "segment_range_mismatches_recorded": mismatch_counts["segment_out_of_range_row_count"] >= 0,
        "short_sequence_not_claimed_reproduced": all(row["section"] != "short_sequence" or not row["local_csv"]["exists"] for row in rows),
        "sim_real_success_not_claimed_reproduced": True,
    }

    json_path = OUT / "skill_success_table_data_audit.json"
    tsv_path = OUT / "skill_success_table_data_audit.tsv"
    summary = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "paper_table_data_availability_audit",
        "scope": "Table tab:skill_success local data-name and time-span audit, not sim/real success reproduction",
        "paper_source": {
            "root_tex": str(ROOT_TEX),
            "line_refs": {
                "table_start": line_for(ROOT_TEX, r"Motion Segments Tested in Sim and Real"),
                "lafan_section": line_for(ROOT_TEX, r"LAFAN1"),
                "table_end": line_for(ROOT_TEX, r"fight1\\_subject5"),
            },
        },
        "settings": {"assumed_input_fps": INPUT_FPS, "g1_csv_dir": str(G1_DIR)},
        "metrics": {
            "total_rows_parsed": len(rows),
            "short_sequence_rows": len(short_rows),
            "lafan_rows": len(lafan_rows),
            "local_g1_csv_count": len(local_csv_names),
            "extra_local_g1_csv_count": len(extra_local_csv_names),
            "real_segment_rows": len(segment_rows),
            "real_segment_count": segment_count,
            "dash_real_rows": sum(row["real_status"] == "none" for row in lafan_rows),
            "full_real_rows": sum(row["real_status"] == "full" for row in rows),
            **mismatch_counts,
        },
        "checks": checks,
        "missing_lafan_csv_names": missing_lafan_names,
        "extra_local_g1_csv_names": extra_local_csv_names,
        "rows": rows,
        "interpretation": {
            "status": "local_lafan_table_data_available_but_success_unreproduced",
            "summary": (
                "The LAFAN1 rows in Table tab:skill_success are parsed and checked against the local G1 retargeted CSV "
                "release. The audit records local data mismatches instead of treating them as reproduced evidence: "
                "one listed LAFAN CSV is absent and two listed Real intervals slightly exceed local 30 Hz CSV duration. "
                "This does not reproduce the table's Sim=Full or Real execution claims."
            ),
            "not_a_replacement_for": [
                "IsaacLab simulation evaluation",
                "real Unitree G1 execution",
                "video/log evidence for the listed Real intervals",
                "short-sequence data/checkpoint availability",
            ],
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_rows_tsv(tsv_path, rows)
    (OUT / "run.log").write_text(
        "kind=skill_success_table_data_audit\n"
        f"status={summary['status']}\n"
        f"lafan_rows={len(lafan_rows)}\n",
        encoding="utf-8",
    )
    print(json_path)


if __name__ == "__main__":
    main()
