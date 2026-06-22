#!/usr/bin/env python3
"""Build a manifest for official released-data MuJoCo MP4 outputs."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
PKG = ROOT / "official_mp4"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def ffprobe(path: Path) -> dict[str, Any]:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=nb_frames,width,height,avg_frame_rate,duration",
            "-show_entries",
            "format=duration,size",
            "-of",
            "json",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return {"returncode": proc.returncode, "stderr": proc.stderr.strip()}
    return json.loads(proc.stdout)


def main() -> None:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted((PKG / "res").glob("*/*_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        outputs = summary.get("outputs", {})
        for kind, out_path in outputs.items():
            path = Path(out_path)
            if not path.exists():
                continue
            row = {
                "path": str(path),
                "relative_path": str(path.relative_to(ROOT)),
                "exists": path.exists(),
                "file_size": path.stat().st_size,
                "sha256": sha256(path),
                "artifact_kind": kind,
                "motion_name": summary.get("motion_name", summary_path.parent.name),
                "experiment_type": summary.get("experiment_type", ""),
                "source": summary.get("source_csv") or summary.get("source_mcap") or "",
                "frames_rendered": summary.get("frames_rendered", ""),
                "backend": summary.get("backend", ""),
                "claim_level": summary.get("claim_level", ""),
                "do_not_commit_large_video": kind == "mp4",
                "notes": "official released-data MuJoCo visualization; not closed-loop policy reproduction",
            }
            if kind == "mp4":
                probe = ffprobe(path)
                streams = probe.get("streams") or [{}]
                fmt = probe.get("format") or {}
                row.update(
                    {
                        "ffprobe_width": streams[0].get("width", ""),
                        "ffprobe_height": streams[0].get("height", ""),
                        "ffprobe_frames": streams[0].get("nb_frames", ""),
                        "ffprobe_duration": streams[0].get("duration") or fmt.get("duration", ""),
                        "ffprobe_avg_frame_rate": streams[0].get("avg_frame_rate", ""),
                    }
                )
            rows.append(row)

    fieldnames = [
        "path",
        "relative_path",
        "exists",
        "file_size",
        "sha256",
        "artifact_kind",
        "motion_name",
        "experiment_type",
        "source",
        "frames_rendered",
        "backend",
        "ffprobe_width",
        "ffprobe_height",
        "ffprobe_frames",
        "ffprobe_duration",
        "ffprobe_avg_frame_rate",
        "claim_level",
        "do_not_commit_large_video",
        "notes",
    ]
    out_json = PKG / "official_mp4_manifest.json"
    out_tsv = PKG / "official_mp4_manifest.tsv"
    with out_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})

    mp4_rows = [row for row in rows if row["artifact_kind"] == "mp4"]
    payload = {
        "status": "ok_official_mp4_manifest",
        "timestamp_utc": utc_now(),
        "row_count": len(rows),
        "mp4_count": len(mp4_rows),
        "mp4_total_size_bytes": sum(int(row["file_size"]) for row in mp4_rows),
        "motion_names": sorted({row["motion_name"] for row in rows}),
        "claim_level": "official released-data MuJoCo visualization manifest; not policy closed-loop and not paper-level completion",
        "checks": {
            "all_paths_exist": all(row["exists"] for row in rows),
            "has_tkd_skill_reference_replay": any(row["motion_name"] == "official_zenodo_tkd_skill" and row["artifact_kind"] == "mp4" for row in rows),
            "has_four_agile_mcap_replays": sum(1 for row in mp4_rows if str(row["motion_name"]).startswith("official_agile_")) == 4,
            "has_fifteen_ablation_mcap_replays": sum(1 for row in mp4_rows if str(row["motion_name"]).startswith("official_ablation_")) == 15,
            "has_walk_and_run_mcap_replays": sum(1 for row in mp4_rows if str(row["motion_name"]).startswith(("official_walk_", "official_run_"))) == 2,
            "has_all_known_official_video_sources": len(mp4_rows) == 22,
            "all_mp4_have_ffprobe_frames": all(str(row.get("ffprobe_frames", "")).isdigit() for row in mp4_rows),
            "all_rows_avoid_policy_claim": all("not policy closed-loop" in str(row["claim_level"]) or "not closed-loop policy" in str(row["claim_level"]) for row in rows),
        },
        "outputs": {"json": str(out_json), "tsv": str(out_tsv)},
        "rows": rows,
    }
    write_json(out_json, payload)
    print(json.dumps({"status": payload["status"], "mp4_count": payload["mp4_count"], "rows": payload["row_count"]}))


if __name__ == "__main__":
    main()
