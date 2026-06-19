#!/usr/bin/env python3
"""Build a full public-motion bundle NPZ for the official tracking MotionLoader.

The official Tracking-Flat-G1-v0 MotionLoader accepts one NPZ path. The local
full public-motion csv_to_npz audit produced 40 per-motion NPZ files. This
script concatenates those 40 official-loop NPZ outputs into a single project
local training/evaluation NPZ and records clip boundaries so downstream runs can
state exactly what was done.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SOURCE_AUDIT = (
    ROOT
    / "res/tracking/official_csv_to_npz_loop_full_dataset_with_enriched_usd/"
    "tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_audit.json"
)
REPLAY_AUDIT = (
    ROOT
    / "res/tracking/official_replay_npz_loop_full_dataset_with_enriched_usd/"
    "tracking_official_replay_npz_loop_full_dataset_with_enriched_usd_audit.json"
)
TASK_EVAL_AUDIT = (
    ROOT
    / "res/tracking/g1_official_csv_loop_full_dataset_task_eval/"
    "tracking_g1_official_csv_loop_full_dataset_task_eval.json"
)
OUT = ROOT / "res/tracking/official_csv_loop_full_bundle_motion_npz"
OUT_NPZ = OUT / "official_csv_loop_full_public_motion_bundle.npz"
OUT_JSON = OUT / "tracking_g1_official_csv_loop_full_bundle_motion_npz.json"
OUT_CLIPS_CSV = OUT / "official_csv_loop_full_public_motion_bundle_clips.csv"
OUT_CLIPS_TSV = OUT / "official_csv_loop_full_public_motion_bundle_clips.tsv"

ARRAY_KEYS = [
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_rows(path: Path, rows: list[dict[str, Any]], delimiter: str) -> None:
    fieldnames = [
        "motion",
        "source_npz",
        "source_sha256",
        "start_frame",
        "end_frame_exclusive",
        "frame_count",
        "fps",
        "joint_shape",
        "body_shape",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = load_json(SOURCE_AUDIT)
    replay = load_json(REPLAY_AUDIT)
    task_eval = load_json(TASK_EVAL_AUDIT)

    arrays: dict[str, list[np.ndarray]] = {key: [] for key in ARRAY_KEYS}
    clip_rows: list[dict[str, Any]] = []
    fps_values: list[int] = []
    cursor = 0

    source_rows = source.get("rows", [])
    for row in source_rows:
        motion = row["motion"]
        npz_path = Path(row["output_npz"])
        data = np.load(npz_path)
        frame_count = int(data["joint_pos"].shape[0])
        fps = int(np.asarray(data["fps"]).reshape(-1)[0])
        fps_values.append(fps)
        for key in ARRAY_KEYS:
            arrays[key].append(np.asarray(data[key]))
        clip_rows.append(
            {
                "motion": motion,
                "source_npz": str(npz_path),
                "source_sha256": sha256_file(npz_path),
                "start_frame": cursor,
                "end_frame_exclusive": cursor + frame_count,
                "frame_count": frame_count,
                "fps": fps,
                "joint_shape": list(data["joint_pos"].shape),
                "body_shape": list(data["body_pos_w"].shape),
            }
        )
        cursor += frame_count

    bundle = {key: np.concatenate(value, axis=0) for key, value in arrays.items()}
    bundle["fps"] = np.asarray([fps_values[0] if fps_values else 50], dtype=np.int64)
    np.savez(OUT_NPZ, **bundle)

    write_rows(OUT_CLIPS_CSV, clip_rows, ",")
    write_rows(OUT_CLIPS_TSV, clip_rows, "\t")

    boundary_jumps = []
    for prev, cur in zip(clip_rows, clip_rows[1:]):
        prev_end = int(prev["end_frame_exclusive"]) - 1
        cur_start = int(cur["start_frame"])
        boundary_jumps.append(
            {
                "from_motion": prev["motion"],
                "to_motion": cur["motion"],
                "frame_before": prev_end,
                "frame_after": cur_start,
                "root_position_jump": float(
                    np.linalg.norm(bundle["body_pos_w"][cur_start, 0] - bundle["body_pos_w"][prev_end, 0])
                ),
                "joint_position_jump": float(
                    np.linalg.norm(bundle["joint_pos"][cur_start] - bundle["joint_pos"][prev_end])
                ),
            }
        )

    summary = {
        "status": "ok_official_csv_loop_full_bundle_motion_npz",
        "experiment_type": "tracking_g1_official_csv_loop_full_bundle_motion_npz",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Concatenates the 40 official csv_to_npz.py loop outputs into one MotionLoader-compatible NPZ for local "
            "full-public-motion PPO/evaluation runs. This is a project-local bundle representation, not an official "
            "BeyondMimic multi-motion sampler."
        ),
        "inputs": {
            "source_audit": str(SOURCE_AUDIT),
            "replay_audit": str(REPLAY_AUDIT),
            "task_eval_audit": str(TASK_EVAL_AUDIT),
        },
        "outputs": {
            "npz": str(OUT_NPZ),
            "json": str(OUT_JSON),
            "clips_csv": str(OUT_CLIPS_CSV),
            "clips_tsv": str(OUT_CLIPS_TSV),
        },
        "bundle": {
            "motion_count": len(clip_rows),
            "total_frames": int(bundle["joint_pos"].shape[0]),
            "fps": int(bundle["fps"][0]),
            "joint_pos_shape": list(bundle["joint_pos"].shape),
            "joint_vel_shape": list(bundle["joint_vel"].shape),
            "body_pos_w_shape": list(bundle["body_pos_w"].shape),
            "body_quat_w_shape": list(bundle["body_quat_w"].shape),
            "body_lin_vel_w_shape": list(bundle["body_lin_vel_w"].shape),
            "body_ang_vel_w_shape": list(bundle["body_ang_vel_w"].shape),
            "npz_size_bytes": OUT_NPZ.stat().st_size,
            "npz_sha256": sha256_file(OUT_NPZ),
            "boundary_count": len(boundary_jumps),
            "max_root_position_boundary_jump": max((x["root_position_jump"] for x in boundary_jumps), default=0.0),
            "max_joint_position_boundary_jump": max((x["joint_position_jump"] for x in boundary_jumps), default=0.0),
        },
        "clip_rows": clip_rows,
        "boundary_jumps": boundary_jumps,
        "checks": {
            "source_full_csv_audit_passed": source.get("status")
            == "ok_official_csv_to_npz_loop_full_dataset_with_enriched_usd",
            "replay_full_dataset_audit_passed": replay.get("status")
            == "ok_official_replay_npz_loop_full_dataset_with_enriched_usd",
            "task_full_dataset_audit_passed": task_eval.get("status") == "ok_official_csv_loop_full_dataset_task_eval",
            "all_40_source_rows_used": len(clip_rows) == 40,
            "all_source_rows_ok": all(row.get("status") == "ok" for row in source_rows),
            "all_fps_50": set(fps_values) == {50},
            "total_frames_11960": int(bundle["joint_pos"].shape[0]) == 11960,
            "joint_shape_11960_29": list(bundle["joint_pos"].shape) == [11960, 29],
            "body_shape_11960_40_3": list(bundle["body_pos_w"].shape) == [11960, 40, 3],
            "npz_written": OUT_NPZ.is_file() and OUT_NPZ.stat().st_size > 0,
            "clip_boundary_manifest_written": OUT_CLIPS_CSV.is_file() and OUT_CLIPS_TSV.is_file(),
            "does_not_claim_official_multimotion_sampler": True,
            "does_not_claim_paper_level_training": True,
            "does_not_claim_real_robot": True,
        },
        "interpretation": {
            "goal_complete": False,
            "claim_level": "local_full_public_motion_bundle_for_official_motionloader",
            "why_not_paper_level": (
                "The official local MotionLoader accepts one NPZ path. Concatenating clips gives a full-public-motion "
                "training input without patching official loader code, but clip boundaries are artificial and this is "
                "not the paper's original teacher motion sampler, official DAgger dataset, or unpatched official asset path."
            ),
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(OUT_JSON), "npz": str(OUT_NPZ)}, sort_keys=True))


if __name__ == "__main__":
    main()
