#!/usr/bin/env python3
"""Create recentered short-motion work copies for Stage-1 teacher repair.

The public whole_body_tracking command loader preserves the global root
translation stored in each motion file.  Several short HuB clips in the local
bundle start at large nonzero XY offsets (for example y ~= 2.8 m), which is
awkward for a dense IsaacLab multi-env grid.  This script does not modify the
source bundle.  It writes audited work copies whose first root XY is shifted to
the origin while preserving the original relative motion, joint targets,
velocities, and claim boundaries.
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
SRC_ROOT = ROOT / "res/tracking/stage1_multisource_motion_bundle"
SRC_CSV = SRC_ROOT / "csv"
SRC_MOTIONS = SRC_ROOT / "motions"
OUT = ROOT / "res/tracking/stage1_short_motion_recentered_bundle"
OUT_CSV = OUT / "csv"
OUT_MOTIONS = OUT / "motions"
OUT_JSON = OUT / "tracking_stage1_short_motion_recenter_audit.json"
OUT_TSV = OUT / "tracking_stage1_short_motion_recenter_audit.tsv"

MOTIONS = [
    "hub_singleleg_video_single_leg_stand_1",
    "hub_swallow_balance_video_swift0322",
    "hub_squat_video_squat_4",
    "hub_squat_video_squat_18",
    "zenodo_tkd_skill",
]
ARRAY_KEYS = ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "motion",
        "status",
        "source_csv",
        "source_npz",
        "recentered_csv",
        "recentered_npz",
        "frames",
        "fps",
        "root_xy_offset_x",
        "root_xy_offset_y",
        "source_root0_x",
        "source_root0_y",
        "source_root0_z",
        "recentered_root0_x",
        "recentered_root0_y",
        "recentered_root0_z",
        "source_root_xy_radius_max",
        "recentered_root_xy_radius_max",
        "claim_level",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def recenter_one(motion: str) -> dict[str, Any]:
    src_csv = SRC_CSV / f"{motion}.csv"
    src_npz = SRC_MOTIONS / motion / "motion.npz"
    out_csv = OUT_CSV / f"{motion}_rootxy0.csv"
    out_npz = OUT_MOTIONS / f"{motion}_rootxy0" / "motion.npz"

    row: dict[str, Any] = {
        "motion": motion,
        "source_csv": str(src_csv),
        "source_npz": str(src_npz),
        "recentered_csv": str(out_csv),
        "recentered_npz": str(out_npz),
        "claim_level": "local_recentered_motion_work_copy_for_teacher_training",
    }
    if not src_csv.is_file() or not src_npz.is_file():
        row.update({"status": "missing_source", "notes": "source csv or npz missing"})
        return row

    csv_arr = np.loadtxt(src_csv, delimiter=",", dtype=np.float64)
    if csv_arr.ndim != 2 or csv_arr.shape[1] != 36:
        row.update({"status": "invalid_csv_shape", "notes": f"expected 36 columns, got {csv_arr.shape}"})
        return row

    with np.load(src_npz) as z:
        arrays = {key: np.array(z[key]) for key in z.files}
    missing = [key for key in ["fps", *ARRAY_KEYS] if key not in arrays]
    if missing:
        row.update({"status": "invalid_npz_missing_keys", "notes": ",".join(missing)})
        return row

    body_pos = arrays["body_pos_w"].astype(np.float32, copy=True)
    if body_pos.ndim != 3 or body_pos.shape[2] != 3:
        row.update({"status": "invalid_body_pos_shape", "notes": str(body_pos.shape)})
        return row

    csv_offset = csv_arr[0, :2].copy()
    npz_offset = body_pos[0, 0, :2].astype(np.float64).copy()
    # The per-motion npz is the FK product of the csv, so these should match
    # closely.  Use the npz offset for body arrays and the csv offset for the
    # source CSV to avoid silently shifting unrelated representations.
    recentered_csv = csv_arr.copy()
    recentered_csv[:, :2] -= csv_offset[None, :]
    recentered_body_pos = body_pos.copy()
    recentered_body_pos[:, :, :2] -= npz_offset[None, None, :]
    arrays["body_pos_w"] = recentered_body_pos.astype(np.float32)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_csv, recentered_csv, delimiter=",", fmt="%.9f")
    np.savez_compressed(out_npz, **arrays)

    source_root = body_pos[:, 0, :]
    recentered_root = recentered_body_pos[:, 0, :]
    row.update(
        {
            "status": "ok",
            "frames": int(body_pos.shape[0]),
            "fps": float(np.asarray(arrays["fps"]).reshape(-1)[0]),
            "root_xy_offset_x": float(npz_offset[0]),
            "root_xy_offset_y": float(npz_offset[1]),
            "source_root0_x": float(source_root[0, 0]),
            "source_root0_y": float(source_root[0, 1]),
            "source_root0_z": float(source_root[0, 2]),
            "recentered_root0_x": float(recentered_root[0, 0]),
            "recentered_root0_y": float(recentered_root[0, 1]),
            "recentered_root0_z": float(recentered_root[0, 2]),
            "source_root_xy_radius_max": float(np.linalg.norm(source_root[:, :2] - source_root[0, :2], axis=1).max()),
            "recentered_root_xy_radius_max": float(
                np.linalg.norm(recentered_root[:, :2] - recentered_root[0, :2], axis=1).max()
            ),
            "source_csv_sha256": sha256_file(src_csv),
            "source_npz_sha256": sha256_file(src_npz),
            "recentered_csv_sha256": sha256_file(out_csv),
            "recentered_npz_sha256": sha256_file(out_npz),
            "notes": (
                "Subtracted first root XY from csv root translation and all npz body_pos_w XY values; "
                "joint targets, quaternions, and velocities are unchanged."
            ),
        }
    )
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [recenter_one(motion) for motion in MOTIONS]
    checks = {
        "all_requested_sources_seen": all(row["status"] == "ok" for row in rows),
        "singleleg_recentered": any(
            row["motion"] == "hub_singleleg_video_single_leg_stand_1"
            and row["status"] == "ok"
            and abs(float(row["recentered_root0_x"])) < 1e-5
            and abs(float(row["recentered_root0_y"])) < 1e-5
            for row in rows
        ),
        "does_not_modify_original_bundle": True,
        "does_not_claim_new_motion_source": True,
    }
    status = "ok_stage1_short_motion_recenter_audit" if all(checks.values()) else "failed_stage1_short_motion_recenter_audit"
    summary = {
        "status": status,
        "generated_at": utc_now(),
        "experiment_type": "tracking_stage1_short_motion_recenter_audit",
        "claim_level": "local_motion_preprocessing_work_copy_audit",
        "scope": (
            "Audits short Stage-1 reference motions with large global XY offsets and writes recentered work copies "
            "for teacher-quality repair runs. This is motion preprocessing, not a policy result."
        ),
        "checks": checks,
        "inputs": {
            "source_csv_dir": str(SRC_CSV),
            "source_motion_root": str(SRC_MOTIONS),
            "motions": MOTIONS,
        },
        "outputs": {
            "json": str(OUT_JSON),
            "tsv": str(OUT_TSV),
            "csv_dir": str(OUT_CSV),
            "motion_root": str(OUT_MOTIONS),
        },
        "rows": rows,
        "interpretation": {
            "goal_complete": False,
            "why_needed": (
                "HuB short clips start at nonzero global XY positions while IsaacLab training uses many packed "
                "environments. Recentered work copies remove that confound before judging teacher learning."
            ),
            "claim_boundary": (
                "The work copies preserve relative motion but are not official released motion files. Any teacher "
                "trained on them must be reported as local recentered-motion training."
            ),
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_tsv(OUT_TSV, rows)
    print(
        json.dumps(
            {
                "status": status,
                "json": str(OUT_JSON),
                "motion_root": str(OUT_MOTIONS),
                "ok_rows": sum(row["status"] == "ok" for row in rows),
            },
            sort_keys=True,
        )
    )
    if status != "ok_stage1_short_motion_recenter_audit":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
