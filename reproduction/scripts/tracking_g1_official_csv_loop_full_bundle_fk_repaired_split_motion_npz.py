#!/usr/bin/env python3
"""Split the FK-repaired full public-motion bundle into per-motion NPZ files."""

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
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_motion_npz/"
    "tracking_g1_official_csv_loop_full_bundle_fk_repaired_motion_npz.json"
)
SOURCE_NPZ = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_motion_npz/"
    "official_csv_loop_full_public_motion_bundle_fk_repaired.npz"
)
OUT = ROOT / "res/tracking/official_csv_loop_full_bundle_fk_repaired_split_motion_npz"
MOTION_DIR = OUT / "motions"

ARRAY_KEYS = [
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def spread_summary(body_pos: np.ndarray) -> dict[str, float]:
    x_spread = body_pos[:, :, 0].max(axis=1) - body_pos[:, :, 0].min(axis=1)
    y_spread = body_pos[:, :, 1].max(axis=1) - body_pos[:, :, 1].min(axis=1)
    z_spread = body_pos[:, :, 2].max(axis=1) - body_pos[:, :, 2].min(axis=1)
    return {
        "x_spread_mean_m": float(np.mean(x_spread)),
        "y_spread_mean_m": float(np.mean(y_spread)),
        "z_spread_mean_m": float(np.mean(z_spread)),
        "z_spread_max_m": float(np.max(z_spread)),
    }


def write_table(path: Path, rows: list[dict[str, Any]], delimiter: str) -> None:
    fieldnames = [
        "motion",
        "frame_count",
        "start_frame",
        "end_frame_exclusive",
        "output_npz",
        "npz_size_bytes",
        "npz_sha256",
        "z_spread_mean_m",
        "z_spread_max_m",
        "left_ankle_mean_z_m",
        "right_ankle_mean_z_m",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not SOURCE_AUDIT.is_file():
        raise FileNotFoundError(SOURCE_AUDIT)
    if not SOURCE_NPZ.is_file():
        raise FileNotFoundError(SOURCE_NPZ)

    OUT.mkdir(parents=True, exist_ok=True)
    MOTION_DIR.mkdir(parents=True, exist_ok=True)
    audit = load_json(SOURCE_AUDIT)
    data = np.load(SOURCE_NPZ)
    clip_rows = audit["clip_rows"]
    fps = np.asarray(data["fps"]).copy()
    rows: list[dict[str, Any]] = []
    for clip in clip_rows:
        motion = clip["motion"]
        start = int(clip["start_frame"])
        end = int(clip["end_frame_exclusive"])
        motion_out = MOTION_DIR / motion
        motion_out.mkdir(parents=True, exist_ok=True)
        npz_path = motion_out / "motion.npz"
        arrays = {key: data[key][start:end].copy() for key in ARRAY_KEYS}
        arrays["fps"] = fps
        np.savez(npz_path, **arrays)
        body_pos = arrays["body_pos_w"]
        spread = spread_summary(body_pos)
        left_ankle_mean = float(body_pos[:, 7, 2].mean())
        right_ankle_mean = float(body_pos[:, 13, 2].mean())
        row = {
            "motion": motion,
            "frame_count": end - start,
            "start_frame": start,
            "end_frame_exclusive": end,
            "output_npz": str(npz_path),
            "npz_size_bytes": npz_path.stat().st_size,
            "npz_sha256": sha256(npz_path),
            "z_spread_mean_m": spread["z_spread_mean_m"],
            "z_spread_max_m": spread["z_spread_max_m"],
            "left_ankle_mean_z_m": left_ankle_mean,
            "right_ankle_mean_z_m": right_ankle_mean,
        }
        rows.append(row)

    total_frames = sum(int(row["frame_count"]) for row in rows)
    checks = {
        "source_status_ok": audit.get("status") == "ok_official_csv_loop_full_bundle_fk_repaired_motion_npz",
        "source_npz_exists": SOURCE_NPZ.is_file(),
        "all_40_motions_written": len(rows) == 40,
        "all_299_frames": all(int(row["frame_count"]) == 299 for row in rows),
        "total_frames_11960": total_frames == 11960,
        "all_outputs_exist": all(Path(row["output_npz"]).is_file() for row in rows),
        "all_outputs_non_degenerate_z_spread": all(float(row["z_spread_mean_m"]) > 0.5 for row in rows),
        "all_ankle_mean_z_below_0_25m": all(
            float(row["left_ankle_mean_z_m"]) < 0.25 and float(row["right_ankle_mean_z_m"]) < 0.25
            for row in rows
        ),
        "does_not_claim_official_csv_to_npz_output": True,
        "does_not_claim_paper_level_tracking": True,
        "does_not_claim_real_robot": True,
    }
    summary = {
        "status": "ok_fk_repaired_split_motion_npz" if all(checks.values()) else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_g1_official_csv_loop_full_bundle_fk_repaired_split_motion_npz",
        "scope": "Splits the local non-Kit FK-repaired full public-motion bundle into 40 per-motion MotionLoader NPZs for isolated IsaacLab task evaluation.",
        "source": {"audit": str(SOURCE_AUDIT), "npz": str(SOURCE_NPZ), "npz_sha256": sha256(SOURCE_NPZ)},
        "metrics": {
            "motion_count": len(rows),
            "total_frames": total_frames,
            "total_npz_size_bytes": sum(int(row["npz_size_bytes"]) for row in rows),
            "z_spread_mean_m_min": min(float(row["z_spread_mean_m"]) for row in rows),
            "z_spread_mean_m_max": max(float(row["z_spread_mean_m"]) for row in rows),
            "left_ankle_mean_z_m_max": max(float(row["left_ankle_mean_z_m"]) for row in rows),
            "right_ankle_mean_z_m_max": max(float(row["right_ankle_mean_z_m"]) for row in rows),
        },
        "checks": checks,
        "rows": rows,
        "outputs": {
            "json": str(OUT / "tracking_g1_official_csv_loop_full_bundle_fk_repaired_split_motion_npz.json"),
            "rows_csv": str(OUT / "tracking_g1_official_csv_loop_full_bundle_fk_repaired_split_motion_npz_rows.csv"),
            "rows_tsv": str(OUT / "tracking_g1_official_csv_loop_full_bundle_fk_repaired_split_motion_npz_rows.tsv"),
            "motion_root": str(MOTION_DIR),
        },
        "interpretation": {
            "claim_level": "local_nonkit_fk_repaired_per_motion_npz_candidates",
            "goal_complete": False,
            "not_paper_level_reasons": [
                "the body poses come from a local non-Kit URDF FK repair candidate",
                "the files are not unmodified official csv_to_npz.py outputs",
                "no PPO, DAgger, VAE/diffusion, TensorRT, or real-robot result is produced",
            ],
        },
    }
    write_table(Path(summary["outputs"]["rows_csv"]), rows, ",")
    write_table(Path(summary["outputs"]["rows_tsv"]), rows, "\t")
    write_json(Path(summary["outputs"]["json"]), summary)
    print(json.dumps({"status": summary["status"], "rows": len(rows), "json": summary["outputs"]["json"]}, sort_keys=True))
    if summary["status"] != "ok_fk_repaired_split_motion_npz":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
