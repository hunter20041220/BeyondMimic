#!/usr/bin/env python3
"""Convert the FK-repaired full bundle from URDF order to IsaacLab robot order.

The first FK-repaired bundle fixed degenerate ``body_pos_w`` values, but a live
probe showed that IsaacLab's runtime articulation body order differs from the
URDF order used by the repair script.  The official ``MotionLoader`` indexes the
motion arrays with runtime robot body indexes, so this script writes a
robot-order version of the same FK data and a 40-motion split for task eval.
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
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_motion_npz/"
    "tracking_g1_official_csv_loop_full_bundle_fk_repaired_motion_npz.json"
)
SOURCE_NPZ = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_motion_npz/"
    "official_csv_loop_full_public_motion_bundle_fk_repaired.npz"
)
BODY_ORDER_PROBE = (
    ROOT
    / "res/tracking/fk_repaired_body_order_runtime_probe/"
    "fk_repaired_body_order_runtime_probe.json"
)
BODY_CONTRACT = ROOT / "res/tracking/motion_npz_fixture/tracking_motion_npz_fixture.json"
OUT = ROOT / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz"
OUT_NPZ = OUT / "official_csv_loop_full_public_motion_bundle_fk_repaired_robot_order.npz"
MOTION_ROOT = OUT / "motions"
BODY_ARRAY_KEYS = ["body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]
PASSTHROUGH_KEYS = ["joint_pos", "joint_vel", "fps"]
TARGET_BODY_NAMES = [
    "pelvis",
    "torso_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str], delimiter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def finite_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "p50": float(np.percentile(values, 50)),
        "max": float(np.max(values)),
    }


def z_stats(body_pos: np.ndarray, body_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    for name in TARGET_BODY_NAMES:
        idx = body_names.index(name)
        z = body_pos[:, idx, 2]
        rows.append({"body_name": name, "body_index": idx, **{f"z_{k}_m": v for k, v in finite_summary(z).items()}})
    return rows


def split_rows(bundle: dict[str, np.ndarray], clip_rows: list[dict[str, Any]], robot_body_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    MOTION_ROOT.mkdir(parents=True, exist_ok=True)
    for clip in clip_rows:
        motion = clip["motion"]
        start = int(clip["start_frame"])
        end = int(clip["end_frame_exclusive"])
        arrays = {key: value[start:end].copy() for key, value in bundle.items() if key != "fps"}
        arrays["fps"] = bundle["fps"].copy()
        motion_dir = MOTION_ROOT / motion
        motion_dir.mkdir(parents=True, exist_ok=True)
        npz_path = motion_dir / "motion.npz"
        np.savez(npz_path, **arrays)
        left_z = float(np.mean(arrays["body_pos_w"][:, robot_body_names.index("left_ankle_roll_link"), 2]))
        right_z = float(np.mean(arrays["body_pos_w"][:, robot_body_names.index("right_ankle_roll_link"), 2]))
        rows.append(
            {
                "motion": motion,
                "start_frame": start,
                "end_frame_exclusive": end,
                "frame_count": end - start,
                "output_npz": str(npz_path),
                "npz_size_bytes": npz_path.stat().st_size,
                "npz_sha256": sha256(npz_path),
                "left_ankle_mean_z_m": left_z,
                "right_ankle_mean_z_m": right_z,
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_audit = load_json(SOURCE_AUDIT)
    probe = load_json(BODY_ORDER_PROBE)
    body_contract = load_json(BODY_CONTRACT)
    urdf_body_names = list(body_contract.get("body_names_urdf_order", []))
    robot_body_names = list(probe.get("robot_body_names", []))
    if not urdf_body_names or not robot_body_names:
        raise RuntimeError("missing body order inputs")
    if sorted(urdf_body_names) != sorted(robot_body_names):
        raise RuntimeError("runtime robot body names and URDF body names differ as sets")
    order_indices = [urdf_body_names.index(name) for name in robot_body_names]

    raw = np.load(SOURCE_NPZ, allow_pickle=True)
    bundle: dict[str, np.ndarray] = {}
    for key in BODY_ARRAY_KEYS:
        bundle[key] = raw[key][:, order_indices].astype(np.float32)
    for key in PASSTHROUGH_KEYS:
        bundle[key] = raw[key].copy()
    np.savez_compressed(OUT_NPZ, **bundle)

    clip_rows = list(source_audit.get("clip_rows", []))
    rows = split_rows(bundle, clip_rows, robot_body_names)

    target_rows = z_stats(bundle["body_pos_w"], robot_body_names)
    alignment_rows = []
    max_z_delta = 0.0
    for name in source_audit.get("target_body_height_rows", []):
        body_name = name.get("body_name")
        if body_name not in robot_body_names or body_name not in urdf_body_names:
            continue
        robot_idx = robot_body_names.index(body_name)
        urdf_idx = urdf_body_names.index(body_name)
        old_z = raw["body_pos_w"][:, urdf_idx, 2]
        new_z = bundle["body_pos_w"][:, robot_idx, 2]
        delta = float(np.max(np.abs(old_z - new_z)))
        max_z_delta = max(max_z_delta, delta)
        alignment_rows.append(
            {
                "body_name": body_name,
                "robot_index": robot_idx,
                "urdf_index": urdf_idx,
                "max_abs_z_delta_m": delta,
                "new_z_mean_m": float(np.mean(new_z)),
            }
        )

    checks = {
        "source_audit_ok": source_audit.get("status") == "ok_official_csv_loop_full_bundle_fk_repaired_motion_npz",
        "body_order_probe_ok": probe.get("status") == "ok_fk_repaired_body_order_runtime_probe",
        "body_order_probe_detected_mismatch": probe.get("checks", {}).get("misindexed_targets_present") is True,
        "source_npz_exists": SOURCE_NPZ.is_file(),
        "output_npz_exists": OUT_NPZ.is_file(),
        "robot_body_count_40": len(robot_body_names) == 40,
        "urdf_body_count_40": len(urdf_body_names) == 40,
        "body_name_sets_match": sorted(urdf_body_names) == sorted(robot_body_names),
        "joint_shape_11960_29": list(bundle["joint_pos"].shape) == [11960, 29],
        "body_shape_11960_40_3": list(bundle["body_pos_w"].shape) == [11960, 40, 3],
        "motion_count_40": len(rows) == 40,
        "all_split_outputs_exist": all(Path(row["output_npz"]).is_file() for row in rows),
        "named_target_z_preserved_after_reorder": max_z_delta < 1e-6,
        "left_right_ankle_mean_z_below_0_25m": all(
            row["z_mean_m"] < 0.25
            for row in target_rows
            if row["body_name"] in {"left_ankle_roll_link", "right_ankle_roll_link"}
        ),
        "does_not_claim_unmodified_official_csv_to_npz": True,
        "does_not_claim_paper_level_tracking": True,
        "does_not_claim_real_robot": True,
    }
    status = "ok_fk_repaired_robot_order_motion_npz" if all(checks.values()) else "failed_fk_repaired_robot_order_motion_npz"

    rows_fields = [
        "motion",
        "start_frame",
        "end_frame_exclusive",
        "frame_count",
        "output_npz",
        "npz_size_bytes",
        "npz_sha256",
        "left_ankle_mean_z_m",
        "right_ankle_mean_z_m",
    ]
    write_rows(OUT / "fk_repaired_robot_order_motion_rows.csv", rows, rows_fields, ",")
    write_rows(OUT / "fk_repaired_robot_order_motion_rows.tsv", rows, rows_fields, "\t")
    write_rows(
        OUT / "fk_repaired_robot_order_alignment.tsv",
        alignment_rows,
        ["body_name", "robot_index", "urdf_index", "max_abs_z_delta_m", "new_z_mean_m"],
        "\t",
    )

    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_g1_official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz",
        "scope": "Reorders the local FK-repaired public G1 motion bundle from URDF body order to IsaacLab runtime robot body order.",
        "source": {
            "fk_repaired_audit": str(SOURCE_AUDIT),
            "fk_repaired_npz": str(SOURCE_NPZ),
            "fk_repaired_npz_sha256": sha256(SOURCE_NPZ),
            "body_order_probe": str(BODY_ORDER_PROBE),
            "body_contract": str(BODY_CONTRACT),
        },
        "checks": checks,
        "metrics": {
            "motion_count": len(rows),
            "total_frames": int(bundle["joint_pos"].shape[0]),
            "fps": int(np.asarray(bundle["fps"]).reshape(-1)[0]),
            "npz_size_bytes": OUT_NPZ.stat().st_size,
            "npz_sha256": sha256(OUT_NPZ),
            "max_named_target_z_delta_after_reorder_m": max_z_delta,
            "left_ankle_mean_z_m": next(row["z_mean_m"] for row in target_rows if row["body_name"] == "left_ankle_roll_link"),
            "right_ankle_mean_z_m": next(row["z_mean_m"] for row in target_rows if row["body_name"] == "right_ankle_roll_link"),
        },
        "robot_body_names": robot_body_names,
        "urdf_body_names": urdf_body_names,
        "order_indices_urdf_for_robot_order": order_indices,
        "rows": rows,
        "target_body_height_rows": target_rows,
        "alignment_rows": alignment_rows,
        "outputs": {
            "json": str(OUT / "tracking_g1_official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz.json"),
            "npz": str(OUT_NPZ),
            "motion_root": str(MOTION_ROOT),
            "rows_csv": str(OUT / "fk_repaired_robot_order_motion_rows.csv"),
            "rows_tsv": str(OUT / "fk_repaired_robot_order_motion_rows.tsv"),
            "alignment_tsv": str(OUT / "fk_repaired_robot_order_alignment.tsv"),
        },
        "interpretation": {
            "goal_complete": False,
            "claim_level": "local_robot_order_fk_repaired_motion_candidate_not_paper_level",
            "main_finding": (
                "The previous FK-repaired body arrays were plausible in URDF order but mismatched the official "
                "MotionLoader runtime indexing contract. This robot-order candidate is the appropriate input for "
                "the next full task eval before rerunning PPO."
            ),
        },
    }
    write_json(Path(summary["outputs"]["json"]), summary)

    md = OUT / "README.md"
    md.write_text(
        "\n".join(
            [
                "# FK-Repaired Robot-Order Motion Bundle",
                "",
                f"Status: `{status}`",
                "",
                "This directory contains a local FK-repaired public-motion bundle reordered to IsaacLab runtime robot body order.",
                "",
                f"- Motion count: `{len(rows)}`",
                f"- Total frames: `{summary['metrics']['total_frames']}`",
                f"- Max named-target z delta after reorder: `{max_z_delta}` m",
                f"- Left/right ankle mean z: `{summary['metrics']['left_ankle_mean_z_m']}` / `{summary['metrics']['right_ankle_mean_z_m']}` m",
                "",
                "Claim boundary: this is a local repair candidate, not unmodified official csv_to_npz.py output, not paper-level tracking, not DAgger, not VAE/diffusion, and not real robot.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "json": summary["outputs"]["json"], "npz": str(OUT_NPZ)}, sort_keys=True))
    if status != "ok_fk_repaired_robot_order_motion_npz":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
