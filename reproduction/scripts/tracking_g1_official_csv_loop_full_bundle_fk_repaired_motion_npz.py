#!/usr/bin/env python3
"""Build a full public-motion FK-repaired candidate bundle.

The current official-loop full bundle has the right NPZ schema but degenerate
``body_pos_w`` values: all rigid bodies are root-like. This script keeps the
same public-motion scope used by the existing local PPO/eval runs
(`40` G1 CSVs, frame range 1..180, 299 output frames each) and rebuilds the
body pose arrays using deterministic non-Kit forward kinematics from the G1
URDF.

This is a repair candidate for debugging the tracking pipeline. It is not the
official Isaac/Kit ``csv_to_npz.py`` articulation export, not a DAgger rollout,
not a paper-level tracking metric, and not real-robot evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SCRIPT_DIR = ROOT / "reproduction/scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from build_level_c_motion_state_fixture import (  # noqa: E402
    DEFAULT_URDF,
    OFFICIAL_CSV_JOINT_NAMES,
    angular_velocity_from_quats,
    compute_fk,
    load_and_interpolate_motion,
    matrix_to_quat_xyzw,
    parse_urdf,
)
from validate_motion_npz_contract import REQUIRED_KEYS  # noqa: E402


CSV_ROOT = ROOT / "download/official/LAFAN1_Retargeting_Dataset/g1"
BODY_CONTRACT = ROOT / "res/tracking/motion_npz_fixture/tracking_motion_npz_fixture.json"
SOURCE_BUNDLE_JSON = (
    ROOT / "res/tracking/official_csv_loop_full_bundle_motion_npz/tracking_g1_official_csv_loop_full_bundle_motion_npz.json"
)
DEGENERACY_AUDIT = (
    ROOT
    / "res/report_assets/official_importer_export_motion_bundle_body_position_degeneracy/"
    "motion_bundle_body_position_degeneracy_audit.json"
)
OUT = ROOT / "res/tracking/official_csv_loop_full_bundle_fk_repaired_motion_npz"
REPORT = ROOT / "res/report_assets/official_csv_loop_full_bundle_fk_repaired_motion_npz"
OUT_NPZ = OUT / "official_csv_loop_full_public_motion_bundle_fk_repaired.npz"
OUT_JSON = OUT / "tracking_g1_official_csv_loop_full_bundle_fk_repaired_motion_npz.json"
OUT_CLIPS_CSV = OUT / "official_csv_loop_full_public_motion_bundle_fk_repaired_clips.csv"
OUT_CLIPS_TSV = OUT / "official_csv_loop_full_public_motion_bundle_fk_repaired_clips.tsv"

INPUT_FPS = 30
OUTPUT_FPS = 50
START_FRAME = 1
END_FRAME = 180

TARGET_BODY_NAMES = [
    "pelvis",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]
ARRAY_KEYS = ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], delimiter: str = ",") -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fieldnames} for row in rows])


def validate_npz(path: Path) -> dict[str, Any]:
    with np.load(path) as data:
        missing = sorted(set(REQUIRED_KEYS) - set(data.files))
        if missing:
            raise ValueError(f"missing keys: {missing}")
        summary: dict[str, Any] = {}
        for key, ndim in REQUIRED_KEYS.items():
            arr = data[key]
            if arr.ndim != ndim:
                raise ValueError(f"{key}: expected ndim {ndim}, got {arr.ndim}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{key}: non-finite values present")
            summary[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if abs(fps - OUTPUT_FPS) > 1e-6:
            raise ValueError(f"fps expected {OUTPUT_FPS}, got {fps}")
        steps = data["joint_pos"].shape[0]
        for key in ["joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]:
            if data[key].shape[0] != steps:
                raise ValueError(f"{key}: timestep mismatch")
        quat_norm = np.linalg.norm(data["body_quat_w"], axis=-1)
        summary["time_steps"] = int(steps)
        summary["fps_value"] = fps
        summary["body_quat_norm_max_abs_error_from_1"] = float(np.max(np.abs(quat_norm - 1.0)))
        return summary


def body_spread_stats(body_pos: np.ndarray) -> dict[str, float]:
    spread = np.ptp(body_pos, axis=1)
    return {
        "x_spread_mean_m": float(np.mean(spread[:, 0])),
        "x_spread_p95_m": float(np.percentile(spread[:, 0], 95)),
        "x_spread_max_m": float(np.max(spread[:, 0])),
        "y_spread_mean_m": float(np.mean(spread[:, 1])),
        "y_spread_p95_m": float(np.percentile(spread[:, 1], 95)),
        "y_spread_max_m": float(np.max(spread[:, 1])),
        "z_spread_mean_m": float(np.mean(spread[:, 2])),
        "z_spread_p95_m": float(np.percentile(spread[:, 2], 95)),
        "z_spread_max_m": float(np.max(spread[:, 2])),
        "max_abs_body_minus_body0_m": float(np.max(np.abs(body_pos - body_pos[:, :1, :]))),
        "max_abs_body_minus_body0_z_m": float(np.max(np.abs(body_pos[:, :, 2] - body_pos[:, :1, 2]))),
    }


def build_one_motion(
    csv_path: Path,
    body_names: list[str],
    children_by_parent: dict[str, Any],
) -> dict[str, Any]:
    motion = load_and_interpolate_motion(
        csv_path=csv_path,
        input_fps=INPUT_FPS,
        output_fps=OUTPUT_FPS,
        start_frame=START_FRAME,
        end_frame=END_FRAME,
    )
    root_pos = motion["base_pos"]
    root_quat_xyzw = motion["base_quat_xyzw"]
    joint_pos = motion["joint_pos"]
    dt = 1.0 / OUTPUT_FPS
    frames = int(joint_pos.shape[0])
    body_pos = np.zeros((frames, len(body_names), 3), dtype=np.float32)
    body_quat_xyzw = np.zeros((frames, len(body_names), 4), dtype=np.float32)
    missing: set[str] = set()
    for t in range(frames):
        joint_values = {name: float(joint_pos[t, idx]) for idx, name in enumerate(OFFICIAL_CSV_JOINT_NAMES)}
        transforms = compute_fk(children_by_parent, root_pos[t], root_quat_xyzw[t], joint_values)
        for b, body_name in enumerate(body_names):
            tf = transforms.get(body_name)
            if tf is None:
                missing.add(body_name)
                continue
            body_pos[t, b] = tf[:3, 3].astype(np.float32)
            body_quat_xyzw[t, b] = matrix_to_quat_xyzw(tf[:3, :3]).astype(np.float32)
    if missing:
        raise RuntimeError(f"{csv_path}: FK missed bodies {sorted(missing)}")

    joint_pos = joint_pos.astype(np.float32)
    joint_vel = np.gradient(joint_pos, dt, axis=0).astype(np.float32)
    body_lin_vel = np.gradient(body_pos, dt, axis=0).astype(np.float32)
    body_ang_vel = np.stack(
        [angular_velocity_from_quats(body_quat_xyzw[:, body_idx].astype(np.float64), dt) for body_idx in range(len(body_names))],
        axis=1,
    ).astype(np.float32)
    body_quat_wxyz = body_quat_xyzw[:, :, [3, 0, 1, 2]].astype(np.float32)
    return {
        "motion": csv_path.stem,
        "input_csv": str(csv_path),
        "input_csv_sha256": sha256_file(csv_path),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos,
        "body_quat_w": body_quat_wxyz,
        "body_lin_vel_w": body_lin_vel,
        "body_ang_vel_w": body_ang_vel,
        "frame_count": frames,
    }


def target_height_rows(body_pos: np.ndarray, body_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in TARGET_BODY_NAMES:
        idx = body_names.index(name)
        z = body_pos[:, idx, 2]
        rows.append(
            {
                "body_name": name,
                "body_index": idx,
                "z_mean_m": float(np.mean(z)),
                "z_min_m": float(np.min(z)),
                "z_p05_m": float(np.percentile(z, 5)),
                "z_p50_m": float(np.percentile(z, 50)),
                "z_p95_m": float(np.percentile(z, 95)),
                "z_max_m": float(np.max(z)),
            }
        )
    return rows


def plot_target_heights(rows: list[dict[str, Any]], path: Path) -> None:
    names = [row["body_name"] for row in rows]
    means = [row["z_mean_m"] for row in rows]
    lows = [row["z_p05_m"] for row in rows]
    highs = [row["z_p95_m"] for row in rows]
    x = np.arange(len(names))
    plt.figure(figsize=(9, 4.6))
    plt.bar(x, means, color="#2563eb")
    plt.vlines(x, lows, highs, color="#111827", linewidth=2)
    plt.xticks(x, names, rotation=35, ha="right")
    plt.ylabel("z position (m)")
    plt.title("FK-repaired full-bundle target body heights")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_per_motion_spread(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [row["motion"] for row in rows]
    values = [row["z_spread_mean_m"] for row in rows]
    x = np.arange(len(labels))
    plt.figure(figsize=(14, 4.8))
    plt.bar(x, values, color="#16a34a")
    plt.xticks(x, labels, rotation=75, ha="right", fontsize=7)
    plt.ylabel("Mean per-frame z spread (m)")
    plt.title("FK-repaired body-position spread by motion")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    body_contract = load_json(BODY_CONTRACT)
    source_bundle = load_json(SOURCE_BUNDLE_JSON)
    degeneracy = load_json(DEGENERACY_AUDIT)
    body_names = list(body_contract["body_names_urdf_order"])
    children_by_parent = parse_urdf(DEFAULT_URDF)
    csv_paths = sorted(CSV_ROOT.glob("*.csv"))

    arrays: dict[str, list[np.ndarray]] = {key: [] for key in ARRAY_KEYS}
    clip_rows: list[dict[str, Any]] = []
    per_motion_rows: list[dict[str, Any]] = []
    cursor = 0
    for csv_path in csv_paths:
        item = build_one_motion(csv_path, body_names, children_by_parent)
        frame_count = int(item["frame_count"])
        for key in ARRAY_KEYS:
            arrays[key].append(item[key])
        body_pos = item["body_pos_w"]
        spread = body_spread_stats(body_pos)
        clip_rows.append(
            {
                "motion": item["motion"],
                "input_csv": item["input_csv"],
                "input_csv_sha256": item["input_csv_sha256"],
                "start_frame": cursor,
                "end_frame_exclusive": cursor + frame_count,
                "frame_count": frame_count,
                "fps": OUTPUT_FPS,
                "joint_shape": list(item["joint_pos"].shape),
                "body_shape": list(body_pos.shape),
            }
        )
        row = {"motion": item["motion"], **spread}
        for target in target_height_rows(body_pos, body_names):
            row[f"{target['body_name']}_z_mean_m"] = target["z_mean_m"]
        per_motion_rows.append(row)
        cursor += frame_count

    bundle = {key: np.concatenate(value, axis=0).astype(np.float32) for key, value in arrays.items()}
    bundle["fps"] = np.asarray([OUTPUT_FPS], dtype=np.int64)
    np.savez_compressed(OUT_NPZ, **bundle)
    validator = validate_npz(OUT_NPZ)

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
                "pelvis_position_jump_m": float(np.linalg.norm(bundle["body_pos_w"][cur_start, 0] - bundle["body_pos_w"][prev_end, 0])),
                "joint_position_jump": float(np.linalg.norm(bundle["joint_pos"][cur_start] - bundle["joint_pos"][prev_end])),
            }
        )

    bundle_spread = body_spread_stats(bundle["body_pos_w"])
    target_rows = target_height_rows(bundle["body_pos_w"], body_names)
    report_json = REPORT / "fk_repaired_motion_bundle_assets.json"
    target_csv = REPORT / "fk_repaired_target_body_heights.csv"
    per_motion_csv = REPORT / "fk_repaired_per_motion_spread.csv"
    spread_csv = REPORT / "fk_repaired_bundle_spread.csv"
    target_png = REPORT / "fk_repaired_target_body_heights.png"
    per_motion_png = REPORT / "fk_repaired_per_motion_spread.png"
    readme = REPORT / "README.md"

    clip_fields = ["motion", "input_csv", "input_csv_sha256", "start_frame", "end_frame_exclusive", "frame_count", "fps", "joint_shape", "body_shape"]
    write_rows(OUT_CLIPS_CSV, clip_rows, clip_fields)
    write_rows(OUT_CLIPS_TSV, clip_rows, clip_fields, delimiter="\t")
    write_rows(
        target_csv,
        target_rows,
        ["body_name", "body_index", "z_mean_m", "z_min_m", "z_p05_m", "z_p50_m", "z_p95_m", "z_max_m"],
    )
    per_motion_fields = ["motion", *body_spread_stats(bundle["body_pos_w"][:1]).keys(), *[f"{name}_z_mean_m" for name in TARGET_BODY_NAMES]]
    write_rows(per_motion_csv, per_motion_rows, per_motion_fields)
    write_rows(spread_csv, [{"source": "fk_repaired_full_bundle", **bundle_spread}], ["source", *bundle_spread.keys()])
    plot_target_heights(target_rows, target_png)
    plot_per_motion_spread(per_motion_rows, per_motion_png)
    readme.write_text(
        "\n".join(
            [
                "# FK-Repaired Full Public-Motion Bundle",
                "",
                "This directory summarizes a non-Kit URDF-FK repaired candidate for the full local G1 public-motion bundle.",
                "",
                "## Claim Boundary",
                "",
                "This is a local repair candidate for debugging. It is not official Isaac/Kit csv_to_npz output, not paper-level tracking, not DAgger, not VAE/diffusion evidence, and not real-robot evidence.",
                "",
                "## Key Metrics",
                "",
                f"- Motion count: `{len(clip_rows)}`",
                f"- Total frames: `{int(bundle['joint_pos'].shape[0])}`",
                f"- Mean z spread: `{bundle_spread['z_spread_mean_m']}` m",
                f"- Left/right ankle mean z: `{next(row['z_mean_m'] for row in target_rows if row['body_name'] == 'left_ankle_roll_link')}` / `{next(row['z_mean_m'] for row in target_rows if row['body_name'] == 'right_ankle_roll_link')}` m",
                "",
                "## Outputs",
                "",
                f"- Candidate NPZ: `{OUT_NPZ}`",
                f"- Summary JSON: `{OUT_JSON}`",
                f"- Target height plot: `{target_png}`",
                f"- Per-motion spread plot: `{per_motion_png}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    old_spread = (degeneracy.get("bundle") or {}).get("spread", {})
    checks = {
        "all_40_csvs_used": len(csv_paths) == 40,
        "total_frames_11960": int(bundle["joint_pos"].shape[0]) == 11960,
        "joint_shape_11960_29": list(bundle["joint_pos"].shape) == [11960, 29],
        "body_shape_11960_40_3": list(bundle["body_pos_w"].shape) == [11960, 40, 3],
        "all_arrays_finite": all(np.isfinite(bundle[key]).all() for key in ARRAY_KEYS),
        "body_quaternions_unit": validator["body_quat_norm_max_abs_error_from_1"] < 1e-5,
        "fk_repaired_z_spread_non_degenerate_gt_0_5m": bundle_spread["z_spread_mean_m"] > 0.5,
        "old_bundle_degeneracy_context_available": bool(old_spread),
        "old_to_new_z_spread_improves_by_1e6x": (
            bool(old_spread)
            and bundle_spread["z_spread_mean_m"] / max(float(old_spread.get("z_spread_mean_m", 0.0)), 1e-12) > 1e6
        ),
        "left_right_ankle_mean_z_below_0_25m": all(
            row["z_mean_m"] < 0.25 for row in target_rows if row["body_name"] in {"left_ankle_roll_link", "right_ankle_roll_link"}
        ),
        "report_assets_exist": all(path.is_file() for path in [target_csv, per_motion_csv, spread_csv, target_png, per_motion_png, readme]),
        "does_not_claim_official_csv_to_npz_output": True,
        "does_not_claim_paper_level_tracking": True,
        "does_not_claim_real_robot": True,
    }
    status = "ok_official_csv_loop_full_bundle_fk_repaired_motion_npz" if all(checks.values()) else "failed"

    summary: dict[str, Any] = {
        "status": status,
        "experiment_type": "tracking_g1_official_csv_loop_full_bundle_fk_repaired_motion_npz",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Full public-motion FK-repaired candidate bundle matching the existing local official-loop bundle scope: "
            "40 public G1 CSV motions, frame range 1..180, 299 output frames per motion, 11960 frames total."
        ),
        "inputs": {
            "csv_root": str(CSV_ROOT),
            "urdf": str(DEFAULT_URDF),
            "urdf_sha256": sha256_file(DEFAULT_URDF),
            "body_contract": str(BODY_CONTRACT),
            "source_bundle_json": str(SOURCE_BUNDLE_JSON),
            "source_bundle_status": source_bundle.get("status"),
            "degeneracy_audit": str(DEGENERACY_AUDIT),
            "degeneracy_status": degeneracy.get("status"),
        },
        "outputs": {
            "npz": str(OUT_NPZ),
            "json": str(OUT_JSON),
            "clips_csv": str(OUT_CLIPS_CSV),
            "clips_tsv": str(OUT_CLIPS_TSV),
            "report_assets_json": str(report_json),
            "target_body_heights_csv": str(target_csv),
            "per_motion_spread_csv": str(per_motion_csv),
            "bundle_spread_csv": str(spread_csv),
            "target_body_heights_png": str(target_png),
            "per_motion_spread_png": str(per_motion_png),
            "readme": str(readme),
        },
        "bundle": {
            "motion_count": len(clip_rows),
            "total_frames": int(bundle["joint_pos"].shape[0]),
            "fps": OUTPUT_FPS,
            "joint_pos_shape": list(bundle["joint_pos"].shape),
            "joint_vel_shape": list(bundle["joint_vel"].shape),
            "body_pos_w_shape": list(bundle["body_pos_w"].shape),
            "body_quat_w_shape": list(bundle["body_quat_w"].shape),
            "body_lin_vel_w_shape": list(bundle["body_lin_vel_w"].shape),
            "body_ang_vel_w_shape": list(bundle["body_ang_vel_w"].shape),
            "npz_size_bytes": OUT_NPZ.stat().st_size,
            "npz_sha256": sha256_file(OUT_NPZ),
            "boundary_count": len(boundary_jumps),
            "max_pelvis_position_boundary_jump_m": max((x["pelvis_position_jump_m"] for x in boundary_jumps), default=0.0),
            "max_joint_position_boundary_jump": max((x["joint_position_jump"] for x in boundary_jumps), default=0.0),
            "spread": bundle_spread,
            "validator": validator,
        },
        "target_body_height_rows": target_rows,
        "clip_rows": clip_rows,
        "per_motion_rows": per_motion_rows,
        "boundary_jumps": boundary_jumps,
        "checks": checks,
        "interpretation": {
            "claim_level": "local_nonkit_fk_repaired_motion_bundle_candidate",
            "goal_complete": False,
            "why_useful": (
                "This candidate repairs the body-position target degeneracy while preserving the existing full-bundle "
                "motion scope. It can be used for the next replay/task diagnostic to test whether the tracking target "
                "path is now physically plausible before any new PPO or downstream teacher claim."
            ),
            "not_paper_level_reasons": [
                "body poses are computed by non-Kit URDF FK rather than official Isaac articulation state",
                "the bundle is a local repair candidate, not unmodified official csv_to_npz.py output",
                "no PPO training/evaluation is started by this script",
                "no DAgger, VAE, diffusion, TensorRT, or real-robot result is produced",
            ],
        },
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_assets = {
        "status": "ok_fk_repaired_motion_bundle_report_assets" if status.startswith("ok_") else "failed",
        "claim_level": summary["interpretation"]["claim_level"],
        "source_summary": str(OUT_JSON),
        "assets": {
            "target_body_heights_csv": str(target_csv),
            "per_motion_spread_csv": str(per_motion_csv),
            "bundle_spread_csv": str(spread_csv),
            "target_body_heights_png": str(target_png),
            "per_motion_spread_png": str(per_motion_png),
            "readme": str(readme),
        },
        "metrics": {
            "motion_count": summary["bundle"]["motion_count"],
            "total_frames": summary["bundle"]["total_frames"],
            "z_spread_mean_m": bundle_spread["z_spread_mean_m"],
            "left_ankle_z_mean_m": next(row["z_mean_m"] for row in target_rows if row["body_name"] == "left_ankle_roll_link"),
            "right_ankle_z_mean_m": next(row["z_mean_m"] for row in target_rows if row["body_name"] == "right_ankle_roll_link"),
        },
        "checks": checks,
    }
    report_json.write_text(json.dumps(report_assets, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "json": str(OUT_JSON), "npz": str(OUT_NPZ)}, sort_keys=True))
    if not status.startswith("ok_"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
