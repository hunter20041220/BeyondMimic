#!/usr/bin/env python3
"""Inventory official BeyondMimic released dataset sources for MuJoCo MP4 use."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
PKG = ROOT / "official_mp4"
DATASET = ROOT / "Dataset_beyondmimic"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    tmp.replace(path)


def line_count(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for _ in f)


def inspect_numeric_csv(path: Path) -> dict[str, Any]:
    try:
        arr = np.genfromtxt(path, delimiter=",", max_rows=5)
        if arr.ndim == 1:
            cols = int(arr.shape[0])
            rows_seen = 1
        else:
            rows_seen = int(arr.shape[0])
            cols = int(arr.shape[1])
        finite = bool(np.isfinite(arr).all())
        return {"numeric": True, "columns": cols, "rows_seen": rows_seen, "finite_sample": finite}
    except Exception as exc:  # noqa: BLE001
        return {"numeric": False, "columns": "", "rows_seen": "", "finite_sample": "", "error": str(exc)[:240]}


def category_for(path: Path) -> tuple[str, str, str, str]:
    rel = path.relative_to(DATASET)
    parts = rel.parts
    suffix = path.suffix.lower()
    if rel.as_posix() == "ablation/tkd_skill.csv":
        return (
            "g1_36col_reference_motion",
            "direct_mujoco_g1_replay",
            "required",
            "36-col G1 generalized coordinates: root xyz, root quaternion, 29 joint positions.",
        )
    if parts and parts[0] == "GRF":
        if "real_walk" in parts or "real_run" in parts:
            return (
                "real_robot_grf_csv",
                "plot_or_metric_overlay_only",
                "optional",
                "Real robot force/wrench CSV for GRF plots; not a robot motion trajectory.",
            )
        if "walk_ref" in parts or "run_ref" in parts:
            return (
                "human_reference_grf",
                "plot_or_metric_overlay_only",
                "optional",
                "Human reference ground reaction force data; not a G1 qpos trajectory.",
            )
        return ("grf_support_file", "plot_or_metric_overlay_only", "optional", "GRF plotting support file.")
    if parts and parts[0] == "base_imu":
        return (
            "real_robot_imu_csv",
            "plot_or_metric_overlay_only",
            "optional",
            "Real robot IMU data for paper-curve reproduction; not a G1 qpos trajectory.",
        )
    if parts and parts[0] == "adaptive_sample":
        return (
            "adaptive_sampling_metric",
            "plot_or_metric_overlay_only",
            "optional",
            "Adaptive sampling/failure heatmap metric data; not a robot motion trajectory.",
        )
    if parts and parts[0] == "rosbag_ablation" and "global" in parts and suffix == ".csv":
        return (
            "mocap_global_csv",
            "mocap_marker_visualization_pending",
            "optional",
            "Motive/global mocap rigid-body and marker CSV; useful for trajectory analysis, not direct G1 joint qpos.",
        )
    if suffix == ".mcap":
        return (
            "ros2_mcap",
            "mujoco_joint_odom_replay_if_joint_odom_available",
            "optional",
            "ROS2 MCAP can be rendered as MuJoCo state replay when /joint_states and /odom are present.",
        )
    if suffix in {".png", ".html"}:
        return ("existing_plot_or_viewer", "report_asset_only", "optional", "Existing released-data visual/report asset.")
    if suffix in {".py", ".json", ".yaml", ".txt", ".xlsx"}:
        return ("support_or_plot_script", "documentation_or_plotting", "optional", "Support metadata or plotting script.")
    return ("other", "not_applicable", "optional", "Not currently used for MuJoCo MP4 generation.")


def main() -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(DATASET.rglob("*")):
        if not path.is_file():
            continue
        category, use, required, notes = category_for(path)
        rel = path.relative_to(ROOT).as_posix()
        inspect: dict[str, Any] = {}
        if path.suffix.lower() == ".csv" and path.stat().st_size < 10 * 1024 * 1024:
            inspect = inspect_numeric_csv(path)
        rows.append(
            {
                "source_path": str(path),
                "relative_path": rel,
                "exists": path.exists(),
                "file_size": path.stat().st_size,
                "sha256": sha256(path) if path.stat().st_size <= 200 * 1024 * 1024 else "skipped_large_file",
                "category": category,
                "official_mp4_use": use,
                "required_or_optional": required,
                "direct_video_candidate": use == "direct_mujoco_g1_replay",
                "large_file": path.stat().st_size > 100 * 1024 * 1024,
                "line_count": line_count(path) if path.suffix.lower() == ".csv" and path.stat().st_size < 10 * 1024 * 1024 else "",
                "sample_numeric": inspect.get("numeric", ""),
                "sample_columns": inspect.get("columns", ""),
                "sample_finite": inspect.get("finite_sample", ""),
                "claim_level": "official_released_data_inventory",
                "notes": notes,
            }
        )

    fieldnames = [
        "source_path",
        "relative_path",
        "exists",
        "file_size",
        "sha256",
        "category",
        "official_mp4_use",
        "required_or_optional",
        "direct_video_candidate",
        "large_file",
        "line_count",
        "sample_numeric",
        "sample_columns",
        "sample_finite",
        "claim_level",
        "notes",
    ]
    out_json = PKG / "official_dataset_inventory.json"
    out_tsv = PKG / "official_dataset_inventory.tsv"
    summary = {
        "status": "ok_official_dataset_inventory",
        "timestamp_utc": utc_now(),
        "dataset_root": str(DATASET),
        "row_count": len(rows),
        "counts_by_category": {cat: sum(1 for r in rows if r["category"] == cat) for cat in sorted({r["category"] for r in rows})},
        "direct_video_candidates": [r for r in rows if r["direct_video_candidate"]],
        "claim_level": "inventory_only; not a policy rollout and not paper-level reproduction",
        "checks": {
            "dataset_exists": DATASET.is_dir(),
            "has_tkd_skill_direct_candidate": any(r["relative_path"] == "Dataset_beyondmimic/ablation/tkd_skill.csv" for r in rows),
            "grf_marked_not_motion": all(
                r["official_mp4_use"] != "direct_mujoco_g1_replay"
                for r in rows
                if r["relative_path"].startswith("Dataset_beyondmimic/GRF/")
            ),
            "mcap_marked_as_joint_odom_replay_candidate": all(
                r["official_mp4_use"] == "mujoco_joint_odom_replay_if_joint_odom_available"
                for r in rows
                if r["category"] == "ros2_mcap"
            ),
        },
        "outputs": {"json": str(out_json), "tsv": str(out_tsv)},
        "rows": rows,
    }
    write_json(out_json, summary)
    write_tsv(out_tsv, rows, fieldnames)
    print(json.dumps({"status": summary["status"], "rows": len(rows), "direct_candidates": len(summary["direct_video_candidates"])}))


if __name__ == "__main__":
    main()
