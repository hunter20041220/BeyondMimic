#!/usr/bin/env python3
"""Create a report-ready kinematic replay video for the official-importer-export full dataset audit."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np

from official_csv_loop_reference_replay_video_asset import (
    BODY_CONTRACT,
    OFFICIAL_SOURCE_CONTRACT,
    ROOT,
    draw_frame,
    load_json,
    set_axes_equal,
    sha256_file,
    write_summary_csv,
)


DATASET_AUDIT = (
    ROOT
    / "res/tracking/official_csv_to_npz_loop_full_dataset_with_official_importer_export/"
    "tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_audit.json"
)
ROWS_CSV = (
    ROOT
    / "res/tracking/official_csv_to_npz_loop_full_dataset_with_official_importer_export/"
    "tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_rows.csv"
)
OUT = ROOT / "res/visualization/official_importer_export_full_dataset_reference_replay"
PREFIX = "official_importer_export_full_dataset_reference_replay"
DEFAULT_MOTION = "walk1_subject1"
TITLE_PREFIX = "Official-importer-export G1 reference motion"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def select_motion(rows: list[dict[str, str]]) -> dict[str, str]:
    requested_motion = os.environ.get("BM_IMPORTER_EXPORT_REFERENCE_REPLAY_MOTION", DEFAULT_MOTION)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        raise ValueError("No ok rows found in official-importer-export full dataset csv_to_npz audit")
    for row in ok_rows:
        if row.get("motion") == requested_motion:
            return row
    for row in ok_rows:
        if row.get("motion") == DEFAULT_MOTION:
            return row
    return ok_rows[0]


def read_body_data(motion_npz: Path) -> tuple[np.ndarray, int]:
    data = np.load(motion_npz)
    body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
    fps = int(np.asarray(data["fps"]).reshape(-1)[0])
    return body_pos, fps


def write_readme(
    path: Path,
    video_path: Path,
    keyframes_path: Path,
    summary_csv: Path,
    asset_json: Path,
    motion: str,
    dataset_audit: dict[str, Any],
) -> None:
    aggregate = dataset_audit["aggregate"]
    path.write_text(
        "\n".join(
            [
                "# Official-Importer-Export Full-Dataset Reference Replay Visualization",
                "",
                "This directory contains a local kinematic reference visualization generated from one representative",
                "motion selected from the full public-motion official `csv_to_npz.py` loop audit that used the G1",
                "USDA captured from the official Isaac Sim URDF importer.",
                "",
                "## Dataset Context",
                "",
                f"- selected motion: `{motion}`",
                f"- full audit rows: `{aggregate['row_count']}`",
                f"- full audit ok rows: `{aggregate['ok_count']}`",
                f"- full audit failed rows: `{aggregate['failed_count']}`",
                f"- full audit total frames: `{aggregate['total_frames']}`",
                "",
                "## Assets",
                "",
                f"- `{video_path}`",
                f"- `{keyframes_path}`",
                f"- `{summary_csv}`",
                f"- `{asset_json}`",
                "",
                "## Claim Level",
                "",
                "local_kinematic_reference_visualization / report asset only. This is not an IsaacLab closed-loop",
                "policy rollout, not an unmodified live official converter-entry capture, not paper Fig. 5/Fig. 6",
                "guided diffusion evidence, and not a real-robot result.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dataset_audit = load_json(DATASET_AUDIT)
    rows = load_rows(ROWS_CSV)
    selected_row = select_motion(rows)
    motion_npz = Path(selected_row["output_npz"])
    if not motion_npz.is_file():
        raise FileNotFoundError(motion_npz)

    body_pos, fps = read_body_data(motion_npz)
    body_contract = load_json(BODY_CONTRACT)
    source_contract = load_json(OFFICIAL_SOURCE_CONTRACT)
    names = list(body_contract["body_names_urdf_order"])
    target_names = list(source_contract["flat_env"]["body_names"])
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    missing_targets = [name for name in target_names if name not in name_to_idx]
    if body_pos.shape[1] != len(names):
        raise ValueError(f"body count mismatch: body_pos has {body_pos.shape[1]}, names has {len(names)}")
    if missing_targets:
        raise ValueError(f"missing target bodies={missing_targets}")

    video_path = OUT / f"{PREFIX}_kinematic.mp4"
    keyframes_path = OUT / f"{PREFIX}_keyframes.png"
    summary_csv = OUT / f"{PREFIX}_summary.csv"
    readme = OUT / "README.md"
    asset_json = OUT / f"{PREFIX}_video_asset.json"
    target_body_pos = body_pos[:, [name_to_idx[name] for name in target_names], :]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    set_axes_equal(ax, target_body_pos)
    ax.view_init(elev=18, azim=-68)
    writer = FFMpegWriter(fps=min(fps, 30), metadata={"title": "BeyondMimic official-importer reference replay"})
    with writer.saving(fig, str(video_path), dpi=150):
        for frame in range(body_pos.shape[0]):
            ax.cla()
            draw_frame(ax, body_pos, names, target_names, frame, TITLE_PREFIX)
            set_axes_equal(ax, target_body_pos)
            ax.view_init(elev=18, azim=-68)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_zlabel("z (m)")
            writer.grab_frame()
    plt.close(fig)

    keyframes = [0, body_pos.shape[0] // 3, 2 * body_pos.shape[0] // 3, body_pos.shape[0] - 1]
    fig = plt.figure(figsize=(13, 7))
    for idx, frame in enumerate(keyframes, start=1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        draw_frame(ax, body_pos, names, target_names, frame, TITLE_PREFIX)
        set_axes_equal(ax, target_body_pos)
        ax.view_init(elev=18, azim=-68)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
    fig.tight_layout()
    fig.savefig(keyframes_path, dpi=180)
    plt.close(fig)

    metric_summary = write_summary_csv(summary_csv, body_pos, names)
    write_readme(
        readme,
        video_path,
        keyframes_path,
        summary_csv,
        asset_json,
        selected_row["motion"],
        dataset_audit,
    )

    aggregate = dataset_audit["aggregate"]
    checks = dataset_audit["checks"]
    asset_summary = {
        "status": "ok_official_importer_export_full_dataset_reference_replay_video_asset",
        "experiment_type": "official_importer_export_full_dataset_reference_replay_video_asset",
        "claim_level": "local_kinematic_reference_visualization",
        "source_dataset_audit": str(DATASET_AUDIT),
        "source_dataset_rows_csv": str(ROWS_CSV),
        "source_dataset_status": dataset_audit["status"],
        "source_dataset_aggregate": aggregate,
        "selected_motion": selected_row["motion"],
        "source_motion_npz": str(motion_npz),
        "body_contract": str(BODY_CONTRACT),
        "official_source_contract": str(OFFICIAL_SOURCE_CONTRACT),
        "frame_count": int(body_pos.shape[0]),
        "body_count": int(body_pos.shape[1]),
        "target_body_count": len(target_names),
        "fps": fps,
        "metrics": metric_summary,
        "assets": {
            "mp4": str(video_path),
            "keyframes_png": str(keyframes_path),
            "summary_csv": str(summary_csv),
            "readme": str(readme),
        },
        "asset_sizes": {
            "mp4_bytes": video_path.stat().st_size,
            "keyframes_png_bytes": keyframes_path.stat().st_size,
            "summary_csv_bytes": summary_csv.stat().st_size,
        },
        "asset_sha256": {
            "mp4": sha256_file(video_path),
            "keyframes_png": sha256_file(keyframes_path),
            "summary_csv": sha256_file(summary_csv),
            "readme": sha256_file(readme),
        },
        "checks": {
            "source_dataset_audit_ok": (
                dataset_audit["status"] == "ok_official_csv_to_npz_loop_full_dataset_with_official_importer_export"
            ),
            "source_dataset_40_of_40_ok": aggregate["row_count"] == 40
            and aggregate["ok_count"] == 40
            and aggregate["failed_count"] == 0,
            "source_dataset_total_frames_11960": aggregate["total_frames"] == 11960,
            "uses_official_importer_export_usd": checks["uses_official_importer_export_usd"],
            "does_not_use_resource_adjusted_enriched_usd": checks["does_not_use_resource_adjusted_enriched_usd"],
            "selected_motion_from_ok_row": selected_row.get("status") == "ok",
            "body_shape_299_40_3": list(body_pos.shape) == [299, 40, 3],
            "target_bodies_all_present": not missing_targets,
            "video_exists_nonempty": video_path.is_file() and video_path.stat().st_size > 0,
            "keyframes_exist_nonempty": keyframes_path.is_file() and keyframes_path.stat().st_size > 0,
            "does_not_claim_closed_loop_rollout": True,
            "does_not_claim_paper_fig5_fig6": True,
            "does_not_claim_real_robot": True,
            "does_not_start_training_or_kit": True,
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "report_asset_only_not_closed_loop",
            "why_not_complete": (
                "The MP4 visualizes a representative saved reference body trajectory selected from the full "
                "official-importer-export csv_to_npz audit. It is useful for the reading report/PPT, but it is not "
                "an IsaacLab rendered closed-loop policy rollout, not guided diffusion, not unmodified live official "
                "converter-entry evidence, and not real-robot validation."
            ),
        },
    }
    asset_json.write_text(json.dumps(asset_summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(asset_json), "mp4": str(video_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
