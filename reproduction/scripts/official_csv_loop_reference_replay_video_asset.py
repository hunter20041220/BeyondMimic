#!/usr/bin/env python3
"""Create a report-ready kinematic reference replay video from official-loop motion data."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
MOTION_NPZ = (
    ROOT
    / "res/tracking/official_csv_to_npz_loop_with_enriched_usd/"
    "walk1_subject1_frames_1_180_official_loop_enriched_usd_motion.npz"
)
MOTION_AUDIT = (
    ROOT
    / "res/tracking/official_csv_to_npz_loop_with_enriched_usd/"
    "tracking_official_csv_to_npz_loop_with_enriched_usd_audit.json"
)
BODY_CONTRACT = ROOT / "res/tracking/motion_npz_fixture/tracking_motion_npz_fixture.json"
OFFICIAL_SOURCE_CONTRACT = (
    ROOT / "res/tracking/official_source_contract_audit/tracking_official_source_contract_audit.json"
)
OUT = ROOT / "res/visualization/official_csv_loop_reference_replay"


EDGES = [
    ("pelvis", "left_hip_roll_link"),
    ("left_hip_roll_link", "left_knee_link"),
    ("left_knee_link", "left_ankle_roll_link"),
    ("pelvis", "right_hip_roll_link"),
    ("right_hip_roll_link", "right_knee_link"),
    ("right_knee_link", "right_ankle_roll_link"),
    ("pelvis", "torso_link"),
    ("torso_link", "left_shoulder_roll_link"),
    ("left_shoulder_roll_link", "left_elbow_link"),
    ("left_elbow_link", "left_wrist_yaw_link"),
    ("torso_link", "right_shoulder_roll_link"),
    ("right_shoulder_roll_link", "right_elbow_link"),
    ("right_elbow_link", "right_wrist_yaw_link"),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_axes_equal(ax: Any, xyz: np.ndarray) -> None:
    mins = xyz.min(axis=(0, 1))
    maxs = xyz.max(axis=(0, 1))
    centers = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 0.4)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(max(0.0, centers[2] - radius), centers[2] + radius)


def draw_frame(
    ax: Any,
    body_pos: np.ndarray,
    names: list[str],
    target_names: list[str],
    frame: int,
    title_prefix: str = "Official-loop G1 reference motion",
) -> None:
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    xyz = body_pos[frame]
    target_indices = [name_to_idx[name] for name in target_names]
    target_xyz = xyz[target_indices]
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=10, color="#9ca3af", alpha=0.25, depthshade=False)
    ax.scatter(
        target_xyz[:, 0],
        target_xyz[:, 1],
        target_xyz[:, 2],
        s=38,
        color="#2563eb",
        alpha=0.95,
        depthshade=False,
    )
    for a, b in EDGES:
        ia = name_to_idx[a]
        ib = name_to_idx[b]
        pts = xyz[[ia, ib]]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#111827", linewidth=2.0)
    pelvis = xyz[name_to_idx["pelvis"]]
    torso = xyz[name_to_idx["torso_link"]]
    ax.scatter([pelvis[0]], [pelvis[1]], [pelvis[2]], s=56, color="#dc2626", depthshade=False)
    ax.scatter([torso[0]], [torso[1]], [torso[2]], s=56, color="#16a34a", depthshade=False)
    ax.set_title(f"{title_prefix}, frame {frame:03d}", fontsize=11)


def write_summary_csv(path: Path, body_pos: np.ndarray, names: list[str]) -> dict[str, float]:
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    pelvis = body_pos[:, name_to_idx["pelvis"]]
    torso = body_pos[:, name_to_idx["torso_link"]]
    left_foot = body_pos[:, name_to_idx["left_ankle_roll_link"]]
    right_foot = body_pos[:, name_to_idx["right_ankle_roll_link"]]
    root_step_distance = float(np.linalg.norm(pelvis[-1, :2] - pelvis[0, :2]))
    summary = {
        "frame_count": float(body_pos.shape[0]),
        "body_count": float(body_pos.shape[1]),
        "pelvis_height_min": float(pelvis[:, 2].min()),
        "pelvis_height_max": float(pelvis[:, 2].max()),
        "torso_height_mean": float(torso[:, 2].mean()),
        "left_foot_height_min": float(left_foot[:, 2].min()),
        "right_foot_height_min": float(right_foot[:, 2].min()),
        "pelvis_xy_displacement": root_step_distance,
    }
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in summary.items():
            writer.writerow({"metric": key, "value": value})
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(MOTION_NPZ)
    body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
    fps = int(np.asarray(data["fps"]).reshape(-1)[0])
    body_contract = load_json(BODY_CONTRACT)
    source_contract = load_json(OFFICIAL_SOURCE_CONTRACT)
    motion_audit = load_json(MOTION_AUDIT)
    names = list(body_contract["body_names_urdf_order"])
    target_names = list(source_contract["flat_env"]["body_names"])
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    missing_targets = [name for name in target_names if name not in name_to_idx]
    missing_edges = [edge for edge in EDGES if edge[0] not in name_to_idx or edge[1] not in name_to_idx]
    if body_pos.shape[1] != len(names):
        raise ValueError(f"body count mismatch: body_pos has {body_pos.shape[1]}, names has {len(names)}")
    if missing_targets or missing_edges:
        raise ValueError(f"missing target bodies={missing_targets}, missing edges={missing_edges}")

    video_path = OUT / "official_csv_loop_reference_replay_kinematic.mp4"
    keyframes_path = OUT / "official_csv_loop_reference_replay_keyframes.png"
    summary_csv = OUT / "official_csv_loop_reference_replay_summary.csv"
    readme = OUT / "README.md"
    asset_json = OUT / "official_csv_loop_reference_replay_video_asset.json"

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    set_axes_equal(ax, body_pos[:, [name_to_idx[name] for name in target_names], :])
    ax.view_init(elev=18, azim=-68)
    writer = FFMpegWriter(fps=min(fps, 30), metadata={"title": "BeyondMimic official-loop reference replay"})
    with writer.saving(fig, str(video_path), dpi=150):
        for frame in range(body_pos.shape[0]):
            ax.cla()
            draw_frame(ax, body_pos, names, target_names, frame)
            set_axes_equal(ax, body_pos[:, [name_to_idx[name] for name in target_names], :])
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
        draw_frame(ax, body_pos, names, target_names, frame)
        set_axes_equal(ax, body_pos[:, [name_to_idx[name] for name in target_names], :])
        ax.view_init(elev=18, azim=-68)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
    fig.tight_layout()
    fig.savefig(keyframes_path, dpi=180)
    plt.close(fig)

    metric_summary = write_summary_csv(summary_csv, body_pos, names)
    readme.write_text(
        "\n".join(
            [
                "# Official-Loop Reference Replay Visualization",
                "",
                "This directory contains a local kinematic reference visualization generated from the",
                "official `csv_to_npz.py` loop output that used the enriched-USD runtime patch.",
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
                "local_kinematic_reference_visualization / report asset only. This is not an IsaacLab",
                "closed-loop rollout video, not unpatched official replay output, not paper Fig. 5/Fig. 6",
                "guided diffusion evidence, and not a real-robot result.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    asset_summary = {
        "status": "ok",
        "experiment_type": "official_csv_loop_reference_replay_video_asset",
        "claim_level": "local_kinematic_reference_visualization",
        "source_motion_npz": str(MOTION_NPZ),
        "source_motion_audit": str(MOTION_AUDIT),
        "source_status": motion_audit["status"],
        "source_latest_blocker": motion_audit["latest_blocker"],
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
            "source_motion_audit_ok": motion_audit["status"] == "ok_official_csv_to_npz_loop_with_enriched_usd_patch",
            "body_shape_299_40_3": list(body_pos.shape) == [299, 40, 3],
            "target_bodies_all_present": not missing_targets,
            "edge_bodies_all_present": not missing_edges,
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
                "The MP4 visualizes the saved reference body trajectory from the resource-adjusted official-loop "
                "motion conversion. It is useful for the reading report/PPT, but it is not an IsaacLab rendered "
                "closed-loop replay, not guided diffusion, and not real-robot validation."
            ),
        },
    }
    asset_json.write_text(json.dumps(asset_summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(asset_json), "mp4": str(video_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
