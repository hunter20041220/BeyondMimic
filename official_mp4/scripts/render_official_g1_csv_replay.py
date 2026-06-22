#!/usr/bin/env python3
"""Render MuJoCo G1 mesh videos from official 36-column G1 CSV reference motions."""

from __future__ import annotations

import csv
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
PKG = ROOT / "official_mp4"
MUJOCO_SCRIPTS = ROOT / "mujoco_mp4/scripts"
if str(MUJOCO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MUJOCO_SCRIPTS))

from mujoco_common import render_frame  # noqa: E402


DEFAULT_MODEL = ROOT / "mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_mocap_29dof.xml"
DEFAULT_CSV = ROOT / "Dataset_beyondmimic/ablation/tkd_skill.csv"
REPLAY_CAMERA = "bm_official_replay_track"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def traceback_payload(exc: BaseException) -> dict[str, str]:
    return {"exception_type": type(exc).__name__, "exception": str(exc), "traceback": traceback.format_exc()}


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_36col_csv(path: Path) -> np.ndarray:
    arr = np.genfromtxt(path, delimiter=",")
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != 36:
        raise ValueError(f"Expected 36 columns for G1 generalized coordinates, got {arr.shape} from {path}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values in {path}")
    return arr.astype(np.float64)


def quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    return np.stack([q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]], axis=1)


def camera_xml(target_x: float, target_y: float, distance: float = 4.4) -> str:
    return (
        f'<camera name="{REPLAY_CAMERA}" mode="fixed" '
        f'pos="{target_x - 1.8:.3f} {target_y - distance:.3f} 1.85" '
        'xyaxes="1 0 0 0 0.34 0.94" fovy="42"/>'
    )


def patch_model_with_camera(model_xml: Path, root_xy: np.ndarray, out_xml: Path) -> Path:
    text = model_xml.read_text(encoding="utf-8")
    target = root_xy[len(root_xy) // 2]
    insert = camera_xml(float(target[0]), float(target[1]))
    if f'name="{REPLAY_CAMERA}"' in text:
        text = text.replace(text[text.find(f'<camera name="{REPLAY_CAMERA}"') : text.find("/>", text.find(f'<camera name="{REPLAY_CAMERA}"')) + 2], insert)
    else:
        marker = "</worldbody>"
        if marker not in text:
            raise ValueError(f"Cannot add camera: {marker} not found in {model_xml}")
        text = text.replace(marker, insert + "\n  " + marker, 1)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    out_xml.write_text(text, encoding="utf-8")
    return out_xml


def render_csv(csv_path: Path, motion_name: str, frames_limit: int, fps: int, width: int, height: int) -> dict[str, Any]:
    import mujoco

    q = load_36col_csv(csv_path)
    frames_total = int(q.shape[0])
    frames = min(frames_total, frames_limit) if frames_limit > 0 else frames_total
    q = q[:frames]
    root_xyz = q[:, 0:3].copy()
    root_quat_wxyz = quat_xyzw_to_wxyz(q[:, 3:7])
    joint_pos = q[:, 7:36]
    joint_vel = np.gradient(joint_pos, 1.0 / fps, axis=0)

    out_dir = PKG / "videos" / motion_name
    res_dir = PKG / "res" / motion_name
    tmp_dir = DEFAULT_MODEL.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    patched_xml = tmp_dir / f"g1_mocap_29dof_{motion_name}_camera.xml"
    patch_model_with_camera(DEFAULT_MODEL, root_xyz[:, :2], patched_xml)
    model = mujoco.MjModel.from_xml_path(str(patched_xml))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    mp4_path = out_dir / f"{motion_name}_mujoco_reference_replay.mp4"
    keyframe_path = out_dir / f"{motion_name}_keyframe.png"
    metrics_path = res_dir / f"{motion_name}_metrics.csv"
    summary_path = res_dir / f"{motion_name}_summary.json"

    rows: list[dict[str, Any]] = []
    try:
        with imageio.get_writer(mp4_path, fps=min(fps, 30), codec="libx264", quality=8, macro_block_size=1) as writer:
            for frame_idx in range(frames):
                data.qpos[:] = 0.0
                data.qvel[:] = 0.0
                data.qpos[0:3] = root_xyz[frame_idx]
                quat = root_quat_wxyz[frame_idx]
                norm = np.linalg.norm(quat)
                data.qpos[3:7] = quat / norm if norm > 0 else np.array([1.0, 0.0, 0.0, 0.0])
                data.qpos[7 : 7 + 29] = joint_pos[frame_idx]
                data.qvel[6 : 6 + 29] = joint_vel[frame_idx]
                mujoco.mj_forward(model, data)
                frame = render_frame(model, data, renderer, camera=REPLAY_CAMERA)
                if frame_idx == 0:
                    imageio.imwrite(keyframe_path, frame)
                writer.append_data(frame)
                rows.append(
                    {
                        "frame": frame_idx,
                        "time_s": frame_idx / fps,
                        "root_x": float(root_xyz[frame_idx, 0]),
                        "root_y": float(root_xyz[frame_idx, 1]),
                        "root_z": float(root_xyz[frame_idx, 2]),
                        "joint_abs_mean": float(np.mean(np.abs(joint_pos[frame_idx]))),
                        "joint_vel_abs_mean": float(np.mean(np.abs(joint_vel[frame_idx]))),
                    }
                )
    finally:
        renderer.close()

    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame", "time_s", "root_x", "root_y", "root_z", "joint_abs_mean", "joint_vel_abs_mean"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "status": "ok",
        "timestamp_utc": utc_now(),
        "experiment_type": "official_released_data_mujoco_g1_csv_reference_replay",
        "backend": os.environ.get("MUJOCO_GL", "egl"),
        "claim_level": "official released 36-column G1 reference data rendered in MuJoCo; kinematic replay, not policy closed-loop, not real robot",
        "source_csv": str(csv_path),
        "source_sha256": sha256(csv_path),
        "source_format": "36-column generalized coordinates: root xyz, root quat xyzw, 29 G1 joint positions",
        "model_xml": str(DEFAULT_MODEL),
        "patched_xml": str(patched_xml),
        "motion_name": motion_name,
        "frames_total": frames_total,
        "frames_rendered": frames,
        "fps_assumed": fps,
        "outputs": {
            "mp4": str(mp4_path),
            "keyframe_png": str(keyframe_path),
            "metrics_csv": str(metrics_path),
            "summary_json": str(summary_path),
        },
        "file_sizes": {
            "mp4": mp4_path.stat().st_size if mp4_path.exists() else 0,
            "keyframe_png": keyframe_path.stat().st_size if keyframe_path.exists() else 0,
            "metrics_csv": metrics_path.stat().st_size if metrics_path.exists() else 0,
        },
        "checks": {
            "mp4_exists": mp4_path.is_file() and mp4_path.stat().st_size > 0,
            "keyframe_exists": keyframe_path.is_file() and keyframe_path.stat().st_size > 0,
            "metrics_exists": metrics_path.is_file() and metrics_path.stat().st_size > 0,
            "input_is_36_columns": True,
            "joint_dim_29": int(joint_pos.shape[1]) == 29,
            "does_not_claim_policy_rollout": True,
            "does_not_claim_real_robot": True,
            "does_not_claim_paper_level_fig5_fig6": True,
        },
        "limitations": [
            "This is MuJoCo kinematic reference replay using mj_forward, not closed-loop policy control.",
            "GRF/contact dynamics are not validated by this replay because qpos is imposed frame-by-frame.",
        ],
    }
    write_json(summary_path, payload)
    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--motion-name", default="")
    parser.add_argument("--frames", type=int, default=0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    motion_name = args.motion_name or args.csv.stem
    summary_path = PKG / "res" / motion_name / f"{motion_name}_summary.json"
    try:
        payload = render_csv(args.csv, motion_name, args.frames, args.fps, args.width, args.height)
        print(json.dumps({"status": "ok", "mp4": payload["outputs"]["mp4"], "frames": payload["frames_rendered"]}))
    except Exception as exc:  # noqa: BLE001
        payload = {
            "status": "failed",
            "timestamp_utc": utc_now(),
            "experiment_type": "official_released_data_mujoco_g1_csv_reference_replay",
            "claim_level": "failed MuJoCo G1 CSV replay attempt",
            "source_csv": str(args.csv),
            "error": traceback_payload(exc),
        }
        write_json(summary_path, payload)
        print(json.dumps({"status": "failed", "summary": str(summary_path)}))
        raise


if __name__ == "__main__":
    main()
