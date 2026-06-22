#!/usr/bin/env python3
"""Render a MuJoCo G1 reference replay video from an existing motion NPZ.

This is not a closed-loop policy rollout.  It writes the reference root pose
and joint angles into MuJoCo state, calls mj_forward, and renders the mesh.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mujoco_common import PKG, ROOT, render_frame, sha256, traceback_payload, utc_now, write_json


DEFAULT_MODEL = ROOT / "mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_mocap_29dof.xml"
DEFAULT_MOTION = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/motions/walk1_subject1/motion.npz"
)
REPLAY_CAMERA = "bm_replay_track"


def camera_xml(target_x: float, target_y: float) -> str:
    return f"""
    <camera name="{REPLAY_CAMERA}" mode="fixed" pos="{target_x - 1.8:.3f} {target_y - 4.2:.3f} 1.9" xyaxes="1 0 0 0 0.34 0.94" fovy="42"/>
    """


def patch_model_with_camera(model_xml: Path, motion_root_xy: np.ndarray, out_xml: Path) -> Path:
    text = model_xml.read_text(encoding="utf-8")
    if f'name="{REPLAY_CAMERA}"' in text:
        out_xml.write_text(text, encoding="utf-8")
        return out_xml
    target = motion_root_xy[len(motion_root_xy) // 2]
    insert = camera_xml(float(target[0]), float(target[1]))
    marker = "</worldbody>"
    if marker not in text:
        raise ValueError(f"Cannot add camera: {marker} not found in {model_xml}")
    out_xml.write_text(text.replace(marker, insert + "\n  " + marker, 1), encoding="utf-8")
    return out_xml


def load_motion(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required = ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "fps"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing required motion keys: {missing}")
    motion = {key: data[key] for key in required}
    if motion["joint_pos"].ndim != 2 or motion["joint_pos"].shape[1] != 29:
        raise ValueError(f"Expected joint_pos shape (T, 29), got {motion['joint_pos'].shape}")
    if motion["body_pos_w"].ndim != 3 or motion["body_pos_w"].shape[2] != 3:
        raise ValueError(f"Expected body_pos_w shape (T, B, 3), got {motion['body_pos_w'].shape}")
    if motion["body_quat_w"].ndim != 3 or motion["body_quat_w"].shape[2] != 4:
        raise ValueError(f"Expected body_quat_w shape (T, B, 4), got {motion['body_quat_w'].shape}")
    return motion


def main() -> None:
    import mujoco

    backend = os.environ.get("MUJOCO_GL", "egl")
    model_path = Path(os.environ.get("BM_MUJOCO_G1_XML", str(DEFAULT_MODEL))).expanduser()
    motion_path = Path(os.environ.get("BM_MUJOCO_MOTION_NPZ", str(DEFAULT_MOTION))).expanduser()
    motion_name = os.environ.get("BM_MUJOCO_MOTION_NAME", motion_path.parent.name or motion_path.stem)
    max_frames = int(os.environ.get("BM_MUJOCO_REPLAY_FRAMES", "0"))
    width = int(os.environ.get("BM_MUJOCO_WIDTH", "1280"))
    height = int(os.environ.get("BM_MUJOCO_HEIGHT", "720"))

    out_dir = PKG / "res/reference_replay" / motion_name
    log_dir = PKG / "logs/reference_replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "reference_replay_summary.json"

    try:
        motion = load_motion(motion_path)
        frames_total = int(motion["joint_pos"].shape[0])
        frames = min(frames_total, max_frames) if max_frames > 0 else frames_total
        fps = int(np.asarray(motion["fps"]).reshape(-1)[0])
        render_fps = int(os.environ.get("BM_MUJOCO_RENDER_FPS", str(min(fps, 30))))
        patched_xml = model_path.parent / f"{model_path.stem}_{motion_name}_camera.xml"
        patched_xml.parent.mkdir(parents=True, exist_ok=True)
        patch_model_with_camera(model_path, motion["body_pos_w"][:frames, 0, :2], patched_xml)

        model = mujoco.MjModel.from_xml_path(str(patched_xml))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=height, width=width)

        mp4_path = out_dir / "reference_replay.mp4"
        keyframe_path = out_dir / "reference_replay_keyframe.png"
        metrics_path = out_dir / "reference_replay_metrics.csv"
        root_xyz = motion["body_pos_w"][:frames, 0, :].astype(np.float64)
        root_quat = motion["body_quat_w"][:frames, 0, :].astype(np.float64)
        joint_pos = motion["joint_pos"][:frames].astype(np.float64)
        joint_vel = motion["joint_vel"][:frames].astype(np.float64)

        metrics_rows: list[dict[str, object]] = []
        with imageio.get_writer(mp4_path, fps=render_fps, codec="libx264", quality=8, macro_block_size=1) as writer:
            for frame_idx in range(frames):
                data.qpos[:] = 0.0
                data.qvel[:] = 0.0
                data.qpos[0:3] = root_xyz[frame_idx]
                quat = root_quat[frame_idx]
                norm = np.linalg.norm(quat)
                data.qpos[3:7] = quat / norm if norm > 0 else np.array([1.0, 0.0, 0.0, 0.0])
                data.qpos[7 : 7 + 29] = joint_pos[frame_idx]
                data.qvel[6 : 6 + 29] = joint_vel[frame_idx]
                mujoco.mj_forward(model, data)
                frame = render_frame(model, data, renderer, camera=REPLAY_CAMERA)
                if frame_idx == 0:
                    imageio.imwrite(keyframe_path, frame)
                writer.append_data(frame)
                metrics_rows.append(
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
        renderer.close()

        with metrics_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["frame", "time_s", "root_x", "root_y", "root_z", "joint_abs_mean", "joint_vel_abs_mean"],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(metrics_rows)

        payload = {
            "status": "ok",
            "timestamp_utc": utc_now(),
            "experiment_type": "mujoco_reference_replay_video",
            "backend": backend,
            "claim_level": "MuJoCo reference replay visualization; not policy closed-loop, not IsaacLab, not real robot",
            "source_model_xml": str(model_path),
            "patched_model_xml": str(patched_xml),
            "motion_npz": str(motion_path),
            "motion_sha256": sha256(motion_path),
            "motion_name": motion_name,
            "frames_total": frames_total,
            "frames_rendered": frames,
            "source_fps": fps,
            "render_fps": render_fps,
            "model_dims": {
                "nq": model.nq,
                "nv": model.nv,
                "nu": model.nu,
                "nbody": model.nbody,
                "njnt": model.njnt,
                "ngeom": model.ngeom,
                "nmesh": model.nmesh,
            },
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
                "metrics_csv_exists": metrics_path.is_file() and metrics_path.stat().st_size > 0,
                "joint_dim_29": int(joint_pos.shape[1]) == 29,
                "does_not_claim_policy_rollout": True,
                "does_not_claim_isaaclab": True,
                "does_not_claim_real_robot": True,
            },
            "limitations": [
                "Reference state is imposed frame-by-frame with mj_forward, so contacts and control stability are not evaluated.",
                "Joint order follows local FK-repaired robot-order bundle and G1 MuJoCo hinge order; this is a local visualization bridge.",
            ],
        }
        write_json(summary_path, payload)
        print(json.dumps({"status": "ok", "mp4": str(mp4_path), "frames": frames, "backend": backend}))
    except Exception as exc:  # noqa: BLE001
        payload = {
            "status": "failed",
            "timestamp_utc": utc_now(),
            "experiment_type": "mujoco_reference_replay_video",
            "backend": backend,
            "claim_level": "failed MuJoCo reference replay attempt",
            "source_model_xml": str(model_path),
            "motion_npz": str(motion_path),
            "error": traceback_payload(exc),
        }
        write_json(summary_path, payload)
        print(json.dumps({"status": "failed", "summary": str(summary_path)}))
        raise


if __name__ == "__main__":
    main()
