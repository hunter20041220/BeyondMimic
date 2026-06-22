#!/usr/bin/env python3
"""Render MuJoCo G1 mesh replay from official released ROS2 MCAP joint/odom data."""

from __future__ import annotations

import csv
import json
import os
import sys
import traceback
from bisect import bisect_left
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
from mcap_ros2.reader import read_ros2_messages


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
PKG = ROOT / "official_mp4"
MUJOCO_SCRIPTS = ROOT / "mujoco_mp4/scripts"
if str(MUJOCO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(MUJOCO_SCRIPTS))

from mujoco_common import render_frame  # noqa: E402


DEFAULT_MODEL = ROOT / "mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_mocap_29dof.xml"
JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
REPLAY_CAMERA = "bm_official_mcap_track"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def traceback_payload(exc: BaseException) -> dict[str, str]:
    return {"exception_type": type(exc).__name__, "exception": str(exc), "traceback": traceback.format_exc()}


def stamp_sec(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def nearest_index(times: np.ndarray, t: float) -> int:
    idx = bisect_left(times, t)
    if idx <= 0:
        return 0
    if idx >= len(times):
        return len(times) - 1
    return idx if abs(times[idx] - t) < abs(times[idx - 1] - t) else idx - 1


def read_mcap_states(path: Path, max_messages: int = 0, target_joint_rows: int = 0) -> dict[str, Any]:
    joint_rows: list[tuple[float, np.ndarray, np.ndarray]] = []
    odom_rows: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    joint_name_mismatches = 0
    msg_count = 0
    with path.open("rb") as f:
        for message in read_ros2_messages(f, topics=["/joint_states", "/odom"]):
            msg_count += 1
            if max_messages and msg_count > max_messages:
                break
            topic = message.channel.topic
            msg = message.ros_msg
            if topic == "/joint_states":
                names = list(msg.name)
                name_to_idx = {name: i for i, name in enumerate(names)}
                if any(name not in name_to_idx for name in JOINT_NAMES):
                    joint_name_mismatches += 1
                    continue
                pos = np.asarray([float(msg.position[name_to_idx[name]]) for name in JOINT_NAMES], dtype=np.float64)
                if len(msg.velocity) >= len(names):
                    vel = np.asarray([float(msg.velocity[name_to_idx[name]]) for name in JOINT_NAMES], dtype=np.float64)
                else:
                    vel = np.zeros(29, dtype=np.float64)
                joint_rows.append((stamp_sec(msg.header.stamp), pos, vel))
            elif topic == "/odom":
                pose = msg.pose.pose
                twist = msg.twist.twist
                pos = np.asarray([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)
                quat_wxyz = quat_xyzw_to_wxyz(
                    np.asarray(
                        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
                        dtype=np.float64,
                    )
                )
                lin = np.asarray([twist.linear.x, twist.linear.y, twist.linear.z], dtype=np.float64)
                ang = np.asarray([twist.angular.x, twist.angular.y, twist.angular.z], dtype=np.float64)
                odom_rows.append((stamp_sec(msg.header.stamp), pos, quat_wxyz, lin, ang))
            if target_joint_rows and len(joint_rows) >= target_joint_rows and odom_rows:
                if odom_rows[-1][0] >= joint_rows[-1][0]:
                    break
    if not joint_rows:
        raise ValueError(f"No usable /joint_states rows with 29 expected G1 joints in {path}")
    if not odom_rows:
        raise ValueError(f"No /odom rows in {path}")

    joint_rows.sort(key=lambda row: row[0])
    odom_rows.sort(key=lambda row: row[0])
    jt = np.asarray([row[0] for row in joint_rows], dtype=np.float64)
    ot = np.asarray([row[0] for row in odom_rows], dtype=np.float64)
    qpos = np.asarray([row[1] for row in joint_rows], dtype=np.float64)
    qvel = np.asarray([row[2] for row in joint_rows], dtype=np.float64)
    opos = np.asarray([row[1] for row in odom_rows], dtype=np.float64)
    oq = np.asarray([row[2] for row in odom_rows], dtype=np.float64)
    olin = np.asarray([row[3] for row in odom_rows], dtype=np.float64)
    oang = np.asarray([row[4] for row in odom_rows], dtype=np.float64)
    return {
        "joint_times": jt,
        "joint_pos": qpos,
        "joint_vel": qvel,
        "odom_times": ot,
        "odom_pos": opos,
        "odom_quat": oq,
        "odom_lin_vel": olin,
        "odom_ang_vel": oang,
        "joint_name_mismatches": joint_name_mismatches,
    }


def camera_xml(target_x: float, target_y: float, distance: float = 4.5) -> str:
    return (
        f'<camera name="{REPLAY_CAMERA}" mode="fixed" '
        f'pos="{target_x - 1.7:.3f} {target_y - distance:.3f} 1.95" '
        'xyaxes="1 0 0 0 0.34 0.94" fovy="42"/>'
    )


def patch_model_with_camera(model_xml: Path, root_xy: np.ndarray, out_xml: Path) -> Path:
    text = model_xml.read_text(encoding="utf-8")
    target = root_xy[len(root_xy) // 2]
    insert = camera_xml(float(target[0]), float(target[1]))
    if f'name="{REPLAY_CAMERA}"' in text:
        start = text.find(f'<camera name="{REPLAY_CAMERA}"')
        end = text.find("/>", start) + 2
        text = text[:start] + insert + text[end:]
    else:
        text = text.replace("</worldbody>", insert + "\n  </worldbody>", 1)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    out_xml.write_text(text, encoding="utf-8")
    return out_xml


def render_mcap(path: Path, motion_name: str, frames_limit: int, stride: int, width: int, height: int) -> dict[str, Any]:
    import mujoco

    target_joint_rows = 0
    if frames_limit > 0:
        target_joint_rows = frames_limit * max(1, stride) + 2
    states = read_mcap_states(path, target_joint_rows=target_joint_rows)
    jt = states["joint_times"]
    ot = states["odom_times"]
    indices = list(range(0, len(jt), max(1, stride)))
    if frames_limit > 0:
        indices = indices[:frames_limit]
    if not indices:
        raise ValueError("No frames selected")
    odom_indices = [nearest_index(ot, float(jt[i])) for i in indices]

    root_pos = states["odom_pos"][odom_indices].copy()
    root_quat = states["odom_quat"][odom_indices].copy()
    joint_pos = states["joint_pos"][indices].copy()
    joint_vel = states["joint_vel"][indices].copy()
    times = jt[indices] - jt[indices[0]]
    if len(times) > 1:
        render_fps = int(round(1.0 / max(np.median(np.diff(times)), 1e-6)))
        render_fps = max(10, min(30, render_fps))
    else:
        render_fps = 30

    out_dir = PKG / "videos" / motion_name
    res_dir = PKG / "res" / motion_name
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    patched_xml = PKG / "tmp" / f"{DEFAULT_MODEL.stem}_{motion_name}_camera.xml"
    patch_model_with_camera(DEFAULT_MODEL, root_pos[:, :2], patched_xml)

    model = mujoco.MjModel.from_xml_path(str(patched_xml))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    mp4_path = out_dir / f"{motion_name}_mujoco_mcap_joint_replay.mp4"
    keyframe_path = out_dir / f"{motion_name}_keyframe.png"
    metrics_path = res_dir / f"{motion_name}_metrics.csv"
    summary_path = res_dir / f"{motion_name}_summary.json"
    rows: list[dict[str, Any]] = []
    try:
        with imageio.get_writer(mp4_path, fps=render_fps, codec="libx264", quality=8, macro_block_size=1) as writer:
            for out_idx, frame_idx in enumerate(range(len(indices))):
                data.qpos[:] = 0.0
                data.qvel[:] = 0.0
                data.qpos[0:3] = root_pos[frame_idx]
                q = root_quat[frame_idx]
                norm = np.linalg.norm(q)
                data.qpos[3:7] = q / norm if norm > 0 else np.array([1.0, 0.0, 0.0, 0.0])
                data.qpos[7 : 7 + 29] = joint_pos[frame_idx]
                data.qvel[0:3] = states["odom_lin_vel"][odom_indices[frame_idx]]
                data.qvel[3:6] = states["odom_ang_vel"][odom_indices[frame_idx]]
                data.qvel[6 : 6 + 29] = joint_vel[frame_idx]
                mujoco.mj_forward(model, data)
                image = render_frame(model, data, renderer, camera=REPLAY_CAMERA)
                if out_idx == 0:
                    imageio.imwrite(keyframe_path, image)
                writer.append_data(image)
                rows.append(
                    {
                        "frame": out_idx,
                        "source_joint_index": indices[frame_idx],
                        "time_s": float(times[frame_idx]),
                        "root_x": float(root_pos[frame_idx, 0]),
                        "root_y": float(root_pos[frame_idx, 1]),
                        "root_z": float(root_pos[frame_idx, 2]),
                        "joint_abs_mean": float(np.mean(np.abs(joint_pos[frame_idx]))),
                        "joint_vel_abs_mean": float(np.mean(np.abs(joint_vel[frame_idx]))),
                    }
                )
    finally:
        renderer.close()

    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["frame", "source_joint_index", "time_s", "root_x", "root_y", "root_z", "joint_abs_mean", "joint_vel_abs_mean"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "status": "ok",
        "timestamp_utc": utc_now(),
        "experiment_type": "official_released_rosbag_mcap_mujoco_joint_odom_replay",
        "backend": os.environ.get("MUJOCO_GL", "osmesa"),
        "claim_level": "official released real-robot rosbag joint/odom data rendered in MuJoCo; kinematic replay, not policy closed-loop",
        "source_mcap": str(path),
        "motion_name": motion_name,
        "joint_message_count": int(len(jt)),
        "odom_message_count": int(len(ot)),
        "frames_rendered": len(indices),
        "stride": stride,
        "render_fps": render_fps,
        "time_span_s": float(times[-1]) if len(times) else 0.0,
        "joint_name_mismatches": int(states["joint_name_mismatches"]),
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
            "joint_dim_29": int(joint_pos.shape[1]) == 29,
            "does_not_claim_policy_rollout": True,
            "does_not_claim_real_robot_deployment": True,
            "does_not_claim_paper_level_fig5_fig6": True,
        },
        "limitations": [
            "This replays recorded real-robot joint and odom states in MuJoCo with mj_forward.",
            "It is not a closed-loop controller and does not validate contact dynamics or policy stability.",
        ],
    }
    write_json(summary_path, payload)
    return payload


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mcap", type=Path, required=True)
    parser.add_argument("--motion-name", default="")
    parser.add_argument("--frames", type=int, default=450)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    motion_name = args.motion_name or args.mcap.stem
    summary_path = PKG / "res" / motion_name / f"{motion_name}_summary.json"
    try:
        payload = render_mcap(args.mcap, motion_name, args.frames, args.stride, args.width, args.height)
        print(json.dumps({"status": "ok", "mp4": payload["outputs"]["mp4"], "frames": payload["frames_rendered"]}))
    except Exception as exc:  # noqa: BLE001
        payload = {
            "status": "failed",
            "timestamp_utc": utc_now(),
            "experiment_type": "official_released_rosbag_mcap_mujoco_joint_odom_replay",
            "claim_level": "failed MuJoCo MCAP joint/odom replay attempt",
            "source_mcap": str(args.mcap),
            "error": traceback_payload(exc),
        }
        write_json(summary_path, payload)
        print(json.dumps({"status": "failed", "summary": str(summary_path)}))
        raise


if __name__ == "__main__":
    main()
