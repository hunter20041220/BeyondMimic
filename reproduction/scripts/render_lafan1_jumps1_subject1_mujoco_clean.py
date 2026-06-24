#!/usr/bin/env python3
"""Render clean MuJoCo evidence for LAFAN1 ``jumps1_subject1``.

The script makes two deliberately separated artifacts from the original
Unitree-retargeted LAFAN1 CSV:

1. ``original_csv_reference_replay``: frame-by-frame MuJoCo mesh replay via
   ``mj_forward``.  This proves the source motion can be displayed.
2. ``reference_action_control``: MuJoCo ``mj_step`` PD tracking of the same
   joint targets with the local G1 position actuators and pelvis root assist.

Claim boundary: these outputs are source/reference baselines only.  They are
not teacher/RL, VAE, diffusion, guidance, IsaacLab rendered rollout, Fig. 5/6,
or real-robot evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SCRIPT_DIR = ROOT / "mujoco_mp4/scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mujoco_common import render_frame  # noqa: E402
from mujoco_pd_control_video import (  # noqa: E402
    apply_root_assist,
    actuator_joint_order,
    load_action_rows,
    patch_joints_and_actuators,
    quat_error_rotvec,
    quat_to_roll_pitch_yaw,
)


OUT_ROOT = ROOT / "res/visualization/lafan1_jumps1_subject1_mujoco"
MODEL_XML = ROOT / "mujoco_mp4/assets/work_g1/gmr_unitree_g1/g1_mocap_29dof.xml"
SOURCE_CSV = ROOT / "download/official/LAFAN1_Retargeting_Dataset/g1/jumps1_subject1.csv"
FULL_MOTION_NPZ = ROOT / "res/tracking/stage1_multisource_motion_bundle/motions/lafan1_jumps1_subject1/motion.npz"
CAMERA_REPLAY = "bm_jumps1_clean_replay"
WINDOWS = {
    "high_dynamic_52s_67s": {"start_frame": 1560, "end_frame": 2010, "fps": 30},
    "stable_dynamic_164s_179s": {"start_frame": 4920, "end_frame": 5370, "fps": 30},
}


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


def traceback_payload(exc: BaseException) -> dict[str, str]:
    return {"exception_type": type(exc).__name__, "exception": str(exc), "traceback": traceback.format_exc()}


def load_36col_csv(path: Path) -> np.ndarray:
    arr = np.genfromtxt(path, delimiter=",")
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] != 36:
        raise ValueError(f"Expected 36 columns, got {arr.shape} from {path}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Non-finite values in {path}")
    return arr.astype(np.float64)


def normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm <= 0.0):
        raise ValueError("zero-norm quaternion")
    return q / norm


def quat_xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    return np.stack([q_xyzw[:, 3], q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2]], axis=1)


def ffprobe(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,duration,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    except FileNotFoundError:
        return {"ok": False, "error": "ffprobe_not_found"}
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.strip()}
    streams = json.loads(proc.stdout).get("streams", [])
    if not streams:
        return {"ok": False, "error": "no_video_stream"}
    stream = streams[0]
    return {
        "ok": True,
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "duration": float(stream.get("duration", 0.0)),
        "nb_frames": int(stream.get("nb_frames", 0)),
        "r_frame_rate": stream.get("r_frame_rate", ""),
    }


def patch_replay_camera(model_xml: Path, root_xy: np.ndarray, out_xml: Path) -> Path:
    tree = ET.parse(model_xml)
    root = tree.getroot()
    world = root.find("worldbody")
    if world is None:
        raise ValueError(f"No worldbody in {model_xml}")
    for cam in world.findall("camera"):
        if cam.attrib.get("name") == CAMERA_REPLAY:
            world.remove(cam)
    center = np.mean(root_xy, axis=0)
    span = np.ptp(root_xy, axis=0)
    distance = max(5.2, float(np.linalg.norm(span)) * 1.25 + 3.0)
    cam = ET.Element("camera", {"name": CAMERA_REPLAY, "mode": "fixed"})
    cam.set("pos", f"{float(center[0]):.3f} {float(center[1]) - distance:.3f} 2.05")
    cam.set("xyaxes", "1 0 0 0 0.34 0.94")
    cam.set("fovy", "50")
    world.append(cam)
    stat = root.find("statistic")
    if stat is not None:
        stat.set("center", f"{float(center[0]):.3f} {float(center[1]):.3f} 0.85")
        stat.set("extent", "2.8")
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_xml, encoding="utf-8", xml_declaration=False)
    return out_xml


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No metric rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def window_from_csv(window_name: str) -> dict[str, Any]:
    if window_name not in WINDOWS:
        raise KeyError(f"Unknown window {window_name}; available={sorted(WINDOWS)}")
    spec = WINDOWS[window_name]
    full = load_36col_csv(SOURCE_CSV)
    start, end, fps = int(spec["start_frame"]), int(spec["end_frame"]), int(spec["fps"])
    q = full[start:end]
    root_xyz = q[:, 0:3].copy()
    recenter_root_xy = os.environ.get("BM_JUMPS1_RECENTER_ROOT_XY", "1") == "1"
    if recenter_root_xy:
        root_xyz[:, 0:2] -= root_xyz[0:1, 0:2]
    root_quat = normalize_quat_wxyz(quat_xyzw_to_wxyz(q[:, 3:7]))
    joint_pos = q[:, 7:36].copy()
    joint_vel = np.gradient(joint_pos, 1.0 / fps, axis=0)
    return {
        "window_name": window_name,
        "start_frame": start,
        "end_frame": end,
        "source_fps": fps,
        "source_start_time_s": start / fps,
        "source_end_time_s": end / fps,
        "root_xyz": root_xyz,
        "root_xy_recentered_for_visualization": recenter_root_xy,
        "root_quat_wxyz": root_quat,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
    }


def render_reference_replay(window: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    import mujoco

    case = "original_csv_reference_replay"
    out_dir = OUT_ROOT / window["window_name"] / case
    out_dir.mkdir(parents=True, exist_ok=True)
    patched_xml = MODEL_XML.parent / f"g1_mocap_29dof_lafan1_jumps1_{window['window_name']}_camera.xml"
    patch_replay_camera(MODEL_XML, window["root_xyz"][:, :2], patched_xml)
    model = mujoco.MjModel.from_xml_path(str(patched_xml))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    mp4 = out_dir / f"{case}.mp4"
    metrics_csv = out_dir / f"{case}_metrics.csv"
    summary_json = out_dir / f"{case}_summary.json"
    keyframes = {
        0: out_dir / f"{case}_keyframe_first.png",
        len(window["joint_pos"]) // 2: out_dir / f"{case}_keyframe_mid.png",
        len(window["joint_pos"]) - 1: out_dir / f"{case}_keyframe_last.png",
    }
    rows: list[dict[str, Any]] = []
    try:
        with imageio.get_writer(mp4, fps=window["source_fps"], codec="libx264", quality=8, macro_block_size=1) as writer:
            for i in range(len(window["joint_pos"])):
                data.qpos[:] = 0.0
                data.qvel[:] = 0.0
                data.qpos[0:3] = window["root_xyz"][i]
                data.qpos[3:7] = window["root_quat_wxyz"][i]
                data.qpos[7 : 7 + 29] = window["joint_pos"][i]
                data.qvel[6 : 6 + 29] = window["joint_vel"][i]
                mujoco.mj_forward(model, data)
                frame = render_frame(model, data, renderer, camera=CAMERA_REPLAY)
                if i in keyframes:
                    imageio.imwrite(keyframes[i], frame)
                writer.append_data(frame)
                rows.append(
                    {
                        "frame": i,
                        "source_frame": window["start_frame"] + i,
                        "time_s": i / window["source_fps"],
                        "source_time_s": window["source_start_time_s"] + i / window["source_fps"],
                        "root_x": float(window["root_xyz"][i, 0]),
                        "root_y": float(window["root_xyz"][i, 1]),
                        "root_z": float(window["root_xyz"][i, 2]),
                        "joint_abs_mean": float(np.mean(np.abs(window["joint_pos"][i]))),
                        "joint_vel_abs_mean": float(np.mean(np.abs(window["joint_vel"][i]))),
                    }
                )
    finally:
        renderer.close()
    write_metrics(metrics_csv, rows)
    probe = ffprobe(mp4)
    payload = {
        "status": "ok",
        "timestamp_utc": utc_now(),
        "experiment_type": "lafan1_jumps1_subject1_original_csv_reference_replay",
        "backend": os.environ.get("MUJOCO_GL", "osmesa"),
        "claim_level": "Original Unitree-retargeted LAFAN1 jumps1_subject1 CSV rendered in MuJoCo via mj_forward; not policy control.",
        "window": {k: window[k] for k in ["window_name", "start_frame", "end_frame", "source_fps", "source_start_time_s", "source_end_time_s"]},
        "source_csv": str(SOURCE_CSV),
        "source_sha256": sha256(SOURCE_CSV),
        "model_xml": str(MODEL_XML),
        "patched_xml": str(patched_xml),
        "frames_rendered": int(len(window["joint_pos"])),
        "duration_seconds": len(window["joint_pos"]) / window["source_fps"],
        "source_summary": {
            "root_xy_displacement_m": float(np.linalg.norm(window["root_xyz"][-1, :2] - window["root_xyz"][0, :2])),
            "root_xy_recentered_for_visualization": bool(window["root_xy_recentered_for_visualization"]),
            "root_z_min": float(np.min(window["root_xyz"][:, 2])),
            "root_z_max": float(np.max(window["root_xyz"][:, 2])),
            "root_z_range": float(np.ptp(window["root_xyz"][:, 2])),
            "joint_dim": int(window["joint_pos"].shape[1]),
            "max_joint_step": float(np.max(np.abs(np.diff(window["joint_pos"], axis=0)))),
        },
        "outputs": {
            "mp4": str(mp4),
            "metrics_csv": str(metrics_csv),
            "summary_json": str(summary_json),
            "keyframe_first": str(keyframes[0]),
            "keyframe_mid": str(keyframes[len(window["joint_pos"]) // 2]),
            "keyframe_last": str(keyframes[len(window["joint_pos"]) - 1]),
        },
        "ffprobe": probe,
        "checks": {
            "mp4_exists": mp4.is_file() and mp4.stat().st_size > 0,
            "ffprobe_ok": bool(probe.get("ok")),
            "ffprobe_frame_count_matches": int(probe.get("nb_frames", -1)) == len(window["joint_pos"]) if probe.get("ok") else False,
            "keyframes_exist": all(path.is_file() and path.stat().st_size > 0 for path in keyframes.values()),
            "metrics_csv_exists": metrics_csv.is_file() and metrics_csv.stat().st_size > 0,
            "input_is_original_lafan1_36col_csv": True,
            "joint_dim_29": int(window["joint_pos"].shape[1]) == 29,
            "does_not_claim_policy_rollout": True,
            "does_not_claim_real_robot": True,
        },
    }
    write_json(summary_json, payload)
    return payload


def render_reference_action_control(window: dict[str, Any], width: int, height: int) -> dict[str, Any]:
    import mujoco

    case = "reference_action_control"
    out_dir = OUT_ROOT / window["window_name"] / case
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_cfg = load_action_rows()
    old_camera = {
        "BM_MUJOCO_PD_CAMERA_POS": os.environ.get("BM_MUJOCO_PD_CAMERA_POS"),
        "BM_MUJOCO_PD_CAMERA_XYAXES": os.environ.get("BM_MUJOCO_PD_CAMERA_XYAXES"),
        "BM_MUJOCO_PD_CAMERA_FOVY": os.environ.get("BM_MUJOCO_PD_CAMERA_FOVY"),
    }
    os.environ["BM_MUJOCO_PD_CAMERA_POS"] = "0.000 -5.300 1.95"
    os.environ["BM_MUJOCO_PD_CAMERA_XYAXES"] = "1 0 0 0 0.34 0.94"
    os.environ["BM_MUJOCO_PD_CAMERA_FOVY"] = "52"
    patched_xml = MODEL_XML.parent / f"g1_mocap_29dof_lafan1_jumps1_{window['window_name']}_pd.xml"
    try:
        patch_joints_and_actuators(MODEL_XML, patched_xml, rows_cfg)
    finally:
        for key, value in old_camera.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    model = mujoco.MjModel.from_xml_path(str(patched_xml))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    names = actuator_joint_order(model)
    expected_names = [row["joint_name"] for row in rows_cfg]
    if names != expected_names:
        raise RuntimeError("Actuator joint order does not match action-scale audit order")
    pelvis_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_body < 0:
        raise RuntimeError("MuJoCo body 'pelvis' not found")

    # Keep the root centered for this reference-action control baseline.  This
    # avoids making camera drift look like a controller failure.
    root_targets = window["root_xyz"].copy()
    root_targets[:, 0:2] = 0.0
    root_quats = window["root_quat_wxyz"].copy()
    joint_targets = window["joint_pos"].copy()
    fps = int(window["source_fps"])
    substeps = int(os.environ.get("BM_JUMPS1_CONTROL_SUBSTEPS", "4"))
    settle_steps = int(os.environ.get("BM_JUMPS1_CONTROL_SETTLE_STEPS", "40"))

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = root_targets[0]
    data.qpos[3:7] = root_quats[0]
    data.qpos[7 : 7 + 29] = joint_targets[0]
    data.ctrl[:] = np.clip(joint_targets[0], model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    mujoco.mj_forward(model, data)
    for _ in range(settle_steps):
        data.xfrc_applied[:] = 0.0
        apply_root_assist(model, data, pelvis_body, root_targets[0], root_quats[0])
        mujoco.mj_step(model, data)

    mp4 = out_dir / f"{case}.mp4"
    metrics_csv = out_dir / f"{case}_metrics.csv"
    summary_json = out_dir / f"{case}_summary.json"
    keyframes = {
        0: out_dir / f"{case}_keyframe_first.png",
        len(joint_targets) // 2: out_dir / f"{case}_keyframe_mid.png",
        len(joint_targets) - 1: out_dir / f"{case}_keyframe_last.png",
    }
    rows: list[dict[str, Any]] = []
    try:
        with imageio.get_writer(mp4, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
            for i in range(len(joint_targets)):
                target = np.clip(joint_targets[i], model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
                data.ctrl[:] = target
                for _ in range(substeps):
                    data.xfrc_applied[:] = 0.0
                    apply_root_assist(model, data, pelvis_body, root_targets[i], root_quats[i])
                    mujoco.mj_step(model, data)
                frame = render_frame(model, data, renderer, camera="bm_pd_fixed_center")
                if i in keyframes:
                    imageio.imwrite(keyframes[i], frame)
                writer.append_data(frame)
                q = data.qpos[7 : 7 + 29].copy()
                qd = data.qvel[6 : 6 + 29].copy()
                err = q - target
                roll, pitch, yaw = quat_to_roll_pitch_yaw(data.qpos[3:7])
                rows.append(
                    {
                        "frame": i,
                        "time_s": i / fps,
                        "source_frame": window["start_frame"] + i,
                        "source_time_s": window["source_start_time_s"] + i / fps,
                        "root_x": float(data.qpos[0]),
                        "root_y": float(data.qpos[1]),
                        "root_z": float(data.qpos[2]),
                        "root_roll": roll,
                        "root_pitch": pitch,
                        "root_yaw": yaw,
                        "joint_target_abs_mean": float(np.mean(np.abs(target))),
                        "joint_error_abs_mean": float(np.mean(np.abs(err))),
                        "joint_error_abs_max": float(np.max(np.abs(err))),
                        "joint_velocity_abs_mean": float(np.mean(np.abs(qd))),
                        "ctrl_abs_mean": float(np.mean(np.abs(data.ctrl))),
                        "root_target_x": float(root_targets[i, 0]),
                        "root_target_y": float(root_targets[i, 1]),
                        "root_target_z": float(root_targets[i, 2]),
                        "root_position_error_m": float(np.linalg.norm(data.xpos[pelvis_body] - root_targets[i])),
                        "root_orientation_error_rad": float(np.linalg.norm(quat_error_rotvec(root_quats[i], data.xquat[pelvis_body]))),
                        "contact_count": int(data.ncon),
                        "fall_proxy": bool(data.qpos[2] < 0.45 or abs(roll) > 1.2 or abs(pitch) > 1.2),
                    }
                )
    finally:
        renderer.close()
    write_metrics(metrics_csv, rows)
    probe = ffprobe(mp4)
    fall_count = sum(1 for row in rows if row["fall_proxy"])
    payload = {
        "status": "ok",
        "timestamp_utc": utc_now(),
        "experiment_type": "lafan1_jumps1_subject1_reference_action_control",
        "backend": os.environ.get("MUJOCO_GL", "osmesa"),
        "claim_level": "MuJoCo mj_step PD tracking of original LAFAN1 jumps1_subject1 joint targets with pelvis root assist; not learned policy control.",
        "window": {k: window[k] for k in ["window_name", "start_frame", "end_frame", "source_fps", "source_start_time_s", "source_end_time_s"]},
        "source_csv": str(SOURCE_CSV),
        "source_sha256": sha256(SOURCE_CSV),
        "source_full_motion_npz_for_context": str(FULL_MOTION_NPZ),
        "model_xml": str(MODEL_XML),
        "patched_xml": str(patched_xml),
        "frames_rendered": int(len(joint_targets)),
        "duration_seconds": len(joint_targets) / fps,
        "simulation": {
            "uses_mj_step": True,
            "writes_qpos_each_frame": False,
            "actuator_type": "position",
            "actuator_count": int(model.nu),
            "control_substeps_per_frame": substeps,
            "settle_steps": settle_steps,
            "timestep": float(model.opt.timestep),
            "root_assist_enabled": True,
            "root_assist_type": "external pelvis force/torque stabilizer applied before mj_step",
            "root_xy_recentered_targets": True,
            "source_root_xy_recentered_for_visualization": bool(window["root_xy_recentered_for_visualization"]),
        },
        "metrics": {
            "joint_error_abs_mean": float(np.mean([row["joint_error_abs_mean"] for row in rows])),
            "joint_error_abs_max": float(np.max([row["joint_error_abs_max"] for row in rows])),
            "root_position_error_mean_m": float(np.mean([row["root_position_error_m"] for row in rows])),
            "root_position_error_max_m": float(np.max([row["root_position_error_m"] for row in rows])),
            "root_orientation_error_mean_rad": float(np.mean([row["root_orientation_error_rad"] for row in rows])),
            "root_height_min": float(np.min([row["root_z"] for row in rows])),
            "root_height_max": float(np.max([row["root_z"] for row in rows])),
            "root_xy_abs_max": float(np.max([max(abs(row["root_x"]), abs(row["root_y"])) for row in rows])),
            "contact_count_mean": float(np.mean([row["contact_count"] for row in rows])),
            "fall_proxy_count": int(fall_count),
        },
        "outputs": {
            "mp4": str(mp4),
            "metrics_csv": str(metrics_csv),
            "summary_json": str(summary_json),
            "keyframe_first": str(keyframes[0]),
            "keyframe_mid": str(keyframes[len(joint_targets) // 2]),
            "keyframe_last": str(keyframes[len(joint_targets) - 1]),
        },
        "ffprobe": probe,
        "checks": {
            "mp4_exists": mp4.is_file() and mp4.stat().st_size > 0,
            "ffprobe_ok": bool(probe.get("ok")),
            "ffprobe_frame_count_matches": int(probe.get("nb_frames", -1)) == len(joint_targets) if probe.get("ok") else False,
            "keyframes_exist": all(path.is_file() and path.stat().st_size > 0 for path in keyframes.values()),
            "metrics_csv_exists": metrics_csv.is_file() and metrics_csv.stat().st_size > 0,
            "uses_mj_step": True,
            "does_not_write_qpos_each_frame": True,
            "uses_root_assist_controller": True,
            "uses_29_position_actuators": int(model.nu) == 29,
            "native_mujoco_ppo_obs_adapter": False,
            "does_not_claim_native_mujoco_policy_controller": True,
            "does_not_claim_teacher_vae_diffusion_guidance": True,
            "does_not_claim_real_robot": True,
            "fall_proxy_count_zero": int(fall_count) == 0,
        },
        "limitations": [
            "This is a source/reference PD baseline, not teacher/RL, VAE, diffusion, or guidance.",
            "Root XY targets are recentered and root assist is enabled to keep the robot centered and visible.",
            "The default replay visualization recenters root XY; global path displacement is not evaluated by this clean video.",
            "Passing this gate does not permit downstream long training; formula/parameter and teacher quality gates still apply.",
        ],
    }
    write_json(summary_json, payload)
    return payload


def write_index(window: dict[str, Any], replay: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    summary_path = OUT_ROOT / window["window_name"] / "lafan1_jumps1_subject1_mujoco_summary.json"
    readme_path = OUT_ROOT / window["window_name"] / "README.md"
    status = "ok" if replay["status"] == "ok" and control["status"] == "ok" else "failed"
    payload = {
        "status": status,
        "timestamp_utc": utc_now(),
        "experiment_type": "lafan1_jumps1_subject1_mujoco_clean_deliverable",
        "claim_level": "Local MuJoCo source/reference baseline for LAFAN1 jumps1_subject1; not learned BeyondMimic control.",
        "window": {k: window[k] for k in ["window_name", "start_frame", "end_frame", "source_fps", "source_start_time_s", "source_end_time_s"]},
        "source_csv": str(SOURCE_CSV),
        "source_sha256": sha256(SOURCE_CSV),
        "source_full_motion_npz_for_context": str(FULL_MOTION_NPZ),
        "cases": {
            "original_csv_reference_replay": replay,
            "reference_action_control": control,
        },
        "checks": {
            "source_csv_exists": SOURCE_CSV.is_file(),
            "source_csv_is_original_download": "download/official/LAFAN1_Retargeting_Dataset" in str(SOURCE_CSV),
            "reference_replay_ok": replay["status"] == "ok" and replay["checks"]["mp4_exists"],
            "reference_action_control_ok": control["status"] == "ok" and control["checks"]["mp4_exists"],
            "reference_action_control_uses_mj_step": control["checks"]["uses_mj_step"],
            "reference_action_control_fall_proxy_zero": control["checks"]["fall_proxy_count_zero"],
            "does_not_claim_teacher_vae_diffusion_guidance": True,
            "does_not_claim_paper_level": True,
            "does_not_claim_real_robot": True,
            "does_not_claim_goal_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": "Only jumps1 source/reference baselines were generated; teacher/RL, VAE, diffusion, guidance, and unassisted native MuJoCo policy gates remain blocked.",
        },
        "outputs": {
            "summary_json": str(summary_path),
            "readme": str(readme_path),
        },
    }
    write_json(summary_path, payload)
    readme_path.write_text(
        "\n".join(
            [
                "# LAFAN1 jumps1_subject1 MuJoCo Clean Baseline",
                "",
                "本目录展示原始 Unitree-retargeted LAFAN1 `jumps1_subject1.csv` 的 MuJoCo baseline。",
                "",
                "Claim boundary: source/reference baseline only; not teacher/RL, not VAE, not diffusion, not guidance, not real robot.",
                "",
                f"- Window: `{window['window_name']}` frames `{window['start_frame']}:{window['end_frame']}` "
                f"({window['source_start_time_s']:.2f}-{window['source_end_time_s']:.2f}s at {window['source_fps']} FPS)",
                f"- Original CSV replay MP4: `{replay['outputs']['mp4']}`",
                f"- Reference action control MP4: `{control['outputs']['mp4']}`",
                f"- Reference action control fall_proxy_count: `{control['metrics']['fall_proxy_count']}`",
                f"- Reference action control mean joint error: `{control['metrics']['joint_error_abs_mean']:.6f}`",
                "",
                "这一步只证明原始动作和 reference PD baseline 可在 MuJoCo 中展示；不能解锁 teacher/VAE/diffusion/guidance 长训练。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    window_name = os.environ.get("BM_JUMPS1_WINDOW", "high_dynamic_52s_67s")
    width = int(os.environ.get("BM_JUMPS1_WIDTH", "640"))
    height = int(os.environ.get("BM_JUMPS1_HEIGHT", "360"))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        window = window_from_csv(window_name)
        replay = render_reference_replay(window, width, height)
        control = render_reference_action_control(window, width, height)
        index = write_index(window, replay, control)
        print(json.dumps({"status": index["status"], "window": window_name, "out": str(OUT_ROOT / window_name)}))
        if index["status"] != "ok":
            raise SystemExit(1)
    except Exception as exc:  # noqa: BLE001
        out = OUT_ROOT / window_name / "lafan1_jumps1_subject1_mujoco_failed_summary.json"
        write_json(
            out,
            {
                "status": "failed",
                "timestamp_utc": utc_now(),
                "experiment_type": "lafan1_jumps1_subject1_mujoco_clean_deliverable",
                "window_name": window_name,
                "error": traceback_payload(exc),
            },
        )
        print(json.dumps({"status": "failed", "summary": str(out)}))
        raise


if __name__ == "__main__":
    main()
