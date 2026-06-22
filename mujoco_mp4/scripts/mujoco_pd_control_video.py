#!/usr/bin/env python3
"""Generate MuJoCo mj_step PD-control videos for existing local rollout targets.

This script is deliberately different from ``mujoco_trace_mesh_video.py``:
it does not write the robot qpos every rendered frame.  Instead it converts a
target motion/trace into a 29-DoF joint target sequence, drives the MuJoCo G1
with position-servo actuators, calls ``mj_step`` for physics, and records the
result.

Claim boundary: these are MuJoCo PD closed-loop tracking-control
visualizations.  They are not native MuJoCo PPO rollouts, because the local
IsaacLab 160-D observation manager has not been faithfully reconstructed in
MuJoCo and the video trace files do not contain the full 29-D action sequence.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mujoco_common import PKG, ROOT, render_frame, sha256, traceback_payload, utc_now, write_json
from mujoco_trace_mesh_video import (
    BODY_NAMES,
    DEFAULT_MODEL,
    DEFAULT_MOTION,
    TRACE_SPECS,
    body_ids,
    load_trace,
    resample_array,
    set_initial_pose,
    solve_ik_frame,
)


ACTION_SCALE_AUDIT = ROOT / "res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json"
PD_CAMERA = "bm_pd_fixed_center"

CONTROL_SPECS: dict[str, dict[str, Any]] = {
    "reference_control": {
        "target_source": "motion_joint_pos",
        "claim": "MuJoCo PD closed-loop tracking of FK-repaired reference joint targets",
    },
    "ppo_policy_control": {
        "target_source": "ik_from_ppo_policy_body_trace",
        "trace_spec": "ppo_policy",
        "claim": "MuJoCo PD closed-loop tracking of IK targets fitted from existing local IsaacLab PPO body trace; not native MuJoCo PPO policy",
    },
    "vae_base_control": {
        "target_source": "ik_from_vae_base_body_trace",
        "trace_spec": "vae_base",
        "claim": "MuJoCo PD closed-loop tracking of IK targets fitted from existing local IsaacLab VAE-base body trace; not native MuJoCo VAE controller",
    },
    "denoised_latent_control": {
        "target_source": "ik_from_denoised_latent_body_trace",
        "trace_spec": "denoised_latent",
        "claim": "MuJoCo PD closed-loop tracking of IK targets fitted from existing local IsaacLab denoised-latent body trace; not native MuJoCo denoiser controller",
    },
    "guided_latent_control": {
        "target_source": "ik_from_guided_latent_body_trace",
        "trace_spec": "guided_latent",
        "claim": "MuJoCo PD closed-loop tracking of IK targets fitted from existing local IsaacLab guided-latent body trace; not native MuJoCo guided controller",
    },
}


def load_action_rows() -> list[dict[str, Any]]:
    payload = json.loads(ACTION_SCALE_AUDIT.read_text(encoding="utf-8"))
    rows = payload.get("joint_rows", [])
    if len(rows) != 29:
        raise ValueError(f"Expected 29 action-scale rows, got {len(rows)}")
    return rows


def add_or_update_option(root: ET.Element) -> None:
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.set("timestep", os.environ.get("BM_MUJOCO_PD_TIMESTEP", "0.005"))
    option.set("gravity", "0 0 -9.81")
    option.set("integrator", os.environ.get("BM_MUJOCO_PD_INTEGRATOR", "implicitfast"))


def add_fixed_camera(root: ET.Element) -> None:
    for cam in root.findall(".//camera"):
        if cam.attrib.get("name") == PD_CAMERA:
            break
    else:
        world = root.find("worldbody")
        if world is None:
            raise ValueError("Cannot add camera: no worldbody in MuJoCo XML")
        cam = ET.SubElement(world, "camera", {"name": PD_CAMERA})
    cam.set("mode", "fixed")
    cam.set("pos", os.environ.get("BM_MUJOCO_PD_CAMERA_POS", "-0.35 -4.80 1.75"))
    cam.set("xyaxes", os.environ.get("BM_MUJOCO_PD_CAMERA_XYAXES", "1 0 0 0 0.32 0.947"))
    cam.set("fovy", os.environ.get("BM_MUJOCO_PD_CAMERA_FOVY", "48"))

    statistic = root.find("statistic")
    if statistic is not None:
        statistic.set("center", "0 0 0.85")
        statistic.set("extent", "2.0")


def patch_joints_and_actuators(model_xml: Path, out_xml: Path, rows: list[dict[str, Any]]) -> Path:
    import mujoco

    source_model = mujoco.MjModel.from_xml_path(str(model_xml))
    joint_ranges: dict[str, tuple[float, float]] = {}
    for j in range(source_model.njnt):
        name = mujoco.mj_id2name(source_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if not name or name == "pelvis":
            continue
        joint_ranges[name] = (float(source_model.jnt_range[j, 0]), float(source_model.jnt_range[j, 1]))

    tree = ET.parse(model_xml)
    root = tree.getroot()
    add_or_update_option(root)
    add_fixed_camera(root)

    rows_by_joint = {str(row["joint_name"]): row for row in rows}
    for joint in root.findall(".//joint"):
        name = joint.attrib.get("name", "")
        row = rows_by_joint.get(name)
        if not row:
            continue
        joint.set("armature", f"{float(row['armature']):.8g}")
        joint.set("damping", f"{float(row['damping']):.8g}")
        joint.set("actuatorfrcrange", f"{-float(row['effort_limit_sim']):.6g} {float(row['effort_limit_sim']):.6g}")

    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    actuator.clear()
    kp_scale = float(os.environ.get("BM_MUJOCO_PD_KP_SCALE", "1.0"))
    kv_scale = float(os.environ.get("BM_MUJOCO_PD_KV_SCALE", "1.0"))
    for row in rows:
        joint = str(row["joint_name"])
        lo, hi = joint_ranges[joint]
        effort = float(row["effort_limit_sim"])
        ET.SubElement(
            actuator,
            "position",
            {
                "name": joint,
                "joint": joint,
                "kp": f"{float(row['stiffness']) * kp_scale:.8g}",
                "kv": f"{float(row['damping']) * kv_scale:.8g}",
                "ctrlrange": f"{lo:.8g} {hi:.8g}",
                "ctrllimited": "true",
                "forcerange": f"{-effort:.8g} {effort:.8g}",
                "forcelimited": "true",
            },
        )
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_xml, encoding="utf-8", xml_declaration=False)
    return out_xml


def normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    return quat / norm if norm > 0 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = [float(v) for v in q1]
    w2, x2, y2, z2 = [float(v) for v in q2]
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_error_rotvec(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    q_err = normalize_quat_wxyz(quat_mul(normalize_quat_wxyz(target), quat_conj(normalize_quat_wxyz(current))))
    if q_err[0] < 0:
        q_err = -q_err
    vec = q_err[1:4]
    vec_norm = float(np.linalg.norm(vec))
    if vec_norm < 1e-8:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vec_norm, float(q_err[0]))
    return vec / vec_norm * angle


def load_reference_targets(frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    motion_path = Path(os.environ.get("BM_MUJOCO_MOTION_NPZ", str(DEFAULT_MOTION))).expanduser()
    motion = np.load(motion_path, allow_pickle=True)
    targets = resample_array(np.asarray(motion["joint_pos"], dtype=np.float64), frames)
    root_pos = resample_array(np.asarray(motion["body_pos_w"][:, 0, :], dtype=np.float64), frames)
    root_pos[:, 0:2] = 0.0
    root_quat = resample_array(np.asarray(motion["body_quat_w"][:, 0, :], dtype=np.float64), frames)
    root_quat = np.stack([normalize_quat_wxyz(q) for q in root_quat], axis=0)
    return targets, root_pos, root_quat, {
        "target_source_file": str(motion_path),
        "target_source_sha256": sha256(motion_path),
        "target_source_frames": int(motion["joint_pos"].shape[0]),
        "initial_root_z": float(root_pos[0, 2]),
        "root_xy_recentered_targets": True,
        "ik_target_error_mean_m": 0.0,
        "ik_target_error_max_m": 0.0,
    }


def trace_to_ik_targets(spec_name: str, frames: int, model_xml: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import mujoco

    trace_path, arrays, meta = load_trace(spec_name)
    pose = resample_array(arrays["pose"], frames)
    target = pose.copy()
    root_xy = target[:, 0, :2].copy()
    target[:, :, 0:2] -= root_xy[:, None, :]
    root_z = float(np.median(target[:, 0, 2]))

    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    ids = body_ids(model)
    set_initial_pose(model, data, target[0])
    prev_qpos: np.ndarray | None = None
    joint_targets = np.zeros((frames, 29), dtype=np.float64)
    root_pos_targets = np.zeros((frames, 3), dtype=np.float64)
    root_quat_targets = np.zeros((frames, 4), dtype=np.float64)
    ik_rows: list[dict[str, float]] = []
    for frame_idx in range(frames):
        qpos, metrics = solve_ik_frame(model, data, ids, target[frame_idx], prev_qpos)
        prev_qpos = qpos
        joint_targets[frame_idx] = qpos[7 : 7 + 29]
        root_pos_targets[frame_idx] = qpos[0:3]
        root_pos_targets[frame_idx, 0:2] = 0.0
        root_quat_targets[frame_idx] = normalize_quat_wxyz(qpos[3:7])
        ik_rows.append(metrics)
    return joint_targets, root_pos_targets, root_quat_targets, {
        "target_source_file": str(trace_path),
        "target_source_sha256": sha256(trace_path),
        "target_pose_key": meta["pose_key"],
        "target_reference_key": meta["reference_key"],
        "target_source_frames": int(arrays["pose"].shape[0]),
        "initial_root_z": root_z,
        "root_xy_recentered_targets": True,
        "ik_target_error_mean_m": float(np.mean([row["ik_error_mean_m"] for row in ik_rows])),
        "ik_target_error_max_m": float(np.max([row["ik_error_max_m"] for row in ik_rows])),
    }


def apply_root_assist(model, data, body_id: int, target_pos: np.ndarray, target_quat: np.ndarray) -> None:
    use_assist = os.environ.get("BM_MUJOCO_ROOT_ASSIST", "1") == "1"
    if not use_assist:
        return
    kp_pos = float(os.environ.get("BM_MUJOCO_ROOT_POS_KP", "900.0"))
    kd_pos = float(os.environ.get("BM_MUJOCO_ROOT_POS_KD", "120.0"))
    kp_rot = float(os.environ.get("BM_MUJOCO_ROOT_ROT_KP", "220.0"))
    kd_rot = float(os.environ.get("BM_MUJOCO_ROOT_ROT_KD", "35.0"))
    max_force = float(os.environ.get("BM_MUJOCO_ROOT_MAX_FORCE", "2500.0"))
    max_torque = float(os.environ.get("BM_MUJOCO_ROOT_MAX_TORQUE", "900.0"))
    current_pos = data.xpos[body_id].copy()
    current_quat = normalize_quat_wxyz(data.xquat[body_id].copy())
    lin_vel = data.qvel[0:3].copy()
    ang_vel = data.qvel[3:6].copy()
    force = kp_pos * (target_pos - current_pos) - kd_pos * lin_vel
    torque = kp_rot * quat_error_rotvec(target_quat, current_quat) - kd_rot * ang_vel
    force = np.clip(force, -max_force, max_force)
    torque = np.clip(torque, -max_torque, max_torque)
    data.xfrc_applied[body_id, 0:3] = force
    data.xfrc_applied[body_id, 3:6] = torque


def actuator_joint_order(model) -> list[str]:
    import mujoco

    names: list[str] = []
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or "")
    return names


def quat_to_roll_pitch_yaw(quat: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = [float(v) for v in quat]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def render_control_video(spec_name: str) -> dict[str, Any]:
    import mujoco

    if spec_name not in CONTROL_SPECS:
        raise KeyError(f"Unknown control spec {spec_name}; available={sorted(CONTROL_SPECS)}")
    spec = CONTROL_SPECS[spec_name]
    backend = os.environ.get("MUJOCO_GL", "egl")
    frames = int(os.environ.get("BM_MUJOCO_CONTROL_FRAMES", "450"))
    fps = int(os.environ.get("BM_MUJOCO_VIDEO_FPS", "30"))
    width = int(os.environ.get("BM_MUJOCO_WIDTH", "960"))
    height = int(os.environ.get("BM_MUJOCO_HEIGHT", "540"))
    substeps = int(os.environ.get("BM_MUJOCO_CONTROL_SUBSTEPS", "4"))
    settle_steps = int(os.environ.get("BM_MUJOCO_CONTROL_SETTLE_STEPS", "40"))
    model_path = Path(os.environ.get("BM_MUJOCO_G1_XML", str(DEFAULT_MODEL))).expanduser()
    rows = load_action_rows()
    patched_xml = model_path.parent / "g1_mocap_29dof_pd_control.xml"
    patch_joints_and_actuators(model_path, patched_xml, rows)

    if spec["target_source"] == "motion_joint_pos":
        joint_targets, root_pos_targets, root_quat_targets, target_meta = load_reference_targets(frames)
    else:
        joint_targets, root_pos_targets, root_quat_targets, target_meta = trace_to_ik_targets(str(spec["trace_spec"]), frames, model_path)

    model = mujoco.MjModel.from_xml_path(str(patched_xml))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    names = actuator_joint_order(model)
    if names != [row["joint_name"] for row in rows]:
        raise RuntimeError("Actuator joint order does not match action-scale audit order")
    pelvis_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_body < 0:
        raise RuntimeError("MuJoCo body 'pelvis' not found")

    out_dir = PKG / "res/control_videos" / spec_name
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / f"{spec_name}.mp4"
    keyframe_path = out_dir / f"{spec_name}_keyframe.png"
    metrics_path = out_dir / f"{spec_name}_metrics.csv"
    summary_path = out_dir / f"{spec_name}_summary.json"

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = root_pos_targets[0]
    data.qpos[3:7] = root_quat_targets[0]
    data.qpos[7 : 7 + 29] = joint_targets[0]
    data.ctrl[:] = np.clip(joint_targets[0], model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    mujoco.mj_forward(model, data)
    for _ in range(settle_steps):
        data.xfrc_applied[:] = 0.0
        apply_root_assist(model, data, pelvis_body, root_pos_targets[0], root_quat_targets[0])
        mujoco.mj_step(model, data)

    rows_out: list[dict[str, Any]] = []
    with imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame_idx in range(frames):
            target = np.clip(joint_targets[frame_idx], model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            data.ctrl[:] = target
            for _ in range(substeps):
                data.xfrc_applied[:] = 0.0
                apply_root_assist(model, data, pelvis_body, root_pos_targets[frame_idx], root_quat_targets[frame_idx])
                mujoco.mj_step(model, data)
            frame = render_frame(model, data, renderer, camera=PD_CAMERA)
            if frame_idx == 0:
                imageio.imwrite(keyframe_path, frame)
            writer.append_data(frame)
            q = data.qpos[7 : 7 + 29].copy()
            qd = data.qvel[6 : 6 + 29].copy()
            err = q - target
            roll, pitch, yaw = quat_to_roll_pitch_yaw(data.qpos[3:7])
            rows_out.append(
                {
                    "frame": frame_idx,
                    "video_time_s": frame_idx / fps,
                    "sim_time_s": float(data.time),
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
                    "root_target_x": float(root_pos_targets[frame_idx, 0]),
                    "root_target_y": float(root_pos_targets[frame_idx, 1]),
                    "root_target_z": float(root_pos_targets[frame_idx, 2]),
                    "root_position_error_m": float(np.linalg.norm(data.xpos[pelvis_body] - root_pos_targets[frame_idx])),
                    "root_orientation_error_rad": float(
                        np.linalg.norm(quat_error_rotvec(root_quat_targets[frame_idx], data.xquat[pelvis_body]))
                    ),
                    "contact_count": int(data.ncon),
                    "fall_proxy": bool(data.qpos[2] < 0.45 or abs(roll) > 1.2 or abs(pitch) > 1.2),
                }
            )
    renderer.close()

    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        fields = list(rows_out[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)

    fall_count = sum(1 for row in rows_out if row["fall_proxy"])
    payload = {
        "status": "ok",
        "timestamp_utc": utc_now(),
        "experiment_type": "mujoco_pd_closed_loop_control_video",
        "spec_name": spec_name,
        "backend": backend,
        "claim_level": spec["claim"],
        "source_model_xml": str(model_path),
        "patched_pd_model_xml": str(patched_xml),
        "action_scale_audit": str(ACTION_SCALE_AUDIT),
        "target_source": spec["target_source"],
        "target_metadata": target_meta,
        "frames_rendered": frames,
        "video_fps": fps,
        "duration_seconds": frames / fps,
        "simulation": {
            "uses_mj_step": True,
            "writes_qpos_each_frame": False,
            "actuator_type": "position",
            "actuator_count": int(model.nu),
            "control_substeps_per_frame": substeps,
            "settle_steps": settle_steps,
            "timestep": float(model.opt.timestep),
            "sim_time_s": float(data.time),
            "root_assist_enabled": os.environ.get("BM_MUJOCO_ROOT_ASSIST", "1") == "1",
            "root_assist_type": "external pelvis force/torque stabilizer applied before mj_step",
        },
        "camera": {
            "name": PD_CAMERA,
            "mode": "fixed",
            "dynamic_camera": False,
            "position": os.environ.get("BM_MUJOCO_PD_CAMERA_POS", "-0.35 -4.80 1.75"),
            "fovy": os.environ.get("BM_MUJOCO_PD_CAMERA_FOVY", "48"),
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
        "metrics": {
            "joint_error_abs_mean": float(np.mean([row["joint_error_abs_mean"] for row in rows_out])),
            "joint_error_abs_max": float(np.max([row["joint_error_abs_max"] for row in rows_out])),
            "root_position_error_mean_m": float(np.mean([row["root_position_error_m"] for row in rows_out])),
            "root_position_error_max_m": float(np.max([row["root_position_error_m"] for row in rows_out])),
            "root_orientation_error_mean_rad": float(np.mean([row["root_orientation_error_rad"] for row in rows_out])),
            "root_height_min": float(np.min([row["root_z"] for row in rows_out])),
            "root_height_max": float(np.max([row["root_z"] for row in rows_out])),
            "root_xy_abs_max": float(np.max([max(abs(row["root_x"]), abs(row["root_y"])) for row in rows_out])),
            "contact_count_mean": float(np.mean([row["contact_count"] for row in rows_out])),
            "fall_proxy_count": int(fall_count),
        },
        "checks": {
            "mp4_exists": mp4_path.is_file() and mp4_path.stat().st_size > 0,
            "keyframe_exists": keyframe_path.is_file() and keyframe_path.stat().st_size > 0,
            "metrics_csv_exists": metrics_path.is_file() and metrics_path.stat().st_size > 0,
            "uses_mujoco_g1_mesh": True,
            "uses_mj_step": True,
            "does_not_write_qpos_each_frame": True,
            "uses_root_assist_controller": os.environ.get("BM_MUJOCO_ROOT_ASSIST", "1") == "1",
            "uses_29_position_actuators": int(model.nu) == 29,
            "native_mujoco_ppo_obs_adapter": False,
            "does_not_claim_native_mujoco_policy_controller": True,
            "does_not_claim_isaaclab_render": True,
            "does_not_claim_real_robot": True,
        },
        "limitations": [
            "This is a MuJoCo PD target-tracking controller, not the native IsaacLab PPO observation/action adapter.",
            "A pelvis force/torque root-assist controller is used to keep the robot centered and upright in the fixed camera; this is not an unassisted humanoid balance policy.",
            "Trace-derived targets use offline IK from existing local IsaacLab rollout body positions because full 29-D action traces were not saved in the video trace files.",
            "The floating base is not directly actuated; falls or drift are physical outcomes of this local MuJoCo controller.",
        ],
    }
    write_json(summary_path, payload)
    print(json.dumps({"status": "ok", "spec": spec_name, "mp4": str(mp4_path), "fall_proxy_count": fall_count}))
    return payload


def render_side_by_side(left: dict[str, Any], right: dict[str, Any], output: str = "guided_vs_unguided_control") -> dict[str, Any]:
    out_dir = PKG / "res/control_videos" / output
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / f"{output}.mp4"
    keyframe = out_dir / f"{output}_keyframe.png"
    summary = out_dir / f"{output}_summary.json"
    fps = int(os.environ.get("BM_MUJOCO_VIDEO_FPS", "30"))
    reader_l = imageio.get_reader(left["outputs"]["mp4"])
    reader_r = imageio.get_reader(right["outputs"]["mp4"])
    n = min(int(left["frames_rendered"]), int(right["frames_rendered"]))
    with imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for i in range(n):
            frame = np.concatenate([reader_l.get_data(i), reader_r.get_data(i)], axis=1)
            if i == 0:
                imageio.imwrite(keyframe, frame)
            writer.append_data(frame)
    reader_l.close()
    reader_r.close()
    payload = {
        "status": "ok",
        "timestamp_utc": utc_now(),
        "experiment_type": "mujoco_pd_guided_vs_unguided_control_side_by_side",
        "claim_level": "Side-by-side MuJoCo PD closed-loop tracking-control videos: VAE-base target tracking on the left, guided-latent target tracking on the right; not native MuJoCo guidance",
        "left_label": "unguided / VAE-base target tracking",
        "right_label": "guided / guided-latent target tracking",
        "left": left["outputs"],
        "right": right["outputs"],
        "frames_rendered": n,
        "video_fps": fps,
        "duration_seconds": n / fps,
        "outputs": {"mp4": str(out_mp4), "keyframe_png": str(keyframe), "summary_json": str(summary)},
        "checks": {
            "mp4_exists": out_mp4.is_file() and out_mp4.stat().st_size > 0,
            "keyframe_exists": keyframe.is_file() and keyframe.stat().st_size > 0,
            "uses_mj_step_source_videos": left["checks"]["uses_mj_step"] and right["checks"]["uses_mj_step"],
            "does_not_claim_native_mujoco_guidance": True,
            "does_not_claim_real_robot": True,
        },
    }
    write_json(summary, payload)
    print(json.dumps({"status": "ok", "spec": output, "mp4": str(out_mp4)}))
    return payload


def main() -> None:
    specs_env = os.environ.get(
        "BM_MUJOCO_CONTROL_SPECS",
        "reference_control,ppo_policy_control,vae_base_control,denoised_latent_control,guided_latent_control",
    )
    specs = [spec.strip() for spec in specs_env.split(",") if spec.strip()]
    rendered: dict[str, Any] = {}
    try:
        for spec_name in specs:
            rendered[spec_name] = render_control_video(spec_name)
        if "vae_base_control" in rendered and "guided_latent_control" in rendered:
            render_side_by_side(rendered["vae_base_control"], rendered["guided_latent_control"])
    except Exception as exc:  # noqa: BLE001
        out = PKG / "res/control_videos/failed_control_video_summary.json"
        write_json(
            out,
            {
                "status": "failed",
                "timestamp_utc": utc_now(),
                "experiment_type": "mujoco_pd_closed_loop_control_video",
                "error": traceback_payload(exc),
            },
        )
        raise


if __name__ == "__main__":
    main()
