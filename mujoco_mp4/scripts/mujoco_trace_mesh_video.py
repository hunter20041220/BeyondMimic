#!/usr/bin/env python3
"""Render existing IsaacLab closed-loop body-pose traces as MuJoCo G1 mesh videos.

This script intentionally does not claim to reconstruct the IsaacLab policy
observation manager inside MuJoCo.  It uses already-captured local IsaacLab
closed-loop traces, fits the MuJoCo G1 mesh to the 14 tracked body positions
with damped least-squares IK, and renders report-ready MP4 assets.
"""

from __future__ import annotations

import csv
import json
import math
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
BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]

TRACE_SPECS = {
    "ppo_policy": {
        "trace": ROOT
        / "res/runs/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_policy_rollout_video_capture/"
        "policy_rollout_capture_20260621_131648_seed20260722/official_csv_loop_policy_rollout_body_pose_trace.npz",
        "pose_key": "robot_body_pos_w",
        "reference_key": "reference_body_pos_w",
        "metric_prefix": "",
        "claim": "MuJoCo mesh rendering of existing local IsaacLab PPO closed-loop body-pose trace; not native MuJoCo PPO controller",
    },
    "vae_base": {
        "trace": ROOT
        / "res/runs/tracking_g1_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval/"
        "joystick/receding_latent_guidance_rollout_20260621_064022_seed20260705/"
        "official_csv_loop_receding_latent_guidance_rollout_trace.npz",
        "pose_key": "vae_base_robot_body_pos_w",
        "reference_key": "vae_base_reference_body_pos_w",
        "metric_prefix": "vae_base_",
        "claim": "MuJoCo mesh rendering of existing local IsaacLab VAE-base closed-loop trace; not native MuJoCo VAE controller",
    },
    "guided_latent": {
        "trace": ROOT
        / "res/runs/tracking_g1_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval/"
        "joystick/receding_latent_guidance_rollout_20260621_064022_seed20260705/"
        "official_csv_loop_receding_latent_guidance_rollout_trace.npz",
        "pose_key": "receding_latent_guided_robot_body_pos_w",
        "reference_key": "receding_latent_guided_reference_body_pos_w",
        "metric_prefix": "receding_latent_guided_",
        "claim": "MuJoCo mesh rendering of existing local IsaacLab guided latent closed-loop trace; not native MuJoCo guided controller",
    },
    "denoised_latent": {
        "trace": ROOT
        / "res/runs/tracking_g1_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval/"
        "joystick/receding_latent_guidance_rollout_20260621_064022_seed20260705/"
        "official_csv_loop_receding_latent_guidance_rollout_trace.npz",
        "pose_key": "denoised_latent_robot_body_pos_w",
        "reference_key": "denoised_latent_reference_body_pos_w",
        "metric_prefix": "denoised_latent_",
        "claim": "MuJoCo mesh rendering of existing local IsaacLab denoised latent closed-loop trace; not native MuJoCo denoiser controller",
    },
}

DEFAULT_MOTION = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/motions/walk1_subject1/motion.npz"
)


def resample_array(array: np.ndarray, frames: int) -> np.ndarray:
    if array.shape[0] == frames:
        return array.copy()
    old_x = np.linspace(0.0, 1.0, array.shape[0])
    new_x = np.linspace(0.0, 1.0, frames)
    flat = array.reshape(array.shape[0], -1)
    out = np.empty((frames, flat.shape[1]), dtype=np.float64)
    for col in range(flat.shape[1]):
        out[:, col] = np.interp(new_x, old_x, flat[:, col])
    return out.reshape((frames,) + array.shape[1:])


def add_camera_xml(model_xml: Path, out_xml: Path, center_xy: np.ndarray | None = None) -> Path:
    camera_name = "bm_trace_track"
    text = model_xml.read_text(encoding="utf-8")
    if center_xy is None:
        center_xy = np.array([0.0, 0.0], dtype=np.float64)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    if f'name="{camera_name}"' not in text:
        camera = (
            f'<camera name="{camera_name}" mode="fixed" pos="{cx - 0.20:.4f} {cy - 4.45:.4f} 1.70" '
            'xyaxes="1 0 0 0 0.30 0.95" fovy="42"/>'
        )
        text = text.replace("</worldbody>", camera + "\n  </worldbody>", 1)
    out_xml.write_text(text, encoding="utf-8")
    return out_xml


def load_trace(spec_name: str, override: str | None = None) -> tuple[Path, dict[str, np.ndarray], dict[str, object]]:
    spec = TRACE_SPECS[spec_name]
    path = Path(override).expanduser() if override else Path(spec["trace"])
    data = np.load(path, allow_pickle=True)
    pose_key = str(spec["pose_key"])
    reference_key = str(spec["reference_key"])
    if pose_key not in data:
        raise KeyError(f"{pose_key} missing from {path}")
    pose = np.asarray(data[pose_key], dtype=np.float64)
    if pose.ndim != 3 or pose.shape[1:] != (14, 3):
        raise ValueError(f"{pose_key} expected shape (T, 14, 3), got {pose.shape}")
    ref = np.asarray(data[reference_key], dtype=np.float64) if reference_key in data else pose.copy()
    aux: dict[str, np.ndarray] = {"pose": pose, "reference": ref}
    prefix = str(spec.get("metric_prefix", ""))
    for metric in ["rewards", "dones", "action_abs_mean", "target_body_error_mean", "guidance_cost_before", "guidance_cost_after"]:
        key = prefix + metric
        if key in data:
            aux[metric] = np.asarray(data[key])
        elif metric in data:
            aux[metric] = np.asarray(data[metric])
    meta = {"claim_level": spec["claim"], "pose_key": pose_key, "reference_key": reference_key, "metric_prefix": prefix}
    return path, aux, meta


def body_ids(model) -> list[int]:
    import mujoco

    ids = []
    for name in BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"MuJoCo body not found: {name}")
        ids.append(bid)
    return ids


def set_initial_pose(model, data, target: np.ndarray) -> None:
    import mujoco

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    pelvis = target[0]
    data.qpos[0:3] = pelvis
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    if model.nq >= 36:
        data.qpos[7:36] = 0.0
    mujoco.mj_forward(model, data)


def solve_ik_frame(model, data, ids: list[int], target: np.ndarray, prev_qpos: np.ndarray | None) -> tuple[np.ndarray, dict[str, float]]:
    import mujoco

    if prev_qpos is not None:
        data.qpos[:] = prev_qpos
        mujoco.mj_forward(model, data)
    target_center = target[0]
    current_pelvis = data.xpos[ids[0]].copy()
    data.qpos[0:3] += target_center - current_pelvis
    mujoco.mj_forward(model, data)

    nv = model.nv
    lambda2 = float(os.environ.get("BM_MUJOCO_IK_DAMPING", "0.002"))
    step_scale = float(os.environ.get("BM_MUJOCO_IK_STEP_SCALE", "0.75"))
    iters = int(os.environ.get("BM_MUJOCO_IK_ITERS", "18"))
    max_step = float(os.environ.get("BM_MUJOCO_IK_MAX_STEP", "0.10"))
    body_weight = np.ones(len(ids), dtype=np.float64)
    body_weight[0] = 2.0
    body_weight[7] = 1.6

    for _ in range(iters):
        residuals = []
        jac_blocks = []
        for wi, bid in zip(body_weight, ids):
            err = (target[ids.index(bid)] - data.xpos[bid]) * wi
            jp = np.zeros((3, nv), dtype=np.float64)
            jr = np.zeros((3, nv), dtype=np.float64)
            mujoco.mj_jacBody(model, data, jp, jr, bid)
            residuals.append(err)
            jac_blocks.append(jp * wi)
        r = np.concatenate(residuals)
        j = np.vstack(jac_blocks)
        if float(np.linalg.norm(r)) < 1e-4:
            break
        lhs = j.T @ j + lambda2 * np.eye(nv)
        rhs = j.T @ r
        dq = np.linalg.solve(lhs, rhs)
        norm = float(np.linalg.norm(dq))
        if norm > max_step:
            dq *= max_step / norm
        mujoco.mj_integratePos(model, data.qpos, dq * step_scale, 1.0)
        mujoco.mj_forward(model, data)

    err_vec = np.stack([target[i] - data.xpos[bid] for i, bid in enumerate(ids)], axis=0)
    norms = np.linalg.norm(err_vec, axis=1)
    return data.qpos.copy(), {
        "ik_error_mean_m": float(np.mean(norms)),
        "ik_error_max_m": float(np.max(norms)),
        "pelvis_error_m": float(norms[0]),
        "torso_error_m": float(norms[7]),
    }


def update_camera(model, data, center: np.ndarray) -> None:
    import mujoco

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "bm_trace_track")
    if cam_id < 0:
        return
    data.cam_xpos[cam_id] = np.array([center[0] - 0.25, center[1] - 4.35, max(1.35, center[2] + 0.80)])
    # Fixed xyaxes keeps the robot centered without requiring quaternion math.
    data.cam_xmat[cam_id] = np.array([1.0, 0.0, 0.0, 0.0, 0.30, 0.9539392, 0.0, -0.9539392, 0.30])


def render_single(spec_name: str, output_name: str | None = None, trace_override: str | None = None) -> dict[str, object]:
    import mujoco

    backend = os.environ.get("MUJOCO_GL", "egl")
    model_path = Path(os.environ.get("BM_MUJOCO_G1_XML", str(DEFAULT_MODEL))).expanduser()
    trace_path, arrays, meta = load_trace(spec_name, trace_override)
    pose = arrays["pose"]
    reference = arrays["reference"]
    render_mode = os.environ.get("BM_MUJOCO_TRACE_RENDER_MODE", "root_qpos_replay")
    requested_frames = int(os.environ.get("BM_MUJOCO_TRACE_FRAMES", "450"))
    source_frames = int(pose.shape[0])
    frames = requested_frames
    pose_render = resample_array(pose, frames)
    reference_render = resample_array(reference, frames)
    output = output_name or spec_name
    out_dir = PKG / "res/rollout_videos" / output
    out_dir.mkdir(parents=True, exist_ok=True)
    patched_xml = model_path.parent / f"{model_path.stem}_{output}_trace_camera.xml"
    trace_center_xy = np.mean(pose_render[:, 0, :2], axis=0)
    add_camera_xml(model_path, patched_xml, trace_center_xy)

    model = mujoco.MjModel.from_xml_path(str(patched_xml))
    data = mujoco.MjData(model)
    ids = body_ids(model)
    set_initial_pose(model, data, pose[0])
    renderer = mujoco.Renderer(model, height=int(os.environ.get("BM_MUJOCO_HEIGHT", "720")), width=int(os.environ.get("BM_MUJOCO_WIDTH", "1280")))
    motion_path = Path(os.environ.get("BM_MUJOCO_MOTION_NPZ", str(DEFAULT_MOTION))).expanduser()
    motion = np.load(motion_path, allow_pickle=True)
    motion_joint_pos = resample_array(np.asarray(motion["joint_pos"], dtype=np.float64), frames)
    motion_joint_vel = resample_array(np.asarray(motion["joint_vel"], dtype=np.float64), frames)
    motion_body_quat = resample_array(np.asarray(motion["body_quat_w"][:, 0, :], dtype=np.float64), frames)

    fps = int(os.environ.get("BM_MUJOCO_VIDEO_FPS", "30"))
    mp4_path = out_dir / f"{output}.mp4"
    keyframe_path = out_dir / f"{output}_keyframe.png"
    metrics_path = out_dir / f"{output}_metrics.csv"
    summary_path = out_dir / f"{output}_summary.json"
    rows: list[dict[str, object]] = []
    prev_qpos: np.ndarray | None = None
    with imageio.get_writer(mp4_path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame_idx in range(frames):
            if render_mode == "ik":
                prev_qpos, ik = solve_ik_frame(model, data, ids, pose_render[frame_idx], prev_qpos)
            elif render_mode == "root_qpos_replay":
                data.qpos[:] = 0.0
                data.qvel[:] = 0.0
                data.qpos[0:3] = pose_render[frame_idx, 0]
                quat = motion_body_quat[frame_idx]
                norm = float(np.linalg.norm(quat))
                data.qpos[3:7] = quat / norm if norm > 0 else np.array([1.0, 0.0, 0.0, 0.0])
                data.qpos[7 : 7 + 29] = motion_joint_pos[frame_idx]
                data.qvel[6 : 6 + 29] = motion_joint_vel[frame_idx]
                mujoco.mj_forward(model, data)
                fitted = np.stack([data.xpos[bid].copy() for bid in ids], axis=0)
                err = np.linalg.norm(fitted - pose_render[frame_idx], axis=1)
                ik = {
                    "ik_error_mean_m": float(np.mean(err)),
                    "ik_error_max_m": float(np.max(err)),
                    "pelvis_error_m": float(err[0]),
                    "torso_error_m": float(err[7]),
                }
            else:
                raise ValueError(f"Unknown BM_MUJOCO_TRACE_RENDER_MODE={render_mode}")
            frame = render_frame(model, data, renderer, camera="bm_trace_track")
            if frame_idx == 0:
                imageio.imwrite(keyframe_path, frame)
            writer.append_data(frame)
            ref_err = np.linalg.norm(pose_render[frame_idx] - reference_render[frame_idx], axis=1)
            row = {
                "frame": frame_idx,
                "time_s": frame_idx / fps,
                "root_x": float(pose_render[frame_idx, 0, 0]),
                "root_y": float(pose_render[frame_idx, 0, 1]),
                "root_z": float(pose_render[frame_idx, 0, 2]),
                "trace_reference_error_mean_m": float(np.mean(ref_err)),
                "trace_reference_error_max_m": float(np.max(ref_err)),
                **ik,
            }
            for metric in ["rewards", "dones", "action_abs_mean", "target_body_error_mean", "guidance_cost_before", "guidance_cost_after"]:
                if metric in arrays and frame_idx < len(arrays[metric]):
                    values = np.asarray(arrays[metric])
                    metric_values = resample_array(values.reshape(values.shape[0], 1), frames).reshape(frames)
                    row[metric] = float(metric_values[frame_idx])
            rows.append(row)
    renderer.close()
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        fields = sorted({key for row in rows for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "status": "ok",
        "timestamp_utc": utc_now(),
        "experiment_type": "mujoco_trace_mesh_video",
        "spec_name": spec_name,
        "backend": backend,
        "claim_level": meta["claim_level"],
        "source_trace": str(trace_path),
        "source_trace_sha256": sha256(trace_path),
        "source_pose_key": meta["pose_key"],
        "source_reference_key": meta["reference_key"],
        "source_model_xml": str(model_path),
        "motion_qpos_npz": str(motion_path),
        "motion_qpos_sha256": sha256(motion_path),
        "render_mode": render_mode,
        "source_frames": source_frames,
        "frames_rendered": frames,
        "video_fps": fps,
        "duration_seconds": frames / fps,
        "body_names": BODY_NAMES,
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
            "ik_error_mean_m": float(np.mean([r["ik_error_mean_m"] for r in rows])),
            "ik_error_max_m": float(np.max([r["ik_error_max_m"] for r in rows])),
            "trace_reference_error_mean_m": float(np.mean([r["trace_reference_error_mean_m"] for r in rows])),
            "trace_reference_error_max_m": float(np.max([r["trace_reference_error_max_m"] for r in rows])),
        },
        "checks": {
            "mp4_exists": mp4_path.is_file() and mp4_path.stat().st_size > 0,
            "keyframe_exists": keyframe_path.is_file() and keyframe_path.stat().st_size > 0,
            "metrics_csv_exists": metrics_path.is_file() and metrics_path.stat().st_size > 0,
            "uses_mujoco_g1_mesh": True,
            "uses_existing_closed_loop_trace": True,
            "uses_motion_qpos_for_fast_mesh_pose": render_mode == "root_qpos_replay",
            "does_not_claim_native_mujoco_policy_controller": True,
            "does_not_claim_isaaclab_render": True,
            "does_not_claim_real_robot": True,
        },
        "limitations": [
            "The source body-pose trace was produced by local IsaacLab closed-loop/proxy rollout code.",
            "MuJoCo is used here as a mesh renderer with IK fitting, not as the original controller simulator.",
            "This resolves report-ready mesh visualization, not the native MuJoCo 160D observation adapter.",
        ],
    }
    write_json(summary_path, payload)
    print(json.dumps({"status": "ok", "spec": spec_name, "mp4": str(mp4_path), "frames": frames}))
    return payload


def render_side_by_side(left: dict[str, object], right: dict[str, object], output: str = "guided_vs_unguided") -> dict[str, object]:
    from PIL import Image

    left_mp4 = Path(left["outputs"]["mp4"])
    right_mp4 = Path(right["outputs"]["mp4"])
    out_dir = PKG / "res/rollout_videos" / output
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / f"{output}.mp4"
    keyframe = out_dir / f"{output}_keyframe.png"
    summary = out_dir / f"{output}_summary.json"
    fps = int(os.environ.get("BM_MUJOCO_VIDEO_FPS", "30"))
    reader_l = imageio.get_reader(left_mp4)
    reader_r = imageio.get_reader(right_mp4)
    n = min(int(left["frames_rendered"]), int(right["frames_rendered"]))
    with imageio.get_writer(out_mp4, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for i in range(n):
            fl = reader_l.get_data(i)
            fr = reader_r.get_data(i)
            frame = np.concatenate([fl, fr], axis=1)
            if i == 0:
                imageio.imwrite(keyframe, frame)
            writer.append_data(frame)
    reader_l.close()
    reader_r.close()
    payload = {
        "status": "ok",
        "timestamp_utc": utc_now(),
        "experiment_type": "mujoco_guided_vs_unguided_side_by_side",
        "claim_level": "Side-by-side MuJoCo mesh rendering of existing local IsaacLab VAE-base and guided latent traces; not native MuJoCo guided controller",
        "left": left["outputs"],
        "right": right["outputs"],
        "frames_rendered": n,
        "video_fps": fps,
        "duration_seconds": n / fps,
        "outputs": {"mp4": str(out_mp4), "keyframe_png": str(keyframe), "summary_json": str(summary)},
        "checks": {
            "mp4_exists": out_mp4.is_file() and out_mp4.stat().st_size > 0,
            "keyframe_exists": keyframe.is_file() and keyframe.stat().st_size > 0,
            "does_not_claim_native_mujoco_guidance": True,
            "does_not_claim_real_robot": True,
        },
    }
    write_json(summary, payload)
    print(json.dumps({"status": "ok", "spec": output, "mp4": str(out_mp4), "frames": n}))
    return payload


def main() -> None:
    specs_env = os.environ.get("BM_MUJOCO_TRACE_SPECS", "ppo_policy,vae_base,guided_latent")
    specs = [s.strip() for s in specs_env.split(",") if s.strip()]
    rendered = {}
    for spec in specs:
        rendered[spec] = render_single(spec)
    if "vae_base" in rendered and "guided_latent" in rendered:
        render_side_by_side(rendered["vae_base"], rendered["guided_latent"])


if __name__ == "__main__":
    main()
