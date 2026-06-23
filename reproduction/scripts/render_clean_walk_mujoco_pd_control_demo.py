#!/usr/bin/env python3
"""Render a clean long MuJoCo walk-control demo from a continuous LAFAN1 walk.

This script is intentionally narrower than the Stage-1 teacher/VAE/diffusion
video suite.  The latest automatic six-video selection proved useful for
diagnosis, but it selected short and/or weak controller segments.  Here we
first produce one report-ready walking-control baseline with a clean source:

    continuous LAFAN1 walk joint targets -> MuJoCo 29-DoF position actuators
    -> mj_step physics -> centered G1 mesh MP4.

Claim boundary: this is a local MuJoCo PD/root-assist reference-action walk
demo.  It is not a PPO teacher rollout, not VAE/diffusion/guidance control,
not IsaacLab rendered output, not a real robot result, and not a paper-level
BeyondMimic Fig. 5/Fig. 6 result.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

ROOT = Path(os.environ.get("BM_ROOT", "/mnt/infini-data/test/BeyondMimic")).resolve()
MUJOCO_SCRIPT_DIR = ROOT / "mujoco_mp4/scripts"
if str(MUJOCO_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(MUJOCO_SCRIPT_DIR))

from mujoco_common import render_frame, sha256, traceback_payload, utc_now, write_json  # noqa: E402
from mujoco_pd_control_video import (  # noqa: E402
    PD_CAMERA,
    actuator_joint_order,
    apply_root_assist,
    load_action_rows,
    normalize_quat_wxyz,
    patch_joints_and_actuators,
    quat_error_rotvec,
    quat_to_roll_pitch_yaw,
)
from mujoco_trace_mesh_video import DEFAULT_MODEL  # noqa: E402


DEFAULT_MOTION = (
    ROOT / "res/tracking/stage1_multisource_motion_bundle/motions/lafan1_walk1_subject1/motion.npz"
)
OUT_DIR = ROOT / "res/visualization/clean_walk_mujoco_pd_control_demo"
OUT_MP4 = OUT_DIR / "clean_lafan1_walk1_subject1_pd_control_15s.mp4"
OUT_KEYFRAME = OUT_DIR / "clean_lafan1_walk1_subject1_pd_control_keyframe.png"
OUT_KEYFRAMES = OUT_DIR / "clean_lafan1_walk1_subject1_pd_control_keyframes.png"
OUT_METRICS = OUT_DIR / "clean_lafan1_walk1_subject1_pd_control_metrics.csv"
OUT_SUMMARY = OUT_DIR / "clean_lafan1_walk1_subject1_pd_control_summary.json"
OUT_README = OUT_DIR / "README.md"
OUT_FAILURE_AUDIT = OUT_DIR / "why_previous_stage1_six_videos_failed.json"


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    score: float
    root_z_min: float
    root_z_mean: float
    root_z_max: float
    root_z_range: float
    root_xy_displacement_m: float


def scalar_fps(value: np.ndarray) -> float:
    arr = np.asarray(value).reshape(-1)
    return float(arr[0]) if arr.size else 50.0


def choose_stable_walk_segment(
    root_pos: np.ndarray,
    motion_fps: float,
    duration_s: float,
    stride_s: float,
    min_root_z: float,
    max_root_z_range: float,
) -> Segment:
    total = int(root_pos.shape[0])
    desired = min(total, max(1, int(round(duration_s * motion_fps))))
    stride = max(1, int(round(stride_s * motion_fps)))
    best: Segment | None = None
    fallback: Segment | None = None
    for start in range(0, total - desired + 1, stride):
        end = start + desired
        window = root_pos[start:end]
        z = window[:, 2]
        xy = window[:, 0:2]
        root_z_min = float(np.min(z))
        root_z_mean = float(np.mean(z))
        root_z_max = float(np.max(z))
        root_z_range = root_z_max - root_z_min
        xy_disp = float(np.linalg.norm(xy[-1] - xy[0]))
        score = root_z_min + root_z_mean - 0.75 * root_z_range + 0.04 * min(xy_disp, 2.0)
        candidate = Segment(
            start=start,
            end=end,
            score=score,
            root_z_min=root_z_min,
            root_z_mean=root_z_mean,
            root_z_max=root_z_max,
            root_z_range=root_z_range,
            root_xy_displacement_m=xy_disp,
        )
        if fallback is None or candidate.score > fallback.score:
            fallback = candidate
        if root_z_min < min_root_z or root_z_range > max_root_z_range:
            continue
        if best is None or candidate.score > best.score:
            best = candidate
    if best is not None:
        return best
    if fallback is None:
        raise RuntimeError("No non-empty motion segment candidates were found")
    return fallback


def segment_from_start(root_pos: np.ndarray, start: int, motion_fps: float, duration_s: float) -> Segment:
    total = int(root_pos.shape[0])
    desired = min(total - start, max(1, int(round(duration_s * motion_fps))))
    if start < 0 or start >= total or desired <= 0:
        raise ValueError(f"Invalid clean-walk start index {start} for {total} motion frames")
    end = start + desired
    window = root_pos[start:end]
    z = window[:, 2]
    xy = window[:, 0:2]
    root_z_min = float(np.min(z))
    root_z_mean = float(np.mean(z))
    root_z_max = float(np.max(z))
    root_z_range = root_z_max - root_z_min
    xy_disp = float(np.linalg.norm(xy[-1] - xy[0]))
    return Segment(
        start=start,
        end=end,
        score=root_z_min + root_z_mean - 0.75 * root_z_range + 0.04 * min(xy_disp, 2.0),
        root_z_min=root_z_min,
        root_z_mean=root_z_mean,
        root_z_max=root_z_max,
        root_z_range=root_z_range,
        root_xy_displacement_m=xy_disp,
    )


def downsample_motion_indices(segment: Segment, motion_fps: float, video_fps: int) -> np.ndarray:
    duration_s = (segment.end - segment.start) / motion_fps
    frames = max(1, int(round(duration_s * video_fps)))
    t = np.arange(frames, dtype=np.float64) / float(video_fps)
    offsets = np.minimum(np.rint(t * motion_fps).astype(np.int64), segment.end - segment.start - 1)
    return segment.start + offsets


def make_keyframe_strip(frames: list[np.ndarray], path: Path) -> None:
    if not frames:
        return
    imageio.imwrite(path, np.concatenate(frames, axis=1))


def write_failure_audit() -> dict[str, Any]:
    previous_root = ROOT / "res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos"
    old_summaries: dict[str, Any] = {}
    for name in [
        "reference_joint_pd_control",
        "teacher_policy_action_control",
        "vae_reconstructed_action_control",
        "diffusion_denoised_latent_action_control",
        "guided_latent_action_control",
        "native_ppo_obs_adapter_probe",
    ]:
        path = previous_root / name / f"{name}_summary.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        old_summaries[name] = {
            "summary_path": str(path),
            "claim_level": payload.get("claim_level"),
            "frames": payload.get("frames")
            or payload.get("rendered_frames")
            or payload.get("frames_rendered"),
            "duration_seconds": payload.get("duration_seconds"),
            "metrics": payload.get("metrics"),
        }
    audit = {
        "status": "ok",
        "timestamp_utc": utc_now(),
        "experiment_type": "stage1_mujoco_video_failure_and_clean_walk_redirect",
        "diagnosis": [
            "The previous Stage-1 six-video suite is not suitable as a normal walking demonstration.",
            "The original continuous selector selected a near-floor root target in lafan1_walk3_subject4, so the target itself was invalid.",
            "The later quality-gated suite used a normal-height segment but only had 30 frames / 1 second of stable evidence.",
            "On the normal-height segment, reference joint PD control was stable, while teacher/VAE/diffusion action-derived targets still drove the robot downward or into crouched poses.",
            "Therefore the immediate report asset should be a clean continuous walk reference-action PD control demo; teacher/VAE/diffusion videos remain diagnostic until the controller/adapter is fixed.",
        ],
        "previous_summaries": old_summaries,
        "claim_boundary": (
            "This audit explains video quality; it does not prove paper-level tracking, diffusion guidance, "
            "IsaacLab rendered rollout, or real-robot deployment."
        ),
    }
    write_json(OUT_FAILURE_AUDIT, audit)
    return audit


def render_walk_demo() -> dict[str, Any]:
    import mujoco

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("BM_MUJOCO_PD_CAMERA_POS", "-0.15 -3.95 1.50")
    os.environ.setdefault("BM_MUJOCO_PD_CAMERA_XYAXES", "1 0 0 0 0.30 0.954")
    os.environ.setdefault("BM_MUJOCO_PD_CAMERA_FOVY", "38")

    backend = os.environ.get("MUJOCO_GL", "egl")
    motion_path = Path(os.environ.get("BM_CLEAN_WALK_MOTION_NPZ", str(DEFAULT_MOTION))).expanduser()
    duration_s = float(os.environ.get("BM_CLEAN_WALK_SECONDS", "15.0"))
    video_fps = int(os.environ.get("BM_CLEAN_WALK_VIDEO_FPS", "30"))
    width = int(os.environ.get("BM_CLEAN_WALK_WIDTH", "960"))
    height = int(os.environ.get("BM_CLEAN_WALK_HEIGHT", "540"))
    substeps = int(os.environ.get("BM_CLEAN_WALK_SUBSTEPS", "4"))
    settle_steps = int(os.environ.get("BM_CLEAN_WALK_SETTLE_STEPS", "50"))
    min_root_z = float(os.environ.get("BM_CLEAN_WALK_MIN_ROOT_Z", "0.60"))
    max_root_z_range = float(os.environ.get("BM_CLEAN_WALK_MAX_ROOT_Z_RANGE", "0.20"))
    stride_s = float(os.environ.get("BM_CLEAN_WALK_WINDOW_STRIDE_S", "2.0"))
    center_xy = os.environ.get("BM_CLEAN_WALK_CENTER_ROOT_XY", "1") == "1"
    explicit_start = os.environ.get("BM_CLEAN_WALK_START_INDEX")

    motion = np.load(motion_path, allow_pickle=True)
    motion_fps = scalar_fps(motion["fps"]) if "fps" in motion else 50.0
    joint_all = np.asarray(motion["joint_pos"], dtype=np.float64)
    root_pos_all = np.asarray(motion["body_pos_w"][:, 0, :], dtype=np.float64)
    root_quat_all = np.asarray(motion["body_quat_w"][:, 0, :], dtype=np.float64)
    if explicit_start is not None:
        segment = segment_from_start(root_pos_all, int(explicit_start), motion_fps, duration_s)
        selector_mode = "explicit_start_index"
    else:
        segment = choose_stable_walk_segment(
            root_pos_all,
            motion_fps=motion_fps,
            duration_s=duration_s,
            stride_s=stride_s,
            min_root_z=min_root_z,
            max_root_z_range=max_root_z_range,
        )
        selector_mode = "stable_root_height_window"
    indices = downsample_motion_indices(segment, motion_fps, video_fps)
    joint_targets = joint_all[indices].copy()
    root_pos_targets = root_pos_all[indices].copy()
    if center_xy:
        root_pos_targets[:, 0:2] = 0.0
    else:
        root_pos_targets[:, 0:2] -= root_pos_targets[0:1, 0:2]
    root_quat_targets = np.stack([normalize_quat_wxyz(q) for q in root_quat_all[indices]], axis=0)

    model_path = Path(os.environ.get("BM_MUJOCO_G1_XML", str(DEFAULT_MODEL))).expanduser()
    action_rows = load_action_rows()
    # Keep the patched XML beside the source MJCF so relative ``meshes/*.STL``
    # paths still resolve correctly when MuJoCo loads the model.
    patched_xml = model_path.parent / "g1_clean_walk_pd_control.xml"
    patch_joints_and_actuators(model_path, patched_xml, action_rows)

    model = mujoco.MjModel.from_xml_path(str(patched_xml))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    if actuator_joint_order(model) != [row["joint_name"] for row in action_rows]:
        raise RuntimeError("Actuator joint order does not match action-scale audit")
    pelvis_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_body < 0:
        raise RuntimeError("MuJoCo body 'pelvis' not found")

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

    keyframe_indices = set(
        int(round(v))
        for v in np.linspace(0, max(0, len(indices) - 1), 5)
    )
    strip_frames: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    with imageio.get_writer(OUT_MP4, fps=video_fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame_idx, motion_idx in enumerate(indices):
            target = np.clip(joint_targets[frame_idx], model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            data.ctrl[:] = target
            for _ in range(substeps):
                data.xfrc_applied[:] = 0.0
                apply_root_assist(model, data, pelvis_body, root_pos_targets[frame_idx], root_quat_targets[frame_idx])
                mujoco.mj_step(model, data)
            frame = render_frame(model, data, renderer, camera=PD_CAMERA)
            if frame_idx == 0:
                imageio.imwrite(OUT_KEYFRAME, frame)
            if frame_idx in keyframe_indices:
                strip_frames.append(frame)
            writer.append_data(frame)
            q = data.qpos[7 : 7 + 29].copy()
            qd = data.qvel[6 : 6 + 29].copy()
            err = q - target
            roll, pitch, yaw = quat_to_roll_pitch_yaw(data.qpos[3:7])
            rows.append(
                {
                    "frame": frame_idx,
                    "motion_index": int(motion_idx),
                    "video_time_s": frame_idx / float(video_fps),
                    "source_motion_time_s": (int(motion_idx) - segment.start) / motion_fps,
                    "sim_time_s": float(data.time),
                    "root_x": float(data.qpos[0]),
                    "root_y": float(data.qpos[1]),
                    "root_z": float(data.qpos[2]),
                    "root_roll": roll,
                    "root_pitch": pitch,
                    "root_yaw": yaw,
                    "root_target_x": float(root_pos_targets[frame_idx, 0]),
                    "root_target_y": float(root_pos_targets[frame_idx, 1]),
                    "root_target_z": float(root_pos_targets[frame_idx, 2]),
                    "joint_target_abs_mean": float(np.mean(np.abs(target))),
                    "joint_error_abs_mean": float(np.mean(np.abs(err))),
                    "joint_error_abs_max": float(np.max(np.abs(err))),
                    "joint_velocity_abs_mean": float(np.mean(np.abs(qd))),
                    "ctrl_abs_mean": float(np.mean(np.abs(data.ctrl))),
                    "root_position_error_m": float(np.linalg.norm(data.xpos[pelvis_body] - root_pos_targets[frame_idx])),
                    "root_orientation_error_rad": float(
                        np.linalg.norm(quat_error_rotvec(root_quat_targets[frame_idx], data.xquat[pelvis_body]))
                    ),
                    "contact_count": int(data.ncon),
                    "fall_proxy": bool(data.qpos[2] < 0.45 or abs(roll) > 1.2 or abs(pitch) > 1.2),
                }
            )
    renderer.close()
    make_keyframe_strip(strip_frames, OUT_KEYFRAMES)

    with OUT_METRICS.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    failure_audit = write_failure_audit()
    fall_count = sum(1 for row in rows if row["fall_proxy"])
    summary = {
        "status": "ok",
        "timestamp_utc": utc_now(),
        "experiment_type": "clean_lafan1_walk_mujoco_pd_control_demo",
        "backend": backend,
        "claim_level": (
            "Local MuJoCo centered PD/root-assist walk-control demonstration from a continuous "
            "LAFAN1 reference motion. Not PPO teacher, not VAE/diffusion/guidance, not IsaacLab rendered, "
            "not real robot, and not paper-level BeyondMimic."
        ),
        "source_motion": {
            "path": str(motion_path),
            "sha256": sha256(motion_path),
            "fps": motion_fps,
            "total_frames": int(joint_all.shape[0]),
            "selected_start_index": segment.start,
            "selected_end_index_exclusive": segment.end,
            "selected_motion_duration_s": (segment.end - segment.start) / motion_fps,
            "selected_root_z_min": segment.root_z_min,
            "selected_root_z_mean": segment.root_z_mean,
            "selected_root_z_max": segment.root_z_max,
            "selected_root_z_range": segment.root_z_range,
            "selected_root_xy_displacement_m": segment.root_xy_displacement_m,
            "selector": {
                "mode": selector_mode,
                "explicit_start_index": int(explicit_start) if explicit_start is not None else None,
                "min_root_z": min_root_z,
                "max_root_z_range": max_root_z_range,
                "window_stride_s": stride_s,
                "score": segment.score,
            },
        },
        "video": {
            "frames_rendered": int(len(indices)),
            "video_fps": video_fps,
            "duration_seconds": len(indices) / float(video_fps),
            "width": width,
            "height": height,
            "motion_downsampled_from_50hz_to_video_fps": True,
            "temporal_stretching": False,
            "root_xy_centered_for_camera": center_xy,
        },
        "simulation": {
            "uses_mj_step": True,
            "writes_qpos_each_frame": False,
            "actuator_type": "position",
            "actuator_count": int(model.nu),
            "control_substeps_per_frame": substeps,
            "settle_steps": settle_steps,
            "timestep": float(model.opt.timestep),
            "root_assist_enabled": os.environ.get("BM_MUJOCO_ROOT_ASSIST", "1") == "1",
            "root_assist_type": "external pelvis force/torque stabilizer before mj_step",
            "source_model_xml": str(model_path),
            "patched_pd_model_xml": str(patched_xml),
        },
        "camera": {
            "name": PD_CAMERA,
            "mode": "fixed",
            "position": os.environ.get("BM_MUJOCO_PD_CAMERA_POS", "-0.15 -3.95 1.50"),
            "xyaxes": os.environ.get("BM_MUJOCO_PD_CAMERA_XYAXES", "1 0 0 0 0.30 0.954"),
            "fovy": os.environ.get("BM_MUJOCO_PD_CAMERA_FOVY", "38"),
            "robot_centered_by_root_xy_target": center_xy,
        },
        "outputs": {
            "mp4": str(OUT_MP4),
            "keyframe_png": str(OUT_KEYFRAME),
            "keyframes_png": str(OUT_KEYFRAMES),
            "metrics_csv": str(OUT_METRICS),
            "summary_json": str(OUT_SUMMARY),
            "readme": str(OUT_README),
            "failure_audit_json": str(OUT_FAILURE_AUDIT),
        },
        "file_sizes": {
            "mp4": OUT_MP4.stat().st_size if OUT_MP4.exists() else 0,
            "keyframe_png": OUT_KEYFRAME.stat().st_size if OUT_KEYFRAME.exists() else 0,
            "keyframes_png": OUT_KEYFRAMES.stat().st_size if OUT_KEYFRAMES.exists() else 0,
            "metrics_csv": OUT_METRICS.stat().st_size if OUT_METRICS.exists() else 0,
        },
        "metrics": {
            "fall_proxy_count": int(fall_count),
            "root_height_min": float(np.min([row["root_z"] for row in rows])),
            "root_height_mean": float(np.mean([row["root_z"] for row in rows])),
            "root_height_max": float(np.max([row["root_z"] for row in rows])),
            "root_xy_abs_max": float(np.max([max(abs(row["root_x"]), abs(row["root_y"])) for row in rows])),
            "joint_error_abs_mean": float(np.mean([row["joint_error_abs_mean"] for row in rows])),
            "joint_error_abs_max": float(np.max([row["joint_error_abs_max"] for row in rows])),
            "root_position_error_mean_m": float(np.mean([row["root_position_error_m"] for row in rows])),
            "root_position_error_max_m": float(np.max([row["root_position_error_m"] for row in rows])),
            "root_orientation_error_mean_rad": float(np.mean([row["root_orientation_error_rad"] for row in rows])),
            "contact_count_mean": float(np.mean([row["contact_count"] for row in rows])),
        },
        "checks": {
            "mp4_exists": OUT_MP4.is_file() and OUT_MP4.stat().st_size > 0,
            "keyframe_exists": OUT_KEYFRAME.is_file() and OUT_KEYFRAME.stat().st_size > 0,
            "keyframes_exists": OUT_KEYFRAMES.is_file() and OUT_KEYFRAMES.stat().st_size > 0,
            "metrics_csv_exists": OUT_METRICS.is_file() and OUT_METRICS.stat().st_size > 0,
            "uses_mujoco_g1_mesh": True,
            "uses_mj_step": True,
            "does_not_write_qpos_each_frame": True,
            "uses_29_position_actuators": int(model.nu) == 29,
            "fall_proxy_zero": fall_count == 0,
            "root_height_above_threshold": min(row["root_z"] for row in rows) > 0.55,
            "video_duration_at_least_10s": len(indices) / float(video_fps) >= 10.0,
            "temporal_stretching_disabled": True,
            "does_not_claim_teacher_policy": True,
            "does_not_claim_diffusion_guidance": True,
            "does_not_claim_real_robot": True,
        },
        "previous_video_failure_audit": failure_audit,
        "limitations": [
            "This uses reference joint targets through MuJoCo position actuators; it is not an autonomous PPO/VAE/diffusion policy.",
            "A root-assist stabilizer keeps the pelvis centered for a readable camera view, so this is not unassisted humanoid balance.",
            "The output is a clean walk demonstration asset, not a paper-level BeyondMimic metric result.",
        ],
    }
    write_json(OUT_SUMMARY, summary)
    OUT_README.write_text(
        "\n".join(
            [
                "# Clean MuJoCo Walk PD Control Demo",
                "",
                "This directory contains a clean local MuJoCo walking-control demonstration generated after the automatic Stage-1 six-video suite was judged unsuitable for presentation.",
                "",
                "## Main Output",
                "",
                f"- MP4: `{OUT_MP4}`",
                f"- Keyframe strip: `{OUT_KEYFRAMES}`",
                f"- Metrics CSV: `{OUT_METRICS}`",
                f"- Summary JSON: `{OUT_SUMMARY}`",
                f"- Failure audit: `{OUT_FAILURE_AUDIT}`",
                "",
                "## Claim Level",
                "",
                "Local MuJoCo centered PD/root-assist walk-control demonstration from a continuous LAFAN1 reference motion. It is not PPO teacher control, not VAE/diffusion/guidance, not IsaacLab rendered output, not real robot, and not paper-level BeyondMimic.",
                "",
                "## Why This Replaces The Previous Walk Display",
                "",
                "The previous Stage-1 six videos were diagnostic: the first selector picked a near-floor target segment, and the later quality-gated videos were only 30 frames / 1 second while teacher/VAE/diffusion action targets still drove the robot downward. This clean walk demo uses a long continuous walking source and does not apply temporal stretching.",
                "",
                "## Key Metrics",
                "",
                f"- Duration: {summary['video']['duration_seconds']:.2f} s",
                f"- Fall proxy count: {summary['metrics']['fall_proxy_count']}",
                f"- Root height min/mean/max: {summary['metrics']['root_height_min']:.3f} / {summary['metrics']['root_height_mean']:.3f} / {summary['metrics']['root_height_max']:.3f} m",
                f"- Mean joint error: {summary['metrics']['joint_error_abs_mean']:.4f} rad",
                f"- Mean root position error: {summary['metrics']['root_position_error_mean_m']:.4f} m",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "mp4": str(OUT_MP4), "summary": str(OUT_SUMMARY)}))
    return summary


def main() -> None:
    try:
        render_walk_demo()
    except Exception as exc:  # noqa: BLE001
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_json(
            OUT_DIR / "clean_lafan1_walk1_subject1_pd_control_failed_summary.json",
            {
                "status": "failed",
                "timestamp_utc": utc_now(),
                "experiment_type": "clean_lafan1_walk_mujoco_pd_control_demo",
                "error": traceback_payload(exc),
            },
        )
        raise


if __name__ == "__main__":
    main()
