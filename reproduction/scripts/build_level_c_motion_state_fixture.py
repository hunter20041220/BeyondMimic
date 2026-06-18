#!/usr/bin/env python3
"""Build a non-Kit Level C motion-state fixture from retargeted LAFAN1 CSV.

The official `csv_to_npz.py` uses Isaac/Kit articulation state to produce
body poses and velocities. That path is currently blocked by the host inotify
limit, so this script creates a clearly marked debug-only fixture by parsing
the G1 URDF and running deterministic forward kinematics outside Kit.

It is not a replacement for official `motion.npz`, DAgger rollouts, VAE
latents, or the paper's final state-latent trajectory dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DEFAULT_CSV = ROOT / "download/official/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv"
DEFAULT_URDF = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking"
    / "assets/unitree_description/urdf/g1/main.urdf"
)
OUT_DATA = ROOT / "reproduction/data/level_c_fixtures"
OUT_RES = ROOT / "res/level_c/motion_state_fixture"

OFFICIAL_CSV_JOINT_NAMES = [
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

G1_TRACKING_BODY_NAMES = [
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


@dataclass(frozen=True)
class JointSpec:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rotx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def roty(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rotz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    return rotz(float(rpy[2])) @ roty(float(rpy[1])) @ rotx(float(rpy[0]))


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    c1 = 1.0 - c
    return np.array(
        [
            [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
            [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
            [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1],
        ],
        dtype=np.float64,
    )


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_xyzw(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    if q[3] < 0.0:
        q *= -1.0
    return q / np.linalg.norm(q)


def quat_slerp_xyzw(a: np.ndarray, b: np.ndarray, blend: float) -> np.ndarray:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:
        out = a + blend * (b - a)
        return out / np.linalg.norm(out)
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta_0 * blend
    s0 = math.cos(theta) - dot * math.sin(theta) / math.sin(theta_0)
    s1 = math.sin(theta) / math.sin(theta_0)
    return s0 * a + s1 * b


def yaw_from_matrix(rot: np.ndarray) -> float:
    return math.atan2(float(rot[1, 0]), float(rot[0, 0]))


def rot6d(rot: np.ndarray) -> np.ndarray:
    return rot[:, :2].reshape(6, order="F")


def transform_matrix(rot: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = xyz
    return out


def parse_vec(text: str | None, default: str) -> np.ndarray:
    raw = text if text is not None else default
    return np.fromstring(raw, sep=" ", dtype=np.float64)


def parse_urdf(urdf: Path) -> dict[str, list[JointSpec]]:
    tree = ET.parse(urdf)
    root = tree.getroot()
    children_by_parent: dict[str, list[JointSpec]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        joint_type = joint.attrib.get("type", "fixed")
        parent_el = joint.find("parent")
        child_el = joint.find("child")
        if parent_el is None or child_el is None:
            continue
        origin_el = joint.find("origin")
        axis_el = joint.find("axis")
        spec = JointSpec(
            name=name,
            joint_type=joint_type,
            parent=parent_el.attrib["link"],
            child=child_el.attrib["link"],
            origin_xyz=parse_vec(origin_el.attrib.get("xyz") if origin_el is not None else None, "0 0 0"),
            origin_rpy=parse_vec(origin_el.attrib.get("rpy") if origin_el is not None else None, "0 0 0"),
            axis=parse_vec(axis_el.attrib.get("xyz") if axis_el is not None else None, "0 0 0"),
        )
        children_by_parent.setdefault(spec.parent, []).append(spec)
    return children_by_parent


def load_and_interpolate_motion(
    csv_path: Path,
    input_fps: int,
    output_fps: int,
    start_frame: int,
    end_frame: int,
) -> dict[str, np.ndarray]:
    data = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    if start_frame < 1 or end_frame < start_frame or end_frame > data.shape[0]:
        raise ValueError(f"invalid one-based frame range [{start_frame}, {end_frame}] for {data.shape[0]} frames")
    data = data[start_frame - 1 : end_frame]
    input_dt = 1.0 / input_fps
    output_dt = 1.0 / output_fps
    duration = (data.shape[0] - 1) * input_dt
    times = np.arange(0.0, duration, output_dt, dtype=np.float64)
    phase = times / duration
    idx0 = np.floor(phase * (data.shape[0] - 1)).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, data.shape[0] - 1)
    blend = phase * (data.shape[0] - 1) - idx0

    base_pos_in = data[:, :3]
    base_quat_xyzw_in = data[:, 3:7]
    dof_pos_in = data[:, 7:]
    base_pos = base_pos_in[idx0] * (1.0 - blend[:, None]) + base_pos_in[idx1] * blend[:, None]
    dof_pos = dof_pos_in[idx0] * (1.0 - blend[:, None]) + dof_pos_in[idx1] * blend[:, None]
    base_quat = np.stack(
        [quat_slerp_xyzw(base_quat_xyzw_in[i0], base_quat_xyzw_in[i1], float(b)) for i0, i1, b in zip(idx0, idx1, blend)],
        axis=0,
    )
    return {
        "times": times,
        "base_pos": base_pos,
        "base_quat_xyzw": base_quat,
        "joint_pos": dof_pos,
        "duration": np.array([duration], dtype=np.float64),
    }


def compute_fk(
    children_by_parent: dict[str, list[JointSpec]],
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    joint_values: dict[str, float],
) -> dict[str, np.ndarray]:
    root_tf = transform_matrix(quat_xyzw_to_matrix(root_quat_xyzw), root_pos)
    transforms = {"pelvis": root_tf}
    stack = ["pelvis"]
    while stack:
        parent = stack.pop()
        parent_tf = transforms[parent]
        for spec in children_by_parent.get(parent, []):
            origin_tf = transform_matrix(rpy_to_matrix(spec.origin_rpy), spec.origin_xyz)
            motion_tf = np.eye(4, dtype=np.float64)
            if spec.joint_type in {"revolute", "continuous"}:
                motion_tf[:3, :3] = axis_angle_to_matrix(spec.axis, float(joint_values.get(spec.name, 0.0)))
            elif spec.joint_type == "prismatic":
                motion_tf[:3, 3] = spec.axis * float(joint_values.get(spec.name, 0.0))
            transforms[spec.child] = parent_tf @ origin_tf @ motion_tf
            stack.append(spec.child)
    return transforms


def angular_velocity_from_quats(quats_xyzw: np.ndarray, dt: float) -> np.ndarray:
    rots = np.stack([quat_xyzw_to_matrix(q) for q in quats_xyzw], axis=0)
    out = np.zeros((len(quats_xyzw), 3), dtype=np.float64)
    if len(quats_xyzw) < 3:
        return out
    for i in range(1, len(quats_xyzw) - 1):
        rdot = (rots[i + 1] - rots[i - 1]) / (2.0 * dt)
        skew = rdot @ rots[i].T
        out[i] = np.array([skew[2, 1], skew[0, 2], skew[1, 0]], dtype=np.float64)
    out[0] = out[1]
    out[-1] = out[-2]
    return out


def build_hybrid_state(
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    root_lin_vel: np.ndarray,
    root_ang_vel: np.ndarray,
    body_pos_w: np.ndarray,
    body_quat_w: np.ndarray,
    body_lin_vel_w: np.ndarray,
) -> tuple[np.ndarray, dict[str, list[int]]]:
    frames, bodies = body_pos_w.shape[:2]
    parts = []
    slices: dict[str, list[int]] = {}

    def add(name: str, value: np.ndarray) -> None:
        start = sum(part.shape[1] for part in parts)
        parts.append(value.reshape(frames, -1))
        slices[name] = [start, start + parts[-1].shape[1]]

    root_rot = np.stack([quat_xyzw_to_matrix(q) for q in root_quat_xyzw], axis=0)
    body_rot = np.empty((frames, bodies, 3, 3), dtype=np.float64)
    for t in range(frames):
        for b in range(bodies):
            body_rot[t, b] = quat_xyzw_to_matrix(body_quat_w[t, b])

    root_height = root_pos[:, 2:3]
    root_rot_yaw_removed = np.empty((frames, 6), dtype=np.float64)
    root_lin_vel_yaw = np.empty((frames, 3), dtype=np.float64)
    root_ang_vel_yaw = np.empty((frames, 3), dtype=np.float64)
    body_pos_yaw = np.empty((frames, bodies, 3), dtype=np.float64)
    body_lin_vel_yaw = np.empty((frames, bodies, 3), dtype=np.float64)
    body_rot_yaw_removed = np.empty((frames, bodies, 6), dtype=np.float64)

    for t in range(frames):
        yaw = yaw_from_matrix(root_rot[t])
        yaw_inv = rotz(-yaw)
        root_rot_yaw_removed[t] = rot6d(yaw_inv @ root_rot[t])
        root_lin_vel_yaw[t] = yaw_inv @ root_lin_vel[t]
        root_ang_vel_yaw[t] = yaw_inv @ root_ang_vel[t]
        body_pos_yaw[t] = (yaw_inv @ (body_pos_w[t] - root_pos[t]).T).T
        body_lin_vel_yaw[t] = (yaw_inv @ body_lin_vel_w[t].T).T
        for b in range(bodies):
            body_rot_yaw_removed[t, b] = rot6d(yaw_inv @ body_rot[t, b])

    add("root_height", root_height)
    add("root_rot6d_without_yaw", root_rot_yaw_removed)
    add("root_lin_vel_yaw_frame", root_lin_vel_yaw)
    add("root_ang_vel_yaw_frame", root_ang_vel_yaw)
    add("body_pos_yaw_frame", body_pos_yaw)
    add("body_lin_vel_yaw_frame", body_lin_vel_yaw)
    add("body_rot6d_without_yaw", body_rot_yaw_removed)
    return np.concatenate(parts, axis=1), slices


def build_windows(state: np.ndarray, history: int, horizon: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    length = history + 1 + horizon
    starts = np.arange(history, state.shape[0] - horizon, stride, dtype=np.int64)
    windows = np.stack([state[s - history : s + horizon + 1] for s in starts], axis=0)
    return windows, starts


def write_tsv(path: Path, rows: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["key", "value"])
        for key in sorted(rows):
            value = rows[key]
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, sort_keys=True)
            writer.writerow([key, value])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--input-fps", type=int, default=30)
    parser.add_argument("--output-fps", type=int, default=50)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--end-frame", type=int, default=180)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--window-stride", type=int, default=10)
    args = parser.parse_args()

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)

    motion = load_and_interpolate_motion(
        args.input_csv,
        args.input_fps,
        args.output_fps,
        args.start_frame,
        args.end_frame,
    )
    children_by_parent = parse_urdf(args.urdf)
    dt = 1.0 / args.output_fps
    joint_pos = motion["joint_pos"]
    joint_vel = np.gradient(joint_pos, dt, axis=0)
    root_pos = motion["base_pos"]
    root_quat = motion["base_quat_xyzw"]
    root_lin_vel = np.gradient(root_pos, dt, axis=0)
    root_ang_vel = angular_velocity_from_quats(root_quat, dt)

    frames = root_pos.shape[0]
    body_count = len(G1_TRACKING_BODY_NAMES)
    body_pos_w = np.zeros((frames, body_count, 3), dtype=np.float64)
    body_quat_w = np.zeros((frames, body_count, 4), dtype=np.float64)
    missing_bodies: set[str] = set()

    for t in range(frames):
        joint_values = {name: float(joint_pos[t, idx]) for idx, name in enumerate(OFFICIAL_CSV_JOINT_NAMES)}
        transforms = compute_fk(children_by_parent, root_pos[t], root_quat[t], joint_values)
        for b, body_name in enumerate(G1_TRACKING_BODY_NAMES):
            tf = transforms.get(body_name)
            if tf is None:
                missing_bodies.add(body_name)
                continue
            body_pos_w[t, b] = tf[:3, 3]
            body_quat_w[t, b] = matrix_to_quat_xyzw(tf[:3, :3])

    if missing_bodies:
        raise RuntimeError(f"missing target bodies from FK: {sorted(missing_bodies)}")

    body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0)
    body_ang_vel_w = np.stack([angular_velocity_from_quats(body_quat_w[:, b], dt) for b in range(body_count)], axis=1)
    hybrid_state, feature_slices = build_hybrid_state(
        root_pos,
        root_quat,
        root_lin_vel,
        root_ang_vel,
        body_pos_w,
        body_quat_w,
        body_lin_vel_w,
    )
    windows, window_start_indices = build_windows(hybrid_state, args.history, args.horizon, args.window_stride)

    translated_root = root_pos + np.array([3.0, -2.0, 0.0], dtype=np.float64)
    yaw_delta = 1.234
    yaw_rot = rotz(yaw_delta)
    transformed_root = (yaw_rot @ translated_root.T).T
    transformed_root_quat = np.stack([matrix_to_quat_xyzw(yaw_rot @ quat_xyzw_to_matrix(q)) for q in root_quat], axis=0)
    transformed_root_lin_vel = (yaw_rot @ root_lin_vel.T).T
    transformed_root_ang_vel = (yaw_rot @ root_ang_vel.T).T
    transformed_body_pos = (yaw_rot @ (body_pos_w + np.array([3.0, -2.0, 0.0])).reshape(-1, 3).T).T.reshape(body_pos_w.shape)
    transformed_body_quat = np.empty_like(body_quat_w)
    for t in range(frames):
        for b in range(body_count):
            transformed_body_quat[t, b] = matrix_to_quat_xyzw(yaw_rot @ quat_xyzw_to_matrix(body_quat_w[t, b]))
    transformed_body_lin_vel = (yaw_rot @ body_lin_vel_w.reshape(-1, 3).T).T.reshape(body_lin_vel_w.shape)
    transformed_hybrid, _ = build_hybrid_state(
        transformed_root,
        transformed_root_quat,
        transformed_root_lin_vel,
        transformed_root_ang_vel,
        transformed_body_pos,
        transformed_body_quat,
        transformed_body_lin_vel,
    )
    invariance_max_abs_error = float(np.max(np.abs(hybrid_state - transformed_hybrid)))

    emphasis = np.ones(hybrid_state.shape[1], dtype=np.float64)
    for name in [
        "root_height",
        "root_rot6d_without_yaw",
        "root_lin_vel_yaw_frame",
        "root_ang_vel_yaw_frame",
    ]:
        lo, hi = feature_slices[name]
        emphasis[lo:hi] = 6.0
    projected = hybrid_state * emphasis
    reconstructed = projected / emphasis
    emphasis_reconstruction_max_abs_error = float(np.max(np.abs(reconstructed - hybrid_state)))

    base_name = f"{args.input_csv.stem}_frames_{args.start_frame}_{args.end_frame}_state_fixture"
    npz_path = OUT_DATA / f"{base_name}.npz"
    manifest_path = OUT_RES / f"{base_name}.json"
    tsv_path = OUT_RES / f"{base_name}.tsv"
    np.savez_compressed(
        npz_path,
        fps=np.array([args.output_fps], dtype=np.float64),
        times=motion["times"],
        root_pos_w=root_pos,
        root_quat_xyzw_w=root_quat,
        root_lin_vel_w=root_lin_vel,
        root_ang_vel_w=root_ang_vel,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_xyzw_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
        candidate_hybrid_state=hybrid_state,
        candidate_hybrid_state_windows=windows,
        window_start_indices=window_start_indices,
        emphasis_projection_weights=emphasis,
    )

    manifest: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "motion-derived non-Kit FK fixture for Level C schema/provenance/transform tests",
        "not_a_replacement_for": [
            "official csv_to_npz Isaac/Kit motion.npz",
            "teacher rollout",
            "DAgger dataset",
            "VAE latent dataset",
            "paper-exact state-latent trajectory dataset",
        ],
        "input_csv": str(args.input_csv),
        "input_csv_sha256": sha256_file(args.input_csv),
        "urdf": str(args.urdf),
        "urdf_sha256": sha256_file(args.urdf),
        "official_csv_joint_names": OFFICIAL_CSV_JOINT_NAMES,
        "tracking_body_names": G1_TRACKING_BODY_NAMES,
        "input_fps": args.input_fps,
        "output_fps": args.output_fps,
        "start_frame_one_based": args.start_frame,
        "end_frame_one_based_inclusive": args.end_frame,
        "output_frames": int(frames),
        "history": args.history,
        "horizon": args.horizon,
        "window_stride": args.window_stride,
        "window_count": int(windows.shape[0]),
        "shapes": {
            "joint_pos": list(joint_pos.shape),
            "body_pos_w": list(body_pos_w.shape),
            "candidate_hybrid_state": list(hybrid_state.shape),
            "candidate_hybrid_state_windows": list(windows.shape),
        },
        "feature_slices": feature_slices,
        "checks": {
            "finite_all_arrays": bool(
                np.isfinite(joint_pos).all()
                and np.isfinite(body_pos_w).all()
                and np.isfinite(hybrid_state).all()
                and np.isfinite(windows).all()
            ),
            "global_xy_yaw_invariance_max_abs_error": invariance_max_abs_error,
            "emphasis_coefficient_for_root_features": 6.0,
            "emphasis_projection_pseudoinverse_max_abs_error": emphasis_reconstruction_max_abs_error,
        },
        "outputs": {
            "npz": str(npz_path),
            "manifest_json": str(manifest_path),
            "manifest_tsv": str(tsv_path),
        },
        "source_evidence": {
            "official_csv_to_npz": str(
                ROOT / "reproduction/third_party/official/whole_body_tracking/scripts/csv_to_npz.py"
            ),
            "g1_flat_env_cfg": str(
                ROOT
                / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py"
            ),
            "goal": str(ROOT / "goal.md"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, manifest)
    print(json.dumps({"status": "ok", "npz": str(npz_path), "manifest": str(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
