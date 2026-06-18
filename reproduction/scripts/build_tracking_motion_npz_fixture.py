#!/usr/bin/env python3
"""Build debug motion.npz fixtures for the tracking MotionLoader contract.

The official producer runs inside Isaac/Kit and records articulation state.
This script stays outside Kit: it reuses the local URDF forward-kinematics
fixture code to create full-body, URDF-order arrays with the same keys and
shape ranks consumed by whole_body_tracking.tasks.tracking.mdp.MotionLoader.

These files are contract/debug fixtures, not official csv_to_npz outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SCRIPTS = ROOT / "reproduction/scripts"
sys.path.insert(0, str(SCRIPTS))

from build_level_c_motion_state_fixture import (  # noqa: E402
    DEFAULT_URDF,
    G1_TRACKING_BODY_NAMES,
    OFFICIAL_CSV_JOINT_NAMES,
    angular_velocity_from_quats,
    compute_fk,
    load_and_interpolate_motion,
    matrix_to_quat_xyzw,
    parse_urdf,
    sha256_file,
)
from validate_motion_npz_contract import REQUIRED_KEYS  # noqa: E402


OUT_DATA = ROOT / "reproduction/data/tracking_motion_npz_fixtures"
OUT_RES = ROOT / "res/tracking/motion_npz_fixture"
DEFAULT_CSVS = [
    ROOT / "download/official/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv",
    ROOT / "download/official/LAFAN1_Retargeting_Dataset/g1/run2_subject1.csv",
    ROOT / "download/official/LAFAN1_Retargeting_Dataset/g1/jumps1_subject1.csv",
]


def urdf_link_order(path: Path) -> list[str]:
    root = ET.parse(path).getroot()
    return [link.attrib["name"] for link in root.findall("link")]


def validate_npz(path: Path, expect_fps: float) -> dict[str, Any]:
    with np.load(path) as data:
        missing = sorted(set(REQUIRED_KEYS) - set(data.files))
        if missing:
            raise ValueError(f"{path}: missing keys {missing}")
        summary: dict[str, Any] = {}
        for key, ndim in REQUIRED_KEYS.items():
            arr = data[key]
            if arr.ndim != ndim:
                raise ValueError(f"{path}: {key} expected ndim {ndim}, got {arr.ndim}")
            if not np.isfinite(arr).all():
                raise ValueError(f"{path}: {key} contains non-finite values")
            summary[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if abs(fps - expect_fps) > 1e-6:
            raise ValueError(f"{path}: expected fps {expect_fps}, got {fps}")
        steps = int(data["joint_pos"].shape[0])
        for key in ["joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]:
            if int(data[key].shape[0]) != steps:
                raise ValueError(f"{path}: {key} timestep mismatch")
        if data["body_pos_w"].shape[-1] != 3:
            raise ValueError(f"{path}: body_pos_w final dim must be 3")
        if data["body_quat_w"].shape[-1] != 4:
            raise ValueError(f"{path}: body_quat_w final dim must be 4")
        quat_norm = np.linalg.norm(data["body_quat_w"], axis=-1)
        summary["time_steps"] = steps
        summary["fps_value"] = fps
        summary["body_quat_norm_max_abs_error_from_1"] = float(np.max(np.abs(quat_norm - 1.0)))
        return summary


def build_one(csv_path: Path, args: argparse.Namespace, link_names: list[str]) -> dict[str, Any]:
    children_by_parent = parse_urdf(args.urdf)
    motion = load_and_interpolate_motion(
        csv_path=csv_path,
        input_fps=args.input_fps,
        output_fps=args.output_fps,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )
    root_pos = motion["base_pos"]
    root_quat_xyzw = motion["base_quat_xyzw"]
    joint_pos = motion["joint_pos"]
    frames = int(joint_pos.shape[0])
    dt = 1.0 / float(args.output_fps)
    joint_vel = np.gradient(joint_pos, dt, axis=0)

    body_pos_w = np.zeros((frames, len(link_names), 3), dtype=np.float64)
    body_quat_xyzw_w = np.zeros((frames, len(link_names), 4), dtype=np.float64)
    missing: set[str] = set()
    for t in range(frames):
        joint_values = {name: float(joint_pos[t, idx]) for idx, name in enumerate(OFFICIAL_CSV_JOINT_NAMES)}
        transforms = compute_fk(children_by_parent, root_pos[t], root_quat_xyzw[t], joint_values)
        for body_idx, body_name in enumerate(link_names):
            tf = transforms.get(body_name)
            if tf is None:
                missing.add(body_name)
                continue
            body_pos_w[t, body_idx] = tf[:3, 3]
            body_quat_xyzw_w[t, body_idx] = matrix_to_quat_xyzw(tf[:3, :3])
    if missing:
        raise RuntimeError(f"{csv_path}: URDF FK missed links {sorted(missing)}")

    body_lin_vel_w = np.gradient(body_pos_w, dt, axis=0)
    body_ang_vel_w = np.stack(
        [angular_velocity_from_quats(body_quat_xyzw_w[:, body_idx], dt) for body_idx in range(len(link_names))],
        axis=1,
    )
    body_quat_wxyz_w = body_quat_xyzw_w[:, :, [3, 0, 1, 2]]

    base_name = f"{csv_path.stem}_frames_{args.start_frame}_{args.end_frame}_debug_motion"
    npz_path = OUT_DATA / f"{base_name}.npz"
    validator_json = OUT_RES / f"{base_name}_validator.json"
    np.savez_compressed(
        npz_path,
        fps=np.array([args.output_fps], dtype=np.float64),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_wxyz_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    validator = validate_npz(npz_path, float(args.output_fps))
    validator_json.write_text(json.dumps(validator, indent=2, sort_keys=True), encoding="utf-8")

    tracking_indices = [link_names.index(name) for name in G1_TRACKING_BODY_NAMES]
    selected = body_pos_w[:, tracking_indices]
    quat_norm = np.linalg.norm(body_quat_wxyz_w, axis=-1)
    return {
        "name": base_name,
        "input_csv": str(csv_path),
        "input_csv_sha256": sha256_file(csv_path),
        "output_npz": str(npz_path),
        "validator_json": str(validator_json),
        "time_steps": frames,
        "joint_count": int(joint_pos.shape[1]),
        "urdf_body_count": len(link_names),
        "tracking_body_count": len(tracking_indices),
        "tracking_body_indices_urdf_order": tracking_indices,
        "finite": bool(
            np.isfinite(joint_pos).all()
            and np.isfinite(joint_vel).all()
            and np.isfinite(body_pos_w).all()
            and np.isfinite(body_quat_wxyz_w).all()
            and np.isfinite(body_lin_vel_w).all()
            and np.isfinite(body_ang_vel_w).all()
        ),
        "body_quat_norm_max_abs_error_from_1": float(np.max(np.abs(quat_norm - 1.0))),
        "selected_tracking_body_pos_shape": list(selected.shape),
        "validator": validator,
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "input_csv",
        "output_npz",
        "time_steps",
        "joint_count",
        "urdf_body_count",
        "tracking_body_count",
        "finite",
        "body_quat_norm_max_abs_error_from_1",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--input-fps", type=int, default=30)
    parser.add_argument("--output-fps", type=int, default=50)
    parser.add_argument("--start-frame", type=int, default=1)
    parser.add_argument("--end-frame", type=int, default=180)
    parser.add_argument("--csv", type=Path, action="append", default=None)
    args = parser.parse_args()

    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)
    csv_paths = args.csv or DEFAULT_CSVS
    link_names = urdf_link_order(args.urdf)
    rows = [build_one(path, args, link_names) for path in csv_paths]
    tsv_path = OUT_RES / "tracking_motion_npz_fixture.tsv"
    json_path = OUT_RES / "tracking_motion_npz_fixture.json"
    write_tsv(tsv_path, rows)

    max_quat_error = max(row["body_quat_norm_max_abs_error_from_1"] for row in rows)
    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only_contract_fixture",
        "scope": "non-Kit URDF-FK motion.npz fixtures matching the tracking MotionLoader key/rank contract",
        "not_a_replacement_for": [
            "official csv_to_npz.py Isaac/Kit articulation export",
            "rendered replay_npz.py validation",
            "PPO tracking training smoke",
            "paper-level motion-tracking rollout metrics",
        ],
        "sources": {
            "urdf": str(args.urdf),
            "urdf_sha256": sha256_file(args.urdf),
            "fixture_builder": str(ROOT / "reproduction/scripts/build_tracking_motion_npz_fixture.py"),
            "validator": str(ROOT / "reproduction/scripts/validate_motion_npz_contract.py"),
            "official_csv_to_npz": str(
                ROOT / "reproduction/third_party/official/whole_body_tracking/scripts/csv_to_npz.py"
            ),
            "official_motion_loader": str(
                ROOT
                / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py"
            ),
        },
        "settings": {
            "input_fps": args.input_fps,
            "output_fps": args.output_fps,
            "start_frame_one_based": args.start_frame,
            "end_frame_one_based_inclusive": args.end_frame,
            "body_order": "URDF link order, recorded as debug evidence because live Isaac body order is unavailable while Kit is blocked",
            "body_quaternion_convention": "wxyz, converted from the FK helper's xyzw convention to match Isaac/whole_body_tracking usage",
        },
        "body_names_urdf_order": link_names,
        "tracking_body_names": G1_TRACKING_BODY_NAMES,
        "rows": rows,
        "metrics": {
            "fixture_count": len(rows),
            "total_time_steps": int(sum(row["time_steps"] for row in rows)),
            "min_time_steps": int(min(row["time_steps"] for row in rows)),
            "max_time_steps": int(max(row["time_steps"] for row in rows)),
            "joint_count": len(OFFICIAL_CSV_JOINT_NAMES),
            "urdf_body_count": len(link_names),
            "tracking_body_count": len(G1_TRACKING_BODY_NAMES),
            "max_body_quat_norm_abs_error_from_1": float(max_quat_error),
        },
        "checks": {
            "all_outputs_exist": all(Path(row["output_npz"]).is_file() for row in rows),
            "all_validator_contracts_pass": all(set(REQUIRED_KEYS).issubset(row["validator"].keys()) for row in rows),
            "all_arrays_finite": all(row["finite"] for row in rows),
            "output_fps_50": args.output_fps == 50,
            "joint_count_29": len(OFFICIAL_CSV_JOINT_NAMES) == 29 and all(row["joint_count"] == 29 for row in rows),
            "full_urdf_body_count_40": len(link_names) == 40 and all(row["urdf_body_count"] == 40 for row in rows),
            "tracking_body_names_present": set(G1_TRACKING_BODY_NAMES).issubset(set(link_names)),
            "selected_tracking_body_count_14": all(row["tracking_body_count"] == 14 for row in rows),
            "quaternions_unit_within_1e_minus_10": max_quat_error < 1e-10,
            "kit_execution_boundary_recorded": True,
        },
        "interpretation": {
            "paper_level_status": "partial",
            "why_not_complete": (
                "The fixtures validate the local motion.npz key/rank contract and provide full-body URDF-FK arrays "
                "for downstream static/debug consumers. They do not execute Isaac/Kit, cannot confirm live Isaac "
                "body ordering, and are not official articulation-state exports."
            ),
        },
        "outputs": {
            "json": str(json_path),
            "tsv": str(tsv_path),
            "fixture_dir": str(OUT_DATA),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(json_path), "fixture_count": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
