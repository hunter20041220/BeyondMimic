#!/usr/bin/env python3
"""Validate the Level C motion-state fixture contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_KEYS = {
    "fps": 1,
    "times": 1,
    "root_pos_w": 2,
    "root_quat_xyzw_w": 2,
    "root_lin_vel_w": 2,
    "root_ang_vel_w": 2,
    "joint_pos": 2,
    "joint_vel": 2,
    "body_pos_w": 3,
    "body_quat_xyzw_w": 3,
    "body_lin_vel_w": 3,
    "body_ang_vel_w": 3,
    "candidate_hybrid_state": 2,
    "candidate_hybrid_state_windows": 3,
    "window_start_indices": 1,
    "emphasis_projection_weights": 1,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture_npz", type=Path)
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--expect-fps", type=float, default=50.0)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=16)
    args = parser.parse_args()

    data = np.load(args.fixture_npz)
    manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))

    missing = sorted(set(REQUIRED_KEYS) - set(data.files))
    if missing:
        raise ValueError(f"missing fixture keys: {missing}")

    summary = {"keys": sorted(data.files), "arrays": {}}
    for key, ndim in REQUIRED_KEYS.items():
        arr = data[key]
        if arr.ndim != ndim:
            raise ValueError(f"{key}: expected ndim {ndim}, got {arr.ndim}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{key}: non-finite values present")
        summary["arrays"][key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}

    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if abs(fps - args.expect_fps) > 1e-9:
        raise ValueError(f"fps mismatch: expected {args.expect_fps}, got {fps}")

    frames = data["joint_pos"].shape[0]
    for key in [
        "times",
        "root_pos_w",
        "root_quat_xyzw_w",
        "root_lin_vel_w",
        "root_ang_vel_w",
        "joint_vel",
        "body_pos_w",
        "body_quat_xyzw_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "candidate_hybrid_state",
    ]:
        if data[key].shape[0] != frames:
            raise ValueError(f"{key}: frame mismatch with joint_pos")

    if data["joint_pos"].shape[1] != 29:
        raise ValueError("joint_pos must have 29 G1 retargeted DOFs")
    if data["body_pos_w"].shape[1:] != (14, 3):
        raise ValueError("body_pos_w must be [T, 14, 3]")
    if data["body_quat_xyzw_w"].shape[1:] != (14, 4):
        raise ValueError("body_quat_xyzw_w must be [T, 14, 4]")

    expected_window_len = args.history + 1 + args.horizon
    windows = data["candidate_hybrid_state_windows"]
    if windows.shape[1] != expected_window_len:
        raise ValueError(f"window length mismatch: expected {expected_window_len}, got {windows.shape[1]}")
    if windows.shape[2] != data["candidate_hybrid_state"].shape[1]:
        raise ValueError("window feature dimension mismatch")
    if windows.shape[0] != data["window_start_indices"].shape[0]:
        raise ValueError("window count mismatch")

    checks = manifest.get("checks", {})
    if manifest.get("experiment_type") != "debug_only":
        raise ValueError("manifest must mark this fixture as debug_only")
    if manifest.get("status") != "ok":
        raise ValueError("manifest status is not ok")
    if not checks.get("finite_all_arrays", False):
        raise ValueError("manifest finite_all_arrays check is not true")
    if float(checks.get("global_xy_yaw_invariance_max_abs_error", 1.0)) > 1e-9:
        raise ValueError("global XY/yaw invariance check is too large")
    if float(checks.get("emphasis_projection_pseudoinverse_max_abs_error", 1.0)) > 1e-9:
        raise ValueError("emphasis projection pseudoinverse check is too large")

    summary["manifest"] = {
        "status": manifest["status"],
        "experiment_type": manifest["experiment_type"],
        "scope": manifest["scope"],
        "checks": checks,
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
