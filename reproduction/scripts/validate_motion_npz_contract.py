#!/usr/bin/env python3
"""Validate the motion.npz contract used by whole_body_tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


REQUIRED_KEYS = {
    "fps": 1,
    "joint_pos": 2,
    "joint_vel": 2,
    "body_pos_w": 3,
    "body_quat_w": 3,
    "body_lin_vel_w": 3,
    "body_ang_vel_w": 3,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("motion_npz", type=Path)
    parser.add_argument("--expect-fps", type=float, default=50.0)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()

    if not args.motion_npz.is_file():
        raise FileNotFoundError(args.motion_npz)

    data = np.load(args.motion_npz)
    missing = sorted(set(REQUIRED_KEYS) - set(data.files))
    if missing:
        raise ValueError(f"missing keys: {missing}")

    summary = {}
    for key, ndim in REQUIRED_KEYS.items():
        arr = data[key]
        if arr.ndim != ndim:
            raise ValueError(f"{key}: expected ndim {ndim}, got {arr.ndim}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{key}: non-finite values present")
        summary[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}

    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if abs(fps - args.expect_fps) > 1e-6:
        raise ValueError(f"fps: expected {args.expect_fps}, got {fps}")

    time_steps = data["joint_pos"].shape[0]
    for key in ["joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]:
        if data[key].shape[0] != time_steps:
            raise ValueError(f"{key}: timestep mismatch with joint_pos")
    if data["body_pos_w"].shape[-1] != 3 or data["body_lin_vel_w"].shape[-1] != 3 or data["body_ang_vel_w"].shape[-1] != 3:
        raise ValueError("body vector arrays must have final dimension 3")
    if data["body_quat_w"].shape[-1] != 4:
        raise ValueError("body_quat_w must have final dimension 4")

    summary["time_steps"] = time_steps
    summary["fps_value"] = fps
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
