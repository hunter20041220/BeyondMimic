#!/usr/bin/env python3
"""Audit trajectory coordinate transforms and inverse reconstruction for Level C."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DEFAULT_FIXTURE = ROOT / "reproduction/data/level_c_fixtures/walk1_subject1_frames_1_180_state_fixture.npz"
OUT = ROOT / "res/level_c/trajectory_inverse_transform_audit"


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
    out = np.array([x, y, z, w], dtype=np.float64)
    if out[3] < 0.0:
        out *= -1.0
    return out / np.linalg.norm(out)


def rotz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def yaw_from_matrix(rot: np.ndarray) -> float:
    return math.atan2(float(rot[1, 0]), float(rot[0, 0]))


def quat_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = abs(float(np.dot(a, b)))
    dot = max(-1.0, min(1.0, dot))
    return 2.0 * math.acos(dot)


def paper_window_transform(
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    root_lin_vel: np.ndarray,
    root_ang_vel: np.ndarray,
    body_pos: np.ndarray,
    body_lin_vel: np.ndarray,
    current_index: int,
) -> dict[str, np.ndarray]:
    root_rot = np.stack([quat_xyzw_to_matrix(q) for q in root_quat], axis=0)
    current_yaw = yaw_from_matrix(root_rot[current_index])
    current_yaw_rot = rotz(current_yaw)
    current_yaw_inv = current_yaw_rot.T
    root_yaw_rot = np.stack([rotz(yaw_from_matrix(rot)) for rot in root_rot], axis=0)

    return {
        "root_pos_rel_current": (current_yaw_inv @ (root_pos - root_pos[current_index]).T).T,
        "root_rot_rel_current": np.stack([current_yaw_inv @ rot for rot in root_rot], axis=0),
        "root_lin_vel_rel_current": (current_yaw_inv @ (root_lin_vel - root_lin_vel[current_index]).T).T,
        "root_ang_vel_rel_current": (current_yaw_inv @ root_ang_vel.T).T,
        "body_pos_local": np.stack(
            [(root_yaw_rot[t].T @ (body_pos[t] - root_pos[t]).T).T for t in range(root_pos.shape[0])],
            axis=0,
        ),
        "body_lin_vel_local": np.stack(
            [(root_yaw_rot[t].T @ (body_lin_vel[t] - root_lin_vel[t]).T).T for t in range(root_pos.shape[0])],
            axis=0,
        ),
        "root_yaw_rot": root_yaw_rot,
        "current_yaw_rot": current_yaw_rot,
    }


def inverse_paper_window_transform(
    features: dict[str, np.ndarray],
    current_root_pos: np.ndarray,
    current_root_lin_vel: np.ndarray,
) -> dict[str, np.ndarray]:
    current_yaw_rot = features["current_yaw_rot"]
    root_yaw_rot = features["root_yaw_rot"]
    root_pos = (current_yaw_rot @ features["root_pos_rel_current"].T).T + current_root_pos
    root_rot = np.stack([current_yaw_rot @ rot for rot in features["root_rot_rel_current"]], axis=0)
    root_quat = np.stack([matrix_to_quat_xyzw(rot) for rot in root_rot], axis=0)
    root_lin_vel = (current_yaw_rot @ features["root_lin_vel_rel_current"].T).T + current_root_lin_vel
    root_ang_vel = (current_yaw_rot @ features["root_ang_vel_rel_current"].T).T
    body_pos = np.stack(
        [(root_yaw_rot[t] @ features["body_pos_local"][t].T).T + root_pos[t] for t in range(root_pos.shape[0])],
        axis=0,
    )
    body_lin_vel = np.stack(
        [(root_yaw_rot[t] @ features["body_lin_vel_local"][t].T).T + root_lin_vel[t] for t in range(root_pos.shape[0])],
        axis=0,
    )
    return {
        "root_pos_w": root_pos,
        "root_quat_xyzw_w": root_quat,
        "root_lin_vel_w": root_lin_vel,
        "root_ang_vel_w": root_ang_vel,
        "body_pos_w": body_pos,
        "body_lin_vel_w": body_lin_vel,
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["window_id", "current_frame", "root_pos_max_abs", "root_vel_max_abs", "root_quat_max_angle", "body_pos_max_abs", "body_vel_max_abs"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(DEFAULT_FIXTURE)
    starts = data["window_start_indices"].astype(np.int64)
    history = 4
    horizon = 16
    length = history + 1 + horizon
    rows: list[dict[str, Any]] = []
    reconstructed_payload: dict[str, np.ndarray] = {}

    for window_id, current in enumerate(starts):
        sl = slice(current - history, current + horizon + 1)
        root_pos = data["root_pos_w"][sl]
        root_quat = data["root_quat_xyzw_w"][sl]
        root_lin_vel = data["root_lin_vel_w"][sl]
        root_ang_vel = data["root_ang_vel_w"][sl]
        body_pos = data["body_pos_w"][sl]
        body_lin_vel = data["body_lin_vel_w"][sl]
        features = paper_window_transform(
            root_pos,
            root_quat,
            root_lin_vel,
            root_ang_vel,
            body_pos,
            body_lin_vel,
            current_index=history,
        )
        recon = inverse_paper_window_transform(features, root_pos[history], root_lin_vel[history])
        quat_angle = max(quat_distance(a, b) for a, b in zip(recon["root_quat_xyzw_w"], root_quat))
        row = {
            "window_id": window_id,
            "current_frame": int(current),
            "root_pos_max_abs": float(np.max(np.abs(recon["root_pos_w"] - root_pos))),
            "root_vel_max_abs": float(np.max(np.abs(recon["root_lin_vel_w"] - root_lin_vel))),
            "root_quat_max_angle": float(quat_angle),
            "body_pos_max_abs": float(np.max(np.abs(recon["body_pos_w"] - body_pos))),
            "body_vel_max_abs": float(np.max(np.abs(recon["body_lin_vel_w"] - body_lin_vel))),
        }
        rows.append(row)
        if window_id == 0:
            for key, value in features.items():
                if key.endswith("_rot"):
                    continue
                reconstructed_payload[f"first_window_{key}"] = value
            for key, value in recon.items():
                reconstructed_payload[f"first_window_reconstructed_{key}"] = value

    candidate_state = data["candidate_hybrid_state"]
    expected_paper_window_root_translation_dims = 3
    candidate_root_translation_encoded = False
    root_pos_errors = np.asarray([row["root_pos_max_abs"] for row in rows], dtype=np.float64)
    root_vel_errors = np.asarray([row["root_vel_max_abs"] for row in rows], dtype=np.float64)
    root_quat_errors = np.asarray([row["root_quat_max_angle"] for row in rows], dtype=np.float64)
    body_pos_errors = np.asarray([row["body_pos_max_abs"] for row in rows], dtype=np.float64)
    body_vel_errors = np.asarray([row["body_vel_max_abs"] for row in rows], dtype=np.float64)

    json_path = OUT / "level_c_trajectory_inverse_transform_audit.json"
    tsv_path = OUT / "level_c_trajectory_inverse_transform_audit.tsv"
    npz_path = OUT / "level_c_trajectory_inverse_transform_audit.npz"
    np.savez_compressed(
        npz_path,
        root_pos_errors=root_pos_errors,
        root_vel_errors=root_vel_errors,
        root_quat_errors=root_quat_errors,
        body_pos_errors=body_pos_errors,
        body_vel_errors=body_vel_errors,
        **reconstructed_payload,
    )
    write_tsv(tsv_path, rows)

    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "paper-formula trajectory coordinate transform and inverse round-trip audit on the Level C fixture",
        "paper_evidence": {
            "character_yaw_frame": str(ROOT / "reproduction/paper/source/root.tex:482-489"),
            "root_relative_formula": str(ROOT / "reproduction/paper/source/root.tex:490-506"),
            "body_local_formula": str(ROOT / "reproduction/paper/source/root.tex:508-519"),
            "emphasis_projection_pseudoinverse": str(ROOT / "reproduction/paper/source/root.tex:524-532"),
            "goal_inverse_transform": str(ROOT / "goal.md:1478-1484"),
        },
        "not_a_replacement_for": [
            "official Isaac/Kit motion.npz",
            "teacher rollout trajectory dataset",
            "trained VAE latent trajectory dataset",
            "paper-level diffusion training",
        ],
        "settings": {
            "fixture_npz": str(DEFAULT_FIXTURE),
            "window_count": int(len(starts)),
            "history": history,
            "horizon": horizon,
            "sequence_length": length,
            "state_dim_in_existing_debug_fixture": int(candidate_state.shape[1]),
            "expected_paper_window_root_translation_dims": expected_paper_window_root_translation_dims,
            "existing_debug_fixture_root_translation_encoded": candidate_root_translation_encoded,
        },
        "metrics": {
            "max_root_pos_roundtrip_abs_error": float(root_pos_errors.max()),
            "max_root_lin_vel_roundtrip_abs_error": float(root_vel_errors.max()),
            "max_root_quat_roundtrip_angle_rad": float(root_quat_errors.max()),
            "max_body_pos_roundtrip_abs_error": float(body_pos_errors.max()),
            "max_body_lin_vel_roundtrip_abs_error": float(body_vel_errors.max()),
        },
        "checks": {
            "all_windows_checked": len(rows) == len(starts) and len(rows) > 0,
            "paper_formula_root_inverse_roundtrip": bool(root_pos_errors.max() < 1e-12 and root_vel_errors.max() < 1e-12),
            "paper_formula_root_rotation_roundtrip": bool(root_quat_errors.max() < 1e-12),
            "paper_formula_body_inverse_roundtrip": bool(body_pos_errors.max() < 1e-12 and body_vel_errors.max() < 1e-12),
            "existing_debug_fixture_not_full_paper_root_window_state": not candidate_root_translation_encoded,
        },
        "interpretation": {
            "paper_level_status": "partial",
            "why_not_complete": (
                "The paper-formula transform and inverse round-trip pass on the local fixture, but the existing debug "
                "candidate_hybrid_state does not encode the full paper window-current-frame root translation. This audit "
                "therefore strengthens the math gate without claiming the official trainable state-latent dataset."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
