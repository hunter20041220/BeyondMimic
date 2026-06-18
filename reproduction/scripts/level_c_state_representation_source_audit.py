#!/usr/bin/env python3
"""Audit Level C state representation against the paper/source formulas."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/state_representation_source_audit"
FIXTURE_DIR = ROOT / "reproduction/data/level_c_fixtures"
MANIFEST_DIR = ROOT / "res/level_c/motion_state_fixture"
FIXTURE_NAMES = [
    "walk1_subject1_frames_1_180_state_fixture",
    "run2_subject1_frames_1_180_state_fixture",
    "jumps1_subject1_frames_1_180_state_fixture",
]
METHOD_TEX = ROOT / "reproduction/paper/source/tex/method.tex"
ROOT_TEX = ROOT / "reproduction/paper/source/root.tex"
HISTORY = 4
HORIZON = 16
BODY_COUNT = 14


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


def rotz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def yaw_from_matrix(rot: np.ndarray) -> float:
    return math.atan2(float(rot[1, 0]), float(rot[0, 0]))


def rot6d(rot: np.ndarray) -> np.ndarray:
    return rot[:, :2].reshape(6, order="F")


def text_has_patterns(path: Path, patterns: list[str]) -> bool:
    text = path.read_text(encoding="utf-8")
    return all(pattern in text for pattern in patterns)


def slc(feature_slices: dict[str, list[int]], name: str) -> slice:
    lo, hi = feature_slices[name]
    return slice(lo, hi)


def paper_window_features(data: np.lib.npyio.NpzFile, current_frame: int) -> dict[str, np.ndarray]:
    seq_len = HISTORY + 1 + HORIZON
    sl = slice(current_frame - HISTORY, current_frame + HORIZON + 1)
    root_pos = data["root_pos_w"][sl]
    root_quat = data["root_quat_xyzw_w"][sl]
    root_lin_vel = data["root_lin_vel_w"][sl]
    root_ang_vel = data["root_ang_vel_w"][sl]
    body_pos = data["body_pos_w"][sl]
    body_lin_vel = data["body_lin_vel_w"][sl]
    root_rot = np.stack([quat_xyzw_to_matrix(q) for q in root_quat], axis=0)
    current = HISTORY
    current_yaw_inv = rotz(-yaw_from_matrix(root_rot[current]))

    root_pos_rel_current = (current_yaw_inv @ (root_pos - root_pos[current]).T).T
    root_rot_rel_current = np.stack([rot6d(current_yaw_inv @ rot) for rot in root_rot], axis=0)
    root_lin_vel_rel_current = (current_yaw_inv @ (root_lin_vel - root_lin_vel[current]).T).T
    root_ang_vel_rel_current = (current_yaw_inv @ root_ang_vel.T).T

    body_pos_local = np.empty((seq_len, BODY_COUNT, 3), dtype=np.float64)
    body_lin_vel_local = np.empty((seq_len, BODY_COUNT, 3), dtype=np.float64)
    for i, rot in enumerate(root_rot):
        yaw_inv = rotz(-yaw_from_matrix(rot))
        body_pos_local[i] = (yaw_inv @ (body_pos[i] - root_pos[i]).T).T
        body_lin_vel_local[i] = (yaw_inv @ (body_lin_vel[i] - root_lin_vel[i]).T).T

    return {
        "root_pos_rel_current": root_pos_rel_current,
        "root_rot6d_rel_current": root_rot_rel_current,
        "root_lin_vel_rel_current": root_lin_vel_rel_current,
        "root_ang_vel_rel_current": root_ang_vel_rel_current,
        "body_pos_local": body_pos_local.reshape(seq_len, -1),
        "body_lin_vel_local": body_lin_vel_local.reshape(seq_len, -1),
    }


def analyze_fixture(name: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    npz_path = FIXTURE_DIR / f"{name}.npz"
    manifest_path = MANIFEST_DIR / f"{name}.json"
    data = np.load(npz_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    slices = manifest["feature_slices"]
    starts = data["window_start_indices"].astype(np.int64)
    windows = data["candidate_hybrid_state_windows"].astype(np.float64)

    body_pos_errors = []
    body_vel_errors = []
    root_rot_errors = []
    root_lin_vel_errors = []
    root_ang_vel_errors = []
    root_current_zero_errors = []
    paper_root_xy_span = []
    first_paper_window: dict[str, np.ndarray] = {}

    for window_id, current_frame in enumerate(starts):
        paper = paper_window_features(data, int(current_frame))
        candidate = windows[window_id]
        if window_id == 0:
            first_paper_window = {f"first_{key}": value for key, value in paper.items()}

        body_pos_errors.append(
            float(np.max(np.abs(candidate[:, slc(slices, "body_pos_yaw_frame")] - paper["body_pos_local"])))
        )
        body_vel_errors.append(
            float(np.max(np.abs(candidate[:, slc(slices, "body_lin_vel_yaw_frame")] - paper["body_lin_vel_local"])))
        )
        root_rot_errors.append(
            float(np.max(np.abs(candidate[:, slc(slices, "root_rot6d_without_yaw")] - paper["root_rot6d_rel_current"])))
        )
        root_lin_vel_errors.append(
            float(np.max(np.abs(candidate[:, slc(slices, "root_lin_vel_yaw_frame")] - paper["root_lin_vel_rel_current"])))
        )
        root_ang_vel_errors.append(
            float(np.max(np.abs(candidate[:, slc(slices, "root_ang_vel_yaw_frame")] - paper["root_ang_vel_rel_current"])))
        )
        root_current_zero_errors.append(float(np.max(np.abs(paper["root_pos_rel_current"][HISTORY]))))
        paper_root_xy_span.append(float(np.max(np.abs(paper["root_pos_rel_current"][:, :2]))))

    emphasis = data["emphasis_projection_weights"].astype(np.float64)
    root_feature_names = [
        "root_height",
        "root_rot6d_without_yaw",
        "root_lin_vel_yaw_frame",
        "root_ang_vel_yaw_frame",
    ]
    root_feature_mask = np.zeros_like(emphasis, dtype=bool)
    for feature_name in root_feature_names:
        root_feature_mask[slc(slices, feature_name)] = True

    candidate_dim = int(data["candidate_hybrid_state"].shape[1])
    paper_min_state_dim_without_body_orientation = 3 + 6 + 3 + 3 + BODY_COUNT * 3 + BODY_COUNT * 3
    row = {
        "fixture_name": name,
        "status": "ok",
        "experiment_type": manifest["experiment_type"],
        "window_count": int(len(starts)),
        "sequence_length": int(windows.shape[1]),
        "candidate_state_dim": candidate_dim,
        "paper_min_state_dim_without_body_orientation": paper_min_state_dim_without_body_orientation,
        "max_body_position_local_error": float(max(body_pos_errors)),
        "max_body_velocity_local_error": float(max(body_vel_errors)),
        "max_root_rotation_current_frame_error": float(max(root_rot_errors)),
        "max_root_linear_velocity_current_frame_error": float(max(root_lin_vel_errors)),
        "max_root_angular_velocity_current_frame_error": float(max(root_ang_vel_errors)),
        "max_paper_current_root_position_zero_error": float(max(root_current_zero_errors)),
        "max_paper_root_xy_span": float(max(paper_root_xy_span)),
        "candidate_has_body_orientation_features": "body_rot6d_without_yaw" in slices,
        "candidate_encodes_full_root_relative_position": False,
        "candidate_body_velocity_subtracts_root_velocity": bool(max(body_vel_errors) < 1e-12),
        "candidate_uses_window_current_frame_for_root_features": bool(
            max(root_rot_errors + root_lin_vel_errors + root_ang_vel_errors) < 1e-12
        ),
        "candidate_emphasis_is_diagonal_weights": bool(
            np.allclose(emphasis[root_feature_mask], 6.0) and np.allclose(emphasis[~root_feature_mask], 1.0)
        ),
    }
    return row, first_paper_window


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "fixture_name",
        "window_count",
        "sequence_length",
        "candidate_state_dim",
        "paper_min_state_dim_without_body_orientation",
        "max_body_position_local_error",
        "max_body_velocity_local_error",
        "max_root_rotation_current_frame_error",
        "max_root_linear_velocity_current_frame_error",
        "max_root_angular_velocity_current_frame_error",
        "max_paper_current_root_position_zero_error",
        "max_paper_root_xy_span",
        "candidate_has_body_orientation_features",
        "candidate_encodes_full_root_relative_position",
        "candidate_body_velocity_subtracts_root_velocity",
        "candidate_uses_window_current_frame_for_root_features",
        "candidate_emphasis_is_diagonal_weights",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    method_patterns = [
        "hybrid character",
        "with respect to the current root frame",
        "expressed relative to their local root frame",
        "state\u2013latent trajectories",
        "individual denoising steps",
    ]
    root_patterns = [
        "character frame",
        "current character frame",
        "p}_{\\text{root}}^{\\,\\text{rel}}",
        "p}_{b}^{\\,\\text{local}}",
        "emphasis projection",
        "c = 6",
        "pseudoinverse",
    ]
    rows: list[dict[str, Any]] = []
    npz_payload: dict[str, np.ndarray] = {}
    for fixture_name in FIXTURE_NAMES:
        row, first_payload = analyze_fixture(fixture_name)
        rows.append(row)
        for key, value in first_payload.items():
            npz_payload[f"{fixture_name}_{key}"] = value

    max_body_pos_error = max(row["max_body_position_local_error"] for row in rows)
    max_body_vel_error = max(row["max_body_velocity_local_error"] for row in rows)
    max_root_current_frame_error = max(
        max(
            row["max_root_rotation_current_frame_error"],
            row["max_root_linear_velocity_current_frame_error"],
            row["max_root_angular_velocity_current_frame_error"],
        )
        for row in rows
    )
    json_path = OUT / "level_c_state_representation_source_audit.json"
    tsv_path = OUT / "level_c_state_representation_source_audit.tsv"
    npz_path = OUT / "level_c_state_representation_source_audit.npz"
    write_tsv(tsv_path, rows)
    np.savez_compressed(npz_path, **npz_payload)

    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "source-aligned audit of the Level C hybrid character-yaw state representation and local fixture boundary",
        "paper_evidence": {
            "trajectory_modeling_state_summary": str(METHOD_TEX) + ":139-145",
            "state_latent_trajectory": str(METHOD_TEX) + ":171-185",
            "diffusion_state_character_frame": str(ROOT_TEX) + ":482-532",
            "diffusion_horizon_history": str(ROOT_TEX) + ":827-856",
        },
        "not_a_replacement_for": [
            "official paper state-latent rollout dataset",
            "trained VAE latent trajectories",
            "paper-exact emphasis projection matrix",
            "full diffusion training",
        ],
        "settings": {
            "fixture_names": FIXTURE_NAMES,
            "history": HISTORY,
            "horizon": HORIZON,
            "sequence_length": HISTORY + 1 + HORIZON,
            "body_count": BODY_COUNT,
            "paper_min_state_dim_without_body_orientation": 99,
            "candidate_state_dim": 181,
        },
        "rows": rows,
        "aggregate_metrics": {
            "max_body_position_local_error": max_body_pos_error,
            "max_body_velocity_local_error": max_body_vel_error,
            "max_root_current_frame_feature_error": max_root_current_frame_error,
        },
        "checks": {
            "method_source_patterns_found": text_has_patterns(METHOD_TEX, method_patterns),
            "supplement_source_patterns_found": text_has_patterns(ROOT_TEX, root_patterns),
            "all_fixtures_debug_only": all(row["experiment_type"] == "debug_only" for row in rows),
            "all_fixture_windows_match_history_horizon": all(row["sequence_length"] == HISTORY + 1 + HORIZON for row in rows),
            "candidate_body_position_matches_paper_local_formula": bool(max_body_pos_error < 1e-12),
            "candidate_body_velocity_missing_root_velocity_subtraction_detected": bool(max_body_vel_error > 1e-6),
            "candidate_root_current_frame_difference_detected": bool(max_root_current_frame_error > 1e-6),
            "candidate_full_root_relative_position_missing_detected": all(
                not row["candidate_encodes_full_root_relative_position"] for row in rows
            ),
            "candidate_body_orientation_extra_features_recorded": all(
                row["candidate_has_body_orientation_features"] for row in rows
            ),
            "candidate_emphasis_diagonal_simplification_recorded": all(
                row["candidate_emphasis_is_diagonal_weights"] for row in rows
            ),
            "paper_exact_trainable_state_not_claimed": True,
        },
        "interpretation": {
            "paper_level_status": "partial",
            "why_not_complete": (
                "The audit confirms source alignment for sequence length and body-position local-frame features, "
                "but it also quantifies that the current debug fixture is not the full paper S3 state: root features "
                "are not encoded in the window-current frame with full relative position, body velocities do not "
                "subtract root velocity, and the emphasis projection is a diagonal debug simplification rather than "
                "the random Gaussian/pseudoinverse projection described in the source."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
