#!/usr/bin/env python3
"""Build paper-formula Level C state-window debug artifacts from motion fixtures."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
FIXTURE_DIR = ROOT / "reproduction/data/level_c_fixtures"
MANIFEST_DIR = ROOT / "res/level_c/motion_state_fixture"
OUT_DATA = ROOT / "reproduction/data/level_c_paper_state_windows"
OUT_RES = ROOT / "res/level_c/paper_state_windows"
DEFAULT_FIXTURE_NAMES = [
    "walk1_subject1_frames_1_180_state_fixture",
    "run2_subject1_frames_1_180_state_fixture",
    "jumps1_subject1_frames_1_180_state_fixture",
]
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


def add_feature(parts: list[np.ndarray], slices: dict[str, list[int]], name: str, value: np.ndarray) -> None:
    start = sum(part.shape[-1] for part in parts)
    flat = value.reshape(value.shape[0], -1)
    parts.append(flat)
    slices[name] = [start, start + flat.shape[-1]]


def paper_window_state(data: np.lib.npyio.NpzFile, current_frame: int) -> tuple[np.ndarray, dict[str, list[int]]]:
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
    root_rot6d_rel_current = np.stack([rot6d(current_yaw_inv @ rot) for rot in root_rot], axis=0)
    root_lin_vel_rel_current = (current_yaw_inv @ (root_lin_vel - root_lin_vel[current]).T).T
    root_ang_vel_rel_current = (current_yaw_inv @ root_ang_vel.T).T

    body_pos_local = np.empty((seq_len, BODY_COUNT, 3), dtype=np.float64)
    body_lin_vel_local = np.empty((seq_len, BODY_COUNT, 3), dtype=np.float64)
    for i, rot in enumerate(root_rot):
        yaw_inv = rotz(-yaw_from_matrix(rot))
        body_pos_local[i] = (yaw_inv @ (body_pos[i] - root_pos[i]).T).T
        body_lin_vel_local[i] = (yaw_inv @ (body_lin_vel[i] - root_lin_vel[i]).T).T

    parts: list[np.ndarray] = []
    feature_slices: dict[str, list[int]] = {}
    add_feature(parts, feature_slices, "root_pos_rel_current_frame", root_pos_rel_current)
    add_feature(parts, feature_slices, "root_rot6d_rel_current_frame", root_rot6d_rel_current)
    add_feature(parts, feature_slices, "root_lin_vel_rel_current_frame", root_lin_vel_rel_current)
    add_feature(parts, feature_slices, "root_ang_vel_rel_current_frame", root_ang_vel_rel_current)
    add_feature(parts, feature_slices, "body_pos_local_root_frame", body_pos_local)
    add_feature(parts, feature_slices, "body_lin_vel_local_root_frame", body_lin_vel_local)
    return np.concatenate(parts, axis=-1), feature_slices


def build_motion(name: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    source_npz = FIXTURE_DIR / f"{name}.npz"
    source_manifest = MANIFEST_DIR / f"{name}.json"
    data = np.load(source_npz)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    starts = data["window_start_indices"].astype(np.int64)
    windows: list[np.ndarray] = []
    feature_slices: dict[str, list[int]] | None = None
    for current_frame in starts:
        window, slices = paper_window_state(data, int(current_frame))
        windows.append(window)
        if feature_slices is None:
            feature_slices = slices
        elif feature_slices != slices:
            raise ValueError(f"{name}: inconsistent feature slices")
    if feature_slices is None:
        raise ValueError(f"{name}: no windows")

    paper_windows = np.stack(windows, axis=0)
    root_current = paper_windows[:, HISTORY, feature_slices["root_pos_rel_current_frame"][0] : feature_slices["root_pos_rel_current_frame"][1]]
    root_lin_current = paper_windows[
        :, HISTORY, feature_slices["root_lin_vel_rel_current_frame"][0] : feature_slices["root_lin_vel_rel_current_frame"][1]
    ]
    finite = bool(np.isfinite(paper_windows).all())
    out_npz = OUT_DATA / f"{name}_paper_state_windows.npz"
    np.savez_compressed(
        out_npz,
        paper_state_windows=paper_windows,
        window_start_indices=starts,
        source_candidate_hybrid_state_windows=data["candidate_hybrid_state_windows"],
        feature_slices_json=np.array([json.dumps(feature_slices, sort_keys=True)]),
        history=np.array([HISTORY], dtype=np.int64),
        horizon=np.array([HORIZON], dtype=np.int64),
        fps=data["fps"],
    )
    row = {
        "name": name,
        "status": "ok",
        "experiment_type": "debug_only",
        "source_fixture_npz": str(source_npz),
        "source_fixture_manifest": str(source_manifest),
        "output_npz": str(out_npz),
        "window_count": int(paper_windows.shape[0]),
        "sequence_length": int(paper_windows.shape[1]),
        "paper_state_dim": int(paper_windows.shape[2]),
        "source_candidate_state_dim": int(data["candidate_hybrid_state_windows"].shape[2]),
        "feature_slices": feature_slices,
        "source_fixture_experiment_type": manifest.get("experiment_type"),
        "checks": {
            "finite_all_arrays": finite,
            "history_horizon_sequence_length": paper_windows.shape[1] == HISTORY + 1 + HORIZON,
            "state_dim_matches_source_formula_terms": paper_windows.shape[2] == 99,
            "current_root_relative_position_zero": float(np.max(np.abs(root_current))) < 1e-12,
            "current_root_relative_linear_velocity_zero": float(np.max(np.abs(root_lin_current))) < 1e-12,
            "source_fixture_debug_only": manifest.get("experiment_type") == "debug_only",
        },
        "metrics": {
            "max_current_root_relative_position_abs": float(np.max(np.abs(root_current))),
            "max_current_root_relative_linear_velocity_abs": float(np.max(np.abs(root_lin_current))),
            "paper_state_abs_max": float(np.max(np.abs(paper_windows))),
            "paper_state_std": float(np.std(paper_windows)),
        },
    }
    payload = {f"{name}_paper_state_windows": paper_windows}
    return row, payload


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "window_count",
        "sequence_length",
        "paper_state_dim",
        "source_candidate_state_dim",
        "max_current_root_relative_position_abs",
        "max_current_root_relative_linear_velocity_abs",
        "paper_state_abs_max",
        "paper_state_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "name": row["name"],
                    "window_count": row["window_count"],
                    "sequence_length": row["sequence_length"],
                    "paper_state_dim": row["paper_state_dim"],
                    "source_candidate_state_dim": row["source_candidate_state_dim"],
                    "max_current_root_relative_position_abs": row["metrics"][
                        "max_current_root_relative_position_abs"
                    ],
                    "max_current_root_relative_linear_velocity_abs": row["metrics"][
                        "max_current_root_relative_linear_velocity_abs"
                    ],
                    "paper_state_abs_max": row["metrics"]["paper_state_abs_max"],
                    "paper_state_std": row["metrics"]["paper_state_std"],
                }
            )


def main() -> None:
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    OUT_RES.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    payload: dict[str, np.ndarray] = {}
    for name in DEFAULT_FIXTURE_NAMES:
        row, motion_payload = build_motion(name)
        rows.append(row)
        payload.update(motion_payload)

    all_checks = [value for row in rows for value in row["checks"].values()]
    summary_npz = OUT_RES / "level_c_paper_state_windows_summary.npz"
    json_path = OUT_RES / "level_c_paper_state_windows.json"
    tsv_path = OUT_RES / "level_c_paper_state_windows.tsv"
    np.savez_compressed(summary_npz, **payload)
    write_tsv(tsv_path, rows)
    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "paper-formula state-window debug artifacts built from non-Kit motion fixtures",
        "paper_evidence": {
            "trajectory_modeling_state_summary": str(ROOT / "reproduction/paper/source/tex/method.tex:139-145"),
            "diffusion_state_character_frame": str(ROOT / "reproduction/paper/source/root.tex:482-532"),
            "diffusion_horizon_history": str(ROOT / "reproduction/paper/source/root.tex:827-856"),
        },
        "not_a_replacement_for": [
            "teacher rollout state-latent dataset",
            "VAE latent trajectories",
            "paper-exact emphasis projection",
            "full diffusion training",
        ],
        "settings": {
            "fixture_names": DEFAULT_FIXTURE_NAMES,
            "history": HISTORY,
            "horizon": HORIZON,
            "sequence_length": HISTORY + 1 + HORIZON,
            "body_count": BODY_COUNT,
            "paper_state_dim": 99,
            "state_terms": [
                "root_pos_rel_current_frame",
                "root_rot6d_rel_current_frame",
                "root_lin_vel_rel_current_frame",
                "root_ang_vel_rel_current_frame",
                "body_pos_local_root_frame",
                "body_lin_vel_local_root_frame",
            ],
        },
        "rows": rows,
        "counts": {
            "motion_count": len(rows),
            "window_count": int(sum(row["window_count"] for row in rows)),
            "sample_count": int(sum(row["window_count"] * row["sequence_length"] for row in rows)),
        },
        "checks": {
            "all_motion_outputs_written": all(Path(row["output_npz"]).exists() for row in rows),
            "all_source_fixtures_debug_only": all(row["checks"]["source_fixture_debug_only"] for row in rows),
            "all_arrays_finite": all(row["checks"]["finite_all_arrays"] for row in rows),
            "all_windows_match_history_horizon": all(row["checks"]["history_horizon_sequence_length"] for row in rows),
            "all_state_dims_match_source_formula_terms": all(
                row["checks"]["state_dim_matches_source_formula_terms"] for row in rows
            ),
            "all_current_root_relative_positions_zero": all(
                row["checks"]["current_root_relative_position_zero"] for row in rows
            ),
            "all_current_root_relative_linear_velocities_zero": all(
                row["checks"]["current_root_relative_linear_velocity_zero"] for row in rows
            ),
            "all_checks_pass": bool(all(all_checks)),
            "paper_exact_trainable_state_not_claimed": True,
        },
        "interpretation": {
            "paper_level_status": "partial",
            "why_not_complete": (
                "This builds paper-formula 99-D state windows from debug motion fixtures and fixes the previously "
                "identified root-current-frame/body-velocity state mismatch for this local artifact. It still lacks "
                "VAE latents, teacher rollouts, live accept/reject collection, paper-exact emphasis projection, and "
                "trained diffusion checkpoints."
            ),
        },
        "outputs": {
            "json": str(json_path),
            "tsv": str(tsv_path),
            "summary_npz": str(summary_npz),
            "data_dir": str(OUT_DATA),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
