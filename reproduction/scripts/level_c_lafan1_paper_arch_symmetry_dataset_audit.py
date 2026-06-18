#!/usr/bin/env python3
"""Build and audit a trainable sagittal-symmetry public LAFAN1 dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SCRIPTS = ROOT / "reproduction/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_level_c_motion_state_fixture as motion_fk  # noqa: E402
from level_c_symmetry_mapping_audit import CENTER_SIGN, SYMMETRY_PAIRS  # noqa: E402


OUT = ROOT / "res/level_c/lafan1_paper_arch_symmetry_dataset"
SOURCE_JSON = (
    ROOT
    / "res/level_c/lafan1_paper_arch_vae_diffusion_training/"
    / "lafan1_paper_arch_vae_diffusion_training.json"
)
SOURCE_NPZ = (
    ROOT
    / "res/level_c/lafan1_paper_arch_vae_diffusion_training/"
    / "lafan1_paper_arch_training_dataset.npz"
)
SYMMETRY_JSON = ROOT / "res/level_c/symmetry_mapping_audit/level_c_symmetry_mapping_audit.json"

FEATURE_SLICES = {
    "root_pos_rel_current_frame": (0, 3),
    "root_rot6d_rel_current_frame": (3, 9),
    "root_lin_vel_rel_current_frame": (9, 12),
    "root_ang_vel_rel_current_frame": (12, 15),
    "body_pos_local_root_frame": (15, 57),
    "body_lin_vel_local_root_frame": (57, 99),
}
POLAR_SIGN = np.asarray([1.0, -1.0, 1.0], dtype=np.float32)
AXIAL_SIGN = np.asarray([-1.0, 1.0, -1.0], dtype=np.float32)
SAGITTAL_REFLECTION = np.diag([1.0, -1.0, 1.0]).astype(np.float64)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "source_window_count",
        "mirror_window_count",
        "augmented_window_count",
        "source_token_count",
        "mirror_token_count",
        "augmented_token_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def rot6d_to_matrix(values: np.ndarray) -> np.ndarray:
    c1 = values[..., 0:3].astype(np.float64)
    c2 = values[..., 3:6].astype(np.float64)
    b1 = c1 / np.maximum(np.linalg.norm(c1, axis=-1, keepdims=True), 1e-12)
    c2_orth = c2 - np.sum(b1 * c2, axis=-1, keepdims=True) * b1
    b2 = c2_orth / np.maximum(np.linalg.norm(c2_orth, axis=-1, keepdims=True), 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def matrix_to_rot6d(rot: np.ndarray) -> np.ndarray:
    return rot[..., :, :2].reshape(rot.shape[:-2] + (6,), order="F").astype(np.float32)


def mirror_rot6d(values: np.ndarray) -> np.ndarray:
    rot = rot6d_to_matrix(values)
    mirrored = SAGITTAL_REFLECTION @ rot @ SAGITTAL_REFLECTION
    return matrix_to_rot6d(mirrored)


def mirror_index_and_sign(names: list[str]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    target_by_name: dict[str, tuple[str, float, str]] = {}
    for left, right, sign in SYMMETRY_PAIRS:
        target_by_name[left] = (right, float(sign), "left_right_pair")
        target_by_name[right] = (left, float(sign), "left_right_pair")
    for name, sign in CENTER_SIGN.items():
        target_by_name[name] = (name, float(sign), "center")

    name_to_index = {name: idx for idx, name in enumerate(names)}
    mirror_index = np.zeros(len(names), dtype=np.int64)
    mirror_sign = np.zeros(len(names), dtype=np.float32)
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        target, sign, kind = target_by_name.get(name, ("", 0.0, "missing"))
        mirror_index[idx] = name_to_index[target] if target in name_to_index else -1
        mirror_sign[idx] = sign
        rows.append(
            {
                "index": idx,
                "name": name,
                "mirror_name": target,
                "mirror_index": int(mirror_index[idx]),
                "sign": float(sign),
                "kind": kind,
                "covered": bool(target in name_to_index),
            }
        )
    return mirror_index, mirror_sign, rows


def body_mirror_index(names: list[str]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    name_to_index = {name: idx for idx, name in enumerate(names)}
    mirror_index = np.zeros(len(names), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        if name.startswith("left_"):
            target = "right_" + name[len("left_") :]
            kind = "left_right_pair"
        elif name.startswith("right_"):
            target = "left_" + name[len("right_") :]
            kind = "left_right_pair"
        else:
            target = name
            kind = "center"
        mirror_index[idx] = name_to_index[target] if target in name_to_index else -1
        rows.append(
            {
                "index": idx,
                "name": name,
                "mirror_name": target,
                "mirror_index": int(mirror_index[idx]),
                "kind": kind if target in name_to_index else "missing",
                "covered": bool(target in name_to_index),
            }
        )
    return mirror_index, rows


def mirror_actions(actions: np.ndarray, mirror_index: np.ndarray, mirror_sign: np.ndarray) -> np.ndarray:
    return (actions[..., mirror_index] * mirror_sign).astype(np.float32)


def mirror_body_vectors(values: np.ndarray, mirror_index: np.ndarray) -> np.ndarray:
    shaped = values.reshape(values.shape[:-1] + (len(mirror_index), 3))
    mirrored = shaped[..., mirror_index, :] * POLAR_SIGN
    return mirrored.reshape(values.shape).astype(np.float32)


def mirror_states(states: np.ndarray, body_index: np.ndarray) -> np.ndarray:
    mirrored = np.empty_like(states, dtype=np.float32)
    lo, hi = FEATURE_SLICES["root_pos_rel_current_frame"]
    mirrored[..., lo:hi] = states[..., lo:hi] * POLAR_SIGN
    lo, hi = FEATURE_SLICES["root_rot6d_rel_current_frame"]
    mirrored[..., lo:hi] = mirror_rot6d(states[..., lo:hi])
    lo, hi = FEATURE_SLICES["root_lin_vel_rel_current_frame"]
    mirrored[..., lo:hi] = states[..., lo:hi] * POLAR_SIGN
    lo, hi = FEATURE_SLICES["root_ang_vel_rel_current_frame"]
    mirrored[..., lo:hi] = states[..., lo:hi] * AXIAL_SIGN
    lo, hi = FEATURE_SLICES["body_pos_local_root_frame"]
    mirrored[..., lo:hi] = mirror_body_vectors(states[..., lo:hi], body_index)
    lo, hi = FEATURE_SLICES["body_lin_vel_local_root_frame"]
    mirrored[..., lo:hi] = mirror_body_vectors(states[..., lo:hi], body_index)
    return mirrored.astype(np.float32)


def split_rows(split_labels: np.ndarray, seq_len: int) -> list[dict[str, Any]]:
    rows = []
    for split in ["train", "validation", "test"]:
        count = int(np.sum(split_labels == split))
        rows.append(
            {
                "split": split,
                "source_window_count": count,
                "mirror_window_count": count,
                "augmented_window_count": count * 2,
                "source_token_count": count * seq_len,
                "mirror_token_count": count * seq_len,
                "augmented_token_count": count * seq_len * 2,
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    symmetry_summary = json.loads(SYMMETRY_JSON.read_text(encoding="utf-8"))
    source = np.load(SOURCE_NPZ, allow_pickle=True)

    states = source["states"].astype(np.float32)
    actions = source["actions"].astype(np.float32)
    split_labels = source["split_labels"].astype(str)
    motion_labels = source["motion_labels"].astype(str)
    projection = source["projection"].astype(np.float32)
    projection_inverse = source["projection_inverse"].astype(np.float32)

    joint_names = list(motion_fk.OFFICIAL_CSV_JOINT_NAMES)
    body_names = list(motion_fk.G1_TRACKING_BODY_NAMES)
    joint_index, joint_sign, joint_rows = mirror_index_and_sign(joint_names)
    body_index, body_rows = body_mirror_index(body_names)

    mirrored_states = mirror_states(states, body_index)
    mirrored_actions = mirror_actions(actions, joint_index, joint_sign)
    double_states = mirror_states(mirrored_states, body_index)
    double_actions = mirror_actions(mirrored_actions, joint_index, joint_sign)
    mirrored_projected = np.einsum("pd,ntd->ntp", projection, mirrored_states).astype(np.float32)

    augmented_states = np.concatenate([states, mirrored_states], axis=0).astype(np.float32)
    augmented_actions = np.concatenate([actions, mirrored_actions], axis=0).astype(np.float32)
    augmented_projected = np.einsum("pd,ntd->ntp", projection, augmented_states).astype(np.float32)
    augmented_split_labels = np.concatenate([split_labels, split_labels])
    augmented_motion_labels = np.concatenate([motion_labels, np.char.add(motion_labels, "::mirror")])

    tsv_path = OUT / "level_c_lafan1_paper_arch_symmetry_dataset_splits.tsv"
    npz_path = OUT / "lafan1_paper_arch_symmetry_augmented_dataset.npz"
    json_path = OUT / "lafan1_paper_arch_symmetry_dataset_audit.json"
    rows = split_rows(split_labels, int(states.shape[1]))
    np.savez_compressed(
        npz_path,
        augmented_states=augmented_states,
        augmented_projected_states=augmented_projected,
        augmented_actions=augmented_actions,
        augmented_split_labels=augmented_split_labels,
        augmented_motion_labels=augmented_motion_labels,
        mirrored_states=mirrored_states,
        mirrored_projected_states=mirrored_projected,
        mirrored_actions=mirrored_actions,
        source_states=states,
        source_actions=actions,
        source_split_labels=split_labels,
        source_motion_labels=motion_labels,
        projection=projection,
        projection_inverse=projection_inverse,
    )
    write_tsv(tsv_path, rows)

    split_counts_doubled = all(
        int(np.sum(augmented_split_labels == split)) == int(np.sum(split_labels == split)) * 2
        for split in ["train", "validation", "test"]
    )
    projected_mirror_recomputed = np.einsum("pd,ntd->ntp", projection, mirrored_states).astype(np.float32)
    double_state_max_abs = float(np.max(np.abs(double_states - states)))
    double_action_max_abs = float(np.max(np.abs(double_actions - actions)))
    projected_mirror_max_abs = float(np.max(np.abs(projected_mirror_recomputed - mirrored_projected)))
    source_projected_max_abs = float(
        np.max(np.abs(np.einsum("pd,ntd->ntp", projection, states).astype(np.float32) - source["projected_states"]))
    )
    checks = {
        "source_training_status_ok": source_summary["status"] == "ok",
        "symmetry_mapping_status_ok": symmetry_summary["status"] == "ok",
        "source_window_count_matches_training": int(states.shape[0])
        == int(source_summary["metrics"]["window_count"]),
        "source_state_shape_matches_paper_arch": list(states.shape[1:]) == [21, 99],
        "source_action_shape_matches_controller": list(actions.shape[1:]) == [21, 29],
        "projected_state_shape_matches_emphasis_projection": list(augmented_projected.shape[1:]) == [21, 207],
        "all_joint_mirror_rows_covered": all(row["covered"] for row in joint_rows),
        "all_body_mirror_rows_covered": all(row["covered"] for row in body_rows),
        "augmented_window_count_doubles_source": int(augmented_states.shape[0]) == int(states.shape[0]) * 2,
        "split_counts_doubled": split_counts_doubled,
        "double_mirror_state_within_tolerance": double_state_max_abs < 2e-5,
        "double_mirror_action_exact_float32": double_action_max_abs < 1e-7,
        "mirrored_projection_matches_recomputed": projected_mirror_max_abs < 1e-7,
        "source_projection_matches_source_dataset": source_projected_max_abs < 1e-5,
        "all_augmented_arrays_finite": bool(
            np.isfinite(augmented_states).all()
            and np.isfinite(augmented_actions).all()
            and np.isfinite(augmented_projected).all()
        ),
        "npz_written": npz_path.is_file(),
        "tsv_written": tsv_path.is_file(),
        "public_lafan1_dataset_boundary_recorded": True,
        "does_not_claim_official_sign_table": True,
        "does_not_claim_retrained_checkpoint": True,
        "goal_remains_incomplete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "public_lafan1_paper_arch_symmetry_augmented_dataset_audit",
        "scope": (
            "Applies the audited candidate Unitree G1 sagittal mirror mapping to the full public LAFAN1 "
            "paper-architecture training windows. The output is a trainable augmented dataset artifact, not a "
            "new VAE/diffusion checkpoint."
        ),
        "source_artifacts": {
            "source_training_json": str(SOURCE_JSON),
            "source_training_npz": str(SOURCE_NPZ),
            "source_training_npz_sha256": sha256_file(SOURCE_NPZ),
            "symmetry_mapping_json": str(SYMMETRY_JSON),
        },
        "settings": {
            "history": 4,
            "horizon": 16,
            "sequence_length": int(states.shape[1]),
            "state_dim": int(states.shape[2]),
            "projected_state_dim": int(augmented_projected.shape[2]),
            "action_dim": int(actions.shape[2]),
            "body_names": body_names,
            "joint_names": joint_names,
            "root_vector_mirror_rule": {"polar": [1.0, -1.0, 1.0], "axial": [-1.0, 1.0, -1.0]},
            "rot6d_rule": "decode first two columns, apply M @ R @ M with M=diag(1,-1,1), re-encode first two columns",
        },
        "metrics": {
            "source_window_count": int(states.shape[0]),
            "mirror_window_count": int(mirrored_states.shape[0]),
            "augmented_window_count": int(augmented_states.shape[0]),
            "source_token_count": int(states.shape[0] * states.shape[1]),
            "augmented_token_count": int(augmented_states.shape[0] * augmented_states.shape[1]),
            "public_lafan1_motion_count": int(len(np.unique(motion_labels))),
            "augmented_motion_label_count": int(len(np.unique(augmented_motion_labels))),
            "source_state_abs_max": float(np.max(np.abs(states))),
            "mirrored_state_abs_max": float(np.max(np.abs(mirrored_states))),
            "augmented_action_abs_max": float(np.max(np.abs(augmented_actions))),
            "double_mirror_state_max_abs_error": double_state_max_abs,
            "double_mirror_action_max_abs_error": double_action_max_abs,
            "mirrored_projection_recompute_max_abs_error": projected_mirror_max_abs,
            "source_projection_recompute_max_abs_error": source_projected_max_abs,
        },
        "checks": checks,
        "split_rows": rows,
        "joint_mirror_rows": joint_rows,
        "body_mirror_rows": body_rows,
        "interpretation": {
            "paper_level_status": "partial",
            "goal_complete": False,
            "why_not_complete": (
                "This produces a full public-data symmetry-augmented dataset, but the official paper teacher "
                "rollout dataset, DAgger/VAE rollout states, official sagittal sign table, and closed-loop robot "
                "success metrics are still unavailable."
            ),
        },
        "outputs": {
            "json": str(json_path),
            "tsv": str(tsv_path),
            "npz": str(npz_path),
        },
    }
    write_json(json_path, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "source_windows": summary["metrics"]["source_window_count"],
                "augmented_windows": summary["metrics"]["augmented_window_count"],
                "json": str(json_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
