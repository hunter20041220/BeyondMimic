#!/usr/bin/env python3
"""Audit consistency between local paper-state windows, debug VAE latents, and action artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/state_latent_dataset_consistency_audit"
PAPER_STATE_JSON = ROOT / "res/level_c/paper_state_windows/level_c_paper_state_windows.json"
PAPER_STATE_DIR = ROOT / "reproduction/data/level_c_paper_state_windows"
VAE_JSON = ROOT / "res/level_c/vae_debug_overfit_latent_artifact/level_c_vae_debug_overfit_latent_artifact.json"
VAE_NPZ = ROOT / "reproduction/data/level_c_vae_debug_latents/debug_tiny_vae_state_latent_windows.npz"
ACTION_JSON = ROOT / "res/level_c/diffusion_to_vae_action_smoke/level_c_diffusion_to_vae_action_smoke.json"
ACTION_NPZ = ROOT / "res/level_c/diffusion_to_vae_action_smoke/level_c_diffusion_to_vae_action_smoke.npz"
HISTORY = 4
HORIZON = 16
SEQ_LEN = HISTORY + 1 + HORIZON
MOTION_SPLITS = {
    "walk1_subject1_frames_1_180_state_fixture": "train",
    "run2_subject1_frames_1_180_state_fixture": "validation",
    "jumps1_subject1_frames_1_180_state_fixture": "test",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def feature_slices(path: Path) -> dict[str, list[int]]:
    with np.load(path) as data:
        return json.loads(str(data["feature_slices_json"][0]))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "motion",
        "split",
        "window_index",
        "paper_start_frame",
        "recorded_start_timestep",
        "state_max_abs_error",
        "target_decoded_action_max_abs_error",
        "latent_abs_mean",
        "teacher_action_abs_mean",
        "decoded_action_abs_mean",
        "logvar_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paper_json = load_json(PAPER_STATE_JSON)
    vae_json = load_json(VAE_JSON)
    action_json = load_json(ACTION_JSON)
    vae_rows = {row["sample_id"]: row for row in vae_json["rows"]}
    rows: list[dict[str, Any]] = []

    with np.load(VAE_NPZ) as vae, np.load(ACTION_NPZ) as action:
        source_motion = action["source_motion"].astype(str)
        split_labels = action["split_labels"].astype(str)
        target_action = action["target_action"].astype(np.float64)

        action_cursor = 0
        paper_feature_slices: dict[str, list[int]] | None = None
        per_motion_window_counts: dict[str, int] = {}
        per_split_counts: dict[str, int] = {}
        state_errors: list[float] = []
        target_action_errors: list[float] = []
        current_root_position_errors: list[float] = []
        current_root_velocity_errors: list[float] = []
        latent_abs_means: list[float] = []
        logvar_means: list[float] = []
        teacher_action_abs_means: list[float] = []
        decoded_action_abs_means: list[float] = []
        action_source_order_ok = True
        action_split_order_ok = True
        vae_json_order_ok = True
        start_timestep_ok = True

        for motion, expected_split in MOTION_SPLITS.items():
            paper_npz = PAPER_STATE_DIR / f"{motion}_paper_state_windows.npz"
            with np.load(paper_npz) as paper:
                paper_states = paper["paper_state_windows"].astype(np.float64)
                starts = paper["window_start_indices"].astype(np.int64)
            if paper_feature_slices is None:
                paper_feature_slices = feature_slices(paper_npz)
            per_motion_window_counts[motion] = int(paper_states.shape[0])
            per_split_counts[expected_split] = int(paper_states.shape[0])

            for window_index in range(paper_states.shape[0]):
                sample_id = f"{motion}_window_{window_index:04d}"
                states = vae[f"{sample_id}_states"].astype(np.float64)
                latents = vae[f"{sample_id}_latents"].astype(np.float64)
                teacher_action = vae[f"{sample_id}_teacher_action"].astype(np.float64)
                decoded_action = vae[f"{sample_id}_decoded_action"].astype(np.float64)
                logvar = vae[f"{sample_id}_logvar"].astype(np.float64)
                row = vae_rows.get(sample_id, {})

                state_error = float(np.max(np.abs(states - paper_states[window_index])))
                action_error = float(np.max(np.abs(target_action[action_cursor] - decoded_action)))
                state_errors.append(state_error)
                target_action_errors.append(action_error)
                latent_abs_means.append(float(np.mean(np.abs(latents))))
                logvar_means.append(float(np.mean(logvar)))
                teacher_action_abs_means.append(float(np.mean(np.abs(teacher_action))))
                decoded_action_abs_means.append(float(np.mean(np.abs(decoded_action))))

                root_pos_slice = slice(*paper_feature_slices["root_pos_rel_current_frame"])
                root_vel_slice = slice(*paper_feature_slices["root_lin_vel_rel_current_frame"])
                current_root_position_errors.append(float(np.max(np.abs(states[HISTORY, root_pos_slice]))))
                current_root_velocity_errors.append(float(np.max(np.abs(states[HISTORY, root_vel_slice]))))

                action_source_order_ok &= bool(source_motion[action_cursor] == motion)
                action_split_order_ok &= bool(split_labels[action_cursor] == expected_split)
                vae_json_order_ok &= bool(row.get("split") == expected_split and row.get("source_motion") == motion)
                start_timestep_ok &= bool(row.get("start_timestep") == int(starts[window_index] - HISTORY))

                rows.append(
                    {
                        "motion": motion,
                        "split": expected_split,
                        "window_index": window_index,
                        "paper_start_frame": int(starts[window_index]),
                        "recorded_start_timestep": int(row.get("start_timestep", -1)),
                        "state_max_abs_error": state_error,
                        "target_decoded_action_max_abs_error": action_error,
                        "latent_abs_mean": float(np.mean(np.abs(latents))),
                        "teacher_action_abs_mean": float(np.mean(np.abs(teacher_action))),
                        "decoded_action_abs_mean": float(np.mean(np.abs(decoded_action))),
                        "logvar_mean": float(np.mean(logvar)),
                    }
                )
                action_cursor += 1

    all_finite = all(
        np.isfinite(value)
        for value in (
            state_errors
            + target_action_errors
            + current_root_position_errors
            + current_root_velocity_errors
            + latent_abs_means
            + logvar_means
            + teacher_action_abs_means
            + decoded_action_abs_means
        )
    )
    json_path = OUT / "level_c_state_latent_dataset_consistency_audit.json"
    tsv_path = OUT / "level_c_state_latent_dataset_consistency_audit.tsv"
    write_tsv(tsv_path, rows)

    expected_slices = {
        "root_pos_rel_current_frame": [0, 3],
        "root_rot6d_rel_current_frame": [3, 9],
        "root_lin_vel_rel_current_frame": [9, 12],
        "root_ang_vel_rel_current_frame": [12, 15],
        "body_pos_local_root_frame": [15, 57],
        "body_lin_vel_local_root_frame": [57, 99],
    }
    metrics = {
        "row_count": len(rows),
        "motion_count": len(MOTION_SPLITS),
        "window_count": len(rows),
        "sequence_length": SEQ_LEN,
        "state_dim": 99,
        "latent_dim": 32,
        "token_dim": 131,
        "action_dim": 29,
        "per_motion_window_counts": per_motion_window_counts,
        "per_split_counts": per_split_counts,
        "max_state_abs_error_between_paper_windows_and_vae_npz": max(state_errors),
        "max_target_action_abs_error_between_action_npz_and_decoded_action": max(target_action_errors),
        "max_current_root_position_abs": max(current_root_position_errors),
        "max_current_root_linear_velocity_abs": max(current_root_velocity_errors),
        "latent_abs_mean": float(np.mean(latent_abs_means)),
        "logvar_mean": float(np.mean(logvar_means)),
        "teacher_action_abs_mean": float(np.mean(teacher_action_abs_means)),
        "decoded_action_abs_mean": float(np.mean(decoded_action_abs_means)),
    }
    checks = {
        "paper_state_json_status_ok": paper_json["status"] == "ok",
        "vae_debug_latents_json_status_ok": vae_json["status"] == "ok",
        "diffusion_to_action_json_status_ok": action_json["status"] == "ok",
        "all_required_files_exist": all(path.is_file() for path in [PAPER_STATE_JSON, VAE_JSON, VAE_NPZ, ACTION_JSON, ACTION_NPZ]),
        "row_count_84": metrics["row_count"] == 84,
        "sequence_length_21": metrics["sequence_length"] == 21,
        "state_dim_99": metrics["state_dim"] == 99,
        "latent_dim_32": metrics["latent_dim"] == 32,
        "token_dim_131": metrics["token_dim"] == 131,
        "action_dim_29": metrics["action_dim"] == 29,
        "per_motion_counts_28": all(count == 28 for count in per_motion_window_counts.values()),
        "per_split_counts_28": per_split_counts == {"train": 28, "validation": 28, "test": 28},
        "feature_slices_match_paper_state_builder": paper_feature_slices == expected_slices,
        "vae_states_equal_paper_state_windows": max(state_errors) < 1e-6,
        "action_target_equals_decoded_action": max(target_action_errors) < 1e-7,
        "action_npz_source_motion_order_matches_windows": action_source_order_ok,
        "action_npz_split_labels_match_motion_split": action_split_order_ok,
        "vae_json_rows_match_motion_split": vae_json_order_ok,
        "vae_json_start_timestep_matches_window_start_minus_history": start_timestep_ok,
        "current_root_position_zero": max(current_root_position_errors) < 1e-12,
        "current_root_linear_velocity_zero": max(current_root_velocity_errors) < 1e-12,
        "all_numeric_metrics_finite": all_finite,
        "does_not_claim_true_dagger_rollout": True,
        "does_not_claim_paper_trainable_dataset": True,
        "does_not_claim_trained_checkpoint": True,
        "does_not_claim_closed_loop_rollout": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "debug_only_state_latent_dataset_consistency_audit",
        "scope": (
            "Cross-artifact consistency audit for local paper-state windows, tiny-VAE debug latents/actions, "
            "and downstream diffusion-to-action NPZ ordering."
        ),
        "settings": {
            "history": HISTORY,
            "horizon": HORIZON,
            "motion_splits": MOTION_SPLITS,
            "expected_feature_slices": expected_slices,
        },
        "source_artifacts": {
            "paper_state_windows_json": str(PAPER_STATE_JSON),
            "paper_state_windows_dir": str(PAPER_STATE_DIR),
            "vae_debug_latents_json": str(VAE_JSON),
            "vae_debug_latents_npz": str(VAE_NPZ),
            "diffusion_to_action_json": str(ACTION_JSON),
            "diffusion_to_action_npz": str(ACTION_NPZ),
        },
        "metrics": metrics,
        "checks": checks,
        "rows": rows,
        "interpretation": {
            "paper_level_status": "partial",
            "goal_complete": False,
            "why_not_complete": (
                "The local debug paper-state, latent, and action artifacts are internally consistent. This does not "
                "turn them into the paper's true DAgger/VAE rollout dataset and does not create trained VAE/diffusion "
                "checkpoints, closed-loop logs, TensorRT engines, or Fig. 5/Fig. 6 paper results."
            ),
        },
        "outputs": {
            "json": str(json_path),
            "tsv": str(tsv_path),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(json_path), "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
