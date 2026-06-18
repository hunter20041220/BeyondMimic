#!/usr/bin/env python3
"""Debug-only receding-horizon rollout smoke using exported tiny-VAE latents.

This consumes the nonzero debug latents and decoded actions exported by
``level_c_vae_debug_overfit_latent_artifact.py``. It rolls the current action
index across each 21-step state-latent window to validate the inference-facing
contract: the current latent/proprioception token produces the current action.
The source data are synthetic teacher projections from local fixtures, not true
DAgger rollouts or a trained BeyondMimic VAE checkpoint.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SOURCE_NPZ = ROOT / "reproduction/data/level_c_vae_debug_latents/debug_tiny_vae_state_latent_windows.npz"
SOURCE_JSON = (
    ROOT
    / "res/level_c/vae_debug_overfit_latent_artifact/level_c_vae_debug_overfit_latent_artifact.json"
)
OUT = ROOT / "res/level_c/vae_receding_horizon_rollout_smoke"
HISTORY = 4
CONTROL_FREQUENCY_HZ = 25.0


def sample_keys(npz: np.lib.npyio.NpzFile) -> list[str]:
    suffix = "_states"
    return sorted(key[: -len(suffix)] for key in npz.files if key.endswith(suffix))


def action_mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(pred - target)))


def mean_action_delta(actions: np.ndarray) -> float:
    if actions.shape[0] < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(actions, axis=0), axis=1)))


def source_motion(sample_id: str) -> str:
    marker = "_state_fixture_window_"
    if marker not in sample_id:
        return sample_id
    return sample_id.split(marker)[0]


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "source_motion",
        "sequence_length",
        "state_dim",
        "latent_dim",
        "action_dim",
        "current_index",
        "current_action_mse",
        "full_window_action_mse",
        "current_action_max_abs_error",
        "current_action_norm",
        "next_latent_action_delta",
        "mean_action_delta",
        "latent_norm_mean",
        "decoded_action_finite",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    with np.load(SOURCE_NPZ) as data:
        rows: list[dict[str, Any]] = []
        current_actions = []
        teacher_current_actions = []
        sample_ids = sample_keys(data)
        for sample_id in sample_ids:
            states = data[f"{sample_id}_states"].astype(np.float64)
            latents = data[f"{sample_id}_latents"].astype(np.float64)
            teacher_action = data[f"{sample_id}_teacher_action"].astype(np.float64)
            decoded_action = data[f"{sample_id}_decoded_action"].astype(np.float64)
            if states.shape[:1] != latents.shape[:1] or teacher_action.shape != decoded_action.shape:
                raise ValueError(f"{sample_id}: inconsistent window shapes")
            current_idx = min(HISTORY, decoded_action.shape[0] - 1)
            next_idx = min(current_idx + 1, decoded_action.shape[0] - 1)
            current_error = decoded_action[current_idx] - teacher_action[current_idx]
            current_actions.append(decoded_action[current_idx])
            teacher_current_actions.append(teacher_action[current_idx])
            rows.append(
                {
                    "sample_id": sample_id,
                    "source_motion": source_motion(sample_id),
                    "sequence_length": int(states.shape[0]),
                    "state_dim": int(states.shape[1]),
                    "latent_dim": int(latents.shape[1]),
                    "action_dim": int(decoded_action.shape[1]),
                    "current_index": int(current_idx),
                    "current_action_mse": action_mse(decoded_action[current_idx], teacher_action[current_idx]),
                    "full_window_action_mse": action_mse(decoded_action, teacher_action),
                    "current_action_max_abs_error": float(np.max(np.abs(current_error))),
                    "current_action_norm": float(np.linalg.norm(decoded_action[current_idx])),
                    "next_latent_action_delta": float(
                        np.linalg.norm(decoded_action[current_idx] - decoded_action[next_idx])
                    ),
                    "mean_action_delta": mean_action_delta(decoded_action),
                    "latent_norm_mean": float(np.mean(np.linalg.norm(latents, axis=1))),
                    "decoded_action_finite": bool(np.all(np.isfinite(decoded_action))),
                }
            )

    current_actions_arr = np.stack(current_actions)
    teacher_current_arr = np.stack(teacher_current_actions)
    current_mse_values = np.asarray([row["current_action_mse"] for row in rows], dtype=np.float64)
    full_mse_values = np.asarray([row["full_window_action_mse"] for row in rows], dtype=np.float64)
    next_delta_values = np.asarray([row["next_latent_action_delta"] for row in rows], dtype=np.float64)
    action_delta_values = np.asarray([row["mean_action_delta"] for row in rows], dtype=np.float64)
    split_counts = source_summary["metrics"]["split_counts"]
    motion_counts = Counter(row["source_motion"] for row in rows)

    json_path = OUT / "level_c_vae_receding_horizon_rollout_smoke.json"
    tsv_path = OUT / "level_c_vae_receding_horizon_rollout_smoke.tsv"
    npz_path = OUT / "level_c_vae_receding_horizon_rollout_smoke.npz"
    np.savez_compressed(
        npz_path,
        current_decoded_actions=current_actions_arr.astype(np.float32),
        current_teacher_actions=teacher_current_arr.astype(np.float32),
        current_action_mse=current_mse_values.astype(np.float32),
        sample_id=np.asarray([row["sample_id"] for row in rows]),
        source_motion=np.asarray([row["source_motion"] for row in rows]),
    )
    write_tsv(tsv_path, rows)

    checks = {
        "source_npz_exists": SOURCE_NPZ.is_file() and SOURCE_NPZ.stat().st_size > 0,
        "source_json_status_ok": source_summary.get("status") == "ok",
        "row_count_84": len(rows) == 84,
        "motion_count_3": len(motion_counts) == 3,
        "split_counts_match_source": split_counts == {"test": 28, "train": 28, "validation": 28},
        "sequence_length_21": all(row["sequence_length"] == 21 for row in rows),
        "state_dim_99": all(row["state_dim"] == 99 for row in rows),
        "latent_dim_32": all(row["latent_dim"] == 32 for row in rows),
        "action_dim_29": all(row["action_dim"] == 29 for row in rows),
        "current_index_is_history": all(row["current_index"] == HISTORY for row in rows),
        "all_decoded_actions_finite": all(row["decoded_action_finite"] for row in rows),
        "current_action_mse_below_0_01": float(np.mean(current_mse_values)) < 0.01,
        "full_window_mse_below_0_01": float(np.mean(full_mse_values)) < 0.01,
        "next_latent_changes_action": float(np.mean(next_delta_values)) > 0.0,
        "rollout_action_delta_finite": bool(np.all(np.isfinite(action_delta_values))),
        "npz_written": npz_path.is_file() and npz_path.stat().st_size > 0,
        "does_not_claim_true_dagger_rollout": True,
        "does_not_claim_trained_paper_checkpoint": True,
        "does_not_claim_closed_loop_simulation": True,
        "does_not_claim_goal_complete": True,
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "debug_only_vae_receding_horizon_rollout_smoke",
        "scope": (
            "roll current-action inference over exported tiny-VAE debug state-latent windows using decoded actions "
            "from the source artifact"
        ),
        "paper_evidence": {
            "decoder_equation": str(ROOT / "reproduction/paper/source/tex/method.tex:157-162"),
            "current_latent_action": str(ROOT / "reproduction/paper/source/tex/method.tex:197-206"),
            "control_frequency_and_cpu_decoder": str(ROOT / "reproduction/paper/source/root.tex:589-593"),
            "goal_vae_dagger": str(ROOT / "goal.md:1148-1190,1431-1467"),
        },
        "source_artifacts": {
            "source_json": str(SOURCE_JSON),
            "source_npz": str(SOURCE_NPZ),
            "source_experiment_type": source_summary["experiment_type"],
            "source_metrics": source_summary["metrics"],
        },
        "settings": {
            "history": HISTORY,
            "current_index": HISTORY,
            "control_frequency_hz": CONTROL_FREQUENCY_HZ,
            "control_dt_seconds": 1.0 / CONTROL_FREQUENCY_HZ,
        },
        "metrics": {
            "row_count": len(rows),
            "motion_counts": dict(sorted(motion_counts.items())),
            "split_counts": split_counts,
            "current_action_mse_mean": float(np.mean(current_mse_values)),
            "current_action_mse_max": float(np.max(current_mse_values)),
            "full_window_action_mse_mean": float(np.mean(full_mse_values)),
            "full_window_action_mse_max": float(np.max(full_mse_values)),
            "current_action_max_abs_error_max": float(
                np.max([row["current_action_max_abs_error"] for row in rows])
            ),
            "next_latent_action_delta_mean": float(np.mean(next_delta_values)),
            "mean_action_delta_mean": float(np.mean(action_delta_values)),
            "current_action_norm_mean": float(
                np.mean([row["current_action_norm"] for row in rows])
            ),
        },
        "checks": checks,
        "rows": rows,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This validates the local tiny-VAE action decoding contract over exported debug windows only. It "
                "does not create true DAgger rollouts, a trained paper VAE checkpoint, closed-loop Isaac evaluation, "
                "tracking-policy survival metrics, or required paper videos."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "rows": len(rows), "json": str(json_path)}, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
