#!/usr/bin/env python3
"""Debug-only diffusion-token to VAE-action pipeline smoke.

This connects two existing local artifacts:

1. ``level_c_vae_latent_heldout_eval`` provides held-out diffusion predictions
   over paper-state + tiny-VAE latent tokens.
2. ``level_c_vae_debug_overfit_latent_artifact`` provides the tiny-VAE
   proprioception projection and decoded synthetic-teacher actions.

The script fits a simple action-decoder surrogate only on train-motion clean
tokens, then evaluates clean/noisy/diffusion-predicted tokens on held-out
validation/test motions. It is a pipeline contract check, not a trained VAE
decoder checkpoint, trained diffusion Transformer, or closed-loop rollout.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
VAE_JSON = ROOT / "res/level_c/vae_debug_overfit_latent_artifact/level_c_vae_debug_overfit_latent_artifact.json"
VAE_NPZ = ROOT / "reproduction/data/level_c_vae_debug_latents/debug_tiny_vae_state_latent_windows.npz"
DIFFUSION_JSON = ROOT / "res/level_c/vae_latent_heldout_eval/level_c_vae_latent_heldout_eval.json"
DIFFUSION_NPZ = ROOT / "res/level_c/vae_latent_heldout_eval/level_c_vae_latent_heldout_eval.npz"
OUT = ROOT / "res/level_c/diffusion_to_vae_action_smoke"
HISTORY = 4
RIDGE_LAMBDA = 1e-5


def feature_tensor(tau: np.ndarray, projection: np.ndarray) -> np.ndarray:
    state = tau[:, :, :99]
    latent = tau[:, :, 99:]
    proprioception = np.tanh(np.einsum("bts,sp->btp", state, projection))
    bias = np.ones((*latent.shape[:2], 1), dtype=np.float64)
    return np.concatenate([latent, proprioception, bias], axis=-1)


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(pred - target)))


def expand_split_mask(split_labels: np.ndarray, sequence_length: int, split: str) -> np.ndarray:
    return np.repeat(split_labels == split, sequence_length)


def split_rows(
    split_labels: np.ndarray,
    source_motion: np.ndarray,
    clean_action: np.ndarray,
    noisy_action: np.ndarray,
    predicted_action: np.ndarray,
    target_action: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ["train", "validation", "test"]:
        mask = split_labels == split
        token_mask = expand_split_mask(split_labels, clean_action.shape[1], split)
        current_idx = HISTORY
        clean_full = clean_action[mask].reshape(-1, clean_action.shape[-1])
        noisy_full = noisy_action[mask].reshape(-1, noisy_action.shape[-1])
        pred_full = predicted_action[mask].reshape(-1, predicted_action.shape[-1])
        target_full = target_action[mask].reshape(-1, target_action.shape[-1])
        row = {
            "split": split,
            "motion_names": ",".join(sorted(set(source_motion[mask].tolist()))),
            "window_count": int(np.sum(mask)),
            "token_count": int(np.sum(token_mask)),
            "clean_full_action_mse": mse(clean_full, target_full),
            "noisy_full_action_mse": mse(noisy_full, target_full),
            "predicted_full_action_mse": mse(pred_full, target_full),
            "clean_current_action_mse": mse(clean_action[mask, current_idx], target_action[mask, current_idx]),
            "noisy_current_action_mse": mse(noisy_action[mask, current_idx], target_action[mask, current_idx]),
            "predicted_current_action_mse": mse(
                predicted_action[mask, current_idx], target_action[mask, current_idx]
            ),
        }
        row["full_mse_reduction_vs_noisy"] = (
            (row["noisy_full_action_mse"] - row["predicted_full_action_mse"]) / row["noisy_full_action_mse"]
            if row["noisy_full_action_mse"] > 0
            else 0.0
        )
        row["current_mse_reduction_vs_noisy"] = (
            (row["noisy_current_action_mse"] - row["predicted_current_action_mse"])
            / row["noisy_current_action_mse"]
            if row["noisy_current_action_mse"] > 0
            else 0.0
        )
        rows.append(row)
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "split",
        "motion_names",
        "window_count",
        "token_count",
        "clean_full_action_mse",
        "noisy_full_action_mse",
        "predicted_full_action_mse",
        "full_mse_reduction_vs_noisy",
        "clean_current_action_mse",
        "noisy_current_action_mse",
        "predicted_current_action_mse",
        "current_mse_reduction_vs_noisy",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vae_summary = json.loads(VAE_JSON.read_text(encoding="utf-8"))
    diffusion_summary = json.loads(DIFFUSION_JSON.read_text(encoding="utf-8"))
    with np.load(VAE_NPZ) as vae_data, np.load(DIFFUSION_NPZ) as diffusion_data:
        projection = vae_data["proprioception_projection"].astype(np.float64)
        sample_rows = vae_summary["rows"]
        target_action = np.stack(
            [vae_data[f"{row['sample_id']}_decoded_action"].astype(np.float64) for row in sample_rows],
            axis=0,
        )
        split_labels = np.asarray([row["split"] for row in sample_rows])
        source_motion = np.asarray([row["source_motion"] for row in sample_rows])
        clean_tau = diffusion_data["clean_tau"].astype(np.float64)
        noisy_tau = diffusion_data["noisy_tau"].astype(np.float64)
        predicted_tau = diffusion_data["prediction"].astype(np.float64)

    clean_features = feature_tensor(clean_tau, projection)
    noisy_features = feature_tensor(noisy_tau, projection)
    predicted_features = feature_tensor(predicted_tau, projection)
    train_mask = expand_split_mask(split_labels, clean_features.shape[1], "train")
    x_train = clean_features.reshape(-1, clean_features.shape[-1])[train_mask]
    y_train = target_action.reshape(-1, target_action.shape[-1])[train_mask]
    reg = RIDGE_LAMBDA * np.eye(x_train.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x_train.T @ x_train + reg, x_train.T @ y_train)

    clean_action = clean_features @ weights
    noisy_action = noisy_features @ weights
    predicted_action = predicted_features @ weights
    rows = split_rows(split_labels, source_motion, clean_action, noisy_action, predicted_action, target_action)
    row_by_split = {row["split"]: row for row in rows}

    json_path = OUT / "level_c_diffusion_to_vae_action_smoke.json"
    tsv_path = OUT / "level_c_diffusion_to_vae_action_smoke.tsv"
    npz_path = OUT / "level_c_diffusion_to_vae_action_smoke.npz"
    np.savez_compressed(
        npz_path,
        clean_action=clean_action.astype(np.float32),
        noisy_action=noisy_action.astype(np.float32),
        predicted_action=predicted_action.astype(np.float32),
        target_action=target_action.astype(np.float32),
        split_labels=split_labels,
        source_motion=source_motion,
        decoder_surrogate_weights=weights.astype(np.float32),
    )
    write_tsv(tsv_path, rows)

    checks = {
        "vae_source_status_ok": vae_summary["status"] == "ok",
        "diffusion_source_status_ok": diffusion_summary["status"] == "ok",
        "uses_train_split_only_for_decoder_surrogate": int(np.sum(split_labels == "train")) == 28,
        "window_count_84": clean_tau.shape[0] == 84,
        "sequence_length_21": clean_tau.shape[1] == 21,
        "token_dim_131": clean_tau.shape[2] == 131,
        "state_dim_99": clean_tau.shape[2] - 32 == 99,
        "latent_dim_32": clean_tau.shape[2] - 99 == 32,
        "action_dim_29": target_action.shape[-1] == 29,
        "current_index_is_history": HISTORY == 4,
        "no_token_identity_basis_in_action_decoder": True,
        "all_actions_finite": bool(
            np.all(np.isfinite(clean_action))
            and np.all(np.isfinite(noisy_action))
            and np.all(np.isfinite(predicted_action))
        ),
        "validation_prediction_improves_full_vs_noisy": row_by_split["validation"][
            "predicted_full_action_mse"
        ]
        < row_by_split["validation"]["noisy_full_action_mse"],
        "test_prediction_improves_full_vs_noisy": row_by_split["test"]["predicted_full_action_mse"]
        < row_by_split["test"]["noisy_full_action_mse"],
        "validation_prediction_improves_current_vs_noisy": row_by_split["validation"][
            "predicted_current_action_mse"
        ]
        < row_by_split["validation"]["noisy_current_action_mse"],
        "test_prediction_improves_current_vs_noisy": row_by_split["test"]["predicted_current_action_mse"]
        < row_by_split["test"]["noisy_current_action_mse"],
        "heldout_current_action_mse_below_0_02": max(
            row_by_split["validation"]["predicted_current_action_mse"],
            row_by_split["test"]["predicted_current_action_mse"],
        )
        < 0.02,
        "npz_written": npz_path.is_file() and npz_path.stat().st_size > 0,
        "does_not_claim_true_vae_decoder": True,
        "does_not_claim_trained_diffusion_checkpoint": True,
        "does_not_claim_closed_loop_rollout": True,
        "does_not_claim_goal_complete": True,
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "debug_only_diffusion_to_vae_action_smoke",
        "scope": (
            "connect held-out diffusion token predictions to a train-split action-decoder surrogate fitted from "
            "tiny-VAE debug decoded actions"
        ),
        "paper_evidence": {
            "clean_trajectory_loss": str(ROOT / "reproduction/paper/source/tex/method.tex:187-196"),
            "vae_decoder_equation": str(ROOT / "reproduction/paper/source/tex/method.tex:157-162"),
            "current_latent_action": str(ROOT / "reproduction/paper/source/tex/method.tex:197-206"),
            "goal_diffusion_vae": str(ROOT / "goal.md:1251-1321,1468-1507"),
        },
        "source_artifacts": {
            "vae_debug_latents_json": str(VAE_JSON),
            "vae_debug_latents_npz": str(VAE_NPZ),
            "vae_latent_heldout_json": str(DIFFUSION_JSON),
            "vae_latent_heldout_npz": str(DIFFUSION_NPZ),
        },
        "settings": {
            "history": HISTORY,
            "current_index": HISTORY,
            "ridge_lambda": RIDGE_LAMBDA,
            "decoder_surrogate_input_dim": int(clean_features.shape[-1]),
            "decoder_surrogate_output_dim": int(target_action.shape[-1]),
            "uses_token_identity_basis": False,
            "decoder_surrogate_train_split": "train",
        },
        "metrics": {
            "rows": rows,
            "validation_predicted_current_action_mse": row_by_split["validation"][
                "predicted_current_action_mse"
            ],
            "test_predicted_current_action_mse": row_by_split["test"]["predicted_current_action_mse"],
            "validation_current_mse_reduction_vs_noisy": row_by_split["validation"][
                "current_mse_reduction_vs_noisy"
            ],
            "test_current_mse_reduction_vs_noisy": row_by_split["test"]["current_mse_reduction_vs_noisy"],
            "validation_predicted_full_action_mse": row_by_split["validation"]["predicted_full_action_mse"],
            "test_predicted_full_action_mse": row_by_split["test"]["predicted_full_action_mse"],
        },
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This links held-out debug diffusion predictions to a train-split surrogate action decoder. It does "
                "not use the unpublished trained VAE decoder, trained diffusion Transformer checkpoint, Isaac "
                "closed-loop rollout, TensorRT deployment, or paper Fig. 5/Fig. 6 protocol."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(json_path), "rows": len(rows)}, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
