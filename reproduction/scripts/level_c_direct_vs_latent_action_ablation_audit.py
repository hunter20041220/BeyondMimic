#!/usr/bin/env python3
"""Debug-only direct-state vs state-latent action ablation.

The paper reports a direct state/action diffusion ablation for cartwheel success.
The local artifact set does not contain the trained checkpoints or rollout logs
needed to reproduce that result. This audit instead compares two local offline
interfaces on the same paper-state debug windows:

* direct branch: train-split ridge action decoder using only the 99-D state token
* latent branch: existing state+latent diffusion-to-action smoke artifact

It is a data-path ablation, not a closed-loop success-rate reproduction.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/direct_vs_latent_action_ablation_audit"
VAE_JSON = ROOT / "res/level_c/vae_debug_overfit_latent_artifact/level_c_vae_debug_overfit_latent_artifact.json"
VAE_NPZ = ROOT / "reproduction/data/level_c_vae_debug_latents/debug_tiny_vae_state_latent_windows.npz"
DIFFUSION_JSON = ROOT / "res/level_c/vae_latent_heldout_eval/level_c_vae_latent_heldout_eval.json"
DIFFUSION_NPZ = ROOT / "res/level_c/vae_latent_heldout_eval/level_c_vae_latent_heldout_eval.npz"
LATENT_ACTION_JSON = ROOT / "res/level_c/diffusion_to_vae_action_smoke/level_c_diffusion_to_vae_action_smoke.json"
LATENT_ACTION_NPZ = ROOT / "res/level_c/diffusion_to_vae_action_smoke/level_c_diffusion_to_vae_action_smoke.npz"
HISTORY = 4
RIDGE_LAMBDA = 1e-5


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def state_features(tau: np.ndarray) -> np.ndarray:
    state = tau[:, :, :99].astype(np.float64)
    bias = np.ones((*state.shape[:2], 1), dtype=np.float64)
    return np.concatenate([state, np.tanh(state), bias], axis=-1)


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.square(pred - target)))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "split",
        "motion_names",
        "window_count",
        "token_count",
        "direct_clean_full_action_mse",
        "direct_noisy_full_action_mse",
        "direct_predicted_full_action_mse",
        "latent_predicted_full_action_mse",
        "direct_clean_current_action_mse",
        "direct_noisy_current_action_mse",
        "direct_predicted_current_action_mse",
        "latent_predicted_current_action_mse",
        "latent_vs_direct_current_mse_ratio",
        "latent_vs_direct_full_mse_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    vae_summary = load_json(VAE_JSON)
    diffusion_summary = load_json(DIFFUSION_JSON)
    latent_action_summary = load_json(LATENT_ACTION_JSON)

    with np.load(VAE_NPZ) as vae, np.load(DIFFUSION_NPZ) as diffusion, np.load(LATENT_ACTION_NPZ) as latent_action:
        sample_rows = vae_summary["rows"]
        target_action = np.stack(
            [vae[f"{row['sample_id']}_decoded_action"].astype(np.float64) for row in sample_rows],
            axis=0,
        )
        split_labels = np.asarray([row["split"] for row in sample_rows])
        source_motion = np.asarray([row["source_motion"] for row in sample_rows])
        clean_tau = diffusion["clean_tau"].astype(np.float64)
        noisy_tau = diffusion["noisy_tau"].astype(np.float64)
        predicted_tau = diffusion["prediction"].astype(np.float64)
        latent_predicted_action = latent_action["predicted_action"].astype(np.float64)
        latent_target_action = latent_action["target_action"].astype(np.float64)
        latent_split_labels = latent_action["split_labels"].astype(str)
        latent_source_motion = latent_action["source_motion"].astype(str)

    clean_x = state_features(clean_tau)
    noisy_x = state_features(noisy_tau)
    pred_x = state_features(predicted_tau)
    train_mask = np.repeat(split_labels == "train", clean_x.shape[1])
    x_train = clean_x.reshape(-1, clean_x.shape[-1])[train_mask]
    y_train = target_action.reshape(-1, target_action.shape[-1])[train_mask]
    reg = RIDGE_LAMBDA * np.eye(x_train.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x_train.T @ x_train + reg, x_train.T @ y_train)

    direct_clean_action = clean_x @ weights
    direct_noisy_action = noisy_x @ weights
    direct_predicted_action = pred_x @ weights

    rows: list[dict[str, Any]] = []
    for split in ["train", "validation", "test"]:
        mask = split_labels == split
        current = HISTORY
        direct_clean_full = direct_clean_action[mask].reshape(-1, target_action.shape[-1])
        direct_noisy_full = direct_noisy_action[mask].reshape(-1, target_action.shape[-1])
        direct_pred_full = direct_predicted_action[mask].reshape(-1, target_action.shape[-1])
        latent_pred_full = latent_predicted_action[mask].reshape(-1, target_action.shape[-1])
        target_full = target_action[mask].reshape(-1, target_action.shape[-1])
        direct_pred_current = mse(direct_predicted_action[mask, current], target_action[mask, current])
        latent_pred_current = mse(latent_predicted_action[mask, current], target_action[mask, current])
        direct_pred_full_mse = mse(direct_pred_full, target_full)
        latent_pred_full_mse = mse(latent_pred_full, target_full)
        rows.append(
            {
                "split": split,
                "motion_names": ",".join(sorted(set(source_motion[mask].tolist()))),
                "window_count": int(np.sum(mask)),
                "token_count": int(np.sum(mask) * clean_tau.shape[1]),
                "direct_clean_full_action_mse": mse(direct_clean_full, target_full),
                "direct_noisy_full_action_mse": mse(direct_noisy_full, target_full),
                "direct_predicted_full_action_mse": direct_pred_full_mse,
                "latent_predicted_full_action_mse": latent_pred_full_mse,
                "direct_clean_current_action_mse": mse(direct_clean_action[mask, current], target_action[mask, current]),
                "direct_noisy_current_action_mse": mse(direct_noisy_action[mask, current], target_action[mask, current]),
                "direct_predicted_current_action_mse": direct_pred_current,
                "latent_predicted_current_action_mse": latent_pred_current,
                "latent_vs_direct_current_mse_ratio": (
                    latent_pred_current / direct_pred_current if direct_pred_current > 0 else 0.0
                ),
                "latent_vs_direct_full_mse_ratio": (
                    latent_pred_full_mse / direct_pred_full_mse if direct_pred_full_mse > 0 else 0.0
                ),
            }
        )
    row_by_split = {row["split"]: row for row in rows}

    json_path = OUT / "level_c_direct_vs_latent_action_ablation_audit.json"
    tsv_path = OUT / "level_c_direct_vs_latent_action_ablation_audit.tsv"
    npz_path = OUT / "level_c_direct_vs_latent_action_ablation_audit.npz"
    np.savez_compressed(
        npz_path,
        direct_clean_action=direct_clean_action.astype(np.float32),
        direct_noisy_action=direct_noisy_action.astype(np.float32),
        direct_predicted_action=direct_predicted_action.astype(np.float32),
        latent_predicted_action=latent_predicted_action.astype(np.float32),
        target_action=target_action.astype(np.float32),
        split_labels=split_labels,
        source_motion=source_motion,
        direct_state_decoder_weights=weights.astype(np.float32),
    )
    write_tsv(tsv_path, rows)

    heldout_direct_current = [
        row_by_split["validation"]["direct_predicted_current_action_mse"],
        row_by_split["test"]["direct_predicted_current_action_mse"],
    ]
    heldout_latent_current = [
        row_by_split["validation"]["latent_predicted_current_action_mse"],
        row_by_split["test"]["latent_predicted_current_action_mse"],
    ]
    checks = {
        "vae_source_status_ok": vae_summary["status"] == "ok",
        "diffusion_source_status_ok": diffusion_summary["status"] == "ok",
        "latent_action_source_status_ok": latent_action_summary["status"] == "ok",
        "latent_action_target_matches_vae_target": bool(np.max(np.abs(latent_target_action - target_action)) < 1e-7),
        "latent_action_order_matches_vae_rows": bool(
            np.array_equal(latent_split_labels, split_labels) and np.array_equal(latent_source_motion, source_motion)
        ),
        "uses_train_split_only_for_direct_decoder": int(np.sum(split_labels == "train")) == 28,
        "window_count_84": clean_tau.shape[0] == 84,
        "sequence_length_21": clean_tau.shape[1] == 21,
        "state_dim_99": clean_tau.shape[2] - 32 == 99,
        "latent_dim_32": clean_tau.shape[2] - 99 == 32,
        "token_dim_131": clean_tau.shape[2] == 131,
        "action_dim_29": target_action.shape[-1] == 29,
        "direct_decoder_uses_state_only": weights.shape[0] == 199,
        "direct_decoder_no_token_identity_basis": True,
        "all_actions_finite": bool(
            np.all(np.isfinite(direct_predicted_action)) and np.all(np.isfinite(latent_predicted_action))
        ),
        "heldout_direct_current_mse_finite": bool(np.all(np.isfinite(heldout_direct_current))),
        "heldout_latent_current_mse_finite": bool(np.all(np.isfinite(heldout_latent_current))),
        "latent_better_than_direct_on_validation_current": row_by_split["validation"][
            "latent_predicted_current_action_mse"
        ]
        < row_by_split["validation"]["direct_predicted_current_action_mse"],
        "latent_better_than_direct_on_test_current": row_by_split["test"]["latent_predicted_current_action_mse"]
        < row_by_split["test"]["direct_predicted_current_action_mse"],
        "does_not_claim_paper_direct_diffusion_success": True,
        "does_not_claim_trained_direct_checkpoint": True,
        "does_not_claim_trained_latent_checkpoint": True,
        "does_not_claim_closed_loop_rollout": True,
    }
    metrics = {
        "row_count": len(rows),
        "window_count": int(clean_tau.shape[0]),
        "sequence_length": int(clean_tau.shape[1]),
        "state_dim": 99,
        "latent_dim": 32,
        "token_dim": 131,
        "action_dim": 29,
        "direct_feature_dim": int(weights.shape[0]),
        "validation_direct_current_action_mse": row_by_split["validation"][
            "direct_predicted_current_action_mse"
        ],
        "test_direct_current_action_mse": row_by_split["test"]["direct_predicted_current_action_mse"],
        "validation_latent_current_action_mse": row_by_split["validation"][
            "latent_predicted_current_action_mse"
        ],
        "test_latent_current_action_mse": row_by_split["test"]["latent_predicted_current_action_mse"],
        "validation_latent_vs_direct_current_mse_ratio": row_by_split["validation"][
            "latent_vs_direct_current_mse_ratio"
        ],
        "test_latent_vs_direct_current_mse_ratio": row_by_split["test"]["latent_vs_direct_current_mse_ratio"],
        "validation_direct_full_action_mse": row_by_split["validation"]["direct_predicted_full_action_mse"],
        "test_direct_full_action_mse": row_by_split["test"]["direct_predicted_full_action_mse"],
        "validation_latent_full_action_mse": row_by_split["validation"]["latent_predicted_full_action_mse"],
        "test_latent_full_action_mse": row_by_split["test"]["latent_predicted_full_action_mse"],
        "rows": rows,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "debug_only_direct_vs_latent_action_ablation",
        "scope": (
            "Offline direct-state action decoder vs existing state-latent downstream action pipeline over local "
            "paper-state debug windows."
        ),
        "paper_evidence": {
            "direct_vs_latent_cartwheel_claim": str(ROOT / "res/comparison/paper_vs_reproduction.json"),
            "goal_ablation_phase": str(ROOT / "goal.md:1488-1537"),
            "vae_decoder_equation": str(ROOT / "reproduction/paper/source/tex/method.tex:157-162"),
            "state_latent_dataset": str(ROOT / "reproduction/paper/source/root.tex:536-546"),
        },
        "settings": {
            "history": HISTORY,
            "current_index": HISTORY,
            "ridge_lambda": RIDGE_LAMBDA,
            "direct_branch_features": ["state_99", "tanh_state_99", "bias"],
            "direct_branch_train_split": "train",
            "latent_branch_source": str(LATENT_ACTION_JSON),
            "uses_token_identity_basis": False,
        },
        "source_artifacts": {
            "vae_debug_latents_json": str(VAE_JSON),
            "vae_debug_latents_npz": str(VAE_NPZ),
            "vae_latent_heldout_json": str(DIFFUSION_JSON),
            "vae_latent_heldout_npz": str(DIFFUSION_NPZ),
            "latent_action_json": str(LATENT_ACTION_JSON),
            "latent_action_npz": str(LATENT_ACTION_NPZ),
        },
        "metrics": metrics,
        "checks": checks,
        "interpretation": {
            "paper_level_status": "partial",
            "goal_complete": False,
            "why_not_complete": (
                "This is an offline debug data-path ablation. It does not reproduce the paper's direct-vs-latent "
                "cartwheel success rates, does not train direct or latent diffusion checkpoints, and does not run "
                "closed-loop Isaac/robot evaluation."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(json_path), "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
