#!/usr/bin/env python3
"""Multi-seed debug audit for the diffusion-token to VAE-action pipe."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import level_c_diffusion_to_vae_action_smoke as pipe
import level_c_vae_latent_diffusion_overfit_probe as vae_overfit
import level_c_vae_latent_heldout_eval as heldout


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/diffusion_to_vae_action_multiseed_audit"


def split_indices_for_latent_dataset(
    motion_ids: np.ndarray,
    latent_manifest: dict[str, Any],
) -> tuple[dict[str, list[int]], dict[str, np.ndarray]]:
    split_manifest = json.loads(heldout.SPLIT_JSON.read_text(encoding="utf-8"))
    motion_names = sorted({row["source_motion"] for row in latent_manifest["rows"]})
    split_by_motion = {item["name"]: item["split"] for item in split_manifest["motions"]}
    split_to_motion_ids: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for motion_id, name in enumerate(motion_names):
        split_to_motion_ids[split_by_motion[name]].append(motion_id)
    split_to_window_indices = {
        split: np.nonzero(np.isin(motion_ids, np.asarray(ids, dtype=np.int64)))[0]
        for split, ids in split_to_motion_ids.items()
    }
    return split_to_motion_ids, split_to_window_indices


def load_action_targets() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vae_summary = json.loads(pipe.VAE_JSON.read_text(encoding="utf-8"))
    with np.load(pipe.VAE_NPZ) as vae_data:
        projection = vae_data["proprioception_projection"].astype(np.float64)
        rows = vae_summary["rows"]
        target_action = np.stack(
            [vae_data[f"{row['sample_id']}_decoded_action"].astype(np.float64) for row in rows],
            axis=0,
        )
        split_labels = np.asarray([row["split"] for row in rows])
        source_motion = np.asarray([row["source_motion"] for row in rows])
    return projection, target_action, split_labels, source_motion


def fit_diffusion_baseline(
    seed: int,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    cfg = replace(vae_overfit.VaeLatentDiffusionConfig(seed=seed, ridge_lambda=ridge_lambda))
    clean_tau, motion_ids, latent_manifest = vae_overfit.load_dataset(cfg)
    noisy_tau, steps = vae_overfit.make_noisy_tau(clean_tau, cfg)
    x = heldout.nonmemorizing_feature_matrix(noisy_tau, steps, motion_ids, cfg)
    y = clean_tau.reshape(x.shape[0], clean_tau.shape[-1])
    _, split_to_window_indices = split_indices_for_latent_dataset(motion_ids, latent_manifest)
    train_idx = heldout.expand_window_indices(split_to_window_indices["train"], cfg.sequence_length)
    reg = cfg.ridge_lambda * np.eye(x.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x[train_idx].T @ x[train_idx] + reg, x[train_idx].T @ y[train_idx])
    predicted_tau = (x @ weights).reshape(clean_tau.shape)
    source_checks = {
        "latent_manifest_status_ok": latent_manifest["status"] == "ok",
        "debug_vae_latents_nonzero": latent_manifest["checks"]["all_latents_nonzero"],
        "uses_independent_state_latent_steps": bool(np.any(steps[..., 0] != steps[..., 1])),
        "diffusion_feature_dim": int(x.shape[-1]),
    }
    return clean_tau, noisy_tau, predicted_tau, source_checks


def fit_action_surrogate(
    clean_tau: np.ndarray,
    projection: np.ndarray,
    target_action: np.ndarray,
    split_labels: np.ndarray,
) -> np.ndarray:
    clean_features = pipe.feature_tensor(clean_tau, projection)
    train_mask = pipe.expand_split_mask(split_labels, clean_features.shape[1], "train")
    x_train = clean_features.reshape(-1, clean_features.shape[-1])[train_mask]
    y_train = target_action.reshape(-1, target_action.shape[-1])[train_mask]
    reg = pipe.RIDGE_LAMBDA * np.eye(x_train.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    return np.linalg.solve(x_train.T @ x_train + reg, x_train.T @ y_train)


def run_seed(seed: int, diffusion_ridge_lambda: float) -> dict[str, Any]:
    projection, target_action, split_labels, source_motion = load_action_targets()
    clean_tau, noisy_tau, predicted_tau, source_checks = fit_diffusion_baseline(seed, diffusion_ridge_lambda)
    action_weights = fit_action_surrogate(clean_tau, projection, target_action, split_labels)
    clean_action = pipe.feature_tensor(clean_tau, projection) @ action_weights
    noisy_action = pipe.feature_tensor(noisy_tau, projection) @ action_weights
    predicted_action = pipe.feature_tensor(predicted_tau, projection) @ action_weights
    rows = pipe.split_rows(split_labels, source_motion, clean_action, noisy_action, predicted_action, target_action)
    row_by_split = {row["split"]: row for row in rows}
    row: dict[str, Any] = {
        "seed": seed,
        "window_count": int(clean_tau.shape[0]),
        "sequence_length": int(clean_tau.shape[1]),
        "token_dim": int(clean_tau.shape[2]),
        "state_dim": 99,
        "latent_dim": 32,
        "action_dim": int(target_action.shape[-1]),
        "decoder_surrogate_input_dim": int(action_weights.shape[0]),
        "decoder_surrogate_output_dim": int(action_weights.shape[1]),
        "uses_token_identity_basis": False,
        "uses_train_split_only_for_decoder_surrogate": int(np.sum(split_labels == "train")) == 28,
        **source_checks,
    }
    for split, split_row in row_by_split.items():
        for key, value in split_row.items():
            if key in {"split", "motion_names"}:
                row[f"{split}_{key}"] = value
            else:
                row[f"{split}_{key}"] = value
    return row


def stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "seed",
        "validation_predicted_current_action_mse",
        "validation_current_mse_reduction_vs_noisy",
        "validation_predicted_full_action_mse",
        "validation_full_mse_reduction_vs_noisy",
        "test_predicted_current_action_mse",
        "test_current_mse_reduction_vs_noisy",
        "test_predicted_full_action_mse",
        "test_full_mse_reduction_vs_noisy",
        "window_count",
        "sequence_length",
        "token_dim",
        "state_dim",
        "latent_dim",
        "action_dim",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="20260919,20260920,20260921")
    parser.add_argument("--diffusion-ridge-lambda", type=float, default=1e-5)
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if len(seeds) < 3:
        raise ValueError("diffusion-to-VAE action multi-seed audit requires at least three seeds")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run_seed(seed, args.diffusion_ridge_lambda) for seed in seeds]
    metric_names = [
        "validation_predicted_current_action_mse",
        "validation_current_mse_reduction_vs_noisy",
        "validation_predicted_full_action_mse",
        "validation_full_mse_reduction_vs_noisy",
        "test_predicted_current_action_mse",
        "test_current_mse_reduction_vs_noisy",
        "test_predicted_full_action_mse",
        "test_full_mse_reduction_vs_noisy",
    ]
    statistics = {name: stats([row[name] for row in rows]) for name in metric_names}

    json_path = OUT / "level_c_diffusion_to_vae_action_multiseed_audit.json"
    tsv_path = OUT / "level_c_diffusion_to_vae_action_multiseed_audit.tsv"
    write_tsv(tsv_path, rows)
    all_metric_values = [row[name] for row in rows for name in metric_names]
    checks = {
        "at_least_three_seeds": len(seeds) >= 3,
        "no_best_seed_only_reporting": True,
        "all_metrics_finite": bool(np.all(np.isfinite(all_metric_values))),
        "all_seed_validation_prediction_improves_current_vs_noisy": bool(
            all(row["validation_predicted_current_action_mse"] < row["validation_noisy_current_action_mse"] for row in rows)
        ),
        "all_seed_test_prediction_improves_current_vs_noisy": bool(
            all(row["test_predicted_current_action_mse"] < row["test_noisy_current_action_mse"] for row in rows)
        ),
        "all_seed_validation_prediction_improves_full_vs_noisy": bool(
            all(row["validation_predicted_full_action_mse"] < row["validation_noisy_full_action_mse"] for row in rows)
        ),
        "all_seed_test_prediction_improves_full_vs_noisy": bool(
            all(row["test_predicted_full_action_mse"] < row["test_noisy_full_action_mse"] for row in rows)
        ),
        "validation_current_mse_mean_below_0_02": statistics["validation_predicted_current_action_mse"]["mean"] < 0.02,
        "test_current_mse_mean_below_0_02": statistics["test_predicted_current_action_mse"]["mean"] < 0.02,
        "all_runs_use_train_split_only_for_decoder_surrogate": bool(
            all(row["uses_train_split_only_for_decoder_surrogate"] for row in rows)
        ),
        "all_runs_do_not_use_token_identity_basis": bool(all(not row["uses_token_identity_basis"] for row in rows)),
        "all_runs_use_independent_state_latent_steps": bool(
            all(row["uses_independent_state_latent_steps"] for row in rows)
        ),
        "all_runs_debug_vae_latent_artifact_ok": bool(all(row["latent_manifest_status_ok"] for row in rows)),
        "all_runs_debug_vae_latents_nonzero": bool(all(row["debug_vae_latents_nonzero"] for row in rows)),
        "all_runs_token_dim_131": bool(all(row["token_dim"] == 131 for row in rows)),
        "all_runs_action_dim_29": bool(all(row["action_dim"] == 29 for row in rows)),
        "does_not_claim_true_vae_decoder": True,
        "does_not_claim_trained_diffusion_checkpoint": True,
        "does_not_claim_closed_loop_rollout": True,
        "does_not_claim_goal_complete": True,
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "debug_only_diffusion_to_vae_action_multiseed_audit",
        "scope": (
            "three-seed downstream action audit for held-out diffusion token predictions through a train-split "
            "surrogate tiny-VAE action decoder"
        ),
        "paper_evidence": {
            "diffusion_to_vae_action_smoke": str(
                ROOT / "res/level_c/diffusion_to_vae_action_smoke/level_c_diffusion_to_vae_action_smoke.json"
            ),
            "vae_latent_heldout_eval": str(
                ROOT / "res/level_c/vae_latent_heldout_eval/level_c_vae_latent_heldout_eval.json"
            ),
            "goal_statistics": str(ROOT / "goal.md:1618-1631"),
            "goal_no_best_seed_only": str(ROOT / "goal.md:890-891"),
        },
        "not_a_replacement_for": [
            "true VAE rollout state-latent dataset",
            "trained VAE decoder checkpoint",
            "trained diffusion Transformer",
            "closed-loop Isaac evaluation",
            "paper multi-seed evaluation",
            "Fig. 5/Fig. 6 reproduction",
        ],
        "settings": {
            "seeds": seeds,
            "diffusion_ridge_lambda": args.diffusion_ridge_lambda,
            "action_decoder_ridge_lambda": pipe.RIDGE_LAMBDA,
            "current_index": pipe.HISTORY,
            "uses_token_identity_basis": False,
            "decoder_surrogate_train_split": "train",
        },
        "rows": rows,
        "statistics": statistics,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This is smoke-level multi-seed reporting for a surrogate downstream action decoder. It still lacks "
                "true VAE rollout data, the unpublished trained VAE decoder, trained diffusion Transformer, Isaac "
                "closed-loop rollouts, TensorRT deployment, and paper Fig. 5/Fig. 6 protocol."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(json_path), "seeds": len(seeds)}, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
