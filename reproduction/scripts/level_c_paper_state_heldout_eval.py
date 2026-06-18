#!/usr/bin/env python3
"""Held-out debug evaluation using paper-formula state windows."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

import level_c_paper_state_overfit_probe as paper_overfit


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/paper_state_heldout_eval"
SPLIT_JSON = ROOT / "res/level_c/small_dataset_split_manifest/small_dataset_split_manifest_summary.json"


def one_hot(indices: np.ndarray, classes: int) -> np.ndarray:
    out = np.zeros(indices.shape + (classes,), dtype=np.float64)
    flat = out.reshape(-1, classes)
    flat[np.arange(indices.size), indices.reshape(-1)] = 1.0
    return out


def nonmemorizing_feature_matrix(
    noisy_tau: np.ndarray,
    steps: np.ndarray,
    motion_ids: np.ndarray,
    cfg: paper_overfit.PaperStateConfig,
) -> np.ndarray:
    b, t, d = noisy_tau.shape
    sample_count = b * t
    motion_per_token = np.repeat(motion_ids, t)
    return np.concatenate(
        [
            noisy_tau.reshape(sample_count, d),
            one_hot(steps[..., 0], cfg.denoising_steps).reshape(sample_count, cfg.denoising_steps),
            one_hot(steps[..., 1], cfg.denoising_steps).reshape(sample_count, cfg.denoising_steps),
            one_hot(motion_per_token, int(motion_ids.max()) + 1),
            np.ones((sample_count, 1), dtype=np.float64),
        ],
        axis=1,
    )


def expand_window_indices(window_indices: np.ndarray, sequence_length: int) -> np.ndarray:
    return np.concatenate(
        [np.arange(idx * sequence_length, (idx + 1) * sequence_length, dtype=np.int64) for idx in window_indices]
    )


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["split", "motion_names", "window_count", "baseline_loss", "prediction_loss", "loss_reduction_ratio"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--ridge-lambda", type=float, default=1e-5)
    args = parser.parse_args()

    cfg = replace(paper_overfit.PaperStateConfig(seed=args.seed, ridge_lambda=args.ridge_lambda))
    OUT.mkdir(parents=True, exist_ok=True)
    split_manifest = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    clean_tau, motion_ids, state_summary, state_dim = paper_overfit.load_dataset(cfg)
    noisy_tau, steps = paper_overfit.make_noisy_tau(clean_tau, state_dim, cfg)
    x = nonmemorizing_feature_matrix(noisy_tau, steps, motion_ids, cfg)
    y = clean_tau.reshape(x.shape[0], clean_tau.shape[-1])
    baseline = noisy_tau.reshape(y.shape)

    names = [row["name"] for row in state_summary["rows"]]
    split_by_motion = {item["name"]: item["split"] for item in split_manifest["motions"]}
    split_to_motion_ids: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for motion_id, name in enumerate(names):
        split_to_motion_ids[split_by_motion[name]].append(motion_id)
    split_to_window_indices: dict[str, np.ndarray] = {}
    for split, ids in split_to_motion_ids.items():
        mask = np.isin(motion_ids, np.asarray(ids, dtype=np.int64))
        split_to_window_indices[split] = np.nonzero(mask)[0]

    train_idx = expand_window_indices(split_to_window_indices["train"], cfg.sequence_length)
    reg = cfg.ridge_lambda * np.eye(x.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x[train_idx].T @ x[train_idx] + reg, x[train_idx].T @ y[train_idx])
    pred = x @ weights

    rows: list[dict[str, Any]] = []
    metrics: dict[str, float] = {}
    for split in ["train", "validation", "test"]:
        window_idx = split_to_window_indices[split]
        token_idx = expand_window_indices(window_idx, cfg.sequence_length)
        baseline_loss = paper_overfit.mse(baseline[token_idx], y[token_idx])
        pred_loss = paper_overfit.mse(pred[token_idx], y[token_idx])
        ratio = (baseline_loss - pred_loss) / baseline_loss if baseline_loss > 0 else 0.0
        motion_names = [names[motion_id] for motion_id in split_to_motion_ids[split]]
        rows.append(
            {
                "split": split,
                "motion_names": ",".join(motion_names),
                "window_count": int(len(window_idx)),
                "baseline_loss": baseline_loss,
                "prediction_loss": pred_loss,
                "loss_reduction_ratio": ratio,
            }
        )
        metrics[f"{split}_baseline_loss"] = baseline_loss
        metrics[f"{split}_prediction_loss"] = pred_loss
        metrics[f"{split}_loss_reduction_ratio"] = ratio

    json_path = OUT / "level_c_paper_state_heldout_eval.json"
    tsv_path = OUT / "level_c_paper_state_heldout_eval.tsv"
    npz_path = OUT / "level_c_paper_state_heldout_eval.npz"
    write_tsv(tsv_path, rows)
    np.savez_compressed(
        npz_path,
        clean_tau=clean_tau,
        noisy_tau=noisy_tau,
        diffusion_steps=steps,
        motion_ids=motion_ids,
        prediction=pred.reshape(clean_tau.shape),
        ridge_weights=weights,
    )

    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "non-memorizing held-out ridge baseline using paper-formula 99-D state windows plus synthetic latents",
        "paper_evidence": {
            "paper_state_windows": str(paper_overfit.STATE_SUMMARY),
            "clean_trajectory_loss": str(ROOT / "reproduction/paper/source/tex/method.tex:187-196"),
            "motion_level_split": str(SPLIT_JSON),
        },
        "not_a_replacement_for": [
            "trained diffusion Transformer",
            "paper validation/test protocol",
            "teacher rollout state-latent dataset",
            "trained VAE latent trajectories",
            "Fig. 5/Fig. 6 reproduction",
        ],
        "settings": {
            **asdict(cfg),
            "paper_state_dim": state_dim,
            "token_dim": int(clean_tau.shape[-1]),
            "feature_dim": int(x.shape[-1]),
            "motion_count": int(len(names)),
            "window_count": int(clean_tau.shape[0]),
            "uses_token_identity_basis": False,
            "uses_motion_id_basis": True,
            "motion_level_split": split_manifest["split_policy"],
        },
        "metrics": {**metrics, "rows": rows},
        "checks": {
            "paper_state_windows_status_ok": state_summary["status"] == "ok",
            "paper_state_windows_all_checks_pass": state_summary["checks"]["all_checks_pass"],
            "uses_paper_state_dim_99": state_dim == 99,
            "losses_are_finite": bool(
                np.all(np.isfinite([value for key, value in metrics.items() if key.endswith("_loss")]))
            ),
            "heldout_splits_reported": all(f"{split}_prediction_loss" in metrics for split in ["train", "validation", "test"]),
            "uses_motion_level_split_manifest": split_manifest["checks"]["no_motion_crosses_splits"],
            "does_not_use_token_identity_basis": True,
            "uses_independent_state_latent_steps": bool(np.any(steps[..., 0] != steps[..., 1])),
            "train_loss_decreases_vs_noisy_baseline": metrics["train_prediction_loss"] < metrics["train_baseline_loss"],
            "validation_loss_decreases_vs_noisy_baseline": metrics["validation_prediction_loss"]
            < metrics["validation_baseline_loss"],
            "test_loss_decreases_vs_noisy_baseline": metrics["test_prediction_loss"] < metrics["test_baseline_loss"],
            "debug_fixture_boundary_recorded": state_summary["experiment_type"] == "debug_only",
        },
        "interpretation": {
            "paper_level_status": "partial",
            "why_not_complete": (
                "This is a held-out debug baseline over paper-formula state windows and synthetic latents. It does "
                "not replace a trained diffusion Transformer, true VAE rollout latents, or the paper evaluation."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
