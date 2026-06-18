#!/usr/bin/env python3
"""Debug-only multi-seed held-out statistics for the Level C fixture dataset."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

import level_c_small_dataset_heldout_eval as heldout
import level_c_small_dataset_overfit_probe as overfit


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/small_dataset_heldout_multiseed_audit"


def split_window_indices(
    names: list[str],
    motion_ids: np.ndarray,
    split_manifest: dict[str, Any],
) -> tuple[dict[str, list[int]], dict[str, np.ndarray]]:
    split_by_motion = {item["name"]: item["split"] for item in split_manifest["motions"]}
    split_to_motion_ids: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for motion_id, name in enumerate(names):
        split_to_motion_ids[split_by_motion[name]].append(motion_id)

    split_to_window_indices: dict[str, np.ndarray] = {}
    for split, ids in split_to_motion_ids.items():
        mask = np.isin(motion_ids, np.asarray(ids, dtype=np.int64))
        split_to_window_indices[split] = np.nonzero(mask)[0]
    return split_to_motion_ids, split_to_window_indices


def run_seed(seed: int, fixture_names: list[str], ridge_lambda: float) -> dict[str, Any]:
    cfg = replace(overfit.SmallDatasetConfig(seed=seed, ridge_lambda=ridge_lambda))
    split_manifest = json.loads(heldout.SPLIT_JSON.read_text(encoding="utf-8"))
    clean_tau, motion_ids, manifests, state_dim = overfit.load_dataset(cfg, fixture_names)
    noisy_tau, steps = overfit.make_noisy_tau(clean_tau, state_dim, cfg)
    x = heldout.nonmemorizing_feature_matrix(noisy_tau, steps, motion_ids, cfg)
    y = clean_tau.reshape(x.shape[0], clean_tau.shape[-1])
    baseline = noisy_tau.reshape(y.shape)

    split_to_motion_ids, split_to_window_indices = split_window_indices(fixture_names, motion_ids, split_manifest)
    train_idx = heldout.expand_window_indices(split_to_window_indices["train"], cfg.sequence_length)
    reg = cfg.ridge_lambda * np.eye(x.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x[train_idx].T @ x[train_idx] + reg, x[train_idx].T @ y[train_idx])
    pred = x @ weights

    row: dict[str, Any] = {
        "seed": seed,
        "motion_count": len(fixture_names),
        "window_count": int(clean_tau.shape[0]),
        "sample_count": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "uses_token_identity_basis": False,
        "uses_motion_id_basis": True,
        "uses_motion_level_split_manifest": split_manifest["checks"]["no_motion_crosses_splits"],
        "uses_independent_state_latent_steps": bool(np.any(steps[..., 0] != steps[..., 1])),
        "debug_fixture_boundary_recorded": all(item["experiment_type"] == "debug_only" for item in manifests),
    }
    for split in ["train", "validation", "test"]:
        token_idx = heldout.expand_window_indices(split_to_window_indices[split], cfg.sequence_length)
        baseline_loss = overfit.mse(baseline[token_idx], y[token_idx])
        pred_loss = overfit.mse(pred[token_idx], y[token_idx])
        ratio = (baseline_loss - pred_loss) / baseline_loss if baseline_loss > 0 else 0.0
        motion_names = [fixture_names[motion_id] for motion_id in split_to_motion_ids[split]]
        row[f"{split}_motion_names"] = ",".join(motion_names)
        row[f"{split}_window_count"] = int(len(split_to_window_indices[split]))
        row[f"{split}_baseline_loss"] = baseline_loss
        row[f"{split}_prediction_loss"] = pred_loss
        row[f"{split}_loss_reduction_ratio"] = ratio
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
        "train_baseline_loss",
        "train_prediction_loss",
        "train_loss_reduction_ratio",
        "validation_baseline_loss",
        "validation_prediction_loss",
        "validation_loss_reduction_ratio",
        "test_baseline_loss",
        "test_prediction_loss",
        "test_loss_reduction_ratio",
        "motion_count",
        "window_count",
        "sample_count",
        "feature_dim",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-names", type=str, default=",".join(overfit.DEFAULT_FIXTURE_NAMES))
    parser.add_argument("--seeds", type=str, default="20260907,20260908,20260909")
    parser.add_argument("--ridge-lambda", type=float, default=1e-5)
    args = parser.parse_args()

    fixture_names = [item.strip() for item in args.fixture_names.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if len(seeds) < 3:
        raise ValueError("held-out multi-seed audit requires at least 3 seeds")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run_seed(seed, fixture_names, args.ridge_lambda) for seed in seeds]
    metric_names = [
        "train_baseline_loss",
        "train_prediction_loss",
        "train_loss_reduction_ratio",
        "validation_baseline_loss",
        "validation_prediction_loss",
        "validation_loss_reduction_ratio",
        "test_baseline_loss",
        "test_prediction_loss",
        "test_loss_reduction_ratio",
    ]
    statistics = {name: stats([row[name] for row in rows]) for name in metric_names}

    json_path = OUT / "level_c_small_dataset_heldout_multiseed_audit.json"
    tsv_path = OUT / "level_c_small_dataset_heldout_multiseed_audit.tsv"
    write_tsv(tsv_path, rows)
    all_metric_values = [row[name] for row in rows for name in metric_names]
    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "3-seed held-out debug evaluation for a non-memorizing ridge baseline",
        "paper_evidence": {
            "goal_evaluation": str(ROOT / "goal.md:1550-1634"),
            "goal_statistics": str(ROOT / "goal.md:1618-1631"),
            "goal_no_best_seed_only": str(ROOT / "goal.md:890-891"),
            "small_dataset_heldout_eval": str(
                ROOT / "res/level_c/small_dataset_heldout_eval/level_c_small_dataset_heldout_eval.json"
            ),
            "small_dataset_split": str(heldout.SPLIT_JSON),
        },
        "not_a_replacement_for": [
            "trained diffusion Transformer",
            "paper validation/test protocol",
            "multi-seed paper evaluation",
            "checkpoint reproduction",
            "Fig. 5/Fig. 6 reproduction",
        ],
        "settings": {
            "fixture_names": fixture_names,
            "seeds": seeds,
            "ridge_lambda": args.ridge_lambda,
            "uses_token_identity_basis": False,
            "uses_motion_id_basis": True,
            **asdict(overfit.SmallDatasetConfig(ridge_lambda=args.ridge_lambda)),
        },
        "rows": rows,
        "statistics": statistics,
        "checks": {
            "at_least_three_seeds": len(seeds) >= 3,
            "no_best_seed_only_reporting": True,
            "all_losses_finite": bool(np.all(np.isfinite(all_metric_values))),
            "all_seed_train_loss_decreases": bool(
                all(row["train_prediction_loss"] < row["train_baseline_loss"] for row in rows)
            ),
            "all_seed_validation_loss_decreases": bool(
                all(row["validation_prediction_loss"] < row["validation_baseline_loss"] for row in rows)
            ),
            "all_seed_test_loss_decreases": bool(
                all(row["test_prediction_loss"] < row["test_baseline_loss"] for row in rows)
            ),
            "all_runs_use_motion_level_split_manifest": bool(
                all(row["uses_motion_level_split_manifest"] for row in rows)
            ),
            "all_runs_do_not_use_token_identity_basis": bool(
                all(not row["uses_token_identity_basis"] for row in rows)
            ),
            "all_runs_use_independent_state_latent_steps": bool(
                all(row["uses_independent_state_latent_steps"] for row in rows)
            ),
            "debug_fixture_boundary_recorded": bool(
                all(row["debug_fixture_boundary_recorded"] for row in rows)
            ),
        },
        "interpretation": {
            "paper_level_status": "partial",
            "why_not_complete": (
                "This reports three seeds for a simple non-memorizing ridge held-out debug baseline. It is useful "
                "for smoke-level statistical reporting, but it is not a trained diffusion Transformer or paper "
                "evaluation protocol."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
