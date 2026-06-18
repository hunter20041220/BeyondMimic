#!/usr/bin/env python3
"""Debug-only multi-seed statistics for the Level C small-dataset overfit gate."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

import level_c_small_dataset_overfit_probe as overfit


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/small_dataset_multiseed_audit"


def run_seed(seed: int, fixture_names: list[str], ridge_lambda: float) -> dict[str, Any]:
    cfg = replace(overfit.SmallDatasetConfig(seed=seed, ridge_lambda=ridge_lambda))
    clean_tau, motion_ids, manifests, state_dim = overfit.load_dataset(cfg, fixture_names)
    noisy_tau, steps = overfit.make_noisy_tau(clean_tau, state_dim, cfg)
    x = overfit.feature_matrix(noisy_tau, steps, motion_ids, cfg)
    y = clean_tau.reshape(x.shape[0], clean_tau.shape[-1])
    noisy_baseline = noisy_tau.reshape(y.shape)
    weights = overfit.ridge_fit(x, y, cfg.ridge_lambda)
    pred = x @ weights
    baseline_loss = overfit.mse(noisy_baseline, y)
    final_loss = overfit.mse(pred, y)
    return {
        "seed": seed,
        "baseline_loss": baseline_loss,
        "final_overfit_loss": final_loss,
        "loss_reduction_ratio": (baseline_loss - final_loss) / baseline_loss if baseline_loss > 0 else 0.0,
        "motion_count": len(fixture_names),
        "window_count": int(clean_tau.shape[0]),
        "sample_count": int(x.shape[0]),
        "feature_dim": int(x.shape[1]),
        "uses_independent_state_latent_steps": bool(np.any(steps[..., 0] != steps[..., 1])),
        "debug_fixture_boundary_recorded": all(item["experiment_type"] == "debug_only" for item in manifests),
    }


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
        "baseline_loss",
        "final_overfit_loss",
        "loss_reduction_ratio",
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
    parser.add_argument("--seeds", type=str, default="20260904,20260905,20260906")
    parser.add_argument("--ridge-lambda", type=float, default=1e-8)
    args = parser.parse_args()
    fixture_names = [item.strip() for item in args.fixture_names.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if len(seeds) < 3:
        raise ValueError("multi-seed audit requires at least 3 seeds")

    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run_seed(seed, fixture_names, args.ridge_lambda) for seed in seeds]
    final_losses = [row["final_overfit_loss"] for row in rows]
    ratios = [row["loss_reduction_ratio"] for row in rows]
    baseline_losses = [row["baseline_loss"] for row in rows]

    json_path = OUT / "level_c_small_dataset_multiseed_audit.json"
    tsv_path = OUT / "level_c_small_dataset_multiseed_audit.tsv"
    write_tsv(tsv_path, rows)
    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "3-seed statistics for the small-dataset overfit memorization gate",
        "paper_evidence": {
            "goal_statistics": str(ROOT / "goal.md:1618-1631"),
            "goal_no_best_seed_only": str(ROOT / "goal.md:890-891"),
            "small_dataset_overfit": str(
                ROOT / "res/level_c/small_dataset_overfit_probe/level_c_small_dataset_overfit_probe.json"
            ),
        },
        "not_a_replacement_for": [
            "multi-seed full diffusion training",
            "held-out validation statistics",
            "paper evaluation metrics",
            "checkpoint reproduction",
            "Fig. 5/Fig. 6 reproduction",
        ],
        "settings": {
            "fixture_names": fixture_names,
            "seeds": seeds,
            "ridge_lambda": args.ridge_lambda,
            **asdict(overfit.SmallDatasetConfig(ridge_lambda=args.ridge_lambda)),
        },
        "rows": rows,
        "statistics": {
            "baseline_loss": stats(baseline_losses),
            "final_overfit_loss": stats(final_losses),
            "loss_reduction_ratio": stats(ratios),
        },
        "checks": {
            "at_least_three_seeds": len(seeds) >= 3,
            "all_losses_finite": bool(np.all(np.isfinite(final_losses + baseline_losses + ratios))),
            "all_seeds_reduce_loss": bool(all(row["final_overfit_loss"] < row["baseline_loss"] for row in rows)),
            "all_seed_final_loss_below_1e_minus_8": bool(all(row["final_overfit_loss"] < 1e-8 for row in rows)),
            "all_seed_loss_reduction_ratio_at_least_0_99": bool(all(row["loss_reduction_ratio"] >= 0.99 for row in rows)),
            "no_best_seed_only_reporting": True,
            "all_runs_use_independent_state_latent_steps": bool(
                all(row["uses_independent_state_latent_steps"] for row in rows)
            ),
            "debug_fixture_boundary_recorded": bool(all(row["debug_fixture_boundary_recorded"] for row in rows)),
        },
        "interpretation": {
            "paper_level_status": "partial",
            "why_not_complete": (
                "This is a 3-seed statistic for a debug memorization gate only. It satisfies the reporting habit of "
                "not selecting a best seed, but it is not multi-seed paper training or evaluation."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
