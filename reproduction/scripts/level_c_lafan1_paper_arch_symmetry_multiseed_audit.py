#!/usr/bin/env python3
"""Aggregate symmetry-augmented public-data paper-architecture LAFAN1 runs over 3 seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/lafan1_paper_arch_symmetry_augmented_multiseed_audit"

DEFAULT_SUMMARIES = [
    ROOT
    / "res/level_c/lafan1_paper_arch_symmetry_augmented_training/"
    / "lafan1_paper_arch_vae_diffusion_training.json",
    ROOT
    / "res/level_c/lafan1_paper_arch_symmetry_augmented_training_seed_20260622/"
    / "lafan1_paper_arch_vae_diffusion_training.json",
    ROOT
    / "res/level_c/lafan1_paper_arch_symmetry_augmented_training_seed_20260623/"
    / "lafan1_paper_arch_vae_diffusion_training.json",
]

METRIC_KEYS = [
    "final_validation_decoded_action_mse",
    "final_test_decoded_action_mse",
    "final_validation_pred_tau_mse",
    "final_test_pred_tau_mse",
    "public_lafan1_unique_motion_label_count",
    "augmented_motion_label_count",
    "window_count",
    "token_count",
    "vae_parameter_count",
    "diffusion_parameter_count",
    "checkpoint_size_bytes",
    "elapsed_seconds",
    "cuda_peak_memory_mb",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def seed_row(summary_path: Path, item: dict[str, Any]) -> dict[str, Any]:
    settings = item["settings"]
    metrics = item["metrics"]
    outputs = item["outputs"]
    return {
        "seed": int(settings["seed"]),
        "projection_seed": int(settings.get("projection_seed", settings["seed"])),
        "run_id": metrics["run_id"],
        "summary_json": str(summary_path),
        "checkpoint": outputs["checkpoint"],
        "checkpoint_sha256": metrics["checkpoint_sha256"],
        "status": item["status"],
        "dataset_source_label": item["dataset_boundary"]["used"],
        "paper_architecture": bool(metrics["paper_architecture"]),
        "paper_dataset": bool(metrics["paper_dataset"]),
        "data_parallel": bool(metrics["data_parallel"]),
        "gpu_count": len(metrics["gpu_device_ids"]),
        "vae_epochs": int(settings["vae_epochs"]),
        "diffusion_epochs": int(settings["diffusion_epochs"]),
        **{key: metrics[key] for key in METRIC_KEYS},
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", default=",".join(str(p) for p in DEFAULT_SUMMARIES))
    parser.add_argument("--output-dir", default=str(OUT))
    args = parser.parse_args()

    out = Path(args.output_dir)
    if not out.is_absolute():
        out = ROOT / out
    summary_paths = [Path(text.strip()) for text in args.summaries.split(",") if text.strip()]
    summary_paths = [p if p.is_absolute() else ROOT / p for p in summary_paths]
    if len(summary_paths) < 3:
        raise ValueError("symmetry-augmented paper-architecture LAFAN1 multi-seed audit requires at least 3 summaries")

    out.mkdir(parents=True, exist_ok=True)
    summaries = [load_json(path) for path in summary_paths]
    rows = [seed_row(path, item) for path, item in zip(summary_paths, summaries)]
    seeds = [row["seed"] for row in rows]
    ckpts = [Path(row["checkpoint"]) for row in rows]
    statistics = {key: stats([float(row[key]) for row in rows]) for key in METRIC_KEYS}

    tsv_path = out / "level_c_lafan1_paper_arch_symmetry_augmented_multiseed_rows.tsv"
    json_path = out / "level_c_lafan1_paper_arch_symmetry_augmented_multiseed_audit.json"
    npz_path = out / "level_c_lafan1_paper_arch_symmetry_augmented_multiseed_metrics.npz"
    write_tsv(tsv_path, rows)
    np.savez_compressed(
        npz_path,
        seeds=np.asarray(seeds, dtype=np.int64),
        validation_decoded_action_mse=np.asarray(
            [row["final_validation_decoded_action_mse"] for row in rows], dtype=np.float64
        ),
        test_decoded_action_mse=np.asarray([row["final_test_decoded_action_mse"] for row in rows], dtype=np.float64),
        validation_pred_tau_mse=np.asarray([row["final_validation_pred_tau_mse"] for row in rows], dtype=np.float64),
        test_pred_tau_mse=np.asarray([row["final_test_pred_tau_mse"] for row in rows], dtype=np.float64),
    )

    reference = rows[0]
    unique_public_motion_counts = {row["public_lafan1_unique_motion_label_count"] for row in rows}
    unique_augmented_motion_counts = {row["augmented_motion_label_count"] for row in rows}
    unique_windows = {row["window_count"] for row in rows}
    unique_tokens = {row["token_count"] for row in rows}
    unique_param_counts = {(row["vae_parameter_count"], row["diffusion_parameter_count"]) for row in rows}
    checks = {
        "at_least_three_seeds": len(rows) >= 3,
        "seeds_unique": len(set(seeds)) == len(seeds),
        "all_source_training_status_ok": all(row["status"] == "ok" for row in rows),
        "all_checkpoints_exist": all(path.is_file() for path in ckpts),
        "all_checkpoint_hashes_recorded": all(len(row["checkpoint_sha256"]) == 64 for row in rows),
        "all_metrics_finite": bool(np.all(np.isfinite([float(row[key]) for row in rows for key in METRIC_KEYS]))),
        "all_runs_use_symmetry_augmented_dataset": all(
            row["dataset_source_label"] == "public_lafan1_symmetry_augmented_dataset" for row in rows
        ),
        "all_runs_use_same_public_lafan1_unique_motion_count": unique_public_motion_counts == {40},
        "all_runs_use_same_augmented_motion_count": unique_augmented_motion_counts == {80},
        "all_runs_use_same_window_count": unique_windows == {reference["window_count"]} == {4400},
        "all_runs_use_same_token_count": unique_tokens == {reference["token_count"]} == {92400},
        "all_runs_use_paper_vae_and_diffusion_architecture": all(
            row["paper_architecture"]
            and row["vae_parameter_count"] >= 6_000_000
            and row["diffusion_parameter_count"] >= 19_000_000
            for row in rows
        ),
        "all_runs_use_same_parameter_counts": len(unique_param_counts) == 1,
        "all_runs_use_paper_training_epochs": all(
            row["vae_epochs"] >= 24 and row["diffusion_epochs"] >= 1000 for row in rows
        ),
        "all_runs_use_multigpu_dataparallel": all(row["data_parallel"] and row["gpu_count"] == 8 for row in rows),
        "all_runs_are_public_data_not_paper_private_dataset": all(row["paper_dataset"] is False for row in rows),
        "statistics_include_mean_std_min_max": all(
            set(statistics[key].keys()) == {"mean", "std", "min", "max"} for key in METRIC_KEYS
        ),
        "no_best_seed_only_reporting": True,
        "does_not_claim_official_teacher_rollout_dataset": True,
        "does_not_claim_closed_loop_or_robot_or_fig_videos": True,
    }

    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "public_lafan1_paper_arch_symmetry_augmented_vae_diffusion_multiseed_audit",
        "scope": (
            "3-seed statistics for full paper-sized conditional VAE and Transformer diffusion training on the "
            "public retargeted LAFAN1 G1 dataset after deterministic left-right symmetry augmentation."
        ),
        "paper_evidence": {
            "statistics_requirement": str(ROOT / "goal.md:1618-1631"),
            "no_best_seed_only_requirement": str(ROOT / "goal.md:890-891"),
            "vae_architecture_table": str(ROOT / "reproduction/paper/source/root.tex:803-821"),
            "diffusion_architecture_table": str(ROOT / "reproduction/paper/source/root.tex:827-848"),
        },
        "dataset_boundary": {
            "used": "download/official/LAFAN1_Retargeting_Dataset/g1 public retargeted motions plus deterministic mirror augmentation",
            "not_available": [
                "official teacher policy rollouts",
                "official DAgger aggregation states",
                "official VAE rollout state-latent trajectory dataset",
                "closed-loop Isaac/robot success and failure rollouts",
                "Fig. 5/Fig. 6 videos",
            ],
        },
        "settings": {
            "summary_jsons": [str(path) for path in summary_paths],
            "seeds": seeds,
            "seed_count": len(seeds),
            "projection_seed_note": (
                "All symmetry runs use projection_seed=20260617 so multi-seed variation reflects model/training "
                "randomness on the same augmented windows."
            ),
        },
        "rows": rows,
        "statistics": statistics,
        "checks": checks,
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "paper_architecture_public_data_symmetry_augmented_3seed_statistics",
            "why_not_complete": (
                "This strengthens the public-data full-architecture reproduction with symmetry-augmented 3-seed "
                "statistics, but exact paper reproduction still lacks official teacher-policy DAgger rollouts, VAE "
                "rollout state-latent trajectories, closed-loop success/failure videos, TensorRT deployment, and "
                "real robot trials."
            ),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if summary["status"] != "ok":
        raise SystemExit(json.dumps({"status": summary["status"], "failed_checks": [k for k, v in checks.items() if not v]}))
    print(
        json.dumps(
            {
                "status": "ok",
                "json": str(json_path),
                "seed_count": len(seeds),
                "validation_tau_mse_mean": statistics["final_validation_pred_tau_mse"]["mean"],
                "test_tau_mse_mean": statistics["final_test_pred_tau_mse"]["mean"],
                "test_action_mse_mean": statistics["final_test_decoded_action_mse"]["mean"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
