#!/usr/bin/env python3
"""Compare public LAFAN1 paper-architecture training with/without symmetry augmentation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/lafan1_paper_arch_symmetry_training_comparison"
BASE_JSON = (
    ROOT
    / "res/level_c/lafan1_paper_arch_vae_diffusion_training/"
    / "lafan1_paper_arch_vae_diffusion_training.json"
)
SYM_DATASET_JSON = (
    ROOT
    / "res/level_c/lafan1_paper_arch_symmetry_dataset/"
    / "lafan1_paper_arch_symmetry_dataset_audit.json"
)
SYM_TRAIN_JSON = (
    ROOT
    / "res/level_c/lafan1_paper_arch_symmetry_augmented_training/"
    / "lafan1_paper_arch_vae_diffusion_training.json"
)
BASE_DATASET_NPZ = (
    ROOT
    / "res/level_c/lafan1_paper_arch_vae_diffusion_training/"
    / "lafan1_paper_arch_training_dataset.npz"
)
SYM_TRAIN_DATASET_NPZ = (
    ROOT
    / "res/level_c/lafan1_paper_arch_symmetry_augmented_training/"
    / "lafan1_paper_arch_training_dataset.npz"
)


METRIC_KEYS = [
    "final_validation_decoded_action_mse",
    "final_test_decoded_action_mse",
    "final_validation_pred_tau_mse",
    "final_test_pred_tau_mse",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ratio_delta(base: float, augmented: float) -> dict[str, float]:
    delta = augmented - base
    return {
        "base": base,
        "symmetry_augmented": augmented,
        "delta": delta,
        "relative_delta": delta / base if base else 0.0,
        "lower_is_better_improved": augmented < base,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "metric",
        "base",
        "symmetry_augmented",
        "delta",
        "relative_delta",
        "lower_is_better_improved",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def projection_sha256(path: Path) -> str:
    data = np.load(path)
    return hashlib.sha256(data["projection"].tobytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_json(BASE_JSON)
    sym_dataset = load_json(SYM_DATASET_JSON)
    sym_train = load_json(SYM_TRAIN_JSON)
    base_projection_sha256 = projection_sha256(BASE_DATASET_NPZ)
    sym_projection_sha256 = projection_sha256(SYM_TRAIN_DATASET_NPZ)
    metric_rows = []
    metric_summary = {}
    for key in METRIC_KEYS:
        item = ratio_delta(float(base["metrics"][key]), float(sym_train["metrics"][key]))
        metric_summary[key] = item
        metric_rows.append({"metric": key, **item})

    json_path = OUT / "level_c_lafan1_paper_arch_symmetry_training_comparison_audit.json"
    tsv_path = OUT / "level_c_lafan1_paper_arch_symmetry_training_comparison.tsv"
    npz_path = OUT / "level_c_lafan1_paper_arch_symmetry_training_comparison_metrics.npz"
    base_windows = int(base["metrics"]["window_count"])
    sym_windows = int(sym_train["metrics"]["window_count"])
    base_tokens = int(base["metrics"]["token_count"])
    sym_tokens = int(sym_train["metrics"]["token_count"])
    action_improved = [
        metric_summary["final_validation_decoded_action_mse"]["lower_is_better_improved"],
        metric_summary["final_test_decoded_action_mse"]["lower_is_better_improved"],
    ]
    tau_finite = [
        np.isfinite(metric_summary["final_validation_pred_tau_mse"]["symmetry_augmented"]),
        np.isfinite(metric_summary["final_test_pred_tau_mse"]["symmetry_augmented"]),
    ]
    checks = {
        "base_training_status_ok": base["status"] == "ok",
        "symmetry_dataset_status_ok": sym_dataset["status"] == "ok",
        "symmetry_training_status_ok": sym_train["status"] == "ok",
        "same_paper_vae_architecture": base["checks"]["paper_vae_architecture_used"]
        and sym_train["checks"]["paper_vae_architecture_used"],
        "same_paper_diffusion_architecture": base["checks"]["paper_diffusion_architecture_used"]
        and sym_train["checks"]["paper_diffusion_architecture_used"],
        "same_projection_matrix_sha256": base_projection_sha256 == sym_projection_sha256,
        "base_uses_public_lafan1_40_motions": int(base["metrics"]["public_lafan1_motion_count"]) == 40,
        "symmetry_training_uses_public_lafan1_40_original_motions": int(
            sym_train["metrics"]["public_lafan1_unique_motion_label_count"]
        )
        == 40,
        "symmetry_training_uses_80_augmented_labels": int(sym_train["metrics"]["augmented_motion_label_count"]) == 80,
        "window_count_doubled": sym_windows == base_windows * 2,
        "token_count_doubled": sym_tokens == base_tokens * 2,
        "both_runs_use_8gpu_dataparallel": bool(base["metrics"]["data_parallel"])
        and bool(sym_train["metrics"]["data_parallel"])
        and len(base["metrics"]["gpu_device_ids"]) == 8
        and len(sym_train["metrics"]["gpu_device_ids"]) == 8,
        "validation_action_mse_improved_with_symmetry": bool(action_improved[0]),
        "test_action_mse_improved_with_symmetry": bool(action_improved[1]),
        "diffusion_tau_metrics_finite": bool(all(tau_finite)),
        "comparison_tsv_written": tsv_path.is_file(),
        "comparison_npz_written": npz_path.is_file(),
        "does_not_claim_closed_loop_ablation": True,
        "does_not_claim_official_teacher_rollout_dataset": True,
        "goal_remains_incomplete": True,
    }
    np.savez_compressed(
        npz_path,
        metric_names=np.asarray(METRIC_KEYS),
        base_values=np.asarray([metric_summary[k]["base"] for k in METRIC_KEYS], dtype=np.float64),
        symmetry_augmented_values=np.asarray(
            [metric_summary[k]["symmetry_augmented"] for k in METRIC_KEYS], dtype=np.float64
        ),
        deltas=np.asarray([metric_summary[k]["delta"] for k in METRIC_KEYS], dtype=np.float64),
        relative_deltas=np.asarray([metric_summary[k]["relative_delta"] for k in METRIC_KEYS], dtype=np.float64),
    )
    write_tsv(tsv_path, metric_rows)
    checks["comparison_tsv_written"] = tsv_path.is_file()
    checks["comparison_npz_written"] = npz_path.is_file()
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "public_lafan1_paper_arch_symmetry_training_comparison_audit",
        "scope": (
            "Public-data paper-architecture comparison between the original LAFAN1 training run and the "
            "candidate sagittal-symmetry augmented training run. This is not a closed-loop paper success ablation."
        ),
        "source_artifacts": {
            "base_training_json": str(BASE_JSON),
            "base_training_dataset_npz": str(BASE_DATASET_NPZ),
            "symmetry_dataset_json": str(SYM_DATASET_JSON),
            "symmetry_augmented_training_json": str(SYM_TRAIN_JSON),
            "symmetry_augmented_training_dataset_npz": str(SYM_TRAIN_DATASET_NPZ),
        },
        "metrics": {
            "base_window_count": base_windows,
            "symmetry_augmented_window_count": sym_windows,
            "base_token_count": base_tokens,
            "symmetry_augmented_token_count": sym_tokens,
            "window_count_ratio": sym_windows / base_windows,
            "token_count_ratio": sym_tokens / base_tokens,
            "base_checkpoint_size_bytes": int(base["metrics"]["checkpoint_size_bytes"]),
            "symmetry_augmented_checkpoint_size_bytes": int(sym_train["metrics"]["checkpoint_size_bytes"]),
            "base_elapsed_seconds": float(base["metrics"]["elapsed_seconds"]),
            "symmetry_augmented_elapsed_seconds": float(sym_train["metrics"]["elapsed_seconds"]),
            "base_projection_sha256": base_projection_sha256,
            "symmetry_augmented_projection_sha256": sym_projection_sha256,
            "metric_summary": metric_summary,
        },
        "checks": checks,
        "metric_rows": metric_rows,
        "interpretation": {
            "paper_level_status": "public_data_training_comparison",
            "goal_complete": False,
            "summary": (
                "On the local public LAFAN1 paper-architecture setup, the symmetry-augmented run improves "
                "validation/test decoded-action MSE while keeping diffusion tau MSE in the same finite range. "
                "It does not prove paper closed-loop success because official teacher rollouts and task logs are absent."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    write_json(json_path, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(json_path),
                "base_windows": base_windows,
                "symmetry_augmented_windows": sym_windows,
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
