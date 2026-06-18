#!/usr/bin/env python3
"""Action smoothness audit for the debug diffusion-to-VAE action pipe."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SOURCE_JSON = ROOT / "res/level_c/diffusion_to_vae_action_smoke/level_c_diffusion_to_vae_action_smoke.json"
SOURCE_NPZ = ROOT / "res/level_c/diffusion_to_vae_action_smoke/level_c_diffusion_to_vae_action_smoke.npz"
MULTISEED_JSON = (
    ROOT
    / "res/level_c/diffusion_to_vae_action_multiseed_audit/level_c_diffusion_to_vae_action_multiseed_audit.json"
)
OUT = ROOT / "res/level_c/diffusion_to_vae_action_smoothness_audit"
CONTROL_FREQUENCY_HZ = 25.0


def stream_metrics(actions: np.ndarray, prefix: str) -> dict[str, float]:
    first = np.diff(actions, axis=1)
    second = np.diff(actions, n=2, axis=1)
    dt = 1.0 / CONTROL_FREQUENCY_HZ
    action_rate = first / dt
    action_acceleration = second / (dt * dt)
    return {
        f"{prefix}_first_difference_mean_norm": float(np.linalg.norm(first, axis=-1).mean()),
        f"{prefix}_first_difference_max_norm": float(np.linalg.norm(first, axis=-1).max()),
        f"{prefix}_second_difference_mean_norm": float(np.linalg.norm(second, axis=-1).mean()),
        f"{prefix}_second_difference_max_norm": float(np.linalg.norm(second, axis=-1).max()),
        f"{prefix}_smoothness_penalty": float(np.mean(second**2)),
        f"{prefix}_action_rate_mean_norm_at_25hz": float(np.linalg.norm(action_rate, axis=-1).mean()),
        f"{prefix}_action_rate_max_norm_at_25hz": float(np.linalg.norm(action_rate, axis=-1).max()),
        f"{prefix}_action_acceleration_mean_norm_at_25hz": float(
            np.linalg.norm(action_acceleration, axis=-1).mean()
        ),
        f"{prefix}_action_acceleration_max_norm_at_25hz": float(
            np.linalg.norm(action_acceleration, axis=-1).max()
        ),
    }


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.square(a - b)))


def split_rows(data: np.lib.npyio.NpzFile) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_labels = data["split_labels"]
    source_motion = data["source_motion"]
    streams = {
        "target": data["target_action"].astype(np.float64),
        "clean": data["clean_action"].astype(np.float64),
        "noisy": data["noisy_action"].astype(np.float64),
        "predicted": data["predicted_action"].astype(np.float64),
    }
    for split in ["train", "validation", "test"]:
        mask = split_labels == split
        row: dict[str, Any] = {
            "split": split,
            "motion_names": ",".join(sorted(set(source_motion[mask].tolist()))),
            "window_count": int(np.sum(mask)),
            "sequence_length": int(streams["target"].shape[1]),
            "action_dim": int(streams["target"].shape[-1]),
        }
        for name, actions in streams.items():
            row.update(stream_metrics(actions[mask], name))
        row["predicted_vs_target_action_mse"] = mse(streams["predicted"][mask], streams["target"][mask])
        row["noisy_vs_target_action_mse"] = mse(streams["noisy"][mask], streams["target"][mask])
        row["clean_vs_target_action_mse"] = mse(streams["clean"][mask], streams["target"][mask])
        row["predicted_smoothness_penalty_reduction_vs_noisy"] = (
            (row["noisy_smoothness_penalty"] - row["predicted_smoothness_penalty"])
            / row["noisy_smoothness_penalty"]
            if row["noisy_smoothness_penalty"] > 0
            else 0.0
        )
        row["predicted_action_rate_reduction_vs_noisy"] = (
            (row["noisy_action_rate_mean_norm_at_25hz"] - row["predicted_action_rate_mean_norm_at_25hz"])
            / row["noisy_action_rate_mean_norm_at_25hz"]
            if row["noisy_action_rate_mean_norm_at_25hz"] > 0
            else 0.0
        )
        row["predicted_action_acceleration_reduction_vs_noisy"] = (
            (
                row["noisy_action_acceleration_mean_norm_at_25hz"]
                - row["predicted_action_acceleration_mean_norm_at_25hz"]
            )
            / row["noisy_action_acceleration_mean_norm_at_25hz"]
            if row["noisy_action_acceleration_mean_norm_at_25hz"] > 0
            else 0.0
        )
        rows.append(row)
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "split",
        "motion_names",
        "window_count",
        "sequence_length",
        "action_dim",
        "target_smoothness_penalty",
        "clean_smoothness_penalty",
        "noisy_smoothness_penalty",
        "predicted_smoothness_penalty",
        "predicted_smoothness_penalty_reduction_vs_noisy",
        "target_action_rate_mean_norm_at_25hz",
        "clean_action_rate_mean_norm_at_25hz",
        "noisy_action_rate_mean_norm_at_25hz",
        "predicted_action_rate_mean_norm_at_25hz",
        "predicted_action_rate_reduction_vs_noisy",
        "target_action_acceleration_mean_norm_at_25hz",
        "clean_action_acceleration_mean_norm_at_25hz",
        "noisy_action_acceleration_mean_norm_at_25hz",
        "predicted_action_acceleration_mean_norm_at_25hz",
        "predicted_action_acceleration_reduction_vs_noisy",
        "predicted_vs_target_action_mse",
        "noisy_vs_target_action_mse",
        "clean_vs_target_action_mse",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    multiseed_summary = json.loads(MULTISEED_JSON.read_text(encoding="utf-8"))
    with np.load(SOURCE_NPZ) as data:
        rows = split_rows(data)
    row_by_split = {row["split"]: row for row in rows}

    json_path = OUT / "level_c_diffusion_to_vae_action_smoothness_audit.json"
    tsv_path = OUT / "level_c_diffusion_to_vae_action_smoothness_audit.tsv"
    write_tsv(tsv_path, rows)

    checks = {
        "source_smoke_status_ok": source_summary["status"] == "ok",
        "multiseed_source_status_ok": multiseed_summary["status"] == "ok",
        "row_count_3_splits": len(rows) == 3,
        "action_dim_29": all(row["action_dim"] == 29 for row in rows),
        "sequence_length_21": all(row["sequence_length"] == 21 for row in rows),
        "control_frequency_25hz": CONTROL_FREQUENCY_HZ == 25.0,
        "all_metrics_finite": bool(
            all(
                np.isfinite(value)
                for row in rows
                for key, value in row.items()
                if key not in {"split", "motion_names"}
            )
        ),
        "validation_predicted_smoother_than_noisy": row_by_split["validation"][
            "predicted_smoothness_penalty"
        ]
        < row_by_split["validation"]["noisy_smoothness_penalty"],
        "test_predicted_smoother_than_noisy": row_by_split["test"]["predicted_smoothness_penalty"]
        < row_by_split["test"]["noisy_smoothness_penalty"],
        "validation_predicted_action_rate_below_noisy": row_by_split["validation"][
            "predicted_action_rate_mean_norm_at_25hz"
        ]
        < row_by_split["validation"]["noisy_action_rate_mean_norm_at_25hz"],
        "test_predicted_action_rate_below_noisy": row_by_split["test"][
            "predicted_action_rate_mean_norm_at_25hz"
        ]
        < row_by_split["test"]["noisy_action_rate_mean_norm_at_25hz"],
        "validation_predicted_acceleration_below_noisy": row_by_split["validation"][
            "predicted_action_acceleration_mean_norm_at_25hz"
        ]
        < row_by_split["validation"]["noisy_action_acceleration_mean_norm_at_25hz"],
        "test_predicted_acceleration_below_noisy": row_by_split["test"][
            "predicted_action_acceleration_mean_norm_at_25hz"
        ]
        < row_by_split["test"]["noisy_action_acceleration_mean_norm_at_25hz"],
        "does_not_claim_closed_loop_action_smoothness": True,
        "does_not_claim_trained_policy_rollout": True,
        "does_not_claim_goal_complete": True,
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "debug_only_diffusion_to_vae_action_smoothness_audit",
        "scope": "action-rate and second-difference smoothness audit for downstream diffusion-to-VAE action streams",
        "paper_evidence": {
            "action_smoothness_context": str(ROOT / "reproduction/paper/source/tex/method.tex:76"),
            "control_frequency_and_latency": str(ROOT / "reproduction/paper/source/root.tex:589-593"),
            "goal_action_smoothness_metric": str(ROOT / "goal.md:1603-1615"),
        },
        "source_artifacts": {
            "diffusion_to_vae_action_smoke": str(SOURCE_JSON),
            "diffusion_to_vae_action_npz": str(SOURCE_NPZ),
            "diffusion_to_vae_action_multiseed": str(MULTISEED_JSON),
        },
        "settings": {
            "control_frequency_hz": CONTROL_FREQUENCY_HZ,
            "control_dt_seconds": 1.0 / CONTROL_FREQUENCY_HZ,
            "streams": ["target", "clean", "noisy", "predicted"],
            "splits": ["train", "validation", "test"],
        },
        "rows": rows,
        "metrics": {
            "validation_predicted_smoothness_penalty": row_by_split["validation"][
                "predicted_smoothness_penalty"
            ],
            "test_predicted_smoothness_penalty": row_by_split["test"]["predicted_smoothness_penalty"],
            "validation_predicted_smoothness_reduction_vs_noisy": row_by_split["validation"][
                "predicted_smoothness_penalty_reduction_vs_noisy"
            ],
            "test_predicted_smoothness_reduction_vs_noisy": row_by_split["test"][
                "predicted_smoothness_penalty_reduction_vs_noisy"
            ],
            "validation_predicted_action_rate_mean_norm_at_25hz": row_by_split["validation"][
                "predicted_action_rate_mean_norm_at_25hz"
            ],
            "test_predicted_action_rate_mean_norm_at_25hz": row_by_split["test"][
                "predicted_action_rate_mean_norm_at_25hz"
            ],
            "validation_predicted_action_acceleration_mean_norm_at_25hz": row_by_split["validation"][
                "predicted_action_acceleration_mean_norm_at_25hz"
            ],
            "test_predicted_action_acceleration_mean_norm_at_25hz": row_by_split["test"][
                "predicted_action_acceleration_mean_norm_at_25hz"
            ],
        },
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This audits action smoothness for local debug downstream action streams only. It does not measure "
                "a trained closed-loop policy, real robot actions, TensorRT latency, or paper Fig. 5/Fig. 6 task "
                "rollouts."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(json_path), "rows": len(rows)}, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
