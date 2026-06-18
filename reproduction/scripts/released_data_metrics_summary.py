#!/usr/bin/env python3
"""Build numeric tables from reproduced released-data figure CSVs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
RELEASED = ROOT / "res/released_figures"
OUT = ROOT / "res/tables/released_data_metrics_summary"

ABLATION_IDS = [
    "ablation_orientation_representation",
    "ablation_observation_history",
    "ablation_armature",
    "ablation_latency",
    "ablation_pd_gain",
]
GRF_IDS = [
    "grf_walk_human_reference",
    "grf_walk_robot_real",
    "grf_run_human_reference",
    "grf_run_robot_real",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_stats(values: pd.Series) -> dict[str, float]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "min": float("nan"), "max": float("nan"), "abs_max": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "abs_max": float(np.max(np.abs(arr))),
    }


def source_row(path: Path, figure_id: str) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "relative_path": str(path.relative_to(ROOT)),
        "absolute_path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def summarize_ablations(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for figure_id in ABLATION_IDS:
        path = RELEASED / figure_id / f"{figure_id}_processed.csv"
        source_rows.append(source_row(path, figure_id))
        df = pd.read_csv(path)
        for (scope, metric), sub in df.groupby(["scope", "metric"], sort=True):
            baseline_exp = "origin" if "origin" in set(sub["experiment"]) else str(sub.iloc[0]["experiment"])
            baseline = float(sub[sub["experiment"] == baseline_exp].iloc[0]["mean"])
            best = sub.loc[sub["mean"].astype(float).idxmin()]
            worst = sub.loc[sub["mean"].astype(float).idxmax()]
            rows.append(
                {
                    "figure_id": figure_id,
                    "table": "released_tracking_ablation",
                    "scope": scope,
                    "metric": metric,
                    "baseline_experiment": baseline_exp,
                    "baseline_mean": baseline,
                    "best_experiment": str(best["experiment"]),
                    "best_mean": float(best["mean"]),
                    "best_vs_baseline_ratio": float(best["mean"]) / baseline if baseline != 0.0 else float("nan"),
                    "worst_experiment": str(worst["experiment"]),
                    "worst_mean": float(worst["mean"]),
                    "row_count": int(len(sub)),
                    "source_csv": str(path),
                }
            )
    return rows


def summarize_grf(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for figure_id in GRF_IDS:
        path = RELEASED / figure_id / f"{figure_id}_processed.csv"
        source_rows.append(source_row(path, figure_id))
        df = pd.read_csv(path)
        for axis in ["Fx", "Fy", "Fz"]:
            column = axis if axis in df.columns else f"{axis}_mean"
            stats = finite_stats(df[column])
            rows.append(
                {
                    "figure_id": figure_id,
                    "table": "released_grf",
                    "axis": axis,
                    "mean": stats["mean"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "abs_max": stats["abs_max"],
                    "sample_count": int(pd.to_numeric(df[column], errors="coerce").notna().sum()),
                    "source_csv": str(path),
                }
            )
    return rows


def summarize_imu(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    figure_id = "imu_orientation_accel_angular_velocity"
    path = RELEASED / figure_id / f"{figure_id}_processed.csv"
    source_rows.append(source_row(path, figure_id))
    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for signal in ["roll", "pitch", "yaw", "acc_x", "acc_y", "acc_z", "ang_x", "ang_y", "ang_z"]:
        stats = finite_stats(df[signal])
        rows.append(
            {
                "figure_id": figure_id,
                "table": "released_imu",
                "signal": signal,
                "mean": stats["mean"],
                "min": stats["min"],
                "max": stats["max"],
                "abs_max": stats["abs_max"],
                "sample_count": int(pd.to_numeric(df[signal], errors="coerce").notna().sum()),
                "source_csv": str(path),
            }
        )
    time = pd.to_numeric(df["time"], errors="coerce").to_numpy(dtype=float)
    time = time[np.isfinite(time)]
    if time.size:
        rows.append(
            {
                "figure_id": figure_id,
                "table": "released_imu",
                "signal": "time_duration_s",
                "mean": float(time[-1] - time[0]),
                "min": float(time[0]),
                "max": float(time[-1]),
                "abs_max": float(time[-1] - time[0]),
                "sample_count": int(time.size),
                "source_csv": str(path),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Released-data metrics summary",
        "",
        "This table is derived from already reproduced released-data processed CSV files under `res/released_figures`.",
        "It is Level A released-data evidence, not a new policy training run.",
        "",
        f"- Status: `{summary['status']}`",
        f"- Source CSV count: `{summary['metrics']['source_csv_count']}`",
        f"- Ablation summary rows: `{summary['metrics']['ablation_row_count']}`",
        f"- GRF summary rows: `{summary['metrics']['grf_row_count']}`",
        f"- IMU summary rows: `{summary['metrics']['imu_row_count']}`",
        f"- Best global position ablation: `{summary['metrics']['best_global_position_ablation']}`",
        f"- Peak vertical GRF abs value: `{summary['metrics']['peak_vertical_grf_abs']}`",
        f"- IMU duration seconds: `{summary['metrics']['imu_duration_s']}`",
        "",
        "Outputs:",
        f"- `{summary['outputs']['json']}`",
        f"- `{summary['outputs']['ablation_csv']}`",
        f"- `{summary['outputs']['grf_csv']}`",
        f"- `{summary['outputs']['imu_csv']}`",
        f"- `{summary['outputs']['source_hashes_tsv']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_rows: list[dict[str, Any]] = []
    ablation_rows = summarize_ablations(source_rows)
    grf_rows = summarize_grf(source_rows)
    imu_rows = summarize_imu(source_rows)

    ablation_csv = OUT / "released_tracking_ablation_metrics.csv"
    grf_csv = OUT / "released_grf_metrics.csv"
    imu_csv = OUT / "released_imu_metrics.csv"
    source_tsv = OUT / "source_hashes.tsv"
    json_path = OUT / "released_data_metrics_summary.json"
    md_path = OUT / "released_data_metrics_summary.md"
    write_csv(ablation_csv, ablation_rows)
    write_csv(grf_csv, grf_rows)
    write_csv(imu_csv, imu_rows)
    with source_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=["figure_id", "relative_path", "absolute_path", "size_bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(source_rows)

    global_pos = [
        row for row in ablation_rows if row["scope"] == "global" and row["metric"] == "pos_err"
    ]
    best_global_position = min(global_pos, key=lambda row: row["best_mean"])
    fz_rows = [row for row in grf_rows if row["axis"] == "Fz"]
    peak_fz = max(float(row["abs_max"]) for row in fz_rows)
    imu_duration = next(row for row in imu_rows if row["signal"] == "time_duration_s")

    checks = {
        "all_expected_source_csvs_present": len(source_rows) == 10 and all(Path(r["absolute_path"]).is_file() for r in source_rows),
        "source_hashes_recorded": len(source_rows) == 10 and all(r["sha256"] for r in source_rows),
        "ablation_rows_cover_five_groups": len({r["figure_id"] for r in ablation_rows}) == 5,
        "grf_rows_cover_four_groups": len({r["figure_id"] for r in grf_rows}) == 4,
        "imu_rows_cover_orientation_accel_angular_velocity": len(imu_rows) == 10,
        "metrics_are_finite": all(
            np.isfinite(float(row[key]))
            for rows in [ablation_rows, grf_rows, imu_rows]
            for row in rows
            for key in row
            if key.endswith("mean") or key in {"min", "max", "abs_max", "best_vs_baseline_ratio"}
        ),
        "does_not_claim_training": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "released_data_metrics_summary",
        "scope": "numeric summary tables derived from Level A released-data figure reproductions",
        "metrics": {
            "source_csv_count": len(source_rows),
            "ablation_row_count": len(ablation_rows),
            "grf_row_count": len(grf_rows),
            "imu_row_count": len(imu_rows),
            "best_global_position_ablation": {
                "figure_id": best_global_position["figure_id"],
                "best_experiment": best_global_position["best_experiment"],
                "best_mean": best_global_position["best_mean"],
                "baseline_experiment": best_global_position["baseline_experiment"],
                "baseline_mean": best_global_position["baseline_mean"],
            },
            "peak_vertical_grf_abs": peak_fz,
            "imu_duration_s": float(imu_duration["mean"]),
        },
        "checks": checks,
        "source_rows": source_rows,
        "not_a_replacement_for": [
            "new official-code PPO training",
            "new VAE/diffusion training",
            "Fig. 5/Fig. 6 rollout metrics",
            "real-robot reproduction",
        ],
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "released-data-only",
            "why_not_complete": (
                "These are summary tables over released processed data and local reproduced figures. They strengthen "
                "Level A reporting but do not create missing official-code training, diffusion checkpoints, or rollout videos."
            ),
        },
        "outputs": {
            "json": str(json_path),
            "markdown": str(md_path),
            "ablation_csv": str(ablation_csv),
            "grf_csv": str(grf_csv),
            "imu_csv": str(imu_csv),
            "source_hashes_tsv": str(source_tsv),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_md(md_path, summary)
    print(json.dumps({"status": summary["status"], "json": str(json_path), "source_csvs": len(source_rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
