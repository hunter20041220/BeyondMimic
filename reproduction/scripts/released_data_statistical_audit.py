#!/usr/bin/env python3
"""Statistical audit for Level A released-data reproductions.

This computes confidence intervals, normalized differences, and paper-claim
sanity checks from the reproduced released-data processed CSVs. It does not
claim new policy training or paper-level closed-loop evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
RELEASED = ROOT / "res/released_figures"
OUT = ROOT / "res/tables/released_data_statistical_audit"

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


def finite_array(values: Any) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    return arr[np.isfinite(arr)]


def mean_ci(values: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "abs_max": float("nan"),
        }
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    mean = float(np.mean(arr))
    margin = 1.96 * sem
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "abs_max": float(np.max(np.abs(arr))),
    }


def pooled_effect(delta: float, spread_a: float, spread_b: float) -> float:
    pooled = math.sqrt(max((spread_a * spread_a + spread_b * spread_b) / 2.0, 0.0))
    return float(delta / pooled) if pooled > 1e-12 else float("nan")


def source_row(path: Path, figure_id: str) -> dict[str, Any]:
    return {
        "figure_id": figure_id,
        "relative_path": str(path.relative_to(ROOT)),
        "absolute_path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def ablation_statistics(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for figure_id in ABLATION_IDS:
        path = RELEASED / figure_id / f"{figure_id}_processed.csv"
        source_rows.append(source_row(path, figure_id))
        df = pd.read_csv(path)
        for (scope, metric), sub in df.groupby(["scope", "metric"], sort=True):
            baseline_name = "origin" if "origin" in set(sub["experiment"]) else str(sub.iloc[0]["experiment"])
            baseline = sub[sub["experiment"] == baseline_name].iloc[0]
            baseline_mean = float(baseline["mean"])
            baseline_range_spread = max((float(baseline["max"]) - float(baseline["min"])) / 3.92, 1e-12)
            best = sub.loc[sub["mean"].astype(float).idxmin()]
            best_mean = float(best["mean"])
            best_range_spread = max((float(best["max"]) - float(best["min"])) / 3.92, 1e-12)
            delta = baseline_mean - best_mean
            rows.append(
                {
                    "figure_id": figure_id,
                    "source_csv": str(path),
                    "scope": scope,
                    "metric": metric,
                    "baseline_experiment": baseline_name,
                    "baseline_mean": baseline_mean,
                    "baseline_min": float(baseline["min"]),
                    "baseline_max": float(baseline["max"]),
                    "best_experiment": str(best["experiment"]),
                    "best_mean": best_mean,
                    "best_min": float(best["min"]),
                    "best_max": float(best["max"]),
                    "absolute_improvement": delta,
                    "relative_improvement": delta / baseline_mean if abs(baseline_mean) > 1e-12 else float("nan"),
                    "range_based_effect_size": pooled_effect(delta, baseline_range_spread, best_range_spread),
                    "row_count": int(len(sub)),
                    "best_beats_baseline": bool(best_mean < baseline_mean),
                }
            )
    return rows


def grf_statistics(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for figure_id in GRF_IDS:
        path = RELEASED / figure_id / f"{figure_id}_processed.csv"
        source_rows.append(source_row(path, figure_id))
        df = pd.read_csv(path)
        for axis in ["Fx", "Fy", "Fz"]:
            column = axis if axis in df.columns else f"{axis}_mean"
            stats = mean_ci(finite_array(df[column]))
            rows.append(
                {
                    "figure_id": figure_id,
                    "source_csv": str(path),
                    "signal": axis,
                    **stats,
                }
            )
    return rows


def imu_statistics(source_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    figure_id = "imu_orientation_accel_angular_velocity"
    path = RELEASED / figure_id / f"{figure_id}_processed.csv"
    source_rows.append(source_row(path, figure_id))
    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for signal in ["roll", "pitch", "yaw", "acc_x", "acc_y", "acc_z", "ang_x", "ang_y", "ang_z"]:
        rows.append({"figure_id": figure_id, "source_csv": str(path), "signal": signal, **mean_ci(finite_array(df[signal]))})

    acc = df[["acc_x", "acc_y", "acc_z"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    ang = df[["ang_x", "ang_y", "ang_z"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    acc_norm = np.linalg.norm(acc[np.isfinite(acc).all(axis=1)], axis=1)
    ang_norm = np.linalg.norm(ang[np.isfinite(ang).all(axis=1)], axis=1)
    rows.append({"figure_id": figure_id, "source_csv": str(path), "signal": "acc_norm", **mean_ci(acc_norm)})
    rows.append({"figure_id": figure_id, "source_csv": str(path), "signal": "ang_norm", **mean_ci(ang_norm)})
    time = finite_array(df["time"])
    duration = float(time[-1] - time[0]) if time.size else float("nan")
    paper_claims = {
        "paper_peak_acc_norm_m_s2": 31.0,
        "paper_peak_ang_norm_rad_s": 20.0,
        "paper_mean_ang_norm_rad_s": 7.01,
        "released_peak_acc_norm_m_s2": float(np.max(acc_norm)) if acc_norm.size else float("nan"),
        "released_peak_ang_norm_rad_s": float(np.max(ang_norm)) if ang_norm.size else float("nan"),
        "released_mean_ang_norm_rad_s": float(np.mean(ang_norm)) if ang_norm.size else float("nan"),
        "duration_s": duration,
    }
    paper_claims["peak_acc_abs_error"] = abs(
        paper_claims["released_peak_acc_norm_m_s2"] - paper_claims["paper_peak_acc_norm_m_s2"]
    )
    paper_claims["peak_ang_abs_error"] = abs(
        paper_claims["released_peak_ang_norm_rad_s"] - paper_claims["paper_peak_ang_norm_rad_s"]
    )
    paper_claims["mean_ang_abs_error"] = abs(
        paper_claims["released_mean_ang_norm_rad_s"] - paper_claims["paper_mean_ang_norm_rad_s"]
    )
    return rows, paper_claims


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Released-data statistical audit",
        "",
        "This audit adds confidence intervals and effect-size style summaries to Level A released-data outputs.",
        "It is released-data evidence only, not a new training or rollout result.",
        "",
        f"- Status: `{summary['status']}`",
        f"- Source CSV count: `{summary['metrics']['source_csv_count']}`",
        f"- Ablation comparison rows: `{summary['metrics']['ablation_comparison_rows']}`",
        f"- GRF CI rows: `{summary['metrics']['grf_ci_rows']}`",
        f"- IMU CI rows: `{summary['metrics']['imu_ci_rows']}`",
        f"- Best relative ablation improvement: `{summary['metrics']['best_relative_ablation_improvement']}`",
        f"- IMU paper-claim comparison: `{json.dumps(summary['metrics']['imu_paper_claim_comparison'], sort_keys=True)}`",
        "",
        "Outputs:",
        f"- `{summary['outputs']['json']}`",
        f"- `{summary['outputs']['ablation_csv']}`",
        f"- `{summary['outputs']['grf_csv']}`",
        f"- `{summary['outputs']['imu_csv']}`",
        f"- `{summary['outputs']['markdown']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_rows: list[dict[str, Any]] = []
    ablation_rows = ablation_statistics(source_rows)
    grf_rows = grf_statistics(source_rows)
    imu_rows, imu_claims = imu_statistics(source_rows)

    best_relative = max(ablation_rows, key=lambda row: row["relative_improvement"])
    json_path = OUT / "released_data_statistical_audit.json"
    ablation_csv = OUT / "released_ablation_effect_sizes.csv"
    grf_csv = OUT / "released_grf_confidence_intervals.csv"
    imu_csv = OUT / "released_imu_confidence_intervals.csv"
    source_tsv = OUT / "source_hashes.tsv"
    md_path = OUT / "released_data_statistical_audit.md"
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

    checks = {
        "all_expected_source_csvs_present": len(source_rows) == 10 and all(Path(r["absolute_path"]).is_file() for r in source_rows),
        "source_hashes_recorded": len(source_rows) == 10 and all(r["sha256"] for r in source_rows),
        "ablation_comparison_rows_30": len(ablation_rows) == 30,
        "grf_ci_rows_12": len(grf_rows) == 12,
        "imu_ci_rows_11": len(imu_rows) == 11,
        "all_confidence_intervals_finite": all(
            np.isfinite(float(row[key]))
            for rows in [grf_rows, imu_rows]
            for row in rows
            for key in ["mean", "std", "sem", "ci95_low", "ci95_high", "min", "max", "abs_max"]
        ),
        "all_ablation_effect_sizes_finite": all(np.isfinite(float(row["range_based_effect_size"])) for row in ablation_rows),
        "at_least_one_ablation_beats_baseline": any(row["best_beats_baseline"] for row in ablation_rows),
        "imu_norm_claim_metrics_present": all(np.isfinite(float(imu_claims[key])) for key in imu_claims),
        "does_not_claim_training": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "released_data_statistical_audit",
        "scope": "confidence intervals, effect-size style summaries, and paper-claim sanity checks for Level A released data",
        "metrics": {
            "source_csv_count": len(source_rows),
            "ablation_comparison_rows": len(ablation_rows),
            "grf_ci_rows": len(grf_rows),
            "imu_ci_rows": len(imu_rows),
            "best_relative_ablation_improvement": {
                "figure_id": best_relative["figure_id"],
                "scope": best_relative["scope"],
                "metric": best_relative["metric"],
                "baseline_experiment": best_relative["baseline_experiment"],
                "best_experiment": best_relative["best_experiment"],
                "relative_improvement": best_relative["relative_improvement"],
                "range_based_effect_size": best_relative["range_based_effect_size"],
            },
            "imu_paper_claim_comparison": imu_claims,
        },
        "checks": checks,
        "source_rows": source_rows,
        "not_a_replacement_for": [
            "new tracking training statistics",
            "paper Fig.5/Fig.6 rollout statistics",
            "paper user-study statistical notebook",
            "hardware deployment metrics",
        ],
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "released-data-only",
            "why_not_complete": (
                "The audit strengthens Level A released-data reporting with uncertainty summaries and paper-claim "
                "sanity checks, but it does not produce missing trained-policy, diffusion, hardware, or Fig.5/Fig.6 "
                "evaluation results."
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
    write_markdown(md_path, summary)
    print(json.dumps({"status": summary["status"], "json": str(json_path), "sources": len(source_rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
