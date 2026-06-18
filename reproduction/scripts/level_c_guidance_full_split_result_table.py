#!/usr/bin/env python3
"""Summarize full-split public-data guidance results into tables and figures."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/guidance_full_split_result_table"
OFFLINE_JSON = (
    ROOT
    / "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/"
    / "level_c_lafan1_paper_arch_guidance_eval.json"
)
REVERSE_JSON = (
    ROOT
    / "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/"
    / "level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.json"
)
TASK_ORDER = ["joystick", "waypoint", "obstacle_avoidance", "inpainting", "composed_objectives"]
LABELS = {
    "joystick": "Joystick",
    "waypoint": "Waypoint",
    "obstacle_avoidance": "Obstacle",
    "inpainting": "Inpainting",
    "composed_objectives": "Composed",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def save_bar_figure(rows: list[dict[str, Any]], stem: Path) -> list[str]:
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.8), sharex=True, constrained_layout=True)
    x = np.arange(len(TASK_ORDER))
    width = 0.36
    colors = {"offline": "#3977b7", "reverse": "#b85c38"}

    for offset, mode in [(-width / 2, "offline"), (width / 2, "reverse")]:
        mode_df = df[df["mode"] == mode].set_index("task").loc[TASK_ORDER]
        axes[0].bar(
            x + offset,
            mode_df["mean_best_cost_delta"].to_numpy(dtype=float),
            width=width,
            label=mode,
            color=colors[mode],
        )
        axes[1].bar(
            x + offset,
            mode_df["positive_best_cost_delta_fraction"].to_numpy(dtype=float),
            width=width,
            label=mode,
            color=colors[mode],
        )

    axes[0].axhline(0.0, color="#333333", linewidth=0.9)
    axes[0].set_ylabel("Mean best cost delta")
    axes[0].set_title("Full-split public-data guidance cost change")
    axes[0].legend(frameon=False, ncols=2)
    axes[1].set_ylabel("Positive delta fraction")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xticks(x, [LABELS[t] for t in TASK_ORDER], rotation=20, ha="right")
    axes[1].axhline(0.5, color="#777777", linewidth=0.8, linestyle="--")
    for ax in axes:
        ax.grid(axis="y", color="#d8d8d8", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    outputs = []
    for ext in ["pdf", "svg", "png"]:
        path = stem.with_suffix(f".{ext}")
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = 220
        fig.savefig(path, **kwargs)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def offline_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task in TASK_ORDER:
        summary = data["task_summaries"][task]
        rows.append(
            {
                "mode": "offline",
                "task": task,
                "task_label": LABELS[task],
                "window_count": int(summary["window_count"]),
                "scale_count": int(summary["scale_count"]),
                "row_count": int(summary["window_count"] * summary["scale_count"]),
                "mean_base_cost": float(summary["mean_base_cost"]),
                "mean_best_cost": float(summary["mean_best_cost"]),
                "mean_best_cost_delta": float(summary["mean_cost_delta"]),
                "median_best_cost_delta": float("nan"),
                "positive_best_cost_delta_fraction": 1.0 if summary["all_best_costs_improve"] else float("nan"),
                "primary_improved_count": int(summary["best_rows_primary_improved_count"]),
                "all_best_costs_improve": bool(summary["all_best_costs_improve"]),
                "mean_best_gradient_norm": float(summary["mean_gradient_norm"]),
            }
        )
    return rows


def reverse_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for task in TASK_ORDER:
        summary = data["task_summaries"][task]
        rows.append(
            {
                "mode": "reverse",
                "task": task,
                "task_label": LABELS[task],
                "window_count": int(summary["window_count"]),
                "scale_count": int(summary["scale_count"]),
                "row_count": int(summary["window_count"] * summary["scale_count"]),
                "mean_base_cost": float("nan"),
                "mean_best_cost": float("nan"),
                "mean_best_cost_delta": float(summary["mean_best_cost_delta"]),
                "median_best_cost_delta": float(summary["median_best_cost_delta"]),
                "positive_best_cost_delta_fraction": float(summary["positive_best_cost_delta_fraction"]),
                "primary_improved_count": int(summary["best_rows_primary_improved_count"]),
                "all_best_costs_improve": bool(summary["all_best_costs_improve"]),
                "mean_best_gradient_norm": float(summary["mean_best_gradient_norm"]),
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    offline = load_json(OFFLINE_JSON)
    reverse = load_json(REVERSE_JSON)
    rows = offline_rows(offline) + reverse_rows(reverse)
    df = pd.DataFrame(rows)

    json_path = OUT / "level_c_guidance_full_split_result_table.json"
    tsv_path = OUT / "level_c_guidance_full_split_result_table.tsv"
    csv_path = OUT / "level_c_guidance_full_split_result_table.csv"
    figure_paths = save_bar_figure(rows, OUT / "level_c_guidance_full_split_cost_delta")
    write_tsv(tsv_path, rows)
    df.to_csv(csv_path, index=False)

    mode_summary = {}
    for mode in ["offline", "reverse"]:
        mode_rows = [row for row in rows if row["mode"] == mode]
        mode_summary[mode] = {
            "task_count": len(mode_rows),
            "total_rows": int(sum(row["row_count"] for row in mode_rows)),
            "tasks_with_all_best_costs_improve": int(sum(row["all_best_costs_improve"] for row in mode_rows)),
            "mean_positive_best_cost_delta_fraction": float(
                np.nanmean([row["positive_best_cost_delta_fraction"] for row in mode_rows])
            ),
            "mean_best_cost_delta_by_task": {
                row["task"]: float(row["mean_best_cost_delta"]) for row in mode_rows
            },
        }

    checks = {
        "offline_status_ok": offline.get("status") == "ok",
        "reverse_status_ok": reverse.get("status") == "ok",
        "two_modes_recorded": sorted({row["mode"] for row in rows}) == ["offline", "reverse"],
        "five_tasks_per_mode": all(
            sorted(row["task"] for row in rows if row["mode"] == mode) == sorted(TASK_ORDER)
            for mode in ["offline", "reverse"]
        ),
        "offline_row_count_matches_source": mode_summary["offline"]["total_rows"] == int(offline["row_count"]),
        "reverse_row_count_matches_source": mode_summary["reverse"]["total_rows"] == int(reverse["metrics"]["row_count"]),
        "all_mean_cost_deltas_finite": all(finite(row["mean_best_cost_delta"]) for row in rows),
        "all_primary_improved_counts_nonzero": all(row["primary_improved_count"] > 0 for row in rows),
        "offline_records_all_cost_improve": mode_summary["offline"]["tasks_with_all_best_costs_improve"] == 5,
        "reverse_records_mixed_cost_outcome": mode_summary["reverse"]["tasks_with_all_best_costs_improve"] == 2,
        "figure_files_written": all(Path(path).is_file() and Path(path).stat().st_size > 0 for path in figure_paths),
        "does_not_claim_closed_loop_rollout": True,
        "does_not_claim_fig5_fig6_reproduction": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "level_c_guidance_full_split_result_table",
        "scope": (
            "Paper-formula task-guidance result table over the full public-data validation/test splits, comparing "
            "one-shot offline guidance with batched reverse-denoising guidance for the symmetry-augmented checkpoint."
        ),
        "inputs": {
            "offline_guidance_json": str(OFFLINE_JSON),
            "reverse_guidance_json": str(REVERSE_JSON),
        },
        "metrics": {
            "row_count": len(rows),
            "mode_count": 2,
            "task_count": len(TASK_ORDER),
            "offline_source_rows": int(offline["row_count"]),
            "reverse_source_rows": int(reverse["metrics"]["row_count"]),
            "reverse_min_after_reserve_used_mb": float(reverse["metrics"]["min_after_reserve_used_mb"]),
        },
        "mode_summary": mode_summary,
        "rows": rows,
        "checks": checks,
        "outputs": {
            "json": str(json_path),
            "tsv": str(tsv_path),
            "csv": str(csv_path),
            "figures": figure_paths,
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "public_data_full_split_guidance_summary",
            "why_not_complete": (
                "This table and figure summarize full-split public-data paper-formula guidance metrics. They are not "
                "closed-loop success/failure videos, Fig. 5/Fig. 6 reproduction, TensorRT deployment, or real robot results."
            ),
        },
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(json_path), "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
