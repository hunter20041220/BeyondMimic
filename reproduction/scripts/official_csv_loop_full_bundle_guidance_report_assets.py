#!/usr/bin/env python3
"""Create report assets for full-bundle offline state-latent guidance."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
GUIDANCE_JSON = (
    ROOT
    / "res/level_c/official_csv_loop_full_bundle_state_latent_guidance_eval/"
    "level_c_official_csv_loop_full_bundle_state_latent_guidance_eval.json"
)
OUT = ROOT / "res/report_assets/official_csv_loop_full_bundle_guidance"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    guidance = load_json(GUIDANCE_JSON)
    worker = guidance["worker_summary"]
    rows = worker["rows"]
    task_summaries = worker["task_summaries"]

    best_rows = []
    for task, summary in sorted(task_summaries.items()):
        for split, row in sorted(summary["splits"].items()):
            best_rows.append(
                {
                    "task": task,
                    "split": split,
                    "best_scale": row["scale"],
                    "base_cost_mean": row["base_cost_mean"],
                    "guided_cost_mean": row["guided_cost_mean"],
                    "cost_delta_mean": row["cost_delta_mean"],
                    "positive_delta_fraction": row["positive_delta_fraction"],
                    "gradient_norm_mean": row["gradient_norm_mean"],
                    "window_count": row["window_count"],
                }
            )

    best_csv = OUT / "full_bundle_guidance_best_rows.csv"
    write_csv(
        best_csv,
        best_rows,
        [
            "task",
            "split",
            "best_scale",
            "base_cost_mean",
            "guided_cost_mean",
            "cost_delta_mean",
            "positive_delta_fraction",
            "gradient_norm_mean",
            "window_count",
        ],
    )

    rows_csv = OUT / "full_bundle_guidance_scale_rows.csv"
    write_csv(
        rows_csv,
        rows,
        [
            "task",
            "split",
            "scale",
            "window_count",
            "base_cost_mean",
            "guided_cost_mean",
            "cost_delta_mean",
            "positive_delta_fraction",
            "gradient_norm_mean",
            "finite",
        ],
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5.4))
    labels = [f"{row['task']}\n{row['split']}" for row in best_rows]
    values = [row["cost_delta_mean"] for row in best_rows]
    colors = ["#2563eb" if row["split"] == "validation" else "#f59e0b" for row in best_rows]
    ax.bar(range(len(best_rows)), values, color=colors)
    ax.set_xticks(range(len(best_rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Best cost delta (base - guided)")
    ax.set_title("Full-bundle offline guidance best-scale improvements")
    fig.tight_layout()
    best_png = OUT / "full_bundle_guidance_best_cost_delta.png"
    fig.savefig(best_png, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharex=True)
    axes_flat = axes.reshape(-1)
    for ax, task in zip(axes_flat, sorted(task_summaries)):
        for split, color in [("validation", "#2563eb"), ("test", "#f59e0b")]:
            subset = [row for row in rows if row["task"] == task and row["split"] == split]
            subset = sorted(subset, key=lambda item: item["scale"])
            ax.plot(
                [row["scale"] for row in subset],
                [row["cost_delta_mean"] for row in subset],
                marker="o",
                label=split,
                color=color,
            )
        ax.set_title(task)
        ax.set_ylabel("Cost delta")
        ax.legend(loc="best")
    for ax in axes_flat[-2:]:
        ax.set_xlabel("Guidance scale")
    fig.suptitle("Full-bundle offline guidance scale response", y=0.995)
    fig.tight_layout()
    scale_png = OUT / "full_bundle_guidance_scale_response.png"
    fig.savefig(scale_png, dpi=180)
    plt.close(fig)

    summary = {
        "status": "ok",
        "claim_level": "local_virtual_full_bundle_offline_guidance_report_asset",
        "source_guidance_json": str(GUIDANCE_JSON),
        "metrics": {
            "total_selected_windows": worker["metrics"]["total_selected_windows"],
            "row_count": worker["metrics"]["row_count"],
            "task_count": worker["metrics"]["task_count"],
            "tasks_with_all_best_costs_improve": worker["metrics"]["tasks_with_all_best_costs_improve"],
            "tasks_with_nonzero_best_gradients": worker["metrics"]["tasks_with_nonzero_best_gradients"],
            "selected_split_counts": worker["settings"]["selected_split_counts"],
            "task_mean_best_cost_delta": {
                task: task_summary["mean_best_cost_delta"] for task, task_summary in task_summaries.items()
            },
        },
        "assets": {
            "best_cost_delta_png": str(best_png),
            "scale_response_png": str(scale_png),
            "best_rows_csv": str(best_csv),
            "scale_rows_csv": str(rows_csv),
            "summary_md": str(OUT / "README.md"),
        },
        "checks": {
            "guidance_status_ok": guidance["status"]
            == "ok_official_csv_loop_full_bundle_state_latent_guidance_eval",
            "full_split_evaluated": worker["settings"]["selected_split_counts"] == {"validation": 28569, "test": 28570},
            "all_tasks_improve": worker["metrics"]["tasks_with_all_best_costs_improve"] == 4,
            "png_assets_exist": best_png.is_file() and scale_png.is_file(),
            "csv_assets_exist": best_csv.is_file() and rows_csv.is_file(),
            "does_not_claim_closed_loop_guidance": True,
            "does_not_claim_fig5_fig6": True,
            "does_not_claim_real_robot": True,
        },
        "limitation": (
            "These assets summarize offline task-cost guidance over local full-bundle denoiser outputs. They are not "
            "closed-loop IsaacLab rollouts, paper Fig. 5/Fig. 6 videos, TensorRT deployment, or real-robot evidence."
        ),
    }

    readme = OUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Full-Bundle Offline Guidance Assets",
                "",
                "These assets summarize offline guidance over the local full-bundle state-latent denoiser.",
                "",
                "## Key Metrics",
                "",
                f"- Total selected windows: `{summary['metrics']['total_selected_windows']}`",
                f"- Selected split counts: `{summary['metrics']['selected_split_counts']}`",
                f"- Tasks with all best costs improved: `{summary['metrics']['tasks_with_all_best_costs_improve']}`",
                "",
                "## Claim Level",
                "",
                "local_virtual_full_bundle_offline_guidance_report_asset. This is not closed-loop guidance,",
                "not paper Fig. 5/Fig. 6 reproduction, and not real-robot evidence.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary["checks"]["summary_md_exists"] = readme.is_file()
    summary_json = OUT / "official_csv_loop_full_bundle_guidance_report_assets.json"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(summary_json), "assets": summary["assets"]}, sort_keys=True))


if __name__ == "__main__":
    main()
