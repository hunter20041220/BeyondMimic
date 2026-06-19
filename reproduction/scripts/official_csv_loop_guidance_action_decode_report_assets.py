#!/usr/bin/env python3
"""Create report-ready plots for official-loop guided VAE action decoding."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
EVAL_JSON = (
    ROOT
    / "res/level_c/official_csv_loop_guidance_vae_action_decode_eval/"
    "level_c_official_csv_loop_guidance_vae_action_decode_eval.json"
)
OUT = ROOT / "res/report_assets/official_csv_loop_guidance_vae_action_decode"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads(EVAL_JSON.read_text(encoding="utf-8"))
    worker = summary["worker_summary"]
    rows_path = Path(worker["outputs"]["rows_tsv"])
    rows = read_rows(rows_path)
    tasks = ["velocity_command", "latent_smoothness", "latent_magnitude", "composed"]
    splits = ["validation", "test"]

    delta_by_task = []
    mse_delta_by_task = []
    for task in tasks:
        task_rows = [row for row in rows if row["task"] == task]
        delta_by_task.append(np.mean([as_float(row, "guided_base_action_l2_mean") for row in task_rows]))
        mse_delta_by_task.append(np.mean([as_float(row, "guided_minus_base_teacher_mse") for row in task_rows]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(tasks))
    axes[0].bar(x, delta_by_task, color="#3b82f6")
    axes[0].set_xticks(x, tasks, rotation=20, ha="right")
    axes[0].set_ylabel("Mean L2(action_guided - action_base)")
    axes[0].set_title("Guided vs base decoded action change")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, mse_delta_by_task, color="#10b981")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(x, tasks, rotation=20, ha="right")
    axes[1].set_ylabel("Guided - base teacher-action MSE")
    axes[1].set_title("Teacher-action MSE delta after guidance")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    comparison_png = OUT / "guided_vs_base_action_decode_metrics.png"
    fig.savefig(comparison_png, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.35
    for idx, split in enumerate(splits):
        values = [
            as_float(next(row for row in rows if row["task"] == task and row["split"] == split), "guided_teacher_action_mse")
            for task in tasks
        ]
        ax.bar(x + (idx - 0.5) * width, values, width=width, label=split)
    ax.set_xticks(x, tasks, rotation=20, ha="right")
    ax.set_ylabel("Guided decoded action vs teacher MSE")
    ax.set_title("Guided decoded action MSE by split")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    split_png = OUT / "guided_action_teacher_mse_by_split.png"
    fig.savefig(split_png, dpi=180)
    plt.close(fig)

    metrics_csv = OUT / "guided_action_decode_metrics.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "task",
            "split",
            "scale",
            "window_count",
            "action_count",
            "base_teacher_action_mse",
            "guided_teacher_action_mse",
            "guided_minus_base_teacher_mse",
            "guided_base_action_l2_mean",
            "finite",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    summary_md = OUT / "README.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Official-Loop Guidance VAE Action Decode Assets",
                "",
                "These assets visualize the offline bridge from guided state-latent denoiser outputs to decoded",
                "29D VAE actions. They are report/PPT assets, not paper-level closed-loop rollout evidence.",
                "",
                "## Source",
                "",
                f"- Eval JSON: `{EVAL_JSON}`",
                f"- Rows TSV: `{rows_path}`",
                f"- Status: `{summary['status']}`",
                f"- Total windows: `{worker['metrics']['total_windows']}`",
                f"- Tasks with finite decoded actions: `{worker['metrics']['tasks_with_finite_actions']}`",
                "",
                "## Assets",
                "",
                f"- `{comparison_png}`",
                f"- `{split_png}`",
                f"- `{metrics_csv}`",
                "",
                "## Claim Level",
                "",
                "qualitative_only / offline action-decode gate. This does not claim IsaacLab closed-loop guidance,",
                "Fig. 5/Fig. 6 reproduction, TensorRT deployment, or real robot validation.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    asset_summary = {
        "status": "ok",
        "source_eval_json": str(EVAL_JSON),
        "claim_level": "qualitative_only_offline_action_decode_asset",
        "limitation": "Not closed-loop IsaacLab guidance, not Fig.5/Fig.6 paper-level video/metric evidence.",
        "assets": {
            "guided_vs_base_action_decode_metrics_png": str(comparison_png),
            "guided_action_teacher_mse_by_split_png": str(split_png),
            "metrics_csv": str(metrics_csv),
            "summary_md": str(summary_md),
        },
        "checks": {
            "eval_status_ok": summary["status"] == "ok_official_csv_loop_guidance_vae_action_decode_eval",
            "png_assets_exist": comparison_png.is_file() and split_png.is_file(),
            "metrics_csv_exists": metrics_csv.is_file(),
            "summary_md_exists": summary_md.is_file(),
            "does_not_claim_closed_loop": True,
        },
    }
    asset_json = OUT / "official_csv_loop_guidance_vae_action_decode_assets.json"
    asset_json.write_text(json.dumps(asset_summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(asset_json), "assets": asset_summary["assets"]}, sort_keys=True))


if __name__ == "__main__":
    main()
