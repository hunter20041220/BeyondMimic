#!/usr/bin/env python3
"""Create report-ready assets for the official-loop PPO checkpoint evaluation."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
EVAL_AUDIT = (
    ROOT
    / "res/tracking/g1_official_csv_loop_ppo_checkpoint_eval/"
    "tracking_g1_official_csv_loop_ppo_checkpoint_eval.json"
)
OUT = ROOT / "res/report_assets/official_csv_loop_ppo_checkpoint_eval"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_numeric(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else float("nan")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = load_json(EVAL_AUDIT)
    outputs = audit["outputs"]
    metrics = audit["run"]["metrics"]
    timeseries_path = Path(outputs["timeseries_csv"])
    gpu_metrics_path = Path(outputs["gpu_metrics_csv"])
    eval_metrics_path = Path(outputs["metrics_json"])
    df = pd.read_csv(timeseries_path)
    gpu = pd.read_csv(gpu_metrics_path)
    gpu.columns = [col.strip() for col in gpu.columns]
    for col in ["memory.used [MiB]", "memory.total [MiB]", "utilization.gpu [%]", "power.draw [W]"]:
        if col in gpu:
            gpu[col] = gpu[col].map(clean_numeric)
    if "index" in gpu:
        gpu["index"] = gpu["index"].astype(int)

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    axes[0].plot(df["step"], df["error_anchor_pos"], label="anchor pos", color="#2563eb")
    axes[0].plot(df["step"], df["error_body_pos"], label="body pos", color="#16a34a")
    axes[0].plot(df["step"], df["error_joint_pos"], label="joint pos", color="#dc2626")
    axes[0].set_ylabel("Mean tracking error")
    axes[0].set_title("Official-loop PPO checkpoint tracking errors")
    axes[0].legend(loc="upper right")

    axes[1].plot(df["step"], df["error_anchor_lin_vel"], label="anchor lin vel", color="#7c3aed")
    axes[1].plot(df["step"], df["error_body_lin_vel"], label="body lin vel", color="#ea580c")
    axes[1].plot(df["step"], df["error_joint_vel"], label="joint vel", color="#0891b2")
    axes[1].set_xlabel("Evaluation step")
    axes[1].set_ylabel("Mean velocity error")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    tracking_error_png = OUT / "tracking_error_timeseries.png"
    fig.savefig(tracking_error_png, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.0), sharex=True)
    axes[0].plot(df["step"], df["reward_mean"], label="mean", color="#059669")
    axes[0].fill_between(df["step"], df["reward_min"], df["reward_max"], color="#34d399", alpha=0.25, label="min/max")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Reward envelope and termination counts")
    axes[0].legend(loc="upper right")
    axes[1].bar(df["step"], df["done_count"], width=1.0, label="done count", color="#f97316")
    axes[1].bar(df["step"], df["timeout_count"], width=1.0, label="timeout count", color="#64748b", alpha=0.65)
    axes[1].set_xlabel("Evaluation step")
    axes[1].set_ylabel("Env count")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    reward_done_png = OUT / "reward_done_timeseries.png"
    fig.savefig(reward_done_png, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.0), sharex=True)
    for gpu_index, group in gpu.groupby("index"):
        axes[0].plot(range(len(group)), group["utilization.gpu [%]"], label=f"GPU {gpu_index}")
        axes[1].plot(range(len(group)), group["memory.used [MiB]"] / 1024.0, label=f"GPU {gpu_index}")
    axes[0].set_ylabel("GPU util (%)")
    axes[0].set_title("GPU telemetry during PPO checkpoint evaluation")
    axes[0].legend(loc="upper right")
    axes[1].set_xlabel("Telemetry sample")
    axes[1].set_ylabel("Memory used (GiB)")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    gpu_png = OUT / "gpu_usage_eval.png"
    fig.savefig(gpu_png, dpi=180)
    plt.close(fig)

    summary_rows = [
        {
            "metric": "status",
            "value": audit["status"],
            "claim_level": "local_virtual_resource_adjusted",
        },
        {"metric": "loaded_iteration", "value": metrics["loaded_iteration"], "claim_level": "local_checkpoint"},
        {"metric": "num_envs", "value": metrics["num_envs"], "claim_level": "local_eval"},
        {"metric": "eval_steps", "value": metrics["eval_steps"], "claim_level": "local_eval"},
        {"metric": "total_env_steps", "value": metrics["total_env_steps"], "claim_level": "local_eval"},
        {"metric": "reward_mean", "value": metrics["reward"]["mean_over_steps"]["mean"], "claim_level": "local_eval"},
        {"metric": "done_count_total", "value": metrics["done_count_total"], "claim_level": "local_eval"},
        {"metric": "timeout_count_total", "value": metrics["timeout_count_total"], "claim_level": "local_eval"},
        {
            "metric": "error_anchor_pos_mean",
            "value": metrics["motion_metrics"]["error_anchor_pos"]["mean"],
            "claim_level": "local_eval",
        },
        {
            "metric": "error_body_pos_mean",
            "value": metrics["motion_metrics"]["error_body_pos"]["mean"],
            "claim_level": "local_eval",
        },
        {
            "metric": "error_joint_pos_mean",
            "value": metrics["motion_metrics"]["error_joint_pos"]["mean"],
            "claim_level": "local_eval",
        },
    ]
    summary_csv = OUT / "ppo_checkpoint_eval_summary.csv"
    write_csv(summary_csv, summary_rows, ["metric", "value", "claim_level"])

    gpu_rows = []
    for gpu_index, group in gpu.groupby("index"):
        gpu_rows.append(
            {
                "gpu_index": int(gpu_index),
                "samples": int(len(group)),
                "mean_utilization_gpu_percent": float(group["utilization.gpu [%]"].mean()),
                "peak_memory_used_gib": float((group["memory.used [MiB]"] / 1024.0).max()),
                "mean_power_draw_w": float(group["power.draw [W]"].mean()),
            }
        )
    gpu_summary_csv = OUT / "ppo_checkpoint_eval_gpu_summary.csv"
    write_csv(
        gpu_summary_csv,
        gpu_rows,
        ["gpu_index", "samples", "mean_utilization_gpu_percent", "peak_memory_used_gib", "mean_power_draw_w"],
    )

    readme = OUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Official-Loop PPO Checkpoint Evaluation Assets",
                "",
                "These plots and tables summarize the local virtual PPO checkpoint evaluation on the official-loop",
                "motion chain. They are intended for the English reading report and presentation slides.",
                "",
                "## Source",
                "",
                f"- Eval audit: `{EVAL_AUDIT}`",
                f"- Eval metrics: `{eval_metrics_path}`",
                f"- Timeseries: `{timeseries_path}`",
                f"- GPU telemetry: `{gpu_metrics_path}`",
                f"- Checkpoint: `{metrics['checkpoint']}`",
                f"- Status: `{audit['status']}`",
                "",
                "## Assets",
                "",
                f"- `{tracking_error_png}`",
                f"- `{reward_done_png}`",
                f"- `{gpu_png}`",
                f"- `{summary_csv}`",
                f"- `{gpu_summary_csv}`",
                "",
                "## Claim Level",
                "",
                "local_virtual_resource_adjusted / approximately comparable engineering evidence. This is not",
                "unpatched official BeyondMimic PPO evaluation, not paper-scale teacher training, not Fig. 5/Fig. 6",
                "guided diffusion, and not real-robot validation.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    asset_summary = {
        "status": "ok",
        "experiment_type": "official_csv_loop_ppo_checkpoint_eval_report_assets",
        "source_eval_audit": str(EVAL_AUDIT),
        "source_timeseries_csv": str(timeseries_path),
        "source_gpu_metrics_csv": str(gpu_metrics_path),
        "claim_level": "local_virtual_resource_adjusted_report_asset",
        "limitation": (
            "Uses a local iteration-299 PPO checkpoint trained/evaluated through the enriched-USD runtime patch; "
            "not unpatched official paper-level PPO evaluation and not real-robot evidence."
        ),
        "metrics": {
            "num_envs": metrics["num_envs"],
            "eval_steps": metrics["eval_steps"],
            "total_env_steps": metrics["total_env_steps"],
            "done_count_total": metrics["done_count_total"],
            "timeout_count_total": metrics["timeout_count_total"],
            "error_anchor_pos_mean": metrics["motion_metrics"]["error_anchor_pos"]["mean"],
            "error_body_pos_mean": metrics["motion_metrics"]["error_body_pos"]["mean"],
            "error_joint_pos_mean": metrics["motion_metrics"]["error_joint_pos"]["mean"],
        },
        "assets": {
            "tracking_error_timeseries_png": str(tracking_error_png),
            "reward_done_timeseries_png": str(reward_done_png),
            "gpu_usage_eval_png": str(gpu_png),
            "summary_csv": str(summary_csv),
            "gpu_summary_csv": str(gpu_summary_csv),
            "summary_md": str(readme),
        },
        "checks": {
            "eval_status_ok": audit["status"] == "ok_official_csv_loop_ppo_checkpoint_eval_completed",
            "timeseries_has_299_rows": len(df) == 299,
            "summary_csv_exists": summary_csv.is_file(),
            "gpu_summary_csv_exists": gpu_summary_csv.is_file(),
            "png_assets_exist": tracking_error_png.is_file() and reward_done_png.is_file() and gpu_png.is_file(),
            "does_not_claim_paper_level_eval": metrics["paper_level_tracking_eval"] is False,
            "does_not_claim_official_unpatched_output": metrics["official_csv_to_npz_output"] is False,
            "does_not_claim_real_robot": True,
        },
    }
    asset_json = OUT / "official_csv_loop_ppo_checkpoint_eval_assets.json"
    asset_json.write_text(json.dumps(asset_summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(asset_json), "assets": asset_summary["assets"]}, sort_keys=True))


if __name__ == "__main__":
    main()
