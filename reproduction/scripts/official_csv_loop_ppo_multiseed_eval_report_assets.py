#!/usr/bin/env python3
"""Create report-ready assets for the official-csv-loop PPO multi-seed eval."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
AUDIT = (
    ROOT
    / "res/tracking/g1_official_csv_loop_ppo_checkpoint_multiseed_eval/"
    "tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval.json"
)
OUT = ROOT / "res/report_assets/official_csv_loop_ppo_checkpoint_multiseed_eval"


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
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_timeseries(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frames = []
    for row in rows:
        copied_path = (
            ROOT
            / "res/tracking/g1_official_csv_loop_ppo_checkpoint_multiseed_eval"
            / f"seed_{row['seed']}"
            / f"seed_{row['seed']}_eval_timeseries.csv"
        )
        path = copied_path if copied_path.is_file() else Path(row["timeseries_csv"])
        df = pd.read_csv(path)
        df["seed"] = int(row["seed"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_gpu_metrics(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frames = []
    for row in rows:
        copied_path = (
            ROOT
            / "res/tracking/g1_official_csv_loop_ppo_checkpoint_multiseed_eval"
            / f"seed_{row['seed']}"
            / f"seed_{row['seed']}_gpu_metrics.csv"
        )
        path = copied_path if copied_path.is_file() else Path(row["gpu_metrics_csv"])
        df = pd.read_csv(path)
        df.columns = [col.strip() for col in df.columns]
        for col in ["memory.used [MiB]", "memory.total [MiB]", "utilization.gpu [%]", "power.draw [W]"]:
            if col in df:
                df[col] = df[col].map(clean_numeric)
        df["seed"] = int(row["seed"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = load_json(AUDIT)
    rows = audit["rows"]
    aggregate = audit["aggregate"]
    ts = load_timeseries(rows)
    gpu = load_gpu_metrics(rows)

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    for seed, group in ts.groupby("seed"):
        axes[0].plot(group["step"], group["reward_mean"], label=f"seed {seed}")
        axes[1].plot(group["step"], group["error_body_pos"], label=f"seed {seed}")
    axes[0].set_title("Official-csv-loop PPO checkpoint multi-seed reward")
    axes[0].set_ylabel("Reward mean")
    axes[0].legend(loc="upper right")
    axes[1].set_xlabel("Evaluation step")
    axes[1].set_ylabel("Body position error")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    reward_error_png = OUT / "multiseed_reward_body_error_timeseries.png"
    fig.savefig(reward_error_png, dpi=180)
    plt.close(fig)

    metric_rows = []
    for metric in [
        "reward_mean",
        "error_anchor_pos_mean",
        "error_body_pos_mean",
        "error_joint_pos_mean",
        "done_count_total",
        "timeout_count_total",
    ]:
        item = aggregate[metric]
        metric_rows.append(
            {
                "metric": metric,
                "mean": item["mean"],
                "std": item["std"],
                "min": item["min"],
                "max": item["max"],
                "count": item["count"],
                "claim_level": "local_virtual_multiseed_tracking_eval",
            }
        )
    aggregate_csv = OUT / "multiseed_eval_aggregate_summary.csv"
    write_csv(aggregate_csv, metric_rows, ["metric", "mean", "std", "min", "max", "count", "claim_level"])

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    plot_metrics = ["reward_mean", "error_body_pos_mean", "error_joint_pos_mean"]
    means = [float(aggregate[name]["mean"]) for name in plot_metrics]
    stds = [float(aggregate[name]["std"]) for name in plot_metrics]
    ax.bar(plot_metrics, means, yerr=stds, capsize=6, color=["#059669", "#2563eb", "#dc2626"])
    ax.set_title("Multi-seed local checkpoint eval aggregate")
    ax.set_ylabel("Mean across seeds")
    fig.tight_layout()
    aggregate_png = OUT / "multiseed_eval_aggregate_bars.png"
    fig.savefig(aggregate_png, dpi=180)
    plt.close(fig)

    gpu_rows = []
    if not gpu.empty:
        for seed, group in gpu.groupby("seed"):
            gpu_rows.append(
                {
                    "seed": int(seed),
                    "gpu_index": int(pd.to_numeric(group["index"]).mode().iloc[0]),
                    "samples": int(len(group)),
                    "mean_utilization_gpu_percent": float(group["utilization.gpu [%]"].mean()),
                    "peak_memory_used_gib": float((group["memory.used [MiB]"] / 1024.0).max()),
                    "mean_power_draw_w": float(group["power.draw [W]"].mean()),
                }
            )
    gpu_summary_csv = OUT / "multiseed_eval_gpu_summary.csv"
    write_csv(
        gpu_summary_csv,
        gpu_rows,
        ["seed", "gpu_index", "samples", "mean_utilization_gpu_percent", "peak_memory_used_gib", "mean_power_draw_w"],
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.0), sharex=True)
    for seed, group in gpu.groupby("seed"):
        axes[0].plot(range(len(group)), group["utilization.gpu [%]"], label=f"seed {seed}")
        axes[1].plot(range(len(group)), group["memory.used [MiB]"] / 1024.0, label=f"seed {seed}")
    axes[0].set_title("GPU telemetry during multi-seed eval")
    axes[0].set_ylabel("GPU util (%)")
    axes[0].legend(loc="upper right")
    axes[1].set_xlabel("Telemetry sample")
    axes[1].set_ylabel("Memory used (GiB)")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    gpu_png = OUT / "multiseed_eval_gpu_usage.png"
    fig.savefig(gpu_png, dpi=180)
    plt.close(fig)

    readme = OUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Official-CSV-Loop PPO Multi-Seed Evaluation Assets",
                "",
                "These plots and tables summarize three full 299-step local virtual evaluations of the",
                "iteration-299 official-csv-loop PPO checkpoint.",
                "",
                "## Source",
                "",
                f"- Multi-seed audit: `{AUDIT}`",
                f"- Rows CSV: `{audit['outputs']['rows_csv']}`",
                f"- Seeds: `{audit['config']['seeds']}`",
                f"- GPU assignment: `{audit['config']['gpu_assignment']}`",
                f"- Total env steps: `{audit['metrics']['total_env_steps']}`",
                "",
                "## Key Aggregate Metrics",
                "",
                f"- reward_mean: `{aggregate['reward_mean']['mean']}` +/- `{aggregate['reward_mean']['std']}`",
                f"- body_pos_error_mean: `{aggregate['error_body_pos_mean']['mean']}` +/- `{aggregate['error_body_pos_mean']['std']}`",
                f"- joint_pos_error_mean: `{aggregate['error_joint_pos_mean']['mean']}` +/- `{aggregate['error_joint_pos_mean']['std']}`",
                f"- done_count_total: `{aggregate['done_count_total']['mean']}` +/- `{aggregate['done_count_total']['std']}`",
                "",
                "## Assets",
                "",
                f"- `{reward_error_png}`",
                f"- `{aggregate_png}`",
                f"- `{gpu_png}`",
                f"- `{aggregate_csv}`",
                f"- `{gpu_summary_csv}`",
                "",
                "## Claim Level",
                "",
                "local_virtual_multiseed_tracking_eval. This is useful stability evidence for the reading report,",
                "but it is not the unpatched official BeyondMimic tracking teacher, not paper-scale PPO evaluation,",
                "not DAgger, not Fig. 5/Fig. 6 guided diffusion, and not real-robot validation.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    asset_summary = {
        "status": "ok",
        "experiment_type": "official_csv_loop_ppo_checkpoint_multiseed_eval_report_assets",
        "source_audit": str(AUDIT),
        "claim_level": "local_virtual_multiseed_tracking_eval_report_asset",
        "metrics": {
            "seed_count": audit["metrics"]["seed_count"],
            "ok_seed_count": audit["metrics"]["ok_seed_count"],
            "total_env_steps": audit["metrics"]["total_env_steps"],
            "reward_mean": aggregate["reward_mean"],
            "error_body_pos_mean": aggregate["error_body_pos_mean"],
            "error_joint_pos_mean": aggregate["error_joint_pos_mean"],
        },
        "assets": {
            "reward_body_error_timeseries_png": str(reward_error_png),
            "aggregate_bars_png": str(aggregate_png),
            "gpu_usage_png": str(gpu_png),
            "aggregate_summary_csv": str(aggregate_csv),
            "gpu_summary_csv": str(gpu_summary_csv),
            "summary_md": str(readme),
        },
        "checks": {
            "audit_status_ok": audit["status"] == "ok_official_csv_loop_ppo_checkpoint_multiseed_eval_completed",
            "three_seeds_completed": audit["metrics"]["ok_seed_count"] == 3,
            "total_env_steps_459264": audit["metrics"]["total_env_steps"] == 459264,
            "timeseries_rows_897": len(ts) == 897,
            "assets_exist": all(
                path.is_file() for path in [reward_error_png, aggregate_png, gpu_png, aggregate_csv, gpu_summary_csv]
            ),
            "does_not_claim_paper_level_eval": True,
            "does_not_claim_unpatched_official_asset": True,
            "does_not_claim_real_robot": True,
            "does_not_claim_goal_complete": True,
        },
        "limitation": (
            "The run evaluates a local iteration-299 checkpoint trained/evaluated through the enriched-USD runtime "
            "patch. It is not the paper-scale official BeyondMimic tracking teacher or official DAgger evidence."
        ),
    }
    asset_json = OUT / "official_csv_loop_ppo_checkpoint_multiseed_eval_assets.json"
    asset_json.write_text(json.dumps(asset_summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(asset_json), "assets": asset_summary["assets"]}, sort_keys=True))


if __name__ == "__main__":
    main()
