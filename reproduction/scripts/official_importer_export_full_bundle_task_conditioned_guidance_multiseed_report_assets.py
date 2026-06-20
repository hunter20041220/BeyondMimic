#!/usr/bin/env python3
"""Create report assets for importer-export multi-seed task guidance."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SUMMARY_JSON = (
    ROOT
    / "res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval/"
    "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json"
)
OUT = ROOT / "res/report_assets/official_importer_export_full_bundle_task_conditioned_guidance_multiseed"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    tmp.replace(path)


def safe_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def build_summary_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in summary["aggregate"]:
        rows.append(
            {
                "task": row["task"],
                "seed_count": row["seed_count"],
                "guided_reward_mean": row["guided_reward_mean_mean"],
                "guided_reward_std": row["guided_reward_mean_std"],
                "guided_target_body_error_mean": row["guided_target_body_error_mean_mean"],
                "guided_target_body_error_std": row["guided_target_body_error_mean_std"],
                "guided_done_count_total_mean": row["guided_done_count_total_mean"],
                "guided_done_count_total_std": row["guided_done_count_total_std"],
                "guidance_cost_delta_mean": row["guidance_cost_delta_mean_mean"],
                "guidance_cost_delta_std": row["guidance_cost_delta_mean_std"],
                "guided_teacher_action_mse_mean": row["guided_teacher_action_mse_mean_mean"],
                "guided_teacher_action_mse_std": row["guided_teacher_action_mse_mean_std"],
            }
        )
    return rows


def plot_bars(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [row["task"].replace("_", "\n") for row in rows]
    x = range(len(rows))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    panels = [
        ("guided_reward_mean", "guided_reward_std", "Guided reward"),
        ("guided_target_body_error_mean", "guided_target_body_error_std", "Target-body error"),
        ("guided_done_count_total_mean", "guided_done_count_total_std", "Done count"),
        ("guidance_cost_delta_mean", "guidance_cost_delta_std", "Guidance cost delta"),
    ]
    colors = ["#3f6f9d", "#b24b3c", "#3e8d68", "#c98337"]
    for ax, (mean_key, std_key, title) in zip(axes.flat, panels):
        means = [safe_float(row[mean_key]) for row in rows]
        stds = [safe_float(row[std_key]) for row in rows]
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="#222222", linewidth=0.8)
        ax.set_xticks(list(x), labels)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    seed_counts = sorted({int(row["seed_count"]) for row in rows})
    seed_count_label = seed_counts[0] if len(seed_counts) == 1 else "/".join(str(value) for value in seed_counts)
    fig.suptitle(f"Official importer-export task-conditioned latent guidance, {seed_count_label} seed groups")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_seed_scatter(summary: dict[str, Any], path: Path) -> None:
    rows = summary["rows"]
    fig, ax = plt.subplots(figsize=(10, 5.8), constrained_layout=True)
    marker_cycle = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
    seed_groups = sorted({row["seed_group"] for row in rows})
    markers = {seed_group: marker_cycle[index % len(marker_cycle)] for index, seed_group in enumerate(seed_groups)}
    colors = {"joystick": "#3f6f9d", "waypoint": "#b24b3c", "obstacle_avoidance": "#3e8d68", "composed": "#c98337"}
    for row in rows:
        task = row["task"]
        group = row["seed_group"]
        ax.scatter(
            safe_float(row["guided_target_body_error_mean"]),
            safe_float(row["guided_reward_mean"]),
            s=72,
            marker=markers.get(group, "o"),
            color=colors.get(task, "#555555"),
            edgecolor="#222222",
            linewidth=0.7,
            label=f"{task} / {group}",
            alpha=0.9,
        )
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=7, ncol=2, loc="best")
    ax.set_xlabel("Guided target-body error mean")
    ax.set_ylabel("Guided reward mean")
    ax.set_title("Importer-export per-seed guided reward/error tradeoff")
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = load_json(SUMMARY_JSON)
    rows = build_summary_rows(summary)
    aggregate_csv = OUT / "importer_export_task_conditioned_guidance_multiseed_aggregate.csv"
    bars_png = OUT / "importer_export_task_conditioned_guidance_multiseed_bars.png"
    scatter_png = OUT / "importer_export_task_conditioned_guidance_multiseed_seed_scatter.png"
    readme = OUT / "README.md"
    fields = [
        "task",
        "seed_count",
        "guided_reward_mean",
        "guided_reward_std",
        "guided_target_body_error_mean",
        "guided_target_body_error_std",
        "guided_done_count_total_mean",
        "guided_done_count_total_std",
        "guidance_cost_delta_mean",
        "guidance_cost_delta_std",
        "guided_teacher_action_mse_mean",
        "guided_teacher_action_mse_std",
    ]
    write_csv(aggregate_csv, rows, fields)
    plot_bars(rows, bars_png)
    plot_seed_scatter(summary, scatter_png)
    readme.write_text(
        "\n".join(
            [
                "# Official importer-export full-bundle task-conditioned guidance multiseed assets",
                "",
                "These assets summarize local virtual multi-seed task-conditioned receding latent-guidance "
                "rollouts over the 40-motion public bundle using the official-importer-export G1 USDA path.",
                "",
                "Claim level: local virtual closed-loop guidance evidence only. Not paper Fig. 5/Fig. 6 "
                "reproduction, not official BeyondMimic checkpoints, not TensorRT deployment, and not "
                "real-robot evidence.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assets = {
        "status": "ok_official_importer_export_full_bundle_task_conditioned_guidance_multiseed_assets",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_summary": str(SUMMARY_JSON),
        "row_count": summary["metrics"]["row_count"],
        "task_count": summary["metrics"]["task_count"],
        "seed_group_count": summary["metrics"]["seed_group_count"],
        "aggregate": rows,
        "assets": {
            "aggregate_csv": str(aggregate_csv),
            "bars_png": str(bars_png),
            "seed_scatter_png": str(scatter_png),
            "readme": str(readme),
        },
        "checks": {
            "summary_status_ok": (
                summary["status"]
                == "ok_official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval"
            ),
            "all_rows_ok": summary["checks"]["all_rows_ok"],
            "seed_group_count_at_least_3": summary["checks"]["seed_group_count_at_least_3"],
            "seed_group_count_at_least_5": summary["checks"].get("seed_group_count_at_least_5", False),
            "four_tasks_per_seed_group": summary["checks"]["four_tasks_per_seed_group"],
            "uses_full_public_motion_bundle": summary["checks"]["uses_full_public_motion_bundle"],
            "full_bundle_motion_count_40": summary["checks"]["full_bundle_motion_count_40"],
            "uses_official_importer_export_usd": summary["checks"]["uses_official_importer_export_usd"],
            "does_not_claim_fig5_fig6": True,
            "does_not_claim_real_robot": True,
            "does_not_claim_goal_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "claim_level": "local_virtual_official_importer_export_task_conditioned_latent_guidance_multiseed_assets",
            "why_not_paper_level": (
                "The source rollouts use local official-importer-export PPO/VAE/denoiser checkpoints and proxy "
                "task costs. They are useful report/PPT evidence but not official BeyondMimic Fig. 5/Fig. 6 "
                "metrics, TensorRT deployment, or real-robot validation."
            ),
        },
    }
    write_json(OUT / "official_importer_export_full_bundle_task_conditioned_guidance_multiseed_assets.json", assets)
    print(
        json.dumps(
            {
                "status": assets["status"],
                "json": str(OUT / "official_importer_export_full_bundle_task_conditioned_guidance_multiseed_assets.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
