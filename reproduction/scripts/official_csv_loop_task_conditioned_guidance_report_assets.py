#!/usr/bin/env python3
"""Create report assets for task-conditioned latent-guidance rollouts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
VARIANT = os.environ.get("BM_TASK_CONDITIONED_REPORT_VARIANT", "standard")
if VARIANT not in {"standard", "full_bundle", "official_importer_export_scaled_ppo"}:
    raise ValueError(f"Unsupported BM_TASK_CONDITIONED_REPORT_VARIANT={VARIANT!r}")
IS_FULL_BUNDLE = VARIANT == "full_bundle"
IS_SCALED_IMPORTER = VARIANT == "official_importer_export_scaled_ppo"
SUMMARY_JSON = ROOT / (
    (
        "res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval/"
        "level_c_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval.json"
    )
    if IS_SCALED_IMPORTER
    else (
        "res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval/"
        "level_c_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval.json"
    )
    if IS_FULL_BUNDLE
    else (
        "res/level_c/official_csv_loop_task_conditioned_latent_guidance_rollout_eval/"
        "level_c_official_csv_loop_task_conditioned_latent_guidance_rollout_eval.json"
    )
)
OUT = ROOT / (
    "res/report_assets/official_importer_export_scaled_ppo_task_conditioned_guidance_summary"
    if IS_SCALED_IMPORTER
    else (
        "res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_summary"
        if IS_FULL_BUNDLE
        else "res/report_assets/official_csv_loop_task_conditioned_guidance_summary"
    )
)
TASK_ORDER = ["joystick", "waypoint", "obstacle_avoidance", "composed"]
VARIANT_ORDER = ["teacher", "vae_base", "denoised_latent", "receding_latent_guided"]
VARIANT_LABELS = {
    "teacher": "Teacher",
    "vae_base": "VAE base",
    "denoised_latent": "Denoised",
    "receding_latent_guided": "Guided",
}
TASK_LABELS = {
    "joystick": "Joystick",
    "waypoint": "Waypoint",
    "obstacle_avoidance": "Obstacle",
    "composed": "Composed",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "task",
        "variant",
        "reward_mean",
        "target_body_error_mean",
        "done_count_total",
        "guided_teacher_action_mse_mean",
        "guided_base_action_mse_mean",
        "guidance_cost_delta_mean",
        "mp4",
        "asset_json",
        "claim_level",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = load_json(SUMMARY_JSON)
    rows = []
    guided_rows = []

    for task_row in summary["rows"]:
        task = task_row["task"]
        task_summary = load_json(Path(task_row["summary_json"]))
        task_asset = load_json(Path(task_row["asset_json"]))
        metrics = task_summary["metrics"]["variant_metrics"]
        for variant in VARIANT_ORDER:
            payload = metrics[variant]
            row = {
                "task": task,
                "variant": variant,
                "reward_mean": payload["reward_mean"],
                "target_body_error_mean": payload["target_body_error_mean"],
                "done_count_total": payload["done_count_total"],
                "guided_teacher_action_mse_mean": payload.get("guided_teacher_action_mse_mean"),
                "guided_base_action_mse_mean": payload.get("guided_base_action_mse_mean"),
                "guidance_cost_delta_mean": payload.get("guidance_cost_delta_mean"),
                "mp4": task_row["mp4"],
                "asset_json": task_row["asset_json"],
                "claim_level": task_asset["claim_level"],
            }
            rows.append(row)
            if variant == "receding_latent_guided":
                guided_rows.append(row)

    metrics_csv = OUT / "task_conditioned_guidance_metrics.csv"
    guided_csv = OUT / "task_conditioned_guided_summary.csv"
    write_csv(metrics_csv, rows)
    write_csv(guided_csv, guided_rows)
    frame = pd.DataFrame(rows)
    guided = pd.DataFrame(guided_rows)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    colors = {
        "teacher": "#059669",
        "vae_base": "#2563eb",
        "denoised_latent": "#9333ea",
        "receding_latent_guided": "#dc2626",
    }
    for idx, metric in enumerate(["reward_mean", "target_body_error_mean", "done_count_total", "guided_teacher_action_mse_mean"]):
        ax = axes[idx]
        pivot = frame.pivot(index="task", columns="variant", values=metric).loc[TASK_ORDER, VARIANT_ORDER]
        pivot.rename(index=TASK_LABELS, columns=VARIANT_LABELS).plot(
            kind="bar",
            ax=ax,
            color=[colors[v] for v in VARIANT_ORDER],
            width=0.78,
        )
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        if idx == 0:
            ax.legend(loc="best", fontsize=8)
        else:
            ax.get_legend().remove()
    fig.suptitle("Local Task-Conditioned Receding Latent Guidance Rollouts")
    fig.tight_layout()
    overview_png = OUT / "task_conditioned_guidance_overview.png"
    fig.savefig(overview_png, dpi=180)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(10, 5.8))
    x = range(len(guided))
    ax1.bar(x, guided["guidance_cost_delta_mean"], color="#dc2626", alpha=0.78, label="Guidance cost delta")
    ax1.set_ylabel("Guidance cost delta")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([TASK_LABELS[t] for t in guided["task"]], rotation=15)
    ax2 = ax1.twinx()
    ax2.plot(x, guided["target_body_error_mean"], color="#2563eb", marker="o", linewidth=2.0, label="Target-body error")
    ax2.set_ylabel("Target-body error mean")
    ax1.set_title("Guidance Cost Reduction vs. Tracking Error")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    fig.tight_layout()
    tradeoff_png = OUT / "task_conditioned_guidance_tradeoff.png"
    fig.savefig(tradeoff_png, dpi=180)
    plt.close(fig)

    readme = OUT / "README.md"
    if IS_SCALED_IMPORTER:
        title_prefix = "Official-Importer-Export Scaled PPO"
        claim_level = "local_virtual_official_importer_export_scaled_ppo_task_conditioned_guidance_summary"
    elif IS_FULL_BUNDLE:
        title_prefix = "Official-CSV-Loop Full-Bundle"
        claim_level = "local_virtual_full_bundle_task_conditioned_receding_horizon_latent_guidance_rollout_summary"
    else:
        title_prefix = "Official-CSV-Loop"
        claim_level = "local_virtual_task_conditioned_receding_horizon_latent_guidance_rollout_summary"
    readme.write_text(
        "\n".join(
            [
                f"# {title_prefix} Task-Conditioned Guidance Summary",
                "",
                "This directory aggregates four local IsaacLab closed-loop task-conditioned latent-guidance rollouts.",
                "",
                "Tasks: joystick, waypoint, obstacle_avoidance, composed.",
                "",
                "## Claim Level",
                "",
                f"{claim_level}.",
                "",
                "These assets are report/PPT evidence only. They are not official BeyondMimic Fig. 5/Fig. 6 results, not official checkpoints, not TensorRT/asynchronous deployment evidence, and not real-robot validation.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assets = {
        "metrics_csv": str(metrics_csv),
        "guided_summary_csv": str(guided_csv),
        "overview_png": str(overview_png),
        "tradeoff_png": str(tradeoff_png),
        "readme": str(readme),
    }
    asset_json = OUT / "official_csv_loop_task_conditioned_guidance_summary_assets.json"
    result = {
        "status": "ok",
        "experiment_type": (
            "official_importer_export_scaled_ppo_task_conditioned_guidance_report_assets"
            if IS_SCALED_IMPORTER
            else
            "official_csv_loop_full_bundle_task_conditioned_guidance_report_assets"
            if IS_FULL_BUNDLE
            else "official_csv_loop_task_conditioned_guidance_report_assets"
        ),
        "variant": VARIANT,
        "claim_level": claim_level,
        "source_summary": str(SUMMARY_JSON),
        "task_count": len(TASK_ORDER),
        "variant_count": len(VARIANT_ORDER),
        "rows": guided_rows,
        "assets": assets,
        "asset_sizes": {key: Path(value).stat().st_size for key, value in assets.items()},
        "asset_sha256": {key: sha256_file(Path(value)) for key, value in assets.items()},
        "checks": {
            "four_tasks_recorded": sorted(guided["task"].tolist()) == sorted(TASK_ORDER),
            "all_assets_nonempty": all(Path(value).is_file() and Path(value).stat().st_size > 0 for value in assets.values()),
            "all_guided_cost_deltas_present": bool(guided["guidance_cost_delta_mean"].notna().all()),
            "does_not_claim_paper_level": True,
            "does_not_claim_fig5_fig6": True,
            "does_not_claim_real_robot": True,
            "does_not_claim_goal_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "These are aggregate report assets for local task-conditioned proxy rollouts. They do not create "
                "official checkpoints, task success/failure rates, paper Fig. 5/Fig. 6 evidence, TensorRT deployment, "
                "or real-robot validation."
            ),
        },
    }
    asset_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "json": str(asset_json), "assets": assets}, sort_keys=True))


if __name__ == "__main__":
    main()
