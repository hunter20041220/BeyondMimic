#!/usr/bin/env python3
"""Summarize official-importer-export tracking eval evidence for reports."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
TASK_EVAL = (
    ROOT
    / "res/tracking/g1_official_importer_export_full_dataset_task_eval/"
    "tracking_g1_official_importer_export_full_dataset_task_eval.json"
)
TASK_ASSETS = (
    ROOT
    / "res/report_assets/official_importer_export_full_dataset_task_eval/"
    "official_importer_export_full_dataset_task_eval_assets.json"
)
SCALED_PPO_EVAL = (
    ROOT
    / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/"
    "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json"
)
SCALED_PPO_ASSETS = (
    ROOT
    / "res/report_assets/official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/"
    "official_importer_export_full_bundle_scaled_ppo_checkpoint_eval_assets.json"
)
POLICY_VIDEO = (
    ROOT
    / "res/visualization/official_importer_export_full_bundle_scaled_ppo_policy_rollout/"
    "official_importer_export_full_bundle_scaled_ppo_policy_rollout_video_asset.json"
)
OUT = ROOT / "res/report_assets/official_importer_export_tracking_eval_summary"
ASSET_JSON = OUT / "official_importer_export_tracking_eval_summary_assets.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    task = load_json(TASK_EVAL)
    task_assets = load_json(TASK_ASSETS)
    ppo = load_json(SCALED_PPO_EVAL)
    ppo_assets = load_json(SCALED_PPO_ASSETS)
    video = load_json(POLICY_VIDEO)

    task_agg = task["aggregate"]
    ppo_metrics = ppo["run"]["metrics"]
    video_metrics = video["metrics"]
    rows = [
        {
            "stage": "full_dataset_task_diagnostic",
            "status": task["status"],
            "motion_count": task_agg["row_count"],
            "env_count": "",
            "step_count": task_agg["total_steps"],
            "reward_mean": task_agg["reward_mean"]["mean"],
            "done_count_total": task_agg["total_done_count"],
            "timeout_count_total": "",
            "anchor_pos_error_mean": task_agg["error_anchor_pos"]["mean"],
            "body_pos_error_mean": task_agg["error_body_pos"]["mean"],
            "joint_pos_error_mean": task_agg["error_joint_pos"]["mean"],
            "completion_rate": task_agg["ok_count"] / task_agg["row_count"],
            "claim_level": "local_virtual_zero_action_task_diagnostic",
        },
        {
            "stage": "scaled_ppo_checkpoint_eval",
            "status": ppo["status"],
            "motion_count": ppo_metrics["motion_count"],
            "env_count": ppo_metrics["num_envs"],
            "step_count": ppo_metrics["total_env_steps"],
            "reward_mean": ppo_metrics["reward"]["mean_over_steps"]["mean"],
            "done_count_total": ppo_metrics["done_count_total"],
            "timeout_count_total": ppo_metrics["timeout_count_total"],
            "anchor_pos_error_mean": ppo_metrics["motion_metrics"]["error_anchor_pos"]["mean"],
            "body_pos_error_mean": ppo_metrics["motion_metrics"]["error_body_pos"]["mean"],
            "joint_pos_error_mean": ppo_metrics["motion_metrics"]["error_joint_pos"]["mean"],
            "completion_rate": "",
            "claim_level": "local_virtual_scaled_ppo_checkpoint_eval",
        },
        {
            "stage": "scaled_ppo_policy_video",
            "status": video["status"],
            "motion_count": 1,
            "env_count": 1,
            "step_count": 299,
            "reward_mean": video_metrics["reward_mean"],
            "done_count_total": video_metrics["done_count_total"],
            "timeout_count_total": "",
            "anchor_pos_error_mean": "",
            "body_pos_error_mean": video_metrics["target_body_error_mean"],
            "joint_pos_error_mean": "",
            "completion_rate": "",
            "claim_level": video["claim_level"],
        },
    ]
    metrics_csv = OUT / "official_importer_export_tracking_eval_summary_metrics.csv"
    write_csv(
        metrics_csv,
        rows,
        [
            "stage",
            "status",
            "motion_count",
            "env_count",
            "step_count",
            "reward_mean",
            "done_count_total",
            "timeout_count_total",
            "anchor_pos_error_mean",
            "body_pos_error_mean",
            "joint_pos_error_mean",
            "completion_rate",
            "claim_level",
        ],
    )

    boundary_rows = [
        {
            "claim": "AppLauncher headless gate",
            "status": "passed_current_gate",
            "evidence": "res/setup/isaaclab_current_headless_gate/isaaclab_current_headless_gate.json",
            "limitation": "startup gate only",
        },
        {
            "claim": "official importer-export replay/task coverage",
            "status": "local_virtual_full_public_bundle",
            "evidence": str(TASK_EVAL),
            "limitation": "captured importer-export USDA, not unmodified live converter-entry success",
        },
        {
            "claim": "scaled PPO checkpoint evaluation",
            "status": "local_virtual_checkpoint_eval",
            "evidence": str(SCALED_PPO_EVAL),
            "limitation": "local checkpoint with high termination count, not official paper teacher",
        },
        {
            "claim": "report video",
            "status": "local_virtual_policy_rollout_video",
            "evidence": video["assets"]["mp4"],
            "limitation": "local MP4 path, not paper Fig.5/Fig.6 or real robot",
        },
        {
            "claim": "paper-level BeyondMimic tracking reproduction",
            "status": "not_claimed",
            "evidence": "required artifact absence and master audit",
            "limitation": "official checkpoints, DAgger logs, paper rollout videos, TensorRT, and real robot evidence remain absent",
        },
    ]
    boundary_csv = OUT / "official_importer_export_tracking_eval_claim_boundary.csv"
    write_csv(boundary_csv, boundary_rows, ["claim", "status", "evidence", "limitation"])

    plt.style.use("seaborn-v0_8-whitegrid")
    labels = ["task diagnostic", "scaled PPO eval", "policy video"]
    reward = [float(row["reward_mean"]) for row in rows]
    body_error = [float(row["body_pos_error_mean"]) for row in rows]
    done = [float(row["done_count_total"]) for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), sharex=True)
    axes[0].bar(labels, reward, color="#2563eb")
    axes[0].set_ylabel("Reward mean")
    axes[0].set_title("Official-importer-export tracking evidence summary")
    axes[1].bar(labels, body_error, color="#16a34a")
    axes[1].set_ylabel("Body/target-body error mean")
    axes[2].bar(labels, done, color="#f97316")
    axes[2].set_ylabel("Done count total")
    axes[2].set_yscale("symlog")
    axes[2].set_xlabel("Evidence stage")
    fig.tight_layout()
    overview_png = OUT / "official_importer_export_tracking_eval_overview.png"
    fig.savefig(overview_png, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    error_labels = ["anchor", "body", "joint"]
    task_errors = [
        task_agg["error_anchor_pos"]["mean"],
        task_agg["error_body_pos"]["mean"],
        task_agg["error_joint_pos"]["mean"],
    ]
    ppo_errors = [
        ppo_metrics["motion_metrics"]["error_anchor_pos"]["mean"],
        ppo_metrics["motion_metrics"]["error_body_pos"]["mean"],
        ppo_metrics["motion_metrics"]["error_joint_pos"]["mean"],
    ]
    x = range(len(error_labels))
    ax.bar([v - 0.18 for v in x], task_errors, width=0.36, label="task diagnostic", color="#64748b")
    ax.bar([v + 0.18 for v in x], ppo_errors, width=0.36, label="scaled PPO eval", color="#2563eb")
    ax.set_xticks(list(x), error_labels)
    ax.set_ylabel("Mean tracking error")
    ax.set_title("Task diagnostic vs scaled PPO checkpoint tracking errors")
    ax.legend(loc="upper right")
    fig.tight_layout()
    errors_png = OUT / "official_importer_export_tracking_error_comparison.png"
    fig.savefig(errors_png, dpi=180)
    plt.close(fig)

    readme = OUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Official-Importer-Export Tracking Eval Summary",
                "",
                "This directory summarizes the current official-importer-export tracking evidence for the English",
                "reading report and PPT.",
                "",
                "## Included Evidence",
                "",
                "- full public-motion task diagnostic: 40/40 rows, 11960 task steps",
                "- scaled PPO checkpoint evaluation: 2048 envs x 299 steps",
                "- scaled PPO policy rollout video: local 299-frame MP4 path indexed in JSON",
                "",
                "## Claim Boundary",
                "",
                "These assets are local virtual evidence. They are not official BeyondMimic paper-level tracking",
                "teacher results, not official DAgger logs, not Fig.5/Fig.6 guided diffusion, not TensorRT deployment,",
                "and not real-robot validation.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assets = {
        "metrics_csv": str(metrics_csv),
        "claim_boundary_csv": str(boundary_csv),
        "overview_png": str(overview_png),
        "tracking_error_comparison_png": str(errors_png),
        "readme": str(readme),
        "task_eval_assets_json": str(TASK_ASSETS),
        "scaled_ppo_eval_assets_json": str(SCALED_PPO_ASSETS),
        "policy_video_asset_json": str(POLICY_VIDEO),
        "policy_video_mp4": video["assets"]["mp4"],
    }
    checks = {
        "task_eval_status_ok": task["status"] == "ok_official_importer_export_full_dataset_task_eval",
        "task_eval_40_of_40_ok": task_agg["row_count"] == 40 and task_agg["ok_count"] == 40 and task_agg["failed_count"] == 0,
        "scaled_ppo_eval_status_ok": ppo["status"] == "ok_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval_completed",
        "scaled_ppo_uses_official_importer_export_usd": bool(ppo_metrics["uses_official_importer_export_usd"]),
        "policy_video_status_ok": video["status"] == "ok_official_importer_export_full_bundle_scaled_ppo_policy_rollout_video_asset",
        "policy_video_exists": Path(video["assets"]["mp4"]).is_file() and Path(video["assets"]["mp4"]).stat().st_size > 0,
        "all_report_assets_exist": all(
            Path(path).is_file() and Path(path).stat().st_size > 0
            for key, path in assets.items()
            if key != "policy_video_mp4"
        ),
        "does_not_claim_paper_level_tracking": True,
        "does_not_claim_fig5_fig6": True,
        "does_not_claim_real_robot": True,
    }
    summary = {
        "status": "ok_official_importer_export_tracking_eval_summary_assets" if all(checks.values()) else "failed",
        "experiment_type": "official_importer_export_tracking_eval_summary_assets",
        "source_artifacts": {
            "task_eval": str(TASK_EVAL),
            "task_assets": str(TASK_ASSETS),
            "scaled_ppo_eval": str(SCALED_PPO_EVAL),
            "scaled_ppo_assets": str(SCALED_PPO_ASSETS),
            "policy_video": str(POLICY_VIDEO),
        },
        "metrics": {row["stage"]: row for row in rows},
        "claim_boundary": boundary_rows,
        "assets": assets,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "paper_level_tracking_eval_complete": False,
            "claim_level": "local_virtual_official_importer_export_tracking_eval_summary",
            "why_not_complete": (
                "This summarizes current local official-importer-export tracking evidence for reporting. It does not "
                "supply official paper checkpoints, DAgger logs, official Fig.5/Fig.6 rollouts, TensorRT deployment, "
                "or real robot validation."
            ),
        },
    }
    ASSET_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(ASSET_JSON)}, sort_keys=True))
    if summary["status"] != "ok_official_importer_export_tracking_eval_summary_assets":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
