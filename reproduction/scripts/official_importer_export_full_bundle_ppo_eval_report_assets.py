#!/usr/bin/env python3
"""Create report assets for official-importer-export full-bundle PPO eval."""

from __future__ import annotations

import csv
import json
import re
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
EVAL_AUDIT = Path(
    os.environ.get(
        "BM_IMPORTER_PPO_REPORT_EVAL_AUDIT",
        str(
            ROOT
            / "res/tracking/g1_official_importer_export_full_bundle_ppo_checkpoint_eval/"
            "tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval.json"
        ),
    )
)
TRAINING_AUDIT = Path(
    os.environ.get(
        "BM_IMPORTER_PPO_REPORT_TRAINING_AUDIT",
        str(
            ROOT
            / "res/tracking/g1_official_importer_export_full_bundle_ppo_training_run/"
            "tracking_g1_official_importer_export_full_bundle_ppo_training_run.json"
        ),
    )
)
OUT = Path(
    os.environ.get(
        "BM_IMPORTER_PPO_REPORT_OUT",
        str(ROOT / "res/report_assets/official_importer_export_full_bundle_ppo_checkpoint_eval"),
    )
)
ASSET_JSON_NAME = os.environ.get(
    "BM_IMPORTER_PPO_REPORT_ASSET_JSON_NAME", "official_importer_export_full_bundle_ppo_checkpoint_eval_assets.json"
)
REPORT_TITLE = os.environ.get(
    "BM_IMPORTER_PPO_REPORT_TITLE", "Official-importer-export PPO checkpoint tracking errors"
)
TRAINING_TITLE = os.environ.get(
    "BM_IMPORTER_PPO_REPORT_TRAINING_TITLE", "Official-importer-export PPO training curve"
)
CLAIM_LEVEL = os.environ.get("BM_IMPORTER_PPO_REPORT_CLAIM_LEVEL", "local_virtual_official_importer_export_report_asset")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean_numeric(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else float("nan")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_training_log(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    patterns = {
        "computation_steps_per_second": "Computation:",
        "mean_action_noise_std": "Mean action noise std:",
        "mean_value_function_loss": "Mean value_function loss:",
        "mean_surrogate_loss": "Mean surrogate loss:",
        "mean_entropy_loss": "Mean entropy loss:",
        "mean_reward": "Mean reward:",
        "mean_episode_length": "Mean episode length:",
        "error_anchor_pos": "Metrics/motion/error_anchor_pos:",
        "error_body_pos": "Metrics/motion/error_body_pos:",
        "error_joint_pos": "Metrics/motion/error_joint_pos:",
        "total_timesteps": "Total timesteps:",
    }
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "Learning iteration" in line:
            if current is not None:
                rows.append(current)
            match = re.search(r"Learning iteration\s+(\d+)/(\d+)", line)
            if match:
                current = {"iteration": int(match.group(1)), "max_iteration": int(match.group(2))}
            else:
                current = {}
            continue
        if current is None:
            continue
        for key, marker in patterns.items():
            if marker in line:
                if key == "computation_steps_per_second":
                    match = re.search(r"Computation:\s+([0-9.]+)\s+steps/s", line)
                else:
                    match = re.search(r":\s+(-?[0-9.]+)", line)
                if match:
                    current[key] = float(match.group(1))
    if current is not None:
        rows.append(current)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = load_json(EVAL_AUDIT)
    training_audit = load_json(TRAINING_AUDIT)
    outputs = audit["outputs"]
    training_outputs = training_audit["outputs"]
    metrics = audit["run"]["metrics"]
    eval_status_ok = str(audit.get("status", "")).startswith("ok_") and str(audit.get("status", "")).endswith(
        "_checkpoint_eval_completed"
    )
    timeseries_path = Path(outputs["timeseries_csv"])
    gpu_metrics_path = Path(outputs["gpu_metrics_csv"])
    eval_metrics_path = Path(outputs["metrics_json"])
    training_log_path = Path(training_outputs["log"])
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
    axes[0].set_title(REPORT_TITLE)
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
    axes[0].set_title(f"GPU telemetry during {REPORT_TITLE}")
    axes[0].legend(loc="upper right")
    axes[1].set_xlabel("Telemetry sample")
    axes[1].set_ylabel("Memory used (GiB)")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    gpu_png = OUT / "gpu_usage_eval.png"
    fig.savefig(gpu_png, dpi=180)
    plt.close(fig)

    training_rows = parse_training_log(training_log_path)
    training_curve_csv = OUT / "training_curve.csv"
    training_fields = [
        "iteration",
        "max_iteration",
        "total_timesteps",
        "computation_steps_per_second",
        "mean_reward",
        "mean_value_function_loss",
        "mean_surrogate_loss",
        "mean_entropy_loss",
        "mean_action_noise_std",
        "error_anchor_pos",
        "error_body_pos",
        "error_joint_pos",
    ]
    write_csv(training_curve_csv, training_rows, training_fields)
    train_df = pd.DataFrame(training_rows)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9.0), sharex=True)
    axes[0].plot(train_df["iteration"], train_df["mean_reward"], color="#059669", label="mean reward")
    axes[0].set_ylabel("Reward")
    axes[0].set_title(TRAINING_TITLE)
    axes[0].legend(loc="upper left")
    axes[1].plot(train_df["iteration"], train_df["error_anchor_pos"], color="#2563eb", label="anchor pos")
    axes[1].plot(train_df["iteration"], train_df["error_body_pos"], color="#16a34a", label="body pos")
    axes[1].plot(train_df["iteration"], train_df["error_joint_pos"], color="#dc2626", label="joint pos")
    axes[1].set_ylabel("Tracking error")
    axes[1].legend(loc="upper right")
    axes[2].plot(
        train_df["iteration"],
        train_df["mean_action_noise_std"],
        color="#7c3aed",
        label="action noise std",
    )
    axes[2].plot(
        train_df["iteration"],
        train_df["mean_entropy_loss"],
        color="#ea580c",
        label="entropy loss",
    )
    axes[2].set_xlabel("Learning iteration")
    axes[2].set_ylabel("Policy scale")
    axes[2].legend(loc="best")
    fig.tight_layout()
    training_curve_png = OUT / "training_curve.png"
    fig.savefig(training_curve_png, dpi=180)
    plt.close(fig)

    summary_rows = [
        {"metric": "status", "value": audit["status"], "claim_level": "local_virtual_official_importer_export"},
        {"metric": "loaded_iteration", "value": metrics["loaded_iteration"], "claim_level": "local_checkpoint"},
        {"metric": "motion_count", "value": metrics.get("motion_count"), "claim_level": "local_bundle"},
        {"metric": "total_motion_frames", "value": metrics.get("total_motion_frames"), "claim_level": "local_bundle"},
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
                f"# {TRAINING_TITLE} Eval Assets",
                "",
                "These plots and tables summarize the local virtual PPO checkpoint evaluation using the",
                "official-importer GPU4 G1 USDA export and the 40-motion official-loop public bundle.",
                "",
                "## Source",
                "",
                f"- Eval audit: `{EVAL_AUDIT}`",
                f"- Eval metrics: `{eval_metrics_path}`",
                f"- Training audit: `{TRAINING_AUDIT}`",
                f"- Training log: `{training_log_path}`",
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
                f"- `{training_curve_png}`",
                f"- `{training_curve_csv}`",
                f"- `{summary_csv}`",
                f"- `{gpu_summary_csv}`",
                "",
                "## Claim Level",
                "",
                f"{CLAIM_LEVEL} / qualitative engineering evidence. This is not a",
                "released official BeyondMimic PPO checkpoint, not paper-scale teacher training, not Fig. 5/Fig. 6",
                "guided diffusion, and not real-robot validation.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    asset_summary = {
        "status": "ok",
        "experiment_type": "official_importer_export_full_bundle_ppo_checkpoint_eval_report_assets",
        "source_eval_audit": str(EVAL_AUDIT),
        "source_timeseries_csv": str(timeseries_path),
        "source_gpu_metrics_csv": str(gpu_metrics_path),
        "claim_level": CLAIM_LEVEL,
        "limitation": (
            f"Uses a local iteration-{metrics.get('loaded_iteration', 'unknown')} PPO checkpoint trained/evaluated "
            "on a 40-motion concatenated public bundle "
            "with a local official-importer USDA asset; not an official BeyondMimic checkpoint, not paper-scale "
            "teacher training, and not real-robot evidence."
        ),
        "metrics": {
            "num_envs": metrics["num_envs"],
            "eval_steps": metrics["eval_steps"],
            "total_env_steps": metrics["total_env_steps"],
            "motion_count": metrics.get("motion_count"),
            "total_motion_frames": metrics.get("total_motion_frames"),
            "done_count_total": metrics["done_count_total"],
            "timeout_count_total": metrics["timeout_count_total"],
            "error_anchor_pos_mean": metrics["motion_metrics"]["error_anchor_pos"]["mean"],
            "error_body_pos_mean": metrics["motion_metrics"]["error_body_pos"]["mean"],
            "error_joint_pos_mean": metrics["motion_metrics"]["error_joint_pos"]["mean"],
            "reward_mean": metrics["reward"]["mean_over_steps"]["mean"],
            "training_iteration_count": len(training_rows),
            "training_duration_seconds": training_audit["run"].get("duration_seconds"),
            "training_checkpoint_count": training_audit["run"].get("checkpoint_count"),
        },
        "assets": {
            "tracking_error_timeseries_png": str(tracking_error_png),
            "reward_done_timeseries_png": str(reward_done_png),
            "gpu_usage_eval_png": str(gpu_png),
            "training_curve_png": str(training_curve_png),
            "training_curve_csv": str(training_curve_csv),
            "summary_csv": str(summary_csv),
            "gpu_summary_csv": str(gpu_summary_csv),
            "summary_md": str(readme),
        },
        "checks": {
            "eval_status_ok": eval_status_ok,
            "timeseries_has_299_rows": len(df) == 299,
            "summary_csv_exists": summary_csv.is_file(),
            "gpu_summary_csv_exists": gpu_summary_csv.is_file(),
            "png_assets_exist": tracking_error_png.is_file() and reward_done_png.is_file() and gpu_png.is_file(),
            "training_curve_assets_exist": training_curve_png.is_file()
            and training_curve_csv.is_file()
            and len(training_rows) >= 300,
            "does_not_claim_paper_level_eval": metrics["paper_level_tracking_eval"] is False,
            "uses_official_importer_export_usd": metrics["uses_official_importer_export_usd"] is True,
            "does_not_claim_real_robot": True,
        },
    }
    asset_json = OUT / ASSET_JSON_NAME
    asset_json.write_text(json.dumps(asset_summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(asset_json), "assets": asset_summary["assets"]}, sort_keys=True))


if __name__ == "__main__":
    main()
