#!/usr/bin/env python3
"""Create report assets for the official-importer-export VAE closed-loop rollout."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SUMMARY_JSON = (
    ROOT
    / "res/level_c/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/"
    "tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_eval.json"
)
OUT = ROOT / "res/report_assets/official_importer_export_full_bundle_vae_closed_loop_rollout_eval"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_timeseries(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty timeseries: {path}")
    columns: dict[str, list[float]] = {key: [] for key in rows[0]}
    for row in rows:
        for key, value in row.items():
            columns[key].append(float(value))
    return {key: np.asarray(values, dtype=np.float64) for key, values in columns.items()}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def mean_stack(series: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    return np.stack([item[key] for item in series], axis=0).mean(axis=0)


def parse_metric(value: str) -> float:
    return float(str(value).strip().split()[0])


def read_gpu_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = load_json(SUMMARY_JSON)
    shard_metrics = summary["run"]["shard_metrics"]
    series = [read_timeseries(Path(item["timeseries_csv"])) for item in shard_metrics]

    steps = mean_stack(series, "step")
    reward_mean = mean_stack(series, "reward_mean")
    done_count = mean_stack(series, "done_count")
    teacher_vae_mse = mean_stack(series, "teacher_vae_action_mse")
    teacher_vae_abs = mean_stack(series, "teacher_vae_action_abs_error_mean")
    teacher_abs = mean_stack(series, "teacher_action_abs_mean")
    vae_abs = mean_stack(series, "vae_action_abs_mean")
    body_pos = mean_stack(series, "error_body_pos")
    joint_pos = mean_stack(series, "error_joint_pos")
    anchor_pos = mean_stack(series, "error_anchor_pos")

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
    axes[0].plot(steps, reward_mean, color="#2563eb", label="mean reward")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Official-importer VAE closed-loop rollout reward")
    axes[0].legend(loc="upper right")
    axes[1].bar(steps, done_count, width=1.0, color="#dc2626", label="mean done count per rank")
    axes[1].set_xlabel("Rollout step")
    axes[1].set_ylabel("Env count")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    reward_png = OUT / "official_importer_vae_closed_loop_reward_done_timeseries.png"
    fig.savefig(reward_png, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.plot(steps, teacher_vae_mse, color="#7c3aed", label="teacher vs VAE action MSE")
    ax.plot(steps, teacher_vae_abs, color="#f59e0b", label="teacher vs VAE mean |error|")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Action reconstruction error")
    ax.set_title("Official-importer VAE action reconstruction error")
    ax.legend(loc="upper right")
    fig.tight_layout()
    action_error_png = OUT / "official_importer_vae_closed_loop_action_reconstruction_error.png"
    fig.savefig(action_error_png, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.plot(steps, teacher_abs, color="#0f766e", label="teacher mean |action|")
    ax.plot(steps, vae_abs, color="#be123c", label="VAE decoded mean |action|")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Action magnitude")
    ax.set_title("Teacher and decoded VAE action magnitudes")
    ax.legend(loc="upper right")
    fig.tight_layout()
    action_mag_png = OUT / "official_importer_vae_closed_loop_action_magnitude.png"
    fig.savefig(action_mag_png, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.4))
    ax.plot(steps, body_pos, color="#1d4ed8", label="body position error")
    ax.plot(steps, joint_pos, color="#b45309", label="joint position error")
    ax.plot(steps, anchor_pos, color="#047857", label="anchor position error")
    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Mean tracking error")
    ax.set_title("Selected tracking errors during official-importer VAE rollout")
    ax.legend(loc="upper right")
    fig.tight_layout()
    tracking_png = OUT / "official_importer_vae_closed_loop_tracking_errors.png"
    fig.savefig(tracking_png, dpi=180)
    plt.close(fig)

    gpu_summary_rows = []
    gpu_rows = read_gpu_rows(Path(summary["run"]["gpu_metrics_csv"]))
    gpu_png = OUT / "official_importer_vae_closed_loop_gpu_memory.png"
    if gpu_rows:
        per_gpu: dict[str, dict[str, list[float]]] = {}
        for row in gpu_rows:
            normalized = {key.strip(): value.strip() for key, value in row.items() if key is not None}
            index = normalized["index"]
            item = per_gpu.setdefault(index, {"mem": [], "util": []})
            item["mem"].append(parse_metric(normalized["memory.used [MiB]"]))
            item["util"].append(parse_metric(normalized["utilization.gpu [%]"]))
        fig, ax = plt.subplots(figsize=(11, 5.4))
        for index, vals in sorted(per_gpu.items()):
            x = np.arange(len(vals["mem"]))
            ax.plot(x, vals["mem"], label=f"GPU {index} memory MiB")
            gpu_summary_rows.append(
                {
                    "gpu": index,
                    "samples": len(vals["mem"]),
                    "peak_memory_mb": float(np.max(vals["mem"])),
                    "mean_memory_mb": float(np.mean(vals["mem"])),
                    "mean_utilization_percent": float(np.mean(vals["util"])),
                    "peak_memory_at_least_10gb": bool(np.max(vals["mem"]) >= 10240.0),
                }
            )
        ax.set_xlabel("Telemetry sample")
        ax.set_ylabel("Memory used MiB")
        ax.set_title("GPU telemetry during official-importer VAE rollout")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(gpu_png, dpi=180)
        plt.close(fig)

    shard_rows = [
        {
            "rank": item["rank"],
            "device": item["device"],
            "num_envs": item["num_envs"],
            "rollout_steps": item["rollout_steps"],
            "total_env_steps": item["total_env_steps"],
            "reward_mean": item["reward_mean"]["mean"],
            "done_count_total": item["done_count_total"],
            "teacher_vae_action_mse_mean": item["teacher_vae_action_mse"]["mean"],
            "teacher_vae_action_abs_error_mean": item["teacher_vae_action_abs_error_mean"]["mean"],
            "body_pos_error_mean": item["motion_metrics"]["error_body_pos"]["mean"],
            "joint_pos_error_mean": item["motion_metrics"]["error_joint_pos"]["mean"],
        }
        for item in shard_metrics
    ]
    shard_csv = OUT / "official_importer_vae_closed_loop_shard_summary.csv"
    write_csv(
        shard_csv,
        shard_rows,
        [
            "rank",
            "device",
            "num_envs",
            "rollout_steps",
            "total_env_steps",
            "reward_mean",
            "done_count_total",
            "teacher_vae_action_mse_mean",
            "teacher_vae_action_abs_error_mean",
            "body_pos_error_mean",
            "joint_pos_error_mean",
        ],
    )

    gpu_summary_csv = OUT / "official_importer_vae_closed_loop_gpu_summary.csv"
    write_csv(
        gpu_summary_csv,
        gpu_summary_rows,
        [
            "gpu",
            "samples",
            "peak_memory_mb",
            "mean_memory_mb",
            "mean_utilization_percent",
            "peak_memory_at_least_10gb",
        ],
    )

    aggregate = summary["run"]["aggregate_metrics"]
    assets_json = OUT / "official_importer_export_full_bundle_vae_closed_loop_rollout_assets.json"
    readme = OUT / "README.md"
    asset_summary = {
        "status": "ok_official_importer_export_full_bundle_vae_closed_loop_rollout_assets",
        "source_summary_json": str(SUMMARY_JSON),
        "claim_level": "local_virtual_official_importer_export_vae_action_reconstruction_closed_loop_report_asset",
        "metrics": {
            "total_env_steps": aggregate["total_env_steps"],
            "total_num_envs": aggregate["total_num_envs"],
            "rollout_steps": aggregate["rollout_steps"],
            "teacher_vae_action_mse_mean": aggregate["teacher_vae_action_mse"]["mean"],
            "teacher_vae_action_abs_error_mean": aggregate["teacher_vae_action_abs_error"]["mean"],
            "reward_mean": aggregate["reward_mean"]["mean"],
            "done_count_total": aggregate["done_count_total"],
            "timeout_count_total": aggregate["timeout_count_total"],
        },
        "assets": {
            "reward_done_timeseries_png": str(reward_png),
            "action_reconstruction_error_png": str(action_error_png),
            "action_magnitude_png": str(action_mag_png),
            "tracking_errors_png": str(tracking_png),
            "gpu_memory_png": str(gpu_png),
            "shard_summary_csv": str(shard_csv),
            "gpu_summary_csv": str(gpu_summary_csv),
            "summary_md": str(readme),
        },
        "checks": {
            "source_status_ok": summary["status"]
            == "ok_official_importer_export_full_bundle_vae_closed_loop_rollout_eval",
            "two_shards_completed": aggregate["shard_count"] == 2,
            "rollout_steps_299": aggregate["rollout_steps"] == 299,
            "total_env_steps_matches_summary": aggregate["total_env_steps"]
            == summary["config"]["expected_total_env_steps"],
            "png_assets_exist": all(
                path.is_file() for path in [reward_png, action_error_png, action_mag_png, tracking_png, gpu_png]
            ),
            "csv_assets_exist": shard_csv.is_file() and gpu_summary_csv.is_file(),
            "does_not_claim_official_beyondmimic_vae": True,
            "does_not_claim_guided_diffusion": True,
            "does_not_claim_fig5_fig6": True,
            "does_not_claim_real_robot": True,
        },
        "limitation": (
            "These assets summarize a local virtual VAE action-reconstruction rollout on the "
            "official-importer-export G1 USDA. They do not use unreleased official BeyondMimic VAE/diffusion "
            "checkpoints, do not evaluate receding-horizon guided diffusion, do not reproduce Fig. 5/Fig. 6, "
            "and do not contain real-robot evidence."
        ),
    }
    readme.write_text(
        "\n".join(
            [
                "# Official-Importer-Export VAE Closed-Loop Rollout Assets",
                "",
                "These plots summarize a local virtual rollout where the full-bundle PPO teacher action is encoded",
                "and decoded through the local official-importer-export conditional action VAE before stepping IsaacLab.",
                "",
                "## Source",
                "",
                f"- Rollout summary: `{SUMMARY_JSON}`",
                f"- Status: `{summary['status']}`",
                f"- Total env steps: `{aggregate['total_env_steps']}`",
                f"- Teacher/VAE action MSE mean: `{aggregate['teacher_vae_action_mse']['mean']}`",
                f"- Teacher/VAE action absolute-error mean: `{aggregate['teacher_vae_action_abs_error']['mean']}`",
                "",
                "## Assets",
                "",
                f"- `{reward_png}`",
                f"- `{action_error_png}`",
                f"- `{action_mag_png}`",
                f"- `{tracking_png}`",
                f"- `{gpu_png}`",
                f"- `{shard_csv}`",
                f"- `{gpu_summary_csv}`",
                "",
                "## Boundary",
                "",
                asset_summary["limitation"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    assets_json.write_text(json.dumps(asset_summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": asset_summary["status"], "json": str(assets_json)}, sort_keys=True))


if __name__ == "__main__":
    main()
