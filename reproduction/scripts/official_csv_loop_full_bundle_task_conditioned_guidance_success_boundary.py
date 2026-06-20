#!/usr/bin/env python3
"""Summarize local proxy success boundaries for full-bundle guidance rollouts."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SOURCE_JSON = (
    ROOT
    / "res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval/"
    "official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json"
)
OUT = ROOT / "res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_success_boundary"


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


def rate(values: list[bool]) -> float:
    return sum(bool(v) for v in values) / len(values) if values else 0.0


def build_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary["rows"]:
        reward_delta_denoised = row["guided_reward_mean"] - row["denoised_reward_mean"]
        error_delta_denoised = row["guided_target_body_error_mean"] - row["denoised_target_body_error_mean"]
        reward_delta_vae = row["guided_reward_mean"] - row["vae_base_reward_mean"]
        error_delta_vae = row["guided_target_body_error_mean"] - row["vae_base_target_body_error_mean"]
        completed_299 = row["rollout_steps"] == 299 and str(row["status"]).startswith("ok_")
        guidance_signal_positive = row["guidance_cost_delta_mean"] > 0.0
        action_changed = row["guided_base_action_mse_mean"] > 1e-6
        reward_improved = reward_delta_denoised > 0.0
        error_not_worse = error_delta_denoised <= 0.0
        local_proxy_pass = completed_299 and guidance_signal_positive and action_changed and (reward_improved or error_not_worse)
        rows.append(
            {
                "task": row["task"],
                "seed_group": row["seed_group"],
                "seed": "" if row.get("seed") is None else row["seed"],
                "rollout_steps": row["rollout_steps"],
                "completed_299": completed_299,
                "guided_reward_mean": row["guided_reward_mean"],
                "denoised_reward_mean": row["denoised_reward_mean"],
                "vae_base_reward_mean": row["vae_base_reward_mean"],
                "guided_target_body_error_mean": row["guided_target_body_error_mean"],
                "denoised_target_body_error_mean": row["denoised_target_body_error_mean"],
                "vae_base_target_body_error_mean": row["vae_base_target_body_error_mean"],
                "reward_delta_vs_denoised": reward_delta_denoised,
                "error_delta_vs_denoised": error_delta_denoised,
                "reward_delta_vs_vae_base": reward_delta_vae,
                "error_delta_vs_vae_base": error_delta_vae,
                "guidance_cost_delta_mean": row["guidance_cost_delta_mean"],
                "guidance_signal_positive": guidance_signal_positive,
                "guided_base_action_mse_mean": row["guided_base_action_mse_mean"],
                "action_changed": action_changed,
                "reward_improved_vs_denoised": reward_improved,
                "tracking_error_not_worse_vs_denoised": error_not_worse,
                "local_proxy_pass": local_proxy_pass,
                "mp4": row["mp4"],
                "claim_level": "local_proxy_success_boundary_not_paper_fig5_fig6",
            }
        )
    return rows


def build_aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tasks = sorted({row["task"] for row in rows})
    out: list[dict[str, Any]] = []
    for task in tasks:
        task_rows = [row for row in rows if row["task"] == task]
        out.append(
            {
                "task": task,
                "row_count": len(task_rows),
                "completion_rate_299": rate([row["completed_299"] for row in task_rows]),
                "guidance_signal_positive_rate": rate([row["guidance_signal_positive"] for row in task_rows]),
                "action_changed_rate": rate([row["action_changed"] for row in task_rows]),
                "reward_improved_vs_denoised_rate": rate([row["reward_improved_vs_denoised"] for row in task_rows]),
                "tracking_error_not_worse_vs_denoised_rate": rate(
                    [row["tracking_error_not_worse_vs_denoised"] for row in task_rows]
                ),
                "local_proxy_pass_rate": rate([row["local_proxy_pass"] for row in task_rows]),
                "reward_delta_vs_denoised_mean": mean(row["reward_delta_vs_denoised"] for row in task_rows),
                "error_delta_vs_denoised_mean": mean(row["error_delta_vs_denoised"] for row in task_rows),
                "reward_delta_vs_vae_base_mean": mean(row["reward_delta_vs_vae_base"] for row in task_rows),
                "error_delta_vs_vae_base_mean": mean(row["error_delta_vs_vae_base"] for row in task_rows),
                "guidance_cost_delta_mean": mean(row["guidance_cost_delta_mean"] for row in task_rows),
                "guided_base_action_mse_mean": mean(row["guided_base_action_mse_mean"] for row in task_rows),
            }
        )
    return out


def plot_aggregate(rows: list[dict[str, Any]], path: Path) -> None:
    labels = [row["task"].replace("_", "\n") for row in rows]
    x = list(range(len(rows)))
    width = 0.17
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    rate_panels = [
        ("completion_rate_299", "299-step"),
        ("reward_improved_vs_denoised_rate", "reward>denoised"),
        ("tracking_error_not_worse_vs_denoised_rate", "error<=denoised"),
        ("local_proxy_pass_rate", "proxy pass"),
    ]
    colors = ["#3f6f9d", "#3e8d68", "#c98337", "#b24b3c"]
    for idx, (key, label) in enumerate(rate_panels):
        axes[0].bar(
            [v + (idx - 1.5) * width for v in x],
            [row[key] for row in rows],
            width=width,
            label=label,
            color=colors[idx],
            edgecolor="#222222",
            linewidth=0.6,
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Rate across seed groups")
    axes[0].set_title("Local proxy completion/improvement rates")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].bar(
        [v - width / 2 for v in x],
        [row["reward_delta_vs_denoised_mean"] for row in rows],
        width=width,
        label="reward delta",
        color="#3f6f9d",
        edgecolor="#222222",
        linewidth=0.6,
    )
    axes[1].bar(
        [v + width / 2 for v in x],
        [row["error_delta_vs_denoised_mean"] for row in rows],
        width=width,
        label="error delta",
        color="#b24b3c",
        edgecolor="#222222",
        linewidth=0.6,
    )
    axes[1].axhline(0.0, color="#222222", linewidth=0.8)
    axes[1].set_xticks(x, labels)
    axes[1].set_title("Guided minus denoised means")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("Full-bundle task-conditioned guidance: local proxy success boundary")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = load_json(SOURCE_JSON)
    rows = build_rows(summary)
    aggregate = build_aggregate(rows)
    rows_csv = OUT / "local_proxy_success_boundary_rows.csv"
    aggregate_csv = OUT / "local_proxy_success_boundary_aggregate.csv"
    plot_png = OUT / "local_proxy_success_boundary_rates.png"
    readme = OUT / "README.md"
    row_fields = [
        "task",
        "seed_group",
        "seed",
        "rollout_steps",
        "completed_299",
        "guided_reward_mean",
        "denoised_reward_mean",
        "vae_base_reward_mean",
        "guided_target_body_error_mean",
        "denoised_target_body_error_mean",
        "vae_base_target_body_error_mean",
        "reward_delta_vs_denoised",
        "error_delta_vs_denoised",
        "reward_delta_vs_vae_base",
        "error_delta_vs_vae_base",
        "guidance_cost_delta_mean",
        "guidance_signal_positive",
        "guided_base_action_mse_mean",
        "action_changed",
        "reward_improved_vs_denoised",
        "tracking_error_not_worse_vs_denoised",
        "local_proxy_pass",
        "mp4",
        "claim_level",
    ]
    aggregate_fields = [
        "task",
        "row_count",
        "completion_rate_299",
        "guidance_signal_positive_rate",
        "action_changed_rate",
        "reward_improved_vs_denoised_rate",
        "tracking_error_not_worse_vs_denoised_rate",
        "local_proxy_pass_rate",
        "reward_delta_vs_denoised_mean",
        "error_delta_vs_denoised_mean",
        "reward_delta_vs_vae_base_mean",
        "error_delta_vs_vae_base_mean",
        "guidance_cost_delta_mean",
        "guided_base_action_mse_mean",
    ]
    write_csv(rows_csv, rows, row_fields)
    write_csv(aggregate_csv, aggregate, aggregate_fields)
    plot_aggregate(aggregate, plot_png)
    readme.write_text(
        "\n".join(
            [
                "# Local proxy success boundary for full-bundle guidance",
                "",
                "This folder summarizes the five-seed local virtual task-conditioned latent-guidance rollouts.",
                "The rates are local proxy diagnostics only: 299-step completion, positive guidance signal,",
                "action change, guided reward improvement over the denoised baseline, and guided tracking error",
                "not worsening relative to the denoised baseline.",
                "",
                "Claim boundary: these are not official BeyondMimic Fig. 5/Fig. 6 success rates, not official",
                "checkpoints, not TensorRT deployment metrics, and not real-robot validation.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    checks = {
        "source_status_ok": summary["status"]
        == "ok_official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval",
        "row_count_20": len(rows) == 20,
        "five_seed_groups": summary["metrics"]["seed_group_count"] == 5,
        "four_tasks": summary["metrics"]["task_count"] == 4,
        "all_rows_completed_299": all(row["completed_299"] for row in rows),
        "all_guidance_signals_positive": all(row["guidance_signal_positive"] for row in rows),
        "all_rows_have_mp4_paths": all(Path(row["mp4"]).is_file() and Path(row["mp4"]).stat().st_size > 0 for row in rows),
        "does_not_claim_fig5_fig6": True,
        "does_not_claim_real_robot": True,
        "does_not_claim_goal_complete": True,
    }
    payload = {
        "status": "ok_official_csv_loop_full_bundle_task_conditioned_guidance_success_boundary",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_summary": str(SOURCE_JSON),
        "rows": rows,
        "aggregate": aggregate,
        "metrics": {
            "row_count": len(rows),
            "task_count": len(aggregate),
            "seed_group_count": summary["metrics"]["seed_group_count"],
            "overall_completion_rate_299": rate([row["completed_299"] for row in rows]),
            "overall_guidance_signal_positive_rate": rate([row["guidance_signal_positive"] for row in rows]),
            "overall_action_changed_rate": rate([row["action_changed"] for row in rows]),
            "overall_reward_improved_vs_denoised_rate": rate([row["reward_improved_vs_denoised"] for row in rows]),
            "overall_tracking_error_not_worse_vs_denoised_rate": rate(
                [row["tracking_error_not_worse_vs_denoised"] for row in rows]
            ),
            "overall_local_proxy_pass_rate": rate([row["local_proxy_pass"] for row in rows]),
        },
        "assets": {
            "rows_csv": str(rows_csv),
            "aggregate_csv": str(aggregate_csv),
            "plot_png": str(plot_png),
            "readme": str(readme),
        },
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "claim_level": "local_proxy_success_boundary_not_paper_fig5_fig6",
            "why_not_paper_level": (
                "The proxy rates summarize local virtual rollouts with local checkpoints, local proxy tasks, "
                "and an enriched USD scaffold. They are useful for the reading report and PPT, but they are not "
                "the official BeyondMimic Fig. 5/Fig. 6 success/fall/collision protocol."
            ),
        },
    }
    out_json = OUT / "local_proxy_success_boundary.json"
    write_json(out_json, payload)
    print(json.dumps({"status": payload["status"], "json": str(out_json)}, sort_keys=True))


if __name__ == "__main__":
    main()
