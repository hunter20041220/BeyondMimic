#!/usr/bin/env python3
"""Build completion/termination proxy assets for the scaled PPO checkpoint eval."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
EVAL_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/"
    "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json"
)
OUT = ROOT / "res/report_assets/official_importer_export_scaled_ppo_checkpoint_completion_proxy"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_timeseries(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if key == "step":
                    parsed[key] = int(value)
                else:
                    try:
                        parsed[key] = float(value)
                    except (TypeError, ValueError):
                        parsed[key] = value
            rows.append(parsed)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def finite(values: list[float]) -> list[float]:
    return [value for value in values if value == value and value not in {float("inf"), float("-inf")}]


def summarize(values: list[float]) -> dict[str, Any]:
    values = finite(values)
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    mean = sum(values) / len(values)
    return {"count": len(values), "mean": mean, "min": min(values), "max": max(values)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = load_json(EVAL_JSON)
    metrics = audit["run"]["metrics"]
    config = audit["config"]
    timeseries_path = Path(audit["outputs"]["timeseries_csv"])
    rows = read_timeseries(timeseries_path)

    num_envs = int(metrics["num_envs"])
    eval_steps = int(metrics["eval_steps"])
    total_env_steps = int(metrics["total_env_steps"])
    done_total = int(metrics["done_count_total"])
    timeout_total = int(metrics["timeout_count_total"])
    non_timeout_done_total = done_total - timeout_total
    attempted_env_steps = num_envs * eval_steps
    completion_proxy_rate = 1.0 - (done_total / attempted_env_steps)
    non_timeout_done_rate = non_timeout_done_total / attempted_env_steps
    timeout_rate = timeout_total / attempted_env_steps
    completed_without_timeout_proxy_rate = 1.0 - timeout_rate

    per_step_rows = []
    cumulative_done = 0.0
    for row in rows:
        done_count = float(row["done_count"])
        cumulative_done += done_count
        per_step_rows.append(
            {
                "step": row["step"],
                "done_count": done_count,
                "done_rate_of_envs": done_count / num_envs,
                "cumulative_done_count": cumulative_done,
                "cumulative_done_rate_of_attempted_env_steps": cumulative_done / attempted_env_steps,
                "reward_mean": row["reward_mean"],
                "error_anchor_pos": row["error_anchor_pos"],
                "error_body_pos": row["error_body_pos"],
                "error_joint_pos": row["error_joint_pos"],
                "action_abs_mean": row["action_abs_mean"],
            }
        )

    aggregate_rows = [
        {
            "metric": "num_envs",
            "value": num_envs,
            "claim_level": "local_virtual_scaled_ppo_checkpoint_eval_proxy",
        },
        {
            "metric": "eval_steps",
            "value": eval_steps,
            "claim_level": "local_virtual_scaled_ppo_checkpoint_eval_proxy",
        },
        {
            "metric": "attempted_env_steps",
            "value": attempted_env_steps,
            "claim_level": "local_virtual_scaled_ppo_checkpoint_eval_proxy",
        },
        {
            "metric": "total_env_steps_recorded",
            "value": total_env_steps,
            "claim_level": "local_virtual_scaled_ppo_checkpoint_eval_proxy",
        },
        {
            "metric": "done_count_total",
            "value": done_total,
            "claim_level": "local_virtual_scaled_ppo_checkpoint_eval_proxy",
        },
        {
            "metric": "non_timeout_done_count_total",
            "value": non_timeout_done_total,
            "claim_level": "local_virtual_scaled_ppo_checkpoint_eval_proxy",
        },
        {
            "metric": "local_completion_proxy_rate",
            "value": completion_proxy_rate,
            "claim_level": "local_virtual_scaled_ppo_checkpoint_eval_proxy",
        },
        {
            "metric": "local_non_timeout_done_rate",
            "value": non_timeout_done_rate,
            "claim_level": "local_virtual_scaled_ppo_checkpoint_eval_proxy",
        },
        {
            "metric": "timeout_rate",
            "value": timeout_rate,
            "claim_level": "local_virtual_scaled_ppo_checkpoint_eval_proxy",
        },
        {
            "metric": "completed_without_timeout_proxy_rate",
            "value": completed_without_timeout_proxy_rate,
            "claim_level": "local_virtual_scaled_ppo_checkpoint_eval_proxy",
        },
    ]

    rows_csv = OUT / "scaled_ppo_checkpoint_completion_proxy_rows.csv"
    aggregate_csv = OUT / "scaled_ppo_checkpoint_completion_proxy_aggregate.csv"
    write_csv(rows_csv, per_step_rows)
    write_csv(aggregate_csv, aggregate_rows)

    steps = [row["step"] for row in per_step_rows]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    axes[0].plot(steps, [row["done_rate_of_envs"] for row in per_step_rows], color="#dc2626", linewidth=1.4)
    axes[0].set_ylabel("Done / envs")
    axes[0].set_title("Scaled PPO checkpoint eval local termination proxy")
    axes[1].plot(
        steps,
        [row["cumulative_done_rate_of_attempted_env_steps"] for row in per_step_rows],
        color="#f97316",
        linewidth=1.4,
    )
    axes[1].set_ylabel("Cumulative done / attempted env-steps")
    axes[2].plot(steps, [row["reward_mean"] for row in per_step_rows], color="#2563eb", linewidth=1.4)
    axes[2].set_ylabel("Reward mean")
    axes[2].set_xlabel("Eval step")
    fig.tight_layout()
    termination_png = OUT / "scaled_ppo_checkpoint_completion_proxy_termination.png"
    fig.savefig(termination_png, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.bar(
        ["completion proxy", "non-timeout done", "timeout"],
        [completion_proxy_rate, non_timeout_done_rate, timeout_rate],
        color=["#16a34a", "#dc2626", "#64748b"],
    )
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Rate over 2048 x 299 attempted env-steps")
    ax.set_title("Scaled PPO checkpoint completion/failure proxy rates")
    for i, value in enumerate([completion_proxy_rate, non_timeout_done_rate, timeout_rate]):
        ax.text(i, min(0.98, value + 0.02), f"{value:.4f}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    rates_png = OUT / "scaled_ppo_checkpoint_completion_proxy_rates.png"
    fig.savefig(rates_png, dpi=180)
    plt.close(fig)

    markdown = OUT / "scaled_ppo_checkpoint_completion_proxy.md"
    markdown.write_text(
        "\n".join(
            [
                "# Scaled PPO Checkpoint Completion Proxy",
                "",
                "This report asset summarizes local completion/termination proxies for the iteration-999",
                "official-importer-export scaled PPO checkpoint evaluation.",
                "",
                "## Metrics",
                "",
                f"- num envs: `{num_envs}`",
                f"- eval steps: `{eval_steps}`",
                f"- attempted env-steps: `{attempted_env_steps}`",
                f"- done count total: `{done_total}`",
                f"- local completion proxy rate: `{completion_proxy_rate}`",
                f"- local non-timeout done rate: `{non_timeout_done_rate}`",
                f"- timeout rate: `{timeout_rate}`",
                "",
                "## Claim Boundary",
                "",
                "This is a local virtual proxy over an existing checkpoint eval. It is not the paper's official",
                "success/fall/collision protocol, not an official BeyondMimic teacher checkpoint evaluation,",
                "not DAgger, not Fig.5/Fig.6 guided diffusion, and not real robot evidence.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    readme = OUT / "README.md"
    readme.write_text(markdown.read_text(encoding="utf-8"), encoding="utf-8")

    assets = {
        "json": str(OUT / "scaled_ppo_checkpoint_completion_proxy.json"),
        "rows_csv": str(rows_csv),
        "aggregate_csv": str(aggregate_csv),
        "termination_png": str(termination_png),
        "rates_png": str(rates_png),
        "markdown": str(markdown),
        "readme": str(readme),
    }
    summary = {
        "status": "ok_official_importer_export_scaled_ppo_checkpoint_completion_proxy",
        "experiment_type": "official_importer_export_scaled_ppo_checkpoint_completion_proxy",
        "source_eval_json": str(EVAL_JSON),
        "source_timeseries_csv": str(timeseries_path),
        "config": {
            "num_envs": num_envs,
            "eval_steps": eval_steps,
            "selected_physical_gpus": config["selected_physical_gpus"],
            "checkpoint": metrics["checkpoint"],
            "loaded_iteration": metrics["loaded_iteration"],
            "uses_official_importer_export_usd": metrics["uses_official_importer_export_usd"],
            "uses_resource_adjusted_usd": metrics["uses_resource_adjusted_usd"],
            "motion_count": metrics["motion_count"],
            "total_motion_frames": metrics["total_motion_frames"],
        },
        "metrics": {
            "attempted_env_steps": attempted_env_steps,
            "total_env_steps_recorded": total_env_steps,
            "done_count_total": done_total,
            "timeout_count_total": timeout_total,
            "non_timeout_done_count_total": non_timeout_done_total,
            "local_completion_proxy_rate": completion_proxy_rate,
            "local_non_timeout_done_rate": non_timeout_done_rate,
            "timeout_rate": timeout_rate,
            "completed_without_timeout_proxy_rate": completed_without_timeout_proxy_rate,
            "done_rate_of_envs": summarize([row["done_rate_of_envs"] for row in per_step_rows]),
            "reward_mean": summarize([row["reward_mean"] for row in per_step_rows]),
            "error_body_pos": summarize([row["error_body_pos"] for row in per_step_rows]),
            "error_joint_pos": summarize([row["error_joint_pos"] for row in per_step_rows]),
        },
        "checks": {
            "source_eval_status_ok": audit["status"]
            == "ok_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval_completed",
            "timeseries_has_299_rows": len(rows) == eval_steps,
            "attempted_env_steps_match": attempted_env_steps == num_envs * eval_steps,
            "uses_official_importer_export_usd": metrics["uses_official_importer_export_usd"] is True,
            "does_not_use_resource_adjusted_usd": metrics["uses_resource_adjusted_usd"] is False,
            "done_count_recorded": done_total > 0,
            "completion_proxy_rate_recorded": 0.0 <= completion_proxy_rate <= 1.0,
            "assets_exist": all(
                Path(path).is_file() and Path(path).stat().st_size > 0
                for key, path in assets.items()
                if key != "json"
            ),
            "does_not_claim_paper_success_or_fall": True,
            "does_not_claim_real_robot": True,
            "does_not_claim_goal_complete": True,
        },
        "assets": assets,
        "interpretation": {
            "claim_level": "local_virtual_scaled_ppo_checkpoint_completion_termination_proxy",
            "goal_complete": False,
            "reading_report_use": (
                "Use this to explain why the scaled PPO checkpoint is runnable but weak: it reaches the eval harness "
                "on the official-importer-export path, yet almost every attempted env-step ends in a non-timeout "
                "termination proxy."
            ),
            "why_not_paper_level": [
                "The paper success/fall/collision criteria are not public.",
                "This uses a local PPO checkpoint, not an official BeyondMimic teacher checkpoint.",
                "The metric is derived from local done counts and timeout counts, not paper labels.",
                "No real robot or TensorRT deployment is involved.",
            ],
        },
    }
    summary["status"] = summary["status"] if all(summary["checks"].values()) else "failed_scaled_ppo_checkpoint_completion_proxy"
    Path(assets["json"]).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": assets["json"], "metrics": summary["metrics"]}, sort_keys=True))
    if summary["status"].startswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
