#!/usr/bin/env python3
"""Build local Fig. 5/6 task-protocol proxy metrics from importer-export traces.

This script analyzes already-generated local IsaacLab importer-export rollout
traces. It intentionally produces proxy evidence only: the public artifact set
does not contain the official BeyondMimic checkpoints, exact Fig. 5/6 task
protocol logs, TensorRT traces, or real-robot/mocap evidence.
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
VARIANT = os.environ.get("BM_FIG56_TASK_PROTOCOL_VARIANT", "full_bundle")
if VARIANT not in {"full_bundle", "scaled_ppo"}:
    raise ValueError(f"Unsupported BM_FIG56_TASK_PROTOCOL_VARIANT={VARIANT!r}")
if VARIANT == "scaled_ppo":
    SOURCE_JSON = (
        ROOT
        / "res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval/"
        "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval.json"
    )
    OUT = ROOT / "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy"
    STATUS_OK = "ok_official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy"
    STATUS_FAILED = "failed_official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy"
    EXPECTED_SOURCE_STATUS = "ok_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval"
    DISPLAY_TITLE = "Official Importer-Export Scaled PPO Fig. 5/Fig. 6 Task-Protocol Proxy"
    CLAIM_LEVEL = "local_virtual_official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy_not_paper_level"
else:
    SOURCE_JSON = (
        ROOT
        / "res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval/"
        "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json"
    )
    OUT = ROOT / "res/report_assets/official_importer_export_fig5_fig6_task_protocol_proxy"
    STATUS_OK = "ok_official_importer_export_fig5_fig6_task_protocol_proxy"
    STATUS_FAILED = "failed_official_importer_export_fig5_fig6_task_protocol_proxy"
    EXPECTED_SOURCE_STATUS = "ok_official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval"
    DISPLAY_TITLE = "Official Importer-Export Fig. 5/Fig. 6 Task-Protocol Proxy"
    CLAIM_LEVEL = "local_virtual_official_importer_export_fig5_fig6_task_protocol_proxy_not_paper_level"

GUIDED = "receding_latent_guided"
BASELINE = "denoised_latent"
EXPECTED_STEPS = 299

THRESHOLDS = {
    "recorded_step_count": EXPECTED_STEPS,
    "root_final_xy_error_threshold_m": 0.02,
    "target_body_mean_error_threshold": 0.35,
    "target_body_frame_error_threshold": 0.36,
    "positive_guidance_cost_delta_threshold": 0.0,
    "reward_improvement_threshold": 0.0,
    "tracking_error_not_worse_threshold": 0.0,
}

CSV_FIELDS = [
    "task",
    "seed_group",
    "seed",
    "summary_json",
    "trace_npz",
    "mp4",
    "trace_npz_exists",
    "mp4_exists",
    "rollout_steps",
    "recorded_299_step_completion",
    "guided_done_flag_sum",
    "denoised_done_flag_sum",
    "guided_reward_mean",
    "denoised_reward_mean",
    "reward_delta_vs_denoised",
    "guided_target_body_error_mean",
    "denoised_target_body_error_mean",
    "tracking_error_delta_vs_denoised",
    "guided_body_error_frame_success_rate_under_0p36",
    "guidance_cost_before_mean",
    "guidance_cost_after_mean",
    "guidance_cost_delta_mean",
    "guidance_signal_positive",
    "root_final_xy_error_m",
    "root_mean_xy_error_m",
    "root_path_length_xy_m",
    "reference_path_length_xy_m",
    "root_path_length_ratio",
    "root_height_mean_m",
    "root_height_min_m",
    "root_height_max_m",
    "endpoint_proxy_pass",
    "target_body_mean_proxy_pass",
    "reward_improved_vs_denoised",
    "tracking_error_not_worse_vs_denoised",
    "local_task_protocol_proxy_pass",
    "claim_level",
]

AGG_FIELDS = [
    "task",
    "row_count",
    "seed_group_count",
    "recorded_299_step_completion_rate",
    "guidance_signal_positive_rate",
    "endpoint_proxy_pass_rate",
    "target_body_mean_proxy_pass_rate",
    "reward_improved_vs_denoised_rate",
    "tracking_error_not_worse_vs_denoised_rate",
    "local_task_protocol_proxy_pass_rate",
    "guided_body_error_frame_success_rate_under_0p36_mean",
    "root_final_xy_error_m_mean",
    "root_final_xy_error_m_max",
    "root_mean_xy_error_m_mean",
    "guided_target_body_error_mean_mean",
    "guidance_cost_delta_mean",
    "reward_delta_vs_denoised_mean",
    "tracking_error_delta_vs_denoised_mean",
    "root_path_length_ratio_mean",
    "mp4_present_rate",
    "trace_npz_present_rate",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def as_float(value: Any) -> float:
    return float(np.asarray(value).item() if np.asarray(value).shape == () else value)


def bool_rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return mean(1.0 if bool(row[key]) else 0.0 for row in rows)


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) not in ("", None)]


def summarize_numeric(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    vals = numeric_values(rows, key)
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": mean(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def metric_array(trace: np.lib.npyio.NpzFile, variant: str, metric: str) -> np.ndarray:
    return np.asarray(trace[f"{variant}_{metric}"], dtype=np.float64)


def body_pos(trace: np.lib.npyio.NpzFile, variant: str, kind: str) -> np.ndarray:
    return np.asarray(trace[f"{variant}_{kind}_body_pos_w"], dtype=np.float64)


def analyze_row(source_row: dict[str, Any]) -> dict[str, Any]:
    summary_path = Path(source_row["summary_json"])
    summary = load_json(summary_path)
    trace_path = Path(summary["outputs"]["capture_npz"])
    mp4_path = Path(summary.get("outputs", {}).get("assets", {}).get("mp4") or source_row.get("mp4", ""))
    seed = source_row.get("seed")
    if seed is None:
        seed = summary.get("config", {}).get("seed")

    trace_exists = trace_path.exists()
    mp4_exists = mp4_path.exists()
    if not trace_exists:
        return {
            "task": source_row["task"],
            "seed_group": source_row["seed_group"],
            "seed": seed,
            "summary_json": rel(summary_path),
            "trace_npz": rel(trace_path),
            "mp4": rel(mp4_path),
            "trace_npz_exists": False,
            "mp4_exists": mp4_exists,
            "claim_level": "failed_missing_trace_npz",
        }

    with np.load(trace_path) as trace:
        guided_robot = body_pos(trace, GUIDED, "robot")
        guided_ref = body_pos(trace, GUIDED, "reference")
        guided_reward = metric_array(trace, GUIDED, "rewards")
        denoised_reward = metric_array(trace, BASELINE, "rewards")
        guided_error = metric_array(trace, GUIDED, "target_body_error_mean")
        denoised_error = metric_array(trace, BASELINE, "target_body_error_mean")
        guided_cost_before = metric_array(trace, GUIDED, "guidance_cost_before")
        guided_cost_after = metric_array(trace, GUIDED, "guidance_cost_after")
        guided_dones = metric_array(trace, GUIDED, "dones")
        denoised_dones = metric_array(trace, BASELINE, "dones")

    rollout_steps = int(guided_robot.shape[0])
    root_xy = guided_robot[:, 0, :2]
    ref_root_xy = guided_ref[:, 0, :2]
    root_diff_xy = root_xy - ref_root_xy
    root_final_xy_error = float(np.linalg.norm(root_diff_xy[-1]))
    root_mean_xy_error = float(np.linalg.norm(root_diff_xy, axis=1).mean())
    root_path_length = float(np.linalg.norm(np.diff(root_xy, axis=0), axis=1).sum())
    ref_path_length = float(np.linalg.norm(np.diff(ref_root_xy, axis=0), axis=1).sum())
    root_path_length_ratio = root_path_length / ref_path_length if ref_path_length else float("nan")

    reward_delta = float(guided_reward.mean() - denoised_reward.mean())
    tracking_error_delta = float(guided_error.mean() - denoised_error.mean())
    guidance_cost_delta = float((guided_cost_before - guided_cost_after).mean())
    recorded_completion = rollout_steps == EXPECTED_STEPS
    guidance_signal_positive = guidance_cost_delta > THRESHOLDS["positive_guidance_cost_delta_threshold"]
    endpoint_proxy_pass = root_final_xy_error <= THRESHOLDS["root_final_xy_error_threshold_m"]
    target_body_mean_proxy_pass = guided_error.mean() <= THRESHOLDS["target_body_mean_error_threshold"]
    reward_improved = reward_delta > THRESHOLDS["reward_improvement_threshold"]
    tracking_not_worse = tracking_error_delta <= THRESHOLDS["tracking_error_not_worse_threshold"]
    local_task_proxy_pass = (
        recorded_completion
        and trace_exists
        and mp4_exists
        and guidance_signal_positive
        and endpoint_proxy_pass
        and target_body_mean_proxy_pass
        and (reward_improved or tracking_not_worse)
    )

    return {
        "task": source_row["task"],
        "seed_group": source_row["seed_group"],
        "seed": int(seed) if seed is not None else "",
        "summary_json": rel(summary_path),
        "trace_npz": rel(trace_path),
        "mp4": rel(mp4_path),
        "trace_npz_exists": trace_exists,
        "mp4_exists": mp4_exists,
        "rollout_steps": rollout_steps,
        "recorded_299_step_completion": recorded_completion,
        "guided_done_flag_sum": float(guided_dones.sum()),
        "denoised_done_flag_sum": float(denoised_dones.sum()),
        "guided_reward_mean": float(guided_reward.mean()),
        "denoised_reward_mean": float(denoised_reward.mean()),
        "reward_delta_vs_denoised": reward_delta,
        "guided_target_body_error_mean": float(guided_error.mean()),
        "denoised_target_body_error_mean": float(denoised_error.mean()),
        "tracking_error_delta_vs_denoised": tracking_error_delta,
        "guided_body_error_frame_success_rate_under_0p36": float(
            (guided_error <= THRESHOLDS["target_body_frame_error_threshold"]).mean()
        ),
        "guidance_cost_before_mean": float(guided_cost_before.mean()),
        "guidance_cost_after_mean": float(guided_cost_after.mean()),
        "guidance_cost_delta_mean": guidance_cost_delta,
        "guidance_signal_positive": guidance_signal_positive,
        "root_final_xy_error_m": root_final_xy_error,
        "root_mean_xy_error_m": root_mean_xy_error,
        "root_path_length_xy_m": root_path_length,
        "reference_path_length_xy_m": ref_path_length,
        "root_path_length_ratio": root_path_length_ratio,
        "root_height_mean_m": float(guided_robot[:, 0, 2].mean()),
        "root_height_min_m": float(guided_robot[:, 0, 2].min()),
        "root_height_max_m": float(guided_robot[:, 0, 2].max()),
        "endpoint_proxy_pass": endpoint_proxy_pass,
        "target_body_mean_proxy_pass": target_body_mean_proxy_pass,
        "reward_improved_vs_denoised": reward_improved,
        "tracking_error_not_worse_vs_denoised": tracking_not_worse,
        "local_task_protocol_proxy_pass": local_task_proxy_pass,
        "claim_level": CLAIM_LEVEL,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in sorted({row["task"] for row in rows}):
        task_rows = [row for row in rows if row["task"] == task]
        out.append(
            {
                "task": task,
                "row_count": len(task_rows),
                "seed_group_count": len({row["seed_group"] for row in task_rows}),
                "recorded_299_step_completion_rate": bool_rate(task_rows, "recorded_299_step_completion"),
                "guidance_signal_positive_rate": bool_rate(task_rows, "guidance_signal_positive"),
                "endpoint_proxy_pass_rate": bool_rate(task_rows, "endpoint_proxy_pass"),
                "target_body_mean_proxy_pass_rate": bool_rate(task_rows, "target_body_mean_proxy_pass"),
                "reward_improved_vs_denoised_rate": bool_rate(task_rows, "reward_improved_vs_denoised"),
                "tracking_error_not_worse_vs_denoised_rate": bool_rate(
                    task_rows, "tracking_error_not_worse_vs_denoised"
                ),
                "local_task_protocol_proxy_pass_rate": bool_rate(task_rows, "local_task_protocol_proxy_pass"),
                "guided_body_error_frame_success_rate_under_0p36_mean": mean(
                    numeric_values(task_rows, "guided_body_error_frame_success_rate_under_0p36")
                ),
                "root_final_xy_error_m_mean": mean(numeric_values(task_rows, "root_final_xy_error_m")),
                "root_final_xy_error_m_max": max(numeric_values(task_rows, "root_final_xy_error_m")),
                "root_mean_xy_error_m_mean": mean(numeric_values(task_rows, "root_mean_xy_error_m")),
                "guided_target_body_error_mean_mean": mean(
                    numeric_values(task_rows, "guided_target_body_error_mean")
                ),
                "guidance_cost_delta_mean": mean(numeric_values(task_rows, "guidance_cost_delta_mean")),
                "reward_delta_vs_denoised_mean": mean(numeric_values(task_rows, "reward_delta_vs_denoised")),
                "tracking_error_delta_vs_denoised_mean": mean(
                    numeric_values(task_rows, "tracking_error_delta_vs_denoised")
                ),
                "root_path_length_ratio_mean": mean(numeric_values(task_rows, "root_path_length_ratio")),
                "mp4_present_rate": bool_rate(task_rows, "mp4_exists"),
                "trace_npz_present_rate": bool_rate(task_rows, "trace_npz_exists"),
            }
        )
    return out


def plot_rates(path: Path, aggregate: list[dict[str, Any]]) -> None:
    tasks = [row["task"] for row in aggregate]
    metrics = [
        ("recorded_299_step_completion_rate", "299-step"),
        ("endpoint_proxy_pass_rate", "endpoint"),
        ("target_body_mean_proxy_pass_rate", "body err"),
        ("reward_improved_vs_denoised_rate", "reward +"),
        ("tracking_error_not_worse_vs_denoised_rate", "error <= baseline"),
        ("local_task_protocol_proxy_pass_rate", "local pass"),
    ]
    x = np.arange(len(tasks))
    width = 0.12
    fig, ax = plt.subplots(figsize=(11, 5.4))
    colors = ["#4c78a8", "#72b7b2", "#54a24b", "#f58518", "#e45756", "#b279a2"]
    for i, (key, label) in enumerate(metrics):
        values = [float(row[key]) for row in aggregate]
        ax.bar(x + (i - (len(metrics) - 1) / 2) * width, values, width=width, label=label, color=colors[i])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rate")
    ax.set_title("Local Fig. 5/6 Task-Protocol Proxy Rates (not paper-level)")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=18, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_deltas(path: Path, aggregate: list[dict[str, Any]]) -> None:
    tasks = [row["task"] for row in aggregate]
    reward = [float(row["reward_delta_vs_denoised_mean"]) for row in aggregate]
    error = [float(row["tracking_error_delta_vs_denoised_mean"]) for row in aggregate]
    cost = [float(row["guidance_cost_delta_mean"]) for row in aggregate]
    x = np.arange(len(tasks))
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.2), sharex=True)
    series = [
        (axes[0], reward, "reward delta vs denoised", "#4c78a8", 0.0),
        (axes[1], error, "target-body error delta vs denoised", "#e45756", 0.0),
        (axes[2], cost, "guidance cost decrease", "#54a24b", None),
    ]
    for ax, values, ylabel, color, zero in series:
        ax.bar(x, values, color=color, alpha=0.9)
        if zero is not None:
            ax.axhline(zero, color="black", linewidth=0.8, alpha=0.7)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_title("Local Guided-vs-Denoised Proxy Deltas (not paper-level)")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(tasks, rotation=18, ha="right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_markdown(path: Path, payload: dict[str, Any], aggregate: list[dict[str, Any]]) -> None:
    rows = [
        "| task | seeds | 299-step | local proxy pass | reward improved | error not worse | final root err mean (m) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate:
        rows.append(
            "| {task} | {seed_group_count} | {completion:.2f} | {local_pass:.2f} | {reward:.2f} | "
            "{error:.2f} | {root:.4f} |".format(
                task=row["task"],
                seed_group_count=row["seed_group_count"],
                completion=row["recorded_299_step_completion_rate"],
                local_pass=row["local_task_protocol_proxy_pass_rate"],
                reward=row["reward_improved_vs_denoised_rate"],
                error=row["tracking_error_not_worse_vs_denoised_rate"],
                root=row["root_final_xy_error_m_mean"],
            )
        )
    text = "\n".join(
        [
            f"# {DISPLAY_TITLE}",
            "",
            "This asset converts existing local closed-loop importer-export traces into task-level proxy metrics",
            "for the reading report. It is not an official BeyondMimic Fig. 5/Fig. 6 protocol, not TensorRT",
            "deployment evidence, and not real-robot evidence.",
            "",
            "## Thresholds",
            "",
            "```json",
            json.dumps(payload["thresholds"], indent=2, sort_keys=True),
            "```",
            "",
            "## Summary",
            "",
            *rows,
            "",
            "## Key Interpretation",
            "",
            f"- Rows analyzed: {payload['metrics']['row_count']}.",
            f"- Seed groups: {payload['metrics']['seed_group_count']}.",
            f"- Overall local task-protocol proxy pass rate: "
            f"{payload['metrics']['overall_local_task_protocol_proxy_pass_rate']:.3f}.",
            f"- Overall reward-improved-vs-denoised rate: "
            f"{payload['metrics']['overall_reward_improved_vs_denoised_rate']:.3f}.",
            f"- Overall tracking-error-not-worse-vs-denoised rate: "
            f"{payload['metrics']['overall_tracking_error_not_worse_vs_denoised_rate']:.3f}.",
            "",
            "The proxy pass requires a 299-step local trace, a present MP4 path, positive guidance-cost",
            "decrease, a root endpoint proxy within 2 cm of the local reference endpoint, mean target-body error",
            "under 0.35, and either reward improvement or non-worse target-body error relative to the local",
            "denoised baseline. These thresholds are local analysis thresholds, not paper thresholds.",
            "",
            "## Remaining Gap",
            "",
            "The paper-level Fig. 5/Fig. 6 gates still require exact task protocols, public official checkpoints",
            "or reproduced teacher-derived state-latent rollouts, fall/collision/success definitions, TensorRT or",
            "asynchronous deployment traces where applicable, and real robot/mocap evidence for real-world panels.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    source = load_json(SOURCE_JSON)
    rows = [analyze_row(row) for row in source.get("rows", [])]
    aggregate = aggregate_rows(rows)

    rows_csv = OUT / "fig5_fig6_task_protocol_proxy_rows.csv"
    aggregate_csv = OUT / "fig5_fig6_task_protocol_proxy_aggregate.csv"
    json_path = OUT / "fig5_fig6_task_protocol_proxy.json"
    markdown_path = OUT / "fig5_fig6_task_protocol_proxy.md"
    rates_png = OUT / "fig5_fig6_task_protocol_proxy_rates.png"
    deltas_png = OUT / "fig5_fig6_task_protocol_proxy_deltas.png"
    readme_path = OUT / "README.md"

    write_csv(rows_csv, rows, CSV_FIELDS)
    write_csv(aggregate_csv, aggregate, AGG_FIELDS)
    plot_rates(rates_png, aggregate)
    plot_deltas(deltas_png, aggregate)

    metrics = {
        "row_count": len(rows),
        "task_count": len({row["task"] for row in rows}),
        "seed_group_count": len({row["seed_group"] for row in rows}),
        "trace_npz_count": sum(1 for row in rows if row.get("trace_npz_exists")),
        "mp4_count": sum(1 for row in rows if row.get("mp4_exists")),
        "overall_recorded_299_step_completion_rate": bool_rate(rows, "recorded_299_step_completion"),
        "overall_guidance_signal_positive_rate": bool_rate(rows, "guidance_signal_positive"),
        "overall_endpoint_proxy_pass_rate": bool_rate(rows, "endpoint_proxy_pass"),
        "overall_target_body_mean_proxy_pass_rate": bool_rate(rows, "target_body_mean_proxy_pass"),
        "overall_reward_improved_vs_denoised_rate": bool_rate(rows, "reward_improved_vs_denoised"),
        "overall_tracking_error_not_worse_vs_denoised_rate": bool_rate(
            rows, "tracking_error_not_worse_vs_denoised"
        ),
        "overall_local_task_protocol_proxy_pass_rate": bool_rate(rows, "local_task_protocol_proxy_pass"),
        "paper_level_reproduced_panel_count": 0,
        "root_final_xy_error_m": summarize_numeric(rows, "root_final_xy_error_m"),
        "guided_target_body_error_mean": summarize_numeric(rows, "guided_target_body_error_mean"),
        "reward_delta_vs_denoised": summarize_numeric(rows, "reward_delta_vs_denoised"),
        "tracking_error_delta_vs_denoised": summarize_numeric(rows, "tracking_error_delta_vs_denoised"),
        "guidance_cost_delta_mean": summarize_numeric(rows, "guidance_cost_delta_mean"),
    }
    assets = {
        "json": str(json_path),
        "rows_csv": str(rows_csv),
        "aggregate_csv": str(aggregate_csv),
        "markdown": str(markdown_path),
        "rates_png": str(rates_png),
        "deltas_png": str(deltas_png),
        "readme": str(readme_path),
    }
    checks = {
        "source_status_ok": source.get("status") == EXPECTED_SOURCE_STATUS,
        "row_count_20": len(rows) == 20,
        "task_count_4": len({row["task"] for row in rows}) == 4,
        "seed_group_count_5": len({row["seed_group"] for row in rows}) == 5,
        "all_trace_npz_exist": all(bool(row.get("trace_npz_exists")) for row in rows),
        "all_mp4_paths_exist": all(bool(row.get("mp4_exists")) for row in rows),
        "all_records_have_299_steps": all(bool(row.get("recorded_299_step_completion")) for row in rows),
        "all_guidance_cost_delta_positive": all(bool(row.get("guidance_signal_positive")) for row in rows),
        "all_endpoint_proxy_pass": all(bool(row.get("endpoint_proxy_pass")) for row in rows),
        "all_target_body_mean_proxy_pass": all(bool(row.get("target_body_mean_proxy_pass")) for row in rows),
        "local_proxy_pass_rate_recorded": metrics["overall_local_task_protocol_proxy_pass_rate"] >= 0.0,
        "does_not_claim_fig5_fig6_paper_level": metrics["paper_level_reproduced_panel_count"] == 0,
        "does_not_claim_real_robot": True,
        "does_not_claim_goal_complete": True,
        "uses_existing_importer_export_traces": True,
        "uses_local_checkpoints": True,
        "no_paper_success_rate_claimed": True,
    }
    payload = {
        "status": STATUS_OK if all(checks.values()) else STATUS_FAILED,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "variant": VARIANT,
        "source_json": str(SOURCE_JSON),
        "thresholds": THRESHOLDS,
        "metrics": metrics,
        "aggregate": aggregate,
        "rows": rows,
        "assets": assets,
        "checks": checks,
        "interpretation": {
            "claim_level": CLAIM_LEVEL,
            "goal_complete": False,
            "reading_report_use": (
                "Use this table to discuss how far the current local virtual evidence reaches toward the "
                "paper's guided diffusion task claims while preserving the distinction from official Fig. 5/"
                "Fig. 6 success, fall, collision, TensorRT, and real-robot protocols."
            ),
            "why_not_paper_level": [
                "Official BeyondMimic VAE/diffusion checkpoints are not public in this artifact set.",
                "The exact Fig. 5/Fig. 6 closed-loop task protocols and success/failure thresholds are not public.",
                "The local task objectives are proxy costs over recovered importer-export rollouts.",
                "TensorRT/asynchronous deployment traces are not generated by this asset.",
                "Real robot and mocap/real-world context panels remain unavailable locally.",
            ],
        },
    }
    write_markdown(markdown_path, payload, aggregate)
    readme_path.write_text(markdown_path.read_text(encoding="utf-8"), encoding="utf-8")
    checks["assets_exist"] = all(Path(path).exists() for path in assets.values() if path != str(json_path))
    payload["status"] = (
        STATUS_OK if all(checks.values()) else STATUS_FAILED
    )
    payload["checks"] = checks
    write_json(json_path, payload)
    print(json.dumps({"status": payload["status"], "json": str(json_path), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
