#!/usr/bin/env python3
"""Build scaled PPO local Fig. 5/6 success/fall/collision proxy assets.

The source traces do not contain the paper's official task-success, fall, or
collision labels. This script therefore creates explicit local proxy metrics
from already-generated scaled PPO closed-loop traces and records the boundary
in the output JSON.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SOURCE_JSON = (
    ROOT
    / "res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval/"
    "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval.json"
)
OUT = ROOT / "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy"

VARIANT = "receding_latent_guided"
BASELINE = "denoised_latent"
EXPECTED_STEPS = 299
THRESHOLDS = {
    "expected_steps": EXPECTED_STEPS,
    "root_final_xy_success_threshold_m": 0.02,
    "target_body_mean_success_threshold": 0.35,
    "target_body_p95_success_threshold": 0.36,
    "positive_guidance_cost_delta_threshold": 0.0,
    "fall_relative_root_height_drop_threshold_m": -0.25,
    "body_error_spike_anomaly_threshold": 0.50,
}

ROW_FIELDS = [
    "task",
    "seed_group",
    "seed",
    "summary_json",
    "trace_npz",
    "mp4",
    "trace_npz_exists",
    "mp4_exists",
    "rollout_steps",
    "completed_299",
    "root_final_xy_error_m",
    "root_mean_xy_error_m",
    "target_body_error_mean",
    "target_body_error_p95",
    "target_body_error_max",
    "reward_mean",
    "denoised_reward_mean",
    "reward_delta_vs_denoised",
    "tracking_error_delta_vs_denoised",
    "guidance_cost_delta_mean",
    "guidance_signal_positive",
    "min_root_height_m",
    "min_reference_root_height_m",
    "min_relative_root_height_m",
    "max_abs_relative_root_height_m",
    "done_flag_sum",
    "fall_height_proxy",
    "body_error_spike_anomaly_proxy",
    "collision_contact_signal_available",
    "collision_proxy_available",
    "success_proxy",
    "failure_reason",
    "claim_level",
]

AGG_FIELDS = [
    "task",
    "row_count",
    "seed_group_count",
    "success_proxy_rate",
    "fall_height_proxy_rate",
    "body_error_spike_anomaly_proxy_rate",
    "completed_299_rate",
    "guidance_signal_positive_rate",
    "root_final_xy_success_rate",
    "target_body_mean_success_rate",
    "target_body_p95_success_rate",
    "root_final_xy_error_m_mean",
    "target_body_error_mean_mean",
    "target_body_error_p95_mean",
    "reward_delta_vs_denoised_mean",
    "tracking_error_delta_vs_denoised_mean",
    "min_relative_root_height_m_min",
    "target_body_error_max_max",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


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


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return mean(1.0 if bool(row[key]) else 0.0 for row in rows) if rows else 0.0


def numeric(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) not in (None, "")]


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    vals = numeric(rows, key)
    if not vals:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": mean(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def metric(trace: np.lib.npyio.NpzFile, variant: str, name: str) -> np.ndarray:
    return np.asarray(trace[f"{variant}_{name}"], dtype=np.float64)


def body_pos(trace: np.lib.npyio.NpzFile, variant: str, kind: str) -> np.ndarray:
    return np.asarray(trace[f"{variant}_{kind}_body_pos_w"], dtype=np.float64)


def analyze_row(source_row: dict[str, Any]) -> dict[str, Any]:
    summary_path = Path(source_row["summary_json"])
    summary = load_json(summary_path)
    trace_path = Path(summary["outputs"]["capture_npz"])
    mp4_path = Path(summary.get("outputs", {}).get("assets", {}).get("mp4") or source_row.get("mp4", ""))
    seed = source_row.get("seed") or summary.get("config", {}).get("seed", "")

    trace_exists = trace_path.is_file()
    mp4_exists = mp4_path.is_file() and mp4_path.stat().st_size > 0
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
            "collision_contact_signal_available": False,
            "collision_proxy_available": False,
            "success_proxy": False,
            "failure_reason": "missing_trace_npz",
            "claim_level": "local_proxy_failed_missing_trace_npz",
        }

    with np.load(trace_path) as trace:
        robot = body_pos(trace, VARIANT, "robot")
        reference = body_pos(trace, VARIANT, "reference")
        target_error = metric(trace, VARIANT, "target_body_error_mean")
        reward = metric(trace, VARIANT, "rewards")
        dones = metric(trace, VARIANT, "dones")
        guidance_cost_delta = metric(trace, VARIANT, "guidance_cost_before") - metric(
            trace, VARIANT, "guidance_cost_after"
        )
        denoised_reward = metric(trace, BASELINE, "rewards")
        denoised_error = metric(trace, BASELINE, "target_body_error_mean")

    rollout_steps = int(robot.shape[0])
    root_xy = robot[:, 0, :2]
    ref_root_xy = reference[:, 0, :2]
    root_xy_error = np.linalg.norm(root_xy - ref_root_xy, axis=1)
    root_height = robot[:, 0, 2]
    ref_root_height = reference[:, 0, 2]
    relative_root_height = root_height - ref_root_height

    root_final_ok = float(root_xy_error[-1]) <= THRESHOLDS["root_final_xy_success_threshold_m"]
    target_mean_ok = float(target_error.mean()) <= THRESHOLDS["target_body_mean_success_threshold"]
    target_p95 = float(np.percentile(target_error, 95))
    target_p95_ok = target_p95 <= THRESHOLDS["target_body_p95_success_threshold"]
    guidance_positive = float(guidance_cost_delta.mean()) > THRESHOLDS["positive_guidance_cost_delta_threshold"]
    completed_299 = rollout_steps == EXPECTED_STEPS
    fall_proxy = float(relative_root_height.min()) < THRESHOLDS["fall_relative_root_height_drop_threshold_m"]
    body_error_spike_proxy = float(target_error.max()) > THRESHOLDS["body_error_spike_anomaly_threshold"]

    failure_reasons: list[str] = []
    if not completed_299:
        failure_reasons.append("not_299_steps")
    if not root_final_ok:
        failure_reasons.append("endpoint_proxy_fail")
    if not target_mean_ok:
        failure_reasons.append("target_body_mean_proxy_fail")
    if not target_p95_ok:
        failure_reasons.append("target_body_p95_proxy_fail")
    if not guidance_positive:
        failure_reasons.append("guidance_signal_nonpositive")
    if fall_proxy:
        failure_reasons.append("fall_height_proxy")
    if body_error_spike_proxy:
        failure_reasons.append("body_error_spike_anomaly_proxy")
    success_proxy = not failure_reasons and mp4_exists
    if not mp4_exists:
        failure_reasons.append("missing_mp4")

    return {
        "task": source_row["task"],
        "seed_group": source_row["seed_group"],
        "seed": seed,
        "summary_json": rel(summary_path),
        "trace_npz": rel(trace_path),
        "mp4": rel(mp4_path),
        "trace_npz_exists": trace_exists,
        "mp4_exists": mp4_exists,
        "rollout_steps": rollout_steps,
        "completed_299": completed_299,
        "root_final_xy_error_m": float(root_xy_error[-1]),
        "root_mean_xy_error_m": float(root_xy_error.mean()),
        "target_body_error_mean": float(target_error.mean()),
        "target_body_error_p95": target_p95,
        "target_body_error_max": float(target_error.max()),
        "reward_mean": float(reward.mean()),
        "denoised_reward_mean": float(denoised_reward.mean()),
        "reward_delta_vs_denoised": float(reward.mean() - denoised_reward.mean()),
        "tracking_error_delta_vs_denoised": float(target_error.mean() - denoised_error.mean()),
        "guidance_cost_delta_mean": float(guidance_cost_delta.mean()),
        "guidance_signal_positive": guidance_positive,
        "min_root_height_m": float(root_height.min()),
        "min_reference_root_height_m": float(ref_root_height.min()),
        "min_relative_root_height_m": float(relative_root_height.min()),
        "max_abs_relative_root_height_m": float(np.abs(relative_root_height).max()),
        "done_flag_sum": float(dones.sum()),
        "fall_height_proxy": fall_proxy,
        "body_error_spike_anomaly_proxy": body_error_spike_proxy,
        "collision_contact_signal_available": False,
        "collision_proxy_available": False,
        "success_proxy": success_proxy,
        "failure_reason": ";".join(failure_reasons) if failure_reasons else "",
        "claim_level": "local_virtual_scaled_ppo_success_fall_collision_proxy_not_paper_level",
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
                "success_proxy_rate": rate(task_rows, "success_proxy"),
                "fall_height_proxy_rate": rate(task_rows, "fall_height_proxy"),
                "body_error_spike_anomaly_proxy_rate": rate(task_rows, "body_error_spike_anomaly_proxy"),
                "completed_299_rate": rate(task_rows, "completed_299"),
                "guidance_signal_positive_rate": rate(task_rows, "guidance_signal_positive"),
                "root_final_xy_success_rate": mean(
                    1.0
                    if row["root_final_xy_error_m"] <= THRESHOLDS["root_final_xy_success_threshold_m"]
                    else 0.0
                    for row in task_rows
                ),
                "target_body_mean_success_rate": mean(
                    1.0
                    if row["target_body_error_mean"] <= THRESHOLDS["target_body_mean_success_threshold"]
                    else 0.0
                    for row in task_rows
                ),
                "target_body_p95_success_rate": mean(
                    1.0
                    if row["target_body_error_p95"] <= THRESHOLDS["target_body_p95_success_threshold"]
                    else 0.0
                    for row in task_rows
                ),
                "root_final_xy_error_m_mean": mean(numeric(task_rows, "root_final_xy_error_m")),
                "target_body_error_mean_mean": mean(numeric(task_rows, "target_body_error_mean")),
                "target_body_error_p95_mean": mean(numeric(task_rows, "target_body_error_p95")),
                "reward_delta_vs_denoised_mean": mean(numeric(task_rows, "reward_delta_vs_denoised")),
                "tracking_error_delta_vs_denoised_mean": mean(numeric(task_rows, "tracking_error_delta_vs_denoised")),
                "min_relative_root_height_m_min": min(numeric(task_rows, "min_relative_root_height_m")),
                "target_body_error_max_max": max(numeric(task_rows, "target_body_error_max")),
            }
        )
    return out


def plot_rates(path: Path, aggregate: list[dict[str, Any]]) -> None:
    tasks = [row["task"].replace("_", "\n") for row in aggregate]
    x = np.arange(len(tasks))
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    bars = [
        ("success_proxy_rate", "success proxy", "#3f6f9d"),
        ("fall_height_proxy_rate", "fall proxy", "#b24b3c"),
        ("body_error_spike_anomaly_proxy_rate", "body-error spike", "#c98337"),
        ("completed_299_rate", "299-step", "#3e8d68"),
    ]
    for idx, (key, label, color) in enumerate(bars):
        axes[0].bar(x + (idx - 1.5) * width, [row[key] for row in aggregate], width, label=label, color=color)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xticks(x, tasks)
    axes[0].set_ylabel("Rate across seed groups")
    axes[0].set_title("Scaled PPO local success/fall/anomaly proxies")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].bar(x - width / 2, [row["reward_delta_vs_denoised_mean"] for row in aggregate], width, label="reward delta")
    axes[1].bar(
        x + width / 2,
        [row["tracking_error_delta_vs_denoised_mean"] for row in aggregate],
        width,
        label="tracking error delta",
    )
    axes[1].axhline(0.0, color="#222222", linewidth=0.8)
    axes[1].set_xticks(x, tasks)
    axes[1].set_title("Guided minus denoised")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("Local proxy metrics only, not official BeyondMimic Fig. 5/6")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    rows = [
        "| task | seeds | success proxy | fall proxy | body-error spike | 299-step |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["aggregate"]:
        rows.append(
            "| {task} | {seed_group_count} | {success_proxy_rate:.2f} | {fall_height_proxy_rate:.2f} | "
            "{body_error_spike_anomaly_proxy_rate:.2f} | {completed_299_rate:.2f} |".format(**row)
        )
    text = "\n".join(
        [
            "# Scaled PPO Fig. 5/Fig. 6 Success/Fall/Collision Proxy",
            "",
            "This asset summarizes existing scaled PPO closed-loop local virtual traces. It does not use the paper's",
            "official Fig. 5/Fig. 6 success, fall, or collision labels.",
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
            "## Collision Boundary",
            "",
            "The traces do not include contact or obstacle-collision sensor channels. The reported body-error spike",
            "rate is an anomaly proxy only and must not be described as a paper collision rate.",
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = load_json(SOURCE_JSON)
    rows = [analyze_row(row) for row in source["rows"]]
    aggregate = aggregate_rows(rows)
    rows_csv = OUT / "success_fall_collision_proxy_rows.csv"
    aggregate_csv = OUT / "success_fall_collision_proxy_aggregate.csv"
    plot_png = OUT / "success_fall_collision_proxy_rates.png"
    md_path = OUT / "success_fall_collision_proxy.md"
    readme_path = OUT / "README.md"
    json_path = OUT / "success_fall_collision_proxy.json"

    write_csv(rows_csv, rows, ROW_FIELDS)
    write_csv(aggregate_csv, aggregate, AGG_FIELDS)
    plot_rates(plot_png, aggregate)
    metrics = {
        "row_count": len(rows),
        "task_count": len({row["task"] for row in rows}),
        "seed_group_count": len({row["seed_group"] for row in rows}),
        "trace_npz_count": sum(1 for row in rows if row["trace_npz_exists"]),
        "mp4_count": sum(1 for row in rows if row["mp4_exists"]),
        "overall_success_proxy_rate": rate(rows, "success_proxy"),
        "overall_fall_height_proxy_rate": rate(rows, "fall_height_proxy"),
        "overall_body_error_spike_anomaly_proxy_rate": rate(rows, "body_error_spike_anomaly_proxy"),
        "overall_completed_299_rate": rate(rows, "completed_299"),
        "overall_guidance_signal_positive_rate": rate(rows, "guidance_signal_positive"),
        "paper_level_success_rate_available": False,
        "paper_level_fall_rate_available": False,
        "paper_level_collision_rate_available": False,
        "collision_contact_signal_available": False,
        "root_final_xy_error_m": summarize(rows, "root_final_xy_error_m"),
        "target_body_error_mean": summarize(rows, "target_body_error_mean"),
        "target_body_error_max": summarize(rows, "target_body_error_max"),
        "min_relative_root_height_m": summarize(rows, "min_relative_root_height_m"),
    }
    assets = {
        "json": str(json_path),
        "rows_csv": str(rows_csv),
        "aggregate_csv": str(aggregate_csv),
        "plot_png": str(plot_png),
        "markdown": str(md_path),
        "readme": str(readme_path),
    }
    checks = {
        "source_status_ok": source["status"]
        == "ok_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval",
        "row_count_20": metrics["row_count"] == 20,
        "task_count_4": metrics["task_count"] == 4,
        "seed_group_count_5": metrics["seed_group_count"] == 5,
        "all_trace_npz_exist": metrics["trace_npz_count"] == 20,
        "all_mp4_paths_exist": metrics["mp4_count"] == 20,
        "all_rows_completed_299": all(row["completed_299"] for row in rows),
        "guidance_signal_positive_rate_recorded": metrics["overall_guidance_signal_positive_rate"] >= 0.0,
        "success_proxy_rate_recorded": metrics["overall_success_proxy_rate"] >= 0.0,
        "fall_proxy_rate_recorded": metrics["overall_fall_height_proxy_rate"] >= 0.0,
        "collision_contact_signal_unavailable_recorded": metrics["collision_contact_signal_available"] is False,
        "does_not_claim_paper_success_rate": metrics["paper_level_success_rate_available"] is False,
        "does_not_claim_paper_fall_rate": metrics["paper_level_fall_rate_available"] is False,
        "does_not_claim_paper_collision_rate": metrics["paper_level_collision_rate_available"] is False,
        "does_not_claim_real_robot": True,
        "does_not_claim_goal_complete": True,
    }
    payload = {
        "status": "ok_official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy"
        if all(checks.values())
        else "failed_official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_json": str(SOURCE_JSON),
        "thresholds": THRESHOLDS,
        "rows": rows,
        "aggregate": aggregate,
        "metrics": metrics,
        "assets": assets,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "claim_level": "local_virtual_scaled_ppo_success_fall_collision_proxy_not_paper_level",
            "reading_report_use": (
                "Use this as a stricter local virtual summary of the scaled PPO Fig. 5/Fig. 6-adjacent rollouts. "
                "It is useful for reporting completion/fall/anomaly proxies, but it is not the official paper protocol."
            ),
            "why_not_paper_level": [
                "Official BeyondMimic Fig. 5/Fig. 6 success/fall/collision definitions are not public.",
                "The local traces do not contain contact/collision sensor labels.",
                "The thresholds are local analysis thresholds over local virtual traces.",
                "The checkpoints are local scaled PPO/VAE/denoiser checkpoints, not official BeyondMimic checkpoints.",
                "No real robot or mocap/real-world context is used.",
            ],
        },
    }
    write_markdown(md_path, payload)
    readme_path.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
    checks["assets_exist"] = all(
        Path(path).is_file() and Path(path).stat().st_size > 0
        for key, path in assets.items()
        if key != "json"
    )
    payload["checks"] = checks
    payload["status"] = (
        "ok_official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy"
        if all(checks.values())
        else "failed_official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy"
    )
    write_json(json_path, payload)
    print(json.dumps({"status": payload["status"], "json": str(json_path), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
