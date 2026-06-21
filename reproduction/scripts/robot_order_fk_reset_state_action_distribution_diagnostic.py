#!/usr/bin/env python3
"""Compare reset/action/velocity traces across same-seed robot-order FK evals."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/robot_order_fk_reset_state_action_distribution_diagnostic"

VARIANTS = {
    "baseline": (
        "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval/"
        "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval.json"
    ),
    "reset_command_warmup_seed_matched": (
        "res/tracking/"
        "g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_seed_matched/"
        "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_seed_matched.json"
    ),
    "reset_target_refresh_no_advance": (
        "res/tracking/"
        "g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance/"
        "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance.json"
    ),
}

METRIC_COLUMNS = [
    "reward_mean",
    "reward_min",
    "reward_max",
    "done_count",
    "timeout_count",
    "action_abs_mean",
    "action_abs_max",
    "error_anchor_pos",
    "error_anchor_rot",
    "error_anchor_lin_vel",
    "error_anchor_ang_vel",
    "error_body_pos",
    "error_body_rot",
    "error_body_lin_vel",
    "error_body_ang_vel",
    "error_joint_pos",
    "error_joint_vel",
    "sampling_entropy",
    "sampling_top1_prob",
    "sampling_top1_bin",
]

WINDOWS = {
    "step0": lambda row: int(row["step"]) == 0,
    "step1": lambda row: int(row["step"]) == 1,
    "step2": lambda row: int(row["step"]) == 2,
    "first5": lambda row: int(row["step"]) < 5,
    "first20": lambda row: int(row["step"]) < 20,
    "post_step0": lambda row: int(row["step"]) > 0,
    "all": lambda row: True,
}


def load_json(rel_path: str | Path) -> dict[str, Any]:
    path = Path(rel_path)
    if not path.is_absolute():
        path = ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    tmp.replace(path)


def read_timeseries(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            converted: dict[str, float] = {}
            for key, value in row.items():
                converted[key] = float(value) if value != "" else float("nan")
            rows.append(converted)
    return rows


def mean(values: list[float]) -> float | None:
    finite = [float(v) for v in values if v == v]
    if not finite:
        return None
    return sum(finite) / len(finite)


def summarize_window(rows: list[dict[str, float]], num_envs: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "row_count": len(rows),
        "num_envs": num_envs,
        "env_step_count": len(rows) * num_envs,
    }
    if not rows:
        return summary
    done_total = sum(row.get("done_count", 0.0) for row in rows)
    summary["done_count_total"] = done_total
    summary["done_rate"] = done_total / float(len(rows) * num_envs)
    for column in METRIC_COLUMNS:
        values = [row[column] for row in rows if column in row]
        if not values:
            continue
        summary[column] = mean(values)
        summary[f"{column}_min"] = min(values)
        summary[f"{column}_max"] = max(values)
    return summary


def flatten_window_row(variant: str, window: str, summary: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "row_count",
        "env_step_count",
        "done_count_total",
        "done_rate",
        "reward_mean",
        "action_abs_mean",
        "action_abs_max",
        "error_body_pos",
        "error_body_ang_vel",
        "error_body_lin_vel",
        "error_joint_pos",
        "error_joint_vel",
        "error_anchor_pos",
        "error_anchor_ang_vel",
        "sampling_top1_bin",
        "sampling_top1_prob",
    ]
    row = {"variant": variant, "window": window}
    for field in fields:
        row[field] = summary.get(field)
    return row


def get_mean_metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, dict):
        if "mean" in value:
            return float(value["mean"])
        nested = value.get("mean_over_steps")
        if isinstance(nested, dict) and "mean" in nested:
            return float(nested["mean"])
    if isinstance(value, (int, float)):
        return float(value)
    return None


def extract_termination_rows(variant: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    num_envs = int(metrics.get("num_envs", 1) or 1)
    rows: list[dict[str, Any]] = []
    for key, value in sorted(metrics.get("episode_log_metrics", {}).items()):
        if not key.startswith("Episode_Termination/") or not isinstance(value, dict):
            continue
        component = key.removeprefix("Episode_Termination/")
        mean_count = value.get("mean")
        if mean_count is None:
            continue
        rows.append(
            {
                "variant": variant,
                "component": component,
                "mean_count_per_step": float(mean_count),
                "fraction_of_envs_per_step": float(mean_count) / float(num_envs),
                "num_envs": num_envs,
            }
        )
    return rows


def extract_motion_metric_means(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.get("motion_metrics", {}).items():
        if isinstance(value, dict) and "mean" in value:
            out[key] = float(value["mean"])
    return out


def make_markdown(diagnostic: dict[str, Any], window_rows: list[dict[str, Any]], delta_rows: list[dict[str, Any]]) -> str:
    metrics = diagnostic["metrics"]
    interpretation = diagnostic["interpretation"]
    lines = [
        "# Robot-Order FK Reset State/Action Distribution Diagnostic",
        "",
        "## Scope",
        "",
        (
            "This diagnostic reads the existing same-seed full-evaluation traces for the robot-order "
            "FK-repaired local PPO checkpoint. It does not launch Isaac Sim, does not train, and does "
            "not claim paper-level tracking."
        ),
        "",
        "## Key Metrics",
        "",
    ]
    for key in [
        "baseline_step0_body_error",
        "target_refresh_step0_body_error",
        "target_refresh_step0_joint_vel",
        "baseline_step0_joint_vel",
        "target_refresh_first5_action_abs_mean_delta",
        "target_refresh_post_step0_done_rate_delta",
        "target_refresh_ee_body_pos_termination_fraction_delta",
    ]:
        lines.append(f"- `{key}`: `{metrics.get(key)}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Primary bottleneck: {interpretation['primary_bottleneck']}",
            f"- Recommended next experiment: {interpretation['recommended_next_experiment']}",
            "",
            "## Window Summary",
            "",
            "| variant | window | done_rate | reward_mean | action_abs_mean | body_pos | joint_vel | body_ang_vel |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in window_rows:
        if row["window"] not in {"step0", "first5", "post_step0", "all"}:
            continue
        lines.append(
            "| {variant} | {window} | {done_rate} | {reward_mean} | {action_abs_mean} | "
            "{error_body_pos} | {error_joint_vel} | {error_body_ang_vel} |".format(
                **{k: row.get(k, "") for k in row}
            )
        )
    lines.extend(
        [
            "",
            "## Deltas Versus Baseline",
            "",
            "| comparison | window | done_rate_delta | action_abs_mean_delta | body_pos_delta | joint_vel_delta |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in delta_rows:
        if row["window"] not in {"step0", "first5", "post_step0", "all"}:
            continue
        lines.append(
            "| {comparison} | {window} | {done_rate_delta} | {action_abs_mean_delta} | "
            "{error_body_pos_delta} | {error_joint_vel_delta} |".format(
                **{k: row.get(k, "") for k in row}
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    variant_summaries: dict[str, dict[str, Any]] = {}
    window_rows: list[dict[str, Any]] = []
    termination_rows: list[dict[str, Any]] = []

    for variant, rel_path in VARIANTS.items():
        audit = load_json(rel_path)
        metrics_json = Path(audit["outputs"]["metrics_json"])
        timeseries_csv = Path(audit["outputs"]["timeseries_csv"])
        metrics = load_json(metrics_json)
        timeseries = read_timeseries(timeseries_csv)
        num_envs = int(audit["config"]["num_envs"])
        windows: dict[str, dict[str, Any]] = {}
        for window_name, predicate in WINDOWS.items():
            selected = [row for row in timeseries if predicate(row)]
            windows[window_name] = summarize_window(selected, num_envs)
            window_rows.append(flatten_window_row(variant, window_name, windows[window_name]))
        termination = extract_termination_rows(variant, metrics)
        termination_rows.extend(termination)
        variant_summaries[variant] = {
            "status": audit["status"],
            "config": audit["config"],
            "outputs": {
                "json": str(ROOT / rel_path),
                "metrics_json": str(metrics_json),
                "timeseries_csv": str(timeseries_csv),
            },
            "windows": windows,
            "termination_fractions": {
                row["component"]: row["fraction_of_envs_per_step"] for row in termination
            },
            "motion_metric_means": extract_motion_metric_means(metrics),
        }

    delta_rows: list[dict[str, Any]] = []
    baseline = variant_summaries["baseline"]
    for variant in ["reset_command_warmup_seed_matched", "reset_target_refresh_no_advance"]:
        for window_name in WINDOWS:
            base_window = baseline["windows"][window_name]
            other_window = variant_summaries[variant]["windows"][window_name]
            delta = {
                "comparison": f"{variant}_minus_baseline",
                "window": window_name,
            }
            for key in [
                "done_rate",
                "reward_mean",
                "action_abs_mean",
                "action_abs_max",
                "error_body_pos",
                "error_body_ang_vel",
                "error_body_lin_vel",
                "error_joint_pos",
                "error_joint_vel",
                "error_anchor_pos",
                "error_anchor_ang_vel",
                "sampling_top1_bin",
            ]:
                if base_window.get(key) is None or other_window.get(key) is None:
                    continue
                delta[f"{key}_delta"] = other_window[key] - base_window[key]
            delta_rows.append(delta)

    target = variant_summaries["reset_target_refresh_no_advance"]
    warmup = variant_summaries["reset_command_warmup_seed_matched"]
    metrics = {
        "same_seed": all(v["config"].get("seed") == 20260721 for v in variant_summaries.values()),
        "same_num_envs": len({v["config"].get("num_envs") for v in variant_summaries.values()}) == 1,
        "same_eval_steps": len({v["config"].get("eval_steps") for v in variant_summaries.values()}) == 1,
        "baseline_step0_body_error": baseline["windows"]["step0"].get("error_body_pos"),
        "target_refresh_step0_body_error": target["windows"]["step0"].get("error_body_pos"),
        "target_refresh_step0_body_error_delta": (
            target["windows"]["step0"].get("error_body_pos") - baseline["windows"]["step0"].get("error_body_pos")
        ),
        "baseline_step0_joint_vel": baseline["windows"]["step0"].get("error_joint_vel"),
        "target_refresh_step0_joint_vel": target["windows"]["step0"].get("error_joint_vel"),
        "target_refresh_step0_joint_vel_delta": (
            target["windows"]["step0"].get("error_joint_vel") - baseline["windows"]["step0"].get("error_joint_vel")
        ),
        "baseline_first5_action_abs_mean": baseline["windows"]["first5"].get("action_abs_mean"),
        "target_refresh_first5_action_abs_mean": target["windows"]["first5"].get("action_abs_mean"),
        "target_refresh_first5_action_abs_mean_delta": (
            target["windows"]["first5"].get("action_abs_mean")
            - baseline["windows"]["first5"].get("action_abs_mean")
        ),
        "baseline_post_step0_done_rate": baseline["windows"]["post_step0"].get("done_rate"),
        "target_refresh_post_step0_done_rate": target["windows"]["post_step0"].get("done_rate"),
        "target_refresh_post_step0_done_rate_delta": (
            target["windows"]["post_step0"].get("done_rate") - baseline["windows"]["post_step0"].get("done_rate")
        ),
        "warmup_post_step0_done_rate_delta": (
            warmup["windows"]["post_step0"].get("done_rate") - baseline["windows"]["post_step0"].get("done_rate")
        ),
        "baseline_ee_body_pos_termination_fraction": baseline["termination_fractions"].get("ee_body_pos"),
        "target_refresh_ee_body_pos_termination_fraction": target["termination_fractions"].get("ee_body_pos"),
        "target_refresh_ee_body_pos_termination_fraction_delta": (
            target["termination_fractions"].get("ee_body_pos")
            - baseline["termination_fractions"].get("ee_body_pos")
        ),
    }
    checks = {
        "all_variants_loaded": len(variant_summaries) == 3,
        "same_seed_scope_all": metrics["same_seed"] and metrics["same_num_envs"] and metrics["same_eval_steps"],
        "target_refresh_reduces_step0_body_error": metrics["target_refresh_step0_body_error_delta"] < -40.0,
        "target_refresh_records_step0_joint_velocity_spike": metrics["target_refresh_step0_joint_vel_delta"] > 10.0,
        "target_refresh_records_first5_action_transient": metrics["target_refresh_first5_action_abs_mean_delta"] > 0.02,
        "target_refresh_post_step0_done_rate_worse": metrics["target_refresh_post_step0_done_rate_delta"] > 0.0,
        "ee_body_pos_termination_recorded": metrics["target_refresh_ee_body_pos_termination_fraction"] is not None,
        "does_not_claim_paper_level_tracking": True,
        "does_not_claim_goal_complete": True,
        "does_not_claim_real_robot": True,
    }

    windows_csv = OUT / "robot_order_fk_reset_state_action_distribution_windows.csv"
    deltas_csv = OUT / "robot_order_fk_reset_state_action_distribution_deltas.csv"
    termination_csv = OUT / "robot_order_fk_reset_state_action_distribution_termination.csv"
    json_out = OUT / "robot_order_fk_reset_state_action_distribution_diagnostic.json"
    md_out = OUT / "robot_order_fk_reset_state_action_distribution_diagnostic.md"
    write_csv(windows_csv, window_rows, list(window_rows[0].keys()))
    write_csv(deltas_csv, delta_rows, sorted(delta_rows[0].keys()))
    write_csv(termination_csv, termination_rows, list(termination_rows[0].keys()))

    diagnostic = {
        "status": "ok_robot_order_fk_reset_state_action_distribution_diagnostic",
        "experiment_type": "robot_order_fk_reset_state_action_distribution_diagnostic",
        "scope": (
            "Static trace diagnostic over same-seed 2048-env x 299-step robot-order FK-repaired PPO evals; "
            "no new simulator launch, no training, no paper-level tracking claim."
        ),
        "source_variants": VARIANTS,
        "variant_summaries": variant_summaries,
        "metrics": metrics,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "paper_level_tracking_reproduced": False,
            "primary_bottleneck": (
                "No-advance target refresh removes the stale step-0 body target, but it exposes or creates a large "
                "initial joint-velocity/action transient and still worsens post-step0 done rate. The current teacher "
                "should not be used as the final DAgger/VAE/diffusion data source."
            ),
            "recommended_next_experiment": (
                "Patch reset-state and reset-action consistency before rerunning PPO: align initial joint velocities "
                "and last-action observations with the refreshed target, then rerun the full robot-order FK task eval "
                "and only proceed to full PPO if post-step0 done rate improves."
            ),
        },
        "outputs": {
            "json": str(json_out),
            "windows_csv": str(windows_csv),
            "deltas_csv": str(deltas_csv),
            "termination_csv": str(termination_csv),
            "markdown": str(md_out),
        },
    }
    diagnostic["checks"]["windows_csv_exists"] = windows_csv.is_file()
    diagnostic["checks"]["deltas_csv_exists"] = deltas_csv.is_file()
    diagnostic["checks"]["termination_csv_exists"] = termination_csv.is_file()

    write_json(json_out, diagnostic)
    md_out.write_text(make_markdown(diagnostic, window_rows, delta_rows), encoding="utf-8")

    print(json.dumps({"status": diagnostic["status"], "metrics": metrics, "checks": checks}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
