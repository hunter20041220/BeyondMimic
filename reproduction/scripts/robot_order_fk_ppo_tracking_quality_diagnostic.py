#!/usr/bin/env python3
"""Diagnose the current robot-order FK PPO tracking quality bottleneck.

This script does not run simulation. It reads the existing full-step eval and
multi-seed eval artifacts, then separates reset/bootstrap spikes from
post-bootstrap tracking quality so the next tracking fix has a concrete target.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SINGLE_EVAL = (
    ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval.json"
)
MULTISEED_EVAL = (
    ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_multiseed_eval/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_multiseed_eval.json"
)
OUT = ROOT / "res/tracking/robot_order_fk_ppo_tracking_quality_diagnostic"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    tmp.replace(path)


def fnum(value: Any) -> float:
    try:
        if value in {"", None}:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def summarize(values: list[float]) -> dict[str, Any]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = sum(finite) / len(finite)
    var = sum((value - mean) ** 2 for value in finite) / len(finite)
    return {
        "count": len(finite),
        "mean": mean,
        "std": math.sqrt(var),
        "min": min(finite),
        "max": max(finite),
    }


def mean_rows(rows: list[dict[str, str]], column: str, start_step: int = 0) -> float | None:
    vals = [fnum(row.get(column)) for row in rows if int(float(row.get("step", "0"))) >= start_step]
    finite = [value for value in vals if math.isfinite(value)]
    if not finite:
        return None
    return sum(finite) / len(finite)


def sum_rows(rows: list[dict[str, str]], column: str, start_step: int = 0) -> float:
    vals = [fnum(row.get(column)) for row in rows if int(float(row.get("step", "0"))) >= start_step]
    return sum(value for value in vals if math.isfinite(value))


def max_rows(rows: list[dict[str, str]], column: str, start_step: int = 0) -> float | None:
    vals = [fnum(row.get(column)) for row in rows if int(float(row.get("step", "0"))) >= start_step]
    finite = [value for value in vals if math.isfinite(value)]
    return max(finite) if finite else None


def first_value(rows: list[dict[str, str]], column: str) -> float | None:
    if not rows:
        return None
    value = fnum(rows[0].get(column))
    return value if math.isfinite(value) else None


def row_for_eval(label: str, seed: int | str, metrics_path: Path, timeseries_path: Path) -> dict[str, Any]:
    metrics = load_json(metrics_path)
    rows = read_csv_rows(timeseries_path)
    eval_steps = int(metrics.get("eval_steps") or len(rows) or 0)
    num_envs = int(metrics.get("num_envs") or 0)
    total_env_steps = int(metrics.get("total_env_steps") or (eval_steps * num_envs))
    step0_done = int(first_value(rows, "done_count") or 0)
    done_total = int(metrics.get("done_count_total") or sum_rows(rows, "done_count"))
    post1_done = int(sum_rows(rows, "done_count", start_step=1))
    episode_log = metrics.get("episode_log_metrics", {})
    ee_mean = episode_log.get("Episode_Termination/ee_body_pos", {}).get("mean")
    anchor_pos_mean = episode_log.get("Episode_Termination/anchor_pos", {}).get("mean")
    anchor_ori_mean = episode_log.get("Episode_Termination/anchor_ori", {}).get("mean")
    motion = metrics.get("motion_metrics", {})
    return {
        "label": label,
        "seed": seed,
        "metrics_json": str(metrics_path),
        "timeseries_csv": str(timeseries_path),
        "num_envs": num_envs,
        "eval_steps": eval_steps,
        "total_env_steps": total_env_steps,
        "done_total": done_total,
        "done_rate_all_steps": (done_total / total_env_steps) if total_env_steps else "",
        "step0_done_count": step0_done,
        "step0_done_rate": (step0_done / num_envs) if num_envs else "",
        "post_step0_done_total": post1_done,
        "post_step0_done_rate": (post1_done / max(total_env_steps - num_envs, 1)) if total_env_steps and num_envs else "",
        "timeout_total": int(metrics.get("timeout_count_total") or sum_rows(rows, "timeout_count")),
        "step0_error_body_pos": first_value(rows, "error_body_pos"),
        "step0_error_anchor_pos": first_value(rows, "error_anchor_pos"),
        "step0_error_joint_pos": first_value(rows, "error_joint_pos"),
        "mean_error_body_pos_all_steps": motion.get("error_body_pos", {}).get("mean"),
        "mean_error_body_pos_post_step0": mean_rows(rows, "error_body_pos", start_step=1),
        "max_error_body_pos_post_step0": max_rows(rows, "error_body_pos", start_step=1),
        "mean_error_anchor_pos_all_steps": motion.get("error_anchor_pos", {}).get("mean"),
        "mean_error_anchor_pos_post_step0": mean_rows(rows, "error_anchor_pos", start_step=1),
        "mean_error_joint_pos_all_steps": motion.get("error_joint_pos", {}).get("mean"),
        "mean_error_joint_pos_post_step0": mean_rows(rows, "error_joint_pos", start_step=1),
        "mean_reward_all_steps": metrics.get("reward", {}).get("mean_over_steps", {}).get("mean"),
        "mean_action_abs_all_steps": metrics.get("action_abs_mean_over_steps", {}).get("mean"),
        "mean_episode_termination_ee_body_pos": ee_mean,
        "mean_episode_termination_anchor_pos": anchor_pos_mean,
        "mean_episode_termination_anchor_ori": anchor_ori_mean,
        "uses_official_importer_export_usd": bool(metrics.get("uses_official_importer_export_usd")),
        "uses_resource_adjusted_usd": bool(metrics.get("uses_resource_adjusted_usd")),
        "paper_level_tracking_eval": bool(metrics.get("paper_level_tracking_eval")),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    single = load_json(SINGLE_EVAL)
    multi = load_json(MULTISEED_EVAL)
    eval_rows: list[dict[str, Any]] = []

    single_outputs = single.get("outputs", {})
    if single_outputs.get("metrics_json") and single_outputs.get("timeseries_csv"):
        eval_rows.append(
            row_for_eval(
                "single_seed_reference",
                single.get("config", {}).get("seed", ""),
                Path(single_outputs["metrics_json"]),
                Path(single_outputs["timeseries_csv"]),
            )
        )
    for row in multi.get("rows", []):
        if row.get("metrics_json") and row.get("timeseries_csv"):
            eval_rows.append(
                row_for_eval(
                    "multiseed_eval",
                    row.get("seed", ""),
                    Path(row["metrics_json"]),
                    Path(row["timeseries_csv"]),
                )
            )

    fields = [
        "label",
        "seed",
        "num_envs",
        "eval_steps",
        "total_env_steps",
        "done_total",
        "done_rate_all_steps",
        "step0_done_count",
        "step0_done_rate",
        "post_step0_done_total",
        "post_step0_done_rate",
        "timeout_total",
        "step0_error_body_pos",
        "step0_error_anchor_pos",
        "step0_error_joint_pos",
        "mean_error_body_pos_all_steps",
        "mean_error_body_pos_post_step0",
        "max_error_body_pos_post_step0",
        "mean_error_anchor_pos_all_steps",
        "mean_error_anchor_pos_post_step0",
        "mean_error_joint_pos_all_steps",
        "mean_error_joint_pos_post_step0",
        "mean_reward_all_steps",
        "mean_action_abs_all_steps",
        "mean_episode_termination_ee_body_pos",
        "mean_episode_termination_anchor_pos",
        "mean_episode_termination_anchor_ori",
        "uses_official_importer_export_usd",
        "uses_resource_adjusted_usd",
        "paper_level_tracking_eval",
        "metrics_json",
        "timeseries_csv",
    ]
    rows_csv = OUT / "robot_order_fk_ppo_tracking_quality_diagnostic_rows.csv"
    write_csv(rows_csv, eval_rows, fields)

    multi_rows = [row for row in eval_rows if row["label"] == "multiseed_eval"]
    aggregate = {
        "row_count": len(eval_rows),
        "multiseed_row_count": len(multi_rows),
        "step0_done_rate": summarize([fnum(row["step0_done_rate"]) for row in multi_rows]),
        "done_rate_all_steps": summarize([fnum(row["done_rate_all_steps"]) for row in multi_rows]),
        "post_step0_done_rate": summarize([fnum(row["post_step0_done_rate"]) for row in multi_rows]),
        "step0_error_body_pos": summarize([fnum(row["step0_error_body_pos"]) for row in multi_rows]),
        "mean_error_body_pos_all_steps": summarize([fnum(row["mean_error_body_pos_all_steps"]) for row in multi_rows]),
        "mean_error_body_pos_post_step0": summarize(
            [fnum(row["mean_error_body_pos_post_step0"]) for row in multi_rows]
        ),
        "mean_error_joint_pos_post_step0": summarize(
            [fnum(row["mean_error_joint_pos_post_step0"]) for row in multi_rows]
        ),
        "mean_episode_termination_ee_body_pos": summarize(
            [fnum(row["mean_episode_termination_ee_body_pos"]) for row in multi_rows]
        ),
        "mean_episode_termination_anchor_pos": summarize(
            [fnum(row["mean_episode_termination_anchor_pos"]) for row in multi_rows]
        ),
    }
    checks = {
        "single_eval_loaded": bool(single),
        "multiseed_eval_loaded": bool(multi),
        "three_multiseed_rows": len(multi_rows) == 3,
        "all_rows_have_299_steps": all(int(row["eval_steps"]) == 299 for row in eval_rows),
        "all_multiseed_step0_done_rate_one": all(abs(fnum(row["step0_done_rate"]) - 1.0) < 1e-12 for row in multi_rows),
        "step0_body_spike_gt_40m": aggregate["step0_error_body_pos"]["mean"] is not None
        and aggregate["step0_error_body_pos"]["mean"] > 40.0,
        "post_step0_body_error_lt_all_step_mean": aggregate["mean_error_body_pos_post_step0"]["mean"] is not None
        and aggregate["mean_error_body_pos_all_steps"]["mean"] is not None
        and aggregate["mean_error_body_pos_post_step0"]["mean"]
        < aggregate["mean_error_body_pos_all_steps"]["mean"],
        "post_step0_done_rate_still_high": aggregate["post_step0_done_rate"]["mean"] is not None
        and aggregate["post_step0_done_rate"]["mean"] > 0.15,
        "does_not_claim_paper_level_tracking": True,
        "does_not_claim_goal_complete": True,
    }
    next_actions = [
        "Inspect why the first eval step reports 100% done and roughly 43m body-position error despite the robot-order FK bundle.",
        "Run a controlled reset/alignment probe before policy actions: env.reset/get_observations/zero-action first step with motion IDs and body_pos_w diagnostics.",
        "Check whether the eval loop should discard the reset/bootstrap step or whether the environment reset target state is misaligned.",
        "Inspect ee_body_pos termination thresholds and endpoint body mapping because post-step0 done rate remains around 0.176 even after removing the reset spike.",
        "Do not collect final DAgger/VAE/diffusion data from this teacher until step-0 alignment and ee-body termination are understood.",
    ]
    summary = {
        "status": "ok",
        "experiment_type": "robot_order_fk_ppo_tracking_quality_diagnostic",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Post-hoc diagnostic of the current robot-order FK-repaired PPO checkpoint eval. It separates reset/"
            "bootstrap effects from post-bootstrap tracking quality and termination sources."
        ),
        "inputs": {
            "single_eval": str(SINGLE_EVAL),
            "multiseed_eval": str(MULTISEED_EVAL),
        },
        "outputs": {
            "json": str(OUT / "robot_order_fk_ppo_tracking_quality_diagnostic.json"),
            "rows_csv": str(rows_csv),
            "markdown": str(OUT / "robot_order_fk_ppo_tracking_quality_diagnostic.md"),
        },
        "aggregate": aggregate,
        "checks": checks,
        "rows": eval_rows,
        "interpretation": {
            "goal_complete": False,
            "paper_level_tracking_eval_complete": False,
            "primary_bottleneck": (
                "The current eval has a deterministic reset/bootstrap spike: every multi-seed run reports "
                "2048/2048 done at step 0 and body-position error around 43m. Removing step 0 lowers the body-position "
                "mean substantially, but the post-step0 done rate remains high, so the next fix should target reset/"
                "target alignment and ee_body_pos termination rather than downstream VAE/diffusion reruns."
            ),
            "next_actions": next_actions,
        },
    }
    write_json(OUT / "robot_order_fk_ppo_tracking_quality_diagnostic.json", summary)
    md = OUT / "robot_order_fk_ppo_tracking_quality_diagnostic.md"
    md.write_text(
        "# Robot-Order FK PPO Tracking Quality Diagnostic\n\n"
        "This is a post-hoc diagnostic for the current local virtual PPO checkpoint. It is not a paper-level "
        "BeyondMimic tracking result.\n\n"
        "## Key Findings\n\n"
        f"- Multi-seed row count: `{len(multi_rows)}`.\n"
        f"- Step-0 done-rate mean: `{aggregate['step0_done_rate']['mean']}`.\n"
        f"- Step-0 body-position error mean: `{aggregate['step0_error_body_pos']['mean']}`.\n"
        f"- All-step body-position error mean: `{aggregate['mean_error_body_pos_all_steps']['mean']}`.\n"
        f"- Post-step0 body-position error mean: `{aggregate['mean_error_body_pos_post_step0']['mean']}`.\n"
        f"- All-step done-rate mean: `{aggregate['done_rate_all_steps']['mean']}`.\n"
        f"- Post-step0 done-rate mean: `{aggregate['post_step0_done_rate']['mean']}`.\n\n"
        "## Interpretation\n\n"
        f"{summary['interpretation']['primary_bottleneck']}\n\n"
        "## Next Actions\n\n"
        + "\n".join(f"- {item}" for item in next_actions)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "json": summary["outputs"]["json"], "rows": len(eval_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
