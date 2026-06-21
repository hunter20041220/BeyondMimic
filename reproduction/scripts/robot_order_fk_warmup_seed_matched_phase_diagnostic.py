#!/usr/bin/env python3
"""Diagnose the seed-matched robot-order FK warmup eval phase effect.

This is a post-hoc analysis over completed full checkpoint-eval traces. It
does not launch IsaacLab. The goal is to decide whether reset-command warmup
actually improves the current teacher, or whether it only removes the stale
step-0 target while introducing a worse post-reset phase/termination pattern.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
NON_WARMUP = (
    ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval.json"
)
WARMUP_ORIGINAL_SEED = (
    ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup.json"
)
WARMUP_SEED_MATCHED = (
    ROOT
    / "res/tracking/"
    "g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_seed_matched/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_seed_matched.json"
)
COMMAND_SOURCE = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/commands.py"
)
TERMINATION_SOURCE = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/terminations.py"
)
OUT = ROOT / "res/tracking/robot_order_fk_warmup_seed_matched_phase_diagnostic"

SEGMENTS = [
    ("step0", 0, 0),
    ("early_1_2", 1, 2),
    ("early_3_10", 3, 10),
    ("post_reset_11_50", 11, 50),
    ("mid_51_100", 51, 100),
    ("mid_101_200", 101, 200),
    ("late_201_298", 201, 298),
]


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


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    tmp.replace(path)


def fnum(value: Any) -> float:
    try:
        if value in {"", None}:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else None


def row_subset(rows: list[dict[str, str]], start: int, end: int) -> list[dict[str, str]]:
    out = []
    for row in rows:
        step = int(fnum(row.get("step")))
        if start <= step <= end:
            out.append(row)
    return out


def sum_column(rows: list[dict[str, str]], column: str) -> float:
    return sum(value for value in (fnum(row.get(column)) for row in rows) if math.isfinite(value))


def mean_column(rows: list[dict[str, str]], column: str) -> float | None:
    return mean([fnum(row.get(column)) for row in rows])


def termination_fraction(metrics: dict[str, Any], component: str) -> float | None:
    episode = metrics.get("episode_log_metrics", {})
    payload = episode.get(f"Episode_Termination/{component}", {})
    if not isinstance(payload, dict) or payload.get("mean") is None:
        return None
    num_envs = metrics.get("num_envs") or 1
    return float(payload["mean"]) / float(num_envs)


def eval_record(label: str, audit: dict[str, Any]) -> dict[str, Any]:
    metrics = audit["run"]["metrics"]
    rows = read_csv_rows(Path(audit["outputs"]["timeseries_csv"]))
    num_envs = int(metrics["num_envs"])
    total_env_steps = int(metrics["total_env_steps"])
    done_total = int(metrics["done_count_total"])
    step0 = rows[0]
    post_rows = rows[1:]
    return {
        "label": label,
        "status": audit.get("status"),
        "seed": audit.get("config", {}).get("seed"),
        "num_envs": num_envs,
        "eval_steps": int(metrics["eval_steps"]),
        "total_env_steps": total_env_steps,
        "done_count_total": done_total,
        "done_rate": done_total / total_env_steps,
        "post_step0_done_rate": sum_column(post_rows, "done_count") / float(total_env_steps - num_envs),
        "step0_done_count": fnum(step0.get("done_count")),
        "step0_body_error": fnum(step0.get("error_body_pos")),
        "step0_joint_error": fnum(step0.get("error_joint_pos")),
        "body_error_mean": metrics["motion_metrics"]["error_body_pos"]["mean"],
        "joint_error_mean": metrics["motion_metrics"]["error_joint_pos"]["mean"],
        "post_step0_body_error_mean": mean_column(post_rows, "error_body_pos"),
        "post_step0_joint_error_mean": mean_column(post_rows, "error_joint_pos"),
        "reward_mean": metrics["reward"]["mean_over_steps"]["mean"],
        "anchor_pos_termination_fraction": termination_fraction(metrics, "anchor_pos"),
        "anchor_ori_termination_fraction": termination_fraction(metrics, "anchor_ori"),
        "ee_body_pos_termination_fraction": termination_fraction(metrics, "ee_body_pos"),
        "time_out_termination_fraction": termination_fraction(metrics, "time_out"),
        "sampling_top1_bin_post_step0_mean": mean_column(post_rows, "sampling_top1_bin"),
        "sampling_entropy_post_step0_mean": mean_column(post_rows, "sampling_entropy"),
        "timeseries_csv": audit["outputs"]["timeseries_csv"],
        "metrics_json": audit["outputs"]["metrics_json"],
    }


def segment_rows(label: str, audit: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = audit["run"]["metrics"]
    rows = read_csv_rows(Path(audit["outputs"]["timeseries_csv"]))
    num_envs = int(metrics["num_envs"])
    out: list[dict[str, Any]] = []
    for segment, start, end in SEGMENTS:
        subset = row_subset(rows, start, end)
        step_count = len(subset)
        done_count = sum_column(subset, "done_count")
        out.append(
            {
                "label": label,
                "seed": audit.get("config", {}).get("seed"),
                "segment": segment,
                "start_step": start,
                "end_step": end,
                "step_count": step_count,
                "done_count": done_count,
                "done_rate": done_count / float(max(step_count * num_envs, 1)),
                "error_body_pos_mean": mean_column(subset, "error_body_pos"),
                "error_joint_pos_mean": mean_column(subset, "error_joint_pos"),
                "error_anchor_pos_mean": mean_column(subset, "error_anchor_pos"),
                "reward_mean": mean_column(subset, "reward_mean"),
                "action_abs_mean": mean_column(subset, "action_abs_mean"),
                "sampling_top1_bin_mean": mean_column(subset, "sampling_top1_bin"),
                "sampling_entropy_mean": mean_column(subset, "sampling_entropy"),
            }
        )
    return out


def step_delta_rows(old_audit: dict[str, Any], warmup_audit: dict[str, Any]) -> list[dict[str, Any]]:
    old_rows = read_csv_rows(Path(old_audit["outputs"]["timeseries_csv"]))
    warm_rows = read_csv_rows(Path(warmup_audit["outputs"]["timeseries_csv"]))
    out: list[dict[str, Any]] = []
    for old, warm in zip(old_rows, warm_rows):
        step = int(fnum(old.get("step")))
        out.append(
            {
                "step": step,
                "old_done_count": fnum(old.get("done_count")),
                "warmup_done_count": fnum(warm.get("done_count")),
                "done_count_delta": fnum(warm.get("done_count")) - fnum(old.get("done_count")),
                "old_error_body_pos": fnum(old.get("error_body_pos")),
                "warmup_error_body_pos": fnum(warm.get("error_body_pos")),
                "error_body_pos_delta": fnum(warm.get("error_body_pos")) - fnum(old.get("error_body_pos")),
                "old_error_joint_pos": fnum(old.get("error_joint_pos")),
                "warmup_error_joint_pos": fnum(warm.get("error_joint_pos")),
                "error_joint_pos_delta": fnum(warm.get("error_joint_pos")) - fnum(old.get("error_joint_pos")),
                "old_reward_mean": fnum(old.get("reward_mean")),
                "warmup_reward_mean": fnum(warm.get("reward_mean")),
                "reward_mean_delta": fnum(warm.get("reward_mean")) - fnum(old.get("reward_mean")),
                "old_sampling_top1_bin": fnum(old.get("sampling_top1_bin")),
                "warmup_sampling_top1_bin": fnum(warm.get("sampling_top1_bin")),
                "sampling_top1_bin_delta": fnum(warm.get("sampling_top1_bin")) - fnum(old.get("sampling_top1_bin")),
            }
        )
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    old = load_json(NON_WARMUP)
    warm_original = load_json(WARMUP_ORIGINAL_SEED)
    warm_seed = load_json(WARMUP_SEED_MATCHED)
    old_record = eval_record("non_warmup_seed20260721", old)
    warm_original_record = eval_record("warmup_seed20260741", warm_original)
    warm_seed_record = eval_record("warmup_seed_matched_20260721", warm_seed)

    segments = []
    segments.extend(segment_rows("non_warmup_seed20260721", old))
    segments.extend(segment_rows("warmup_seed20260741", warm_original))
    segments.extend(segment_rows("warmup_seed_matched_20260721", warm_seed))
    segment_csv = OUT / "robot_order_fk_warmup_seed_matched_phase_segments.csv"
    write_csv(segment_csv, segments, list(segments[0].keys()))

    deltas = step_delta_rows(old, warm_seed)
    delta_csv = OUT / "robot_order_fk_warmup_seed_matched_step_deltas.csv"
    write_csv(delta_csv, deltas, list(deltas[0].keys()))

    done_delta_after_step0 = warm_seed_record["post_step0_done_rate"] - old_record["post_step0_done_rate"]
    body_delta_after_step0 = (
        warm_seed_record["post_step0_body_error_mean"] - old_record["post_step0_body_error_mean"]
    )
    ee_delta = (
        warm_seed_record["ee_body_pos_termination_fraction"] - old_record["ee_body_pos_termination_fraction"]
    )
    sampling_bin_delta = (
        warm_seed_record["sampling_top1_bin_post_step0_mean"] - old_record["sampling_top1_bin_post_step0_mean"]
    )
    summary = {
        "status": "ok_robot_order_fk_warmup_seed_matched_phase_diagnostic",
        "experiment_type": "robot_order_fk_warmup_seed_matched_phase_diagnostic",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Post-hoc full-trace analysis of non-warmup, original-seed warmup, and seed-matched warmup evals for "
            "the local robot-order FK-repaired PPO checkpoint. This decides whether the warmup intervention is a "
            "teacher-quality fix or a reset/phase diagnostic."
        ),
        "inputs": {
            "non_warmup": str(NON_WARMUP),
            "warmup_original_seed": str(WARMUP_ORIGINAL_SEED),
            "warmup_seed_matched": str(WARMUP_SEED_MATCHED),
            "command_source": str(COMMAND_SOURCE),
            "termination_source": str(TERMINATION_SOURCE),
        },
        "outputs": {
            "json": str(OUT / "robot_order_fk_warmup_seed_matched_phase_diagnostic.json"),
            "segment_csv": str(segment_csv),
            "step_delta_csv": str(delta_csv),
            "markdown": str(OUT / "robot_order_fk_warmup_seed_matched_phase_diagnostic.md"),
        },
        "evals": [old_record, warm_original_record, warm_seed_record],
        "metrics": {
            "non_warmup_done_rate": old_record["done_rate"],
            "warmup_original_seed_done_rate": warm_original_record["done_rate"],
            "warmup_seed_matched_done_rate": warm_seed_record["done_rate"],
            "same_seed_done_rate_delta": warm_seed_record["done_rate"] - old_record["done_rate"],
            "same_seed_post_step0_done_rate_delta": done_delta_after_step0,
            "step0_done_count_delta": warm_seed_record["step0_done_count"] - old_record["step0_done_count"],
            "step0_body_error_delta": warm_seed_record["step0_body_error"] - old_record["step0_body_error"],
            "same_seed_post_step0_body_error_delta": body_delta_after_step0,
            "same_seed_ee_body_pos_termination_fraction_delta": ee_delta,
            "same_seed_anchor_pos_termination_fraction_delta": (
                warm_seed_record["anchor_pos_termination_fraction"] - old_record["anchor_pos_termination_fraction"]
            ),
            "same_seed_sampling_top1_bin_post_step0_delta": sampling_bin_delta,
        },
        "checks": {
            "non_warmup_completed": old.get("status")
            == "ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_completed",
            "warmup_seed_matched_completed": warm_seed.get("status")
            == "ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_seed_matched_completed",
            "same_seed_as_non_warmup_eval": old_record["seed"] == warm_seed_record["seed"] == 20260721,
            "same_full_eval_scope": old_record["total_env_steps"] == warm_seed_record["total_env_steps"] == 612352,
            "step0_done_count_improves": warm_seed_record["step0_done_count"] < old_record["step0_done_count"],
            "step0_body_error_improves": warm_seed_record["step0_body_error"] < old_record["step0_body_error"],
            "total_done_rate_worse_after_warmup_same_seed": warm_seed_record["done_rate"] > old_record["done_rate"],
            "post_step0_done_rate_worse_after_warmup_same_seed": done_delta_after_step0 > 0.0,
            "ee_body_pos_termination_fraction_increases": ee_delta > 0.0,
            "anchor_pos_termination_fraction_does_not_increase": (
                warm_seed_record["anchor_pos_termination_fraction"] <= old_record["anchor_pos_termination_fraction"]
            ),
            "sampling_top1_bin_unchanged_same_seed": abs(sampling_bin_delta) < 1e-6,
            "segment_csv_exists": segment_csv.is_file(),
            "step_delta_csv_exists": delta_csv.is_file(),
            "does_not_claim_paper_level_tracking": True,
            "does_not_claim_goal_complete": True,
            "does_not_claim_real_robot": True,
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_tracking_eval_complete": False,
            "primary_bottleneck": (
                "Seed matching confirms that reset-command warmup is not a teacher-quality fix. It removes the "
                "stale step-0 body target spike, but post-step0 done rate and ee_body_pos termination increase while "
                "the adaptive-sampling top bin stays unchanged. The likely next target is command/observation phase "
                "consistency: refresh motion targets after reset without introducing a one-command-step mismatch, or "
                "apply the same reset warmup consistently during training and evaluation."
            ),
            "recommended_next_experiment": (
                "Run a targeted reset-target refresh variant that recomputes body_pos_relative_w at reset without "
                "advancing MotionCommand.time_steps, then only run full PPO after this termination gate improves."
            ),
        },
    }
    write_json(OUT / "robot_order_fk_warmup_seed_matched_phase_diagnostic.json", summary)

    md = OUT / "robot_order_fk_warmup_seed_matched_phase_diagnostic.md"
    md.write_text(
        "\n".join(
            [
                "# Robot-Order FK Warmup Seed-Matched Phase Diagnostic",
                "",
                "This is a post-hoc analysis over completed full checkpoint-eval traces. It does not claim paper-level tracking.",
                "",
                "## Key Findings",
                "",
                f"- Non-warmup done rate: `{old_record['done_rate']}`.",
                f"- Seed-matched warmup done rate: `{warm_seed_record['done_rate']}`.",
                f"- Same-seed done-rate delta: `{summary['metrics']['same_seed_done_rate_delta']}`.",
                f"- Step-0 done count delta: `{summary['metrics']['step0_done_count_delta']}`.",
                f"- Step-0 body-error delta: `{summary['metrics']['step0_body_error_delta']}`.",
                f"- Post-step0 done-rate delta: `{done_delta_after_step0}`.",
                f"- ee_body_pos termination fraction delta: `{ee_delta}`.",
                f"- Sampling top-bin post-step0 delta: `{sampling_bin_delta}`.",
                "",
                "## Interpretation",
                "",
                summary["interpretation"]["primary_bottleneck"],
                "",
                "## Next Experiment",
                "",
                summary["interpretation"]["recommended_next_experiment"],
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"]}, sort_keys=True))


if __name__ == "__main__":
    main()
