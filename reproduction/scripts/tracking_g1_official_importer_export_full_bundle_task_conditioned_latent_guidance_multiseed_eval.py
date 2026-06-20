#!/usr/bin/env python3
"""Run/aggregate official-importer-export multi-seed task-conditioned guidance.

This orchestrates several local virtual seed groups for the official importer
export G1 path. It reuses the validated single-seed importer-export
task-conditioned latent-guidance runner and aggregates four task-conditioned
closed-loop variants per seed group.
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = (
    ROOT
    / "reproduction/scripts/"
    "tracking_g1_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval.py"
)
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
SINGLE_SUMMARY = (
    ROOT
    / "res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval/"
    "level_c_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval.json"
)
OUT = ROOT / "res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval"
VIS_ROOT = ROOT / "res/visualization/official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_rollout"
LOG_ROOT = ROOT / "logs/tracking_g1_official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval"
FAILED_ROOT = (
    ROOT
    / "res/failed_runs/"
    "tracking_g1_official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval"
)
REPORT_OUT = ROOT / "res/report_assets/official_importer_export_full_bundle_task_conditioned_guidance_multiseed"
FULL_BUNDLE_AUDIT = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_motion_npz/"
    "tracking_g1_official_csv_loop_full_bundle_motion_npz.json"
)
OK_SINGLE_STATUS = "ok_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval"
OK_MULTI_STATUS = "ok_official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval"
FAILED_MULTI_STATUS = "failed_official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval"
TASKS = ["joystick", "waypoint", "obstacle_avoidance", "composed"]
BASELINE_SEED_GROUP = "seed_group_0_existing"
NEW_SEED_GROUPS = {
    "seed_group_1": {
        "joystick": 20260801,
        "waypoint": 20260802,
        "obstacle_avoidance": 20260803,
        "composed": 20260804,
    },
    "seed_group_2": {
        "joystick": 20260811,
        "waypoint": 20260812,
        "obstacle_avoidance": 20260813,
        "composed": 20260814,
    },
}
EXTRA_SEED_GROUPS = {
    key: {task: int(seed) for task, seed in value.items()}
    for key, value in json.loads(os.environ.get("BM_IMPORTER_EXPORT_EXTRA_SEED_GROUPS_JSON", "{}")).items()
}
REUSE_EXISTING_GROUPS = os.environ.get("BM_IMPORTER_EXPORT_REUSE_EXISTING_SEED_GROUPS", "0") == "1"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def summarize(values: list[float]) -> dict[str, Any]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = sum(finite) / len(finite)
    var = sum((value - mean) ** 2 for value in finite) / len(finite)
    return {"count": len(finite), "mean": mean, "std": math.sqrt(var), "min": min(finite), "max": max(finite)}


def safe_float(value: Any) -> float:
    try:
        if value in {"", None}:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def task_metrics_from_summary(summary: dict[str, Any], task: str, seed_group: str, seed: int | None) -> dict[str, Any]:
    metrics = summary.get("metrics", {})
    variants = metrics.get("variant_metrics", {})
    guided = variants.get("receding_latent_guided", {})
    teacher = variants.get("teacher", {})
    vae_base = variants.get("vae_base", {})
    denoised = variants.get("denoised_latent", {})
    outputs = summary.get("outputs", {})
    return {
        "seed_group": seed_group,
        "task": task,
        "seed": seed,
        "status": summary.get("status", ""),
        "selected_physical_gpu": summary.get("config", {}).get("selected_physical_gpu"),
        "rollout_steps": metrics.get("rollout_steps"),
        "guided_reward_mean": guided.get("reward_mean"),
        "teacher_reward_mean": teacher.get("reward_mean"),
        "vae_base_reward_mean": vae_base.get("reward_mean"),
        "denoised_reward_mean": denoised.get("reward_mean"),
        "guided_target_body_error_mean": guided.get("target_body_error_mean"),
        "teacher_target_body_error_mean": teacher.get("target_body_error_mean"),
        "vae_base_target_body_error_mean": vae_base.get("target_body_error_mean"),
        "denoised_target_body_error_mean": denoised.get("target_body_error_mean"),
        "guided_done_count_total": guided.get("done_count_total"),
        "teacher_done_count_total": teacher.get("done_count_total"),
        "guidance_cost_delta_mean": guided.get("guidance_cost_delta_mean"),
        "guidance_grad_norm_mean": guided.get("guidance_grad_norm_mean"),
        "guided_teacher_action_mse_mean": guided.get("guided_teacher_action_mse_mean"),
        "guided_base_action_mse_mean": guided.get("guided_base_action_mse_mean"),
        "summary_json": outputs.get("json", ""),
        "asset_json": outputs.get("asset_json", ""),
        "mp4": outputs.get("assets", {}).get("mp4", ""),
        "metrics_csv": outputs.get("assets", {}).get("metrics_csv", ""),
        "keyframes_png": outputs.get("assets", {}).get("keyframes_png", ""),
        "metrics_png": outputs.get("assets", {}).get("metrics_png", ""),
        "claim_level": "local_virtual_official_importer_export_task_conditioned_latent_guidance_multiseed",
    }


def collect_baseline_rows() -> list[dict[str, Any]]:
    summary = load_json(SINGLE_SUMMARY)
    rows: list[dict[str, Any]] = []
    for row in summary.get("rows", []):
        task = row["task"]
        task_summary = load_json(Path(row["summary_json"]))
        rows.append(task_metrics_from_summary(task_summary, task, BASELINE_SEED_GROUP, None))
    return rows


def run_seed_group(seed_group: str, task_seeds: dict[str, int]) -> list[dict[str, Any]]:
    summary_json = OUT / seed_group / f"{seed_group}_importer_export_task_conditioned_latent_guidance_rollout_eval.json"
    summary_tsv = OUT / seed_group / f"{seed_group}_importer_export_task_conditioned_latent_guidance_rollout_eval.tsv"
    process_log = LOG_ROOT / seed_group / f"{seed_group}_process.log"
    process_log.parent.mkdir(parents=True, exist_ok=True)
    if REUSE_EXISTING_GROUPS and summary_json.is_file():
        group_summary = load_json(summary_json)
        rows = []
        for row in group_summary.get("rows", []):
            task = row["task"]
            task_summary = load_json(Path(row["summary_json"]))
            rows.append(task_metrics_from_summary(task_summary, task, seed_group, task_seeds.get(task)))
        if len(rows) == len(TASKS) and all(row["status"] == OK_SINGLE_STATUS for row in rows):
            return rows

    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "BM_TASK_CONDITIONED_OUT_ROOT": str(VIS_ROOT / seed_group),
            "BM_TASK_CONDITIONED_SUMMARY_ROOT": str(OUT / seed_group),
            "BM_TASK_CONDITIONED_SUMMARY_JSON": str(summary_json),
            "BM_TASK_CONDITIONED_SUMMARY_TSV": str(summary_tsv),
            "BM_TASK_CONDITIONED_LOG_ROOT": str(LOG_ROOT / seed_group),
            "BM_TASK_CONDITIONED_FAILED_ROOT": str(FAILED_ROOT / seed_group),
            "BM_TASK_CONDITIONED_RUN_ROOT": str(RUN_ROOT / seed_group),
            "BM_TASK_CONDITIONED_TASKS": ",".join(TASKS),
            "BM_TASK_CONDITIONED_TASK_SEEDS_JSON": json.dumps(task_seeds, sort_keys=True),
        }
    )
    with process_log.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            [str(TRACKING_PY), str(BASE_SCRIPT)],
            cwd=ROOT,
            env=env,
            check=False,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    group_summary = load_json(summary_json)
    rows = []
    for row in group_summary.get("rows", []):
        task = row["task"]
        task_summary = load_json(Path(row["summary_json"]))
        rows.append(task_metrics_from_summary(task_summary, task, seed_group, task_seeds[task]))

    process = {
        "returncode": proc.returncode,
        "log": str(process_log),
        "summary_json": str(summary_json),
        "summary_tsv": str(summary_tsv),
        "group_summary_status": group_summary.get("status", ""),
    }
    if proc.returncode != 0 or not rows:
        failed_summary = {
            "status": "failed_importer_export_task_conditioned_latent_guidance_seed_group_process",
            "experiment_type": "importer_export_task_conditioned_latent_guidance_seed_group",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "seed_group": seed_group,
            "task_seeds": task_seeds,
            "rows": rows,
            "process": process,
            "checks": {
                "four_tasks_completed": len(rows) == 4 and all(row["status"] == OK_SINGLE_STATUS for row in rows),
                "does_not_claim_fig5_fig6": True,
                "does_not_claim_real_robot": True,
                "does_not_claim_goal_complete": True,
            },
        }
        write_json(summary_json, failed_summary)
        return rows

    group_summary["process"] = process
    write_json(summary_json, group_summary)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    tmp.replace(path)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_names = [
        "guided_reward_mean",
        "teacher_reward_mean",
        "vae_base_reward_mean",
        "denoised_reward_mean",
        "guided_target_body_error_mean",
        "teacher_target_body_error_mean",
        "vae_base_target_body_error_mean",
        "denoised_target_body_error_mean",
        "guided_done_count_total",
        "teacher_done_count_total",
        "guidance_cost_delta_mean",
        "guidance_grad_norm_mean",
        "guided_teacher_action_mse_mean",
        "guided_base_action_mse_mean",
    ]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    aggregate = []
    for task in TASKS:
        task_rows = by_task[task]
        item: dict[str, Any] = {"task": task, "seed_count": len(task_rows)}
        for metric in metric_names:
            stats = summarize([safe_float(row.get(metric)) for row in task_rows])
            item[f"{metric}_mean"] = stats["mean"]
            item[f"{metric}_std"] = stats["std"]
            item[f"{metric}_min"] = stats["min"]
            item[f"{metric}_max"] = stats["max"]
        aggregate.append(item)
    return aggregate


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    bundle = load_json(FULL_BUNDLE_AUDIT)
    all_rows = collect_baseline_rows()
    attempted = [{"seed_group": BASELINE_SEED_GROUP, "source": str(SINGLE_SUMMARY), "status": "reused_existing"}]
    requested_seed_groups = {**NEW_SEED_GROUPS, **EXTRA_SEED_GROUPS}
    for seed_group, task_seeds in requested_seed_groups.items():
        rows = run_seed_group(seed_group, task_seeds)
        all_rows.extend(rows)
        group_json = OUT / seed_group / f"{seed_group}_importer_export_task_conditioned_latent_guidance_rollout_eval.json"
        attempted.append(
            {
                "seed_group": seed_group,
                "task_seeds": task_seeds,
                "status": "reused_existing"
                if REUSE_EXISTING_GROUPS
                and group_json.is_file()
                and all(row["status"] == OK_SINGLE_STATUS for row in rows)
                else "ok"
                if all(row["status"] == OK_SINGLE_STATUS for row in rows)
                else "failed",
            }
        )

    row_fields = [
        "seed_group",
        "task",
        "seed",
        "status",
        "selected_physical_gpu",
        "rollout_steps",
        "guided_reward_mean",
        "teacher_reward_mean",
        "vae_base_reward_mean",
        "denoised_reward_mean",
        "guided_target_body_error_mean",
        "teacher_target_body_error_mean",
        "vae_base_target_body_error_mean",
        "denoised_target_body_error_mean",
        "guided_done_count_total",
        "teacher_done_count_total",
        "guidance_cost_delta_mean",
        "guidance_grad_norm_mean",
        "guided_teacher_action_mse_mean",
        "guided_base_action_mse_mean",
        "summary_json",
        "asset_json",
        "mp4",
        "metrics_csv",
        "keyframes_png",
        "metrics_png",
        "claim_level",
    ]
    aggregate = aggregate_rows(all_rows)
    aggregate_fields = sorted({key for row in aggregate for key in row.keys()})

    rows_csv = OUT / "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_rows.csv"
    rows_tsv = OUT / "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_rows.tsv"
    aggregate_csv = OUT / "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_aggregate.csv"
    write_csv(rows_csv, all_rows, row_fields)
    write_csv(rows_tsv, all_rows, row_fields, delimiter="\t")
    write_csv(aggregate_csv, aggregate, aggregate_fields)

    status_ok = all(row["status"] == OK_SINGLE_STATUS for row in all_rows)
    summary = {
        "status": OK_MULTI_STATUS if status_ok else FAILED_MULTI_STATUS,
        "experiment_type": "tracking_g1_official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Aggregates local virtual seed groups for four official-importer-export full-bundle closed-loop "
            "task-conditioned latent-guidance rollouts: joystick, waypoint, obstacle avoidance, and composed "
            "objectives."
        ),
        "tasks": TASKS,
        "seed_groups": [BASELINE_SEED_GROUP, *requested_seed_groups.keys()],
        "bundle": {
            "motion_count": bundle.get("bundle", {}).get("motion_count"),
            "total_frames": bundle.get("bundle", {}).get("total_frames"),
            "fps": bundle.get("bundle", {}).get("fps"),
            "clip_boundary_count": bundle.get("bundle", {}).get("boundary_count"),
        },
        "attempted": attempted,
        "rows": all_rows,
        "aggregate": aggregate,
        "metrics": {
            "task_count": len(TASKS),
            "seed_group_count": len({row["seed_group"] for row in all_rows}),
            "row_count": len(all_rows),
            "total_rollout_variant_steps": sum(int(row.get("rollout_steps") or 0) * 4 for row in all_rows),
            "video_row_count": sum(1 for row in all_rows if row.get("mp4")),
        },
        "checks": {
            "uses_full_public_motion_bundle": bundle.get("status") == "ok_official_csv_loop_full_bundle_motion_npz",
            "full_bundle_motion_count_40": bundle.get("bundle", {}).get("motion_count") == 40,
            "uses_official_importer_export_usd": True,
            "seed_group_count_at_least_3": len({row["seed_group"] for row in all_rows}) >= 3,
            "four_tasks_per_seed_group": len(all_rows) == len({row["seed_group"] for row in all_rows}) * len(TASKS),
            "all_rows_ok": status_ok,
            "all_rows_have_mp4_paths": all(bool(row.get("mp4")) for row in all_rows),
            "all_rollouts_299_steps": all(int(row.get("rollout_steps") or 0) == 299 for row in all_rows),
            "does_not_claim_fig5_fig6": True,
            "does_not_claim_official_checkpoint": True,
            "does_not_claim_real_robot": True,
            "does_not_claim_goal_complete": True,
        },
        "outputs": {
            "json": str(OUT / "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json"),
            "rows_csv": str(rows_csv),
            "rows_tsv": str(rows_tsv),
            "aggregate_csv": str(aggregate_csv),
            "report_dir": str(REPORT_OUT),
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "local_virtual_official_importer_export_task_conditioned_latent_guidance_multiseed_eval",
            "why_not_complete": (
                "This is a multi-seed local virtual official-importer-export full-bundle closed-loop guidance "
                "evaluation using local PPO/VAE/denoiser checkpoints and proxy costs. It is useful paper-facing "
                "evidence, but not official BeyondMimic Fig. 5/Fig. 6 reproduction, TensorRT deployment, or "
                "real-robot evidence."
            ),
        },
    }
    write_json(OUT / "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json", summary)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(all_rows)}, sort_keys=True))
    if not status_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
