#!/usr/bin/env python3
"""Run full-step multi-seed evals for the official-csv-loop PPO checkpoint."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_g1_official_csv_loop_ppo_checkpoint_eval.py"
SOURCE_SINGLE_EVAL_JSON = (
    ROOT
    / "res/tracking/g1_official_csv_loop_ppo_checkpoint_eval/"
    "tracking_g1_official_csv_loop_ppo_checkpoint_eval.json"
)
OUT = ROOT / "res/tracking/g1_official_csv_loop_ppo_checkpoint_multiseed_eval"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval"
LOG_DIR = ROOT / "logs/tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval"
REPORT_OUT = ROOT / "res/report_assets/official_csv_loop_ppo_checkpoint_multiseed_eval"
DEFAULT_SEEDS = [20260640, 20260641, 20260642]
DEFAULT_GPU_ASSIGNMENT = [4, 7, 4]


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    tmp.replace(path)


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_official_csv_loop_single_eval", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base official eval script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_mean(metrics: dict[str, Any], section: str, name: str | None = None) -> float:
    if name is None:
        value = metrics.get(section, {})
    else:
        value = metrics.get(section, {}).get(name, {})
    if isinstance(value, dict):
        if "mean" in value:
            return float(value["mean"])
        if "mean_over_steps" in value and isinstance(value["mean_over_steps"], dict):
            return float(value["mean_over_steps"]["mean"])
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


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any) -> float:
    try:
        if value in {"", None}:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def extract_row(seed: int, gpu: int, summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("run", {}).get("metrics", {})
    motion = metrics.get("motion_metrics", {})
    config = summary.get("config", {})
    return {
        "seed": seed,
        "assigned_gpu": gpu,
        "status": summary.get("status", ""),
        "selected_physical_gpus": json.dumps(config.get("selected_physical_gpus", [])),
        "duration_seconds": summary.get("run", {}).get("duration_seconds", ""),
        "num_envs": metrics.get("num_envs", ""),
        "eval_steps": metrics.get("eval_steps", ""),
        "total_env_steps": metrics.get("total_env_steps", ""),
        "reward_mean": metrics.get("reward", {}).get("mean_over_steps", {}).get("mean", ""),
        "done_count_total": metrics.get("done_count_total", ""),
        "timeout_count_total": metrics.get("timeout_count_total", ""),
        "error_anchor_pos_mean": motion.get("error_anchor_pos", {}).get("mean", ""),
        "error_body_pos_mean": motion.get("error_body_pos", {}).get("mean", ""),
        "error_joint_pos_mean": motion.get("error_joint_pos", {}).get("mean", ""),
        "error_anchor_lin_vel_mean": motion.get("error_anchor_lin_vel", {}).get("mean", ""),
        "error_body_lin_vel_mean": motion.get("error_body_lin_vel", {}).get("mean", ""),
        "error_joint_vel_mean": motion.get("error_joint_vel", {}).get("mean", ""),
        "run_dir": summary.get("outputs", {}).get("run_dir", ""),
        "json": summary.get("outputs", {}).get("json", ""),
        "timeseries_csv": summary.get("outputs", {}).get("timeseries_csv", ""),
        "gpu_metrics_csv": summary.get("outputs", {}).get("gpu_metrics_csv", ""),
    }


def copy_seed_outputs(seed: int, source_summary: dict[str, Any]) -> dict[str, str]:
    seed_dir = OUT / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    copies: dict[str, str] = {}
    for key in ["json", "metrics_json", "timeseries_csv", "gpu_metrics_csv", "log"]:
        src = Path(source_summary.get("outputs", {}).get(key) or source_summary.get("run", {}).get(key, ""))
        if src.is_file():
            dst = seed_dir / f"seed_{seed}_{src.name}"
            shutil.copy2(src, dst)
            copies[key] = str(dst)
    return copies


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.mkdir(parents=True, exist_ok=True)

    seeds = [
        int(item.strip())
        for item in os.environ.get("BM_OFFICIAL_CSV_LOOP_PPO_MULTI_SEEDS", ",".join(map(str, DEFAULT_SEEDS))).split(",")
        if item.strip()
    ]
    gpu_assignment = [
        int(item.strip())
        for item in os.environ.get(
            "BM_OFFICIAL_CSV_LOOP_PPO_MULTI_GPUS",
            ",".join(map(str, DEFAULT_GPU_ASSIGNMENT[: len(seeds)])),
        ).split(",")
        if item.strip()
    ]
    if len(gpu_assignment) < len(seeds):
        gpu_assignment.extend(DEFAULT_GPU_ASSIGNMENT[: len(seeds) - len(gpu_assignment)])

    base = load_base_module()
    source_single = load_json(SOURCE_SINGLE_EVAL_JSON)
    rows: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []
    attempted = []

    for index, seed in enumerate(seeds):
        gpu = gpu_assignment[index]
        base.OUT = OUT / f"seed_{seed}_scratch"
        base.LOG_DIR = LOG_DIR
        base.RUN_ROOT = RUN_ROOT / f"seed_{seed}"
        base.TARGET_GPUS = [gpu]
        os.environ["BM_OFFICIAL_CSV_LOOP_PPO_EVAL_SEED"] = str(seed)
        os.environ["BM_PPO_EVAL_SEED"] = str(seed)
        os.environ["BM_PPO_EVAL_CANDIDATE_GPUS"] = str(gpu)
        os.environ["BM_PPO_EVAL_VISIBLE_GPU_LIMIT"] = "1"
        os.environ.setdefault("BM_PPO_EVAL_NUM_ENVS", "512")
        os.environ.setdefault("BM_PPO_EVAL_STEPS", "299")
        started_at = datetime.now(timezone.utc).isoformat()
        base.main()
        final_json = base.OUT / "tracking_g1_official_csv_loop_ppo_checkpoint_eval.json"
        summary = load_json(final_json)
        copies = copy_seed_outputs(seed, summary)
        summary["multiseed_copy_outputs"] = copies
        summary["multiseed_seed"] = seed
        summary["multiseed_assigned_gpu"] = gpu
        summary["multiseed_started_at"] = started_at
        seed_summary_path = OUT / f"seed_{seed}" / f"tracking_g1_official_csv_loop_ppo_checkpoint_eval_seed_{seed}.json"
        write_json_atomic(seed_summary_path, summary)
        attempted.append(
            {
                "seed": seed,
                "assigned_gpu": gpu,
                "status": summary.get("status", ""),
                "json": str(seed_summary_path),
            }
        )
        row = extract_row(seed, gpu, summary)
        row["json"] = str(seed_summary_path)
        rows.append(row)
        seed_summaries.append(summary)

    metric_names = [
        "reward_mean",
        "done_count_total",
        "timeout_count_total",
        "error_anchor_pos_mean",
        "error_body_pos_mean",
        "error_joint_pos_mean",
        "error_anchor_lin_vel_mean",
        "error_body_lin_vel_mean",
        "error_joint_vel_mean",
    ]
    aggregate = {
        name: summarize([safe_float(row.get(name)) for row in rows])
        for name in metric_names
    }
    ok_count = sum(row["status"] == "ok_official_csv_loop_ppo_checkpoint_eval_completed" for row in rows)
    total_env_steps = sum(safe_int(row.get("total_env_steps")) for row in rows)

    rows_csv = OUT / "tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval_rows.csv"
    rows_tsv = OUT / "tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval_rows.tsv"
    fields = list(rows[0].keys()) if rows else []
    write_csv(rows_csv, rows, fields)
    write_csv(rows_tsv, rows, fields)

    summary = {
        "status": "ok_official_csv_loop_ppo_checkpoint_multiseed_eval_completed"
        if ok_count == len(seeds)
        else "partial_official_csv_loop_ppo_checkpoint_multiseed_eval",
        "experiment_type": "tracking_official_csv_loop_ppo_checkpoint_multiseed_eval",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Runs multiple full 299-step local virtual evaluations of the official-csv-loop PPO checkpoint in "
            "Tracking-Flat-G1-v0. This improves stability evidence over a single checkpoint eval, but still uses "
            "the enriched-USD runtime patch and a 300-iteration local PPO checkpoint rather than the paper teacher."
        ),
        "config": {
            "seeds": seeds,
            "gpu_assignment": gpu_assignment[: len(seeds)],
            "num_envs": int(os.environ.get("BM_PPO_EVAL_NUM_ENVS", "512")),
            "eval_steps": int(os.environ.get("BM_PPO_EVAL_STEPS", "299")),
            "source_single_eval_json": str(SOURCE_SINGLE_EVAL_JSON),
            "source_single_eval_status": source_single.get("status", ""),
        },
        "attempted": attempted,
        "rows": rows,
        "aggregate": aggregate,
        "metrics": {
            "seed_count": len(seeds),
            "ok_seed_count": ok_count,
            "total_env_steps": total_env_steps,
        },
        "checks": {
            "all_seeds_completed": ok_count == len(seeds),
            "all_eval_steps_299": all(safe_int(row.get("eval_steps")) == 299 for row in rows),
            "all_num_envs_512": all(safe_int(row.get("num_envs")) == 512 for row in rows),
            "total_env_steps_positive": total_env_steps > 0,
            "rows_csv_exists": rows_csv.is_file(),
            "rows_tsv_exists": rows_tsv.is_file(),
            "does_not_claim_paper_level_eval": True,
            "does_not_claim_unpatched_official_asset": True,
            "does_not_claim_real_robot": True,
            "does_not_claim_goal_complete": True,
        },
        "outputs": {
            "json": str(OUT / "tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval.json"),
            "rows_csv": str(rows_csv),
            "rows_tsv": str(rows_tsv),
            "report_dir": str(REPORT_OUT),
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "local_virtual_multiseed_tracking_eval",
            "why_not_complete": (
                "This is multi-seed local virtual checkpoint evaluation. It is not the paper-scale official "
                "BeyondMimic tracking teacher, not unpatched official G1 replay/training, not DAgger, and not real robot."
            ),
        },
    }
    write_json_atomic(OUT / "tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval.json", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "ok_seed_count": ok_count,
                "seed_count": len(seeds),
                "total_env_steps": total_env_steps,
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if ok_count != len(seeds):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
