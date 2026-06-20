#!/usr/bin/env python3
"""Run multi-seed evals for the scaled official-importer-export PPO checkpoint."""

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
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.py"
SOURCE_SINGLE_EVAL_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/"
    "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json"
)
OUT = ROOT / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_multiseed_eval"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_multiseed_eval"
LOG_DIR = ROOT / "logs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_multiseed_eval"
DEFAULT_SEEDS = [20260710, 20260711, 20260712]
DEFAULT_NUM_ENVS = 2048
DEFAULT_EVAL_STEPS = 299


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str], delimiter: str = ",") -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    tmp.replace(path)


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_scaled_importer_single_eval", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base scaled eval script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_float(value: Any) -> float:
    try:
        if value in {"", None}:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


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


def metric_mean(metrics: dict[str, Any], section: str, name: str | None = None) -> Any:
    value = metrics.get(section, {}) if name is None else metrics.get(section, {}).get(name, {})
    if isinstance(value, dict):
        if "mean" in value:
            return value["mean"]
        if "mean_over_steps" in value and isinstance(value["mean_over_steps"], dict):
            return value["mean_over_steps"].get("mean")
    return ""


def extract_row(seed: int, summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("run", {}).get("metrics", {})
    motion = metrics.get("motion_metrics", {})
    config = summary.get("config", {})
    return {
        "seed": seed,
        "status": summary.get("status", ""),
        "selected_physical_gpus": json.dumps(config.get("selected_physical_gpus", [])),
        "duration_seconds": summary.get("run", {}).get("duration_seconds", ""),
        "num_envs": metrics.get("num_envs", ""),
        "eval_steps": metrics.get("eval_steps", ""),
        "total_env_steps": metrics.get("total_env_steps", ""),
        "loaded_iteration": metrics.get("loaded_iteration", ""),
        "reward_mean": metric_mean(metrics, "reward"),
        "done_count_total": metrics.get("done_count_total", ""),
        "timeout_count_total": metrics.get("timeout_count_total", ""),
        "error_anchor_pos_mean": metric_mean(motion, "error_anchor_pos"),
        "error_body_pos_mean": metric_mean(motion, "error_body_pos"),
        "error_joint_pos_mean": metric_mean(motion, "error_joint_pos"),
        "error_anchor_lin_vel_mean": metric_mean(motion, "error_anchor_lin_vel"),
        "error_body_lin_vel_mean": metric_mean(motion, "error_body_lin_vel"),
        "error_joint_vel_mean": metric_mean(motion, "error_joint_vel"),
        "uses_official_importer_export_usd": metrics.get("uses_official_importer_export_usd", ""),
        "uses_resource_adjusted_usd": metrics.get("uses_resource_adjusted_usd", ""),
        "motion_count": metrics.get("motion_count", ""),
        "total_motion_frames": metrics.get("total_motion_frames", ""),
        "json": summary.get("outputs", {}).get("json", ""),
        "metrics_json": summary.get("outputs", {}).get("metrics_json", ""),
        "timeseries_csv": summary.get("outputs", {}).get("timeseries_csv", ""),
        "gpu_metrics_csv": summary.get("outputs", {}).get("gpu_metrics_csv", ""),
        "log": summary.get("outputs", {}).get("log", ""),
    }


def copy_seed_outputs(seed: int, summary: dict[str, Any]) -> dict[str, str]:
    seed_dir = OUT / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    copies: dict[str, str] = {}
    for key in ["json", "metrics_json", "timeseries_csv", "gpu_metrics_csv", "log"]:
        src = Path(summary.get("outputs", {}).get(key) or summary.get("run", {}).get(key, ""))
        if src.is_file():
            dst = seed_dir / f"seed_{seed}_{src.name}"
            shutil.copy2(src, dst)
            copies[key] = str(dst)
    return copies


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [
        int(item.strip())
        for item in os.environ.get(
            "BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_MULTI_SEEDS",
            ",".join(map(str, DEFAULT_SEEDS)),
        ).split(",")
        if item.strip()
    ]
    num_envs = int(os.environ.get("BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_MULTI_NUM_ENVS", str(DEFAULT_NUM_ENVS)))
    eval_steps = int(os.environ.get("BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_MULTI_EVAL_STEPS", str(DEFAULT_EVAL_STEPS)))

    base = load_base_module()
    source_single = load_json(SOURCE_SINGLE_EVAL_JSON)
    rows: list[dict[str, Any]] = []
    attempted: list[dict[str, Any]] = []
    seed_summaries: list[dict[str, Any]] = []

    for seed in seeds:
        base.OUT = OUT / f"seed_{seed}_scratch"
        base.LOG_DIR = LOG_DIR
        base.RUN_ROOT = RUN_ROOT / f"seed_{seed}"
        os.environ["BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_EVAL_SEED"] = str(seed)
        os.environ["BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_EVAL_NUM_ENVS"] = str(num_envs)
        os.environ["BM_PPO_EVAL_STEPS"] = str(eval_steps)
        os.environ["BM_PPO_EVAL_CANDIDATE_GPUS"] = "4,7"
        os.environ["BM_PPO_EVAL_VISIBLE_GPU_LIMIT"] = "2"
        started_at = datetime.now(timezone.utc).isoformat()
        base.main()
        final_json = base.OUT / "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json"
        summary = load_json(final_json)
        copies = copy_seed_outputs(seed, summary)
        summary["multiseed_copy_outputs"] = copies
        summary["multiseed_seed"] = seed
        summary["multiseed_started_at"] = started_at
        seed_summary_path = OUT / f"seed_{seed}" / (
            f"tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval_seed_{seed}.json"
        )
        write_json(seed_summary_path, summary)
        row = extract_row(seed, summary)
        row["json"] = str(seed_summary_path)
        rows.append(row)
        seed_summaries.append(summary)
        attempted.append(
            {
                "seed": seed,
                "status": summary.get("status", ""),
                "json": str(seed_summary_path),
                "started_at": started_at,
            }
        )

    fields = list(rows[0].keys()) if rows else []
    rows_csv = OUT / "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_multiseed_eval_rows.csv"
    rows_tsv = OUT / "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_multiseed_eval_rows.tsv"
    write_csv(rows_csv, rows, fields)
    write_csv(rows_tsv, rows, fields, delimiter="\t")

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
    aggregate = {name: summarize([safe_float(row.get(name)) for row in rows]) for name in metric_names}
    ok_status = "ok_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval_completed"
    ok_count = sum(row["status"] == ok_status for row in rows)
    total_env_steps = sum(safe_int(row.get("total_env_steps")) for row in rows)
    summary = {
        "status": "ok_official_importer_export_full_bundle_scaled_ppo_checkpoint_multiseed_eval_completed"
        if ok_count == len(seeds)
        else "partial_official_importer_export_full_bundle_scaled_ppo_checkpoint_multiseed_eval",
        "experiment_type": "tracking_official_importer_export_full_bundle_scaled_ppo_checkpoint_multiseed_eval",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Runs multiple full-step local virtual evaluations of the iteration-999 scaled PPO checkpoint trained on "
            "the official-importer-export G1 USDA and the 40-motion public official-csv-loop bundle. This improves "
            "stability evidence over a single eval, but remains local virtual tracking evidence."
        ),
        "config": {
            "seeds": seeds,
            "candidate_physical_gpus": [4, 7],
            "selected_physical_gpus": [4, 7],
            "num_envs": num_envs,
            "eval_steps": eval_steps,
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
            "all_eval_steps_299": all(safe_int(row.get("eval_steps")) == eval_steps for row in rows),
            "all_num_envs_match_config": all(safe_int(row.get("num_envs")) == num_envs for row in rows),
            "all_use_official_importer_export_usd": all(
                bool(row.get("uses_official_importer_export_usd")) for row in rows
            ),
            "no_rows_use_resource_adjusted_usd": all(
                row.get("uses_resource_adjusted_usd") is False for row in rows
            ),
            "all_motion_count_40": all(safe_int(row.get("motion_count")) == 40 for row in rows),
            "all_total_motion_frames_11960": all(safe_int(row.get("total_motion_frames")) == 11960 for row in rows),
            "total_env_steps_positive": total_env_steps > 0,
            "rows_csv_exists": rows_csv.is_file(),
            "rows_tsv_exists": rows_tsv.is_file(),
            "does_not_claim_paper_level_eval": True,
            "does_not_claim_official_beyondmimic_checkpoint": True,
            "does_not_claim_real_robot": True,
            "does_not_claim_goal_complete": True,
        },
        "outputs": {
            "json": str(OUT / "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_multiseed_eval.json"),
            "rows_csv": str(rows_csv),
            "rows_tsv": str(rows_tsv),
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "local_virtual_multiseed_scaled_tracking_eval",
            "why_not_complete": (
                "This is multi-seed local virtual checkpoint evaluation. It uses a local scaled PPO checkpoint and "
                "captured official-importer USDA, not an official BeyondMimic teacher checkpoint, not the paper's full "
                "teacher training/evaluation protocol, not DAgger, and not real robot."
            ),
        },
    }
    write_json(OUT / "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_multiseed_eval.json", summary)
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
