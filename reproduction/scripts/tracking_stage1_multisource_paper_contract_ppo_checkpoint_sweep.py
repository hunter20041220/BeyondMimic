#!/usr/bin/env python3
"""Sweep checkpoints from the GPUs 5/6 Stage-1 multi-source PPO run."""

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
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_stage1_multisource_paper_contract_ppo_checkpoint_eval.py"
TRAINING_RUN_JSON = (
    ROOT
    / "res/tracking/stage1_multisource_paper_contract_ppo_training_run/"
    "tracking_stage1_multisource_paper_contract_ppo_training_run.json"
)
DEFAULT_RUN_DIR = (
    ROOT
    / "res/runs/stage1_multisource_paper_contract_ppo_training/"
    "resource_adjusted_ppo_20260622_114146_seed20260851"
)
OUT = ROOT / "res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep"
RUN_ROOT = ROOT / "res/runs/stage1_multisource_paper_contract_ppo_checkpoint_sweep"
LOG_DIR = ROOT / "logs/tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep"
DEFAULT_NUM_ENVS = 256
DEFAULT_EVAL_STEPS = 299
DEFAULT_SEED = 20260853
OK_STATUS = "ok_stage1_multisource_paper_contract_ppo_checkpoint_eval_completed"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict[str, Any]], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["status"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_stage1_multisource_checkpoint_eval", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load eval wrapper: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_iteration(path: Path) -> int:
    try:
        return int(path.stem.split("_", maxsplit=1)[1])
    except Exception:
        return -1


def training_run_dir(training: dict[str, Any]) -> Path:
    override = os.environ.get("BM_STAGE1_MULTISOURCE_CHECKPOINT_SWEEP_RUN_DIR", "").strip()
    if override:
        return Path(override)
    run_dir = Path(training.get("outputs", {}).get("run_dir", ""))
    return run_dir if run_dir.is_dir() else DEFAULT_RUN_DIR


def discover_checkpoints(training: dict[str, Any]) -> list[Path]:
    run_dir = training_run_dir(training)
    checkpoints = sorted((run_dir / "rank_0").glob("model_*.pt"), key=checkpoint_iteration)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints under {run_dir / 'rank_0'}")
    selected = os.environ.get("BM_STAGE1_MULTISOURCE_CHECKPOINT_SWEEP_ITERATIONS", "").strip()
    if selected:
        wanted = {int(item.strip()) for item in selected.split(",") if item.strip()}
        checkpoints = [path for path in checkpoints if checkpoint_iteration(path) in wanted]
    else:
        stride = int(os.environ.get("BM_STAGE1_MULTISOURCE_CHECKPOINT_SWEEP_STRIDE", "2500"))
        if stride > 0:
            checkpoints = [
                path
                for path in checkpoints
                if checkpoint_iteration(path) == 0 or checkpoint_iteration(path) % stride == 0 or path == checkpoints[-1]
            ]
    if not checkpoints:
        raise FileNotFoundError("Checkpoint selection is empty")
    return checkpoints


def make_training_shim(training: dict[str, Any], checkpoint: Path, iteration: int) -> Path:
    shim_root = OUT / "training_shims" / f"iter_{iteration:05d}"
    rank0 = shim_root / "run_dir/rank_0"
    rank0.mkdir(parents=True, exist_ok=True)
    link = rank0 / checkpoint.name
    if link.exists() or link.is_symlink():
        link.unlink()
    try:
        link.symlink_to(checkpoint)
    except OSError:
        shutil.copy2(checkpoint, link)
    shim = json.loads(json.dumps(training))
    shim["status"] = "ok_stage1_multisource_paper_contract_ppo_training_completed"
    shim.setdefault("outputs", {})
    shim["outputs"]["run_dir"] = str(shim_root / "run_dir")
    shim.setdefault("inputs", {})
    shim["inputs"]["sweep_source_training_run_json"] = str(TRAINING_RUN_JSON)
    shim["inputs"]["sweep_selected_checkpoint"] = str(checkpoint)
    shim["inputs"]["sweep_selected_iteration"] = iteration
    path = shim_root / f"training_summary_iter_{iteration:05d}.json"
    write_json(path, shim)
    return path


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def metric_mean(metrics: dict[str, Any], section: str) -> Any:
    value = metrics.get(section, {})
    if isinstance(value, dict):
        if "mean" in value:
            return value["mean"]
        if "mean_over_steps" in value and isinstance(value["mean_over_steps"], dict):
            return value["mean_over_steps"].get("mean")
    return ""


def extract_row(iteration: int, checkpoint: Path, summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("run", {}).get("metrics", {})
    motion = metrics.get("motion_metrics", {})
    total_env_steps = safe_float(metrics.get("total_env_steps"), 0.0)
    done = safe_float(metrics.get("done_count_total"), 0.0)
    timeout = safe_float(metrics.get("timeout_count_total"), 0.0)
    return {
        "iteration": iteration,
        "checkpoint": str(checkpoint),
        "status": summary.get("status", ""),
        "num_envs": metrics.get("num_envs", ""),
        "eval_steps": metrics.get("eval_steps", ""),
        "total_env_steps": metrics.get("total_env_steps", ""),
        "loaded_iteration": metrics.get("loaded_iteration", ""),
        "reward_mean": metric_mean(metrics, "reward"),
        "done_count_total": metrics.get("done_count_total", ""),
        "timeout_count_total": metrics.get("timeout_count_total", ""),
        "local_non_timeout_done_rate": (done - timeout) / total_env_steps if total_env_steps > 0 else math.nan,
        "error_anchor_pos_mean": metric_mean(motion, "error_anchor_pos"),
        "error_body_pos_mean": metric_mean(motion, "error_body_pos"),
        "error_joint_pos_mean": metric_mean(motion, "error_joint_pos"),
        "error_joint_vel_mean": metric_mean(motion, "error_joint_vel"),
        "motion_count": metrics.get("motion_count", ""),
        "total_motion_frames": metrics.get("total_motion_frames", ""),
        "summary_json": summary.get("outputs", {}).get("json", ""),
        "metrics_json": summary.get("outputs", {}).get("metrics_json", ""),
        "timeseries_csv": summary.get("outputs", {}).get("timeseries_csv", ""),
        "log": summary.get("outputs", {}).get("log", ""),
    }


def rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        safe_float(row.get("local_non_timeout_done_rate"), math.inf),
        safe_float(row.get("error_body_pos_mean"), math.inf),
        safe_float(row.get("error_joint_pos_mean"), math.inf),
        -safe_float(row.get("reward_mean"), -math.inf),
    )


def summarize(values: list[float]) -> dict[str, Any]:
    finite = [value for value in values if math.isfinite(value)]
    return {"count": len(finite), "mean": sum(finite) / len(finite), "min": min(finite), "max": max(finite)} if finite else {
        "count": 0,
        "mean": None,
        "min": None,
        "max": None,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    training = load_json(TRAINING_RUN_JSON)
    if not training:
        raise FileNotFoundError(f"Training summary is not ready: {TRAINING_RUN_JSON}")
    checkpoints = discover_checkpoints(training)
    base = load_base_module()
    rows: list[dict[str, Any]] = []
    attempted: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        iteration = checkpoint_iteration(checkpoint)
        shim = make_training_shim(training, checkpoint, iteration)
        scratch = OUT / f"iter_{iteration:05d}_scratch"
        base.OUT = scratch
        base.LOG_DIR = LOG_DIR
        base.RUN_ROOT = RUN_ROOT / f"iter_{iteration:05d}"
        base.TRAINING_RUN_JSON = shim
        os.environ["BM_STAGE1_MULTISOURCE_PPO_EVAL_SEED"] = os.environ.get(
            "BM_STAGE1_MULTISOURCE_CHECKPOINT_SWEEP_SEED", str(DEFAULT_SEED)
        )
        os.environ["BM_STAGE1_MULTISOURCE_PPO_EVAL_NUM_ENVS"] = os.environ.get(
            "BM_STAGE1_MULTISOURCE_CHECKPOINT_SWEEP_NUM_ENVS", str(DEFAULT_NUM_ENVS)
        )
        os.environ["BM_STAGE1_MULTISOURCE_PPO_EVAL_STEPS"] = os.environ.get(
            "BM_STAGE1_MULTISOURCE_CHECKPOINT_SWEEP_EVAL_STEPS", str(DEFAULT_EVAL_STEPS)
        )
        os.environ["BM_STAGE1_MULTISOURCE_PPO_EVAL_TARGET_GPUS"] = os.environ.get(
            "BM_STAGE1_MULTISOURCE_CHECKPOINT_SWEEP_TARGET_GPUS", "5,6"
        )
        started_at = datetime.now(timezone.utc).isoformat()
        base.main()
        eval_summary = load_json(scratch / "tracking_stage1_multisource_paper_contract_ppo_checkpoint_eval.json")
        eval_summary["checkpoint_sweep_iteration"] = iteration
        eval_summary["checkpoint_sweep_checkpoint"] = str(checkpoint)
        eval_summary_path = OUT / f"iter_{iteration:05d}" / (
            f"tracking_stage1_multisource_paper_contract_ppo_checkpoint_eval_iter_{iteration:05d}.json"
        )
        write_json(eval_summary_path, eval_summary)
        row = extract_row(iteration, checkpoint, eval_summary)
        row["summary_json"] = str(eval_summary_path)
        rows.append(row)
        attempted.append({"iteration": iteration, "status": eval_summary.get("status", ""), "json": str(eval_summary_path), "started_at": started_at})

    rows_csv = OUT / "tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep_rows.csv"
    rows_tsv = OUT / "tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep_rows.tsv"
    write_rows(rows_csv, rows)
    write_rows(rows_tsv, rows, delimiter="\t")
    ok_rows = [row for row in rows if row.get("status") == OK_STATUS]
    ranked_rows = sorted(ok_rows, key=rank_key)
    best = ranked_rows[0] if ranked_rows else {}
    aggregate = {
        name: summarize([safe_float(row.get(name)) for row in rows])
        for name in ["reward_mean", "local_non_timeout_done_rate", "error_body_pos_mean", "error_joint_pos_mean"]
    }
    summary = {
        "status": "ok_stage1_multisource_paper_contract_ppo_checkpoint_sweep_completed"
        if ok_rows and len(ok_rows) == len(rows)
        else "partial_stage1_multisource_paper_contract_ppo_checkpoint_sweep",
        "experiment_type": "tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Screens checkpoints from the 5/6 multi-source Stage-1 PPO run before downstream teacher rollout.",
        "config": {
            "checkpoint_count": len(checkpoints),
            "iterations": [checkpoint_iteration(path) for path in checkpoints],
            "target_physical_gpus": os.environ.get("BM_STAGE1_MULTISOURCE_CHECKPOINT_SWEEP_TARGET_GPUS", "5,6"),
            "num_envs": int(os.environ.get("BM_STAGE1_MULTISOURCE_CHECKPOINT_SWEEP_NUM_ENVS", str(DEFAULT_NUM_ENVS))),
            "eval_steps": int(os.environ.get("BM_STAGE1_MULTISOURCE_CHECKPOINT_SWEEP_EVAL_STEPS", str(DEFAULT_EVAL_STEPS))),
            "seed": int(os.environ.get("BM_STAGE1_MULTISOURCE_CHECKPOINT_SWEEP_SEED", str(DEFAULT_SEED))),
            "source_training_run_json": str(TRAINING_RUN_JSON),
        },
        "attempted": attempted,
        "rows": rows,
        "aggregate": aggregate,
        "best_checkpoint": best,
        "metrics": {
            "checkpoint_count": len(checkpoints),
            "ok_checkpoint_count": len(ok_rows),
            "best_iteration": int(best.get("iteration", -1)) if best else None,
            "best_checkpoint": best.get("checkpoint", ""),
            "best_reward_mean": safe_float(best.get("reward_mean")) if best else None,
            "best_local_non_timeout_done_rate": safe_float(best.get("local_non_timeout_done_rate")) if best else None,
            "best_error_body_pos_mean": safe_float(best.get("error_body_pos_mean")) if best else None,
            "best_error_joint_pos_mean": safe_float(best.get("error_joint_pos_mean")) if best else None,
        },
        "checks": {
            "training_summary_exists": TRAINING_RUN_JSON.is_file(),
            "training_status_completed": training.get("status") == "ok_stage1_multisource_paper_contract_ppo_training_completed",
            "best_checkpoint_recorded": bool(best),
            "does_not_claim_official_beyondmimic_checkpoint": True,
            "does_not_claim_paper_level_eval": True,
            "does_not_claim_real_robot": True,
        },
        "outputs": {
            "json": str(OUT / "tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.json"),
            "best_teacher_json": str(OUT / "stage1_multisource_best_teacher.json"),
            "rows_csv": str(rows_csv),
            "rows_tsv": str(rows_tsv),
        },
        "interpretation": {
            "goal_complete": False,
            "claim_level": "local_multisource_stage1_checkpoint_screening_not_paper_level",
            "why_not_complete": "This selects a local public-plus-available-data teacher candidate, not the official BeyondMimic teacher.",
        },
    }
    write_json(OUT / "tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.json", summary)
    write_json(
        OUT / "stage1_multisource_best_teacher.json",
        {
            "status": "ok" if best else "missing",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_sweep_json": summary["outputs"]["json"],
            "best_checkpoint": best,
            "claim_level": "local_best_candidate_teacher_from_stage1_multisource_sweep",
            "official_beyondmimic_checkpoint": False,
            "paper_level_tracking_eval": False,
        },
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "checkpoint_count": len(checkpoints),
                "ok_checkpoint_count": len(ok_rows),
                "best_iteration": summary["metrics"]["best_iteration"],
                "best_checkpoint": summary["metrics"]["best_checkpoint"],
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if not ok_rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
