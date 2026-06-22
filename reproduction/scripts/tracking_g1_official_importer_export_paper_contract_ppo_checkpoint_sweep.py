#!/usr/bin/env python3
"""Sweep paper-contract PPO checkpoints after the 4/7 LAFAN1 teacher run.

This is a post-training selector for the long paper-contract PPO run.  It
evaluates saved ``model_*.pt`` files with the same official-importer-export
G1 USD and robot-order FK-repaired 40-motion LAFAN1 bundle used by the run.

Claim boundary: the selected checkpoint is the best local candidate under this
screening protocol.  It is not an official BeyondMimic teacher checkpoint and
is not a paper-level tracking result.
"""

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
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_eval.py"
TRAINING_RUN_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_paper_contract_ppo_training_run/"
    "tracking_g1_official_importer_export_paper_contract_ppo_training_run.json"
)
TRAINING_LOG = (
    ROOT / "logs/tracking_g1_official_importer_export_paper_contract_ppo_training_run/"
    "tracking_g1_resource_adjusted_ppo_training_run.log"
)
DEFAULT_RUN_DIR = (
    ROOT
    / "res/runs/tracking_g1_official_importer_export_paper_contract_ppo_training/"
    "resource_adjusted_ppo_20260622_084243_seed20260801"
)
OUT = ROOT / "res/tracking/g1_official_importer_export_paper_contract_ppo_checkpoint_sweep"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_sweep"
LOG_DIR = ROOT / "logs/tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_sweep"
DEFAULT_NUM_ENVS = 256
DEFAULT_EVAL_STEPS = 299
DEFAULT_SEED = 20260803
OK_STATUS = "ok_official_importer_export_paper_contract_ppo_checkpoint_eval_completed"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_rows(path: Path, rows: list[dict[str, Any]], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["status"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    tmp.replace(path)


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_paper_contract_checkpoint_eval", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base eval wrapper: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_iteration(path: Path) -> int:
    try:
        return int(path.stem.split("_", maxsplit=1)[1])
    except Exception:
        return -1


def training_run_dir(training: dict[str, Any]) -> Path:
    override = os.environ.get("BM_PAPER_CONTRACT_CHECKPOINT_SWEEP_RUN_DIR", "").strip()
    if override:
        return Path(override)
    run_dir = Path(training.get("outputs", {}).get("run_dir", ""))
    return run_dir if run_dir.is_dir() else DEFAULT_RUN_DIR


def discover_checkpoints(training: dict[str, Any]) -> list[Path]:
    run_dir = training_run_dir(training)
    checkpoints = sorted((run_dir / "rank_0").glob("model_*.pt"), key=checkpoint_iteration)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints under {run_dir / 'rank_0'}")
    selected = os.environ.get("BM_PAPER_CONTRACT_CHECKPOINT_SWEEP_ITERATIONS", "").strip()
    if selected:
        wanted = {int(item.strip()) for item in selected.split(",") if item.strip()}
        checkpoints = [path for path in checkpoints if checkpoint_iteration(path) in wanted]
    stride = int(os.environ.get("BM_PAPER_CONTRACT_CHECKPOINT_SWEEP_STRIDE", "0"))
    if stride > 0 and not selected:
        checkpoints = [
            path
            for path in checkpoints
            if checkpoint_iteration(path) == 0
            or checkpoint_iteration(path) % stride == 0
            or path == checkpoints[-1]
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
    shim["status"] = "ok_official_importer_export_paper_contract_ppo_training_completed"
    shim.setdefault("outputs", {})
    shim["outputs"]["run_dir"] = str(shim_root / "run_dir")
    shim.setdefault("inputs", {})
    shim["inputs"]["sweep_source_training_run_json"] = str(TRAINING_RUN_JSON)
    shim["inputs"]["sweep_source_training_log"] = str(TRAINING_LOG)
    shim["inputs"]["sweep_selected_checkpoint"] = str(checkpoint)
    shim["inputs"]["sweep_selected_iteration"] = iteration
    shim.setdefault("interpretation", {})
    shim["interpretation"]["checkpoint_sweep_shim"] = (
        "This shim redirects the paper-contract checkpoint-eval harness to one saved checkpoint. "
        "The authoritative training audit remains the paper-contract training JSON/log."
    )
    shim_path = shim_root / f"training_summary_iter_{iteration:05d}.json"
    write_json(shim_path, shim)
    return shim_path


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value in {"", None}:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in {"", None}:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def metric_mean(metrics: dict[str, Any], section: str, name: str | None = None) -> Any:
    value = metrics.get(section, {}) if name is None else metrics.get(section, {}).get(name, {})
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
    non_timeout_done_rate = (done - timeout) / total_env_steps if total_env_steps > 0 else math.nan
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
        "local_non_timeout_done_rate": non_timeout_done_rate,
        "error_anchor_pos_mean": metric_mean(motion, "error_anchor_pos"),
        "error_anchor_rot_mean": metric_mean(motion, "error_anchor_rot"),
        "error_body_pos_mean": metric_mean(motion, "error_body_pos"),
        "error_body_rot_mean": metric_mean(motion, "error_body_rot"),
        "error_joint_pos_mean": metric_mean(motion, "error_joint_pos"),
        "error_joint_vel_mean": metric_mean(motion, "error_joint_vel"),
        "error_body_lin_vel_mean": metric_mean(motion, "error_body_lin_vel"),
        "error_body_ang_vel_mean": metric_mean(motion, "error_body_ang_vel"),
        "uses_official_importer_export_usd": metrics.get("uses_official_importer_export_usd", ""),
        "motion_count": metrics.get("motion_count", ""),
        "total_motion_frames": metrics.get("total_motion_frames", ""),
        "summary_json": summary.get("outputs", {}).get("json", ""),
        "metrics_json": summary.get("outputs", {}).get("metrics_json", ""),
        "timeseries_csv": summary.get("outputs", {}).get("timeseries_csv", ""),
        "log": summary.get("outputs", {}).get("log", ""),
    }


def rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        safe_float(row.get("local_non_timeout_done_rate"), math.inf),
        safe_float(row.get("error_body_pos_mean"), math.inf),
        safe_float(row.get("error_joint_pos_mean"), math.inf),
        safe_float(row.get("error_anchor_pos_mean"), math.inf),
        -safe_float(row.get("reward_mean"), -math.inf),
    )


def copy_outputs(iteration: int, summary: dict[str, Any]) -> dict[str, str]:
    out_dir = OUT / f"iter_{iteration:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for key in ["json", "metrics_json", "timeseries_csv", "gpu_metrics_csv", "log"]:
        src = Path(summary.get("outputs", {}).get(key, "") or summary.get("run", {}).get(key, ""))
        if src.is_file():
            dst = out_dir / f"iter_{iteration:05d}_{src.name}"
            shutil.copy2(src, dst)
            copied[key] = str(dst)
    return copied


def summarize(values: list[float]) -> dict[str, Any]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "min": min(finite),
        "max": max(finite),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    training = load_json(TRAINING_RUN_JSON)
    if not training:
        raise FileNotFoundError(f"Training summary is not ready yet: {TRAINING_RUN_JSON}")

    checkpoints = discover_checkpoints(training)
    num_envs = int(os.environ.get("BM_PAPER_CONTRACT_CHECKPOINT_SWEEP_NUM_ENVS", str(DEFAULT_NUM_ENVS)))
    eval_steps = int(os.environ.get("BM_PAPER_CONTRACT_CHECKPOINT_SWEEP_EVAL_STEPS", str(DEFAULT_EVAL_STEPS)))
    seed = int(os.environ.get("BM_PAPER_CONTRACT_CHECKPOINT_SWEEP_SEED", str(DEFAULT_SEED)))
    target_gpus = os.environ.get("BM_PAPER_CONTRACT_CHECKPOINT_SWEEP_TARGET_GPUS", "4,7")
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
        os.environ["BM_PAPER_CONTRACT_PPO_EVAL_SEED"] = str(seed)
        os.environ["BM_PAPER_CONTRACT_PPO_EVAL_NUM_ENVS"] = str(num_envs)
        os.environ["BM_PAPER_CONTRACT_PPO_EVAL_STEPS"] = str(eval_steps)
        os.environ["BM_PAPER_CONTRACT_PPO_EVAL_TARGET_GPUS"] = target_gpus
        started_at = datetime.now(timezone.utc).isoformat()
        base.main()
        final_json = scratch / "tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_eval.json"
        eval_summary = load_json(final_json)
        copied = copy_outputs(iteration, eval_summary)
        eval_summary["checkpoint_sweep_iteration"] = iteration
        eval_summary["checkpoint_sweep_checkpoint"] = str(checkpoint)
        eval_summary["checkpoint_sweep_copy_outputs"] = copied
        eval_summary_path = OUT / f"iter_{iteration:05d}" / (
            f"tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_eval_iter_{iteration:05d}.json"
        )
        write_json(eval_summary_path, eval_summary)
        row = extract_row(iteration, checkpoint, eval_summary)
        row["summary_json"] = str(eval_summary_path)
        rows.append(row)
        attempted.append({"iteration": iteration, "status": eval_summary.get("status", ""), "json": str(eval_summary_path), "started_at": started_at})

    rows_csv = OUT / "tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_sweep_rows.csv"
    rows_tsv = OUT / "tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_sweep_rows.tsv"
    write_rows(rows_csv, rows)
    write_rows(rows_tsv, rows, delimiter="\t")

    ok_rows = [row for row in rows if row.get("status") == OK_STATUS]
    ranked_rows = sorted(ok_rows, key=rank_key)
    best = ranked_rows[0] if ranked_rows else {}
    metric_names = [
        "reward_mean",
        "local_non_timeout_done_rate",
        "error_anchor_pos_mean",
        "error_body_pos_mean",
        "error_joint_pos_mean",
        "error_joint_vel_mean",
    ]
    aggregate = {name: summarize([safe_float(row.get(name)) for row in rows]) for name in metric_names}
    summary = {
        "status": "ok_official_importer_export_paper_contract_ppo_checkpoint_sweep_completed"
        if ok_rows and len(ok_rows) == len(rows)
        else "partial_official_importer_export_paper_contract_ppo_checkpoint_sweep",
        "experiment_type": "tracking_official_importer_export_paper_contract_ppo_checkpoint_sweep",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Screens saved checkpoints from the 4/7 LAFAN1 paper-contract PPO retraining run. "
            "Ranking prefers low non-timeout termination rate, low body/joint tracking error, and then high reward."
        ),
        "config": {
            "checkpoint_count": len(checkpoints),
            "iterations": [checkpoint_iteration(path) for path in checkpoints],
            "target_physical_gpus": target_gpus,
            "num_envs": num_envs,
            "eval_steps": eval_steps,
            "seed": seed,
            "source_training_run_json": str(TRAINING_RUN_JSON),
            "source_training_log": str(TRAINING_LOG),
        },
        "attempted": attempted,
        "rows": rows,
        "aggregate": aggregate,
        "best_checkpoint": best,
        "metrics": {
            "checkpoint_count": len(checkpoints),
            "ok_checkpoint_count": len(ok_rows),
            "best_iteration": safe_int(best.get("iteration"), -1) if best else None,
            "best_checkpoint": best.get("checkpoint", ""),
            "best_reward_mean": safe_float(best.get("reward_mean")) if best else None,
            "best_local_non_timeout_done_rate": safe_float(best.get("local_non_timeout_done_rate")) if best else None,
            "best_error_body_pos_mean": safe_float(best.get("error_body_pos_mean")) if best else None,
            "best_error_joint_pos_mean": safe_float(best.get("error_joint_pos_mean")) if best else None,
        },
        "checks": {
            "training_summary_exists": TRAINING_RUN_JSON.is_file(),
            "training_status_completed": training.get("status") == "ok_official_importer_export_paper_contract_ppo_training_completed",
            "all_checkpoints_completed": len(ok_rows) == len(rows),
            "best_checkpoint_recorded": bool(best),
            "does_not_claim_official_beyondmimic_checkpoint": True,
            "does_not_claim_paper_level_eval": True,
            "does_not_claim_real_robot": True,
            "does_not_claim_goal_complete": True,
        },
        "outputs": {
            "json": str(OUT / "tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_sweep.json"),
            "best_teacher_json": str(OUT / "paper_contract_best_teacher.json"),
            "rows_csv": str(rows_csv),
            "rows_tsv": str(rows_tsv),
        },
        "interpretation": {
            "goal_complete": False,
            "claim_level": "local_virtual_paper_contract_checkpoint_screening_not_paper_level",
            "why_not_complete": (
                "The sweep selects a local teacher candidate from public-resource retraining. It is not the official "
                "BeyondMimic teacher, not official DAgger data, not Fig. 5/Fig. 6, and not real-robot evidence."
            ),
        },
    }
    write_json(OUT / "tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_sweep.json", summary)
    write_json(
        OUT / "paper_contract_best_teacher.json",
        {
            "status": "ok" if best else "missing",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source_sweep_json": summary["outputs"]["json"],
            "best_checkpoint": best,
            "claim_level": "local_best_candidate_teacher_from_paper_contract_sweep",
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
