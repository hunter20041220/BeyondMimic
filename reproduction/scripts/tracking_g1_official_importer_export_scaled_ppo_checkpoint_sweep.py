#!/usr/bin/env python3
"""Screen saved scaled-PPO checkpoints on the official-importer-export path.

The previous eval only measured the final iteration-999 checkpoint. This sweep
reuses the already validated IsaacLab evaluation wrapper and redirects its
training-summary run_dir to one checkpoint at a time. The goal is to determine
whether an earlier checkpoint is a better local tracking teacher before spending
more GPU time on downstream VAE/diffusion stages or longer PPO training.
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
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.py"
TRAINING_RUN_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_training_run/"
    "tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run.json"
)
OUT = ROOT / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_sweep"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_sweep"
LOG_DIR = ROOT / "logs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_sweep"
REPORT_OUT = ROOT / "res/report_assets/official_importer_export_scaled_ppo_checkpoint_sweep"
DEFAULT_NUM_ENVS = 256
DEFAULT_EVAL_STEPS = 299
DEFAULT_SEED = 20260730


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
    spec = importlib.util.spec_from_file_location("bm_scaled_importer_checkpoint_eval", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base scaled eval script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_iteration(path: Path) -> int:
    try:
        return int(path.stem.split("_")[1])
    except Exception:
        return -1


def discover_checkpoints(training: dict[str, Any]) -> list[Path]:
    run_dir = Path(training.get("outputs", {}).get("run_dir", ""))
    candidates = sorted((run_dir / "rank_0").glob("model_*.pt"), key=checkpoint_iteration) if run_dir.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"No rank_0/model_*.pt checkpoints under {run_dir}")
    selected = os.environ.get("BM_SCALED_PPO_CHECKPOINT_SWEEP_ITERATIONS", "").strip()
    if not selected:
        return candidates
    wanted = {int(item.strip()) for item in selected.split(",") if item.strip()}
    return [path for path in candidates if checkpoint_iteration(path) in wanted]


def make_checkpoint_training_shim(training: dict[str, Any], checkpoint: Path, iteration: int) -> Path:
    shim_root = OUT / "training_shims" / f"iter_{iteration:04d}"
    rank0 = shim_root / "run_dir" / "rank_0"
    rank0.mkdir(parents=True, exist_ok=True)
    link = rank0 / checkpoint.name
    if link.exists() or link.is_symlink():
        link.unlink()
    try:
        link.symlink_to(checkpoint)
    except OSError:
        shutil.copy2(checkpoint, link)
    shim = json.loads(json.dumps(training))
    shim.setdefault("outputs", {})
    shim["outputs"]["run_dir"] = str(shim_root / "run_dir")
    shim.setdefault("inputs", {})
    shim["inputs"]["sweep_source_training_run_json"] = str(TRAINING_RUN_JSON)
    shim["inputs"]["sweep_selected_checkpoint"] = str(checkpoint)
    shim["inputs"]["sweep_selected_iteration"] = iteration
    shim.setdefault("interpretation", {})
    shim["interpretation"]["checkpoint_sweep_shim"] = (
        "This shim redirects the existing scaled checkpoint-eval harness to a single saved checkpoint. "
        "The authoritative training audit remains the original scaled PPO training JSON."
    )
    shim_path = shim_root / f"training_summary_iter_{iteration:04d}.json"
    write_json(shim_path, shim)
    return shim_path


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
    config = summary.get("config", {})
    return {
        "iteration": iteration,
        "checkpoint": str(checkpoint),
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
        "local_non_timeout_done_rate": (
            safe_float(metrics.get("done_count_total")) / safe_float(metrics.get("total_env_steps"))
            if safe_float(metrics.get("total_env_steps")) > 0
            else ""
        ),
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


def summarize(values: list[float]) -> dict[str, Any]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = sum(finite) / len(finite)
    var = sum((value - mean) ** 2 for value in finite) / len(finite)
    return {"count": len(finite), "mean": mean, "std": math.sqrt(var), "min": min(finite), "max": max(finite)}


def copy_checkpoint_outputs(iteration: int, summary: dict[str, Any]) -> dict[str, str]:
    iter_dir = OUT / f"iter_{iteration:04d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    copies: dict[str, str] = {}
    for key in ["json", "metrics_json", "timeseries_csv", "gpu_metrics_csv", "log"]:
        src = Path(summary.get("outputs", {}).get(key) or summary.get("run", {}).get(key, ""))
        if src.is_file():
            dst = iter_dir / f"iter_{iteration:04d}_{src.name}"
            shutil.copy2(src, dst)
            copies[key] = str(dst)
    return copies


def write_report_assets(rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, str]:
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(rows).sort_values("iteration")
    for col in [
        "reward_mean",
        "done_count_total",
        "local_non_timeout_done_rate",
        "error_anchor_pos_mean",
        "error_body_pos_mean",
        "error_joint_pos_mean",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(df["iteration"], df["reward_mean"], marker="o", color="#059669")
    axes[0].set_ylabel("Reward mean")
    axes[0].set_title("Scaled PPO checkpoint sweep on official-importer-export full bundle")
    axes[1].plot(df["iteration"], df["local_non_timeout_done_rate"], marker="o", color="#ea580c")
    axes[1].set_ylabel("Non-timeout done rate")
    axes[2].plot(df["iteration"], df["error_anchor_pos_mean"], marker="o", label="anchor pos", color="#2563eb")
    axes[2].plot(df["iteration"], df["error_body_pos_mean"], marker="o", label="body pos", color="#16a34a")
    axes[2].plot(df["iteration"], df["error_joint_pos_mean"], marker="o", label="joint pos", color="#dc2626")
    axes[2].set_xlabel("Checkpoint iteration")
    axes[2].set_ylabel("Mean tracking error")
    axes[2].legend(loc="upper right")
    fig.tight_layout()
    sweep_png = REPORT_OUT / "scaled_ppo_checkpoint_sweep_metrics.png"
    fig.savefig(sweep_png, dpi=180)
    plt.close(fig)

    ranked = df.sort_values(
        ["local_non_timeout_done_rate", "error_body_pos_mean", "reward_mean"],
        ascending=[True, True, False],
    )
    ranked_csv = REPORT_OUT / "scaled_ppo_checkpoint_sweep_ranked.csv"
    ranked.to_csv(ranked_csv, index=False)
    readme = REPORT_OUT / "README.md"
    best = summary["best_checkpoint"]
    readme.write_text(
        "\n".join(
            [
                "# Scaled PPO Checkpoint Sweep",
                "",
                "Local virtual screening over saved scaled PPO checkpoints on the official-importer-export G1 USDA",
                "and the full 40-motion public bundle.",
                "",
                f"Best local screening checkpoint: iteration `{best.get('iteration')}`.",
                f"Best reward mean: `{best.get('reward_mean')}`.",
                f"Best non-timeout done rate: `{best.get('local_non_timeout_done_rate')}`.",
                "",
                "Claim level: local virtual checkpoint screening only. This is not an official BeyondMimic teacher",
                "checkpoint, not paper success/fall/collision evaluation, and not a real-robot result.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"ranked_csv": str(ranked_csv), "metrics_png": str(sweep_png), "readme": str(readme)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    training = load_json(TRAINING_RUN_JSON)
    checkpoints = discover_checkpoints(training)
    if os.environ.get("BM_SCALED_PPO_CHECKPOINT_SWEEP_INCLUDE_FINAL_ONLY", "0") == "1":
        checkpoints = checkpoints[-1:]
    num_envs = int(os.environ.get("BM_SCALED_PPO_CHECKPOINT_SWEEP_NUM_ENVS", str(DEFAULT_NUM_ENVS)))
    eval_steps = int(os.environ.get("BM_SCALED_PPO_CHECKPOINT_SWEEP_EVAL_STEPS", str(DEFAULT_EVAL_STEPS)))
    seed = int(os.environ.get("BM_SCALED_PPO_CHECKPOINT_SWEEP_SEED", str(DEFAULT_SEED)))

    base = load_base_module()
    rows: list[dict[str, Any]] = []
    attempted: list[dict[str, Any]] = []
    ok_status = "ok_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval_completed"

    for checkpoint in checkpoints:
        iteration = checkpoint_iteration(checkpoint)
        shim = make_checkpoint_training_shim(training, checkpoint, iteration)
        scratch = OUT / f"iter_{iteration:04d}_scratch"
        base.OUT = scratch
        base.LOG_DIR = LOG_DIR
        base.RUN_ROOT = RUN_ROOT / f"iter_{iteration:04d}"
        base.TRAINING_RUN_JSON = shim
        os.environ["BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_EVAL_SEED"] = str(seed)
        os.environ["BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_EVAL_NUM_ENVS"] = str(num_envs)
        os.environ["BM_PPO_EVAL_STEPS"] = str(eval_steps)
        os.environ["BM_PPO_EVAL_CANDIDATE_GPUS"] = "4,7"
        os.environ["BM_PPO_EVAL_VISIBLE_GPU_LIMIT"] = "2"
        started_at = datetime.now(timezone.utc).isoformat()
        base.main()
        final_json = scratch / "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json"
        eval_summary = load_json(final_json)
        copies = copy_checkpoint_outputs(iteration, eval_summary)
        eval_summary["checkpoint_sweep_iteration"] = iteration
        eval_summary["checkpoint_sweep_checkpoint"] = str(checkpoint)
        eval_summary["checkpoint_sweep_copy_outputs"] = copies
        eval_summary_path = OUT / f"iter_{iteration:04d}" / (
            f"tracking_g1_official_importer_export_scaled_ppo_checkpoint_eval_iter_{iteration:04d}.json"
        )
        write_json(eval_summary_path, eval_summary)
        row = extract_row(iteration, checkpoint, eval_summary)
        row["json"] = str(eval_summary_path)
        rows.append(row)
        attempted.append({"iteration": iteration, "status": eval_summary.get("status", ""), "json": str(eval_summary_path), "started_at": started_at})

    fields = list(rows[0].keys()) if rows else []
    rows_csv = OUT / "tracking_g1_official_importer_export_scaled_ppo_checkpoint_sweep_rows.csv"
    rows_tsv = OUT / "tracking_g1_official_importer_export_scaled_ppo_checkpoint_sweep_rows.tsv"
    write_csv(rows_csv, rows, fields)
    write_csv(rows_tsv, rows, fields, delimiter="\t")

    ok_rows = [row for row in rows if row.get("status") == ok_status]
    ranked_rows = sorted(
        ok_rows,
        key=lambda row: (
            safe_float(row.get("local_non_timeout_done_rate")),
            safe_float(row.get("error_body_pos_mean")),
            -safe_float(row.get("reward_mean")),
        ),
    )
    best = ranked_rows[0] if ranked_rows else {}
    metric_names = [
        "reward_mean",
        "done_count_total",
        "local_non_timeout_done_rate",
        "error_anchor_pos_mean",
        "error_body_pos_mean",
        "error_joint_pos_mean",
    ]
    aggregate = {name: summarize([safe_float(row.get(name)) for row in rows]) for name in metric_names}
    total_env_steps = sum(safe_int(row.get("total_env_steps")) for row in rows)
    summary = {
        "status": "ok_official_importer_export_scaled_ppo_checkpoint_sweep_completed"
        if len(ok_rows) == len(rows)
        else "partial_official_importer_export_scaled_ppo_checkpoint_sweep",
        "experiment_type": "tracking_official_importer_export_scaled_ppo_checkpoint_sweep",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Screens saved iteration checkpoints from the scaled PPO training run on the official-importer-export "
            "G1 USDA and full 40-motion public bundle. This identifies the best local checkpoint candidate before "
            "longer PPO training or downstream teacher-data regeneration."
        ),
        "config": {
            "checkpoint_count": len(checkpoints),
            "iterations": [checkpoint_iteration(path) for path in checkpoints],
            "candidate_physical_gpus": [4, 7],
            "selected_physical_gpus": [4, 7],
            "num_envs": num_envs,
            "eval_steps": eval_steps,
            "seed": seed,
            "source_training_run_json": str(TRAINING_RUN_JSON),
        },
        "attempted": attempted,
        "rows": rows,
        "aggregate": aggregate,
        "best_checkpoint": best,
        "metrics": {
            "checkpoint_count": len(checkpoints),
            "ok_checkpoint_count": len(ok_rows),
            "total_env_steps": total_env_steps,
            "best_iteration": safe_int(best.get("iteration"), -1) if best else None,
            "best_reward_mean": safe_float(best.get("reward_mean")) if best else None,
            "best_local_non_timeout_done_rate": safe_float(best.get("local_non_timeout_done_rate")) if best else None,
            "best_error_body_pos_mean": safe_float(best.get("error_body_pos_mean")) if best else None,
        },
        "checks": {
            "all_checkpoints_completed": len(ok_rows) == len(rows),
            "all_eval_steps_match_config": all(safe_int(row.get("eval_steps")) == eval_steps for row in rows),
            "all_num_envs_match_config": all(safe_int(row.get("num_envs")) == num_envs for row in rows),
            "all_use_official_importer_export_usd": all(bool(row.get("uses_official_importer_export_usd")) for row in rows),
            "no_rows_use_resource_adjusted_usd": all(row.get("uses_resource_adjusted_usd") is False for row in rows),
            "all_motion_count_40": all(safe_int(row.get("motion_count")) == 40 for row in rows),
            "all_total_motion_frames_11960": all(safe_int(row.get("total_motion_frames")) == 11960 for row in rows),
            "best_checkpoint_recorded": bool(best),
            "does_not_claim_paper_level_eval": True,
            "does_not_claim_official_beyondmimic_checkpoint": True,
            "does_not_claim_real_robot": True,
            "does_not_claim_goal_complete": True,
        },
        "outputs": {
            "json": str(OUT / "tracking_g1_official_importer_export_scaled_ppo_checkpoint_sweep.json"),
            "rows_csv": str(rows_csv),
            "rows_tsv": str(rows_tsv),
        },
        "interpretation": {
            "goal_complete": False,
            "claim_level": "local_virtual_scaled_ppo_checkpoint_sweep_not_paper_level",
            "why_not_complete": (
                "This is a local checkpoint-screening experiment over saved PPO checkpoints. It does not use official "
                "BeyondMimic checkpoints, does not evaluate the paper's success/fall/collision protocol, and does not "
                "produce real-robot evidence."
            ),
        },
    }
    assets = write_report_assets(rows, summary)
    summary["report_assets"] = assets
    write_json(OUT / "tracking_g1_official_importer_export_scaled_ppo_checkpoint_sweep.json", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "checkpoint_count": len(checkpoints),
                "ok_checkpoint_count": len(ok_rows),
                "total_env_steps": total_env_steps,
                "best_iteration": summary["metrics"]["best_iteration"],
                "best_reward_mean": summary["metrics"]["best_reward_mean"],
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if len(ok_rows) != len(rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
