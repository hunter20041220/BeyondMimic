#!/usr/bin/env python3
"""Full-size confirmation eval for the best checkpoint from the scaled PPO sweep."""

from __future__ import annotations

import csv
import importlib.util
import json
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
SWEEP_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_sweep/"
    "tracking_g1_official_importer_export_scaled_ppo_checkpoint_sweep.json"
)
FINAL_EVAL_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/"
    "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json"
)
OUT = ROOT / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_best_checkpoint_confirmation_eval"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_best_checkpoint_confirmation_eval"
LOG_DIR = ROOT / "logs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_best_checkpoint_confirmation_eval"
REPORT_OUT = ROOT / "res/report_assets/official_importer_export_scaled_ppo_best_checkpoint_confirmation_eval"
DEFAULT_NUM_ENVS = 2048
DEFAULT_EVAL_STEPS = 299
DEFAULT_SEED = 20260731


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


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


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_scaled_importer_best_checkpoint_eval", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base scaled eval script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_float(value: Any) -> float | None:
    try:
        if value in {"", None}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_mean(metrics: dict[str, Any], section: str, name: str | None = None) -> Any:
    value = metrics.get(section, {}) if name is None else metrics.get(section, {}).get(name, {})
    if isinstance(value, dict):
        if "mean" in value:
            return value["mean"]
        if "mean_over_steps" in value and isinstance(value["mean_over_steps"], dict):
            return value["mean_over_steps"].get("mean")
    return ""


def checkpoint_iteration(path: Path) -> int:
    try:
        return int(path.stem.split("_")[1])
    except Exception:
        return -1


def make_training_shim(training: dict[str, Any], checkpoint: Path, iteration: int) -> Path:
    shim_root = OUT / "training_shim"
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
    shim["inputs"]["best_checkpoint_confirmation_source_training_run_json"] = str(TRAINING_RUN_JSON)
    shim["inputs"]["best_checkpoint_confirmation_source_sweep_json"] = str(SWEEP_JSON)
    shim["inputs"]["best_checkpoint_confirmation_checkpoint"] = str(checkpoint)
    shim["inputs"]["best_checkpoint_confirmation_iteration"] = iteration
    shim.setdefault("interpretation", {})
    shim["interpretation"]["best_checkpoint_confirmation_shim"] = (
        "This shim redirects the existing scaled checkpoint-eval harness to the best checkpoint selected by the "
        "local checkpoint sweep. The authoritative training audit remains the original scaled PPO training JSON."
    )
    path = shim_root / "best_checkpoint_training_summary.json"
    write_json(path, shim)
    return path


def extract_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    metrics = summary.get("run", {}).get("metrics", {})
    motion = metrics.get("motion_metrics", {})
    total_steps = safe_float(metrics.get("total_env_steps")) or 0.0
    done_count = safe_float(metrics.get("done_count_total")) or 0.0
    timeout_count = safe_float(metrics.get("timeout_count_total")) or 0.0
    return {
        "status": summary.get("status", ""),
        "num_envs": metrics.get("num_envs", ""),
        "eval_steps": metrics.get("eval_steps", ""),
        "total_env_steps": metrics.get("total_env_steps", ""),
        "loaded_iteration": metrics.get("loaded_iteration", ""),
        "reward_mean": metric_mean(metrics, "reward"),
        "done_count_total": metrics.get("done_count_total", ""),
        "timeout_count_total": metrics.get("timeout_count_total", ""),
        "local_non_timeout_done_rate": done_count / total_steps if total_steps else "",
        "local_timeout_rate": timeout_count / total_steps if total_steps else "",
        "error_anchor_pos_mean": metric_mean(motion, "error_anchor_pos"),
        "error_body_pos_mean": metric_mean(motion, "error_body_pos"),
        "error_joint_pos_mean": metric_mean(motion, "error_joint_pos"),
        "error_anchor_lin_vel_mean": metric_mean(motion, "error_anchor_lin_vel"),
        "error_body_lin_vel_mean": metric_mean(motion, "error_body_lin_vel"),
        "error_joint_vel_mean": metric_mean(motion, "error_joint_vel"),
        "motion_count": metrics.get("motion_count", ""),
        "total_motion_frames": metrics.get("total_motion_frames", ""),
        "uses_official_importer_export_usd": metrics.get("uses_official_importer_export_usd", ""),
        "uses_resource_adjusted_usd": metrics.get("uses_resource_adjusted_usd", ""),
    }


def delta(best: dict[str, Any], final: dict[str, Any], key: str) -> float | None:
    lhs = safe_float(best.get(key))
    rhs = safe_float(final.get(key))
    if lhs is None or rhs is None:
        return None
    return lhs - rhs


def write_report_assets(summary: dict[str, Any]) -> dict[str, str]:
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    import pandas as pd

    rows = summary["comparison_rows"]
    df = pd.DataFrame(rows)
    csv_path = REPORT_OUT / "best_vs_final_checkpoint_confirmation.csv"
    df.to_csv(csv_path, index=False)
    plot_df = df.set_index("checkpoint_label")
    for col in ["reward_mean", "local_non_timeout_done_rate", "error_body_pos_mean", "error_joint_pos_mean"]:
        plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    plot_specs = [
        ("reward_mean", "Reward mean", "#059669"),
        ("local_non_timeout_done_rate", "Non-timeout done rate", "#ea580c"),
        ("error_body_pos_mean", "Body position error", "#2563eb"),
        ("error_joint_pos_mean", "Joint position error", "#dc2626"),
    ]
    for ax, (col, title, color) in zip(axes, plot_specs):
        plot_df[col].plot(kind="bar", ax=ax, color=color)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle("Full-size confirmation: sweep-selected best checkpoint vs final checkpoint", y=1.02)
    fig.tight_layout()
    plot_path = REPORT_OUT / "best_vs_final_checkpoint_confirmation.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    readme = REPORT_OUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Best Checkpoint Confirmation Eval",
                "",
                "Full-size local virtual confirmation of the sweep-selected scaled PPO checkpoint against the",
                "iteration-999 final checkpoint on the official-importer-export G1 USDA and full public motion bundle.",
                "",
                f"Best checkpoint iteration: `{summary['best_iteration']}`.",
                f"Best reward mean: `{summary['best_metrics'].get('reward_mean')}`.",
                f"Final reward mean: `{summary['final_metrics'].get('reward_mean')}`.",
                f"Reward delta best-minus-final: `{summary['deltas'].get('reward_mean')}`.",
                "",
                "Claim level: local virtual checkpoint confirmation only. This is not an official BeyondMimic",
                "teacher checkpoint, not paper success/fall/collision evaluation, and not a real-robot result.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"comparison_csv": str(csv_path), "comparison_png": str(plot_path), "readme": str(readme)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    training = load_json(TRAINING_RUN_JSON)
    sweep = load_json(SWEEP_JSON)
    final_eval = load_json(FINAL_EVAL_JSON)
    checkpoint = Path(sweep.get("best_checkpoint", {}).get("checkpoint", ""))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Sweep-selected checkpoint missing: {checkpoint}")
    iteration = int(sweep.get("best_checkpoint", {}).get("iteration", checkpoint_iteration(checkpoint)))
    num_envs = int(os.environ.get("BM_BEST_CHECKPOINT_CONFIRMATION_NUM_ENVS", str(DEFAULT_NUM_ENVS)))
    eval_steps = int(os.environ.get("BM_BEST_CHECKPOINT_CONFIRMATION_EVAL_STEPS", str(DEFAULT_EVAL_STEPS)))
    seed = int(os.environ.get("BM_BEST_CHECKPOINT_CONFIRMATION_SEED", str(DEFAULT_SEED)))

    shim = make_training_shim(training, checkpoint, iteration)
    base = load_base_module()
    base.OUT = OUT / "scratch"
    base.LOG_DIR = LOG_DIR
    base.RUN_ROOT = RUN_ROOT
    base.TRAINING_RUN_JSON = shim
    os.environ["BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_EVAL_SEED"] = str(seed)
    os.environ["BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_EVAL_NUM_ENVS"] = str(num_envs)
    os.environ["BM_PPO_EVAL_STEPS"] = str(eval_steps)
    os.environ["BM_PPO_EVAL_CANDIDATE_GPUS"] = "4,7"
    os.environ["BM_PPO_EVAL_VISIBLE_GPU_LIMIT"] = "2"
    base.main()

    eval_json = base.OUT / "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json"
    best_eval = load_json(eval_json)
    best_metrics = extract_metrics(best_eval)
    final_metrics = extract_metrics(final_eval)
    comparison_rows = [
        {"checkpoint_label": f"best_iter_{iteration}", **best_metrics},
        {"checkpoint_label": "final_iter_999", **final_metrics},
    ]
    fields = list(comparison_rows[0].keys())
    rows_csv = OUT / "best_vs_final_checkpoint_confirmation.csv"
    write_csv(rows_csv, comparison_rows, fields)

    copied_outputs: dict[str, str] = {}
    for key in ["json", "metrics_json", "timeseries_csv", "gpu_metrics_csv", "log"]:
        src = Path(best_eval.get("outputs", {}).get(key) or best_eval.get("run", {}).get(key, ""))
        if src.is_file():
            dst = OUT / f"best_iter_{iteration}_{src.name}"
            shutil.copy2(src, dst)
            copied_outputs[key] = str(dst)

    deltas = {
        "reward_mean": delta(best_metrics, final_metrics, "reward_mean"),
        "local_non_timeout_done_rate": delta(best_metrics, final_metrics, "local_non_timeout_done_rate"),
        "error_body_pos_mean": delta(best_metrics, final_metrics, "error_body_pos_mean"),
        "error_joint_pos_mean": delta(best_metrics, final_metrics, "error_joint_pos_mean"),
    }
    summary = {
        "status": "ok_official_importer_export_scaled_ppo_best_checkpoint_confirmation_eval"
        if best_eval.get("status") == "ok_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval_completed"
        else "failed_official_importer_export_scaled_ppo_best_checkpoint_confirmation_eval",
        "experiment_type": "tracking_official_importer_export_scaled_ppo_best_checkpoint_confirmation_eval",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Runs a full-size 2048-env confirmation eval for the checkpoint selected by the scaled PPO checkpoint "
            "sweep, then compares it with the existing iteration-999 final-checkpoint eval."
        ),
        "config": {
            "best_iteration": iteration,
            "best_checkpoint": str(checkpoint),
            "num_envs": num_envs,
            "eval_steps": eval_steps,
            "seed": seed,
            "selected_physical_gpus": [4, 7],
            "source_sweep_json": str(SWEEP_JSON),
            "final_eval_json": str(FINAL_EVAL_JSON),
        },
        "best_iteration": iteration,
        "best_eval": best_eval,
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
        "comparison_rows": comparison_rows,
        "deltas": deltas,
        "checks": {
            "best_eval_completed": best_eval.get("status")
            == "ok_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval_completed",
            "num_envs_2048": best_metrics.get("num_envs") == 2048,
            "eval_steps_299": best_metrics.get("eval_steps") == 299,
            "uses_official_importer_export_usd": best_metrics.get("uses_official_importer_export_usd") is True,
            "does_not_use_resource_adjusted_usd": best_metrics.get("uses_resource_adjusted_usd") is False,
            "motion_count_40": best_metrics.get("motion_count") == 40,
            "total_motion_frames_11960": best_metrics.get("total_motion_frames") == 11960,
            "final_eval_loaded_iteration_999": final_metrics.get("loaded_iteration") == 999,
            "best_iteration_matches_sweep": iteration == int(sweep.get("metrics", {}).get("best_iteration")),
            "does_not_claim_paper_level_eval": True,
            "does_not_claim_official_beyondmimic_checkpoint": True,
            "does_not_claim_real_robot": True,
            "does_not_claim_goal_complete": True,
        },
        "outputs": {
            "json": str(OUT / "tracking_g1_official_importer_export_scaled_ppo_best_checkpoint_confirmation_eval.json"),
            "comparison_csv": str(rows_csv),
            **copied_outputs,
        },
        "interpretation": {
            "goal_complete": False,
            "claim_level": "local_virtual_scaled_ppo_best_checkpoint_confirmation_not_paper_level",
            "why_not_complete": (
                "This confirms a sweep-selected local PPO checkpoint against the local final checkpoint. It is not an "
                "official BeyondMimic teacher checkpoint, not paper success/fall/collision evaluation, not DAgger, "
                "and not real robot evidence."
            ),
        },
    }
    summary["report_assets"] = write_report_assets(summary)
    write_json(OUT / "tracking_g1_official_importer_export_scaled_ppo_best_checkpoint_confirmation_eval.json", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "best_iteration": iteration,
                "best_reward_mean": best_metrics.get("reward_mean"),
                "final_reward_mean": final_metrics.get("reward_mean"),
                "reward_delta_best_minus_final": deltas["reward_mean"],
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if not summary["checks"]["best_eval_completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
