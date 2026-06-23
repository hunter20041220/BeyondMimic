#!/usr/bin/env python3
"""Generate a detailed BeyondMimic reproduction report package.

The report is intentionally evidence-first: it scans local project files and
loads existing JSON/CSV artifacts instead of inventing reproduction claims.
Generated content lives under ROOT/report and can be regenerated after new
experiments.
"""

from __future__ import annotations

import csv
import html
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(os.environ.get("BM_ROOT", "/mnt/infini-data/test/BeyondMimic")).resolve()
REPORT = ROOT / "report"


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def abs_path(rel_path: str) -> Path:
    return ROOT / rel_path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(rel_path: str, default: Any | None = None) -> Any:
    path = abs_path(rel_path)
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(rel_path: str, limit: int | None = None) -> str:
    path = abs_path(rel_path)
    if not path.exists():
        return f"NOT FOUND: {rel_path}"
    text = path.read_text(encoding="utf-8", errors="replace")
    return text if limit is None else text[:limit]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def run_capture(cmd: list[str], cwd: Path = ROOT, timeout: int = 120) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as exc:  # pragma: no cover - report fallback path
        return 999, "", repr(exc)


def ensure_dirs() -> None:
    dirs = [
        REPORT,
        REPORT / "figures",
        REPORT / "video_frames",
        REPORT / "tables",
        REPORT / "logs_summary",
        REPORT / "data" / "processed_data_examples",
        REPORT / "pipeline",
        REPORT / "code_review",
        REPORT / "experiments" / "training_curves",
        REPORT / "figures" / "paper_reproduced",
        REPORT / "figures" / "generated_schematics",
        REPORT / "figures" / "training_curves",
        REPORT / "figures" / "metric_plots",
        REPORT / "figures" / "video_frames",
        REPORT / "figures" / "failure_cases",
        REPORT / "videos" / "selected_success",
        REPORT / "videos" / "selected_failure",
        REPORT / "videos" / "thumbnails",
        REPORT / "logs" / "important_logs",
        REPORT / "logs" / "failure_logs",
        REPORT / "appendix",
    ]
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)


def make_inventory_files() -> None:
    patterns = (
        "*.py",
        "*.yaml",
        "*.yml",
        "*.json",
        "*.csv",
        "*.npz",
        "*.pt",
        "*.pth",
        "*.onnx",
        "*.mp4",
        "*.mkv",
        "*.avi",
        "*.gif",
        "*.log",
        "*.txt",
        "*.md",
        "*.pdf",
    )
    depth4: list[str] = []
    relevant: list[str] = []
    for path in ROOT.rglob("*"):
        if "report/" in path.as_posix():
            continue
        if not path.is_file():
            continue
        try:
            relative = path.relative_to(ROOT)
        except ValueError:
            continue
        if len(relative.parts) <= 4:
            depth4.append(str(relative))
        if any(path.match(pattern) for pattern in patterns):
            relevant.append(str(relative))
    write_text(REPORT / "file_inventory.txt", "\n".join(sorted(depth4)))
    write_text(REPORT / "file_tree_depth4.txt", "\n".join(sorted(depth4)))
    write_text(REPORT / "all_relevant_files.txt", "\n".join(sorted(relevant)))
    code, out2, err2 = run_capture(["du", "-h", "--max-depth=2", str(ROOT)], timeout=240)
    write_text(REPORT / "disk_usage_depth2.txt", out2 if code == 0 else err2)
    code, out3, err3 = run_capture(["du", "-h", "--max-depth=3", str(ROOT)], timeout=240)
    write_text(REPORT / "disk_usage_depth3.txt", out3 if code == 0 else err3)


def path_status(path: str) -> str:
    return "FOUND" if abs_path(path).exists() else "not found in current project"


def collect_artifacts() -> dict[str, Any]:
    return {
        "paper_pdf": "download/papers/BeyondMimic_2508.08241.pdf",
        "paper_source_tar": "download/papers/BeyondMimic_2508.08241_source.tar",
        "whole_body_tracking": "download/official/whole_body_tracking",
        "motion_tracking_controller": "download/official/motion_tracking_controller",
        "isaaclab": "download/dependencies/IsaacLab-v2.1.0",
        "lafan1_original": "download/official/ubisoft-laforge-animation-dataset/lafan1/lafan1.zip",
        "lafan1_retargeted_g1": "download/official/LAFAN1_Retargeting_Dataset/g1",
        "zenodo_dataset": "Dataset_beyondmimic",
        "mujoco_package": "mujoco_mp4",
        "official_mp4": "official_mp4",
        "stage1_motion_bundle": "res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle.json",
        "stage1_training": (
            "res/tracking/stage1_multisource_paper_contract_ppo_training_run/"
            "tracking_stage1_multisource_paper_contract_ppo_training_run.json"
        ),
        "stage1_sweep": (
            "res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/"
            "tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.json"
        ),
        "stage1_rollout": (
            "res/tracking/stage1_multisource_best_teacher_rollout_dataset/"
            "tracking_stage1_multisource_best_teacher_rollout_dataset.json"
        ),
        "stage1_vae": (
            "res/level_c/stage1_multisource_teacher_rollout_vae_training/"
            "level_c_stage1_multisource_teacher_rollout_vae_training.json"
        ),
        "stage1_state_latent": (
            "res/level_c/stage1_multisource_teacher_rollout_state_latent_dataset/"
            "level_c_stage1_multisource_teacher_rollout_state_latent_dataset.json"
        ),
        "stage1_diffusion": (
            "res/level_c/stage1_multisource_state_latent_diffusion_training/"
            "level_c_stage1_multisource_state_latent_diffusion_training.json"
        ),
        "stage1_guidance": (
            "res/level_c/stage1_multisource_state_latent_guidance_eval/"
            "level_c_stage1_multisource_state_latent_guidance_eval.json"
        ),
        "stage1_videos": (
            "res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/"
            "stage1_multisource_continuous_video_suite_summary.json"
        ),
        "required_absence": "res/required_artifact_absence/required_artifact_absence_audit.json",
        "master_audit": "res/master_audit/reproduction_master_audit.json",
    }


def metric_bundle() -> dict[str, Any]:
    motion = load_json("res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle.json")
    training = load_json(
        "res/tracking/stage1_multisource_paper_contract_ppo_training_run/"
        "tracking_stage1_multisource_paper_contract_ppo_training_run.json"
    )
    sweep = load_json(
        "res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/"
        "tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.json"
    )
    rollout = load_json(
        "res/tracking/stage1_multisource_best_teacher_rollout_dataset/"
        "tracking_stage1_multisource_best_teacher_rollout_dataset.json"
    )
    vae = load_json(
        "res/level_c/stage1_multisource_teacher_rollout_vae_training/"
        "level_c_stage1_multisource_teacher_rollout_vae_training.json"
    )
    state = load_json(
        "res/level_c/stage1_multisource_teacher_rollout_state_latent_dataset/"
        "level_c_stage1_multisource_teacher_rollout_state_latent_dataset.json"
    )
    diffusion = load_json(
        "res/level_c/stage1_multisource_state_latent_diffusion_training/"
        "level_c_stage1_multisource_state_latent_diffusion_training.json"
    )
    guidance = load_json(
        "res/level_c/stage1_multisource_state_latent_guidance_eval/"
        "level_c_stage1_multisource_state_latent_guidance_eval.json"
    )
    videos = load_json(
        "res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/"
        "stage1_multisource_continuous_video_suite_summary.json"
    )
    official_mp4 = load_json("official_mp4/official_mp4_manifest.json")
    comparison = load_json("res/comparison/paper_vs_reproduction.json")
    absence = load_json("res/required_artifact_absence/required_artifact_absence_audit.json")
    master = load_json("res/master_audit/reproduction_master_audit.json")
    return {
        "motion": motion,
        "training": training,
        "sweep": sweep,
        "rollout": rollout,
        "vae": vae,
        "state": state,
        "diffusion": diffusion,
        "guidance": guidance,
        "videos": videos,
        "official_mp4": official_mp4,
        "comparison": comparison,
        "absence": absence,
        "master": master,
    }


def get_nested(data: dict[str, Any], *keys: str, default: Any = "") -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def fmt_float(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def motion_hours(metrics: dict[str, Any]) -> Any:
    motion_metrics = metrics.get("motion", {}).get("metrics", {})
    return (
        motion_metrics.get("total_duration_hours")
        or motion_metrics.get("total_motion_hours")
        or get_nested(metrics.get("training", {}), "config", "motion_duration_hours")
    )


def write_dataset_tables(metrics: dict[str, Any]) -> None:
    motion = metrics["motion"]
    m = motion.get("metrics", {})
    inputs = motion.get("inputs", {})
    dataset_rows = [
        {
            "dataset": "Original LAFAN1",
            "source_path": "download/official/ubisoft-laforge-animation-dataset/lafan1/lafan1.zip",
            "format": "BVH zip",
            "used_in_current_stage1": "No direct use in latest 5/6 run",
            "purpose": "Original human motion source / provenance.",
            "status": path_status("download/official/ubisoft-laforge-animation-dataset/lafan1/lafan1.zip"),
            "notes": "Retargeted G1 CSVs are used for current teacher training instead of raw BVH.",
        },
        {
            "dataset": "Unitree-retargeted LAFAN1 G1",
            "source_path": inputs.get("lafan_root", "download/official/LAFAN1_Retargeting_Dataset/g1"),
            "format": "36-column generalized-coordinate CSV",
            "used_in_current_stage1": "Yes",
            "purpose": "Primary reference motions for Stage 1 PPO teacher.",
            "status": "FOUND",
            "notes": f"Current bundle includes {m.get('source_counts', {}).get('Unitree-retargeted LAFAN1', '')} LAFAN1 motions.",
        },
        {
            "dataset": "BeyondMimic Zenodo released data",
            "source_path": "Dataset_beyondmimic",
            "format": "CSV, MCAP/rosbag-derived result data, plotting scripts, GRF/IMU/ablation files",
            "used_in_current_stage1": "Partially",
            "purpose": "Mostly paper released-result analysis; tkd_skill.csv is a direct 36-column reference candidate.",
            "status": path_status("Dataset_beyondmimic"),
            "notes": "Not the full official diffusion training dataset or official checkpoints.",
        },
        {
            "dataset": "HuB supplemental motions",
            "source_path": "download/_supplemental/hub_data/drive_folder",
            "format": "29-DoF pkl candidates",
            "used_in_current_stage1": "Yes, available audited candidates",
            "purpose": "Additional balance / skill motions in multi-source Stage 1 candidate bundle.",
            "status": path_status("download/_supplemental/hub_data/drive_folder"),
            "notes": f"Current bundle includes {m.get('source_counts', {}).get('HuB supplemental 29-DoF pkl', '')} HuB motions.",
        },
        {
            "dataset": "PBHC/KungfuBot sidekick and ASAP Ronaldo sources",
            "source_path": "download/reference_code/PBHC and download/reference_code/ASAP if present",
            "format": "Non-29-DoF pkl/reference code sources",
            "used_in_current_stage1": "No",
            "purpose": "Candidate future motion sources.",
            "status": "partial / not train-ready",
            "notes": "Explicitly skipped until audited 23-to-29 G1 mapping exists.",
        },
        {
            "dataset": "Online animation packs",
            "source_path": "not found in current project scan",
            "format": "Not available locally as train-ready generalized-coordinate motions",
            "used_in_current_stage1": "No",
            "purpose": "Paper-described source category; not publicly reconstructed here.",
            "status": "not found in current project",
            "notes": "Do not claim full paper curated 2.5h data recovery from this project alone.",
        },
    ]
    write_csv(REPORT / "tables" / "dataset_inventory.csv", dataset_rows)
    write_csv(REPORT / "data" / "data_provenance_table.csv", dataset_rows)

    duration_rows: list[dict[str, Any]] = []
    for row in motion.get("rows", []):
        duration_rows.append(
            {
                "motion": row.get("motion"),
                "source_family": row.get("source_family"),
                "source_kind": row.get("source_kind"),
                "source_path": row.get("source_path"),
                "frame_count": row.get("frame_count"),
                "fps": 50,
                "duration_seconds": row.get("duration_seconds"),
                "entered_stage1_training": "yes",
            }
        )
    write_csv(REPORT / "data" / "motion_duration_summary.csv", duration_rows)
    write_csv(REPORT / "data" / "motion_file_manifest.csv", duration_rows)
    write_csv(REPORT / "tables" / "motion_duration_summary.csv", duration_rows)


def module_status_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    sweep_metrics = metrics["sweep"].get("metrics", {})
    rollout_metrics = metrics["rollout"].get("aggregate_metrics", {})
    vae_eval = get_nested(metrics["vae"], "worker_summary", "evaluation", "test", default={})
    state_dataset = get_nested(metrics["state"], "worker_summary", "dataset", default={})
    diffusion_eval = get_nested(metrics["diffusion"], "worker_summary", "evaluation", "test", default={})
    guidance_metrics = get_nested(metrics["guidance"], "worker_summary", "metrics", default={})
    videos = metrics["videos"]
    return [
        {
            "module": "Data collection and motion bundle",
            "status": "PARTIAL",
            "evidence_path": "res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle.json",
            "current_result": (
                f"{get_nested(metrics['motion'], 'metrics', 'motion_count')} motions, "
                f"{fmt_float(motion_hours(metrics), 3)} h"
            ),
            "gap": "Available bundle approximates paper duration but not the exact unreleased curated collection.",
        },
        {
            "module": "PPO motion tracking teacher",
            "status": "FAILED/PARTIAL",
            "evidence_path": "res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.json",
            "current_result": (
                f"best iteration {sweep_metrics.get('best_iteration')}, reward "
                f"{sweep_metrics.get('best_reward_mean')}, body error {sweep_metrics.get('best_error_body_pos_mean')}"
            ),
            "gap": "Teacher is weak; current reward/error/done metrics are not paper-quality tracking.",
        },
        {
            "module": "Teacher rollout dataset",
            "status": "PARTIAL",
            "evidence_path": "res/tracking/stage1_multisource_best_teacher_rollout_dataset/tracking_stage1_multisource_best_teacher_rollout_dataset.json",
            "current_result": f"{rollout_metrics.get('total_env_steps')} env steps, done_count={rollout_metrics.get('done_count_total')}",
            "gap": "Useful local state-action data but not official DAgger rollout logs.",
        },
        {
            "module": "Conditional VAE",
            "status": "PARTIAL",
            "evidence_path": "res/level_c/stage1_multisource_teacher_rollout_vae_training/level_c_stage1_multisource_teacher_rollout_vae_training.json",
            "current_result": f"test action MSE {vae_eval.get('action_mse')}",
            "gap": "Offline reconstruction only; true DAgger and stable closed-loop VAE control remain missing.",
        },
        {
            "module": "State-latent dataset",
            "status": "PARTIAL",
            "evidence_path": "res/level_c/stage1_multisource_teacher_rollout_state_latent_dataset/level_c_stage1_multisource_teacher_rollout_state_latent_dataset.json",
            "current_result": f"{state_dataset.get('window_count')} windows, token_dim={state_dataset.get('token_dim')}",
            "gap": "Generated from weak local teacher, not official paper rollout dataset.",
        },
        {
            "module": "Diffusion denoiser",
            "status": "PARTIAL",
            "evidence_path": "res/level_c/stage1_multisource_state_latent_diffusion_training/level_c_stage1_multisource_state_latent_diffusion_training.json",
            "current_result": (
                f"test pred token MSE {diffusion_eval.get('pred_token_mse')}, noisy token MSE "
                f"{diffusion_eval.get('noisy_token_mse')}"
            ),
            "gap": "Token denoising improves, but closed-loop control is not stable.",
        },
        {
            "module": "Classifier/task guidance",
            "status": "PARTIAL",
            "evidence_path": "res/level_c/stage1_multisource_state_latent_guidance_eval/level_c_stage1_multisource_state_latent_guidance_eval.json",
            "current_result": (
                f"{guidance_metrics.get('total_selected_windows')} offline windows, "
                f"{guidance_metrics.get('tasks_with_all_best_costs_improve')} tasks improve in offline proxy"
            ),
            "gap": "Offline proxy only; not paper Fig. 5/Fig. 6 closed-loop task success.",
        },
        {
            "module": "MuJoCo action-control videos",
            "status": "FAILED/PARTIAL",
            "evidence_path": "res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/stage1_multisource_continuous_video_suite_summary.json",
            "current_result": (
                f"{len(videos.get('videos', {}))} videos, checks={videos.get('checks', {})}"
            ),
            "gap": "Videos are continuous and physically stepped, but fall proxies and QACC warnings show poor motion quality.",
        },
    ]


def write_module_tables(metrics: dict[str, Any]) -> None:
    rows = module_status_rows(metrics)
    write_csv(REPORT / "tables" / "module_status.csv", rows)
    lines = ["# Module Status", "", "| module | status | current result | evidence | gap |", "|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['module']} | {row['status']} | {row['current_result']} | `{row['evidence_path']}` | {row['gap']} |"
        )
    write_text(REPORT / "reproduction_status.md", "\n".join(lines))


def write_metric_tables(metrics: dict[str, Any]) -> dict[str, float]:
    diffusion_eval = get_nested(metrics["diffusion"], "worker_summary", "evaluation", "test", default={})
    noisy = float(diffusion_eval.get("noisy_token_mse", 0.07281625297452722))
    pred = float(diffusion_eval.get("pred_token_mse", 0.04322136765612023))
    improvement = (noisy - pred) / noisy if noisy else 0.0
    sweep_metrics = metrics["sweep"].get("metrics", {})
    vae_eval = get_nested(metrics["vae"], "worker_summary", "evaluation", "test", default={})
    guidance_metrics = get_nested(metrics["guidance"], "worker_summary", "metrics", default={})
    video_segment = metrics["videos"].get("selected_continuous_segment", {})
    rows = [
        {"metric": "Stage1 motion count", "value": get_nested(metrics["motion"], "metrics", "motion_count"), "unit": "motions", "source": "stage1 motion bundle"},
        {"metric": "Stage1 motion duration", "value": motion_hours(metrics), "unit": "hours", "source": "stage1 motion bundle"},
        {"metric": "Best teacher reward mean", "value": sweep_metrics.get("best_reward_mean"), "unit": "reward", "source": "checkpoint sweep"},
        {"metric": "Best teacher body error mean", "value": sweep_metrics.get("best_error_body_pos_mean"), "unit": "m/proxy", "source": "checkpoint sweep"},
        {"metric": "Best teacher joint error mean", "value": sweep_metrics.get("best_error_joint_pos_mean"), "unit": "rad/proxy", "source": "checkpoint sweep"},
        {"metric": "Teacher rollout env steps", "value": get_nested(metrics["rollout"], "aggregate_metrics", "total_env_steps"), "unit": "steps", "source": "teacher rollout dataset"},
        {"metric": "VAE test action MSE", "value": vae_eval.get("action_mse"), "unit": "MSE", "source": "stage1 VAE"},
        {"metric": "Noisy token MSE", "value": noisy, "unit": "MSE", "source": "stage1 diffusion"},
        {"metric": "Test pred token MSE", "value": pred, "unit": "MSE", "source": "stage1 diffusion"},
        {"metric": "Relative denoising improvement", "value": improvement, "unit": "ratio", "source": "computed"},
        {"metric": "Offline guidance selected windows", "value": guidance_metrics.get("total_selected_windows"), "unit": "windows", "source": "stage1 guidance"},
        {"metric": "Continuous MuJoCo video duration", "value": video_segment.get("duration_seconds"), "unit": "seconds", "source": "stage1 video suite"},
    ]
    write_csv(REPORT / "tables" / "metrics_summary.csv", rows)
    write_csv(REPORT / "experiments" / "metrics_summary.csv", rows)
    lines = [
        "# Metrics Summary",
        "",
        "| metric | value | unit | source |",
        "|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['metric']} | {row['value']} | {row['unit']} | {row['source']} |")
    write_text(REPORT / "experiments" / "metrics_summary.md", "\n".join(lines))
    return {"noisy": noisy, "pred": pred, "improvement": improvement}


def draw_mse_chart(values: dict[str, float]) -> None:
    names = ["Noisy token MSE", "Predicted token MSE"]
    vals = [values["noisy"], values["pred"]]
    colors = ["#8f8f8f", "#2f6fdf"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylabel("MSE")
    ax.set_title(f"Stage1 multi-source denoising improvement: {values['improvement'] * 100:.1f}%")
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + max(vals) * 0.02, f"{val:.5f}", ha="center")
    fig.tight_layout()
    for out in [
        REPORT / "figures" / "denoising_mse_improvement.png",
        REPORT / "figures" / "metric_plots" / "denoising_mse_improvement.png",
    ]:
        fig.savefig(out, dpi=180)
    for out in [
        REPORT / "figures" / "denoising_mse_improvement.svg",
        REPORT / "figures" / "metric_plots" / "denoising_mse_improvement.svg",
    ]:
        fig.savefig(out)
    plt.close(fig)


def draw_checkpoint_chart(metrics: dict[str, Any]) -> None:
    csv_path = abs_path(
        "res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/"
        "tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep_rows.csv"
    )
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if "iteration" not in df or "reward_mean" not in df:
        return
    fig, ax1 = plt.subplots(figsize=(8.2, 4.6))
    ax1.plot(df["iteration"], df["reward_mean"], marker="o", color="#2f6fdf", label="reward_mean")
    ax1.set_xlabel("checkpoint iteration")
    ax1.set_ylabel("reward_mean", color="#2f6fdf")
    ax1.tick_params(axis="y", labelcolor="#2f6fdf")
    if "error_body_pos_mean" in df:
        ax2 = ax1.twinx()
        ax2.plot(df["iteration"], df["error_body_pos_mean"], marker="s", color="#c33c3c", label="body_error")
        ax2.set_ylabel("body-position error mean", color="#c33c3c")
        ax2.tick_params(axis="y", labelcolor="#c33c3c")
    ax1.set_title("Stage1 multi-source checkpoint sweep")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORT / "figures" / "metric_plots" / "stage1_checkpoint_sweep.png", dpi=180)
    fig.savefig(REPORT / "figures" / "metric_plots" / "stage1_checkpoint_sweep.svg")
    plt.close(fig)


def draw_training_curve(rel_csv: str, out_stem: str, y_cols: list[str], title: str) -> None:
    path = abs_path(rel_csv)
    if not path.exists():
        return
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception:
        df = pd.read_csv(path)
    x = df["epoch"] if "epoch" in df else range(len(df))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for col in y_cols:
        if col in df:
            ax.plot(x, df[col], marker="o", linewidth=1.6, label=col)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss / metric")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT / "figures" / "training_curves" / f"{out_stem}.png", dpi=180)
    fig.savefig(REPORT / "figures" / "training_curves" / f"{out_stem}.svg")
    fig.savefig(REPORT / "experiments" / "training_curves" / f"{out_stem}.png", dpi=180)
    plt.close(fig)


def write_svg_flow(path: Path, title: str, nodes: list[tuple[str, str]]) -> None:
    width = 1200
    node_h = 82
    gap = 28
    top = 80
    height = top + len(nodes) * (node_h + gap) + 50
    colors = ["#e8f1ff", "#eef8ec", "#fff2dc", "#f5ebff", "#ffecec", "#eef7f7"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif}.title{font-size:30px;font-weight:700}.label{font-size:20px;font-weight:700}.desc{font-size:15px}</style>",
        f'<text class="title" x="{width/2}" y="42" text-anchor="middle">{html.escape(title)}</text>',
    ]
    x = 170
    w = 860
    for idx, (label, desc) in enumerate(nodes):
        y = top + idx * (node_h + gap)
        fill = colors[idx % len(colors)]
        parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{node_h}" rx="10" fill="{fill}" stroke="#333" stroke-width="1.2"/>')
        parts.append(f'<text class="label" x="{x+24}" y="{y+31}">{html.escape(label)}</text>')
        parts.append(f'<text class="desc" x="{x+24}" y="{y+58}">{html.escape(desc[:128])}</text>')
        if idx < len(nodes) - 1:
            y1 = y + node_h
            y2 = y + node_h + gap - 6
            cx = x + w / 2
            parts.append(f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2}" stroke="#444" stroke-width="2"/>')
            parts.append(f'<polygon points="{cx-7},{y2-1} {cx+7},{y2-1} {cx},{y2+10}" fill="#444"/>')
    parts.append("</svg>")
    write_text(path, "\n".join(parts))


def draw_flow_png(path: Path, title: str, nodes: list[tuple[str, str]]) -> None:
    fig_h = max(5.0, 1.0 + len(nodes) * 0.85)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=18, weight="bold", pad=18)
    y_positions = list(reversed(range(len(nodes))))
    for idx, ((label, desc), y) in enumerate(zip(nodes, y_positions)):
        ax.text(
            0.5,
            y,
            f"{label}\n{desc}",
            ha="center",
            va="center",
            fontsize=10.5,
            bbox=dict(boxstyle="round,pad=0.55", fc="#f4f7fb", ec="#2b2b2b", lw=1.0),
        )
        if idx < len(nodes) - 1:
            ax.annotate("", xy=(0.5, y - 0.58), xytext=(0.5, y - 0.18), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, len(nodes) - 0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_flow_diagrams() -> None:
    diagrams: dict[str, tuple[str, list[tuple[str, str]]]] = {
        "pipeline_overview": (
            "BeyondMimic reproduction pipeline",
            [
                ("Data sources", "LAFAN1 G1 CSV, Zenodo result data, HuB candidates; exact paper curated set not fully public."),
                ("Motion preprocessing", "Validate G1 generalized coordinates, FK/body tensors, metadata, duration statistics."),
                ("PPO teacher", "Official whole_body_tracking / IsaacLab task; latest 5/6 run completes but teacher remains weak."),
                ("Teacher rollout", "Collect obs/action/reward/done/motion_time_steps from selected local teacher."),
                ("Conditional VAE", "Encode teacher action distribution into 32-D latent and decode action from obs+z."),
                ("State-latent diffusion", "Train denoiser over 21-token state+latent windows; token MSE improves."),
                ("Classifier guidance", "Offline joystick/waypoint/smoothness proxy costs; not paper-level closed loop."),
                ("MuJoCo video", "Render continuous action-to-PD diagnostics; videos reveal instability/fall proxies."),
            ],
        ),
        "data_flow": (
            "Data flow",
            [
                ("Raw / released sources", "BVH, 36-col CSV, MCAP/rosbag-derived released evidence."),
                ("Train-ready references", "G1 generalized-coordinate CSV/NPZ with joint/body tensors."),
                ("Stage1 bundle", "49 motions / 2.49 h local public+available bundle."),
                ("Rollout shards", "612,352 local weak-teacher state-action samples."),
                ("State-latent windows", "571,392 windows with 160-D obs + 32-D latent."),
            ],
        ),
        "stage1_tracking": (
            "Stage 1: PPO motion tracking",
            [
                ("Input", "Reference motion NPZ, Unitree G1 asset, tracking reward, PPO config."),
                ("Policy loop", "obs -> PPO actor -> 29-D action -> PD target in physics."),
                ("Reward/termination", "Tracking terms and reset gates decide learning signal."),
                ("Current result", "Best 5/6 checkpoint reward mean ~0.024 and high error: weak teacher."),
            ],
        ),
        "stage2_vae": (
            "Stage 2: Conditional action VAE",
            [
                ("Input", "Teacher rollout obs/action pairs."),
                ("Encoder", "q(z|obs, action) produces 32-D latent posterior."),
                ("Decoder", "D(obs,z) reconstructs 29-D action."),
                ("Current result", "Low offline action MSE, but closed-loop stability not proven."),
            ],
        ),
        "stage3_diffusion": (
            "Stage 3: State-latent diffusion",
            [
                ("Input", "State-latent token windows from teacher rollout and VAE."),
                ("Noising", "x_t = sqrt(a_bar)x_0 + sqrt(1-a_bar)eps."),
                ("Denoising", "MLP denoiser predicts clean token from noisy token + timestep."),
                ("Current result", "Noisy MSE 0.0728 -> pred MSE 0.04322, about 40.6% improvement."),
            ],
        ),
        "stage4_guidance": (
            "Stage 4: Test-time guidance",
            [
                ("Input", "Unconditional denoiser output and task cost."),
                ("Gradient", "Compute task-cost gradient w.r.t. trajectory tokens."),
                ("Receding action", "Decode current latent to action and step simulation."),
                ("Current status", "Offline proxy guidance only for this chain; closed-loop task success absent."),
            ],
        ),
        "mujoco_or_isaac_rendering": (
            "MuJoCo / Isaac video rendering",
            [
                ("Isaac", "Blocked on H20 Vulkan/Kit rendering stack for true Isaac rendered MP4."),
                ("MuJoCo", "Loads G1 MJCF, maps action to joint setpoints, steps physics, renders MP4."),
                ("Current videos", "Continuous 298-frame local diagnostics, but unstable QACC/fall proxy remains."),
                ("Claim level", "Local virtual evidence only, not real robot and not official paper video."),
            ],
        ),
        "failure_diagnosis": (
            "Failure diagnosis map",
            [
                ("Data", "Check retargeting, joint order, root frame, FPS, impossible segments."),
                ("Teacher", "Low reward/high done means downstream models imitate weak behavior."),
                ("VAE/diffusion", "Offline MSE can improve while physical rollout remains invalid."),
                ("Deployment", "Check action scale, PD gain, default pose, obs normalization, joint map."),
            ],
        ),
    }
    for name, (title, nodes) in diagrams.items():
        write_svg_flow(REPORT / "figures" / f"{name}.svg", title, nodes)
        draw_flow_png(REPORT / "figures" / f"{name}.png", title, nodes)
        write_text(
            REPORT / "pipeline" / f"{name}.md",
            "# " + title + "\n\n" + "\n".join(f"{i+1}. **{a}**: {b}" for i, (a, b) in enumerate(nodes)),
        )
        write_svg_flow(REPORT / "pipeline" / f"{name}.svg", title, nodes)
        draw_flow_png(REPORT / "pipeline" / f"{name}.png", title, nodes)
    # Compatibility names requested by the longer prompt.
    copies = {
        "pipeline_overview": "full_pipeline",
        "stage2_vae": "stage2_vae_dagger",
        "mujoco_or_isaac_rendering": "mujoco_deployment",
    }
    for src, dst in copies.items():
        for ext in ["svg", "png"]:
            shutil.copyfile(REPORT / "figures" / f"{src}.{ext}", REPORT / "pipeline" / f"{dst}.{ext}")


def ffprobe(video: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,duration,nb_frames",
        "-of",
        "json",
        str(video),
    ]
    code, out, _ = run_capture(cmd, timeout=20)
    if code != 0:
        return {}
    try:
        data = json.loads(out)
        stream = data.get("streams", [{}])[0]
        return stream
    except Exception:
        return {}


def extract_video_frames(video_rows: list[dict[str, Any]]) -> None:
    selected = [
        row
        for row in video_rows
        if "stage1_multisource_continuous_mujoco_action_control_videos" in row["video_path"]
    ][:6]
    montage_images: list[Path] = []
    for row in selected:
        video = abs_path(row["video_path"])
        stem = video.parent.name
        duration = float(row.get("duration_seconds") or 0.0)
        times = [0.2, max(duration * 0.5, 0.2), max(duration - 0.3, 0.2)] if duration else [0.2, 1.0, 2.0]
        for idx, t in enumerate(times):
            out = REPORT / "video_frames" / f"{stem}_{idx}.png"
            cmd = ["ffmpeg", "-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", "-q:v", "2", str(out)]
            code, _, _ = run_capture(cmd, timeout=40)
            if code == 0 and out.exists():
                montage_images.append(out)
                shutil.copyfile(out, REPORT / "figures" / "video_frames" / out.name)
    if not montage_images:
        return
    import matplotlib.image as mpimg

    imgs = []
    for path in montage_images[:18]:
        try:
            imgs.append((path, mpimg.imread(path)))
        except Exception:
            pass
    if not imgs:
        return
    cols = 3
    rows = (len(imgs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.0))
    axes_list = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax, (path, img) in zip(axes_list, imgs):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(path.stem, fontsize=8)
    for ax in axes_list[len(imgs) :]:
        ax.axis("off")
    fig.suptitle("Stage1 multi-source MuJoCo video frame montage (diagnostic)", fontsize=14)
    fig.tight_layout()
    fig.savefig(REPORT / "figures" / "failure_montage.png", dpi=180)
    fig.savefig(REPORT / "figures" / "failure_cases" / "failure_montage.png", dpi=180)
    plt.close(fig)


def write_video_index(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    video_paths = sorted(
        [
            p
            for p in ROOT.rglob("*")
            if p.is_file()
            and p.suffix.lower() in {".mp4", ".mkv", ".avi", ".gif"}
            and "report/" not in p.as_posix()
        ]
    )
    rows: list[dict[str, Any]] = []
    for p in video_paths:
        info = ffprobe(p) if p.suffix.lower() != ".gif" else {}
        r = rel(p)
        stage = "unknown"
        status = "unknown"
        suspected = ""
        if "stage1_multisource_continuous_mujoco_action_control_videos" in r:
            stage = "stage1 multi-source MuJoCo action-control diagnostic"
            status = "failure/diagnostic"
            suspected = "weak teacher and sim-control mismatch; fall proxy high"
        elif "official_mp4" in r:
            stage = "official released-data MuJoCo state replay"
            status = "visualization only"
            suspected = "not policy closed-loop control"
        elif "isaac_mp4" in r:
            stage = "Isaac rendered MP4 gate"
            status = "blocked/failed"
            suspected = "H20 Vulkan/Kit rendering stack"
        rows.append(
            {
                "video_path": r,
                "stage": stage,
                "duration_seconds": info.get("duration", ""),
                "resolution": f"{info.get('width', '')}x{info.get('height', '')}".strip("x"),
                "frames": info.get("nb_frames", ""),
                "status": status,
                "visual_description": "indexed local video; inspect frames/thumbnails for qualitative judgment",
                "suspected_problem": suspected,
            }
        )
    write_csv(REPORT / "videos" / "video_index.csv", rows)
    lines = ["# Video Index", "", "| video | stage | duration | resolution | status | suspected problem |", "|---|---|---:|---|---|---|"]
    for row in rows:
        lines.append(
            f"| `{row['video_path']}` | {row['stage']} | {row['duration_seconds']} | {row['resolution']} | {row['status']} | {row['suspected_problem']} |"
        )
    write_text(REPORT / "videos" / "video_index.md", "\n".join(lines))
    write_text(REPORT / "video_index.md", "\n".join(lines))
    extract_video_frames(rows)
    return rows


def numbered_snippet(path: str, start: int, end: int) -> str:
    p = abs_path(path)
    if not p.exists():
        return f"```text\nNOT FOUND: {path}\n```"
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(1, start)
    end = min(len(lines), end)
    body = "\n".join(f"{idx:04d}: {lines[idx-1]}" for idx in range(start, end + 1))
    return f"```python\n{body}\n```"


def find_snippet(path: str, pattern: str, before: int = 8, after: int = 30) -> tuple[int, int]:
    p = abs_path(path)
    if not p.exists():
        return (1, 1)
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    rx = re.compile(pattern)
    for idx, line in enumerate(lines, start=1):
        if rx.search(line):
            return (max(1, idx - before), min(len(lines), idx + after))
    return (1, min(len(lines), before + after))


def code_index_rows() -> list[dict[str, Any]]:
    specs = [
        ("data preprocessing", "csv_to_npz", "download/official/whole_body_tracking/scripts/csv_to_npz.py", "official", "CSV generalized coordinates to motion NPZ / registry workflow"),
        ("tracking", "MotionCommand", "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py", "official", "Reference-motion command and tracking target computation"),
        ("tracking", "reward terms", "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py", "official", "DeepMimic-style tracking rewards and smoothing terms"),
        ("tracking", "observations", "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py", "official", "Policy observation construction"),
        ("tracking", "terminations", "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py", "official", "Early termination conditions"),
        ("tracking", "PPO train entry", "download/official/whole_body_tracking/scripts/rsl_rl/train.py", "official", "RSL-RL PPO training entry"),
        ("teacher rollout", "rollout collection", "reproduction/scripts/tracking_g1_resource_adjusted_teacher_rollout_dataset.py", "custom wrapper", "Collect state/action/reward/done rollout shards"),
        ("teacher rollout", "5/6 wrapper", "reproduction/scripts/tracking_stage1_multisource_best_teacher_rollout_dataset.py", "custom wrapper", "Bind best multi-source checkpoint to rollout collector"),
        ("VAE", "ConditionalActionVAE", "reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py", "paper-faithful local", "Encoder/decoder action VAE"),
        ("state-latent", "state-latent dataset", "reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py", "paper-faithful local", "Build token windows from obs and VAE latents"),
        ("diffusion", "StateLatentDenoiser", "reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py", "paper-faithful local", "Noising and denoising training loop"),
        ("guidance", "offline guidance", "reproduction/scripts/level_c_resource_adjusted_state_latent_guidance_eval.py", "paper-faithful local", "Task-cost guidance proxy evaluation"),
        ("MuJoCo", "PD video rendering", "mujoco_mp4/scripts/mujoco_pd_control_video.py", "custom MuJoCo", "Action-to-PD control video rendering"),
        ("MuJoCo", "continuous video suite", "reproduction/scripts/stage1_multisource_continuous_mujoco_action_control_videos.py", "custom wrapper", "Fresh 5/6 continuous MuJoCo video suite"),
    ]
    rows: list[dict[str, Any]] = []
    for stage, func, path, origin, desc in specs:
        p = abs_path(path)
        line_range = ""
        if p.exists():
            start, end = find_snippet(path, re.escape(func.split()[0]) if func != "reward terms" else "def ")
            line_range = f"{start}-{end}"
        rows.append(
            {
                "stage": stage,
                "functionality": func,
                "file_path": path,
                "class_or_function": func,
                "line_range_if_detected": line_range,
                "whether_official_paper_faithful_custom": origin,
                "description": desc,
                "report_section": stage,
                "status": "FOUND" if p.exists() else "not found in current project scan",
            }
        )
    return rows


def snippet_block(title: str, path: str, pattern: str, purpose: str, paper_relation: str) -> str:
    start, end = find_snippet(path, pattern)
    return "\n".join(
        [
            f"## {title}",
            "",
            f"File: `{path}`",
            f"Function/Class: `{pattern}`",
            f"Purpose: {purpose}",
            "Input: see function signature and surrounding module context.",
            "Output: see return values / written artifacts.",
            f"Paper relation: {paper_relation}",
            "",
            numbered_snippet(path, start, end),
            "",
        ]
    )


def write_code_reports() -> None:
    rows = code_index_rows()
    write_csv(REPORT / "code_review" / "key_code_index.csv", rows)
    lines = ["# Key Code Index", "", "| stage | functionality | file | line range | origin | status | description |", "|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['stage']} | {row['functionality']} | `{row['file_path']}` | {row['line_range_if_detected']} | "
            f"{row['whether_official_paper_faithful_custom']} | {row['status']} | {row['description']} |"
        )
    write_text(REPORT / "code_review" / "key_code_index.md", "\n".join(lines))
    write_text(REPORT / "code_review" / "code_inventory.md", "\n".join(lines))

    tracking = [
        snippet_block(
            "Official motion command / target computation",
            "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py",
            r"class .*Command|def .*command|def .*metrics",
            "Loads reference motion and computes target/error signals for the tracking MDP.",
            "Corresponds to Stage 1 motion tracking target and anchor-centered command logic.",
        ),
        snippet_block(
            "Official reward terms",
            "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py",
            r"def ",
            "Defines tracking reward and smoothing terms.",
            "Corresponds to paper tracking reward implementation in the public Stage-1 repo.",
        ),
        snippet_block(
            "Teacher rollout collection wrapper",
            "reproduction/scripts/tracking_g1_resource_adjusted_teacher_rollout_dataset.py",
            r"def main",
            "Runs the selected teacher and writes rollout shards.",
            "Provides local state-action data for downstream VAE/diffusion experiments.",
        ),
    ]
    vae = [
        snippet_block(
            "Conditional action VAE",
            "reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py",
            r"class ConditionalActionVAE",
            "Encodes obs+action into a latent and decodes obs+z back to action.",
            "Paper-faithful local reimplementation of latent action distillation, but not official checkpoint.",
        )
    ]
    diffusion = [
        snippet_block(
            "State-latent denoiser",
            "reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py",
            r"class StateLatentDenoiser",
            "Predicts clean state-latent tokens from noisy tokens and timestep.",
            "Local denoising objective corresponding to state-latent diffusion training.",
        ),
        snippet_block(
            "Diffusion noising and loss loop",
            "reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py",
            r"noisy = torch.sqrt",
            "Applies DDPM-style noising and computes token MSE.",
            "Matches the clean-token denoising formulation used for local training.",
        ),
    ]
    guidance = [
        snippet_block(
            "Offline state-latent guidance evaluation",
            "reproduction/scripts/level_c_resource_adjusted_state_latent_guidance_eval.py",
            r"def .*cost|def main|tasks",
            "Evaluates task-cost gradients and scale response over denoiser outputs.",
            "Implements local classifier/task guidance proxies for joystick/waypoint/smoothness-style costs.",
        )
    ]
    mujoco = [
        snippet_block(
            "MuJoCo action-to-PD rendering",
            "mujoco_mp4/scripts/mujoco_pd_control_video.py",
            r"def .*render|def .*step|def .*action",
            "Loads G1 model, maps action to joint setpoints, steps MuJoCo, and renders MP4.",
            "Local simulation visualization path; not official Isaac rendering.",
        ),
        snippet_block(
            "Continuous 5/6 video suite wrapper",
            "reproduction/scripts/stage1_multisource_continuous_mujoco_action_control_videos.py",
            r"def patch_artifact_bindings",
            "Binds the latest multi-source teacher/VAE/denoiser artifacts and filters continuous segments.",
            "Prevents reset-spliced video artifacts and records honest claim level.",
        ),
    ]
    write_text(REPORT / "code_review" / "key_snippets_tracking.md", "\n".join(["# Tracking Code Snippets", "", *tracking]))
    write_text(REPORT / "code_review" / "key_snippets_vae.md", "\n".join(["# VAE Code Snippets", "", *vae]))
    write_text(REPORT / "code_review" / "key_snippets_diffusion.md", "\n".join(["# Diffusion Code Snippets", "", *diffusion]))
    write_text(REPORT / "code_review" / "key_snippets_guidance.md", "\n".join(["# Guidance Code Snippets", "", *guidance]))
    write_text(REPORT / "code_review" / "key_snippets_mujoco.md", "\n".join(["# MuJoCo Code Snippets", "", *mujoco]))
    full = "\n".join(["# Full Code Snippets", "", *(tracking + vae + diffusion + guidance + mujoco)])
    write_text(REPORT / "appendix" / "full_code_snippets.md", full)
    write_text(REPORT / "code_snippets.md", full)


def pseudocode_text() -> str:
    return """# Pseudocode for All Stages

## Algorithm 1: Motion Preprocessing

```text
Input: retargeted G1 CSV / BVH-like source, expected joint order, FPS
Output: reference motion NPZ with q_ref, v_ref, body poses, body velocities, metadata
1. Load source motion table.
2. Validate root pose and 29 actuated joint columns.
3. Convert or verify generalized coordinates in Unitree G1 order.
4. Run FK to recover body positions/orientations/velocities.
5. Reject sources whose DoF mapping is not audited.
6. Save per-motion NPZ and a bundle manifest with duration/source provenance.
```

## Algorithm 2: PPO Motion Tracking Teacher Training

```text
Input: processed reference bundle, G1 asset, tracking task config, PPO config
Output: teacher policy checkpoint
for each environment reset:
    sample a motion and phase
    initialize robot near the reference state
for each control step:
    build observation from proprioception and reference tracking cues
    action = PPO_actor(observation)
    theta_sp = theta_default + action_scale * action
    physics.step(PD(theta_sp))
    reward = tracking_reward + regularization
    done = termination_checks(robot_state, reference_error)
    store transition
after horizon:
    update PPO actor/critic with RSL-RL
    save checkpoint every configured interval
```

## Algorithm 3: Teacher Rollout Collection

```text
Input: trained teacher checkpoint, reference bundle, simulation environment
Output: state-action rollout shards
load teacher policy
for each rank/env:
    reset to sampled motion phase
    for T steps:
        obs_t = env.get_obs()
        a_t = teacher(obs_t)
        obs_{t+1}, reward_t, done_t = env.step(a_t)
        record obs_t, a_t, reward_t, done_t, motion_time_step_t
        if done: record reset boundary; do not stitch across resets for video evidence
save NPZ shards and metrics
```

## Algorithm 4: Conditional VAE Training

```text
Input: teacher rollout obs/action pairs
Output: VAE checkpoint and latent action representation
for minibatch (obs, action):
    mu, logvar = Encoder([obs, action])
    z = mu + exp(0.5 * logvar) * epsilon
    action_hat = Decoder([obs, z])
    loss = MSE(action_hat, action) + beta * KL(q(z|obs,action) || N(0,I))
    update encoder and decoder
```

## Algorithm 5: DAgger Loop (paper target; current project is partial)

```text
Input: student/VAE policy, teacher policy, simulation environment
Output: aggregated on-policy state-action dataset
dataset = initial teacher rollouts
repeat:
    rollout student policy in simulation
    at visited states, query teacher action
    append (student_state, teacher_action) to dataset
    retrain or fine-tune VAE/student
Current project: offline teacher-rollout VAE exists; full official DAgger logs are not available.
```

## Algorithm 6: State-Latent Trajectory Dataset

```text
Input: teacher rollout observations, actions, trained VAE
Output: windows of tau = [state, latent] tokens
for each rollout shard:
    infer z_t from VAE posterior
    concatenate token_t = [obs_t, z_t]
    create fixed-length windows of 21 tokens
    split windows into train/validation/test
save state-latent dataset and index metadata
```

## Algorithm 7: Diffusion Denoiser Training

```text
Input: clean state-latent token windows x0
Output: denoiser checkpoint
for minibatch x0:
    sample timestep k
    epsilon ~ Normal(0, I)
    x_k = sqrt(alpha_bar[k]) * x0 + sqrt(1 - alpha_bar[k]) * epsilon
    x0_hat = Denoiser(x_k, k)
    loss = MSE(x0_hat, x0)
    update denoiser
```

## Algorithm 8: Guided Diffusion Inference

```text
Input: current state, denoiser, task cost C(tau), VAE decoder
Output: current action for receding-horizon control
initialize noisy future trajectory tau_K
for k from K to 1:
    tau_hat = denoiser(tau_k, k)
    cost = C(tau_hat)
    grad = d cost / d tau_hat
    tau_{k-1} = reverse_step(tau_k, tau_hat) - guidance_scale * grad
take current latent z_t from tau_0
action_t = VAE_decoder(current_proprioception, z_t)
execute action_t in physics
```

## Algorithm 9: MuJoCo / Isaac Video Rendering

```text
Input: robot model, action or qpos sequence, camera config, metrics config
Output: MP4, keyframes, metrics CSV
load G1 MJCF/USD/URDF-derived model
map actions to robot joint order
if action-control:
    theta_sp = theta_default + action_scale * clip(action)
    for each frame: set actuator ctrl, step MuJoCo, render RGB
if reference replay:
    write qpos for visualization and call mj_forward
save frames to MP4 and compute root/action/fall/error metrics
```

## Algorithm 10: Failure Diagnosis Checklist

```text
Check data: joint order, FPS, root frame, ground contact, impossible segments.
Check teacher: reward, done rate, body/joint error, termination breakdown.
Check VAE: reconstruction MSE, KL, closed-loop rollout, DAgger coverage.
Check diffusion: token scaling, inverse transform, physically valid trajectory.
Check guidance: cost frame, gradient target, guidance scale.
Check deployment: action scale, PD gains, default pose, control frequency, observation normalization.
```
"""


def write_pseudocode() -> None:
    text = pseudocode_text()
    write_text(REPORT / "pseudocode.md", text)
    write_text(REPORT / "code_review" / "pseudocode_all_stages.md", text)


def paper_project_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "module": "Data collection",
            "paper_requirement": "About 2.5h diverse motions from prior work, LAFAN1, online animation.",
            "current_project_status": "PARTIAL",
            "evidence_path": "res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle.json",
            "current_result": (
                f"{get_nested(metrics['motion'], 'metrics', 'motion_count')} motions / "
                f"{fmt_float(motion_hours(metrics), 3)} h"
            ),
            "gap": "Not exact paper curated set; some sources skipped or unavailable.",
            "next_action": "Audit missing Sidekick/Ronaldo/online animation mappings before training claims.",
            "status": "PARTIAL",
        },
        {
            "module": "Motion preprocessing",
            "paper_requirement": "Retargeted generalized coordinates with FK body states.",
            "current_project_status": "Implemented for local bundle",
            "evidence_path": "res/tracking/stage1_multisource_motion_bundle/",
            "current_result": "G1 robot-order FK repaired bundle exists.",
            "gap": "Needs per-source visual QA and impossible-motion filtering.",
            "next_action": "Run per-motion stability/replay filters.",
            "status": "PARTIAL",
        },
        {
            "module": "PPO teacher",
            "paper_requirement": "Robust tracking policies for diverse motions.",
            "current_project_status": "Training completed but weak",
            "evidence_path": "res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/",
            "current_result": f"reward {get_nested(metrics['sweep'], 'metrics', 'best_reward_mean')}, body err {get_nested(metrics['sweep'], 'metrics', 'best_error_body_pos_mean')}",
            "gap": "Does not yet achieve stable paper-like tracking.",
            "next_action": "Debug reward/termination/action-scale/reset and retrain.",
            "status": "FAILED",
        },
        {
            "module": "Teacher rollout",
            "paper_requirement": "Reliable teacher state-action trajectories.",
            "current_project_status": "Collected from weak teacher",
            "evidence_path": "res/tracking/stage1_multisource_best_teacher_rollout_dataset/",
            "current_result": f"{get_nested(metrics['rollout'], 'aggregate_metrics', 'total_env_steps')} samples",
            "gap": "Not official DAgger data; many done/failure signals.",
            "next_action": "Recollect after teacher quality improves.",
            "status": "PARTIAL",
        },
        {
            "module": "Conditional VAE",
            "paper_requirement": "DAgger-trained latent action policy.",
            "current_project_status": "Offline VAE trained",
            "evidence_path": "res/level_c/stage1_multisource_teacher_rollout_vae_training/",
            "current_result": f"test action MSE {get_nested(metrics['vae'], 'worker_summary', 'evaluation', 'test', 'action_mse')}",
            "gap": "Full DAgger closed-loop not reproduced.",
            "next_action": "Implement student rollout + teacher query loop after teacher repair.",
            "status": "PARTIAL",
        },
        {
            "module": "State-latent diffusion",
            "paper_requirement": "State-latent trajectory diffusion model.",
            "current_project_status": "Local denoiser trained",
            "evidence_path": "res/level_c/stage1_multisource_state_latent_diffusion_training/",
            "current_result": f"pred MSE {get_nested(metrics['diffusion'], 'worker_summary', 'evaluation', 'test', 'pred_token_mse')}",
            "gap": "Not official Transformer/checkpoint and not stable controller.",
            "next_action": "Train after high-quality rollouts; verify inverse transforms.",
            "status": "PARTIAL",
        },
        {
            "module": "Guidance",
            "paper_requirement": "Joystick, waypoint, inpainting, obstacle avoidance via classifier guidance.",
            "current_project_status": "Offline proxy guidance",
            "evidence_path": "res/level_c/stage1_multisource_state_latent_guidance_eval/",
            "current_result": f"{get_nested(metrics['guidance'], 'worker_summary', 'metrics', 'total_selected_windows')} windows",
            "gap": "Closed-loop task success videos/metrics missing for current chain.",
            "next_action": "Receding-horizon MuJoCo closed-loop after teacher/VAE repair.",
            "status": "PARTIAL",
        },
        {
            "module": "MuJoCo/Isaac rendering",
            "paper_requirement": "High-quality simulation and real robot videos.",
            "current_project_status": "MuJoCo diagnostics; Isaac H20 rendering blocked",
            "evidence_path": "res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/",
            "current_result": "6 continuous diagnostic videos, unstable motion",
            "gap": "Not true Isaac MP4, not stable control, not real robot.",
            "next_action": "Fix control quality, use RTX for true Isaac rendered MP4 if needed.",
            "status": "FAILED",
        },
    ]


def write_paper_alignment(metrics: dict[str, Any]) -> None:
    rows = paper_project_rows(metrics)
    write_csv(REPORT / "tables" / "paper_project_comparison.csv", rows)
    write_csv(REPORT / "data" / "paper_project_comparison.csv", rows)
    lines = ["# Paper vs Project Alignment", "", "| module | status | paper requirement | current result | evidence | gap | next action |", "|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['module']} | {row['status']} | {row['paper_requirement']} | {row['current_result']} | "
            f"`{row['evidence_path']}` | {row['gap']} | {row['next_action']} |"
        )
    text = "\n".join(lines)
    write_text(REPORT / "paper_vs_project.md", text)
    write_text(REPORT / "paper_alignment.md", text)


def write_data_report(metrics: dict[str, Any]) -> None:
    m = metrics["motion"].get("metrics", {})
    checks = metrics["motion"].get("checks", {})
    inputs = metrics["motion"].get("inputs", {})
    text = f"""# Data Sources and Processing Report

## Summary

The latest 5/6-GPU Stage 1 run used a local multi-source motion bundle recorded in:

`/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle.json`

Current bundle metrics:

- Motion count: `{m.get('motion_count')}`
- Total frames: `{m.get('total_motion_frames')}`
- Total duration: `{m.get('total_duration_hours', m.get('total_motion_hours'))}` h
- Source counts: `{json.dumps(m.get('source_counts', {}), ensure_ascii=False)}`
- Checks: `{json.dumps(checks, ensure_ascii=False, sort_keys=True)}`

## Important Boundary

The paper describes about 2.5 hours of diverse motions from prior works, Unitree-retargeted LAFAN1, and online animation data. The current project should not claim it has exactly reconstructed the authors' private curated collection. What exists locally is:

- G1-retargeted LAFAN1 CSVs at `{inputs.get('lafan_root')}`;
- one train-ready BeyondMimic Zenodo reference CSV: `{inputs.get('zenodo_tkd_csv')}`;
- several HuB supplemental 29-DoF pkl candidates;
- PBHC sidekick / ASAP Ronaldo-like sources that were skipped because their 23-DoF mapping is not audited;
- Zenodo GRF/IMU/MCAP/ablation data, which is mainly released-result evidence rather than the full diffusion training dataset.

## Processing Flow

```text
raw BVH / downloaded CSV / pkl candidates
    -> audited G1 generalized coordinates
    -> FK-repaired body positions and velocities
    -> stage1_multisource_public_plus_available_motion_bundle_fk_repaired_robot_order.npz
    -> IsaacLab motion-tracking teacher training
    -> teacher rollout shards
    -> VAE and state-latent diffusion training data
```

## Generated Tables

- `report/tables/dataset_inventory.csv`
- `report/data/motion_duration_summary.csv`
- `report/data/motion_file_manifest.csv`
"""
    write_text(REPORT / "data_report.md", text)
    write_text(REPORT / "data" / "dataset_inventory.md", text)
    write_text(REPORT / "data" / "data_processing_flow.md", text)


def write_experiment_results(metrics: dict[str, Any], denoise: dict[str, float]) -> None:
    text = f"""# Experiment Results

## Stage 1 Multi-Source Teacher

- Training run: `{collect_artifacts()['stage1_training']}`
- Checkpoint sweep: `{collect_artifacts()['stage1_sweep']}`
- Best checkpoint: `{get_nested(metrics['sweep'], 'metrics', 'best_checkpoint')}`
- Best iteration: `{get_nested(metrics['sweep'], 'metrics', 'best_iteration')}`
- Best reward mean: `{get_nested(metrics['sweep'], 'metrics', 'best_reward_mean')}`
- Best body-position error mean: `{get_nested(metrics['sweep'], 'metrics', 'best_error_body_pos_mean')}`
- Best joint-position error mean: `{get_nested(metrics['sweep'], 'metrics', 'best_error_joint_pos_mean')}`

Interpretation: the 5/6 training completed and the checkpoint sweep is real, but the best teacher is still weak.

## Teacher Rollout and VAE

- Teacher rollout samples: `{get_nested(metrics['rollout'], 'aggregate_metrics', 'total_env_steps')}`
- Rollout done count: `{get_nested(metrics['rollout'], 'aggregate_metrics', 'done_count_total')}`
- VAE test action MSE: `{get_nested(metrics['vae'], 'worker_summary', 'evaluation', 'test', 'action_mse')}`
- VAE test absolute action error mean: `{get_nested(metrics['vae'], 'worker_summary', 'evaluation', 'test', 'action_abs_error_mean')}`

## State-Latent Diffusion

The denoiser reduces token prediction error from `{denoise['noisy']:.6f}` to `{denoise['pred']:.6f}`, corresponding to approximately `{denoise['improvement'] * 100:.1f}%` relative denoising improvement.

This indicates that the diffusion model has learned a non-trivial denoising mapping at the token level. However, token-level MSE improvement does not imply closed-loop humanoid control success. The current videos still show unstable or incomplete motion, so the diffusion model is not yet a successful BeyondMimic controller.

Figures:

- `report/figures/denoising_mse_improvement.png`
- `report/figures/metric_plots/stage1_checkpoint_sweep.png`

## Guidance

- Guidance status: `{metrics['guidance'].get('status')}`
- Selected windows: `{get_nested(metrics['guidance'], 'worker_summary', 'metrics', 'total_selected_windows')}`
- Tasks with all best costs improved: `{get_nested(metrics['guidance'], 'worker_summary', 'metrics', 'tasks_with_all_best_costs_improve')}`
- Tasks with nonzero best gradients: `{get_nested(metrics['guidance'], 'worker_summary', 'metrics', 'tasks_with_nonzero_best_gradients')}`

This is offline guidance evidence only, not paper-level Fig. 5/Fig. 6 closed-loop success.
"""
    write_text(REPORT / "experiment_results.md", text)
    write_text(REPORT / "experiments" / "mse_denoising_analysis.md", text)
    write_text(REPORT / "experiments" / "tracking_analysis.md", text)
    write_text(REPORT / "experiments" / "vae_analysis.md", text)
    write_text(REPORT / "experiments" / "diffusion_analysis.md", text)
    write_text(REPORT / "experiments" / "guidance_analysis.md", text)
    write_text(REPORT / "experiments" / "experiment_inventory.md", text)


def write_failure_analysis(metrics: dict[str, Any]) -> None:
    video = metrics["videos"]
    text = f"""# Failure Analysis

Current videos cannot be described as successful BeyondMimic reproduction. The latest six MuJoCo videos are continuous and generated through simulation stepping, but they are diagnostics. They still show poor motion quality, high fall proxies, and MuJoCo instability warnings such as QACC warnings.

## Evidence

- Video suite: `{collect_artifacts()['stage1_videos']}`
- Checks: `{json.dumps(video.get('checks', {}), sort_keys=True)}`
- Selected segment: `{json.dumps(video.get('selected_continuous_segment', {}), ensure_ascii=False)[:1200]}`
- MuJoCo warning log: `logs/mujoco/MUJOCO_LOG_stage1_multisource_continuous_rerun_20260623_143500.txt`

## Data Layer

Evidence: the current bundle includes `{get_nested(metrics['motion'], 'metrics', 'motion_count')}` motions and explicitly records skipped sources. It does not silently pad PBHC/ASAP 23-DoF sources into 29-DoF G1 actions.

Possible causes:

- Some motions may be kinematically valid but dynamically difficult.
- Root frame / ground height may still differ from the training model.
- The exact paper curated 2.5h dataset is not fully reconstructed.

How to verify:

- Replay each source motion with FK and ground-contact checks.
- Plot ankle/wrist/root heights for each motion.
- Reject impossible clips before PPO training.

## Teacher Policy Layer

Evidence: best 5/6 checkpoint reward mean is `{get_nested(metrics['sweep'], 'metrics', 'best_reward_mean')}`, body-position error mean is `{get_nested(metrics['sweep'], 'metrics', 'best_error_body_pos_mean')}`, and non-timeout done rate is `{get_nested(metrics['sweep'], 'metrics', 'best_local_non_timeout_done_rate')}`.

Likely impact: downstream VAE and diffusion imitate weak behavior. Better denoising cannot rescue a failed teacher distribution.

How to verify:

- Inspect reward component breakdown and termination reason distribution.
- Compare action scale / PD targets / reset pose against official whole_body_tracking config.
- Run smaller single-motion training until the teacher visibly tracks before multi-source training.

## VAE Layer

Evidence: VAE test action MSE is `{get_nested(metrics['vae'], 'worker_summary', 'evaluation', 'test', 'action_mse')}`. This is good offline reconstruction but not a proof of closed-loop VAE stability.

Risks:

- Offline reconstruction may match bad teacher actions.
- KL collapse / latent degeneracy may still occur.
- Full DAgger is not reproduced from official logs.

## Diffusion Layer

Evidence: test pred token MSE improves from `{get_nested(metrics['diffusion'], 'worker_summary', 'evaluation', 'test', 'noisy_token_mse')}` to `{get_nested(metrics['diffusion'], 'worker_summary', 'evaluation', 'test', 'pred_token_mse')}`.

Risks:

- Token MSE does not enforce physically executable trajectory.
- Inverse transform / scaling / horizon alignment may be wrong.
- The denoiser is trained from weak-teacher trajectories.

## Guidance Layer

Evidence: offline task-cost gradients are nonzero and best costs improve for proxy tasks, but no current paper-level receding-horizon closed-loop task success is available for this chain.

Risks:

- Guidance may act on token variables that do not map cleanly to stable actions.
- Guidance scale can break the learned distribution.
- Current implementation is not the paper TensorRT/asynchronous deployment path.

## MuJoCo / Isaac Deployment Layer

Risks to audit:

| risk | current evidence | how to test | priority |
|---|---|---|---|
| joint order mismatch | action-scale and G1 mapping audits exist but video instability remains | one-joint impulse test and compare visual joint | high |
| action scale mismatch | theta_sp formula is recorded in video summaries | sweep action scale while holding reference action | high |
| default pose mismatch | controller default position is imported from motion_tracking_controller config | compare default pose in Isaac/MuJoCo | high |
| PD gain mismatch | local MuJoCo uses position actuators/root assist | audit gain/armature against official controller | high |
| root frame mismatch | reference/video recenters root XY for display | compare root transform convention with training obs | high |
| observation normalization mismatch | PPO obs adapter is local | log obs mean/std and compare training normalization | high |
| diffusion inverse transform mismatch | local token windows are custom | decode known training windows and compare action reconstruction | medium |

## Priority Fix

Do not optimize the report videos first. The first fix is to make a single-motion Stage 1 teacher reliably track in physics, then recollect rollout data and retrain VAE/diffusion. Longer videos will only make weak-control failure more visible.
"""
    write_text(REPORT / "failure_analysis.md", text)
    write_text(REPORT / "logs" / "failure_logs" / "failure_analysis.md", text)


def write_next_steps() -> None:
    text = """# Next Steps

## Highest Priority

1. Single-motion teacher sanity retraining: pick one clean LAFAN1 walking/running motion and make PPO visibly track it before multi-source training.
2. Reward/termination audit: dump reward components, done causes, body/joint errors, and reset phases for the weak 5/6 checkpoint.
3. MuJoCo/Isaac action contract audit: verify joint order, action scale, default pose, PD gains, armature, and control frequency with one-joint tests.

## After Teacher Repair

1. Recollect teacher rollout shards from stable policy.
2. Retrain VAE and check closed-loop VAE rollout.
3. Rebuild state-latent dataset and retrain denoiser.
4. Implement receding-horizon guided MuJoCo closed-loop tasks.
5. Run RTX Isaac rendered MP4 if true Isaac visuals are still needed.

## Files Most Likely To Inspect Next

- `/mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py`
- `/mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_pd_control_video.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_stage1_multisource_paper_contract_ppo_checkpoint_eval.py`
"""
    write_text(REPORT / "next_steps.md", text)
    write_text(REPORT / "limitations_and_next_steps.md", text)


def write_appendix(metrics: dict[str, Any]) -> None:
    env = f"""# Environment Summary

- ROOT: `{ROOT}`
- Report generated at: `{now_utc()}`
- bm_analysis: `{path_status('envs/bm_analysis/bin/python3.10')}`
- bm_diffusion: `{path_status('envs/bm_diffusion/bin/python3.10')}`
- bm_tracking: `{path_status('envs/bm_tracking/bin/python3.10')}`
- MuJoCo venv: `{path_status('mujoco_mp4/.venv/bin/python')}`
- IsaacLab directory: `{path_status('download/dependencies/IsaacLab-v2.1.0')}`
- Isaac rendering status: H20 true Isaac rendered MP4 remains blocked by Kit/Hydra/Vulkan rendering stack.
"""
    write_text(REPORT / "appendix" / "environment_summary.md", env)
    equations = r"""# Equations and Implementation Notes

## Action to PD target

$$
\theta^{sp} = \theta^0 + \alpha \odot a
$$

MuJoCo diagnostic controller:

$$
\tau \approx K_p(\theta^{sp} - \theta) - K_d \dot{\theta}
$$

## VAE objective

$$
\mathcal{L}_{VAE} = \|a - D(o,z)\|_2^2 + \beta D_{KL}(q_\phi(z|o,a)\|N(0,I))
$$

## Diffusion noising

$$
x_k = \sqrt{\bar{\alpha}_k}x_0 + \sqrt{1-\bar{\alpha}_k}\epsilon
$$

## Denoising loss

$$
\mathcal{L}_{diff} = \|\hat{x}_0 - x_0\|_2^2
$$

## Guidance

At test time, task cost gradients modify reverse diffusion samples:

$$
x \leftarrow x - \lambda \nabla_x C(x)
$$

This report treats current guidance as offline/local proxy evidence unless a physical closed-loop rollout exists.
"""
    write_text(REPORT / "appendix" / "equations.md", equations)
    commands = f"""# Command History and Reproduction Commands

The following commands are reconstructed from current project files and this report-generation round. Items marked inferred should be treated as commands-to-run, not guaranteed original shell history.

## Stage1 multi-source checkpoint sweep

```bash
CUDA_VISIBLE_DEVICES=5,6 BM_STAGE1_MULTISOURCE_CHECKPOINT_SWEEP_TARGET_GPUS=5,6 BM_STAGE1_MULTISOURCE_CHECKPOINT_SWEEP_NUM_ENVS=256 BM_STAGE1_MULTISOURCE_CHECKPOINT_SWEEP_EVAL_STEPS=299 BM_STAGE1_MULTISOURCE_CHECKPOINT_SWEEP_STRIDE=2500 python3 reproduction/scripts/tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.py
```

## Teacher rollout

```bash
CUDA_VISIBLE_DEVICES=5,6 BM_STAGE1_MULTISOURCE_TEACHER_ROLLOUT_NUM_ENVS_PER_RANK=1024 BM_TEACHER_ROLLOUT_STEPS=299 python3 reproduction/scripts/tracking_stage1_multisource_best_teacher_rollout_dataset.py
```

## VAE / diffusion / guidance

```bash
CUDA_VISIBLE_DEVICES=5,6 python3 reproduction/scripts/level_c_stage1_multisource_teacher_rollout_vae_training.py
CUDA_VISIBLE_DEVICES=5,6 python3 reproduction/scripts/level_c_stage1_multisource_teacher_rollout_state_latent_dataset.py
CUDA_VISIBLE_DEVICES=5,6 python3 reproduction/scripts/level_c_stage1_multisource_state_latent_diffusion_training.py
CUDA_VISIBLE_DEVICES=5,6 python3 reproduction/scripts/level_c_stage1_multisource_state_latent_guidance_eval.py
```

## MuJoCo video suite

```bash
MUJOCO_GL=egl mujoco_mp4/.venv/bin/python reproduction/scripts/stage1_multisource_continuous_mujoco_action_control_videos.py
```
"""
    write_text(REPORT / "appendix" / "command_history.md", commands)
    write_text(REPORT / "appendix" / "references.md", """# References

- BeyondMimic paper PDF: `/mnt/infini-data/test/BeyondMimic/download/papers/BeyondMimic_2508.08241.pdf`
- Paper source tar: `/mnt/infini-data/test/BeyondMimic/download/papers/BeyondMimic_2508.08241_source.tar`
- Official motion tracking code: `/mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking`
- Official controller code: `/mnt/infini-data/test/BeyondMimic/download/official/motion_tracking_controller`
- Zenodo released data copy: `/mnt/infini-data/test/BeyondMimic/Dataset_beyondmimic`
""")
    write_text(REPORT / "appendix" / "unresolved_details.md", """# Unresolved Details

- Exact private curated 2.5h paper motion set is not fully reconstructed.
- Official BeyondMimic VAE/diffusion checkpoints are not available.
- True DAgger logs are not available.
- H20 server does not produce true Isaac rendered MP4 due rendering stack blocker.
- Current teacher quality is weak, so downstream videos remain diagnostic.
""")


def write_module_pipeline(metrics: dict[str, Any]) -> None:
    text = """# Module Pipeline

## Module 0: Project code base

- Official Stage 1 code: `download/official/whole_body_tracking`
- Official deployment/controller reference: `download/official/motion_tracking_controller`
- IsaacLab: `download/dependencies/IsaacLab-v2.1.0`
- Local reproduction scripts: `reproduction/scripts`
- MuJoCo package: `mujoco_mp4`
- Report-ready official released-data replays: `official_mp4`

## Module 1: Data

G1-retargeted LAFAN1, one train-ready Zenodo tkd CSV, and HuB candidates were converted into the current 49-motion / 2.49h local bundle.

## Module 2: Stage1 teacher

PPO training completed on GPUs 5/6, but current reward/error metrics show a weak teacher.

## Module 3: Teacher rollout

The selected weak teacher generated local rollout shards used by VAE/diffusion.

## Module 4: Conditional VAE

Offline action reconstruction works, but true DAgger/closed-loop VAE success is not proven.

## Module 5: State-latent diffusion

Token denoising improves by about 40.6%, but physical control remains unstable.

## Module 6: Guidance

Offline guidance proxy works over selected windows. Paper-level closed-loop joystick/waypoint/inpainting/obstacle success is not reproduced.

## Module 7: MuJoCo/Isaac rendering

MuJoCo videos are generated; true Isaac rendered MP4 on H20 is blocked by rendering stack.
"""
    write_text(REPORT / "module_pipeline.md", text)
    write_text(REPORT / "pipeline" / "full_pipeline_mermaid.md", """# Full Pipeline Mermaid Source

```mermaid
flowchart TD
    A[Data sources] --> B[Motion preprocessing]
    B --> C[PPO motion tracking teacher]
    C --> D[Teacher rollout state-action data]
    D --> E[Conditional VAE]
    E --> F[State-latent trajectory dataset]
    F --> G[Diffusion denoiser]
    G --> H[Test-time guidance]
    H --> I[MuJoCo / Isaac video]
    I --> J[Metrics and failure analysis]
```
""")


def write_report_main(metrics: dict[str, Any], denoise: dict[str, float]) -> None:
    artifacts = collect_artifacts()
    rows = module_status_rows(metrics)
    status_lines = "\n".join(
        f"- **{row['module']}**: `{row['status']}`; {row['current_result']}; evidence `{row['evidence_path']}`."
        for row in rows
    )
    text = f"""# BeyondMimic Reproduction Technical Report

Generated at: `{now_utc()}`

## 0. Executive Summary

This project currently has a substantial, auditable reproduction codebase for BeyondMimic, but it does **not** fully reproduce BeyondMimic at paper level. The latest GPUs 5/6 multi-source Stage 1 teacher training completed and downstream VAE/state-latent/diffusion/guidance artifacts were generated. However, the teacher remains weak, and the newest MuJoCo action-control videos still do not show stable paper-quality humanoid motion.

Most important current finding: the diffusion denoiser reduces token MSE from `{denoise['noisy']:.6f}` to `{denoise['pred']:.6f}` (`{denoise['improvement'] * 100:.1f}%` improvement), but token-level denoising success does not imply closed-loop humanoid control success.

## 1. Paper Method Overview

BeyondMimic can be understood as a four-stage pipeline:

```text
Human motions
    -> RL motion tracking teacher policies
    -> DAgger / conditional VAE latent action policy
    -> state-latent diffusion model
    -> test-time guidance for new tasks
```

The key control equation used by the local MuJoCo diagnostics is:

$$\\theta^{{sp}} = \\theta^0 + \\alpha \\odot a$$

where the policy or VAE decoder produces action `a`, `alpha` is action scale, and the simulator executes a PD-like position target. The project currently has local approximations for this path, but official BeyondMimic VAE/diffusion checkpoints and real robot logs are not public.

## 2. Current Project Inventory

- Paper PDF: `{artifacts['paper_pdf']}` ({path_status(artifacts['paper_pdf'])})
- Paper source tar: `{artifacts['paper_source_tar']}` ({path_status(artifacts['paper_source_tar'])})
- Official Stage 1 code: `{artifacts['whole_body_tracking']}` ({path_status(artifacts['whole_body_tracking'])})
- Official controller reference: `{artifacts['motion_tracking_controller']}` ({path_status(artifacts['motion_tracking_controller'])})
- IsaacLab: `{artifacts['isaaclab']}` ({path_status(artifacts['isaaclab'])})
- MuJoCo experiment package: `{artifacts['mujoco_package']}` ({path_status(artifacts['mujoco_package'])})
- Report file inventory: `report/file_inventory.txt`

## 3. Data Sources and Preprocessing

The latest local Stage 1 bundle contains `{get_nested(metrics['motion'], 'metrics', 'motion_count')}` motions and `{fmt_float(motion_hours(metrics), 3)}` hours. Source counts are:

```json
{json.dumps(get_nested(metrics['motion'], 'metrics', 'source_counts', default={}), indent=2, ensure_ascii=False)}
```

This is close in duration to the paper's reported 2.5h, but it is **not** guaranteed to be the authors' exact private curated set. See `report/data_report.md` and `report/tables/dataset_inventory.csv`.

## 4. Module 1: Motion Tracking Teacher

Input: processed reference motion bundle, Unitree G1 model, IsaacLab task, reward/termination/PPO config.

Processing: PPO policy maps observation to 29-D action, action becomes PD target, environment returns tracking rewards and done signals.

Current result: checkpoint sweep selected iteration `{get_nested(metrics['sweep'], 'metrics', 'best_iteration')}` with reward mean `{get_nested(metrics['sweep'], 'metrics', 'best_reward_mean')}`, body-position error mean `{get_nested(metrics['sweep'], 'metrics', 'best_error_body_pos_mean')}`, and joint-position error mean `{get_nested(metrics['sweep'], 'metrics', 'best_error_joint_pos_mean')}`.

Gap to paper: the teacher is weak and cannot be treated as robust BeyondMimic motion tracking.

## 5. Module 2: Teacher Rollout

The selected teacher produced `{get_nested(metrics['rollout'], 'aggregate_metrics', 'total_env_steps')}` rollout samples across `{get_nested(metrics['rollout'], 'aggregate_metrics', 'shard_count')}` shards. Done count is `{get_nested(metrics['rollout'], 'aggregate_metrics', 'done_count_total')}`. These samples are useful local training data, but not official DAgger rollouts.

## 6. Module 3: Conditional VAE and DAgger

The local VAE uses teacher rollout obs/action pairs. Current test action MSE is `{get_nested(metrics['vae'], 'worker_summary', 'evaluation', 'test', 'action_mse')}`. This is offline reconstruction evidence, not full DAgger closed-loop reproduction.

Formula:

$$\\mathcal{{L}}_{{VAE}} = \\|a - D(o,z)\\|_2^2 + \\beta D_{{KL}}(q(z|o,a)\\|N(0,I))$$

## 7. Module 4: State-Latent Trajectory Diffusion

The local state-latent dataset contains `{get_nested(metrics['state'], 'worker_summary', 'dataset', 'window_count')}` windows with token dimension `{get_nested(metrics['state'], 'worker_summary', 'dataset', 'token_dim')}`.

Denoising result:

- Noisy token MSE: `{denoise['noisy']:.6f}`
- Test pred token MSE: `{denoise['pred']:.6f}`
- Relative improvement: `{denoise['improvement'] * 100:.1f}%`

This is a meaningful token-level result, but it does not prove physically stable humanoid control.

## 8. Module 5: Test-Time Guidance

Current guidance status: `{metrics['guidance'].get('status')}`. The local run evaluates `{get_nested(metrics['guidance'], 'worker_summary', 'metrics', 'total_selected_windows')}` windows and records nonzero gradients / improving best costs for proxy tasks. This is offline evidence, not paper Fig. 5/Fig. 6 closed-loop evaluation.

## 9. Module 6: MuJoCo / Isaac Rendering

H20 true Isaac rendered MP4 remains blocked by the Isaac Sim Kit/Hydra/Vulkan rendering stack. MuJoCo rendering works and generated six continuous videos under:

`{get_nested(metrics['videos'], 'output_root')}`

These videos use continuous motion-time steps and no reset stitching, but they remain failure/diagnostic videos because the current teacher and action-control chain are unstable.

## 10. Current Quantitative Results

{status_lines}

Detailed metrics are in:

- `report/tables/metrics_summary.csv`
- `report/experiment_results.md`
- `report/figures/denoising_mse_improvement.png`

## 11. Current Qualitative Video Results

Video index: `report/videos/video_index.md`

Failure montage: `report/figures/failure_montage.png`

The latest six videos are:

```json
{json.dumps(list(get_nested(metrics['videos'], 'videos', default={}).keys()), indent=2)}
```

## 12. Failure Analysis

The most likely high-level cause is not one isolated rendering bug. The current teacher is weak, so VAE/diffusion/guidance inherit weak action distributions. Deployment mismatch may amplify the weakness. See `report/failure_analysis.md`.

## 13. Paper-vs-Project Alignment

See `report/paper_vs_project.md` and `report/tables/paper_project_comparison.csv`.

Summary: data/preprocessing and local downstream code are partially reproduced; paper-level teacher quality, official DAgger, official VAE/diffusion checkpoints, TensorRT/asynchronous deployment, Fig. 5/Fig. 6 videos, and real robot results are not reproduced.

## 14. Next Debugging Priorities

1. Fix Stage 1 teacher on a single clean motion before more multi-source training.
2. Audit reward/termination/reset/action-scale/PD gain contract.
3. Recollect rollout data only after the teacher has stable closed-loop tracking.
4. Retrain VAE/diffusion and rerun receding-horizon MuJoCo videos.

## 15. Links to Code Snippets and Pseudocode

- `report/code_snippets.md`
- `report/pseudocode.md`
- `report/code_review/key_code_index.md`
- `report/appendix/equations.md`

## Non-Claim Boundary

This project does not fully reproduce BeyondMimic at paper-level. Current MP4s are local MuJoCo virtual diagnostics, not real robot results and not official Isaac rendered paper videos.
"""
    write_text(REPORT / "report_main.md", text)
    write_text(REPORT / "executive_summary.md", text.split("## 1. Paper Method Overview")[0])


def write_readme(pdf_status: str) -> None:
    text = f"""# BeyondMimic Report Package

Generated at: `{now_utc()}`

Main files:

- `report_main.md`: detailed technical report
- `report_main.html`: HTML export
- `module_pipeline.md`: module-by-module flow
- `data_report.md`: data provenance and 2.5h-motion boundary
- `code_snippets.md`: key code excerpts
- `pseudocode.md`: algorithms for each stage
- `experiment_results.md`: current metrics
- `failure_analysis.md`: why videos are still poor
- `next_steps.md`: recommended repair path

Figures:

- `figures/pipeline_overview.png`
- `figures/denoising_mse_improvement.png`
- `figures/failure_montage.png`

HTML/PDF:

- HTML: `report_main.html`
- PDF: {pdf_status}

Claim boundary: this report documents current local reproduction work. It does not claim complete paper-level BeyondMimic reproduction.
"""
    write_text(REPORT / "README.md", text)


def export_html_pdf() -> str:
    md = REPORT / "report_main.md"
    html_out = REPORT / "report_main.html"
    pdf_out = REPORT / "report_main.pdf"
    if shutil.which("pandoc"):
        code, _, err = run_capture(["pandoc", str(md), "-o", str(html_out), "--standalone"], timeout=120)
        if code != 0:
            write_text(html_out, f"<pre>{html.escape(md.read_text(encoding='utf-8'))}</pre>")
            write_text(REPORT / "report_main_html_reason.txt", err)
        code, _, err = run_capture(["pandoc", str(md), "-o", str(pdf_out)], timeout=180)
        if code == 0 and pdf_out.exists():
            return "`report_main.pdf` generated"
        reason = f"`report_main.pdf` not generated: {err.strip()[:1000]}"
        write_text(REPORT / "report_main_pdf_reason.txt", reason)
        return reason
    body = md.read_text(encoding="utf-8")
    write_text(html_out, f"<html><body><pre>{html.escape(body)}</pre></body></html>")
    reason = "pandoc not found; PDF skipped"
    write_text(REPORT / "report_main_pdf_reason.txt", reason)
    return reason


def write_logs_summary() -> None:
    logs = sorted([p for p in (ROOT / "logs").rglob("*") if p.is_file()])[:500] if (ROOT / "logs").exists() else []
    rows = [
        {
            "log_path": rel(p),
            "size_bytes": p.stat().st_size,
            "modified_time": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
        }
        for p in logs
    ]
    write_csv(REPORT / "logs_summary" / "log_inventory.csv", rows)
    write_csv(REPORT / "logs" / "log_inventory.csv", rows)
    lines = ["# Log Inventory", "", "| log | size bytes | modified |", "|---|---:|---|"]
    for row in rows[:200]:
        lines.append(f"| `{row['log_path']}` | {row['size_bytes']} | {row['modified_time']} |")
    write_text(REPORT / "logs_summary" / "log_inventory.md", "\n".join(lines))
    write_text(REPORT / "logs" / "log_inventory.md", "\n".join(lines))


def main() -> None:
    ensure_dirs()
    make_inventory_files()
    metrics = metric_bundle()
    write_dataset_tables(metrics)
    write_module_tables(metrics)
    denoise = write_metric_tables(metrics)
    draw_mse_chart(denoise)
    draw_checkpoint_chart(metrics)
    draw_training_curve(
        "res/level_c/stage1_multisource_teacher_rollout_vae_training/level_c_stage1_multisource_teacher_rollout_vae_training.tsv",
        "stage1_vae_training",
        ["train_loss", "validation_loss", "validation_action_mse", "test_action_mse"],
        "Stage1 multi-source VAE training",
    )
    draw_training_curve(
        "res/level_c/stage1_multisource_state_latent_diffusion_training/level_c_stage1_multisource_state_latent_diffusion_training.tsv",
        "stage1_diffusion_training",
        ["train_token_mse", "validation_pred_token_mse", "validation_noisy_token_mse"],
        "Stage1 multi-source diffusion training",
    )
    write_flow_diagrams()
    video_rows = write_video_index(metrics)
    write_code_reports()
    write_pseudocode()
    write_paper_alignment(metrics)
    write_data_report(metrics)
    write_experiment_results(metrics, denoise)
    write_failure_analysis(metrics)
    write_next_steps()
    write_module_pipeline(metrics)
    write_appendix(metrics)
    write_logs_summary()
    write_report_main(metrics, denoise)
    pdf_status = export_html_pdf()
    write_readme(pdf_status)
    summary = {
        "status": "ok",
        "generated_at": now_utc(),
        "report_root": str(REPORT),
        "main_report": str(REPORT / "report_main.md"),
        "html_report": str(REPORT / "report_main.html"),
        "pdf_status": pdf_status,
        "video_count_indexed": len(video_rows),
        "key_outputs": {
            "denoising_mse_png": str(REPORT / "figures" / "denoising_mse_improvement.png"),
            "failure_montage": str(REPORT / "figures" / "failure_montage.png"),
            "paper_project_csv": str(REPORT / "tables" / "paper_project_comparison.csv"),
            "code_snippets": str(REPORT / "code_snippets.md"),
        },
        "claim_boundary": "Current report documents local reproduction progress; it does not claim complete paper-level BeyondMimic reproduction.",
    }
    write_text(REPORT / "report_generation_summary.json", json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
