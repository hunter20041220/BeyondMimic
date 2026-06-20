#!/usr/bin/env python3
"""Run a local importer-export walk-to-run transition guidance proxy.

This script injects a single ``transition`` task into the validated
official-importer-export task-conditioned receding-latent guidance runner.  The
task uses a local walk-to-run velocity-ramp proxy cost and produces report
assets for Fig. 5B/5D-adjacent discussion.  It is local virtual evidence only:
not the paper transition protocol, not the paper t-SNE, and not an official
BeyondMimic checkpoint result.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SCRIPTS = ROOT / "reproduction/scripts"
BASE_SCRIPT = SCRIPTS / "tracking_g1_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval.py"
SUMMARY_ROOT = ROOT / "res/level_c/official_importer_export_full_bundle_transition_guidance_rollout_eval"
VIS_ROOT = ROOT / "res/visualization/official_importer_export_full_bundle_transition_guidance_rollout"
REPORT_ROOT = ROOT / "res/report_assets/official_importer_export_full_bundle_transition_guidance"
LOG_ROOT = ROOT / "logs/tracking_g1_official_importer_export_full_bundle_transition_guidance_rollout_eval"
FAILED_ROOT = ROOT / "res/failed_runs/tracking_g1_official_importer_export_full_bundle_transition_guidance_rollout_eval"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_importer_export_full_bundle_transition_guidance_rollout_eval"
SUMMARY_JSON = SUMMARY_ROOT / "level_c_official_importer_export_full_bundle_transition_guidance_rollout_eval.json"
SUMMARY_TSV = SUMMARY_ROOT / "level_c_official_importer_export_full_bundle_transition_guidance_rollout_eval.tsv"
UNDERLYING_JSON = SUMMARY_ROOT / "underlying_transition_task.json"
UNDERLYING_TSV = SUMMARY_ROOT / "underlying_transition_task.tsv"
REPORT_JSON = REPORT_ROOT / "transition_guidance_report_assets.json"
TASK = "transition"
SEED = 20260903
DT = 0.02


TRANSITION_TASK_COST_CODE = r'''
    def task_cost(tau, task_name):
        obs_part = tau[..., :160]
        latent_part = tau[..., 160:]
        root_xy = obs_part[..., :2]
        horizon = root_xy.shape[1]
        dtype = tau.dtype
        device = tau.device
        dt = 0.02
        path_xy = torch.cumsum(root_xy * dt, dim=1)
        t = torch.linspace(0.0, 1.0, horizon, dtype=dtype, device=device).view(1, horizon, 1)

        joystick_target = torch.tensor([0.35, 0.0], dtype=dtype, device=device).view(1, 1, 2)
        joystick = torch.mean((root_xy - joystick_target) ** 2)

        waypoint_target = torch.cat([0.35 * t, 0.12 * torch.sin(torch.pi * t)], dim=-1)
        waypoint = torch.mean((path_xy - waypoint_target) ** 2)

        transition_target = torch.cat([0.16 + 0.58 * t, torch.zeros_like(t)], dim=-1)
        transition_velocity = torch.mean((root_xy - transition_target) ** 2)
        transition_path_target = torch.cat(
            [(0.16 * t + 0.29 * t * t) * horizon * dt, torch.zeros_like(t)],
            dim=-1,
        )
        transition_path = torch.mean((path_xy - transition_path_target) ** 2)
        transition_smooth = torch.mean((root_xy[:, 1:, :] - root_xy[:, :-1, :]) ** 2)

        key_count = min(5, horizon)
        keyframe_indices = torch.linspace(0, horizon - 1, key_count, device=device).round().long()
        keyframe_t = t[:, keyframe_indices, :]
        keyframe_target = torch.cat(
            [0.30 * keyframe_t, 0.10 * torch.sin(2.0 * torch.pi * keyframe_t)],
            dim=-1,
        )
        keyframe_xy = path_xy[:, keyframe_indices, :]
        inpainting = torch.mean((keyframe_xy - keyframe_target) ** 2)

        obstacle_center = torch.tensor([0.18, 0.0], dtype=dtype, device=device).view(1, 1, 2)
        clearance = torch.linalg.vector_norm(path_xy - obstacle_center, dim=-1) - 0.18
        obstacle = torch.mean(torch.relu(0.04 - clearance) ** 2)

        latent_smooth = torch.mean((latent_part[:, 1:, :] - latent_part[:, :-1, :]) ** 2)
        latent_mag = torch.mean(latent_part**2)
        regularizer = 0.25 * latent_smooth + 0.1 * latent_mag

        if task_name == "joystick":
            return joystick + regularizer
        if task_name == "waypoint":
            return waypoint + 0.2 * joystick + regularizer
        if task_name == "obstacle_avoidance":
            return obstacle + 0.1 * joystick + regularizer
        if task_name == "composed":
            return joystick + 0.5 * waypoint + 2.0 * obstacle + regularizer
        if task_name == "inpainting":
            return inpainting + 0.1 * joystick + regularizer
        if task_name == "transition":
            return transition_velocity + 0.35 * transition_path + 0.15 * transition_smooth + regularizer
        raise ValueError(f"Unknown guidance task: {task_name}")

    def best_task_scale(path, task_name):
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            return float(summary["worker_summary"]["task_summaries"][task_name]["splits"]["validation"]["scale"])
        except Exception:
            return 0.01
'''


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "variant",
        "step_count",
        "x_progress_m",
        "mean_speed_mps",
        "first_third_speed_mps",
        "last_third_speed_mps",
        "late_minus_early_speed_mps",
        "speed_slope_mps_per_step",
        "target_speed_rmse_mps",
        "speed_target_corr",
        "lateral_abs_mean_m",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def import_base_module():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("bm_importer_transition_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_underlying() -> tuple[int, str]:
    os.environ.update(
        {
            "BM_TASK_CONDITIONED_TASKS": TASK,
            "BM_TASK_CONDITIONED_TASK_SEEDS_JSON": json.dumps({TASK: SEED}),
            "BM_TASK_CONDITIONED_OUT_ROOT": str(VIS_ROOT),
            "BM_TASK_CONDITIONED_SUMMARY_ROOT": str(SUMMARY_ROOT / "underlying_tasks"),
            "BM_TASK_CONDITIONED_SUMMARY_JSON": str(UNDERLYING_JSON),
            "BM_TASK_CONDITIONED_SUMMARY_TSV": str(UNDERLYING_TSV),
            "BM_TASK_CONDITIONED_LOG_ROOT": str(LOG_ROOT / "underlying_tasks"),
            "BM_TASK_CONDITIONED_FAILED_ROOT": str(FAILED_ROOT / "underlying_tasks"),
            "BM_TASK_CONDITIONED_RUN_ROOT": str(RUN_ROOT / "underlying_tasks"),
            "BM_RECEDING_LATENT_GUIDANCE_SCALE_MULT": "1.0",
        }
    )
    for path in [SUMMARY_ROOT, VIS_ROOT, REPORT_ROOT, LOG_ROOT, FAILED_ROOT, RUN_ROOT]:
        path.mkdir(parents=True, exist_ok=True)

    module = import_base_module()
    module.base_task.TASK_COST_CODE = TRANSITION_TASK_COST_CODE
    module.base_task.DEFAULT_TASK_SEEDS[TASK] = SEED
    module.base_task.TASK_SEEDS[TASK] = SEED
    try:
        module.main()
        return 0, ""
    except SystemExit as exc:
        code = int(exc.code or 0)
        return code, f"SystemExit({code})"
    except BaseException as exc:  # preserve failure in wrapper JSON
        return 1, repr(exc)


def target_speed(step_count: int) -> np.ndarray:
    return np.linspace(0.16, 0.74, step_count, dtype=np.float64)


def variant_transition_metrics(npz: np.lib.npyio.NpzFile, variant: str) -> dict[str, Any]:
    key = f"{variant}_robot_body_pos_w"
    if key not in npz:
        return {"variant": variant, "available": False, "reason": f"missing {key}"}
    body_pos = np.asarray(npz[key], dtype=np.float64)
    root_xy = body_pos[:, 0, :2]
    delta = np.diff(root_xy, axis=0, prepend=root_xy[0:1])
    speed = np.linalg.norm(delta, axis=-1) / DT
    target = target_speed(speed.shape[0])
    third = max(speed.shape[0] // 3, 1)
    x = np.arange(speed.shape[0], dtype=np.float64)
    slope = float(np.polyfit(x, speed, deg=1)[0]) if speed.shape[0] >= 2 else 0.0
    corr = float(np.corrcoef(speed, target)[0, 1]) if np.std(speed) > 1e-12 else 0.0
    return {
        "variant": variant,
        "available": True,
        "step_count": int(speed.shape[0]),
        "x_progress_m": float(root_xy[-1, 0] - root_xy[0, 0]),
        "y_progress_m": float(root_xy[-1, 1] - root_xy[0, 1]),
        "mean_speed_mps": float(speed.mean()),
        "first_third_speed_mps": float(speed[:third].mean()),
        "last_third_speed_mps": float(speed[-third:].mean()),
        "late_minus_early_speed_mps": float(speed[-third:].mean() - speed[:third].mean()),
        "speed_slope_mps_per_step": slope,
        "target_speed_rmse_mps": float(np.sqrt(np.mean((speed - target) ** 2))),
        "speed_target_corr": corr,
        "lateral_abs_mean_m": float(np.mean(np.abs(root_xy[:, 1] - root_xy[0, 1]))),
        "speed_profile": speed.tolist(),
        "target_speed_profile": target.tolist(),
        "root_xy": root_xy.tolist(),
    }


def plot_transition_profiles(metrics: dict[str, dict[str, Any]], assets: dict[str, str]) -> None:
    variants = ["teacher", "vae_base", "denoised_latent", "receding_latent_guided"]
    labels = {
        "teacher": "teacher",
        "vae_base": "VAE base",
        "denoised_latent": "denoised latent",
        "receding_latent_guided": "guided latent",
    }
    colors = {
        "teacher": "#059669",
        "vae_base": "#2563eb",
        "denoised_latent": "#9333ea",
        "receding_latent_guided": "#dc2626",
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    target = None
    for variant in variants:
        row = metrics.get(variant, {})
        if not row.get("available"):
            continue
        speed = np.asarray(row["speed_profile"], dtype=np.float64)
        target = np.asarray(row["target_speed_profile"], dtype=np.float64)
        ax.plot(speed, label=labels[variant], color=colors[variant], linewidth=1.8)
    if target is not None:
        ax.plot(target, label="local transition target", color="#111827", linestyle="--", linewidth=1.5)
    ax.set_title("Official-importer local walk-to-run transition proxy speed profile")
    ax.set_xlabel("step")
    ax.set_ylabel("root speed proxy (m/s)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    speed_png = REPORT_ROOT / "transition_speed_profile.png"
    fig.savefig(speed_png, dpi=180)
    plt.close(fig)
    assets["speed_profile_png"] = str(speed_png)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    for variant in variants:
        row = metrics.get(variant, {})
        if not row.get("available"):
            continue
        root_xy = np.asarray(row["root_xy"], dtype=np.float64)
        ax.plot(root_xy[:, 0] - root_xy[0, 0], root_xy[:, 1] - root_xy[0, 1], label=labels[variant], color=colors[variant])
    ax.set_title("Official-importer local transition proxy root XY path")
    ax.set_xlabel("relative x (m)")
    ax.set_ylabel("relative y (m)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    path_png = REPORT_ROOT / "transition_root_path.png"
    fig.savefig(path_png, dpi=180)
    plt.close(fig)
    assets["root_path_png"] = str(path_png)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.6), constrained_layout=True)
    names = [variant for variant in variants if metrics.get(variant, {}).get("available")]
    axes[0].bar(
        [labels[name] for name in names],
        [metrics[name]["late_minus_early_speed_mps"] for name in names],
        color=[colors[name] for name in names],
    )
    axes[0].set_ylabel("late - early speed (m/s)")
    axes[0].set_title("Transition acceleration proxy")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(
        [labels[name] for name in names],
        [metrics[name]["target_speed_rmse_mps"] for name in names],
        color=[colors[name] for name in names],
    )
    axes[1].set_ylabel("target speed RMSE (m/s)")
    axes[1].set_title("Ramp-tracking proxy")
    axes[1].tick_params(axis="x", rotation=25)
    bars_png = REPORT_ROOT / "transition_metric_bars.png"
    fig.savefig(bars_png, dpi=180)
    plt.close(fig)
    assets["metric_bars_png"] = str(bars_png)


def write_report_readme(metrics: dict[str, dict[str, Any]], assets: dict[str, str]) -> None:
    guided = metrics.get("receding_latent_guided", {})
    readme = REPORT_ROOT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Official-Importer Transition Guidance Proxy",
                "",
                "This folder contains report assets for a local walk-to-run transition guidance proxy.",
                "The run uses the recovered official-importer-export G1 USDA path and local PPO/VAE/denoiser checkpoints.",
                "",
                "It is not the paper Fig. 5B transition protocol, not the paper Fig. 5D t-SNE panel, not an official checkpoint result, and not real-robot evidence.",
                "",
                "## Guided Variant Metrics",
                "",
                f"- Steps: `{guided.get('step_count')}`",
                f"- Late minus early speed: `{guided.get('late_minus_early_speed_mps')}`",
                f"- Target speed RMSE: `{guided.get('target_speed_rmse_mps')}`",
                f"- Speed-target correlation: `{guided.get('speed_target_corr')}`",
                f"- X progress: `{guided.get('x_progress_m')}`",
                "",
                "## Assets",
                "",
                *[f"- `{path}`" for path in assets.values()],
                "",
            ]
        ),
        encoding="utf-8",
    )
    assets["readme"] = str(readme)


def build_summary(underlying_returncode: int, underlying_error: str) -> dict[str, Any]:
    underlying = load_json(UNDERLYING_JSON)
    rows = underlying.get("rows", [])
    row = rows[0] if rows else {}
    task_summary = load_json(Path(row.get("summary_json", ""))) if row else {}
    capture_npz = Path(task_summary.get("outputs", {}).get("capture_npz", ""))
    assets = dict(task_summary.get("outputs", {}).get("assets", {}))
    report_assets: dict[str, str] = {}
    transition_metrics: dict[str, dict[str, Any]] = {}
    if capture_npz.is_file():
        with np.load(capture_npz) as npz:
            for variant in ["teacher", "vae_base", "denoised_latent", "receding_latent_guided"]:
                transition_metrics[variant] = variant_transition_metrics(npz, variant)
        plot_transition_profiles(transition_metrics, report_assets)
        write_tsv(SUMMARY_TSV, list(transition_metrics.values()))
        write_report_readme(transition_metrics, report_assets)

    report_assets.update(
        {
            "json": str(REPORT_JSON),
            "summary_tsv": str(SUMMARY_TSV),
            "underlying_json": str(UNDERLYING_JSON),
            "mp4": assets.get("mp4", row.get("mp4", "")),
            "keyframes_png": assets.get("keyframes_png", ""),
            "metrics_png": assets.get("metrics_png", ""),
            "metrics_csv": assets.get("metrics_csv", ""),
        }
    )
    all_report_assets_exist = all(
        Path(path).is_file() and Path(path).stat().st_size > 0
        for key, path in report_assets.items()
        if key not in {"json", "mp4"} and path
    )
    checks = {
        "underlying_returncode_zero": underlying_returncode == 0,
        "underlying_status_ok": underlying.get("status")
        == "ok_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval",
        "single_transition_task_attempted": len(rows) == 1 and row.get("task") == TASK,
        "task_row_status_ok": row.get("status") == "ok_official_csv_loop_task_conditioned_latent_guidance_rollout_eval",
        "rollout_299_steps": row.get("rollout_steps") == 299,
        "capture_npz_exists": capture_npz.is_file(),
        "mp4_path_recorded": bool(report_assets.get("mp4")),
        "transition_metrics_recorded": bool(transition_metrics)
        and all(item.get("available") for item in transition_metrics.values()),
        "report_assets_exist": all_report_assets_exist,
        "uses_official_importer_export_usd": underlying.get("checks", {}).get("uses_official_importer_export_usd")
        is True,
        "uses_full_public_motion_bundle": underlying.get("checks", {}).get("uses_full_public_motion_bundle")
        is True,
        "does_not_claim_fig5b_paper_protocol": True,
        "does_not_claim_fig5d_tsne": True,
        "does_not_claim_official_checkpoint": True,
        "does_not_claim_real_robot": True,
        "does_not_claim_goal_complete": True,
    }
    guided = transition_metrics.get("receding_latent_guided", {})
    result_row = {
        "task": TASK,
        "status": "ok" if all(checks.values()) else "failed",
        "rollout_steps": row.get("rollout_steps"),
        "selected_physical_gpu": row.get("selected_physical_gpu"),
        "guided_reward_mean": row.get("guided_reward_mean"),
        "guided_target_body_error_mean": row.get("guided_target_body_error_mean"),
        "guidance_cost_delta_mean": row.get("guidance_cost_delta_mean"),
        "guided_late_minus_early_speed_mps": guided.get("late_minus_early_speed_mps"),
        "guided_target_speed_rmse_mps": guided.get("target_speed_rmse_mps"),
        "guided_speed_target_corr": guided.get("speed_target_corr"),
        "capture_npz": str(capture_npz) if str(capture_npz) else "",
        "mp4": report_assets.get("mp4", ""),
    }
    payload = {
        "status": "ok_official_importer_export_full_bundle_transition_guidance_rollout_eval"
        if all(checks.values())
        else "failed_official_importer_export_full_bundle_transition_guidance_rollout_eval",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_g1_official_importer_export_full_bundle_transition_guidance_rollout_eval",
        "scope": (
            "Runs one local official-importer-export IsaacLab closed-loop walk-to-run transition guidance proxy. "
            "This is Fig. 5B/5D-adjacent local virtual evidence only."
        ),
        "underlying_returncode": underlying_returncode,
        "underlying_error": underlying_error,
        "underlying_summary": underlying,
        "rows": [result_row],
        "transition_metrics": transition_metrics,
        "checks": checks,
        "outputs": {
            "json": str(SUMMARY_JSON),
            "tsv": str(SUMMARY_TSV),
            "report_json": str(REPORT_JSON),
            "underlying_json": str(UNDERLYING_JSON),
            "underlying_tsv": str(UNDERLYING_TSV),
            "visualization_root": str(VIS_ROOT),
            "report_assets_root": str(REPORT_ROOT),
            "capture_npz": str(capture_npz) if str(capture_npz) else "",
            "assets": report_assets,
        },
        "interpretation": {
            "goal_complete": False,
            "claim_level": "local_virtual_official_importer_export_walk_to_run_transition_proxy",
            "paper_level_status": "qualitative_proxy_only",
            "why_not_paper_level": (
                "The run uses local PPO/VAE/denoiser checkpoints, a local velocity-ramp transition proxy cost, "
                "and the recovered official-importer-export G1 USDA path. It is not the paper Fig. 5B transition "
                "protocol, not the paper Fig. 5D t-SNE panel, not an official BeyondMimic checkpoint result, not "
                "TensorRT deployment, and not real-robot evidence."
            ),
        },
    }
    return payload


def main() -> None:
    code, error = run_underlying()
    summary = build_summary(code, error)
    write_json(SUMMARY_JSON, summary)
    write_json(REPORT_JSON, summary)
    print(json.dumps({"status": summary["status"], "json": str(SUMMARY_JSON)}, sort_keys=True))
    if summary["status"] != "ok_official_importer_export_full_bundle_transition_guidance_rollout_eval":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
