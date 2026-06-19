#!/usr/bin/env python3
"""Run task-conditioned local receding latent-guidance rollouts.

This script reuses the validated official-csv-loop receding latent-guidance
runner and executes several paper-facing guidance proxy tasks in closed-loop
IsaacLab simulation. It is local resource-adjusted virtual evidence only: it
does not use the official BeyondMimic diffusion checkpoint and must not be
reported as Fig. 5/Fig. 6 paper-level reproduction.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_g1_official_csv_loop_receding_latent_guidance_rollout_eval.py"


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default)))


OUT_ROOT = env_path(
    "BM_TASK_CONDITIONED_OUT_ROOT",
    ROOT / "res/visualization/official_csv_loop_task_conditioned_latent_guidance_rollout",
)
SUMMARY_ROOT = env_path(
    "BM_TASK_CONDITIONED_SUMMARY_ROOT",
    ROOT / "res/level_c/official_csv_loop_task_conditioned_latent_guidance_rollout_eval",
)
SUMMARY_JSON = env_path(
    "BM_TASK_CONDITIONED_SUMMARY_JSON",
    SUMMARY_ROOT / "level_c_official_csv_loop_task_conditioned_latent_guidance_rollout_eval.json",
)
SUMMARY_TSV = env_path(
    "BM_TASK_CONDITIONED_SUMMARY_TSV",
    SUMMARY_ROOT / "level_c_official_csv_loop_task_conditioned_latent_guidance_rollout_eval.tsv",
)
LOG_ROOT = env_path(
    "BM_TASK_CONDITIONED_LOG_ROOT",
    ROOT / "logs/tracking_g1_official_csv_loop_task_conditioned_latent_guidance_rollout_eval",
)
FAILED_ROOT = env_path(
    "BM_TASK_CONDITIONED_FAILED_ROOT",
    ROOT / "res/failed_runs/tracking_g1_official_csv_loop_task_conditioned_latent_guidance_rollout_eval",
)
RUN_ROOT = env_path(
    "BM_TASK_CONDITIONED_RUN_ROOT",
    ROOT / "res/runs/tracking_g1_official_csv_loop_task_conditioned_latent_guidance_rollout_eval",
)
TASKS = [
    item.strip()
    for item in os.environ.get("BM_TASK_CONDITIONED_TASKS", "joystick,waypoint,obstacle_avoidance,composed").split(",")
    if item.strip()
]
DEFAULT_TASK_SEEDS = {
    "joystick": 20260641,
    "waypoint": 20260642,
    "obstacle_avoidance": 20260643,
    "composed": 20260644,
}
TASK_SEEDS = DEFAULT_TASK_SEEDS.copy()
if os.environ.get("BM_TASK_CONDITIONED_TASK_SEEDS_JSON"):
    TASK_SEEDS.update({key: int(value) for key, value in json.loads(os.environ["BM_TASK_CONDITIONED_TASK_SEEDS_JSON"]).items()})


TASK_COST_CODE = r'''
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
        raise ValueError(f"Unknown guidance task: {task_name}")

    def best_task_scale(path, task_name):
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            return float(summary["worker_summary"]["task_summaries"][task_name]["splits"]["validation"]["scale"])
        except Exception:
            return 0.01
'''


def import_base_module():
    spec = importlib.util.spec_from_file_location("bm_receding_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_worker_code(base_worker: str) -> str:
    code = re.sub(
        r"\n    def composed_cost\(tau\):.*?\n    def best_composed_scale\(path\):.*?\n            return 0\.01\n",
        "\n" + TASK_COST_CODE + "\n",
        base_worker,
        flags=re.S,
    )
    code = code.replace(
        'guidance_scale = best_composed_scale(guidance_json) * guidance_scale_mult',
        'task_name = os.environ.get("BM_GUIDANCE_TASK", "composed")\n'
        '    guidance_scale = best_task_scale(guidance_json, task_name) * guidance_scale_mult',
    )
    code = code.replace("cost_before = composed_cost(variable)", "cost_before = task_cost(variable, task_name)")
    code = code.replace("cost_after = composed_cost(guided)", "cost_after = task_cost(guided, task_name)")
    code = code.replace(
        '"type": "local_receding_horizon_state_latent_composed_cost",\n'
        '            "base_scale_from_offline_guidance": best_composed_scale(guidance_json),',
        '"type": f"local_receding_horizon_state_latent_{task_name}_cost",\n'
        '            "task": task_name,\n'
        '            "base_scale_from_offline_guidance": best_task_scale(guidance_json, task_name),',
    )
    return code


def make_render_code(base_render: str) -> str:
    code = base_render.replace(
        "official_csv_loop_receding_latent_guidance_rollout",
        "official_csv_loop_task_conditioned_latent_guidance_rollout",
    )
    code = code.replace(
        "Local Receding-Horizon Latent Guidance Closed-Loop Rollout",
        "Local Task-Conditioned Receding-Horizon Latent Guidance Rollout",
    )
    code = code.replace(
        "local_virtual_receding_horizon_latent_guidance_rollout",
        "local_virtual_task_conditioned_receding_horizon_latent_guidance_rollout",
    )
    code = code.replace(
        "This directory contains a local virtual closed-loop rollout comparing teacher, VAE base, denoised-latent, and guided-latent variants.",
        "This directory contains one local virtual closed-loop task-conditioned rollout comparing teacher, VAE base, denoised-latent, and guided-latent variants.",
    )
    return code


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task",
        "status",
        "selected_physical_gpu",
        "rollout_steps",
        "guided_reward_mean",
        "guided_target_body_error_mean",
        "guided_done_count_total",
        "guided_teacher_action_mse_mean",
        "guided_base_action_mse_mean",
        "guidance_cost_delta_mean",
        "asset_json",
        "mp4",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def run_task(task: str) -> dict[str, Any]:
    base = import_base_module()
    base.OUT = OUT_ROOT / task
    base.SUMMARY_JSON = SUMMARY_ROOT / task / f"{task}_task_conditioned_latent_guidance_rollout_eval.json"
    base.SUMMARY_TSV = SUMMARY_ROOT / task / f"{task}_task_conditioned_latent_guidance_rollout_eval.tsv"
    base.LOG_DIR = LOG_ROOT / task
    base.FAILED_DIR = FAILED_ROOT / task
    base.RUN_ROOT = RUN_ROOT / task
    base.SEED = TASK_SEEDS[task]
    base.WORKER_CODE = make_worker_code(base.WORKER_CODE)
    base.RENDER_CODE = make_render_code(base.RENDER_CODE)

    original_base_env = base.base_env

    def task_base_env(*args, **kwargs):
        env = original_base_env(*args, **kwargs)
        env["BM_GUIDANCE_TASK"] = task
        return env

    base.base_env = task_base_env
    base.main()

    summary = load_json(base.SUMMARY_JSON)
    asset_json = Path(summary.get("outputs", {}).get("asset_json", ""))
    asset = load_json(asset_json) if str(asset_json) else {}
    capture = summary.get("run", {}).get("capture_metrics", {})
    capture_guidance = capture.get("guidance", {})
    guided_metrics = capture.get("variant_metrics", {}).get("receding_latent_guided", {})

    summary["status"] = (
        "ok_official_csv_loop_task_conditioned_latent_guidance_rollout_eval"
        if summary.get("checks", {}).get("capture_ok") and summary.get("checks", {}).get("render_ok")
        else summary.get("status")
    )
    summary["experiment_type"] = "tracking_g1_official_csv_loop_task_conditioned_latent_guidance_rollout_eval"
    summary["task"] = task
    summary["scope"] = (
        f"Runs a local task-conditioned receding latent-guidance closed-loop rollout for {task}. "
        "This is local virtual bridge evidence and not paper Fig. 5/Fig. 6 reproduction."
    )
    summary.setdefault("metrics", {})["task"] = task
    summary.setdefault("metrics", {})["task_guidance"] = capture_guidance
    summary.setdefault("checks", {})["task_guidance_recorded"] = capture_guidance.get("task") == task
    summary.setdefault("checks", {})["does_not_claim_task_success_rate"] = True
    summary.setdefault("checks", {})["does_not_claim_fig5_fig6"] = True
    summary.setdefault("interpretation", {})["paper_level_status"] = (
        "local_virtual_task_conditioned_receding_horizon_latent_guidance_rollout"
        if summary.get("status", "").startswith("ok_")
        else "not_completed"
    )
    write_json(base.SUMMARY_JSON, summary)

    if asset:
        asset["task"] = task
        asset["claim_level"] = "local_virtual_task_conditioned_receding_horizon_latent_guidance_rollout"
        asset.setdefault("checks", {})["task_conditioned_local_asset"] = True
        write_json(asset_json, asset)

    assets = summary.get("outputs", {}).get("assets", {})
    return {
        "task": task,
        "status": summary.get("status"),
        "selected_physical_gpu": summary.get("config", {}).get("selected_physical_gpu"),
        "rollout_steps": capture.get("rollout_steps"),
        "guided_reward_mean": guided_metrics.get("reward_mean"),
        "guided_target_body_error_mean": guided_metrics.get("target_body_error_mean"),
        "guided_done_count_total": guided_metrics.get("done_count_total"),
        "guided_teacher_action_mse_mean": guided_metrics.get("guided_teacher_action_mse_mean"),
        "guided_base_action_mse_mean": guided_metrics.get("guided_base_action_mse_mean"),
        "guidance_cost_delta_mean": guided_metrics.get("guidance_cost_delta_mean"),
        "asset_json": str(asset_json) if asset_json else "",
        "mp4": assets.get("mp4", ""),
        "summary_json": str(base.SUMMARY_JSON),
        "summary_tsv": str(base.SUMMARY_TSV),
        "claim_level": "local_virtual_task_conditioned_receding_horizon_latent_guidance_rollout",
    }


def main() -> None:
    rows = [run_task(task) for task in TASKS]
    status_ok = all(row["status"] == "ok_official_csv_loop_task_conditioned_latent_guidance_rollout_eval" for row in rows)
    summary = {
        "status": "ok_official_csv_loop_task_conditioned_latent_guidance_rollout_eval" if status_ok else "failed",
        "experiment_type": "tracking_g1_official_csv_loop_task_conditioned_latent_guidance_rollout_eval",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Aggregates local closed-loop task-conditioned latent guidance rollouts for joystick, waypoint, "
            "obstacle-avoidance, and composed proxy objectives in IsaacLab."
        ),
        "tasks": TASKS,
        "rows": rows,
        "checks": {
            "four_tasks_attempted": len(rows) == 4,
            "all_tasks_ok": status_ok,
            "all_tasks_have_asset_json": all(Path(row["asset_json"]).is_file() for row in rows),
            "all_tasks_have_mp4_path_recorded": all(bool(row["mp4"]) for row in rows),
            "all_tasks_have_guidance_cost_delta": all(row["guidance_cost_delta_mean"] is not None for row in rows),
            "does_not_claim_paper_level_guidance": True,
            "does_not_claim_fig5_fig6": True,
            "does_not_claim_real_robot": True,
            "does_not_claim_goal_complete": True,
        },
        "outputs": {
            "json": str(SUMMARY_JSON),
            "tsv": str(SUMMARY_TSV),
            "task_summaries": [row["summary_json"] for row in rows],
            "task_assets": [row["asset_json"] for row in rows],
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "local_virtual_task_conditioned_receding_horizon_latent_guidance_rollout",
            "why_not_complete": (
                "These are closed-loop local virtual task-conditioned guidance rollouts using local PPO/VAE/denoiser "
                "checkpoints and proxy costs. They are useful report evidence but are not official BeyondMimic "
                "Fig. 5/Fig. 6 results, not TensorRT/asynchronous deployment, and not real-robot validation."
            ),
        },
    }
    write_json(SUMMARY_JSON, summary)
    write_tsv(SUMMARY_TSV, rows)
    print(json.dumps({"status": summary["status"], "json": str(SUMMARY_JSON), "rows": len(rows)}, sort_keys=True))
    if not status_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
