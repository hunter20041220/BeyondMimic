#!/usr/bin/env python3
"""Formula-level task guidance scale sweeps for Phase 8 debug coverage."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from level_c_guidance_formula_probe import (
    DEFAULT_FIXTURE,
    DEFAULT_MANIFEST,
    extract_future,
    joystick_cost,
    keyframe_candidate_cost,
    sdf_obstacle_cost,
    waypoint_cost,
)


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/guidance_task_scale_sweep"
SCALES = [0.0, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5]


def rebuild_tensors(future: torch.Tensor) -> dict[str, torch.Tensor]:
    body_pos = future[:, 15:57].reshape(future.shape[0], 14, 3)
    root_vel = future[:, 9:12]
    root_xy = torch.cumsum(root_vel[:, :2] * 0.04, dim=0)
    return {
        "future": future,
        "root_xy": root_xy,
        "root_vel_xy": root_vel[:, :2],
        "body_pos_xy": body_pos[:, :, :2],
    }


def task_cost(task: str, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    command_velocity = torch.tensor([0.35, 0.0], dtype=torch.float64)
    goal_xy = torch.tensor([0.08, 0.04], dtype=torch.float64)
    obstacle_center = torch.tensor([0.04, 0.0], dtype=torch.float64)
    keyframe_xy = torch.tensor([0.04, -0.02], dtype=torch.float64)
    if task == "joystick":
        return joystick_cost(tensors["root_vel_xy"], command_velocity)
    if task == "waypoint":
        return waypoint_cost(tensors["root_xy"], tensors["root_vel_xy"], goal_xy)
    if task == "obstacle_avoidance":
        return sdf_obstacle_cost(tensors["body_pos_xy"], obstacle_center, 0.2, 0.05, 0.1)
    if task == "inpainting":
        return keyframe_candidate_cost(tensors["root_xy"], keyframe_xy, min(5, tensors["root_xy"].shape[0] - 1))
    if task == "composed_objectives":
        return (
            joystick_cost(tensors["root_vel_xy"], command_velocity)
            + waypoint_cost(tensors["root_xy"], tensors["root_vel_xy"], goal_xy)
            + sdf_obstacle_cost(tensors["body_pos_xy"], obstacle_center, 0.2, 0.05, 0.1)
        )
    raise ValueError(f"unknown task {task}")


def run_task(task: str, initial_future: torch.Tensor) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    future = initial_future.detach().clone().requires_grad_(True)
    initial_cost = task_cost(task, rebuild_tensors(future))
    initial_cost.backward()
    grad = future.grad.detach().clone()
    grad_norm = float(torch.linalg.norm(grad).item())
    rows: list[dict[str, Any]] = []
    for scale in SCALES:
        stepped = (initial_future.detach() - scale * grad).requires_grad_(True)
        cost_after = task_cost(task, rebuild_tensors(stepped))
        rows.append(
            {
                "task": task,
                "scale": scale,
                "initial_cost": float(initial_cost.detach().cpu()),
                "cost_after": float(cost_after.detach().cpu()),
                "cost_delta": float(initial_cost.detach().cpu() - cost_after.detach().cpu()),
                "relative_cost_delta": float((initial_cost.detach().cpu() - cost_after.detach().cpu()) / (abs(initial_cost.detach().cpu()) + 1e-12)),
                "gradient_norm": grad_norm,
                "finite": bool(torch.isfinite(cost_after).detach().cpu().item()),
            }
        )
    best = min(rows, key=lambda row: row["cost_after"])
    summary = {
        "initial_cost": float(initial_cost.detach().cpu()),
        "gradient_norm": grad_norm,
        "best_scale": best["scale"],
        "best_cost_after": best["cost_after"],
        "best_cost_delta": best["cost_delta"],
        "all_finite": all(row["finite"] for row in rows),
        "best_improves_over_zero": best["cost_after"] < rows[0]["cost_after"],
    }
    return rows, summary


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "task",
        "scale",
        "initial_cost",
        "cost_after",
        "cost_delta",
        "relative_cost_delta",
        "gradient_norm",
        "finite",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fixture = np.load(DEFAULT_FIXTURE)
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    tensors = extract_future(fixture["candidate_hybrid_state_windows"], manifest["feature_slices"], history=4)
    initial_future = tensors["future"].detach().clone()
    tasks = ["joystick", "waypoint", "obstacle_avoidance", "inpainting", "composed_objectives"]
    all_rows: list[dict[str, Any]] = []
    task_summaries: dict[str, Any] = {}
    for task in tasks:
        rows, task_summary = run_task(task, initial_future)
        all_rows.extend(rows)
        task_summaries[task] = task_summary

    checks = {
        "five_tasks_swept": sorted(task_summaries) == sorted(tasks),
        "scale_count_per_task": all(sum(1 for row in all_rows if row["task"] == task) == len(SCALES) for task in tasks),
        "all_rows_finite": all(row["finite"] for row in all_rows),
        "all_gradients_nonzero": all(task_summaries[task]["gradient_norm"] > 0.0 for task in tasks),
        "all_tasks_have_improving_positive_scale": all(task_summaries[task]["best_improves_over_zero"] for task in tasks),
        "does_not_claim_rollout_or_video": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "guidance_task_scale_sweep",
        "scope": "formula-level multi-scale guidance sweeps for Phase 8 task costs on one motion-derived future state",
        "settings": {
            "scales": SCALES,
            "tasks": tasks,
            "fixture": str(DEFAULT_FIXTURE),
            "manifest": str(DEFAULT_MANIFEST),
            "future_shape": list(initial_future.shape),
        },
        "task_summaries": task_summaries,
        "row_count": len(all_rows),
        "rows": all_rows,
        "checks": checks,
        "not_a_replacement_for": [
            "closed-loop guided diffusion rollout",
            "task validation/test scale selection",
            "success/failure videos",
            "Fig. 5/Fig. 6 reproduction",
        ],
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "partial",
            "why_not_complete": (
                "This sweeps formula-level task costs on one debug future-state tensor. It improves Phase 8 task-scale "
                "coverage but does not run a trained denoising policy, closed-loop simulation, or video evaluation."
            ),
        },
        "outputs": {
            "json": str(OUT / "level_c_guidance_task_scale_sweep.json"),
            "tsv": str(OUT / "level_c_guidance_task_scale_sweep.tsv"),
        },
    }
    (OUT / "level_c_guidance_task_scale_sweep.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_tsv(OUT / "level_c_guidance_task_scale_sweep.tsv", all_rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(all_rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
