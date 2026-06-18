#!/usr/bin/env python3
"""Offline guidance sweeps on the full public-LAFAN1 paper-architecture checkpoint.

This applies paper-style task-cost gradients to predicted clean state-latent
trajectories from the trained local diffusion checkpoint. It is an offline
trajectory adjustment/evaluation, not a closed-loop rollout.
"""

from __future__ import annotations

import csv
import hashlib
import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SCRIPTS = ROOT / "reproduction/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_lafan1_paper_level_vae_diffusion as paper_train  # noqa: E402
from level_c_guidance_formula_probe import (  # noqa: E402
    joystick_cost,
    keyframe_candidate_cost,
    sdf_obstacle_cost,
    waypoint_cost,
)


OUT = ROOT / "res/level_c/lafan1_paper_arch_guidance_eval"
TRAINING_JSON = (
    ROOT
    / "res/level_c/lafan1_paper_arch_vae_diffusion_training/"
    / "lafan1_paper_arch_vae_diffusion_training.json"
)
OFFLINE_JSON = (
    ROOT
    / "res/level_c/lafan1_paper_arch_offline_metrics/"
    / "level_c_lafan1_paper_arch_offline_metrics_audit.json"
)
TASKS = ["joystick", "waypoint", "obstacle_avoidance", "inpainting", "composed_objectives"]
SCALES = [0.0, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(path: str | None, default: Path) -> Path:
    if path is None:
        return default
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def state_tensors(state99: torch.Tensor) -> dict[str, torch.Tensor]:
    # Paper S3 local state layout used by local windows:
    # root pose/twist 18, body pos 42, body velocity 42 minus compact overlaps -> body pos starts at 15 in fixtures.
    body_pos = state99[:, 15:57].reshape(state99.shape[0], 14, 3)
    root_vel = state99[:, 9:12]
    root_xy = torch.cumsum(root_vel[:, :2] * 0.04, dim=0)
    return {
        "root_xy": root_xy,
        "root_vel_xy": root_vel[:, :2],
        "body_pos_xy": body_pos[:, :, :2],
    }


def task_cost(task: str, state99: torch.Tensor) -> torch.Tensor:
    tensors = state_tensors(state99)
    command_velocity = torch.tensor([0.35, 0.0], dtype=state99.dtype, device=state99.device)
    goal_xy = torch.tensor([0.08, 0.04], dtype=state99.dtype, device=state99.device)
    obstacle_center = torch.tensor([0.04, 0.0], dtype=state99.dtype, device=state99.device)
    keyframe_xy = torch.tensor([0.04, -0.02], dtype=state99.dtype, device=state99.device)
    if task == "joystick":
        return joystick_cost(tensors["root_vel_xy"], command_velocity)
    if task == "waypoint":
        return waypoint_cost(tensors["root_xy"], tensors["root_vel_xy"], goal_xy)
    if task == "obstacle_avoidance":
        return sdf_obstacle_cost(tensors["body_pos_xy"], obstacle_center, 0.2, 0.05, 0.1)
    if task == "inpainting":
        return keyframe_candidate_cost(tensors["root_xy"], keyframe_xy, min(5, state99.shape[0] - 1))
    if task == "composed_objectives":
        return (
            joystick_cost(tensors["root_vel_xy"], command_velocity)
            + waypoint_cost(tensors["root_xy"], tensors["root_vel_xy"], goal_xy)
            + sdf_obstacle_cost(tensors["body_pos_xy"], obstacle_center, 0.2, 0.05, 0.1)
        )
    raise ValueError(task)


def primary_metric(task: str, state99: torch.Tensor) -> torch.Tensor:
    tensors = state_tensors(state99)
    command_velocity = torch.tensor([0.35, 0.0], dtype=state99.dtype, device=state99.device)
    goal_xy = torch.tensor([0.08, 0.04], dtype=state99.dtype, device=state99.device)
    obstacle_center = torch.tensor([0.04, 0.0], dtype=state99.dtype, device=state99.device)
    keyframe_xy = torch.tensor([0.04, -0.02], dtype=state99.dtype, device=state99.device)
    if task == "joystick":
        return torch.mean((tensors["root_vel_xy"] - command_velocity) ** 2)
    if task == "waypoint":
        return torch.linalg.norm(tensors["root_xy"][-1] - goal_xy)
    if task == "obstacle_avoidance":
        clearance = torch.linalg.norm(tensors["body_pos_xy"] - obstacle_center, dim=-1) - 0.2 - 0.05
        return torch.min(clearance)
    if task == "inpainting":
        idx = min(5, state99.shape[0] - 1)
        return torch.linalg.norm(tensors["root_xy"][idx] - keyframe_xy)
    if task == "composed_objectives":
        clearance = torch.linalg.norm(tensors["body_pos_xy"] - obstacle_center, dim=-1) - 0.2 - 0.05
        return torch.min(clearance)
    raise ValueError(task)


def direction(task: str) -> str:
    return "higher_is_better" if task in {"obstacle_avoidance", "composed_objectives"} else "lower_is_better"


def improves(task: str, before: float, after: float) -> bool:
    return after > before if direction(task) == "higher_is_better" else after < before


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = resolve_path(args.output_dir, OUT)
    training_json = resolve_path(args.training_json, TRAINING_JSON)
    offline_json = resolve_path(args.offline_json, OFFLINE_JSON)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    training = load_json(training_json)
    offline = load_json(offline_json)
    checkpoint_path = Path(training["outputs"]["checkpoint"])
    dataset_path = Path(training["outputs"]["dataset_npz"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = paper_train.TrainConfig(**payload["config"])
    diffusion = paper_train.DiffusionTransformer(cfg)
    diffusion.load_state_dict(strip_module_prefix(payload["diffusion_state_dict"]))
    diffusion.eval()

    with np.load(dataset_path, allow_pickle=True) as data:
        states = data["states"].astype(np.float32)
        projected = data["projected_states"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        projection = data["projection"].astype(np.float32)
        projection_inverse = data["projection_inverse"].astype(np.float32)
        split_labels = data["split_labels"].astype(str)

    requested_splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    selected_parts = []
    split_index_counts: dict[str, int] = {}
    for split in requested_splits:
        split_indices = np.nonzero(split_labels == split)[0]
        if len(split_indices) == 0:
            raise RuntimeError(f"split {split!r} does not have any windows")
        if args.max_windows_per_split > 0:
            split_indices = split_indices[: args.max_windows_per_split]
        selected_parts.append(split_indices)
        split_index_counts[split] = int(len(split_indices))
    selected = np.concatenate(selected_parts).astype(np.int64)
    selected_split_labels = split_labels[selected]
    state_tokens = torch.from_numpy(states[selected].reshape(-1, cfg.state_dim))
    action_tokens = torch.from_numpy(actions[selected].reshape(-1, cfg.action_dim))
    vae = paper_train.ConditionalVAE(cfg)
    vae.load_state_dict(strip_module_prefix(payload["vae_state_dict"]))
    vae.eval()
    with torch.no_grad():
        latent = vae.encode(state_tokens, action_tokens).reshape(len(selected), cfg.seq_len, cfg.latent_dim)
    clean_tau = torch.cat([torch.from_numpy(projected[selected]), latent], dim=-1)
    noisy, steps = paper_train.noised_tau(clean_tau, cfg, torch.device("cpu"))
    with torch.no_grad():
        pred_tau = diffusion(noisy, steps)

    pinv = torch.from_numpy(projection_inverse)
    projection_t = torch.from_numpy(projection)
    rows: list[dict[str, Any]] = []
    task_summaries: dict[str, Any] = {}
    all_guided_tau: dict[str, list[np.ndarray]] = {task: [] for task in TASKS}
    for task in TASKS:
        task_rows: list[dict[str, Any]] = []
        for local_i, window_idx in enumerate(selected):
            pred_proj = pred_tau[local_i, :, : cfg.projected_state_dim].detach()
            pred_latent = pred_tau[local_i, :, cfg.projected_state_dim :].detach()
            pred_state99 = pred_proj @ pinv.T
            base_cost = float(task_cost(task, pred_state99).detach().cpu())
            base_primary = float(primary_metric(task, pred_state99).detach().cpu())
            variable = pred_state99.detach().clone().requires_grad_(True)
            cost = task_cost(task, variable)
            cost.backward()
            grad = variable.grad.detach()
            grad_norm = float(torch.linalg.norm(grad).cpu())
            for scale in SCALES:
                guided_state99 = (pred_state99 - scale * grad).detach()
                # Use the exact projection matrix from the training dataset to return to the model token basis.
                guided_projected = guided_state99 @ projection_t.T
                guided_tau = torch.cat([guided_projected, pred_latent], dim=-1)
                guided_cost = float(task_cost(task, guided_state99).detach().cpu())
                guided_primary = float(primary_metric(task, guided_state99).detach().cpu())
                state_mse_to_pred = float(torch.mean((guided_state99 - pred_state99) ** 2).cpu())
                state_mse_to_reference = float(torch.mean((guided_state99 - torch.from_numpy(states[window_idx])) ** 2).cpu())
                tau_mse_to_clean = float(torch.mean((guided_tau - clean_tau[local_i]) ** 2).cpu())
                task_rows.append(
                    {
                        "task": task,
                        "split": str(selected_split_labels[local_i]),
                        "window_index": int(window_idx),
                        "scale": scale,
                        "base_cost": base_cost,
                        "guided_cost": guided_cost,
                        "cost_delta": base_cost - guided_cost,
                        "base_primary_metric": base_primary,
                        "guided_primary_metric": guided_primary,
                        "primary_delta": guided_primary - base_primary,
                        "primary_improved": improves(task, base_primary, guided_primary),
                        "direction": direction(task),
                        "gradient_norm": grad_norm,
                        "state_mse_to_unguided_pred": state_mse_to_pred,
                        "state_mse_to_reference": state_mse_to_reference,
                        "tau_mse_to_clean": tau_mse_to_clean,
                        "finite": bool(np.isfinite(guided_cost) and np.isfinite(guided_primary)),
                    }
                )
                if scale == max(SCALES):
                    all_guided_tau[task].append(guided_tau.detach().numpy())
        rows.extend(task_rows)
        nonzero = [row for row in task_rows if row["scale"] > 0.0]
        best_rows = []
        for window_idx in selected:
            candidates = [row for row in nonzero if row["window_index"] == int(window_idx)]
            best = min(candidates, key=lambda r: r["guided_cost"])
            best_rows.append(best)
        task_summaries[task] = {
            "window_count": len(selected),
            "scale_count": len(SCALES),
            "mean_base_cost": float(np.mean([row["base_cost"] for row in task_rows if row["scale"] == 0.0])),
            "mean_best_cost": float(np.mean([row["guided_cost"] for row in best_rows])),
            "mean_cost_delta": float(np.mean([row["cost_delta"] for row in best_rows])),
            "mean_gradient_norm": float(np.mean([row["gradient_norm"] for row in task_rows if row["scale"] == 0.0])),
            "best_rows_primary_improved_count": int(sum(row["primary_improved"] for row in best_rows)),
            "all_best_costs_improve": all(row["cost_delta"] > 0.0 for row in best_rows),
        }

    npz_path = out / "level_c_lafan1_paper_arch_guidance_eval.npz"
    np.savez_compressed(
        npz_path,
        selected_indices=selected.astype(np.int64),
        clean_tau=clean_tau.detach().numpy().astype(np.float32),
        noisy_tau=noisy.detach().numpy().astype(np.float32),
        unguided_pred_tau=pred_tau.detach().numpy().astype(np.float32),
        **{f"guided_tau_{task}_max_scale": np.stack(values).astype(np.float32) for task, values in all_guided_tau.items()},
    )
    checks = {
        "source_training_status_ok": training["status"] == "ok",
        "offline_metrics_status_ok": offline["status"] == "ok",
        "source_checkpoint_hash_matches": sha256(checkpoint_path) == training["metrics"]["checkpoint_sha256"],
        "paper_architecture_checkpoint": payload.get("paper_architecture") is True,
        "public_dataset_boundary_recorded": payload.get("paper_dataset") is False,
        "five_tasks_evaluated": sorted(task_summaries) == sorted(TASKS),
        "scale_grid_evaluated": all(summary["scale_count"] == len(SCALES) for summary in task_summaries.values()),
        "requested_splits_evaluated": set(split_index_counts) == set(requested_splits),
        "selected_window_count_matches_split_counts": len(selected) == sum(split_index_counts.values()),
        "all_rows_finite": all(row["finite"] for row in rows),
        "all_task_best_costs_improve": all(summary["all_best_costs_improve"] for summary in task_summaries.values()),
        "gradients_nonzero": all(summary["mean_gradient_norm"] > 0.0 for summary in task_summaries.values()),
        "npz_written": npz_path.is_file() and npz_path.stat().st_size > 0,
        "does_not_claim_closed_loop_rollout": True,
        "does_not_claim_success_failure_videos": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "public_lafan1_paper_arch_guidance_eval",
        "scope": (
            "Offline multi-scale task-cost guidance applied to predicted clean state-latent trajectories from the full "
            "paper-architecture public-LAFAN1 diffusion checkpoint."
        ),
        "settings": {
            "tasks": TASKS,
            "scales": SCALES,
            "splits": requested_splits,
            "max_windows_per_split": args.max_windows_per_split,
            "split_window_counts": split_index_counts,
            "selected_window_indices": selected.tolist(),
            "state_dim": cfg.state_dim,
            "projected_state_dim": cfg.projected_state_dim,
            "latent_dim": cfg.latent_dim,
            "seq_len": cfg.seq_len,
            "training_json": str(training_json),
            "offline_metrics_json": str(offline_json),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
        },
        "task_summaries": task_summaries,
        "row_count": len(rows),
        "rows": rows,
        "checks": checks,
        "outputs": {
            "json": str(out / "level_c_lafan1_paper_arch_guidance_eval.json"),
            "tsv": str(out / "level_c_lafan1_paper_arch_guidance_eval.tsv"),
            "npz": str(npz_path),
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "paper_architecture_public_data_offline_guidance",
            "why_not_complete": (
                "This evaluates task-cost gradients on full-checkpoint predicted trajectories and records with/without "
                "guidance and scale sweeps. It is not a reverse-denoising controller, closed-loop IsaacLab rollout, "
                "success/failure video, Fig. 5/Fig. 6 reproduction, TensorRT deployment, or real robot test."
            ),
        },
    }
    write_json_atomic(Path(summary["outputs"]["json"]), summary)
    write_tsv(Path(summary["outputs"]["tsv"]), rows)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "rows": len(rows),
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run offline task-cost guidance sweeps for a public-LAFAN1 paper-architecture checkpoint."
    )
    parser.add_argument(
        "--training-json",
        default=None,
        help="Training summary JSON to evaluate. Defaults to the original public-LAFAN1 paper-architecture run.",
    )
    parser.add_argument(
        "--offline-json",
        default=None,
        help="Offline metrics JSON paired with the training checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for JSON/TSV/NPZ. Defaults to the original guidance-eval directory.",
    )
    parser.add_argument(
        "--splits",
        default="validation",
        help="Comma-separated dataset splits to evaluate. Default preserves the original validation-only audit.",
    )
    parser.add_argument(
        "--max-windows-per-split",
        type=int,
        default=5,
        help="Maximum windows per split; use -1 to evaluate each requested split fully.",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
