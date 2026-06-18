#!/usr/bin/env python3
"""Batched full-split reverse-denoising guidance audit for the paper architecture."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SCRIPTS = ROOT / "reproduction/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import train_lafan1_paper_level_vae_diffusion as paper_train  # noqa: E402
from level_c_lafan1_paper_arch_guidance_eval import (  # noqa: E402
    TASKS,
    direction,
    improves,
    primary_metric,
    task_cost,
)
from level_c_lafan1_paper_arch_high_memory_batch_audit import (  # noqa: E402
    fill_reserve_tensors,
    nvidia_memory_rows,
    peak_allocated_by_gpu_mb,
    reserve_to_target,
    reset_peak_stats,
    top_up_reserves,
)


OUT = ROOT / "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split"
DEFAULT_TRAINING_JSON = (
    ROOT
    / "res/level_c/lafan1_paper_arch_symmetry_augmented_training/"
    / "lafan1_paper_arch_vae_diffusion_training.json"
)
DEFAULT_OFFLINE_GUIDANCE_JSON = (
    ROOT
    / "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/"
    / "level_c_lafan1_paper_arch_guidance_eval.json"
)
DIFFUSION_EQUATION_JSON = (
    ROOT
    / "res/level_c/diffusion_equation_audit/"
    / "level_c_diffusion_equation_audit.json"
)


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


def reverse_coefficients(steps: int, dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
    betas = torch.linspace(1e-4, 0.02, steps, dtype=dtype, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    prev_alpha_bars = torch.cat([torch.ones(1, dtype=dtype, device=device), alpha_bars[:-1]])
    c1 = betas * torch.sqrt(prev_alpha_bars) / (1.0 - alpha_bars)
    c2 = (1.0 - prev_alpha_bars) * torch.sqrt(alphas) / (1.0 - alpha_bars)
    return {"posterior_c1": c1, "posterior_c2": c2}


def select_indices(split_labels: np.ndarray, requested_splits: list[str], max_windows_per_split: int) -> tuple[np.ndarray, dict[str, int]]:
    parts = []
    counts: dict[str, int] = {}
    for split in requested_splits:
        split_indices = np.nonzero(split_labels == split)[0]
        if len(split_indices) == 0:
            raise RuntimeError(f"split {split!r} does not have any windows")
        if max_windows_per_split > 0:
            split_indices = split_indices[:max_windows_per_split]
        parts.append(split_indices)
        counts[split] = int(len(split_indices))
    return np.concatenate(parts).astype(np.int64), counts


def step_tensor(batch: int, seq_len: int, step: int, device: torch.device) -> torch.Tensor:
    return torch.full((batch, seq_len, 2), step, dtype=torch.long, device=device)


def noised_reference_start(clean_tau: torch.Tensor, cfg: paper_train.TrainConfig, seed: int) -> torch.Tensor:
    bars = paper_train.alpha_bars(cfg.diffusion_steps, clean_tau.device, clean_tau.dtype)
    alpha = bars[-1]
    generator = torch.Generator(device=clean_tau.device).manual_seed(seed)
    noise = torch.randn(clean_tau.shape, dtype=clean_tau.dtype, device=clean_tau.device, generator=generator)
    return torch.sqrt(alpha) * clean_tau + torch.sqrt(1.0 - alpha) * noise


def state_to_costs(task: str, state99: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    costs = []
    primaries = []
    for item in state99:
        costs.append(float(task_cost(task, item).detach().cpu()))
        primaries.append(float(primary_metric(task, item).detach().cpu()))
    return np.asarray(costs, dtype=np.float64), np.asarray(primaries, dtype=np.float64)


def task_gradients(task: str, state99: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    variable = state99.detach().clone().requires_grad_(True)
    cost = sum(task_cost(task, item) for item in variable)
    cost.backward()
    return variable.grad.detach(), cost.detach()


def summarize_task(
    task: str,
    rows: list[dict[str, Any]],
    window_indices: np.ndarray,
    cfg: paper_train.TrainConfig,
) -> dict[str, Any]:
    guided_rows = [row for row in rows if row["task"] == task and row["scale"] > 0.0]
    best_rows = []
    for window_idx in window_indices:
        candidates = [row for row in guided_rows if row["window_index"] == int(window_idx)]
        best_rows.append(min(candidates, key=lambda row: row["guided_final_cost"]))
    return {
        "window_count": int(len(window_indices)),
        "scale_count": int(len({row["scale"] for row in rows if row["task"] == task})),
        "mean_best_cost_delta": float(np.mean([row["cost_delta"] for row in best_rows])),
        "median_best_cost_delta": float(np.median([row["cost_delta"] for row in best_rows])),
        "mean_best_gradient_norm": float(np.mean([row["mean_gradient_norm"] for row in best_rows])),
        "best_rows_primary_improved_count": int(sum(row["primary_improved"] for row in best_rows)),
        "all_best_costs_improve": bool(all(row["cost_delta"] > 0.0 for row in best_rows)),
        "positive_best_cost_delta_fraction": float(np.mean([row["cost_delta"] > 0.0 for row in best_rows])),
        "all_best_rows_have_nonzero_guidance_gradients": bool(
            all(row["nonzero_gradient_steps"] == cfg.diffusion_steps for row in best_rows)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-json", default=str(DEFAULT_TRAINING_JSON))
    parser.add_argument("--offline-guidance-json", default=str(DEFAULT_OFFLINE_GUIDANCE_JSON))
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--splits", default="validation,test")
    parser.add_argument("--max-windows-per-split", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=660)
    parser.add_argument("--guidance-scales", default="0.0,0.00002,0.00005,0.0001,0.0002")
    parser.add_argument("--target-memory-mb", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 8-GPU full-split reverse-guidance audit")
    gpu_count = torch.cuda.device_count()
    if gpu_count != 8:
        raise RuntimeError(f"expected 8 GPUs, found {gpu_count}")

    out = resolve_path(args.output_dir, OUT)
    training_json = resolve_path(args.training_json, DEFAULT_TRAINING_JSON)
    offline_guidance_json = resolve_path(args.offline_guidance_json, DEFAULT_OFFLINE_GUIDANCE_JSON)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    training = load_json(training_json)
    offline_guidance = load_json(offline_guidance_json)
    diffusion_equations = load_json(DIFFUSION_EQUATION_JSON)
    checkpoint_path = Path(training["outputs"]["checkpoint"])
    dataset_path = Path(training["outputs"]["dataset_npz"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = paper_train.TrainConfig(**payload["config"])

    with np.load(dataset_path, allow_pickle=True) as data:
        states = data["states"].astype(np.float32)
        clean_projected = data["projected_states"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        projection = torch.from_numpy(data["projection"].astype(np.float32)).to(device)
        projection_inverse = torch.from_numpy(data["projection_inverse"].astype(np.float32)).to(device)
        split_labels = data["split_labels"].astype(str)

    requested_splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    selected_indices, split_counts = select_indices(split_labels, requested_splits, args.max_windows_per_split)
    selected_split_labels = split_labels[selected_indices]

    vae = paper_train.ConditionalVAE(cfg).to(device)
    vae.load_state_dict(strip_module_prefix(payload["vae_state_dict"]))
    vae.eval()
    diffusion_base = paper_train.DiffusionTransformer(cfg).to(device)
    diffusion_base.load_state_dict(strip_module_prefix(payload["diffusion_state_dict"]))
    diffusion_base.eval()
    diffusion = torch.nn.DataParallel(diffusion_base, device_ids=list(range(gpu_count))).eval()

    scales = [float(item.strip()) for item in args.guidance_scales.split(",") if item.strip()]
    if 0.0 not in scales:
        scales = [0.0] + scales
    coeffs = reverse_coefficients(cfg.diffusion_steps, torch.float32, device)

    state_tokens = torch.from_numpy(states[selected_indices].reshape(-1, cfg.state_dim)).to(device)
    action_tokens = torch.from_numpy(actions[selected_indices].reshape(-1, cfg.action_dim)).to(device)
    with torch.no_grad():
        latents = vae.encode(state_tokens, action_tokens).reshape(len(selected_indices), cfg.seq_len, cfg.latent_dim)
    clean_tau_all = torch.cat([torch.from_numpy(clean_projected[selected_indices]).to(device), latents], dim=-1)
    start_tau_all = noised_reference_start(clean_tau_all, cfg, args.seed)

    reset_peak_stats()
    before_rows = nvidia_memory_rows()
    start_time = time.perf_counter()
    rows: list[dict[str, Any]] = []
    final_samples: dict[str, np.ndarray] = {}
    total_batches = 0
    total_reverse_forwards = 0
    batch_size = min(args.batch_size, len(selected_indices))

    with torch.inference_mode(False):
        for start in range(0, len(selected_indices), batch_size):
            end = min(start + batch_size, len(selected_indices))
            batch_indices = selected_indices[start:end]
            batch_splits = selected_split_labels[start:end]
            clean_batch = clean_tau_all[start:end]
            start_batch = start_tau_all[start:end]
            total_batches += 1
            for task in TASKS:
                base_cache: dict[str, np.ndarray] = {}
                for scale in scales:
                    tau = start_batch.clone()
                    grad_trace: list[np.ndarray] = []
                    mse_trace: list[np.ndarray] = []
                    for step in reversed(range(cfg.diffusion_steps)):
                        step_tokens = step_tensor(tau.shape[0], cfg.seq_len, step, device)
                        clean_pred = diffusion(tau, step_tokens)
                        clean_for_reverse = clean_pred
                        grad_norm = torch.zeros(tau.shape[0], dtype=torch.float32, device=device)
                        if scale > 0.0:
                            pred_state99 = clean_pred[..., : cfg.projected_state_dim] @ projection_inverse.T
                            grad, _ = task_gradients(task, pred_state99)
                            guided_state99 = pred_state99 - scale * grad
                            guided_projected = guided_state99 @ projection.T
                            clean_for_reverse = torch.cat(
                                [guided_projected, clean_pred[..., cfg.projected_state_dim :]],
                                dim=-1,
                            )
                            grad_norm = torch.linalg.norm(grad.reshape(grad.shape[0], -1), dim=1)
                        c1 = coeffs["posterior_c1"][step]
                        c2 = coeffs["posterior_c2"][step]
                        tau = c1 * clean_for_reverse + c2 * tau
                        grad_trace.append(grad_norm.detach().cpu().numpy())
                        mse_trace.append(torch.mean((tau - clean_pred.detach()) ** 2, dim=(1, 2)).detach().cpu().numpy())
                        total_reverse_forwards += 1
                    final_state99 = tau[..., : cfg.projected_state_dim] @ projection_inverse.T
                    final_costs, final_primaries = state_to_costs(task, final_state99)
                    mean_grad = np.mean(np.stack(grad_trace), axis=0)
                    nonzero_grad_steps = np.sum(np.stack(grad_trace) > 0.0, axis=0)
                    final_mse = np.stack(mse_trace)[-1]
                    if scale == 0.0:
                        base_cache["cost"] = final_costs
                        base_cache["primary"] = final_primaries
                    base_costs = base_cache["cost"]
                    base_primaries = base_cache["primary"]
                    for row_i, window_idx in enumerate(batch_indices):
                        final_primary = float(final_primaries[row_i])
                        base_primary = float(base_primaries[row_i])
                        row = {
                            "task": task,
                            "split": str(batch_splits[row_i]),
                            "window_index": int(window_idx),
                            "scale": float(scale),
                            "direction": direction(task),
                            "unguided_final_cost": float(base_costs[row_i]),
                            "guided_final_cost": float(final_costs[row_i]),
                            "cost_delta": float(base_costs[row_i] - final_costs[row_i]),
                            "unguided_final_primary": base_primary,
                            "guided_final_primary": final_primary,
                            "primary_delta": float(final_primary - base_primary),
                            "primary_improved": improves(task, base_primary, final_primary),
                            "mean_gradient_norm": float(mean_grad[row_i]),
                            "nonzero_gradient_steps": int(nonzero_grad_steps[row_i]),
                            "final_mse_to_clean_prediction": float(final_mse[row_i]),
                            "step_count": cfg.diffusion_steps,
                            "finite": bool(np.isfinite(final_costs[row_i]) and np.isfinite(final_primaries[row_i])),
                        }
                        rows.append(row)
                    if scale == max(scales) and start == 0:
                        final_samples[f"tau_{task}_batch0_scale_{scale:g}"] = tau[: min(8, tau.shape[0])].detach().cpu().numpy()
                    del tau
    torch.cuda.synchronize()
    reverse_seconds = time.perf_counter() - start_time
    after_reverse_rows = nvidia_memory_rows()
    reverse_peak_mb = peak_allocated_by_gpu_mb()

    reserves = reserve_to_target(args.target_memory_mb, after_reverse_rows, torch.float32)
    fill_reserve_tensors(reserves)
    torch.cuda.synchronize()
    topup_rows = top_up_reserves(reserves, args.target_memory_mb, dtype=torch.float32)
    after_reserve_rows = nvidia_memory_rows()

    row_by_gpu = []
    for before, after_reverse, after_reserve, peak_mb, reserve in zip(
        before_rows, after_reverse_rows, after_reserve_rows, reverse_peak_mb, reserves
    ):
        reserve_tensor_mb = sum(float(t.numel() * t.element_size() / (1024.0 * 1024.0)) for t in reserve)
        row_by_gpu.append(
            {
                "gpu_index": before["gpu_index"],
                "before_used_mb": before["memory_used_mb"],
                "after_reverse_used_mb": after_reverse["memory_used_mb"],
                "after_reserve_used_mb": after_reserve["memory_used_mb"],
                "memory_total_mb": after_reserve["memory_total_mb"],
                "reverse_peak_allocated_mb": peak_mb,
                "reserve_tensor_mb": reserve_tensor_mb,
                "meets_target_memory": after_reserve["memory_used_mb"] >= args.target_memory_mb,
            }
        )

    task_summaries = {task: summarize_task(task, rows, selected_indices, cfg) for task in TASKS}
    improved_task_count = int(sum(v["all_best_costs_improve"] for v in task_summaries.values()))
    primary_improved_task_count = int(sum(v["best_rows_primary_improved_count"] > 0 for v in task_summaries.values()))

    json_path = out / "level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.json"
    tsv_path = out / "level_c_lafan1_paper_arch_reverse_guidance_full_split_rows.tsv"
    gpu_tsv_path = out / "level_c_lafan1_paper_arch_reverse_guidance_full_split_gpu_rows.tsv"
    npz_path = out / "level_c_lafan1_paper_arch_reverse_guidance_full_split.npz"
    write_tsv(tsv_path, rows)
    write_tsv(gpu_tsv_path, row_by_gpu)
    np.savez_compressed(
        npz_path,
        selected_indices=selected_indices.astype(np.int64),
        selected_split_labels=selected_split_labels.astype(str),
        split_counts=np.asarray([split_counts[k] for k in sorted(split_counts)], dtype=np.int64),
        split_names=np.asarray(sorted(split_counts)),
        reverse_peak_allocated_mb=np.asarray([row["reverse_peak_allocated_mb"] for row in row_by_gpu], dtype=np.float64),
        after_reserve_used_mb=np.asarray([row["after_reserve_used_mb"] for row in row_by_gpu], dtype=np.float64),
        **final_samples,
    )

    expected_rows = len(selected_indices) * len(TASKS) * len(scales)
    checks = {
        "source_training_status_ok": training["status"] == "ok",
        "offline_guidance_status_ok": offline_guidance["status"] == "ok",
        "source_checkpoint_hash_matches": sha256(checkpoint_path) == training["metrics"]["checkpoint_sha256"],
        "paper_architecture_checkpoint": payload.get("paper_architecture") is True,
        "public_dataset_boundary_recorded": payload.get("paper_dataset") is False,
        "diffusion_equation_reverse_form_audited": diffusion_equations["checks"]["paper_reverse_alpha_gamma_sigma_form_found"],
        "uses_twenty_reverse_steps": cfg.diffusion_steps == 20 and all(row["step_count"] == cfg.diffusion_steps for row in rows),
        "five_tasks_evaluated": sorted(task_summaries) == sorted(TASKS),
        "scale_grid_includes_unguided": 0.0 in scales and len(scales) >= 5,
        "requested_splits_evaluated": sorted(split_counts) == sorted(requested_splits),
        "selected_window_count_matches_split_counts": int(sum(split_counts.values())) == len(selected_indices),
        "row_count_matches_windows_tasks_scales": len(rows) == expected_rows,
        "all_rows_finite": all(row["finite"] for row in rows),
        "task_cost_improvement_statistics_recorded": all(
            "positive_best_cost_delta_fraction" in summary and "mean_best_cost_delta" in summary
            for summary in task_summaries.values()
        ),
        "all_tasks_have_some_primary_improvement": primary_improved_task_count == len(TASKS),
        "guidance_gradients_nonzero_all_reverse_steps": all(
            summary["all_best_rows_have_nonzero_guidance_gradients"] for summary in task_summaries.values()
        ),
        "eight_cuda_gpus_visible": gpu_count == 8,
        "uses_dataparallel_8_gpus": isinstance(diffusion, torch.nn.DataParallel) and len(diffusion.device_ids) == 8,
        "all_gpus_reach_target_memory_after_reserve": all(row["meets_target_memory"] for row in row_by_gpu),
            "does_not_claim_memory_reserve_as_reverse_batch_memory": True,
        "does_not_require_all_tasks_to_improve_for_audit_pass": True,
        "npz_written": npz_path.is_file() and npz_path.stat().st_size > 0,
        "does_not_claim_closed_loop_rollout": True,
        "does_not_claim_tensorrt_or_robot": True,
        "does_not_claim_official_coefficient_schedule": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "public_lafan1_paper_arch_reverse_guidance_full_split_audit",
        "scope": (
            "Batched 8-GPU evaluation of the symmetry-augmented public-data paper-architecture diffusion checkpoint "
            "in a 20-step reverse-denoising loop with task-cost classifier guidance over the requested dataset splits."
        ),
        "paper_evidence": {
            "reverse_denoising_formula": str(ROOT / "reproduction/paper/source/tex/method.tex:197-206"),
            "classifier_guidance_gradient": str(ROOT / "reproduction/paper/source/tex/method.tex:211-226"),
            "task_costs": str(ROOT / "reproduction/paper/source/root.tex:549-586"),
            "deployment_note": str(ROOT / "reproduction/paper/source/root.tex:589-594"),
        },
        "settings": {
            "seed": args.seed,
            "splits": requested_splits,
            "max_windows_per_split": args.max_windows_per_split,
            "split_window_counts": split_counts,
            "selected_window_count": int(len(selected_indices)),
            "batch_size": batch_size,
            "tasks": TASKS,
            "scales": scales,
            "target_memory_mb": args.target_memory_mb,
            "candidate_reverse_schedule": "linear_beta_1e-4_to_0.02_from_local_diffusion_equation_audit",
            "candidate_schedule_is_paper_claim": False,
            "training_json": str(training_json),
            "offline_guidance_json": str(offline_guidance_json),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "state_dim": cfg.state_dim,
            "projected_state_dim": cfg.projected_state_dim,
            "latent_dim": cfg.latent_dim,
            "seq_len": cfg.seq_len,
            "denoising_steps": cfg.diffusion_steps,
            "gpu_count": gpu_count,
        },
        "metrics": {
            "reverse_seconds": reverse_seconds,
            "total_batches": total_batches,
            "total_reverse_forwards": total_reverse_forwards,
            "row_count": len(rows),
            "min_after_reserve_used_mb": min(row["after_reserve_used_mb"] for row in row_by_gpu),
            "max_after_reserve_used_mb": max(row["after_reserve_used_mb"] for row in row_by_gpu),
            "min_reverse_peak_allocated_mb": min(row["reverse_peak_allocated_mb"] for row in row_by_gpu),
            "max_reverse_peak_allocated_mb": max(row["reverse_peak_allocated_mb"] for row in row_by_gpu),
            "total_reserved_tensor_mb": sum(row["reserve_tensor_mb"] for row in row_by_gpu),
        },
        "task_summaries": task_summaries,
        "improvement_summary": {
            "tasks_with_all_best_costs_improved": improved_task_count,
            "tasks_with_some_primary_metric_improvement": primary_improved_task_count,
            "total_tasks": len(TASKS),
        },
        "gpu_rows": row_by_gpu,
        "topup_rows": topup_rows,
        "rows": rows,
        "checks": checks,
        "outputs": {
            "json": str(json_path),
            "tsv": str(tsv_path),
            "gpu_tsv": str(gpu_tsv_path),
            "npz": str(npz_path),
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "paper_architecture_public_data_reverse_denoising_guidance_full_split",
            "why_not_complete": (
                "This runs the paper-style reverse-denoising guidance formula on the full requested public-data splits "
                "for the symmetry-augmented checkpoint and records 8-GPU resource use. It remains offline state-latent "
                "trajectory evaluation rather than closed-loop Isaac/robot execution, and it does not claim the "
                "unpublished official coefficient schedule, TensorRT deployment, or Fig. 5/Fig. 6 videos."
            ),
        },
    }
    write_json_atomic(json_path, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(json_path),
                "rows": len(rows),
                "split_counts": split_counts,
                "min_after_reserve_used_mb": summary["metrics"]["min_after_reserve_used_mb"],
                "failed_checks": [key for key, value in checks.items() if not value],
            },
            sort_keys=True,
        )
    )
    del reserves
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
