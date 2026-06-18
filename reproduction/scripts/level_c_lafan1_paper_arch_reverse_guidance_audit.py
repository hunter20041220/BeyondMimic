#!/usr/bin/env python3
"""Full-checkpoint guided reverse-denoising audit for public LAFAN1 paper architecture."""

from __future__ import annotations

import argparse
import csv
import hashlib
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
from level_c_lafan1_paper_arch_guidance_eval import (  # noqa: E402
    TASKS,
    direction,
    improves,
    primary_metric,
    task_cost,
)


OUT = ROOT / "res/level_c/lafan1_paper_arch_reverse_guidance"
TRAINING_JSON = (
    ROOT
    / "res/level_c/lafan1_paper_arch_vae_diffusion_training/"
    / "lafan1_paper_arch_vae_diffusion_training.json"
)
OFFLINE_GUIDANCE_JSON = (
    ROOT
    / "res/level_c/lafan1_paper_arch_guidance_eval/"
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


def reverse_coefficients(steps: int, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    betas = torch.linspace(1e-4, 0.02, steps, dtype=dtype)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    prev_alpha_bars = torch.cat([torch.ones(1, dtype=dtype), alpha_bars[:-1]])
    c1 = betas * torch.sqrt(prev_alpha_bars) / (1.0 - alpha_bars)
    c2 = (1.0 - prev_alpha_bars) * torch.sqrt(alphas) / (1.0 - alpha_bars)
    posterior_variance = betas * (1.0 - prev_alpha_bars) / (1.0 - alpha_bars)
    return {
        "betas": betas,
        "alpha_bars": alpha_bars,
        "posterior_c1": c1,
        "posterior_c2": c2,
        "posterior_sigma": torch.sqrt(torch.clamp(posterior_variance, min=0.0)),
    }


def noised_reference_start(clean_tau: torch.Tensor, cfg: paper_train.TrainConfig, seed: int) -> torch.Tensor:
    bars = paper_train.alpha_bars(cfg.diffusion_steps, torch.device("cpu"), clean_tau.dtype)
    alpha = bars[-1]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(clean_tau.shape, dtype=clean_tau.dtype, generator=generator)
    return torch.sqrt(alpha) * clean_tau + torch.sqrt(1.0 - alpha) * noise


def step_tensor(batch: int, seq_len: int, step: int) -> torch.Tensor:
    return torch.full((batch, seq_len, 2), step, dtype=torch.long)


def project_state(state99: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    return state99 @ projection.T


def tau_to_state99(tau: torch.Tensor, cfg: paper_train.TrainConfig, projection_inverse: torch.Tensor) -> torch.Tensor:
    return tau[..., : cfg.projected_state_dim] @ projection_inverse.T


def guided_clean_prediction(
    clean_pred: torch.Tensor,
    task: str,
    cfg: paper_train.TrainConfig,
    projection: torch.Tensor,
    projection_inverse: torch.Tensor,
    guidance_scale: float,
) -> tuple[torch.Tensor, float, float, float]:
    state99 = tau_to_state99(clean_pred, cfg, projection_inverse).detach().clone().requires_grad_(True)
    cost = task_cost(task, state99[0])
    cost.backward()
    grad = state99.grad.detach()
    guided_state99 = (state99 - guidance_scale * grad).detach()
    guided_projected = project_state(guided_state99, projection)
    guided_tau = torch.cat([guided_projected, clean_pred[..., cfg.projected_state_dim :].detach()], dim=-1)
    guided_cost = float(task_cost(task, guided_state99[0]).detach().cpu())
    grad_norm = float(torch.linalg.norm(grad).detach().cpu())
    state_delta = float(torch.mean((guided_state99 - state99.detach()) ** 2).detach().cpu())
    return guided_tau, float(cost.detach().cpu()), guided_cost, grad_norm + state_delta * 0.0


def reverse_loop(
    diffusion: paper_train.DiffusionTransformer,
    start_tau: torch.Tensor,
    task: str,
    cfg: paper_train.TrainConfig,
    projection: torch.Tensor,
    projection_inverse: torch.Tensor,
    guidance_scale: float,
    stochastic: bool,
    seed: int,
) -> dict[str, Any]:
    coeffs = reverse_coefficients(cfg.diffusion_steps, start_tau.dtype)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    tau = start_tau.clone()
    cost_trace: list[float] = []
    primary_trace: list[float] = []
    mse_to_clean_pred_trace: list[float] = []
    grad_norm_trace: list[float] = []
    step_trace: list[int] = []

    for step in reversed(range(cfg.diffusion_steps)):
        steps = step_tensor(tau.shape[0], tau.shape[1], step)
        clean_pred = diffusion(tau, steps)
        clean_for_reverse = clean_pred
        raw_clean_state = tau_to_state99(clean_pred, cfg, projection_inverse)[0].detach()
        raw_cost = float(task_cost(task, raw_clean_state).detach().cpu())
        raw_primary = float(primary_metric(task, raw_clean_state).detach().cpu())
        grad_norm = 0.0
        if guidance_scale > 0.0:
            clean_for_reverse, raw_cost, _, grad_norm = guided_clean_prediction(
                clean_pred,
                task,
                cfg,
                projection,
                projection_inverse,
                guidance_scale,
            )
        c1 = coeffs["posterior_c1"][step]
        c2 = coeffs["posterior_c2"][step]
        mean = c1 * clean_for_reverse + c2 * tau
        if stochastic and step > 0:
            noise = torch.randn(tau.shape, dtype=tau.dtype, generator=generator)
            tau = mean + coeffs["posterior_sigma"][step] * noise
        else:
            tau = mean
        final_state = tau_to_state99(tau, cfg, projection_inverse)[0].detach()
        cost_trace.append(float(task_cost(task, final_state).detach().cpu()))
        primary_trace.append(float(primary_metric(task, final_state).detach().cpu()))
        mse_to_clean_pred_trace.append(float(torch.mean((tau - clean_pred.detach()) ** 2).detach().cpu()))
        grad_norm_trace.append(grad_norm)
        step_trace.append(step)
        _ = raw_cost
        _ = raw_primary

    final_state99 = tau_to_state99(tau, cfg, projection_inverse)[0].detach()
    return {
        "final_tau": tau.detach(),
        "final_state99": final_state99,
        "cost_trace": cost_trace,
        "primary_trace": primary_trace,
        "mse_to_clean_pred_trace": mse_to_clean_pred_trace,
        "gradient_norm_trace": grad_norm_trace,
        "step_trace": step_trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-json",
        default=None,
        help="Training summary JSON to evaluate. Defaults to the original public-LAFAN1 paper-architecture run.",
    )
    parser.add_argument(
        "--offline-guidance-json",
        default=None,
        help="Offline guidance JSON paired with the training checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for JSON/TSV/NPZ. Defaults to the original reverse-guidance directory.",
    )
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--guidance-scales", default="0.0,0.00002,0.00005,0.0001,0.0002")
    parser.add_argument("--start-mode", choices=["gaussian", "noised_reference"], default="noised_reference")
    parser.add_argument("--stochastic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260621)
    args = parser.parse_args()

    out = resolve_path(args.output_dir, OUT)
    training_json = resolve_path(args.training_json, TRAINING_JSON)
    offline_guidance_json = resolve_path(args.offline_guidance_json, OFFLINE_GUIDANCE_JSON)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    training = load_json(training_json)
    offline_guidance = load_json(offline_guidance_json)
    diffusion_equations = load_json(DIFFUSION_EQUATION_JSON)
    checkpoint_path = Path(training["outputs"]["checkpoint"])
    dataset_path = Path(training["outputs"]["dataset_npz"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = paper_train.TrainConfig(**payload["config"])
    diffusion = paper_train.DiffusionTransformer(cfg)
    diffusion.load_state_dict(strip_module_prefix(payload["diffusion_state_dict"]))
    diffusion.eval()
    scales = [float(item.strip()) for item in args.guidance_scales.split(",") if item.strip()]
    if 0.0 not in scales:
        scales = [0.0] + scales

    with np.load(dataset_path, allow_pickle=True) as data:
        states = data["states"].astype(np.float32)
        clean_projected = data["projected_states"].astype(np.float32)
        actions = data["actions"].astype(np.float32)
        projection = torch.from_numpy(data["projection"].astype(np.float32))
        projection_inverse = torch.from_numpy(data["projection_inverse"].astype(np.float32))
        split_labels = data["split_labels"].astype(str)
    validation_indices = np.nonzero(split_labels == "validation")[0][: args.windows]
    if len(validation_indices) != args.windows:
        raise RuntimeError("not enough validation windows")

    vae = paper_train.ConditionalVAE(cfg)
    vae.load_state_dict(strip_module_prefix(payload["vae_state_dict"]))
    vae.eval()
    with torch.no_grad():
        selected_state_tokens = torch.from_numpy(states[validation_indices].reshape(-1, cfg.state_dim))
        selected_action_tokens = torch.from_numpy(actions[validation_indices].reshape(-1, cfg.action_dim))
        selected_latents = vae.encode(selected_state_tokens, selected_action_tokens).reshape(
            len(validation_indices), cfg.seq_len, cfg.latent_dim
        )
    selected_clean_tau = torch.cat([torch.from_numpy(clean_projected[validation_indices]), selected_latents], dim=-1)

    rows: list[dict[str, Any]] = []
    final_tau_arrays: dict[str, np.ndarray] = {}
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    for local_i, window_idx in enumerate(validation_indices):
        if args.start_mode == "gaussian":
            start_tau = torch.randn((1, cfg.seq_len, cfg.token_dim), dtype=torch.float32, generator=rng)
        else:
            start_tau = noised_reference_start(selected_clean_tau[local_i : local_i + 1], cfg, args.seed + local_i)
        base_results: dict[str, dict[str, Any]] = {}
        for task in TASKS:
            for scale in scales:
                result = reverse_loop(
                    diffusion,
                    start_tau,
                    task,
                    cfg,
                    projection,
                    projection_inverse,
                    scale,
                    args.stochastic,
                    args.seed + local_i * 1000 + int(scale * 1e8),
                )
                final_state = result["final_state99"]
                final_cost = float(task_cost(task, final_state).detach().cpu())
                final_primary = float(primary_metric(task, final_state).detach().cpu())
                if scale == 0.0:
                    base_results[task] = {"cost": final_cost, "primary": final_primary}
                base = base_results[task]
                row = {
                    "task": task,
                    "window_index": int(window_idx),
                    "scale": scale,
                    "direction": direction(task),
                    "unguided_final_cost": base["cost"],
                    "guided_final_cost": final_cost,
                    "cost_delta": base["cost"] - final_cost,
                    "unguided_final_primary": base["primary"],
                    "guided_final_primary": final_primary,
                    "primary_delta": final_primary - base["primary"],
                    "primary_improved": improves(task, base["primary"], final_primary),
                    "mean_gradient_norm": float(np.mean(result["gradient_norm_trace"])),
                    "nonzero_gradient_steps": int(sum(v > 0.0 for v in result["gradient_norm_trace"])),
                    "final_mse_to_clean_prediction": result["mse_to_clean_pred_trace"][-1],
                    "cost_trace_first": result["cost_trace"][0],
                    "cost_trace_last": result["cost_trace"][-1],
                    "step_count": len(result["step_trace"]),
                    "finite": bool(np.isfinite(final_cost) and np.isfinite(final_primary)),
                }
                rows.append(row)
                if scale == max(scales):
                    final_tau_arrays[f"tau_{task}_window_{int(window_idx)}_scale_{scale:g}"] = (
                        result["final_tau"].detach().numpy().astype(np.float32)
                    )

    task_summaries: dict[str, Any] = {}
    for task in TASKS:
        task_rows = [row for row in rows if row["task"] == task and row["scale"] > 0.0]
        best_rows = []
        for window_idx in validation_indices:
            candidates = [row for row in task_rows if row["window_index"] == int(window_idx)]
            best_rows.append(min(candidates, key=lambda row: row["guided_final_cost"]))
        task_summaries[task] = {
            "window_count": int(len(validation_indices)),
            "scale_count": int(len(scales)),
            "mean_best_cost_delta": float(np.mean([row["cost_delta"] for row in best_rows])),
            "mean_best_gradient_norm": float(np.mean([row["mean_gradient_norm"] for row in best_rows])),
            "best_rows_primary_improved_count": int(sum(row["primary_improved"] for row in best_rows)),
            "all_best_costs_improve": bool(all(row["cost_delta"] > 0.0 for row in best_rows)),
            "all_best_rows_have_nonzero_guidance_gradients": bool(
                all(row["nonzero_gradient_steps"] == cfg.diffusion_steps for row in best_rows)
            ),
        }

    improved_task_count = int(sum(v["all_best_costs_improve"] for v in task_summaries.values()))
    primary_improved_task_count = int(
        sum(v["best_rows_primary_improved_count"] > 0 for v in task_summaries.values())
    )
    npz_path = out / "level_c_lafan1_paper_arch_reverse_guidance.npz"
    json_path = out / "level_c_lafan1_paper_arch_reverse_guidance_audit.json"
    tsv_path = out / "level_c_lafan1_paper_arch_reverse_guidance_rows.tsv"
    np.savez_compressed(
        npz_path,
        validation_indices=validation_indices.astype(np.int64),
        clean_projected_reference=clean_projected[validation_indices].astype(np.float32),
        **final_tau_arrays,
    )
    write_tsv(tsv_path, rows)
    checks = {
        "source_training_status_ok": training["status"] == "ok",
        "offline_guidance_status_ok": offline_guidance["status"] == "ok",
        "source_checkpoint_hash_matches": sha256(checkpoint_path) == training["metrics"]["checkpoint_sha256"],
        "paper_architecture_checkpoint": payload.get("paper_architecture") is True,
        "public_dataset_boundary_recorded": payload.get("paper_dataset") is False,
        "diffusion_equation_reverse_form_audited": diffusion_equations["checks"]["paper_reverse_alpha_gamma_sigma_form_found"],
        "uses_twenty_reverse_steps": cfg.diffusion_steps == 20
        and all(row["step_count"] == cfg.diffusion_steps for row in rows),
        "five_tasks_evaluated": sorted(task_summaries) == sorted(TASKS),
        "scale_grid_includes_unguided": 0.0 in scales and len(scales) >= 5,
        "all_rows_finite": all(row["finite"] for row in rows),
        "at_least_three_tasks_best_costs_improve": improved_task_count >= 3,
        "all_tasks_have_some_primary_improvement": primary_improved_task_count == len(TASKS),
        "guidance_gradients_nonzero_all_reverse_steps": all(
            summary["all_best_rows_have_nonzero_guidance_gradients"] for summary in task_summaries.values()
        ),
        "npz_written": npz_path.is_file() and npz_path.stat().st_size > 0,
        "does_not_claim_closed_loop_rollout": True,
        "does_not_claim_tensorrt_or_robot": True,
        "does_not_claim_official_coefficient_schedule": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "public_lafan1_paper_arch_reverse_guidance_audit",
        "scope": (
            "Full public-data paper-architecture diffusion checkpoint evaluated in a 20-step reverse-denoising loop "
            "with classifier-guidance task-cost gradients applied inside each denoising iteration."
        ),
        "paper_evidence": {
            "reverse_denoising_formula": str(ROOT / "reproduction/paper/source/tex/method.tex:197-206"),
            "classifier_guidance_gradient": str(ROOT / "reproduction/paper/source/tex/method.tex:211-226"),
            "task_costs": str(ROOT / "reproduction/paper/source/root.tex:549-586"),
            "deployment_note": str(ROOT / "reproduction/paper/source/root.tex:589-594"),
        },
        "settings": {
            "seed": args.seed,
            "windows": int(args.windows),
            "validation_window_indices": validation_indices.tolist(),
            "start_mode": args.start_mode,
            "tasks": TASKS,
            "scales": scales,
            "stochastic_reverse_noise": args.stochastic,
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
        },
        "task_summaries": task_summaries,
        "improvement_summary": {
            "tasks_with_all_best_costs_improved": improved_task_count,
            "tasks_with_some_primary_metric_improvement": primary_improved_task_count,
            "total_tasks": len(TASKS),
        },
        "row_count": len(rows),
        "rows": rows,
        "checks": checks,
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "paper_architecture_public_data_reverse_denoising_guidance",
            "why_not_complete": (
                "This uses the full trained public-data diffusion checkpoint and applies cost gradients inside a "
                "20-step reverse loop, which is closer to the paper inference algorithm than one-shot offline guidance. "
                "The audit records whether task costs improve under the chosen offline start mode rather than forcing "
                "a success claim. It still is not closed-loop Isaac/robot execution, does not use the unpublished "
                "official coefficient schedule, and does not reproduce TensorRT/asynchronous deployment or Fig. 5/Fig. 6 videos."
            ),
        },
    }
    write_json_atomic(json_path, summary)
    print(json.dumps({"status": summary["status"], "rows": len(rows), "json": str(json_path)}, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
