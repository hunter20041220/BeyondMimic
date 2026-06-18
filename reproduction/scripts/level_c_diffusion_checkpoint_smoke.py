#!/usr/bin/env python3
"""Debug-only diffusion Transformer checkpoint save/load/resume smoke."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from level_c_paper_state_transformer_arch_probe import (
    PAPER_STATE_JSON,
    PAPER_STATE_NPZ,
    PaperStateDiffusionTransformer,
    ProbeConfig,
    diffusion_alpha_bars,
    grad_norm,
    load_paper_state_tau,
    parameter_count,
    seed_everything,
)
from level_c_training_schedule_probe import DiffusionTrainingScheduleConfig, ema_decay_at_step, learning_rate_at_step
from level_c_transformer_ema_smoke import clone_state, ema_update, mean_state_l2_delta, state_digest


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/diffusion_checkpoint_smoke"


def optimizer_digest(optimizer: torch.optim.Optimizer) -> str:
    digest = hashlib.sha256()
    state = optimizer.state_dict()
    digest.update(json.dumps(state["param_groups"], sort_keys=True, default=str).encode("utf-8"))
    for param_id, values in sorted(state["state"].items()):
        digest.update(str(param_id).encode("utf-8"))
        for key, value in sorted(values.items()):
            digest.update(str(key).encode("utf-8"))
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu().contiguous().numpy()
                digest.update(json.dumps(list(tensor.shape)).encode("utf-8"))
                digest.update(str(tensor.dtype).encode("utf-8"))
                digest.update(tensor.tobytes())
            else:
                digest.update(str(value).encode("utf-8"))
    return digest.hexdigest()


def make_batch(
    clean_tau: torch.Tensor,
    state_dim: int,
    cfg: ProbeConfig,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    seed_everything(seed)
    alpha_bars = diffusion_alpha_bars(cfg.denoising_steps, device, clean_tau.dtype)
    steps = torch.randint(0, cfg.denoising_steps, (cfg.batch_size, cfg.sequence_length, 2), device=device)
    noise = torch.randn_like(clean_tau)
    state_alpha = alpha_bars[steps[..., 0]].unsqueeze(-1).expand(-1, -1, state_dim)
    latent_alpha = alpha_bars[steps[..., 1]].unsqueeze(-1).expand(-1, -1, cfg.latent_dim)
    alpha = torch.cat([state_alpha, latent_alpha], dim=-1)
    noisy_tau = torch.sqrt(alpha) * clean_tau + torch.sqrt(1.0 - alpha) * noise
    return noisy_tau, steps


def train_step(
    model: PaperStateDiffusionTransformer,
    optimizer: torch.optim.Optimizer,
    ema_shadow: dict[str, torch.Tensor],
    clean_tau: torch.Tensor,
    state_dim: int,
    cfg: ProbeConfig,
    schedule: DiffusionTrainingScheduleConfig,
    step: int,
    device: torch.device,
) -> dict[str, float]:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate_at_step(step, schedule)
    noisy_tau, denoising_steps = make_batch(clean_tau, state_dim, cfg, device, cfg.seed + 200 + step)
    with torch.no_grad():
        loss_before = F.mse_loss(model(noisy_tau, denoising_steps), clean_tau)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(noisy_tau, denoising_steps)
    loss = F.mse_loss(prediction, clean_tau)
    loss.backward()
    total_grad_norm = grad_norm(model)
    optimizer.step()
    ema_update(ema_shadow, model, ema_decay_at_step(step, schedule))
    with torch.no_grad():
        loss_after = F.mse_loss(model(noisy_tau, denoising_steps), clean_tau)
    return {
        "step": float(step),
        "learning_rate": float(learning_rate_at_step(step, schedule)),
        "ema_decay": float(ema_decay_at_step(step, schedule)),
        "loss_before": float(loss_before.detach().cpu().item()),
        "loss_after": float(loss_after.detach().cpu().item()),
        "grad_norm": float(total_grad_norm),
    }


def init_training_objects(
    cfg: ProbeConfig,
    token_dim: int,
    schedule: DiffusionTrainingScheduleConfig,
    device: torch.device,
) -> tuple[PaperStateDiffusionTransformer, torch.optim.Optimizer, dict[str, torch.Tensor]]:
    seed_everything(cfg.seed)
    model = PaperStateDiffusionTransformer(token_dim=token_dim, cfg=cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=schedule.learning_rate, weight_decay=schedule.weight_decay)
    ema_shadow = clone_state(model)
    return model, optimizer, ema_shadow


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "phase",
        "step",
        "learning_rate",
        "ema_decay",
        "loss_before",
        "loss_after",
        "grad_norm",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    cfg = ProbeConfig(batch_size=1, seed=20260904)
    schedule = DiffusionTrainingScheduleConfig(probe_total_gradient_steps=50000)
    namespace = argparse.Namespace(
        paper_state_json=PAPER_STATE_JSON,
        paper_state_npz=PAPER_STATE_NPZ,
        motion_key="walk1_subject1_frames_1_180_state_fixture_paper_state_windows",
    )
    clean_tau, state_dim, paper_state_summary = load_paper_state_tau(namespace, cfg, device)
    token_dim = int(clean_tau.shape[-1])

    baseline_model, baseline_optimizer, baseline_ema = init_training_objects(cfg, token_dim, schedule, device)
    baseline_rows = []
    for step in [0, 1]:
        row = train_step(baseline_model, baseline_optimizer, baseline_ema, clean_tau, state_dim, cfg, schedule, step, device)
        baseline_rows.append({"phase": "uninterrupted", **row})

    resumed_model, resumed_optimizer, resumed_ema = init_training_objects(cfg, token_dim, schedule, device)
    first_row = train_step(resumed_model, resumed_optimizer, resumed_ema, clean_tau, state_dim, cfg, schedule, 0, device)
    checkpoint_path = OUT / "debug_diffusion_transformer_checkpoint_smoke.pt"
    checkpoint_payload = {
        "experiment_type": "debug_only_diffusion_checkpoint_smoke",
        "paper_evidence": {
            "paper_state_windows": str(ROOT / "res/level_c/paper_state_windows/level_c_paper_state_windows.json"),
            "diffusion_hyperparameters": str(ROOT / "reproduction/paper/source/root.tex:827-856"),
            "clean_trajectory_loss": str(ROOT / "reproduction/paper/source/tex/method.tex:187-196"),
            "goal_checkpoint_requirement": str(ROOT / "goal.md:1251-1290,1468-1487,1825"),
        },
        "config": asdict(cfg),
        "schedule": asdict(schedule),
        "model_state_dict": resumed_model.state_dict(),
        "optimizer_state_dict": resumed_optimizer.state_dict(),
        "ema_state_dict": resumed_ema,
        "step": 1,
        "seed": cfg.seed,
        "is_trained_paper_checkpoint": False,
        "is_ema_paper_checkpoint": False,
        "not_a_replacement_for": [
            "trained diffusion Transformer checkpoint",
            "paper EMA checkpoint",
            "true VAE rollout state-latent dataset",
            "TensorRT deployment",
            "Fig. 5/Fig. 6 rollout evaluation",
        ],
    }
    torch.save(checkpoint_payload, checkpoint_path)

    loaded_payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    loaded_cfg = ProbeConfig(**loaded_payload["config"])
    loaded_schedule = DiffusionTrainingScheduleConfig(**loaded_payload["schedule"])
    loaded_model = PaperStateDiffusionTransformer(token_dim=token_dim, cfg=loaded_cfg).to(device)
    loaded_optimizer = torch.optim.AdamW(
        loaded_model.parameters(),
        lr=loaded_schedule.learning_rate,
        weight_decay=loaded_schedule.weight_decay,
    )
    loaded_model.load_state_dict(loaded_payload["model_state_dict"])
    loaded_optimizer.load_state_dict(loaded_payload["optimizer_state_dict"])
    loaded_ema = {name: tensor.detach().clone().to(device) for name, tensor in loaded_payload["ema_state_dict"].items()}
    second_row = train_step(
        loaded_model,
        loaded_optimizer,
        loaded_ema,
        clean_tau,
        state_dim,
        loaded_cfg,
        loaded_schedule,
        1,
        device,
    )
    resumed_rows = [{"phase": "before_save", **first_row}, {"phase": "after_load_resume", **second_row}]

    baseline_state = clone_state(baseline_model)
    loaded_state = clone_state(loaded_model)
    baseline_model_sha = state_digest(baseline_state)
    loaded_model_sha = state_digest(loaded_state)
    baseline_ema_sha = state_digest(baseline_ema)
    loaded_ema_sha = state_digest(loaded_ema)
    baseline_optimizer_sha = optimizer_digest(baseline_optimizer)
    loaded_optimizer_sha = optimizer_digest(loaded_optimizer)
    checkpoint_size = checkpoint_path.stat().st_size if checkpoint_path.exists() else 0
    eval_noisy, eval_steps = make_batch(clean_tau, state_dim, cfg, device, cfg.seed + 999)
    with torch.no_grad():
        baseline_eval = baseline_model(eval_noisy, eval_steps).detach().cpu()
        loaded_eval = loaded_model(eval_noisy, eval_steps).detach().cpu()
    eval_max_abs_error = float(torch.max(torch.abs(baseline_eval - loaded_eval)).item())
    model_l2_after_resume = mean_state_l2_delta(baseline_state, loaded_state)
    ema_l2_after_resume = mean_state_l2_delta(baseline_ema, loaded_ema)

    checks = {
        "checkpoint_file_exists": checkpoint_path.is_file() and checkpoint_size > 0,
        "checkpoint_has_required_keys": all(
            key in loaded_payload
            for key in [
                "config",
                "schedule",
                "model_state_dict",
                "optimizer_state_dict",
                "ema_state_dict",
                "step",
                "seed",
                "is_trained_paper_checkpoint",
            ]
        ),
        "uses_paper_state_token_dim_131": token_dim == 131,
        "uses_paper_transformer_hyperparameters": loaded_cfg.embedding_dim == 512
        and loaded_cfg.attention_heads == 8
        and loaded_cfg.transformer_layers == 6
        and loaded_cfg.denoising_steps == 20,
        "uses_paper_optimizer_hyperparameters": loaded_schedule.learning_rate == 1e-4
        and loaded_schedule.weight_decay == 0.001
        and loaded_schedule.ema_power == 0.75
        and loaded_schedule.ema_max == 0.9999,
        "loaded_model_matches_uninterrupted_after_resume": baseline_model_sha == loaded_model_sha,
        "loaded_ema_matches_uninterrupted_after_resume": baseline_ema_sha == loaded_ema_sha,
        "loaded_optimizer_matches_uninterrupted_after_resume": baseline_optimizer_sha == loaded_optimizer_sha,
        "loaded_eval_prediction_matches_uninterrupted": eval_max_abs_error == 0.0,
        "resume_losses_finite": bool(
            np.all(np.isfinite([row["loss_before"] for row in baseline_rows + resumed_rows]))
            and np.all(np.isfinite([row["loss_after"] for row in baseline_rows + resumed_rows]))
        ),
        "resume_grad_norms_positive": all(row["grad_norm"] > 0.0 for row in baseline_rows + resumed_rows),
        "marks_not_trained_paper_checkpoint": loaded_payload["is_trained_paper_checkpoint"] is False,
        "marks_not_ema_paper_checkpoint": loaded_payload["is_ema_paper_checkpoint"] is False,
        "does_not_claim_true_vae_rollout": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "diffusion_checkpoint_smoke",
        "scope": "debug-only paper-state Transformer checkpoint save/load/resume mechanics",
        "settings": {
            **asdict(cfg),
            "device": str(device),
            "torch_threads": args.torch_threads,
            "token_dim": token_dim,
            "state_dim": state_dim,
            "paper_state_window_count": paper_state_summary["counts"]["window_count"],
            "learning_rate": schedule.learning_rate,
            "weight_decay": schedule.weight_decay,
            "warmup_gradient_steps": schedule.warmup_gradient_steps,
            "ema_power": schedule.ema_power,
            "ema_max": schedule.ema_max,
        },
        "model": {
            "class": "PaperStateDiffusionTransformer",
            "parameter_count": parameter_count(loaded_model),
            "state_dict_tensor_count": len(loaded_model.state_dict()),
        },
        "step_rows": baseline_rows + resumed_rows,
        "metrics": {
            "checkpoint_size_bytes": checkpoint_size,
            "baseline_model_sha256": baseline_model_sha,
            "loaded_model_sha256": loaded_model_sha,
            "baseline_ema_sha256": baseline_ema_sha,
            "loaded_ema_sha256": loaded_ema_sha,
            "baseline_optimizer_sha256": baseline_optimizer_sha,
            "loaded_optimizer_sha256": loaded_optimizer_sha,
            "model_l2_after_resume": model_l2_after_resume,
            "ema_l2_after_resume": ema_l2_after_resume,
            "eval_max_abs_error_after_resume": eval_max_abs_error,
            "final_uninterrupted_loss_after": baseline_rows[-1]["loss_after"],
            "final_resumed_loss_after": resumed_rows[-1]["loss_after"],
        },
        "checks": checks,
        "not_a_replacement_for": loaded_payload["not_a_replacement_for"],
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "partial",
            "why_not_complete": (
                "This saves and reloads a debug checkpoint after one CPU smoke step, then verifies exact resume "
                "agreement with an uninterrupted second step. It is not paper-scale diffusion training, not a "
                "trained Transformer checkpoint, and not a deployable EMA/TensorRT artifact."
            ),
        },
        "outputs": {
            "json": str(OUT / "level_c_diffusion_checkpoint_smoke.json"),
            "tsv": str(OUT / "level_c_diffusion_checkpoint_smoke.tsv"),
            "checkpoint": str(checkpoint_path),
        },
    }
    (OUT / "level_c_diffusion_checkpoint_smoke.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_tsv(OUT / "level_c_diffusion_checkpoint_smoke.tsv", baseline_rows + resumed_rows)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": summary["outputs"]["json"],
                "checkpoint": summary["outputs"]["checkpoint"],
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
