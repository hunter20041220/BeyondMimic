#!/usr/bin/env python3
"""Debug-only AdamW/LR/EMA smoke on the paper-state Transformer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/transformer_ema_smoke"


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(json.dumps(list(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.tobytes())
    return digest.hexdigest()


def clone_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def ema_update(shadow: dict[str, torch.Tensor], model: torch.nn.Module, decay: float) -> None:
    current = model.state_dict()
    for name, value in current.items():
        shadow[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)


def max_state_abs_delta(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    return float(max(torch.max(torch.abs(a[name].detach().cpu() - b[name].detach().cpu())).item() for name in a))


def mean_state_l2_delta(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    total = torch.zeros(())
    count = 0
    for name in a:
        diff = a[name].detach().cpu() - b[name].detach().cpu()
        total = total + diff.pow(2).sum()
        count += diff.numel()
    return float(torch.sqrt(total / max(count, 1)).item())


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "step",
            "learning_rate",
            "ema_decay",
            "loss_before",
            "loss_after",
            "grad_norm",
            "model_vs_initial_l2",
            "ema_vs_model_l2",
        ]
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device)
    cfg = ProbeConfig(batch_size=1, seed=20260904)
    schedule = DiffusionTrainingScheduleConfig(probe_total_gradient_steps=50000)
    seed_everything(cfg.seed)
    namespace = argparse.Namespace(
        paper_state_json=PAPER_STATE_JSON,
        paper_state_npz=PAPER_STATE_NPZ,
        motion_key="walk1_subject1_frames_1_180_state_fixture_paper_state_windows",
    )
    clean_tau, state_dim, _paper_state_summary = load_paper_state_tau(namespace, cfg, device)
    token_dim = int(clean_tau.shape[-1])
    model = PaperStateDiffusionTransformer(token_dim=token_dim, cfg=cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=schedule.learning_rate, weight_decay=schedule.weight_decay)
    alpha_bars = diffusion_alpha_bars(cfg.denoising_steps, device, clean_tau.dtype)

    initial_state = clone_state(model)
    ema_shadow = clone_state(model)
    rows: list[dict[str, Any]] = []
    for step in range(args.steps):
        lr = learning_rate_at_step(step, schedule)
        ema_decay = ema_decay_at_step(step, schedule)
        for group in optimizer.param_groups:
            group["lr"] = lr
        seed_everything(cfg.seed + 10 + step)
        denoising_steps = torch.randint(0, cfg.denoising_steps, (cfg.batch_size, cfg.sequence_length, 2), device=device)
        noise = torch.randn_like(clean_tau)
        state_alpha = alpha_bars[denoising_steps[..., 0]].unsqueeze(-1).expand(-1, -1, state_dim)
        latent_alpha = alpha_bars[denoising_steps[..., 1]].unsqueeze(-1).expand(-1, -1, cfg.latent_dim)
        alpha = torch.cat([state_alpha, latent_alpha], dim=-1)
        noisy_tau = torch.sqrt(alpha) * clean_tau + torch.sqrt(1.0 - alpha) * noise

        with torch.no_grad():
            loss_before = F.mse_loss(model(noisy_tau, denoising_steps), clean_tau)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(noisy_tau, denoising_steps)
        loss = F.mse_loss(prediction, clean_tau)
        loss.backward()
        total_grad_norm = grad_norm(model)
        optimizer.step()
        ema_update(ema_shadow, model, ema_decay)
        with torch.no_grad():
            loss_after = F.mse_loss(model(noisy_tau, denoising_steps), clean_tau)
        rows.append(
            {
                "step": step,
                "learning_rate": lr,
                "ema_decay": ema_decay,
                "loss_before": float(loss_before.detach().cpu().item()),
                "loss_after": float(loss_after.detach().cpu().item()),
                "grad_norm": total_grad_norm,
                "model_vs_initial_l2": mean_state_l2_delta(clone_state(model), initial_state),
                "ema_vs_model_l2": mean_state_l2_delta(ema_shadow, clone_state(model)),
            }
        )

    final_state = clone_state(model)
    metrics = {
        "step_count": args.steps,
        "parameter_count": parameter_count(model),
        "initial_model_sha256": state_digest(initial_state),
        "final_model_sha256": state_digest(final_state),
        "final_ema_sha256": state_digest(ema_shadow),
        "final_loss_before": rows[-1]["loss_before"],
        "final_loss_after": rows[-1]["loss_after"],
        "loss_after_min": min(row["loss_after"] for row in rows),
        "model_vs_initial_l2": mean_state_l2_delta(final_state, initial_state),
        "ema_vs_model_l2": mean_state_l2_delta(ema_shadow, final_state),
        "ema_vs_model_max_abs": max_state_abs_delta(ema_shadow, final_state),
    }
    checks = {
        "uses_paper_state_token_dim_131": token_dim == 131,
        "uses_paper_transformer_hyperparameters": cfg.embedding_dim == 512
        and cfg.attention_heads == 8
        and cfg.transformer_layers == 6
        and cfg.denoising_steps == 20,
        "uses_paper_optimizer_hyperparameters": schedule.learning_rate == 1e-4
        and schedule.weight_decay == 0.001
        and schedule.ema_power == 0.75
        and schedule.ema_max == 0.9999,
        "learning_rate_schedule_applied": rows[0]["learning_rate"] == learning_rate_at_step(0, schedule)
        and rows[-1]["learning_rate"] == learning_rate_at_step(args.steps - 1, schedule),
        "ema_decay_schedule_applied": rows[0]["ema_decay"] == ema_decay_at_step(0, schedule)
        and rows[-1]["ema_decay"] == ema_decay_at_step(args.steps - 1, schedule),
        "all_losses_finite": all(np.isfinite(row["loss_before"]) and np.isfinite(row["loss_after"]) for row in rows),
        "all_grad_norms_positive": all(row["grad_norm"] > 0.0 for row in rows),
        "model_parameters_changed": metrics["initial_model_sha256"] != metrics["final_model_sha256"]
        and metrics["model_vs_initial_l2"] > 0.0,
        "ema_shadow_differs_from_final_model": metrics["final_ema_sha256"] != metrics["final_model_sha256"]
        and metrics["ema_vs_model_l2"] > 0.0,
        "loss_after_not_worse_than_before_on_last_batch": metrics["final_loss_after"] <= metrics["final_loss_before"],
        "does_not_write_weight_checkpoint": True,
        "does_not_claim_training": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "transformer_ema_smoke",
        "scope": "debug-only AdamW, learning-rate, and EMA shadow update smoke on the paper-state Transformer",
        "settings": {
            "device": str(device),
            "steps": args.steps,
            "token_dim": token_dim,
            "state_dim": state_dim,
            "latent_dim": cfg.latent_dim,
            "sequence_length": cfg.sequence_length,
            "learning_rate": schedule.learning_rate,
            "weight_decay": schedule.weight_decay,
            "warmup_gradient_steps": schedule.warmup_gradient_steps,
            "ema_power": schedule.ema_power,
            "ema_max": schedule.ema_max,
        },
        "step_rows": rows,
        "metrics": metrics,
        "checks": checks,
        "not_a_replacement_for": [
            "full diffusion training",
            "trained diffusion checkpoint",
            "EMA checkpoint",
            "validation/test metrics",
            "TensorRT deployment",
            "Fig. 5/Fig. 6 rollout evaluation",
        ],
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "partial",
            "why_not_complete": (
                "This executes two optimizer/EMA smoke steps on one debug paper-state batch. It validates local "
                "training mechanics, but it is not paper-scale training and does not save a trained checkpoint."
            ),
        },
        "outputs": {
            "json": str(OUT / "level_c_transformer_ema_smoke.json"),
            "tsv": str(OUT / "level_c_transformer_ema_smoke.tsv"),
        },
    }
    (OUT / "level_c_transformer_ema_smoke.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_tsv(OUT / "level_c_transformer_ema_smoke.tsv", rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "steps": args.steps}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
