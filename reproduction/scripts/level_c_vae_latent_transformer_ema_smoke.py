#!/usr/bin/env python3
"""Debug-only AdamW/LR/EMA smoke on paper Transformer with debug VAE latents."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

import level_c_vae_latent_diffusion_overfit_probe as vae_latent
from level_c_paper_state_transformer_arch_probe import (
    PaperStateDiffusionTransformer,
    ProbeConfig,
    diffusion_alpha_bars,
    grad_norm,
    parameter_count,
    seed_everything,
)
from level_c_training_schedule_probe import DiffusionTrainingScheduleConfig, ema_decay_at_step, learning_rate_at_step
from level_c_transformer_ema_smoke import (
    clone_state,
    ema_update,
    max_state_abs_delta,
    mean_state_l2_delta,
    state_digest,
)


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/vae_latent_transformer_ema_smoke"


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
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
    with path.open("w", encoding="utf-8", newline="") as f:
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

    latent_cfg = vae_latent.VaeLatentDiffusionConfig(seed=cfg.seed)
    clean_np, motion_ids, latent_manifest = vae_latent.load_dataset(latent_cfg)
    clean_tau = torch.from_numpy(clean_np[: cfg.batch_size].astype(np.float32)).to(device)
    state_dim = latent_cfg.state_dim
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

        seed_everything(cfg.seed + 410 + step)
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
    latent_slice = clean_np[..., state_dim:]
    metrics = {
        "step_count": args.steps,
        "parameter_count": parameter_count(model),
        "input_window_count": int(clean_np.shape[0]),
        "input_motion_count": int(len(set(motion_ids.tolist()))),
        "input_motion_ids_first_batch": [int(x) for x in motion_ids[: cfg.batch_size].tolist()],
        "latent_abs_mean": float(np.mean(np.abs(latent_slice))),
        "latent_abs_max": float(np.max(np.abs(latent_slice))),
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
        "debug_vae_latent_artifact_status_ok": latent_manifest["status"] == "ok",
        "debug_vae_latents_nonzero": latent_manifest["checks"]["all_latents_nonzero"],
        "uses_paper_state_token_dim_131": token_dim == 131,
        "uses_paper_state_dim_99": state_dim == 99,
        "uses_debug_vae_latent_dim_32": cfg.latent_dim == 32,
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
        "does_not_claim_true_vae_rollout": True,
        "does_not_claim_goal_complete": True,
    }
    json_path = OUT / "level_c_vae_latent_transformer_ema_smoke.json"
    tsv_path = OUT / "level_c_vae_latent_transformer_ema_smoke.tsv"
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "vae_latent_transformer_ema_smoke",
        "scope": (
            "debug-only AdamW, learning-rate, and EMA shadow update smoke on the paper-sized Transformer "
            "using nonzero tiny-VAE debug latents"
        ),
        "paper_evidence": {
            "debug_vae_latents": str(vae_latent.VAE_LATENT_JSON),
            "state_latent_dataset": str(ROOT / "reproduction/paper/source/root.tex:536-546"),
            "diffusion_hyperparameters": str(ROOT / "reproduction/paper/source/root.tex:827-856"),
            "clean_trajectory_loss": str(ROOT / "reproduction/paper/source/tex/method.tex:187-196"),
            "individual_denoising_steps": str(ROOT / "reproduction/paper/source/tex/method.tex:171-185"),
        },
        "settings": {
            "device": str(device),
            "torch_threads": args.torch_threads,
            "steps": args.steps,
            "token_dim": token_dim,
            "state_dim": state_dim,
            "latent_dim": cfg.latent_dim,
            "sequence_length": cfg.sequence_length,
            "latent_source": "debug_tiny_vae_mu_nonzero_synthetic_teacher",
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
            "true VAE rollout state-latent dataset",
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
                "This executes two optimizer/EMA smoke steps on one debug state-latent batch that includes nonzero "
                "tiny-VAE latents. It validates local training mechanics on the intended token shape, but it is not "
                "paper-scale training, does not use true rollout latents, and does not save a trained checkpoint."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, rows)
    print(json.dumps({"status": summary["status"], "json": str(json_path), "steps": args.steps}, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
