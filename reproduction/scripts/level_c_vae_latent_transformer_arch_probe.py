#!/usr/bin/env python3
"""Paper-sized Transformer probe using nonzero debug VAE latents."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

import level_c_vae_latent_diffusion_overfit_probe as vae_latent
from level_c_paper_state_transformer_arch_probe import (
    PaperStateDiffusionTransformer,
    ProbeConfig,
    choose_device,
    diffusion_alpha_bars,
    grad_norm,
    parameter_count,
    seed_everything,
)


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/vae_latent_transformer_arch_probe"


def write_tsv(path: Path, rows: dict[str, Any]) -> None:
    flat: list[tuple[str, str]] = []

    def rec(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                rec(f"{prefix}.{key}" if prefix else str(key), value[key])
        elif isinstance(value, list):
            flat.append((prefix, json.dumps(value, sort_keys=True)))
        else:
            flat.append((prefix, str(value)))

    rec("", rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["key", "value"])
        writer.writerows(flat)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    cfg = ProbeConfig(batch_size=args.batch_size, seed=args.seed)
    seed_everything(cfg.seed)
    torch.set_num_threads(args.torch_threads)
    device = choose_device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()

    latent_cfg = vae_latent.VaeLatentDiffusionConfig(seed=args.seed)
    clean_np, motion_ids, latent_manifest = vae_latent.load_dataset(latent_cfg)
    if args.batch_size > clean_np.shape[0]:
        raise ValueError(f"batch_size {args.batch_size} exceeds {clean_np.shape[0]} debug VAE-latent windows")
    clean_tau = torch.from_numpy(clean_np[: args.batch_size].astype(np.float32)).to(device)
    batch_motion_ids = motion_ids[: args.batch_size]
    state_dim = latent_cfg.state_dim
    token_dim = int(clean_tau.shape[-1])
    model = PaperStateDiffusionTransformer(token_dim=token_dim, cfg=cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.paper_learning_rate, weight_decay=cfg.paper_weight_decay)

    alpha_bars = diffusion_alpha_bars(cfg.denoising_steps, device, clean_tau.dtype)
    steps = torch.randint(0, cfg.denoising_steps, (cfg.batch_size, cfg.sequence_length, 2), device=device)
    noise = torch.randn_like(clean_tau)
    state_alpha = alpha_bars[steps[..., 0]].unsqueeze(-1).expand(-1, -1, state_dim)
    latent_alpha = alpha_bars[steps[..., 1]].unsqueeze(-1).expand(-1, -1, cfg.latent_dim)
    alpha = torch.cat([state_alpha, latent_alpha], dim=-1)
    noisy_tau = torch.sqrt(alpha) * clean_tau + torch.sqrt(1.0 - alpha) * noise

    prediction = model(noisy_tau, steps)
    loss = F.mse_loss(prediction, clean_tau)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    total_grad_norm = grad_norm(model)
    optimizer.step()

    cuda_peak_memory_mb = None
    if device.type == "cuda":
        cuda_peak_memory_mb = float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))

    json_path = OUT / "level_c_vae_latent_transformer_arch_probe.json"
    tsv_path = OUT / "level_c_vae_latent_transformer_arch_probe.tsv"
    npz_path = OUT / "level_c_vae_latent_transformer_arch_probe.npz"
    np.savez_compressed(
        npz_path,
        clean_tau=clean_tau.detach().cpu().numpy(),
        noisy_tau=noisy_tau.detach().cpu().numpy(),
        denoising_steps=steps.detach().cpu().numpy(),
        prediction=prediction.detach().cpu().numpy(),
        motion_ids=batch_motion_ids,
    )
    checks = {
        "debug_vae_latent_artifact_status_ok": latent_manifest["status"] == "ok",
        "debug_vae_latents_nonzero": latent_manifest["checks"]["all_latents_nonzero"],
        "uses_paper_state_dim_99": state_dim == 99,
        "uses_debug_vae_latent_dim_32": cfg.latent_dim == 32,
        "uses_token_dim_131": token_dim == 131,
        "uses_paper_embedding_dim": cfg.embedding_dim == 512,
        "uses_paper_attention_heads": cfg.attention_heads == 8,
        "uses_paper_transformer_layers": cfg.transformer_layers == 6,
        "uses_paper_denoising_steps": cfg.denoising_steps == 20,
        "uses_paper_history_horizon": cfg.history == 4 and cfg.horizon == 16,
        "step_tensor_is_independent_state_latent": list(steps.shape) == [cfg.batch_size, cfg.sequence_length, 2]
        and bool(torch.any(steps[..., 0] != steps[..., 1]).detach().cpu().item()),
        "prediction_shape_matches_clean_tau": list(prediction.shape) == list(clean_tau.shape),
        "loss_is_finite": bool(torch.isfinite(loss).detach().cpu().item()),
        "grad_norm_is_positive": total_grad_norm > 0.0,
        "does_not_claim_training": True,
        "does_not_claim_true_vae_rollout": True,
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "debug_only",
        "scope": "paper-sized Transformer forward/backward probe on paper states plus nonzero tiny-VAE debug latents",
        "paper_evidence": {
            "debug_vae_latents": str(vae_latent.VAE_LATENT_JSON),
            "state_latent_dataset": str(ROOT / "reproduction/paper/source/root.tex:536-546"),
            "diffusion_hyperparameters": str(ROOT / "reproduction/paper/source/root.tex:827-856"),
            "clean_trajectory_loss": str(ROOT / "reproduction/paper/source/tex/method.tex:187-196"),
            "individual_denoising_steps": str(ROOT / "reproduction/paper/source/tex/method.tex:171-185"),
        },
        "not_a_replacement_for": [
            "true VAE rollout state-latent dataset",
            "long diffusion training",
            "trained diffusion checkpoint",
            "TensorRT deployment",
            "guided rollout evaluation",
            "Fig. 5/Fig. 6 reproduction",
        ],
        "settings": {
            **asdict(cfg),
            "device": str(device),
            "torch_threads": args.torch_threads,
            "state_dim": state_dim,
            "token_dim": token_dim,
            "vae_latent_window_count": int(clean_np.shape[0]),
            "latent_source": "debug_tiny_vae_mu_nonzero_synthetic_teacher",
        },
        "model": {
            "class": "PaperStateDiffusionTransformer",
            "parameter_count": parameter_count(model),
            "input_projection_shape": list(model.input_proj.weight.shape),
            "output_projection_shape": list(model.output_proj.weight.shape),
            "position_embedding_shape": list(model.pos_embed.shape),
            "state_step_embedding_shape": list(model.state_step_embed.weight.shape),
            "latent_step_embedding_shape": list(model.latent_step_embed.weight.shape),
            "encoder_layers": cfg.transformer_layers,
            "attention_heads": cfg.attention_heads,
            "embedding_dim": cfg.embedding_dim,
        },
        "batch": {
            "clean_tau_shape": list(clean_tau.shape),
            "noisy_tau_shape": list(noisy_tau.shape),
            "denoising_steps_shape": list(steps.shape),
            "prediction_shape": list(prediction.shape),
            "motion_ids": [int(x) for x in batch_motion_ids.tolist()],
            "unique_state_steps": sorted(int(x) for x in torch.unique(steps[..., 0]).detach().cpu().tolist()),
            "unique_latent_steps": sorted(int(x) for x in torch.unique(steps[..., 1]).detach().cpu().tolist()),
        },
        "metrics": {
            "clean_trajectory_mse": float(loss.detach().cpu().item()),
            "total_grad_norm": total_grad_norm,
            "cuda_peak_memory_mb": cuda_peak_memory_mb,
        },
        "checks": checks,
        "interpretation": {
            "paper_level_status": "partial",
            "goal_complete": False,
            "why_not_complete": (
                "This verifies one paper-sized Transformer forward/backward pass on local paper-state windows with "
                "nonzero tiny-VAE debug latents. It still does not use true VAE rollout latents, train the diffusion "
                "model, produce a trained checkpoint, run TensorRT, or evaluate rollouts."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    summary = run_probe(args)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "npz": summary["outputs"]["npz"]}, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
