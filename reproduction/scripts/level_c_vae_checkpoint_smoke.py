#!/usr/bin/env python3
"""Debug-only VAE checkpoint save/load consistency smoke."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from level_c_vae_accumulation_probe import (
    ConditionalVAE,
    VAEAccumulationConfig,
    grad_norm,
    mlp,
    parameter_count,
    parameter_delta_norm,
)


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/vae_checkpoint_smoke"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_synthetic_step(cfg: VAEAccumulationConfig, device: torch.device) -> tuple[ConditionalVAE, torch.optim.Optimizer, dict[str, float]]:
    seed_everything(cfg.seed)
    teacher = mlp(cfg.teacher_input_dim, cfg.teacher_hidden_dims, cfg.action_dim).to(device)
    vae = ConditionalVAE(cfg).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.learning_rate)
    before = [param.detach().cpu().clone() for param in vae.parameters()]
    optimizer.zero_grad(set_to_none=True)
    accumulated_loss = 0.0
    for _ in range(cfg.gradient_accumulation_steps):
        reference_and_anchor = torch.randn(cfg.micro_batch_size, cfg.encoder_input_dim, device=device)
        proprioception = torch.randn(cfg.micro_batch_size, cfg.proprioception_dim, device=device)
        with torch.no_grad():
            teacher_action = teacher(torch.cat([proprioception, reference_and_anchor], dim=-1))
        predicted, mu, logvar = vae(reference_and_anchor, proprioception)
        reconstruction_loss = F.mse_loss(predicted, teacher_action)
        kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        loss = reconstruction_loss + cfg.kl_coefficient * kl_loss
        (loss / cfg.gradient_accumulation_steps).backward()
        accumulated_loss += float(loss.detach().cpu())
    grad_norm_before_step = grad_norm(vae)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return (
        vae,
        optimizer,
        {
            "mean_training_loss": accumulated_loss / cfg.gradient_accumulation_steps,
            "grad_norm_before_optimizer_step": grad_norm_before_step,
            "parameter_update_norm": parameter_delta_norm(before, vae),
        },
    )


def eval_action(vae: ConditionalVAE, reference_and_anchor: torch.Tensor, proprioception: torch.Tensor) -> torch.Tensor:
    vae.eval()
    stats = vae.encoder(reference_and_anchor)
    mu, _logvar = stats.chunk(2, dim=-1)
    return vae.decoder(torch.cat([mu, proprioception], dim=-1))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["check", "passed", "detail"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    cfg = VAEAccumulationConfig(seed=args.seed)
    vae, optimizer, train_metrics = train_one_synthetic_step(cfg, device)

    seed_everything(cfg.seed + 1000)
    reference_and_anchor = torch.randn(4, cfg.encoder_input_dim, device=device)
    proprioception = torch.randn(4, cfg.proprioception_dim, device=device)
    with torch.no_grad():
        action_before = eval_action(vae, reference_and_anchor, proprioception).detach().cpu()

    checkpoint_path = OUT / "debug_conditional_vae_checkpoint_smoke.pt"
    checkpoint_payload = {
        "experiment_type": "debug_only_vae_checkpoint_smoke",
        "paper_evidence": {
            "vae_hyperparameters": str(ROOT / "reproduction/paper/source/root.tex:801-825"),
            "vae_dagger_loss": str(ROOT / "reproduction/paper/source/tex/method.tex:150-170"),
            "goal_checkpoint_requirement": str(ROOT / "goal.md:1148-1190,1825"),
        },
        "config": asdict(cfg),
        "model_state_dict": vae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": 1,
        "seed": cfg.seed,
        "is_trained_paper_checkpoint": False,
        "not_a_replacement_for": [
            "trained conditional VAE checkpoint",
            "true DAgger rollout aggregation",
            "closed-loop VAE survival evaluation",
            "paper Fig. 5/Fig. 6 reproduction",
        ],
    }
    torch.save(checkpoint_payload, checkpoint_path)

    loaded_payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    loaded_cfg = VAEAccumulationConfig(**loaded_payload["config"])
    loaded_vae = ConditionalVAE(loaded_cfg).to(device)
    loaded_optimizer = torch.optim.Adam(loaded_vae.parameters(), lr=loaded_cfg.learning_rate)
    loaded_vae.load_state_dict(loaded_payload["model_state_dict"])
    loaded_optimizer.load_state_dict(loaded_payload["optimizer_state_dict"])
    with torch.no_grad():
        action_after = eval_action(loaded_vae, reference_and_anchor, proprioception).detach().cpu()
    max_abs_action_error = float(torch.max(torch.abs(action_after - action_before)).item())
    optimizer_state_param_count = sum(
        int(value.numel())
        for state in loaded_optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    checks = {
        "checkpoint_file_exists": checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0,
        "checkpoint_has_required_keys": all(
            key in loaded_payload
            for key in ["config", "model_state_dict", "optimizer_state_dict", "step", "seed", "is_trained_paper_checkpoint"]
        ),
        "paper_dimensions_match": loaded_cfg.latent_dim == 32
        and loaded_cfg.encoder_hidden_dims == (2048, 1024, 512)
        and loaded_cfg.decoder_hidden_dims == (2048, 1024, 512)
        and loaded_cfg.teacher_hidden_dims == (512, 256, 128),
        "gradient_accumulation_matches_paper": loaded_cfg.gradient_accumulation_steps == 15,
        "loaded_eval_action_matches_saved_model": math.isclose(max_abs_action_error, 0.0, abs_tol=1e-12),
        "optimizer_state_restored": optimizer_state_param_count > parameter_count(loaded_vae),
        "parameter_update_nonzero": train_metrics["parameter_update_norm"] > 0.0,
        "marks_not_trained_paper_checkpoint": loaded_payload["is_trained_paper_checkpoint"] is False,
        "does_not_claim_true_dagger_rollout": True,
        "does_not_claim_goal_complete": True,
    }
    rows = [{"check": key, "passed": value, "detail": ""} for key, value in checks.items()]
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "vae_checkpoint_smoke",
        "scope": "debug-only conditional VAE checkpoint save/load and deterministic eval consistency",
        "settings": asdict(cfg) | {"device": str(device)},
        "model": {
            "vae_parameter_count": parameter_count(vae),
            "encoder_output_dim": cfg.latent_dim * 2,
            "decoder_output_dim": cfg.action_dim,
            "optimizer_state_tensor_count": optimizer_state_param_count,
        },
        "metrics": {
            **train_metrics,
            "max_abs_loaded_eval_action_error": max_abs_action_error,
            "checkpoint_size_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else 0,
            "eval_batch_size": int(reference_and_anchor.shape[0]),
        },
        "checks": checks,
        "not_a_replacement_for": loaded_payload["not_a_replacement_for"],
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "partial",
            "why_not_complete": (
                "The smoke validates checkpoint save/load mechanics for the synthetic VAE debug model, but it is not "
                "a trained BeyondMimic VAE checkpoint from true DAgger rollout data and cannot support paper metrics."
            ),
        },
        "outputs": {
            "json": str(OUT / "level_c_vae_checkpoint_smoke.json"),
            "tsv": str(OUT / "level_c_vae_checkpoint_smoke.tsv"),
            "checkpoint": str(checkpoint_path),
        },
    }
    (OUT / "level_c_vae_checkpoint_smoke.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_tsv(OUT / "level_c_vae_checkpoint_smoke.tsv", rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "checkpoint": str(checkpoint_path)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
