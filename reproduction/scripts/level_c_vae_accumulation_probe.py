#!/usr/bin/env python3
"""Debug-only VAE gradient-accumulation and DAgger-batch schema probe."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/vae_accumulation_probe"


@dataclass(frozen=True)
class VAEAccumulationConfig:
    seed: int = 20260830
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 15
    num_joints: int = 29
    reference_motion_dim: int = 58
    anchor_error_dim: int = 9
    proprioception_dim: int = 96
    latent_dim: int = 32
    encoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512)
    decoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512)
    teacher_hidden_dims: tuple[int, ...] = (512, 256, 128)
    learning_rate: float = 5e-4
    kl_coefficient: float = 0.01

    @property
    def encoder_input_dim(self) -> int:
        return self.reference_motion_dim + self.anchor_error_dim

    @property
    def decoder_input_dim(self) -> int:
        return self.latent_dim + self.proprioception_dim

    @property
    def teacher_input_dim(self) -> int:
        return self.proprioception_dim + self.encoder_input_dim

    @property
    def action_dim(self) -> int:
        return self.num_joints

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for hidden in hidden_dims:
        layers.extend([nn.Linear(prev, hidden), nn.ELU()])
        prev = hidden
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class ConditionalVAE(nn.Module):
    def __init__(self, cfg: VAEAccumulationConfig) -> None:
        super().__init__()
        self.encoder = mlp(cfg.encoder_input_dim, cfg.encoder_hidden_dims, cfg.latent_dim * 2)
        self.decoder = mlp(cfg.decoder_input_dim, cfg.decoder_hidden_dims, cfg.action_dim)

    def forward(self, reference_and_anchor: torch.Tensor, proprioception: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stats = self.encoder(reference_and_anchor)
        mu, logvar = stats.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        action = self.decoder(torch.cat([z, proprioception], dim=-1))
        return action, mu, logvar


def parameter_count(module: nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters()))


def grad_norm(module: nn.Module) -> float:
    total = torch.zeros(())
    for param in module.parameters():
        if param.grad is not None:
            total = total + param.grad.detach().cpu().pow(2).sum()
    return float(torch.sqrt(total).item())


def parameter_delta_norm(before: list[torch.Tensor], module: nn.Module) -> float:
    total = torch.zeros(())
    for old, param in zip(before, module.parameters(), strict=True):
        total = total + (param.detach().cpu() - old).pow(2).sum()
    return float(torch.sqrt(total).item())


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "micro_step",
        "scaled_total_loss",
        "unscaled_total_loss",
        "reconstruction_loss",
        "kl_loss",
        "teacher_action_norm",
        "grad_norm_after_backward",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    cfg = VAEAccumulationConfig(seed=args.seed)
    seed_everything(cfg.seed)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)

    teacher = mlp(cfg.teacher_input_dim, cfg.teacher_hidden_dims, cfg.action_dim).to(device)
    vae = ConditionalVAE(cfg).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.learning_rate)
    before_params = [param.detach().cpu().clone() for param in vae.parameters()]
    optimizer.zero_grad(set_to_none=True)

    rows: list[dict[str, Any]] = []
    accumulated_unscaled_total = 0.0
    accumulated_reconstruction = 0.0
    accumulated_kl = 0.0
    for micro_step in range(cfg.gradient_accumulation_steps):
        reference_and_anchor = torch.randn(cfg.micro_batch_size, cfg.encoder_input_dim, device=device)
        proprioception = torch.randn(cfg.micro_batch_size, cfg.proprioception_dim, device=device)
        teacher_input = torch.cat([proprioception, reference_and_anchor], dim=-1)
        with torch.no_grad():
            teacher_action = teacher(teacher_input)

        predicted_action, mu, logvar = vae(reference_and_anchor, proprioception)
        reconstruction_loss = F.mse_loss(predicted_action, teacher_action)
        kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        unscaled_loss = reconstruction_loss + cfg.kl_coefficient * kl_loss
        scaled_loss = unscaled_loss / cfg.gradient_accumulation_steps
        scaled_loss.backward()

        accumulated_unscaled_total += float(unscaled_loss.detach().cpu())
        accumulated_reconstruction += float(reconstruction_loss.detach().cpu())
        accumulated_kl += float(kl_loss.detach().cpu())
        rows.append(
            {
                "micro_step": micro_step + 1,
                "scaled_total_loss": float(scaled_loss.detach().cpu()),
                "unscaled_total_loss": float(unscaled_loss.detach().cpu()),
                "reconstruction_loss": float(reconstruction_loss.detach().cpu()),
                "kl_loss": float(kl_loss.detach().cpu()),
                "teacher_action_norm": float(teacher_action.norm().detach().cpu()),
                "grad_norm_after_backward": grad_norm(vae),
            }
        )

    grad_norm_before_step = grad_norm(vae)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    parameter_update_norm = parameter_delta_norm(before_params, vae)
    grad_norm_after_zero = grad_norm(vae)

    json_path = OUT / "level_c_vae_accumulation_probe.json"
    tsv_path = OUT / "level_c_vae_accumulation_probe.tsv"
    npz_path = OUT / "level_c_vae_accumulation_probe.npz"
    write_tsv(tsv_path, rows)
    np.savez_compressed(
        npz_path,
        micro_step=np.asarray([row["micro_step"] for row in rows], dtype=np.int64),
        unscaled_total_loss=np.asarray([row["unscaled_total_loss"] for row in rows], dtype=np.float64),
        scaled_total_loss=np.asarray([row["scaled_total_loss"] for row in rows], dtype=np.float64),
        grad_norm_after_backward=np.asarray([row["grad_norm_after_backward"] for row in rows], dtype=np.float64),
    )

    synthetic_dagger_manifest = {
        "teacher_policy_source": "synthetic_teacher_mlp_debug_only",
        "student_policy_source": "conditional_vae_debug_only",
        "teacher_query_rule": "query synthetic teacher for every micro-batch sample",
        "aggregation_mode": "single optimizer update from 15 accumulated synthetic micro-batches",
        "rollout_source": "none_no_live_isaac_or_teacher_policy",
        "is_true_dagger_rollout": False,
    }
    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "paper-parameter VAE gradient-accumulation and synthetic DAgger-batch schema probe",
        "paper_evidence": {
            "vae_hyperparameters": str(ROOT / "reproduction/paper/source/root.tex:801-825"),
            "vae_dagger_loss": str(ROOT / "reproduction/paper/source/tex/method.tex:150-170"),
            "goal_dagger_requirement": str(ROOT / "goal.md:1148-1185"),
        },
        "not_a_replacement_for": [
            "real teacher policy rollout",
            "true DAgger aggregation over environment states",
            "VAE checkpoint reproduction",
            "VAE rollout stability evaluation",
            "latent analysis",
        ],
        "settings": asdict(cfg) | {"device": str(device)},
        "model": {
            "teacher_parameter_count": parameter_count(teacher),
            "vae_parameter_count": parameter_count(vae),
            "encoder_output_dim": cfg.latent_dim * 2,
            "decoder_output_dim": cfg.action_dim,
        },
        "synthetic_dagger_manifest": synthetic_dagger_manifest,
        "micro_step_rows": rows,
        "metrics": {
            "effective_batch_size": cfg.effective_batch_size,
            "optimizer_steps": 1,
            "micro_batches": cfg.gradient_accumulation_steps,
            "mean_unscaled_total_loss": accumulated_unscaled_total / cfg.gradient_accumulation_steps,
            "mean_reconstruction_loss": accumulated_reconstruction / cfg.gradient_accumulation_steps,
            "mean_kl_loss": accumulated_kl / cfg.gradient_accumulation_steps,
            "grad_norm_before_optimizer_step": grad_norm_before_step,
            "grad_norm_after_zero_grad": grad_norm_after_zero,
            "parameter_update_norm": parameter_update_norm,
        },
        "checks": {
            "latent_dim_matches_paper": cfg.latent_dim == 32,
            "encoder_dims_match_paper": cfg.encoder_hidden_dims == (2048, 1024, 512),
            "decoder_dims_match_paper": cfg.decoder_hidden_dims == (2048, 1024, 512),
            "teacher_dims_match_paper": cfg.teacher_hidden_dims == (512, 256, 128),
            "learning_rate_matches_paper": math.isclose(cfg.learning_rate, 5e-4),
            "kl_coefficient_matches_paper": math.isclose(cfg.kl_coefficient, 0.01),
            "gradient_accumulation_matches_paper": cfg.gradient_accumulation_steps == 15,
            "all_micro_losses_finite": bool(all(math.isfinite(row["unscaled_total_loss"]) for row in rows)),
            "all_micro_grad_norms_positive": bool(all(row["grad_norm_after_backward"] > 0.0 for row in rows)),
            "single_optimizer_step_updates_parameters": parameter_update_norm > 0.0,
            "zero_grad_clears_gradients": math.isclose(grad_norm_after_zero, 0.0, abs_tol=1e-12),
            "manifest_marks_not_true_dagger_rollout": synthetic_dagger_manifest["is_true_dagger_rollout"] is False,
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
