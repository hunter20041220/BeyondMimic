#!/usr/bin/env python3
"""Synthetic Level C smoke for BeyondMimic VAE and guided diffusion.

This is deliberately a smoke test, not a faithful training reproduction. It
checks that paper-specified dimensions and losses can execute end-to-end on a
tiny synthetic batch before any long VAE/diffusion job is considered.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/synthetic_smoke"


@dataclass(frozen=True)
class SmokeConfig:
    seed: int = 20260825
    batch_size: int = 2
    num_joints: int = 29
    target_bodies: int = 14
    reference_motion_dim: int = 58
    anchor_error_dim: int = 9
    proprioception_dim: int = 96
    vae_latent_dim: int = 32
    vae_encoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512)
    vae_decoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512)
    teacher_hidden_dims: tuple[int, ...] = (512, 256, 128)
    vae_learning_rate: float = 5e-4
    vae_kl_coefficient: float = 0.01
    paper_accumulated_gradient_steps: int = 15
    diffusion_horizon: int = 16
    observation_history: int = 4
    diffusion_embedding_dim: int = 512
    diffusion_attention_heads: int = 8
    diffusion_transformer_layers: int = 6
    diffusion_denoising_steps: int = 20
    diffusion_batch_size_paper: int = 512
    diffusion_epochs_paper: int = 1000
    diffusion_learning_rate: float = 1e-4
    diffusion_weight_decay: float = 0.001
    diffusion_warmup_gradient_steps: int = 10000
    diffusion_ema_power: float = 0.75
    diffusion_ema_max: float = 0.9999
    smoke_diffusion_embedding_dim: int = 128
    smoke_diffusion_attention_heads: int = 4
    smoke_diffusion_transformer_layers: int = 2
    root_state_dim: int = 15
    body_position_velocity_dim: int = 84

    @property
    def vae_encoder_input_dim(self) -> int:
        return self.reference_motion_dim + self.anchor_error_dim

    @property
    def vae_decoder_input_dim(self) -> int:
        return self.vae_latent_dim + self.proprioception_dim

    @property
    def action_dim(self) -> int:
        return self.num_joints

    @property
    def state_dim(self) -> int:
        return self.root_state_dim + self.body_position_velocity_dim

    @property
    def diffusion_token_dim(self) -> int:
        return self.state_dim + self.vae_latent_dim

    @property
    def diffusion_sequence_length(self) -> int:
        return self.observation_history + 1 + self.diffusion_horizon


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    forced = os.environ.get("BM_LEVEL_C_DEVICE")
    if forced:
        return torch.device(forced)
    return torch.device("cpu")


def mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for hidden in hidden_dims:
        layers.extend([nn.Linear(prev, hidden), nn.ELU()])
        prev = hidden
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class ConditionalVAE(nn.Module):
    def __init__(self, cfg: SmokeConfig) -> None:
        super().__init__()
        self.encoder = mlp(
            cfg.vae_encoder_input_dim,
            cfg.vae_encoder_hidden_dims,
            cfg.vae_latent_dim * 2,
        )
        self.decoder = mlp(
            cfg.vae_decoder_input_dim,
            cfg.vae_decoder_hidden_dims,
            cfg.action_dim,
        )

    def forward(self, reference_and_anchor: torch.Tensor, proprioception: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stats = self.encoder(reference_and_anchor)
        mu, logvar = stats.chunk(2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        action = self.decoder(torch.cat([z, proprioception], dim=-1))
        return action, mu, logvar, z


class DiffusionTransformer(nn.Module):
    def __init__(self, cfg: SmokeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(cfg.diffusion_token_dim, cfg.smoke_diffusion_embedding_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.diffusion_sequence_length, cfg.smoke_diffusion_embedding_dim))
        self.state_step_embed = nn.Embedding(cfg.diffusion_denoising_steps, cfg.smoke_diffusion_embedding_dim)
        self.latent_step_embed = nn.Embedding(cfg.diffusion_denoising_steps, cfg.smoke_diffusion_embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.smoke_diffusion_embedding_dim,
            nhead=cfg.smoke_diffusion_attention_heads,
            dim_feedforward=cfg.smoke_diffusion_embedding_dim * 4,
            activation="gelu",
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.smoke_diffusion_transformer_layers)
        self.output_proj = nn.Linear(cfg.smoke_diffusion_embedding_dim, cfg.diffusion_token_dim)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, noisy_tokens: torch.Tensor, diffusion_steps: torch.Tensor) -> torch.Tensor:
        if diffusion_steps.shape != noisy_tokens.shape[:2] + (2,):
            raise ValueError(
                "diffusion_steps must have shape [batch, sequence, 2] for independent state/latent timesteps"
            )
        x = self.input_proj(noisy_tokens)
        x = x + self.pos_embed[:, : x.shape[1]]
        state_steps = diffusion_steps[..., 0]
        latent_steps = diffusion_steps[..., 1]
        x = x + self.state_step_embed(state_steps) + self.latent_step_embed(latent_steps)
        return self.output_proj(self.encoder(x))


def make_diffusion_schedule(steps: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    betas = torch.linspace(1e-4, 0.02, steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alpha_bars


def guidance_costs(tokens: torch.Tensor, cfg: SmokeConfig) -> dict[str, torch.Tensor]:
    start = cfg.observation_history
    future = tokens[:, start:, :]
    root_xy = future[..., 0:2]
    root_velocity_xy = future[..., 9:11]

    command_velocity = torch.tensor([0.8, 0.0], device=tokens.device, dtype=tokens.dtype)
    waypoint = torch.tensor([1.5, 0.25], device=tokens.device, dtype=tokens.dtype)
    keyframe = torch.zeros(cfg.diffusion_token_dim, device=tokens.device, dtype=tokens.dtype)
    obstacle_center = torch.tensor([0.75, 0.05], device=tokens.device, dtype=tokens.dtype)
    obstacle_radius = torch.tensor(0.35, device=tokens.device, dtype=tokens.dtype)
    margin = torch.tensor(0.1, device=tokens.device, dtype=tokens.dtype)

    distance = torch.linalg.norm(root_xy - obstacle_center, dim=-1)
    signed_distance = distance - obstacle_radius

    return {
        "joystick_velocity": F.mse_loss(root_velocity_xy, command_velocity.expand_as(root_velocity_xy)),
        "waypoint": F.mse_loss(root_xy[:, -1, :], waypoint.expand_as(root_xy[:, -1, :])),
        "keyframe_inpainting": F.mse_loss(future[:, cfg.diffusion_horizon // 2, :], keyframe.expand_as(future[:, cfg.diffusion_horizon // 2, :])),
        "sdf_obstacle": torch.relu(margin - signed_distance).pow(2).mean(),
    }


def tensor_shape(tensor: torch.Tensor) -> list[int]:
    return [int(dim) for dim in tensor.shape]


def parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def flatten(prefix: str, value: Any, rows: list[tuple[str, str]]) -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            flatten(f"{prefix}.{key}" if prefix else str(key), value[key], rows)
    elif isinstance(value, (list, tuple)):
        rows.append((prefix, json.dumps(value)))
    else:
        rows.append((prefix, str(value)))


def write_tsv(path: Path, summary: dict[str, Any]) -> None:
    rows: list[tuple[str, str]] = []
    flatten("", summary, rows)
    with path.open("w", encoding="utf-8") as f:
        f.write("key\tvalue\n")
        for key, value in rows:
            f.write(f"{key}\t{value}\n")


def main() -> None:
    cfg = SmokeConfig()
    OUT.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg.seed)
    torch.set_num_threads(int(os.environ.get("BM_LEVEL_C_TORCH_THREADS", "2")))
    device = choose_device()

    teacher = mlp(
        cfg.proprioception_dim + cfg.vae_encoder_input_dim,
        cfg.teacher_hidden_dims,
        cfg.action_dim,
    ).to(device)
    vae = ConditionalVAE(cfg).to(device)
    diffusion = DiffusionTransformer(cfg).to(device)

    reference_and_anchor = torch.randn(cfg.batch_size, cfg.vae_encoder_input_dim, device=device)
    proprioception = torch.randn(cfg.batch_size, cfg.proprioception_dim, device=device)
    teacher_input = torch.cat([proprioception, reference_and_anchor], dim=-1)
    with torch.no_grad():
        target_action = teacher(teacher_input)

    vae_opt = torch.optim.Adam(vae.parameters(), lr=cfg.vae_learning_rate)
    vae_action, mu, logvar, latent = vae(reference_and_anchor, proprioception)
    reconstruction_loss = F.mse_loss(vae_action, target_action)
    kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    vae_loss = reconstruction_loss + cfg.vae_kl_coefficient * kl_loss
    vae_opt.zero_grad(set_to_none=True)
    vae_loss.backward()
    vae_grad_norm = torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1000.0)
    vae_opt.step()

    diffusion_opt = torch.optim.AdamW(
        diffusion.parameters(),
        lr=cfg.diffusion_learning_rate,
        weight_decay=cfg.diffusion_weight_decay,
    )
    x0 = torch.randn(
        cfg.batch_size,
        cfg.diffusion_sequence_length,
        cfg.diffusion_token_dim,
        device=device,
    )
    diffusion_steps = torch.randint(
        low=0,
        high=cfg.diffusion_denoising_steps,
        size=(cfg.batch_size, cfg.diffusion_sequence_length, 2),
        device=device,
        dtype=torch.long,
    )
    _betas, alpha_bars = make_diffusion_schedule(cfg.diffusion_denoising_steps, device)
    noise = torch.randn_like(x0)
    state_alpha_bar = alpha_bars[diffusion_steps[..., 0]].unsqueeze(-1)
    latent_alpha_bar = alpha_bars[diffusion_steps[..., 1]].unsqueeze(-1)
    alpha_bar = torch.cat(
        [
            state_alpha_bar.expand(-1, -1, cfg.state_dim),
            latent_alpha_bar.expand(-1, -1, cfg.vae_latent_dim),
        ],
        dim=-1,
    )
    xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise
    predicted_clean = diffusion(xt, diffusion_steps)
    diffusion_loss = F.mse_loss(predicted_clean, x0)
    diffusion_opt.zero_grad(set_to_none=True)
    diffusion_loss.backward()
    diffusion_grad_norm = torch.nn.utils.clip_grad_norm_(diffusion.parameters(), max_norm=1000.0)
    diffusion_opt.step()

    guided_tokens = x0.detach().clone().requires_grad_(True)
    costs = guidance_costs(guided_tokens, cfg)
    guidance_total = sum(costs.values())
    guidance_total.backward()
    guidance_grad_norm = guided_tokens.grad.norm()

    summary: dict[str, Any] = {
        "status": "ok",
        "scope": "synthetic architecture/loss/guidance smoke only; not a faithful trained reproduction",
        "no_long_training": True,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "seed": cfg.seed,
        "paper_parameters_used": {
            "vae_latent_dim": cfg.vae_latent_dim,
            "vae_encoder_hidden_dims": list(cfg.vae_encoder_hidden_dims),
            "vae_decoder_hidden_dims": list(cfg.vae_decoder_hidden_dims),
            "teacher_hidden_dims": list(cfg.teacher_hidden_dims),
            "vae_learning_rate": cfg.vae_learning_rate,
            "vae_kl_coefficient": cfg.vae_kl_coefficient,
            "paper_accumulated_gradient_steps": cfg.paper_accumulated_gradient_steps,
            "diffusion_horizon": cfg.diffusion_horizon,
            "observation_history": cfg.observation_history,
            "diffusion_embedding_dim": cfg.diffusion_embedding_dim,
            "diffusion_attention_heads": cfg.diffusion_attention_heads,
            "diffusion_transformer_layers": cfg.diffusion_transformer_layers,
            "diffusion_denoising_steps": cfg.diffusion_denoising_steps,
            "diffusion_batch_size_paper": cfg.diffusion_batch_size_paper,
            "diffusion_epochs_paper": cfg.diffusion_epochs_paper,
            "diffusion_learning_rate": cfg.diffusion_learning_rate,
            "diffusion_weight_decay": cfg.diffusion_weight_decay,
            "diffusion_warmup_gradient_steps": cfg.diffusion_warmup_gradient_steps,
            "diffusion_ema_power": cfg.diffusion_ema_power,
            "diffusion_ema_max": cfg.diffusion_ema_max,
        },
        "smoke_runtime_overrides": {
            "device_default": "cpu",
            "diffusion_embedding_dim": cfg.smoke_diffusion_embedding_dim,
            "diffusion_attention_heads": cfg.smoke_diffusion_attention_heads,
            "diffusion_transformer_layers": cfg.smoke_diffusion_transformer_layers,
            "reason": "keep the synthetic smoke fast and deterministic; paper hyperparameters remain recorded above",
        },
        "paper_fidelity_checks": {
            "diffusion_training_target": "clean_state_latent_trajectory",
            "diffusion_training_target_evidence": str(ROOT / "reproduction/paper/source/tex/method.tex:187-196"),
            "independent_state_latent_timesteps": True,
            "independent_timestep_evidence": str(ROOT / "reproduction/paper/source/tex/method.tex:171-185"),
            "reference_motion_conditioning_in_diffusion": False,
        },
        "synthetic_schema": asdict(cfg)
        | {
            "vae_encoder_input_dim": cfg.vae_encoder_input_dim,
            "vae_decoder_input_dim": cfg.vae_decoder_input_dim,
            "state_dim": cfg.state_dim,
            "diffusion_token_dim": cfg.diffusion_token_dim,
            "diffusion_sequence_length": cfg.diffusion_sequence_length,
        },
        "model_parameter_counts": {
            "teacher": parameter_count(teacher),
            "conditional_vae": parameter_count(vae),
            "diffusion_transformer": parameter_count(diffusion),
        },
        "shapes": {
            "reference_and_anchor": tensor_shape(reference_and_anchor),
            "proprioception": tensor_shape(proprioception),
            "teacher_action": tensor_shape(target_action),
            "vae_action": tensor_shape(vae_action),
            "vae_mu": tensor_shape(mu),
            "vae_logvar": tensor_shape(logvar),
            "vae_latent": tensor_shape(latent),
            "diffusion_x0": tensor_shape(x0),
            "diffusion_xt": tensor_shape(xt),
            "diffusion_predicted_clean": tensor_shape(predicted_clean),
            "diffusion_steps": tensor_shape(diffusion_steps),
            "diffusion_state_alpha_bar": tensor_shape(state_alpha_bar),
            "diffusion_latent_alpha_bar": tensor_shape(latent_alpha_bar),
        },
        "losses": {
            "vae_reconstruction_mse": float(reconstruction_loss.detach().cpu()),
            "vae_kl": float(kl_loss.detach().cpu()),
            "vae_total": float(vae_loss.detach().cpu()),
            "diffusion_clean_trajectory_mse": float(diffusion_loss.detach().cpu()),
            "guidance_total": float(guidance_total.detach().cpu()),
        },
        "guidance_costs": {key: float(value.detach().cpu()) for key, value in costs.items()},
        "gradient_norms": {
            "vae": float(vae_grad_norm.detach().cpu()),
            "diffusion": float(diffusion_grad_norm.detach().cpu()),
            "guidance_tokens": float(guidance_grad_norm.detach().cpu()),
        },
        "source_evidence": {
            "method_tex": str(ROOT / "reproduction/paper/source/tex/method.tex"),
            "hyperparameter_table_tex": str(ROOT / "reproduction/paper/source/root.tex"),
            "level_c_plan": str(ROOT / "reproduction/docs/level_c_diffusion_plan.md"),
        },
    }

    if not math.isfinite(summary["losses"]["vae_total"]):
        raise RuntimeError("non-finite VAE loss")
    if not math.isfinite(summary["losses"]["diffusion_clean_trajectory_mse"]):
        raise RuntimeError("non-finite diffusion loss")
    if summary["gradient_norms"]["guidance_tokens"] <= 0.0:
        raise RuntimeError("guidance costs produced zero gradient")

    json_path = OUT / "level_c_synthetic_smoke.json"
    tsv_path = OUT / "level_c_synthetic_smoke.tsv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, summary)
    print(json.dumps({"status": "ok", "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
