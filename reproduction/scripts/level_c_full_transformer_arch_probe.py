#!/usr/bin/env python3
"""Debug-only full-paper Transformer architecture probe for Level C diffusion.

This probe instantiates the paper-specified Transformer dimensions and runs one
tiny clean-trajectory prediction backward pass. It is not training and does not
replace the missing state-latent rollout dataset or checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DEFAULT_FIXTURE = ROOT / "reproduction/data/level_c_fixtures/walk1_subject1_frames_1_180_state_fixture.npz"
DEFAULT_MANIFEST = ROOT / "res/level_c/motion_state_fixture/walk1_subject1_frames_1_180_state_fixture.json"
OUT = ROOT / "res/level_c/full_transformer_arch_probe"


@dataclass(frozen=True)
class ProbeConfig:
    seed: int = 20260829
    batch_size: int = 1
    history: int = 4
    horizon: int = 16
    latent_dim: int = 32
    denoising_steps: int = 20
    embedding_dim: int = 512
    attention_heads: int = 8
    transformer_layers: int = 6
    dim_feedforward_multiplier: int = 4
    paper_batch_size: int = 512
    paper_epochs: int = 1000
    paper_learning_rate: float = 1e-4
    paper_weight_decay: float = 0.001
    paper_warmup_gradient_steps: int = 10000
    paper_ema_power: float = 0.75
    paper_ema_max: float = 0.9999

    @property
    def sequence_length(self) -> int:
        return self.history + 1 + self.horizon


class FullPaperDiffusionTransformer(nn.Module):
    def __init__(self, *, token_dim: int, cfg: ProbeConfig) -> None:
        super().__init__()
        self.input_proj = nn.Linear(token_dim, cfg.embedding_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.sequence_length, cfg.embedding_dim))
        self.state_step_embed = nn.Embedding(cfg.denoising_steps, cfg.embedding_dim)
        self.latent_step_embed = nn.Embedding(cfg.denoising_steps, cfg.embedding_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embedding_dim,
            nhead=cfg.attention_heads,
            dim_feedforward=cfg.embedding_dim * cfg.dim_feedforward_multiplier,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.transformer_layers)
        self.output_proj = nn.Linear(cfg.embedding_dim, token_dim)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, noisy_tau: torch.Tensor, denoising_steps: torch.Tensor) -> torch.Tensor:
        expected = noisy_tau.shape[:2] + (2,)
        if denoising_steps.shape != expected:
            raise ValueError(f"denoising_steps shape {tuple(denoising_steps.shape)} != expected {tuple(expected)}")
        x = self.input_proj(noisy_tau)
        x = x + self.pos_embed[:, : x.shape[1]]
        x = x + self.state_step_embed(denoising_steps[..., 0])
        x = x + self.latent_step_embed(denoising_steps[..., 1])
        return self.output_proj(self.encoder(x))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def diffusion_alpha_bars(steps: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    betas = torch.linspace(1e-4, 0.02, steps, device=device, dtype=dtype)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def choose_device(arg_device: str | None) -> torch.device:
    forced = arg_device or os.environ.get("BM_LEVEL_C_DEVICE")
    if forced:
        return torch.device(forced)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def grad_norm(module: nn.Module) -> float:
    total = torch.zeros((), device=next(module.parameters()).device)
    for param in module.parameters():
        if param.grad is not None:
            total = total + param.grad.detach().pow(2).sum()
    return float(torch.sqrt(total).cpu().item())


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


def load_tau(args: argparse.Namespace, cfg: ProbeConfig, device: torch.device) -> tuple[torch.Tensor, int, dict[str, Any]]:
    fixture = np.load(args.fixture_npz)
    manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    windows = fixture["candidate_hybrid_state_windows"]
    if windows.shape[1] != cfg.sequence_length:
        raise ValueError(f"fixture sequence length {windows.shape[1]} != expected {cfg.sequence_length}")
    if int(manifest["history"]) != cfg.history or int(manifest["horizon"]) != cfg.horizon:
        raise ValueError("fixture manifest history/horizon does not match paper probe config")

    rng = np.random.default_rng(cfg.seed)
    state = windows[: cfg.batch_size].astype(np.float32)
    latents = rng.standard_normal((cfg.batch_size, cfg.sequence_length, cfg.latent_dim), dtype=np.float32)
    tau = np.concatenate([state, latents], axis=-1)
    return torch.from_numpy(tau).to(device), int(state.shape[-1]), manifest


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    cfg = ProbeConfig(batch_size=args.batch_size, seed=args.seed)
    seed_everything(cfg.seed)
    torch.set_num_threads(args.torch_threads)
    device = choose_device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()

    clean_tau, state_dim, manifest = load_tau(args, cfg, device)
    token_dim = int(clean_tau.shape[-1])
    model = FullPaperDiffusionTransformer(token_dim=token_dim, cfg=cfg).to(device)
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

    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "paper-hyperparameter Transformer architecture forward/backward probe",
        "paper_evidence": {
            "diffusion_hyperparameters": str(ROOT / "reproduction/paper/source/root.tex:827-856"),
            "clean_trajectory_loss": str(ROOT / "reproduction/paper/source/tex/method.tex:187-196"),
            "individual_denoising_steps": str(ROOT / "reproduction/paper/source/tex/method.tex:171-185"),
        },
        "not_a_replacement_for": [
            "long diffusion training",
            "teacher rollout state-latent dataset",
            "trained checkpoint",
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
            "fixture_scope": manifest.get("scope"),
        },
        "model": {
            "class": "FullPaperDiffusionTransformer",
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
            "unique_state_steps": sorted(int(x) for x in torch.unique(steps[..., 0]).detach().cpu().tolist()),
            "unique_latent_steps": sorted(int(x) for x in torch.unique(steps[..., 1]).detach().cpu().tolist()),
        },
        "metrics": {
            "clean_trajectory_mse": float(loss.detach().cpu().item()),
            "total_grad_norm": total_grad_norm,
            "cuda_peak_memory_mb": cuda_peak_memory_mb,
        },
        "checks": {
            "uses_paper_embedding_dim": cfg.embedding_dim == 512,
            "uses_paper_attention_heads": cfg.attention_heads == 8,
            "uses_paper_transformer_layers": cfg.transformer_layers == 6,
            "uses_paper_denoising_steps": cfg.denoising_steps == 20,
            "uses_paper_history_horizon": cfg.history == 4 and cfg.horizon == 16,
            "step_tensor_is_independent_state_latent": list(steps.shape) == [cfg.batch_size, cfg.sequence_length, 2],
            "prediction_shape_matches_clean_tau": list(prediction.shape) == list(clean_tau.shape),
            "loss_is_finite": bool(torch.isfinite(loss).detach().cpu().item()),
            "grad_norm_is_positive": total_grad_norm > 0.0,
        },
    }

    npz_path = OUT / "level_c_full_transformer_arch_probe.npz"
    json_path = OUT / "level_c_full_transformer_arch_probe.json"
    tsv_path = OUT / "level_c_full_transformer_arch_probe.tsv"
    np.savez_compressed(
        npz_path,
        clean_tau=clean_tau.detach().cpu().numpy(),
        noisy_tau=noisy_tau.detach().cpu().numpy(),
        denoising_steps=steps.detach().cpu().numpy(),
        prediction=prediction.detach().cpu().numpy(),
    )
    summary["outputs"] = {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)}
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-npz", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--torch-threads", type=int, default=int(os.environ.get("BM_LEVEL_C_TORCH_THREADS", "2")))
    args = parser.parse_args()
    summary = run_probe(args)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"]}, sort_keys=True))


if __name__ == "__main__":
    main()
