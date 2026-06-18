#!/usr/bin/env python3
"""Debug-only VAE reparameterization, KL, and latent interpolation probe."""

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
OUT = ROOT / "res/level_c/vae_latent_probe"


@dataclass(frozen=True)
class VAELatentProbeConfig:
    seed: int = 20260915
    batch_size: int = 8
    interpolation_points: int = 11
    num_joints: int = 29
    reference_motion_dim: int = 58
    anchor_error_dim: int = 9
    proprioception_dim: int = 96
    latent_dim: int = 32
    encoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512)
    decoder_hidden_dims: tuple[int, ...] = (2048, 1024, 512)
    kl_coefficient: float = 0.01

    @property
    def encoder_input_dim(self) -> int:
        return self.reference_motion_dim + self.anchor_error_dim

    @property
    def decoder_input_dim(self) -> int:
        return self.latent_dim + self.proprioception_dim

    @property
    def action_dim(self) -> int:
        return self.num_joints


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
    def __init__(self, cfg: VAELatentProbeConfig) -> None:
        super().__init__()
        self.encoder = mlp(cfg.encoder_input_dim, cfg.encoder_hidden_dims, cfg.latent_dim * 2)
        self.decoder = mlp(cfg.decoder_input_dim, cfg.decoder_hidden_dims, cfg.action_dim)

    def encode(self, reference_and_anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self.encoder(reference_and_anchor)
        return stats.chunk(2, dim=-1)

    def decode(self, latent: torch.Tensor, proprioception: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([latent, proprioception], dim=-1))


def parameter_count(module: nn.Module) -> int:
    return int(sum(param.numel() for param in module.parameters()))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "seed",
        "kl_loss",
        "reparameterization_max_abs_error",
        "mean_latent_std",
        "interpolation_endpoint_start_error",
        "interpolation_endpoint_end_error",
        "max_neighbor_action_delta",
        "mean_neighbor_action_delta",
        "curvature_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def run_seed(seed: int, device: torch.device) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cfg = VAELatentProbeConfig(seed=seed)
    seed_everything(seed)
    vae = ConditionalVAE(cfg).to(device)
    vae.eval()

    reference_and_anchor = torch.randn(cfg.batch_size, cfg.encoder_input_dim, device=device)
    proprioception = torch.randn(cfg.batch_size, cfg.proprioception_dim, device=device)
    eps = torch.randn(cfg.batch_size, cfg.latent_dim, device=device)

    with torch.no_grad():
        mu, logvar = vae.encode(reference_and_anchor)
        std = torch.exp(0.5 * logvar)
        z = mu + std * eps
        reconstructed_z = mu + torch.exp(0.5 * logvar) * eps
        action = vae.decode(z, proprioception)
        deterministic_action = vae.decode(mu, proprioception)
        kl_per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        kl_loss = torch.mean(kl_per_sample)

        z0 = mu[0]
        z1 = mu[1]
        fixed_prop = proprioception[0].expand(cfg.interpolation_points, -1)
        alphas = torch.linspace(0.0, 1.0, cfg.interpolation_points, device=device).unsqueeze(-1)
        interpolated_latents = (1.0 - alphas) * z0 + alphas * z1
        interpolated_actions = vae.decode(interpolated_latents, fixed_prop)
        start_action = vae.decode(z0.unsqueeze(0), proprioception[0:1]).squeeze(0)
        end_action_same_prop = vae.decode(z1.unsqueeze(0), proprioception[0:1]).squeeze(0)

    action_np = interpolated_actions.detach().cpu().numpy()
    neighbor_delta = np.linalg.norm(np.diff(action_np, axis=0), axis=1)
    curvature = np.linalg.norm(action_np[2:] - 2.0 * action_np[1:-1] + action_np[:-2], axis=1)
    manual_kl = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
    row = {
        "seed": seed,
        "kl_loss": float(kl_loss.detach().cpu()),
        "manual_kl_loss": float(manual_kl.detach().cpu()),
        "reparameterization_max_abs_error": float(torch.max(torch.abs(z - reconstructed_z)).detach().cpu()),
        "mean_latent_std": float(torch.mean(std).detach().cpu()),
        "mean_latent_norm": float(torch.linalg.norm(z, dim=-1).mean().detach().cpu()),
        "deterministic_vs_sampled_action_mse": float(F.mse_loss(deterministic_action, action).detach().cpu()),
        "interpolation_endpoint_start_error": float(
            torch.linalg.norm(interpolated_actions[0] - start_action).detach().cpu()
        ),
        "interpolation_endpoint_end_error": float(
            torch.linalg.norm(interpolated_actions[-1] - end_action_same_prop).detach().cpu()
        ),
        "max_neighbor_action_delta": float(neighbor_delta.max()),
        "mean_neighbor_action_delta": float(neighbor_delta.mean()),
        "curvature_mean": float(curvature.mean()),
        "action_all_finite": bool(torch.all(torch.isfinite(action)).detach().cpu()),
        "interpolation_actions_all_finite": bool(torch.all(torch.isfinite(interpolated_actions)).detach().cpu()),
        "vae_parameter_count": parameter_count(vae),
    }
    payload = {
        f"seed_{seed}_mu": mu.detach().cpu().numpy(),
        f"seed_{seed}_logvar": logvar.detach().cpu().numpy(),
        f"seed_{seed}_eps": eps.detach().cpu().numpy(),
        f"seed_{seed}_z": z.detach().cpu().numpy(),
        f"seed_{seed}_interpolated_actions": action_np,
    }
    return row, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", default="20260915,20260916,20260917")
    args = parser.parse_args()

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if len(seeds) < 3:
        raise ValueError("VAE latent probe requires at least 3 seeds")
    device = torch.device(args.device)
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    payload: dict[str, np.ndarray] = {}
    for seed in seeds:
        row, seed_payload = run_seed(seed, device)
        rows.append(row)
        payload.update(seed_payload)

    cfg = VAELatentProbeConfig(seed=seeds[0])
    metric_names = [
        "kl_loss",
        "reparameterization_max_abs_error",
        "mean_latent_std",
        "mean_latent_norm",
        "deterministic_vs_sampled_action_mse",
        "interpolation_endpoint_start_error",
        "interpolation_endpoint_end_error",
        "max_neighbor_action_delta",
        "mean_neighbor_action_delta",
        "curvature_mean",
    ]
    statistics = {name: stats([row[name] for row in rows]) for name in metric_names}

    json_path = OUT / "level_c_vae_latent_probe.json"
    tsv_path = OUT / "level_c_vae_latent_probe.tsv"
    npz_path = OUT / "level_c_vae_latent_probe.npz"
    write_tsv(tsv_path, rows)
    np.savez_compressed(npz_path, **payload)
    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "VAE reparameterization, KL, latent smoothness, and interpolation schema probe",
        "paper_evidence": {
            "vae_encoder_decoder_equations": str(ROOT / "reproduction/paper/source/tex/method.tex:150-170"),
            "vae_hyperparameters": str(ROOT / "reproduction/paper/source/root.tex:801-825"),
            "goal_vae_metrics": str(ROOT / "goal.md:1582-1594"),
            "goal_reparameterization_checklist": str(ROOT / "goal.md:1733-1738"),
        },
        "not_a_replacement_for": [
            "trained VAE checkpoint",
            "true DAgger rollout",
            "teacher-student closed-loop evaluation",
            "paper latent analysis over real rollout dataset",
        ],
        "settings": asdict(cfg) | {"seeds": seeds, "device": str(device)},
        "rows": rows,
        "statistics": statistics,
        "checks": {
            "at_least_three_seeds": len(seeds) >= 3,
            "latent_dim_matches_paper": cfg.latent_dim == 32,
            "encoder_dims_match_paper": cfg.encoder_hidden_dims == (2048, 1024, 512),
            "decoder_dims_match_paper": cfg.decoder_hidden_dims == (2048, 1024, 512),
            "kl_coefficient_matches_paper": math.isclose(cfg.kl_coefficient, 0.01),
            "all_reparameterization_exact": bool(
                all(row["reparameterization_max_abs_error"] == 0.0 for row in rows)
            ),
            "all_kl_formula_matches_manual": bool(
                all(math.isclose(row["kl_loss"], row["manual_kl_loss"], rel_tol=0.0, abs_tol=1e-12) for row in rows)
            ),
            "all_latent_std_positive": bool(all(row["mean_latent_std"] > 0.0 for row in rows)),
            "all_interpolation_endpoints_match": bool(
                all(row["interpolation_endpoint_start_error"] < 1e-6 for row in rows)
                and all(row["interpolation_endpoint_end_error"] < 1e-6 for row in rows)
            ),
            "all_interpolation_actions_finite": bool(all(row["interpolation_actions_all_finite"] for row in rows)),
            "all_sampled_actions_finite": bool(all(row["action_all_finite"] for row in rows)),
            "debug_only_boundary_recorded": True,
        },
        "interpretation": {
            "paper_level_status": "partial",
            "why_not_complete": (
                "This exercises the VAE latent math and interpolation path with paper dimensions on synthetic inputs. "
                "It does not replace true DAgger rollout, a trained VAE checkpoint, or paper latent analysis over "
                "state-action trajectories."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
