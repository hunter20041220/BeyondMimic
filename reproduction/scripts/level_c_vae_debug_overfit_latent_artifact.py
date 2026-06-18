#!/usr/bin/env python3
"""Train a tiny debug VAE on paper-state windows and export nonzero latents.

This is a local smoke/overfit gate for the state-latent data path. It uses
deterministic synthetic teacher actions derived from paper-state windows, not
true DAgger rollout data or trained motion-tracking teacher policies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SRC = ROOT / "reproduction/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beyondmimic_reimpl.trajectory import build_state_latent_window, split_counts, stack_state_latent_tokens


PAPER_STATE_JSON = ROOT / "res/level_c/paper_state_windows/level_c_paper_state_windows.json"
SPLIT_JSON = ROOT / "res/level_c/small_dataset_split_manifest/small_dataset_split_manifest_summary.json"
SPLIT_TSV = ROOT / "res/level_c/small_dataset_split_manifest/small_dataset_window_provenance.tsv"
PAPER_STATE_DIR = ROOT / "reproduction/data/level_c_paper_state_windows"
OUT = ROOT / "res/level_c/vae_debug_overfit_latent_artifact"
OUT_DATA = ROOT / "reproduction/data/level_c_vae_debug_latents"


@dataclass(frozen=True)
class DebugVAEConfig:
    seed: int = 20260916
    latent_dim: int = 32
    encoder_input_dim: int = 67
    proprioception_dim: int = 96
    action_dim: int = 29
    encoder_hidden_dims: tuple[int, ...] = (256, 128)
    decoder_hidden_dims: tuple[int, ...] = (256, 128)
    teacher_seed: int = 20260917
    epochs: int = 350
    learning_rate: float = 1e-3
    kl_coefficient: float = 1e-4


class TinyConditionalVAE(nn.Module):
    def __init__(self, cfg: DebugVAEConfig) -> None:
        super().__init__()
        self.encoder = mlp(cfg.encoder_input_dim, cfg.encoder_hidden_dims, cfg.latent_dim * 2)
        self.decoder = mlp(cfg.latent_dim + cfg.proprioception_dim, cfg.decoder_hidden_dims, cfg.action_dim)

    def forward(
        self,
        encoder_input: torch.Tensor,
        proprioception: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stats = self.encoder(encoder_input)
        mu, logvar = stats.chunk(2, dim=-1)
        if deterministic:
            z = mu
        else:
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        action = self.decoder(torch.cat([z, proprioception], dim=-1))
        return action, mu, logvar, z


def mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for hidden in hidden_dims:
        layers.extend([nn.Linear(prev, hidden), nn.ELU()])
        prev = hidden
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def paper_state_npz(name: str) -> Path:
    return PAPER_STATE_DIR / f"{name}_paper_state_windows.npz"


def projection_matrix(seed: int, in_dim: int, out_dim: int, scale: float = 0.15) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.normal(0.0, scale, size=(in_dim, out_dim))
    return mat.astype(np.float32)


def build_dataset(cfg: DebugVAEConfig) -> dict[str, Any]:
    split_summary = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    provenance = read_tsv(SPLIT_TSV)
    by_motion: dict[str, list[dict[str, str]]] = {}
    for row in provenance:
        by_motion.setdefault(row["source_motion"], []).append(row)

    prop_proj = projection_matrix(cfg.seed + 11, 99, cfg.proprioception_dim)
    teacher_proj = projection_matrix(cfg.teacher_seed, 99 + cfg.proprioception_dim, cfg.action_dim, scale=0.08)
    teacher_bias = np.linspace(-0.05, 0.05, cfg.action_dim, dtype=np.float32)

    states_rows: list[np.ndarray] = []
    encoder_rows: list[np.ndarray] = []
    proprio_rows: list[np.ndarray] = []
    action_rows: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    window_states: list[np.ndarray] = []
    window_proprio: list[np.ndarray] = []
    window_actions: list[np.ndarray] = []
    window_meta: list[dict[str, Any]] = []

    for motion in split_summary["motions"]:
        name = motion["name"]
        with np.load(paper_state_npz(name)) as data:
            states = data["paper_state_windows"].astype(np.float32)
        motion_rows = sorted(by_motion[name], key=lambda row: int(row["start_timestep"]))
        if len(motion_rows) != states.shape[0]:
            raise ValueError(f"{name}: provenance rows {len(motion_rows)} != windows {states.shape[0]}")
        for idx, prov in enumerate(motion_rows):
            state_window = states[idx]
            proprio = np.tanh(state_window @ prop_proj)
            teacher_input = np.concatenate([state_window, proprio], axis=-1)
            teacher_action = np.tanh(teacher_input @ teacher_proj + teacher_bias)
            encoder_input = state_window[:, : cfg.encoder_input_dim]
            window_states.append(state_window)
            window_proprio.append(proprio.astype(np.float32))
            window_actions.append(teacher_action.astype(np.float32))
            window_meta.append(
                {
                    "sample_id": prov["sample_id"],
                    "source_motion": name,
                    "start_timestep": int(prov["start_timestep"]),
                    "split": prov["motion_split"],
                    "accept_reject": prov["accept_reject"],
                }
            )
            for t in range(state_window.shape[0]):
                states_rows.append(state_window[t])
                encoder_rows.append(encoder_input[t])
                proprio_rows.append(proprio[t])
                action_rows.append(teacher_action[t])
                meta_rows.append(window_meta[-1] | {"token_index": t})

    return {
        "states": np.stack(states_rows).astype(np.float32),
        "encoder_input": np.stack(encoder_rows).astype(np.float32),
        "proprioception": np.stack(proprio_rows).astype(np.float32),
        "teacher_action": np.stack(action_rows).astype(np.float32),
        "meta_rows": meta_rows,
        "window_states": np.stack(window_states).astype(np.float32),
        "window_proprioception": np.stack(window_proprio).astype(np.float32),
        "window_teacher_action": np.stack(window_actions).astype(np.float32),
        "window_meta": window_meta,
        "prop_projection": prop_proj,
        "teacher_projection": teacher_proj,
        "teacher_bias": teacher_bias,
    }


def train_debug_vae(cfg: DebugVAEConfig, data: dict[str, Any], device: torch.device) -> tuple[TinyConditionalVAE, dict[str, float]]:
    seed_everything(cfg.seed)
    model = TinyConditionalVAE(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    encoder_input = torch.from_numpy(data["encoder_input"]).to(device)
    proprioception = torch.from_numpy(data["proprioception"]).to(device)
    teacher_action = torch.from_numpy(data["teacher_action"]).to(device)

    with torch.no_grad():
        initial_pred, initial_mu, initial_logvar, _ = model(encoder_input, proprioception, deterministic=True)
        initial_recon = F.mse_loss(initial_pred, teacher_action).item()
        initial_kl = (-0.5 * torch.mean(1.0 + initial_logvar - initial_mu.pow(2) - initial_logvar.exp())).item()

    last_loss = math.inf
    for _ in range(cfg.epochs):
        pred, mu, logvar, _ = model(encoder_input, proprioception, deterministic=False)
        recon = F.mse_loss(pred, teacher_action)
        kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + cfg.kl_coefficient * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())

    with torch.no_grad():
        final_pred, final_mu, final_logvar, _ = model(encoder_input, proprioception, deterministic=True)
        final_recon = F.mse_loss(final_pred, teacher_action).item()
        final_kl = (-0.5 * torch.mean(1.0 + final_logvar - final_mu.pow(2) - final_logvar.exp())).item()
        latent_std_mean = float(torch.exp(0.5 * final_logvar).mean().cpu())
        mu_abs_mean = float(final_mu.abs().mean().cpu())
        mu_std = float(final_mu.std(unbiased=False).cpu())
    return model, {
        "initial_reconstruction_mse": float(initial_recon),
        "initial_kl": float(initial_kl),
        "final_total_loss": last_loss,
        "final_reconstruction_mse": float(final_recon),
        "final_kl": float(final_kl),
        "reconstruction_loss_reduction_ratio": float((initial_recon - final_recon) / max(initial_recon, 1e-12)),
        "latent_std_mean": latent_std_mean,
        "latent_mu_abs_mean": mu_abs_mean,
        "latent_mu_std": mu_std,
    }


def export_windows(cfg: DebugVAEConfig, model: TinyConditionalVAE, data: dict[str, Any], device: torch.device) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    payload: dict[str, np.ndarray] = {
        "proprioception_projection": data["prop_projection"],
        "synthetic_teacher_projection": data["teacher_projection"],
        "synthetic_teacher_bias": data["teacher_bias"],
    }
    model.eval()
    with torch.no_grad():
        for idx, meta in enumerate(data["window_meta"]):
            enc = torch.from_numpy(data["window_states"][idx, :, : cfg.encoder_input_dim]).to(device)
            prop = torch.from_numpy(data["window_proprioception"][idx]).to(device)
            action, mu, logvar, _ = model(enc, prop, deterministic=True)
            states = data["window_states"][idx].astype(np.float64)
            latents = mu.cpu().numpy().astype(np.float64)
            tokens = stack_state_latent_tokens(states, latents)
            window = build_state_latent_window(
                sample_id=meta["sample_id"],
                source_motion=meta["source_motion"],
                start_timestep=int(meta["start_timestep"]),
                split=meta["split"],  # type: ignore[arg-type]
                accepted=meta["accept_reject"].startswith("accepted_debug"),
                states=states,
                latents=latents,
            )
            key = meta["sample_id"]
            payload[f"{key}_states"] = window.states
            payload[f"{key}_latents"] = window.latents
            payload[f"{key}_teacher_action"] = data["window_teacher_action"][idx].astype(np.float64)
            payload[f"{key}_decoded_action"] = action.cpu().numpy().astype(np.float64)
            payload[f"{key}_logvar"] = logvar.cpu().numpy().astype(np.float64)
            rows.append(
                {
                    "sample_id": window.sample_id,
                    "source_motion": window.source_motion,
                    "start_timestep": window.start_timestep,
                    "split": window.split,
                    "accepted": window.accepted,
                    "state_shape": list(window.states.shape),
                    "latent_shape": list(window.latents.shape),
                    "token_shape": list(tokens.shape),
                    "latent_source": "debug_tiny_vae_mu_nonzero_synthetic_teacher",
                    "latent_abs_mean": float(np.mean(np.abs(window.latents))),
                    "latent_std": float(np.std(window.latents)),
                    "finite": bool(np.isfinite(tokens).all()),
                }
            )
    return rows, payload


def write_rows_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "sample_id",
        "source_motion",
        "start_timestep",
        "split",
        "accepted",
        "state_shape",
        "latent_shape",
        "token_shape",
        "latent_source",
        "latent_abs_mean",
        "latent_std",
        "finite",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key]) if isinstance(row[key], list) else row[key] for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    cfg = DebugVAEConfig(seed=args.seed, epochs=args.epochs)
    device = torch.device(args.device)
    data = build_dataset(cfg)
    model, metrics = train_debug_vae(cfg, data, device)
    rows, payload = export_windows(cfg, model, data, device)

    counts = split_counts(
        [
            build_state_latent_window(
                sample_id=row["sample_id"],
                source_motion=row["source_motion"],
                start_timestep=row["start_timestep"],
                split=row["split"],  # type: ignore[arg-type]
                accepted=row["accepted"],
                states=payload[f"{row['sample_id']}_states"],
                latents=payload[f"{row['sample_id']}_latents"],
            )
            for row in rows
        ]
    )
    latent_abs = np.asarray([row["latent_abs_mean"] for row in rows], dtype=np.float64)
    latent_std = np.asarray([row["latent_std"] for row in rows], dtype=np.float64)
    npz_path = OUT_DATA / "debug_tiny_vae_state_latent_windows.npz"
    json_path = OUT / "level_c_vae_debug_overfit_latent_artifact.json"
    tsv_path = OUT / "level_c_vae_debug_overfit_latent_artifact.tsv"
    np.savez_compressed(npz_path, **payload)
    write_rows_tsv(tsv_path, rows)

    checks = {
        "paper_state_inputs_exist": PAPER_STATE_JSON.is_file() and SPLIT_JSON.is_file() and SPLIT_TSV.is_file(),
        "uses_package_state_latent_api": True,
        "row_count_84": len(rows) == 84,
        "split_counts_match_manifest": counts == {"train": 28, "validation": 28, "test": 28},
        "all_tokens_shape_21x131": all(row["token_shape"] == [21, 131] for row in rows),
        "all_latents_nonzero": bool(np.min(latent_abs) > 1e-6),
        "all_latents_finite": all(row["finite"] for row in rows),
        "latent_dim_32": cfg.latent_dim == 32 and all(row["latent_shape"] == [21, 32] for row in rows),
        "reconstruction_loss_decreases": metrics["final_reconstruction_mse"] < metrics["initial_reconstruction_mse"],
        "loss_reduction_ratio_above_half": metrics["reconstruction_loss_reduction_ratio"] > 0.5,
        "kl_finite_positive": math.isfinite(metrics["final_kl"]) and metrics["final_kl"] >= 0.0,
        "debug_only_boundary_recorded": True,
        "does_not_claim_true_dagger_rollout": True,
        "does_not_claim_trained_paper_checkpoint": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "debug_only_vae_overfit_latent_artifact",
        "scope": "tiny CPU VAE overfit gate that exports nonzero 32-D debug latents for local paper-state windows",
        "paper_evidence": {
            "vae_stage": str(ROOT / "reproduction/paper/source/root.tex:253"),
            "vae_elbo": str(ROOT / "reproduction/paper/source/tex/method.tex:109-115"),
            "state_latent_dataset": str(ROOT / "reproduction/paper/source/root.tex:536-546"),
            "goal_level_c": str(ROOT / "goal.md:1148-1190,1431-1467"),
        },
        "not_a_replacement_for": [
            "true DAgger rollout",
            "trained paper conditional VAE checkpoint",
            "VAE rollout stability evaluation",
            "paper state-latent trajectory dataset",
            "latent-space t-SNE or paper Fig. 5 analysis",
        ],
        "settings": asdict(cfg) | {"device": str(device)},
        "metrics": {
            **metrics,
            "row_count": len(rows),
            "token_count": int(len(rows) * 21),
            "split_counts": counts,
            "latent_abs_mean_min": float(np.min(latent_abs)),
            "latent_abs_mean_mean": float(np.mean(latent_abs)),
            "latent_std_min": float(np.min(latent_std)),
            "latent_std_mean": float(np.mean(latent_std)),
        },
        "checks": checks,
        "rows": rows,
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "partial",
            "why_not_complete": (
                "This replaces zero-placeholder schema-only latents with nonzero debug latents from a tiny VAE overfit "
                "gate, but the teacher actions are deterministic synthetic projections of local paper-state windows. "
                "It is not a true DAgger rollout, trained paper VAE, or accepted VAE rollout dataset."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(json_path), "npz": str(npz_path)}, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
