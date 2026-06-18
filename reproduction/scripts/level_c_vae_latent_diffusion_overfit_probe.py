#!/usr/bin/env python3
"""Diffusion clean-trajectory overfit probe using debug VAE latents.

This mirrors the paper-state overfit gate, but replaces random synthetic
latents with nonzero latents exported by the tiny debug VAE artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

import level_c_paper_state_overfit_probe as paper_overfit


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
VAE_LATENT_JSON = ROOT / "res/level_c/vae_debug_overfit_latent_artifact/level_c_vae_debug_overfit_latent_artifact.json"
VAE_LATENT_NPZ = ROOT / "reproduction/data/level_c_vae_debug_latents/debug_tiny_vae_state_latent_windows.npz"
OUT = ROOT / "res/level_c/vae_latent_diffusion_overfit_probe"


@dataclass(frozen=True)
class VaeLatentDiffusionConfig:
    seed: int = 20260918
    history: int = 4
    horizon: int = 16
    state_dim: int = 99
    latent_dim: int = 32
    denoising_steps: int = 20
    ridge_lambda: float = 1e-8

    @property
    def sequence_length(self) -> int:
        return self.history + 1 + self.horizon


def load_dataset(cfg: VaeLatentDiffusionConfig) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    manifest = json.loads(VAE_LATENT_JSON.read_text(encoding="utf-8"))
    rows = manifest["rows"]
    with np.load(VAE_LATENT_NPZ) as data:
        taus: list[np.ndarray] = []
        motion_name_to_id: dict[str, int] = {}
        motion_ids: list[int] = []
        for row in rows:
            sample_id = row["sample_id"]
            states = data[f"{sample_id}_states"].astype(np.float64)
            latents = data[f"{sample_id}_latents"].astype(np.float64)
            if states.shape != (cfg.sequence_length, cfg.state_dim):
                raise ValueError(f"{sample_id}: state shape {states.shape}")
            if latents.shape != (cfg.sequence_length, cfg.latent_dim):
                raise ValueError(f"{sample_id}: latent shape {latents.shape}")
            taus.append(np.concatenate([states, latents], axis=-1))
            source_motion = row["source_motion"]
            if source_motion not in motion_name_to_id:
                motion_name_to_id[source_motion] = len(motion_name_to_id)
            motion_ids.append(motion_name_to_id[source_motion])
    return np.stack(taus, axis=0), np.asarray(motion_ids, dtype=np.int64), manifest


def make_noisy_tau(
    clean_tau: np.ndarray,
    cfg: VaeLatentDiffusionConfig,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed + 1)
    steps = rng.integers(0, cfg.denoising_steps, size=clean_tau.shape[:2] + (2,), dtype=np.int64)
    noise = rng.standard_normal(clean_tau.shape)
    bars = paper_overfit.alpha_bars(cfg.denoising_steps)
    state_alpha = np.repeat(bars[steps[..., 0]][..., None], cfg.state_dim, axis=-1)
    latent_alpha = np.repeat(bars[steps[..., 1]][..., None], cfg.latent_dim, axis=-1)
    alpha = np.concatenate([state_alpha, latent_alpha], axis=-1)
    return np.sqrt(alpha) * clean_tau + np.sqrt(1.0 - alpha) * noise, steps


def feature_matrix(
    noisy_tau: np.ndarray,
    steps: np.ndarray,
    motion_ids: np.ndarray,
    cfg: VaeLatentDiffusionConfig,
) -> np.ndarray:
    b, t, d = noisy_tau.shape
    sample_count = b * t
    motion_per_token = np.repeat(motion_ids, t)
    return np.concatenate(
        [
            noisy_tau.reshape(sample_count, d),
            paper_overfit.one_hot(steps[..., 0], cfg.denoising_steps).reshape(sample_count, cfg.denoising_steps),
            paper_overfit.one_hot(steps[..., 1], cfg.denoising_steps).reshape(sample_count, cfg.denoising_steps),
            paper_overfit.one_hot(motion_per_token, int(motion_ids.max()) + 1),
            np.eye(sample_count, dtype=np.float64),
            np.ones((sample_count, 1), dtype=np.float64),
        ],
        axis=1,
    )


def expand_window_indices(window_indices: np.ndarray, sequence_length: int) -> np.ndarray:
    return np.concatenate(
        [np.arange(idx * sequence_length, (idx + 1) * sequence_length, dtype=np.int64) for idx in window_indices]
    )


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["subset", "window_count", "baseline_loss", "overfit_loss", "loss_reduction_ratio"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--ridge-lambda", type=float, default=1e-8)
    args = parser.parse_args()

    cfg = replace(VaeLatentDiffusionConfig(seed=args.seed, ridge_lambda=args.ridge_lambda))
    OUT.mkdir(parents=True, exist_ok=True)
    clean_tau, motion_ids, latent_manifest = load_dataset(cfg)
    noisy_tau, steps = make_noisy_tau(clean_tau, cfg)
    x = feature_matrix(noisy_tau, steps, motion_ids, cfg)
    y = clean_tau.reshape(x.shape[0], clean_tau.shape[-1])
    baseline = noisy_tau.reshape(y.shape)
    weights = paper_overfit.ridge_fit(x, y, cfg.ridge_lambda)
    pred = x @ weights

    rows: list[dict[str, Any]] = []
    metrics: dict[str, float] = {}
    subsets: list[tuple[str, np.ndarray]] = [("all_debug_vae_latent_windows", np.arange(clean_tau.shape[0]))]
    motion_names = sorted({row["source_motion"] for row in latent_manifest["rows"]})
    for motion_id, name in enumerate(motion_names):
        subsets.append((f"motion:{name}", np.nonzero(motion_ids == motion_id)[0]))
    for subset, window_idx in subsets:
        token_idx = expand_window_indices(window_idx, cfg.sequence_length)
        baseline_loss = paper_overfit.mse(baseline[token_idx], y[token_idx])
        overfit_loss = paper_overfit.mse(pred[token_idx], y[token_idx])
        ratio = (baseline_loss - overfit_loss) / baseline_loss if baseline_loss > 0 else 0.0
        rows.append(
            {
                "subset": subset,
                "window_count": int(len(window_idx)),
                "baseline_loss": baseline_loss,
                "overfit_loss": overfit_loss,
                "loss_reduction_ratio": ratio,
            }
        )
        key = subset.replace(":", "_")
        metrics[f"{key}_baseline_loss"] = baseline_loss
        metrics[f"{key}_overfit_loss"] = overfit_loss
        metrics[f"{key}_loss_reduction_ratio"] = ratio

    json_path = OUT / "level_c_vae_latent_diffusion_overfit_probe.json"
    tsv_path = OUT / "level_c_vae_latent_diffusion_overfit_probe.tsv"
    npz_path = OUT / "level_c_vae_latent_diffusion_overfit_probe.npz"
    write_tsv(tsv_path, rows)
    np.savez_compressed(
        npz_path,
        clean_tau=clean_tau,
        noisy_tau=noisy_tau,
        diffusion_steps=steps,
        motion_ids=motion_ids,
        prediction=pred.reshape(clean_tau.shape),
        ridge_weights=weights,
    )

    all_key = "all_debug_vae_latent_windows"
    checks = {
        "debug_vae_latent_artifact_status_ok": latent_manifest["status"] == "ok",
        "debug_vae_latents_nonzero": latent_manifest["checks"]["all_latents_nonzero"],
        "uses_paper_state_dim_99": cfg.state_dim == 99,
        "uses_debug_vae_latent_dim_32": cfg.latent_dim == 32,
        "token_dim_131": clean_tau.shape[-1] == 131,
        "window_count_84": clean_tau.shape[0] == 84,
        "uses_multiple_motions": len(set(motion_ids.tolist())) == 3,
        "uses_independent_state_latent_steps": bool(np.any(steps[..., 0] != steps[..., 1])),
        "loss_decreases": metrics[f"{all_key}_overfit_loss"] < metrics[f"{all_key}_baseline_loss"],
        "loss_reduction_ratio_at_least_0_99": metrics[f"{all_key}_loss_reduction_ratio"] >= 0.99,
        "final_loss_below_1e_minus_8": metrics[f"{all_key}_overfit_loss"] < 1e-8,
        "overparameterized_memorization_basis_recorded": True,
        "debug_only_boundary_recorded": True,
        "does_not_claim_true_vae_rollout": True,
        "does_not_claim_trained_diffusion_checkpoint": True,
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "debug_only",
        "scope": "clean-trajectory overfit gate using paper-formula states and nonzero tiny-VAE debug latents",
        "paper_evidence": {
            "clean_trajectory_loss": str(ROOT / "reproduction/paper/source/tex/method.tex:187-196"),
            "state_latent_dataset": str(ROOT / "reproduction/paper/source/root.tex:536-546"),
            "debug_vae_latents": str(VAE_LATENT_JSON),
        },
        "not_a_replacement_for": [
            "true VAE rollout state-latent dataset",
            "trained diffusion Transformer",
            "held-out diffusion evaluation",
            "paper Fig. 5/Fig. 6 reproduction",
        ],
        "settings": {
            **asdict(cfg),
            "token_dim": int(clean_tau.shape[-1]),
            "motion_count": int(len(set(motion_ids.tolist()))),
            "window_count": int(clean_tau.shape[0]),
            "sample_count": int(x.shape[0]),
            "feature_dim": int(x.shape[1]),
            "uses_overparameterized_token_identity_basis": True,
            "latent_source": "debug_tiny_vae_mu_nonzero_synthetic_teacher",
        },
        "metrics": {**metrics, "rows": rows},
        "checks": checks,
        "interpretation": {
            "paper_level_status": "partial",
            "why_not_complete": (
                "This confirms the local diffusion clean-trajectory loss path can consume nonzero debug VAE latents. "
                "It is still an overparameterized memorization gate using synthetic-teacher VAE latents, not true "
                "VAE rollout data, a trained diffusion Transformer, or paper evaluation."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
