#!/usr/bin/env python3
"""Debug-only NumPy single-batch overfit probe for Level C diffusion readiness."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DEFAULT_FIXTURE = ROOT / "reproduction/data/level_c_fixtures/walk1_subject1_frames_1_180_state_fixture.npz"
DEFAULT_MANIFEST = ROOT / "res/level_c/motion_state_fixture/walk1_subject1_frames_1_180_state_fixture.json"
OUT = ROOT / "res/level_c/single_batch_overfit_probe"


@dataclass(frozen=True)
class OverfitConfig:
    seed: int = 20260902
    batch_size: int = 4
    history: int = 4
    horizon: int = 16
    latent_dim: int = 32
    denoising_steps: int = 20
    ridge_lambda: float = 1e-6

    @property
    def sequence_length(self) -> int:
        return self.history + 1 + self.horizon


def alpha_bars(steps: int) -> np.ndarray:
    betas = np.linspace(1e-4, 0.02, steps, dtype=np.float64)
    alphas = 1.0 - betas
    return np.cumprod(alphas)


def one_hot(indices: np.ndarray, classes: int) -> np.ndarray:
    out = np.zeros(indices.shape + (classes,), dtype=np.float64)
    flat = out.reshape(-1, classes)
    flat[np.arange(indices.size), indices.reshape(-1)] = 1.0
    return out


def make_tau(cfg: OverfitConfig) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
    fixture = np.load(DEFAULT_FIXTURE)
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    windows = fixture["candidate_hybrid_state_windows"]
    if windows.shape[1] != cfg.sequence_length:
        raise ValueError(f"fixture sequence length {windows.shape[1]} != expected {cfg.sequence_length}")
    if int(manifest["history"]) != cfg.history or int(manifest["horizon"]) != cfg.horizon:
        raise ValueError("fixture manifest history/horizon does not match overfit config")

    count = min(cfg.batch_size, windows.shape[0])
    state = windows[:count].astype(np.float64)
    rng = np.random.default_rng(cfg.seed)
    latents = rng.standard_normal((count, cfg.sequence_length, cfg.latent_dim))
    tau = np.concatenate([state, latents], axis=-1)
    return tau, fixture["window_start_indices"][:count], int(state.shape[-1]), manifest


def make_noisy_tau(clean_tau: np.ndarray, state_dim: int, cfg: OverfitConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed + 1)
    steps = rng.integers(0, cfg.denoising_steps, size=clean_tau.shape[:2] + (2,), dtype=np.int64)
    noise = rng.standard_normal(clean_tau.shape)
    bars = alpha_bars(cfg.denoising_steps)
    state_alpha = np.repeat(bars[steps[..., 0]][..., None], state_dim, axis=-1)
    latent_alpha = np.repeat(bars[steps[..., 1]][..., None], cfg.latent_dim, axis=-1)
    token_alpha = np.concatenate([state_alpha, latent_alpha], axis=-1)
    noisy_tau = np.sqrt(token_alpha) * clean_tau + np.sqrt(1.0 - token_alpha) * noise
    return noisy_tau, steps


def feature_matrix(noisy_tau: np.ndarray, steps: np.ndarray, cfg: OverfitConfig) -> np.ndarray:
    b, t, d = noisy_tau.shape
    noisy_flat = noisy_tau.reshape(b * t, d)
    state_step = one_hot(steps[..., 0], cfg.denoising_steps).reshape(b * t, cfg.denoising_steps)
    latent_step = one_hot(steps[..., 1], cfg.denoising_steps).reshape(b * t, cfg.denoising_steps)
    bias = np.ones((b * t, 1), dtype=np.float64)
    return np.concatenate([noisy_flat, state_step, latent_step, bias], axis=1)


def ridge_fit(features: np.ndarray, targets: np.ndarray, ridge_lambda: float) -> np.ndarray:
    xtx = features.T @ features
    reg = ridge_lambda * np.eye(xtx.shape[0], dtype=np.float64)
    reg[-1, -1] = 0.0
    xty = features.T @ targets
    return np.linalg.solve(xtx + reg, xty)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["stage", "loss"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--ridge-lambda", type=float, default=1e-6)
    args = parser.parse_args()

    cfg = OverfitConfig(batch_size=args.batch_size, ridge_lambda=args.ridge_lambda)
    OUT.mkdir(parents=True, exist_ok=True)
    clean_tau, window_start_indices, state_dim, manifest = make_tau(cfg)
    noisy_tau, steps = make_noisy_tau(clean_tau, state_dim, cfg)
    features = feature_matrix(noisy_tau, steps, cfg)
    targets = clean_tau.reshape(features.shape[0], clean_tau.shape[-1])

    zero_prediction = np.zeros_like(targets)
    identity_prediction = noisy_tau.reshape(targets.shape)
    weights = ridge_fit(features, targets, cfg.ridge_lambda)
    fitted_prediction = features @ weights

    initial_loss = mse(identity_prediction, targets)
    zero_loss = mse(zero_prediction, targets)
    final_loss = mse(fitted_prediction, targets)
    rows = [
        {"stage": "zero_prediction", "loss": zero_loss},
        {"stage": "noisy_identity_baseline", "loss": initial_loss},
        {"stage": "ridge_overfit_solution", "loss": final_loss},
    ]

    json_path = OUT / "level_c_single_batch_overfit_probe.json"
    tsv_path = OUT / "level_c_single_batch_overfit_probe.tsv"
    npz_path = OUT / "level_c_single_batch_overfit_probe.npz"
    write_tsv(tsv_path, rows)
    np.savez_compressed(
        npz_path,
        clean_tau=clean_tau,
        noisy_tau=noisy_tau,
        diffusion_steps=steps,
        features=features,
        ridge_weights=weights,
        fitted_prediction=fitted_prediction.reshape(clean_tau.shape),
        window_start_indices=window_start_indices,
    )

    loss_reduction_ratio = (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0.0
    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "single-batch clean-trajectory overfit probe before any long Level C diffusion training",
        "paper_evidence": {
            "clean_trajectory_loss": str(ROOT / "reproduction/paper/source/tex/method.tex:187-196"),
            "diffusion_hyperparameters": str(ROOT / "reproduction/paper/source/root.tex:827-856"),
            "goal_phase7_overfit_gate": str(ROOT / "goal.md:1470-1484"),
        },
        "not_a_replacement_for": [
            "full diffusion training",
            "official BeyondMimic diffusion implementation",
            "teacher rollout state-latent dataset",
            "validation/test metrics",
            "checkpoint reproduction",
            "Fig. 5/Fig. 6 reproduction",
        ],
        "settings": {
            **asdict(cfg),
            "state_dim": state_dim,
            "token_dim": int(clean_tau.shape[-1]),
            "feature_dim": int(features.shape[-1]),
            "sample_count": int(features.shape[0]),
            "fixture_scope": manifest.get("scope"),
        },
        "metrics": {
            "zero_prediction_loss": zero_loss,
            "initial_noisy_identity_loss": initial_loss,
            "final_overfit_loss": final_loss,
            "loss_reduction": initial_loss - final_loss,
            "loss_reduction_ratio": loss_reduction_ratio,
            "condition_number_xtx": float(np.linalg.cond(features.T @ features + cfg.ridge_lambda * np.eye(features.shape[1]))),
            "rows": rows,
        },
        "checks": {
            "losses_are_finite": bool(np.all(np.isfinite([zero_loss, initial_loss, final_loss]))),
            "loss_decreases_vs_noisy_identity": final_loss < initial_loss,
            "loss_reduction_ratio_at_least_0_99": loss_reduction_ratio >= 0.99,
            "final_loss_below_1e_minus_8": final_loss < 1e-8,
            "uses_clean_trajectory_target": True,
            "uses_independent_state_latent_steps": bool(np.any(steps[..., 0] != steps[..., 1])),
            "debug_fixture_boundary_recorded": manifest.get("experiment_type") == "debug_only",
        },
        "interpretation": {
            "paper_level_status": "partial",
            "why_not_complete": (
                "The single-batch overfit gate proves the local noised trajectory target can be fit on the debug fixture. "
                "It is a pre-training math/data-path check, not trained diffusion reproduction."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
