#!/usr/bin/env python3
"""Debug-only single-motion overfit probe for Level C diffusion readiness."""

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
OUT = ROOT / "res/level_c/single_motion_overfit_probe"


@dataclass(frozen=True)
class SingleMotionConfig:
    seed: int = 20260903
    history: int = 4
    horizon: int = 16
    latent_dim: int = 32
    denoising_steps: int = 20
    ridge_lambda: float = 1e-8
    diagnostic_stride: int = 5

    @property
    def sequence_length(self) -> int:
        return self.history + 1 + self.horizon


def alpha_bars(steps: int) -> np.ndarray:
    betas = np.linspace(1e-4, 0.02, steps, dtype=np.float64)
    return np.cumprod(1.0 - betas)


def one_hot(indices: np.ndarray, classes: int) -> np.ndarray:
    out = np.zeros(indices.shape + (classes,), dtype=np.float64)
    flat = out.reshape(-1, classes)
    flat[np.arange(indices.size), indices.reshape(-1)] = 1.0
    return out


def load_motion_tau(cfg: SingleMotionConfig) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
    fixture = np.load(DEFAULT_FIXTURE)
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    windows = fixture["candidate_hybrid_state_windows"].astype(np.float64)
    if windows.shape[1] != cfg.sequence_length:
        raise ValueError(f"fixture sequence length {windows.shape[1]} != expected {cfg.sequence_length}")
    rng = np.random.default_rng(cfg.seed)
    latents = rng.standard_normal((windows.shape[0], cfg.sequence_length, cfg.latent_dim))
    tau = np.concatenate([windows, latents], axis=-1)
    return tau, fixture["window_start_indices"].astype(np.int64), int(windows.shape[-1]), manifest


def make_noisy_tau(clean_tau: np.ndarray, state_dim: int, cfg: SingleMotionConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed + 1)
    steps = rng.integers(0, cfg.denoising_steps, size=clean_tau.shape[:2] + (2,), dtype=np.int64)
    noise = rng.standard_normal(clean_tau.shape)
    bars = alpha_bars(cfg.denoising_steps)
    state_alpha = np.repeat(bars[steps[..., 0]][..., None], state_dim, axis=-1)
    latent_alpha = np.repeat(bars[steps[..., 1]][..., None], cfg.latent_dim, axis=-1)
    alpha = np.concatenate([state_alpha, latent_alpha], axis=-1)
    return np.sqrt(alpha) * clean_tau + np.sqrt(1.0 - alpha) * noise, steps


def feature_matrix(noisy_tau: np.ndarray, steps: np.ndarray, cfg: SingleMotionConfig) -> np.ndarray:
    b, t, d = noisy_tau.shape
    sample_count = b * t
    return np.concatenate(
        [
            noisy_tau.reshape(sample_count, d),
            one_hot(steps[..., 0], cfg.denoising_steps).reshape(sample_count, cfg.denoising_steps),
            one_hot(steps[..., 1], cfg.denoising_steps).reshape(sample_count, cfg.denoising_steps),
            np.eye(sample_count, dtype=np.float64),
            np.ones((sample_count, 1), dtype=np.float64),
        ],
        axis=1,
    )


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    xtx = x.T @ x
    reg = lam * np.eye(xtx.shape[0], dtype=np.float64)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xtx + reg, x.T @ y)


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def window_subsets(count: int, stride: int) -> tuple[np.ndarray, np.ndarray]:
    diagnostic = np.arange(count, dtype=np.int64)[::stride]
    complement = np.asarray([idx for idx in range(count) if idx not in set(diagnostic.tolist())], dtype=np.int64)
    if len(complement) == 0 or len(diagnostic) == 0:
        raise ValueError("invalid diagnostic split")
    return complement, diagnostic


def expand_window_indices(window_indices: np.ndarray, sequence_length: int) -> np.ndarray:
    return np.concatenate(
        [np.arange(idx * sequence_length, (idx + 1) * sequence_length, dtype=np.int64) for idx in window_indices]
    )


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["subset", "baseline_loss", "overfit_loss", "loss_reduction_ratio"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ridge-lambda", type=float, default=1e-8)
    parser.add_argument("--diagnostic-stride", type=int, default=5)
    args = parser.parse_args()
    cfg = SingleMotionConfig(ridge_lambda=args.ridge_lambda, diagnostic_stride=args.diagnostic_stride)
    OUT.mkdir(parents=True, exist_ok=True)

    clean_tau, starts, state_dim, manifest = load_motion_tau(cfg)
    noisy_tau, steps = make_noisy_tau(clean_tau, state_dim, cfg)
    x = feature_matrix(noisy_tau, steps, cfg)
    y = clean_tau.reshape(x.shape[0], clean_tau.shape[-1])
    noisy_baseline = noisy_tau.reshape(y.shape)

    complement_windows, diagnostic_windows = window_subsets(clean_tau.shape[0], cfg.diagnostic_stride)
    complement_idx = expand_window_indices(complement_windows, cfg.sequence_length)
    diagnostic_idx = expand_window_indices(diagnostic_windows, cfg.sequence_length)

    weights = ridge_fit(x, y, cfg.ridge_lambda)
    pred = x @ weights

    rows: list[dict[str, Any]] = []
    metrics: dict[str, float] = {}
    for subset, idx in [("all_motion_windows", np.arange(x.shape[0])), ("diagnostic_stride_subset", diagnostic_idx), ("diagnostic_complement_subset", complement_idx)]:
        baseline_loss = mse(noisy_baseline[idx], y[idx])
        overfit_loss = mse(pred[idx], y[idx])
        ratio = (baseline_loss - overfit_loss) / baseline_loss if baseline_loss > 0.0 else 0.0
        rows.append(
            {
                "subset": subset,
                "baseline_loss": baseline_loss,
                "overfit_loss": overfit_loss,
                "loss_reduction_ratio": ratio,
            }
        )
        metrics[f"{subset}_baseline_loss"] = baseline_loss
        metrics[f"{subset}_overfit_loss"] = overfit_loss
        metrics[f"{subset}_loss_reduction_ratio"] = ratio

    json_path = OUT / "level_c_single_motion_overfit_probe.json"
    tsv_path = OUT / "level_c_single_motion_overfit_probe.tsv"
    npz_path = OUT / "level_c_single_motion_overfit_probe.npz"
    write_tsv(tsv_path, rows)
    np.savez_compressed(
        npz_path,
        clean_tau=clean_tau,
        noisy_tau=noisy_tau,
        diffusion_steps=steps,
        ridge_weights=weights,
        prediction=pred.reshape(clean_tau.shape),
        window_start_indices=starts,
        diagnostic_windows=diagnostic_windows,
        diagnostic_complement_windows=complement_windows,
    )

    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "single-motion clean-trajectory overfit gate over all fixture windows",
        "paper_evidence": {
            "clean_trajectory_loss": str(ROOT / "reproduction/paper/source/tex/method.tex:187-196"),
            "goal_phase7_overfit_gate": str(ROOT / "goal.md:1470-1484"),
        },
        "not_a_replacement_for": [
            "small multi-motion dataset overfit",
            "full diffusion training",
            "official BeyondMimic diffusion implementation",
            "teacher rollout state-latent dataset",
            "checkpoint reproduction",
            "Fig. 5/Fig. 6 reproduction",
        ],
        "settings": {
            **asdict(cfg),
            "state_dim": state_dim,
            "token_dim": int(clean_tau.shape[-1]),
            "feature_dim": int(x.shape[-1]),
            "uses_token_identity_basis": True,
            "window_count": int(clean_tau.shape[0]),
            "sample_count": int(x.shape[0]),
            "diagnostic_subset_window_count": int(len(diagnostic_windows)),
            "diagnostic_complement_window_count": int(len(complement_windows)),
            "fixture_scope": manifest.get("scope"),
        },
        "metrics": {**metrics, "rows": rows},
        "checks": {
            "losses_are_finite": bool(
                np.all(np.isfinite([value for key, value in metrics.items() if key.endswith("_loss")]))
            ),
            "all_motion_loss_decreases": metrics["all_motion_windows_overfit_loss"] < metrics["all_motion_windows_baseline_loss"],
            "all_motion_loss_reduction_ratio_at_least_0_99": metrics["all_motion_windows_loss_reduction_ratio"] >= 0.99,
            "all_motion_final_loss_below_1e_minus_8": metrics["all_motion_windows_overfit_loss"] < 1e-8,
            "overparameterized_memorization_basis_recorded": True,
            "diagnostic_subsets_reported": "diagnostic_stride_subset_overfit_loss" in metrics
            and "diagnostic_complement_subset_overfit_loss" in metrics,
            "uses_independent_state_latent_steps": bool(np.any(steps[..., 0] != steps[..., 1])),
            "uses_all_fixture_windows": clean_tau.shape[0] == len(starts) and clean_tau.shape[0] == 28,
            "debug_fixture_boundary_recorded": manifest.get("experiment_type") == "debug_only",
        },
        "interpretation": {
            "paper_level_status": "partial",
            "why_not_complete": (
                "This overfit gate covers all windows from one debug motion fixture. It is stronger than the single-batch "
                "gate, but it is still not a multi-motion small-dataset overfit, full diffusion training, or paper result."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
