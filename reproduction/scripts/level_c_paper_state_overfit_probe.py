#!/usr/bin/env python3
"""Debug-only clean-trajectory overfit probe using paper-formula state windows."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
STATE_SUMMARY = ROOT / "res/level_c/paper_state_windows/level_c_paper_state_windows.json"
OUT = ROOT / "res/level_c/paper_state_overfit_probe"


@dataclass(frozen=True)
class PaperStateConfig:
    seed: int = 20260910
    history: int = 4
    horizon: int = 16
    latent_dim: int = 32
    denoising_steps: int = 20
    ridge_lambda: float = 1e-8

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


def load_dataset(cfg: PaperStateConfig) -> tuple[np.ndarray, np.ndarray, dict[str, Any], int]:
    rng = np.random.default_rng(cfg.seed)
    summary = json.loads(STATE_SUMMARY.read_text(encoding="utf-8"))
    taus: list[np.ndarray] = []
    motion_ids: list[np.ndarray] = []
    for motion_id, row in enumerate(summary["rows"]):
        data = np.load(row["output_npz"])
        windows = data["paper_state_windows"].astype(np.float64)
        if windows.shape[1] != cfg.sequence_length:
            raise ValueError(f"{row['name']}: sequence length mismatch")
        latents = rng.standard_normal((windows.shape[0], cfg.sequence_length, cfg.latent_dim))
        taus.append(np.concatenate([windows, latents], axis=-1))
        motion_ids.append(np.full(windows.shape[0], motion_id, dtype=np.int64))
    state_dim = int(summary["settings"]["paper_state_dim"])
    return np.concatenate(taus, axis=0), np.concatenate(motion_ids, axis=0), summary, state_dim


def make_noisy_tau(clean_tau: np.ndarray, state_dim: int, cfg: PaperStateConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed + 1)
    steps = rng.integers(0, cfg.denoising_steps, size=clean_tau.shape[:2] + (2,), dtype=np.int64)
    noise = rng.standard_normal(clean_tau.shape)
    bars = alpha_bars(cfg.denoising_steps)
    state_alpha = np.repeat(bars[steps[..., 0]][..., None], state_dim, axis=-1)
    latent_alpha = np.repeat(bars[steps[..., 1]][..., None], cfg.latent_dim, axis=-1)
    alpha = np.concatenate([state_alpha, latent_alpha], axis=-1)
    return np.sqrt(alpha) * clean_tau + np.sqrt(1.0 - alpha) * noise, steps


def feature_matrix(noisy_tau: np.ndarray, steps: np.ndarray, motion_ids: np.ndarray, cfg: PaperStateConfig) -> np.ndarray:
    b, t, d = noisy_tau.shape
    sample_count = b * t
    motion_per_token = np.repeat(motion_ids, t)
    return np.concatenate(
        [
            noisy_tau.reshape(sample_count, d),
            one_hot(steps[..., 0], cfg.denoising_steps).reshape(sample_count, cfg.denoising_steps),
            one_hot(steps[..., 1], cfg.denoising_steps).reshape(sample_count, cfg.denoising_steps),
            one_hot(motion_per_token, int(motion_ids.max()) + 1),
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
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--ridge-lambda", type=float, default=1e-8)
    args = parser.parse_args()

    cfg = replace(PaperStateConfig(seed=args.seed, ridge_lambda=args.ridge_lambda))
    OUT.mkdir(parents=True, exist_ok=True)
    clean_tau, motion_ids, state_summary, state_dim = load_dataset(cfg)
    noisy_tau, steps = make_noisy_tau(clean_tau, state_dim, cfg)
    x = feature_matrix(noisy_tau, steps, motion_ids, cfg)
    y = clean_tau.reshape(x.shape[0], clean_tau.shape[-1])
    baseline = noisy_tau.reshape(y.shape)
    weights = ridge_fit(x, y, cfg.ridge_lambda)
    pred = x @ weights

    rows: list[dict[str, Any]] = []
    metrics: dict[str, float] = {}
    subsets: list[tuple[str, np.ndarray]] = [("all_paper_state_windows", np.arange(clean_tau.shape[0]))]
    for motion_id, row in enumerate(state_summary["rows"]):
        subsets.append((f"motion:{row['name']}", np.nonzero(motion_ids == motion_id)[0]))
    for subset, window_idx in subsets:
        token_idx = expand_window_indices(window_idx, cfg.sequence_length)
        baseline_loss = mse(baseline[token_idx], y[token_idx])
        overfit_loss = mse(pred[token_idx], y[token_idx])
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

    json_path = OUT / "level_c_paper_state_overfit_probe.json"
    tsv_path = OUT / "level_c_paper_state_overfit_probe.tsv"
    npz_path = OUT / "level_c_paper_state_overfit_probe.npz"
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

    all_key = "all_paper_state_windows"
    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "clean-trajectory overfit gate using paper-formula 99-D state windows plus synthetic latents",
        "paper_evidence": {
            "paper_state_windows": str(STATE_SUMMARY),
            "clean_trajectory_loss": str(ROOT / "reproduction/paper/source/tex/method.tex:187-196"),
            "diffusion_state_formula": str(ROOT / "reproduction/paper/source/root.tex:482-532"),
        },
        "not_a_replacement_for": [
            "full diffusion training",
            "teacher rollout state-latent dataset",
            "trained VAE latent trajectories",
            "paper evaluation",
        ],
        "settings": {
            **asdict(cfg),
            "paper_state_dim": state_dim,
            "token_dim": int(clean_tau.shape[-1]),
            "motion_count": int(len(state_summary["rows"])),
            "window_count": int(clean_tau.shape[0]),
            "sample_count": int(x.shape[0]),
            "feature_dim": int(x.shape[1]),
            "uses_overparameterized_token_identity_basis": True,
        },
        "metrics": {**metrics, "rows": rows},
        "checks": {
            "paper_state_windows_status_ok": state_summary["status"] == "ok",
            "paper_state_windows_all_checks_pass": state_summary["checks"]["all_checks_pass"],
            "uses_paper_state_dim_99": state_dim == 99,
            "uses_multiple_motions": len(state_summary["rows"]) >= 3,
            "uses_independent_state_latent_steps": bool(np.any(steps[..., 0] != steps[..., 1])),
            "loss_decreases": metrics[f"{all_key}_overfit_loss"] < metrics[f"{all_key}_baseline_loss"],
            "loss_reduction_ratio_at_least_0_99": metrics[f"{all_key}_loss_reduction_ratio"] >= 0.99,
            "final_loss_below_1e_minus_8": metrics[f"{all_key}_overfit_loss"] < 1e-8,
            "overparameterized_memorization_basis_recorded": True,
            "debug_fixture_boundary_recorded": state_summary["experiment_type"] == "debug_only",
        },
        "interpretation": {
            "paper_level_status": "partial",
            "why_not_complete": (
                "This confirms the local denoising loss path can memorize paper-formula state windows plus synthetic "
                "latents. It is still a debug memorization gate with token identity features, not full diffusion "
                "training, held-out evaluation, or a teacher/VAE rollout dataset."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
