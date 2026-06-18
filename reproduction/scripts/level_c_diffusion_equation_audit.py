#!/usr/bin/env python3
"""Audit the paper's diffusion equations and coefficient-schedule boundary."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
METHOD_TEX = ROOT / "reproduction/paper/source/tex/method.tex"
ROOT_TEX = ROOT / "reproduction/paper/source/root.tex"
OUT = ROOT / "res/level_c/diffusion_equation_audit"


def linear_beta_schedule(steps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> np.ndarray:
    return np.linspace(beta_start, beta_end, steps, dtype=np.float64)


def clean_prediction_reverse_coefficients(betas: np.ndarray) -> dict[str, np.ndarray]:
    """Convert the clean-prediction DDPM posterior mean into the paper's reverse form.

    The paper writes the reverse step as
    alpha_k * (x_k - gamma_k * (x_k - x0_pred)) + sigma_k * noise.
    For a standard DDPM posterior with a clean-data predictor:

      mean = c1 * x0_pred + c2 * x_k

    the equivalent paper-form coefficients are:

      paper_alpha = c1 + c2
      paper_gamma = c1 / (c1 + c2)
      paper_sigma = sqrt(posterior_variance)

    This is an equivalence probe for the published algebraic form. The exact
    BeyondMimic beta/alpha/gamma/sigma schedule is not published in the local
    paper/source artifact set currently available.
    """
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    prev_alpha_bars = np.concatenate([[1.0], alpha_bars[:-1]])
    c1 = betas * np.sqrt(prev_alpha_bars) / (1.0 - alpha_bars)
    c2 = (1.0 - prev_alpha_bars) * np.sqrt(alphas) / (1.0 - alpha_bars)
    paper_alpha = c1 + c2
    paper_gamma = c1 / paper_alpha
    posterior_variance = betas * (1.0 - prev_alpha_bars) / (1.0 - alpha_bars)
    paper_sigma = np.sqrt(np.maximum(posterior_variance, 0.0))
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "posterior_c1": c1,
        "posterior_c2": c2,
        "paper_form_alpha": paper_alpha,
        "paper_form_gamma": paper_gamma,
        "paper_form_sigma": paper_sigma,
    }


def paper_reverse_mean(x_k: np.ndarray, x0_pred: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    return alpha * (x_k - gamma * (x_k - x0_pred))


def posterior_mean(x_k: np.ndarray, x0_pred: np.ndarray, c1: float, c2: float) -> np.ndarray:
    return c1 * x0_pred + c2 * x_k


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["step", "beta", "alpha_bar", "paper_form_alpha", "paper_form_gamma", "paper_form_sigma"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def find_required_text() -> dict[str, bool]:
    method = METHOD_TEX.read_text(encoding="utf-8")
    root = ROOT_TEX.read_text(encoding="utf-8")
    return {
        "forward_posterior_formula_found": "q_\\text{forward}" in method and "\\bar{\\alpha}_k" in method,
        "clean_latent_prediction_loss_found": "z_{\\phi}(\\mathbf{z}^k, k) - \\mathbf{z}^0" in method,
        "clean_trajectory_prediction_loss_found": "z_{\\phi}(\\tau^{\\mathbf{k}},\\, \\mathbf{k}) - \\tau" in method,
        "reverse_alpha_gamma_sigma_form_found": "\\alpha_k\\!\\left" in method
        and "\\gamma_k" in method
        and "\\sigma_k" in method,
        "state_latent_reverse_form_found": "\\alpha_{\\mathbf{k}}" in method
        and "\\gamma_{\\mathbf{k}}" in method
        and "\\sigma_{\\mathbf{k}}" in method,
        "denoising_steps_20_found": "Denoising steps & 20" in root,
        "explicit_beta_schedule_found": any(
            needle in method.lower() or needle in root.lower()
            for needle in ["linear beta", "cosine beta", "variance schedule", "beta schedule", "noise schedule: "]
        ),
        "explicit_alpha_gamma_sigma_numeric_schedule_found": any(
            needle in method or needle in root
            for needle in ["\\alpha_k =", "\\gamma_k =", "\\sigma_k =", "\\alpha_{\\mathbf{k}} =", "\\gamma_{\\mathbf{k}} ="]
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    denoising_steps = 20
    betas = linear_beta_schedule(denoising_steps)
    coeffs = clean_prediction_reverse_coefficients(betas)
    rng = np.random.default_rng(20260901)
    x0 = rng.standard_normal((32, 9))
    eps = rng.standard_normal(x0.shape)

    equivalence_errors: list[float] = []
    forward_mse_rows: list[dict[str, float]] = []
    for idx in range(denoising_steps):
        alpha_bar = coeffs["alpha_bars"][idx]
        x_k = math.sqrt(float(alpha_bar)) * x0 + math.sqrt(float(1.0 - alpha_bar)) * eps
        paper_mean = paper_reverse_mean(
            x_k,
            x0,
            float(coeffs["paper_form_alpha"][idx]),
            float(coeffs["paper_form_gamma"][idx]),
        )
        ddpm_mean = posterior_mean(
            x_k,
            x0,
            float(coeffs["posterior_c1"][idx]),
            float(coeffs["posterior_c2"][idx]),
        )
        equivalence_errors.append(float(np.max(np.abs(paper_mean - ddpm_mean))))
        forward_mse_rows.append(
            {
                "step": float(idx + 1),
                "noisy_vs_clean_mse": float(np.mean((x_k - x0) ** 2)),
                "oracle_clean_prediction_loss": 0.0,
            }
        )

    coefficient_rows = [
        {
            "step": idx + 1,
            "beta": float(coeffs["betas"][idx]),
            "alpha_bar": float(coeffs["alpha_bars"][idx]),
            "paper_form_alpha": float(coeffs["paper_form_alpha"][idx]),
            "paper_form_gamma": float(coeffs["paper_form_gamma"][idx]),
            "paper_form_sigma": float(coeffs["paper_form_sigma"][idx]),
        }
        for idx in range(denoising_steps)
    ]

    npz_path = OUT / "level_c_diffusion_equation_audit.npz"
    json_path = OUT / "level_c_diffusion_equation_audit.json"
    tsv_path = OUT / "level_c_diffusion_equation_audit.tsv"
    np.savez_compressed(
        npz_path,
        **coeffs,
        equivalence_errors=np.asarray(equivalence_errors, dtype=np.float64),
        forward_noisy_mse=np.asarray([row["noisy_vs_clean_mse"] for row in forward_mse_rows], dtype=np.float64),
    )
    write_tsv(tsv_path, coefficient_rows)

    source_findings = find_required_text()
    schedule_missing = (
        not source_findings["explicit_beta_schedule_found"]
        and not source_findings["explicit_alpha_gamma_sigma_numeric_schedule_found"]
    )
    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "formula_audit",
        "scope": "paper diffusion forward/loss/reverse equations plus coefficient-schedule availability boundary",
        "paper_evidence": {
            "ldm_forward_loss_reverse": str(ROOT / "reproduction/paper/source/tex/method.tex:117-132"),
            "state_latent_training_and_reverse": str(ROOT / "reproduction/paper/source/tex/method.tex:171-206"),
            "diffusion_hyperparameter_table": str(ROOT / "reproduction/paper/source/root.tex:827-856"),
            "goal_math_unit_tests": str(ROOT / "goal.md:1728-1737"),
        },
        "not_a_replacement_for": [
            "official BeyondMimic diffusion implementation",
            "exact unpublished coefficient schedule",
            "trained denoising network",
            "long diffusion training",
            "Fig. 5/Fig. 6 reproduction",
        ],
        "source_findings": source_findings,
        "settings": {
            "denoising_steps": denoising_steps,
            "candidate_schedule_used_for_equivalence_probe": "linear_beta_1e-4_to_0.02",
            "candidate_schedule_is_paper_claim": False,
            "seed": 20260901,
        },
        "metrics": {
            "posterior_to_paper_form_max_abs_error": float(np.max(equivalence_errors)),
            "posterior_to_paper_form_mean_abs_error": float(np.mean(equivalence_errors)),
            "forward_noisy_mse_first_step": forward_mse_rows[0]["noisy_vs_clean_mse"],
            "forward_noisy_mse_last_step": forward_mse_rows[-1]["noisy_vs_clean_mse"],
            "paper_form_alpha_min": float(np.min(coeffs["paper_form_alpha"])),
            "paper_form_alpha_max": float(np.max(coeffs["paper_form_alpha"])),
            "paper_form_gamma_min": float(np.min(coeffs["paper_form_gamma"])),
            "paper_form_gamma_max": float(np.max(coeffs["paper_form_gamma"])),
            "paper_form_sigma_min": float(np.min(coeffs["paper_form_sigma"])),
            "paper_form_sigma_max": float(np.max(coeffs["paper_form_sigma"])),
        },
        "coefficient_rows": coefficient_rows,
        "checks": {
            "paper_forward_posterior_formula_found": source_findings["forward_posterior_formula_found"],
            "paper_clean_prediction_loss_found": source_findings["clean_latent_prediction_loss_found"]
            and source_findings["clean_trajectory_prediction_loss_found"],
            "paper_reverse_alpha_gamma_sigma_form_found": source_findings["reverse_alpha_gamma_sigma_form_found"]
            and source_findings["state_latent_reverse_form_found"],
            "paper_denoising_steps_20_found": source_findings["denoising_steps_20_found"],
            "public_source_exact_coefficient_schedule_missing": schedule_missing,
            "candidate_coefficients_finite": bool(
                np.all(np.isfinite(coeffs["paper_form_alpha"]))
                and np.all(np.isfinite(coeffs["paper_form_gamma"]))
                and np.all(np.isfinite(coeffs["paper_form_sigma"]))
            ),
            "candidate_gamma_in_unit_interval": bool(
                np.all(coeffs["paper_form_gamma"] >= 0.0) and np.all(coeffs["paper_form_gamma"] <= 1.0)
            ),
            "candidate_sigma_nonnegative": bool(np.all(coeffs["paper_form_sigma"] >= 0.0)),
            "posterior_to_paper_form_equivalent": bool(np.max(equivalence_errors) < 1e-12),
            "forward_noise_increases_mse_over_probe_steps": bool(
                forward_mse_rows[-1]["noisy_vs_clean_mse"] > forward_mse_rows[0]["noisy_vs_clean_mse"]
            ),
        },
        "interpretation": {
            "paper_level_status": "partial",
            "why_not_complete": (
                "The public paper/source states the forward posterior, clean-prediction loss, reverse alpha/gamma/sigma "
                "form, and K=20 denoising steps, but does not expose the exact numeric beta or alpha/gamma/sigma "
                "coefficient schedule used by the trained BeyondMimic diffusion model."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
