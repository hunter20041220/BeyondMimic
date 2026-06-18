#!/usr/bin/env python3
"""Debug-only reverse denoising probe for Level C diffusion mechanics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DEFAULT_TIMESTEP_NPZ = ROOT / "res/level_c/timestep_mask_probe/level_c_timestep_mask_probe.npz"
DEFAULT_TIMESTEP_JSON = ROOT / "res/level_c/timestep_mask_probe/level_c_timestep_mask_probe.json"
OUT = ROOT / "res/level_c/reverse_denoising_probe"


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


def split_alpha(alpha_bars: np.ndarray, steps: np.ndarray, state_dim: int, latent_dim: int) -> np.ndarray:
    state_alpha = alpha_bars[steps[:, 0]][:, None]
    latent_alpha = alpha_bars[steps[:, 1]][:, None]
    return np.concatenate(
        [
            np.repeat(state_alpha, state_dim, axis=1),
            np.repeat(latent_alpha, latent_dim, axis=1),
        ],
        axis=1,
    )


def decrement_steps(steps: np.ndarray) -> np.ndarray:
    return np.maximum(steps - 1, 0)


def oracle_reverse_step(
    tau_k: np.ndarray,
    tau_clean: np.ndarray,
    steps: np.ndarray,
    alpha_bars: np.ndarray,
    state_dim: int,
    latent_dim: int,
    observed_mask: np.ndarray,
    stochastic_sigma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Candidate deterministic DDIM-style step with optional tiny noise.

    This is a mechanics probe. The paper writes alpha/gamma/sigma coefficients
    but does not provide their complete schedule in the released source.
    """
    rng = np.random.default_rng(seed)
    next_steps = decrement_steps(steps)
    alpha_next = split_alpha(alpha_bars, next_steps, state_dim, latent_dim)
    current_alpha = split_alpha(alpha_bars, steps, state_dim, latent_dim)
    tau_next = np.sqrt(alpha_next) * tau_clean
    residual = tau_k - np.sqrt(current_alpha) * tau_clean
    tau_next += np.sqrt(np.maximum(1.0 - alpha_next, 0.0)) * residual
    if stochastic_sigma > 0.0:
        tau_next += stochastic_sigma * rng.standard_normal(tau_next.shape)

    full_mask = np.concatenate(
        [
            np.repeat(observed_mask[:, 0:1], state_dim, axis=1),
            np.repeat(observed_mask[:, 1:2], latent_dim, axis=1),
        ],
        axis=1,
    )
    tau_next = np.where(full_mask, tau_clean, tau_next)
    return tau_next, next_steps


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestep-npz", type=Path, default=DEFAULT_TIMESTEP_NPZ)
    parser.add_argument("--timestep-json", type=Path, default=DEFAULT_TIMESTEP_JSON)
    parser.add_argument("--schedule", choices=["history_conditioning", "future_keyframe_inpainting"], default="future_keyframe_inpainting")
    parser.add_argument("--stochastic-sigma", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260828)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    data = np.load(args.timestep_npz)
    timestep_summary = json.loads(args.timestep_json.read_text(encoding="utf-8"))
    tau_clean = data["tau_clean"]
    alpha_bars = data["alpha_bars"]
    steps = data[f"{args.schedule}_steps"].astype(np.int64)
    observed_mask = data[f"{args.schedule}_observed_mask"].astype(bool)
    tau_start = data[f"{args.schedule}_noised_tau"]
    state_dim = int(timestep_summary["settings"]["state_dim"])
    latent_dim = int(timestep_summary["settings"]["latent_dim"])

    tau_next, next_steps = oracle_reverse_step(
        tau_start,
        tau_clean,
        steps,
        alpha_bars,
        state_dim,
        latent_dim,
        observed_mask,
        args.stochastic_sigma,
        args.seed,
    )
    tau_final = tau_start.copy()
    iter_steps = steps.copy()
    trajectory_mse = [mse(tau_final, tau_clean)]
    step_trace = [int(np.max(iter_steps))]
    for i in range(int(np.max(steps))):
        tau_final, iter_steps = oracle_reverse_step(
            tau_final,
            tau_clean,
            iter_steps,
            alpha_bars,
            state_dim,
            latent_dim,
            observed_mask,
            args.stochastic_sigma,
            args.seed + i + 1,
        )
        trajectory_mse.append(mse(tau_final, tau_clean))
        step_trace.append(int(np.max(iter_steps)))

    full_mask = np.concatenate(
        [
            np.repeat(observed_mask[:, 0:1], state_dim, axis=1),
            np.repeat(observed_mask[:, 1:2], latent_dim, axis=1),
        ],
        axis=1,
    )
    observed_clamp_error = float(np.max(np.abs(tau_next[full_mask] - tau_clean[full_mask]))) if np.any(full_mask) else 0.0

    npz_path = OUT / "level_c_reverse_denoising_probe.npz"
    json_path = OUT / "level_c_reverse_denoising_probe.json"
    tsv_path = OUT / "level_c_reverse_denoising_probe.tsv"
    np.savez_compressed(
        npz_path,
        tau_clean=tau_clean,
        tau_start=tau_start,
        tau_after_one_step=tau_next,
        tau_after_full_oracle_reverse=tau_final,
        initial_steps=steps,
        one_step_next_steps=next_steps,
        final_steps=iter_steps,
        trajectory_mse=np.asarray(trajectory_mse, dtype=np.float64),
        max_step_trace=np.asarray(step_trace, dtype=np.int64),
    )

    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "oracle clean-predictor reverse denoising mechanics probe",
        "paper_evidence": {
            "reverse_formula": str(ROOT / "reproduction/paper/source/tex/method.tex:197-206"),
            "individual_denoising_steps": str(ROOT / "reproduction/paper/source/tex/method.tex:171-185"),
        },
        "not_a_replacement_for": [
            "paper-exact alpha/gamma/sigma schedule",
            "trained denoising network",
            "TensorRT deployment",
            "guided rollout evaluation",
        ],
        "settings": {
            "schedule": args.schedule,
            "state_dim": state_dim,
            "latent_dim": latent_dim,
            "tau_shape": list(tau_clean.shape),
            "stochastic_sigma": args.stochastic_sigma,
            "seed": args.seed,
        },
        "metrics": {
            "initial_mse": trajectory_mse[0],
            "one_step_mse": mse(tau_next, tau_clean),
            "final_mse": trajectory_mse[-1],
            "observed_clamp_max_abs_error": observed_clamp_error,
            "initial_max_step": int(np.max(steps)),
            "one_step_max_step": int(np.max(next_steps)),
            "final_max_step": int(np.max(iter_steps)),
            "mse_trace": trajectory_mse,
            "max_step_trace": step_trace,
        },
        "checks": {
            "one_step_reduces_mse": bool(mse(tau_next, tau_clean) < trajectory_mse[0]),
            "full_reverse_reduces_mse": bool(trajectory_mse[-1] < trajectory_mse[0]),
            "all_steps_reach_zero": bool(np.max(iter_steps) == 0),
            "observed_tokens_clamped": bool(observed_clamp_error < 1e-12),
            "step_trace_monotone_nonincreasing": bool(np.all(np.diff(np.asarray(step_trace)) <= 0)),
        },
        "outputs": {"npz": str(npz_path), "json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, summary)
    print(json.dumps({"status": "ok", "json": str(json_path), "npz": str(npz_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
