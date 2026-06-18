#!/usr/bin/env python3
"""Debug-only timestep/mask and oracle reverse probe on 99-D paper-state windows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
PAPER_STATE_JSON = ROOT / "res/level_c/paper_state_windows/level_c_paper_state_windows.json"
PAPER_STATE_NPZ = ROOT / "res/level_c/paper_state_windows/level_c_paper_state_windows_summary.npz"
OUT = ROOT / "res/level_c/paper_state_mask_reverse_probe"


def diffusion_alpha_bars(steps: int) -> np.ndarray:
    betas = np.linspace(1e-4, 0.02, steps, dtype=np.float64)
    return np.cumprod(1.0 - betas)


def build_schedules(length: int, history: int, denoising_steps: int, seed: int) -> dict[str, dict[str, np.ndarray | str]]:
    rng = np.random.default_rng(seed)
    max_step = denoising_steps - 1
    current = history
    keyframe_indices = [min(current + 5, length - 1), min(current + 10, length - 1), min(current + 15, length - 1)]

    training = rng.integers(0, denoising_steps, size=(length, 2), dtype=np.int64)

    history_conditioning = np.full((length, 2), max_step, dtype=np.int64)
    history_conditioning[: current + 1] = 0

    future_keyframe_inpainting = np.full((length, 2), max_step, dtype=np.int64)
    future_keyframe_inpainting[: current + 1] = 0
    for idx in keyframe_indices:
        future_keyframe_inpainting[idx, 0] = 0

    return {
        "training_uniform_random": {
            "steps": training,
            "description": "paper-state training schedule with independent random state/latent timesteps",
        },
        "history_conditioning": {
            "steps": history_conditioning,
            "description": "paper-state history/current state and latent tokens clamped clean",
        },
        "future_keyframe_inpainting": {
            "steps": future_keyframe_inpainting,
            "description": "paper-state history/current clean plus sparse future state keyframes clean",
        },
    }


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


def full_mask(observed_mask: np.ndarray, state_dim: int, latent_dim: int) -> np.ndarray:
    return np.concatenate(
        [
            np.repeat(observed_mask[:, 0:1], state_dim, axis=1),
            np.repeat(observed_mask[:, 1:2], latent_dim, axis=1),
        ],
        axis=1,
    )


def apply_tokenwise_noise(
    tau_clean: np.ndarray,
    state_dim: int,
    latent_dim: int,
    steps: np.ndarray,
    alpha_bars: np.ndarray,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    alpha = split_alpha(alpha_bars, steps, state_dim, latent_dim)
    tau_noised = np.sqrt(alpha) * tau_clean + np.sqrt(1.0 - alpha) * rng.standard_normal(tau_clean.shape)
    mask = full_mask(steps == 0, state_dim, latent_dim)
    return np.where(mask, tau_clean, tau_noised)


def oracle_reverse_step(
    tau_k: np.ndarray,
    tau_clean: np.ndarray,
    steps: np.ndarray,
    alpha_bars: np.ndarray,
    state_dim: int,
    latent_dim: int,
    observed_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    next_steps = np.maximum(steps - 1, 0)
    alpha_next = split_alpha(alpha_bars, next_steps, state_dim, latent_dim)
    alpha_current = split_alpha(alpha_bars, steps, state_dim, latent_dim)
    residual = tau_k - np.sqrt(alpha_current) * tau_clean
    tau_next = np.sqrt(alpha_next) * tau_clean + np.sqrt(np.maximum(1.0 - alpha_next, 0.0)) * residual
    return np.where(full_mask(observed_mask, state_dim, latent_dim), tau_clean, tau_next), next_steps


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-state-json", type=Path, default=PAPER_STATE_JSON)
    parser.add_argument("--paper-state-npz", type=Path, default=PAPER_STATE_NPZ)
    parser.add_argument("--motion-key", default="walk1_subject1_frames_1_180_state_fixture_paper_state_windows")
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument("--denoising-steps", type=int, default=20)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    paper_state_summary = json.loads(args.paper_state_json.read_text(encoding="utf-8"))
    paper_state_data = np.load(args.paper_state_npz)
    if args.motion_key not in paper_state_data.files:
        raise KeyError(f"{args.motion_key} not found in {args.paper_state_npz}; keys={paper_state_data.files}")

    motion_windows = paper_state_data[args.motion_key]
    state = motion_windows[args.window_index].astype(np.float64)
    history = int(paper_state_summary["settings"]["history"])
    horizon = int(paper_state_summary["settings"]["horizon"])
    length = int(paper_state_summary["settings"]["sequence_length"])
    state_dim = int(paper_state_summary["settings"]["paper_state_dim"])
    if state.shape != (length, state_dim):
        raise ValueError(f"paper-state window shape {state.shape} does not match {(length, state_dim)}")

    rng = np.random.default_rng(args.seed)
    latents = rng.standard_normal((length, args.latent_dim))
    tau_clean = np.concatenate([state, latents], axis=1)
    alpha_bars = diffusion_alpha_bars(args.denoising_steps)
    schedules = build_schedules(length, history, args.denoising_steps, args.seed)

    npz_payload: dict[str, np.ndarray] = {
        "paper_state_clean": state,
        "synthetic_latents": latents,
        "tau_clean": tau_clean,
        "alpha_bars": alpha_bars,
    }
    schedule_summaries: dict[str, Any] = {}
    for name, item in schedules.items():
        steps = item["steps"]
        if not isinstance(steps, np.ndarray):
            raise TypeError(f"{name}: steps missing")
        observed_mask = steps == 0
        noised = apply_tokenwise_noise(tau_clean, state_dim, args.latent_dim, steps, alpha_bars, args.seed + len(name))
        state_steps = steps[:, 0]
        latent_steps = steps[:, 1]
        npz_payload[f"{name}_steps"] = steps
        npz_payload[f"{name}_observed_mask"] = observed_mask
        npz_payload[f"{name}_noised_tau"] = noised
        schedule_summaries[name] = {
            "description": item["description"],
            "shape": list(steps.shape),
            "state_step_minmax": [int(state_steps.min()), int(state_steps.max())],
            "latent_step_minmax": [int(latent_steps.min()), int(latent_steps.max())],
            "clean_state_tokens": int(np.sum(state_steps == 0)),
            "clean_latent_tokens": int(np.sum(latent_steps == 0)),
            "state_latent_step_mismatch_tokens": int(np.sum(state_steps != latent_steps)),
            "observed_mask_true_count": int(np.sum(observed_mask)),
            "mean_abs_delta_from_clean": float(np.mean(np.abs(noised - tau_clean))),
        }

    schedule_name = "future_keyframe_inpainting"
    steps = npz_payload[f"{schedule_name}_steps"].astype(np.int64)
    observed_mask = npz_payload[f"{schedule_name}_observed_mask"].astype(bool)
    tau_start = npz_payload[f"{schedule_name}_noised_tau"]
    tau_final = tau_start.copy()
    iter_steps = steps.copy()
    mse_trace = [mse(tau_final, tau_clean)]
    max_step_trace = [int(np.max(iter_steps))]
    for _ in range(int(np.max(steps))):
        tau_final, iter_steps = oracle_reverse_step(
            tau_final,
            tau_clean,
            iter_steps,
            alpha_bars,
            state_dim,
            args.latent_dim,
            observed_mask,
        )
        mse_trace.append(mse(tau_final, tau_clean))
        max_step_trace.append(int(np.max(iter_steps)))

    one_step_tau, one_step_steps = oracle_reverse_step(
        tau_start,
        tau_clean,
        steps,
        alpha_bars,
        state_dim,
        args.latent_dim,
        observed_mask,
    )
    mask = full_mask(observed_mask, state_dim, args.latent_dim)
    observed_clamp_error = float(np.max(np.abs(tau_final[mask] - tau_clean[mask]))) if np.any(mask) else 0.0

    npz_path = OUT / "level_c_paper_state_mask_reverse_probe.npz"
    json_path = OUT / "level_c_paper_state_mask_reverse_probe.json"
    tsv_path = OUT / "level_c_paper_state_mask_reverse_probe.tsv"
    np.savez_compressed(
        npz_path,
        **npz_payload,
        reverse_tau_start=tau_start,
        reverse_tau_after_one_step=one_step_tau,
        reverse_tau_final=tau_final,
        reverse_initial_steps=steps,
        reverse_one_step_steps=one_step_steps,
        reverse_final_steps=iter_steps,
        reverse_mse_trace=np.asarray(mse_trace, dtype=np.float64),
        reverse_max_step_trace=np.asarray(max_step_trace, dtype=np.int64),
    )

    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "paper-formula 99-D state-window timestep/mask and oracle reverse mechanics probe",
        "paper_evidence": {
            "paper_state_windows": str(ROOT / "res/level_c/paper_state_windows/level_c_paper_state_windows.json"),
            "individual_denoising_steps": str(ROOT / "reproduction/paper/source/tex/method.tex:171-185"),
            "clean_trajectory_loss": str(ROOT / "reproduction/paper/source/tex/method.tex:187-196"),
            "reverse_formula": str(ROOT / "reproduction/paper/source/tex/method.tex:197-206"),
        },
        "not_a_replacement_for": [
            "trained denoising network",
            "paper-exact mask policy",
            "paper-exact alpha/gamma/sigma coefficient schedule",
            "VAE latents from a trained encoder",
            "TensorRT deployment",
            "Fig. 6 rollout evaluation",
        ],
        "settings": {
            "motion_key": args.motion_key,
            "window_index": args.window_index,
            "history": history,
            "horizon": horizon,
            "sequence_length": length,
            "denoising_steps": args.denoising_steps,
            "paper_state_dim": state_dim,
            "latent_dim": args.latent_dim,
            "tau_dim": int(tau_clean.shape[1]),
            "seed": args.seed,
            "reverse_schedule": schedule_name,
        },
        "schedules": schedule_summaries,
        "metrics": {
            "reverse_initial_mse": mse_trace[0],
            "reverse_one_step_mse": mse(one_step_tau, tau_clean),
            "reverse_final_mse": mse_trace[-1],
            "reverse_initial_max_step": int(np.max(steps)),
            "reverse_one_step_max_step": int(np.max(one_step_steps)),
            "reverse_final_max_step": int(np.max(iter_steps)),
            "reverse_observed_clamp_max_abs_error": observed_clamp_error,
            "reverse_mse_trace": mse_trace,
            "reverse_max_step_trace": max_step_trace,
        },
        "checks": {
            "paper_state_source_status_ok": paper_state_summary["status"] == "ok",
            "paper_state_dim_99": state_dim == 99,
            "tau_dim_131": tau_clean.shape[1] == 131,
            "all_step_tensors_shape_21x2": all(
                npz_payload[f"{name}_steps"].shape == (length, 2) for name in schedules
            ),
            "training_has_independent_state_latent_steps": (
                schedule_summaries["training_uniform_random"]["state_latent_step_mismatch_tokens"] > 0
            ),
            "history_conditioning_clean_prefix": bool(
                np.all(npz_payload["history_conditioning_steps"][: history + 1] == 0)
                and np.all(npz_payload["history_conditioning_steps"][history + 1 :] == args.denoising_steps - 1)
            ),
            "keyframe_inpainting_has_future_clean_state_only": (
                schedule_summaries["future_keyframe_inpainting"]["clean_state_tokens"]
                > schedule_summaries["future_keyframe_inpainting"]["clean_latent_tokens"]
                and schedule_summaries["future_keyframe_inpainting"]["state_latent_step_mismatch_tokens"] > 0
            ),
            "noised_tau_shapes_match_clean": all(
                npz_payload[f"{name}_noised_tau"].shape == tau_clean.shape for name in schedules
            ),
            "reverse_one_step_reduces_mse": mse(one_step_tau, tau_clean) < mse_trace[0],
            "reverse_full_reduces_mse": mse_trace[-1] < mse_trace[0],
            "reverse_steps_reach_zero": int(np.max(iter_steps)) == 0,
            "reverse_observed_tokens_clamped": observed_clamp_error < 1e-12,
            "reverse_step_trace_monotone_nonincreasing": bool(np.all(np.diff(np.asarray(max_step_trace)) <= 0)),
        },
        "interpretation": {
            "paper_level_status": "partial",
            "goal_complete": False,
            "why_not_complete": (
                "This closes the local artifact gap for applying timestep/mask and oracle reverse mechanics to the "
                "99-D paper-formula state windows, but it remains a debug-only probe with synthetic latents and an "
                "oracle clean predictor. It does not establish the unpublished deployed mask policy, trained diffusion "
                "network, exact coefficient schedule, TensorRT runtime, or Fig.6 rollout behavior."
            ),
        },
        "outputs": {"npz": str(npz_path), "json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, summary)
    print(json.dumps({"status": "ok", "json": str(json_path), "npz": str(npz_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
