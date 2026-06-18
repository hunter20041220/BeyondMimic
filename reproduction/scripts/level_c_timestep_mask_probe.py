#!/usr/bin/env python3
"""Debug-only probe for independent diffusion timesteps and task masks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DEFAULT_FIXTURE = ROOT / "reproduction/data/level_c_fixtures/walk1_subject1_frames_1_180_state_fixture.npz"
DEFAULT_MANIFEST = ROOT / "res/level_c/motion_state_fixture/walk1_subject1_frames_1_180_state_fixture.json"
OUT = ROOT / "res/level_c/timestep_mask_probe"


def diffusion_alpha_bars(steps: int) -> np.ndarray:
    betas = np.linspace(1e-4, 0.02, steps, dtype=np.float64)
    alphas = 1.0 - betas
    return np.cumprod(alphas)


def build_schedules(length: int, history: int, horizon: int, denoising_steps: int, seed: int) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    max_step = denoising_steps - 1
    current = history
    keyframe_indices = [min(current + 5, length - 1), min(current + 10, length - 1), min(current + 15, length - 1)]

    training = rng.integers(0, denoising_steps, size=(length, 2), dtype=np.int64)

    history_conditioning = np.full((length, 2), max_step, dtype=np.int64)
    history_conditioning[: current + 1, 0] = 0
    history_conditioning[: current + 1, 1] = 0

    keyframe_inpainting = np.full((length, 2), max_step, dtype=np.int64)
    keyframe_inpainting[: current + 1, 0] = 0
    keyframe_inpainting[: current + 1, 1] = 0
    for idx in keyframe_indices:
        keyframe_inpainting[idx, 0] = 0

    return {
        "training_uniform_random": {
            "steps": training,
            "observed_mask": training == 0,
            "description": "self-supervised training schedule with independent random state/latent timesteps",
        },
        "history_conditioning": {
            "steps": history_conditioning,
            "observed_mask": history_conditioning == 0,
            "description": "current and history state/latent tokens clamped clean; future remains noised",
        },
        "future_keyframe_inpainting": {
            "steps": keyframe_inpainting,
            "observed_mask": keyframe_inpainting == 0,
            "description": "history/current clean plus sparse future state keyframes clean; future latents remain generated",
        },
    }


def apply_tokenwise_noise(x0: np.ndarray, state_dim: int, latent_dim: int, steps: np.ndarray, alpha_bars: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(x0.shape)
    state_alpha = alpha_bars[steps[:, 0]][:, None]
    latent_alpha = alpha_bars[steps[:, 1]][:, None]
    alpha = np.concatenate(
        [
            np.repeat(state_alpha, state_dim, axis=1),
            np.repeat(latent_alpha, latent_dim, axis=1),
        ],
        axis=1,
    )
    return np.sqrt(alpha) * x0 + np.sqrt(1.0 - alpha) * noise


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
    parser.add_argument("--fixture-npz", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--manifest-json", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--denoising-steps", type=int, default=20)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    fixture = np.load(args.fixture_npz)
    manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))
    windows = fixture["candidate_hybrid_state_windows"]
    state = windows[0]
    history = int(manifest["history"])
    horizon = int(manifest["horizon"])
    length = history + 1 + horizon
    if state.shape[0] != length:
        raise ValueError(f"window length mismatch: state has {state.shape[0]}, expected {length}")

    rng = np.random.default_rng(args.seed)
    latents = rng.standard_normal((length, args.latent_dim))
    tau = np.concatenate([state, latents], axis=1)
    state_dim = state.shape[1]
    schedules = build_schedules(length, history, horizon, args.denoising_steps, args.seed)
    alpha_bars = diffusion_alpha_bars(args.denoising_steps)

    npz_payload: dict[str, np.ndarray] = {"tau_clean": tau, "alpha_bars": alpha_bars}
    schedule_summaries: dict[str, Any] = {}
    for name, item in schedules.items():
        steps = item["steps"]
        noised = apply_tokenwise_noise(tau, state_dim, args.latent_dim, steps, alpha_bars, args.seed + len(name))
        npz_payload[f"{name}_steps"] = steps
        npz_payload[f"{name}_observed_mask"] = item["observed_mask"]
        npz_payload[f"{name}_noised_tau"] = noised
        state_steps = steps[:, 0]
        latent_steps = steps[:, 1]
        clean_state = int(np.sum(state_steps == 0))
        clean_latent = int(np.sum(latent_steps == 0))
        independent_pairs = int(np.sum(state_steps != latent_steps))
        schedule_summaries[name] = {
            "description": item["description"],
            "shape": list(steps.shape),
            "state_step_minmax": [int(state_steps.min()), int(state_steps.max())],
            "latent_step_minmax": [int(latent_steps.min()), int(latent_steps.max())],
            "clean_state_tokens": clean_state,
            "clean_latent_tokens": clean_latent,
            "state_latent_step_mismatch_tokens": independent_pairs,
            "observed_mask_true_count": int(np.sum(item["observed_mask"])),
            "mean_abs_delta_from_clean": float(np.mean(np.abs(noised - tau))),
            "max_abs_delta_for_clean_tokens": float(np.max(np.abs(noised[item["observed_mask"].all(axis=1)] - tau[item["observed_mask"].all(axis=1)])))
            if np.any(item["observed_mask"].all(axis=1))
            else None,
        }

    npz_path = OUT / "level_c_timestep_mask_probe.npz"
    json_path = OUT / "level_c_timestep_mask_probe.json"
    tsv_path = OUT / "level_c_timestep_mask_probe.tsv"
    np.savez_compressed(npz_path, **npz_payload)

    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "independent state/latent diffusion timestep and task-mask probe",
        "paper_evidence": {
            "individual_denoising_steps": str(ROOT / "reproduction/paper/source/tex/method.tex:171-185"),
            "clean_trajectory_loss": str(ROOT / "reproduction/paper/source/tex/method.tex:187-196"),
            "future_keyframe_inpainting_figure": str(ROOT / "reproduction/paper/source/root.tex:237-243"),
            "goal_independent_timestep": str(ROOT / "goal.md:1249-1287"),
        },
        "not_a_replacement_for": [
            "paper-exact inpainting mask policy",
            "reverse denoising implementation",
            "guided rollout evaluation",
            "Fig. 6 reproduction",
        ],
        "settings": {
            "history": history,
            "horizon": horizon,
            "sequence_length": length,
            "denoising_steps": args.denoising_steps,
            "state_dim": state_dim,
            "latent_dim": args.latent_dim,
            "tau_dim": int(tau.shape[1]),
            "seed": args.seed,
        },
        "schedules": schedule_summaries,
        "checks": {
            "all_step_tensors_shape_21x2": bool(all(item["steps"].shape == (length, 2) for item in schedules.values())),
            "training_has_independent_state_latent_steps": bool(schedule_summaries["training_uniform_random"]["state_latent_step_mismatch_tokens"] > 0),
            "history_conditioning_clean_prefix": bool(
                np.all(schedules["history_conditioning"]["steps"][: history + 1] == 0)
                and np.all(schedules["history_conditioning"]["steps"][history + 1 :] == args.denoising_steps - 1)
            ),
            "keyframe_inpainting_has_future_clean_state_only": bool(
                schedule_summaries["future_keyframe_inpainting"]["state_latent_step_mismatch_tokens"] > 0
                and schedule_summaries["future_keyframe_inpainting"]["clean_state_tokens"]
                > schedule_summaries["future_keyframe_inpainting"]["clean_latent_tokens"]
            ),
            "noised_tau_shapes_match_clean": bool(
                all(npz_payload[f"{name}_noised_tau"].shape == tau.shape for name in schedules)
            ),
        },
        "outputs": {"npz": str(npz_path), "json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, summary)
    print(json.dumps({"status": "ok", "json": str(json_path), "npz": str(npz_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
