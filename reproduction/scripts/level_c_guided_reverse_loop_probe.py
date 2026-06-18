#!/usr/bin/env python3
"""Debug-only guided reverse-loop probe for Level C diffusion mechanics.

This connects the earlier reverse denoising mechanics probe with differentiable
guidance costs. It uses an oracle clean predictor and a candidate gradient step,
so it validates wiring and invariants rather than a faithful trained controller.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DEFAULT_TIMESTEP_NPZ = ROOT / "res/level_c/timestep_mask_probe/level_c_timestep_mask_probe.npz"
DEFAULT_TIMESTEP_JSON = ROOT / "res/level_c/timestep_mask_probe/level_c_timestep_mask_probe.json"
OUT = ROOT / "res/level_c/guided_reverse_loop_probe"


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
) -> tuple[np.ndarray, np.ndarray]:
    next_steps = decrement_steps(steps)
    alpha_next = split_alpha(alpha_bars, next_steps, state_dim, latent_dim)
    current_alpha = split_alpha(alpha_bars, steps, state_dim, latent_dim)
    tau_next = np.sqrt(alpha_next) * tau_clean
    residual = tau_k - np.sqrt(current_alpha) * tau_clean
    tau_next += np.sqrt(np.maximum(1.0 - alpha_next, 0.0)) * residual
    return clamp_observed(tau_next, tau_clean, observed_mask, state_dim, latent_dim), next_steps


def full_observed_mask(observed_mask: np.ndarray, state_dim: int, latent_dim: int) -> np.ndarray:
    return np.concatenate(
        [
            np.repeat(observed_mask[:, 0:1], state_dim, axis=1),
            np.repeat(observed_mask[:, 1:2], latent_dim, axis=1),
        ],
        axis=1,
    )


def clamp_observed(
    tau: np.ndarray,
    tau_clean: np.ndarray,
    observed_mask: np.ndarray,
    state_dim: int,
    latent_dim: int,
) -> np.ndarray:
    mask = full_observed_mask(observed_mask, state_dim, latent_dim)
    return np.where(mask, tau_clean, tau)


def joystick_guidance_cost(
    tau: torch.Tensor,
    *,
    history: int,
    root_lin_vel_slice: slice,
    command_velocity: torch.Tensor,
) -> torch.Tensor:
    future = tau[history:]
    root_vel_xy = future[:, root_lin_vel_slice][:, :2]
    return 0.5 * torch.sum((root_vel_xy - command_velocity) ** 2)


def apply_guidance(
    tau: np.ndarray,
    tau_clean: np.ndarray,
    observed_mask: np.ndarray,
    state_dim: int,
    latent_dim: int,
    history: int,
    root_lin_vel_slice: slice,
    command_velocity_xy: list[float],
    guidance_scale: float,
) -> tuple[np.ndarray, float, float]:
    tau_t = torch.tensor(tau, dtype=torch.float64, requires_grad=True)
    command = torch.tensor(command_velocity_xy, dtype=torch.float64)
    cost = joystick_guidance_cost(
        tau_t,
        history=history,
        root_lin_vel_slice=root_lin_vel_slice,
        command_velocity=command,
    )
    cost.backward()
    grad = tau_t.grad.detach().cpu().numpy()
    guided = tau - guidance_scale * grad
    guided = clamp_observed(guided, tau_clean, observed_mask, state_dim, latent_dim)
    return guided, float(cost.detach().cpu()), float(np.linalg.norm(grad))


def eval_guidance_cost(
    tau: np.ndarray,
    history: int,
    root_lin_vel_slice: slice,
    command_velocity_xy: list[float],
) -> float:
    tau_t = torch.tensor(tau, dtype=torch.float64)
    command = torch.tensor(command_velocity_xy, dtype=torch.float64)
    return float(
        joystick_guidance_cost(
            tau_t,
            history=history,
            root_lin_vel_slice=root_lin_vel_slice,
            command_velocity=command,
        )
        .detach()
        .cpu()
    )


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestep-npz", type=Path, default=DEFAULT_TIMESTEP_NPZ)
    parser.add_argument("--timestep-json", type=Path, default=DEFAULT_TIMESTEP_JSON)
    parser.add_argument("--schedule", choices=["history_conditioning", "future_keyframe_inpainting"], default="history_conditioning")
    parser.add_argument("--guidance-scale", type=float, default=0.002)
    parser.add_argument("--command-velocity-x", type=float, default=0.35)
    parser.add_argument("--command-velocity-y", type=float, default=0.0)
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
    history = int(timestep_summary["settings"]["history"])
    feature_slices = timestep_summary.get("fixture_feature_slices")
    if feature_slices is None:
        # The timestep summary is built from the motion-state fixture, whose feature layout is stable and documented.
        # root_lin_vel_yaw_frame occupies [9, 12] in the 181-D candidate hybrid state.
        root_lin_vel_slice = slice(9, 12)
    else:
        lo, hi = feature_slices["root_lin_vel_yaw_frame"]
        root_lin_vel_slice = slice(lo, hi)

    command_velocity = [args.command_velocity_x, args.command_velocity_y]
    tau_unguided = tau_start.copy()
    tau_guided = tau_start.copy()
    iter_steps = steps.copy()
    max_steps = int(np.max(steps))

    unguided_cost_trace = [eval_guidance_cost(tau_unguided, history, root_lin_vel_slice, command_velocity)]
    guided_cost_trace = [eval_guidance_cost(tau_guided, history, root_lin_vel_slice, command_velocity)]
    unguided_mse_trace = [mse(tau_unguided, tau_clean)]
    guided_mse_trace = [mse(tau_guided, tau_clean)]
    gradient_norm_trace: list[float] = []
    max_step_trace = [int(np.max(iter_steps))]

    for _ in range(max_steps):
        tau_unguided, _ = oracle_reverse_step(
            tau_unguided,
            tau_clean,
            iter_steps,
            alpha_bars,
            state_dim,
            latent_dim,
            observed_mask,
        )
        tau_guided, next_steps = oracle_reverse_step(
            tau_guided,
            tau_clean,
            iter_steps,
            alpha_bars,
            state_dim,
            latent_dim,
            observed_mask,
        )
        tau_guided, _, grad_norm = apply_guidance(
            tau_guided,
            tau_clean,
            observed_mask,
            state_dim,
            latent_dim,
            history,
            root_lin_vel_slice,
            command_velocity,
            args.guidance_scale,
        )
        gradient_norm_trace.append(grad_norm)
        iter_steps = next_steps
        unguided_cost_trace.append(eval_guidance_cost(tau_unguided, history, root_lin_vel_slice, command_velocity))
        guided_cost_trace.append(eval_guidance_cost(tau_guided, history, root_lin_vel_slice, command_velocity))
        unguided_mse_trace.append(mse(tau_unguided, tau_clean))
        guided_mse_trace.append(mse(tau_guided, tau_clean))
        max_step_trace.append(int(np.max(iter_steps)))

    mask = full_observed_mask(observed_mask, state_dim, latent_dim)
    guided_clamp_error = float(np.max(np.abs(tau_guided[mask] - tau_clean[mask]))) if np.any(mask) else 0.0
    unguided_clamp_error = float(np.max(np.abs(tau_unguided[mask] - tau_clean[mask]))) if np.any(mask) else 0.0

    npz_path = OUT / "level_c_guided_reverse_loop_probe.npz"
    json_path = OUT / "level_c_guided_reverse_loop_probe.json"
    tsv_path = OUT / "level_c_guided_reverse_loop_probe.tsv"
    np.savez_compressed(
        npz_path,
        tau_clean=tau_clean,
        tau_start=tau_start,
        tau_unguided_final=tau_unguided,
        tau_guided_final=tau_guided,
        initial_steps=steps,
        final_steps=iter_steps,
        unguided_guidance_cost_trace=np.asarray(unguided_cost_trace, dtype=np.float64),
        guided_guidance_cost_trace=np.asarray(guided_cost_trace, dtype=np.float64),
        unguided_mse_trace=np.asarray(unguided_mse_trace, dtype=np.float64),
        guided_mse_trace=np.asarray(guided_mse_trace, dtype=np.float64),
        gradient_norm_trace=np.asarray(gradient_norm_trace, dtype=np.float64),
        max_step_trace=np.asarray(max_step_trace, dtype=np.int64),
    )

    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "candidate guided reverse-loop probe with oracle clean predictor and joystick cost",
        "paper_evidence": {
            "classifier_guidance_gradient": str(ROOT / "reproduction/paper/source/tex/method.tex:212-226"),
            "reverse_formula": str(ROOT / "reproduction/paper/source/tex/method.tex:197-206"),
            "joystick_cost_formula": str(ROOT / "reproduction/paper/source/root.tex:548-559"),
        },
        "not_a_replacement_for": [
            "trained denoising network",
            "paper-exact guidance scale protocol",
            "paper-exact alpha/gamma/sigma schedule",
            "validation/test scene split",
            "guided rollout metrics",
            "TensorRT deployment",
            "Fig. 5/Fig. 6 reproduction",
        ],
        "settings": {
            "schedule": args.schedule,
            "state_dim": state_dim,
            "latent_dim": latent_dim,
            "tau_shape": list(tau_clean.shape),
            "history": history,
            "denoising_steps": int(len(alpha_bars)),
            "guidance_scale": args.guidance_scale,
            "command_velocity_xy": command_velocity,
            "root_lin_vel_slice": [root_lin_vel_slice.start, root_lin_vel_slice.stop],
        },
        "metrics": {
            "initial_guidance_cost": unguided_cost_trace[0],
            "unguided_final_guidance_cost": unguided_cost_trace[-1],
            "guided_final_guidance_cost": guided_cost_trace[-1],
            "guided_cost_improvement_vs_unguided_final": unguided_cost_trace[-1] - guided_cost_trace[-1],
            "initial_mse": unguided_mse_trace[0],
            "unguided_final_mse": unguided_mse_trace[-1],
            "guided_final_mse": guided_mse_trace[-1],
            "unguided_observed_clamp_max_abs_error": unguided_clamp_error,
            "guided_observed_clamp_max_abs_error": guided_clamp_error,
            "initial_max_step": int(np.max(steps)),
            "final_max_step": int(np.max(iter_steps)),
            "gradient_norm_trace": gradient_norm_trace,
            "unguided_guidance_cost_trace": unguided_cost_trace,
            "guided_guidance_cost_trace": guided_cost_trace,
            "unguided_mse_trace": unguided_mse_trace,
            "guided_mse_trace": guided_mse_trace,
            "max_step_trace": max_step_trace,
        },
        "checks": {
            "all_steps_reach_zero": bool(np.max(iter_steps) == 0),
            "observed_tokens_clamped_unguided": bool(unguided_clamp_error < 1e-12),
            "observed_tokens_clamped_guided": bool(guided_clamp_error < 1e-12),
            "guidance_gradients_nonzero": bool(all(g > 0.0 for g in gradient_norm_trace)),
            "guided_final_cost_below_unguided_final": bool(guided_cost_trace[-1] < unguided_cost_trace[-1]),
            "guided_cost_decreases_from_initial": bool(guided_cost_trace[-1] < guided_cost_trace[0]),
            "unguided_reverse_reduces_mse": bool(unguided_mse_trace[-1] < unguided_mse_trace[0]),
            "guided_reverse_keeps_finite_mse": bool(np.isfinite(guided_mse_trace[-1])),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, summary)
    print(json.dumps({"status": "ok", "json": str(json_path), "npz": str(npz_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
