#!/usr/bin/env python3
"""Debug-only guidance-scale sweep for the Level C guided reverse loop."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SCRIPTS = ROOT / "reproduction/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from level_c_guided_reverse_loop_probe import (  # noqa: E402
    DEFAULT_TIMESTEP_JSON,
    DEFAULT_TIMESTEP_NPZ,
    apply_guidance,
    clamp_observed,
    eval_guidance_cost,
    full_observed_mask,
    mse,
    oracle_reverse_step,
)


OUT = ROOT / "res/level_c/guidance_scale_sweep_probe"


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "scale",
        "initial_guidance_cost",
        "unguided_final_guidance_cost",
        "guided_final_guidance_cost",
        "improvement_vs_unguided_final",
        "initial_mse",
        "unguided_final_mse",
        "guided_final_mse",
        "guided_clamp_error",
        "final_max_step",
        "mean_gradient_norm",
        "valid",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def parse_scales(text: str) -> list[float]:
    scales = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not scales:
        raise ValueError("at least one guidance scale is required")
    if len(set(scales)) != len(scales):
        raise ValueError(f"duplicate guidance scales are not allowed: {scales}")
    return scales


def run_scale(
    *,
    scale: float,
    tau_start: np.ndarray,
    tau_clean: np.ndarray,
    steps: np.ndarray,
    alpha_bars: np.ndarray,
    observed_mask: np.ndarray,
    state_dim: int,
    latent_dim: int,
    history: int,
    root_lin_vel_slice: slice,
    command_velocity: list[float],
) -> dict[str, Any]:
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
        if scale != 0.0:
            tau_guided, _, grad_norm = apply_guidance(
                tau_guided,
                tau_clean,
                observed_mask,
                state_dim,
                latent_dim,
                history,
                root_lin_vel_slice,
                command_velocity,
                scale,
            )
        else:
            tau_guided = clamp_observed(tau_guided, tau_clean, observed_mask, state_dim, latent_dim)
            grad_norm = 0.0
        gradient_norm_trace.append(grad_norm)
        iter_steps = next_steps
        unguided_cost_trace.append(eval_guidance_cost(tau_unguided, history, root_lin_vel_slice, command_velocity))
        guided_cost_trace.append(eval_guidance_cost(tau_guided, history, root_lin_vel_slice, command_velocity))
        unguided_mse_trace.append(mse(tau_unguided, tau_clean))
        guided_mse_trace.append(mse(tau_guided, tau_clean))
        max_step_trace.append(int(np.max(iter_steps)))

    mask = full_observed_mask(observed_mask, state_dim, latent_dim)
    guided_clamp_error = float(np.max(np.abs(tau_guided[mask] - tau_clean[mask]))) if np.any(mask) else 0.0
    row: dict[str, Any] = {
        "scale": scale,
        "initial_guidance_cost": unguided_cost_trace[0],
        "unguided_final_guidance_cost": unguided_cost_trace[-1],
        "guided_final_guidance_cost": guided_cost_trace[-1],
        "improvement_vs_unguided_final": unguided_cost_trace[-1] - guided_cost_trace[-1],
        "initial_mse": unguided_mse_trace[0],
        "unguided_final_mse": unguided_mse_trace[-1],
        "guided_final_mse": guided_mse_trace[-1],
        "guided_clamp_error": guided_clamp_error,
        "final_max_step": int(np.max(iter_steps)),
        "mean_gradient_norm": float(np.mean(gradient_norm_trace)) if gradient_norm_trace else 0.0,
        "valid": bool(np.isfinite(guided_mse_trace[-1]) and guided_clamp_error < 1e-12 and np.max(iter_steps) == 0),
        "traces": {
            "unguided_guidance_cost": unguided_cost_trace,
            "guided_guidance_cost": guided_cost_trace,
            "unguided_mse": unguided_mse_trace,
            "guided_mse": guided_mse_trace,
            "gradient_norm": gradient_norm_trace,
            "max_step": max_step_trace,
        },
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestep-npz", type=Path, default=DEFAULT_TIMESTEP_NPZ)
    parser.add_argument("--timestep-json", type=Path, default=DEFAULT_TIMESTEP_JSON)
    parser.add_argument("--schedule", choices=["history_conditioning", "future_keyframe_inpainting"], default="history_conditioning")
    parser.add_argument("--scales", type=str, default="0,0.0005,0.001,0.002,0.005,0.01,0.02")
    parser.add_argument("--command-velocity-x", type=float, default=0.35)
    parser.add_argument("--command-velocity-y", type=float, default=0.0)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    scales = parse_scales(args.scales)
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
    root_lin_vel_slice = slice(9, 12)
    command_velocity = [args.command_velocity_x, args.command_velocity_y]

    rows = [
        run_scale(
            scale=scale,
            tau_start=tau_start,
            tau_clean=tau_clean,
            steps=steps,
            alpha_bars=alpha_bars,
            observed_mask=observed_mask,
            state_dim=state_dim,
            latent_dim=latent_dim,
            history=history,
            root_lin_vel_slice=root_lin_vel_slice,
            command_velocity=command_velocity,
        )
        for scale in scales
    ]
    valid_rows = [row for row in rows if row["valid"]]
    if not valid_rows:
        raise RuntimeError("no valid guidance-scale sweep rows")
    best_row = min(valid_rows, key=lambda row: row["guided_final_guidance_cost"])
    baseline = next(row for row in rows if row["scale"] == 0.0) if 0.0 in scales else rows[0]

    json_path = OUT / "level_c_guidance_scale_sweep_probe.json"
    tsv_path = OUT / "level_c_guidance_scale_sweep_probe.tsv"
    npz_path = OUT / "level_c_guidance_scale_sweep_probe.npz"
    write_tsv(tsv_path, rows)
    np.savez_compressed(
        npz_path,
        scales=np.asarray([row["scale"] for row in rows], dtype=np.float64),
        guided_final_guidance_cost=np.asarray([row["guided_final_guidance_cost"] for row in rows], dtype=np.float64),
        improvement_vs_unguided_final=np.asarray([row["improvement_vs_unguided_final"] for row in rows], dtype=np.float64),
        guided_final_mse=np.asarray([row["guided_final_mse"] for row in rows], dtype=np.float64),
        guided_clamp_error=np.asarray([row["guided_clamp_error"] for row in rows], dtype=np.float64),
    )

    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "candidate joystick guidance-scale sweep over oracle reverse-loop probe",
        "paper_evidence": {
            "classifier_guidance_gradient": str(ROOT / "reproduction/paper/source/tex/method.tex:212-226"),
            "joystick_cost_formula": str(ROOT / "reproduction/paper/source/root.tex:548-559"),
            "goal_scale_selection": str(ROOT / "goal.md:1290-1355"),
        },
        "not_a_replacement_for": [
            "paper-exact guidance scale protocol",
            "validation/test scene split",
            "trained denoising network",
            "guided rollout metrics",
            "TensorRT deployment",
            "Fig. 5/Fig. 6 reproduction",
        ],
        "settings": {
            "schedule": args.schedule,
            "scales": scales,
            "state_dim": state_dim,
            "latent_dim": latent_dim,
            "tau_shape": list(tau_clean.shape),
            "history": history,
            "command_velocity_xy": command_velocity,
            "root_lin_vel_slice": [root_lin_vel_slice.start, root_lin_vel_slice.stop],
        },
        "baseline_scale_zero": {
            key: baseline[key]
            for key in [
                "scale",
                "guided_final_guidance_cost",
                "guided_final_mse",
                "guided_clamp_error",
                "final_max_step",
            ]
        },
        "selected_best": {
            key: best_row[key]
            for key in [
                "scale",
                "guided_final_guidance_cost",
                "improvement_vs_unguided_final",
                "guided_final_mse",
                "guided_clamp_error",
                "final_max_step",
                "mean_gradient_norm",
            ]
        },
        "rows": rows,
        "checks": {
            "all_rows_valid": bool(all(row["valid"] for row in rows)),
            "all_rows_reach_zero_step": bool(all(row["final_max_step"] == 0 for row in rows)),
            "all_rows_keep_clamps": bool(all(row["guided_clamp_error"] < 1e-12 for row in rows)),
            "best_improves_over_zero_scale": bool(best_row["guided_final_guidance_cost"] < baseline["guided_final_guidance_cost"]),
            "positive_scales_have_nonzero_gradients": bool(
                all(row["mean_gradient_norm"] > 0.0 for row in rows if row["scale"] > 0.0)
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
