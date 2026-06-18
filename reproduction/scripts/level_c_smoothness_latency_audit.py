#!/usr/bin/env python3
"""Debug-only trajectory/action smoothness and latency-budget audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/smoothness_latency_audit"
GUIDED_JSON = ROOT / "res/level_c/guided_reverse_loop_probe/level_c_guided_reverse_loop_probe.json"
GUIDED_NPZ = ROOT / "res/level_c/guided_reverse_loop_probe/level_c_guided_reverse_loop_probe.npz"
DECODER_JSON = ROOT / "res/level_c/receding_horizon_decoder_probe/level_c_receding_horizon_decoder_probe.json"
DECODER_NPZ = ROOT / "res/level_c/receding_horizon_decoder_probe/level_c_receding_horizon_decoder_probe.npz"
ROOT_TEX = ROOT / "reproduction/paper/source/root.tex"
METHOD_TEX = ROOT / "reproduction/paper/source/tex/method.tex"


def finite_difference_metrics(values: np.ndarray, prefix: str) -> dict[str, float]:
    first = np.diff(values, axis=0)
    second = np.diff(values, n=2, axis=0)
    return {
        f"{prefix}_first_difference_mean_norm": float(np.linalg.norm(first, axis=-1).mean()),
        f"{prefix}_first_difference_max_norm": float(np.linalg.norm(first, axis=-1).max()),
        f"{prefix}_second_difference_mean_norm": float(np.linalg.norm(second, axis=-1).mean()),
        f"{prefix}_second_difference_max_norm": float(np.linalg.norm(second, axis=-1).max()),
    }


def row_metrics(name: str, tau: np.ndarray, state_dim: int, latent_dim: int) -> dict[str, Any]:
    state = tau[:, :state_dim]
    latent = tau[:, state_dim : state_dim + latent_dim]
    root_xy = state[:, :2]
    root_vel_xy = state[:, 9:11]
    row: dict[str, Any] = {"trajectory_name": name}
    row.update(finite_difference_metrics(state, "state"))
    row.update(finite_difference_metrics(latent, "latent"))
    row.update(finite_difference_metrics(root_xy, "root_xy"))
    row.update(finite_difference_metrics(root_vel_xy, "root_velocity_xy"))
    return row


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def text_has_patterns(path: Path, patterns: list[str]) -> bool:
    text = path.read_text(encoding="utf-8")
    return all(pattern in text for pattern in patterns)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    guided_summary = json.loads(GUIDED_JSON.read_text(encoding="utf-8"))
    decoder_summary = json.loads(DECODER_JSON.read_text(encoding="utf-8"))
    guided_data = np.load(GUIDED_NPZ)
    decoder_data = np.load(DECODER_NPZ)

    state_dim = int(guided_summary["settings"]["state_dim"])
    latent_dim = int(guided_summary["settings"]["latent_dim"])
    tau_names = ["tau_clean", "tau_start", "tau_unguided_final", "tau_guided_final"]
    rows = [row_metrics(name, guided_data[name], state_dim, latent_dim) for name in tau_names]

    action_pair = np.stack(
        [decoder_data["current_action"].astype(np.float64), decoder_data["next_latent_action_for_contrast"].astype(np.float64)],
        axis=0,
    )
    action_first_delta = float(np.linalg.norm(np.diff(action_pair, axis=0), axis=-1)[0])
    guidance_trace = np.asarray(guided_summary["metrics"]["guided_guidance_cost_trace"], dtype=np.float64)
    mse_trace = np.asarray(guided_summary["metrics"]["guided_mse_trace"], dtype=np.float64)
    gradient_trace = np.asarray(guided_summary["metrics"]["gradient_norm_trace"], dtype=np.float64)

    denoising_steps = int(guided_summary["settings"]["denoising_steps"])
    control_dt = float(decoder_summary["metrics"]["control_dt_seconds"])
    paper_control_period_ms = control_dt * 1000.0
    paper_denoising_total_ms = 20.0
    paper_decoder_target_ms = 1.0
    paper_async_remaining_budget_ms = paper_control_period_ms - paper_denoising_total_ms

    json_path = OUT / "level_c_smoothness_latency_audit.json"
    tsv_path = OUT / "level_c_smoothness_latency_audit.tsv"
    npz_path = OUT / "level_c_smoothness_latency_audit.npz"
    write_tsv(tsv_path, rows)
    np.savez_compressed(
        npz_path,
        trajectory_rows=np.asarray([[row[key] for key in row if key != "trajectory_name"] for row in rows], dtype=np.float64),
        guided_guidance_cost_trace=guidance_trace,
        guided_mse_trace=mse_trace,
        gradient_norm_trace=gradient_trace,
        action_pair=action_pair,
    )

    guided_row = next(row for row in rows if row["trajectory_name"] == "tau_guided_final")
    clean_row = next(row for row in rows if row["trajectory_name"] == "tau_clean")
    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "debug_only",
        "scope": "smoothness, guidance-cost trace, and paper latency-budget audit over debug guided reverse-loop artifacts",
        "paper_evidence": {
            "action_smoothness_context": str(METHOD_TEX) + ":76",
            "latency_context": str(METHOD_TEX) + ":147",
            "control_rate_and_diffusion_latency": str(ROOT_TEX) + ":589-593",
            "goal_diffusion_metrics": str(ROOT / "goal.md:1603-1615"),
        },
        "not_a_replacement_for": [
            "trained diffusion rollout smoothness",
            "real TensorRT latency benchmark",
            "closed-loop action sequence",
            "paper Fig. 5/Fig. 6 evaluation",
        ],
        "settings": {
            "state_dim": state_dim,
            "latent_dim": latent_dim,
            "trajectory_shape": list(guided_data["tau_guided_final"].shape),
            "denoising_steps": denoising_steps,
            "control_dt_seconds": control_dt,
            "paper_control_period_ms": paper_control_period_ms,
            "paper_denoising_total_ms": paper_denoising_total_ms,
            "paper_decoder_target_ms": paper_decoder_target_ms,
        },
        "trajectory_rows": rows,
        "metrics": {
            "guided_final_state_second_difference_mean_norm": guided_row["state_second_difference_mean_norm"],
            "clean_state_second_difference_mean_norm": clean_row["state_second_difference_mean_norm"],
            "guided_final_latent_second_difference_mean_norm": guided_row["latent_second_difference_mean_norm"],
            "clean_latent_second_difference_mean_norm": clean_row["latent_second_difference_mean_norm"],
            "schema_action_delta_current_vs_next_latent": action_first_delta,
            "initial_guidance_cost": float(guidance_trace[0]),
            "final_guidance_cost": float(guidance_trace[-1]),
            "guidance_cost_reduction": float(guidance_trace[0] - guidance_trace[-1]),
            "initial_mse": float(mse_trace[0]),
            "final_mse": float(mse_trace[-1]),
            "mse_reduction": float(mse_trace[0] - mse_trace[-1]),
            "mean_guidance_gradient_norm": float(gradient_trace.mean()),
            "paper_async_remaining_budget_ms": paper_async_remaining_budget_ms,
            "paper_denoising_fraction_of_control_period": paper_denoising_total_ms / paper_control_period_ms,
            "paper_decoder_fraction_of_control_period_upper_bound": paper_decoder_target_ms / paper_control_period_ms,
        },
        "checks": {
            "paper_source_latency_patterns_found": text_has_patterns(
                ROOT_TEX, ["25 Hz", "20\\,\\mathrm{ms}", "20 denoising steps", "under $1.0$~ms"]
            ),
            "goal_metric_names_indexed": text_has_patterns(
                ROOT / "goal.md",
                ["state reconstruction error", "latent reconstruction error", "action smoothness", "trajectory smoothness", "inference latency", "denoising latency", "guidance cost"],
            ),
            "guided_tau_shape_matches_settings": list(guided_data["tau_guided_final"].shape) == [21, state_dim + latent_dim],
            "all_smoothness_metrics_finite": bool(
                all(np.isfinite(value) for row in rows for key, value in row.items() if key != "trajectory_name")
            ),
            "guidance_cost_decreases": bool(guidance_trace[-1] < guidance_trace[0]),
            "guided_mse_decreases": bool(mse_trace[-1] < mse_trace[0]),
            "guidance_gradients_finite_positive": bool(np.all(np.isfinite(gradient_trace)) and np.all(gradient_trace > 0.0)),
            "control_rate_25hz_recorded": abs(paper_control_period_ms - 40.0) < 1e-12,
            "paper_denoising_latency_within_control_period": paper_denoising_total_ms < paper_control_period_ms,
            "paper_decoder_target_within_remaining_budget": paper_decoder_target_ms < paper_async_remaining_budget_ms,
            "schema_action_delta_finite_positive": bool(np.isfinite(action_first_delta) and action_first_delta > 0.0),
            "debug_only_boundary_recorded": True,
        },
        "interpretation": {
            "paper_level_status": "partial",
            "why_not_complete": (
                "This reports smoothness and latency-budget metrics for debug oracle/guided artifacts and paper latency "
                "targets. It does not measure a trained diffusion TensorRT engine, a real closed-loop action sequence, "
                "or paper rollout smoothness."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
