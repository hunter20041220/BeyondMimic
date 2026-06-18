#!/usr/bin/env python3
"""Runtime integration audit for the lightweight BeyondMimic reimplementation package."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SRC = ROOT / "reproduction/src"
LATENTS_NPZ = ROOT / "reproduction/data/level_c_vae_debug_latents/debug_tiny_vae_state_latent_windows.npz"
ACTION_NPZ = ROOT / "res/level_c/diffusion_to_vae_action_smoke/level_c_diffusion_to_vae_action_smoke.npz"
FIXTURE_NPZ = ROOT / "reproduction/data/level_c_fixtures/walk1_subject1_frames_1_180_state_fixture.npz"
OUT = ROOT / "res/code/reimpl_runtime_integration_audit"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beyondmimic_reimpl.dagger import build_dagger_sample, teacher_student_discrepancy
from beyondmimic_reimpl.diffusion import apply_observation_mask, denoise_one_step_with_oracle_eps, q_sample
from beyondmimic_reimpl.evaluation import action_mse, survival_rate, tracking_error
from beyondmimic_reimpl.guidance import finite_difference_grad, gaussian_reward, sdf_barrier
from beyondmimic_reimpl.state import emphasis_projection, smoothness_penalty
from beyondmimic_reimpl.trajectory import build_state_latent_window, split_counts, stack_state_latent_tokens
from beyondmimic_reimpl.vae import kl_standard_normal, reparameterize


MOTION_SPLITS = {
    "walk1_subject1_frames_1_180_state_fixture": "train",
    "run2_subject1_frames_1_180_state_fixture": "validation",
    "jumps1_subject1_frames_1_180_state_fixture": "test",
}


def finite(value: float) -> bool:
    return math.isfinite(float(value))


def load_windows(npz: np.lib.npyio.NpzFile) -> list[Any]:
    windows = []
    for motion, split in MOTION_SPLITS.items():
        for idx in range(28):
            prefix = f"{motion}_window_{idx:04d}"
            states = np.asarray(npz[f"{prefix}_states"], dtype=np.float64)
            latents = np.asarray(npz[f"{prefix}_latents"], dtype=np.float64)
            windows.append(
                build_state_latent_window(
                    sample_id=prefix,
                    source_motion=motion,
                    start_timestep=idx,
                    split=split,  # type: ignore[arg-type]
                    accepted=True,
                    states=states,
                    latents=latents,
                )
            )
    return windows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    latent_npz = np.load(LATENTS_NPZ, allow_pickle=False)
    action_npz = np.load(ACTION_NPZ, allow_pickle=False)
    fixture_npz = np.load(FIXTURE_NPZ, allow_pickle=False)

    windows = load_windows(latent_npz)
    tokens = np.stack([stack_state_latent_tokens(w.states, w.latents) for w in windows], axis=0)
    split_count_map = split_counts(windows)
    teacher_actions = np.stack(
        [
            np.asarray(latent_npz[f"{w.sample_id}_teacher_action"], dtype=np.float64)
            for w in windows
        ],
        axis=0,
    )
    decoded_actions = np.stack(
        [
            np.asarray(latent_npz[f"{w.sample_id}_decoded_action"], dtype=np.float64)
            for w in windows
        ],
        axis=0,
    )
    logvars = np.stack(
        [np.asarray(latent_npz[f"{w.sample_id}_logvar"], dtype=np.float64) for w in windows],
        axis=0,
    )

    p, p_inv = emphasis_projection(seed=17, state_dim=99, root_dim=18, coefficient=6)
    projected = windows[0].states @ p.T
    recovered = projected @ p_inv.T
    projection_reconstruction_max_error = float(np.max(np.abs(recovered - windows[0].states)))

    rng = np.random.default_rng(20261001)
    eps = rng.normal(size=tokens[0].shape)
    noisy = q_sample(tokens[0], eps, alpha_bar=0.42)
    reversed_tokens = denoise_one_step_with_oracle_eps(noisy, tokens[0], alpha_bar_t=0.42, alpha_bar_prev=0.72)
    diffusion_mse_before = float(np.mean((noisy - tokens[0]) ** 2))
    diffusion_mse_after = float(np.mean((reversed_tokens - tokens[0]) ** 2))

    observed = np.zeros_like(tokens[0], dtype=bool)
    observed[:5, :] = True
    observed[[10, 15, 20], :99] = True
    clamped = apply_observation_mask(noisy, tokens[0], observed)
    observation_clamp_max_error = float(np.max(np.abs(clamped[observed] - tokens[0][observed])))

    decoded_teacher_action_mse = action_mse(decoded_actions, teacher_actions)
    downstream_action_mse = action_mse(action_npz["predicted_action"], action_npz["target_action"])
    current_downstream_action_mse = action_mse(
        action_npz["predicted_action"][:, 4, :],
        action_npz["target_action"][:, 4, :],
    )
    smoothness = smoothness_penalty(action_npz["predicted_action"].reshape(-1, 29))

    dagger_samples = []
    for idx, window in enumerate(windows[:30]):
        dagger_samples.append(
            build_dagger_sample(
                sample_id=f"runtime_integration_{idx:04d}",
                rollout_id=window.source_motion,
                timestep=4,
                state=tokens[idx, 4],
                student_action=decoded_actions[idx, 4],
                teacher_action=teacher_actions[idx, 4],
                teacher_queried=True,
                accepted=True,
                split=window.split,
            )
        )
    dagger_metrics = teacher_student_discrepancy(dagger_samples)

    mu = windows[0].latents[4]
    logvar = logvars[0, 4]
    eps_latent = np.zeros_like(mu)
    z = reparameterize(mu, logvar, eps_latent)
    kl = kl_standard_normal(mu, logvar)
    reparameterize_zero_eps_error = float(np.max(np.abs(z - mu)))

    target_velocity = np.array([0.5, -0.2], dtype=np.float64)
    current_velocity = tokens[0, 4, :2].copy()

    def joystick_cost(v: np.ndarray) -> float:
        err = v - target_velocity
        return float(0.5 * np.dot(err, err))

    joystick_grad = finite_difference_grad(joystick_cost, current_velocity)
    gaussian_tracking_reward = gaussian_reward(current_velocity - target_velocity, sigma=0.3)
    sdf_cost = sdf_barrier(np.array([0.05, 0.12, 0.3], dtype=np.float64), delta=0.1)

    body_pos = np.asarray(fixture_npz["body_pos_w"][:21], dtype=np.float64)
    tracking_metrics = tracking_error(body_pos + 0.01, body_pos)
    survival = survival_rate(np.array([21, 21, 18, 25], dtype=np.float64), horizon=21)

    metrics: dict[str, Any] = {
        "window_count": len(windows),
        "token_shape": list(tokens.shape),
        "split_counts": split_count_map,
        "projection_shape": list(p.shape),
        "projection_reconstruction_max_error": projection_reconstruction_max_error,
        "diffusion_mse_before": diffusion_mse_before,
        "diffusion_mse_after": diffusion_mse_after,
        "observation_clamp_max_error": observation_clamp_max_error,
        "decoded_teacher_action_mse": decoded_teacher_action_mse,
        "downstream_action_mse": downstream_action_mse,
        "current_downstream_action_mse": current_downstream_action_mse,
        "predicted_action_smoothness_penalty": smoothness,
        "dagger_action_mse": dagger_metrics["action_mse"],
        "dagger_teacher_query_count": dagger_metrics["teacher_query_count"],
        "vae_kl_current_token": kl,
        "vae_reparameterize_zero_eps_error": reparameterize_zero_eps_error,
        "joystick_grad_norm": float(np.linalg.norm(joystick_grad)),
        "gaussian_tracking_reward": gaussian_tracking_reward,
        "sdf_barrier_cost": sdf_cost,
        "tracking_mean_error": tracking_metrics["mean_error"],
        "tracking_max_error": tracking_metrics["max_error"],
        "survival_rate": survival,
    }
    rows = [
        {"metric": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value}
        for key, value in metrics.items()
    ]
    checks = {
        "latents_npz_exists": LATENTS_NPZ.is_file(),
        "action_npz_exists": ACTION_NPZ.is_file(),
        "fixture_npz_exists": FIXTURE_NPZ.is_file(),
        "window_count_84": len(windows) == 84,
        "split_counts_28_each": split_count_map == {"train": 28, "validation": 28, "test": 28},
        "token_shape_84_21_131": list(tokens.shape) == [84, 21, 131],
        "state_dim_99": tokens.shape[-1] - windows[0].latents.shape[-1] == 99,
        "latent_dim_32": windows[0].latents.shape[-1] == 32,
        "action_dim_29": decoded_actions.shape[-1] == 29,
        "all_metrics_finite": all(
            finite(value)
            for value in metrics.values()
            if not isinstance(value, (dict, list))
        ),
        "projection_reconstructs_state": projection_reconstruction_max_error < 1e-10,
        "diffusion_reverse_reduces_mse": diffusion_mse_after < diffusion_mse_before,
        "observation_mask_clamps_exactly": observation_clamp_max_error == 0.0,
        "decoded_teacher_action_mse_below_0_01": decoded_teacher_action_mse < 0.01,
        "dagger_queries_recorded": dagger_metrics["teacher_query_count"] == 30.0,
        "vae_reparameterize_zero_eps_matches_mu": reparameterize_zero_eps_error == 0.0,
        "guidance_gradient_nonzero": metrics["joystick_grad_norm"] > 0.0,
        "tracking_error_positive": tracking_metrics["mean_error"] > 0.0,
        "survival_rate_in_unit_interval": 0.0 <= survival <= 1.0,
        "does_not_claim_official_training_code": True,
        "does_not_claim_trained_checkpoint": True,
        "does_not_claim_closed_loop_rollout": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "reimpl_runtime_integration_audit",
        "scope": "package-level runtime integration over local debug state-latent and action fixtures",
        "metrics": metrics,
        "checks": checks,
        "rows": rows,
        "api_groups_exercised": [
            "trajectory",
            "state",
            "diffusion",
            "evaluation",
            "dagger",
            "vae",
            "guidance",
        ],
        "inputs": {
            "latents_npz": str(LATENTS_NPZ),
            "action_npz": str(ACTION_NPZ),
            "fixture_npz": str(FIXTURE_NPZ),
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The clean-room package APIs run together on local debug fixtures, but this is still not the "
                "unpublished official training/deployment code, a trained checkpoint, Isaac closed-loop rollout, "
                "TensorRT engine, or paper-level evaluation."
            ),
        },
        "outputs": {
            "json": str(OUT / "reimpl_runtime_integration_audit.json"),
            "tsv": str(OUT / "reimpl_runtime_integration_audit.tsv"),
        },
    }
    (OUT / "reimpl_runtime_integration_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (OUT / "reimpl_runtime_integration_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "metrics": len(metrics)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
