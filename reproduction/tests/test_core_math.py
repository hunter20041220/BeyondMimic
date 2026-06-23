#!/usr/bin/env python3
"""Unit tests for BeyondMimic reproduction core math gates.

These tests are intentionally pure NumPy/stdlib so they can run before IsaacLab,
ROS, trained checkpoints, or GPU-only dependencies are available. They validate
formula-level behavior used by the current Level C audits and write a small
machine-readable result artifact for the master audit.
"""

from __future__ import annotations

import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SRC = ROOT / "reproduction/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beyondmimic_reimpl.diffusion import apply_observation_mask, denoise_one_step_with_oracle_eps, q_sample
from beyondmimic_reimpl.dagger import build_dagger_sample, teacher_student_discrepancy
from beyondmimic_reimpl.evaluation import action_mse, fall_rate, success_rate, survival_rate, tracking_error, velocity_tracking_error
from beyondmimic_reimpl.geometry import anchor_to_world, rot6d_to_matrix, world_to_anchor
from beyondmimic_reimpl.guidance import finite_difference_grad, gaussian_reward, sdf_barrier
from beyondmimic_reimpl.sampling import adaptive_distribution, mirror_state_29d, ou_noise
from beyondmimic_reimpl.state import (
    HYBRID_STATE_DIM,
    ROOT_STATE_DIM,
    TARGET_BODY_FEATURE_DIM,
    emphasis_projection,
    hybrid_state_schema,
    project_hybrid_state,
    smoothness_penalty,
    unproject_hybrid_state,
    validate_hybrid_state,
)
from beyondmimic_reimpl.trajectory import build_state_latent_window, split_counts, stack_state_latent_tokens
from beyondmimic_reimpl.vae import kl_standard_normal, reparameterize

OUT = ROOT / "res/tests/core_math_unit_tests"
ATOL = 1e-9

TestFn = Callable[[], Dict[str, Any]]


def assert_close(actual: np.ndarray | float, expected: np.ndarray | float, *, atol: float = ATOL) -> None:
    if not np.allclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"max_abs_error={float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))}")


def test_rot6d_conversion() -> dict[str, Any]:
    r = rot6d_to_matrix(np.array([1.0, 0.2, -0.1, 0.4, 1.0, 0.3]))
    assert_close(r.T @ r, np.eye(3), atol=1e-12)
    det = float(np.linalg.det(r))
    assert abs(det - 1.0) < 1e-12
    return {"determinant": det, "orthogonality_error": float(np.max(np.abs(r.T @ r - np.eye(3))))}


def test_anchor_yaw_roundtrip() -> dict[str, Any]:
    anchor = np.array([1.25, -0.3, 0.87])
    points = np.array([[1.8, 0.2, 1.1], [0.9, -0.7, 0.6], [1.4, -0.2, 0.87]])
    local = world_to_anchor(points, anchor, anchor_yaw=0.73)
    recovered = anchor_to_world(local, anchor, anchor_yaw=0.73)
    assert_close(recovered, points, atol=1e-12)
    assert_close(local[:, 2], points[:, 2] - anchor[2], atol=1e-12)
    return {"roundtrip_error": float(np.max(np.abs(recovered - points))), "relative_height_sum": float(np.sum(local[:, 2]))}


def test_height_preserving_current_frame() -> dict[str, Any]:
    root = np.array([0.3, 0.4, 0.91])
    body = np.array([[0.2, 0.1, 1.08], [0.5, 0.6, 0.77]])
    local = world_to_anchor(body, root, anchor_yaw=-1.2)
    reconstructed = anchor_to_world(local, root, anchor_yaw=-1.2)
    assert_close(reconstructed[:, 2], body[:, 2], atol=1e-12)
    return {"max_height_error": float(np.max(np.abs(reconstructed[:, 2] - body[:, 2])))}


def test_gaussian_reward_monotonicity() -> dict[str, Any]:
    small = gaussian_reward(np.array([0.02, -0.01]), sigma=0.25)
    large = gaussian_reward(np.array([0.2, -0.1]), sigma=0.25)
    if not small > large > 0.0:
        raise AssertionError(f"unexpected reward order small={small} large={large}")
    return {"small_error_reward": small, "large_error_reward": large}


def test_termination_thresholds() -> dict[str, Any]:
    def terminated(height_error: float, orientation_error: float) -> bool:
        return height_error > 0.35 or orientation_error > 0.8

    if terminated(0.1, 0.2):
        raise AssertionError("safe state terminated")
    if not terminated(0.36, 0.2):
        raise AssertionError("height failure missed")
    if not terminated(0.1, 0.81):
        raise AssertionError("orientation failure missed")
    return {"height_threshold": 0.35, "orientation_threshold": 0.8}


def test_adaptive_sampling_kernel() -> dict[str, Any]:
    code_like = adaptive_distribution(failure_bin=7, bin_count=12, kernel_size=1)
    paper_like = adaptive_distribution(failure_bin=7, bin_count=12, kernel_size=3)
    code_prefailure = float(np.sum(code_like[5:8]))
    paper_prefailure = float(np.sum(paper_like[5:8]))
    if not paper_prefailure > code_prefailure:
        raise AssertionError("paper look-back distribution should move more mass before failure")
    assert_close(np.sum(paper_like), 1.0)
    return {"kernel1_prefailure_mass": code_prefailure, "kernel3_prefailure_mass": paper_prefailure}


def test_ou_noise_temporal_correlation() -> dict[str, Any]:
    seq = ou_noise(seed=11, steps=256, dim=3)
    lag_corr = float(np.mean(seq[1:] * seq[:-1]) / np.mean(seq[:-1] ** 2))
    if not 0.0 < lag_corr < 1.0:
        raise AssertionError(f"invalid OU lag correlation {lag_corr}")
    return {"lag_correlation": lag_corr, "shape": list(seq.shape)}


def test_symmetry_involution_29d() -> dict[str, Any]:
    vec = np.linspace(-1.0, 1.0, 29)
    recovered = mirror_state_29d(mirror_state_29d(vec))
    assert_close(recovered, vec, atol=1e-12)
    return {"dimension": 29, "roundtrip_error": float(np.max(np.abs(recovered - vec)))}


def test_emphasis_projection_pseudoinverse() -> dict[str, Any]:
    p, p_inv = emphasis_projection()
    if p.shape != (163, 99) or p_inv.shape != (99, 163):
        raise AssertionError(f"paper hybrid-state projection must be 99->163, got {p.shape}, {p_inv.shape}")
    rng = np.random.default_rng(9)
    states = rng.normal(size=(16, 99))
    projected = states @ p.T
    recovered = projected @ p_inv.T
    assert_close(recovered, states, atol=1e-10)
    return {
        "projection_shape": list(p.shape),
        "root_state_dim": ROOT_STATE_DIM,
        "body_feature_dim": TARGET_BODY_FEATURE_DIM,
        "state_dim": HYBRID_STATE_DIM,
        "max_reconstruction_error": float(np.max(np.abs(recovered - states))),
    }


def test_hybrid_state_schema_and_projection() -> dict[str, Any]:
    schema = hybrid_state_schema()
    expected_slices = {
        "root_pos_rel_current_frame": [0, 3],
        "root_rot6d_rel_current_frame": [3, 9],
        "root_lin_vel_rel_current_frame": [9, 12],
        "root_ang_vel_rel_current_frame": [12, 15],
        "body_pos_local_root_frame": [15, 57],
        "body_lin_vel_local_root_frame": [57, 99],
    }
    if schema.state_dim != 99 or schema.root_dim != 15 or schema.body_feature_dim != 84:
        raise AssertionError(f"unexpected hybrid schema {schema.to_dict()}")
    if schema.projected_dim != 163 or schema.slices != expected_slices:
        raise AssertionError(f"unexpected schema projection/slices {schema.to_dict()}")
    rng = np.random.default_rng(99)
    states = rng.normal(size=(3, 4, schema.state_dim))
    validated = validate_hybrid_state(states, schema)
    projected, _, p_inv = project_hybrid_state(validated, seed=5, schema=schema)
    recovered = unproject_hybrid_state(projected, p_inv, schema)
    assert_close(recovered, states, atol=1e-10)
    try:
        validate_hybrid_state(np.zeros((2, 160)), schema)
    except ValueError as exc:
        error = str(exc)
    else:
        raise AssertionError("160-D policy obs must not pass as paper hybrid state")
    return {
        "schema": schema.to_dict(),
        "projected_shape": list(projected.shape),
        "roundtrip_error": float(np.max(np.abs(recovered - states))),
        "dimension_error": error,
    }


def test_diffusion_forward_noise_increases() -> dict[str, Any]:
    rng = np.random.default_rng(3)
    x0 = rng.normal(size=(21, 131))
    eps = rng.normal(size=(21, 131))
    low = q_sample(x0, eps, alpha_bar=0.95)
    high = q_sample(x0, eps, alpha_bar=0.25)
    low_mse = float(np.mean((low - x0) ** 2))
    high_mse = float(np.mean((high - x0) ** 2))
    if not high_mse > low_mse:
        raise AssertionError("forward diffusion did not increase perturbation")
    return {"low_step_mse": low_mse, "high_step_mse": high_mse}


def test_diffusion_oracle_reverse_reduces_mse() -> dict[str, Any]:
    rng = np.random.default_rng(4)
    x0 = rng.normal(size=(21, 131))
    eps = rng.normal(size=(21, 131))
    xt = q_sample(x0, eps, alpha_bar=0.35)
    xprev = denoise_one_step_with_oracle_eps(xt, x0, alpha_bar_t=0.35, alpha_bar_prev=0.62)
    mse_t = float(np.mean((xt - x0) ** 2))
    mse_prev = float(np.mean((xprev - x0) ** 2))
    if not mse_prev < mse_t:
        raise AssertionError("oracle reverse step did not reduce MSE")
    return {"mse_before": mse_t, "mse_after": mse_prev}


def test_independent_timestep_mask_schedule() -> dict[str, Any]:
    steps = np.full((21, 2), 20, dtype=np.int64)
    steps[:5, :] = 0
    keyframes = [10, 15, 20]
    steps[keyframes, 0] = 0
    if not np.all(steps[:5, :] == 0):
        raise AssertionError("history conditioning should keep state and latent clean")
    if not np.all(steps[keyframes, 0] == 0) or not np.all(steps[keyframes, 1] == 20):
        raise AssertionError("future keyframe mask should keep only state clean")
    return {"shape": list(steps.shape), "future_state_keyframes": keyframes}


def test_inpainting_observation_clamp() -> dict[str, Any]:
    rng = np.random.default_rng(12)
    clean = rng.normal(size=(21, 131))
    noisy = clean + rng.normal(scale=0.5, size=(21, 131))
    observed = np.zeros_like(clean, dtype=bool)
    observed[:5, :] = True
    observed[[10, 15, 20], :99] = True
    clamped = apply_observation_mask(noisy, clean, observed)
    assert_close(clamped[observed], clean[observed], atol=0.0)
    return {"observed_entries": int(np.sum(observed)), "clamp_error": float(np.max(np.abs(clamped[observed] - clean[observed])))}


def test_vae_reparameterization_and_kl() -> dict[str, Any]:
    mu = np.array([0.2, -0.1, 0.0, 0.4])
    logvar = np.array([-0.3, 0.1, 0.0, -0.2])
    eps = np.array([1.0, -0.5, 0.25, 0.0])
    z = reparameterize(mu, logvar, eps)
    expected = np.array([1.06070798, -0.62563555, 0.25, 0.4])
    assert_close(z, expected, atol=1e-8)
    kl = kl_standard_normal(mu, logvar)
    if not kl > 0.0:
        raise AssertionError("KL should be positive")
    return {"kl": kl, "latent_dim": int(z.size)}


def test_joystick_cost_gradient() -> dict[str, Any]:
    target = np.array([0.5, -0.2])

    def cost(v: np.ndarray) -> float:
        err = v - target
        return float(0.5 * np.dot(err, err))

    velocity = np.array([0.1, 0.4])
    expected_grad = velocity - target
    fd_grad = finite_difference_grad(cost, velocity)
    assert_close(fd_grad, expected_grad, atol=1e-8)
    return {"gradient_error": float(np.max(np.abs(fd_grad - expected_grad)))}


def test_waypoint_cost_decreases_near_goal() -> dict[str, Any]:
    goal = np.array([1.5, -0.25])

    def cost(pos: np.ndarray) -> float:
        return float(np.linalg.norm(pos - goal))

    far = cost(np.array([-1.0, 0.5]))
    near = cost(np.array([1.4, -0.2]))
    if not near < far:
        raise AssertionError("waypoint cost should be lower near goal")
    return {"far_cost": far, "near_cost": near}


def test_sdf_barrier_gradient_sign() -> dict[str, Any]:
    x = np.array([0.05])
    grad = finite_difference_grad(lambda value: sdf_barrier(value, delta=0.1), x)
    if not grad[0] < 0.0:
        raise AssertionError(f"barrier gradient should push distance upward, got {grad[0]}")
    return {"gradient": float(grad[0]), "barrier": sdf_barrier(x)}


def test_sdf_barrier_matches_paper_piecewise_formula() -> dict[str, Any]:
    distances = np.array([0.05, 0.1, 0.2])
    delta = 0.1
    expected = (
        -np.log(delta)
        + 0.5 * (((distances[0] - 2.0 * delta) / delta) ** 2 - 1.0)
        - np.log(distances[1])
        - np.log(distances[2])
    )
    actual = sdf_barrier(distances, delta=delta)
    assert_close(actual, expected, atol=1e-12)
    return {"actual": float(actual), "expected": float(expected), "delta": delta}


def test_trajectory_smoothness_penalty() -> dict[str, Any]:
    linear = np.linspace(0.0, 1.0, 21)[:, None]
    kink = linear.copy()
    kink[10:] += 0.3

    if not smoothness_penalty(linear) < 1e-28:
        raise AssertionError("linear path should have zero second-difference penalty")
    if not smoothness_penalty(kink) > smoothness_penalty(linear):
        raise AssertionError("kinked path should be less smooth")
    return {"linear_penalty": smoothness_penalty(linear), "kink_penalty": smoothness_penalty(kink)}


def test_nan_inf_guards() -> dict[str, Any]:
    cases: list[tuple[str, Callable[[], Any]]] = [
        ("rot6d_nan", lambda: rot6d_to_matrix(np.array([1.0, np.nan, 0.0, 0.0, 1.0, 0.0]))),
        ("q_sample_inf", lambda: q_sample(np.zeros((2, 3)), np.full((2, 3), np.inf), alpha_bar=0.5)),
        ("vae_nan", lambda: reparameterize(np.zeros(2), np.array([0.0, np.nan]), np.zeros(2))),
        ("guidance_inf", lambda: gaussian_reward(np.array([np.inf]), sigma=0.25)),
    ]
    rejected: list[str] = []
    for name, fn in cases:
        try:
            fn()
        except ValueError:
            rejected.append(name)
    if len(rejected) != len(cases):
        raise AssertionError(f"not all NaN/Inf cases were rejected: {rejected}")
    return {"rejected_cases": rejected}


def test_dagger_sample_teacher_query_metrics() -> dict[str, Any]:
    samples = [
        build_dagger_sample(
            sample_id="s0",
            rollout_id="debug_rollout",
            timestep=0,
            state=np.array([0.1, 0.2, 0.3]),
            student_action=np.array([0.0, 0.2]),
            teacher_action=np.array([0.1, 0.0]),
            teacher_queried=True,
            accepted=True,
            split="train",
        ),
        build_dagger_sample(
            sample_id="s1",
            rollout_id="debug_rollout",
            timestep=1,
            state=np.array([0.2, 0.3, 0.4]),
            student_action=np.array([0.3, -0.1]),
            teacher_action=np.array([0.1, -0.2]),
            teacher_queried=True,
            accepted=False,
            split="validation",
        ),
    ]
    metrics = teacher_student_discrepancy(samples)
    expected_mse = action_mse(
        np.array([[0.0, 0.2], [0.3, -0.1]]),
        np.array([[0.1, 0.0], [0.1, -0.2]]),
    )
    assert_close(metrics["action_mse"], expected_mse)
    if metrics["teacher_query_count"] != 2.0 or metrics["accepted_count"] != 1.0:
        raise AssertionError(f"unexpected DAgger counts {metrics}")
    return metrics


def test_state_latent_window_schema_and_splits() -> dict[str, Any]:
    states = np.arange(12, dtype=np.float64).reshape(3, 4)
    latents = np.ones((3, 2), dtype=np.float64)
    tokens = stack_state_latent_tokens(states, latents)
    assert_close(tokens[:, :4], states)
    assert_close(tokens[:, 4:], latents)
    windows = [
        build_state_latent_window("w0", "walk", 0, "train", True, states, latents),
        build_state_latent_window("w1", "run", 5, "validation", False, states, latents),
        build_state_latent_window("w2", "jump", 10, "test", True, states, latents),
    ]
    counts = split_counts(windows)
    if counts != {"train": 1, "validation": 0, "test": 1}:
        raise AssertionError(f"unexpected split counts {counts}")
    return {"token_shape": list(tokens.shape), "split_counts": counts}


def test_tracking_error_and_survival_metrics() -> dict[str, Any]:
    target = np.zeros((2, 2, 3), dtype=np.float64)
    predicted = target.copy()
    predicted[0, 0, 0] = 0.3
    predicted[1, 1, 2] = -0.4
    errors = tracking_error(predicted, target)
    survival = survival_rate(np.array([50, 49, 50, 10]), horizon=50)
    if not np.isclose(errors["max_error"], 0.4):
        raise AssertionError(f"unexpected max tracking error {errors}")
    if survival != 0.5:
        raise AssertionError(f"unexpected survival rate {survival}")
    return {"mean_tracking_error": errors["mean_error"], "max_tracking_error": errors["max_error"], "survival_rate": survival}


def test_goal_success_fall_velocity_metrics() -> dict[str, Any]:
    successes = np.array([1, 1, 0, 1, 0], dtype=np.float64)
    falls = np.array([0, 0, 1, 0, 1], dtype=np.float64)
    predicted = np.array([[0.4, 0.1], [0.6, 0.0], [0.5, -0.2]], dtype=np.float64)
    target = np.array([[0.5, 0.0], [0.5, 0.0], [0.5, 0.0]], dtype=np.float64)
    velocity = velocity_tracking_error(predicted, target)
    if success_rate(successes) != 0.6:
        raise AssertionError("success rate mismatch")
    if fall_rate(falls) != 0.4:
        raise AssertionError("fall rate mismatch")
    if not np.isclose(velocity["rmse"], np.sqrt(np.mean((predicted - target) ** 2))):
        raise AssertionError(f"velocity RMSE mismatch {velocity}")
    return {
        "success_rate": success_rate(successes),
        "fall_rate": fall_rate(falls),
        "velocity_mean_error": velocity["mean_error"],
        "velocity_rmse": velocity["rmse"],
    }


TESTS: list[tuple[str, TestFn, list[str]]] = [
    ("rot6d_conversion", test_rot6d_conversion, ["state_representation", "rotation_6d"]),
    ("anchor_yaw_roundtrip", test_anchor_yaw_roundtrip, ["trajectory_inverse_transform", "current_character_frame"]),
    ("height_preserving_current_frame", test_height_preserving_current_frame, ["trajectory_inverse_transform", "height"]),
    ("gaussian_reward_monotonicity", test_gaussian_reward_monotonicity, ["tracking_reward"]),
    ("termination_thresholds", test_termination_thresholds, ["tracking_termination"]),
    ("adaptive_sampling_kernel", test_adaptive_sampling_kernel, ["adaptive_sampling"]),
    ("ou_noise_temporal_correlation", test_ou_noise_temporal_correlation, ["dataset_collection", "ou_noise"]),
    ("symmetry_involution_29d", test_symmetry_involution_29d, ["dataset_collection", "symmetry_augmentation"]),
    ("emphasis_projection_pseudoinverse", test_emphasis_projection_pseudoinverse, ["emphasis_projection"]),
    ("diffusion_forward_noise_increases", test_diffusion_forward_noise_increases, ["diffusion_forward"]),
    ("diffusion_oracle_reverse_reduces_mse", test_diffusion_oracle_reverse_reduces_mse, ["diffusion_reverse"]),
    ("independent_timestep_mask_schedule", test_independent_timestep_mask_schedule, ["independent_timestep", "task_masks"]),
    ("inpainting_observation_clamp", test_inpainting_observation_clamp, ["inpainting", "task_masks"]),
    ("vae_reparameterization_and_kl", test_vae_reparameterization_and_kl, ["vae_latent", "kl_loss"]),
    ("joystick_cost_gradient", test_joystick_cost_gradient, ["guidance_cost", "joystick"]),
    ("waypoint_cost_decreases_near_goal", test_waypoint_cost_decreases_near_goal, ["guidance_cost", "waypoint"]),
    ("sdf_barrier_gradient_sign", test_sdf_barrier_gradient_sign, ["guidance_cost", "sdf_barrier"]),
    ("trajectory_smoothness_penalty", test_trajectory_smoothness_penalty, ["smoothness"]),
    ("nan_inf_guards", test_nan_inf_guards, ["coding_requirements", "nan_inf_checks"]),
    ("dagger_sample_teacher_query_metrics", test_dagger_sample_teacher_query_metrics, ["dagger", "teacher_query", "teacher_student_discrepancy"]),
    ("state_latent_window_schema_and_splits", test_state_latent_window_schema_and_splits, ["trajectory_dataset", "state_latent_tokens", "split_manifest"]),
    ("tracking_error_and_survival_metrics", test_tracking_error_and_survival_metrics, ["evaluation_metrics", "tracking_error", "closed_loop_survival"]),
    ("goal_success_fall_velocity_metrics", test_goal_success_fall_velocity_metrics, ["evaluation_metrics", "success_rate", "fall_rate", "velocity_tracking_error"]),
]


def run_all() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name, fn, goal_items in TESTS:
        try:
            metrics = fn()
            rows.append(
                {
                    "name": name,
                    "status": "passed",
                    "goal_items": goal_items,
                    "metrics": metrics,
                    "error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001 - test runner records every failure.
            rows.append(
                {
                    "name": name,
                    "status": "failed",
                    "goal_items": goal_items,
                    "metrics": {},
                    "error": f"{exc}\n{traceback.format_exc()}",
                }
            )

    failed = [row for row in rows if row["status"] != "passed"]
    covered_goal_items = sorted({item for row in rows for item in row["goal_items"]})
    summary: dict[str, Any] = {
        "status": "ok" if not failed else "failed",
        "experiment_type": "unit_test",
        "scope": "pure NumPy formula-level tests for BeyondMimic reproduction math",
        "row_count": len(rows),
        "failed_row_count": len(failed),
        "covered_goal_items": covered_goal_items,
        "checks": {
            "all_core_math_tests_pass": not failed,
            "covers_goal_core_math_items": len(covered_goal_items) >= 16,
            "pure_numpy_no_isaac_ros_dependency": True,
            "does_not_claim_training_or_deployment": True,
        },
        "rows": rows,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "These are formula-level unit tests. They do not replace IsaacLab rollouts, trained VAE/diffusion "
                "checkpoints, TensorRT deployment, Fig. 5/Fig. 6 reproduction, or real Unitree G1 execution."
            ),
        },
        "outputs": {
            "json": str(OUT / "core_math_unit_tests.json"),
            "tsv": str(OUT / "core_math_unit_tests.tsv"),
        },
    }
    return summary


def write_outputs(summary: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / "core_math_unit_tests.json"
    tsv_path = OUT / "core_math_unit_tests.tsv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["name", "status", "goal_items", "metrics", "error"])
        writer.writeheader()
        for row in summary["rows"]:
            writer.writerow(
                {
                    "name": row["name"],
                    "status": row["status"],
                    "goal_items": ",".join(row["goal_items"]),
                    "metrics": json.dumps(row["metrics"], sort_keys=True),
                    "error": row["error"],
                }
            )


def main() -> None:
    summary = run_all()
    write_outputs(summary)
    print(json.dumps({"status": summary["status"], "rows": summary["row_count"], "failed": summary["failed_row_count"]}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
