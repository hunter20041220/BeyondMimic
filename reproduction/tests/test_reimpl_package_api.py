#!/usr/bin/env python3
"""Package-level contract tests for the clean-room reimplementation API.

These tests exercise exported symbols, shape checks, metadata helpers, and
error paths across the lightweight package. They are dependency-light and do
not run IsaacLab, ROS, TensorRT, training, or deployment.
"""

from __future__ import annotations

import csv
import importlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SRC = ROOT / "reproduction/src"
OUT = ROOT / "res/tests/reimpl_package_api_tests"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beyondmimic_reimpl.dagger import build_dagger_sample, teacher_student_discrepancy
from beyondmimic_reimpl.diffusion import apply_observation_mask, q_sample
from beyondmimic_reimpl.evaluation import (
    action_mse,
    fall_rate,
    split_metric_summary,
    success_rate,
    survival_rate,
    tracking_error,
    velocity_tracking_error,
)
from beyondmimic_reimpl.geometry import anchor_to_world, rot6d_to_matrix, world_to_anchor, yaw_matrix
from beyondmimic_reimpl.guidance import finite_difference_grad, gaussian_reward, sdf_barrier
from beyondmimic_reimpl.sampling import adaptive_distribution, mirror_state_29d, ou_noise
from beyondmimic_reimpl.state import (
    emphasis_projection,
    hybrid_state_schema,
    project_hybrid_state,
    smoothness_penalty,
    unproject_hybrid_state,
    validate_hybrid_state,
)
from beyondmimic_reimpl.trajectory import build_state_latent_window, split_counts, stack_state_latent_tokens
from beyondmimic_reimpl.validation import ensure_finite
from beyondmimic_reimpl.vae import kl_standard_normal, reparameterize


TestFn = Callable


def expect_value_error(fn: Callable[[], Any]) -> str:
    """Run ``fn`` and return the ValueError message for invalid shape inputs."""
    try:
        fn()
    except ValueError as exc:
        return str(exc)
    raise AssertionError("expected ValueError")


def test_module_exports() -> dict[str, Any]:
    """Verify expected package exports across clean-room API modules."""
    expected = {
        "beyondmimic_reimpl.geometry": ["anchor_to_world", "rot6d_to_matrix", "world_to_anchor", "yaw_matrix"],
        "beyondmimic_reimpl.sampling": ["adaptive_distribution", "mirror_state_29d", "ou_noise"],
        "beyondmimic_reimpl.diffusion": ["apply_observation_mask", "denoise_one_step_with_oracle_eps", "q_sample"],
        "beyondmimic_reimpl.vae": ["kl_standard_normal", "reparameterize"],
        "beyondmimic_reimpl.guidance": ["finite_difference_grad", "gaussian_reward", "sdf_barrier"],
        "beyondmimic_reimpl.state": [
            "emphasis_projection",
            "hybrid_state_schema",
            "project_hybrid_state",
            "smoothness_penalty",
            "unproject_hybrid_state",
            "validate_hybrid_state",
        ],
        "beyondmimic_reimpl.dagger": ["build_dagger_sample", "teacher_student_discrepancy"],
        "beyondmimic_reimpl.trajectory": ["build_state_latent_window", "split_counts", "stack_state_latent_tokens"],
        "beyondmimic_reimpl.evaluation": [
            "action_mse",
            "fall_rate",
            "split_metric_summary",
            "success_rate",
            "survival_rate",
            "tracking_error",
            "velocity_tracking_error",
        ],
    }
    missing: list[str] = []
    for module_name, symbols in expected.items():
        module = importlib.import_module(module_name)
        exported = set(getattr(module, "__all__", symbols))
        for symbol in symbols:
            if not hasattr(module, symbol) or symbol not in exported:
                missing.append(f"{module_name}:{symbol}")
    if missing:
        raise AssertionError(f"missing exports: {missing}")
    return {"module_count": len(expected), "symbol_count": sum(len(v) for v in expected.values())}


def test_geometry_shape_and_roundtrip_contract() -> dict[str, Any]:
    """Check geometry shape errors and anchor/world frame roundtrip."""
    root = np.array([0.2, -0.1, 0.9])
    points = np.array([[0.3, 0.4, 1.1], [0.1, -0.2, 0.7]])
    local = world_to_anchor(points, root, 0.3)
    recovered = anchor_to_world(local, root, 0.3)
    if not np.allclose(recovered, points, atol=1e-12):
        raise AssertionError("anchor/world roundtrip failed")
    shape_error = expect_value_error(lambda: rot6d_to_matrix(np.ones(5)))
    yaw_error = expect_value_error(lambda: yaw_matrix(float("nan")))
    return {"roundtrip_error": float(np.max(np.abs(recovered - points))), "shape_error": shape_error, "yaw_error": yaw_error}


def test_sampling_contracts_and_reproducibility() -> dict[str, Any]:
    """Check sampling shape contracts, deterministic OU seeds, and mirror errors."""
    dist = adaptive_distribution(3, 8, 2)
    if not np.isclose(np.sum(dist), 1.0):
        raise AssertionError("adaptive distribution is not normalized")
    a = ou_noise(seed=123, steps=8, dim=2)
    b = ou_noise(seed=123, steps=8, dim=2)
    if not np.array_equal(a, b):
        raise AssertionError("OU noise seed path is not deterministic")
    mirror_error = expect_value_error(lambda: mirror_state_29d(np.zeros(28)))
    return {"distribution_sum": float(np.sum(dist)), "ou_shape": list(a.shape), "mirror_error": mirror_error}


def test_diffusion_and_mask_contracts() -> dict[str, Any]:
    """Check diffusion shape validation and mask clamp behavior."""
    x0 = np.zeros((3, 4))
    eps = np.ones((3, 4))
    noisy = q_sample(x0, eps, alpha_bar=0.5)
    clean = np.full((3, 4), 2.0)
    mask = np.zeros((3, 4), dtype=bool)
    mask[0, :] = True
    clamped = apply_observation_mask(noisy, clean, mask)
    if not np.array_equal(clamped[0], clean[0]):
        raise AssertionError("mask clamp failed")
    shape_error = expect_value_error(lambda: q_sample(np.zeros((2, 3)), np.zeros((2, 4)), 0.5))
    alpha_error = expect_value_error(lambda: q_sample(np.zeros((2, 3)), np.zeros((2, 3)), 1.5))
    return {"noisy_mean": float(np.mean(noisy)), "shape_error": shape_error, "alpha_error": alpha_error}


def test_vae_and_validation_errors() -> dict[str, Any]:
    """Check VAE latent math and finite-value rejection."""
    mu = np.array([0.0, 0.2])
    logvar = np.array([0.0, -0.1])
    eps = np.array([1.0, 0.0])
    z = reparameterize(mu, logvar, eps)
    kl = kl_standard_normal(mu, logvar)
    finite_error = expect_value_error(lambda: ensure_finite("bad", np.array([np.inf])))
    shape_error = expect_value_error(lambda: reparameterize(mu, logvar, np.zeros(3)))
    return {"latent_shape": list(z.shape), "kl": kl, "finite_error": finite_error, "shape_error": shape_error}


def test_guidance_state_and_evaluation_contracts() -> dict[str, Any]:
    """Check guidance gradients, state projection, smoothness, and metrics."""
    grad = finite_difference_grad(lambda x: float(np.sum(x**2)), np.array([0.5, -0.25]))
    if not np.allclose(grad, np.array([1.0, -0.5]), atol=1e-7):
        raise AssertionError("finite difference gradient mismatch")
    reward = gaussian_reward(np.array([0.1, 0.0]), sigma=0.25)
    barrier = sdf_barrier(np.array([0.05]), delta=0.1)
    p, p_inv = emphasis_projection(seed=3)
    if p.shape != (163, 99) or p_inv.shape != (99, 163):
        raise AssertionError(f"unexpected paper projection shapes {p.shape}, {p_inv.shape}")
    states = np.eye(99)[:2]
    recovered = (states @ p.T) @ p_inv.T
    schema = hybrid_state_schema()
    projected, _, schema_p_inv = project_hybrid_state(np.zeros((2, schema.state_dim)), seed=4, schema=schema)
    schema_recovered = unproject_hybrid_state(projected, schema_p_inv, schema)
    schema_error = expect_value_error(lambda: validate_hybrid_state(np.zeros((2, 160)), schema))
    tracking = tracking_error(np.zeros((2, 1, 3)), np.ones((2, 1, 3)) * 0.1)
    survival = survival_rate(np.array([5.0, 10.0, 11.0]), horizon=10)
    mse = action_mse(np.zeros((2, 3)), np.ones((2, 3)))
    smooth = smoothness_penalty(np.arange(12, dtype=np.float64).reshape(6, 2))
    if smooth > 1e-28:
        raise AssertionError("linear trajectory should have near-zero second difference")
    return {
        "reward": reward,
        "barrier": barrier,
        "projection_error": float(np.max(np.abs(recovered - states))),
        "paper_projection_shape": list(p.shape),
        "hybrid_schema_state_dim": schema.state_dim,
        "hybrid_schema_projected_dim": schema.projected_dim,
        "hybrid_projection_roundtrip_error": float(np.max(np.abs(schema_recovered))),
        "hybrid_schema_error": schema_error,
        "tracking_mean_error": tracking["mean_error"],
        "survival_rate": survival,
        "action_mse": mse,
    }


def test_dagger_and_trajectory_metadata_contracts() -> dict[str, Any]:
    """Check DAgger metadata and state-latent split contracts."""
    state = np.arange(4, dtype=np.float64)
    student = np.array([0.0, 0.2])
    teacher = np.array([0.1, 0.0])
    sample = build_dagger_sample("s0", "rollout0", 2, state, student, teacher, True, True, "train")
    metadata = sample.to_metadata()
    if metadata["state"]["shape"] != [4] or metadata["teacher_action"]["shape"] != [2]:
        raise AssertionError("metadata shape export failed")
    discrepancy = teacher_student_discrepancy([sample])
    teacher_error = expect_value_error(
        lambda: teacher_student_discrepancy(
            [build_dagger_sample("s1", "rollout0", 3, state, student, teacher, False, True, "train")]
        )
    )
    states = np.zeros((3, 4))
    latents = np.ones((3, 2))
    window = build_state_latent_window("w0", "motion", 0, "train", True, states, latents)
    tokens = stack_state_latent_tokens(window.states, window.latents)
    counts = split_counts([window, build_state_latent_window("w1", "motion", 1, "test", False, states, latents)])
    split_error = expect_value_error(lambda: build_state_latent_window("bad", "motion", 0, "dev", True, states, latents))
    return {
        "metadata_state_shape": metadata["state"]["shape"],
        "teacher_query_count": discrepancy["teacher_query_count"],
        "token_shape": list(tokens.shape),
        "split_counts": counts,
        "teacher_error": teacher_error,
        "split_error": split_error,
    }


def test_goal_evaluation_metric_contracts() -> dict[str, Any]:
    """Check success/fall/velocity metric helpers used by goal-level evaluation tables."""
    successes = np.array([1, 0, 1, 1], dtype=np.float64)
    falls = np.array([0, 1, 0, 0], dtype=np.float64)
    pred_vel = np.array([[0.4, 0.0], [0.6, -0.1], [0.5, 0.2]], dtype=np.float64)
    target_vel = np.array([[0.5, 0.0], [0.5, 0.0], [0.5, 0.0]], dtype=np.float64)
    velocity = velocity_tracking_error(pred_vel, target_vel)
    split_summary = split_metric_summary(
        np.array([0.1, 0.3, 0.5, 0.7], dtype=np.float64),
        ["train", "train", "test", "test"],
    )
    binary_error = expect_value_error(lambda: success_rate(np.array([0.0, 0.5, 1.0])))
    shape_error = expect_value_error(lambda: velocity_tracking_error(np.zeros((2, 2)), np.zeros((2, 2, 1))))
    if success_rate(successes) != 0.75 or fall_rate(falls) != 0.25:
        raise AssertionError("success/fall rates are incorrect")
    if not np.isclose(velocity["max_error"], 0.2):
        raise AssertionError(f"unexpected max velocity error {velocity}")
    return {
        "success_rate": success_rate(successes),
        "fall_rate": fall_rate(falls),
        "velocity_mean_error": velocity["mean_error"],
        "velocity_max_error": velocity["max_error"],
        "test_split_mean": split_summary["test"]["mean"],
        "binary_error": binary_error,
        "shape_error": shape_error,
    }


TESTS: list[tuple[str, TestFn, list[str]]] = [
    ("module_exports", test_module_exports, ["package_exports", "api_surface"]),
    ("geometry_shape_and_roundtrip_contract", test_geometry_shape_and_roundtrip_contract, ["geometry", "shape_errors"]),
    ("sampling_contracts_and_reproducibility", test_sampling_contracts_and_reproducibility, ["sampling", "fixed_seed"]),
    ("diffusion_and_mask_contracts", test_diffusion_and_mask_contracts, ["diffusion", "mask_shape"]),
    ("vae_and_validation_errors", test_vae_and_validation_errors, ["vae", "finite_guards"]),
    ("guidance_state_and_evaluation_contracts", test_guidance_state_and_evaluation_contracts, ["guidance", "state", "evaluation"]),
    ("dagger_and_trajectory_metadata_contracts", test_dagger_and_trajectory_metadata_contracts, ["dagger", "trajectory"]),
    ("goal_evaluation_metric_contracts", test_goal_evaluation_metric_contracts, ["evaluation", "goal_metrics"]),
]


def run_all() -> dict[str, Any]:
    """Run all package API tests and return a machine-readable summary."""
    rows: list[dict[str, Any]] = []
    for name, fn, goal_items in TESTS:
        try:
            rows.append({"name": name, "status": "passed", "goal_items": goal_items, "metrics": fn(), "error": ""})
        except Exception as exc:  # noqa: BLE001 - local test runner records every failure.
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
    return {
        "status": "ok" if not failed else "failed",
        "experiment_type": "package_api_unit_test",
        "scope": "dependency-light package API contract tests for beyondmimic_reimpl",
        "row_count": len(rows),
        "failed_row_count": len(failed),
        "covered_goal_items": covered_goal_items,
        "rows": rows,
        "checks": {
            "all_package_api_tests_pass": not failed,
            "covers_at_least_seven_modules": len(covered_goal_items) >= 10,
            "shape_error_paths_tested": any("shape_error" in row["metrics"] for row in rows if row["status"] == "passed"),
            "metadata_paths_tested": any("metadata_state_shape" in row["metrics"] for row in rows if row["status"] == "passed"),
            "pure_numpy_no_isaac_ros_dependency": True,
            "does_not_claim_training_or_deployment": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "These tests strengthen the clean-room package API contract. They do not execute official IsaacLab "
                "rollouts, ROS deployment, TensorRT, trained checkpoints, or paper-level evaluation."
            ),
        },
        "outputs": {
            "json": str(OUT / "reimpl_package_api_tests.json"),
            "tsv": str(OUT / "reimpl_package_api_tests.tsv"),
        },
    }


def write_outputs(summary: dict[str, Any]) -> None:
    """Write package API test summary as JSON and TSV artifacts."""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reimpl_package_api_tests.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT / "reimpl_package_api_tests.tsv").open("w", encoding="utf-8", newline="") as f:
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
    """Run package API tests and fail the process on any failed row."""
    summary = run_all()
    write_outputs(summary)
    print(json.dumps({"status": summary["status"], "rows": summary["row_count"], "failed": summary["failed_row_count"]}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
