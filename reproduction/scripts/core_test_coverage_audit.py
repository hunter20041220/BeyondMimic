#!/usr/bin/env python3
"""Audit coverage of the explicit goal.md "at least test" core-math checklist."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
CORE_TESTS = ROOT / "res/tests/core_math_unit_tests/core_math_unit_tests.json"
OUT = ROOT / "res/tests/core_test_coverage_audit"

REQUIRED = [
    ("rot6d_conversion", ["rot6d_conversion"], ["rotation_6d"]),
    ("anchor_transform", ["anchor_yaw_roundtrip"], ["trajectory_inverse_transform", "current_character_frame"]),
    ("yaw_alignment", ["anchor_yaw_roundtrip"], ["current_character_frame"]),
    ("height_preserving_transform", ["height_preserving_current_frame"], ["height"]),
    ("reward_components", ["gaussian_reward_monotonicity"], ["tracking_reward"]),
    ("termination", ["termination_thresholds"], ["tracking_termination"]),
    ("adaptive_sampling", ["adaptive_sampling_kernel"], ["adaptive_sampling"]),
    ("ou_noise", ["ou_noise_temporal_correlation"], ["ou_noise"]),
    ("symmetry_augmentation", ["symmetry_involution_29d"], ["symmetry_augmentation"]),
    ("trajectory_coordinate_transform", ["anchor_yaw_roundtrip", "height_preserving_current_frame"], ["trajectory_inverse_transform"]),
    ("emphasis_projection", ["emphasis_projection_pseudoinverse"], ["emphasis_projection"]),
    ("pseudoinverse_reconstruction", ["emphasis_projection_pseudoinverse"], ["emphasis_projection"]),
    ("diffusion_forward_process", ["diffusion_forward_noise_increases"], ["diffusion_forward"]),
    ("diffusion_reverse_process", ["diffusion_oracle_reverse_reduces_mse"], ["diffusion_reverse"]),
    ("independent_timestep_mask", ["independent_timestep_mask_schedule"], ["independent_timestep", "task_masks"]),
    ("vae_reparameterization", ["vae_reparameterization_and_kl"], ["vae_latent"]),
    ("joystick_cost", ["joystick_cost_gradient"], ["joystick"]),
    ("waypoint_cost", ["waypoint_cost_decreases_near_goal"], ["waypoint"]),
    ("sdf_cost", ["sdf_barrier_gradient_sign"], ["sdf_barrier"]),
    ("inpainting_mask", ["inpainting_observation_clamp"], ["inpainting", "task_masks"]),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    core = json.loads(CORE_TESTS.read_text(encoding="utf-8"))
    passed_tests = {row["name"]: row for row in core["rows"] if row["status"] == "passed"}
    covered_items = set(core["covered_goal_items"])
    rows: list[dict[str, Any]] = []
    for requirement, test_names, goal_items in REQUIRED:
        tests_present = [name in passed_tests for name in test_names]
        goals_present = [item in covered_items for item in goal_items]
        rows.append(
            {
                "requirement": requirement,
                "test_names": test_names,
                "goal_items": goal_items,
                "tests_present": tests_present,
                "goal_items_present": goals_present,
                "passed": all(tests_present) and all(goals_present),
                "evidence": str(CORE_TESTS),
            }
        )
    missing = [row for row in rows if not row["passed"]]
    goal_metric_items = {"success_rate", "fall_rate", "velocity_tracking_error", "evaluation_metrics"}
    summary = {
        "status": "ok" if core["status"] == "ok" and not missing else "failed",
        "experiment_type": "coverage_audit",
        "scope": "explicit goal.md core-math 'at least test' checklist coverage",
        "required_count": len(rows),
        "missing_count": len(missing),
        "core_test_row_count": core["row_count"],
        "core_test_failed_row_count": core["failed_row_count"],
        "covered_goal_item_count": len(covered_items),
        "goal_metric_items": sorted(goal_metric_items),
        "rows": rows,
        "missing_rows": missing,
        "checks": {
            "core_math_tests_pass": core["status"] == "ok" and core["failed_row_count"] == 0,
            "all_20_required_items_have_test_evidence": not missing,
            "goal_metric_items_have_test_evidence": goal_metric_items.issubset(covered_items),
            "does_not_claim_training_or_deployment": core["checks"]["does_not_claim_training_or_deployment"],
            "pure_numpy_no_isaac_ros_dependency": core["checks"]["pure_numpy_no_isaac_ros_dependency"],
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The explicit formula/test checklist is covered by pure NumPy tests, but this does not replace "
                "IsaacLab rollouts, trained checkpoints, deployment, paper Fig. 5/Fig. 6 reproduction, or real robot "
                "execution."
            ),
        },
        "outputs": {
            "json": str(OUT / "core_test_coverage_audit.json"),
            "tsv": str(OUT / "core_test_coverage_audit.tsv"),
        },
    }
    (OUT / "core_test_coverage_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT / "core_test_coverage_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=["requirement", "test_names", "goal_items", "tests_present", "goal_items_present", "passed", "evidence"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "requirement": row["requirement"],
                    "test_names": ",".join(row["test_names"]),
                    "goal_items": ",".join(row["goal_items"]),
                    "tests_present": json.dumps(row["tests_present"]),
                    "goal_items_present": json.dumps(row["goal_items_present"]),
                    "passed": row["passed"],
                    "evidence": row["evidence"],
                }
            )
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
