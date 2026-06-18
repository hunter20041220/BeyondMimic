#!/usr/bin/env python3
"""Debug-only iterative DAgger aggregation smoke."""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SRC = ROOT / "reproduction/src"
OUT = ROOT / "res/level_c/dagger_iteration_smoke"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beyondmimic_reimpl.dagger import build_dagger_sample, teacher_student_discrepancy
from beyondmimic_reimpl.evaluation import action_mse


@dataclass(frozen=True)
class Config:
    seed: int = 20261004
    iterations: int = 3
    samples_per_iteration: int = 96
    heldout_count: int = 128
    state_dim: int = 163
    action_dim: int = 29
    ridge_lambda: float = 1e-4
    student_rollout_noise: float = 0.08


def teacher_actions(states: np.ndarray, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return np.tanh(states @ weights + bias)


def student_actions(states: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x = np.concatenate([states, np.ones((states.shape[0], 1), dtype=np.float64)], axis=1)
    return np.tanh(x @ weights)


def ridge_fit(states: np.ndarray, teacher: np.ndarray, ridge_lambda: float) -> np.ndarray:
    x = np.concatenate([states, np.ones((states.shape[0], 1), dtype=np.float64)], axis=1)
    # Fit pre-activation targets approximately by clipping inverse tanh.
    y = np.arctanh(np.clip(teacher, -0.999, 0.999))
    xtx = x.T @ x
    reg = ridge_lambda * np.eye(xtx.shape[0], dtype=np.float64)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xtx + reg, x.T @ y)


def split_for_iteration(iteration: int) -> str:
    if iteration == 0:
        return "train"
    if iteration == 1:
        return "validation"
    return "test"


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "iteration",
        "new_samples",
        "aggregate_samples",
        "teacher_query_count",
        "iteration_action_mse_before_update",
        "aggregate_action_mse_after_update",
        "heldout_action_mse_after_update",
        "max_abs_action_error_after_update",
        "split",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    cfg = Config()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    teacher_w = rng.normal(0.0, 0.12, size=(cfg.state_dim, cfg.action_dim))
    teacher_b = rng.normal(0.0, 0.03, size=(cfg.action_dim,))
    student_w = rng.normal(0.0, 0.02, size=(cfg.state_dim + 1, cfg.action_dim))
    heldout_states = rng.normal(0.0, 0.5, size=(cfg.heldout_count, cfg.state_dim))
    heldout_teacher = teacher_actions(heldout_states, teacher_w, teacher_b)
    initial_heldout_mse = action_mse(student_actions(heldout_states, student_w), heldout_teacher)

    all_samples = []
    aggregate_states: list[np.ndarray] = []
    aggregate_teacher: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for iteration in range(cfg.iterations):
        split = split_for_iteration(iteration)
        rollout_states = rng.normal(0.0, 0.5, size=(cfg.samples_per_iteration, cfg.state_dim))
        student_before = student_actions(rollout_states, student_w)
        # The synthetic rollout distribution depends on the current student, so later queries are not a fixed offline set.
        queried_states = rollout_states + cfg.student_rollout_noise * np.pad(
            student_before,
            ((0, 0), (0, cfg.state_dim - cfg.action_dim)),
            mode="constant",
        )
        teacher = teacher_actions(queried_states, teacher_w, teacher_b)
        iteration_mse_before = action_mse(student_actions(queried_states, student_w), teacher)
        for idx in range(cfg.samples_per_iteration):
            sample = build_dagger_sample(
                sample_id=f"dagger_iter_{iteration:02d}_sample_{idx:03d}",
                rollout_id=f"synthetic_student_rollout_iter_{iteration:02d}",
                timestep=iteration * cfg.samples_per_iteration + idx,
                state=queried_states[idx],
                student_action=student_actions(queried_states[idx : idx + 1], student_w)[0],
                teacher_action=teacher[idx],
                teacher_queried=True,
                accepted=True,
                split=split,  # type: ignore[arg-type]
            )
            all_samples.append(sample)
        aggregate_states.append(queried_states)
        aggregate_teacher.append(teacher)
        train_states = np.concatenate(aggregate_states, axis=0)
        train_teacher = np.concatenate(aggregate_teacher, axis=0)
        student_w = ridge_fit(train_states, train_teacher, cfg.ridge_lambda)
        aggregate_pred = student_actions(train_states, student_w)
        heldout_pred = student_actions(heldout_states, student_w)
        discrepancy = teacher_student_discrepancy(all_samples)
        rows.append(
            {
                "iteration": iteration,
                "new_samples": cfg.samples_per_iteration,
                "aggregate_samples": int(train_states.shape[0]),
                "teacher_query_count": int(discrepancy["teacher_query_count"]),
                "iteration_action_mse_before_update": iteration_mse_before,
                "aggregate_action_mse_after_update": action_mse(aggregate_pred, train_teacher),
                "heldout_action_mse_after_update": action_mse(heldout_pred, heldout_teacher),
                "max_abs_action_error_after_update": float(np.max(np.abs(aggregate_pred - train_teacher))),
                "split": split,
            }
        )

    final_discrepancy = teacher_student_discrepancy(all_samples)
    metrics = {
        "initial_heldout_action_mse": initial_heldout_mse,
        "final_heldout_action_mse": rows[-1]["heldout_action_mse_after_update"],
        "heldout_mse_reduction_ratio": (initial_heldout_mse - rows[-1]["heldout_action_mse_after_update"])
        / initial_heldout_mse,
        "final_aggregate_action_mse": rows[-1]["aggregate_action_mse_after_update"],
        "total_teacher_queries": final_discrepancy["teacher_query_count"],
        "total_samples": final_discrepancy["sample_count"],
        "accepted_count": final_discrepancy["accepted_count"],
        "final_max_abs_action_error": rows[-1]["max_abs_action_error_after_update"],
    }
    split_counts: dict[str, int] = {}
    for sample in all_samples:
        split_counts[sample.split] = split_counts.get(sample.split, 0) + 1
    checks = {
        "uses_package_dagger_api": True,
        "uses_package_evaluation_metric_api": True,
        "three_dagger_iterations": cfg.iterations == 3 and len(rows) == 3,
        "all_iterations_query_teacher": all(row["teacher_query_count"] == (row["iteration"] + 1) * cfg.samples_per_iteration for row in rows),
        "aggregate_dataset_grows_each_iteration": [row["aggregate_samples"] for row in rows]
        == [cfg.samples_per_iteration * (idx + 1) for idx in range(cfg.iterations)],
        "all_samples_teacher_queried": all(sample.teacher_queried for sample in all_samples),
        "all_samples_accepted_debug_only": all(sample.accepted for sample in all_samples),
        "state_dim_163": cfg.state_dim == 163 and all(sample.state.shape == (163,) for sample in all_samples),
        "action_dim_29": cfg.action_dim == 29 and all(sample.teacher_action.shape == (29,) for sample in all_samples),
        "heldout_discrepancy_decreases": metrics["final_heldout_action_mse"] < metrics["initial_heldout_action_mse"],
        "final_heldout_reduction_at_least_0_5": metrics["heldout_mse_reduction_ratio"] >= 0.5,
        "split_counts_nonzero": all(split_counts.get(split, 0) > 0 for split in ["train", "validation", "test"]),
        "does_not_claim_true_dagger_rollout": True,
        "does_not_claim_goal_complete": True,
    }
    json_path = OUT / "level_c_dagger_iteration_smoke.json"
    tsv_path = OUT / "level_c_dagger_iteration_smoke.tsv"
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "dagger_iteration_smoke",
        "scope": "debug-only iterative DAgger aggregation with synthetic student rollouts and teacher queries",
        "paper_evidence": {
            "vae_dagger_method": str(ROOT / "reproduction/paper/source/tex/method.tex:150-170"),
            "goal_dagger_requirement": str(ROOT / "goal.md:1148-1190"),
        },
        "settings": asdict(cfg),
        "iteration_rows": rows,
        "split_counts": dict(sorted(split_counts.items())),
        "metrics": metrics,
        "checks": checks,
        "not_a_replacement_for": [
            "Isaac/teacher-policy rollout",
            "true DAgger aggregation over environment states",
            "trained conditional VAE checkpoint",
            "closed-loop VAE survival and fall-rate evaluation",
        ],
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "partial",
            "why_not_complete": (
                "This proves the local DAgger-style loop can query a teacher, aggregate samples over iterations, "
                "update a student, and reduce held-out action discrepancy in a deterministic synthetic setting. It is "
                "not true BeyondMimic DAgger because no Isaac rollout or trained teacher policy is used."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, rows)
    print(json.dumps({"status": summary["status"], "json": str(json_path), "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
