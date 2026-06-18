#!/usr/bin/env python3
"""Audit DAgger sample schema and discrepancy metrics via package APIs."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SRC = ROOT / "reproduction/src"
OUT = ROOT / "res/level_c/dagger_schema_audit"
VAE_ACCUMULATION_JSON = ROOT / "res/level_c/vae_accumulation_probe/level_c_vae_accumulation_probe.json"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beyondmimic_reimpl.dagger import build_dagger_sample, teacher_student_discrepancy
from beyondmimic_reimpl.evaluation import action_mse


def split_for_index(index: int) -> str:
    if index < 20:
        return "train"
    if index < 25:
        return "validation"
    return "test"


def build_synthetic_samples(settings: dict[str, Any], sample_count: int) -> tuple[list[Any], list[dict[str, Any]]]:
    state_dim = int(settings["proprioception_dim"] + settings["reference_motion_dim"] + settings["anchor_error_dim"])
    action_dim = int(settings["num_joints"])
    rng = np.random.default_rng(int(settings["seed"]) + 17)
    samples = []
    rows: list[dict[str, Any]] = []
    for idx in range(sample_count):
        micro_step = idx // int(settings["micro_batch_size"]) + 1
        micro_index = idx % int(settings["micro_batch_size"])
        state = rng.normal(loc=0.0, scale=0.35, size=state_dim)
        teacher_action = np.tanh(state[:action_dim] * 0.2 + rng.normal(0.0, 0.05, size=action_dim))
        student_action = teacher_action + rng.normal(0.0, 0.035, size=action_dim)
        split = split_for_index(idx)
        sample = build_dagger_sample(
            sample_id=f"synthetic_dagger_schema_{idx:03d}",
            rollout_id="synthetic_dagger_schema_debug_only",
            timestep=idx,
            state=state,
            student_action=student_action,
            teacher_action=teacher_action,
            teacher_queried=True,
            accepted=True,
            split=split,  # type: ignore[arg-type]
        )
        samples.append(sample)
        diff = sample.student_action - sample.teacher_action
        rows.append(
            {
                "sample_id": sample.sample_id,
                "rollout_id": sample.rollout_id,
                "timestep": sample.timestep,
                "micro_step": micro_step,
                "micro_batch_index": micro_index,
                "split": sample.split,
                "teacher_queried": sample.teacher_queried,
                "accepted": sample.accepted,
                "state_shape": list(sample.state.shape),
                "student_action_shape": list(sample.student_action.shape),
                "teacher_action_shape": list(sample.teacher_action.shape),
                "sample_action_mse": float(np.mean(diff**2)),
                "sample_action_rmse": float(np.sqrt(np.mean(diff**2))),
                "max_abs_action_error": float(np.max(np.abs(diff))),
                "finite": bool(
                    np.all(np.isfinite(sample.state))
                    and np.all(np.isfinite(sample.student_action))
                    and np.all(np.isfinite(sample.teacher_action))
                ),
                "source": "synthetic_teacher_student_pair_schema_only",
            }
        )
    return samples, rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    accumulation = json.loads(VAE_ACCUMULATION_JSON.read_text(encoding="utf-8"))
    settings = accumulation["settings"]
    sample_count = int(accumulation["metrics"]["effective_batch_size"])
    samples, rows = build_synthetic_samples(settings, sample_count)
    metrics = teacher_student_discrepancy(samples)
    student = np.stack([sample.student_action for sample in samples], axis=0)
    teacher = np.stack([sample.teacher_action for sample in samples], axis=0)
    evaluation_action_mse = action_mse(student, teacher)
    split_counts = Counter(row["split"] for row in rows)
    micro_step_counts = Counter(row["micro_step"] for row in rows)
    state_shape_counts = Counter(str(row["state_shape"]) for row in rows)
    action_shape_counts = Counter(str(row["teacher_action_shape"]) for row in rows)
    synthetic_manifest = {
        "source": "deterministic_synthetic_pairs_from_vae_accumulation_settings",
        "teacher_policy_source": accumulation["synthetic_dagger_manifest"]["teacher_policy_source"],
        "student_policy_source": accumulation["synthetic_dagger_manifest"]["student_policy_source"],
        "teacher_query_rule": "teacher_queried=True for every synthetic schema sample",
        "rollout_source": accumulation["synthetic_dagger_manifest"]["rollout_source"],
        "is_true_dagger_rollout": False,
    }
    checks = {
        "all_evidence_paths_exist": VAE_ACCUMULATION_JSON.exists(),
        "uses_package_dagger_api": True,
        "uses_package_evaluation_metric_api": True,
        "sample_count_matches_effective_batch": len(rows) == sample_count == 30,
        "micro_batch_layout_matches_accumulation": dict(micro_step_counts) == {
            step: int(settings["micro_batch_size"]) for step in range(1, int(settings["gradient_accumulation_steps"]) + 1)
        },
        "state_dim_163": state_shape_counts == {"[163]": sample_count},
        "action_dim_29": action_shape_counts == {"[29]": sample_count},
        "all_teacher_queried": all(row["teacher_queried"] for row in rows),
        "all_rows_accepted_debug_only": all(row["accepted"] for row in rows),
        "all_rows_finite": all(row["finite"] for row in rows),
        "split_counts_nonzero": all(split_counts.get(split, 0) > 0 for split in ["train", "validation", "test"]),
        "discrepancy_metrics_finite": all(np.isfinite(value) for value in metrics.values()),
        "evaluation_action_mse_matches_dagger_metric": abs(evaluation_action_mse - metrics["action_mse"]) < 1e-15,
        "manifest_marks_not_true_dagger_rollout": synthetic_manifest["is_true_dagger_rollout"] is False,
        "does_not_claim_true_dagger_rollout": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "dagger_schema_audit",
        "scope": "package-level DAgger teacher-query sample schema and synthetic discrepancy metrics",
        "row_count": len(rows),
        "split_counts": dict(sorted(split_counts.items())),
        "micro_step_counts": {str(k): v for k, v in sorted(micro_step_counts.items())},
        "state_shape_counts": dict(sorted(state_shape_counts.items())),
        "action_shape_counts": dict(sorted(action_shape_counts.items())),
        "metrics": {**metrics, "evaluation_action_mse": evaluation_action_mse},
        "settings": {
            "seed": int(settings["seed"]) + 17,
            "state_dim": int(settings["proprioception_dim"] + settings["reference_motion_dim"] + settings["anchor_error_dim"]),
            "action_dim": int(settings["num_joints"]),
            "micro_batch_size": int(settings["micro_batch_size"]),
            "gradient_accumulation_steps": int(settings["gradient_accumulation_steps"]),
            "effective_batch_size": sample_count,
        },
        "synthetic_dagger_manifest": synthetic_manifest,
        "checks": checks,
        "missing_evidence_rows": [],
        "rows": rows,
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "partial",
            "why_not_complete": (
                "This validates the DAgger sample schema and teacher/student discrepancy math with deterministic "
                "synthetic pairs derived from the VAE accumulation settings. It does not run Isaac, query a real "
                "teacher policy, aggregate true environment states, or train a VAE checkpoint."
            ),
        },
        "outputs": {
            "json": str(OUT / "dagger_schema_audit.json"),
            "tsv": str(OUT / "dagger_schema_audit.tsv"),
        },
    }
    (OUT / "dagger_schema_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT / "dagger_schema_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "sample_id",
            "rollout_id",
            "timestep",
            "micro_step",
            "micro_batch_index",
            "split",
            "teacher_queried",
            "accepted",
            "state_shape",
            "student_action_shape",
            "teacher_action_shape",
            "sample_action_mse",
            "sample_action_rmse",
            "max_abs_action_error",
            "finite",
            "source",
        ]
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(row[key]) if isinstance(row[key], list) else row[key] for key in fieldnames})
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
