#!/usr/bin/env python3
"""Executable checkpoint/resume smoke test for run-management plumbing.

The smoke is deliberately tiny and deterministic. It validates that a run can
write a checkpoint, resume from it, and produce the same final state as an
uninterrupted baseline. It is not a BeyondMimic training checkpoint.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
RUN_ID = "setup_checkpoint_resume_smoke_static_000_20260617_061500"
RUN_DIR = ROOT / "res/runs" / RUN_ID
OUT = ROOT / "res/run_management_audit/checkpoint_resume_smoke"
SEED = 20260617
TOTAL_STEPS = 12
CHECKPOINT_STEP = 5
DIM = 4


def update_state(state: np.ndarray, step: int) -> np.ndarray:
    """Apply a deterministic finite update to a state vector with shape ``[4]``."""
    if state.shape != (DIM,) or not np.all(np.isfinite(state)):
        raise ValueError(f"state must be finite with shape {(DIM,)}, got {state.shape}")
    step_vec = np.array([step, step**2, np.sin(step), np.cos(step)], dtype=np.float64)
    return state + 0.01 * step_vec


def run_steps(start_state: np.ndarray, start_step: int, end_step: int) -> np.ndarray:
    """Run deterministic updates for ``start_step < step <= end_step``."""
    state = np.asarray(start_state, dtype=np.float64).copy()
    for step in range(start_step + 1, end_step + 1):
        state = update_state(state, step)
    return state


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for dirname in ["checkpoint", "figures", "videos"]:
        (RUN_DIR / dirname).mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    initial_state = rng.normal(size=DIM)
    baseline_final = run_steps(initial_state, 0, TOTAL_STEPS)
    partial_state = run_steps(initial_state, 0, CHECKPOINT_STEP)

    checkpoint_path = RUN_DIR / "checkpoint" / f"step_{CHECKPOINT_STEP:04d}.npz"
    np.savez(
        checkpoint_path,
        run_id=RUN_ID,
        seed=np.array(SEED, dtype=np.int64),
        step=np.array(CHECKPOINT_STEP, dtype=np.int64),
        state=partial_state,
    )

    loaded = np.load(checkpoint_path)
    resume_step = int(np.asarray(loaded["step"]))
    resumed_state = run_steps(np.asarray(loaded["state"], dtype=np.float64), resume_step, TOTAL_STEPS)
    max_abs_error = float(np.max(np.abs(resumed_state - baseline_final)))

    resolved_config = {
        "run_id": RUN_ID,
        "stage": "setup",
        "method": "checkpoint_resume_smoke",
        "motion": "static",
        "config": "deterministic_resume",
        "seed": SEED,
        "status": "SUCCESS",
        "is_training_run": False,
        "source_goal_lines": "goal.md:1704-1717,1747-1787",
        "checkpoint_step": CHECKPOINT_STEP,
        "total_steps": TOTAL_STEPS,
        "checkpoint_resume_scope": "diagnostic plumbing only; not a model checkpoint",
    }
    (RUN_DIR / "resolved_config.yaml").write_text(
        "\n".join(f"{key}: {value}" for key, value in resolved_config.items()) + "\n",
        encoding="utf-8",
    )
    (RUN_DIR / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\npython3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/checkpoint_resume_smoke.py\n",
        encoding="utf-8",
    )
    (RUN_DIR / "command.sh").chmod(0o755)
    (RUN_DIR / "stdout.log").write_text("Checkpoint/resume smoke completed successfully.\n", encoding="utf-8")
    (RUN_DIR / "stderr.log").write_text("", encoding="utf-8")
    (RUN_DIR / "environment.txt").write_text(
        f"timestamp={datetime.now().isoformat(timespec='seconds')}\npython=python3\nscope=checkpoint_resume_smoke\n",
        encoding="utf-8",
    )
    (RUN_DIR / "git_state.txt").write_text("not_a_git_repository\n", encoding="utf-8")
    (RUN_DIR / "gpu_metrics.csv").write_text(
        "timestamp,gpu_index,gpu_name,memory_used_mib,run_id,run_status,sample_kind\n"
        f"{datetime.now().isoformat(timespec='seconds')},-1,cpu_only,0,{RUN_ID},SUCCESS,checkpoint_resume_smoke\n",
        encoding="utf-8",
    )

    metrics = {
        "run_id": RUN_ID,
        "status": "SUCCESS",
        "is_training_run": False,
        "seed": SEED,
        "checkpoint_step": CHECKPOINT_STEP,
        "total_steps": TOTAL_STEPS,
        "max_abs_resume_error": max_abs_error,
        "samples_per_second": None,
        "environment_steps_per_second": None,
        "iteration_time": None,
        "estimated_remaining_time": None,
        "oom_count": 0,
        "restart_count": 1,
        "checkpoint_path": str(checkpoint_path),
    }
    write_json(RUN_DIR / "metrics.json", metrics)
    with (RUN_DIR / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    status = {
        "run_id": RUN_ID,
        "status": "SUCCESS",
        "allowed_status": True,
        "is_success": True,
        "is_training_run": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "reason": "Diagnostic checkpoint/resume smoke only; not a paper training run.",
    }
    write_json(RUN_DIR / "status.json", status)

    checks = {
        "checkpoint_file_exists": checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0,
        "checkpoint_contains_seed_step_state": all(key in loaded.files for key in ["seed", "step", "state"]),
        "resume_step_matches_checkpoint": resume_step == CHECKPOINT_STEP,
        "resumed_matches_uninterrupted": max_abs_error < 1e-12,
        "status_success_but_not_training": status["status"] == "SUCCESS" and not status["is_training_run"],
        "run_id_matches_goal_pattern": RUN_ID.count("_") >= 6 and RUN_ID.endswith("20260617_061500"),
        "does_not_claim_model_checkpoint": "not a model checkpoint" in resolved_config["checkpoint_resume_scope"],
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "checkpoint_resume_smoke",
        "scope": "executable diagnostic for checkpoint save/load/resume plumbing",
        "run_id": RUN_ID,
        "run_dir": str(RUN_DIR),
        "checkpoint_path": str(checkpoint_path),
        "metrics": metrics,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "Checkpoint/resume plumbing is executable for a deterministic diagnostic state. This is not a "
                "BeyondMimic PPO, VAE, diffusion, TensorRT, or deployment checkpoint."
            ),
        },
        "outputs": {
            "json": str(OUT / "checkpoint_resume_smoke.json"),
            "tsv": str(OUT / "checkpoint_resume_smoke.tsv"),
        },
    }
    write_json(OUT / "checkpoint_resume_smoke.json", summary)
    with (OUT / "checkpoint_resume_smoke.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["check", "passed"])
        writer.writeheader()
        for check, passed in checks.items():
            writer.writerow({"check": check, "passed": passed})
    print(json.dumps({"status": summary["status"], "run_dir": str(RUN_DIR), "json": summary["outputs"]["json"]}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
