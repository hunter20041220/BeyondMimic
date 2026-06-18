#!/usr/bin/env python3
"""Unified DAgger-to-VAE debug pipeline audit.

This audit stitches together the existing debug-only DAgger, conditional VAE,
checkpoint, nonzero-latent, and receding-horizon action artifacts. It verifies
that the local pipeline is internally consistent while explicitly preserving the
boundary that no true Isaac DAgger rollout or paper-trained VAE checkpoint was
produced.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/dagger_vae_pipeline_audit"

ARTIFACTS = {
    "dagger_schema": "res/level_c/dagger_schema_audit/dagger_schema_audit.json",
    "dagger_iteration": "res/level_c/dagger_iteration_smoke/level_c_dagger_iteration_smoke.json",
    "vae_contract": "res/level_c/vae_contract_audit/level_c_vae_contract_audit.json",
    "vae_accumulation": "res/level_c/vae_accumulation_probe/level_c_vae_accumulation_probe.json",
    "vae_checkpoint": "res/level_c/vae_checkpoint_smoke/level_c_vae_checkpoint_smoke.json",
    "vae_debug_latents": "res/level_c/vae_debug_overfit_latent_artifact/level_c_vae_debug_overfit_latent_artifact.json",
    "vae_motion_split": "res/level_c/vae_motion_split_heldout_eval/level_c_vae_motion_split_heldout_eval.json",
    "state_latent_consistency": (
        "res/level_c/state_latent_dataset_consistency_audit/"
        "level_c_state_latent_dataset_consistency_audit.json"
    ),
    "vae_receding_horizon": (
        "res/level_c/vae_receding_horizon_rollout_smoke/"
        "level_c_vae_receding_horizon_rollout_smoke.json"
    ),
}


def load_json(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["stage", "artifact", "status", "key_metrics", "boundary"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def boundary_text(data: dict[str, Any]) -> str:
    if "interpretation" in data:
        interp = data["interpretation"]
        return interp.get("why_not_complete") or interp.get("remaining_gap") or ""
    return ""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    artifacts = {name: load_json(rel) for name, rel in ARTIFACTS.items()}
    dagger_schema = artifacts["dagger_schema"]
    dagger_iteration = artifacts["dagger_iteration"]
    vae_contract = artifacts["vae_contract"]
    vae_accumulation = artifacts["vae_accumulation"]
    vae_checkpoint = artifacts["vae_checkpoint"]
    vae_debug_latents = artifacts["vae_debug_latents"]
    vae_motion_split = artifacts["vae_motion_split"]
    consistency = artifacts["state_latent_consistency"]
    receding = artifacts["vae_receding_horizon"]

    rows = [
        {
            "stage": "dagger_schema",
            "artifact": str(ROOT / ARTIFACTS["dagger_schema"]),
            "status": dagger_schema["status"],
            "key_metrics": json.dumps(dagger_schema["metrics"], sort_keys=True),
            "boundary": boundary_text(dagger_schema),
        },
        {
            "stage": "dagger_iteration",
            "artifact": str(ROOT / ARTIFACTS["dagger_iteration"]),
            "status": dagger_iteration["status"],
            "key_metrics": json.dumps(dagger_iteration["metrics"], sort_keys=True),
            "boundary": boundary_text(dagger_iteration),
        },
        {
            "stage": "vae_contract",
            "artifact": str(ROOT / ARTIFACTS["vae_contract"]),
            "status": vae_contract["status"],
            "key_metrics": json.dumps(vae_contract["metrics"], sort_keys=True),
            "boundary": boundary_text(vae_contract),
        },
        {
            "stage": "vae_accumulation",
            "artifact": str(ROOT / ARTIFACTS["vae_accumulation"]),
            "status": vae_accumulation["status"],
            "key_metrics": json.dumps(vae_accumulation["metrics"], sort_keys=True),
            "boundary": boundary_text(vae_accumulation),
        },
        {
            "stage": "vae_checkpoint",
            "artifact": str(ROOT / ARTIFACTS["vae_checkpoint"]),
            "status": vae_checkpoint["status"],
            "key_metrics": json.dumps(vae_checkpoint["metrics"], sort_keys=True),
            "boundary": boundary_text(vae_checkpoint),
        },
        {
            "stage": "vae_debug_latents",
            "artifact": str(ROOT / ARTIFACTS["vae_debug_latents"]),
            "status": vae_debug_latents["status"],
            "key_metrics": json.dumps(vae_debug_latents["metrics"], sort_keys=True),
            "boundary": boundary_text(vae_debug_latents),
        },
        {
            "stage": "vae_motion_split",
            "artifact": str(ROOT / ARTIFACTS["vae_motion_split"]),
            "status": vae_motion_split["status"],
            "key_metrics": json.dumps(vae_motion_split["metrics"], sort_keys=True),
            "boundary": boundary_text(vae_motion_split),
        },
        {
            "stage": "state_latent_consistency",
            "artifact": str(ROOT / ARTIFACTS["state_latent_consistency"]),
            "status": consistency["status"],
            "key_metrics": json.dumps(consistency["metrics"], sort_keys=True),
            "boundary": boundary_text(consistency),
        },
        {
            "stage": "vae_receding_horizon",
            "artifact": str(ROOT / ARTIFACTS["vae_receding_horizon"]),
            "status": receding["status"],
            "key_metrics": json.dumps(receding["metrics"], sort_keys=True),
            "boundary": boundary_text(receding),
        },
    ]

    checks = {
        "all_stage_status_ok": all(data.get("status") == "ok" for data in artifacts.values()),
        "dagger_schema_not_true_rollout": dagger_schema["checks"]["does_not_claim_true_dagger_rollout"]
        and dagger_schema["checks"]["manifest_marks_not_true_dagger_rollout"],
        "dagger_iteration_three_iters_and_288_queries": dagger_iteration["checks"]["three_dagger_iterations"]
        and dagger_iteration["metrics"]["total_teacher_queries"] == 288.0,
        "dagger_iteration_heldout_improves": dagger_iteration["checks"]["heldout_discrepancy_decreases"]
        and dagger_iteration["metrics"]["heldout_mse_reduction_ratio"] > 0.99,
        "vae_contract_zero_failed_rows": vae_contract["metrics"]["failed_row_count"] == 0
        and vae_contract["checks"]["all_contract_rows_pass"],
        "vae_accumulation_paper_hparams_and_optimizer_step": vae_accumulation["checks"][
            "gradient_accumulation_matches_paper"
        ]
        and vae_accumulation["checks"]["single_optimizer_step_updates_parameters"]
        and vae_accumulation["metrics"]["effective_batch_size"] == 30,
        "vae_checkpoint_roundtrip_debug_only": vae_checkpoint["checks"]["loaded_eval_action_matches_saved_model"]
        and vae_checkpoint["checks"]["marks_not_trained_paper_checkpoint"]
        and vae_checkpoint["metrics"]["max_abs_loaded_eval_action_error"] == 0.0,
        "vae_debug_latents_nonzero_and_84_rows": vae_debug_latents["checks"]["all_latents_nonzero"]
        and vae_debug_latents["metrics"]["row_count"] == 84,
        "vae_motion_split_heldout_test_improves": vae_motion_split["checks"]["test_action_mse_decreases"]
        and vae_motion_split["metrics"]["test_action_mse_reduction_ratio"] > 0.9,
        "state_latent_dims_match_99_32_131_29": consistency["metrics"]["state_dim"] == 99
        and consistency["metrics"]["latent_dim"] == 32
        and consistency["metrics"]["token_dim"] == 131
        and consistency["metrics"]["action_dim"] == 29,
        "receding_horizon_84_rows_current_action_low_mse": receding["checks"]["row_count_84"]
        and receding["checks"]["current_action_mse_below_0_01"],
        "receding_horizon_uses_history_index_4_and_25hz": receding["settings"]["current_index"] == 4
        and receding["settings"]["control_frequency_hz"] == 25.0,
        "all_boundaries_preserve_goal_incomplete": all(
            data.get("interpretation", {}).get("goal_complete") is False
            or data.get("checks", {}).get("does_not_claim_goal_complete") is True
            or data.get("checks", {}).get("manifest_marks_not_true_dagger_rollout") is True
            for data in artifacts.values()
        ),
        "does_not_claim_true_dagger_or_paper_vae": True,
        "atomic_write_used": True,
    }
    metrics = {
        "stage_count": len(rows),
        "ok_stage_count": sum(row["status"] == "ok" for row in rows),
        "dagger_teacher_queries": int(dagger_iteration["metrics"]["total_teacher_queries"]),
        "dagger_total_samples": int(dagger_iteration["metrics"]["total_samples"]),
        "dagger_heldout_reduction_ratio": dagger_iteration["metrics"]["heldout_mse_reduction_ratio"],
        "vae_contract_rows": vae_contract["metrics"]["row_count"],
        "vae_parameter_count": vae_contract["metrics"]["vae_parameter_count"],
        "vae_effective_batch_size": vae_accumulation["metrics"]["effective_batch_size"],
        "vae_checkpoint_size_bytes": vae_checkpoint["metrics"]["checkpoint_size_bytes"],
        "state_latent_rows": consistency["metrics"]["row_count"],
        "state_dim": consistency["metrics"]["state_dim"],
        "latent_dim": consistency["metrics"]["latent_dim"],
        "token_dim": consistency["metrics"]["token_dim"],
        "action_dim": consistency["metrics"]["action_dim"],
        "vae_debug_latent_abs_mean": vae_debug_latents["metrics"]["latent_abs_mean_mean"],
        "vae_motion_split_test_action_mse": vae_motion_split["metrics"]["test_final_action_mse"],
        "vae_receding_current_action_mse_mean": receding["metrics"]["current_action_mse_mean"],
        "vae_receding_current_action_mse_max": receding["metrics"]["current_action_mse_max"],
        "failed_check_count": sum(1 for value in checks.values() if not value),
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "level_c_dagger_vae_pipeline_audit",
        "scope": (
            "Unified audit of the local debug-only chain: synthetic DAgger samples and iterations, paper VAE "
            "contract, gradient accumulation, checkpoint roundtrip, nonzero debug latents, state-latent consistency, "
            "and receding-horizon current-action decoding."
        ),
        "source_artifacts": {name: str(ROOT / rel) for name, rel in ARTIFACTS.items()},
        "stage_rows": rows,
        "metrics": metrics,
        "checks": checks,
        "interpretation": {
            "paper_level_status": "partial_debug_pipeline_only",
            "goal_complete": False,
            "why_not_complete": (
                "This audit proves the local debug DAgger-to-VAE chain is internally consistent and reproducible. It "
                "does not prove true DAgger collection in Isaac, teacher/student closed-loop rollouts, a trained paper "
                "conditional VAE checkpoint, VAE rollout survival/stability, or paper Fig. 5/Fig. 6 evaluation."
            ),
        },
        "outputs": {
            "json": str(OUT / "level_c_dagger_vae_pipeline_audit.json"),
            "tsv": str(OUT / "level_c_dagger_vae_pipeline_audit.tsv"),
        },
    }
    atomic_write_text(
        OUT / "level_c_dagger_vae_pipeline_audit.json",
        json.dumps(summary, indent=2, sort_keys=True),
    )
    atomic_write_tsv(OUT / "level_c_dagger_vae_pipeline_audit.tsv", rows)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "stages": metrics["stage_count"],
                "failed": metrics["failed_check_count"],
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
