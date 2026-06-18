#!/usr/bin/env python3
"""Audit whether current Level-C state-latent artifacts satisfy the paper training-dataset contract."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/state_latent_training_dataset_contract_audit"
PAPER_STATE_JSON = ROOT / "res/level_c/paper_state_windows/level_c_paper_state_windows.json"
STATE_LATENT_SCHEMA_JSON = ROOT / "res/level_c/state_latent_schema_audit/state_latent_schema_audit.json"
STATE_LATENT_CONSISTENCY_JSON = (
    ROOT / "res/level_c/state_latent_dataset_consistency_audit/level_c_state_latent_dataset_consistency_audit.json"
)
DATASET_PROTOCOL_JSON = ROOT / "res/level_c/dataset_collection_protocol_audit/level_c_dataset_collection_protocol_audit.json"
ROLLOUT_MANIFEST_JSON = ROOT / "res/level_c/rollout_rejection_manifest_probe/level_c_rollout_rejection_manifest_probe.json"
DAGGER_SCHEMA_JSON = ROOT / "res/level_c/dagger_schema_audit/dagger_schema_audit.json"
VAE_LATENTS_JSON = ROOT / "res/level_c/vae_debug_overfit_latent_artifact/level_c_vae_debug_overfit_latent_artifact.json"
VAE_LATENTS_NPZ = ROOT / "reproduction/data/level_c_vae_debug_latents/debug_tiny_vae_state_latent_windows.npz"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_tsv_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "requirement_id",
        "requirement",
        "paper_required",
        "current_status",
        "evidence",
        "gap",
        "paper_trainable_satisfied",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})
    tmp.replace(path)


def npz_summary(path: Path) -> dict[str, Any]:
    with np.load(path) as data:
        keys = sorted(data.files)
        state_keys = [key for key in keys if key.endswith("_states")]
        latent_keys = [key for key in keys if key.endswith("_latents")]
        teacher_keys = [key for key in keys if key.endswith("_teacher_action")]
        decoded_keys = [key for key in keys if key.endswith("_decoded_action")]
        mu_keys = [key for key in keys if key.endswith("_mu")]
        logvar_keys = [key for key in keys if key.endswith("_logvar")]
        latent_abs = []
        state_shape_counts: dict[str, int] = {}
        latent_shape_counts: dict[str, int] = {}
        for key in latent_keys:
            arr = data[key]
            latent_abs.append(float(np.mean(np.abs(arr))))
            shape = "x".join(str(dim) for dim in arr.shape)
            latent_shape_counts[shape] = latent_shape_counts.get(shape, 0) + 1
        for key in state_keys:
            shape = "x".join(str(dim) for dim in data[key].shape)
            state_shape_counts[shape] = state_shape_counts.get(shape, 0) + 1
    return {
        "path": str(path),
        "key_count": len(keys),
        "sample_count_from_state_keys": len(state_keys),
        "state_key_count": len(state_keys),
        "latent_key_count": len(latent_keys),
        "teacher_action_key_count": len(teacher_keys),
        "decoded_action_key_count": len(decoded_keys),
        "mu_key_count": len(mu_keys),
        "logvar_key_count": len(logvar_keys),
        "state_shape_counts": state_shape_counts,
        "latent_shape_counts": latent_shape_counts,
        "latent_abs_mean": float(np.mean(latent_abs)) if latent_abs else 0.0,
        "latent_abs_min": float(np.min(latent_abs)) if latent_abs else 0.0,
        "latent_abs_max": float(np.max(latent_abs)) if latent_abs else 0.0,
    }


def row(
    requirement_id: str,
    requirement: str,
    current_status: str,
    evidence: list[str],
    gap: str,
    paper_trainable_satisfied: bool,
) -> dict[str, Any]:
    return {
        "requirement_id": requirement_id,
        "requirement": requirement,
        "paper_required": True,
        "current_status": current_status,
        "evidence": ";".join(evidence),
        "gap": gap,
        "paper_trainable_satisfied": paper_trainable_satisfied,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paper_state = load_json(PAPER_STATE_JSON)
    schema = load_json(STATE_LATENT_SCHEMA_JSON)
    consistency = load_json(STATE_LATENT_CONSISTENCY_JSON)
    protocol = load_json(DATASET_PROTOCOL_JSON)
    rollout = load_json(ROLLOUT_MANIFEST_JSON)
    dagger = load_json(DAGGER_SCHEMA_JSON)
    vae_latents = load_json(VAE_LATENTS_JSON)
    npz = npz_summary(VAE_LATENTS_NPZ)

    evidence_paths = [
        PAPER_STATE_JSON,
        STATE_LATENT_SCHEMA_JSON,
        STATE_LATENT_CONSISTENCY_JSON,
        DATASET_PROTOCOL_JSON,
        ROLLOUT_MANIFEST_JSON,
        DAGGER_SCHEMA_JSON,
        VAE_LATENTS_JSON,
        VAE_LATENTS_NPZ,
    ]
    rows = [
        row(
            "paper_state_tokens",
            "Use paper-formula state tokens with history 4, horizon 16, sequence length 21.",
            "debug_satisfied",
            [str(PAPER_STATE_JSON), str(STATE_LATENT_CONSISTENCY_JSON)],
            "",
            True,
        ),
        row(
            "state_latent_dimensions",
            "Provide 99-D state, 32-D latent, and 131-D concatenated state-latent tokens.",
            "debug_satisfied",
            [str(STATE_LATENT_SCHEMA_JSON), str(STATE_LATENT_CONSISTENCY_JSON), str(VAE_LATENTS_NPZ)],
            "",
            True,
        ),
        row(
            "motion_level_split",
            "Keep train/validation/test motion splits without cross-motion leakage.",
            "debug_satisfied",
            [str(STATE_LATENT_SCHEMA_JSON), str(STATE_LATENT_CONSISTENCY_JSON)],
            "",
            True,
        ),
        row(
            "trained_vae_rollout_latents",
            "Record latent trajectories from a trained conditional VAE policy rollout.",
            "missing",
            [str(VAE_LATENTS_JSON), str(DATASET_PROTOCOL_JSON)],
            "Current latents are nonzero debug tiny-VAE artifacts, not trained paper VAE rollout latents.",
            False,
        ),
        row(
            "true_dagger_teacher_student_rollout",
            "Collect states/actions from true DAgger teacher/student closed-loop rollouts.",
            "missing",
            [str(DAGGER_SCHEMA_JSON), str(DATASET_PROTOCOL_JSON)],
            "Current DAgger samples are deterministic synthetic/debug schema samples, not Isaac teacher-policy rollouts.",
            False,
        ),
        row(
            "ou_action_perturbation_collection",
            "Apply paper OU action perturbations during live rollout collection.",
            "debug_protocol_only",
            [str(DATASET_PROTOCOL_JSON), str(ROLLOUT_MANIFEST_JSON)],
            "OU constants and a synthetic manifest are audited; no live policy rollout with perturbations exists.",
            False,
        ),
        row(
            "two_point_five_second_recording",
            "Record 2.5 second state-latent rollout windows from live VAE policy execution.",
            "debug_protocol_only",
            [str(DATASET_PROTOCOL_JSON), str(ROLLOUT_MANIFEST_JSON)],
            "Window lengths are represented in a debug manifest; no live VAE policy execution log is present.",
            False,
        ),
        row(
            "five_second_stability_rejection",
            "Verify 5 second stability and reject failed episodes using real rollout failure signals.",
            "missing",
            [str(DATASET_PROTOCOL_JSON), str(ROLLOUT_MANIFEST_JSON)],
            "Accept/reject fields are debug placeholders and failure signals are explicitly missing.",
            False,
        ),
        row(
            "approximately_100x_sample_coverage",
            "Cover each start sample approximately 100 times with accepted rollout data.",
            "debug_protocol_only",
            [str(DATASET_PROTOCOL_JSON), str(ROLLOUT_MANIFEST_JSON)],
            "The debug manifest has 100x synthetic coverage; no accepted live rollout coverage measurement exists.",
            False,
        ),
        row(
            "sagittal_symmetric_augmented_dataset",
            "Include sagittal-symmetric augmentation in the trainable state-latent dataset.",
            "debug_protocol_only",
            [str(DATASET_PROTOCOL_JSON)],
            "Candidate symmetry mechanics are audited elsewhere, but no augmented trainable dataset was produced.",
            False,
        ),
        row(
            "trainable_dataset_provenance_manifest",
            "Store per-window provenance: source motion, teacher/student policy, timestep, latent source, augmentation, accept/reject, and split.",
            "partial_debug_only",
            [str(STATE_LATENT_SCHEMA_JSON), str(DATASET_PROTOCOL_JSON), str(VAE_LATENTS_JSON)],
            "Debug provenance exists for source/split/window ordering, but teacher/student policy, true latent source, augmentation, and accept/reject are not paper-level evidence.",
            False,
        ),
        row(
            "paper_scale_training_dataset_size",
            "Provide enough accepted windows for paper-scale diffusion training rather than a tiny debug fixture.",
            "missing",
            [str(STATE_LATENT_CONSISTENCY_JSON), str(VAE_LATENTS_NPZ)],
            "Current dataset has 84 debug windows over 3 short motions; it is not a paper-scale trainable dataset.",
            False,
        ),
    ]

    missing_rows = [item for item in rows if not item["paper_trainable_satisfied"]]
    checks = {
        "all_evidence_paths_exist": all(path.exists() for path in evidence_paths),
        "source_audits_status_ok": all(
            data["status"] == "ok" for data in [paper_state, schema, consistency, protocol, rollout, dagger, vae_latents]
        ),
        "npz_has_84_state_latent_samples": npz["state_key_count"] == 84 and npz["latent_key_count"] == 84,
        "npz_shapes_match_21x99_and_21x32": npz["state_shape_counts"] == {"21x99": 84}
        and npz["latent_shape_counts"] == {"21x32": 84},
        "debug_dataset_dimensions_match_contract": consistency["metrics"]["state_dim"] == 99
        and consistency["metrics"]["latent_dim"] == 32
        and consistency["metrics"]["token_dim"] == 131
        and consistency["metrics"]["sequence_length"] == 21,
        "debug_dataset_split_counts_present": consistency["metrics"]["per_split_counts"]
        == {"test": 28, "train": 28, "validation": 28},
        "missing_training_requirements_recorded": len(missing_rows) == 9,
        "no_current_artifact_qualifies_as_paper_trainable_dataset": not all(
            item["paper_trainable_satisfied"] for item in rows
        ),
        "does_not_claim_goal_complete": True,
    }

    metrics = {
        "contract_row_count": len(rows),
        "paper_trainable_satisfied_count": sum(int(item["paper_trainable_satisfied"]) for item in rows),
        "missing_or_debug_only_count": len(missing_rows),
        "debug_npz_sample_count": npz["sample_count_from_state_keys"],
        "debug_npz_key_count": npz["key_count"],
        "debug_state_shape_counts": npz["state_shape_counts"],
        "debug_latent_shape_counts": npz["latent_shape_counts"],
        "debug_latent_abs_mean": npz["latent_abs_mean"],
        "consistency_row_count": consistency["metrics"]["row_count"],
        "rollout_manifest_episode_rows": rollout["metrics"]["episode_manifest_rows"],
        "dagger_schema_sample_count": dagger["metrics"]["sample_count"],
        "failed_check_count": sum(int(not value) for value in checks.values()),
    }
    json_path = OUT / "level_c_state_latent_training_dataset_contract_audit.json"
    tsv_path = OUT / "level_c_state_latent_training_dataset_contract_audit.tsv"
    summary = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "training_dataset_contract_audit",
        "scope": (
            "Contract check for whether current Level-C state-latent artifacts can be treated as the "
            "paper's trainable diffusion dataset."
        ),
        "source_artifacts": {
            "paper_state_windows": str(PAPER_STATE_JSON),
            "state_latent_schema": str(STATE_LATENT_SCHEMA_JSON),
            "state_latent_consistency": str(STATE_LATENT_CONSISTENCY_JSON),
            "dataset_collection_protocol": str(DATASET_PROTOCOL_JSON),
            "rollout_manifest": str(ROLLOUT_MANIFEST_JSON),
            "dagger_schema": str(DAGGER_SCHEMA_JSON),
            "vae_debug_latents": str(VAE_LATENTS_JSON),
            "vae_debug_latents_npz": str(VAE_LATENTS_NPZ),
        },
        "current_npz_summary": npz,
        "rows": rows,
        "missing_rows": missing_rows,
        "metrics": metrics,
        "checks": checks,
        "interpretation": {
            "paper_level_status": "partial_debug_dataset_only",
            "goal_complete": False,
            "trainable_dataset_available": False,
            "why_not_complete": (
                "The current Level-C artifacts are internally consistent debug state-latent fixtures. They do not "
                "satisfy the paper's trainable state-latent dataset contract because true DAgger/teacher rollouts, "
                "trained VAE rollout latents, live OU-perturbed collection, real 2.5s/5s accept-reject decisions, "
                "paper-scale coverage, and augmented accepted dataset provenance are missing."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path)},
    }
    write_tsv_atomic(tsv_path, rows)
    write_json_atomic(json_path, summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(json_path),
                "rows": len(rows),
                "missing_or_debug_only": len(missing_rows),
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
