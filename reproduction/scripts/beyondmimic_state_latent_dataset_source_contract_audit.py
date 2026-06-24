#!/usr/bin/env python3
"""Audit whether current state-latent datasets really use BeyondMimic hybrid states.

This is a pre-training hard gate.  It does not train a model.  It checks that
the rollout shards contain the raw simulator state needed to construct the
paper's 99-D hybrid yaw-centric state, and that downstream diffusion code is
not silently learning from the 160-D Stage-1 policy observation.
"""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/audits/state_latent_dataset_source_contract"

PAPER_ROOT = ROOT / "reproduction/paper/source/root.tex"
PAPER_METHOD = ROOT / "reproduction/paper/source/tex/method.tex"
HYBRID_SCHEMA_JSON = (
    ROOT / "res/audits/hybrid_state_schema_contract/beyondmimic_hybrid_state_schema_contract_audit.json"
)
PAPER_CONTRACT_DATASET_JSON = (
    ROOT
    / "res/level_c/official_importer_export_paper_contract_teacher_rollout_state_latent_dataset/"
    "level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.json"
)
STAGE1_MULTI_DATASET_JSON = (
    ROOT
    / "res/level_c/stage1_multisource_teacher_rollout_state_latent_dataset/"
    "level_c_stage1_multisource_teacher_rollout_state_latent_dataset.json"
)
PAPER_CONTRACT_TEACHER_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_paper_contract_best_teacher_rollout_dataset/"
    "tracking_g1_official_importer_export_paper_contract_best_teacher_rollout_dataset.json"
)

STATE_LATENT_BUILDER = ROOT / "reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py"
TEACHER_ROLLOUT_COLLECTOR = ROOT / "reproduction/scripts/tracking_g1_resource_adjusted_teacher_rollout_dataset.py"
PAPER_DATASET_WRAPPER = (
    ROOT / "reproduction/scripts/level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.py"
)
CORE_STATE_HELPERS = ROOT / "reproduction/src/beyondmimic_reimpl/state.py"
CORE_MATH_TEST = ROOT / "reproduction/tests/test_core_math.py"
RESOURCE_DIFFUSION = ROOT / "reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py"
PAPER_TRANSFORMER_DIFFUSION = (
    ROOT / "reproduction/scripts/level_c_paper_contract_transformer_state_latent_diffusion_training.py"
)

REQUIRED_WORLD_STATE_FIELDS = {
    "root_pos_w": ["root_pos_w", "root_position_w", "root_pos", "robot_anchor_pos_w"],
    "root_quat_w_or_rot": [
        "root_quat_w",
        "root_quat_xyzw_w",
        "root_rot_w",
        "root_rotation",
        "robot_anchor_quat_w",
    ],
    "root_lin_vel_w": ["root_lin_vel_w", "root_velocity_w", "root_lin_vel", "robot_anchor_lin_vel_w"],
    "root_ang_vel_w": ["root_ang_vel_w", "root_angular_velocity_w", "root_ang_vel", "robot_anchor_ang_vel_w"],
    "body_pos_w": ["body_pos_w", "target_body_pos_w", "rigid_body_pos_w", "body_positions_w", "robot_body_pos_w"],
    "body_lin_vel_w": [
        "body_lin_vel_w",
        "target_body_lin_vel_w",
        "rigid_body_lin_vel_w",
        "body_velocities_w",
        "robot_body_lin_vel_w",
    ],
}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def nested_find_npz_paths(obj: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(obj, dict):
        for value in obj.values():
            paths.extend(nested_find_npz_paths(value))
    elif isinstance(obj, list):
        for value in obj:
            paths.extend(nested_find_npz_paths(value))
    elif isinstance(obj, str) and obj.endswith(".npz"):
        paths.append(obj)
    return paths


def dataset_meta(summary: dict[str, Any]) -> dict[str, Any]:
    return summary.get("worker_summary", {}).get("dataset", {})


def dataset_shards(summary: dict[str, Any]) -> list[dict[str, Any]]:
    shards = summary.get("worker_summary", {}).get("shards", [])
    return shards if isinstance(shards, list) else []


def first_teacher_shards(summary: dict[str, Any]) -> list[str]:
    direct = summary.get("run", {}).get("shard_npz_paths")
    if isinstance(direct, list) and direct:
        return [str(item) for item in direct]
    outputs = summary.get("outputs", {}).get("shard_npz_paths")
    if isinstance(outputs, list) and outputs:
        return [str(item) for item in outputs]
    return nested_find_npz_paths(summary)


def inspect_npz(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    row: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "keys": [], "shapes": {}, "missing": []}
    if not path.is_file():
        row["missing"] = list(REQUIRED_WORLD_STATE_FIELDS)
        return row
    with np.load(path, mmap_mode="r") as data:
        keys = list(data.files)
        row["keys"] = keys
        row["shapes"] = {key: list(data[key].shape) for key in keys}
    key_set = set(row["keys"])
    missing = []
    matched = {}
    for field, aliases in REQUIRED_WORLD_STATE_FIELDS.items():
        hit = [name for name in aliases if name in key_set]
        matched[field] = hit
        if not hit:
            missing.append(field)
    row["matched_required_fields"] = matched
    row["missing"] = missing
    row["has_required_world_state_fields"] = not missing
    return row


def has_done_filter(text: str) -> bool:
    lower = text.lower()
    filters_done = "valid_contiguous_window_mask" in lower or "np.any(dones" in lower or ("dones[" in lower and "continue" in lower)
    rejects = "reject" in lower or "accepted" in lower or "5 seconds" in lower or "5s" in lower
    return filters_done and rejects


def add_row(
    rows: list[dict[str, Any]],
    contract: str,
    expected: str,
    observed: str,
    passed: bool,
    evidence: list[str],
    required_fix: str,
    status_if_fail: str = "blocked",
) -> None:
    rows.append(
        {
            "contract": contract,
            "expected_from_paper_or_gate": expected,
            "observed_in_current_project": observed,
            "passed": bool(passed),
            "status": "pass" if passed else status_if_fail,
            "evidence": evidence,
            "required_fix_before_long_training": required_fix,
        }
    )


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "contract",
        "status",
        "passed",
        "expected_from_paper_or_gate",
        "observed_in_current_project",
        "required_fix_before_long_training",
        "evidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["evidence"] = "; ".join(row.get("evidence", []))
            writer.writerow(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paper_root = read_text(PAPER_ROOT)
    paper_method = read_text(PAPER_METHOD)
    schema = read_json(HYBRID_SCHEMA_JSON)
    paper_dataset = read_json(PAPER_CONTRACT_DATASET_JSON)
    stage1_dataset = read_json(STAGE1_MULTI_DATASET_JSON)
    teacher_summary = read_json(PAPER_CONTRACT_TEACHER_JSON)

    builder_text = read_text(STATE_LATENT_BUILDER)
    collector_text = read_text(TEACHER_ROLLOUT_COLLECTOR)
    wrapper_text = read_text(PAPER_DATASET_WRAPPER)
    core_state_text = read_text(CORE_STATE_HELPERS)
    core_test_text = read_text(CORE_MATH_TEST)
    resource_diffusion_text = read_text(RESOURCE_DIFFUSION)
    paper_transformer_text = read_text(PAPER_TRANSFORMER_DIFFUSION)

    paper_meta = dataset_meta(paper_dataset)
    stage1_meta = dataset_meta(stage1_dataset)
    paper_shards = dataset_shards(paper_dataset)
    teacher_shards = first_teacher_shards(teacher_summary)
    inspected_shards = [inspect_npz(path) for path in teacher_shards[:2]]
    shard_missing = sorted({field for shard in inspected_shards for field in shard.get("missing", [])})

    expected_state_dim = int(schema.get("schema", {}).get("hybrid_state_dim", 99))
    expected_projected_state_dim = int(schema.get("schema", {}).get("projected_state_dim", 163))
    expected_latent_dim = int(paper_meta.get("latent_dim", stage1_meta.get("latent_dim", 32)) or 32)
    expected_original_token_dim = expected_state_dim + expected_latent_dim
    expected_projected_token_dim = expected_projected_state_dim + expected_latent_dim

    paper_state_patterns = [
        "hybrid character",
        "yaw-centric",
        "root position",
        "angular velocity",
        "Ornstein-Uhlenbeck",
        "5 seconds",
        "morphological symmetry",
    ]
    paper_sources_have_patterns = all(pattern.lower() in (paper_root + paper_method).lower() for pattern in paper_state_patterns)

    current_uses_policy_obs = "policy_obs" in str(paper_meta.get("state_source", "")).lower()
    current_obs_dim = int(paper_meta.get("obs_dim", -1) or -1)
    current_token_dim = int(paper_meta.get("token_dim", -1) or -1)
    total_done_count = int(sum(int(shard.get("done_count", 0) or 0) for shard in paper_shards))
    total_window_count = int(paper_meta.get("window_count", 0) or 0)
    reusable_hybrid_builder_available = (
        "def build_paper_hybrid_state_window" in core_state_text
        and "quat_format" in core_state_text
        and "body_lin_vel - root_lin_vel" in core_state_text
    )
    reusable_hybrid_builder_tested = (
        "test_paper_hybrid_state_window_from_raw_rollout_invariance" in core_test_text
        and "quat_format=\"wxyz\"" in core_test_text
    )

    paper_transformer_reads_policy_obs = '["policy_obs"]' in paper_transformer_text or "policy_obs" in paper_transformer_text
    resource_diffusion_has_legacy_policy_obs_fallback = "policy_obs" in resource_diffusion_text
    diffusion_reads_policy_obs = paper_transformer_reads_policy_obs
    builder_has_legacy_policy_obs_path = 'data["policy_obs"]' in builder_text or "legacy_policy_obs" in builder_text
    builder_has_paper_hybrid_path = (
        "build_paper_hybrid_state_window" in builder_text
        and "state_tokens" in builder_text
        and "BM_STATE_LATENT_REQUIRE_RAW_STATE" in builder_text
    )
    wrapper_overrides_to_policy_obs = "policy_obs in local paper-contract" in wrapper_text
    wrapper_requires_hybrid_state = (
        "BM_STATE_LATENT_STATE_MODE" in wrapper_text
        and "paper_hybrid" in wrapper_text
        and "BM_STATE_LATENT_REQUIRE_RAW_STATE" in wrapper_text
        and "BM_STATE_LATENT_REQUIRE_PAPER_CONTRACT_VAE" in wrapper_text
    )
    windows_filter_done_and_rejection = has_done_filter(builder_text)
    ou_collection_recorded = bool(re.search(r"\b(ornstein|uhlenbeck|ou[-_\s]?noise)\b", builder_text.lower()))
    symmetry_recorded = "symmetry" in builder_text.lower() or "mirror" in builder_text.lower()
    collector_records_required_world_state = all(
        any(alias in collector_text for alias in aliases)
        for aliases in REQUIRED_WORLD_STATE_FIELDS.values()
    )

    rows: list[dict[str, Any]] = []
    add_row(
        rows,
        "Paper state-latent source contract is readable",
        "Paper S3 requires hybrid character-yaw-centric state, OU perturbation collection, 5s rejection, and symmetry augmentation.",
        f"patterns_found={paper_sources_have_patterns}",
        paper_sources_have_patterns,
        [rel(PAPER_ROOT), rel(PAPER_METHOD)],
        "Restore/read paper source before claiming formula-level data alignment.",
    )
    add_row(
        rows,
        "Reusable hybrid state schema gate is available",
        "Previous gate must define 99-D hybrid state and 163-D projected state.",
        f"status={schema.get('status')}, hybrid_state_dim={expected_state_dim}, projected_state_dim={expected_projected_state_dim}",
        schema.get("checks", {}).get("hybrid_state_dim_99") is True
        and schema.get("checks", {}).get("projected_state_dim_163") is True,
        [rel(HYBRID_SCHEMA_JSON)],
        "Run/fix hybrid schema audit before building a trainable state-latent dataset.",
    )
    add_row(
        rows,
        "Reusable raw rollout to paper hybrid-state builder is implemented and tested",
        "Training dataset builders must call a single tested helper that converts contiguous raw root/body simulator state into the paper 99-D yaw-centric hybrid state.",
        f"builder_available={reusable_hybrid_builder_available}, builder_tested={reusable_hybrid_builder_tested}",
        reusable_hybrid_builder_available and reusable_hybrid_builder_tested,
        [rel(CORE_STATE_HELPERS), rel(CORE_MATH_TEST)],
        "Implement/test raw rollout -> 99-D hybrid-state construction before rebuilding teacher rollout state-latent datasets.",
    )
    add_row(
        rows,
        "Current paper-contract state-latent dataset is not 160-D policy observation",
        f"Expected 99-D hybrid state or 163-D projected state; token dim should be {expected_original_token_dim} or {expected_projected_token_dim} with 32-D latent.",
        f"state_source={paper_meta.get('state_source')!r}, obs_dim={current_obs_dim}, token_dim={current_token_dim}",
        (not current_uses_policy_obs)
        and current_obs_dim in {expected_state_dim, expected_projected_state_dim}
        and current_token_dim in {expected_original_token_dim, expected_projected_token_dim},
        [rel(PAPER_CONTRACT_DATASET_JSON)],
        "Rebuild state-latent shards from raw simulator/root/body state, not from Stage-1 policy_obs.",
    )
    add_row(
        rows,
        "Teacher rollout collector source records fields needed for future paper hybrid-state shards",
        "Collector code should save root pose/twist and target-body world positions/velocities so future shards can build 99-D hybrid states.",
        f"collector_records_required_world_state={collector_records_required_world_state}",
        collector_records_required_world_state,
        [rel(TEACHER_ROLLOUT_COLLECTOR)],
        "Patch the teacher rollout collector before recollecting rollout shards.",
    )
    add_row(
        rows,
        "Teacher rollout shards contain raw world-state fields required to construct paper hybrid state",
        "Rollout shards must contain root pose/twist and target-body world positions/velocities.",
        f"inspected_shards={len(inspected_shards)}, missing_required_fields={shard_missing}, keys={[s.get('keys') for s in inspected_shards]}",
        bool(inspected_shards) and all(shard.get("has_required_world_state_fields") for shard in inspected_shards),
        [rel(PAPER_CONTRACT_TEACHER_JSON)] + [s["path"] for s in inspected_shards],
        "Modify teacher rollout collection to save root/body world states and velocities before building state-latent data.",
    )
    add_row(
        rows,
        "Paper-contract state-latent builder path requires raw hybrid state instead of policy_obs",
        "The paper-contract wrapper must force raw rollout -> hybrid/projected state and must not override state_source to policy_obs. A legacy resource-adjusted path may remain only if clearly labeled.",
        f"builder_has_legacy_policy_obs_path={builder_has_legacy_policy_obs_path}, builder_has_paper_hybrid_path={builder_has_paper_hybrid_path}, wrapper_requires_hybrid_state={wrapper_requires_hybrid_state}, wrapper_overrides_state_source_to_policy_obs={wrapper_overrides_to_policy_obs}",
        builder_has_paper_hybrid_path and wrapper_requires_hybrid_state and (not wrapper_overrides_to_policy_obs),
        [rel(STATE_LATENT_BUILDER), rel(PAPER_DATASET_WRAPPER)],
        "Replace policy_obs encoding path with explicit hybrid-state construction and record schema version.",
    )
    add_row(
        rows,
        "Window index excludes done/reset discontinuities and implements paper 5s rejection",
        "Samples should be accepted only when the robot remains stable for the 5s verification horizon; windows must not cross done/reset discontinuities.",
        f"done_count_in_source_shards={total_done_count}, window_count={total_window_count}, builder_has_done_rejection_filter={windows_filter_done_and_rejection}",
        total_done_count == 0 and windows_filter_done_and_rejection,
        [rel(PAPER_CONTRACT_DATASET_JSON), rel(STATE_LATENT_BUILDER), rel(PAPER_ROOT) + ":546"],
        "Implement accepted-episode filtering and reject/omit all windows crossing dones/resets before diffusion training.",
    )
    add_row(
        rows,
        "OU-noise collection and symmetry augmentation are recorded in the trainable dataset protocol",
        "Paper uses OU action perturbations for DAgger-like rollout data and sagittal symmetry augmentation for VAE/diffusion.",
        f"ou_collection_recorded={ou_collection_recorded}, symmetry_recorded={symmetry_recorded}",
        ou_collection_recorded and symmetry_recorded,
        [rel(PAPER_ROOT) + ":539", rel(PAPER_ROOT) + ":591", rel(STATE_LATENT_BUILDER)],
        "Add explicit OU perturbation metadata and symmetry augmentation outputs for train/val/test splits.",
    )
    add_row(
        rows,
        "Diffusion training scripts consume paper hybrid/projected state, not policy_obs",
        "Denoiser input should be state-latent tokens built from hybrid/projected state and VAE latent.",
        f"paper_transformer_reads_policy_obs={paper_transformer_reads_policy_obs}, resource_diffusion_has_legacy_policy_obs_fallback={resource_diffusion_has_legacy_policy_obs_fallback}",
        not diffusion_reads_policy_obs,
        [rel(RESOURCE_DIFFUSION), rel(PAPER_TRANSFORMER_DIFFUSION)],
        "Gate full diffusion training until scripts read corrected state-latent arrays instead of source_shard['policy_obs'].",
    )

    blocked_count = sum(1 for row in rows if row["status"] == "blocked")
    pass_count = sum(1 for row in rows if row["status"] == "pass")
    status = (
        "ok_state_latent_dataset_source_contract"
        if blocked_count == 0
        else "blocked_state_latent_dataset_source_uses_policy_obs_and_missing_rollout_state"
    )
    checks = {
        "paper_hybrid_state_required": paper_sources_have_patterns,
        "hybrid_schema_gate_available": schema.get("checks", {}).get("hybrid_state_dim_99") is True,
        "reusable_raw_rollout_to_hybrid_state_builder_available": reusable_hybrid_builder_available,
        "reusable_raw_rollout_to_hybrid_state_builder_tested": reusable_hybrid_builder_tested,
        "existing_dataset_uses_policy_obs": current_uses_policy_obs,
        "existing_dataset_has_corrected_hybrid_state": any(
            row["contract"] == "Current paper-contract state-latent dataset is not 160-D policy observation"
            and row["passed"]
            for row in rows
        ),
        "teacher_rollout_collector_records_required_world_state": collector_records_required_world_state,
        "teacher_rollout_shards_have_required_world_state": any(
            row["contract"] == "Teacher rollout shards contain raw world-state fields required to construct paper hybrid state"
            and row["passed"]
            for row in rows
        ),
        "builder_has_legacy_policy_obs_path": builder_has_legacy_policy_obs_path,
        "builder_has_paper_hybrid_path": builder_has_paper_hybrid_path,
        "paper_contract_wrapper_requires_hybrid_state": wrapper_requires_hybrid_state,
        "paper_contract_wrapper_overrides_policy_obs": wrapper_overrides_to_policy_obs,
        "paper_transformer_diffusion_reads_policy_obs": paper_transformer_reads_policy_obs,
        "resource_diffusion_has_legacy_policy_obs_fallback": resource_diffusion_has_legacy_policy_obs_fallback,
        "windows_filter_done_and_5s_rejection": windows_filter_done_and_rejection and total_done_count == 0,
        "ou_noise_collection_recorded": ou_collection_recorded,
        "symmetry_augmentation_recorded": symmetry_recorded,
        "blocks_downstream_training": blocked_count > 0,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "pretraining_contract_audit",
        "row_count": len(rows),
        "pass_count": pass_count,
        "blocked_count": blocked_count,
        "checks": checks,
        "expected_schema": {
            "hybrid_state_dim": expected_state_dim,
            "projected_state_dim": expected_projected_state_dim,
            "latent_dim": expected_latent_dim,
            "original_state_latent_token_dim": expected_original_token_dim,
            "projected_state_latent_token_dim": expected_projected_token_dim,
        },
        "observed_dataset": {
            "paper_contract_state_source": paper_meta.get("state_source"),
            "paper_contract_obs_dim": current_obs_dim,
            "paper_contract_token_dim": current_token_dim,
            "paper_contract_window_count": total_window_count,
            "paper_contract_done_count_in_source_shards": total_done_count,
            "stage1_multisource_state_source": stage1_meta.get("state_source"),
            "stage1_multisource_obs_dim": stage1_meta.get("obs_dim"),
            "stage1_multisource_token_dim": stage1_meta.get("token_dim"),
        },
        "teacher_rollout_shard_probe": inspected_shards,
        "permission": {
            "start_downstream_vae_training": False,
            "start_state_latent_diffusion_training": False,
            "start_guided_closed_loop_video_generation": False,
            "use_existing_policy_obs_state_latent_dataset_for_long_training": False,
            "allowed_next_step": "modify teacher rollout collection to save raw root/body world state, then rebuild continuous accepted 99-D hybrid-state windows",
        },
        "interpretation": {
            "goal_complete": False,
            "claim_level": "pre-training source-contract audit only; current trainable state-latent data is blocked",
            "why_this_explains_bad_videos": (
                "The current downstream VAE/diffusion path learns from 160-D policy observations and windows that can include "
                "done/reset discontinuities, while the paper diffusion model is defined over continuous accepted hybrid "
                "state-latent trajectories. This mismatch can collapse the learned control chain toward a weak forward-leaning "
                "posture even when a reference pose replay looks reasonable."
            ),
        },
        "outputs": {
            "json": str(OUT / "beyondmimic_state_latent_dataset_source_contract_audit.json"),
            "tsv": str(OUT / "beyondmimic_state_latent_dataset_source_contract_audit.tsv"),
            "markdown": str(OUT / "beyondmimic_state_latent_dataset_source_contract_audit.md"),
        },
        "rows": rows,
    }

    json_path = OUT / "beyondmimic_state_latent_dataset_source_contract_audit.json"
    tsv_path = OUT / "beyondmimic_state_latent_dataset_source_contract_audit.tsv"
    md_path = OUT / "beyondmimic_state_latent_dataset_source_contract_audit.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, rows)
    md_lines = [
        "# BeyondMimic State-Latent Dataset Source Contract Audit",
        "",
        f"- Status: `{status}`",
        f"- Rows: `{len(rows)}` pass `{pass_count}` blocked `{blocked_count}`",
        f"- Current paper-contract dataset state source: `{paper_meta.get('state_source')}`",
        f"- Current dims: obs/state `{current_obs_dim}`, token `{current_token_dim}`",
        f"- Expected dims: hybrid state `{expected_state_dim}`, projected state `{expected_projected_state_dim}`, token `{expected_original_token_dim}` or `{expected_projected_token_dim}` with latent `{expected_latent_dim}`",
        f"- Teacher rollout missing fields: `{shard_missing}`",
        f"- Permission: `{json.dumps(summary['permission'], sort_keys=True)}`",
        "",
        "## Blocking Rows",
    ]
    for row in rows:
        if row["status"] != "pass":
            md_lines.extend(
                [
                    "",
                    f"### {row['contract']}",
                    f"- Observed: {row['observed_in_current_project']}",
                    f"- Required fix: {row['required_fix_before_long_training']}",
                ]
            )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "json": str(json_path), "blocked_count": blocked_count}, sort_keys=True))


if __name__ == "__main__":
    main()
