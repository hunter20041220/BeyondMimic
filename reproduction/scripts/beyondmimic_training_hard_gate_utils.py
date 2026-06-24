#!/usr/bin/env python3
"""Shared hard-gate helpers for paper-contract training entrypoints."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
PRETRAINING_HARD_GATE_JSON = (
    ROOT / "res/audits/pretraining_hard_gate/beyondmimic_pretraining_hard_gate_audit.json"
)
STATE_LATENT_SOURCE_CONTRACT_JSON = (
    ROOT / "res/audits/state_latent_dataset_source_contract/beyondmimic_state_latent_dataset_source_contract_audit.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def pretraining_permission_block_reasons(permission_key: str) -> list[str]:
    gate = load_json(PRETRAINING_HARD_GATE_JSON)
    permission = gate.get("permission", {})
    reasons: list[str] = []
    if not gate:
        reasons.append(f"missing_pretraining_hard_gate:{PRETRAINING_HARD_GATE_JSON}")
    if permission.get(permission_key) is not True:
        reasons.append(f"pretraining_permission_{permission_key}_is_{permission.get(permission_key)!r}")
    if str(gate.get("status", "")).startswith("blocked"):
        reasons.append(f"pretraining_hard_gate_status={gate.get('status')}")
    return reasons


def state_latent_dataset_block_reasons(dataset_json: Path) -> list[str]:
    dataset_summary = load_json(dataset_json)
    source_contract = load_json(STATE_LATENT_SOURCE_CONTRACT_JSON)
    dataset_meta = dataset_summary.get("worker_summary", {}).get("dataset", {})
    checks = dataset_summary.get("checks", {})
    source = str(dataset_meta.get("state_source", "")).lower()
    state_dim = dataset_meta.get("state_dim", dataset_meta.get("obs_dim"))
    token_dim = dataset_meta.get("token_dim")
    reasons: list[str] = []
    if not dataset_summary:
        reasons.append(f"missing_state_latent_dataset:{dataset_json}")
    if source_contract.get("permission", {}).get("use_existing_policy_obs_state_latent_dataset_for_long_training") is False:
        reasons.append("source_contract_disallows_existing_policy_obs_state_latent_dataset")
    if str(source_contract.get("status", "")).startswith("blocked"):
        reasons.append(f"state_latent_source_contract_status={source_contract.get('status')}")
    if "policy_obs" in source:
        reasons.append(f"state_latent_dataset_uses_policy_obs:{dataset_meta.get('state_source')}")
    if checks.get("state_source_is_raw_hybrid_or_projected") is not True:
        reasons.append("state_source_is_not_raw_hybrid_or_projected")
    if checks.get("state_dim_matches_paper_contract") is not True:
        reasons.append(f"state_dim_not_paper_contract:{state_dim}")
    if checks.get("window_filter_rejects_discontinuities") is not True:
        reasons.append("window_filter_does_not_prove_done_reset_rejection")
    if state_dim not in {99, 163}:
        reasons.append(f"state_dim_not_99_or_163:{state_dim}")
    if token_dim not in {131, 195}:
        reasons.append(f"token_dim_not_state_plus_32_latent:{token_dim}")
    return reasons


def write_blocked_summary(
    *,
    json_path: Path,
    tsv_path: Path | None,
    status: str,
    experiment_type: str,
    permission_key: str,
    blocking_reasons: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": experiment_type,
        "blocking_reasons": blocking_reasons,
        "checks": {
            f"pretraining_permission_{permission_key}": False,
            "blocked_before_worker_launch": True,
            "does_not_start_training": True,
            "does_not_claim_goal_complete": True,
        },
        "permission": {
            "requested_permission": permission_key,
            "start_worker": False,
            "start_training": False,
        },
        "interpretation": {
            "goal_complete": False,
            "claim_level": "blocked training entrypoint guard",
            "why_not_complete": (
                "The requested training/evaluation entrypoint is blocked because the current project hard gates do "
                "not yet prove paper-contract teacher quality, adapter parity, and corrected hybrid state-latent data."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path) if tsv_path else None},
    }
    if extra:
        summary.update(extra)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if tsv_path is not None:
        tsv_path.parent.mkdir(parents=True, exist_ok=True)
        tsv_path.write_text(
            "status\tblocking_reasons\n" + f"{status}\t{'; '.join(blocking_reasons)}\n",
            encoding="utf-8",
        )
    return summary
