#!/usr/bin/env python3
"""Audit that downstream training entrypoints obey the current hard gates."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/audits/training_entrypoint_hard_gate"

UTIL = ROOT / "reproduction/scripts/beyondmimic_training_hard_gate_utils.py"
VAE_WRAPPER = ROOT / "reproduction/scripts/level_c_official_importer_export_paper_contract_teacher_rollout_vae_training.py"
DIFFUSION_WRAPPER = (
    ROOT / "reproduction/scripts/level_c_official_importer_export_paper_contract_state_latent_diffusion_training.py"
)
GUIDANCE_WRAPPER = (
    ROOT / "reproduction/scripts/level_c_official_importer_export_paper_contract_state_latent_guidance_eval.py"
)
TRANSFORMER = ROOT / "reproduction/scripts/level_c_paper_contract_transformer_state_latent_diffusion_training.py"

VAE_JSON = (
    ROOT
    / "res/level_c/official_importer_export_paper_contract_teacher_rollout_vae_training/"
    "level_c_official_importer_export_paper_contract_teacher_rollout_vae_training.json"
)
DIFFUSION_JSON = (
    ROOT
    / "res/level_c/official_importer_export_paper_contract_state_latent_diffusion_training/"
    "level_c_official_importer_export_paper_contract_state_latent_diffusion_training.json"
)
GUIDANCE_JSON = (
    ROOT
    / "res/level_c/official_importer_export_paper_contract_state_latent_guidance_eval/"
    "level_c_official_importer_export_paper_contract_state_latent_guidance_eval.json"
)
TRANSFORMER_JSON = (
    ROOT
    / "res/level_c/paper_contract_transformer_state_latent_diffusion_training/"
    "paper_contract_transformer_state_latent_diffusion_training.json"
)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def add_row(rows: list[dict[str, Any]], name: str, expected: str, observed: str, passed: bool, evidence: list[Path]) -> None:
    rows.append(
        {
            "name": name,
            "expected": expected,
            "observed": observed,
            "passed": bool(passed),
            "status": "pass" if passed else "blocked",
            "evidence": [rel(path) for path in evidence],
        }
    )


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "status", "passed", "expected", "observed", "evidence"],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["evidence"] = "; ".join(row["evidence"])
            writer.writerow(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    util_text = read_text(UTIL)
    vae_text = read_text(VAE_WRAPPER)
    diffusion_text = read_text(DIFFUSION_WRAPPER)
    guidance_text = read_text(GUIDANCE_WRAPPER)
    transformer_text = read_text(TRANSFORMER)
    summaries = {
        "vae": read_json(VAE_JSON),
        "diffusion": read_json(DIFFUSION_JSON),
        "guidance": read_json(GUIDANCE_JSON),
        "transformer": read_json(TRANSFORMER_JSON),
    }

    rows: list[dict[str, Any]] = []
    add_row(
        rows,
        "Shared training hard-gate utility exists",
        "Entrypoints should use one shared helper for pretraining permission and state-latent source checks.",
        f"utility_exists={UTIL.is_file()}, has_state_latent_dataset_block_reasons={'state_latent_dataset_block_reasons' in util_text}",
        UTIL.is_file()
        and "pretraining_permission_block_reasons" in util_text
        and "state_latent_dataset_block_reasons" in util_text
        and "write_blocked_summary" in util_text,
        [UTIL],
    )
    for label, path, text, permission in [
        ("vae", VAE_WRAPPER, vae_text, "start_downstream_vae_training"),
        ("diffusion", DIFFUSION_WRAPPER, diffusion_text, "start_state_latent_diffusion_training"),
        ("guidance", GUIDANCE_WRAPPER, guidance_text, "start_guided_closed_loop_video_generation"),
    ]:
        add_row(
            rows,
            f"{label} wrapper checks hard gate before worker launch",
            f"Wrapper must call enforce_hard_gate and requested permission `{permission}` before invoking the worker.",
            f"has_enforce={'enforce_hard_gate' in text}, has_permission={permission in text}",
            "enforce_hard_gate" in text and permission in text and "if not enforce_hard_gate()" in text,
            [path],
        )
    add_row(
        rows,
        "Transformer full training is blocked while dry-run remains code-contract only",
        "Transformer script should allow dry-run probes but block --full-train when state-latent hard gates fail.",
        f"has_full_train_guard={'if dry_run:' in transformer_text and 'blocked_paper_contract_transformer_diffusion_training_hard_gate' in transformer_text}",
        "enforce_hard_gate" in transformer_text
        and "if dry_run:" in transformer_text
        and "blocked_paper_contract_transformer_diffusion_training_hard_gate" in transformer_text,
        [TRANSFORMER],
    )

    expected_status = {
        "vae": "blocked_official_importer_export_paper_contract_teacher_rollout_vae_training_hard_gate",
        "diffusion": "blocked_official_importer_export_paper_contract_state_latent_diffusion_training_hard_gate",
        "guidance": "blocked_official_importer_export_paper_contract_state_latent_guidance_eval_hard_gate",
        "transformer": "blocked_paper_contract_transformer_diffusion_training_hard_gate",
    }
    expected_paths = {
        "vae": VAE_JSON,
        "diffusion": DIFFUSION_JSON,
        "guidance": GUIDANCE_JSON,
        "transformer": TRANSFORMER_JSON,
    }
    for label, status in expected_status.items():
        summary = summaries[label]
        add_row(
            rows,
            f"{label} latest output is blocked before worker launch",
            "After the guard smoke checks, output JSON should record blocked_before_worker_launch and does_not_start_training.",
            f"status={summary.get('status')!r}, checks={summary.get('checks')}",
            summary.get("status") == status
            and summary.get("checks", {}).get("blocked_before_worker_launch") is True
            and summary.get("checks", {}).get("does_not_start_training") is True,
            [expected_paths[label]],
        )

    blocked_count = sum(1 for row in rows if row["status"] == "blocked")
    pass_count = len(rows) - blocked_count
    status = "ok_training_entrypoint_hard_gate_audit" if blocked_count == 0 else "blocked_training_entrypoint_hard_gate_audit"
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "training_entrypoint_hard_gate_audit",
        "row_count": len(rows),
        "pass_count": pass_count,
        "blocked_count": blocked_count,
        "checks": {
            "all_training_entrypoints_guarded": blocked_count == 0,
            "does_not_start_training": True,
            "does_not_claim_goal_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "claim_level": "entrypoint guard audit only",
            "why_this_matters": (
                "The current project should not continue VAE/diffusion/guidance training from policy_obs-derived "
                "state-latent data. These guards turn the audit finding into executable protection."
            ),
        },
        "outputs": {
            "json": str(OUT / "beyondmimic_training_entrypoint_hard_gate_audit.json"),
            "tsv": str(OUT / "beyondmimic_training_entrypoint_hard_gate_audit.tsv"),
            "md": str(OUT / "beyondmimic_training_entrypoint_hard_gate_audit.md"),
        },
        "rows": rows,
    }
    json_path = OUT / "beyondmimic_training_entrypoint_hard_gate_audit.json"
    tsv_path = OUT / "beyondmimic_training_entrypoint_hard_gate_audit.tsv"
    md_path = OUT / "beyondmimic_training_entrypoint_hard_gate_audit.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_tsv(tsv_path, rows)
    md_path.write_text(
        "# BeyondMimic Training Entrypoint Hard-Gate Audit\n\n"
        f"- Status: `{status}`\n"
        f"- Rows: `{len(rows)}` pass `{pass_count}` blocked `{blocked_count}`\n"
        "- Result: downstream VAE/diffusion/guidance training entrypoints now block before worker launch while current hard gates fail.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "json": str(json_path), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
