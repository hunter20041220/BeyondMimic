#!/usr/bin/env python3
"""Audit the paper S3 hybrid state schema before downstream training.

This gate is narrower than the full code/formula audit: it verifies that the
local reusable state helpers now encode the paper's 99-D hybrid state layout and
15-D root emphasis projection.  It deliberately keeps long VAE/diffusion
training blocked until a trainable rollout dataset is rebuilt with this schema.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SRC = ROOT / "reproduction/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from beyondmimic_reimpl.state import (  # noqa: E402
    HYBRID_STATE_DIM,
    ROOT_STATE_DIM,
    TARGET_BODY_FEATURE_DIM,
    emphasis_projection,
    hybrid_state_schema,
    project_hybrid_state,
    unproject_hybrid_state,
    validate_hybrid_state,
)

OUT = ROOT / "res/audits/hybrid_state_schema_contract"
JSON_OUT = OUT / "beyondmimic_hybrid_state_schema_contract_audit.json"
TSV_OUT = OUT / "beyondmimic_hybrid_state_schema_contract_audit.tsv"
MD_OUT = OUT / "beyondmimic_hybrid_state_schema_contract_audit.md"

PAPER_ROOT = ROOT / "reproduction/paper/source/root.tex"
PAPER_METHOD = ROOT / "reproduction/paper/source/tex/method.tex"
CODE_FORMULA_AUDIT = ROOT / "res/audits/code_formula_appendix_contract/beyondmimic_code_formula_appendix_contract_audit.json"
STATE_DATASET_AUDIT = ROOT / "res/level_c/state_latent_dataset_consistency_audit/level_c_state_latent_dataset_consistency_audit.json"
STAGE1_DATASET_AUDIT = ROOT / (
    "res/level_c/stage1_multisource_teacher_rollout_state_latent_dataset/"
    "level_c_stage1_multisource_teacher_rollout_state_latent_dataset.json"
)


def read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def line_ref(path: Path, needle: str) -> str:
    text = read_text(path)
    for idx, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return f"{path}:{idx}"
    return f"{path}:not_found:{needle}"


def row(
    contract: str,
    expected: str,
    observed: str,
    status: str,
    evidence: list[str],
    required_fix: str,
) -> dict[str, Any]:
    return {
        "contract": contract,
        "expected_from_paper_or_gate": expected,
        "observed_in_current_project": observed,
        "status": status,
        "passed": status == "pass",
        "evidence": evidence,
        "required_fix_before_long_training": required_fix,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    schema = hybrid_state_schema()
    p, p_inv = emphasis_projection(seed=123)
    rng = np.random.default_rng(1234)
    states = rng.normal(size=(5, 7, schema.state_dim))
    projected, projection, projection_inverse = project_hybrid_state(states, seed=123, schema=schema)
    recovered = unproject_hybrid_state(projected, projection_inverse, schema)
    roundtrip_error = float(np.max(np.abs(recovered - states)))
    dimension_error = ""
    try:
        validate_hybrid_state(np.zeros((2, 160)), schema)
    except ValueError as exc:
        dimension_error = str(exc)

    code_formula = read_json(CODE_FORMULA_AUDIT)
    state_dataset = read_json(STATE_DATASET_AUDIT)
    stage1_dataset = read_json(STAGE1_DATASET_AUDIT)

    paper_patterns_found = all(
        pattern in read_text(PAPER_ROOT) + read_text(PAPER_METHOD)
        for pattern in [
            "hybrid trajectory representation",
            "root features",
            "body features",
            "pseudoinverse",
            "c = 6",
        ]
    )
    trainable_dataset_already_gate_passed = bool(
        code_formula.get("checks", {}).get("state_latent_uses_hybrid_state", False)
    )
    existing_dataset_debug_only = bool(
        state_dataset.get("interpretation", {}).get("goal_complete") is False
        or stage1_dataset.get("interpretation", {}).get("goal_complete") is False
    )

    rows = [
        row(
            "Paper S3 hybrid state source is readable",
            "Paper states root features in current yaw frame and body features in local root frames.",
            f"patterns_found={paper_patterns_found}",
            "pass" if paper_patterns_found else "blocked",
            [
                line_ref(PAPER_ROOT, "This hybrid trajectory representation"),
                line_ref(PAPER_METHOD, "hybrid character"),
            ],
            "Restore paper source files before claiming formula-level state alignment.",
        ),
        row(
            "Root feature dimension",
            "3 root position + 6 rot6d + 3 linear velocity + 3 angular velocity = 15.",
            f"root_dim={schema.root_dim}",
            "pass" if schema.root_dim == 15 and ROOT_STATE_DIM == 15 else "mismatch",
            [str(SRC / "beyondmimic_reimpl/state.py")],
            "Keep root_dim=15 in reusable helpers and downstream training scripts.",
        ),
        row(
            "Target body feature dimension",
            "14 target bodies * (3 local position + 3 local velocity) = 84.",
            f"target_body_count={schema.target_body_count}, body_feature_dim={schema.body_feature_dim}",
            "pass" if schema.target_body_count == 14 and schema.body_feature_dim == 84 else "mismatch",
            [str(SRC / "beyondmimic_reimpl/state.py")],
            "Use the same target-body set when rebuilding the trainable state-latent dataset.",
        ),
        row(
            "Hybrid state dimension",
            "15 root features + 84 target-body features = 99.",
            f"state_dim={schema.state_dim}",
            "pass" if schema.state_dim == 99 and HYBRID_STATE_DIM == 99 else "mismatch",
            [str(SRC / "beyondmimic_reimpl/state.py")],
            "Reject 160-D policy observations or other non-paper features before diffusion training.",
        ),
        row(
            "Emphasis projection dimension",
            "Projected dimension = 99 + 64 Gaussian emphasis rows = 163 with P shape [163, 99]; c=6 scales root features.",
            f"projection_shape={list(p.shape)}, inverse_shape={list(p_inv.shape)}",
            "pass" if p.shape == (163, 99) and p_inv.shape == (99, 163) else "mismatch",
            [str(SRC / "beyondmimic_reimpl/state.py")],
            "Do not use the previous 18-D-root / 207-D projected state default or a mistaken 6*root_dim row count.",
        ),
        row(
            "Projection pseudoinverse round-trip",
            "Projected states recover original 99-D states through P pseudoinverse.",
            f"max_abs_error={roundtrip_error:.3e}",
            "pass" if roundtrip_error < 1e-10 and projection.shape == (163, 99) else "mismatch",
            [str(JSON_OUT)],
            "Fix projection or inverse before training diffusion on projected states.",
        ),
        row(
            "Policy obs rejection",
            "Paper diffusion state must not silently accept 160-D tracking policy observations.",
            f"dimension_error={dimension_error}",
            "pass" if "160" in dimension_error and "99" in dimension_error else "mismatch",
            [str(SRC / "beyondmimic_reimpl/state.py")],
            "Keep shape validation at all state-latent dataset construction boundaries.",
        ),
        row(
            "Trainable state-latent dataset rebuilt with corrected schema",
            "Long downstream VAE/diffusion training should use freshly rebuilt 99-D hybrid states, not stale debug windows.",
            (
                f"code_formula_state_latent_uses_hybrid_state={trainable_dataset_already_gate_passed}, "
                f"existing_dataset_debug_only={existing_dataset_debug_only}"
            ),
            "blocked",
            [str(CODE_FORMULA_AUDIT), str(STATE_DATASET_AUDIT), str(STAGE1_DATASET_AUDIT)],
            (
                "Rebuild the trainable teacher rollout/state-latent dataset with root/body hybrid features, OU "
                "noise collection, 5s rejection, and symmetry augmentation before long VAE/diffusion training."
            ),
        ),
    ]

    status_counts: dict[str, int] = {}
    for item in rows:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
    checks = {
        "paper_sources_readable": paper_patterns_found,
        "root_state_dim_15": schema.root_dim == 15 and ROOT_STATE_DIM == 15,
        "body_feature_dim_84": schema.body_feature_dim == 84 and TARGET_BODY_FEATURE_DIM == 84,
        "hybrid_state_dim_99": schema.state_dim == 99,
        "emphasis_coefficient_6": schema.coefficient == 6,
        "gaussian_emphasis_rows_64": schema.gaussian_rows == 64,
        "projected_state_dim_163": schema.projected_dim == 163 and p.shape == (163, 99),
        "projection_pseudoinverse_roundtrip": roundtrip_error < 1e-10,
        "rejects_160d_policy_obs": "160" in dimension_error and "99" in dimension_error,
        "trainable_dataset_rebuilt_with_corrected_schema": trainable_dataset_already_gate_passed,
        "does_not_allow_long_training_yet": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "blocked_hybrid_state_schema_ready_but_trainable_dataset_missing",
        "experiment_type": "pretraining_contract_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "row_count": len(rows),
        "status_counts": status_counts,
        "schema": schema.to_dict(),
        "metrics": {
            "projection_shape": list(p.shape),
            "projection_inverse_shape": list(p_inv.shape),
            "roundtrip_max_abs_error": roundtrip_error,
            "projected_sample_shape": list(projected.shape),
        },
        "checks": checks,
        "permission": {
            "start_downstream_vae_training": False,
            "start_state_latent_diffusion_training": False,
            "start_guided_closed_loop_video_generation": False,
            "reason": (
                "The reusable schema/projection code is corrected, but the actual trainable rollout dataset still "
                "needs to be rebuilt and re-audited with this schema."
            ),
        },
        "rows": rows,
        "interpretation": {
            "goal_complete": False,
            "claim_level": "formula/schema helper alignment only; not a successful trainable diffusion dataset",
            "next_required_step": (
                "Rebuild teacher rollout state-latent windows from continuous MuJoCo/IsaacLab rollouts using the "
                "paper hybrid state representation, then rerun the code/formula appendix gate."
            ),
        },
        "outputs": {"json": str(JSON_OUT), "tsv": str(TSV_OUT), "markdown": str(MD_OUT)},
    }

    JSON_OUT.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with TSV_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "contract",
                "expected_from_paper_or_gate",
                "observed_in_current_project",
                "status",
                "passed",
                "required_fix_before_long_training",
                "evidence",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for item in rows:
            out = dict(item)
            out["evidence"] = json.dumps(item["evidence"], ensure_ascii=False)
            writer.writerow(out)
    lines = [
        "# Hybrid State Schema Contract Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Schema: `{json.dumps(summary['schema'], sort_keys=True)}`",
        f"- Checks: `{json.dumps(summary['checks'], sort_keys=True)}`",
        "",
        "## Required Fix Before Long Training",
        "",
        summary["interpretation"]["next_required_step"],
    ]
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(JSON_OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
