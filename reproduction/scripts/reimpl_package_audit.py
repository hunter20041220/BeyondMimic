#!/usr/bin/env python3
"""Audit the lightweight beyondmimic_reimpl package surface."""

from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SRC = ROOT / "reproduction/src"
OUT = ROOT / "res/code/reimpl_package_audit"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EXPECTED_APIS = {
    "beyondmimic_reimpl.geometry": ["anchor_to_world", "rot6d_to_matrix", "world_to_anchor", "yaw_matrix"],
    "beyondmimic_reimpl.sampling": ["adaptive_distribution", "mirror_state_29d", "ou_noise"],
    "beyondmimic_reimpl.diffusion": ["apply_observation_mask", "denoise_one_step_with_oracle_eps", "q_sample"],
    "beyondmimic_reimpl.vae": ["kl_standard_normal", "reparameterize"],
    "beyondmimic_reimpl.guidance": ["finite_difference_grad", "gaussian_reward", "sdf_barrier"],
    "beyondmimic_reimpl.state": ["emphasis_projection", "smoothness_penalty"],
    "beyondmimic_reimpl.dagger": ["build_dagger_sample", "teacher_student_discrepancy"],
    "beyondmimic_reimpl.trajectory": ["build_state_latent_window", "split_counts", "stack_state_latent_tokens"],
    "beyondmimic_reimpl.evaluation": [
        "action_mse",
        "fall_rate",
        "split_metric_summary",
        "success_rate",
        "survival_rate",
        "tracking_error",
        "velocity_tracking_error",
    ],
}


def audit_modules() -> tuple[list[dict[str, Any]], dict[str, bool]]:
    rows: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    for module_name, symbols in EXPECTED_APIS.items():
        try:
            module = importlib.import_module(module_name)
            imported = True
            error = ""
        except Exception as exc:  # noqa: BLE001 - audit records import failures.
            module = None
            imported = False
            error = str(exc)
        for symbol in symbols:
            exists = bool(imported and hasattr(module, symbol))
            rows.append(
                {
                    "module": module_name,
                    "symbol": symbol,
                    "imported": imported,
                    "exists": exists,
                    "error": error,
                }
            )
            checks[f"{module_name}:{symbol}"] = exists
    return rows, checks


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / "reproduction/src/beyondmimic_reimpl").rglob("*.py"))
    rows, symbol_checks = audit_modules()
    all_symbols_exist = all(symbol_checks.values())
    summary = {
        "status": "ok" if all_symbols_exist else "failed",
        "experiment_type": "code_audit",
        "scope": "lightweight BeyondMimic reimplementation package API",
        "source_root": str(SRC / "beyondmimic_reimpl"),
        "python_file_count": len(files),
        "modules_checked": sorted(EXPECTED_APIS),
        "symbol_row_count": len(rows),
        "failed_symbol_rows": [row for row in rows if not row["exists"]],
        "checks": {
            "package_has_python_files": len(files) >= 10,
            "all_expected_modules_import": all(row["imported"] for row in rows),
            "all_expected_symbols_exist": all_symbols_exist,
            "core_math_tests_use_package_api": "beyondmimic_reimpl" in (ROOT / "reproduction/tests/test_core_math.py").read_text(
                encoding="utf-8"
            ),
            "does_not_claim_official_training_implementation": True,
        },
        "rows": rows,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The package now exposes reusable formula-level utilities, but it is not the unpublished official "
                "BeyondMimic VAE/diffusion training code and does not contain trained checkpoints or deployment engines."
            ),
        },
        "outputs": {
            "json": str(OUT / "reimpl_package_audit.json"),
            "tsv": str(OUT / "reimpl_package_audit.tsv"),
        },
    }
    (OUT / "reimpl_package_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT / "reimpl_package_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["module", "symbol", "imported", "exists", "error"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
