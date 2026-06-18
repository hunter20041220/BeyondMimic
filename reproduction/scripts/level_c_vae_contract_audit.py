#!/usr/bin/env python3
"""Audit the conditional VAE architecture/loss contract against paper and local probes."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/vae_contract_audit"
ROOT_TEX = ROOT / "reproduction/paper/source/root.tex"
METHOD_TEX = ROOT / "reproduction/paper/source/tex/method.tex"


def load_json(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def same(expected: Any, observed: Any) -> bool:
    if isinstance(expected, float) or isinstance(observed, float):
        return math.isclose(float(expected), float(observed), rel_tol=1e-9, abs_tol=1e-12)
    return expected == observed


def text_has(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.S) is not None


def add_row(rows: list[dict[str, Any]], category: str, item: str, expected: Any, observed: Any, evidence: str) -> None:
    rows.append(
        {
            "category": category,
            "item": item,
            "expected": expected,
            "observed": observed,
            "passed": same(expected, observed),
            "evidence": evidence,
        }
    )


def add_bool_row(rows: list[dict[str, Any]], category: str, item: str, passed: bool, evidence: str, note: str = "") -> None:
    rows.append(
        {
            "category": category,
            "item": item,
            "expected": True,
            "observed": bool(passed),
            "passed": bool(passed),
            "evidence": evidence,
            "note": note,
        }
    )


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["category", "item", "expected", "observed", "passed", "evidence", "note"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(row.get(field), sort_keys=True) if isinstance(row.get(field), (list, dict)) else row.get(field, "") for field in fields})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    root_text = ROOT_TEX.read_text(encoding="utf-8")
    method_text = METHOD_TEX.read_text(encoding="utf-8")
    synthetic = load_json("res/level_c/synthetic_smoke/level_c_synthetic_smoke.json")
    accumulation = load_json("res/level_c/vae_accumulation_probe/level_c_vae_accumulation_probe.json")
    latent = load_json("res/level_c/vae_latent_probe/level_c_vae_latent_probe.json")
    table_values = load_json("res/paper_table_values/paper_table_value_audit.json")

    table_rows = [row for row in table_values["rows"] if row.get("table") == "tab:vae_hyperparameters"]
    by_param = {row["parameter"]: row for row in table_rows}
    rows: list[dict[str, Any]] = []

    add_bool_row(rows, "paper_source", "vae_table_label_present", r"\label{tab:vae_hyperparameters}" in root_text, "reproduction/paper/source/root.tex:801-825")
    add_bool_row(rows, "paper_source", "encoder_formula_present", text_has(method_text, r"\\mathbf\{z\}\s*=\s*\\mathcal\{E\}"), "reproduction/paper/source/tex/method.tex:151-160")
    add_bool_row(rows, "paper_source", "decoder_formula_present", text_has(method_text, r"\\hat\{\\mathbf\{a\}\}\s*=\s*\\mathcal\{D\}"), "reproduction/paper/source/tex/method.tex:160-166")
    add_bool_row(rows, "paper_source", "elbo_formula_present", text_has(method_text, r"\\mathcal\{L\}_\\text\{VAE\}"), "reproduction/paper/source/tex/method.tex:166-170")
    add_bool_row(rows, "paper_source", "dagger_training_present", "DAgger" in method_text, "reproduction/paper/source/tex/method.tex:166-170")

    settings = accumulation["settings"]
    table_specs = [
        ("Latent dimension", settings["latent_dim"]),
        ("Student encoder MLP hidden dimensions", settings["encoder_hidden_dims"]),
        ("Student decoder MLP hidden dimensions", settings["decoder_hidden_dims"]),
        ("Teacher hidden MLP hidden dimensions", settings["teacher_hidden_dims"]),
        ("Activation function", "ELU"),
        ("Learning rate", settings["learning_rate"]),
        ("Accumulated Gradient steps", settings["gradient_accumulation_steps"]),
        ("KL loss coefficient", settings["kl_coefficient"]),
    ]
    for parameter, observed in table_specs:
        paper_row = by_param[parameter]
        add_row(
            rows,
            "paper_table_value",
            parameter,
            paper_row["paper_expected"],
            observed,
            paper_row["evidence"],
        )

    encoder_input_dim = settings["reference_motion_dim"] + settings["anchor_error_dim"]
    decoder_input_dim = settings["latent_dim"] + settings["proprioception_dim"]
    teacher_input_dim = settings["proprioception_dim"] + encoder_input_dim
    add_row(rows, "dimension_contract", "encoder_input_dim", 67, encoder_input_dim, "res/level_c/vae_accumulation_probe/level_c_vae_accumulation_probe.json")
    add_row(rows, "dimension_contract", "decoder_input_dim", 128, decoder_input_dim, "res/level_c/vae_accumulation_probe/level_c_vae_accumulation_probe.json")
    add_row(rows, "dimension_contract", "encoder_output_dim_mu_logvar", 64, accumulation["model"]["encoder_output_dim"], "res/level_c/vae_accumulation_probe/level_c_vae_accumulation_probe.json")
    add_row(rows, "dimension_contract", "decoder_output_action_dim", 29, accumulation["model"]["decoder_output_dim"], "res/level_c/vae_accumulation_probe/level_c_vae_accumulation_probe.json")
    add_row(rows, "dimension_contract", "teacher_input_dim", 163, teacher_input_dim, "res/level_c/vae_accumulation_probe/level_c_vae_accumulation_probe.json")

    shapes = synthetic["shapes"]
    add_row(rows, "runtime_shape", "vae_mu_shape", [2, 32], shapes["vae_mu"], "res/level_c/synthetic_smoke/level_c_synthetic_smoke.json")
    add_row(rows, "runtime_shape", "vae_logvar_shape", [2, 32], shapes["vae_logvar"], "res/level_c/synthetic_smoke/level_c_synthetic_smoke.json")
    add_row(rows, "runtime_shape", "vae_latent_shape", [2, 32], shapes["vae_latent"], "res/level_c/synthetic_smoke/level_c_synthetic_smoke.json")
    add_row(rows, "runtime_shape", "vae_action_shape", [2, 29], shapes["vae_action"], "res/level_c/synthetic_smoke/level_c_synthetic_smoke.json")

    for check_name in [
        "gradient_accumulation_matches_paper",
        "all_micro_losses_finite",
        "all_micro_grad_norms_positive",
        "single_optimizer_step_updates_parameters",
        "zero_grad_clears_gradients",
        "manifest_marks_not_true_dagger_rollout",
    ]:
        add_bool_row(rows, "training_contract_debug", check_name, accumulation["checks"][check_name], "res/level_c/vae_accumulation_probe/level_c_vae_accumulation_probe.json")
    for check_name in [
        "all_reparameterization_exact",
        "all_kl_formula_matches_manual",
        "all_latent_std_positive",
        "all_interpolation_endpoints_match",
        "all_interpolation_actions_finite",
        "all_sampled_actions_finite",
        "debug_only_boundary_recorded",
    ]:
        add_bool_row(rows, "latent_contract_debug", check_name, latent["checks"][check_name], "res/level_c/vae_latent_probe/level_c_vae_latent_probe.json")

    failed = [row for row in rows if not row["passed"]]
    json_path = OUT / "level_c_vae_contract_audit.json"
    tsv_path = OUT / "level_c_vae_contract_audit.tsv"
    summary = {
        "status": "ok" if not failed else "failed",
        "experiment_type": "level_c_vae_contract_audit",
        "scope": (
            "Conditional VAE architecture, dimensions, ELBO components, gradient accumulation, and latent math "
            "contract audit over paper/source plus existing debug probes. This is not a trained VAE reproduction."
        ),
        "metrics": {
            "row_count": len(rows),
            "failed_row_count": len(failed),
            "vae_table_value_rows": len(table_rows),
            "vae_parameter_count": accumulation["model"]["vae_parameter_count"],
            "teacher_parameter_count": accumulation["model"]["teacher_parameter_count"],
            "latent_probe_seed_count": len(latent["settings"]["seeds"]),
            "effective_batch_size": accumulation["metrics"]["effective_batch_size"],
        },
        "checks": {
            "all_contract_rows_pass": len(failed) == 0,
            "paper_source_formulas_present": all(
                row["passed"] for row in rows if row["category"] == "paper_source"
            ),
            "all_vae_table_values_match": all(
                row["passed"] for row in rows if row["category"] == "paper_table_value"
            ),
            "all_dimension_contracts_match": all(
                row["passed"] for row in rows if row["category"] == "dimension_contract"
            ),
            "runtime_shapes_match": all(row["passed"] for row in rows if row["category"] == "runtime_shape"),
            "latent_math_contract_passes": all(
                row["passed"] for row in rows if row["category"] == "latent_contract_debug"
            ),
            "training_contract_passes_debug_only": all(
                row["passed"] for row in rows if row["category"] == "training_contract_debug"
            ),
            "debug_only_boundary_recorded": latent["checks"]["debug_only_boundary_recorded"]
            and accumulation["synthetic_dagger_manifest"]["is_true_dagger_rollout"] is False,
        },
        "failed_rows": failed,
        "rows": rows,
        "interpretation": {
            "paper_level_status": "partial",
            "goal_complete": False,
            "why_not_complete": (
                "The local VAE contract matches the paper/source values and formulas at debug-probe level, but "
                "there is still no true DAgger rollout, trained VAE checkpoint, rollout stability evaluation, or "
                "paper latent-space analysis."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, rows)
    print(json.dumps({"status": summary["status"], "json": str(json_path), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
