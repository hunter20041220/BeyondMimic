#!/usr/bin/env python3
"""Audit local diffusion Transformer parameter counts against the paper's ~19.8M statement."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/transformer_parameter_count_audit"
ROOT_TEX = ROOT / "reproduction/paper/source/root.tex"


def load_json(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def find_line(path: Path, pattern: str) -> int | None:
    rx = re.compile(pattern)
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if rx.search(line):
            return idx
    return None


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "variant",
        "state_dim",
        "latent_dim",
        "token_dim",
        "parameter_count",
        "paper_reference_count",
        "absolute_delta",
        "relative_delta",
        "passed_local_arch_checks",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def local_arch_ok(data: dict[str, Any], *, require_paper_state: bool) -> bool:
    checks = data["checks"]
    required = [
        checks["uses_paper_embedding_dim"],
        checks["uses_paper_attention_heads"],
        checks["uses_paper_transformer_layers"],
        checks["uses_paper_denoising_steps"],
        checks["uses_paper_history_horizon"],
        checks["step_tensor_is_independent_state_latent"],
        checks["prediction_shape_matches_clean_tau"],
        checks["loss_is_finite"],
        checks["grad_norm_is_positive"],
    ]
    if require_paper_state:
        required.extend([checks["uses_paper_state_dim_99"], checks["uses_token_dim_131"]])
    return bool(all(required))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full = load_json("res/level_c/full_transformer_arch_probe/level_c_full_transformer_arch_probe.json")
    paper_state = load_json(
        "res/level_c/paper_state_transformer_arch_probe/level_c_paper_state_transformer_arch_probe.json"
    )
    root_text = ROOT_TEX.read_text(encoding="utf-8")
    paper_reference_count = 19_800_000
    paper_line = find_line(ROOT_TEX, r"parameter count.*19\.8M")

    variants = [
        (
            "fixture_181d_state_tau213",
            full,
            False,
            "Older debug fixture state uses 181-D candidate token plus 32-D synthetic latent.",
        ),
        (
            "paper_state_99d_tau131",
            paper_state,
            True,
            "Paper-formula 99-D state windows plus 32-D synthetic latent.",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for name, data, require_paper_state, notes in variants:
        parameter_count = int(data["model"]["parameter_count"])
        delta = parameter_count - paper_reference_count
        rows.append(
            {
                "variant": name,
                "state_dim": int(data["settings"]["state_dim"]),
                "latent_dim": int(data["settings"]["latent_dim"]),
                "token_dim": int(data["settings"]["token_dim"]),
                "parameter_count": parameter_count,
                "paper_reference_count": paper_reference_count,
                "absolute_delta": delta,
                "relative_delta": float(delta / paper_reference_count),
                "passed_local_arch_checks": local_arch_ok(data, require_paper_state=require_paper_state),
                "notes": notes,
            }
        )

    token_dim_delta = rows[0]["token_dim"] - rows[1]["token_dim"]
    parameter_count_delta = rows[0]["parameter_count"] - rows[1]["parameter_count"]
    expected_delta_from_io_projection = token_dim_delta * (512 + 1) + token_dim_delta * 512
    json_path = OUT / "level_c_transformer_parameter_count_audit.json"
    tsv_path = OUT / "level_c_transformer_parameter_count_audit.tsv"
    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "level_c_transformer_parameter_count_audit",
        "scope": "Diffusion Transformer parameter-count source and local architecture-boundary audit",
        "paper_evidence": {
            "root_tex": str(ROOT_TEX),
            "line": paper_line,
            "parameter_count_statement_found": "parameter count of \\~19.8M" in root_text
            or "parameter count of \\~19.8M" in root_text.replace(" ", ""),
            "diffusion_hyperparameter_table": str(ROOT / "reproduction/paper/source/root.tex:827-856"),
        },
        "paper_reference": {
            "parameter_count_text": "~19.8M",
            "parameter_count_numeric_for_audit": paper_reference_count,
            "is_approximate_statement": True,
        },
        "rows": rows,
        "metrics": {
            "variant_count": len(rows),
            "paper_reference_count": paper_reference_count,
            "fixture_181d_parameter_count": rows[0]["parameter_count"],
            "paper_state_99d_parameter_count": rows[1]["parameter_count"],
            "fixture_relative_delta_vs_paper": rows[0]["relative_delta"],
            "paper_state_relative_delta_vs_paper": rows[1]["relative_delta"],
            "token_dim_delta": token_dim_delta,
            "parameter_count_delta_between_local_variants": parameter_count_delta,
            "expected_delta_from_input_output_projection": expected_delta_from_io_projection,
        },
        "checks": {
            "paper_parameter_count_statement_found": paper_line is not None,
            "all_local_architecture_checks_pass": all(row["passed_local_arch_checks"] for row in rows),
            "both_local_counts_within_5_percent_of_paper_approx": all(
                abs(row["relative_delta"]) < 0.05 for row in rows
            ),
            "local_counts_not_exact_paper_count": any(row["absolute_delta"] != 0 for row in rows),
            "variant_delta_explained_by_token_io_projection": parameter_count_delta
            == expected_delta_from_io_projection,
            "paper_state_token_dim_131_recorded": rows[1]["token_dim"] == 131,
            "does_not_claim_exact_paper_parameter_count": True,
        },
        "interpretation": {
            "paper_level_status": "partial",
            "goal_complete": False,
            "why_not_complete": (
                "The local Transformer probes match the published embedding/head/layer/step hyperparameters and are "
                "within 5 percent of the paper's approximate ~19.8M statement, but neither local count exactly matches "
                "19.8M. The difference likely reflects unpublished implementation details and token/input-output "
                "projection dimensions, so exact paper checkpoint architecture is not claimed."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, rows)
    print(json.dumps({"status": summary["status"], "json": str(json_path), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
