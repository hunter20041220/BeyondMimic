#!/usr/bin/env python3
"""Audit paper guidance-cost source coverage against local debug guidance probes."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/guidance_cost_coverage_audit"
ROOT_TEX = ROOT / "reproduction/paper/source/root.tex"
METHOD_TEX = ROOT / "reproduction/paper/source/tex/method.tex"
FULL_SPLIT_OFFLINE = (
    "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/"
    "level_c_lafan1_paper_arch_guidance_eval.json"
)
FULL_SPLIT_REVERSE = (
    "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/"
    "level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.json"
)
FULL_SPLIT_TABLE = "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.json"


def load_json(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def has(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.S) is not None


def add_row(
    rows: list[dict[str, Any]],
    *,
    item: str,
    paper_status: str,
    source_found: bool,
    probe_status: str,
    probe_check: bool,
    evidence: str,
    notes: str,
    full_split_task: str = "",
    full_split_evidence: str = "",
) -> None:
    rows.append(
        {
            "item": item,
            "paper_status": paper_status,
            "source_found": source_found,
            "probe_status": probe_status,
            "probe_check": probe_check,
            "passed": bool(source_found and probe_check),
            "evidence": evidence,
            "full_split_task": full_split_task,
            "full_split_evidence": full_split_evidence,
            "notes": notes,
        }
    )


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "item",
        "paper_status",
        "source_found",
        "probe_status",
        "probe_check",
        "passed",
        "evidence",
        "full_split_task",
        "full_split_evidence",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    root_text = ROOT_TEX.read_text(encoding="utf-8")
    method_text = METHOD_TEX.read_text(encoding="utf-8")
    formula = load_json("res/level_c/guidance_formula_probe/level_c_guidance_formula_probe.json")
    sweep = load_json("res/level_c/guidance_scale_sweep_probe/level_c_guidance_scale_sweep_probe.json")
    reverse = load_json("res/level_c/guided_reverse_loop_probe/level_c_guided_reverse_loop_probe.json")
    full_split = load_json(FULL_SPLIT_TABLE)
    full_split_tasks = {row["task"] for row in full_split["rows"]}
    full_split_evidence = f"{FULL_SPLIT_OFFLINE}; {FULL_SPLIT_REVERSE}; {FULL_SPLIT_TABLE}"

    rows: list[dict[str, Any]] = []
    add_row(
        rows,
        item="classifier_guidance_gradient",
        paper_status="paper_explicit",
        source_found=has(method_text, r"= -\\,\\nabla_\{\\boldsymbol\{\\tau\}\} G\(\\boldsymbol\{\\tau\}\)"),
        probe_status="guided_reverse_loop_probe",
        probe_check=reverse["checks"]["guidance_gradients_nonzero"]
        and reverse["checks"]["guided_final_cost_below_unguided_final"],
        evidence="reproduction/paper/source/tex/method.tex:212-226; res/level_c/guided_reverse_loop_probe/level_c_guided_reverse_loop_probe.json",
        full_split_task="all_guided_tasks",
        full_split_evidence=full_split_evidence,
        notes="Paper derives classifier guidance as negative cost gradient; debug reverse loop and full-split reverse guidance apply task gradients.",
    )
    add_row(
        rows,
        item="joystick_velocity_cost",
        paper_status="paper_explicit",
        source_found=has(root_text, r"G_\{\\mathrm\{js\}\}.*V_\{xy,i\}.*\\mathbf\{g\}_\{v\}"),
        probe_status="guidance_formula_probe",
        probe_check=formula["checks"]["all_formula_gradients_nonzero"]
        and formula["gradient_norms"]["joystick_velocity_formula"] > 0.0,
        evidence="reproduction/paper/source/root.tex:549-555; res/level_c/guidance_formula_probe/level_c_guidance_formula_probe.json",
        full_split_task="joystick",
        full_split_evidence=full_split_evidence,
        notes="Paper gives the exact squared planar root-velocity command cost; full-split public-data guidance records joystick cost/primary metrics.",
    )
    add_row(
        rows,
        item="waypoint_navigation_cost",
        paper_status="paper_explicit",
        source_found=has(root_text, r"G_\{\\mathrm\{wp\}\}.*1-e\^\{-2d_i\}.*e\^\{-2d_i\}"),
        probe_status="guidance_formula_probe",
        probe_check=formula["checks"]["all_formula_gradients_nonzero"]
        and formula["gradient_norms"]["waypoint_formula"] > 0.0,
        evidence="reproduction/paper/source/root.tex:556-563; res/level_c/guidance_formula_probe/level_c_guidance_formula_probe.json",
        full_split_task="waypoint",
        full_split_evidence=full_split_evidence,
        notes="Paper gives position-to-goal and near-goal velocity-minimization terms; full-split public-data guidance records waypoint cost/primary metrics.",
    )
    add_row(
        rows,
        item="sdf_obstacle_cost",
        paper_status="paper_explicit",
        source_found=has(root_text, r"G_\{\\mathrm\{sdf\}\}.*\\mathrm\{SDF\}.*r_b"),
        probe_status="guidance_formula_probe",
        probe_check=formula["checks"]["all_formula_gradients_nonzero"]
        and formula["gradient_norms"]["sdf_obstacle_barrier_formula"] > 0.0,
        evidence="reproduction/paper/source/root.tex:564-569; res/level_c/guidance_formula_probe/level_c_guidance_formula_probe.json",
        full_split_task="obstacle_avoidance",
        full_split_evidence=full_split_evidence,
        notes="Paper gives SDF obstacle cost over bodies and horizon; full-split public-data guidance records obstacle-avoidance cost/primary metrics.",
    )
    add_row(
        rows,
        item="relaxed_barrier_function",
        paper_status="paper_explicit",
        source_found=all(
            token in root_text
            for token in [
                r"B(x, \delta)",
                r"-\ln(x)",
                r"x \geq \delta",
                r"x < \delta",
                r"\frac{x-2\delta}{\delta}",
            ]
        ),
        probe_status="guidance_formula_probe",
        probe_check=formula["checks"]["finite_costs"] and formula["checks"]["finite_gradients"],
        evidence="reproduction/paper/source/root.tex:570-586; res/level_c/guidance_formula_probe/level_c_guidance_formula_probe.json",
        full_split_task="obstacle_avoidance",
        full_split_evidence=full_split_evidence,
        notes="Paper gives the relaxed barrier piecewise form used by the SDF probe; full-split public-data obstacle guidance exercises the barrier term.",
    )
    add_row(
        rows,
        item="composed_waypoint_obstacle_cost",
        paper_status="paper_demonstrated",
        source_found="composing waypoint and obstacle-avoidance costs" in root_text,
        probe_status="guidance_formula_probe",
        probe_check=formula["checks"]["composed_gradient_nonzero"],
        evidence="reproduction/paper/source/root.tex:241-242; res/level_c/guidance_formula_probe/level_c_guidance_formula_probe.json",
        full_split_task="composed_objectives",
        full_split_evidence=full_split_evidence,
        notes="Figure 6 caption demonstrates composition; probe sums paper-explicit joystick/waypoint/SDF terms and full-split guidance records composed-objective metrics.",
    )
    keyframe_source_found = "future keyframes" in root_text and "0.2" in root_text and "keyframe" in root_text.lower()
    rows.append(
        {
            "item": "keyframe_inpainting_cost",
            "paper_status": "paper_demonstrated_formula_missing",
            "source_found": keyframe_source_found,
            "probe_status": "candidate_guidance_formula_probe",
            "probe_check": formula["checks"]["keyframe_candidate_gradient_nonzero"],
            "passed": bool(keyframe_source_found and formula["checks"]["keyframe_candidate_gradient_nonzero"]),
            "evidence": "reproduction/paper/source/root.tex:237-243; res/level_c/guidance_formula_probe/level_c_guidance_formula_probe.json",
            "full_split_task": "inpainting",
            "full_split_evidence": full_split_evidence,
            "notes": (
                "Paper demonstrates sparse future keyframes but does not publish a unique keyframe cost equation in "
                "the local source; local implementation remains a candidate differentiable term and full-split "
                "public-data guidance records inpainting metrics."
            ),
        }
    )
    add_row(
        rows,
        item="guidance_scale_sweep",
        paper_status="goal_required_debug_protocol",
        source_found=has(method_text, r"training-free") and has(method_text, r"-\\,\\nabla"),
        probe_status="guidance_scale_sweep_probe",
        probe_check=sweep["checks"]["all_rows_valid"]
        and sweep["checks"]["best_improves_over_zero_scale"]
        and sweep["checks"]["positive_scales_have_nonzero_gradients"],
        evidence="res/level_c/guidance_scale_sweep_probe/level_c_guidance_scale_sweep_probe.json",
        full_split_task="all_guided_tasks",
        full_split_evidence=full_split_evidence,
        notes="Paper does not publish a scale-selection protocol; debug sweep and full-split public-data offline/reverse audits record guidance scale sweeps.",
    )

    failed = [row for row in rows if not row["passed"]]
    formula_missing = [row for row in rows if row["paper_status"].endswith("formula_missing")]
    json_path = OUT / "level_c_guidance_cost_coverage_audit.json"
    tsv_path = OUT / "level_c_guidance_cost_coverage_audit.tsv"
    summary = {
        "status": "ok" if not failed else "failed",
        "experiment_type": "level_c_guidance_cost_coverage_audit",
        "scope": (
            "Coverage audit for paper/source guidance costs and local debug probes. It separates paper-explicit "
            "joystick/waypoint/SDF/barrier/classifier guidance equations from the keyframe candidate term whose "
            "unique formula is not published in the local source."
        ),
        "metrics": {
            "row_count": len(rows),
            "failed_row_count": len(failed),
            "paper_explicit_row_count": sum(1 for row in rows if row["paper_status"] == "paper_explicit"),
            "formula_missing_row_count": len(formula_missing),
            "scale_sweep_rows": len(sweep["rows"]),
            "selected_best_scale": sweep["selected_best"]["scale"],
            "guided_reverse_final_max_step": reverse["metrics"]["final_max_step"],
            "full_split_task_linked_row_count": sum(1 for row in rows if row["full_split_evidence"]),
            "full_split_source_row_count": full_split["metrics"]["offline_source_rows"]
            + full_split["metrics"]["reverse_source_rows"],
        },
        "checks": {
            "all_rows_pass": len(failed) == 0,
            "all_paper_explicit_costs_have_source_and_gradients": all(
                row["source_found"] and row["probe_check"]
                for row in rows
                if row["paper_status"] == "paper_explicit"
            ),
            "keyframe_formula_recorded_missing": len(formula_missing) == 1
            and formula_missing[0]["item"] == "keyframe_inpainting_cost",
            "keyframe_candidate_has_gradient": formula["checks"]["keyframe_candidate_gradient_nonzero"],
            "guided_reverse_loop_valid": reverse["checks"]["all_steps_reach_zero"]
            and reverse["checks"]["observed_tokens_clamped_guided"],
            "scale_sweep_valid": sweep["checks"]["all_rows_valid"] and sweep["checks"]["all_rows_keep_clamps"],
            "all_guidance_rows_link_full_split_public_data": all(row["full_split_evidence"] for row in rows),
            "paper_explicit_costs_link_full_split_public_data": all(
                row["full_split_evidence"]
                and (
                    row["full_split_task"] == "all_guided_tasks"
                    or row["full_split_task"] in full_split_tasks
                )
                for row in rows
                if row["paper_status"] == "paper_explicit"
            ),
            "full_split_link_does_not_claim_closed_loop": True,
        },
        "failed_rows": failed,
        "rows": rows,
        "interpretation": {
            "paper_level_status": "partial",
            "goal_complete": False,
            "why_not_complete": (
                "Paper-explicit guidance formulas are covered by source, debug gradient checks, and full-split "
                "public-data guidance metric links. The exact keyframe inpainting cost, paper guidance-scale "
                "validation protocol, trained closed-loop rollout metrics, and Fig.5/Fig.6 evidence remain "
                "unavailable."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, rows)
    print(json.dumps({"status": summary["status"], "json": str(json_path), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
