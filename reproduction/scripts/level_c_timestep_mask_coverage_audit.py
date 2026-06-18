#!/usr/bin/env python3
"""Audit paper/source coverage for independent timestep and task-mask probes."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/timestep_mask_coverage_audit"
ROOT_TEX = ROOT / "reproduction/paper/source/root.tex"
METHOD_TEX = ROOT / "reproduction/paper/source/tex/method.tex"


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
    local_artifact: str,
    local_check: bool,
    notes: str,
) -> None:
    rows.append(
        {
            "item": item,
            "paper_status": paper_status,
            "source_found": bool(source_found),
            "local_artifact": local_artifact,
            "local_check": bool(local_check),
            "passed": bool(source_found and local_check),
            "notes": notes,
        }
    )


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["item", "paper_status", "source_found", "local_artifact", "local_check", "passed", "notes"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    root_text = ROOT_TEX.read_text(encoding="utf-8")
    method_text = METHOD_TEX.read_text(encoding="utf-8")
    timestep = load_json("res/level_c/timestep_mask_probe/level_c_timestep_mask_probe.json")
    reverse = load_json("res/level_c/reverse_denoising_probe/level_c_reverse_denoising_probe.json")
    guided = load_json("res/level_c/guided_reverse_loop_probe/level_c_guided_reverse_loop_probe.json")
    paper_state_windows = load_json("res/level_c/paper_state_windows/level_c_paper_state_windows.json")
    paper_state_mask_reverse = load_json(
        "res/level_c/paper_state_mask_reverse_probe/level_c_paper_state_mask_reverse_probe.json"
    )

    rows: list[dict[str, Any]] = []
    add_row(
        rows,
        item="trajectory_state_latent_sequence",
        paper_status="paper_explicit",
        source_found=has(method_text, r"\\tau = \[.*\\mathbf\{s\}_\{t-N\}.*\\mathbf\{z\}_\{t\+H\}"),
        local_artifact="res/level_c/timestep_mask_probe/level_c_timestep_mask_probe.json",
        local_check=timestep["settings"]["sequence_length"] == 21
        and timestep["settings"]["state_dim"] == 181
        and timestep["settings"]["latent_dim"] == 32
        and timestep["settings"]["tau_dim"] == 213,
        notes=(
            "Paper defines tau as state-latent trajectory; this older timestep debug artifact uses 181-D fixture state "
            "+ 32-D synthetic latent over 21 tokens, while the later paper-state 99-D windows are audited separately."
        ),
    )
    add_row(
        rows,
        item="individual_state_latent_denoising_steps",
        paper_status="paper_explicit",
        source_found="individual denoising steps" in method_text and r"k_{\mathbf{s}_t}" in method_text,
        local_artifact="res/level_c/timestep_mask_probe/level_c_timestep_mask_probe.json",
        local_check=timestep["checks"]["all_step_tensors_shape_21x2"],
        notes="Paper defines separate k_s and k_z entries; local schedules store [state_step, latent_step] per token.",
    )
    add_row(
        rows,
        item="uniform_training_steps",
        paper_status="paper_explicit",
        source_found=has(method_text, r"\\textbf\{k\}_\{s_i\}.*\\mathcal\{U\}\(0, K\)"),
        local_artifact="res/level_c/timestep_mask_probe/level_c_timestep_mask_probe.json",
        local_check=timestep["checks"]["training_has_independent_state_latent_steps"],
        notes="Paper samples state/latent denoising steps uniformly; local debug schedule samples independent random steps.",
    )
    add_row(
        rows,
        item="varying_noise_for_inpainting",
        paper_status="paper_explicit_high_level",
        source_found="inpainting observations and future poses" in method_text,
        local_artifact="res/level_c/timestep_mask_probe/level_c_timestep_mask_probe.json",
        local_check=timestep["checks"]["history_conditioning_clean_prefix"]
        and timestep["checks"]["keyframe_inpainting_has_future_clean_state_only"],
        notes="Paper states varying noise enables inpainting; local probe implements history conditioning and sparse future state keyframes.",
    )
    add_row(
        rows,
        item="future_keyframe_demo",
        paper_status="paper_demonstrated_policy_unspecified",
        source_found="future keyframes" in root_text and "0.2" in root_text,
        local_artifact="res/level_c/timestep_mask_probe/level_c_timestep_mask_probe.json",
        local_check=timestep["checks"]["keyframe_inpainting_has_future_clean_state_only"],
        notes="Figure 6 describes sparse future keyframes, but exact deployed mask/keyframe policy is not published.",
    )
    add_row(
        rows,
        item="reverse_loop_clamps_observed_tokens",
        paper_status="debug_contract_for_inpainting",
        source_found="iteratively denoised" in method_text and "clean" in method_text,
        local_artifact="res/level_c/reverse_denoising_probe/level_c_reverse_denoising_probe.json",
        local_check=reverse["checks"]["observed_tokens_clamped"] and reverse["checks"]["all_steps_reach_zero"],
        notes="Local oracle reverse probe verifies observed tokens stay clamped and max steps reach zero.",
    )
    add_row(
        rows,
        item="guided_reverse_loop_with_masks",
        paper_status="debug_contract_for_guided_inference",
        source_found="classifier guidance" in method_text and "cost function" in method_text,
        local_artifact="res/level_c/guided_reverse_loop_probe/level_c_guided_reverse_loop_probe.json",
        local_check=guided["checks"]["observed_tokens_clamped_guided"]
        and guided["checks"]["all_steps_reach_zero"]
        and guided["checks"]["guidance_gradients_nonzero"],
        notes="Local guided reverse loop keeps observed masks clamped while applying debug joystick guidance.",
    )
    add_row(
        rows,
        item="paper_state_99d_mask_reverse_debug_artifact",
        paper_status="paper_state_debug_artifact_present",
        source_found=paper_state_windows["settings"]["paper_state_dim"] == 99,
        local_artifact="res/level_c/paper_state_mask_reverse_probe/level_c_paper_state_mask_reverse_probe.json",
        local_check=paper_state_mask_reverse["checks"]["paper_state_dim_99"]
        and paper_state_mask_reverse["checks"]["tau_dim_131"]
        and paper_state_mask_reverse["checks"]["all_step_tensors_shape_21x2"]
        and paper_state_mask_reverse["checks"]["keyframe_inpainting_has_future_clean_state_only"]
        and paper_state_mask_reverse["checks"]["reverse_steps_reach_zero"]
        and paper_state_mask_reverse["checks"]["reverse_observed_tokens_clamped"],
        notes=(
            "A separate 99-D paper-formula state-window debug artifact now validates local mask schedules and oracle "
            "reverse mechanics with synthetic 32-D latents. This closes the earlier local artifact gap but remains "
            "debug-only and does not prove the unpublished deployment mask policy or trained network behavior."
        ),
    )

    failed = [row for row in rows if not row["passed"]]
    policy_unspecified = [row for row in rows if "unspecified" in row["paper_status"]]
    paper_state_debug_rows = [row for row in rows if row["paper_status"] == "paper_state_debug_artifact_present"]
    json_path = OUT / "level_c_timestep_mask_coverage_audit.json"
    tsv_path = OUT / "level_c_timestep_mask_coverage_audit.tsv"
    summary = {
        "status": "ok" if not failed else "failed",
        "experiment_type": "level_c_timestep_mask_coverage_audit",
        "scope": (
            "Coverage audit for paper/source independent state/latent denoising steps and local debug task-mask "
            "schedules. It separates paper-explicit k-vector mechanics from concrete deployed history/keyframe mask "
            "policies that are not published in local artifacts."
        ),
        "metrics": {
            "row_count": len(rows),
            "failed_row_count": len(failed),
            "paper_explicit_or_high_level_rows": sum(
                1 for row in rows if row["paper_status"].startswith("paper_explicit")
            ),
            "policy_unspecified_row_count": len(policy_unspecified),
            "paper_state_debug_artifact_row_count": len(paper_state_debug_rows),
            "sequence_length": timestep["settings"]["sequence_length"],
            "state_dim": timestep["settings"]["state_dim"],
            "latent_dim": timestep["settings"]["latent_dim"],
            "tau_dim": timestep["settings"]["tau_dim"],
            "paper_state_mask_state_dim": paper_state_mask_reverse["settings"]["paper_state_dim"],
            "paper_state_mask_tau_dim": paper_state_mask_reverse["settings"]["tau_dim"],
            "paper_state_mask_reverse_final_mse": paper_state_mask_reverse["metrics"]["reverse_final_mse"],
            "reverse_final_max_step": reverse["metrics"]["final_max_step"],
            "guided_final_max_step": guided["metrics"]["final_max_step"],
        },
        "checks": {
            "all_rows_pass": len(failed) == 0,
            "paper_individual_steps_found": any(
                row["item"] == "individual_state_latent_denoising_steps" and row["source_found"] for row in rows
            ),
            "all_step_tensors_shape_21x2": timestep["checks"]["all_step_tensors_shape_21x2"],
            "training_independent_steps": timestep["checks"]["training_has_independent_state_latent_steps"],
            "history_and_keyframe_debug_masks_pass": timestep["checks"]["history_conditioning_clean_prefix"]
            and timestep["checks"]["keyframe_inpainting_has_future_clean_state_only"],
            "deployed_mask_policy_recorded_unspecified": len(policy_unspecified) >= 1,
            "paper_state_mask_reverse_debug_artifact_present": len(paper_state_debug_rows) == 1
            and paper_state_mask_reverse["checks"]["paper_state_dim_99"]
            and paper_state_mask_reverse["checks"]["tau_dim_131"],
            "reverse_and_guided_loops_reach_zero": reverse["checks"]["all_steps_reach_zero"]
            and guided["checks"]["all_steps_reach_zero"],
            "paper_state_reverse_reaches_zero": paper_state_mask_reverse["checks"]["reverse_steps_reach_zero"],
        },
        "failed_rows": failed,
        "rows": rows,
        "interpretation": {
            "paper_level_status": "partial",
            "goal_complete": False,
            "why_not_complete": (
                "The k-vector and clean-trajectory mechanics are source-covered and debug-checked. The exact deployed "
                "task mask/keyframe policy, trained denoising network behavior, exact coefficient schedule, Fig.6 "
                "execution, and rollout metrics remain unavailable."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, rows)
    print(json.dumps({"status": summary["status"], "json": str(json_path), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
