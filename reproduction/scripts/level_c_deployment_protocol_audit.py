#!/usr/bin/env python3
"""Source-indexed audit for the Level C diffusion deployment protocol."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/deployment_protocol_audit"
ROOT_TEX = ROOT / "reproduction/paper/source/root.tex"
METHOD_TEX = ROOT / "reproduction/paper/source/tex/method.tex"


def load_json(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def line_for(path: Path, pattern: str) -> str | None:
    regex = re.compile(pattern)
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if regex.search(line):
            return f"{path}:{idx}"
    return None


def contains(path: Path, pattern: str) -> bool:
    return line_for(path, pattern) is not None


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "item",
        "paper_source",
        "paper_claim",
        "local_evidence",
        "local_status",
        "boundary",
        "passed",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    smoothness = load_json("res/level_c/smoothness_latency_audit/level_c_smoothness_latency_audit.json")
    decoder = load_json("res/level_c/receding_horizon_decoder_probe/level_c_receding_horizon_decoder_probe.json")
    official = load_json("res/level_c/official_artifact_audit/level_c_official_artifact_audit.json")
    guidance = load_json("res/level_c/guidance_cost_coverage_audit/level_c_guidance_cost_coverage_audit.json")

    source_lines = {
        "control_25hz": line_for(ROOT_TEX, r"25 Hz"),
        "deployment_stack": line_for(ROOT_TEX, r"RTX 4060 Mobile GPU"),
        "tensorrt": line_for(ROOT_TEX, r"TensorRT"),
        "async": line_for(ROOT_TEX, r"asynchronously in a separate thread"),
        "latency_20ms": line_for(ROOT_TEX, r"20.*ms"),
        "cpu_decoder": line_for(ROOT_TEX, r"VAE decoder.*CPU"),
        "cppad": line_for(ROOT_TEX, r"CppAD"),
        "proprioceptive_state": line_for(ROOT_TEX, r"proprioceptive state estimation"),
        "mocap_context": line_for(ROOT_TEX, r"motion capture data"),
        "decoder_equation": line_for(METHOD_TEX, r"mathcal.*D.*mathbf.*z"),
        "current_latent": line_for(METHOD_TEX, r"current denoised latent"),
        "guidance_gradient": line_for(METHOD_TEX, r"nabla.*G"),
    }

    control_period_ms = float(smoothness["settings"]["paper_control_period_ms"])
    paper_denoising_latency_ms = float(smoothness["settings"]["paper_denoising_total_ms"])
    paper_denoising_steps = int(smoothness["settings"]["denoising_steps"])
    denoising_fraction = float(smoothness["metrics"]["paper_denoising_fraction_of_control_period"])
    decoder_cpu_probe = bool(decoder["checks"]["decoder_runs_on_cpu_by_default"])
    guidance_debug_gradients = bool(guidance["checks"]["all_paper_explicit_costs_have_source_and_gradients"])
    official_engine_absent = not bool(official["conclusion"]["official_beyondmimic_checkpoint_or_engine_found"])

    rows: list[dict[str, Any]] = [
        {
            "item": "control_frequency_25hz",
            "paper_source": source_lines["control_25hz"],
            "paper_claim": "Tracking policies are trained and deployed at 25 Hz to leave diffusion inference time.",
            "local_evidence": "smoothness_latency_audit records 40 ms control period and 25 Hz checks.",
            "local_status": "debug_budget_check",
            "boundary": "No trained closed-loop controller is running at 25 Hz.",
            "passed": bool(smoothness["checks"]["control_rate_25hz_recorded"]),
        },
        {
            "item": "diffusion_latency_20ms_20_steps",
            "paper_source": source_lines["latency_20ms"],
            "paper_claim": "Diffusion inference takes about 20 ms using 20 denoising steps.",
            "local_evidence": f"Budget check records {paper_denoising_latency_ms} ms / {paper_denoising_steps} steps, fraction {denoising_fraction}.",
            "local_status": "paper_budget_indexed",
            "boundary": "No TensorRT engine benchmark or measured deployment latency exists locally.",
            "passed": bool(smoothness["checks"]["paper_denoising_latency_within_control_period"]),
        },
        {
            "item": "async_diffusion_thread",
            "paper_source": source_lines["async"],
            "paper_claim": "Diffusion inference is asynchronous in a separate thread.",
            "local_evidence": "Recorded as a deployment boundary; local probes are offline/synchronous Python scripts.",
            "local_status": "expected_missing_boundary",
            "boundary": "No C++ async runtime integration or thread scheduling measurement exists locally.",
            "passed": source_lines["async"] is not None,
        },
        {
            "item": "tensorrt_acceleration",
            "paper_source": source_lines["tensorrt"],
            "paper_claim": "TensorRT accelerates diffusion inference on the onboard GPU.",
            "local_evidence": "official_artifact_audit found no official checkpoint/TensorRT engine candidate.",
            "local_status": "expected_missing_boundary",
            "boundary": "No TensorRT engine, conversion script, or engine latency benchmark exists locally.",
            "passed": source_lines["tensorrt"] is not None and official_engine_absent,
        },
        {
            "item": "rtx4060_mobile_mini_pc",
            "paper_source": source_lines["deployment_stack"],
            "paper_claim": "Deployment uses a portable Mini PC with NVIDIA RTX 4060 Mobile GPU.",
            "local_evidence": "Host has RTX 4090 D GPUs, not the stated onboard Mini PC stack.",
            "local_status": "expected_missing_boundary",
            "boundary": "This host cannot validate the paper onboard hardware latency claim.",
            "passed": source_lines["deployment_stack"] is not None,
        },
        {
            "item": "vae_decoder_cpu_sync_current_latent",
            "paper_source": f"{source_lines['cpu_decoder']}; {source_lines['decoder_equation']}; {source_lines['current_latent']}",
            "paper_claim": "Lightweight VAE decoder runs synchronously on CPU and decodes current denoised latent.",
            "local_evidence": "receding_horizon_decoder_probe decodes current index 4 latent to a 29-D action on CPU.",
            "local_status": "debug_schema_probe",
            "boundary": "The local decoder is randomly initialized and lacks the trained checkpoint/proprioception layout.",
            "passed": decoder_cpu_probe and bool(decoder["checks"]["uses_current_latent_only_for_action"]),
        },
        {
            "item": "cppad_guidance_gradients",
            "paper_source": f"{source_lines['cppad']}; {source_lines['guidance_gradient']}",
            "paper_claim": "Guiding-cost gradients are computed automatically with CppAD inside each denoising iteration.",
            "local_evidence": "guidance_cost_coverage_audit verifies paper-explicit costs have source and debug gradients.",
            "local_status": "debug_autograd_probe",
            "boundary": "Local gradients use Python/PyTorch-style debug probes, not CppAD in deployed C++.",
            "passed": source_lines["cppad"] is not None and guidance_debug_gradients,
        },
        {
            "item": "joystick_inpainting_state_estimation",
            "paper_source": source_lines["proprioceptive_state"],
            "paper_claim": "Joystick and inpainting tasks use proprioceptive state estimation for pose/velocities.",
            "local_evidence": "Decoder/schema probes use fixture-derived state vectors, not live state estimation.",
            "local_status": "expected_missing_boundary",
            "boundary": "No 500 Hz estimator, live robot state, or task rollout log exists locally.",
            "passed": source_lines["proprioceptive_state"] is not None,
        },
        {
            "item": "waypoint_obstacle_mocap_context",
            "paper_source": source_lines["mocap_context"],
            "paper_claim": "Waypoint/obstacle tasks use motion capture for environment context and localization.",
            "local_evidence": "Guidance formulas are debug-checked, but no mocap/environment context is available.",
            "local_status": "expected_missing_boundary",
            "boundary": "No mocap stream, obstacle scene log, or closed-loop task trajectory exists locally.",
            "passed": source_lines["mocap_context"] is not None,
        },
    ]

    failed_rows = [row for row in rows if not row["passed"]]
    deployment_boundary_rows = [row for row in rows if row["local_status"] == "expected_missing_boundary"]
    debug_rows = [row for row in rows if row["local_status"].startswith("debug")]

    checks = {
        "paper_deployment_protocol_source_found": all(
            source_lines[key] is not None
            for key in [
                "control_25hz",
                "deployment_stack",
                "tensorrt",
                "async",
                "latency_20ms",
                "cpu_decoder",
                "cppad",
                "proprioceptive_state",
                "mocap_context",
            ]
        ),
        "method_decoder_and_guidance_sources_found": all(
            source_lines[key] is not None for key in ["decoder_equation", "current_latent", "guidance_gradient"]
        ),
        "control_rate_25hz_recorded": bool(smoothness["checks"]["control_rate_25hz_recorded"]),
        "paper_latency_budget_recorded": paper_denoising_steps == 20
        and paper_denoising_latency_ms == 20.0
        and denoising_fraction <= 1.0,
        "decoder_cpu_schema_probe_present": decoder_cpu_probe,
        "guidance_gradients_debug_present": guidance_debug_gradients,
        "official_tensorrt_engine_absent_recorded": official_engine_absent,
        "async_tensorrt_cppad_boundaries_recorded": all(
            row["local_status"] in {"expected_missing_boundary", "debug_autograd_probe"}
            for row in rows
            if row["item"] in {"async_diffusion_thread", "tensorrt_acceleration", "cppad_guidance_gradients"}
        ),
        "all_rows_pass_or_record_expected_boundary": not failed_rows,
        "does_not_claim_deployment_reproduction": True,
    }

    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "source_indexed_deployment_protocol_audit",
        "scope": "Level C diffusion deployment protocol source claims, debug evidence, and explicit deployment boundaries",
        "paper_evidence": source_lines,
        "metrics": {
            "row_count": len(rows),
            "failed_row_count": len(failed_rows),
            "paper_explicit_row_count": len(rows),
            "implemented_debug_row_count": len(debug_rows),
            "deployment_boundary_row_count": len(deployment_boundary_rows),
            "control_period_ms": control_period_ms,
            "paper_diffusion_latency_ms": paper_denoising_latency_ms,
            "paper_denoising_steps": paper_denoising_steps,
            "denoising_fraction_of_control_period": denoising_fraction,
        },
        "checks": checks,
        "rows": rows,
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "partial",
            "why_not_complete": (
                "The paper deployment protocol is source-indexed and partially cross-checked with debug latency/decoder/"
                "guidance artifacts, but TensorRT, asynchronous C++ integration, CppAD deployment, RTX 4060 Mobile Mini PC "
                "latency, live state estimation, mocap context, and closed-loop task logs are not available locally."
            ),
        },
        "outputs": {
            "json": str(OUT / "level_c_deployment_protocol_audit.json"),
            "tsv": str(OUT / "level_c_deployment_protocol_audit.tsv"),
        },
    }

    (OUT / "level_c_deployment_protocol_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_tsv(OUT / "level_c_deployment_protocol_audit.tsv", rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"]}, sort_keys=True))


if __name__ == "__main__":
    main()
