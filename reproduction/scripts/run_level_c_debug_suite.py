#!/usr/bin/env python3
"""Run lightweight Level-C VAE/diffusion/guidance debug gates."""

from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/debug_suite"
LOG_DIR = ROOT / "logs/level_c_debug_suite"
PYTHON = "python3"
TORCH_PYTHON = str(ROOT / "envs/bm_tracking/bin/python")


COMMANDS = [
    (
        "vae_accumulation_probe",
        [TORCH_PYTHON, str(ROOT / "reproduction/scripts/level_c_vae_accumulation_probe.py")],
        "res/level_c/vae_accumulation_probe/level_c_vae_accumulation_probe.json",
    ),
    (
        "vae_latent_probe",
        [TORCH_PYTHON, str(ROOT / "reproduction/scripts/level_c_vae_latent_probe.py")],
        "res/level_c/vae_latent_probe/level_c_vae_latent_probe.json",
    ),
    (
        "diffusion_equation_audit",
        [PYTHON, str(ROOT / "reproduction/scripts/level_c_diffusion_equation_audit.py")],
        "res/level_c/diffusion_equation_audit/level_c_diffusion_equation_audit.json",
    ),
    (
        "reverse_denoising_probe",
        [PYTHON, str(ROOT / "reproduction/scripts/level_c_reverse_denoising_probe.py")],
        "res/level_c/reverse_denoising_probe/level_c_reverse_denoising_probe.json",
    ),
    (
        "paper_state_mask_reverse_probe",
        [PYTHON, str(ROOT / "reproduction/scripts/level_c_paper_state_mask_reverse_probe.py")],
        "res/level_c/paper_state_mask_reverse_probe/level_c_paper_state_mask_reverse_probe.json",
    ),
    (
        "guidance_formula_probe",
        [TORCH_PYTHON, str(ROOT / "reproduction/scripts/level_c_guidance_formula_probe.py")],
        "res/level_c/guidance_formula_probe/level_c_guidance_formula_probe.json",
    ),
    (
        "guided_reverse_loop_probe",
        [TORCH_PYTHON, str(ROOT / "reproduction/scripts/level_c_guided_reverse_loop_probe.py")],
        "res/level_c/guided_reverse_loop_probe/level_c_guided_reverse_loop_probe.json",
    ),
    (
        "guidance_scale_sweep_probe",
        [TORCH_PYTHON, str(ROOT / "reproduction/scripts/level_c_guidance_scale_sweep_probe.py")],
        "res/level_c/guidance_scale_sweep_probe/level_c_guidance_scale_sweep_probe.json",
    ),
    (
        "receding_horizon_decoder_probe",
        [TORCH_PYTHON, str(ROOT / "reproduction/scripts/level_c_receding_horizon_decoder_probe.py")],
        "res/level_c/receding_horizon_decoder_probe/level_c_receding_horizon_decoder_probe.json",
    ),
    (
        "diffusion_to_vae_action_smoke",
        [PYTHON, str(ROOT / "reproduction/scripts/level_c_diffusion_to_vae_action_smoke.py")],
        "res/level_c/diffusion_to_vae_action_smoke/level_c_diffusion_to_vae_action_smoke.json",
    ),
]


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_json(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def run_step(name: str, command: list[str], output_rel: str) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=240,
        check=False,
    )
    duration = time.perf_counter() - start
    log_path = LOG_DIR / f"{name}.log"
    atomic_write_text(log_path, proc.stdout)
    output_path = ROOT / output_rel
    output = load_json(output_rel)
    return {
        "name": name,
        "command": " ".join(command),
        "returncode": proc.returncode,
        "duration_sec": duration,
        "passed": proc.returncode == 0 and output.get("status") == "ok",
        "output": str(output_path),
        "output_exists": output_path.is_file() and output_path.stat().st_size > 0,
        "output_status": output.get("status"),
        "log": str(log_path),
        "stdout_tail": proc.stdout[-2000:],
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["name", "command", "returncode", "duration_sec", "passed", "output", "output_exists", "output_status", "log"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    steps = [run_step(name, command, output_rel) for name, command, output_rel in COMMANDS]

    vae_accum = load_json("res/level_c/vae_accumulation_probe/level_c_vae_accumulation_probe.json")
    vae_latent = load_json("res/level_c/vae_latent_probe/level_c_vae_latent_probe.json")
    diffusion_eq = load_json("res/level_c/diffusion_equation_audit/level_c_diffusion_equation_audit.json")
    reverse = load_json("res/level_c/reverse_denoising_probe/level_c_reverse_denoising_probe.json")
    mask_reverse = load_json("res/level_c/paper_state_mask_reverse_probe/level_c_paper_state_mask_reverse_probe.json")
    guidance = load_json("res/level_c/guidance_formula_probe/level_c_guidance_formula_probe.json")
    guided_loop = load_json("res/level_c/guided_reverse_loop_probe/level_c_guided_reverse_loop_probe.json")
    scale = load_json("res/level_c/guidance_scale_sweep_probe/level_c_guidance_scale_sweep_probe.json")
    decoder = load_json("res/level_c/receding_horizon_decoder_probe/level_c_receding_horizon_decoder_probe.json")
    action = load_json("res/level_c/diffusion_to_vae_action_smoke/level_c_diffusion_to_vae_action_smoke.json")

    checks = {
        "all_steps_pass": all(step["passed"] for step in steps),
        "vae_accumulation_matches_paper": vae_accum.get("checks", {}).get("gradient_accumulation_matches_paper") is True,
        "vae_latent_three_seeds": vae_latent.get("checks", {}).get("at_least_three_seeds") is True,
        "diffusion_equation_denoising_steps_20": diffusion_eq.get("checks", {}).get("paper_denoising_steps_20_found") is True,
        "reverse_reduces_mse": reverse.get("checks", {}).get("full_reverse_reduces_mse") is True,
        "paper_state_mask_reverse_reaches_zero": mask_reverse.get("checks", {}).get("reverse_steps_reach_zero") is True,
        "guidance_formula_gradients_nonzero": guidance.get("checks", {}).get("all_formula_gradients_nonzero") is True,
        "guided_reverse_loop_valid": guided_loop.get("checks", {}).get("all_steps_reach_zero") is True,
        "guidance_scale_sweep_improves": scale.get("checks", {}).get("best_improves_over_zero_scale") is True,
        "decoder_uses_current_latent": decoder.get("checks", {}).get("uses_current_latent_only_for_action") is True,
        "diffusion_to_action_heldout_mse_below_0_02": action.get("checks", {}).get("heldout_current_action_mse_below_0_02") is True,
        "atomic_write_used": True,
        "does_not_claim_training_or_deployment": True,
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "level_c_debug_suite",
        "scope": "Unified execution of lightweight Level-C VAE, diffusion, guidance, decoder, and action-interface debug gates.",
        "step_count": len(steps),
        "pass_count": sum(1 for step in steps if step["passed"]),
        "steps": steps,
        "checks": checks,
        "metrics": {
            "vae_accumulation_optimizer_steps": vae_accum.get("metrics", {}).get("optimizer_steps"),
            "reverse_initial_mse": reverse.get("metrics", {}).get("initial_mse"),
            "reverse_final_mse": reverse.get("metrics", {}).get("final_mse"),
            "paper_state_reverse_final_mse": mask_reverse.get("metrics", {}).get("reverse_final_mse"),
            "guided_final_mse": guided_loop.get("metrics", {}).get("guided_final_mse"),
            "guided_cost_improvement_vs_unguided_final": guided_loop.get("metrics", {}).get(
                "guided_cost_improvement_vs_unguided_final"
            ),
            "decoder_current_index": decoder.get("metrics", {}).get("current_index"),
            "validation_predicted_current_action_mse": action.get("metrics", {}).get(
                "validation_predicted_current_action_mse"
            ),
            "test_predicted_current_action_mse": action.get("metrics", {}).get("test_predicted_current_action_mse"),
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This suite runs lightweight debug gates only. It does not train paper-scale VAE/diffusion policies, "
                "produce official checkpoints, execute TensorRT deployment, reproduce Fig. 5/Fig. 6, or run real "
                "Unitree G1 hardware."
            ),
        },
        "outputs": {
            "json": str(OUT / "level_c_debug_suite.json"),
            "tsv": str(OUT / "level_c_debug_suite.tsv"),
            "log_dir": str(LOG_DIR),
        },
    }
    atomic_write_text(OUT / "level_c_debug_suite.json", json.dumps(summary, indent=2, sort_keys=True))
    write_tsv(OUT / "level_c_debug_suite.tsv", steps)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "steps": summary["step_count"],
                "pass": summary["pass_count"],
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
