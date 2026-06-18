#!/usr/bin/env python3
"""Unified Level-C extended debug evidence suite.

This suite consolidates nontraining Level-C evidence beyond the lightweight
formula/mechanics suite: state-latent consistency, tiny-VAE heldout behavior,
non-memorizing diffusion heldout baselines, downstream action interface, and
action smoothness. It is intentionally not paper-scale training.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/extended_debug_suite"
LOG = ROOT / "logs/level_c_extended_debug_suite"
PY = "python3"
TORCH_PY = str(ROOT / "envs/bm_tracking/bin/python")


STEPS = [
    (
        "state_latent_dataset_consistency",
        [PY, str(ROOT / "reproduction/scripts/level_c_state_latent_dataset_consistency_audit.py")],
        "res/level_c/state_latent_dataset_consistency_audit/level_c_state_latent_dataset_consistency_audit.json",
    ),
    (
        "vae_debug_overfit_latent_artifact",
        [TORCH_PY, str(ROOT / "reproduction/scripts/level_c_vae_debug_overfit_latent_artifact.py"), "--device", "cpu"],
        "res/level_c/vae_debug_overfit_latent_artifact/level_c_vae_debug_overfit_latent_artifact.json",
    ),
    (
        "vae_motion_split_heldout_eval",
        [TORCH_PY, str(ROOT / "reproduction/scripts/level_c_vae_motion_split_heldout_eval.py"), "--device", "cpu"],
        "res/level_c/vae_motion_split_heldout_eval/level_c_vae_motion_split_heldout_eval.json",
    ),
    (
        "vae_latent_diffusion_overfit",
        [PY, str(ROOT / "reproduction/scripts/level_c_vae_latent_diffusion_overfit_probe.py")],
        "res/level_c/vae_latent_diffusion_overfit_probe/level_c_vae_latent_diffusion_overfit_probe.json",
    ),
    (
        "vae_latent_heldout_eval",
        [PY, str(ROOT / "reproduction/scripts/level_c_vae_latent_heldout_eval.py")],
        "res/level_c/vae_latent_heldout_eval/level_c_vae_latent_heldout_eval.json",
    ),
    (
        "vae_latent_heldout_multiseed",
        [PY, str(ROOT / "reproduction/scripts/level_c_vae_latent_heldout_multiseed_audit.py")],
        "res/level_c/vae_latent_heldout_multiseed_audit/level_c_vae_latent_heldout_multiseed_audit.json",
    ),
    (
        "diffusion_to_vae_action_smoke",
        [PY, str(ROOT / "reproduction/scripts/level_c_diffusion_to_vae_action_smoke.py")],
        "res/level_c/diffusion_to_vae_action_smoke/level_c_diffusion_to_vae_action_smoke.json",
    ),
    (
        "diffusion_to_vae_action_multiseed",
        [PY, str(ROOT / "reproduction/scripts/level_c_diffusion_to_vae_action_multiseed_audit.py")],
        "res/level_c/diffusion_to_vae_action_multiseed_audit/level_c_diffusion_to_vae_action_multiseed_audit.json",
    ),
    (
        "diffusion_to_vae_action_smoothness",
        [PY, str(ROOT / "reproduction/scripts/level_c_diffusion_to_vae_action_smoothness_audit.py")],
        "res/level_c/diffusion_to_vae_action_smoothness_audit/level_c_diffusion_to_vae_action_smoothness_audit.json",
    ),
    (
        "small_dataset_heldout_multiseed",
        [PY, str(ROOT / "reproduction/scripts/level_c_small_dataset_heldout_multiseed_audit.py")],
        "res/level_c/small_dataset_heldout_multiseed_audit/level_c_small_dataset_heldout_multiseed_audit.json",
    ),
]


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def load_json(rel: str) -> dict[str, Any]:
    path = ROOT / rel
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def run_step(name: str, command: list[str], output_rel: str) -> dict[str, Any]:
    LOG.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
        check=False,
    )
    log_path = LOG / f"{name}.log"
    atomic_write_text(log_path, proc.stdout)
    output = load_json(output_rel)
    return {
        "name": name,
        "command": " ".join(command),
        "return_code": proc.returncode,
        "passed": proc.returncode == 0 and output.get("status") == "ok",
        "artifact": str(ROOT / output_rel),
        "artifact_status": output.get("status"),
        "log": str(log_path),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run_step(*step) for step in STEPS]
    consistency = load_json(STEPS[0][2])
    vae_latents = load_json(STEPS[1][2])
    vae_heldout = load_json(STEPS[2][2])
    overfit = load_json(STEPS[3][2])
    heldout = load_json(STEPS[4][2])
    heldout_multi = load_json(STEPS[5][2])
    action = load_json(STEPS[6][2])
    action_multi = load_json(STEPS[7][2])
    smoothness = load_json(STEPS[8][2])
    small_multi = load_json(STEPS[9][2])

    checks = {
        "all_steps_pass": all(row["passed"] for row in rows),
        "step_count_10": len(rows) == 10,
        "pass_count_10": sum(row["passed"] for row in rows) == 10,
        "state_latent_rows_84": consistency["metrics"]["row_count"] == 84,
        "state_latent_dims_99_32_131_29": consistency["metrics"]["state_dim"] == 99
        and consistency["metrics"]["latent_dim"] == 32
        and consistency["metrics"]["token_dim"] == 131
        and consistency["metrics"]["action_dim"] == 29,
        "vae_debug_latents_nonzero": vae_latents["checks"]["all_latents_nonzero"],
        "vae_motion_split_heldout_reduces_test_mse": vae_heldout["checks"]["test_action_mse_decreases"],
        "vae_latent_overfit_loss_below_1e_minus_8": overfit["checks"]["final_loss_below_1e_minus_8"],
        "vae_latent_heldout_test_loss_decreases": heldout["checks"]["test_loss_decreases_vs_noisy_baseline"],
        "vae_latent_multiseed_three_seeds": heldout_multi["checks"]["at_least_three_seeds"],
        "action_smoke_action_dim_29": action["checks"]["action_dim_29"],
        "action_multiseed_reduces_test_mse": action_multi["checks"][
            "all_seed_test_prediction_improves_current_vs_noisy"
        ],
        "smoothness_control_frequency_25hz": smoothness["checks"]["control_frequency_25hz"],
        "small_dataset_multiseed_three_seeds": small_multi["checks"]["at_least_three_seeds"],
        "atomic_write_used": True,
        "does_not_claim_paper_training_or_deployment": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "scope": "Level-C extended debug evidence suite; nontraining VAE/diffusion/action checks only",
        "step_count": len(rows),
        "pass_count": sum(row["passed"] for row in rows),
        "steps": rows,
        "metrics": {
            "state_latent_rows": consistency["metrics"]["row_count"],
            "state_dim": consistency["metrics"]["state_dim"],
            "latent_dim": consistency["metrics"]["latent_dim"],
            "token_dim": consistency["metrics"]["token_dim"],
            "action_dim": consistency["metrics"]["action_dim"],
            "vae_debug_latent_abs_mean": vae_latents["metrics"]["latent_abs_mean_mean"],
            "vae_heldout_test_action_mse": vae_heldout["metrics"]["test_final_action_mse"],
            "vae_latent_overfit_loss": overfit["metrics"]["all_debug_vae_latent_windows_overfit_loss"],
            "vae_latent_test_prediction_loss": heldout["metrics"]["test_prediction_loss"],
            "vae_latent_multiseed_test_reduction_mean": heldout_multi["statistics"]["test_loss_reduction_ratio"]["mean"],
            "action_test_current_mse": action["metrics"]["test_predicted_current_action_mse"],
            "action_multiseed_test_current_mse_mean": action_multi["statistics"]["test_predicted_current_action_mse"]["mean"],
            "action_smoothness_test_penalty": smoothness["metrics"]["test_predicted_smoothness_penalty"],
            "small_dataset_heldout_test_reduction_mean": small_multi["statistics"]["test_loss_reduction_ratio"]["mean"],
        },
        "checks": checks,
        "interpretation": {
            "evidence_level": "level_c_extended_debug_suite",
            "goal_complete": False,
            "remaining_gap": (
                "This suite consolidates nontraining debug evidence. It does not create true VAE rollout latents, "
                "paper-scale diffusion training, official checkpoints, Fig. 5/Fig. 6 rollouts, TensorRT deployment, "
                "or hardware results."
            ),
        },
    }
    atomic_write_text(OUT / "level_c_extended_debug_suite.json", json.dumps(summary, indent=2, sort_keys=True))
    atomic_write_tsv(
        OUT / "level_c_extended_debug_suite.tsv",
        rows,
        ["name", "command", "return_code", "passed", "artifact", "artifact_status", "log"],
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(OUT / "level_c_extended_debug_suite.json"),
                "steps": summary["step_count"],
                "pass": summary["pass_count"],
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
