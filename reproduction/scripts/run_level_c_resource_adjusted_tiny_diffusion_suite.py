#!/usr/bin/env python3
"""Unified resource-adjusted tiny diffusion evidence suite."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/resource_adjusted_tiny_diffusion_suite"
LOG = ROOT / "logs/level_c_resource_adjusted_tiny_diffusion_suite"
PY = str(ROOT / "envs/bm_diffusion/bin/python")


STEPS = [
    (
        "resource_adjusted_tiny_diffusion_training_run",
        [
            PY,
            str(ROOT / "reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_training_run.py"),
            "--device",
            "cpu",
            "--torch-threads",
            "2",
            "--epochs",
            "180",
        ],
        "res/level_c/resource_adjusted_tiny_diffusion_training_run/"
        "level_c_resource_adjusted_tiny_diffusion_training_run.json",
    ),
    (
        "resource_adjusted_tiny_diffusion_multiseed_audit",
        [
            PY,
            str(ROOT / "reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_multiseed_audit.py"),
            "--device",
            "cpu",
            "--torch-threads",
            "2",
            "--epochs",
            "80",
        ],
        "res/level_c/resource_adjusted_tiny_diffusion_multiseed_audit/"
        "level_c_resource_adjusted_tiny_diffusion_multiseed_audit.json",
    ),
    (
        "resource_adjusted_tiny_diffusion_checkpoint_eval",
        [
            PY,
            str(ROOT / "reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_checkpoint_eval.py"),
            "--device",
            "cpu",
            "--torch-threads",
            "2",
        ],
        "res/level_c/resource_adjusted_tiny_diffusion_checkpoint_eval/"
        "level_c_resource_adjusted_tiny_diffusion_checkpoint_eval.json",
    ),
    (
        "resource_adjusted_tiny_diffusion_onnx_export_inference",
        [PY, str(ROOT / "reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_onnx_export_inference.py")],
        "res/level_c/resource_adjusted_tiny_diffusion_onnx_export_inference/"
        "level_c_resource_adjusted_tiny_diffusion_onnx_export_inference.json",
    ),
    (
        "resource_adjusted_tiny_diffusion_latency_audit",
        [PY, str(ROOT / "reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_latency_audit.py")],
        "res/level_c/resource_adjusted_tiny_diffusion_latency_audit/"
        "level_c_resource_adjusted_tiny_diffusion_latency_audit.json",
    ),
    (
        "resource_adjusted_tiny_diffusion_video_preview",
        [PY, str(ROOT / "reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_video_preview.py")],
        "res/level_c/resource_adjusted_tiny_diffusion_video_preview/"
        "level_c_resource_adjusted_tiny_diffusion_video_preview.json",
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
        timeout=420,
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
    training = load_json(STEPS[0][2])
    multiseed = load_json(STEPS[1][2])
    checkpoint = load_json(STEPS[2][2])
    onnx = load_json(STEPS[3][2])
    latency = load_json(STEPS[4][2])
    video = load_json(STEPS[5][2])

    checks = {
        "all_steps_pass": all(row["passed"] for row in rows),
        "step_count_6": len(rows) == 6,
        "pass_count_6": sum(row["passed"] for row in rows) == 6,
        "training_checkpoint_exists": training["checks"]["checkpoint_file_exists"],
        "training_heldout_token_mse_better_than_noisy": training["metrics"]["validation_pred_token_mse"]
        < training["eval_rows"][1]["noisy_token_mse"]
        and training["metrics"]["test_pred_token_mse"] < training["eval_rows"][2]["noisy_token_mse"],
        "multiseed_three_seeds": multiseed["checks"]["seed_count_3"],
        "multiseed_heldout_improves": multiseed["checks"]["all_validation_token_improves_vs_noisy"]
        and multiseed["checks"]["all_test_token_improves_vs_noisy"],
        "checkpoint_eval_reproduces_training_metrics": checkpoint["checks"]["max_token_mse_delta_below_1e_minus_12"]
        and checkpoint["checks"]["max_action_mse_delta_below_1e_minus_12"],
        "onnx_matches_torch": onnx["checks"]["onnx_matches_torch"],
        "latency_under_debug_budget": latency["checks"]["tiny_onnx_reference_p95_under_paper_20ms_debug_budget"]
        and latency["checks"]["tiny_torch_p95_under_paper_20ms_debug_budget"],
        "video_previews_written": video["checks"]["two_debug_gifs_written"] and video["checks"]["posters_written"],
        "atomic_write_used": True,
        "does_not_claim_paper_training_or_deployment": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "scope": "Resource-adjusted Level-C tiny diffusion suite: debug training, stability, reload, ONNX, latency, and preview evidence",
        "step_count": len(rows),
        "pass_count": sum(row["passed"] for row in rows),
        "steps": rows,
        "metrics": {
            "parameter_count": training["metrics"]["parameter_count"],
            "epochs": training["metrics"]["epochs"],
            "validation_pred_token_mse": training["metrics"]["validation_pred_token_mse"],
            "test_pred_token_mse": training["metrics"]["test_pred_token_mse"],
            "validation_pred_current_action_mse": training["metrics"]["validation_pred_current_action_mse"],
            "test_pred_current_action_mse": training["metrics"]["test_pred_current_action_mse"],
            "multiseed_validation_token_mse_mean": multiseed["statistics"]["validation_pred_token_mse"]["mean"],
            "multiseed_test_token_mse_mean": multiseed["statistics"]["test_pred_token_mse"]["mean"],
            "checkpoint_max_token_delta": checkpoint["metrics"]["max_abs_pred_token_mse_delta_vs_source"],
            "onnx_max_abs_vs_torch": onnx["metrics"]["max_abs_onnx_vs_torch"],
            "onnx_size_bytes": onnx["metrics"]["onnx_size_bytes"],
            "onnx_reference_cpu_p95_ms": latency["metrics"]["onnx_reference_cpu_p95_ms"],
            "torch_cpu_p95_ms": latency["metrics"]["torch_cpu_p95_ms"],
            "video_preview_count": len(video["rows"]),
        },
        "checks": checks,
        "interpretation": {
            "evidence_level": "resource_adjusted_debug_suite",
            "goal_complete": False,
            "remaining_gap": (
                "This suite reruns a real resource-adjusted debug tiny denoiser chain. It still is not the paper "
                "Transformer, true VAE rollout data, paper-scale diffusion training, official checkpoint, TensorRT, "
                "closed-loop Fig. 5/Fig. 6 rollout, or hardware evidence."
            ),
        },
    }
    atomic_write_text(
        OUT / "level_c_resource_adjusted_tiny_diffusion_suite.json",
        json.dumps(summary, indent=2, sort_keys=True),
    )
    atomic_write_tsv(
        OUT / "level_c_resource_adjusted_tiny_diffusion_suite.tsv",
        rows,
        ["name", "command", "return_code", "passed", "artifact", "artifact_status", "log"],
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(OUT / "level_c_resource_adjusted_tiny_diffusion_suite.json"),
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
