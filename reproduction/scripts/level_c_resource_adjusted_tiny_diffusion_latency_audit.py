#!/usr/bin/env python3
"""Latency audit for the resource-adjusted tiny diffusion debug ONNX."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SCRIPTS = ROOT / "reproduction/scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import level_c_resource_adjusted_tiny_diffusion_training_run as tiny  # noqa: E402


OUT = ROOT / "res/level_c/resource_adjusted_tiny_diffusion_latency_audit"
SOURCE_JSON = (
    ROOT
    / "res/level_c/resource_adjusted_tiny_diffusion_onnx_export_inference/"
    / "level_c_resource_adjusted_tiny_diffusion_onnx_export_inference.json"
)
TRAINING_JSON = (
    ROOT
    / "res/level_c/resource_adjusted_tiny_diffusion_training_run/"
    / "level_c_resource_adjusted_tiny_diffusion_training_run.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=[
                "runtime",
                "iterations",
                "mean_ms",
                "median_ms",
                "p95_ms",
                "min_ms",
                "max_ms",
                "paper_20ms_fraction_p95",
                "control_40ms_fraction_p95",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def timed_loop(fn, *, warmup: int, iterations: int) -> tuple[dict[str, float], list[float]]:
    for _ in range(warmup):
        fn()
    values: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        values.append((time.perf_counter() - start) * 1000.0)
    return stats(values), values


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    source = load_json(SOURCE_JSON)
    training = load_json(TRAINING_JSON)
    onnx_path = Path(source["outputs"]["onnx"])
    io_npz = Path(source["outputs"]["npz"])
    checkpoint_path = Path(training["outputs"]["checkpoint"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = tiny.TinyConfig(**payload["config"])
    model = tiny.TinyDenoiser(cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    with np.load(io_npz) as data:
        noisy_np = data["noisy_tau"].astype(np.float32)
        steps_np = data["diffusion_steps"].astype(np.int64)
    noisy_torch = torch.from_numpy(noisy_np)
    steps_torch = torch.from_numpy(steps_np)
    proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(proto)
    evaluator = ReferenceEvaluator(proto)

    with torch.no_grad():
        torch_pred = model(noisy_torch, steps_torch).detach().cpu().numpy()
    onnx_pred = evaluator.run(None, {"noisy_tau": noisy_np, "diffusion_steps": steps_np})[0]
    max_abs = float(np.max(np.abs(torch_pred - onnx_pred)))

    warmup = 5
    iterations = 100

    def run_torch() -> None:
        with torch.no_grad():
            model(noisy_torch, steps_torch)

    def run_onnx_reference() -> None:
        evaluator.run(None, {"noisy_tau": noisy_np, "diffusion_steps": steps_np})

    torch_stats, torch_values = timed_loop(run_torch, warmup=warmup, iterations=iterations)
    onnx_stats, onnx_values = timed_loop(run_onnx_reference, warmup=warmup, iterations=iterations)
    paper_budget_ms = 20.0
    control_period_ms = 40.0
    rows: list[dict[str, Any]] = []
    for runtime, row_stats in [("pytorch_cpu", torch_stats), ("onnx_reference_cpu", onnx_stats)]:
        rows.append(
            {
                "runtime": runtime,
                "iterations": iterations,
                **row_stats,
                "paper_20ms_fraction_p95": row_stats["p95_ms"] / paper_budget_ms,
                "control_40ms_fraction_p95": row_stats["p95_ms"] / control_period_ms,
            }
        )

    npz_path = OUT / "level_c_resource_adjusted_tiny_diffusion_latency_audit.npz"
    np.savez_compressed(
        npz_path,
        torch_latency_ms=np.asarray(torch_values, dtype=np.float64),
        onnx_reference_latency_ms=np.asarray(onnx_values, dtype=np.float64),
        torch_predicted_clean_tau=torch_pred.astype(np.float32),
        onnx_predicted_clean_tau=onnx_pred.astype(np.float32),
    )
    json_path = OUT / "level_c_resource_adjusted_tiny_diffusion_latency_audit.json"
    tsv_path = OUT / "level_c_resource_adjusted_tiny_diffusion_latency_audit.tsv"
    checks = {
        "source_onnx_export_status_ok": source["status"] == "ok",
        "training_status_ok": training["status"] == "ok",
        "onnx_file_exists": onnx_path.is_file() and onnx_path.stat().st_size > 0,
        "io_fixture_exists": io_npz.is_file() and io_npz.stat().st_size > 0,
        "onnx_checker_passed": True,
        "onnx_matches_torch": max_abs <= 1e-5,
        "torch_latency_finite_positive": bool(
            torch_stats["p95_ms"] > 0.0 and np.isfinite(torch_stats["p95_ms"])
        ),
        "onnx_reference_latency_finite_positive": bool(
            onnx_stats["p95_ms"] > 0.0 and np.isfinite(onnx_stats["p95_ms"])
        ),
        "tiny_onnx_reference_p95_under_paper_20ms_debug_budget": onnx_stats["p95_ms"] < paper_budget_ms,
        "tiny_torch_p95_under_paper_20ms_debug_budget": torch_stats["p95_ms"] < paper_budget_ms,
        "npz_written": npz_path.is_file() and npz_path.stat().st_size > 0,
        "does_not_claim_tensorrt": True,
        "does_not_claim_paper_model_latency": True,
        "does_not_claim_goal_complete": True,
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "resource_adjusted_tiny_diffusion_latency_audit",
        "scope": (
            "CPU latency audit for the resource-adjusted tiny debug denoiser in PyTorch and ONNX reference runtime, "
            "using the same validation-window fixture as the ONNX export/inference audit"
        ),
        "settings": {
            "warmup_iterations": warmup,
            "timed_iterations": iterations,
            "torch_threads": torch.get_num_threads(),
            "paper_diffusion_latency_budget_ms": paper_budget_ms,
            "paper_control_period_ms": control_period_ms,
            "sequence_length": cfg.sequence_length,
            "token_dim": cfg.token_dim,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
        },
        "metrics": {
            "max_abs_onnx_vs_torch": max_abs,
            "torch_cpu_p95_ms": torch_stats["p95_ms"],
            "onnx_reference_cpu_p95_ms": onnx_stats["p95_ms"],
            "torch_cpu_median_ms": torch_stats["median_ms"],
            "onnx_reference_cpu_median_ms": onnx_stats["median_ms"],
            "onnx_reference_p95_fraction_of_paper_20ms": onnx_stats["p95_ms"] / paper_budget_ms,
            "onnx_reference_p95_fraction_of_control_40ms": onnx_stats["p95_ms"] / control_period_ms,
            "torch_p95_fraction_of_paper_20ms": torch_stats["p95_ms"] / paper_budget_ms,
            "torch_p95_fraction_of_control_40ms": torch_stats["p95_ms"] / control_period_ms,
        },
        "rows": rows,
        "checks": checks,
        "outputs": {"json": str(json_path), "tsv": str(tsv_path), "npz": str(npz_path)},
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "resource_adjusted_debug_latency_only",
            "why_not_complete": (
                "This measures a small debug denoiser on this host with PyTorch CPU and ONNX's reference evaluator. "
                "It is not TensorRT, not the paper Transformer checkpoint, not the RTX 4060 Mobile Mini PC, not "
                "asynchronous deployment, and not a closed-loop controller latency benchmark."
            ),
        },
    }
    write_json_atomic(json_path, summary)
    write_tsv(tsv_path, rows)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "torch_p95_ms": torch_stats["p95_ms"],
                "onnx_reference_p95_ms": onnx_stats["p95_ms"],
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
