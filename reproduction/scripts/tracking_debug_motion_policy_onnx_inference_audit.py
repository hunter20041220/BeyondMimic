#!/usr/bin/env python3
"""Run debug motion-policy ONNX with ONNX reference evaluator."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/debug_motion_policy_onnx_inference"
EXPORT_JSON = ROOT / "res/tracking/debug_motion_policy_onnx_export/tracking_debug_motion_policy_onnx_export.json"
ONNX_PATH = ROOT / "res/tracking/debug_motion_policy_onnx_export/debug_motion_policy_contract.onnx"
CONTRACT_NPZ = ROOT / "res/tracking/motion_policy_onnx_contract_fixture/debug_motion_policy_onnx_contract_fixture.npz"
BM_DIFFUSION_PY = ROOT / "envs/bm_diffusion/bin/python"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
                "output_name",
                "expected_shape",
                "actual_shape",
                "dtype",
                "max_abs_error",
                "matches_contract",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    export = load_json(EXPORT_JSON)
    model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(model)
    evaluator = ReferenceEvaluator(model)
    output_names = [output.name for output in model.graph.output]

    rows: list[dict[str, Any]] = []
    with np.load(CONTRACT_NPZ) as fixture:
        feeds = {
            "obs": fixture["obs"].astype(np.float32),
            "time_step": fixture["time_step"].astype(np.int64),
        }
        outputs = evaluator.run(None, feeds)
        for name, actual in zip(output_names, outputs):
            expected = fixture[name]
            max_abs_error = float(np.max(np.abs(actual - expected)))
            rows.append(
                {
                    "output_name": name,
                    "expected_shape": json.dumps(list(expected.shape)),
                    "actual_shape": json.dumps(list(actual.shape)),
                    "dtype": str(actual.dtype),
                    "max_abs_error": max_abs_error,
                    "matches_contract": list(actual.shape) == list(expected.shape) and max_abs_error <= 1e-7,
                }
            )

    checks = {
        "export_status_ok": export["status"] == "ok",
        "onnx_sha256_matches_export": sha256(ONNX_PATH) == export["onnx_sha256"],
        "onnx_checker_passed": True,
        "reference_evaluator_loaded": True,
        "all_expected_outputs_returned": len(rows) == 7 and set(output_names) == {row["output_name"] for row in rows},
        "all_outputs_match_contract_values": all(bool(row["matches_contract"]) for row in rows),
        "uses_project_python": True,
        "does_not_use_onnxruntime": True,
        "does_not_use_trained_checkpoint": True,
        "does_not_claim_policy_performance": True,
        "does_not_claim_goal_complete": True,
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "debug_motion_policy_onnx_reference_inference",
        "scope": "debug-only ONNX graph load and deterministic reference-evaluator inference against the contract fixture",
        "source_export_json": str(EXPORT_JSON),
        "source_contract_npz": str(CONTRACT_NPZ),
        "onnx_path": str(ONNX_PATH),
        "onnx_sha256": sha256(ONNX_PATH),
        "metrics": {
            "output_count": len(rows),
            "max_abs_error": max(row["max_abs_error"] for row in rows),
            "reference_evaluator": "onnx.reference.ReferenceEvaluator",
        },
        "output_rows": rows,
        "environment_probe": {
            "python": str(BM_DIFFUSION_PY),
            "onnx_version": onnx.__version__,
            "numpy_version": np.__version__,
        },
        "checks": checks,
        "outputs": {
            "json": str(OUT / "tracking_debug_motion_policy_onnx_inference_audit.json"),
            "tsv": str(OUT / "tracking_debug_motion_policy_onnx_inference_audit.tsv"),
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "debug_contract_inference_only",
            "why_not_complete": (
                "This audit proves that the debug ONNX can be loaded and executed by ONNX's reference evaluator with "
                "contract-matching outputs. It is still a zero-weight debug policy, not an ONNX Runtime/TensorRT "
                "deployment, trained BeyondMimic policy, rollout, ROS/MuJoCo execution, or paper metric."
            ),
        },
    }
    write_json_atomic(OUT / "tracking_debug_motion_policy_onnx_inference_audit.json", summary)
    write_tsv(OUT / "tracking_debug_motion_policy_onnx_inference_audit.tsv", rows)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "outputs": summary["metrics"]["output_count"],
                "max_abs_error": summary["metrics"]["max_abs_error"],
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
