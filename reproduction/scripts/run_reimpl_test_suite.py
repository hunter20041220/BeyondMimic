#!/usr/bin/env python3
"""Run the pure-Python clean-room reimplementation test suite."""

from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tests/reimpl_test_suite"
LOG_DIR = ROOT / "logs/reimpl_test_suite"


COMMANDS = [
    (
        "core_math_unit_tests",
        ["python3", str(ROOT / "reproduction/tests/test_core_math.py")],
        "res/tests/core_math_unit_tests/core_math_unit_tests.json",
    ),
    (
        "reimpl_package_api_tests",
        ["python3", str(ROOT / "reproduction/tests/test_reimpl_package_api.py")],
        "res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json",
    ),
    (
        "reimpl_package_audit",
        ["python3", str(ROOT / "reproduction/scripts/reimpl_package_audit.py")],
        "res/code/reimpl_package_audit/reimpl_package_audit.json",
    ),
    (
        "reimpl_runtime_integration_audit",
        ["python3", str(ROOT / "reproduction/scripts/reimpl_runtime_integration_audit.py")],
        "res/code/reimpl_runtime_integration_audit/reimpl_runtime_integration_audit.json",
    ),
    (
        "core_test_coverage_audit",
        ["python3", str(ROOT / "reproduction/scripts/core_test_coverage_audit.py")],
        "res/tests/core_test_coverage_audit/core_test_coverage_audit.json",
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
        timeout=180,
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
        "row_count": output.get("row_count"),
        "failed_row_count": output.get("failed_row_count"),
        "log": str(log_path),
        "stdout_tail": proc.stdout[-2000:],
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "command",
        "returncode",
        "duration_sec",
        "passed",
        "output",
        "output_exists",
        "output_status",
        "row_count",
        "failed_row_count",
        "log",
    ]
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

    core = load_json("res/tests/core_math_unit_tests/core_math_unit_tests.json")
    api = load_json("res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json")
    runtime = load_json("res/code/reimpl_runtime_integration_audit/reimpl_runtime_integration_audit.json")
    coverage = load_json("res/tests/core_test_coverage_audit/core_test_coverage_audit.json")
    package = load_json("res/code/reimpl_package_audit/reimpl_package_audit.json")

    checks = {
        "all_steps_pass": all(step["passed"] for step in steps),
        "core_math_rows_at_least_27": core.get("row_count", 0) >= 27 and core.get("failed_row_count") == 0,
        "api_rows_8": api.get("row_count") == 8 and api.get("failed_row_count") == 0,
        "package_symbols_at_least_37": package.get("symbol_row_count", 0) >= 37,
        "runtime_window_count_84": runtime.get("metrics", {}).get("window_count") == 84,
        "runtime_token_shape_84_21_131": runtime.get("metrics", {}).get("token_shape") == [84, 21, 131],
        "coverage_required_20": coverage.get("required_count") == 20 and coverage.get("missing_count") == 0,
        "pure_numpy_no_isaac_ros_dependency": (
            core.get("checks", {}).get("pure_numpy_no_isaac_ros_dependency") is True
            and api.get("checks", {}).get("pure_numpy_no_isaac_ros_dependency") is True
            and coverage.get("checks", {}).get("pure_numpy_no_isaac_ros_dependency") is True
        ),
        "atomic_write_used": True,
        "does_not_claim_training_or_deployment": True,
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "reimpl_test_suite",
        "scope": "Unified execution of pure-Python clean-room formula/API/runtime tests and audits.",
        "step_count": len(steps),
        "pass_count": sum(1 for step in steps if step["passed"]),
        "steps": steps,
        "checks": checks,
        "metrics": {
            "core_math_row_count": core.get("row_count"),
            "api_row_count": api.get("row_count"),
            "package_symbol_count": package.get("symbol_row_count"),
            "runtime_window_count": runtime.get("metrics", {}).get("window_count"),
            "runtime_token_shape": runtime.get("metrics", {}).get("token_shape"),
            "coverage_required_count": coverage.get("required_count"),
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This suite executes the clean-room pure-Python formula/API/runtime checks. It does not execute "
                "IsaacLab/Kit rollouts, ROS 2 deployment, TensorRT, long PPO/VAE/diffusion training, Fig. 5/Fig. 6 "
                "paper evaluation, or real Unitree G1 hardware."
            ),
        },
        "outputs": {
            "json": str(OUT / "reimpl_test_suite.json"),
            "tsv": str(OUT / "reimpl_test_suite.tsv"),
            "log_dir": str(LOG_DIR),
        },
    }
    atomic_write_text(OUT / "reimpl_test_suite.json", json.dumps(summary, indent=2, sort_keys=True))
    write_tsv(OUT / "reimpl_test_suite.tsv", steps)
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
