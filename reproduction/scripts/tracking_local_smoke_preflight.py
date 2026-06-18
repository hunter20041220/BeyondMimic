#!/usr/bin/env python3
"""Run non-Kit preflight checks for the local whole_body_tracking smoke."""

from __future__ import annotations

import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/local_smoke_preflight"
LOG_DIR = ROOT / "logs/tracking_local_smoke_preflight"
GEN = ROOT / "reproduction/generated/whole_body_tracking_local"
FIXTURE_DIR = ROOT / "reproduction/data/tracking_motion_npz_fixtures"


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def run_step(name: str, command: list[str], timeout: int = 120) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    duration = time.perf_counter() - start
    log_path = LOG_DIR / f"{name}.log"
    atomic_write_text(log_path, proc.stdout)
    return {
        "name": name,
        "command": " ".join(command),
        "returncode": proc.returncode,
        "duration_sec": duration,
        "passed": proc.returncode == 0,
        "log": str(log_path),
        "stdout_tail": proc.stdout[-2000:],
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["name", "command", "returncode", "duration_sec", "passed", "log"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    steps: list[dict[str, Any]] = []
    steps.append(
        run_step(
            "prepare_tracking_local_smoke",
            ["python3", str(ROOT / "reproduction/scripts/prepare_tracking_local_smoke.py")],
        )
    )
    steps.append(run_step("runner_bash_syntax", ["bash", "-n", str(ROOT / "reproduction/scripts/run_tracking_local_smoke.sh")]))
    generated_scripts = [
        GEN / "csv_to_npz_local.py",
        GEN / "replay_npz_local.py",
        GEN / "rsl_rl/train_local.py",
        GEN / "rsl_rl/cli_args.py",
    ]
    steps.append(run_step("generated_script_compile", ["python3", "-m", "py_compile", *map(str, generated_scripts)]))

    fixture_npzs = sorted(FIXTURE_DIR.glob("*_debug_motion.npz"))
    validator_jsons = []
    for npz_path in fixture_npzs:
        summary_path = OUT / f"{npz_path.stem}_validator.json"
        validator_jsons.append(summary_path)
        steps.append(
            run_step(
                f"validate_{npz_path.stem}",
                [
                    "python3",
                    str(ROOT / "reproduction/scripts/validate_motion_npz_contract.py"),
                    str(npz_path),
                    "--summary-json",
                    str(summary_path),
                ],
            )
        )

    generated_exist = all(path.is_file() for path in generated_scripts)
    generated_executable = all(path.stat().st_mode & 0o111 for path in generated_scripts if path.is_file())
    validator_outputs_exist = all(path.is_file() and path.stat().st_size > 0 for path in validator_jsons)
    all_steps_passed = all(step["passed"] for step in steps)
    checks = {
        "prepare_script_executed": steps[0]["passed"] if steps else False,
        "runner_bash_syntax_valid": any(step["name"] == "runner_bash_syntax" and step["passed"] for step in steps),
        "generated_scripts_exist": generated_exist,
        "generated_scripts_executable": bool(generated_scripts) and generated_executable,
        "generated_scripts_compile": any(step["name"] == "generated_script_compile" and step["passed"] for step in steps),
        "fixture_count_3": len(fixture_npzs) == 3,
        "fixture_validators_pass": validator_outputs_exist and all(
            step["passed"] for step in steps if step["name"].startswith("validate_")
        ),
        "atomic_write_used": True,
        "does_not_launch_kit_or_training": True,
        "does_not_claim_tracking_reproduction_complete": True,
    }
    summary: dict[str, Any] = {
        "status": "ok" if all_steps_passed and all(checks.values()) else "failed",
        "experiment_type": "tracking_local_smoke_preflight",
        "scope": "Executed non-Kit local smoke preparation, generated-script syntax checks, and debug motion.npz validators.",
        "fixture_count": len(fixture_npzs),
        "step_count": len(steps),
        "pass_count": sum(1 for step in steps if step["passed"]),
        "generated_scripts": [str(path) for path in generated_scripts],
        "fixture_npzs": [str(path) for path in fixture_npzs],
        "validator_jsons": [str(path) for path in validator_jsons],
        "steps": steps,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This preflight executes only non-Kit preparation and validator checks. It does not run the official "
                "IsaacLab/Kit csv_to_npz, rendered replay, PPO training smoke, rollout evaluation, or paper-level "
                "tracking metrics."
            ),
        },
        "outputs": {
            "json": str(OUT / "tracking_local_smoke_preflight.json"),
            "tsv": str(OUT / "tracking_local_smoke_preflight.tsv"),
            "log_dir": str(LOG_DIR),
        },
    }
    atomic_write_text(OUT / "tracking_local_smoke_preflight.json", json.dumps(summary, indent=2, sort_keys=True))
    write_tsv(OUT / "tracking_local_smoke_preflight.tsv", steps)
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
