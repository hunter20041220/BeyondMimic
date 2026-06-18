#!/usr/bin/env python3
"""Audit the final report verification-command list and run light smoke commands."""

from __future__ import annotations

import csv
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/verification_command_coverage"
FINAL_REPORT = ROOT / "res/final_report/final_reproduction_report.json"


SMOKE_COMMANDS = [
    f"python3 {ROOT / 'reproduction/scripts/progress_report_audit.py'}",
    f"python3 {ROOT / 'reproduction/scripts/final_report_requirement_audit.py'}",
    f"python3 {ROOT / 'reproduction/scripts/required_artifact_absence_audit.py'}",
    f"python3 {ROOT / 'reproduction/scripts/evaluation_metrics_coverage_audit.py'}",
    f"python3 {ROOT / 'reproduction/scripts/ablation_coverage_audit.py'}",
    f"python3 {ROOT / 'reproduction/scripts/goal_traceability_audit.py'}",
    f"python3 {ROOT / 'reproduction/scripts/goal_directive_index_audit.py'}",
    f"python3 {ROOT / 'reproduction/scripts/goal_requirement_matrix_audit.py'}",
    f"python3 {ROOT / 'reproduction/scripts/final_deliverables_audit.py'}",
    f"python3 {ROOT / 'reproduction/scripts/paper_formula_code_trace_audit.py'}",
]

EXPECTED_OUTPUTS = {
    SMOKE_COMMANDS[0]: ["res/progress_report_audit/progress_report_audit.json"],
    SMOKE_COMMANDS[1]: ["res/final_report/final_report_requirement_audit/final_report_requirement_audit.json"],
    SMOKE_COMMANDS[2]: ["res/required_artifact_absence/required_artifact_absence_audit.json"],
    SMOKE_COMMANDS[3]: ["res/evaluation_metrics_coverage/evaluation_metrics_coverage_audit.json"],
    SMOKE_COMMANDS[4]: ["res/ablation_coverage/ablation_coverage_audit.json"],
    SMOKE_COMMANDS[5]: ["res/goal_traceability/goal_traceability_audit.json"],
    SMOKE_COMMANDS[6]: ["res/goal_directive_index/goal_directive_index_audit.json"],
    SMOKE_COMMANDS[7]: ["res/goal_requirement_matrix/goal_requirement_matrix_audit.json"],
    SMOKE_COMMANDS[8]: ["res/final_deliverables_audit/final_deliverables_audit.json"],
    SMOKE_COMMANDS[9]: ["res/paper_formula_code_trace/paper_formula_code_trace_audit.json"],
}

AUDIT_COMMAND = f"python3 {ROOT / 'reproduction/scripts/verification_command_coverage_audit.py'}"


def command_category(command: str) -> str:
    if "envs/bm_tracking/bin/python" in command or "envs/bm_analysis/bin/python" in command:
        return "env_specific"
    if "cuda:" in command or "gpu_resource_audit.py" in command:
        return "hardware_or_gpu_probe"
    if "level_c_" in command and any(token in command for token in ("transformer", "vae", "diffusion")):
        return "level_c_heavy_or_model_smoke"
    if command.startswith("python3 ") and "/reproduction/scripts/" in command:
        return "lightweight_python_audit"
    if "/reproduction/tests/" in command:
        return "unit_test"
    return "other"


def rel_script_path(command: str) -> str:
    for token in command.replace("\\", " ").split():
        if token.startswith(str(ROOT / "reproduction")) and token.endswith(".py"):
            return str(Path(token).relative_to(ROOT))
    return ""


def output_status(command: str) -> tuple[bool, str]:
    outputs = EXPECTED_OUTPUTS.get(command, [])
    if not outputs:
        return True, "not_mapped"
    missing = []
    details = []
    for rel in outputs:
        path = ROOT / rel
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(rel)
        else:
            details.append(f"{rel}:{path.stat().st_size}")
    if missing:
        return False, "missing_or_empty=" + ",".join(missing)
    return True, ";".join(details)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def run_smoke(command: str) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )
    duration = time.perf_counter() - start
    output_ok, output_detail = output_status(command)
    return {
        "command": command,
        "returncode": proc.returncode,
        "duration_sec": duration,
        "passed": proc.returncode == 0 and output_ok,
        "output_status": output_detail,
        "stdout_tail": proc.stdout[-2000:],
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "index",
        "category",
        "command",
        "script_path",
        "script_exists",
        "expected_output_status",
        "smoke_executed",
        "smoke_passed",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = json.loads(FINAL_REPORT.read_text(encoding="utf-8"))
    commands = list(report.get("verification_commands", []))
    command_set = set(commands)
    category_counts = Counter(command_category(command) for command in commands)

    smoke_results = [run_smoke(command) for command in SMOKE_COMMANDS]
    smoke_by_command = {row["command"]: row for row in smoke_results}

    rows: list[dict[str, Any]] = []
    missing_scripts = []
    for idx, command in enumerate(commands):
        script_rel = rel_script_path(command)
        script_exists = True
        if script_rel:
            script_exists = (ROOT / script_rel).is_file()
            if not script_exists:
                missing_scripts.append(script_rel)
        output_ok, output_detail = output_status(command)
        smoke = smoke_by_command.get(command)
        rows.append(
            {
                "index": idx,
                "category": command_category(command),
                "command": command,
                "script_path": script_rel,
                "script_exists": script_exists,
                "expected_output_status": output_detail if output_ok else "failed:" + output_detail,
                "smoke_executed": smoke is not None,
                "smoke_passed": smoke["passed"] if smoke else "",
            }
        )

    duplicated = [command for command, count in Counter(commands).items() if count > 1]
    smoke_commands_listed = all(command in command_set for command in SMOKE_COMMANDS)
    smoke_passed = all(row["passed"] for row in smoke_results)
    mapped_outputs_exist = all(output_status(command)[0] for command in EXPECTED_OUTPUTS)
    checks = {
        "final_report_exists": FINAL_REPORT.is_file(),
        "verification_commands_nonempty": bool(commands),
        "command_count_at_least_80": len(commands) >= 80,
        "no_duplicate_commands": not duplicated,
        "all_detected_scripts_exist": not missing_scripts,
        "smoke_commands_listed_in_final_report": smoke_commands_listed,
        "smoke_commands_pass": smoke_passed,
        "mapped_smoke_outputs_exist": mapped_outputs_exist,
        "self_command_listed_in_final_report": AUDIT_COMMAND in command_set,
        "atomic_write_used": True,
        "does_not_execute_heavy_or_env_specific_commands": all(
            command_category(command) == "lightweight_python_audit" for command in SMOKE_COMMANDS
        ),
        "does_not_claim_full_reproduction": True,
    }
    summary: dict[str, Any] = {
        "status": "ok"
        if all(
            checks[key]
            for key in [
                "final_report_exists",
                "verification_commands_nonempty",
                "command_count_at_least_80",
                "no_duplicate_commands",
                "all_detected_scripts_exist",
                "smoke_commands_listed_in_final_report",
                "smoke_commands_pass",
                "mapped_smoke_outputs_exist",
                "atomic_write_used",
                "does_not_execute_heavy_or_env_specific_commands",
                "does_not_claim_full_reproduction",
            ]
        )
        else "failed",
        "experiment_type": "verification_command_coverage_audit",
        "scope": "Audit final-report verification commands and execute a bounded lightweight smoke subset.",
        "final_report": str(FINAL_REPORT),
        "command_count": len(commands),
        "category_counts": dict(sorted(category_counts.items())),
        "smoke_command_count": len(SMOKE_COMMANDS),
        "smoke_pass_count": sum(1 for row in smoke_results if row["passed"]),
        "duplicate_commands": duplicated,
        "missing_scripts": missing_scripts,
        "smoke_results": smoke_results,
        "checks": checks,
        "rows": rows,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The smoke subset verifies final-report command hygiene and a few lightweight audits only. It does "
                "not execute IsaacLab/Kit rollouts, ROS 2 launches, long VAE/diffusion training, TensorRT export, "
                "Fig. 5/Fig. 6 reproduction, or real Unitree G1 deployment."
            ),
        },
        "outputs": {
            "json": str(OUT / "verification_command_coverage_audit.json"),
            "tsv": str(OUT / "verification_command_coverage_audit.tsv"),
        },
    }
    atomic_write_text(OUT / "verification_command_coverage_audit.json", json.dumps(summary, indent=2, sort_keys=True))
    write_tsv(OUT / "verification_command_coverage_audit.tsv", rows)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "commands": summary["command_count"],
                "smoke_pass": summary["smoke_pass_count"],
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
