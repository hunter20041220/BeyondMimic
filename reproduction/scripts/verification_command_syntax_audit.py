#!/usr/bin/env python3
"""Compile Python scripts referenced by final-report verification commands."""

from __future__ import annotations

import csv
import json
import py_compile
import shlex
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
FINAL_REPORT = ROOT / "res/final_report/final_reproduction_report.json"
OUT = ROOT / "res/verification_command_syntax"


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def extract_python_script(command: str) -> Path | None:
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    for part in parts:
        path = Path(part)
        if path.suffix == ".py" and str(path).startswith(str(ROOT)):
            return path
    return None


def compile_script(path: Path) -> tuple[bool, str]:
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:
        return False, str(exc)
    return True, ""


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "index",
        "command",
        "script_path",
        "script_exists",
        "compiled",
        "error",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = json.loads(FINAL_REPORT.read_text(encoding="utf-8")) if FINAL_REPORT.is_file() else {}
    commands = list(report.get("verification_commands", []))

    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    skipped_commands = 0
    for index, command in enumerate(commands):
        script = extract_python_script(command)
        if script is None:
            skipped_commands += 1
            continue
        if script in seen:
            continue
        seen.add(script)
        exists = script.is_file()
        compiled = False
        error = ""
        if exists:
            compiled, error = compile_script(script)
        else:
            error = "missing_script"
        rows.append(
            {
                "index": index,
                "command": command,
                "script_path": str(script.relative_to(ROOT)),
                "script_exists": exists,
                "compiled": compiled,
                "error": error or "none",
            }
        )

    failed_rows = [row for row in rows if not row["compiled"]]
    checks = {
        "final_report_exists": FINAL_REPORT.is_file(),
        "verification_commands_nonempty": bool(commands),
        "python_script_count_at_least_80": len(rows) >= 80,
        "all_scripts_exist": all(row["script_exists"] for row in rows) and bool(rows),
        "all_scripts_compile": not failed_rows and bool(rows),
        "atomic_write_used": True,
        "does_not_execute_commands": True,
        "does_not_claim_full_reproduction": True,
    }
    summary: dict[str, Any] = {
        "status": "ok"
        if all(
            checks[key]
            for key in [
                "final_report_exists",
                "verification_commands_nonempty",
                "python_script_count_at_least_80",
                "all_scripts_exist",
                "all_scripts_compile",
                "atomic_write_used",
                "does_not_execute_commands",
                "does_not_claim_full_reproduction",
            ]
        )
        else "failed",
        "experiment_type": "verification_command_syntax_audit",
        "scope": "Static syntax compilation for unique Python scripts referenced by final-report verification commands.",
        "final_report": str(FINAL_REPORT),
        "command_count": len(commands),
        "python_script_count": len(rows),
        "skipped_command_count": skipped_commands,
        "failed_count": len(failed_rows),
        "failed_rows": failed_rows,
        "checks": checks,
        "rows": rows,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This audit compiles referenced Python scripts only. It does not run IsaacLab/Kit rollouts, ROS 2 "
                "launches, full PPO/VAE/diffusion training, TensorRT export, Fig. 5/Fig. 6 reproduction, or real "
                "Unitree G1 deployment."
            ),
        },
        "outputs": {
            "json": str(OUT / "verification_command_syntax_audit.json"),
            "tsv": str(OUT / "verification_command_syntax_audit.tsv"),
        },
    }
    atomic_write_text(OUT / "verification_command_syntax_audit.json", json.dumps(summary, indent=2, sort_keys=True))
    write_tsv(OUT / "verification_command_syntax_audit.tsv", rows)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "scripts": summary["python_script_count"],
                "failed": summary["failed_count"],
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
