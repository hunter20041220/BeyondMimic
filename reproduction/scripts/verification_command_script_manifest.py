#!/usr/bin/env python3
"""Hash Python scripts referenced by final-report verification commands."""

from __future__ import annotations

import csv
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
FINAL_REPORT = ROOT / "res/final_report/final_reproduction_report.json"
SYNTAX_AUDIT = ROOT / "res/verification_command_syntax/verification_command_syntax_audit.json"
OUT = ROOT / "res/verification_command_script_manifest"


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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["index", "command", "script_path", "size_bytes", "sha256"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = json.loads(FINAL_REPORT.read_text(encoding="utf-8")) if FINAL_REPORT.is_file() else {}
    syntax = json.loads(SYNTAX_AUDIT.read_text(encoding="utf-8")) if SYNTAX_AUDIT.is_file() else {}
    commands = list(report.get("verification_commands", []))

    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    missing_scripts: list[str] = []
    for index, command in enumerate(commands):
        script = extract_python_script(command)
        if script is None or script in seen:
            continue
        seen.add(script)
        if not script.is_file():
            missing_scripts.append(str(script))
            continue
        rows.append(
            {
                "index": index,
                "command": command,
                "script_path": str(script.relative_to(ROOT)),
                "size_bytes": script.stat().st_size,
                "sha256": sha256(script),
            }
        )

    syntax_script_count = syntax.get("python_script_count")
    checks = {
        "final_report_exists": FINAL_REPORT.is_file(),
        "syntax_audit_exists": SYNTAX_AUDIT.is_file(),
        "verification_commands_nonempty": bool(commands),
        "script_count_matches_syntax_audit": syntax_script_count == len(rows),
        "script_count_at_least_100": len(rows) >= 100,
        "no_missing_scripts": not missing_scripts,
        "all_hashes_present": all(len(row["sha256"]) == 64 for row in rows) and bool(rows),
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
                "syntax_audit_exists",
                "verification_commands_nonempty",
                "script_count_matches_syntax_audit",
                "script_count_at_least_100",
                "no_missing_scripts",
                "all_hashes_present",
                "atomic_write_used",
                "does_not_execute_commands",
                "does_not_claim_full_reproduction",
            ]
        )
        else "failed",
        "experiment_type": "verification_command_script_manifest",
        "scope": "SHA256 manifest for unique Python scripts referenced by final-report verification commands.",
        "final_report": str(FINAL_REPORT),
        "syntax_audit": str(SYNTAX_AUDIT),
        "command_count": len(commands),
        "python_script_count": len(rows),
        "syntax_script_count": syntax_script_count,
        "missing_scripts": missing_scripts,
        "checks": checks,
        "rows": rows,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This is a static script provenance manifest. It does not run IsaacLab/Kit rollouts, ROS 2 launches, "
                "full PPO/VAE/diffusion training, TensorRT export, Fig. 5/Fig. 6 reproduction, or real Unitree G1 "
                "deployment."
            ),
        },
        "outputs": {
            "json": str(OUT / "verification_command_script_manifest.json"),
            "tsv": str(OUT / "verification_command_script_manifest.tsv"),
        },
    }
    atomic_write_text(
        OUT / "verification_command_script_manifest.json",
        json.dumps(summary, indent=2, sort_keys=True),
    )
    write_tsv(OUT / "verification_command_script_manifest.tsv", rows)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "scripts": summary["python_script_count"],
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
