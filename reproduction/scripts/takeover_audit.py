#!/usr/bin/env python3
"""Takeover audit for the migrated BeyondMimic workspace.

This script is intentionally non-training: it checks host/runtime state,
workspace layout, readable evidence artifacts, and basic script syntax/import
surfaces after moving the old work area to a new absolute ROOT.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OLD_ROOT = "/shared_disk/zzy/BeyondMimic"
OUT = ROOT / "res/takeover_audit"
LOG_DIR = ROOT / "logs/takeover_audit"


KEY_FILES = [
    "goal.md",
    "other/goal.md",
    "README.md",
    "other/README.md",
    "reproduction/PROGRESS.md",
    "reproduction/RUNBOOK.md",
    "reproduction/docs/final_reproduction_report.md",
    "reproduction/docs/known_limitations.md",
    "reproduction/docs/experiment_protocol.md",
    "res/comparison/paper_vs_reproduction.json",
    "res/artifact_manifest/artifact_manifest.json",
    "res/master_audit/reproduction_master_audit.json",
    "res/required_artifact_absence/required_artifact_absence_audit.json",
]

JSON_ARTIFACTS = [
    "res/comparison/paper_vs_reproduction.json",
    "res/artifact_manifest/artifact_manifest.json",
    "res/master_audit/reproduction_master_audit.json",
    "res/required_artifact_absence/required_artifact_absence_audit.json",
]

SMOKE_SCRIPT_PATHS = [
    "reproduction/scripts/artifact_manifest.py",
    "reproduction/scripts/paper_vs_reproduction_comparison.py",
    "reproduction/scripts/final_reproduction_report.py",
    "reproduction/scripts/completion_matrix_status_audit.py",
    "reproduction/scripts/verification_command_syntax_audit.py",
    "reproduction/scripts/verification_command_script_manifest.py",
    "reproduction/scripts/verification_command_coverage_audit.py",
    "reproduction/scripts/reproduction_master_audit.py",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_command(name: str, command: list[str], timeout: int = 60) -> dict:
    started = now_iso()
    try:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        status = {
            "name": name,
            "command": command,
            "started_at": started,
            "finished_at": now_iso(),
            "returncode": proc.returncode,
            "stdout_tail": proc.stdout[-6000:],
            "stderr_tail": proc.stderr[-6000:],
        }
    except FileNotFoundError as exc:
        status = {
            "name": name,
            "command": command,
            "started_at": started,
            "finished_at": now_iso(),
            "returncode": 127,
            "stdout_tail": "",
            "stderr_tail": str(exc),
        }
    except subprocess.TimeoutExpired as exc:
        status = {
            "name": name,
            "command": command,
            "started_at": started,
            "finished_at": now_iso(),
            "returncode": 124,
            "stdout_tail": (exc.stdout or "")[-6000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-6000:] if isinstance(exc.stderr, str) else "",
            "timeout_sec": timeout,
        }
    return status


def count_files(path: Path) -> int | None:
    if not path.exists():
        return None
    return sum(1 for p in path.rglob("*") if p.is_file())


def dir_summary(path: Path) -> dict:
    exists = path.exists()
    return {
        "path": str(path),
        "exists": exists,
        "is_dir": path.is_dir(),
        "file_count": count_files(path) if exists else None,
        "writable": os.access(path, os.W_OK) if exists else None,
    }


def read_json_summary(rel: str) -> dict:
    path = ROOT / rel
    item = {
        "path": rel,
        "exists": path.is_file(),
        "sha256": sha256(path),
        "readable_json": False,
        "top_level_keys": [],
    }
    if not path.is_file():
        return item
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - audit must preserve parse error.
        item["error"] = repr(exc)
        return item
    item["readable_json"] = True
    if isinstance(data, dict):
        item["top_level_keys"] = sorted(data.keys())[:40]
        for key in ("status", "artifact_count", "total_rows", "row_count", "goal_complete"):
            if key in data:
                item[key] = data[key]
        if "completion_matrix_counts" in data:
            item["completion_matrix_counts"] = data["completion_matrix_counts"]
        if "status_counts" in data:
            item["status_counts"] = data["status_counts"]
    elif isinstance(data, list):
        item["list_length"] = len(data)
    return item


def key_file_summary(rel: str) -> dict:
    path = ROOT / rel
    return {
        "path": rel,
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256(path),
    }


def old_path_scan() -> dict:
    command = [
        "rg",
        "-l",
        OLD_ROOT,
        "goal.md",
        "README.md",
        "reproduction",
        "res",
        "logs",
        "--glob",
        "!**/*.pt",
        "--glob",
        "!**/*.pth",
        "--glob",
        "!**/*.ckpt",
        "--glob",
        "!**/*.onnx",
        "--glob",
        "!**/*.npz",
        "--glob",
        "!**/*.npy",
        "--glob",
        "!**/*.mcap",
        "--glob",
        "!**/*.png",
        "--glob",
        "!**/*.pdf",
        "--glob",
        "!**/*.zip",
        "--glob",
        "!**/*.tar",
        "--glob",
        "!**/*.gz",
        "--glob",
        "!**/*.rar",
        "--glob",
        "!**/*.dae",
        "--glob",
        "!**/*.usd",
        "--glob",
        "!**/*.usda",
        "--glob",
        "!**/*.obj",
        "--glob",
        "!**/*.stl",
        "--glob",
        "!**/*.mp4",
        "--glob",
        "!**/*.avi",
        "--glob",
        "!**/*.mkv",
    ]
    result = run_command("old_root_text_path_scan", command, timeout=120)
    files = [line for line in result["stdout_tail"].splitlines() if line.strip()]
    expected_self_references = {
        "reproduction/scripts/takeover_audit.py",
        "res/takeover_audit/takeover_audit.json",
    }
    unexpected_files = [f for f in files if f not in expected_self_references]
    result["matching_file_count_from_tail"] = len(files)
    result["expected_self_reference_files"] = sorted(set(files) & expected_self_references)
    result["unexpected_matching_files"] = unexpected_files
    result["unexpected_matching_file_count"] = len(unexpected_files)
    result["status"] = "ok" if result["returncode"] in (0, 1) and not unexpected_files else "old_paths_remaining"
    return result


def python_syntax_checks() -> list[dict]:
    rows = []
    for rel in SMOKE_SCRIPT_PATHS:
        path = ROOT / rel
        if not path.is_file():
            rows.append({"script": rel, "exists": False, "returncode": None})
            continue
        rows.append(run_command(f"py_compile:{rel}", [sys.executable, "-m", "py_compile", str(path)], timeout=60))
        rows[-1]["script"] = rel
        rows[-1]["exists"] = True
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for rel in ("envs", "cache", "tmp", "logs", "res"):
        (ROOT / rel).mkdir(parents=True, exist_ok=True)

    analysis_python = ROOT / "envs/bm_analysis/bin/python"
    diffusion_python = ROOT / "envs/bm_diffusion/bin/python"
    tracking_python = ROOT / "envs/bm_tracking/bin/python"

    commands = [
        ("uname", ["uname", "-a"]),
        ("os_release", ["bash", "-lc", "cat /etc/os-release 2>/dev/null || true"]),
        ("python_version", [sys.executable, "--version"]),
        ("python_executable", ["bash", "-lc", "command -v python3 && python3 -c 'import sys; print(sys.executable)'"]),
        ("nvidia_smi", ["nvidia-smi"]),
        ("nvcc_version", ["bash", "-lc", "command -v nvcc && nvcc --version"]),
        ("cuda_visible_devices", ["bash", "-lc", "printf '%s\\n' \"${CUDA_VISIBLE_DEVICES-<unset>}\""]),
        (
            "torch_cuda_probe_devices_5_6",
            [
                "bash",
                "-lc",
                (
                    f"CUDA_VISIBLE_DEVICES=5,6 PYTHONNOUSERSITE=1 {diffusion_python} -c "
                    "'import torch; print(torch.__version__); print(torch.cuda.is_available()); "
                    "print(torch.cuda.device_count()); "
                    "[print(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]'"
                ),
            ],
        ),
        (
            "analysis_import_probe",
            [
                "bash",
                "-lc",
                f"PYTHONNOUSERSITE=1 {analysis_python} -c 'import json, numpy, pandas, matplotlib; print(\"analysis_imports_ok\")'",
            ],
        ),
        (
            "onnxruntime_import_probe",
            [
                "bash",
                "-lc",
                f"PYTHONNOUSERSITE=1 {analysis_python} -c 'import onnxruntime as ort; print(ort.__version__, ort.__file__)'",
            ],
        ),
        (
            "isaaclab_import_probe",
            [
                "bash",
                "-lc",
                f"PYTHONNOUSERSITE=1 {tracking_python} -c 'import isaaclab; print(\"isaaclab_import_ok\")'",
            ],
        ),
        ("disk_root", ["df", "-h", str(ROOT)]),
        ("disk_inodes_root", ["df", "-ih", str(ROOT)]),
        ("download_permissions", ["bash", "-lc", "stat -c '%A %U %G %n' download other reproduction res logs"]),
        ("top_level_sizes", ["du", "-sh", "download", "other", "reproduction", "res", "logs"]),
        ("git_status", ["bash", "-lc", "git status --short --branch 2>&1 || true"]),
    ]
    command_results = [run_command(name, cmd, timeout=90) for name, cmd in commands]

    dirs = {
        name: dir_summary(ROOT / name)
        for name in ("download", "other", "reproduction", "res", "logs", "envs", "cache", "tmp")
    }
    key_files = [key_file_summary(rel) for rel in KEY_FILES]
    json_artifacts = [read_json_summary(rel) for rel in JSON_ARTIFACTS]
    syntax_rows = python_syntax_checks()
    old_paths = old_path_scan()

    command_failures = [r for r in command_results if r.get("returncode") not in (0,)]
    missing_dirs = [name for name, row in dirs.items() if not row["exists"]]
    missing_key_files = [row["path"] for row in key_files if not row["exists"]]
    unreadable_json = [row["path"] for row in json_artifacts if not row["readable_json"]]
    syntax_failures = [row for row in syntax_rows if row.get("returncode") not in (0,)]
    hard_failures = missing_dirs + missing_key_files + unreadable_json + [r.get("script", r["name"]) for r in syntax_failures]

    audit = {
        "status": "ok_with_runtime_warnings" if command_failures and not hard_failures else ("ok" if not hard_failures else "failed"),
        "generated_at": now_iso(),
        "scope": "takeover_audit_after_root_migration_no_training",
        "root": str(ROOT),
        "old_root": OLD_ROOT,
        "platform": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "cwd": str(Path.cwd()),
            "uid": os.getuid() if hasattr(os, "getuid") else None,
        },
        "directories": dirs,
        "key_files": key_files,
        "json_artifacts": json_artifacts,
        "command_results": command_results,
        "command_failures": command_failures,
        "syntax_checks": syntax_rows,
        "old_root_scan": old_paths,
        "checks": {
            "download_present": dirs["download"]["exists"],
            "other_backup_present": dirs["other"]["exists"],
            "workspace_promoted": dirs["reproduction"]["exists"] and dirs["res"]["exists"] and dirs["logs"]["exists"],
            "key_files_present": not missing_key_files,
            "json_artifacts_readable": not unreadable_json,
            "smoke_scripts_compile": not syntax_failures,
            "old_root_text_paths_absent": old_paths["status"] == "ok",
            "download_treated_read_only": True,
            "training_started": False,
        },
        "missing_dirs": missing_dirs,
        "missing_key_files": missing_key_files,
        "unreadable_json_artifacts": unreadable_json,
        "interpretation": (
            "This takeover audit checks the migrated workspace only. It does not train policies, "
            "does not create checkpoints/videos, and does not claim full BeyondMimic reproduction."
        ),
    }

    json_path = OUT / "takeover_audit.json"
    md_path = OUT / "takeover_audit.md"
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# Takeover Audit",
        "",
        f"- Status: `{audit['status']}`",
        f"- Generated at: `{audit['generated_at']}`",
        f"- ROOT: `{ROOT}`",
        f"- Missing key files: `{len(missing_key_files)}`",
        f"- Unreadable JSON artifacts: `{len(unreadable_json)}`",
        f"- Runtime command warnings/failures: `{len(command_failures)}`",
        f"- Script syntax failures: `{len(syntax_failures)}`",
        f"- Old ROOT text path scan: `{old_paths['status']}`",
        "",
        "## Runtime Warnings",
    ]
    if command_failures:
        for row in command_failures:
            lines.append(f"- `{row['name']}` returned `{row['returncode']}`")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "No training, rollout, checkpoint fabrication, video generation, TensorRT benchmarking, or real-robot execution was performed.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if not hard_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
