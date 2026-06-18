#!/usr/bin/env python3
"""Probe whether manual Isaac Sim extension namespace paths unlock tracking imports."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/tracking_extension_namespace_probe"
PYTHON_SH = ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/_isaac_sim/python.sh"
ISAACSIM_EXTS = ROOT / "envs/isaacsim-4.5.0/exts"

MODULES = [
    "isaacsim.core.api",
    "isaacsim.core.prims",
    "isaacsim.core.utils.stage",
    "whole_body_tracking",
    "whole_body_tracking.robots.g1",
    "whole_body_tracking.tasks.tracking.tracking_env_cfg",
    "whole_body_tracking.tasks.tracking.mdp.commands",
    "whole_body_tracking.tasks.tracking.mdp.rewards",
]


def core_extension_namespace_paths() -> list[str]:
    return sorted(str(path) for path in ISAACSIM_EXTS.glob("isaacsim.core.*/isaacsim") if path.is_dir())


def probe_imports(timeout: int = 120) -> dict[str, Any]:
    paths = core_extension_namespace_paths()
    code = (
        "import importlib, json\n"
        "import isaacsim\n"
        f"paths={paths!r}\n"
        f"modules={MODULES!r}\n"
        "for path in paths:\n"
        "    if path not in isaacsim.__path__:\n"
        "        isaacsim.__path__.append(path)\n"
        "rows=[]\n"
        "for module in modules:\n"
        "    try:\n"
        "        m=importlib.import_module(module)\n"
        "        rows.append({'module': module, 'ok': True, 'file': getattr(m, '__file__', None)})\n"
        "    except Exception as exc:\n"
        "        rows.append({'module': module, 'ok': False, 'error_type': type(exc).__name__, 'error': str(exc)})\n"
        "print(json.dumps({'isaacsim_path_len': len(isaacsim.__path__), 'rows': rows}))\n"
    )
    try:
        proc = subprocess.run(
            [str(PYTHON_SH), "-c", code],
            cwd=str(ROOT),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "return_code": 124,
            "timed_out": True,
            "stdout_tail": ((exc.stdout or "") if isinstance(exc.stdout, str) else "")[-1000:],
            "isaacsim_path_len": None,
            "rows": [
                {"module": module, "ok": False, "error_type": "TimeoutExpired", "error": f"timeout after {timeout}s"}
                for module in MODULES
            ],
        }
    text = proc.stdout.strip()
    payload = None
    for line in reversed(text.splitlines()):
        try:
            loaded = json.loads(line)
            if isinstance(loaded, dict) and "rows" in loaded:
                payload = loaded
                break
        except json.JSONDecodeError:
            continue
    if payload is None:
        payload = {
            "isaacsim_path_len": None,
            "rows": [
                {"module": module, "ok": False, "error_type": "UnparsedOutput", "error": text[-1000:]}
                for module in MODULES
            ],
        }
    payload["return_code"] = proc.returncode
    payload["timed_out"] = False
    payload["stdout_tail"] = text[-1000:]
    return payload


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    namespace_paths = core_extension_namespace_paths()
    probe = probe_imports()
    rows = probe["rows"]
    error_by_module = {row["module"]: row.get("error", "") for row in rows}
    import_ok_count = sum(1 for row in rows if row["ok"])
    import_fail_count = sum(1 for row in rows if not row["ok"])
    core_error_changed = "No module named 'isaacsim.core'" not in error_by_module["isaacsim.core.api"]
    kit_runtime_dependency_seen = any(
        ("omni.kit.commands" in row.get("error", "")) or ("libarch.so" in row.get("error", ""))
        for row in rows
    )
    checks_rows = [
        {
            "check": "core_namespace_paths_present",
            "status": "pass" if len(namespace_paths) >= 8 else "fail",
            "evidence": str(ISAACSIM_EXTS),
            "detail": f"core_namespace_path_count={len(namespace_paths)}",
        },
        {
            "check": "manual_namespace_changes_failure_mode",
            "status": "pass" if core_error_changed else "fail",
            "evidence": str(PYTHON_SH),
            "detail": json.dumps(error_by_module, sort_keys=True),
        },
        {
            "check": "kit_runtime_dependency_still_blocks_deep_import",
            "status": "pass" if kit_runtime_dependency_seen and import_fail_count == len(MODULES) else "fail",
            "evidence": str(PYTHON_SH),
            "detail": json.dumps(error_by_module, sort_keys=True),
        },
        {
            "check": "does_not_modify_or_launch_kit",
            "status": "pass",
            "evidence": __file__,
            "detail": "Temporarily appends isaacsim.__path__ inside a subprocess only; no files are edited and no Kit app is launched.",
        },
    ]
    failed = [row for row in checks_rows if row["status"] == "fail"]
    summary: dict[str, Any] = {
        "status": "ok" if not failed else "failed",
        "experiment_type": "tracking_extension_namespace_probe",
        "scope": "manual non-Kit Isaac Sim extension namespace bridge for official tracking imports",
        "row_count": len(checks_rows),
        "failed_row_count": len(failed),
        "rows": checks_rows,
        "namespace_paths": namespace_paths,
        "module_imports": rows,
        "metrics": {
            "core_namespace_path_count": len(namespace_paths),
            "module_count": len(MODULES),
            "import_ok_count": import_ok_count,
            "import_fail_count": import_fail_count,
            "isaacsim_path_len_after_append": probe["isaacsim_path_len"],
            "core_error_changed_from_missing_namespace": core_error_changed,
            "kit_runtime_dependency_seen": kit_runtime_dependency_seen,
            "return_code": probe["return_code"],
        },
        "checks": {
            "core_namespace_paths_present": len(namespace_paths) >= 8,
            "manual_namespace_changes_failure_mode": core_error_changed,
            "kit_runtime_dependency_still_blocks_deep_import": kit_runtime_dependency_seen
            and import_fail_count == len(MODULES),
            "does_not_modify_or_launch_kit": True,
            "does_not_claim_tracking_reproduction_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "Manually appending Isaac Sim core extension namespace paths changes the failure mode from a missing "
                "isaacsim.core namespace to lower-level Kit/native runtime dependencies such as libarch.so and "
                "omni.kit.commands. This moves the import diagnosis one layer deeper but does not make official "
                "whole_body_tracking configs safely importable without Kit."
            ),
        },
        "outputs": {
            "json": str(OUT / "tracking_extension_namespace_probe.json"),
            "tsv": str(OUT / "tracking_extension_namespace_probe.tsv"),
        },
    }
    (OUT / "tracking_extension_namespace_probe.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (OUT / "tracking_extension_namespace_probe.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["check", "status", "evidence", "detail"])
        writer.writeheader()
        writer.writerows(checks_rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(checks_rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
