#!/usr/bin/env python3
"""Audit non-Kit importability of official whole_body_tracking modules."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/tracking_import_gate_audit"
PYTHON_EXE = ROOT / "envs/bm_tracking/bin/python"
ISAACSIM_CORE_API = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/exts/isaacsim.core.api/isaacsim/core/api/__init__.py"
)
TRACKING_ROOT = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking"
)

MODULES = [
    "isaaclab",
    "isaacsim.core.api",
    "whole_body_tracking",
    "whole_body_tracking.robots.g1",
    "whole_body_tracking.tasks.tracking.tracking_env_cfg",
    "whole_body_tracking.tasks.tracking.mdp.commands",
    "whole_body_tracking.tasks.tracking.mdp.rewards",
]


def import_modules(modules: list[str], timeout: int = 120) -> list[dict[str, Any]]:
    code = (
        "import importlib, json\n"
        f"modules={modules!r}\n"
        "rows=[]\n"
        "for module in modules:\n"
        "    try:\n"
        "        m=importlib.import_module(module)\n"
        "        rows.append({'module': module, 'ok': True, 'file': getattr(m, '__file__', None)})\n"
        "    except Exception as exc:\n"
        "        rows.append({'module': module, 'ok': False, 'error_type': type(exc).__name__, 'error': str(exc)})\n"
        "print(json.dumps(rows))\n"
    )
    try:
        proc = subprocess.run(
            [str(PYTHON_EXE), "-c", code],
            cwd=str(ROOT),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            env={
                **dict(__import__("os").environ),
                "PYTHONNOUSERSITE": "1",
                "OMNI_KIT_ACCEPT_EULA": "YES",
            },
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        return [
            {
                "module": module,
                "ok": False,
                "return_code": 124,
                "error_type": "TimeoutExpired",
                "error": f"batch timeout after {timeout}s",
                "stdout": stdout[-1000:],
            }
            for module in modules
        ]
    text = proc.stdout.strip()
    payload: list[dict[str, Any]] | None = None
    for line in reversed(text.splitlines()):
        try:
            loaded = json.loads(line)
            if isinstance(loaded, list):
                payload = loaded
                break
            if isinstance(loaded, dict):
                payload = [loaded]
                break
        except json.JSONDecodeError:
            continue
    if payload is None:
        payload = [
            {"module": module, "ok": False, "error_type": "UnparsedOutput", "error": text[-1000:]}
            for module in modules
        ]
    for row in payload:
        row["return_code"] = proc.returncode
        row["stdout_tail"] = text[-1000:]
    return payload


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    imports = import_modules(MODULES)
    by_module = {row["module"]: row for row in imports}
    tracking_py_files = sorted(str(path) for path in TRACKING_ROOT.rglob("*.py"))
    expected_failures = [module for module in MODULES if module != "isaaclab" and by_module[module]["ok"] is False]
    expected_failure_errors = {
        module: by_module[module].get("error", "") for module in expected_failures
    }
    tracking_deep_failures = [module for module in expected_failures if module.startswith("whole_body_tracking")]
    kit_namespace_markers = ("No module named 'isaacsim.core'", "No module named 'omni.kit'")
    rows = [
        {
            "check": "isaaclab_plain_import_passes",
            "status": "pass" if by_module["isaaclab"]["ok"] else "fail",
            "evidence": str(PYTHON_EXE),
            "detail": json.dumps(by_module["isaaclab"], sort_keys=True),
        },
        {
            "check": "isaacsim_core_extension_source_exists",
            "status": "pass" if ISAACSIM_CORE_API.exists() else "fail",
            "evidence": str(ISAACSIM_CORE_API),
            "detail": f"exists={ISAACSIM_CORE_API.exists()}",
        },
        {
            "check": "plain_python_does_not_enable_isaacsim_core_extension",
            "status": "pass"
            if by_module["isaacsim.core.api"]["ok"] is False
            and "No module named 'isaacsim.core'" in by_module["isaacsim.core.api"].get("error", "")
            else "fail",
            "evidence": str(PYTHON_EXE),
            "detail": json.dumps(by_module["isaacsim.core.api"], sort_keys=True),
        },
        {
            "check": "whole_body_tracking_deep_imports_blocked_by_kit_namespace",
            "status": "pass"
            if len(tracking_deep_failures) >= 5
            and all(
                any(marker in expected_failure_errors[m] for marker in kit_namespace_markers)
                for m in tracking_deep_failures
            )
            else "fail",
            "evidence": str(TRACKING_ROOT),
            "detail": json.dumps(expected_failure_errors, sort_keys=True),
        },
        {
            "check": "official_tracking_python_sources_present",
            "status": "pass" if len(tracking_py_files) >= 15 else "fail",
            "evidence": str(TRACKING_ROOT),
            "detail": f"python_file_count={len(tracking_py_files)}",
        },
        {
            "check": "does_not_launch_kit_or_training",
            "status": "pass",
            "evidence": __file__,
            "detail": "Subprocess import probes only; no SimulationApp/AppLauncher, Kit app, PPO, replay, or training is run.",
        },
    ]
    failed = [row for row in rows if row["status"] == "fail"]
    summary: dict[str, Any] = {
        "status": "ok" if not failed else "failed",
        "experiment_type": "tracking_import_gate_audit",
        "scope": "non-Kit import boundary for official whole_body_tracking and Isaac Sim extension modules",
        "row_count": len(rows),
        "failed_row_count": len(failed),
        "rows": rows,
        "module_imports": imports,
        "metrics": {
            "module_count": len(MODULES),
            "import_ok_count": sum(1 for row in imports if row["ok"]),
            "import_fail_count": sum(1 for row in imports if not row["ok"]),
            "tracking_python_file_count": len(tracking_py_files),
            "expected_kit_namespace_failure_count": len(expected_failures),
        },
        "checks": {
            "isaaclab_plain_import_passes": by_module["isaaclab"]["ok"] is True,
            "isaacsim_core_extension_source_exists": ISAACSIM_CORE_API.exists(),
            "plain_python_lacks_isaacsim_core_extension": by_module["isaacsim.core.api"]["ok"] is False,
            "tracking_deep_imports_blocked_by_kit_namespace": len(tracking_deep_failures) >= 5
            and all(
                any(marker in expected_failure_errors[m] for marker in kit_namespace_markers)
                for m in tracking_deep_failures
            ),
            "official_tracking_python_sources_present": len(tracking_py_files) >= 15,
            "does_not_launch_kit_or_training": True,
            "does_not_claim_tracking_reproduction_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The bm_tracking Python can import pip Isaac Sim and local editable IsaacLab, but plain Python does "
                "not enable the isaacsim.core/omni.kit extension namespaces. "
                "Official whole_body_tracking deep config modules depend on Kit extension namespaces and remain gated "
                "by a live Kit/SimulationApp extension-manager context, which is currently blocked by host Kit/Vulkan "
                "startup errors recorded in logs/setup/isaaclab_headless_app_gate.log."
            ),
        },
        "outputs": {
            "json": str(OUT / "tracking_import_gate_audit.json"),
            "tsv": str(OUT / "tracking_import_gate_audit.tsv"),
        },
    }
    (OUT / "tracking_import_gate_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (OUT / "tracking_import_gate_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["check", "status", "evidence", "detail"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
