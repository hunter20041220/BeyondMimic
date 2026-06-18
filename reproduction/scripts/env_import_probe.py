#!/usr/bin/env python3
"""Probe restored conda environments without launching training."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/setup/env_probe"
LOG = ROOT / "logs/env_probe/env_import_probe.log"
LEGACY_LIVE_GATE = ROOT / "res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json"
CURRENT_HEADLESS_GATE = ROOT / "res/setup/isaaclab_current_headless_gate/isaaclab_current_headless_gate.json"


def run(name: str, cmd: list[str], env: dict[str, str] | None = None, timeout: int = 120) -> dict[str, Any]:
    merged_env = os.environ.copy()
    merged_env["PYTHONNOUSERSITE"] = "1"
    if env:
        merged_env.update(env)
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=merged_env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return {
            "name": name,
            "cmd": cmd,
            "returncode": proc.returncode,
            "ok": proc.returncode == 0,
            "stdout_tail": proc.stdout[-4000:],
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        return {
            "name": name,
            "cmd": cmd,
            "returncode": 124,
            "ok": False,
            "stdout_tail": stdout[-4000:],
            "error": f"timeout after {timeout}s",
        }


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)

    py_analysis = ROOT / "envs/bm_analysis/bin/python"
    py_diffusion = ROOT / "envs/bm_diffusion/bin/python"
    py_tracking = ROOT / "envs/bm_tracking/bin/python"

    probes = [
        run(
            "analysis_imports",
            [
                str(py_analysis),
                "-c",
                "import numpy,pandas,matplotlib,onnxruntime; print('analysis_imports_ok')",
            ],
        ),
        run(
            "diffusion_torch_cuda_visible_devices_5_6",
            [
                str(py_diffusion),
                "-c",
                (
                    "import json, torch; "
                    "print(json.dumps({'torch': torch.__version__, "
                    "'cuda_available': torch.cuda.is_available(), "
                    "'device_count': torch.cuda.device_count(), "
                    "'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))"
                ),
            ],
            env={"CUDA_VISIBLE_DEVICES": "5,6"},
        ),
        run(
            "tracking_basic_imports",
            [
                str(py_tracking),
                "-c",
                "import numpy,torch,onnxruntime,gymnasium; print('tracking_basic_imports_ok')",
            ],
        ),
        run(
            "tracking_isaacsim_isaaclab_imports",
            [
                str(py_tracking),
                "-c",
                (
                    "import json, os, isaacsim, isaaclab; "
                    "print(json.dumps({'isaacsim': isaacsim.__file__, "
                    "'isaaclab': isaaclab.__file__, "
                    "'ISAAC_PATH': os.environ.get('ISAAC_PATH')}))"
                ),
            ],
            env={"OMNI_KIT_ACCEPT_EULA": "YES"},
        ),
        run("tracking_pip_check", [str(py_tracking), "-m", "pip", "check"]),
    ]

    legacy_live_gate = load_json(LEGACY_LIVE_GATE)
    current_headless_gate = load_json(CURRENT_HEADLESS_GATE)
    live_headless_gate_ok = bool(
        legacy_live_gate.get("checks", {}).get("app_launcher_reached_success_sentinel")
        or current_headless_gate.get("checks", {}).get("app_launcher_headless_success_sentinel")
    )
    checks = {
        "analysis_imports_ok": probes[0]["ok"],
        "diffusion_torch_cuda_visible_devices_5_6_ok": probes[1]["ok"],
        "tracking_basic_imports_ok": probes[2]["ok"],
        "isaacsim_import_ok": probes[3]["ok"],
        "isaaclab_import_ok": probes[3]["ok"],
        "tracking_pip_check_ok": probes[4]["ok"],
        "isaaclab_live_headless_gate_ok": live_headless_gate_ok,
        "training_started": False,
    }
    status = "ok_with_runtime_warning" if all(v for k, v in checks.items() if k != "training_started") else "partial_blocked"

    summary: dict[str, Any] = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "conda prefix import probes after IsaacLab/Isaac Sim pip-runtime recovery; no training",
        "checks": checks,
        "probes": probes,
        "live_gate_evidence": {
            "legacy_live_gate_json": str(LEGACY_LIVE_GATE),
            "legacy_live_gate_status": legacy_live_gate.get("status"),
            "legacy_live_gate_success_sentinel": bool(
                legacy_live_gate.get("checks", {}).get("app_launcher_reached_success_sentinel")
            ),
            "current_headless_gate_json": str(CURRENT_HEADLESS_GATE),
            "current_headless_gate_status": current_headless_gate.get("status"),
            "current_headless_gate_success_sentinel": bool(
                current_headless_gate.get("checks", {}).get("app_launcher_headless_success_sentinel")
            ),
            "current_headless_gate_runtime_warning": bool(
                current_headless_gate.get("run", {}).get("markers", {}).get("cuda_p2p_iommu_warning")
            ),
        },
        "interpretation": (
            "bm_tracking imports pip Isaac Sim 4.5 and local editable IsaacLab source packages, and the current "
            "AppLauncher(headless=True) gate reaches a success sentinel. The remaining active tracking blocker is "
            "official G1 conversion/replay, not AppLauncher startup. The gate retains CUDA P2P/IOMMU runtime warnings "
            "and does not prove replay, PPO, DAgger, Fig.5/Fig.6, or real robot behavior."
        ),
        "log": "logs/env_probe/env_import_probe.log",
    }
    LOG.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (OUT / "env_import_probe.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": status, "json": str(OUT / "env_import_probe.json")}))
    if status == "partial_blocked":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
