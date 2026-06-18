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

    checks = {
        "analysis_imports_ok": probes[0]["ok"],
        "diffusion_torch_cuda_visible_devices_5_6_ok": probes[1]["ok"],
        "tracking_basic_imports_ok": probes[2]["ok"],
        "isaacsim_import_ok": probes[3]["ok"],
        "isaaclab_import_ok": probes[3]["ok"],
        "tracking_pip_check_ok": probes[4]["ok"],
        "isaaclab_live_headless_gate_ok": False,
        "training_started": False,
    }
    status = "ok_with_live_kit_warning" if all(v for k, v in checks.items() if k not in {"isaaclab_live_headless_gate_ok", "training_started"}) else "partial_blocked"

    summary: dict[str, Any] = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "conda prefix import probes after IsaacLab/Isaac Sim pip-runtime recovery; no training",
        "checks": checks,
        "probes": probes,
        "interpretation": (
            "bm_tracking now imports pip Isaac Sim 4.5 and local editable IsaacLab source packages. "
            "This does not prove live IsaacLab rollout: the separate headless AppLauncher gate is still "
            "blocked by host Kit/Vulkan startup errors recorded under logs/setup/isaaclab_headless_app_gate.log."
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
