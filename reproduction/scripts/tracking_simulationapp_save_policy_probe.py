#!/usr/bin/env python3
"""Compare Isaac Sim SimulationApp and IsaacLab AppLauncher USD save policy."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/simulationapp_save_policy_probe"
LOG_DIR = ROOT / "logs/tracking_simulationapp_save_policy_probe"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
ISAACSIM_BASE_PYTHON_KIT = (
    ROOT / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/apps/isaacsim.exp.base.python.kit"
)
ISAACLAB_HEADLESS_KIT = ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/apps/isaaclab.python.headless.kit"
TIMEOUT_SECONDS = 120


CASE_CODE = r"""
import json
import os
from pathlib import Path

MODE = os.environ["BM_SAVE_POLICY_MODE"]
EXPERIENCE = os.environ.get("BM_SAVE_POLICY_EXPERIENCE", "")
ROOT = Path("/mnt/infini-data/test/BeyondMimic")
CASE_DIR = ROOT / "tmp/simulationapp_save_policy_probe" / MODE
CASE_DIR.mkdir(parents=True, exist_ok=True)

if MODE.startswith("applauncher"):
    from isaaclab.app import AppLauncher

    print("BM_SENTINEL:before_app", flush=True)
    launcher = AppLauncher(
        headless=True,
        enable_cameras=False,
        device="cuda:6",
        fast_shutdown=False,
        multi_gpu=False,
        experience=EXPERIENCE,
        kit_args="--/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1 --/renderer/activeGpu=6 --/physics/cudaDevice=6",
    )
    app = launcher.app
    print("BM_SENTINEL:after_app", flush=True)
else:
    from isaacsim.simulation_app import SimulationApp

    print("BM_SENTINEL:before_app", flush=True)
    config = {
        "headless": True,
        "active_gpu": 6,
        "physics_gpu": 6,
        "multi_gpu": False,
        "fast_shutdown": False,
        "create_new_stage": True,
        "extra_args": [
            "--/renderer/multiGpu/autoEnable=false",
            "--/renderer/multiGpu/maxGpuCount=1",
            "--/renderer/activeGpu=6",
            "--/physics/cudaDevice=6",
        ],
    }
    app = SimulationApp(config, experience=EXPERIENCE)
    print("BM_SENTINEL:after_app", flush=True)

payload = {"mode": MODE, "experience": EXPERIENCE}

try:
    from pxr import Sdf, Usd, UsdGeom

    payload["pxr_import_ok"] = True

    def try_save(label, force_permission=False, use_export=False):
        path = CASE_DIR / f"{label}.usda"
        result = {"label": label, "path": str(path), "force_permission": force_permission, "use_export": use_export}
        try:
            stage = Usd.Stage.CreateNew(str(path))
            xform = UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
            stage.SetDefaultPrim(xform.GetPrim())
            layer = stage.GetRootLayer()
            result["identifier"] = layer.identifier
            result["real_path"] = layer.realPath
            result["permission_initial"] = bool(layer.permissionToSave)
            if force_permission:
                layer.SetPermissionToSave(True)
                result["permission_after_force"] = bool(layer.permissionToSave)
            try:
                if use_export:
                    layer.Export(str(path))
                else:
                    layer.Save()
                result["save_ok"] = True
            except Exception as exc:
                result["save_ok"] = False
                result["save_exception"] = repr(exc)
            result["exists"] = path.exists()
            result["size"] = path.stat().st_size if path.exists() else None
        except Exception as exc:
            result["create_exception"] = repr(exc)
        return result

    payload["attempts"] = [
        try_save("plain"),
        try_save("force_permission", force_permission=True),
        try_save("export", use_export=True),
    ]
except Exception as exc:
    payload["pxr_import_ok"] = False
    payload["exception"] = repr(exc)

print("BM_SENTINEL:payload:" + json.dumps(payload, sort_keys=True), flush=True)
app.close()
print("BM_SENTINEL:after_close", flush=True)
"""


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "ENABLE_CAMERAS": "0",
            "TMPDIR": str(ROOT / "tmp"),
            "HOME": str(ROOT / "cache/home"),
            "XDG_CACHE_HOME": str(ROOT / "cache/xdg"),
            "PIP_CACHE_DIR": str(ROOT / "cache/pip"),
            "OMNI_USER_DIR": str(ROOT / "cache/omniverse/user"),
            "OMNI_LOGS_DIR": str(ROOT / "logs/omniverse"),
            "OMNI_CACHE_DIR": str(ROOT / "cache/omniverse/cache"),
            "OV_USER_DIR": str(ROOT / "cache/omniverse/user"),
            "OV_CACHE_DIR": str(ROOT / "cache/omniverse/cache"),
            "VK_ICD_FILENAMES": str(PROJECT_EGL_ICD),
            "LD_LIBRARY_PATH": f"{GPU_FOUNDATION_DEPS}:{os.environ.get('LD_LIBRARY_PATH', '')}",
        }
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def classify_output(text: str, timed_out: bool) -> dict[str, bool]:
    lowered = text.lower()
    return {
        "timed_out": timed_out,
        "sentinel_after_app": "BM_SENTINEL:after_app" in text,
        "sentinel_payload": "BM_SENTINEL:payload:" in text,
        "sentinel_after_close": "BM_SENTINEL:after_close" in text,
        "usd_save_not_allowed": "saving not allowed" in lowered,
        "vulkan_device_lost": "error_device_lost" in lowered or "gpu crash is detected" in lowered,
        "traceback": "traceback (most recent call last)" in lowered,
        "p2p_iommu_warning": "p2pbandwidthlatencytest" in lowered
        or "cuda peer-to-peer observed bandwidth" in lowered,
    }


def extract_payload(text: str) -> dict[str, Any]:
    for line in text.splitlines():
        if line.startswith("BM_SENTINEL:payload:"):
            return json.loads(line.split("BM_SENTINEL:payload:", 1)[1])
    return {}


def run_case(name: str, experience: Path | str) -> dict[str, Any]:
    env = base_env()
    env["BM_SAVE_POLICY_MODE"] = name
    env["BM_SAVE_POLICY_EXPERIENCE"] = str(experience)
    timed_out = False
    try:
        proc = subprocess.run(
            [str(TRACKING_PY), "-c", textwrap.dedent(CASE_CODE)],
            cwd=ROOT,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=TIMEOUT_SECONDS,
        )
        output = proc.stdout
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        if isinstance(exc.stdout, bytes):
            output = exc.stdout.decode("utf-8", errors="ignore")
        elif isinstance(exc.stdout, str):
            output = exc.stdout
        else:
            output = ""

    log_path = LOG_DIR / f"{name}.log"
    log_path.write_text(output, encoding="utf-8", errors="ignore")
    payload = extract_payload(output)
    attempts = payload.get("attempts", [])
    return {
        "name": name,
        "experience": str(experience),
        "returncode": returncode,
        "markers": classify_output(output, timed_out),
        "payload": payload,
        "log": str(log_path),
        "save_ok_count": sum(1 for row in attempts if row.get("save_ok") is True),
        "permission_false_count": sum(1 for row in attempts if row.get("permission_initial") is False),
        "force_after_false_count": sum(
            1
            for row in attempts
            if row.get("force_permission") is True and row.get("permission_after_force") is False
        ),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(ROOT / "tmp/simulationapp_save_policy_probe", ignore_errors=True)
    (ROOT / "tmp/simulationapp_save_policy_probe").mkdir(parents=True, exist_ok=True)

    cases = [
        run_case("simulationapp_isaacsim_base_python", ISAACSIM_BASE_PYTHON_KIT),
        run_case("simulationapp_isaaclab_headless", ISAACLAB_HEADLESS_KIT),
        run_case("applauncher_isaaclab_headless", ISAACLAB_HEADLESS_KIT),
    ]
    any_save_ok = any(case["save_ok_count"] > 0 for case in cases)
    simapp_base = cases[0]
    simapp_isaaclab = cases[1]
    app_launcher = cases[-1]
    simapp_base_crash_recorded = (
        simapp_base["returncode"] != 0
        and simapp_base["markers"]["vulkan_device_lost"]
        and not simapp_base["markers"]["sentinel_payload"]
    )
    simapp_isaaclab_permission_false = (
        simapp_isaaclab["markers"]["sentinel_payload"]
        and simapp_isaaclab["save_ok_count"] == 0
        and simapp_isaaclab["permission_false_count"] > 0
    )
    applauncher_permission_false = (
        app_launcher["markers"]["sentinel_payload"]
        and app_launcher["save_ok_count"] == 0
        and app_launcher["permission_false_count"] > 0
    )
    if any_save_ok:
        blocker = "none_simulationapp_or_applauncher_has_usd_save_path"
    elif simapp_base_crash_recorded and simapp_isaaclab_permission_false and applauncher_permission_false:
        blocker = "isaaclab_headless_experience_layers_permission_to_save_false_with_isaacsim_base_vulkan_crash"
    elif simapp_isaaclab_permission_false and applauncher_permission_false:
        blocker = "isaaclab_headless_experience_layers_permission_to_save_false"
    elif applauncher_permission_false:
        blocker = "applauncher_layers_permission_to_save_false"
    else:
        blocker = "unclassified_simulationapp_save_policy_failure"

    checks = {
        "tracking_python_exists": TRACKING_PY.is_file(),
        "isaacsim_base_python_kit_exists": ISAACSIM_BASE_PYTHON_KIT.is_file(),
        "isaaclab_headless_kit_exists": ISAACLAB_HEADLESS_KIT.is_file(),
        "simulationapp_base_reached_payload": simapp_base["markers"]["sentinel_payload"],
        "simulationapp_base_vulkan_device_lost_recorded": simapp_base["markers"]["vulkan_device_lost"],
        "simulationapp_base_crash_recorded": simapp_base_crash_recorded,
        "simulationapp_base_save_failed": simapp_base["save_ok_count"] == 0,
        "simulationapp_base_permission_false": simapp_base["permission_false_count"] > 0,
        "simulationapp_base_force_permission_failed": simapp_base["force_after_false_count"] > 0,
        "simulationapp_isaaclab_headless_reached_payload": simapp_isaaclab["markers"]["sentinel_payload"],
        "simulationapp_isaaclab_headless_save_failed": simapp_isaaclab["save_ok_count"] == 0,
        "simulationapp_isaaclab_headless_permission_false": simapp_isaaclab["permission_false_count"] > 0,
        "simulationapp_isaaclab_headless_force_permission_failed": simapp_isaaclab[
            "force_after_false_count"
        ]
        > 0,
        "applauncher_reached_payload": app_launcher["markers"]["sentinel_payload"],
        "applauncher_save_failed": app_launcher["save_ok_count"] == 0,
        "applauncher_permission_false": app_launcher["permission_false_count"] > 0,
        "applauncher_force_permission_failed": app_launcher["force_after_false_count"] > 0,
        "any_usd_save_path_found": any_save_ok,
        "does_not_start_replay_or_training": True,
        "does_not_claim_motion_npz": True,
    }
    status = (
        "ok"
        if any_save_ok
        else "ok_with_blocker_classified"
        if simapp_isaaclab_permission_false and applauncher_permission_false
        else "failed"
    )
    summary: dict[str, Any] = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_simulationapp_save_policy_probe",
        "scope": (
            "Compare local USD save policy between raw Isaac Sim SimulationApp and IsaacLab AppLauncher. "
            "No replay, training, task smoke, video, checkpoint, or robot execution is performed."
        ),
        "current_blocker": blocker,
        "cases": cases,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This probe only compares Kit launchers and USD layer save policy. It does not generate a valid "
                "G1 USD, motion.npz, official replay, PPO checkpoint, teacher rollout, or paper-level result."
            ),
        },
        "outputs": {"json": str(OUT / "tracking_simulationapp_save_policy_probe.json")},
    }
    (OUT / "tracking_simulationapp_save_policy_probe.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": status, "current_blocker": blocker, "json": summary["outputs"]["json"]}, sort_keys=True))
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
