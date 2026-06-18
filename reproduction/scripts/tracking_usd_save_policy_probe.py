#!/usr/bin/env python3
"""Probe USD layer save policy inside and outside Isaac Sim AppLauncher."""

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
OUT = ROOT / "res/tracking/usd_save_policy_probe"
LOG_DIR = ROOT / "logs/tracking_usd_save_policy_probe"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
TIMEOUT_SECONDS = 120


PLAIN_PXR_CODE = r"""
import json
from pathlib import Path

payload = {}
try:
    from pxr import Sdf, Usd, UsdGeom

    path = Path("/mnt/infini-data/test/BeyondMimic/tmp/usd_save_policy_probe/plain_python/plain_stage.usda")
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    xform = UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
    stage.SetDefaultPrim(xform.GetPrim())
    layer = stage.GetRootLayer()
    payload["pxr_import_ok"] = True
    payload["permission_to_save_initial"] = bool(layer.permissionToSave)
    try:
        layer.Save()
        payload["save_ok"] = True
    except Exception as exc:
        payload["save_ok"] = False
        payload["save_exception"] = repr(exc)
    payload["path"] = str(path)
    payload["exists"] = path.exists()
    payload["size"] = path.stat().st_size if path.exists() else None
except Exception as exc:
    payload["pxr_import_ok"] = False
    payload["exception"] = repr(exc)
print(json.dumps(payload, sort_keys=True))
"""


KIT_PROBE_CODE = r"""
import json
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE = ROOT / "tmp/usd_save_policy_probe/app_launcher"
TEST_DIRS = {
    "tmp": BASE / "tmp",
    "cache": ROOT / "cache/usd_save_policy_probe",
    "res": ROOT / "res/tracking/usd_save_policy_probe/app_launcher_stage",
}
for path in TEST_DIRS.values():
    path.mkdir(parents=True, exist_ok=True)

print("BM_SENTINEL:before_app", flush=True)
launcher = AppLauncher(
    headless=True,
    enable_cameras=False,
    device="cuda:6",
    fast_shutdown=False,
    multi_gpu=False,
    kit_args="--/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1 --/renderer/activeGpu=6 --/physics/cudaDevice=6",
)
app = launcher.app
print("BM_SENTINEL:after_app", flush=True)

payload = {"test_dirs": {k: str(v) for k, v in TEST_DIRS.items()}}

try:
    from pxr import Sdf, Usd, UsdGeom

    payload["pxr_import_ok_inside_app"] = True

    def try_save(label, path, force_permission=False, use_export=False):
        result = {
            "label": label,
            "path": str(path),
            "force_permission": force_permission,
            "use_export": use_export,
        }
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

    attempts = []
    for dir_label, directory in TEST_DIRS.items():
        for ext in ["usda", "usd"]:
            attempts.append(try_save(f"{dir_label}_{ext}_plain", directory / f"plain.{ext}"))
            attempts.append(try_save(f"{dir_label}_{ext}_force", directory / f"force.{ext}", force_permission=True))
            attempts.append(try_save(f"{dir_label}_{ext}_export", directory / f"export.{ext}", use_export=True))
    payload["attempts"] = attempts
except Exception as exc:
    payload["pxr_import_ok_inside_app"] = False
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


def run_plain_python() -> tuple[int, str]:
    proc = subprocess.run(
        [str(TRACKING_PY), "-c", textwrap.dedent(PLAIN_PXR_CODE)],
        cwd=ROOT,
        env=base_env(),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    return proc.returncode, proc.stdout


def run_app_probe() -> tuple[int, str, bool]:
    timed_out = False
    try:
        proc = subprocess.run(
            [str(TRACKING_PY), "-c", textwrap.dedent(KIT_PROBE_CODE)],
            cwd=ROOT,
            env=base_env(),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=TIMEOUT_SECONDS,
        )
        return proc.returncode, proc.stdout, timed_out
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        if isinstance(exc.stdout, bytes):
            output = exc.stdout.decode("utf-8", errors="ignore")
        elif isinstance(exc.stdout, str):
            output = exc.stdout
        else:
            output = ""
        return 124, output, timed_out


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


def parse_plain_output(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {"parse_error": True, "raw_tail": text[-1000:]}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(ROOT / "tmp/usd_save_policy_probe", ignore_errors=True)
    (ROOT / "tmp/usd_save_policy_probe").mkdir(parents=True, exist_ok=True)

    plain_rc, plain_output = run_plain_python()
    plain_log = LOG_DIR / "plain_python_pxr_probe.log"
    plain_log.write_text(plain_output, encoding="utf-8", errors="ignore")
    plain_payload = parse_plain_output(plain_output)

    app_rc, app_output, timed_out = run_app_probe()
    app_log = LOG_DIR / "app_launcher_usd_save_policy_probe.log"
    app_log.write_text(app_output, encoding="utf-8", errors="ignore")
    markers = classify_output(app_output, timed_out)
    app_payload = extract_payload(app_output)
    attempts = app_payload.get("attempts", [])
    save_ok_count = sum(1 for row in attempts if row.get("save_ok") is True)
    force_save_ok_count = sum(
        1 for row in attempts if row.get("force_permission") is True and row.get("save_ok") is True
    )
    export_ok_count = sum(1 for row in attempts if row.get("use_export") is True and row.get("save_ok") is True)
    permission_false_count = sum(1 for row in attempts if row.get("permission_initial") is False)

    if save_ok_count > 0:
        blocker = "none_usd_save_policy_has_working_path"
    elif app_payload and permission_false_count == len(attempts) and attempts:
        blocker = "app_launcher_layers_permission_to_save_false"
    elif markers["usd_save_not_allowed"] and markers["vulkan_device_lost"]:
        blocker = "app_launcher_usd_save_forbidden_and_vulkan_device_lost"
    elif plain_payload.get("pxr_import_ok") is False and not app_payload:
        blocker = "pxr_only_available_inside_kit_and_app_probe_failed"
    else:
        blocker = "unclassified_usd_save_policy_failure"

    checks = {
        "tracking_python_exists": TRACKING_PY.is_file(),
        "plain_python_pxr_import_ok": plain_payload.get("pxr_import_ok") is True,
        "plain_python_records_absent_pxr": plain_payload.get("pxr_import_ok") is False,
        "app_launcher_reached_after_app": markers["sentinel_after_app"],
        "app_launcher_payload_recorded": markers["sentinel_payload"],
        "app_launcher_closed_or_timeout_recorded": markers["sentinel_after_close"] or markers["timed_out"],
        "pxr_import_ok_inside_app": app_payload.get("pxr_import_ok_inside_app") is True,
        "attempts_recorded": len(attempts) >= 12,
        "permission_false_recorded": permission_false_count > 0,
        "all_attempts_failed_to_save": len(attempts) > 0 and save_ok_count == 0,
        "force_permission_attempts_failed": force_save_ok_count == 0,
        "export_attempts_failed": export_ok_count == 0,
        "usd_save_not_allowed_recorded": markers["usd_save_not_allowed"],
        "vulkan_device_lost_recorded": markers["vulkan_device_lost"],
        "does_not_start_replay_or_training": True,
        "does_not_claim_motion_npz": True,
    }
    status = "ok" if save_ok_count > 0 else "ok_with_blocker_classified" if app_payload else "failed"
    summary: dict[str, Any] = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_usd_save_policy_probe",
        "scope": (
            "Minimal USD save-policy diagnostic for the official replay blocker. It tests plain Python pxr import "
            "and AppLauncher USD layer saving across tmp/cache/res paths, .usd/.usda, Save, Export, and "
            "SetPermissionToSave(True). No replay, training, task smoke, video, or robot execution is performed."
        ),
        "plain_python": {"returncode": plain_rc, "payload": plain_payload, "log": str(plain_log)},
        "app_launcher": {
            "returncode": app_rc,
            "markers": markers,
            "payload": app_payload,
            "log": str(app_log),
            "save_ok_count": save_ok_count,
            "force_save_ok_count": force_save_ok_count,
            "export_ok_count": export_ok_count,
            "permission_false_count": permission_false_count,
        },
        "current_blocker": blocker,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This probe only diagnoses USD layer save policy. It does not generate a valid G1 USD, motion.npz, "
                "official replay, tracking metric, PPO checkpoint, teacher rollout, or paper-level closed-loop result."
            ),
        },
        "outputs": {"json": str(OUT / "tracking_usd_save_policy_probe.json")},
    }
    (OUT / "tracking_usd_save_policy_probe.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": status, "current_blocker": blocker, "json": summary["outputs"]["json"]}, sort_keys=True))
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
