#!/usr/bin/env python3
"""Probe USD write API variants inside IsaacLab headless Kit."""

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
OUT = ROOT / "res/tracking/usd_api_variant_probe"
LOG_DIR = ROOT / "logs/tracking_usd_api_variant_probe"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
ISAACLAB_HEADLESS_KIT = ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/apps/isaaclab.python.headless.kit"
TIMEOUT_SECONDS = 160


CASE_CODE = r"""
import asyncio
import json
import os
from pathlib import Path

ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT_DIR = ROOT / "tmp/usd_api_variant_probe"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXPERIENCE = os.environ["BM_USD_API_EXPERIENCE"]

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

payload = {"experience": EXPERIENCE, "attempts": []}

from pxr import Sdf, Usd, UsdGeom

payload["pxr_import_ok"] = True


def record(label, path, func):
    result = {"label": label, "path": str(path)}
    try:
        value = func(path, result)
        result["api_return"] = repr(value)
    except Exception as exc:
        result["exception"] = repr(exc)
    result["exists"] = path.exists()
    result["size"] = path.stat().st_size if path.exists() else None
    payload["attempts"].append(result)


def add_world(stage):
    xform = UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
    stage.SetDefaultPrim(xform.GetPrim())


def create_new_save(path, result):
    stage = Usd.Stage.CreateNew(str(path))
    add_world(stage)
    layer = stage.GetRootLayer()
    result["root_identifier"] = layer.identifier
    result["permission_initial"] = bool(layer.permissionToSave)
    return layer.Save()


def create_new_stage_export(path, result):
    stage = Usd.Stage.CreateNew(str(path))
    add_world(stage)
    layer = stage.GetRootLayer()
    result["root_identifier"] = layer.identifier
    result["permission_initial"] = bool(layer.permissionToSave)
    return stage.Export(str(path))


def create_in_memory_stage_export(path, result):
    stage = Usd.Stage.CreateInMemory()
    add_world(stage)
    layer = stage.GetRootLayer()
    result["root_identifier"] = layer.identifier
    result["anonymous"] = bool(layer.anonymous)
    result["permission_initial"] = bool(layer.permissionToSave)
    return stage.Export(str(path))


def sdf_layer_create_new_export(path, result):
    layer = Sdf.Layer.CreateNew(str(path))
    result["root_identifier"] = layer.identifier
    result["permission_initial"] = bool(layer.permissionToSave)
    layer.ImportFromString("#usda 1.0\n\ndef Xform \"World\" {\n}\n")
    return layer.Export(str(path))


def sdf_layer_create_anonymous_export(path, result):
    layer = Sdf.Layer.CreateAnonymous("bm_api_variant.usda")
    result["root_identifier"] = layer.identifier
    result["anonymous"] = bool(layer.anonymous)
    result["permission_initial"] = bool(layer.permissionToSave)
    layer.ImportFromString("#usda 1.0\n\ndef Xform \"World\" {\n}\n")
    return layer.Export(str(path))


def omni_usd_save_as(path, result):
    import omni.usd

    context = omni.usd.get_context()

    async def run():
        await context.new_stage_async()
        stage = context.get_stage()
        add_world(stage)
        root = stage.GetRootLayer()
        result["root_identifier"] = root.identifier
        result["anonymous"] = bool(root.anonymous)
        result["permission_initial"] = bool(root.permissionToSave)
        return await context.save_as_stage_async(str(path))

    return asyncio.get_event_loop().run_until_complete(run())


def omni_usd_export_as(path, result):
    import omni.usd

    context = omni.usd.get_context()

    async def run():
        await context.new_stage_async()
        stage = context.get_stage()
        add_world(stage)
        root = stage.GetRootLayer()
        result["root_identifier"] = root.identifier
        result["anonymous"] = bool(root.anonymous)
        result["permission_initial"] = bool(root.permissionToSave)
        return await context.export_as_stage_async(str(path))

    return asyncio.get_event_loop().run_until_complete(run())


for label, func in [
    ("create_new_save", create_new_save),
    ("create_new_stage_export", create_new_stage_export),
    ("create_in_memory_stage_export", create_in_memory_stage_export),
    ("sdf_layer_create_new_export", sdf_layer_create_new_export),
    ("sdf_layer_create_anonymous_export", sdf_layer_create_anonymous_export),
    # The omni.usd context save_as/export_as calls can hang in this host's
    # headless session, so this probe keeps to direct PXR/Sdf APIs.
]:
    record(label, OUT_DIR / f"{label}.usda", func)

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
            "BM_USD_API_EXPERIENCE": str(ISAACLAB_HEADLESS_KIT),
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
    }


def extract_payload(text: str) -> dict[str, Any]:
    for line in text.splitlines():
        if line.startswith("BM_SENTINEL:payload:"):
            return json.loads(line.split("BM_SENTINEL:payload:", 1)[1])
    return {}


def run_probe() -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timed_out = False
    try:
        proc = subprocess.run(
            [str(TRACKING_PY), "-c", textwrap.dedent(CASE_CODE)],
            cwd=ROOT,
            env=base_env(),
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
    log_path = LOG_DIR / "tracking_usd_api_variant_probe.log"
    log_path.write_text(output, encoding="utf-8", errors="ignore")
    return {
        "returncode": returncode,
        "log": str(log_path),
        "markers": classify_output(output, timed_out),
        "payload": extract_payload(output),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(ROOT / "tmp/usd_api_variant_probe", ignore_errors=True)
    (ROOT / "tmp/usd_api_variant_probe").mkdir(parents=True, exist_ok=True)
    probe = run_probe()
    attempts = probe.get("payload", {}).get("attempts", [])
    successful = [row for row in attempts if row.get("exists") and row.get("size", 0) > 20 and not row.get("exception")]
    in_memory_success = any(row.get("label") == "create_in_memory_stage_export" for row in successful)
    stage_export_success = any(
        row.get("label") in {"create_new_stage_export", "create_in_memory_stage_export"} for row in successful
    )
    create_new_blocked = any(
        row.get("label") in {"create_new_save", "create_new_stage_export"}
        and row.get("permission_initial") is False
        and row.get("exception")
        for row in attempts
    )
    if in_memory_success and stage_export_success:
        status = "ok_with_stage_export_workaround"
        blocker = "layer_save_blocked_but_stage_export_succeeds"
    elif successful:
        status = "ok_with_alternate_usd_write_workaround"
        blocker = "create_new_layers_permission_to_save_false_but_alternate_usd_api_succeeds"
    elif probe["markers"]["sentinel_payload"] and create_new_blocked:
        status = "ok_with_blocker_confirmed"
        blocker = "all_tested_usd_write_api_variants_blocked_or_empty"
    else:
        status = "failed"
        blocker = "usd_api_variant_probe_failed_before_classification"

    checks = {
        "tracking_python_exists": TRACKING_PY.is_file(),
        "isaaclab_headless_kit_exists": ISAACLAB_HEADLESS_KIT.is_file(),
        "app_reached_after_app": probe["markers"]["sentinel_after_app"],
        "payload_recorded": probe["markers"]["sentinel_payload"],
        "pxr_import_ok": probe.get("payload", {}).get("pxr_import_ok") is True,
        "attempts_recorded": len(attempts) >= 5,
        "create_new_blocked_by_permission_false": create_new_blocked,
        "stage_export_success": stage_export_success,
        "in_memory_stage_export_success": in_memory_success,
        "omni_usd_context_save_or_export_not_tested_to_avoid_known_hang": True,
        "any_nonempty_usd_written": bool(successful),
        "does_not_start_replay_or_training": True,
        "does_not_claim_motion_npz": True,
    }
    summary: dict[str, Any] = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_usd_api_variant_probe",
        "scope": (
            "Bounded USD write API variant probe inside IsaacLab headless Kit. No official replay, PPO training, "
            "policy evaluation, DAgger rollout, VAE/diffusion run, video, checkpoint, or robot execution is performed."
        ),
        "current_blocker": blocker,
        "probe": probe,
        "successful_attempt_labels": [row["label"] for row in successful],
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This probe only identifies whether a basic USD write workaround exists inside the current Kit "
                "session. Even a successful USD write path is not yet a valid G1 conversion, motion.npz, replay, "
                "tracking task smoke, PPO result, or paper-level reproduction."
            ),
        },
        "outputs": {"json": str(OUT / "tracking_usd_api_variant_probe.json")},
    }
    (OUT / "tracking_usd_api_variant_probe.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "current_blocker": blocker,
                "successful_attempt_labels": summary["successful_attempt_labels"],
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
