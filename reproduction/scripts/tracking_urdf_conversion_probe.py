#!/usr/bin/env python3
"""Probe Isaac Sim URDF-to-USD conversion for the official G1 asset only."""

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
OUT = ROOT / "res/tracking/urdf_conversion_probe"
LOG_DIR = ROOT / "logs/tracking_urdf_conversion_probe"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
G1_URDF = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf"
)
TIMEOUT_SECONDS = 180


PROBE_CODE = r"""
import json
import os
import stat
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path("/mnt/infini-data/test/BeyondMimic")
URDF = ROOT / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf"
OUT = ROOT / "res/tracking/urdf_conversion_probe"
USD_DIR = ROOT / "tmp/isaaclab_urdf_probe/g1"
USD_DIR.mkdir(parents=True, exist_ok=True)

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

payload = {"urdf": str(URDF), "usd_dir": str(USD_DIR)}
try:
    import omni.kit.commands
    import omni.usd
    from pxr import Usd
    from isaaclab.sim.converters import UrdfConverterCfg
    from isaaclab.sim.converters.urdf_converter import UrdfConverter
    from isaaclab.sim import converters

    cfg = UrdfConverterCfg(
        asset_path=str(URDF),
        usd_dir=str(USD_DIR),
        usd_file_name="g1_probe.usd",
        force_usd_conversion=True,
        fix_base=False,
        make_instanceable=False,
        replace_cylinders_with_capsules=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    )
    converter = UrdfConverter(cfg)
    usd_path = Path(converter.usd_path)
    payload["converter_usd_path"] = str(usd_path)
    payload["converter_usd_exists"] = usd_path.exists()
    payload["converter_usd_mode"] = oct(stat.S_IMODE(usd_path.stat().st_mode)) if usd_path.exists() else None
    payload["converter_usd_size"] = usd_path.stat().st_size if usd_path.exists() else None

    stage = Usd.Stage.Open(str(usd_path)) if usd_path.exists() else None
    payload["stage_open_ok"] = stage is not None
    if stage is not None:
        default_prim = stage.GetDefaultPrim()
        payload["default_prim_valid"] = bool(default_prim)
        payload["default_prim_path"] = str(default_prim.GetPath()) if default_prim else None
        prim_count = 0
        rigid_body_like = 0
        for prim in stage.Traverse():
            prim_count += 1
            if prim.HasAPI("PhysicsRigidBodyAPI") or prim.GetAttribute("physics:rigidBodyEnabled"):
                rigid_body_like += 1
        payload["prim_count"] = prim_count
        payload["rigid_body_like_count"] = rigid_body_like
        try:
            stage.Save()
            payload["stage_save_ok_after_open"] = True
        except Exception as exc:
            payload["stage_save_ok_after_open"] = False
            payload["stage_save_error"] = repr(exc)
except Exception as exc:
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


def classify(text: str, timed_out: bool) -> dict[str, bool]:
    lowered = text.lower()
    return {
        "timed_out": timed_out,
        "sentinel_after_app": "BM_SENTINEL:after_app" in text,
        "sentinel_payload": "BM_SENTINEL:payload:" in text,
        "sentinel_after_close": "BM_SENTINEL:after_close" in text,
        "usd_save_not_allowed": "saving not allowed" in lowered,
        "libglu_missing": "libglu.so.1" in lowered and "cannot open shared object file" in lowered,
        "p2p_iommu_warning": "p2pbandwidthlatencytest" in lowered
        or "cuda peer-to-peer observed bandwidth" in lowered,
        "traceback": "traceback (most recent call last)" in lowered,
    }


def extract_payload(text: str) -> dict[str, Any]:
    for line in text.splitlines():
        if line.startswith("BM_SENTINEL:payload:"):
            return json.loads(line.split("BM_SENTINEL:payload:", 1)[1])
    return {}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for path in [ROOT / "tmp/isaaclab_urdf_probe/g1", ROOT / "cache/home", ROOT / "cache/omniverse"]:
        path.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(ROOT / "tmp/isaaclab_urdf_probe/g1", ignore_errors=True)
    (ROOT / "tmp/isaaclab_urdf_probe/g1").mkdir(parents=True, exist_ok=True)

    timed_out = False
    try:
        proc = subprocess.run(
            [str(TRACKING_PY), "-c", textwrap.dedent(PROBE_CODE)],
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
        output = exc.stdout if isinstance(exc.stdout, str) else ""

    log_path = LOG_DIR / "tracking_urdf_conversion_probe.log"
    log_path.write_text(output, encoding="utf-8", errors="ignore")
    markers = classify(output, timed_out)
    payload = extract_payload(output)
    checks = {
        "tracking_python_exists": TRACKING_PY.is_file(),
        "g1_urdf_exists": G1_URDF.is_file(),
        "project_egl_icd_exists": PROJECT_EGL_ICD.is_file(),
        "app_launcher_reached_payload": markers["sentinel_after_app"] and markers["sentinel_payload"],
        "app_launcher_closed": markers["sentinel_after_close"],
        "libglu_missing_absent": not markers["libglu_missing"],
        "usd_save_blocker_observed": markers["usd_save_not_allowed"]
        or payload.get("stage_save_ok_after_open") is False,
        "no_probe_exception": "exception" not in payload,
        "does_not_start_replay_or_training": True,
        "does_not_claim_motion_npz": True,
    }
    status = (
        "ok_with_urdf_usd_blocker"
        if checks["app_launcher_reached_payload"] and checks["no_probe_exception"]
        else "failed"
    )
    summary: dict[str, Any] = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_urdf_conversion_probe",
        "scope": "Minimal Isaac Sim URDF-to-USD conversion probe for official G1 asset only; no replay or training.",
        "returncode": returncode,
        "markers": markers,
        "payload": payload,
        "checks": checks,
        "log": str(log_path),
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This probe isolates the official G1 URDF conversion layer. It does not produce motion.npz, replay, "
                "tracking smoke metrics, PPO, or paper-level rollout evidence."
            ),
        },
        "outputs": {"json": str(OUT / "tracking_urdf_conversion_probe.json")},
    }
    (OUT / "tracking_urdf_conversion_probe.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": status, "json": summary["outputs"]["json"]}, sort_keys=True))
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
