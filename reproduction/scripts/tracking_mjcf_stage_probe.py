#!/usr/bin/env python3
"""Probe USD stage saving and MJCF conversion as a G1 replay bypass path."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import textwrap
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/mjcf_stage_probe"
LOG_DIR = ROOT / "logs/tracking_mjcf_stage_probe"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
G1_ASSET_ROOT = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description"
)
G1_MJCF = G1_ASSET_ROOT / "mjcf/g1.xml"
TIMEOUT_SECONDS = 120


PROBE_CODE = r"""
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE = ROOT / "tmp/isaaclab_mjcf_stage_probe"
G1_MJCF = ROOT / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml"
TINY_MJCF = BASE / "tiny/tiny_box.xml"
for path in [BASE / "stage", BASE / "tiny", BASE / "tiny/usd", BASE / "g1/usd"]:
    path.mkdir(parents=True, exist_ok=True)

TINY_MJCF.write_text(
    '''<mujoco model="tiny_box">
  <worldbody>
    <body name="box" pos="0 0 0.1">
      <freejoint/>
      <geom name="box_geom" type="box" size="0.1 0.1 0.1" density="1000"/>
    </body>
  </worldbody>
</mujoco>
''',
    encoding="utf-8",
)

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

payload = {
    "g1_mjcf": str(G1_MJCF),
    "tiny_mjcf": str(TINY_MJCF),
}

try:
    import omni.kit.app
    import omni.kit.commands
    import omni.usd
    from isaacsim.core.utils.extensions import enable_extension
    from pxr import Sdf, Usd, UsdGeom
    from isaaclab.sim.converters import MjcfConverterCfg
    from isaaclab.sim.converters.mjcf_converter import MjcfConverter

    manager = omni.kit.app.get_app().get_extension_manager()
    payload["mjcf_extension_enabled_before"] = manager.is_extension_enabled("isaacsim.asset.importer.mjcf")
    if not payload["mjcf_extension_enabled_before"]:
        enable_extension("isaacsim.asset.importer.mjcf")
        for _ in range(5):
            app.update()
    payload["mjcf_extension_enabled_after"] = manager.is_extension_enabled("isaacsim.asset.importer.mjcf")

    def inspect_stage(path):
        result = {"usd_path": str(path), "usd_exists": path.exists()}
        result["usd_size"] = path.stat().st_size if path.exists() else None
        stage = Usd.Stage.Open(str(path)) if path.exists() else None
        result["stage_open_ok"] = stage is not None
        if stage is not None:
            default_prim = stage.GetDefaultPrim()
            result["default_prim_valid"] = bool(default_prim)
            result["default_prim_path"] = str(default_prim.GetPath()) if default_prim else None
            prim_paths = []
            prim_count = 0
            for prim in stage.Traverse():
                prim_count += 1
                if len(prim_paths) < 12:
                    prim_paths.append(str(prim.GetPath()))
            result["prim_count"] = prim_count
            result["prim_path_sample"] = prim_paths
        return result

    stage_path = BASE / "stage/minimal_stage.usda"
    try:
        stage = Usd.Stage.CreateNew(str(stage_path))
        xform = UsdGeom.Xform.Define(stage, Sdf.Path("/World"))
        stage.SetDefaultPrim(xform.GetPrim())
        stage.GetRootLayer().Save()
        payload["minimal_stage_save"] = inspect_stage(stage_path)
    except Exception as exc:
        payload["minimal_stage_save"] = {"exception": repr(exc), "usd_path": str(stage_path)}

    def convert_mjcf(label, mjcf_path, usd_dir):
        result = {"label": label, "mjcf_path": str(mjcf_path), "usd_dir": str(usd_dir)}
        try:
            cfg = MjcfConverterCfg(
                asset_path=str(mjcf_path),
                usd_dir=str(usd_dir),
                usd_file_name=f"{label}.usd",
                force_usd_conversion=True,
                fix_base=False,
                make_instanceable=False,
                import_sites=True,
                self_collision=False,
            )
            converter = MjcfConverter(cfg)
            result["converter_usd_path"] = converter.usd_path
            result["stage"] = inspect_stage(Path(converter.usd_path))
        except Exception as exc:
            result["conversion_exception"] = repr(exc)
        return result

    payload["conversions"] = {
        "tiny_mjcf": convert_mjcf("tiny_mjcf", TINY_MJCF, BASE / "tiny/usd"),
        "g1_mjcf": convert_mjcf("g1_mjcf", G1_MJCF, BASE / "g1/usd"),
    }
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


def mjcf_mesh_audit() -> dict[str, Any]:
    root = ET.parse(G1_MJCF).getroot()
    compiler = root.find("compiler")
    meshdir = compiler.attrib.get("meshdir", "") if compiler is not None else ""
    mesh_root = (G1_MJCF.parent / meshdir).resolve()
    refs = [mesh.attrib["file"] for mesh in root.findall(".//mesh") if mesh.attrib.get("file")]
    rows = []
    missing = []
    for ref in refs:
        path = mesh_root / ref
        row = {"ref": ref, "resolved_path": str(path), "exists": path.is_file()}
        rows.append(row)
        if not path.is_file():
            missing.append(row)
    return {
        "mjcf": str(G1_MJCF),
        "meshdir": meshdir,
        "mesh_root": str(mesh_root),
        "mesh_ref_count": len(refs),
        "missing_count": len(missing),
        "all_mesh_refs_resolve": len(missing) == 0,
        "sample": rows[:10],
        "missing": missing,
    }


def classify_output(text: str, timed_out: bool) -> dict[str, bool]:
    lowered = text.lower()
    return {
        "timed_out": timed_out,
        "sentinel_after_app": "BM_SENTINEL:after_app" in text,
        "sentinel_payload": "BM_SENTINEL:payload:" in text,
        "sentinel_after_close": "BM_SENTINEL:after_close" in text,
        "libglu_missing": "libglu.so.1" in lowered and "cannot open shared object file" in lowered,
        "traceback": "traceback (most recent call last)" in lowered,
        "usd_save_not_allowed": "saving not allowed" in lowered,
        "vulkan_device_lost": "error_device_lost" in lowered or "gpu crash is detected" in lowered,
        "p2p_iommu_warning": "p2pbandwidthlatencytest" in lowered
        or "cuda peer-to-peer observed bandwidth" in lowered,
    }


def extract_payload(text: str) -> dict[str, Any]:
    for line in text.splitlines():
        if line.startswith("BM_SENTINEL:payload:"):
            return json.loads(line.split("BM_SENTINEL:payload:", 1)[1])
    return {}


def stage_success(payload: dict[str, Any], key: str) -> bool:
    if key == "minimal_stage_save":
        stage = payload.get(key, {})
    else:
        stage = payload.get("conversions", {}).get(key, {}).get("stage", {})
    return bool(stage.get("stage_open_ok") and stage.get("default_prim_valid") and stage.get("prim_count", 0) > 0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(ROOT / "tmp/isaaclab_mjcf_stage_probe", ignore_errors=True)
    (ROOT / "tmp/isaaclab_mjcf_stage_probe").mkdir(parents=True, exist_ok=True)

    static_mjcf_audit = mjcf_mesh_audit()
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
        if isinstance(exc.stdout, bytes):
            output = exc.stdout.decode("utf-8", errors="ignore")
        elif isinstance(exc.stdout, str):
            output = exc.stdout
        else:
            output = ""

    log_path = LOG_DIR / "tracking_mjcf_stage_probe.log"
    log_path.write_text(output, encoding="utf-8", errors="ignore")
    markers = classify_output(output, timed_out)
    payload = extract_payload(output)

    minimal_stage_ok = stage_success(payload, "minimal_stage_save")
    tiny_mjcf_ok = stage_success(payload, "tiny_mjcf")
    g1_mjcf_ok = stage_success(payload, "g1_mjcf")
    if g1_mjcf_ok:
        blocker = "none_mjcf_g1_usd_conversion_available"
    elif markers["sentinel_after_app"] and markers["usd_save_not_allowed"] and markers["vulkan_device_lost"]:
        blocker = "mjcf_or_stage_usd_save_forbidden_and_vulkan_device_lost"
    elif minimal_stage_ok and tiny_mjcf_ok and not g1_mjcf_ok:
        blocker = "g1_mjcf_specific_conversion_failure"
    elif minimal_stage_ok and not tiny_mjcf_ok:
        blocker = "mjcf_importer_failure_after_basic_stage_save"
    elif not minimal_stage_ok:
        blocker = "basic_usd_stage_save_failure"
    else:
        blocker = "unclassified_mjcf_stage_probe_failure"

    checks = {
        "tracking_python_exists": TRACKING_PY.is_file(),
        "g1_mjcf_exists": G1_MJCF.is_file(),
        "project_egl_icd_exists": PROJECT_EGL_ICD.is_file(),
        "all_g1_mjcf_mesh_refs_resolve_statically": static_mjcf_audit["all_mesh_refs_resolve"],
        "app_launcher_reached_after_app": markers["sentinel_after_app"],
        "app_launcher_payload_or_blocker_recorded": markers["sentinel_payload"] or blocker
        in {"mjcf_or_stage_usd_save_forbidden_and_vulkan_device_lost", "basic_usd_stage_save_failure"},
        "app_launcher_closed_or_timeout_recorded": markers["sentinel_after_close"] or markers["timed_out"],
        "libglu_missing_absent": not markers["libglu_missing"],
        "minimal_stage_save_success": minimal_stage_ok,
        "tiny_mjcf_conversion_success": tiny_mjcf_ok,
        "g1_mjcf_conversion_success": g1_mjcf_ok,
        "does_not_start_replay_or_training": True,
        "does_not_claim_motion_npz": True,
    }
    status = "ok" if g1_mjcf_ok else "ok_with_blocker_classified" if checks["app_launcher_payload_or_blocker_recorded"] else "failed"
    summary: dict[str, Any] = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_mjcf_stage_probe",
        "scope": (
            "Headless Isaac Sim probe for basic USD stage saving and MJCF conversion using a tiny MJCF and the "
            "official G1 MJCF. No csv_to_npz, replay, task smoke, PPO, VAE, diffusion, video, or robot execution "
            "is performed."
        ),
        "returncode": returncode,
        "markers": markers,
        "static_mjcf_audit": static_mjcf_audit,
        "payload": payload,
        "current_blocker": blocker,
        "checks": checks,
        "log": str(log_path),
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This probe only tests whether MJCF or direct USD stage save can bypass the URDF conversion blocker. "
                "It does not create an official motion.npz, replay video, tracking metric, PPO checkpoint, teacher "
                "rollout, or paper-level closed-loop evidence."
            ),
        },
        "outputs": {"json": str(OUT / "tracking_mjcf_stage_probe.json")},
    }
    (OUT / "tracking_mjcf_stage_probe.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": status, "current_blocker": blocker, "json": summary["outputs"]["json"]}, sort_keys=True))
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
