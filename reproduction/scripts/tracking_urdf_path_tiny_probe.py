#!/usr/bin/env python3
"""Contrast tiny URDF conversion with official G1 package/path variants."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/urdf_path_tiny_probe"
LOG_DIR = ROOT / "logs/tracking_urdf_path_tiny_probe"
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
G1_URDF = G1_ASSET_ROOT / "urdf/g1/main.urdf"
TIMEOUT_SECONDS = 120


PROBE_CODE = r"""
import json
import os
import shutil
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE = ROOT / "tmp/isaaclab_urdf_path_tiny_probe"
G1_ASSET_ROOT = ROOT / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description"
G1_URDF = G1_ASSET_ROOT / "urdf/g1/main.urdf"
TINY_URDF = BASE / "tiny/tiny_box.urdf"
G1_ABS_URDF = BASE / "g1_abs/main_abs_meshes.urdf"

for path in [BASE / "tiny", BASE / "g1_original", BASE / "g1_abs"]:
    path.mkdir(parents=True, exist_ok=True)

TINY_URDF.write_text(
    '''<?xml version="1.0"?>
<robot name=\"tiny_box\">
  <link name=\"base\">
    <inertial>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <mass value=\"1.0\"/>
      <inertia ixx=\"0.01\" ixy=\"0\" ixz=\"0\" iyy=\"0.01\" iyz=\"0\" izz=\"0.01\"/>
    </inertial>
    <visual>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <geometry><box size=\"0.2 0.2 0.2\"/></geometry>
    </visual>
    <collision>
      <origin xyz=\"0 0 0\" rpy=\"0 0 0\"/>
      <geometry><box size=\"0.2 0.2 0.2\"/></geometry>
    </collision>
  </link>
</robot>
''',
    encoding="utf-8",
)

G1_ABS_URDF.write_text(
    G1_URDF.read_text(encoding="utf-8").replace(
        "package://unitree_description/", str(G1_ASSET_ROOT).rstrip("/") + "/"
    ),
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
    "tiny_urdf": str(TINY_URDF),
    "g1_original_urdf": str(G1_URDF),
    "g1_abs_urdf": str(G1_ABS_URDF),
    "ros_package_path": os.environ.get("ROS_PACKAGE_PATH"),
}

try:
    import omni.kit.app
    import omni.kit.commands
    from isaacsim.core.utils.extensions import enable_extension
    from pxr import Usd
    from isaaclab.sim.converters import UrdfConverterCfg
    from isaaclab.sim.converters.urdf_converter import UrdfConverter

    manager = omni.kit.app.get_app().get_extension_manager()
    payload["urdf_extension_enabled_before"] = manager.is_extension_enabled("isaacsim.asset.importer.urdf")
    if not payload["urdf_extension_enabled_before"]:
        enable_extension("isaacsim.asset.importer.urdf")
        for _ in range(5):
            app.update()
    payload["urdf_extension_enabled_after"] = manager.is_extension_enabled("isaacsim.asset.importer.urdf")

    ok, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    payload["import_config_create_ok"] = bool(ok)
    payload["import_config_is_none"] = import_config is None
    payload["import_config_path_methods"] = sorted(
        name
        for name in dir(import_config)
        if any(token in name.lower() for token in ["path", "package", "root", "search", "mesh"])
    )
    payload["import_config_selected_setters"] = sorted(
        name
        for name in dir(import_config)
        if name.startswith("set_")
        and any(token in name.lower() for token in ["default", "physics", "mesh", "joint", "base", "collision"])
    )

    def summarize_robot(robot):
        out = {"type": type(robot).__name__}
        for attr in ["name", "root_link"]:
            try:
                out[attr] = str(getattr(robot, attr))
            except Exception:
                pass
        for attr in ["links", "joints", "materials"]:
            try:
                value = getattr(robot, attr)
                out[f"{attr}_count"] = len(value)
                out[f"{attr}_sample"] = list(value.keys())[:8] if hasattr(value, "keys") else None
            except Exception as exc:
                out[f"{attr}_error"] = repr(exc)
        return out

    def inspect_stage(path):
        result = {"usd_path": str(path), "usd_exists": path.exists()}
        result["usd_size"] = path.stat().st_size if path.exists() else None
        stage = Usd.Stage.Open(str(path)) if path.exists() else None
        result["stage_open_ok"] = stage is not None
        if stage is not None:
            default_prim = stage.GetDefaultPrim()
            result["default_prim_valid"] = bool(default_prim)
            result["default_prim_path"] = str(default_prim.GetPath()) if default_prim else None
            prim_count = 0
            prim_paths = []
            rigid_body_like = 0
            for prim in stage.Traverse():
                prim_count += 1
                if len(prim_paths) < 12:
                    prim_paths.append(str(prim.GetPath()))
                if prim.HasAPI("PhysicsRigidBodyAPI") or prim.GetAttribute("physics:rigidBodyEnabled"):
                    rigid_body_like += 1
            result["prim_count"] = prim_count
            result["prim_path_sample"] = prim_paths
            result["rigid_body_like_count"] = rigid_body_like
        return result

    def parse(path):
        ok_parse, robot = omni.kit.commands.execute("URDFParseFile", urdf_path=str(path), import_config=import_config)
        return {"parse_ok": bool(ok_parse), "robot": summarize_robot(robot) if ok_parse else None}

    def convert(label, urdf_path, usd_dir, fix_base):
        result = {"label": label, "urdf_path": str(urdf_path), "usd_dir": str(usd_dir)}
        try:
            result["parse"] = parse(urdf_path)
        except Exception as exc:
            result["parse_exception"] = repr(exc)
        try:
            cfg = UrdfConverterCfg(
                asset_path=str(urdf_path),
                usd_dir=str(usd_dir),
                usd_file_name=f"{label}.usd",
                force_usd_conversion=True,
                fix_base=fix_base,
                make_instanceable=False,
                replace_cylinders_with_capsules=True,
                joint_drive=None,
            )
            converter = UrdfConverter(cfg)
            result["converter_usd_path"] = converter.usd_path
            result["stage"] = inspect_stage(Path(converter.usd_path))
        except Exception as exc:
            result["conversion_exception"] = repr(exc)
        return result

    payload["conversions"] = {
        "tiny_box": convert("tiny_box", TINY_URDF, BASE / "tiny/usd", True),
        "g1_original": convert("g1_original", G1_URDF, BASE / "g1_original/usd", False),
        "g1_abs_meshes": convert("g1_abs_meshes", G1_ABS_URDF, BASE / "g1_abs/usd", False),
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
            "ROS_PACKAGE_PATH": str(G1_ASSET_ROOT.parent),
        }
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def mesh_reference_audit() -> dict[str, Any]:
    text = G1_URDF.read_text(encoding="utf-8")
    refs = re.findall(r'filename="([^"]+)"', text)
    package_refs = [ref for ref in refs if ref.startswith("package://unitree_description/")]
    resolved = []
    missing = []
    for ref in package_refs:
        suffix = ref.removeprefix("package://unitree_description/")
        path = G1_ASSET_ROOT / suffix
        row = {"ref": ref, "resolved_path": str(path), "exists": path.is_file()}
        resolved.append(row)
        if not path.is_file():
            missing.append(row)
    return {
        "urdf": str(G1_URDF),
        "mesh_ref_count": len(refs),
        "package_ref_count": len(package_refs),
        "missing_count": len(missing),
        "all_package_refs_resolve": len(missing) == 0,
        "sample": resolved[:10],
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
        "segmentation_fault": "segmentation fault" in lowered,
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


def conversion_success(payload: dict[str, Any], name: str) -> bool:
    stage = payload.get("conversions", {}).get(name, {}).get("stage", {})
    return bool(stage.get("stage_open_ok") and stage.get("default_prim_valid") and stage.get("prim_count", 0) > 0)


def conversion_empty(payload: dict[str, Any], name: str) -> bool:
    stage = payload.get("conversions", {}).get(name, {}).get("stage", {})
    return bool(stage.get("stage_open_ok") and stage.get("prim_count") == 0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(ROOT / "tmp/isaaclab_urdf_path_tiny_probe", ignore_errors=True)
    (ROOT / "tmp/isaaclab_urdf_path_tiny_probe").mkdir(parents=True, exist_ok=True)

    static_mesh_audit = mesh_reference_audit()
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

    log_path = LOG_DIR / "tracking_urdf_path_tiny_probe.log"
    log_path.write_text(output, encoding="utf-8", errors="ignore")
    markers = classify_output(output, timed_out)
    payload = extract_payload(output)

    tiny_success = conversion_success(payload, "tiny_box")
    g1_original_success = conversion_success(payload, "g1_original")
    g1_abs_success = conversion_success(payload, "g1_abs_meshes")
    g1_original_empty = conversion_empty(payload, "g1_original")
    g1_abs_empty = conversion_empty(payload, "g1_abs_meshes")
    if markers["sentinel_after_app"] and markers["usd_save_not_allowed"] and markers["vulkan_device_lost"]:
        blocker = "usd_layer_save_forbidden_and_vulkan_device_lost_before_payload"
    elif markers["timed_out"] and markers["sentinel_payload"] and not markers["sentinel_after_close"]:
        blocker = "kit_shutdown_timeout_after_urdf_payload"
    elif markers["timed_out"] and not markers["sentinel_payload"]:
        blocker = "kit_timeout_before_urdf_payload"
    elif g1_original_success:
        blocker = "none_for_g1_urdf_conversion"
    elif tiny_success and g1_abs_success and g1_original_empty:
        blocker = "package_uri_resolution_for_official_g1_urdf"
    elif tiny_success and g1_original_empty and g1_abs_empty:
        blocker = "g1_specific_urdf_importer_empty_usd"
    elif not tiny_success and (g1_original_empty or g1_abs_empty):
        blocker = "urdf_importer_global_empty_usd_or_headless_importer_failure"
    else:
        blocker = "unclassified_urdf_conversion_failure"

    checks = {
        "tracking_python_exists": TRACKING_PY.is_file(),
        "g1_urdf_exists": G1_URDF.is_file(),
        "project_egl_icd_exists": PROJECT_EGL_ICD.is_file(),
        "all_g1_package_mesh_refs_resolve_statically": static_mesh_audit["all_package_refs_resolve"],
        "app_launcher_reached_payload": markers["sentinel_after_app"] and markers["sentinel_payload"],
        "app_launcher_closed": markers["sentinel_after_close"],
        "libglu_missing_absent": not markers["libglu_missing"],
        "tiny_urdf_conversion_success": tiny_success,
        "g1_original_conversion_success": g1_original_success,
        "g1_abs_mesh_conversion_success": g1_abs_success,
        "g1_original_empty_usd_recorded": g1_original_empty,
        "g1_abs_empty_usd_recorded": g1_abs_empty,
        "does_not_start_replay_or_training": True,
        "does_not_claim_motion_npz": True,
    }
    status = (
        "ok_with_blocker_classified"
        if checks["app_launcher_reached_payload"]
        or blocker
        in {
            "kit_shutdown_timeout_after_urdf_payload",
            "usd_layer_save_forbidden_and_vulkan_device_lost_before_payload",
        }
        else "failed"
    )
    summary: dict[str, Any] = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_urdf_path_tiny_probe",
        "scope": (
            "Contrasts Isaac Sim URDF conversion for a project-local tiny URDF, the official G1 URDF with "
            "package:// mesh paths, and a generated G1 URDF with absolute mesh paths. No replay, task smoke, "
            "PPO, VAE, diffusion, video, or robot execution is performed."
        ),
        "returncode": returncode,
        "markers": markers,
        "static_mesh_audit": static_mesh_audit,
        "payload": payload,
        "current_blocker": blocker,
        "checks": checks,
        "log": str(log_path),
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This is an importer/path diagnostic gate only. It does not generate a valid official motion.npz, "
                "rendered replay, tracking task metric, PPO checkpoint, teacher rollout dataset, or paper-level "
                "closed-loop BeyondMimic evidence."
            ),
        },
        "outputs": {"json": str(OUT / "tracking_urdf_path_tiny_probe.json")},
    }
    (OUT / "tracking_urdf_path_tiny_probe.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": status, "current_blocker": blocker, "json": summary["outputs"]["json"]}, sort_keys=True))
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
