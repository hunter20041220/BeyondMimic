#!/usr/bin/env python3
"""Probe G1 URDF in-memory import under raw SimulationApp with IsaacLab headless experience."""

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
OUT = ROOT / "res/tracking/g1_urdf_simulationapp_in_memory_import"
LOG_DIR = ROOT / "logs/tracking_g1_urdf_simulationapp_in_memory_import"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
ISAACLAB_HEADLESS_KIT = ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/apps/isaaclab.python.headless.kit"
G1_URDF = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/"
    "unitree_description/urdf/g1/main.urdf"
)
TIMEOUT_SECONDS = 180


PROBE_CODE = r"""
import json
import os
from pathlib import Path

ROOT = Path("/mnt/infini-data/test/BeyondMimic")
EXPERIENCE = ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/apps/isaaclab.python.headless.kit"
G1_URDF = ROOT / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf"
BASE = ROOT / "tmp/g1_urdf_simulationapp_in_memory_import"
BASE.mkdir(parents=True, exist_ok=True)
EXPORT = BASE / "g1_current_stage_export.usda"

print("BM_SENTINEL:before_app", flush=True)
from isaacsim.simulation_app import SimulationApp

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
app = SimulationApp(config, experience=str(EXPERIENCE))
print("BM_SENTINEL:after_app", flush=True)

payload = {
    "runtime": "isaacsim.simulation_app.SimulationApp",
    "experience": str(EXPERIENCE),
    "config": config,
    "g1_urdf": str(G1_URDF),
    "export": str(EXPORT),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}

try:
    import omni.kit.app
    import omni.kit.commands
    import omni.usd
    from isaacsim.core.utils.extensions import enable_extension
    from pxr import Usd, UsdGeom, UsdPhysics

    def inspect_stage(path_or_stage, label):
        result = {"label": label}
        if isinstance(path_or_stage, (str, Path)):
            path = Path(path_or_stage)
            result.update(
                {
                    "usd_path": str(path),
                    "usd_exists": path.exists(),
                    "usd_size": path.stat().st_size if path.exists() else None,
                }
            )
            stage = Usd.Stage.Open(str(path)) if path.exists() else None
        else:
            stage = path_or_stage
            result.update({"usd_path": None, "usd_exists": None, "usd_size": None})
        result["stage_open_ok"] = stage is not None
        if stage is not None:
            default_prim = stage.GetDefaultPrim()
            result["default_prim_valid"] = bool(default_prim)
            result["default_prim_path"] = str(default_prim.GetPath()) if default_prim else None
            prim_count = 0
            joint_count = 0
            rigid_body_like_count = 0
            articulation_api_count = 0
            g1_path_present = False
            pelvis_path_present = False
            prim_sample = []
            for prim in stage.Traverse():
                prim_count += 1
                path_s = str(prim.GetPath())
                if len(prim_sample) < 80:
                    prim_sample.append({"path": path_s, "type": prim.GetTypeName()})
                if "Joint" in prim.GetTypeName():
                    joint_count += 1
                if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.GetAttribute("physics:rigidBodyEnabled"):
                    rigid_body_like_count += 1
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    articulation_api_count += 1
                if path_s == "/g1":
                    g1_path_present = True
                if path_s == "/g1/pelvis":
                    pelvis_path_present = True
            result["prim_count"] = prim_count
            result["joint_count"] = joint_count
            result["rigid_body_like_count"] = rigid_body_like_count
            result["articulation_api_count"] = articulation_api_count
            result["g1_path_present"] = g1_path_present
            result["pelvis_path_present"] = pelvis_path_present
            result["prim_sample"] = prim_sample
        return result

    manager = omni.kit.app.get_app().get_extension_manager()
    payload["urdf_extension_enabled_before"] = manager.is_extension_enabled("isaacsim.asset.importer.urdf")
    if not payload["urdf_extension_enabled_before"]:
        enable_extension("isaacsim.asset.importer.urdf")
        for _ in range(5):
            app.update()
    payload["urdf_extension_enabled_after"] = manager.is_extension_enabled("isaacsim.asset.importer.urdf")

    context = omni.usd.get_context()
    stage = context.get_stage()
    payload["explicit_new_stage_called"] = False
    payload["stage_before_import"] = inspect_stage(stage, "current_stage_before_import")
    if stage is not None:
        try:
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            payload["set_stage_up_axis_ok"] = True
        except Exception as exc:
            payload["set_stage_up_axis_exception"] = repr(exc)

    ok, import_config = omni.kit.commands.execute("URDFCreateImportConfig")
    payload["create_import_config_ok"] = bool(ok)
    import_config.set_distance_scale(1.0)
    import_config.set_make_default_prim(True)
    import_config.set_create_physics_scene(False)
    import_config.set_density(0.0)
    import_config.set_convex_decomp(False)
    import_config.set_collision_from_visuals(False)
    import_config.set_merge_fixed_joints(False)
    import_config.set_fix_base(False)
    import_config.set_self_collision(False)
    import_config.set_parse_mimic(True)
    import_config.set_replace_cylinders_with_capsules(True)

    try:
        result = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path=str(G1_URDF),
            import_config=import_config,
            dest_path="",
            get_articulation_root=True,
        )
        payload["parse_and_import_result"] = repr(result)
    except Exception as exc:
        payload["parse_and_import_exception"] = repr(exc)

    for _ in range(10):
        app.update()

    current_stage = context.get_stage()
    payload["current_stage_after_import"] = inspect_stage(current_stage, "current_stage_after_import")
    try:
        payload["current_stage_export_return"] = bool(current_stage.Export(str(EXPORT)))
    except Exception as exc:
        payload["current_stage_export_exception"] = repr(exc)
    payload["exported_stage"] = inspect_stage(EXPORT, "exported_current_stage")

    config_layers = []
    for path in sorted((BASE / "configuration").glob("*")):
        if path.is_file():
            config_layers.append({"path": str(path), "size": path.stat().st_size})
    payload["configuration_files"] = config_layers
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


def classify_output(text: str, timed_out: bool) -> dict[str, bool]:
    lowered = text.lower()
    return {
        "timed_out": timed_out,
        "sentinel_after_app": "BM_SENTINEL:after_app" in text,
        "sentinel_payload": "BM_SENTINEL:payload:" in text,
        "sentinel_after_close": "BM_SENTINEL:after_close" in text,
        "usd_save_not_allowed": "saving not allowed" in lowered,
        "vulkan_device_lost": "error_device_lost" in lowered or "gpu crash is detected" in lowered,
        "in_memory_stage_branch": "creating asset in an in-memory stage" in lowered,
        "traceback": "traceback (most recent call last)" in lowered,
        "p2p_iommu_warning": "p2pbandwidthlatencytest" in lowered
        or "cuda peer-to-peer observed bandwidth" in lowered,
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
    log_path = LOG_DIR / "tracking_g1_urdf_simulationapp_in_memory_import_probe.log"
    log_path.write_text(output, encoding="utf-8", errors="ignore")
    return {
        "returncode": returncode,
        "log": str(log_path),
        "markers": classify_output(output, timed_out),
        "payload": extract_payload(output),
    }


def stage_has_robot(stage_summary: dict[str, Any]) -> bool:
    return bool(
        stage_summary.get("stage_open_ok")
        and stage_summary.get("prim_count", 0) > 10
        and (
            stage_summary.get("joint_count", 0) > 0
            or stage_summary.get("rigid_body_like_count", 0) > 0
            or stage_summary.get("g1_path_present")
        )
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(ROOT / "tmp/g1_urdf_simulationapp_in_memory_import", ignore_errors=True)
    (ROOT / "tmp/g1_urdf_simulationapp_in_memory_import").mkdir(parents=True, exist_ok=True)
    probe = run_probe()
    payload = probe.get("payload", {})
    current_robot_ok = stage_has_robot(payload.get("current_stage_after_import", {}))
    exported_robot_ok = stage_has_robot(payload.get("exported_stage", {}))
    parse_ok = str(payload.get("parse_and_import_result", "")).startswith("(True,")

    if exported_robot_ok:
        status = "ok_with_valid_exported_current_stage_g1_usd"
        blocker = "none_simulationapp_in_memory_import_generated_valid_g1_usd"
    elif current_robot_ok:
        status = "ok_with_current_stage_robot_but_export_invalid"
        blocker = "simulationapp_in_memory_import_reaches_current_stage_but_export_invalid"
    elif probe["markers"]["sentinel_payload"] and parse_ok:
        status = "ok_with_parse_success_but_current_stage_empty"
        blocker = "simulationapp_in_memory_import_parse_success_but_current_stage_empty"
    elif probe["markers"]["sentinel_after_app"] and probe["markers"]["vulkan_device_lost"] and not probe["markers"]["sentinel_payload"]:
        status = "ok_with_vulkan_device_lost_before_payload"
        blocker = "simulationapp_in_memory_import_vulkan_device_lost_before_payload"
    elif probe["markers"]["sentinel_payload"]:
        status = "ok_with_simulationapp_in_memory_import_failed"
        blocker = "simulationapp_in_memory_import_failed_before_robot_stage"
    else:
        status = "failed"
        blocker = "simulationapp_g1_in_memory_import_probe_failed_before_classification"

    checks = {
        "tracking_python_exists": TRACKING_PY.is_file(),
        "g1_urdf_exists": G1_URDF.is_file(),
        "isaaclab_headless_experience_exists": ISAACLAB_HEADLESS_KIT.is_file(),
        "project_egl_icd_exists": PROJECT_EGL_ICD.is_file(),
        "uses_simulationapp_runtime": payload.get("runtime") == "isaacsim.simulation_app.SimulationApp",
        "uses_isaaclab_headless_experience": payload.get("experience") == str(ISAACLAB_HEADLESS_KIT),
        "app_reached_after_app": probe["markers"]["sentinel_after_app"],
        "payload_recorded": probe["markers"]["sentinel_payload"],
        "vulkan_device_lost_recorded": probe["markers"]["vulkan_device_lost"],
        "in_memory_stage_branch_recorded": probe["markers"]["in_memory_stage_branch"],
        "urdf_extension_enabled": payload.get("urdf_extension_enabled_after") is True,
        "parse_and_import_returned_true": parse_ok,
        "current_stage_has_robot": current_robot_ok,
        "exported_stage_has_robot": exported_robot_ok,
        "current_stage_export_attempted": "current_stage_export_return" in payload
        or "current_stage_export_exception" in payload,
        "dest_path_was_empty": True,
        "does_not_start_replay_or_training": True,
        "does_not_claim_motion_npz": True,
    }
    summary: dict[str, Any] = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_g1_urdf_simulationapp_in_memory_import_probe",
        "scope": (
            "Bounded G1 URDF importer probe that uses raw SimulationApp with the IsaacLab headless experience and "
            "dest_path=\"\" to avoid the URDF importer's layered file-output path. It validates generated USD "
            "structure only. No csv_to_npz replay, PPO, policy evaluation, VAE/diffusion run, video, checkpoint, "
            "or robot execution is performed."
        ),
        "current_blocker": blocker,
        "probe": probe,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "A valid exported G1 USD would still need official csv_to_npz wiring, motion.npz validation, "
                "replay_npz execution, and tracking task smoke/evaluation before any tracking reproduction claim. "
                "A blocked in-memory import remains an Isaac/Kit gate, not a paper-level reproduction result."
            ),
        },
        "outputs": {"json": str(OUT / "tracking_g1_urdf_simulationapp_in_memory_import_probe.json")},
    }
    (OUT / "tracking_g1_urdf_simulationapp_in_memory_import_probe.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "current_blocker": blocker,
                "parse_ok": parse_ok,
                "current_stage_has_robot": current_robot_ok,
                "exported_stage_has_robot": exported_robot_ok,
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
