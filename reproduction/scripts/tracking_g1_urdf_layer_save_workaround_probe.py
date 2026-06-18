#!/usr/bin/env python3
"""Probe whether the deeper URDF importer Sdf.Layer.Save path can be redirected to Stage.Export."""

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
OUT = ROOT / "res/tracking/g1_urdf_layer_save_workaround"
LOG_DIR = ROOT / "logs/tracking_g1_urdf_layer_save_workaround"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
G1_URDF = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/"
    "unitree_description/urdf/g1/main.urdf"
)
TIMEOUT_SECONDS = 180


PROBE_CODE = r"""
import json
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path("/mnt/infini-data/test/BeyondMimic")
G1_URDF = ROOT / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf"
BASE = ROOT / "tmp/g1_urdf_layer_save_workaround"
BASE.mkdir(parents=True, exist_ok=True)
DEST = BASE / "g1_parse_and_import_layer_save_patch.usd"

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

payload = {"g1_urdf": str(G1_URDF), "dest": str(DEST)}

try:
    import omni.kit.app
    import omni.kit.commands
    import omni.usd
    from isaacsim.core.utils.extensions import enable_extension
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    def inspect_stage(path_or_stage, label):
        result = {"label": label}
        if isinstance(path_or_stage, (str, Path)):
            path = Path(path_or_stage)
            result.update({"usd_path": str(path), "usd_exists": path.exists(), "usd_size": path.stat().st_size if path.exists() else None})
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
            prim_sample = []
            for prim in stage.Traverse():
                prim_count += 1
                if len(prim_sample) < 40:
                    prim_sample.append({"path": str(prim.GetPath()), "type": prim.GetTypeName()})
                if "Joint" in prim.GetTypeName():
                    joint_count += 1
                if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.GetAttribute("physics:rigidBodyEnabled"):
                    rigid_body_like_count += 1
                if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                    articulation_api_count += 1
            result["prim_count"] = prim_count
            result["joint_count"] = joint_count
            result["rigid_body_like_count"] = rigid_body_like_count
            result["articulation_api_count"] = articulation_api_count
            result["prim_sample"] = prim_sample
        return result

    def inspect_layer_file(path, label):
        path = Path(path)
        out = inspect_stage(path, label)
        out["contains_robotish_text"] = False
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")[:200000]
            out["contains_robotish_text"] = any(token in text for token in ["def Xform", "physics:joint", "ArticulationRootAPI", "revolute"])
        return out

    manager = omni.kit.app.get_app().get_extension_manager()
    payload["urdf_extension_enabled_before"] = manager.is_extension_enabled("isaacsim.asset.importer.urdf")
    if not payload["urdf_extension_enabled_before"]:
        enable_extension("isaacsim.asset.importer.urdf")
        for _ in range(5):
            app.update()
    payload["urdf_extension_enabled_after"] = manager.is_extension_enabled("isaacsim.asset.importer.urdf")

    original_create_new = Usd.Stage.CreateNew
    original_layer_save = Sdf.Layer.Save
    layer_save_events = []
    stage_save_events = []

    def export_stage_or_layer(layer):
        path = layer.realPath or layer.identifier
        if not path or layer.anonymous:
            return False
        try:
            stage = Usd.Stage.Open(layer)
            if stage is not None:
                return bool(stage.Export(path))
        except Exception:
            pass
        return bool(layer.Export(path))

    def patched_layer_save(self, *args, **kwargs):
        event = {
            "identifier": self.identifier,
            "realPath": self.realPath,
            "anonymous": bool(self.anonymous),
            "permission_to_save": bool(self.permissionToSave),
        }
        try:
            event["export_return"] = export_stage_or_layer(self)
            layer_save_events.append(event)
            return event["export_return"]
        except Exception as exc:
            event["exception"] = repr(exc)
            layer_save_events.append(event)
            return original_layer_save(self, *args, **kwargs)

    class StageSaveAsExportProxy:
        def __init__(self, stage, path):
            self._stage = stage
            self._path = str(path)

        def Save(self):
            stage_save_events.append({"method": "Save", "path": self._path, "routed_to": "Usd.Stage.Export"})
            return self._stage.Export(self._path)

        def __getattr__(self, name):
            return getattr(self._stage, name)

    def patched_create_new(path, *args, **kwargs):
        stage = original_create_new(path, *args, **kwargs)
        stage_save_events.append({"method": "CreateNew", "path": str(path)})
        return StageSaveAsExportProxy(stage, path)

    payload["layer_save_patch_assignment_ok"] = False
    try:
        Sdf.Layer.Save = patched_layer_save
        payload["layer_save_patch_assignment_ok"] = True
    except Exception as exc:
        payload["layer_save_patch_assignment_exception"] = repr(exc)

    payload["stage_create_new_patch_assignment_ok"] = False
    try:
        Usd.Stage.CreateNew = patched_create_new
        payload["stage_create_new_patch_assignment_ok"] = True
    except Exception as exc:
        payload["stage_create_new_patch_assignment_exception"] = repr(exc)

    direct_layer_path = BASE / "direct_layer_save_patch.usda"
    try:
        direct_stage = original_create_new(str(direct_layer_path))
        UsdGeom.Xform.Define(direct_stage, "/direct")
        direct_stage.SetDefaultPrim(direct_stage.GetPrimAtPath("/direct"))
        payload["direct_layer_save_return"] = bool(direct_stage.GetRootLayer().Save())
    except Exception as exc:
        payload["direct_layer_save_exception"] = repr(exc)
    payload["direct_layer_stage"] = inspect_layer_file(direct_layer_path, "direct_layer_save_patch_test")

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
        try:
            result = omni.kit.commands.execute(
                "URDFParseAndImportFile",
                urdf_path=str(G1_URDF),
                import_config=import_config,
                dest_path=str(DEST),
                get_articulation_root=True,
            )
            payload["parse_and_import_result"] = repr(result)
        except Exception as exc:
            payload["parse_and_import_exception"] = repr(exc)
    finally:
        try:
            Usd.Stage.CreateNew = original_create_new
        except Exception as exc:
            payload["stage_create_new_restore_exception"] = repr(exc)
        if payload.get("layer_save_patch_assignment_ok"):
            try:
                Sdf.Layer.Save = original_layer_save
            except Exception as exc:
                payload["layer_save_restore_exception"] = repr(exc)

    for _ in range(5):
        app.update()

    payload["stage_save_events"] = stage_save_events
    payload["layer_save_events"] = layer_save_events
    payload["dest_stage"] = inspect_layer_file(DEST, "dest_stage_after_parse_and_import")
    current_stage = omni.usd.get_context().get_stage()
    payload["current_stage"] = inspect_stage(current_stage, "current_omni_usd_stage")
    current_export_path = BASE / "current_stage_export.usda"
    try:
        payload["current_stage_export_return"] = bool(current_stage.Export(str(current_export_path)))
    except Exception as exc:
        payload["current_stage_export_exception"] = repr(exc)
    payload["current_stage_export"] = inspect_layer_file(current_export_path, "current_stage_export")

    config_layers = []
    for path in sorted((BASE / "configuration").glob("*.usd")) + sorted((BASE / "configuration").glob("*.usda")):
        config_layers.append(inspect_layer_file(path, f"configuration/{path.name}"))
    payload["configuration_layers"] = config_layers
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
        "layer_save_patch_type_error": "cannot set" in lowered and "save" in lowered,
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
    log_path = LOG_DIR / "tracking_g1_urdf_layer_save_workaround_probe.log"
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
        and stage_summary.get("default_prim_valid")
        and stage_summary.get("prim_count", 0) > 10
        and stage_summary.get("joint_count", 0) > 0
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(ROOT / "tmp/g1_urdf_layer_save_workaround", ignore_errors=True)
    (ROOT / "tmp/g1_urdf_layer_save_workaround").mkdir(parents=True, exist_ok=True)
    probe = run_probe()
    payload = probe.get("payload", {})
    dest_robot_ok = stage_has_robot(payload.get("dest_stage", {}))
    current_robot_ok = stage_has_robot(payload.get("current_stage", {}))
    current_export_ok = stage_has_robot(payload.get("current_stage_export", {}))
    config_layer_robotish_count = sum(
        1 for row in payload.get("configuration_layers", []) if row.get("prim_count", 0) > 0 or row.get("contains_robotish_text")
    )
    layer_save_patch_ok = payload.get("layer_save_patch_assignment_ok") is True
    stage_create_new_patch_ok = payload.get("stage_create_new_patch_assignment_ok") is True
    layer_save_events = payload.get("layer_save_events", [])
    importer_configuration_layer_save_intercepted = any(
        "/configuration/" in str(row.get("identifier", "")) or "/configuration/" in str(row.get("realPath", ""))
        for row in layer_save_events
    )
    direct_layer_save_ok = payload.get("direct_layer_stage", {}).get("stage_open_ok") is True and payload.get(
        "direct_layer_stage", {}
    ).get("prim_count", 0) > 0

    if dest_robot_ok:
        status = "ok_with_valid_g1_usd"
        blocker = "none_layer_save_patch_generated_valid_g1_usd"
    elif current_robot_ok or current_export_ok:
        status = "ok_with_current_stage_robot_not_dest"
        blocker = "g1_import_reaches_current_stage_but_dest_export_incomplete_after_layer_save_patch"
    elif probe["markers"]["sentinel_payload"] and not layer_save_patch_ok:
        status = "ok_with_layer_save_patch_unavailable"
        blocker = "sdf_layer_save_python_monkeypatch_unavailable"
    elif probe["markers"]["sentinel_payload"] and layer_save_patch_ok and not importer_configuration_layer_save_intercepted:
        status = "ok_with_cpp_importer_save_path_not_intercepted"
        blocker = "sdf_layer_save_patch_applied_but_cpp_importer_save_path_not_intercepted"
    elif probe["markers"]["sentinel_payload"] and layer_save_patch_ok:
        status = "ok_with_importer_still_empty_after_layer_save_patch"
        blocker = "layer_save_patch_applied_but_importer_output_empty"
    else:
        status = "failed"
        blocker = "g1_layer_save_workaround_probe_failed_before_classification"

    checks = {
        "tracking_python_exists": TRACKING_PY.is_file(),
        "g1_urdf_exists": G1_URDF.is_file(),
        "project_egl_icd_exists": PROJECT_EGL_ICD.is_file(),
        "app_reached_after_app": probe["markers"]["sentinel_after_app"],
        "payload_recorded": probe["markers"]["sentinel_payload"],
        "urdf_extension_enabled": payload.get("urdf_extension_enabled_after") is True,
        "sdf_layer_save_patch_assignment_ok": layer_save_patch_ok,
        "stage_create_new_patch_assignment_ok": stage_create_new_patch_ok,
        "layer_save_events_recorded": len(layer_save_events) > 0,
        "importer_configuration_layer_save_intercepted": importer_configuration_layer_save_intercepted,
        "direct_layer_save_patch_test_opened": direct_layer_save_ok,
        "dest_stage_has_robot": dest_robot_ok,
        "current_stage_has_robot": current_robot_ok,
        "current_stage_export_has_robot": current_export_ok,
        "configuration_layer_count": len(payload.get("configuration_layers", [])),
        "configuration_layer_robotish_count": config_layer_robotish_count,
        "does_not_start_replay_or_training": True,
        "does_not_claim_motion_npz": True,
    }
    summary: dict[str, Any] = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_g1_urdf_layer_save_workaround_probe",
        "scope": (
            "Bounded G1 URDF importer probe that tests whether Sdf.Layer.Save can be monkeypatched to export "
            "deeper URDF importer base/physics/sensor layers. It validates generated USD structure only. No "
            "csv_to_npz replay, PPO, policy evaluation, VAE/diffusion run, video, checkpoint, or robot execution "
            "is performed."
        ),
        "current_blocker": blocker,
        "probe": probe,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "A valid official replay still requires a non-empty official G1 USD, csv_to_npz execution, "
                "motion.npz validation, replay_npz execution, and tracking task smoke/evaluation. This probe only "
                "localizes whether the deeper USD layer-save path is patchable in Python."
            ),
        },
        "outputs": {"json": str(OUT / "tracking_g1_urdf_layer_save_workaround_probe.json")},
    }
    (OUT / "tracking_g1_urdf_layer_save_workaround_probe.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "current_blocker": blocker,
                "layer_save_patch_ok": layer_save_patch_ok,
                "layer_save_event_count": len(layer_save_events),
                "importer_configuration_layer_save_intercepted": importer_configuration_layer_save_intercepted,
                "dest_stage_has_robot": dest_robot_ok,
                "configuration_layer_count": len(payload.get("configuration_layers", [])),
                "configuration_layer_robotish_count": config_layer_robotish_count,
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
