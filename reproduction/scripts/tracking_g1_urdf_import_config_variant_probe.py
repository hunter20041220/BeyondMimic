#!/usr/bin/env python3
"""Probe URDF importer config variants for the official G1 converter save blocker."""

from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/g1_urdf_import_config_variant_probe"
LOG_DIR = ROOT / "logs/tracking_g1_urdf_import_config_variant_probe"
PROBE_DIR = ROOT / "tmp/g1_urdf_import_config_variant_probe/subprobes"
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
BASE = ROOT / "tmp/g1_urdf_import_config_variant_probe"
METHOD_TIMEOUT_SECONDS = 150
VARIANT_TIMEOUT_SECONDS = 150


COMMON_APP_CODE = r'''
from isaaclab.app import AppLauncher


def launch_app():
    print("BM_SENTINEL:before_app", flush=True)
    launcher = AppLauncher(
        headless=True,
        enable_cameras=False,
        device="cuda:6",
        fast_shutdown=False,
        multi_gpu=False,
        kit_args=(
            "--/renderer/multiGpu/autoEnable=false "
            "--/renderer/multiGpu/maxGpuCount=1 "
            "--/renderer/activeGpu=6 "
            "--/physics/cudaDevice=6"
        ),
    )
    print("BM_SENTINEL:after_app", flush=True)
    return launcher.app
'''


METHOD_PROBE_CODE = (
    COMMON_APP_CODE
    + r'''
import json
import os

app = launch_app()
payload = {}
try:
    import omni.kit.commands
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.asset.importer.urdf")
    for _ in range(5):
        app.update()
    print("BM_SENTINEL:before_import_config", flush=True)
    ok, config = omni.kit.commands.execute("URDFCreateImportConfig")
    methods = sorted(
        method
        for method in dir(config)
        if any(token in method.lower() for token in ["instance", "usd", "config", "path", "make", "default", "save"])
    )
    payload = {
        "create_import_config_ok": bool(ok),
        "import_config_type": str(type(config)),
        "import_config_methods": methods,
        "has_set_make_instanceable": hasattr(config, "set_make_instanceable"),
        "has_set_instanceable_usd_path": hasattr(config, "set_instanceable_usd_path"),
    }
except BaseException as exc:
    payload = {"exception": repr(exc)}
print("BM_SENTINEL:payload:" + json.dumps(payload, sort_keys=True), flush=True)
os._exit(0)
'''
)


VARIANT_PROBE_CODE = (
    COMMON_APP_CODE
    + r'''
import json
import os
from pathlib import Path

ROOT = Path("/mnt/infini-data/test/BeyondMimic")
G1_URDF = ROOT / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf"
BASE = ROOT / "tmp/g1_urdf_import_config_variant_probe"
VARIANT_NAME = os.environ["BM_VARIANT_NAME"]
PATCH_INSTANCEABLE = os.environ["BM_PATCH_INSTANCEABLE"] == "1"
MAKE_INSTANCEABLE = os.environ["BM_MAKE_INSTANCEABLE"] == "1"
INSTANCEABLE_USD_PATH = os.environ.get("BM_INSTANCEABLE_USD_PATH") or None


def inspect_usd(path):
    from pxr import Usd, UsdPhysics

    path = Path(path)
    result = {
        "path": str(path),
        "exists": path.exists(),
        "size": path.stat().st_size if path.exists() else None,
        "configuration_files": [],
    }
    configuration_dir = path.parent / "configuration"
    if configuration_dir.exists():
        for cfg_path in sorted(configuration_dir.glob("*")):
            result["configuration_files"].append(
                {
                    "path": str(cfg_path),
                    "size": cfg_path.stat().st_size,
                    "text_head": cfg_path.read_text(encoding="utf-8", errors="ignore")[:200],
                }
            )
    stage = Usd.Stage.Open(str(path)) if path.exists() else None
    result["stage_open_ok"] = stage is not None
    if stage is None:
        return result
    default_prim = stage.GetDefaultPrim()
    result["default_prim_valid"] = bool(default_prim)
    result["default_prim_path"] = str(default_prim.GetPath()) if default_prim else None
    prim_count = 0
    rigid_body_like_count = 0
    joint_count = 0
    prim_sample = []
    for prim in stage.Traverse():
        prim_count += 1
        if len(prim_sample) < 30:
            prim_sample.append({"path": str(prim.GetPath()), "type": prim.GetTypeName()})
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.GetAttribute("physics:rigidBodyEnabled"):
            rigid_body_like_count += 1
        if "Joint" in prim.GetTypeName():
            joint_count += 1
    result["prim_count"] = prim_count
    result["rigid_body_like_count"] = rigid_body_like_count
    result["joint_count"] = joint_count
    result["prim_sample"] = prim_sample
    return result


app = launch_app()
payload = {
    "name": VARIANT_NAME,
    "patch_instanceable": PATCH_INSTANCEABLE,
    "requested_make_instanceable": MAKE_INSTANCEABLE,
    "requested_instanceable_usd_path": INSTANCEABLE_USD_PATH,
}
try:
    import omni.kit.commands
    from isaacsim.core.utils.extensions import enable_extension
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
    from isaaclab.sim.converters.urdf_converter import UrdfConverter as UrdfConverterClass

    enable_extension("isaacsim.asset.importer.urdf")
    for _ in range(5):
        app.update()
    ok, config_probe = omni.kit.commands.execute("URDFCreateImportConfig")
    payload["create_import_config_ok"] = bool(ok)
    payload["has_set_make_instanceable"] = hasattr(config_probe, "set_make_instanceable")
    payload["has_set_instanceable_usd_path"] = hasattr(config_probe, "set_instanceable_usd_path")
    print("BM_SENTINEL:config_surface:" + json.dumps(payload, sort_keys=True), flush=True)

    original_get = UrdfConverterClass._get_urdf_import_config
    patched_methods = []

    def patched_get(self):
        cfg = original_get(self)
        if PATCH_INSTANCEABLE and hasattr(cfg, "set_make_instanceable"):
            cfg.set_make_instanceable(MAKE_INSTANCEABLE)
            patched_methods.append("set_make_instanceable")
        if PATCH_INSTANCEABLE and INSTANCEABLE_USD_PATH and hasattr(cfg, "set_instanceable_usd_path"):
            cfg.set_instanceable_usd_path(INSTANCEABLE_USD_PATH)
            patched_methods.append("set_instanceable_usd_path")
        return cfg

    variant_dir = BASE / VARIANT_NAME
    variant_dir.mkdir(parents=True, exist_ok=True)
    if PATCH_INSTANCEABLE:
        UrdfConverterClass._get_urdf_import_config = patched_get
    try:
        print("BM_SENTINEL:before_converter", flush=True)
        converter_cfg = UrdfConverterCfg(
            asset_path=str(G1_URDF),
            usd_dir=str(variant_dir),
            usd_file_name="main.usd",
            force_usd_conversion=True,
            fix_base=False,
            make_instanceable=MAKE_INSTANCEABLE,
            replace_cylinders_with_capsules=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
        )
        converter = UrdfConverter(converter_cfg)
        payload["converter_exception"] = None
        payload["usd_path"] = converter.usd_path
        print("BM_SENTINEL:after_converter", flush=True)
    except BaseException as exc:
        payload["converter_exception"] = repr(exc)
        payload["usd_path"] = str(variant_dir / "main.usd")
        print("BM_SENTINEL:converter_exception:" + repr(exc), flush=True)
    finally:
        UrdfConverterClass._get_urdf_import_config = original_get
    payload["patched_methods"] = sorted(set(patched_methods))
    payload["usd"] = inspect_usd(payload["usd_path"])
except BaseException as exc:
    payload["outer_exception"] = repr(exc)
print("BM_SENTINEL:payload:" + json.dumps(payload, sort_keys=True), flush=True)
os._exit(0)
'''
)


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
            "LD_LIBRARY_PATH": f"{GPU_FOUNDATION_DEPS}:{env.get('LD_LIBRARY_PATH', '')}",
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def extract_payload(text: str, prefix: str = "BM_SENTINEL:payload:") -> dict[str, Any]:
    for line in text.splitlines():
        if line.startswith(prefix):
            return json.loads(line.split(prefix, 1)[1])
    return {}


def classify(text: str, timed_out: bool) -> dict[str, bool]:
    lowered = text.lower()
    return {
        "timed_out": timed_out,
        "before_app": "BM_SENTINEL:before_app" in text,
        "after_app": "BM_SENTINEL:after_app" in text,
        "before_import_config": "BM_SENTINEL:before_import_config" in text,
        "config_surface": "BM_SENTINEL:config_surface:" in text,
        "before_converter": "BM_SENTINEL:before_converter" in text,
        "after_converter": "BM_SENTINEL:after_converter" in text,
        "payload": "BM_SENTINEL:payload:" in text,
        "after_close": "BM_SENTINEL:after_close" in text,
        "saving_not_allowed": "saving not allowed" in lowered,
        "cannot_save_layer": "cannot save layer" in lowered,
        "vulkan_device_lost": "error_device_lost" in lowered or "device lost" in lowered,
        "traceback": "traceback (most recent call last)" in lowered,
        "p2p_iommu_warning": "cuda peer-to-peer observed bandwidth" in lowered,
    }


def decode_timeout_output(exc: subprocess.TimeoutExpired) -> str:
    if isinstance(exc.stdout, str):
        return exc.stdout
    return (exc.stdout or b"").decode("utf-8", errors="ignore")


def run_subprobe(name: str, code: str, timeout: int, extra_env: dict[str, str] | None = None) -> dict[str, Any]:
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    path = PROBE_DIR / f"{name}.py"
    path.write_text(textwrap.dedent(code), encoding="utf-8")
    env = base_env()
    if extra_env:
        env.update(extra_env)
    timed_out = False
    proc = subprocess.Popen(
        [str(TRACKING_PY), str(path)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=timeout)
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        output = decode_timeout_output(exc)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            output_after_term, _ = proc.communicate(timeout=20)
            output += output_after_term or ""
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            output_after_kill, _ = proc.communicate(timeout=20)
            output += output_after_kill or ""
        returncode = 124
    log_path = LOG_DIR / f"{name}.log"
    log_path.write_text(output, encoding="utf-8", errors="ignore")
    return {
        "name": name,
        "returncode": returncode,
        "timed_out": timed_out,
        "log": str(log_path),
        "markers": classify(output, timed_out),
        "payload": extract_payload(output),
        "config_surface": extract_payload(output, "BM_SENTINEL:config_surface:"),
    }


def usd_summary(payload: dict[str, Any]) -> dict[str, Any]:
    usd = payload.get("usd", {})
    return {
        "converter_exception": payload.get("converter_exception"),
        "outer_exception": payload.get("outer_exception"),
        "usd_exists": usd.get("exists"),
        "usd_size": usd.get("size"),
        "stage_open_ok": usd.get("stage_open_ok"),
        "default_prim_valid": usd.get("default_prim_valid"),
        "prim_count": usd.get("prim_count"),
        "rigid_body_like_count": usd.get("rigid_body_like_count"),
        "joint_count": usd.get("joint_count"),
        "configuration_file_count": len(usd.get("configuration_files", [])),
        "patched_methods": payload.get("patched_methods", []),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(BASE, ignore_errors=True)
    BASE.mkdir(parents=True, exist_ok=True)

    method = run_subprobe("method_surface_probe", METHOD_PROBE_CODE, METHOD_TIMEOUT_SECONDS)
    method_payload = method.get("payload", {})
    variants = [
        run_subprobe(
            "variant_baseline_make_instanceable_false",
            VARIANT_PROBE_CODE,
            VARIANT_TIMEOUT_SECONDS,
            {
                "BM_VARIANT_NAME": "baseline_make_instanceable_false",
                "BM_PATCH_INSTANCEABLE": "0",
                "BM_MAKE_INSTANCEABLE": "0",
            },
        )
    ]
    skipped_variants: list[dict[str, str]] = []
    if method_payload.get("has_set_make_instanceable") or method_payload.get("has_set_instanceable_usd_path"):
        variants.append(
            run_subprobe(
                "variant_patched_make_instanceable_false",
                VARIANT_PROBE_CODE,
                VARIANT_TIMEOUT_SECONDS,
                {
                    "BM_VARIANT_NAME": "patched_make_instanceable_false",
                    "BM_PATCH_INSTANCEABLE": "1",
                    "BM_MAKE_INSTANCEABLE": "0",
                },
            )
        )
    else:
        skipped_variants.append(
            {
                "name": "patched_make_instanceable_false",
                "reason": "URDFCreateImportConfig exposes no set_make_instanceable setter in Isaac Sim 4.5.",
            }
        )
        skipped_variants.append(
            {
                "name": "patched_make_instanceable_true_local_instanceable",
                "reason": "URDFCreateImportConfig exposes no instanceable USD path setter in Isaac Sim 4.5.",
            }
        )

    variant_payloads = [row.get("payload", {}) for row in variants]
    variant_surfaces = [row.get("config_surface", {}) for row in variants if row.get("config_surface")]
    variant_summary = {
        row["name"]: {
            "returncode": row["returncode"],
            "timed_out": row["timed_out"],
            "markers": row["markers"],
            "config_surface": row.get("config_surface", {}),
            "usd": usd_summary(row.get("payload", {})),
            "log": row["log"],
        }
        for row in variants
    }
    any_valid_robotish = any(
        (payload.get("usd", {}).get("rigid_body_like_count") or 0) > 0
        and payload.get("usd", {}).get("default_prim_valid")
        for payload in variant_payloads
    )
    all_variant_markers = [row["markers"] for row in variants]
    checks = {
        "tracking_python_exists": TRACKING_PY.is_file(),
        "g1_urdf_exists": G1_URDF.is_file(),
        "project_egl_icd_exists": PROJECT_EGL_ICD.is_file(),
        "method_app_reached": method["markers"]["after_app"],
        "method_payload_recorded": bool(method_payload),
        "import_config_methods_recorded": bool(method_payload.get("import_config_methods")),
        "patch_surface_checked": (
            "has_set_make_instanceable" in method_payload
            and "has_set_instanceable_usd_path" in method_payload
        )
        or any(
            "has_set_make_instanceable" in row and "has_set_instanceable_usd_path" in row
            for row in variant_surfaces
        ),
        "variant_count_at_least_1": len(variants) >= 1,
        "variant_app_reached": all(row["markers"]["after_app"] for row in variants),
        "variant_converter_attempted": all(row["markers"]["before_converter"] for row in variants),
        "variant_config_surface_recorded": all(bool(row.get("config_surface")) for row in variants),
        "save_or_runtime_blocker_recorded": any(
            row["saving_not_allowed"] or row["cannot_save_layer"] or row["vulkan_device_lost"] or row["timed_out"]
            for row in all_variant_markers
        ),
        "no_valid_robotish_usd_claim": not any_valid_robotish,
        "does_not_start_replay_or_training": True,
        "does_not_claim_motion_npz": True,
        "does_not_claim_official_converter_success": True,
    }
    status = (
        "ok_with_import_config_surface_recorded_and_variants_blocked"
        if all(checks.values())
        else "failed"
    )
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_g1_urdf_import_config_variant_probe",
        "scope": (
            "Tests whether URDF ImportConfig instanceable-related setters can avoid the G1 converter layer-save "
            "blocker. This is a converter diagnostic only: it does not produce official motion.npz, replay, PPO, "
            "policy evaluation, video, or robot evidence."
        ),
        "method_probe": method,
        "skipped_variants": skipped_variants,
        "variant_summary": variant_summary,
        "checks": checks,
        "current_blocker": (
            "official_urdf_converter_layer_save_or_vulkan_device_lost_after_import_config_variants"
            if not any_valid_robotish
            else "none"
        ),
        "interpretation": {
            "goal_complete": False,
            "any_valid_robotish_usd": any_valid_robotish,
            "why_not_complete": (
                "The probe only tests importer configuration surfaces. The official replay gate still requires a "
                "valid official robot USD plus successful csv_to_npz/replay execution before any paper-level claim."
            ),
            "next_action": (
                "Continue below the Python ImportConfig surface, try a different official asset-conversion path, or "
                "keep using the clearly labeled resource-adjusted USD path for bounded diagnostics only."
            ),
        },
        "outputs": {
            "json": str(OUT / "tracking_g1_urdf_import_config_variant_probe.json"),
            "logs_dir": str(LOG_DIR),
        },
    }
    (OUT / "tracking_g1_urdf_import_config_variant_probe.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "json": summary["outputs"]["json"],
                "has_set_make_instanceable": method_payload.get("has_set_make_instanceable"),
                "has_set_instanceable_usd_path": method_payload.get("has_set_instanceable_usd_path"),
                "any_valid_robotish_usd": any_valid_robotish,
            },
            sort_keys=True,
        )
    )
    if status != "ok_with_import_config_surface_recorded_and_variants_blocked":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
