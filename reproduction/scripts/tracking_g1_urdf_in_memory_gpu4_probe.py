#!/usr/bin/env python3
"""Probe official G1 URDF importer in-memory on the current GPU4 headless gate."""

from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe"
LOG_DIR = ROOT / "logs/tracking_g1_urdf_in_memory_gpu4_probe"
FAILED_DIR = ROOT / "res/failed_runs/tracking_g1_urdf_in_memory_gpu4_probe"
TMP_DIR = ROOT / "tmp/g1_urdf_in_memory_gpu4_probe"
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
EXPORT_USD = OUT / "g1_official_importer_in_memory_gpu4_export.usda"
TIMEOUT_SECONDS = 240


PROBE_CODE = r"""
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path("/mnt/infini-data/test/BeyondMimic")
G1_URDF = ROOT / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf"
EXPORT_USD = ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda"


def inspect_stage(stage, label):
    from pxr import UsdPhysics

    result = {"label": label, "stage_open_ok": stage is not None}
    if stage is None:
        return result
    default_prim = stage.GetDefaultPrim()
    result["default_prim_valid"] = bool(default_prim)
    result["default_prim_path"] = str(default_prim.GetPath()) if default_prim else None
    prim_count = 0
    rigid_body_like_count = 0
    joint_count = 0
    articulation_api_count = 0
    target_body_hits = {}
    joint_hits = {}
    sample = []
    target_body_names = [
        "torso_link",
        "left_hip_pitch_link",
        "right_hip_pitch_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_shoulder_pitch_link",
        "right_shoulder_pitch_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "pelvis",
    ]
    action_joint_names = [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
    ]
    for prim in stage.Traverse():
        prim_count += 1
        path = str(prim.GetPath())
        name = prim.GetName()
        if len(sample) < 50:
            sample.append({"path": path, "type": prim.GetTypeName()})
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.GetAttribute("physics:rigidBodyEnabled"):
            rigid_body_like_count += 1
        if "Joint" in prim.GetTypeName():
            joint_count += 1
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_api_count += 1
        if name in target_body_names:
            target_body_hits[name] = path
        if name in action_joint_names:
            joint_hits[name] = path
    result.update(
        {
            "prim_count": prim_count,
            "rigid_body_like_count": rigid_body_like_count,
            "joint_count": joint_count,
            "articulation_api_count": articulation_api_count,
            "target_body_hit_count": len(target_body_hits),
            "target_body_hits": target_body_hits,
            "sample_action_joint_hit_count": len(joint_hits),
            "sample_action_joint_hits": joint_hits,
            "prim_sample": sample,
        }
    )
    return result


payload = {"g1_urdf": str(G1_URDF), "export_usd": str(EXPORT_USD)}
print("BM_SENTINEL:before_app", flush=True)
launcher = AppLauncher(
    headless=True,
    enable_cameras=False,
    device="cuda:4",
    fast_shutdown=False,
    multi_gpu=False,
    kit_args=(
        "--/renderer/multiGpu/enabled=false "
        "--/renderer/multiGpu/autoEnable=false "
        "--/renderer/multiGpu/maxGpuCount=1 "
        "--/renderer/activeGpu=4 "
        "--/physics/cudaDevice=4"
    ),
)
app = launcher.app
print("BM_SENTINEL:after_app", flush=True)
try:
    import omni.kit.commands
    import omni.usd
    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.asset.importer.urdf")
    for _ in range(8):
        app.update()

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

    print("BM_SENTINEL:before_parse_import_in_memory", flush=True)
    result = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(G1_URDF),
        import_config=import_config,
        dest_path="",
        get_articulation_root=True,
    )
    payload["parse_import_result_repr"] = repr(result)
    print("BM_SENTINEL:after_parse_import_in_memory", flush=True)
    for _ in range(8):
        app.update()

    stage = omni.usd.get_context().get_stage()
    payload["current_stage"] = inspect_stage(stage, "current_stage_after_in_memory_import")
    try:
        EXPORT_USD.parent.mkdir(parents=True, exist_ok=True)
        payload["export_return"] = bool(stage.Export(str(EXPORT_USD)))
    except BaseException as exc:
        payload["export_exception"] = repr(exc)
    payload["export_exists"] = EXPORT_USD.exists()
    payload["export_size"] = EXPORT_USD.stat().st_size if EXPORT_USD.exists() else None
    if EXPORT_USD.exists():
        from pxr import Usd

        exported_stage = Usd.Stage.Open(str(EXPORT_USD))
        payload["exported_stage"] = inspect_stage(exported_stage, "exported_stage")
except BaseException as exc:
    payload["exception"] = repr(exc)
print("BM_SENTINEL:payload:" + json.dumps(payload, sort_keys=True), flush=True)
try:
    app.close()
    print("BM_SENTINEL:after_close", flush=True)
except BaseException as exc:
    print("BM_SENTINEL:close_exception=" + repr(exc), flush=True)
os._exit(0)
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
            "LD_LIBRARY_PATH": f"{GPU_FOUNDATION_DEPS}:{env.get('LD_LIBRARY_PATH', '')}",
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def extract_payload(text: str) -> dict[str, Any]:
    for line in text.splitlines():
        if line.startswith("BM_SENTINEL:payload:"):
            return json.loads(line.split("BM_SENTINEL:payload:", 1)[1])
    return {}


def classify(text: str, timed_out: bool) -> dict[str, bool]:
    lowered = text.lower()
    return {
        "timed_out": timed_out,
        "before_app": "BM_SENTINEL:before_app" in text,
        "after_app": "BM_SENTINEL:after_app" in text,
        "before_parse_import_in_memory": "BM_SENTINEL:before_parse_import_in_memory" in text,
        "after_parse_import_in_memory": "BM_SENTINEL:after_parse_import_in_memory" in text,
        "payload": "BM_SENTINEL:payload:" in text,
        "after_close": "BM_SENTINEL:after_close" in text,
        "vulkan_device_lost": "error_device_lost" in lowered or "gpu crash is detected" in lowered,
        "usd_save_not_allowed": "saving not allowed" in lowered,
        "traceback": "traceback (most recent call last)" in lowered,
        "segfault": "segmentation fault" in lowered,
        "p2p_iommu_warning": "cuda peer-to-peer observed bandwidth" in lowered or "iommu is enabled" in lowered,
    }


def determine_status(payload: dict[str, Any], markers: dict[str, bool], returncode: int) -> str:
    current_stage = payload.get("current_stage", {})
    exported_stage = payload.get("exported_stage", {})
    current_ok = current_stage.get("rigid_body_like_count", 0) > 0 and current_stage.get("joint_count", 0) > 0
    exported_ok = exported_stage.get("rigid_body_like_count", 0) > 0 and exported_stage.get("joint_count", 0) > 0
    if markers["after_parse_import_in_memory"] and current_ok and exported_ok:
        return "ok_official_g1_in_memory_import_export"
    if markers["vulkan_device_lost"]:
        return "ok_with_vulkan_device_lost_blocker"
    if markers["after_parse_import_in_memory"] and not current_ok:
        return "ok_with_empty_stage_after_in_memory_import"
    if "exception" in payload:
        return "ok_with_in_memory_import_exception"
    if returncode not in (0,):
        return "failed_process_returncode"
    return "failed_unknown"


def latest_blocker(status: str, markers: dict[str, bool], checks: dict[str, bool]) -> str:
    if status == "ok_official_g1_in_memory_import_export":
        return "none_official_g1_in_memory_import_export_completed"
    if markers["vulkan_device_lost"] and checks["export_exists"]:
        return "official_g1_in_memory_import_exported_stage_but_vulkan_device_lost_before_payload"
    if markers["vulkan_device_lost"]:
        return "official_g1_in_memory_import_vulkan_device_lost_before_payload_or_export"
    if checks["export_exists"] and not (checks["export_has_rigid_bodies"] and checks["export_has_joints"]):
        return "official_g1_in_memory_import_export_missing_physics_contract"
    if markers["after_parse_import_in_memory"] and not checks["payload_recorded"]:
        return "official_g1_in_memory_import_returned_without_payload_record"
    return "official_g1_in_memory_import_unclassified_blocker"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    if EXPORT_USD.exists():
        EXPORT_USD.unlink()
    shutil.rmtree(TMP_DIR / "local_import_cache", ignore_errors=True)
    log_path = LOG_DIR / "tracking_g1_urdf_in_memory_gpu4_probe.log"
    start = time.time()
    timed_out = False
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        proc = subprocess.Popen(
            [str(TRACKING_PY), "-c", textwrap.dedent(PROBE_CODE)],
            cwd=ROOT,
            env=base_env(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        while proc.poll() is None:
            if time.time() - start > TIMEOUT_SECONDS:
                timed_out = True
                log_file.write(f"\nBM_TIMEOUT:exceeded_{TIMEOUT_SECONDS}s\n")
                log_file.flush()
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=60)
                break
            time.sleep(5)
    returncode = proc.returncode
    output = log_path.read_text(encoding="utf-8", errors="replace")
    markers = classify(output, timed_out)
    payload = extract_payload(output)
    status = determine_status(payload, markers, returncode)
    checks = {
        "tracking_python_exists": TRACKING_PY.is_file(),
        "g1_urdf_exists": G1_URDF.is_file(),
        "project_egl_icd_exists": PROJECT_EGL_ICD.is_file(),
        "app_launcher_reached": markers["after_app"],
        "in_memory_import_returned": markers["after_parse_import_in_memory"],
        "payload_recorded": markers["payload"],
        "export_exists": EXPORT_USD.is_file(),
        "export_has_rigid_bodies": payload.get("exported_stage", {}).get("rigid_body_like_count", 0) > 0,
        "export_has_joints": payload.get("exported_stage", {}).get("joint_count", 0) > 0,
        "does_not_start_replay_or_training": True,
        "does_not_claim_motion_npz": True,
        "does_not_claim_paper_level_replay": True,
    }
    blocker = latest_blocker(status, markers, checks)
    failed_copy = ""
    if status != "ok_official_g1_in_memory_import_export":
        failed_copy_path = FAILED_DIR / "tracking_g1_urdf_in_memory_gpu4_probe.log"
        failed_copy_path.write_text(output, encoding="utf-8", errors="replace")
        failed_copy = str(failed_copy_path)
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_g1_urdf_in_memory_gpu4_probe",
        "scope": (
            "Uses the official Isaac Sim URDF importer command surface for the official G1 URDF with dest_path='' "
            "on the current physical GPU4 headless gate, then attempts to export the current stage into res/. "
            "This probes a possible official converter workaround only; it does not run csv_to_npz, replay_npz, "
            "tracking evaluation, PPO, DAgger, or paper-level rollout."
        ),
        "returncode": returncode,
        "duration_seconds": round(time.time() - start, 3),
        "markers": markers,
        "payload": payload,
        "checks": checks,
        "latest_blocker": blocker,
        "outputs": {
            "json": str(OUT / "tracking_g1_urdf_in_memory_gpu4_probe.json"),
            "log": str(log_path),
            "failed_log_copy": failed_copy,
            "export_usd": str(EXPORT_USD) if EXPORT_USD.is_file() else "",
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "A successful in-memory import/export would still need to be wired into official csv_to_npz/replay "
                "and task evaluation before paper-facing tracking claims. A failed probe keeps the official "
                "G1 converter/replay gate incomplete."
            ),
        },
    }
    (OUT / "tracking_g1_urdf_in_memory_gpu4_probe.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "json": summary["outputs"]["json"],
                "returncode": returncode,
                "export_exists": checks["export_exists"],
                "export_has_rigid_bodies": checks["export_has_rigid_bodies"],
                "export_has_joints": checks["export_has_joints"],
            },
            sort_keys=True,
        )
    )
    if status.startswith("failed_"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
