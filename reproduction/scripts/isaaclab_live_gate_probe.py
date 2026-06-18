#!/usr/bin/env python3
"""Probe IsaacLab live AppLauncher gate without starting training."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/setup/isaaclab_live_gate_probe"
LOG_DIR = ROOT / "logs/setup/isaaclab_live_gate_probe"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
TIMEOUT_SECONDS = 180


APP_PROBE = r"""
import json
import os
import sys

print("BM_SENTINEL:before_import", flush=True)
from isaaclab.app import AppLauncher

print("BM_SENTINEL:before_app", flush=True)
launcher_kwargs = {
    "headless": True,
    "enable_cameras": False,
    "device": os.environ.get("BM_ISAACLAB_DEVICE", "cuda:0"),
}
if os.environ.get("BM_ISAACLAB_KIT_ARGS"):
    launcher_kwargs["kit_args"] = os.environ["BM_ISAACLAB_KIT_ARGS"]
if os.environ.get("BM_ISAACLAB_MULTI_GPU") is not None:
    launcher_kwargs["multi_gpu"] = os.environ["BM_ISAACLAB_MULTI_GPU"].lower() in {"1", "true", "yes"}
if os.environ.get("BM_ISAACLAB_FAST_SHUTDOWN") is not None:
    launcher_kwargs["fast_shutdown"] = os.environ["BM_ISAACLAB_FAST_SHUTDOWN"].lower() in {"1", "true", "yes"}
launcher = AppLauncher(**launcher_kwargs)
print("BM_SENTINEL:after_app", flush=True)
app = launcher.app
payload = {
    "is_running": bool(app.is_running()),
    "device": os.environ.get("BM_ISAACLAB_DEVICE", "cuda:0"),
    "fast_shutdown": launcher_kwargs.get("fast_shutdown", "isaacsim_default_true"),
    "multi_gpu": launcher_kwargs.get("multi_gpu", "isaaclab_default"),
}
print("BM_SENTINEL:payload:" + json.dumps(payload, sort_keys=True), flush=True)
app.close()
print("BM_SENTINEL:after_close", flush=True)
"""


def read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def command_output(args: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(args, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except OSError as exc:
        return 127, str(exc)
    return proc.returncode, proc.stdout.strip()


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "TMPDIR": str(ROOT / "tmp"),
            "HOME": str(ROOT / "cache/home"),
            "XDG_CACHE_HOME": str(ROOT / "cache/xdg"),
            "PIP_CACHE_DIR": str(ROOT / "cache/pip"),
            "OMNI_USER_DIR": str(ROOT / "cache/omniverse/user"),
            "OMNI_LOGS_DIR": str(ROOT / "logs/omniverse"),
            "OMNI_CACHE_DIR": str(ROOT / "cache/omniverse/cache"),
            "OV_USER_DIR": str(ROOT / "cache/omniverse/user"),
            "OV_CACHE_DIR": str(ROOT / "cache/omniverse/cache"),
        }
    )
    return env


def classify_output(text: str, timed_out: bool) -> dict[str, bool]:
    lowered = text.lower()
    return {
        "timed_out": timed_out,
        "sentinel_import_payload": "BM_SENTINEL:import_payload:" in text,
        "sentinel_before_import": "BM_SENTINEL:before_import" in text,
        "sentinel_before_app": "BM_SENTINEL:before_app" in text,
        "sentinel_after_app": "BM_SENTINEL:after_app" in text,
        "sentinel_payload": "BM_SENTINEL:payload:" in text,
        "sentinel_after_close": "BM_SENTINEL:after_close" in text,
        "vulkan_incompatible_driver": "error_incompatible_driver" in lowered
        or "vkcreateinstance failed" in lowered,
        "gpu_foundation_create_instance_failed": "carb::graphics::createinstance failed" in lowered,
        "cuda_visible_devices_warning": "cuda_visible_devices environment variable is set" in lowered,
        "cuda_p2p_iommu_validation_failure": "peer access is already enabled" in lowered
        or "p2pbandwidthlatencytest" in lowered,
        "iommu_enabled_warning": "iommu is enabled" in lowered,
        "gpu_foundation_no_device_created": "no device could be created" in lowered,
        "gpu_foundation_cuda_bad_state": "cuda being in bad state" in lowered,
        "active_gpu_incompatible": "activegpu" in lowered and "not compatible" in lowered,
        "inotify_errno28": "errno=28" in lowered or "failed to create change watch" in lowered,
        "eula_prompt_or_failure": "do you accept the eula" in lowered or "omniverse license agreement" in lowered,
        "omnihub_inaccessible": "omnihub is inaccessible" in lowered,
        "traceback": "traceback (most recent call last)" in lowered,
    }


def has_fatal_app_error(markers: dict[str, bool]) -> bool:
    return any(
        markers[key]
        for key in [
            "timed_out",
            "vulkan_incompatible_driver",
            "gpu_foundation_create_instance_failed",
            "gpu_foundation_no_device_created",
            "gpu_foundation_cuda_bad_state",
            "active_gpu_incompatible",
            "inotify_errno28",
            "eula_prompt_or_failure",
            "traceback",
        ]
    )


def run_probe(
    name: str,
    code: str,
    env_updates: dict[str, str] | None = None,
    timeout: int = TIMEOUT_SECONDS,
    probe_type: str = "app_launcher",
) -> dict[str, Any]:
    env = base_env()
    if env_updates:
        for key, value in env_updates.items():
            if value == "":
                env.pop(key, None)
            else:
                env[key] = value
    cmd = [str(TRACKING_PY), "-c", code]
    timed_out = False
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        returncode = proc.returncode
        output = proc.stdout
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        output = exc.stdout if isinstance(exc.stdout, str) else ""
    (LOG_DIR / f"{name}.log").write_text(output, encoding="utf-8", errors="ignore")
    markers = classify_output(output, timed_out)
    expected_fast_shutdown = env.get("BM_ISAACLAB_FAST_SHUTDOWN", "true").lower() in {"1", "true", "yes"}
    clean_app_payload_without_critical_errors = (
        returncode == 0
        and markers["sentinel_after_app"]
        and markers["sentinel_payload"]
        and not has_fatal_app_error(markers)
    )
    if probe_type == "import":
        ok = returncode == 0 and markers["sentinel_import_payload"]
    else:
        clean_shutdown_observed = clean_app_payload_without_critical_errors and markers["sentinel_after_close"]
        clean_expected_fast_exit = clean_app_payload_without_critical_errors and expected_fast_shutdown
        ok = clean_shutdown_observed or clean_expected_fast_exit
    return {
        "name": name,
        "probe_type": probe_type,
        "cmd": cmd,
        "returncode": returncode,
        "ok": ok,
        "expected_fast_shutdown": expected_fast_shutdown,
        "clean_app_payload_without_critical_errors": clean_app_payload_without_critical_errors,
        "markers": markers,
        "log": str(LOG_DIR / f"{name}.log"),
        "stdout_tail": output[-5000:],
    }


def current_blocker(probes: list[dict[str, Any]]) -> str:
    app_probes = [probe for probe in probes if probe["name"].startswith("app_launcher")]
    if any(probe["ok"] for probe in app_probes):
        return "none"
    advanced_probes = [
        probe
        for probe in app_probes
        if probe["markers"]["sentinel_after_app"]
        and probe["markers"]["sentinel_payload"]
        and not probe["markers"]["vulkan_incompatible_driver"]
        and not probe["markers"]["gpu_foundation_create_instance_failed"]
    ]
    if any(probe["markers"]["cuda_p2p_iommu_validation_failure"] for probe in advanced_probes):
        return "cuda_p2p_iommu_runtime_warning"
    if any(probe["markers"]["gpu_foundation_no_device_created"] for probe in app_probes):
        return "gpu_foundation_no_device_created"
    if any(probe["markers"]["vulkan_incompatible_driver"] for probe in app_probes):
        return "vulkan_incompatible_driver"
    if any(probe["markers"]["inotify_errno28"] for probe in app_probes):
        return "inotify_errno28"
    if any(probe["markers"]["eula_prompt_or_failure"] for probe in app_probes):
        return "eula_prompt_or_failure"
    if any(probe["markers"]["timed_out"] for probe in app_probes):
        return "timeout_before_sentinel"
    if any(probe["returncode"] != 0 for probe in app_probes):
        return "nonzero_return_before_sentinel"
    return "missing_success_sentinel"


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "returncode",
        "ok",
        "expected_fast_shutdown",
        "clean_app_payload_without_critical_errors",
        "sentinel_before_app",
        "sentinel_after_app",
        "sentinel_payload",
        "sentinel_after_close",
        "vulkan_incompatible_driver",
        "inotify_errno28",
        "cuda_visible_devices_warning",
        "cuda_p2p_iommu_validation_failure",
        "iommu_enabled_warning",
        "gpu_foundation_no_device_created",
        "gpu_foundation_cuda_bad_state",
        "active_gpu_incompatible",
        "log",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            markers = row["markers"]
            writer.writerow(
                {
                    "name": row["name"],
                    "returncode": row["returncode"],
                    "ok": row["ok"],
                    "expected_fast_shutdown": row.get("expected_fast_shutdown"),
                    "clean_app_payload_without_critical_errors": row.get(
                        "clean_app_payload_without_critical_errors"
                    ),
                    "sentinel_before_app": markers["sentinel_before_app"],
                    "sentinel_after_app": markers["sentinel_after_app"],
                    "sentinel_payload": markers["sentinel_payload"],
                    "sentinel_after_close": markers["sentinel_after_close"],
                    "vulkan_incompatible_driver": markers["vulkan_incompatible_driver"],
                    "inotify_errno28": markers["inotify_errno28"],
                    "cuda_visible_devices_warning": markers["cuda_visible_devices_warning"],
                    "cuda_p2p_iommu_validation_failure": markers["cuda_p2p_iommu_validation_failure"],
                    "iommu_enabled_warning": markers["iommu_enabled_warning"],
                    "gpu_foundation_no_device_created": markers["gpu_foundation_no_device_created"],
                    "gpu_foundation_cuda_bad_state": markers["gpu_foundation_cuda_bad_state"],
                    "active_gpu_incompatible": markers["active_gpu_incompatible"],
                    "log": row["log"],
                }
            )
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    for path in [ROOT / "tmp", ROOT / "cache/home", ROOT / "cache/xdg", ROOT / "cache/pip", ROOT / "cache/omniverse"]:
        path.mkdir(parents=True, exist_ok=True)

    nvidia_smi_rc, nvidia_smi_out = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,cuda_version,memory.used,memory.total,display_mode,display_active",
            "--format=csv,noheader,nounits",
        ]
    )
    vulkaninfo_path = shutil.which("vulkaninfo")

    import_code = textwrap.dedent(
        """
        import json, os
        import isaacsim, isaaclab
        print("BM_SENTINEL:import_payload:" + json.dumps({
            "isaacsim": isaacsim.__file__,
            "isaaclab": isaaclab.__file__,
            "ISAAC_PATH": os.environ.get("ISAAC_PATH"),
            "EXP_PATH": os.environ.get("EXP_PATH"),
        }, sort_keys=True), flush=True)
        """
    )
    probes = [
        run_probe("package_import_eula_accepted", import_code, timeout=60, probe_type="import"),
        run_probe(
            "app_launcher_no_cuda_visible_devices_device_cuda6",
            APP_PROBE,
            {"CUDA_VISIBLE_DEVICES": "", "BM_ISAACLAB_DEVICE": "cuda:6"},
        ),
        run_probe(
            "app_launcher_project_egl_icd_no_cuda_visible_devices_device_cuda6",
            APP_PROBE,
            {
                "CUDA_VISIBLE_DEVICES": "",
                "BM_ISAACLAB_DEVICE": "cuda:6",
                "VK_ICD_FILENAMES": str(PROJECT_EGL_ICD),
                "LD_LIBRARY_PATH": f"{GPU_FOUNDATION_DEPS}:{os.environ.get('LD_LIBRARY_PATH', '')}",
            },
        ),
        run_probe(
            "app_launcher_project_egl_icd_disable_p2p_candidate_a",
            APP_PROBE,
            {
                "CUDA_VISIBLE_DEVICES": "",
                "BM_ISAACLAB_DEVICE": "cuda:6",
                "VK_ICD_FILENAMES": str(PROJECT_EGL_ICD),
                "LD_LIBRARY_PATH": f"{GPU_FOUNDATION_DEPS}:{os.environ.get('LD_LIBRARY_PATH', '')}",
                "BM_ISAACLAB_KIT_ARGS": "--/plugins/carb.cudainterop.plugin/enableP2P=false --/plugins/carb.cudainterop.plugin/disableP2PAccess=true",
            },
        ),
        run_probe(
            "app_launcher_project_egl_icd_disable_p2p_candidate_b",
            APP_PROBE,
            {
                "CUDA_VISIBLE_DEVICES": "",
                "BM_ISAACLAB_DEVICE": "cuda:6",
                "VK_ICD_FILENAMES": str(PROJECT_EGL_ICD),
                "LD_LIBRARY_PATH": f"{GPU_FOUNDATION_DEPS}:{os.environ.get('LD_LIBRARY_PATH', '')}",
                "BM_ISAACLAB_KIT_ARGS": "--/plugins/carb.cudainterop/enableP2P=false --/plugins/carb.cudainterop/disableP2PAccess=true",
            },
        ),
        run_probe(
            "app_launcher_project_egl_icd_single_gpu_renderer",
            APP_PROBE,
            {
                "CUDA_VISIBLE_DEVICES": "",
                "BM_ISAACLAB_DEVICE": "cuda:6",
                "VK_ICD_FILENAMES": str(PROJECT_EGL_ICD),
                "LD_LIBRARY_PATH": f"{GPU_FOUNDATION_DEPS}:{os.environ.get('LD_LIBRARY_PATH', '')}",
                "BM_ISAACLAB_KIT_ARGS": "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1 --/renderer/activeGpu=6 --/physics/cudaDevice=6",
            },
        ),
        run_probe(
            "app_launcher_project_egl_icd_simapp_multi_gpu_false",
            APP_PROBE,
            {
                "CUDA_VISIBLE_DEVICES": "",
                "BM_ISAACLAB_DEVICE": "cuda:6",
                "BM_ISAACLAB_MULTI_GPU": "false",
                "VK_ICD_FILENAMES": str(PROJECT_EGL_ICD),
                "LD_LIBRARY_PATH": f"{GPU_FOUNDATION_DEPS}:{os.environ.get('LD_LIBRARY_PATH', '')}",
                "BM_ISAACLAB_KIT_ARGS": "--/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1 --/renderer/activeGpu=6 --/physics/cudaDevice=6",
            },
        ),
        run_probe(
            "app_launcher_project_egl_icd_simapp_multi_gpu_false_fast_shutdown_false",
            APP_PROBE,
            {
                "CUDA_VISIBLE_DEVICES": "",
                "BM_ISAACLAB_DEVICE": "cuda:6",
                "BM_ISAACLAB_MULTI_GPU": "false",
                "BM_ISAACLAB_FAST_SHUTDOWN": "false",
                "VK_ICD_FILENAMES": str(PROJECT_EGL_ICD),
                "LD_LIBRARY_PATH": f"{GPU_FOUNDATION_DEPS}:{os.environ.get('LD_LIBRARY_PATH', '')}",
                "BM_ISAACLAB_KIT_ARGS": "--/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1 --/renderer/activeGpu=6 --/physics/cudaDevice=6",
            },
        ),
        run_probe(
            "app_launcher_project_egl_icd_cuda_visible_devices_6_single_gpu_renderer",
            APP_PROBE,
            {
                "CUDA_VISIBLE_DEVICES": "6",
                "BM_ISAACLAB_DEVICE": "cuda:0",
                "VK_ICD_FILENAMES": str(PROJECT_EGL_ICD),
                "LD_LIBRARY_PATH": f"{GPU_FOUNDATION_DEPS}:{os.environ.get('LD_LIBRARY_PATH', '')}",
                "BM_ISAACLAB_KIT_ARGS": "--/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1 --/renderer/activeGpu=0 --/physics/cudaDevice=0",
            },
        ),
        run_probe(
            "app_launcher_project_egl_icd_cpu_device_single_gpu_renderer",
            APP_PROBE,
            {
                "CUDA_VISIBLE_DEVICES": "",
                "BM_ISAACLAB_DEVICE": "cpu",
                "BM_ISAACLAB_MULTI_GPU": "false",
                "VK_ICD_FILENAMES": str(PROJECT_EGL_ICD),
                "LD_LIBRARY_PATH": f"{GPU_FOUNDATION_DEPS}:{os.environ.get('LD_LIBRARY_PATH', '')}",
                "BM_ISAACLAB_KIT_ARGS": "--/renderer/multiGpu/autoEnable=false --/renderer/multiGpu/maxGpuCount=1",
            },
        ),
        run_probe(
            "app_launcher_cuda_visible_devices_6_device_cuda0",
            APP_PROBE,
            {"CUDA_VISIBLE_DEVICES": "6", "BM_ISAACLAB_DEVICE": "cuda:0"},
        ),
    ]

    app_probe_ok = any(probe["ok"] for probe in probes if probe["name"].startswith("app_launcher"))
    blocker = current_blocker(probes)
    p2p_runtime_warning_retained = any(
        probe["name"].startswith("app_launcher")
        and probe["markers"]["cuda_p2p_iommu_validation_failure"]
        and probe["markers"]["sentinel_after_app"]
        for probe in probes
    )
    checks = {
        "tracking_python_exists": TRACKING_PY.is_file(),
        "package_import_probe_ok": probes[0]["ok"],
        "app_launcher_reached_success_sentinel": app_probe_ok,
        "no_training_started": True,
        "does_not_claim_tracking_reproduction_complete": True,
        "current_inotify_limits_meet_targets": (read_int("/proc/sys/fs/inotify/max_user_watches") or 0) >= 524288
        and (read_int("/proc/sys/fs/inotify/max_user_instances") or 0) >= 1024,
        "project_egl_icd_exists": PROJECT_EGL_ICD.is_file(),
        "project_egl_icd_removes_vulkan_error": any(
            probe["name"].startswith("app_launcher_project_egl_icd")
            and probe["markers"]["sentinel_after_app"]
            and not probe["markers"]["vulkan_incompatible_driver"]
            and not probe["markers"]["gpu_foundation_create_instance_failed"]
            for probe in probes
        ),
        "single_gpu_renderer_limits_active_gpu": any(
            probe["name"] == "app_launcher_project_egl_icd_single_gpu_renderer"
            and probe["markers"]["sentinel_after_app"]
            and not probe["markers"]["vulkan_incompatible_driver"]
            for probe in probes
        ),
        "cuda_visible_devices_single_gpu_not_viable": any(
            probe["name"] == "app_launcher_project_egl_icd_cuda_visible_devices_6_single_gpu_renderer"
            and probe["markers"]["gpu_foundation_no_device_created"]
            for probe in probes
        ),
        "fast_shutdown_false_candidate_recorded": any(
            probe["name"] == "app_launcher_project_egl_icd_simapp_multi_gpu_false_fast_shutdown_false"
            and probe.get("expected_fast_shutdown") is False
            for probe in probes
        ),
        "fast_shutdown_semantics_recorded": any(
            probe["name"].startswith("app_launcher")
            and probe.get("expected_fast_shutdown") is True
            and probe["markers"]["sentinel_after_app"]
            and not probe["markers"]["sentinel_after_close"]
            for probe in probes
        ),
        "cuda_p2p_iommu_runtime_warning_retained": p2p_runtime_warning_retained,
    }
    summary: dict[str, Any] = {
        "status": "ok_with_runtime_warning" if app_probe_ok and p2p_runtime_warning_retained else "ok"
        if app_probe_ok
        else "blocked",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "isaaclab_live_gate_probe",
        "scope": "IsaacLab AppLauncher(headless=True) sentinel diagnostics only; no motion replay, no PPO, no training",
        "checks": checks,
        "current_blocker": blocker,
        "host": {
            "nvidia_smi_return_code": nvidia_smi_rc,
            "nvidia_smi_csv": nvidia_smi_out,
            "vulkaninfo_path": vulkaninfo_path,
            "max_user_watches": read_int("/proc/sys/fs/inotify/max_user_watches"),
            "max_user_instances": read_int("/proc/sys/fs/inotify/max_user_instances"),
        },
        "probes": probes,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This is a live Kit gate diagnostic. Passing it would only permit the next official replay/smoke "
                "stage; it would not prove PPO training, DAgger rollouts, Fig. 5/Fig. 6 results, or real robot behavior."
            ),
            "next_action": (
                "If status is ok_with_runtime_warning, proceed only to official replay preflight/smoke while retaining "
                "the CUDA P2P/IOMMU warning as a runtime risk; do not start PPO or closed-loop paper experiments."
            ),
        },
        "outputs": {
            "json": str(OUT / "isaaclab_live_gate_probe.json"),
            "tsv": str(OUT / "isaaclab_live_gate_probe.tsv"),
            "log_dir": str(LOG_DIR),
        },
    }
    (OUT / "isaaclab_live_gate_probe.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(OUT / "isaaclab_live_gate_probe.tsv", probes)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "current_blocker": blocker,
                "app_launcher_reached_success_sentinel": app_probe_ok,
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if not checks["package_import_probe_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
