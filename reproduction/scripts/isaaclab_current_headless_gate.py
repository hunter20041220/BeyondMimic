#!/usr/bin/env python3
"""Run the current IsaacLab headless AppLauncher gate on the mainline GPU policy."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/setup/isaaclab_current_headless_gate"
LOG_DIR = ROOT / "logs/setup/isaaclab_current_headless_gate"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
CANDIDATE_GPUS = [4, 7]
SELECTED_GPU = int(os.environ.get("BM_HEADLESS_GATE_GPU", "4"))
MIN_FREE_MB = 20_000
MAX_BUSY_UTIL = 50
TIMEOUT_SECONDS = int(os.environ.get("BM_HEADLESS_GATE_TIMEOUT", "240"))
WANGJC_PATH_MARKER = "/mnt/infini-data/test/wangjc/"


APP_PROBE = r"""
import json
import os
import sys

selected_gpu = os.environ.get("BM_SELECTED_PHYSICAL_GPU", "4")
print("BM_SENTINEL:current_gate:before_import", flush=True)
from isaaclab.app import AppLauncher

print("BM_SENTINEL:current_gate:before_app", flush=True)
launcher = AppLauncher(
    headless=True,
    enable_cameras=False,
    device=os.environ.get("BM_ISAACLAB_DEVICE", "cuda:4"),
    multi_gpu=False,
    fast_shutdown=True,
    kit_args=(
        "--/renderer/multiGpu/enabled=false "
        "--/renderer/multiGpu/autoEnable=false "
        "--/renderer/multiGpu/maxGpuCount=1 "
        f"--/renderer/activeGpu={selected_gpu} "
        f"--/physics/cudaDevice={selected_gpu}"
    ),
)
print("BM_SENTINEL:current_gate:after_app", flush=True)
app = launcher.app
payload = {
    "is_running": bool(app.is_running()),
    "device": os.environ.get("BM_ISAACLAB_DEVICE", "cuda:0"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "multi_gpu": False,
    "headless": True,
}
print("BM_SENTINEL:current_gate:payload:" + json.dumps(payload, sort_keys=True), flush=True)
app.close()
print("BM_SENTINEL:current_gate:after_close", flush=True)
"""


def run(args: list[str], env: dict[str, str] | None = None, timeout: int | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout


def query_gpus() -> list[dict[str, Any]]:
    rc, out = run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
            "-i",
            ",".join(str(gpu) for gpu in CANDIDATE_GPUS),
        ],
        timeout=30,
    )
    if rc != 0:
        return [{"error": out.strip()}]
    rows = []
    for raw in csv.reader(out.strip().splitlines()):
        if len(raw) < 7:
            continue
        index, uuid, name, mem_used, mem_total, util, power = [item.strip() for item in raw[:7]]
        mem_used_i = int(float(mem_used))
        mem_total_i = int(float(mem_total))
        rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_used_mb": mem_used_i,
                "memory_total_mb": mem_total_i,
                "memory_free_mb": mem_total_i - mem_used_i,
                "utilization_gpu_percent": int(float(util)),
                "power_draw_w": float(power),
            }
        )
    return rows


def read_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace")
    except OSError:
        return ""


def query_compute_processes() -> list[dict[str, Any]]:
    uuid_to_index = {row.get("uuid"): row.get("index") for row in query_gpus() if row.get("uuid")}
    rc, out = run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
            "-i",
            ",".join(str(gpu) for gpu in CANDIDATE_GPUS),
        ],
        timeout=30,
    )
    if rc != 0:
        return [{"error": out.strip()}]
    rows = []
    for raw in csv.reader(out.strip().splitlines()):
        if len(raw) < 4:
            continue
        pid = int(raw[1].strip())
        cmdline = read_cmdline(pid)
        rows.append(
            {
                "gpu_uuid": raw[0].strip(),
                "gpu_index": uuid_to_index.get(raw[0].strip()),
                "pid": pid,
                "process_name": raw[2].strip(),
                "cmdline": cmdline,
                "cmdline_contains_allowed_marker": WANGJC_PATH_MARKER in cmdline or WANGJC_PATH_MARKER in raw[2],
                "used_memory_mb": int(float(raw[3].strip())),
            }
        )
    return rows


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "BM_SELECTED_PHYSICAL_GPU": str(SELECTED_GPU),
            "BM_ISAACLAB_DEVICE": f"cuda:{SELECTED_GPU}",
            "VK_ICD_FILENAMES": str(PROJECT_EGL_ICD),
            "LD_LIBRARY_PATH": f"{GPU_FOUNDATION_DEPS}:{env.get('LD_LIBRARY_PATH', '')}",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
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
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def classify_output(text: str, timed_out: bool) -> dict[str, bool]:
    lowered = text.lower()
    return {
        "timed_out": timed_out,
        "sentinel_before_import": "BM_SENTINEL:current_gate:before_import" in text,
        "sentinel_before_app": "BM_SENTINEL:current_gate:before_app" in text,
        "sentinel_after_app": "BM_SENTINEL:current_gate:after_app" in text,
        "sentinel_payload": "BM_SENTINEL:current_gate:payload:" in text,
        "sentinel_after_close": "BM_SENTINEL:current_gate:after_close" in text,
        "vulkan_incompatible_driver": "error_incompatible_driver" in lowered
        or "vkcreateinstance failed" in lowered,
        "gpu_foundation_create_instance_failed": "carb::graphics::createinstance failed" in lowered,
        "gpu_foundation_no_device_created": "no device could be created" in lowered,
        "active_gpu_incompatible": "activegpu" in lowered and "not compatible" in lowered,
        "inotify_errno28": "errno=28" in lowered or "failed to create change watch" in lowered,
        "cuda_p2p_iommu_warning": "peer access is already enabled" in lowered
        or "p2pbandwidthlatencytest" in lowered
        or "iommu is enabled" in lowered,
        "traceback": "traceback (most recent call last)" in lowered,
    }


def extract_payload(text: str) -> dict[str, Any]:
    prefix = "BM_SENTINEL:current_gate:payload:"
    for line in text.splitlines():
        if line.startswith(prefix):
            try:
                return json.loads(line.split(prefix, 1)[1])
            except json.JSONDecodeError:
                return {}
    return {}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"isaaclab_current_headless_gate_{timestamp}.log"
    gpu_before = query_gpus()
    processes_before = query_compute_processes()
    selected_row = next((row for row in gpu_before if row.get("index") == SELECTED_GPU), {})
    selected_processes = [proc for proc in processes_before if proc.get("gpu_index") == SELECTED_GPU]
    resource_ready = (
        selected_row.get("memory_free_mb", 0) >= MIN_FREE_MB
        and selected_row.get("utilization_gpu_percent", 100) <= MAX_BUSY_UTIL
        and not selected_processes
    )
    input_checks = {
        "tracking_python_exists": TRACKING_PY.is_file(),
        "selected_gpu_in_candidate_set": SELECTED_GPU in CANDIDATE_GPUS,
        "selected_gpu_resource_ready": resource_ready,
        "project_egl_icd_exists": PROJECT_EGL_ICD.is_file(),
        "gpu_foundation_deps_exists": GPU_FOUNDATION_DEPS.is_dir(),
        "no_training_started": True,
    }

    attempted = False
    returncode = None
    output = ""
    duration_seconds = 0.0
    timed_out = False
    if all(input_checks.values()):
        attempted = True
        start = time.time()
        try:
            returncode, output = run([str(TRACKING_PY), "-c", APP_PROBE], env=base_env(), timeout=TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = 124
            output = exc.stdout if isinstance(exc.stdout, str) else ""
        duration_seconds = round(time.time() - start, 3)
        log_path.write_text(output, encoding="utf-8", errors="ignore")

    markers = classify_output(output, timed_out)
    payload = extract_payload(output)
    fatal = (
        markers["timed_out"]
        or markers["vulkan_incompatible_driver"]
        or markers["gpu_foundation_create_instance_failed"]
        or markers["gpu_foundation_no_device_created"]
        or markers["active_gpu_incompatible"]
        or markers["inotify_errno28"]
        or markers["traceback"]
    )
    gate_ok = (
        attempted
        and returncode == 0
        and markers["sentinel_after_app"]
        and markers["sentinel_payload"]
        and bool(payload.get("is_running"))
        and not fatal
    )
    status = "ok" if gate_ok else "ok_with_resource_unavailable_before_gate" if not all(input_checks.values()) else "blocked"
    summary = {
        "status": status,
        "experiment_type": "isaaclab_current_headless_gate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Current AppLauncher(headless=True) sentinel gate on fixed mainline GPU policy; no replay, no PPO, no training.",
        "config": {
            "candidate_physical_gpus": CANDIDATE_GPUS,
            "selected_physical_gpu": SELECTED_GPU,
            "cuda_visible_devices": "",
            "device": f"cuda:{SELECTED_GPU}",
            "timeout_seconds": TIMEOUT_SECONDS,
            "min_free_mb_required": MIN_FREE_MB,
            "max_busy_util_percent": MAX_BUSY_UTIL,
        },
        "gpu_preflight": {
            "before": gpu_before,
            "compute_processes_before": [
                {k: ("<redacted>" if k == "cmdline" and v else v) for k, v in proc.items()}
                for proc in processes_before
            ],
            "selected_gpu_resource_ready": resource_ready,
        },
        "input_checks": input_checks,
        "run": {
            "attempted": attempted,
            "returncode": returncode,
            "duration_seconds": duration_seconds,
            "log": str(log_path),
            "markers": markers,
            "payload": payload,
        },
        "checks": {
            "app_launcher_headless_success_sentinel": gate_ok,
            "sentinel_after_app": markers["sentinel_after_app"],
            "sentinel_payload": markers["sentinel_payload"],
            "payload_is_running": bool(payload.get("is_running")),
            "no_fatal_runtime_error": not fatal,
            "no_training_started": True,
            "does_not_claim_tracking_reproduction_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "official_replay_complete": False,
            "paper_level_tracking_complete": False,
            "why_not_complete": (
                "This only verifies current IsaacLab/Isaac Sim AppLauncher(headless=True) startup. It permits replay/"
                "task smoke work, but it is not official replay, PPO training, DAgger data, Fig.5/Fig.6, or robot evidence."
            ),
        },
        "outputs": {
            "json": str(OUT / "isaaclab_current_headless_gate.json"),
            "log": str(log_path),
        },
    }
    (OUT / "isaaclab_current_headless_gate.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": status, "gate_ok": gate_ok, "json": summary["outputs"]["json"]}, sort_keys=True))
    if status == "blocked":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
