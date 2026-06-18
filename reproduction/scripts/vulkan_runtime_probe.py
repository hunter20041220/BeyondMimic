#!/usr/bin/env python3
"""Audit Vulkan loader/ICD availability without launching Isaac Sim Kit."""

from __future__ import annotations

import csv
import ctypes
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/setup/vulkan_runtime_probe"
ISAACSIM = ROOT / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim"
GPU_FOUNDATION_DEPS = (
    ISAACSIM / "extscache/omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
PROJECT_EGL_ICD = OUT / "nvidia_egl_icd.json"
VK_SUCCESS = 0


class VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.POINTER(VkApplicationInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def command_output(args: list[str], env: dict[str, str] | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            args,
            cwd=ROOT,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)
    return proc.returncode, proc.stdout.strip()


def vk_make_version(major: int, minor: int, patch: int) -> int:
    return (major << 22) | (minor << 12) | patch


def decode_vk_version(version: int | None) -> str | None:
    if version is None:
        return None
    major = version >> 22
    minor = (version >> 12) & 0x3FF
    patch = version & 0xFFF
    return f"{major}.{minor}.{patch}"


def probe_loader(name: str, library: str, env_updates: dict[str, str] | None = None) -> dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["TMPDIR"] = str(ROOT / "tmp")
    if env_updates:
        env.update(env_updates)

    script = f"""
import ctypes, json, os
VK_SUCCESS = 0
class VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]
class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.POINTER(VkApplicationInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]
def make_version(major, minor, patch):
    return (major << 22) | (minor << 12) | patch
out = {{"library": {library!r}}}
try:
    lib = ctypes.CDLL({library!r})
    out["load_ok"] = True
except OSError as exc:
    out["load_ok"] = False
    out["load_error"] = str(exc)
    print(json.dumps(out, sort_keys=True))
    raise SystemExit(0)
version = ctypes.c_uint32(0)
try:
    enum_version = lib.vkEnumerateInstanceVersion
    enum_version.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    enum_version.restype = ctypes.c_int32
    rc = int(enum_version(ctypes.byref(version)))
    out["enumerate_instance_version_rc"] = rc
    out["instance_version_raw"] = int(version.value)
except AttributeError as exc:
    out["enumerate_instance_version_error"] = str(exc)
app = VkApplicationInfo()
app.sType = 0
app.pApplicationName = b"BeyondMimicVulkanProbe"
app.applicationVersion = make_version(0, 0, 1)
app.pEngineName = b"NoEngine"
app.engineVersion = make_version(0, 0, 1)
app.apiVersion = make_version(1, 1, 0)
create = VkInstanceCreateInfo()
create.sType = 1
create.pApplicationInfo = ctypes.pointer(app)
instance = ctypes.c_void_p()
try:
    vk_create = lib.vkCreateInstance
    vk_create.argtypes = [ctypes.POINTER(VkInstanceCreateInfo), ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    vk_create.restype = ctypes.c_int32
    rc = int(vk_create(ctypes.byref(create), None, ctypes.byref(instance)))
    out["vk_create_instance_rc"] = rc
    out["vk_create_instance_success"] = rc == VK_SUCCESS
    if rc == VK_SUCCESS:
        destroy = lib.vkDestroyInstance
        destroy.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        destroy(instance, None)
except Exception as exc:
    out["vk_create_instance_error"] = repr(exc)
print(json.dumps(out, sort_keys=True))
"""
    rc, stdout = command_output(["python3", "-c", script], env=env)
    parsed: dict[str, Any] = {}
    try:
        parsed = json.loads(stdout.splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        parsed = {"parse_error": True, "stdout": stdout}
    parsed.update(
        {
            "name": name,
            "returncode": rc,
            "stdout_tail": stdout[-3000:],
            "env_updates": env_updates or {},
        }
    )
    return parsed


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "library",
        "returncode",
        "load_ok",
        "enumerate_instance_version_rc",
        "instance_version",
        "vk_create_instance_rc",
        "vk_create_instance_success",
        "load_error",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "name": row.get("name"),
                    "library": row.get("library"),
                    "returncode": row.get("returncode"),
                    "load_ok": row.get("load_ok"),
                    "enumerate_instance_version_rc": row.get("enumerate_instance_version_rc"),
                    "instance_version": decode_vk_version(row.get("instance_version_raw")),
                    "vk_create_instance_rc": row.get("vk_create_instance_rc"),
                    "vk_create_instance_success": row.get("vk_create_instance_success"),
                    "load_error": row.get("load_error"),
                }
            )
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    PROJECT_EGL_ICD.write_text(
        json.dumps(
            {
                "file_format_version": "1.0.1",
                "ICD": {"library_path": "libEGL_nvidia.so.0", "api_version": "1.4.303"},
            },
            indent=4,
        ),
        encoding="utf-8",
    )
    icd_paths = [Path("/etc/vulkan/icd.d/nvidia_icd.json"), Path("/usr/share/vulkan/icd.d/nvidia_icd.json")]
    existing_icds = [path for path in icd_paths if path.is_file()]
    bundled_loader = GPU_FOUNDATION_DEPS / "libvulkan.so.1"
    system_env = {"VK_ICD_FILENAMES": str(existing_icds[0])} if existing_icds else {}
    bundled_env = {
        "VK_ICD_FILENAMES": str(existing_icds[0]) if existing_icds else "",
        "LD_LIBRARY_PATH": f"{GPU_FOUNDATION_DEPS}:{os.environ.get('LD_LIBRARY_PATH', '')}",
    }
    bundled_egl_env = {
        "VK_ICD_FILENAMES": str(PROJECT_EGL_ICD),
        "LD_LIBRARY_PATH": f"{GPU_FOUNDATION_DEPS}:{os.environ.get('LD_LIBRARY_PATH', '')}",
    }

    probes = [
        probe_loader("system_default_libvulkan", "libvulkan.so.1"),
        probe_loader("system_libvulkan_with_nvidia_icd", "libvulkan.so.1", system_env),
        probe_loader("isaac_bundled_gpu_foundation_libvulkan", str(bundled_loader), bundled_env),
        probe_loader("isaac_bundled_loader_with_project_egl_icd", str(bundled_loader), bundled_egl_env),
    ]
    nvidia_smi_rc, nvidia_smi_out = command_output(["nvidia-smi", "-L"])
    checks = {
        "nvidia_icd_json_exists": bool(existing_icds),
        "nvidia_icd_mentions_libglx_nvidia": any("libGLX_nvidia.so.0" in read_text(path) for path in existing_icds),
        "libglx_nvidia_resolves": command_output(["bash", "-lc", "ldconfig -p | grep -q libGLX_nvidia.so.0"])[0] == 0,
        "system_loader_create_instance_ok": any(
            row.get("name", "").startswith("system_") and row.get("vk_create_instance_success") for row in probes
        ),
        "isaac_bundled_loader_create_instance_ok": any(
            row.get("name", "").startswith("isaac_bundled") and row.get("vk_create_instance_success")
            for row in probes
        ),
        "project_egl_icd_written": PROJECT_EGL_ICD.is_file(),
        "does_not_launch_kit_or_training": True,
        "does_not_claim_isaaclab_gate_passed": True,
    }
    status = "ok" if checks["system_loader_create_instance_ok"] or checks["isaac_bundled_loader_create_instance_ok"] else "blocked"
    summary: dict[str, Any] = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "vulkan_runtime_probe",
        "scope": "ctypes Vulkan loader/ICD probe only; does not launch Isaac Sim Kit or any training",
        "icd_files": [{"path": str(path), "text": read_text(path)} for path in existing_icds],
        "bundled_loader": str(bundled_loader),
        "project_egl_icd": str(PROJECT_EGL_ICD),
        "nvidia_smi": {"returncode": nvidia_smi_rc, "stdout": nvidia_smi_out},
        "probes": probes,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "A minimal Vulkan loader probe is only a prerequisite diagnostic. IsaacLab live AppLauncher, "
                "official replay, PPO training, closed-loop diffusion evaluation, and real robot validation remain "
                "separate gates."
            ),
        },
        "outputs": {
            "json": str(OUT / "vulkan_runtime_probe.json"),
            "tsv": str(OUT / "vulkan_runtime_probe.tsv"),
        },
    }
    (OUT / "vulkan_runtime_probe.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(OUT / "vulkan_runtime_probe.tsv", probes)
    print(
        json.dumps(
            {
                "status": status,
                "system_loader_create_instance_ok": checks["system_loader_create_instance_ok"],
                "isaac_bundled_loader_create_instance_ok": checks["isaac_bundled_loader_create_instance_ok"],
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if status != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
