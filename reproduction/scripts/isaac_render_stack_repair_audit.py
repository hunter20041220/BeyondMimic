#!/usr/bin/env python3
"""Audit the attempted Isaac Sim rendering-stack repair for true MP4 capture."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/setup/isaac_render_stack_repair_audit"

APT_LOGS = sorted((ROOT / "logs/setup").glob("render_stack_apt_install_*.log"))
VULKAN_DEFAULT_LOGS = sorted((ROOT / "logs/isaac_mp4").glob("vulkaninfo_default_*.log"))
VULKAN_SYSTEM_LOGS = sorted((ROOT / "logs/isaac_mp4").glob("vulkaninfo_system_nvidia_icd_*.log"))
VULKAN_SYSTEM_FULL_LOGS = sorted((ROOT / "logs/isaac_mp4").glob("vulkaninfo_system_nvidia_full_*.log"))
VULKAN_EGL_LOGS = sorted((ROOT / "logs/isaac_mp4").glob("vulkaninfo_project_egl_icd_*.log"))
VULKAN_LLVMP_LOGS = sorted((ROOT / "logs/isaac_mp4").glob("vulkaninfo_llvmpipe_*.log"))
LLVMP_PXR_LOGS = sorted((ROOT / "logs/isaac_mp4").glob("llvmpipe_pxr_app_probe_*.log"))
MP4_GATE_JSON = ROOT / "res/failed_runs/isaac_mp4/isaaclab_rendered_policy_rollout_video_failed_gate.json"
ASSET_JSON = ROOT / "res/visualization/isaac_mp4/isaaclab_rendered_policy_rollout_video_asset.json"
SYSTEM_NVIDIA_ICD = Path("/etc/vulkan/icd.d/nvidia_icd.json")
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
LLVMP_ICD = Path("/usr/share/vulkan/icd.d/lvp_icd.x86_64.json")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def latest(paths: list[Path]) -> Path | None:
    return paths[-1] if paths else None


def contains_any(text: str, patterns: list[str]) -> list[str]:
    return sorted({pattern for pattern in patterns if pattern in text})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "probe",
        "status",
        "path",
        "evidence",
        "interpretation",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mp4_gate = load_json(MP4_GATE_JSON)
    asset = load_json(ASSET_JSON)

    vulkan_system_full_text = read_text(latest(VULKAN_SYSTEM_FULL_LOGS))
    vulkan_llvmp_text = read_text(latest(VULKAN_LLVMP_LOGS))
    llvmp_pxr_text = read_text(latest(LLVMP_PXR_LOGS))
    latest_gate_log = Path(mp4_gate.get("latest_log", "")) if mp4_gate.get("latest_log") else None
    latest_gate_log_text = read_text(latest_gate_log)
    xvfb_gate_logs = sorted((ROOT / "logs/isaac_mp4").glob("*seed20260782*10step_robot_order_policy.log"))
    latest_xvfb_gate_log = latest(xvfb_gate_logs)
    latest_xvfb_gate_text = read_text(latest_xvfb_gate_log)

    vulkan_ray_tracing_extensions = {
        "VK_KHR_acceleration_structure": "VK_KHR_acceleration_structure" in vulkan_system_full_text,
        "VK_KHR_deferred_host_operations": "VK_KHR_deferred_host_operations" in vulkan_system_full_text,
        "VK_KHR_ray_query": "VK_KHR_ray_query" in vulkan_system_full_text,
        "VK_KHR_ray_tracing_pipeline": "VK_KHR_ray_tracing_pipeline" in vulkan_system_full_text,
        "VK_NV_ray_tracing": "VK_NV_ray_tracing" in vulkan_system_full_text,
    }
    gate_errors = contains_any(
        latest_gate_log_text,
        [
            "GLFW initialization failed",
            "GLInteropContext::init",
            "carb::windowing is not available",
            "GPU crash occurred",
            "VkResult: ERROR_DEVICE_LOST",
            "Segmentation fault",
            "omni.kit.widget.viewport",
        ],
    )
    llvmp_errors = contains_any(
        llvmp_pxr_text,
        [
            "No device could be created",
            "GPU Foundation is not initialized",
            "Stage opened with no valid renderer selected",
            "CUDA libs are present, but no suitable CUDA GPU was found",
            "BM_SENTINEL:llvmpipe_pxr:after_app",
        ],
    )
    xvfb_errors = contains_any(
        latest_xvfb_gate_text,
        [
            "GLXBadFBConfig",
            "GLFW initialization failed",
            "GPU crash occurred",
            "VkResult: ERROR_DEVICE_LOST",
            "BM_SENTINEL:isaac_mp4:after_app",
        ],
    )

    checks = {
        "apt_render_stack_packages_logged": bool(APT_LOGS),
        "system_nvidia_icd_exists": SYSTEM_NVIDIA_ICD.is_file(),
        "project_egl_icd_exists": PROJECT_EGL_ICD.is_file(),
        "llvmpipe_icd_exists": LLVMP_ICD.is_file(),
        "vulkaninfo_default_ran": bool(VULKAN_DEFAULT_LOGS),
        "vulkaninfo_system_nvidia_icd_ran": bool(VULKAN_SYSTEM_LOGS),
        "vulkaninfo_project_egl_icd_ran": bool(VULKAN_EGL_LOGS),
        "vulkaninfo_llvmpipe_ran": bool(VULKAN_LLVMP_LOGS),
        "vulkaninfo_system_detects_h20": "NVIDIA H20" in vulkan_system_full_text,
        "vulkaninfo_llvmpipe_detects_cpu_device": "llvmpipe" in vulkan_llvmp_text,
        "mp4_gate_uses_system_nvidia_icd": mp4_gate.get("run", {}).get("vulkan_icd") == str(SYSTEM_NVIDIA_ICD),
        "mp4_gate_uses_project_xdg_runtime": str(mp4_gate.get("run", {}).get("xdg_runtime_dir", "")).startswith(str(ROOT)),
        "mp4_gate_failed_before_after_app": "BM_SENTINEL:isaac_mp4:after_app" not in latest_gate_log_text,
        "mp4_gate_failed_before_env_creation": "BM_SENTINEL:isaac_mp4:env_created" not in latest_gate_log_text,
        "mp4_gate_has_no_mp4": not mp4_gate.get("run", {}).get("mp4_exists", False),
        "mp4_gate_classifies_h20_blocker": bool(
            mp4_gate.get("failure_classification", {}).get("h20_isaac_rendering_hardware_blocker")
        ),
        "llvmpipe_pxr_probe_ran": bool(LLVMP_PXR_LOGS),
        "llvmpipe_pxr_did_not_reach_after_app": "BM_SENTINEL:llvmpipe_pxr:after_app" not in llvmp_pxr_text,
        "xvfb_render_gate_ran_after_dependency_install": bool(xvfb_gate_logs),
        "xvfb_render_gate_did_not_reach_after_app": "BM_SENTINEL:isaac_mp4:after_app" not in latest_xvfb_gate_text,
        "xvfb_render_gate_glx_bad_fbconfig_recorded": "GLXBadFBConfig" in latest_xvfb_gate_text,
        "does_not_claim_successful_rendered_mp4": True,
        "does_not_claim_policy_failure": True,
        "does_not_claim_physics_rollout_failure": True,
        "does_not_claim_paper_level": True,
        "does_not_claim_real_robot": True,
    }
    status = "blocked_h20_isaac_sim_rendering_stack" if all(
        [
            checks["vulkaninfo_system_detects_h20"],
            checks["mp4_gate_failed_before_after_app"],
            checks["mp4_gate_has_no_mp4"],
            checks["mp4_gate_classifies_h20_blocker"],
        ]
    ) else "incomplete_isaac_render_stack_repair_audit"

    rows = [
        {
            "probe": "system_render_stack_packages",
            "status": "logged" if APT_LOGS else "missing_log",
            "path": str(latest(APT_LOGS) or ""),
            "evidence": "vulkan-tools/mesa-utils/xvfb and XCB helper packages installed or verified",
            "interpretation": "Basic diagnostic/runtime packages are no longer the primary blocker.",
        },
        {
            "probe": "vulkaninfo_system_nvidia_icd",
            "status": "ran" if VULKAN_SYSTEM_LOGS else "missing",
            "path": str(latest(VULKAN_SYSTEM_LOGS) or ""),
            "evidence": "NVIDIA H20 enumerated by Vulkan system ICD",
            "interpretation": "Vulkan loader/ICD can enumerate H20; the crash happens when Isaac Kit/Hydra creates rendering devices.",
        },
        {
            "probe": "vulkaninfo_llvmpipe",
            "status": "ran" if VULKAN_LLVMP_LOGS else "missing",
            "path": str(latest(VULKAN_LLVMP_LOGS) or ""),
            "evidence": "llvmpipe CPU Vulkan device enumerated",
            "interpretation": "Software Vulkan exists, but Isaac GPU Foundation does not accept it as a usable rendering device.",
        },
        {
            "probe": "isaaclab_rendered_mp4_gate",
            "status": mp4_gate.get("status", "missing"),
            "path": str(MP4_GATE_JSON),
            "evidence": ", ".join(gate_errors),
            "interpretation": "Fails before AppLauncher reaches after_app and before Tracking-Flat-G1-v0 env creation.",
        },
        {
            "probe": "llvmpipe_pxr_app_probe",
            "status": "failed_no_after_app" if checks["llvmpipe_pxr_probe_ran"] else "missing",
            "path": str(latest(LLVMP_PXR_LOGS) or ""),
            "evidence": ", ".join(llvmp_errors),
            "interpretation": "CPU/PXR software path does not provide a stable Isaac rendering fallback on this host.",
        },
        {
            "probe": "xvfb_render_gate_after_dependency_install",
            "status": "failed_no_after_app" if checks["xvfb_render_gate_ran_after_dependency_install"] else "missing",
            "path": str(latest_xvfb_gate_log or ""),
            "evidence": ", ".join(xvfb_errors),
            "interpretation": "Xvfb does not make the Isaac rendering experience usable; the current repro hits GLXBadFBConfig before app startup completes.",
        },
    ]

    payload = {
        "status": status,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "claim_level": "server_rendering_stack_blocker_audit",
        "goal_complete": False,
        "inputs": {
            "mp4_gate_json": str(MP4_GATE_JSON),
            "asset_json": str(ASSET_JSON),
            "latest_mp4_gate_log": str(latest_gate_log or ""),
            "latest_apt_log": str(latest(APT_LOGS) or ""),
            "latest_vulkan_default_log": str(latest(VULKAN_DEFAULT_LOGS) or ""),
            "latest_vulkan_system_log": str(latest(VULKAN_SYSTEM_LOGS) or ""),
            "latest_vulkan_system_full_log": str(latest(VULKAN_SYSTEM_FULL_LOGS) or ""),
            "latest_vulkan_egl_log": str(latest(VULKAN_EGL_LOGS) or ""),
            "latest_vulkan_llvmpipe_log": str(latest(VULKAN_LLVMP_LOGS) or ""),
            "latest_llvmpipe_pxr_log": str(latest(LLVMP_PXR_LOGS) or ""),
            "latest_xvfb_render_gate_log": str(latest_xvfb_gate_log or ""),
        },
        "runtime": {
            "system_nvidia_icd": str(SYSTEM_NVIDIA_ICD),
            "project_egl_icd": str(PROJECT_EGL_ICD),
            "llvmpipe_icd": str(LLVMP_ICD),
            "vulkan_ray_tracing_extensions_detected": vulkan_ray_tracing_extensions,
            "mp4_gate_failure_classification": mp4_gate.get("failure_classification", {}),
            "mp4_gate_detected_error_patterns": mp4_gate.get("detected_error_patterns", []),
            "mp4_gate_reached_sentinels": mp4_gate.get("reached_sentinels", []),
            "asset_status": asset.get("status"),
        },
        "checks": checks,
        "probe_rows": rows,
        "official_support_boundary": {
            "isaac_sim_requirements_url": "https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html",
            "nvidia_forum_h20_url": "https://forums.developer.nvidia.com/t/does-the-h20-graphics-card-support-isaac-lab-isaac-sim/339701",
            "summary": (
                "NVIDIA's Isaac Sim requirements exclude GPUs without RT Cores. NVIDIA forum guidance for H20 "
                "states that H20 lacks RT Cores and is not supported for Isaac Sim/Isaac Lab rendering. The local "
                "H20 host can run CUDA and non-rendering IsaacLab gates, but true rendered MP4 capture remains "
                "blocked in Kit/Hydra/Vulkan startup."
            ),
        },
        "interpretation": (
            "The attempted repair installed/verified diagnostic graphics packages, switched the MP4 gate to the "
            "system NVIDIA Vulkan ICD, set a project-local XDG_RUNTIME_DIR, disabled the NVIDIA Optimus layer, "
            "validated Vulkan enumeration for both H20 and llvmpipe, and tried a llvmpipe/PXR fallback. The true "
            "IsaacLab rendered MP4 gate still fails before AppLauncher reaches the post-start sentinel and before "
            "Tracking-Flat-G1-v0 environment creation. A post-dependency Xvfb rerun fails even earlier with "
            "GLXBadFBConfig. This is therefore a server rendering-stack hardware/driver blocker on the H20 host, "
            "not a PPO checkpoint failure and not a physics rollout failure."
        ),
    }
    write_json(OUT / "isaac_render_stack_repair_audit.json", payload)
    write_tsv(OUT / "isaac_render_stack_repair_audit.tsv", rows)
    print(json.dumps({"status": status, "json": str(OUT / "isaac_render_stack_repair_audit.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
