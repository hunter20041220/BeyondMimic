#!/usr/bin/env python3
"""Audit local Kit settings tried for the IsaacLab gpu.foundation P2P gate."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/setup/isaaclab_gpu_foundation_settings_audit"
LIVE = ROOT / "res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json"


def run_rg(pattern: str) -> tuple[int, str]:
    args = [
        "rg",
        "-n",
        pattern,
        "envs/bm_tracking/lib/python3.10/site-packages/isaacsim",
        "reproduction/third_party/official/IsaacLab-v2.1.0",
        "-g",
        "*.kit",
        "-g",
        "*.toml",
        "-g",
        "*.json",
        "-g",
        "*.py",
    ]
    proc = subprocess.run(args, cwd=ROOT, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["candidate", "attempted", "result", "evidence"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    live = load_json(LIVE)
    probes = {row["name"]: row for row in live.get("probes", [])}
    rg_rc, rg_out = run_rg("gpu.foundation|cudainterop|IOMMU|p2p|P2P|renderer.multiGpu|multi_gpu|active_gpu|physics_gpu")
    (OUT / "settings_surface_rg.txt").write_text(rg_out, encoding="utf-8")

    rows = [
        {
            "candidate": "project EGL ICD",
            "attempted": True,
            "result": "moves past Vulkan incompatible-driver error",
            "evidence": "project_egl_icd_removes_vulkan_error=true",
        },
        {
            "candidate": "renderer single-GPU kit args",
            "attempted": True,
            "result": "limits active GPU to GPU 6 but does not stop IOMMU P2P validation",
            "evidence": "app_launcher_project_egl_icd_single_gpu_renderer",
        },
        {
            "candidate": "SimulationApp multi_gpu=False",
            "attempted": True,
            "result": "limits active GPU to GPU 6 but does not stop IOMMU P2P validation",
            "evidence": "app_launcher_project_egl_icd_simapp_multi_gpu_false",
        },
        {
            "candidate": "CUDA_VISIBLE_DEVICES=6",
            "attempted": True,
            "result": "not viable because gpu.foundation reports CUDA bad state / no device could be created",
            "evidence": "app_launcher_project_egl_icd_cuda_visible_devices_6_single_gpu_renderer",
        },
        {
            "candidate": "device=cpu",
            "attempted": True,
            "result": "still loads gpu.foundation and triggers IOMMU P2P validation",
            "evidence": "app_launcher_project_egl_icd_cpu_device_single_gpu_renderer",
        },
    ]
    checks = {
        "settings_surface_search_ran": rg_rc in {0, 1},
        "project_egl_icd_removes_vulkan_error": live["checks"].get("project_egl_icd_removes_vulkan_error") is True,
        "single_gpu_renderer_limits_active_gpu": live["checks"].get("single_gpu_renderer_limits_active_gpu") is True,
        "cuda_visible_devices_single_gpu_not_viable": live["checks"].get("cuda_visible_devices_single_gpu_not_viable")
        is True,
        "simapp_multi_gpu_false_attempt_recorded": "app_launcher_project_egl_icd_simapp_multi_gpu_false" in probes,
        "cpu_device_attempt_recorded": "app_launcher_project_egl_icd_cpu_device_single_gpu_renderer" in probes,
        "app_launcher_still_blocked": live.get("status") == "blocked",
        "does_not_launch_kit_or_training": True,
        "does_not_claim_gate_passed": True,
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "isaaclab_gpu_foundation_settings_audit",
        "scope": "static/settings-surface summary of local gpu.foundation/cudainterop gate repair attempts",
        "live_gate_json": str(LIVE),
        "settings_surface_rg": str(OUT / "settings_surface_rg.txt"),
        "rows": rows,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The current local settings attempts do not produce a clean AppLauncher close sentinel. Official replay, "
                "tracking smoke, PPO, and closed-loop diffusion remain blocked."
            ),
        },
        "outputs": {
            "json": str(OUT / "isaaclab_gpu_foundation_settings_audit.json"),
            "tsv": str(OUT / "isaaclab_gpu_foundation_settings_audit.tsv"),
            "settings_surface_rg": str(OUT / "settings_surface_rg.txt"),
        },
    }
    (OUT / "isaaclab_gpu_foundation_settings_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_tsv(OUT / "isaaclab_gpu_foundation_settings_audit.tsv", rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"]}, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
