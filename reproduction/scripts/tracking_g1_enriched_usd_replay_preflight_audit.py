#!/usr/bin/env python3
"""Run and audit a bounded replay preflight for the enriched G1 USD scaffold."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/g1_enriched_usd_replay_preflight"
LOG_DIR = ROOT / "logs/tracking_g1_enriched_usd_replay_preflight"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
ENTRYPOINT = ROOT / "reproduction/generated/whole_body_tracking_local/replay_npz_enriched_usd_preflight.py"
ENRICHED_USD = (
    ROOT
    / "res/tracking/g1_resource_adjusted_enriched_usd/"
    "g1_resource_adjusted_29dof_enriched_scaffold.usda"
)
ENRICHED_USD_AUDIT = (
    ROOT
    / "res/tracking/g1_resource_adjusted_enriched_usd/"
    "tracking_g1_resource_adjusted_enriched_usd_probe.json"
)
MOTION_FIXTURE = ROOT / "reproduction/data/tracking_motion_npz_fixtures/walk1_subject1_frames_1_180_debug_motion.npz"
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
TIMEOUT_SECONDS = 210


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def classify(text: str) -> dict[str, bool]:
    lowered = text.lower()
    return {
        "before_app": "bm_sentinel:before_app" in lowered,
        "after_app": "bm_sentinel:after_app" in lowered,
        "motion_contract": "bm_sentinel:motion_contract" in lowered,
        "sim_created": "bm_sentinel:sim_created" in lowered,
        "scene_created": "bm_sentinel:scene_created" in lowered,
        "sim_reset": "bm_sentinel:sim_reset" in lowered,
        "robot_contract": "bm_sentinel:robot_contract" in lowered,
        "step_reached": "bm_sentinel:step=" in lowered,
        "success": "bm_sentinel:enriched_usd_replay_preflight_success" in lowered,
        "after_close": "bm_sentinel:after_close" in lowered,
        "traceback": "traceback (most recent call last)" in lowered,
        "vulkan_device_lost": "vkresult: error_device_lost" in lowered or "device lost" in lowered,
        "no_physics_articulation": "failed to find articulation" in lowered
        or "articulation root" in lowered
        or "no bodies found in articulation" in lowered,
        "usd_reference_or_spawn_error": "failed to open layer" in lowered
        or "could not open asset" in lowered
        or "unresolved reference" in lowered,
        "eula_prompt": "do you accept the eula" in lowered or "omniverse license agreement" in lowered,
        "argparse_enable_cameras_false_error": "unrecognized arguments: false" in lowered,
        "cuda_invalid_device_ordinal": "invalid device ordinal" in lowered
        or "requested cuda device ordinal" in lowered,
        "timeout": "bm_TIMEOUT" in text,
    }


def summarize_blocker(markers: dict[str, bool], returncode: int) -> str:
    if returncode == 0 and markers["success"]:
        return "none_resource_adjusted_preflight_passed"
    if returncode == 124 and markers["success"]:
        return "kit_shutdown_timeout_after_resource_adjusted_step_gate_passed"
    if markers["vulkan_device_lost"]:
        return "kit_vulkan_device_lost_during_enriched_usd_replay_preflight"
    if markers["no_physics_articulation"]:
        return "enriched_usd_spawned_without_valid_physics_articulation"
    if markers["usd_reference_or_spawn_error"]:
        return "enriched_usd_reference_or_spawn_error"
    if markers["eula_prompt"]:
        return "isaacsim_eula_prompt_not_suppressed"
    if markers["argparse_enable_cameras_false_error"]:
        return "argparse_enable_cameras_false_error"
    if markers["cuda_invalid_device_ordinal"]:
        return "cuda_visible_devices_kit_ordinal_mismatch"
    if markers["scene_created"] and not markers["sim_reset"]:
        return "isaaclab_scene_created_but_sim_reset_failed"
    if markers["after_app"] and not markers["sim_created"]:
        return "kit_started_but_sim_context_or_scene_failed"
    if returncode == 124 or markers["timeout"]:
        return "bounded_preflight_timeout"
    if markers["traceback"]:
        return "python_traceback_unclassified"
    return "preflight_failed_before_or_without_known_marker"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_path = LOG_DIR / "enriched_usd_replay_preflight.log"
    env = os.environ.copy()
    env.update(
        {
            "VK_ICD_FILENAMES": str(PROJECT_EGL_ICD),
            "LD_LIBRARY_PATH": f"{GPU_FOUNDATION_DEPS}:{env.get('LD_LIBRARY_PATH', '')}",
            "PYTHONUNBUFFERED": "1",
            "ISAAC_PATH": str(ROOT / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim"),
            "OMNI_USER_DIR": str(ROOT / "cache/omni/user"),
            "OMNI_CACHE_DIR": str(ROOT / "cache/omni/cache"),
            "OMNI_DATA_DIR": str(ROOT / "cache/omni/data"),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
        }
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    command = [
        "timeout",
        str(TIMEOUT_SECONDS),
        str(TRACKING_PY),
        str(ENTRYPOINT),
        "--motion_file",
        str(MOTION_FIXTURE),
        "--usd_path",
        str(ENRICHED_USD),
        "--max_steps",
        "4",
        "--headless",
        "--device",
        "cuda:6",
    ]

    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=TIMEOUT_SECONDS + 30,
    )
    output = proc.stdout
    if proc.returncode == 124:
        output += "\nBM_TIMEOUT:timeout_return_code_124\n"
    log_path.write_text(output, encoding="utf-8")

    enriched_audit = load_json(ENRICHED_USD_AUDIT)
    markers = classify(output)
    latest_blocker = summarize_blocker(markers, proc.returncode)
    step_gate_passed = markers["success"] and markers["robot_contract"] and markers["step_reached"]
    success = proc.returncode == 0 and step_gate_passed
    shutdown_timeout_after_success = proc.returncode == 124 and step_gate_passed
    checks = {
        "entrypoint_exists": ENTRYPOINT.is_file(),
        "tracking_python_exists": TRACKING_PY.is_file(),
        "motion_fixture_exists": MOTION_FIXTURE.is_file(),
        "enriched_usd_exists": ENRICHED_USD.is_file(),
        "project_egl_icd_exists": PROJECT_EGL_ICD.is_file(),
        "enriched_usd_readback_ok": enriched_audit.get("status") == "ok_with_resource_adjusted_enriched_usd_scaffold",
        "bounded_command_executed": True,
        "kit_reached_after_app": markers["after_app"],
        "sim_context_reached": markers["sim_created"],
        "scene_creation_reached": markers["scene_created"],
        "robot_contract_reached": markers["robot_contract"],
        "render_step_reached": markers["step_reached"],
        "resource_adjusted_step_gate_passed": step_gate_passed,
        "resource_adjusted_preflight_clean_exit": success,
        "kit_shutdown_timeout_after_step_gate": shutdown_timeout_after_success,
        "does_not_claim_official_replay_success": True,
        "does_not_claim_paper_level_rollout": True,
        "does_not_start_training": True,
    }
    if success:
        status = "ok_resource_adjusted_preflight_passed"
    elif shutdown_timeout_after_success:
        status = "ok_with_resource_adjusted_step_gate_passed_shutdown_timeout"
    else:
        status = "ok_with_resource_adjusted_replay_blocker"
    summary = {
        "status": status,
        "experiment_type": "tracking_resource_adjusted_replay_preflight",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Bounded IsaacLab replay gate using the local resource-adjusted enriched G1 USD scaffold and a "
            "debug motion fixture. This explicitly does not validate official csv_to_npz output, official replay, "
            "PPO training, DAgger, or paper-level closed-loop BeyondMimic performance."
        ),
        "command": command,
        "timeout_seconds": TIMEOUT_SECONDS,
        "returncode": proc.returncode,
        "latest_blocker": latest_blocker,
        "markers": markers,
        "checks": checks,
        "inputs": {
            "entrypoint": str(ENTRYPOINT),
            "motion_fixture": str(MOTION_FIXTURE),
            "enriched_usd": str(ENRICHED_USD),
            "enriched_usd_audit": str(ENRICHED_USD_AUDIT),
        },
        "outputs": {
            "json": str(OUT / "tracking_g1_enriched_usd_replay_preflight_audit.json"),
            "log": str(log_path),
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_tracking_replay_complete": False,
            "official_beyondmimic_replay_complete": False,
            "why_not_complete": (
                "This gate uses a generated resource-adjusted scaffold and a debug fixture. Even if it passes, it "
                "does not replace official URDF conversion, official motion replay, tracking evaluation metrics, "
                "teacher rollout data, or Fig. 5/Fig. 6 closed-loop results."
            ),
        },
    }
    (OUT / "tracking_g1_enriched_usd_replay_preflight_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": status, "returncode": proc.returncode, "latest_blocker": latest_blocker}, sort_keys=True))


if __name__ == "__main__":
    main()
