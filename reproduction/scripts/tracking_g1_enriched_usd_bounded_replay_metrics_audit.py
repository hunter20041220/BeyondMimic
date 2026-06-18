#!/usr/bin/env python3
"""Run a bounded resource-adjusted replay metrics gate for enriched G1 USD."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/g1_enriched_usd_bounded_replay_metrics"
LOG_DIR = ROOT / "logs/tracking_g1_enriched_usd_bounded_replay_metrics"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
ENTRYPOINT = ROOT / "reproduction/generated/whole_body_tracking_local/replay_npz_enriched_usd_preflight.py"
ENRICHED_USD = (
    ROOT
    / "res/tracking/g1_resource_adjusted_enriched_usd/"
    "g1_resource_adjusted_29dof_enriched_scaffold.usda"
)
ENRICHED_STEP_GATE = (
    ROOT
    / "res/tracking/g1_enriched_usd_replay_preflight/"
    "tracking_g1_enriched_usd_replay_preflight_audit.json"
)
MOTION_FIXTURE = ROOT / "reproduction/data/tracking_motion_npz_fixtures/walk1_subject1_frames_1_180_debug_motion.npz"
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
METRICS_JSON = OUT / "walk1_subject1_64step_resource_adjusted_replay_metrics.json"
TIMEOUT_SECONDS = 240


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
        "step_64": "bm_sentinel:step=64" in lowered,
        "metrics_file": "bm_sentinel:metrics_file=" in lowered,
        "success": "bm_sentinel:enriched_usd_replay_preflight_success" in lowered,
        "explicit_exit_after_success": "bm_sentinel:explicit_exit_after_success" in lowered,
        "traceback": "traceback (most recent call last)" in lowered,
        "vulkan_device_lost": "vkresult: error_device_lost" in lowered or "device lost" in lowered,
        "timeout": "bm_TIMEOUT" in text,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "walk1_subject1_64step_resource_adjusted_replay_metrics.log"
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
        "64",
        "--metrics_file",
        str(METRICS_JSON),
        "--headless",
        "--device",
        "cuda:6",
        "--exit_after_success",
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

    markers = classify(output)
    metrics = load_json(METRICS_JSON)
    step_gate = load_json(ENRICHED_STEP_GATE)
    checks = {
        "entrypoint_exists": ENTRYPOINT.is_file(),
        "motion_fixture_exists": MOTION_FIXTURE.is_file(),
        "enriched_usd_exists": ENRICHED_USD.is_file(),
        "prior_4step_gate_passed": step_gate.get("checks", {}).get("resource_adjusted_step_gate_passed") is True,
        "bounded_command_executed": True,
        "process_returned_zero": proc.returncode == 0,
        "success_sentinel_seen": markers["success"],
        "explicit_exit_after_success": markers["explicit_exit_after_success"],
        "metrics_file_written": METRICS_JSON.is_file() and markers["metrics_file"],
        "step_64_reached": markers["step_64"],
        "robot_contract_reached": markers["robot_contract"],
        "robot_joint_count_29": metrics.get("robot_num_joints") == 29,
        "robot_body_count_40": metrics.get("robot_num_bodies") == 40,
        "executed_steps_64": metrics.get("executed_steps") == 64,
        "motion_total_steps_ge_64": metrics.get("motion_total_steps", 0) >= 64,
        "joint_pos_error_recorded": metrics.get("max_joint_pos_abs_error") is not None,
        "root_state_error_recorded": metrics.get("max_root_pos_abs_error") is not None,
        "uses_resource_adjusted_usd": metrics.get("uses_resource_adjusted_usd") is True,
        "does_not_claim_official_csv_to_npz_output": metrics.get("official_csv_to_npz_output") is False,
        "does_not_claim_paper_level_rollout": metrics.get("paper_level_rollout") is False,
        "does_not_start_training": True,
    }
    passed = all(checks.values())
    status = "ok_resource_adjusted_64step_metrics_gate" if passed else "ok_with_resource_adjusted_metrics_blocker"
    summary = {
        "status": status,
        "experiment_type": "tracking_resource_adjusted_bounded_replay_metrics",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "64-step bounded metrics gate using the generated resource-adjusted enriched G1 USD scaffold and debug "
            "motion fixture. This is not official csv_to_npz/replay output and not paper-level tracking evaluation."
        ),
        "command": command,
        "timeout_seconds": TIMEOUT_SECONDS,
        "returncode": proc.returncode,
        "markers": markers,
        "metrics": metrics,
        "checks": checks,
        "inputs": {
            "entrypoint": str(ENTRYPOINT),
            "motion_fixture": str(MOTION_FIXTURE),
            "enriched_usd": str(ENRICHED_USD),
            "prior_step_gate": str(ENRICHED_STEP_GATE),
        },
        "outputs": {
            "json": str(OUT / "tracking_g1_enriched_usd_bounded_replay_metrics_audit.json"),
            "metrics_json": str(METRICS_JSON),
            "log": str(log_path),
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_tracking_replay_complete": False,
            "official_beyondmimic_replay_complete": False,
            "why_not_complete": (
                "This diagnostic uses a generated resource-adjusted USD and debug fixture; it does not replace "
                "official motion conversion, official tracking replay, evaluation metrics, PPO, DAgger, teacher "
                "rollout data, or Fig. 5/Fig. 6 closed-loop results."
            ),
        },
    }
    (OUT / "tracking_g1_enriched_usd_bounded_replay_metrics_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": status, "returncode": proc.returncode, "executed_steps": metrics.get("executed_steps")}, sort_keys=True))


if __name__ == "__main__":
    main()
