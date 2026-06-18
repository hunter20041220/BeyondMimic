#!/usr/bin/env python3
"""Replay the resource-adjusted official-CSV conversion for all available steps."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/g1_resource_adjusted_csv_full_replay"
LOG_DIR = ROOT / "logs/tracking_g1_resource_adjusted_csv_full_replay"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
ENTRYPOINT = ROOT / "reproduction/generated/whole_body_tracking_local/replay_npz_enriched_usd_preflight.py"
ENRICHED_USD = (
    ROOT
    / "res/tracking/g1_resource_adjusted_enriched_usd/"
    "g1_resource_adjusted_29dof_enriched_scaffold.usda"
)
CSV_CONVERSION_AUDIT = (
    ROOT
    / "res/tracking/g1_resource_adjusted_csv_conversion/"
    "tracking_g1_resource_adjusted_csv_conversion_audit.json"
)
MOTION_NPZ = (
    ROOT
    / "res/tracking/g1_resource_adjusted_csv_conversion/"
    "walk1_subject1_frames_1_180_resource_adjusted_motion.npz"
)
METRICS_JSON = OUT / "walk1_subject1_frames_1_180_resource_adjusted_full_replay_metrics.json"
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
STALL_SECONDS = 900


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def classify(text: str) -> dict[str, Any]:
    lowered = text.lower()
    return {
        "before_app": "bm_sentinel:before_app" in lowered,
        "after_app": "bm_sentinel:after_app" in lowered,
        "motion_contract": "bm_sentinel:motion_contract" in lowered,
        "sim_created": "bm_sentinel:sim_created" in lowered,
        "scene_created": "bm_sentinel:scene_created" in lowered,
        "sim_reset": "bm_sentinel:sim_reset" in lowered,
        "robot_contract": "bm_sentinel:robot_contract" in lowered,
        "step_299": "bm_sentinel:step=299" in lowered,
        "metrics_file": "bm_sentinel:metrics_file=" in lowered,
        "success": "bm_sentinel:enriched_usd_replay_preflight_success" in lowered,
        "explicit_exit_after_success": "bm_sentinel:explicit_exit_after_success" in lowered,
        "traceback": "traceback (most recent call last)" in lowered,
        "vulkan_device_lost": "vkresult: error_device_lost" in lowered or "device lost" in lowered,
        "stall_timeout": "bm_stall_timeout" in lowered,
    }


def env_vars() -> dict[str, str]:
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
    return env


def run_replay(log_path: Path) -> dict[str, Any]:
    command = [
        str(TRACKING_PY),
        str(ENTRYPOINT),
        "--motion_file",
        str(MOTION_NPZ),
        "--usd_path",
        str(ENRICHED_USD),
        "--max_steps",
        "299",
        "--metrics_file",
        str(METRICS_JSON),
        "--headless",
        "--device",
        "cuda:6",
        "--exit_after_success",
    ]
    start = time.time()
    stalled = False
    last_size = -1
    last_change = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env_vars(),
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        while proc.poll() is None:
            time.sleep(10)
            current_size = log_path.stat().st_size if log_path.is_file() else 0
            if current_size != last_size:
                last_size = current_size
                last_change = time.time()
            elif time.time() - last_change > STALL_SECONDS:
                stalled = True
                log_file.write(f"\nBM_STALL_TIMEOUT:no_log_progress_for_{STALL_SECONDS}s\n")
                log_file.flush()
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=60)
                break
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    return {
        "command": command,
        "returncode": proc.returncode,
        "duration_seconds": round(time.time() - start, 3),
        "stalled": stalled,
        "markers": classify(text),
        "log": str(log_path),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if METRICS_JSON.exists():
        METRICS_JSON.unlink()
    log_path = LOG_DIR / "tracking_g1_resource_adjusted_csv_full_replay.log"
    run = run_replay(log_path)
    metrics = load_json(METRICS_JSON)
    conversion = load_json(CSV_CONVERSION_AUDIT)
    checks = {
        "entrypoint_exists": ENTRYPOINT.is_file(),
        "motion_npz_exists": MOTION_NPZ.is_file(),
        "enriched_usd_exists": ENRICHED_USD.is_file(),
        "prior_csv_conversion_passed": conversion.get("status") == "ok_resource_adjusted_csv_conversion",
        "process_returned_zero": run["returncode"] == 0,
        "no_stall_timeout": run["stalled"] is False,
        "app_reached": run["markers"]["after_app"],
        "motion_contract_seen": run["markers"]["motion_contract"],
        "robot_contract_seen": run["markers"]["robot_contract"],
        "step_299_reached": run["markers"]["step_299"],
        "success_sentinel_seen": run["markers"]["success"],
        "explicit_exit_after_success": run["markers"]["explicit_exit_after_success"],
        "metrics_file_written": METRICS_JSON.is_file(),
        "executed_steps_299": metrics.get("executed_steps") == 299,
        "motion_total_steps_299": metrics.get("motion_total_steps") == 299,
        "robot_joint_count_29": metrics.get("robot_num_joints") == 29,
        "robot_body_count_40": metrics.get("robot_num_bodies") == 40,
        "joint_pos_shape_299_29": metrics.get("joint_pos_shape") == [299, 29],
        "body_pos_shape_299_40_3": metrics.get("body_pos_w_shape") == [299, 40, 3],
        "joint_pos_error_recorded": metrics.get("max_joint_pos_abs_error") is not None,
        "root_state_error_recorded": metrics.get("max_root_pos_abs_error") is not None,
        "uses_resource_adjusted_usd": metrics.get("uses_resource_adjusted_usd") is True,
        "does_not_claim_official_csv_to_npz_output": metrics.get("official_csv_to_npz_output") is False,
        "does_not_claim_paper_level_rollout": metrics.get("paper_level_rollout") is False,
        "does_not_start_training": True,
    }
    passed = all(checks.values())
    if passed:
        status = "ok_resource_adjusted_csv_full_replay"
        latest_blocker = "none_resource_adjusted_csv_full_replay_passed"
    elif run["markers"]["vulkan_device_lost"]:
        status = "ok_with_resource_adjusted_csv_full_replay_blocker"
        latest_blocker = "vulkan_device_lost"
    elif run["stalled"]:
        status = "ok_with_resource_adjusted_csv_full_replay_blocker"
        latest_blocker = "stall_timeout"
    elif run["markers"]["traceback"]:
        status = "ok_with_resource_adjusted_csv_full_replay_blocker"
        latest_blocker = "python_traceback"
    else:
        status = "ok_with_resource_adjusted_csv_full_replay_blocker"
        latest_blocker = "failed_checks"
    summary = {
        "status": status,
        "experiment_type": "tracking_resource_adjusted_csv_full_replay",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Full 299-step replay metrics for a resource-adjusted motion.npz converted from the official G1 LAFAN "
            "CSV. This tests the replay surface for an official-source motion segment, but still uses generated "
            "resource-adjusted USD and is not official replay/evaluation or paper-level evidence."
        ),
        "latest_blocker": latest_blocker,
        "run": run,
        "metrics": metrics,
        "checks": checks,
        "inputs": {
            "entrypoint": str(ENTRYPOINT),
            "motion_npz": str(MOTION_NPZ),
            "enriched_usd": str(ENRICHED_USD),
            "csv_conversion_audit": str(CSV_CONVERSION_AUDIT),
        },
        "outputs": {
            "json": str(OUT / "tracking_g1_resource_adjusted_csv_full_replay_audit.json"),
            "metrics_json": str(METRICS_JSON),
            "log": str(log_path),
        },
        "interpretation": {
            "goal_complete": False,
            "official_replay_complete": False,
            "paper_level_tracking_eval_complete": False,
            "why_not_complete": (
                "This full replay uses a motion derived from official CSV data, but the robot asset path is still the "
                "generated enriched USD scaffold rather than the official URDF converter output. It narrows the "
                "blocker and validates the replay surface, but does not replace official csv_to_npz/replay, PPO, "
                "DAgger, teacher rollout data, or paper-level metrics."
            ),
        },
    }
    (OUT / "tracking_g1_resource_adjusted_csv_full_replay_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": status, "latest_blocker": latest_blocker}, sort_keys=True))
    if status.endswith("_blocker") and not METRICS_JSON.is_file():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
