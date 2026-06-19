#!/usr/bin/env python3
"""Evaluate the official-CSV-derived resource-adjusted motion in Tracking-Flat-G1-v0."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/g1_resource_adjusted_csv_task_eval"
LOG_DIR = ROOT / "logs/tracking_g1_resource_adjusted_csv_task_eval"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
METRICS_JSON = OUT / "tracking_g1_resource_adjusted_csv_task_eval_metrics.json"
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
ENRICHED_USD = (
    ROOT
    / "res/tracking/g1_resource_adjusted_enriched_usd/"
    "g1_resource_adjusted_29dof_enriched_scaffold.usda"
)
CSV_MOTION_NPZ = (
    ROOT
    / "res/tracking/g1_resource_adjusted_csv_conversion/"
    "walk1_subject1_frames_1_180_resource_adjusted_motion.npz"
)
CSV_CONTRACT = (
    ROOT
    / "res/tracking/g1_resource_adjusted_csv_conversion/"
    "walk1_subject1_frames_1_180_resource_adjusted_motion_contract.json"
)
CSV_FULL_REPLAY = (
    ROOT
    / "res/tracking/g1_resource_adjusted_csv_full_replay/"
    "tracking_g1_resource_adjusted_csv_full_replay_audit.json"
)
STALL_SECONDS = 900
CANDIDATE_GPUS = [4, 7]
TARGET_GPU = int(os.environ.get("BM_CSV_TASK_EVAL_GPU", "4"))
PROMOTE_CANONICAL = os.environ.get("BM_CSV_TASK_EVAL_PROMOTE", "0") == "1"


PROBE_CODE = r"""
import json
import os
from pathlib import Path

OUT = Path(os.environ["BM_CSV_TASK_METRICS"])
ENRICHED_USD = Path(os.environ["BM_ENRICHED_USD"])
MOTION_FILE = Path(os.environ["BM_MOTION_FILE"])

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
target_gpu = os.environ.get("BM_TARGET_GPU", "4")
args.headless = True
args.enable_cameras = False
args.device = os.environ.get("BM_DEVICE", f"cuda:{target_gpu}")
args.multi_gpu = False
args.kit_args = (
    "--/renderer/multiGpu/enabled=false "
    "--/renderer/multiGpu/autoEnable=false "
    "--/renderer/multiGpu/maxGpuCount=1 "
    f"--/renderer/activeGpu={target_gpu} "
    f"--/physics/cudaDevice={target_gpu}"
)

print("BM_SENTINEL:before_app", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("BM_SENTINEL:after_app", flush=True)

try:
    import gymnasium as gym
    import torch
    import isaaclab.sim as sim_utils
    import whole_body_tracking.tasks  # noqa: F401
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

    def tensor_shape(x):
        if hasattr(x, "shape"):
            return list(x.shape)
        if isinstance(x, dict):
            return {k: tensor_shape(v) for k, v in x.items()}
        return str(type(x))

    print(f"BM_SENTINEL:motion_file={MOTION_FILE}", flush=True)
    env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=str(ENRICHED_USD),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    )
    env_cfg.commands.motion.motion_file = str(MOTION_FILE)
    env_cfg.commands.motion.debug_vis = False
    env_cfg.scene.contact_forces.debug_vis = False
    env_cfg.sim.device = args.device
    env_cfg.episode_length_s = 0.24
    env_cfg.seed = 123

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print("BM_SENTINEL:env_created", flush=True)
    obs, extras = env.reset()
    print("BM_SENTINEL:env_reset", flush=True)

    action_dim = int(env.unwrapped.action_manager.total_action_dim)
    action = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
    command = env.unwrapped.command_manager.get_term("motion")
    step_count = int(command.motion.time_step_total)
    rewards = []
    terminated_counts = []
    truncated_counts = []
    for i in range(step_count):
        obs, reward, terminated, truncated, extras = env.step(action)
        rewards.append(float(reward.detach().cpu().mean().item()))
        terminated_counts.append(int(terminated.detach().cpu().sum().item()))
        truncated_counts.append(int(truncated.detach().cpu().sum().item()))
        if (i + 1) % 50 == 0 or (i + 1) == step_count:
            print(f"BM_SENTINEL:env_step={i + 1}/{step_count}", flush=True)

    command_metrics = {
        k: float(v.detach().cpu().mean().item())
        for k, v in command.metrics.items()
        if hasattr(v, "detach") and v.numel() > 0
    }
    metrics = {
        "task": "Tracking-Flat-G1-v0",
        "motion_file": str(MOTION_FILE),
        "num_envs": int(env.unwrapped.num_envs),
        "device": str(env.unwrapped.device),
        "action_dim": action_dim,
        "policy_observation_dim": int(env.unwrapped.observation_manager.group_obs_dim["policy"][0]),
        "critic_observation_dim": int(env.unwrapped.observation_manager.group_obs_dim["critic"][0]),
        "observation_shapes": tensor_shape(obs),
        "action_terms": list(env.unwrapped.action_manager.active_terms),
        "reward_terms": list(env.unwrapped.reward_manager.active_terms),
        "termination_terms": list(env.unwrapped.termination_manager.active_terms),
        "command_terms": list(env.unwrapped.command_manager.active_terms),
        "event_modes": list(env.unwrapped.event_manager.available_modes),
        "step_count": step_count,
        "reward_mean": sum(rewards) / len(rewards),
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "terminated_total": sum(terminated_counts),
        "truncated_total": sum(truncated_counts),
        "command_metrics": command_metrics,
        "robot_num_joints": int(env.unwrapped.scene["robot"].num_joints),
        "robot_num_bodies": int(env.unwrapped.scene["robot"].num_bodies),
        "usd_path": str(ENRICHED_USD),
        "uses_resource_adjusted_usd": True,
        "official_csv_to_npz_output": False,
        "official_csv_source": True,
        "paper_level_rollout": False,
        "ppo_training": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"BM_SENTINEL:metrics_file={OUT}", flush=True)
    print("BM_SENTINEL:csv_task_eval_success", flush=True)
    os._exit(0)
except Exception as exc:
    print("BM_SENTINEL:exception=" + repr(exc), flush=True)
    raise
finally:
    try:
        simulation_app.close(wait_for_replicator=False)
        print("BM_SENTINEL:after_close", flush=True)
    except Exception:
        pass
"""


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def classify(text: str) -> dict[str, bool]:
    lowered = text.lower()
    return {
        "before_app": "bm_sentinel:before_app" in lowered,
        "after_app": "bm_sentinel:after_app" in lowered,
        "env_created": "bm_sentinel:env_created" in lowered,
        "env_reset": "bm_sentinel:env_reset" in lowered,
        "step_299": "bm_sentinel:env_step=299/299" in lowered,
        "metrics_file": "bm_sentinel:metrics_file=" in lowered,
        "success": "bm_sentinel:csv_task_eval_success" in lowered,
        "traceback": "traceback (most recent call last)" in lowered,
        "exception": "bm_sentinel:exception=" in lowered,
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
            "BM_ENRICHED_USD": str(ENRICHED_USD),
            "BM_MOTION_FILE": str(CSV_MOTION_NPZ),
            "BM_CSV_TASK_METRICS": str(METRICS_JSON),
            "BM_TARGET_GPU": str(TARGET_GPU),
            "BM_DEVICE": f"cuda:{TARGET_GPU}",
        }
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def run_probe(script_path: Path, log_path: Path) -> dict[str, Any]:
    command = [str(TRACKING_PY), str(script_path), "--headless", "--device", f"cuda:{TARGET_GPU}"]
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
            start_new_session=True,
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
    global METRICS_JSON
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    candidate_dir = OUT / "candidate_gpu47_reruns" / run_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    script_path = OUT / "tracking_g1_resource_adjusted_csv_task_eval_probe.py"
    log_path = LOG_DIR / f"tracking_g1_resource_adjusted_csv_task_eval_{run_id}_gpu{TARGET_GPU}.log"
    candidate_metrics_json = candidate_dir / "tracking_g1_resource_adjusted_csv_task_eval_metrics.json"
    script_path.write_text(PROBE_CODE, encoding="utf-8")
    previous_metrics = load_json(METRICS_JSON)
    if candidate_metrics_json.exists():
        candidate_metrics_json.unlink()
    METRICS_JSON = candidate_metrics_json
    run = run_probe(script_path, log_path)
    metrics = load_json(candidate_metrics_json)
    csv_contract = load_json(CSV_CONTRACT)
    csv_full_replay = load_json(CSV_FULL_REPLAY)
    checks = {
        "probe_script_written": script_path.is_file(),
        "motion_npz_exists": CSV_MOTION_NPZ.is_file(),
        "csv_contract_exists": CSV_CONTRACT.is_file(),
        "enriched_usd_exists": ENRICHED_USD.is_file(),
        "prior_csv_full_replay_passed": csv_full_replay.get("status") == "ok_resource_adjusted_csv_full_replay",
        "contract_output_frames_299": csv_contract.get("output_frames") == 299,
        "contract_joint_shape_299_29": csv_contract.get("joint_pos_shape") == [299, 29],
        "process_returned_zero": run["returncode"] == 0,
        "no_stall_timeout": run["stalled"] is False,
        "app_reached": run["markers"]["after_app"],
        "env_created": run["markers"]["env_created"],
        "env_reset": run["markers"]["env_reset"],
        "step_299_reached": run["markers"]["step_299"],
        "success_sentinel_seen": run["markers"]["success"],
        "metrics_file_written": METRICS_JSON.is_file() and run["markers"]["metrics_file"],
        "step_count_299": metrics.get("step_count") == 299,
        "action_dim_29": metrics.get("action_dim") == 29,
        "policy_observation_dim_160": metrics.get("policy_observation_dim") == 160,
        "critic_observation_dim_286": metrics.get("critic_observation_dim") == 286,
        "reward_terms_9": len(metrics.get("reward_terms", [])) == 9,
        "termination_terms_4": len(metrics.get("termination_terms", [])) == 4,
        "robot_joint_count_29": metrics.get("robot_num_joints") == 29,
        "robot_body_count_40": metrics.get("robot_num_bodies") == 40,
        "uses_resource_adjusted_usd": metrics.get("uses_resource_adjusted_usd") is True,
        "official_csv_source": metrics.get("official_csv_source") is True,
        "does_not_claim_official_csv_to_npz_output": metrics.get("official_csv_to_npz_output") is False,
        "does_not_claim_paper_level_rollout": metrics.get("paper_level_rollout") is False,
        "does_not_start_training": metrics.get("ppo_training") is False,
    }
    passed = all(checks.values())
    if passed:
        status = "ok_resource_adjusted_csv_task_eval"
        latest_blocker = "none_resource_adjusted_csv_task_eval_passed"
    elif run["markers"]["vulkan_device_lost"]:
        status = "ok_with_resource_adjusted_csv_task_eval_blocker"
        latest_blocker = "vulkan_device_lost"
    elif run["stalled"]:
        status = "ok_with_resource_adjusted_csv_task_eval_blocker"
        latest_blocker = "stall_timeout"
    elif run["markers"]["traceback"]:
        status = "ok_with_resource_adjusted_csv_task_eval_blocker"
        latest_blocker = "python_traceback"
    else:
        status = "ok_with_resource_adjusted_csv_task_eval_blocker"
        latest_blocker = "failed_checks"
    summary = {
        "status": status,
        "experiment_type": "tracking_resource_adjusted_csv_task_eval",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Full 299-step Tracking-Flat-G1-v0 task diagnostics for a motion converted from official G1 LAFAN CSV "
            "through the resource-adjusted enriched USD path. This verifies task manager surfaces on official-source "
            "motion data, but is not official csv_to_npz/replay/evaluation, PPO, or paper-level evidence."
        ),
        "latest_blocker": latest_blocker,
        "run": run,
        "metrics": metrics,
        "checks": checks,
        "inputs": {
            "motion_npz": str(CSV_MOTION_NPZ),
            "csv_contract": str(CSV_CONTRACT),
            "enriched_usd": str(ENRICHED_USD),
            "prior_csv_full_replay": str(CSV_FULL_REPLAY),
            "candidate_physical_gpus": CANDIDATE_GPUS,
            "target_physical_gpu": TARGET_GPU,
        },
        "outputs": {
            "json": str(candidate_dir / "tracking_g1_resource_adjusted_csv_task_eval_audit.json"),
            "metrics_json": str(candidate_metrics_json),
            "canonical_json": str(OUT / "tracking_g1_resource_adjusted_csv_task_eval_audit.json"),
            "canonical_metrics_json": str(OUT / "tracking_g1_resource_adjusted_csv_task_eval_metrics.json"),
            "probe_script": str(script_path),
            "log": str(log_path),
        },
        "interpretation": {
            "goal_complete": False,
            "official_tracking_eval_complete": False,
            "paper_level_tracking_eval_complete": False,
            "why_not_complete": (
                "The motion source is an official downloaded CSV, but the robot asset path still uses the generated "
                "resource-adjusted enriched USD. This gate validates the official task manager contract and full "
                "motion length for that source-derived data; it does not replace official replay/evaluation, PPO, "
                "DAgger, teacher rollout data, or paper-level tracking metrics."
            ),
        },
        "canonical_promotion": {
            "requested": PROMOTE_CANONICAL,
            "performed": bool(passed and PROMOTE_CANONICAL),
            "why_default_is_false": (
                "Failed reruns on required GPUs 4/7 must not overwrite the earlier canonical successful task-eval "
                "artifact. Set BM_CSV_TASK_EVAL_PROMOTE=1 only after a candidate rerun completes all 299 steps."
            ),
            "previous_canonical_metrics_present": bool(previous_metrics),
        },
    }
    (candidate_dir / "tracking_g1_resource_adjusted_csv_task_eval_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    if passed and PROMOTE_CANONICAL:
        (OUT / "tracking_g1_resource_adjusted_csv_task_eval_audit.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        (OUT / "tracking_g1_resource_adjusted_csv_task_eval_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(json.dumps({"status": status, "latest_blocker": latest_blocker}, sort_keys=True))
    if status.endswith("_blocker") and not candidate_metrics_json.is_file():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
