#!/usr/bin/env python3
"""Run resource-adjusted multi-fixture official tracking task diagnostics."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/g1_resource_adjusted_multi_fixture_eval"
LOG_DIR = ROOT / "logs/tracking_g1_resource_adjusted_multi_fixture_eval"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
METRICS_JSON = OUT / "tracking_g1_resource_adjusted_multi_fixture_eval_metrics.json"
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
TASK_SMOKE = ROOT / "res/tracking/g1_resource_adjusted_task_smoke/tracking_g1_resource_adjusted_task_smoke_audit.json"
FIXTURES = [
    ROOT / "reproduction/data/tracking_motion_npz_fixtures/walk1_subject1_frames_1_180_debug_motion.npz",
    ROOT / "reproduction/data/tracking_motion_npz_fixtures/run2_subject1_frames_1_180_debug_motion.npz",
    ROOT / "reproduction/data/tracking_motion_npz_fixtures/jumps1_subject1_frames_1_180_debug_motion.npz",
]
STALL_SECONDS = 900


PROBE_CODE = r"""
import json
import os
from pathlib import Path

OUT = Path(os.environ["BM_FIXTURE_METRICS"])
ENRICHED_USD = Path(os.environ["BM_ENRICHED_USD"])
MOTION_FIXTURE = Path(os.environ["BM_MOTION_FIXTURE"])

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

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

    print(f"BM_SENTINEL:fixture_start={MOTION_FIXTURE.name}", flush=True)
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
    env_cfg.commands.motion.motion_file = str(MOTION_FIXTURE)
    env_cfg.commands.motion.debug_vis = False
    env_cfg.scene.contact_forces.debug_vis = False
    env_cfg.sim.device = args.device
    env_cfg.episode_length_s = 0.24
    env_cfg.seed = 123

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print("BM_SENTINEL:env_created", flush=True)
    obs, extras = env.reset()
    print(f"BM_SENTINEL:fixture_reset={MOTION_FIXTURE.name}", flush=True)

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
            print(f"BM_SENTINEL:fixture_step={MOTION_FIXTURE.name}:{i + 1}/{step_count}", flush=True)

    command_metrics = {
        k: float(v.detach().cpu().mean().item())
        for k, v in command.metrics.items()
        if hasattr(v, "detach") and v.numel() > 0
    }
    metrics = {
        "fixture": MOTION_FIXTURE.name,
        "motion_file": str(MOTION_FIXTURE),
        "task": "Tracking-Flat-G1-v0",
        "num_envs": int(env.unwrapped.num_envs),
        "device": str(env.unwrapped.device),
        "action_dim": action_dim,
        "policy_observation_dim": int(env.unwrapped.observation_manager.group_obs_dim["policy"][0]),
        "critic_observation_dim": int(env.unwrapped.observation_manager.group_obs_dim["critic"][0]),
        "observation_shapes": tensor_shape(obs),
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
        "paper_level_rollout": False,
        "ppo_training": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"BM_SENTINEL:metrics_file={OUT}", flush=True)
    print(f"BM_SENTINEL:fixture_done={MOTION_FIXTURE.name}", flush=True)
    print("BM_SENTINEL:fixture_eval_success", flush=True)
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


def classify(text: str) -> dict[str, Any]:
    lowered = text.lower()
    return {
        "before_app": "bm_sentinel:before_app" in lowered,
        "after_app": "bm_sentinel:after_app" in lowered,
        "env_created": "bm_sentinel:env_created" in lowered,
        "fixture_start": "bm_sentinel:fixture_start=" in lowered,
        "fixture_reset": "bm_sentinel:fixture_reset=" in lowered,
        "fixture_step_299": ":299/299" in lowered,
        "fixture_done": "bm_sentinel:fixture_done=" in lowered,
        "metrics_file": "bm_sentinel:metrics_file=" in lowered,
        "success": "bm_sentinel:fixture_eval_success" in lowered,
        "traceback": "traceback (most recent call last)" in lowered,
        "exception": "bm_sentinel:exception=" in lowered,
        "vulkan_device_lost": "vkresult: error_device_lost" in lowered or "device lost" in lowered,
        "stall_timeout": "bm_stall_timeout" in lowered,
    }


def base_env() -> dict[str, str]:
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
        }
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def run_fixture(script_path: Path, fixture: Path) -> dict[str, Any]:
    metric_path = OUT / f"{fixture.stem}_task_eval_metrics.json"
    log_path = LOG_DIR / f"{fixture.stem}.log"
    env = base_env()
    env.update({"BM_MOTION_FIXTURE": str(fixture), "BM_FIXTURE_METRICS": str(metric_path)})
    command = [str(TRACKING_PY), str(script_path), "--headless", "--device", "cuda:6"]
    start = time.time()
    stalled = False
    last_size = -1
    last_change = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        while proc.poll() is None:
            time.sleep(10)
            try:
                current_size = log_path.stat().st_size
            except FileNotFoundError:
                current_size = 0
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
        "fixture": fixture.name,
        "command": command,
        "returncode": proc.returncode,
        "duration_seconds": round(time.time() - start, 3),
        "stalled": stalled,
        "markers": classify(text),
        "metrics_path": str(metric_path),
        "log": str(log_path),
        "metrics": load_json(metric_path),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    script_path = OUT / "tracking_g1_resource_adjusted_single_fixture_eval_probe.py"
    script_path.write_text(PROBE_CODE, encoding="utf-8")

    fixture_runs = [run_fixture(script_path, fixture) for fixture in FIXTURES]
    rows = [run["metrics"] for run in fixture_runs if run["metrics"]]
    metrics = {
        "task": "Tracking-Flat-G1-v0",
        "fixture_count": len(rows),
        "total_steps": sum(int(r.get("step_count", 0)) for r in rows),
        "fixtures": [r.get("fixture") for r in rows],
        "rows": rows,
        "action_dim_all_29": all(r.get("action_dim") == 29 for r in rows),
        "policy_observation_dim_all_160": all(r.get("policy_observation_dim") == 160 for r in rows),
        "critic_observation_dim_all_286": all(r.get("critic_observation_dim") == 286 for r in rows),
        "reward_terms_all_9": all(len(r.get("reward_terms", [])) == 9 for r in rows),
        "termination_terms_all_4": all(len(r.get("termination_terms", [])) == 4 for r in rows),
        "robot_num_joints_all_29": all(r.get("robot_num_joints") == 29 for r in rows),
        "robot_num_bodies_all_40": all(r.get("robot_num_bodies") == 40 for r in rows),
        "uses_resource_adjusted_usd": all(r.get("uses_resource_adjusted_usd") is True for r in rows),
        "official_csv_to_npz_output": any(r.get("official_csv_to_npz_output") is True for r in rows),
        "paper_level_rollout": any(r.get("paper_level_rollout") is True for r in rows),
        "ppo_training": any(r.get("ppo_training") is True for r in rows),
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    task_smoke = load_json(TASK_SMOKE)
    checks = {
        "probe_script_written": script_path.is_file(),
        "enriched_usd_exists": ENRICHED_USD.is_file(),
        "all_fixtures_exist": all(p.is_file() for p in FIXTURES),
        "prior_task_smoke_passed": task_smoke.get("status") == "ok_resource_adjusted_tracking_task_smoke",
        "all_fixture_processes_returned_zero": all(run["returncode"] == 0 for run in fixture_runs),
        "no_fixture_stall_timeout": not any(run["stalled"] for run in fixture_runs),
        "fixture_start_count_3": sum(1 for run in fixture_runs if run["markers"]["fixture_start"]) == 3,
        "fixture_done_count_3": sum(1 for run in fixture_runs if run["markers"]["fixture_done"]) == 3,
        "fixture_step_299_count_3": sum(1 for run in fixture_runs if run["markers"]["fixture_step_299"]) == 3,
        "metrics_file_count_3": sum(1 for run in fixture_runs if run["markers"]["metrics_file"]) == 3,
        "success_sentinel_count_3": sum(1 for run in fixture_runs if run["markers"]["success"]) == 3,
        "aggregate_metrics_file_written": METRICS_JSON.is_file(),
        "fixture_count_3": metrics.get("fixture_count") == 3,
        "total_steps_897": metrics.get("total_steps") == 897,
        "all_fixture_steps_299": all(int(row.get("step_count", 0)) == 299 for row in rows),
        "action_dim_all_29": metrics.get("action_dim_all_29") is True,
        "policy_observation_dim_all_160": metrics.get("policy_observation_dim_all_160") is True,
        "critic_observation_dim_all_286": metrics.get("critic_observation_dim_all_286") is True,
        "reward_terms_all_9": metrics.get("reward_terms_all_9") is True,
        "termination_terms_all_4": metrics.get("termination_terms_all_4") is True,
        "robot_num_joints_all_29": metrics.get("robot_num_joints_all_29") is True,
        "robot_num_bodies_all_40": metrics.get("robot_num_bodies_all_40") is True,
        "uses_resource_adjusted_usd": metrics.get("uses_resource_adjusted_usd") is True,
        "does_not_claim_official_csv_to_npz_output": metrics.get("official_csv_to_npz_output") is False,
        "does_not_claim_paper_level_rollout": metrics.get("paper_level_rollout") is False,
        "does_not_start_training": metrics.get("ppo_training") is False,
    }
    passed = all(checks.values())
    blockers = []
    for run in fixture_runs:
        if run["returncode"] != 0:
            blockers.append(f"{run['fixture']}:returncode_{run['returncode']}")
        if run["stalled"]:
            blockers.append(f"{run['fixture']}:stall_timeout")
        if run["markers"]["exception"]:
            blockers.append(f"{run['fixture']}:python_exception")
        if run["markers"]["vulkan_device_lost"]:
            blockers.append(f"{run['fixture']}:vulkan_device_lost")
    if passed:
        status = "ok_resource_adjusted_multi_fixture_task_eval"
        latest_blocker = "none_resource_adjusted_multi_fixture_eval_passed"
    else:
        status = "ok_with_resource_adjusted_multi_fixture_eval_blocker"
        latest_blocker = ";".join(blockers) if blockers else "failed_checks"
    summary = {
        "status": status,
        "experiment_type": "tracking_resource_adjusted_multi_fixture_task_eval",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Full available fixture diagnostics for official Tracking-Flat-G1-v0 manager stack using the generated "
            "enriched G1 USD and three debug motion fixtures. Each fixture runs in an isolated Kit process for all "
            "299 available motion steps. This is not official csv_to_npz/replay evaluation, PPO, DAgger, or "
            "paper-level closed-loop evidence."
        ),
        "stall_seconds": STALL_SECONDS,
        "returncode": 0 if passed else 1,
        "latest_blocker": latest_blocker,
        "fixture_runs": fixture_runs,
        "metrics": metrics,
        "checks": checks,
        "inputs": {
            "enriched_usd": str(ENRICHED_USD),
            "fixtures": [str(p) for p in FIXTURES],
            "task_smoke": str(TASK_SMOKE),
        },
        "outputs": {
            "json": str(OUT / "tracking_g1_resource_adjusted_multi_fixture_eval_audit.json"),
            "metrics_json": str(METRICS_JSON),
            "probe_script": str(script_path),
            "logs": [run["log"] for run in fixture_runs],
            "fixture_metrics": [run["metrics_path"] for run in fixture_runs],
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_tracking_eval_complete": False,
            "official_beyondmimic_replay_complete": False,
            "why_not_complete": (
                "This diagnostic reuses the official task manager stack but substitutes generated resource-adjusted "
                "USD and debug fixtures. It validates full fixture execution and task contracts, but it does not "
                "replace official motion conversion, official replay/evaluation, PPO training, DAgger, or paper-level "
                "closed-loop results."
            ),
        },
    }
    (OUT / "tracking_g1_resource_adjusted_multi_fixture_eval_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": status, "latest_blocker": latest_blocker, "total_steps": metrics["total_steps"]}, sort_keys=True))


if __name__ == "__main__":
    main()
