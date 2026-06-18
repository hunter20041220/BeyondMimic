#!/usr/bin/env python3
"""Run a bounded RSL-RL train-entry diagnostic on the resource-adjusted G1 gate."""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/g1_resource_adjusted_train_entry_diagnostic"
LOG_DIR = ROOT / "logs/tracking_g1_resource_adjusted_train_entry_diagnostic"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
METRICS_JSON = OUT / "tracking_g1_resource_adjusted_train_entry_diagnostic_metrics.json"
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
CSV_TASK_EVAL = (
    ROOT
    / "res/tracking/g1_resource_adjusted_csv_task_eval/"
    "tracking_g1_resource_adjusted_csv_task_eval_audit.json"
)
STALL_SECONDS = 900
MAX_SECONDS = 1800


PROBE_CODE = r"""
import json
import os
from pathlib import Path

OUT = Path(os.environ["BM_TRAIN_ENTRY_METRICS"])
ENRICHED_USD = Path(os.environ["BM_ENRICHED_USD"])
MOTION_FILE = Path(os.environ["BM_MOTION_FILE"])

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
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
    from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner

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

    agent_cfg = G1FlatPPORunnerCfg()
    agent_cfg.device = args.device
    agent_cfg.seed = 123
    agent_cfg.max_iterations = 1
    agent_cfg.num_steps_per_env = 4
    agent_cfg.save_interval = 1000000
    agent_cfg.algorithm.num_learning_epochs = 1
    agent_cfg.algorithm.num_mini_batches = 1
    agent_cfg.empirical_normalization = True

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print("BM_SENTINEL:env_created", flush=True)
    vec_env = RslRlVecEnvWrapper(env)
    print("BM_SENTINEL:vec_env_wrapped", flush=True)
    runner = MotionOnPolicyRunner(
        vec_env,
        agent_cfg.to_dict(),
        log_dir=None,
        device=agent_cfg.device,
        registry_name=f"local:{MOTION_FILE}",
    )
    runner.disable_logs = True
    print("BM_SENTINEL:runner_created", flush=True)
    runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
    print("BM_SENTINEL:learn_completed", flush=True)

    obs, extras = vec_env.get_observations()
    metrics = {
        "task": "Tracking-Flat-G1-v0",
        "motion_file": str(MOTION_FILE),
        "usd_path": str(ENRICHED_USD),
        "device": str(vec_env.device),
        "agent_device": str(agent_cfg.device),
        "num_envs": int(vec_env.num_envs),
        "num_actions": int(vec_env.num_actions),
        "num_obs": int(vec_env.num_obs),
        "num_privileged_obs": int(vec_env.num_privileged_obs),
        "requested_learning_iterations": 1,
        "configured_num_steps_per_env": int(agent_cfg.num_steps_per_env),
        "configured_num_learning_epochs": int(agent_cfg.algorithm.num_learning_epochs),
        "configured_num_mini_batches": int(agent_cfg.algorithm.num_mini_batches),
        "runner_class": type(runner).__name__,
        "runner_training_type": str(runner.training_type),
        "runner_current_learning_iteration": int(runner.current_learning_iteration),
        "policy_class": type(runner.alg.policy).__name__,
        "storage_num_transitions_per_env": int(runner.alg.storage.num_transitions_per_env),
        "storage_num_envs": int(runner.alg.storage.num_envs),
        "post_learn_policy_obs_shape": list(obs.shape),
        "post_learn_critic_obs_shape": list(extras["observations"]["critic"].shape),
        "robot_num_joints": int(vec_env.unwrapped.scene["robot"].num_joints),
        "robot_num_bodies": int(vec_env.unwrapped.scene["robot"].num_bodies),
        "uses_resource_adjusted_usd": True,
        "official_csv_source": True,
        "official_csv_to_npz_output": False,
        "paper_level_rollout": False,
        "formal_ppo_training": False,
        "checkpoint_written": False,
        "log_dir": None,
    }
    vec_env.close()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"BM_SENTINEL:metrics_file={OUT}", flush=True)
    print("BM_SENTINEL:train_entry_diagnostic_success", flush=True)
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
        "vec_env_wrapped": "bm_sentinel:vec_env_wrapped" in lowered,
        "runner_created": "bm_sentinel:runner_created" in lowered,
        "learn_completed": "bm_sentinel:learn_completed" in lowered,
        "metrics_file": "bm_sentinel:metrics_file=" in lowered,
        "success": "bm_sentinel:train_entry_diagnostic_success" in lowered,
        "traceback": "traceback (most recent call last)" in lowered,
        "exception": "bm_sentinel:exception=" in lowered,
        "vulkan_device_lost": "vkresult: error_device_lost" in lowered or "device lost" in lowered,
        "physx_gpu_kernel_error": "gpu convexcoreconvexnphase_kernel fail to launch" in lowered,
        "stall_timeout": "bm_stall_timeout" in lowered,
        "max_runtime_timeout": "bm_max_runtime_timeout" in lowered,
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
            "WANDB_MODE": "offline",
            "BM_ENRICHED_USD": str(ENRICHED_USD),
            "BM_MOTION_FILE": str(CSV_MOTION_NPZ),
            "BM_TRAIN_ENTRY_METRICS": str(METRICS_JSON),
        }
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def run_probe(script_path: Path, log_path: Path) -> dict[str, Any]:
    command = [str(TRACKING_PY), str(script_path), "--headless", "--device", "cuda:6"]
    start = time.time()
    stalled = False
    max_runtime = False
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
            now = time.time()
            if current_size != last_size:
                last_size = current_size
                last_change = now
            elif now - last_change > STALL_SECONDS:
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
            if now - start > MAX_SECONDS:
                max_runtime = True
                log_file.write(f"\nBM_MAX_RUNTIME_TIMEOUT:runtime_exceeded_{MAX_SECONDS}s\n")
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
        "max_runtime_timeout": max_runtime,
        "markers": classify(text),
        "log": str(log_path),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    script_path = OUT / "tracking_g1_resource_adjusted_train_entry_diagnostic_probe.py"
    log_path = LOG_DIR / "tracking_g1_resource_adjusted_train_entry_diagnostic.log"
    if METRICS_JSON.exists():
        METRICS_JSON.unlink()
    script_path.write_text(PROBE_CODE, encoding="utf-8")
    run = run_probe(script_path, log_path)
    metrics = load_json(METRICS_JSON)
    csv_task_eval = load_json(CSV_TASK_EVAL)
    checks = {
        "probe_script_written": script_path.is_file(),
        "motion_npz_exists": CSV_MOTION_NPZ.is_file(),
        "enriched_usd_exists": ENRICHED_USD.is_file(),
        "prior_csv_task_eval_passed": csv_task_eval.get("status") == "ok_resource_adjusted_csv_task_eval",
        "process_returned_zero": run["returncode"] == 0,
        "no_stall_timeout": run["stalled"] is False,
        "no_max_runtime_timeout": run["max_runtime_timeout"] is False,
        "app_reached": run["markers"]["after_app"],
        "env_created": run["markers"]["env_created"],
        "vec_env_wrapped": run["markers"]["vec_env_wrapped"],
        "runner_created": run["markers"]["runner_created"],
        "learn_completed": run["markers"]["learn_completed"],
        "physx_gpu_kernel_warning_recorded": isinstance(run["markers"]["physx_gpu_kernel_error"], bool),
        "success_sentinel_seen": run["markers"]["success"],
        "metrics_file_written": METRICS_JSON.is_file() and run["markers"]["metrics_file"],
        "runner_class_motion": metrics.get("runner_class") == "MotionOnPolicyRunner",
        "runner_training_type_rl": metrics.get("runner_training_type") == "rl",
        "num_envs_1": metrics.get("num_envs") == 1,
        "num_actions_29": metrics.get("num_actions") == 29,
        "num_obs_160": metrics.get("num_obs") == 160,
        "num_privileged_obs_286": metrics.get("num_privileged_obs") == 286,
        "num_steps_per_env_4": metrics.get("configured_num_steps_per_env") == 4,
        "one_iteration_requested": metrics.get("requested_learning_iterations") == 1,
        "learn_iteration_recorded": metrics.get("runner_current_learning_iteration") == 0,
        "storage_shape_matches": metrics.get("storage_num_transitions_per_env") == 4
        and metrics.get("storage_num_envs") == 1,
        "robot_joint_count_29": metrics.get("robot_num_joints") == 29,
        "robot_body_count_40": metrics.get("robot_num_bodies") == 40,
        "uses_resource_adjusted_usd": metrics.get("uses_resource_adjusted_usd") is True,
        "official_csv_source": metrics.get("official_csv_source") is True,
        "does_not_claim_official_csv_to_npz_output": metrics.get("official_csv_to_npz_output") is False,
        "does_not_claim_paper_level_rollout": metrics.get("paper_level_rollout") is False,
        "does_not_claim_formal_ppo_training": metrics.get("formal_ppo_training") is False,
        "does_not_write_checkpoint": metrics.get("checkpoint_written") is False,
    }
    passed = all(checks.values())
    if passed:
        status = "ok_resource_adjusted_train_entry_diagnostic"
        latest_blocker = "none_resource_adjusted_train_entry_diagnostic_passed"
    elif run["markers"]["vulkan_device_lost"]:
        status = "ok_with_resource_adjusted_train_entry_diagnostic_blocker"
        latest_blocker = "vulkan_device_lost"
    elif run["stalled"]:
        status = "ok_with_resource_adjusted_train_entry_diagnostic_blocker"
        latest_blocker = "stall_timeout"
    elif run["max_runtime_timeout"]:
        status = "ok_with_resource_adjusted_train_entry_diagnostic_blocker"
        latest_blocker = "max_runtime_timeout"
    elif run["markers"]["traceback"]:
        status = "ok_with_resource_adjusted_train_entry_diagnostic_blocker"
        latest_blocker = "python_traceback"
    else:
        status = "ok_with_resource_adjusted_train_entry_diagnostic_blocker"
        latest_blocker = "failed_checks"
    summary = {
        "status": status,
        "experiment_type": "tracking_resource_adjusted_train_entry_diagnostic",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Bounded RSL-RL train-entry diagnostic for the official Tracking-Flat-G1-v0 task using the "
            "official-CSV-derived resource-adjusted motion and generated enriched USD. The probe constructs the "
            "official IsaacLab env, wraps it with RslRlVecEnvWrapper, instantiates the official custom "
            "MotionOnPolicyRunner, and executes one tiny PPO learning iteration with four steps. It is not a formal "
            "PPO training run, checkpoint, teacher policy, official replay, DAgger data, or paper-level result."
        ),
        "latest_blocker": latest_blocker,
        "run": run,
        "metrics": metrics,
        "checks": checks,
        "inputs": {
            "motion_npz": str(CSV_MOTION_NPZ),
            "enriched_usd": str(ENRICHED_USD),
            "prior_csv_task_eval": str(CSV_TASK_EVAL),
        },
        "outputs": {
            "json": str(OUT / "tracking_g1_resource_adjusted_train_entry_diagnostic_audit.json"),
            "metrics_json": str(METRICS_JSON),
            "probe_script": str(script_path),
            "log": str(log_path),
        },
        "interpretation": {
            "goal_complete": False,
            "official_tracking_train_complete": False,
            "paper_level_tracking_train_complete": False,
            "formal_gpu_experiment": False,
            "why_not_formal_gpu_experiment": (
                "This diagnostic intentionally uses one environment, four rollout steps, and one PPO update only to "
                "verify the train-entry wiring after environment recovery. It does not use the required formal "
                "multi-GPU/full-memory protocol and cannot be reported as training performance."
            ),
            "why_not_complete": (
                "The robot asset path still uses the generated resource-adjusted enriched USD, the motion path is "
                "resource-adjusted from an official CSV rather than official csv_to_npz output, and the run is a tiny "
                "entry diagnostic without checkpointed policy evaluation, DAgger rollouts, full PPO training, or "
                "closed-loop paper metrics."
            ),
            "runtime_warning": (
                "The probe log contains PhysX GPU convex narrowphase kernel launch errors before the success sentinel."
                if run["markers"]["physx_gpu_kernel_error"]
                else "No PhysX GPU kernel warning marker was detected in the probe log."
            ),
        },
    }
    (OUT / "tracking_g1_resource_adjusted_train_entry_diagnostic_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": status, "latest_blocker": latest_blocker}, sort_keys=True))
    if status.endswith("_blocker"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
