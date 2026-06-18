#!/usr/bin/env python3
"""Run a bounded resource-adjusted official tracking task smoke probe."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/g1_resource_adjusted_task_smoke"
LOG_DIR = ROOT / "logs/tracking_g1_resource_adjusted_task_smoke"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
METRICS_JSON = OUT / "tracking_g1_resource_adjusted_task_smoke_metrics.json"
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
MOTION_FIXTURE = ROOT / "reproduction/data/tracking_motion_npz_fixtures/walk1_subject1_frames_1_180_debug_motion.npz"
BOUNDED_METRICS_GATE = (
    ROOT
    / "res/tracking/g1_enriched_usd_bounded_replay_metrics/"
    "tracking_g1_enriched_usd_bounded_replay_metrics_audit.json"
)
TIMEOUT_SECONDS = 260


PROBE_CODE = r"""
import json
import os
from pathlib import Path

ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = Path(os.environ["BM_TASK_SMOKE_METRICS"])
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

    print("BM_SENTINEL:before_cfg", flush=True)
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
    print("BM_SENTINEL:cfg_ready", flush=True)

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print("BM_SENTINEL:env_created", flush=True)
    obs, extras = env.reset()
    print("BM_SENTINEL:env_reset", flush=True)

    def tensor_shape(x):
        if hasattr(x, "shape"):
            return list(x.shape)
        if isinstance(x, dict):
            return {k: tensor_shape(v) for k, v in x.items()}
        return str(type(x))

    action_dim = int(env.unwrapped.action_manager.total_action_dim)
    action = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
    rewards = []
    terminated_counts = []
    truncated_counts = []
    step_count = 8
    for i in range(step_count):
        obs, reward, terminated, truncated, extras = env.step(action)
        rewards.append(float(reward.detach().cpu().mean().item()))
        terminated_counts.append(int(terminated.detach().cpu().sum().item()))
        truncated_counts.append(int(truncated.detach().cpu().sum().item()))
        print(f"BM_SENTINEL:env_step={i + 1}", flush=True)

    command = env.unwrapped.command_manager.get_term("motion")
    command_metrics = {
        k: float(v.detach().cpu().mean().item())
        for k, v in command.metrics.items()
        if hasattr(v, "detach") and v.numel() > 0
    }
    metrics = {
        "task": "Tracking-Flat-G1-v0",
        "num_envs": int(env.unwrapped.num_envs),
        "device": str(env.unwrapped.device),
        "action_dim": action_dim,
        "observation_shapes": tensor_shape(obs),
        "single_observation_space": str(env.unwrapped.single_observation_space),
        "single_action_space": str(env.unwrapped.single_action_space),
        "policy_observation_dim": int(env.unwrapped.observation_manager.group_obs_dim["policy"][0]),
        "critic_observation_dim": int(env.unwrapped.observation_manager.group_obs_dim["critic"][0]),
        "action_terms": list(env.unwrapped.action_manager.active_terms),
        "observation_terms": env.unwrapped.observation_manager.active_terms,
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
        "motion_file": str(MOTION_FIXTURE),
        "usd_path": str(ENRICHED_USD),
        "uses_resource_adjusted_usd": True,
        "official_csv_to_npz_output": False,
        "paper_level_rollout": False,
        "ppo_training": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"BM_SENTINEL:metrics_file={OUT}", flush=True)
    print("BM_SENTINEL:task_smoke_success", flush=True)
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
        "cfg_ready": "bm_sentinel:cfg_ready" in lowered,
        "env_created": "bm_sentinel:env_created" in lowered,
        "env_reset": "bm_sentinel:env_reset" in lowered,
        "env_step_8": "bm_sentinel:env_step=8" in lowered,
        "metrics_file": "bm_sentinel:metrics_file=" in lowered,
        "success": "bm_sentinel:task_smoke_success" in lowered,
        "traceback": "traceback (most recent call last)" in lowered,
        "exception": "bm_sentinel:exception=" in lowered,
        "argparse_conflict": "already has the field 'headless'" in lowered
        or "already has the field 'device'" in lowered,
        "vulkan_device_lost": "vkresult: error_device_lost" in lowered or "device lost" in lowered,
        "timeout": "BM_TIMEOUT" in text,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    script_path = OUT / "tracking_g1_resource_adjusted_task_smoke_probe.py"
    log_path = LOG_DIR / "tracking_g1_resource_adjusted_task_smoke.log"
    script_path.write_text(PROBE_CODE, encoding="utf-8")
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
            "BM_TASK_SMOKE_METRICS": str(METRICS_JSON),
            "BM_ENRICHED_USD": str(ENRICHED_USD),
            "BM_MOTION_FIXTURE": str(MOTION_FIXTURE),
        }
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    command = [
        "timeout",
        str(TIMEOUT_SECONDS),
        str(TRACKING_PY),
        str(script_path),
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
    markers = classify(output)
    metrics = load_json(METRICS_JSON)
    bounded_gate = load_json(BOUNDED_METRICS_GATE)
    checks = {
        "probe_script_written": script_path.is_file(),
        "enriched_usd_exists": ENRICHED_USD.is_file(),
        "motion_fixture_exists": MOTION_FIXTURE.is_file(),
        "bounded_replay_metrics_gate_passed": bounded_gate.get("status") == "ok_resource_adjusted_64step_metrics_gate",
        "process_returned_zero": proc.returncode == 0,
        "cfg_ready": markers["cfg_ready"],
        "env_created": markers["env_created"],
        "env_reset": markers["env_reset"],
        "env_step_8": markers["env_step_8"],
        "metrics_file_written": METRICS_JSON.is_file() and markers["metrics_file"],
        "action_dim_29": metrics.get("action_dim") == 29,
        "policy_observation_dim_160": metrics.get("policy_observation_dim") == 160,
        "critic_observation_dim_286": metrics.get("critic_observation_dim") == 286,
        "reward_terms_9": len(metrics.get("reward_terms", [])) == 9,
        "termination_terms_4": len(metrics.get("termination_terms", [])) == 4,
        "robot_joint_count_29": metrics.get("robot_num_joints") == 29,
        "robot_body_count_40": metrics.get("robot_num_bodies") == 40,
        "step_count_8": metrics.get("step_count") == 8,
        "uses_resource_adjusted_usd": metrics.get("uses_resource_adjusted_usd") is True,
        "does_not_claim_official_csv_to_npz_output": metrics.get("official_csv_to_npz_output") is False,
        "does_not_claim_paper_level_rollout": metrics.get("paper_level_rollout") is False,
        "does_not_start_training": metrics.get("ppo_training") is False,
    }
    passed = all(checks.values())
    if passed:
        status = "ok_resource_adjusted_tracking_task_smoke"
        latest_blocker = "none_resource_adjusted_task_smoke_passed"
    elif markers["argparse_conflict"]:
        status = "ok_with_resource_adjusted_task_smoke_blocker"
        latest_blocker = "argparse_conflict"
    elif markers["vulkan_device_lost"]:
        status = "ok_with_resource_adjusted_task_smoke_blocker"
        latest_blocker = "vulkan_device_lost"
    elif markers["exception"]:
        status = "ok_with_resource_adjusted_task_smoke_blocker"
        latest_blocker = "python_exception"
    elif proc.returncode == 124:
        status = "ok_with_resource_adjusted_task_smoke_blocker"
        latest_blocker = "timeout"
    else:
        status = "ok_with_resource_adjusted_task_smoke_blocker"
        latest_blocker = "unclassified"
    summary = {
        "status": status,
        "experiment_type": "tracking_resource_adjusted_task_smoke",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Bounded task-level smoke for official Tracking-Flat-G1-v0 manager stack using the generated "
            "resource-adjusted enriched G1 USD and debug motion fixture. This is not official replay/evaluation, "
            "PPO training, DAgger, or paper-level evidence."
        ),
        "command": command,
        "timeout_seconds": TIMEOUT_SECONDS,
        "returncode": proc.returncode,
        "latest_blocker": latest_blocker,
        "markers": markers,
        "metrics": metrics,
        "checks": checks,
        "inputs": {
            "enriched_usd": str(ENRICHED_USD),
            "motion_fixture": str(MOTION_FIXTURE),
            "bounded_metrics_gate": str(BOUNDED_METRICS_GATE),
        },
        "outputs": {
            "json": str(OUT / "tracking_g1_resource_adjusted_task_smoke_audit.json"),
            "metrics_json": str(METRICS_JSON),
            "probe_script": str(script_path),
            "log": str(log_path),
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_tracking_eval_complete": False,
            "official_beyondmimic_replay_complete": False,
            "why_not_complete": (
                "This smoke test reuses the official task manager stack but substitutes a generated resource-adjusted "
                "USD and debug fixture. It does not replace official motion conversion, official replay/evaluation, "
                "PPO training, DAgger, or paper-level closed-loop results."
            ),
        },
    }
    (OUT / "tracking_g1_resource_adjusted_task_smoke_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": status, "returncode": proc.returncode, "latest_blocker": latest_blocker}, sort_keys=True))


if __name__ == "__main__":
    main()
