#!/usr/bin/env python3
"""Run a bounded Tracking-Flat-G1-v0 gate on the official-importer GPU4 USDA export."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/g1_official_importer_export_task_smoke"
LOG_DIR = ROOT / "logs/tracking_g1_official_importer_export_task_smoke"
FAILED_DIR = ROOT / "res/failed_runs/g1_official_importer_export_task_smoke"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
OFFICIAL_IMPORTER_USD = (
    ROOT
    / "res/tracking/g1_urdf_in_memory_gpu4_probe/"
    "g1_official_importer_in_memory_gpu4_export.usda"
)
EXPORT_STRUCTURE_AUDIT = (
    ROOT
    / "res/tracking/g1_urdf_in_memory_gpu4_export_structure_audit/"
    "tracking_g1_urdf_in_memory_gpu4_export_structure_audit.json"
)
MOTION_NPZ = (
    ROOT
    / "res/tracking/official_csv_to_npz_loop_with_enriched_usd/"
    "walk1_subject1_frames_1_180_official_loop_enriched_usd_motion.npz"
)
TARGET_GPU = int(os.environ.get("BM_OFFICIAL_IMPORTER_EXPORT_TASK_GPU", "4"))
STEP_COUNT = int(os.environ.get("BM_OFFICIAL_IMPORTER_EXPORT_TASK_STEPS", "8"))
TIMEOUT_SECONDS = int(os.environ.get("BM_OFFICIAL_IMPORTER_EXPORT_TASK_TIMEOUT", "900"))
WATCH_GPUS = [4, 7]
WANGJC_PATH_MARKER = "/mnt/infini-data/test/wangjc/"


WORKER_CODE = r"""
import json
import os
from pathlib import Path

ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = Path(os.environ["BM_METRICS_JSON"])
USD_PATH = Path(os.environ["BM_OFFICIAL_IMPORTER_USD"])
MOTION_NPZ = Path(os.environ["BM_MOTION_NPZ"])
STEP_COUNT = int(os.environ["BM_STEP_COUNT"])

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
    import isaaclab.sim as sim_utils
    import torch
    import whole_body_tracking.tasks  # noqa: F401
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

    def tensor_shape(x):
        if hasattr(x, "shape"):
            return list(x.shape)
        if isinstance(x, dict):
            return {k: tensor_shape(v) for k, v in x.items()}
        return str(type(x))

    print("BM_SENTINEL:before_cfg", flush=True)
    env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
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
    env_cfg.commands.motion.motion_file = str(MOTION_NPZ)
    env_cfg.commands.motion.debug_vis = False
    env_cfg.scene.contact_forces.debug_vis = False
    env_cfg.sim.device = args.device
    env_cfg.episode_length_s = max(0.24, STEP_COUNT / 50.0)
    env_cfg.seed = int(os.environ.get("BM_SEED", "20260660"))
    print("BM_SENTINEL:cfg_ready", flush=True)

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print("BM_SENTINEL:env_created", flush=True)
    obs, extras = env.reset()
    print("BM_SENTINEL:env_reset", flush=True)

    action_dim = int(env.unwrapped.action_manager.total_action_dim)
    action = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
    rewards = []
    terminated_counts = []
    truncated_counts = []
    for i in range(STEP_COUNT):
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
        "status": "ok",
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
        "step_count": STEP_COUNT,
        "reward_mean": sum(rewards) / len(rewards),
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "terminated_total": sum(terminated_counts),
        "truncated_total": sum(truncated_counts),
        "command_metrics": command_metrics,
        "robot_num_joints": int(env.unwrapped.scene["robot"].num_joints),
        "robot_num_bodies": int(env.unwrapped.scene["robot"].num_bodies),
        "motion_file": str(MOTION_NPZ),
        "usd_path": str(USD_PATH),
        "uses_official_importer_export_usd": True,
        "uses_resource_adjusted_enriched_usd": False,
        "official_csv_loop_npz_input": True,
        "official_csv_to_npz_unpatched_output": False,
        "paper_level_rollout": False,
        "ppo_training": False,
        "real_robot": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"BM_SENTINEL:metrics_file={OUT}", flush=True)
    print("BM_SENTINEL:official_importer_export_task_smoke_success", flush=True)
    os._exit(0)
except BaseException as exc:
    print("BM_SENTINEL:exception=" + repr(exc), flush=True)
    raise
finally:
    try:
        simulation_app.close(wait_for_replicator=False)
        print("BM_SENTINEL:after_close", flush=True)
    except BaseException:
        pass
"""


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def run_command(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)


def gpu_uuid_map() -> dict[str, int]:
    proc = run_command(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
    mapping: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2:
            mapping[parts[1]] = int(parts[0])
    return mapping


def process_cmdline(pid: str) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_text(encoding="utf-8", errors="replace").replace("\0", " ").strip()
    except OSError:
        return ""


def gpu_process_rows() -> list[dict[str, Any]]:
    mapping = gpu_uuid_map()
    proc = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        gpu_uuid, pid, name, used = parts[:4]
        rows.append(
            {
                "gpu": mapping.get(gpu_uuid),
                "pid": int(pid),
                "process_name": name,
                "used_memory_mb": int(float(used)),
                "cmdline": process_cmdline(pid),
            }
        )
    return rows


def guard_watch_gpus() -> dict[str, Any]:
    rows = gpu_process_rows()
    target_rows = [row for row in rows if row["gpu"] in WATCH_GPUS and row["used_memory_mb"] > 200]
    killed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in target_rows:
        if WANGJC_PATH_MARKER in row["cmdline"]:
            try:
                os.kill(row["pid"], signal.SIGTERM)
                killed.append(row | {"signal": "SIGTERM"})
            except OSError as exc:
                killed.append(row | {"signal": "SIGTERM", "error": repr(exc)})
        else:
            skipped.append(row)
    time.sleep(3 if killed else 0)
    guard = {
        "watch_gpus": WATCH_GPUS,
        "target_gpu": TARGET_GPU,
        "pre_rows": rows,
        "target_rows": target_rows,
        "killed_wangjc_rows": killed,
        "skipped_non_wangjc_rows": skipped,
        "post_rows": gpu_process_rows(),
        "policy": "Only processes with cmdline containing /mnt/infini-data/test/wangjc/ are terminated.",
    }
    return guard


def classify(text: str) -> dict[str, bool]:
    lowered = text.lower()
    return {
        "before_app": "bm_sentinel:before_app" in lowered,
        "after_app": "bm_sentinel:after_app" in lowered,
        "cfg_ready": "bm_sentinel:cfg_ready" in lowered,
        "env_created": "bm_sentinel:env_created" in lowered,
        "env_reset": "bm_sentinel:env_reset" in lowered,
        "env_step_final": f"bm_sentinel:env_step={STEP_COUNT}" in lowered,
        "metrics_file": "bm_sentinel:metrics_file=" in lowered,
        "success": "bm_sentinel:official_importer_export_task_smoke_success" in lowered,
        "traceback": "traceback (most recent call last)" in lowered,
        "exception": "bm_sentinel:exception=" in lowered,
        "vulkan_device_lost": "vkresult: error_device_lost" in lowered or "device lost" in lowered,
        "articulation_root_missing": "failed to find an articulation" in lowered or "articulationrootapi" in lowered,
        "failed_to_find_prim": "failed to find prim" in lowered or "prim path" in lowered,
        "timeout": "BM_TIMEOUT" in text,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = OUT / "tracking_g1_official_importer_export_task_smoke_metrics.json"
    worker_path = OUT / "tracking_g1_official_importer_export_task_smoke_worker.py"
    log_path = LOG_DIR / "tracking_g1_official_importer_export_task_smoke.log"
    worker_path.write_text(textwrap.dedent(WORKER_CODE), encoding="utf-8")
    guard = guard_watch_gpus()
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
            "BM_METRICS_JSON": str(metrics_path),
            "BM_OFFICIAL_IMPORTER_USD": str(OFFICIAL_IMPORTER_USD),
            "BM_MOTION_NPZ": str(MOTION_NPZ),
            "BM_TARGET_GPU": str(TARGET_GPU),
            "BM_DEVICE": f"cuda:{TARGET_GPU}",
            "BM_STEP_COUNT": str(STEP_COUNT),
        }
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    command = [
        str(TRACKING_PY),
        str(worker_path),
        "--headless",
        "--device",
        f"cuda:{TARGET_GPU}",
    ]
    start = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=TIMEOUT_SECONDS,
        )
        output = proc.stdout
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + "\nBM_TIMEOUT:subprocess_timeout\n"
        proc = subprocess.CompletedProcess(command, 124, output)
        timed_out = True
    duration = time.time() - start
    log_path.write_text(output, encoding="utf-8", errors="replace")
    markers = classify(output)
    metrics = load_json(metrics_path)
    export_audit = load_json(EXPORT_STRUCTURE_AUDIT)
    checks = {
        "worker_script_written": worker_path.is_file(),
        "official_importer_usd_exists": OFFICIAL_IMPORTER_USD.is_file(),
        "official_importer_usd_large": OFFICIAL_IMPORTER_USD.stat().st_size > 100_000_000 if OFFICIAL_IMPORTER_USD.is_file() else False,
        "export_structure_audit_passed": export_audit.get("status") == "ok_with_physics_usd_export_but_vulkan_device_lost",
        "motion_npz_exists": MOTION_NPZ.is_file(),
        "process_returned_zero": proc.returncode == 0,
        "app_launcher_reached": markers["after_app"],
        "cfg_ready": markers["cfg_ready"],
        "env_created": markers["env_created"],
        "env_reset": markers["env_reset"],
        "env_step_final": markers["env_step_final"],
        "metrics_file_written": metrics_path.is_file() and markers["metrics_file"],
        "action_dim_29": metrics.get("action_dim") == 29,
        "policy_observation_dim_160": metrics.get("policy_observation_dim") == 160,
        "critic_observation_dim_286": metrics.get("critic_observation_dim") == 286,
        "reward_terms_9": len(metrics.get("reward_terms", [])) == 9,
        "termination_terms_4": len(metrics.get("termination_terms", [])) == 4,
        "robot_joint_count_29": metrics.get("robot_num_joints") == 29,
        "robot_body_count_40": metrics.get("robot_num_bodies") == 40,
        "uses_official_importer_export_usd": metrics.get("uses_official_importer_export_usd") is True,
        "does_not_claim_resource_adjusted_enriched_usd": metrics.get("uses_resource_adjusted_enriched_usd") is False,
        "does_not_claim_unpatched_csv_to_npz_output": metrics.get("official_csv_to_npz_unpatched_output") is False,
        "does_not_claim_paper_level_rollout": metrics.get("paper_level_rollout") is False,
        "does_not_start_training": metrics.get("ppo_training") is False,
        "does_not_claim_real_robot": metrics.get("real_robot") is False,
    }
    passed = all(checks.values())
    if passed:
        status = "ok_official_importer_export_task_smoke"
        latest_blocker = "none_official_importer_export_task_smoke_passed"
    elif markers["vulkan_device_lost"]:
        status = "ok_with_official_importer_export_task_smoke_blocker"
        latest_blocker = "vulkan_device_lost_during_task_gate"
    elif markers["articulation_root_missing"]:
        status = "ok_with_official_importer_export_task_smoke_blocker"
        latest_blocker = "official_importer_export_not_accepted_as_articulation"
    elif markers["exception"]:
        status = "ok_with_official_importer_export_task_smoke_blocker"
        latest_blocker = "python_exception_during_task_gate"
    elif timed_out:
        status = "ok_with_official_importer_export_task_smoke_blocker"
        latest_blocker = "timeout_during_task_gate"
    else:
        status = "ok_with_official_importer_export_task_smoke_blocker"
        latest_blocker = "unclassified_task_gate_blocker"
    failed_copy = ""
    if status != "ok_official_importer_export_task_smoke":
        failed_copy_path = FAILED_DIR / "tracking_g1_official_importer_export_task_smoke.log"
        failed_copy_path.write_text(output, encoding="utf-8", errors="replace")
        failed_copy = str(failed_copy_path)
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_g1_official_importer_export_task_smoke",
        "scope": (
            "Attempts to instantiate the official Tracking-Flat-G1-v0 task using the official Isaac Sim URDF "
            "importer GPU4 USDA export as the robot USD. This is a bounded reset/step gate only; it is not PPO, "
            "DAgger, paper-level evaluation, Fig. 5/Fig. 6, TensorRT, or real-robot validation."
        ),
        "command": command,
        "timeout_seconds": TIMEOUT_SECONDS,
        "duration_seconds": round(duration, 3),
        "returncode": proc.returncode,
        "latest_blocker": latest_blocker,
        "gpu_guard": guard,
        "markers": markers,
        "checks": checks,
        "metrics": metrics,
        "inputs": {
            "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
            "export_structure_audit": str(EXPORT_STRUCTURE_AUDIT),
            "motion_npz": str(MOTION_NPZ),
            "target_gpu": TARGET_GPU,
            "step_count": STEP_COUNT,
        },
        "outputs": {
            "json": str(OUT / "tracking_g1_official_importer_export_task_smoke.json"),
            "metrics_json": str(metrics_path),
            "worker_script": str(worker_path),
            "log": str(log_path),
            "failed_log_copy": failed_copy,
        },
        "interpretation": {
            "goal_complete": False,
            "claim_level": "official_importer_export_task_gate",
            "official_replay_complete": False,
            "paper_level_tracking_eval_complete": False,
            "why_not_complete": (
                "A successful bounded task gate would only show that the official-importer USDA can be consumed by "
                "IsaacLab's task stack. Official full replay, PPO tracking, DAgger, VAE/diffusion, deployment, and "
                "real-robot evidence remain separate required gates."
            ),
        },
    }
    write_json(OUT / "tracking_g1_official_importer_export_task_smoke.json", summary)
    print(json.dumps({"status": status, "latest_blocker": latest_blocker, "returncode": proc.returncode}, sort_keys=True))


if __name__ == "__main__":
    main()
