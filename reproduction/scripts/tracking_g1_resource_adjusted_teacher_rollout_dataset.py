#!/usr/bin/env python3
"""Collect a resource-adjusted teacher rollout dataset from the local PPO checkpoint.

This is intentionally labeled as resource-adjusted virtual evidence. It is not
an official BeyondMimic DAgger rollout dataset and not a paper-level result.
"""

from __future__ import annotations

import csv
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
OUT = ROOT / "res/tracking/g1_resource_adjusted_teacher_rollout_dataset"
LOG_DIR = ROOT / "logs/tracking_g1_resource_adjusted_teacher_rollout_dataset"
RUN_ROOT = ROOT / "res/runs/tracking_g1_resource_adjusted_teacher_rollout_dataset"
GPU_GUARD_DIR = ROOT / "res/gpu_guard"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
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
TRAINING_RUN_JSON = (
    ROOT
    / "res/tracking/g1_resource_adjusted_ppo_training_run/"
    "tracking_g1_resource_adjusted_ppo_training_run.json"
)
CANDIDATE_GPUS = [4, 7]
MIN_FREE_MB = 20_000
MAX_BUSY_UTIL = 50
NUM_ENVS_PER_RANK = int(os.environ.get("BM_TEACHER_ROLLOUT_NUM_ENVS_PER_RANK", "512"))
ROLLOUT_STEPS = int(os.environ.get("BM_TEACHER_ROLLOUT_STEPS", "299"))
SEED = int(os.environ.get("BM_TEACHER_ROLLOUT_SEED", "20260621"))
WANGJC_PATH_MARKER = "/mnt/infini-data/test/wangjc/"


def cuda_visible_devices() -> str:
    return ",".join(str(gpu) for gpu in CANDIDATE_GPUS)


WORKER_CODE = r"""
import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
rank = int(os.environ.get("RANK", str(local_rank)))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
device = f"cuda:{local_rank}"

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
args.device = device

print(f"BM_SENTINEL:teacher_rollout:before_app:rank={rank}:device={args.device}", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"BM_SENTINEL:teacher_rollout:after_app:rank={rank}", flush=True)

try:
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import numpy as np
    import torch
    import whole_body_tracking.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

    enriched_usd = Path(os.environ["BM_ENRICHED_USD"])
    motion_file = Path(os.environ["BM_MOTION_FILE"])
    checkpoint = Path(os.environ["BM_CHECKPOINT"])
    run_dir = Path(os.environ["BM_RUN_DIR"])
    shard_dir = run_dir / f"rank_{rank}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_npz = shard_dir / "teacher_rollout_shard.npz"
    shard_metrics_path = shard_dir / "teacher_rollout_shard_metrics.json"

    env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = int(os.environ["BM_NUM_ENVS_PER_RANK"])
    env_cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=str(enriched_usd),
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
    env_cfg.commands.motion.motion_file = str(motion_file)
    env_cfg.commands.motion.debug_vis = False
    env_cfg.scene.contact_forces.debug_vis = False
    env_cfg.sim.device = args.device
    env_cfg.seed = int(os.environ["BM_TEACHER_ROLLOUT_SEED"]) + rank

    agent_cfg = G1FlatPPORunnerCfg()
    agent_cfg.device = args.device
    agent_cfg.seed = int(os.environ["BM_TEACHER_ROLLOUT_SEED"]) + rank
    agent_cfg.empirical_normalization = True
    agent_cfg.logger = "tensorboard"

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print(f"BM_SENTINEL:teacher_rollout:env_created:rank={rank}:num_envs={env.unwrapped.num_envs}", flush=True)
    vec_env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    loaded_infos = runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
    obs, extras = vec_env.get_observations()
    command = vec_env.unwrapped.command_manager.get_term("motion")
    rollout_steps = int(os.environ["BM_TEACHER_ROLLOUT_STEPS"])

    policy_obs = []
    critic_obs = []
    actions_out = []
    rewards_out = []
    dones_out = []
    timeouts_out = []
    motion_time_steps = []
    metric_series = {}
    episode_log_accum = {}
    metric_names = [
        "error_anchor_pos",
        "error_anchor_rot",
        "error_anchor_lin_vel",
        "error_anchor_ang_vel",
        "error_body_pos",
        "error_body_rot",
        "error_body_lin_vel",
        "error_body_ang_vel",
        "error_joint_pos",
        "error_joint_vel",
        "sampling_entropy",
        "sampling_top1_prob",
        "sampling_top1_bin",
    ]

    with torch.inference_mode():
        for step in range(rollout_steps):
            policy_obs.append(obs.detach().cpu().numpy().astype(np.float32))
            critic = extras["observations"]["critic"]
            critic_obs.append(critic.detach().cpu().numpy().astype(np.float32))
            if hasattr(command, "time_steps"):
                motion_time_steps.append(command.time_steps.detach().cpu().numpy().astype(np.int32))
            else:
                motion_time_steps.append(np.zeros((vec_env.num_envs,), dtype=np.int32))

            actions = policy(obs)
            actions_out.append(actions.detach().cpu().numpy().astype(np.float32))
            obs, rew, dones, step_extras = vec_env.step(actions)
            extras = step_extras
            rewards_out.append(rew.detach().cpu().numpy().astype(np.float32))
            dones_out.append(dones.detach().cpu().numpy().astype(np.bool_))
            timeout_tensor = step_extras.get("time_outs", torch.zeros_like(dones))
            timeouts_out.append(timeout_tensor.detach().cpu().numpy().astype(np.bool_))
            for name in metric_names:
                tensor = command.metrics.get(name)
                if tensor is None:
                    value = float("nan")
                else:
                    value = float(tensor.mean().detach().cpu())
                metric_series.setdefault(name, []).append(value)
            for key, value in step_extras.get("log", {}).items():
                try:
                    scalar = float(value.detach().mean().cpu()) if hasattr(value, "detach") else float(value)
                except Exception:
                    continue
                episode_log_accum.setdefault(key, []).append(scalar)

    policy_obs_arr = np.stack(policy_obs, axis=0)
    critic_obs_arr = np.stack(critic_obs, axis=0)
    actions_arr = np.stack(actions_out, axis=0)
    rewards_arr = np.stack(rewards_out, axis=0)
    dones_arr = np.stack(dones_out, axis=0)
    timeouts_arr = np.stack(timeouts_out, axis=0)
    motion_time_steps_arr = np.stack(motion_time_steps, axis=0)
    final_obs, final_extras = vec_env.get_observations()
    np.savez_compressed(
        shard_npz,
        policy_obs=policy_obs_arr,
        critic_obs=critic_obs_arr,
        actions=actions_arr,
        rewards=rewards_arr,
        dones=dones_arr,
        timeouts=timeouts_arr,
        motion_time_steps=motion_time_steps_arr,
        final_policy_obs=final_obs.detach().cpu().numpy().astype(np.float32),
        final_critic_obs=final_extras["observations"]["critic"].detach().cpu().numpy().astype(np.float32),
        rank=np.array([rank], dtype=np.int32),
        world_size=np.array([world_size], dtype=np.int32),
        seed=np.array([int(os.environ["BM_TEACHER_ROLLOUT_SEED"]) + rank], dtype=np.int32),
    )

    def summarize(values):
        values = list(values)
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "first": values[0],
            "last": values[-1],
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    metrics = {
        "status": "ok",
        "rank": rank,
        "world_size": world_size,
        "task": "Tracking-Flat-G1-v0",
        "checkpoint": str(checkpoint),
        "loaded_iteration": int(runner.current_learning_iteration),
        "loaded_infos_type": type(loaded_infos).__name__,
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "num_envs": int(vec_env.num_envs),
        "rollout_steps": rollout_steps,
        "total_env_steps": int(vec_env.num_envs) * rollout_steps,
        "num_actions": int(vec_env.num_actions),
        "num_obs": int(vec_env.num_obs),
        "num_privileged_obs": int(vec_env.num_privileged_obs),
        "policy_obs_shape": list(policy_obs_arr.shape),
        "critic_obs_shape": list(critic_obs_arr.shape),
        "actions_shape": list(actions_arr.shape),
        "rewards_shape": list(rewards_arr.shape),
        "dones_shape": list(dones_arr.shape),
        "motion_time_steps_shape": list(motion_time_steps_arr.shape),
        "reward_mean": float(rewards_arr.mean()),
        "reward_min": float(rewards_arr.min()),
        "reward_max": float(rewards_arr.max()),
        "done_count_total": int(dones_arr.sum()),
        "timeout_count_total": int(timeouts_arr.sum()),
        "action_abs_mean": float(np.abs(actions_arr).mean()),
        "action_abs_max": float(np.abs(actions_arr).max()),
        "motion_metrics": {name: summarize(values) for name, values in metric_series.items()},
        "episode_log_metrics": {name: summarize(values) for name, values in episode_log_accum.items()},
        "dataset_npz": str(shard_npz),
        "dataset_npz_size_bytes": shard_npz.stat().st_size,
        "uses_resource_adjusted_usd": True,
        "official_csv_source": True,
        "official_dagger_rollout_dataset": False,
        "paper_level_teacher_rollout_dataset": False,
    }
    shard_metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    vec_env.close()
    print(f"BM_SENTINEL:teacher_rollout:metrics_written:rank={rank}:{shard_metrics_path}", flush=True)
    os._exit(0)
except BaseException as exc:
    print(f"BM_SENTINEL:teacher_rollout:exception:rank={rank}:{exc!r}", flush=True)
    raise
finally:
    try:
        simulation_app.close(wait_for_replicator=False)
    except BaseException:
        pass
"""


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run(args: list[str], env: dict[str, str] | None = None, timeout: int | None = None) -> tuple[int, str]:
    proc = subprocess.run(
        args,
        cwd=ROOT,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout


def query_gpus() -> list[dict[str, Any]]:
    rc, out = run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
            "-i",
            ",".join(str(gpu) for gpu in CANDIDATE_GPUS),
        ],
        timeout=30,
    )
    if rc != 0:
        return [{"error": out.strip()}]
    rows = []
    for raw in csv.reader(out.strip().splitlines()):
        if len(raw) < 7:
            continue
        index, uuid, name, mem_used, mem_total, util, power = [item.strip() for item in raw[:7]]
        mem_used_i = int(float(mem_used))
        mem_total_i = int(float(mem_total))
        rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_used_mb": mem_used_i,
                "memory_total_mb": mem_total_i,
                "memory_free_mb": mem_total_i - mem_used_i,
                "utilization_gpu_percent": int(float(util)),
                "power_draw_w": float(power),
            }
        )
    return rows


def read_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace")
    except OSError:
        return ""


def query_compute_processes() -> list[dict[str, Any]]:
    uuid_to_index = {row.get("uuid"): row.get("index") for row in query_gpus() if row.get("uuid")}
    rc, out = run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
            "-i",
            ",".join(str(gpu) for gpu in CANDIDATE_GPUS),
        ],
        timeout=30,
    )
    if rc != 0:
        return [{"error": out.strip()}]
    rows = []
    for raw in csv.reader(out.strip().splitlines()):
        if len(raw) < 4:
            continue
        pid = int(raw[1].strip())
        rows.append(
            {
                "gpu_uuid": raw[0].strip(),
                "gpu_index": uuid_to_index.get(raw[0].strip()),
                "pid": pid,
                "process_name": raw[2].strip(),
                "cmdline": read_cmdline(pid),
                "used_memory_mb": int(float(raw[3].strip())),
            }
        )
    return rows


def terminate_wangjc_processes(processes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    terminated = []
    for proc in processes:
        marker_source = f"{proc.get('process_name', '')} {proc.get('cmdline', '')}"
        if WANGJC_PATH_MARKER not in marker_source:
            continue
        pid = int(proc["pid"])
        record = dict(proc)
        record["policy"] = "terminated_wangjc_process_on_gpu_4_or_7"
        try:
            os.kill(pid, signal.SIGTERM)
            record["sigterm_sent"] = True
            for _ in range(30):
                if not Path(f"/proc/{pid}").exists():
                    break
                time.sleep(1)
            if Path(f"/proc/{pid}").exists():
                os.kill(pid, signal.SIGKILL)
                record["sigkill_sent"] = True
            else:
                record["sigkill_sent"] = False
            record["error"] = ""
        except ProcessLookupError:
            record["sigterm_sent"] = False
            record["sigkill_sent"] = False
            record["error"] = "process_already_exited"
        except PermissionError as exc:
            record["sigterm_sent"] = False
            record["sigkill_sent"] = False
            record["error"] = f"permission_error={exc}"
        terminated.append(record)
    return terminated


def write_gpu_guard(timestamp: str) -> dict[str, Any]:
    GPU_GUARD_DIR.mkdir(parents=True, exist_ok=True)
    before_gpus = query_gpus()
    before_processes = query_compute_processes()
    terminated = terminate_wangjc_processes(before_processes)
    if terminated:
        time.sleep(5)
    after_gpus = query_gpus()
    after_processes = query_compute_processes()
    guard = {
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "GPU 4/7 guard for resource-adjusted teacher rollout collection.",
        "candidate_physical_gpus": CANDIDATE_GPUS,
        "policy": {
            "allowed_to_terminate_path_marker": WANGJC_PATH_MARKER,
            "only_gpu_4_7": True,
            "non_wangjc_processes_terminated": False,
        },
        "before": {"gpus": before_gpus, "compute_processes": before_processes},
        "terminated_processes": terminated,
        "after": {"gpus": after_gpus, "compute_processes": after_processes},
    }
    gpu_tag = "gpu" + "".join(str(gpu) for gpu in CANDIDATE_GPUS)
    path = GPU_GUARD_DIR / f"{timestamp}_{gpu_tag}_wangjc_teacher_rollout_guard.json"
    guard["path"] = str(path)
    path.write_text(json.dumps(guard, indent=2, sort_keys=True), encoding="utf-8")
    return guard


def select_checkpoint(training_run: dict[str, Any]) -> Path:
    run_dir = Path(training_run.get("outputs", {}).get("run_dir", ""))
    if run_dir.is_dir():
        candidates = sorted((run_dir / "rank_0").glob("model_*.pt"))
        if candidates:
            def key(path: Path) -> int:
                try:
                    return int(path.stem.split("_")[1])
                except Exception:
                    return -1

            return max(candidates, key=key)
    return ROOT / "res/runs/tracking_g1_resource_adjusted_ppo_training/resource_adjusted_ppo_20260618_182241_seed20260619/rank_0/model_99.pt"


def base_env(run_dir: Path, checkpoint: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": cuda_visible_devices(),
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
            "BM_CHECKPOINT": str(checkpoint),
            "BM_RUN_DIR": str(run_dir),
            "BM_NUM_ENVS_PER_RANK": str(NUM_ENVS_PER_RANK),
            "BM_TEACHER_ROLLOUT_STEPS": str(ROLLOUT_STEPS),
            "BM_TEACHER_ROLLOUT_SEED": str(SEED),
        }
    )
    return env


def start_gpu_monitor(path: Path) -> subprocess.Popen[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv",
            "-i",
            cuda_visible_devices(),
            "-l",
            "5",
            "-f",
            str(path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def summarize_gpu_metrics(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    per_gpu: dict[str, dict[str, Any]] = {}
    for row in rows:
        normalized = {key.strip(): value.strip() for key, value in row.items() if key is not None}
        index = str(normalized.get("index", "")).strip()
        if not index:
            continue
        mem_used = float(normalized.get("memory.used [MiB]", "0").split()[0])
        mem_total = float(normalized.get("memory.total [MiB]", "0").split()[0])
        util = float(normalized.get("utilization.gpu [%]", "0").split()[0])
        item = per_gpu.setdefault(index, {"samples": 0, "peak_memory_used_mb": 0.0, "utilization_samples": []})
        item["samples"] += 1
        item["peak_memory_used_mb"] = max(item["peak_memory_used_mb"], mem_used)
        item["memory_total_mb"] = mem_total
        item["utilization_samples"].append(util)
    for item in per_gpu.values():
        samples = item.pop("utilization_samples")
        item["mean_utilization_gpu_percent"] = sum(samples) / len(samples) if samples else 0.0
    return {"exists": True, "row_count": len(rows), "per_gpu": per_gpu}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    worker_path = OUT / "tracking_g1_resource_adjusted_teacher_rollout_worker.py"
    worker_path.write_text(textwrap.dedent(WORKER_CODE), encoding="utf-8")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RUN_ROOT / f"resource_adjusted_teacher_rollout_{timestamp}_seed{SEED}"
    training_run = load_json(TRAINING_RUN_JSON)
    checkpoint = select_checkpoint(training_run)
    gpu_guard = write_gpu_guard(timestamp)
    gpu_snapshot = query_gpus()
    compute_processes = query_compute_processes()
    selected_gpus = list(CANDIDATE_GPUS)
    resource_ready = (
        len([row for row in gpu_snapshot if row.get("index") in selected_gpus]) == 2
        and all(
            row.get("memory_free_mb", 0) >= MIN_FREE_MB and row.get("utilization_gpu_percent", 100) <= MAX_BUSY_UTIL
            for row in gpu_snapshot
            if row.get("index") in selected_gpus
        )
        and not [
            proc for proc in compute_processes
            if WANGJC_PATH_MARKER not in f"{proc.get('process_name', '')} {proc.get('cmdline', '')}"
        ]
    )
    input_checks = {
        "tracking_python_exists": TRACKING_PY.is_file(),
        "training_run_completed": training_run.get("status")
        in {
            "ok_resource_adjusted_ppo_training_completed",
            "ok_official_csv_loop_ppo_training_completed",
        },
        "checkpoint_exists": checkpoint.is_file(),
        "enriched_usd_exists": ENRICHED_USD.is_file(),
        "motion_npz_exists": CSV_MOTION_NPZ.is_file(),
        "candidate_gpu_count_matches_config": len([row for row in gpu_snapshot if "index" in row])
        == len(CANDIDATE_GPUS),
        "selected_gpus_match_config": selected_gpus == CANDIDATE_GPUS,
        "selected_gpus_have_required_free_memory_and_low_utilization": resource_ready,
    }
    command = [
        str(TRACKING_PY),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=2",
        str(worker_path),
    ]
    telemetry_path = run_dir / "gpu_metrics.csv"
    log_path = LOG_DIR / "tracking_g1_resource_adjusted_teacher_rollout_dataset.log"
    rollout_run: dict[str, Any] = {
        "attempted_rollout": False,
        "command": command,
        "run_dir": str(run_dir),
        "log": str(log_path),
        "gpu_metrics_csv": str(telemetry_path),
    }

    if all(input_checks.values()):
        run_dir.mkdir(parents=True, exist_ok=True)
        monitor = start_gpu_monitor(telemetry_path)
        start = time.time()
        env = base_env(run_dir, checkpoint)
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            returncode = proc.wait()
        monitor.terminate()
        try:
            monitor.wait(timeout=20)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait(timeout=20)
        shard_metrics = [load_json(path) for path in sorted(run_dir.glob("rank_*/teacher_rollout_shard_metrics.json"))]
        shard_npz_paths = sorted(str(path) for path in run_dir.glob("rank_*/teacher_rollout_shard.npz"))
        rollout_run.update(
            {
                "attempted_rollout": True,
                "returncode": returncode,
                "duration_seconds": round(time.time() - start, 3),
                "shard_metrics": shard_metrics,
                "shard_count": len(shard_metrics),
                "shard_npz_paths": shard_npz_paths,
                "shard_npz_total_size_bytes": sum(Path(path).stat().st_size for path in shard_npz_paths),
                "gpu_metrics_summary": summarize_gpu_metrics(telemetry_path),
            }
        )
    else:
        rollout_run["reason_not_started"] = (
            f"GPU set {CANDIDATE_GPUS} was not fully free after the wangjc-only guard, or required checkpoint/USD/motion inputs were "
            "missing. The script did not start IsaacLab rollout collection."
        )

    rollout_ok = (
        rollout_run.get("attempted_rollout")
        and rollout_run.get("returncode") == 0
        and rollout_run.get("shard_count") == 2
        and all(row.get("status") == "ok" for row in rollout_run.get("shard_metrics", []))
    )
    if rollout_ok:
        status = "ok_resource_adjusted_teacher_rollout_dataset_completed"
    elif not all(input_checks.values()):
        status = "ok_with_gpu_resource_unavailable_before_teacher_rollout"
    else:
        status = "failed_resource_adjusted_teacher_rollout_dataset"

    total_env_steps = sum(row.get("total_env_steps", 0) for row in rollout_run.get("shard_metrics", []))
    summary = {
        "status": status,
        "experiment_type": "tracking_resource_adjusted_teacher_rollout_dataset",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Collects policy observation, critic observation, action, reward, done, timeout, and motion timestep "
            "arrays from the local resource-adjusted PPO teacher checkpoint in the official Tracking-Flat-G1-v0 task "
            "stack. This is a virtual resource-adjusted teacher dataset candidate for downstream VAE/state-latent "
            "experiments, not an official DAgger dataset."
        ),
        "config": {
            "candidate_physical_gpus": CANDIDATE_GPUS,
            "selected_physical_gpus": selected_gpus,
            "cuda_visible_devices": cuda_visible_devices(),
            "world_size": 2,
            "num_envs_per_rank": NUM_ENVS_PER_RANK,
            "total_num_envs": NUM_ENVS_PER_RANK * 2,
            "rollout_steps": ROLLOUT_STEPS,
            "expected_total_env_steps": NUM_ENVS_PER_RANK * 2 * ROLLOUT_STEPS,
            "seed": SEED,
            "min_free_mb_required_per_gpu": MIN_FREE_MB,
            "max_busy_util_percent_for_start": MAX_BUSY_UTIL,
        },
        "gpu_guard": gpu_guard,
        "gpu_preflight": {
            "snapshot_after_guard": gpu_snapshot,
            "compute_processes_after_guard": compute_processes,
            "resource_ready": resource_ready,
            "selected_gpus": selected_gpus,
        },
        "inputs": {
            "training_run_json": str(TRAINING_RUN_JSON),
            "checkpoint": str(checkpoint),
            "enriched_usd": str(ENRICHED_USD),
            "motion_npz": str(CSV_MOTION_NPZ),
        },
        "input_checks": input_checks,
        "run": rollout_run,
        "outputs": {
            "json": str(OUT / "tracking_g1_resource_adjusted_teacher_rollout_dataset.json"),
            "worker_script": str(worker_path),
            "run_dir": str(run_dir),
            "log": str(log_path),
            "gpu_metrics_csv": str(telemetry_path),
            "gpu_guard_json": gpu_guard["path"],
        },
        "aggregate_metrics": {
            "total_env_steps": total_env_steps,
            "shard_count": rollout_run.get("shard_count", 0),
            "dataset_npz_total_size_bytes": rollout_run.get("shard_npz_total_size_bytes", 0),
            "reward_mean_by_rank": [row.get("reward_mean") for row in rollout_run.get("shard_metrics", [])],
            "done_count_total": sum(row.get("done_count_total", 0) for row in rollout_run.get("shard_metrics", [])),
            "timeout_count_total": sum(row.get("timeout_count_total", 0) for row in rollout_run.get("shard_metrics", [])),
        },
        "interpretation": {
            "goal_complete": False,
            "official_dagger_dataset_complete": False,
            "paper_level_teacher_rollout_dataset_complete": False,
            "resource_adjusted_teacher_rollout_dataset_complete": bool(rollout_ok),
            "why_not_paper_level": (
                "The dataset comes from a locally trained resource-adjusted PPO checkpoint and generated "
                "resource-adjusted USD/motion path. The paper's true DAgger rollouts and official teacher policy "
                "checkpoints are not public in this workspace."
            ),
        },
    }
    (OUT / "tracking_g1_resource_adjusted_teacher_rollout_dataset.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "json": summary["outputs"]["json"],
                "attempted_rollout": rollout_run["attempted_rollout"],
                "shard_count": rollout_run.get("shard_count", 0),
                "total_env_steps": total_env_steps,
            },
            sort_keys=True,
        )
    )
    if status == "failed_resource_adjusted_teacher_rollout_dataset":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
