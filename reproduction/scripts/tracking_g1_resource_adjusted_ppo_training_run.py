#!/usr/bin/env python3
"""Launch or preflight a resource-adjusted G1 PPO training run with GPU telemetry."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/g1_resource_adjusted_ppo_training_run"
LOG_DIR = ROOT / "logs/tracking_g1_resource_adjusted_ppo_training_run"
RUN_ROOT = ROOT / "res/runs/tracking_g1_resource_adjusted_ppo_training"
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
TRAIN_ENTRY = (
    ROOT
    / "res/tracking/g1_resource_adjusted_train_entry_diagnostic/"
    "tracking_g1_resource_adjusted_train_entry_diagnostic_audit.json"
)
CANDIDATE_GPUS = [4, 5, 6, 7]
MIN_FREE_MB = 20_000
MAX_BUSY_UTIL = 50
NUM_ENVS_PER_RANK = int(os.environ.get("BM_PPO_NUM_ENVS_PER_RANK", "512"))
MAX_ITERATIONS = int(os.environ.get("BM_PPO_MAX_ITERATIONS", "100"))
NUM_STEPS_PER_ENV = 24
SEED = int(os.environ.get("BM_PPO_SEED", "20260619"))


WORKER_CODE = r"""
import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
global_rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
args.device = f"cuda:{local_rank}"

print(f"BM_SENTINEL:rank={global_rank}:before_app:device={args.device}", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"BM_SENTINEL:rank={global_rank}:after_app", flush=True)

try:
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import torch
    import whole_body_tracking.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
    from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner

    enriched_usd = Path(os.environ["BM_ENRICHED_USD"])
    motion_file = Path(os.environ["BM_MOTION_FILE"])
    run_dir = Path(os.environ["BM_RUN_DIR"])
    rank_dir = run_dir / f"rank_{global_rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)

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
    env_cfg.seed = int(os.environ["BM_PPO_SEED"]) + global_rank

    agent_cfg = G1FlatPPORunnerCfg()
    agent_cfg.device = args.device
    agent_cfg.seed = int(os.environ["BM_PPO_SEED"]) + global_rank
    agent_cfg.max_iterations = int(os.environ["BM_MAX_ITERATIONS"])
    agent_cfg.num_steps_per_env = int(os.environ["BM_NUM_STEPS_PER_ENV"])
    agent_cfg.save_interval = max(1, min(50, agent_cfg.max_iterations))
    agent_cfg.empirical_normalization = True
    agent_cfg.logger = "tensorboard"
    agent_cfg.run_name = f"resource_adjusted_rank{global_rank}"

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print(f"BM_SENTINEL:rank={global_rank}:env_created:num_envs={env.unwrapped.num_envs}", flush=True)
    vec_env = RslRlVecEnvWrapper(env)
    runner = MotionOnPolicyRunner(
        vec_env,
        agent_cfg.to_dict(),
        log_dir=str(rank_dir),
        device=agent_cfg.device,
        registry_name=f"local:{motion_file}",
    )
    print(f"BM_SENTINEL:rank={global_rank}:runner_created", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"BM_SENTINEL:rank={global_rank}:learn_completed", flush=True)

    obs, extras = vec_env.get_observations()
    checkpoints = sorted(str(path) for path in rank_dir.glob("model_*.pt"))
    metrics = {
        "rank": global_rank,
        "world_size": world_size,
        "device": args.device,
        "num_envs": int(vec_env.num_envs),
        "num_actions": int(vec_env.num_actions),
        "num_obs": int(vec_env.num_obs),
        "num_privileged_obs": int(vec_env.num_privileged_obs),
        "num_steps_per_env": int(agent_cfg.num_steps_per_env),
        "max_iterations": int(agent_cfg.max_iterations),
        "current_learning_iteration": int(runner.current_learning_iteration),
        "tot_timesteps": int(runner.tot_timesteps),
        "tot_time": float(runner.tot_time),
        "robot_num_joints": int(vec_env.unwrapped.scene["robot"].num_joints),
        "robot_num_bodies": int(vec_env.unwrapped.scene["robot"].num_bodies),
        "post_train_policy_obs_shape": list(obs.shape),
        "post_train_critic_obs_shape": list(extras["observations"]["critic"].shape),
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
        "uses_resource_adjusted_usd": True,
        "official_csv_source": True,
        "official_csv_to_npz_output": False,
        "paper_level_training": False,
    }
    (rank_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    vec_env.close()
    print(f"BM_SENTINEL:rank={global_rank}:metrics_written={rank_dir / 'training_metrics.json'}", flush=True)
    os._exit(0)
except BaseException as exc:
    print(f"BM_SENTINEL:rank={global_rank}:exception={exc!r}", flush=True)
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
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
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
        if len(raw) < 6:
            continue
        index, name, mem_used, mem_total, util, power = [item.strip() for item in raw[:6]]
        mem_used_i = int(float(mem_used))
        mem_total_i = int(float(mem_total))
        rows.append(
            {
                "index": int(index),
                "name": name,
                "memory_used_mb": mem_used_i,
                "memory_total_mb": mem_total_i,
                "memory_free_mb": mem_total_i - mem_used_i,
                "utilization_gpu_percent": int(float(util)),
                "power_draw_w": float(power),
            }
        )
    return rows


def query_compute_processes() -> list[dict[str, Any]]:
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
        rows.append(
            {
                "gpu_uuid": raw[0].strip(),
                "pid": int(raw[1].strip()),
                "process_name": raw[2].strip(),
                "used_memory_mb": int(float(raw[3].strip())),
            }
        )
    return rows


def base_env(run_dir: Path, selected_cuda_visible_devices: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": selected_cuda_visible_devices,
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
            "BM_RUN_DIR": str(run_dir),
            "BM_NUM_ENVS_PER_RANK": str(NUM_ENVS_PER_RANK),
            "BM_MAX_ITERATIONS": str(MAX_ITERATIONS),
            "BM_NUM_STEPS_PER_ENV": str(NUM_STEPS_PER_ENV),
            "BM_PPO_SEED": str(SEED),
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
            os.environ["BM_SELECTED_GPU_CSV"],
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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    worker_path = OUT / "tracking_g1_resource_adjusted_ppo_worker.py"
    worker_path.write_text(textwrap.dedent(WORKER_CODE), encoding="utf-8")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RUN_ROOT / f"resource_adjusted_ppo_{timestamp}_seed{SEED}"
    gpu_snapshot = query_gpus()
    compute_processes = query_compute_processes()
    train_entry = load_json(TRAIN_ENTRY)
    available_gpus = [
        row["index"]
        for row in gpu_snapshot
        if "index" in row
        and row.get("memory_free_mb", 0) >= MIN_FREE_MB
        and row.get("utilization_gpu_percent", 100) <= MAX_BUSY_UTIL
    ]
    selected_gpus = available_gpus[:2]
    resource_ready = len(selected_gpus) >= 1
    selected_cuda_visible_devices = ",".join(str(gpu) for gpu in selected_gpus)
    selected_world_size = len(selected_gpus)
    input_checks = {
        "tracking_python_exists": TRACKING_PY.is_file(),
        "enriched_usd_exists": ENRICHED_USD.is_file(),
        "motion_npz_exists": CSV_MOTION_NPZ.is_file(),
        "train_entry_smoke_passed": train_entry.get("status") == "ok_resource_adjusted_train_entry_diagnostic",
        "candidate_gpu_count_4": len([row for row in gpu_snapshot if "index" in row]) == 4,
        "selected_gpu_count_at_least_1": selected_world_size >= 1,
        "selected_gpus_have_required_free_memory_and_low_utilization": resource_ready,
    }
    command = [
        str(TRACKING_PY),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={selected_world_size}",
        str(worker_path),
    ]
    telemetry_path = run_dir / "gpu_metrics.csv"
    log_path = LOG_DIR / "tracking_g1_resource_adjusted_ppo_training_run.log"
    run: dict[str, Any] = {
        "attempted_training": False,
        "command": command,
        "run_dir": str(run_dir),
        "log": str(log_path),
        "gpu_metrics_csv": str(telemetry_path),
    }

    if all(input_checks.values()):
        run_dir.mkdir(parents=True, exist_ok=True)
        os.environ["BM_SELECTED_GPU_CSV"] = selected_cuda_visible_devices
        monitor = start_gpu_monitor(telemetry_path)
        start = time.time()
        env = base_env(run_dir, selected_cuda_visible_devices)
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
        rank_metrics = []
        for metric_path in sorted(run_dir.glob("rank_*/training_metrics.json")):
            rank_metrics.append(load_json(metric_path))
        run.update(
            {
                "attempted_training": True,
                "returncode": returncode,
                "duration_seconds": round(time.time() - start, 3),
                "rank_metrics": rank_metrics,
                "checkpoint_count": sum(row.get("checkpoint_count", 0) for row in rank_metrics),
            }
        )
    else:
        run["reason_not_started"] = (
            "No GPU in candidate set 4-7 satisfied the configured free-memory/utilization preflight. The script did not start IsaacLab training "
            "to avoid interfering with existing compute jobs."
        )

    trained_ok = run.get("attempted_training") and run.get("returncode") == 0 and run.get("checkpoint_count", 0) > 0
    if trained_ok:
        status = "ok_resource_adjusted_ppo_training_completed"
    elif not all(input_checks.values()):
        status = "ok_with_gpu_resource_unavailable_before_training"
    else:
        status = "ok_with_resource_adjusted_ppo_training_blocker"

    summary = {
        "status": status,
        "experiment_type": "tracking_resource_adjusted_ppo_training_run",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Formalized resource-adjusted PPO training harness for the official Tracking-Flat-G1-v0 manager stack. "
            "It selects available GPUs from physical GPUs 4-7 via torch.distributed, records GPU telemetry, and "
            "writes checkpoints under res/runs. Because the asset remains resource-adjusted, even a completed run is "
            "not official BeyondMimic paper-level training."
        ),
        "config": {
            "candidate_physical_gpus": CANDIDATE_GPUS,
            "selected_physical_gpus": selected_gpus,
            "cuda_visible_devices": selected_cuda_visible_devices,
            "num_envs_per_rank": NUM_ENVS_PER_RANK,
            "world_size": selected_world_size,
            "total_num_envs": NUM_ENVS_PER_RANK * selected_world_size,
            "num_steps_per_env": NUM_STEPS_PER_ENV,
            "max_iterations": MAX_ITERATIONS,
            "seed": SEED,
            "min_free_mb_required_per_gpu": MIN_FREE_MB,
            "max_busy_util_percent_for_start": MAX_BUSY_UTIL,
        },
        "gpu_preflight": {
            "snapshot": gpu_snapshot,
            "compute_processes": compute_processes,
            "resource_ready": resource_ready,
            "available_gpus": available_gpus,
            "selected_gpus": selected_gpus,
        },
        "inputs": {
            "enriched_usd": str(ENRICHED_USD),
            "motion_npz": str(CSV_MOTION_NPZ),
            "train_entry_smoke": str(TRAIN_ENTRY),
        },
        "input_checks": input_checks,
        "run": run,
        "outputs": {
            "json": str(OUT / "tracking_g1_resource_adjusted_ppo_training_run.json"),
            "worker_script": str(worker_path),
            "run_dir": str(run_dir),
            "log": str(log_path),
            "gpu_metrics_csv": str(telemetry_path),
        },
        "interpretation": {
            "goal_complete": False,
            "official_ppo_training_complete": False,
            "paper_level_tracking_training_complete": False,
            "resource_adjusted_training_harness_ready": True,
            "why_not_paper_level": (
                "The task still uses the generated resource-adjusted enriched USD and resource-adjusted motion.npz "
                "from official CSV data, not official csv_to_npz/replay output. Current status may also be a preflight "
                "defer if all candidate GPUs 4-7 are occupied."
            ),
        },
    }
    (OUT / "tracking_g1_resource_adjusted_ppo_training_run.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "resource_ready": resource_ready,
                "attempted_training": run["attempted_training"],
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if status == "ok_with_resource_adjusted_ppo_training_blocker":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
