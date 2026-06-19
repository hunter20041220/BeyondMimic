#!/usr/bin/env python3
"""Evaluate the resource-adjusted G1 PPO checkpoint with the official task stack."""

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
OUT = ROOT / "res/tracking/g1_resource_adjusted_ppo_checkpoint_eval"
LOG_DIR = ROOT / "logs/tracking_g1_resource_adjusted_ppo_checkpoint_eval"
RUN_ROOT = ROOT / "res/runs/tracking_g1_resource_adjusted_ppo_checkpoint_eval"
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
CANDIDATE_GPUS = [
    int(item.strip())
    for item in os.environ.get("BM_PPO_EVAL_CANDIDATE_GPUS", "4,7").split(",")
    if item.strip()
]
MIN_FREE_MB = 20_000
MAX_BUSY_UTIL = 50
VISIBLE_GPU_LIMIT = int(os.environ.get("BM_PPO_EVAL_VISIBLE_GPU_LIMIT", "2"))
NUM_ENVS = int(os.environ.get("BM_PPO_EVAL_NUM_ENVS", "512"))
EVAL_STEPS = int(os.environ.get("BM_PPO_EVAL_STEPS", "299"))
SEED = int(os.environ.get("BM_PPO_EVAL_SEED", "20260620"))


WORKER_CODE = r"""
import argparse
import csv
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
args.device = "cuda:0"

print(f"BM_SENTINEL:eval:before_app:device={args.device}", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("BM_SENTINEL:eval:after_app", flush=True)

try:
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import torch
    import whole_body_tracking.tasks  # noqa: F401
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

    enriched_usd = Path(os.environ["BM_ENRICHED_USD"])
    motion_file = Path(os.environ["BM_MOTION_FILE"])
    checkpoint = Path(os.environ["BM_CHECKPOINT"])
    run_dir = Path(os.environ["BM_RUN_DIR"])
    metrics_path = run_dir / "eval_metrics.json"
    timeseries_path = run_dir / "eval_timeseries.csv"
    run_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = int(os.environ["BM_NUM_ENVS"])
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
    env_cfg.seed = int(os.environ["BM_EVAL_SEED"])

    agent_cfg = G1FlatPPORunnerCfg()
    agent_cfg.device = args.device
    agent_cfg.seed = int(os.environ["BM_EVAL_SEED"])
    agent_cfg.empirical_normalization = True
    agent_cfg.logger = "tensorboard"

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print(f"BM_SENTINEL:eval:env_created:num_envs={env.unwrapped.num_envs}", flush=True)
    vec_env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    loaded_infos = runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
    obs, extras = vec_env.get_observations()
    command = vec_env.unwrapped.command_manager.get_term("motion")

    reward_means = []
    reward_mins = []
    reward_maxs = []
    done_counts = []
    timeout_counts = []
    action_abs_means = []
    action_abs_maxs = []
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
    fieldnames = [
        "step",
        "reward_mean",
        "reward_min",
        "reward_max",
        "done_count",
        "timeout_count",
        "action_abs_mean",
        "action_abs_max",
    ] + metric_names

    with timeseries_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        with torch.inference_mode():
            for step in range(int(os.environ["BM_EVAL_STEPS"])):
                actions = policy(obs)
                obs, rew, dones, step_extras = vec_env.step(actions)
                row = {
                    "step": step,
                    "reward_mean": float(rew.mean().detach().cpu()),
                    "reward_min": float(rew.min().detach().cpu()),
                    "reward_max": float(rew.max().detach().cpu()),
                    "done_count": int(dones.sum().detach().cpu()),
                    "timeout_count": int(step_extras.get("time_outs", torch.zeros_like(dones)).sum().detach().cpu()),
                    "action_abs_mean": float(actions.abs().mean().detach().cpu()),
                    "action_abs_max": float(actions.abs().max().detach().cpu()),
                }
                for name in metric_names:
                    tensor = command.metrics.get(name)
                    value = float(tensor.mean().detach().cpu()) if tensor is not None else float("nan")
                    row[name] = value
                    metric_series.setdefault(name, []).append(value)
                writer.writerow(row)

                reward_means.append(row["reward_mean"])
                reward_mins.append(row["reward_min"])
                reward_maxs.append(row["reward_max"])
                done_counts.append(row["done_count"])
                timeout_counts.append(row["timeout_count"])
                action_abs_means.append(row["action_abs_mean"])
                action_abs_maxs.append(row["action_abs_max"])
                for key, value in step_extras.get("log", {}).items():
                    try:
                        scalar = float(value.detach().mean().cpu()) if hasattr(value, "detach") else float(value)
                    except Exception:
                        continue
                    episode_log_accum.setdefault(key, []).append(scalar)

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

    obs, extras = vec_env.get_observations()
    metrics = {
        "status": "ok",
        "task": "Tracking-Flat-G1-v0",
        "checkpoint": str(checkpoint),
        "loaded_iteration": int(runner.current_learning_iteration),
        "loaded_infos_type": type(loaded_infos).__name__,
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "num_envs": int(vec_env.num_envs),
        "eval_steps": int(os.environ["BM_EVAL_STEPS"]),
        "total_env_steps": int(vec_env.num_envs) * int(os.environ["BM_EVAL_STEPS"]),
        "num_actions": int(vec_env.num_actions),
        "num_obs": int(vec_env.num_obs),
        "num_privileged_obs": int(vec_env.num_privileged_obs),
        "policy_obs_shape": list(obs.shape),
        "critic_obs_shape": list(extras["observations"]["critic"].shape),
        "robot_num_joints": int(vec_env.unwrapped.scene["robot"].num_joints),
        "robot_num_bodies": int(vec_env.unwrapped.scene["robot"].num_bodies),
        "reward": {
            "mean_over_steps": summarize(reward_means),
            "min_over_steps": summarize(reward_mins),
            "max_over_steps": summarize(reward_maxs),
        },
        "done_count_total": int(sum(done_counts)),
        "timeout_count_total": int(sum(timeout_counts)),
        "action_abs_mean_over_steps": summarize(action_abs_means),
        "action_abs_max_over_steps": summarize(action_abs_maxs),
        "motion_metrics": {name: summarize(values) for name, values in metric_series.items()},
        "episode_log_metrics": {name: summarize(values) for name, values in episode_log_accum.items()},
        "uses_resource_adjusted_usd": True,
        "official_csv_source": True,
        "official_csv_to_npz_output": False,
        "official_replay_output": False,
        "paper_level_tracking_eval": False,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    vec_env.close()
    print(f"BM_SENTINEL:eval:metrics_written={metrics_path}", flush=True)
    os._exit(0)
except BaseException as exc:
    print(f"BM_SENTINEL:eval:exception={exc!r}", flush=True)
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


def classify_process_policy(processes: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify GPU occupants without killing unrelated shared-server jobs."""
    own_project_processes = []
    external_processes = []
    for proc in processes:
        name = proc.get("process_name", "")
        if "BeyondMimic" in name or "/mnt/infini-data/test/BeyondMimic/" in name:
            own_project_processes.append(proc)
        else:
            external_processes.append(proc)
    return {
        "auto_kill_enabled": False,
        "auto_kill_reason": (
            "The audit records GPU occupants but does not kill non-BeyondMimic processes automatically on the shared "
            "server. Only stale project-owned processes should be manually inspected before termination."
        ),
        "own_project_processes": own_project_processes,
        "external_processes": external_processes,
        "external_process_count": len(external_processes),
    }


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


def base_env(run_dir: Path, selected_cuda_visible_devices: str, checkpoint: Path) -> dict[str, str]:
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
            "BM_CHECKPOINT": str(checkpoint),
            "BM_RUN_DIR": str(run_dir),
            "BM_NUM_ENVS": str(NUM_ENVS),
            "BM_EVAL_STEPS": str(EVAL_STEPS),
            "BM_EVAL_SEED": str(SEED),
        }
    )
    return env


def start_gpu_monitor(path: Path, selected_cuda_visible_devices: str) -> subprocess.Popen[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv",
            "-i",
            selected_cuda_visible_devices,
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
    worker_path = OUT / "tracking_g1_resource_adjusted_ppo_checkpoint_eval_worker.py"
    worker_path.write_text(textwrap.dedent(WORKER_CODE), encoding="utf-8")

    training_run = load_json(TRAINING_RUN_JSON)
    checkpoint = select_checkpoint(training_run)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RUN_ROOT / f"resource_adjusted_ppo_eval_{timestamp}_seed{SEED}"
    gpu_snapshot = query_gpus()
    compute_processes = query_compute_processes()
    process_policy = classify_process_policy(compute_processes)
    available_gpus = [
        row["index"]
        for row in gpu_snapshot
        if "index" in row
        and row.get("memory_free_mb", 0) >= MIN_FREE_MB
        and row.get("utilization_gpu_percent", 100) <= MAX_BUSY_UTIL
    ]
    selected_gpus = available_gpus[:VISIBLE_GPU_LIMIT]
    selected_cuda_visible_devices = ",".join(str(gpu) for gpu in selected_gpus)
    resource_ready = len(selected_gpus) >= 1

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
        "candidate_gpu_count_2": len([row for row in gpu_snapshot if "index" in row]) == len(CANDIDATE_GPUS),
        "selected_gpu_count_at_least_1": len(selected_gpus) >= 1,
        "selected_gpus_have_required_free_memory_and_low_utilization": resource_ready,
    }

    telemetry_path = run_dir / "gpu_metrics.csv"
    log_path = LOG_DIR / "tracking_g1_resource_adjusted_ppo_checkpoint_eval.log"
    metrics_path = run_dir / "eval_metrics.json"
    timeseries_path = run_dir / "eval_timeseries.csv"
    command = [str(TRACKING_PY), str(worker_path)]
    eval_run: dict[str, Any] = {
        "attempted_eval": False,
        "command": command,
        "run_dir": str(run_dir),
        "log": str(log_path),
        "gpu_metrics_csv": str(telemetry_path),
        "metrics_json": str(metrics_path),
        "timeseries_csv": str(timeseries_path),
    }

    if all(input_checks.values()):
        run_dir.mkdir(parents=True, exist_ok=True)
        monitor = start_gpu_monitor(telemetry_path, selected_cuda_visible_devices)
        start = time.time()
        env = base_env(run_dir, selected_cuda_visible_devices, checkpoint)
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
        metrics = load_json(metrics_path)
        eval_run.update(
            {
                "attempted_eval": True,
                "returncode": returncode,
                "duration_seconds": round(time.time() - start, 3),
                "metrics": metrics,
                "metrics_exists": metrics_path.is_file(),
                "timeseries_exists": timeseries_path.is_file(),
            }
        )
    else:
        eval_run["reason_not_started"] = (
            "No candidate GPU in physical set 4/7 satisfied the free-memory/utilization preflight, or required "
            "training/checkpoint inputs were missing. The script did not start IsaacLab evaluation."
        )

    eval_ok = (
        eval_run.get("attempted_eval")
        and eval_run.get("returncode") == 0
        and eval_run.get("metrics_exists")
        and eval_run.get("metrics", {}).get("status") == "ok"
    )
    if eval_ok:
        status = "ok_resource_adjusted_ppo_checkpoint_eval_completed"
    elif not all(input_checks.values()):
        status = "ok_with_resource_unavailable_before_eval"
    else:
        status = "failed_resource_adjusted_ppo_checkpoint_eval"

    summary = {
        "status": status,
        "experiment_type": "tracking_resource_adjusted_ppo_checkpoint_eval",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Loads the resource-adjusted PPO checkpoint with the official RSL-RL OnPolicyRunner API and evaluates it "
            "inside the official Tracking-Flat-G1-v0 task stack on the generated resource-adjusted G1 USD and "
            "official-CSV-derived motion. This is virtual checkpoint evaluation, not official paper-level tracking."
        ),
        "config": {
            "candidate_physical_gpus": CANDIDATE_GPUS,
            "selected_physical_gpus": selected_gpus,
            "cuda_visible_devices": selected_cuda_visible_devices,
            "visible_gpu_limit": VISIBLE_GPU_LIMIT,
            "num_envs": NUM_ENVS,
            "eval_steps": EVAL_STEPS,
            "total_env_steps": NUM_ENVS * EVAL_STEPS,
            "seed": SEED,
            "min_free_mb_required_per_gpu": MIN_FREE_MB,
            "max_busy_util_percent_for_start": MAX_BUSY_UTIL,
        },
        "gpu_preflight": {
            "snapshot": gpu_snapshot,
            "compute_processes": compute_processes,
            "process_policy": process_policy,
            "resource_ready": resource_ready,
            "available_gpus": available_gpus,
            "selected_gpus": selected_gpus,
        },
        "inputs": {
            "training_run_json": str(TRAINING_RUN_JSON),
            "checkpoint": str(checkpoint),
            "enriched_usd": str(ENRICHED_USD),
            "motion_npz": str(CSV_MOTION_NPZ),
        },
        "input_checks": input_checks,
        "run": eval_run,
        "outputs": {
            "json": str(OUT / "tracking_g1_resource_adjusted_ppo_checkpoint_eval.json"),
            "worker_script": str(worker_path),
            "run_dir": str(run_dir),
            "log": str(log_path),
            "gpu_metrics_csv": str(telemetry_path),
            "metrics_json": str(metrics_path),
            "timeseries_csv": str(timeseries_path),
        },
        "interpretation": {
            "goal_complete": False,
            "official_tracking_eval_complete": False,
            "paper_level_tracking_eval_complete": False,
            "resource_adjusted_checkpoint_eval_complete": bool(eval_ok),
            "why_not_paper_level": (
                "The evaluated checkpoint was trained on a generated resource-adjusted USD and resource-adjusted "
                "motion.npz derived from official CSV data. It is not produced by the official converter/replay path "
                "and is not a paper-scale teacher-policy evaluation."
            ),
        },
    }
    (OUT / "tracking_g1_resource_adjusted_ppo_checkpoint_eval.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": status, "json": summary["outputs"]["json"], "attempted_eval": eval_run["attempted_eval"]}, sort_keys=True))
    if status == "failed_resource_adjusted_ppo_checkpoint_eval":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
