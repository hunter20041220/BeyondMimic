#!/usr/bin/env python3
"""Evaluate local VAE-reconstructed teacher actions in IsaacLab closed loop.

This runs the official-csv-loop tracking task with the local PPO teacher policy.
At each step the teacher action is encoded by the local official-csv-loop action
VAE, decoded from the posterior mean, and the decoded action is sent to the
environment. This is a VAE action-reconstruction closed-loop evaluation, not
autonomous diffusion guidance and not paper-level BeyondMimic evidence.
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
OUT = ROOT / "res/level_c/official_csv_loop_vae_closed_loop_rollout_eval"
LOG_DIR = ROOT / "logs/tracking_g1_official_csv_loop_vae_closed_loop_rollout_eval"
FAILED_DIR = ROOT / "res/failed_runs/tracking_g1_official_csv_loop_vae_closed_loop_rollout_eval"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_csv_loop_vae_closed_loop_rollout_eval"
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
OFFICIAL_LOOP_MOTION_NPZ = (
    ROOT
    / "res/tracking/official_csv_to_npz_loop_with_enriched_usd/"
    "walk1_subject1_frames_1_180_official_loop_enriched_usd_motion.npz"
)
TRAINING_RUN_JSON = (
    ROOT
    / "res/tracking/g1_official_csv_loop_ppo_training_run/"
    "tracking_g1_official_csv_loop_ppo_training_run.json"
)
TEACHER_ROLLOUT_JSON = (
    ROOT
    / "res/tracking/g1_official_csv_loop_teacher_rollout_dataset/"
    "tracking_g1_official_csv_loop_teacher_rollout_dataset.json"
)
VAE_TRAINING_JSON = (
    ROOT
    / "res/level_c/official_csv_loop_teacher_rollout_vae_training/"
    "level_c_official_csv_loop_teacher_rollout_vae_training.json"
)
CANDIDATE_GPUS = [4, 7]
MIN_FREE_MB = 20_000
MAX_BUSY_UTIL = 90
NUM_ENVS_PER_RANK = int(os.environ.get("BM_VAE_CLOSED_LOOP_NUM_ENVS_PER_RANK", "1024"))
ROLLOUT_STEPS = int(os.environ.get("BM_VAE_CLOSED_LOOP_STEPS", "299"))
SEED = int(os.environ.get("BM_VAE_CLOSED_LOOP_SEED", "20260639"))
WANGJC_PATH_MARKER = "/mnt/infini-data/test/wangjc/"


WORKER_CODE = r"""
import argparse
import csv
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

print(f"BM_SENTINEL:vae_closed_loop:before_app:rank={rank}:device={args.device}", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(f"BM_SENTINEL:vae_closed_loop:after_app:rank={rank}", flush=True)

try:
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import numpy as np
    import torch
    from torch import nn
    import whole_body_tracking.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

    class ConditionalActionVAE(nn.Module):
        def __init__(self, obs_dim, action_dim, latent_dim, hidden_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, latent_dim * 2),
            )
            self.decoder = nn.Sequential(
                nn.Linear(obs_dim + latent_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, action_dim),
            )

        def encode(self, obs, action):
            mu_logvar = self.encoder(torch.cat([obs, action], dim=-1))
            return torch.chunk(mu_logvar, 2, dim=-1)

        def decode(self, obs, latent):
            return self.decoder(torch.cat([obs, latent], dim=-1))

    enriched_usd = Path(os.environ["BM_ENRICHED_USD"])
    motion_file = Path(os.environ["BM_MOTION_FILE"])
    checkpoint = Path(os.environ["BM_CHECKPOINT"])
    vae_checkpoint = Path(os.environ["BM_VAE_CHECKPOINT"])
    run_dir = Path(os.environ["BM_RUN_DIR"])
    rollout_steps = int(os.environ["BM_ROLLOUT_STEPS"])
    shard_dir = run_dir / f"rank_{rank}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_metrics_path = shard_dir / "vae_closed_loop_rollout_metrics.json"
    shard_timeseries_path = shard_dir / "vae_closed_loop_rollout_timeseries.csv"
    shard_npz = shard_dir / "vae_closed_loop_rollout_summary_arrays.npz"

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
    env_cfg.seed = int(os.environ["BM_SEED"]) + rank

    agent_cfg = G1FlatPPORunnerCfg()
    agent_cfg.device = args.device
    agent_cfg.seed = int(os.environ["BM_SEED"]) + rank
    agent_cfg.empirical_normalization = True
    agent_cfg.logger = "tensorboard"

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print(f"BM_SENTINEL:vae_closed_loop:env_created:rank={rank}:num_envs={env.unwrapped.num_envs}", flush=True)
    vec_env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    loaded_infos = runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)

    vae_payload = torch.load(vae_checkpoint, map_location="cpu")
    vae_cfg = vae_payload["config"]
    vae = ConditionalActionVAE(
        vae_cfg["obs_dim"],
        vae_cfg["action_dim"],
        vae_cfg["latent_dim"],
        vae_cfg["hidden_dim"],
    ).to(vec_env.unwrapped.device)
    vae.load_state_dict(vae_payload["model_state_dict"])
    vae.eval()

    obs, extras = vec_env.get_observations()
    command = vec_env.unwrapped.command_manager.get_term("motion")
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
        "teacher_action_abs_mean",
        "vae_action_abs_mean",
        "teacher_vae_action_mse",
        "teacher_vae_action_abs_error_mean",
        "latent_mu_abs_mean",
    ] + metric_names

    rows = []
    reward_means = []
    done_counts = []
    timeout_counts = []
    teacher_vae_mse = []
    teacher_vae_abs = []
    latent_abs = []
    teacher_action_abs = []
    vae_action_abs = []
    metric_series = {name: [] for name in metric_names}

    with shard_timeseries_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        with torch.no_grad():
            for step in range(rollout_steps):
                teacher_action = policy(obs)
                mu, logvar = vae.encode(obs, teacher_action)
                vae_action = vae.decode(obs, mu)
                action_delta = vae_action - teacher_action
                obs, rew, dones, step_extras = vec_env.step(vae_action)
                row = {
                    "step": step,
                    "reward_mean": float(rew.mean().detach().cpu()),
                    "reward_min": float(rew.min().detach().cpu()),
                    "reward_max": float(rew.max().detach().cpu()),
                    "done_count": int(dones.sum().detach().cpu()),
                    "timeout_count": int(step_extras.get("time_outs", torch.zeros_like(dones)).sum().detach().cpu()),
                    "teacher_action_abs_mean": float(teacher_action.abs().mean().detach().cpu()),
                    "vae_action_abs_mean": float(vae_action.abs().mean().detach().cpu()),
                    "teacher_vae_action_mse": float(torch.mean(action_delta.square()).detach().cpu()),
                    "teacher_vae_action_abs_error_mean": float(torch.mean(action_delta.abs()).detach().cpu()),
                    "latent_mu_abs_mean": float(mu.abs().mean().detach().cpu()),
                }
                for name in metric_names:
                    tensor = command.metrics.get(name)
                    value = float(tensor.mean().detach().cpu()) if tensor is not None else float("nan")
                    row[name] = value
                    metric_series[name].append(value)
                writer.writerow(row)
                rows.append(row)
                reward_means.append(row["reward_mean"])
                done_counts.append(row["done_count"])
                timeout_counts.append(row["timeout_count"])
                teacher_vae_mse.append(row["teacher_vae_action_mse"])
                teacher_vae_abs.append(row["teacher_vae_action_abs_error_mean"])
                latent_abs.append(row["latent_mu_abs_mean"])
                teacher_action_abs.append(row["teacher_action_abs_mean"])
                vae_action_abs.append(row["vae_action_abs_mean"])
                if (step + 1) % 50 == 0 or (step + 1) == rollout_steps:
                    print(f"BM_SENTINEL:vae_closed_loop:rank={rank}:step={step + 1}/{rollout_steps}", flush=True)

    def summarize(values):
        values = list(values)
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "first": float(values[0]),
            "last": float(values[-1]),
            "mean": float(sum(values) / len(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }

    np.savez_compressed(
        shard_npz,
        reward_mean=np.asarray(reward_means, dtype=np.float32),
        done_count=np.asarray(done_counts, dtype=np.int32),
        timeout_count=np.asarray(timeout_counts, dtype=np.int32),
        teacher_vae_action_mse=np.asarray(teacher_vae_mse, dtype=np.float32),
        teacher_vae_action_abs_error_mean=np.asarray(teacher_vae_abs, dtype=np.float32),
        latent_mu_abs_mean=np.asarray(latent_abs, dtype=np.float32),
        teacher_action_abs_mean=np.asarray(teacher_action_abs, dtype=np.float32),
        vae_action_abs_mean=np.asarray(vae_action_abs, dtype=np.float32),
    )
    metrics = {
        "status": "ok",
        "rank": rank,
        "world_size": world_size,
        "task": "Tracking-Flat-G1-v0",
        "checkpoint": str(checkpoint),
        "vae_checkpoint": str(vae_checkpoint),
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
        "robot_num_joints": int(vec_env.unwrapped.scene["robot"].num_joints),
        "robot_num_bodies": int(vec_env.unwrapped.scene["robot"].num_bodies),
        "vae_config": vae_cfg,
        "reward_mean": summarize(reward_means),
        "done_count_total": int(sum(done_counts)),
        "timeout_count_total": int(sum(timeout_counts)),
        "teacher_vae_action_mse": summarize(teacher_vae_mse),
        "teacher_vae_action_abs_error_mean": summarize(teacher_vae_abs),
        "teacher_action_abs_mean": summarize(teacher_action_abs),
        "vae_action_abs_mean": summarize(vae_action_abs),
        "latent_mu_abs_mean": summarize(latent_abs),
        "motion_metrics": {name: summarize(values) for name, values in metric_series.items()},
        "timeseries_csv": str(shard_timeseries_path),
        "summary_npz": str(shard_npz),
        "summary_npz_size_bytes": shard_npz.stat().st_size,
        "uses_resource_adjusted_usd": True,
        "official_csv_loop_motion": True,
        "official_beyondmimic_vae_checkpoint": False,
        "paper_level_vae_closed_loop": False,
        "paper_level_guided_diffusion": False,
        "real_robot": False,
    }
    shard_metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    vec_env.close()
    print(f"BM_SENTINEL:vae_closed_loop:metrics_written:rank={rank}:{shard_metrics_path}", flush=True)
    os._exit(0)
except BaseException as exc:
    print(f"BM_SENTINEL:vae_closed_loop:exception:rank={rank}:{exc!r}", flush=True)
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
        record["policy"] = "terminated_wangjc_process_on_gpu_4_or_7_for_formal_eval"
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
        "scope": "GPU 4/7 guard for official-csv-loop VAE closed-loop rollout evaluation.",
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
    path = GPU_GUARD_DIR / f"{timestamp}_gpu47_wangjc_vae_closed_loop_rollout_guard.json"
    guard["path"] = str(path)
    path.write_text(json.dumps(guard, indent=2, sort_keys=True), encoding="utf-8")
    return guard


def select_checkpoint() -> Path:
    training_run = load_json(TRAINING_RUN_JSON)
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
    return ROOT / "res/runs/tracking_g1_official_csv_loop_ppo_training/resource_adjusted_ppo_20260618_224626_seed20260629/rank_0/model_299.pt"


def select_vae_checkpoint() -> Path:
    summary = load_json(VAE_TRAINING_JSON)
    return Path(summary.get("worker_summary", {}).get("outputs", {}).get("checkpoint", ""))


def base_env(run_dir: Path, checkpoint: Path, vae_checkpoint: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "4,7",
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
            "BM_MOTION_FILE": str(OFFICIAL_LOOP_MOTION_NPZ),
            "BM_CHECKPOINT": str(checkpoint),
            "BM_VAE_CHECKPOINT": str(vae_checkpoint),
            "BM_RUN_DIR": str(run_dir),
            "BM_NUM_ENVS_PER_RANK": str(NUM_ENVS_PER_RANK),
            "BM_ROLLOUT_STEPS": str(ROLLOUT_STEPS),
            "BM_SEED": str(SEED),
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
            "4,7",
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


def summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": float(sum(values) / len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def aggregate_shards(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    shard_metrics = []
    for path in sorted(run_dir.glob("rank_*/vae_closed_loop_rollout_metrics.json")):
        shard_metrics.append(load_json(path))
    aggregate = {
        "shard_count": len(shard_metrics),
        "total_env_steps": int(sum(item.get("total_env_steps", 0) for item in shard_metrics)),
        "total_num_envs": int(sum(item.get("num_envs", 0) for item in shard_metrics)),
        "rollout_steps": ROLLOUT_STEPS,
        "done_count_total": int(sum(item.get("done_count_total", 0) for item in shard_metrics)),
        "timeout_count_total": int(sum(item.get("timeout_count_total", 0) for item in shard_metrics)),
        "reward_mean_by_rank": [item.get("reward_mean", {}).get("mean") for item in shard_metrics],
        "teacher_vae_action_mse_by_rank": [
            item.get("teacher_vae_action_mse", {}).get("mean") for item in shard_metrics
        ],
        "teacher_vae_action_abs_error_by_rank": [
            item.get("teacher_vae_action_abs_error_mean", {}).get("mean") for item in shard_metrics
        ],
    }
    aggregate["reward_mean"] = summarize_values([v for v in aggregate["reward_mean_by_rank"] if v is not None])
    aggregate["teacher_vae_action_mse"] = summarize_values(
        [v for v in aggregate["teacher_vae_action_mse_by_rank"] if v is not None]
    )
    aggregate["teacher_vae_action_abs_error"] = summarize_values(
        [v for v in aggregate["teacher_vae_action_abs_error_by_rank"] if v is not None]
    )
    return shard_metrics, aggregate


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    worker_path = OUT / "tracking_g1_official_csv_loop_vae_closed_loop_rollout_worker.py"
    worker_path.write_text(textwrap.dedent(WORKER_CODE), encoding="utf-8")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RUN_ROOT / f"vae_closed_loop_rollout_{timestamp}_seed{SEED}"
    checkpoint = select_checkpoint()
    vae_checkpoint = select_vae_checkpoint()
    gpu_guard = write_gpu_guard(timestamp)
    gpu_snapshot = query_gpus()
    compute_processes = query_compute_processes()
    resource_ready = (
        len([row for row in gpu_snapshot if row.get("index") in CANDIDATE_GPUS]) == 2
        and all(
            row.get("memory_free_mb", 0) >= MIN_FREE_MB and row.get("utilization_gpu_percent", 100) <= MAX_BUSY_UTIL
            for row in gpu_snapshot
            if row.get("index") in CANDIDATE_GPUS
        )
        and not [
            proc for proc in compute_processes
            if WANGJC_PATH_MARKER not in f"{proc.get('process_name', '')} {proc.get('cmdline', '')}"
        ]
    )
    input_checks = {
        "tracking_python_exists": TRACKING_PY.is_file(),
        "teacher_training_json_exists": TRAINING_RUN_JSON.is_file(),
        "teacher_rollout_json_exists": TEACHER_ROLLOUT_JSON.is_file(),
        "vae_training_json_exists": VAE_TRAINING_JSON.is_file(),
        "checkpoint_exists": checkpoint.is_file(),
        "vae_checkpoint_exists": vae_checkpoint.is_file(),
        "enriched_usd_exists": ENRICHED_USD.is_file(),
        "motion_npz_exists": OFFICIAL_LOOP_MOTION_NPZ.is_file(),
        "selected_gpus_exactly_4_7": CANDIDATE_GPUS == [4, 7],
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
    log_path = LOG_DIR / "tracking_g1_official_csv_loop_vae_closed_loop_rollout_eval.log"
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
        env = base_env(run_dir, checkpoint, vae_checkpoint)
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, text=True, stdout=log_file, stderr=subprocess.STDOUT)
            returncode = proc.wait()
        monitor.terminate()
        try:
            monitor.wait(timeout=20)
        except subprocess.TimeoutExpired:
            monitor.kill()
            monitor.wait(timeout=20)
        shard_metrics, aggregate = aggregate_shards(run_dir)
        rollout_run.update(
            {
                "attempted_rollout": True,
                "returncode": returncode,
                "duration_seconds": round(time.time() - start, 3),
                "shard_count": len(shard_metrics),
                "shard_metrics": shard_metrics,
                "aggregate_metrics": aggregate,
                "gpu_metrics_summary": summarize_gpu_metrics(telemetry_path),
            }
        )
    else:
        rollout_run["reason_not_started"] = "Required inputs missing or GPU 4/7 resource preflight failed."
        shard_metrics, aggregate = [], {}

    success = (
        rollout_run.get("attempted_rollout")
        and rollout_run.get("returncode") == 0
        and aggregate.get("shard_count") == 2
        and aggregate.get("rollout_steps") == ROLLOUT_STEPS
    )
    if not success and log_path.is_file():
        failed_copy = FAILED_DIR / f"vae_closed_loop_rollout_{timestamp}.log"
        failed_copy.write_text(log_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        rollout_run["failed_log_copy"] = str(failed_copy)

    status = (
        "ok_official_csv_loop_vae_closed_loop_rollout_eval"
        if success
        else "failed_official_csv_loop_vae_closed_loop_rollout_eval"
        if rollout_run.get("attempted_rollout")
        else "ok_with_resource_unavailable_before_vae_closed_loop_rollout_eval"
    )
    gpu_summary = rollout_run.get("gpu_metrics_summary", {})
    peak_memory_by_gpu = {
        gpu: item.get("peak_memory_used_mb")
        for gpu, item in gpu_summary.get("per_gpu", {}).items()
    } if isinstance(gpu_summary, dict) else {}
    summary = {
        "status": status,
        "experiment_type": "tracking_official_csv_loop_vae_closed_loop_rollout_eval",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Full 299-step, two-GPU local VAE action-reconstruction closed-loop rollout. Each step uses the local PPO "
            "teacher action, encodes it with the local official-csv-loop action VAE, decodes from posterior mean, "
            "and steps IsaacLab with the decoded action."
        ),
        "config": {
            "candidate_physical_gpus": CANDIDATE_GPUS,
            "cuda_visible_devices": "4,7",
            "num_envs_per_rank": NUM_ENVS_PER_RANK,
            "world_size": 2,
            "rollout_steps": ROLLOUT_STEPS,
            "expected_total_env_steps": NUM_ENVS_PER_RANK * 2 * ROLLOUT_STEPS,
            "seed": SEED,
            "formal_gpu_experiment": True,
            "min_free_mb_required_per_gpu": MIN_FREE_MB,
            "max_busy_util_percent_for_start": MAX_BUSY_UTIL,
            "single_gpu_peak_memory_target_mb": 10_240,
        },
        "gpu_preflight": {"snapshot": gpu_snapshot, "compute_processes": compute_processes, "gpu_guard": gpu_guard},
        "inputs": {
            "training_run_json": str(TRAINING_RUN_JSON),
            "teacher_rollout_json": str(TEACHER_ROLLOUT_JSON),
            "vae_training_json": str(VAE_TRAINING_JSON),
            "checkpoint": str(checkpoint),
            "vae_checkpoint": str(vae_checkpoint),
            "motion_npz": str(OFFICIAL_LOOP_MOTION_NPZ),
            "enriched_usd": str(ENRICHED_USD),
        },
        "input_checks": input_checks,
        "run": rollout_run,
        "aggregate_metrics": aggregate,
        "outputs": {
            "json": str(OUT / "tracking_g1_official_csv_loop_vae_closed_loop_rollout_eval.json"),
            "worker_script": str(worker_path),
            "run_dir": str(run_dir),
            "log": str(log_path),
            "gpu_metrics_csv": str(telemetry_path),
        },
        "checks": {
            "rollout_success": bool(success),
            "uses_gpus_4_7": True,
            "two_shards_completed": aggregate.get("shard_count") == 2,
            "rollout_steps_299": aggregate.get("rollout_steps") == 299,
            "total_env_steps_full": aggregate.get("total_env_steps") == NUM_ENVS_PER_RANK * 2 * ROLLOUT_STEPS,
            "peak_memory_each_gpu_at_least_10gb": all(
                (value or 0) >= 10_240 for value in peak_memory_by_gpu.values()
            ) and len(peak_memory_by_gpu) == 2,
            "does_not_claim_official_beyondmimic_vae": True,
            "does_not_claim_autonomous_vae_policy": True,
            "does_not_claim_guided_diffusion": True,
            "does_not_claim_fig5_fig6": True,
            "does_not_claim_real_robot": True,
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "local_virtual_vae_action_reconstruction_closed_loop_eval" if success else "not_completed",
            "why_not_paper_level": (
                "This evaluates local VAE reconstruction of teacher actions in closed loop under the enriched-USD "
                "official-csv-loop path. It is not the unreleased official BeyondMimic VAE checkpoint, not an "
                "autonomous VAE rollout policy, not diffusion guidance, not TensorRT, and not real robot evidence."
            ),
        },
    }
    (OUT / "tracking_g1_official_csv_loop_vae_closed_loop_rollout_eval.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": status, "json": summary["outputs"]["json"]}, sort_keys=True))
    if status.startswith("failed_"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
