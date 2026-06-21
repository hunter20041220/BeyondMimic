#!/usr/bin/env python3
"""Trace wrist/ankle z-errors for the scaled PPO checkpoint under the official ee_body_pos gate."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_endpoint_z_error_trace"
REPORT_OUT = ROOT / "res/report_assets/official_importer_export_scaled_ppo_endpoint_z_error_trace"
LOG_DIR = ROOT / "logs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_endpoint_z_error_trace"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_endpoint_z_error_trace"
OFFICIAL_IMPORTER_USD = (
    ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda"
)
MOTION_NPZ = (
    ROOT / "res/tracking/official_csv_loop_full_bundle_motion_npz/official_csv_loop_full_public_motion_bundle.npz"
)
TRAINING_JSON = (
    ROOT
    / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_training_run/"
    "tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run.json"
)
SOURCE_AUDIT_JSON = (
    ROOT
    / "res/report_assets/official_importer_export_scaled_ppo_ee_body_pos_termination_source_audit/"
    "ee_body_pos_termination_source_audit.json"
)
ENV_PYTHON = ROOT / "envs/bm_tracking/bin/python"
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
TARGET_GPUS = [4, 7]
DEFAULT_NUM_ENVS = 2048
DEFAULT_EVAL_STEPS = 299
DEFAULT_SEED = 20260801
THRESHOLD_M = 0.25
TERMINATION_BODIES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    tmp.replace(path)


def query_gpu_snapshot() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    rows: list[dict[str, Any]] = []
    if result.returncode != 0:
        return rows
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mb": int(float(parts[2])),
                "memory_total_mb": int(float(parts[3])),
                "utilization_gpu_percent": int(float(parts[4])),
            }
        )
    return rows


def latest_checkpoint(training: dict[str, Any]) -> Path:
    direct = training.get("outputs", {}).get("latest_checkpoint")
    if direct and Path(direct).is_file():
        return Path(direct)
    run_dir = Path(training.get("outputs", {}).get("run_dir", ""))
    candidates = sorted(run_dir.glob("rank_0/model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not candidates:
        raise FileNotFoundError(f"No scaled PPO checkpoint found under {run_dir}")
    return candidates[-1]


def write_worker(path: Path) -> None:
    path.write_text(
        r'''
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

print(f"BM_SENTINEL:endpoint_trace:before_app:device={args.device}", flush=True)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("BM_SENTINEL:endpoint_trace:after_app", flush=True)

try:
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import torch
    import whole_body_tracking.tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

    usd = Path(os.environ["BM_OFFICIAL_IMPORTER_USD"])
    motion_file = Path(os.environ["BM_MOTION_FILE"])
    checkpoint = Path(os.environ["BM_CHECKPOINT"])
    run_dir = Path(os.environ["BM_RUN_DIR"])
    num_envs = int(os.environ["BM_NUM_ENVS"])
    eval_steps = int(os.environ["BM_EVAL_STEPS"])
    seed = int(os.environ["BM_EVAL_SEED"])
    threshold = float(os.environ["BM_EE_BODY_POS_THRESHOLD"])
    termination_bodies = os.environ["BM_TERMINATION_BODIES"].split(",")
    run_dir.mkdir(parents=True, exist_ok=True)
    step_csv = run_dir / "endpoint_z_error_timeseries.csv"
    body_csv = run_dir / "endpoint_z_error_by_body.csv"
    metrics_json = run_dir / "endpoint_z_error_metrics.json"

    env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=str(usd),
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
    env_cfg.seed = seed

    agent_cfg = G1FlatPPORunnerCfg()
    agent_cfg.device = args.device
    agent_cfg.seed = seed
    agent_cfg.empirical_normalization = True
    agent_cfg.logger = "tensorboard"

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    vec_env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(vec_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    loaded_infos = runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=vec_env.unwrapped.device)
    obs, extras = vec_env.get_observations()
    command = vec_env.unwrapped.command_manager.get_term("motion")

    body_indexes = [command.cfg.body_names.index(name) for name in termination_bodies]
    body_accum = {
        name: {
            "mean_abs_z": [],
            "p95_abs_z": [],
            "max_abs_z": [],
            "exceed_rate": [],
            "mean_signed_z": [],
        }
        for name in termination_bodies
    }
    aggregate_rows = []
    step_fields = [
        "step",
        "done_count",
        "timeout_count",
        "ee_body_pos_count",
        "aggregate_mean_abs_z",
        "aggregate_p95_abs_z",
        "aggregate_max_abs_z",
        "aggregate_exceed_rate",
    ]
    for name in termination_bodies:
        step_fields.extend(
            [
                f"{name}_mean_abs_z",
                f"{name}_p95_abs_z",
                f"{name}_max_abs_z",
                f"{name}_exceed_rate",
                f"{name}_mean_signed_z",
            ]
        )

    with step_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=step_fields, lineterminator="\n")
        writer.writeheader()
        with torch.inference_mode():
            for step in range(eval_steps):
                actions = policy(obs)
                obs, rew, dones, step_extras = vec_env.step(actions)
                target_z = command.body_pos_relative_w[:, body_indexes, 2]
                robot_z = command.robot_body_pos_w[:, body_indexes, 2]
                signed_z = robot_z - target_z
                abs_z = signed_z.abs()
                aggregate_abs = abs_z.reshape(-1)
                exceed_any = torch.any(abs_z > threshold, dim=1)
                log = step_extras.get("log", {})
                ee_count = log.get("Episode_Termination/ee_body_pos")
                if hasattr(ee_count, "detach"):
                    ee_value = int(float(ee_count.detach().mean().cpu()))
                else:
                    ee_value = int(float(ee_count)) if ee_count is not None else int(exceed_any.sum().detach().cpu())
                row = {
                    "step": step,
                    "done_count": int(dones.sum().detach().cpu()),
                    "timeout_count": int(step_extras.get("time_outs", torch.zeros_like(dones)).sum().detach().cpu()),
                    "ee_body_pos_count": ee_value,
                    "aggregate_mean_abs_z": float(aggregate_abs.mean().detach().cpu()),
                    "aggregate_p95_abs_z": float(torch.quantile(aggregate_abs, 0.95).detach().cpu()),
                    "aggregate_max_abs_z": float(aggregate_abs.max().detach().cpu()),
                    "aggregate_exceed_rate": float(exceed_any.float().mean().detach().cpu()),
                }
                for idx, name in enumerate(termination_bodies):
                    values = abs_z[:, idx]
                    signed_values = signed_z[:, idx]
                    body_exceed = (values > threshold).float()
                    stats = {
                        "mean_abs_z": float(values.mean().detach().cpu()),
                        "p95_abs_z": float(torch.quantile(values, 0.95).detach().cpu()),
                        "max_abs_z": float(values.max().detach().cpu()),
                        "exceed_rate": float(body_exceed.mean().detach().cpu()),
                        "mean_signed_z": float(signed_values.mean().detach().cpu()),
                    }
                    for key, value in stats.items():
                        body_accum[name][key].append(value)
                        row[f"{name}_{key}"] = value
                aggregate_rows.append(row)
                writer.writerow(row)

    body_rows = []
    for name, stats in body_accum.items():
        row = {"body_name": name, "threshold_m": threshold}
        for key, values in stats.items():
            row[f"{key}_mean_over_steps"] = sum(values) / len(values)
            row[f"{key}_max_over_steps"] = max(values)
            row[f"{key}_first"] = values[0]
            row[f"{key}_last"] = values[-1]
        body_rows.append(row)
    with body_csv.open("w", encoding="utf-8", newline="") as f:
        fields = list(body_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(body_rows)

    def series_summary(key):
        values = [float(row[key]) for row in aggregate_rows]
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
        "task": "Tracking-Flat-G1-v0",
        "checkpoint": str(checkpoint),
        "loaded_iteration": int(runner.current_learning_iteration),
        "loaded_infos_type": type(loaded_infos).__name__,
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "num_envs": int(vec_env.num_envs),
        "eval_steps": eval_steps,
        "total_env_steps": int(vec_env.num_envs) * eval_steps,
        "num_actions": int(vec_env.num_actions),
        "num_obs": int(vec_env.num_obs),
        "num_privileged_obs": int(vec_env.num_privileged_obs),
        "robot_num_joints": int(vec_env.unwrapped.scene["robot"].num_joints),
        "robot_num_bodies": int(vec_env.unwrapped.scene["robot"].num_bodies),
        "threshold_m": threshold,
        "termination_bodies": termination_bodies,
        "body_indexes": body_indexes,
        "aggregate": {
            "mean_abs_z": series_summary("aggregate_mean_abs_z"),
            "p95_abs_z": series_summary("aggregate_p95_abs_z"),
            "max_abs_z": series_summary("aggregate_max_abs_z"),
            "exceed_rate": series_summary("aggregate_exceed_rate"),
            "done_count": series_summary("done_count"),
            "ee_body_pos_count": series_summary("ee_body_pos_count"),
        },
        "body_rows": body_rows,
        "outputs": {
            "step_csv": str(step_csv),
            "body_csv": str(body_csv),
            "metrics_json": str(metrics_json),
        },
        "paper_level_tracking_eval": False,
        "uses_official_importer_export_usd": True,
        "official_csv_loop_full_public_bundle": True,
    }
    metrics_json.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    vec_env.close()
    print(f"BM_SENTINEL:endpoint_trace:metrics_written={metrics_json}", flush=True)
    os._exit(0)
except BaseException as exc:
    print(f"BM_SENTINEL:endpoint_trace:exception={exc!r}", flush=True)
    raise
finally:
    try:
        simulation_app.close(wait_for_replicator=False)
    except BaseException:
        pass
''',
        encoding="utf-8",
    )


def start_gpu_monitor(path: Path) -> subprocess.Popen[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv",
            "-i",
            ",".join(str(gpu) for gpu in TARGET_GPUS),
            "-l",
            "2",
            "-f",
            str(path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def run_worker(worker: Path, run_dir: Path, checkpoint: Path, num_envs: int, eval_steps: int, seed: int) -> tuple[int, float]:
    log_path = LOG_DIR / "tracking_g1_official_importer_export_scaled_ppo_endpoint_z_error_trace.log"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": "4,7",
            "VK_ICD_FILENAMES": str(PROJECT_EGL_ICD),
            "LD_LIBRARY_PATH": f"{GPU_FOUNDATION_DEPS}:{env.get('LD_LIBRARY_PATH', '')}",
            "ISAAC_PATH": str(ROOT / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim"),
            "OMNI_USER_DIR": str(ROOT / "cache/omni/user"),
            "OMNI_CACHE_DIR": str(ROOT / "cache/omni/cache"),
            "OMNI_DATA_DIR": str(ROOT / "cache/omni/data"),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "WANDB_MODE": "offline",
            "BM_OFFICIAL_IMPORTER_USD": str(OFFICIAL_IMPORTER_USD),
            "BM_MOTION_FILE": str(MOTION_NPZ),
            "BM_CHECKPOINT": str(checkpoint),
            "BM_RUN_DIR": str(run_dir),
            "BM_NUM_ENVS": str(num_envs),
            "BM_EVAL_STEPS": str(eval_steps),
            "BM_EVAL_SEED": str(seed),
            "BM_EE_BODY_POS_THRESHOLD": str(THRESHOLD_M),
            "BM_TERMINATION_BODIES": ",".join(TERMINATION_BODIES),
        }
    )
    cmd = [str(ENV_PYTHON), str(worker)]
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    return proc.returncode, time.time() - start


def make_report_assets(summary: dict[str, Any]) -> dict[str, str]:
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    import pandas as pd

    step_csv = Path(summary["outputs"]["step_csv"])
    body_csv = Path(summary["outputs"]["body_csv"])
    step_df = pd.read_csv(step_csv)
    body_df = pd.read_csv(body_csv)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(step_df["step"], step_df["aggregate_mean_abs_z"], label="mean abs z")
    ax.plot(step_df["step"], step_df["aggregate_p95_abs_z"], label="p95 abs z")
    ax.axhline(THRESHOLD_M, color="#dc2626", linestyle="--", label="0.25 m threshold")
    ax.set_xlabel("Step")
    ax.set_ylabel("Endpoint z-error (m)")
    ax.set_title("Scaled PPO wrist/ankle z-error against official ee_body_pos threshold")
    ax.legend()
    fig.tight_layout()
    timeseries_png = REPORT_OUT / "endpoint_z_error_timeseries.png"
    fig.savefig(timeseries_png, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    body_df.plot(
        x="body_name",
        y="exceed_rate_mean_over_steps",
        kind="bar",
        ax=ax,
        color="#ea580c",
        legend=False,
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Mean exceed rate over steps")
    ax.set_title("Endpoint body contribution to z-threshold violations")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    body_png = REPORT_OUT / "endpoint_z_error_by_body.png"
    fig.savefig(body_png, dpi=180)
    plt.close(fig)

    report_csv = REPORT_OUT / "endpoint_z_error_by_body.csv"
    body_df.to_csv(report_csv, index=False)
    report_md = REPORT_OUT / "endpoint_z_error_trace.md"
    report_md.write_text(
        "\n".join(
            [
                "# Scaled PPO Endpoint Z-Error Trace",
                "",
                f"Checkpoint: `{summary['inputs']['checkpoint']}`.",
                f"Eval size: `{summary['config']['num_envs']}` envs x `{summary['config']['eval_steps']}` steps.",
                f"Threshold: `{THRESHOLD_M}` m on `{TERMINATION_BODIES}`.",
                f"Aggregate exceed-rate mean: `{summary['metrics']['aggregate']['exceed_rate']['mean']}`.",
                f"Aggregate p95 z-error mean: `{summary['metrics']['aggregate']['p95_abs_z']['mean']}`.",
                "",
                "Claim level: local full-size checkpoint diagnostic. This is not a paper-level BeyondMimic tracking",
                "metric, not an official checkpoint, and not a real-robot result.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    readme = REPORT_OUT / "README.md"
    readme.write_text(report_md.read_text(encoding="utf-8"), encoding="utf-8")
    return {
        "timeseries_png": str(timeseries_png),
        "body_png": str(body_png),
        "body_csv": str(report_csv),
        "markdown": str(report_md),
        "readme": str(readme),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.mkdir(parents=True, exist_ok=True)

    training = load_json(TRAINING_JSON)
    source_audit = load_json(SOURCE_AUDIT_JSON)
    checkpoint = latest_checkpoint(training)
    num_envs = int(os.environ.get("BM_ENDPOINT_TRACE_NUM_ENVS", str(DEFAULT_NUM_ENVS)))
    eval_steps = int(os.environ.get("BM_ENDPOINT_TRACE_EVAL_STEPS", str(DEFAULT_EVAL_STEPS)))
    seed = int(os.environ.get("BM_ENDPOINT_TRACE_SEED", str(DEFAULT_SEED)))
    run_id = f"endpoint_z_error_trace_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_seed{seed}"
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    worker = OUT / "tracking_g1_official_importer_export_scaled_ppo_endpoint_z_error_trace_worker.py"
    write_worker(worker)
    shutil.copyfile(worker, run_dir / worker.name)

    gpu_before = query_gpu_snapshot()
    target_gpu_rows = [row for row in gpu_before if row["index"] in TARGET_GPUS]
    input_checks = {
        "env_python_exists": ENV_PYTHON.is_file(),
        "official_importer_usd_exists": OFFICIAL_IMPORTER_USD.is_file(),
        "motion_npz_exists": MOTION_NPZ.is_file(),
        "training_json_exists": TRAINING_JSON.is_file(),
        "source_audit_exists": SOURCE_AUDIT_JSON.is_file(),
        "source_audit_threshold_matches": source_audit.get("source_config", {}).get("threshold_m") == THRESHOLD_M,
        "checkpoint_exists": checkpoint.is_file(),
        "target_gpus_present": len(target_gpu_rows) == len(TARGET_GPUS),
        "target_gpus_have_memory": all(
            (row["memory_total_mb"] - row["memory_used_mb"]) >= 20000 for row in target_gpu_rows
        ),
        "target_gpus_not_busy": all(row["utilization_gpu_percent"] <= 50 for row in target_gpu_rows),
    }

    gpu_metrics_csv = run_dir / "gpu_metrics.csv"
    monitor: subprocess.Popen[str] | None = None
    returncode = -1
    duration = 0.0
    if all(input_checks.values()):
        monitor = start_gpu_monitor(gpu_metrics_csv)
        try:
            returncode, duration = run_worker(worker, run_dir, checkpoint, num_envs, eval_steps, seed)
        finally:
            if monitor is not None:
                monitor.terminate()
                try:
                    monitor.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    monitor.kill()
    metrics_path = run_dir / "endpoint_z_error_metrics.json"
    metrics = load_json(metrics_path)
    metrics_ok = returncode == 0 and metrics.get("status") == "ok"
    if metrics_ok:
        report_assets = make_report_assets(
            {
                "inputs": {"checkpoint": str(checkpoint)},
                "config": {"num_envs": num_envs, "eval_steps": eval_steps},
                "metrics": metrics,
                "outputs": metrics["outputs"],
            }
        )
    else:
        report_assets = {}

    summary = {
        "status": (
            "ok_official_importer_export_scaled_ppo_endpoint_z_error_trace"
            if metrics_ok
            else "failed_official_importer_export_scaled_ppo_endpoint_z_error_trace"
        ),
        "experiment_type": "tracking_official_importer_export_scaled_ppo_endpoint_z_error_trace",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Full-size local checkpoint diagnostic for the official ee_body_pos z-only wrist/ankle termination gate. "
            "This reuses the official-importer-export G1 USDA, full public motion bundle, and iteration-999 scaled "
            "PPO checkpoint, but does not claim paper-level tracking performance."
        ),
        "inputs": {
            "training_json": str(TRAINING_JSON),
            "source_audit_json": str(SOURCE_AUDIT_JSON),
            "checkpoint": str(checkpoint),
            "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
            "motion_npz": str(MOTION_NPZ),
        },
        "input_checks": input_checks,
        "config": {
            "candidate_physical_gpus": TARGET_GPUS,
            "selected_physical_gpus": TARGET_GPUS,
            "cuda_visible_devices": "4,7",
            "num_envs": num_envs,
            "eval_steps": eval_steps,
            "total_env_steps": num_envs * eval_steps,
            "seed": seed,
            "termination_bodies": TERMINATION_BODIES,
            "threshold_m": THRESHOLD_M,
            "formal_gpu_experiment": False,
            "why_not_formal_gpu_experiment": (
                "This is a checkpoint diagnostic, not a formal training experiment; GPU memory is recorded but "
                "the run is not required to consume >=10GB/card."
            ),
        },
        "gpu_preflight": {
            "before": gpu_before,
            "target_gpu_rows": target_gpu_rows,
        },
        "run": {
            "attempted": all(input_checks.values()),
            "returncode": returncode,
            "duration_seconds": duration,
            "metrics_exists": metrics_path.is_file(),
            "metrics": metrics,
        },
        "outputs": {
            "json": str(OUT / "tracking_g1_official_importer_export_scaled_ppo_endpoint_z_error_trace.json"),
            "run_dir": str(run_dir),
            "worker_script": str(worker),
            "worker_copy": str(run_dir / worker.name),
            "log": str(LOG_DIR / "tracking_g1_official_importer_export_scaled_ppo_endpoint_z_error_trace.log"),
            "gpu_metrics_csv": str(gpu_metrics_csv),
            "metrics_json": str(metrics_path),
            "step_csv": metrics.get("outputs", {}).get("step_csv", ""),
            "body_csv": metrics.get("outputs", {}).get("body_csv", ""),
            "report_assets": report_assets,
        },
        "checks": {
            "process_returned_zero": returncode == 0,
            "metrics_status_ok": metrics.get("status") == "ok",
            "eval_shape_full_size": metrics.get("num_envs") == num_envs and metrics.get("eval_steps") == eval_steps,
            "threshold_matches_source": source_audit.get("source_config", {}).get("threshold_m") == THRESHOLD_M,
            "termination_bodies_match_source": source_audit.get("source_config", {}).get("termination_body_names")
            == TERMINATION_BODIES,
            "records_four_endpoint_bodies": len(metrics.get("body_rows", [])) == 4,
            "aggregate_exceed_rate_recorded": "exceed_rate" in metrics.get("aggregate", {}),
            "report_assets_exist": bool(report_assets)
            and all(Path(path).is_file() for path in report_assets.values()),
            "does_not_claim_paper_level_eval": True,
            "does_not_claim_official_beyondmimic_checkpoint": True,
            "does_not_claim_real_robot": True,
            "does_not_claim_goal_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "claim_level": "local_full_size_checkpoint_endpoint_z_error_diagnostic_not_paper_level",
            "main_finding": (
                "This diagnostic measures which of the official ee_body_pos wrist/ankle z-threshold bodies drive "
                "termination in the current local scaled PPO teacher. It is meant to guide the next PPO/retargeting "
                "fix before downstream rollout data is trusted."
            ),
            "why_not_paper_level": (
                "The checkpoint is locally trained on public data and the diagnostic measures a local termination "
                "mechanism. It is not an official BeyondMimic metric, not official DAgger data, not Fig. 5/Fig. 6 "
                "closed-loop guidance, not TensorRT deployment, and not real robot validation."
            ),
        },
    }
    write_json(Path(summary["outputs"]["json"]), summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": summary["outputs"]["json"],
                "attempted": summary["run"]["attempted"],
                "returncode": returncode,
                "duration_seconds": duration,
                "aggregate_exceed_rate_mean": metrics.get("aggregate", {}).get("exceed_rate", {}).get("mean"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
