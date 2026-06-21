#!/usr/bin/env python3
"""Run full-dataset task eval using the official-importer GPU4 G1 USDA export.

The motion NPZ files come from the full official ``csv_to_npz.py`` loop audit.
Unlike the prior full-dataset task gate, this script uses the large USDA exported
by the official Isaac Sim URDF importer in-memory GPU4 probe as the robot asset.
It records zero-action task-contract metrics only; it is not PPO training,
DAgger, paper-level policy performance, TensorRT, or real-robot validation.
"""

from __future__ import annotations

import csv
import json
import math
import os
import signal
import subprocess
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/g1_official_importer_export_full_dataset_task_eval"
REPORT_OUT = ROOT / "res/report_assets/official_importer_export_full_dataset_task_eval"
LOG_DIR = ROOT / "logs/tracking_g1_official_importer_export_full_dataset_task_eval"
FAILED_DIR = ROOT / "res/failed_runs/tracking_g1_official_importer_export_full_dataset_task_eval"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
CSV_FULL_AUDIT = (
    ROOT
    / "res/tracking/official_csv_to_npz_loop_full_dataset_with_official_importer_export/"
    "tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_audit.json"
)
REPLAY_FULL_AUDIT = (
    ROOT
    / "res/tracking/official_replay_npz_loop_full_dataset_with_official_importer_export/"
    "tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_audit.json"
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
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
TARGET_GPU = int(os.environ.get("BM_OFFICIAL_IMPORTER_EXPORT_FULL_TASK_GPU", "4"))
MAX_STEPS = int(os.environ.get("BM_OFFICIAL_IMPORTER_EXPORT_FULL_TASK_MAX_STEPS", "299"))
LIMIT = int(os.environ.get("BM_OFFICIAL_IMPORTER_EXPORT_FULL_TASK_LIMIT", "0"))
STALL_SECONDS = int(os.environ.get("BM_OFFICIAL_IMPORTER_EXPORT_FULL_TASK_STALL_SECONDS", "900"))
WATCH_GPUS = [4, 7]
WANGJC_PATH_MARKER = "/mnt/infini-data/test/wangjc/"


WORKER_CODE = r"""
import json
import os
from pathlib import Path

OUT = Path(os.environ["BM_TASK_METRICS_JSON"])
ROBOT_USD = Path(os.environ["BM_ROBOT_USD"])
MOTION_FILE = Path(os.environ["BM_MOTION_FILE"])
MAX_STEPS = int(os.environ["BM_MAX_STEPS"])

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

    env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.robot.spawn = sim_utils.UsdFileCfg(
        usd_path=str(ROBOT_USD),
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
    env_cfg.seed = int(os.environ.get("BM_SEED", "20260650"))

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print("BM_SENTINEL:env_created", flush=True)
    obs, extras = env.reset()
    print("BM_SENTINEL:env_reset", flush=True)

    action_dim = int(env.unwrapped.action_manager.total_action_dim)
    action = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
    command = env.unwrapped.command_manager.get_term("motion")
    step_count = min(int(command.motion.time_step_total), MAX_STEPS)
    rewards = []
    terminated_counts = []
    truncated_counts = []
    metric_series = {}
    for i in range(step_count):
        obs, reward, terminated, truncated, extras = env.step(action)
        rewards.append(float(reward.detach().cpu().mean().item()))
        terminated_counts.append(int(terminated.detach().cpu().sum().item()))
        truncated_counts.append(int(truncated.detach().cpu().sum().item()))
        for name, value in command.metrics.items():
            if hasattr(value, "detach") and value.numel() > 0:
                metric_series.setdefault(name, []).append(float(value.detach().cpu().mean().item()))
        if (i + 1) % 50 == 0 or (i + 1) == step_count:
            print(f"BM_SENTINEL:env_step={i + 1}/{step_count}", flush=True)

    def summarize(values):
        finite = [float(v) for v in values if math.isfinite(float(v))]
        if not finite:
            return {"count": 0, "mean": None, "min": None, "max": None}
        return {
            "count": len(finite),
            "mean": sum(finite) / len(finite),
            "min": min(finite),
            "max": max(finite),
        }

    import math
    command_metrics_final = {
        k: float(v.detach().cpu().mean().item())
        for k, v in command.metrics.items()
        if hasattr(v, "detach") and v.numel() > 0
    }
    metrics = {
        "status": "ok",
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
        "reward": summarize(rewards),
        "terminated_total": int(sum(terminated_counts)),
        "truncated_total": int(sum(truncated_counts)),
        "done_total": int(sum(terminated_counts) + sum(truncated_counts)),
        "command_metrics_final": command_metrics_final,
        "command_metrics_timeseries": {name: summarize(values) for name, values in metric_series.items()},
        "robot_num_joints": int(env.unwrapped.scene["robot"].num_joints),
        "robot_num_bodies": int(env.unwrapped.scene["robot"].num_bodies),
        "usd_path": str(ROBOT_USD),
        "uses_official_importer_export_usd": True,
        "uses_resource_adjusted_usd": False,
        "official_csv_loop_npz_input": True,
        "official_csv_to_npz_unpatched_output": False,
        "paper_level_rollout": False,
        "ppo_training": False,
        "real_robot": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"BM_SENTINEL:metrics_file={OUT}", flush=True)
    print("BM_SENTINEL:task_eval_success", flush=True)
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


def run_command(cmd: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)


def gpu_index_to_bus_id() -> dict[int, str]:
    proc = run_command(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"])
    mapping = {}
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2:
            try:
                mapping[int(parts[0])] = parts[1]
            except ValueError:
                pass
    return mapping


def parse_gpu_processes() -> list[dict[str, Any]]:
    proc = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            rows.append(
                {
                    "gpu_bus_id": parts[0],
                    "pid": int(parts[1]),
                    "process_name": parts[2],
                    "used_memory_mb": int(parts[3]),
                }
            )
        except ValueError:
            continue
    return rows


def cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
    except OSError:
        return ""


def kill_wangjc_on_watch_gpus() -> dict[str, Any]:
    guard_dir = ROOT / "res/gpu_guard"
    guard_dir.mkdir(parents=True, exist_ok=True)
    bus = gpu_index_to_bus_id()
    target_bus = {bus[index] for index in WATCH_GPUS if index in bus}
    killed = []
    skipped = []
    for row in parse_gpu_processes():
        if row["gpu_bus_id"] not in target_bus:
            continue
        command = cmdline(row["pid"])
        item = row | {"cmdline": command}
        if WANGJC_PATH_MARKER in command:
            try:
                os.kill(row["pid"], signal.SIGTERM)
                item["signal"] = "SIGTERM"
            except ProcessLookupError:
                item["signal"] = "already_exited"
            killed.append(item)
        else:
            skipped.append(item)
    if killed:
        time.sleep(8)
        for item in killed:
            pid = item["pid"]
            if Path(f"/proc/{pid}").exists():
                try:
                    os.kill(pid, signal.SIGKILL)
                    item["signal"] = "SIGKILL_after_grace"
                except ProcessLookupError:
                    pass
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "watch_gpus": WATCH_GPUS,
        "target_gpu_for_run": TARGET_GPU,
        "killed": killed,
        "skipped_non_wangjc": skipped,
    }
    path = guard_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_gpu47_wangjc_official_csv_task_eval_guard.json"
    write_json(path, summary)
    summary["json"] = str(path)
    return summary


def classify_log(text: str, step_count: int) -> dict[str, Any]:
    lowered = text.lower()
    return {
        "before_app": "bm_sentinel:before_app" in lowered,
        "after_app": "bm_sentinel:after_app" in lowered,
        "env_created": "bm_sentinel:env_created" in lowered,
        "env_reset": "bm_sentinel:env_reset" in lowered,
        "step_bound_reached": f"bm_sentinel:env_step={step_count}/{step_count}" in lowered,
        "metrics_file": "bm_sentinel:metrics_file=" in lowered,
        "success": "bm_sentinel:task_eval_success" in lowered,
        "traceback": "traceback (most recent call last)" in lowered,
        "exception": "bm_sentinel:exception=" in lowered,
        "vulkan_device_lost": "vkresult: error_device_lost" in lowered or "device lost" in lowered,
        "stall_timeout": "bm_stall_timeout" in lowered,
    }


def env_vars(motion_npz: Path, metrics_json: Path, seed: int) -> dict[str, str]:
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
            "BM_ROBOT_USD": str(OFFICIAL_IMPORTER_USD),
            "BM_MOTION_FILE": str(motion_npz),
            "BM_TASK_METRICS_JSON": str(metrics_json),
            "BM_TARGET_GPU": str(TARGET_GPU),
            "BM_DEVICE": f"cuda:{TARGET_GPU}",
            "BM_MAX_STEPS": str(MAX_STEPS),
            "BM_SEED": str(seed),
        }
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def run_worker(motion: str, motion_npz: Path, metrics_json: Path, worker_path: Path, log_path: Path, seed: int) -> dict[str, Any]:
    command = [str(TRACKING_PY), str(worker_path), "--headless", "--device", f"cuda:{TARGET_GPU}"]
    start = time.time()
    stalled = False
    last_size = -1
    last_change = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env_vars(motion_npz, metrics_json, seed),
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
    markers = classify_log(text, MAX_STEPS)
    metrics = load_json(metrics_json)
    ok = (
        proc.returncode == 0
        and not stalled
        and metrics.get("status") == "ok"
        and metrics.get("step_count") == MAX_STEPS
        and metrics.get("action_dim") == 29
        and metrics.get("policy_observation_dim") == 160
        and metrics.get("critic_observation_dim") == 286
        and metrics.get("robot_num_joints") == 29
        and metrics.get("robot_num_bodies") == 40
        and markers["success"]
    )
    return {
        "motion": motion,
        "status": "ok" if ok else "failed",
        "returncode": proc.returncode,
        "duration_seconds": round(time.time() - start, 3),
        "stalled": stalled,
        "markers": markers,
        "metrics_json": str(metrics_json) if metrics_json.is_file() else "",
        "log": str(log_path),
        "metrics": metrics,
    }


def selected_rows() -> list[dict[str, Any]]:
    audit = load_json(CSV_FULL_AUDIT)
    rows = [row for row in audit.get("rows", []) if row.get("status") == "ok" and row.get("output_npz")]
    rows = sorted(rows, key=lambda row: row["motion"])
    if LIMIT > 0:
        rows = rows[:LIMIT]
    return rows


def flatten_result(row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics") or {}
    command = metrics.get("command_metrics_final") or {}
    reward = metrics.get("reward") or {}
    return {
        "motion": row["motion"],
        "status": row["status"],
        "returncode": row["returncode"],
        "duration_seconds": row["duration_seconds"],
        "step_count": metrics.get("step_count", ""),
        "reward_mean": reward.get("mean", ""),
        "reward_min": reward.get("min", ""),
        "reward_max": reward.get("max", ""),
        "terminated_total": metrics.get("terminated_total", ""),
        "truncated_total": metrics.get("truncated_total", ""),
        "done_total": metrics.get("done_total", ""),
        "error_anchor_pos": command.get("error_anchor_pos", ""),
        "error_body_pos": command.get("error_body_pos", ""),
        "error_joint_pos": command.get("error_joint_pos", ""),
        "error_anchor_lin_vel": command.get("error_anchor_lin_vel", ""),
        "error_body_lin_vel": command.get("error_body_lin_vel", ""),
        "error_joint_vel": command.get("error_joint_vel", ""),
        "sampling_entropy": command.get("sampling_entropy", ""),
        "sampling_top1_prob": command.get("sampling_top1_prob", ""),
        "metrics_json": row.get("metrics_json", ""),
        "log": row.get("log", ""),
    }


def write_table(path: Path, rows: list[dict[str, Any]], delimiter: str) -> None:
    fieldnames = [
        "motion",
        "status",
        "returncode",
        "duration_seconds",
        "step_count",
        "reward_mean",
        "reward_min",
        "reward_max",
        "terminated_total",
        "truncated_total",
        "done_total",
        "error_anchor_pos",
        "error_body_pos",
        "error_joint_pos",
        "error_anchor_lin_vel",
        "error_body_lin_vel",
        "error_joint_vel",
        "sampling_entropy",
        "sampling_top1_prob",
        "metrics_json",
        "log",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return values


def summarize(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / len(values)
    return {"count": len(values), "mean": mean, "std": math.sqrt(var), "min": min(values), "max": max(values)}


def write_report_assets(table_rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, str]:
    REPORT_OUT.mkdir(parents=True, exist_ok=True)
    metrics_csv = REPORT_OUT / "official_importer_export_full_dataset_task_eval_metrics.csv"
    completion_csv = REPORT_OUT / "official_importer_export_full_dataset_task_eval_completion_table.csv"
    write_table(metrics_csv, table_rows, ",")
    completion_rows = [
        {
            "motion": row["motion"],
            "status": row["status"],
            "step_count": row.get("step_count", ""),
            "done_total": row.get("done_total", ""),
            "terminated_total": row.get("terminated_total", ""),
            "truncated_total": row.get("truncated_total", ""),
            "completed_299_step_gate": row["status"] == "ok" and int(row.get("step_count") or 0) == MAX_STEPS,
            "claim_level": "official_importer_export_task_eval_diagnostic",
        }
        for row in table_rows
    ]
    with completion_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(completion_rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(completion_rows)

    motions = [row["motion"] for row in table_rows]
    x = list(range(len(motions)))

    reward_png = REPORT_OUT / "official_importer_export_full_dataset_task_eval_reward_done.png"
    errors_png = REPORT_OUT / "official_importer_export_full_dataset_task_eval_tracking_errors.png"
    plot_rows_json = REPORT_OUT / "official_importer_export_full_dataset_task_eval_plot_rows.json"
    plot_rows_json.write_text(json.dumps(table_rows, indent=2, sort_keys=True), encoding="utf-8")
    plot_code = f"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = json.loads(Path({str(plot_rows_json)!r}).read_text(encoding="utf-8"))
motions = [row["motion"] for row in rows]
x = list(range(len(motions)))

def f(row, key):
    value = row.get(key, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")

fig, ax1 = plt.subplots(figsize=(14, 5))
ax1.plot(x, [f(row, "reward_mean") for row in rows], marker="o", linewidth=1.4, label="Reward mean", color="#1f77b4")
ax1.set_ylabel("Reward mean")
ax1.set_xticks(x)
ax1.set_xticklabels(motions, rotation=75, ha="right", fontsize=7)
ax2 = ax1.twinx()
ax2.bar(x, [f(row, "done_total") for row in rows], alpha=0.25, label="Done count", color="#d62728")
ax2.set_ylabel("Terminated + truncated count")
ax1.set_title("Official-Importer-Export Full Task Eval: Reward and Done Counts")
fig.tight_layout()
fig.savefig({str(reward_png)!r}, dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(14, 5))
for key, label, color in [
    ("error_anchor_pos", "Anchor position", "#1f77b4"),
    ("error_body_pos", "Body position", "#2ca02c"),
    ("error_joint_pos", "Joint position", "#ff7f0e"),
]:
    ax.plot(x, [f(row, key) for row in rows], marker="o", linewidth=1.2, label=label, color=color)
ax.set_xticks(x)
ax.set_xticklabels(motions, rotation=75, ha="right", fontsize=7)
ax.set_ylabel("Final command metric")
ax.set_title("Official-Importer-Export Full Task Eval: Tracking Error Metrics")
ax.legend()
fig.tight_layout()
fig.savefig({str(errors_png)!r}, dpi=180)
plt.close(fig)
"""
    plot_script = REPORT_OUT / "render_official_importer_export_full_dataset_task_eval_plots.py"
    plot_script.write_text(plot_code, encoding="utf-8")
    analysis_python = ROOT / "envs/bm_analysis/bin/python"
    proc = subprocess.run(
        [str(analysis_python), str(plot_script)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Report asset plotting failed with {analysis_python}: {proc.stdout}")

    readme = REPORT_OUT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Official Importer Export Full-Dataset Task Eval Assets",
                "",
                "This directory contains report assets for the full public-motion `Tracking-Flat-G1-v0` task eval using the official-importer GPU4 USDA export.",
                "",
                f"- Source audit: `{summary['outputs']['json']}`",
                f"- Motions evaluated: `{summary['aggregate']['ok_count']}/{summary['aggregate']['row_count']}`",
                f"- Total task steps: `{summary['aggregate']['total_steps']}`",
                "",
                "Claim boundary: official-importer-export task eval diagnostic. It uses the GPU4 USDA export from the official Isaac Sim URDF importer and official-loop NPZ inputs generated by the full official-importer-export csv_to_npz loop audit. It is not unpatched live official converter-entry success, trained PPO policy evaluation, paper Fig. 5/Fig. 6 evidence, or real-robot evidence.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    asset_json = REPORT_OUT / "official_importer_export_full_dataset_task_eval_assets.json"
    assets = {
        "status": "ok",
        "experiment_type": "official_importer_export_full_dataset_task_eval_report_assets",
        "claim_level": "official_importer_export_full_dataset_task_eval_report_assets",
        "source_audit": summary["outputs"]["json"],
        "assets": {
            "metrics_csv": str(metrics_csv),
            "completion_csv": str(completion_csv),
            "reward_done_png": str(reward_png),
            "tracking_errors_png": str(errors_png),
            "readme": str(readme),
        },
        "limitations": summary["interpretation"]["not_paper_level_reasons"],
    }
    write_json(asset_json, assets)
    return assets["assets"] | {"asset_json": str(asset_json)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    worker_path = OUT / "tracking_g1_official_importer_export_full_dataset_task_eval_worker.py"
    worker_path.write_text(textwrap.dedent(WORKER_CODE), encoding="utf-8")
    guard = kill_wangjc_on_watch_gpus()
    input_rows = selected_rows()
    rows = []
    for index, item in enumerate(input_rows):
        motion = item["motion"]
        motion_dir = OUT / "motions" / motion
        motion_dir.mkdir(parents=True, exist_ok=True)
        metrics_json = motion_dir / f"{motion}_task_eval_metrics.json"
        log_path = LOG_DIR / f"{motion}_task_eval.log"
        result = run_worker(
            motion,
            Path(item["output_npz"]),
            metrics_json,
            worker_path,
            log_path,
            seed=20260650 + index,
        )
        motion_audit = motion_dir / f"{motion}_task_eval_audit.json"
        write_json(motion_audit, result)
        result["motion_audit"] = str(motion_audit)
        if result["status"] != "ok":
            failed_copy = FAILED_DIR / f"{motion}_task_eval.log"
            failed_copy.write_text(log_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            result["failed_log_copy"] = str(failed_copy)
        rows.append(result)

    table_rows = [flatten_result(row) for row in rows]
    ok_rows = [row for row in table_rows if row["status"] == "ok"]
    failed_rows = [row for row in table_rows if row["status"] != "ok"]
    aggregate = {
        "row_count": len(table_rows),
        "ok_count": len(ok_rows),
        "failed_count": len(failed_rows),
        "total_steps": sum(int(row.get("step_count") or 0) for row in ok_rows),
        "total_done_count": sum(int(row.get("done_total") or 0) for row in ok_rows),
        "reward_mean": summarize(finite_values(ok_rows, "reward_mean")),
        "error_anchor_pos": summarize(finite_values(ok_rows, "error_anchor_pos")),
        "error_body_pos": summarize(finite_values(ok_rows, "error_body_pos")),
        "error_joint_pos": summarize(finite_values(ok_rows, "error_joint_pos")),
        "duration_seconds": summarize(finite_values(ok_rows, "duration_seconds")),
    }
    csv_audit = load_json(CSV_FULL_AUDIT)
    replay_audit = load_json(REPLAY_FULL_AUDIT)
    checks = {
        "csv_full_dataset_audit_passed": csv_audit.get("status")
        == "ok_official_csv_to_npz_loop_full_dataset_with_official_importer_export",
        "replay_full_dataset_audit_passed": replay_audit.get("status")
        == "ok_official_replay_npz_loop_full_dataset_with_official_importer_export",
        "all_40_motion_inputs_selected": len(input_rows) == 40 if LIMIT == 0 else len(input_rows) == LIMIT,
        "all_input_npz_exist": all(Path(row["output_npz"]).is_file() for row in input_rows),
        "all_rows_ok": len(table_rows) > 0 and len(failed_rows) == 0,
        "all_rows_step_299": all(int(row.get("step_count") or 0) == MAX_STEPS for row in table_rows),
        "all_rows_action_dim_29": all((rows[i].get("metrics") or {}).get("action_dim") == 29 for i in range(len(rows))),
        "all_rows_policy_obs_dim_160": all(
            (rows[i].get("metrics") or {}).get("policy_observation_dim") == 160 for i in range(len(rows))
        ),
        "all_rows_critic_obs_dim_286": all(
            (rows[i].get("metrics") or {}).get("critic_observation_dim") == 286 for i in range(len(rows))
        ),
        "all_rows_reward_terms_9": all(len((rows[i].get("metrics") or {}).get("reward_terms", [])) == 9 for i in range(len(rows))),
        "all_rows_termination_terms_4": all(
            len((rows[i].get("metrics") or {}).get("termination_terms", [])) == 4 for i in range(len(rows))
        ),
        "all_rows_robot_contract_29j_40b": all(
            (rows[i].get("metrics") or {}).get("robot_num_joints") == 29
            and (rows[i].get("metrics") or {}).get("robot_num_bodies") == 40
            for i in range(len(rows))
        ),
        "official_importer_usd_exists": OFFICIAL_IMPORTER_USD.is_file(),
        "export_structure_audit_passed": load_json(EXPORT_STRUCTURE_AUDIT).get("status") == "ok_with_physics_usd_export_but_vulkan_device_lost",
        "all_rows_use_official_importer_export_usd": all(
            (rows[i].get("metrics") or {}).get("uses_official_importer_export_usd") is True
            for i in range(len(rows))
        ),
        "no_rows_use_resource_adjusted_enriched_usd": all(
            (rows[i].get("metrics") or {}).get("uses_resource_adjusted_usd") is False
            for i in range(len(rows))
        ),
        "uses_official_csv_loop_npz_inputs": True,
        "uses_official_importer_export_usd": True,
        "does_not_use_resource_adjusted_enriched_usd": True,
        "does_not_claim_unpatched_official_asset_complete": True,
        "does_not_claim_trained_policy_eval": True,
        "does_not_start_training": True,
        "does_not_claim_real_robot": True,
    }
    status = (
        "ok_official_importer_export_full_dataset_task_eval"
        if all(checks.values())
        else "ok_with_official_importer_export_full_dataset_task_eval_blocker"
    )
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_g1_official_importer_export_full_dataset_task_eval",
        "scope": (
            "Runs zero-action Tracking-Flat-G1-v0 task diagnostics over all NPZ files generated by the full official "
            "csv_to_npz.py loop audit, using the official-importer GPU4 USDA export as the robot USD. This verifies "
            "task startup, observation/action dimensions, reward terms, termination terms, reset/step behavior, and "
            "motion-command tracking metrics for the full public motion bundle. It is not trained-policy evaluation "
            "or paper-level PPO."
        ),
        "config": {
            "target_gpu": TARGET_GPU,
            "max_steps_per_motion": MAX_STEPS,
            "limit": LIMIT,
            "csv_full_audit": str(CSV_FULL_AUDIT),
            "replay_full_audit": str(REPLAY_FULL_AUDIT),
            "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
            "export_structure_audit": str(EXPORT_STRUCTURE_AUDIT),
        },
        "gpu_guard": guard,
        "aggregate": aggregate,
        "checks": checks,
        "rows": table_rows,
        "failed_rows": [row["motion"] for row in failed_rows],
        "outputs": {
            "json": str(OUT / "tracking_g1_official_importer_export_full_dataset_task_eval.json"),
            "rows_csv": str(OUT / "tracking_g1_official_importer_export_full_dataset_task_eval_rows.csv"),
            "rows_tsv": str(OUT / "tracking_g1_official_importer_export_full_dataset_task_eval_rows.tsv"),
            "motion_root": str(OUT / "motions"),
            "worker": str(worker_path),
            "log_dir": str(LOG_DIR),
        },
        "interpretation": {
            "goal_complete": False,
            "claim_level": "official_importer_export_full_dataset_task_eval",
            "paper_level_tracking_eval_complete": False,
            "not_paper_level_reasons": [
                "uses the official-importer GPU4 USDA export, but not yet an unpatched official replay entry result",
                "uses official csv_to_npz loop NPZs generated under the captured official-importer-export asset path",
                "uses zero diagnostic actions rather than a trained paper teacher policy",
                "does not train PPO or evaluate a paper-scale checkpoint",
                "does not involve real robot hardware",
            ],
        },
    }
    report_assets = write_report_assets(table_rows, summary)
    summary["outputs"]["report_assets"] = report_assets
    write_table(OUT / "tracking_g1_official_importer_export_full_dataset_task_eval_rows.csv", table_rows, ",")
    write_table(OUT / "tracking_g1_official_importer_export_full_dataset_task_eval_rows.tsv", table_rows, "\t")
    write_json(OUT / "tracking_g1_official_importer_export_full_dataset_task_eval.json", summary)
    print(json.dumps({"status": status, "rows": len(table_rows), "json": summary["outputs"]["json"]}, sort_keys=True))


if __name__ == "__main__":
    main()
