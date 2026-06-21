#!/usr/bin/env python3
"""Run a full-bundle task diagnostic on the FK-repaired motion candidate.

This launches one IsaacLab/Kit AppLauncher process with the official-importer
G1 USDA and the non-Kit FK-repaired full public-motion bundle. It runs a
zero-action `Tracking-Flat-G1-v0` diagnostic for the full 11960-frame bundle to
verify that the repaired motion target can be consumed by the official task
stack before any PPO retraining is attempted.
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
OUT = ROOT / "res/tracking/g1_official_importer_export_fk_repaired_full_bundle_task_eval"
REPORT = ROOT / "res/report_assets/official_importer_export_fk_repaired_full_bundle_task_eval"
LOG_DIR = ROOT / "logs/tracking_g1_official_importer_export_fk_repaired_full_bundle_task_eval"
FAILED_DIR = ROOT / "res/failed_runs/tracking_g1_official_importer_export_fk_repaired_full_bundle_task_eval"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
FK_BUNDLE_JSON = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_motion_npz/"
    "tracking_g1_official_csv_loop_full_bundle_fk_repaired_motion_npz.json"
)
FK_BUNDLE_NPZ = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_motion_npz/"
    "official_csv_loop_full_public_motion_bundle_fk_repaired.npz"
)
OFFICIAL_IMPORTER_USD = (
    ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda"
)
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
TARGET_GPU = int(os.environ.get("BM_FK_REPAIRED_FULL_TASK_GPU", "4"))
MAX_STEPS = int(os.environ.get("BM_FK_REPAIRED_FULL_TASK_MAX_STEPS", "11960"))
STALL_SECONDS = int(os.environ.get("BM_FK_REPAIRED_FULL_TASK_STALL_SECONDS", "1200"))
WATCH_GPUS = [4, 7]
WANGJC_PATH_MARKER = "/mnt/infini-data/test/wangjc/"


WORKER_CODE = r"""
import json
import math
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

    def summarize(values):
        finite = [float(v) for v in values if math.isfinite(float(v))]
        if not finite:
            return {"count": 0, "mean": None, "min": None, "max": None}
        mean = sum(finite) / len(finite)
        var = sum((v - mean) ** 2 for v in finite) / len(finite)
        return {"count": len(finite), "mean": mean, "std": math.sqrt(var), "min": min(finite), "max": max(finite)}

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
    env_cfg.seed = int(os.environ.get("BM_SEED", "20260690"))

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print("BM_SENTINEL:env_created", flush=True)
    obs, extras = env.reset()
    print("BM_SENTINEL:env_reset", flush=True)

    action_dim = int(env.unwrapped.action_manager.total_action_dim)
    action = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
    command = env.unwrapped.command_manager.get_term("motion")
    step_count = min(int(command.motion.time_step_total), MAX_STEPS)
    reward_values = []
    terminated_counts = []
    truncated_counts = []
    metric_series = {}
    boundary_rows = []
    for i in range(step_count):
        obs, reward, terminated, truncated, extras = env.step(action)
        reward_values.append(float(reward.detach().cpu().mean().item()))
        terminated_counts.append(int(terminated.detach().cpu().sum().item()))
        truncated_counts.append(int(truncated.detach().cpu().sum().item()))
        for name, value in command.metrics.items():
            if hasattr(value, "detach") and value.numel() > 0:
                metric_series.setdefault(name, []).append(float(value.detach().cpu().mean().item()))
        if (i + 1) % 299 == 0 or (i + 1) == step_count:
            boundary_rows.append(
                {
                    "step": i + 1,
                    "reward_mean_so_far": sum(reward_values) / len(reward_values),
                    "terminated_total_so_far": sum(terminated_counts),
                    "truncated_total_so_far": sum(truncated_counts),
                    "error_anchor_pos": metric_series.get("error_anchor_pos", [None])[-1],
                    "error_body_pos": metric_series.get("error_body_pos", [None])[-1],
                    "error_joint_pos": metric_series.get("error_joint_pos", [None])[-1],
                }
            )
        if (i + 1) % 500 == 0 or (i + 1) == step_count:
            print(f"BM_SENTINEL:env_step={i + 1}/{step_count}", flush=True)

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
        "reward_terms": list(env.unwrapped.reward_manager.active_terms),
        "termination_terms": list(env.unwrapped.termination_manager.active_terms),
        "command_terms": list(env.unwrapped.command_manager.active_terms),
        "step_count": step_count,
        "reward": summarize(reward_values),
        "terminated_total": int(sum(terminated_counts)),
        "truncated_total": int(sum(truncated_counts)),
        "done_total": int(sum(terminated_counts) + sum(truncated_counts)),
        "command_metrics_final": command_metrics_final,
        "command_metrics_timeseries": {name: summarize(values) for name, values in metric_series.items()},
        "boundary_rows": boundary_rows,
        "robot_num_joints": int(env.unwrapped.scene["robot"].num_joints),
        "robot_num_bodies": int(env.unwrapped.scene["robot"].num_bodies),
        "usd_path": str(ROBOT_USD),
        "uses_official_importer_export_usd": True,
        "uses_fk_repaired_motion_bundle": True,
        "uses_resource_adjusted_usd": False,
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


def install_signal_marker(summary_path: Path, log_path: Path) -> None:
    def _handler(signum: int, _frame: Any) -> None:
        marker = {
            "status": "terminated_by_signal",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "signal": signum,
            "experiment_type": "tracking_g1_official_importer_export_fk_repaired_full_bundle_task_eval",
            "scope": "Signal marker for a native/outer termination before normal wrapper summary finalization.",
            "outputs": {"json": str(summary_path), "log": str(log_path)},
            "interpretation": {
                "claim_level": "failed_gate_signal_marker",
                "goal_complete": False,
                "not_paper_level_reasons": [
                    "the wrapper was terminated before task metrics were written",
                    "no PPO, replay, DAgger, VAE/diffusion, TensorRT, or real-robot evidence was produced",
                ],
            },
        }
        try:
            write_json(summary_path, marker)
        finally:
            raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


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
        ["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    )
    rows = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            rows.append({"gpu_bus_id": parts[0], "pid": int(parts[1]), "process_name": parts[2], "used_memory_mb": int(parts[3])})
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
            if Path(f"/proc/{item['pid']}").exists():
                try:
                    os.kill(item["pid"], signal.SIGKILL)
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
    path = guard_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_gpu47_wangjc_fk_repaired_task_eval_guard.json"
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


def env_vars(metrics_json: Path) -> dict[str, str]:
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
            "BM_MOTION_FILE": str(FK_BUNDLE_NPZ),
            "BM_TASK_METRICS_JSON": str(metrics_json),
            "BM_TARGET_GPU": str(TARGET_GPU),
            "BM_DEVICE": f"cuda:{TARGET_GPU}",
            "BM_MAX_STEPS": str(MAX_STEPS),
            "BM_SEED": "20260691",
        }
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def run_worker(worker_path: Path, metrics_json: Path, log_path: Path) -> dict[str, Any]:
    command = [str(TRACKING_PY), str(worker_path), "--headless", "--device", f"cuda:{TARGET_GPU}"]
    start = time.time()
    stalled = False
    last_size = -1
    last_change = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env_vars(metrics_json),
            text=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while proc.poll() is None:
            time.sleep(10)
            current_size = log_path.stat().st_size if log_path.is_file() else 0
            elapsed = round(time.time() - start, 1)
            print(
                json.dumps(
                    {
                        "parent_heartbeat": "fk_repaired_full_bundle_task_eval",
                        "elapsed_seconds": elapsed,
                        "log_size_bytes": current_size,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
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
        "status": "ok" if ok else "failed",
        "returncode": proc.returncode,
        "duration_seconds": round(time.time() - start, 3),
        "stalled": stalled,
        "markers": markers,
        "metrics": metrics,
        "metrics_json": str(metrics_json) if metrics_json.is_file() else "",
        "log": str(log_path),
    }


def summarize(values: list[float]) -> dict[str, Any]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = sum(finite) / len(finite)
    var = sum((v - mean) ** 2 for v in finite) / len(finite)
    return {"count": len(finite), "mean": mean, "std": math.sqrt(var), "min": min(finite), "max": max(finite)}


def write_boundary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "step",
        "reward_mean_so_far",
        "terminated_total_so_far",
        "truncated_total_so_far",
        "error_anchor_pos",
        "error_body_pos",
        "error_joint_pos",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_report_assets(summary: dict[str, Any]) -> dict[str, str]:
    REPORT.mkdir(parents=True, exist_ok=True)
    metrics = summary["run"]["metrics"]
    boundary_rows = metrics.get("boundary_rows", [])
    boundary_csv = REPORT / "fk_repaired_full_bundle_task_eval_boundaries.csv"
    write_boundary_csv(boundary_csv, boundary_rows)
    summary_csv = REPORT / "fk_repaired_full_bundle_task_eval_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"], lineterminator="\n")
        writer.writeheader()
        for key, value in {
            "step_count": metrics.get("step_count"),
            "reward_mean": (metrics.get("reward") or {}).get("mean"),
            "done_total": metrics.get("done_total"),
            "error_anchor_pos_final": (metrics.get("command_metrics_final") or {}).get("error_anchor_pos"),
            "error_body_pos_final": (metrics.get("command_metrics_final") or {}).get("error_body_pos"),
            "error_joint_pos_final": (metrics.get("command_metrics_final") or {}).get("error_joint_pos"),
        }.items():
            writer.writerow({"metric": key, "value": value})
    readme = REPORT / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# FK-Repaired Full-Bundle Task Eval",
                "",
                "Zero-action full-bundle `Tracking-Flat-G1-v0` diagnostic using the official-importer-export G1 USDA and the non-Kit FK-repaired motion bundle.",
                "",
                "Claim boundary: local virtual task-contract diagnostic only. This is not PPO, not paper-level tracking, not DAgger, not VAE/diffusion, and not real robot evidence.",
                "",
                f"- Source audit: `{summary['outputs']['json']}`",
                f"- Boundary CSV: `{boundary_csv}`",
                f"- Summary CSV: `{summary_csv}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    asset_json = REPORT / "fk_repaired_full_bundle_task_eval_assets.json"
    assets = {
        "status": "ok_fk_repaired_full_bundle_task_eval_report_assets" if summary["status"].startswith("ok_") else "failed",
        "claim_level": summary["interpretation"]["claim_level"],
        "source_audit": summary["outputs"]["json"],
        "assets": {"boundary_csv": str(boundary_csv), "summary_csv": str(summary_csv), "readme": str(readme)},
        "metrics": {
            "step_count": metrics.get("step_count"),
            "reward_mean": (metrics.get("reward") or {}).get("mean"),
            "done_total": metrics.get("done_total"),
            "error_body_pos_final": (metrics.get("command_metrics_final") or {}).get("error_body_pos"),
        },
    }
    write_json(asset_json, assets)
    return assets["assets"] | {"asset_json": str(asset_json)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    worker_path = OUT / "tracking_g1_official_importer_export_fk_repaired_full_bundle_task_eval_worker.py"
    worker_path.write_text(textwrap.dedent(WORKER_CODE), encoding="utf-8")
    guard = kill_wangjc_on_watch_gpus()
    metrics_json = OUT / "fk_repaired_full_bundle_task_eval_metrics.json"
    log_path = LOG_DIR / "fk_repaired_full_bundle_task_eval.log"
    summary_path = OUT / "tracking_g1_official_importer_export_fk_repaired_full_bundle_task_eval.json"
    install_signal_marker(summary_path, log_path)
    result = run_worker(worker_path, metrics_json, log_path)
    fk_bundle = load_json(FK_BUNDLE_JSON)
    metrics = result.get("metrics") or {}
    checks = {
        "fk_bundle_status_ok": fk_bundle.get("status") == "ok_official_csv_loop_full_bundle_fk_repaired_motion_npz",
        "fk_bundle_npz_exists": FK_BUNDLE_NPZ.is_file(),
        "official_importer_usd_exists": OFFICIAL_IMPORTER_USD.is_file(),
        "process_returned_zero": result["returncode"] == 0,
        "metrics_status_ok": metrics.get("status") == "ok",
        "full_11960_steps_completed": metrics.get("step_count") == MAX_STEPS,
        "action_dim_29": metrics.get("action_dim") == 29,
        "policy_obs_dim_160": metrics.get("policy_observation_dim") == 160,
        "critic_obs_dim_286": metrics.get("critic_observation_dim") == 286,
        "reward_terms_9": len(metrics.get("reward_terms", [])) == 9,
        "termination_terms_4": len(metrics.get("termination_terms", [])) == 4,
        "robot_contract_29j_40b": metrics.get("robot_num_joints") == 29 and metrics.get("robot_num_bodies") == 40,
        "uses_official_importer_export_usd": metrics.get("uses_official_importer_export_usd") is True,
        "uses_fk_repaired_motion_bundle": metrics.get("uses_fk_repaired_motion_bundle") is True,
        "does_not_claim_paper_level_tracking": metrics.get("paper_level_rollout") is False,
        "does_not_start_training": metrics.get("ppo_training") is False,
        "does_not_claim_real_robot": metrics.get("real_robot") is False,
    }
    status = "ok_official_importer_export_fk_repaired_full_bundle_task_eval" if all(checks.values()) else "failed"
    if status != "ok_official_importer_export_fk_repaired_full_bundle_task_eval":
        failed_copy = FAILED_DIR / "fk_repaired_full_bundle_task_eval.log"
        if log_path.is_file():
            failed_copy.write_text(log_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_g1_official_importer_export_fk_repaired_full_bundle_task_eval",
        "scope": "Full 11960-step zero-action Tracking-Flat-G1-v0 diagnostic on the FK-repaired full public-motion bundle.",
        "config": {
            "target_gpu": TARGET_GPU,
            "max_steps": MAX_STEPS,
            "motion_npz": str(FK_BUNDLE_NPZ),
            "robot_usd": str(OFFICIAL_IMPORTER_USD),
            "stall_seconds": STALL_SECONDS,
        },
        "gpu_guard": guard,
        "run": result,
        "checks": checks,
        "outputs": {
            "json": str(summary_path),
            "metrics_json": str(metrics_json),
            "log": str(log_path),
            "worker": str(worker_path),
            "report_assets_dir": str(REPORT),
        },
        "interpretation": {
            "claim_level": "local_virtual_fk_repaired_full_bundle_task_contract_diagnostic",
            "goal_complete": False,
            "not_paper_level_reasons": [
                "zero-action task diagnostic only",
                "FK-repaired bundle is non-Kit and not unmodified official csv_to_npz output",
                "no PPO policy is trained or evaluated",
                "no DAgger, VAE, diffusion, TensorRT, or real-robot evidence is produced",
            ],
        },
    }
    write_json(Path(summary["outputs"]["json"]), summary)
    assets = write_report_assets(summary)
    summary["outputs"]["report_assets"] = assets
    write_json(Path(summary["outputs"]["json"]), summary)
    print(json.dumps({"status": status, "json": summary["outputs"]["json"]}, sort_keys=True))
    if not status.startswith("ok_"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
