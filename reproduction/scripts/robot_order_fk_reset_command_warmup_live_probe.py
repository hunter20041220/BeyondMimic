#!/usr/bin/env python3
"""Live probe for the robot-order FK reset/command warmup bottleneck.

The previous source-linked audit found a deterministic step-0 done spike in
the robot-order FK PPO eval traces. This script launches a small IsaacLab
Tracking-Flat-G1-v0 process and measures whether manually warming the motion
command targets after reset reduces the endpoint-z termination condition before
we spend time on another full PPO run.
"""

from __future__ import annotations

import csv
import json
import os
import select
import signal
import subprocess
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/robot_order_fk_reset_command_warmup_live_probe"
LOG_DIR = ROOT / "logs/tracking_robot_order_fk_reset_command_warmup_live_probe"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
OFFICIAL_IMPORTER_USD = ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda"
ROBOT_ORDER_MOTION_NPZ = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/"
    "official_csv_loop_full_public_motion_bundle_fk_repaired_robot_order.npz"
)
RESET_ALIGNMENT_AUDIT = (
    ROOT
    / "res/tracking/robot_order_fk_reset_termination_alignment_audit/"
    "robot_order_fk_reset_termination_alignment_audit.json"
)
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)

TARGET_GPU = int(os.environ.get("BM_RESET_WARMUP_GPU", "4"))
NUM_ENVS = int(os.environ.get("BM_RESET_WARMUP_NUM_ENVS", "256"))
STALL_SECONDS = int(os.environ.get("BM_RESET_WARMUP_STALL_SECONDS", "900"))


WORKER_CODE = r"""
import json
import math
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(os.environ["BM_WORKER_METRICS_JSON"])
ROBOT_USD = Path(os.environ["BM_ROBOT_USD"])
MOTION_FILE = Path(os.environ["BM_MOTION_FILE"])
NUM_ENVS = int(os.environ["BM_NUM_ENVS"])
ENDPOINT_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]


def write_payload(payload):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("BM_SENTINEL:worker_metrics_written=" + str(OUT), flush=True)


def stats_tensor(tensor):
    if tensor.numel() == 0:
        return {"count": 0, "mean": None, "min": None, "max": None, "std": None}
    t = tensor.detach().float().reshape(-1).cpu()
    finite = t[torch.isfinite(t)]
    if finite.numel() == 0:
        return {"count": 0, "mean": None, "min": None, "max": None, "std": None}
    return {
        "count": int(finite.numel()),
        "mean": float(finite.mean().item()),
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
        "std": float(finite.std(unbiased=False).item()) if finite.numel() > 1 else 0.0,
    }


def tensor_int(value):
    if hasattr(value, "detach"):
        return int(value.detach().cpu().sum().item())
    return int(value)


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
args.fast_shutdown = True
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

    def snapshot(label, command):
        body_names = list(command.cfg.body_names)
        endpoint_indexes = [body_names.index(name) for name in ENDPOINT_NAMES if name in body_names]
        body_target = command.body_pos_relative_w.detach()
        robot_body = command.robot_body_pos_w.detach()
        body_error = torch.linalg.norm(body_target - robot_body, dim=-1)
        endpoint_z_error = torch.abs(body_target[:, endpoint_indexes, 2] - robot_body[:, endpoint_indexes, 2])
        endpoint_manual_done = torch.any(endpoint_z_error > 0.25, dim=-1)
        anchor_error = torch.linalg.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=-1)
        time_steps = command.time_steps.detach().float()
        return {
            "label": label,
            "body_name_count": len(body_names),
            "endpoint_names_present": [name for name in ENDPOINT_NAMES if name in body_names],
            "endpoint_indexes": endpoint_indexes,
            "time_steps": stats_tensor(time_steps),
            "body_pos_relative_abs_max": float(body_target.abs().max().detach().cpu().item()),
            "robot_body_pos_abs_max": float(robot_body.abs().max().detach().cpu().item()),
            "body_error_m": stats_tensor(body_error),
            "anchor_error_m": stats_tensor(anchor_error),
            "endpoint_z_error_m": stats_tensor(endpoint_z_error),
            "manual_endpoint_z_done_count": int(endpoint_manual_done.detach().cpu().sum().item()),
            "manual_endpoint_z_done_rate": float(endpoint_manual_done.float().mean().detach().cpu().item()),
            "command_metric_error_body_pos_mean": (
                float(command.metrics["error_body_pos"].detach().float().mean().cpu().item())
                if "error_body_pos" in command.metrics
                else None
            ),
        }

    env_cfg = G1FlatEnvCfg()
    env_cfg.scene.num_envs = NUM_ENVS
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
    env_cfg.seed = int(os.environ.get("BM_SEED", "20260760"))

    print("BM_SENTINEL:before_gym_make", flush=True)
    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    print("BM_SENTINEL:env_created", flush=True)
    obs, extras = env.reset()
    print("BM_SENTINEL:env_reset", flush=True)
    command = env.unwrapped.command_manager.get_term("motion")

    before = snapshot("after_reset_before_command_warmup", command)
    env.unwrapped.command_manager.compute(dt=env.unwrapped.step_dt)
    after_warmup = snapshot("after_manual_command_manager_compute", command)

    action_dim = int(env.unwrapped.action_manager.total_action_dim)
    zero_action = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
    obs, reward, terminated, truncated, extras = env.step(zero_action)
    after_step = snapshot("after_zero_action_step_following_warmup", command)
    terminated_count = tensor_int(terminated)
    truncated_count = tensor_int(truncated)
    done_count = terminated_count + truncated_count
    step_summary = {
        "terminated_count": terminated_count,
        "truncated_count": truncated_count,
        "done_count": done_count,
        "done_rate": done_count / float(env.unwrapped.num_envs),
        "reward_mean": float(reward.detach().float().mean().cpu().item()),
    }

    pre_mean = before["endpoint_z_error_m"]["mean"]
    post_mean = after_warmup["endpoint_z_error_m"]["mean"]
    pre_done = before["manual_endpoint_z_done_rate"]
    post_done = after_warmup["manual_endpoint_z_done_rate"]
    checks = {
        "uses_official_importer_export_usd": True,
        "uses_robot_order_fk_repaired_bundle": True,
        "num_envs_positive": int(env.unwrapped.num_envs) == NUM_ENVS,
        "endpoint_names_found": len(after_warmup["endpoint_names_present"]) == len(ENDPOINT_NAMES),
        "pre_warmup_manual_endpoint_done_rate_high": pre_done > 0.90,
        "warmup_reduces_endpoint_z_error_mean": post_mean is not None and pre_mean is not None and post_mean < pre_mean,
        "warmup_reduces_manual_endpoint_done_rate": post_done < pre_done,
        "post_warmup_manual_endpoint_done_rate_low": post_done < 0.05,
        "zero_action_step_after_warmup_not_all_done": step_summary["done_rate"] < 0.95,
        "does_not_claim_paper_level_tracking": True,
        "does_not_claim_goal_complete": True,
        "does_not_claim_real_robot": True,
    }
    if checks["post_warmup_manual_endpoint_done_rate_low"] and checks["zero_action_step_after_warmup_not_all_done"]:
        diagnosis = "command_warmup_clears_reset_endpoint_z_spike"
    elif checks["warmup_reduces_endpoint_z_error_mean"]:
        diagnosis = "command_warmup_partially_reduces_reset_endpoint_z_spike"
    else:
        diagnosis = "command_warmup_does_not_clear_reset_endpoint_z_spike"

    payload = {
        "status": "ok_robot_order_fk_reset_command_warmup_live_probe",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "robot_order_fk_reset_command_warmup_live_probe_worker",
        "scope": "Live IsaacLab reset/command warmup diagnostic for the robot-order FK repaired tracking pipeline.",
        "device": str(env.unwrapped.device),
        "target_gpu": int(target_gpu),
        "num_envs": int(env.unwrapped.num_envs),
        "step_dt": float(env.unwrapped.step_dt),
        "physics_dt": float(env.unwrapped.physics_dt),
        "action_dim": action_dim,
        "policy_observation_dim": int(env.unwrapped.observation_manager.group_obs_dim["policy"][0]),
        "critic_observation_dim": int(env.unwrapped.observation_manager.group_obs_dim["critic"][0]),
        "robot_num_joints": int(env.unwrapped.scene["robot"].num_joints),
        "robot_num_bodies": int(env.unwrapped.scene["robot"].num_bodies),
        "robot_usd": str(ROBOT_USD),
        "motion_file": str(MOTION_FILE),
        "snapshots": [before, after_warmup, after_step],
        "step_summary": step_summary,
        "checks": checks,
        "diagnosis": diagnosis,
        "interpretation": {
            "claim_level": "live_tracking_reset_diagnostic",
            "goal_complete": False,
            "paper_level_tracking_reproduced": False,
            "real_robot": False,
            "next_step_if_cleared": (
                "Patch local tracking train/eval wrappers to warm command targets immediately after reset, then "
                "rerun full robot-order FK tracking eval/PPO."
            ),
            "next_step_if_not_cleared": (
                "Inspect motion target frame, endpoint body mapping, and termination thresholds before another "
                "full PPO run."
            ),
        },
    }
    write_payload(payload)
    print("BM_SENTINEL:live_probe_success", flush=True)
    os._exit(0)
except BaseException as exc:
    payload = {
        "status": "failed_robot_order_fk_reset_command_warmup_live_probe",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "exception": repr(exc),
        "traceback": traceback.format_exc(),
        "robot_usd": str(ROBOT_USD),
        "motion_file": str(MOTION_FILE),
    }
    write_payload(payload)
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    tmp.replace(path)


def env_for(worker_metrics: Path) -> dict[str, str]:
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
            "BM_WORKER_METRICS_JSON": str(worker_metrics),
            "BM_ROBOT_USD": str(OFFICIAL_IMPORTER_USD),
            "BM_MOTION_FILE": str(ROBOT_ORDER_MOTION_NPZ),
            "BM_NUM_ENVS": str(NUM_ENVS),
            "BM_TARGET_GPU": str(TARGET_GPU),
            "BM_DEVICE": f"cuda:{TARGET_GPU}",
            "BM_SEED": "20260760",
        }
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def run_worker(worker: Path, worker_metrics: Path) -> dict[str, Any]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "robot_order_fk_reset_command_warmup_live_probe.log"
    cmd = [str(TRACKING_PY), str(worker), "--headless", "--device", f"cuda:{TARGET_GPU}"]
    start = time.time()
    last_change = time.time()
    last_size = -1
    stdout_tail: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env_for(worker_metrics),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert proc.stdout is not None
        while proc.poll() is None:
            ready, _, _ = select.select([proc.stdout], [], [], 5)
            if ready:
                line = proc.stdout.readline()
                if line:
                    stdout_tail.append(line.rstrip())
                    stdout_tail = stdout_tail[-80:]
                    log_file.write(line)
                    log_file.flush()
                    if line.startswith("BM_SENTINEL") or "Traceback" in line or "Error" in line:
                        print(line.rstrip(), flush=True)
            size = log_path.stat().st_size if log_path.is_file() else 0
            if size != last_size:
                last_size = size
                last_change = time.time()
            elif time.time() - last_change > STALL_SECONDS:
                log_file.write(f"\nBM_STALL_TIMEOUT:no_log_progress_for_{STALL_SECONDS}s\n")
                log_file.flush()
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait(timeout=30)
                break
        for line in proc.stdout:
            stdout_tail.append(line.rstrip())
            stdout_tail = stdout_tail[-80:]
            log_file.write(line)
    duration = time.time() - start
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "duration_seconds": duration,
        "log_path": str(log_path),
        "stdout_tail": stdout_tail,
    }


def snapshot_rows(worker: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in worker.get("snapshots", []):
        endpoint_stats = item.get("endpoint_z_error_m", {})
        body_stats = item.get("body_error_m", {})
        anchor_stats = item.get("anchor_error_m", {})
        rows.append(
            {
                "stage": item.get("label"),
                "manual_endpoint_z_done_count": item.get("manual_endpoint_z_done_count"),
                "manual_endpoint_z_done_rate": item.get("manual_endpoint_z_done_rate"),
                "endpoint_z_error_mean_m": endpoint_stats.get("mean"),
                "endpoint_z_error_max_m": endpoint_stats.get("max"),
                "body_error_mean_m": body_stats.get("mean"),
                "body_error_max_m": body_stats.get("max"),
                "anchor_error_mean_m": anchor_stats.get("mean"),
                "body_pos_relative_abs_max": item.get("body_pos_relative_abs_max"),
                "time_steps_mean": item.get("time_steps", {}).get("mean"),
            }
        )
    return rows


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    worker = summary.get("worker_metrics", {})
    checks = worker.get("checks", {})
    rows = snapshot_rows(worker)
    lines = [
        "# Robot-Order FK Reset Command-Warmup Live Probe",
        "",
        f"Generated: {summary['generated_at']}",
        "",
        "## Goal",
        "",
        "Diagnose whether the robot-order FK tracking step-0 done spike is caused by stale/zero motion command targets immediately after reset.",
        "",
        "## Result",
        "",
        f"- Status: `{summary['status']}`",
        f"- Worker status: `{worker.get('status')}`",
        f"- Diagnosis: `{worker.get('diagnosis')}`",
        f"- GPU: `{summary['config']['target_gpu']}`",
        f"- Num envs: `{summary['config']['num_envs']}`",
        f"- Log: `{summary['outputs']['log']}`",
        "",
        "## Snapshot Metrics",
        "",
        "| Stage | Endpoint done rate | Endpoint z mean (m) | Endpoint z max (m) | Body error mean (m) | Body target abs max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {stage} | {done} | {zmean} | {zmax} | {body} | {absmax} |".format(
                stage=row["stage"],
                done=row["manual_endpoint_z_done_rate"],
                zmean=row["endpoint_z_error_mean_m"],
                zmax=row["endpoint_z_error_max_m"],
                body=row["body_error_mean_m"],
                absmax=row["body_pos_relative_abs_max"],
            )
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
        ]
    )
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            worker.get("interpretation", {}).get("next_step_if_cleared", ""),
            "",
            "This is a live diagnostic, not paper-level tracking reproduction, not PPO training, not DAgger, not VAE/diffusion guidance, and not a real-robot result.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    worker = OUT / "robot_order_fk_reset_command_warmup_live_probe_worker.py"
    worker_metrics_path = OUT / "robot_order_fk_reset_command_warmup_live_probe_worker_metrics.json"
    worker.write_text(WORKER_CODE, encoding="utf-8")

    run = run_worker(worker, worker_metrics_path)
    worker_metrics = load_json(worker_metrics_path)
    reset_alignment = load_json(RESET_ALIGNMENT_AUDIT)
    passed = run["returncode"] == 0 and worker_metrics.get("status") == "ok_robot_order_fk_reset_command_warmup_live_probe"
    checks = worker_metrics.get("checks", {}) if worker_metrics else {}
    status = "ok_robot_order_fk_reset_command_warmup_live_probe" if passed else "failed_robot_order_fk_reset_command_warmup_live_probe"

    rows = snapshot_rows(worker_metrics)
    tsv_path = OUT / "robot_order_fk_reset_command_warmup_live_probe.tsv"
    write_tsv(
        tsv_path,
        rows,
        [
            "stage",
            "manual_endpoint_z_done_count",
            "manual_endpoint_z_done_rate",
            "endpoint_z_error_mean_m",
            "endpoint_z_error_max_m",
            "body_error_mean_m",
            "body_error_max_m",
            "anchor_error_mean_m",
            "body_pos_relative_abs_max",
            "time_steps_mean",
        ],
    )

    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "robot_order_fk_reset_command_warmup_live_probe",
        "scope": "Live IsaacLab command warmup diagnostic before rerunning robot-order FK tracking PPO.",
        "config": {
            "target_gpu": TARGET_GPU,
            "num_envs": NUM_ENVS,
            "stall_seconds": STALL_SECONDS,
            "tracking_python": str(TRACKING_PY),
            "robot_usd": str(OFFICIAL_IMPORTER_USD),
            "motion_file": str(ROBOT_ORDER_MOTION_NPZ),
        },
        "run": run,
        "worker_metrics": worker_metrics,
        "previous_reset_alignment_audit": {
            "path": str(RESET_ALIGNMENT_AUDIT),
            "status": reset_alignment.get("status"),
            "recommended_next_live_probe": reset_alignment.get("recommended_next_live_probe"),
        },
        "checks": {
            "worker_returned_zero": run["returncode"] == 0,
            "worker_status_ok": worker_metrics.get("status") == "ok_robot_order_fk_reset_command_warmup_live_probe",
            "endpoint_names_found": checks.get("endpoint_names_found") is True,
            "pre_warmup_manual_endpoint_done_rate_high": checks.get("pre_warmup_manual_endpoint_done_rate_high") is True,
            "warmup_reduces_endpoint_z_error_mean": checks.get("warmup_reduces_endpoint_z_error_mean") is True,
            "post_warmup_manual_endpoint_done_rate_low": checks.get("post_warmup_manual_endpoint_done_rate_low") is True,
            "zero_action_step_after_warmup_not_all_done": checks.get("zero_action_step_after_warmup_not_all_done") is True,
            "does_not_claim_paper_level_tracking": True,
            "does_not_claim_goal_complete": True,
            "does_not_claim_real_robot": True,
        },
        "interpretation": {
            "claim_level": "tracking_reset_live_diagnostic",
            "goal_complete": False,
            "paper_level_tracking_reproduced": False,
            "real_robot": False,
            "next_mainline_decision": worker_metrics.get("diagnosis", "worker_failed"),
        },
        "outputs": {
            "json": str(OUT / "robot_order_fk_reset_command_warmup_live_probe.json"),
            "tsv": str(tsv_path),
            "md": str(OUT / "robot_order_fk_reset_command_warmup_live_probe.md"),
            "worker": str(worker),
            "worker_metrics": str(worker_metrics_path),
            "log": run["log_path"],
        },
    }
    json_path = OUT / "robot_order_fk_reset_command_warmup_live_probe.json"
    md_path = OUT / "robot_order_fk_reset_command_warmup_live_probe.md"
    write_json(json_path, summary)
    write_markdown(md_path, summary)
    print(json.dumps({"status": status, "json": str(json_path), "diagnosis": worker_metrics.get("diagnosis")}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
