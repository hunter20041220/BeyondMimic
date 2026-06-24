#!/usr/bin/env python3
"""Capture an official IsaacLab observation_manager policy sample.

This is a runtime gate for the MuJoCo native observation adapter.  It creates
the official `Tracking-Flat-G1-v0` environment with one FK-repaired G1 motion,
resets it, computes the official IsaacLab observation manager output, and saves
the 160-D policy observation plus term slices.  It does not train, evaluate a
policy, render video, or claim MuJoCo parity by itself.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import select
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = Path(os.environ.get("BM_ISAACLAB_OBS_SAMPLE_OUT", ROOT / "res/audits/isaaclab_observation_manager_sample_gate"))
LOG_DIR = ROOT / "logs/isaaclab_observation_manager_sample_gate"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
ROBOT_USD = ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda"
MOTION_NPZ = Path(
    os.environ.get(
        "BM_ISAACLAB_OBS_SAMPLE_MOTION",
        ROOT
        / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/motions/"
        "dance1_subject1/motion.npz",
    )
)
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
JSON_OUT = OUT / "isaaclab_observation_manager_sample_gate.json"
TSV_OUT = OUT / "isaaclab_observation_manager_sample_gate.tsv"
SAMPLE_JSON = OUT / "isaaclab_policy_obs_sample.json"
SAMPLE_NPZ = OUT / "isaaclab_policy_obs_sample.npz"
MD_OUT = OUT / "isaaclab_observation_manager_sample_gate.md"
WORKER = OUT / "isaaclab_observation_manager_sample_gate_worker.py"

TARGET_GPU = int(os.environ.get("BM_ISAACLAB_OBS_SAMPLE_GPU", "4"))
TIMEOUT_SECONDS = int(os.environ.get("BM_ISAACLAB_OBS_SAMPLE_TIMEOUT", "360"))
STALL_SECONDS = int(os.environ.get("BM_ISAACLAB_OBS_SAMPLE_STALL_SECONDS", "180"))
MIN_FREE_MB = int(os.environ.get("BM_ISAACLAB_OBS_SAMPLE_MIN_FREE_MB", "20000"))
MAX_BUSY_UTIL = int(os.environ.get("BM_ISAACLAB_OBS_SAMPLE_MAX_BUSY_UTIL", "50"))
WANGJC_PATH_MARKER = "/mnt/infini-data/test/wangjc/"
CAPTURE_AFTER_ZERO_STEP = os.environ.get("BM_ISAACLAB_OBS_SAMPLE_AFTER_ZERO_STEP", "1") not in {
    "0",
    "false",
    "False",
}


WORKER_CODE = r'''
import faulthandler
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

faulthandler.enable(file=sys.stdout, all_threads=True)

OUT_JSON = Path(os.environ["BM_SAMPLE_JSON"])
OUT_NPZ = Path(os.environ["BM_SAMPLE_NPZ"])
ROBOT_USD = Path(os.environ["BM_ROBOT_USD"])
MOTION_FILE = Path(os.environ["BM_MOTION_FILE"])
TARGET_GPU = os.environ.get("BM_TARGET_GPU", "4")
CAPTURE_AFTER_ZERO_STEP = os.environ.get("BM_CAPTURE_AFTER_ZERO_STEP", "1") not in {"0", "false", "False"}


def to_list(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().tolist()
    return value


def write_stage(stage, **extra):
    payload = {
        "stage": stage,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "robot_usd": str(ROBOT_USD),
        "motion_file": str(MOTION_FILE),
    }
    payload.update(extra)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("BM_SENTINEL:obs_sample:state:" + json.dumps(payload, sort_keys=True), flush=True)


write_stage("before_import")
from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
args.device = os.environ.get("BM_DEVICE", f"cuda:{TARGET_GPU}")
args.multi_gpu = False
args.fast_shutdown = True
args.kit_args = (
    "--/renderer/multiGpu/enabled=false "
    "--/renderer/multiGpu/autoEnable=false "
    "--/renderer/multiGpu/maxGpuCount=1 "
    f"--/renderer/activeGpu={TARGET_GPU} "
    f"--/physics/cudaDevice={TARGET_GPU}"
)

write_stage("before_app", target_gpu=TARGET_GPU, device=args.device)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
write_stage("after_app", app_running=bool(simulation_app.is_running()))

try:
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import numpy as np
    import torch
    import whole_body_tracking.tasks  # noqa: F401
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

    write_stage("before_cfg")
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
    env_cfg.seed = int(os.environ.get("BM_SEED", "20260770"))
    write_stage("cfg_ready")

    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    unwrapped = env.unwrapped
    write_stage(
        "env_created",
        num_envs=int(unwrapped.num_envs),
        device=str(unwrapped.device),
        action_dim=int(unwrapped.action_manager.total_action_dim),
        robot_num_joints=int(unwrapped.scene["robot"].num_joints),
        robot_num_bodies=int(unwrapped.scene["robot"].num_bodies),
    )
    obs, extras = env.reset()
    write_stage("env_reset")

    action_dim = int(unwrapped.action_manager.total_action_dim)
    zero_action = torch.zeros((unwrapped.num_envs, action_dim), device=unwrapped.device)
    if CAPTURE_AFTER_ZERO_STEP:
        # Force a deterministic no-op step so last_action and command buffers
        # are populated through the official runtime path.
        obs, reward, terminated, truncated, extras = env.step(zero_action)
        capture_mode = "after_zero_action_step"
    else:
        reward = torch.zeros((unwrapped.num_envs,), device=unwrapped.device)
        terminated = torch.zeros((unwrapped.num_envs,), dtype=torch.bool, device=unwrapped.device)
        truncated = torch.zeros((unwrapped.num_envs,), dtype=torch.bool, device=unwrapped.device)
        capture_mode = "after_reset_no_step"

    obs_dict = unwrapped.observation_manager.compute()
    policy_obs = obs_dict["policy"].detach().cpu()
    critic_obs = obs_dict["critic"].detach().cpu()
    active_terms = unwrapped.observation_manager.active_terms
    term_dims = unwrapped.observation_manager.group_obs_term_dim
    iterable_terms = unwrapped.observation_manager.get_active_iterable_terms(0)
    policy_terms = {
        name.split("-", 1)[1]: values
        for name, values in iterable_terms
        if name.startswith("policy-")
    }
    critic_terms = {
        name.split("-", 1)[1]: values
        for name, values in iterable_terms
        if name.startswith("critic-")
    }
    command = unwrapped.command_manager.get_term("motion")
    motion_time_steps = getattr(command, "time_steps", None)
    motion_time_step_total = getattr(command.motion, "time_step_total", None)
    motion_time_steps_list = to_list(motion_time_steps) if motion_time_steps is not None else None
    robot = unwrapped.scene["robot"]

    raw_state = {
        "command_joint_pos": to_list(command.joint_pos),
        "command_joint_vel": to_list(command.joint_vel),
        "command_body_pos_w": to_list(command.body_pos_w),
        "command_body_quat_w": to_list(command.body_quat_w),
        "command_body_lin_vel_w": to_list(command.body_lin_vel_w),
        "command_body_ang_vel_w": to_list(command.body_ang_vel_w),
        "command_anchor_pos_w": to_list(command.anchor_pos_w),
        "command_anchor_quat_w": to_list(command.anchor_quat_w),
        "command_anchor_lin_vel_w": to_list(command.anchor_lin_vel_w),
        "command_anchor_ang_vel_w": to_list(command.anchor_ang_vel_w),
        "command_body_pos_relative_w": to_list(command.body_pos_relative_w),
        "command_body_quat_relative_w": to_list(command.body_quat_relative_w),
        "robot_body_pos_w": to_list(command.robot_body_pos_w),
        "robot_body_quat_w": to_list(command.robot_body_quat_w),
        "robot_body_lin_vel_w": to_list(command.robot_body_lin_vel_w),
        "robot_body_ang_vel_w": to_list(command.robot_body_ang_vel_w),
        "robot_anchor_pos_w": to_list(command.robot_anchor_pos_w),
        "robot_anchor_quat_w": to_list(command.robot_anchor_quat_w),
        "robot_anchor_lin_vel_w": to_list(command.robot_anchor_lin_vel_w),
        "robot_anchor_ang_vel_w": to_list(command.robot_anchor_ang_vel_w),
        "robot_root_pos_w": to_list(robot.data.root_pos_w),
        "robot_root_quat_w": to_list(robot.data.root_quat_w),
        "robot_root_lin_vel_b": to_list(robot.data.root_lin_vel_b),
        "robot_root_ang_vel_b": to_list(robot.data.root_ang_vel_b),
        "robot_root_lin_vel_w": to_list(robot.data.root_lin_vel_w),
        "robot_root_ang_vel_w": to_list(robot.data.root_ang_vel_w),
        "robot_joint_pos": to_list(robot.data.joint_pos),
        "robot_default_joint_pos": to_list(robot.data.default_joint_pos),
        "robot_joint_vel": to_list(robot.data.joint_vel),
        "action_manager_action": to_list(getattr(unwrapped.action_manager, "action", torch.empty(0, device=unwrapped.device))),
        "action_manager_prev_action": to_list(
            getattr(unwrapped.action_manager, "prev_action", torch.empty(0, device=unwrapped.device))
        ),
        "zero_action": to_list(zero_action),
    }

    sample = {
        "status": "ok_isaaclab_observation_manager_sample_captured",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "Tracking-Flat-G1-v0",
        "claim_level": "official_isaaclab_observation_sample_only; no_mujoco_parity_claim",
        "robot_usd": str(ROBOT_USD),
        "motion_file": str(MOTION_FILE),
        "device": str(unwrapped.device),
        "num_envs": int(unwrapped.num_envs),
        "policy_obs_shape": list(policy_obs.shape),
        "critic_obs_shape": list(critic_obs.shape),
        "policy_obs_dim": int(unwrapped.observation_manager.group_obs_dim["policy"][0]),
        "critic_obs_dim": int(unwrapped.observation_manager.group_obs_dim["critic"][0]),
        "policy_term_names": list(active_terms["policy"]),
        "critic_term_names": list(active_terms["critic"]),
        "policy_term_dims": [list(dim) for dim in term_dims["policy"]],
        "critic_term_dims": [list(dim) for dim in term_dims["critic"]],
        "policy_terms": policy_terms,
        "critic_terms": critic_terms,
        "raw_state": raw_state,
        "policy_obs": policy_obs[0].tolist(),
        "critic_obs_head": critic_obs[0, : min(64, critic_obs.shape[-1])].tolist(),
        "last_action": policy_terms.get("actions", []),
        "capture_mode": capture_mode,
        "zero_action_applied_before_capture": bool(CAPTURE_AFTER_ZERO_STEP),
        "motion_time_steps": motion_time_steps_list,
        "motion_time_step_total": int(motion_time_step_total) if motion_time_step_total is not None else None,
        "robot_anchor_body_index": int(command.robot_anchor_body_index),
        "motion_anchor_body_index": int(command.motion_anchor_body_index),
        "body_indexes": to_list(command.body_indexes),
        "reward_after_zero_step": float(reward.detach().cpu().mean().item()),
        "terminated_after_zero_step": bool(terminated.detach().cpu().any().item()),
        "truncated_after_zero_step": bool(truncated.detach().cpu().any().item()),
        "command_metrics": {
            name: float(value.detach().cpu().mean().item())
            for name, value in command.metrics.items()
            if hasattr(value, "detach") and value.numel() > 0
        },
        "checks": {
            "policy_obs_dim_160": int(unwrapped.observation_manager.group_obs_dim["policy"][0]) == 160,
            "policy_term_count_8": len(active_terms["policy"]) == 8,
            "policy_terms_expected_order": list(active_terms["policy"])
            == [
                "command",
                "motion_anchor_pos_b",
                "motion_anchor_ori_b",
                "base_lin_vel",
                "base_ang_vel",
                "joint_pos",
                "joint_vel",
                "actions",
            ],
            "critic_shared_terms_available": all(
                name in critic_terms
                for name in [
                    "command",
                    "motion_anchor_pos_b",
                    "motion_anchor_ori_b",
                    "base_lin_vel",
                    "base_ang_vel",
                    "joint_pos",
                    "joint_vel",
                    "actions",
                ]
            ),
            "raw_state_available_for_same_state_parity": all(
                name in raw_state
                for name in [
                    "command_joint_pos",
                    "command_joint_vel",
                    "command_anchor_pos_w",
                    "command_anchor_quat_w",
                    "robot_anchor_pos_w",
                    "robot_anchor_quat_w",
                    "robot_root_lin_vel_b",
                    "robot_root_ang_vel_b",
                    "robot_joint_pos",
                    "robot_default_joint_pos",
                    "robot_joint_vel",
                    "zero_action",
                ]
            ),
            "zero_action_applied_before_capture": bool(CAPTURE_AFTER_ZERO_STEP),
            "does_not_claim_mujoco_parity_or_rollout": True,
        },
        "interpretation": {
            "official_observation_manager_sample_available": True,
            "policy_terms_are_training_noisy": True,
            "critic_shared_terms_are_noise_free_reference": True,
            "mujoco_native_observation_parity_ready": False,
            "next_step": "Run a same-state MuJoCo adapter comparison against this sample; this file alone is not parity.",
        },
    }
    OUT_JSON.write_text(json.dumps(sample, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(
        OUT_NPZ,
        policy_obs=policy_obs.numpy(),
        critic_obs=critic_obs.numpy(),
        **{f"policy_{name}": np.asarray(values, dtype=np.float64) for name, values in policy_terms.items()},
        **{f"critic_{name}": np.asarray(values, dtype=np.float64) for name, values in critic_terms.items()},
        **{f"raw_{name}": np.asarray(values, dtype=np.float64) for name, values in raw_state.items()},
    )
    print("BM_SENTINEL:obs_sample:sample_written=" + str(OUT_JSON), flush=True)
    print("BM_SENTINEL:obs_sample:npz_written=" + str(OUT_NPZ), flush=True)
    env.close()
    simulation_app.close(wait_for_replicator=False)
    print("BM_SENTINEL:obs_sample:success", flush=True)
    os._exit(0)
except BaseException as exc:
    write_stage("exception", exception=repr(exc), traceback=traceback.format_exc())
    raise
finally:
    try:
        simulation_app.close(wait_for_replicator=False)
    except BaseException:
        pass
'''


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def read_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace")
    except OSError:
        return ""


def query_gpus() -> list[dict[str, Any]]:
    rc, out = run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=30,
    )
    if rc != 0:
        return [{"error": out.strip()}]
    rows = []
    for raw in csv.reader(out.strip().splitlines()):
        if len(raw) < 6:
            continue
        index, uuid, name, mem_used, mem_total, util = [item.strip() for item in raw[:6]]
        used = int(float(mem_used))
        total = int(float(mem_total))
        rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_used_mb": used,
                "memory_total_mb": total,
                "memory_free_mb": total - used,
                "utilization_gpu_percent": int(float(util)),
            }
        )
    return rows


def query_compute_processes(gpu_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    uuid_to_index = {row.get("uuid"): row.get("index") for row in gpu_rows if row.get("uuid")}
    rc, out = run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
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


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "VK_ICD_FILENAMES": str(PROJECT_EGL_ICD),
            "LD_LIBRARY_PATH": f"{GPU_FOUNDATION_DEPS}:{env.get('LD_LIBRARY_PATH', '')}",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "ISAAC_PATH": str(ROOT / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim"),
            "OMNI_USER_DIR": str(ROOT / "cache/omni/user"),
            "OMNI_CACHE_DIR": str(ROOT / "cache/omni/cache"),
            "OMNI_DATA_DIR": str(ROOT / "cache/omni/data"),
            "OMNI_LOGS_DIR": str(ROOT / "logs/omniverse"),
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "TMPDIR": str(ROOT / "tmp"),
            "BM_SAMPLE_JSON": str(SAMPLE_JSON),
            "BM_SAMPLE_NPZ": str(SAMPLE_NPZ),
            "BM_ROBOT_USD": str(ROBOT_USD),
            "BM_MOTION_FILE": str(MOTION_NPZ),
            "BM_TARGET_GPU": str(TARGET_GPU),
            "BM_DEVICE": f"cuda:{TARGET_GPU}",
            "BM_SEED": "20260770",
            "BM_CAPTURE_AFTER_ZERO_STEP": "1" if CAPTURE_AFTER_ZERO_STEP else "0",
        }
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def classify_log(text: str, timed_out: bool) -> dict[str, bool]:
    lowered = text.lower()
    return {
        "timed_out": timed_out,
        "before_import": "BM_SENTINEL:obs_sample:state:" in text and '"stage": "before_import"' in text,
        "after_app": '"stage": "after_app"' in text,
        "env_created": '"stage": "env_created"' in text,
        "env_reset": '"stage": "env_reset"' in text,
        "sample_written": "BM_SENTINEL:obs_sample:sample_written=" in text,
        "npz_written": "BM_SENTINEL:obs_sample:npz_written=" in text,
        "success": "BM_SENTINEL:obs_sample:success" in text,
        "traceback": "traceback (most recent call last)" in lowered,
        "vulkan_device_lost": "device_lost" in lowered or "vk_error_device_lost" in lowered,
        "inotify_errno28": "errno=28" in lowered or "failed to create change watch" in lowered,
        "no_device_created": "no device could be created" in lowered,
        "active_gpu_incompatible": "activegpu" in lowered and "not compatible" in lowered,
    }


def run_worker(log_path: Path) -> tuple[int, str, bool, float]:
    WORKER.parent.mkdir(parents=True, exist_ok=True)
    WORKER.write_text(WORKER_CODE, encoding="utf-8")
    cmd = [str(TRACKING_PY), str(WORKER), "--headless", "--device", f"cuda:{TARGET_GPU}"]
    env = base_env()
    start = time.time()
    last_change = time.time()
    last_size = -1
    chunks: list[str] = []
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert proc.stdout is not None
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        timed_out = False
        while True:
            ready, _, _ = select.select([proc.stdout], [], [], 1.0)
            if ready:
                line = proc.stdout.readline()
                if line:
                    chunks.append(line)
                    log.write(line)
                    log.flush()
            code = proc.poll()
            if code is not None:
                rest = proc.stdout.read()
                if rest:
                    chunks.append(rest)
                    log.write(rest)
                break
            if log_path.exists():
                size = log_path.stat().st_size
                if size != last_size:
                    last_change = time.time()
                    last_size = size
            if time.time() - start > TIMEOUT_SECONDS or time.time() - last_change > STALL_SECONDS:
                timed_out = True
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=20)
                break
        duration = round(time.time() - start, 3)
        return proc.returncode if proc.returncode is not None else 124, "".join(chunks), timed_out, duration


def build_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    checks = summary.get("checks", {})
    return [
        {"check": key, "passed": bool(value), "notes": "n/a"}
        for key, value in checks.items()
    ]


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "passed", "notes"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_md(summary: dict[str, Any]) -> None:
    sample = summary.get("sample_summary", {})
    failed = [key for key, value in summary.get("checks", {}).items() if not value]
    lines = [
        "# IsaacLab Observation Manager Sample Gate",
        "",
        f"- Status: `{summary['status']}`",
        f"- Generated: `{summary['generated_at']}`",
        "- Scope: official IsaacLab observation sample only; no MuJoCo parity, no training, no video.",
        "- 当前不得声称完整复现 BeyondMimic；本 gate 只是为下一步 MuJoCo obs adapter parity 提供官方样本。",
        "",
        "## Sample",
        "",
        f"- JSON: `{summary['outputs'].get('sample_json')}`",
        f"- NPZ: `{summary['outputs'].get('sample_npz')}`",
        f"- Policy obs dim: `{sample.get('policy_obs_dim')}`",
        f"- Policy term names: `{sample.get('policy_term_names')}`",
        f"- Motion time steps: `{sample.get('motion_time_steps')}`",
        "",
        "## Failed / Blocking Checks",
        "",
    ]
    if failed:
        lines.extend(f"- `{item}`" for item in failed)
    else:
        lines.append("- None for sample capture. MuJoCo parity is still a separate blocking gate.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- 如果本 gate 成功，说明官方 IsaacLab observation_manager 样本已经可捕获。",
            "- 这仍不代表 MuJoCo 160-D observation adapter 正确。",
            "- 下一步必须在同一 reset/state/last_action 条件下对 MuJoCo builder 的 8 个 policy slices 做数值对比。",
            "",
        ]
    )
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"isaaclab_observation_manager_sample_gate_{timestamp}.log"
    gpu_rows = query_gpus()
    processes = query_compute_processes(gpu_rows)
    target_gpu_row = next((row for row in gpu_rows if row.get("index") == TARGET_GPU), {})
    target_processes = [proc for proc in processes if proc.get("gpu_index") == TARGET_GPU]
    target_processes_blocking = [
        proc
        for proc in target_processes
        if WANGJC_PATH_MARKER not in proc.get("cmdline", "") and WANGJC_PATH_MARKER not in proc.get("process_name", "")
    ]
    input_checks = {
        "tracking_python_exists": TRACKING_PY.is_file(),
        "robot_usd_exists": ROBOT_USD.is_file(),
        "motion_npz_exists": MOTION_NPZ.is_file(),
        "project_egl_icd_exists": PROJECT_EGL_ICD.is_file(),
        "gpu_foundation_deps_exists": GPU_FOUNDATION_DEPS.is_dir(),
        "target_gpu_resource_ready": bool(
            target_gpu_row
            and target_gpu_row.get("memory_free_mb", 0) >= MIN_FREE_MB
            and target_gpu_row.get("utilization_gpu_percent", 100) <= MAX_BUSY_UTIL
            and not target_processes_blocking
        ),
        "does_not_start_training": True,
    }
    attempted = False
    returncode: int | None = None
    log_text = ""
    timed_out = False
    duration = 0.0
    if all(input_checks.values()):
        attempted = True
        returncode, log_text, timed_out, duration = run_worker(log_path)
    markers = classify_log(log_text, timed_out)
    sample = json.loads(SAMPLE_JSON.read_text(encoding="utf-8")) if SAMPLE_JSON.is_file() else {}
    sample_ok = sample.get("status") == "ok_isaaclab_observation_manager_sample_captured"
    checks = {
        **input_checks,
        "worker_attempted": attempted,
        "worker_returncode_zero": returncode == 0,
        "sentinel_after_app": markers["after_app"],
        "sentinel_env_created": markers["env_created"],
        "sentinel_env_reset": markers["env_reset"],
        "sample_json_written": SAMPLE_JSON.is_file(),
        "sample_npz_written": SAMPLE_NPZ.is_file(),
        "sample_status_ok": sample_ok,
        "sample_policy_obs_dim_160": sample.get("policy_obs_dim") == 160,
        "sample_policy_terms_expected_order": bool(
            sample.get("checks", {}).get("policy_terms_expected_order")
        ),
        "sample_does_not_claim_mujoco_parity": bool(
            sample.get("checks", {}).get("does_not_claim_mujoco_parity_or_rollout")
        ),
        "mujoco_native_parity_ready": False,
    }
    status = (
        "ok_isaaclab_observation_manager_sample_captured_but_mujoco_parity_pending"
        if sample_ok and checks["sample_policy_obs_dim_160"] and checks["sample_policy_terms_expected_order"]
        else "failed_isaaclab_observation_manager_sample_gate"
    )
    sample_summary = {
        "status": sample.get("status"),
        "policy_obs_dim": sample.get("policy_obs_dim"),
        "policy_obs_shape": sample.get("policy_obs_shape"),
        "critic_obs_shape": sample.get("critic_obs_shape"),
        "policy_term_names": sample.get("policy_term_names"),
        "policy_term_dims": sample.get("policy_term_dims"),
        "motion_time_steps": sample.get("motion_time_steps"),
        "reward_after_zero_step": sample.get("reward_after_zero_step"),
        "terminated_after_zero_step": sample.get("terminated_after_zero_step"),
        "truncated_after_zero_step": sample.get("truncated_after_zero_step"),
        "sample_json_sha256": sha256(SAMPLE_JSON) if SAMPLE_JSON.is_file() else "",
        "sample_npz_sha256": sha256(SAMPLE_NPZ) if SAMPLE_NPZ.is_file() else "",
    }
    summary = {
        "status": status,
        "generated_at": utc_now(),
        "experiment_type": "isaaclab_observation_manager_sample_gate",
        "claim_level": "official_isaaclab_observation_sample_only; no_mujoco_parity_claim",
        "inputs": {
            "tracking_python": str(TRACKING_PY),
            "robot_usd": str(ROBOT_USD),
            "motion_npz": str(MOTION_NPZ),
            "target_gpu": TARGET_GPU,
        },
        "runtime": {
            "attempted": attempted,
            "returncode": returncode,
            "duration_seconds": duration,
            "timed_out": timed_out,
            "log": str(log_path),
            "markers": markers,
            "target_gpu_row": target_gpu_row,
            "target_processes": target_processes,
        },
        "sample_summary": sample_summary,
        "checks": checks,
        "outputs": {
            "json": str(JSON_OUT),
            "tsv": str(TSV_OUT),
            "md": str(MD_OUT),
            "worker": str(WORKER),
            "log": str(log_path),
            "sample_json": str(SAMPLE_JSON),
            "sample_npz": str(SAMPLE_NPZ),
        },
        "hard_blockers": [
            "mujoco_native_parity_ready is false",
            "sample capture does not compare MuJoCo builder output against official observation_manager",
        ],
        "interpretation": {
            "official_observation_manager_sample_available": sample_ok,
            "mujoco_native_observation_parity_ready": False,
            "success_video_claim_allowed": False,
            "goal_complete": False,
            "next_step": (
                "Use this sample as the official reference for a same-state MuJoCo observation builder comparison "
                "covering all 8 policy slices."
            ),
        },
    }
    rows = build_rows(summary)
    write_json(JSON_OUT, summary)
    write_tsv(TSV_OUT, rows)
    write_md(summary)
    print(json.dumps({"status": status, "json": str(JSON_OUT), "sample_json": str(SAMPLE_JSON)}, sort_keys=True))
    if status.startswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
