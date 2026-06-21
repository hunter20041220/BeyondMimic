#!/usr/bin/env python3
"""Diagnose the current Tracking-Flat-G1-v0 env-construction gate."""

from __future__ import annotations

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
OUT = ROOT / "res/tracking/g1_current_task_env_construction_gate"
LOG_DIR = ROOT / "logs/tracking_g1_current_task_env_construction_gate"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
OFFICIAL_IMPORTER_USD = ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda"
LEGACY_OFFICIAL_MOTION_NPZ = (
    ROOT
    / "res/tracking/official_csv_to_npz_loop_full_dataset_with_official_importer_export/motions/"
    "dance1_subject1/motion.npz"
)
FK_MOTION_NPZ = (
    ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_split_motion_npz/motions/"
    "dance1_subject1/motion.npz"
)
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
TARGET_GPU = int(os.environ.get("BM_TASK_ENV_GATE_GPU", "4"))
STALL_SECONDS = int(os.environ.get("BM_TASK_ENV_GATE_STALL_SECONDS", "240"))


WORKER_CODE = r"""
import faulthandler
import json
import os
import signal
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

faulthandler.enable(file=sys.stdout, all_threads=True)

OUT = Path(os.environ["BM_GATE_STATE_JSON"])
ROBOT_USD = Path(os.environ["BM_ROBOT_USD"])
MOTION_FILE = Path(os.environ["BM_MOTION_FILE"])
PROBE_MODE = os.environ["BM_PROBE_MODE"]


def write_state(stage, **extra):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "probe_mode": PROBE_MODE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "robot_usd": str(ROBOT_USD),
        "motion_file": str(MOTION_FILE),
    }
    payload.update(extra)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print("BM_SENTINEL:state:" + json.dumps(payload, sort_keys=True), flush=True)


def handle_signal(signum, frame):
    write_state("signal", signal=signum)
    sys.exit(128 + signum)


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)

write_state("before_import")
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

write_state("before_app", device=args.device, target_gpu=target_gpu)
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
write_state("after_app", app_running=bool(simulation_app.is_running()))

try:
    if PROBE_MODE == "app_only":
        write_state("app_only_success")
        os._exit(0)

    import gymnasium as gym
    import isaaclab.sim as sim_utils
    import whole_body_tracking.tasks  # noqa: F401
    from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg

    write_state("before_cfg")
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
    env_cfg.seed = int(os.environ.get("BM_SEED", "20260760"))
    write_state("cfg_ready")

    if PROBE_MODE == "cfg_only":
        write_state("cfg_only_success")
        os._exit(0)

    write_state("before_gym_make")
    env = gym.make("Tracking-Flat-G1-v0", cfg=env_cfg, render_mode=None)
    env_created_payload = {
        "num_envs": int(env.unwrapped.num_envs),
        "device": str(env.unwrapped.device),
        "action_dim": int(env.unwrapped.action_manager.total_action_dim),
        "robot_num_joints": int(env.unwrapped.scene["robot"].num_joints),
        "robot_num_bodies": int(env.unwrapped.scene["robot"].num_bodies),
    }
    write_state("env_created", **env_created_payload)
    if PROBE_MODE == "gym_make_only":
        write_state("gym_make_only_success", **env_created_payload)
        os._exit(0)

    obs, extras = env.reset()
    write_state(
        "env_reset",
        policy_observation_dim=int(env.unwrapped.observation_manager.group_obs_dim["policy"][0]),
        critic_observation_dim=int(env.unwrapped.observation_manager.group_obs_dim["critic"][0]),
    )
    env.close()
    write_state("env_closed")
    simulation_app.close(wait_for_replicator=False)
    write_state("after_close")
except BaseException as exc:
    write_state("exception", exception=repr(exc), traceback=traceback.format_exc())
    raise
"""


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def env_for(mode: str, state_json: Path, motion_npz: Path) -> dict[str, str]:
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
            "BM_GATE_STATE_JSON": str(state_json),
            "BM_ROBOT_USD": str(OFFICIAL_IMPORTER_USD),
            "BM_MOTION_FILE": str(motion_npz),
            "BM_PROBE_MODE": mode,
            "BM_TARGET_GPU": str(TARGET_GPU),
            "BM_DEVICE": f"cuda:{TARGET_GPU}",
            "BM_SEED": "20260760",
        }
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def run_probe(worker: Path, mode: str, motion_name: str, motion_npz: Path) -> dict[str, Any]:
    state_json = OUT / f"{mode}_{motion_name}_state.json"
    log_path = LOG_DIR / f"{mode}_{motion_name}.log"
    cmd = [str(TRACKING_PY), str(worker), "--headless", "--device", f"cuda:{TARGET_GPU}"]
    start = time.time()
    last_change = time.time()
    last_heartbeat = 0.0
    last_size = -1
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env_for(mode, state_json, motion_npz),
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
                    lines.append(line)
                    log_file.write(line)
                    log_file.flush()
                    if (
                        line.startswith("BM_SENTINEL")
                        or "Traceback" in line
                        or "Error" in line
                        or "Exception" in line
                    ):
                        print(line.rstrip(), flush=True)
            size = log_path.stat().st_size if log_path.is_file() else 0
            if size != last_size:
                last_size = size
                last_change = time.time()
            elif time.time() - last_change > STALL_SECONDS:
                log_file.write(f"\nBM_STALL_TIMEOUT:no_log_progress_for_{STALL_SECONDS}s\n")
                log_file.flush()
                proc.terminate()
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=60)
                break
            now = time.time()
            if now - last_heartbeat >= 10:
                last_heartbeat = now
                print(
                    json.dumps(
                        {
                            "parent_heartbeat": "current_task_env_construction_gate",
                            "mode": mode,
                            "motion": motion_name,
                            "elapsed_seconds": round(now - start, 1),
                            "log_size_bytes": size,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        if proc.stdout:
            rest = proc.stdout.read()
            if rest:
                lines.append(rest)
                log_file.write(rest)
                print(rest, end="", flush=True)
    state = {}
    if state_json.is_file():
        state = json.loads(state_json.read_text(encoding="utf-8"))
    text = "".join(lines)
    return {
        "mode": mode,
        "motion": motion_name,
        "motion_npz": str(motion_npz),
        "status": "ok" if proc.returncode == 0 else "failed",
        "returncode": proc.returncode,
        "duration_seconds": round(time.time() - start, 3),
        "last_state": state,
        "state_json": str(state_json),
        "log": str(log_path),
        "markers": {
            "after_app": "BM_SENTINEL:state:" in text and '"stage": "after_app"' in text,
            "app_only_success": '"stage": "app_only_success"' in text,
            "cfg_ready": '"stage": "cfg_ready"' in text,
            "cfg_only_success": '"stage": "cfg_only_success"' in text,
            "before_gym_make": '"stage": "before_gym_make"' in text,
            "env_created": '"stage": "env_created"' in text,
            "gym_make_only_success": '"stage": "gym_make_only_success"' in text,
            "env_reset": '"stage": "env_reset"' in text,
            "exception": '"stage": "exception"' in text,
            "signal": '"stage": "signal"' in text,
        },
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    worker = OUT / "tracking_g1_current_task_env_construction_gate_worker.py"
    worker.write_text(textwrap.dedent(WORKER_CODE), encoding="utf-8")
    probe_specs = [
        ("gym_make_only", "fk_repaired", FK_MOTION_NPZ),
        ("app_only", "fk_repaired", FK_MOTION_NPZ),
        ("cfg_only", "fk_repaired", FK_MOTION_NPZ),
    ]
    rows = [run_probe(worker, mode, motion, motion_npz) for mode, motion, motion_npz in probe_specs]
    checks = {
        "fk_repaired_gym_make_ok": rows[0]["status"] == "ok"
        and rows[0]["last_state"].get("stage") == "gym_make_only_success",
        "app_only_ok": rows[1]["status"] == "ok" and rows[1]["last_state"].get("stage") == "app_only_success",
        "cfg_only_ok": rows[2]["status"] == "ok" and rows[2]["last_state"].get("stage") == "cfg_only_success",
        "legacy_official_motion_path_missing_recorded": not LEGACY_OFFICIAL_MOTION_NPZ.exists(),
        "does_not_start_training": True,
        "does_not_claim_paper_level_tracking": True,
        "does_not_claim_real_robot": True,
    }
    status = "ok_current_task_env_construction_gate" if all(checks.values()) else "failed_current_task_env_construction_gate"
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_g1_current_task_env_construction_gate",
        "scope": "Current staged AppLauncher/cfg/gym.make diagnostics for Tracking-Flat-G1-v0 on official-importer G1 USDA.",
        "config": {
            "target_gpu": TARGET_GPU,
            "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
            "legacy_official_motion_npz": str(LEGACY_OFFICIAL_MOTION_NPZ),
            "legacy_official_motion_npz_exists": LEGACY_OFFICIAL_MOTION_NPZ.exists(),
            "fk_motion_npz": str(FK_MOTION_NPZ),
        },
        "checks": checks,
        "rows": rows,
        "outputs": {"json": str(OUT / "tracking_g1_current_task_env_construction_gate.json"), "worker": str(worker), "log_dir": str(LOG_DIR)},
        "interpretation": {
            "goal_complete": False,
            "claim_level": "current_live_task_env_construction_gate",
            "not_paper_level_reasons": [
                "diagnostic only, no replay loop, PPO, DAgger, VAE/diffusion, TensorRT, Fig. 5/Fig. 6, or robot evidence",
            ],
        },
    }
    write_json(Path(summary["outputs"]["json"]), summary)
    print(json.dumps({"status": status, "json": summary["outputs"]["json"]}, sort_keys=True))
    if status != "ok_current_task_env_construction_gate":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
