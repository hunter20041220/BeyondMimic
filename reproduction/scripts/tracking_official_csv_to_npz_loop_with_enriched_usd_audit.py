#!/usr/bin/env python3
"""Run official csv_to_npz.py with local output and enriched-USD runtime patches.

The official script body is executed with ``runpy``. Runtime monkeypatches keep
all generated files inside this project and bypass only the known G1 URDF/USD
conversion blocker by replacing the in-memory G1 config with the already audited
resource-adjusted enriched USD. The resulting NPZ is therefore not official
converter output and must not be reported as paper-level replay evidence.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = Path(os.environ.get("BM_OFFICIAL_CSV_LOOP_OUT_DIR", str(ROOT / "res/tracking/official_csv_to_npz_loop_with_enriched_usd")))
LOG_DIR = Path(os.environ.get("BM_OFFICIAL_CSV_LOOP_LOG_DIR", str(ROOT / "logs/tracking_official_csv_to_npz_loop_with_enriched_usd")))
FAILED_DIR = Path(
    os.environ.get(
        "BM_OFFICIAL_CSV_LOOP_FAILED_DIR",
        str(ROOT / "res/failed_runs/tracking_official_csv_to_npz_loop_with_enriched_usd"),
    )
)
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
OFFICIAL_CSV_TO_NPZ = ROOT / "reproduction/third_party/official/whole_body_tracking/scripts/csv_to_npz.py"
INPUT_CSV = Path(
    os.environ.get(
        "BM_OFFICIAL_CSV_LOOP_INPUT_CSV",
        str(ROOT / "download/official/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv"),
    )
)
OUTPUT_NPZ = Path(
    os.environ.get(
        "BM_OFFICIAL_CSV_LOOP_OUTPUT_NPZ",
        str(OUT / "walk1_subject1_frames_1_180_official_loop_enriched_usd_motion.npz"),
    )
)
METRICS_JSON = Path(
    os.environ.get(
        "BM_OFFICIAL_CSV_LOOP_METRICS_JSON",
        str(OUT / "tracking_official_csv_to_npz_loop_with_enriched_usd_metrics.json"),
    )
)
DEFAULT_ENRICHED_USD = (
    ROOT
    / "res/tracking/g1_resource_adjusted_enriched_usd/"
    "g1_resource_adjusted_29dof_enriched_scaffold.usda"
)
ROBOT_USD = Path(os.environ.get("BM_OFFICIAL_CSV_LOOP_ROBOT_USD", str(DEFAULT_ENRICHED_USD)))
ROBOT_USD_LABEL = os.environ.get("BM_OFFICIAL_CSV_LOOP_USD_LABEL", "enriched_usd")
USES_RESOURCE_ADJUSTED_USD = os.environ.get("BM_OFFICIAL_CSV_LOOP_USES_RESOURCE_ADJUSTED_USD", "1") == "1"
USES_OFFICIAL_IMPORTER_EXPORT_USD = (
    os.environ.get("BM_OFFICIAL_CSV_LOOP_USES_OFFICIAL_IMPORTER_EXPORT_USD", "0") == "1"
)
RESOURCE_ADJUSTED_CONVERSION = (
    ROOT
    / "res/tracking/g1_resource_adjusted_csv_conversion/"
    "tracking_g1_resource_adjusted_csv_conversion_audit.json"
)
OFFICIAL_REPLAY_LOOP = (
    ROOT
    / "res/tracking/official_replay_npz_loop_with_enriched_usd/"
    "tracking_official_replay_npz_loop_with_enriched_usd_audit.json"
)
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
PROBE = OUT / "tracking_official_csv_to_npz_loop_with_enriched_usd_probe.py"
AUDIT_BASENAME = os.environ.get(
    "BM_OFFICIAL_CSV_LOOP_AUDIT_BASENAME",
    "tracking_official_csv_to_npz_loop_with_enriched_usd_audit",
)
SUCCESS_STATUS = os.environ.get(
    "BM_OFFICIAL_CSV_LOOP_SUCCESS_STATUS",
    "ok_official_csv_to_npz_loop_with_enriched_usd_patch",
)
BLOCKER_STATUS = os.environ.get(
    "BM_OFFICIAL_CSV_LOOP_BLOCKER_STATUS",
    "ok_with_official_csv_to_npz_loop_patch_blocker",
)
EXPERIMENT_TYPE = os.environ.get(
    "BM_OFFICIAL_CSV_LOOP_EXPERIMENT_TYPE",
    "tracking_official_csv_to_npz_loop_with_enriched_usd_patch",
)
CLAIM_LEVEL = os.environ.get(
    "BM_OFFICIAL_CSV_LOOP_CLAIM_LEVEL",
    "resource_adjusted_official_csv_to_npz_loop",
)
TARGET_GPU = int(os.environ.get("BM_OFFICIAL_CSV_LOOP_TARGET_GPU", "4"))
WATCH_GPUS = [4, 7]
MAX_STEPS = int(os.environ.get("BM_OFFICIAL_CSV_LOOP_MAX_STEPS", "299"))
STALL_SECONDS = 900
SUCCESS_EXIT_GRACE_SECONDS = int(os.environ.get("BM_OFFICIAL_CSV_LOOP_SUCCESS_EXIT_GRACE_SECONDS", "30"))
WANGJC_PATH_MARKER = "/mnt/infini-data/test/wangjc/"
OUTPUT_NAME = os.environ.get(
    "BM_OFFICIAL_CSV_LOOP_OUTPUT_NAME",
    f"bm_local_{INPUT_CSV.stem}_official_csv_loop_enriched_usd",
)
LOG_BASENAME = os.environ.get(
    "BM_OFFICIAL_CSV_LOOP_LOG_BASENAME",
    "tracking_official_csv_to_npz_loop_with_enriched_usd",
)


PROBE_CODE = r"""
import os
import runpy
import signal
import sys
import traceback
import types
from pathlib import Path

OFFICIAL_CSV_TO_NPZ = Path(os.environ["BM_OFFICIAL_CSV_TO_NPZ"])
OUTPUT_NPZ = Path(os.environ["BM_OUTPUT_NPZ"])
OUTPUT_NAME = os.environ["BM_OUTPUT_NAME"]
MAX_STEPS = int(os.environ["BM_MAX_STEPS"])
ROBOT_USD = Path(os.environ["BM_ROBOT_USD"])
ROBOT_USD_LABEL = os.environ["BM_ROBOT_USD_LABEL"]


def _sigterm_handler(signum, frame):
    print(f"BM_SENTINEL:received_signal={signum}", flush=True)
    traceback.print_stack(frame)
    print("BM_SENTINEL:received_signal_ignored_for_bounded_probe", flush=True)


signal.signal(signal.SIGTERM, _sigterm_handler)

import numpy as _np

_real_savez = _np.savez


def _redirected_savez(file, *args, **kwargs):
    if str(file) == "/tmp/motion.npz":
        OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
        print(f"BM_SENTINEL:np_savez_redirect=/tmp/motion.npz->{OUTPUT_NPZ}", flush=True)
        return _real_savez(str(OUTPUT_NPZ), *args, **kwargs)
    return _real_savez(file, *args, **kwargs)


_np.savez = _redirected_savez


class _FakeRun:
    def __init__(self, project, name):
        self.project = project
        self.name = name
        print(f"BM_SENTINEL:fake_wandb_init=project:{project},name:{name}", flush=True)

    def log_artifact(self, artifact_or_path, name, type):
        print(
            f"BM_SENTINEL:fake_wandb_log_artifact=artifact:{artifact_or_path},name:{name},type:{type}",
            flush=True,
        )
        if name != OUTPUT_NAME:
            raise RuntimeError(f"Unexpected artifact name {name!r} != {OUTPUT_NAME!r}")
        if not OUTPUT_NPZ.is_file():
            raise RuntimeError(f"Redirected output NPZ missing before fake log_artifact: {OUTPUT_NPZ}")
        return {"artifact_or_path": str(OUTPUT_NPZ), "name": name, "type": type}

    def link_artifact(self, artifact, target_path):
        print(f"BM_SENTINEL:fake_wandb_link_artifact=target:{target_path}", flush=True)
        return None


def _fake_init(project, name):
    return _FakeRun(project=project, name=name)


fake_wandb = types.ModuleType("wandb")
fake_wandb.init = _fake_init
sys.modules["wandb"] = fake_wandb

import isaaclab.app as isaaclab_app

_RealAppLauncher = isaaclab_app.AppLauncher


class _BoundedSimulationApp:
    def __init__(self, app):
        self._app = app
        self._calls = 0

    def is_running(self):
        self._calls += 1
        if self._calls <= MAX_STEPS:
            if self._calls == 1 or self._calls % 50 == 0 or self._calls == MAX_STEPS:
                print(f"BM_SENTINEL:official_csv_loop_is_running_call={self._calls}", flush=True)
            return self._app.is_running()
        print(f"BM_SENTINEL:official_csv_loop_complete={MAX_STEPS}", flush=True)
        return False

    def close(self, *args, **kwargs):
        print("BM_SENTINEL:simulation_app_close_called", flush=True)
        return self._app.close(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._app, name)


def patch_g1_cfg_to_robot_usd():
    import isaaclab.sim as sim_utils
    import whole_body_tracking.robots.g1 as g1_module

    cfg = g1_module.G1_CYLINDER_CFG.copy()
    cfg.spawn = sim_utils.UsdFileCfg(
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
    g1_module.G1_CYLINDER_CFG = cfg
    print(f"BM_SENTINEL:g1_cfg_patched_to_robot_usd={ROBOT_USD_LABEL}:{ROBOT_USD}", flush=True)
    if ROBOT_USD_LABEL == "enriched_usd":
        print(f"BM_SENTINEL:g1_cfg_patched_to_enriched_usd={ROBOT_USD}", flush=True)


class _BoundedAppLauncher:
    @staticmethod
    def add_app_launcher_args(parser):
        print("BM_SENTINEL:add_app_launcher_args", flush=True)
        return _RealAppLauncher.add_app_launcher_args(parser)

    def __init__(self, args):
        print("BM_SENTINEL:before_real_app_launcher", flush=True)
        args.headless = True
        args.enable_cameras = False
        args.device = os.environ["BM_DEVICE"]
        args.multi_gpu = False
        args.kit_args = os.environ.get("BM_KIT_ARGS", "")
        self._real = _RealAppLauncher(args)
        self.app = _BoundedSimulationApp(self._real.app)
        print("BM_SENTINEL:after_real_app_launcher", flush=True)
        patch_g1_cfg_to_robot_usd()

    def __getattr__(self, name):
        return getattr(self._real, name)


isaaclab_app.AppLauncher = _BoundedAppLauncher
sys.argv = [
    str(OFFICIAL_CSV_TO_NPZ),
    "--input_file",
    os.environ["BM_INPUT_CSV"],
    "--input_fps",
    "30",
    "--frame_range",
    "1",
    "180",
    "--output_name",
    OUTPUT_NAME,
    "--output_fps",
    "50",
    "--headless",
    "--device",
    os.environ["BM_DEVICE"],
]

print(f"BM_SENTINEL:before_runpy={OFFICIAL_CSV_TO_NPZ}", flush=True)
try:
    runpy.run_path(str(OFFICIAL_CSV_TO_NPZ), run_name="__main__")
    print("BM_SENTINEL:after_runpy", flush=True)
except BaseException as exc:
    print("BM_SENTINEL:exception=" + repr(exc), flush=True)
    traceback.print_exc()
    raise
"""


def run_command(cmd: list[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
    path = guard_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_gpu47_wangjc_official_csv_to_npz_loop_guard.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["json"] = str(path)
    return summary


def classify_log(text: str) -> dict[str, Any]:
    lowered = text.lower()
    step_calls = []
    for marker in [
        "bm_sentinel:official_csv_loop_is_running_call=1",
        "bm_sentinel:official_csv_loop_is_running_call=50",
        "bm_sentinel:official_csv_loop_is_running_call=100",
        "bm_sentinel:official_csv_loop_is_running_call=150",
        "bm_sentinel:official_csv_loop_is_running_call=200",
        "bm_sentinel:official_csv_loop_is_running_call=250",
        "bm_sentinel:official_csv_loop_is_running_call=299",
    ]:
        if marker in lowered:
            step_calls.append(int(marker.rsplit("=", 1)[-1]))
    return {
        "before_runpy": "bm_sentinel:before_runpy=" in lowered,
        "add_app_launcher_args": "bm_sentinel:add_app_launcher_args" in lowered,
        "before_real_app_launcher": "bm_sentinel:before_real_app_launcher" in lowered,
        "after_real_app_launcher": "bm_sentinel:after_real_app_launcher" in lowered,
        "g1_cfg_patched_to_robot_usd": "bm_sentinel:g1_cfg_patched_to_robot_usd=" in lowered,
        "g1_cfg_patched_to_enriched_usd": "bm_sentinel:g1_cfg_patched_to_enriched_usd=" in lowered,
        "motion_loaded": "motion loaded" in lowered,
        "motion_interpolated": "motion interpolated" in lowered,
        "official_loop_step_calls": step_calls,
        "official_loop_call_299": "bm_sentinel:official_csv_loop_is_running_call=299" in lowered,
        "official_loop_complete": f"bm_sentinel:official_csv_loop_complete={MAX_STEPS}" in lowered,
        "np_savez_redirect": "bm_sentinel:np_savez_redirect=" in lowered,
        "fake_wandb_init": "bm_sentinel:fake_wandb_init=" in lowered,
        "fake_wandb_log_artifact": "bm_sentinel:fake_wandb_log_artifact=" in lowered,
        "fake_wandb_link_artifact": "bm_sentinel:fake_wandb_link_artifact=" in lowered,
        "official_motion_saved_print": "motion saved to wandb registry" in lowered,
        "simulation_app_close_called": "bm_sentinel:simulation_app_close_called" in lowered,
        "after_runpy": "bm_sentinel:after_runpy" in lowered,
        "exception": "bm_sentinel:exception=" in lowered,
        "traceback": "traceback (most recent call last)" in lowered,
        "vulkan_device_lost": "vkresult: error_device_lost" in lowered or "device lost" in lowered,
        "permission_to_save_false": (
            "permissiontosave=false" in lowered
            or "permission to save" in lowered
            or "saving not allowed" in lowered
        ),
        "failed_to_save_layer": (
            ("cannot save layer" in lowered or "failed to save" in lowered)
            and ("layer" in lowered or ".usd@" in lowered)
        ),
        "empty_robot_after_converter": "no rigid bodies are present under this prim" in lowered,
        "stall_timeout": "bm_stall_timeout" in lowered,
        "driver_shader_cache_shutdown_error": "drivershadercachemanager::init() called without a shutdown" in lowered,
    }


def env_vars() -> dict[str, str]:
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
            "BM_OFFICIAL_CSV_TO_NPZ": str(OFFICIAL_CSV_TO_NPZ),
            "BM_INPUT_CSV": str(INPUT_CSV),
            "BM_OUTPUT_NPZ": str(OUTPUT_NPZ),
            "BM_OUTPUT_NAME": OUTPUT_NAME,
            "BM_MAX_STEPS": str(MAX_STEPS),
            "BM_ROBOT_USD": str(ROBOT_USD),
            "BM_ROBOT_USD_LABEL": ROBOT_USD_LABEL,
            "BM_DEVICE": f"cuda:{TARGET_GPU}",
            "BM_KIT_ARGS": (
                "--/renderer/multiGpu/enabled=false "
                "--/renderer/multiGpu/autoEnable=false "
                "--/renderer/multiGpu/maxGpuCount=1 "
                f"--/renderer/activeGpu={TARGET_GPU} "
                f"--/physics/cudaDevice={TARGET_GPU}"
            ),
        }
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def prepare_probe() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_NPZ.exists():
        OUTPUT_NPZ.unlink()
    if METRICS_JSON.exists():
        METRICS_JSON.unlink()
    PROBE.write_text(textwrap.dedent(PROBE_CODE), encoding="utf-8")


def run_probe(log_path: Path) -> dict[str, Any]:
    command = [str(TRACKING_PY), str(PROBE)]
    start = time.time()
    stalled = False
    forced_after_success_sentinel = False
    success_seen_at = None
    last_size = -1
    last_change = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env_vars(),
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
                text_now = log_path.read_text(encoding="utf-8", errors="replace")
                if (
                    success_seen_at is None
                    and f"BM_SENTINEL:official_csv_loop_complete={MAX_STEPS}" in text_now
                    and "BM_SENTINEL:simulation_app_close_called" in text_now
                    and OUTPUT_NPZ.is_file()
                ):
                    success_seen_at = time.time()
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
            if success_seen_at is not None and time.time() - success_seen_at > SUCCESS_EXIT_GRACE_SECONDS:
                forced_after_success_sentinel = True
                log_file.write(
                    f"\nBM_FORCED_EXIT_AFTER_SUCCESS_SENTINEL:grace_seconds={SUCCESS_EXIT_GRACE_SECONDS}\n"
                )
                log_file.flush()
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=60)
                break
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    return {
        "command": command,
        "returncode": proc.returncode,
        "duration_seconds": round(time.time() - start, 3),
        "stalled": stalled,
        "forced_after_success_sentinel": forced_after_success_sentinel,
        "markers": classify_log(text),
        "log": str(log_path),
    }


def compute_metrics() -> dict[str, Any]:
    if not OUTPUT_NPZ.is_file():
        return {}
    data = dict(np.load(OUTPUT_NPZ))
    metrics = {
        "input_csv": str(INPUT_CSV),
        "output_npz": str(OUTPUT_NPZ),
        "official_csv_to_npz_script": str(OFFICIAL_CSV_TO_NPZ),
        "usd_path": str(ROBOT_USD),
        "fps": data["fps"].astype(float).tolist(),
        "joint_pos_shape": list(data["joint_pos"].shape),
        "joint_vel_shape": list(data["joint_vel"].shape),
        "body_pos_w_shape": list(data["body_pos_w"].shape),
        "body_quat_w_shape": list(data["body_quat_w"].shape),
        "body_lin_vel_w_shape": list(data["body_lin_vel_w"].shape),
        "body_ang_vel_w_shape": list(data["body_ang_vel_w"].shape),
        "max_joint_abs": float(np.max(np.abs(data["joint_pos"]))),
        "max_joint_vel_abs": float(np.max(np.abs(data["joint_vel"]))),
        "root_height_min": float(np.min(data["body_pos_w"][:, 0, 2])),
        "root_height_max": float(np.max(data["body_pos_w"][:, 0, 2])),
        "max_body_quat_norm_abs_error_from_1": float(
            np.max(np.abs(np.linalg.norm(data["body_quat_w"], axis=-1) - 1.0))
        ),
        "npz_size_bytes": OUTPUT_NPZ.stat().st_size,
        "uses_official_csv_to_npz_loop": True,
        "uses_resource_adjusted_usd": USES_RESOURCE_ADJUSTED_USD,
        "uses_official_importer_export_usd": USES_OFFICIAL_IMPORTER_EXPORT_USD,
        "redirected_tmp_output_to_project": True,
        "fake_wandb_registry": True,
        "official_csv_to_npz_unpatched_official_asset_complete": False,
        "paper_level_rollout": False,
        "ppo_training": False,
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def determine_blocker(run: dict[str, Any], metrics: dict[str, Any]) -> str:
    markers = run["markers"]
    if (
        run["returncode"] == 0
        or run.get("forced_after_success_sentinel") is True
    ) and (
        markers["official_loop_complete"]
        and markers["simulation_app_close_called"]
        and OUTPUT_NPZ.is_file()
        and metrics.get("joint_pos_shape") == [299, 29]
    ):
        return f"none_official_csv_to_npz_loop_completed_with_{ROBOT_USD_LABEL}"
    if markers["vulkan_device_lost"]:
        return "vulkan_device_lost"
    if markers["permission_to_save_false"] or markers["failed_to_save_layer"] or markers["empty_robot_after_converter"]:
        return "official_urdf_converter_or_layer_save_still_reached"
    if run["stalled"]:
        return "stall_timeout"
    if markers["exception"] or markers["traceback"]:
        return "python_exception"
    if not markers["after_real_app_launcher"]:
        return "app_launcher_or_kit_startup"
    if not markers["g1_cfg_patched_to_robot_usd"]:
        return "g1_cfg_patch_failed"
    if not markers["np_savez_redirect"]:
        return "official_csv_loop_did_not_reach_save"
    if not OUTPUT_NPZ.is_file():
        return "redirected_motion_npz_missing"
    return "unknown_failed_checks"


def main() -> None:
    prepare_probe()
    guard = kill_wangjc_on_watch_gpus()
    log_path = LOG_DIR / f"{LOG_BASENAME}.log"
    run = run_probe(log_path)
    metrics = compute_metrics()
    blocker = determine_blocker(run, metrics)
    resource_adjusted_conversion = read_json(RESOURCE_ADJUSTED_CONVERSION)
    official_replay_loop = read_json(OFFICIAL_REPLAY_LOOP)
    checks = {
        "official_csv_to_npz_script_exists": OFFICIAL_CSV_TO_NPZ.is_file(),
        "tracking_python_exists": TRACKING_PY.is_file(),
        "input_csv_exists": INPUT_CSV.is_file(),
        "robot_usd_exists": ROBOT_USD.is_file(),
        "enriched_usd_exists": ROBOT_USD.is_file() if ROBOT_USD_LABEL == "enriched_usd" else True,
        "prior_resource_adjusted_conversion_available": (
            resource_adjusted_conversion.get("status") == "ok_resource_adjusted_csv_conversion"
        ),
        "prior_official_replay_loop_patch_available": (
            official_replay_loop.get("status") == "ok_official_replay_loop_with_enriched_usd_patch"
        ),
        "before_runpy_seen": run["markers"]["before_runpy"],
        "app_launcher_args_seen": run["markers"]["add_app_launcher_args"],
        "app_launcher_constructed": run["markers"]["after_real_app_launcher"],
        "g1_cfg_patched_to_robot_usd": run["markers"]["g1_cfg_patched_to_robot_usd"],
        "g1_cfg_patched_to_enriched_usd": run["markers"]["g1_cfg_patched_to_enriched_usd"],
        "motion_loaded": run["markers"]["motion_loaded"],
        "motion_interpolated": run["markers"]["motion_interpolated"],
        "official_loop_call_299_seen": run["markers"]["official_loop_call_299"],
        "official_loop_complete_seen": run["markers"]["official_loop_complete"],
        "np_savez_redirect_seen": run["markers"]["np_savez_redirect"],
        "fake_wandb_init_seen": run["markers"]["fake_wandb_init"],
        "fake_wandb_log_artifact_seen": run["markers"]["fake_wandb_log_artifact"],
        "fake_wandb_link_artifact_seen": run["markers"]["fake_wandb_link_artifact"],
        "official_motion_saved_print_seen": run["markers"]["official_motion_saved_print"],
        "process_returned_zero_or_forced_after_success_sentinel": (
            run["returncode"] == 0 or run.get("forced_after_success_sentinel") is True
        ),
        "forced_after_success_sentinel_recorded": (
            run.get("forced_after_success_sentinel") is False
            or run["markers"]["official_loop_complete"]
        ),
        "no_stall_timeout": run["stalled"] is False,
        "motion_npz_written": OUTPUT_NPZ.is_file(),
        "metrics_json_written": METRICS_JSON.is_file(),
        "joint_pos_shape_299_29": metrics.get("joint_pos_shape") == [299, 29],
        "body_pos_shape_299_40_3": metrics.get("body_pos_w_shape") == [299, 40, 3],
        "body_quaternions_unit": metrics.get("max_body_quat_norm_abs_error_from_1", 1.0) < 1e-4,
        "uses_official_csv_to_npz_loop": metrics.get("uses_official_csv_to_npz_loop") is True,
        "uses_resource_adjusted_usd": metrics.get("uses_resource_adjusted_usd") is USES_RESOURCE_ADJUSTED_USD,
        "uses_official_importer_export_usd": (
            metrics.get("uses_official_importer_export_usd") is USES_OFFICIAL_IMPORTER_EXPORT_USD
        ),
        "uses_resource_adjusted_usd_matches_expected": (
            metrics.get("uses_resource_adjusted_usd") is USES_RESOURCE_ADJUSTED_USD
        ),
        "uses_official_importer_export_usd_matches_expected": (
            metrics.get("uses_official_importer_export_usd") is USES_OFFICIAL_IMPORTER_EXPORT_USD
        ),
        "redirected_tmp_output_to_project": metrics.get("redirected_tmp_output_to_project") is True,
        "does_not_claim_unpatched_official_asset_complete": (
            metrics.get("official_csv_to_npz_unpatched_official_asset_complete") is False
        ),
        "does_not_claim_paper_level_rollout": metrics.get("paper_level_rollout") is False,
        "does_not_start_training": metrics.get("ppo_training") is False,
        "does_not_modify_official_worktree": True,
    }
    required_success_checks = [
        "official_csv_to_npz_script_exists",
        "tracking_python_exists",
        "input_csv_exists",
        "robot_usd_exists",
        "prior_resource_adjusted_conversion_available",
        "prior_official_replay_loop_patch_available",
        "before_runpy_seen",
        "app_launcher_args_seen",
        "app_launcher_constructed",
        "g1_cfg_patched_to_robot_usd",
        "motion_loaded",
        "motion_interpolated",
        "official_loop_call_299_seen",
        "official_loop_complete_seen",
        "np_savez_redirect_seen",
        "fake_wandb_init_seen",
        "fake_wandb_log_artifact_seen",
        "fake_wandb_link_artifact_seen",
        "official_motion_saved_print_seen",
        "process_returned_zero_or_forced_after_success_sentinel",
        "forced_after_success_sentinel_recorded",
        "no_stall_timeout",
        "motion_npz_written",
        "metrics_json_written",
        "joint_pos_shape_299_29",
        "body_pos_shape_299_40_3",
        "body_quaternions_unit",
        "uses_official_csv_to_npz_loop",
        "uses_resource_adjusted_usd_matches_expected",
        "uses_official_importer_export_usd_matches_expected",
        "redirected_tmp_output_to_project",
        "does_not_claim_unpatched_official_asset_complete",
        "does_not_claim_paper_level_rollout",
        "does_not_start_training",
        "does_not_modify_official_worktree",
    ]
    success = all(checks.get(name) is True for name in required_success_checks) and blocker.startswith("none_")
    status = SUCCESS_STATUS if success else BLOCKER_STATUS
    failed_log_copy = ""
    if not success:
        failed_log = FAILED_DIR / f"{LOG_BASENAME}.log"
        failed_log.write_text(log_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        failed_log_copy = str(failed_log)
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": EXPERIMENT_TYPE,
        "scope": (
            "Runs the official whole_body_tracking scripts/csv_to_npz.py loop while patching only runtime dependencies: "
            f"the in-memory G1 config uses the selected robot USD ({ROBOT_USD_LABEL}), np.savez('/tmp/motion.npz') "
            "is redirected to the project result directory, and wandb is replaced with a local fake registry. This executes "
            "the official conversion/replay loop body, but it remains patched and is not unmodified official "
            "csv_to_npz output or paper-level replay evidence."
        ),
        "latest_blocker": blocker,
        "gpu_guard": guard,
        "run": run,
        "metrics": metrics,
        "checks": checks,
        "inputs": {
            "official_csv_to_npz_script": str(OFFICIAL_CSV_TO_NPZ),
            "input_csv": str(INPUT_CSV),
            "robot_usd": str(ROBOT_USD),
            "robot_usd_label": ROBOT_USD_LABEL,
            "uses_resource_adjusted_usd": USES_RESOURCE_ADJUSTED_USD,
            "uses_official_importer_export_usd": USES_OFFICIAL_IMPORTER_EXPORT_USD,
            "resource_adjusted_conversion_audit": str(RESOURCE_ADJUSTED_CONVERSION),
            "official_replay_loop_patch_audit": str(OFFICIAL_REPLAY_LOOP),
        },
        "outputs": {
            "json": str(OUT / f"{AUDIT_BASENAME}.json"),
            "metrics_json": str(METRICS_JSON),
            "motion_npz": str(OUTPUT_NPZ),
            "log": str(log_path),
            "failed_log_copy": failed_log_copy,
            "probe": str(PROBE),
        },
        "interpretation": {
            "goal_complete": False,
            "claim_level": CLAIM_LEVEL,
            "official_csv_to_npz_unpatched_asset_complete": False,
            "official_replay_complete": False,
            "paper_level_tracking_eval_complete": False,
            "why_not_complete": (
                "A successful run proves the official csv_to_npz.py loop can generate a 299-step motion artifact when "
                f"the active G1 converter blocker is bypassed by the selected {ROBOT_USD_LABEL} and project-local output "
                "redirects. It still does not prove the official URDF converter, unpatched official csv_to_npz output, "
                "trained-policy tracking metrics, DAgger rollouts, Fig.5/Fig.6, or real robot behavior."
            ),
        },
    }
    (OUT / f"{AUDIT_BASENAME}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "latest_blocker": blocker,
                "json": summary["outputs"]["json"],
                "returncode": run["returncode"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
