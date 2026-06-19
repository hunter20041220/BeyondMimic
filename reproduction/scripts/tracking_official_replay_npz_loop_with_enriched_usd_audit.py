#!/usr/bin/env python3
"""Run the official replay_npz.py loop with a local enriched-USD asset patch.

This audit intentionally leaves the official whole_body_tracking worktree
unchanged. The probe process monkeypatches the in-memory G1 robot config after
AppLauncher starts, replacing the official URDF converter spawn path with the
resource-adjusted enriched USD that already passed the full replay/task gates.
The replay loop itself still comes from the official replay_npz.py script.
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


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/official_replay_npz_loop_with_enriched_usd"
LOG_DIR = ROOT / "logs/tracking_official_replay_npz_loop_with_enriched_usd"
FAILED_DIR = ROOT / "res/failed_runs/tracking_official_replay_npz_loop_with_enriched_usd"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
OFFICIAL_REPLAY = ROOT / "reproduction/third_party/official/whole_body_tracking/scripts/replay_npz.py"
MOTION_NPZ = Path(
    os.environ.get(
        "BM_OFFICIAL_REPLAY_LOOP_MOTION_NPZ",
        str(
            ROOT
            / "res/tracking/g1_resource_adjusted_csv_conversion/"
            "walk1_subject1_frames_1_180_resource_adjusted_motion.npz"
        ),
    )
)
ENRICHED_USD = (
    ROOT
    / "res/tracking/g1_resource_adjusted_enriched_usd/"
    "g1_resource_adjusted_29dof_enriched_scaffold.usda"
)
CSV_FULL_REPLAY = (
    ROOT
    / "res/tracking/g1_resource_adjusted_csv_full_replay/"
    "tracking_g1_resource_adjusted_csv_full_replay_audit.json"
)
CSV_TASK_EVAL = (
    ROOT
    / "res/tracking/g1_resource_adjusted_csv_task_eval/"
    "tracking_g1_resource_adjusted_csv_task_eval_audit.json"
)
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
FAKE_ARTIFACT_DIR = Path(
    os.environ.get(
        "BM_OFFICIAL_REPLAY_LOOP_FAKE_ARTIFACT_DIR",
        str(ROOT / "tmp/tracking_official_replay_npz_loop_with_enriched_usd/fake_wandb_artifact"),
    )
)
PROBE = OUT / "tracking_official_replay_npz_loop_with_enriched_usd_probe.py"
TARGET_GPU = int(os.environ.get("BM_OFFICIAL_REPLAY_LOOP_TARGET_GPU", "4"))
WATCH_GPUS = [4, 7]
MAX_STEPS = int(os.environ.get("BM_OFFICIAL_REPLAY_LOOP_MAX_STEPS", "299"))
STALL_SECONDS = 900
SUCCESS_EXIT_GRACE_SECONDS = int(os.environ.get("BM_OFFICIAL_REPLAY_LOOP_SUCCESS_EXIT_GRACE_SECONDS", "30"))
WANGJC_PATH_MARKER = "/mnt/infini-data/test/wangjc/"
REGISTRY_NAME = os.environ.get("BM_OFFICIAL_REPLAY_LOOP_REGISTRY_NAME", "bm-local/g1_resource_adjusted_motion:latest")
LOG_BASENAME = os.environ.get(
    "BM_OFFICIAL_REPLAY_LOOP_LOG_BASENAME",
    "tracking_official_replay_npz_loop_with_enriched_usd",
)


PROBE_CODE = r"""
import os
import runpy
import signal
import sys
import traceback
import types
from pathlib import Path

OFFICIAL_REPLAY = Path(os.environ["BM_OFFICIAL_REPLAY"])
FAKE_ARTIFACT_DIR = Path(os.environ["BM_FAKE_ARTIFACT_DIR"])
REGISTRY_NAME = os.environ["BM_FAKE_REGISTRY_NAME"]
MAX_STEPS = int(os.environ["BM_MAX_STEPS"])
ENRICHED_USD = Path(os.environ["BM_ENRICHED_USD"])


def _sigterm_handler(signum, frame):
    print(f"BM_SENTINEL:received_signal={signum}", flush=True)
    traceback.print_stack(frame)
    print("BM_SENTINEL:received_signal_ignored_for_bounded_probe", flush=True)


signal.signal(signal.SIGTERM, _sigterm_handler)


class _FakeArtifact:
    def download(self):
        print(f"BM_SENTINEL:fake_wandb_download={FAKE_ARTIFACT_DIR}", flush=True)
        return str(FAKE_ARTIFACT_DIR)


class _FakeApi:
    def artifact(self, name):
        print(f"BM_SENTINEL:fake_wandb_artifact_request={name}", flush=True)
        if name != REGISTRY_NAME:
            raise RuntimeError(f"Unexpected fake registry name: {name!r} != {REGISTRY_NAME!r}")
        return _FakeArtifact()


fake_wandb = types.ModuleType("wandb")
fake_wandb.Api = _FakeApi
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
                print(f"BM_SENTINEL:official_loop_is_running_call={self._calls}", flush=True)
            return self._app.is_running()
        print(f"BM_SENTINEL:official_loop_complete={MAX_STEPS}", flush=True)
        return False

    def close(self, *args, **kwargs):
        print("BM_SENTINEL:simulation_app_close_called", flush=True)
        return self._app.close(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._app, name)


def patch_g1_cfg_to_enriched_usd():
    import isaaclab.sim as sim_utils
    import whole_body_tracking.robots.g1 as g1_module

    cfg = g1_module.G1_CYLINDER_CFG.copy()
    cfg.spawn = sim_utils.UsdFileCfg(
        usd_path=str(ENRICHED_USD),
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
    print(f"BM_SENTINEL:g1_cfg_patched_to_enriched_usd={ENRICHED_USD}", flush=True)


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
        patch_g1_cfg_to_enriched_usd()

    def __getattr__(self, name):
        return getattr(self._real, name)


isaaclab_app.AppLauncher = _BoundedAppLauncher
sys.argv = [
    str(OFFICIAL_REPLAY),
    "--registry_name",
    REGISTRY_NAME.removesuffix(":latest"),
    "--headless",
    "--device",
    os.environ["BM_DEVICE"],
]

print(f"BM_SENTINEL:before_runpy={OFFICIAL_REPLAY}", flush=True)
try:
    runpy.run_path(str(OFFICIAL_REPLAY), run_name="__main__")
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
    path = guard_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_gpu47_wangjc_official_replay_loop_guard.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["json"] = str(path)
    return summary


def classify_log(text: str) -> dict[str, Any]:
    lowered = text.lower()
    step_calls = []
    for marker in [
        "bm_sentinel:official_loop_is_running_call=1",
        "bm_sentinel:official_loop_is_running_call=50",
        "bm_sentinel:official_loop_is_running_call=100",
        "bm_sentinel:official_loop_is_running_call=150",
        "bm_sentinel:official_loop_is_running_call=200",
        "bm_sentinel:official_loop_is_running_call=250",
        "bm_sentinel:official_loop_is_running_call=299",
    ]:
        if marker in lowered:
            step_calls.append(int(marker.rsplit("=", 1)[-1]))
    return {
        "before_runpy": "bm_sentinel:before_runpy=" in lowered,
        "add_app_launcher_args": "bm_sentinel:add_app_launcher_args" in lowered,
        "before_real_app_launcher": "bm_sentinel:before_real_app_launcher" in lowered,
        "after_real_app_launcher": "bm_sentinel:after_real_app_launcher" in lowered,
        "g1_cfg_patched_to_enriched_usd": "bm_sentinel:g1_cfg_patched_to_enriched_usd=" in lowered,
        "fake_wandb_artifact_request": "bm_sentinel:fake_wandb_artifact_request=" in lowered,
        "fake_wandb_download": "bm_sentinel:fake_wandb_download=" in lowered,
        "official_loop_step_calls": step_calls,
        "official_loop_call_299": "bm_sentinel:official_loop_is_running_call=299" in lowered,
        "official_loop_complete": f"bm_sentinel:official_loop_complete={MAX_STEPS}" in lowered,
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
        "motion_loaded": "motion loaded" in lowered or "time_step_total" in lowered,
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
            "BM_OFFICIAL_REPLAY": str(OFFICIAL_REPLAY),
            "BM_FAKE_ARTIFACT_DIR": str(FAKE_ARTIFACT_DIR),
            "BM_FAKE_REGISTRY_NAME": REGISTRY_NAME,
            "BM_MAX_STEPS": str(MAX_STEPS),
            "BM_ENRICHED_USD": str(ENRICHED_USD),
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
    FAKE_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    link = FAKE_ARTIFACT_DIR / "motion.npz"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(MOTION_NPZ)
    PROBE.write_text(textwrap.dedent(PROBE_CODE), encoding="utf-8")


def run_probe(log_path: Path) -> dict[str, Any]:
    command = [str(TRACKING_PY), str(PROBE)]
    start = time.time()
    stalled = False
    forced_after_success_sentinel = False
    success_seen_at: float | None = None
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
                text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
                markers = classify_log(text)
                if (
                    markers["official_loop_complete"]
                    and markers["simulation_app_close_called"]
                    and success_seen_at is None
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
                    proc.wait(timeout=30)
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


def determine_blocker(run: dict[str, Any]) -> str:
    markers = run["markers"]
    completed_after_success_sentinel = markers["official_loop_complete"] and markers["simulation_app_close_called"]
    if (
        (run["returncode"] == 0 or run.get("forced_after_success_sentinel"))
        and completed_after_success_sentinel
    ):
        return "none_official_replay_loop_completed_with_enriched_usd_patch"
    if completed_after_success_sentinel and run["returncode"] in {-9, -15}:
        return "shutdown_signal_after_official_replay_loop_completed"
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
    if not markers["g1_cfg_patched_to_enriched_usd"]:
        return "g1_cfg_patch_failed"
    if not markers["fake_wandb_download"]:
        return "artifact_or_scene_setup_before_motion_load"
    if not markers["official_loop_call_299"]:
        return "official_replay_loop_did_not_reach_full_motion_bound"
    return "unknown_failed_checks"


def main() -> None:
    prepare_probe()
    guard = kill_wangjc_on_watch_gpus()
    log_path = LOG_DIR / f"{LOG_BASENAME}.log"
    run = run_probe(log_path)
    blocker = determine_blocker(run)
    csv_full_replay = read_json(CSV_FULL_REPLAY)
    csv_task_eval = read_json(CSV_TASK_EVAL)
    checks = {
        "official_replay_script_exists": OFFICIAL_REPLAY.is_file(),
        "tracking_python_exists": TRACKING_PY.is_file(),
        "motion_npz_exists": MOTION_NPZ.is_file(),
        "enriched_usd_exists": ENRICHED_USD.is_file(),
        "fake_artifact_motion_symlink_exists": (FAKE_ARTIFACT_DIR / "motion.npz").exists(),
        "resource_adjusted_full_replay_available": csv_full_replay.get("status") == "ok_resource_adjusted_csv_full_replay",
        "resource_adjusted_task_eval_available": csv_task_eval.get("status") == "ok_resource_adjusted_csv_task_eval",
        "before_runpy_seen": run["markers"]["before_runpy"],
        "app_launcher_args_seen": run["markers"]["add_app_launcher_args"],
        "app_launcher_constructed": run["markers"]["after_real_app_launcher"],
        "g1_cfg_patched_to_enriched_usd": run["markers"]["g1_cfg_patched_to_enriched_usd"],
        "fake_wandb_download_seen": run["markers"]["fake_wandb_download"],
        "official_loop_call_299_seen": run["markers"]["official_loop_call_299"],
        "official_loop_complete_seen": run["markers"]["official_loop_complete"],
        "after_runpy_seen": run["markers"]["after_runpy"],
        "process_returned_zero_or_forced_after_success_sentinel": (
            run["returncode"] == 0 or run.get("forced_after_success_sentinel") is True
        ),
        "forced_after_success_sentinel_recorded": bool(run.get("forced_after_success_sentinel")),
        "no_stall_timeout": run["stalled"] is False,
        "does_not_modify_official_worktree": True,
        "does_not_claim_resource_adjusted_asset_is_official_converter_output": True,
        "does_not_claim_resource_adjusted_motion_is_official_csv_to_npz_output": True,
        "does_not_claim_paper_level_replay": True,
        "does_not_start_training": True,
    }
    success = (
        checks["process_returned_zero_or_forced_after_success_sentinel"]
        and checks["official_loop_complete_seen"]
        and run["markers"]["simulation_app_close_called"]
        and checks["g1_cfg_patched_to_enriched_usd"]
        and checks["fake_wandb_download_seen"]
    )
    completed_with_shutdown_warning = (
        checks["official_loop_complete_seen"]
        and run["markers"]["simulation_app_close_called"]
        and checks["g1_cfg_patched_to_enriched_usd"]
        and checks["fake_wandb_download_seen"]
        and run["returncode"] in {-9, -15}
    )
    if success:
        status = "ok_official_replay_loop_with_enriched_usd_patch"
    elif completed_with_shutdown_warning:
        status = "ok_official_replay_loop_with_enriched_usd_patch_shutdown_warning"
    else:
        status = "ok_with_official_replay_loop_patch_blocker"
    failed_log_copy = ""
    if not success:
        failed_log = FAILED_DIR / f"{LOG_BASENAME}.log"
        failed_log.write_text(log_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        failed_log_copy = str(failed_log)
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_official_replay_npz_loop_with_enriched_usd_patch",
        "scope": (
            "Runs the official whole_body_tracking scripts/replay_npz.py loop while patching only runtime dependencies: "
            "a local fake-WandB artifact provides the official-CSV-derived motion.npz, and the in-memory G1 config uses "
            "the previously validated resource-adjusted enriched USD instead of the official URDF converter path. "
            "This is stronger than the local copied replay script because the loop body is official, but it is still "
            "not official csv_to_npz.py output, not official converter output, not PPO, and not paper-level tracking."
        ),
        "latest_blocker": blocker,
        "gpu_guard": guard,
        "run": run,
        "checks": checks,
        "inputs": {
            "official_replay_script": str(OFFICIAL_REPLAY),
            "motion_npz": str(MOTION_NPZ),
            "enriched_usd": str(ENRICHED_USD),
            "fake_artifact_dir": str(FAKE_ARTIFACT_DIR),
            "resource_adjusted_full_replay_audit": str(CSV_FULL_REPLAY),
            "resource_adjusted_task_eval_audit": str(CSV_TASK_EVAL),
        },
        "outputs": {
            "json": str(OUT / "tracking_official_replay_npz_loop_with_enriched_usd_audit.json"),
            "log": str(log_path),
            "failed_log_copy": failed_log_copy,
            "probe": str(PROBE),
        },
        "interpretation": {
            "goal_complete": False,
            "official_replay_complete": False,
            "paper_level_tracking_eval_complete": False,
            "official_loop_body_completed": bool(success or completed_with_shutdown_warning),
            "shutdown_warning": bool(completed_with_shutdown_warning),
            "why_not_complete": (
                "A successful run proves the official replay_npz.py loop can consume the local official-CSV-derived "
                "motion when the active G1 converter blocker is bypassed by the validated enriched USD. It still does "
                "not prove the official URDF converter, official csv_to_npz.py output, trained-policy tracking metrics, "
                "DAgger rollouts, Fig.5/Fig.6, or real robot behavior."
            ),
        },
    }
    (OUT / "tracking_official_replay_npz_loop_with_enriched_usd_audit.json").write_text(
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
