#!/usr/bin/env python3
"""Run the official replay_npz.py entrypoint with a local bounded fake-WandB artifact."""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/official_replay_npz_entry_diagnostic"
LOG_DIR = ROOT / "logs/tracking_official_replay_npz_entry_diagnostic"
FAILED_DIR = ROOT / "res/failed_runs/tracking_official_replay_npz_entry_diagnostic"
TRACKING_PY = ROOT / "envs/bm_tracking/bin/python"
OFFICIAL_REPLAY = ROOT / "reproduction/third_party/official/whole_body_tracking/scripts/replay_npz.py"
MOTION_NPZ = (
    ROOT
    / "res/tracking/g1_resource_adjusted_csv_conversion/"
    "walk1_subject1_frames_1_180_resource_adjusted_motion.npz"
)
SOURCE_EQUIVALENCE = (
    ROOT
    / "res/tracking/g1_urdf_source_equivalence_audit/tracking_g1_urdf_source_equivalence_audit.json"
)
CSV_FULL_REPLAY = (
    ROOT
    / "res/tracking/g1_resource_adjusted_csv_full_replay/"
    "tracking_g1_resource_adjusted_csv_full_replay_audit.json"
)
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"
GPU_FOUNDATION_DEPS = (
    ROOT
    / "envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/"
    "omni.gpu_foundation-0.0.0+d02c707b.lx64.r.cp310/bin/deps"
)
FAKE_ARTIFACT_DIR = ROOT / "tmp/tracking_official_replay_npz_entry_diagnostic/fake_wandb_artifact"
PROBE = OUT / "tracking_official_replay_npz_entry_diagnostic_probe.py"
MAX_STEPS = 299
STALL_SECONDS = 900


PROBE_CODE = r"""
import os
import runpy
import sys
import traceback
import types
from pathlib import Path

OFFICIAL_REPLAY = Path(os.environ["BM_OFFICIAL_REPLAY"])
FAKE_ARTIFACT_DIR = Path(os.environ["BM_FAKE_ARTIFACT_DIR"])
REGISTRY_NAME = os.environ["BM_FAKE_REGISTRY_NAME"]
MAX_STEPS = int(os.environ["BM_MAX_STEPS"])


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
                print(f"BM_SENTINEL:bounded_is_running_step={self._calls}", flush=True)
            return self._app.is_running()
        print(f"BM_SENTINEL:bounded_loop_complete={MAX_STEPS}", flush=True)
        return False

    def close(self, *args, **kwargs):
        print("BM_SENTINEL:simulation_app_close_called", flush=True)
        return self._app.close(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._app, name)


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def classify_log(text: str) -> dict[str, Any]:
    lowered = text.lower()
    step_markers = []
    for marker in [
        "bm_sentinel:bounded_is_running_step=1",
        "bm_sentinel:bounded_is_running_step=50",
        "bm_sentinel:bounded_is_running_step=100",
        "bm_sentinel:bounded_is_running_step=150",
        "bm_sentinel:bounded_is_running_step=200",
        "bm_sentinel:bounded_is_running_step=250",
        "bm_sentinel:bounded_is_running_step=299",
    ]:
        if marker in lowered:
            step_markers.append(marker.rsplit("=", 1)[-1])
    return {
        "before_runpy": "bm_sentinel:before_runpy=" in lowered,
        "add_app_launcher_args": "bm_sentinel:add_app_launcher_args" in lowered,
        "before_real_app_launcher": "bm_sentinel:before_real_app_launcher" in lowered,
        "after_real_app_launcher": "bm_sentinel:after_real_app_launcher" in lowered,
        "fake_wandb_artifact_request": "bm_sentinel:fake_wandb_artifact_request=" in lowered,
        "fake_wandb_download": "bm_sentinel:fake_wandb_download=" in lowered,
        "bounded_step_markers": step_markers,
        "bounded_step_299": "bm_sentinel:bounded_is_running_step=299" in lowered,
        "bounded_loop_complete": f"bm_sentinel:bounded_loop_complete={MAX_STEPS}" in lowered,
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
        "urdf_importer_signature": "urdf" in lowered and ("import" in lowered or "converter" in lowered),
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
            "BM_FAKE_REGISTRY_NAME": "bm-local/g1_resource_adjusted_motion:latest",
            "BM_MAX_STEPS": str(MAX_STEPS),
            "BM_DEVICE": "cuda:4",
            "BM_KIT_ARGS": (
                "--/renderer/multiGpu/enabled=false "
                "--/renderer/multiGpu/autoEnable=false "
                "--/renderer/multiGpu/maxGpuCount=1 "
                "--/renderer/activeGpu=4 "
                "--/physics/cudaDevice=4"
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
    return {
        "command": command,
        "returncode": proc.returncode,
        "duration_seconds": round(time.time() - start, 3),
        "stalled": stalled,
        "markers": classify_log(text),
        "log": str(log_path),
    }


def determine_blocker(run: dict[str, Any]) -> str:
    markers = run["markers"]
    if run["returncode"] == 0 and markers["bounded_loop_complete"]:
        return "none_bounded_official_replay_entry_completed"
    if markers["vulkan_device_lost"]:
        return "vulkan_device_lost"
    if markers["permission_to_save_false"] or markers["failed_to_save_layer"] or markers["empty_robot_after_converter"]:
        return "official_urdf_converter_layer_save_blocked"
    if run["returncode"] in {-15, 143} and markers["after_real_app_launcher"] and not markers["fake_wandb_download"]:
        return "process_terminated_after_app_launcher_before_motion_load"
    if markers["driver_shader_cache_shutdown_error"] and markers["after_real_app_launcher"]:
        return "kit_driver_shader_cache_shutdown_error_after_app_launcher"
    if markers["urdf_importer_signature"] and markers["exception"]:
        return "official_urdf_converter_or_asset_import_failed"
    if run["stalled"]:
        return "stall_timeout"
    if markers["exception"] or markers["traceback"]:
        return "python_exception"
    if not markers["after_real_app_launcher"]:
        return "app_launcher_or_kit_startup"
    if not markers["fake_wandb_download"]:
        return "artifact_or_scene_setup_before_motion_load"
    if not markers["bounded_step_299"]:
        return "replay_loop_did_not_reach_full_motion"
    return "unknown_failed_checks"


def main() -> None:
    prepare_probe()
    log_path = LOG_DIR / "tracking_official_replay_npz_entry_diagnostic.log"
    run = run_probe(log_path)
    blocker = determine_blocker(run)
    source_equivalence = read_json(SOURCE_EQUIVALENCE)
    csv_full_replay = read_json(CSV_FULL_REPLAY)
    checks = {
        "official_replay_script_exists": OFFICIAL_REPLAY.is_file(),
        "tracking_python_exists": TRACKING_PY.is_file(),
        "motion_npz_exists": MOTION_NPZ.is_file(),
        "fake_artifact_motion_symlink_exists": (FAKE_ARTIFACT_DIR / "motion.npz").exists(),
        "source_equivalence_available": source_equivalence.get("status") == "ok_with_source_differences_recorded",
        "resource_adjusted_full_replay_available": csv_full_replay.get("status") == "ok_resource_adjusted_csv_full_replay",
        "before_runpy_seen": run["markers"]["before_runpy"],
        "app_launcher_args_seen": run["markers"]["add_app_launcher_args"],
        "app_launcher_constructed": run["markers"]["after_real_app_launcher"],
        "fake_wandb_download_seen": run["markers"]["fake_wandb_download"],
        "bounded_step_299_seen": run["markers"]["bounded_step_299"],
        "bounded_loop_complete_seen": run["markers"]["bounded_loop_complete"],
        "process_returned_zero": run["returncode"] == 0,
        "no_stall_timeout": run["stalled"] is False,
        "does_not_modify_official_worktree": True,
        "does_not_claim_resource_adjusted_motion_is_official_motion": True,
        "does_not_claim_paper_level_replay": True,
        "does_not_start_training": True,
    }
    success = checks["bounded_loop_complete_seen"] and checks["process_returned_zero"]
    if success:
        status = "ok_official_replay_npz_entry_diagnostic"
    else:
        status = "ok_with_official_replay_npz_entry_blocker"
        failed_copy = FAILED_DIR / "tracking_official_replay_npz_entry_diagnostic.log"
        failed_copy.write_text(log_path.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_official_replay_npz_entry_diagnostic",
        "scope": (
            "Runs the official whole_body_tracking scripts/replay_npz.py entrypoint without modifying the official "
            "worktree. A local fake-WandB artifact points to the existing official-CSV-derived motion.npz, and a "
            "bounded AppLauncher wrapper limits the official replay loop to 299 is_running() calls. This diagnoses "
            "the official replay entry surface only; it is not official csv_to_npz output, not a trained-policy "
            "evaluation, not PPO, and not paper-level tracking evidence."
        ),
        "latest_blocker": blocker,
        "run": run,
        "checks": checks,
        "inputs": {
            "official_replay_script": str(OFFICIAL_REPLAY),
            "motion_npz": str(MOTION_NPZ),
            "fake_artifact_dir": str(FAKE_ARTIFACT_DIR),
            "source_equivalence_audit": str(SOURCE_EQUIVALENCE),
            "resource_adjusted_full_replay_audit": str(CSV_FULL_REPLAY),
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The diagnostic either localizes the official replay entry failure or confirms the bounded replay "
                "loop can execute with a fake local artifact. In both cases the input motion remains an "
                "official-CSV-derived resource-adjusted motion file, not official csv_to_npz.py output, and no "
                "trained policy, tracking metric, DAgger rollout, or paper-level video is produced."
            ),
            "next_action": (
                "If the official replay entry completed, use it as a gate before controlled PPO diagnostics. If it "
                "failed, use latest_blocker and the retained log to continue official URDF/USD converter recovery."
            ),
        },
        "outputs": {
            "json": str(OUT / "tracking_official_replay_npz_entry_diagnostic_audit.json"),
            "log": str(log_path),
            "failed_log_copy": str(
                FAILED_DIR / "tracking_official_replay_npz_entry_diagnostic.log"
                if status != "ok_official_replay_npz_entry_diagnostic"
                else ""
            ),
            "probe": str(PROBE),
        },
    }
    (OUT / "tracking_official_replay_npz_entry_diagnostic_audit.json").write_text(
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
