#!/usr/bin/env python3
"""Run PPO training using the official csv_to_npz loop motion artifact.

This reuses the existing resource-adjusted PPO harness but switches the motion
input to the NPZ produced by the official ``csv_to_npz.py`` loop under the
enriched-USD runtime patch. The robot asset is still the enriched USD workaround,
so this is not unpatched official BeyondMimic training.
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
BASE_SCRIPT = ROOT / "reproduction/scripts/tracking_g1_resource_adjusted_ppo_training_run.py"
OUT = ROOT / "res/tracking/g1_official_csv_loop_ppo_training_run"
LOG_DIR = ROOT / "logs/tracking_g1_official_csv_loop_ppo_training_run"
RUN_ROOT = ROOT / "res/runs/tracking_g1_official_csv_loop_ppo_training"
OFFICIAL_LOOP_MOTION_NPZ = (
    ROOT
    / "res/tracking/official_csv_to_npz_loop_with_enriched_usd/"
    "walk1_subject1_frames_1_180_official_loop_enriched_usd_motion.npz"
)
OFFICIAL_CSV_LOOP_AUDIT = (
    ROOT
    / "res/tracking/official_csv_to_npz_loop_with_enriched_usd/"
    "tracking_official_csv_to_npz_loop_with_enriched_usd_audit.json"
)
OFFICIAL_REPLAY_LOOP_AUDIT = (
    ROOT
    / "res/tracking/official_replay_npz_loop_with_enriched_usd/"
    "tracking_official_replay_npz_loop_with_enriched_usd_audit.json"
)
GPU_GUARD_DIR = ROOT / "res/gpu_guard"
TARGET_GPUS = [4, 7]
WANGJC_PATH_MARKER = "/mnt/infini-data/test/wangjc/"
DEFAULT_MAX_ITERATIONS = 300
DEFAULT_SEED = 20260629


def run_command(cmd: list[str]) -> str:
    import subprocess

    proc = subprocess.run(cmd, cwd=ROOT, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.stdout


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def gpu_index_to_bus_id() -> dict[int, str]:
    out = run_command(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"])
    mapping: dict[int, str] = {}
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            mapping[int(parts[0])] = parts[1]
        except ValueError:
            pass
    return mapping


def parse_gpu_processes() -> list[dict[str, Any]]:
    out = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[dict[str, Any]] = []
    for line in out.splitlines():
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


def kill_wangjc_on_target_gpus() -> dict[str, Any]:
    GPU_GUARD_DIR.mkdir(parents=True, exist_ok=True)
    bus = gpu_index_to_bus_id()
    target_bus = {bus[index] for index in TARGET_GPUS if index in bus}
    killed = []
    skipped_non_wangjc = []
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
            skipped_non_wangjc.append(item)
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
        "target_gpus": TARGET_GPUS,
        "policy": "kill only processes whose cmdline contains /mnt/infini-data/test/wangjc/",
        "killed": killed,
        "skipped_non_wangjc": skipped_non_wangjc,
    }
    path = GPU_GUARD_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_gpu47_wangjc_official_csv_loop_ppo_guard.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["json"] = str(path)
    return summary


def load_base_module():
    spec = importlib.util.spec_from_file_location("bm_resource_adjusted_ppo_training_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base PPO script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def patch_summary(summary: dict[str, Any], guard: dict[str, Any]) -> dict[str, Any]:
    official_csv_loop = load_json(OFFICIAL_CSV_LOOP_AUDIT)
    official_replay_loop = load_json(OFFICIAL_REPLAY_LOOP_AUDIT)
    summary["experiment_type"] = "tracking_official_csv_loop_ppo_training_run"
    summary["scope"] = (
        "PPO training with the official Tracking-Flat-G1-v0 manager stack using the motion artifact generated by "
        "the official csv_to_npz.py loop under the enriched-USD runtime patch. The robot asset remains the "
        "resource-adjusted enriched USD, so this is not unpatched official BeyondMimic paper-level training."
    )
    summary["gpu_guard"] = guard
    summary["inputs"]["motion_npz"] = str(OFFICIAL_LOOP_MOTION_NPZ)
    summary["inputs"]["official_csv_to_npz_loop_audit"] = str(OFFICIAL_CSV_LOOP_AUDIT)
    summary["inputs"]["official_replay_loop_audit"] = str(OFFICIAL_REPLAY_LOOP_AUDIT)
    summary["input_checks"]["official_csv_loop_motion_npz_exists"] = OFFICIAL_LOOP_MOTION_NPZ.is_file()
    summary["input_checks"]["official_csv_loop_audit_passed"] = (
        official_csv_loop.get("status") == "ok_official_csv_to_npz_loop_with_enriched_usd_patch"
    )
    summary["input_checks"]["official_replay_loop_audit_passed"] = (
        official_replay_loop.get("status") == "ok_official_replay_loop_with_enriched_usd_patch"
    )
    for rank_metric in summary.get("run", {}).get("rank_metrics", []):
        rank_metric["official_csv_to_npz_loop_output"] = True
        rank_metric["official_csv_to_npz_unpatched_output"] = False
        rank_metric["motion_file"] = str(OFFICIAL_LOOP_MOTION_NPZ)
    summary["interpretation"] = {
        "goal_complete": False,
        "official_ppo_training_complete": False,
        "paper_level_tracking_training_complete": False,
        "official_csv_loop_motion_training_complete": summary.get("status")
        == "ok_resource_adjusted_ppo_training_completed",
        "why_not_paper_level": (
            "The motion file was generated by the official csv_to_npz.py loop under the enriched-USD runtime patch, "
            "but the robot asset path still bypasses the unpatched official G1 URDF/USD converter. This is a "
            "substantive virtual training run for the local reproduction chain, not an official paper-level teacher."
        ),
    }
    return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    guard = kill_wangjc_on_target_gpus()

    module = load_base_module()
    module.OUT = OUT
    module.LOG_DIR = LOG_DIR
    module.RUN_ROOT = RUN_ROOT
    module.CSV_MOTION_NPZ = OFFICIAL_LOOP_MOTION_NPZ
    module.CANDIDATE_GPUS = TARGET_GPUS
    module.MAX_ITERATIONS = int(os.environ.get("BM_OFFICIAL_CSV_LOOP_PPO_MAX_ITERATIONS", str(DEFAULT_MAX_ITERATIONS)))
    module.SEED = int(os.environ.get("BM_OFFICIAL_CSV_LOOP_PPO_SEED", str(DEFAULT_SEED)))
    module.main()

    output_json = OUT / "tracking_g1_resource_adjusted_ppo_training_run.json"
    final_json = OUT / "tracking_g1_official_csv_loop_ppo_training_run.json"
    summary = patch_summary(load_json(output_json), guard)
    summary["status"] = (
        "ok_official_csv_loop_ppo_training_completed"
        if summary.get("status") == "ok_resource_adjusted_ppo_training_completed"
        else summary.get("status", "failed_official_csv_loop_ppo_training")
    )
    summary["outputs"]["json"] = str(final_json)
    output_json.unlink(missing_ok=True)
    final_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(final_json),
                "attempted_training": summary.get("run", {}).get("attempted_training"),
                "checkpoint_count": summary.get("run", {}).get("checkpoint_count"),
                "max_iterations": summary.get("config", {}).get("max_iterations"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
