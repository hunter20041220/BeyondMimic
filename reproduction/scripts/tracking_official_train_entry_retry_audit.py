#!/usr/bin/env python3
"""Bounded retry of the official whole_body_tracking training entry point."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/official_train_entry_retry_audit"
LOG_DIR = ROOT / "logs/tracking_official_train_entry_retry"
ISAACLAB = ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0"
WBT = ROOT / "reproduction/third_party/official/whole_body_tracking"
LOCAL_TRAIN = ROOT / "reproduction/generated/whole_body_tracking_local/rsl_rl/train_local.py"
MOTION_NPZ = ROOT / "reproduction/data/tracking_motion_npz_fixtures/walk1_subject1_frames_1_180_debug_motion.npz"
TIMEOUT_SECONDS = 90
WATCH_TARGET = 524288
INSTANCE_TARGET = 1024


def read_int(path: str) -> int | None:
    try:
        return int(Path(path).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def command_output(args: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(args, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except OSError as exc:
        return 127, str(exc)
    return proc.returncode, proc.stdout.strip()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["check", "status", "evidence", "detail"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def classify_output(returncode: int, text: str) -> dict[str, Any]:
    lower = text.lower()
    has_inotify = "failed to create change watch" in lower or "errno=28" in lower
    has_no_space = "no space left on device" in lower
    timed_out = returncode == 124 or "timed out" in lower
    module_missing = "modulenotfounderror" in lower or "no module named" in lower
    if returncode == 0:
        result = "unexpected_success"
        reason = "The bounded official training entry command returned 0; this requires manual review before any training claim."
    elif has_inotify or has_no_space:
        result = "blocked_inotify"
        reason = "The retry reproduced Kit/inotify watcher-budget failure signatures."
    elif timed_out:
        result = "timeout_without_success"
        reason = "The bounded retry timed out before reaching a training endpoint."
    elif module_missing:
        result = "blocked_missing_runtime_module"
        reason = "The retry failed before training because a required Kit/Isaac runtime module was unavailable."
    else:
        result = "failed_other"
        reason = "The retry failed before a valid training endpoint for another captured reason."
    return {
        "retry_result": result,
        "reason": reason,
        "returncode": returncode,
        "has_inotify_watch_failure": has_inotify,
        "has_no_space_left_on_device": has_no_space,
        "timed_out": timed_out,
        "module_missing": module_missing,
    }


def latest_log_dirs() -> list[str]:
    root = WBT / "logs/rsl_rl"
    if not root.is_dir():
        return []
    paths = sorted((path for path in root.rglob("*") if path.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(path) for path in paths[:8]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    max_watches = read_int("/proc/sys/fs/inotify/max_user_watches")
    max_instances = read_int("/proc/sys/fs/inotify/max_user_instances")
    sysctl_rc, sysctl_out = command_output(["sysctl", "fs.inotify.max_user_watches", "fs.inotify.max_user_instances"])
    pgrep_rc, pgrep_out = command_output(
        ["bash", "-lc", "pgrep -af 'kit|isaac|whole_body_tracking|rsl_rl|python.*train.py' || true"]
    )
    env = os.environ.copy()
    env.update(
        {
            "ROOT": str(ROOT),
            "DOWNLOAD_ROOT": str(ROOT / "download"),
            "WORKSPACE": str(ROOT / "reproduction"),
            "ENV_ROOT": str(ROOT / "envs"),
            "CACHE_ROOT": str(ROOT / "cache"),
            "TMP_ROOT": str(ROOT / "tmp"),
            "LOG_ROOT": str(ROOT / "logs"),
            "RES_ROOT": str(ROOT / "res"),
            "CONDA_PKGS_DIRS": str(ROOT / "cache/conda_pkgs"),
            "PIP_CACHE_DIR": str(ROOT / "cache/pip"),
            "HF_HOME": str(ROOT / "cache/huggingface"),
            "TRANSFORMERS_CACHE": str(ROOT / "cache/huggingface/transformers"),
            "TORCH_HOME": str(ROOT / "cache/torch"),
            "XDG_CACHE_HOME": str(ROOT / "cache/xdg"),
            "WANDB_DIR": str(ROOT / "logs/wandb"),
            "TMPDIR": str(ROOT / "tmp"),
            "TEMP": str(ROOT / "tmp"),
            "TMP": str(ROOT / "tmp"),
            "TERM": "xterm",
        }
    )
    command = [
        "timeout",
        f"{TIMEOUT_SECONDS}s",
        str(ISAACLAB / "isaaclab.sh"),
        "-p",
        str(LOCAL_TRAIN),
        "--task=Tracking-Flat-G1-v0",
        "--motion_file",
        str(MOTION_NPZ),
        "--num_envs=1",
        "--max_iterations=1",
        "--headless",
        "--device",
        "cpu",
        "--logger",
        "tensorboard",
        "--run_name",
        "official_train_entry_retry_audit",
    ]
    log_path = LOG_DIR / "official_train_entry_retry.log"
    started = datetime.now().isoformat(timespec="seconds")
    proc = subprocess.run(
        command,
        cwd=WBT,
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=TIMEOUT_SECONDS + 45,
    )
    finished = datetime.now().isoformat(timespec="seconds")
    output = proc.stdout or ""
    log_path.write_text(output, encoding="utf-8", errors="replace")
    tail_path = LOG_DIR / "official_train_entry_retry_tail.log"
    tail_path.write_text("\n".join(output.splitlines()[-120:]) + "\n", encoding="utf-8")
    classification = classify_output(proc.returncode, output)
    current_below_target = (max_watches or 0) < WATCH_TARGET or (max_instances or 0) < INSTANCE_TARGET
    rows = [
        {
            "check": "local_train_script_exists",
            "status": "pass" if LOCAL_TRAIN.is_file() else "fail",
            "evidence": str(LOCAL_TRAIN),
            "detail": f"size={LOCAL_TRAIN.stat().st_size if LOCAL_TRAIN.is_file() else 0}",
        },
        {
            "check": "local_motion_npz_exists",
            "status": "pass" if MOTION_NPZ.is_file() else "fail",
            "evidence": str(MOTION_NPZ),
            "detail": f"size={MOTION_NPZ.stat().st_size if MOTION_NPZ.is_file() else 0}",
        },
        {
            "check": "current_inotify_below_training_retry_target",
            "status": "blocked" if current_below_target else "clear",
            "evidence": "/proc/sys/fs/inotify",
            "detail": f"max_user_watches={max_watches}, max_user_instances={max_instances}",
        },
        {
            "check": "official_train_entry_attempted",
            "status": "pass",
            "evidence": str(log_path),
            "detail": f"returncode={proc.returncode}, timeout_seconds={TIMEOUT_SECONDS}",
        },
        {
            "check": "retry_result_classified",
            "status": classification["retry_result"],
            "evidence": str(tail_path),
            "detail": classification["reason"],
        },
    ]
    required_ok = all(row["status"] != "fail" for row in rows)
    checks = {
        "local_train_script_exists": LOCAL_TRAIN.is_file(),
        "local_motion_npz_exists": MOTION_NPZ.is_file(),
        "log_written": log_path.is_file() and log_path.stat().st_size > 0,
        "tail_log_written": tail_path.is_file() and tail_path.stat().st_size > 0,
        "inotify_recorded": max_watches is not None and max_instances is not None,
        "current_inotify_below_retry_target": current_below_target,
        "command_attempted_once": True,
        "retry_did_not_reach_valid_training_success": classification["retry_result"] != "unexpected_success",
        "does_not_claim_training_success": True,
        "does_not_claim_goal_complete": True,
    }
    summary: dict[str, Any] = {
        "status": "ok" if required_ok and all(checks.values()) else "review_required",
        "experiment_type": "official_train_entry_retry_audit",
        "scope": "bounded retry of the official/locally patched whole_body_tracking training entry point",
        "started_at": started,
        "finished_at": finished,
        "timeout_seconds": TIMEOUT_SECONDS,
        "command": command,
        "cwd": str(WBT),
        "returncode": proc.returncode,
        "classification": classification,
        "sysctl": {
            "returncode": sysctl_rc,
            "output": sysctl_out,
            "max_user_watches": max_watches,
            "max_user_instances": max_instances,
            "watch_target": WATCH_TARGET,
            "instance_target": INSTANCE_TARGET,
        },
        "process_snapshot": {
            "pgrep_return_code": pgrep_rc,
            "matching_processes": pgrep_out.splitlines(),
        },
        "latest_official_log_dirs": latest_log_dirs(),
        "rows": rows,
        "checks": checks,
        "outputs": {
            "json": str(OUT / "tracking_official_train_entry_retry_audit.json"),
            "tsv": str(OUT / "tracking_official_train_entry_retry_audit.tsv"),
            "log": str(log_path),
            "tail_log": str(tail_path),
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This is a bounded entry-point retry. It does not claim PPO training success unless the command reaches "
                "a valid endpoint with complete training artifacts; current host inotify limits remain below the "
                "documented retry threshold."
            ),
        },
    }
    atomic_write_json(OUT / "tracking_official_train_entry_retry_audit.json", summary)
    write_tsv(OUT / "tracking_official_train_entry_retry_audit.tsv", rows)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "result": classification["retry_result"],
                "returncode": proc.returncode,
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if summary["status"] not in {"ok", "review_required"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
