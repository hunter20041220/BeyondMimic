#!/usr/bin/env python3
"""Audit the IsaacLab/Kit inotify budget blocker without launching Kit."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/setup/kit_inotify_budget_audit"
RETRY_LOG = ROOT / "logs/setup/isaaclab_headless_smoke_retry.log"
WATCH_LIMIT_TARGET = 524288
INSTANCE_LIMIT_TARGET = 1024


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


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


def df_available_bytes(path: Path) -> tuple[int, int | None, str]:
    rc, out = command_output(["df", "-PB1", str(path)])
    if rc != 0:
        return rc, None, out
    lines = [line.split() for line in out.splitlines() if line.strip()]
    if len(lines) < 2 or len(lines[1]) < 4:
        return rc, None, out
    try:
        return rc, int(lines[1][3]), out
    except ValueError:
        return rc, None, out


def parse_failed_watch_paths(log_text: str) -> list[str]:
    paths = re.findall(r"Failed to create change watch for `([^`]+)`: errno=28", log_text)
    return sorted(set(paths))


def bounded_directory_count(roots: list[Path], stop_after: int, seconds: float = 60.0) -> dict[str, Any]:
    start = time.monotonic()
    count = 0
    visited_roots: list[str] = []
    truncated = False
    missing_roots: list[str] = []
    for root in roots:
        if not root.exists():
            missing_roots.append(str(root))
            continue
        visited_roots.append(str(root))
        for _dirpath, dirnames, _filenames in os.walk(root):
            count += 1
            if count > stop_after:
                truncated = True
                return {
                    "directory_count_lower_bound": count,
                    "stop_after": stop_after,
                    "truncated": truncated,
                    "elapsed_seconds": time.monotonic() - start,
                    "visited_roots": visited_roots,
                    "missing_roots": missing_roots,
                }
            if time.monotonic() - start > seconds:
                truncated = True
                return {
                    "directory_count_lower_bound": count,
                    "stop_after": stop_after,
                    "truncated": truncated,
                    "elapsed_seconds": time.monotonic() - start,
                    "visited_roots": visited_roots,
                    "missing_roots": missing_roots,
                }
            # Keep traversal deterministic enough for audit output while still cheap.
            dirnames.sort()
    return {
        "directory_count_lower_bound": count,
        "stop_after": stop_after,
        "truncated": truncated,
        "elapsed_seconds": time.monotonic() - start,
        "visited_roots": visited_roots,
        "missing_roots": missing_roots,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log_text = read_text(RETRY_LOG)
    max_watches = read_int("/proc/sys/fs/inotify/max_user_watches")
    max_instances = read_int("/proc/sys/fs/inotify/max_user_instances")
    sysctl_rc, sysctl_out = command_output(["sysctl", "fs.inotify.max_user_watches", "fs.inotify.max_user_instances"])
    df_rc, avail_bytes, df_out = df_available_bytes(ROOT)
    failed_paths = parse_failed_watch_paths(log_text)
    watch_roots = [
        ROOT / "envs/isaacsim-4.5.0",
        ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0",
    ]
    dir_budget = bounded_directory_count(watch_roots, stop_after=(max_watches or 8192))
    enough_disk_for_smoke = (avail_bytes or 0) > 10 * 1024**3
    current_below_target = (max_watches or 0) < WATCH_LIMIT_TARGET or (max_instances or 0) < INSTANCE_LIMIT_TARGET
    directory_pressure_exceeds_current = dir_budget["directory_count_lower_bound"] > (max_watches or 0)

    rows = [
        {
            "check": "retry_log_records_inotify_errno28",
            "status": "pass" if "errno=28" in log_text and failed_paths else "fail",
            "evidence": str(RETRY_LOG),
            "detail": f"unique_failed_watch_paths={len(failed_paths)}, errno28_count={log_text.count('errno=28')}",
        },
        {
            "check": "current_inotify_limits_below_retry_target",
            "status": "pass" if current_below_target else "review",
            "evidence": "/proc/sys/fs/inotify",
            "detail": (
                f"max_user_watches={max_watches}, max_user_instances={max_instances}, "
                f"target={WATCH_LIMIT_TARGET}/{INSTANCE_LIMIT_TARGET}"
            ),
        },
        {
            "check": "watch_tree_directory_pressure_exceeds_current_limit",
            "status": "pass" if directory_pressure_exceeds_current else "review",
            "evidence": ";".join(str(path) for path in watch_roots),
            "detail": (
                f"directory_count_lower_bound={dir_budget['directory_count_lower_bound']}, "
                f"current_max_user_watches={max_watches}, truncated={dir_budget['truncated']}"
            ),
        },
        {
            "check": "filesystem_capacity_not_primary_no_space_cause",
            "status": "pass" if enough_disk_for_smoke else "review",
            "evidence": str(ROOT),
            "detail": f"df_available_bytes={avail_bytes}, df_return_code={df_rc}",
        },
        {
            "check": "does_not_launch_kit_or_training",
            "status": "pass",
            "evidence": __file__,
            "detail": "Static/sysctl/filesystem/log audit only; no SimulationApp, PPO, replay, or training command is run.",
        },
    ]
    failed = [row for row in rows if row["status"] == "fail"]
    summary: dict[str, Any] = {
        "status": "ok" if not failed else "failed",
        "experiment_type": "kit_inotify_budget_audit",
        "scope": "machine-readable evidence for the IsaacLab/Kit inotify blocker and retry preconditions",
        "row_count": len(rows),
        "failed_row_count": len(failed),
        "rows": rows,
        "failed_watch_paths": failed_paths,
        "metrics": {
            "max_user_watches": max_watches,
            "max_user_instances": max_instances,
            "watch_limit_target": WATCH_LIMIT_TARGET,
            "instance_limit_target": INSTANCE_LIMIT_TARGET,
            "sysctl_return_code": sysctl_rc,
            "sysctl_output": sysctl_out,
            "df_return_code": df_rc,
            "df_available_bytes": avail_bytes,
            "df_output": df_out,
            "errno28_count": log_text.count("errno=28"),
            "change_watch_failure_count": log_text.count("Failed to create change watch"),
            "unique_failed_watch_path_count": len(failed_paths),
            **dir_budget,
        },
        "checks": {
            "retry_log_exists": RETRY_LOG.is_file() and RETRY_LOG.stat().st_size > 0,
            "retry_log_records_errno28": "errno=28" in log_text,
            "failed_watch_paths_parsed": len(failed_paths) >= 10,
            "current_inotify_below_retry_target": current_below_target,
            "directory_pressure_exceeds_current_limit": directory_pressure_exceeds_current,
            "filesystem_capacity_not_primary_no_space_cause": enough_disk_for_smoke,
            "does_not_launch_kit_or_training": True,
            "does_not_claim_tracking_reproduction_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The retry log and current host limits still prove an inotify watcher-budget blocker. "
                "Disk capacity is available, so the Kit 'No space left on device' messages should not be "
                "treated as ordinary filesystem capacity exhaustion. Full IsaacLab/Kit smoke remains gated "
                "until host inotify limits are raised and the live smoke is rerun."
            ),
        },
        "outputs": {
            "json": str(OUT / "kit_inotify_budget_audit.json"),
            "tsv": str(OUT / "kit_inotify_budget_audit.tsv"),
        },
    }
    (OUT / "kit_inotify_budget_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (OUT / "kit_inotify_budget_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["check", "status", "evidence", "detail"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
