#!/usr/bin/env python3
"""Audit live inotify fd/watch usage for the current host user."""

from __future__ import annotations

import csv
import json
import os
import pwd
import re
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/setup/inotify_live_usage_audit"
WATCH_TARGET = 524288
INSTANCE_TARGET = 1024
WATCH_RE = re.compile(r"\binotify\b")
WD_RE = re.compile(r"\bwd:")


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


def proc_uid(pid: str) -> int | None:
    try:
        for line in (Path("/proc") / pid / "status").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("Uid:"):
                parts = line.split()
                return int(parts[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def proc_cmd(pid: str) -> str:
    base = Path("/proc") / pid
    try:
        raw = (base / "cmdline").read_bytes()
        text = raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
        if text:
            return text
    except OSError:
        pass
    try:
        return (base / "comm").read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return ""


def count_fdinfo(pid: str) -> tuple[int, int, list[str]]:
    fdinfo_dir = Path("/proc") / pid / "fdinfo"
    fd_count = 0
    watch_count = 0
    samples: list[str] = []
    try:
        entries = sorted(fdinfo_dir.iterdir(), key=lambda p: p.name)
    except OSError:
        return 0, 0, samples
    for entry in entries:
        try:
            text = entry.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if WATCH_RE.search(text):
            fd_count += 1
            watches = len(WD_RE.findall(text))
            watch_count += watches
            if len(samples) < 3:
                first_watch = next((line for line in text.splitlines() if "inotify" in line), "")
                samples.append(f"fd={entry.name} watches={watches} sample={first_watch[:220]}")
    return fd_count, watch_count, samples


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    uid = os.getuid()
    username = pwd.getpwuid(uid).pw_name
    max_watches = read_int("/proc/sys/fs/inotify/max_user_watches")
    max_instances = read_int("/proc/sys/fs/inotify/max_user_instances")
    sysctl_rc, sysctl_out = command_output(["sysctl", "fs.inotify.max_user_watches", "fs.inotify.max_user_instances"])

    processes: list[dict[str, Any]] = []
    for proc in sorted(Path("/proc").iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if not proc.name.isdigit():
            continue
        if proc_uid(proc.name) != uid:
            continue
        fd_count, watch_count, samples = count_fdinfo(proc.name)
        if fd_count == 0 and watch_count == 0:
            continue
        processes.append(
            {
                "pid": int(proc.name),
                "command": proc_cmd(proc.name),
                "inotify_fd_count": fd_count,
                "inotify_watch_count": watch_count,
                "sample_fdinfo": samples,
            }
        )

    total_fds = sum(row["inotify_fd_count"] for row in processes)
    total_watches = sum(row["inotify_watch_count"] for row in processes)
    sorted_processes = sorted(processes, key=lambda row: (-row["inotify_watch_count"], -row["inotify_fd_count"], row["pid"]))
    top_processes = sorted_processes[:20]
    watch_headroom = None if max_watches is None else max_watches - total_watches
    instance_headroom = None if max_instances is None else max_instances - total_fds
    rows = [
        {
            "check": "sysctl_limits_read",
            "status": "pass" if max_watches is not None and max_instances is not None else "fail",
            "evidence": "/proc/sys/fs/inotify",
            "detail": f"max_user_watches={max_watches}, max_user_instances={max_instances}",
        },
        {
            "check": "live_inotify_usage_counted",
            "status": "pass",
            "evidence": "/proc/*/fdinfo",
            "detail": f"user={username}, processes={len(processes)}, fds={total_fds}, watches={total_watches}",
        },
        {
            "check": "live_usage_below_current_limits",
            "status": "pass" if (watch_headroom is None or watch_headroom >= 0) and (instance_headroom is None or instance_headroom >= 0) else "review",
            "evidence": "/proc/*/fdinfo",
            "detail": f"watch_headroom={watch_headroom}, instance_headroom={instance_headroom}",
        },
        {
            "check": "current_limits_below_kit_target",
            "status": "blocked" if (max_watches or 0) < WATCH_TARGET or (max_instances or 0) < INSTANCE_TARGET else "clear",
            "evidence": "/proc/sys/fs/inotify",
            "detail": f"current={max_watches}/{max_instances}, target={WATCH_TARGET}/{INSTANCE_TARGET}",
        },
    ]
    failed = [row for row in rows if row["status"] == "fail"]
    summary: dict[str, Any] = {
        "status": "ok" if not failed else "failed",
        "experiment_type": "inotify_live_usage_audit",
        "scope": "read-only live inotify fd/watch usage audit for the current user",
        "user": {"uid": uid, "name": username},
        "limits": {
            "max_user_watches": max_watches,
            "max_user_instances": max_instances,
            "watch_target": WATCH_TARGET,
            "instance_target": INSTANCE_TARGET,
            "sysctl_return_code": sysctl_rc,
            "sysctl_output": sysctl_out,
        },
        "metrics": {
            "process_count_with_inotify": len(processes),
            "total_inotify_fd_count": total_fds,
            "total_inotify_watch_count": total_watches,
            "watch_headroom": watch_headroom,
            "instance_headroom": instance_headroom,
            "max_watch_process": top_processes[0] if top_processes else None,
        },
        "top_processes": top_processes,
        "rows": rows,
        "failed_checks": failed,
        "checks": {
            "sysctl_limits_read": max_watches is not None and max_instances is not None,
            "process_fdinfo_scanned": True,
            "records_top_processes": True,
            "does_not_launch_kit_or_training": True,
            "does_not_claim_tracking_reproduction_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This audit only quantifies current live inotify consumption. It does not launch IsaacLab/Kit, "
                "does not alter sysctl limits, and cannot prove official PPO training success."
            ),
        },
        "outputs": {
            "json": str(OUT / "inotify_live_usage_audit.json"),
            "tsv": str(OUT / "inotify_live_usage_audit.tsv"),
            "process_tsv": str(OUT / "inotify_live_processes.tsv"),
        },
    }

    (OUT / "inotify_live_usage_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT / "inotify_live_usage_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["check", "status", "evidence", "detail"])
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "inotify_live_processes.tsv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["pid", "inotify_fd_count", "inotify_watch_count", "command"]
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_processes:
            writer.writerow({key: row[key] for key in fieldnames})
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": summary["outputs"]["json"],
                "processes": len(processes),
                "fds": total_fds,
                "watches": total_watches,
                "watch_headroom": watch_headroom,
                "instance_headroom": instance_headroom,
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
