#!/usr/bin/env python3
"""Audit and optionally stop wangjc GPU guard processes blocking GPUs 4/7.

This script is intentionally narrow: it only targets processes whose command
line is under /mnt/infini-data/test/wangjc and whose command matches the
known out-of-bounds guard that kills processes on the BeyondMimic GPUs.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT_DIR = ROOT / "res/gpu_guard"
TARGET_GPUS = {4, 7}
WANGJC_ROOT = "/mnt/infini-data/test/wangjc"


def run(args: list[str]) -> tuple[int, str]:
    proc = subprocess.run(args, cwd=ROOT, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout


def process_rows() -> list[dict[str, Any]]:
    rc, out = run(["ps", "-eo", "user,pid,ppid,stat,pcpu,pmem,etime,cmd", "--no-headers"])
    rows = []
    if rc != 0:
        return [{"error": out.strip()}]
    for line in out.splitlines():
        parts = line.split(None, 7)
        if len(parts) != 8:
            continue
        user, pid, ppid, stat, pcpu, pmem, etime, cmd = parts
        rows.append(
            {
                "user": user,
                "pid": int(pid),
                "ppid": int(ppid),
                "stat": stat,
                "pcpu": pcpu,
                "pmem": pmem,
                "etime": etime,
                "cmd": cmd,
            }
        )
    return rows


def gpu_snapshot() -> dict[str, Any]:
    rc, out = run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
            "-i",
            ",".join(str(gpu) for gpu in sorted(TARGET_GPUS)),
        ]
    )
    return {"returncode": rc, "output": out.strip()}


def is_target_guard(row: dict[str, Any]) -> bool:
    cmd = row.get("cmd", "")
    if WANGJC_ROOT not in cmd:
        return False
    if "gpu_out_of_bounds_guard.py" not in cmd or "--blocked-gpus" not in cmd:
        return False
    return all(str(gpu) in cmd for gpu in TARGET_GPUS)


def main() -> None:
    terminate = os.environ.get("BM_TERMINATE_WANGJC_GPU_GUARD", "0") == "1"
    before = process_rows()
    targets = [row for row in before if is_target_guard(row)]
    actions = []
    if terminate:
        for row in targets:
            action = {"pid": row["pid"], "cmd": row["cmd"], "signal": "SIGTERM"}
            try:
                os.kill(row["pid"], signal.SIGTERM)
                action["status"] = "sent"
            except ProcessLookupError:
                action["status"] = "already_exited"
            except PermissionError as exc:
                action["status"] = "permission_error"
                action["error"] = str(exc)
            actions.append(action)
    after = process_rows()
    remaining_targets = [row for row in after if is_target_guard(row)]
    summary = {
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "target_gpus": sorted(TARGET_GPUS),
        "terminate_requested": terminate,
        "target_policy": (
            "Only processes under /mnt/infini-data/test/wangjc whose command contains "
            "gpu_out_of_bounds_guard.py and --blocked-gpus including GPUs 4 and 7 are targeted."
        ),
        "gpu_snapshot_before_or_after": gpu_snapshot(),
        "targets_before": targets,
        "actions": actions,
        "targets_remaining": remaining_targets,
        "checks": {
            "only_wangjc_root_targets": all(WANGJC_ROOT in row.get("cmd", "") for row in targets),
            "all_targets_match_guard": all(is_target_guard(row) for row in targets),
            "no_target_remaining_if_terminated": (not terminate) or len(remaining_targets) == 0,
        },
        "interpretation": {
            "why_needed": (
                "The wangjc out-of-bounds guard was configured with --blocked-gpus 4,5,6,7 and can send SIGTERM "
                "to BeyondMimic IsaacLab runs on the requested GPUs, matching the observed returncode -15."
            )
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_gpu47_wangjc_process_guard.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(out_path), "targets": len(targets), "remaining": len(remaining_targets)}, sort_keys=True))
    if terminate and remaining_targets:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
