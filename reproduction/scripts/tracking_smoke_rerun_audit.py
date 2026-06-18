#!/usr/bin/env python3
"""Audit the latest safe Level-B tracking smoke rerun and Kit retry conditions."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/tracking_smoke_rerun_audit"
NONKIT_LOG = ROOT / "logs/setup/whole_body_tracking_nokit_smoke_rerun.log"
KIT_RETRY_LOG = ROOT / "logs/setup/isaaclab_headless_smoke_retry.log"


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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nonkit = read_text(NONKIT_LOG)
    kit_retry = read_text(KIT_RETRY_LOG)
    watches = read_int("/proc/sys/fs/inotify/max_user_watches")
    instances = read_int("/proc/sys/fs/inotify/max_user_instances")
    ulimit_rc, ulimit_out = command_output(["bash", "-lc", "ulimit -n"])
    sysctl_rc, sysctl_out = command_output(["sysctl", "fs.inotify.max_user_watches", "fs.inotify.max_user_instances"])

    rows = [
        {
            "gate": "whole_body_tracking_nonkit_package_asset_smoke",
            "status": "passed" if "non-kit package and asset smoke: OK" in nonkit else "failed_or_missing",
            "evidence": str(NONKIT_LOG),
            "detail": "Official package import, G1 config files, URDF, and first mesh reference checked without launching Kit.",
        },
        {
            "gate": "isaaclab_kit_headless_retry_condition",
            "status": "not_safe_to_retry" if (watches or 0) < 524288 or (instances or 0) < 1024 else "retry_possible",
            "evidence": str(KIT_RETRY_LOG),
            "detail": (
                f"Current inotify max_user_watches={watches}, max_user_instances={instances}; "
                "target retry threshold is >=524288 and >=1024."
            ),
        },
        {
            "gate": "previous_kit_failure_evidence",
            "status": "blocked_evidence_present" if "errno=28" in kit_retry else "missing_or_needs_review",
            "evidence": str(KIT_RETRY_LOG),
            "detail": f"Previous Kit retry log errno=28 count={kit_retry.count('errno=28')}.",
        },
    ]
    failed = [row for row in rows if row["status"] in {"failed_or_missing", "missing_or_needs_review"}]
    summary: dict[str, Any] = {
        "status": "ok" if not failed else "failed",
        "experiment_type": "tracking_smoke_rerun_audit",
        "scope": "safe rerun evidence for Level B non-Kit tracking smoke and current Kit retry gate",
        "row_count": len(rows),
        "failed_row_count": len(failed),
        "rows": rows,
        "metrics": {
            "inotify_max_user_watches": watches,
            "inotify_max_user_instances": instances,
            "ulimit_n_return_code": ulimit_rc,
            "ulimit_n": ulimit_out,
            "sysctl_return_code": sysctl_rc,
            "sysctl_output": sysctl_out,
            "nonkit_log_size_bytes": NONKIT_LOG.stat().st_size if NONKIT_LOG.exists() else 0,
            "kit_retry_errno28_count": kit_retry.count("errno=28"),
            "kit_retry_change_watch_failure_count": kit_retry.count("Failed to create change watch"),
        },
        "checks": {
            "nonkit_rerun_log_exists": NONKIT_LOG.is_file() and NONKIT_LOG.stat().st_size > 0,
            "nonkit_rerun_passed": "non-kit package and asset smoke: OK" in nonkit,
            "g1_urdf_checked": "assets/unitree_description/urdf/g1/main.urdf" in nonkit,
            "g1_mesh_checked": "meshes/g1/pelvis.STL" in nonkit,
            "kit_retry_log_records_errno28": "errno=28" in kit_retry,
            "current_inotify_below_retry_threshold": (watches or 0) < 524288 or (instances or 0) < 1024,
            "does_not_launch_kit_or_training": True,
            "does_not_claim_tracking_reproduction_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The safe non-Kit package/asset smoke rerun passes, but current inotify limits remain below the "
                "documented retry threshold. Full IsaacLab/Kit preprocessing, replay, PPO smoke, and rollout gates "
                "remain blocked."
            ),
        },
        "outputs": {
            "json": str(OUT / "tracking_smoke_rerun_audit.json"),
            "tsv": str(OUT / "tracking_smoke_rerun_audit.tsv"),
        },
    }
    (OUT / "tracking_smoke_rerun_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (OUT / "tracking_smoke_rerun_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["gate", "status", "evidence", "detail"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
