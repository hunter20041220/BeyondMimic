#!/usr/bin/env python3
"""Record a goal.md-compliant failed run from an observed blocker."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
RUN_ID = "phase1_isaaclab_headless_smoke_g1_inotify_0_20260616_200654"
FAILED_DIR = ROOT / "res/failed_runs" / RUN_ID
OUT = ROOT / "res/failed_runs/failed_run_audit"

REQUIRED = [
    "run_id.txt",
    "error.txt",
    "config.json",
    "checkpoint.txt",
    "last_log.txt",
    "gpu_status.csv",
    "failure_reason.md",
    "resolution_plan.md",
    "status.json",
]


def load_json(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def tail(path: Path, lines: int = 80) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]) + "\n"


def make_failed_run() -> None:
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    blocked = load_json("res/blocked_gates/blocked_gate_audit.json")
    gate = next(item for item in blocked["gates"] if item["gate_id"] == "isaaclab_kit_inotify")
    log_path = Path(gate["evidence"]["retry_log"])
    inotify_path = ROOT / "logs/setup/inotify_status.txt"
    gpu_path = ROOT / "logs/gpu/gpu_metrics.csv"

    write(FAILED_DIR / "run_id.txt", RUN_ID + "\n")
    write(
        FAILED_DIR / "error.txt",
        "\n".join(
            [
                "IsaacLab/Kit headless smoke failed before SimulationApp startup completed.",
                "Observed repeated change-watch creation failures with errno=28 / No space left on device.",
                "Evidence indicates low inotify limits rather than disk exhaustion.",
                "",
            ]
        ),
    )
    config = {
        "run_id": RUN_ID,
        "stage": "phase1",
        "method": "isaaclab_headless_smoke",
        "motion": "g1",
        "config": "inotify",
        "seed": 0,
        "command_log": str(log_path),
        "blocked_gate_id": gate["gate_id"],
        "blocked_gate_status": gate["status"],
        "blocked_tasks": gate["blocks"],
        "sysctl_output": gate["evidence"]["sysctl_output"],
    }
    write(FAILED_DIR / "config.json", json.dumps(config, indent=2, sort_keys=True) + "\n")
    write(
        FAILED_DIR / "checkpoint.txt",
        "No checkpoint was created. Failure occurred during environment/Kit smoke before training startup.\n",
    )
    write(FAILED_DIR / "last_log.txt", tail(log_path))
    if gpu_path.is_file():
        shutil.copyfile(gpu_path, FAILED_DIR / "gpu_status.csv")
    else:
        write(FAILED_DIR / "gpu_status.csv", "timestamp,gpu_index,gpu_name,memory_used_mib\n")
    write(
        FAILED_DIR / "failure_reason.md",
        "\n".join(
            [
                "# Failure Reason",
                "",
                "- Gate: `isaaclab_kit_inotify`.",
                "- Kit reports repeated `Failed to create change watch` errors with `errno=28`.",
                f"- Current inotify limits from evidence: `{gate['evidence']['sysctl_output'].strip()}`.",
                "- This blocks IsaacLab/Kit smoke, motion preprocessing, replay, PPO smoke, and live rollout evaluation.",
                "",
            ]
        ),
    )
    write(
        FAILED_DIR / "resolution_plan.md",
        "\n".join(
            [
                "# Resolution Plan",
                "",
                f"- {gate['next_action']}",
                "- After limits are raised, rerun the IsaacLab/Kit smoke and then tracking preprocessing/replay/PPO smoke.",
                "- Keep this failed run in `res/failed_runs`; do not delete or hide it.",
                "",
            ]
        ),
    )
    status = {
        "run_id": RUN_ID,
        "status": "FAILED",
        "failed_gate": "isaaclab_kit_inotify",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "is_success": False,
        "kept_for_audit": True,
    }
    write(FAILED_DIR / "status.json", json.dumps(status, indent=2, sort_keys=True) + "\n")


def audit() -> dict[str, Any]:
    files = {
        rel: {"exists": (FAILED_DIR / rel).is_file(), "size": (FAILED_DIR / rel).stat().st_size if (FAILED_DIR / rel).is_file() else 0}
        for rel in REQUIRED
    }
    status = json.loads((FAILED_DIR / "status.json").read_text(encoding="utf-8"))
    config = json.loads((FAILED_DIR / "config.json").read_text(encoding="utf-8"))
    log_text = (FAILED_DIR / "last_log.txt").read_text(encoding="utf-8", errors="replace")
    gpu_rows = 0
    with (FAILED_DIR / "gpu_status.csv").open("r", encoding="utf-8", newline="") as f:
        gpu_rows = max(0, sum(1 for _ in csv.DictReader(f)))
    checks = {
        "all_required_failed_run_files_exist": all(item["exists"] and item["size"] >= 0 for item in files.values()),
        "status_failed_not_success": status["status"] == "FAILED" and not status["is_success"],
        "run_id_recorded": config["run_id"] == RUN_ID,
        "error_recorded": (FAILED_DIR / "error.txt").stat().st_size > 0,
        "config_recorded": config["blocked_gate_id"] == "isaaclab_kit_inotify",
        "checkpoint_absence_recorded": "No checkpoint" in (FAILED_DIR / "checkpoint.txt").read_text(encoding="utf-8"),
        "last_log_records_errno28": "errno=28" in log_text or "No space left on device" in log_text,
        "gpu_status_recorded": gpu_rows > 0,
        "failure_reason_recorded": "isaaclab_kit_inotify" in (FAILED_DIR / "failure_reason.md").read_text(encoding="utf-8"),
        "resolution_plan_recorded": "inotify" in (FAILED_DIR / "resolution_plan.md").read_text(encoding="utf-8"),
    }
    return {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "failed_run_audit",
        "scope": "goal.md failed-run retention record",
        "run_id": RUN_ID,
        "failed_run_dir": str(FAILED_DIR),
        "required_files": files,
        "gpu_status_rows": gpu_rows,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This records one real failed/blocked Kit smoke attempt. It preserves evidence but does not resolve "
                "the inotify blocker or complete the blocked training/evaluation phases."
            ),
        },
        "outputs": {
            "json": str(OUT / "failed_run_audit.json"),
            "tsv": str(OUT / "failed_run_audit.tsv"),
        },
    }


def write_audit(summary: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "failed_run_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT / "failed_run_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["kind", "name", "passed_or_exists", "detail"])
        writer.writeheader()
        for rel, item in summary["required_files"].items():
            writer.writerow({"kind": "file", "name": rel, "passed_or_exists": item["exists"], "detail": item["size"]})
        for name, ok in summary["checks"].items():
            writer.writerow({"kind": "check", "name": name, "passed_or_exists": ok, "detail": ""})


def main() -> None:
    make_failed_run()
    summary = audit()
    write_audit(summary)
    print(json.dumps({"status": summary["status"], "failed_run_dir": str(FAILED_DIR), "json": summary["outputs"]["json"]}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
