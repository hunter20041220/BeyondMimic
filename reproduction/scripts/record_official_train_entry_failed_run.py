#!/usr/bin/env python3
"""Record the bounded official tracking train-entry retry as a failed run."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
RUN_ID = "phase1_official_train_entry_retry_inotify_0_20260617_174742"
FAILED_DIR = ROOT / "res/failed_runs" / RUN_ID
OUT = ROOT / "res/failed_runs/official_train_entry_failed_run_audit"
RETRY_JSON = ROOT / "res/tracking/official_train_entry_retry_audit/tracking_official_train_entry_retry_audit.json"

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


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def tail(path: Path, lines: int = 120) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]) + "\n"


def load_retry() -> dict[str, Any]:
    return json.loads(RETRY_JSON.read_text(encoding="utf-8"))


def make_failed_run() -> None:
    retry = load_retry()
    FAILED_DIR.mkdir(parents=True, exist_ok=True)
    log_path = Path(retry["outputs"]["log"])
    gpu_path = ROOT / "logs/gpu/gpu_metrics.csv"
    write(FAILED_DIR / "run_id.txt", RUN_ID + "\n")
    write(
        FAILED_DIR / "error.txt",
        "\n".join(
            [
                "Bounded official whole_body_tracking train-entry retry did not reach a PPO training endpoint.",
                f"Retry result: {retry['classification']['retry_result']}.",
                f"Return code: {retry['returncode']} under {retry['timeout_seconds']} second timeout.",
                "Observed Kit change-watch failures with errno=28 / No space left on device.",
                "",
            ]
        ),
    )
    config = {
        "run_id": RUN_ID,
        "stage": "phase1",
        "method": "official_whole_body_tracking_train_entry_retry",
        "task": "Tracking-Flat-G1-v0",
        "num_envs": 1,
        "max_iterations": 1,
        "device": "cpu",
        "timeout_seconds": retry["timeout_seconds"],
        "command": retry["command"],
        "cwd": retry["cwd"],
        "retry_json": str(RETRY_JSON),
        "command_log": retry["outputs"]["log"],
        "tail_log": retry["outputs"]["tail_log"],
        "classification": retry["classification"],
        "sysctl": retry["sysctl"],
        "blocked_gate_id": "isaaclab_kit_inotify",
    }
    write(FAILED_DIR / "config.json", json.dumps(config, indent=2, sort_keys=True) + "\n")
    write(
        FAILED_DIR / "checkpoint.txt",
        "No checkpoint was created. The bounded official train-entry retry timed out before PPO training success.\n",
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
                "- The bounded official `whole_body_tracking` train-entry retry reproduced Kit watcher failures.",
                f"- Classification: `{retry['classification']['retry_result']}`.",
                f"- Current inotify limits: `{retry['sysctl']['output']}`.",
                "- The retry did not reach a PPO training endpoint and produced no checkpoint, rollout, or video.",
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
                "- Ask an administrator to raise `fs.inotify.max_user_watches` to at least `524288`.",
                "- Ask an administrator to raise `fs.inotify.max_user_instances` to at least `1024`.",
                "- Rerun `tracking_official_train_entry_retry_audit.py` after the host limits change.",
                "- If the retry reaches the training endpoint, only then schedule a longer official PPO smoke.",
                "- Keep this failed run under `res/failed_runs`; do not delete or hide it.",
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
        "is_training_run": False,
        "kept_for_audit": True,
    }
    write(FAILED_DIR / "status.json", json.dumps(status, indent=2, sort_keys=True) + "\n")


def audit() -> dict[str, Any]:
    retry = load_retry()
    files = {
        rel: {
            "exists": (FAILED_DIR / rel).is_file(),
            "size": (FAILED_DIR / rel).stat().st_size if (FAILED_DIR / rel).is_file() else 0,
        }
        for rel in REQUIRED
    }
    status = json.loads((FAILED_DIR / "status.json").read_text(encoding="utf-8"))
    config = json.loads((FAILED_DIR / "config.json").read_text(encoding="utf-8"))
    log_text = (FAILED_DIR / "last_log.txt").read_text(encoding="utf-8", errors="replace")
    with (FAILED_DIR / "gpu_status.csv").open("r", encoding="utf-8", newline="") as f:
        gpu_rows = max(0, sum(1 for _ in csv.DictReader(f)))
    checks = {
        "all_required_failed_run_files_exist": all(item["exists"] and item["size"] >= 0 for item in files.values()),
        "status_failed_not_success": status["status"] == "FAILED" and not status["is_success"],
        "run_id_recorded": config["run_id"] == RUN_ID,
        "retry_json_exists": RETRY_JSON.is_file(),
        "retry_classified_blocked_inotify": retry["classification"]["retry_result"] == "blocked_inotify",
        "config_records_command": bool(config["command"]) and config["task"] == "Tracking-Flat-G1-v0",
        "checkpoint_absence_recorded": "No checkpoint" in (FAILED_DIR / "checkpoint.txt").read_text(encoding="utf-8"),
        "last_log_records_inotify_failure": "errno=28" in log_text or "No space left on device" in log_text,
        "gpu_status_recorded": gpu_rows > 0,
        "failure_reason_recorded": "isaaclab_kit_inotify" in (FAILED_DIR / "failure_reason.md").read_text(encoding="utf-8"),
        "resolution_plan_recorded": "max_user_watches" in (FAILED_DIR / "resolution_plan.md").read_text(encoding="utf-8"),
        "does_not_claim_training_success": True,
    }
    return {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "official_train_entry_failed_run_audit",
        "scope": "goal.md failed-run retention record for the bounded official train-entry retry",
        "run_id": RUN_ID,
        "failed_run_dir": str(FAILED_DIR),
        "required_files": files,
        "gpu_status_rows": gpu_rows,
        "source_retry_json": str(RETRY_JSON),
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This records the bounded official train-entry retry failure. It preserves evidence but does not "
                "resolve the inotify blocker or complete official PPO training/evaluation."
            ),
        },
        "outputs": {
            "json": str(OUT / "official_train_entry_failed_run_audit.json"),
            "tsv": str(OUT / "official_train_entry_failed_run_audit.tsv"),
        },
    }


def write_audit(summary: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "official_train_entry_failed_run_audit.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(OUT / "official_train_entry_failed_run_audit.json")
    tsv_tmp = OUT / "official_train_entry_failed_run_audit.tsv.tmp"
    with tsv_tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["kind", "name", "passed_or_exists", "detail"])
        writer.writeheader()
        for rel, item in summary["required_files"].items():
            writer.writerow({"kind": "file", "name": rel, "passed_or_exists": item["exists"], "detail": item["size"]})
        for name, ok in summary["checks"].items():
            writer.writerow({"kind": "check", "name": name, "passed_or_exists": ok, "detail": ""})
    tsv_tmp.replace(OUT / "official_train_entry_failed_run_audit.tsv")


def main() -> None:
    make_failed_run()
    summary = audit()
    write_audit(summary)
    print(json.dumps({"status": summary["status"], "failed_run_dir": str(FAILED_DIR), "json": summary["outputs"]["json"]}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
