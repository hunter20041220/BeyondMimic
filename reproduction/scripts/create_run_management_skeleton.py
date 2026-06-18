#!/usr/bin/env python3
"""Create and audit a goal.md-compliant run directory skeleton.

This is a diagnostic run-management artifact, not a completed training run.
It proves the required per-run file layout/status schema exists before long
Isaac/VAE/diffusion jobs are allowed to claim SUCCESS.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
RUN_ID = "setup_run_management_diagnostic_static_000_20260617_050000"
RUN_DIR = ROOT / "res/runs" / RUN_ID
OUT = ROOT / "res/run_management_audit"

REQUIRED_FILES = [
    "resolved_config.yaml",
    "command.sh",
    "stdout.log",
    "stderr.log",
    "environment.txt",
    "git_state.txt",
    "gpu_metrics.csv",
    "metrics.json",
    "metrics.csv",
    "status.json",
]
REQUIRED_DIRS = ["checkpoint", "figures", "videos"]


def read_command(cmd: list[str], timeout: int = 20) -> str:
    try:
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=False)
        return (proc.stdout + proc.stderr).strip()
    except Exception as exc:  # noqa: BLE001 - diagnostic should keep going.
        return f"command_failed: {cmd}: {exc}"


def write_text(path: Path, text: str, executable: bool = False) -> None:
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def make_run_dir() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    for dirname in REQUIRED_DIRS:
        (RUN_DIR / dirname).mkdir(exist_ok=True)

    write_text(
        RUN_DIR / "resolved_config.yaml",
        "\n".join(
            [
                "run_id: setup_run_management_diagnostic_static_000_20260617_050000",
                "stage: setup",
                "method: run_management_diagnostic",
                "motion: static",
                "config: goal_schema",
                "seed: 0",
                "status: INVALID",
                "is_training_run: false",
                "reason: Schema-only diagnostic run directory; no training or evaluation was executed.",
                "source_goal_lines: goal.md:1747-1787",
                "",
            ]
        ),
    )
    write_text(
        RUN_DIR / "command.sh",
        "#!/usr/bin/env bash\nset -euo pipefail\npython3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/create_run_management_skeleton.py\n",
        executable=True,
    )
    write_text(
        RUN_DIR / "stdout.log",
        "Diagnostic run-management skeleton generated. No training command was executed.\n",
    )
    write_text(RUN_DIR / "stderr.log", "")

    env_lines = [
        f"timestamp={datetime.now().isoformat(timespec='seconds')}",
        f"python={read_command(['python3', '--version'])}",
        f"bm_analysis={ROOT / 'envs/bm_analysis'}",
        f"bm_tracking={ROOT / 'envs/bm_tracking'}",
        f"bm_diffusion={ROOT / 'envs/bm_diffusion'}",
        "notes=Diagnostic run; environment lock files remain in env prefixes.",
        "",
    ]
    write_text(RUN_DIR / "environment.txt", "\n".join(env_lines))

    git_state = read_command(["git", "status", "--short"])
    if "not a git repository" in git_state:
        git_state = "not_a_git_repository\n" + git_state
    write_text(RUN_DIR / "git_state.txt", git_state + "\n")

    gpu_src = ROOT / "logs/gpu/gpu_metrics.csv"
    if gpu_src.is_file():
        shutil.copyfile(gpu_src, RUN_DIR / "gpu_metrics.csv")
    else:
        write_text(
            RUN_DIR / "gpu_metrics.csv",
            "timestamp,gpu_index,gpu_uuid,gpu_name,memory_used_mib,memory_total_mib,memory_free_mib,gpu_util_percent,power_draw_w,temperature_c,process_pid,run_id,run_status,sample_kind\n",
        )

    metrics = {
        "run_id": RUN_ID,
        "status": "INVALID",
        "is_training_run": False,
        "samples_per_second": None,
        "environment_steps_per_second": None,
        "iteration_time": None,
        "estimated_remaining_time": None,
        "oom_count": 0,
        "restart_count": 0,
        "why_invalid": "Schema-only diagnostic run directory; no training endpoint or evaluation was reached.",
    }
    write_text(RUN_DIR / "metrics.json", json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    with (RUN_DIR / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)

    status = {
        "run_id": RUN_ID,
        "status": "INVALID",
        "allowed_status": True,
        "is_success": False,
        "is_training_run": False,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "reason": "Diagnostic schema run only. goal.md allows SUCCESS only after training endpoint and evaluation.",
    }
    write_text(RUN_DIR / "status.json", json.dumps(status, indent=2, sort_keys=True) + "\n")


def audit() -> dict[str, Any]:
    file_status = {
        rel: {
            "exists": (RUN_DIR / rel).is_file(),
            "size": (RUN_DIR / rel).stat().st_size if (RUN_DIR / rel).is_file() else 0,
        }
        for rel in REQUIRED_FILES
    }
    dir_status = {rel: (RUN_DIR / rel).is_dir() for rel in REQUIRED_DIRS}
    status_json = json.loads((RUN_DIR / "status.json").read_text(encoding="utf-8"))
    metrics_json = json.loads((RUN_DIR / "metrics.json").read_text(encoding="utf-8"))
    gpu_rows = 0
    with (RUN_DIR / "gpu_metrics.csv").open("r", encoding="utf-8", newline="") as f:
        gpu_rows = max(0, sum(1 for _ in csv.DictReader(f)))
    checks = {
        "run_id_matches_goal_pattern": RUN_ID.count("_") >= 6 and RUN_ID.endswith("20260617_050000"),
        "all_required_files_exist": all(item["exists"] and item["size"] >= 0 for item in file_status.values()),
        "all_required_dirs_exist": all(dir_status.values()),
        "status_is_allowed": status_json["status"] in {
            "QUEUED",
            "RUNNING",
            "SUCCESS",
            "FAILED",
            "FAILED_OOM",
            "INTERRUPTED",
            "INVALID",
        },
        "does_not_mark_success_without_training": status_json["status"] != "SUCCESS" and not status_json["is_success"],
        "metrics_json_has_required_runtime_fields": all(
            key in metrics_json
            for key in [
                "samples_per_second",
                "environment_steps_per_second",
                "iteration_time",
                "estimated_remaining_time",
                "oom_count",
                "restart_count",
            ]
        ),
        "gpu_metrics_present": gpu_rows > 0,
        "checkpoint_figures_videos_dirs_exist": all(dir_status.values()),
    }
    return {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "run_management_audit",
        "scope": "goal.md per-run directory schema diagnostic",
        "run_id": RUN_ID,
        "run_dir": str(RUN_DIR),
        "file_status": file_status,
        "dir_status": dir_status,
        "gpu_metric_rows": gpu_rows,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The run-management directory schema exists, but this diagnostic run is INVALID and does not "
                "replace full training runs with checkpoints, figures, videos, and evaluation metrics."
            ),
        },
        "outputs": {
            "json": str(OUT / "run_management_audit.json"),
            "tsv": str(OUT / "run_management_audit.tsv"),
        },
    }


def write_audit(summary: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run_management_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT / "run_management_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["kind", "name", "exists_or_passed", "detail"])
        writer.writeheader()
        for rel, item in summary["file_status"].items():
            writer.writerow({"kind": "file", "name": rel, "exists_or_passed": item["exists"], "detail": item["size"]})
        for rel, ok in summary["dir_status"].items():
            writer.writerow({"kind": "dir", "name": rel, "exists_or_passed": ok, "detail": ""})
        for name, ok in summary["checks"].items():
            writer.writerow({"kind": "check", "name": name, "exists_or_passed": ok, "detail": ""})


def main() -> None:
    make_run_dir()
    summary = audit()
    write_audit(summary)
    print(json.dumps({"status": summary["status"], "run_dir": str(RUN_DIR), "json": summary["outputs"]["json"]}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
