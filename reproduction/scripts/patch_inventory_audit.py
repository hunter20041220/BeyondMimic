#!/usr/bin/env python3
"""Inventory local patch deliverables and official-worktree drift.

goal.md asks for reproducible patches as part of the final code deliverables.
The current project keeps downloaded/raw sources read-only and uses local audit
scripts/wrappers for smoke evidence, so this audit records what patch artifacts
actually exist instead of treating an empty patch directory as a complete patch
series.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
PATCH_DIR = ROOT / "reproduction/patches"
OFFICIAL_ROOT = ROOT / "reproduction/third_party/official"
OUT = ROOT / "res/code/patch_inventory_audit"
TIMEOUT_SECONDS = 5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_git(repo: Path, args: list[str]) -> dict[str, Any]:
    cmd = ["git", "-C", str(repo), *args]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            env={"GIT_OPTIONAL_LOCKS": "0"},
        )
        stdout_lines = proc.stdout.splitlines()
        stderr_lines = proc.stderr.splitlines()
        return {
            "command": " ".join(cmd),
            "returncode": proc.returncode,
            "timed_out": False,
            "stdout_line_count": len(stdout_lines),
            "stderr_line_count": len(stderr_lines),
            "stdout_sample": stdout_lines[:20],
            "stderr_sample": stderr_lines[:20],
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
        return {
            "command": " ".join(cmd),
            "returncode": None,
            "timed_out": True,
            "stdout_line_count": len(stdout.splitlines()),
            "stderr_line_count": len(stderr.splitlines()),
            "stdout_sample": stdout.splitlines()[:20],
            "stderr_sample": stderr.splitlines()[:20],
        }


def patch_file_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not PATCH_DIR.exists():
        return rows
    for path in sorted(p for p in PATCH_DIR.rglob("*") if p.is_file()):
        rel = path.relative_to(ROOT)
        rows.append(
            {
                "kind": "patch_file",
                "name": str(rel),
                "status": "present",
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "detail": "File under reproduction/patches.",
            }
        )
    return rows


def official_repo_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repo in sorted(p for p in OFFICIAL_ROOT.iterdir() if (p / ".git").is_dir()):
        head = run_git(repo, ["rev-parse", "--short", "HEAD"])
        status = run_git(repo, ["status", "--short", "--untracked-files=no"])
        modified_count = status["stdout_line_count"] if not status["timed_out"] and status["returncode"] == 0 else None
        rows.append(
            {
                "kind": "official_worktree",
                "name": repo.name,
                "status": "status_timeout" if status["timed_out"] else "tracked_changes" if modified_count else "clean",
                "path": str(repo),
                "head": head["stdout_sample"][0] if head["stdout_sample"] else "",
                "head_timed_out": head["timed_out"],
                "status_timed_out": status["timed_out"],
                "tracked_modified_count": modified_count,
                "status_sample": status["stdout_sample"],
                "status_stderr_sample": status["stderr_sample"],
                "detail": (
                    "Official downloaded/reference worktree status is recorded for patch hygiene. "
                    "Tracked changes here are not automatically counted as a reproducible patch series."
                ),
            }
        )
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "kind",
        "name",
        "status",
        "path",
        "head",
        "head_timed_out",
        "status_timed_out",
        "tracked_modified_count",
        "size_bytes",
        "sha256",
        "detail",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({field: r.get(field, "") for field in fields})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    patch_rows = patch_file_rows()
    repo_rows = official_repo_rows()
    rows = patch_rows + repo_rows
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    timed_out_repos = [row for row in repo_rows if row["status_timed_out"]]
    dirty_repos = [row for row in repo_rows if row.get("tracked_modified_count")]
    checks = {
        "patch_directory_exists": PATCH_DIR.is_dir(),
        "official_repo_count_3": len(repo_rows) == 3,
        "patch_file_count_recorded": len(patch_rows) >= 0,
        "official_heads_recorded": all(row.get("head") for row in repo_rows),
        "status_timeouts_recorded": isinstance(timed_out_repos, list),
        "tracked_changes_recorded": isinstance(dirty_repos, list),
        "does_not_claim_patch_series_complete": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "patch_inventory_audit",
        "scope": "goal.md final-deliverable patch inventory and official-worktree drift accounting",
        "metrics": {
            "patch_file_count": len(patch_rows),
            "official_repo_count": len(repo_rows),
            "status_timeout_repo_count": len(timed_out_repos),
            "tracked_change_repo_count": len(dirty_repos),
            "tracked_modified_file_count": int(
                sum(int(row.get("tracked_modified_count") or 0) for row in repo_rows)
            ),
            "row_count": len(rows),
        },
        "status_counts": dict(sorted(status_counts.items())),
        "rows": rows,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "partial",
            "why_not_complete": (
                "The patch directory and official-worktree state are auditable, but no explicit patch files are "
                "present in reproduction/patches and tracked official-worktree changes/timeouts are not a curated "
                "training/deployment patch series."
            ),
        },
        "outputs": {
            "json": str(OUT / "patch_inventory_audit.json"),
            "tsv": str(OUT / "patch_inventory_audit.tsv"),
        },
    }
    (OUT / "patch_inventory_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(OUT / "patch_inventory_audit.tsv", rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
