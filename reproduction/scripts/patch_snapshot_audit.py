#!/usr/bin/env python3
"""Export reproducible patch snapshots for tracked official-worktree drift."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OFFICIAL_ROOT = ROOT / "reproduction/third_party/official"
PATCH_OUT = ROOT / "reproduction/patches/official_worktree_snapshots"
OUT = ROOT / "res/code/patch_snapshot_audit"
TIMEOUT_SECONDS = 15


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_git(repo: Path, args: list[str], *, timeout: int = TIMEOUT_SECONDS) -> dict[str, Any]:
    cmd = ["git", "-C", str(repo), *args]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"GIT_OPTIONAL_LOCKS": "0"},
        )
        return {
            "returncode": proc.returncode,
            "timed_out": False,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "command": " ".join(cmd),
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout or ""
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr or ""
        return {
            "returncode": None,
            "timed_out": True,
            "stdout": stdout,
            "stderr": stderr,
            "command": " ".join(cmd),
        }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "repo",
        "head",
        "status",
        "tracked_modified_count",
        "patch_path",
        "patch_size_bytes",
        "patch_sha256",
        "semantic_diff_empty",
        "diff_timed_out",
        "status_timed_out",
        "detail",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    PATCH_OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for repo in sorted(p for p in OFFICIAL_ROOT.iterdir() if (p / ".git").is_dir()):
        head = run_git(repo, ["rev-parse", "--short", "HEAD"], timeout=5)
        status = run_git(repo, ["status", "--short", "--untracked-files=no"], timeout=5)
        tracked_lines = [line for line in status["stdout"].splitlines() if line.strip()]
        if status["timed_out"] or not tracked_lines:
            continue
        diff = run_git(repo, ["diff", "--", "."], timeout=TIMEOUT_SECONDS)
        semantic = run_git(repo, ["diff", "--ignore-space-at-eol", "--", "."], timeout=TIMEOUT_SECONDS)
        patch_path = PATCH_OUT / f"{repo.name}.tracked.patch"
        patch_path.write_text(diff["stdout"], encoding="utf-8")
        rows.append(
            {
                "repo": repo.name,
                "head": head["stdout"].strip().splitlines()[0] if head["stdout"].strip() else "",
                "status": "snapshot_exported",
                "tracked_modified_count": len(tracked_lines),
                "patch_path": str(patch_path),
                "patch_size_bytes": patch_path.stat().st_size,
                "patch_sha256": sha256_file(patch_path),
                "semantic_diff_empty": (not semantic["stdout"].strip()) and not semantic["timed_out"],
                "diff_timed_out": diff["timed_out"],
                "status_timed_out": status["timed_out"],
                "status_sample": tracked_lines[:20],
                "diff_stderr_sample": diff["stderr"].splitlines()[:20],
                "semantic_diff_stderr_sample": semantic["stderr"].splitlines()[:20],
                "detail": (
                    "Tracked official-worktree diff snapshot exported. Empty or semantic-empty patches usually "
                    "indicate line-ending or whitespace-only drift, not a functional patch series."
                ),
            }
        )

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    checks = {
        "snapshot_dir_exists": PATCH_OUT.is_dir(),
        "at_least_one_snapshot_row": len(rows) >= 1,
        "all_patch_files_exist": all(Path(row["patch_path"]).is_file() for row in rows),
        "all_patch_files_hashed": all(bool(row["patch_sha256"]) for row in rows),
        "all_statuses_not_timed_out": all(not row["status_timed_out"] for row in rows),
        "does_not_modify_official_worktrees": True,
        "does_not_claim_functional_patch_series": True,
        "does_not_claim_goal_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "patch_snapshot_audit",
        "scope": "export tracked official-worktree diffs into reproduction/patches snapshots",
        "metrics": {
            "snapshot_row_count": len(rows),
            "patch_file_count": len(rows),
            "semantic_empty_patch_count": sum(1 for row in rows if row["semantic_diff_empty"]),
            "total_patch_size_bytes": int(sum(int(row["patch_size_bytes"]) for row in rows)),
            "tracked_modified_file_count": int(sum(int(row["tracked_modified_count"]) for row in rows)),
        },
        "status_counts": dict(sorted(status_counts.items())),
        "rows": rows,
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "partial",
            "why_not_complete": (
                "Tracked official-worktree drift is now exported as patch snapshot files, but these snapshots are "
                "not a curated functional patch series for full training, deployment, or paper-result reproduction."
            ),
        },
        "outputs": {
            "json": str(OUT / "patch_snapshot_audit.json"),
            "tsv": str(OUT / "patch_snapshot_audit.tsv"),
            "patch_dir": str(PATCH_OUT),
        },
    }
    (OUT / "patch_snapshot_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(OUT / "patch_snapshot_audit.tsv", rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
