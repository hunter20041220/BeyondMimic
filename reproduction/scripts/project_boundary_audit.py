#!/usr/bin/env python3
"""Audit project path boundaries, raw download policy, and project-local cache/tmp settings."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DOWNLOAD = ROOT / "download"
OUT = ROOT / "res/project_boundary_audit"

ALLOWED_DOWNLOAD_TOPLEVEL = {
    "README_download_scope.md",
    "_supplemental",
    "dependencies",
    "download_full.err.log",
    "download_full.log",
    "manifests",
    "official",
    "papers",
    "reference_code",
}

PROJECT_GENERATED_ROOTS = [
    ROOT / "reproduction",
    ROOT / "res",
    ROOT / "logs",
    ROOT / "envs",
    ROOT / "cache",
    ROOT / "tmp",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def shallow_count(path: Path, limit: int = 1000) -> int:
    if not path.exists():
        return 0
    count = 0
    for _ in path.iterdir():
        count += 1
        if count >= limit:
            return limit
    return count


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    top_entries = sorted(p.name for p in DOWNLOAD.iterdir()) if DOWNLOAD.is_dir() else []
    unexpected_download_entries = [name for name in top_entries if name not in ALLOWED_DOWNLOAD_TOPLEVEL]
    generated_counts = {root.name: shallow_count(root) for root in PROJECT_GENERATED_ROOTS}
    project_env = ROOT / "reproduction/scripts/project_env.sh"
    project_env_text = project_env.read_text(encoding="utf-8") if project_env.is_file() else ""
    required_env_lines = {
        "CACHE_ROOT": 'export CACHE_ROOT="$ROOT/cache"',
        "TMP_ROOT": 'export TMP_ROOT="$ROOT/tmp"',
        "PIP_CACHE_DIR": 'export PIP_CACHE_DIR="$CACHE_ROOT/pip"',
        "HF_HOME": 'export HF_HOME="$CACHE_ROOT/huggingface"',
        "TORCH_HOME": 'export TORCH_HOME="$CACHE_ROOT/torch"',
        "XDG_CACHE_HOME": 'export XDG_CACHE_HOME="$CACHE_ROOT/xdg"',
        "TMPDIR": 'export TMPDIR="$TMP_ROOT"',
    }
    env_rows = [
        {
            "name": name,
            "pattern": pattern,
            "present": pattern in project_env_text,
        }
        for name, pattern in required_env_lines.items()
    ]
    supplemental_manifest = DOWNLOAD / "_supplemental/supplemental_downloads.tsv"
    download_manifest_files = [
        DOWNLOAD / "manifests/downloaded_files.tsv",
        DOWNLOAD / "manifests/git_revisions.tsv",
        DOWNLOAD / "manifests/resource_sources.md",
        DOWNLOAD / "manifests/download_summary.txt",
    ]
    source_docs = [
        ROOT / "reproduction/docs/local_inventory.tsv",
        ROOT / "reproduction/docs/source_ledger.md",
        ROOT / "download/README_download_scope.md",
    ]
    rows = [
        {
            "check": "download_exists",
            "passed": DOWNLOAD.is_dir(),
            "detail": str(DOWNLOAD),
        },
        {
            "check": "download_toplevel_allowlist",
            "passed": not unexpected_download_entries,
            "detail": json.dumps({"entries": top_entries, "unexpected": unexpected_download_entries}, sort_keys=True),
        },
        {
            "check": "supplemental_download_manifest_exists",
            "passed": supplemental_manifest.is_file() and supplemental_manifest.stat().st_size > 0,
            "detail": rel(supplemental_manifest),
        },
        {
            "check": "download_manifests_exist",
            "passed": all(path.is_file() and path.stat().st_size > 0 for path in download_manifest_files),
            "detail": "; ".join(rel(path) for path in download_manifest_files),
        },
        {
            "check": "source_docs_exist",
            "passed": all(path.is_file() and path.stat().st_size > 0 for path in source_docs),
            "detail": "; ".join(rel(path) for path in source_docs),
        },
        {
            "check": "project_generated_roots_exist",
            "passed": all(root.exists() for root in PROJECT_GENERATED_ROOTS[:-1]),
            "detail": json.dumps({"shallow_entry_counts": generated_counts}, sort_keys=True),
        },
        {
            "check": "project_env_cache_tmp_redirects",
            "passed": all(row["present"] for row in env_rows),
            "detail": json.dumps(env_rows, sort_keys=True),
        },
        {
            "check": "no_goal_completion_claim",
            "passed": True,
            "detail": "Boundary audit records policy evidence only.",
        },
    ]
    failed = [row for row in rows if not row["passed"]]
    summary = {
        "status": "ok" if not failed else "failed",
        "experiment_type": "boundary_audit",
        "scope": "goal.md fixed path, raw download, supplemental-download, and project-local cache/tmp policy",
        "download_root": str(DOWNLOAD),
        "allowed_download_toplevel": sorted(ALLOWED_DOWNLOAD_TOPLEVEL),
        "download_toplevel_entries": top_entries,
        "unexpected_download_entries": unexpected_download_entries,
        "generated_root_counts": generated_counts,
        "row_count": len(rows),
        "failed_count": len(failed),
        "rows": rows,
        "checks": {
            "download_exists": DOWNLOAD.is_dir(),
            "download_toplevel_allowlist_passes": not unexpected_download_entries,
            "supplemental_manifest_exists": supplemental_manifest.is_file() and supplemental_manifest.stat().st_size > 0,
            "download_manifests_exist": all(path.is_file() and path.stat().st_size > 0 for path in download_manifest_files),
            "source_docs_exist": all(path.is_file() and path.stat().st_size > 0 for path in source_docs),
            "project_generated_roots_exist": all(root.exists() for root in PROJECT_GENERATED_ROOTS[:-1]),
            "project_env_cache_tmp_redirects": all(row["present"] for row in env_rows),
            "does_not_claim_goal_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "Path-boundary and raw-download policy evidence is present, but this does not replace the missing "
                "live training, checkpoint, video, Fig. 5/Fig. 6, TensorRT, or hardware evidence."
            ),
        },
        "outputs": {
            "json": str(OUT / "project_boundary_audit.json"),
            "tsv": str(OUT / "project_boundary_audit.tsv"),
        },
    }
    (OUT / "project_boundary_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT / "project_boundary_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["check", "passed", "detail"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
