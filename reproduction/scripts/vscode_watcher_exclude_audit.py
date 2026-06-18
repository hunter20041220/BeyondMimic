#!/usr/bin/env python3
"""Audit project-local VS Code watcher excludes for large reproduction directories."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SETTINGS = ROOT / ".vscode/settings.json"
LIVE_USAGE = ROOT / "res/setup/inotify_live_usage_audit/inotify_live_usage_audit.json"
OUT = ROOT / "res/setup/vscode_watcher_exclude_audit"
REQUIRED_PATTERNS = [
    "**/cache/**",
    "**/download/**",
    "**/envs/**",
    "**/logs/**",
    "**/res/**",
    "**/tmp/**",
    "**/reproduction/data/**",
    "**/reproduction/third_party/**",
    "**/reproduction/generated/**",
    "**/__pycache__/**",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    settings = load_json(SETTINGS) if SETTINGS.is_file() else {}
    live = load_json(LIVE_USAGE) if LIVE_USAGE.is_file() else {}
    watcher = settings.get("files.watcherExclude", {})
    search = settings.get("search.exclude", {})
    rows: list[dict[str, Any]] = []
    for pattern in REQUIRED_PATTERNS:
        rows.append(
            {
                "pattern": pattern,
                "watcher_excluded": bool(watcher.get(pattern) is True),
                "search_excluded": bool(search.get(pattern) is True),
            }
        )
    missing = [row for row in rows if not row["watcher_excluded"] or not row["search_excluded"]]
    watch_headroom = live.get("metrics", {}).get("watch_headroom")
    top_process = live.get("metrics", {}).get("max_watch_process") or {}
    live_still_saturated = watch_headroom == 0
    summary: dict[str, Any] = {
        "status": "ok" if not missing else "failed",
        "experiment_type": "vscode_watcher_exclude_audit",
        "scope": "project-local VS Code watcher/search excludes for large BeyondMimic reproduction directories",
        "settings_path": str(SETTINGS),
        "required_pattern_count": len(REQUIRED_PATTERNS),
        "missing_count": len(missing),
        "rows": rows,
        "live_usage_link": str(LIVE_USAGE),
        "live_usage_snapshot": {
            "watch_headroom": watch_headroom,
            "total_inotify_watch_count": live.get("metrics", {}).get("total_inotify_watch_count"),
            "max_watch_process_pid": top_process.get("pid"),
            "max_watch_process_watches": top_process.get("inotify_watch_count"),
            "max_watch_process_command": top_process.get("command"),
            "live_still_saturated_after_settings_write": live_still_saturated,
        },
        "checks": {
            "settings_file_exists": SETTINGS.is_file(),
            "all_required_watcher_excludes_present": not any(not row["watcher_excluded"] for row in rows),
            "all_required_search_excludes_present": not any(not row["search_excluded"] for row in rows),
            "live_usage_audit_exists": LIVE_USAGE.is_file(),
            "does_not_kill_vscode_or_modify_sysctl": True,
            "does_not_claim_kit_unblocked": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The workspace settings reduce future VS Code file-watcher pressure after the server/window reloads, "
                "but this audit does not itself release already-open watches, raise sysctl limits, launch Kit, or "
                "prove official training success."
            ),
        },
        "outputs": {
            "json": str(OUT / "vscode_watcher_exclude_audit.json"),
            "tsv": str(OUT / "vscode_watcher_exclude_audit.tsv"),
        },
    }
    (OUT / "vscode_watcher_exclude_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT / "vscode_watcher_exclude_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["pattern", "watcher_excluded", "search_excluded"])
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": summary["outputs"]["json"],
                "missing": len(missing),
                "live_still_saturated": live_still_saturated,
                "watch_headroom": watch_headroom,
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
