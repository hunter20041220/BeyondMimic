#!/usr/bin/env python3
"""Audit local Kit/Isaac Sim file-watcher configuration surfaces."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/setup/kit_watcher_config_surface_audit"
ISAAC_ROOT = ROOT / "envs/isaacsim-4.5.0"
PYTHON_APP = ISAAC_ROOT / "apps/isaacsim.exp.base.python.kit"
FULL_APP = ISAAC_ROOT / "apps/isaacsim.exp.full.kit"
KIT_CORE = ISAAC_ROOT / "kit/kernel/config/kit-core.json"
RETRY_LOG = ROOT / "logs/setup/isaaclab_headless_smoke_retry.log"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


def parse_failed_watch_paths(log_text: str) -> list[str]:
    return sorted(set(re.findall(r"Failed to create change watch for `([^`]+)`: errno=28", log_text)))


def parse_extension_folders(app_text: str) -> list[str]:
    match = re.search(r"\[settings\.app\.exts\.folders\]\s*'\+\+'\s*=\s*\[(.*?)\]", app_text, re.S)
    if not match:
        return []
    return [item.strip().strip('"').strip("'") for item in match.group(1).split(",") if item.strip()]


def scan_fswatcher_configs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    roots = [
        ISAAC_ROOT / "exts",
        ISAAC_ROOT / "extsDeprecated",
        ISAAC_ROOT / "extsPhysics",
    ]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*/config/extension.toml")):
            text = read_text(path)
            if "[fswatcher" not in text:
                continue
            rows.append(
                {
                    "path": str(path),
                    "has_fswatcher_paths": "[fswatcher.paths]" in text,
                    "has_fswatcher_patterns": "[fswatcher.patterns]" in text,
                    "include_patterns": re.findall(r"include\s*=\s*\[([^\]]*)\]", text),
                    "exclude_patterns": re.findall(r"exclude\s*=\s*\[([^\]]*)\]", text),
                }
            )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    retry_log = read_text(RETRY_LOG)
    python_app = read_text(PYTHON_APP)
    full_app = read_text(FULL_APP)
    core_text = read_text(KIT_CORE)
    failed_paths = parse_failed_watch_paths(retry_log)
    python_folders = parse_extension_folders(python_app)
    fswatcher_configs = scan_fswatcher_configs()
    watched_config_mentions = (
        read_text(ISAAC_ROOT / "kit/compile_commands.json").count("omni.kit.watched_config")
        + core_text.count("watched")
        + python_app.count("watched")
        + full_app.count("watched")
    )
    watch_root_tokens = [
        "envs/isaacsim-4.5.0/apps",
        "envs/isaacsim-4.5.0/exts",
        "envs/isaacsim-4.5.0/extsPhysics",
        "envs/isaacsim-4.5.0/extscache",
        "envs/isaacsim-4.5.0/extsDeprecated",
        "envs/isaacsim-4.5.0/kit/apps",
        "envs/isaacsim-4.5.0/kit/exts",
        "envs/isaacsim-4.5.0/kit/extscore",
        "envs/isaacsim-4.5.0/kit/data/Kit",
        "reproduction/third_party/official/IsaacLab-v2.1.0/apps",
        "reproduction/third_party/official/IsaacLab-v2.1.0/source",
    ]
    failed_roots_matched_by_app = [
        path
        for path in failed_paths
        if any(root in path for root in watch_root_tokens)
    ]
    disable_candidates = []
    for text_name, text in [
        ("python_app", python_app),
        ("full_app", full_app),
        ("kit_core", core_text),
    ]:
        for pattern in [r"disable[^=\n]*watch", r"watch[^=\n]*disable", r"enable[^=\n]*watch"]:
            for match in re.finditer(pattern, text, flags=re.I):
                disable_candidates.append({"source": text_name, "match": match.group(0)})

    rows = [
        {
            "check": "python_app_extension_folders_indexed",
            "status": "pass" if len(python_folders) >= 4 else "fail",
            "evidence": str(PYTHON_APP),
            "detail": f"folders={python_folders}",
        },
        {
            "check": "failed_watch_paths_overlap_app_extension_roots",
            "status": "pass" if len(failed_roots_matched_by_app) >= 10 else "fail",
            "evidence": str(RETRY_LOG),
            "detail": f"matched_failed_paths={len(failed_roots_matched_by_app)}, total_failed_paths={len(failed_paths)}",
        },
        {
            "check": "extension_fswatcher_config_surfaces_exist",
            "status": "pass" if len(fswatcher_configs) >= 5 else "fail",
            "evidence": str(ISAAC_ROOT / "exts"),
            "detail": f"fswatcher_extension_config_count={len(fswatcher_configs)}",
        },
        {
            "check": "kit_watched_config_component_present",
            "status": "pass" if watched_config_mentions > 0 else "review",
            "evidence": str(ISAAC_ROOT / "kit/compile_commands.json"),
            "detail": f"omni.kit.watched_config_mentions={watched_config_mentions}",
        },
        {
            "check": "no_documented_global_disable_found_in_local_app_configs",
            "status": "pass" if not disable_candidates else "review",
            "evidence": f"{PYTHON_APP};{FULL_APP};{KIT_CORE}",
            "detail": f"disable_or_enable_watch_candidates={disable_candidates[:8]}",
        },
        {
            "check": "does_not_modify_or_launch_kit",
            "status": "pass",
            "evidence": __file__,
            "detail": "Source/config audit only; no Kit app, SimulationApp, PPO, replay, or file modification is performed.",
        },
    ]
    failed = [row for row in rows if row["status"] == "fail"]
    summary: dict[str, Any] = {
        "status": "ok" if not failed else "failed",
        "experiment_type": "kit_watcher_config_surface_audit",
        "scope": "local Isaac Sim/Kit file-watcher config surfaces and their relation to the current inotify blocker",
        "row_count": len(rows),
        "failed_row_count": len(failed),
        "rows": rows,
        "failed_watch_paths": failed_paths,
        "python_app_extension_folders": python_folders,
        "fswatcher_configs": fswatcher_configs,
        "metrics": {
            "failed_watch_path_count": len(failed_paths),
            "failed_paths_overlapping_app_extension_roots": len(failed_roots_matched_by_app),
            "watch_root_token_count": len(watch_root_tokens),
            "python_app_extension_folder_count": len(python_folders),
            "fswatcher_extension_config_count": len(fswatcher_configs),
            "fswatcher_paths_config_count": sum(1 for row in fswatcher_configs if row["has_fswatcher_paths"]),
            "fswatcher_patterns_config_count": sum(1 for row in fswatcher_configs if row["has_fswatcher_patterns"]),
            "omni_kit_watched_config_mentions": watched_config_mentions,
            "disable_candidate_count": len(disable_candidates),
        },
        "checks": {
            "python_app_loads_extension_roots": len(python_folders) >= 4,
            "retry_failures_overlap_app_extension_roots": len(failed_roots_matched_by_app) >= 10,
            "fswatcher_config_surfaces_exist": len(fswatcher_configs) >= 5,
            "kit_watched_config_component_present": watched_config_mentions > 0,
            "no_documented_global_disable_found": not disable_candidates,
            "does_not_modify_or_launch_kit": True,
            "does_not_claim_kit_unblocked": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "Local Kit/Isaac Sim configs expose extension folders and per-extension fswatcher pattern surfaces, "
                "and the failed paths overlap those extension roots. This audit did not find a documented global "
                "watcher-disable setting in the local app configs and does not prove a safe bypass under the current "
                "8192/128 inotify limits. Live Kit smoke remains gated by the host inotify setting."
            ),
        },
        "outputs": {
            "json": str(OUT / "kit_watcher_config_surface_audit.json"),
            "tsv": str(OUT / "kit_watcher_config_surface_audit.tsv"),
        },
    }
    (OUT / "kit_watcher_config_surface_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (OUT / "kit_watcher_config_surface_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["check", "status", "evidence", "detail"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"], "rows": len(rows)}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
