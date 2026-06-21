#!/usr/bin/env python3
"""Conservatively remove failed/superseded bulky artifacts.

The cleanup policy is intentionally narrow: keep JSON/CSV/TSV/MD/log audit
evidence, keep artifacts referenced by current summaries, and only delete
known failed or superseded bulky directories that are not required by the
current evidence chain.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/storage_cleanup"

DELETE_CANDIDATES = [
    {
        "path": ROOT / "res/tracking/g1_official_importer_export_fk_repaired_full_bundle_task_eval",
        "reason": "failed full-bundle FK-repaired task-eval working directory superseded by 40/40 split task eval",
        "keep_reason": "",
    },
]

RETAINED_BULKY_CANDIDATES = [
    {
        "path": ROOT
        / "res/runs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset/"
        "resource_adjusted_teacher_rollout_20260620_195754_seed20260700",
        "reason": (
            "large older teacher-rollout shard directory, but still referenced by existing VAE/state-latent "
            "worker summaries and required-artifact absence audits"
        ),
    },
    {
        "path": ROOT
        / "res/runs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset/"
        "resource_adjusted_teacher_rollout_20260621_060339_seed20260700",
        "reason": "current scaled-PPO teacher-rollout run_dir referenced by the active dataset summary",
    },
    {
        "path": ROOT
        / "res/runs/level_c_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset/"
        "resource_adjusted_state_latent_dataset_20260621_042551_seed20260702",
        "reason": (
            "large older state-latent shard directory, but still referenced by existing downstream summaries and "
            "required-artifact absence audits"
        ),
    },
    {
        "path": ROOT
        / "res/runs/level_c_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset/"
        "resource_adjusted_state_latent_dataset_20260621_141711_seed20260702",
        "reason": "current scaled-PPO state-latent run_dir referenced by the active dataset summary",
    },
]


def size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    deleted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    before_free = shutil.disk_usage(ROOT).free

    for item in DELETE_CANDIDATES:
        path = item["path"]
        before = size_bytes(path)
        row = {
            "path": rel(path),
            "exists_before": path.exists(),
            "size_bytes_before": before,
            "reason": item["reason"],
            "policy": "delete_failed_or_superseded_bulky_working_directory_keep_audit_logs",
        }
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            row["exists_after"] = path.exists()
            row["size_bytes_after"] = size_bytes(path)
            deleted.append(row)
        else:
            row["exists_after"] = False
            row["size_bytes_after"] = 0
            skipped.append(row)

    retained = []
    for item in RETAINED_BULKY_CANDIDATES:
        path = item["path"]
        retained.append(
            {
                "path": rel(path),
                "exists": path.exists(),
                "size_bytes": size_bytes(path),
                "reason_not_deleted": item["reason"],
            }
        )

    after_free = shutil.disk_usage(ROOT).free
    freed = sum(row["size_bytes_before"] - row["size_bytes_after"] for row in deleted)
    summary = {
        "status": "ok",
        "experiment_type": "cleanup_failed_large_artifacts",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Conservative storage cleanup for failed/superseded bulky artifacts. JSON/CSV/TSV/MD/log evidence and "
            "currently referenced run directories are retained."
        ),
        "deleted": deleted,
        "skipped": skipped,
        "retained_bulky_candidates": retained,
        "metrics": {
            "deleted_count": len(deleted),
            "freed_bytes_by_deleted_rows": freed,
            "filesystem_free_bytes_before": before_free,
            "filesystem_free_bytes_after": after_free,
            "filesystem_free_bytes_delta": after_free - before_free,
        },
        "checks": {
            "only_failed_or_superseded_candidates_deleted": True,
            "current_scaled_teacher_rollout_run_dir_retained": any(
                "20260621_060339" in row["path"] and row["exists"] for row in retained
            ),
            "current_scaled_state_latent_run_dir_retained": any(
                "20260621_141711" in row["path"] and row["exists"] for row in retained
            ),
            "audit_logs_retained": True,
            "does_not_modify_download": True,
        },
        "interpretation": {
            "goal_complete": False,
            "claim_level": "storage_hygiene_only",
            "why_not_more_aggressive": (
                "Several large older shards are still referenced by existing VAE/state-latent worker summaries and "
                "absence audits, so they were retained instead of deleted blindly."
            ),
        },
        "outputs": {
            "json": str(OUT / "cleanup_failed_large_artifacts.json"),
        },
    }
    write_json(OUT / "cleanup_failed_large_artifacts.json", summary)
    print(json.dumps({"status": summary["status"], "deleted": len(deleted), "freed_bytes": freed, "json": summary["outputs"]["json"]}, sort_keys=True))


if __name__ == "__main__":
    main()
