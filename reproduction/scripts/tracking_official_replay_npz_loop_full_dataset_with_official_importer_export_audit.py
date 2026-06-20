#!/usr/bin/env python3
"""Replay official-importer-export converted full-dataset NPZs via official replay_npz.py."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
SCRIPT = ROOT / "reproduction/scripts/tracking_official_replay_npz_loop_with_enriched_usd_audit.py"
CSV_FULL_AUDIT = (
    ROOT
    / "res/tracking/official_csv_to_npz_loop_full_dataset_with_official_importer_export/"
    "tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_audit.json"
)
OUT = ROOT / "res/tracking/official_replay_npz_loop_full_dataset_with_official_importer_export"
LOG_DIR = ROOT / "logs/tracking_official_replay_npz_loop_full_dataset_with_official_importer_export"
FAILED_DIR = ROOT / "res/failed_runs/tracking_official_replay_npz_loop_full_dataset_with_official_importer_export"
OFFICIAL_IMPORTER_USD = (
    ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda"
)
TARGET_GPU = int(os.environ.get("BM_OFFICIAL_IMPORTER_REPLAY_FULL_DATASET_TARGET_GPU", "4"))
MAX_STEPS = int(os.environ.get("BM_OFFICIAL_IMPORTER_REPLAY_FULL_DATASET_MAX_STEPS", "299"))
LIMIT = int(os.environ.get("BM_OFFICIAL_IMPORTER_REPLAY_FULL_DATASET_LIMIT", "0"))


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_table(rows: list[dict[str, Any]], path: Path, delimiter: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "motion",
        "status",
        "latest_blocker",
        "returncode",
        "duration_seconds",
        "official_loop_body_completed",
        "shutdown_warning",
        "official_loop_call_299",
        "fake_wandb_download",
        "g1_cfg_patched_to_robot_usd",
        "uses_official_importer_export_usd",
        "motion_npz",
        "log",
        "wrapper_stdout_log",
        "motion_audit",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def selected_motion_rows() -> list[dict[str, Any]]:
    audit = load_json(CSV_FULL_AUDIT)
    rows = audit.get("rows") or []
    rows = [row for row in rows if row.get("status") == "ok" and row.get("output_npz")]
    rows = sorted(rows, key=lambda row: row["motion"])
    if LIMIT > 0:
        rows = rows[:LIMIT]
    return rows


def run_one(row: dict[str, Any]) -> dict[str, Any]:
    motion = row["motion"]
    motion_npz = Path(row["output_npz"])
    motion_out = OUT / "motions" / motion
    motion_out.mkdir(parents=True, exist_ok=True)
    fake_artifact_dir = ROOT / "tmp/tracking_official_replay_npz_loop_full_dataset_with_official_importer_export" / motion
    audit_basename = f"{motion}_official_replay_loop_official_importer_export_audit"
    env = os.environ.copy()
    env.update(
        {
            "BM_OFFICIAL_REPLAY_LOOP_MOTION_NPZ": str(motion_npz),
            "BM_OFFICIAL_REPLAY_LOOP_FAKE_ARTIFACT_DIR": str(fake_artifact_dir),
            "BM_OFFICIAL_REPLAY_LOOP_REGISTRY_NAME": f"bm-local/{motion}_official_importer_export_motion:latest",
            "BM_OFFICIAL_REPLAY_LOOP_LOG_BASENAME": f"{motion}_official_replay_loop_official_importer_export",
            "BM_OFFICIAL_REPLAY_LOOP_TARGET_GPU": str(TARGET_GPU),
            "BM_OFFICIAL_REPLAY_LOOP_MAX_STEPS": str(MAX_STEPS),
            "BM_OFFICIAL_REPLAY_LOOP_OUT_DIR": str(motion_out),
            "BM_OFFICIAL_REPLAY_LOOP_LOG_DIR": str(LOG_DIR),
            "BM_OFFICIAL_REPLAY_LOOP_FAILED_DIR": str(FAILED_DIR),
            "BM_OFFICIAL_REPLAY_LOOP_ROBOT_USD": str(OFFICIAL_IMPORTER_USD),
            "BM_OFFICIAL_REPLAY_LOOP_USD_LABEL": "official_importer_export_usd",
            "BM_OFFICIAL_REPLAY_LOOP_USES_RESOURCE_ADJUSTED_USD": "0",
            "BM_OFFICIAL_REPLAY_LOOP_USES_OFFICIAL_IMPORTER_EXPORT_USD": "1",
            "BM_OFFICIAL_REPLAY_LOOP_AUDIT_BASENAME": audit_basename,
            "BM_OFFICIAL_REPLAY_LOOP_SUCCESS_STATUS": "ok_official_replay_loop_with_official_importer_export",
            "BM_OFFICIAL_REPLAY_LOOP_SUCCESS_SHUTDOWN_STATUS": (
                "ok_official_replay_loop_with_official_importer_export_shutdown_warning"
            ),
            "BM_OFFICIAL_REPLAY_LOOP_BLOCKER_STATUS": "ok_with_official_replay_loop_official_importer_export_blocker",
            "BM_OFFICIAL_REPLAY_LOOP_EXPERIMENT_TYPE": "tracking_official_replay_npz_loop_with_official_importer_export",
            "BM_OFFICIAL_REPLAY_LOOP_CLAIM_LEVEL": "local_virtual_official_loop_official_importer_export_reference_replay",
            "PYTHONUNBUFFERED": "1",
        }
    )
    start = time.time()
    proc = subprocess.run(
        ["python3", str(SCRIPT)],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    stdout_log = LOG_DIR / f"{motion}_wrapper_stdout.log"
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stdout_log.write_text(proc.stdout, encoding="utf-8")
    single = load_json(motion_out / f"{audit_basename}.json")
    checks = single.get("checks") or {}
    markers = (single.get("run") or {}).get("markers") or {}
    interpretation = single.get("interpretation") or {}
    ok_statuses = {
        "ok_official_replay_loop_with_official_importer_export",
        "ok_official_replay_loop_with_official_importer_export_shutdown_warning",
    }
    ok = (
        proc.returncode == 0
        and single.get("status") in ok_statuses
        and bool(interpretation.get("official_loop_body_completed"))
        and bool(checks.get("fake_wandb_download_seen"))
        and bool(checks.get("g1_cfg_patched_to_robot_usd"))
        and bool(checks.get("uses_official_importer_export_usd_matches_expected"))
        and checks.get("uses_resource_adjusted_usd_matches_expected") is True
        and bool(checks.get("official_loop_call_299_seen"))
    )
    result = {
        "motion": motion,
        "status": "ok" if ok else "failed",
        "latest_blocker": single.get("latest_blocker", "missing_single_audit"),
        "returncode": proc.returncode,
        "duration_seconds": round(time.time() - start, 3),
        "official_loop_body_completed": bool(interpretation.get("official_loop_body_completed")),
        "shutdown_warning": bool(interpretation.get("shutdown_warning")),
        "official_loop_call_299": bool(markers.get("official_loop_call_299")),
        "fake_wandb_download": bool(markers.get("fake_wandb_download")),
        "g1_cfg_patched_to_robot_usd": bool(markers.get("g1_cfg_patched_to_robot_usd")),
        "uses_official_importer_export_usd": bool(checks.get("uses_official_importer_export_usd")),
        "motion_npz": str(motion_npz),
        "log": single.get("outputs", {}).get("log", ""),
        "wrapper_stdout_log": str(stdout_log),
        "motion_audit": str(motion_out / f"{audit_basename}.json"),
    }
    if not ok:
        FAILED_DIR.mkdir(parents=True, exist_ok=True)
        failed_path = FAILED_DIR / f"{motion}_wrapper_stdout.log"
        failed_path.write_text(proc.stdout, encoding="utf-8")
        result["failed_log_copy"] = str(failed_path)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    motion_rows = selected_motion_rows()
    rows = [run_one(row) for row in motion_rows]
    ok_rows = [row for row in rows if row["status"] == "ok"]
    failed_rows = [row for row in rows if row["status"] != "ok"]
    checks = {
        "official_replay_loop_wrapper_exists": SCRIPT.is_file(),
        "csv_full_dataset_audit_exists": CSV_FULL_AUDIT.is_file(),
        "official_importer_usd_exists": OFFICIAL_IMPORTER_USD.is_file(),
        "all_40_csv_loop_outputs_selected": len(motion_rows) == 40 if LIMIT == 0 else len(motion_rows) == LIMIT,
        "all_motion_npz_paths_exist": all(Path(row["motion_npz"]).is_file() for row in rows),
        "all_rows_ok": len(rows) > 0 and len(failed_rows) == 0,
        "all_rows_reached_official_loop_299": all(row["official_loop_call_299"] for row in rows),
        "all_rows_downloaded_fake_wandb_artifact": all(row["fake_wandb_download"] for row in rows),
        "all_rows_used_official_importer_export_usd": all(row["uses_official_importer_export_usd"] for row in rows),
        "uses_official_replay_npz_loop": True,
        "uses_official_csv_loop_npz_inputs": True,
        "uses_official_importer_export_usd": True,
        "does_not_use_resource_adjusted_enriched_usd": True,
        "does_not_claim_unpatched_official_asset_complete": True,
        "does_not_claim_trained_policy_eval": True,
        "does_not_claim_paper_level_replay": True,
        "does_not_start_training": True,
    }
    aggregate = {
        "row_count": len(rows),
        "ok_count": len(ok_rows),
        "failed_count": len(failed_rows),
        "total_replayed_steps": sum(MAX_STEPS for _ in ok_rows),
        "shutdown_warning_count": sum(1 for row in rows if row.get("shutdown_warning")),
        "total_duration_seconds": round(sum(float(row["duration_seconds"]) for row in rows), 3),
    }
    status = (
        "ok_official_replay_npz_loop_full_dataset_with_official_importer_export"
        if all(checks.values())
        else "ok_with_official_replay_npz_loop_full_dataset_official_importer_export_blocker"
    )
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_official_replay_npz_loop_full_dataset_with_official_importer_export",
        "scope": (
            "Runs the official whole_body_tracking replay_npz.py loop over every NPZ produced by the full-dataset "
            "official csv_to_npz.py loop on the captured official-importer-export G1 USDA. This avoids the generated "
            "enriched scaffold but still bypasses the live official converter."
        ),
        "config": {
            "target_gpu": TARGET_GPU,
            "max_steps_per_motion": MAX_STEPS,
            "limit": LIMIT,
            "csv_full_dataset_audit": str(CSV_FULL_AUDIT),
            "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
        },
        "aggregate": aggregate,
        "checks": checks,
        "rows": rows,
        "failed_rows": [row["motion"] for row in failed_rows],
        "outputs": {
            "json": str(OUT / "tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_audit.json"),
            "rows_csv": str(OUT / "tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_rows.csv"),
            "rows_tsv": str(OUT / "tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_rows.tsv"),
            "motion_root": str(OUT / "motions"),
            "log_dir": str(LOG_DIR),
        },
        "interpretation": {
            "claim_level": "local_virtual_official_loop_official_importer_export_full_public_motion_reference_replay",
            "goal_complete": False,
            "paper_level_tracking_eval_complete": False,
            "not_paper_level_reasons": [
                "uses captured official-importer USDA instead of the live unmodified official URDF converter path",
                "uses official csv_to_npz loop NPZs generated under the same captured-importer path",
                "does not evaluate a trained policy, success/fall rate, reward, or tracking error metrics",
                "does not train PPO and does not produce official paper rollout videos",
                "does not involve real robot hardware",
            ],
        },
    }
    write_table(rows, OUT / "tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_rows.csv", ",")
    write_table(rows, OUT / "tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_rows.tsv", "\t")
    (OUT / "tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "rows": len(rows), "json": summary["outputs"]["json"]}, sort_keys=True))


if __name__ == "__main__":
    main()
