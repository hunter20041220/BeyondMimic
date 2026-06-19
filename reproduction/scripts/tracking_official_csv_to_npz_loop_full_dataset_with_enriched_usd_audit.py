#!/usr/bin/env python3
"""Run official csv_to_npz.py loop over all local G1 LAFAN CSV motions.

This wraps ``tracking_official_csv_to_npz_loop_with_enriched_usd_audit.py`` and
keeps the same claim boundary: the official script body is executed, but the G1
asset path is the audited resource-adjusted enriched USD, not the unpatched
official URDF converter output.
"""

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
SCRIPT = ROOT / "reproduction/scripts/tracking_official_csv_to_npz_loop_with_enriched_usd_audit.py"
CSV_ROOT = ROOT / "download/official/LAFAN1_Retargeting_Dataset/g1"
OUT = ROOT / "res/tracking/official_csv_to_npz_loop_full_dataset_with_enriched_usd"
LOG_DIR = ROOT / "logs/tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd"
FAILED_DIR = ROOT / "res/failed_runs/tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd"
TARGET_GPU = int(os.environ.get("BM_OFFICIAL_CSV_FULL_DATASET_TARGET_GPU", "4"))
MAX_STEPS = int(os.environ.get("BM_OFFICIAL_CSV_FULL_DATASET_MAX_STEPS", "299"))
LIMIT = int(os.environ.get("BM_OFFICIAL_CSV_FULL_DATASET_LIMIT", "0"))


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
        "joint_frames",
        "joint_dim",
        "body_count",
        "max_joint_abs",
        "max_joint_vel_abs",
        "root_height_min",
        "root_height_max",
        "npz_size_bytes",
        "output_npz",
        "metrics_json",
        "log",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_one(csv_path: Path) -> dict[str, Any]:
    motion = csv_path.stem
    motion_out = OUT / "motions" / motion
    motion_out.mkdir(parents=True, exist_ok=True)
    output_npz = motion_out / f"{motion}_official_loop_enriched_usd_motion.npz"
    metrics_json = motion_out / f"{motion}_official_loop_enriched_usd_metrics.json"
    env = os.environ.copy()
    env.update(
        {
            "BM_OFFICIAL_CSV_LOOP_INPUT_CSV": str(csv_path),
            "BM_OFFICIAL_CSV_LOOP_OUTPUT_NPZ": str(output_npz),
            "BM_OFFICIAL_CSV_LOOP_METRICS_JSON": str(metrics_json),
            "BM_OFFICIAL_CSV_LOOP_OUTPUT_NAME": f"bm_local_{motion}_official_csv_loop_enriched_usd",
            "BM_OFFICIAL_CSV_LOOP_LOG_BASENAME": f"{motion}_official_csv_loop_enriched_usd",
            "BM_OFFICIAL_CSV_LOOP_TARGET_GPU": str(TARGET_GPU),
            "BM_OFFICIAL_CSV_LOOP_MAX_STEPS": str(MAX_STEPS),
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
    single = load_json(ROOT / "res/tracking/official_csv_to_npz_loop_with_enriched_usd/tracking_official_csv_to_npz_loop_with_enriched_usd_audit.json")
    metrics = load_json(metrics_json)
    latest_blocker = single.get("latest_blocker", "missing_single_audit")
    ok = (
        proc.returncode == 0
        and single.get("status") == "ok_official_csv_to_npz_loop_with_enriched_usd_patch"
        and metrics.get("joint_pos_shape") == [MAX_STEPS, 29]
        and metrics.get("body_pos_w_shape") == [MAX_STEPS, 40, 3]
    )
    row = {
        "motion": motion,
        "status": "ok" if ok else "failed",
        "latest_blocker": latest_blocker,
        "returncode": proc.returncode,
        "duration_seconds": round(time.time() - start, 3),
        "joint_frames": (metrics.get("joint_pos_shape") or ["", ""])[0],
        "joint_dim": (metrics.get("joint_pos_shape") or ["", ""])[1],
        "body_count": (metrics.get("body_pos_w_shape") or ["", "", ""])[1],
        "max_joint_abs": metrics.get("max_joint_abs", ""),
        "max_joint_vel_abs": metrics.get("max_joint_vel_abs", ""),
        "root_height_min": metrics.get("root_height_min", ""),
        "root_height_max": metrics.get("root_height_max", ""),
        "npz_size_bytes": metrics.get("npz_size_bytes", ""),
        "output_npz": str(output_npz) if output_npz.is_file() else "",
        "metrics_json": str(metrics_json) if metrics_json.is_file() else "",
        "log": single.get("outputs", {}).get("log", ""),
        "wrapper_stdout_log": str(stdout_log),
        "single_audit": single,
    }
    if not ok:
        FAILED_DIR.mkdir(parents=True, exist_ok=True)
        failed_path = FAILED_DIR / f"{motion}_wrapper_stdout.log"
        failed_path.write_text(proc.stdout, encoding="utf-8")
        row["failed_log_copy"] = str(failed_path)
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(CSV_ROOT.glob("*.csv"))
    if LIMIT > 0:
        csv_paths = csv_paths[:LIMIT]
    rows = [run_one(path) for path in csv_paths]
    table_rows = [{key: value for key, value in row.items() if key != "single_audit"} for row in rows]
    ok_rows = [row for row in rows if row["status"] == "ok"]
    failed_rows = [row for row in rows if row["status"] != "ok"]
    checks = {
        "official_csv_to_npz_wrapper_exists": SCRIPT.is_file(),
        "all_40_csvs_selected": len(csv_paths) == 40 if LIMIT == 0 else len(csv_paths) == LIMIT,
        "all_rows_ok": len(failed_rows) == 0 and len(rows) > 0,
        "all_joint_shapes_299_29": all(row["joint_frames"] == MAX_STEPS and row["joint_dim"] == 29 for row in rows),
        "all_body_shapes_299_40": all(row["body_count"] == 40 for row in rows),
        "all_npz_paths_recorded": all(bool(row["output_npz"]) for row in rows),
        "uses_official_csv_to_npz_loop": True,
        "uses_resource_adjusted_usd": True,
        "does_not_claim_unpatched_official_asset_complete": True,
        "does_not_claim_paper_level_replay": True,
        "does_not_start_training": True,
    }
    aggregate = {
        "row_count": len(rows),
        "ok_count": len(ok_rows),
        "failed_count": len(failed_rows),
        "total_frames": sum(int(row["joint_frames"] or 0) for row in rows),
        "total_joint_values": sum(int(row["joint_frames"] or 0) * int(row["joint_dim"] or 0) for row in rows),
        "total_npz_bytes": sum(int(row["npz_size_bytes"] or 0) for row in rows),
        "max_joint_abs_overall": max((float(row["max_joint_abs"]) for row in rows if row["max_joint_abs"] != ""), default=None),
        "max_joint_vel_abs_overall": max(
            (float(row["max_joint_vel_abs"]) for row in rows if row["max_joint_vel_abs"] != ""),
            default=None,
        ),
        "root_height_min_overall": min(
            (float(row["root_height_min"]) for row in rows if row["root_height_min"] != ""),
            default=None,
        ),
        "root_height_max_overall": max(
            (float(row["root_height_max"]) for row in rows if row["root_height_max"] != ""),
            default=None,
        ),
    }
    status = (
        "ok_official_csv_to_npz_loop_full_dataset_with_enriched_usd"
        if all(checks.values())
        else "ok_with_official_csv_to_npz_loop_full_dataset_blocker"
    )
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd",
        "scope": (
            "Runs the official whole_body_tracking csv_to_npz.py loop over all local G1 LAFAN CSV files through the "
            "same resource-adjusted enriched-USD runtime patch used by the single-motion audit. This is full public "
            "motion coverage for the conversion loop body, not unpatched official URDF-converter output, not replay "
            "success/failure metrics, not PPO, and not paper-level tracking."
        ),
        "config": {
            "csv_root": str(CSV_ROOT),
            "target_gpu": TARGET_GPU,
            "max_steps_per_motion": MAX_STEPS,
            "limit": LIMIT,
        },
        "aggregate": aggregate,
        "checks": checks,
        "rows": table_rows,
        "failed_rows": [row["motion"] for row in failed_rows],
        "outputs": {
            "json": str(OUT / "tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_audit.json"),
            "rows_csv": str(OUT / "tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_rows.csv"),
            "rows_tsv": str(OUT / "tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_rows.tsv"),
            "motion_root": str(OUT / "motions"),
            "log_dir": str(LOG_DIR),
        },
        "interpretation": {
            "claim_level": "resource_adjusted_official_loop_full_public_motion_conversion",
            "goal_complete": False,
            "not_paper_level_reasons": [
                "uses resource-adjusted enriched USD instead of the unpatched official G1 URDF converter output",
                "does not evaluate trained tracking policy success or fall metrics",
                "does not train PPO and does not produce official paper rollout metrics",
                "does not involve real robot hardware",
            ],
        },
    }
    write_table(table_rows, OUT / "tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_rows.csv", ",")
    write_table(table_rows, OUT / "tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_rows.tsv", "\t")
    (OUT / "tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "rows": len(rows), "json": summary["outputs"]["json"]}, sort_keys=True))


if __name__ == "__main__":
    main()
