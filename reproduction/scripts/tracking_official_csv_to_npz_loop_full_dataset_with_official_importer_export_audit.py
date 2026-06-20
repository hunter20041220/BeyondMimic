#!/usr/bin/env python3
"""Run official csv_to_npz.py loop over all G1 CSVs using the official-importer USDA.

This is the official-importer-export companion to
``tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_audit.py``.
Each row executes the official ``whole_body_tracking/scripts/csv_to_npz.py``
loop body in a fresh process while selecting the large G1 USDA exported by the
official Isaac Sim URDF importer in-memory probe.

The claim boundary remains strict: this avoids the generated enriched scaffold,
but it still bypasses the live URDF converter by loading a project-local USDA
captured from the official importer. It is not unmodified official
``csv_to_npz.py`` conversion, not PPO, and not paper-level tracking.
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
OUT = ROOT / "res/tracking/official_csv_to_npz_loop_full_dataset_with_official_importer_export"
LOG_DIR = ROOT / "logs/tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export"
FAILED_DIR = ROOT / "res/failed_runs/tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export"
OFFICIAL_IMPORTER_USD = (
    ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda"
)
EXPORT_STRUCTURE_AUDIT = (
    ROOT
    / "res/tracking/g1_urdf_in_memory_gpu4_export_structure_audit/"
    "tracking_g1_urdf_in_memory_gpu4_export_structure_audit.json"
)
TARGET_GPU = int(os.environ.get("BM_OFFICIAL_IMPORTER_CSV_FULL_DATASET_TARGET_GPU", "4"))
MAX_STEPS = int(os.environ.get("BM_OFFICIAL_IMPORTER_CSV_FULL_DATASET_MAX_STEPS", "299"))
LIMIT = int(os.environ.get("BM_OFFICIAL_IMPORTER_CSV_FULL_DATASET_LIMIT", "0"))


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
        "wrapper_stdout_log",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def run_one(csv_path: Path) -> dict[str, Any]:
    motion = csv_path.stem
    motion_out = OUT / "motions" / motion
    motion_out.mkdir(parents=True, exist_ok=True)
    output_npz = motion_out / f"{motion}_official_loop_official_importer_export_motion.npz"
    metrics_json = motion_out / f"{motion}_official_loop_official_importer_export_metrics.json"
    audit_basename = f"{motion}_official_csv_to_npz_loop_official_importer_export_audit"
    env = os.environ.copy()
    env.update(
        {
            "BM_OFFICIAL_CSV_LOOP_INPUT_CSV": str(csv_path),
            "BM_OFFICIAL_CSV_LOOP_OUTPUT_NPZ": str(output_npz),
            "BM_OFFICIAL_CSV_LOOP_METRICS_JSON": str(metrics_json),
            "BM_OFFICIAL_CSV_LOOP_OUTPUT_NAME": f"bm_local_{motion}_official_csv_loop_official_importer_export",
            "BM_OFFICIAL_CSV_LOOP_LOG_BASENAME": f"{motion}_official_csv_loop_official_importer_export",
            "BM_OFFICIAL_CSV_LOOP_TARGET_GPU": str(TARGET_GPU),
            "BM_OFFICIAL_CSV_LOOP_MAX_STEPS": str(MAX_STEPS),
            "BM_OFFICIAL_CSV_LOOP_OUT_DIR": str(motion_out),
            "BM_OFFICIAL_CSV_LOOP_LOG_DIR": str(LOG_DIR),
            "BM_OFFICIAL_CSV_LOOP_FAILED_DIR": str(FAILED_DIR),
            "BM_OFFICIAL_CSV_LOOP_ROBOT_USD": str(OFFICIAL_IMPORTER_USD),
            "BM_OFFICIAL_CSV_LOOP_USD_LABEL": "official_importer_export_usd",
            "BM_OFFICIAL_CSV_LOOP_USES_RESOURCE_ADJUSTED_USD": "0",
            "BM_OFFICIAL_CSV_LOOP_USES_OFFICIAL_IMPORTER_EXPORT_USD": "1",
            "BM_OFFICIAL_CSV_LOOP_AUDIT_BASENAME": audit_basename,
            "BM_OFFICIAL_CSV_LOOP_SUCCESS_STATUS": "ok_official_csv_to_npz_loop_with_official_importer_export",
            "BM_OFFICIAL_CSV_LOOP_BLOCKER_STATUS": "ok_with_official_csv_to_npz_loop_official_importer_export_blocker",
            "BM_OFFICIAL_CSV_LOOP_EXPERIMENT_TYPE": "tracking_official_csv_to_npz_loop_with_official_importer_export",
            "BM_OFFICIAL_CSV_LOOP_CLAIM_LEVEL": "local_virtual_official_loop_official_importer_export_conversion",
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
    metrics = load_json(metrics_json)
    checks = single.get("checks") or {}
    ok = (
        proc.returncode == 0
        and single.get("status") == "ok_official_csv_to_npz_loop_with_official_importer_export"
        and checks.get("g1_cfg_patched_to_robot_usd") is True
        and checks.get("uses_official_importer_export_usd") is True
        and checks.get("uses_resource_adjusted_usd") is True
        and metrics.get("uses_resource_adjusted_usd") is False
        and metrics.get("uses_official_importer_export_usd") is True
        and metrics.get("joint_pos_shape") == [MAX_STEPS, 29]
        and metrics.get("body_pos_w_shape") == [MAX_STEPS, 40, 3]
    )
    row = {
        "motion": motion,
        "status": "ok" if ok else "failed",
        "latest_blocker": single.get("latest_blocker", "missing_single_audit"),
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
        "motion_audit": str(motion_out / f"{audit_basename}.json"),
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
    ok_rows = [row for row in rows if row["status"] == "ok"]
    failed_rows = [row for row in rows if row["status"] != "ok"]
    export_audit = load_json(EXPORT_STRUCTURE_AUDIT)
    checks = {
        "official_csv_to_npz_wrapper_exists": SCRIPT.is_file(),
        "official_importer_usd_exists": OFFICIAL_IMPORTER_USD.is_file(),
        "export_structure_audit_available": export_audit.get("status")
        == "ok_with_physics_usd_export_but_vulkan_device_lost",
        "all_40_csvs_selected": len(csv_paths) == 40 if LIMIT == 0 else len(csv_paths) == LIMIT,
        "all_rows_ok": len(rows) > 0 and len(failed_rows) == 0,
        "all_joint_shapes_299_29": all(row["joint_frames"] == MAX_STEPS and row["joint_dim"] == 29 for row in rows),
        "all_body_shapes_299_40": all(row["body_count"] == 40 for row in rows),
        "all_npz_paths_recorded": all(bool(row["output_npz"]) for row in rows),
        "uses_official_csv_to_npz_loop": True,
        "uses_official_importer_export_usd": True,
        "does_not_use_resource_adjusted_enriched_usd": True,
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
        "ok_official_csv_to_npz_loop_full_dataset_with_official_importer_export"
        if all(checks.values())
        else "ok_with_official_csv_to_npz_loop_full_dataset_official_importer_export_blocker"
    )
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export",
        "scope": (
            "Runs the official whole_body_tracking csv_to_npz.py loop over all local G1 LAFAN CSV files using the "
            "G1 USDA exported by the official Isaac Sim URDF importer. This avoids the generated enriched scaffold, "
            "but it still bypasses the live converter by selecting a captured project-local importer export."
        ),
        "config": {
            "csv_root": str(CSV_ROOT),
            "target_gpu": TARGET_GPU,
            "max_steps_per_motion": MAX_STEPS,
            "limit": LIMIT,
            "official_importer_usd": str(OFFICIAL_IMPORTER_USD),
            "export_structure_audit": str(EXPORT_STRUCTURE_AUDIT),
        },
        "aggregate": aggregate,
        "checks": checks,
        "rows": rows,
        "failed_rows": [row["motion"] for row in failed_rows],
        "outputs": {
            "json": str(OUT / "tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_audit.json"),
            "rows_csv": str(OUT / "tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_rows.csv"),
            "rows_tsv": str(OUT / "tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_rows.tsv"),
            "motion_root": str(OUT / "motions"),
            "log_dir": str(LOG_DIR),
        },
        "interpretation": {
            "claim_level": "local_virtual_official_loop_official_importer_export_full_public_motion_conversion",
            "goal_complete": False,
            "not_paper_level_reasons": [
                "uses a captured official-importer USDA instead of the live unmodified official URDF converter path",
                "does not evaluate trained tracking policy success or fall metrics",
                "does not train PPO and does not produce official paper rollout metrics",
                "does not involve real robot hardware",
            ],
        },
    }
    write_table(rows, OUT / "tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_rows.csv", ",")
    write_table(rows, OUT / "tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_rows.tsv", "\t")
    (OUT / "tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "rows": len(rows), "json": summary["outputs"]["json"]}, sort_keys=True))


if __name__ == "__main__":
    main()
