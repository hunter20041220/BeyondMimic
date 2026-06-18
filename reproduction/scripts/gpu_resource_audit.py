#!/usr/bin/env python3
"""Collect and audit GPU resource snapshots for BeyondMimic runs.

This script implements the logging schema requested in goal.md Section 6.3
without creating any artificial GPU load. Training jobs should reuse the same
CSV schema and set their real run_id/status.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
GPU_LOG = ROOT / "logs/gpu/gpu_metrics.csv"
OUT = ROOT / "res/setup/gpu_resource_audit"

GPU_FIELDS = [
    "index",
    "uuid",
    "name",
    "memory.total",
    "memory.used",
    "memory.free",
    "utilization.gpu",
    "power.draw",
    "temperature.gpu",
]

CSV_FIELDS = [
    "timestamp",
    "gpu_index",
    "gpu_uuid",
    "gpu_name",
    "memory_used_mib",
    "memory_total_mib",
    "memory_free_mib",
    "gpu_util_percent",
    "power_draw_w",
    "temperature_c",
    "process_pid",
    "run_id",
    "run_status",
    "sample_kind",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_float(value: str) -> float:
    value = value.strip()
    if value in {"", "[N/A]", "N/A"}:
        return 0.0
    return float(value)


def parse_int(value: str) -> int:
    return int(round(parse_float(value)))


def query_gpu_rows() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(GPU_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=True)
    rows: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        cols = [part.strip() for part in line.split(",")]
        if len(cols) != len(GPU_FIELDS):
            raise ValueError(f"unexpected nvidia-smi row: {line!r}")
        item = dict(zip(GPU_FIELDS, cols))
        rows.append(
            {
                "gpu_index": parse_int(item["index"]),
                "gpu_uuid": item["uuid"],
                "gpu_name": item["name"],
                "memory_used_mib": parse_int(item["memory.used"]),
                "memory_total_mib": parse_int(item["memory.total"]),
                "memory_free_mib": parse_int(item["memory.free"]),
                "gpu_util_percent": parse_int(item["utilization.gpu"]),
                "power_draw_w": parse_float(item["power.draw"]),
                "temperature_c": parse_int(item["temperature.gpu"]),
            }
        )
    return rows


def query_compute_processes() -> dict[int, list[int]]:
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    by_uuid: dict[str, list[int]] = {}
    if proc.returncode == 0:
        for line in proc.stdout.splitlines():
            cols = [part.strip() for part in line.split(",")]
            if len(cols) >= 2 and cols[0] and cols[1].isdigit():
                by_uuid.setdefault(cols[0], []).append(int(cols[1]))
    uuid_to_index = {row["gpu_uuid"]: row["gpu_index"] for row in query_gpu_rows()}
    return {uuid_to_index[uuid]: pids for uuid, pids in by_uuid.items() if uuid in uuid_to_index}


def collect(samples: int, interval_sec: float, run_id: str, run_status: str, sample_kind: str) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    for sample_idx in range(samples):
        ts = now_iso()
        processes = query_compute_processes()
        for row in query_gpu_rows():
            pids = processes.get(row["gpu_index"], [])
            all_rows.append(
                {
                    "timestamp": ts,
                    **row,
                    "process_pid": ";".join(str(pid) for pid in pids),
                    "run_id": run_id,
                    "run_status": run_status,
                    "sample_kind": sample_kind,
                }
            )
        if sample_idx + 1 < samples:
            time.sleep(interval_sec)
    return all_rows


def append_gpu_log(rows: list[dict[str, Any]]) -> None:
    GPU_LOG.parent.mkdir(parents=True, exist_ok=True)
    exists = GPU_LOG.is_file() and GPU_LOG.stat().st_size > 0
    with GPU_LOG.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    gpu_indices = sorted({row["gpu_index"] for row in rows})
    max_mem_by_gpu = {
        str(idx): max(row["memory_used_mib"] for row in rows if row["gpu_index"] == idx) for idx in gpu_indices
    }
    max_util_by_gpu = {
        str(idx): max(row["gpu_util_percent"] for row in rows if row["gpu_index"] == idx) for idx in gpu_indices
    }
    process_pids = sorted({pid for row in rows for pid in str(row["process_pid"]).split(";") if pid})
    nontrivial_memory_gpus = {
        gpu: mem for gpu, mem in max_mem_by_gpu.items() if mem >= args.nontrivial_memory_threshold_mib
    }
    return {
        "status": "ok",
        "experiment_type": "gpu_resource_audit",
        "scope": "non-training GPU metrics schema and current resource snapshot",
        "run_id": args.run_id,
        "run_status": args.run_status,
        "sample_kind": args.sample_kind,
        "samples_requested": args.samples,
        "rows_written": len(rows),
        "gpu_count": len(gpu_indices),
        "csv_schema": CSV_FIELDS,
        "max_memory_used_mib_by_gpu": max_mem_by_gpu,
        "max_gpu_util_percent_by_gpu": max_util_by_gpu,
        "compute_process_pids_seen": process_pids,
        "nontrivial_memory_gpus": nontrivial_memory_gpus,
        "checks": {
            "gpu_metrics_csv_written": GPU_LOG.is_file() and GPU_LOG.stat().st_size > 0,
            "row_count_matches_samples_x_gpus": len(rows) == args.samples * len(gpu_indices),
            "has_goal_required_columns": all(field in CSV_FIELDS for field in CSV_FIELDS),
            "does_not_modify_power_or_clocks": True,
            "does_not_create_artificial_load": args.sample_kind == "snapshot_no_training_load",
            "records_nontrivial_existing_memory": bool(nontrivial_memory_gpus),
            "does_not_claim_training_utilization": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The GPU metrics CSV schema and current snapshot are now available, but no long PPO/VAE/diffusion "
                "training run has completed, so target 18-22 GiB utilization and per-run throughput metrics remain "
                "unproven."
            ),
        },
        "outputs": {
            "gpu_metrics_csv": str(GPU_LOG),
            "json": str(OUT / "gpu_resource_audit.json"),
            "tsv": str(OUT / "gpu_resource_audit.tsv"),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument("--run-id", default="setup_gpu_resource_snapshot")
    parser.add_argument("--run-status", default="INVALID")
    parser.add_argument("--sample-kind", default="snapshot_no_training_load")
    parser.add_argument("--nontrivial-memory-threshold-mib", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = collect(args.samples, args.interval_sec, args.run_id, args.run_status, args.sample_kind)
    append_gpu_log(rows)
    summary = summarize(rows, args)
    (OUT / "gpu_resource_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(OUT / "gpu_resource_audit.tsv", rows)
    print(json.dumps({"status": summary["status"], "rows": len(rows), "json": summary["outputs"]["json"]}))


if __name__ == "__main__":
    main()
