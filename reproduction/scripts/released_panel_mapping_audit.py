#!/usr/bin/env python3
"""Audit released-data panel mapping against local dataset and generated figures."""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
PANEL_MAP = ROOT / "reproduction/docs/paper_panel_map.tsv"
SUMMARY = ROOT / "res/released_figures/released_figure_summary.tsv"
DATASET = ROOT / "reproduction/data/Dataset_beyondmimic"
ZIP_PATH = ROOT / "download/official/Dataset_beyondmimic.zip"
OUT = ROOT / "res/released_panel_mapping_audit"

EXPECTED_RELEASED_FIGURE_IDS = {
    "ablation_armature",
    "ablation_latency",
    "ablation_observation_history",
    "ablation_orientation_representation",
    "ablation_pd_gain",
    "adaptive_sampling_probability_evolution",
    "adaptive_sampling_w",
    "adaptive_sampling_wo",
    "grf_run_human_reference",
    "grf_run_robot_real",
    "grf_walk_human_reference",
    "grf_walk_robot_real",
    "imu_orientation_accel_angular_velocity",
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def zip_contains(zip_names: set[str], local_path: Path) -> bool:
    try:
        rel = local_path.resolve().relative_to(DATASET.resolve())
    except ValueError:
        return False
    name = "Dataset_beyondmimic/" + rel.as_posix()
    return name in zip_names


def load_zip_names(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as zf:
        return {info.filename for info in zf.infolist()}


def figure_dir_from_summary(row: dict[str, str]) -> Path:
    outputs = [Path(x) for x in row["outputs"].split(",") if x]
    if outputs:
        return outputs[0].parent
    return Path(row["processed_csv"]).parent


def source_hash_rows(figure_dir: Path) -> list[dict[str, str]]:
    path = figure_dir / "source_hashes.tsv"
    if not path.exists():
        return []
    return read_tsv(path)


def source_stats(source_rows: list[dict[str, str]], zip_names: set[str]) -> dict[str, Any]:
    existing = 0
    in_dataset = 0
    in_zip = 0
    bad_hash = 0
    sources: list[dict[str, Any]] = []
    for row in source_rows:
        path = Path(row["path"])
        exists = path.exists()
        is_dataset = False
        is_zip = False
        hash_ok = None
        if exists:
            existing += 1
            try:
                path.resolve().relative_to(DATASET.resolve())
                is_dataset = True
                in_dataset += 1
            except ValueError:
                pass
            is_zip = zip_contains(zip_names, path)
            if is_zip:
                in_zip += 1
            expected_hash = row.get("sha256", "")
            if expected_hash:
                hash_ok = sha256_file(path) == expected_hash
                if not hash_ok:
                    bad_hash += 1
        sources.append(
            {
                "path": str(path),
                "exists": exists,
                "inside_extracted_dataset": is_dataset,
                "present_in_raw_zip": is_zip,
                "hash_ok": hash_ok,
            }
        )
    return {
        "source_count": len(source_rows),
        "existing_source_count": existing,
        "dataset_source_count": in_dataset,
        "raw_zip_source_count": in_zip,
        "bad_hash_count": bad_hash,
        "sources": sources,
    }


def map_panel_to_figure_id(row: dict[str, str]) -> str | None:
    artifact = row["local_reproduction_artifact"]
    if "/res/released_figures/" not in artifact:
        return None
    parts = Path(artifact).parts
    try:
        idx = parts.index("released_figures")
    except ValueError:
        return None
    if idx + 1 >= len(parts):
        return None
    group = parts[idx + 1]
    if group == "adaptive_sampling_converted":
        rel = "/".join(parts[idx + 2 :])
        if rel.startswith("w/"):
            return "adaptive_sampling_w"
        if rel.startswith("wo/"):
            return "adaptive_sampling_wo"
    if group == "adaptive_sampling":
        name = Path(artifact).name
        if "probability" in name:
            return "adaptive_sampling_probability_evolution"
        if "_w_" in name:
            return "adaptive_sampling_w"
        if "_wo_" in name:
            return "adaptive_sampling_wo"
    return group


def build_rows(panel_rows: list[dict[str, str]], summary_rows: list[dict[str, str]], zip_names: set[str]) -> list[dict[str, Any]]:
    by_figure = {row["figure_id"]: row for row in summary_rows}
    output_rows: list[dict[str, Any]] = []
    for panel in panel_rows:
        status = panel["status"]
        if not status.startswith("released-data"):
            continue
        figure_id = map_panel_to_figure_id(panel)
        summary = by_figure.get(figure_id or "")
        processed_path = Path(summary["processed_csv"]) if summary else Path(panel["local_reproduction_artifact"])
        outputs = [Path(x) for x in summary["outputs"].split(",") if x] if summary else []
        figure_dir = figure_dir_from_summary(summary) if summary else processed_path.parent
        stats = source_stats(source_hash_rows(figure_dir), zip_names)
        local_artifact = Path(panel["local_reproduction_artifact"])
        row = {
            "paper_item": panel["paper_item"],
            "label_or_panel": panel["label_or_panel"],
            "source_evidence": panel["source_evidence"],
            "status": status,
            "figure_id": figure_id,
            "summary_row_present": summary is not None,
            "local_artifact": str(local_artifact),
            "local_artifact_exists": local_artifact.exists(),
            "processed_csv": str(processed_path),
            "processed_csv_exists": processed_path.exists(),
            "outputs": [str(p) for p in outputs],
            "output_count": len(outputs),
            "missing_outputs": [str(p) for p in outputs if not p.exists()],
            "run_log": str(figure_dir / "run.log"),
            "run_log_exists": (figure_dir / "run.log").exists(),
            "source_hashes": str(figure_dir / "source_hashes.tsv"),
            "source_hashes_exists": (figure_dir / "source_hashes.tsv").exists(),
            "source_count": stats["source_count"],
            "existing_source_count": stats["existing_source_count"],
            "dataset_source_count": stats["dataset_source_count"],
            "raw_zip_source_count": stats["raw_zip_source_count"],
            "bad_hash_count": stats["bad_hash_count"],
            "notes": panel["notes"],
            "source_details": stats["sources"],
        }
        row["passed"] = (
            row["summary_row_present"]
            and row["local_artifact_exists"]
            and row["processed_csv_exists"]
            and row["run_log_exists"]
            and row["source_hashes_exists"]
            and row["source_count"] > 0
            and row["existing_source_count"] == row["source_count"]
            and row["dataset_source_count"] > 0
            and row["raw_zip_source_count"] > 0
            and row["bad_hash_count"] == 0
            and len(row["missing_outputs"]) == 0
        )
        output_rows.append(row)
    return output_rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "paper_item",
        "label_or_panel",
        "source_evidence",
        "status",
        "figure_id",
        "summary_row_present",
        "local_artifact",
        "local_artifact_exists",
        "processed_csv",
        "processed_csv_exists",
        "output_count",
        "run_log_exists",
        "source_hashes_exists",
        "source_count",
        "existing_source_count",
        "dataset_source_count",
        "raw_zip_source_count",
        "bad_hash_count",
        "passed",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel_rows = read_tsv(PANEL_MAP)
    summary_rows = read_tsv(SUMMARY)
    zip_names = load_zip_names(ZIP_PATH)
    rows = build_rows(panel_rows, summary_rows, zip_names)
    status_counts = Counter(row["status"] for row in panel_rows)
    released_figure_ids = {row["figure_id"] for row in summary_rows}
    mapped_figure_ids = {row["figure_id"] for row in rows if row["figure_id"]}
    failed_rows = [row for row in rows if not row["passed"]]
    json_path = OUT / "released_panel_mapping_audit.json"
    tsv_path = OUT / "released_panel_mapping_audit.tsv"
    summary = {
        "status": "ok" if not failed_rows else "failed",
        "experiment_type": "released_panel_mapping_audit",
        "scope": (
            "Trace released-data paper panel mappings to generated artifacts, extracted Dataset_beyondmimic files, "
            "and the immutable raw zip. This does not reproduce paper-only, blocked Fig.5/Fig.6, or hardware claims."
        ),
        "metrics": {
            "paper_panel_map_rows": len(panel_rows),
            "released_panel_rows": len(rows),
            "released_summary_rows": len(summary_rows),
            "released_panel_pass_count": len(rows) - len(failed_rows),
            "released_panel_fail_count": len(failed_rows),
            "zip_member_count": len(zip_names),
            "expected_released_figure_id_count": len(EXPECTED_RELEASED_FIGURE_IDS),
            "released_figure_id_count": len(released_figure_ids),
            "mapped_released_figure_id_count": len(mapped_figure_ids),
        },
        "status_counts": dict(sorted(status_counts.items())),
        "expected_released_figure_ids": sorted(EXPECTED_RELEASED_FIGURE_IDS),
        "released_figure_ids": sorted(released_figure_ids),
        "mapped_released_figure_ids": sorted(mapped_figure_ids),
        "missing_expected_released_figure_ids": sorted(EXPECTED_RELEASED_FIGURE_IDS - released_figure_ids),
        "unmapped_summary_figure_ids": sorted(released_figure_ids - mapped_figure_ids),
        "failed_rows": [{k: v for k, v in row.items() if k != "source_details"} for row in failed_rows],
        "rows": rows,
        "checks": {
            "raw_zip_exists": ZIP_PATH.exists(),
            "extracted_dataset_exists": DATASET.exists(),
            "all_expected_released_figure_ids_present": EXPECTED_RELEASED_FIGURE_IDS <= released_figure_ids,
            "all_summary_figure_ids_mapped_to_paper_panels": released_figure_ids <= mapped_figure_ids,
            "all_released_panel_rows_pass": len(failed_rows) == 0,
            "non_released_paper_items_not_claimed": all(
                not row["status"].startswith("released-data")
                for row in panel_rows
                if row["paper_item"] in {"Figure 5", "Figure 6"}
            ),
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The released-data panel mapping is complete for Level A traceability, but it does not unblock "
                "live Kit evaluation, trained VAE/diffusion reproduction, Fig.5/Fig.6, or real robot deployment."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, rows)
    print(json.dumps({"status": summary["status"], "json": str(json_path), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
