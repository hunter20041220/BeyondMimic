#!/usr/bin/env python3
"""Inventory and hash the raw ``download`` source bundle used for reproduction.

The audit is intentionally read-only. It records the paper PDF/source tar,
official datasets/code, dependencies, reference code, supplemental packages,
and download manifests as the raw evidence boundary for the rest of the
reproduction.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
DOWNLOAD = ROOT / "download"
OUT = ROOT / "res/source_integrity/download_source_integrity"

REQUIRED_PATHS = {
    "scope_readme": "download/README_download_scope.md",
    "downloaded_files_manifest": "download/manifests/downloaded_files.tsv",
    "git_revisions_manifest": "download/manifests/git_revisions.tsv",
    "resource_sources_manifest": "download/manifests/resource_sources.md",
    "zenodo_metadata": "download/manifests/zenodo_17529720_metadata.json",
    "paper_pdf": "download/papers/BeyondMimic_2508.08241.pdf",
    "paper_source_tar": "download/papers/BeyondMimic_2508.08241_source.tar",
    "official_dataset_zip": "download/official/Dataset_beyondmimic.zip",
    "official_lafan_readme": "download/official/LAFAN1_Retargeting_Dataset/README.md",
    "official_lafan_infos": "download/official/LAFAN1_Retargeting_Dataset/dataset_infos.json",
    "official_whole_body_tracking": "download/official/whole_body_tracking/README.md",
    "official_motion_tracking_controller": "download/official/motion_tracking_controller/README.md",
    "official_unitree_description": "download/official/unitree_description.tar.gz",
    "isaaclab_version": "download/dependencies/IsaacLab-v2.1.0/VERSION",
    "rsl_rl_setup": "download/dependencies/rsl_rl/setup.py",
    "unitree_bringup_package": "download/dependencies/unitree_bringup/package.xml",
    "supplemental_manifest": "download/_supplemental/supplemental_downloads.tsv",
}

REFERENCE_REPOS = [
    "download/reference_code/ASAP/README.md",
    "download/reference_code/GMR/README.md",
    "download/reference_code/PBHC/README.md",
    "download/reference_code/diffuser/README.md",
    "download/reference_code/diffusion-motion-inbetweening/README.md",
    "download/reference_code/guided-motion-diffusion/README.md",
    "download/reference_code/latent-diffusion/README.md",
    "download/reference_code/motion-diffusion-model/README.md",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(rel: Path) -> str:
    parts = rel.parts
    if not parts:
        return "download_root"
    if parts[0] == "papers":
        return "paper_source"
    if parts[0] == "manifests":
        return "download_manifest"
    if parts[0] == "official":
        if len(parts) > 1 and parts[1] in {"Dataset_beyondmimic.zip", "unitree_description.tar.gz"}:
            return "official_archive"
        return "official_code_or_dataset"
    if parts[0] == "dependencies":
        return "dependency_source"
    if parts[0] == "reference_code":
        return "reference_code"
    if parts[0] == "_supplemental":
        return "supplemental"
    if rel.name.endswith(".log"):
        return "download_log"
    if rel.name.startswith("README"):
        return "download_doc"
    return "other"


def normalize_manifest_path(raw_path: str) -> str:
    """Normalize Windows or POSIX manifest paths to ``download/...`` form."""
    normalized = raw_path.replace("\\", "/").lstrip("\ufeff")
    marker = "/download/"
    if marker in normalized:
        normalized = "download/" + normalized.split(marker, 1)[1]
    elif normalized.startswith("download/"):
        pass
    else:
        normalized = f"download/{Path(normalized).name}"
    return normalized


def read_download_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for idx, row in enumerate(reader, start=1):
            rel = normalize_manifest_path(row.get("path", ""))
            try:
                size = int(row.get("bytes", "0") or 0)
            except ValueError:
                size = 0
            download_rel = rel[len("download/") :] if rel.startswith("download/") else rel
            rows.append(
                {
                    "index": idx,
                    "relative_path": rel,
                    "download_relative_path": download_rel,
                    "category": classify(Path(download_rel)),
                    "top_level": Path(download_rel).parts[0] if Path(download_rel).parts else ".",
                    "extension": Path(download_rel).suffix.lower() or "<no_ext>",
                    "size_bytes": size,
                }
            )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_file = DOWNLOAD / "manifests/downloaded_files.tsv"
    rows = read_download_manifest(manifest_file)
    category_counts: Counter[str] = Counter()
    extension_counts: Counter[str] = Counter()
    top_level_counts: Counter[str] = Counter()
    total_bytes = 0

    for row in rows:
        category_counts[row["category"]] += 1
        extension_counts[row["extension"]] += 1
        top_level_counts[row["top_level"]] += 1
        total_bytes += row["size_bytes"]

    required_rows = []
    for key, rel in REQUIRED_PATHS.items():
        path = ROOT / rel
        required_rows.append(
            {
                "key": key,
                "relative_path": rel,
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else 0,
                "sha256": sha256_file(path) if path.is_file() else "",
            }
        )

    reference_rows = []
    for rel in REFERENCE_REPOS:
        path = ROOT / rel
        reference_rows.append(
            {
                "relative_path": rel,
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else 0,
                "sha256": sha256_file(path) if path.is_file() else "",
            }
        )

    missing_required = [row for row in required_rows if not row["exists"]]
    missing_reference = [row for row in reference_rows if not row["exists"]]
    manifest_rows = len(rows)
    manifest_paths = {row["relative_path"] for row in rows}

    summary: dict[str, Any] = {
        "status": "ok" if not missing_required and not missing_reference and len(rows) > 0 else "failed",
        "experiment_type": "download_source_integrity_audit",
        "scope": "read-only inventory and SHA256 manifest for the raw download bundle",
        "download_root": str(DOWNLOAD),
        "file_count": len(rows),
        "total_size_bytes": total_bytes,
        "inventory_source": str(manifest_file),
        "required_hash_file_count": sum(1 for row in required_rows if row["sha256"]),
        "reference_hash_file_count": sum(1 for row in reference_rows if row["sha256"]),
        "downloaded_files_manifest_row_count": manifest_rows,
        "category_counts": dict(sorted(category_counts.items())),
        "extension_counts": dict(sorted(extension_counts.items())),
        "top_level_counts": dict(sorted(top_level_counts.items())),
        "required_rows": required_rows,
        "missing_required_rows": missing_required,
        "reference_rows": reference_rows,
        "missing_reference_rows": missing_reference,
        "largest_files": sorted(
            [
                {
                    "relative_path": row["relative_path"],
                    "category": row["category"],
                    "size_bytes": row["size_bytes"],
                }
                for row in rows
            ],
            key=lambda row: row["size_bytes"],
            reverse=True,
        )[:20],
        "checks": {
            "download_root_exists": DOWNLOAD.is_dir(),
            "downloaded_files_manifest_exists": manifest_file.is_file(),
            "file_manifest_nonempty": len(rows) > 0,
            "all_required_rows_have_sha256": all(bool(row["sha256"]) for row in required_rows if row["exists"]),
            "all_reference_rows_have_sha256": all(bool(row["sha256"]) for row in reference_rows if row["exists"]),
            "required_paths_listed_in_download_manifest": all(row["relative_path"] in manifest_paths for row in required_rows),
            "required_paths_exist": not missing_required,
            "reference_repos_present": not missing_reference,
            "paper_pdf_and_source_present": all(
                row["exists"] for row in required_rows if row["key"] in {"paper_pdf", "paper_source_tar"}
            ),
            "official_dataset_and_code_present": all(
                row["exists"]
                for row in required_rows
                if row["key"]
                in {
                    "official_dataset_zip",
                    "official_lafan_readme",
                    "official_lafan_infos",
                    "official_whole_body_tracking",
                    "official_motion_tracking_controller",
                    "official_unitree_description",
                }
            ),
            "dependency_snapshots_present": all(
                row["exists"]
                for row in required_rows
                if row["key"] in {"isaaclab_version", "rsl_rl_setup", "unitree_bringup_package"}
            ),
            "download_manifests_present": all(
                row["exists"]
                for row in required_rows
                if row["key"]
                in {
                    "downloaded_files_manifest",
                    "git_revisions_manifest",
                    "resource_sources_manifest",
                    "zenodo_metadata",
                    "supplemental_manifest",
                }
            ),
            "does_not_modify_raw_downloads": True,
            "does_not_claim_training_or_deployment": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "This audit strengthens provenance for the local paper/source/download bundle. It does not supply "
                "missing trained checkpoints, Level C official training code, Fig. 5/6 rollout data, videos, "
                "TensorRT engines, ROS deployment evidence, or hardware results."
            ),
        },
        "outputs": {
            "json": str(OUT / "download_source_integrity_audit.json"),
            "tsv": str(OUT / "download_source_integrity_manifest.tsv"),
            "required_tsv": str(OUT / "download_source_integrity_required.tsv"),
            "markdown": str(OUT / "download_source_integrity_audit.md"),
        },
    }

    (OUT / "download_source_integrity_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (OUT / "download_source_integrity_manifest.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=[
                "index",
                "relative_path",
                "download_relative_path",
                "category",
                "top_level",
                "extension",
                "size_bytes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "download_source_integrity_required.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["key", "relative_path", "exists", "size_bytes", "sha256"])
        writer.writeheader()
        writer.writerows(required_rows)

    lines = [
        "# Download Source Integrity Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Files: `{summary['file_count']}`",
        f"- Total size bytes: `{summary['total_size_bytes']}`",
        f"- Missing required rows: `{len(missing_required)}`",
        f"- Missing reference rows: `{len(missing_reference)}`",
        f"- Download manifest rows: `{manifest_rows}`",
        "",
        "## Category Counts",
    ]
    for key, value in sorted(category_counts.items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Outputs"])
    for key, value in summary["outputs"].items():
        lines.append(f"- `{key}`: `{value}`")
    (OUT / "download_source_integrity_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "status": summary["status"],
                "files": summary["file_count"],
                "total_size_bytes": summary["total_size_bytes"],
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
