#!/usr/bin/env python3
"""Inventory motion-tracking sources required by whole_body_tracking README."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/download_audit/motion_tracking_required_sources_inventory"
JSON_OUT = OUT / "motion_tracking_required_sources_inventory.json"
TSV_OUT = OUT / "motion_tracking_required_sources_inventory.tsv"


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(path: Path) -> dict[str, Any]:
    if not (path / ".git").exists():
        return {}

    def run(args: list[str]) -> str:
        proc = subprocess.run(args, cwd=path, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        return proc.stdout.strip()

    return {
        "remote": run(["git", "remote", "get-url", "origin"]),
        "commit": run(["git", "rev-parse", "HEAD"]),
    }


def file_row(
    name: str,
    category: str,
    path: Path,
    source_url: str,
    purpose: str,
    status: str,
    notes: str,
    required_or_optional: str = "required_by_readme_or_paper_table",
    conversion_status: str = "not_applicable",
) -> dict[str, Any]:
    return {
        "name": name,
        "category": category,
        "path": str(path),
        "exists": path.exists(),
        "file_size": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path),
        "source_url": source_url,
        "purpose": purpose,
        "status": status,
        "required_or_optional": required_or_optional,
        "conversion_status": conversion_status,
        "claim_level": "downloaded_or_local_source_inventory_not_training_result",
        "notes": notes,
    }


def repo_row(name: str, path: Path, source_url: str, purpose: str, status: str, notes: str) -> dict[str, Any]:
    info = git_info(path)
    return {
        "name": name,
        "category": "source_repo",
        "path": str(path),
        "exists": path.exists(),
        "file_size": None,
        "sha256": None,
        "source_url": source_url,
        "purpose": purpose,
        "status": status,
        "required_or_optional": "required_by_readme_or_related_reference",
        "conversion_status": "not_applicable",
        "claim_level": "downloaded_source_code_inventory_not_training_result",
        "notes": notes,
        "git_remote": info.get("remote"),
        "git_commit": info.get("commit"),
    }


def count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    rows.extend(
        [
            repo_row(
                "HybridRobotics/whole_body_tracking",
                ROOT / "download/official/whole_body_tracking",
                "https://github.com/HybridRobotics/whole_body_tracking",
                "Official BeyondMimic motion-tracking training code.",
                "available",
                "Editable work copy is under reproduction/third_party/official/whole_body_tracking.",
            ),
            repo_row(
                "HybridRobotics/motion_tracking_controller",
                ROOT / "download/official/motion_tracking_controller",
                "https://github.com/HybridRobotics/motion_tracking_controller",
                "Sim-to-sim/sim-to-real deployment reference linked by whole_body_tracking README.",
                "available",
                "Deployment reference only; not used by the current PPO training run.",
            ),
            repo_row(
                "mujocolab/mjlab",
                ROOT / "download/reference_code/mjlab",
                "https://github.com/mujocolab/mjlab",
                "Alternative MuJoCo-Warp implementation referenced by the README.",
                "available",
                "Reference implementation; not official IsaacLab training evidence.",
            ),
            repo_row(
                "TeleHuman/PBHC",
                ROOT / "download/reference_code/PBHC",
                "https://github.com/TeleHuman/PBHC",
                "KungfuBot/PBHC reference containing side-kick example motion.",
                "available",
                "Contains example/motion_data/Side_kick.pkl; not yet converted to whole_body_tracking CSV/NPZ.",
            ),
            repo_row(
                "LeCAR-Lab/ASAP",
                ROOT / "download/reference_code/ASAP",
                "https://github.com/LeCAR-Lab/ASAP",
                "ASAP reference containing CR7 motion assets.",
                "available",
                "Contains G1 29DoF CR7 pickle files; not yet converted to whole_body_tracking CSV/NPZ.",
            ),
        ]
    )

    lafan = ROOT / "download/official/LAFAN1_Retargeting_Dataset"
    rows.append(
        repo_row(
            "Unitree-retargeted LAFAN1 Dataset",
            lafan,
            "https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset",
            "Retargeted G1/H1/H1_2 reference motion CSVs recommended by whole_body_tracking README.",
            "available",
            f"G1 csv count={count_files(lafan / 'g1', '*.csv')}; currently used by tracking PPO after local NPZ conversion.",
        )
    )
    rows.append(
        repo_row(
            "Ubisoft LaForge LAFAN1 original dataset",
            ROOT / "download/official/ubisoft-laforge-animation-dataset",
            "https://github.com/ubisoft/ubisoft-laforge-animation-dataset",
            "Original LAFAN1 source dataset for provenance and Level-C public-data experiments.",
            "available",
            "Original animation dataset; not directly consumed by Tracking-Flat-G1-v0 without retargeting.",
        )
    )

    rows.extend(
        [
            file_row(
                "KungfuBot side kick motion",
                "motion_file",
                ROOT / "download/reference_code/PBHC/example/motion_data/Side_kick.pkl",
                "https://kungfu-bot.github.io/ and https://github.com/TeleHuman/PBHC",
                "Side-kick short-sequence motion source referenced by whole_body_tracking README and paper table.",
                "available",
                "joblib-readable dict with root_trans_offset/pose_aa/dof/root_rot/smpl_joints/fps/contact_mask.",
                conversion_status="needs_conversion_to_whole_body_tracking_csv_or_npz",
            ),
            file_row(
                "ASAP Cristiano Ronaldo CR7 motion",
                "motion_file",
                ROOT
                / "download/reference_code/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/"
                "TairanTestbed/singles/0-TairanTestbed_TairanTestbed_CR7_video_CR7_level1_filter_amass.pkl",
                "https://github.com/LeCAR-Lab/ASAP",
                "Cristiano Ronaldo celebration short-sequence motion source referenced by whole_body_tracking README.",
                "available",
                "joblib-readable dict with root_trans_offset/pose_aa/dof/root_rot/smpl_joints/fps.",
                conversion_status="needs_conversion_to_whole_body_tracking_csv_or_npz",
            ),
            file_row(
                "HuB single-leg balance motion",
                "motion_file",
                ROOT / "download/_supplemental/hub_data/drive_folder/singleleg.pkl",
                "https://hub-robot.github.io/ and Google Drive folder 1ZrF8HFMzJ7jBcHP6qFKgCyErvFDWlkmc",
                "Single-leg balance source referenced by whole_body_tracking README/paper table.",
                "available",
                "Downloaded from the HuB project dataset link; joblib-readable dict.",
                conversion_status="needs_conversion_to_whole_body_tracking_csv_or_npz",
            ),
            file_row(
                "HuB swallow balance motion",
                "motion_file",
                ROOT / "download/_supplemental/hub_data/drive_folder/swallow_balance.pkl",
                "https://hub-robot.github.io/ and Google Drive folder 1ZrF8HFMzJ7jBcHP6qFKgCyErvFDWlkmc",
                "Swallow balance source referenced by whole_body_tracking README/paper table.",
                "available",
                "Downloaded from the HuB project dataset link; joblib-readable dict.",
                conversion_status="needs_conversion_to_whole_body_tracking_csv_or_npz",
            ),
            file_row(
                "HuB Bruce Lee kick motion",
                "motion_file",
                ROOT / "download/_supplemental/hub_data/drive_folder/bruce_lee.pkl",
                "https://hub-robot.github.io/ and Google Drive folder 1ZrF8HFMzJ7jBcHP6qFKgCyErvFDWlkmc",
                "Additional HuB extreme balance/kick motion source.",
                "available",
                "Downloaded as part of HuB dataset folder.",
                conversion_status="needs_conversion_to_whole_body_tracking_csv_or_npz",
            ),
        ]
    )

    summary = {
        "status": "ok_motion_tracking_required_sources_inventory",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "Inventory of source code and motion datasets named by HybridRobotics/whole_body_tracking README. "
            "This verifies download/local availability only; it does not claim that non-LAFAN pkl motions have been "
            "converted into registry-ready whole_body_tracking NPZs."
        ),
        "rows": rows,
        "counts": {
            "row_count": len(rows),
            "available_count": sum(1 for r in rows if r["exists"]),
            "needs_conversion_count": sum(
                1 for r in rows if r.get("conversion_status") == "needs_conversion_to_whole_body_tracking_csv_or_npz"
            ),
            "unitree_lafan_g1_csv_count": count_files(lafan / "g1", "*.csv"),
            "asap_g1_29dof_single_pkl_count": count_files(
                ROOT
                / "download/reference_code/ASAP/humanoidverse/data/motions/g1_29dof_anneal_23dof/"
                "TairanTestbed/singles",
                "*.pkl",
            ),
            "pbhc_example_motion_pkl_count": count_files(
                ROOT / "download/reference_code/PBHC/example/motion_data", "*.pkl"
            ),
            "hub_downloaded_pkl_count": count_files(ROOT / "download/_supplemental/hub_data/drive_folder", "*.pkl"),
        },
        "claim_boundary": {
            "download_complete_for_named_public_sources": True,
            "all_named_short_sequence_sources_available": True,
            "converted_to_whole_body_tracking_registry_npz": False,
            "paper_level_motion_tracking_dataset": False,
            "real_robot_data": False,
        },
        "next_step": (
            "Build explicit converters from PBHC/ASAP/HuB joblib motion dicts to the generalized-coordinate CSV "
            "convention expected by whole_body_tracking/scripts/csv_to_npz.py, then run conversion/replay gates before "
            "adding them to PPO training."
        ),
    }

    JSON_OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with TSV_OUT.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "name",
            "category",
            "path",
            "exists",
            "file_size",
            "sha256",
            "source_url",
            "purpose",
            "status",
            "required_or_optional",
            "conversion_status",
            "claim_level",
            "notes",
            "git_remote",
            "git_commit",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(json.dumps({"status": summary["status"], "json": str(JSON_OUT), "tsv": str(TSV_OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
