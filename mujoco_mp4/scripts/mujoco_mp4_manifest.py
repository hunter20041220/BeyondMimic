#!/usr/bin/env python3
"""Build a local manifest for the independent MuJoCo MP4 package."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mujoco_common import PKG, ROOT, sha256, utc_now, write_json, write_tsv


SKIP_DIRS = {
    ".venv",
    "cache",
    "__pycache__",
}


def category_for(path: Path) -> str:
    rel = path.relative_to(PKG)
    parts = rel.parts
    suffix = path.suffix.lower()
    if parts[0] == "scripts":
        return "script"
    if parts[0] == "configs":
        return "config"
    if parts[0] == "assets":
        return "asset"
    if parts[0] == "logs":
        return "log"
    if parts[0] == "docs" or suffix == ".md":
        return "doc"
    if suffix == ".mp4":
        return "video"
    if suffix in {".png", ".jpg", ".jpeg"}:
        return "metric"
    if suffix in {".csv", ".tsv", ".json"}:
        return "metric"
    if path.name == "requirements-lock.txt":
        return "env"
    return "helper"


def backend_for(path: Path) -> str:
    name = path.name.lower()
    text = str(path.relative_to(PKG)).lower()
    if "_egl" in name or "/egl" in text or "backend_probe_egl" in text:
        return "egl"
    if "_osmesa" in name or "/osmesa" in text or "backend_probe_osmesa" in text:
        return "osmesa"
    if "_glfw" in name or "/glfw" in text or "backend_probe_glfw" in text:
        return "glfw"
    return "not_applicable"


def claim_for(path: Path, category: str) -> str:
    rel = str(path.relative_to(PKG))
    if "reference_replay" in rel:
        return "MuJoCo reference replay visualization; not policy closed-loop, not IsaacLab, not real robot"
    if "g1_import" in rel or "work_g1" in rel:
        return "MuJoCo G1 asset import/render smoke support; not policy, not IsaacLab, not real robot"
    if "smoke" in rel:
        return "MuJoCo backend/minimal rendering smoke; not robot policy or paper-level result"
    if category in {"script", "config", "doc", "env"}:
        return "Reproduction engineering support artifact"
    return "Local MuJoCo MP4 package artifact"


def source_dependency_for(path: Path) -> str:
    rel = str(path.relative_to(PKG))
    if "work_g1/gmr_unitree_g1" in rel:
        return str(ROOT / "download/reference_code/GMR/assets/unitree_g1")
    if "work_g1/pbhc_g1" in rel:
        return str(ROOT / "download/reference_code/PBHC/description/robots/g1")
    if "work_g1/unitree_rl_mjlab_unitree_g1" in rel:
        return str(ROOT / "download/reference_code/unitree_rl_mjlab/src/assets/robots/unitree_g1")
    if "reference_replay" in rel:
        return str(ROOT / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz")
    return ""


def purpose_for(path: Path, category: str) -> str:
    rel = str(path.relative_to(PKG))
    if path.name == "README_MUJOCO_MP4_H20.md":
        return "Operator documentation for MuJoCo MP4 route on H20"
    if path.name == "requirements-lock.txt":
        return "Python dependency lock from the independent MuJoCo venv"
    if path.name.endswith("_probe.py"):
        return "MuJoCo rendering backend probe script"
    if path.name == "mujoco_minimal_video_smoke.py":
        return "Minimal offscreen rendered MP4 smoke script"
    if path.name == "mujoco_g1_asset_inventory.py":
        return "G1 MuJoCo asset inventory and work-copy builder"
    if path.name == "mujoco_g1_import_smoke.py":
        return "G1 MJCF import and offscreen render smoke script"
    if path.name == "mujoco_reference_replay_video.py":
        return "Render local G1 reference motion as MuJoCo mesh replay"
    if rel.startswith("assets/work_g1"):
        return "Local work copy of G1 MuJoCo robot assets"
    if category == "video":
        return "Local MP4 evidence for MuJoCo rendering/replay"
    if category == "metric":
        return "Machine-readable output, keyframe, or metrics for MuJoCo MP4 route"
    if category == "log":
        return "Command output log for setup or smoke execution"
    return "MuJoCo MP4 package artifact"


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in sorted(PKG.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(PKG)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if rel.name in {"mujoco_mp4_manifest.json", "mujoco_mp4_manifest.tsv"}:
            continue
        files.append(path)
    return files


def main() -> None:
    rows = []
    for path in iter_files():
        rel = path.relative_to(PKG)
        category = category_for(path)
        size = path.stat().st_size
        rows.append(
            {
                "path": str(path),
                "relative_path": str(rel),
                "exists": True,
                "file_size": size,
                "sha256": sha256(path),
                "category": category,
                "purpose": purpose_for(path, category),
                "backend": backend_for(path),
                "claim_level": claim_for(path, category),
                "source_dependency": source_dependency_for(path),
                "large_file": size > 50 * 1024 * 1024,
                "notes": "excluded from git by package policy" if category in {"video", "asset", "log"} else "",
            }
        )

    payload = {
        "status": "ok",
        "timestamp_utc": utc_now(),
        "package_root": str(PKG),
        "experiment_type": "mujoco_mp4_manifest",
        "claim_level": "Manifest for MuJoCo local virtual simulation evidence package",
        "row_count": len(rows),
        "category_counts": {cat: sum(1 for row in rows if row["category"] == cat) for cat in sorted({str(r["category"]) for r in rows})},
        "backend_counts": {backend: sum(1 for row in rows if row["backend"] == backend) for backend in sorted({str(r["backend"]) for r in rows})},
        "rows": rows,
    }
    write_json(PKG / "mujoco_mp4_manifest.json", payload)
    write_tsv(
        PKG / "mujoco_mp4_manifest.tsv",
        rows,
        [
            "path",
            "relative_path",
            "exists",
            "file_size",
            "sha256",
            "category",
            "purpose",
            "backend",
            "claim_level",
            "source_dependency",
            "large_file",
            "notes",
        ],
    )
    print(json.dumps({"status": "ok", "rows": len(rows), "manifest": str(PKG / "mujoco_mp4_manifest.json")}))


if __name__ == "__main__":
    main()
