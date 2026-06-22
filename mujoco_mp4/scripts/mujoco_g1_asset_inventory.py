#!/usr/bin/env python3
"""Inventory Unitree G1 MuJoCo/MJCF/URDF assets for the MuJoCo MP4 route."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mujoco_common import PKG, ROOT, sha256, utc_now, write_json, write_tsv


CANDIDATE_ROOTS = [
    ROOT / "download/reference_code/GMR/assets/unitree_g1",
    ROOT / "download/reference_code/PBHC/description/robots/g1",
    ROOT / "download/reference_code/unitree_rl_mjlab/src/assets/robots/unitree_g1",
    ROOT / "download/reference_code/ASAP/humanoidverse/data/robots/g1",
    ROOT / "download/official/LAFAN1_Retargeting_Dataset/robot_description/g1",
]


def classify(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith(".xml") or lower.endswith(".mjcf"):
        return "mjcf_xml"
    if lower.endswith(".urdf"):
        return "urdf"
    if lower.endswith((".stl", ".obj", ".dae", ".ply")):
        return "mesh"
    if lower.endswith((".yaml", ".yml", ".json", ".npy")):
        return "metadata"
    return "other"


def main() -> None:
    out_dir = PKG / "res/g1_import"
    work_dir = PKG / "assets/work_g1"
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for root in CANDIDATE_ROOTS:
        if not root.exists():
            rows.append(
                {
                    "source_path": str(root),
                    "exists": False,
                    "category": "missing_root",
                    "size": 0,
                    "sha256": "",
                    "recommended": False,
                    "notes": "candidate root missing",
                }
            )
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            category = classify(path)
            if category == "other":
                continue
            recommended = path.name in {
                "g1_mocap_29dof.xml",
                "g1_mocap_29dof_with_hands.xml",
                "g1_29dof_rev_1_0.xml",
                "scene_g1.xml",
                "g1.xml",
            }
            rows.append(
                {
                    "source_path": str(path),
                    "exists": True,
                    "category": category,
                    "size": path.stat().st_size,
                    "sha256": sha256(path),
                    "recommended": recommended,
                    "notes": "preferred_xml_candidate" if recommended and category == "mjcf_xml" else "",
                }
            )

    copy_roots = [
        ("gmr_unitree_g1", ROOT / "download/reference_code/GMR/assets/unitree_g1"),
        ("pbhc_g1", ROOT / "download/reference_code/PBHC/description/robots/g1"),
        ("unitree_rl_mjlab_unitree_g1", ROOT / "download/reference_code/unitree_rl_mjlab/src/assets/robots/unitree_g1"),
    ]
    copied: list[str] = []
    for dst_name, source_root in copy_roots:
        if not source_root.is_dir():
            continue
        dst = work_dir / dst_name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(source_root, dst)
        copied.append(str(dst))

    payload = {
        "status": "ok",
        "timestamp_utc": utc_now(),
        "experiment_type": "mujoco_g1_asset_inventory",
        "claim_level": "asset inventory only; not simulation evidence",
        "candidate_roots": [str(p) for p in CANDIDATE_ROOTS],
        "work_copies": copied,
        "row_count": len(rows),
        "recommended_xml_count": sum(1 for row in rows if row["recommended"] and row["category"] == "mjcf_xml"),
        "category_counts": {cat: sum(1 for row in rows if row["category"] == cat) for cat in sorted({str(r["category"]) for r in rows})},
        "rows": rows,
    }
    write_json(out_dir / "mujoco_g1_asset_inventory.json", payload)
    write_tsv(
        out_dir / "mujoco_g1_asset_inventory.tsv",
        rows,
        ["source_path", "exists", "category", "size", "sha256", "recommended", "notes"],
    )
    print(json.dumps({"status": "ok", "rows": len(rows), "recommended_xml": payload["recommended_xml_count"]}))


if __name__ == "__main__":
    main()
