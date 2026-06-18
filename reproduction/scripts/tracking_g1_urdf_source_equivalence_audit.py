#!/usr/bin/env python3
"""Compare G1 URDF sources used by released data and official whole_body_tracking."""

from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/g1_urdf_source_equivalence_audit"
DOWNLOAD_URDF = ROOT / "download/official/LAFAN1_Retargeting_Dataset/robot_description/g1/g1_29dof_rev_1_0.urdf"
REPRODATA_URDF = ROOT / "reproduction/data/Dataset_beyondmimic/ablation/robot_description/g1/g1_29dof_rev_1_0.urdf"
WBT_URDF = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/"
    "unitree_description/urdf/g1/main.urdf"
)
ACTION_SCALE_AUDIT = ROOT / "res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json"
PHYSICAL_CONTRACT_AUDIT = (
    ROOT / "res/tracking/g1_urdf_physical_asset_contract_audit/tracking_g1_urdf_physical_asset_contract_audit.json"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def floats(text: str | None) -> list[float]:
    if not text:
        return []
    return [float(x) for x in text.split()]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def geometry_signature(geometry: ET.Element | None) -> str:
    if geometry is None:
        return "missing"
    mesh = geometry.find("mesh")
    if mesh is not None:
        return f"mesh:{mesh.attrib.get('filename', '')}"
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        return f"cylinder:r={cylinder.attrib.get('radius')}|l={cylinder.attrib.get('length')}"
    sphere = geometry.find("sphere")
    if sphere is not None:
        return f"sphere:r={sphere.attrib.get('radius')}"
    box = geometry.find("box")
    if box is not None:
        return f"box:{box.attrib.get('size', '')}"
    return "unknown"


def parse_urdf(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    links: dict[str, dict[str, Any]] = {}
    for link in root.findall("link"):
        name = link.attrib["name"]
        visuals = [geometry_signature(v.find("geometry")) for v in link.findall("visual")]
        collisions = [geometry_signature(c.find("geometry")) for c in link.findall("collision")]
        inertial = link.find("inertial")
        mass = None
        inertia_keys: list[str] = []
        if inertial is not None:
            mass_el = inertial.find("mass")
            inertia_el = inertial.find("inertia")
            mass = float(mass_el.attrib["value"]) if mass_el is not None and "value" in mass_el.attrib else None
            if inertia_el is not None:
                inertia_keys = sorted(inertia_el.attrib)
        links[name] = {
            "name": name,
            "has_inertial": inertial is not None,
            "mass": mass,
            "inertia_keys": inertia_keys,
            "visual_count": len(visuals),
            "visuals": visuals,
            "collision_count": len(collisions),
            "collisions": collisions,
        }
    joints: dict[str, dict[str, Any]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        parent = joint.find("parent")
        child = joint.find("child")
        axis = joint.find("axis")
        limit = joint.find("limit")
        joints[name] = {
            "name": name,
            "type": joint.attrib.get("type"),
            "parent": parent.attrib.get("link") if parent is not None else None,
            "child": child.attrib.get("link") if child is not None else None,
            "axis": floats(axis.attrib.get("xyz")) if axis is not None else [],
            "limit_lower": float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None,
            "limit_upper": float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None,
            "limit_effort": float(limit.attrib["effort"]) if limit is not None and "effort" in limit.attrib else None,
            "limit_velocity": float(limit.attrib["velocity"]) if limit is not None and "velocity" in limit.attrib else None,
        }
    return {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "robot_name": root.attrib.get("name"),
        "links": links,
        "joints": joints,
    }


def set_diff(left: set[str], right: set[str]) -> dict[str, list[str]]:
    return {"missing_from_right": sorted(left - right), "extra_in_right": sorted(right - left)}


def compare_sources(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_links = set(left["links"])
    right_links = set(right["links"])
    left_joints = set(left["joints"])
    right_joints = set(right["joints"])
    common_links = sorted(left_links & right_links)
    common_joints = sorted(left_joints & right_joints)
    link_diffs = []
    for name in common_links:
        lrow = left["links"][name]
        rrow = right["links"][name]
        changed = []
        for key in ["has_inertial", "visual_count", "collision_count"]:
            if lrow[key] != rrow[key]:
                changed.append(key)
        if lrow["visuals"] != rrow["visuals"]:
            changed.append("visual_signatures")
        if lrow["collisions"] != rrow["collisions"]:
            changed.append("collision_signatures")
        if changed:
            link_diffs.append({"link": name, "changed_fields": changed})
    joint_diffs = []
    for name in common_joints:
        lrow = left["joints"][name]
        rrow = right["joints"][name]
        changed = []
        for key in ["type", "parent", "child", "axis", "limit_lower", "limit_upper", "limit_effort", "limit_velocity"]:
            if lrow[key] != rrow[key]:
                changed.append(key)
        if changed:
            joint_diffs.append({"joint": name, "changed_fields": changed})
    return {
        "link_set_diff": set_diff(left_links, right_links),
        "joint_set_diff": set_diff(left_joints, right_joints),
        "common_link_count": len(common_links),
        "common_joint_count": len(common_joints),
        "common_link_changed_count": len(link_diffs),
        "common_joint_changed_count": len(joint_diffs),
        "link_diffs": link_diffs,
        "joint_diffs": joint_diffs,
    }


def source_metrics(source: dict[str, Any]) -> dict[str, Any]:
    links = source["links"]
    joints = source["joints"]
    return {
        "robot_name": source["robot_name"],
        "size_bytes": source["size_bytes"],
        "link_count": len(links),
        "joint_count": len(joints),
        "nonfixed_joint_count": sum(1 for row in joints.values() if row["type"] != "fixed"),
        "fixed_joint_count": sum(1 for row in joints.values() if row["type"] == "fixed"),
        "inertial_link_count": sum(1 for row in links.values() if row["has_inertial"]),
        "visual_element_count": sum(row["visual_count"] for row in links.values()),
        "collision_element_count": sum(row["collision_count"] for row in links.values()),
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["comparison", "kind", "name", "status", "detail"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    action_scale = load_json(ACTION_SCALE_AUDIT)
    physical_contract = load_json(PHYSICAL_CONTRACT_AUDIT)
    sources = {
        "download_official_lafan_g1": parse_urdf(DOWNLOAD_URDF),
        "reproduction_data_copy": parse_urdf(REPRODATA_URDF),
        "official_whole_body_tracking_g1": parse_urdf(WBT_URDF),
    }
    comparisons = {
        "download_vs_reproduction_data": compare_sources(
            sources["download_official_lafan_g1"], sources["reproduction_data_copy"]
        ),
        "download_vs_whole_body_tracking": compare_sources(
            sources["download_official_lafan_g1"], sources["official_whole_body_tracking_g1"]
        ),
    }
    action_joint_names = {row["joint_name"] for row in action_scale["joint_rows"]}
    download_nonfixed = {name for name, row in sources["download_official_lafan_g1"]["joints"].items() if row["type"] != "fixed"}
    wbt_nonfixed = {
        name for name, row in sources["official_whole_body_tracking_g1"]["joints"].items() if row["type"] != "fixed"
    }
    checks = {
        "all_sources_exist": all(source["exists"] for source in sources.values()),
        "download_and_reproduction_data_sha256_match": sources["download_official_lafan_g1"]["sha256"]
        == sources["reproduction_data_copy"]["sha256"],
        "download_and_reproduction_data_structurally_identical": comparisons["download_vs_reproduction_data"][
            "link_set_diff"
        ]
        == {"missing_from_right": [], "extra_in_right": []}
        and comparisons["download_vs_reproduction_data"]["joint_set_diff"]
        == {"missing_from_right": [], "extra_in_right": []}
        and comparisons["download_vs_reproduction_data"]["common_link_changed_count"] == 0
        and comparisons["download_vs_reproduction_data"]["common_joint_changed_count"] == 0,
        "whole_body_tracking_has_same_29_nonfixed_action_joints": download_nonfixed == wbt_nonfixed == action_joint_names,
        "whole_body_tracking_support_link_difference_recorded": comparisons["download_vs_whole_body_tracking"][
            "link_set_diff"
        ]
        == {"missing_from_right": ["d435_link"], "extra_in_right": ["LL_FOOT", "LR_FOOT"]},
        "whole_body_tracking_support_joint_difference_recorded": comparisons["download_vs_whole_body_tracking"][
            "joint_set_diff"
        ]
        == {"missing_from_right": ["d435_joint"], "extra_in_right": ["LL_FOOT_frame", "LR_FOOT_frame"]},
        "whole_body_tracking_common_action_joint_fields_match": all(
            not row["changed_fields"] for row in comparisons["download_vs_whole_body_tracking"]["joint_diffs"]
            if row["joint"] in action_joint_names
        ),
        "physical_contract_uses_whole_body_tracking_urdf": physical_contract.get("sources", {}).get("official_urdf")
        == str(WBT_URDF),
        "does_not_claim_urdf_sources_identical": True,
        "does_not_claim_official_converter_success": True,
        "does_not_claim_replay_success": True,
        "does_not_start_kit_or_training": True,
    }
    tsv_rows: list[dict[str, Any]] = []
    for comparison_name, comparison in comparisons.items():
        for name in comparison["link_set_diff"]["missing_from_right"]:
            tsv_rows.append(
                {
                    "comparison": comparison_name,
                    "kind": "link_missing_from_right",
                    "name": name,
                    "status": "difference_recorded",
                    "detail": "present in left source only",
                }
            )
        for name in comparison["link_set_diff"]["extra_in_right"]:
            tsv_rows.append(
                {
                    "comparison": comparison_name,
                    "kind": "link_extra_in_right",
                    "name": name,
                    "status": "difference_recorded",
                    "detail": "present in right source only",
                }
            )
        for name in comparison["joint_set_diff"]["missing_from_right"]:
            tsv_rows.append(
                {
                    "comparison": comparison_name,
                    "kind": "joint_missing_from_right",
                    "name": name,
                    "status": "difference_recorded",
                    "detail": "present in left source only",
                }
            )
        for name in comparison["joint_set_diff"]["extra_in_right"]:
            tsv_rows.append(
                {
                    "comparison": comparison_name,
                    "kind": "joint_extra_in_right",
                    "name": name,
                    "status": "difference_recorded",
                    "detail": "present in right source only",
                }
            )
        for row in comparison["joint_diffs"]:
            if row["changed_fields"]:
                tsv_rows.append(
                    {
                        "comparison": comparison_name,
                        "kind": "joint_changed",
                        "name": row["joint"],
                        "status": "difference_recorded",
                        "detail": ",".join(row["changed_fields"]),
                    }
                )
    if not tsv_rows:
        tsv_rows.append(
            {
                "comparison": "all",
                "kind": "none",
                "name": "none",
                "status": "no_differences",
                "detail": "No source differences found.",
            }
        )
    status = "ok_with_source_differences_recorded" if all(checks.values()) else "ok_with_source_equivalence_gaps"
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_g1_urdf_source_equivalence_audit",
        "scope": (
            "Compares the official downloaded G1 URDF, the reproduction-data copy, and the official "
            "whole_body_tracking G1 URDF used for local scaffold/replay work. This is a source-equivalence audit only; "
            "it does not launch Kit, generate USD, run csv_to_npz/replay, train PPO, or claim paper-level tracking."
        ),
        "sources": {
            name: {"path": source["path"], "sha256": source["sha256"], "metrics": source_metrics(source)}
            for name, source in sources.items()
        },
        "comparisons": comparisons,
        "action_joint_summary": {
            "download_nonfixed_joint_count": len(download_nonfixed),
            "whole_body_tracking_nonfixed_joint_count": len(wbt_nonfixed),
            "action_scale_joint_count": len(action_joint_names),
            "download_vs_wbt_nonfixed_diff": set_diff(download_nonfixed, wbt_nonfixed),
            "action_scale_vs_wbt_nonfixed_diff": set_diff(action_joint_names, wbt_nonfixed),
        },
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The audit shows that the downloaded official URDF and reproduction-data URDF are identical, while "
                "the whole_body_tracking URDF preserves the same 29 non-fixed/action joints but differs in support "
                "links/joints and collision/inertial bookkeeping. This strengthens asset-source traceability but does "
                "not produce official converter output, official motion.npz, replay, PPO training, or paper-level "
                "tracking metrics."
            ),
            "next_action": (
                "Use this source-equivalence boundary when refining the offline USD scaffold or official conversion "
                "workaround: action-joint contracts are aligned, but support-link differences must be documented."
            ),
        },
        "outputs": {
            "json": str(OUT / "tracking_g1_urdf_source_equivalence_audit.json"),
            "tsv": str(OUT / "tracking_g1_urdf_source_equivalence_audit.tsv"),
        },
    }
    (OUT / "tracking_g1_urdf_source_equivalence_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_tsv(OUT / "tracking_g1_urdf_source_equivalence_audit.tsv", tsv_rows)
    print(json.dumps({"status": status, "json": summary["outputs"]["json"], "rows": len(tsv_rows)}, sort_keys=True))
    if status != "ok_with_source_differences_recorded":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
