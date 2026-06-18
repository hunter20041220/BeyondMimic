#!/usr/bin/env python3
"""Audit the official G1 URDF physical fields needed to enrich the skeleton USD."""

from __future__ import annotations

import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/g1_urdf_physical_asset_contract_audit"
OFFICIAL_URDF = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/"
    "unitree_description/urdf/g1/main.urdf"
)
UNITREE_DESCRIPTION_ROOT = OFFICIAL_URDF.parents[2]
ACTION_SCALE_AUDIT = ROOT / "res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json"
SKELETON_USD_AUDIT = (
    ROOT / "res/tracking/g1_official_urdf_skeleton_usd/tracking_g1_official_urdf_skeleton_usd_audit.json"
)
OFFICIAL_SOURCE_AUDIT = ROOT / "res/tracking/official_source_contract_audit/tracking_official_source_contract_audit.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def floats(text: str | None) -> list[float]:
    if not text:
        return []
    return [float(x) for x in text.split()]


def resolve_mesh(filename: str) -> Path:
    prefix = "package://unitree_description/"
    if filename.startswith(prefix):
        return UNITREE_DESCRIPTION_ROOT / filename[len(prefix) :]
    return (OFFICIAL_URDF.parent / filename).resolve()


def parse_origin(element: ET.Element | None) -> dict[str, list[float]]:
    if element is None:
        return {"xyz": [], "rpy": []}
    return {"xyz": floats(element.attrib.get("xyz")), "rpy": floats(element.attrib.get("rpy"))}


def geometry_kind(geometry: ET.Element | None) -> str:
    if geometry is None:
        return "missing"
    for child in list(geometry):
        return child.tag
    return "empty"


def parse_links(root: ET.Element) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    link_rows: list[dict[str, Any]] = []
    mesh_rows: list[dict[str, Any]] = []
    collision_rows: list[dict[str, Any]] = []
    for link in root.findall("link"):
        link_name = link.attrib["name"]
        inertial = link.find("inertial")
        mass = inertial.find("mass") if inertial is not None else None
        inertia = inertial.find("inertia") if inertial is not None else None
        visual_elements = link.findall("visual")
        collision_elements = link.findall("collision")
        link_rows.append(
            {
                "link_name": link_name,
                "has_inertial": inertial is not None,
                "mass": float(mass.attrib["value"]) if mass is not None and "value" in mass.attrib else None,
                "has_full_inertia_tensor": inertia is not None
                and all(k in inertia.attrib for k in ["ixx", "ixy", "ixz", "iyy", "iyz", "izz"]),
                "visual_count": len(visual_elements),
                "collision_count": len(collision_elements),
            }
        )
        for idx, visual in enumerate(visual_elements):
            mesh = visual.find("geometry/mesh")
            filename = mesh.attrib.get("filename") if mesh is not None else ""
            resolved = resolve_mesh(filename) if filename else None
            mesh_rows.append(
                {
                    "link_name": link_name,
                    "visual_index": idx,
                    "mesh_filename": filename,
                    "resolved_path": str(resolved) if resolved is not None else "",
                    "mesh_exists": bool(resolved and resolved.is_file()),
                    "mesh_size_bytes": resolved.stat().st_size if resolved and resolved.is_file() else None,
                    "origin": parse_origin(visual.find("origin")),
                }
            )
        for idx, collision in enumerate(collision_elements):
            geom = collision.find("geometry")
            collision_rows.append(
                {
                    "link_name": link_name,
                    "collision_index": idx,
                    "geometry_type": geometry_kind(geom),
                    "origin": parse_origin(collision.find("origin")),
                }
            )
    return link_rows, mesh_rows, collision_rows


def parse_joints(root: ET.Element, action_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    action_by_name = {row["joint_name"]: row for row in action_rows}
    rows: list[dict[str, Any]] = []
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        limit = joint.find("limit")
        action = action_by_name.get(name, {})
        rows.append(
            {
                "joint_name": name,
                "joint_type": joint.attrib.get("type"),
                "parent": joint.find("parent").attrib.get("link") if joint.find("parent") is not None else None,
                "child": joint.find("child").attrib.get("link") if joint.find("child") is not None else None,
                "origin": parse_origin(joint.find("origin")),
                "axis": floats(joint.find("axis").attrib.get("xyz")) if joint.find("axis") is not None else [],
                "limit_lower": float(limit.attrib["lower"]) if limit is not None and "lower" in limit.attrib else None,
                "limit_upper": float(limit.attrib["upper"]) if limit is not None and "upper" in limit.attrib else None,
                "limit_effort": float(limit.attrib["effort"]) if limit is not None and "effort" in limit.attrib else None,
                "limit_velocity": float(limit.attrib["velocity"])
                if limit is not None and "velocity" in limit.attrib
                else None,
                "has_action_drive_row": name in action_by_name,
                "actuator_group": action.get("actuator_group"),
                "stiffness": action.get("stiffness"),
                "damping": action.get("damping"),
                "armature": action.get("armature"),
                "action_scale": action.get("action_scale"),
                "effort_limit_sim": action.get("effort_limit_sim"),
                "velocity_limit_sim": action.get("velocity_limit_sim"),
            }
        )
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "kind",
        "name",
        "parent",
        "child",
        "status",
        "detail",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    root = ET.parse(OFFICIAL_URDF).getroot()
    action_scale = load_json(ACTION_SCALE_AUDIT)
    skeleton = load_json(SKELETON_USD_AUDIT)
    official_source = load_json(OFFICIAL_SOURCE_AUDIT)
    link_rows, mesh_rows, collision_rows = parse_links(root)
    joint_rows = parse_joints(root, action_scale["joint_rows"])
    nonfixed_joint_rows = [row for row in joint_rows if row["joint_type"] != "fixed"]
    fixed_joint_rows = [row for row in joint_rows if row["joint_type"] == "fixed"]

    missing_inertial_links = sorted(row["link_name"] for row in link_rows if not row["has_inertial"])
    missing_mesh_refs = sorted(row["mesh_filename"] for row in mesh_rows if not row["mesh_exists"])
    nonfixed_missing_axis = sorted(row["joint_name"] for row in nonfixed_joint_rows if len(row["axis"]) != 3)
    nonfixed_missing_limit = sorted(
        row["joint_name"]
        for row in nonfixed_joint_rows
        if any(row[key] is None for key in ["limit_lower", "limit_upper", "limit_effort", "limit_velocity"])
    )
    nonfixed_missing_drive = sorted(row["joint_name"] for row in nonfixed_joint_rows if not row["has_action_drive_row"])
    target_bodies = official_source["flat_env"]["body_names"]
    target_body_missing_inertial = sorted(set(target_bodies) & set(missing_inertial_links))
    collision_type_counts: dict[str, int] = {}
    for row in collision_rows:
        collision_type_counts[row["geometry_type"]] = collision_type_counts.get(row["geometry_type"], 0) + 1

    checks = {
        "official_urdf_exists": OFFICIAL_URDF.is_file(),
        "official_urdf_hash_recorded": True,
        "link_count_40": len(link_rows) == 40,
        "joint_count_39": len(joint_rows) == 39,
        "nonfixed_joint_count_29": len(nonfixed_joint_rows) == 29,
        "fixed_joint_count_10": len(fixed_joint_rows) == 10,
        "visual_mesh_reference_count_35": len(mesh_rows) == 35,
        "all_visual_mesh_files_exist": len(missing_mesh_refs) == 0,
        "collision_element_count_29": len(collision_rows) == 29,
        "inertial_links_recorded": len(link_rows) - len(missing_inertial_links) == 37,
        "missing_inertial_links_recorded": missing_inertial_links == [
            "imu_in_pelvis",
            "imu_in_torso",
            "mid360_link",
        ],
        "all_nonfixed_joints_have_axis": len(nonfixed_missing_axis) == 0,
        "all_nonfixed_joints_have_limits": len(nonfixed_missing_limit) == 0,
        "all_nonfixed_joints_have_action_drive_rows": len(nonfixed_missing_drive) == 0,
        "skeleton_contract_ok": skeleton.get("skeleton_contract_ok") is True,
        "target_bodies_have_inertial": len(target_body_missing_inertial) == 0,
        "ready_for_offline_converter_scaffold": len(missing_mesh_refs) == 0
        and len(nonfixed_missing_axis) == 0
        and len(nonfixed_missing_limit) == 0
        and len(nonfixed_missing_drive) == 0
        and skeleton.get("skeleton_contract_ok") is True,
        "physical_fidelity_complete_for_replay": False,
        "does_not_generate_physical_usd": True,
        "does_not_claim_motion_npz": True,
        "does_not_claim_replay_success": True,
        "does_not_launch_kit_or_training": True,
    }
    status = "ok_with_physical_contract_ready_for_converter_scaffold" if checks[
        "ready_for_offline_converter_scaffold"
    ] else "ok_with_physical_contract_gaps"
    tsv_rows = []
    for row in link_rows:
        tsv_rows.append(
            {
                "kind": "link",
                "name": row["link_name"],
                "parent": "",
                "child": "",
                "status": "missing_inertial" if not row["has_inertial"] else "ok",
                "detail": f"visual={row['visual_count']} collision={row['collision_count']} mass={row['mass']}",
            }
        )
    for row in nonfixed_joint_rows:
        tsv_rows.append(
            {
                "kind": "joint",
                "name": row["joint_name"],
                "parent": row["parent"],
                "child": row["child"],
                "status": "ok" if row["has_action_drive_row"] else "missing_drive",
                "detail": (
                    f"axis={row['axis']} limit=[{row['limit_lower']},{row['limit_upper']}] "
                    f"effort={row['limit_effort']} velocity={row['limit_velocity']} "
                    f"stiffness={row['stiffness']} armature={row['armature']} action_scale={row['action_scale']}"
                ),
            }
        )
    summary: dict[str, Any] = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_g1_urdf_physical_asset_contract_audit",
        "scope": (
            "Parses the official G1 URDF and local action-scale audit to enumerate the physical fields needed to "
            "enrich the minimal skeleton USD. This is a converter contract audit only: it does not generate a "
            "physically faithful USD, motion.npz, replay, PPO training, policy evaluation, video, or robot result."
        ),
        "sources": {
            "official_urdf": str(OFFICIAL_URDF),
            "action_scale_audit": str(ACTION_SCALE_AUDIT),
            "skeleton_usd_audit": str(SKELETON_USD_AUDIT),
            "official_source_audit": str(OFFICIAL_SOURCE_AUDIT),
        },
        "source_hashes": {"official_urdf_sha256": sha256_file(OFFICIAL_URDF)},
        "metrics": {
            "link_count": len(link_rows),
            "joint_count": len(joint_rows),
            "nonfixed_joint_count": len(nonfixed_joint_rows),
            "fixed_joint_count": len(fixed_joint_rows),
            "inertial_link_count": len(link_rows) - len(missing_inertial_links),
            "missing_inertial_link_count": len(missing_inertial_links),
            "visual_mesh_reference_count": len(mesh_rows),
            "missing_mesh_reference_count": len(missing_mesh_refs),
            "collision_element_count": len(collision_rows),
            "collision_link_count": len({row["link_name"] for row in collision_rows}),
            "collision_type_counts": dict(sorted(collision_type_counts.items())),
            "target_body_count": len(target_bodies),
            "target_body_missing_inertial_count": len(target_body_missing_inertial),
        },
        "gaps": {
            "missing_inertial_links": missing_inertial_links,
            "missing_mesh_refs": missing_mesh_refs,
            "nonfixed_missing_axis": nonfixed_missing_axis,
            "nonfixed_missing_limit": nonfixed_missing_limit,
            "nonfixed_missing_drive": nonfixed_missing_drive,
            "target_body_missing_inertial": target_body_missing_inertial,
            "still_missing_for_replay": [
                "USD authoring of inertial/mass tensors",
                "USD visual mesh references or mesh payloads",
                "USD collision geometry authoring",
                "USD revolute joint origin/axis/limit authoring",
                "USD actuator/drive metadata from action-scale rows",
                "csv_to_npz.py validation",
                "replay_npz.py validation",
            ],
        },
        "checks": checks,
        "link_rows": link_rows,
        "joint_rows": joint_rows,
        "mesh_rows": mesh_rows,
        "collision_rows": collision_rows,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The public official URDF has enough mesh, collision, joint-axis/limit, and action-drive contract "
                "data to define an offline converter scaffold, while three non-target support/hand links lack "
                "inertial tags. No physically faithful USD or replay artifact has been generated yet."
            ),
            "next_action": (
                "Author an offline USD enrichment pass from this contract and the skeleton USD, then rerun the "
                "official csv_to_npz/replay gates. Keep results labeled as resource-adjusted until official replay "
                "validation succeeds."
            ),
        },
        "outputs": {
            "json": str(OUT / "tracking_g1_urdf_physical_asset_contract_audit.json"),
            "tsv": str(OUT / "tracking_g1_urdf_physical_asset_contract_audit.tsv"),
        },
    }
    (OUT / "tracking_g1_urdf_physical_asset_contract_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_tsv(OUT / "tracking_g1_urdf_physical_asset_contract_audit.tsv", tsv_rows)
    print(
        json.dumps(
            {
                "status": status,
                "json": summary["outputs"]["json"],
                "link_count": len(link_rows),
                "nonfixed_joint_count": len(nonfixed_joint_rows),
                "missing_inertial_links": missing_inertial_links,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
