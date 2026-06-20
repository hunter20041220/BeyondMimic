#!/usr/bin/env python3
"""Audit the GPU4 in-memory URDF importer USDA export without launching Kit."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/g1_urdf_in_memory_gpu4_export_structure_audit"
PROBE_JSON = ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/tracking_g1_urdf_in_memory_gpu4_probe.json"
EXPORT_USD = ROOT / "res/tracking/g1_urdf_in_memory_gpu4_probe/g1_official_importer_in_memory_gpu4_export.usda"
ACTION_SCALE_AUDIT = ROOT / "res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json"


PATTERNS = {
    "mesh_defs": re.compile(r"\bdef\s+Mesh\b"),
    "capsule_defs": re.compile(r"\bdef\s+Capsule\b"),
    "xform_defs": re.compile(r"\bdef\s+Xform\b"),
    "physics_revolute_joints": re.compile(r"\bdef\s+PhysicsRevoluteJoint\b"),
    "physics_fixed_joints": re.compile(r"\bdef\s+PhysicsFixedJoint\b"),
    "physics_prismatic_joints": re.compile(r"\bdef\s+PhysicsPrismaticJoint\b"),
    "rigid_body_api": re.compile(r"RigidBodyAPI"),
    "articulation_root_api": re.compile(r"ArticulationRootAPI"),
    "physics_drive_api": re.compile(r"PhysicsDriveAPI"),
    "joint_state_api": re.compile(r"PhysicsJointStateAPI"),
}

TARGET_BODIES = [
    "pelvis",
    "torso_link",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def action_joint_names() -> list[str]:
    data = load_json(ACTION_SCALE_AUDIT)
    return [row["joint_name"] for row in data["joint_rows"]]


def scan_export(path: Path, action_joints: list[str]) -> dict[str, Any]:
    counts = {name: 0 for name in PATTERNS}
    target_body_hits = {name: False for name in TARGET_BODIES}
    action_joint_hits = {name: False for name in action_joints}
    sample_physics_lines: list[dict[str, Any]] = []
    default_prim_g1 = False
    root_g1_def = False

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, 1):
            if 'defaultPrim = "g1"' in line:
                default_prim_g1 = True
            if re.search(r'\bdef\s+Xform\s+"g1"', line):
                root_g1_def = True
            for key, pattern in PATTERNS.items():
                if pattern.search(line):
                    counts[key] += 1
                    if len(sample_physics_lines) < 40 and (
                        "Joint" in line or "physics:" in line or "RigidBodyAPI" in line or "ArticulationRootAPI" in line
                    ):
                        sample_physics_lines.append({"line": lineno, "text": line.strip()[:240]})
            for name in target_body_hits:
                if f'"{name}"' in line or f"/{name}" in line:
                    target_body_hits[name] = True
            for name in action_joint_hits:
                if f'"{name}"' in line or f"/{name}" in line:
                    action_joint_hits[name] = True

    hit_action_joints = [name for name, hit in action_joint_hits.items() if hit]
    hit_target_bodies = [name for name, hit in target_body_hits.items() if hit]
    return {
        "counts": counts,
        "default_prim_g1": default_prim_g1,
        "root_g1_def": root_g1_def,
        "action_joint_count_expected": len(action_joints),
        "action_joint_hit_count": len(hit_action_joints),
        "action_joint_hits": hit_action_joints,
        "action_joint_missing": [name for name, hit in action_joint_hits.items() if not hit],
        "target_body_count_checked": len(TARGET_BODIES),
        "target_body_hit_count": len(hit_target_bodies),
        "target_body_hits": hit_target_bodies,
        "target_body_missing": [name for name, hit in target_body_hits.items() if not hit],
        "sample_physics_lines": sample_physics_lines,
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["metric", "value"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    probe = load_json(PROBE_JSON)
    action_joints = action_joint_names()
    scan = scan_export(EXPORT_USD, action_joints) if EXPORT_USD.is_file() else {}
    export_size = EXPORT_USD.stat().st_size if EXPORT_USD.is_file() else 0
    counts = scan.get("counts", {})
    checks = {
        "source_probe_json_exists": PROBE_JSON.is_file(),
        "source_probe_records_vulkan_device_lost": probe.get("markers", {}).get("vulkan_device_lost") is True,
        "source_probe_records_import_returned": probe.get("markers", {}).get("after_parse_import_in_memory") is True,
        "export_exists": EXPORT_USD.is_file(),
        "export_nonempty_large": export_size > 100_000_000,
        "default_prim_g1": scan.get("default_prim_g1") is True,
        "root_g1_def": scan.get("root_g1_def") is True,
        "has_mesh_defs": counts.get("mesh_defs", 0) > 0,
        "has_capsule_defs": counts.get("capsule_defs", 0) > 0,
        "has_rigid_body_api": counts.get("rigid_body_api", 0) > 0,
        "has_articulation_root_api": counts.get("articulation_root_api", 0) > 0,
        "has_physics_revolute_joints": counts.get("physics_revolute_joints", 0) >= 29,
        "all_29_action_joints_present": scan.get("action_joint_hit_count") == len(action_joints),
        "target_bodies_present_for_replay_context": scan.get("target_body_hit_count", 0) >= 14,
        "payload_not_recorded": probe.get("markers", {}).get("payload") is False,
        "process_not_cleanly_closed": probe.get("markers", {}).get("after_close") is False,
        "does_not_claim_motion_npz": True,
        "does_not_claim_official_replay": True,
        "does_not_start_training": True,
    }
    status = (
        "ok_with_physics_usd_export_but_vulkan_device_lost"
        if checks["export_exists"]
        and checks["has_articulation_root_api"]
        and checks["has_physics_revolute_joints"]
        and checks["source_probe_records_vulkan_device_lost"]
        else "blocked_missing_usable_g1_export_structure"
    )
    latest_blocker = (
        "official_g1_importer_exports_physics_stage_but_vulkan_device_lost_before_payload_or_replay"
        if status == "ok_with_physics_usd_export_but_vulkan_device_lost"
        else "official_g1_importer_export_structure_incomplete_or_missing"
    )
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_g1_urdf_in_memory_gpu4_export_structure_audit",
        "scope": (
            "Text-scans the local USDA exported by the official Isaac Sim URDF importer in-memory GPU4 probe. "
            "This audit does not launch Kit, does not modify the export, and does not copy the large USDA into Git. "
            "It determines whether the export contains robot/physics structure while preserving the blocker boundary."
        ),
        "source_probe": {
            "json": str(PROBE_JSON),
            "status": probe.get("status"),
            "returncode": probe.get("returncode"),
            "duration_seconds": probe.get("duration_seconds"),
            "markers": probe.get("markers", {}),
            "latest_blocker": probe.get("latest_blocker"),
        },
        "export": {
            "path": str(EXPORT_USD),
            "size_bytes": export_size,
            "sha256": sha256_file(EXPORT_USD) if EXPORT_USD.is_file() else "",
            **scan,
        },
        "checks": checks,
        "latest_blocker": latest_blocker,
        "interpretation": {
            "evidence_level": "official_importer_structure_audit_blocked_before_replay",
            "goal_complete": False,
            "why_not_complete": (
                "The official importer produced a nonempty G1 physics USD structure, but the Kit process still hit "
                "Vulkan ERROR_DEVICE_LOST before payload/clean shutdown and no official csv_to_npz/replay_npz, PPO, "
                "DAgger, VAE/diffusion, Fig. 5/Fig. 6, TensorRT, or robot validation was run from this export."
            ),
            "large_export_not_committed": True,
        },
        "outputs": {
            "json": str(OUT / "tracking_g1_urdf_in_memory_gpu4_export_structure_audit.json"),
            "tsv": str(OUT / "tracking_g1_urdf_in_memory_gpu4_export_structure_audit.tsv"),
            "readme": str(OUT / "README.md"),
        },
    }
    rows = [{"metric": key, "value": value} for key, value in counts.items()]
    rows.extend(
        [
            {"metric": "export_size_bytes", "value": export_size},
            {"metric": "action_joint_hit_count", "value": scan.get("action_joint_hit_count", 0)},
            {"metric": "target_body_hit_count", "value": scan.get("target_body_hit_count", 0)},
            {"metric": "latest_blocker", "value": latest_blocker},
        ]
    )
    (OUT / "tracking_g1_urdf_in_memory_gpu4_export_structure_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_tsv(OUT / "tracking_g1_urdf_in_memory_gpu4_export_structure_audit.tsv", rows)
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# G1 URDF In-Memory GPU4 Export Structure Audit",
                "",
                "This directory stores lightweight evidence for the local USDA exported by the official Isaac Sim",
                "URDF importer in-memory GPU4 probe. The large USDA itself remains local and ignored by Git.",
                "",
                f"Status: `{status}`",
                f"Latest blocker: `{latest_blocker}`",
                "",
                "Claim boundary: this is an official importer structure audit only. It is not official replay,",
                "not motion preprocessing success, not PPO training, and not paper-level tracking reproduction.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "latest_blocker": latest_blocker, "json": summary["outputs"]["json"]}))


if __name__ == "__main__":
    main()
