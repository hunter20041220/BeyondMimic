#!/usr/bin/env python3
"""Audit the candidate sagittal symmetry joint mapping for Unitree G1."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/level_c/symmetry_mapping_audit"
PAPER = ROOT / "reproduction/paper/source/root.tex"
CONTROLLER_YAML = ROOT / "reproduction/third_party/official/motion_tracking_controller/config/g1/controllers.yaml"
G1_PY = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    / "whole_body_tracking/robots/g1.py"
)
URDF = (
    ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    / "whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf"
)
AUGMENTATION_JSON = ROOT / "res/level_c/augmentation_probe/level_c_augmentation_probe.json"
AUGMENTATION_NPZ = ROOT / "res/level_c/augmentation_probe/level_c_augmentation_probe.npz"


SYMMETRY_PAIRS = [
    ("left_hip_pitch_joint", "right_hip_pitch_joint", 1.0),
    ("left_hip_roll_joint", "right_hip_roll_joint", -1.0),
    ("left_hip_yaw_joint", "right_hip_yaw_joint", -1.0),
    ("left_knee_joint", "right_knee_joint", 1.0),
    ("left_ankle_pitch_joint", "right_ankle_pitch_joint", 1.0),
    ("left_ankle_roll_joint", "right_ankle_roll_joint", -1.0),
    ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint", 1.0),
    ("left_shoulder_roll_joint", "right_shoulder_roll_joint", -1.0),
    ("left_shoulder_yaw_joint", "right_shoulder_yaw_joint", -1.0),
    ("left_elbow_joint", "right_elbow_joint", 1.0),
    ("left_wrist_roll_joint", "right_wrist_roll_joint", -1.0),
    ("left_wrist_pitch_joint", "right_wrist_pitch_joint", 1.0),
    ("left_wrist_yaw_joint", "right_wrist_yaw_joint", -1.0),
]

CENTER_SIGN = {
    "waist_yaw_joint": -1.0,
    "waist_roll_joint": -1.0,
    "waist_pitch_joint": 1.0,
}


def parse_controller_joint_names(text: str) -> list[str]:
    marker = "joint_names:"
    start = text.index(marker)
    block = text[start : text.index("default_position:", start)]
    content = block[block.index("[") + 1 : block.index("]")]
    return [item.strip() for item in content.replace("\n", " ").split(",") if item.strip()]


def parse_urdf_joint_names(text: str) -> list[str]:
    return re.findall(r'<joint[^>]+name="([^"]+)"', text)


def build_mapping(joint_names: list[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    mirror: dict[str, tuple[str, float, str]] = {}
    for left, right, sign in SYMMETRY_PAIRS:
        mirror[left] = (right, sign, "left_right_pair")
        mirror[right] = (left, sign, "left_right_pair")
    for name, sign in CENTER_SIGN.items():
        mirror[name] = (name, sign, "center")

    rows: list[dict[str, Any]] = []
    for idx, name in enumerate(joint_names):
        target, sign, kind = mirror.get(name, ("", 0.0, "missing"))
        rows.append(
            {
                "index": idx,
                "joint_name": name,
                "mirror_joint_name": target,
                "sign": sign,
                "kind": kind,
                "covered": bool(target),
            }
        )
    counts = {
        "joint_count": len(joint_names),
        "covered_joint_count": sum(1 for row in rows if row["covered"]),
        "pair_count": len(SYMMETRY_PAIRS),
        "center_joint_count": len(CENTER_SIGN),
        "pitch_like_sign_positive_count": sum(
            1 for _, _, sign in SYMMETRY_PAIRS if sign > 0
        )
        + sum(1 for sign in CENTER_SIGN.values() if sign > 0),
        "roll_yaw_like_sign_negative_count": sum(
            1 for _, _, sign in SYMMETRY_PAIRS if sign < 0
        )
        + sum(1 for sign in CENTER_SIGN.values() if sign < 0),
    }
    return rows, counts


def mirror_array(values: np.ndarray, rows: list[dict[str, Any]]) -> np.ndarray:
    mirrored = np.zeros_like(values)
    name_to_idx = {row["joint_name"]: int(row["index"]) for row in rows}
    for row in rows:
        idx = int(row["index"])
        target = row["mirror_joint_name"]
        sign = float(row["sign"])
        mirrored[:, idx] = sign * values[:, name_to_idx[target]]
    return mirrored


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["index", "joint_name", "mirror_joint_name", "sign", "kind", "covered"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paper_text = PAPER.read_text(encoding="utf-8")
    controller_text = CONTROLLER_YAML.read_text(encoding="utf-8")
    g1_text = G1_PY.read_text(encoding="utf-8")
    urdf_text = URDF.read_text(encoding="utf-8")
    augmentation = json.loads(AUGMENTATION_JSON.read_text(encoding="utf-8"))
    npz = np.load(AUGMENTATION_NPZ)

    controller_joint_names = parse_controller_joint_names(controller_text)
    urdf_joint_names = parse_urdf_joint_names(urdf_text)
    rows, counts = build_mapping(controller_joint_names)
    original_pos = npz["mirrored_joint_pos_candidate"] * 0.0
    # Recover original via applying the candidate mirror twice to the mirrored output.
    mirrored_pos = npz["mirrored_joint_pos_candidate"]
    double_mirrored_pos = mirror_array(mirrored_pos, rows)
    fixture_joint_count = mirrored_pos.shape[1]

    candidate_joint_names = augmentation["symmetry_candidate"]["joint_names"]
    missing_in_urdf = sorted(set(controller_joint_names) - set(urdf_joint_names))
    extra_urdf_actuated_like = sorted(
        j
        for j in set(urdf_joint_names) - set(controller_joint_names)
        if any(token in j for token in ["hip", "knee", "ankle", "waist", "shoulder", "elbow", "wrist"])
    )
    json_path = OUT / "level_c_symmetry_mapping_audit.json"
    tsv_path = OUT / "level_c_symmetry_mapping_audit.tsv"
    checks = {
        "paper_mentions_sagittal_symmetry": "sagittal" in paper_text.lower() and "symmetry" in paper_text.lower(),
        "paper_does_not_publish_g1_sign_table": "sign table" not in paper_text.lower(),
        "controller_joint_count_29": len(controller_joint_names) == 29,
        "fixture_joint_count_29": fixture_joint_count == 29,
        "candidate_joint_names_match_controller_order": candidate_joint_names == controller_joint_names,
        "all_controller_joints_covered": counts["covered_joint_count"] == len(controller_joint_names),
        "pair_and_center_counts_match_29": counts["pair_count"] * 2 + counts["center_joint_count"] == 29,
        "mapping_is_involution_by_rows": all(
            rows[int(r["index"])]["joint_name"]
            == next(rr for rr in rows if rr["joint_name"] == r["mirror_joint_name"])["mirror_joint_name"]
            for r in rows
        ),
        "augmentation_probe_double_mirror_exact": augmentation["checks"]["symmetry_double_mirror_exact"],
        "velocity_double_mirror_exact": augmentation["checks"]["velocity_symmetry_double_mirror_exact"],
        "controller_joints_exist_in_urdf": len(missing_in_urdf) == 0,
        "no_extra_urdf_actuated_like_joints": len(extra_urdf_actuated_like) == 0,
        "official_source_has_left_right_regexes": ".*_hip_pitch_joint" in g1_text and ".*_shoulder_pitch_joint" in g1_text,
    }
    # The recovered fixture itself is not stored in the augmentation NPZ, so the executable
    # double-mirror check comes from the augmentation probe. We keep this zero-valued placeholder
    # only to record the audited array shape without claiming a second independent recovery.
    _ = original_pos
    metrics = dict(counts)
    metrics.update(
        {
            "urdf_joint_count": len(urdf_joint_names),
            "controller_missing_in_urdf_count": len(missing_in_urdf),
            "extra_urdf_actuated_like_count": len(extra_urdf_actuated_like),
            "mirrored_fixture_shape": list(mirrored_pos.shape),
            "double_mirrored_fixture_shape": list(double_mirrored_pos.shape),
        }
    )
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "level_c_symmetry_mapping_audit",
        "scope": (
            "Static and debug-array audit of the candidate Unitree G1 sagittal symmetry mapping used for Level C "
            "augmentation probes. This records coverage and involution properties but does not prove an official "
            "paper sign table, because none is published in the local paper/source."
        ),
        "metrics": metrics,
        "checks": checks,
        "missing_controller_joints_in_urdf": missing_in_urdf,
        "extra_urdf_actuated_like_joints": extra_urdf_actuated_like,
        "mapping_rows": rows,
        "paper_evidence": {
            "sagittal_symmetry_augmentation": "reproduction/paper/source/root.tex:591-592",
            "official_controller_joint_order": "reproduction/third_party/official/motion_tracking_controller/config/g1/controllers.yaml",
            "official_g1_regex_groups": "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py",
            "official_urdf": "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf",
            "debug_double_mirror_probe": "res/level_c/augmentation_probe/level_c_augmentation_probe.json",
        },
        "interpretation": {
            "paper_level_status": "partial",
            "goal_complete": False,
            "why_not_complete": (
                "The candidate G1 sagittal mirror mapping is complete and involutive for the local controller joint "
                "order and debug fixture, but the paper/local official artifacts do not publish a definitive sign "
                "table or trainable augmented VAE/diffusion dataset."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_tsv(tsv_path, rows)
    print(json.dumps({"status": summary["status"], "json": str(json_path), "rows": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
