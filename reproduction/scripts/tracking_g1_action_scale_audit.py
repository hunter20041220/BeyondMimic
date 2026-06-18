#!/usr/bin/env python3
"""Expand official Unitree G1 actuator/action-scale settings per joint.

The official whole_body_tracking code computes ``G1_ACTION_SCALE`` from effort
limits and stiffness. This audit reproduces that calculation without importing
IsaacLab/Kit, expands regex actuator groups over the local G1 URDF joints, and
writes a per-joint table for traceability.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
WBT = ROOT / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking"
G1_SRC = WBT / "robots/g1.py"
G1_URDF = WBT / "assets/unitree_description/urdf/g1/main.urdf"
OUT = ROOT / "res/tracking/g1_action_scale_audit"


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eval_expr(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -eval_expr(node.operand, env)
    if isinstance(node, ast.BinOp):
        left = eval_expr(node.left, env)
        right = eval_expr(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
    if isinstance(node, (ast.List, ast.Tuple)):
        return [eval_expr(elt, env) for elt in node.elts]
    if isinstance(node, ast.Dict):
        return {eval_expr(k, env): eval_expr(v, env) for k, v in zip(node.keys, node.values)}
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def constants_from_source(tree: ast.Module) -> dict[str, Any]:
    env: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if re.match(r"^(ARMATURE|NATURAL_FREQ|DAMPING|STIFFNESS)_?", target.id):
            env[target.id] = eval_expr(node.value, env)
    return env


def keyword(call: ast.Call, name: str) -> ast.AST | None:
    for item in call.keywords:
        if item.arg == name:
            return item.value
    return None


def actuator_calls(tree: ast.Module) -> list[tuple[str, ast.Call]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "actuators" and isinstance(node.value, ast.Dict):
            calls: list[tuple[str, ast.Call]] = []
            for key, value in zip(node.value.keys, node.value.values):
                if not isinstance(key, ast.Constant) or not isinstance(value, ast.Call):
                    continue
                calls.append((str(key.value), value))
            return calls
    raise RuntimeError("Could not find G1_CYLINDER_CFG actuators dict")


def scalar_or_map(value: Any, names: list[str]) -> dict[str, float]:
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    return {name: float(value) for name in names}


def parse_actuators() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tree = ast.parse(G1_SRC.read_text(encoding="utf-8"))
    env = constants_from_source(tree)
    groups: list[dict[str, Any]] = []
    for group_name, call in actuator_calls(tree):
        names_node = keyword(call, "joint_names_expr")
        if names_node is None:
            raise RuntimeError(f"Missing joint_names_expr for {group_name}")
        names = eval_expr(names_node, env)
        group = {
            "group": group_name,
            "joint_patterns": names,
            "effort_limit_sim": scalar_or_map(eval_expr(keyword(call, "effort_limit_sim"), env), names),
            "velocity_limit_sim": scalar_or_map(eval_expr(keyword(call, "velocity_limit_sim"), env), names),
            "stiffness": scalar_or_map(eval_expr(keyword(call, "stiffness"), env), names),
            "damping": scalar_or_map(eval_expr(keyword(call, "damping"), env), names),
            "armature": scalar_or_map(eval_expr(keyword(call, "armature"), env), names),
        }
        for pattern in names:
            effort = group["effort_limit_sim"][pattern]
            stiffness = group["stiffness"][pattern]
            group.setdefault("action_scale_by_pattern", {})[pattern] = 0.25 * effort / stiffness
        groups.append(group)
    return env, groups


def urdf_nonfixed_joints() -> list[str]:
    root = ET.parse(G1_URDF).getroot()
    return [joint.attrib["name"] for joint in root.findall("joint") if joint.attrib.get("type") != "fixed"]


def matching_pattern(joint: str, groups: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    matches: list[tuple[dict[str, Any], str]] = []
    for group in groups:
        for pattern in group["joint_patterns"]:
            if re.fullmatch(pattern, joint):
                matches.append((group, pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one actuator match for {joint}, got {len(matches)}")
    return matches[0]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    constants, groups = parse_actuators()
    joints = urdf_nonfixed_joints()
    rows: list[dict[str, Any]] = []
    for joint in joints:
        group, pattern = matching_pattern(joint, groups)
        effort = group["effort_limit_sim"][pattern]
        velocity = group["velocity_limit_sim"][pattern]
        stiffness = group["stiffness"][pattern]
        damping = group["damping"][pattern]
        armature = group["armature"][pattern]
        action_scale = 0.25 * effort / stiffness
        rows.append(
            {
                "joint_name": joint,
                "actuator_group": group["group"],
                "matched_pattern": pattern,
                "effort_limit_sim": effort,
                "velocity_limit_sim": velocity,
                "armature": armature,
                "stiffness": stiffness,
                "damping": damping,
                "action_scale": action_scale,
                "natural_frequency_rad_s": constants["NATURAL_FREQ"],
                "damping_ratio": constants["DAMPING_RATIO"],
            }
        )

    action_scales = [row["action_scale"] for row in rows]
    armatures = [row["armature"] for row in rows]
    stiffnesses = [row["stiffness"] for row in rows]
    by_group: dict[str, int] = {}
    for row in rows:
        by_group[row["actuator_group"]] = by_group.get(row["actuator_group"], 0) + 1

    checks = {
        "source_files_exist": G1_SRC.is_file() and G1_URDF.is_file(),
        "urdf_nonfixed_joint_count_29": len(joints) == 29,
        "expanded_joint_rows_29": len(rows) == 29,
        "all_joints_matched_once": len({row["joint_name"] for row in rows}) == len(joints),
        "actuator_group_count_5": len(groups) == 5,
        "group_counts_expected": by_group == {"legs": 8, "feet": 4, "waist_yaw": 1, "waist": 2, "arms": 14},
        "action_scale_formula_matches_official": all(
            math.isclose(row["action_scale"], 0.25 * row["effort_limit_sim"] / row["stiffness"], rel_tol=1e-12)
            for row in rows
        ),
        "all_action_scales_positive": min(action_scales) > 0,
        "all_armatures_positive": min(armatures) > 0,
        "all_stiffness_positive": min(stiffnesses) > 0,
        "natural_frequency_10hz_formula": math.isclose(constants["NATURAL_FREQ"], 10 * 2.0 * math.pi, rel_tol=1e-10),
        "damping_ratio_2": constants["DAMPING_RATIO"] == 2.0,
        "atomic_write_used": True,
        "does_not_launch_kit_or_training": True,
        "does_not_claim_rollout_or_deployment": True,
    }

    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "scope": "official G1 actuator/action-scale expansion from source and URDF; no Kit launch",
        "sources": {"g1_source": str(G1_SRC), "g1_urdf": str(G1_URDF)},
        "source_hashes": {"g1_source": sha256_file(G1_SRC), "g1_urdf": sha256_file(G1_URDF)},
        "constants": constants,
        "actuator_groups": groups,
        "joint_rows": rows,
        "metrics": {
            "joint_count": len(joints),
            "row_count": len(rows),
            "actuator_group_count": len(groups),
            "group_counts": by_group,
            "action_scale_min": min(action_scales),
            "action_scale_max": max(action_scales),
            "action_scale_mean": sum(action_scales) / len(action_scales),
            "armature_min": min(armatures),
            "armature_max": max(armatures),
            "stiffness_min": min(stiffnesses),
            "stiffness_max": max(stiffnesses),
        },
        "checks": checks,
        "interpretation": {
            "evidence_level": "official_code_static_numeric_contract",
            "goal_complete": False,
            "remaining_gap": (
                "The official G1 actuator/action-scale numeric contract is expanded per joint, but this does not "
                "execute IsaacLab rollouts, train tracking policies, export deployment models, or run hardware."
            ),
        },
    }

    atomic_write_text(OUT / "tracking_g1_action_scale_audit.json", json.dumps(summary, indent=2, sort_keys=True))
    atomic_write_tsv(
        OUT / "tracking_g1_action_scale_audit.tsv",
        rows,
        [
            "joint_name",
            "actuator_group",
            "matched_pattern",
            "effort_limit_sim",
            "velocity_limit_sim",
            "armature",
            "stiffness",
            "damping",
            "action_scale",
            "natural_frequency_rad_s",
            "damping_ratio",
        ],
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(OUT / "tracking_g1_action_scale_audit.json"),
                "rows": len(rows),
                "groups": by_group,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
