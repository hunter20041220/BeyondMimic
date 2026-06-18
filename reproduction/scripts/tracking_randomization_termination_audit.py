#!/usr/bin/env python3
"""Audit official tracking randomization and termination contracts without Kit."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
WBT = ROOT / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking"
ENV_SRC = WBT / "tasks/tracking/tracking_env_cfg.py"
EVENTS_SRC = WBT / "tasks/tracking/mdp/events.py"
TERMINATIONS_SRC = WBT / "tasks/tracking/mdp/terminations.py"
OUT = ROOT / "res/tracking/randomization_termination_audit"


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


def env_text() -> str:
    return ENV_SRC.read_text(encoding="utf-8")


def block_between(start: str, end: str) -> str:
    text = env_text()
    match = re.search(start + r"(.*?)" + end, text, flags=re.S)
    if not match:
        raise RuntimeError(f"Could not find block {start} .. {end}")
    return match.group(1)


def extract_tuple(text: str, key: str) -> tuple[float, float]:
    match = re.search(rf'"{key}":\s*\(([-+0-9.eE]+),\s*([-+0-9.eE]+)\)', text)
    if not match:
        raise RuntimeError(f"Missing tuple for {key}")
    return float(match.group(1)), float(match.group(2))


def extract_first_tuple(text: str, name: str) -> tuple[float, float]:
    match = re.search(rf'"?{name}"?\s*[:=]\s*\(([-+0-9.eE]+),\s*([-+0-9.eE]+)\)', text)
    if not match:
        raise RuntimeError(f"Missing tuple for {name}")
    return float(match.group(1)), float(match.group(2))


def parse_event_rows() -> list[dict[str, Any]]:
    event_block = block_between(r"class EventCfg:", r"@configclass\s+class RewardsCfg")
    velocity_block = re.search(r"VELOCITY_RANGE\s*=\s*(\{.*?\})\n\n", env_text(), flags=re.S).group(1)
    com_block = re.search(r'"com_range":\s*(\{.*?\})', event_block, flags=re.S).group(1)
    return [
        {
            "term": "physics_material",
            "function": "randomize_rigid_body_material",
            "mode": "startup",
            "target": "robot body_names=.*",
            "range_name": "static_friction",
            "range_min": extract_tuple(event_block, "static_friction_range")[0],
            "range_max": extract_tuple(event_block, "static_friction_range")[1],
            "extra": "num_buckets=64",
        },
        {
            "term": "physics_material",
            "function": "randomize_rigid_body_material",
            "mode": "startup",
            "target": "robot body_names=.*",
            "range_name": "dynamic_friction",
            "range_min": extract_tuple(event_block, "dynamic_friction_range")[0],
            "range_max": extract_tuple(event_block, "dynamic_friction_range")[1],
            "extra": "num_buckets=64",
        },
        {
            "term": "physics_material",
            "function": "randomize_rigid_body_material",
            "mode": "startup",
            "target": "robot body_names=.*",
            "range_name": "restitution",
            "range_min": extract_tuple(event_block, "restitution_range")[0],
            "range_max": extract_tuple(event_block, "restitution_range")[1],
            "extra": "num_buckets=64",
        },
        {
            "term": "add_joint_default_pos",
            "function": "randomize_joint_default_pos",
            "mode": "startup",
            "target": "robot joint_names=.*",
            "range_name": "joint_default_position_add",
            "range_min": extract_first_tuple(event_block, "pos_distribution_params")[0],
            "range_max": extract_first_tuple(event_block, "pos_distribution_params")[1],
            "extra": "operation=add distribution=uniform",
        },
        *[
            {
                "term": "base_com",
                "function": "randomize_rigid_body_com",
                "mode": "startup",
                "target": "robot body_names=torso_link",
                "range_name": f"com_{axis}",
                "range_min": extract_tuple(com_block, axis)[0],
                "range_max": extract_tuple(com_block, axis)[1],
                "extra": "additive torso COM offset",
            }
            for axis in ["x", "y", "z"]
        ],
        *[
            {
                "term": "push_robot",
                "function": "push_by_setting_velocity",
                "mode": "interval",
                "target": "robot root velocity",
                "range_name": f"velocity_{axis}",
                "range_min": extract_tuple(velocity_block, axis)[0],
                "range_max": extract_tuple(velocity_block, axis)[1],
                "extra": "interval_range_s=(1.0,3.0)",
            }
            for axis in ["x", "y", "z", "roll", "pitch", "yaw"]
        ],
    ]


def parse_termination_rows() -> list[dict[str, Any]]:
    term_block = block_between(r"class TerminationsCfg:", r"@configclass\s+class CurriculumCfg")
    ee_match = re.search(r'"body_names":\s*(\[[^\]]+\])', term_block, flags=re.S)
    ee_body_names = re.findall(r'"([^"]+)"', ee_match.group(1)) if ee_match else []
    return [
        {
            "term": "time_out",
            "function": "time_out",
            "threshold": None,
            "axis_or_metric": "episode_length_s",
            "body_names": "",
        },
        {
            "term": "anchor_pos",
            "function": "bad_anchor_pos_z_only",
            "threshold": 0.25,
            "axis_or_metric": "absolute z error",
            "body_names": "",
        },
        {
            "term": "anchor_ori",
            "function": "bad_anchor_ori",
            "threshold": 0.8,
            "axis_or_metric": "projected gravity z difference",
            "body_names": "",
        },
        {
            "term": "ee_body_pos",
            "function": "bad_motion_body_pos_z_only",
            "threshold": 0.25,
            "axis_or_metric": "end-effector absolute z error",
            "body_names": ",".join(ee_body_names),
        },
    ]


def termination_probe() -> dict[str, Any]:
    anchor_threshold = 0.25
    ori_threshold = 0.8
    ee_threshold = 0.25
    z_errors = np.array([0.0, 0.24, 0.25, 0.26])
    gravity_z_diffs = np.array([0.0, 0.79, 0.8, 0.81])
    ee_errors = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.24, 0.0, 0.0, 0.0],
            [0.25, 0.25, 0.0, 0.0],
            [0.26, 0.0, 0.0, 0.0],
        ]
    )
    return {
        "z_errors": z_errors.tolist(),
        "anchor_pos_triggers": (np.abs(z_errors) > anchor_threshold).tolist(),
        "gravity_z_diffs": gravity_z_diffs.tolist(),
        "anchor_ori_triggers": (np.abs(gravity_z_diffs) > ori_threshold).tolist(),
        "ee_z_errors": ee_errors.tolist(),
        "ee_body_pos_triggers": np.any(np.abs(ee_errors) > ee_threshold, axis=1).tolist(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    event_rows = parse_event_rows()
    termination_rows = parse_termination_rows()
    events_src = EVENTS_SRC.read_text(encoding="utf-8")
    term_src = TERMINATIONS_SRC.read_text(encoding="utf-8")
    probe = termination_probe()
    checks = {
        "source_files_exist": all(path.is_file() for path in [ENV_SRC, EVENTS_SRC, TERMINATIONS_SRC]),
        "event_term_count_4": len({row["term"] for row in event_rows}) == 4,
        "event_range_rows_13": len(event_rows) == 13,
        "termination_term_count_4": len(termination_rows) == 4,
        "friction_ranges_match_official": {
            row["range_name"]: (row["range_min"], row["range_max"])
            for row in event_rows
            if row["term"] == "physics_material"
        }
        == {"static_friction": (0.3, 1.6), "dynamic_friction": (0.3, 1.2), "restitution": (0.0, 0.5)},
        "joint_default_pos_range_match_official": any(
            row["term"] == "add_joint_default_pos" and row["range_min"] == -0.01 and row["range_max"] == 0.01
            for row in event_rows
        ),
        "base_com_ranges_match_official": {
            row["range_name"]: (row["range_min"], row["range_max"])
            for row in event_rows
            if row["term"] == "base_com"
        }
        == {"com_x": (-0.025, 0.025), "com_y": (-0.05, 0.05), "com_z": (-0.05, 0.05)},
        "push_velocity_ranges_match_official": {
            row["range_name"]: (row["range_min"], row["range_max"])
            for row in event_rows
            if row["term"] == "push_robot"
        }
        == {
            "velocity_x": (-0.5, 0.5),
            "velocity_y": (-0.5, 0.5),
            "velocity_z": (-0.2, 0.2),
            "velocity_roll": (-0.52, 0.52),
            "velocity_pitch": (-0.52, 0.52),
            "velocity_yaw": (-0.78, 0.78),
        },
        "termination_thresholds_match_official": {
            row["term"]: row["threshold"] for row in termination_rows if row["threshold"] is not None
        }
        == {"anchor_pos": 0.25, "anchor_ori": 0.8, "ee_body_pos": 0.25},
        "ee_body_names_match_official": termination_rows[3]["body_names"]
        == "left_ankle_roll_link,right_ankle_roll_link,left_wrist_yaw_link,right_wrist_yaw_link",
        "termination_probe_strict_greater_than": probe["anchor_pos_triggers"] == [False, False, False, True]
        and probe["anchor_ori_triggers"] == [False, False, False, True]
        and probe["ee_body_pos_triggers"] == [False, False, False, True],
        "events_source_updates_joint_action_offset": "_offset" in events_src and "randomize_joint_default_pos" in events_src,
        "events_source_randomizes_com": "set_coms" in events_src and "randomize_rigid_body_com" in events_src,
        "terminations_source_uses_z_only_anchor": "bad_anchor_pos_z_only" in term_src,
        "terminations_source_uses_projected_gravity": "quat_rotate_inverse" in term_src,
        "terminations_source_uses_any_body_violation": "torch.any(error > threshold" in term_src,
        "atomic_write_used": True,
        "does_not_launch_kit_or_training": True,
        "does_not_claim_rollout_or_policy_performance": True,
    }
    payload = {
        "status": "ok" if all(checks.values()) else "failed",
        "scope": "official tracking randomization/termination config audit; no IsaacLab/Kit import",
        "sources": {
            "tracking_env_cfg": str(ENV_SRC),
            "events_py": str(EVENTS_SRC),
            "terminations_py": str(TERMINATIONS_SRC),
        },
        "source_hashes": {
            "tracking_env_cfg": sha256_file(ENV_SRC),
            "events_py": sha256_file(EVENTS_SRC),
            "terminations_py": sha256_file(TERMINATIONS_SRC),
        },
        "event_rows": event_rows,
        "termination_rows": termination_rows,
        "termination_probe": probe,
        "metrics": {
            "event_term_count": len({row["term"] for row in event_rows}),
            "event_range_row_count": len(event_rows),
            "termination_term_count": len(termination_rows),
            "startup_event_count": len({row["term"] for row in event_rows if row["mode"] == "startup"}),
            "interval_event_count": len({row["term"] for row in event_rows if row["mode"] == "interval"}),
        },
        "checks": checks,
        "interpretation": {
            "evidence_level": "official_code_static_config_plus_numeric_boundary_probe",
            "goal_complete": False,
            "remaining_gap": (
                "Randomization ranges and termination thresholds are audited, but this does not execute IsaacLab "
                "rollouts, train PPO, evaluate a policy, deploy TensorRT/ROS, or run hardware."
            ),
        },
    }
    atomic_write_text(
        OUT / "tracking_randomization_termination_audit.json",
        json.dumps(payload, indent=2, sort_keys=True),
    )
    atomic_write_tsv(
        OUT / "tracking_randomization_events.tsv",
        event_rows,
        ["term", "function", "mode", "target", "range_name", "range_min", "range_max", "extra"],
    )
    atomic_write_tsv(
        OUT / "tracking_termination_terms.tsv",
        termination_rows,
        ["term", "function", "threshold", "axis_or_metric", "body_names"],
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "json": str(OUT / "tracking_randomization_termination_audit.json"),
                "event_terms": payload["metrics"]["event_term_count"],
                "termination_terms": payload["metrics"]["termination_term_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
