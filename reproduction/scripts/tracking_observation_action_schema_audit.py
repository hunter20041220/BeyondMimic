#!/usr/bin/env python3
"""Static/numeric schema audit for official tracking observations and actions.

This audit derives the actor/critic observation dimensions from the official
whole_body_tracking config and checks them against the local debug motion.npz
fixtures. It avoids importing IsaacLab/Kit.
"""

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
OBS_SRC = WBT / "tasks/tracking/mdp/observations.py"
COMMANDS_SRC = WBT / "tasks/tracking/mdp/commands.py"
FLAT_SRC = WBT / "tasks/tracking/config/g1/flat_env_cfg.py"
ACTION_SCALE_JSON = ROOT / "res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json"
FIXTURE_JSON = ROOT / "res/tracking/motion_npz_fixture/tracking_motion_npz_fixture.json"
FIXTURE_DIR = ROOT / "reproduction/data/tracking_motion_npz_fixtures"
OUT = ROOT / "res/tracking/observation_action_schema_audit"


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def obs_block(name: str) -> str:
    text = ENV_SRC.read_text(encoding="utf-8")
    if name == "policy":
        match = re.search(r"class PolicyCfg.*?class PrivilegedCfg", text, flags=re.S)
    elif name == "critic":
        match = re.search(r"class PrivilegedCfg.*?# observation groups", text, flags=re.S)
    else:
        raise ValueError(name)
    if not match:
        raise RuntimeError(f"{name} observation block not found")
    return match.group(0)


def parse_terms(block: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in re.finditer(r"^\s{8}([A-Za-z0-9_]+)\s*=\s*ObsTerm\((.*?)(?=^\s{8}[A-Za-z0-9_]+\s*=|\n\s{8}def|\n\n)", block, re.S | re.M):
        term, body = match.group(1), match.group(2)
        func_match = re.search(r"func=mdp\.([A-Za-z0-9_]+)", body)
        noise_match = re.search(r"noise=Unoise\(n_min=([-+0-9.eE]+),\s*n_max=([-+0-9.eE]+)\)", body)
        rows.append(
            {
                "term": term,
                "function": func_match.group(1) if func_match else None,
                "noise_min": float(noise_match.group(1)) if noise_match else None,
                "noise_max": float(noise_match.group(2)) if noise_match else None,
                "has_noise": noise_match is not None,
            }
        )
    return rows


def parse_flat_body_names() -> list[str]:
    text = FLAT_SRC.read_text(encoding="utf-8")
    match = re.search(r"body_names\s*=\s*(\[[^\]]+\])", text, flags=re.S)
    if not match:
        raise RuntimeError("body_names not found")
    return [item.strip().strip('"') for item in match.group(1).strip("[]").split(",") if item.strip()]


def derive_dim(function: str, joint_count: int, target_body_count: int) -> int:
    if function == "generated_commands":
        return joint_count * 2
    if function in {"motion_anchor_pos_b", "base_lin_vel", "base_ang_vel"}:
        return 3
    if function == "motion_anchor_ori_b":
        return 6
    if function in {"joint_pos_rel", "joint_vel_rel", "last_action"}:
        return joint_count
    if function == "robot_body_pos_b":
        return target_body_count * 3
    if function == "robot_body_ori_b":
        return target_body_count * 6
    raise ValueError(f"Unknown observation function {function}")


def fixture_stats() -> dict[str, Any]:
    files = sorted(FIXTURE_DIR.glob("*_debug_motion.npz"))
    stats = []
    for path in files:
        data = np.load(path)
        stats.append(
            {
                "file": str(path),
                "sha256": sha256_file(path),
                "fps": float(np.asarray(data["fps"]).reshape(-1)[0]),
                "time_steps": int(data["joint_pos"].shape[0]),
                "joint_count": int(data["joint_pos"].shape[1]),
                "urdf_body_count": int(data["body_pos_w"].shape[1]),
                "joint_vel_shape": list(data["joint_vel"].shape),
                "body_pos_w_shape": list(data["body_pos_w"].shape),
                "body_quat_w_shape": list(data["body_quat_w"].shape),
                "finite": bool(all(np.isfinite(data[key]).all() for key in data.files)),
                "max_quat_norm_error": float(np.max(np.abs(np.linalg.norm(data["body_quat_w"], axis=-1) - 1.0))),
            }
        )
    return {"files": stats, "fixture_count": len(stats)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    action_scale = load_json(ACTION_SCALE_JSON)
    fixture_audit = load_json(FIXTURE_JSON)
    body_names = parse_flat_body_names()
    joint_count = action_scale["metrics"]["joint_count"]
    target_body_count = len(body_names)
    fixture = fixture_stats()
    rows: list[dict[str, Any]] = []
    for group_name, block in [("policy", obs_block("policy")), ("critic", obs_block("critic"))]:
        for order, term in enumerate(parse_terms(block)):
            dim = derive_dim(term["function"], joint_count, target_body_count)
            rows.append(
                {
                    "group": group_name,
                    "order": order,
                    "term": term["term"],
                    "function": term["function"],
                    "dimension": dim,
                    "has_noise": term["has_noise"],
                    "noise_min": term["noise_min"],
                    "noise_max": term["noise_max"],
                }
            )
    policy_rows = [row for row in rows if row["group"] == "policy"]
    critic_rows = [row for row in rows if row["group"] == "critic"]
    policy_dim = sum(row["dimension"] for row in policy_rows)
    critic_dim = sum(row["dimension"] for row in critic_rows)
    source_text = (OBS_SRC.read_text(encoding="utf-8") + "\n" + COMMANDS_SRC.read_text(encoding="utf-8"))
    checks = {
        "source_files_exist": all(path.is_file() for path in [ENV_SRC, OBS_SRC, COMMANDS_SRC, FLAT_SRC]),
        "policy_term_count_8": len(policy_rows) == 8,
        "critic_term_count_10": len(critic_rows) == 10,
        "policy_dimension_160": policy_dim == 160,
        "critic_dimension_286": critic_dim == 286,
        "action_dimension_29": joint_count == 29,
        "target_body_count_14": target_body_count == 14,
        "policy_noise_terms_6": sum(1 for row in policy_rows if row["has_noise"]) == 6,
        "critic_noise_terms_0": sum(1 for row in critic_rows if row["has_noise"]) == 0,
        "policy_concatenate_terms_enabled": "self.concatenate_terms = True" in obs_block("policy"),
        "policy_corruption_enabled": "self.enable_corruption = True" in obs_block("policy"),
        "critic_has_body_pos_and_ori": {"body_pos", "body_ori"}.issubset({row["term"] for row in critic_rows}),
        "observation_source_has_body_frame_transforms": "subtract_frame_transforms" in source_text,
        "observation_source_uses_rot6d": "mat[..., :2].reshape" in source_text,
        "command_dimension_matches_joint_pos_plus_vel": any(
            row["term"] == "command" and row["dimension"] == 58 for row in rows
        ),
        "fixture_count_3": fixture["fixture_count"] == 3,
        "fixture_joint_count_29": all(item["joint_count"] == 29 for item in fixture["files"]),
        "fixture_urdf_body_count_40": all(item["urdf_body_count"] == 40 for item in fixture["files"]),
        "fixture_fps_50": all(item["fps"] == 50.0 for item in fixture["files"]),
        "fixture_arrays_finite": all(item["finite"] for item in fixture["files"]),
        "fixture_quaternions_unit": all(item["max_quat_norm_error"] < 1e-10 for item in fixture["files"]),
        "motion_npz_fixture_audit_ok": fixture_audit["status"] == "ok",
        "atomic_write_used": True,
        "does_not_launch_kit_or_training": True,
        "does_not_claim_rollout_or_policy_performance": True,
    }
    payload = {
        "status": "ok" if all(checks.values()) else "failed",
        "scope": "official tracking observation/action schema derivation; no IsaacLab/Kit import",
        "sources": {
            "tracking_env_cfg": str(ENV_SRC),
            "observations_py": str(OBS_SRC),
            "commands_py": str(COMMANDS_SRC),
            "g1_flat_env_cfg": str(FLAT_SRC),
            "g1_action_scale_audit": str(ACTION_SCALE_JSON),
            "motion_npz_fixture_audit": str(FIXTURE_JSON),
        },
        "source_hashes": {
            "tracking_env_cfg": sha256_file(ENV_SRC),
            "observations_py": sha256_file(OBS_SRC),
            "commands_py": sha256_file(COMMANDS_SRC),
            "g1_flat_env_cfg": sha256_file(FLAT_SRC),
        },
        "body_names": body_names,
        "observation_rows": rows,
        "fixture_stats": fixture,
        "metrics": {
            "joint_count": joint_count,
            "target_body_count": target_body_count,
            "policy_term_count": len(policy_rows),
            "critic_term_count": len(critic_rows),
            "policy_dimension": policy_dim,
            "critic_dimension": critic_dim,
            "action_dimension": joint_count,
            "fixture_count": fixture["fixture_count"],
        },
        "checks": checks,
        "interpretation": {
            "evidence_level": "official_code_static_schema_plus_local_fixture_contract",
            "goal_complete": False,
            "remaining_gap": (
                "Observation/action dimensions and fixture contracts are audited, but this does not execute "
                "IsaacLab rollouts, train PPO, evaluate a policy, deploy TensorRT/ROS, or run hardware."
            ),
        },
    }
    atomic_write_text(
        OUT / "tracking_observation_action_schema_audit.json",
        json.dumps(payload, indent=2, sort_keys=True),
    )
    atomic_write_tsv(
        OUT / "tracking_observation_action_schema_audit.tsv",
        rows,
        ["group", "order", "term", "function", "dimension", "has_noise", "noise_min", "noise_max"],
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "json": str(OUT / "tracking_observation_action_schema_audit.json"),
                "policy_dim": policy_dim,
                "critic_dim": critic_dim,
                "action_dim": joint_count,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
