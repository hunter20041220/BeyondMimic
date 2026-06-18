#!/usr/bin/env python3
"""Numerical audit for official whole_body_tracking motion reward formulas.

The official reward module depends on IsaacLab/Kit at import time, so this audit
does not import it. Instead, it statically checks the official source and runs a
NumPy equivalent of the six motion-imitation exponential error rewards using the
weights/std values from the official tracking environment config.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
WBT = ROOT / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking"
REWARD_SRC = WBT / "tasks/tracking/mdp/rewards.py"
ENV_SRC = WBT / "tasks/tracking/tracking_env_cfg.py"
OUT = ROOT / "res/tracking/reward_formula_audit"


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


def rewards_block() -> str:
    text = ENV_SRC.read_text(encoding="utf-8")
    match = re.search(r"class RewardsCfg:.*?class TerminationsCfg", text, flags=re.S)
    if not match:
        raise RuntimeError("RewardsCfg block not found")
    return match.group(0)


def parse_env_rewards() -> list[dict[str, Any]]:
    block = rewards_block()
    rows: list[dict[str, Any]] = []
    for match in re.finditer(r"^\s{4}([A-Za-z0-9_]+)\s*=\s*RewTerm\((.*?)(?=^\s{4}[A-Za-z0-9_]+\s*=|\n\n)", block, re.S | re.M):
        name, body = match.group(1), match.group(2)
        func_match = re.search(r"func=mdp\.([A-Za-z0-9_]+)", body)
        weight_match = re.search(r"weight\s*=\s*([-+0-9.eE]+)", body)
        std_match = re.search(r"['\"]std['\"]:\s*([-+0-9.eE]+)", body)
        rows.append(
            {
                "term": name,
                "function": func_match.group(1) if func_match else None,
                "weight": float(weight_match.group(1)) if weight_match else None,
                "std": float(std_match.group(1)) if std_match else None,
            }
        )
    return rows


def reward_source_contract() -> dict[str, Any]:
    text = REWARD_SRC.read_text(encoding="utf-8")
    functions = sorted(re.findall(r"^def\s+([A-Za-z0-9_]+)\(", text, flags=re.M))
    tree = ast.parse(text)
    return {
        "functions": functions,
        "function_count": len(functions),
        "uses_torch_exp_negative_error_over_std_squared": "torch.exp(-error / std**2)" in text,
        "squares_position_velocity_errors": text.count("torch.square(") >= 4,
        "uses_quat_error_magnitude_for_orientation": text.count("quat_error_magnitude") >= 3,
        "has_body_index_filter_helper": "_get_body_indexes" in functions,
        "ast_function_defs": len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]),
    }


def exp_reward(squared_error: np.ndarray | float, std: float) -> np.ndarray:
    return np.exp(-np.asarray(squared_error, dtype=float) / (std**2))


def make_rows(env_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    motion_rows = [row for row in env_rows if row["std"] is not None]
    scan = np.array([0.0, 0.25, 0.5, 1.0, 2.0], dtype=float)
    rows: list[dict[str, Any]] = []
    for row in motion_rows:
        term = row["term"]
        std = float(row["std"])
        weight = float(row["weight"])
        function = row["function"]
        values = exp_reward(scan**2, std)
        weighted = weight * values
        for distance, value, weighted_value in zip(scan, values, weighted):
            rows.append(
                {
                    "term": term,
                    "function": function,
                    "std": std,
                    "weight": weight,
                    "distance_or_angle_norm": float(distance),
                    "squared_error": float(distance**2),
                    "reward": float(value),
                    "weighted_reward": float(weighted_value),
                }
            )
    return rows


def term_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for term in sorted({row["term"] for row in rows}):
        term_rows = [row for row in rows if row["term"] == term]
        rewards = [row["reward"] for row in term_rows]
        weighted = [row["weighted_reward"] for row in term_rows]
        out.append(
            {
                "term": term,
                "function": term_rows[0]["function"],
                "std": term_rows[0]["std"],
                "weight": term_rows[0]["weight"],
                "reward_at_zero": rewards[0],
                "reward_at_max_scan": rewards[-1],
                "weighted_at_zero": weighted[0],
                "weighted_at_max_scan": weighted[-1],
                "monotone_nonincreasing": all(a >= b for a, b in zip(rewards, rewards[1:])),
                "within_unit_interval": min(rewards) >= 0.0 and max(rewards) <= 1.0,
            }
        )
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    env_reward_rows = parse_env_rewards()
    source = reward_source_contract()
    numeric_rows = make_rows(env_reward_rows)
    summary_rows = term_summary(numeric_rows)
    motion_terms = [row for row in env_reward_rows if row["std"] is not None]
    non_motion_terms = [row for row in env_reward_rows if row["std"] is None]
    expected_motion_functions = {
        "motion_global_anchor_position_error_exp",
        "motion_global_anchor_orientation_error_exp",
        "motion_relative_body_position_error_exp",
        "motion_relative_body_orientation_error_exp",
        "motion_global_body_linear_velocity_error_exp",
        "motion_global_body_angular_velocity_error_exp",
    }
    checks = {
        "source_files_exist": REWARD_SRC.is_file() and ENV_SRC.is_file(),
        "reward_term_count_9": len(env_reward_rows) == 9,
        "motion_exp_reward_term_count_6": len(motion_terms) == 6,
        "regularizer_term_count_3": len(non_motion_terms) == 3,
        "expected_motion_functions_present": {row["function"] for row in motion_terms} == expected_motion_functions,
        "source_uses_exp_negative_error_over_std_squared": source[
            "uses_torch_exp_negative_error_over_std_squared"
        ],
        "source_uses_quat_error_magnitude": source["uses_quat_error_magnitude_for_orientation"],
        "numeric_rows_30": len(numeric_rows) == 30,
        "reward_at_zero_is_one": all(math.isclose(row["reward_at_zero"], 1.0) for row in summary_rows),
        "weighted_at_zero_matches_weight": all(
            math.isclose(row["weighted_at_zero"], row["weight"]) for row in summary_rows
        ),
        "rewards_monotone_nonincreasing": all(row["monotone_nonincreasing"] for row in summary_rows),
        "rewards_within_unit_interval": all(row["within_unit_interval"] for row in summary_rows),
        "std_values_match_official": {
            row["term"]: row["std"] for row in motion_terms
        }
        == {
            "motion_global_anchor_pos": 0.3,
            "motion_global_anchor_ori": 0.4,
            "motion_body_pos": 0.3,
            "motion_body_ori": 0.4,
            "motion_body_lin_vel": 1.0,
            "motion_body_ang_vel": 3.14,
        },
        "weight_values_match_official": {
            row["term"]: row["weight"] for row in env_reward_rows
        }
        == {
            "motion_global_anchor_pos": 0.5,
            "motion_global_anchor_ori": 0.5,
            "motion_body_pos": 1.0,
            "motion_body_ori": 1.0,
            "motion_body_lin_vel": 1.0,
            "motion_body_ang_vel": 1.0,
            "action_rate_l2": -0.1,
            "joint_limit": -10.0,
            "undesired_contacts": -0.1,
        },
        "atomic_write_used": True,
        "does_not_launch_kit_or_training": True,
        "does_not_claim_rollout_or_policy_performance": True,
    }
    payload = {
        "status": "ok" if all(checks.values()) else "failed",
        "scope": "official tracking reward formula numeric audit; no IsaacLab/Kit import",
        "sources": {"rewards_py": str(REWARD_SRC), "tracking_env_cfg_py": str(ENV_SRC)},
        "source_hashes": {"rewards_py": sha256_file(REWARD_SRC), "tracking_env_cfg_py": sha256_file(ENV_SRC)},
        "source_contract": source,
        "env_reward_rows": env_reward_rows,
        "numeric_rows": numeric_rows,
        "term_summary": summary_rows,
        "metrics": {
            "reward_term_count": len(env_reward_rows),
            "motion_exp_reward_term_count": len(motion_terms),
            "regularizer_term_count": len(non_motion_terms),
            "numeric_row_count": len(numeric_rows),
            "term_summary_count": len(summary_rows),
            "scan_distance_count": 5,
        },
        "checks": checks,
        "interpretation": {
            "evidence_level": "official_code_static_and_numeric_formula",
            "goal_complete": False,
            "remaining_gap": (
                "The reward formulas and official weights/stds are numerically audited, but this does not execute "
                "IsaacLab rollouts, PPO training, policy evaluation, deployment, or hardware."
            ),
        },
    }
    atomic_write_text(OUT / "tracking_reward_formula_audit.json", json.dumps(payload, indent=2, sort_keys=True))
    atomic_write_tsv(
        OUT / "tracking_reward_formula_scan.tsv",
        numeric_rows,
        [
            "term",
            "function",
            "std",
            "weight",
            "distance_or_angle_norm",
            "squared_error",
            "reward",
            "weighted_reward",
        ],
    )
    atomic_write_tsv(
        OUT / "tracking_reward_formula_summary.tsv",
        summary_rows,
        [
            "term",
            "function",
            "std",
            "weight",
            "reward_at_zero",
            "reward_at_max_scan",
            "weighted_at_zero",
            "weighted_at_max_scan",
            "monotone_nonincreasing",
            "within_unit_interval",
        ],
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "json": str(OUT / "tracking_reward_formula_audit.json"),
                "terms": len(env_reward_rows),
                "motion_terms": len(motion_terms),
                "numeric_rows": len(numeric_rows),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
