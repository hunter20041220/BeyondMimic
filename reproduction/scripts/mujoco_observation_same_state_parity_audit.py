#!/usr/bin/env python3
"""Same-state observation slice parity against an IsaacLab sample.

This audit recomputes the eight 160-D actor observation slices from the raw
state tensors captured by `isaaclab_observation_manager_sample_gate.py` and
compares them to the official noise-free critic terms for the same state.

It is intentionally narrower than a native MuJoCo rollout.  Passing this file
means the local NumPy formulas agree with the official IsaacLab observation
manager on one captured same-state fixture.  It does not prove that a MuJoCo
runtime builder, motion_tracking_controller deployment path, teacher policy,
VAE, diffusion model, or guided controller is ready.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/audits/mujoco_observation_same_state_parity"
JSON_OUT = OUT / "mujoco_observation_same_state_parity_audit.json"
TSV_OUT = OUT / "mujoco_observation_same_state_parity_audit.tsv"
MD_OUT = OUT / "mujoco_observation_same_state_parity_audit.md"

SAMPLE_JSON = (
    ROOT / "res/audits/isaaclab_observation_manager_sample_gate/isaaclab_policy_obs_sample.json"
)

EXPECTED_TERMS = [
    "command",
    "motion_anchor_pos_b",
    "motion_anchor_ori_b",
    "base_lin_vel",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
]


def as_env0(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim >= 2 and arr.shape[0] == 1:
        return arr[0].copy()
    return arr.copy()


def norm_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-12)


def qconj(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).copy()
    out[..., 1:] *= -1.0
    return out


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def qapply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = norm_quat(q)
    v = np.asarray(v, dtype=np.float64)
    vq = np.concatenate([np.zeros(v.shape[:-1] + (1,), dtype=np.float64), v], axis=-1)
    return qmul(qmul(q, vq), qconj(q))[..., 1:4]


def qmat(q: np.ndarray) -> np.ndarray:
    q = norm_quat(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def rot6(q: np.ndarray) -> np.ndarray:
    mat = qmat(q)
    return mat[..., :2].reshape(-1)


def local_pos(parent_pos: np.ndarray, parent_quat: np.ndarray, child_pos: np.ndarray) -> np.ndarray:
    return qapply(qconj(parent_quat), child_pos - parent_pos)


def local_quat(parent_quat: np.ndarray, child_quat: np.ndarray) -> np.ndarray:
    return norm_quat(qmul(qconj(parent_quat), child_quat))


def max_error(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def load_sample() -> dict[str, Any]:
    if not SAMPLE_JSON.is_file():
        return {}
    return json.loads(SAMPLE_JSON.read_text(encoding="utf-8"))


def recompute_terms(sample: dict[str, Any]) -> dict[str, np.ndarray]:
    raw = sample.get("raw_state", {})
    command_joint_pos = as_env0(raw["command_joint_pos"])
    command_joint_vel = as_env0(raw["command_joint_vel"])
    robot_anchor_pos_w = as_env0(raw["robot_anchor_pos_w"])
    robot_anchor_quat_w = as_env0(raw["robot_anchor_quat_w"])
    command_anchor_pos_w = as_env0(raw["command_anchor_pos_w"])
    command_anchor_quat_w = as_env0(raw["command_anchor_quat_w"])
    robot_joint_pos = as_env0(raw["robot_joint_pos"])
    robot_default_joint_pos = as_env0(raw["robot_default_joint_pos"])
    robot_joint_vel = as_env0(raw["robot_joint_vel"])
    root_lin_vel_b = as_env0(raw["robot_root_lin_vel_b"])
    root_ang_vel_b = as_env0(raw["robot_root_ang_vel_b"])
    zero_action = as_env0(raw["zero_action"])

    anchor_rel_quat = local_quat(robot_anchor_quat_w, command_anchor_quat_w)
    return {
        "command": np.concatenate([command_joint_pos, command_joint_vel]),
        "motion_anchor_pos_b": local_pos(robot_anchor_pos_w, robot_anchor_quat_w, command_anchor_pos_w).reshape(-1),
        "motion_anchor_ori_b": rot6(anchor_rel_quat),
        "base_lin_vel": root_lin_vel_b.reshape(-1),
        "base_ang_vel": root_ang_vel_b.reshape(-1),
        "joint_pos": (robot_joint_pos - robot_default_joint_pos).reshape(-1),
        "joint_vel": robot_joint_vel.reshape(-1),
        "actions": zero_action.reshape(-1),
    }


def build_summary() -> dict[str, Any]:
    sample = load_sample()
    sample_checks = sample.get("checks", {})
    rows: list[dict[str, Any]] = []
    recomputed: dict[str, np.ndarray] = {}
    error = ""
    try:
        if sample:
            recomputed = recompute_terms(sample)
    except Exception as exc:  # pragma: no cover - audit retains precise failure.
        error = repr(exc)

    critic_terms = sample.get("critic_terms", {})
    policy_terms = sample.get("policy_terms", {})
    tol = 1e-5
    for name in EXPECTED_TERMS:
        official = np.asarray(critic_terms.get(name, []), dtype=np.float64)
        if official.ndim >= 2 and official.shape[0] == 1:
            official = official[0]
        local = recomputed.get(name, np.asarray([], dtype=np.float64))
        policy = np.asarray(policy_terms.get(name, []), dtype=np.float64)
        if policy.ndim >= 2 and policy.shape[0] == 1:
            policy = policy[0]
        err = max_error(local.reshape(-1), official.reshape(-1))
        policy_critic_err = max_error(policy.reshape(-1), official.reshape(-1)) if policy.size and official.size else 0.0
        rows.append(
            {
                "term": name,
                "dimension": int(local.size) if local.size else int(official.size),
                "official_dimension": int(official.size),
                "local_dimension": int(local.size),
                "max_abs_error": err,
                "passed": bool(np.isfinite(err) and err <= tol),
                "policy_vs_critic_max_abs_error": policy_critic_err,
                "notes": (
                    "Compared against critic shared term because policy term contains training-time corruption/noise."
                    if name not in {"command", "actions"}
                    else "Policy and critic are expected to match or be near-identical for this term."
                ),
            }
        )

    checks = {
        "sample_json_available": bool(sample),
        "sample_status_ok": sample.get("status") == "ok_isaaclab_observation_manager_sample_captured",
        "sample_policy_obs_dim_160": sample.get("policy_obs_dim") == 160,
        "sample_policy_terms_expected_order": bool(sample_checks.get("policy_terms_expected_order")),
        "sample_critic_shared_terms_available": bool(sample_checks.get("critic_shared_terms_available")),
        "sample_raw_state_available_for_same_state_parity": bool(
            sample_checks.get("raw_state_available_for_same_state_parity")
        ),
        "local_recompute_succeeded": not error and bool(recomputed),
        "all_same_state_formula_slices_pass": bool(rows) and all(row["passed"] for row in rows),
        "uses_noise_free_critic_reference": True,
        "policy_terms_are_not_used_as_exact_reference": True,
        "mujoco_runtime_builder_executed": False,
        "does_not_claim_mujoco_runtime_or_rollout_success": True,
    }
    status = (
        "ok_same_state_observation_formula_slices_match_official_sample_but_mujoco_runtime_pending"
        if checks["all_same_state_formula_slices_pass"]
        else "failed_same_state_observation_formula_parity"
    )
    failed = [key for key, value in checks.items() if not value]
    return {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "mujoco_observation_same_state_parity_audit",
        "claim_level": "same-state formula parity only; no MuJoCo runtime rollout claim",
        "sample_json": str(SAMPLE_JSON),
        "tolerance": tol,
        "terms": rows,
        "checks": checks,
        "failed_checks": failed,
        "error": error,
        "interpretation": {
            "same_state_formula_parity_ready": checks["all_same_state_formula_slices_pass"],
            "mujoco_runtime_observation_builder_ready": False,
            "native_mujoco_policy_rollout_allowed": False,
            "why_this_matters": (
                "The current failed videos can be caused by semantically wrong observations even when the 160-D actor "
                "input has the right shape. This audit removes the pure formula/slice-order ambiguity for one official "
                "IsaacLab sample, but the actual MuJoCo runtime builder still has to be validated."
            ),
            "next_step": (
                "Initialize or log a native MuJoCo runtime state with the same terms, then compare each slice before "
                "running PPO/VAE/diffusion videos without root assist."
            ),
        },
    }


def write_outputs(summary: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with TSV_OUT.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "term",
            "dimension",
            "official_dimension",
            "local_dimension",
            "max_abs_error",
            "passed",
            "policy_vs_critic_max_abs_error",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary["terms"])

    failed = summary.get("failed_checks", [])
    lines = [
        "# MuJoCo Observation Same-State Parity Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Generated: `{summary['generated_at']}`",
        "- Scope: one captured IsaacLab state, local NumPy formula recomputation, no MuJoCo physics rollout.",
        "- 当前不得声称完整复现 BeyondMimic；本审计只验证 observation 公式/slice 的同状态样本对齐。",
        "",
        "## Term Errors",
        "",
    ]
    for row in summary["terms"]:
        lines.append(
            f"- `{row['term']}` dim={row['dimension']} max_abs_error={row['max_abs_error']:.6e} "
            f"passed=`{row['passed']}` policy_vs_critic={row['policy_vs_critic_max_abs_error']:.6e}"
        )
    lines.extend(["", "## Failed Checks", ""])
    if failed:
        lines.extend(f"- `{item}`" for item in failed)
    else:
        lines.append("- None for same-state formula parity. MuJoCo runtime builder parity is still pending.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- policy observation 在官方配置里带 corruption/noise，因此不能作为精确公式对齐参考。",
            "- 本审计使用 critic 中同名的无噪声 shared terms 作为 deterministic reference。",
            "- 这一步通过也只说明同状态公式和 slice 顺序正确；还没有证明 MuJoCo runtime 的 state、frame alignment、normalizer 和 last_action 全部正确。",
            "",
        ]
    )
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summary = build_summary()
    write_outputs(summary)
    print(json.dumps({"status": summary["status"], "json": str(JSON_OUT)}, sort_keys=True))
    if summary["status"].startswith("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
