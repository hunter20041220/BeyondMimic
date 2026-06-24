#!/usr/bin/env python3
"""Audit MuJoCo observation math parity against IsaacLab helper formulas.

This is a formula-level gate, not a physics rollout and not a substitute for
capturing an IsaacLab observation_manager sample.  It verifies that the local
NumPy quaternion/frame helpers used by the MuJoCo native PPO probe agree with
IsaacLab's official torch helpers for:

* matrix_from_quat / rot6 flattening
* yaw_quat
* subtract_frame_transforms
* body-frame linear/angular velocity conversion

The audit also records an important remaining gap: the current native MuJoCo
probe still uses a local world_to_init reference alignment for the reference
anchor.  That may be useful for centered visualization, but it has not been
numerically proven equivalent to the official training-time observation manager.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/audits/mujoco_observation_math_parity"
JSON_OUT = OUT / "mujoco_observation_math_parity_audit.json"
TSV_OUT = OUT / "mujoco_observation_math_parity_audit.tsv"
MD_OUT = OUT / "mujoco_observation_math_parity_audit.md"

FILES = {
    "isaaclab_math": ROOT
    / "reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab/isaaclab/utils/math.py",
    "official_observations": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/observations.py",
    "official_commands": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/commands.py",
    "native_probe": ROOT / "reproduction/scripts/stage1_multisource_quality_gated_native_ppo_mujoco_probe.py",
    "bm_tracking_python": ROOT / "envs/bm_tracking/bin/python",
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def run_math_fixture() -> dict[str, Any]:
    python = FILES["bm_tracking_python"]
    if not python.is_file():
        return {"status": "failed", "error": f"missing python: {python}"}
    code = r'''
import json
import numpy as np
import torch

from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_inv, quat_mul, subtract_frame_transforms, yaw_quat

rng = np.random.default_rng(20260624)
n = 32
eps = 1e-10

def norm_np(q):
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)

def qmul_np(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    w1, x1, y1, z1 = np.moveaxis(a, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(b, -1, 0)
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)

def qconj_np(q):
    q = np.asarray(q, dtype=np.float64).copy()
    q[..., 1:] *= -1.0
    return q

def qapply_np(q, v):
    q = norm_np(q)
    vq = np.concatenate([np.zeros(v.shape[:-1] + (1,), dtype=np.float64), v], axis=-1)
    return qmul_np(qmul_np(q, vq), qconj_np(q))[..., 1:4]

def qmat_np(q):
    q = norm_np(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        1 - 2 * (y * y + z * z),
        2 * (x * y - z * w),
        2 * (x * z + y * w),
        2 * (x * y + z * w),
        1 - 2 * (x * x + z * z),
        2 * (y * z - x * w),
        2 * (x * z - y * w),
        2 * (y * z + x * w),
        1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))

def yaw_np(q):
    q = norm_np(q)
    qw, qx, qy, qz = np.moveaxis(q, -1, 0)
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    out = np.zeros_like(q)
    out[..., 0] = np.cos(yaw / 2)
    out[..., 3] = np.sin(yaw / 2)
    return norm_np(out)

def sign_inv_max(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(min(np.max(np.abs(a - b)), np.max(np.abs(a + b))))

q1 = norm_np(rng.normal(size=(n, 4)))
q2 = norm_np(rng.normal(size=(n, 4)))
t1 = rng.normal(size=(n, 3))
t2 = rng.normal(size=(n, 3))
vel = rng.normal(size=(n, 3))

q1_t = torch.tensor(q1, dtype=torch.float64)
q2_t = torch.tensor(q2, dtype=torch.float64)
t1_t = torch.tensor(t1, dtype=torch.float64)
t2_t = torch.tensor(t2, dtype=torch.float64)
vel_t = torch.tensor(vel, dtype=torch.float64)

mat_ref = matrix_from_quat(q1_t).cpu().numpy()
mat_np = qmat_np(q1)

yaw_ref = yaw_quat(q1_t).cpu().numpy()
yaw_local = yaw_np(q1)

pos_ref_t, quat_ref_t = subtract_frame_transforms(t1_t, q1_t, t2_t, q2_t)
pos_ref = pos_ref_t.cpu().numpy()
quat_ref = quat_ref_t.cpu().numpy()
pos_local = qapply_np(qconj_np(q1), t2 - t1)
quat_local = norm_np(qmul_np(qconj_np(q1), q2))

rot6_ref = matrix_from_quat(quat_ref_t).cpu().numpy()[..., :2].reshape(n, -1)
rot6_local = qmat_np(quat_local)[..., :2].reshape(n, -1)

base_ref = quat_apply(quat_inv(q1_t), vel_t).cpu().numpy()
base_local = qapply_np(qconj_np(q1), vel)

delta_ori_ref = yaw_quat(quat_mul(q1_t, quat_inv(q2_t))).cpu().numpy()
delta_ori_local = yaw_np(qmul_np(q1, qconj_np(q2)))

out = {
    "status": "ok",
    "fixture_count": n,
    "max_abs_error": {
        "matrix_from_quat": float(np.max(np.abs(mat_ref - mat_np))),
        "yaw_quat_sign_invariant": sign_inv_max(yaw_ref, yaw_local),
        "subtract_frame_transforms_position": float(np.max(np.abs(pos_ref - pos_local))),
        "subtract_frame_transforms_quat_sign_invariant": sign_inv_max(quat_ref, quat_local),
        "rot6_flatten_order": float(np.max(np.abs(rot6_ref - rot6_local))),
        "base_velocity_body_frame": float(np.max(np.abs(base_ref - base_local))),
        "motion_command_delta_yaw_sign_invariant": sign_inv_max(delta_ori_ref, delta_ori_local),
    },
}
print(json.dumps(out, sort_keys=True))
'''
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab")
    proc = subprocess.run(
        [str(python), "-c", code],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
        check=False,
    )
    payload: dict[str, Any] = {
        "python": str(python),
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
    }
    if proc.returncode != 0:
        payload["status"] = "failed"
        payload["error"] = "fixture_subprocess_failed"
        return payload
    try:
        parsed = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:  # pragma: no cover - audit retains failure.
        payload["status"] = "failed"
        payload["error"] = f"json_parse_failed: {exc!r}"
        return payload
    payload.update(parsed)
    return payload


def build_checks(fixture: dict[str, Any], native_source: str) -> dict[str, bool]:
    errors = fixture.get("max_abs_error", {})
    tol = 1e-9
    formula_checks = {
        key: bool(float(value) <= tol)
        for key, value in errors.items()
        if isinstance(value, int | float)
    }
    native_builds_terms = all(
        needle in native_source
        for needle in [
            "command = np.concatenate([ref_joint_pos, ref_joint_vel])",
            "motion_anchor_pos_b = local_pos(robot_anchor_pos, robot_anchor_quat, ref_anchor_pos)",
            "motion_anchor_ori_b = rot6(local_quat(robot_anchor_quat, ref_anchor_quat))",
            "base_lin_vel_b = quat_apply(quat_conj(root_quat), data.qvel[0:3].copy())",
            "base_ang_vel_b = quat_apply(quat_conj(root_quat), data.qvel[3:6].copy())",
            "joint_pos_rel = data.qpos[7 : 7 + 29].copy() - default_joint_pos",
            "joint_vel_rel = data.qvel[6 : 6 + 29].copy()",
            "last_action",
        ]
    )
    uses_world_to_init = "world_to_init_translation" in native_source and "world_to_init_yaw" in native_source
    checks: dict[str, bool] = {
        "isaaclab_math_import_ok": fixture.get("status") == "ok",
        **{f"{key}_parity": passed for key, passed in formula_checks.items()},
        "all_formula_fixtures_pass": bool(formula_checks) and all(formula_checks.values()),
        "native_probe_builds_all_8_policy_terms": native_builds_terms,
        "native_probe_declares_approximate": "approximate 160-D observation" in native_source,
        "native_probe_uses_world_to_init_anchor_alignment": uses_world_to_init,
        "native_probe_anchor_alignment_proven_equivalent_to_observation_manager": False,
        "runtime_observation_manager_sample_available": False,
        "native_obs_runtime_parity_ready": False,
        "does_not_claim_rollout_or_success": True,
    }
    return checks


def write_outputs(summary: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = [
        {
            "check": key,
            "passed": bool(value),
            "notes": "n/a",
        }
        for key, value in summary["checks"].items()
    ]
    with TSV_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["check", "passed", "notes"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    failed = [row["check"] for row in rows if not row["passed"]]
    lines = [
        "# MuJoCo Observation Math Parity Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Generated: `{summary['generated_at']}`",
        "- Scope: formula fixture only; no IsaacLab task rollout, no MuJoCo physics claim.",
        "- 当前不得声称完整复现 BeyondMimic；本审计只证明部分 observation 数学公式与 IsaacLab helper 对齐。",
        "",
        "## Max Absolute Errors",
        "",
    ]
    for key, value in summary["fixture"].get("max_abs_error", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Failed / Blocking Checks", ""])
    if failed:
        lines.extend(f"- `{item}`" for item in failed)
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Quaternion, Rot6D, yaw-only quaternion, frame subtraction, and body-frame velocity formulas match IsaacLab math fixtures.",
            "- The current native probe still uses a local `world_to_init` anchor alignment for centered MuJoCo visualization.",
            "- That alignment has not been proven numerically equivalent to the official IsaacLab observation_manager output.",
            "- Therefore native MuJoCo PPO/VAE/diffusion videos remain blocked until a real observation_manager parity sample is captured.",
            "",
        ]
    )
    MD_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    fixture = run_math_fixture()
    native_source = read_text(FILES["native_probe"])
    checks = build_checks(fixture, native_source)
    status = (
        "blocked_observation_runtime_parity_missing_but_math_fixtures_pass"
        if checks["all_formula_fixtures_pass"]
        else "blocked_observation_math_fixture_mismatch"
    )
    summary = {
        "status": status,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "mujoco_observation_math_parity_audit",
        "claim_level": "formula_fixture_only; no native MuJoCo policy rollout claim",
        "files": {key: str(path) for key, path in FILES.items()},
        "fixture": fixture,
        "checks": checks,
        "hard_blockers": [
            "runtime_observation_manager_sample_available is false",
            "native_probe_anchor_alignment_proven_equivalent_to_observation_manager is false",
            "native_obs_runtime_parity_ready is false",
        ],
        "next_step": (
            "Capture an IsaacLab observation_manager sample for the same motion phase/state/last_action and compare "
            "all eight policy slices against the native MuJoCo observation builder."
        ),
        "interpretation": {
            "formula_math_parity_ready": checks["all_formula_fixtures_pass"],
            "runtime_observation_manager_parity_ready": False,
            "native_obs_adapter_ready": False,
            "success_video_claim_allowed": False,
        },
    }
    write_outputs(summary)
    print(json.dumps({"status": status, "json": str(JSON_OUT), "md": str(MD_OUT), "tsv": str(TSV_OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
