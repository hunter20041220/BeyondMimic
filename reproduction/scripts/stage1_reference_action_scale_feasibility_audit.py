#!/usr/bin/env python3
"""Check whether selected reference motions are reachable under G1 action scale.

The official tracking action is a target joint-position offset:

    theta_sp = theta_default + action_scale * action

This audit reconstructs the nominal default pose and action scales from the
public whole_body_tracking G1 configuration, then computes the action values
that would be required to directly set reference joint targets. This is not a
policy or control success claim; it is a diagnostic for scale/default-pose
feasibility and possible saturation pressure.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT_DIR = ROOT / "res/audits/stage1_reference_action_scale_feasibility"
JSON_OUT = OUT_DIR / "stage1_reference_action_scale_feasibility_audit.json"
TSV_OUT = OUT_DIR / "stage1_reference_action_scale_feasibility_audit.tsv"
MD_OUT = OUT_DIR / "stage1_reference_action_scale_feasibility_audit.md"

RUNTIME_PROBE = ROOT / "res/audits/stage1_singleleg_runtime_contract_probe/stage1_singleleg_runtime_contract_probe.json"
MOTION_BUNDLE_AUDIT = ROOT / "res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle.json"
G1_SOURCE = (
    ROOT
    / "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py"
)

MOTIONS = {
    "hub_singleleg_rootxy0": ROOT
    / "res/tracking/stage1_short_motion_recentered_bundle/motions/hub_singleleg_video_single_leg_stand_1_rootxy0/motion.npz",
    "official_short_walk1_subject1": ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/motions/walk1_subject1/motion.npz",
    "official_short_jumps1_subject1": ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/motions/jumps1_subject1/motion.npz",
    "lafan1_jumps1_subject1_full": ROOT
    / "res/tracking/stage1_multisource_motion_bundle/motions/lafan1_jumps1_subject1/motion.npz",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def official_g1_constants() -> dict[str, float]:
    arm_5020 = 0.003609725
    arm_7520_14 = 0.010177520
    arm_7520_22 = 0.025101925
    arm_4010 = 0.00425
    natural_freq = 10 * 2.0 * math.pi
    return {
        "ARMATURE_5020": arm_5020,
        "ARMATURE_7520_14": arm_7520_14,
        "ARMATURE_7520_22": arm_7520_22,
        "ARMATURE_4010": arm_4010,
        "STIFFNESS_5020": arm_5020 * natural_freq**2,
        "STIFFNESS_7520_14": arm_7520_14 * natural_freq**2,
        "STIFFNESS_7520_22": arm_7520_22 * natural_freq**2,
        "STIFFNESS_4010": arm_4010 * natural_freq**2,
        "NATURAL_FREQ": natural_freq,
    }


def default_joint_pos(joint_names: list[str]) -> np.ndarray:
    defaults = np.zeros(len(joint_names), dtype=np.float64)
    patterns = [
        (r".*_hip_pitch_joint", -0.312),
        (r".*_knee_joint", 0.669),
        (r".*_ankle_pitch_joint", -0.363),
        (r".*_elbow_joint", 0.6),
        (r"left_shoulder_roll_joint", 0.2),
        (r"left_shoulder_pitch_joint", 0.2),
        (r"right_shoulder_roll_joint", -0.2),
        (r"right_shoulder_pitch_joint", 0.2),
    ]
    for i, name in enumerate(joint_names):
        for pat, value in patterns:
            if re.fullmatch(pat.replace(".*", ".*"), name):
                defaults[i] = value
    return defaults


def action_scale_for_joint(name: str, c: dict[str, float]) -> float:
    # G1_ACTION_SCALE is 0.25 * effort_limit / stiffness. These branches mirror
    # whole_body_tracking/robots/g1.py.
    if re.fullmatch(r".*_hip_yaw_joint", name):
        return 0.25 * 88.0 / c["STIFFNESS_7520_14"]
    if re.fullmatch(r".*_hip_roll_joint", name):
        return 0.25 * 139.0 / c["STIFFNESS_7520_22"]
    if re.fullmatch(r".*_hip_pitch_joint", name):
        return 0.25 * 88.0 / c["STIFFNESS_7520_14"]
    if re.fullmatch(r".*_knee_joint", name):
        return 0.25 * 139.0 / c["STIFFNESS_7520_22"]
    if re.fullmatch(r".*_ankle_pitch_joint", name) or re.fullmatch(r".*_ankle_roll_joint", name):
        return 0.25 * 50.0 / (2.0 * c["STIFFNESS_5020"])
    if name in {"waist_roll_joint", "waist_pitch_joint"}:
        return 0.25 * 50.0 / (2.0 * c["STIFFNESS_5020"])
    if name == "waist_yaw_joint":
        return 0.25 * 88.0 / c["STIFFNESS_7520_14"]
    if re.fullmatch(r".*_shoulder_pitch_joint", name) or re.fullmatch(r".*_shoulder_roll_joint", name):
        return 0.25 * 25.0 / c["STIFFNESS_5020"]
    if re.fullmatch(r".*_shoulder_yaw_joint", name) or re.fullmatch(r".*_elbow_joint", name):
        return 0.25 * 25.0 / c["STIFFNESS_5020"]
    if re.fullmatch(r".*_wrist_roll_joint", name):
        return 0.25 * 25.0 / c["STIFFNESS_5020"]
    if re.fullmatch(r".*_wrist_pitch_joint", name) or re.fullmatch(r".*_wrist_yaw_joint", name):
        return 0.25 * 5.0 / c["STIFFNESS_4010"]
    raise KeyError(name)


def joint_names() -> list[str]:
    probe = load_json(RUNTIME_PROBE)
    names = probe.get("robot_joint_names", [])
    if isinstance(names, list) and len(names) == 29:
        return [str(n) for n in names]
    # Fallback to the official Unitree G1 order used by whole_body_tracking export.
    return [
        "left_hip_pitch_joint",
        "right_hip_pitch_joint",
        "waist_yaw_joint",
        "left_hip_roll_joint",
        "right_hip_roll_joint",
        "waist_roll_joint",
        "left_hip_yaw_joint",
        "right_hip_yaw_joint",
        "waist_pitch_joint",
        "left_knee_joint",
        "right_knee_joint",
        "left_shoulder_pitch_joint",
        "right_shoulder_pitch_joint",
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
        "left_shoulder_yaw_joint",
        "right_shoulder_yaw_joint",
        "left_elbow_joint",
        "right_elbow_joint",
        "left_wrist_roll_joint",
        "right_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "right_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_yaw_joint",
    ]


def motion_row(name: str, path: Path, names: list[str], scale: np.ndarray, default: np.ndarray) -> dict[str, Any]:
    row: dict[str, Any] = {"name": name, "path": str(path), "exists": path.is_file(), "sha256": sha256_file(path)}
    if not path.is_file():
        return row
    data = np.load(path)
    q = np.asarray(data["joint_pos"], dtype=np.float64)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    required_action = (q - default[None, :]) / scale[None, :]
    abs_req = np.abs(required_action)
    per_joint_max = np.nanmax(abs_req, axis=0)
    top = np.argsort(-per_joint_max)[:10]
    groups = {
        "legs": [
            i
            for i, joint_name in enumerate(names)
            if any(token in joint_name for token in ["hip_", "knee_", "ankle_"])
        ],
        "torso": [i for i, joint_name in enumerate(names) if joint_name.startswith("waist_")],
        "arms_no_wrists": [
            i
            for i, joint_name in enumerate(names)
            if any(token in joint_name for token in ["shoulder_", "elbow_"])
        ],
        "wrists": [i for i, joint_name in enumerate(names) if "wrist_" in joint_name],
    }

    def group_stats(indexes: list[int]) -> dict[str, float | int]:
        if not indexes:
            return {"joint_count": 0}
        x = abs_req[:, indexes]
        return {
            "joint_count": int(len(indexes)),
            "required_action_abs_mean": float(np.nanmean(x)),
            "required_action_abs_p95": float(np.nanpercentile(x, 95)),
            "required_action_abs_max": float(np.nanmax(x)),
            "fraction_required_abs_gt_1": float(np.mean(x > 1.0)),
            "fraction_required_abs_gt_2": float(np.mean(x > 2.0)),
            "fraction_required_abs_gt_3": float(np.mean(x > 3.0)),
        }

    grouped = {key: group_stats(value) for key, value in groups.items()}
    row.update(
        {
            "frames": int(q.shape[0]),
            "fps": fps,
            "duration_seconds": float(q.shape[0] / fps),
            "action_scale_min": float(np.min(scale)),
            "action_scale_max": float(np.max(scale)),
            "required_action_abs_mean": float(np.nanmean(abs_req)),
            "required_action_abs_p95": float(np.nanpercentile(abs_req, 95)),
            "required_action_abs_p99": float(np.nanpercentile(abs_req, 99)),
            "required_action_abs_max": float(np.nanmax(abs_req)),
            "fraction_required_abs_gt_1": float(np.mean(abs_req > 1.0)),
            "fraction_required_abs_gt_2": float(np.mean(abs_req > 2.0)),
            "fraction_required_abs_gt_3": float(np.mean(abs_req > 3.0)),
            "grouped_required_action_stats": grouped,
            "top_required_action_joints": [
                {
                    "joint_name": names[int(i)],
                    "joint_index": int(i),
                    "scale": float(scale[int(i)]),
                    "default": float(default[int(i)]),
                    "required_abs_max": float(per_joint_max[int(i)]),
                    "reference_min": float(np.nanmin(q[:, int(i)])),
                    "reference_max": float(np.nanmax(q[:, int(i)])),
                }
                for i in top
            ],
        }
    )
    row["action_scale_feasibility_status"] = (
        "high_saturation_pressure"
        if row["fraction_required_abs_gt_1"] > 0.25
        or row["required_action_abs_p95"] > 2.0
        or grouped["legs"].get("required_action_abs_p95", 0.0) > 1.5
        else "moderate_or_low_static_scale_pressure"
    )
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = joint_names()
    constants = official_g1_constants()
    scale = np.asarray([action_scale_for_joint(n, constants) for n in names], dtype=np.float64)
    default = default_joint_pos(names)
    rows = [motion_row(name, path, names, scale, default) for name, path in MOTIONS.items()]
    payload = {
        "status": "ok_stage1_reference_action_scale_feasibility_audit",
        "generated_at": utc_now(),
        "claim_level": "diagnostic_only_not_policy_success",
        "inputs": {
            "runtime_probe": str(RUNTIME_PROBE),
            "motion_bundle_audit": str(MOTION_BUNDLE_AUDIT),
            "g1_source": str(G1_SOURCE),
        },
        "joint_names": names,
        "default_joint_pos": default.tolist(),
        "action_scale": scale.tolist(),
        "checks": {
            "joint_count_29": len(names) == 29,
            "all_action_scales_positive": bool(np.all(scale > 0)),
            "motion_rows_present": all(row.get("exists") for row in rows),
            "goal_complete": False,
        },
        "rows": rows,
        "interpretation": {
            "how_to_read": (
                "Values are the normalized action that would be needed to set the reference joint target directly. "
                "PPO policies are not required to exactly equal this static inverse, but large values indicate strong "
                "saturation pressure under theta_sp = theta_default + action_scale * action."
            ),
            "claim_boundary": (
                "This audit does not prove or disprove policy learning. It is a diagnostic for action-scale/default-pose "
                "compatibility before generating teacher/VAE/diffusion videos."
            ),
            "important_caveat": (
                "Wrists often dominate the global normalized action range because official wrist scales are small. "
                "The grouped leg statistics should be used when diagnosing gait or single-leg support feasibility."
            ),
        },
        "outputs": {"json": str(JSON_OUT), "tsv": str(TSV_OUT), "markdown": str(MD_OUT)},
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with TSV_OUT.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "name",
            "exists",
            "duration_seconds",
            "required_action_abs_p95",
            "required_action_abs_max",
            "fraction_required_abs_gt_1",
            "fraction_required_abs_gt_2",
            "legs_required_action_abs_p95",
            "legs_fraction_required_abs_gt_1",
            "wrists_required_action_abs_p95",
            "action_scale_feasibility_status",
            "path",
        ]
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            out = {k: row.get(k) for k in fields}
            grouped = row.get("grouped_required_action_stats", {})
            if isinstance(grouped, dict):
                out["legs_required_action_abs_p95"] = grouped.get("legs", {}).get("required_action_abs_p95")
                out["legs_fraction_required_abs_gt_1"] = grouped.get("legs", {}).get(
                    "fraction_required_abs_gt_1"
                )
                out["wrists_required_action_abs_p95"] = grouped.get("wrists", {}).get("required_action_abs_p95")
            writer.writerow(out)
    lines = [
        "# Stage-1 Reference Action-Scale Feasibility Audit",
        "",
        f"- Status: `{payload['status']}`",
        f"- Claim level: `{payload['claim_level']}`",
        "",
        "This audit checks the static action magnitude implied by official G1 action scale.",
        "",
        "| Motion | all p95 | legs p95 | wrists p95 | frac > 1 | status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        grouped = row.get("grouped_required_action_stats", {})
        legs_p95 = grouped.get("legs", {}).get("required_action_abs_p95", float("nan"))
        wrists_p95 = grouped.get("wrists", {}).get("required_action_abs_p95", float("nan"))
        lines.append(
            f"| `{row['name']}` | {row.get('required_action_abs_p95', float('nan')):.3f} | "
            f"{legs_p95:.3f} | {wrists_p95:.3f} | "
            f"{row.get('fraction_required_abs_gt_1', float('nan')):.3f} | "
            f"`{row.get('action_scale_feasibility_status')}` |"
        )
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "json": str(JSON_OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
