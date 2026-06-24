#!/usr/bin/env python3
"""Audit why current single-leg/jumps teacher videos should remain blocked.

This is a lightweight evidence script: it reads local motion bundles, official
whole_body_tracking source snippets, and current checkpoint-eval JSON files. It
does not launch IsaacLab, stop training, train models, or render videos.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT_DIR = ROOT / "res/audits/stage1_singleleg_jumps_motion_teacher_failure"
JSON_OUT = OUT_DIR / "stage1_singleleg_jumps_motion_teacher_failure_audit.json"
TSV_OUT = OUT_DIR / "stage1_singleleg_jumps_motion_teacher_failure_audit.tsv"
MD_OUT = OUT_DIR / "stage1_singleleg_jumps_motion_teacher_failure_audit.md"

OFFICIAL_FILES = [
    ROOT
    / "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py",
    ROOT
    / "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py",
    ROOT
    / "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py",
    ROOT
    / "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py",
    ROOT
    / "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py",
    ROOT
    / "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py",
    ROOT
    / "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py",
]

PAPER_FILES = [
    ROOT / "reproduction/paper/source/tex/method.tex",
    ROOT / "reproduction/paper/source/root.tex",
    ROOT / "reproduction/paper/source/tex/results.tex",
]

MOTIONS = {
    "hub_singleleg_rootxy0": ROOT
    / "res/tracking/stage1_short_motion_recentered_bundle/motions/hub_singleleg_video_single_leg_stand_1_rootxy0/motion.npz",
    "hub_singleleg_original": ROOT
    / "res/tracking/stage1_multisource_motion_bundle/motions/hub_singleleg_video_single_leg_stand_1/motion.npz",
    "lafan1_jumps1_subject1_full": ROOT
    / "res/tracking/stage1_multisource_motion_bundle/motions/lafan1_jumps1_subject1/motion.npz",
    "official_short_jumps1_subject1": ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/motions/jumps1_subject1/motion.npz",
    "official_short_walk1_subject1": ROOT
    / "res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/motions/walk1_subject1/motion.npz",
}

EVAL_JSONS = [
    ROOT
    / "res/tracking/hub_singleleg_paper_contract_ppo_checkpoint_eval_rootxy0_model1000_strict_20260624_222900/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.json",
    ROOT
    / "res/tracking/hub_singleleg_paper_contract_ppo_checkpoint_eval_rootxy0_model1000_refresh_strict_20260624_223900/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.json",
    ROOT
    / "res/tracking/hub_singleleg_paper_contract_ppo_checkpoint_eval_rootxy0_refresh_model250_20260624_231000/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.json",
    ROOT
    / "res/tracking/hub_singleleg_paper_contract_ppo_checkpoint_eval_rootxy0_refresh_model250_relaxed0p5_20260624_231400/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.json",
    ROOT
    / "res/tracking/hub_singleleg_paper_contract_ppo_checkpoint_eval_rootxy0_refresh_model500_20260624_231600/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.json",
    ROOT
    / "res/tracking/hub_singleleg_paper_contract_ppo_checkpoint_eval_rootxy0_refresh_model750_20260624_232500/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.json",
    ROOT
    / "res/tracking/hub_singleleg_paper_contract_ppo_checkpoint_eval_rootxy0_refresh_model1000_20260624_234100/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.json",
]

MOTION_BUNDLE_AUDIT = ROOT / "res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle.json"

TARGET_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
ENDPOINT_BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]


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


def get_body_names() -> list[str]:
    data = load_json(MOTION_BUNDLE_AUDIT)
    names = data.get("robot_body_names", [])
    return names if isinstance(names, list) else []


def motion_stats(name: str, path: Path, body_names: list[str]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "name": name,
        "path": str(path),
        "exists": path.is_file(),
        "sha256": sha256_file(path),
    }
    if not path.is_file():
        return row
    data = np.load(path)
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    body_pos = np.asarray(data["body_pos_w"], dtype=np.float64)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    row.update(
        {
            "frames": int(joint_pos.shape[0]),
            "fps": fps,
            "duration_seconds": float(joint_pos.shape[0] / fps),
            "joint_pos_shape": list(joint_pos.shape),
            "body_pos_shape": list(body_pos.shape),
            "joint_global_min": float(np.nanmin(joint_pos)),
            "joint_global_max": float(np.nanmax(joint_pos)),
            "joint_median_range": float(np.nanmedian(np.ptp(joint_pos, axis=0))),
            "joint_max_range": float(np.nanmax(np.ptp(joint_pos, axis=0))),
            "joint_velocity_p95_abs": float(np.nanpercentile(np.abs(np.diff(joint_pos, axis=0) * fps), 95))
            if joint_pos.shape[0] > 1
            else 0.0,
            "body_global_z_min": float(np.nanmin(body_pos[..., 2])),
            "body_global_z_max": float(np.nanmax(body_pos[..., 2])),
        }
    )
    endpoint_stats = {}
    for body in ENDPOINT_BODY_NAMES:
        if body not in body_names:
            endpoint_stats[body] = {"exists": False}
            continue
        idx = body_names.index(body)
        z = body_pos[:, idx, 2]
        endpoint_stats[body] = {
            "exists": True,
            "z_min": float(np.nanmin(z)),
            "z_mean": float(np.nanmean(z)),
            "z_max": float(np.nanmax(z)),
            "z_range": float(np.nanmax(z) - np.nanmin(z)),
        }
    row["endpoint_z_stats"] = endpoint_stats
    if "left_ankle_roll_link" in body_names and "right_ankle_roll_link" in body_names:
        li = body_names.index("left_ankle_roll_link")
        ri = body_names.index("right_ankle_roll_link")
        diff = body_pos[:, li, 2] - body_pos[:, ri, 2]
        row["left_minus_right_ankle_z"] = {
            "min": float(np.nanmin(diff)),
            "mean": float(np.nanmean(diff)),
            "max": float(np.nanmax(diff)),
            "abs_max": float(np.nanmax(np.abs(diff))),
        }
    amp = np.ptp(joint_pos, axis=0)
    row["top_joint_ranges"] = [
        {
            "joint_index": int(idx),
            "range": float(amp[idx]),
            "min": float(np.nanmin(joint_pos[:, idx])),
            "max": float(np.nanmax(joint_pos[:, idx])),
        }
        for idx in np.argsort(-amp)[:8]
    ]
    row["contains_visible_singleleg_target"] = bool(
        name.startswith("hub_singleleg")
        and row.get("left_minus_right_ankle_z", {}).get("abs_max", 0.0) >= 0.25
    )
    row["contains_visible_jump_or_step_target"] = bool(
        ("jumps" in name or "walk" in name)
        and max((v.get("z_range", 0.0) for v in endpoint_stats.values() if v.get("exists")), default=0.0) >= 0.1
    )
    return row


def eval_stats(path: Path) -> dict[str, Any]:
    data = load_json(path)
    row: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "status": data.get("status"),
        "quality_gate": data.get("quality_gate"),
    }
    quality_gate = data.get("quality_gate", {})
    if isinstance(quality_gate, dict):
        row["quality_gate_passed"] = quality_gate.get("passed")
        row.update(
            {
                "reward_mean": quality_gate.get("reward_mean"),
                "error_body_pos_mean": quality_gate.get("error_body_pos_mean"),
                "error_joint_pos_mean": quality_gate.get("error_joint_pos_mean"),
                "local_non_timeout_done_rate": quality_gate.get("local_non_timeout_done_rate"),
            }
        )
    metrics = data.get("metrics", {})
    summary = data.get("summary", {})
    if isinstance(metrics, dict):
        for key in [
            "reward_mean",
            "error_body_pos_mean",
            "error_joint_pos_mean",
            "local_non_timeout_done_rate",
        ]:
            row.setdefault(key, metrics.get(key))
    if isinstance(summary, dict):
        row["summary"] = summary
    return row


def source_contract_rows() -> list[dict[str, Any]]:
    rows = []
    checks = [
        (
            "official_policy_obs_contract",
            "Tracking policy actor observation order is command(58), anchor pos/orientation, base twist, joint state, last action.",
            OFFICIAL_FILES[0],
            ["self.commands.motion.body_names", "self.actions.joint_pos.scale = G1_ACTION_SCALE"],
        ),
        (
            "official_reward_contract",
            "Tracking reward uses Gaussian-shaped task-space terms plus action-rate, joint-limit, and contact regularizers.",
            OFFICIAL_FILES[1],
            ["motion_body_pos", "undesired_contacts", "ee_body_pos"],
        ),
        (
            "official_reset_update_contract",
            "MotionCommand refreshes relative body targets in _update_command; reset-time stale targets are a local runtime blocker if not refreshed.",
            OFFICIAL_FILES[2],
            ["def _resample_command", "def _update_command", "self.body_pos_relative_w"],
        ),
        (
            "official_pd_action_scale_contract",
            "Action is a joint position target offset using G1_ACTION_SCALE = 0.25 * effort / stiffness.",
            OFFICIAL_FILES[5],
            ["G1_ACTION_SCALE", "0.25 * e[n] / s[n]", "NATURAL_FREQ = 10"],
        ),
        (
            "paper_stage1_contract",
            "Paper Stage 1 is RL motion tracking with DeepMimic-style task-space rewards and 50 Hz policies.",
            PAPER_FILES[0],
            ["motion-tracking", "reward", "policy"],
        ),
        (
            "paper_stage2_contract",
            "Paper Stage 2 requires true tracking-policy rollouts, VAE/DAgger, state-latent diffusion, and receding-horizon guidance.",
            PAPER_FILES[0],
            ["DAgger", "state", "latent", "guidance"],
        ),
    ]
    for name, purpose, path, needles in checks:
        text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        rows.append(
            {
                "name": name,
                "category": "source_contract",
                "path": str(path),
                "exists": path.is_file(),
                "purpose": purpose,
                "needles": needles,
                "needles_present": [needle in text for needle in needles],
                "status": "pass" if path.is_file() and all(needle in text for needle in needles) else "blocked",
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    body_names = get_body_names()
    motion_rows = [motion_stats(name, path, body_names) for name, path in MOTIONS.items()]
    eval_rows = [eval_stats(path) for path in EVAL_JSONS]
    contract_rows = source_contract_rows()

    latest_refresh_eval = next(
        (
            row
            for row in reversed(eval_rows)
            if row.get("exists") and "refresh_model1000" in row.get("path", "")
        ),
        {},
    )
    motion_has_singleleg_target = any(row.get("contains_visible_singleleg_target") for row in motion_rows)
    motion_has_jumps_target = any(
        row["name"] == "lafan1_jumps1_subject1_full" and row.get("contains_visible_jump_or_step_target")
        for row in motion_rows
    )
    latest_gate_failed = latest_refresh_eval.get("quality_gate_passed") is False

    blockers = []
    if latest_gate_failed:
        blockers.append(
            "Latest refreshed single-leg checkpoint eval still fails quality gate; reward remains low and non-timeout done rate high."
        )
    if motion_has_singleleg_target:
        blockers.append(
            "Single-leg source motion visibly contains a lifted right ankle target, so missing lifted-leg videos are not explained by empty source motion."
        )
    blockers.append(
        "Downstream VAE/diffusion/guidance videos must remain blocked until a teacher checkpoint passes continuous rollout quality gates."
    )

    status = (
        "blocked_teacher_quality_not_motion_absence"
        if motion_has_singleleg_target and motion_has_jumps_target and latest_gate_failed
        else "partial_stage1_motion_teacher_failure_audit"
    )
    payload = {
        "status": status,
        "generated_at": utc_now(),
        "claim_level": "audit_only_no_training_no_video_no_success_claim",
        "checks": {
            "official_contract_sources_present": all(row["exists"] for row in contract_rows),
            "official_contract_needles_present": all(row["status"] == "pass" for row in contract_rows),
            "singleleg_motion_contains_lifted_ankle_target": motion_has_singleleg_target,
            "jumps1_motion_contains_endpoint_height_target": motion_has_jumps_target,
            "latest_refresh_model1000_quality_gate_failed": latest_gate_failed,
            "downstream_video_generation_allowed": False,
            "goal_complete": False,
        },
        "source_contract_rows": contract_rows,
        "motion_rows": motion_rows,
        "eval_rows": eval_rows,
        "blockers": blockers,
        "interpretation": {
            "root_cause_assessment": (
                "The inspected source motions do contain the requested visible postures. Current failures are therefore "
                "best treated as Stage-1 teacher quality / runtime-contract problems, not as absence of target motion. "
                "The reset-target refresh patch fixes an initial stale-target spike but does not yet produce a usable "
                "single-leg teacher by model_500."
            ),
            "do_not_proceed_to_downstream": (
                "Do not train or render final VAE/diffusion/guidance single-leg videos from the current weak teacher. "
                "The teacher gate must pass first."
            ),
            "next_repair_candidates": [
                "Evaluate refreshed model_750/model_1000 when available.",
                "Run a focused action-scale/default-pose feasibility audit: required reference target offsets versus official G1 action scale.",
                "If gate remains failed, start a clearly labeled curriculum diagnostic with relaxed endpoint termination or ankle-only endpoint termination, then verify against strict evaluator.",
                "Use walk1_subject1 or squat_18 only as easier teacher pipeline probes, not as substitutes for Single Leg Balance and jumps1_subject1.",
            ],
        },
        "outputs": {
            "json": str(JSON_OUT),
            "tsv": str(TSV_OUT),
            "markdown": str(MD_OUT),
        },
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    flat_rows: list[dict[str, Any]] = []
    for row in contract_rows:
        flat_rows.append(
            {
                "category": "source_contract",
                "name": row["name"],
                "status": row["status"],
                "path": row["path"],
                "metric": "",
                "value": "",
                "notes": row["purpose"],
            }
        )
    for row in motion_rows:
        flat_rows.append(
            {
                "category": "motion",
                "name": row["name"],
                "status": "pass" if row.get("exists") else "blocked",
                "path": row["path"],
                "metric": "endpoint_or_joint_target",
                "value": json.dumps(
                    {
                        "duration_seconds": row.get("duration_seconds"),
                        "joint_median_range": row.get("joint_median_range"),
                        "joint_max_range": row.get("joint_max_range"),
                        "left_minus_right_ankle_z": row.get("left_minus_right_ankle_z"),
                    },
                    sort_keys=True,
                ),
                "notes": "Motion source target magnitude check.",
            }
        )
    for row in eval_rows:
        flat_rows.append(
            {
                "category": "eval",
                "name": Path(row["path"]).parent.name,
                "status": "failed_quality_gate"
                if row.get("quality_gate_passed") is False
                else str(row.get("status")),
                "path": row["path"],
                "metric": "reward/body_error/joint_error/done_rate",
                "value": json.dumps(
                    {
                        "reward_mean": row.get("reward_mean"),
                        "error_body_pos_mean": row.get("error_body_pos_mean"),
                        "error_joint_pos_mean": row.get("error_joint_pos_mean"),
                        "local_non_timeout_done_rate": row.get("local_non_timeout_done_rate"),
                    },
                    sort_keys=True,
                ),
                "notes": "Teacher checkpoint quality evidence.",
            }
        )
    with TSV_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["category", "name", "status", "path", "metric", "value", "notes"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(flat_rows)

    lines = [
        "# Stage-1 Single-Leg / Jumps Teacher Failure Audit",
        "",
        f"- Status: `{status}`",
        f"- Claim level: `{payload['claim_level']}`",
        "",
        "## Key Findings",
        "",
        "- The inspected single-leg motion contains a visible lifted-ankle target; the source motion is not empty.",
        "- The inspected jumps1 motion contains endpoint-height variation suitable for a later reference/teacher target.",
        "- The latest refreshed single-leg checkpoint evidence available to this audit still fails the quality gate.",
        "- Downstream VAE/diffusion/guidance videos remain blocked until a continuous teacher rollout quality gate passes.",
        "",
        "## Outputs",
        "",
        f"- JSON: `{JSON_OUT}`",
        f"- TSV: `{TSV_OUT}`",
    ]
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "json": str(JSON_OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
