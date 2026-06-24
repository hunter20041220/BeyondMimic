#!/usr/bin/env python3
"""Audit why the current teacher/VAE/diffusion chain should not produce success videos yet.

The goal of this script is not to re-label failed MuJoCo videos as success.  It
records the hard gate that currently blocks trustworthy downstream VAE,
diffusion, and guidance rollouts: the Stage-1 teacher still fails strict
single-motion screening, and the local MuJoCo videos do not yet implement a
validated native IsaacLab observation/action adapter.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/model_chain_teacher_gate_root_cause"

OFFICIAL_ACTION = (
    ROOT
    / "download/dependencies/IsaacLab-v2.1.0/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py"
)
OFFICIAL_TRACKING_CFG = (
    ROOT
    / "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/"
    "tracking_env_cfg.py"
)
OFFICIAL_REWARDS = (
    ROOT
    / "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/"
    "rewards.py"
)
OFFICIAL_TERMINATIONS = (
    ROOT
    / "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/"
    "terminations.py"
)
OFFICIAL_PPO = (
    ROOT
    / "download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/"
    "agents/rsl_rl_ppo_cfg.py"
)
ACTION_SCALE_AUDIT = ROOT / "res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json"
ADAPTER_GAP = ROOT / "mujoco_mp4/res/adapter_gap/mujoco_ppo_adapter_gap_audit.json"

EVALS = {
    "rootxy0_model250_strict": ROOT
    / "res/tracking/hub_singleleg_paper_contract_ppo_checkpoint_eval_rootxy0_model250_strict_20260624_132900/"
    "tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.json",
    "rootxy0_model500_strict": ROOT
    / "res/tracking/hub_singleleg_paper_contract_ppo_checkpoint_eval_rootxy0_model500_strict_20260624_213700/"
    "tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.json",
    "relaxed_model1000_strict_eval": ROOT
    / "res/tracking/hub_singleleg_paper_contract_ppo_checkpoint_eval_model1000_strict_overridefix_20260624_124800/"
    "tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.json",
}

MOTIONS = {
    "hub_singleleg": ROOT
    / "res/tracking/stage1_multisource_motion_bundle/motions/hub_singleleg_video_single_leg_stand_1/motion.npz",
    "lafan1_walk1_subject1": ROOT
    / "res/tracking/stage1_multisource_motion_bundle/motions/lafan1_walk1_subject1/motion.npz",
    "lafan1_jumps1_subject1": ROOT
    / "res/tracking/stage1_multisource_motion_bundle/motions/lafan1_jumps1_subject1/motion.npz",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore") if path.is_file() else ""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "category",
        "item",
        "status",
        "evidence",
        "claim_level",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def eval_summary(name: str, path: Path) -> dict[str, Any]:
    payload = read_json(path)
    q = payload.get("quality_gate", {})
    run_metrics = payload.get("run", {}).get("metrics", {})
    ee = run_metrics.get("episode_log_metrics", {}).get("Episode_Termination/ee_body_pos", {})
    return {
        "name": name,
        "path": str(path),
        "exists": path.is_file(),
        "checkpoint": payload.get("inputs", {}).get("checkpoint") or run_metrics.get("checkpoint"),
        "loaded_iteration": run_metrics.get("loaded_iteration"),
        "quality_gate_passed": bool(q.get("passed", False)),
        "reward_mean": q.get("reward_mean"),
        "local_non_timeout_done_rate": q.get("local_non_timeout_done_rate"),
        "error_body_pos_mean": q.get("error_body_pos_mean"),
        "error_joint_pos_mean": q.get("error_joint_pos_mean"),
        "ee_body_pos_termination_mean_per_step": ee.get("mean"),
        "action_abs_mean": (run_metrics.get("action_abs_mean_over_steps") or {}).get("mean"),
        "action_abs_max": (run_metrics.get("action_abs_max_over_steps") or {}).get("mean"),
    }


def controller_default_position_by_joint(joint_names: list[str]) -> np.ndarray:
    defaults = []
    for name in joint_names:
        value = 0.0
        if name.endswith("_hip_pitch_joint"):
            value = -0.312
        elif name.endswith("_knee_joint"):
            value = 0.669
        elif name.endswith("_ankle_pitch_joint"):
            value = -0.33
        elif name == "left_shoulder_roll_joint":
            value = 0.2
        elif name == "left_shoulder_pitch_joint":
            value = 0.2
        elif name == "left_elbow_joint":
            value = 0.6
        elif name == "right_shoulder_roll_joint":
            value = -0.2
        elif name == "right_shoulder_pitch_joint":
            value = 0.2
        elif name == "right_elbow_joint":
            value = 0.6
        defaults.append(value)
    return np.asarray(defaults, dtype=np.float64)


def motion_action_equivalent_summary(name: str, path: Path, joint_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not path.is_file():
        return {"name": name, "path": str(path), "exists": False}
    motion = np.load(path, allow_pickle=True)
    joint_pos = np.asarray(motion["joint_pos"], dtype=np.float64)
    joint_names = [str(row["joint_name"]) for row in joint_rows]
    alpha = np.asarray([float(row["action_scale"]) for row in joint_rows], dtype=np.float64)
    theta0 = controller_default_position_by_joint(joint_names)
    action_equiv = (joint_pos - theta0) / alpha
    frac_per_joint = (np.abs(action_equiv) > 1.0).mean(axis=0)
    worst = np.argsort(-frac_per_joint)[:8]
    return {
        "name": name,
        "path": str(path),
        "exists": True,
        "frames": int(joint_pos.shape[0]),
        "mean_abs_action_equivalent": float(np.mean(np.abs(action_equiv))),
        "p95_abs_action_equivalent": float(np.percentile(np.abs(action_equiv), 95)),
        "mean_fraction_dims_outside_abs1": float(frac_per_joint.mean()),
        "joints_outside_abs1_more_than_50pct": int((frac_per_joint > 0.5).sum()),
        "worst_joints": [
            {
                "joint_name": joint_names[int(i)],
                "fraction_outside_abs1": float(frac_per_joint[int(i)]),
                "action_equiv_min": float(action_equiv[:, int(i)].min()),
                "action_equiv_max": float(action_equiv[:, int(i)].max()),
            }
            for i in worst
        ],
    }


def contains(path: Path, needle: str) -> bool:
    return needle in read_text(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    action_scale_payload = read_json(ACTION_SCALE_AUDIT)
    joint_rows = action_scale_payload.get("joint_rows", [])
    evals = {name: eval_summary(name, path) for name, path in EVALS.items()}
    motions = {
        name: motion_action_equivalent_summary(name, path, joint_rows)
        for name, path in MOTIONS.items()
    }
    adapter = read_json(ADAPTER_GAP)

    any_strict_teacher_passed = any(
        row["quality_gate_passed"]
        for key, row in evals.items()
        if "strict" in key or "rootxy0" in key
    )

    checks = {
        "official_joint_action_affine_formula_found": contains(
            OFFICIAL_ACTION, "self._processed_actions = self._raw_actions * self._scale + self._offset"
        ),
        "official_joint_action_has_optional_clip_only": contains(OFFICIAL_ACTION, "if self.cfg.clip is not None"),
        "official_tracking_uses_default_offset": contains(OFFICIAL_TRACKING_CFG, "use_default_offset=True"),
        "official_tracking_obs_terms_present": all(
            contains(OFFICIAL_TRACKING_CFG, term)
            for term in [
                "command = ObsTerm",
                "motion_anchor_pos_b",
                "motion_anchor_ori_b",
                "base_lin_vel",
                "joint_pos",
                "joint_vel",
                "actions = ObsTerm",
            ]
        ),
        "official_reward_terms_present": all(
            contains(OFFICIAL_TRACKING_CFG, term)
            for term in [
                "motion_global_anchor_pos",
                "motion_body_pos",
                "motion_body_lin_vel",
                "action_rate_l2",
                "undesired_contacts",
            ]
        ),
        "official_ppo_30000_iterations": contains(OFFICIAL_PPO, "max_iterations = 30000"),
        "rootxy0_model250_gate_failed": not evals["rootxy0_model250_strict"]["quality_gate_passed"],
        "rootxy0_model500_gate_failed": not evals["rootxy0_model500_strict"]["quality_gate_passed"],
        "strict_teacher_gate_passed": any_strict_teacher_passed,
        "native_mujoco_adapter_complete": bool(
            adapter.get("checks", {}).get("native_mujoco_adapter_complete", False)
        ),
        "downstream_video_generation_allowed": any_strict_teacher_passed
        and bool(adapter.get("checks", {}).get("native_mujoco_adapter_complete", False)),
        "does_not_claim_full_reproduction": True,
        "does_not_claim_real_robot": True,
    }

    rows: list[dict[str, Any]] = [
        {
            "category": "official_formula",
            "item": "joint_action_contract",
            "status": "ok_static_contract",
            "evidence": str(OFFICIAL_ACTION),
            "claim_level": "official IsaacLab action formula",
            "notes": "IsaacLab applies processed_action = raw_action * scale + offset; clipping only exists when configured.",
        },
        {
            "category": "official_formula",
            "item": "tracking_mdp_contract",
            "status": "ok_static_contract",
            "evidence": str(OFFICIAL_TRACKING_CFG),
            "claim_level": "official whole_body_tracking MDP",
            "notes": "Policy obs/rewards/PPO configuration match the public tracking code contract.",
        },
    ]
    for item in evals.values():
        rows.append(
            {
                "category": "teacher_gate",
                "item": item["name"],
                "status": "failed" if not item["quality_gate_passed"] else "passed",
                "evidence": str(item["path"]),
                "claim_level": "local single-motion strict teacher screening",
                "notes": (
                    f"iter={item['loaded_iteration']} reward={item['reward_mean']} "
                    f"done_rate={item['local_non_timeout_done_rate']} body_err={item['error_body_pos_mean']} "
                    f"joint_err={item['error_joint_pos_mean']}"
                ),
            }
        )
    for item in motions.values():
        rows.append(
            {
                "category": "motion_distribution",
                "item": item["name"],
                "status": "informational",
                "evidence": str(item["path"]),
                "claim_level": "reference action-equivalent diagnostic",
                "notes": (
                    f"mean_abs_action_equiv={item.get('mean_abs_action_equivalent')} "
                    f"p95={item.get('p95_abs_action_equivalent')} "
                    f"frac_dims_outside_abs1={item.get('mean_fraction_dims_outside_abs1')}"
                ),
            }
        )
    rows.append(
        {
            "category": "mujoco_adapter",
            "item": "native_mujoco_observation_action_adapter",
            "status": "gap" if not checks["native_mujoco_adapter_complete"] else "ok",
            "evidence": str(ADAPTER_GAP),
            "claim_level": "local MuJoCo adapter audit",
            "notes": adapter.get("interpretation", {}).get("next_step_for_native_control", ""),
        }
    )
    rows.append(
        {
            "category": "decision",
            "item": "downstream_vae_diffusion_guidance_video_gate",
            "status": "blocked_by_teacher_or_adapter" if not checks["downstream_video_generation_allowed"] else "allowed",
            "evidence": "strict teacher quality gate and native MuJoCo adapter audit",
            "claim_level": "hard gate for generating success videos",
            "notes": "Do not generate success-claimed VAE/diffusion/guidance videos until strict teacher gate and adapter gate both pass.",
        }
    )

    payload = {
        "status": "blocked_by_teacher_quality_and_native_adapter_gap"
        if not checks["downstream_video_generation_allowed"]
        else "ok_downstream_video_generation_allowed",
        "timestamp_utc": utc_now(),
        "experiment_type": "model_chain_teacher_gate_root_cause_audit",
        "scope": "Stage-1 teacher gate and formula/adapter root cause for failed single-leg/walk/jump model-chain videos",
        "checks": checks,
        "teacher_evals": evals,
        "motion_action_equivalent": motions,
        "rows": rows,
        "interpretation": {
            "main_conclusion": (
                "The public IsaacLab/whole_body_tracking action, observation, reward, and PPO contracts are present. "
                "The current blocker is not proven to be a paper-formula typo; the latest strict single-leg teachers "
                "still fail the local quality gate, and native MuJoCo PPO/VAE/diffusion observation-action parity is "
                "not complete. Downstream success videos should remain gated."
            ),
            "why_front_leaning_videos_are_not_success": (
                "A weak or frequently-reset teacher rollout induces a narrow failure distribution; VAE and diffusion "
                "trained on that data can reconstruct/denoise the failure distribution without learning the reference "
                "single-leg, walk, or jump posture."
            ),
            "next_steps": [
                "Keep Stage-1 teacher training/eval separate from downstream VAE/diffusion video generation.",
                "Evaluate later strict checkpoints, then only collect teacher rollouts from a checkpoint passing the local gate.",
                "Finish native MuJoCo observation/action parity before claiming MuJoCo PPO closed-loop control.",
                "Regenerate VAE/diffusion/guidance videos only after upstream teacher and adapter gates pass.",
            ],
            "goal_complete": False,
        },
        "outputs": {
            "json": str(OUT / "model_chain_teacher_gate_root_cause_audit.json"),
            "tsv": str(OUT / "model_chain_teacher_gate_root_cause_audit.tsv"),
        },
    }
    write_json(OUT / "model_chain_teacher_gate_root_cause_audit.json", payload)
    write_tsv(OUT / "model_chain_teacher_gate_root_cause_audit.tsv", rows)
    print(json.dumps({"status": payload["status"], "json": payload["outputs"]["json"]}, sort_keys=True))


if __name__ == "__main__":
    main()
