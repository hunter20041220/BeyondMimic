#!/usr/bin/env python3
"""Diagnose why current BeyondMimic control videos are poor.

The report intentionally separates four things that are easy to mix up:

1. Paper/official-code control contract.
2. Current local IsaacLab tracking teacher quality.
3. Current MuJoCo visualization/controller implementation.
4. What must be repaired before producing new PPO/VAE/guided videos.

It does not start training and it does not claim paper-level reproduction.
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT_DIR = ROOT / "res/diagnostics/beyondmimic_control_quality_diagnosis"

EVAL_SOURCES = {
    "scaled_ppo_iter999": ROOT
    / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/"
    "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json",
    "scaled_ppo_best_iter300": ROOT
    / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_best_checkpoint_confirmation_eval/"
    "best_iter_300_tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json",
    "fk_repaired_robot_order_iter999": ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval.json",
    "endpoint_threshold_candidate_iter999": ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_endpoint_threshold_candidate_ppo_checkpoint_eval/"
    "tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_endpoint_threshold_candidate_ppo_checkpoint_eval.json",
}

SUPPORTING_SOURCES = {
    "reward_termination_diagnostic": ROOT
    / "res/report_assets/official_importer_export_scaled_ppo_reward_termination_diagnostic/reward_termination_diagnostic.json",
    "endpoint_threshold_sweep": ROOT
    / "res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_threshold_sweep/"
    "endpoint_threshold_sweep.json",
    "observation_action_schema": ROOT
    / "res/tracking/observation_action_schema_audit/tracking_observation_action_schema_audit.json",
    "action_scale_audit": ROOT / "res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json",
    "mujoco_adapter_gap": ROOT / "mujoco_mp4/res/adapter_gap/mujoco_ppo_adapter_gap_audit.json",
    "mujoco_control_video_summary": ROOT / "mujoco_mp4/res/control_videos/mujoco_control_video_summary.json",
    "required_artifact_absence": ROOT / "res/required_artifact_absence/required_artifact_absence_audit.json",
}

SOURCE_FILES_READ = [
    "prompt06211658.txt",
    "goal.md",
    "README.md",
    "reproduction/PROGRESS.md",
    "reproduction/RUNBOOK.md",
    "reproduction/docs/final_reproduction_report.md",
    "reproduction/docs/known_limitations.md",
    "reproduction/docs/experiment_protocol.md",
    "reproduction/paper/source/tex/method.tex",
    "reproduction/paper/source/root.tex",
    "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py",
    "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py",
    "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py",
    "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py",
    "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py",
    "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py",
    "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py",
    "mujoco_mp4/scripts/mujoco_pd_control_video.py",
    "mujoco_mp4/scripts/mujoco_trace_mesh_video.py",
    "mujoco_mp4/scripts/mujoco_ppo_adapter_gap_audit.py",
    "mujoco_mp4/configs/g1_joint_mapping.yaml",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"_missing": True, "_path": str(path)}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def nested(obj: Any, *keys: str) -> Any:
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def stat_mean(metrics: dict[str, Any], key: str) -> float | None:
    value = nested(metrics, "episode_log_metrics", key, "mean")
    if value is None:
        value = nested(metrics, key, "mean")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def top_stat(metrics: dict[str, Any], key: str) -> float | None:
    value = nested(metrics, key, "mean_over_steps", "mean")
    if value is None:
        value = nested(metrics, key, "mean")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_rate(numerator: Any, denominator: Any) -> float | None:
    try:
        den = float(denominator)
        if den <= 0:
            return None
        return float(numerator) / den
    except (TypeError, ValueError):
        return None


def finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return value if math.isfinite(value) else None


def summarize_eval(name: str, path: Path) -> dict[str, Any]:
    payload = load_json(path)
    metrics = nested(payload, "run", "metrics") or {}
    num_envs = metrics.get("num_envs") or nested(payload, "config", "num_envs")
    total_env_steps = metrics.get("total_env_steps") or nested(payload, "config", "total_env_steps")
    done_count = metrics.get("done_count_total")
    row = {
        "name": name,
        "source_path": str(path),
        "exists": path.is_file(),
        "status": payload.get("status"),
        "checkpoint": metrics.get("checkpoint") or nested(payload, "inputs", "checkpoint"),
        "policy_obs_shape": metrics.get("policy_obs_shape"),
        "critic_obs_shape": metrics.get("critic_obs_shape"),
        "num_envs": num_envs,
        "total_env_steps": total_env_steps,
        "done_count_total": done_count,
        "done_rate": finite_or_none(safe_rate(done_count, total_env_steps)),
        "reward_mean": top_stat(metrics, "reward"),
        "action_abs_mean": finite_or_none(top_stat(metrics, "action_abs_mean_over_steps")),
        "action_abs_max": finite_or_none(top_stat(metrics, "action_abs_max_over_steps")),
        "error_anchor_pos_mean": stat_mean(metrics, "Metrics/motion/error_anchor_pos"),
        "error_anchor_rot_mean": stat_mean(metrics, "Metrics/motion/error_anchor_rot"),
        "error_body_pos_mean": stat_mean(metrics, "Metrics/motion/error_body_pos"),
        "error_body_rot_mean": stat_mean(metrics, "Metrics/motion/error_body_rot"),
        "error_joint_pos_mean": stat_mean(metrics, "Metrics/motion/error_joint_pos"),
        "error_joint_vel_mean": stat_mean(metrics, "Metrics/motion/error_joint_vel"),
        "termination_ee_body_pos_mean_per_step": stat_mean(metrics, "Episode_Termination/ee_body_pos"),
        "termination_anchor_pos_mean_per_step": stat_mean(metrics, "Episode_Termination/anchor_pos"),
        "termination_anchor_ori_mean_per_step": stat_mean(metrics, "Episode_Termination/anchor_ori"),
        "claim_level": nested(payload, "interpretation", "claim_level") or "local_virtual_tracking_eval",
        "paper_level_tracking_eval_complete": nested(payload, "interpretation", "paper_level_tracking_eval_complete"),
    }
    if row["termination_ee_body_pos_mean_per_step"] is not None and num_envs:
        row["termination_ee_body_pos_rate_per_step"] = finite_or_none(
            safe_rate(row["termination_ee_body_pos_mean_per_step"], num_envs)
        )
    if row["termination_anchor_pos_mean_per_step"] is not None and num_envs:
        row["termination_anchor_pos_rate_per_step"] = finite_or_none(
            safe_rate(row["termination_anchor_pos_mean_per_step"], num_envs)
        )
    return row


def summarize_mujoco_videos(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    rows = payload.get("rows", [])
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "name": row.get("name"),
                "mp4": row.get("mp4"),
                "mp4_exists": row.get("mp4_exists"),
                "duration_seconds": row.get("duration_seconds"),
                "uses_mj_step": row.get("uses_mj_step"),
                "uses_root_assist_controller": row.get("uses_root_assist_controller"),
                "native_mujoco_ppo_obs_adapter": row.get("native_mujoco_ppo_obs_adapter"),
                "joint_error_abs_mean": row.get("joint_error_abs_mean"),
                "root_position_error_mean_m": row.get("root_position_error_mean_m"),
                "root_xy_abs_max": row.get("root_xy_abs_max"),
                "claim_level": row.get("claim_level"),
            }
        )
    return out


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    eval_rows = payload["tracking_eval_rows"]
    video_rows = payload["mujoco_video_rows"]
    findings = payload["diagnosis_findings"]
    recommendations = payload["recommended_repair_order"]
    lines = [
        "# BeyondMimic Control Quality Diagnosis",
        "",
        f"Generated: `{payload['timestamp_utc']}`",
        "",
        "## Executive Conclusion",
        "",
        "The current poor MuJoCo movement-control videos are not just a rendering problem. They expose two upstream issues:",
        "",
        "- The best local tracking teachers are still weak local virtual PPO checkpoints, with low reward, high done rates, and large body/joint errors.",
        "- The current MuJoCo control videos do not run a faithful native PPO/VAE/guided controller. They use target joint/IK sequences, MuJoCo PD actuators, and root assist, so they cannot prove BeyondMimic-style learned control.",
        "",
        "This project still must not claim full paper-level BeyondMimic reproduction.",
        "",
        "## Paper And Official-Code Contract",
        "",
        "- Policy observation: 160-D concatenation of generated motion command, anchor position/orientation error, base linear/angular velocity, relative joint position, relative joint velocity, and previous action.",
        "- Action: 29-D normalized joint position setpoint command, converted with per-joint action scale and executed by low-level PD.",
        "- Robot target bodies: 14 G1 bodies with torso as anchor.",
        "- Tracking rewards: anchor global pose plus relative body position/orientation/linear-velocity/angular-velocity terms with paper tolerances around position 0.3 m, orientation 0.4 rad, linear velocity 1.0, angular velocity 3.14.",
        "- Official training scale in source config: 4096 envs and 30000 PPO iterations; local runs are resource-adjusted and shorter.",
        "- Important termination terms: anchor z, anchor orientation, and end-effector z-only body-position errors.",
        "",
        "## Current Tracking Teacher Metrics",
        "",
        "| name | reward_mean | done_rate | ee_term_rate | anchor_pos_term_rate | body_pos_err | joint_pos_err | action_abs_mean | claim |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in eval_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["name"]),
                    fmt(row.get("reward_mean")),
                    fmt(row.get("done_rate")),
                    fmt(row.get("termination_ee_body_pos_rate_per_step")),
                    fmt(row.get("termination_anchor_pos_rate_per_step")),
                    fmt(row.get("error_body_pos_mean")),
                    fmt(row.get("error_joint_pos_mean")),
                    fmt(row.get("action_abs_mean")),
                    str(row.get("claim_level")),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Current MuJoCo Video Evidence",
        "",
        "| video | duration | mj_step | root_assist | native_adapter | joint_error | root_error | claim |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in video_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("name")),
                    fmt(row.get("duration_seconds")),
                    str(row.get("uses_mj_step")),
                    str(row.get("uses_root_assist_controller")),
                    str(row.get("native_mujoco_ppo_obs_adapter")),
                    fmt(row.get("joint_error_abs_mean")),
                    fmt(row.get("root_position_error_mean_m")),
                    str(row.get("claim_level")),
                ]
            )
            + " |"
        )
    lines += [
        "",
        "## Diagnosis Findings",
        "",
    ]
    for item in findings:
        lines.append(f"- **{item['id']}**: {item['finding']} Evidence: {item['evidence']}")
    lines += [
        "",
        "## Recommended Repair Order",
        "",
    ]
    for idx, item in enumerate(recommendations, start=1):
        lines.append(f"{idx}. **{item['step']}**: {item['why']}")
    lines += [
        "",
        "## Claim Boundary",
        "",
        "Current claim level: local virtual partial reproduction and diagnostics only. The MuJoCo videos can be used as diagnostic/report media, but not as official BeyondMimic PPO/VAE/guided control evidence. Real-robot deployment remains unavailable.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eval_rows = [summarize_eval(name, path) for name, path in EVAL_SOURCES.items()]
    support = {name: load_json(path) for name, path in SUPPORTING_SOURCES.items()}
    video_rows = summarize_mujoco_videos(SUPPORTING_SOURCES["mujoco_control_video_summary"])
    reward_diag = support["reward_termination_diagnostic"]
    threshold_sweep = support["endpoint_threshold_sweep"]
    adapter_gap = support["mujoco_adapter_gap"]
    absence = support["required_artifact_absence"]

    finding_rows = [
        {
            "id": "weak_tracking_teacher",
            "severity": "high",
            "finding": "The current strongest local PPO teachers still have low reward and high done/termination rates; downstream VAE/diffusion can only imitate this weak behavior.",
            "evidence": "scaled_ppo_iter999 reward_mean={reward} done_rate={done}; scaled PPO termination diagnostic dominant component={component} fraction={frac}".format(
                reward=fmt(eval_rows[0].get("reward_mean")),
                done=fmt(eval_rows[0].get("done_rate")),
                component=nested(reward_diag, "metrics", "dominant_final_termination_component"),
                frac=fmt(nested(reward_diag, "metrics", "dominant_final_termination_fraction")),
            ),
        },
        {
            "id": "endpoint_body_semantics",
            "severity": "high",
            "finding": "Endpoint z-only body-position termination dominates or remains a major failure source. Relaxing the endpoint threshold improves done rate but changes the evaluator and does not repair paper-level tracking.",
            "evidence": "threshold_sweep best_threshold={thr}, best_done_rate={done}, why_not_paper_level={why}".format(
                thr=nested(threshold_sweep, "comparison_to_baselines", "best_threshold"),
                done=fmt(nested(threshold_sweep, "comparison_to_baselines", "best_done_rate")),
                why=nested(threshold_sweep, "interpretation", "why_not_paper_level"),
            ),
        },
        {
            "id": "mujoco_native_adapter_missing",
            "severity": "high",
            "finding": "The native MuJoCo 160-D IsaacLab-compatible observation/action adapter is not complete, so the IsaacLab PPO checkpoint cannot be honestly claimed as a MuJoCo closed-loop policy.",
            "evidence": "adapter_gap checks native_mujoco_adapter_complete={complete}; actor input/output dimensions are 160/29.".format(
                complete=nested(adapter_gap, "checks", "native_mujoco_adapter_complete")
            ),
        },
        {
            "id": "pd_root_assist_video_boundary",
            "severity": "medium",
            "finding": "The current MuJoCo control videos use PD target tracking plus root-assist external forces; they are useful diagnostics but not learned unassisted humanoid control.",
            "evidence": "all current control rows mark native_mujoco_ppo_obs_adapter=False; most controller rows use_root_assist_controller=True.",
        },
        {
            "id": "missing_official_level_c_artifacts",
            "severity": "medium",
            "finding": "Official BeyondMimic VAE/diffusion checkpoints, true DAgger rollout logs, paper-level Fig.5/Fig.6 videos, TensorRT engine, and real-robot evidence remain absent.",
            "evidence": nested(absence, "interpretation", "why_not_complete"),
        },
    ]

    recommendation_rows = [
        {
            "step": "Stop regenerating paper-claim videos from the current weak teacher",
            "why": "Longer MP4s will not fix the controller if the source PPO reward/done/body-error metrics remain poor.",
        },
        {
            "step": "Repair tracking target semantics before training more",
            "why": "Compare FK-repaired motion body_pos_w/body_quat_w, G1 target-body order, wrist endpoint z, reset pose, and target refresh against the official MotionCommand contract.",
        },
        {
            "step": "Train/evaluate a stronger IsaacLab tracking teacher first",
            "why": "Use the official 160-D observation and 29-D action path in IsaacLab non-render mode; track reward, done breakdown, body/joint errors, action distribution, and multi-seed variance.",
        },
        {
            "step": "Only then rebuild teacher rollout, VAE, state-latent dataset, diffusion, and guidance",
            "why": "Low reconstruction or denoising MSE against a weak teacher does not imply high-quality humanoid control.",
        },
        {
            "step": "Implement a MuJoCo adapter as a separate validation project",
            "why": "A valid adapter must mirror generated_commands, anchor-frame errors, base velocities, joint relative states, last action, action scale, empirical normalization, reset/termination semantics, and actuator limits before loading IsaacLab PPO weights.",
        },
        {
            "step": "If MuJoCo is the final simulator, train a native MuJoCo tracking policy",
            "why": "Directly transferring an IsaacLab PPO checkpoint into MuJoCo without matching physics, observation manager, normalization, and actuator semantics is not reliable.",
        },
    ]

    payload = {
        "status": "ok_beyondmimic_control_quality_diagnosis",
        "timestamp_utc": utc_now(),
        "experiment_type": "beyondmimic_control_quality_diagnosis",
        "scope": "diagnose current poor movement-control videos; no training launched",
        "goal_complete": False,
        "paper_level_reproduction_complete": False,
        "source_files_read": SOURCE_FILES_READ,
        "source_jsons": {name: str(path) for name, path in {**EVAL_SOURCES, **SUPPORTING_SOURCES}.items()},
        "paper_official_contract": {
            "policy_obs_dim": 160,
            "action_dim": 29,
            "target_body_count": 14,
            "anchor_body": "torso_link",
            "control_frequency_hz": 50.0,
            "sim_dt": 0.005,
            "decimation": 4,
            "official_config_num_envs": 4096,
            "official_config_max_iterations": 30000,
            "reward_sigmas": {"position": 0.3, "orientation": 0.4, "linear_velocity": 1.0, "angular_velocity": 3.14},
            "pd_natural_frequency_hz": 10.0,
            "damping_ratio": 2.0,
            "endpoint_z_termination_threshold_m": 0.25,
        },
        "tracking_eval_rows": eval_rows,
        "mujoco_video_rows": video_rows,
        "diagnosis_findings": finding_rows,
        "recommended_repair_order": recommendation_rows,
        "checks": {
            "all_eval_sources_present": all(row["exists"] for row in eval_rows),
            "mujoco_adapter_gap_documented": nested(adapter_gap, "checks", "native_mujoco_adapter_complete") is False,
            "mujoco_control_videos_not_native_policy": all(row.get("native_mujoco_ppo_obs_adapter") is False for row in video_rows),
            "does_not_claim_complete_reproduction": True,
            "does_not_claim_real_robot": True,
            "does_not_launch_training": True,
        },
        "outputs": {
            "json": str(OUT_DIR / "control_quality_diagnosis.json"),
            "tsv": str(OUT_DIR / "control_quality_diagnosis_findings.tsv"),
            "markdown": str(OUT_DIR / "control_quality_diagnosis.md"),
        },
    }

    write_json(OUT_DIR / "control_quality_diagnosis.json", payload)
    write_tsv(
        OUT_DIR / "control_quality_diagnosis_findings.tsv",
        finding_rows,
        ["id", "severity", "finding", "evidence"],
    )
    write_markdown(OUT_DIR / "control_quality_diagnosis.md", payload)
    print(json.dumps({"status": payload["status"], "outputs": payload["outputs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
