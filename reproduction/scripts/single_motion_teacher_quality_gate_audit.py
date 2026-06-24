#!/usr/bin/env python3
"""Audit whether current single-motion teachers are good enough for downstream models."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/audits/single_motion_teacher_quality_gate"

THRESHOLDS = {
    "reward_mean_min": 0.10,
    "error_body_pos_mean_max_m": 0.25,
    "error_joint_pos_mean_max_rad": 1.00,
    "local_non_timeout_done_rate_max": 0.05,
}


def read_json(rel_path: str) -> dict[str, Any]:
    path = ROOT / rel_path
    if not path.is_file():
        return {"_missing": True, "_path": str(path)}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - audit must record parse errors.
        return {"_parse_error": str(exc), "_path": str(path)}
    data["_path"] = str(path)
    return data


def nested(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def metric_mean(data: dict[str, Any], metric_name: str) -> float | None:
    direct = as_float(nested(data, metric_name, "mean"))
    if direct is not None:
        return direct
    return as_float(nested(data, "command_metrics_timeseries", metric_name, "mean"))


def done_rate_from_task_eval(data: dict[str, Any]) -> float | None:
    step_count = as_float(data.get("step_count"))
    num_envs = as_float(data.get("num_envs")) or 1.0
    terminated_total = as_float(data.get("terminated_total"))
    done_total = as_float(data.get("done_total"))
    truncated_total = as_float(data.get("truncated_total")) or 0.0
    if step_count is None or step_count <= 0:
        return None
    denominator = step_count * max(num_envs, 1.0)
    if terminated_total is not None:
        return terminated_total / denominator
    if done_total is not None:
        return max(done_total - truncated_total, 0.0) / denominator
    return None


def gate_row(
    *,
    motion: str,
    source: str,
    evidence_rel_path: str,
    is_teacher_eval: bool,
    claim_level: str,
    reward_mean: float | None,
    error_body_pos_mean: float | None,
    error_joint_pos_mean: float | None,
    local_non_timeout_done_rate: float | None,
    notes: str,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    passes_reward = reward_mean is not None and reward_mean >= THRESHOLDS["reward_mean_min"]
    passes_body_error = (
        error_body_pos_mean is not None and error_body_pos_mean <= THRESHOLDS["error_body_pos_mean_max_m"]
    )
    passes_joint_error = (
        error_joint_pos_mean is not None and error_joint_pos_mean <= THRESHOLDS["error_joint_pos_mean_max_rad"]
    )
    passes_done_rate = (
        local_non_timeout_done_rate is not None
        and local_non_timeout_done_rate <= THRESHOLDS["local_non_timeout_done_rate_max"]
    )
    teacher_quality_passed = (
        is_teacher_eval and passes_reward and passes_body_error and passes_joint_error and passes_done_rate
    )
    return {
        "motion": motion,
        "source": source,
        "evidence_path": str(ROOT / evidence_rel_path),
        "exists": (ROOT / evidence_rel_path).is_file(),
        "is_teacher_eval": is_teacher_eval,
        "claim_level": claim_level,
        "checkpoint": checkpoint or "",
        "reward_mean": reward_mean,
        "error_body_pos_mean": error_body_pos_mean,
        "error_joint_pos_mean": error_joint_pos_mean,
        "local_non_timeout_done_rate": local_non_timeout_done_rate,
        "threshold_reward_mean_min": THRESHOLDS["reward_mean_min"],
        "threshold_error_body_pos_mean_max_m": THRESHOLDS["error_body_pos_mean_max_m"],
        "threshold_error_joint_pos_mean_max_rad": THRESHOLDS["error_joint_pos_mean_max_rad"],
        "threshold_local_non_timeout_done_rate_max": THRESHOLDS["local_non_timeout_done_rate_max"],
        "passes_reward": passes_reward,
        "passes_body_error": passes_body_error,
        "passes_joint_error": passes_joint_error,
        "passes_done_rate": passes_done_rate,
        "teacher_quality_passed": teacher_quality_passed,
        "allowed_for_downstream_vae_diffusion": teacher_quality_passed,
        "allowed_for_success_video_claim": teacher_quality_passed,
        "required_fix": "" if teacher_quality_passed else "retrain_or_revalidate_stage1_teacher_before_downstream",
        "notes": notes,
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    paper_sweep_rel = (
        "res/tracking/g1_official_importer_export_paper_contract_ppo_checkpoint_sweep/"
        "tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_sweep.json"
    )
    paper_sweep = read_json(paper_sweep_rel)
    best = paper_sweep.get("best_checkpoint", {})
    rows.append(
        gate_row(
            motion="paper_contract_public_bundle_best_teacher",
            source="official_importer_export_paper_contract_checkpoint_sweep",
            evidence_rel_path=paper_sweep_rel,
            is_teacher_eval=True,
            claim_level="local_best_candidate_teacher_screening_not_paper_level",
            checkpoint=best.get("checkpoint"),
            reward_mean=as_float(best.get("reward_mean")),
            error_body_pos_mean=as_float(best.get("error_body_pos_mean")),
            error_joint_pos_mean=as_float(best.get("error_joint_pos_mean")),
            local_non_timeout_done_rate=as_float(best.get("local_non_timeout_done_rate")),
            notes="Best public-bundle checkpoint is still too weak for downstream VAE/diffusion success claims.",
        )
    )

    multisource_rel = (
        "res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/"
        "tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.json"
    )
    multisource = read_json(multisource_rel)
    best = multisource.get("best_checkpoint", {})
    rows.append(
        gate_row(
            motion="stage1_multisource_best_teacher",
            source="stage1_multisource_checkpoint_sweep",
            evidence_rel_path=multisource_rel,
            is_teacher_eval=True,
            claim_level="local_multisource_teacher_screening_not_paper_level",
            checkpoint=best.get("checkpoint"),
            reward_mean=as_float(best.get("reward_mean")),
            error_body_pos_mean=as_float(best.get("error_body_pos_mean")),
            error_joint_pos_mean=as_float(best.get("error_joint_pos_mean")),
            local_non_timeout_done_rate=as_float(best.get("local_non_timeout_done_rate")),
            notes="This is the 5/6-card multi-source teacher candidate; it also fails the quality gate.",
        )
    )

    singleleg_rel = (
        "res/tracking/hub_singleleg_paper_contract_ppo_checkpoint_eval/"
        "tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.json"
    )
    singleleg = read_json(singleleg_rel)
    gate = singleleg.get("quality_gate", {})
    rows.append(
        gate_row(
            motion="hub_singleleg_video_single_leg_stand_1",
            source="hub_singleleg_single_motion_teacher_eval",
            evidence_rel_path=singleleg_rel,
            is_teacher_eval=True,
            claim_level="local_single_motion_teacher_quality_screening",
            checkpoint=nested(singleleg, "inputs", "checkpoint"),
            reward_mean=as_float(gate.get("reward_mean")),
            error_body_pos_mean=as_float(gate.get("error_body_pos_mean")),
            error_joint_pos_mean=as_float(gate.get("error_joint_pos_mean")),
            local_non_timeout_done_rate=as_float(gate.get("local_non_timeout_done_rate")),
            notes="Short Sequence target currently fails because non-timeout done rate is high.",
        )
    )

    jumps_robot_rel = (
        "res/tracking/g1_official_importer_export_fk_repaired_robot_order_split_task_eval/"
        "motions/jumps1_subject1/jumps1_subject1_task_eval_metrics.json"
    )
    jumps_robot = read_json(jumps_robot_rel)
    rows.append(
        gate_row(
            motion="lafan1_jumps1_subject1",
            source="robot_order_fk_repaired_single_motion_task_eval",
            evidence_rel_path=jumps_robot_rel,
            is_teacher_eval=True,
            claim_level="local_single_motion_task_eval_screening_not_trained_success",
            checkpoint="",
            reward_mean=metric_mean(jumps_robot, "reward"),
            error_body_pos_mean=metric_mean(jumps_robot, "error_body_pos"),
            error_joint_pos_mean=metric_mean(jumps_robot, "error_joint_pos"),
            local_non_timeout_done_rate=done_rate_from_task_eval(jumps_robot),
            notes="Best existing jumps1 task-eval variant has tolerable body error but low reward, high joint error, and too many terminations.",
        )
    )

    jumps_resource_rel = (
        "res/tracking/g1_official_csv_loop_full_dataset_task_eval/"
        "motions/jumps1_subject1/jumps1_subject1_task_eval_metrics.json"
    )
    jumps_resource = read_json(jumps_resource_rel)
    rows.append(
        gate_row(
            motion="lafan1_jumps1_subject1",
            source="resource_adjusted_single_motion_task_eval",
            evidence_rel_path=jumps_resource_rel,
            is_teacher_eval=True,
            claim_level="local_single_motion_task_eval_screening_not_trained_success",
            checkpoint="",
            reward_mean=metric_mean(jumps_resource, "reward"),
            error_body_pos_mean=metric_mean(jumps_resource, "error_body_pos"),
            error_joint_pos_mean=metric_mean(jumps_resource, "error_joint_pos"),
            local_non_timeout_done_rate=done_rate_from_task_eval(jumps_resource),
            notes="Resource-adjusted jumps1 task eval has lower body error but still fails reward, joint error, and done-rate gates.",
        )
    )

    jumps_ref_rel = (
        "res/visualization/lafan1_jumps1_subject1_mujoco/stable_dynamic_164s_179s/"
        "lafan1_jumps1_subject1_mujoco_summary.json"
    )
    rows.append(
        gate_row(
            motion="lafan1_jumps1_subject1",
            source="stable_dynamic_reference_action_baseline",
            evidence_rel_path=jumps_ref_rel,
            is_teacher_eval=False,
            claim_level="local_mujoco_reference_action_baseline_not_learned_control",
            reward_mean=None,
            error_body_pos_mean=None,
            error_joint_pos_mean=None,
            local_non_timeout_done_rate=None,
            notes="Reference/action baseline can be used for visualization only and does not unlock downstream teacher rollout collection.",
        )
    )

    singleleg_ref_rel = (
        "res/visualization/hub_singleleg_full_chain_scaled_ppo_pure/"
        "source_singleleg_reference_replay/source_singleleg_reference_replay_summary.json"
    )
    rows.append(
        gate_row(
            motion="hub_singleleg_video_single_leg_stand_1",
            source="source_singleleg_reference_replay",
            evidence_rel_path=singleleg_ref_rel,
            is_teacher_eval=False,
            claim_level="source_reference_replay_not_learned_control",
            reward_mean=None,
            error_body_pos_mean=None,
            error_joint_pos_mean=None,
            local_non_timeout_done_rate=None,
            notes="This verifies that a source single-leg pose exists; it does not prove any policy, VAE, or diffusion controller learned it.",
        )
    )

    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "motion",
        "source",
        "exists",
        "is_teacher_eval",
        "claim_level",
        "reward_mean",
        "error_body_pos_mean",
        "error_joint_pos_mean",
        "local_non_timeout_done_rate",
        "passes_reward",
        "passes_body_error",
        "passes_joint_error",
        "passes_done_rate",
        "teacher_quality_passed",
        "allowed_for_downstream_vae_diffusion",
        "allowed_for_success_video_claim",
        "required_fix",
        "evidence_path",
        "checkpoint",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: ("none" if row.get(field) in (None, "") else row.get(field)) for field in fields})


def write_md(path: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    lines = [
        "# 单动作 Teacher 质量硬门控",
        "",
        f"生成时间：`{payload['generated_at']}`",
        "",
        "## 结论",
        "",
        (
            "当前 `jumps1_subject1` 和 Short Sequence `Single Leg Balance` 的 teacher 证据"
            "**没有通过**进入下游 VAE / diffusion / guidance 训练所需的质量门控。"
        ),
        "",
        "Reference replay / reference action-control baseline 仍然可以用于展示源动作，但它们不是 learned control。",
        "",
        "## 门控阈值",
        "",
        "| 指标 | 要求 |",
        "|---|---:|",
        f"| 平均 reward | >= {THRESHOLDS['reward_mean_min']} |",
        f"| 平均 body position error | <= {THRESHOLDS['error_body_pos_mean_max_m']} m |",
        f"| 平均 joint position error | <= {THRESHOLDS['error_joint_pos_mean_max_rad']} rad |",
        f"| non-timeout done rate | <= {THRESHOLDS['local_non_timeout_done_rate_max']} |",
        "",
        "## 证据表",
        "",
        "| 动作 | 来源 | Reward | Body err | Joint err | Done rate | Teacher 通过 | Claim level |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        def fmt(value: Any) -> str:
            if value is None or value == "":
                return "n/a"
            if isinstance(value, float):
                return f"{value:.6g}"
            return str(value)

        lines.append(
            "| "
            + " | ".join(
                [
                    row["motion"],
                    row["source"],
                    fmt(row["reward_mean"]),
                    fmt(row["error_body_pos_mean"]),
                    fmt(row["error_joint_pos_mean"]),
                    fmt(row["local_non_timeout_done_rate"]),
                    str(row["teacher_quality_passed"]),
                    row["claim_level"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- `paper_contract_public_bundle_best_teacher`：reward 约 `0.021`，done rate 约 `0.154`，不能作为可靠 teacher。",
            "- `stage1_multisource_best_teacher`：reward 约 `0.024`，body error 约 `1.01 m`，不应继续喂给下游模型。",
            "- `hub_singleleg_video_single_leg_stand_1`：这是一次真实的本地单动作 teacher 尝试，但主要因为 done rate 太高而失败。",
            "- `jumps1_subject1`：reference baseline 有展示价值，但现有 task/eval 指标不能证明 learned tracking 成功。",
            "",
            "## 对后续流程的影响",
            "",
            "- 当前 teacher 证据不能解锁 VAE / diffusion / guidance 的成功视频生成。",
            "- 下一步应该先做 corrective single-motion Stage-1 teacher training/evaluation，而不是继续长时间训练下游模型。",
            "- 本审计不标记 BeyondMimic 复现目标完成。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    teacher_rows = [row for row in rows if row["is_teacher_eval"]]
    teacher_passed = [row for row in teacher_rows if row["teacher_quality_passed"]]
    payload = {
        "status": "blocked_single_motion_teacher_quality_gate",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claim_level": "audit_gate_only_no_success_video_claim",
        "thresholds": THRESHOLDS,
        "checks": {
            "jumps1_teacher_quality_passed": any(
                row["motion"] == "lafan1_jumps1_subject1" and row["teacher_quality_passed"] for row in rows
            ),
            "singleleg_teacher_quality_passed": any(
                row["motion"] == "hub_singleleg_video_single_leg_stand_1" and row["teacher_quality_passed"]
                for row in rows
            ),
            "any_teacher_quality_passed": bool(teacher_passed),
            "reference_baseline_does_not_unlock_downstream": all(
                not row["allowed_for_downstream_vae_diffusion"] for row in rows if not row["is_teacher_eval"]
            ),
            "downstream_vae_diffusion_allowed": bool(teacher_passed),
            "success_video_generation_allowed": bool(teacher_passed),
            "does_not_claim_goal_complete": True,
        },
        "blocking_reason": (
            "Existing teacher evaluations do not meet the conservative reward/error/done-rate gate. "
            "Reference replay can demonstrate source motions, but cannot be used as proof of learned control."
        ),
        "recommended_next_steps": [
            "Run corrective single-motion Stage-1 training for hub_singleleg_video_single_leg_stand_1.",
            "Run corrective single-motion Stage-1 training for lafan1_jumps1_subject1 stable windows.",
            "Before MuJoCo policy videos, validate same-state observation parity against IsaacLab observation_manager.",
            "Only after a teacher passes this gate should VAE/diffusion/guidance be retrained for success videos.",
        ],
        "rows": rows,
    }
    (OUT / "single_motion_teacher_quality_gate_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_tsv(OUT / "single_motion_teacher_quality_gate_audit.tsv", rows)
    write_md(OUT / "single_motion_teacher_quality_gate_audit.md", payload)
    print(json.dumps({"status": payload["status"], "rows": len(rows), "out": str(OUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
