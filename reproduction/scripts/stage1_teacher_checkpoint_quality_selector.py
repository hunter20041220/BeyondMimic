#!/usr/bin/env python3
"""Select usable Stage-1 teacher checkpoints from existing eval evidence.

This is an audit/selection tool. It never starts training or simulation. The
goal is to prevent weak or discontinuous tracking policies from being promoted
into VAE/diffusion data collection just because a checkpoint exists.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT_DIR = ROOT / "res" / "tracking" / "stage1_teacher_checkpoint_quality_selector"
JSON_OUT = OUT_DIR / "stage1_teacher_checkpoint_quality_selector.json"
TSV_OUT = OUT_DIR / "stage1_teacher_checkpoint_quality_selector.tsv"
MD_OUT = OUT_DIR / "stage1_teacher_checkpoint_quality_selector.md"


READINESS_THRESHOLDS = {
    "downstream_ready_done_rate": 0.05,
    "candidate_done_rate": 0.10,
    "max_body_pos_error_mean": 0.20,
    "max_joint_pos_error_mean": 1.00,
    "max_anchor_pos_error_mean": 0.12,
    "max_action_abs_mean": 1.20,
    "min_eval_steps": 250,
    "min_num_envs": 128,
}


METRIC_KEYS = {
    "error_anchor_pos_mean": ("Metrics/motion/error_anchor_pos", "error_anchor_pos"),
    "error_anchor_rot_mean": ("Metrics/motion/error_anchor_rot", "error_anchor_rot"),
    "error_body_pos_mean": ("Metrics/motion/error_body_pos", "error_body_pos"),
    "error_body_rot_mean": ("Metrics/motion/error_body_rot", "error_body_rot"),
    "error_joint_pos_mean": ("Metrics/motion/error_joint_pos", "error_joint_pos"),
    "error_joint_vel_mean": ("Metrics/motion/error_joint_vel", "error_joint_vel"),
    "error_body_lin_vel_mean": ("Metrics/motion/error_body_lin_vel", "error_body_lin_vel"),
    "error_body_ang_vel_mean": ("Metrics/motion/error_body_ang_vel", "error_body_ang_vel"),
}


@dataclass
class Candidate:
    source_json: str
    category: str
    checkpoint: str | None = None
    iteration: int | None = None
    status: str | None = None
    claim_level: str | None = None
    official_beyondmimic_checkpoint: bool = False
    paper_level_tracking_eval: bool = False
    metrics_json: str | None = None
    timeseries_csv: str | None = None
    eval_steps: int | None = None
    num_envs: int | None = None
    total_env_steps: int | None = None
    motion_count: int | None = None
    total_motion_frames: int | None = None
    done_count_total: int | None = None
    timeout_count_total: int | None = None
    local_non_timeout_done_rate: float | None = None
    reward_mean: float | None = None
    action_abs_mean: float | None = None
    action_abs_max: float | None = None
    metrics: dict[str, float | None] = field(default_factory=dict)
    evidence_quality: str = "insufficient"
    downstream_teacher_ready: bool = False
    candidate_teacher_usable: bool = False
    decision: str = "insufficient_evidence"
    score: float = math.inf
    blockers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        row = {
            "category": self.category,
            "iteration": self.iteration,
            "checkpoint": self.checkpoint,
            "source_json": self.source_json,
            "metrics_json": self.metrics_json,
            "timeseries_csv": self.timeseries_csv,
            "status": self.status,
            "claim_level": self.claim_level,
            "official_beyondmimic_checkpoint": self.official_beyondmimic_checkpoint,
            "paper_level_tracking_eval": self.paper_level_tracking_eval,
            "eval_steps": self.eval_steps,
            "num_envs": self.num_envs,
            "total_env_steps": self.total_env_steps,
            "motion_count": self.motion_count,
            "total_motion_frames": self.total_motion_frames,
            "done_count_total": self.done_count_total,
            "timeout_count_total": self.timeout_count_total,
            "local_non_timeout_done_rate": self.local_non_timeout_done_rate,
            "reward_mean": self.reward_mean,
            "action_abs_mean": self.action_abs_mean,
            "action_abs_max": self.action_abs_max,
            "evidence_quality": self.evidence_quality,
            "candidate_teacher_usable": self.candidate_teacher_usable,
            "downstream_teacher_ready": self.downstream_teacher_ready,
            "decision": self.decision,
            "score": None if math.isinf(self.score) else self.score,
            "blockers": "; ".join(self.blockers),
            "notes": "; ".join(self.notes),
        }
        row.update(self.metrics)
        return row


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def stat_mean(value: Any) -> float | None:
    if isinstance(value, dict):
        v = value.get("mean")
        return float(v) if isinstance(v, (int, float)) else None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def nested_get(dct: dict[str, Any], keys: tuple[str, ...]) -> Any:
    cur: Any = dct
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def parse_iteration_from_path(path: Path) -> int | None:
    for part in reversed(path.parts):
        m = re.search(r"iter[_-]?(\d+)", part)
        if m:
            return int(m.group(1))
    m = re.search(r"model_(\d+)\.pt", str(path))
    return int(m.group(1)) if m else None


def resolve_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = ROOT / p
    return p


def read_eval_metrics(metrics_path: str | None) -> dict[str, Any]:
    p = resolve_path(metrics_path)
    if p is None or not p.exists():
        return {}
    return load_json(p) or {}


def extract_motion_metrics(metrics: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    episode_log = metrics.get("episode_log_metrics") if isinstance(metrics.get("episode_log_metrics"), dict) else {}
    motion_metrics = metrics.get("motion_metrics") if isinstance(metrics.get("motion_metrics"), dict) else {}
    for out_key, candidates in METRIC_KEYS.items():
        value = None
        for key in candidates:
            if key in episode_log:
                value = stat_mean(episode_log[key])
                break
            if key in motion_metrics:
                value = stat_mean(motion_metrics[key])
                break
        out[out_key] = value
    return out


def compute_timeseries_checks(timeseries_path: str | None) -> dict[str, Any]:
    p = resolve_path(timeseries_path)
    if p is None or not p.exists():
        return {"exists": False}
    checks: dict[str, Any] = {"exists": True, "rows": 0}
    try:
        with p.open(newline="") as f:
            reader = csv.DictReader(f)
            done_sum = 0.0
            timeout_sum = 0.0
            rewards: list[float] = []
            for row in reader:
                checks["rows"] += 1
                for key, dest in [("done_count", "done_sum"), ("timeout_count", "timeout_sum")]:
                    try:
                        v = float(row.get(key, 0) or 0)
                    except ValueError:
                        v = 0.0
                    if dest == "done_sum":
                        done_sum += v
                    else:
                        timeout_sum += v
                try:
                    rewards.append(float(row.get("reward_mean", "nan")))
                except ValueError:
                    pass
            checks["done_sum_from_timeseries"] = int(done_sum)
            checks["timeout_sum_from_timeseries"] = int(timeout_sum)
            checks["reward_mean_from_timeseries"] = sum(rewards) / len(rewards) if rewards else None
    except Exception as exc:
        checks["error"] = repr(exc)
    return checks


def infer_category(path: Path, data: dict[str, Any]) -> str:
    text = " ".join([str(path), str(data.get("experiment_type", "")), str(data.get("claim_level", ""))]).lower()
    if "stage1_multisource" in text:
        return "stage1_multisource_5_6_gpu"
    if "paper_contract" in text:
        return "official_importer_export_paper_contract_4_7_gpu"
    if "scaled_ppo" in text:
        return "legacy_scaled_ppo_diagnostic"
    if "fk_repaired_robot_order" in text:
        return "fk_repaired_robot_order_diagnostic"
    if "hub_singleleg" in text:
        return "hub_singleleg_short_or_failed"
    return "other_tracking_eval"


def extract_candidate_from_best(path: Path, data: dict[str, Any]) -> Candidate | None:
    best = data.get("best_checkpoint")
    if not isinstance(best, dict):
        return None
    cand = Candidate(source_json=str(path), category=infer_category(path, data))
    cand.checkpoint = best.get("checkpoint")
    cand.iteration = best.get("iteration") or best.get("loaded_iteration") or parse_iteration_from_path(Path(cand.checkpoint or str(path)))
    cand.status = best.get("status") or data.get("status")
    cand.claim_level = data.get("claim_level")
    cand.official_beyondmimic_checkpoint = bool(data.get("official_beyondmimic_checkpoint", False))
    cand.paper_level_tracking_eval = bool(data.get("paper_level_tracking_eval", False))
    cand.metrics_json = best.get("metrics_json")
    cand.timeseries_csv = best.get("timeseries_csv")
    cand.eval_steps = best.get("eval_steps")
    cand.num_envs = best.get("num_envs")
    cand.total_env_steps = best.get("total_env_steps")
    cand.motion_count = best.get("motion_count")
    cand.total_motion_frames = best.get("total_motion_frames")
    cand.done_count_total = best.get("done_count_total")
    cand.timeout_count_total = best.get("timeout_count_total")
    cand.local_non_timeout_done_rate = best.get("local_non_timeout_done_rate")
    cand.reward_mean = best.get("reward_mean")
    for key in METRIC_KEYS:
        cand.metrics[key] = best.get(key)
    fill_from_metrics(cand, read_eval_metrics(cand.metrics_json))
    return cand


def extract_candidate_from_eval(path: Path, data: dict[str, Any]) -> Candidate | None:
    metrics_path = nested_get(data, ("outputs", "metrics_json"))
    if metrics_path is None and path.name.endswith("eval_metrics.json"):
        metrics_path = str(path)
    metrics = read_eval_metrics(metrics_path) if metrics_path != str(path) else data
    checkpoint = data.get("checkpoint_sweep_checkpoint") or nested_get(data, ("inputs", "checkpoint")) or metrics.get("checkpoint")
    if not checkpoint and not metrics:
        return None
    cand = Candidate(source_json=str(path), category=infer_category(path, data))
    cand.checkpoint = checkpoint
    cand.iteration = data.get("checkpoint_sweep_iteration") or metrics.get("loaded_iteration") or parse_iteration_from_path(path)
    cand.status = data.get("status") or metrics.get("status")
    cand.claim_level = nested_get(data, ("interpretation", "claim_level")) or data.get("claim_level")
    cand.official_beyondmimic_checkpoint = bool(nested_get(data, ("interpretation", "official_beyondmimic_checkpoint")) or data.get("official_beyondmimic_checkpoint", False))
    cand.paper_level_tracking_eval = bool(nested_get(data, ("interpretation", "paper_level_tracking_eval_complete")) or metrics.get("paper_level_tracking_eval", False))
    cand.metrics_json = str(metrics_path) if metrics_path else None
    cand.timeseries_csv = nested_get(data, ("outputs", "timeseries_csv"))
    cand.eval_steps = nested_get(data, ("config", "eval_steps")) or metrics.get("eval_steps")
    cand.num_envs = nested_get(data, ("config", "num_envs")) or metrics.get("num_envs")
    cand.total_env_steps = nested_get(data, ("config", "total_env_steps")) or metrics.get("total_env_steps")
    fill_from_metrics(cand, metrics)
    return cand


def fill_from_metrics(cand: Candidate, metrics: dict[str, Any]) -> None:
    if not metrics:
        return
    cand.eval_steps = cand.eval_steps or metrics.get("eval_steps")
    cand.num_envs = cand.num_envs or metrics.get("num_envs")
    cand.total_env_steps = cand.total_env_steps or metrics.get("total_env_steps")
    cand.done_count_total = cand.done_count_total if cand.done_count_total is not None else metrics.get("done_count_total")
    cand.timeout_count_total = cand.timeout_count_total if cand.timeout_count_total is not None else metrics.get("timeout_count_total")
    cand.checkpoint = cand.checkpoint or metrics.get("checkpoint")
    cand.iteration = cand.iteration or metrics.get("loaded_iteration") or parse_iteration_from_path(Path(cand.checkpoint or ""))
    reward = metrics.get("reward")
    if cand.reward_mean is None:
        cand.reward_mean = stat_mean(reward)
    cand.action_abs_mean = stat_mean(metrics.get("action_abs_mean_over_steps"))
    cand.action_abs_max = stat_mean(metrics.get("action_abs_max_over_steps"))
    for key, value in extract_motion_metrics(metrics).items():
        if cand.metrics.get(key) is None:
            cand.metrics[key] = value
    if cand.local_non_timeout_done_rate is None and cand.done_count_total is not None and cand.eval_steps and cand.num_envs:
        timeouts = cand.timeout_count_total or 0
        cand.local_non_timeout_done_rate = max(0.0, float(cand.done_count_total - timeouts) / float(cand.eval_steps * cand.num_envs))


def finalize_candidate(cand: Candidate) -> Candidate:
    p_ckpt = resolve_path(cand.checkpoint)
    p_metrics = resolve_path(cand.metrics_json)
    p_ts = resolve_path(cand.timeseries_csv)
    if p_ckpt is None or not p_ckpt.exists():
        cand.blockers.append("checkpoint_missing_or_unresolved")
    if p_metrics is None or not p_metrics.exists():
        cand.blockers.append("metrics_json_missing")
    if p_ts is None or not p_ts.exists():
        cand.notes.append("timeseries_csv_missing; continuity cannot be independently checked")
    else:
        checks = compute_timeseries_checks(cand.timeseries_csv)
        if checks.get("exists") and cand.done_count_total is not None and checks.get("done_sum_from_timeseries") != cand.done_count_total:
            cand.notes.append("done_count_total differs from timeseries sum")
    if cand.eval_steps is None or cand.eval_steps < READINESS_THRESHOLDS["min_eval_steps"]:
        cand.blockers.append("eval_steps_too_small_or_missing")
    if cand.num_envs is None or cand.num_envs < READINESS_THRESHOLDS["min_num_envs"]:
        cand.blockers.append("num_envs_too_small_or_missing")
    if cand.local_non_timeout_done_rate is None:
        cand.blockers.append("done_rate_missing")
    else:
        if cand.local_non_timeout_done_rate > READINESS_THRESHOLDS["candidate_done_rate"]:
            cand.blockers.append("done_rate_above_candidate_threshold")
        if cand.local_non_timeout_done_rate > READINESS_THRESHOLDS["downstream_ready_done_rate"]:
            cand.blockers.append("done_rate_above_downstream_threshold")
    for key, threshold in [
        ("error_body_pos_mean", READINESS_THRESHOLDS["max_body_pos_error_mean"]),
        ("error_joint_pos_mean", READINESS_THRESHOLDS["max_joint_pos_error_mean"]),
        ("error_anchor_pos_mean", READINESS_THRESHOLDS["max_anchor_pos_error_mean"]),
    ]:
        value = cand.metrics.get(key)
        if value is None:
            cand.blockers.append(f"{key}_missing")
        elif value > threshold:
            cand.blockers.append(f"{key}_above_threshold")
    if cand.action_abs_mean is not None and cand.action_abs_mean > READINESS_THRESHOLDS["max_action_abs_mean"]:
        cand.blockers.append("action_abs_mean_above_threshold")
    if not cand.official_beyondmimic_checkpoint:
        cand.notes.append("not_official_beyondmimic_checkpoint")
    if not cand.paper_level_tracking_eval:
        cand.notes.append("not_paper_level_tracking_eval")

    required_fields = [
        cand.local_non_timeout_done_rate,
        cand.metrics.get("error_body_pos_mean"),
        cand.metrics.get("error_joint_pos_mean"),
        cand.metrics.get("error_anchor_pos_mean"),
        cand.eval_steps,
        cand.num_envs,
    ]
    cand.evidence_quality = "sufficient_for_local_selection" if all(v is not None for v in required_fields) else "insufficient_for_local_selection"
    cand.candidate_teacher_usable = (
        cand.evidence_quality == "sufficient_for_local_selection"
        and cand.local_non_timeout_done_rate is not None
        and cand.local_non_timeout_done_rate <= READINESS_THRESHOLDS["candidate_done_rate"]
        and cand.metrics.get("error_body_pos_mean") is not None
        and cand.metrics["error_body_pos_mean"] <= READINESS_THRESHOLDS["max_body_pos_error_mean"]
        and cand.metrics.get("error_joint_pos_mean") is not None
        and cand.metrics["error_joint_pos_mean"] <= READINESS_THRESHOLDS["max_joint_pos_error_mean"]
    )
    cand.downstream_teacher_ready = (
        cand.candidate_teacher_usable
        and cand.local_non_timeout_done_rate is not None
        and cand.local_non_timeout_done_rate <= READINESS_THRESHOLDS["downstream_ready_done_rate"]
    )
    if cand.downstream_teacher_ready:
        cand.decision = "ready_for_continuous_teacher_rollout_collection"
    elif cand.candidate_teacher_usable:
        cand.decision = "usable_only_for_diagnostic_teacher_rollout_not_final_vae_diffusion"
    elif cand.evidence_quality == "sufficient_for_local_selection":
        cand.decision = "not_ready_for_vae_diffusion_due_to_quality_gate"
    else:
        cand.decision = "insufficient_evidence_do_not_use"

    done = cand.local_non_timeout_done_rate if cand.local_non_timeout_done_rate is not None else 1.0
    body = cand.metrics.get("error_body_pos_mean") if cand.metrics.get("error_body_pos_mean") is not None else 2.0
    joint = cand.metrics.get("error_joint_pos_mean") if cand.metrics.get("error_joint_pos_mean") is not None else 3.0
    anchor = cand.metrics.get("error_anchor_pos_mean") if cand.metrics.get("error_anchor_pos_mean") is not None else 1.0
    action = cand.action_abs_mean if cand.action_abs_mean is not None else 2.0
    cand.score = 10.0 * done + 1.5 * body + 0.5 * joint + anchor + 0.1 * action
    if not cand.candidate_teacher_usable:
        cand.score += 10.0
    return cand


def discover_candidates() -> list[Candidate]:
    candidate_paths: set[Path] = set()
    roots = [
        ROOT / "res" / "tracking",
        ROOT / "res" / "runs",
    ]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            name = path.name.lower()
            text = str(path).lower()
            if "teacher_checkpoint_quality_selector" in text:
                continue
            if (
                "best_teacher" in name
                or "checkpoint_eval" in name
                or name == "eval_metrics.json"
                or name.endswith("_eval_metrics.json")
            ) and any(token in text for token in ["tracking", "ppo", "stage1", "paper_contract", "fk_repaired", "scaled"]):
                candidate_paths.add(path)
    candidates: list[Candidate] = []
    seen: set[tuple[str | None, int | None, str, str, str]] = set()
    for path in sorted(candidate_paths):
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        cand = extract_candidate_from_best(path, data) if "best_teacher" in path.name.lower() else None
        if cand is None:
            cand = extract_candidate_from_eval(path, data)
        if cand is None:
            continue
        cand = finalize_candidate(cand)
        key = (
            cand.checkpoint,
            cand.iteration,
            cand.category,
            f"{cand.local_non_timeout_done_rate:.8f}" if cand.local_non_timeout_done_rate is not None else "none",
            f"{cand.metrics.get('error_body_pos_mean'):.8f}" if cand.metrics.get("error_body_pos_mean") is not None else "none",
            f"{cand.metrics.get('error_joint_pos_mean'):.8f}" if cand.metrics.get("error_joint_pos_mean") is not None else "none",
        )
        if key in seen:
            continue
        seen.add(key)
        candidates.append(cand)
    candidates.sort(key=lambda c: (c.score, c.category, c.iteration if c.iteration is not None else -1))
    return candidates


def summarize(candidates: list[Candidate]) -> dict[str, Any]:
    ready = [c for c in candidates if c.downstream_teacher_ready]
    usable = [c for c in candidates if c.candidate_teacher_usable]
    best = candidates[0] if candidates else None
    status = "blocked_no_downstream_ready_teacher_checkpoint"
    if ready:
        status = "ok_downstream_ready_teacher_checkpoint_available"
    elif usable:
        status = "diagnostic_only_candidate_available_but_not_downstream_ready"
    counts: dict[str, int] = {}
    decisions: dict[str, int] = {}
    category_best: dict[str, dict[str, Any]] = {}
    for c in candidates:
        counts[c.category] = counts.get(c.category, 0) + 1
        decisions[c.decision] = decisions.get(c.decision, 0) + 1
        if c.category not in category_best or c.score < category_best[c.category]["_score_for_sort"]:
            row = c.as_row()
            row["_score_for_sort"] = c.score
            category_best[c.category] = row
    for row in category_best.values():
        row.pop("_score_for_sort", None)
    return {
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "thresholds": READINESS_THRESHOLDS,
        "candidate_count": len(candidates),
        "category_counts": counts,
        "decision_counts": decisions,
        "category_best_candidates": category_best,
        "best_ranked_candidate": best.as_row() if best else None,
        "ready_candidate_count": len(ready),
        "usable_candidate_count": len(usable),
        "claim_boundary": (
            "This selector ranks local tracking checkpoints for downstream data collection only. "
            "It does not certify official BeyondMimic paper-level tracking, DAgger, VAE, diffusion, "
            "Fig.5/Fig.6, Isaac rendered rollout, or real-robot results."
        ),
        "required_next_action": (
            "Do not train VAE/diffusion or generate success videos from a teacher unless a checkpoint "
            "passes the downstream readiness gate and a continuous no-jump teacher rollout is collected."
        ),
        "candidates": [c.as_row() for c in candidates],
    }


def write_outputs(summary: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    rows = summary["candidates"]
    fieldnames = [
        "category",
        "iteration",
        "decision",
        "score",
        "local_non_timeout_done_rate",
        "reward_mean",
        "error_anchor_pos_mean",
        "error_body_pos_mean",
        "error_joint_pos_mean",
        "error_joint_vel_mean",
        "action_abs_mean",
        "eval_steps",
        "num_envs",
        "motion_count",
        "total_motion_frames",
        "candidate_teacher_usable",
        "downstream_teacher_ready",
        "checkpoint",
        "metrics_json",
        "source_json",
        "blockers",
        "notes",
    ]
    with TSV_OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    best = summary.get("best_ranked_candidate") or {}
    lines = [
        "# Stage 1 Teacher Checkpoint Quality Selector",
        "",
        f"- Status: `{summary['status']}`",
        f"- Candidates scanned: `{summary['candidate_count']}`",
        f"- Usable diagnostic candidates: `{summary['usable_candidate_count']}`",
        f"- Downstream-ready candidates: `{summary['ready_candidate_count']}`",
        "",
        "## Thresholds",
        "",
    ]
    for key, value in READINESS_THRESHOLDS.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Best Ranked Candidate",
            "",
            f"- Category: `{best.get('category')}`",
            f"- Iteration: `{best.get('iteration')}`",
            f"- Decision: `{best.get('decision')}`",
            f"- Done rate: `{best.get('local_non_timeout_done_rate')}`",
            f"- Body position error mean: `{best.get('error_body_pos_mean')}`",
            f"- Joint position error mean: `{best.get('error_joint_pos_mean')}`",
            f"- Checkpoint: `{best.get('checkpoint')}`",
            f"- Blockers: `{best.get('blockers')}`",
            "",
            "## Top 10 Candidates",
            "",
        ]
    )
    for row in rows[:10]:
        lines.append(
            "- "
            f"`{row.get('category')}` iter `{row.get('iteration')}`: "
            f"decision `{row.get('decision')}`, "
            f"done `{row.get('local_non_timeout_done_rate')}`, "
            f"body `{row.get('error_body_pos_mean')}`, "
            f"joint `{row.get('error_joint_pos_mean')}`"
        )
    lines.extend(["", "## Best Candidate Per Family", ""])
    for category, row in sorted(summary.get("category_best_candidates", {}).items()):
        lines.append(
            "- "
            f"`{category}` iter `{row.get('iteration')}`: "
            f"decision `{row.get('decision')}`, "
            f"done `{row.get('local_non_timeout_done_rate')}`, "
            f"body `{row.get('error_body_pos_mean')}`, "
            f"joint `{row.get('error_joint_pos_mean')}`"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            summary["claim_boundary"],
            "",
            "当前不得声称完整复现 BeyondMimic；该选择器只证明本地 teacher checkpoint 是否适合继续采集 VAE/diffusion 数据。",
            "",
        ]
    )
    MD_OUT.write_text("\n".join(lines))


def main() -> int:
    candidates = discover_candidates()
    summary = summarize(candidates)
    write_outputs(summary)
    print(json.dumps({k: summary[k] for k in ["status", "candidate_count", "usable_candidate_count", "ready_candidate_count"]}, ensure_ascii=False, indent=2))
    print(f"Wrote {JSON_OUT}")
    return 0 if candidates else 2


if __name__ == "__main__":
    raise SystemExit(main())
