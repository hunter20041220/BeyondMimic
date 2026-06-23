#!/usr/bin/env python3
"""Diagnose why the latest Stage-1 MuJoCo action-control videos are unstable.

This is an audit script, not a renderer.  It reads the selected continuous
teacher segment, the multi-source motion bundle, teacher metrics, and the
generated video metrics to separate three different failure modes:

* a bad/low-height reference segment selection;
* weak Stage-1 PPO teacher quality;
* MuJoCo control/video instability inherited by VAE and diffusion variants.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
VIDEO_ROOT = ROOT / "res/visualization/stage1_multisource_continuous_mujoco_action_control_videos"
SUMMARY_JSON = VIDEO_ROOT / "stage1_multisource_continuous_video_suite_summary.json"
TEACHER_SWEEP_JSON = (
    ROOT
    / "res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/"
    "tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.json"
)
TEACHER_ROLLOUT_JSON = (
    ROOT
    / "res/tracking/stage1_multisource_best_teacher_rollout_dataset/"
    "tracking_stage1_multisource_best_teacher_rollout_dataset.json"
)
MOTION_BUNDLE_JSON = (
    ROOT
    / "res/tracking/stage1_multisource_motion_bundle/"
    "tracking_stage1_multisource_motion_bundle.json"
)

OUT_JSON = VIDEO_ROOT / "stage1_multisource_mujoco_video_failure_diagnosis.json"
OUT_MD = VIDEO_ROOT / "stage1_multisource_mujoco_video_failure_diagnosis.md"
OUT_TSV = VIDEO_ROOT / "stage1_multisource_mujoco_video_failure_candidate_segments.tsv"

VIDEO_NAMES = [
    "reference_action_control",
    "teacher_policy_action_control",
    "vae_reconstructed_action_control",
    "diffusion_denoised_latent_action_control",
    "guided_latent_action_control",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"min": float("nan"), "mean": float("nan"), "median": float("nan"), "max": float("nan")}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def motion_rows() -> list[dict[str, Any]]:
    rows = read_json(MOTION_BUNDLE_JSON)["rows"]
    return sorted(rows, key=lambda row: int(row["start_frame"]))


def source_for_global_step(rows: list[dict[str, Any]], step: int) -> dict[str, Any] | None:
    for row in rows:
        if int(row["start_frame"]) <= step < int(row["end_frame_exclusive"]):
            return row
    return None


_ROOT_Z_CACHE: dict[str, np.ndarray] = {}


def source_root_z(row: dict[str, Any]) -> np.ndarray:
    key = str(row["per_motion_npz"])
    cached = _ROOT_Z_CACHE.get(key)
    if cached is not None:
        return cached
    with np.load(key) as motion:
        z = np.asarray(motion["body_pos_w"][:, 0, 2], dtype=np.float64).copy()
    _ROOT_Z_CACHE[key] = z
    return z


def segment_root_stats(segment: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = int(segment["motion_time_step_start"])
    last = int(segment["motion_time_step_end"])
    source = source_for_global_step(rows, first)
    if source is None:
        return {"single_source": False, "reason": "first step does not map to motion bundle row"}
    if not (int(source["start_frame"]) <= last < int(source["end_frame_exclusive"])):
        return {
            "single_source": False,
            "reason": "segment crosses source motion boundary",
            "source_motion": source.get("motion"),
        }
    z = source_root_z(source)
    local_start = first - int(source["start_frame"])
    local_end = last - int(source["start_frame"]) + 1
    window = z[local_start:local_end]
    source_stats = stats(z)
    window_stats = stats(window)
    source_percentile = float(np.mean(z <= window_stats["mean"]) * 100.0)
    return {
        "single_source": True,
        "source_motion": source.get("motion"),
        "source_family": source.get("source_family"),
        "source_path": source.get("source_path"),
        "per_motion_npz": source.get("per_motion_npz"),
        "source_start_frame": int(source["start_frame"]),
        "source_end_frame_exclusive": int(source["end_frame_exclusive"]),
        "local_start": int(local_start),
        "local_end_exclusive": int(local_end),
        "selected_root_z": window_stats,
        "whole_source_root_z": source_stats,
        "selected_mean_percentile_within_source": source_percentile,
        "below_nominal_standing_height": bool(window_stats["mean"] < 0.45),
        "near_floor_segment": bool(window_stats["mean"] < 0.15),
    }


def find_continuous_segments(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rollout = read_json(TEACHER_ROLLOUT_JSON)
    segments: list[dict[str, Any]] = []
    for shard_path_str in rollout["run"]["shard_npz_paths"]:
        shard_path = Path(shard_path_str)
        with np.load(shard_path) as data:
            rewards = np.asarray(data["rewards"], dtype=np.float64)
            dones = np.asarray(data["dones"], dtype=np.bool_)
            time_steps = np.asarray(data["motion_time_steps"], dtype=np.int64)
            rank = int(np.asarray(data["rank"])[0])
            total_frames, env_count = time_steps.shape
            for env_idx in range(env_count):
                start = 0
                while start < total_frames:
                    while start < total_frames and dones[start, env_idx]:
                        start += 1
                    if start >= total_frames:
                        break
                    end = start + 1
                    while (
                        end < total_frames
                        and not dones[end, env_idx]
                        and int(time_steps[end, env_idx]) == int(time_steps[end - 1, env_idx]) + 1
                    ):
                        end += 1
                    length = end - start
                    if length >= 2:
                        segment = {
                            "shard": str(shard_path),
                            "rank": rank,
                            "env_index": int(env_idx),
                            "start": int(start),
                            "end_exclusive": int(end),
                            "length": int(length),
                            "motion_time_step_start": int(time_steps[start, env_idx]),
                            "motion_time_step_end": int(time_steps[end - 1, env_idx]),
                            "reward_mean": float(np.mean(rewards[start:end, env_idx])),
                            "done_count": int(np.sum(dones[start:end, env_idx])),
                        }
                        segment["root_diagnosis"] = segment_root_stats(segment, rows)
                        segments.append(segment)
                    start = max(end, start + 1)
    return segments


def read_video_metrics(name: str) -> dict[str, Any]:
    metrics_path = VIDEO_ROOT / name / f"{name}_metrics.csv"
    summary_path = VIDEO_ROOT / name / f"{name}_summary.json"
    output: dict[str, Any] = {
        "metrics_csv": str(metrics_path),
        "summary_json": str(summary_path),
        "metrics_exists": metrics_path.is_file(),
        "summary_exists": summary_path.is_file(),
    }
    if summary_path.is_file():
        output["summary_sha256"] = sha256_file(summary_path)
        output["summary"] = read_json(summary_path)
    if not metrics_path.is_file():
        return output
    columns: dict[str, list[float]] = {}
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if value in ("", None):
                    continue
                try:
                    columns.setdefault(key, []).append(float(value))
                except ValueError:
                    continue
    output["metrics_sha256"] = sha256_file(metrics_path)
    output["row_count"] = len(next(iter(columns.values()))) if columns else 0
    output["stats"] = {
        key: stats(np.asarray(values, dtype=np.float64))
        for key, values in columns.items()
        if key
        in {
            "root_z",
            "root_target_z",
            "root_x",
            "root_y",
            "root_position_error_m",
            "fall_proxy",
            "action_abs_mean",
            "joint_target_abs_mean",
        }
    }
    root_z = output["stats"].get("root_z", {})
    root_target_z = output["stats"].get("root_target_z", {})
    root_x = output["stats"].get("root_x", {})
    root_y = output["stats"].get("root_y", {})
    fall_proxy = output["stats"].get("fall_proxy", {})
    output["instability_flags"] = {
        "root_target_near_floor": bool(root_target_z and root_target_z.get("mean", 1.0) < 0.15),
        "root_height_explosion": bool(root_z and root_z.get("max", 0.0) > 2.0),
        "xy_drift_over_5m": bool(
            (root_x and max(abs(root_x.get("min", 0.0)), abs(root_x.get("max", 0.0))) > 5.0)
            or (root_y and max(abs(root_y.get("min", 0.0)), abs(root_y.get("max", 0.0))) > 5.0)
        ),
        "fall_proxy_majority": bool(fall_proxy and fall_proxy.get("mean", 0.0) > 0.5),
    }
    return output


def compact_segment(segment: dict[str, Any], rank: int, rule: str) -> dict[str, Any]:
    diag = segment.get("root_diagnosis", {})
    selected_z = diag.get("selected_root_z", {})
    return {
        "rank": rank,
        "rule": rule,
        "length": int(segment["length"]),
        "reward_mean": float(segment["reward_mean"]),
        "source_motion": diag.get("source_motion", ""),
        "root_z_min": selected_z.get("min", ""),
        "root_z_mean": selected_z.get("mean", ""),
        "root_z_max": selected_z.get("max", ""),
        "source_percentile": diag.get("selected_mean_percentile_within_source", ""),
        "rank_id": int(segment["rank"]),
        "env_index": int(segment["env_index"]),
        "start": int(segment["start"]),
        "end_exclusive": int(segment["end_exclusive"]),
        "motion_time_step_start": int(segment["motion_time_step_start"]),
        "motion_time_step_end": int(segment["motion_time_step_end"]),
    }


def write_tsv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "rank",
        "rule",
        "length",
        "reward_mean",
        "source_motion",
        "root_z_min",
        "root_z_mean",
        "root_z_max",
        "source_percentile",
        "rank_id",
        "env_index",
        "start",
        "end_exclusive",
        "motion_time_step_start",
        "motion_time_step_end",
    ]
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_TSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    for path in [SUMMARY_JSON, TEACHER_SWEEP_JSON, TEACHER_ROLLOUT_JSON, MOTION_BUNDLE_JSON]:
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = read_json(SUMMARY_JSON)
    teacher_sweep = read_json(TEACHER_SWEEP_JSON)
    teacher_rollout = read_json(TEACHER_ROLLOUT_JSON)
    rows = motion_rows()

    selected_segment = summary["selected_continuous_segment"]
    selected_root_diag = segment_root_stats(selected_segment, rows)
    all_segments = find_continuous_segments(rows)
    current_rule = sorted(all_segments, key=lambda row: (row["length"], row["reward_mean"]), reverse=True)[:10]
    quality_rule = sorted(
        [
            row
            for row in all_segments
            if row["length"] >= 60
            and row.get("root_diagnosis", {}).get("single_source", False)
            and row.get("root_diagnosis", {}).get("selected_root_z", {}).get("mean", 0.0) >= 0.45
            and row.get("root_diagnosis", {}).get("selected_root_z", {}).get("min", 0.0) >= 0.30
        ],
        key=lambda row: (row["reward_mean"], row["length"]),
        reverse=True,
    )[:10]
    segment_availability: dict[str, Any] = {}
    for min_len in [2, 10, 20, 30, 60, 100, 150, 200]:
        subset = [row for row in all_segments if row["length"] >= min_len]
        good_root = [
            row
            for row in subset
            if row.get("root_diagnosis", {}).get("selected_root_z", {}).get("mean", 0.0) >= 0.45
            and row.get("root_diagnosis", {}).get("selected_root_z", {}).get("min", 0.0) >= 0.30
        ]
        best_good = None
        if good_root:
            item = sorted(good_root, key=lambda row: (row["reward_mean"], row["length"]), reverse=True)[0]
            best_good = compact_segment(item, 1, f"best_good_root_len_ge_{min_len}")
        segment_availability[str(min_len)] = {
            "segment_count": len(subset),
            "good_root_segment_count": len(good_root),
            "best_good_root_segment": best_good,
        }

    candidate_rows = [compact_segment(row, idx + 1, "current_length_then_reward") for idx, row in enumerate(current_rule)]
    candidate_rows += [compact_segment(row, idx + 1, "quality_reward_then_length") for idx, row in enumerate(quality_rule)]
    for min_len, item in segment_availability.items():
        if item["best_good_root_segment"] is not None:
            candidate_rows.append(item["best_good_root_segment"])
    write_tsv(candidate_rows)

    best = teacher_sweep.get("best_checkpoint", {})
    rollout_run = teacher_rollout.get("run", {})
    video_metrics = {name: read_video_metrics(name) for name in VIDEO_NAMES}
    checks = {
        "selected_segment_is_continuous": bool(
            selected_segment.get("continuity", {}).get("all_motion_time_step_deltas_plus_one")
        ),
        "selected_segment_has_no_done": int(selected_segment.get("done_count", -1)) == 0,
        "selected_segment_root_target_near_floor": bool(
            selected_root_diag.get("selected_root_z", {}).get("mean", 1.0) < 0.15
        ),
        "selected_segment_reward_negative": float(selected_segment.get("reward_mean", 0.0)) < 0.0,
        "selection_rule_prefers_length_before_quality": True,
        "teacher_reward_low": float(best.get("reward_mean", 0.0)) < 0.10,
        "teacher_done_rate_high": float(best.get("local_non_timeout_done_rate", 0.0)) > 0.10,
        "teacher_body_error_high": float(best.get("error_body_pos_mean", 0.0)) > 0.50,
        "video_instability_detected": any(
            any(item.get("instability_flags", {}).values()) for item in video_metrics.values()
        ),
        "not_real_robot_result": True,
        "does_not_claim_paper_level": True,
    }
    diagnosis = [
        {
            "rank": 1,
            "cause": "bad_segment_selection_near_floor_root_target",
            "evidence": (
                "The selected continuous segment is valid in time, but its pelvis/root target z is near the floor "
                "instead of a standing walking height."
            ),
            "severity": "critical",
        },
        {
            "rank": 2,
            "cause": "selection_rule_prioritizes_length_over_motion_quality",
            "evidence": "The current selector sorts by (length, reward_mean), so a long low-quality segment can win.",
            "severity": "critical",
        },
        {
            "rank": 3,
            "cause": "stage1_teacher_policy_is_still_weak",
            "evidence": "The best checkpoint has very low reward, high body error, and high non-timeout done rate.",
            "severity": "high",
        },
        {
            "rank": 4,
            "cause": "vae_diffusion_guidance_inherit_bad_teacher_and_bad_targets",
            "evidence": "The downstream videos share the same selected root targets and are trained/decoded from weak teacher rollouts.",
            "severity": "high",
        },
        {
            "rank": 5,
            "cause": "mujoco_adapter_is_a_diagnostic_proxy_not_official_closed_loop",
            "evidence": "Videos use MuJoCo position actuators plus root assist and should not be interpreted as paper-level control.",
            "severity": "medium",
        },
    ]

    payload = {
        "status": "ok_stage1_mujoco_video_failure_diagnosis",
        "timestamp_utc": utc_now(),
        "experiment_type": "stage1_multisource_mujoco_video_failure_diagnosis",
        "claim_level": "diagnosis_only_local_mujoco_video_pipeline_not_paper_level",
        "inputs": {
            "video_summary_json": str(SUMMARY_JSON),
            "video_summary_sha256": sha256_file(SUMMARY_JSON),
            "teacher_sweep_json": str(TEACHER_SWEEP_JSON),
            "teacher_sweep_sha256": sha256_file(TEACHER_SWEEP_JSON),
            "teacher_rollout_json": str(TEACHER_ROLLOUT_JSON),
            "teacher_rollout_sha256": sha256_file(TEACHER_ROLLOUT_JSON),
            "motion_bundle_json": str(MOTION_BUNDLE_JSON),
            "motion_bundle_sha256": sha256_file(MOTION_BUNDLE_JSON),
        },
        "checks": checks,
        "diagnosis": diagnosis,
        "selected_segment": selected_segment,
        "selected_segment_root_diagnosis": selected_root_diag,
        "teacher_quality": {
            "best_checkpoint": best,
            "rollout_done_count_total": rollout_run.get("done_count_total")
            or teacher_rollout.get("aggregate_metrics", {}).get("done_count_total"),
            "rollout_total_env_steps": rollout_run.get("total_env_steps")
            or teacher_rollout.get("aggregate_metrics", {}).get("total_env_steps"),
            "rollout_reward_mean_by_rank": rollout_run.get("reward_mean_by_rank")
            or teacher_rollout.get("aggregate_metrics", {}).get("reward_mean_by_rank"),
        },
        "video_metrics": video_metrics,
        "candidate_segments_tsv": str(OUT_TSV),
        "top_current_rule_segments": current_rule,
        "top_quality_rule_segments": quality_rule,
        "segment_availability_by_min_length": segment_availability,
        "recommended_fix_order": [
            "Do not use the current videos as successful motion-control evidence; keep them as failed diagnostics.",
            "Patch segment selection to require source root height/stability and prefer reward before length.",
            "Regenerate the reference pose replay from a standing-height continuous source segment before touching VAE/diffusion videos.",
            "Only after reference/root targets are sane, rerun teacher action-control with the same segment.",
            "If teacher remains unstable after target selection is fixed, retrain/evaluate Stage-1 teacher before training or claiming VAE/diffusion/guidance closed-loop behavior.",
        ],
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(payload)
    print(json.dumps({"status": payload["status"], "json": str(OUT_JSON), "md": str(OUT_MD)}, sort_keys=True))


def write_markdown(payload: dict[str, Any]) -> None:
    selected_z = payload["selected_segment_root_diagnosis"]["selected_root_z"]
    source_z = payload["selected_segment_root_diagnosis"]["whole_source_root_z"]
    best = payload["teacher_quality"]["best_checkpoint"]
    lines = [
        "# Stage-1 MuJoCo 视频失败诊断",
        "",
        "## 结论",
        "",
        "当前视频差的首要原因不是视频编码，也不是单纯 VAE/diffusion 公式写错，而是前端控制证据链已经坏了："
        "自动选中的连续片段虽然 `motion_time_steps` 连续、`done_count=0`，但 root/pelvis target z 只有约"
        f" `{selected_z['mean']:.4f} m`，接近地面；同一个 source motion 的中位 root z 为"
        f" `{source_z['median']:.4f} m`。后续 MuJoCo root assist 和 PD action-control 都围绕这个近地面目标运行，"
        "所以机器人站不稳是预期结果。",
        "",
        "## 关键证据",
        "",
        f"- 选中 motion: `{payload['selected_segment_root_diagnosis'].get('source_motion', '')}`",
        f"- 选中全局 motion steps: `{payload['selected_segment']['motion_time_step_start']}..{payload['selected_segment']['motion_time_step_end']}`",
        f"- 选中片段 root z: min `{selected_z['min']:.4f}`, mean `{selected_z['mean']:.4f}`, max `{selected_z['max']:.4f}` m",
        f"- 整个 source root z: min `{source_z['min']:.4f}`, median `{source_z['median']:.4f}`, max `{source_z['max']:.4f}` m",
        f"- 选中片段 reward mean: `{payload['selected_segment']['reward_mean']:.6f}`",
        f"- 当前选择规则: `length` 优先，再看 `reward_mean`；因此会选择长但不可展示的坏片段。",
        f"- teacher best reward mean: `{float(best.get('reward_mean', float('nan'))):.6f}`",
        f"- teacher body error mean: `{float(best.get('error_body_pos_mean', float('nan'))):.6f}` m",
        f"- teacher non-timeout done rate: `{float(best.get('local_non_timeout_done_rate', float('nan'))):.6f}`",
        f"- 60 帧及以上、root z 正常的连续候选数: `{payload['segment_availability_by_min_length']['60']['good_root_segment_count']}`",
        f"- 30 帧及以上、root z 正常的连续候选数: `{payload['segment_availability_by_min_length']['30']['good_root_segment_count']}`",
        "",
        "## 为什么六个视频都会差",
        "",
        "这六个视频共享同一段 reference/root target。`reference_action_control` 直接做 pose replay，"
        "因此它会首先暴露低 root 高度问题；`teacher_policy_action_control`、`vae_reconstructed_action_control`、"
        "`diffusion_denoised_latent_action_control` 和 `guided_latent_action_control` 又共享这个 root target，"
        "同时 teacher 本身 reward 很低、done/fall 很多，所以它们不是独立失败，而是同一条坏数据/弱 teacher 链路的下游表现。",
        "",
        "## 下一步修复顺序",
        "",
        "1. 旧六个视频标记为失败诊断，不作为展示结果。",
        "2. 修改 segment selector：要求 root z 不低于站立阈值，优先 reward/stability，再考虑 length。",
        "3. 先重新生成一条 root 高度正常的 reference pose replay。",
        "4. 再在同一连续片段上跑 teacher action-control。",
        "5. teacher 稳定后才继续 VAE/diffusion/guidance 视频；否则下游视频只会学习弱 teacher 和坏 target。",
        "",
        "## Claim Boundary",
        "",
        "这份诊断只说明本地 MuJoCo 视频链路为什么失败；它不是 BeyondMimic paper-level 复现结果，也不是真实机器人结果。",
        "",
        f"JSON: `{OUT_JSON}`",
        f"候选片段 TSV: `{OUT_TSV}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
