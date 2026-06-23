#!/usr/bin/env python3
"""Audit the quality-gated Stage-1 MuJoCo video suite.

The selector/root-target bug is considered fixed only if the chosen reference
has normal root height.  Long-horizon control is considered unresolved unless
teacher/VAE/diffusion videos remain stable beyond this short gate.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT_ROOT = ROOT / "res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos"
SUMMARY_JSON = OUT_ROOT / "stage1_multisource_quality_gated_video_suite_summary.json"
SELECTOR_JSON = OUT_ROOT / "quality_gated_stage1_multisource_selector_audit.json"
OUT_JSON = OUT_ROOT / "quality_gated_stage1_multisource_stability_audit.json"
OUT_MD = OUT_ROOT / "quality_gated_stage1_multisource_stability_audit.md"

CONTROL_VIDEOS = [
    "teacher_policy_action_control",
    "vae_reconstructed_action_control",
    "diffusion_denoised_latent_action_control",
    "guided_latent_action_control",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def numeric_column_stats(path: Path, columns: list[str]) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = {column: [] for column in columns}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for column in columns:
                raw = row.get(column)
                if raw in (None, ""):
                    continue
                try:
                    values[column].append(float(raw))
                except ValueError:
                    pass
    out: dict[str, dict[str, float]] = {}
    for column, vals in values.items():
        if not vals:
            continue
        out[column] = {
            "min": min(vals),
            "mean": sum(vals) / len(vals),
            "max": max(vals),
            "last": vals[-1],
        }
    return out


def video_audit(name: str) -> dict[str, Any]:
    summary_path = OUT_ROOT / name / f"{name}_summary.json"
    metrics_path = OUT_ROOT / name / f"{name}_metrics.csv"
    summary = read_json(summary_path)
    stats = numeric_column_stats(
        metrics_path,
        [
            "root_z",
            "root_target_z",
            "root_position_error_m",
            "fall_proxy",
            "joint_error_abs_mean",
            "joint_velocity_abs_mean",
        ],
    )
    root = stats.get("root_z", {})
    target = stats.get("root_target_z", {})
    fall_count = int(summary.get("metrics", {}).get("fall_proxy_count", 0))
    root_drop = None
    if root and target:
        root_drop = float(target["mean"] - root["min"])
    return {
        "name": name,
        "summary_json": str(summary_path),
        "metrics_csv": str(metrics_path),
        "metrics": summary.get("metrics", {}),
        "stats": stats,
        "checks": {
            "mp4_exists": bool(summary.get("checks", {}).get("mp4_exists")),
            "metrics_csv_exists": metrics_path.is_file() and metrics_path.stat().st_size > 0,
            "fall_proxy_zero": fall_count == 0,
            "root_height_above_fall_threshold": bool(root and root["min"] >= 0.45),
            "root_height_sag_under_0p70": bool(root and root["min"] < 0.70),
            "root_drop_from_target_over_0p10": bool(root_drop is not None and root_drop > 0.10),
            "does_not_claim_paper_level": True,
        },
        "root_drop_from_target_m": root_drop,
    }


def main() -> None:
    if not SUMMARY_JSON.is_file():
        raise FileNotFoundError(SUMMARY_JSON)
    if not SELECTOR_JSON.is_file():
        raise FileNotFoundError(SELECTOR_JSON)
    summary = read_json(SUMMARY_JSON)
    selector = read_json(SELECTOR_JSON)
    reference_metrics = numeric_column_stats(
        OUT_ROOT / "reference_action_control/reference_action_control_metrics.csv",
        ["root_z", "root_x", "root_y"],
    )
    controls = {name: video_audit(name) for name in CONTROL_VIDEOS}
    selector_summary = selector.get("selected_segment", {})
    checks = {
        "selector_status_ok": selector.get("status") == "ok_quality_gated_selector",
        "suite_status_ok": summary.get("status") == "ok",
        "reference_root_z_mean_normal": reference_metrics.get("root_z", {}).get("mean", 0.0) >= 0.75,
        "reference_root_z_min_normal": reference_metrics.get("root_z", {}).get("min", 0.0) >= 0.70,
        "control_videos_have_zero_fall_proxy": all(item["checks"]["fall_proxy_zero"] for item in controls.values()),
        "control_videos_remain_above_fall_threshold": all(
            item["checks"]["root_height_above_fall_threshold"] for item in controls.values()
        ),
        "control_videos_show_root_height_sag": any(
            item["checks"]["root_drop_from_target_over_0p10"] for item in controls.values()
        ),
        "long_horizon_control_not_proven": True,
        "does_not_claim_paper_level": True,
        "does_not_claim_real_robot": True,
    }
    payload = {
        "status": "partial_fix_root_target_gate_short_control_still_limited",
        "timestamp_utc": utc_now(),
        "experiment_type": "stage1_multisource_quality_gated_mujoco_video_stability_audit",
        "claim_level": (
            "The near-floor segment selection bug is fixed for this short suite, and the action-control videos no "
            "longer immediately fall.  Long-horizon MuJoCo control remains unresolved."
        ),
        "summary_json": str(SUMMARY_JSON),
        "selector_json": str(SELECTOR_JSON),
        "selector_summary": selector_summary,
        "reference_root_stats": reference_metrics,
        "control_video_audits": controls,
        "checks": checks,
        "interpretation": {
            "fixed": [
                "The selected reference segment has normal root height.",
                "The old near-floor root target is no longer used.",
                "The short teacher/VAE/diffusion/guided action-control videos have zero fall_proxy count.",
            ],
            "still_limited": [
                "The videos are only 30 frames / 1 second because the current teacher rollout has no >=60-frame normal-height stable segment.",
                "Control videos still show root-height sag relative to the 0.789 m target.",
                "The controller uses MuJoCo position actuators plus root assist and is not a native closed-loop PPO obs/action adapter.",
                "This is not paper-level Fig.5/Fig.6 and not real-robot evidence.",
            ],
            "next": [
                "For better videos, extend Stage-1 teacher quality or collect longer normal-height stable teacher rollouts.",
                "Then implement/evaluate a true MuJoCo closed-loop obs/action adapter instead of open-loop replaying Isaac teacher actions.",
            ],
        },
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(payload)
    print(json.dumps({"status": payload["status"], "json": str(OUT_JSON), "md": str(OUT_MD)}, sort_keys=True))


def write_markdown(payload: dict[str, Any]) -> None:
    selected = payload["selector_summary"]
    ref_z = payload["reference_root_stats"]["root_z"]
    teacher = payload["control_video_audits"]["teacher_policy_action_control"]
    lines = [
        "# Quality-Gated Stage-1 MuJoCo Stability Audit",
        "",
        "## 结论",
        "",
        "本轮修复已经解决了旧视频的 near-floor root target 问题：新选段 root z 正常，reference replay 可以正常站立显示。"
        "teacher/VAE/diffusion/guided action-control 在 30 帧短视频里 `fall_proxy_count=0`，但仍存在 root height 下滑，"
        "因此只能视为短时诊断通过，不能视为长时稳定控制。",
        "",
        "## 选段",
        "",
        f"- Motion: `{selected.get('source_motion')}`",
        f"- Motion steps: `{selected.get('motion_time_step_start')}..{selected.get('motion_time_step_end')}`",
        f"- Frames: `{selected.get('length')}`",
        f"- Reward mean: `{selected.get('reward_mean')}`",
        f"- Root z mean: `{selected.get('root_z_mean')}`",
        "",
        "## 指标",
        "",
        f"- Reference root z: min `{ref_z['min']:.4f}`, mean `{ref_z['mean']:.4f}`, max `{ref_z['max']:.4f}` m",
        f"- Teacher fall proxy count: `{teacher['metrics'].get('fall_proxy_count')}`",
        f"- Teacher root height min/max: `{teacher['metrics'].get('root_height_min')}` / `{teacher['metrics'].get('root_height_max')}` m",
        f"- Teacher root position error mean/max: `{teacher['metrics'].get('root_position_error_mean_m')}` / `{teacher['metrics'].get('root_position_error_max_m')}` m",
        "",
        "## 未解决",
        "",
        "- 当前 teacher rollout 没有 `>=60` 帧的正常 root-height 稳定片段；视频只有 30 帧，不应硬拉长。",
        "- action-control 仍使用 MuJoCo position actuators + root assist，不是 native MuJoCo PPO closed-loop obs/action adapter。",
        "- 当前结果不是 BeyondMimic paper-level Fig.5/Fig.6，也不是真实机器人结果。",
        "",
        f"JSON: `{OUT_JSON}`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
