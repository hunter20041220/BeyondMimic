#!/usr/bin/env python3
"""Audit clean MuJoCo baselines for LAFAN1 ``jumps1_subject1``.

The audit records that the stable window is currently the report-safe
reference-action baseline, while the high-dynamic window is retained as a
diagnostic source replay/control stress case.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/audits/lafan1_jumps1_subject1_mujoco_clean"
JSON_OUT = OUT / "lafan1_jumps1_subject1_mujoco_clean_audit.json"
TSV_OUT = OUT / "lafan1_jumps1_subject1_mujoco_clean_audit.tsv"
MD_OUT = OUT / "lafan1_jumps1_subject1_mujoco_clean_audit.md"
BASE = ROOT / "res/visualization/lafan1_jumps1_subject1_mujoco"
WINDOWS = ["high_dynamic_52s_67s", "stable_dynamic_164s_179s"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def file_info(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"path": "", "exists": False, "size_bytes": 0, "sha256": ""}
    p = Path(path)
    exists = p.is_file() and p.stat().st_size > 0
    return {
        "path": str(p),
        "exists": exists,
        "size_bytes": p.stat().st_size if p.exists() else 0,
        "sha256": sha256(p) if exists else "",
    }


def row_for_window(window: str) -> dict[str, Any]:
    summary_path = BASE / window / "lafan1_jumps1_subject1_mujoco_summary.json"
    summary = read_json(summary_path)
    replay = summary.get("cases", {}).get("original_csv_reference_replay", {})
    control = summary.get("cases", {}).get("reference_action_control", {})
    checks = summary.get("checks", {})
    control_metrics = control.get("metrics", {})
    control_checks = control.get("checks", {})
    issues: list[str] = []
    if summary.get("status") != "ok":
        issues.append(f"summary_status={summary.get('status')!r}")
    if not checks.get("reference_replay_ok", False):
        issues.append("reference_replay_not_ok")
    if not checks.get("reference_action_control_ok", False):
        issues.append("reference_action_control_not_ok")
    if not control_checks.get("uses_mj_step", False):
        issues.append("control_does_not_use_mj_step")
    if not control_checks.get("does_not_write_qpos_each_frame", False):
        issues.append("control_writes_qpos_each_frame")
    fall_count = int(control_metrics.get("fall_proxy_count", 999))
    if window == "stable_dynamic_164s_179s" and fall_count != 0:
        issues.append("stable_window_fall_proxy_nonzero")
    if window == "high_dynamic_52s_67s" and fall_count == 0:
        issues.append("high_dynamic_expected_diagnostic_failure_not_recorded")
    return {
        "window": window,
        "summary_path": str(summary_path),
        "summary_exists": summary_path.is_file(),
        "passed": not issues,
        "recommended_for_report": window == "stable_dynamic_164s_179s" and not issues,
        "diagnostic_only": window == "high_dynamic_52s_67s",
        "status": summary.get("status", "missing"),
        "source_start_time_s": summary.get("window", {}).get("source_start_time_s", ""),
        "source_end_time_s": summary.get("window", {}).get("source_end_time_s", ""),
        "frames_rendered": replay.get("frames_rendered", ""),
        "root_z_range": replay.get("source_summary", {}).get("root_z_range", ""),
        "max_joint_step": replay.get("source_summary", {}).get("max_joint_step", ""),
        "control_fall_proxy_count": fall_count,
        "control_joint_error_abs_mean": control_metrics.get("joint_error_abs_mean", ""),
        "control_root_height_min": control_metrics.get("root_height_min", ""),
        "control_root_height_max": control_metrics.get("root_height_max", ""),
        "replay_mp4": file_info(replay.get("outputs", {}).get("mp4")),
        "control_mp4": file_info(control.get("outputs", {}).get("mp4")),
        "issues": issues,
        "claim_level": summary.get("claim_level", ""),
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "window",
        "passed",
        "recommended_for_report",
        "diagnostic_only",
        "status",
        "source_start_time_s",
        "source_end_time_s",
        "frames_rendered",
        "root_z_range",
        "max_joint_step",
        "control_fall_proxy_count",
        "control_joint_error_abs_mean",
        "control_root_height_min",
        "control_root_height_max",
        "replay_mp4_path",
        "control_mp4_path",
        "issues",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "window": row["window"],
                    "passed": row["passed"],
                    "recommended_for_report": row["recommended_for_report"],
                    "diagnostic_only": row["diagnostic_only"],
                    "status": row["status"],
                    "source_start_time_s": row["source_start_time_s"],
                    "source_end_time_s": row["source_end_time_s"],
                    "frames_rendered": row["frames_rendered"],
                    "root_z_range": row["root_z_range"],
                    "max_joint_step": row["max_joint_step"],
                    "control_fall_proxy_count": row["control_fall_proxy_count"],
                    "control_joint_error_abs_mean": row["control_joint_error_abs_mean"],
                    "control_root_height_min": row["control_root_height_min"],
                    "control_root_height_max": row["control_root_height_max"],
                    "replay_mp4_path": row["replay_mp4"]["path"],
                    "control_mp4_path": row["control_mp4"]["path"],
                    "issues": ";".join(row["issues"]),
                }
            )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [row_for_window(window) for window in WINDOWS]
    recommended = [row for row in rows if row["recommended_for_report"]]
    status = "ok_jumps1_stable_reference_action_control_ready" if len(recommended) == 1 else "failed"
    payload = {
        "status": status,
        "timestamp_utc": utc_now(),
        "experiment_type": "lafan1_jumps1_subject1_mujoco_clean_audit",
        "claim_level": "Local MuJoCo LAFAN1 jumps1 source/reference baseline audit; not learned BeyondMimic control.",
        "rows": rows,
        "row_count": len(rows),
        "recommended_window": recommended[0]["window"] if recommended else "",
        "checks": {
            "stable_window_reference_replay_ok": any(
                row["window"] == "stable_dynamic_164s_179s" and row["passed"] for row in rows
            ),
            "stable_window_reference_action_control_fall_proxy_zero": any(
                row["window"] == "stable_dynamic_164s_179s" and row["control_fall_proxy_count"] == 0 for row in rows
            ),
            "high_dynamic_retained_as_diagnostic": any(
                row["window"] == "high_dynamic_52s_67s" and row["diagnostic_only"] for row in rows
            ),
            "does_not_claim_teacher_vae_diffusion_guidance": True,
            "does_not_claim_paper_level": True,
            "does_not_claim_real_robot": True,
            "does_not_claim_goal_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The jumps1 source/reference baseline is ready, but teacher/RL, VAE, diffusion, guidance, "
                "and formula/parameter gates still block downstream training and success claims."
            ),
        },
        "outputs": {"json": str(JSON_OUT), "tsv": str(TSV_OUT), "md": str(MD_OUT)},
    }
    write_json(JSON_OUT, payload)
    write_tsv(TSV_OUT, rows)
    MD_OUT.write_text(
        "\n".join(
            [
                "# LAFAN1 jumps1_subject1 MuJoCo Clean Audit",
                "",
                f"Status: `{status}`",
                "",
                f"Recommended report window: `{payload['recommended_window']}`",
                "",
                "High-dynamic window is retained as a diagnostic stress case because its reference_action_control has fall/instability.",
                "Stable window is the current clean local MuJoCo source/reference baseline.",
                "",
                "Claim boundary: not teacher/RL, not VAE, not diffusion, not guidance, not paper-level, not real robot.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "recommended_window": payload["recommended_window"], "json": str(JSON_OUT)}))
    if status != "ok_jumps1_stable_reference_action_control_ready":
        raise SystemExit(1)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    main()
