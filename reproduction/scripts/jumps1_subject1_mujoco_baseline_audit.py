#!/usr/bin/env python3
"""Audit the current MuJoCo jumps1_subject1 baseline videos.

This audit intentionally separates three claim levels:

1. original 36-column LAFAN1 CSV kinematic replay;
2. FK-repaired motion NPZ kinematic replay;
3. MuJoCo mj_step PD reference-action control.

It does not claim teacher/RL, VAE, diffusion, guided control, IsaacLab
rendering, or real-robot reproduction.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/audits/jumps1_subject1_mujoco_baseline"


CASES = [
    {
        "case_id": "original_csv_reference_replay_osmesa",
        "category": "kinematic_reference_replay",
        "summary": ROOT
        / "official_mp4/res/lafan1_jumps1_subject1_original_csv_osmesa/"
        "lafan1_jumps1_subject1_original_csv_osmesa_summary.json",
        "expected_frames_min": 300,
        "required_checks": [
            "mp4_exists",
            "keyframe_exists",
            "metrics_exists",
            "input_is_36_columns",
            "joint_dim_29",
            "does_not_claim_policy_rollout",
            "does_not_claim_real_robot",
            "does_not_claim_paper_level_fig5_fig6",
        ],
        "claim_level": (
            "Official released 36-column G1 LAFAN1 reference rendered in MuJoCo via "
            "frame-by-frame mj_forward; not policy control."
        ),
    },
    {
        "case_id": "fk_repaired_npz_reference_replay_osmesa",
        "category": "kinematic_reference_replay",
        "summary": ROOT
        / "mujoco_mp4/res/reference_replay/lafan1_jumps1_subject1_fk_repaired_npz_osmesa/"
        "reference_replay_summary.json",
        "expected_frames_min": 299,
        "required_checks": [
            "mp4_exists",
            "keyframe_exists",
            "metrics_csv_exists",
            "joint_dim_29",
            "does_not_claim_policy_rollout",
            "does_not_claim_isaaclab",
            "does_not_claim_real_robot",
        ],
        "claim_level": (
            "FK-repaired local motion NPZ rendered in MuJoCo via frame-by-frame "
            "mj_forward; not policy control."
        ),
    },
    {
        "case_id": "reference_action_control_osmesa",
        "category": "pd_reference_action_control",
        "summary": ROOT
        / "mujoco_mp4/res/jumps1_subject1_control_videos_osmesa/reference_action_control/"
        "reference_action_control_summary.json",
        "expected_frames_min": 299,
        "required_checks": [
            "mp4_exists",
            "keyframe_exists",
            "metrics_csv_exists",
            "uses_mj_step",
            "does_not_write_qpos_each_frame",
            "uses_29_position_actuators",
            "uses_root_assist_controller",
            "native_mujoco_ppo_obs_adapter",
            "does_not_claim_native_mujoco_policy_controller",
            "does_not_claim_isaaclab_render",
            "does_not_claim_real_robot",
        ],
        "required_false_checks": ["native_mujoco_ppo_obs_adapter"],
        "claim_level": (
            "MuJoCo mj_step PD tracking of FK-repaired reference joint targets with "
            "root assist; not a learned teacher/VAE/diffusion controller."
        ),
    },
]


EGL_ABORT_LOG = ROOT / "logs/mujoco/jumps1_subject1/original_csv_replay_300f_egl.log"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ffprobe(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"ok": False, "error": "missing"}
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,duration,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    except FileNotFoundError:
        return {"ok": False, "error": "ffprobe_not_found"}
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.strip()}
    payload = json.loads(proc.stdout)
    streams = payload.get("streams", [])
    if not streams:
        return {"ok": False, "error": "no_video_stream"}
    stream = streams[0]
    return {
        "ok": True,
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "duration": float(stream.get("duration", 0.0)),
        "nb_frames": int(stream.get("nb_frames", 0)),
        "r_frame_rate": stream.get("r_frame_rate", ""),
    }


def path_info(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": "", "exists": False, "size_bytes": 0, "sha256": ""}
    exists = path.is_file() and path.stat().st_size > 0
    return {
        "path": str(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256(path) if exists else "",
    }


def output_path(summary: dict[str, Any], *names: str) -> Path | None:
    outputs = summary.get("outputs", {})
    for name in names:
        value = outputs.get(name)
        if value:
            return Path(value)
    return None


def audit_case(case: dict[str, Any]) -> dict[str, Any]:
    summary_path = Path(case["summary"])
    row: dict[str, Any] = {
        "case_id": case["case_id"],
        "category": case["category"],
        "summary_path": str(summary_path),
        "summary_exists": summary_path.is_file() and summary_path.stat().st_size > 0,
        "claim_level": case["claim_level"],
        "passed": False,
        "issues": [],
    }
    if not row["summary_exists"]:
        row["issues"].append("summary_missing")
        return row

    summary = load_json(summary_path)
    checks = summary.get("checks", {})
    required_true = list(case.get("required_checks", []))
    required_false = set(case.get("required_false_checks", []))
    missing_or_bad = []
    for name in required_true:
        expected = False if name in required_false else True
        if checks.get(name) is not expected:
            missing_or_bad.append(f"{name}!={expected}")

    frames_rendered = int(summary.get("frames_rendered", 0))
    mp4 = output_path(summary, "mp4")
    keyframe = output_path(summary, "keyframe_png")
    metrics = output_path(summary, "metrics_csv")
    probe = ffprobe(mp4) if mp4 else {"ok": False, "error": "mp4_path_missing"}
    mp4_frames = int(probe.get("nb_frames", 0)) if probe.get("ok") else 0
    metrics_lines = 0
    if metrics and metrics.is_file():
        with metrics.open("r", encoding="utf-8") as f:
            metrics_lines = sum(1 for _ in f)

    metrics_summary = summary.get("metrics", {})
    row.update(
        {
            "status": summary.get("status"),
            "backend": summary.get("backend"),
            "frames_rendered": frames_rendered,
            "expected_frames_min": int(case["expected_frames_min"]),
            "mp4_probe": probe,
            "mp4_info": path_info(mp4),
            "keyframe_info": path_info(keyframe),
            "metrics_info": path_info(metrics),
            "metrics_lines": metrics_lines,
            "checks": checks,
            "metrics_summary": metrics_summary,
            "limitations": summary.get("limitations", []),
            "target_metadata": summary.get("target_metadata", {}),
            "simulation": summary.get("simulation", {}),
        }
    )
    if summary.get("status") != "ok":
        row["issues"].append(f"status={summary.get('status')!r}")
    if missing_or_bad:
        row["issues"].extend(missing_or_bad)
    if frames_rendered < int(case["expected_frames_min"]):
        row["issues"].append(f"frames_rendered_lt_{case['expected_frames_min']}")
    if not probe.get("ok"):
        row["issues"].append(f"ffprobe_failed={probe.get('error')}")
    elif mp4_frames < int(case["expected_frames_min"]):
        row["issues"].append(f"mp4_frames_lt_{case['expected_frames_min']}")
    if metrics_lines < int(case["expected_frames_min"]):
        row["issues"].append("metrics_lines_too_few")

    if case["case_id"] == "reference_action_control_osmesa":
        if int(metrics_summary.get("fall_proxy_count", 1)) != 0:
            row["issues"].append("fall_proxy_count_nonzero")
        if float(metrics_summary.get("joint_error_abs_mean", 999.0)) > 0.15:
            row["issues"].append("joint_error_abs_mean_gt_0.15")
        if bool(summary.get("simulation", {}).get("root_assist_enabled")) is not True:
            row["issues"].append("root_assist_not_recorded")

    row["passed"] = not row["issues"]
    return row


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "category",
        "passed",
        "status",
        "backend",
        "frames_rendered",
        "mp4_frames",
        "duration_s",
        "metrics_lines",
        "claim_level",
        "issues",
        "summary_path",
        "mp4_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            probe = row.get("mp4_probe", {})
            writer.writerow(
                {
                    "case_id": row["case_id"],
                    "category": row["category"],
                    "passed": row["passed"],
                    "status": row.get("status", ""),
                    "backend": row.get("backend", ""),
                    "frames_rendered": row.get("frames_rendered", 0),
                    "mp4_frames": probe.get("nb_frames", 0),
                    "duration_s": probe.get("duration", 0.0),
                    "metrics_lines": row.get("metrics_lines", 0),
                    "claim_level": row.get("claim_level", ""),
                    "issues": ";".join(row.get("issues", [])),
                    "summary_path": row.get("summary_path", ""),
                    "mp4_path": row.get("mp4_info", {}).get("path", ""),
                }
            )


def write_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# jumps1_subject1 MuJoCo Baseline Audit",
        "",
        f"- Status: `{summary['status']}`",
        f"- Passed cases: `{summary['pass_count']}/{summary['row_count']}`",
        f"- EGL 300-frame attempt status: `{summary['egl_300_frame_attempt']['status']}`",
        "",
        "## Case Summary",
        "",
        "| Case | Category | Frames | MP4 Frames | Passed | Claim Boundary |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["rows"]:
        probe = row.get("mp4_probe", {})
        lines.append(
            "| "
            + " | ".join(
                [
                    row["case_id"],
                    row["category"],
                    str(row.get("frames_rendered", 0)),
                    str(probe.get("nb_frames", 0)),
                    str(row["passed"]),
                    row["claim_level"],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The original CSV and FK-repaired NPZ videos are reference replays; they prove that the source motion can be rendered on the G1 mesh in MuJoCo, not that a policy controls the robot.",
            "- `reference_action_control` uses MuJoCo `mj_step` and 29 position actuators, but it also uses a pelvis root-assist stabilizer. It is a PD control baseline, not a teacher/RL, VAE, diffusion, or guidance result.",
            "- The EGL 300-frame attempt is retained as a rendering-backend failure. OSMesa is the current stable backend for this local H20 report-video path.",
            "- This audit does not claim paper-level BeyondMimic Fig. 5/Fig. 6 reproduction and does not claim real-robot deployment.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [audit_case(case) for case in CASES]
    egl_log_info = path_info(EGL_ABORT_LOG)
    egl_text = EGL_ABORT_LOG.read_text(encoding="utf-8", errors="replace") if EGL_ABORT_LOG.is_file() else ""
    if EGL_ABORT_LOG.is_file() and "Aborted" in egl_text:
        egl_status = "failed_retained"
    elif EGL_ABORT_LOG.exists() and EGL_ABORT_LOG.stat().st_size == 0:
        egl_status = "empty_log_after_abort_observed_in_command_output"
    else:
        egl_status = "not_observed"
    pass_count = sum(1 for row in rows if row["passed"])
    summary = {
        "status": "ok_jumps1_subject1_mujoco_baseline_audit" if pass_count == len(rows) else "failed",
        "experiment_type": "jumps1_subject1_mujoco_baseline_audit",
        "scope": (
            "Audit current jumps1_subject1 MuJoCo reference replay and PD reference-action "
            "control videos before any teacher/VAE/diffusion/guidance training claims."
        ),
        "row_count": len(rows),
        "pass_count": pass_count,
        "fail_count": len(rows) - pass_count,
        "rows": rows,
        "egl_300_frame_attempt": {
            "status": egl_status,
            "log": egl_log_info,
            "interpretation": (
                "The EGL 300-frame replay attempt aborted on this H20 host; OSMesa completed "
                "the same 300-frame CSV replay and 299-frame NPZ/control videos."
            ),
        },
        "checks": {
            "all_cases_passed": pass_count == len(rows),
            "original_csv_osmesa_300_frames_ok": rows[0]["passed"],
            "fk_repaired_npz_osmesa_299_frames_ok": rows[1]["passed"],
            "reference_action_control_osmesa_299_frames_ok": rows[2]["passed"],
            "reference_action_control_uses_mj_step": bool(rows[2].get("checks", {}).get("uses_mj_step")),
            "reference_action_control_does_not_write_qpos": bool(
                rows[2].get("checks", {}).get("does_not_write_qpos_each_frame")
            ),
            "reference_action_control_root_assist_declared": bool(
                rows[2].get("simulation", {}).get("root_assist_enabled")
            ),
            "reference_action_control_not_native_ppo": rows[2].get("checks", {}).get("native_mujoco_ppo_obs_adapter")
            is False,
            "egl_abort_or_empty_log_recorded": egl_status in {
                "failed_retained",
                "empty_log_after_abort_observed_in_command_output",
            },
            "does_not_claim_teacher_policy": True,
            "does_not_claim_vae_diffusion_guidance": True,
            "does_not_claim_isaaclab_render": True,
            "does_not_claim_real_robot": True,
            "does_not_claim_goal_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The baseline proves jumps1_subject1 can be rendered and PD-tracked locally in MuJoCo "
                "with root assist, but learned teacher/RL, VAE, diffusion, guidance, official IsaacLab "
                "rendered MP4, and real-robot results remain unproven."
            ),
        },
        "outputs": {
            "json": str(OUT / "jumps1_subject1_mujoco_baseline_audit.json"),
            "tsv": str(OUT / "jumps1_subject1_mujoco_baseline_audit.tsv"),
            "markdown": str(OUT / "jumps1_subject1_mujoco_baseline_audit.md"),
        },
    }
    (OUT / "jumps1_subject1_mujoco_baseline_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_tsv(OUT / "jumps1_subject1_mujoco_baseline_audit.tsv", rows)
    write_md(OUT / "jumps1_subject1_mujoco_baseline_audit.md", summary)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "rows": summary["row_count"],
                "pass_count": summary["pass_count"],
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok_jumps1_subject1_mujoco_baseline_audit":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
