#!/usr/bin/env python3
"""Summarize MuJoCo PD control videos and verify their claim boundaries."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from mujoco_common import PKG, sha256, utc_now, write_json, write_tsv


CONTROL_ROOT = PKG / "res/control_videos"


def ffprobe(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size",
        "-show_entries",
        "stream=nb_frames,width,height,avg_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return json.loads(proc.stdout)


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(CONTROL_ROOT.glob("*/*_summary.json")):
        data = load_summary(summary_path)
        outputs = data.get("outputs", {})
        mp4 = Path(outputs.get("mp4", ""))
        probe = ffprobe(mp4) if mp4.is_file() else {"streams": [], "format": {}}
        stream = (probe.get("streams") or [{}])[0]
        fmt = probe.get("format") or {}
        checks = data.get("checks", {})
        simulation = data.get("simulation", {})
        metrics = data.get("metrics", {})
        row = {
            "name": summary_path.parent.name,
            "status": data.get("status", ""),
            "experiment_type": data.get("experiment_type", ""),
            "claim_level": data.get("claim_level", ""),
            "mp4": str(mp4),
            "mp4_exists": mp4.is_file(),
            "mp4_size": int(fmt.get("size", mp4.stat().st_size if mp4.exists() else 0)),
            "sha256": sha256(mp4) if mp4.is_file() else "",
            "width": stream.get("width", ""),
            "height": stream.get("height", ""),
            "avg_frame_rate": stream.get("avg_frame_rate", ""),
            "nb_frames": int(stream.get("nb_frames", data.get("frames_rendered", 0)) or 0),
            "duration_seconds": float(fmt.get("duration", data.get("duration_seconds", 0.0)) or 0.0),
            "uses_mj_step": bool(simulation.get("uses_mj_step") or checks.get("uses_mj_step")),
            "writes_qpos_each_frame": bool(simulation.get("writes_qpos_each_frame", False)),
            "uses_root_assist_controller": bool(checks.get("uses_root_assist_controller", False)),
            "native_mujoco_ppo_obs_adapter": bool(checks.get("native_mujoco_ppo_obs_adapter", False)),
            "fall_proxy_count": metrics.get("fall_proxy_count", ""),
            "root_xy_abs_max": metrics.get("root_xy_abs_max", ""),
            "root_position_error_mean_m": metrics.get("root_position_error_mean_m", ""),
            "joint_error_abs_mean": metrics.get("joint_error_abs_mean", ""),
            "summary_json": str(summary_path),
        }
        rows.append(row)

    expected = {
        "reference_control",
        "ppo_policy_control",
        "vae_base_control",
        "denoised_latent_control",
        "guided_latent_control",
        "guided_vs_unguided_control",
    }
    names = {row["name"] for row in rows}
    single_rows = [row for row in rows if row["name"] != "guided_vs_unguided_control"]
    payload = {
        "status": "ok" if expected.issubset(names) and all(row["mp4_exists"] for row in rows) else "partial_or_failed",
        "timestamp_utc": utc_now(),
        "experiment_type": "mujoco_pd_control_video_summary",
        "claim_level": "MuJoCo local PD closed-loop tracking-control visualization; not native MuJoCo PPO/VAE/guidance, not IsaacLab, not real robot",
        "video_count": len(rows),
        "expected_video_names": sorted(expected),
        "missing_expected": sorted(expected - names),
        "checks": {
            "all_expected_present": expected.issubset(names),
            "all_mp4_exist": all(row["mp4_exists"] for row in rows),
            "all_15s": all(abs(float(row["duration_seconds"]) - 15.0) < 1e-6 for row in rows),
            "all_450_frames": all(int(row["nb_frames"]) == 450 for row in rows),
            "all_single_videos_960x540": all(int(row["width"]) == 960 and int(row["height"]) == 540 for row in single_rows),
            "side_by_side_1920x540": any(
                row["name"] == "guided_vs_unguided_control"
                and int(row["width"]) == 1920
                and int(row["height"]) == 540
                for row in rows
            ),
            "single_videos_use_mj_step": all(bool(row["uses_mj_step"]) for row in single_rows),
            "single_videos_do_not_write_qpos_each_frame": all(not bool(row["writes_qpos_each_frame"]) for row in single_rows),
            "single_videos_use_root_assist": all(bool(row["uses_root_assist_controller"]) for row in single_rows),
            "no_native_mujoco_ppo_adapter_claim": all(not bool(row["native_mujoco_ppo_obs_adapter"]) for row in rows),
            "single_videos_no_fall_proxy": all(int(row["fall_proxy_count"]) == 0 for row in single_rows),
        },
        "interpretation": {
            "what_changed_vs_rollout_videos": (
                "The older mujoco_mp4/res/rollout_videos assets are trace-to-mesh visualizations. "
                "The new mujoco_mp4/res/control_videos assets are generated through MuJoCo mj_step using "
                "29 position actuators and an explicitly marked root-assist stabilizer."
            ),
            "why_not_native_policy_rollout": (
                "The available video trace files do not contain full 29-D action vectors for PPO/VAE/guided variants, "
                "and a faithful MuJoCo reconstruction of the IsaacLab 160-D policy observation manager is still not complete."
            ),
            "guided_vs_unguided": (
                "Unguided means the VAE-base target trace without test-time task guidance. Guided means the latent/action "
                "target trace after the local task guidance objective. The side-by-side video compares MuJoCo PD tracking "
                "of those two target traces, not native MuJoCo latent guidance."
            ),
        },
        "rows": rows,
    }
    out_json = CONTROL_ROOT / "mujoco_control_video_summary.json"
    out_tsv = CONTROL_ROOT / "mujoco_control_video_summary.tsv"
    write_json(out_json, payload)
    fields = [
        "name",
        "status",
        "experiment_type",
        "claim_level",
        "mp4",
        "mp4_exists",
        "mp4_size",
        "sha256",
        "width",
        "height",
        "avg_frame_rate",
        "nb_frames",
        "duration_seconds",
        "uses_mj_step",
        "writes_qpos_each_frame",
        "uses_root_assist_controller",
        "native_mujoco_ppo_obs_adapter",
        "fall_proxy_count",
        "root_xy_abs_max",
        "root_position_error_mean_m",
        "joint_error_abs_mean",
        "summary_json",
    ]
    write_tsv(out_tsv, rows, fields)
    print(json.dumps({"status": payload["status"], "video_count": len(rows), "summary": str(out_json)}))
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
