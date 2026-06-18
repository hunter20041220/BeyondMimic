#!/usr/bin/env python3
"""Preflight official whole_body_tracking replay entry points without running replay."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/official_replay_preflight"
OFFICIAL = ROOT / "reproduction/third_party/official/whole_body_tracking/scripts"
LOCAL = ROOT / "reproduction/generated/whole_body_tracking_local"
MOTION_CSV = ROOT / "download/official/LAFAN1_Retargeting_Dataset/g1/walk1_subject1.csv"
LIVE_GATE = ROOT / "res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json"
PROJECT_EGL_ICD = ROOT / "res/setup/vulkan_runtime_probe/nvidia_egl_icd.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parser_args(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                names.append(arg.value)
                break
    return names


def csv_shape(path: Path) -> tuple[int, int, bool]:
    row_count = 0
    width = None
    finite = True
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if width is None:
                width = len(row)
            elif width != len(row):
                finite = False
            for value in row:
                try:
                    float(value)
                except ValueError:
                    finite = False
            row_count += 1
    return row_count, width or 0, finite


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    official_csv_to_npz = OFFICIAL / "csv_to_npz.py"
    official_replay_npz = OFFICIAL / "replay_npz.py"
    local_csv_to_npz = LOCAL / "csv_to_npz_local.py"
    local_replay_npz = LOCAL / "replay_npz_local.py"
    for label, path in [
        ("official_csv_to_npz", official_csv_to_npz),
        ("official_replay_npz", official_replay_npz),
        ("local_csv_to_npz", local_csv_to_npz),
        ("local_replay_npz", local_replay_npz),
    ]:
        rows.append(
            {
                "label": label,
                "path": str(path),
                "exists": path.is_file(),
                "sha256": sha256(path) if path.is_file() else "",
                "args": parser_args(path) if path.is_file() else [],
            }
        )

    row_count, column_count, finite = csv_shape(MOTION_CSV)
    live = load_json(LIVE_GATE)
    output_npz = ROOT / "res/tracking/official_replay_preflight/walk1_subject1_official_preflight_motion.npz"
    csv_to_npz_command = [
        str(ROOT / "envs/bm_tracking/bin/python"),
        str(official_csv_to_npz),
        "--input_file",
        str(MOTION_CSV),
        "--input_fps",
        "30",
        "--frame_range",
        "1",
        "180",
        "--output_name",
        str(output_npz),
        "--output_fps",
        "50",
        "--headless",
        "--device",
        "cuda:6",
        "--enable_cameras",
        "False",
    ]
    local_replay_command = [
        str(ROOT / "envs/bm_tracking/bin/python"),
        str(local_replay_npz),
        "--motion_file",
        str(output_npz),
        "--headless",
        "--device",
        "cuda:6",
    ]
    checks = {
        "official_csv_to_npz_exists": official_csv_to_npz.is_file(),
        "official_replay_npz_exists": official_replay_npz.is_file(),
        "local_replay_patch_exists": local_replay_npz.is_file(),
        "motion_csv_exists": MOTION_CSV.is_file(),
        "motion_csv_has_36_columns": column_count == 36,
        "motion_csv_finite": finite,
        "motion_csv_has_enough_frames": row_count >= 180,
        "live_gate_allows_replay_preflight": live.get("status") in {"ok", "ok_with_runtime_warning"},
        "project_egl_icd_exists": PROJECT_EGL_ICD.is_file(),
        "does_not_execute_csv_to_npz_or_replay": True,
        "does_not_start_training": True,
        "does_not_claim_tracking_reproduction_complete": True,
    }
    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "tracking_official_replay_preflight",
        "scope": (
            "Static/pre-execution preflight for official whole_body_tracking csv_to_npz.py and replay_npz.py. "
            "This does not execute motion conversion, rendered replay, task smoke, PPO, or policy evaluation."
        ),
        "selected_motion_csv": {
            "path": str(MOTION_CSV),
            "sha256": sha256(MOTION_CSV),
            "row_count": row_count,
            "column_count": column_count,
            "finite": finite,
        },
        "entry_points": rows,
        "commands_planned_not_run": {
            "official_csv_to_npz": csv_to_npz_command,
            "local_replay_npz_after_official_conversion": local_replay_command,
        },
        "runtime_requirements": {
            "python": str(ROOT / "envs/bm_tracking/bin/python"),
            "cuda_visible_devices": "5,6",
            "preferred_device": "cuda:6",
            "vk_icd_filenames": str(PROJECT_EGL_ICD),
            "live_gate_status": live.get("status"),
            "live_gate_warning_retained": live.get("checks", {}).get("cuda_p2p_iommu_runtime_warning_retained"),
        },
        "checks": checks,
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The official replay path is now ready for a bounded execution attempt, but no official motion.npz "
                "conversion, rendered replay, tracking task smoke, PPO training, or paper metric evaluation was run."
            ),
            "next_action": (
                "Run official csv_to_npz.py on the selected short G1 LAFAN1 clip with the live-gate environment, "
                "validate the produced motion.npz, then run local/offline replay only if conversion succeeds."
            ),
        },
        "outputs": {
            "json": str(OUT / "tracking_official_replay_preflight.json"),
            "tsv": str(OUT / "tracking_official_replay_preflight.tsv"),
        },
    }
    (OUT / "tracking_official_replay_preflight.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (OUT / "tracking_official_replay_preflight.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=["label", "path", "exists", "sha256", "args"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": summary["status"], "json": summary["outputs"]["json"]}, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
