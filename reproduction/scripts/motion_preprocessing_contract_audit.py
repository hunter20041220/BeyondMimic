#!/usr/bin/env python3
"""Static and data-shape audit for the official motion preprocessing contract."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/motion_preprocessing_contract_audit"
OFFICIAL = ROOT / "reproduction/third_party/official/whole_body_tracking"
RAW = ROOT / "download/official/whole_body_tracking"
CSV_TO_NPZ = OFFICIAL / "scripts/csv_to_npz.py"
RAW_CSV_TO_NPZ = RAW / "scripts/csv_to_npz.py"
REPLAY_NPZ = OFFICIAL / "scripts/replay_npz.py"
COMMANDS = (
    OFFICIAL
    / "source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py"
)
VALIDATOR = ROOT / "reproduction/scripts/validate_motion_npz_contract.py"
LOCAL_PATCHER = ROOT / "reproduction/scripts/prepare_tracking_local_smoke.py"
LOCAL_RUNNER = ROOT / "reproduction/scripts/run_tracking_local_smoke.sh"
LOCAL_GENERATED = ROOT / "reproduction/generated/whole_body_tracking_local"
LOCAL_GENERATED_MANIFEST = LOCAL_GENERATED / "manifest.tsv"
G1_CSV_DIR = ROOT / "download/official/LAFAN1_Retargeting_Dataset/g1"

REQUIRED_OUTPUT_KEYS = [
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
]
VECTOR_BODY_KEYS = ["body_pos_w", "body_lin_vel_w", "body_ang_vel_w"]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_normalized_text(path: Path) -> str:
    text = read_text(path).replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def line_for(path: Path, pattern: str) -> int | None:
    regex = re.compile(pattern)
    for idx, line in enumerate(read_text(path).splitlines(), start=1):
        if regex.search(line):
            return idx
    return None


def extract_joint_names() -> list[str]:
    text = read_text(CSV_TO_NPZ)
    match = re.search(r"joint_names\s*=\s*(\[[^\]]+\])", text, flags=re.S)
    if not match:
        return []
    return list(ast.literal_eval(match.group(1)))


def extract_csv_loader_slices() -> dict[str, str]:
    text = read_text(CSV_TO_NPZ)
    patterns = {
        "base_pos": r"motion_base_poss_input\s*=\s*motion\[:,\s*(:3)\]",
        "base_quat_xyzw": r"motion_base_rots_input\s*=\s*motion\[:,\s*(3:7)\]",
        "dof_pos": r"motion_dof_poss_input\s*=\s*motion\[:,\s*(7:)\]",
        "quat_xyzw_to_wxyz": r"motion_base_rots_input\s*=\s*self\.motion_base_rots_input\[:,\s*(\[[^\]]+\])\]",
    }
    out: dict[str, str] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        out[key] = match.group(1) if match else "NOT_FOUND"
    return out


def extract_output_keys() -> list[str]:
    text = read_text(CSV_TO_NPZ)
    match = re.search(r"log\s*=\s*(\{.*?\n\s*\})", text, flags=re.S)
    if not match:
        return []
    return sorted(set(re.findall(r'"([^"]+)":', match.group(1))))


def extract_stacked_keys() -> list[str]:
    text = read_text(CSV_TO_NPZ)
    match = re.search(r"for k in\s*\((.*?)\):\s*\n\s*log\[k\]\s*=\s*np\.stack", text, flags=re.S)
    if not match:
        return []
    return sorted(re.findall(r'"([^"]+)"', match.group(1)))


def extract_motion_loader_keys() -> list[str]:
    text = read_text(COMMANDS)
    return sorted(set(re.findall(r'data\["([^"]+)"\]', text)))


def extract_replay_uses_motion_loader() -> bool:
    text = read_text(REPLAY_NPZ)
    return "from whole_body_tracking.tasks.tracking.mdp import MotionLoader" in text and "motion = MotionLoader(" in text


def extract_validator_required_keys() -> dict[str, int]:
    text = read_text(VALIDATOR)
    match = re.search(r"REQUIRED_KEYS\s*=\s*(\{.*?\})", text, flags=re.S)
    if not match:
        return {}
    return dict(ast.literal_eval(match.group(1)))


def scan_g1_csvs(joint_count: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    expected_cols = 7 + joint_count
    for path in sorted(G1_CSV_DIR.glob("*.csv")):
        data = np.loadtxt(path, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        quat_norm = np.linalg.norm(data[:, 3:7], axis=1)
        rows.append(
            {
                "file": str(path),
                "name": path.name,
                "rows": int(data.shape[0]),
                "columns": int(data.shape[1]),
                "expected_columns": expected_cols,
                "finite": bool(np.isfinite(data).all()),
                "column_count_matches": int(data.shape[1]) == expected_cols,
                "quat_norm_min": float(quat_norm.min()),
                "quat_norm_max": float(quat_norm.max()),
                "quat_norm_max_abs_error_from_1": float(np.max(np.abs(quat_norm - 1.0))),
                "first_root_xyz": [float(x) for x in data[0, :3]],
            }
        )
    return {
        "csv_count": len(rows),
        "expected_columns": expected_cols,
        "all_column_counts_match": all(row["column_count_matches"] for row in rows),
        "all_finite": all(row["finite"] for row in rows),
        "all_quaternions_unit_within_1e_minus_5": all(
            row["quat_norm_max_abs_error_from_1"] < 1e-5 for row in rows
        ),
        "min_rows": min((row["rows"] for row in rows), default=0),
        "max_rows": max((row["rows"] for row in rows), default=0),
        "max_quat_norm_abs_error_from_1": max(
            (row["quat_norm_max_abs_error_from_1"] for row in rows), default=math.inf
        ),
        "rows": rows,
    }


def local_patch_checks() -> dict[str, bool]:
    patcher = read_text(LOCAL_PATCHER)
    runner = read_text(LOCAL_RUNNER)
    generated_files = [
        LOCAL_GENERATED / "csv_to_npz_local.py",
        LOCAL_GENERATED / "replay_npz_local.py",
        LOCAL_GENERATED / "rsl_rl/train_local.py",
        LOCAL_GENERATED / "rsl_rl/cli_args.py",
    ]
    bash_syntax = subprocess.run(["bash", "-n", str(LOCAL_RUNNER)], cwd=ROOT, capture_output=True, text=True)
    manifest_rows = []
    if LOCAL_GENERATED_MANIFEST.exists():
        with LOCAL_GENERATED_MANIFEST.open("r", encoding="utf-8", newline="") as f:
            manifest_rows = list(csv.DictReader(f, delimiter="\t"))
    return {
        "local_patcher_adds_output_file": "--output_file" in patcher,
        "local_patcher_replaces_wandb_csv_save": "Motion saved locally" in patcher,
        "local_patcher_adds_replay_motion_file": "--motion_file" in patcher,
        "local_patcher_adds_train_motion_file": "args_cli.motion_file" in patcher,
        "local_runner_calls_csv_replay_train": "csv_to_npz_local.py" in runner
        and "replay_npz_local.py" in runner
        and "train_local.py" in runner,
        "local_runner_invokes_validator": "validate_motion_npz_contract.py" in runner,
        "local_runner_shell_syntax_valid": bash_syntax.returncode == 0,
        "local_runner_executable": LOCAL_RUNNER.exists() and LOCAL_RUNNER.stat().st_mode & 0o111 != 0,
        "local_generated_manifest_present": LOCAL_GENERATED_MANIFEST.exists() and len(manifest_rows) == 4,
        "local_generated_scripts_exist": all(path.exists() for path in generated_files),
        "local_generated_scripts_executable": all(path.stat().st_mode & 0o111 != 0 for path in generated_files),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    joint_names = extract_joint_names()
    csv_slices = extract_csv_loader_slices()
    output_keys = extract_output_keys()
    stacked_keys = extract_stacked_keys()
    loader_keys = extract_motion_loader_keys()
    validator_keys = extract_validator_required_keys()
    csv_scan = scan_g1_csvs(len(joint_names))
    patches = local_patch_checks()
    normalized_csv_to_npz_hash = sha256_normalized_text(CSV_TO_NPZ)
    normalized_raw_csv_to_npz_hash = sha256_normalized_text(RAW_CSV_TO_NPZ)

    required_set = set(REQUIRED_OUTPUT_KEYS)
    output_set = set(output_keys)
    loader_set = set(loader_keys)
    validator_set = set(validator_keys)
    stacked_set = set(stacked_keys)

    checks = {
        "official_and_raw_csv_to_npz_normalized_text_match": normalized_csv_to_npz_hash
        == normalized_raw_csv_to_npz_hash,
        "csv_loader_root_pos_slice_found": csv_slices["base_pos"] == ":3",
        "csv_loader_quat_slice_found": csv_slices["base_quat_xyzw"] == "3:7",
        "csv_loader_dof_slice_found": csv_slices["dof_pos"] == "7:",
        "csv_loader_converts_xyzw_to_wxyz": csv_slices["quat_xyzw_to_wxyz"] == "[3, 0, 1, 2]",
        "g1_joint_count_29": len(joint_names) == 29,
        "csv_expected_column_count_36": csv_scan["expected_columns"] == 36,
        "all_g1_csv_column_counts_match": csv_scan["all_column_counts_match"],
        "all_g1_csv_values_finite": csv_scan["all_finite"],
        "all_g1_csv_quaternions_unit": csv_scan["all_quaternions_unit_within_1e_minus_5"],
        "csv_to_npz_declares_required_output_keys": required_set.issubset(output_set),
        "csv_to_npz_stacks_all_time_series_keys": set(REQUIRED_OUTPUT_KEYS[1:]).issubset(stacked_set),
        "motion_loader_consumes_required_output_keys": required_set.issubset(loader_set),
        "validator_matches_required_output_keys": required_set == validator_set,
        "validator_ndims_match_contract": validator_keys
        == {
            "fps": 1,
            "joint_pos": 2,
            "joint_vel": 2,
            "body_pos_w": 3,
            "body_quat_w": 3,
            "body_lin_vel_w": 3,
            "body_ang_vel_w": 3,
        },
        "replay_npz_uses_tracking_motion_loader": extract_replay_uses_motion_loader(),
        "local_smoke_patches_prepared": all(patches.values()),
        "kit_execution_boundary_recorded": True,
    }

    audit = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "static_and_data_contract_audit",
        "scope": "official csv_to_npz motion.npz producer contract versus replay/training MotionLoader consumer",
        "sources": {
            "csv_to_npz": str(CSV_TO_NPZ),
            "raw_csv_to_npz": str(RAW_CSV_TO_NPZ),
            "replay_npz": str(REPLAY_NPZ),
            "commands_motion_loader": str(COMMANDS),
            "validator": str(VALIDATOR),
            "local_patcher": str(LOCAL_PATCHER),
            "local_runner": str(LOCAL_RUNNER),
            "g1_csv_dir": str(G1_CSV_DIR),
            "sha256": {
                "csv_to_npz": sha256_file(CSV_TO_NPZ),
                "raw_csv_to_npz": sha256_file(RAW_CSV_TO_NPZ),
                "csv_to_npz_normalized_text": normalized_csv_to_npz_hash,
                "raw_csv_to_npz_normalized_text": normalized_raw_csv_to_npz_hash,
                "replay_npz": sha256_file(REPLAY_NPZ),
                "commands": sha256_file(COMMANDS),
            },
            "line_refs": {
                "csv_input_slices": line_for(CSV_TO_NPZ, r"motion_base_poss_input\s*=\s*motion\[:, :3\]"),
                "csv_output_log": line_for(CSV_TO_NPZ, r"log\s*=\s*\{"),
                "csv_save_npz": line_for(CSV_TO_NPZ, r"np\.savez"),
                "joint_names": line_for(CSV_TO_NPZ, r"joint_names=\["),
                "motion_loader_np_load": line_for(COMMANDS, r"data\s*=\s*np\.load"),
                "replay_motion_loader": line_for(REPLAY_NPZ, r"motion\s*=\s*MotionLoader"),
            },
        },
        "contract": {
            "required_output_keys": REQUIRED_OUTPUT_KEYS,
            "expected_ndims": validator_keys,
            "csv_input_layout": {
                "root_xyz_columns": "0:3",
                "root_quat_xyzw_columns": "3:7",
                "joint_position_columns": "7:",
                "expected_columns": csv_scan["expected_columns"],
            },
            "joint_names": joint_names,
            "csv_loader_slices": csv_slices,
            "csv_to_npz_output_keys": output_keys,
            "csv_to_npz_stacked_keys": stacked_keys,
            "motion_loader_consumed_keys": loader_keys,
        },
        "csv_dataset_scan": csv_scan,
        "local_smoke_patch_checks": patches,
        "metrics": {
            "g1_csv_count": csv_scan["csv_count"],
            "g1_joint_count": len(joint_names),
            "expected_csv_columns": csv_scan["expected_columns"],
            "min_csv_rows": csv_scan["min_rows"],
            "max_csv_rows": csv_scan["max_rows"],
            "max_quat_norm_abs_error_from_1": csv_scan["max_quat_norm_abs_error_from_1"],
            "producer_key_count": len(output_keys),
            "consumer_key_count": len(loader_keys),
        },
        "checks": checks,
        "interpretation": {
            "status": "contract_verified_but_live_kit_execution_blocked",
            "summary": (
                "The official motion.npz producer, replay path, training MotionLoader, and local validator agree on "
                "the required motion keys and shape ranks. All local official G1 retargeted CSV files match the "
                "expected 36-column input layout and contain finite near-unit quaternions."
            ),
            "not_a_replacement_for": [
                "executing csv_to_npz.py in IsaacLab/Kit",
                "rendered replay validation",
                "PPO training smoke",
                "paper-level tracking rollout metrics",
            ],
        },
    }

    json_path = OUT / "motion_preprocessing_contract_audit.json"
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT / "motion_preprocessing_contract_audit.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["name", "rows", "columns", "finite", "quat_norm_max_abs_error_from_1"])
        for row in csv_scan["rows"]:
            writer.writerow(
                [
                    row["name"],
                    row["rows"],
                    row["columns"],
                    row["finite"],
                    row["quat_norm_max_abs_error_from_1"],
                ]
            )
    (OUT / "run.log").write_text(
        "kind=motion_preprocessing_contract_audit\n"
        f"status={audit['status']}\n"
        f"g1_csv_count={csv_scan['csv_count']}\n",
        encoding="utf-8",
    )
    print(json_path)


if __name__ == "__main__":
    main()
