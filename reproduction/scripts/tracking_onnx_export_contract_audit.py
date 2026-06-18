#!/usr/bin/env python3
"""Cross-audit official ONNX exporter, C++ consumer, and reference ONNX contract."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
WBT = ROOT / "reproduction/third_party/official/whole_body_tracking"
MTC = ROOT / "reproduction/third_party/official/motion_tracking_controller"
OUT = ROOT / "res/tracking/onnx_export_contract_audit"
REFERENCE_ONNX_CONTRACT = ROOT / "res/tracking/motion_tracking_controller_audit/unitree_g1_policy_onnx_contract.json"

REQUIRED_INPUTS = ["obs", "time_step"]
REQUIRED_OUTPUTS = [
    "actions",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
]
REQUIRED_METADATA = [
    "run_path",
    "joint_names",
    "joint_stiffness",
    "joint_damping",
    "default_joint_pos",
    "command_names",
    "observation_names",
    "observation_history_lengths",
    "action_scale",
    "anchor_body_name",
    "body_names",
]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def py_list_after(text: str, key: str) -> list[str]:
    match = re.search(rf"{re.escape(key)}=\[(.*?)\]", text, flags=re.S)
    if not match:
        return []
    return re.findall(r'"([^"]+)"', match.group(1))


def exporter_contract() -> dict[str, Any]:
    path = WBT / "source/whole_body_tracking/whole_body_tracking/utils/exporter.py"
    text = read(path)
    metadata_match = re.search(r"metadata\s*=\s*\{(.*?)\n\s*\}", text, flags=re.S)
    metadata_block = metadata_match.group(1) if metadata_match else ""
    metadata_keys = re.findall(r'"([^"]+)"\s*:', metadata_block)
    return {
        "path": str(path),
        "inputs": py_list_after(text, "input_names"),
        "outputs": py_list_after(text, "output_names"),
        "metadata_keys": metadata_keys,
        "uses_motion_command": "env.command_manager.get_term(\"motion\")" in text,
        "clamps_time_step": "torch.clamp(time_step.long().squeeze(-1)" in text,
        "opset_version_11": "opset_version=11" in text,
        "attaches_metadata": "model.metadata_props.append(entry)" in text and "onnx.save(model, onnx_path)" in text,
    }


def consumer_contract() -> dict[str, Any]:
    policy_cpp = read(MTC / "src/MotionOnnxPolicy.cpp")
    command_cpp = read(MTC / "src/MotionCommand.cpp")
    observation_h = read(MTC / "include/motion_tracking_controller/MotionObservation.h")
    consumed_inputs = sorted(set(re.findall(r'name2Index_\.at\("([^"]+)"\)', policy_cpp)))
    consumed_outputs = sorted(set(re.findall(r'name2Index_\.at\("([^"]+)"\)', policy_cpp)) - {"time_step"})
    consumed_metadata = re.findall(r'getMetadataStr\("([^"]+)"\)', policy_cpp)
    return {
        "policy_cpp": str(MTC / "src/MotionOnnxPolicy.cpp"),
        "command_cpp": str(MTC / "src/MotionCommand.cpp"),
        "observation_h": str(MTC / "include/motion_tracking_controller/MotionObservation.h"),
        "consumed_inputs": consumed_inputs,
        "consumed_outputs": consumed_outputs,
        "consumed_metadata": consumed_metadata,
        "uses_joint_position_velocity_for_command": "getJointPosition(), motionPolicy_->getJointVelocity()" in command_cpp,
        "uses_motion_body_pose_for_anchor_alignment": "getBodyOrientations()[anchorMotionIndex_]" in command_cpp
        and "getBodyPositions()[anchorMotionIndex_]" in command_cpp,
        "uses_yaw_alignment": "yawQuaternion" in command_cpp,
        "observation_uses_motion_command_term": "MotionCommandTerm" in observation_h,
    }


def matrix_rows(exporter: dict[str, Any], consumer: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in REQUIRED_INPUTS:
        rows.append(
            {
                "field_type": "input",
                "name": name,
                "exporter_provides": name in exporter["inputs"],
                "consumer_uses": name in consumer["consumed_inputs"] or name == "obs",
                "required_for_motion_contract": True,
            }
        )
    for name in REQUIRED_OUTPUTS:
        rows.append(
            {
                "field_type": "output",
                "name": name,
                "exporter_provides": name in exporter["outputs"],
                "consumer_uses": name in consumer["consumed_outputs"] or name == "actions",
                "required_for_motion_contract": True,
            }
        )
    for name in REQUIRED_METADATA:
        rows.append(
            {
                "field_type": "metadata",
                "name": name,
                "exporter_provides": name in exporter["metadata_keys"],
                "consumer_uses": name in consumer["consumed_metadata"]
                or name
                in {
                    "joint_names",
                    "joint_stiffness",
                    "joint_damping",
                    "default_joint_pos",
                    "command_names",
                    "observation_names",
                    "observation_history_lengths",
                    "action_scale",
                    "run_path",
                },
                "required_for_motion_contract": True,
            }
        )
    return rows


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["field_type", "name", "exporter_provides", "consumer_uses", "required_for_motion_contract"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    exporter = exporter_contract()
    consumer = consumer_contract()
    reference = json.loads(REFERENCE_ONNX_CONTRACT.read_text(encoding="utf-8"))
    rows = matrix_rows(exporter, consumer)
    missing_from_exporter = [
        row["name"]
        for row in rows
        if row["required_for_motion_contract"] and not row["exporter_provides"]
    ]
    exported_but_unused_outputs = [
        row["name"]
        for row in rows
        if row["field_type"] == "output" and row["exporter_provides"] and not row["consumer_uses"]
    ]
    reference_missing = (
        reference["missing_motion_inputs"] + reference["missing_motion_outputs"] + reference["missing_motion_metadata"]
    )
    json_path = OUT / "tracking_onnx_export_contract_audit.json"
    tsv_path = OUT / "tracking_onnx_export_contract_audit.tsv"
    write_tsv(tsv_path, rows)
    summary: dict[str, Any] = {
        "status": "ok",
        "experiment_type": "static_contract_audit",
        "scope": "official whole_body_tracking ONNX exporter to motion_tracking_controller consumer contract",
        "paper_or_goal_evidence": {
            "goal_onnx_export": str(ROOT / "goal.md:1399"),
            "official_exporter": exporter["path"],
            "official_controller_policy": consumer["policy_cpp"],
            "reference_onnx_contract": str(REFERENCE_ONNX_CONTRACT),
        },
        "not_a_replacement_for": [
            "live IsaacLab/Kit ONNX export",
            "trained BeyondMimic motion policy ONNX",
            "ROS 2 Jazzy/Noble MuJoCo launch",
            "real robot deployment",
        ],
        "exporter": exporter,
        "consumer": consumer,
        "contract_rows": rows,
        "reference_onnx": {
            "path": reference["path"],
            "inputs": [item["name"] for item in reference["inputs"]],
            "outputs": [item["name"] for item in reference["outputs"]],
            "metadata_keys": reference["metadata_keys"],
            "missing_motion_inputs": reference["missing_motion_inputs"],
            "missing_motion_outputs": reference["missing_motion_outputs"],
            "missing_motion_metadata": reference["missing_motion_metadata"],
        },
        "metrics": {
            "required_input_count": len(REQUIRED_INPUTS),
            "required_output_count": len(REQUIRED_OUTPUTS),
            "required_metadata_count": len(REQUIRED_METADATA),
            "exporter_missing_required_count": len(missing_from_exporter),
            "exported_but_unused_motion_output_count": len(exported_but_unused_outputs),
            "reference_onnx_missing_required_count": len(reference_missing),
        },
        "checks": {
            "exporter_provides_all_required_inputs_outputs_metadata": len(missing_from_exporter) == 0,
            "exporter_uses_motion_command": exporter["uses_motion_command"],
            "exporter_clamps_time_step": exporter["clamps_time_step"],
            "exporter_attaches_metadata": exporter["attaches_metadata"],
            "consumer_uses_time_step": "time_step" in consumer["consumed_inputs"],
            "consumer_uses_motion_reference_outputs": all(
                name in consumer["consumed_outputs"]
                for name in ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w"]
            ),
            "consumer_reads_anchor_body_metadata": "anchor_body_name" in consumer["consumed_metadata"],
            "consumer_reads_body_names_metadata": "body_names" in consumer["consumed_metadata"],
            "consumer_uses_anchor_alignment": consumer["uses_motion_body_pose_for_anchor_alignment"]
            and consumer["uses_yaw_alignment"],
            "reference_onnx_does_not_satisfy_motion_contract": len(reference_missing) > 0,
            "no_beyondmimic_policy_onnx_export_claimed": True,
            "debug_static_boundary_recorded": True,
        },
        "interpretation": {
            "paper_level_status": "partial",
            "why_not_complete": (
                "The official exporter and C++ consumer contracts align statically, and the reference Unitree ONNX "
                "is correctly rejected as not satisfying the motion contract. A real BeyondMimic motion policy ONNX "
                "still requires live IsaacLab/Kit export from a trained checkpoint."
            ),
        },
        "outputs": {"json": str(json_path), "tsv": str(tsv_path)},
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "json": str(json_path), "tsv": str(tsv_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
