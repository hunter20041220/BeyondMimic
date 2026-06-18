#!/usr/bin/env python3
"""Build a debug-only motion policy ONNX contract fixture.

The host does not have the Python `onnx` package, and no trained BeyondMimic
motion policy checkpoint is available. This script therefore does not create a
real .onnx model. Instead, it materializes the exact input/output/metadata
contract required by the official exporter and C++ consumer as JSON/TSV/NPZ so
the interface can be audited without pretending a trained policy exists.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/motion_policy_onnx_contract_fixture"
CONTRACT_AUDIT = ROOT / "res/tracking/onnx_export_contract_audit/tracking_onnx_export_contract_audit.json"
ACTION_SCALE = ROOT / "res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json"
OBS_SCHEMA = ROOT / "res/tracking/observation_action_schema_audit/tracking_observation_action_schema_audit.json"
MOTION_FIXTURE = ROOT / "res/tracking/motion_npz_fixture/tracking_motion_npz_fixture.json"

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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_value(values: list[Any]) -> str:
    return ",".join(str(value) for value in values)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "field_type",
        "name",
        "shape",
        "dtype",
        "present_in_fixture",
        "required_by_exporter",
        "used_by_consumer",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    contract = load_json(CONTRACT_AUDIT)
    action_scale = load_json(ACTION_SCALE)
    obs_schema = load_json(OBS_SCHEMA)
    motion_fixture = load_json(MOTION_FIXTURE)

    joint_rows = action_scale["joint_rows"]
    joint_names = [row["joint_name"] for row in joint_rows]
    stiffness = np.asarray([row["stiffness"] for row in joint_rows], dtype=np.float32)
    damping = np.asarray([row["damping"] for row in joint_rows], dtype=np.float32)
    scale = np.asarray([row["action_scale"] for row in joint_rows], dtype=np.float32)
    default_joint_pos = np.zeros(len(joint_names), dtype=np.float32)
    body_names = obs_schema["body_names"]
    anchor_body_name = "torso_link"
    if anchor_body_name not in body_names:
        # The official flat env uses torso_link; local target list can be pelvis-anchored.
        anchor_body_name = body_names[0]

    obs_dim = int(obs_schema["metrics"]["policy_dimension"])
    action_dim = int(action_scale["metrics"]["joint_count"])
    body_count = len(body_names)
    batch = 1
    obs = np.zeros((batch, obs_dim), dtype=np.float32)
    time_step = np.asarray([[0]], dtype=np.int64)
    actions = np.zeros((batch, action_dim), dtype=np.float32)
    joint_pos = default_joint_pos.reshape(batch, action_dim).astype(np.float32)
    joint_vel = np.zeros((batch, action_dim), dtype=np.float32)
    body_pos_w = np.zeros((body_count, 3), dtype=np.float32)
    body_quat_w = np.zeros((body_count, 4), dtype=np.float32)
    body_quat_w[:, 0] = 1.0
    body_lin_vel_w = np.zeros((body_count, 3), dtype=np.float32)
    body_ang_vel_w = np.zeros((body_count, 3), dtype=np.float32)

    metadata = {
        "run_path": "debug_only_motion_policy_onnx_contract_fixture_not_trained",
        "joint_names": csv_value(joint_names),
        "joint_stiffness": csv_value([f"{value:.9g}" for value in stiffness.tolist()]),
        "joint_damping": csv_value([f"{value:.9g}" for value in damping.tolist()]),
        "default_joint_pos": csv_value([f"{value:.9g}" for value in default_joint_pos.tolist()]),
        "command_names": "motion",
        "observation_names": ",".join(row["term"] for row in obs_schema["observation_rows"] if row["group"] == "policy"),
        "observation_history_lengths": csv_value([1 for row in obs_schema["observation_rows"] if row["group"] == "policy"]),
        "action_scale": csv_value([f"{value:.9g}" for value in scale.tolist()]),
        "anchor_body_name": anchor_body_name,
        "body_names": csv_value(body_names),
    }

    npz_path = OUT / "debug_motion_policy_onnx_contract_fixture.npz"
    tmp_npz = npz_path.with_suffix(npz_path.suffix + ".tmp")
    with tmp_npz.open("wb") as f:
        np.savez_compressed(
            f,
            obs=obs,
            time_step=time_step,
            actions=actions,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            body_pos_w=body_pos_w,
            body_quat_w=body_quat_w,
            body_lin_vel_w=body_lin_vel_w,
            body_ang_vel_w=body_ang_vel_w,
            joint_names=np.asarray(joint_names),
            body_names=np.asarray(body_names),
            action_scale=scale,
        )
    tmp_npz.replace(npz_path)

    input_shapes = {"obs": list(obs.shape), "time_step": list(time_step.shape)}
    output_shapes = {
        "actions": list(actions.shape),
        "joint_pos": list(joint_pos.shape),
        "joint_vel": list(joint_vel.shape),
        "body_pos_w": list(body_pos_w.shape),
        "body_quat_w": list(body_quat_w.shape),
        "body_lin_vel_w": list(body_lin_vel_w.shape),
        "body_ang_vel_w": list(body_ang_vel_w.shape),
    }
    dtype_by_field = {
        "obs": str(obs.dtype),
        "time_step": str(time_step.dtype),
        "actions": str(actions.dtype),
        "joint_pos": str(joint_pos.dtype),
        "joint_vel": str(joint_vel.dtype),
        "body_pos_w": str(body_pos_w.dtype),
        "body_quat_w": str(body_quat_w.dtype),
        "body_lin_vel_w": str(body_lin_vel_w.dtype),
        "body_ang_vel_w": str(body_ang_vel_w.dtype),
    }
    exporter_fields = {
        "input": set(contract["exporter"]["inputs"]),
        "output": set(contract["exporter"]["outputs"]),
        "metadata": set(contract["exporter"]["metadata_keys"]),
    }
    consumer_fields = {
        "input": {"obs", "time_step"},
        "output": set(contract["consumer"]["consumed_outputs"]) | {"actions"},
        "metadata": set(contract["consumer"]["consumed_metadata"])
        | {
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
    }
    rows: list[dict[str, Any]] = []
    for name in REQUIRED_INPUTS:
        rows.append(
            {
                "field_type": "input",
                "name": name,
                "shape": json.dumps(input_shapes[name]),
                "dtype": dtype_by_field[name],
                "present_in_fixture": True,
                "required_by_exporter": name in exporter_fields["input"],
                "used_by_consumer": name in consumer_fields["input"],
            }
        )
    for name in REQUIRED_OUTPUTS:
        rows.append(
            {
                "field_type": "output",
                "name": name,
                "shape": json.dumps(output_shapes[name]),
                "dtype": dtype_by_field[name],
                "present_in_fixture": True,
                "required_by_exporter": name in exporter_fields["output"],
                "used_by_consumer": name in consumer_fields["output"],
            }
        )
    for name in REQUIRED_METADATA:
        rows.append(
            {
                "field_type": "metadata",
                "name": name,
                "shape": "",
                "dtype": "string",
                "present_in_fixture": name in metadata,
                "required_by_exporter": name in exporter_fields["metadata"],
                "used_by_consumer": name in consumer_fields["metadata"],
            }
        )

    checks = {
        "contract_audit_status_ok": contract["status"] == "ok",
        "npz_fixture_written": npz_path.is_file() and npz_path.stat().st_size > 0,
        "all_required_inputs_present": all(name in input_shapes for name in REQUIRED_INPUTS),
        "all_required_outputs_present": all(name in output_shapes for name in REQUIRED_OUTPUTS),
        "all_required_metadata_present": all(name in metadata for name in REQUIRED_METADATA),
        "obs_shape_matches_policy_dimension": input_shapes["obs"] == [1, obs_dim] and obs_dim == 160,
        "time_step_shape_1x1_int64": input_shapes["time_step"] == [1, 1] and time_step.dtype == np.int64,
        "action_and_joint_dims_29": output_shapes["actions"] == [1, 29]
        and output_shapes["joint_pos"] == [1, 29]
        and output_shapes["joint_vel"] == [1, 29],
        "body_outputs_match_14_target_bodies": output_shapes["body_pos_w"] == [14, 3]
        and output_shapes["body_quat_w"] == [14, 4]
        and output_shapes["body_lin_vel_w"] == [14, 3]
        and output_shapes["body_ang_vel_w"] == [14, 3],
        "body_quaternions_unit_wxyz": bool(np.allclose(np.linalg.norm(body_quat_w, axis=1), 1.0)),
        "metadata_joint_lengths_29": len(joint_names) == 29
        and len(stiffness) == 29
        and len(damping) == 29
        and len(scale) == 29,
        "metadata_body_names_14": len(body_names) == 14,
        "metadata_action_scale_matches_official_audit": bool(np.all(scale > 0.0))
        and action_scale["checks"]["action_scale_formula_matches_official"],
        "fixture_rows_cover_official_exporter_and_consumer": all(
            row["present_in_fixture"] and row["required_by_exporter"] for row in rows
        )
        and all(row["used_by_consumer"] for row in rows if row["name"] not in {"body_lin_vel_w", "body_ang_vel_w"}),
        "does_not_write_real_onnx": not (OUT / "debug_motion_policy_onnx_contract_fixture.onnx").exists(),
        "does_not_claim_trained_policy": True,
        "does_not_claim_goal_complete": True,
        "atomic_write_used": True,
    }
    metrics = {
        "input_count": len(REQUIRED_INPUTS),
        "output_count": len(REQUIRED_OUTPUTS),
        "metadata_count": len(REQUIRED_METADATA),
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "joint_count": len(joint_names),
        "target_body_count": body_count,
        "npz_size_bytes": npz_path.stat().st_size,
        "npz_sha256": sha256_file(npz_path),
        "consumer_unused_output_count": sum(
            1 for name in REQUIRED_OUTPUTS if name not in consumer_fields["output"]
        ),
        "failed_check_count": sum(1 for value in checks.values() if not value),
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "tracking_motion_policy_onnx_contract_fixture",
        "scope": (
            "Debug-only materialization of the official motion policy ONNX input/output/metadata contract. "
            "This is not a .onnx file and not a trained BeyondMimic policy."
        ),
        "source_artifacts": {
            "onnx_export_contract_audit": str(CONTRACT_AUDIT),
            "g1_action_scale_audit": str(ACTION_SCALE),
            "observation_action_schema_audit": str(OBS_SCHEMA),
            "motion_npz_fixture": str(MOTION_FIXTURE),
        },
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "metadata": metadata,
        "contract_rows": rows,
        "metrics": metrics,
        "checks": checks,
        "not_a_replacement_for": [
            "trained BeyondMimic motion-tracking policy checkpoint",
            "real ONNX export through live IsaacLab/Kit exporter",
            "ONNX Runtime inference",
            "ROS 2 motion_tracking_controller execution",
            "MuJoCo or real Unitree G1 deployment",
        ],
        "interpretation": {
            "paper_level_status": "debug_contract_fixture_only",
            "goal_complete": False,
            "why_not_complete": (
                "The fixture proves that the local metadata, tensor shapes, and official exporter/consumer field names "
                "can be materialized consistently. It does not provide a trained policy, an actual ONNX protobuf, "
                "IsaacLab export, rollout metrics, or deployment evidence."
            ),
        },
        "outputs": {
            "json": str(OUT / "tracking_motion_policy_onnx_contract_fixture.json"),
            "tsv": str(OUT / "tracking_motion_policy_onnx_contract_fixture.tsv"),
            "npz": str(npz_path),
        },
    }
    atomic_write_text(
        OUT / "tracking_motion_policy_onnx_contract_fixture.json",
        json.dumps(summary, indent=2, sort_keys=True),
    )
    atomic_write_tsv(OUT / "tracking_motion_policy_onnx_contract_fixture.tsv", rows)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "inputs": metrics["input_count"],
                "outputs": metrics["output_count"],
                "metadata": metrics["metadata_count"],
                "failed": metrics["failed_check_count"],
                "json": summary["outputs"]["json"],
            },
            sort_keys=True,
        )
    )
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
