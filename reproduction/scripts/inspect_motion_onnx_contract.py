#!/usr/bin/env python3
"""Inspect and validate the BeyondMimic motion-policy ONNX contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx


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


def value_info_summary(value_info) -> dict[str, object]:
    shape = []
    tensor_type = value_info.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.dim_value:
            shape.append(dim.dim_value)
        elif dim.dim_param:
            shape.append(dim.dim_param)
        else:
            shape.append(None)
    return {
        "name": value_info.name,
        "elem_type": tensor_type.elem_type,
        "shape": shape,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path", type=Path)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--require-motion-contract", action="store_true")
    args = parser.parse_args()

    model = onnx.load(args.onnx_path)
    inputs = [value_info_summary(v) for v in model.graph.input]
    outputs = [value_info_summary(v) for v in model.graph.output]
    metadata = {p.key: p.value for p in model.metadata_props}

    summary = {
        "path": str(args.onnx_path),
        "ir_version": model.ir_version,
        "opset_imports": [{"domain": o.domain, "version": o.version} for o in model.opset_import],
        "inputs": inputs,
        "outputs": outputs,
        "metadata_keys": sorted(metadata),
        "missing_motion_inputs": [k for k in REQUIRED_INPUTS if k not in {i["name"] for i in inputs}],
        "missing_motion_outputs": [k for k in REQUIRED_OUTPUTS if k not in {o["name"] for o in outputs}],
        "missing_motion_metadata": [k for k in REQUIRED_METADATA if k not in metadata],
    }

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))

    if args.require_motion_contract:
        missing = (
            summary["missing_motion_inputs"]
            + summary["missing_motion_outputs"]
            + summary["missing_motion_metadata"]
        )
        if missing:
            raise SystemExit(f"ONNX does not satisfy BeyondMimic motion contract; missing: {missing}")


if __name__ == "__main__":
    main()
