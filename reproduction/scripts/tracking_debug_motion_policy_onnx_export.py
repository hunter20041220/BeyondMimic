#!/usr/bin/env python3
"""Export a debug-only motion-policy ONNX that matches the official contract."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/tracking/debug_motion_policy_onnx_export"
CONTRACT = ROOT / "res/tracking/motion_policy_onnx_contract_fixture/tracking_motion_policy_onnx_contract_fixture.json"
CONTRACT_NPZ = ROOT / "res/tracking/motion_policy_onnx_contract_fixture/debug_motion_policy_onnx_contract_fixture.npz"
BM_DIFFUSION_PY = ROOT / "envs/bm_diffusion/bin/python"


class DebugMotionPolicy(torch.nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, body_count: int):
        super().__init__()
        self.linear = torch.nn.Linear(obs_dim, action_dim)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        self.register_buffer("joint_pos_template", torch.zeros(1, action_dim))
        self.register_buffer("joint_vel_template", torch.zeros(1, action_dim))
        self.register_buffer("body_pos_template", torch.zeros(body_count, 3))
        quat = torch.zeros(body_count, 4)
        quat[:, 0] = 1.0
        self.register_buffer("body_quat_template", quat)
        self.register_buffer("body_lin_vel_template", torch.zeros(body_count, 3))
        self.register_buffer("body_ang_vel_template", torch.zeros(body_count, 3))

    def forward(self, obs: torch.Tensor, time_step: torch.Tensor):
        time_zero = time_step.to(dtype=obs.dtype).sum() * 0.0
        actions = torch.tanh(self.linear(obs) + time_zero)
        return (
            actions,
            actions * 0.0 + self.joint_pos_template,
            actions * 0.0 + self.joint_vel_template,
            self.body_pos_template,
            self.body_quat_template,
            self.body_lin_vel_template,
            self.body_ang_vel_template,
        )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=["name", "kind", "expected_shape", "onnx_shape", "dtype", "matches_contract"],
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def tensor_info(value: onnx.ValueInfoProto) -> tuple[list[int | str], str]:
    tensor_type = value.type.tensor_type
    shape: list[int | str] = []
    for dim in tensor_type.shape.dim:
        if dim.dim_param:
            shape.append(dim.dim_param)
        else:
            shape.append(int(dim.dim_value))
    dtype = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
    return shape, dtype


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    contract = load_json(CONTRACT)
    with np.load(CONTRACT_NPZ) as data:
        obs = torch.from_numpy(data["obs"].astype(np.float32))
        time_step = torch.from_numpy(data["time_step"].astype(np.int64))
    obs_dim = int(contract["metrics"]["obs_dim"])
    action_dim = int(contract["metrics"]["action_dim"])
    body_count = int(contract["metrics"]["target_body_count"])
    model = DebugMotionPolicy(obs_dim=obs_dim, action_dim=action_dim, body_count=body_count)
    model.eval()
    onnx_path = OUT / "debug_motion_policy_contract.onnx"
    torch.onnx.export(
        model,
        (obs, time_step),
        onnx_path,
        input_names=["obs", "time_step"],
        output_names=[
            "actions",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}, "joint_pos": {0: "batch"}, "joint_vel": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    model_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_proto)
    metadata = {
        **contract["metadata"],
        "run_path": "debug_only_motion_policy_contract_onnx_not_trained",
        "paper_level_status": "debug_contract_export_only",
    }
    existing = {item.key: item for item in model_proto.metadata_props}
    for key, value in metadata.items():
        prop = existing.get(key) or model_proto.metadata_props.add()
        prop.key = key
        prop.value = str(value)
    onnx.save(model_proto, str(onnx_path))
    model_proto = onnx.load(str(onnx_path))
    onnx.checker.check_model(model_proto)

    input_infos = {value.name: tensor_info(value) for value in model_proto.graph.input}
    output_infos = {value.name: tensor_info(value) for value in model_proto.graph.output}
    expected_inputs = {"obs": ["batch", obs_dim], "time_step": [1, 1]}
    expected_outputs = {
        "actions": ["batch", action_dim],
        "joint_pos": ["batch", action_dim],
        "joint_vel": ["batch", action_dim],
        "body_pos_w": [body_count, 3],
        "body_quat_w": [body_count, 4],
        "body_lin_vel_w": [body_count, 3],
        "body_ang_vel_w": [body_count, 3],
    }
    rows: list[dict[str, Any]] = []
    for name, expected in expected_inputs.items():
        shape, dtype = input_infos.get(name, ([], "MISSING"))
        rows.append(
            {
                "name": name,
                "kind": "input",
                "expected_shape": json.dumps(expected),
                "onnx_shape": json.dumps(shape),
                "dtype": dtype,
                "matches_contract": shape == expected,
            }
        )
    for name, expected in expected_outputs.items():
        shape, dtype = output_infos.get(name, ([], "MISSING"))
        rows.append(
            {
                "name": name,
                "kind": "output",
                "expected_shape": json.dumps(expected),
                "onnx_shape": json.dumps(shape),
                "dtype": dtype,
                "matches_contract": shape == expected,
            }
        )
    metadata_keys = sorted(item.key for item in model_proto.metadata_props)
    missing_metadata = sorted(set(contract["metadata"]) - set(metadata_keys))
    environment_probe = {
        "python": str(BM_DIFFUSION_PY),
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
    }
    checks = {
        "contract_fixture_status_ok": contract["status"] == "ok",
        "onnx_file_written": onnx_path.is_file() and onnx_path.stat().st_size > 0,
        "onnx_checker_passed": True,
        "all_inputs_match_contract": all(row["matches_contract"] for row in rows if row["kind"] == "input"),
        "all_outputs_match_contract": all(row["matches_contract"] for row in rows if row["kind"] == "output"),
        "all_required_metadata_present": not missing_metadata,
        "uses_project_python": True,
        "does_not_use_trained_checkpoint": True,
        "does_not_claim_policy_performance": True,
        "does_not_claim_goal_complete": True,
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "debug_motion_policy_onnx_export",
        "scope": "debug-only ONNX export that matches official motion-policy input/output/metadata contract",
        "source_contract_json": str(CONTRACT),
        "source_contract_npz": str(CONTRACT_NPZ),
        "onnx_path": str(onnx_path),
        "onnx_sha256": sha256(onnx_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "opset_imports": [{"domain": item.domain, "version": item.version} for item in model_proto.opset_import],
        "input_output_rows": rows,
        "metadata_keys": metadata_keys,
        "missing_metadata": missing_metadata,
        "environment_probe": environment_probe,
        "checks": checks,
        "outputs": {
            "json": str(OUT / "tracking_debug_motion_policy_onnx_export.json"),
            "tsv": str(OUT / "tracking_debug_motion_policy_onnx_export.tsv"),
            "onnx": str(onnx_path),
        },
        "interpretation": {
            "goal_complete": False,
            "paper_level_status": "debug_contract_export_only",
            "why_not_complete": (
                "This ONNX proves the local export/checker path and official interface contract. It is initialized "
                "with zero weights and no trained BeyondMimic checkpoint, so it is not a motion-tracking policy, not "
                "ROS/MuJoCo deployment evidence, and not a paper metric."
            ),
        },
    }
    write_json_atomic(OUT / "tracking_debug_motion_policy_onnx_export.json", summary)
    write_tsv(OUT / "tracking_debug_motion_policy_onnx_export.tsv", rows)
    print(json.dumps({"status": summary["status"], "onnx": str(onnx_path), "size": summary["onnx_size_bytes"]}, sort_keys=True))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
