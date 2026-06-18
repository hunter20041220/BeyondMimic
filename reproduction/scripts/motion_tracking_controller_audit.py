#!/usr/bin/env python3
"""Audit the official motion_tracking_controller deployment repository."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
MTC = ROOT / "reproduction/third_party/official/motion_tracking_controller"
UNITREE = ROOT / "download/dependencies/unitree_bringup"
WBT = ROOT / "reproduction/third_party/official/whole_body_tracking"
OUT = ROOT / "res/tracking/motion_tracking_controller_audit"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def git_info(path: Path) -> dict[str, object]:
    head = (path / ".git/HEAD").read_text(encoding="utf-8").strip()
    ref = None
    commit = head
    if head.startswith("ref: "):
        ref = head.removeprefix("ref: ")
        ref_path = path / ".git" / ref
        if ref_path.exists():
            commit = ref_path.read_text(encoding="utf-8").strip()
    modified = []
    for rel in [
        "CMakeLists.txt",
        "README.md",
        "package.xml",
        "config/g1/controllers.yaml",
        "include/motion_tracking_controller/MotionCommand.h",
        "include/motion_tracking_controller/MotionObservation.h",
        "include/motion_tracking_controller/MotionOnnxPolicy.h",
        "include/motion_tracking_controller/MotionTrackingController.h",
        "src/MotionCommand.cpp",
        "src/MotionOnnxPolicy.cpp",
        "src/MotionTrackingController.cpp",
    ]:
        modified.append({"path": rel, "sha256": sha256_file(path / rel)})
    return {"head": head, "ref": ref, "commit": commit, "tracked_source_hashes": modified}


def package_deps() -> dict[str, object]:
    root = ET.parse(MTC / "package.xml").getroot()
    deps = []
    for child in root:
        if child.tag.endswith("depend") or child.tag == "buildtool_depend" or child.tag == "build_depend":
            deps.append({"tag": child.tag, "name": (child.text or "").strip()})
    return {
        "name": root.findtext("name"),
        "version": root.findtext("version"),
        "license": root.findtext("license"),
        "dependencies": deps,
    }


def cmake_audit() -> dict[str, object]:
    text = read(MTC / "CMakeLists.txt")
    return {
        "uses_ament_cmake_auto": "ament_cmake_auto" in text,
        "finds_eigen3": "find_package(Eigen3 REQUIRED)" in text,
        "exports_pluginlib": "pluginlib_export_plugin_description_file" in text,
        "installs_config_launch": "INSTALL_TO_SHARE config launch" in text,
        "sha256": sha256_file(MTC / "CMakeLists.txt"),
    }


def controllers_yaml(path: Path) -> dict[str, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    manager = data["controller_manager"]["ros__parameters"]
    walking = data["walking_controller"]["ros__parameters"]
    standby = data["standby_controller"]["ros__parameters"]
    return {
        "path": str(path),
        "manager_update_rate": manager.get("update_rate"),
        "controllers": {k: v["type"] for k, v in manager.items() if isinstance(v, dict) and "type" in v},
        "walking_update_rate": walking.get("update_rate"),
        "walking_policy_path": (walking.get("policy") or {}).get("path"),
        "standby_joint_count": len(standby.get("joint_names", [])),
        "standby_default_position_count": len(standby.get("default_position", [])),
        "standby_kp_count": len(standby.get("kp", [])),
        "standby_kd_count": len(standby.get("kd", [])),
        "sha256": sha256_file(path),
    }


def code_patterns() -> dict[str, object]:
    files = {
        "MotionOnnxPolicy.cpp": MTC / "src/MotionOnnxPolicy.cpp",
        "MotionCommand.cpp": MTC / "src/MotionCommand.cpp",
        "MotionTrackingController.cpp": MTC / "src/MotionTrackingController.cpp",
        "MotionObservation.h": MTC / "include/motion_tracking_controller/MotionObservation.h",
    }
    patterns = {
        "time_step_input": r'name2Index_\.at\("time_step"\)',
        "reference_outputs": r'joint_pos|joint_vel|body_pos_w|body_quat_w',
        "metadata_anchor": r'getMetadataStr\("anchor_body_name"\)',
        "metadata_body_names": r'getMetadataStr\("body_names"\)',
        "yaw_alignment": r"yawQuaternion",
        "rot6_observation": r"vector_t rot6\(6\)",
        "parser_motion_command": r'name == "motion"',
        "parser_motion_anchor_pos": r"motion_anchor_pos_b",
        "parser_motion_anchor_ori": r"motion_anchor_ori_b",
    }
    out = {}
    for name, path in files.items():
        text = read(path)
        out[name] = {
            "sha256": sha256_file(path),
            "patterns": {key: bool(re.search(pattern, text)) for key, pattern in patterns.items()},
        }
    return out


def launch_audit(path: Path) -> dict[str, object]:
    text = read(path)
    return {
        "path": str(path),
        "declares_policy_path": "'policy_path'" in text or '"policy_path"' in text,
        "declares_wandb_path": "'wandb_path'" in text or '"wandb_path"' in text,
        "declares_start_step": "'start_step'" in text or '"start_step"' in text,
        "uses_unitree_description": "FindPackageShare(\"unitree_description\")" in text,
        "uses_legged_bringup_utils": "legged_bringup.launch_utils" in text,
        "uses_mujoco": "mujoco_sim_ros2" in text,
        "uses_rosbag": "ros2', 'bag', 'record'" in text,
        "sha256": sha256_file(path),
    }


def exporter_contract() -> dict[str, object]:
    path = WBT / "source/whole_body_tracking/whole_body_tracking/utils/exporter.py"
    text = read(path)
    output_names = re.findall(r'"(actions|joint_pos|joint_vel|body_pos_w|body_quat_w|body_lin_vel_w|body_ang_vel_w)"', text)
    metadata_keys = re.findall(r'"(run_path|joint_names|joint_stiffness|joint_damping|default_joint_pos|command_names|observation_names|observation_history_lengths|action_scale|anchor_body_name|body_names)"\s*:', text)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "imports_onnx": "import onnx" in text,
        "input_names": ["obs", "time_step"] if 'input_names=["obs", "time_step"]' in text else [],
        "output_names": sorted(set(output_names), key=output_names.index),
        "metadata_keys": sorted(set(metadata_keys), key=metadata_keys.index),
        "uses_motion_command": "cmd: MotionCommand = env.command_manager.get_term(\"motion\")" in text,
        "time_step_clamp": "torch.clamp(time_step.long().squeeze(-1)" in text,
    }


def onnx_audit(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() else None,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    audit = {
        "motion_tracking_controller": git_info(MTC),
        "package": package_deps(),
        "cmake": cmake_audit(),
        "controllers": {
            "official_motion_tracking": controllers_yaml(MTC / "config/g1/controllers.yaml"),
            "unitree_bringup_reference": controllers_yaml(UNITREE / "config/g1/controllers.yaml"),
        },
        "code_patterns": code_patterns(),
        "launch": {
            "mujoco": launch_audit(MTC / "launch/mujoco.launch.py"),
            "real": launch_audit(MTC / "launch/real.launch.py"),
        },
        "whole_body_tracking_exporter": exporter_contract(),
        "onnx_reference": onnx_audit(UNITREE / "config/g1/policy.onnx"),
    }
    json_path = OUT / "motion_tracking_controller_audit.json"
    json_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    with (OUT / "motion_tracking_controller_audit.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["section", "key", "value"])
        for section, value in audit.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    writer.writerow([section, k, json.dumps(v, sort_keys=True)])
            else:
                writer.writerow([section, "", json.dumps(value, sort_keys=True)])
    (OUT / "run.log").write_text("kind=motion_tracking_controller_static_audit\nstatus=ok\n", encoding="utf-8")
    print(json_path)


if __name__ == "__main__":
    main()
