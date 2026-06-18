#!/usr/bin/env python3
"""Audit official motion_tracking_controller runtime semantics without running ROS.

This is a static, source-level audit for the official C++ deployment controller.
It verifies the controller's ONNX reference-motion contract, motion-alignment
semantics, observation aliases, launch-mode differences, and host runtime gate.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
MTC = ROOT / "reproduction/third_party/official/motion_tracking_controller"
OUT = ROOT / "res/tracking/deployment_controller_semantics_audit"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def os_release() -> dict[str, str]:
    path = Path("/etc/os-release")
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def source_hashes() -> list[dict[str, Any]]:
    rels = [
        "README.md",
        "CMakeLists.txt",
        "package.xml",
        "motion_tracking_controller.xml",
        "config/g1/controllers.yaml",
        "include/motion_tracking_controller/MotionCommand.h",
        "include/motion_tracking_controller/MotionObservation.h",
        "include/motion_tracking_controller/MotionOnnxPolicy.h",
        "include/motion_tracking_controller/MotionTrackingController.h",
        "src/MotionCommand.cpp",
        "src/MotionOnnxPolicy.cpp",
        "src/MotionTrackingController.cpp",
        "launch/mujoco.launch.py",
        "launch/real.launch.py",
    ]
    return [
        {
            "path": str(MTC / rel),
            "relative_path": rel,
            "size_bytes": (MTC / rel).stat().st_size,
            "sha256": sha256_file(MTC / rel),
        }
        for rel in rels
    ]


def git_info() -> dict[str, Any]:
    head_path = MTC / ".git/HEAD"
    head = head_path.read_text(encoding="utf-8").strip() if head_path.is_file() else ""
    ref = ""
    commit = head
    if head.startswith("ref: "):
        ref = head[len("ref: ") :]
        ref_path = MTC / ".git" / ref
        if ref_path.is_file():
            commit = ref_path.read_text(encoding="utf-8").strip()
    config = read(MTC / ".git/config") if (MTC / ".git/config").is_file() else ""
    remotes = re.findall(r"url\s*=\s*(.+)", config)
    return {"head": head, "ref": ref, "commit": commit, "remotes": remotes}


def package_contract() -> dict[str, Any]:
    package_root = ET.parse(MTC / "package.xml").getroot()
    plugin_root = ET.parse(MTC / "motion_tracking_controller.xml").getroot()
    plugin_class = plugin_root.find("class")
    deps = [
        {"tag": child.tag, "name": (child.text or "").strip()}
        for child in package_root
        if (child.text or "").strip()
    ]
    return {
        "package_name": package_root.findtext("name"),
        "version": package_root.findtext("version"),
        "license": package_root.findtext("license"),
        "dependency_names": [item["name"] for item in deps],
        "plugin_class_name": plugin_class.attrib.get("name") if plugin_class is not None else "",
        "plugin_class_type": plugin_class.attrib.get("type") if plugin_class is not None else "",
        "plugin_base_class_type": plugin_class.attrib.get("base_class_type") if plugin_class is not None else "",
    }


def yaml_contract() -> dict[str, Any]:
    data = yaml.safe_load(read(MTC / "config/g1/controllers.yaml"))
    manager = data["controller_manager"]["ros__parameters"]
    state = data["state_estimator"]["ros__parameters"]
    standby = data["standby_controller"]["ros__parameters"]
    walking = data["walking_controller"]["ros__parameters"]
    return {
        "manager_update_rate_hz": manager.get("update_rate"),
        "controller_types": {
            key: value["type"]
            for key, value in manager.items()
            if isinstance(value, dict) and "type" in value
        },
        "state_estimator_base_name": state.get("model", {}).get("base_name"),
        "state_estimator_contact_names": state.get("model", {}).get("six_dof_contact_names", []),
        "state_estimator_position_frame_id": state.get("estimation", {}).get("position", {}).get("frame_id"),
        "standby_joint_count": len(standby.get("joint_names", [])),
        "standby_default_position_count": len(standby.get("default_position", [])),
        "standby_kp_count": len(standby.get("kp", [])),
        "standby_kd_count": len(standby.get("kd", [])),
        "walking_update_rate_hz": walking.get("update_rate"),
        "walking_policy_path_configured_in_static_yaml": bool((walking.get("policy") or {}).get("path")),
    }


def launch_contract() -> dict[str, Any]:
    mujoco = read(MTC / "launch/mujoco.launch.py")
    real = read(MTC / "launch/real.launch.py")
    return {
        "mujoco_active_controllers": re.findall(r'active_list\s*=\s*\[(.*?)\]', mujoco, flags=re.S),
        "real_active_controllers": re.findall(r'active_list\s*=\s*\[(.*?)\]', real, flags=re.S),
        "mujoco_ext_pos_topic": "/mid360" if "/mid360" in mujoco else "",
        "real_ext_pos_topic": "/glim/odom" if "/glim/odom" in real else "",
        "mujoco_uses_mujoco_ros2_control_plugin": "mujoco_ros2_control::MujocoRos2ControlPlugin" in mujoco,
        "mujoco_uses_unitree_description_and_robot_state_publisher": "FindPackageShare(\"unitree_description\")" in mujoco
        and "robot_state_publisher" in mujoco,
        "real_uses_ros2_control_node": 'executable="ros2_control_node"' in real,
        "real_uses_unitree_description_and_robot_state_publisher": "FindPackageShare(\"unitree_description\")" in real
        and "robot_state_publisher" in real,
        "real_passes_network_interface_to_xacro": "network_interface:=" in real,
        "real_records_mcap_rosbag": "'ros2', 'bag', 'record', '-s', 'mcap', '-a'" in real,
        "both_resolve_policy_path_or_wandb": all(
            text.count("download_wandb_onnx") >= 1
            and "walking_controller.policy.path" in text
            and "os.path.abspath" in text
            for text in [mujoco, real]
        ),
        "both_support_start_step_override": all(
            "walking_controller.motion.start_step" in text for text in [mujoco, real]
        ),
    }


def code_contract() -> dict[str, Any]:
    controller_cpp = read(MTC / "src/MotionTrackingController.cpp")
    policy_cpp = read(MTC / "src/MotionOnnxPolicy.cpp")
    policy_h = read(MTC / "include/motion_tracking_controller/MotionOnnxPolicy.h")
    command_cpp = read(MTC / "src/MotionCommand.cpp")
    command_h = read(MTC / "include/motion_tracking_controller/MotionCommand.h")
    obs_h = read(MTC / "include/motion_tracking_controller/MotionObservation.h")
    readme = read(MTC / "README.md")
    return {
        "controller_reads_policy_path_and_start_step": 'get_parameter("policy.path")' in controller_cpp
        and 'get_parameter("motion.start_step")' in controller_cpp,
        "controller_initializes_motion_onnx_policy": "std::make_shared<MotionOnnxPolicy>(policyPath, startStep)" in controller_cpp
        and "policy_->init()" in controller_cpp,
        "controller_populates_anchor_and_body_names_from_metadata": "cfg_.anchorBody = policy->getAnchorBodyName()" in controller_cpp
        and "cfg_.bodyNames = policy->getBodyNames()" in controller_cpp,
        "controller_registers_motion_command_term": 'name == "motion"' in controller_cpp
        and "std::make_shared<MotionCommandTerm>" in controller_cpp,
        "controller_registers_motion_observation_aliases": all(
            token in controller_cpp
            for token in [
                "motion_ref_pos_b",
                "motion_anchor_pos_b",
                "motion_ref_ori_b",
                "motion_anchor_ori_b",
                "robot_body_pos",
                "robot_body_ori",
            ]
        ),
        "policy_reset_sets_start_step_and_primes_outputs": "timeStep_ = startStep_" in policy_cpp
        and "forward(vector_t::Zero(getObservationSize()))" in policy_cpp,
        "policy_forward_feeds_incrementing_time_step": 'name2Index_.at("time_step")' in policy_cpp
        and "timeStep_++" in policy_cpp,
        "policy_reads_motion_reference_outputs": all(
            token in policy_cpp for token in ['"joint_pos"', '"joint_vel"', '"body_pos_w"', '"body_quat_w"']
        ),
        "policy_reconstructs_wxyz_quaternion": "ori.w() = quat(0)" in policy_cpp
        and "ori.coeffs().head(3) = quat.tail(3)" in policy_cpp,
        "policy_metadata_anchor_body_names": 'getMetadataStr("anchor_body_name")' in policy_cpp
        and 'getMetadataStr("body_names")' in policy_cpp,
        "policy_header_exposes_reference_getters": all(
            token in policy_h
            for token in [
                "getJointPosition",
                "getJointVelocity",
                "getBodyPositions",
                "getBodyOrientations",
                "getAnchorBodyName",
                "getBodyNames",
            ]
        ),
        "command_value_is_joint_pos_vel_concat": "motionPolicy_->getJointPosition(), motionPolicy_->getJointVelocity()" in command_cpp
        and "2 * model_->getNumJoints()" in command_h,
        "command_validates_anchor_and_body_frames": "Anchor body " in command_cpp
        and "not found" in command_cpp
        and "Frame " in command_cpp,
        "command_finds_anchor_motion_index_from_body_names": "cfg_.bodyNames[i] == cfg_.anchorBody" in command_cpp
        and "anchorMotionIndex_" in command_cpp,
        "command_aligns_initial_motion_to_robot_anchor_yaw": "yawQuaternion" in command_cpp
        and "worldToInit_ = worldToAnchor * initToAnchor.inverse()" in command_cpp,
        "command_computes_local_anchor_position_orientation": "getAnchorPositionLocal" in command_cpp
        and "getAnchorOrientationLocal" in command_cpp
        and "actInv" in command_cpp,
        "command_computes_local_robot_body_position_orientation": "getRobotBodyPositionLocal" in command_cpp
        and "getRobotBodyOrientationLocal" in command_cpp,
        "observation_dimensions_are_formula_based": all(
            token in obs_h
            for token in [
                "return 3;",
                "return 6;",
                "3 * commandTerm_->getCfg().bodyNames.size()",
                "6 * commandTerm_->getCfg().bodyNames.size()",
            ]
        ),
        "readme_has_research_only_real_robot_disclaimer": "dangerous" in readme.lower()
        and "research only" in readme.lower()
        and "no responsibility" in readme.lower(),
        "readme_documents_remote_controller_switches": all(
            token in readme for token in ["L1 + A", "R1 + A", "E-stop", "B"]
        ),
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["section", "key", "value"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    package = package_contract()
    config = yaml_contract()
    launch = launch_contract()
    code = code_contract()
    host_os = os_release()
    host_runtime = {
        "python_version": platform.python_version(),
        "ubuntu_version_id": host_os.get("VERSION_ID", ""),
        "ubuntu_codename": host_os.get("VERSION_CODENAME") or host_os.get("UBUNTU_CODENAME", ""),
        "ros2_path": shutil.which("ros2"),
        "colcon_path": shutil.which("colcon"),
        "rosdep_path": shutil.which("rosdep"),
    }

    controller_types = set(config["controller_types"].values())
    checks = {
        "official_source_tree_present": MTC.is_dir(),
        "git_commit_recorded": bool(git_info()["commit"]),
        "package_and_plugin_match_motion_tracking_controller": package["package_name"] == "motion_tracking_controller"
        and package["plugin_class_name"] == "motion_tracking_controller/MotionTrackingController"
        and package["plugin_base_class_type"] == "controller_interface::ControllerInterface",
        "package_declares_core_controller_dependencies": {
            "legged_rl_controllers",
            "legged_bringup",
            "rosbag2-storage-mcap",
        }.issubset(set(package["dependency_names"])),
        "launch_references_unitree_description_and_robot_state_publisher": launch[
            "mujoco_uses_unitree_description_and_robot_state_publisher"
        ]
        and launch["real_uses_unitree_description_and_robot_state_publisher"],
        "controller_yaml_has_expected_rates_and_29_joint_standby": config["manager_update_rate_hz"] == 500
        and config["walking_update_rate_hz"] == 50
        and config["standby_joint_count"] == 29
        and config["standby_default_position_count"] == 29
        and config["standby_kp_count"] == 29
        and config["standby_kd_count"] == 29,
        "controller_yaml_has_state_standby_walking_types": {
            "legged_controllers/StateEstimator",
            "legged_controllers/StandbyController",
            "motion_tracking_controller/MotionTrackingController",
        }.issubset(controller_types),
        "sim_launch_starts_walking_real_launch_starts_standby": "walking_controller" in "".join(
            launch["mujoco_active_controllers"]
        )
        and "standby_controller" in "".join(launch["real_active_controllers"]),
        "launches_support_policy_path_wandb_and_start_step": launch["both_resolve_policy_path_or_wandb"]
        and launch["both_support_start_step_override"],
        "real_launch_has_network_interface_and_rosbag_boundary": launch["real_uses_ros2_control_node"]
        and launch["real_passes_network_interface_to_xacro"]
        and launch["real_records_mcap_rosbag"],
        "policy_time_step_reference_output_metadata_contract": code["policy_forward_feeds_incrementing_time_step"]
        and code["policy_reads_motion_reference_outputs"]
        and code["policy_metadata_anchor_body_names"],
        "controller_motion_command_and_observation_alias_contract": code["controller_registers_motion_command_term"]
        and code["controller_registers_motion_observation_aliases"],
        "motion_alignment_uses_anchor_yaw_and_local_frame_observations": code[
            "command_aligns_initial_motion_to_robot_anchor_yaw"
        ]
        and code["command_computes_local_anchor_position_orientation"]
        and code["command_computes_local_robot_body_position_orientation"],
        "observation_dimensions_are_dynamic_body_count_formulas": code["observation_dimensions_are_formula_based"],
        "readme_records_real_robot_risk_and_remote_switches": code["readme_has_research_only_real_robot_disclaimer"]
        and code["readme_documents_remote_controller_switches"],
        "host_ros2_runtime_unavailable_or_not_jazzy_noble": host_runtime["ros2_path"] is None
        or host_runtime["ubuntu_version_id"] not in {"24.04"},
        "does_not_execute_ros_mujoco_or_robot": True,
        "does_not_claim_deployment_reproduction": True,
    }

    source_rows = source_hashes()
    metrics = {
        "audited_source_file_count": len(source_rows),
        "declared_dependency_count": len(package["dependency_names"]),
        "controller_type_count": len(config["controller_types"]),
        "manager_update_rate_hz": config["manager_update_rate_hz"],
        "walking_update_rate_hz": config["walking_update_rate_hz"],
        "standby_joint_count": config["standby_joint_count"],
        "motion_command_dim_for_29_joints": 58,
        "motion_observation_dim_formula_for_n_bodies": "9 + 9 * body_names_count",
        "motion_observation_dim_for_14_target_bodies": 135,
        "failed_check_count": sum(1 for value in checks.values() if not value),
    }
    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "failed",
        "experiment_type": "deployment_controller_semantics_audit",
        "scope": (
            "Static source-level semantics audit for the official motion_tracking_controller C++ deployment stack; "
            "no ROS, MuJoCo, rosbag, network interface, or Unitree hardware command is executed."
        ),
        "sources": {
            "motion_tracking_controller": str(MTC),
            "goal": str(ROOT / "goal.md"),
            "readme": str(MTC / "README.md"),
            "controller_cpp": str(MTC / "src/MotionTrackingController.cpp"),
            "policy_cpp": str(MTC / "src/MotionOnnxPolicy.cpp"),
            "command_cpp": str(MTC / "src/MotionCommand.cpp"),
            "observation_header": str(MTC / "include/motion_tracking_controller/MotionObservation.h"),
            "controllers_yaml": str(MTC / "config/g1/controllers.yaml"),
            "mujoco_launch": str(MTC / "launch/mujoco.launch.py"),
            "real_launch": str(MTC / "launch/real.launch.py"),
        },
        "not_a_replacement_for": [
            "colcon build on ROS 2 Jazzy/Noble",
            "ros2 launch motion_tracking_controller mujoco.launch.py execution",
            "trained BeyondMimic ONNX policy export",
            "rosbag from a live run",
            "TensorRT diffusion deployment",
            "real Unitree G1 deployment",
        ],
        "git": git_info(),
        "source_hashes": source_rows,
        "package": package,
        "controllers_yaml": config,
        "launch": launch,
        "code_semantics": code,
        "host_runtime": host_runtime,
        "metrics": metrics,
        "checks": checks,
        "interpretation": {
            "paper_level_status": "partial",
            "goal_complete": False,
            "why_not_complete": (
                "The official C++ deployment controller semantics are traceable at source level, but the current host "
                "does not provide a ROS 2 Jazzy/Noble runtime or real G1 hardware. This audit therefore cannot prove "
                "MuJoCo sim-to-sim execution, rosbag evidence, trained-policy ONNX behavior, TensorRT latency, or "
                "real-robot safety/performance."
            ),
        },
        "outputs": {
            "json": str(OUT / "tracking_deployment_controller_semantics_audit.json"),
            "tsv": str(OUT / "tracking_deployment_controller_semantics_audit.tsv"),
        },
    }

    rows = []
    for section in ["metrics", "checks", "host_runtime"]:
        for key, value in summary[section].items():
            rows.append({"section": section, "key": key, "value": json.dumps(value, sort_keys=True)})
    atomic_write_text(
        OUT / "tracking_deployment_controller_semantics_audit.json",
        json.dumps(summary, indent=2, sort_keys=True),
    )
    write_tsv(OUT / "tracking_deployment_controller_semantics_audit.tsv", rows)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "checks": len(checks),
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
