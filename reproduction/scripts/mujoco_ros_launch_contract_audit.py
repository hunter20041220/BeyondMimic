#!/usr/bin/env python3
"""Audit the official MuJoCo/ROS launch contract without executing ROS.

The host does not provide ROS 2 Jazzy/Noble, so this audit verifies the static
launch/package/config contract and records the runtime boundary explicitly.
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
OUT = ROOT / "res/tracking/mujoco_ros_launch_contract_audit"


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
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"')
    return values


def launch_arguments(text: str) -> dict[str, str | None]:
    args: dict[str, str | None] = {}
    pattern = re.compile(
        r"DeclareLaunchArgument\(\s*['\"](?P<name>[^'\"]+)['\"](?P<body>.*?)\)",
        flags=re.S,
    )
    for match in pattern.finditer(text):
        body = match.group("body")
        default_match = re.search(r"default_value\s*=\s*['\"]([^'\"]*)['\"]", body)
        args[match.group("name")] = default_match.group(1) if default_match else None
    return args


def launch_contract(path: Path, launch_kind: str) -> dict[str, Any]:
    text = read(path)
    args = launch_arguments(text)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "launch_arguments": args,
        "uses_policy_path_override": "walking_controller.policy.path" in text and "os.path.abspath" in text,
        "uses_wandb_download_when_policy_empty": "download_wandb_onnx" in text
        and "if not policy_path_value and wandb_path_value" in text,
        "uses_start_step_override": "walking_controller.motion.start_step" in text,
        "uses_ext_pos_corr_override": "height_sensor_noise" in text,
        "uses_unitree_description_xacro": "FindPackageShare(\"unitree_description\")" in text
        and "robot.xacro" in text,
        "uses_unitree_teleop_launch": "FindPackageShare('unitree_bringup')" in text
        and "teleop.launch.py" in text,
        "active_controllers": re.findall(r'active_list\s*=\s*\[(.*?)\]', text, flags=re.S),
        "inactive_controllers_from_config": "get_controller_names" in text,
        "uses_controller_spawner": "control_spawner" in text,
        "mujoco_sim_node": launch_kind == "mujoco" and "mujoco_sim_ros2" in text and "mujoco_sim" in text,
        "mujoco_ros2_control_plugin": "mujoco_ros2_control::MujocoRos2ControlPlugin" in text,
        "real_ros2_control_node": launch_kind == "real" and "ros2_control_node" in text,
        "real_network_interface_xacro_arg": launch_kind == "real" and "network_interface:=" in text,
        "real_rosbag_mcap_all_topics": launch_kind == "real"
        and "'ros2', 'bag', 'record', '-s', 'mcap', '-a'" in text,
        "real_rosbag_exclude_regex": launch_kind == "real" and "--exclude-regex" in text,
    }


def package_contract() -> dict[str, Any]:
    root = ET.parse(MTC / "package.xml").getroot()
    deps = [(child.tag, (child.text or "").strip()) for child in root]
    plugin = ET.parse(MTC / "motion_tracking_controller.xml").getroot()
    plugin_class = plugin.find("class")
    cmake = read(MTC / "CMakeLists.txt")
    return {
        "package_name": root.findtext("name"),
        "version": root.findtext("version"),
        "license": root.findtext("license"),
        "dependencies": [{"tag": tag, "name": name} for tag, name in deps if name],
        "dependency_names": [name for _, name in deps if name],
        "plugin_class_name": plugin_class.attrib.get("name") if plugin_class is not None else "",
        "plugin_class_type": plugin_class.attrib.get("type") if plugin_class is not None else "",
        "plugin_base_class_type": plugin_class.attrib.get("base_class_type") if plugin_class is not None else "",
        "cmake_uses_ament_auto": "ament_auto_find_build_dependencies" in cmake,
        "cmake_exports_pluginlib": "pluginlib_export_plugin_description_file" in cmake,
        "cmake_installs_config_launch": "INSTALL_TO_SHARE config launch" in cmake,
        "sha256": {
            "package_xml": sha256_file(MTC / "package.xml"),
            "plugin_xml": sha256_file(MTC / "motion_tracking_controller.xml"),
            "cmake": sha256_file(MTC / "CMakeLists.txt"),
        },
    }


def controller_config_contract() -> dict[str, Any]:
    path = MTC / "config/g1/controllers.yaml"
    data = yaml.safe_load(read(path))
    manager = data["controller_manager"]["ros__parameters"]
    state = data["state_estimator"]["ros__parameters"]
    standby = data["standby_controller"]["ros__parameters"]
    walking = data["walking_controller"]["ros__parameters"]
    joint_names = standby.get("joint_names", [])
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "manager_update_rate_hz": manager.get("update_rate"),
        "controller_types": {k: v["type"] for k, v in manager.items() if isinstance(v, dict) and "type" in v},
        "state_estimator_base_name": state.get("model", {}).get("base_name"),
        "state_estimator_contact_names": state.get("model", {}).get("six_dof_contact_names", []),
        "state_estimator_position_frame_id": state.get("estimation", {}).get("position", {}).get("frame_id"),
        "standby_joint_count": len(joint_names),
        "standby_default_position_count": len(standby.get("default_position", [])),
        "standby_kp_count": len(standby.get("kp", [])),
        "standby_kd_count": len(standby.get("kd", [])),
        "walking_update_rate_hz": walking.get("update_rate"),
    }


def readme_contract() -> dict[str, Any]:
    text = read(MTC / "README.md")
    commands = re.findall(r"ros2 launch motion_tracking_controller [^\n]+", text)
    return {
        "path": str(MTC / "README.md"),
        "sha256": sha256_file(MTC / "README.md"),
        "mentions_ros2_jazzy": "ROS 2 Jazzy" in text,
        "mentions_legged_control2": "legged_control2" in text,
        "mentions_unitree_bringup": "unitree_bringup" in text,
        "mentions_mujoco_launch": "mujoco.launch.py" in text,
        "mentions_real_launch": "real.launch.py" in text,
        "mentions_network_interface": "network_interface" in text,
        "mentions_policy_path": "policy_path" in text,
        "mentions_wandb_path": "wandb_path" in text,
        "ros2_launch_commands": commands,
    }


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["section", "key", "value"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mujoco = launch_contract(MTC / "launch/mujoco.launch.py", "mujoco")
    real = launch_contract(MTC / "launch/real.launch.py", "real")
    package = package_contract()
    config = controller_config_contract()
    readme = readme_contract()
    os_info = os_release()
    host = {
        "python_version": platform.python_version(),
        "ubuntu_version_id": os_info.get("VERSION_ID", ""),
        "ubuntu_codename": os_info.get("VERSION_CODENAME") or os_info.get("UBUNTU_CODENAME", ""),
        "ros2_path": shutil.which("ros2"),
        "colcon_path": shutil.which("colcon"),
        "rosdep_path": shutil.which("rosdep"),
    }

    required_mujoco_args = {"robot_type", "policy_path", "start_step", "ext_pos_corr", "wandb_path"}
    required_real_args = required_mujoco_args | {"network_interface"}
    dependency_names = set(package["dependency_names"])
    controller_types = config["controller_types"]
    checks = {
        "readme_mentions_ros2_jazzy": readme["mentions_ros2_jazzy"],
        "readme_has_mujoco_policy_path_command": any(
            "mujoco.launch.py" in cmd and "policy_path:=" in cmd for cmd in readme["ros2_launch_commands"]
        ),
        "readme_has_real_network_interface_command": any(
            "real.launch.py" in cmd and "network_interface:=" in cmd for cmd in readme["ros2_launch_commands"]
        ),
        "package_declares_required_deps": {"legged_rl_controllers", "legged_bringup", "rosbag2-storage-mcap"}.issubset(
            dependency_names
        ),
        "plugin_declares_motion_tracking_controller": package["plugin_class_name"]
        == "motion_tracking_controller/MotionTrackingController",
        "cmake_exports_plugin_and_installs_launch_config": package["cmake_exports_pluginlib"]
        and package["cmake_installs_config_launch"],
        "controllers_yaml_has_expected_controllers": {
            "state_estimator",
            "standby_controller",
            "walking_controller",
        }.issubset(controller_types),
        "controllers_yaml_500hz_manager_50hz_walking": config["manager_update_rate_hz"] == 500
        and config["walking_update_rate_hz"] == 50,
        "standby_joint_impedance_counts_match_29": config["standby_joint_count"] == 29
        and config["standby_default_position_count"] == 29
        and config["standby_kp_count"] == 29
        and config["standby_kd_count"] == 29,
        "mujoco_launch_declares_required_args": required_mujoco_args.issubset(mujoco["launch_arguments"]),
        "mujoco_launch_uses_sim_node_and_plugin": mujoco["mujoco_sim_node"] and mujoco["mujoco_ros2_control_plugin"],
        "mujoco_launch_uses_policy_or_wandb_resolution": mujoco["uses_policy_path_override"]
        and mujoco["uses_wandb_download_when_policy_empty"],
        "mujoco_launch_spawns_state_estimator_and_walking": "state_estimator" in "".join(
            mujoco["active_controllers"]
        )
        and "walking_controller" in "".join(mujoco["active_controllers"]),
        "real_launch_declares_required_args": required_real_args.issubset(real["launch_arguments"]),
        "real_launch_uses_ros2_control_and_network_interface": real["real_ros2_control_node"]
        and real["real_network_interface_xacro_arg"],
        "real_launch_records_mcap_with_exclusions": real["real_rosbag_mcap_all_topics"]
        and real["real_rosbag_exclude_regex"],
        "real_launch_spawns_state_estimator_and_standby": "state_estimator" in "".join(real["active_controllers"])
        and "standby_controller" in "".join(real["active_controllers"]),
        "host_runtime_gate_recorded": host["ros2_path"] is None
        and host["ubuntu_version_id"] != "24.04",
    }

    summary: dict[str, Any] = {
        "status": "ok" if all(checks.values()) else "check_failed",
        "experiment_type": "static_launch_contract_audit",
        "scope": "official motion_tracking_controller MuJoCo/ROS launch contract, package dependencies, and host runtime boundary",
        "sources": {
            "motion_tracking_controller": str(MTC),
            "mujoco_launch": str(MTC / "launch/mujoco.launch.py"),
            "real_launch": str(MTC / "launch/real.launch.py"),
            "controllers_yaml": str(MTC / "config/g1/controllers.yaml"),
            "package_xml": str(MTC / "package.xml"),
            "plugin_xml": str(MTC / "motion_tracking_controller.xml"),
            "readme": str(MTC / "README.md"),
        },
        "not_a_replacement_for": [
            "ROS 2 Jazzy/Noble workspace build",
            "MuJoCo sim-to-sim launch execution",
            "ROS bag recording from a live run",
            "real Unitree G1 deployment",
        ],
        "host_runtime": host,
        "readme": readme,
        "package": package,
        "controllers_yaml": config,
        "launch": {"mujoco": mujoco, "real": real},
        "metrics": {
            "readme_ros2_launch_command_count": len(readme["ros2_launch_commands"]),
            "mujoco_launch_argument_count": len(mujoco["launch_arguments"]),
            "real_launch_argument_count": len(real["launch_arguments"]),
            "declared_dependency_count": len(package["dependency_names"]),
            "controller_type_count": len(controller_types),
            "standby_joint_count": config["standby_joint_count"],
            "manager_update_rate_hz": config["manager_update_rate_hz"],
            "walking_update_rate_hz": config["walking_update_rate_hz"],
        },
        "checks": checks,
        "interpretation": {
            "paper_level_status": "partial",
            "why_not_complete": (
                "The official sim-to-sim/real launch contract is statically consistent, but this host lacks ROS 2 "
                "Jazzy/Noble tooling and remains Ubuntu 20.04, so no MuJoCo launch, rosbag, or robot deployment was run."
            ),
        },
    }
    json_path = OUT / "mujoco_ros_launch_contract_audit.json"
    tsv_path = OUT / "mujoco_ros_launch_contract_audit.tsv"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    rows = []
    for section in ["metrics", "checks", "host_runtime"]:
        for key, value in summary[section].items():
            rows.append({"section": section, "key": key, "value": json.dumps(value, sort_keys=True)})
    write_tsv(tsv_path, rows)
    print(json.dumps({"status": summary["status"], "json": str(json_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
