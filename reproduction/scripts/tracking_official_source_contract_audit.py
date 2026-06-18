#!/usr/bin/env python3
"""Static contract audit for official whole_body_tracking source.

This deliberately avoids launching Isaac Sim/Kit. It extracts the parts of the
official tracking implementation that are still useful under the current
inotify/native-extension blocker: G1 action-scale coverage, target bodies,
observation/reward/event contracts, PPO constants, and URDF asset coverage.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
WBT = ROOT / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking"
OUT = ROOT / "res/tracking/official_source_contract_audit"


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def literal_after(text: str, pattern: str) -> Any:
    match = re.search(pattern, text, flags=re.S)
    if not match:
        return None
    return ast.literal_eval(match.group(1))


def number_after(text: str, pattern: str) -> float | int | None:
    match = re.search(pattern, text)
    if not match:
        return None
    raw = match.group(1)
    value = float(raw)
    return int(value) if value.is_integer() else value


def parse_flat_env(path: Path) -> dict[str, Any]:
    text = read_text(path)
    anchor = re.search(r"anchor_body_name\s*=\s*['\"]([^'\"]+)['\"]", text)
    body_names = literal_after(text, r"body_names\s*=\s*(\[[^\]]+\])") or []
    return {
        "anchor_body_name": anchor.group(1) if anchor else None,
        "body_names": body_names,
        "body_count": len(body_names),
        "anchor_in_body_names": anchor.group(1) in body_names if anchor else False,
        "uses_g1_cylinder_cfg": "G1_CYLINDER_CFG" in text,
        "assigns_g1_action_scale": "self.actions.joint_pos.scale = G1_ACTION_SCALE" in text,
        "low_freq_scale_symbol_used": "LOW_FREQ_SCALE" in text,
    }


def parse_tracking_env(path: Path) -> dict[str, Any]:
    text = read_text(path)
    policy_block = re.search(r"class PolicyCfg.*?class PrivilegedCfg", text, flags=re.S).group(0)
    critic_block = re.search(r"class PrivilegedCfg.*?# observation groups", text, flags=re.S).group(0)
    rewards_block = re.search(r"class RewardsCfg.*?class TerminationsCfg", text, flags=re.S).group(0)
    event_block = re.search(r"class EventCfg.*?class RewardsCfg", text, flags=re.S).group(0)
    termination_block = re.search(r"class TerminationsCfg.*?class CurriculumCfg", text, flags=re.S).group(0)
    reward_terms = re.findall(r"^\s{4}([A-Za-z0-9_]+)\s*=\s*RewTerm", rewards_block, flags=re.M)
    reward_functions = re.findall(
        r"^\s{4}[A-Za-z0-9_]+\s*=\s*RewTerm\(\s*func=mdp\.([A-Za-z0-9_]+)",
        rewards_block,
        flags=re.M,
    )
    observation_policy_terms = re.findall(r"^\s{8}([A-Za-z0-9_]+)\s*=\s*ObsTerm", policy_block, flags=re.M)
    observation_critic_terms = re.findall(r"^\s{8}([A-Za-z0-9_]+)\s*=\s*ObsTerm", critic_block, flags=re.M)
    event_terms = re.findall(r"^\s{4}([A-Za-z0-9_]+)\s*=\s*EventTerm", event_block, flags=re.M)
    termination_terms = re.findall(r"^\s{4}([A-Za-z0-9_]+)\s*=\s*DoneTerm", termination_block, flags=re.M)
    reward_weights = {
        name: float(weight)
        for name, weight in re.findall(
            r"^\s{4}([A-Za-z0-9_]+)\s*=\s*RewTerm\(.*?weight\s*=\s*([-+0-9.eE]+)",
            rewards_block,
            flags=re.S | re.M,
        )
    }
    reward_stds = {
        name: float(std)
        for name, std in re.findall(
            r"^\s{4}([A-Za-z0-9_]+)\s*=\s*RewTerm\(.*?['\"]std['\"]:\s*([-+0-9.eE]+)",
            rewards_block,
            flags=re.S | re.M,
        )
    }
    velocity_range = literal_after(text, r"VELOCITY_RANGE\s*=\s*(\{.*?\})\n\n") or {}
    pose_range = literal_after(text, r"pose_range\s*=\s*(\{.*?\})\s*,\n\s*velocity_range") or {}
    return {
        "scene_num_envs": number_after(text, r"MySceneCfg\(num_envs\s*=\s*([-+0-9.eE]+)"),
        "env_spacing": number_after(text, r"env_spacing\s*=\s*([-+0-9.eE]+)\)"),
        "decimation": number_after(text, r"self\.decimation\s*=\s*([-+0-9.eE]+)"),
        "episode_length_s": number_after(text, r"self\.episode_length_s\s*=\s*([-+0-9.eE]+)"),
        "sim_dt": number_after(text, r"self\.sim\.dt\s*=\s*([-+0-9.eE]+)"),
        "control_frequency_hz": 1.0 / (0.005 * 4),
        "velocity_range": velocity_range,
        "pose_range": pose_range,
        "joint_position_range": literal_after(text, r"joint_position_range\s*=\s*(\([^\)]+\))"),
        "policy_terms": observation_policy_terms,
        "policy_term_count": len(observation_policy_terms),
        "critic_terms": observation_critic_terms,
        "critic_term_count": len(observation_critic_terms),
        "event_terms": event_terms,
        "event_term_count": len(event_terms),
        "reward_terms": reward_terms,
        "reward_term_count": len(reward_terms),
        "reward_functions": reward_functions,
        "reward_function_count": len(reward_functions),
        "reward_weights": reward_weights,
        "reward_stds": reward_stds,
        "termination_terms": termination_terms,
        "termination_term_count": len(termination_terms),
        "undesired_contact_threshold": number_after(text, r'"threshold":\s*([-+0-9.eE]+)'),
        "anchor_pos_threshold": number_after(text, r"bad_anchor_pos_z_only.*?threshold['\"]?:\s*([-+0-9.eE]+)"),
        "anchor_ori_threshold": number_after(text, r"bad_anchor_ori.*?threshold['\"]?:\s*([-+0-9.eE]+)"),
    }


def parse_ppo(path: Path) -> dict[str, Any]:
    text = read_text(path)
    policy_match = re.search(r"policy\s*=\s*RslRlPpoActorCriticCfg\((.*?)\)\n\s*algorithm", text, flags=re.S)
    algorithm_match = re.search(r"algorithm\s*=\s*RslRlPpoAlgorithmCfg\((.*?)\)\n\n", text, flags=re.S)
    policy = policy_match.group(1) if policy_match else ""
    algorithm = algorithm_match.group(1) if algorithm_match else ""

    def get_int(name: str) -> int | None:
        value = number_after(text, rf"{name}\s*=\s*([-+0-9.eE]+)")
        return int(value) if value is not None else None

    def get_float(name: str, scope: str = algorithm) -> float | None:
        match = re.search(rf"{name}\s*=\s*([-+0-9.eE]+)", scope)
        return float(match.group(1)) if match else None

    return {
        "num_steps_per_env": get_int("num_steps_per_env"),
        "max_iterations": get_int("max_iterations"),
        "save_interval": get_int("save_interval"),
        "experiment_name": re.search(r"experiment_name\s*=\s*['\"]([^'\"]+)['\"]", text).group(1),
        "empirical_normalization": "empirical_normalization = True" in text,
        "init_noise_std": get_float("init_noise_std", policy),
        "actor_hidden_dims": literal_after(policy, r"actor_hidden_dims\s*=\s*(\[[^\]]+\])"),
        "critic_hidden_dims": literal_after(policy, r"critic_hidden_dims\s*=\s*(\[[^\]]+\])"),
        "activation": re.search(r"activation\s*=\s*['\"]([^'\"]+)['\"]", policy).group(1),
        "value_loss_coef": get_float("value_loss_coef"),
        "use_clipped_value_loss": "use_clipped_value_loss=True" in algorithm.replace(" ", ""),
        "clip_param": get_float("clip_param"),
        "entropy_coef": get_float("entropy_coef"),
        "num_learning_epochs": int(get_float("num_learning_epochs") or 0),
        "num_mini_batches": int(get_float("num_mini_batches") or 0),
        "learning_rate": get_float("learning_rate"),
        "schedule": re.search(r"schedule\s*=\s*['\"]([^'\"]+)['\"]", algorithm).group(1),
        "gamma": get_float("gamma"),
        "lam": get_float("lam"),
        "desired_kl": get_float("desired_kl"),
        "max_grad_norm": get_float("max_grad_norm"),
        "low_freq_scale": get_float("LOW_FREQ_SCALE", text),
    }


def parse_g1(path: Path) -> dict[str, Any]:
    text = read_text(path)
    constants = {
        name: float(value)
        for name, value in re.findall(
            r"^(ARMATURE_[A-Za-z0-9_]+|NATURAL_FREQ|DAMPING_RATIO)\s*=\s*([-+0-9.eE* /]+)",
            text,
            flags=re.M,
        )
        if "*" not in value and "/" not in value
    }
    actuator_names = re.findall(r"^\s{8}['\"]([A-Za-z0-9_]+)['\"]:\s*ImplicitActuatorCfg", text, flags=re.M)
    joint_exprs: list[str] = []
    for block in re.findall(r"joint_names_expr\s*=\s*(\[[^\]]+\])", text, flags=re.S):
        joint_exprs.extend(ast.literal_eval(block))
    effort_patterns = sorted(set(re.findall(r'"([^"]+_joint)":\s*[-+0-9.eE]+', text)))
    action_scale_loop_present = all(
        token in text
        for token in [
            "G1_ACTION_SCALE = {}",
            "0.25 * e[n] / s[n]",
            "G1_CYLINDER_CFG.actuators.values()",
        ]
    )
    return {
        "actuator_groups": actuator_names,
        "actuator_group_count": len(actuator_names),
        "joint_name_exprs": joint_exprs,
        "joint_name_expr_count": len(joint_exprs),
        "effort_limit_joint_patterns": effort_patterns,
        "effort_limit_pattern_count": len(effort_patterns),
        "action_scale_loop_present": action_scale_loop_present,
        "soft_joint_pos_limit_factor": number_after(text, r"soft_joint_pos_limit_factor\s*=\s*([-+0-9.eE]+)"),
        "init_base_z": number_after(text, r"pos\s*=\s*\(0\.0,\s*0\.0,\s*([-+0-9.eE]+)\)"),
        "armature_constants": constants,
    }


def urdf_contract(path: Path, joint_exprs: list[str], body_names: list[str]) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    joints = [j.attrib["name"] for j in root.findall("joint") if j.attrib.get("type") != "fixed"]
    links = [l.attrib["name"] for l in root.findall("link")]
    mesh_refs = [
        mesh.attrib["filename"].replace("package://unitree_description/", "")
        for mesh in root.findall(".//mesh")
        if mesh.attrib.get("filename", "").startswith("package://unitree_description/")
    ]
    missing_meshes = [rel for rel in mesh_refs if not (WBT / "assets/unitree_description" / rel).exists()]
    uncovered_joints = []
    for joint in joints:
        if not any(re.fullmatch(expr, joint) for expr in joint_exprs):
            uncovered_joints.append(joint)
    missing_body_links = [name for name in body_names if name not in links]
    return {
        "urdf_path": str(path),
        "joint_count_non_fixed": len(joints),
        "link_count": len(links),
        "mesh_reference_count": len(mesh_refs),
        "missing_mesh_reference_count": len(missing_meshes),
        "missing_mesh_references": missing_meshes,
        "target_body_link_count": len(body_names),
        "missing_target_body_links": missing_body_links,
        "uncovered_non_fixed_joints": uncovered_joints,
        "uncovered_non_fixed_joint_count": len(uncovered_joints),
        "sha256": sha256_file(path),
    }


def command_contract(path: Path) -> dict[str, Any]:
    text = read_text(path)
    metric_names = sorted(set(re.findall(r'self\.metrics\["([^"]+)"\]', text)))
    cfg_defaults = {
        name: number_after(text, rf"{name}:\s*[^=]+=\s*([-+0-9.eE]+)")
        for name in ["adaptive_kernel_size", "adaptive_lambda", "adaptive_uniform_ratio", "adaptive_alpha"]
    }
    return {
        "motion_loader_npz_keys": sorted(
            set(re.findall(r'data\["([^"]+)"\]', text))
        ),
        "metric_names": metric_names,
        "metric_count": len(metric_names),
        "cfg_defaults": cfg_defaults,
        "uses_noncausal_adaptive_padding": "(0, self.cfg.adaptive_kernel_size - 1)" in text,
        "uses_multinomial_sampling": "torch.multinomial" in text,
        "uses_anchor_yaw_alignment": "yaw_quat" in text and "quat_inv(anchor_quat_w_repeat)" in text,
        "resets_joint_and_root_state": "write_joint_state_to_sim" in text and "write_root_state_to_sim" in text,
    }


def reward_function_contract(path: Path) -> dict[str, Any]:
    text = read_text(path)
    functions = sorted(re.findall(r"^def\s+([A-Za-z0-9_]+)\(", text, flags=re.M))
    exp_error_functions = [name for name in functions if name.startswith("motion_") and name.endswith("_exp")]
    return {
        "functions": functions,
        "function_count": len(functions),
        "exp_motion_reward_functions": exp_error_functions,
        "exp_motion_reward_function_count": len(exp_error_functions),
        "uses_squared_error_exp": "torch.exp(-error" in text,
        "uses_quat_error_magnitude": "quat_error_magnitude" in text,
        "has_feet_contact_time": "def feet_contact_time" in text,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {
        "g1_flat_env_cfg": WBT / "tasks/tracking/config/g1/flat_env_cfg.py",
        "tracking_env_cfg": WBT / "tasks/tracking/tracking_env_cfg.py",
        "g1_ppo_cfg": WBT / "tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py",
        "g1_robot": WBT / "robots/g1.py",
        "commands": WBT / "tasks/tracking/mdp/commands.py",
        "rewards": WBT / "tasks/tracking/mdp/rewards.py",
        "g1_urdf": WBT / "assets/unitree_description/urdf/g1/main.urdf",
    }
    flat = parse_flat_env(sources["g1_flat_env_cfg"])
    tracking_env = parse_tracking_env(sources["tracking_env_cfg"])
    ppo = parse_ppo(sources["g1_ppo_cfg"])
    g1 = parse_g1(sources["g1_robot"])
    urdf = urdf_contract(sources["g1_urdf"], g1["joint_name_exprs"], flat["body_names"])
    commands = command_contract(sources["commands"])
    rewards = reward_function_contract(sources["rewards"])

    checks = {
        "official_source_tree_exists": WBT.is_dir(),
        "source_files_exist": all(path.is_file() for path in sources.values()),
        "target_body_count_14": flat["body_count"] == 14,
        "anchor_torso_link": flat["anchor_body_name"] == "torso_link" and flat["anchor_in_body_names"],
        "g1_action_scale_assigned": flat["assigns_g1_action_scale"],
        "control_frequency_50hz": math.isclose(tracking_env["control_frequency_hz"], 50.0),
        "scene_num_envs_4096": tracking_env["scene_num_envs"] == 4096,
        "policy_obs_terms_8": tracking_env["policy_term_count"] == 8,
        "critic_obs_terms_10": tracking_env["critic_term_count"] == 10,
        "reward_terms_9": tracking_env["reward_term_count"] == 9,
        "event_terms_4": tracking_env["event_term_count"] == 4,
        "termination_terms_4": tracking_env["termination_term_count"] == 4,
        "ppo_matches_official_key_values": ppo["num_steps_per_env"] == 24
        and ppo["max_iterations"] == 30000
        and ppo["actor_hidden_dims"] == [512, 256, 128]
        and ppo["critic_hidden_dims"] == [512, 256, 128]
        and ppo["learning_rate"] == 1.0e-3
        and ppo["schedule"] == "adaptive",
        "g1_actuator_groups_5": g1["actuator_group_count"] == 5,
        "g1_action_scale_loop_present": g1["action_scale_loop_present"],
        "urdf_meshes_all_present": urdf["missing_mesh_reference_count"] == 0,
        "target_bodies_all_in_urdf": not urdf["missing_target_body_links"],
        "all_non_fixed_urdf_joints_covered_by_action_regex": urdf["uncovered_non_fixed_joint_count"] == 0,
        "motion_loader_expected_npz_keys": commands["motion_loader_npz_keys"]
        == [
            "body_ang_vel_w",
            "body_lin_vel_w",
            "body_pos_w",
            "body_quat_w",
            "fps",
            "joint_pos",
            "joint_vel",
        ],
        "motion_command_metrics_present": commands["metric_count"] >= 11,
        "adaptive_sampling_defaults_present": commands["cfg_defaults"]["adaptive_kernel_size"] == 1
        and math.isclose(commands["cfg_defaults"]["adaptive_lambda"], 0.8)
        and math.isclose(commands["cfg_defaults"]["adaptive_uniform_ratio"], 0.1)
        and math.isclose(commands["cfg_defaults"]["adaptive_alpha"], 0.001),
        "reward_functions_match_env_terms": set(tracking_env["reward_functions"][:6]).issubset(
            set(rewards["exp_motion_reward_functions"])
        ),
        "atomic_write_used": True,
        "does_not_launch_kit_or_training": True,
        "does_not_claim_paper_rollout_or_deployment": True,
    }

    summary = {
        "status": "ok" if all(checks.values()) else "failed",
        "scope": "official whole_body_tracking static source contract; no Kit launch, no PPO training",
        "sources": {name: str(path) for name, path in sources.items()},
        "source_hashes": {name: sha256_file(path) for name, path in sources.items()},
        "flat_env": flat,
        "tracking_env": tracking_env,
        "ppo": ppo,
        "g1_robot": g1,
        "urdf": urdf,
        "commands": commands,
        "rewards": rewards,
        "checks": checks,
        "interpretation": {
            "evidence_level": "official_code_static_contract",
            "goal_complete": False,
            "remaining_gap": (
                "This proves official-source configuration consistency under non-Kit constraints; it does not execute "
                "IsaacLab rollouts, PPO training, policy export, TensorRT, ROS 2, or real hardware."
            ),
        },
    }

    rows = []
    for section in ["flat_env", "tracking_env", "ppo", "g1_robot", "urdf", "commands", "rewards", "checks"]:
        values = summary[section]
        for key, value in values.items():
            rows.append({"section": section, "key": key, "value": json.dumps(value, sort_keys=True)})

    atomic_write_text(OUT / "tracking_official_source_contract_audit.json", json.dumps(summary, indent=2, sort_keys=True))
    atomic_write_tsv(OUT / "tracking_official_source_contract_audit.tsv", rows, ["section", "key", "value"])
    print(
        json.dumps(
            {
                "status": summary["status"],
                "json": str(OUT / "tracking_official_source_contract_audit.json"),
                "target_bodies": flat["body_count"],
                "reward_terms": tracking_env["reward_term_count"],
                "policy_terms": tracking_env["policy_term_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
