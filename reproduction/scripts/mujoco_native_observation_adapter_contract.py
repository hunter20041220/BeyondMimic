#!/usr/bin/env python3
"""Audit the native MuJoCo 160-D observation adapter contract.

This is a contract gate, not a rollout.  The current MuJoCo videos are weak
because a trained IsaacLab/RSL-RL PPO checkpoint is only meaningful when the
runtime reconstructs the exact policy observation semantics used during
training:

    [command, motion_anchor_pos_b, motion_anchor_ori_b, base_lin_vel,
     base_ang_vel, joint_pos_rel, joint_vel_rel, last_action]

The dangerous failure mode is to concatenate an arbitrary 160-D vector and feed
it to the actor.  That can produce plausible-looking but wrong actions and
collapse videos into a generic leaning pose.  This audit records the official
source contract, the MuJoCo reconstruction requirements, the empirical
normalizer requirement, and the remaining validation gaps before any native
MuJoCo PPO/VAE/diffusion rollout can be claimed.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/audits/mujoco_native_observation_adapter_contract"
JSON_OUT = OUT / "mujoco_native_observation_adapter_contract.json"
TSV_OUT = OUT / "mujoco_native_observation_adapter_contract.tsv"
MD_OUT = OUT / "mujoco_native_observation_adapter_contract.md"

FILES = {
    "schema_audit": ROOT / "res/tracking/observation_action_schema_audit/tracking_observation_action_schema_audit.json",
    "motion_controller_audit": ROOT / "res/tracking/motion_tracking_controller_audit/motion_tracking_controller_audit.json",
    "onnx_contract_audit": ROOT / "res/tracking/onnx_export_contract_audit/tracking_onnx_export_contract_audit.json",
    "mujoco_action_adapter": ROOT
    / "res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json",
    "mujoco_observation_math_parity": ROOT
    / "res/audits/mujoco_observation_math_parity/mujoco_observation_math_parity_audit.json",
    "mujoco_observation_same_state_parity": ROOT
    / "res/audits/mujoco_observation_same_state_parity/mujoco_observation_same_state_parity_audit.json",
    "mujoco_observation_runtime_parity": ROOT
    / "res/audits/mujoco_observation_runtime_parity/mujoco_observation_runtime_parity_audit.json",
    "mujoco_torso_frame_offset": ROOT
    / "res/audits/mujoco_torso_frame_offset/mujoco_torso_frame_offset_audit.json",
    "isaaclab_observation_sample_gate": ROOT
    / "res/audits/isaaclab_observation_manager_sample_gate/isaaclab_observation_manager_sample_gate.json",
    "isaaclab_observation_sample": ROOT
    / "res/audits/isaaclab_observation_manager_sample_gate/isaaclab_policy_obs_sample.json",
    "mujoco_control_contract": ROOT / "res/audits/mujoco_control_contract_audit/mujoco_control_contract_audit.json",
    "official_tracking_cfg": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/tracking_env_cfg.py",
    "official_g1_ppo_cfg": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py",
    "official_observations": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/observations.py",
    "official_commands": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/commands.py",
    "official_exporter": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/utils/exporter.py",
    "isaaclab_exporter": ROOT
    / "reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab_rl/isaaclab_rl/"
    "rsl_rl/exporter.py",
    "controller_motion_observation_h": ROOT
    / "reproduction/third_party/official/motion_tracking_controller/include/motion_tracking_controller/"
    "MotionObservation.h",
    "controller_motion_command_cpp": ROOT
    / "reproduction/third_party/official/motion_tracking_controller/src/MotionCommand.cpp",
    "controller_motion_policy_cpp": ROOT
    / "reproduction/third_party/official/motion_tracking_controller/src/MotionOnnxPolicy.cpp",
    "native_probe_script": ROOT / "reproduction/scripts/stage1_multisource_quality_gated_native_ppo_mujoco_probe.py",
    "clean_walk_suite_script": ROOT / "reproduction/scripts/render_clean_walk_mujoco_control_suite.py",
}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def line_ref(path: Path, needle: str) -> str:
    text = read_text(path)
    for idx, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return f"{path}:{idx}"
    return str(path)


def has(path_key: str, *needles: str) -> bool:
    text = read_text(FILES[path_key])
    return all(needle in text for needle in needles)


def trailing_dim_is_160(shape: list[int]) -> bool:
    return bool(shape and shape[-1] == 160)


def observation_rows(schema: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    offset = 0
    for row in schema.get("observation_rows", []):
        if row.get("group") != "policy":
            continue
        dim = int(row.get("dimension", 0))
        rows.append(
            {
                "term": row.get("term"),
                "function": row.get("function"),
                "order": int(row.get("order", len(rows))),
                "dimension": dim,
                "slice_start": offset,
                "slice_end": offset + dim,
                "has_training_noise": bool(row.get("has_noise")),
                "noise_min": row.get("noise_min"),
                "noise_max": row.get("noise_max"),
            }
        )
        offset += dim
    return rows


def term_requirements() -> dict[str, dict[str, Any]]:
    return {
        "command": {
            "mujoco_source_needed": "reference motion joint_pos and joint_vel at the same continuous time_step",
            "official_semantics": "MotionCommand.command = cat([joint_pos, joint_vel])",
            "implementation_status": "contract_known_not_runtime_validated",
            "required_validation": "compare exact 58-D command slice against IsaacLab observation_manager for the same motion file and time_step",
            "failure_mode": "wrong phase/time-step or reset-spliced commands can make a good policy chase discontinuous targets",
            "notes": "Requires the exact motion file, joint order, phase/time-step handling, and no reset-spliced time jumps.",
        },
        "motion_anchor_pos_b": {
            "mujoco_source_needed": "robot anchor pose and reference anchor pose after official yaw/translation alignment",
            "official_semantics": "subtract_frame_transforms(robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w)",
            "implementation_status": "approximate_probe_exists_not_validated_against_isaaclab",
            "required_validation": "match IsaacLab subtract_frame_transforms output after MotionCommand yaw-only re-anchoring",
            "failure_mode": "meter-scale fake anchor error can drive the actor into a persistent leaning/recovery posture",
            "notes": "This is the most failure-prone term.  Wrong world-to-init or yaw alignment makes the actor see meter-scale fake error.",
        },
        "motion_anchor_ori_b": {
            "mujoco_source_needed": "robot anchor quaternion and aligned reference anchor quaternion converted to rot6D",
            "official_semantics": "relative anchor orientation via subtract_frame_transforms, then first two rotation-matrix columns",
            "implementation_status": "approximate_probe_exists_not_validated_against_isaaclab",
            "required_validation": "match IsaacLab matrix_from_quat(...)[..., :2].reshape ordering and quaternion convention",
            "failure_mode": "rot6D column/order mismatch can make the actor believe the target is rotated even while standing",
            "notes": "Rot6D order must match IsaacLab matrix_from_quat(...)[..., :2].reshape.",
        },
        "base_lin_vel": {
            "mujoco_source_needed": "robot base/root linear velocity in the same body/local frame as IsaacLab mdp.base_lin_vel",
            "official_semantics": "IsaacLab mdp.base_lin_vel from ArticulationData.root_lin_vel_b",
            "implementation_status": "approximate_probe_exists_not_validated_against_isaaclab",
            "required_validation": "match IsaacLab root_lin_vel_b for the same MuJoCo/IsaacLab state, not world-frame qvel",
            "failure_mode": "world/body-frame velocity mismatch looks like a constant disturbance and biases recovery actions",
            "notes": "Using world-frame qvel directly is wrong; body-frame conversion and root/body choice must be verified.",
        },
        "base_ang_vel": {
            "mujoco_source_needed": "robot base/root angular velocity in the same body/local frame as IsaacLab mdp.base_ang_vel",
            "official_semantics": "IsaacLab mdp.base_ang_vel from ArticulationData.root_ang_vel_b",
            "implementation_status": "approximate_probe_exists_not_validated_against_isaaclab",
            "required_validation": "match IsaacLab root_ang_vel_b and MuJoCo quaternion/qvel angular convention",
            "failure_mode": "angular velocity convention mismatch can suppress leg-lift behavior and over-trigger torso stabilization",
            "notes": "MuJoCo qvel angular convention must be checked against IsaacLab root_ang_vel_b.",
        },
        "joint_pos": {
            "mujoco_source_needed": "current 29 joint positions minus the exact IsaacLab default_joint_pos used by the policy",
            "official_semantics": "mdp.joint_pos_rel",
            "implementation_status": "partial_default_pose_warning",
            "required_validation": "verify default_joint_pos from exported metadata or IsaacLab ArticulationData, then compare 29-D slice",
            "failure_mode": "default-pose mismatch shifts every policy input and can make neutral standing look like crouching/leaning",
            "notes": "Deployment default and IsaacLab nominal default differ at ankle pitch by about 0.033 rad; prefer ONNX metadata/default_joint_pos.",
        },
        "joint_vel": {
            "mujoco_source_needed": "current 29 joint velocities in official joint order",
            "official_semantics": "mdp.joint_vel_rel",
            "implementation_status": "contract_known_not_runtime_validated",
            "required_validation": "compare MuJoCo qvel joint slice against IsaacLab joint_vel_rel in official joint order",
            "failure_mode": "joint-order or qvel-index mismatch corrupts feedback even if the action order is correct",
            "notes": "Requires MuJoCo qvel order to match official action/joint rows exactly.",
        },
        "actions": {
            "mujoco_source_needed": "previous clipped normalized policy action actually applied to PD setpoints",
            "official_semantics": "mdp.last_action",
            "implementation_status": "bug_fixed_in_clean_walk_suite_but_not_global_gate",
            "required_validation": "prove last_action is the previous normalized action applied by the same controller variant",
            "failure_mode": "feeding teacher or reference actions as last_action contaminates VAE/diffusion closed-loop observations",
            "notes": "Using teacher action as last_action for VAE/diffusion variants contaminates downstream observations.",
        },
    }


def build_term_rows(schema: dict[str, Any]) -> list[dict[str, Any]]:
    requirements = term_requirements()
    rows = []
    for row in observation_rows(schema):
        req = requirements.get(row["term"], {})
        rows.append(
            {
                **row,
                "mujoco_source_needed": req.get("mujoco_source_needed", ""),
                "official_semantics": req.get("official_semantics", ""),
                "implementation_status": req.get("implementation_status", "unknown"),
                "required_validation": req.get("required_validation", ""),
                "failure_mode": req.get("failure_mode", ""),
                "notes": req.get("notes", ""),
            }
        )
    return rows


def build_validation_matrix(term_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in term_rows:
        status = str(row.get("implementation_status", ""))
        rows.append(
            {
                "term": row["term"],
                "slice": f"{row['slice_start']}:{row['slice_end']}",
                "dimension": row["dimension"],
                "official_semantics": row.get("official_semantics", ""),
                "required_validation": row.get("required_validation", ""),
                "validated_against_isaaclab_observation_manager": False,
                "validated_against_motion_tracking_controller": False,
                "ready_for_native_mujoco_policy_rollout": False,
                "status": status,
                "why_blocking": row.get("failure_mode", ""),
            }
        )
    return rows


def checkpoint_normalizer_summary() -> dict[str, Any]:
    best = read_json(
        ROOT / "res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/stage1_multisource_best_teacher.json"
    )
    checkpoint = Path((best.get("best_checkpoint") or {}).get("checkpoint", ""))
    summary: dict[str, Any] = {
        "best_teacher_json": str(
            ROOT
            / "res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/stage1_multisource_best_teacher.json"
        ),
        "checkpoint": str(checkpoint) if str(checkpoint) != "." else "",
        "checkpoint_exists": checkpoint.is_file(),
        "obs_norm_state_dict_present": False,
        "model_state_dict_present": False,
        "actor_input_dim_160": False,
        "actor_output_dim_29": False,
        "loaded_with_torch": False,
        "error": "",
    }
    if not checkpoint.is_file():
        return summary
    try:
        import torch

        payload = torch.load(checkpoint, map_location="cpu")
        state = payload.get("model_state_dict", {})
        summary.update(
            {
                "loaded_with_torch": True,
                "model_state_dict_present": bool(state),
                "obs_norm_state_dict_present": "obs_norm_state_dict" in payload,
                "actor_input_dim_160": False,
                "actor_output_dim_29": False,
                "checkpoint_iter": int(payload.get("iter", -1)),
            }
        )
        if "actor.0.weight" in state:
            summary["actor_0_weight_shape"] = list(state["actor.0.weight"].shape)
            summary["actor_input_dim_160"] = list(state["actor.0.weight"].shape) == [512, 160]
        if "actor.6.weight" in state:
            summary["actor_6_weight_shape"] = list(state["actor.6.weight"].shape)
            summary["actor_output_dim_29"] = list(state["actor.6.weight"].shape) == [29, 128]
        if "obs_norm_state_dict" in payload:
            norm = payload["obs_norm_state_dict"]
            summary["obs_norm_keys"] = sorted(norm.keys())
            for key in ["_mean", "_std"]:
                value = norm.get(key)
                if value is not None:
                    shape = list(value.shape)
                    summary[f"obs_norm_{key}_shape"] = shape
                    summary[f"obs_norm_{key}_trailing_dim_160"] = trailing_dim_is_160(shape)
    except Exception as exc:  # pragma: no cover - audit must retain failure.
        summary["error"] = repr(exc)
        for python in [
            ROOT / "envs/bm_tracking/bin/python",
            ROOT / "envs/bm_diffusion/bin/python",
            ROOT / "envs/bm_tracking/bin/python3",
            ROOT / "envs/bm_diffusion/bin/python3",
        ]:
            if not python.is_file():
                continue
            code = r"""
import json
import sys
import torch
path = sys.argv[1]
payload = torch.load(path, map_location="cpu")
state = payload.get("model_state_dict", {})
out = {
    "loaded_with_torch": True,
    "torch_python": sys.executable,
    "model_state_dict_present": bool(state),
    "obs_norm_state_dict_present": "obs_norm_state_dict" in payload,
    "actor_input_dim_160": False,
    "actor_output_dim_29": False,
    "checkpoint_iter": int(payload.get("iter", -1)),
}
if "actor.0.weight" in state:
    out["actor_0_weight_shape"] = list(state["actor.0.weight"].shape)
    out["actor_input_dim_160"] = list(state["actor.0.weight"].shape) == [512, 160]
if "actor.6.weight" in state:
    out["actor_6_weight_shape"] = list(state["actor.6.weight"].shape)
    out["actor_output_dim_29"] = list(state["actor.6.weight"].shape) == [29, 128]
if "obs_norm_state_dict" in payload:
    norm = payload["obs_norm_state_dict"]
    out["obs_norm_keys"] = sorted(norm.keys())
    for key in ["_mean", "_std"]:
        value = norm.get(key)
        if value is not None:
            shape = list(value.shape)
            out[f"obs_norm_{key}_shape"] = shape
            out[f"obs_norm_{key}_trailing_dim_160"] = bool(shape and shape[-1] == 160)
print(json.dumps(out, sort_keys=True))
"""
            proc = subprocess.run(
                [str(python), "-c", code, str(checkpoint)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
                check=False,
            )
            if proc.returncode == 0:
                try:
                    payload_summary = json.loads(proc.stdout.strip().splitlines()[-1])
                except Exception as parse_exc:  # pragma: no cover
                    summary["fallback_parse_error"] = repr(parse_exc)
                    summary["fallback_stdout_tail"] = proc.stdout[-1000:]
                    continue
                summary.update(payload_summary)
                summary["error"] = ""
                break
            summary["fallback_error"] = proc.stderr[-1000:] or proc.stdout[-1000:]
    return summary


def build_checks(
    schema: dict[str, Any],
    term_rows: list[dict[str, Any]],
    normalizer: dict[str, Any],
    math_parity: dict[str, Any],
    same_state_parity: dict[str, Any],
    runtime_parity: dict[str, Any],
    torso_frame_offset: dict[str, Any],
    sample_gate: dict[str, Any],
    sample: dict[str, Any],
) -> dict[str, bool]:
    tracking_cfg = read_text(FILES["official_tracking_cfg"])
    g1_ppo_cfg = read_text(FILES["official_g1_ppo_cfg"])
    native_probe = read_text(FILES["native_probe_script"])
    action_adapter = read_json(FILES["mujoco_action_adapter"])
    control_contract = read_json(FILES["mujoco_control_contract"])
    dim_sum = sum(int(row["dimension"]) for row in term_rows)
    action_interpretation = action_adapter.get("interpretation", {})
    action_checks = action_adapter.get("checks", {})
    math_checks = math_parity.get("checks", {})
    math_interpretation = math_parity.get("interpretation", {})
    same_state_checks = same_state_parity.get("checks", {})
    same_state_interpretation = same_state_parity.get("interpretation", {})
    runtime_checks = runtime_parity.get("checks", {})
    runtime_interpretation = runtime_parity.get("interpretation", {})
    torso_checks = torso_frame_offset.get("checks", {})
    torso_interpretation = torso_frame_offset.get("interpretation", {})
    sample_checks = sample.get("checks", {})
    native_probe_builds_obs = "def build_obs(" in native_probe and "if obs.shape != (160,)" in native_probe
    native_probe_declares_approx = "approximate 160-D observation" in native_probe
    return {
        "official_schema_json_available": bool(schema),
        "policy_observation_dim_160": dim_sum == 160 == int(schema.get("metrics", {}).get("policy_dimension", -1)),
        "policy_observation_order_8_terms": [row["term"] for row in term_rows]
        == [
            "command",
            "motion_anchor_pos_b",
            "motion_anchor_ori_b",
            "base_lin_vel",
            "base_ang_vel",
            "joint_pos",
            "joint_vel",
            "actions",
        ],
        "official_policy_corruption_enabled": "self.enable_corruption = True" in tracking_cfg,
        "official_empirical_normalization_enabled": "empirical_normalization = True" in g1_ppo_cfg,
        "checkpoint_obs_normalizer_present": bool(normalizer.get("obs_norm_state_dict_present")),
        "checkpoint_obs_norm_mean_trailing_dim_160": bool(normalizer.get("obs_norm__mean_trailing_dim_160")),
        "checkpoint_obs_norm_std_trailing_dim_160": bool(normalizer.get("obs_norm__std_trailing_dim_160")),
        "checkpoint_actor_input_dim_160": bool(normalizer.get("actor_input_dim_160")),
        "checkpoint_actor_output_dim_29": bool(normalizer.get("actor_output_dim_29")),
        "official_exporter_wraps_normalizer": has("official_exporter", "self.actor(self.normalizer(x))")
        and has("isaaclab_exporter", "self.normalizer = copy.deepcopy(normalizer)"),
        "motion_command_alignment_formula_available": has(
            "official_commands",
            "delta_ori_w = yaw_quat",
            "self.body_pos_relative_w = delta_pos_w + quat_apply",
        ),
        "official_reference_command_is_joint_pos_vel": has(
            "official_commands", "return torch.cat([self.joint_pos, self.joint_vel], dim=1)"
        ),
        "official_reanchoring_uses_robot_anchor_xy_and_reference_z": has(
            "official_commands",
            "delta_pos_w = robot_anchor_pos_w_repeat",
            "delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]",
        ),
        "deployment_world_to_init_alignment_available": has(
            "controller_motion_command_cpp", "worldToInit_ = worldToAnchor * initToAnchor.inverse()"
        ),
        "deployment_local_frame_observations_available": has(
            "controller_motion_command_cpp",
            "getAnchorPositionLocal",
            "getAnchorOrientationLocal",
            "getRobotBodyPositionLocal",
            "getRobotBodyOrientationLocal",
        ),
        "deployment_motion_policy_outputs_reference_arrays": has(
            "controller_motion_policy_cpp",
            'name2Index_.at("time_step")',
            'name2Index_.at("joint_pos")',
            'name2Index_.at("body_pos_w")',
        ),
        "native_probe_builds_160d_obs": native_probe_builds_obs,
        "native_probe_uses_obs_normalizer": "obs_norm_state_dict" in native_probe and "self.obs_mean" in native_probe,
        "native_probe_declared_approximate": native_probe_declares_approx,
        "dimension_only_probe_is_explicitly_not_success": bool(native_probe_builds_obs and native_probe_declares_approx),
        "native_adapter_validated_against_isaaclab_observation_manager": False,
        "native_adapter_validated_against_deployment_controller": False,
        "native_adapter_all_terms_numerically_validated": False,
        "native_adapter_has_no_root_assist_rollout_success": bool(
            control_contract.get("checks", {}).get("mujoco_video_has_no_root_assist")
            and control_contract.get("checks", {}).get("mujoco_video_uses_native_policy_action_semantics")
        ),
        "native_action_adapter_formula_ready": bool(action_interpretation.get("formula_adapter_ready")),
        "native_action_adapter_rollout_ready": bool(action_interpretation.get("rollout_adapter_ready")),
        "native_action_adapter_ctrlrange_allows_rollout": bool(action_checks.get("unit_targets_inside_mujoco_ctrlrange")),
        "native_action_adapter_ctrlrange_warning_cleared": not bool(
            action_interpretation.get("mujoco_ctrlrange_warning_blocks_final_rollout_claim")
        ),
        "observation_math_fixture_available": bool(math_parity),
        "observation_math_fixture_formulas_pass": bool(math_checks.get("all_formula_fixtures_pass")),
        "observation_math_fixture_runtime_still_blocked": not bool(
            math_interpretation.get("runtime_observation_manager_parity_ready")
        ),
        "observation_math_fixture_does_not_claim_success": bool(math_checks.get("does_not_claim_rollout_or_success")),
        "same_state_observation_parity_available": bool(same_state_parity),
        "same_state_observation_formula_slices_pass": bool(
            same_state_checks.get("all_same_state_formula_slices_pass")
        ),
        "same_state_observation_uses_noise_free_critic_reference": bool(
            same_state_checks.get("uses_noise_free_critic_reference")
        ),
        "same_state_observation_runtime_builder_still_blocked": not bool(
            same_state_interpretation.get("mujoco_runtime_observation_builder_ready")
        ),
        "same_state_observation_does_not_claim_runtime_rollout_success": bool(
            same_state_checks.get("does_not_claim_mujoco_runtime_or_rollout_success")
        ),
        "runtime_observation_parity_available": bool(runtime_parity),
        "runtime_observation_builder_executed": bool(runtime_checks.get("mujoco_runtime_builder_executed")),
        "runtime_observation_all_slices_pass": bool(runtime_checks.get("all_runtime_observation_slices_pass")),
        "runtime_observation_anchor_pose_matches_isaaclab": bool(
            runtime_checks.get("mujoco_anchor_pose_matches_isaaclab_sample")
        ),
        "runtime_observation_any_candidate_model_anchor_frame_matches": bool(
            runtime_checks.get("candidate_mujoco_models_any_anchor_orientation_matches_isaaclab")
        ),
        "runtime_observation_does_not_claim_rollout_success": bool(
            runtime_checks.get("does_not_claim_rollout_or_success")
        ),
        "runtime_observation_policy_rollout_still_blocked": not bool(
            runtime_interpretation.get("native_mujoco_policy_rollout_allowed")
        ),
        "torso_frame_offset_audit_available": bool(torso_frame_offset),
        "torso_frame_offset_hypothesis_supported": bool(
            torso_interpretation.get("torso_frame_offset_hypothesis_supported")
        ),
        "torso_frame_offset_restores_primary_anchor_terms": bool(
            torso_interpretation.get("primary_offset_restores_anchor_observation_terms")
        ),
        "torso_frame_offset_sample_quality_blocks_patch": bool(
            torso_interpretation.get("sample_quality_blocks_patch_claim")
        ),
        "torso_frame_offset_not_used_as_patch_yet": not bool(
            torso_interpretation.get("native_obs_adapter_patch_allowed")
        ),
        "torso_frame_offset_independent_walk_validation_pending": not bool(
            torso_checks.get("offset_validated_across_independent_walk_sample")
        ),
        "isaaclab_observation_sample_gate_available": bool(sample_gate),
        "isaaclab_observation_sample_captured": sample.get("status") == "ok_isaaclab_observation_manager_sample_captured",
        "isaaclab_observation_sample_policy_dim_160": bool(sample_checks.get("policy_obs_dim_160")),
        "isaaclab_observation_sample_policy_terms_expected_order": bool(
            sample_checks.get("policy_terms_expected_order")
        ),
        "isaaclab_observation_sample_critic_shared_terms_available": bool(
            sample_checks.get("critic_shared_terms_available")
        ),
        "isaaclab_observation_sample_raw_state_available_for_same_state_parity": bool(
            sample_checks.get("raw_state_available_for_same_state_parity")
        ),
        "isaaclab_observation_sample_has_motion_time_steps": isinstance(sample.get("motion_time_steps"), list)
        and len(sample.get("motion_time_steps")) == 1,
        "isaaclab_observation_sample_does_not_claim_mujoco_parity": bool(
            sample_checks.get("does_not_claim_mujoco_parity_or_rollout")
        ),
        "native_rollout_preconditions_ready": False,
        "does_not_claim_rollout_or_success": True,
    }


def write_outputs(summary: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with TSV_OUT.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "order",
            "term",
            "dimension",
            "slice_start",
            "slice_end",
            "function",
            "has_training_noise",
            "noise_min",
            "noise_max",
            "implementation_status",
            "mujoco_source_needed",
            "official_semantics",
            "required_validation",
            "failure_mode",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary["term_rows"])

    failed = [key for key, ok in summary["checks"].items() if not ok]
    md = [
        "# MuJoCo Native Observation Adapter Contract",
        "",
        f"- Status: `{summary['status']}`",
        f"- Generated: `{summary['generated_at']}`",
        "- Scope: official 160-D observation contract and native MuJoCo reconstruction gate; no physics rollout.",
        "- 结论：当前不能把任意 160 维拼接 obs 喂给 IsaacLab PPO actor 后声称 MuJoCo native policy rollout 成功。",
        "- 当前不得声称完整复现 BeyondMimic；本审计只给出后续修 native obs/action adapter 的逐项合同。",
        "",
        "## Failed / Blocking Checks",
        "",
    ]
    md.extend(f"- `{item}`" for item in failed)
    md.extend(["", "## Policy Observation Layout", ""])
    for row in summary["term_rows"]:
        md.append(
            f"- `{row['slice_start']}:{row['slice_end']}` `{row['term']}` "
            f"({row['dimension']}D): {row['official_semantics']} | status=`{row['implementation_status']}`"
        )
    md.extend(["", "## Normalizer Gate", ""])
    norm = summary["checkpoint_normalizer"]
    md.append(f"- Checkpoint: `{norm.get('checkpoint')}`")
    md.append(f"- `obs_norm_state_dict_present`: `{norm.get('obs_norm_state_dict_present')}`")
    md.append(f"- `obs_norm__mean_shape`: `{norm.get('obs_norm__mean_shape')}`")
    md.append(f"- `obs_norm__std_shape`: `{norm.get('obs_norm__std_shape')}`")
    md.append(f"- `obs_norm__mean_trailing_dim_160`: `{norm.get('obs_norm__mean_trailing_dim_160')}`")
    md.append(f"- `obs_norm__std_trailing_dim_160`: `{norm.get('obs_norm__std_trailing_dim_160')}`")
    md.append(f"- `actor_input_dim_160`: `{norm.get('actor_input_dim_160')}`")
    md.append(f"- `actor_output_dim_29`: `{norm.get('actor_output_dim_29')}`")
    md.extend(["", "## Same-State Formula Parity", ""])
    same_state = summary.get("same_state_observation_parity", {})
    md.append(f"- Status: `{same_state.get('status')}`")
    md.append(f"- Claim level: `{same_state.get('claim_level')}`")
    md.append(
        "- 解释：这里比较的是同一个 IsaacLab captured state 下，本地 NumPy 公式重算值 "
        "vs. official critic/noise-free shared observation terms；它不是 MuJoCo runtime rollout。"
    )
    for row in same_state.get("terms", []):
        md.append(
            f"- `{row['term']}` dim={row['dimension']} max_abs_error={float(row['max_abs_error']):.6e} "
            f"passed=`{row['passed']}`"
        )
    md.extend(["", "## MuJoCo Runtime Injected-State Parity", ""])
    runtime = summary.get("runtime_observation_parity", {})
    md.append(f"- Status: `{runtime.get('status')}`")
    md.append(f"- Claim level: `{runtime.get('claim_level')}`")
    md.append(
        "- 解释：这里加载 MuJoCo G1 XML，把 IsaacLab captured root/joint/qvel 状态注入 MuJoCo，"
        "执行 `mj_forward` 后再构造 160-D observation；它仍不是 policy rollout。"
    )
    for row in runtime.get("terms", []):
        md.append(
            f"- `{row['term']}` dim={row['dimension']} max_abs_error={float(row['max_abs_error']):.6e} "
            f"passed=`{row['passed']}`"
        )
    diagnostics = runtime.get("diagnostics", {})
    anchor_error = diagnostics.get("anchor_world_pose_error") or {}
    md.append(
        "- Anchor frame diagnostic: "
        f"position_m=`{anchor_error.get('position_m')}`, "
        f"quat_sign_invariant=`{anchor_error.get('quat_sign_invariant')}`"
    )
    candidates = diagnostics.get("candidate_model_frame_errors") or []
    if candidates:
        md.append("- Candidate MJCF torso frame errors:")
        for item in candidates:
            model_xml = str(item.get("model_xml", "")).replace(str(ROOT) + "/", "")
            md.append(
                f"  - `{model_xml}` loaded=`{item.get('loaded')}` "
                f"torso_quat_err=`{item.get('torso_quat_sign_invariant_error')}` "
                f"torso_pos_err=`{item.get('torso_position_error_m')}`"
            )
    md.extend(["", "## MuJoCo Torso Frame Offset Hypothesis", ""])
    torso = summary.get("torso_frame_offset", {})
    primary_torso = torso.get("primary_model_result", {})
    md.append(f"- Status: `{torso.get('status')}`")
    md.append(f"- Claim level: `{torso.get('claim_level')}`")
    md.append(f"- Raw anchor pos/orient error: `{primary_torso.get('raw_anchor_pos_b_error')}` / `{primary_torso.get('raw_anchor_ori_b_error')}`")
    md.append(
        f"- Corrected anchor pos/orient error: `{primary_torso.get('corrected_anchor_pos_b_error')}` / "
        f"`{primary_torso.get('corrected_anchor_ori_b_error')}`"
    )
    md.append(f"- Candidate quaternion offset: `{primary_torso.get('q_offset_right_mujoco_to_isaac')}`")
    md.append(f"- Sample terminated after zero step: `{torso.get('sample', {}).get('terminated_after_zero_step')}`")
    md.append("- 解释：该 offset 只支持 frame-mismatch 假设；因为样本已 terminated，不能直接作为 rollout adapter 修复。")
    md.extend(["", "## Runtime Validation Matrix", ""])
    for row in summary["validation_matrix"]:
        md.append(
            f"- `{row['term']}` `{row['slice']}`: isaaclab=`{row['validated_against_isaaclab_observation_manager']}`, "
            f"deployment=`{row['validated_against_motion_tracking_controller']}`, "
            f"ready=`{row['ready_for_native_mujoco_policy_rollout']}`. {row['why_blocking']}"
        )
    md.extend(["", "## Required Next Implementation Steps", ""])
    md.extend(f"- {item}" for item in summary["required_next_steps"])
    md.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "Passing this audit in the future would only clear the observation-adapter precondition. It would still not prove teacher quality, VAE reconstruction quality, diffusion closed-loop stability, guided task success, true Isaac rendering, or real-robot deployment.",
            "",
        ]
    )
    MD_OUT.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    schema = read_json(FILES["schema_audit"])
    term_rows = build_term_rows(schema)
    validation_matrix = build_validation_matrix(term_rows)
    normalizer = checkpoint_normalizer_summary()
    math_parity = read_json(FILES["mujoco_observation_math_parity"])
    same_state_parity = read_json(FILES["mujoco_observation_same_state_parity"])
    runtime_parity = read_json(FILES["mujoco_observation_runtime_parity"])
    torso_frame_offset = read_json(FILES["mujoco_torso_frame_offset"])
    sample_gate = read_json(FILES["isaaclab_observation_sample_gate"])
    sample = read_json(FILES["isaaclab_observation_sample"])
    checks = build_checks(
        schema,
        term_rows,
        normalizer,
        math_parity,
        same_state_parity,
        runtime_parity,
        torso_frame_offset,
        sample_gate,
        sample,
    )
    summary = {
        "status": "blocked_native_mujoco_observation_adapter_not_validated",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "mujoco_native_observation_adapter_contract",
        "claim_level": "contract audit only; no native MuJoCo PPO/VAE/diffusion rollout claim",
        "scope": (
            "Audits the exact 160-D IsaacLab policy observation contract, the official/deployment frame semantics, "
            "and the empirical-normalizer requirement before using IsaacLab PPO checkpoints in native MuJoCo."
        ),
        "files": {key: str(path) for key, path in FILES.items()},
        "paper_and_official_contract": {
            "policy_obs_order": [
                "command",
                "motion_anchor_pos_b",
                "motion_anchor_ori_b",
                "base_lin_vel",
                "base_ang_vel",
                "joint_pos_rel",
                "joint_vel_rel",
                "last_action",
            ],
            "policy_obs_dim": 160,
            "action_dim": 29,
            "actor_hidden_dims": [512, 256, 128],
            "empirical_normalization_required": True,
            "training_observation_corruption_required": True,
            "motion_reanchoring": (
                "MotionCommand increments continuous time_steps, resets via adaptive sampling, then aligns desired "
                "body poses by robot anchor x/y, reference anchor z, and yaw_quat(robot_anchor * inv(reference_anchor))."
            ),
        },
        "source_line_refs": {
            "policy_obs_terms": line_ref(FILES["official_tracking_cfg"], "class PolicyCfg"),
            "policy_corruption": line_ref(FILES["official_tracking_cfg"], "self.enable_corruption = True"),
            "empirical_normalization": line_ref(FILES["official_g1_ppo_cfg"], "empirical_normalization = True"),
            "motion_anchor_pos_b": line_ref(FILES["official_observations"], "def motion_anchor_pos_b"),
            "motion_anchor_ori_b": line_ref(FILES["official_observations"], "def motion_anchor_ori_b"),
            "command_joint_pos_vel": line_ref(FILES["official_commands"], "return torch.cat([self.joint_pos, self.joint_vel]"),
            "motion_relative_alignment": line_ref(FILES["official_commands"], "delta_ori_w = yaw_quat"),
            "onnx_exporter_normalizer": line_ref(FILES["official_exporter"], "self.actor(self.normalizer(x))"),
            "deployment_world_to_init": line_ref(FILES["controller_motion_command_cpp"], "worldToInit_ = worldToAnchor"),
        },
        "term_rows": term_rows,
        "validation_matrix": validation_matrix,
        "checkpoint_normalizer": normalizer,
        "observation_math_parity": {
            "status": math_parity.get("status"),
            "json": str(FILES["mujoco_observation_math_parity"]),
            "claim_level": math_parity.get("claim_level"),
            "max_abs_error": math_parity.get("fixture", {}).get("max_abs_error", {}),
            "failed_checks": [key for key, ok in math_parity.get("checks", {}).items() if not ok],
        },
        "same_state_observation_parity": {
            "status": same_state_parity.get("status"),
            "json": str(FILES["mujoco_observation_same_state_parity"]),
            "claim_level": same_state_parity.get("claim_level"),
            "terms": same_state_parity.get("terms", []),
            "failed_checks": [key for key, ok in same_state_parity.get("checks", {}).items() if not ok],
        },
        "runtime_observation_parity": {
            "status": runtime_parity.get("status"),
            "json": str(FILES["mujoco_observation_runtime_parity"]),
            "claim_level": runtime_parity.get("claim_level"),
            "terms": runtime_parity.get("terms", []),
            "diagnostics": {
                "anchor_world_pose_error": runtime_parity.get("diagnostics", {}).get("anchor_world_pose_error"),
                "base_velocity_from_qvel_vs_raw_body_frame": runtime_parity.get("diagnostics", {}).get(
                    "base_velocity_from_qvel_vs_raw_body_frame"
                ),
                "motion_anchor_from_file_vs_sample": runtime_parity.get("diagnostics", {}).get(
                    "motion_anchor_from_file_vs_sample"
                ),
                "candidate_model_frame_errors": runtime_parity.get("diagnostics", {}).get(
                    "candidate_model_frame_errors", []
                ),
            },
            "failed_checks": [key for key, ok in runtime_parity.get("checks", {}).items() if not ok],
        },
        "torso_frame_offset": {
            "status": torso_frame_offset.get("status"),
            "json": str(FILES["mujoco_torso_frame_offset"]),
            "claim_level": torso_frame_offset.get("claim_level"),
            "sample": torso_frame_offset.get("sample"),
            "primary_model_result": torso_frame_offset.get("primary_model_result"),
            "failed_checks": [key for key, ok in torso_frame_offset.get("checks", {}).items() if not ok],
            "interpretation": torso_frame_offset.get("interpretation"),
        },
        "isaaclab_observation_sample_gate": {
            "status": sample_gate.get("status"),
            "json": str(FILES["isaaclab_observation_sample_gate"]),
            "sample_json": str(FILES["isaaclab_observation_sample"]),
            "claim_level": sample_gate.get("claim_level"),
            "policy_obs_dim": sample.get("policy_obs_dim"),
            "policy_term_names": sample.get("policy_term_names"),
            "policy_term_dims": sample.get("policy_term_dims"),
            "critic_term_names": sample.get("critic_term_names"),
            "critic_term_dims": sample.get("critic_term_dims"),
            "motion_time_steps": sample.get("motion_time_steps"),
            "body_indexes": sample.get("body_indexes"),
            "failed_checks": [key for key, ok in sample_gate.get("checks", {}).items() if not ok],
        },
        "checks": checks,
        "hard_blockers": [
            "native_adapter_validated_against_isaaclab_observation_manager is false",
            "native_adapter_validated_against_deployment_controller is false",
            "native_adapter_all_terms_numerically_validated is false",
            "native_adapter_has_no_root_assist_rollout_success is false",
            "MuJoCo injected-state runtime parity fails at torso_link/motion_anchor pose frame mismatch",
            "torso frame offset hypothesis is currently based on one terminated dance sample and must be validated on a non-terminated walk sample",
            "current MuJoCo PPO probe is explicitly approximate, not an official observation-manager match",
            "empirical_normalization must be preserved for any direct actor checkpoint inference",
            "dimension-correct 160-D observation is not sufficient evidence of semantic correctness",
        ],
        "required_next_steps": [
            "Export an official motion policy ONNX with metadata and embedded normalizer, or load the checkpoint obs normalizer exactly.",
            "Implement a native MuJoCo observation builder that returns the exact eight policy terms and slices in this audit.",
            "Validate that builder numerically against the captured IsaacLab observation_manager sample for the same reset state, motion time_steps, and last_action.",
            "Resolve the MuJoCo MJCF versus IsaacLab USD/URDF torso_link frame mismatch before feeding native MuJoCo observations to the actor.",
            "Capture a non-terminated low-dynamic walk observation_manager sample and verify whether the same MuJoCo-to-IsaacLab torso frame offset restores anchor terms.",
            "Validate frame-alignment semantics against motion_tracking_controller worldToInit_/Pinocchio local-frame formulas.",
            "Validate body-frame base velocity, Rot6D column ordering, default_joint_pos source, and previous-action semantics with finite numeric fixtures.",
            "Use the no-action-clipping MuJoCo actuator XML from the action adapter audit for any later no-root-assist policy videos.",
            "Combine the validated obs builder with the native action adapter fixture, disable root assist, and log raw/clipped normalized actions plus PD setpoints.",
            "Only after the above gates pass should a native MuJoCo PPO rollout video be treated as motion-control evidence.",
        ],
        "interpretation": {
            "native_obs_adapter_ready": False,
            "native_action_adapter_formula_ready": checks["native_action_adapter_formula_ready"],
            "native_action_adapter_rollout_ready": checks["native_action_adapter_rollout_ready"],
            "observation_formula_math_parity_ready": checks["observation_math_fixture_formulas_pass"],
            "same_state_observation_formula_parity_ready": checks["same_state_observation_formula_slices_pass"],
            "runtime_observation_parity_available": checks["runtime_observation_parity_available"],
            "runtime_observation_builder_executed": checks["runtime_observation_builder_executed"],
            "runtime_observation_all_slices_pass": checks["runtime_observation_all_slices_pass"],
            "runtime_anchor_frame_mismatch_detected": not checks["runtime_observation_anchor_pose_matches_isaaclab"],
            "torso_frame_offset_hypothesis_supported": checks["torso_frame_offset_hypothesis_supported"],
            "torso_frame_offset_restores_primary_anchor_terms": checks[
                "torso_frame_offset_restores_primary_anchor_terms"
            ],
            "torso_frame_offset_requires_independent_walk_validation": checks[
                "torso_frame_offset_independent_walk_validation_pending"
            ],
            "isaaclab_observation_sample_available": checks["isaaclab_observation_sample_captured"],
            "observation_runtime_parity_ready": False,
            "native_rollout_preconditions_ready": checks["native_rollout_preconditions_ready"],
            "normalizer_required": checks["official_empirical_normalization_enabled"],
            "normalizer_available_in_best_checkpoint": checks["checkpoint_obs_normalizer_present"],
            "normalizer_shape_ready": checks["checkpoint_obs_norm_mean_trailing_dim_160"]
            and checks["checkpoint_obs_norm_std_trailing_dim_160"],
            "success_video_claim_allowed": False,
            "why_current_videos_can_fail_even_if_actor_checkpoint_loads": (
                "The actor was trained on normalized IsaacLab observations with exact command/frame semantics. "
                "A MuJoCo-side approximate 160-D vector can be dimension-correct but semantically wrong, producing "
                "front-leaning or collapsed postures instead of lifted-leg/walk control."
            ),
        },
    }
    write_outputs(summary)
    print(json.dumps({"status": summary["status"], "json": str(JSON_OUT), "md": str(MD_OUT), "tsv": str(TSV_OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
