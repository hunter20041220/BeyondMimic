#!/usr/bin/env python3
"""Audit the local teacher/VAE/diffusion/video chain against the paper contract.

This script is intentionally conservative.  It does not try to rescue weak
videos by changing their wording; it records whether each local component
matches the equations and data flow described in BeyondMimic.  A failed audit
means downstream videos must remain diagnostic artifacts until a teacher and
model chain pass the quality gates.
"""

from __future__ import annotations

import json
import os
import csv
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/audits/model_chain_paper_contract_audit"
JSON_OUT = OUT / "beyondmimic_model_chain_paper_contract_audit.json"
MD_OUT = OUT / "beyondmimic_model_chain_paper_contract_audit.md"
TSV_OUT = OUT / "beyondmimic_model_chain_paper_contract_audit.tsv"


FILES = {
    "paper_method": ROOT / "reproduction/paper/source/tex/method.tex",
    "paper_supplement": ROOT / "reproduction/paper/source/root.tex",
    "official_tracking_cfg": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/tracking_env_cfg.py",
    "official_g1_flat_cfg": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py",
    "official_rewards": ROOT
    / "reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/"
    "whole_body_tracking/tasks/tracking/mdp/rewards.py",
    "resource_vae": ROOT / "reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py",
    "resource_diffusion": ROOT / "reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py",
    "paper_contract_stage1_audit": ROOT
    / "res/tracking/stage1_tracking_parameter_contract_audit/"
    "stage1_tracking_parameter_contract_audit.json",
    "paper_contract_vae_script": ROOT / "reproduction/scripts/level_c_paper_contract_teacher_rollout_vae_training.py",
    "paper_contract_vae_json": ROOT
    / "res/level_c/paper_contract_teacher_rollout_vae_training/"
    "level_c_paper_contract_teacher_rollout_vae_training.json",
    "paper_contract_state_latent_dataset_json": ROOT
    / "res/level_c/official_importer_export_paper_contract_teacher_rollout_state_latent_dataset/"
    "level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.json",
    "paper_contract_diffusion_json": ROOT
    / "res/level_c/official_importer_export_paper_contract_state_latent_diffusion_training/"
    "level_c_official_importer_export_paper_contract_state_latent_diffusion_training.json",
    "paper_contract_transformer_diffusion_json": ROOT
    / "res/level_c/paper_contract_transformer_state_latent_diffusion_training/"
    "paper_contract_transformer_state_latent_diffusion_training.json",
    "paper_contract_guidance_json": ROOT
    / "res/level_c/official_importer_export_paper_contract_state_latent_guidance_eval/"
    "level_c_official_importer_export_paper_contract_state_latent_guidance_eval.json",
    "mujoco_control_contract_audit_json": ROOT
    / "res/audits/mujoco_control_contract_audit/"
    "mujoco_control_contract_audit.json",
    "mujoco_native_action_adapter_json": ROOT
    / "res/audits/mujoco_native_action_adapter_contract/"
    "mujoco_native_action_adapter_contract.json",
    "mujoco_native_observation_adapter_json": ROOT
    / "res/audits/mujoco_native_observation_adapter_contract/"
    "mujoco_native_observation_adapter_contract.json",
    "paper_arch_train": ROOT / "reproduction/scripts/train_lafan1_paper_level_vae_diffusion.py",
    "clean_walk_video": ROOT / "reproduction/scripts/render_clean_walk_mujoco_control_suite.py",
    "mujoco_pd_video": ROOT / "mujoco_mp4/scripts/mujoco_pd_control_video.py",
    "singleleg_train_json": ROOT
    / "res/tracking/hub_singleleg_paper_contract_ppo_training_run/"
    "tracking_hub_singleleg_paper_contract_ppo_training_run.json",
    "singleleg_eval_json": ROOT
    / "res/tracking/hub_singleleg_paper_contract_ppo_checkpoint_eval/"
    "tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.json",
    "singleleg_train_log": ROOT
    / "logs/tracking_hub_singleleg_paper_contract_ppo_training_run/"
    "tracking_g1_resource_adjusted_ppo_training_run.log",
}


FAILED_VIDEO_DIRS = [
    ROOT / "res/visualization/hub_singleleg_full_chain_scaled_ppo_pure",
    ROOT / "res/visualization/final_clean_walk_six_mujoco_videos_scaled_ppo_pure",
    ROOT / "res/visualization/clean_walk_mujoco_control_suite",
    ROOT / "res/visualization/clean_walk_mujoco_control_suite_sweep",
    ROOT / "res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos",
]


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def contains(text: str, *patterns: str) -> bool:
    return all(pattern in text for pattern in patterns)


def rg_line(path: Path, pattern: str) -> str | None:
    if not path.is_file():
        return None
    for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if pattern in line:
            return f"{path}:{idx}"
    return None


def shell_json(args: list[str]) -> list[dict[str, Any]]:
    proc = subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    rows: list[dict[str, Any]] = []
    if proc.returncode != 0:
        return [{"error": proc.stderr.strip() or proc.stdout.strip()}]
    for line in proc.stdout.strip().splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) >= 6 and parts[0].isdigit():
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": int(float(parts[2])),
                    "memory_total_mb": int(float(parts[3])),
                    "utilization_gpu_percent": int(float(parts[4])),
                    "power_draw_w": float(parts[5]),
                }
            )
    return rows


def latest_training_metrics_from_log(path: Path) -> dict[str, Any]:
    text = read_text(path)
    if not text:
        return {"log_exists": False}
    tail = "\n".join(text.splitlines()[-220:])
    patterns = {
        "iteration": r"Learning iteration\s+(\d+)/(\d+)",
        "mean_reward": r"Mean reward:\s+([-+0-9.eE]+)",
        "mean_episode_length": r"Mean episode length:\s+([-+0-9.eE]+)",
        "error_anchor_pos": r"Metrics/motion/error_anchor_pos:\s+([-+0-9.eE]+)",
        "error_body_pos": r"Metrics/motion/error_body_pos:\s+([-+0-9.eE]+)",
        "error_joint_pos": r"Metrics/motion/error_joint_pos:\s+([-+0-9.eE]+)",
        "termination_anchor_pos": r"Episode_Termination/anchor_pos:\s+([-+0-9.eE]+)",
        "termination_ee_body_pos": r"Episode_Termination/ee_body_pos:\s+([-+0-9.eE]+)",
        "eta": r"ETA:\s+([0-9:]+)",
    }
    out: dict[str, Any] = {"log_exists": True}
    for key, pattern in patterns.items():
        matches = re.findall(pattern, tail)
        if not matches:
            continue
        value = matches[-1]
        if key == "iteration":
            out["iteration"] = int(value[0])
            out["max_iterations"] = int(value[1])
        elif key == "eta":
            out[key] = value
        else:
            out[key] = float(value)
    return out


def component_rows() -> list[dict[str, Any]]:
    paper = read_text(FILES["paper_method"])
    supp = read_text(FILES["paper_supplement"])
    official_cfg = read_text(FILES["official_tracking_cfg"])
    official_g1 = read_text(FILES["official_g1_flat_cfg"])
    official_rewards = read_text(FILES["official_rewards"])
    resource_vae = read_text(FILES["resource_vae"])
    resource_diffusion = read_text(FILES["resource_diffusion"])
    paper_contract_stage1 = read_json(FILES["paper_contract_stage1_audit"])
    paper_contract_vae_script = read_text(FILES["paper_contract_vae_script"])
    paper_contract_vae = read_json(FILES["paper_contract_vae_json"])
    paper_contract_state_latent = read_json(FILES["paper_contract_state_latent_dataset_json"])
    paper_contract_diffusion = read_json(FILES["paper_contract_diffusion_json"])
    paper_contract_transformer_diffusion = read_json(FILES["paper_contract_transformer_diffusion_json"])
    paper_contract_guidance = read_json(FILES["paper_contract_guidance_json"])
    mujoco_control_contract = read_json(FILES["mujoco_control_contract_audit_json"])
    mujoco_native_action_adapter = read_json(FILES["mujoco_native_action_adapter_json"])
    mujoco_native_observation_adapter = read_json(FILES["mujoco_native_observation_adapter_json"])
    paper_arch = read_text(FILES["paper_arch_train"])
    video = read_text(FILES["clean_walk_video"])
    mujoco_pd = read_text(FILES["mujoco_pd_video"])

    return [
        {
            "component": "stage1_tracking_parameter_contract_gate",
            "status": paper_contract_stage1.get("status", "missing"),
            "matches_paper": paper_contract_stage1.get("status")
            in {"ok_stage1_tracking_parameter_contract_audited", "blocked_stage1_teacher_contract_has_required_followups"},
            "evidence": [str(FILES["paper_contract_stage1_audit"])],
            "notes": (
                "Stage-1 formula/parameter contract is now separately audited. The public/official code mostly "
                "matches paper contracts, but the current teacher quality gate remains blocked, so downstream "
                "VAE/diffusion cannot be treated as final."
            ),
            "detected_patterns": {
                "stage1_contract_status": paper_contract_stage1.get("status"),
                "teacher_quality_blocked": paper_contract_stage1.get("status")
                == "blocked_stage1_teacher_contract_has_required_followups",
            },
        },
        {
            "component": "paper_contract_tracking_observation_action_reward",
            "status": "reference_contract_available",
            "matches_paper": True,
            "evidence": [
                rg_line(FILES["paper_method"], r"\\mathbf{o}=["),
                rg_line(FILES["paper_method"], r"\\boldsymbol{\\theta}^{\\text{sp}}"),
                rg_line(FILES["paper_method"], "r_{\\text{task}}"),
                rg_line(FILES["paper_supplement"], "Termination occurs"),
            ],
            "notes": (
                "The paper defines motion phase, anchor error, IMU/root twist, joint state, and last action; "
                "actions are normalized PD setpoints; rewards are exponential body tracking terms plus three regularizers."
            ),
        },
        {
            "component": "official_whole_body_tracking_stage1",
            "status": "mostly_matches_available_official_stage1_code",
            "matches_paper": bool(
                contains(official_cfg, "motion_global_anchor_pos", "motion_body_pos", "action_rate_l2", "joint_limit")
                and contains(official_g1, "G1_ACTION_SCALE", "anchor_body_name", "torso_link")
                and contains(official_rewards, "motion_relative_body_position_error_exp")
            ),
            "evidence": [
                str(FILES["official_tracking_cfg"]),
                str(FILES["official_g1_flat_cfg"]),
                str(FILES["official_rewards"]),
            ],
            "notes": (
                "Stage 1 should continue to use the official whole_body_tracking IsaacLab/RSL-RL task. "
                "The weak local videos are not evidence that the official Stage-1 formulation is wrong; "
                "they show the local teacher/checkpoint/data quality is not yet good enough."
            ),
        },
        {
            "component": "resource_adjusted_teacher_rollout_vae",
            "status": "fails_paper_vae_contract",
            "matches_paper": False,
            "evidence": [str(FILES["resource_vae"])],
            "notes": (
                "Local VAE encoder is obs+action -> latent and decoder is obs+latent -> action. "
                "The paper VAE encoder should encode reference motion intent E(psi, e_anchor), while the decoder "
                "combines z with proprioception. This mismatch can collapse learned outputs toward a generic posture."
            ),
            "detected_patterns": {
                "encoder_obs_action": "self.encoder(torch.cat([obs, action], dim=-1))" in resource_vae,
                "decoder_obs_latent": "self.decoder(torch.cat([obs, z], dim=-1))" in resource_vae,
                "no_reference_anchor_split": "psi" not in resource_vae and "e_anchor" not in resource_vae,
            },
        },
        {
            "component": "paper_contract_teacher_rollout_vae",
            "status": paper_contract_vae.get("status", "missing"),
            "matches_paper": bool(
                paper_contract_vae.get("checks", {}).get("encoder_uses_reference_intent_only")
                and paper_contract_vae.get("checks", {}).get("decoder_uses_proprioception_plus_latent")
                and paper_contract_vae.get("checks", {}).get("action_dim_29")
            ),
            "evidence": [str(FILES["paper_contract_vae_script"]), str(FILES["paper_contract_vae_json"])],
            "notes": (
                "This is the preferred local VAE route because it repairs the major formula-level interface: "
                "encoder uses reference intent terms and decoder uses proprioception plus latent. It still cannot "
                "be called paper-level because the source teacher rollout is local, not official DAgger, and the "
                "source teacher quality gate remains weak."
            ),
            "detected_patterns": {
                "encoder_terms_recorded": "encoder_input_terms" in paper_contract_vae_script,
                "encoder_uses_reference_intent_only": paper_contract_vae.get("checks", {}).get(
                    "encoder_uses_reference_intent_only"
                ),
                "decoder_uses_proprioception_plus_latent": paper_contract_vae.get("checks", {}).get(
                    "decoder_uses_proprioception_plus_latent"
                ),
                "source_teacher_done_rate_low_enough": paper_contract_vae.get("checks", {}).get(
                    "source_teacher_done_rate_low_enough_for_downstream"
                ),
                "official_dagger": paper_contract_vae.get("source_teacher_rollout", {}).get("official_dagger_dataset"),
            },
        },
        {
            "component": "resource_adjusted_state_latent_diffusion",
            "status": "fails_paper_diffusion_contract",
            "matches_paper": False,
            "evidence": [str(FILES["resource_diffusion"])],
            "notes": (
                "This path trains an MLP denoiser over policy_obs+latent windows. The paper uses a state-latent "
                "trajectory with hybrid character-yaw state, emphasis projection, individual state/latent denoising "
                "steps, and a Transformer denoiser. It is useful as a local diagnostic, not as the success video chain."
            ),
            "detected_patterns": {
                "mlp_denoiser": "class StateLatentDenoiser" in resource_diffusion
                and "nn.Sequential" in resource_diffusion,
                "uses_policy_obs_as_state": 'data["policy_obs"]' in resource_diffusion,
                "no_transformer_encoder": "TransformerEncoder" not in resource_diffusion,
            },
        },
        {
            "component": "paper_contract_state_latent_dataset",
            "status": paper_contract_state_latent.get("status", "missing"),
            "matches_paper": bool(
                paper_contract_state_latent.get("checks", {}).get("paper_contract_best_teacher_rollout_source")
                and paper_contract_state_latent.get("checks", {}).get("paper_contract_vae_source")
                and paper_contract_state_latent.get("checks", {}).get("has_train_validation_test_splits")
            ),
            "evidence": [str(FILES["paper_contract_state_latent_dataset_json"])],
            "notes": (
                "This dataset is the correct local route after paper-contract VAE, but its state source is still "
                "policy_obs rather than the full hybrid character-yaw state described in the paper, and it uses "
                "local teacher rollouts with many done events. It is a local contract candidate, not paper-level data."
            ),
            "detected_patterns": {
                "token_dim": paper_contract_state_latent.get("worker_summary", {}).get("dataset", {}).get("token_dim"),
                "obs_dim": paper_contract_state_latent.get("worker_summary", {}).get("dataset", {}).get("obs_dim"),
                "latent_dim": paper_contract_state_latent.get("worker_summary", {}).get("dataset", {}).get("latent_dim"),
                "done_counts": [
                    row.get("done_count")
                    for row in paper_contract_state_latent.get("worker_summary", {}).get("shards", [])
                ],
                "paper_level_state_latent_dataset": paper_contract_state_latent.get("paper_level", False),
            },
        },
        {
            "component": "paper_contract_state_latent_diffusion",
            "status": paper_contract_diffusion.get("status", "missing"),
            "matches_paper": bool(
                paper_contract_diffusion.get("checks", {}).get("paper_contract_state_latent_dataset_source")
                and paper_contract_diffusion.get("checks", {}).get("test_denoising_improves_over_noisy")
            ),
            "evidence": [str(FILES["paper_contract_diffusion_json"])],
            "notes": (
                "This is the current preferred local denoiser output because it improves denoising on a full "
                "paper-contract local window dataset. However the base implementation still uses the resource-adjusted "
                "MLP denoiser path, not the paper's 6-layer Transformer with 512-d embeddings and per-state/latent "
                "denoising embeddings. It therefore remains local diagnostic evidence."
            ),
            "detected_patterns": {
                "test_denoising_improvement_ratio": paper_contract_diffusion.get("worker_summary", {})
                .get("evaluation", {})
                .get("test", {})
                .get("denoising_improvement_ratio"),
                "denoising_steps": paper_contract_diffusion.get("worker_summary", {}).get("training", {}).get(
                    "denoising_steps"
                ),
                "hidden_dim": paper_contract_diffusion.get("worker_summary", {}).get("training", {}).get("hidden_dim"),
                "paper_level_diffusion": paper_contract_diffusion.get("checks", {}).get("does_not_claim_paper_level_diffusion")
                is False,
            },
        },
        {
            "component": "paper_contract_transformer_state_latent_diffusion_code_contract",
            "status": paper_contract_transformer_diffusion.get("status", "missing"),
            "matches_paper": bool(
                paper_contract_transformer_diffusion.get("checks", {}).get("paper_contract_architecture_checks_pass")
                and paper_contract_transformer_diffusion.get("worker_summary", {})
                .get("checks", {})
                .get("uses_transformer_encoder")
                and paper_contract_transformer_diffusion.get("worker_summary", {})
                .get("checks", {})
                .get("uses_individual_state_and_latent_denoising_steps")
                and paper_contract_transformer_diffusion.get("worker_summary", {})
                .get("checks", {})
                .get("forward_backward_ok")
            ),
            "evidence": [str(FILES["paper_contract_transformer_diffusion_json"])],
            "notes": (
                "This is the corrected local code-contract route for the paper-style state-latent diffusion model: "
                "6-layer Transformer, 512-d embeddings, 8 attention heads, 20 denoising steps, separate state/latent "
                "denoising-step embeddings, and clean-trajectory prediction. It has only been dry-run tested on a tiny "
                "local subset, so it proves architecture/gradient viability but not full training quality or closed-loop control."
            ),
            "detected_patterns": {
                "dry_run": paper_contract_transformer_diffusion.get("dry_run"),
                "parameter_count": paper_contract_transformer_diffusion.get("worker_summary", {})
                .get("architecture", {})
                .get("parameter_count"),
                "embedding_dim": paper_contract_transformer_diffusion.get("worker_summary", {})
                .get("architecture", {})
                .get("embedding_dim"),
                "attention_heads": paper_contract_transformer_diffusion.get("worker_summary", {})
                .get("architecture", {})
                .get("attention_heads"),
                "transformer_layers": paper_contract_transformer_diffusion.get("worker_summary", {})
                .get("architecture", {})
                .get("transformer_layers"),
                "test_denoising_improvement_ratio_after_one_dry_step": paper_contract_transformer_diffusion.get(
                    "worker_summary", {}
                )
                .get("evaluation", {})
                .get("test", {})
                .get("denoising_improvement_ratio"),
                "paper_level_diffusion": paper_contract_transformer_diffusion.get("interpretation", {}).get(
                    "paper_level_diffusion"
                ),
            },
        },
        {
            "component": "paper_contract_offline_guidance",
            "status": paper_contract_guidance.get("status", "missing"),
            "matches_paper": bool(
                paper_contract_guidance.get("checks", {}).get("all_best_guidance_gradients_nonzero")
                and paper_contract_guidance.get("checks", {}).get("all_best_costs_improve")
            ),
            "evidence": [str(FILES["paper_contract_guidance_json"])],
            "notes": (
                "The paper-contract guidance audit verifies differentiable cost gradients and cost reduction offline. "
                "It does not perform receding-horizon closed-loop MuJoCo/Isaac control, and its cost reductions are "
                "very small, so it cannot support Fig.5/Fig.6-style claims yet."
            ),
            "detected_patterns": {
                "task_count": paper_contract_guidance.get("worker_summary", {}).get("metrics", {}).get("task_count"),
                "row_count": paper_contract_guidance.get("worker_summary", {}).get("metrics", {}).get("row_count"),
                "all_best_costs_improve": paper_contract_guidance.get("checks", {}).get("all_best_costs_improve"),
                "closed_loop_rollout": not paper_contract_guidance.get("checks", {}).get(
                    "does_not_claim_closed_loop_rollout", True
                ),
            },
        },
        {
            "component": "mujoco_control_contract_gate",
            "status": mujoco_control_contract.get("status", "missing"),
            "matches_paper": bool(
                mujoco_control_contract.get("checks", {}).get("mujoco_video_uses_native_policy_action_semantics")
                and mujoco_control_contract.get("checks", {}).get("mujoco_video_has_no_root_assist")
                and mujoco_control_contract.get("checks", {}).get("mujoco_floor_material_matches_official_training")
            ),
            "evidence": [str(FILES["mujoco_control_contract_audit_json"])],
            "notes": (
                "The local MuJoCo videos have useful official G1 PD/action-scale numbers, but the current video "
                "adapter still uses absolute/IK joint targets, default root assist, and material/friction differences. "
                "It is therefore a diagnostic visualization route, not the native paper control path."
            ),
            "detected_patterns": {
                "pd_values_match": mujoco_control_contract.get("checks", {}).get(
                    "mujoco_pd_xml_parameter_patch_matches_official_numbers"
                ),
                "native_policy_action_semantics": mujoco_control_contract.get("checks", {}).get(
                    "mujoco_video_uses_native_policy_action_semantics"
                ),
                "root_assist_disabled": mujoco_control_contract.get("checks", {}).get(
                    "mujoco_video_has_no_root_assist"
                ),
                "floor_material_matches": mujoco_control_contract.get("checks", {}).get(
                    "mujoco_floor_material_matches_official_training"
                ),
            },
        },
        {
            "component": "mujoco_native_action_adapter_formula_gate",
            "status": mujoco_native_action_adapter.get("status", "missing"),
            "matches_paper": bool(
                mujoco_native_action_adapter.get("checks", {}).get("paper_formula_available")
                and mujoco_native_action_adapter.get("checks", {}).get(
                    "isaaclab_affine_joint_action_semantics_available"
                )
                and mujoco_native_action_adapter.get("checks", {}).get("official_action_scale_rows_29")
                and mujoco_native_action_adapter.get("checks", {}).get("zero_default_fallback_not_used")
                and mujoco_native_action_adapter.get("checks", {}).get("mujoco_mapping_order_matches_action_rows")
                and mujoco_native_action_adapter.get("checks", {}).get("pd_actuator_order_matches_action_rows")
                and mujoco_native_action_adapter.get("checks", {}).get("zero_action_returns_default_pose")
                and mujoco_native_action_adapter.get("checks", {}).get("unit_action_delta_matches_action_scale")
                and mujoco_native_action_adapter.get("checks", {}).get(
                    "negative_unit_action_delta_matches_action_scale"
                )
                and mujoco_native_action_adapter.get("checks", {}).get("large_action_clips_to_unit_scale")
            ),
            "evidence": [str(FILES["mujoco_native_action_adapter_json"])],
            "notes": (
                "The normalized-action-to-PD-setpoint formula gate is now available: theta_sp = theta_default + "
                "action_scale * clipped_action, with official joint order, deployment default pose, and action-scale rows. "
                "This is only a formula/order fixture; it does not prove native observations, physics stability, or video success."
            ),
            "detected_patterns": {
                "formula_adapter_ready": mujoco_native_action_adapter.get("interpretation", {}).get(
                    "formula_adapter_ready"
                ),
                "native_obs_adapter_ready": mujoco_native_action_adapter.get("interpretation", {}).get(
                    "native_obs_adapter_ready"
                ),
                "physics_rollout_performed": mujoco_native_action_adapter.get("interpretation", {}).get(
                    "physics_rollout_performed"
                ),
                "success_video_claim_allowed": mujoco_native_action_adapter.get("interpretation", {}).get(
                    "success_video_claim_allowed"
                ),
                "mujoco_ctrlrange_warning_blocks_final_rollout_claim": mujoco_native_action_adapter.get(
                    "interpretation", {}
                ).get("mujoco_ctrlrange_warning_blocks_final_rollout_claim"),
                "rollout_readiness_warnings": mujoco_native_action_adapter.get("rollout_readiness_warnings", []),
            },
        },
        {
            "component": "mujoco_native_observation_adapter_gate",
            "status": mujoco_native_observation_adapter.get("status", "missing"),
            "matches_paper": bool(
                mujoco_native_observation_adapter.get("checks", {}).get("policy_observation_dim_160")
                and mujoco_native_observation_adapter.get("checks", {}).get("policy_observation_order_8_terms")
                and mujoco_native_observation_adapter.get("checks", {}).get("checkpoint_obs_normalizer_present")
                and mujoco_native_observation_adapter.get("checks", {}).get(
                    "native_adapter_validated_against_isaaclab_observation_manager"
                )
                and mujoco_native_observation_adapter.get("checks", {}).get(
                    "native_adapter_validated_against_deployment_controller"
                )
                and mujoco_native_observation_adapter.get("checks", {}).get(
                    "native_adapter_has_no_root_assist_rollout_success"
                )
            ),
            "evidence": [str(FILES["mujoco_native_observation_adapter_json"])],
            "notes": (
                "The exact 160-D observation layout, empirical-normalizer requirement, and official deployment "
                "frame semantics are now enumerated. This gate remains blocked because the MuJoCo builder has not "
                "been numerically validated against IsaacLab observation_manager or motion_tracking_controller "
                "worldToInit_/Pinocchio local-frame semantics, and no no-root-assist native rollout has passed."
            ),
            "detected_patterns": {
                "policy_observation_dim_160": mujoco_native_observation_adapter.get("checks", {}).get(
                    "policy_observation_dim_160"
                ),
                "policy_observation_order_8_terms": mujoco_native_observation_adapter.get("checks", {}).get(
                    "policy_observation_order_8_terms"
                ),
                "official_empirical_normalization_enabled": mujoco_native_observation_adapter.get("checks", {}).get(
                    "official_empirical_normalization_enabled"
                ),
                "checkpoint_obs_normalizer_present": mujoco_native_observation_adapter.get("checks", {}).get(
                    "checkpoint_obs_normalizer_present"
                ),
                "native_probe_declared_approximate": mujoco_native_observation_adapter.get("checks", {}).get(
                    "native_probe_declared_approximate"
                ),
                "native_adapter_validated_against_isaaclab": mujoco_native_observation_adapter.get("checks", {}).get(
                    "native_adapter_validated_against_isaaclab_observation_manager"
                ),
                "native_adapter_validated_against_deployment": mujoco_native_observation_adapter.get("checks", {}).get(
                    "native_adapter_validated_against_deployment_controller"
                ),
                "success_video_claim_allowed": mujoco_native_observation_adapter.get("interpretation", {}).get(
                    "success_video_claim_allowed"
                ),
                "hard_blockers": mujoco_native_observation_adapter.get("hard_blockers", []),
            },
        },
        {
            "component": "lafan1_paper_arch_training",
            "status": "paper_architecture_public_data_approximation_not_full_paper_contract",
            "matches_paper": False,
            "evidence": [str(FILES["paper_arch_train"])],
            "notes": (
                "This path correctly uses a Transformer diffusion backbone, individual state/latent denoising steps, "
                "state projection, 20 diffusion steps, and paper-scale VAE/Transformer dimensions. However its VAE "
                "still encodes state+action rather than E(psi, e_anchor), and its dataset is public retargeted LAFAN1 "
                "reference data rather than teacher DAgger/VAE rollout trajectories."
            ),
            "detected_patterns": {
                "transformer_diffusion": "class DiffusionTransformer" in paper_arch and "nn.TransformerEncoder" in paper_arch,
                "individual_state_latent_steps": "state_step_embed" in paper_arch and "latent_step_embed" in paper_arch,
                "emphasis_projection": "emphasis_projection" in paper_arch,
                "vae_encoder_state_action": "self.encoder = mlp(cfg.state_dim + cfg.action_dim" in paper_arch,
                "dataset_public_lafan1": "CSV_ROOT" in paper_arch and "LAFAN1_Retargeting_Dataset" in paper_arch,
            },
        },
        {
            "component": "clean_walk_and_singleleg_video_chain",
            "status": "fails_success_claim_video_contract",
            "matches_paper": False,
            "evidence": [str(FILES["clean_walk_video"]), str(FILES["mujoco_pd_video"])],
            "notes": (
                "The video chain uses reference anchoring/model-target blending, one-step latent denoise, latent "
                "interpolation as 'guidance', and MuJoCo root assist in several scripts. These are valid diagnostics "
                "only. They do not prove teacher/VAE/diffusion learned single-leg or walking control."
            ),
            "detected_patterns": {
                "reference_blending": "model_target_weight" in video,
                "one_step_denoise": "def denoise_latent" in video and "steps - 1" in video,
                "guidance_is_interpolation": "denoised + guidance_scale * (latent.cpu().numpy() - denoised)" in video,
                "root_assist": "apply_root_assist" in video or "root_assist" in mujoco_pd,
            },
        },
    ]


def video_rows() -> list[dict[str, Any]]:
    rows = []
    for path in FAILED_VIDEO_DIRS:
        mp4s = sorted(path.rglob("*.mp4")) if path.is_dir() else []
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "mp4_count": len(mp4s),
                "status": "failed_or_diagnostic_until_revalidated",
                "claim_level": "diagnostic_visualization_not_success",
                "recommended_action": "archive_under_res_failed_runs_after_success_folder_exists",
                "notes": (
                    "User-visible posture is weak/collapsed or the chain uses non-paper assists. "
                    "Do not use as the final successful single-leg/walk result."
                ),
            }
        )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = component_rows()
    train_json = read_json(FILES["singleleg_train_json"])
    eval_json = read_json(FILES["singleleg_eval_json"])
    latest_train = latest_training_metrics_from_log(FILES["singleleg_train_log"])
    gpu_rows = shell_json(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
            "-i",
            "5,6",
        ]
    )
    gpu_high_memory = all(
        row.get("memory_used_mb", 0) >= 80000 for row in gpu_rows if "memory_used_mb" in row
    )
    train_quality_pass = bool(eval_json.get("quality_gate", {}).get("passed"))
    hard_checks = {
        "paper_source_readable": FILES["paper_method"].is_file() and FILES["paper_supplement"].is_file(),
        "official_stage1_code_readable": FILES["official_tracking_cfg"].is_file(),
        "stage1_official_code_contract_available": next(
            row["matches_paper"] for row in rows if row["component"] == "official_whole_body_tracking_stage1"
        ),
        "legacy_resource_vae_must_remain_disabled_for_success_claims": not next(
            row["matches_paper"] for row in rows if row["component"] == "resource_adjusted_teacher_rollout_vae"
        ),
        "legacy_resource_diffusion_must_remain_disabled_for_success_claims": not next(
            row["matches_paper"] for row in rows if row["component"] == "resource_adjusted_state_latent_diffusion"
        ),
        "paper_contract_vae_interface_available": next(
            row["matches_paper"] for row in rows if row["component"] == "paper_contract_teacher_rollout_vae"
        ),
        "paper_contract_state_latent_dataset_available": next(
            row["matches_paper"] for row in rows if row["component"] == "paper_contract_state_latent_dataset"
        ),
        "paper_contract_diffusion_denoising_available": next(
            row["matches_paper"] for row in rows if row["component"] == "paper_contract_state_latent_diffusion"
        ),
        "paper_contract_transformer_diffusion_code_contract_available": next(
            row["matches_paper"]
            for row in rows
            if row["component"] == "paper_contract_transformer_state_latent_diffusion_code_contract"
        ),
        "paper_contract_offline_guidance_available": next(
            row["matches_paper"] for row in rows if row["component"] == "paper_contract_offline_guidance"
        ),
        "mujoco_control_contract_native_ready": next(
            row["matches_paper"] for row in rows if row["component"] == "mujoco_control_contract_gate"
        ),
        "mujoco_native_action_adapter_formula_ready": next(
            row["matches_paper"] for row in rows if row["component"] == "mujoco_native_action_adapter_formula_gate"
        ),
        "mujoco_native_observation_adapter_ready": next(
            row["matches_paper"] for row in rows if row["component"] == "mujoco_native_observation_adapter_gate"
        ),
        "public_lafan1_arch_full_vae_contract": next(
            row["matches_paper"] for row in rows if row["component"] == "lafan1_paper_arch_training"
        ),
        "video_chain_success_claim_allowed": next(
            row["matches_paper"] for row in rows if row["component"] == "clean_walk_and_singleleg_video_chain"
        ),
        "singleleg_teacher_quality_gate_passed": train_quality_pass,
        "current_gpu_memory_target_reached_80gb_each": gpu_high_memory,
    }
    summary = {
        "status": "blocked_model_chain_not_paper_contract_and_teacher_quality_not_ready",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "beyondmimic_model_chain_paper_contract_audit",
        "scope": (
            "Audits whether local teacher/VAE/diffusion/MuJoCo video code can honestly be used to claim correct "
            "single-leg standing or walking control reproduction."
        ),
        "paper_contract": {
            "stage1": "official whole_body_tracking RL motion tracking with paper observation/action/reward terms",
            "vae": "encoder E(psi, e_anchor), decoder D(z, proprioception), trained with DAgger modified ELBO",
            "diffusion": "state-latent trajectory diffusion with hybrid character-yaw state, emphasis projection, Transformer denoiser, individual state/latent denoising steps",
            "guidance": "classifier guidance applies -grad_tau G(tau) inside reverse denoising",
            "video_success_gate": "teacher and downstream models must reproduce the full lifted-leg pose, not merely remain upright or lean forward",
        },
        "component_rows": rows,
        "video_rows": video_rows(),
        "route_decision": {
            "disable_for_success_claims": [
                "resource_adjusted_teacher_rollout_vae",
                "resource_adjusted_state_latent_diffusion",
                "clean_walk_and_singleleg_video_chain",
            ],
            "preferred_local_diagnostic_route": [
                "stage1_tracking_parameter_contract_gate",
                "paper_contract_teacher_rollout_vae",
                "paper_contract_state_latent_dataset",
                "paper_contract_transformer_state_latent_diffusion_code_contract",
                "paper_contract_state_latent_diffusion",
                "paper_contract_offline_guidance",
            ],
            "why_still_blocked": [
                "Stage-1 teacher quality gate has not passed.",
                "The state-latent dataset still uses local policy_obs rather than the full paper hybrid state.",
                "The paper-style Transformer denoiser has only passed a tiny dry-run code-contract gate; it has not been fully trained or evaluated.",
                "Guidance is offline cost-gradient evaluation, not receding-horizon closed-loop MuJoCo/Isaac control.",
                "MuJoCo video/control adapter uses absolute joint targets, IK traces, root assist, and material differences; it is not yet native normalized-action control.",
                "The native action formula adapter is ready as a fixture, but native observation reconstruction and no-root-assist physics rollout are still missing.",
                "The native 160-D observation adapter remains blocked until it is numerically validated against IsaacLab observation_manager output and motion_tracking_controller frame-alignment semantics.",
                "Official G1 PPO uses empirical observation normalization; native MuJoCo inference must preserve the exported normalizer or checkpoint obs_norm_state_dict.",
                "Existing videos use blending/root assist or weak teacher actions and cannot be the final single-leg success folder.",
            ],
        },
        "singleleg_training": {
            "training_json": str(FILES["singleleg_train_json"]),
            "training_status": train_json.get("status"),
            "eval_json": str(FILES["singleleg_eval_json"]),
            "eval_status": eval_json.get("status"),
            "eval_quality_gate": eval_json.get("quality_gate"),
            "latest_log_metrics": latest_train,
            "gpu_5_6_snapshot": gpu_rows,
            "gpu_high_memory_target_reached_80gb_each": gpu_high_memory,
        },
        "checks": hard_checks,
        "conclusion": {
            "can_claim_teacher_vae_diffusion_singleleg_success": False,
            "can_generate_final_success_folder_now": False,
            "should_train_downstream_vae_diffusion_from_current_teacher": False,
            "should_archive_existing_failed_videos": True,
            "recommended_next_steps": [
                "Do not use current clean_walk/hub_singleleg learned videos as success evidence.",
                "Use the paper-contract VAE route for future diagnostics, not the legacy obs+action resource VAE.",
                "Run full training/evaluation of the paper-contract Transformer diffusion route only after teacher quality improves.",
                "Implement or verify the native MuJoCo/Isaac action adapter before producing final videos: obs -> model -> normalized action -> theta0 + alpha * action -> PD -> physics step.",
                "Implement and validate the native 160-D MuJoCo observation adapter before any direct PPO/VAE/diffusion actor-video claim.",
                "Use the new action adapter formula fixture, but keep logging raw setpoints versus MuJoCo ctrlrange-clipped setpoints for ankle-roll joints.",
                "Train/evaluate a high-throughput Stage-1 teacher with official whole_body_tracking until done rate and posture metrics pass.",
                "Only after the teacher quality gate passes, collect continuous rollouts, train the corrected VAE/diffusion chain, then render one final success folder.",
            ],
        },
    }
    JSON_OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with TSV_OUT.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "component",
            "status",
            "matches_paper",
            "notes",
            "evidence",
            "detected_patterns",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "component": row["component"],
                    "status": row["status"],
                    "matches_paper": row["matches_paper"],
                    "notes": row["notes"],
                    "evidence": json.dumps(row.get("evidence", []), ensure_ascii=False),
                    "detected_patterns": json.dumps(row.get("detected_patterns", {}), ensure_ascii=False),
                }
            )

    failed = [name for name, ok in hard_checks.items() if not ok]
    md = [
        "# BeyondMimic 模型链论文合同审计",
        "",
        f"- 状态：`{summary['status']}`",
        f"- 生成时间：`{summary['generated_at']}`",
        "- 结论：当前 teacher/VAE/diffusion 视频链不能声称已经学会单脚站立或正常走路。",
        "- 当前不得声称完整复现 BeyondMimic，也不得把现有前倾站姿视频作为成功结果。",
        "- 旧 resource-adjusted VAE/diffusion 链条必须继续标记为 diagnostic；新的 paper-contract VAE 链条可作为后续候选，但仍被 teacher quality、Transformer diffusion、closed-loop guidance gate 阻塞。",
        "",
        "## 失败检查",
        "",
    ]
    md.extend(f"- `{item}`" for item in failed)
    md.extend(["", "## 模块结论", ""])
    for row in rows:
        md.append(f"### {row['component']}")
        md.append(f"- 状态：`{row['status']}`")
        md.append(f"- 是否满足论文合同：`{row['matches_paper']}`")
        md.append(f"- 说明：{row['notes']}")
        md.append("")
    md.extend(["## 路由决定", ""])
    for key, values in summary["route_decision"].items():
        md.append(f"### {key}")
        for item in values:
            md.append(f"- {item}")
        md.append("")
    md.extend(
        [
            "## 当前训练状态",
            "",
            f"- 最新训练日志指标：`{latest_train}`",
            f"- GPU 5/6 快照：`{gpu_rows}`",
            f"- 是否达到 80GB/卡目标：`{gpu_high_memory}`",
            "",
            "## 下一步",
            "",
        ]
    )
    md.extend(f"- {item}" for item in summary["conclusion"]["recommended_next_steps"])
    md.append("")
    MD_OUT.write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "json": str(JSON_OUT), "md": str(MD_OUT), "tsv": str(TSV_OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
