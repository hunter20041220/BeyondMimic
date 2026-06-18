#!/usr/bin/env python3
"""Generate a consolidated BeyondMimic reproduction evidence report."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/final_report"
DOC_OUT = ROOT / "reproduction/docs/final_reproduction_report.md"
GOAL_DOC_OUT = OUT / "reproduction_report.md"


def load_json(rel: str) -> dict[str, Any]:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def count_tsv_rows(rel: str) -> int:
    path = ROOT / rel
    with path.open("r", encoding="utf-8", newline="") as f:
        return max(0, sum(1 for _ in csv.DictReader(f, delimiter="\t")))


def rel(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def md_link(path: str | Path) -> str:
    p = Path(path)
    label = p.name
    return f"[{label}]({p})"


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def gather_summary() -> dict[str, Any]:
    master = load_json("res/master_audit/reproduction_master_audit.json")
    blocked = load_json("res/blocked_gates/blocked_gate_audit.json")
    takeover = load_json("res/takeover_audit/takeover_audit.json")
    env_import_probe = load_json("res/setup/env_probe/env_import_probe.json")
    vulkan_runtime_probe = load_json("res/setup/vulkan_runtime_probe/vulkan_runtime_probe.json")
    cuda_p2p_runtime_probe = load_json("res/setup/cuda_p2p_runtime_probe/cuda_p2p_runtime_probe.json")
    isaaclab_live_gate_probe = load_json("res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json")
    isaaclab_gpu_foundation_settings_audit = load_json(
        "res/setup/isaaclab_gpu_foundation_settings_audit/isaaclab_gpu_foundation_settings_audit.json"
    )
    bm_diffusion_env = load_json("res/setup/bm_diffusion_env_audit/bm_diffusion_env_audit.json")
    gpu_resource = load_json("res/setup/gpu_resource_audit/gpu_resource_audit.json")
    run_management = load_json("res/run_management_audit/run_management_audit.json")
    checkpoint_resume = load_json("res/run_management_audit/checkpoint_resume_smoke/checkpoint_resume_smoke.json")
    full_run_deliverable_gap = load_json(
        "res/run_management_audit/full_run_deliverable_gap_audit/full_run_deliverable_gap_audit.json"
    )
    failed_run = load_json("res/failed_runs/failed_run_audit/failed_run_audit.json")
    official_train_entry_failed_run = load_json(
        "res/failed_runs/official_train_entry_failed_run_audit/official_train_entry_failed_run_audit.json"
    )
    patch_inventory = load_json("res/code/patch_inventory_audit/patch_inventory_audit.json")
    patch_snapshot = load_json("res/code/patch_snapshot_audit/patch_snapshot_audit.json")
    reimpl_package = load_json("res/code/reimpl_package_audit/reimpl_package_audit.json")
    reimpl_runtime = load_json("res/code/reimpl_runtime_integration_audit/reimpl_runtime_integration_audit.json")
    coding_requirements = load_json("res/code/coding_requirements_audit/coding_requirements_audit.json")
    resolved_config = load_json("res/config/resolved_reproduction_config.json")
    artifact_manifest = load_json("res/artifact_manifest/artifact_manifest.json")
    download_source_integrity = load_json(
        "res/source_integrity/download_source_integrity/download_source_integrity_audit.json"
    )
    run_log_config_catalog = load_json("res/run_log_config_catalog/run_log_config_catalog.json")
    experiment_protocol = load_json("res/docs/experiment_protocol_audit/experiment_protocol_audit.json")
    readme_audit = load_json("res/docs/readme_audit/readme_audit.json")
    progress_audit = load_json("res/progress_report_audit/progress_report_audit.json")
    completion_matrix_status = load_json(
        "res/docs/completion_matrix_status_audit/completion_matrix_status_audit.json"
    )
    project_boundary = load_json("res/project_boundary_audit/project_boundary_audit.json")
    final_deliverables = load_json("res/final_deliverables_audit/final_deliverables_audit.json")
    visual_media_inventory = load_json("res/visual_media_inventory/visual_media_inventory_audit.json")
    verification_command_coverage = load_json(
        "res/verification_command_coverage/verification_command_coverage_audit.json"
    )
    verification_command_syntax = load_json(
        "res/verification_command_syntax/verification_command_syntax_audit.json"
    )
    verification_command_script_manifest = load_json(
        "res/verification_command_script_manifest/verification_command_script_manifest.json"
    )
    required_artifact_absence = load_json("res/required_artifact_absence/required_artifact_absence_audit.json")
    evaluation_metrics = load_json("res/evaluation_metrics_coverage/evaluation_metrics_coverage_audit.json")
    trial_failure_accounting = load_json(
        "res/evaluation/trial_failure_accounting_audit/trial_failure_accounting_audit.json"
    )
    ablation_coverage = load_json("res/ablation_coverage/ablation_coverage_audit.json")
    metrics_catalog = load_json("res/metrics/metrics_catalog/metrics_catalog.json")
    released_data_metrics = load_json(
        "res/tables/released_data_metrics_summary/released_data_metrics_summary.json"
    )
    released_data_statistical = load_json(
        "res/tables/released_data_statistical_audit/released_data_statistical_audit.json"
    )
    level_a_released_data_suite = load_json("res/level_a/released_data_suite/level_a_released_data_suite.json")
    guidance_task_coverage = load_json("res/guidance_task_coverage/guidance_task_coverage_audit.json")
    coverage = load_json("res/paper_source_coverage/paper_source_coverage_audit.json")
    paper_latex_inventory = load_json("res/paper_latex_inventory/paper_latex_inventory_audit.json")
    paper_formula_code_trace = load_json("res/paper_formula_code_trace/paper_formula_code_trace_audit.json")
    paper_pdf_source_consistency = load_json(
        "res/paper_pdf_source_consistency/paper_pdf_source_consistency_audit.json"
    )
    table_values = load_json("res/paper_table_values/paper_table_value_audit.json")
    skill_success_table = load_json("res/paper_skill_success_table_audit/skill_success_table_data_audit.json")
    released_panel_mapping = load_json("res/released_panel_mapping_audit/released_panel_mapping_audit.json")
    comparison = load_json("res/comparison/paper_vs_reproduction.json")
    results_claims = load_json("res/results_claims_audit/results_claims_audit.json")
    traceability = load_json("res/goal_traceability/goal_traceability_audit.json")
    goal_directive_index = load_json("res/goal_directive_index/goal_directive_index_audit.json")
    goal_matrix = load_json("res/goal_requirement_matrix/goal_requirement_matrix_audit.json")
    adaptive_sampling_discrepancy = load_json(
        "res/tracking/adaptive_sampling_discrepancy_audit/adaptive_sampling_discrepancy_audit.json"
    )
    tracking_smoke_rerun = load_json(
        "res/tracking/tracking_smoke_rerun_audit/tracking_smoke_rerun_audit.json"
    )
    tracking_official_train_entry_retry = load_json(
        "res/tracking/official_train_entry_retry_audit/tracking_official_train_entry_retry_audit.json"
    )
    kit_inotify_budget = load_json(
        "res/setup/kit_inotify_budget_audit/kit_inotify_budget_audit.json"
    )
    inotify_live_usage = load_json(
        "res/setup/inotify_live_usage_audit/inotify_live_usage_audit.json"
    )
    vscode_watcher_exclude = load_json(
        "res/setup/vscode_watcher_exclude_audit/vscode_watcher_exclude_audit.json"
    )
    kit_watcher_config_surface = load_json(
        "res/setup/kit_watcher_config_surface_audit/kit_watcher_config_surface_audit.json"
    )
    tracking_import_gate = load_json(
        "res/tracking/tracking_import_gate_audit/tracking_import_gate_audit.json"
    )
    tracking_extension_namespace = load_json(
        "res/tracking/tracking_extension_namespace_probe/tracking_extension_namespace_probe.json"
    )
    tracking_official_source_contract = load_json(
        "res/tracking/official_source_contract_audit/tracking_official_source_contract_audit.json"
    )
    tracking_g1_action_scale = load_json(
        "res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json"
    )
    tracking_reward_formula = load_json(
        "res/tracking/reward_formula_audit/tracking_reward_formula_audit.json"
    )
    tracking_observation_action_schema = load_json(
        "res/tracking/observation_action_schema_audit/tracking_observation_action_schema_audit.json"
    )
    tracking_randomization_termination = load_json(
        "res/tracking/randomization_termination_audit/tracking_randomization_termination_audit.json"
    )
    level_b_tracking_nonkit_suite = load_json(
        "res/tracking/nonkit_suite/level_b_tracking_nonkit_suite.json"
    )
    motion_preprocessing_contract = load_json(
        "res/tracking/motion_preprocessing_contract_audit/motion_preprocessing_contract_audit.json"
    )
    tracking_motion_npz_fixture = load_json("res/tracking/motion_npz_fixture/tracking_motion_npz_fixture.json")
    tracking_official_replay_preflight = load_json(
        "res/tracking/official_replay_preflight/tracking_official_replay_preflight.json"
    )
    tracking_official_replay_conversion = load_json(
        "res/tracking/official_replay_conversion/tracking_official_replay_conversion_audit.json"
    )
    tracking_urdf_conversion_probe = load_json("res/tracking/urdf_conversion_probe/tracking_urdf_conversion_probe.json")
    tracking_urdf_path_tiny_probe = load_json(
        "res/tracking/urdf_path_tiny_probe/tracking_urdf_path_tiny_probe.json"
    )
    tracking_mjcf_stage_probe = load_json("res/tracking/mjcf_stage_probe/tracking_mjcf_stage_probe.json")
    tracking_usd_save_policy_probe = load_json(
        "res/tracking/usd_save_policy_probe/tracking_usd_save_policy_probe.json"
    )
    tracking_simulationapp_save_policy_probe = load_json(
        "res/tracking/simulationapp_save_policy_probe/tracking_simulationapp_save_policy_probe.json"
    )
    tracking_local_smoke_preflight = load_json(
        "res/tracking/local_smoke_preflight/tracking_local_smoke_preflight.json"
    )
    tracking_onnx_contract = load_json(
        "res/tracking/onnx_export_contract_audit/tracking_onnx_export_contract_audit.json"
    )
    tracking_motion_policy_onnx_fixture = load_json(
        "res/tracking/motion_policy_onnx_contract_fixture/tracking_motion_policy_onnx_contract_fixture.json"
    )
    tracking_debug_motion_policy_onnx = load_json(
        "res/tracking/debug_motion_policy_onnx_export/tracking_debug_motion_policy_onnx_export.json"
    )
    tracking_debug_motion_policy_onnx_inference = load_json(
        "res/tracking/debug_motion_policy_onnx_inference/tracking_debug_motion_policy_onnx_inference_audit.json"
    )
    mujoco_ros_launch_contract = load_json(
        "res/tracking/mujoco_ros_launch_contract_audit/mujoco_ros_launch_contract_audit.json"
    )
    tracking_deployment_controller_semantics = load_json(
        "res/tracking/deployment_controller_semantics_audit/tracking_deployment_controller_semantics_audit.json"
    )
    official_artifacts = load_json("res/level_c/official_artifact_audit/level_c_official_artifact_audit.json")
    fig56 = load_json("res/level_c/fig5_fig6_feasibility_audit/level_c_fig5_fig6_feasibility_audit.json")
    level_c_debug_suite = load_json("res/level_c/debug_suite/level_c_debug_suite.json")
    level_c_extended_debug_suite = load_json(
        "res/level_c/extended_debug_suite/level_c_extended_debug_suite.json"
    )
    diffusion_equations = load_json("res/level_c/diffusion_equation_audit/level_c_diffusion_equation_audit.json")
    inverse_transform = load_json(
        "res/level_c/trajectory_inverse_transform_audit/level_c_trajectory_inverse_transform_audit.json"
    )
    emphasis_projection = load_json("res/level_c/emphasis_projection_audit/level_c_emphasis_projection_audit.json")
    state_representation = load_json(
        "res/level_c/state_representation_source_audit/level_c_state_representation_source_audit.json"
    )
    dataset_collection_protocol = load_json(
        "res/level_c/dataset_collection_protocol_audit/level_c_dataset_collection_protocol_audit.json"
    )
    rollout_rejection_manifest = load_json(
        "res/level_c/rollout_rejection_manifest_probe/level_c_rollout_rejection_manifest_probe.json"
    )
    state_latent_schema = load_json("res/level_c/state_latent_schema_audit/state_latent_schema_audit.json")
    dagger_schema = load_json("res/level_c/dagger_schema_audit/dagger_schema_audit.json")
    paper_state_windows = load_json("res/level_c/paper_state_windows/level_c_paper_state_windows.json")
    state_latent_dataset_consistency = load_json(
        "res/level_c/state_latent_dataset_consistency_audit/level_c_state_latent_dataset_consistency_audit.json"
    )
    state_latent_training_dataset_contract = load_json(
        "res/level_c/state_latent_training_dataset_contract_audit/"
        "level_c_state_latent_training_dataset_contract_audit.json"
    )
    paper_state_overfit = load_json("res/level_c/paper_state_overfit_probe/level_c_paper_state_overfit_probe.json")
    dagger_iteration = load_json("res/level_c/dagger_iteration_smoke/level_c_dagger_iteration_smoke.json")
    vae_latent_diffusion_overfit = load_json(
        "res/level_c/vae_latent_diffusion_overfit_probe/level_c_vae_latent_diffusion_overfit_probe.json"
    )
    vae_latent_heldout = load_json("res/level_c/vae_latent_heldout_eval/level_c_vae_latent_heldout_eval.json")
    paper_state_heldout = load_json("res/level_c/paper_state_heldout_eval/level_c_paper_state_heldout_eval.json")
    paper_state_heldout_multiseed = load_json(
        "res/level_c/paper_state_heldout_multiseed_audit/level_c_paper_state_heldout_multiseed_audit.json"
    )
    vae_latent_heldout_multiseed = load_json(
        "res/level_c/vae_latent_heldout_multiseed_audit/level_c_vae_latent_heldout_multiseed_audit.json"
    )
    paper_state_transformer = load_json(
        "res/level_c/paper_state_transformer_arch_probe/level_c_paper_state_transformer_arch_probe.json"
    )
    vae_latent_transformer = load_json(
        "res/level_c/vae_latent_transformer_arch_probe/level_c_vae_latent_transformer_arch_probe.json"
    )
    transformer_parameter_count = load_json(
        "res/level_c/transformer_parameter_count_audit/level_c_transformer_parameter_count_audit.json"
    )
    transformer_state_dict = load_json(
        "res/level_c/transformer_state_dict_manifest/level_c_transformer_state_dict_manifest.json"
    )
    transformer_ema = load_json("res/level_c/transformer_ema_smoke/level_c_transformer_ema_smoke.json")
    vae_latent_transformer_ema = load_json(
        "res/level_c/vae_latent_transformer_ema_smoke/level_c_vae_latent_transformer_ema_smoke.json"
    )
    diffusion_checkpoint_smoke = load_json(
        "res/level_c/diffusion_checkpoint_smoke/level_c_diffusion_checkpoint_smoke.json"
    )
    bounded_debug_diffusion_training_run = load_json(
        "res/level_c/bounded_debug_diffusion_training_run/level_c_bounded_debug_diffusion_training_run.json"
    )
    bounded_debug_diffusion_checkpoint_eval = load_json(
        "res/level_c/bounded_debug_diffusion_checkpoint_eval/level_c_bounded_debug_diffusion_checkpoint_eval.json"
    )
    bounded_debug_diffusion_action_eval = load_json(
        "res/level_c/bounded_debug_diffusion_action_eval/level_c_bounded_debug_diffusion_action_eval.json"
    )
    resource_adjusted_tiny_diffusion = load_json(
        "res/level_c/resource_adjusted_tiny_diffusion_training_run/"
        "level_c_resource_adjusted_tiny_diffusion_training_run.json"
    )
    resource_adjusted_tiny_suite = load_json(
        "res/level_c/resource_adjusted_tiny_diffusion_suite/"
        "level_c_resource_adjusted_tiny_diffusion_suite.json"
    )
    resource_adjusted_tiny_multiseed = load_json(
        "res/level_c/resource_adjusted_tiny_diffusion_multiseed_audit/"
        "level_c_resource_adjusted_tiny_diffusion_multiseed_audit.json"
    )
    resource_adjusted_tiny_checkpoint_eval = load_json(
        "res/level_c/resource_adjusted_tiny_diffusion_checkpoint_eval/"
        "level_c_resource_adjusted_tiny_diffusion_checkpoint_eval.json"
    )
    resource_adjusted_tiny_onnx = load_json(
        "res/level_c/resource_adjusted_tiny_diffusion_onnx_export_inference/"
        "level_c_resource_adjusted_tiny_diffusion_onnx_export_inference.json"
    )
    resource_adjusted_tiny_latency = load_json(
        "res/level_c/resource_adjusted_tiny_diffusion_latency_audit/"
        "level_c_resource_adjusted_tiny_diffusion_latency_audit.json"
    )
    resource_adjusted_tiny_video_preview = load_json(
        "res/level_c/resource_adjusted_tiny_diffusion_video_preview/"
        "level_c_resource_adjusted_tiny_diffusion_video_preview.json"
    )
    lafan1_paper_arch_training = load_json(
        "res/level_c/lafan1_paper_arch_vae_diffusion_training/"
        "lafan1_paper_arch_vae_diffusion_training.json"
    )
    lafan1_paper_arch_multiseed = load_json(
        "res/level_c/lafan1_paper_arch_multiseed_audit/"
        "level_c_lafan1_paper_arch_multiseed_audit.json"
    )
    lafan1_paper_arch_symmetry_multiseed = load_json(
        "res/level_c/lafan1_paper_arch_symmetry_augmented_multiseed_audit/"
        "level_c_lafan1_paper_arch_symmetry_augmented_multiseed_audit.json"
    )
    lafan1_paper_arch_high_memory = load_json(
        "res/level_c/lafan1_paper_arch_high_memory_batch_audit/"
        "level_c_lafan1_paper_arch_high_memory_batch_audit.json"
    )
    lafan1_paper_arch_symmetry_dataset = load_json(
        "res/level_c/lafan1_paper_arch_symmetry_dataset/"
        "lafan1_paper_arch_symmetry_dataset_audit.json"
    )
    lafan1_paper_arch_symmetry_training = load_json(
        "res/level_c/lafan1_paper_arch_symmetry_augmented_training/"
        "lafan1_paper_arch_vae_diffusion_training.json"
    )
    lafan1_paper_arch_symmetry_training_comparison = load_json(
        "res/level_c/lafan1_paper_arch_symmetry_training_comparison/"
        "level_c_lafan1_paper_arch_symmetry_training_comparison_audit.json"
    )
    lafan1_paper_arch_onnx_latency = load_json(
        "res/level_c/lafan1_paper_arch_onnx_latency/"
        "level_c_lafan1_paper_arch_onnx_latency_audit.json"
    )
    lafan1_paper_arch_symmetry_onnx_latency = load_json(
        "res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/"
        "level_c_lafan1_paper_arch_onnx_latency_audit.json"
    )
    lafan1_paper_arch_offline_metrics = load_json(
        "res/level_c/lafan1_paper_arch_offline_metrics/"
        "level_c_lafan1_paper_arch_offline_metrics_audit.json"
    )
    lafan1_paper_arch_symmetry_offline_metrics = load_json(
        "res/level_c/lafan1_paper_arch_symmetry_augmented_offline_metrics/"
        "level_c_lafan1_paper_arch_offline_metrics_audit.json"
    )
    lafan1_paper_arch_guidance_eval = load_json(
        "res/level_c/lafan1_paper_arch_guidance_eval/"
        "level_c_lafan1_paper_arch_guidance_eval.json"
    )
    lafan1_paper_arch_symmetry_guidance_eval = load_json(
        "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval/"
        "level_c_lafan1_paper_arch_guidance_eval.json"
    )
    lafan1_paper_arch_symmetry_guidance_eval_full_split = load_json(
        "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/"
        "level_c_lafan1_paper_arch_guidance_eval.json"
    )
    lafan1_paper_arch_reverse_guidance = load_json(
        "res/level_c/lafan1_paper_arch_reverse_guidance/"
        "level_c_lafan1_paper_arch_reverse_guidance_audit.json"
    )
    lafan1_paper_arch_symmetry_reverse_guidance = load_json(
        "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance/"
        "level_c_lafan1_paper_arch_reverse_guidance_audit.json"
    )
    lafan1_paper_arch_symmetry_reverse_guidance_full_split = load_json(
        "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/"
        "level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.json"
    )
    single_batch_overfit = load_json("res/level_c/single_batch_overfit_probe/level_c_single_batch_overfit_probe.json")
    single_motion_overfit = load_json("res/level_c/single_motion_overfit_probe/level_c_single_motion_overfit_probe.json")
    small_dataset_overfit = load_json("res/level_c/small_dataset_overfit_probe/level_c_small_dataset_overfit_probe.json")
    small_dataset_split = load_json(
        "res/level_c/small_dataset_split_manifest/small_dataset_split_manifest_summary.json"
    )
    small_dataset_multiseed = load_json(
        "res/level_c/small_dataset_multiseed_audit/level_c_small_dataset_multiseed_audit.json"
    )
    small_dataset_heldout = load_json("res/level_c/small_dataset_heldout_eval/level_c_small_dataset_heldout_eval.json")
    small_dataset_heldout_multiseed = load_json(
        "res/level_c/small_dataset_heldout_multiseed_audit/level_c_small_dataset_heldout_multiseed_audit.json"
    )
    vae_checkpoint_smoke = load_json("res/level_c/vae_checkpoint_smoke/level_c_vae_checkpoint_smoke.json")
    vae_debug_overfit_latents = load_json(
        "res/level_c/vae_debug_overfit_latent_artifact/level_c_vae_debug_overfit_latent_artifact.json"
    )
    vae_motion_split_heldout = load_json(
        "res/level_c/vae_motion_split_heldout_eval/level_c_vae_motion_split_heldout_eval.json"
    )
    vae_receding_horizon_rollout = load_json(
        "res/level_c/vae_receding_horizon_rollout_smoke/level_c_vae_receding_horizon_rollout_smoke.json"
    )
    diffusion_to_vae_action = load_json(
        "res/level_c/diffusion_to_vae_action_smoke/level_c_diffusion_to_vae_action_smoke.json"
    )
    diffusion_to_vae_action_multiseed = load_json(
        "res/level_c/diffusion_to_vae_action_multiseed_audit/level_c_diffusion_to_vae_action_multiseed_audit.json"
    )
    diffusion_to_vae_action_smoothness = load_json(
        "res/level_c/diffusion_to_vae_action_smoothness_audit/level_c_diffusion_to_vae_action_smoothness_audit.json"
    )
    direct_vs_latent_action_ablation = load_json(
        "res/level_c/direct_vs_latent_action_ablation_audit/level_c_direct_vs_latent_action_ablation_audit.json"
    )
    vae_contract = load_json("res/level_c/vae_contract_audit/level_c_vae_contract_audit.json")
    dagger_vae_pipeline = load_json(
        "res/level_c/dagger_vae_pipeline_audit/level_c_dagger_vae_pipeline_audit.json"
    )
    vae_latent = load_json("res/level_c/vae_latent_probe/level_c_vae_latent_probe.json")
    symmetry_mapping = load_json("res/level_c/symmetry_mapping_audit/level_c_symmetry_mapping_audit.json")
    guidance_task_scale = load_json("res/level_c/guidance_task_scale_sweep/level_c_guidance_task_scale_sweep.json")
    guidance_debug_visualization = load_json(
        "res/level_c/guidance_debug_visualization/level_c_guidance_debug_visualization.json"
    )
    guidance_task_metric = load_json(
        "res/level_c/guidance_task_metric_audit/level_c_guidance_task_metric_audit.json"
    )
    guidance_full_split_result_table = load_json(
        "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.json"
    )
    guidance_checkpoint_visualization = load_json(
        "res/level_c/guidance_checkpoint_visualization/level_c_guidance_checkpoint_visualization.json"
    )
    guidance_visual_deliverables = load_json(
        "res/level_c/guidance_visual_deliverables_audit/level_c_guidance_visual_deliverables_audit.json"
    )
    guidance_cost_coverage = load_json(
        "res/level_c/guidance_cost_coverage_audit/level_c_guidance_cost_coverage_audit.json"
    )
    core_math_unit_tests = load_json("res/tests/core_math_unit_tests/core_math_unit_tests.json")
    reimpl_package_api_tests = load_json("res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json")
    reimpl_test_suite = load_json("res/tests/reimpl_test_suite/reimpl_test_suite.json")
    core_test_coverage = load_json("res/tests/core_test_coverage_audit/core_test_coverage_audit.json")
    timestep_mask_coverage = load_json(
        "res/level_c/timestep_mask_coverage_audit/level_c_timestep_mask_coverage_audit.json"
    )
    paper_state_mask_reverse = load_json(
        "res/level_c/paper_state_mask_reverse_probe/level_c_paper_state_mask_reverse_probe.json"
    )
    smoothness_latency = load_json(
        "res/level_c/smoothness_latency_audit/level_c_smoothness_latency_audit.json"
    )
    deployment_protocol = load_json(
        "res/level_c/deployment_protocol_audit/level_c_deployment_protocol_audit.json"
    )

    released_rows = count_tsv_rows("res/released_figures/released_figure_summary.tsv")
    paper_panel_rows = count_tsv_rows("reproduction/docs/paper_panel_map.tsv")

    return {
        "status": "ok",
        "experiment_type": "report",
        "scope": "consolidated BeyondMimic reproduction evidence and remaining gates",
        "goal_complete": False,
        "master": {
            "artifact_count": master["artifact_count"],
            "artifact_pass_count": master["artifact_pass_count"],
            "artifact_fail_count": master["artifact_fail_count"],
            "completion_matrix_counts": master["completion_matrix_counts"],
            "why_not_complete": master["interpretation"]["why_not_complete"],
        },
        "level_a_released_data": {
            "status": "complete_for_released_dataset_scope",
            "released_figure_rows": released_rows,
            "paper_panel_map_rows": paper_panel_rows,
            "released_panel_mapping_status": released_panel_mapping["status"],
            "released_panel_mapping_metrics": released_panel_mapping["metrics"],
            "released_data_metrics_status": released_data_metrics["status"],
            "released_data_metrics": released_data_metrics["metrics"],
            "released_data_metrics_checks": released_data_metrics["checks"],
            "released_data_statistical_audit_status": released_data_statistical["status"],
            "released_data_statistical_audit_metrics": released_data_statistical["metrics"],
            "released_data_statistical_audit_checks": released_data_statistical["checks"],
            "released_data_suite_status": level_a_released_data_suite["status"],
            "released_data_suite_step_count": level_a_released_data_suite["step_count"],
            "released_data_suite_pass_count": level_a_released_data_suite["pass_count"],
            "released_data_suite_metrics": level_a_released_data_suite["metrics"],
            "released_data_suite_checks": level_a_released_data_suite["checks"],
            "released_data_suite_json": str(
                ROOT / "res/level_a/released_data_suite/level_a_released_data_suite.json"
            ),
            "released_panel_mapping_json": str(
                ROOT / "res/released_panel_mapping_audit/released_panel_mapping_audit.json"
            ),
            "released_data_metrics_json": str(
                ROOT / "res/tables/released_data_metrics_summary/released_data_metrics_summary.json"
            ),
            "released_data_metrics_markdown": str(
                ROOT / "res/tables/released_data_metrics_summary/released_data_metrics_summary.md"
            ),
            "released_data_statistical_audit_json": str(
                ROOT / "res/tables/released_data_statistical_audit/released_data_statistical_audit.json"
            ),
            "released_data_statistical_audit_markdown": str(
                ROOT / "res/tables/released_data_statistical_audit/released_data_statistical_audit.md"
            ),
            "summary_tsv": str(ROOT / "res/released_figures/released_figure_summary.tsv"),
            "panel_map_tsv": str(ROOT / "reproduction/docs/paper_panel_map.tsv"),
        },
        "level_b_tracking": {
            "status": "partial_blocked_for_live_kit_and_deployment",
            "tracking_config_audit": str(ROOT / "res/tracking/smoke_config_audit/tracking_config_audit.json"),
            "tracking_smoke_rerun_status": tracking_smoke_rerun["status"],
            "tracking_smoke_rerun_metrics": tracking_smoke_rerun["metrics"],
            "tracking_smoke_rerun_checks": tracking_smoke_rerun["checks"],
            "tracking_smoke_rerun_json": str(
                ROOT / "res/tracking/tracking_smoke_rerun_audit/tracking_smoke_rerun_audit.json"
            ),
            "tracking_official_train_entry_retry_status": tracking_official_train_entry_retry["status"],
            "tracking_official_train_entry_retry_classification": tracking_official_train_entry_retry[
                "classification"
            ],
            "tracking_official_train_entry_retry_checks": tracking_official_train_entry_retry["checks"],
            "tracking_official_train_entry_retry_outputs": tracking_official_train_entry_retry["outputs"],
            "tracking_official_train_entry_retry_json": str(
                ROOT
                / "res/tracking/official_train_entry_retry_audit/"
                / "tracking_official_train_entry_retry_audit.json"
            ),
            "kit_inotify_budget_status": kit_inotify_budget["status"],
            "kit_inotify_budget_metrics": kit_inotify_budget["metrics"],
            "kit_inotify_budget_checks": kit_inotify_budget["checks"],
            "kit_inotify_budget_json": str(
                ROOT / "res/setup/kit_inotify_budget_audit/kit_inotify_budget_audit.json"
            ),
            "inotify_live_usage_status": inotify_live_usage["status"],
            "inotify_live_usage_metrics": inotify_live_usage["metrics"],
            "inotify_live_usage_checks": inotify_live_usage["checks"],
            "inotify_live_usage_json": str(
                ROOT / "res/setup/inotify_live_usage_audit/inotify_live_usage_audit.json"
            ),
            "vscode_watcher_exclude_status": vscode_watcher_exclude["status"],
            "vscode_watcher_exclude_snapshot": vscode_watcher_exclude["live_usage_snapshot"],
            "vscode_watcher_exclude_checks": vscode_watcher_exclude["checks"],
            "vscode_watcher_exclude_json": str(
                ROOT / "res/setup/vscode_watcher_exclude_audit/vscode_watcher_exclude_audit.json"
            ),
            "kit_watcher_config_surface_status": kit_watcher_config_surface["status"],
            "kit_watcher_config_surface_metrics": kit_watcher_config_surface["metrics"],
            "kit_watcher_config_surface_checks": kit_watcher_config_surface["checks"],
            "kit_watcher_config_surface_json": str(
                ROOT / "res/setup/kit_watcher_config_surface_audit/kit_watcher_config_surface_audit.json"
            ),
            "tracking_import_gate_status": tracking_import_gate["status"],
            "tracking_import_gate_metrics": tracking_import_gate["metrics"],
            "tracking_import_gate_checks": tracking_import_gate["checks"],
            "tracking_import_gate_json": str(
                ROOT / "res/tracking/tracking_import_gate_audit/tracking_import_gate_audit.json"
            ),
            "tracking_extension_namespace_status": tracking_extension_namespace["status"],
            "tracking_extension_namespace_metrics": tracking_extension_namespace["metrics"],
            "tracking_extension_namespace_checks": tracking_extension_namespace["checks"],
            "tracking_extension_namespace_json": str(
                ROOT / "res/tracking/tracking_extension_namespace_probe/tracking_extension_namespace_probe.json"
            ),
            "tracking_official_source_contract_status": tracking_official_source_contract["status"],
            "tracking_official_source_contract_checks": tracking_official_source_contract["checks"],
            "tracking_official_source_contract_metrics": {
                "target_body_count": tracking_official_source_contract["flat_env"]["body_count"],
                "policy_term_count": tracking_official_source_contract["tracking_env"]["policy_term_count"],
                "critic_term_count": tracking_official_source_contract["tracking_env"]["critic_term_count"],
                "reward_term_count": tracking_official_source_contract["tracking_env"]["reward_term_count"],
                "event_term_count": tracking_official_source_contract["tracking_env"]["event_term_count"],
                "termination_term_count": tracking_official_source_contract["tracking_env"]["termination_term_count"],
                "ppo_max_iterations": tracking_official_source_contract["ppo"]["max_iterations"],
                "urdf_non_fixed_joint_count": tracking_official_source_contract["urdf"]["joint_count_non_fixed"],
                "urdf_uncovered_non_fixed_joint_count": tracking_official_source_contract["urdf"][
                    "uncovered_non_fixed_joint_count"
                ],
            },
            "tracking_official_source_contract_json": str(
                ROOT
                / "res/tracking/official_source_contract_audit/"
                / "tracking_official_source_contract_audit.json"
            ),
            "tracking_g1_action_scale_status": tracking_g1_action_scale["status"],
            "tracking_g1_action_scale_metrics": tracking_g1_action_scale["metrics"],
            "tracking_g1_action_scale_checks": tracking_g1_action_scale["checks"],
            "tracking_g1_action_scale_json": str(
                ROOT / "res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json"
            ),
            "tracking_g1_action_scale_tsv": str(
                ROOT / "res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.tsv"
            ),
            "tracking_reward_formula_status": tracking_reward_formula["status"],
            "tracking_reward_formula_metrics": tracking_reward_formula["metrics"],
            "tracking_reward_formula_checks": tracking_reward_formula["checks"],
            "tracking_reward_formula_json": str(
                ROOT / "res/tracking/reward_formula_audit/tracking_reward_formula_audit.json"
            ),
            "tracking_reward_formula_scan_tsv": str(
                ROOT / "res/tracking/reward_formula_audit/tracking_reward_formula_scan.tsv"
            ),
            "tracking_reward_formula_summary_tsv": str(
                ROOT / "res/tracking/reward_formula_audit/tracking_reward_formula_summary.tsv"
            ),
            "tracking_observation_action_schema_status": tracking_observation_action_schema["status"],
            "tracking_observation_action_schema_metrics": tracking_observation_action_schema["metrics"],
            "tracking_observation_action_schema_checks": tracking_observation_action_schema["checks"],
            "tracking_observation_action_schema_json": str(
                ROOT
                / "res/tracking/observation_action_schema_audit/"
                / "tracking_observation_action_schema_audit.json"
            ),
            "tracking_observation_action_schema_tsv": str(
                ROOT
                / "res/tracking/observation_action_schema_audit/"
                / "tracking_observation_action_schema_audit.tsv"
            ),
            "tracking_randomization_termination_status": tracking_randomization_termination["status"],
            "tracking_randomization_termination_metrics": tracking_randomization_termination["metrics"],
            "tracking_randomization_termination_checks": tracking_randomization_termination["checks"],
            "tracking_randomization_termination_json": str(
                ROOT
                / "res/tracking/randomization_termination_audit/"
                / "tracking_randomization_termination_audit.json"
            ),
            "tracking_randomization_events_tsv": str(
                ROOT / "res/tracking/randomization_termination_audit/tracking_randomization_events.tsv"
            ),
            "tracking_termination_terms_tsv": str(
                ROOT / "res/tracking/randomization_termination_audit/tracking_termination_terms.tsv"
            ),
            "tracking_nonkit_suite_status": level_b_tracking_nonkit_suite["status"],
            "tracking_nonkit_suite_step_count": level_b_tracking_nonkit_suite["step_count"],
            "tracking_nonkit_suite_pass_count": level_b_tracking_nonkit_suite["pass_count"],
            "tracking_nonkit_suite_metrics": level_b_tracking_nonkit_suite["metrics"],
            "tracking_nonkit_suite_checks": level_b_tracking_nonkit_suite["checks"],
            "tracking_nonkit_suite_json": str(
                ROOT / "res/tracking/nonkit_suite/level_b_tracking_nonkit_suite.json"
            ),
            "tracking_nonkit_suite_tsv": str(
                ROOT / "res/tracking/nonkit_suite/level_b_tracking_nonkit_suite.tsv"
            ),
            "adaptive_sampling_discrepancy_status": adaptive_sampling_discrepancy["status"],
            "adaptive_sampling_discrepancy_metrics": adaptive_sampling_discrepancy["metrics"],
            "adaptive_sampling_discrepancy_checks": adaptive_sampling_discrepancy["checks"],
            "adaptive_sampling_discrepancy_json": str(
                ROOT
                / "res/tracking/adaptive_sampling_discrepancy_audit/"
                / "adaptive_sampling_discrepancy_audit.json"
            ),
            "motion_preprocessing_contract_status": motion_preprocessing_contract["status"],
            "motion_preprocessing_contract_metrics": motion_preprocessing_contract["metrics"],
            "motion_preprocessing_contract_checks": motion_preprocessing_contract["checks"],
            "motion_preprocessing_contract_json": str(
                ROOT
                / "res/tracking/motion_preprocessing_contract_audit/"
                / "motion_preprocessing_contract_audit.json"
            ),
            "tracking_motion_npz_fixture_status": tracking_motion_npz_fixture["status"],
            "tracking_motion_npz_fixture_metrics": tracking_motion_npz_fixture["metrics"],
            "tracking_motion_npz_fixture_checks": tracking_motion_npz_fixture["checks"],
            "tracking_motion_npz_fixture_json": str(
                ROOT / "res/tracking/motion_npz_fixture/tracking_motion_npz_fixture.json"
            ),
            "tracking_official_replay_preflight_status": tracking_official_replay_preflight["status"],
            "tracking_official_replay_preflight_checks": tracking_official_replay_preflight["checks"],
            "tracking_official_replay_preflight_runtime_requirements": tracking_official_replay_preflight[
                "runtime_requirements"
            ],
            "tracking_official_replay_preflight_commands": tracking_official_replay_preflight[
                "commands_planned_not_run"
            ],
            "tracking_official_replay_preflight_json": str(
                ROOT / "res/tracking/official_replay_preflight/tracking_official_replay_preflight.json"
            ),
            "tracking_official_replay_conversion_status": tracking_official_replay_conversion["status"],
            "tracking_official_replay_conversion_checks": tracking_official_replay_conversion["checks"],
            "tracking_official_replay_conversion_latest_blocker": tracking_official_replay_conversion[
                "latest_blocker"
            ],
            "tracking_official_replay_conversion_repairs": tracking_official_replay_conversion[
                "environment_repairs"
            ],
            "tracking_official_replay_conversion_json": str(
                ROOT / "res/tracking/official_replay_conversion/tracking_official_replay_conversion_audit.json"
            ),
            "tracking_urdf_conversion_probe_status": tracking_urdf_conversion_probe["status"],
            "tracking_urdf_conversion_probe_payload": tracking_urdf_conversion_probe["payload"],
            "tracking_urdf_conversion_probe_json": str(
                ROOT / "res/tracking/urdf_conversion_probe/tracking_urdf_conversion_probe.json"
            ),
            "tracking_urdf_path_tiny_probe_status": tracking_urdf_path_tiny_probe["status"],
            "tracking_urdf_path_tiny_probe_current_blocker": tracking_urdf_path_tiny_probe["current_blocker"],
            "tracking_urdf_path_tiny_probe_markers": tracking_urdf_path_tiny_probe["markers"],
            "tracking_urdf_path_tiny_probe_checks": tracking_urdf_path_tiny_probe["checks"],
            "tracking_urdf_path_tiny_probe_json": str(
                ROOT / "res/tracking/urdf_path_tiny_probe/tracking_urdf_path_tiny_probe.json"
            ),
            "tracking_mjcf_stage_probe_status": tracking_mjcf_stage_probe["status"],
            "tracking_mjcf_stage_probe_current_blocker": tracking_mjcf_stage_probe["current_blocker"],
            "tracking_mjcf_stage_probe_markers": tracking_mjcf_stage_probe["markers"],
            "tracking_mjcf_stage_probe_checks": tracking_mjcf_stage_probe["checks"],
            "tracking_mjcf_stage_probe_json": str(
                ROOT / "res/tracking/mjcf_stage_probe/tracking_mjcf_stage_probe.json"
            ),
            "tracking_usd_save_policy_probe_status": tracking_usd_save_policy_probe["status"],
            "tracking_usd_save_policy_probe_current_blocker": tracking_usd_save_policy_probe["current_blocker"],
            "tracking_usd_save_policy_probe_checks": tracking_usd_save_policy_probe["checks"],
            "tracking_usd_save_policy_probe_counts": {
                "save_ok_count": tracking_usd_save_policy_probe["app_launcher"]["save_ok_count"],
                "force_save_ok_count": tracking_usd_save_policy_probe["app_launcher"]["force_save_ok_count"],
                "export_ok_count": tracking_usd_save_policy_probe["app_launcher"]["export_ok_count"],
                "permission_false_count": tracking_usd_save_policy_probe["app_launcher"]["permission_false_count"],
            },
            "tracking_usd_save_policy_probe_json": str(
                ROOT / "res/tracking/usd_save_policy_probe/tracking_usd_save_policy_probe.json"
            ),
            "tracking_simulationapp_save_policy_probe_status": tracking_simulationapp_save_policy_probe["status"],
            "tracking_simulationapp_save_policy_probe_current_blocker": tracking_simulationapp_save_policy_probe[
                "current_blocker"
            ],
            "tracking_simulationapp_save_policy_probe_checks": tracking_simulationapp_save_policy_probe["checks"],
            "tracking_simulationapp_save_policy_probe_cases": [
                {
                    "name": case["name"],
                    "returncode": case["returncode"],
                    "save_ok_count": case["save_ok_count"],
                    "permission_false_count": case["permission_false_count"],
                    "force_after_false_count": case["force_after_false_count"],
                    "markers": case["markers"],
                }
                for case in tracking_simulationapp_save_policy_probe["cases"]
            ],
            "tracking_simulationapp_save_policy_probe_json": str(
                ROOT / "res/tracking/simulationapp_save_policy_probe/tracking_simulationapp_save_policy_probe.json"
            ),
            "tracking_local_smoke_preflight_status": tracking_local_smoke_preflight["status"],
            "tracking_local_smoke_preflight_step_count": tracking_local_smoke_preflight["step_count"],
            "tracking_local_smoke_preflight_pass_count": tracking_local_smoke_preflight["pass_count"],
            "tracking_local_smoke_preflight_checks": tracking_local_smoke_preflight["checks"],
            "tracking_local_smoke_preflight_json": str(
                ROOT / "res/tracking/local_smoke_preflight/tracking_local_smoke_preflight.json"
            ),
            "deployment_audit": str(ROOT / "res/tracking/motion_tracking_controller_audit/motion_tracking_controller_audit.json"),
            "mujoco_ros_launch_contract_status": mujoco_ros_launch_contract["status"],
            "mujoco_ros_launch_contract_metrics": mujoco_ros_launch_contract["metrics"],
            "mujoco_ros_launch_contract_checks": mujoco_ros_launch_contract["checks"],
            "mujoco_ros_launch_contract_host_runtime": mujoco_ros_launch_contract["host_runtime"],
            "mujoco_ros_launch_contract_json": str(
                ROOT / "res/tracking/mujoco_ros_launch_contract_audit/mujoco_ros_launch_contract_audit.json"
            ),
            "deployment_controller_semantics_status": tracking_deployment_controller_semantics["status"],
            "deployment_controller_semantics_metrics": tracking_deployment_controller_semantics["metrics"],
            "deployment_controller_semantics_checks": tracking_deployment_controller_semantics["checks"],
            "deployment_controller_semantics_host_runtime": tracking_deployment_controller_semantics["host_runtime"],
            "deployment_controller_semantics_json": str(
                ROOT
                / "res/tracking/deployment_controller_semantics_audit/"
                / "tracking_deployment_controller_semantics_audit.json"
            ),
            "onnx_export_contract_status": tracking_onnx_contract["status"],
            "onnx_export_contract_metrics": tracking_onnx_contract["metrics"],
            "onnx_export_contract_checks": tracking_onnx_contract["checks"],
            "onnx_export_contract_json": str(
                ROOT / "res/tracking/onnx_export_contract_audit/tracking_onnx_export_contract_audit.json"
            ),
            "motion_policy_onnx_contract_fixture_status": tracking_motion_policy_onnx_fixture["status"],
            "motion_policy_onnx_contract_fixture_metrics": tracking_motion_policy_onnx_fixture["metrics"],
            "motion_policy_onnx_contract_fixture_checks": tracking_motion_policy_onnx_fixture["checks"],
            "motion_policy_onnx_contract_fixture_json": str(
                ROOT
                / "res/tracking/motion_policy_onnx_contract_fixture/"
                / "tracking_motion_policy_onnx_contract_fixture.json"
            ),
            "motion_policy_onnx_contract_fixture_npz": tracking_motion_policy_onnx_fixture["outputs"]["npz"],
            "debug_motion_policy_onnx_export_status": tracking_debug_motion_policy_onnx["status"],
            "debug_motion_policy_onnx_export_checks": tracking_debug_motion_policy_onnx["checks"],
            "debug_motion_policy_onnx_export_path": tracking_debug_motion_policy_onnx["onnx_path"],
            "debug_motion_policy_onnx_export_sha256": tracking_debug_motion_policy_onnx["onnx_sha256"],
            "debug_motion_policy_onnx_export_size_bytes": tracking_debug_motion_policy_onnx["onnx_size_bytes"],
            "debug_motion_policy_onnx_export_json": str(
                ROOT / "res/tracking/debug_motion_policy_onnx_export/tracking_debug_motion_policy_onnx_export.json"
            ),
            "debug_motion_policy_onnx_inference_status": tracking_debug_motion_policy_onnx_inference["status"],
            "debug_motion_policy_onnx_inference_checks": tracking_debug_motion_policy_onnx_inference["checks"],
            "debug_motion_policy_onnx_inference_metrics": tracking_debug_motion_policy_onnx_inference["metrics"],
            "debug_motion_policy_onnx_inference_json": str(
                ROOT
                / "res/tracking/debug_motion_policy_onnx_inference/"
                / "tracking_debug_motion_policy_onnx_inference_audit.json"
            ),
            "blocking_gates": [
                gate["gate_id"]
                for gate in blocked["gates"]
                if gate["gate_id"] in {"isaaclab_kit_inotify", "ros2_jazzy_noble_controller", "unitree_g1_hardware"}
            ],
        },
        "environments": {
            "takeover": {
                "status": takeover["status"],
                "checks": takeover["checks"],
                "command_failures": [
                    {"name": row["name"], "returncode": row["returncode"]}
                    for row in takeover.get("command_failures", [])
                ],
                "json": str(ROOT / "res/takeover_audit/takeover_audit.json"),
            },
            "env_import_probe": {
                "status": env_import_probe["status"],
                "checks": env_import_probe["checks"],
                "json": str(ROOT / "res/setup/env_probe/env_import_probe.json"),
                "log": str(ROOT / "logs/env_probe/env_import_probe.log"),
            },
            "isaaclab_live_gate_probe": {
                "status": isaaclab_live_gate_probe["status"],
                "checks": isaaclab_live_gate_probe["checks"],
                "current_blocker": isaaclab_live_gate_probe["current_blocker"],
                "json": str(ROOT / "res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json"),
                "log_dir": str(ROOT / "logs/setup/isaaclab_live_gate_probe"),
            },
            "vulkan_runtime_probe": {
                "status": vulkan_runtime_probe["status"],
                "checks": vulkan_runtime_probe["checks"],
                "json": str(ROOT / "res/setup/vulkan_runtime_probe/vulkan_runtime_probe.json"),
            },
            "cuda_p2p_runtime_probe": {
                "status": cuda_p2p_runtime_probe["status"],
                "checks": cuda_p2p_runtime_probe["checks"],
                "json": str(ROOT / "res/setup/cuda_p2p_runtime_probe/cuda_p2p_runtime_probe.json"),
            },
            "isaaclab_gpu_foundation_settings_audit": {
                "status": isaaclab_gpu_foundation_settings_audit["status"],
                "checks": isaaclab_gpu_foundation_settings_audit["checks"],
                "json": str(
                    ROOT
                    / "res/setup/isaaclab_gpu_foundation_settings_audit/isaaclab_gpu_foundation_settings_audit.json"
                ),
            },
            "bm_analysis": {
                "status": "ok",
                "path": str(ROOT / "envs/bm_analysis"),
                "lock_files": [
                    str(ROOT / "envs/bm_analysis/environment.yml"),
                    str(ROOT / "envs/bm_analysis/requirements-lock.txt"),
                    str(ROOT / "envs/bm_analysis/pip-freeze.txt"),
                    str(ROOT / "envs/bm_analysis/conda-list-explicit.txt"),
                ],
            },
            "bm_tracking": {
                "status": "partial_blocked_for_kit",
                "path": str(ROOT / "envs/bm_tracking"),
                "lock_files": [
                    str(ROOT / "envs/bm_tracking/environment.yml"),
                    str(ROOT / "envs/bm_tracking/requirements-lock.txt"),
                    str(ROOT / "envs/bm_tracking/pip-freeze.txt"),
                    str(ROOT / "envs/bm_tracking/conda-list-explicit.txt"),
                ],
            },
            "bm_diffusion": {
                "status": bm_diffusion_env["status"],
                "path": bm_diffusion_env["environment_path"],
                "checks": bm_diffusion_env["checks"],
                "status_counts": bm_diffusion_env["status_counts"],
                "json": str(ROOT / "res/setup/bm_diffusion_env_audit/bm_diffusion_env_audit.json"),
            },
        },
        "gpu_resource_monitoring": {
            "status": gpu_resource["status"],
            "rows_written": gpu_resource["rows_written"],
            "gpu_count": gpu_resource["gpu_count"],
            "nontrivial_memory_gpus": gpu_resource["nontrivial_memory_gpus"],
            "checks": gpu_resource["checks"],
            "gpu_metrics_csv": str(ROOT / "logs/gpu/gpu_metrics.csv"),
            "json": str(ROOT / "res/setup/gpu_resource_audit/gpu_resource_audit.json"),
        },
        "run_management": {
            "status": run_management["status"],
            "run_id": run_management["run_id"],
            "run_dir": run_management["run_dir"],
            "gpu_metric_rows": run_management["gpu_metric_rows"],
            "checks": run_management["checks"],
            "json": str(ROOT / "res/run_management_audit/run_management_audit.json"),
        },
        "checkpoint_resume_smoke": {
            "status": checkpoint_resume["status"],
            "run_id": checkpoint_resume["run_id"],
            "run_dir": checkpoint_resume["run_dir"],
            "checkpoint_path": checkpoint_resume["checkpoint_path"],
            "max_abs_resume_error": checkpoint_resume["metrics"]["max_abs_resume_error"],
            "checks": checkpoint_resume["checks"],
            "json": str(ROOT / "res/run_management_audit/checkpoint_resume_smoke/checkpoint_resume_smoke.json"),
            "tsv": str(ROOT / "res/run_management_audit/checkpoint_resume_smoke/checkpoint_resume_smoke.tsv"),
        },
        "full_run_deliverable_gap_audit": {
            "status": full_run_deliverable_gap["status"],
            "metrics": full_run_deliverable_gap["metrics"],
            "checks": full_run_deliverable_gap["checks"],
            "json": str(
                ROOT / "res/run_management_audit/full_run_deliverable_gap_audit/full_run_deliverable_gap_audit.json"
            ),
            "tsv": str(
                ROOT / "res/run_management_audit/full_run_deliverable_gap_audit/full_run_deliverable_gap_audit.tsv"
            ),
        },
        "failed_run_retention": {
            "status": failed_run["status"],
            "run_id": failed_run["run_id"],
            "failed_run_dir": failed_run["failed_run_dir"],
            "gpu_status_rows": failed_run["gpu_status_rows"],
            "checks": failed_run["checks"],
            "json": str(ROOT / "res/failed_runs/failed_run_audit/failed_run_audit.json"),
        },
        "official_train_entry_failed_run_retention": {
            "status": official_train_entry_failed_run["status"],
            "run_id": official_train_entry_failed_run["run_id"],
            "failed_run_dir": official_train_entry_failed_run["failed_run_dir"],
            "gpu_status_rows": official_train_entry_failed_run["gpu_status_rows"],
            "checks": official_train_entry_failed_run["checks"],
            "json": str(
                ROOT
                / "res/failed_runs/official_train_entry_failed_run_audit/"
                / "official_train_entry_failed_run_audit.json"
            ),
        },
        "patch_inventory": {
            "status": patch_inventory["status"],
            "metrics": patch_inventory["metrics"],
            "status_counts": patch_inventory["status_counts"],
            "checks": patch_inventory["checks"],
            "json": str(ROOT / "res/code/patch_inventory_audit/patch_inventory_audit.json"),
            "tsv": str(ROOT / "res/code/patch_inventory_audit/patch_inventory_audit.tsv"),
        },
        "patch_snapshot": {
            "status": patch_snapshot["status"],
            "metrics": patch_snapshot["metrics"],
            "status_counts": patch_snapshot["status_counts"],
            "checks": patch_snapshot["checks"],
            "json": str(ROOT / "res/code/patch_snapshot_audit/patch_snapshot_audit.json"),
            "tsv": str(ROOT / "res/code/patch_snapshot_audit/patch_snapshot_audit.tsv"),
            "patch_dir": patch_snapshot["outputs"]["patch_dir"],
        },
        "reimplementation_package": {
            "status": reimpl_package["status"],
            "source_root": reimpl_package["source_root"],
            "python_file_count": reimpl_package["python_file_count"],
            "symbol_row_count": reimpl_package["symbol_row_count"],
            "checks": reimpl_package["checks"],
            "json": str(ROOT / "res/code/reimpl_package_audit/reimpl_package_audit.json"),
        },
        "reimplementation_runtime_integration": {
            "status": reimpl_runtime["status"],
            "metrics": {
                "window_count": reimpl_runtime["metrics"]["window_count"],
                "token_shape": reimpl_runtime["metrics"]["token_shape"],
                "split_counts": reimpl_runtime["metrics"]["split_counts"],
                "decoded_teacher_action_mse": reimpl_runtime["metrics"]["decoded_teacher_action_mse"],
                "current_downstream_action_mse": reimpl_runtime["metrics"]["current_downstream_action_mse"],
                "predicted_action_smoothness_penalty": reimpl_runtime["metrics"][
                    "predicted_action_smoothness_penalty"
                ],
                "diffusion_mse_before": reimpl_runtime["metrics"]["diffusion_mse_before"],
                "diffusion_mse_after": reimpl_runtime["metrics"]["diffusion_mse_after"],
                "dagger_teacher_query_count": reimpl_runtime["metrics"]["dagger_teacher_query_count"],
                "tracking_mean_error": reimpl_runtime["metrics"]["tracking_mean_error"],
                "survival_rate": reimpl_runtime["metrics"]["survival_rate"],
            },
            "checks": reimpl_runtime["checks"],
            "json": str(ROOT / "res/code/reimpl_runtime_integration_audit/reimpl_runtime_integration_audit.json"),
            "tsv": str(ROOT / "res/code/reimpl_runtime_integration_audit/reimpl_runtime_integration_audit.tsv"),
        },
        "coding_requirements": {
            "status": coding_requirements["status"],
            "function_row_count": coding_requirements["function_row_count"],
            "requirement_row_count": coding_requirements["requirement_row_count"],
            "failed_requirement_count": coding_requirements["failed_requirement_count"],
            "checks": coding_requirements["checks"],
            "json": str(ROOT / "res/code/coding_requirements_audit/coding_requirements_audit.json"),
            "tsv": str(ROOT / "res/code/coding_requirements_audit/coding_requirements_audit.tsv"),
            "functions_tsv": str(ROOT / "res/code/coding_requirements_audit/coding_requirements_functions.tsv"),
        },
        "reimpl_package_api_tests": {
            "status": reimpl_package_api_tests["status"],
            "row_count": reimpl_package_api_tests["row_count"],
            "failed_row_count": reimpl_package_api_tests["failed_row_count"],
            "covered_goal_items": reimpl_package_api_tests["covered_goal_items"],
            "checks": reimpl_package_api_tests["checks"],
            "json": str(ROOT / "res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json"),
            "tsv": str(ROOT / "res/tests/reimpl_package_api_tests/reimpl_package_api_tests.tsv"),
        },
        "reimpl_test_suite": {
            "status": reimpl_test_suite["status"],
            "step_count": reimpl_test_suite["step_count"],
            "pass_count": reimpl_test_suite["pass_count"],
            "metrics": reimpl_test_suite["metrics"],
            "checks": reimpl_test_suite["checks"],
            "json": str(ROOT / "res/tests/reimpl_test_suite/reimpl_test_suite.json"),
            "tsv": str(ROOT / "res/tests/reimpl_test_suite/reimpl_test_suite.tsv"),
        },
        "resolved_reproduction_config": {
            "status": resolved_config["status"],
            "tracking_control_frequency_hz": resolved_config["tracking"]["control_frequency_hz"],
            "tracking_ppo_max_iterations": resolved_config["tracking"]["ppo"]["max_iterations"],
            "vae_latent_dim": resolved_config["vae"]["latent_dim"],
            "diffusion_batch_size": resolved_config["diffusion"]["training"]["batch_size"],
            "diffusion_denoising_steps": resolved_config["diffusion"]["denoising_steps"],
            "checks": resolved_config["checks"],
            "json": str(ROOT / "res/config/resolved_reproduction_config.json"),
            "yaml": str(ROOT / "res/config/resolved_reproduction_config.yaml"),
            "csv": str(ROOT / "res/config/resolved_reproduction_config.csv"),
        },
        "artifact_manifest": {
            "status": artifact_manifest["status"],
            "artifact_count": artifact_manifest["artifact_count"],
            "missing_count": artifact_manifest["missing_count"],
            "category_counts": artifact_manifest["category_counts"],
            "checks": artifact_manifest["checks"],
            "json": str(ROOT / "res/artifact_manifest/artifact_manifest.json"),
            "tsv": str(ROOT / "res/artifact_manifest/artifact_manifest.tsv"),
        },
        "download_source_integrity": {
            "status": download_source_integrity["status"],
            "file_count": download_source_integrity["file_count"],
            "total_size_bytes": download_source_integrity["total_size_bytes"],
            "downloaded_files_manifest_row_count": download_source_integrity[
                "downloaded_files_manifest_row_count"
            ],
            "required_hash_file_count": download_source_integrity["required_hash_file_count"],
            "reference_hash_file_count": download_source_integrity["reference_hash_file_count"],
            "category_counts": download_source_integrity["category_counts"],
            "checks": download_source_integrity["checks"],
            "json": str(
                ROOT
                / "res/source_integrity/download_source_integrity/download_source_integrity_audit.json"
            ),
            "tsv": str(
                ROOT
                / "res/source_integrity/download_source_integrity/download_source_integrity_manifest.tsv"
            ),
            "required_tsv": str(
                ROOT
                / "res/source_integrity/download_source_integrity/download_source_integrity_required.tsv"
            ),
        },
        "run_log_config_catalog": {
            "status": run_log_config_catalog["status"],
            "metrics": run_log_config_catalog["metrics"],
            "category_counts": run_log_config_catalog["category_counts"],
            "checks": run_log_config_catalog["checks"],
            "json": str(ROOT / "res/run_log_config_catalog/run_log_config_catalog.json"),
            "csv": str(ROOT / "res/run_log_config_catalog/run_log_config_catalog.csv"),
            "markdown": str(ROOT / "res/run_log_config_catalog/run_log_config_catalog.md"),
        },
        "experiment_protocol": {
            "status": experiment_protocol["status"],
            "document": experiment_protocol["document"],
            "row_count": experiment_protocol["row_count"],
            "missing_count": experiment_protocol["missing_count"],
            "checks": experiment_protocol["checks"],
            "json": str(ROOT / "res/docs/experiment_protocol_audit/experiment_protocol_audit.json"),
        },
        "top_level_readme": {
            "status": readme_audit["status"],
            "document": readme_audit["document"],
            "row_count": readme_audit["row_count"],
            "missing_count": readme_audit["missing_count"],
            "checks": readme_audit["checks"],
            "json": str(ROOT / "res/docs/readme_audit/readme_audit.json"),
        },
        "final_deliverables": {
            "status": final_deliverables["status"],
            "row_count": final_deliverables["row_count"],
            "category_counts": final_deliverables["category_counts"],
            "status_counts": final_deliverables["status_counts"],
            "missing_evidence_rows": len(final_deliverables["missing_evidence_rows"]),
            "blocked_or_missing_rows": final_deliverables["blocked_or_missing_rows"],
            "checks": final_deliverables["checks"],
            "json": str(ROOT / "res/final_deliverables_audit/final_deliverables_audit.json"),
            "tsv": str(ROOT / "res/final_deliverables_audit/final_deliverables_audit.tsv"),
        },
        "visual_media_inventory": {
            "status": visual_media_inventory["status"],
            "row_count": visual_media_inventory["row_count"],
            "kind_counts": visual_media_inventory["kind_counts"],
            "category_counts": visual_media_inventory["category_counts"],
            "checks": visual_media_inventory["checks"],
            "json": str(ROOT / "res/visual_media_inventory/visual_media_inventory_audit.json"),
            "tsv": str(ROOT / "res/visual_media_inventory/visual_media_inventory_audit.tsv"),
        },
        "verification_command_coverage": {
            "status": verification_command_coverage["status"],
            "command_count": verification_command_coverage["command_count"],
            "category_counts": verification_command_coverage["category_counts"],
            "smoke_command_count": verification_command_coverage["smoke_command_count"],
            "smoke_pass_count": verification_command_coverage["smoke_pass_count"],
            "checks": verification_command_coverage["checks"],
            "json": str(ROOT / "res/verification_command_coverage/verification_command_coverage_audit.json"),
            "tsv": str(ROOT / "res/verification_command_coverage/verification_command_coverage_audit.tsv"),
        },
        "verification_command_syntax": {
            "status": verification_command_syntax["status"],
            "python_script_count": verification_command_syntax["python_script_count"],
            "failed_count": verification_command_syntax["failed_count"],
            "checks": verification_command_syntax["checks"],
            "json": str(ROOT / "res/verification_command_syntax/verification_command_syntax_audit.json"),
            "tsv": str(ROOT / "res/verification_command_syntax/verification_command_syntax_audit.tsv"),
        },
        "verification_command_script_manifest": {
            "status": verification_command_script_manifest["status"],
            "python_script_count": verification_command_script_manifest["python_script_count"],
            "checks": verification_command_script_manifest["checks"],
            "json": str(
                ROOT
                / "res/verification_command_script_manifest/verification_command_script_manifest.json"
            ),
            "tsv": str(
                ROOT
                / "res/verification_command_script_manifest/verification_command_script_manifest.tsv"
            ),
        },
        "required_artifact_absence": {
            "status": required_artifact_absence["status"],
            "row_count": required_artifact_absence["row_count"],
            "status_counts": required_artifact_absence["status_counts"],
            "local_scan_counts": required_artifact_absence["local_scan_counts"],
            "download_reference_counts": required_artifact_absence["download_reference_counts"],
            "checks": required_artifact_absence["checks"],
            "json": str(ROOT / "res/required_artifact_absence/required_artifact_absence_audit.json"),
            "tsv": str(ROOT / "res/required_artifact_absence/required_artifact_absence_audit.tsv"),
        },
        "evaluation_metrics_coverage": {
            "status": evaluation_metrics["status"],
            "row_count": evaluation_metrics["row_count"],
            "status_counts": evaluation_metrics["status_counts"],
            "section_counts": evaluation_metrics["section_counts"],
            "missing_evidence_rows": len(evaluation_metrics["missing_evidence_rows"]),
            "checks": evaluation_metrics["checks"],
            "json": str(ROOT / "res/evaluation_metrics_coverage/evaluation_metrics_coverage_audit.json"),
            "tsv": str(ROOT / "res/evaluation_metrics_coverage/evaluation_metrics_coverage_audit.tsv"),
        },
        "trial_failure_accounting": {
            "status": trial_failure_accounting["status"],
            "metrics": trial_failure_accounting["metrics"],
            "status_counts": trial_failure_accounting["status_counts"],
            "checks": trial_failure_accounting["checks"],
            "json": str(ROOT / "res/evaluation/trial_failure_accounting_audit/trial_failure_accounting_audit.json"),
            "tsv": str(ROOT / "res/evaluation/trial_failure_accounting_audit/trial_failure_accounting_audit.tsv"),
        },
        "metrics_catalog": {
            "status": metrics_catalog["status"],
            "metrics": metrics_catalog["metrics"],
            "level_counts": metrics_catalog["level_counts"],
            "checks": metrics_catalog["checks"],
            "json": str(ROOT / "res/metrics/metrics_catalog/metrics_catalog.json"),
            "csv": str(ROOT / "res/metrics/metrics_catalog/metrics_catalog.csv"),
            "markdown": str(ROOT / "res/metrics/metrics_catalog/metrics_catalog.md"),
        },
        "ablation_coverage": {
            "status": ablation_coverage["status"],
            "row_count": ablation_coverage["row_count"],
            "status_counts": ablation_coverage["status_counts"],
            "group_counts": ablation_coverage["group_counts"],
            "missing_evidence_rows": len(ablation_coverage["missing_evidence_rows"]),
            "symmetry_training_comparison_status": lafan1_paper_arch_symmetry_training_comparison["status"],
            "symmetry_training_comparison_metrics": lafan1_paper_arch_symmetry_training_comparison["metrics"],
            "symmetry_training_comparison_checks": lafan1_paper_arch_symmetry_training_comparison["checks"],
            "symmetry_training_comparison_outputs": lafan1_paper_arch_symmetry_training_comparison["outputs"],
            "checks": ablation_coverage["checks"],
            "json": str(ROOT / "res/ablation_coverage/ablation_coverage_audit.json"),
            "tsv": str(ROOT / "res/ablation_coverage/ablation_coverage_audit.tsv"),
        },
        "guidance_task_coverage": {
            "status": guidance_task_coverage["status"],
            "row_count": guidance_task_coverage["row_count"],
            "status_counts": guidance_task_coverage["status_counts"],
            "task_counts": guidance_task_coverage["task_counts"],
            "requirement_counts": guidance_task_coverage["requirement_counts"],
            "missing_evidence_rows": len(guidance_task_coverage["missing_evidence_rows"]),
            "checks": guidance_task_coverage["checks"],
            "json": str(ROOT / "res/guidance_task_coverage/guidance_task_coverage_audit.json"),
            "tsv": str(ROOT / "res/guidance_task_coverage/guidance_task_coverage_audit.tsv"),
        },
        "progress_report": {
            "status": progress_audit["status"],
            "required_field_count": progress_audit["required_field_count"],
            "progress_marker_count": progress_audit["progress_marker_count"],
            "missing_count": progress_audit["missing_count"],
            "master_count_mentions": progress_audit["master_count_mentions"],
            "checks": progress_audit["checks"],
            "json": str(ROOT / "res/progress_report_audit/progress_report_audit.json"),
            "tsv": str(ROOT / "res/progress_report_audit/progress_report_audit.tsv"),
        },
        "completion_matrix_status": {
            "status": completion_matrix_status["status"],
            "row_count": completion_matrix_status["row_count"],
            "header_count": completion_matrix_status["header_count"],
            "invalid_status_count": completion_matrix_status["invalid_status_count"],
            "invalid_row_count": completion_matrix_status["invalid_row_count"],
            "status_counts": completion_matrix_status["status_counts"],
            "checks": completion_matrix_status["checks"],
            "json": str(ROOT / "res/docs/completion_matrix_status_audit/completion_matrix_status_audit.json"),
            "tsv": str(ROOT / "res/docs/completion_matrix_status_audit/completion_matrix_status_audit.tsv"),
        },
        "project_boundary": {
            "status": project_boundary["status"],
            "row_count": project_boundary["row_count"],
            "failed_count": project_boundary["failed_count"],
            "download_toplevel_entries": project_boundary["download_toplevel_entries"],
            "generated_root_counts": project_boundary["generated_root_counts"],
            "checks": project_boundary["checks"],
            "json": str(ROOT / "res/project_boundary_audit/project_boundary_audit.json"),
            "tsv": str(ROOT / "res/project_boundary_audit/project_boundary_audit.tsv"),
        },
        "core_test_coverage": {
            "status": core_test_coverage["status"],
            "required_count": core_test_coverage["required_count"],
            "missing_count": core_test_coverage["missing_count"],
            "core_test_row_count": core_test_coverage["core_test_row_count"],
            "core_test_failed_row_count": core_test_coverage["core_test_failed_row_count"],
            "checks": core_test_coverage["checks"],
            "json": str(ROOT / "res/tests/core_test_coverage_audit/core_test_coverage_audit.json"),
            "tsv": str(ROOT / "res/tests/core_test_coverage_audit/core_test_coverage_audit.tsv"),
        },
        "level_c_diffusion": {
            "status": "debug_mechanics_and_audits_only",
            "official_level_c_code_found": official_artifacts["conclusion"]["official_beyondmimic_vae_diffusion_code_found"],
            "official_level_c_checkpoint_or_engine_found": official_artifacts["conclusion"][
                "official_beyondmimic_checkpoint_or_engine_found"
            ],
            "fig5_fig6_possible_from_current_artifacts": fig56["conclusion"][
                "fig5_fig6_paper_reproduction_possible_from_current_local_artifacts"
            ],
            "debug_suite_status": level_c_debug_suite["status"],
            "debug_suite_step_count": level_c_debug_suite["step_count"],
            "debug_suite_pass_count": level_c_debug_suite["pass_count"],
            "debug_suite_metrics": level_c_debug_suite["metrics"],
            "debug_suite_checks": level_c_debug_suite["checks"],
            "debug_suite_json": str(ROOT / "res/level_c/debug_suite/level_c_debug_suite.json"),
            "extended_debug_suite_status": level_c_extended_debug_suite["status"],
            "extended_debug_suite_step_count": level_c_extended_debug_suite["step_count"],
            "extended_debug_suite_pass_count": level_c_extended_debug_suite["pass_count"],
            "extended_debug_suite_metrics": level_c_extended_debug_suite["metrics"],
            "extended_debug_suite_checks": level_c_extended_debug_suite["checks"],
            "extended_debug_suite_json": str(
                ROOT / "res/level_c/extended_debug_suite/level_c_extended_debug_suite.json"
            ),
            "diffusion_equation_audit_status": diffusion_equations["status"],
            "diffusion_exact_coefficient_schedule_missing": diffusion_equations["checks"][
                "public_source_exact_coefficient_schedule_missing"
            ],
            "diffusion_equation_audit_json": str(
                ROOT / "res/level_c/diffusion_equation_audit/level_c_diffusion_equation_audit.json"
            ),
            "trajectory_inverse_transform_audit_status": inverse_transform["status"],
            "trajectory_inverse_transform_checks": inverse_transform["checks"],
            "trajectory_inverse_transform_json": str(
                ROOT / "res/level_c/trajectory_inverse_transform_audit/level_c_trajectory_inverse_transform_audit.json"
            ),
            "emphasis_projection_audit_status": emphasis_projection["status"],
            "emphasis_projection_audit_metrics": emphasis_projection["metrics"],
            "emphasis_projection_audit_checks": emphasis_projection["checks"],
            "emphasis_projection_audit_json": str(
                ROOT / "res/level_c/emphasis_projection_audit/level_c_emphasis_projection_audit.json"
            ),
            "state_representation_source_audit_status": state_representation["status"],
            "state_representation_source_audit_metrics": state_representation["aggregate_metrics"],
            "state_representation_source_audit_checks": state_representation["checks"],
            "state_representation_source_audit_json": str(
                ROOT / "res/level_c/state_representation_source_audit/level_c_state_representation_source_audit.json"
            ),
            "dataset_collection_protocol_audit_status": dataset_collection_protocol["status"],
            "dataset_collection_protocol_audit_metrics": dataset_collection_protocol["metrics"],
            "dataset_collection_protocol_audit_checks": dataset_collection_protocol["checks"],
            "dataset_collection_protocol_audit_json": str(
                ROOT
                / "res/level_c/dataset_collection_protocol_audit/"
                / "level_c_dataset_collection_protocol_audit.json"
            ),
            "rollout_rejection_manifest_probe_status": rollout_rejection_manifest["status"],
            "rollout_rejection_manifest_probe_metrics": rollout_rejection_manifest["metrics"],
            "rollout_rejection_manifest_probe_checks": rollout_rejection_manifest["checks"],
            "rollout_rejection_manifest_probe_json": str(
                ROOT
                / "res/level_c/rollout_rejection_manifest_probe/"
                / "level_c_rollout_rejection_manifest_probe.json"
            ),
            "state_latent_schema_audit_status": state_latent_schema["status"],
            "state_latent_schema_audit_row_count": state_latent_schema["row_count"],
            "state_latent_schema_audit_split_counts": state_latent_schema["split_counts"],
            "state_latent_schema_audit_token_shape_counts": state_latent_schema["token_shape_counts"],
            "state_latent_schema_audit_latent_source_counts": state_latent_schema["latent_source_counts"],
            "state_latent_schema_audit_checks": state_latent_schema["checks"],
            "state_latent_schema_audit_json": str(
                ROOT / "res/level_c/state_latent_schema_audit/state_latent_schema_audit.json"
            ),
            "dagger_schema_audit_status": dagger_schema["status"],
            "dagger_schema_audit_row_count": dagger_schema["row_count"],
            "dagger_schema_audit_split_counts": dagger_schema["split_counts"],
            "dagger_schema_audit_metrics": dagger_schema["metrics"],
            "dagger_schema_audit_checks": dagger_schema["checks"],
            "dagger_schema_audit_json": str(ROOT / "res/level_c/dagger_schema_audit/dagger_schema_audit.json"),
            "dagger_iteration_smoke_status": dagger_iteration["status"],
            "dagger_iteration_smoke_metrics": dagger_iteration["metrics"],
            "dagger_iteration_smoke_checks": dagger_iteration["checks"],
            "dagger_iteration_smoke_json": str(
                ROOT / "res/level_c/dagger_iteration_smoke/level_c_dagger_iteration_smoke.json"
            ),
            "paper_state_windows_status": paper_state_windows["status"],
            "paper_state_windows_counts": paper_state_windows["counts"],
            "paper_state_windows_checks": paper_state_windows["checks"],
            "paper_state_windows_json": str(
                ROOT / "res/level_c/paper_state_windows/level_c_paper_state_windows.json"
            ),
            "state_latent_dataset_consistency_status": state_latent_dataset_consistency["status"],
            "state_latent_dataset_consistency_metrics": {
                "row_count": state_latent_dataset_consistency["metrics"]["row_count"],
                "state_dim": state_latent_dataset_consistency["metrics"]["state_dim"],
                "latent_dim": state_latent_dataset_consistency["metrics"]["latent_dim"],
                "token_dim": state_latent_dataset_consistency["metrics"]["token_dim"],
                "action_dim": state_latent_dataset_consistency["metrics"]["action_dim"],
                "per_split_counts": state_latent_dataset_consistency["metrics"]["per_split_counts"],
                "max_state_abs_error_between_paper_windows_and_vae_npz": state_latent_dataset_consistency[
                    "metrics"
                ]["max_state_abs_error_between_paper_windows_and_vae_npz"],
                "max_target_action_abs_error_between_action_npz_and_decoded_action": (
                    state_latent_dataset_consistency["metrics"][
                        "max_target_action_abs_error_between_action_npz_and_decoded_action"
                    ]
                ),
                "latent_abs_mean": state_latent_dataset_consistency["metrics"]["latent_abs_mean"],
            },
            "state_latent_dataset_consistency_checks": state_latent_dataset_consistency["checks"],
            "state_latent_dataset_consistency_json": str(
                ROOT
                / "res/level_c/state_latent_dataset_consistency_audit/"
                / "level_c_state_latent_dataset_consistency_audit.json"
            ),
            "state_latent_training_dataset_contract_status": state_latent_training_dataset_contract["status"],
            "state_latent_training_dataset_contract_metrics": state_latent_training_dataset_contract["metrics"],
            "state_latent_training_dataset_contract_checks": state_latent_training_dataset_contract["checks"],
            "state_latent_training_dataset_contract_json": str(
                ROOT
                / "res/level_c/state_latent_training_dataset_contract_audit/"
                / "level_c_state_latent_training_dataset_contract_audit.json"
            ),
            "paper_state_overfit_status": paper_state_overfit["status"],
            "paper_state_overfit_metrics": {
                "all_paper_state_windows_baseline_loss": paper_state_overfit["metrics"][
                    "all_paper_state_windows_baseline_loss"
                ],
                "all_paper_state_windows_overfit_loss": paper_state_overfit["metrics"][
                    "all_paper_state_windows_overfit_loss"
                ],
                "all_paper_state_windows_loss_reduction_ratio": paper_state_overfit["metrics"][
                    "all_paper_state_windows_loss_reduction_ratio"
                ],
                "paper_state_dim": paper_state_overfit["settings"]["paper_state_dim"],
                "token_dim": paper_state_overfit["settings"]["token_dim"],
            },
            "paper_state_overfit_json": str(
                ROOT / "res/level_c/paper_state_overfit_probe/level_c_paper_state_overfit_probe.json"
            ),
            "vae_latent_diffusion_overfit_status": vae_latent_diffusion_overfit["status"],
            "vae_latent_diffusion_overfit_metrics": {
                "all_debug_vae_latent_windows_baseline_loss": vae_latent_diffusion_overfit["metrics"][
                    "all_debug_vae_latent_windows_baseline_loss"
                ],
                "all_debug_vae_latent_windows_overfit_loss": vae_latent_diffusion_overfit["metrics"][
                    "all_debug_vae_latent_windows_overfit_loss"
                ],
                "all_debug_vae_latent_windows_loss_reduction_ratio": vae_latent_diffusion_overfit["metrics"][
                    "all_debug_vae_latent_windows_loss_reduction_ratio"
                ],
                "token_dim": vae_latent_diffusion_overfit["settings"]["token_dim"],
                "latent_source": vae_latent_diffusion_overfit["settings"]["latent_source"],
            },
            "vae_latent_diffusion_overfit_checks": vae_latent_diffusion_overfit["checks"],
            "vae_latent_diffusion_overfit_json": str(
                ROOT
                / "res/level_c/vae_latent_diffusion_overfit_probe/level_c_vae_latent_diffusion_overfit_probe.json"
            ),
            "paper_state_heldout_status": paper_state_heldout["status"],
            "paper_state_heldout_metrics": {
                "train_prediction_loss": paper_state_heldout["metrics"]["train_prediction_loss"],
                "validation_prediction_loss": paper_state_heldout["metrics"]["validation_prediction_loss"],
                "test_prediction_loss": paper_state_heldout["metrics"]["test_prediction_loss"],
                "validation_loss_reduction_ratio": paper_state_heldout["metrics"]["validation_loss_reduction_ratio"],
                "test_loss_reduction_ratio": paper_state_heldout["metrics"]["test_loss_reduction_ratio"],
                "paper_state_dim": paper_state_heldout["settings"]["paper_state_dim"],
                "token_dim": paper_state_heldout["settings"]["token_dim"],
            },
            "paper_state_heldout_json": str(
                ROOT / "res/level_c/paper_state_heldout_eval/level_c_paper_state_heldout_eval.json"
            ),
            "vae_latent_heldout_status": vae_latent_heldout["status"],
            "vae_latent_heldout_metrics": {
                "train_prediction_loss": vae_latent_heldout["metrics"]["train_prediction_loss"],
                "validation_prediction_loss": vae_latent_heldout["metrics"]["validation_prediction_loss"],
                "test_prediction_loss": vae_latent_heldout["metrics"]["test_prediction_loss"],
                "validation_loss_reduction_ratio": vae_latent_heldout["metrics"][
                    "validation_loss_reduction_ratio"
                ],
                "test_loss_reduction_ratio": vae_latent_heldout["metrics"]["test_loss_reduction_ratio"],
                "token_dim": vae_latent_heldout["settings"]["token_dim"],
                "latent_source": vae_latent_heldout["settings"]["latent_source"],
            },
            "vae_latent_heldout_checks": vae_latent_heldout["checks"],
            "vae_latent_heldout_json": str(
                ROOT / "res/level_c/vae_latent_heldout_eval/level_c_vae_latent_heldout_eval.json"
            ),
            "paper_state_heldout_multiseed_status": paper_state_heldout_multiseed["status"],
            "paper_state_heldout_multiseed_statistics": {
                "validation_prediction_loss": paper_state_heldout_multiseed["statistics"][
                    "validation_prediction_loss"
                ],
                "test_prediction_loss": paper_state_heldout_multiseed["statistics"]["test_prediction_loss"],
                "validation_loss_reduction_ratio": paper_state_heldout_multiseed["statistics"][
                    "validation_loss_reduction_ratio"
                ],
                "test_loss_reduction_ratio": paper_state_heldout_multiseed["statistics"][
                    "test_loss_reduction_ratio"
                ],
            },
            "paper_state_heldout_multiseed_json": str(
                ROOT
                / "res/level_c/paper_state_heldout_multiseed_audit/"
                / "level_c_paper_state_heldout_multiseed_audit.json"
            ),
            "vae_latent_heldout_multiseed_status": vae_latent_heldout_multiseed["status"],
            "vae_latent_heldout_multiseed_statistics": {
                "validation_prediction_loss": vae_latent_heldout_multiseed["statistics"][
                    "validation_prediction_loss"
                ],
                "test_prediction_loss": vae_latent_heldout_multiseed["statistics"]["test_prediction_loss"],
                "validation_loss_reduction_ratio": vae_latent_heldout_multiseed["statistics"][
                    "validation_loss_reduction_ratio"
                ],
                "test_loss_reduction_ratio": vae_latent_heldout_multiseed["statistics"][
                    "test_loss_reduction_ratio"
                ],
            },
            "vae_latent_heldout_multiseed_checks": vae_latent_heldout_multiseed["checks"],
            "vae_latent_heldout_multiseed_json": str(
                ROOT
                / "res/level_c/vae_latent_heldout_multiseed_audit/"
                / "level_c_vae_latent_heldout_multiseed_audit.json"
            ),
            "paper_state_transformer_arch_probe_status": paper_state_transformer["status"],
            "paper_state_transformer_arch_probe_settings": {
                "state_dim": paper_state_transformer["settings"]["state_dim"],
                "token_dim": paper_state_transformer["settings"]["token_dim"],
                "embedding_dim": paper_state_transformer["settings"]["embedding_dim"],
                "attention_heads": paper_state_transformer["settings"]["attention_heads"],
                "transformer_layers": paper_state_transformer["settings"]["transformer_layers"],
                "denoising_steps": paper_state_transformer["settings"]["denoising_steps"],
            },
            "paper_state_transformer_arch_probe_metrics": {
                "parameter_count": paper_state_transformer["model"]["parameter_count"],
                "clean_trajectory_mse": paper_state_transformer["metrics"]["clean_trajectory_mse"],
                "total_grad_norm": paper_state_transformer["metrics"]["total_grad_norm"],
                "cuda_peak_memory_mb": paper_state_transformer["metrics"]["cuda_peak_memory_mb"],
            },
            "paper_state_transformer_arch_probe_checks": paper_state_transformer["checks"],
            "paper_state_transformer_arch_probe_json": str(
                ROOT
                / "res/level_c/paper_state_transformer_arch_probe/"
                / "level_c_paper_state_transformer_arch_probe.json"
            ),
            "vae_latent_transformer_arch_probe_status": vae_latent_transformer["status"],
            "vae_latent_transformer_arch_probe_settings": {
                "state_dim": vae_latent_transformer["settings"]["state_dim"],
                "token_dim": vae_latent_transformer["settings"]["token_dim"],
                "latent_source": vae_latent_transformer["settings"]["latent_source"],
                "embedding_dim": vae_latent_transformer["settings"]["embedding_dim"],
                "attention_heads": vae_latent_transformer["settings"]["attention_heads"],
                "transformer_layers": vae_latent_transformer["settings"]["transformer_layers"],
                "denoising_steps": vae_latent_transformer["settings"]["denoising_steps"],
            },
            "vae_latent_transformer_arch_probe_metrics": {
                "parameter_count": vae_latent_transformer["model"]["parameter_count"],
                "clean_trajectory_mse": vae_latent_transformer["metrics"]["clean_trajectory_mse"],
                "total_grad_norm": vae_latent_transformer["metrics"]["total_grad_norm"],
                "cuda_peak_memory_mb": vae_latent_transformer["metrics"]["cuda_peak_memory_mb"],
            },
            "vae_latent_transformer_arch_probe_checks": vae_latent_transformer["checks"],
            "vae_latent_transformer_arch_probe_json": str(
                ROOT
                / "res/level_c/vae_latent_transformer_arch_probe/"
                / "level_c_vae_latent_transformer_arch_probe.json"
            ),
            "transformer_parameter_count_audit_status": transformer_parameter_count["status"],
            "transformer_parameter_count_audit_metrics": transformer_parameter_count["metrics"],
            "transformer_parameter_count_audit_checks": transformer_parameter_count["checks"],
            "transformer_parameter_count_audit_json": str(
                ROOT
                / "res/level_c/transformer_parameter_count_audit/"
                / "level_c_transformer_parameter_count_audit.json"
            ),
            "transformer_state_dict_manifest_status": transformer_state_dict["status"],
            "transformer_state_dict_manifest_metrics": transformer_state_dict["metrics"],
            "transformer_state_dict_manifest_checks": transformer_state_dict["checks"],
            "transformer_state_dict_manifest_json": str(
                ROOT
                / "res/level_c/transformer_state_dict_manifest/"
                / "level_c_transformer_state_dict_manifest.json"
            ),
            "transformer_ema_smoke_status": transformer_ema["status"],
            "transformer_ema_smoke_metrics": transformer_ema["metrics"],
            "transformer_ema_smoke_checks": transformer_ema["checks"],
            "transformer_ema_smoke_json": str(
                ROOT / "res/level_c/transformer_ema_smoke/level_c_transformer_ema_smoke.json"
            ),
            "vae_latent_transformer_ema_smoke_status": vae_latent_transformer_ema["status"],
            "vae_latent_transformer_ema_smoke_metrics": vae_latent_transformer_ema["metrics"],
            "vae_latent_transformer_ema_smoke_checks": vae_latent_transformer_ema["checks"],
            "vae_latent_transformer_ema_smoke_json": str(
                ROOT
                / "res/level_c/vae_latent_transformer_ema_smoke/"
                / "level_c_vae_latent_transformer_ema_smoke.json"
            ),
            "diffusion_checkpoint_smoke_status": diffusion_checkpoint_smoke["status"],
            "diffusion_checkpoint_smoke_metrics": {
                "checkpoint_size_bytes": diffusion_checkpoint_smoke["metrics"]["checkpoint_size_bytes"],
                "eval_max_abs_error_after_resume": diffusion_checkpoint_smoke["metrics"][
                    "eval_max_abs_error_after_resume"
                ],
                "model_l2_after_resume": diffusion_checkpoint_smoke["metrics"]["model_l2_after_resume"],
                "ema_l2_after_resume": diffusion_checkpoint_smoke["metrics"]["ema_l2_after_resume"],
                "final_uninterrupted_loss_after": diffusion_checkpoint_smoke["metrics"][
                    "final_uninterrupted_loss_after"
                ],
                "final_resumed_loss_after": diffusion_checkpoint_smoke["metrics"]["final_resumed_loss_after"],
            },
            "diffusion_checkpoint_smoke_checks": diffusion_checkpoint_smoke["checks"],
            "diffusion_checkpoint_smoke_checkpoint": diffusion_checkpoint_smoke["outputs"]["checkpoint"],
            "diffusion_checkpoint_smoke_json": str(
                ROOT / "res/level_c/diffusion_checkpoint_smoke/level_c_diffusion_checkpoint_smoke.json"
            ),
            "bounded_debug_diffusion_training_run_status": bounded_debug_diffusion_training_run["status"],
            "bounded_debug_diffusion_training_run_metrics": {
                "debug_step_count": bounded_debug_diffusion_training_run["metrics"]["debug_step_count"],
                "debug_token_dim": bounded_debug_diffusion_training_run["metrics"]["debug_token_dim"],
                "parameter_count": bounded_debug_diffusion_training_run["metrics"]["parameter_count"],
                "initial_loss_before": bounded_debug_diffusion_training_run["metrics"]["initial_loss_before"],
                "final_loss_after": bounded_debug_diffusion_training_run["metrics"]["final_loss_after"],
                "checkpoint_size_bytes": bounded_debug_diffusion_training_run["metrics"]["checkpoint_size_bytes"],
                "loss_figure_size_bytes": bounded_debug_diffusion_training_run["metrics"]["loss_figure_size_bytes"],
                "is_training_run": bounded_debug_diffusion_training_run["metrics"]["is_training_run"],
                "paper_level": bounded_debug_diffusion_training_run["metrics"]["paper_level"],
            },
            "bounded_debug_diffusion_training_run_checks": bounded_debug_diffusion_training_run["checks"],
            "bounded_debug_diffusion_training_run_checkpoint": bounded_debug_diffusion_training_run["outputs"][
                "checkpoint"
            ],
            "bounded_debug_diffusion_training_run_figure": bounded_debug_diffusion_training_run["outputs"]["figure"],
            "bounded_debug_diffusion_training_run_run_dir": bounded_debug_diffusion_training_run["outputs"]["run_dir"],
            "bounded_debug_diffusion_training_run_json": str(
                ROOT
                / "res/level_c/bounded_debug_diffusion_training_run/"
                / "level_c_bounded_debug_diffusion_training_run.json"
            ),
            "bounded_debug_diffusion_checkpoint_eval_status": bounded_debug_diffusion_checkpoint_eval["status"],
            "bounded_debug_diffusion_checkpoint_eval_rows": bounded_debug_diffusion_checkpoint_eval["rows"],
            "bounded_debug_diffusion_checkpoint_eval_metrics": bounded_debug_diffusion_checkpoint_eval["metrics"],
            "bounded_debug_diffusion_checkpoint_eval_checks": bounded_debug_diffusion_checkpoint_eval["checks"],
            "bounded_debug_diffusion_checkpoint_eval_json": str(
                ROOT
                / "res/level_c/bounded_debug_diffusion_checkpoint_eval/"
                / "level_c_bounded_debug_diffusion_checkpoint_eval.json"
            ),
            "bounded_debug_diffusion_action_eval_status": bounded_debug_diffusion_action_eval["status"],
            "bounded_debug_diffusion_action_eval_rows": bounded_debug_diffusion_action_eval["rows"],
            "bounded_debug_diffusion_action_eval_metrics": {
                "validation_checkpoint_current_action_mse": bounded_debug_diffusion_action_eval["metrics"][
                    "validation_checkpoint_current_action_mse"
                ],
                "test_checkpoint_current_action_mse": bounded_debug_diffusion_action_eval["metrics"][
                    "test_checkpoint_current_action_mse"
                ],
                "validation_checkpoint_full_action_mse": bounded_debug_diffusion_action_eval["metrics"][
                    "validation_checkpoint_full_action_mse"
                ],
                "test_checkpoint_full_action_mse": bounded_debug_diffusion_action_eval["metrics"][
                    "test_checkpoint_full_action_mse"
                ],
            },
            "bounded_debug_diffusion_action_eval_checks": bounded_debug_diffusion_action_eval["checks"],
            "bounded_debug_diffusion_action_eval_json": str(
                ROOT
                / "res/level_c/bounded_debug_diffusion_action_eval/"
                / "level_c_bounded_debug_diffusion_action_eval.json"
            ),
            "resource_adjusted_tiny_diffusion_status": resource_adjusted_tiny_diffusion["status"],
            "resource_adjusted_tiny_diffusion_metrics": {
                "epochs": resource_adjusted_tiny_diffusion["metrics"]["epochs"],
                "parameter_count": resource_adjusted_tiny_diffusion["metrics"]["parameter_count"],
                "train_pred_token_mse": resource_adjusted_tiny_diffusion["metrics"]["train_pred_token_mse"],
                "validation_pred_token_mse": resource_adjusted_tiny_diffusion["metrics"][
                    "validation_pred_token_mse"
                ],
                "test_pred_token_mse": resource_adjusted_tiny_diffusion["metrics"]["test_pred_token_mse"],
                "validation_pred_current_action_mse": resource_adjusted_tiny_diffusion["metrics"][
                    "validation_pred_current_action_mse"
                ],
                "test_pred_current_action_mse": resource_adjusted_tiny_diffusion["metrics"][
                    "test_pred_current_action_mse"
                ],
                "checkpoint_size_bytes": resource_adjusted_tiny_diffusion["metrics"]["checkpoint_size_bytes"],
            },
            "resource_adjusted_tiny_diffusion_checks": resource_adjusted_tiny_diffusion["checks"],
            "resource_adjusted_tiny_diffusion_run_dir": resource_adjusted_tiny_diffusion["outputs"]["run_dir"],
            "resource_adjusted_tiny_diffusion_checkpoint": resource_adjusted_tiny_diffusion["outputs"][
                "checkpoint"
            ],
            "resource_adjusted_tiny_diffusion_json": str(
                ROOT
                / "res/level_c/resource_adjusted_tiny_diffusion_training_run/"
                / "level_c_resource_adjusted_tiny_diffusion_training_run.json"
            ),
            "resource_adjusted_tiny_suite_status": resource_adjusted_tiny_suite["status"],
            "resource_adjusted_tiny_suite_step_count": resource_adjusted_tiny_suite["step_count"],
            "resource_adjusted_tiny_suite_pass_count": resource_adjusted_tiny_suite["pass_count"],
            "resource_adjusted_tiny_suite_metrics": resource_adjusted_tiny_suite["metrics"],
            "resource_adjusted_tiny_suite_checks": resource_adjusted_tiny_suite["checks"],
            "resource_adjusted_tiny_suite_json": str(
                ROOT
                / "res/level_c/resource_adjusted_tiny_diffusion_suite/"
                / "level_c_resource_adjusted_tiny_diffusion_suite.json"
            ),
            "resource_adjusted_tiny_multiseed_status": resource_adjusted_tiny_multiseed["status"],
            "resource_adjusted_tiny_multiseed_statistics": resource_adjusted_tiny_multiseed["statistics"],
            "resource_adjusted_tiny_multiseed_rows": resource_adjusted_tiny_multiseed["rows"],
            "resource_adjusted_tiny_multiseed_checks": resource_adjusted_tiny_multiseed["checks"],
            "resource_adjusted_tiny_multiseed_json": str(
                ROOT
                / "res/level_c/resource_adjusted_tiny_diffusion_multiseed_audit/"
                / "level_c_resource_adjusted_tiny_diffusion_multiseed_audit.json"
            ),
            "resource_adjusted_tiny_checkpoint_eval_status": resource_adjusted_tiny_checkpoint_eval["status"],
            "resource_adjusted_tiny_checkpoint_eval_metrics": resource_adjusted_tiny_checkpoint_eval["metrics"],
            "resource_adjusted_tiny_checkpoint_eval_rows": resource_adjusted_tiny_checkpoint_eval["rows"],
            "resource_adjusted_tiny_checkpoint_eval_checks": resource_adjusted_tiny_checkpoint_eval["checks"],
            "resource_adjusted_tiny_checkpoint_eval_json": str(
                ROOT
                / "res/level_c/resource_adjusted_tiny_diffusion_checkpoint_eval/"
                / "level_c_resource_adjusted_tiny_diffusion_checkpoint_eval.json"
            ),
            "resource_adjusted_tiny_onnx_status": resource_adjusted_tiny_onnx["status"],
            "resource_adjusted_tiny_onnx_metrics": resource_adjusted_tiny_onnx["metrics"],
            "resource_adjusted_tiny_onnx_checks": resource_adjusted_tiny_onnx["checks"],
            "resource_adjusted_tiny_onnx_outputs": resource_adjusted_tiny_onnx["outputs"],
            "resource_adjusted_tiny_onnx_json": str(
                ROOT
                / "res/level_c/resource_adjusted_tiny_diffusion_onnx_export_inference/"
                / "level_c_resource_adjusted_tiny_diffusion_onnx_export_inference.json"
            ),
            "resource_adjusted_tiny_latency_status": resource_adjusted_tiny_latency["status"],
            "resource_adjusted_tiny_latency_metrics": resource_adjusted_tiny_latency["metrics"],
            "resource_adjusted_tiny_latency_checks": resource_adjusted_tiny_latency["checks"],
            "resource_adjusted_tiny_latency_json": str(
                ROOT
                / "res/level_c/resource_adjusted_tiny_diffusion_latency_audit/"
                / "level_c_resource_adjusted_tiny_diffusion_latency_audit.json"
            ),
            "resource_adjusted_tiny_video_preview_status": resource_adjusted_tiny_video_preview["status"],
            "resource_adjusted_tiny_video_preview_rows": resource_adjusted_tiny_video_preview["rows"],
            "resource_adjusted_tiny_video_preview_checks": resource_adjusted_tiny_video_preview["checks"],
            "resource_adjusted_tiny_video_preview_outputs": resource_adjusted_tiny_video_preview["outputs"],
            "resource_adjusted_tiny_video_preview_json": str(
                ROOT
                / "res/level_c/resource_adjusted_tiny_diffusion_video_preview/"
                / "level_c_resource_adjusted_tiny_diffusion_video_preview.json"
            ),
            "lafan1_paper_arch_training_status": lafan1_paper_arch_training["status"],
            "lafan1_paper_arch_training_metrics": {
                "public_lafan1_motion_count": lafan1_paper_arch_training["metrics"][
                    "public_lafan1_motion_count"
                ],
                "window_count": lafan1_paper_arch_training["metrics"]["window_count"],
                "token_count": lafan1_paper_arch_training["metrics"]["token_count"],
                "vae_parameter_count": lafan1_paper_arch_training["metrics"]["vae_parameter_count"],
                "diffusion_parameter_count": lafan1_paper_arch_training["metrics"][
                    "diffusion_parameter_count"
                ],
                "final_validation_decoded_action_mse": lafan1_paper_arch_training["metrics"][
                    "final_validation_decoded_action_mse"
                ],
                "final_test_decoded_action_mse": lafan1_paper_arch_training["metrics"][
                    "final_test_decoded_action_mse"
                ],
                "final_validation_pred_tau_mse": lafan1_paper_arch_training["metrics"][
                    "final_validation_pred_tau_mse"
                ],
                "final_test_pred_tau_mse": lafan1_paper_arch_training["metrics"][
                    "final_test_pred_tau_mse"
                ],
                "checkpoint_size_bytes": lafan1_paper_arch_training["metrics"]["checkpoint_size_bytes"],
                "data_parallel": lafan1_paper_arch_training["metrics"]["data_parallel"],
                "gpu_device_ids": lafan1_paper_arch_training["metrics"]["gpu_device_ids"],
            },
            "lafan1_paper_arch_training_checks": lafan1_paper_arch_training["checks"],
            "lafan1_paper_arch_training_outputs": lafan1_paper_arch_training["outputs"],
            "lafan1_paper_arch_training_json": str(
                ROOT
                / "res/level_c/lafan1_paper_arch_vae_diffusion_training/"
                / "lafan1_paper_arch_vae_diffusion_training.json"
            ),
            "lafan1_paper_arch_multiseed_status": lafan1_paper_arch_multiseed["status"],
            "lafan1_paper_arch_multiseed_statistics": lafan1_paper_arch_multiseed["statistics"],
            "lafan1_paper_arch_multiseed_checks": lafan1_paper_arch_multiseed["checks"],
            "lafan1_paper_arch_multiseed_outputs": lafan1_paper_arch_multiseed["outputs"],
            "lafan1_paper_arch_multiseed_json": str(
                ROOT
                / "res/level_c/lafan1_paper_arch_multiseed_audit/"
                / "level_c_lafan1_paper_arch_multiseed_audit.json"
            ),
            "lafan1_paper_arch_symmetry_multiseed_status": lafan1_paper_arch_symmetry_multiseed[
                "status"
            ],
            "lafan1_paper_arch_symmetry_multiseed_statistics": lafan1_paper_arch_symmetry_multiseed[
                "statistics"
            ],
            "lafan1_paper_arch_symmetry_multiseed_checks": lafan1_paper_arch_symmetry_multiseed["checks"],
            "lafan1_paper_arch_symmetry_multiseed_outputs": lafan1_paper_arch_symmetry_multiseed[
                "outputs"
            ],
            "lafan1_paper_arch_symmetry_multiseed_json": str(
                ROOT
                / "res/level_c/lafan1_paper_arch_symmetry_augmented_multiseed_audit/"
                / "level_c_lafan1_paper_arch_symmetry_augmented_multiseed_audit.json"
            ),
            "lafan1_paper_arch_high_memory_status": lafan1_paper_arch_high_memory["status"],
            "lafan1_paper_arch_high_memory_metrics": lafan1_paper_arch_high_memory["metrics"],
            "lafan1_paper_arch_high_memory_checks": lafan1_paper_arch_high_memory["checks"],
            "lafan1_paper_arch_high_memory_outputs": lafan1_paper_arch_high_memory["outputs"],
            "lafan1_paper_arch_high_memory_json": str(
                ROOT
                / "res/level_c/lafan1_paper_arch_high_memory_batch_audit/"
                / "level_c_lafan1_paper_arch_high_memory_batch_audit.json"
            ),
            "lafan1_paper_arch_symmetry_dataset_status": lafan1_paper_arch_symmetry_dataset["status"],
            "lafan1_paper_arch_symmetry_dataset_metrics": lafan1_paper_arch_symmetry_dataset["metrics"],
            "lafan1_paper_arch_symmetry_dataset_checks": lafan1_paper_arch_symmetry_dataset["checks"],
            "lafan1_paper_arch_symmetry_dataset_outputs": lafan1_paper_arch_symmetry_dataset["outputs"],
            "lafan1_paper_arch_symmetry_dataset_json": str(
                ROOT
                / "res/level_c/lafan1_paper_arch_symmetry_dataset/"
                / "lafan1_paper_arch_symmetry_dataset_audit.json"
            ),
            "lafan1_paper_arch_symmetry_training_status": lafan1_paper_arch_symmetry_training["status"],
            "lafan1_paper_arch_symmetry_training_metrics": {
                "public_lafan1_unique_motion_label_count": lafan1_paper_arch_symmetry_training["metrics"][
                    "public_lafan1_unique_motion_label_count"
                ],
                "augmented_motion_label_count": lafan1_paper_arch_symmetry_training["metrics"][
                    "augmented_motion_label_count"
                ],
                "window_count": lafan1_paper_arch_symmetry_training["metrics"]["window_count"],
                "token_count": lafan1_paper_arch_symmetry_training["metrics"]["token_count"],
                "vae_parameter_count": lafan1_paper_arch_symmetry_training["metrics"]["vae_parameter_count"],
                "diffusion_parameter_count": lafan1_paper_arch_symmetry_training["metrics"][
                    "diffusion_parameter_count"
                ],
                "final_validation_decoded_action_mse": lafan1_paper_arch_symmetry_training["metrics"][
                    "final_validation_decoded_action_mse"
                ],
                "final_test_decoded_action_mse": lafan1_paper_arch_symmetry_training["metrics"][
                    "final_test_decoded_action_mse"
                ],
                "final_validation_pred_tau_mse": lafan1_paper_arch_symmetry_training["metrics"][
                    "final_validation_pred_tau_mse"
                ],
                "final_test_pred_tau_mse": lafan1_paper_arch_symmetry_training["metrics"][
                    "final_test_pred_tau_mse"
                ],
                "checkpoint_size_bytes": lafan1_paper_arch_symmetry_training["metrics"][
                    "checkpoint_size_bytes"
                ],
                "data_parallel": lafan1_paper_arch_symmetry_training["metrics"]["data_parallel"],
                "gpu_device_ids": lafan1_paper_arch_symmetry_training["metrics"]["gpu_device_ids"],
                "elapsed_seconds": lafan1_paper_arch_symmetry_training["metrics"]["elapsed_seconds"],
            },
            "lafan1_paper_arch_symmetry_training_checks": lafan1_paper_arch_symmetry_training["checks"],
            "lafan1_paper_arch_symmetry_training_outputs": lafan1_paper_arch_symmetry_training["outputs"],
            "lafan1_paper_arch_symmetry_training_json": str(
                ROOT
                / "res/level_c/lafan1_paper_arch_symmetry_augmented_training/"
                / "lafan1_paper_arch_vae_diffusion_training.json"
            ),
            "lafan1_paper_arch_onnx_latency_status": lafan1_paper_arch_onnx_latency["status"],
            "lafan1_paper_arch_onnx_latency_metrics": lafan1_paper_arch_onnx_latency["metrics"],
            "lafan1_paper_arch_onnx_latency_checks": lafan1_paper_arch_onnx_latency["checks"],
            "lafan1_paper_arch_onnx_latency_outputs": lafan1_paper_arch_onnx_latency["outputs"],
            "lafan1_paper_arch_onnx_latency_json": str(
                ROOT
                / "res/level_c/lafan1_paper_arch_onnx_latency/"
                / "level_c_lafan1_paper_arch_onnx_latency_audit.json"
            ),
            "lafan1_paper_arch_symmetry_onnx_latency_status": lafan1_paper_arch_symmetry_onnx_latency[
                "status"
            ],
            "lafan1_paper_arch_symmetry_onnx_latency_metrics": lafan1_paper_arch_symmetry_onnx_latency[
                "metrics"
            ],
            "lafan1_paper_arch_symmetry_onnx_latency_checks": lafan1_paper_arch_symmetry_onnx_latency[
                "checks"
            ],
            "lafan1_paper_arch_symmetry_onnx_latency_outputs": lafan1_paper_arch_symmetry_onnx_latency[
                "outputs"
            ],
            "lafan1_paper_arch_symmetry_onnx_latency_json": str(
                ROOT
                / "res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/"
                / "level_c_lafan1_paper_arch_onnx_latency_audit.json"
            ),
            "lafan1_paper_arch_offline_metrics_status": lafan1_paper_arch_offline_metrics["status"],
            "lafan1_paper_arch_offline_metrics_metrics": lafan1_paper_arch_offline_metrics["metrics"],
            "lafan1_paper_arch_offline_metrics_checks": lafan1_paper_arch_offline_metrics["checks"],
            "lafan1_paper_arch_offline_metrics_outputs": lafan1_paper_arch_offline_metrics["outputs"],
            "lafan1_paper_arch_offline_metrics_json": str(
                ROOT
                / "res/level_c/lafan1_paper_arch_offline_metrics/"
                / "level_c_lafan1_paper_arch_offline_metrics_audit.json"
            ),
            "lafan1_paper_arch_symmetry_offline_metrics_status": lafan1_paper_arch_symmetry_offline_metrics[
                "status"
            ],
            "lafan1_paper_arch_symmetry_offline_metrics_metrics": lafan1_paper_arch_symmetry_offline_metrics[
                "metrics"
            ],
            "lafan1_paper_arch_symmetry_offline_metrics_checks": lafan1_paper_arch_symmetry_offline_metrics[
                "checks"
            ],
            "lafan1_paper_arch_symmetry_offline_metrics_outputs": lafan1_paper_arch_symmetry_offline_metrics[
                "outputs"
            ],
            "lafan1_paper_arch_symmetry_offline_metrics_json": str(
                ROOT
                / "res/level_c/lafan1_paper_arch_symmetry_augmented_offline_metrics/"
                / "level_c_lafan1_paper_arch_offline_metrics_audit.json"
            ),
            "lafan1_paper_arch_guidance_eval_status": lafan1_paper_arch_guidance_eval["status"],
            "lafan1_paper_arch_guidance_eval_task_summaries": lafan1_paper_arch_guidance_eval["task_summaries"],
            "lafan1_paper_arch_guidance_eval_checks": lafan1_paper_arch_guidance_eval["checks"],
            "lafan1_paper_arch_guidance_eval_outputs": lafan1_paper_arch_guidance_eval["outputs"],
            "lafan1_paper_arch_guidance_eval_json": str(
                ROOT
                / "res/level_c/lafan1_paper_arch_guidance_eval/"
                / "level_c_lafan1_paper_arch_guidance_eval.json"
            ),
            "lafan1_paper_arch_symmetry_guidance_eval_status": lafan1_paper_arch_symmetry_guidance_eval[
                "status"
            ],
            "lafan1_paper_arch_symmetry_guidance_eval_task_summaries": lafan1_paper_arch_symmetry_guidance_eval[
                "task_summaries"
            ],
            "lafan1_paper_arch_symmetry_guidance_eval_checks": lafan1_paper_arch_symmetry_guidance_eval[
                "checks"
            ],
            "lafan1_paper_arch_symmetry_guidance_eval_outputs": lafan1_paper_arch_symmetry_guidance_eval[
                "outputs"
            ],
            "lafan1_paper_arch_symmetry_guidance_eval_json": str(
                ROOT
                / "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval/"
                / "level_c_lafan1_paper_arch_guidance_eval.json"
            ),
            "lafan1_paper_arch_symmetry_guidance_eval_full_split_status": (
                lafan1_paper_arch_symmetry_guidance_eval_full_split["status"]
            ),
            "lafan1_paper_arch_symmetry_guidance_eval_full_split_row_count": (
                lafan1_paper_arch_symmetry_guidance_eval_full_split["row_count"]
            ),
            "lafan1_paper_arch_symmetry_guidance_eval_full_split_settings": {
                "splits": lafan1_paper_arch_symmetry_guidance_eval_full_split["settings"]["splits"],
                "max_windows_per_split": lafan1_paper_arch_symmetry_guidance_eval_full_split["settings"][
                    "max_windows_per_split"
                ],
                "split_window_counts": lafan1_paper_arch_symmetry_guidance_eval_full_split["settings"][
                    "split_window_counts"
                ],
            },
            "lafan1_paper_arch_symmetry_guidance_eval_full_split_task_summaries": (
                lafan1_paper_arch_symmetry_guidance_eval_full_split["task_summaries"]
            ),
            "lafan1_paper_arch_symmetry_guidance_eval_full_split_checks": (
                lafan1_paper_arch_symmetry_guidance_eval_full_split["checks"]
            ),
            "lafan1_paper_arch_symmetry_guidance_eval_full_split_outputs": (
                lafan1_paper_arch_symmetry_guidance_eval_full_split["outputs"]
            ),
            "lafan1_paper_arch_symmetry_guidance_eval_full_split_json": str(
                ROOT
                / "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/"
                / "level_c_lafan1_paper_arch_guidance_eval.json"
            ),
            "lafan1_paper_arch_reverse_guidance_status": lafan1_paper_arch_reverse_guidance["status"],
            "lafan1_paper_arch_reverse_guidance_task_summaries": lafan1_paper_arch_reverse_guidance[
                "task_summaries"
            ],
            "lafan1_paper_arch_reverse_guidance_improvement_summary": lafan1_paper_arch_reverse_guidance[
                "improvement_summary"
            ],
            "lafan1_paper_arch_reverse_guidance_checks": lafan1_paper_arch_reverse_guidance["checks"],
            "lafan1_paper_arch_reverse_guidance_outputs": lafan1_paper_arch_reverse_guidance["outputs"],
            "lafan1_paper_arch_reverse_guidance_json": str(
                ROOT
                / "res/level_c/lafan1_paper_arch_reverse_guidance/"
                / "level_c_lafan1_paper_arch_reverse_guidance_audit.json"
            ),
            "lafan1_paper_arch_symmetry_reverse_guidance_status": lafan1_paper_arch_symmetry_reverse_guidance[
                "status"
            ],
            "lafan1_paper_arch_symmetry_reverse_guidance_task_summaries": lafan1_paper_arch_symmetry_reverse_guidance[
                "task_summaries"
            ],
            "lafan1_paper_arch_symmetry_reverse_guidance_improvement_summary": (
                lafan1_paper_arch_symmetry_reverse_guidance["improvement_summary"]
            ),
            "lafan1_paper_arch_symmetry_reverse_guidance_checks": lafan1_paper_arch_symmetry_reverse_guidance[
                "checks"
            ],
            "lafan1_paper_arch_symmetry_reverse_guidance_outputs": lafan1_paper_arch_symmetry_reverse_guidance[
                "outputs"
            ],
            "lafan1_paper_arch_symmetry_reverse_guidance_json": str(
                ROOT
                / "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance/"
                / "level_c_lafan1_paper_arch_reverse_guidance_audit.json"
            ),
            "lafan1_paper_arch_symmetry_reverse_guidance_full_split_status": (
                lafan1_paper_arch_symmetry_reverse_guidance_full_split["status"]
            ),
            "lafan1_paper_arch_symmetry_reverse_guidance_full_split_metrics": {
                "row_count": lafan1_paper_arch_symmetry_reverse_guidance_full_split["metrics"]["row_count"],
                "total_batches": lafan1_paper_arch_symmetry_reverse_guidance_full_split["metrics"][
                    "total_batches"
                ],
                "total_reverse_forwards": lafan1_paper_arch_symmetry_reverse_guidance_full_split["metrics"][
                    "total_reverse_forwards"
                ],
                "min_after_reserve_used_mb": lafan1_paper_arch_symmetry_reverse_guidance_full_split[
                    "metrics"
                ]["min_after_reserve_used_mb"],
                "min_reverse_peak_allocated_mb": lafan1_paper_arch_symmetry_reverse_guidance_full_split[
                    "metrics"
                ]["min_reverse_peak_allocated_mb"],
            },
            "lafan1_paper_arch_symmetry_reverse_guidance_full_split_settings": {
                "splits": lafan1_paper_arch_symmetry_reverse_guidance_full_split["settings"]["splits"],
                "split_window_counts": lafan1_paper_arch_symmetry_reverse_guidance_full_split["settings"][
                    "split_window_counts"
                ],
                "selected_window_count": lafan1_paper_arch_symmetry_reverse_guidance_full_split["settings"][
                    "selected_window_count"
                ],
                "batch_size": lafan1_paper_arch_symmetry_reverse_guidance_full_split["settings"]["batch_size"],
                "scales": lafan1_paper_arch_symmetry_reverse_guidance_full_split["settings"]["scales"],
            },
            "lafan1_paper_arch_symmetry_reverse_guidance_full_split_improvement_summary": (
                lafan1_paper_arch_symmetry_reverse_guidance_full_split["improvement_summary"]
            ),
            "lafan1_paper_arch_symmetry_reverse_guidance_full_split_task_summaries": (
                lafan1_paper_arch_symmetry_reverse_guidance_full_split["task_summaries"]
            ),
            "lafan1_paper_arch_symmetry_reverse_guidance_full_split_checks": (
                lafan1_paper_arch_symmetry_reverse_guidance_full_split["checks"]
            ),
            "lafan1_paper_arch_symmetry_reverse_guidance_full_split_outputs": (
                lafan1_paper_arch_symmetry_reverse_guidance_full_split["outputs"]
            ),
            "lafan1_paper_arch_symmetry_reverse_guidance_full_split_json": str(
                ROOT
                / "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/"
                / "level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.json"
            ),
            "single_batch_overfit_status": single_batch_overfit["status"],
            "single_batch_overfit_metrics": {
                "initial_noisy_identity_loss": single_batch_overfit["metrics"]["initial_noisy_identity_loss"],
                "final_overfit_loss": single_batch_overfit["metrics"]["final_overfit_loss"],
                "loss_reduction_ratio": single_batch_overfit["metrics"]["loss_reduction_ratio"],
            },
            "single_batch_overfit_json": str(
                ROOT / "res/level_c/single_batch_overfit_probe/level_c_single_batch_overfit_probe.json"
            ),
            "single_motion_overfit_status": single_motion_overfit["status"],
            "single_motion_overfit_metrics": {
                "all_motion_baseline_loss": single_motion_overfit["metrics"]["all_motion_windows_baseline_loss"],
                "all_motion_overfit_loss": single_motion_overfit["metrics"]["all_motion_windows_overfit_loss"],
                "all_motion_loss_reduction_ratio": single_motion_overfit["metrics"][
                    "all_motion_windows_loss_reduction_ratio"
                ],
            },
            "single_motion_overfit_json": str(
                ROOT / "res/level_c/single_motion_overfit_probe/level_c_single_motion_overfit_probe.json"
            ),
            "small_dataset_overfit_status": small_dataset_overfit["status"],
            "small_dataset_overfit_metrics": {
                "motion_count": small_dataset_overfit["settings"]["motion_count"],
                "window_count": small_dataset_overfit["settings"]["window_count"],
                "all_small_dataset_baseline_loss": small_dataset_overfit["metrics"][
                    "all_small_dataset_windows_baseline_loss"
                ],
                "all_small_dataset_overfit_loss": small_dataset_overfit["metrics"][
                    "all_small_dataset_windows_overfit_loss"
                ],
                "all_small_dataset_loss_reduction_ratio": small_dataset_overfit["metrics"][
                    "all_small_dataset_windows_loss_reduction_ratio"
                ],
            },
            "small_dataset_overfit_json": str(
                ROOT / "res/level_c/small_dataset_overfit_probe/level_c_small_dataset_overfit_probe.json"
            ),
            "small_dataset_split_status": small_dataset_split["status"],
            "small_dataset_split_counts": small_dataset_split["counts"],
            "small_dataset_split_json": str(
                ROOT / "res/level_c/small_dataset_split_manifest/small_dataset_split_manifest_summary.json"
            ),
            "small_dataset_multiseed_status": small_dataset_multiseed["status"],
            "small_dataset_multiseed_statistics": small_dataset_multiseed["statistics"],
            "small_dataset_multiseed_json": str(
                ROOT / "res/level_c/small_dataset_multiseed_audit/level_c_small_dataset_multiseed_audit.json"
            ),
            "small_dataset_heldout_status": small_dataset_heldout["status"],
            "small_dataset_heldout_metrics": {
                "train_prediction_loss": small_dataset_heldout["metrics"]["train_prediction_loss"],
                "validation_prediction_loss": small_dataset_heldout["metrics"]["validation_prediction_loss"],
                "test_prediction_loss": small_dataset_heldout["metrics"]["test_prediction_loss"],
                "validation_loss_reduction_ratio": small_dataset_heldout["metrics"][
                    "validation_loss_reduction_ratio"
                ],
                "test_loss_reduction_ratio": small_dataset_heldout["metrics"]["test_loss_reduction_ratio"],
            },
            "small_dataset_heldout_json": str(
                ROOT / "res/level_c/small_dataset_heldout_eval/level_c_small_dataset_heldout_eval.json"
            ),
            "small_dataset_heldout_multiseed_status": small_dataset_heldout_multiseed["status"],
            "small_dataset_heldout_multiseed_statistics": {
                "validation_prediction_loss": small_dataset_heldout_multiseed["statistics"][
                    "validation_prediction_loss"
                ],
                "test_prediction_loss": small_dataset_heldout_multiseed["statistics"]["test_prediction_loss"],
                "validation_loss_reduction_ratio": small_dataset_heldout_multiseed["statistics"][
                    "validation_loss_reduction_ratio"
                ],
                "test_loss_reduction_ratio": small_dataset_heldout_multiseed["statistics"][
                    "test_loss_reduction_ratio"
                ],
            },
            "small_dataset_heldout_multiseed_json": str(
                ROOT
                / "res/level_c/small_dataset_heldout_multiseed_audit/"
                / "level_c_small_dataset_heldout_multiseed_audit.json"
            ),
            "vae_checkpoint_smoke_status": vae_checkpoint_smoke["status"],
            "vae_checkpoint_smoke_metrics": vae_checkpoint_smoke["metrics"],
            "vae_checkpoint_smoke_checks": vae_checkpoint_smoke["checks"],
            "vae_checkpoint_smoke_json": str(
                ROOT / "res/level_c/vae_checkpoint_smoke/level_c_vae_checkpoint_smoke.json"
            ),
            "vae_checkpoint_smoke_checkpoint": vae_checkpoint_smoke["outputs"]["checkpoint"],
            "vae_debug_overfit_latent_artifact_status": vae_debug_overfit_latents["status"],
            "vae_debug_overfit_latent_artifact_metrics": vae_debug_overfit_latents["metrics"],
            "vae_debug_overfit_latent_artifact_checks": vae_debug_overfit_latents["checks"],
            "vae_debug_overfit_latent_artifact_json": str(
                ROOT
                / "res/level_c/vae_debug_overfit_latent_artifact/"
                / "level_c_vae_debug_overfit_latent_artifact.json"
            ),
            "vae_motion_split_heldout_status": vae_motion_split_heldout["status"],
            "vae_motion_split_heldout_metrics": {
                "train_final_action_mse": vae_motion_split_heldout["metrics"]["train_final_action_mse"],
                "validation_final_action_mse": vae_motion_split_heldout["metrics"][
                    "validation_final_action_mse"
                ],
                "test_final_action_mse": vae_motion_split_heldout["metrics"]["test_final_action_mse"],
                "validation_action_mse_reduction_ratio": vae_motion_split_heldout["metrics"][
                    "validation_action_mse_reduction_ratio"
                ],
                "test_action_mse_reduction_ratio": vae_motion_split_heldout["metrics"][
                    "test_action_mse_reduction_ratio"
                ],
                "final_validation_kl_mean_sum_over_latent": vae_motion_split_heldout["metrics"][
                    "final_validation_kl_mean_sum_over_latent"
                ],
                "token_counts": vae_motion_split_heldout["metrics"]["token_counts"],
            },
            "vae_motion_split_heldout_checks": vae_motion_split_heldout["checks"],
            "vae_motion_split_heldout_json": str(
                ROOT
                / "res/level_c/vae_motion_split_heldout_eval/"
                / "level_c_vae_motion_split_heldout_eval.json"
            ),
            "vae_receding_horizon_rollout_status": vae_receding_horizon_rollout["status"],
            "vae_receding_horizon_rollout_metrics": {
                "row_count": vae_receding_horizon_rollout["metrics"]["row_count"],
                "current_action_mse_mean": vae_receding_horizon_rollout["metrics"][
                    "current_action_mse_mean"
                ],
                "current_action_mse_max": vae_receding_horizon_rollout["metrics"][
                    "current_action_mse_max"
                ],
                "full_window_action_mse_mean": vae_receding_horizon_rollout["metrics"][
                    "full_window_action_mse_mean"
                ],
                "next_latent_action_delta_mean": vae_receding_horizon_rollout["metrics"][
                    "next_latent_action_delta_mean"
                ],
                "mean_action_delta_mean": vae_receding_horizon_rollout["metrics"][
                    "mean_action_delta_mean"
                ],
            },
            "vae_receding_horizon_rollout_checks": vae_receding_horizon_rollout["checks"],
            "vae_receding_horizon_rollout_json": str(
                ROOT
                / "res/level_c/vae_receding_horizon_rollout_smoke/"
                / "level_c_vae_receding_horizon_rollout_smoke.json"
            ),
            "diffusion_to_vae_action_status": diffusion_to_vae_action["status"],
            "diffusion_to_vae_action_metrics": {
                "validation_predicted_current_action_mse": diffusion_to_vae_action["metrics"][
                    "validation_predicted_current_action_mse"
                ],
                "test_predicted_current_action_mse": diffusion_to_vae_action["metrics"][
                    "test_predicted_current_action_mse"
                ],
                "validation_current_mse_reduction_vs_noisy": diffusion_to_vae_action["metrics"][
                    "validation_current_mse_reduction_vs_noisy"
                ],
                "test_current_mse_reduction_vs_noisy": diffusion_to_vae_action["metrics"][
                    "test_current_mse_reduction_vs_noisy"
                ],
                "validation_predicted_full_action_mse": diffusion_to_vae_action["metrics"][
                    "validation_predicted_full_action_mse"
                ],
                "test_predicted_full_action_mse": diffusion_to_vae_action["metrics"][
                    "test_predicted_full_action_mse"
                ],
            },
            "diffusion_to_vae_action_checks": diffusion_to_vae_action["checks"],
            "diffusion_to_vae_action_json": str(
                ROOT
                / "res/level_c/diffusion_to_vae_action_smoke/"
                / "level_c_diffusion_to_vae_action_smoke.json"
            ),
            "diffusion_to_vae_action_multiseed_status": diffusion_to_vae_action_multiseed["status"],
            "diffusion_to_vae_action_multiseed_statistics": {
                "validation_predicted_current_action_mse": diffusion_to_vae_action_multiseed["statistics"][
                    "validation_predicted_current_action_mse"
                ],
                "test_predicted_current_action_mse": diffusion_to_vae_action_multiseed["statistics"][
                    "test_predicted_current_action_mse"
                ],
                "validation_current_mse_reduction_vs_noisy": diffusion_to_vae_action_multiseed["statistics"][
                    "validation_current_mse_reduction_vs_noisy"
                ],
                "test_current_mse_reduction_vs_noisy": diffusion_to_vae_action_multiseed["statistics"][
                    "test_current_mse_reduction_vs_noisy"
                ],
            },
            "diffusion_to_vae_action_multiseed_checks": diffusion_to_vae_action_multiseed["checks"],
            "diffusion_to_vae_action_multiseed_json": str(
                ROOT
                / "res/level_c/diffusion_to_vae_action_multiseed_audit/"
                / "level_c_diffusion_to_vae_action_multiseed_audit.json"
            ),
            "diffusion_to_vae_action_smoothness_status": diffusion_to_vae_action_smoothness["status"],
            "diffusion_to_vae_action_smoothness_metrics": {
                "validation_predicted_smoothness_penalty": diffusion_to_vae_action_smoothness["metrics"][
                    "validation_predicted_smoothness_penalty"
                ],
                "test_predicted_smoothness_penalty": diffusion_to_vae_action_smoothness["metrics"][
                    "test_predicted_smoothness_penalty"
                ],
                "validation_predicted_smoothness_reduction_vs_noisy": diffusion_to_vae_action_smoothness[
                    "metrics"
                ]["validation_predicted_smoothness_reduction_vs_noisy"],
                "test_predicted_smoothness_reduction_vs_noisy": diffusion_to_vae_action_smoothness["metrics"][
                    "test_predicted_smoothness_reduction_vs_noisy"
                ],
                "validation_predicted_action_rate_mean_norm_at_25hz": diffusion_to_vae_action_smoothness[
                    "metrics"
                ]["validation_predicted_action_rate_mean_norm_at_25hz"],
                "test_predicted_action_rate_mean_norm_at_25hz": diffusion_to_vae_action_smoothness["metrics"][
                    "test_predicted_action_rate_mean_norm_at_25hz"
                ],
                "validation_predicted_action_acceleration_mean_norm_at_25hz": diffusion_to_vae_action_smoothness[
                    "metrics"
                ]["validation_predicted_action_acceleration_mean_norm_at_25hz"],
                "test_predicted_action_acceleration_mean_norm_at_25hz": diffusion_to_vae_action_smoothness[
                    "metrics"
                ]["test_predicted_action_acceleration_mean_norm_at_25hz"],
            },
            "diffusion_to_vae_action_smoothness_checks": diffusion_to_vae_action_smoothness["checks"],
            "diffusion_to_vae_action_smoothness_json": str(
                ROOT
                / "res/level_c/diffusion_to_vae_action_smoothness_audit/"
                / "level_c_diffusion_to_vae_action_smoothness_audit.json"
            ),
            "direct_vs_latent_action_ablation_status": direct_vs_latent_action_ablation["status"],
            "direct_vs_latent_action_ablation_metrics": {
                "validation_direct_current_action_mse": direct_vs_latent_action_ablation["metrics"][
                    "validation_direct_current_action_mse"
                ],
                "test_direct_current_action_mse": direct_vs_latent_action_ablation["metrics"][
                    "test_direct_current_action_mse"
                ],
                "validation_latent_current_action_mse": direct_vs_latent_action_ablation["metrics"][
                    "validation_latent_current_action_mse"
                ],
                "test_latent_current_action_mse": direct_vs_latent_action_ablation["metrics"][
                    "test_latent_current_action_mse"
                ],
                "validation_latent_vs_direct_current_mse_ratio": direct_vs_latent_action_ablation["metrics"][
                    "validation_latent_vs_direct_current_mse_ratio"
                ],
                "test_latent_vs_direct_current_mse_ratio": direct_vs_latent_action_ablation["metrics"][
                    "test_latent_vs_direct_current_mse_ratio"
                ],
            },
            "direct_vs_latent_action_ablation_checks": direct_vs_latent_action_ablation["checks"],
            "direct_vs_latent_action_ablation_json": str(
                ROOT
                / "res/level_c/direct_vs_latent_action_ablation_audit/"
                / "level_c_direct_vs_latent_action_ablation_audit.json"
            ),
            "vae_contract_audit_status": vae_contract["status"],
            "vae_contract_audit_metrics": vae_contract["metrics"],
            "vae_contract_audit_checks": vae_contract["checks"],
            "vae_contract_audit_json": str(
                ROOT / "res/level_c/vae_contract_audit/level_c_vae_contract_audit.json"
            ),
            "dagger_vae_pipeline_audit_status": dagger_vae_pipeline["status"],
            "dagger_vae_pipeline_audit_metrics": dagger_vae_pipeline["metrics"],
            "dagger_vae_pipeline_audit_checks": dagger_vae_pipeline["checks"],
            "dagger_vae_pipeline_audit_json": str(
                ROOT / "res/level_c/dagger_vae_pipeline_audit/level_c_dagger_vae_pipeline_audit.json"
            ),
            "vae_latent_probe_status": vae_latent["status"],
            "vae_latent_probe_statistics": {
                "kl_loss": vae_latent["statistics"]["kl_loss"],
                "mean_latent_std": vae_latent["statistics"]["mean_latent_std"],
                "mean_neighbor_action_delta": vae_latent["statistics"]["mean_neighbor_action_delta"],
                "curvature_mean": vae_latent["statistics"]["curvature_mean"],
            },
            "vae_latent_probe_json": str(ROOT / "res/level_c/vae_latent_probe/level_c_vae_latent_probe.json"),
            "symmetry_mapping_audit_status": symmetry_mapping["status"],
            "symmetry_mapping_audit_metrics": symmetry_mapping["metrics"],
            "symmetry_mapping_audit_checks": symmetry_mapping["checks"],
            "symmetry_mapping_audit_json": str(
                ROOT / "res/level_c/symmetry_mapping_audit/level_c_symmetry_mapping_audit.json"
            ),
            "guidance_task_scale_sweep_status": guidance_task_scale["status"],
            "guidance_task_scale_sweep_row_count": guidance_task_scale["row_count"],
            "guidance_task_scale_sweep_task_summaries": guidance_task_scale["task_summaries"],
            "guidance_task_scale_sweep_checks": guidance_task_scale["checks"],
            "guidance_task_scale_sweep_json": str(
                ROOT / "res/level_c/guidance_task_scale_sweep/level_c_guidance_task_scale_sweep.json"
            ),
            "guidance_debug_visualization_status": guidance_debug_visualization["status"],
            "guidance_debug_visualization_primary_metrics": {
                task: {
                    "primary_metric": payload["primary_metric"],
                    "before": payload["metrics"][payload["primary_metric"]]["before"],
                    "after": payload["metrics"][payload["primary_metric"]]["after"],
                    "delta": payload["metrics"][payload["primary_metric"]]["delta"],
                }
                for task, payload in guidance_debug_visualization["per_task"].items()
            },
            "guidance_debug_visualization_checks": guidance_debug_visualization["checks"],
            "guidance_debug_visualization_outputs": guidance_debug_visualization["outputs"],
            "guidance_task_metric_audit_status": guidance_task_metric["status"],
            "guidance_task_metric_audit_metrics": guidance_task_metric["metrics"],
            "guidance_task_metric_audit_primary_metrics": guidance_task_metric["task_primary_metrics"],
            "guidance_task_metric_audit_checks": guidance_task_metric["checks"],
            "guidance_task_metric_audit_json": str(
                ROOT / "res/level_c/guidance_task_metric_audit/level_c_guidance_task_metric_audit.json"
            ),
            "guidance_full_split_result_table_status": guidance_full_split_result_table["status"],
            "guidance_full_split_result_table_metrics": guidance_full_split_result_table["metrics"],
            "guidance_full_split_result_table_mode_summary": guidance_full_split_result_table["mode_summary"],
            "guidance_full_split_result_table_checks": guidance_full_split_result_table["checks"],
            "guidance_full_split_result_table_outputs": guidance_full_split_result_table["outputs"],
            "guidance_full_split_result_table_json": str(
                ROOT
                / "res/level_c/guidance_full_split_result_table/"
                / "level_c_guidance_full_split_result_table.json"
            ),
            "guidance_checkpoint_visualization_status": guidance_checkpoint_visualization["status"],
            "guidance_checkpoint_visualization_metrics": guidance_checkpoint_visualization["metrics"],
            "guidance_checkpoint_visualization_checks": guidance_checkpoint_visualization["checks"],
            "guidance_checkpoint_visualization_outputs": guidance_checkpoint_visualization["outputs"],
            "guidance_checkpoint_visualization_json": str(
                ROOT
                / "res/level_c/guidance_checkpoint_visualization/"
                / "level_c_guidance_checkpoint_visualization.json"
            ),
            "guidance_visual_deliverables_audit_status": guidance_visual_deliverables["status"],
            "guidance_visual_deliverables_audit_metrics": guidance_visual_deliverables["metrics"],
            "guidance_visual_deliverables_audit_checks": guidance_visual_deliverables["checks"],
            "guidance_visual_deliverables_audit_json": str(
                ROOT
                / "res/level_c/guidance_visual_deliverables_audit/"
                / "level_c_guidance_visual_deliverables_audit.json"
            ),
            "guidance_cost_coverage_audit_status": guidance_cost_coverage["status"],
            "guidance_cost_coverage_audit_metrics": guidance_cost_coverage["metrics"],
            "guidance_cost_coverage_audit_checks": guidance_cost_coverage["checks"],
            "guidance_cost_coverage_audit_json": str(
                ROOT / "res/level_c/guidance_cost_coverage_audit/level_c_guidance_cost_coverage_audit.json"
            ),
            "core_math_unit_tests_status": core_math_unit_tests["status"],
            "core_math_unit_tests_metrics": {
                "row_count": core_math_unit_tests["row_count"],
                "failed_row_count": core_math_unit_tests["failed_row_count"],
                "covered_goal_item_count": len(core_math_unit_tests["covered_goal_items"]),
            },
            "core_math_unit_tests_checks": core_math_unit_tests["checks"],
            "core_math_unit_tests_json": str(
                ROOT / "res/tests/core_math_unit_tests/core_math_unit_tests.json"
            ),
            "timestep_mask_coverage_audit_status": timestep_mask_coverage["status"],
            "timestep_mask_coverage_audit_metrics": timestep_mask_coverage["metrics"],
            "timestep_mask_coverage_audit_checks": timestep_mask_coverage["checks"],
            "timestep_mask_coverage_audit_json": str(
                ROOT / "res/level_c/timestep_mask_coverage_audit/level_c_timestep_mask_coverage_audit.json"
            ),
            "paper_state_mask_reverse_probe_status": paper_state_mask_reverse["status"],
            "paper_state_mask_reverse_probe_settings": paper_state_mask_reverse["settings"],
            "paper_state_mask_reverse_probe_metrics": {
                "reverse_initial_mse": paper_state_mask_reverse["metrics"]["reverse_initial_mse"],
                "reverse_final_mse": paper_state_mask_reverse["metrics"]["reverse_final_mse"],
                "reverse_final_max_step": paper_state_mask_reverse["metrics"]["reverse_final_max_step"],
                "reverse_observed_clamp_max_abs_error": paper_state_mask_reverse["metrics"][
                    "reverse_observed_clamp_max_abs_error"
                ],
            },
            "paper_state_mask_reverse_probe_checks": paper_state_mask_reverse["checks"],
            "paper_state_mask_reverse_probe_json": str(
                ROOT / "res/level_c/paper_state_mask_reverse_probe/level_c_paper_state_mask_reverse_probe.json"
            ),
            "smoothness_latency_audit_status": smoothness_latency["status"],
            "smoothness_latency_audit_metrics": {
                "guided_final_state_second_difference_mean_norm": smoothness_latency["metrics"][
                    "guided_final_state_second_difference_mean_norm"
                ],
                "guided_final_latent_second_difference_mean_norm": smoothness_latency["metrics"][
                    "guided_final_latent_second_difference_mean_norm"
                ],
                "schema_action_delta_current_vs_next_latent": smoothness_latency["metrics"][
                    "schema_action_delta_current_vs_next_latent"
                ],
                "guidance_cost_reduction": smoothness_latency["metrics"]["guidance_cost_reduction"],
                "paper_denoising_fraction_of_control_period": smoothness_latency["metrics"][
                    "paper_denoising_fraction_of_control_period"
                ],
            },
            "smoothness_latency_audit_json": str(
                ROOT / "res/level_c/smoothness_latency_audit/level_c_smoothness_latency_audit.json"
            ),
            "deployment_protocol_audit_status": deployment_protocol["status"],
            "deployment_protocol_audit_metrics": deployment_protocol["metrics"],
            "deployment_protocol_audit_checks": deployment_protocol["checks"],
            "deployment_protocol_audit_json": str(
                ROOT / "res/level_c/deployment_protocol_audit/level_c_deployment_protocol_audit.json"
            ),
            "debug_probe_dirs": sorted(str(p) for p in (ROOT / "res/level_c").iterdir() if p.is_dir()),
        },
        "paper_source_coverage": {
            "status": coverage["status"],
            "counts": coverage["counts"],
            "bucket_counts": coverage["bucket_counts"],
            "json": str(ROOT / "res/paper_source_coverage/paper_source_coverage_audit.json"),
        },
        "paper_latex_inventory": {
            "status": paper_latex_inventory["status"],
            "counts": paper_latex_inventory["counts"],
            "equation_topic_counts": paper_latex_inventory["equation_topic_counts"],
            "checks": paper_latex_inventory["checks"],
            "json": str(ROOT / "res/paper_latex_inventory/paper_latex_inventory_audit.json"),
            "equations_tsv": str(ROOT / "res/paper_latex_inventory/paper_latex_equations.tsv"),
            "settings_tsv": str(ROOT / "res/paper_latex_inventory/paper_latex_experiment_settings.tsv"),
        },
        "paper_formula_code_trace": {
            "status": paper_formula_code_trace["status"],
            "row_count": paper_formula_code_trace["row_count"],
            "missing_evidence_row_count": paper_formula_code_trace["missing_evidence_row_count"],
            "status_counts": paper_formula_code_trace["status_counts"],
            "source_counts": paper_formula_code_trace["source_counts"],
            "checks": paper_formula_code_trace["checks"],
            "json": str(ROOT / "res/paper_formula_code_trace/paper_formula_code_trace_audit.json"),
            "tsv": str(ROOT / "res/paper_formula_code_trace/paper_formula_code_trace_audit.tsv"),
        },
        "paper_pdf_source_consistency": {
            "status": paper_pdf_source_consistency["status"],
            "metrics": paper_pdf_source_consistency["metrics"],
            "checks": paper_pdf_source_consistency["checks"],
            "json": str(
                ROOT / "res/paper_pdf_source_consistency/paper_pdf_source_consistency_audit.json"
            ),
            "anchors_tsv": str(ROOT / "res/paper_pdf_source_consistency/paper_pdf_anchor_audit.tsv"),
            "source_tar_tsv": str(ROOT / "res/paper_pdf_source_consistency/paper_source_tar_audit.tsv"),
        },
        "paper_table_values": {
            "status": table_values["status"],
            "counts": table_values["counts"],
            "json": str(ROOT / "res/paper_table_values/paper_table_value_audit.json"),
        },
        "skill_success_table": {
            "status": skill_success_table["status"],
            "metrics": skill_success_table["metrics"],
            "checks": skill_success_table["checks"],
            "missing_lafan_csv_names": skill_success_table["missing_lafan_csv_names"],
            "json": str(ROOT / "res/paper_skill_success_table_audit/skill_success_table_data_audit.json"),
        },
        "paper_vs_reproduction": {
            "status": comparison["status"],
            "total_rows": comparison["total_rows"],
            "comparison_type_counts": comparison["comparison_type_counts"],
            "missing_goal_checkpoint_rows": comparison["missing_goal_checkpoint_rows"],
            "csv": str(ROOT / "res/comparison/paper_vs_reproduction.csv"),
            "markdown": str(ROOT / "res/comparison/paper_vs_reproduction.md"),
            "json": str(ROOT / "res/comparison/paper_vs_reproduction.json"),
        },
        "results_claims": {
            "status": results_claims["status"],
            "metrics": results_claims["metrics"],
            "checks": results_claims["checks"],
            "local_status_counts": results_claims["local_status_counts"],
            "json": str(ROOT / "res/results_claims_audit/results_claims_audit.json"),
            "tsv": str(ROOT / "res/results_claims_audit/results_claims_audit.tsv"),
        },
        "goal_traceability": {
            "status": traceability["status"],
            "goal_line_count": traceability["goal_line_count"],
            "heading_count": traceability["heading_count"],
            "trace_row_count": traceability["trace_row_count"],
            "status_counts": traceability["status_counts"],
            "missing_evidence_rows": len(traceability["missing_evidence_rows"]),
            "json": str(ROOT / "res/goal_traceability/goal_traceability_audit.json"),
        },
        "goal_directive_index": {
            "status": goal_directive_index["status"],
            "line_count": goal_directive_index["line_count"],
            "heading_count": goal_directive_index["heading_count"],
            "directive_row_count": goal_directive_index["directive_row_count"],
            "tag_counts": goal_directive_index["tag_counts"],
            "checks": goal_directive_index["checks"],
            "json": str(ROOT / "res/goal_directive_index/goal_directive_index_audit.json"),
            "directives_tsv": str(ROOT / "res/goal_directive_index/goal_directive_rows.tsv"),
            "headings_tsv": str(ROOT / "res/goal_directive_index/goal_heading_rows.tsv"),
        },
        "goal_requirement_matrix": {
            "status": goal_matrix["status"],
            "goal_line_count": goal_matrix["goal_line_count"],
            "requirement_row_count": goal_matrix["requirement_row_count"],
            "status_counts": goal_matrix["status_counts"],
            "missing_evidence_rows": len(goal_matrix["missing_evidence_rows"]),
            "checks": goal_matrix["checks"],
            "json": str(ROOT / "res/goal_requirement_matrix/goal_requirement_matrix_audit.json"),
            "tsv": str(ROOT / "res/goal_requirement_matrix/goal_requirement_matrix_audit.tsv"),
        },
        "blocked_gates": {
            "status": blocked["status"],
            "gate_status_counts": blocked["gate_status_counts"],
            "gates": blocked["gates"],
            "json": str(ROOT / "res/blocked_gates/blocked_gate_audit.json"),
        },
        "verification_commands": [
            f"python3 {ROOT / 'reproduction/scripts/reproduction_master_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/blocked_gate_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/paper_source_coverage_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/paper_latex_inventory_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/paper_formula_code_trace_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/paper_pdf_source_consistency_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/paper_table_value_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/skill_success_table_data_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/released_panel_mapping_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/paper_vs_reproduction_comparison.py'}",
            f"python3 {ROOT / 'reproduction/scripts/results_claims_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/bm_diffusion_env_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/gpu_resource_audit.py'} --samples 3 --interval-sec 1",
            f"python3 {ROOT / 'reproduction/scripts/create_run_management_skeleton.py'}",
            f"python3 {ROOT / 'reproduction/scripts/checkpoint_resume_smoke.py'}",
            f"python3 {ROOT / 'reproduction/scripts/full_run_deliverable_gap_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/record_failed_run.py'}",
            f"python3 {ROOT / 'reproduction/scripts/record_official_train_entry_failed_run.py'}",
            f"python3 {ROOT / 'reproduction/scripts/resolved_reproduction_config.py'}",
            f"python3 {ROOT / 'reproduction/scripts/artifact_manifest.py'}",
            f"python3 {ROOT / 'reproduction/scripts/download_source_integrity_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/run_log_config_catalog.py'}",
            f"python3 {ROOT / 'reproduction/scripts/experiment_protocol_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/readme_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/patch_inventory_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/patch_snapshot_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/final_report_requirement_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/final_deliverables_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/visual_media_inventory_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/verification_command_coverage_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/verification_command_syntax_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/verification_command_script_manifest.py'}",
            f"python3 {ROOT / 'reproduction/scripts/required_artifact_absence_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/trial_failure_accounting_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/evaluation_metrics_coverage_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/ablation_coverage_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/metrics_catalog.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/released_data_metrics_summary.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/released_data_statistical_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/run_level_a_released_data_suite.py'}",
            f"python3 {ROOT / 'reproduction/scripts/guidance_task_coverage_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/progress_report_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/completion_matrix_status_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/project_boundary_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/core_test_coverage_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/coding_requirements_audit.py'}",
            f"python3 {ROOT / 'reproduction/tests/test_reimpl_package_api.py'}",
            f"python3 {ROOT / 'reproduction/scripts/run_reimpl_test_suite.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_diffusion_equation_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_trajectory_inverse_transform_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_emphasis_projection_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_state_representation_source_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_dataset_collection_protocol_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_rollout_rejection_manifest_probe.py'}",
            f"python3 {ROOT / 'reproduction/scripts/state_latent_schema_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/dagger_schema_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_dagger_iteration_smoke.py'}",
            f"python3 {ROOT / 'reproduction/scripts/build_level_c_paper_state_windows.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_state_latent_dataset_consistency_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_state_latent_training_dataset_contract_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_paper_state_overfit_probe.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_vae_latent_diffusion_overfit_probe.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_paper_state_heldout_eval.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_vae_latent_heldout_eval.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_paper_state_heldout_multiseed_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_vae_latent_heldout_multiseed_audit.py'}",
            f"BM_LEVEL_C_TORCH_THREADS=2 {ROOT / 'envs/bm_tracking/bin/python'} {ROOT / 'reproduction/scripts/level_c_paper_state_transformer_arch_probe.py'} --device cuda:0 --batch-size 1",
            f"{ROOT / 'envs/bm_tracking/bin/python'} {ROOT / 'reproduction/scripts/level_c_vae_latent_transformer_arch_probe.py'} --device cpu --batch-size 1",
            f"python3 {ROOT / 'reproduction/scripts/level_c_transformer_parameter_count_audit.py'}",
            f"{ROOT / 'envs/bm_tracking/bin/python'} {ROOT / 'reproduction/scripts/level_c_transformer_state_dict_manifest.py'}",
            f"{ROOT / 'envs/bm_tracking/bin/python'} {ROOT / 'reproduction/scripts/level_c_transformer_ema_smoke.py'} --device cpu --steps 2",
            f"{ROOT / 'envs/bm_tracking/bin/python'} {ROOT / 'reproduction/scripts/level_c_vae_latent_transformer_ema_smoke.py'} --device cpu --steps 2",
            f"{ROOT / 'envs/bm_tracking/bin/python'} {ROOT / 'reproduction/scripts/level_c_diffusion_checkpoint_smoke.py'} --device cpu",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_bounded_debug_diffusion_training_run.py'} --device cpu --steps 3 --batch-size 1 --torch-threads 2",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_bounded_debug_diffusion_checkpoint_eval.py'} --device cpu --torch-threads 2",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_bounded_debug_diffusion_action_eval.py'} --device cpu --torch-threads 2",
            f"python3 {ROOT / 'reproduction/scripts/run_level_c_resource_adjusted_tiny_diffusion_suite.py'}",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_training_run.py'} --device cpu --torch-threads 2 --epochs 180",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_multiseed_audit.py'} --device cpu --torch-threads 2 --epochs 80",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_checkpoint_eval.py'} --device cpu --torch-threads 2",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_onnx_export_inference.py'}",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_latency_audit.py'}",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_video_preview.py'}",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/train_lafan1_paper_level_vae_diffusion.py'} --device cuda:0 --max-motions 40 --max-frames-per-motion 420 --vae-epochs 24 --diffusion-epochs 1000 --diffusion-batch-size 512 --data-parallel",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_lafan1_paper_arch_multiseed_audit.py'}",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_lafan1_paper_arch_symmetry_dataset_audit.py'}",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/train_lafan1_paper_level_vae_diffusion.py'} --device cuda:0 --seed 20260621 --projection-seed 20260617 --run-id level_c_lafan1_paper_arch_symmetry_augmented_static_000_20260617_215500 --output-dir {ROOT / 'res/level_c/lafan1_paper_arch_symmetry_augmented_training'} --dataset-npz {ROOT / 'res/level_c/lafan1_paper_arch_symmetry_dataset/lafan1_paper_arch_symmetry_augmented_dataset.npz'} --dataset-source-label public_lafan1_symmetry_augmented_dataset --max-motions 40 --max-frames-per-motion 420 --vae-epochs 24 --diffusion-epochs 1000 --diffusion-batch-size 512 --data-parallel",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/train_lafan1_paper_level_vae_diffusion.py'} --device cuda:0 --seed 20260622 --projection-seed 20260617 --run-id level_c_lafan1_paper_arch_symmetry_augmented_seed_20260622_static_000_20260617_215500 --output-dir {ROOT / 'res/level_c/lafan1_paper_arch_symmetry_augmented_training_seed_20260622'} --dataset-npz {ROOT / 'res/level_c/lafan1_paper_arch_symmetry_dataset/lafan1_paper_arch_symmetry_augmented_dataset.npz'} --dataset-source-label public_lafan1_symmetry_augmented_dataset --max-motions 40 --max-frames-per-motion 420 --vae-epochs 24 --diffusion-epochs 1000 --diffusion-batch-size 512 --data-parallel",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/train_lafan1_paper_level_vae_diffusion.py'} --device cuda:0 --seed 20260623 --projection-seed 20260617 --run-id level_c_lafan1_paper_arch_symmetry_augmented_seed_20260623_static_000_20260617_215500 --output-dir {ROOT / 'res/level_c/lafan1_paper_arch_symmetry_augmented_training_seed_20260623'} --dataset-npz {ROOT / 'res/level_c/lafan1_paper_arch_symmetry_dataset/lafan1_paper_arch_symmetry_augmented_dataset.npz'} --dataset-source-label public_lafan1_symmetry_augmented_dataset --max-motions 40 --max-frames-per-motion 420 --vae-epochs 24 --diffusion-epochs 1000 --diffusion-batch-size 512 --data-parallel",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_lafan1_paper_arch_symmetry_multiseed_audit.py'}",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_lafan1_paper_arch_high_memory_batch_audit.py'} --initial-batch-size 8192 --max-batch-size 32768 --target-memory-mb 20000",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_lafan1_symmetry_training_comparison_audit.py'}",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_lafan1_paper_arch_onnx_latency_audit.py'}",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_lafan1_paper_arch_onnx_latency_audit.py'} --training-json {ROOT / 'res/level_c/lafan1_paper_arch_symmetry_augmented_training/lafan1_paper_arch_vae_diffusion_training.json'} --output-dir {ROOT / 'res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency'}",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_lafan1_paper_arch_offline_metrics_audit.py'}",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_lafan1_paper_arch_offline_metrics_audit.py'} --training-json {ROOT / 'res/level_c/lafan1_paper_arch_symmetry_augmented_training/lafan1_paper_arch_vae_diffusion_training.json'} --output-dir {ROOT / 'res/level_c/lafan1_paper_arch_symmetry_augmented_offline_metrics'}",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_lafan1_paper_arch_guidance_eval.py'}",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_lafan1_paper_arch_guidance_eval.py'} --training-json {ROOT / 'res/level_c/lafan1_paper_arch_symmetry_augmented_training/lafan1_paper_arch_vae_diffusion_training.json'} --offline-json {ROOT / 'res/level_c/lafan1_paper_arch_symmetry_augmented_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_audit.json'} --output-dir {ROOT / 'res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval'}",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_lafan1_paper_arch_guidance_eval.py'} --training-json {ROOT / 'res/level_c/lafan1_paper_arch_symmetry_augmented_training/lafan1_paper_arch_vae_diffusion_training.json'} --offline-json {ROOT / 'res/level_c/lafan1_paper_arch_symmetry_augmented_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_audit.json'} --output-dir {ROOT / 'res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split'} --splits validation,test --max-windows-per-split -1",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_lafan1_paper_arch_reverse_guidance_audit.py'}",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_lafan1_paper_arch_reverse_guidance_audit.py'} --training-json {ROOT / 'res/level_c/lafan1_paper_arch_symmetry_augmented_training/lafan1_paper_arch_vae_diffusion_training.json'} --offline-guidance-json {ROOT / 'res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.json'} --output-dir {ROOT / 'res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance'}",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.py'} --output-dir {ROOT / 'res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split'} --splits validation,test --max-windows-per-split -1 --batch-size 660 --target-memory-mb 10000",
            f"python3 {ROOT / 'reproduction/scripts/level_c_single_batch_overfit_probe.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_single_motion_overfit_probe.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_small_dataset_overfit_probe.py'}",
            f"python3 {ROOT / 'reproduction/scripts/build_level_c_small_dataset_split_manifest.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_small_dataset_multiseed_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_small_dataset_heldout_eval.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_small_dataset_heldout_multiseed_audit.py'}",
            f"{ROOT / 'envs/bm_tracking/bin/python'} {ROOT / 'reproduction/scripts/level_c_vae_checkpoint_smoke.py'} --device cpu",
            f"{ROOT / 'envs/bm_tracking/bin/python'} {ROOT / 'reproduction/scripts/level_c_vae_debug_overfit_latent_artifact.py'} --device cpu",
            f"{ROOT / 'envs/bm_tracking/bin/python'} {ROOT / 'reproduction/scripts/level_c_vae_motion_split_heldout_eval.py'} --device cpu",
            f"{ROOT / 'envs/bm_tracking/bin/python'} {ROOT / 'reproduction/scripts/level_c_vae_receding_horizon_rollout_smoke.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_diffusion_to_vae_action_smoke.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_diffusion_to_vae_action_multiseed_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_diffusion_to_vae_action_smoothness_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_direct_vs_latent_action_ablation_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_vae_contract_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/run_level_c_dagger_vae_pipeline_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_vae_latent_probe.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_symmetry_mapping_audit.py'}",
            f"{ROOT / 'envs/bm_tracking/bin/python'} {ROOT / 'reproduction/scripts/level_c_guidance_task_scale_sweep.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/level_c_guidance_debug_visualization.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_guidance_task_metric_audit.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/level_c_guidance_full_split_result_table.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/level_c_guidance_checkpoint_visualization.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_guidance_visual_deliverables_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_guidance_cost_coverage_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/run_level_c_debug_suite.py'}",
            f"python3 {ROOT / 'reproduction/scripts/run_level_c_extended_debug_suite.py'}",
            f"python3 {ROOT / 'reproduction/tests/test_core_math.py'}",
            f"python3 {ROOT / 'reproduction/scripts/reimpl_package_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/reimpl_runtime_integration_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_timestep_mask_coverage_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_paper_state_mask_reverse_probe.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_smoothness_latency_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/level_c_deployment_protocol_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/tracking_smoke_rerun_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/tracking_official_train_entry_retry_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/kit_inotify_budget_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/inotify_live_usage_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/vulkan_runtime_probe.py'}",
            f"python3 {ROOT / 'reproduction/scripts/cuda_p2p_runtime_probe.py'}",
            f"python3 {ROOT / 'reproduction/scripts/isaaclab_live_gate_probe.py'}",
            f"python3 {ROOT / 'reproduction/scripts/isaaclab_gpu_foundation_settings_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/vscode_watcher_exclude_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/kit_watcher_config_surface_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/tracking_import_gate_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/tracking_extension_namespace_probe.py'}",
            f"python3 {ROOT / 'reproduction/scripts/tracking_official_source_contract_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/tracking_g1_action_scale_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/tracking_reward_formula_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/tracking_observation_action_schema_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/tracking_randomization_termination_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/run_level_b_tracking_nonkit_suite.py'}",
            f"python3 {ROOT / 'reproduction/scripts/adaptive_sampling_discrepancy_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/motion_preprocessing_contract_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/tracking_local_smoke_preflight.py'}",
            f"python3 {ROOT / 'reproduction/scripts/build_tracking_motion_npz_fixture.py'}",
            f"python3 {ROOT / 'reproduction/scripts/mujoco_ros_launch_contract_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/tracking_deployment_controller_semantics_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/tracking_onnx_export_contract_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/tracking_motion_policy_onnx_contract_fixture.py'}",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/tracking_debug_motion_policy_onnx_export.py'}",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/tracking_debug_motion_policy_onnx_inference_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/goal_traceability_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/goal_directive_index_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/goal_requirement_matrix_audit.py'}",
            f"python3 {ROOT / 'reproduction/scripts/final_reproduction_report.py'}",
        ],
        "checks": {
            "atomic_write_used": True,
            "does_not_claim_goal_complete": True,
        },
        "outputs": {
            "json": str(OUT / "final_reproduction_report.json"),
            "markdown": str(DOC_OUT),
            "goal_markdown": str(GOAL_DOC_OUT),
        },
    }


def write_markdown(summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Final Reproduction Evidence Report")
    lines.append("")
    lines.append("This report consolidates the current BeyondMimic reproduction evidence. It is generated from machine-readable audits and does not mark the full goal complete.")
    lines.append("")
    lines.append("## Current Status")
    m = summary["master"]
    lines.append(f"- Master audit: `{m['artifact_pass_count']}/{m['artifact_count']}` artifacts pass, failures `{m['artifact_fail_count']}`.")
    lines.append(f"- Completion matrix counts: `{json.dumps(m['completion_matrix_counts'], sort_keys=True)}`.")
    lines.append(f"- Goal complete: `{summary['goal_complete']}`.")
    lines.append(f"- Why not complete: {m['why_not_complete']}")
    lines.append("")
    lines.append("## Level Summary")
    env = summary["environments"]
    lines.append(
        f"- Takeover audit: `{env['takeover']['status']}`; checks "
        f"`{json.dumps(env['takeover']['checks'], sort_keys=True)}`; runtime warnings "
        f"`{json.dumps(env['takeover']['command_failures'], sort_keys=True)}`."
    )
    lines.append(
        f"- Environments: bm_analysis `{env['bm_analysis']['status']}`, "
        f"bm_tracking `{env['bm_tracking']['status']}`, "
        f"bm_diffusion `{env['bm_diffusion']['status']}` with bm_diffusion checks "
        f"`{json.dumps(env['bm_diffusion']['checks'], sort_keys=True)}`."
    )
    lines.append(
        f"- Environment import probe: `{env['env_import_probe']['status']}`; checks "
        f"`{json.dumps(env['env_import_probe']['checks'], sort_keys=True)}`."
    )
    lines.append(
        f"- IsaacLab live gate probe: `{env['isaaclab_live_gate_probe']['status']}`; current blocker "
        f"`{env['isaaclab_live_gate_probe']['current_blocker']}`; checks "
        f"`{json.dumps(env['isaaclab_live_gate_probe']['checks'], sort_keys=True)}`."
    )
    lines.append(
        f"- Vulkan runtime probe: `{env['vulkan_runtime_probe']['status']}`; checks "
        f"`{json.dumps(env['vulkan_runtime_probe']['checks'], sort_keys=True)}`."
    )
    lines.append(
        f"- CUDA P2P runtime probe: `{env['cuda_p2p_runtime_probe']['status']}`; checks "
        f"`{json.dumps(env['cuda_p2p_runtime_probe']['checks'], sort_keys=True)}`."
    )
    lines.append(
        f"- IsaacLab GPU foundation settings audit: `{env['isaaclab_gpu_foundation_settings_audit']['status']}`; checks "
        f"`{json.dumps(env['isaaclab_gpu_foundation_settings_audit']['checks'], sort_keys=True)}`."
    )
    gpu = summary["gpu_resource_monitoring"]
    lines.append(
        f"- GPU resource monitoring: `{gpu['status']}`; `{gpu['rows_written']}` snapshot rows over "
        f"`{gpu['gpu_count']}` GPUs, nontrivial existing memory `{json.dumps(gpu['nontrivial_memory_gpus'], sort_keys=True)}`."
    )
    run = summary["run_management"]
    lines.append(
        f"- Run management schema: `{run['status']}`; diagnostic run `{run['run_id']}` has "
        f"`{run['gpu_metric_rows']}` GPU metric rows and required run files/directories."
    )
    resume = summary["checkpoint_resume_smoke"]
    lines.append(
        f"- Checkpoint/resume smoke: `{resume['status']}`; run `{resume['run_id']}` resumes with max abs error "
        f"`{resume['max_abs_resume_error']}` and writes `{resume['checkpoint_path']}`."
    )
    full_run_gap = summary["full_run_deliverable_gap_audit"]
    lines.append(
        f"- Full run deliverable gap audit: `{full_run_gap['status']}`; metrics "
        f"`{json.dumps(full_run_gap['metrics'], sort_keys=True)}`."
    )
    failed = summary["failed_run_retention"]
    lines.append(
        f"- Failed-run retention: `{failed['status']}`; failed run `{failed['run_id']}` preserved with "
        f"`{failed['gpu_status_rows']}` GPU status rows."
    )
    official_failed = summary["official_train_entry_failed_run_retention"]
    lines.append(
        f"- Official train-entry failed-run retention: `{official_failed['status']}`; failed run "
        f"`{official_failed['run_id']}` preserved with `{official_failed['gpu_status_rows']}` GPU status rows."
    )
    patches = summary["patch_inventory"]
    lines.append(
        f"- Patch inventory audit: `{patches['status']}`; "
        f"metrics `{json.dumps(patches['metrics'], sort_keys=True)}`, "
        f"status counts `{json.dumps(patches['status_counts'], sort_keys=True)}`."
    )
    patch_snapshot = summary["patch_snapshot"]
    lines.append(
        f"- Patch snapshot audit: `{patch_snapshot['status']}`; "
        f"metrics `{json.dumps(patch_snapshot['metrics'], sort_keys=True)}`, "
        f"patch directory `{patch_snapshot['patch_dir']}`."
    )
    reimpl = summary["reimplementation_package"]
    lines.append(
        f"- Reimplementation package: `{reimpl['status']}`; `{reimpl['python_file_count']}` Python files and "
        f"`{reimpl['symbol_row_count']}` checked formula API symbols under `{reimpl['source_root']}`."
    )
    reimpl_runtime = summary["reimplementation_runtime_integration"]
    lines.append(
        f"- Reimplementation runtime integration audit: `{reimpl_runtime['status']}`; "
        f"metrics `{json.dumps(reimpl_runtime['metrics'], sort_keys=True)}`."
    )
    coding = summary["coding_requirements"]
    lines.append(
        f"- Coding requirements audit: `{coding['status']}`; `{coding['requirement_row_count']}` goal.md coding rows, "
        f"failed `{coding['failed_requirement_count']}`, public function rows `{coding['function_row_count']}`."
    )
    api_tests = summary["reimpl_package_api_tests"]
    lines.append(
        f"- Reimplementation package API tests: `{api_tests['status']}`; `{api_tests['row_count']}` rows, "
        f"failed `{api_tests['failed_row_count']}`, covered items `{json.dumps(api_tests['covered_goal_items'], sort_keys=True)}`."
    )
    suite = summary["reimpl_test_suite"]
    lines.append(
        f"- Reimplementation test suite: `{suite['status']}`; `{suite['pass_count']}/{suite['step_count']}` "
        f"pure-Python code/test/audit steps passed, metrics `{json.dumps(suite['metrics'], sort_keys=True)}`."
    )
    cfg = summary["resolved_reproduction_config"]
    lines.append(
        f"- Resolved config manifest: `{cfg['status']}`; tracking `{cfg['tracking_control_frequency_hz']}` Hz, "
        f"PPO max iterations `{cfg['tracking_ppo_max_iterations']}`, VAE latent `{cfg['vae_latent_dim']}`, "
        f"diffusion batch `{cfg['diffusion_batch_size']}`, denoising steps `{cfg['diffusion_denoising_steps']}`."
    )
    manifest = summary["artifact_manifest"]
    lines.append(
        f"- Artifact manifest: `{manifest['status']}`; `{manifest['artifact_count']}` hashed key artifacts, "
        f"missing `{manifest['missing_count']}`."
    )
    matrix_status = summary["completion_matrix_status"]
    lines.append(
        f"- Completion matrix status audit: `{matrix_status['status']}`; `{matrix_status['row_count']}` rows, "
        f"invalid statuses `{matrix_status['invalid_status_count']}`, "
        f"status counts `{json.dumps(matrix_status['status_counts'], sort_keys=True)}`."
    )
    source_integrity = summary["download_source_integrity"]
    lines.append(
        f"- Download source integrity audit: `{source_integrity['status']}`; "
        f"`{source_integrity['file_count']}` manifest rows, total bytes `{source_integrity['total_size_bytes']}`, "
        f"required hashes `{source_integrity['required_hash_file_count']}`, "
        f"reference hashes `{source_integrity['reference_hash_file_count']}`."
    )
    run_log_catalog = summary["run_log_config_catalog"]
    lines.append(
        f"- Run/log/config catalog: `{run_log_catalog['status']}`; "
        f"metrics `{json.dumps(run_log_catalog['metrics'], sort_keys=True)}`."
    )
    protocol = summary["experiment_protocol"]
    lines.append(
        f"- Experiment protocol: `{protocol['status']}`; `{protocol['row_count']}` required protocol patterns, "
        f"missing `{protocol['missing_count']}`."
    )
    readme = summary["top_level_readme"]
    lines.append(
        f"- Top-level README: `{readme['status']}`; `{readme['row_count']}` required entry-point patterns, "
        f"missing `{readme['missing_count']}`."
    )
    deliverables = summary["final_deliverables"]
    lines.append(
        f"- Final deliverables audit: `{deliverables['status']}`; `{deliverables['row_count']}` deliverable rows, "
        f"status counts `{json.dumps(deliverables['status_counts'], sort_keys=True)}`, missing evidence rows "
        f"`{deliverables['missing_evidence_rows']}`."
    )
    visual_media = summary["visual_media_inventory"]
    lines.append(
        f"- Visual media inventory: `{visual_media['status']}`; `{visual_media['row_count']}` media files, "
        f"kind counts `{json.dumps(visual_media['kind_counts'], sort_keys=True)}`, category counts "
        f"`{json.dumps(visual_media['category_counts'], sort_keys=True)}`; paper-required rollout/robot videos remain absent."
    )
    verification = summary["verification_command_coverage"]
    lines.append(
        f"- Verification command coverage audit: `{verification['status']}`; `{verification['command_count']}` "
        f"final-report commands categorized, lightweight smoke pass `{verification['smoke_pass_count']}/"
        f"{verification['smoke_command_count']}`."
    )
    syntax = summary["verification_command_syntax"]
    lines.append(
        f"- Verification command syntax audit: `{syntax['status']}`; `{syntax['python_script_count']}` unique "
        f"Python command scripts compiled, failed `{syntax['failed_count']}`."
    )
    script_manifest = summary["verification_command_script_manifest"]
    lines.append(
        f"- Verification command script manifest: `{script_manifest['status']}`; "
        f"`{script_manifest['python_script_count']}` unique Python command scripts hashed with SHA256."
    )
    required_artifacts = summary["required_artifact_absence"]
    lines.append(
        f"- Required artifact absence audit: `{required_artifacts['status']}`; `{required_artifacts['row_count']}` "
        f"trained/deployment artifact rows, status counts "
        f"`{json.dumps(required_artifacts['status_counts'], sort_keys=True)}`, local model files "
        f"`{required_artifacts['local_scan_counts']['local_reproduction_model_files_excluding_diagnostic']}`, "
        f"local videos `{required_artifacts['local_scan_counts']['local_video_files']}`."
    )
    eval_metrics = summary["evaluation_metrics_coverage"]
    lines.append(
        f"- Evaluation metrics coverage audit: `{eval_metrics['status']}`; `{eval_metrics['row_count']}` "
        f"`goal.md` Section 12 metrics, status counts "
        f"`{json.dumps(eval_metrics['status_counts'], sort_keys=True)}`, missing evidence rows "
        f"`{eval_metrics['missing_evidence_rows']}`."
    )
    trial_failure = summary["trial_failure_accounting"]
    lines.append(
        f"- Trial/failure accounting audit: `{trial_failure['status']}`; "
        f"metrics `{json.dumps(trial_failure['metrics'], sort_keys=True)}`, "
        f"status counts `{json.dumps(trial_failure['status_counts'], sort_keys=True)}`."
    )
    metrics_catalog = summary["metrics_catalog"]
    lines.append(
        f"- Metrics catalog: `{metrics_catalog['status']}`; "
        f"metrics `{json.dumps(metrics_catalog['metrics'], sort_keys=True)}`, "
        f"level counts `{json.dumps(metrics_catalog['level_counts'], sort_keys=True)}`."
    )
    ablations = summary["ablation_coverage"]
    lines.append(
        f"- Ablation coverage audit: `{ablations['status']}`; `{ablations['row_count']}` Phase 9 items, "
        f"group counts `{json.dumps(ablations['group_counts'], sort_keys=True)}`, status counts "
        f"`{json.dumps(ablations['status_counts'], sort_keys=True)}`."
    )
    lines.append(
        f"- Public LAFAN1 symmetry training comparison: "
        f"`{ablations['symmetry_training_comparison_status']}`; metrics "
        f"`{json.dumps(ablations['symmetry_training_comparison_metrics'], sort_keys=True)}`."
    )
    guidance_tasks = summary["guidance_task_coverage"]
    lines.append(
        f"- Guidance task coverage audit: `{guidance_tasks['status']}`; `{guidance_tasks['row_count']}` Phase 8 "
        f"task-requirement rows, task counts `{json.dumps(guidance_tasks['task_counts'], sort_keys=True)}`, "
        f"status counts `{json.dumps(guidance_tasks['status_counts'], sort_keys=True)}`."
    )
    progress = summary["progress_report"]
    lines.append(
        f"- Progress report audit: `{progress['status']}`; `{progress['required_field_count']}` required fields, "
        f"`{progress['progress_marker_count']}` key progress markers, missing `{progress['missing_count']}`."
    )
    boundary = summary["project_boundary"]
    lines.append(
        f"- Project boundary audit: `{boundary['status']}`; `{boundary['row_count']}` path/download/cache checks, "
        f"failures `{boundary['failed_count']}`."
    )
    core_cov = summary["core_test_coverage"]
    lines.append(
        f"- Core test coverage audit: `{core_cov['status']}`; `{core_cov['required_count']}` explicit "
        f"`goal.md` checklist items, missing `{core_cov['missing_count']}`, core-test failures "
        f"`{core_cov['core_test_failed_row_count']}`."
    )
    level_a = summary["level_a_released_data"]
    lines.append(f"- Level A released data: `{level_a['status']}` with `{level_a['released_figure_rows']}` released-figure rows and `{level_a['paper_panel_map_rows']}` panel-map rows.")
    lines.append(
        f"- Released panel mapping audit: `{level_a['released_panel_mapping_status']}`; "
        f"metrics `{json.dumps(level_a['released_panel_mapping_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Released-data metrics summary: `{level_a['released_data_metrics_status']}`; "
        f"metrics `{json.dumps(level_a['released_data_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Released-data statistical audit: `{level_a['released_data_statistical_audit_status']}`; "
        f"metrics `{json.dumps(level_a['released_data_statistical_audit_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Level A released-data/table suite: `{level_a['released_data_suite_status']}`; "
        f"`{level_a['released_data_suite_pass_count']}/{level_a['released_data_suite_step_count']}` "
        f"released-data/table steps passed, metrics "
        f"`{json.dumps(level_a['released_data_suite_metrics'], sort_keys=True)}`."
    )
    lines.append(f"- Level B tracking/deployment: `{summary['level_b_tracking']['status']}`; live Kit and ROS/deployment gates remain listed below.")
    lines.append(
        f"- Level B tracking smoke rerun audit: `{summary['level_b_tracking']['tracking_smoke_rerun_status']}`; "
        f"metrics `{json.dumps(summary['level_b_tracking']['tracking_smoke_rerun_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Level B Kit/inotify budget audit: `{summary['level_b_tracking']['kit_inotify_budget_status']}`; "
        f"metrics `{json.dumps(summary['level_b_tracking']['kit_inotify_budget_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Level B live inotify usage audit: `{summary['level_b_tracking']['inotify_live_usage_status']}`; "
        f"metrics `{json.dumps(summary['level_b_tracking']['inotify_live_usage_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- VS Code watcher exclude audit: `{summary['level_b_tracking']['vscode_watcher_exclude_status']}`; "
        f"snapshot `{json.dumps(summary['level_b_tracking']['vscode_watcher_exclude_snapshot'], sort_keys=True)}`."
    )
    lines.append(
        f"- Level B Kit watcher config surface audit: "
        f"`{summary['level_b_tracking']['kit_watcher_config_surface_status']}`; "
        f"metrics `{json.dumps(summary['level_b_tracking']['kit_watcher_config_surface_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Level B tracking import gate audit: "
        f"`{summary['level_b_tracking']['tracking_import_gate_status']}`; "
        f"metrics `{json.dumps(summary['level_b_tracking']['tracking_import_gate_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Level B tracking extension namespace probe: "
        f"`{summary['level_b_tracking']['tracking_extension_namespace_status']}`; "
        f"metrics `{json.dumps(summary['level_b_tracking']['tracking_extension_namespace_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Level B official source contract audit: "
        f"`{summary['level_b_tracking']['tracking_official_source_contract_status']}`; "
        f"metrics `{json.dumps(summary['level_b_tracking']['tracking_official_source_contract_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Level B G1 action-scale audit: "
        f"`{summary['level_b_tracking']['tracking_g1_action_scale_status']}`; "
        f"metrics `{json.dumps(summary['level_b_tracking']['tracking_g1_action_scale_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Level B tracking reward formula audit: "
        f"`{summary['level_b_tracking']['tracking_reward_formula_status']}`; "
        f"metrics `{json.dumps(summary['level_b_tracking']['tracking_reward_formula_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Level B tracking observation/action schema audit: "
        f"`{summary['level_b_tracking']['tracking_observation_action_schema_status']}`; "
        f"metrics `{json.dumps(summary['level_b_tracking']['tracking_observation_action_schema_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Level B tracking randomization/termination audit: "
        f"`{summary['level_b_tracking']['tracking_randomization_termination_status']}`; "
        f"metrics `{json.dumps(summary['level_b_tracking']['tracking_randomization_termination_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Level B non-Kit tracking suite: "
        f"`{summary['level_b_tracking']['tracking_nonkit_suite_status']}`; "
        f"`{summary['level_b_tracking']['tracking_nonkit_suite_pass_count']}/"
        f"{summary['level_b_tracking']['tracking_nonkit_suite_step_count']}` steps passed, metrics "
        f"`{json.dumps(summary['level_b_tracking']['tracking_nonkit_suite_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Level B ONNX export contract: `{summary['level_b_tracking']['onnx_export_contract_status']}`; "
        f"metrics `{json.dumps(summary['level_b_tracking']['onnx_export_contract_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Level B motion-policy ONNX contract fixture: "
        f"`{summary['level_b_tracking']['motion_policy_onnx_contract_fixture_status']}`; "
        f"metrics `{json.dumps(summary['level_b_tracking']['motion_policy_onnx_contract_fixture_metrics'], sort_keys=True)}`; "
        f"NPZ `{summary['level_b_tracking']['motion_policy_onnx_contract_fixture_npz']}` is debug-only, not a real trained ONNX."
    )
    lines.append(
        f"- Level B debug motion-policy ONNX export: "
        f"`{summary['level_b_tracking']['debug_motion_policy_onnx_export_status']}`; "
        f"ONNX `{summary['level_b_tracking']['debug_motion_policy_onnx_export_path']}` "
        f"({summary['level_b_tracking']['debug_motion_policy_onnx_export_size_bytes']} bytes, "
        f"sha256 `{summary['level_b_tracking']['debug_motion_policy_onnx_export_sha256']}`) matches the contract but is not trained."
    )
    lines.append(
        f"- Level B debug motion-policy ONNX inference: "
        f"`{summary['level_b_tracking']['debug_motion_policy_onnx_inference_status']}`; "
        f"reference-evaluator metrics "
        f"`{json.dumps(summary['level_b_tracking']['debug_motion_policy_onnx_inference_metrics'], sort_keys=True)}`. "
        f"This proves graph load/inference for the debug contract only, not a trained policy."
    )
    lines.append(
        f"- Level B adaptive sampling discrepancy audit: "
        f"`{summary['level_b_tracking']['adaptive_sampling_discrepancy_status']}`; "
        f"metrics `{json.dumps(summary['level_b_tracking']['adaptive_sampling_discrepancy_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Level B motion preprocessing contract audit: "
        f"`{summary['level_b_tracking']['motion_preprocessing_contract_status']}`; "
        f"metrics `{json.dumps(summary['level_b_tracking']['motion_preprocessing_contract_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Level B debug motion.npz fixture: "
        f"`{summary['level_b_tracking']['tracking_motion_npz_fixture_status']}`; "
        f"metrics `{json.dumps(summary['level_b_tracking']['tracking_motion_npz_fixture_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Level B official replay preflight: "
        f"`{summary['level_b_tracking']['tracking_official_replay_preflight_status']}`; "
        f"checks `{json.dumps(summary['level_b_tracking']['tracking_official_replay_preflight_checks'], sort_keys=True)}`. "
        "This plans official conversion/replay commands only; it does not execute rendered replay or PPO."
    )
    lines.append(
        f"- Level B official replay conversion attempt: "
        f"`{summary['level_b_tracking']['tracking_official_replay_conversion_status']}`; "
        f"latest blocker `{summary['level_b_tracking']['tracking_official_replay_conversion_latest_blocker']}`; "
        f"checks `{json.dumps(summary['level_b_tracking']['tracking_official_replay_conversion_checks'], sort_keys=True)}`. "
        "RSL-RL and libGLU environment issues were repaired, but no valid official motion.npz was produced."
    )
    lines.append(
        f"- Level B G1 URDF conversion probe: "
        f"`{summary['level_b_tracking']['tracking_urdf_conversion_probe_status']}`; "
        f"payload `{json.dumps(summary['level_b_tracking']['tracking_urdf_conversion_probe_payload'], sort_keys=True)}`. "
        "The isolated converter opens a tiny USD but records zero traversed prims and no valid default prim."
    )
    lines.append(
        f"- Level B URDF path/tiny contrast probe: "
        f"`{summary['level_b_tracking']['tracking_urdf_path_tiny_probe_status']}`; "
        f"current blocker `{summary['level_b_tracking']['tracking_urdf_path_tiny_probe_current_blocker']}`; "
        f"markers `{json.dumps(summary['level_b_tracking']['tracking_urdf_path_tiny_probe_markers'], sort_keys=True)}`. "
        "This shows the official replay blocker is now localized to Isaac Sim URDF USD write/runtime behavior, not to "
        "missing local G1 mesh files."
    )
    lines.append(
        f"- Level B MJCF/stage bypass probe: "
        f"`{summary['level_b_tracking']['tracking_mjcf_stage_probe_status']}`; "
        f"current blocker `{summary['level_b_tracking']['tracking_mjcf_stage_probe_current_blocker']}`; "
        f"checks `{json.dumps(summary['level_b_tracking']['tracking_mjcf_stage_probe_checks'], sort_keys=True)}`. "
        "The minimal USD stage save itself fails with a save-forbidden error, and both tiny MJCF and official G1 MJCF "
        "conversion produce empty USD layers, so the blocker is below the URDF/MJCF asset-format choice."
    )
    lines.append(
        f"- Level B USD save-policy probe: "
        f"`{summary['level_b_tracking']['tracking_usd_save_policy_probe_status']}`; "
        f"current blocker `{summary['level_b_tracking']['tracking_usd_save_policy_probe_current_blocker']}`; "
        f"counts `{json.dumps(summary['level_b_tracking']['tracking_usd_save_policy_probe_counts'], sort_keys=True)}`. "
        "Plain `bm_tracking` Python cannot import `pxr`, while AppLauncher can import `pxr` but creates all tested "
        "local layers with `permissionToSave=False`; Save, Export, and SetPermissionToSave(True) attempts all fail."
    )
    lines.append(
        f"- Level B SimulationApp/AppLauncher save-policy comparison: "
        f"`{summary['level_b_tracking']['tracking_simulationapp_save_policy_probe_status']}`; "
        f"current blocker "
        f"`{summary['level_b_tracking']['tracking_simulationapp_save_policy_probe_current_blocker']}`; "
        f"cases `{json.dumps(summary['level_b_tracking']['tracking_simulationapp_save_policy_probe_cases'], sort_keys=True)}`. "
        "Raw SimulationApp with the IsaacLab headless experience reaches payload and shows the same local USD "
        "permissionToSave=False behavior as AppLauncher; the Isaac Sim base python experience records a Vulkan "
        "device-lost crash before payload. This keeps the official replay gate blocked and does not produce motion.npz."
    )
    lines.append(
        f"- Level B local tracking smoke preflight: "
        f"`{summary['level_b_tracking']['tracking_local_smoke_preflight_status']}`; "
        f"`{summary['level_b_tracking']['tracking_local_smoke_preflight_pass_count']}/"
        f"{summary['level_b_tracking']['tracking_local_smoke_preflight_step_count']}` non-Kit steps passed."
    )
    lines.append(
        f"- Level B official train entry retry: "
        f"`{summary['level_b_tracking']['tracking_official_train_entry_retry_status']}`; "
        f"classification "
        f"`{json.dumps(summary['level_b_tracking']['tracking_official_train_entry_retry_classification'], sort_keys=True)}`."
    )
    lines.append(
        f"- Level B MuJoCo/ROS launch contract: "
        f"`{summary['level_b_tracking']['mujoco_ros_launch_contract_status']}`; "
        f"metrics `{json.dumps(summary['level_b_tracking']['mujoco_ros_launch_contract_metrics'], sort_keys=True)}`; "
        f"host `{json.dumps(summary['level_b_tracking']['mujoco_ros_launch_contract_host_runtime'], sort_keys=True)}`."
    )
    lines.append(
        f"- Level B deployment controller semantics: "
        f"`{summary['level_b_tracking']['deployment_controller_semantics_status']}`; "
        f"metrics `{json.dumps(summary['level_b_tracking']['deployment_controller_semantics_metrics'], sort_keys=True)}`; "
        f"host `{json.dumps(summary['level_b_tracking']['deployment_controller_semantics_host_runtime'], sort_keys=True)}`."
    )
    lines.append(f"- Level C VAE/diffusion: `{summary['level_c_diffusion']['status']}`; official Level C code found `{summary['level_c_diffusion']['official_level_c_code_found']}`, checkpoint/engine found `{summary['level_c_diffusion']['official_level_c_checkpoint_or_engine_found']}`.")
    lines.append(
        f"- Level C debug suite: `{summary['level_c_diffusion']['debug_suite_status']}`; "
        f"`{summary['level_c_diffusion']['debug_suite_pass_count']}/"
        f"{summary['level_c_diffusion']['debug_suite_step_count']}` lightweight VAE/diffusion/guidance/debug-action "
        f"steps passed, metrics `{json.dumps(summary['level_c_diffusion']['debug_suite_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Diffusion equation audit: `{summary['level_c_diffusion']['diffusion_equation_audit_status']}`; "
        f"exact coefficient schedule missing from public source `{summary['level_c_diffusion']['diffusion_exact_coefficient_schedule_missing']}`."
    )
    lines.append(
        f"- Trajectory inverse transform audit: `{summary['level_c_diffusion']['trajectory_inverse_transform_audit_status']}`; "
        f"root/body round-trip checks `{json.dumps(summary['level_c_diffusion']['trajectory_inverse_transform_checks'], sort_keys=True)}`."
    )
    lines.append(
        f"- Emphasis projection audit: `{summary['level_c_diffusion']['emphasis_projection_audit_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['emphasis_projection_audit_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- State representation source audit: "
        f"`{summary['level_c_diffusion']['state_representation_source_audit_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['state_representation_source_audit_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Dataset collection protocol audit: "
        f"`{summary['level_c_diffusion']['dataset_collection_protocol_audit_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['dataset_collection_protocol_audit_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Rollout rejection manifest probe: "
        f"`{summary['level_c_diffusion']['rollout_rejection_manifest_probe_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['rollout_rejection_manifest_probe_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- State-latent schema audit: `{summary['level_c_diffusion']['state_latent_schema_audit_status']}`; "
        f"`{summary['level_c_diffusion']['state_latent_schema_audit_row_count']}` windows, split counts "
        f"`{json.dumps(summary['level_c_diffusion']['state_latent_schema_audit_split_counts'], sort_keys=True)}`, "
        f"token shapes `{json.dumps(summary['level_c_diffusion']['state_latent_schema_audit_token_shape_counts'], sort_keys=True)}`."
    )
    lines.append(
        f"- DAgger schema audit: `{summary['level_c_diffusion']['dagger_schema_audit_status']}`; "
        f"`{summary['level_c_diffusion']['dagger_schema_audit_row_count']}` synthetic teacher-query samples, "
        f"split counts `{json.dumps(summary['level_c_diffusion']['dagger_schema_audit_split_counts'], sort_keys=True)}`, "
        f"metrics `{json.dumps(summary['level_c_diffusion']['dagger_schema_audit_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- DAgger iteration smoke: `{summary['level_c_diffusion']['dagger_iteration_smoke_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['dagger_iteration_smoke_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Paper-formula state windows: `{summary['level_c_diffusion']['paper_state_windows_status']}`; "
        f"counts `{json.dumps(summary['level_c_diffusion']['paper_state_windows_counts'], sort_keys=True)}`."
    )
    lines.append(
        f"- State-latent dataset consistency audit: "
        f"`{summary['level_c_diffusion']['state_latent_dataset_consistency_status']}`; "
        f"metrics "
        f"`{json.dumps(summary['level_c_diffusion']['state_latent_dataset_consistency_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- State-latent training dataset contract audit: "
        f"`{summary['level_c_diffusion']['state_latent_training_dataset_contract_status']}`; "
        f"metrics "
        f"`{json.dumps(summary['level_c_diffusion']['state_latent_training_dataset_contract_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Paper-state overfit gate: `{summary['level_c_diffusion']['paper_state_overfit_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['paper_state_overfit_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Debug-VAE-latent diffusion overfit gate: "
        f"`{summary['level_c_diffusion']['vae_latent_diffusion_overfit_status']}`; "
        f"metrics "
        f"`{json.dumps(summary['level_c_diffusion']['vae_latent_diffusion_overfit_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Paper-state held-out eval: `{summary['level_c_diffusion']['paper_state_heldout_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['paper_state_heldout_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Debug-VAE-latent held-out eval: `{summary['level_c_diffusion']['vae_latent_heldout_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['vae_latent_heldout_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Paper-state held-out multi-seed audit: "
        f"`{summary['level_c_diffusion']['paper_state_heldout_multiseed_status']}`; "
        f"statistics "
        f"`{json.dumps(summary['level_c_diffusion']['paper_state_heldout_multiseed_statistics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Debug-VAE-latent held-out multi-seed audit: "
        f"`{summary['level_c_diffusion']['vae_latent_heldout_multiseed_status']}`; "
        f"statistics "
        f"`{json.dumps(summary['level_c_diffusion']['vae_latent_heldout_multiseed_statistics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Level C extended debug suite: "
        f"`{summary['level_c_diffusion']['extended_debug_suite_status']}`; "
        f"`{summary['level_c_diffusion']['extended_debug_suite_pass_count']}/"
        f"{summary['level_c_diffusion']['extended_debug_suite_step_count']}` steps passed, metrics "
        f"`{json.dumps(summary['level_c_diffusion']['extended_debug_suite_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Paper-state Transformer architecture probe: "
        f"`{summary['level_c_diffusion']['paper_state_transformer_arch_probe_status']}`; "
        f"settings `{json.dumps(summary['level_c_diffusion']['paper_state_transformer_arch_probe_settings'], sort_keys=True)}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['paper_state_transformer_arch_probe_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Debug-VAE-latent Transformer architecture probe: "
        f"`{summary['level_c_diffusion']['vae_latent_transformer_arch_probe_status']}`; "
        f"settings `{json.dumps(summary['level_c_diffusion']['vae_latent_transformer_arch_probe_settings'], sort_keys=True)}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['vae_latent_transformer_arch_probe_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Transformer parameter-count audit: "
        f"`{summary['level_c_diffusion']['transformer_parameter_count_audit_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['transformer_parameter_count_audit_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Transformer state-dict manifest: "
        f"`{summary['level_c_diffusion']['transformer_state_dict_manifest_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['transformer_state_dict_manifest_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Transformer EMA smoke: `{summary['level_c_diffusion']['transformer_ema_smoke_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['transformer_ema_smoke_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Debug-VAE-latent Transformer EMA smoke: "
        f"`{summary['level_c_diffusion']['vae_latent_transformer_ema_smoke_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['vae_latent_transformer_ema_smoke_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Diffusion checkpoint smoke: `{summary['level_c_diffusion']['diffusion_checkpoint_smoke_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['diffusion_checkpoint_smoke_metrics'], sort_keys=True)}`; "
        f"checkpoint `{summary['level_c_diffusion']['diffusion_checkpoint_smoke_checkpoint']}` is debug-only."
    )
    lines.append(
        f"- Bounded debug diffusion training run: "
        f"`{summary['level_c_diffusion']['bounded_debug_diffusion_training_run_status']}`; "
        f"metrics "
        f"`{json.dumps(summary['level_c_diffusion']['bounded_debug_diffusion_training_run_metrics'], sort_keys=True)}`; "
        f"run dir `{summary['level_c_diffusion']['bounded_debug_diffusion_training_run_run_dir']}`."
    )
    lines.append(
        f"- Bounded debug diffusion checkpoint eval: "
        f"`{summary['level_c_diffusion']['bounded_debug_diffusion_checkpoint_eval_status']}`; "
        f"rows `{json.dumps(summary['level_c_diffusion']['bounded_debug_diffusion_checkpoint_eval_rows'], sort_keys=True)}`."
    )
    lines.append(
        f"- Bounded debug diffusion action eval: "
        f"`{summary['level_c_diffusion']['bounded_debug_diffusion_action_eval_status']}`; "
        f"metrics "
        f"`{json.dumps(summary['level_c_diffusion']['bounded_debug_diffusion_action_eval_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Resource-adjusted tiny diffusion training run: "
        f"`{summary['level_c_diffusion']['resource_adjusted_tiny_diffusion_status']}`; "
        f"metrics "
        f"`{json.dumps(summary['level_c_diffusion']['resource_adjusted_tiny_diffusion_metrics'], sort_keys=True)}`; "
        f"run dir `{summary['level_c_diffusion']['resource_adjusted_tiny_diffusion_run_dir']}`."
    )
    lines.append(
        f"- Public LAFAN1 paper-architecture VAE/diffusion training: "
        f"`{summary['level_c_diffusion']['lafan1_paper_arch_training_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_training_metrics'], sort_keys=True)}`; "
        f"outputs `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_training_outputs'], sort_keys=True)}`."
    )
    lines.append(
        f"- Public LAFAN1 paper-architecture 3-seed statistics: "
        f"`{summary['level_c_diffusion']['lafan1_paper_arch_multiseed_status']}`; "
        f"statistics `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_multiseed_statistics'], sort_keys=True)}`; "
        f"outputs `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_multiseed_outputs'], sort_keys=True)}`."
    )
    lines.append(
        f"- Public LAFAN1 paper-architecture symmetry-augmented 3-seed statistics: "
        f"`{summary['level_c_diffusion']['lafan1_paper_arch_symmetry_multiseed_status']}`; "
        f"statistics `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_symmetry_multiseed_statistics'], sort_keys=True)}`; "
        f"outputs `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_symmetry_multiseed_outputs'], sort_keys=True)}`."
    )
    lines.append(
        f"- Public LAFAN1 paper-architecture 8-GPU high-memory batch audit: "
        f"`{summary['level_c_diffusion']['lafan1_paper_arch_high_memory_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_high_memory_metrics'], sort_keys=True)}`; "
        f"outputs `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_high_memory_outputs'], sort_keys=True)}`."
    )
    lines.append(
        f"- Public LAFAN1 paper-architecture symmetry-augmented dataset: "
        f"`{summary['level_c_diffusion']['lafan1_paper_arch_symmetry_dataset_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_symmetry_dataset_metrics'], sort_keys=True)}`; "
        f"outputs `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_symmetry_dataset_outputs'], sort_keys=True)}`."
    )
    lines.append(
        f"- Public LAFAN1 paper-architecture symmetry-augmented VAE/diffusion training: "
        f"`{summary['level_c_diffusion']['lafan1_paper_arch_symmetry_training_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_symmetry_training_metrics'], sort_keys=True)}`; "
        f"outputs `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_symmetry_training_outputs'], sort_keys=True)}`."
    )
    lines.append(
        f"- Public LAFAN1 paper-architecture ONNX/latency audit: "
        f"`{summary['level_c_diffusion']['lafan1_paper_arch_onnx_latency_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_onnx_latency_metrics'], sort_keys=True)}`; "
        f"outputs `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_onnx_latency_outputs'], sort_keys=True)}`."
    )
    lines.append(
        f"- Public LAFAN1 paper-architecture symmetry-augmented ONNX/latency audit: "
        f"`{summary['level_c_diffusion']['lafan1_paper_arch_symmetry_onnx_latency_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_symmetry_onnx_latency_metrics'], sort_keys=True)}`; "
        f"outputs `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_symmetry_onnx_latency_outputs'], sort_keys=True)}`."
    )
    lines.append(
        f"- Public LAFAN1 paper-architecture offline metrics audit: "
        f"`{summary['level_c_diffusion']['lafan1_paper_arch_offline_metrics_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_offline_metrics_metrics'], sort_keys=True)}`; "
        f"outputs `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_offline_metrics_outputs'], sort_keys=True)}`."
    )
    lines.append(
        f"- Public LAFAN1 paper-architecture symmetry-augmented offline metrics audit: "
        f"`{summary['level_c_diffusion']['lafan1_paper_arch_symmetry_offline_metrics_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_symmetry_offline_metrics_metrics'], sort_keys=True)}`; "
        f"outputs `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_symmetry_offline_metrics_outputs'], sort_keys=True)}`."
    )
    lines.append(
        f"- Public LAFAN1 paper-architecture offline guidance eval: "
        f"`{summary['level_c_diffusion']['lafan1_paper_arch_guidance_eval_status']}`; "
        f"task summaries `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_guidance_eval_task_summaries'], sort_keys=True)}`; "
        f"outputs `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_guidance_eval_outputs'], sort_keys=True)}`."
    )
    lines.append(
        f"- Public LAFAN1 paper-architecture symmetry-augmented offline guidance eval: "
        f"`{summary['level_c_diffusion']['lafan1_paper_arch_symmetry_guidance_eval_status']}`; "
        f"task summaries `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_symmetry_guidance_eval_task_summaries'], sort_keys=True)}`; "
        f"outputs `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_symmetry_guidance_eval_outputs'], sort_keys=True)}`."
    )
    lines.append(
        f"- Public LAFAN1 paper-architecture symmetry-augmented full-split offline guidance eval: "
        f"`{summary['level_c_diffusion']['lafan1_paper_arch_symmetry_guidance_eval_full_split_status']}`; "
        f"rows `{summary['level_c_diffusion']['lafan1_paper_arch_symmetry_guidance_eval_full_split_row_count']}`; "
        f"settings `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_symmetry_guidance_eval_full_split_settings'], sort_keys=True)}`; "
        f"task summaries `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_symmetry_guidance_eval_full_split_task_summaries'], sort_keys=True)}`; "
        f"outputs `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_symmetry_guidance_eval_full_split_outputs'], sort_keys=True)}`."
    )
    lines.append(
        f"- Public LAFAN1 paper-architecture reverse-denoising guidance audit: "
        f"`{summary['level_c_diffusion']['lafan1_paper_arch_reverse_guidance_status']}`; "
        f"improvement summary `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_reverse_guidance_improvement_summary'], sort_keys=True)}`; "
        f"outputs `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_reverse_guidance_outputs'], sort_keys=True)}`."
    )
    lines.append(
        f"- Public LAFAN1 paper-architecture symmetry-augmented reverse-denoising guidance audit: "
        f"`{summary['level_c_diffusion']['lafan1_paper_arch_symmetry_reverse_guidance_status']}`; "
        f"improvement summary `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_symmetry_reverse_guidance_improvement_summary'], sort_keys=True)}`; "
        f"outputs `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_symmetry_reverse_guidance_outputs'], sort_keys=True)}`."
    )
    lines.append(
        f"- Public LAFAN1 paper-architecture symmetry-augmented full-split reverse-denoising guidance audit: "
        f"`{summary['level_c_diffusion']['lafan1_paper_arch_symmetry_reverse_guidance_full_split_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_symmetry_reverse_guidance_full_split_metrics'], sort_keys=True)}`; "
        f"settings `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_symmetry_reverse_guidance_full_split_settings'], sort_keys=True)}`; "
        f"improvement summary `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_symmetry_reverse_guidance_full_split_improvement_summary'], sort_keys=True)}`; "
        f"outputs `{json.dumps(summary['level_c_diffusion']['lafan1_paper_arch_symmetry_reverse_guidance_full_split_outputs'], sort_keys=True)}`."
    )
    lines.append(
        f"- Resource-adjusted tiny diffusion suite: "
        f"`{summary['level_c_diffusion']['resource_adjusted_tiny_suite_status']}`; "
        f"`{summary['level_c_diffusion']['resource_adjusted_tiny_suite_pass_count']}/"
        f"{summary['level_c_diffusion']['resource_adjusted_tiny_suite_step_count']}` steps passed, metrics "
        f"`{json.dumps(summary['level_c_diffusion']['resource_adjusted_tiny_suite_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Resource-adjusted tiny diffusion multi-seed audit: "
        f"`{summary['level_c_diffusion']['resource_adjusted_tiny_multiseed_status']}`; "
        f"statistics "
        f"`{json.dumps(summary['level_c_diffusion']['resource_adjusted_tiny_multiseed_statistics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Resource-adjusted tiny diffusion checkpoint eval: "
        f"`{summary['level_c_diffusion']['resource_adjusted_tiny_checkpoint_eval_status']}`; "
        f"metrics "
        f"`{json.dumps(summary['level_c_diffusion']['resource_adjusted_tiny_checkpoint_eval_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Resource-adjusted tiny diffusion ONNX export/inference: "
        f"`{summary['level_c_diffusion']['resource_adjusted_tiny_onnx_status']}`; "
        f"metrics "
        f"`{json.dumps(summary['level_c_diffusion']['resource_adjusted_tiny_onnx_metrics'], sort_keys=True)}`; "
        f"ONNX `{summary['level_c_diffusion']['resource_adjusted_tiny_onnx_outputs']['onnx']}`."
    )
    lines.append(
        f"- Resource-adjusted tiny diffusion latency audit: "
        f"`{summary['level_c_diffusion']['resource_adjusted_tiny_latency_status']}`; "
        f"metrics "
        f"`{json.dumps(summary['level_c_diffusion']['resource_adjusted_tiny_latency_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Resource-adjusted tiny diffusion video preview: "
        f"`{summary['level_c_diffusion']['resource_adjusted_tiny_video_preview_status']}`; "
        f"rows `{json.dumps(summary['level_c_diffusion']['resource_adjusted_tiny_video_preview_rows'], sort_keys=True)}`."
    )
    lines.append(
        f"- Single-batch overfit gate: `{summary['level_c_diffusion']['single_batch_overfit_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['single_batch_overfit_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Single-motion overfit gate: `{summary['level_c_diffusion']['single_motion_overfit_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['single_motion_overfit_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Small-dataset overfit gate: `{summary['level_c_diffusion']['small_dataset_overfit_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['small_dataset_overfit_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Small-dataset split manifest: `{summary['level_c_diffusion']['small_dataset_split_status']}`; "
        f"counts `{json.dumps(summary['level_c_diffusion']['small_dataset_split_counts'], sort_keys=True)}`."
    )
    lines.append(
        f"- Small-dataset multi-seed audit: `{summary['level_c_diffusion']['small_dataset_multiseed_status']}`; "
        f"statistics `{json.dumps(summary['level_c_diffusion']['small_dataset_multiseed_statistics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Small-dataset held-out eval: `{summary['level_c_diffusion']['small_dataset_heldout_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['small_dataset_heldout_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Small-dataset held-out multi-seed audit: "
        f"`{summary['level_c_diffusion']['small_dataset_heldout_multiseed_status']}`; "
        f"statistics "
        f"`{json.dumps(summary['level_c_diffusion']['small_dataset_heldout_multiseed_statistics'], sort_keys=True)}`."
    )
    lines.append(
        f"- VAE checkpoint smoke: `{summary['level_c_diffusion']['vae_checkpoint_smoke_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['vae_checkpoint_smoke_metrics'], sort_keys=True)}`; "
        f"checkpoint `{summary['level_c_diffusion']['vae_checkpoint_smoke_checkpoint']}` is debug-only."
    )
    lines.append(
        f"- VAE debug overfit latent artifact: "
        f"`{summary['level_c_diffusion']['vae_debug_overfit_latent_artifact_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['vae_debug_overfit_latent_artifact_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- VAE motion-split held-out eval: "
        f"`{summary['level_c_diffusion']['vae_motion_split_heldout_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['vae_motion_split_heldout_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- VAE receding-horizon rollout smoke: "
        f"`{summary['level_c_diffusion']['vae_receding_horizon_rollout_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['vae_receding_horizon_rollout_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Diffusion-to-VAE action smoke: "
        f"`{summary['level_c_diffusion']['diffusion_to_vae_action_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['diffusion_to_vae_action_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Diffusion-to-VAE action multi-seed audit: "
        f"`{summary['level_c_diffusion']['diffusion_to_vae_action_multiseed_status']}`; "
        f"statistics `{json.dumps(summary['level_c_diffusion']['diffusion_to_vae_action_multiseed_statistics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Diffusion-to-VAE action smoothness audit: "
        f"`{summary['level_c_diffusion']['diffusion_to_vae_action_smoothness_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['diffusion_to_vae_action_smoothness_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Direct-vs-latent action ablation audit: "
        f"`{summary['level_c_diffusion']['direct_vs_latent_action_ablation_status']}`; "
        f"metrics "
        f"`{json.dumps(summary['level_c_diffusion']['direct_vs_latent_action_ablation_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- VAE contract audit: `{summary['level_c_diffusion']['vae_contract_audit_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['vae_contract_audit_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- DAgger-to-VAE debug pipeline audit: "
        f"`{summary['level_c_diffusion']['dagger_vae_pipeline_audit_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['dagger_vae_pipeline_audit_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- VAE latent probe: `{summary['level_c_diffusion']['vae_latent_probe_status']}`; "
        f"statistics `{json.dumps(summary['level_c_diffusion']['vae_latent_probe_statistics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Symmetry mapping audit: `{summary['level_c_diffusion']['symmetry_mapping_audit_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['symmetry_mapping_audit_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Guidance task scale sweep: `{summary['level_c_diffusion']['guidance_task_scale_sweep_status']}`; "
        f"`{summary['level_c_diffusion']['guidance_task_scale_sweep_row_count']}` rows; "
        f"task summaries `{json.dumps(summary['level_c_diffusion']['guidance_task_scale_sweep_task_summaries'], sort_keys=True)}`."
    )
    lines.append(
        f"- Guidance debug visualization: "
        f"`{summary['level_c_diffusion']['guidance_debug_visualization_status']}`; "
        f"primary metrics `{json.dumps(summary['level_c_diffusion']['guidance_debug_visualization_primary_metrics'], sort_keys=True)}`; "
        f"outputs `{json.dumps(summary['level_c_diffusion']['guidance_debug_visualization_outputs'], sort_keys=True)}`."
    )
    lines.append(
        f"- Guidance task metric audit: "
        f"`{summary['level_c_diffusion']['guidance_task_metric_audit_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['guidance_task_metric_audit_metrics'], sort_keys=True)}`; "
        f"primary metrics `{json.dumps(summary['level_c_diffusion']['guidance_task_metric_audit_primary_metrics'], sort_keys=True)}`; "
        f"full-split linkage included."
    )
    lines.append(
        f"- Guidance full-split result table: "
        f"`{summary['level_c_diffusion']['guidance_full_split_result_table_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['guidance_full_split_result_table_metrics'], sort_keys=True)}`; "
        f"mode summary `{json.dumps(summary['level_c_diffusion']['guidance_full_split_result_table_mode_summary'], sort_keys=True)}`; "
        f"outputs `{json.dumps(summary['level_c_diffusion']['guidance_full_split_result_table_outputs'], sort_keys=True)}`."
    )
    lines.append(
        f"- Guidance checkpoint visualization: "
        f"`{summary['level_c_diffusion']['guidance_checkpoint_visualization_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['guidance_checkpoint_visualization_metrics'], sort_keys=True)}`; "
        f"outputs `{json.dumps(summary['level_c_diffusion']['guidance_checkpoint_visualization_outputs'], sort_keys=True)}`."
    )
    lines.append(
        f"- Guidance visual deliverables audit: "
        f"`{summary['level_c_diffusion']['guidance_visual_deliverables_audit_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['guidance_visual_deliverables_audit_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Guidance cost coverage audit: "
        f"`{summary['level_c_diffusion']['guidance_cost_coverage_audit_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['guidance_cost_coverage_audit_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Core math unit tests: `{summary['level_c_diffusion']['core_math_unit_tests_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['core_math_unit_tests_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Timestep/mask source coverage audit: "
        f"`{summary['level_c_diffusion']['timestep_mask_coverage_audit_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['timestep_mask_coverage_audit_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Paper-state mask/reverse probe: "
        f"`{summary['level_c_diffusion']['paper_state_mask_reverse_probe_status']}`; "
        f"settings `{json.dumps(summary['level_c_diffusion']['paper_state_mask_reverse_probe_settings'], sort_keys=True)}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['paper_state_mask_reverse_probe_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Smoothness/latency audit: `{summary['level_c_diffusion']['smoothness_latency_audit_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['smoothness_latency_audit_metrics'], sort_keys=True)}`."
    )
    lines.append(
        f"- Deployment protocol audit: `{summary['level_c_diffusion']['deployment_protocol_audit_status']}`; "
        f"metrics `{json.dumps(summary['level_c_diffusion']['deployment_protocol_audit_metrics'], sort_keys=True)}`."
    )
    lines.append("")
    lines.append("## Paper Coverage")
    source_integrity = summary["download_source_integrity"]
    lines.append(
        f"- Download source integrity audit: `{source_integrity['status']}`; "
        f"category counts `{json.dumps(source_integrity['category_counts'], sort_keys=True)}`."
    )
    cov = summary["paper_source_coverage"]
    lines.append(f"- LaTeX/source coverage audit: `{cov['status']}`; counts `{json.dumps(cov['counts'], sort_keys=True)}`.")
    lines.append(f"- Coverage buckets: `{json.dumps(cov['bucket_counts'], sort_keys=True)}`.")
    latex = summary["paper_latex_inventory"]
    lines.append(
        f"- LaTeX inventory audit: `{latex['status']}`; counts `{json.dumps(latex['counts'], sort_keys=True)}`; "
        f"equation topics `{json.dumps(latex['equation_topic_counts'], sort_keys=True)}`."
    )
    formula_trace = summary["paper_formula_code_trace"]
    lines.append(
        f"- Paper formula/code trace audit: `{formula_trace['status']}`; `{formula_trace['row_count']}` rows, "
        f"missing evidence `{formula_trace['missing_evidence_row_count']}`, source counts "
        f"`{json.dumps(formula_trace['source_counts'], sort_keys=True)}`."
    )
    pdf = summary["paper_pdf_source_consistency"]
    lines.append(
        f"- PDF/source consistency audit: `{pdf['status']}`; "
        f"metrics `{json.dumps(pdf['metrics'], sort_keys=True)}`."
    )
    vals = summary["paper_table_values"]
    lines.append(f"- Table value audit: `{vals['status']}`; counts `{json.dumps(vals['counts'], sort_keys=True)}`.")
    skill = summary["skill_success_table"]
    lines.append(
        f"- Skill-success table data audit: `{skill['status']}`; "
        f"metrics `{json.dumps(skill['metrics'], sort_keys=True)}`; "
        f"missing LAFAN CSVs `{json.dumps(skill['missing_lafan_csv_names'])}`."
    )
    cmp = summary["paper_vs_reproduction"]
    lines.append(
        f"- Paper-vs-reproduction comparison: `{cmp['status']}`; `{cmp['total_rows']}` rows, "
        f"type counts `{json.dumps(cmp['comparison_type_counts'], sort_keys=True)}`, "
        f"missing goal checkpoint rows `{len(cmp['missing_goal_checkpoint_rows'])}`."
    )
    results = summary["results_claims"]
    lines.append(
        f"- Results claims audit: `{results['status']}`; "
        f"metrics `{json.dumps(results['metrics'], sort_keys=True)}`; "
        f"status counts `{json.dumps(results['local_status_counts'], sort_keys=True)}`."
    )
    trace = summary["goal_traceability"]
    lines.append(
        f"- Goal traceability audit: `{trace['status']}`; `{trace['trace_row_count']}` trace rows over "
        f"`{trace['heading_count']}` headings, status counts `{json.dumps(trace['status_counts'], sort_keys=True)}`, "
        f"missing evidence rows `{trace['missing_evidence_rows']}`."
    )
    directives = summary["goal_directive_index"]
    lines.append(
        f"- goal.md directive index: `{directives['status']}`; `{directives['directive_row_count']}` directive rows over "
        f"`{directives['line_count']}` lines and `{directives['heading_count']}` headings, tag counts "
        f"`{json.dumps(directives['tag_counts'], sort_keys=True)}`."
    )
    matrix = summary["goal_requirement_matrix"]
    lines.append(
        f"- Goal requirement matrix: `{matrix['status']}`; `{matrix['requirement_row_count']}` requirement rows over "
        f"`{matrix['goal_line_count']}` goal.md lines, status counts "
        f"`{json.dumps(matrix['status_counts'], sort_keys=True)}`, missing evidence rows "
        f"`{matrix['missing_evidence_rows']}`."
    )
    lines.append("")
    lines.append("## Blocked Gates")
    lines.append(f"- Gate status counts: `{json.dumps(summary['blocked_gates']['gate_status_counts'], sort_keys=True)}`.")
    for gate in summary["blocked_gates"]["gates"]:
        lines.append(f"- `{gate['gate_id']}`: `{gate['status']}`; blocks {', '.join(gate['blocks'])}.")
    lines.append("")
    lines.append("## Goal.md Final Report Requirements")
    lines.append("- Official code used: Level B tracking/configuration audits use the downloaded official `whole_body_tracking` and `motion_tracking_controller` trees where accessible; live IsaacLab/Kit and ROS deployment execution remain blocked.")
    lines.append("- Paper-faithful reimplementation: Level C VAE, diffusion, trajectory transforms, masks, and guidance mechanics are reimplemented from paper formulas and local source audits as debug/package evidence, not as unpublished official checkpoints.")
    lines.append("- Released-data reproduction: Level A uses public released data for directly redrawable panels and released-figure summaries.")
    lines.append("- Retrained results: no paper-scale PPO, VAE, diffusion, TensorRT, or real-robot result has been retrained to completion; only debug overfit, held-out, and multi-seed smoke probes were run.")
    lines.append("- Qualitative-only comparison: qualitative/debug-only rows are separated in `res/comparison/paper_vs_reproduction.csv` and the Results claims audit.")
    lines.append("- Not publicly reproducible: missing official Level C code/checkpoints, Fig. 5/Fig. 6 rollout data, TensorRT engine, and Unitree G1 hardware evidence are recorded as blocked or out of scope.")
    lines.append("- Result differences: paper-vs-reproduction comparison rows record exact, approximate, qualitative-only, not-publicly-reproducible, and real-robot-required comparison types.")
    lines.append("- Difference sources: blocked gates, adaptive-sampling discrepancy, missing checkpoints, missing deployment stack, and hardware absence are recorded as likely sources.")
    lines.append("- Current reproduction credibility: strong for inventory, paper/source value audits, released-data panels, and unit-tested formula mechanics; partial or debug-only for tracking and Level C; blocked for paper-level deployment and Fig. 5/Fig. 6.")
    lines.append("- Completed and incomplete scope: completion matrix, goal traceability audit, and blocked-gate audit separate complete, partial, blocked, and out-of-scope items.")
    lines.append("- Hardware cost and training time: current evidence records non-training GPU resource snapshots, diagnostic/debug runtime artifacts, and failed-run retention; full training hardware cost and wall-clock time are missing because no long paper-scale training reached SUCCESS.")
    lines.append("- One-command rerun path: the Verification Commands section and `reproduction/RUNBOOK.md` provide the current rerun sequence for all audited artifacts.")
    lines.append("")
    lines.append("## Key Evidence")
    evidence_paths = [
        "res/master_audit/reproduction_master_audit.json",
        "res/blocked_gates/blocked_gate_audit.json",
        "res/tracking/tracking_smoke_rerun_audit/tracking_smoke_rerun_audit.json",
        "res/tracking/official_train_entry_retry_audit/tracking_official_train_entry_retry_audit.json",
        "logs/tracking_official_train_entry_retry/official_train_entry_retry.log",
        "logs/tracking_official_train_entry_retry/official_train_entry_retry_tail.log",
        "res/setup/kit_inotify_budget_audit/kit_inotify_budget_audit.json",
        "res/setup/inotify_live_usage_audit/inotify_live_usage_audit.json",
        "res/setup/vscode_watcher_exclude_audit/vscode_watcher_exclude_audit.json",
        ".vscode/settings.json",
        "res/setup/kit_watcher_config_surface_audit/kit_watcher_config_surface_audit.json",
        "res/tracking/tracking_import_gate_audit/tracking_import_gate_audit.json",
        "res/tracking/tracking_extension_namespace_probe/tracking_extension_namespace_probe.json",
        "res/tracking/official_source_contract_audit/tracking_official_source_contract_audit.json",
        "res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json",
        "res/tracking/reward_formula_audit/tracking_reward_formula_audit.json",
        "res/tracking/observation_action_schema_audit/tracking_observation_action_schema_audit.json",
        "res/tracking/randomization_termination_audit/tracking_randomization_termination_audit.json",
        "res/tracking/nonkit_suite/level_b_tracking_nonkit_suite.json",
        "res/tracking/adaptive_sampling_discrepancy_audit/adaptive_sampling_discrepancy_audit.json",
        "res/tracking/motion_preprocessing_contract_audit/motion_preprocessing_contract_audit.json",
        "res/tracking/local_smoke_preflight/tracking_local_smoke_preflight.json",
        "res/tracking/motion_npz_fixture/tracking_motion_npz_fixture.json",
        "res/tracking/official_replay_preflight/tracking_official_replay_preflight.json",
        "res/tracking/official_replay_conversion/tracking_official_replay_conversion_audit.json",
        "res/tracking/urdf_conversion_probe/tracking_urdf_conversion_probe.json",
        "res/tracking/urdf_path_tiny_probe/tracking_urdf_path_tiny_probe.json",
        "res/tracking/mjcf_stage_probe/tracking_mjcf_stage_probe.json",
        "res/tracking/usd_save_policy_probe/tracking_usd_save_policy_probe.json",
        "res/tracking/simulationapp_save_policy_probe/tracking_simulationapp_save_policy_probe.json",
        "res/tracking/mujoco_ros_launch_contract_audit/mujoco_ros_launch_contract_audit.json",
        "res/tracking/deployment_controller_semantics_audit/tracking_deployment_controller_semantics_audit.json",
        "res/tracking/onnx_export_contract_audit/tracking_onnx_export_contract_audit.json",
        "res/tracking/motion_policy_onnx_contract_fixture/tracking_motion_policy_onnx_contract_fixture.json",
        "res/tracking/debug_motion_policy_onnx_export/tracking_debug_motion_policy_onnx_export.json",
        "res/tracking/debug_motion_policy_onnx_export/debug_motion_policy_contract.onnx",
        "res/takeover_audit/takeover_audit.json",
        "res/setup/env_probe/env_import_probe.json",
        "res/setup/bm_diffusion_env_audit/bm_diffusion_env_audit.json",
        "res/setup/gpu_resource_audit/gpu_resource_audit.json",
        "logs/gpu/gpu_metrics.csv",
        "res/run_management_audit/run_management_audit.json",
        "res/runs/setup_run_management_diagnostic_static_000_20260617_050000/status.json",
        "res/run_management_audit/checkpoint_resume_smoke/checkpoint_resume_smoke.json",
        "res/runs/setup_checkpoint_resume_smoke_static_000_20260617_061500/status.json",
        "res/run_management_audit/full_run_deliverable_gap_audit/full_run_deliverable_gap_audit.json",
        "res/failed_runs/failed_run_audit/failed_run_audit.json",
        "res/failed_runs/phase1_isaaclab_headless_smoke_g1_inotify_0_20260616_200654/status.json",
        "res/failed_runs/official_train_entry_failed_run_audit/official_train_entry_failed_run_audit.json",
        "res/failed_runs/phase1_official_train_entry_retry_inotify_0_20260617_174742/status.json",
        "res/code/patch_inventory_audit/patch_inventory_audit.json",
        "res/code/patch_snapshot_audit/patch_snapshot_audit.json",
        "res/code/reimpl_package_audit/reimpl_package_audit.json",
        "res/code/reimpl_runtime_integration_audit/reimpl_runtime_integration_audit.json",
        "res/code/coding_requirements_audit/coding_requirements_audit.json",
        "res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json",
        "res/tests/reimpl_test_suite/reimpl_test_suite.json",
        "reproduction/src/beyondmimic_reimpl/__init__.py",
        "res/config/resolved_reproduction_config.json",
        "res/config/resolved_reproduction_config.yaml",
        "res/artifact_manifest/artifact_manifest.json",
        "res/artifact_manifest/artifact_manifest.tsv",
        "res/source_integrity/download_source_integrity/download_source_integrity_audit.json",
        "res/source_integrity/download_source_integrity/download_source_integrity_manifest.tsv",
        "res/source_integrity/download_source_integrity/download_source_integrity_required.tsv",
        "res/run_log_config_catalog/run_log_config_catalog.json",
        "res/run_log_config_catalog/run_log_config_catalog.csv",
        "README.md",
        "res/final_report/reproduction_report.md",
        "res/final_report/final_report_requirement_audit/final_report_requirement_audit.json",
        "res/final_deliverables_audit/final_deliverables_audit.json",
        "res/visual_media_inventory/visual_media_inventory_audit.json",
        "res/visual_media_inventory/visual_media_inventory_audit.tsv",
        "res/verification_command_coverage/verification_command_coverage_audit.json",
        "res/verification_command_syntax/verification_command_syntax_audit.json",
        "res/verification_command_script_manifest/verification_command_script_manifest.json",
        "res/required_artifact_absence/required_artifact_absence_audit.json",
        "res/evaluation/trial_failure_accounting_audit/trial_failure_accounting_audit.json",
        "res/evaluation_metrics_coverage/evaluation_metrics_coverage_audit.json",
        "res/ablation_coverage/ablation_coverage_audit.json",
        "res/metrics/metrics_catalog/metrics_catalog.json",
        "res/metrics/metrics_catalog/metrics_catalog.csv",
        "res/tables/released_data_metrics_summary/released_data_metrics_summary.json",
        "res/tables/released_data_metrics_summary/released_tracking_ablation_metrics.csv",
        "res/tables/released_data_metrics_summary/released_grf_metrics.csv",
        "res/tables/released_data_metrics_summary/released_imu_metrics.csv",
        "res/tables/released_data_statistical_audit/released_data_statistical_audit.json",
        "res/tables/released_data_statistical_audit/released_ablation_effect_sizes.csv",
        "res/tables/released_data_statistical_audit/released_grf_confidence_intervals.csv",
        "res/tables/released_data_statistical_audit/released_imu_confidence_intervals.csv",
        "res/level_a/released_data_suite/level_a_released_data_suite.json",
        "res/guidance_task_coverage/guidance_task_coverage_audit.json",
        "res/progress_report_audit/progress_report_audit.json",
        "res/docs/completion_matrix_status_audit/completion_matrix_status_audit.json",
        "res/project_boundary_audit/project_boundary_audit.json",
        "reproduction/docs/experiment_protocol.md",
        "res/docs/experiment_protocol_audit/experiment_protocol_audit.json",
        "res/docs/readme_audit/readme_audit.json",
        "res/paper_source_coverage/paper_source_coverage_audit.json",
        "res/paper_latex_inventory/paper_latex_inventory_audit.json",
        "res/paper_latex_inventory/paper_latex_equations.tsv",
        "res/paper_latex_inventory/paper_latex_experiment_settings.tsv",
        "res/paper_formula_code_trace/paper_formula_code_trace_audit.json",
        "res/paper_formula_code_trace/paper_formula_code_trace_audit.tsv",
        "res/paper_pdf_source_consistency/paper_pdf_source_consistency_audit.json",
        "res/paper_pdf_source_consistency/paper_pdf_anchor_audit.tsv",
        "res/paper_pdf_source_consistency/paper_source_tar_audit.tsv",
        "res/paper_table_values/paper_table_value_audit.json",
        "res/paper_skill_success_table_audit/skill_success_table_data_audit.json",
        "res/released_panel_mapping_audit/released_panel_mapping_audit.json",
        "res/comparison/paper_vs_reproduction.csv",
        "res/comparison/paper_vs_reproduction.md",
        "res/level_c/diffusion_equation_audit/level_c_diffusion_equation_audit.json",
        "res/results_claims_audit/results_claims_audit.json",
        "res/level_c/trajectory_inverse_transform_audit/level_c_trajectory_inverse_transform_audit.json",
        "res/level_c/emphasis_projection_audit/level_c_emphasis_projection_audit.json",
        "res/level_c/state_representation_source_audit/level_c_state_representation_source_audit.json",
        "res/level_c/dataset_collection_protocol_audit/level_c_dataset_collection_protocol_audit.json",
        "res/level_c/rollout_rejection_manifest_probe/level_c_rollout_rejection_manifest_probe.json",
        "res/level_c/state_latent_schema_audit/state_latent_schema_audit.json",
        "res/level_c/dagger_schema_audit/dagger_schema_audit.json",
        "res/level_c/dagger_iteration_smoke/level_c_dagger_iteration_smoke.json",
        "res/level_c/paper_state_windows/level_c_paper_state_windows.json",
        "res/level_c/state_latent_dataset_consistency_audit/level_c_state_latent_dataset_consistency_audit.json",
        "res/level_c/state_latent_training_dataset_contract_audit/"
        "level_c_state_latent_training_dataset_contract_audit.json",
        "res/level_c/paper_state_overfit_probe/level_c_paper_state_overfit_probe.json",
        "res/level_c/vae_latent_diffusion_overfit_probe/level_c_vae_latent_diffusion_overfit_probe.json",
        "res/level_c/paper_state_heldout_eval/level_c_paper_state_heldout_eval.json",
        "res/level_c/vae_latent_heldout_eval/level_c_vae_latent_heldout_eval.json",
        "res/level_c/paper_state_heldout_multiseed_audit/level_c_paper_state_heldout_multiseed_audit.json",
        "res/level_c/vae_latent_heldout_multiseed_audit/level_c_vae_latent_heldout_multiseed_audit.json",
        "res/level_c/paper_state_transformer_arch_probe/level_c_paper_state_transformer_arch_probe.json",
        "res/level_c/vae_latent_transformer_arch_probe/level_c_vae_latent_transformer_arch_probe.json",
        "res/level_c/transformer_parameter_count_audit/level_c_transformer_parameter_count_audit.json",
        "res/level_c/transformer_state_dict_manifest/level_c_transformer_state_dict_manifest.json",
        "res/level_c/transformer_ema_smoke/level_c_transformer_ema_smoke.json",
        "res/level_c/vae_latent_transformer_ema_smoke/level_c_vae_latent_transformer_ema_smoke.json",
        "res/level_c/diffusion_checkpoint_smoke/level_c_diffusion_checkpoint_smoke.json",
        "res/level_c/bounded_debug_diffusion_training_run/level_c_bounded_debug_diffusion_training_run.json",
        "res/runs/level_c_bounded_debug_diffusion_static_000_20260617_083000/status.json",
        "res/runs/level_c_bounded_debug_diffusion_static_000_20260617_083000/metrics.json",
        "res/runs/level_c_bounded_debug_diffusion_static_000_20260617_083000/figures/debug_training_loss.png",
        "res/level_c/bounded_debug_diffusion_checkpoint_eval/level_c_bounded_debug_diffusion_checkpoint_eval.json",
        "res/level_c/bounded_debug_diffusion_action_eval/level_c_bounded_debug_diffusion_action_eval.json",
        "res/level_c/resource_adjusted_tiny_diffusion_training_run/level_c_resource_adjusted_tiny_diffusion_training_run.json",
        "res/level_c/resource_adjusted_tiny_diffusion_suite/level_c_resource_adjusted_tiny_diffusion_suite.json",
        "res/level_c/resource_adjusted_tiny_diffusion_suite/level_c_resource_adjusted_tiny_diffusion_suite.tsv",
        "res/level_c/resource_adjusted_tiny_diffusion_multiseed_audit/level_c_resource_adjusted_tiny_diffusion_multiseed_audit.json",
        "res/level_c/resource_adjusted_tiny_diffusion_multiseed_audit/level_c_resource_adjusted_tiny_diffusion_multiseed_audit.tsv",
        "res/level_c/resource_adjusted_tiny_diffusion_multiseed_audit/level_c_resource_adjusted_tiny_diffusion_multiseed_audit.npz",
        "res/level_c/resource_adjusted_tiny_diffusion_checkpoint_eval/level_c_resource_adjusted_tiny_diffusion_checkpoint_eval.json",
        "res/level_c/resource_adjusted_tiny_diffusion_checkpoint_eval/level_c_resource_adjusted_tiny_diffusion_checkpoint_eval.tsv",
        "res/level_c/resource_adjusted_tiny_diffusion_checkpoint_eval/level_c_resource_adjusted_tiny_diffusion_checkpoint_eval.npz",
        "res/level_c/resource_adjusted_tiny_diffusion_onnx_export_inference/level_c_resource_adjusted_tiny_diffusion_onnx_export_inference.json",
        "res/level_c/resource_adjusted_tiny_diffusion_onnx_export_inference/level_c_resource_adjusted_tiny_diffusion_onnx_export_inference.tsv",
        "res/level_c/resource_adjusted_tiny_diffusion_onnx_export_inference/resource_adjusted_tiny_denoiser_onnx_debug_io.npz",
        "res/level_c/resource_adjusted_tiny_diffusion_onnx_export_inference/resource_adjusted_tiny_denoiser_debug.onnx",
        "res/level_c/resource_adjusted_tiny_diffusion_latency_audit/level_c_resource_adjusted_tiny_diffusion_latency_audit.json",
        "res/level_c/resource_adjusted_tiny_diffusion_latency_audit/level_c_resource_adjusted_tiny_diffusion_latency_audit.tsv",
        "res/level_c/resource_adjusted_tiny_diffusion_latency_audit/level_c_resource_adjusted_tiny_diffusion_latency_audit.npz",
        "res/level_c/resource_adjusted_tiny_diffusion_video_preview/level_c_resource_adjusted_tiny_diffusion_video_preview.json",
        "res/level_c/resource_adjusted_tiny_diffusion_video_preview/level_c_resource_adjusted_tiny_diffusion_video_preview.tsv",
        "res/level_c/resource_adjusted_tiny_diffusion_video_preview/tiny_diffusion_validation_debug_preview_poster.png",
        "res/level_c/resource_adjusted_tiny_diffusion_video_preview/tiny_diffusion_test_debug_preview_poster.png",
        "res/runs/level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500/videos/tiny_diffusion_validation_debug_preview.gif",
        "res/runs/level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500/videos/tiny_diffusion_test_debug_preview.gif",
        "res/runs/level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500/status.json",
        "res/runs/level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500/metrics.json",
        "res/runs/level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500/figures/tiny_denoiser_loss.png",
        "res/runs/level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500/figures/tiny_denoiser_eval_mse.png",
        "res/level_c/lafan1_paper_arch_multiseed_audit/level_c_lafan1_paper_arch_multiseed_audit.json",
        "res/level_c/lafan1_paper_arch_multiseed_audit/level_c_lafan1_paper_arch_multiseed_rows.tsv",
        "res/level_c/lafan1_paper_arch_multiseed_audit/level_c_lafan1_paper_arch_multiseed_metrics.npz",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_multiseed_audit/level_c_lafan1_paper_arch_symmetry_augmented_multiseed_audit.json",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_multiseed_audit/level_c_lafan1_paper_arch_symmetry_augmented_multiseed_rows.tsv",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_multiseed_audit/level_c_lafan1_paper_arch_symmetry_augmented_multiseed_metrics.npz",
        "res/level_c/lafan1_paper_arch_high_memory_batch_audit/level_c_lafan1_paper_arch_high_memory_batch_audit.json",
        "res/level_c/lafan1_paper_arch_high_memory_batch_audit/level_c_lafan1_paper_arch_high_memory_batch_rows.tsv",
        "res/level_c/lafan1_paper_arch_high_memory_batch_audit/level_c_lafan1_paper_arch_high_memory_batch_fixture.npz",
        "res/level_c/lafan1_paper_arch_symmetry_dataset/lafan1_paper_arch_symmetry_dataset_audit.json",
        "res/level_c/lafan1_paper_arch_symmetry_dataset/level_c_lafan1_paper_arch_symmetry_dataset_splits.tsv",
        "res/level_c/lafan1_paper_arch_symmetry_dataset/lafan1_paper_arch_symmetry_augmented_dataset.npz",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_training/lafan1_paper_arch_vae_diffusion_training.json",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_training/lafan1_paper_arch_training_dataset.npz",
        "res/runs/level_c_lafan1_paper_arch_symmetry_augmented_static_000_20260617_215500/checkpoint/lafan1_paper_arch_vae_diffusion.pt",
        "res/runs/level_c_lafan1_paper_arch_symmetry_augmented_static_000_20260617_215500/metrics.json",
        "res/level_c/lafan1_paper_arch_symmetry_training_comparison/level_c_lafan1_paper_arch_symmetry_training_comparison_audit.json",
        "res/level_c/lafan1_paper_arch_symmetry_training_comparison/level_c_lafan1_paper_arch_symmetry_training_comparison.tsv",
        "res/level_c/lafan1_paper_arch_symmetry_training_comparison/level_c_lafan1_paper_arch_symmetry_training_comparison_metrics.npz",
        "res/level_c/lafan1_paper_arch_onnx_latency/level_c_lafan1_paper_arch_onnx_latency_audit.json",
        "res/level_c/lafan1_paper_arch_onnx_latency/level_c_lafan1_paper_arch_onnx_latency_components.tsv",
        "res/level_c/lafan1_paper_arch_onnx_latency/level_c_lafan1_paper_arch_onnx_latency_rows.tsv",
        "res/level_c/lafan1_paper_arch_onnx_latency/lafan1_paper_arch_onnx_io_fixture.npz",
        "res/level_c/lafan1_paper_arch_onnx_latency/lafan1_paper_arch_vae_decoder.onnx",
        "res/level_c/lafan1_paper_arch_onnx_latency/lafan1_paper_arch_diffusion_denoiser.onnx",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/level_c_lafan1_paper_arch_onnx_latency_audit.json",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/level_c_lafan1_paper_arch_onnx_latency_components.tsv",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/level_c_lafan1_paper_arch_onnx_latency_rows.tsv",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/lafan1_paper_arch_onnx_io_fixture.npz",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/lafan1_paper_arch_vae_decoder.onnx",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/lafan1_paper_arch_diffusion_denoiser.onnx",
        "res/level_c/lafan1_paper_arch_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_audit.json",
        "res/level_c/lafan1_paper_arch_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_rows.tsv",
        "res/level_c/lafan1_paper_arch_offline_metrics/level_c_lafan1_paper_arch_offline_metrics.npz",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_audit.json",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_rows.tsv",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_offline_metrics/level_c_lafan1_paper_arch_offline_metrics.npz",
        "res/level_c/lafan1_paper_arch_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.json",
        "res/level_c/lafan1_paper_arch_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.tsv",
        "res/level_c/lafan1_paper_arch_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.npz",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.json",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.tsv",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.npz",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/level_c_lafan1_paper_arch_guidance_eval.json",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/level_c_lafan1_paper_arch_guidance_eval.tsv",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/level_c_lafan1_paper_arch_guidance_eval.npz",
        "res/level_c/lafan1_paper_arch_reverse_guidance/level_c_lafan1_paper_arch_reverse_guidance_audit.json",
        "res/level_c/lafan1_paper_arch_reverse_guidance/level_c_lafan1_paper_arch_reverse_guidance_rows.tsv",
        "res/level_c/lafan1_paper_arch_reverse_guidance/level_c_lafan1_paper_arch_reverse_guidance.npz",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance/level_c_lafan1_paper_arch_reverse_guidance_audit.json",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance/level_c_lafan1_paper_arch_reverse_guidance_rows.tsv",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance/level_c_lafan1_paper_arch_reverse_guidance.npz",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.json",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/level_c_lafan1_paper_arch_reverse_guidance_full_split_rows.tsv",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/level_c_lafan1_paper_arch_reverse_guidance_full_split_gpu_rows.tsv",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/level_c_lafan1_paper_arch_reverse_guidance_full_split.npz",
        "res/level_c/debug_suite/level_c_debug_suite.json",
        "res/level_c/extended_debug_suite/level_c_extended_debug_suite.json",
        "res/level_c/single_batch_overfit_probe/level_c_single_batch_overfit_probe.json",
        "res/level_c/single_motion_overfit_probe/level_c_single_motion_overfit_probe.json",
        "res/level_c/small_dataset_overfit_probe/level_c_small_dataset_overfit_probe.json",
        "res/level_c/small_dataset_split_manifest/small_dataset_split_manifest_summary.json",
        "res/level_c/small_dataset_multiseed_audit/level_c_small_dataset_multiseed_audit.json",
        "res/level_c/small_dataset_heldout_eval/level_c_small_dataset_heldout_eval.json",
        "res/level_c/small_dataset_heldout_multiseed_audit/level_c_small_dataset_heldout_multiseed_audit.json",
        "res/level_c/vae_checkpoint_smoke/level_c_vae_checkpoint_smoke.json",
        "res/level_c/vae_debug_overfit_latent_artifact/level_c_vae_debug_overfit_latent_artifact.json",
        "res/level_c/vae_motion_split_heldout_eval/level_c_vae_motion_split_heldout_eval.json",
        "res/level_c/vae_receding_horizon_rollout_smoke/level_c_vae_receding_horizon_rollout_smoke.json",
        "res/level_c/diffusion_to_vae_action_smoke/level_c_diffusion_to_vae_action_smoke.json",
        "res/level_c/diffusion_to_vae_action_multiseed_audit/level_c_diffusion_to_vae_action_multiseed_audit.json",
        "res/level_c/diffusion_to_vae_action_smoothness_audit/level_c_diffusion_to_vae_action_smoothness_audit.json",
        "res/level_c/direct_vs_latent_action_ablation_audit/level_c_direct_vs_latent_action_ablation_audit.json",
        "res/level_c/vae_contract_audit/level_c_vae_contract_audit.json",
        "res/level_c/dagger_vae_pipeline_audit/level_c_dagger_vae_pipeline_audit.json",
        "res/level_c/vae_latent_probe/level_c_vae_latent_probe.json",
        "res/level_c/symmetry_mapping_audit/level_c_symmetry_mapping_audit.json",
        "res/level_c/guidance_task_scale_sweep/level_c_guidance_task_scale_sweep.json",
        "res/level_c/guidance_debug_visualization/level_c_guidance_debug_visualization.json",
        "res/level_c/guidance_task_metric_audit/level_c_guidance_task_metric_audit.json",
        "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.json",
        "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.tsv",
        "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.csv",
        "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_cost_delta.pdf",
        "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_cost_delta.svg",
        "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_cost_delta.png",
        "res/level_c/guidance_checkpoint_visualization/level_c_guidance_checkpoint_visualization.json",
        "res/level_c/guidance_checkpoint_visualization/level_c_guidance_checkpoint_visualization.tsv",
        "res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_joystick.png",
        "res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_waypoint.png",
        "res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_obstacle_avoidance.png",
        "res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_inpainting.png",
        "res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_composed_objectives.png",
        "res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_composed_preview.gif",
        "res/level_c/guidance_visual_deliverables_audit/level_c_guidance_visual_deliverables_audit.json",
        "res/level_c/guidance_cost_coverage_audit/level_c_guidance_cost_coverage_audit.json",
        "res/tests/core_math_unit_tests/core_math_unit_tests.json",
        "res/tests/core_test_coverage_audit/core_test_coverage_audit.json",
        "res/level_c/timestep_mask_coverage_audit/level_c_timestep_mask_coverage_audit.json",
        "res/level_c/paper_state_mask_reverse_probe/level_c_paper_state_mask_reverse_probe.json",
        "res/level_c/smoothness_latency_audit/level_c_smoothness_latency_audit.json",
        "res/level_c/deployment_protocol_audit/level_c_deployment_protocol_audit.json",
        "res/goal_traceability/goal_traceability_audit.json",
        "res/goal_directive_index/goal_directive_index_audit.json",
        "res/goal_directive_index/goal_directive_rows.tsv",
        "res/goal_directive_index/goal_heading_rows.tsv",
        "res/goal_requirement_matrix/goal_requirement_matrix_audit.json",
        "res/released_figures/released_figure_summary.tsv",
        "reproduction/docs/completion_matrix.md",
        "reproduction/docs/level_c_diffusion_plan.md",
        "reproduction/docs/unresolved_details.md",
    ]
    for path in evidence_paths:
        lines.append(f"- {md_link(ROOT / path)}")
    lines.append("")
    lines.append("## Verification Commands")
    lines.extend(f"```bash\n{cmd}\n```" for cmd in summary["verification_commands"])
    lines.append("")
    lines.append("## Boundary")
    lines.append("The report is a consolidation artifact. It does not replace live IsaacLab rollouts, true DAgger data collection, trained VAE/diffusion checkpoints, TensorRT deployment, Fig. 5/Fig. 6 paper reproduction, or real Unitree G1 execution.")
    text = "\n".join(lines) + "\n"
    atomic_write_text(DOC_OUT, text)
    atomic_write_text(GOAL_DOC_OUT, text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DOC_OUT.parent.mkdir(parents=True, exist_ok=True)
    GOAL_DOC_OUT.parent.mkdir(parents=True, exist_ok=True)
    summary = gather_summary()
    json_path = OUT / "final_reproduction_report.json"
    write_markdown(summary)
    atomic_write_text(json_path, json.dumps(summary, indent=2, sort_keys=True))
    print(
        json.dumps(
            {"status": "ok", "json": str(json_path), "markdown": str(DOC_OUT), "goal_markdown": str(GOAL_DOC_OUT)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
