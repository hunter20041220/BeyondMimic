#!/usr/bin/env python3
"""Create a SHA256 manifest for key reproduction deliverables."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/infini-data/test/BeyondMimic")
OUT = ROOT / "res/artifact_manifest"

ARTIFACTS = [
    ("raw_paper_pdf", "download/papers/BeyondMimic_2508.08241.pdf", "raw_source"),
    ("raw_paper_source_tar", "download/papers/BeyondMimic_2508.08241_source.tar", "raw_source"),
    (
        "download_source_integrity",
        "res/source_integrity/download_source_integrity/download_source_integrity_audit.json",
        "raw_source",
    ),
    ("top_level_readme", "README.md", "documentation"),
    ("source_ledger", "reproduction/docs/source_ledger.md", "documentation"),
    ("local_inventory", "reproduction/docs/local_inventory.tsv", "documentation"),
    ("environment_doc", "reproduction/docs/environment.md", "documentation"),
    ("experiment_protocol", "reproduction/docs/experiment_protocol.md", "documentation"),
    ("bm_analysis_lock", "envs/bm_analysis/requirements-lock.txt", "environment"),
    ("bm_tracking_lock", "envs/bm_tracking/requirements-lock.txt", "environment"),
    ("bm_diffusion_lock", "envs/bm_diffusion/requirements-lock.txt", "environment"),
    ("takeover_audit", "res/takeover_audit/takeover_audit.json", "environment"),
    ("env_import_probe", "res/setup/env_probe/env_import_probe.json", "environment"),
    ("resolved_config_json", "res/config/resolved_reproduction_config.json", "config"),
    ("resolved_config_yaml", "res/config/resolved_reproduction_config.yaml", "config"),
    ("core_math_tests", "reproduction/tests/test_core_math.py", "code"),
    ("reimpl_package_api_tests", "res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json", "code_audit"),
    ("reimpl_test_suite", "res/tests/reimpl_test_suite/reimpl_test_suite.json", "code_audit"),
    ("core_test_coverage_audit", "res/tests/core_test_coverage_audit/core_test_coverage_audit.json", "code_audit"),
    ("coding_requirements_audit", "res/code/coding_requirements_audit/coding_requirements_audit.json", "code_audit"),
    ("patch_inventory_audit", "res/code/patch_inventory_audit/patch_inventory_audit.json", "code_audit"),
    ("patch_snapshot_audit", "res/code/patch_snapshot_audit/patch_snapshot_audit.json", "code_audit"),
    ("reimpl_package_audit", "res/code/reimpl_package_audit/reimpl_package_audit.json", "code_audit"),
    (
        "reimpl_runtime_integration_audit",
        "res/code/reimpl_runtime_integration_audit/reimpl_runtime_integration_audit.json",
        "code_audit",
    ),
    ("gpu_metrics_csv", "logs/gpu/gpu_metrics.csv", "run_log"),
    ("run_log_config_catalog", "res/run_log_config_catalog/run_log_config_catalog.json", "run_log"),
    ("run_management_audit", "res/run_management_audit/run_management_audit.json", "run_log"),
    ("checkpoint_resume_smoke", "res/run_management_audit/checkpoint_resume_smoke/checkpoint_resume_smoke.json", "run_log"),
    (
        "full_run_deliverable_gap_audit",
        "res/run_management_audit/full_run_deliverable_gap_audit/full_run_deliverable_gap_audit.json",
        "run_log",
    ),
    ("failed_run_audit", "res/failed_runs/failed_run_audit/failed_run_audit.json", "run_log"),
    (
        "official_train_entry_failed_run_audit",
        "res/failed_runs/official_train_entry_failed_run_audit/official_train_entry_failed_run_audit.json",
        "run_log",
    ),
    ("kit_inotify_budget_audit", "res/setup/kit_inotify_budget_audit/kit_inotify_budget_audit.json", "environment"),
    ("inotify_live_usage_audit", "res/setup/inotify_live_usage_audit/inotify_live_usage_audit.json", "environment"),
    (
        "vscode_watcher_exclude_audit",
        "res/setup/vscode_watcher_exclude_audit/vscode_watcher_exclude_audit.json",
        "environment",
    ),
    (
        "kit_watcher_config_surface_audit",
        "res/setup/kit_watcher_config_surface_audit/kit_watcher_config_surface_audit.json",
        "environment",
    ),
    ("experiment_protocol_audit", "res/docs/experiment_protocol_audit/experiment_protocol_audit.json", "documentation"),
    ("readme_audit", "res/docs/readme_audit/readme_audit.json", "documentation"),
    ("progress_report_audit", "res/progress_report_audit/progress_report_audit.json", "documentation"),
    (
        "completion_matrix_status_audit",
        "res/docs/completion_matrix_status_audit/completion_matrix_status_audit.json",
        "documentation",
    ),
    ("project_boundary_audit", "res/project_boundary_audit/project_boundary_audit.json", "documentation"),
    ("goal_directive_index", "res/goal_directive_index/goal_directive_index_audit.json", "documentation"),
    ("paper_source_coverage", "res/paper_source_coverage/paper_source_coverage_audit.json", "paper_audit"),
    ("paper_latex_inventory", "res/paper_latex_inventory/paper_latex_inventory_audit.json", "paper_audit"),
    ("paper_formula_code_trace", "res/paper_formula_code_trace/paper_formula_code_trace_audit.json", "paper_audit"),
    ("paper_pdf_source_consistency", "res/paper_pdf_source_consistency/paper_pdf_source_consistency_audit.json", "paper_audit"),
    ("paper_table_values", "res/paper_table_values/paper_table_value_audit.json", "paper_audit"),
    ("evaluation_metrics_coverage", "res/evaluation_metrics_coverage/evaluation_metrics_coverage_audit.json", "evaluation"),
    (
        "trial_failure_accounting_audit",
        "res/evaluation/trial_failure_accounting_audit/trial_failure_accounting_audit.json",
        "evaluation",
    ),
    ("ablation_coverage", "res/ablation_coverage/ablation_coverage_audit.json", "evaluation"),
    (
        "released_data_metrics_summary",
        "res/tables/released_data_metrics_summary/released_data_metrics_summary.json",
        "evaluation",
    ),
    (
        "released_data_statistical_audit",
        "res/tables/released_data_statistical_audit/released_data_statistical_audit.json",
        "evaluation",
    ),
    (
        "level_a_released_data_suite",
        "res/level_a/released_data_suite/level_a_released_data_suite.json",
        "evaluation",
    ),
    ("metrics_catalog", "res/metrics/metrics_catalog/metrics_catalog.json", "evaluation"),
    ("guidance_task_coverage", "res/guidance_task_coverage/guidance_task_coverage_audit.json", "evaluation"),
    ("tracking_config_audit", "res/tracking/smoke_config_audit/tracking_config_audit.json", "tracking"),
    (
        "tracking_official_source_contract",
        "res/tracking/official_source_contract_audit/tracking_official_source_contract_audit.json",
        "tracking",
    ),
    (
        "tracking_g1_action_scale_audit",
        "res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json",
        "tracking",
    ),
    (
        "tracking_reward_formula_audit",
        "res/tracking/reward_formula_audit/tracking_reward_formula_audit.json",
        "tracking",
    ),
    (
        "tracking_observation_action_schema_audit",
        "res/tracking/observation_action_schema_audit/tracking_observation_action_schema_audit.json",
        "tracking",
    ),
    (
        "tracking_randomization_termination_audit",
        "res/tracking/randomization_termination_audit/tracking_randomization_termination_audit.json",
        "tracking",
    ),
    (
        "level_b_tracking_nonkit_suite",
        "res/tracking/nonkit_suite/level_b_tracking_nonkit_suite.json",
        "tracking",
    ),
    ("tracking_smoke_rerun_audit", "res/tracking/tracking_smoke_rerun_audit/tracking_smoke_rerun_audit.json", "tracking"),
    (
        "tracking_official_train_entry_retry_audit",
        "res/tracking/official_train_entry_retry_audit/tracking_official_train_entry_retry_audit.json",
        "tracking",
    ),
    ("tracking_import_gate_audit", "res/tracking/tracking_import_gate_audit/tracking_import_gate_audit.json", "tracking"),
    (
        "tracking_extension_namespace_probe",
        "res/tracking/tracking_extension_namespace_probe/tracking_extension_namespace_probe.json",
        "tracking",
    ),
    ("motion_preprocessing_contract", "res/tracking/motion_preprocessing_contract_audit/motion_preprocessing_contract_audit.json", "tracking"),
    ("tracking_local_smoke_runner", "reproduction/scripts/run_tracking_local_smoke.sh", "tracking"),
    ("tracking_local_smoke_patcher", "reproduction/scripts/prepare_tracking_local_smoke.py", "tracking"),
    (
        "tracking_local_smoke_generated_manifest",
        "reproduction/generated/whole_body_tracking_local/manifest.tsv",
        "tracking",
    ),
    (
        "tracking_local_csv_to_npz_script",
        "reproduction/generated/whole_body_tracking_local/csv_to_npz_local.py",
        "tracking",
    ),
    (
        "tracking_local_replay_npz_script",
        "reproduction/generated/whole_body_tracking_local/replay_npz_local.py",
        "tracking",
    ),
    (
        "tracking_local_train_script",
        "reproduction/generated/whole_body_tracking_local/rsl_rl/train_local.py",
        "tracking",
    ),
    ("tracking_local_smoke_preflight", "res/tracking/local_smoke_preflight/tracking_local_smoke_preflight.json", "tracking"),
    ("tracking_motion_npz_fixture", "res/tracking/motion_npz_fixture/tracking_motion_npz_fixture.json", "tracking"),
    ("mujoco_ros_launch_contract", "res/tracking/mujoco_ros_launch_contract_audit/mujoco_ros_launch_contract_audit.json", "tracking"),
    (
        "tracking_deployment_controller_semantics",
        "res/tracking/deployment_controller_semantics_audit/tracking_deployment_controller_semantics_audit.json",
        "tracking",
    ),
    (
        "tracking_motion_policy_onnx_contract_fixture",
        "res/tracking/motion_policy_onnx_contract_fixture/tracking_motion_policy_onnx_contract_fixture.json",
        "tracking",
    ),
    (
        "tracking_debug_motion_policy_onnx_export",
        "res/tracking/debug_motion_policy_onnx_export/tracking_debug_motion_policy_onnx_export.json",
        "tracking",
    ),
    (
        "tracking_debug_motion_policy_onnx_file",
        "res/tracking/debug_motion_policy_onnx_export/debug_motion_policy_contract.onnx",
        "tracking",
    ),
    (
        "tracking_debug_motion_policy_onnx_inference",
        "res/tracking/debug_motion_policy_onnx_inference/tracking_debug_motion_policy_onnx_inference_audit.json",
        "tracking",
    ),
    ("level_c_debug_suite", "res/level_c/debug_suite/level_c_debug_suite.json", "level_c"),
    ("level_c_dataset_protocol", "res/level_c/dataset_collection_protocol_audit/level_c_dataset_collection_protocol_audit.json", "level_c"),
    ("level_c_state_latent_schema", "res/level_c/state_latent_schema_audit/state_latent_schema_audit.json", "level_c"),
    ("level_c_dagger_schema", "res/level_c/dagger_schema_audit/dagger_schema_audit.json", "level_c"),
    ("level_c_dagger_iteration_smoke", "res/level_c/dagger_iteration_smoke/level_c_dagger_iteration_smoke.json", "level_c"),
    (
        "level_c_state_latent_dataset_consistency",
        "res/level_c/state_latent_dataset_consistency_audit/level_c_state_latent_dataset_consistency_audit.json",
        "level_c",
    ),
    (
        "level_c_state_latent_training_dataset_contract_audit",
        "res/level_c/state_latent_training_dataset_contract_audit/"
        "level_c_state_latent_training_dataset_contract_audit.json",
        "level_c",
    ),
    (
        "level_c_extended_debug_suite",
        "res/level_c/extended_debug_suite/level_c_extended_debug_suite.json",
        "level_c",
    ),
    (
        "level_c_resource_adjusted_tiny_diffusion_suite",
        "res/level_c/resource_adjusted_tiny_diffusion_suite/"
        "level_c_resource_adjusted_tiny_diffusion_suite.json",
        "level_c",
    ),
    (
        "level_c_dagger_vae_pipeline_audit",
        "res/level_c/dagger_vae_pipeline_audit/level_c_dagger_vae_pipeline_audit.json",
        "level_c",
    ),
    ("level_c_vae_checkpoint_smoke", "res/level_c/vae_checkpoint_smoke/level_c_vae_checkpoint_smoke.json", "level_c"),
    (
        "level_c_vae_debug_overfit_latent_artifact",
        "res/level_c/vae_debug_overfit_latent_artifact/level_c_vae_debug_overfit_latent_artifact.json",
        "level_c",
    ),
    (
        "level_c_vae_motion_split_heldout_eval",
        "res/level_c/vae_motion_split_heldout_eval/level_c_vae_motion_split_heldout_eval.json",
        "level_c",
    ),
    (
        "level_c_vae_receding_horizon_rollout_smoke",
        "res/level_c/vae_receding_horizon_rollout_smoke/level_c_vae_receding_horizon_rollout_smoke.json",
        "level_c",
    ),
    (
        "level_c_diffusion_to_vae_action_smoke",
        "res/level_c/diffusion_to_vae_action_smoke/level_c_diffusion_to_vae_action_smoke.json",
        "level_c",
    ),
    (
        "level_c_diffusion_to_vae_action_multiseed_audit",
        "res/level_c/diffusion_to_vae_action_multiseed_audit/level_c_diffusion_to_vae_action_multiseed_audit.json",
        "level_c",
    ),
    (
        "level_c_diffusion_to_vae_action_smoothness_audit",
        "res/level_c/diffusion_to_vae_action_smoothness_audit/level_c_diffusion_to_vae_action_smoothness_audit.json",
        "level_c",
    ),
    (
        "level_c_direct_vs_latent_action_ablation_audit",
        "res/level_c/direct_vs_latent_action_ablation_audit/level_c_direct_vs_latent_action_ablation_audit.json",
        "level_c",
    ),
    (
        "level_c_vae_latent_diffusion_overfit",
        "res/level_c/vae_latent_diffusion_overfit_probe/level_c_vae_latent_diffusion_overfit_probe.json",
        "level_c",
    ),
    (
        "level_c_vae_latent_heldout_eval",
        "res/level_c/vae_latent_heldout_eval/level_c_vae_latent_heldout_eval.json",
        "level_c",
    ),
    (
        "level_c_vae_latent_heldout_multiseed_audit",
        "res/level_c/vae_latent_heldout_multiseed_audit/level_c_vae_latent_heldout_multiseed_audit.json",
        "level_c",
    ),
    ("level_c_transformer_state_dict_manifest", "res/level_c/transformer_state_dict_manifest/level_c_transformer_state_dict_manifest.json", "level_c"),
    (
        "level_c_vae_latent_transformer_arch_probe",
        "res/level_c/vae_latent_transformer_arch_probe/level_c_vae_latent_transformer_arch_probe.json",
        "level_c",
    ),
    (
        "level_c_vae_latent_transformer_ema_smoke",
        "res/level_c/vae_latent_transformer_ema_smoke/level_c_vae_latent_transformer_ema_smoke.json",
        "level_c",
    ),
    ("level_c_transformer_ema_smoke", "res/level_c/transformer_ema_smoke/level_c_transformer_ema_smoke.json", "level_c"),
    (
        "level_c_diffusion_checkpoint_smoke",
        "res/level_c/diffusion_checkpoint_smoke/level_c_diffusion_checkpoint_smoke.json",
        "level_c",
    ),
    (
        "level_c_bounded_debug_diffusion_training_run",
        "res/level_c/bounded_debug_diffusion_training_run/level_c_bounded_debug_diffusion_training_run.json",
        "level_c",
    ),
    (
        "level_c_bounded_debug_diffusion_checkpoint_eval",
        "res/level_c/bounded_debug_diffusion_checkpoint_eval/level_c_bounded_debug_diffusion_checkpoint_eval.json",
        "level_c",
    ),
    (
        "level_c_bounded_debug_diffusion_action_eval",
        "res/level_c/bounded_debug_diffusion_action_eval/level_c_bounded_debug_diffusion_action_eval.json",
        "level_c",
    ),
    (
        "level_c_resource_adjusted_tiny_diffusion_training_run",
        "res/level_c/resource_adjusted_tiny_diffusion_training_run/"
        "level_c_resource_adjusted_tiny_diffusion_training_run.json",
        "level_c",
    ),
    (
        "level_c_resource_adjusted_tiny_diffusion_multiseed_audit",
        "res/level_c/resource_adjusted_tiny_diffusion_multiseed_audit/"
        "level_c_resource_adjusted_tiny_diffusion_multiseed_audit.json",
        "level_c",
    ),
    (
        "level_c_resource_adjusted_tiny_diffusion_checkpoint_eval",
        "res/level_c/resource_adjusted_tiny_diffusion_checkpoint_eval/"
        "level_c_resource_adjusted_tiny_diffusion_checkpoint_eval.json",
        "level_c",
    ),
    (
        "level_c_resource_adjusted_tiny_diffusion_onnx_export_inference",
        "res/level_c/resource_adjusted_tiny_diffusion_onnx_export_inference/"
        "level_c_resource_adjusted_tiny_diffusion_onnx_export_inference.json",
        "level_c",
    ),
    (
        "level_c_resource_adjusted_tiny_diffusion_onnx_file",
        "res/level_c/resource_adjusted_tiny_diffusion_onnx_export_inference/"
        "resource_adjusted_tiny_denoiser_debug.onnx",
        "level_c",
    ),
    (
        "level_c_resource_adjusted_tiny_diffusion_latency_audit",
        "res/level_c/resource_adjusted_tiny_diffusion_latency_audit/"
        "level_c_resource_adjusted_tiny_diffusion_latency_audit.json",
        "level_c",
    ),
    (
        "level_c_resource_adjusted_tiny_diffusion_video_preview",
        "res/level_c/resource_adjusted_tiny_diffusion_video_preview/"
        "level_c_resource_adjusted_tiny_diffusion_video_preview.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_vae_diffusion_training",
        "res/level_c/lafan1_paper_arch_vae_diffusion_training/"
        "lafan1_paper_arch_vae_diffusion_training.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_vae_diffusion_dataset",
        "res/level_c/lafan1_paper_arch_vae_diffusion_training/lafan1_paper_arch_training_dataset.npz",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_vae_diffusion_checkpoint",
        "res/runs/level_c_lafan1_paper_arch_vae_diffusion_static_000_20260617_203000/"
        "checkpoint/lafan1_paper_arch_vae_diffusion.pt",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_vae_diffusion_run_metrics",
        "res/runs/level_c_lafan1_paper_arch_vae_diffusion_static_000_20260617_203000/metrics.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_vae_diffusion_script",
        "reproduction/scripts/train_lafan1_paper_level_vae_diffusion.py",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_vae_diffusion_seed_20260618",
        "res/level_c/lafan1_paper_arch_vae_diffusion_training_seed_20260618/"
        "lafan1_paper_arch_vae_diffusion_training.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_vae_diffusion_seed_20260618_checkpoint",
        "res/runs/level_c_lafan1_paper_arch_vae_diffusion_seed_20260618_static_000_20260617_203000/"
        "checkpoint/lafan1_paper_arch_vae_diffusion.pt",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_vae_diffusion_seed_20260619",
        "res/level_c/lafan1_paper_arch_vae_diffusion_training_seed_20260619/"
        "lafan1_paper_arch_vae_diffusion_training.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_vae_diffusion_seed_20260619_checkpoint",
        "res/runs/level_c_lafan1_paper_arch_vae_diffusion_seed_20260619_static_000_20260617_203000/"
        "checkpoint/lafan1_paper_arch_vae_diffusion.pt",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_multiseed_audit",
        "res/level_c/lafan1_paper_arch_multiseed_audit/level_c_lafan1_paper_arch_multiseed_audit.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_multiseed_rows",
        "res/level_c/lafan1_paper_arch_multiseed_audit/level_c_lafan1_paper_arch_multiseed_rows.tsv",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_multiseed_npz",
        "res/level_c/lafan1_paper_arch_multiseed_audit/level_c_lafan1_paper_arch_multiseed_metrics.npz",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_multiseed_script",
        "reproduction/scripts/level_c_lafan1_paper_arch_multiseed_audit.py",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_multiseed_audit",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_multiseed_audit/"
        "level_c_lafan1_paper_arch_symmetry_augmented_multiseed_audit.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_multiseed_rows",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_multiseed_audit/"
        "level_c_lafan1_paper_arch_symmetry_augmented_multiseed_rows.tsv",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_multiseed_npz",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_multiseed_audit/"
        "level_c_lafan1_paper_arch_symmetry_augmented_multiseed_metrics.npz",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_multiseed_script",
        "reproduction/scripts/level_c_lafan1_paper_arch_symmetry_multiseed_audit.py",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_high_memory_batch_audit",
        "res/level_c/lafan1_paper_arch_high_memory_batch_audit/"
        "level_c_lafan1_paper_arch_high_memory_batch_audit.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_high_memory_batch_rows",
        "res/level_c/lafan1_paper_arch_high_memory_batch_audit/"
        "level_c_lafan1_paper_arch_high_memory_batch_rows.tsv",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_high_memory_batch_fixture",
        "res/level_c/lafan1_paper_arch_high_memory_batch_audit/"
        "level_c_lafan1_paper_arch_high_memory_batch_fixture.npz",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_high_memory_batch_script",
        "reproduction/scripts/level_c_lafan1_paper_arch_high_memory_batch_audit.py",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_dataset_audit",
        "res/level_c/lafan1_paper_arch_symmetry_dataset/"
        "lafan1_paper_arch_symmetry_dataset_audit.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_dataset",
        "res/level_c/lafan1_paper_arch_symmetry_dataset/"
        "lafan1_paper_arch_symmetry_augmented_dataset.npz",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_dataset_splits",
        "res/level_c/lafan1_paper_arch_symmetry_dataset/"
        "level_c_lafan1_paper_arch_symmetry_dataset_splits.tsv",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_dataset_script",
        "reproduction/scripts/level_c_lafan1_paper_arch_symmetry_dataset_audit.py",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_training",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_training/"
        "lafan1_paper_arch_vae_diffusion_training.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_training_dataset",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_training/"
        "lafan1_paper_arch_training_dataset.npz",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_training_checkpoint",
        "res/runs/level_c_lafan1_paper_arch_symmetry_augmented_static_000_20260617_215500/"
        "checkpoint/lafan1_paper_arch_vae_diffusion.pt",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_training_metrics",
        "res/runs/level_c_lafan1_paper_arch_symmetry_augmented_static_000_20260617_215500/"
        "metrics.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_training_seed_20260622",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_training_seed_20260622/"
        "lafan1_paper_arch_vae_diffusion_training.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_training_seed_20260622_checkpoint",
        "res/runs/level_c_lafan1_paper_arch_symmetry_augmented_seed_20260622_static_000_20260617_215500/"
        "checkpoint/lafan1_paper_arch_vae_diffusion.pt",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_training_seed_20260622_metrics",
        "res/runs/level_c_lafan1_paper_arch_symmetry_augmented_seed_20260622_static_000_20260617_215500/"
        "metrics.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_training_seed_20260623",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_training_seed_20260623/"
        "lafan1_paper_arch_vae_diffusion_training.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_training_seed_20260623_checkpoint",
        "res/runs/level_c_lafan1_paper_arch_symmetry_augmented_seed_20260623_static_000_20260617_215500/"
        "checkpoint/lafan1_paper_arch_vae_diffusion.pt",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_training_seed_20260623_metrics",
        "res/runs/level_c_lafan1_paper_arch_symmetry_augmented_seed_20260623_static_000_20260617_215500/"
        "metrics.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_training_comparison",
        "res/level_c/lafan1_paper_arch_symmetry_training_comparison/"
        "level_c_lafan1_paper_arch_symmetry_training_comparison_audit.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_training_comparison_rows",
        "res/level_c/lafan1_paper_arch_symmetry_training_comparison/"
        "level_c_lafan1_paper_arch_symmetry_training_comparison.tsv",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_training_comparison_npz",
        "res/level_c/lafan1_paper_arch_symmetry_training_comparison/"
        "level_c_lafan1_paper_arch_symmetry_training_comparison_metrics.npz",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_training_comparison_script",
        "reproduction/scripts/level_c_lafan1_symmetry_training_comparison_audit.py",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_offline_metrics_audit",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_offline_metrics/"
        "level_c_lafan1_paper_arch_offline_metrics_audit.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_offline_metrics_rows",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_offline_metrics/"
        "level_c_lafan1_paper_arch_offline_metrics_rows.tsv",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_offline_metrics_npz",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_offline_metrics/"
        "level_c_lafan1_paper_arch_offline_metrics.npz",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_guidance_eval",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval/"
        "level_c_lafan1_paper_arch_guidance_eval.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_guidance_eval_rows",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval/"
        "level_c_lafan1_paper_arch_guidance_eval.tsv",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_guidance_eval_npz",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval/"
        "level_c_lafan1_paper_arch_guidance_eval.npz",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/"
        "level_c_lafan1_paper_arch_guidance_eval.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split_rows",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/"
        "level_c_lafan1_paper_arch_guidance_eval.tsv",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split_npz",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/"
        "level_c_lafan1_paper_arch_guidance_eval.npz",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/"
        "level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split_rows",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/"
        "level_c_lafan1_paper_arch_reverse_guidance_full_split_rows.tsv",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split_gpu_rows",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/"
        "level_c_lafan1_paper_arch_reverse_guidance_full_split_gpu_rows.tsv",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split_npz",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/"
        "level_c_lafan1_paper_arch_reverse_guidance_full_split.npz",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split_script",
        "reproduction/scripts/level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.py",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_onnx_latency_audit",
        "res/level_c/lafan1_paper_arch_onnx_latency/"
        "level_c_lafan1_paper_arch_onnx_latency_audit.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_vae_decoder_onnx",
        "res/level_c/lafan1_paper_arch_onnx_latency/lafan1_paper_arch_vae_decoder.onnx",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_diffusion_denoiser_onnx",
        "res/level_c/lafan1_paper_arch_onnx_latency/lafan1_paper_arch_diffusion_denoiser.onnx",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_onnx_latency_script",
        "reproduction/scripts/level_c_lafan1_paper_arch_onnx_latency_audit.py",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_onnx_latency_audit",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/"
        "level_c_lafan1_paper_arch_onnx_latency_audit.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_vae_decoder_onnx",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/lafan1_paper_arch_vae_decoder.onnx",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_diffusion_denoiser_onnx",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/"
        "lafan1_paper_arch_diffusion_denoiser.onnx",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_onnx_latency_rows",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/"
        "level_c_lafan1_paper_arch_onnx_latency_rows.tsv",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_onnx_io_fixture",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/"
        "lafan1_paper_arch_onnx_io_fixture.npz",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_offline_metrics_audit",
        "res/level_c/lafan1_paper_arch_offline_metrics/"
        "level_c_lafan1_paper_arch_offline_metrics_audit.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_offline_metrics_npz",
        "res/level_c/lafan1_paper_arch_offline_metrics/level_c_lafan1_paper_arch_offline_metrics.npz",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_offline_metrics_script",
        "reproduction/scripts/level_c_lafan1_paper_arch_offline_metrics_audit.py",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_guidance_eval",
        "res/level_c/lafan1_paper_arch_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_guidance_eval_npz",
        "res/level_c/lafan1_paper_arch_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.npz",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_guidance_eval_script",
        "reproduction/scripts/level_c_lafan1_paper_arch_guidance_eval.py",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_reverse_guidance",
        "res/level_c/lafan1_paper_arch_reverse_guidance/"
        "level_c_lafan1_paper_arch_reverse_guidance_audit.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_reverse_guidance_rows",
        "res/level_c/lafan1_paper_arch_reverse_guidance/"
        "level_c_lafan1_paper_arch_reverse_guidance_rows.tsv",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_reverse_guidance_npz",
        "res/level_c/lafan1_paper_arch_reverse_guidance/"
        "level_c_lafan1_paper_arch_reverse_guidance.npz",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_reverse_guidance_script",
        "reproduction/scripts/level_c_lafan1_paper_arch_reverse_guidance_audit.py",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_reverse_guidance",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance/"
        "level_c_lafan1_paper_arch_reverse_guidance_audit.json",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_reverse_guidance_rows",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance/"
        "level_c_lafan1_paper_arch_reverse_guidance_rows.tsv",
        "level_c",
    ),
    (
        "level_c_lafan1_paper_arch_symmetry_augmented_reverse_guidance_npz",
        "res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance/"
        "level_c_lafan1_paper_arch_reverse_guidance.npz",
        "level_c",
    ),
    ("level_c_diffusion_equations", "res/level_c/diffusion_equation_audit/level_c_diffusion_equation_audit.json", "level_c"),
    ("level_c_guidance_task_scale_sweep", "res/level_c/guidance_task_scale_sweep/level_c_guidance_task_scale_sweep.json", "level_c"),
    ("level_c_guidance_debug_visualization", "res/level_c/guidance_debug_visualization/level_c_guidance_debug_visualization.json", "level_c"),
    ("level_c_guidance_task_metric_audit", "res/level_c/guidance_task_metric_audit/level_c_guidance_task_metric_audit.json", "level_c"),
    (
        "level_c_guidance_full_split_result_table",
        "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.json",
        "level_c",
    ),
    (
        "level_c_guidance_full_split_result_table_tsv",
        "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.tsv",
        "level_c",
    ),
    (
        "level_c_guidance_full_split_result_table_csv",
        "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.csv",
        "level_c",
    ),
    (
        "level_c_guidance_full_split_result_table_pdf",
        "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_cost_delta.pdf",
        "level_c",
    ),
    (
        "level_c_guidance_full_split_result_table_svg",
        "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_cost_delta.svg",
        "level_c",
    ),
    (
        "level_c_guidance_full_split_result_table_png",
        "res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_cost_delta.png",
        "level_c",
    ),
    (
        "level_c_guidance_full_split_result_table_script",
        "reproduction/scripts/level_c_guidance_full_split_result_table.py",
        "level_c",
    ),
    (
        "level_c_guidance_checkpoint_visualization",
        "res/level_c/guidance_checkpoint_visualization/level_c_guidance_checkpoint_visualization.json",
        "level_c",
    ),
    (
        "level_c_guidance_checkpoint_visualization_rows",
        "res/level_c/guidance_checkpoint_visualization/level_c_guidance_checkpoint_visualization.tsv",
        "level_c",
    ),
    (
        "level_c_guidance_checkpoint_visualization_gif",
        "res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_composed_preview.gif",
        "level_c",
    ),
    (
        "level_c_guidance_checkpoint_visualization_joystick_png",
        "res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_joystick.png",
        "level_c",
    ),
    (
        "level_c_guidance_checkpoint_visualization_waypoint_png",
        "res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_waypoint.png",
        "level_c",
    ),
    (
        "level_c_guidance_checkpoint_visualization_obstacle_png",
        "res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_obstacle_avoidance.png",
        "level_c",
    ),
    (
        "level_c_guidance_checkpoint_visualization_inpainting_png",
        "res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_inpainting.png",
        "level_c",
    ),
    (
        "level_c_guidance_checkpoint_visualization_composed_png",
        "res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_composed_objectives.png",
        "level_c",
    ),
    (
        "level_c_guidance_checkpoint_visualization_script",
        "reproduction/scripts/level_c_guidance_checkpoint_visualization.py",
        "level_c",
    ),
    (
        "level_c_guidance_visual_deliverables_audit",
        "res/level_c/guidance_visual_deliverables_audit/level_c_guidance_visual_deliverables_audit.json",
        "level_c",
    ),
    ("level_c_guidance_coverage", "res/level_c/guidance_cost_coverage_audit/level_c_guidance_cost_coverage_audit.json", "level_c"),
    ("comparison_csv", "res/comparison/paper_vs_reproduction.csv", "comparison"),
    ("final_report_json", "res/final_report/final_reproduction_report.json", "final_report"),
    ("final_report_md", "reproduction/docs/final_reproduction_report.md", "final_report"),
    ("goal_final_report_md", "res/final_report/reproduction_report.md", "final_report"),
    (
        "final_report_requirement_audit",
        "res/final_report/final_report_requirement_audit/final_report_requirement_audit.json",
        "final_report",
    ),
    ("final_deliverables_audit", "res/final_deliverables_audit/final_deliverables_audit.json", "final_report"),
    ("visual_media_inventory_audit", "res/visual_media_inventory/visual_media_inventory_audit.json", "final_report"),
    (
        "verification_command_coverage_audit",
        "res/verification_command_coverage/verification_command_coverage_audit.json",
        "final_report",
    ),
    (
        "verification_command_syntax_audit",
        "res/verification_command_syntax/verification_command_syntax_audit.json",
        "final_report",
    ),
    (
        "verification_command_script_manifest",
        "res/verification_command_script_manifest/verification_command_script_manifest.json",
        "final_report",
    ),
    (
        "required_artifact_absence_audit",
        "res/required_artifact_absence/required_artifact_absence_audit.json",
        "final_report",
    ),
    ("runbook", "reproduction/RUNBOOK.md", "documentation"),
    ("progress", "reproduction/PROGRESS.md", "documentation"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for name, rel, category in ARTIFACTS:
        path = ROOT / rel
        exists = path.is_file()
        rows.append(
            {
                "name": name,
                "category": category,
                "relative_path": rel,
                "absolute_path": str(path),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else 0,
                "sha256": sha256(path) if exists else "",
            }
        )
    missing = [row for row in rows if not row["exists"]]
    categories: dict[str, int] = {}
    for row in rows:
        categories[row["category"]] = categories.get(row["category"], 0) + 1
    summary = {
        "status": "ok" if not missing else "failed",
        "experiment_type": "artifact_manifest",
        "scope": "SHA256 manifest for key BeyondMimic reproduction deliverables",
        "artifact_count": len(rows),
        "missing_count": len(missing),
        "category_counts": dict(sorted(categories.items())),
        "rows": rows,
        "checks": {
            "all_manifest_artifacts_exist": not missing,
            "raw_sources_hashed": all(row["exists"] and row["sha256"] for row in rows if row["category"] == "raw_source"),
            "env_locks_hashed": all(row["exists"] and row["sha256"] for row in rows if row["category"] == "environment"),
            "final_report_hashed": all(row["exists"] and row["sha256"] for row in rows if row["category"] == "final_report"),
            "tracking_local_smoke_scripts_hashed": all(
                row["exists"] and row["sha256"]
                for row in rows
                if row["name"].startswith("tracking_local_")
            ),
            "atomic_write_used": True,
            "does_not_modify_raw_downloads": True,
            "does_not_claim_goal_complete": True,
        },
        "interpretation": {
            "goal_complete": False,
            "why_not_complete": (
                "The manifest improves artifact traceability, but it does not create missing trained checkpoints, "
                "videos, TensorRT engines, Fig. 5/6 paper results, or hardware deployment evidence."
            ),
        },
        "outputs": {
            "json": str(OUT / "artifact_manifest.json"),
            "tsv": str(OUT / "artifact_manifest.tsv"),
        },
    }
    atomic_write_text(OUT / "artifact_manifest.json", json.dumps(summary, indent=2, sort_keys=True))
    tsv_tmp = OUT / "artifact_manifest.tsv.tmp"
    with tsv_tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            delimiter="\t",
            fieldnames=["name", "category", "relative_path", "absolute_path", "exists", "size_bytes", "sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)
    tsv_tmp.replace(OUT / "artifact_manifest.tsv")
    print(json.dumps({"status": summary["status"], "artifacts": len(rows), "json": summary["outputs"]["json"]}))
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
