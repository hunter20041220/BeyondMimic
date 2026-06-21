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
ENGLISH_READING_REPORT_DOC = ROOT / "reproduction/docs/english_reading_report.md"
ENGLISH_READING_REPORT_FINAL = ROOT / "res/final_report/english_reading_report.md"
CURRENT_STATUS_REPORT_DOC = ROOT / "reproduction/docs/current_environment_and_reproduction_status.md"
CURRENT_STATUS_REPORT_FINAL = ROOT / "res/final_report/current_environment_and_reproduction_status.md"


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
    isaaclab_current_headless_gate = load_json(
        "res/setup/isaaclab_current_headless_gate/isaaclab_current_headless_gate.json"
    )
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
    tracking_csv_task_eval_gpu47_failed_rerun = load_json(
        "res/failed_runs/tracking_g1_resource_adjusted_csv_task_eval_gpu47_20260619_124125/status.json"
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
    visual_evidence_index = load_json("res/report_assets/visual_evidence_index/visual_evidence_index.json")
    guided_vs_unguided_closed_loop_matrix = load_json(
        "res/report_assets/guided_vs_unguided_closed_loop_matrix/"
        "guided_vs_unguided_closed_loop_matrix.json"
    )
    official_importer_export_fig5_fig6_proxy_protocol_matrix = load_json(
        "res/report_assets/official_importer_export_fig5_fig6_proxy_protocol_matrix/"
        "fig5_fig6_proxy_protocol_matrix.json"
    )
    official_importer_export_fig5_fig6_task_protocol_proxy = load_json(
        "res/report_assets/official_importer_export_fig5_fig6_task_protocol_proxy/"
        "fig5_fig6_task_protocol_proxy.json"
    )
    official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy = load_json(
        "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy/"
        "fig5_fig6_task_protocol_proxy.json"
    )
    official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy = load_json(
        "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy/"
        "success_fall_collision_proxy.json"
    )
    official_importer_export_full_bundle_latent_projection_report_assets = load_json(
        "res/report_assets/official_importer_export_full_bundle_latent_projection/"
        "official_importer_export_full_bundle_latent_projection_assets.json"
    )
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
    tracking_official_replay_npz_entry_diagnostic = load_json(
        "res/tracking/official_replay_npz_entry_diagnostic/"
        "tracking_official_replay_npz_entry_diagnostic_audit.json"
    )
    tracking_official_replay_npz_loop_with_enriched_usd = load_json(
        "res/tracking/official_replay_npz_loop_with_enriched_usd/"
        "tracking_official_replay_npz_loop_with_enriched_usd_audit.json"
    )
    tracking_official_replay_npz_loop_full_dataset_with_enriched_usd = load_json(
        "res/tracking/official_replay_npz_loop_full_dataset_with_enriched_usd/"
        "tracking_official_replay_npz_loop_full_dataset_with_enriched_usd_audit.json"
    )
    tracking_official_replay_npz_loop_full_dataset_with_official_importer_export = load_json(
        "res/tracking/official_replay_npz_loop_full_dataset_with_official_importer_export/"
        "tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_audit.json"
    )
    official_importer_export_replay_full_dataset_report_assets = load_json(
        "res/report_assets/official_importer_export_replay_full_dataset/"
        "official_importer_export_replay_full_dataset_report_assets.json"
    )
    tracking_official_csv_to_npz_loop_with_enriched_usd = load_json(
        "res/tracking/official_csv_to_npz_loop_with_enriched_usd/"
        "tracking_official_csv_to_npz_loop_with_enriched_usd_audit.json"
    )
    tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd = load_json(
        "res/tracking/official_csv_to_npz_loop_full_dataset_with_enriched_usd/"
        "tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_audit.json"
    )
    tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export = load_json(
        "res/tracking/official_csv_to_npz_loop_full_dataset_with_official_importer_export/"
        "tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_audit.json"
    )
    tracking_g1_official_csv_loop_full_dataset_task_eval = load_json(
        "res/tracking/g1_official_csv_loop_full_dataset_task_eval/"
        "tracking_g1_official_csv_loop_full_dataset_task_eval.json"
    )
    tracking_g1_official_importer_export_task_smoke = load_json(
        "res/tracking/g1_official_importer_export_task_smoke/"
        "tracking_g1_official_importer_export_task_smoke.json"
    )
    tracking_g1_official_importer_export_full_dataset_task_eval = load_json(
        "res/tracking/g1_official_importer_export_full_dataset_task_eval/"
        "tracking_g1_official_importer_export_full_dataset_task_eval.json"
    )
    tracking_g1_urdf_import_config_variant_probe = load_json(
        "res/tracking/g1_urdf_import_config_variant_probe/"
        "tracking_g1_urdf_import_config_variant_probe.json"
    )
    tracking_g1_enriched_usd_replay_preflight = load_json(
        "res/tracking/g1_enriched_usd_replay_preflight/tracking_g1_enriched_usd_replay_preflight_audit.json"
    )
    tracking_g1_enriched_usd_bounded_replay_metrics = load_json(
        "res/tracking/g1_enriched_usd_bounded_replay_metrics/"
        "tracking_g1_enriched_usd_bounded_replay_metrics_audit.json"
    )
    tracking_g1_resource_adjusted_task_smoke = load_json(
        "res/tracking/g1_resource_adjusted_task_smoke/tracking_g1_resource_adjusted_task_smoke_audit.json"
    )
    tracking_g1_resource_adjusted_multi_fixture_eval = load_json(
        "res/tracking/g1_resource_adjusted_multi_fixture_eval/"
        "tracking_g1_resource_adjusted_multi_fixture_eval_audit.json"
    )
    tracking_g1_resource_adjusted_csv_conversion = load_json(
        "res/tracking/g1_resource_adjusted_csv_conversion/"
        "tracking_g1_resource_adjusted_csv_conversion_audit.json"
    )
    tracking_g1_resource_adjusted_csv_full_replay = load_json(
        "res/tracking/g1_resource_adjusted_csv_full_replay/"
        "tracking_g1_resource_adjusted_csv_full_replay_audit.json"
    )
    tracking_g1_resource_adjusted_csv_task_eval = load_json(
        "res/tracking/g1_resource_adjusted_csv_task_eval/"
        "tracking_g1_resource_adjusted_csv_task_eval_audit.json"
    )
    tracking_g1_resource_adjusted_train_entry_diagnostic = load_json(
        "res/tracking/g1_resource_adjusted_train_entry_diagnostic/"
        "tracking_g1_resource_adjusted_train_entry_diagnostic_audit.json"
    )
    tracking_g1_resource_adjusted_ppo_training_run = load_json(
        "res/tracking/g1_resource_adjusted_ppo_training_run/"
        "tracking_g1_resource_adjusted_ppo_training_run.json"
    )
    tracking_g1_resource_adjusted_ppo_checkpoint_eval = load_json(
        "res/tracking/g1_resource_adjusted_ppo_checkpoint_eval/"
        "tracking_g1_resource_adjusted_ppo_checkpoint_eval.json"
    )
    tracking_g1_official_csv_loop_ppo_training_run = load_json(
        "res/tracking/g1_official_csv_loop_ppo_training_run/"
        "tracking_g1_official_csv_loop_ppo_training_run.json"
    )
    tracking_g1_official_csv_loop_ppo_checkpoint_eval = load_json(
        "res/tracking/g1_official_csv_loop_ppo_checkpoint_eval/"
        "tracking_g1_official_csv_loop_ppo_checkpoint_eval.json"
    )
    tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval = load_json(
        "res/tracking/g1_official_csv_loop_ppo_checkpoint_multiseed_eval/"
        "tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval.json"
    )
    tracking_g1_official_csv_loop_full_bundle_motion_npz = load_json(
        "res/tracking/official_csv_loop_full_bundle_motion_npz/"
        "tracking_g1_official_csv_loop_full_bundle_motion_npz.json"
    )
    tracking_g1_official_csv_loop_full_bundle_ppo_training_run = load_json(
        "res/tracking/g1_official_csv_loop_full_bundle_ppo_training_run/"
        "tracking_g1_official_csv_loop_full_bundle_ppo_training_run.json"
    )
    tracking_g1_official_csv_loop_full_bundle_ppo_checkpoint_eval = load_json(
        "res/tracking/g1_official_csv_loop_full_bundle_ppo_checkpoint_eval/"
        "tracking_g1_official_csv_loop_full_bundle_ppo_checkpoint_eval.json"
    )
    official_csv_loop_ppo_eval_report_assets = load_json(
        "res/report_assets/official_csv_loop_ppo_checkpoint_eval/official_csv_loop_ppo_checkpoint_eval_assets.json"
    )
    official_csv_loop_ppo_multiseed_eval_report_assets = load_json(
        "res/report_assets/official_csv_loop_ppo_checkpoint_multiseed_eval/"
        "official_csv_loop_ppo_checkpoint_multiseed_eval_assets.json"
    )
    official_csv_loop_full_bundle_ppo_eval_report_assets = load_json(
        "res/report_assets/official_csv_loop_full_bundle_ppo_checkpoint_eval/"
        "official_csv_loop_full_bundle_ppo_checkpoint_eval_assets.json"
    )
    tracking_g1_official_importer_export_full_bundle_ppo_training_run = load_json(
        "res/tracking/g1_official_importer_export_full_bundle_ppo_training_run/"
        "tracking_g1_official_importer_export_full_bundle_ppo_training_run.json"
    )
    tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval = load_json(
        "res/tracking/g1_official_importer_export_full_bundle_ppo_checkpoint_eval/"
        "tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval.json"
    )
    official_importer_export_full_bundle_ppo_eval_report_assets = load_json(
        "res/report_assets/official_importer_export_full_bundle_ppo_checkpoint_eval/"
        "official_importer_export_full_bundle_ppo_checkpoint_eval_assets.json"
    )
    tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run = load_json(
        "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_training_run/"
        "tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run.json"
    )
    tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval = load_json(
        "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/"
        "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json"
    )
    official_importer_export_full_bundle_scaled_ppo_eval_report_assets = load_json(
        "res/report_assets/official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/"
        "official_importer_export_full_bundle_scaled_ppo_checkpoint_eval_assets.json"
    )
    official_importer_export_scaled_ppo_checkpoint_completion_proxy = load_json(
        "res/report_assets/official_importer_export_scaled_ppo_checkpoint_completion_proxy/"
        "scaled_ppo_checkpoint_completion_proxy.json"
    )
    tracking_g1_official_importer_export_scaled_ppo_checkpoint_sweep = load_json(
        "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_sweep/"
        "tracking_g1_official_importer_export_scaled_ppo_checkpoint_sweep.json"
    )
    tracking_g1_official_importer_export_scaled_ppo_best_checkpoint_confirmation_eval = load_json(
        "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_best_checkpoint_confirmation_eval/"
        "tracking_g1_official_importer_export_scaled_ppo_best_checkpoint_confirmation_eval.json"
    )
    official_importer_export_scaled_ppo_reward_termination_diagnostic = load_json(
        "res/report_assets/official_importer_export_scaled_ppo_reward_termination_diagnostic/"
        "reward_termination_diagnostic.json"
    )
    official_importer_export_scaled_ppo_ee_body_pos_termination_source_audit = load_json(
        "res/report_assets/official_importer_export_scaled_ppo_ee_body_pos_termination_source_audit/"
        "ee_body_pos_termination_source_audit.json"
    )
    tracking_g1_official_importer_export_scaled_ppo_endpoint_z_error_trace = load_json(
        "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_endpoint_z_error_trace/"
        "tracking_g1_official_importer_export_scaled_ppo_endpoint_z_error_trace.json"
    )
    tracking_g1_official_importer_export_scaled_ppo_checkpoint_multiseed_eval = load_json(
        "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_multiseed_eval/"
        "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_multiseed_eval.json"
    )
    official_importer_export_scaled_ppo_checkpoint_multiseed_eval_report_assets = load_json(
        "res/report_assets/official_importer_export_scaled_ppo_checkpoint_multiseed_eval/"
        "official_importer_export_scaled_ppo_checkpoint_multiseed_eval_assets.json"
    )
    official_importer_export_full_bundle_scaled_ppo_policy_rollout_capture = load_json(
        "res/visualization/official_importer_export_full_bundle_scaled_ppo_policy_rollout/"
        "tracking_g1_official_importer_export_full_bundle_scaled_ppo_policy_rollout_capture.json"
    )
    official_importer_export_full_bundle_scaled_ppo_policy_rollout_video_asset = load_json(
        "res/visualization/official_importer_export_full_bundle_scaled_ppo_policy_rollout/"
        "official_importer_export_full_bundle_scaled_ppo_policy_rollout_video_asset.json"
    )
    official_importer_export_tracking_eval_summary_assets = load_json(
        "res/report_assets/official_importer_export_tracking_eval_summary/"
        "official_importer_export_tracking_eval_summary_assets.json"
    )
    tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset = load_json(
        "res/tracking/g1_official_importer_export_full_bundle_teacher_rollout_dataset/"
        "tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset.json"
    )
    official_importer_export_full_bundle_teacher_rollout_report_assets = load_json(
        "res/report_assets/official_importer_export_full_bundle_teacher_rollout_dataset/"
        "official_importer_export_full_bundle_teacher_rollout_report_assets.json"
    )
    tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset = load_json(
        "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset/"
        "tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset.json"
    )
    official_importer_export_full_bundle_scaled_ppo_teacher_rollout_report_assets = load_json(
        "res/report_assets/official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset/"
        "official_importer_export_full_bundle_teacher_rollout_report_assets.json"
    )
    official_csv_loop_reference_replay_video_asset = load_json(
        "res/visualization/official_csv_loop_reference_replay/"
        "official_csv_loop_reference_replay_video_asset.json"
    )
    official_importer_export_full_dataset_reference_replay_video_asset = load_json(
        "res/visualization/official_importer_export_full_dataset_reference_replay/"
        "official_importer_export_full_dataset_reference_replay_video_asset.json"
    )
    official_csv_loop_policy_rollout_capture = load_json(
        "res/visualization/official_csv_loop_policy_rollout/"
        "tracking_g1_official_csv_loop_policy_rollout_capture.json"
    )
    official_csv_loop_policy_rollout_video_asset = load_json(
        "res/visualization/official_csv_loop_policy_rollout/official_csv_loop_policy_rollout_video_asset.json"
    )
    official_csv_loop_full_bundle_policy_rollout_capture = load_json(
        "res/visualization/official_csv_loop_full_bundle_policy_rollout/"
        "tracking_g1_official_csv_loop_policy_rollout_capture.json"
    )
    official_csv_loop_full_bundle_policy_rollout_video_asset = load_json(
        "res/visualization/official_csv_loop_full_bundle_policy_rollout/"
        "official_csv_loop_policy_rollout_video_asset.json"
    )
    official_csv_loop_vae_closed_loop_rollout_capture = load_json(
        "res/visualization/official_csv_loop_vae_closed_loop_rollout/"
        "tracking_g1_official_csv_loop_vae_closed_loop_rollout_capture.json"
    )
    official_csv_loop_vae_closed_loop_rollout_video_asset = load_json(
        "res/visualization/official_csv_loop_vae_closed_loop_rollout/"
        "official_csv_loop_vae_closed_loop_rollout_video_asset.json"
    )
    official_importer_export_full_bundle_vae_closed_loop_rollout_capture = load_json(
        "res/visualization/official_importer_export_full_bundle_vae_closed_loop_rollout/"
        "tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_capture.json"
    )
    official_importer_export_full_bundle_vae_closed_loop_rollout_video_asset = load_json(
        "res/visualization/official_importer_export_full_bundle_vae_closed_loop_rollout/"
        "official_importer_export_full_bundle_vae_closed_loop_rollout_video_asset.json"
    )
    official_csv_loop_action_guidance_rollout_eval = load_json(
        "res/level_c/official_csv_loop_action_guidance_rollout_eval/"
        "level_c_official_csv_loop_action_guidance_rollout_eval.json"
    )
    official_csv_loop_action_guidance_rollout_asset = load_json(
        "res/visualization/official_csv_loop_action_guidance_rollout/"
        "official_csv_loop_action_guidance_rollout_asset.json"
    )
    official_csv_loop_receding_latent_guidance_rollout_eval = load_json(
        "res/level_c/official_csv_loop_receding_latent_guidance_rollout_eval/"
        "level_c_official_csv_loop_receding_latent_guidance_rollout_eval.json"
    )
    official_csv_loop_receding_latent_guidance_rollout_asset = load_json(
        "res/visualization/official_csv_loop_receding_latent_guidance_rollout/"
        "official_csv_loop_receding_latent_guidance_rollout_asset.json"
    )
    official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval = load_json(
        "res/level_c/official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval/"
        "level_c_official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval.json"
    )
    official_csv_loop_full_bundle_receding_latent_guidance_rollout_asset = load_json(
        "res/visualization/official_csv_loop_full_bundle_receding_latent_guidance_rollout/"
        "official_csv_loop_receding_latent_guidance_rollout_asset.json"
    )
    official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval = load_json(
        "res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval/"
        "level_c_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval.json"
    )
    official_csv_loop_full_bundle_task_conditioned_guidance_summary_assets = load_json(
        "res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_summary/"
        "official_csv_loop_task_conditioned_guidance_summary_assets.json"
    )
    official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval = load_json(
        "res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval/"
        "official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json"
    )
    official_csv_loop_full_bundle_task_conditioned_guidance_multiseed_assets = load_json(
        "res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_multiseed/"
        "official_csv_loop_full_bundle_task_conditioned_guidance_multiseed_assets.json"
    )
    official_csv_loop_full_bundle_task_conditioned_guidance_success_boundary = load_json(
        "res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_success_boundary/"
        "local_proxy_success_boundary.json"
    )
    official_csv_loop_full_bundle_guidance_video_contact_sheet = load_json(
        "res/report_assets/official_csv_loop_full_bundle_guidance_video_contact_sheet/"
        "full_bundle_guidance_video_index.json"
    )
    official_csv_loop_task_conditioned_latent_guidance_rollout_eval = load_json(
        "res/level_c/official_csv_loop_task_conditioned_latent_guidance_rollout_eval/"
        "level_c_official_csv_loop_task_conditioned_latent_guidance_rollout_eval.json"
    )
    official_csv_loop_task_conditioned_guidance_summary_assets = load_json(
        "res/report_assets/official_csv_loop_task_conditioned_guidance_summary/"
        "official_csv_loop_task_conditioned_guidance_summary_assets.json"
    )
    official_csv_loop_task_conditioned_latent_guidance_multiseed_eval = load_json(
        "res/level_c/official_csv_loop_task_conditioned_latent_guidance_multiseed_eval/"
        "official_csv_loop_task_conditioned_latent_guidance_multiseed_eval.json"
    )
    official_csv_loop_task_conditioned_guidance_multiseed_assets = load_json(
        "res/report_assets/official_csv_loop_task_conditioned_guidance_multiseed/"
        "official_csv_loop_task_conditioned_guidance_multiseed_assets.json"
    )
    official_csv_loop_teacher_rollout_report_assets = load_json(
        "res/report_assets/official_csv_loop_teacher_rollout_dataset/"
        "official_csv_loop_teacher_rollout_report_assets.json"
    )
    official_csv_loop_full_bundle_teacher_rollout_report_assets = load_json(
        "res/report_assets/official_csv_loop_full_bundle_teacher_rollout_dataset/"
        "official_csv_loop_full_bundle_teacher_rollout_report_assets.json"
    )
    tracking_g1_resource_adjusted_teacher_rollout_dataset = load_json(
        "res/tracking/g1_resource_adjusted_teacher_rollout_dataset/"
        "tracking_g1_resource_adjusted_teacher_rollout_dataset.json"
    )
    tracking_g1_official_csv_loop_teacher_rollout_dataset = load_json(
        "res/tracking/g1_official_csv_loop_teacher_rollout_dataset/"
        "tracking_g1_official_csv_loop_teacher_rollout_dataset.json"
    )
    tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset = load_json(
        "res/tracking/g1_official_csv_loop_full_bundle_teacher_rollout_dataset/"
        "tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset.json"
    )
    tracking_g1_urdf_in_memory_gpu4_probe = load_json(
        "res/tracking/g1_urdf_in_memory_gpu4_probe/tracking_g1_urdf_in_memory_gpu4_probe.json"
    )
    tracking_g1_urdf_in_memory_gpu4_export_structure = load_json(
        "res/tracking/g1_urdf_in_memory_gpu4_export_structure_audit/"
        "tracking_g1_urdf_in_memory_gpu4_export_structure_audit.json"
    )
    train_entry_runtime_warning = tracking_g1_resource_adjusted_train_entry_diagnostic.get(
        "interpretation", {}
    ).get("runtime_warning")
    if train_entry_runtime_warning is None:
        train_entry_markers = tracking_g1_resource_adjusted_train_entry_diagnostic.get("run", {}).get("markers", {})
        train_entry_runtime_warning = (
            "The probe log contains PhysX GPU kernel warnings before the success sentinel."
            if train_entry_markers.get("physx_gpu_kernel_error")
            else "Runtime warning field is absent in this audit JSON; see the raw probe log if needed."
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
    tracking_usd_api_variant_probe = load_json("res/tracking/usd_api_variant_probe/tracking_usd_api_variant_probe.json")
    tracking_g1_urdf_stage_export_probe = load_json(
        "res/tracking/g1_urdf_stage_export_workaround/tracking_g1_urdf_stage_export_workaround_probe.json"
    )
    tracking_g1_urdf_layer_save_probe = load_json(
        "res/tracking/g1_urdf_layer_save_workaround/tracking_g1_urdf_layer_save_workaround_probe.json"
    )
    tracking_g1_urdf_in_memory_probe = load_json(
        "res/tracking/g1_urdf_in_memory_import/tracking_g1_urdf_in_memory_import_probe.json"
    )
    tracking_g1_urdf_simulationapp_in_memory_probe = load_json(
        "res/tracking/g1_urdf_simulationapp_in_memory_import/"
        "tracking_g1_urdf_simulationapp_in_memory_import_probe.json"
    )
    tracking_g1_urdf_in_memory_variant_matrix_probe = load_json(
        "res/tracking/g1_urdf_in_memory_variant_matrix/tracking_g1_urdf_in_memory_variant_matrix_probe.json"
    )
    tracking_g1_preconverted_asset_audit = load_json(
        "res/tracking/g1_preconverted_asset_audit/tracking_g1_preconverted_asset_audit.json"
    )
    tracking_g1_reference_usd_compatibility_audit = load_json(
        "res/tracking/g1_reference_usd_compatibility_audit/"
        "tracking_g1_reference_usd_compatibility_audit.json"
    )
    tracking_g1_official_urdf_skeleton_usd_audit = load_json(
        "res/tracking/g1_official_urdf_skeleton_usd/tracking_g1_official_urdf_skeleton_usd_audit.json"
    )
    tracking_g1_urdf_physical_asset_contract_audit = load_json(
        "res/tracking/g1_urdf_physical_asset_contract_audit/tracking_g1_urdf_physical_asset_contract_audit.json"
    )
    tracking_g1_urdf_source_equivalence_audit = load_json(
        "res/tracking/g1_urdf_source_equivalence_audit/tracking_g1_urdf_source_equivalence_audit.json"
    )
    tracking_g1_resource_adjusted_enriched_usd_probe = load_json(
        "res/tracking/g1_resource_adjusted_enriched_usd/tracking_g1_resource_adjusted_enriched_usd_probe.json"
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
    resource_adjusted_teacher_rollout_vae_training = load_json(
        "res/level_c/resource_adjusted_teacher_rollout_vae_training/"
        "level_c_resource_adjusted_teacher_rollout_vae_training.json"
    )
    official_csv_loop_teacher_rollout_vae_training = load_json(
        "res/level_c/official_csv_loop_teacher_rollout_vae_training/"
        "level_c_official_csv_loop_teacher_rollout_vae_training.json"
    )
    official_csv_loop_teacher_rollout_state_latent_dataset = load_json(
        "res/level_c/official_csv_loop_teacher_rollout_state_latent_dataset/"
        "level_c_official_csv_loop_teacher_rollout_state_latent_dataset.json"
    )
    official_csv_loop_state_latent_diffusion_training = load_json(
        "res/level_c/official_csv_loop_state_latent_diffusion_training/"
        "level_c_official_csv_loop_state_latent_diffusion_training.json"
    )
    official_csv_loop_full_bundle_teacher_rollout_vae_training = load_json(
        "res/level_c/official_csv_loop_full_bundle_teacher_rollout_vae_training/"
        "level_c_official_csv_loop_full_bundle_teacher_rollout_vae_training.json"
    )
    official_importer_export_full_bundle_teacher_rollout_vae_training = load_json(
        "res/level_c/official_importer_export_full_bundle_teacher_rollout_vae_training/"
        "level_c_official_importer_export_full_bundle_teacher_rollout_vae_training.json"
    )
    official_importer_export_scaled_ppo_teacher_rollout_vae_training = load_json(
        "res/level_c/official_importer_export_scaled_ppo_teacher_rollout_vae_training/"
        "level_c_official_importer_export_scaled_ppo_teacher_rollout_vae_training.json"
    )
    official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset = load_json(
        "res/level_c/official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset/"
        "level_c_official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset.json"
    )
    official_csv_loop_full_bundle_state_latent_diffusion_training = load_json(
        "res/level_c/official_csv_loop_full_bundle_state_latent_diffusion_training/"
        "level_c_official_csv_loop_full_bundle_state_latent_diffusion_training.json"
    )
    official_importer_export_full_bundle_teacher_rollout_state_latent_dataset = load_json(
        "res/level_c/official_importer_export_full_bundle_teacher_rollout_state_latent_dataset/"
        "level_c_official_importer_export_full_bundle_teacher_rollout_state_latent_dataset.json"
    )
    official_importer_export_full_bundle_state_latent_diffusion_training = load_json(
        "res/level_c/official_importer_export_full_bundle_state_latent_diffusion_training/"
        "level_c_official_importer_export_full_bundle_state_latent_diffusion_training.json"
    )
    official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset = load_json(
        "res/level_c/official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset/"
        "level_c_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset.json"
    )
    official_importer_export_scaled_ppo_state_latent_diffusion_training = load_json(
        "res/level_c/official_importer_export_scaled_ppo_state_latent_diffusion_training/"
        "level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training.json"
    )
    official_csv_loop_full_bundle_downstream_assets = load_json(
        "res/report_assets/official_csv_loop_full_bundle_downstream/"
        "official_csv_loop_full_bundle_downstream_report_assets.json"
    )
    official_importer_export_full_bundle_downstream_assets = load_json(
        "res/report_assets/official_importer_export_full_bundle_downstream/"
        "official_importer_export_full_bundle_downstream_report_assets.json"
    )
    official_importer_export_scaled_ppo_downstream_assets = load_json(
        "res/report_assets/official_importer_export_scaled_ppo_downstream/"
        "official_importer_export_full_bundle_downstream_report_assets.json"
    )
    official_importer_export_full_bundle_vae_assets = load_json(
        "res/report_assets/official_importer_export_full_bundle_vae_training/"
        "official_importer_export_full_bundle_vae_training_assets.json"
    )
    official_importer_export_full_bundle_vae_closed_loop_rollout_eval = load_json(
        "res/level_c/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/"
        "tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_eval.json"
    )
    official_importer_export_full_bundle_vae_closed_loop_rollout_assets = load_json(
        "res/report_assets/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/"
        "official_importer_export_full_bundle_vae_closed_loop_rollout_assets.json"
    )
    official_csv_loop_state_latent_guidance_eval = load_json(
        "res/level_c/official_csv_loop_state_latent_guidance_eval/"
        "level_c_official_csv_loop_state_latent_guidance_eval.json"
    )
    official_csv_loop_full_bundle_state_latent_guidance_eval = load_json(
        "res/level_c/official_csv_loop_full_bundle_state_latent_guidance_eval/"
        "level_c_official_csv_loop_full_bundle_state_latent_guidance_eval.json"
    )
    official_importer_export_full_bundle_state_latent_guidance_eval = load_json(
        "res/level_c/official_importer_export_full_bundle_state_latent_guidance_eval/"
        "level_c_official_importer_export_full_bundle_state_latent_guidance_eval.json"
    )
    official_importer_export_scaled_ppo_state_latent_guidance_eval = load_json(
        "res/level_c/official_importer_export_scaled_ppo_state_latent_guidance_eval/"
        "level_c_official_importer_export_scaled_ppo_state_latent_guidance_eval.json"
    )
    official_importer_export_scaled_ppo_guidance_assets = load_json(
        "res/report_assets/official_importer_export_scaled_ppo_guidance/"
        "official_importer_export_scaled_ppo_guidance_report_assets.json"
    )
    official_csv_loop_full_bundle_guidance_assets = load_json(
        "res/report_assets/official_csv_loop_full_bundle_guidance/"
        "official_csv_loop_full_bundle_guidance_report_assets.json"
    )
    official_csv_loop_guidance_vae_action_decode_eval = load_json(
        "res/level_c/official_csv_loop_guidance_vae_action_decode_eval/"
        "level_c_official_csv_loop_guidance_vae_action_decode_eval.json"
    )
    official_csv_loop_guidance_vae_action_decode_assets = load_json(
        "res/report_assets/official_csv_loop_guidance_vae_action_decode/"
        "official_csv_loop_guidance_vae_action_decode_assets.json"
    )
    official_csv_loop_guided_action_rollout_probe = load_json(
        "res/level_c/official_csv_loop_guided_action_rollout_probe/"
        "tracking_g1_official_csv_loop_guided_action_rollout_probe.json"
    )
    official_csv_loop_guided_action_rollout_probe_assets = load_json(
        "res/level_c/official_csv_loop_guided_action_rollout_probe/"
        "official_csv_loop_guided_action_rollout_probe_assets.json"
    )
    official_csv_loop_vae_closed_loop_rollout_eval = load_json(
        "res/level_c/official_csv_loop_vae_closed_loop_rollout_eval/"
        "tracking_g1_official_csv_loop_vae_closed_loop_rollout_eval.json"
    )
    official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval = load_json(
        "res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval/"
        "level_c_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval.json"
    )
    official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval = load_json(
        "res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval/"
        "level_c_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval.json"
    )
    official_importer_export_scaled_ppo_task_conditioned_guidance_summary_assets = load_json(
        "res/report_assets/official_importer_export_scaled_ppo_task_conditioned_guidance_summary/"
        "official_csv_loop_task_conditioned_guidance_summary_assets.json"
    )
    official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval = load_json(
        "res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval/"
        "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json"
    )
    official_importer_export_full_bundle_task_conditioned_guidance_multiseed_assets = load_json(
        "res/report_assets/official_importer_export_full_bundle_task_conditioned_guidance_multiseed/"
        "official_importer_export_full_bundle_task_conditioned_guidance_multiseed_assets.json"
    )
    official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval = load_json(
        "res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval/"
        "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval.json"
    )
    official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed_assets = load_json(
        "res/report_assets/official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed/"
        "official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed_assets.json"
    )
    official_importer_export_full_bundle_task_conditioned_guidance_success_boundary = load_json(
        "res/report_assets/official_importer_export_full_bundle_task_conditioned_guidance_success_boundary/"
        "local_proxy_success_boundary.json"
    )
    official_importer_export_full_bundle_inpainting_guidance_rollout_eval = load_json(
        "res/level_c/official_importer_export_full_bundle_inpainting_guidance_rollout_eval/"
        "level_c_official_importer_export_full_bundle_inpainting_guidance_rollout_eval.json"
    )
    official_importer_export_full_bundle_transition_guidance_rollout_eval = load_json(
        "res/level_c/official_importer_export_full_bundle_transition_guidance_rollout_eval/"
        "level_c_official_importer_export_full_bundle_transition_guidance_rollout_eval.json"
    )
    official_importer_export_full_bundle_guidance_video_contact_sheet = load_json(
        "res/report_assets/official_importer_export_full_bundle_guidance_video_contact_sheet/"
        "importer_export_guidance_video_index.json"
    )
    official_csv_loop_vae_closed_loop_rollout_assets = load_json(
        "res/report_assets/official_csv_loop_vae_closed_loop_rollout_eval/"
        "official_csv_loop_vae_closed_loop_rollout_assets.json"
    )
    official_csv_loop_vae_denoiser_onnx_async_audit = load_json(
        "res/level_c/official_csv_loop_vae_denoiser_onnx_async/"
        "level_c_official_csv_loop_vae_denoiser_onnx_async_audit.json"
    )
    official_csv_loop_full_bundle_vae_denoiser_onnx_async_audit = load_json(
        "res/level_c/official_csv_loop_full_bundle_vae_denoiser_onnx_async/"
        "level_c_official_csv_loop_full_bundle_vae_denoiser_onnx_async_audit.json"
    )
    official_importer_export_full_bundle_vae_denoiser_onnx_async_audit = load_json(
        "res/level_c/official_importer_export_full_bundle_vae_denoiser_onnx_async/"
        "level_c_official_importer_export_full_bundle_vae_denoiser_onnx_async_audit.json"
    )
    official_importer_export_scaled_ppo_vae_denoiser_onnx_async_audit = load_json(
        "res/level_c/official_importer_export_scaled_ppo_vae_denoiser_onnx_async/"
        "level_c_official_importer_export_scaled_ppo_vae_denoiser_onnx_async_audit.json"
    )
    resource_adjusted_teacher_rollout_state_latent_dataset = load_json(
        "res/level_c/resource_adjusted_teacher_rollout_state_latent_dataset/"
        "level_c_resource_adjusted_teacher_rollout_state_latent_dataset.json"
    )
    resource_adjusted_state_latent_diffusion_training = load_json(
        "res/level_c/resource_adjusted_state_latent_diffusion_training/"
        "level_c_resource_adjusted_state_latent_diffusion_training.json"
    )
    resource_adjusted_state_latent_guidance_eval = load_json(
        "res/level_c/resource_adjusted_state_latent_guidance_eval/"
        "level_c_resource_adjusted_state_latent_guidance_eval.json"
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
            "tracking_official_replay_npz_entry_diagnostic_status": tracking_official_replay_npz_entry_diagnostic[
                "status"
            ],
            "tracking_official_replay_npz_entry_diagnostic_latest_blocker": (
                tracking_official_replay_npz_entry_diagnostic["latest_blocker"]
            ),
            "tracking_official_replay_npz_entry_diagnostic_checks": tracking_official_replay_npz_entry_diagnostic[
                "checks"
            ],
            "tracking_official_replay_npz_entry_diagnostic_markers": tracking_official_replay_npz_entry_diagnostic[
                "run"
            ]["markers"],
            "tracking_official_replay_npz_entry_diagnostic_json": str(
                ROOT
                / "res/tracking/official_replay_npz_entry_diagnostic/"
                "tracking_official_replay_npz_entry_diagnostic_audit.json"
            ),
            "tracking_official_replay_npz_loop_with_enriched_usd_status": (
                tracking_official_replay_npz_loop_with_enriched_usd["status"]
            ),
            "tracking_official_replay_npz_loop_with_enriched_usd_latest_blocker": (
                tracking_official_replay_npz_loop_with_enriched_usd["latest_blocker"]
            ),
            "tracking_official_replay_npz_loop_with_enriched_usd_checks": (
                tracking_official_replay_npz_loop_with_enriched_usd["checks"]
            ),
            "tracking_official_replay_npz_loop_with_enriched_usd_markers": (
                tracking_official_replay_npz_loop_with_enriched_usd["run"]["markers"]
            ),
            "tracking_official_replay_npz_loop_with_enriched_usd_json": str(
                ROOT
                / "res/tracking/official_replay_npz_loop_with_enriched_usd/"
                "tracking_official_replay_npz_loop_with_enriched_usd_audit.json"
            ),
            "tracking_official_replay_npz_loop_full_dataset_with_enriched_usd_status": (
                tracking_official_replay_npz_loop_full_dataset_with_enriched_usd["status"]
            ),
            "tracking_official_replay_npz_loop_full_dataset_with_enriched_usd_aggregate": (
                tracking_official_replay_npz_loop_full_dataset_with_enriched_usd["aggregate"]
            ),
            "tracking_official_replay_npz_loop_full_dataset_with_enriched_usd_checks": (
                tracking_official_replay_npz_loop_full_dataset_with_enriched_usd["checks"]
            ),
            "tracking_official_replay_npz_loop_full_dataset_with_enriched_usd_json": str(
                ROOT
                / "res/tracking/official_replay_npz_loop_full_dataset_with_enriched_usd/"
                "tracking_official_replay_npz_loop_full_dataset_with_enriched_usd_audit.json"
            ),
            "tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_status": (
                tracking_official_replay_npz_loop_full_dataset_with_official_importer_export["status"]
            ),
            "tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_aggregate": (
                tracking_official_replay_npz_loop_full_dataset_with_official_importer_export["aggregate"]
            ),
            "tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_checks": (
                tracking_official_replay_npz_loop_full_dataset_with_official_importer_export["checks"]
            ),
            "tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_json": str(
                ROOT
                / "res/tracking/official_replay_npz_loop_full_dataset_with_official_importer_export/"
                "tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_audit.json"
            ),
            "official_importer_export_replay_full_dataset_report_assets": (
                official_importer_export_replay_full_dataset_report_assets
            ),
            "tracking_official_csv_to_npz_loop_with_enriched_usd_status": (
                tracking_official_csv_to_npz_loop_with_enriched_usd["status"]
            ),
            "tracking_official_csv_to_npz_loop_with_enriched_usd_latest_blocker": (
                tracking_official_csv_to_npz_loop_with_enriched_usd["latest_blocker"]
            ),
            "tracking_official_csv_to_npz_loop_with_enriched_usd_checks": (
                tracking_official_csv_to_npz_loop_with_enriched_usd["checks"]
            ),
            "tracking_official_csv_to_npz_loop_with_enriched_usd_metrics": (
                tracking_official_csv_to_npz_loop_with_enriched_usd["metrics"]
            ),
            "tracking_official_csv_to_npz_loop_with_enriched_usd_markers": (
                tracking_official_csv_to_npz_loop_with_enriched_usd["run"]["markers"]
            ),
            "tracking_official_csv_to_npz_loop_with_enriched_usd_json": str(
                ROOT
                / "res/tracking/official_csv_to_npz_loop_with_enriched_usd/"
                "tracking_official_csv_to_npz_loop_with_enriched_usd_audit.json"
            ),
            "tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_status": (
                tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd["status"]
            ),
            "tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_aggregate": (
                tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd["aggregate"]
            ),
            "tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_checks": (
                tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd["checks"]
            ),
            "tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_json": str(
                ROOT
                / "res/tracking/official_csv_to_npz_loop_full_dataset_with_enriched_usd/"
                "tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_audit.json"
            ),
            "tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_status": (
                tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export["status"]
            ),
            "tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_aggregate": (
                tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export["aggregate"]
            ),
            "tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_checks": (
                tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export["checks"]
            ),
            "tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_json": str(
                ROOT
                / "res/tracking/official_csv_to_npz_loop_full_dataset_with_official_importer_export/"
                "tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_audit.json"
            ),
            "tracking_g1_official_csv_loop_full_dataset_task_eval_status": (
                tracking_g1_official_csv_loop_full_dataset_task_eval["status"]
            ),
            "tracking_g1_official_csv_loop_full_dataset_task_eval_aggregate": (
                tracking_g1_official_csv_loop_full_dataset_task_eval["aggregate"]
            ),
            "tracking_g1_official_csv_loop_full_dataset_task_eval_checks": (
                tracking_g1_official_csv_loop_full_dataset_task_eval["checks"]
            ),
            "tracking_g1_official_csv_loop_full_dataset_task_eval_interpretation": (
                tracking_g1_official_csv_loop_full_dataset_task_eval["interpretation"]
            ),
            "tracking_g1_official_csv_loop_full_dataset_task_eval_report_assets": (
                tracking_g1_official_csv_loop_full_dataset_task_eval["outputs"]["report_assets"]
            ),
            "tracking_g1_official_csv_loop_full_dataset_task_eval_json": str(
                ROOT
                / "res/tracking/g1_official_csv_loop_full_dataset_task_eval/"
                "tracking_g1_official_csv_loop_full_dataset_task_eval.json"
            ),
            "tracking_g1_official_importer_export_task_smoke_status": (
                tracking_g1_official_importer_export_task_smoke["status"]
            ),
            "tracking_g1_official_importer_export_task_smoke_checks": (
                tracking_g1_official_importer_export_task_smoke["checks"]
            ),
            "tracking_g1_official_importer_export_task_smoke_metrics": (
                tracking_g1_official_importer_export_task_smoke["metrics"]
            ),
            "tracking_g1_official_importer_export_task_smoke_json": str(
                ROOT
                / "res/tracking/g1_official_importer_export_task_smoke/"
                "tracking_g1_official_importer_export_task_smoke.json"
            ),
            "tracking_g1_official_importer_export_full_dataset_task_eval_status": (
                tracking_g1_official_importer_export_full_dataset_task_eval["status"]
            ),
            "tracking_g1_official_importer_export_full_dataset_task_eval_aggregate": (
                tracking_g1_official_importer_export_full_dataset_task_eval["aggregate"]
            ),
            "tracking_g1_official_importer_export_full_dataset_task_eval_checks": (
                tracking_g1_official_importer_export_full_dataset_task_eval["checks"]
            ),
            "tracking_g1_official_importer_export_full_dataset_task_eval_interpretation": (
                tracking_g1_official_importer_export_full_dataset_task_eval["interpretation"]
            ),
            "tracking_g1_official_importer_export_full_dataset_task_eval_report_assets": (
                tracking_g1_official_importer_export_full_dataset_task_eval["outputs"]["report_assets"]
            ),
            "tracking_g1_official_importer_export_full_dataset_task_eval_json": str(
                ROOT
                / "res/tracking/g1_official_importer_export_full_dataset_task_eval/"
                "tracking_g1_official_importer_export_full_dataset_task_eval.json"
            ),
            "tracking_g1_urdf_import_config_variant_probe_status": (
                tracking_g1_urdf_import_config_variant_probe["status"]
            ),
            "tracking_g1_urdf_import_config_variant_probe_current_blocker": (
                tracking_g1_urdf_import_config_variant_probe["current_blocker"]
            ),
            "tracking_g1_urdf_import_config_variant_probe_method_payload": (
                tracking_g1_urdf_import_config_variant_probe["method_probe"]["payload"]
            ),
            "tracking_g1_urdf_import_config_variant_probe_baseline_usd": (
                tracking_g1_urdf_import_config_variant_probe["variant_summary"][
                    "variant_baseline_make_instanceable_false"
                ]["usd"]
            ),
            "tracking_g1_urdf_import_config_variant_probe_skipped_variants": (
                tracking_g1_urdf_import_config_variant_probe["skipped_variants"]
            ),
            "tracking_g1_urdf_import_config_variant_probe_json": str(
                ROOT
                / "res/tracking/g1_urdf_import_config_variant_probe/"
                "tracking_g1_urdf_import_config_variant_probe.json"
            ),
            "tracking_g1_enriched_usd_replay_preflight_status": tracking_g1_enriched_usd_replay_preflight["status"],
            "tracking_g1_enriched_usd_replay_preflight_latest_blocker": tracking_g1_enriched_usd_replay_preflight[
                "latest_blocker"
            ],
            "tracking_g1_enriched_usd_replay_preflight_checks": tracking_g1_enriched_usd_replay_preflight["checks"],
            "tracking_g1_enriched_usd_replay_preflight_markers": tracking_g1_enriched_usd_replay_preflight[
                "markers"
            ],
            "tracking_g1_enriched_usd_replay_preflight_json": str(
                ROOT
                / "res/tracking/g1_enriched_usd_replay_preflight/"
                "tracking_g1_enriched_usd_replay_preflight_audit.json"
            ),
            "tracking_g1_enriched_usd_bounded_replay_metrics_status": tracking_g1_enriched_usd_bounded_replay_metrics[
                "status"
            ],
            "tracking_g1_enriched_usd_bounded_replay_metrics_metrics": tracking_g1_enriched_usd_bounded_replay_metrics[
                "metrics"
            ],
            "tracking_g1_enriched_usd_bounded_replay_metrics_checks": tracking_g1_enriched_usd_bounded_replay_metrics[
                "checks"
            ],
            "tracking_g1_enriched_usd_bounded_replay_metrics_json": str(
                ROOT
                / "res/tracking/g1_enriched_usd_bounded_replay_metrics/"
                "tracking_g1_enriched_usd_bounded_replay_metrics_audit.json"
            ),
            "tracking_g1_resource_adjusted_task_smoke_status": tracking_g1_resource_adjusted_task_smoke["status"],
            "tracking_g1_resource_adjusted_task_smoke_metrics": tracking_g1_resource_adjusted_task_smoke["metrics"],
            "tracking_g1_resource_adjusted_task_smoke_checks": tracking_g1_resource_adjusted_task_smoke["checks"],
            "tracking_g1_resource_adjusted_task_smoke_json": str(
                ROOT
                / "res/tracking/g1_resource_adjusted_task_smoke/"
                "tracking_g1_resource_adjusted_task_smoke_audit.json"
            ),
            "tracking_g1_resource_adjusted_multi_fixture_eval_status": tracking_g1_resource_adjusted_multi_fixture_eval[
                "status"
            ],
            "tracking_g1_resource_adjusted_multi_fixture_eval_metrics": tracking_g1_resource_adjusted_multi_fixture_eval[
                "metrics"
            ],
            "tracking_g1_resource_adjusted_multi_fixture_eval_checks": tracking_g1_resource_adjusted_multi_fixture_eval[
                "checks"
            ],
            "tracking_g1_resource_adjusted_multi_fixture_eval_json": str(
                ROOT
                / "res/tracking/g1_resource_adjusted_multi_fixture_eval/"
                "tracking_g1_resource_adjusted_multi_fixture_eval_audit.json"
            ),
            "tracking_g1_resource_adjusted_csv_conversion_status": tracking_g1_resource_adjusted_csv_conversion[
                "status"
            ],
            "tracking_g1_resource_adjusted_csv_conversion_metrics": tracking_g1_resource_adjusted_csv_conversion[
                "metrics"
            ],
            "tracking_g1_resource_adjusted_csv_conversion_checks": tracking_g1_resource_adjusted_csv_conversion[
                "checks"
            ],
            "tracking_g1_resource_adjusted_csv_conversion_json": str(
                ROOT
                / "res/tracking/g1_resource_adjusted_csv_conversion/"
                "tracking_g1_resource_adjusted_csv_conversion_audit.json"
            ),
            "tracking_g1_resource_adjusted_csv_full_replay_status": tracking_g1_resource_adjusted_csv_full_replay[
                "status"
            ],
            "tracking_g1_resource_adjusted_csv_full_replay_metrics": tracking_g1_resource_adjusted_csv_full_replay[
                "metrics"
            ],
            "tracking_g1_resource_adjusted_csv_full_replay_checks": tracking_g1_resource_adjusted_csv_full_replay[
                "checks"
            ],
            "tracking_g1_resource_adjusted_csv_full_replay_json": str(
                ROOT
                / "res/tracking/g1_resource_adjusted_csv_full_replay/"
                "tracking_g1_resource_adjusted_csv_full_replay_audit.json"
            ),
            "tracking_g1_resource_adjusted_csv_task_eval_status": tracking_g1_resource_adjusted_csv_task_eval[
                "status"
            ],
            "tracking_g1_resource_adjusted_csv_task_eval_metrics": tracking_g1_resource_adjusted_csv_task_eval[
                "metrics"
            ],
            "tracking_g1_resource_adjusted_csv_task_eval_checks": tracking_g1_resource_adjusted_csv_task_eval[
                "checks"
            ],
            "tracking_g1_resource_adjusted_csv_task_eval_json": str(
                ROOT
                / "res/tracking/g1_resource_adjusted_csv_task_eval/"
                "tracking_g1_resource_adjusted_csv_task_eval_audit.json"
            ),
            "tracking_g1_resource_adjusted_train_entry_diagnostic_status": (
                tracking_g1_resource_adjusted_train_entry_diagnostic["status"]
            ),
            "tracking_g1_resource_adjusted_train_entry_diagnostic_metrics": (
                tracking_g1_resource_adjusted_train_entry_diagnostic["metrics"]
            ),
            "tracking_g1_resource_adjusted_train_entry_diagnostic_checks": (
                tracking_g1_resource_adjusted_train_entry_diagnostic["checks"]
            ),
            "tracking_g1_resource_adjusted_train_entry_diagnostic_warning": (
                train_entry_runtime_warning
            ),
            "tracking_g1_resource_adjusted_train_entry_diagnostic_json": str(
                ROOT
                / "res/tracking/g1_resource_adjusted_train_entry_diagnostic/"
                "tracking_g1_resource_adjusted_train_entry_diagnostic_audit.json"
            ),
            "tracking_g1_resource_adjusted_ppo_training_run_status": (
                tracking_g1_resource_adjusted_ppo_training_run["status"]
            ),
            "tracking_g1_resource_adjusted_ppo_training_run_config": (
                tracking_g1_resource_adjusted_ppo_training_run["config"]
            ),
            "tracking_g1_resource_adjusted_ppo_training_run_gpu_preflight": (
                tracking_g1_resource_adjusted_ppo_training_run["gpu_preflight"]
            ),
            "tracking_g1_resource_adjusted_ppo_training_run_attempted": (
                tracking_g1_resource_adjusted_ppo_training_run["run"]["attempted_training"]
            ),
            "tracking_g1_resource_adjusted_ppo_training_run_reason_not_started": (
                tracking_g1_resource_adjusted_ppo_training_run["run"].get("reason_not_started", "")
            ),
            "tracking_g1_resource_adjusted_ppo_training_run_checkpoint_count": (
                tracking_g1_resource_adjusted_ppo_training_run["run"].get("checkpoint_count", 0)
            ),
            "tracking_g1_resource_adjusted_ppo_training_run_json": str(
                ROOT
                / "res/tracking/g1_resource_adjusted_ppo_training_run/"
                "tracking_g1_resource_adjusted_ppo_training_run.json"
            ),
            "tracking_g1_resource_adjusted_ppo_checkpoint_eval_status": (
                tracking_g1_resource_adjusted_ppo_checkpoint_eval["status"]
            ),
            "tracking_g1_resource_adjusted_ppo_checkpoint_eval_config": (
                tracking_g1_resource_adjusted_ppo_checkpoint_eval["config"]
            ),
            "tracking_g1_resource_adjusted_ppo_checkpoint_eval_metrics": (
                tracking_g1_resource_adjusted_ppo_checkpoint_eval["run"].get("metrics", {})
            ),
            "tracking_g1_resource_adjusted_ppo_checkpoint_eval_duration_seconds": (
                tracking_g1_resource_adjusted_ppo_checkpoint_eval["run"].get("duration_seconds")
            ),
            "tracking_g1_resource_adjusted_ppo_checkpoint_eval_json": str(
                ROOT
                / "res/tracking/g1_resource_adjusted_ppo_checkpoint_eval/"
                "tracking_g1_resource_adjusted_ppo_checkpoint_eval.json"
            ),
            "tracking_g1_official_csv_loop_ppo_training_run_status": (
                tracking_g1_official_csv_loop_ppo_training_run["status"]
            ),
            "tracking_g1_official_csv_loop_ppo_training_run_config": (
                tracking_g1_official_csv_loop_ppo_training_run["config"]
            ),
            "tracking_g1_official_csv_loop_ppo_training_run_rank_metrics": (
                tracking_g1_official_csv_loop_ppo_training_run["run"].get("rank_metrics", [])
            ),
            "tracking_g1_official_csv_loop_ppo_training_run_duration_seconds": (
                tracking_g1_official_csv_loop_ppo_training_run["run"].get("duration_seconds")
            ),
            "tracking_g1_official_csv_loop_ppo_training_run_checkpoint_count": (
                tracking_g1_official_csv_loop_ppo_training_run["run"].get("checkpoint_count", 0)
            ),
            "tracking_g1_official_csv_loop_ppo_training_run_json": str(
                ROOT
                / "res/tracking/g1_official_csv_loop_ppo_training_run/"
                "tracking_g1_official_csv_loop_ppo_training_run.json"
            ),
            "tracking_g1_official_csv_loop_ppo_checkpoint_eval_status": (
                tracking_g1_official_csv_loop_ppo_checkpoint_eval["status"]
            ),
            "tracking_g1_official_csv_loop_ppo_checkpoint_eval_config": (
                tracking_g1_official_csv_loop_ppo_checkpoint_eval["config"]
            ),
            "tracking_g1_official_csv_loop_ppo_checkpoint_eval_metrics": (
                tracking_g1_official_csv_loop_ppo_checkpoint_eval["run"].get("metrics", {})
            ),
            "tracking_g1_official_csv_loop_ppo_checkpoint_eval_duration_seconds": (
                tracking_g1_official_csv_loop_ppo_checkpoint_eval["run"].get("duration_seconds")
            ),
            "tracking_g1_official_csv_loop_ppo_checkpoint_eval_json": str(
                ROOT
                / "res/tracking/g1_official_csv_loop_ppo_checkpoint_eval/"
                "tracking_g1_official_csv_loop_ppo_checkpoint_eval.json"
            ),
            "tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval_status": (
                tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval["status"]
            ),
            "tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval_config": (
                tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval["config"]
            ),
            "tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval_metrics": (
                tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval["metrics"]
            ),
            "tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval_aggregate": (
                tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval["aggregate"]
            ),
            "tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval_json": str(
                ROOT
                / "res/tracking/g1_official_csv_loop_ppo_checkpoint_multiseed_eval/"
                "tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval.json"
            ),
            "tracking_g1_official_csv_loop_full_bundle_motion_npz_status": (
                tracking_g1_official_csv_loop_full_bundle_motion_npz["status"]
            ),
            "tracking_g1_official_csv_loop_full_bundle_motion_npz_bundle": (
                tracking_g1_official_csv_loop_full_bundle_motion_npz["bundle"]
            ),
            "tracking_g1_official_csv_loop_full_bundle_motion_npz_interpretation": (
                tracking_g1_official_csv_loop_full_bundle_motion_npz["interpretation"]
            ),
            "tracking_g1_official_csv_loop_full_bundle_ppo_training_run_status": (
                tracking_g1_official_csv_loop_full_bundle_ppo_training_run["status"]
            ),
            "tracking_g1_official_csv_loop_full_bundle_ppo_training_run_config": (
                tracking_g1_official_csv_loop_full_bundle_ppo_training_run["config"]
            ),
            "tracking_g1_official_csv_loop_full_bundle_ppo_training_run_rank_metrics": (
                tracking_g1_official_csv_loop_full_bundle_ppo_training_run["run"].get("rank_metrics", [])
            ),
            "tracking_g1_official_csv_loop_full_bundle_ppo_training_run_duration_seconds": (
                tracking_g1_official_csv_loop_full_bundle_ppo_training_run["run"].get("duration_seconds")
            ),
            "tracking_g1_official_csv_loop_full_bundle_ppo_training_run_checkpoint_count": (
                tracking_g1_official_csv_loop_full_bundle_ppo_training_run["run"].get("checkpoint_count", 0)
            ),
            "tracking_g1_official_csv_loop_full_bundle_ppo_checkpoint_eval_status": (
                tracking_g1_official_csv_loop_full_bundle_ppo_checkpoint_eval["status"]
            ),
            "tracking_g1_official_csv_loop_full_bundle_ppo_checkpoint_eval_config": (
                tracking_g1_official_csv_loop_full_bundle_ppo_checkpoint_eval["config"]
            ),
            "tracking_g1_official_csv_loop_full_bundle_ppo_checkpoint_eval_metrics": (
                tracking_g1_official_csv_loop_full_bundle_ppo_checkpoint_eval["run"].get("metrics", {})
            ),
            "tracking_g1_official_csv_loop_full_bundle_ppo_checkpoint_eval_duration_seconds": (
                tracking_g1_official_csv_loop_full_bundle_ppo_checkpoint_eval["run"].get("duration_seconds")
            ),
            "official_csv_loop_ppo_eval_report_assets": official_csv_loop_ppo_eval_report_assets,
            "official_csv_loop_ppo_multiseed_eval_report_assets": official_csv_loop_ppo_multiseed_eval_report_assets,
            "official_csv_loop_full_bundle_ppo_eval_report_assets": official_csv_loop_full_bundle_ppo_eval_report_assets,
            "tracking_g1_official_importer_export_full_bundle_ppo_training_run_status": (
                tracking_g1_official_importer_export_full_bundle_ppo_training_run["status"]
            ),
            "tracking_g1_official_importer_export_full_bundle_ppo_training_run_config": (
                tracking_g1_official_importer_export_full_bundle_ppo_training_run["config"]
            ),
            "tracking_g1_official_importer_export_full_bundle_ppo_training_run_rank_metrics": (
                tracking_g1_official_importer_export_full_bundle_ppo_training_run["run"].get("rank_metrics", [])
            ),
            "tracking_g1_official_importer_export_full_bundle_ppo_training_run_duration_seconds": (
                tracking_g1_official_importer_export_full_bundle_ppo_training_run["run"].get("duration_seconds")
            ),
            "tracking_g1_official_importer_export_full_bundle_ppo_training_run_checkpoint_count": (
                tracking_g1_official_importer_export_full_bundle_ppo_training_run["run"].get("checkpoint_count", 0)
            ),
            "tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval_status": (
                tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval["status"]
            ),
            "tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval_config": (
                tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval["config"]
            ),
            "tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval_metrics": (
                tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval["run"].get("metrics", {})
            ),
            "tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval_duration_seconds": (
                tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval["run"].get("duration_seconds")
            ),
            "official_importer_export_full_bundle_ppo_eval_report_assets": (
                official_importer_export_full_bundle_ppo_eval_report_assets
            ),
            "tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run_status": (
                tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run["status"]
            ),
            "tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run_config": (
                tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run["config"]
            ),
            "tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run_rank_metrics": (
                tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run["run"].get(
                    "rank_metrics", []
                )
            ),
            "tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run_duration_seconds": (
                tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run["run"].get(
                    "duration_seconds"
                )
            ),
            "tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run_checkpoint_count": (
                tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run["run"].get(
                    "checkpoint_count", 0
                )
            ),
            "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval_status": (
                tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval["status"]
            ),
            "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval_config": (
                tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval["config"]
            ),
            "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval_metrics": (
                tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval["run"].get("metrics", {})
            ),
            "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval_duration_seconds": (
                tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval["run"].get(
                    "duration_seconds"
                )
            ),
            "official_importer_export_full_bundle_scaled_ppo_eval_report_assets": (
                official_importer_export_full_bundle_scaled_ppo_eval_report_assets
            ),
            "official_importer_export_scaled_ppo_checkpoint_completion_proxy": (
                official_importer_export_scaled_ppo_checkpoint_completion_proxy
            ),
            "tracking_g1_official_importer_export_scaled_ppo_checkpoint_sweep": (
                tracking_g1_official_importer_export_scaled_ppo_checkpoint_sweep
            ),
            "tracking_g1_official_importer_export_scaled_ppo_best_checkpoint_confirmation_eval": (
                tracking_g1_official_importer_export_scaled_ppo_best_checkpoint_confirmation_eval
            ),
            "official_importer_export_scaled_ppo_reward_termination_diagnostic": (
                official_importer_export_scaled_ppo_reward_termination_diagnostic
            ),
            "official_importer_export_scaled_ppo_ee_body_pos_termination_source_audit": (
                official_importer_export_scaled_ppo_ee_body_pos_termination_source_audit
            ),
            "tracking_g1_official_importer_export_scaled_ppo_endpoint_z_error_trace": (
                tracking_g1_official_importer_export_scaled_ppo_endpoint_z_error_trace
            ),
            "tracking_g1_official_importer_export_scaled_ppo_checkpoint_multiseed_eval_status": (
                tracking_g1_official_importer_export_scaled_ppo_checkpoint_multiseed_eval["status"]
            ),
            "tracking_g1_official_importer_export_scaled_ppo_checkpoint_multiseed_eval_config": (
                tracking_g1_official_importer_export_scaled_ppo_checkpoint_multiseed_eval["config"]
            ),
            "tracking_g1_official_importer_export_scaled_ppo_checkpoint_multiseed_eval_metrics": (
                tracking_g1_official_importer_export_scaled_ppo_checkpoint_multiseed_eval["metrics"]
            ),
            "tracking_g1_official_importer_export_scaled_ppo_checkpoint_multiseed_eval_aggregate": (
                tracking_g1_official_importer_export_scaled_ppo_checkpoint_multiseed_eval["aggregate"]
            ),
            "official_importer_export_scaled_ppo_checkpoint_multiseed_eval_report_assets": (
                official_importer_export_scaled_ppo_checkpoint_multiseed_eval_report_assets
            ),
            "official_importer_export_full_bundle_scaled_ppo_policy_rollout_capture": (
                official_importer_export_full_bundle_scaled_ppo_policy_rollout_capture
            ),
            "official_importer_export_full_bundle_scaled_ppo_policy_rollout_video_asset": (
                official_importer_export_full_bundle_scaled_ppo_policy_rollout_video_asset
            ),
            "official_importer_export_tracking_eval_summary_assets": (
                official_importer_export_tracking_eval_summary_assets
            ),
            "tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset_status": (
                tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset["status"]
            ),
            "tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset_config": (
                tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset["config"]
            ),
            "tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset_aggregate": (
                tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset["aggregate_metrics"]
            ),
            "tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset_duration_seconds": (
                tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset["run"].get(
                    "duration_seconds"
                )
            ),
            "tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset_gpu_metrics": (
                tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset["run"].get(
                    "gpu_metrics_summary", {}
                )
            ),
            "tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset_json": str(
                ROOT
                / "res/tracking/g1_official_importer_export_full_bundle_teacher_rollout_dataset/"
                "tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset.json"
            ),
            "official_importer_export_full_bundle_teacher_rollout_report_assets": (
                official_importer_export_full_bundle_teacher_rollout_report_assets
            ),
            "tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset_status": (
                tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset["status"]
            ),
            "tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset_config": (
                tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset["config"]
            ),
            "tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset_aggregate": (
                tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset[
                    "aggregate_metrics"
                ]
            ),
            "tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset_duration_seconds": (
                tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset["run"].get(
                    "duration_seconds"
                )
            ),
            "tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset_gpu_metrics": (
                tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset["run"].get(
                    "gpu_metrics_summary", {}
                )
            ),
            "tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset_json": str(
                ROOT
                / "res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset/"
                "tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset.json"
            ),
            "official_importer_export_full_bundle_scaled_ppo_teacher_rollout_report_assets": (
                official_importer_export_full_bundle_scaled_ppo_teacher_rollout_report_assets
            ),
            "official_csv_loop_reference_replay_video_asset": official_csv_loop_reference_replay_video_asset,
            "official_importer_export_full_dataset_reference_replay_video_asset": (
                official_importer_export_full_dataset_reference_replay_video_asset
            ),
            "official_csv_loop_policy_rollout_capture": official_csv_loop_policy_rollout_capture,
            "official_csv_loop_policy_rollout_video_asset": official_csv_loop_policy_rollout_video_asset,
            "official_csv_loop_full_bundle_policy_rollout_capture": (
                official_csv_loop_full_bundle_policy_rollout_capture
            ),
            "official_csv_loop_full_bundle_policy_rollout_video_asset": (
                official_csv_loop_full_bundle_policy_rollout_video_asset
            ),
            "official_csv_loop_vae_closed_loop_rollout_capture": official_csv_loop_vae_closed_loop_rollout_capture,
            "official_csv_loop_vae_closed_loop_rollout_video_asset": (
                official_csv_loop_vae_closed_loop_rollout_video_asset
            ),
            "official_importer_export_full_bundle_vae_closed_loop_rollout_capture": (
                official_importer_export_full_bundle_vae_closed_loop_rollout_capture
            ),
            "official_importer_export_full_bundle_vae_closed_loop_rollout_video_asset": (
                official_importer_export_full_bundle_vae_closed_loop_rollout_video_asset
            ),
            "official_csv_loop_action_guidance_rollout_asset": official_csv_loop_action_guidance_rollout_asset,
            "official_csv_loop_teacher_rollout_report_assets": official_csv_loop_teacher_rollout_report_assets,
            "official_csv_loop_full_bundle_teacher_rollout_report_assets": (
                official_csv_loop_full_bundle_teacher_rollout_report_assets
            ),
            "tracking_g1_resource_adjusted_teacher_rollout_dataset_status": (
                tracking_g1_resource_adjusted_teacher_rollout_dataset["status"]
            ),
            "tracking_g1_resource_adjusted_teacher_rollout_dataset_config": (
                tracking_g1_resource_adjusted_teacher_rollout_dataset["config"]
            ),
            "tracking_g1_resource_adjusted_teacher_rollout_dataset_aggregate": (
                tracking_g1_resource_adjusted_teacher_rollout_dataset["aggregate_metrics"]
            ),
            "tracking_g1_resource_adjusted_teacher_rollout_dataset_duration_seconds": (
                tracking_g1_resource_adjusted_teacher_rollout_dataset["run"].get("duration_seconds")
            ),
            "tracking_g1_resource_adjusted_teacher_rollout_dataset_gpu_metrics": (
                tracking_g1_resource_adjusted_teacher_rollout_dataset["run"].get("gpu_metrics_summary", {})
            ),
            "tracking_g1_resource_adjusted_teacher_rollout_dataset_json": str(
                ROOT
                / "res/tracking/g1_resource_adjusted_teacher_rollout_dataset/"
                "tracking_g1_resource_adjusted_teacher_rollout_dataset.json"
            ),
            "tracking_g1_official_csv_loop_teacher_rollout_dataset_status": (
                tracking_g1_official_csv_loop_teacher_rollout_dataset["status"]
            ),
            "tracking_g1_official_csv_loop_teacher_rollout_dataset_config": (
                tracking_g1_official_csv_loop_teacher_rollout_dataset["config"]
            ),
            "tracking_g1_official_csv_loop_teacher_rollout_dataset_aggregate": (
                tracking_g1_official_csv_loop_teacher_rollout_dataset["aggregate_metrics"]
            ),
            "tracking_g1_official_csv_loop_teacher_rollout_dataset_duration_seconds": (
                tracking_g1_official_csv_loop_teacher_rollout_dataset["run"].get("duration_seconds")
            ),
            "tracking_g1_official_csv_loop_teacher_rollout_dataset_gpu_metrics": (
                tracking_g1_official_csv_loop_teacher_rollout_dataset["run"].get("gpu_metrics_summary", {})
            ),
            "tracking_g1_official_csv_loop_teacher_rollout_dataset_json": str(
                ROOT
                / "res/tracking/g1_official_csv_loop_teacher_rollout_dataset/"
                "tracking_g1_official_csv_loop_teacher_rollout_dataset.json"
            ),
            "tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset_status": (
                tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset["status"]
            ),
            "tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset_config": (
                tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset["config"]
            ),
            "tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset_aggregate": (
                tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset["aggregate_metrics"]
            ),
            "tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset_duration_seconds": (
                tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset["run"].get("duration_seconds")
            ),
            "tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset_gpu_metrics": (
                tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset["run"].get(
                    "gpu_metrics_summary", {}
                )
            ),
            "tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset_json": str(
                ROOT
                / "res/tracking/g1_official_csv_loop_full_bundle_teacher_rollout_dataset/"
                "tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset.json"
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
            "tracking_usd_api_variant_probe_status": tracking_usd_api_variant_probe["status"],
            "tracking_usd_api_variant_probe_current_blocker": tracking_usd_api_variant_probe["current_blocker"],
            "tracking_usd_api_variant_probe_successful_attempt_labels": tracking_usd_api_variant_probe[
                "successful_attempt_labels"
            ],
            "tracking_usd_api_variant_probe_checks": tracking_usd_api_variant_probe["checks"],
            "tracking_usd_api_variant_probe_json": str(
                ROOT / "res/tracking/usd_api_variant_probe/tracking_usd_api_variant_probe.json"
            ),
            "tracking_g1_urdf_stage_export_probe_status": tracking_g1_urdf_stage_export_probe["status"],
            "tracking_g1_urdf_stage_export_probe_current_blocker": tracking_g1_urdf_stage_export_probe[
                "current_blocker"
            ],
            "tracking_g1_urdf_stage_export_probe_checks": tracking_g1_urdf_stage_export_probe["checks"],
            "tracking_g1_urdf_stage_export_probe_parse_result": tracking_g1_urdf_stage_export_probe["probe"][
                "payload"
            ].get("parse_and_import_result"),
            "tracking_g1_urdf_stage_export_probe_patch_events": tracking_g1_urdf_stage_export_probe["probe"][
                "payload"
            ].get("patch_events"),
            "tracking_g1_urdf_stage_export_probe_json": str(
                ROOT
                / "res/tracking/g1_urdf_stage_export_workaround/tracking_g1_urdf_stage_export_workaround_probe.json"
            ),
            "tracking_g1_urdf_layer_save_probe_status": tracking_g1_urdf_layer_save_probe["status"],
            "tracking_g1_urdf_layer_save_probe_current_blocker": tracking_g1_urdf_layer_save_probe[
                "current_blocker"
            ],
            "tracking_g1_urdf_layer_save_probe_checks": tracking_g1_urdf_layer_save_probe["checks"],
            "tracking_g1_urdf_layer_save_probe_parse_result": tracking_g1_urdf_layer_save_probe["probe"][
                "payload"
            ].get("parse_and_import_result"),
            "tracking_g1_urdf_layer_save_probe_layer_save_exception": tracking_g1_urdf_layer_save_probe["probe"][
                "payload"
            ].get("layer_save_patch_assignment_exception"),
            "tracking_g1_urdf_layer_save_probe_layer_save_events": tracking_g1_urdf_layer_save_probe["probe"][
                "payload"
            ].get("layer_save_events"),
            "tracking_g1_urdf_layer_save_probe_json": str(
                ROOT / "res/tracking/g1_urdf_layer_save_workaround/tracking_g1_urdf_layer_save_workaround_probe.json"
            ),
            "tracking_g1_urdf_in_memory_probe_status": tracking_g1_urdf_in_memory_probe["status"],
            "tracking_g1_urdf_in_memory_probe_current_blocker": tracking_g1_urdf_in_memory_probe["current_blocker"],
            "tracking_g1_urdf_in_memory_probe_checks": tracking_g1_urdf_in_memory_probe["checks"],
            "tracking_g1_urdf_in_memory_probe_markers": tracking_g1_urdf_in_memory_probe["probe"]["markers"],
            "tracking_g1_urdf_in_memory_probe_parse_result": tracking_g1_urdf_in_memory_probe["probe"][
                "payload"
            ].get("parse_and_import_result"),
            "tracking_g1_urdf_in_memory_probe_json": str(
                ROOT / "res/tracking/g1_urdf_in_memory_import/tracking_g1_urdf_in_memory_import_probe.json"
            ),
            "tracking_g1_urdf_simulationapp_in_memory_probe_status": tracking_g1_urdf_simulationapp_in_memory_probe[
                "status"
            ],
            "tracking_g1_urdf_simulationapp_in_memory_probe_current_blocker": tracking_g1_urdf_simulationapp_in_memory_probe[
                "current_blocker"
            ],
            "tracking_g1_urdf_simulationapp_in_memory_probe_checks": tracking_g1_urdf_simulationapp_in_memory_probe[
                "checks"
            ],
            "tracking_g1_urdf_simulationapp_in_memory_probe_markers": tracking_g1_urdf_simulationapp_in_memory_probe[
                "probe"
            ][
                "markers"
            ],
            "tracking_g1_urdf_simulationapp_in_memory_probe_returncode": tracking_g1_urdf_simulationapp_in_memory_probe[
                "probe"
            ][
                "returncode"
            ],
            "tracking_g1_urdf_simulationapp_in_memory_probe_parse_result": tracking_g1_urdf_simulationapp_in_memory_probe[
                "probe"
            ][
                "payload"
            ].get(
                "parse_and_import_result"
            ),
            "tracking_g1_urdf_simulationapp_in_memory_probe_log": tracking_g1_urdf_simulationapp_in_memory_probe[
                "probe"
            ][
                "log"
            ],
            "tracking_g1_urdf_simulationapp_in_memory_probe_json": str(
                ROOT
                / "res/tracking/g1_urdf_simulationapp_in_memory_import/"
                "tracking_g1_urdf_simulationapp_in_memory_import_probe.json"
            ),
            "tracking_g1_urdf_in_memory_variant_matrix_probe_status": tracking_g1_urdf_in_memory_variant_matrix_probe[
                "status"
            ],
            "tracking_g1_urdf_in_memory_variant_matrix_probe_current_blocker": tracking_g1_urdf_in_memory_variant_matrix_probe[
                "current_blocker"
            ],
            "tracking_g1_urdf_in_memory_variant_matrix_probe_checks": tracking_g1_urdf_in_memory_variant_matrix_probe[
                "checks"
            ],
            "tracking_g1_urdf_in_memory_variant_matrix_probe_cases": [
                {
                    "name": case["name"],
                    "gpu": case["gpu"],
                    "status": case["status"],
                    "current_blocker": case["current_blocker"],
                    "returncode": case["returncode"],
                    "markers": case["markers"],
                    "log": case["log"],
                }
                for case in tracking_g1_urdf_in_memory_variant_matrix_probe["cases"]
            ],
            "tracking_g1_urdf_in_memory_variant_matrix_probe_json": str(
                ROOT
                / "res/tracking/g1_urdf_in_memory_variant_matrix/"
                "tracking_g1_urdf_in_memory_variant_matrix_probe.json"
            ),
            "tracking_g1_urdf_in_memory_gpu4_probe_status": tracking_g1_urdf_in_memory_gpu4_probe["status"],
            "tracking_g1_urdf_in_memory_gpu4_probe_returncode": tracking_g1_urdf_in_memory_gpu4_probe[
                "returncode"
            ],
            "tracking_g1_urdf_in_memory_gpu4_probe_duration_seconds": tracking_g1_urdf_in_memory_gpu4_probe[
                "duration_seconds"
            ],
            "tracking_g1_urdf_in_memory_gpu4_probe_checks": tracking_g1_urdf_in_memory_gpu4_probe["checks"],
            "tracking_g1_urdf_in_memory_gpu4_probe_latest_blocker": tracking_g1_urdf_in_memory_gpu4_probe.get(
                "latest_blocker", ""
            ),
            "tracking_g1_urdf_in_memory_gpu4_probe_json": str(
                ROOT
                / "res/tracking/g1_urdf_in_memory_gpu4_probe/"
                "tracking_g1_urdf_in_memory_gpu4_probe.json"
            ),
            "tracking_g1_urdf_in_memory_gpu4_export_structure_status": tracking_g1_urdf_in_memory_gpu4_export_structure[
                "status"
            ],
            "tracking_g1_urdf_in_memory_gpu4_export_structure_latest_blocker": (
                tracking_g1_urdf_in_memory_gpu4_export_structure["latest_blocker"]
            ),
            "tracking_g1_urdf_in_memory_gpu4_export_structure_export": {
                "size_bytes": tracking_g1_urdf_in_memory_gpu4_export_structure["export"]["size_bytes"],
                "counts": tracking_g1_urdf_in_memory_gpu4_export_structure["export"]["counts"],
                "action_joint_hit_count": tracking_g1_urdf_in_memory_gpu4_export_structure["export"][
                    "action_joint_hit_count"
                ],
                "action_joint_count_expected": tracking_g1_urdf_in_memory_gpu4_export_structure["export"][
                    "action_joint_count_expected"
                ],
                "target_body_hit_count": tracking_g1_urdf_in_memory_gpu4_export_structure["export"][
                    "target_body_hit_count"
                ],
                "target_body_count_checked": tracking_g1_urdf_in_memory_gpu4_export_structure["export"][
                    "target_body_count_checked"
                ],
            },
            "tracking_g1_urdf_in_memory_gpu4_export_structure_checks": (
                tracking_g1_urdf_in_memory_gpu4_export_structure["checks"]
            ),
            "tracking_g1_urdf_in_memory_gpu4_export_structure_json": str(
                ROOT
                / "res/tracking/g1_urdf_in_memory_gpu4_export_structure_audit/"
                "tracking_g1_urdf_in_memory_gpu4_export_structure_audit.json"
            ),
            "tracking_g1_preconverted_asset_audit_status": tracking_g1_preconverted_asset_audit["status"],
            "tracking_g1_preconverted_asset_audit_counts": {
                "candidate_count": tracking_g1_preconverted_asset_audit["candidate_count"],
                "usd_candidate_count": tracking_g1_preconverted_asset_audit["usd_candidate_count"],
                "official_mesh_usd_count": tracking_g1_preconverted_asset_audit["official_mesh_usd_count"],
                "official_full_robot_preconverted_g1_usd_count": tracking_g1_preconverted_asset_audit[
                    "official_full_robot_preconverted_g1_usd_count"
                ],
                "reference_g1_usd_count": tracking_g1_preconverted_asset_audit["reference_g1_usd_count"],
                "validated_reference_robotish_usd_count": tracking_g1_preconverted_asset_audit[
                    "validated_reference_robotish_usd_count"
                ],
            },
            "tracking_g1_preconverted_asset_audit_reference_usd": [
                {
                    "relative_path": row["relative_path"],
                    "has_robotish_stage": row["has_robotish_stage"],
                    "usable_as_official_beyondmimic_asset": row["usable_as_official_beyondmimic_asset"],
                    "payload_summary": {
                        key: row["probe"]["payload"].get(key)
                        for key in [
                            "stage_open_ok",
                            "default_prim_path",
                            "prim_count",
                            "joint_count",
                            "rigid_body_like_count",
                            "articulation_api_count",
                            "pelvis_path_present",
                            "torso_path_present",
                        ]
                    },
                }
                for row in tracking_g1_preconverted_asset_audit["kit_validated_reference_usd"]
            ],
            "tracking_g1_preconverted_asset_audit_json": str(
                ROOT / "res/tracking/g1_preconverted_asset_audit/tracking_g1_preconverted_asset_audit.json"
            ),
            "tracking_g1_reference_usd_compatibility_audit_status": tracking_g1_reference_usd_compatibility_audit[
                "status"
            ],
            "tracking_g1_reference_usd_compatibility_audit_compatible": tracking_g1_reference_usd_compatibility_audit[
                "compatible_for_resource_adjusted_replay"
            ],
            "tracking_g1_reference_usd_compatibility_audit_official_contract": tracking_g1_reference_usd_compatibility_audit[
                "official_contract"
            ],
            "tracking_g1_reference_usd_compatibility_audit_reference_contract": tracking_g1_reference_usd_compatibility_audit[
                "reference_contract"
            ],
            "tracking_g1_reference_usd_compatibility_audit_missing_action_joints": tracking_g1_reference_usd_compatibility_audit[
                "diffs"
            ][
                "official_action_joints_vs_reference_revolute_joints"
            ][
                "missing_from_right"
            ],
            "tracking_g1_reference_usd_compatibility_audit_missing_target_bodies": tracking_g1_reference_usd_compatibility_audit[
                "diffs"
            ][
                "official_target_bodies_vs_reference_links"
            ][
                "missing_from_right"
            ],
            "tracking_g1_reference_usd_compatibility_audit_json": str(
                ROOT
                / "res/tracking/g1_reference_usd_compatibility_audit/"
                "tracking_g1_reference_usd_compatibility_audit.json"
            ),
            "tracking_g1_official_urdf_skeleton_usd_audit_status": tracking_g1_official_urdf_skeleton_usd_audit[
                "status"
            ],
            "tracking_g1_official_urdf_skeleton_usd_contract_ok": tracking_g1_official_urdf_skeleton_usd_audit[
                "skeleton_contract_ok"
            ],
            "tracking_g1_official_urdf_skeleton_usd_official_contract": tracking_g1_official_urdf_skeleton_usd_audit[
                "official_contract"
            ],
            "tracking_g1_official_urdf_skeleton_usd_skeleton_contract": tracking_g1_official_urdf_skeleton_usd_audit[
                "skeleton_contract"
            ],
            "tracking_g1_official_urdf_skeleton_usd_checks": tracking_g1_official_urdf_skeleton_usd_audit["checks"],
            "tracking_g1_official_urdf_skeleton_usd_json": str(
                ROOT
                / "res/tracking/g1_official_urdf_skeleton_usd/"
                "tracking_g1_official_urdf_skeleton_usd_audit.json"
            ),
            "tracking_g1_official_urdf_skeleton_usd_path": tracking_g1_official_urdf_skeleton_usd_audit["usd_path"],
            "tracking_g1_urdf_physical_asset_contract_status": tracking_g1_urdf_physical_asset_contract_audit[
                "status"
            ],
            "tracking_g1_urdf_physical_asset_contract_metrics": tracking_g1_urdf_physical_asset_contract_audit[
                "metrics"
            ],
            "tracking_g1_urdf_physical_asset_contract_gaps": tracking_g1_urdf_physical_asset_contract_audit["gaps"],
            "tracking_g1_urdf_physical_asset_contract_checks": tracking_g1_urdf_physical_asset_contract_audit[
                "checks"
            ],
            "tracking_g1_urdf_physical_asset_contract_json": str(
                ROOT
                / "res/tracking/g1_urdf_physical_asset_contract_audit/"
                "tracking_g1_urdf_physical_asset_contract_audit.json"
            ),
            "tracking_g1_urdf_source_equivalence_status": tracking_g1_urdf_source_equivalence_audit["status"],
            "tracking_g1_urdf_source_equivalence_source_metrics": {
                key: value["metrics"]
                for key, value in tracking_g1_urdf_source_equivalence_audit["sources"].items()
            },
            "tracking_g1_urdf_source_equivalence_action_joint_summary": (
                tracking_g1_urdf_source_equivalence_audit["action_joint_summary"]
            ),
            "tracking_g1_urdf_source_equivalence_download_vs_wbt": (
                tracking_g1_urdf_source_equivalence_audit["comparisons"]["download_vs_whole_body_tracking"]
            ),
            "tracking_g1_urdf_source_equivalence_checks": tracking_g1_urdf_source_equivalence_audit["checks"],
            "tracking_g1_urdf_source_equivalence_json": str(
                ROOT
                / "res/tracking/g1_urdf_source_equivalence_audit/"
                "tracking_g1_urdf_source_equivalence_audit.json"
            ),
            "tracking_g1_resource_adjusted_enriched_usd_status": tracking_g1_resource_adjusted_enriched_usd_probe[
                "status"
            ],
            "tracking_g1_resource_adjusted_enriched_usd_readback": tracking_g1_resource_adjusted_enriched_usd_probe[
                "readback"
            ],
            "tracking_g1_resource_adjusted_enriched_usd_checks": tracking_g1_resource_adjusted_enriched_usd_probe[
                "checks"
            ],
            "tracking_g1_resource_adjusted_enriched_usd_json": str(
                ROOT
                / "res/tracking/g1_resource_adjusted_enriched_usd/"
                "tracking_g1_resource_adjusted_enriched_usd_probe.json"
            ),
            "tracking_g1_resource_adjusted_enriched_usd_path": tracking_g1_resource_adjusted_enriched_usd_probe[
                "outputs"
            ]["enriched_usd"],
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
                if gate["status"] in {"blocked", "needs_review"}
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
            "isaaclab_current_headless_gate": {
                "status": isaaclab_current_headless_gate["status"],
                "checks": isaaclab_current_headless_gate["checks"],
                "config": isaaclab_current_headless_gate["config"],
                "run": isaaclab_current_headless_gate["run"],
                "json": str(ROOT / "res/setup/isaaclab_current_headless_gate/isaaclab_current_headless_gate.json"),
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
        "tracking_csv_task_eval_gpu47_failed_rerun": {
            "status": tracking_csv_task_eval_gpu47_failed_rerun["status"],
            "returncode": tracking_csv_task_eval_gpu47_failed_rerun["returncode"],
            "target_physical_gpu": tracking_csv_task_eval_gpu47_failed_rerun["target_physical_gpu"],
            "markers": tracking_csv_task_eval_gpu47_failed_rerun["markers"],
            "claim_level": tracking_csv_task_eval_gpu47_failed_rerun["claim_level"],
            "json": str(
                ROOT / "res/failed_runs/tracking_g1_resource_adjusted_csv_task_eval_gpu47_20260619_124125/status.json"
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
        "english_reading_report": {
            "doc_path": str(ENGLISH_READING_REPORT_DOC),
            "final_path": str(ENGLISH_READING_REPORT_FINAL),
            "doc_exists": ENGLISH_READING_REPORT_DOC.is_file(),
            "final_exists": ENGLISH_READING_REPORT_FINAL.is_file(),
            "word_count": (
                len(ENGLISH_READING_REPORT_DOC.read_text(encoding="utf-8").split())
                if ENGLISH_READING_REPORT_DOC.is_file()
                else 0
            ),
            "contains_no_full_reproduction_claim": (
                "does not fully reproduce BeyondMimic at paper-level"
                in ENGLISH_READING_REPORT_DOC.read_text(encoding="utf-8")
                if ENGLISH_READING_REPORT_DOC.is_file()
                else False
            ),
            "mentions_official_loop_virtual_chain": (
                "official-loop tracking/PPO eval"
                in ENGLISH_READING_REPORT_DOC.read_text(encoding="utf-8")
                if ENGLISH_READING_REPORT_DOC.is_file()
                else False
            ),
        },
        "current_environment_and_reproduction_status": {
            "doc_path": str(CURRENT_STATUS_REPORT_DOC),
            "final_path": str(CURRENT_STATUS_REPORT_FINAL),
            "doc_exists": CURRENT_STATUS_REPORT_DOC.is_file(),
            "final_exists": CURRENT_STATUS_REPORT_FINAL.is_file(),
            "word_count": (
                len(CURRENT_STATUS_REPORT_DOC.read_text(encoding="utf-8").split())
                if CURRENT_STATUS_REPORT_DOC.is_file()
                else 0
            ),
            "contains_no_full_reproduction_boundary": (
                "cannot claim a full BeyondMimic reproduction"
                in CURRENT_STATUS_REPORT_DOC.read_text(encoding="utf-8")
                if CURRENT_STATUS_REPORT_DOC.is_file()
                else False
            ),
            "mentions_simulation_next_steps": (
                "What Can Still Be Verified In Simulation"
                in CURRENT_STATUS_REPORT_DOC.read_text(encoding="utf-8")
                if CURRENT_STATUS_REPORT_DOC.is_file()
                else False
            ),
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
        "visual_evidence_index": {
            "status": visual_evidence_index["status"],
            "metrics": visual_evidence_index["metrics"],
            "checks": visual_evidence_index["checks"],
            "json": str(ROOT / "res/report_assets/visual_evidence_index/visual_evidence_index.json"),
            "csv": str(ROOT / "res/report_assets/visual_evidence_index/visual_evidence_index.csv"),
            "md": str(ROOT / "res/report_assets/visual_evidence_index/visual_evidence_index.md"),
        },
        "guided_vs_unguided_closed_loop_matrix": {
            "status": guided_vs_unguided_closed_loop_matrix["status"],
            "claim_level": guided_vs_unguided_closed_loop_matrix["claim_level"],
            "metrics": guided_vs_unguided_closed_loop_matrix["metrics"],
            "checks": guided_vs_unguided_closed_loop_matrix["checks"],
            "interpretation": guided_vs_unguided_closed_loop_matrix["interpretation"],
            "json": str(
                ROOT
                / "res/report_assets/guided_vs_unguided_closed_loop_matrix/"
                "guided_vs_unguided_closed_loop_matrix.json"
            ),
            "csv": str(
                ROOT
                / "res/report_assets/guided_vs_unguided_closed_loop_matrix/"
                "guided_vs_unguided_closed_loop_matrix.csv"
            ),
            "aggregate_csv": str(
                ROOT
                / "res/report_assets/guided_vs_unguided_closed_loop_matrix/"
                "guided_vs_unguided_closed_loop_aggregate.csv"
            ),
            "md": str(
                ROOT
                / "res/report_assets/guided_vs_unguided_closed_loop_matrix/"
                "guided_vs_unguided_closed_loop_matrix.md"
            ),
        },
        "official_importer_export_fig5_fig6_proxy_protocol_matrix": {
            "status": official_importer_export_fig5_fig6_proxy_protocol_matrix["status"],
            "metrics": official_importer_export_fig5_fig6_proxy_protocol_matrix["metrics"],
            "checks": official_importer_export_fig5_fig6_proxy_protocol_matrix["checks"],
            "interpretation": official_importer_export_fig5_fig6_proxy_protocol_matrix["interpretation"],
            "json": str(
                ROOT
                / "res/report_assets/official_importer_export_fig5_fig6_proxy_protocol_matrix/"
                "fig5_fig6_proxy_protocol_matrix.json"
            ),
            "csv": str(
                ROOT
                / "res/report_assets/official_importer_export_fig5_fig6_proxy_protocol_matrix/"
                "fig5_fig6_proxy_protocol_matrix.csv"
            ),
            "markdown": str(
                ROOT
                / "res/report_assets/official_importer_export_fig5_fig6_proxy_protocol_matrix/"
                "fig5_fig6_proxy_protocol_matrix.md"
            ),
            "plot_png": str(
                ROOT
                / "res/report_assets/official_importer_export_fig5_fig6_proxy_protocol_matrix/"
                "fig5_fig6_proxy_protocol_rates.png"
            ),
        },
        "official_importer_export_fig5_fig6_task_protocol_proxy": {
            "status": official_importer_export_fig5_fig6_task_protocol_proxy["status"],
            "metrics": official_importer_export_fig5_fig6_task_protocol_proxy["metrics"],
            "aggregate": official_importer_export_fig5_fig6_task_protocol_proxy["aggregate"],
            "checks": official_importer_export_fig5_fig6_task_protocol_proxy["checks"],
            "thresholds": official_importer_export_fig5_fig6_task_protocol_proxy["thresholds"],
            "interpretation": official_importer_export_fig5_fig6_task_protocol_proxy["interpretation"],
            "json": str(
                ROOT
                / "res/report_assets/official_importer_export_fig5_fig6_task_protocol_proxy/"
                "fig5_fig6_task_protocol_proxy.json"
            ),
            "rows_csv": str(
                ROOT
                / "res/report_assets/official_importer_export_fig5_fig6_task_protocol_proxy/"
                "fig5_fig6_task_protocol_proxy_rows.csv"
            ),
            "aggregate_csv": str(
                ROOT
                / "res/report_assets/official_importer_export_fig5_fig6_task_protocol_proxy/"
                "fig5_fig6_task_protocol_proxy_aggregate.csv"
            ),
            "markdown": str(
                ROOT
                / "res/report_assets/official_importer_export_fig5_fig6_task_protocol_proxy/"
                "fig5_fig6_task_protocol_proxy.md"
            ),
            "rates_png": str(
                ROOT
                / "res/report_assets/official_importer_export_fig5_fig6_task_protocol_proxy/"
                "fig5_fig6_task_protocol_proxy_rates.png"
            ),
            "deltas_png": str(
                ROOT
                / "res/report_assets/official_importer_export_fig5_fig6_task_protocol_proxy/"
                "fig5_fig6_task_protocol_proxy_deltas.png"
            ),
        },
        "official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy": {
            "status": official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy["status"],
            "metrics": official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy["metrics"],
            "aggregate": official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy["aggregate"],
            "checks": official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy["checks"],
            "thresholds": official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy["thresholds"],
            "interpretation": official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy["interpretation"],
            "json": str(
                ROOT
                / "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy/"
                "fig5_fig6_task_protocol_proxy.json"
            ),
            "rows_csv": str(
                ROOT
                / "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy/"
                "fig5_fig6_task_protocol_proxy_rows.csv"
            ),
            "aggregate_csv": str(
                ROOT
                / "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy/"
                "fig5_fig6_task_protocol_proxy_aggregate.csv"
            ),
            "markdown": str(
                ROOT
                / "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy/"
                "fig5_fig6_task_protocol_proxy.md"
            ),
            "rates_png": str(
                ROOT
                / "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy/"
                "fig5_fig6_task_protocol_proxy_rates.png"
            ),
            "deltas_png": str(
                ROOT
                / "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy/"
                "fig5_fig6_task_protocol_proxy_deltas.png"
            ),
        },
        "official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy": {
            "status": official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy["status"],
            "metrics": official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy["metrics"],
            "aggregate": official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy["aggregate"],
            "checks": official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy["checks"],
            "thresholds": official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy["thresholds"],
            "interpretation": official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy[
                "interpretation"
            ],
            "json": str(
                ROOT
                / "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy/"
                "success_fall_collision_proxy.json"
            ),
            "rows_csv": str(
                ROOT
                / "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy/"
                "success_fall_collision_proxy_rows.csv"
            ),
            "aggregate_csv": str(
                ROOT
                / "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy/"
                "success_fall_collision_proxy_aggregate.csv"
            ),
            "markdown": str(
                ROOT
                / "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy/"
                "success_fall_collision_proxy.md"
            ),
            "rates_png": str(
                ROOT
                / "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy/"
                "success_fall_collision_proxy_rates.png"
            ),
        },
        "official_importer_export_full_bundle_latent_projection_report_assets": {
            "status": official_importer_export_full_bundle_latent_projection_report_assets["status"],
            "metrics": official_importer_export_full_bundle_latent_projection_report_assets["metrics"],
            "checks": official_importer_export_full_bundle_latent_projection_report_assets["checks"],
            "interpretation": official_importer_export_full_bundle_latent_projection_report_assets[
                "interpretation"
            ],
            "assets": official_importer_export_full_bundle_latent_projection_report_assets["assets"],
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
            "resource_adjusted_teacher_rollout_vae_training_status": (
                resource_adjusted_teacher_rollout_vae_training["status"]
            ),
            "resource_adjusted_teacher_rollout_vae_training_worker": (
                resource_adjusted_teacher_rollout_vae_training["worker_summary"]
            ),
            "resource_adjusted_teacher_rollout_vae_training_gpu_metrics": (
                resource_adjusted_teacher_rollout_vae_training.get("gpu_metrics_summary", {})
            ),
            "resource_adjusted_teacher_rollout_vae_training_checks": (
                resource_adjusted_teacher_rollout_vae_training["checks"]
            ),
            "resource_adjusted_teacher_rollout_vae_training_outputs": (
                resource_adjusted_teacher_rollout_vae_training["outputs"]
            ),
            "resource_adjusted_teacher_rollout_vae_training_json": str(
                ROOT
                / "res/level_c/resource_adjusted_teacher_rollout_vae_training/"
                / "level_c_resource_adjusted_teacher_rollout_vae_training.json"
            ),
            "official_csv_loop_teacher_rollout_vae_training_status": (
                official_csv_loop_teacher_rollout_vae_training["status"]
            ),
            "official_csv_loop_teacher_rollout_vae_training_worker": (
                official_csv_loop_teacher_rollout_vae_training["worker_summary"]
            ),
            "official_csv_loop_teacher_rollout_vae_training_gpu_metrics": (
                official_csv_loop_teacher_rollout_vae_training.get("gpu_metrics_summary", {})
            ),
            "official_csv_loop_teacher_rollout_vae_training_checks": (
                official_csv_loop_teacher_rollout_vae_training["checks"]
            ),
            "official_csv_loop_teacher_rollout_vae_training_outputs": (
                official_csv_loop_teacher_rollout_vae_training["outputs"]
            ),
            "official_csv_loop_teacher_rollout_vae_training_json": str(
                ROOT
                / "res/level_c/official_csv_loop_teacher_rollout_vae_training/"
                / "level_c_official_csv_loop_teacher_rollout_vae_training.json"
            ),
            "official_csv_loop_teacher_rollout_state_latent_dataset_status": (
                official_csv_loop_teacher_rollout_state_latent_dataset["status"]
            ),
            "official_csv_loop_teacher_rollout_state_latent_dataset_worker": (
                official_csv_loop_teacher_rollout_state_latent_dataset["worker_summary"]
            ),
            "official_csv_loop_teacher_rollout_state_latent_dataset_gpu_metrics": (
                official_csv_loop_teacher_rollout_state_latent_dataset.get("gpu_metrics_summary", {})
            ),
            "official_csv_loop_teacher_rollout_state_latent_dataset_checks": (
                official_csv_loop_teacher_rollout_state_latent_dataset["checks"]
            ),
            "official_csv_loop_teacher_rollout_state_latent_dataset_json": str(
                ROOT
                / "res/level_c/official_csv_loop_teacher_rollout_state_latent_dataset/"
                / "level_c_official_csv_loop_teacher_rollout_state_latent_dataset.json"
            ),
            "official_csv_loop_state_latent_diffusion_training_status": (
                official_csv_loop_state_latent_diffusion_training["status"]
            ),
            "official_csv_loop_state_latent_diffusion_training_worker": (
                official_csv_loop_state_latent_diffusion_training["worker_summary"]
            ),
            "official_csv_loop_state_latent_diffusion_training_gpu_metrics": (
                official_csv_loop_state_latent_diffusion_training.get("gpu_metrics_summary", {})
            ),
            "official_csv_loop_state_latent_diffusion_training_checks": (
                official_csv_loop_state_latent_diffusion_training["checks"]
            ),
            "official_csv_loop_state_latent_diffusion_training_json": str(
                ROOT
                / "res/level_c/official_csv_loop_state_latent_diffusion_training/"
                / "level_c_official_csv_loop_state_latent_diffusion_training.json"
            ),
            "official_csv_loop_full_bundle_teacher_rollout_vae_training_status": (
                official_csv_loop_full_bundle_teacher_rollout_vae_training["status"]
            ),
            "official_csv_loop_full_bundle_teacher_rollout_vae_training_worker": (
                official_csv_loop_full_bundle_teacher_rollout_vae_training["worker_summary"]
            ),
            "official_csv_loop_full_bundle_teacher_rollout_vae_training_gpu_metrics": (
                official_csv_loop_full_bundle_teacher_rollout_vae_training.get("gpu_metrics_summary", {})
            ),
            "official_csv_loop_full_bundle_teacher_rollout_vae_training_checks": (
                official_csv_loop_full_bundle_teacher_rollout_vae_training["checks"]
            ),
            "official_csv_loop_full_bundle_teacher_rollout_vae_training_json": str(
                ROOT
                / "res/level_c/official_csv_loop_full_bundle_teacher_rollout_vae_training/"
                / "level_c_official_csv_loop_full_bundle_teacher_rollout_vae_training.json"
            ),
            "official_importer_export_full_bundle_teacher_rollout_vae_training_status": (
                official_importer_export_full_bundle_teacher_rollout_vae_training["status"]
            ),
            "official_importer_export_full_bundle_teacher_rollout_vae_training_worker": (
                official_importer_export_full_bundle_teacher_rollout_vae_training["worker_summary"]
            ),
            "official_importer_export_full_bundle_teacher_rollout_vae_training_gpu_metrics": (
                official_importer_export_full_bundle_teacher_rollout_vae_training.get("gpu_metrics_summary", {})
            ),
            "official_importer_export_full_bundle_teacher_rollout_vae_training_checks": (
                official_importer_export_full_bundle_teacher_rollout_vae_training["checks"]
            ),
            "official_importer_export_full_bundle_teacher_rollout_vae_training_json": str(
                ROOT
                / "res/level_c/official_importer_export_full_bundle_teacher_rollout_vae_training/"
                / "level_c_official_importer_export_full_bundle_teacher_rollout_vae_training.json"
            ),
            "official_importer_export_scaled_ppo_teacher_rollout_vae_training_status": (
                official_importer_export_scaled_ppo_teacher_rollout_vae_training["status"]
            ),
            "official_importer_export_scaled_ppo_teacher_rollout_vae_training_worker": (
                official_importer_export_scaled_ppo_teacher_rollout_vae_training["worker_summary"]
            ),
            "official_importer_export_scaled_ppo_teacher_rollout_vae_training_gpu_metrics": (
                official_importer_export_scaled_ppo_teacher_rollout_vae_training.get("gpu_metrics_summary", {})
            ),
            "official_importer_export_scaled_ppo_teacher_rollout_vae_training_checks": (
                official_importer_export_scaled_ppo_teacher_rollout_vae_training["checks"]
            ),
            "official_importer_export_scaled_ppo_teacher_rollout_vae_training_json": str(
                ROOT
                / "res/level_c/official_importer_export_scaled_ppo_teacher_rollout_vae_training/"
                / "level_c_official_importer_export_scaled_ppo_teacher_rollout_vae_training.json"
            ),
            "official_importer_export_full_bundle_vae_closed_loop_rollout_eval_status": (
                official_importer_export_full_bundle_vae_closed_loop_rollout_eval["status"]
            ),
            "official_importer_export_full_bundle_vae_closed_loop_rollout_eval_config": (
                official_importer_export_full_bundle_vae_closed_loop_rollout_eval["config"]
            ),
            "official_importer_export_full_bundle_vae_closed_loop_rollout_eval_run": (
                official_importer_export_full_bundle_vae_closed_loop_rollout_eval["run"]
            ),
            "official_importer_export_full_bundle_vae_closed_loop_rollout_eval_checks": (
                official_importer_export_full_bundle_vae_closed_loop_rollout_eval["checks"]
            ),
            "official_importer_export_full_bundle_vae_closed_loop_rollout_assets": (
                official_importer_export_full_bundle_vae_closed_loop_rollout_assets
            ),
            "official_importer_export_full_bundle_vae_closed_loop_rollout_eval_json": str(
                ROOT
                / "res/level_c/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/"
                / "tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_eval.json"
            ),
            "official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset_status": (
                official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset["status"]
            ),
            "official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset_worker": (
                official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset["worker_summary"]
            ),
            "official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset_gpu_metrics": (
                official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset.get("gpu_metrics_summary", {})
            ),
            "official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset_checks": (
                official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset["checks"]
            ),
            "official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset_json": str(
                ROOT
                / "res/level_c/official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset/"
                / "level_c_official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset.json"
            ),
            "official_csv_loop_full_bundle_state_latent_diffusion_training_status": (
                official_csv_loop_full_bundle_state_latent_diffusion_training["status"]
            ),
            "official_csv_loop_full_bundle_state_latent_diffusion_training_worker": (
                official_csv_loop_full_bundle_state_latent_diffusion_training["worker_summary"]
            ),
            "official_csv_loop_full_bundle_state_latent_diffusion_training_gpu_metrics": (
                official_csv_loop_full_bundle_state_latent_diffusion_training.get("gpu_metrics_summary", {})
            ),
            "official_csv_loop_full_bundle_state_latent_diffusion_training_checks": (
                official_csv_loop_full_bundle_state_latent_diffusion_training["checks"]
            ),
            "official_csv_loop_full_bundle_state_latent_diffusion_training_json": str(
                ROOT
                / "res/level_c/official_csv_loop_full_bundle_state_latent_diffusion_training/"
                / "level_c_official_csv_loop_full_bundle_state_latent_diffusion_training.json"
            ),
            "official_csv_loop_full_bundle_downstream_assets": official_csv_loop_full_bundle_downstream_assets,
            "official_importer_export_full_bundle_vae_assets": official_importer_export_full_bundle_vae_assets,
            "official_importer_export_full_bundle_teacher_rollout_state_latent_dataset_status": (
                official_importer_export_full_bundle_teacher_rollout_state_latent_dataset["status"]
            ),
            "official_importer_export_full_bundle_teacher_rollout_state_latent_dataset_worker": (
                official_importer_export_full_bundle_teacher_rollout_state_latent_dataset["worker_summary"]
            ),
            "official_importer_export_full_bundle_teacher_rollout_state_latent_dataset_gpu_metrics": (
                official_importer_export_full_bundle_teacher_rollout_state_latent_dataset.get(
                    "gpu_metrics_summary", {}
                )
            ),
            "official_importer_export_full_bundle_teacher_rollout_state_latent_dataset_checks": (
                official_importer_export_full_bundle_teacher_rollout_state_latent_dataset["checks"]
            ),
            "official_importer_export_full_bundle_teacher_rollout_state_latent_dataset_json": str(
                ROOT
                / "res/level_c/official_importer_export_full_bundle_teacher_rollout_state_latent_dataset/"
                / "level_c_official_importer_export_full_bundle_teacher_rollout_state_latent_dataset.json"
            ),
            "official_importer_export_full_bundle_state_latent_diffusion_training_status": (
                official_importer_export_full_bundle_state_latent_diffusion_training["status"]
            ),
            "official_importer_export_full_bundle_state_latent_diffusion_training_worker": (
                official_importer_export_full_bundle_state_latent_diffusion_training["worker_summary"]
            ),
            "official_importer_export_full_bundle_state_latent_diffusion_training_gpu_metrics": (
                official_importer_export_full_bundle_state_latent_diffusion_training.get("gpu_metrics_summary", {})
            ),
            "official_importer_export_full_bundle_state_latent_diffusion_training_checks": (
                official_importer_export_full_bundle_state_latent_diffusion_training["checks"]
            ),
            "official_importer_export_full_bundle_state_latent_diffusion_training_json": str(
                ROOT
                / "res/level_c/official_importer_export_full_bundle_state_latent_diffusion_training/"
                / "level_c_official_importer_export_full_bundle_state_latent_diffusion_training.json"
            ),
            "official_importer_export_full_bundle_downstream_assets": (
                official_importer_export_full_bundle_downstream_assets
            ),
            "official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset_status": (
                official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset["status"]
            ),
            "official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset_worker": (
                official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset["worker_summary"]
            ),
            "official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset_gpu_metrics": (
                official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset.get(
                    "gpu_metrics_summary", {}
                )
            ),
            "official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset_checks": (
                official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset["checks"]
            ),
            "official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset_json": str(
                ROOT
                / "res/level_c/official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset/"
                / "level_c_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset.json"
            ),
            "official_importer_export_scaled_ppo_state_latent_diffusion_training_status": (
                official_importer_export_scaled_ppo_state_latent_diffusion_training["status"]
            ),
            "official_importer_export_scaled_ppo_state_latent_diffusion_training_worker": (
                official_importer_export_scaled_ppo_state_latent_diffusion_training["worker_summary"]
            ),
            "official_importer_export_scaled_ppo_state_latent_diffusion_training_gpu_metrics": (
                official_importer_export_scaled_ppo_state_latent_diffusion_training.get("gpu_metrics_summary", {})
            ),
            "official_importer_export_scaled_ppo_state_latent_diffusion_training_checks": (
                official_importer_export_scaled_ppo_state_latent_diffusion_training["checks"]
            ),
            "official_importer_export_scaled_ppo_state_latent_diffusion_training_json": str(
                ROOT
                / "res/level_c/official_importer_export_scaled_ppo_state_latent_diffusion_training/"
                / "level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training.json"
            ),
            "official_importer_export_scaled_ppo_downstream_assets": (
                official_importer_export_scaled_ppo_downstream_assets
            ),
            "official_csv_loop_state_latent_guidance_eval_status": (
                official_csv_loop_state_latent_guidance_eval["status"]
            ),
            "official_csv_loop_state_latent_guidance_eval_worker": (
                official_csv_loop_state_latent_guidance_eval["worker_summary"]
            ),
            "official_csv_loop_state_latent_guidance_eval_gpu_metrics": (
                official_csv_loop_state_latent_guidance_eval.get("gpu_metrics_summary", {})
            ),
            "official_csv_loop_state_latent_guidance_eval_checks": (
                official_csv_loop_state_latent_guidance_eval["checks"]
            ),
            "official_csv_loop_state_latent_guidance_eval_json": str(
                ROOT
                / "res/level_c/official_csv_loop_state_latent_guidance_eval/"
                / "level_c_official_csv_loop_state_latent_guidance_eval.json"
            ),
            "official_csv_loop_full_bundle_state_latent_guidance_eval_status": (
                official_csv_loop_full_bundle_state_latent_guidance_eval["status"]
            ),
            "official_csv_loop_full_bundle_state_latent_guidance_eval_worker": (
                official_csv_loop_full_bundle_state_latent_guidance_eval["worker_summary"]
            ),
            "official_csv_loop_full_bundle_state_latent_guidance_eval_gpu_metrics": (
                official_csv_loop_full_bundle_state_latent_guidance_eval.get("gpu_metrics_summary", {})
            ),
            "official_csv_loop_full_bundle_state_latent_guidance_eval_checks": (
                official_csv_loop_full_bundle_state_latent_guidance_eval["checks"]
            ),
            "official_csv_loop_full_bundle_state_latent_guidance_eval_json": str(
                ROOT
                / "res/level_c/official_csv_loop_full_bundle_state_latent_guidance_eval/"
                / "level_c_official_csv_loop_full_bundle_state_latent_guidance_eval.json"
            ),
            "official_csv_loop_full_bundle_guidance_assets": official_csv_loop_full_bundle_guidance_assets,
            "official_importer_export_full_bundle_state_latent_guidance_eval_status": (
                official_importer_export_full_bundle_state_latent_guidance_eval["status"]
            ),
            "official_importer_export_full_bundle_state_latent_guidance_eval_worker": (
                official_importer_export_full_bundle_state_latent_guidance_eval["worker_summary"]
            ),
            "official_importer_export_full_bundle_state_latent_guidance_eval_gpu_metrics": (
                official_importer_export_full_bundle_state_latent_guidance_eval.get("gpu_metrics_summary", {})
            ),
            "official_importer_export_full_bundle_state_latent_guidance_eval_checks": (
                official_importer_export_full_bundle_state_latent_guidance_eval["checks"]
            ),
            "official_importer_export_full_bundle_state_latent_guidance_eval_json": str(
                ROOT
                / "res/level_c/official_importer_export_full_bundle_state_latent_guidance_eval/"
                / "level_c_official_importer_export_full_bundle_state_latent_guidance_eval.json"
            ),
            "official_importer_export_scaled_ppo_state_latent_guidance_eval_status": (
                official_importer_export_scaled_ppo_state_latent_guidance_eval["status"]
            ),
            "official_importer_export_scaled_ppo_state_latent_guidance_eval_worker": (
                official_importer_export_scaled_ppo_state_latent_guidance_eval["worker_summary"]
            ),
            "official_importer_export_scaled_ppo_state_latent_guidance_eval_gpu_metrics": (
                official_importer_export_scaled_ppo_state_latent_guidance_eval.get("gpu_metrics_summary", {})
            ),
            "official_importer_export_scaled_ppo_state_latent_guidance_eval_checks": (
                official_importer_export_scaled_ppo_state_latent_guidance_eval["checks"]
            ),
            "official_importer_export_scaled_ppo_state_latent_guidance_eval_json": str(
                ROOT
                / "res/level_c/official_importer_export_scaled_ppo_state_latent_guidance_eval/"
                / "level_c_official_importer_export_scaled_ppo_state_latent_guidance_eval.json"
            ),
            "official_importer_export_scaled_ppo_guidance_assets": (
                official_importer_export_scaled_ppo_guidance_assets
            ),
            "official_csv_loop_guidance_vae_action_decode_eval_status": (
                official_csv_loop_guidance_vae_action_decode_eval["status"]
            ),
            "official_csv_loop_guidance_vae_action_decode_eval_worker": (
                official_csv_loop_guidance_vae_action_decode_eval["worker_summary"]
            ),
            "official_csv_loop_guidance_vae_action_decode_eval_gpu_metrics": (
                official_csv_loop_guidance_vae_action_decode_eval.get("gpu_metrics_summary", {})
            ),
            "official_csv_loop_guidance_vae_action_decode_eval_checks": (
                official_csv_loop_guidance_vae_action_decode_eval["checks"]
            ),
            "official_csv_loop_guidance_vae_action_decode_eval_json": str(
                ROOT
                / "res/level_c/official_csv_loop_guidance_vae_action_decode_eval/"
                / "level_c_official_csv_loop_guidance_vae_action_decode_eval.json"
            ),
            "official_csv_loop_guidance_vae_action_decode_assets": (
                official_csv_loop_guidance_vae_action_decode_assets
            ),
            "official_csv_loop_guided_action_rollout_probe_status": (
                official_csv_loop_guided_action_rollout_probe["status"]
            ),
            "official_csv_loop_guided_action_rollout_probe_config": (
                official_csv_loop_guided_action_rollout_probe["config"]
            ),
            "official_csv_loop_guided_action_rollout_probe_metrics": (
                official_csv_loop_guided_action_rollout_probe["metrics"]
            ),
            "official_csv_loop_guided_action_rollout_probe_checks": (
                official_csv_loop_guided_action_rollout_probe["checks"]
            ),
            "official_csv_loop_guided_action_rollout_probe_assets": (
                official_csv_loop_guided_action_rollout_probe_assets
            ),
            "official_csv_loop_guided_action_rollout_probe_json": str(
                ROOT
                / "res/level_c/official_csv_loop_guided_action_rollout_probe/"
                / "tracking_g1_official_csv_loop_guided_action_rollout_probe.json"
            ),
            "official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval_status": (
                official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval["status"]
            ),
            "official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval_bundle": (
                official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval["bundle"]
            ),
            "official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval_tasks": (
                official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval["tasks"]
            ),
            "official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval_rows": (
                official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval["rows"]
            ),
            "official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval_checks": (
                official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval["checks"]
            ),
            "official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval_outputs": (
                official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval["outputs"]
            ),
            "official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval_json": str(
                ROOT
                / "res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval/"
                / "level_c_official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval.json"
            ),
            "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval_status": (
                official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval["status"]
            ),
            "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval_bundle": (
                official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval["bundle"]
            ),
            "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval_tasks": (
                official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval["tasks"]
            ),
            "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval_rows": (
                official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval["rows"]
            ),
            "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval_checks": (
                official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval["checks"]
            ),
            "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval_inputs": (
                official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval["inputs"]
            ),
            "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval_json": str(
                ROOT
                / "res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval/"
                / "level_c_official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval.json"
            ),
            "official_importer_export_scaled_ppo_task_conditioned_guidance_summary_assets": (
                official_importer_export_scaled_ppo_task_conditioned_guidance_summary_assets
            ),
            "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval_status": (
                official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval["status"]
            ),
            "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval_bundle": (
                official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval["bundle"]
            ),
            "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval_metrics": (
                official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval["metrics"]
            ),
            "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval_checks": (
                official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval["checks"]
            ),
            "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval_aggregate": (
                official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval["aggregate"]
            ),
            "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval_json": str(
                ROOT
                / "res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval/"
                / "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json"
            ),
            "official_importer_export_full_bundle_task_conditioned_guidance_multiseed_assets": (
                official_importer_export_full_bundle_task_conditioned_guidance_multiseed_assets
            ),
            "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval_status": (
                official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval["status"]
            ),
            "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval_bundle": (
                official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval["bundle"]
            ),
            "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval_metrics": (
                official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval["metrics"]
            ),
            "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval_checks": (
                official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval["checks"]
            ),
            "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval_aggregate": (
                official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval["aggregate"]
            ),
            "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval_inputs": (
                official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval["inputs"]
            ),
            "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval_json": str(
                ROOT
                / "res/level_c/official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval/"
                / "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval.json"
            ),
            "official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed_assets": (
                official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed_assets
            ),
            "official_importer_export_full_bundle_task_conditioned_guidance_success_boundary": (
                official_importer_export_full_bundle_task_conditioned_guidance_success_boundary
            ),
            "official_importer_export_full_bundle_inpainting_guidance_rollout_eval": (
                official_importer_export_full_bundle_inpainting_guidance_rollout_eval
            ),
            "official_importer_export_full_bundle_transition_guidance_rollout_eval": (
                official_importer_export_full_bundle_transition_guidance_rollout_eval
            ),
            "official_importer_export_full_bundle_guidance_video_contact_sheet": (
                official_importer_export_full_bundle_guidance_video_contact_sheet
            ),
            "official_csv_loop_action_guidance_rollout_eval_status": (
                official_csv_loop_action_guidance_rollout_eval["status"]
            ),
            "official_csv_loop_action_guidance_rollout_eval_config": (
                official_csv_loop_action_guidance_rollout_eval["config"]
            ),
            "official_csv_loop_action_guidance_rollout_eval_metrics": (
                official_csv_loop_action_guidance_rollout_eval["metrics"]
            ),
            "official_csv_loop_action_guidance_rollout_eval_checks": (
                official_csv_loop_action_guidance_rollout_eval["checks"]
            ),
            "official_csv_loop_action_guidance_rollout_asset": (
                official_csv_loop_action_guidance_rollout_asset
            ),
            "official_csv_loop_action_guidance_rollout_eval_json": str(
                ROOT
                / "res/level_c/official_csv_loop_action_guidance_rollout_eval/"
                / "level_c_official_csv_loop_action_guidance_rollout_eval.json"
            ),
            "official_csv_loop_receding_latent_guidance_rollout_eval_status": (
                official_csv_loop_receding_latent_guidance_rollout_eval["status"]
            ),
            "official_csv_loop_receding_latent_guidance_rollout_eval_config": (
                official_csv_loop_receding_latent_guidance_rollout_eval["config"]
            ),
            "official_csv_loop_receding_latent_guidance_rollout_eval_metrics": (
                official_csv_loop_receding_latent_guidance_rollout_eval["metrics"]
            ),
            "official_csv_loop_receding_latent_guidance_rollout_eval_checks": (
                official_csv_loop_receding_latent_guidance_rollout_eval["checks"]
            ),
            "official_csv_loop_receding_latent_guidance_rollout_asset": (
                official_csv_loop_receding_latent_guidance_rollout_asset
            ),
            "official_csv_loop_receding_latent_guidance_rollout_eval_json": str(
                ROOT
                / "res/level_c/official_csv_loop_receding_latent_guidance_rollout_eval/"
                / "level_c_official_csv_loop_receding_latent_guidance_rollout_eval.json"
            ),
            "official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval_status": (
                official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval["status"]
            ),
            "official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval_bundle": (
                official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval["bundle"]
            ),
            "official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval_config": (
                official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval["config"]
            ),
            "official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval_metrics": (
                official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval["metrics"]
            ),
            "official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval_checks": (
                official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval["checks"]
            ),
            "official_csv_loop_full_bundle_receding_latent_guidance_rollout_asset": (
                official_csv_loop_full_bundle_receding_latent_guidance_rollout_asset
            ),
            "official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval_json": str(
                ROOT
                / "res/level_c/official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval/"
                / "level_c_official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval.json"
            ),
            "official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval_status": (
                official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval["status"]
            ),
            "official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval_bundle": (
                official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval["bundle"]
            ),
            "official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval_rows": (
                official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval["rows"]
            ),
            "official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval_checks": (
                official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval["checks"]
            ),
            "official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval_json": str(
                ROOT
                / "res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval/"
                / "level_c_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval.json"
            ),
            "official_csv_loop_full_bundle_task_conditioned_guidance_summary_assets": (
                official_csv_loop_full_bundle_task_conditioned_guidance_summary_assets
            ),
            "official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval_status": (
                official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval["status"]
            ),
            "official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval_bundle": (
                official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval["bundle"]
            ),
            "official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval_metrics": (
                official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval["metrics"]
            ),
            "official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval_checks": (
                official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval["checks"]
            ),
            "official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval_aggregate": (
                official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval["aggregate"]
            ),
            "official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval_json": str(
                ROOT
                / "res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval/"
                / "official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json"
            ),
            "official_csv_loop_full_bundle_task_conditioned_guidance_multiseed_assets": (
                official_csv_loop_full_bundle_task_conditioned_guidance_multiseed_assets
            ),
            "official_csv_loop_full_bundle_task_conditioned_guidance_success_boundary": (
                official_csv_loop_full_bundle_task_conditioned_guidance_success_boundary
            ),
            "official_csv_loop_full_bundle_guidance_video_contact_sheet": (
                official_csv_loop_full_bundle_guidance_video_contact_sheet
            ),
            "official_csv_loop_task_conditioned_latent_guidance_rollout_eval_status": (
                official_csv_loop_task_conditioned_latent_guidance_rollout_eval["status"]
            ),
            "official_csv_loop_task_conditioned_latent_guidance_rollout_eval_rows": (
                official_csv_loop_task_conditioned_latent_guidance_rollout_eval["rows"]
            ),
            "official_csv_loop_task_conditioned_latent_guidance_rollout_eval_checks": (
                official_csv_loop_task_conditioned_latent_guidance_rollout_eval["checks"]
            ),
            "official_csv_loop_task_conditioned_latent_guidance_rollout_eval_json": str(
                ROOT
                / "res/level_c/official_csv_loop_task_conditioned_latent_guidance_rollout_eval/"
                / "level_c_official_csv_loop_task_conditioned_latent_guidance_rollout_eval.json"
            ),
            "official_csv_loop_task_conditioned_guidance_summary_assets": (
                official_csv_loop_task_conditioned_guidance_summary_assets
            ),
            "official_csv_loop_task_conditioned_latent_guidance_multiseed_eval_status": (
                official_csv_loop_task_conditioned_latent_guidance_multiseed_eval["status"]
            ),
            "official_csv_loop_task_conditioned_latent_guidance_multiseed_eval_metrics": (
                official_csv_loop_task_conditioned_latent_guidance_multiseed_eval["metrics"]
            ),
            "official_csv_loop_task_conditioned_latent_guidance_multiseed_eval_checks": (
                official_csv_loop_task_conditioned_latent_guidance_multiseed_eval["checks"]
            ),
            "official_csv_loop_task_conditioned_latent_guidance_multiseed_eval_aggregate": (
                official_csv_loop_task_conditioned_latent_guidance_multiseed_eval["aggregate"]
            ),
            "official_csv_loop_task_conditioned_latent_guidance_multiseed_eval_json": str(
                ROOT
                / "res/level_c/official_csv_loop_task_conditioned_latent_guidance_multiseed_eval/"
                / "official_csv_loop_task_conditioned_latent_guidance_multiseed_eval.json"
            ),
            "official_csv_loop_task_conditioned_guidance_multiseed_assets": (
                official_csv_loop_task_conditioned_guidance_multiseed_assets
            ),
            "official_csv_loop_vae_closed_loop_rollout_eval_status": (
                official_csv_loop_vae_closed_loop_rollout_eval["status"]
            ),
            "official_csv_loop_vae_closed_loop_rollout_eval_config": (
                official_csv_loop_vae_closed_loop_rollout_eval["config"]
            ),
            "official_csv_loop_vae_closed_loop_rollout_eval_run": (
                official_csv_loop_vae_closed_loop_rollout_eval["run"]
            ),
            "official_csv_loop_vae_closed_loop_rollout_eval_checks": (
                official_csv_loop_vae_closed_loop_rollout_eval["checks"]
            ),
            "official_csv_loop_vae_closed_loop_rollout_assets": (
                official_csv_loop_vae_closed_loop_rollout_assets
            ),
            "official_csv_loop_vae_closed_loop_rollout_eval_json": str(
                ROOT
                / "res/level_c/official_csv_loop_vae_closed_loop_rollout_eval/"
                / "tracking_g1_official_csv_loop_vae_closed_loop_rollout_eval.json"
            ),
            "official_csv_loop_vae_denoiser_onnx_async_status": (
                official_csv_loop_vae_denoiser_onnx_async_audit["status"]
            ),
            "official_csv_loop_vae_denoiser_onnx_async_settings": (
                official_csv_loop_vae_denoiser_onnx_async_audit["settings"]
            ),
            "official_csv_loop_vae_denoiser_onnx_async_consistency": (
                official_csv_loop_vae_denoiser_onnx_async_audit["consistency"]
            ),
            "official_csv_loop_vae_denoiser_onnx_async_summary": (
                official_csv_loop_vae_denoiser_onnx_async_audit["async_summary"]
            ),
            "official_csv_loop_vae_denoiser_onnx_async_checks": (
                official_csv_loop_vae_denoiser_onnx_async_audit["checks"]
            ),
            "official_csv_loop_vae_denoiser_onnx_async_outputs": (
                official_csv_loop_vae_denoiser_onnx_async_audit["outputs"]
            ),
            "official_csv_loop_vae_denoiser_onnx_async_json": str(
                ROOT
                / "res/level_c/official_csv_loop_vae_denoiser_onnx_async/"
                / "level_c_official_csv_loop_vae_denoiser_onnx_async_audit.json"
            ),
            "official_csv_loop_full_bundle_vae_denoiser_onnx_async_status": (
                official_csv_loop_full_bundle_vae_denoiser_onnx_async_audit["status"]
            ),
            "official_csv_loop_full_bundle_vae_denoiser_onnx_async_settings": (
                official_csv_loop_full_bundle_vae_denoiser_onnx_async_audit["settings"]
            ),
            "official_csv_loop_full_bundle_vae_denoiser_onnx_async_consistency": (
                official_csv_loop_full_bundle_vae_denoiser_onnx_async_audit["consistency"]
            ),
            "official_csv_loop_full_bundle_vae_denoiser_onnx_async_summary": (
                official_csv_loop_full_bundle_vae_denoiser_onnx_async_audit["async_summary"]
            ),
            "official_csv_loop_full_bundle_vae_denoiser_onnx_async_checks": (
                official_csv_loop_full_bundle_vae_denoiser_onnx_async_audit["checks"]
            ),
            "official_csv_loop_full_bundle_vae_denoiser_onnx_async_outputs": (
                official_csv_loop_full_bundle_vae_denoiser_onnx_async_audit["outputs"]
            ),
            "official_csv_loop_full_bundle_vae_denoiser_onnx_async_json": str(
                ROOT
                / "res/level_c/official_csv_loop_full_bundle_vae_denoiser_onnx_async/"
                / "level_c_official_csv_loop_full_bundle_vae_denoiser_onnx_async_audit.json"
            ),
            "official_importer_export_full_bundle_vae_denoiser_onnx_async_status": (
                official_importer_export_full_bundle_vae_denoiser_onnx_async_audit["status"]
            ),
            "official_importer_export_full_bundle_vae_denoiser_onnx_async_settings": (
                official_importer_export_full_bundle_vae_denoiser_onnx_async_audit["settings"]
            ),
            "official_importer_export_full_bundle_vae_denoiser_onnx_async_consistency": (
                official_importer_export_full_bundle_vae_denoiser_onnx_async_audit["consistency"]
            ),
            "official_importer_export_full_bundle_vae_denoiser_onnx_async_summary": (
                official_importer_export_full_bundle_vae_denoiser_onnx_async_audit["async_summary"]
            ),
            "official_importer_export_full_bundle_vae_denoiser_onnx_async_checks": (
                official_importer_export_full_bundle_vae_denoiser_onnx_async_audit["checks"]
            ),
            "official_importer_export_full_bundle_vae_denoiser_onnx_async_outputs": (
                official_importer_export_full_bundle_vae_denoiser_onnx_async_audit["outputs"]
            ),
            "official_importer_export_full_bundle_vae_denoiser_onnx_async_json": str(
                ROOT
                / "res/level_c/official_importer_export_full_bundle_vae_denoiser_onnx_async/"
                / "level_c_official_importer_export_full_bundle_vae_denoiser_onnx_async_audit.json"
            ),
            "official_importer_export_scaled_ppo_vae_denoiser_onnx_async_status": (
                official_importer_export_scaled_ppo_vae_denoiser_onnx_async_audit["status"]
            ),
            "official_importer_export_scaled_ppo_vae_denoiser_onnx_async_settings": (
                official_importer_export_scaled_ppo_vae_denoiser_onnx_async_audit["settings"]
            ),
            "official_importer_export_scaled_ppo_vae_denoiser_onnx_async_consistency": (
                official_importer_export_scaled_ppo_vae_denoiser_onnx_async_audit["consistency"]
            ),
            "official_importer_export_scaled_ppo_vae_denoiser_onnx_async_summary": (
                official_importer_export_scaled_ppo_vae_denoiser_onnx_async_audit["async_summary"]
            ),
            "official_importer_export_scaled_ppo_vae_denoiser_onnx_async_checks": (
                official_importer_export_scaled_ppo_vae_denoiser_onnx_async_audit["checks"]
            ),
            "official_importer_export_scaled_ppo_vae_denoiser_onnx_async_outputs": (
                official_importer_export_scaled_ppo_vae_denoiser_onnx_async_audit["outputs"]
            ),
            "official_importer_export_scaled_ppo_vae_denoiser_onnx_async_json": str(
                ROOT
                / "res/level_c/official_importer_export_scaled_ppo_vae_denoiser_onnx_async/"
                / "level_c_official_importer_export_scaled_ppo_vae_denoiser_onnx_async_audit.json"
            ),
            "resource_adjusted_teacher_rollout_state_latent_dataset_status": (
                resource_adjusted_teacher_rollout_state_latent_dataset["status"]
            ),
            "resource_adjusted_teacher_rollout_state_latent_dataset_worker": (
                resource_adjusted_teacher_rollout_state_latent_dataset["worker_summary"]
            ),
            "resource_adjusted_teacher_rollout_state_latent_dataset_gpu_metrics": (
                resource_adjusted_teacher_rollout_state_latent_dataset.get("gpu_metrics_summary", {})
            ),
            "resource_adjusted_teacher_rollout_state_latent_dataset_checks": (
                resource_adjusted_teacher_rollout_state_latent_dataset["checks"]
            ),
            "resource_adjusted_teacher_rollout_state_latent_dataset_json": str(
                ROOT
                / "res/level_c/resource_adjusted_teacher_rollout_state_latent_dataset/"
                / "level_c_resource_adjusted_teacher_rollout_state_latent_dataset.json"
            ),
            "resource_adjusted_state_latent_diffusion_training_status": (
                resource_adjusted_state_latent_diffusion_training["status"]
            ),
            "resource_adjusted_state_latent_diffusion_training_worker": (
                resource_adjusted_state_latent_diffusion_training["worker_summary"]
            ),
            "resource_adjusted_state_latent_diffusion_training_gpu_metrics": (
                resource_adjusted_state_latent_diffusion_training.get("gpu_metrics_summary", {})
            ),
            "resource_adjusted_state_latent_diffusion_training_checks": (
                resource_adjusted_state_latent_diffusion_training["checks"]
            ),
            "resource_adjusted_state_latent_diffusion_training_json": str(
                ROOT
                / "res/level_c/resource_adjusted_state_latent_diffusion_training/"
                / "level_c_resource_adjusted_state_latent_diffusion_training.json"
            ),
            "resource_adjusted_state_latent_guidance_eval_status": (
                resource_adjusted_state_latent_guidance_eval["status"]
            ),
            "resource_adjusted_state_latent_guidance_eval_worker": (
                resource_adjusted_state_latent_guidance_eval["worker_summary"]
            ),
            "resource_adjusted_state_latent_guidance_eval_gpu_metrics": (
                resource_adjusted_state_latent_guidance_eval.get("gpu_metrics_summary", {})
            ),
            "resource_adjusted_state_latent_guidance_eval_checks": (
                resource_adjusted_state_latent_guidance_eval["checks"]
            ),
            "resource_adjusted_state_latent_guidance_eval_json": str(
                ROOT
                / "res/level_c/resource_adjusted_state_latent_guidance_eval/"
                / "level_c_resource_adjusted_state_latent_guidance_eval.json"
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
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/visual_evidence_index.py'}",
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
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/level_c_resource_adjusted_state_latent_guidance_eval.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/official_importer_export_full_bundle_task_conditioned_guidance_success_boundary.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/official_importer_export_scaled_ppo_checkpoint_completion_proxy.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_checkpoint_sweep.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_best_checkpoint_confirmation_eval.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/official_importer_export_scaled_ppo_reward_termination_diagnostic.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/official_importer_export_scaled_ppo_ee_body_pos_termination_source_audit.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_endpoint_z_error_trace.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/official_importer_export_fig5_fig6_task_protocol_proxy.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy.py'}",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_official_importer_export_full_bundle_vae_denoiser_onnx_async_audit.py'}",
            f"{ROOT / 'envs/bm_diffusion/bin/python'} {ROOT / 'reproduction/scripts/level_c_official_importer_export_scaled_ppo_vae_denoiser_onnx_async_audit.py'}",
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
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/tracking_official_replay_conversion_audit.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/tracking_official_replay_npz_entry_diagnostic_audit.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/tracking_official_csv_to_npz_loop_with_enriched_usd_audit.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/tracking_official_replay_npz_loop_with_enriched_usd_audit.py'}",
            f"{ROOT / 'envs/bm_tracking/bin/python'} {ROOT / 'reproduction/scripts/tracking_urdf_conversion_probe.py'}",
            f"{ROOT / 'envs/bm_tracking/bin/python'} {ROOT / 'reproduction/scripts/tracking_urdf_path_tiny_probe.py'}",
            f"{ROOT / 'envs/bm_tracking/bin/python'} {ROOT / 'reproduction/scripts/tracking_mjcf_stage_probe.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/tracking_usd_save_policy_probe.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/tracking_simulationapp_save_policy_probe.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/tracking_usd_api_variant_probe.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/tracking_g1_urdf_stage_export_workaround_probe.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/tracking_g1_urdf_layer_save_workaround_probe.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/tracking_g1_urdf_in_memory_import_probe.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/tracking_g1_urdf_simulationapp_in_memory_import_probe.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/tracking_g1_urdf_in_memory_variant_matrix_probe.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/tracking_g1_preconverted_asset_audit.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/tracking_g1_reference_usd_compatibility_audit.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/tracking_g1_official_urdf_skeleton_usd_audit.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/tracking_g1_urdf_physical_asset_contract_audit.py'}",
            f"{ROOT / 'envs/bm_analysis/bin/python'} {ROOT / 'reproduction/scripts/tracking_g1_resource_adjusted_enriched_usd_probe.py'}",
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
        f"- Current IsaacLab headless AppLauncher gate: `{env['isaaclab_current_headless_gate']['status']}`; "
        f"config `{json.dumps(env['isaaclab_current_headless_gate']['config'], sort_keys=True)}`; checks "
        f"`{json.dumps(env['isaaclab_current_headless_gate']['checks'], sort_keys=True)}`. "
        "This confirms the current headless startup sentinel on physical GPU 4 without claiming official replay, PPO, "
        "DAgger, Fig. 5/Fig. 6, or real-robot completion."
    )
    failed_gpu47 = summary["tracking_csv_task_eval_gpu47_failed_rerun"]
    lines.append(
        f"- Current GPU4/7 resource-adjusted CSV task-eval rerun: `{failed_gpu47['status']}`; "
        f"target GPU `{failed_gpu47['target_physical_gpu']}`; return code `{failed_gpu47['returncode']}`; "
        f"markers `{json.dumps(failed_gpu47['markers'], sort_keys=True)}`. "
        "This candidate rerun reached AppLauncher, environment creation, and reset, then was killed before 299-step "
        "metrics were written. It is retained as a failed current-GPU rerun and does not overwrite the earlier "
        "canonical successful GPU6 resource-adjusted 299-step task-eval artifact."
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
    reading_report = summary["english_reading_report"]
    lines.append(
        f"- English reading report draft: doc exists `{reading_report['doc_exists']}`, final copy exists "
        f"`{reading_report['final_exists']}`, word count `{reading_report['word_count']}`; "
        f"no-full-reproduction boundary `{reading_report['contains_no_full_reproduction_claim']}`, "
        f"official-loop virtual-chain evidence `{reading_report['mentions_official_loop_virtual_chain']}`."
    )
    visual_media = summary["visual_media_inventory"]
    lines.append(
        f"- Visual media inventory: `{visual_media['status']}`; `{visual_media['row_count']}` media files, "
        f"kind counts `{json.dumps(visual_media['kind_counts'], sort_keys=True)}`, category counts "
        f"`{json.dumps(visual_media['category_counts'], sort_keys=True)}`; paper-required rollout/robot videos remain absent."
    )
    visual_index = summary["visual_evidence_index"]
    lines.append(
        f"- Visual evidence index: `{visual_index['status']}`; metrics "
        f"`{json.dumps(visual_index['metrics'], sort_keys=True)}`; checks "
        f"`{json.dumps(visual_index['checks'], sort_keys=True)}`. "
        "This records report/PPT-ready MP4, PNG, table, and README assets while explicitly keeping large videos out "
        "of GitHub and avoiding paper-level or real-robot claims."
    )
    guided_matrix = summary["guided_vs_unguided_closed_loop_matrix"]
    lines.append(
        f"- Guided-vs-unguided closed-loop report matrix: `{guided_matrix['status']}`; metrics "
        f"`{json.dumps(guided_matrix['metrics'], sort_keys=True)}`; claim level "
        f"`{guided_matrix['claim_level']}`. This aggregates existing local virtual action-guidance, receding-latent, "
        "and task-conditioned guidance rollouts into CSV/JSON/PNG report assets, while explicitly keeping the "
        "evidence below official Fig. 5/Fig. 6 and real-robot claims."
    )
    success_boundary = summary["level_c_diffusion"][
        "official_csv_loop_full_bundle_task_conditioned_guidance_success_boundary"
    ]
    lines.append(
        f"- Full-bundle task-conditioned guidance local proxy success boundary: "
        f"`{success_boundary['status']}`; metrics `{json.dumps(success_boundary['metrics'], sort_keys=True)}`; "
        f"assets `{json.dumps(success_boundary['assets'], sort_keys=True)}`. "
        "This converts the 20 local closed-loop full-bundle guidance videos into report-facing proxy completion and "
        "guided-vs-denoised improvement rates. It is useful for the reading report/PPT, but it is not the official "
        "BeyondMimic Fig. 5/Fig. 6 success/fall/collision protocol."
    )
    video_contact_sheet = summary["level_c_diffusion"][
        "official_csv_loop_full_bundle_guidance_video_contact_sheet"
    ]
    lines.append(
        f"- Full-bundle task-conditioned guidance video contact sheet: "
        f"`{video_contact_sheet['status']}`; metrics "
        f"`{json.dumps(video_contact_sheet['metrics'], sort_keys=True)}`; assets "
        f"`{json.dumps(video_contact_sheet['assets'], sort_keys=True)}`. "
        "This indexes 20 local MP4 rollouts with SHA256 hashes and builds a compact keyframe contact sheet for the "
        "English reading report/PPT. The MP4 files remain local and are not committed to GitHub; the claim level is "
        "local virtual/resource-adjusted evidence, not official Fig. 5/Fig. 6, TensorRT, or real-robot evidence."
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
    official_entry_markers = summary["level_b_tracking"]["tracking_official_replay_npz_entry_diagnostic_markers"]
    official_entry_summary = {
        "app_launcher_constructed": summary["level_b_tracking"][
            "tracking_official_replay_npz_entry_diagnostic_checks"
        ]["app_launcher_constructed"],
        "blocked_before_artifact_download": summary["level_b_tracking"][
            "tracking_official_replay_npz_entry_diagnostic_checks"
        ]["fake_wandb_download_seen"]
        is False,
        "failed_to_save_layer": official_entry_markers["failed_to_save_layer"],
        "empty_robot_after_converter": official_entry_markers["empty_robot_after_converter"],
    }
    lines.append(
        f"- Level B official `replay_npz.py` entry diagnostic: "
        f"`{summary['level_b_tracking']['tracking_official_replay_npz_entry_diagnostic_status']}`; "
        f"latest blocker "
        f"`{summary['level_b_tracking']['tracking_official_replay_npz_entry_diagnostic_latest_blocker']}`; "
        f"summary `{json.dumps(official_entry_summary, sort_keys=True)}`. "
        "This runs the official replay entrypoint with a local fake-WandB artifact and bounded AppLauncher wrapper "
        "without modifying the official worktree. It reaches AppLauncher but blocks in the official URDF converter "
        "layer-save path before artifact download or replay-loop execution, leaving an empty robot prim. This is "
        "retained failure evidence, not official replay success or paper-level tracking."
    )
    official_csv_markers = summary["level_b_tracking"][
        "tracking_official_csv_to_npz_loop_with_enriched_usd_markers"
    ]
    official_csv_metrics = summary["level_b_tracking"][
        "tracking_official_csv_to_npz_loop_with_enriched_usd_metrics"
    ]
    official_csv_summary = {
        "app_launcher_constructed": summary["level_b_tracking"][
            "tracking_official_csv_to_npz_loop_with_enriched_usd_checks"
        ]["app_launcher_constructed"],
        "g1_cfg_patched_to_enriched_usd": summary["level_b_tracking"][
            "tracking_official_csv_to_npz_loop_with_enriched_usd_checks"
        ]["g1_cfg_patched_to_enriched_usd"],
        "motion_loaded": summary["level_b_tracking"][
            "tracking_official_csv_to_npz_loop_with_enriched_usd_checks"
        ]["motion_loaded"],
        "official_loop_call_299_seen": summary["level_b_tracking"][
            "tracking_official_csv_to_npz_loop_with_enriched_usd_checks"
        ]["official_loop_call_299_seen"],
        "np_savez_redirect_seen": summary["level_b_tracking"][
            "tracking_official_csv_to_npz_loop_with_enriched_usd_checks"
        ]["np_savez_redirect_seen"],
        "fake_wandb_log_artifact_seen": summary["level_b_tracking"][
            "tracking_official_csv_to_npz_loop_with_enriched_usd_checks"
        ]["fake_wandb_log_artifact_seen"],
        "joint_pos_shape": official_csv_metrics.get("joint_pos_shape"),
        "body_pos_w_shape": official_csv_metrics.get("body_pos_w_shape"),
        "simulation_app_close_called": official_csv_markers["simulation_app_close_called"],
    }
    lines.append(
        f"- Level B official `csv_to_npz.py` loop with enriched-USD runtime patch: "
        f"`{summary['level_b_tracking']['tracking_official_csv_to_npz_loop_with_enriched_usd_status']}`; "
        f"latest blocker "
        f"`{summary['level_b_tracking']['tracking_official_csv_to_npz_loop_with_enriched_usd_latest_blocker']}`; "
        f"summary `{json.dumps(official_csv_summary, sort_keys=True)}`. "
        "This executes the official csv_to_npz loop body to the 299-step bound, redirects the script's hard-coded "
        "`/tmp/motion.npz` output into the project result directory, and replaces wandb with a local fake registry. "
        "It remains resource-adjusted because the G1 config is patched in memory to use the validated enriched USD; "
        "therefore it is not unpatched official converter output and not paper-level replay/evaluation."
    )
    full_csv = summary["level_b_tracking"][
        "tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_aggregate"
    ]
    lines.append(
        f"- Level B full public-motion official `csv_to_npz.py` loop coverage: "
        f"`{summary['level_b_tracking']['tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_status']}`; "
        f"converted `{full_csv['ok_count']}/{full_csv['row_count']}` local G1 LAFAN CSV motions with "
        f"`{full_csv['failed_count']}` failures, `{full_csv['total_frames']}` total 50 Hz frames, and "
        f"`{full_csv['total_joint_values']}` joint values. This moves beyond the previous single-motion gate by "
        "covering the full local public motion bundle through the official loop body. It is still resource-adjusted "
        "because the same enriched-USD runtime patch is used, and it is not policy replay, PPO evaluation, or a "
        "paper-level tracking result."
    )
    full_csv_importer = summary["level_b_tracking"][
        "tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_aggregate"
    ]
    lines.append(
        f"- Level B full public-motion official `csv_to_npz.py` loop on captured official-importer-export G1 USDA: "
        f"`{summary['level_b_tracking']['tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_status']}`; "
        f"converted `{full_csv_importer['ok_count']}/{full_csv_importer['row_count']}` local G1 LAFAN CSV motions "
        f"with `{full_csv_importer['failed_count']}` failures, `{full_csv_importer['total_frames']}` total 50 Hz "
        f"frames, `{full_csv_importer['total_joint_values']}` joint values, and "
        f"`{full_csv_importer['total_npz_bytes']}` bytes of project-local NPZ outputs. This removes the generated "
        "enriched-USD scaffold from the full official loop test by selecting the USDA exported by the official Isaac "
        "Sim URDF importer. It still uses a captured local importer export rather than the live unmodified official "
        "converter entry, and it is not policy replay, PPO evaluation, or a paper-level tracking result."
    )
    official_loop_markers = summary["level_b_tracking"][
        "tracking_official_replay_npz_loop_with_enriched_usd_markers"
    ]
    official_loop_summary = {
        "app_launcher_constructed": summary["level_b_tracking"][
            "tracking_official_replay_npz_loop_with_enriched_usd_checks"
        ]["app_launcher_constructed"],
        "g1_cfg_patched_to_enriched_usd": summary["level_b_tracking"][
            "tracking_official_replay_npz_loop_with_enriched_usd_checks"
        ]["g1_cfg_patched_to_enriched_usd"],
        "fake_wandb_download_seen": summary["level_b_tracking"][
            "tracking_official_replay_npz_loop_with_enriched_usd_checks"
        ]["fake_wandb_download_seen"],
        "official_loop_call_299_seen": summary["level_b_tracking"][
            "tracking_official_replay_npz_loop_with_enriched_usd_checks"
        ]["official_loop_call_299_seen"],
        "official_loop_complete_seen": summary["level_b_tracking"][
            "tracking_official_replay_npz_loop_with_enriched_usd_checks"
        ]["official_loop_complete_seen"],
        "simulation_app_close_called": official_loop_markers["simulation_app_close_called"],
    }
    lines.append(
        f"- Level B official `replay_npz.py` loop with enriched-USD runtime patch: "
        f"`{summary['level_b_tracking']['tracking_official_replay_npz_loop_with_enriched_usd_status']}`; "
        f"latest blocker "
        f"`{summary['level_b_tracking']['tracking_official_replay_npz_loop_with_enriched_usd_latest_blocker']}`; "
        f"summary `{json.dumps(official_loop_summary, sort_keys=True)}`. "
        "This executes the official replay loop body to the 299-step bound after patching runtime dependencies only: "
        "the G1 robot config uses the validated resource-adjusted enriched USD and a local fake-WandB artifact points "
        "to the official-CSV-derived motion. This is stronger than the copied local replay script, but it remains "
        "resource-adjusted because the official URDF converter and official `csv_to_npz.py` output are still not "
        "validated."
    )
    full_replay = summary["level_b_tracking"][
        "tracking_official_replay_npz_loop_full_dataset_with_enriched_usd_aggregate"
    ]
    lines.append(
        f"- Level B full public-motion official `replay_npz.py` loop coverage: "
        f"`{summary['level_b_tracking']['tracking_official_replay_npz_loop_full_dataset_with_enriched_usd_status']}`; "
        f"replayed `{full_replay['ok_count']}/{full_replay['row_count']}` official-loop NPZ motions with "
        f"`{full_replay['failed_count']}` failures and `{full_replay['total_replayed_steps']}` total reference replay "
        f"steps. This extends the single-motion official replay-loop gate to the complete local public G1 LAFAN motion "
        "bundle. It remains resource-adjusted because both the robot asset and the NPZ inputs come from the enriched-USD "
        "runtime patch, and it is not trained-policy evaluation, PPO performance, DAgger, Fig. 5/Fig. 6, or real robot "
        "evidence."
    )
    full_replay_importer = summary["level_b_tracking"][
        "tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_aggregate"
    ]
    lines.append(
        f"- Level B full public-motion official `replay_npz.py` loop on captured official-importer-export G1 USDA: "
        f"`{summary['level_b_tracking']['tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_status']}`; "
        f"replayed `{full_replay_importer['ok_count']}/{full_replay_importer['row_count']}` matching official-loop "
        f"NPZ motions with `{full_replay_importer['failed_count']}` failures, "
        f"`{full_replay_importer['total_replayed_steps']}` total reference replay steps, and "
        f"`{full_replay_importer['shutdown_warning_count']}` shutdown warnings. This is the strongest current "
        "reference-replay loop evidence on the official-importer-export asset path, but it still bypasses the live "
        "unmodified converter entry and does not evaluate a trained policy, DAgger dataset, Fig. 5/Fig. 6 guidance, "
        "TensorRT deployment, or real robot."
    )
    replay_assets = summary["level_b_tracking"]["official_importer_export_replay_full_dataset_report_assets"]
    lines.append(
        f"- Official-importer-export full-dataset replay report assets: `{replay_assets['status']}`; "
        f"aggregate `{json.dumps(replay_assets['aggregate'], sort_keys=True)}`; assets "
        f"`{json.dumps(replay_assets['assets'], sort_keys=True)}`. "
        "These assets turn the 40/40 replay-loop audit into report/PPT-ready completion, family-summary, duration, "
        "and reference-video evidence. The MP4 remains local and is intentionally not a paper-level claim: this is not "
        "trained policy evaluation, unmodified live converter-entry success, Fig. 5/Fig. 6 guided diffusion, TensorRT "
        "deployment, or real robot."
    )
    full_task_eval = summary["level_b_tracking"][
        "tracking_g1_official_csv_loop_full_dataset_task_eval_aggregate"
    ]
    full_task_assets = summary["level_b_tracking"][
        "tracking_g1_official_csv_loop_full_dataset_task_eval_report_assets"
    ]
    full_task_summary = {
        "ok_count": full_task_eval["ok_count"],
        "row_count": full_task_eval["row_count"],
        "failed_count": full_task_eval["failed_count"],
        "total_steps": full_task_eval["total_steps"],
        "reward_mean": full_task_eval["reward_mean"]["mean"],
        "error_anchor_pos_mean": full_task_eval["error_anchor_pos"]["mean"],
        "error_body_pos_mean": full_task_eval["error_body_pos"]["mean"],
        "error_joint_pos_mean": full_task_eval["error_joint_pos"]["mean"],
    }
    lines.append(
        f"- Level B full public-motion `Tracking-Flat-G1-v0` task diagnostic: "
        f"`{summary['level_b_tracking']['tracking_g1_official_csv_loop_full_dataset_task_eval_status']}`; "
        f"summary `{json.dumps(full_task_summary, sort_keys=True)}`; report assets "
        f"`{json.dumps(full_task_assets, sort_keys=True)}`. The audit feeds all 40 official csv-loop NPZ motions "
        "into the official tracking task, reaches 299 steps for every motion, and validates action dim 29, policy obs "
        "dim 160, critic obs dim 286, nine reward terms, four termination terms, and the 29-joint/40-body G1 contract. "
        "It uses zero diagnostic actions and the enriched-USD runtime patch, so it is task-contract evidence rather "
        "than trained PPO teacher performance, unpatched official replay, DAgger, Fig. 5/Fig. 6, or real-robot evidence."
    )
    importer_smoke_metrics = summary["level_b_tracking"][
        "tracking_g1_official_importer_export_task_smoke_metrics"
    ]
    importer_smoke_summary = {
        "action_dim": importer_smoke_metrics.get("action_dim"),
        "policy_observation_dim": importer_smoke_metrics.get("policy_observation_dim"),
        "critic_observation_dim": importer_smoke_metrics.get("critic_observation_dim"),
        "robot_num_joints": importer_smoke_metrics.get("robot_num_joints"),
        "robot_num_bodies": importer_smoke_metrics.get("robot_num_bodies"),
        "step_count": importer_smoke_metrics.get("step_count"),
        "reward_mean": importer_smoke_metrics.get("reward_mean"),
    }
    lines.append(
        f"- Level B official-importer-export `Tracking-Flat-G1-v0` smoke gate: "
        f"`{summary['level_b_tracking']['tracking_g1_official_importer_export_task_smoke_status']}`; "
        f"summary `{json.dumps(importer_smoke_summary, sort_keys=True)}`. This replaces the generated enriched "
        "scaffold with the 311,027,678-byte USDA exported by the official Isaac Sim URDF importer GPU4 probe, then "
        "creates the official tracking task, resets it, and runs 8 zero-action steps on `cuda:4`. This proves the "
        "official-importer export can be consumed by IsaacLab's task stack, but it is still not a trained policy, "
        "paper-level replay, PPO, DAgger, Fig. 5/Fig. 6, TensorRT, or real-robot result."
    )
    importer_full_eval = summary["level_b_tracking"][
        "tracking_g1_official_importer_export_full_dataset_task_eval_aggregate"
    ]
    importer_full_assets = summary["level_b_tracking"][
        "tracking_g1_official_importer_export_full_dataset_task_eval_report_assets"
    ]
    importer_full_summary = {
        "ok_count": importer_full_eval["ok_count"],
        "row_count": importer_full_eval["row_count"],
        "failed_count": importer_full_eval["failed_count"],
        "total_steps": importer_full_eval["total_steps"],
        "reward_mean": importer_full_eval["reward_mean"]["mean"],
        "error_anchor_pos_mean": importer_full_eval["error_anchor_pos"]["mean"],
        "error_body_pos_mean": importer_full_eval["error_body_pos"]["mean"],
        "error_joint_pos_mean": importer_full_eval["error_joint_pos"]["mean"],
    }
    lines.append(
        f"- Level B full public-motion official-importer-export task diagnostic: "
        f"`{summary['level_b_tracking']['tracking_g1_official_importer_export_full_dataset_task_eval_status']}`; "
        f"summary `{json.dumps(importer_full_summary, sort_keys=True)}`; report assets "
        f"`{json.dumps(importer_full_assets, sort_keys=True)}`. The audit runs all 40 official-loop public G1 motion "
        "NPZs through `Tracking-Flat-G1-v0`, reaches 299 steps for every motion (`11960` total task steps), and "
        "validates action dim 29, policy obs dim 160, critic obs dim 286, nine reward terms, four termination terms, "
        "and the 29-joint/40-body G1 contract while using the official-importer GPU4 USDA export instead of the "
        "generated enriched scaffold. It now also consumes the full official-importer-export csv/replay loop outputs "
        "rather than the older enriched-USD NPZ set. It still uses zero diagnostic actions and a captured "
        "official-importer-export asset path rather than unpatched live converter-entry success, so it is not trained "
        "PPO teacher performance, DAgger, Fig. 5/Fig. 6, TensorRT, or real-robot evidence."
    )
    import_config_summary = {
        "has_set_make_instanceable": summary["level_b_tracking"][
            "tracking_g1_urdf_import_config_variant_probe_method_payload"
        ]["has_set_make_instanceable"],
        "has_set_instanceable_usd_path": summary["level_b_tracking"][
            "tracking_g1_urdf_import_config_variant_probe_method_payload"
        ]["has_set_instanceable_usd_path"],
        "baseline_stage_open_ok": summary["level_b_tracking"][
            "tracking_g1_urdf_import_config_variant_probe_baseline_usd"
        ]["stage_open_ok"],
        "baseline_prim_count": summary["level_b_tracking"][
            "tracking_g1_urdf_import_config_variant_probe_baseline_usd"
        ]["prim_count"],
        "baseline_joint_count": summary["level_b_tracking"][
            "tracking_g1_urdf_import_config_variant_probe_baseline_usd"
        ]["joint_count"],
        "baseline_rigid_body_like_count": summary["level_b_tracking"][
            "tracking_g1_urdf_import_config_variant_probe_baseline_usd"
        ]["rigid_body_like_count"],
    }
    lines.append(
        f"- Level B G1 URDF ImportConfig surface probe: "
        f"`{summary['level_b_tracking']['tracking_g1_urdf_import_config_variant_probe_status']}`; "
        f"current blocker "
        f"`{summary['level_b_tracking']['tracking_g1_urdf_import_config_variant_probe_current_blocker']}`; "
        f"summary `{json.dumps(import_config_summary, sort_keys=True)}`. "
        "In Isaac Sim 4.5 the official URDF import config exposed drive/default-prim setters but no "
        "`set_make_instanceable` or instanceable USD path setter, so the attempted Python-level instanceable patch "
        "surface is not available. The baseline official G1 URDF conversion produced an openable but empty USD "
        "(zero prims, joints, or rigid bodies). This closes one converter-debug path and points the next reproduction "
        "work back to runnable replay/task evaluation routes rather than more ImportConfig patching."
    )
    lines.append(
        f"- Level B resource-adjusted enriched USD replay preflight: "
        f"`{summary['level_b_tracking']['tracking_g1_enriched_usd_replay_preflight_status']}`; "
        f"latest blocker "
        f"`{summary['level_b_tracking']['tracking_g1_enriched_usd_replay_preflight_latest_blocker']}`; "
        f"checks "
        f"`{json.dumps(summary['level_b_tracking']['tracking_g1_enriched_usd_replay_preflight_checks'], sort_keys=True)}`. "
        "This bounded gate directly loads the generated enriched USD through IsaacLab `UsdFileCfg`, reaches "
        "`num_joints=29` and `num_bodies=40`, renders four fixture steps on `cuda:6`, and now returns from the "
        "bounded gate via an explicit success-after-sentinel process exit. Clean Kit shutdown is still not verified. "
        "It is a resource-adjusted environment/articulation gate only, not official csv_to_npz, official motion "
        "replay, PPO, DAgger, or paper-level closed-loop evidence."
    )
    lines.append(
        f"- Level B resource-adjusted enriched USD bounded replay metrics: "
        f"`{summary['level_b_tracking']['tracking_g1_enriched_usd_bounded_replay_metrics_status']}`; "
        f"metrics "
        f"`{json.dumps(summary['level_b_tracking']['tracking_g1_enriched_usd_bounded_replay_metrics_metrics'], sort_keys=True)}`. "
        "This extends the resource-adjusted gate to 64 debug-fixture steps, writes root and joint state, and records "
        "joint/root consistency metrics on `cuda:6`. It still uses a generated scaffold and debug fixture, so it is "
        "not official `csv_to_npz.py` output, official replay/evaluation, PPO, DAgger, or paper-level evidence."
    )
    lines.append(
        f"- Level B resource-adjusted official tracking task smoke: "
        f"`{summary['level_b_tracking']['tracking_g1_resource_adjusted_task_smoke_status']}`; "
        f"metrics "
        f"`{json.dumps(summary['level_b_tracking']['tracking_g1_resource_adjusted_task_smoke_metrics'], sort_keys=True)}`. "
        "This instantiates the official `Tracking-Flat-G1-v0` ManagerBasedRLEnv stack with the generated enriched USD "
        "and debug fixture, reaches reset, performs eight zero-action steps, and verifies action dimension 29, policy "
        "observation dimension 160, critic observation dimension 286, nine reward terms, and four termination terms. "
        "It is a resource-adjusted task smoke/eval gate, not official replay/evaluation, PPO, DAgger, or paper-level "
        "tracking evidence."
    )
    multi_fixture_metrics = summary["level_b_tracking"]["tracking_g1_resource_adjusted_multi_fixture_eval_metrics"]
    multi_fixture_summary = {
        key: multi_fixture_metrics[key]
        for key in [
            "fixture_count",
            "total_steps",
            "action_dim_all_29",
            "policy_observation_dim_all_160",
            "critic_observation_dim_all_286",
            "reward_terms_all_9",
            "termination_terms_all_4",
            "robot_num_joints_all_29",
            "robot_num_bodies_all_40",
        ]
    }
    lines.append(
        f"- Level B resource-adjusted official tracking task full fixture eval: "
        f"`{summary['level_b_tracking']['tracking_g1_resource_adjusted_multi_fixture_eval_status']}`; "
        f"metrics `{json.dumps(multi_fixture_summary, sort_keys=True)}`. "
        "This runs the official `Tracking-Flat-G1-v0` manager stack for all available steps in the three local debug "
        "fixtures (`walk`, `run`, `jump`), using isolated Kit processes to avoid observed teardown/recreate hangs, and "
        "records per-fixture rewards and termination counts. It is stronger resource-adjusted task-contract evidence "
        "than the eight-step smoke, but it is still not official `csv_to_npz.py` conversion, official replay/evaluation, "
        "PPO training, DAgger rollout data, or paper-level closed-loop tracking performance."
    )
    csv_conversion_metrics = summary["level_b_tracking"]["tracking_g1_resource_adjusted_csv_conversion_metrics"]
    csv_conversion_summary = {
        "joint_pos_shape": csv_conversion_metrics["joint_pos_shape"],
        "body_pos_w_shape": csv_conversion_metrics["body_pos_w_shape"],
        "root_height_min": csv_conversion_metrics["root_height_min"],
        "root_height_max": csv_conversion_metrics["root_height_max"],
        "npz_size_bytes": csv_conversion_metrics["npz_size_bytes"],
    }
    lines.append(
        f"- Level B resource-adjusted official-CSV conversion gate: "
        f"`{summary['level_b_tracking']['tracking_g1_resource_adjusted_csv_conversion_status']}`; "
        f"metrics `{json.dumps(csv_conversion_summary, sort_keys=True)}`. "
        "This converts the downloaded official G1 LAFAN `walk1_subject1.csv` frame range 1-180 into a 299-step "
        "`motion.npz` using the official interpolation/logging schema plus the generated enriched G1 USD. It narrows "
        "the replay blocker to the official URDF/USD conversion path, but the resulting `motion.npz` is explicitly "
        "resource-adjusted and must not be reported as official `csv_to_npz.py` output."
    )
    csv_full_replay_metrics = summary["level_b_tracking"]["tracking_g1_resource_adjusted_csv_full_replay_metrics"]
    csv_full_replay_summary = {
        key: csv_full_replay_metrics[key]
        for key in [
            "executed_steps",
            "motion_total_steps",
            "joint_pos_shape",
            "body_pos_w_shape",
            "max_joint_pos_abs_error",
            "max_root_pos_abs_error",
            "root_height_min",
            "root_height_max",
        ]
    }
    lines.append(
        f"- Level B resource-adjusted official-CSV full replay gate: "
        f"`{summary['level_b_tracking']['tracking_g1_resource_adjusted_csv_full_replay_status']}`; "
        f"metrics `{json.dumps(csv_full_replay_summary, sort_keys=True)}`. "
        "This replays the official-CSV-derived resource-adjusted motion for all 299 steps through the enriched USD "
        "replay surface and records zero joint/root write-read errors. It is still not official replay/evaluation, "
        "PPO, DAgger, or paper-level tracking performance."
    )
    csv_task_metrics = summary["level_b_tracking"]["tracking_g1_resource_adjusted_csv_task_eval_metrics"]
    csv_task_summary = {
        key: csv_task_metrics[key]
        for key in [
            "step_count",
            "action_dim",
            "policy_observation_dim",
            "critic_observation_dim",
            "reward_mean",
            "reward_min",
            "reward_max",
            "terminated_total",
            "truncated_total",
            "robot_num_joints",
            "robot_num_bodies",
        ]
    }
    lines.append(
        f"- Level B resource-adjusted official-CSV tracking task eval: "
        f"`{summary['level_b_tracking']['tracking_g1_resource_adjusted_csv_task_eval_status']}`; "
        f"metrics `{json.dumps(csv_task_summary, sort_keys=True)}`. "
        "This feeds the official-CSV-derived resource-adjusted `motion.npz` into the official `Tracking-Flat-G1-v0` "
        "ManagerBasedRLEnv stack for all 299 available steps and verifies action, observation, reward, termination, "
        "and robot-contract dimensions. It uses zero diagnostic actions and a generated enriched USD, so termination "
        "counts are not policy-quality evidence and the result is not official replay/evaluation or PPO."
    )
    train_entry_metrics = summary["level_b_tracking"]["tracking_g1_resource_adjusted_train_entry_diagnostic_metrics"]
    train_entry_summary = {
        key: train_entry_metrics[key]
        for key in [
            "requested_learning_iterations",
            "configured_num_steps_per_env",
            "num_envs",
            "num_actions",
            "num_obs",
            "num_privileged_obs",
            "runner_class",
            "runner_training_type",
            "checkpoint_written",
        ]
    }
    lines.append(
        f"- Level B resource-adjusted RSL-RL train-entry diagnostic: "
        f"`{summary['level_b_tracking']['tracking_g1_resource_adjusted_train_entry_diagnostic_status']}`; "
        f"metrics `{json.dumps(train_entry_summary, sort_keys=True)}`. "
        "This constructs the official `Tracking-Flat-G1-v0` env, wraps it with `RslRlVecEnvWrapper`, instantiates the "
        "official custom `MotionOnPolicyRunner`, and executes one tiny PPO learning iteration with four rollout steps. "
        "It verifies train-entry wiring only: no checkpoint is written, it is not formal PPO training, and it is not "
        "paper-level tracking performance. Runtime warning: "
        f"{summary['level_b_tracking']['tracking_g1_resource_adjusted_train_entry_diagnostic_warning']}"
    )
    ppo_config = summary["level_b_tracking"]["tracking_g1_resource_adjusted_ppo_training_run_config"]
    ppo_preflight = summary["level_b_tracking"]["tracking_g1_resource_adjusted_ppo_training_run_gpu_preflight"]
    ppo_summary = {
        "candidate_physical_gpus": ppo_config["candidate_physical_gpus"],
        "selected_physical_gpus": ppo_config["selected_physical_gpus"],
        "world_size": ppo_config["world_size"],
        "total_num_envs": ppo_config["total_num_envs"],
        "num_steps_per_env": ppo_config["num_steps_per_env"],
        "max_iterations": ppo_config["max_iterations"],
        "attempted_training": summary["level_b_tracking"]["tracking_g1_resource_adjusted_ppo_training_run_attempted"],
        "resource_ready": ppo_preflight["resource_ready"],
        "checkpoint_count": summary["level_b_tracking"][
            "tracking_g1_resource_adjusted_ppo_training_run_checkpoint_count"
        ],
    }
    lines.append(
        f"- Level B resource-adjusted PPO training run: "
        f"`{summary['level_b_tracking']['tracking_g1_resource_adjusted_ppo_training_run_status']}`; "
        f"summary `{json.dumps(ppo_summary, sort_keys=True)}`. "
        "The harness selects available GPUs from physical GPUs 4-7 and launches `torch.distributed` with the "
        "official `Tracking-Flat-G1-v0` manager stack, official PPO rollout length, GPU telemetry, checkpoints, and "
        "run metadata. The current run completed 100 resource-adjusted iterations on GPUs selected by preflight. "
        "The asset/motion path remains resource-adjusted, so this is evidence of virtual training execution and not "
        "official paper-level PPO training or a validated BeyondMimic teacher."
    )
    ppo_eval_config = summary["level_b_tracking"]["tracking_g1_resource_adjusted_ppo_checkpoint_eval_config"]
    ppo_eval_metrics = summary["level_b_tracking"]["tracking_g1_resource_adjusted_ppo_checkpoint_eval_metrics"]
    ppo_eval_motion = ppo_eval_metrics.get("motion_metrics", {})
    ppo_eval_summary = {
        "selected_physical_gpus": ppo_eval_config["selected_physical_gpus"],
        "num_envs": ppo_eval_config["num_envs"],
        "eval_steps": ppo_eval_config["eval_steps"],
        "total_env_steps": ppo_eval_config["total_env_steps"],
        "loaded_iteration": ppo_eval_metrics.get("loaded_iteration"),
        "duration_seconds": summary["level_b_tracking"][
            "tracking_g1_resource_adjusted_ppo_checkpoint_eval_duration_seconds"
        ],
        "reward_mean": ppo_eval_metrics.get("reward", {}).get("mean_over_steps", {}).get("mean"),
        "error_anchor_pos_mean": ppo_eval_motion.get("error_anchor_pos", {}).get("mean"),
        "error_body_pos_mean": ppo_eval_motion.get("error_body_pos", {}).get("mean"),
        "error_joint_pos_mean": ppo_eval_motion.get("error_joint_pos", {}).get("mean"),
    }
    lines.append(
        f"- Level B resource-adjusted PPO checkpoint evaluation: "
        f"`{summary['level_b_tracking']['tracking_g1_resource_adjusted_ppo_checkpoint_eval_status']}`; "
        f"summary `{json.dumps(ppo_eval_summary, sort_keys=True)}`. "
        "The evaluator loads `model_99.pt` with the official RSL-RL `OnPolicyRunner` inference API and runs "
        "`Tracking-Flat-G1-v0` for 512 environments x 299 steps while recording reward, termination, action, GPU, and "
        "motion-command tracking metrics. This is useful virtual policy-evaluation evidence, but it remains "
        "resource-adjusted and below official paper-level tracking evaluation."
    )
    csv_loop_ppo_config = summary["level_b_tracking"]["tracking_g1_official_csv_loop_ppo_training_run_config"]
    csv_loop_rank_metrics = summary["level_b_tracking"][
        "tracking_g1_official_csv_loop_ppo_training_run_rank_metrics"
    ]
    csv_loop_rank0 = next((item for item in csv_loop_rank_metrics if item.get("rank") == 0), {})
    csv_loop_ppo_summary = {
        "selected_physical_gpus": csv_loop_ppo_config["selected_physical_gpus"],
        "world_size": csv_loop_ppo_config["world_size"],
        "total_num_envs": csv_loop_ppo_config["total_num_envs"],
        "num_steps_per_env": csv_loop_ppo_config["num_steps_per_env"],
        "max_iterations": csv_loop_ppo_config["max_iterations"],
        "duration_seconds": summary["level_b_tracking"][
            "tracking_g1_official_csv_loop_ppo_training_run_duration_seconds"
        ],
        "checkpoint_count": summary["level_b_tracking"][
            "tracking_g1_official_csv_loop_ppo_training_run_checkpoint_count"
        ],
        "rank0_learning_iteration": csv_loop_rank0.get("current_learning_iteration"),
        "rank0_timesteps": csv_loop_rank0.get("tot_timesteps"),
    }
    lines.append(
        f"- Level B official csv-loop motion PPO training run: "
        f"`{summary['level_b_tracking']['tracking_g1_official_csv_loop_ppo_training_run_status']}`; "
        f"summary `{json.dumps(csv_loop_ppo_summary, sort_keys=True)}`. "
        "This run used the motion NPZ generated by the official `csv_to_npz.py` loop under the enriched-USD runtime "
        "patch, launched official `Tracking-Flat-G1-v0` and RSL-RL PPO through `torch.distributed` on GPUs 4 and 7, "
        "and wrote seven checkpoints through iteration 299. GPU utilization averaged about 98% on both cards, but "
        "peak memory was about 7.8GB/card, so it is a substantive virtual training run but below the requested "
        "10GB/card formal high-memory threshold. It remains resource-adjusted and is not paper-level PPO teacher "
        "training."
    )
    csv_loop_eval_config = summary["level_b_tracking"][
        "tracking_g1_official_csv_loop_ppo_checkpoint_eval_config"
    ]
    csv_loop_eval_metrics = summary["level_b_tracking"][
        "tracking_g1_official_csv_loop_ppo_checkpoint_eval_metrics"
    ]
    csv_loop_eval_motion = csv_loop_eval_metrics.get("motion_metrics", {})
    csv_loop_eval_summary = {
        "selected_physical_gpus": csv_loop_eval_config["selected_physical_gpus"],
        "num_envs": csv_loop_eval_config["num_envs"],
        "eval_steps": csv_loop_eval_config["eval_steps"],
        "total_env_steps": csv_loop_eval_config["total_env_steps"],
        "loaded_iteration": csv_loop_eval_metrics.get("loaded_iteration"),
        "duration_seconds": summary["level_b_tracking"][
            "tracking_g1_official_csv_loop_ppo_checkpoint_eval_duration_seconds"
        ],
        "reward_mean": csv_loop_eval_metrics.get("reward", {}).get("mean_over_steps", {}).get("mean"),
        "done_count_total": csv_loop_eval_metrics.get("done_count_total"),
        "error_anchor_pos_mean": csv_loop_eval_motion.get("error_anchor_pos", {}).get("mean"),
        "error_body_pos_mean": csv_loop_eval_motion.get("error_body_pos", {}).get("mean"),
        "error_joint_pos_mean": csv_loop_eval_motion.get("error_joint_pos", {}).get("mean"),
        "sampling_top1_prob_mean": csv_loop_eval_motion.get("sampling_top1_prob", {}).get("mean"),
    }
    lines.append(
        f"- Level B official csv-loop motion PPO checkpoint evaluation: "
        f"`{summary['level_b_tracking']['tracking_g1_official_csv_loop_ppo_checkpoint_eval_status']}`; "
        f"summary `{json.dumps(csv_loop_eval_summary, sort_keys=True)}`. "
        "The evaluator loads the iteration-299 checkpoint through the official RSL-RL `OnPolicyRunner` inference API "
        "and runs `Tracking-Flat-G1-v0` for 512 environments x 299 steps. This is stronger local virtual tracking "
        "evidence than the earlier model-99 evaluation, but it still depends on the enriched-USD runtime patch and "
        "does not establish official paper-level tracking performance."
    )
    ppo_eval_assets = summary["level_b_tracking"]["official_csv_loop_ppo_eval_report_assets"]
    lines.append(
        f"- Official csv-loop PPO eval report assets: `{ppo_eval_assets['status']}`; "
        f"assets `{json.dumps(ppo_eval_assets['assets'], sort_keys=True)}`; "
        f"claim level `{ppo_eval_assets['claim_level']}`. These provide report-ready tracking error, reward/done, "
        "GPU-usage plots and summary tables for the local virtual checkpoint evaluation, without claiming unpatched "
        "official PPO evaluation, Fig. 5/Fig. 6 guided diffusion, or real-robot validation."
    )
    multiseed_config = summary["level_b_tracking"][
        "tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval_config"
    ]
    multiseed_metrics = summary["level_b_tracking"][
        "tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval_metrics"
    ]
    multiseed_aggregate = summary["level_b_tracking"][
        "tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval_aggregate"
    ]
    multiseed_summary = {
        "seeds": multiseed_config["seeds"],
        "gpu_assignment": multiseed_config["gpu_assignment"],
        "num_envs": multiseed_config["num_envs"],
        "eval_steps": multiseed_config["eval_steps"],
        "total_env_steps": multiseed_metrics["total_env_steps"],
        "reward_mean": multiseed_aggregate["reward_mean"],
        "error_body_pos_mean": multiseed_aggregate["error_body_pos_mean"],
        "error_joint_pos_mean": multiseed_aggregate["error_joint_pos_mean"],
    }
    lines.append(
        f"- Level B official csv-loop motion PPO checkpoint multi-seed evaluation: "
        f"`{summary['level_b_tracking']['tracking_g1_official_csv_loop_ppo_checkpoint_multiseed_eval_status']}`; "
        f"summary `{json.dumps(multiseed_summary, sort_keys=True)}`. "
        "This repeats the full 512-env x 299-step local virtual evaluator for three seeds on GPUs 4/7/4, totaling "
        "459264 environment steps. It provides mean/std stability evidence for the local official-csv-loop tracking "
        "checkpoint, but it still uses the enriched-USD runtime patch and the reduced iteration-299 checkpoint, so "
        "it is not unpatched official paper-level PPO tracking evaluation."
    )
    multiseed_assets = summary["level_b_tracking"]["official_csv_loop_ppo_multiseed_eval_report_assets"]
    lines.append(
        f"- Official csv-loop PPO multi-seed eval report assets: `{multiseed_assets['status']}`; "
        f"assets `{json.dumps(multiseed_assets['assets'], sort_keys=True)}`; "
        f"claim level `{multiseed_assets['claim_level']}`. These provide report-ready per-seed reward/error plots, "
        "aggregate bars, GPU telemetry, and CSV summaries for the English reading report without promoting the run "
        "to official paper-level tracking, DAgger, Fig. 5/Fig. 6, or real-robot evidence."
    )
    full_bundle = summary["level_b_tracking"]["tracking_g1_official_csv_loop_full_bundle_motion_npz_bundle"]
    lines.append(
        f"- Official csv-loop full public motion bundle: "
        f"`{summary['level_b_tracking']['tracking_g1_official_csv_loop_full_bundle_motion_npz_status']}`; "
        f"summary `{json.dumps(full_bundle, sort_keys=True)}`. "
        "This concatenates all 40 public official-loop motion NPZs into one MotionLoader-compatible file without "
        "patching official loader code. It improves public-motion coverage for local virtual PPO, but the 39 clip "
        "boundaries are artificial and this is not the paper's original teacher motion sampler or DAgger dataset."
    )
    full_bundle_ppo_config = summary["level_b_tracking"][
        "tracking_g1_official_csv_loop_full_bundle_ppo_training_run_config"
    ]
    full_bundle_rank_metrics = summary["level_b_tracking"][
        "tracking_g1_official_csv_loop_full_bundle_ppo_training_run_rank_metrics"
    ]
    full_bundle_rank0 = next((item for item in full_bundle_rank_metrics if item.get("rank") == 0), {})
    full_bundle_training_summary = {
        "selected_physical_gpus": full_bundle_ppo_config["selected_physical_gpus"],
        "world_size": full_bundle_ppo_config["world_size"],
        "total_num_envs": full_bundle_ppo_config["total_num_envs"],
        "num_steps_per_env": full_bundle_ppo_config["num_steps_per_env"],
        "max_iterations": full_bundle_ppo_config["max_iterations"],
        "duration_seconds": summary["level_b_tracking"][
            "tracking_g1_official_csv_loop_full_bundle_ppo_training_run_duration_seconds"
        ],
        "checkpoint_count": summary["level_b_tracking"][
            "tracking_g1_official_csv_loop_full_bundle_ppo_training_run_checkpoint_count"
        ],
        "rank0_learning_iteration": full_bundle_rank0.get("current_learning_iteration"),
        "rank0_timesteps": full_bundle_rank0.get("tot_timesteps"),
        "motion_count": full_bundle["motion_count"],
        "total_motion_frames": full_bundle["total_frames"],
    }
    lines.append(
        f"- Level B official csv-loop full-bundle PPO training run: "
        f"`{summary['level_b_tracking']['tracking_g1_official_csv_loop_full_bundle_ppo_training_run_status']}`; "
        f"summary `{json.dumps(full_bundle_training_summary, sort_keys=True)}`. "
        "The run used GPUs 4 and 7 for a 300-iteration RSL-RL PPO training job over the 40-motion public bundle. "
        "Per-card memory peaked below 10GB because the official 512-env/rank harness fit in about 8GB/card; this is "
        "recorded as a real long virtual training run, not a smoke test, and no artificial memory inflation was used."
    )
    full_bundle_eval_config = summary["level_b_tracking"][
        "tracking_g1_official_csv_loop_full_bundle_ppo_checkpoint_eval_config"
    ]
    full_bundle_eval_metrics = summary["level_b_tracking"][
        "tracking_g1_official_csv_loop_full_bundle_ppo_checkpoint_eval_metrics"
    ]
    full_bundle_eval_motion = full_bundle_eval_metrics.get("motion_metrics", {})
    full_bundle_eval_summary = {
        "selected_physical_gpus": full_bundle_eval_config["selected_physical_gpus"],
        "num_envs": full_bundle_eval_config["num_envs"],
        "eval_steps": full_bundle_eval_config["eval_steps"],
        "total_env_steps": full_bundle_eval_config["total_env_steps"],
        "loaded_iteration": full_bundle_eval_metrics.get("loaded_iteration"),
        "duration_seconds": summary["level_b_tracking"][
            "tracking_g1_official_csv_loop_full_bundle_ppo_checkpoint_eval_duration_seconds"
        ],
        "reward_mean": full_bundle_eval_metrics.get("reward", {}).get("mean_over_steps", {}).get("mean"),
        "done_count_total": full_bundle_eval_metrics.get("done_count_total"),
        "error_anchor_pos_mean": full_bundle_eval_motion.get("error_anchor_pos", {}).get("mean"),
        "error_body_pos_mean": full_bundle_eval_motion.get("error_body_pos", {}).get("mean"),
        "error_joint_pos_mean": full_bundle_eval_motion.get("error_joint_pos", {}).get("mean"),
        "motion_count": full_bundle_eval_metrics.get("motion_count"),
        "total_motion_frames": full_bundle_eval_metrics.get("total_motion_frames"),
    }
    lines.append(
        f"- Level B official csv-loop full-bundle PPO checkpoint evaluation: "
        f"`{summary['level_b_tracking']['tracking_g1_official_csv_loop_full_bundle_ppo_checkpoint_eval_status']}`; "
        f"summary `{json.dumps(full_bundle_eval_summary, sort_keys=True)}`. "
        "The evaluator loaded the iteration-299 checkpoint and ran `Tracking-Flat-G1-v0` for 512 environments x "
        "299 steps. This is the current strongest local virtual tracking evidence, but it still uses enriched-USD "
        "assets and artificial bundle boundaries, so it remains below official paper-level teacher evaluation."
    )
    full_bundle_assets = summary["level_b_tracking"]["official_csv_loop_full_bundle_ppo_eval_report_assets"]
    lines.append(
        f"- Official csv-loop full-bundle PPO eval report assets: `{full_bundle_assets['status']}`; "
        f"assets `{json.dumps(full_bundle_assets['assets'], sort_keys=True)}`; "
        f"claim level `{full_bundle_assets['claim_level']}`. These add report-ready plots and summary tables for "
        "the full-public-motion PPO checkpoint evaluation while explicitly avoiding Fig. 5/Fig. 6, DAgger, "
        "unpatched official, or real-robot claims."
    )
    importer_ppo_config = summary["level_b_tracking"][
        "tracking_g1_official_importer_export_full_bundle_ppo_training_run_config"
    ]
    importer_rank_metrics = summary["level_b_tracking"][
        "tracking_g1_official_importer_export_full_bundle_ppo_training_run_rank_metrics"
    ]
    importer_rank0 = next((item for item in importer_rank_metrics if item.get("rank") == 0), {})
    importer_training_summary = {
        "selected_physical_gpus": importer_ppo_config["selected_physical_gpus"],
        "world_size": importer_ppo_config["world_size"],
        "total_num_envs": importer_ppo_config["total_num_envs"],
        "num_steps_per_env": importer_ppo_config["num_steps_per_env"],
        "max_iterations": importer_ppo_config["max_iterations"],
        "duration_seconds": summary["level_b_tracking"][
            "tracking_g1_official_importer_export_full_bundle_ppo_training_run_duration_seconds"
        ],
        "checkpoint_count": summary["level_b_tracking"][
            "tracking_g1_official_importer_export_full_bundle_ppo_training_run_checkpoint_count"
        ],
        "rank0_learning_iteration": importer_rank0.get("current_learning_iteration"),
        "rank0_timesteps": importer_rank0.get("tot_timesteps"),
        "uses_official_importer_export_usd": importer_rank0.get("uses_official_importer_export_usd"),
    }
    lines.append(
        f"- Level B official-importer-export full-bundle PPO training run: "
        f"`{summary['level_b_tracking']['tracking_g1_official_importer_export_full_bundle_ppo_training_run_status']}`; "
        f"summary `{json.dumps(importer_training_summary, sort_keys=True)}`. "
        "This is a 300-iteration two-GPU RSL-RL PPO run using the large G1 USDA exported by the official Isaac Sim "
        "URDF importer and the 40-motion public bundle. It is a stronger tracking step than the enriched-USD PPO "
        "run because the robot asset is no longer the resource-adjusted scaffold, but it remains a local exported "
        "asset and not the official paper teacher-policy run."
    )
    importer_eval_config = summary["level_b_tracking"][
        "tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval_config"
    ]
    importer_eval_metrics = summary["level_b_tracking"][
        "tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval_metrics"
    ]
    importer_eval_motion = importer_eval_metrics.get("motion_metrics", {})
    importer_eval_summary = {
        "selected_physical_gpus": importer_eval_config["selected_physical_gpus"],
        "num_envs": importer_eval_config["num_envs"],
        "eval_steps": importer_eval_config["eval_steps"],
        "total_env_steps": importer_eval_config["total_env_steps"],
        "loaded_iteration": importer_eval_metrics.get("loaded_iteration"),
        "duration_seconds": summary["level_b_tracking"][
            "tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval_duration_seconds"
        ],
        "reward_mean": importer_eval_metrics.get("reward", {}).get("mean_over_steps", {}).get("mean"),
        "done_count_total": importer_eval_metrics.get("done_count_total"),
        "error_anchor_pos_mean": importer_eval_motion.get("error_anchor_pos", {}).get("mean"),
        "error_body_pos_mean": importer_eval_motion.get("error_body_pos", {}).get("mean"),
        "error_joint_pos_mean": importer_eval_motion.get("error_joint_pos", {}).get("mean"),
        "motion_count": importer_eval_metrics.get("motion_count"),
        "total_motion_frames": importer_eval_metrics.get("total_motion_frames"),
    }
    lines.append(
        f"- Level B official-importer-export full-bundle PPO checkpoint evaluation: "
        f"`{summary['level_b_tracking']['tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval_status']}`; "
        f"summary `{json.dumps(importer_eval_summary, sort_keys=True)}`. "
        "The evaluator loaded the iteration-299 checkpoint and ran 512 environments x 299 steps. It is local "
        "virtual policy evidence for the tracking teacher path, not DAgger, not VAE/diffusion guidance, not "
        "TensorRT deployment, and not real-robot validation."
    )
    importer_assets = summary["level_b_tracking"]["official_importer_export_full_bundle_ppo_eval_report_assets"]
    lines.append(
        f"- Official-importer-export PPO eval report assets: `{importer_assets['status']}`; "
        f"assets `{json.dumps(importer_assets['assets'], sort_keys=True)}`; "
        f"claim level `{importer_assets['claim_level']}`. These include training-curve, eval-error, reward/done, "
        "and GPU telemetry plots for the English report/PPT while preserving the non-paper-level claim boundary."
    )
    scaled_importer_ppo_config = summary["level_b_tracking"][
        "tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run_config"
    ]
    scaled_importer_rank_metrics = summary["level_b_tracking"][
        "tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run_rank_metrics"
    ]
    scaled_importer_rank0 = next((item for item in scaled_importer_rank_metrics if item.get("rank") == 0), {})
    scaled_importer_training_summary = {
        "selected_physical_gpus": scaled_importer_ppo_config["selected_physical_gpus"],
        "world_size": scaled_importer_ppo_config["world_size"],
        "total_num_envs": scaled_importer_ppo_config["total_num_envs"],
        "num_steps_per_env": scaled_importer_ppo_config["num_steps_per_env"],
        "max_iterations": scaled_importer_ppo_config["max_iterations"],
        "duration_seconds": summary["level_b_tracking"][
            "tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run_duration_seconds"
        ],
        "checkpoint_count": summary["level_b_tracking"][
            "tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run_checkpoint_count"
        ],
        "num_envs_per_rank": scaled_importer_ppo_config.get("num_envs_per_rank"),
        "rank0_learning_iteration": scaled_importer_rank0.get("current_learning_iteration"),
        "rank0_timesteps": scaled_importer_rank0.get("tot_timesteps"),
        "uses_official_importer_export_usd": scaled_importer_rank0.get("uses_official_importer_export_usd"),
    }
    lines.append(
        f"- Level B official-importer-export scaled PPO training run: "
        f"`{summary['level_b_tracking']['tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run_status']}`; "
        f"summary `{json.dumps(scaled_importer_training_summary, sort_keys=True)}`. "
        "This extends the same official-importer-export/public-bundle path to 1000 PPO iterations with "
        f"{scaled_importer_ppo_config['total_num_envs']} total environments "
        f"({scaled_importer_ppo_config.get('num_envs_per_rank')} per rank) on GPUs 4 and 7. "
        "It is stronger local virtual training evidence, but the observed "
        "peak memory stayed below the requested 10GB/card threshold and the resulting teacher is still not a paper "
        "checkpoint."
    )
    scaled_importer_eval_config = summary["level_b_tracking"][
        "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval_config"
    ]
    scaled_importer_eval_metrics = summary["level_b_tracking"][
        "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval_metrics"
    ]
    scaled_importer_eval_motion = scaled_importer_eval_metrics.get("motion_metrics", {})
    scaled_importer_eval_summary = {
        "selected_physical_gpus": scaled_importer_eval_config["selected_physical_gpus"],
        "num_envs": scaled_importer_eval_config["num_envs"],
        "eval_steps": scaled_importer_eval_config["eval_steps"],
        "total_env_steps": scaled_importer_eval_config["total_env_steps"],
        "loaded_iteration": scaled_importer_eval_metrics.get("loaded_iteration"),
        "duration_seconds": summary["level_b_tracking"][
            "tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval_duration_seconds"
        ],
        "reward_mean": scaled_importer_eval_metrics.get("reward", {}).get("mean_over_steps", {}).get("mean"),
        "done_count_total": scaled_importer_eval_metrics.get("done_count_total"),
        "error_anchor_pos_mean": scaled_importer_eval_motion.get("error_anchor_pos", {}).get("mean"),
        "error_body_pos_mean": scaled_importer_eval_motion.get("error_body_pos", {}).get("mean"),
        "error_joint_pos_mean": scaled_importer_eval_motion.get("error_joint_pos", {}).get("mean"),
        "motion_count": scaled_importer_eval_metrics.get("motion_count"),
        "total_motion_frames": scaled_importer_eval_metrics.get("total_motion_frames"),
    }
    lines.append(
        f"- Level B official-importer-export scaled PPO checkpoint evaluation: "
        f"`{summary['level_b_tracking']['tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval_status']}`; "
        f"summary `{json.dumps(scaled_importer_eval_summary, sort_keys=True)}`. "
        "The evaluator loaded the iteration-999 checkpoint and ran 2048 environments x 299 steps. The high done count "
        "and weak reward show that the scaled local checkpoint is not a mature tracking teacher, so this remains "
        "qualitative/local virtual evidence for the reading report rather than paper-level tracking reproduction."
    )
    scaled_importer_assets = summary["level_b_tracking"][
        "official_importer_export_full_bundle_scaled_ppo_eval_report_assets"
    ]
    lines.append(
        f"- Official-importer-export scaled PPO eval report assets: `{scaled_importer_assets['status']}`; "
        f"assets `{json.dumps(scaled_importer_assets['assets'], sort_keys=True)}`; "
        f"claim level `{scaled_importer_assets['claim_level']}`. These plots and CSVs document the larger local PPO "
        "run while preserving the boundary from official BeyondMimic teacher checkpoints, DAgger logs, Fig. 5/Fig. 6 "
        "rollouts, TensorRT deployment, and real-robot validation."
    )
    scaled_importer_completion_proxy = summary["level_b_tracking"][
        "official_importer_export_scaled_ppo_checkpoint_completion_proxy"
    ]
    scaled_importer_completion_metrics = scaled_importer_completion_proxy["metrics"]
    lines.append(
        f"- Official-importer-export scaled PPO completion/termination proxy: "
        f"`{scaled_importer_completion_proxy['status']}`; metrics "
        f"`{json.dumps(scaled_importer_completion_metrics, sort_keys=True)}`; assets "
        f"`{json.dumps(scaled_importer_completion_proxy['assets'], sort_keys=True)}`. This converts the "
        "2048-env x 299-step checkpoint evaluation into local termination/completion plots and makes the negative "
        "evidence explicit: almost all attempted virtual env-steps ended in non-timeout done. It is not a paper "
        "success/fall/collision metric and not an official teacher result."
    )
    scaled_importer_sweep = summary["level_b_tracking"][
        "tracking_g1_official_importer_export_scaled_ppo_checkpoint_sweep"
    ]
    scaled_importer_sweep_summary = {
        "metrics": scaled_importer_sweep["metrics"],
        "best_checkpoint": {
            "iteration": scaled_importer_sweep["best_checkpoint"].get("iteration"),
            "reward_mean": scaled_importer_sweep["best_checkpoint"].get("reward_mean"),
            "local_non_timeout_done_rate": scaled_importer_sweep["best_checkpoint"].get("local_non_timeout_done_rate"),
            "error_body_pos_mean": scaled_importer_sweep["best_checkpoint"].get("error_body_pos_mean"),
        },
        "report_assets": scaled_importer_sweep.get("report_assets", {}),
    }
    lines.append(
        f"- Official-importer-export scaled PPO checkpoint sweep: `{scaled_importer_sweep['status']}`; "
        f"summary `{json.dumps(scaled_importer_sweep_summary, sort_keys=True)}`. This screens 21 saved local PPO "
        "checkpoints at 256 envs x 299 steps on the official-importer-export G1 USDA and full public motion bundle. "
        "The best local screening row is iteration 300, but the local non-timeout done rate is still 1.0, so the "
        "result argues for teacher/training diagnosis rather than paper-level tracking success."
    )
    scaled_importer_best_confirmation = summary["level_b_tracking"][
        "tracking_g1_official_importer_export_scaled_ppo_best_checkpoint_confirmation_eval"
    ]
    scaled_importer_best_confirmation_summary = {
        "config": scaled_importer_best_confirmation["config"],
        "best_metrics": scaled_importer_best_confirmation["best_metrics"],
        "final_metrics": scaled_importer_best_confirmation["final_metrics"],
        "deltas": scaled_importer_best_confirmation["deltas"],
        "report_assets": scaled_importer_best_confirmation["report_assets"],
    }
    lines.append(
        f"- Official-importer-export scaled PPO best-checkpoint confirmation eval: "
        f"`{scaled_importer_best_confirmation['status']}`; summary "
        f"`{json.dumps(scaled_importer_best_confirmation_summary, sort_keys=True)}`. This reruns the sweep-selected "
        "iteration-300 checkpoint at the same 2048-env x 299-step scale as the final iteration-999 eval. The "
        "confirmation shows iteration 300 does not beat final iteration 999 in reward or tracking error, so the "
        "sweep is useful diagnosis but not evidence of a stronger paper-level tracking teacher."
    )
    scaled_importer_reward_diag = summary["level_b_tracking"][
        "official_importer_export_scaled_ppo_reward_termination_diagnostic"
    ]
    lines.append(
        f"- Official-importer-export scaled PPO reward/termination diagnostic: "
        f"`{scaled_importer_reward_diag['status']}`; metrics "
        f"`{json.dumps(scaled_importer_reward_diag['metrics'], sort_keys=True)}`; assets "
        f"`{json.dumps(scaled_importer_reward_diag['assets'], sort_keys=True)}`. The dominant non-timeout "
        "termination component is ee_body_pos for both the sweep-selected iteration-300 checkpoint and final "
        "iteration-999 checkpoint, with per-step fractions above 0.99. This points the next mainline debugging step "
        "toward body-tracking/termination configuration or teacher quality rather than checkpoint selection alone."
    )
    scaled_importer_ee_body_source = summary["level_b_tracking"][
        "official_importer_export_scaled_ppo_ee_body_pos_termination_source_audit"
    ]
    lines.append(
        f"- Official-importer-export scaled PPO ee_body_pos source-linked termination audit: "
        f"`{scaled_importer_ee_body_source['status']}`; source config "
        f"`{json.dumps(scaled_importer_ee_body_source['source_config'], sort_keys=True)}`; motion bundle "
        f"`{json.dumps(scaled_importer_ee_body_source['motion_bundle'], sort_keys=True)}`; assets "
        f"`{json.dumps(scaled_importer_ee_body_source['assets'], sort_keys=True)}`. The official source confirms "
        "`ee_body_pos` uses the z-only body-position termination with a 0.25 m threshold on the left/right ankles "
        "and wrists, while the local scaled PPO best and final checkpoints trip that gate for more than 99% of "
        "env-steps. This sharpens the next tracking debug target without claiming a paper-level teacher result."
    )
    scaled_importer_endpoint_trace = summary["level_b_tracking"][
        "tracking_g1_official_importer_export_scaled_ppo_endpoint_z_error_trace"
    ]
    endpoint_metrics = scaled_importer_endpoint_trace["run"]["metrics"]
    lines.append(
        f"- Official-importer-export scaled PPO endpoint z-error full-size trace: "
        f"`{scaled_importer_endpoint_trace['status']}`; config "
        f"`{json.dumps(scaled_importer_endpoint_trace['config'], sort_keys=True)}`; aggregate "
        f"`{json.dumps(endpoint_metrics['aggregate'], sort_keys=True)}`; body rows "
        f"`{json.dumps(endpoint_metrics['body_rows'], sort_keys=True)}`; report assets "
        f"`{json.dumps(scaled_importer_endpoint_trace['outputs']['report_assets'], sort_keys=True)}`. This "
        "2048-env x 299-step trace confirms the current local teacher mainly fails the official endpoint gate through "
        "ankle height: left/right ankle mean absolute z-errors are roughly 0.71/0.72 m against the 0.25 m threshold, "
        "with near-unit exceed rates. The result points the next mainline fix toward retargeted ankle height, body "
        "index consistency, and termination/curriculum handling before downstream teacher rollouts are trusted."
    )
    scaled_importer_multiseed_summary = {
        "config": summary["level_b_tracking"][
            "tracking_g1_official_importer_export_scaled_ppo_checkpoint_multiseed_eval_config"
        ],
        "metrics": summary["level_b_tracking"][
            "tracking_g1_official_importer_export_scaled_ppo_checkpoint_multiseed_eval_metrics"
        ],
        "aggregate": summary["level_b_tracking"][
            "tracking_g1_official_importer_export_scaled_ppo_checkpoint_multiseed_eval_aggregate"
        ],
    }
    scaled_importer_multiseed_assets = summary["level_b_tracking"][
        "official_importer_export_scaled_ppo_checkpoint_multiseed_eval_report_assets"
    ]
    lines.append(
        f"- Official-importer-export scaled PPO checkpoint multiseed evaluation: "
        f"`{summary['level_b_tracking']['tracking_g1_official_importer_export_scaled_ppo_checkpoint_multiseed_eval_status']}`; "
        f"summary `{json.dumps(scaled_importer_multiseed_summary, sort_keys=True)}`; report assets "
        f"`{json.dumps(scaled_importer_multiseed_assets['assets'], sort_keys=True)}`. This reruns the iteration-999 "
        "local checkpoint for three full 2048-env x 299-step seeds and records reward/done/tracking-error plots. It "
        "is stronger robustness evidence than the single-seed eval, but it is still local virtual evidence rather than "
        "an official BeyondMimic teacher checkpoint or paper-level tracking metric."
    )
    scaled_importer_policy_video = summary["level_b_tracking"][
        "official_importer_export_full_bundle_scaled_ppo_policy_rollout_video_asset"
    ]
    scaled_importer_policy_capture = summary["level_b_tracking"][
        "official_importer_export_full_bundle_scaled_ppo_policy_rollout_capture"
    ]
    lines.append(
        f"- Official-importer-export scaled PPO policy rollout video: "
        f"`{scaled_importer_policy_capture['status']}`; claim level "
        f"`{scaled_importer_policy_video['claim_level']}`; metrics "
        f"`{json.dumps(scaled_importer_policy_video['metrics'], sort_keys=True)}`; assets "
        f"`{json.dumps(scaled_importer_policy_video['assets'], sort_keys=True)}`. This is a 299-frame "
        "single-environment policy-vs-reference visualization from the iteration-999 scaled local PPO checkpoint "
        "on the official-importer-export G1 USDA and 40-motion public bundle. It is report/PPT media for the "
        "tracking pipeline only: not an official BeyondMimic teacher checkpoint, not a paper-level tracking metric, "
        "not Fig. 5/Fig. 6 guided diffusion, not TensorRT deployment, and not real-robot evidence."
    )
    tracking_eval_summary_assets = summary["level_b_tracking"][
        "official_importer_export_tracking_eval_summary_assets"
    ]
    tracking_eval_summary_brief = {
        "status": tracking_eval_summary_assets["status"],
        "claim_level": tracking_eval_summary_assets["interpretation"]["claim_level"],
        "task_diagnostic": tracking_eval_summary_assets["metrics"]["full_dataset_task_diagnostic"],
        "scaled_ppo_checkpoint_eval": tracking_eval_summary_assets["metrics"]["scaled_ppo_checkpoint_eval"],
        "scaled_ppo_policy_video": tracking_eval_summary_assets["metrics"]["scaled_ppo_policy_video"],
    }
    lines.append(
        f"- Official-importer-export tracking evaluation summary assets: "
        f"`{tracking_eval_summary_assets['status']}`; summary "
        f"`{json.dumps(tracking_eval_summary_brief, sort_keys=True)}`; assets "
        f"`{json.dumps(tracking_eval_summary_assets['assets'], sort_keys=True)}`. This is the reading-report bridge "
        "between the 40/40 task diagnostic, the scaled PPO checkpoint evaluation, and the scaled local policy video. "
        "It is deliberately scoped as local virtual evidence, not an official BeyondMimic teacher checkpoint, not "
        "paper Fig. 5/Fig. 6 closed-loop guidance, not TensorRT deployment, and not real-robot validation."
    )
    importer_teacher_config = summary["level_b_tracking"][
        "tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset_config"
    ]
    importer_teacher_aggregate = summary["level_b_tracking"][
        "tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset_aggregate"
    ]
    importer_teacher_summary = {
        "selected_physical_gpus": importer_teacher_config["selected_physical_gpus"],
        "world_size": importer_teacher_config["world_size"],
        "num_envs_per_rank": importer_teacher_config["num_envs_per_rank"],
        "rollout_steps": importer_teacher_config["rollout_steps"],
        "total_env_steps": importer_teacher_aggregate["total_env_steps"],
        "motion_count": importer_teacher_aggregate["motion_count"],
        "total_motion_frames": importer_teacher_aggregate["total_motion_frames"],
        "shard_count": importer_teacher_aggregate["shard_count"],
        "dataset_npz_total_size_bytes": importer_teacher_aggregate["dataset_npz_total_size_bytes"],
        "reward_mean_by_rank": importer_teacher_aggregate["reward_mean_by_rank"],
        "done_count_total": importer_teacher_aggregate["done_count_total"],
        "duration_seconds": summary["level_b_tracking"][
            "tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset_duration_seconds"
        ],
        "gpu_metrics_summary": summary["level_b_tracking"][
            "tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset_gpu_metrics"
        ],
    }
    lines.append(
        f"- Level B official-importer-export full-bundle teacher rollout dataset gate: "
        f"`{summary['level_b_tracking']['tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset_status']}`; "
        f"summary `{json.dumps(importer_teacher_summary, sort_keys=True)}`. "
        "This collects two raw ignored `.npz` shards from the local iteration-299 PPO checkpoint trained with the "
        "official-importer-export G1 USDA and the 40-motion public bundle. It is now the strongest local virtual "
        "teacher-data candidate on the more official robot-asset path, but it is still a short local PPO-derived "
        "dataset, not the official BeyondMimic DAgger rollout log and not paper-level Fig. 5/Fig. 6 closed-loop "
        "guided diffusion evidence."
    )
    importer_teacher_assets = summary["level_b_tracking"][
        "official_importer_export_full_bundle_teacher_rollout_report_assets"
    ]
    lines.append(
        f"- Official-importer-export full-bundle teacher rollout report assets: "
        f"`{importer_teacher_assets['status']}`; metrics "
        f"`{json.dumps(importer_teacher_assets['metrics'], sort_keys=True)}`; assets "
        f"`{json.dumps(importer_teacher_assets['assets'], sort_keys=True)}`. "
        "These add report-ready reward/done, action-distribution, and motion-step coverage plots for the "
        "official-importer-export local teacher rollout while preserving the non-official, non-real-robot claim level."
    )
    scaled_importer_teacher_config = summary["level_b_tracking"][
        "tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset_config"
    ]
    scaled_importer_teacher_aggregate = summary["level_b_tracking"][
        "tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset_aggregate"
    ]
    scaled_importer_teacher_summary = {
        "selected_physical_gpus": scaled_importer_teacher_config["selected_physical_gpus"],
        "world_size": scaled_importer_teacher_config["world_size"],
        "num_envs_per_rank": scaled_importer_teacher_config["num_envs_per_rank"],
        "rollout_steps": scaled_importer_teacher_config["rollout_steps"],
        "total_env_steps": scaled_importer_teacher_aggregate["total_env_steps"],
        "motion_count": scaled_importer_teacher_aggregate["motion_count"],
        "total_motion_frames": scaled_importer_teacher_aggregate["total_motion_frames"],
        "shard_count": scaled_importer_teacher_aggregate["shard_count"],
        "dataset_npz_total_size_bytes": scaled_importer_teacher_aggregate["dataset_npz_total_size_bytes"],
        "reward_mean_by_rank": scaled_importer_teacher_aggregate["reward_mean_by_rank"],
        "done_count_total": scaled_importer_teacher_aggregate["done_count_total"],
        "duration_seconds": summary["level_b_tracking"][
            "tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset_duration_seconds"
        ],
        "gpu_metrics_summary": summary["level_b_tracking"][
            "tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset_gpu_metrics"
        ],
    }
    lines.append(
        f"- Level B official-importer-export scaled PPO teacher rollout dataset gate: "
        f"`{summary['level_b_tracking']['tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset_status']}`; "
        f"summary `{json.dumps(scaled_importer_teacher_summary, sort_keys=True)}`. "
        "This collects two raw ignored `.npz` shards from the local iteration-999 scaled PPO checkpoint, using "
        "2048 envs/rank for 1,224,704 virtual env steps on the official-importer-export G1 USDA and 40-motion "
        "public bundle. It is the strongest current local teacher-data candidate for future downstream VAE/state-"
        "latent experiments, but it is still local virtual data, not the official BeyondMimic DAgger rollout log, "
        "not paper Fig. 5/Fig. 6 guidance, and not real-robot evidence."
    )
    scaled_importer_teacher_assets = summary["level_b_tracking"][
        "official_importer_export_full_bundle_scaled_ppo_teacher_rollout_report_assets"
    ]
    lines.append(
        f"- Official-importer-export scaled PPO teacher rollout report assets: "
        f"`{scaled_importer_teacher_assets['status']}`; metrics "
        f"`{json.dumps(scaled_importer_teacher_assets['metrics'], sort_keys=True)}`; assets "
        f"`{json.dumps(scaled_importer_teacher_assets['assets'], sort_keys=True)}`. "
        "These add report-ready reward/done, action-distribution, and motion-step coverage plots for the latest "
        "iteration-999 teacher-rollout dataset while preserving the non-official, non-real-robot claim level."
    )
    reference_video = summary["level_b_tracking"]["official_csv_loop_reference_replay_video_asset"]
    lines.append(
        f"- Official csv-loop reference replay visualization: `{reference_video['status']}`; "
        f"frames `{reference_video['frame_count']}`, bodies `{reference_video['body_count']}`, "
        f"target bodies `{reference_video['target_body_count']}`, claim level "
        f"`{reference_video['claim_level']}`. Assets are recorded under "
        f"`{reference_video['assets']['readme']}` and include a local MP4 SHA256 plus a report keyframe PNG. "
        "This is a kinematic visualization of saved reference motion only, not an IsaacLab closed-loop rollout "
        "video, not Fig. 5/Fig. 6 guided diffusion evidence, and not real-robot validation."
    )
    importer_reference_video = summary["level_b_tracking"][
        "official_importer_export_full_dataset_reference_replay_video_asset"
    ]
    lines.append(
        f"- Official-importer-export full-dataset reference replay visualization: "
        f"`{importer_reference_video['status']}`; selected motion "
        f"`{importer_reference_video['selected_motion']}`, frames `{importer_reference_video['frame_count']}`, "
        f"full audit rows `{importer_reference_video['source_dataset_aggregate']['ok_count']}/"
        f"{importer_reference_video['source_dataset_aggregate']['row_count']}`, claim level "
        f"`{importer_reference_video['claim_level']}`. Assets are recorded under "
        f"`{importer_reference_video['assets']['readme']}` and include a local MP4 SHA256 plus a report keyframe PNG. "
        "This complements the 40/40 official-importer-export conversion/replay loop evidence with visual reference "
        "motion context, but it remains a kinematic saved-trajectory visualization, not an IsaacLab closed-loop "
        "policy rollout, not unmodified live official converter-entry output, not Fig. 5/Fig. 6 guided diffusion "
        "evidence, and not real-robot validation."
    )
    policy_video = summary["level_b_tracking"]["official_csv_loop_policy_rollout_video_asset"]
    policy_capture = summary["level_b_tracking"]["official_csv_loop_policy_rollout_capture"]
    lines.append(
        f"- Official csv-loop local policy rollout video: `{policy_capture['status']}`; "
        f"claim level `{policy_video['claim_level']}`; metrics "
        f"`{json.dumps(policy_video['metrics'], sort_keys=True)}`; assets "
        f"`{json.dumps(policy_video['assets'], sort_keys=True)}`. This captures a 299-step single-env rollout from "
        "the local iteration-299 PPO checkpoint, records robot/reference target-body poses, and renders a local "
        "policy-vs-reference MP4 plus keyframes. It is the first robot policy video artifact in the project, but it "
        "remains resource-adjusted local virtual evidence, not unpatched official replay, not Fig. 5/Fig. 6 guided "
        "diffusion, and not real-robot validation."
    )
    full_bundle_policy_video = summary["level_b_tracking"][
        "official_csv_loop_full_bundle_policy_rollout_video_asset"
    ]
    full_bundle_policy_capture = summary["level_b_tracking"][
        "official_csv_loop_full_bundle_policy_rollout_capture"
    ]
    lines.append(
        f"- Official csv-loop full-bundle local policy rollout video: "
        f"`{full_bundle_policy_capture['status']}`; claim level "
        f"`{full_bundle_policy_video['claim_level']}`; bundle "
        f"`{json.dumps(full_bundle_policy_video.get('bundle', {}), sort_keys=True)}`; metrics "
        f"`{json.dumps(full_bundle_policy_video['metrics'], sort_keys=True)}`; assets "
        f"`{json.dumps(full_bundle_policy_video['assets'], sort_keys=True)}`. This upgrades the earlier "
        "single-motion policy video to the 40-motion public official-csv-loop bundle and gives the English report "
        "a clearer robot policy-vs-reference visualization from the full-bundle local PPO checkpoint. It remains "
        "resource-adjusted local virtual evidence, not an official BeyondMimic checkpoint, not paper-level "
        "Fig. 5/Fig. 6 guided diffusion, not TensorRT deployment evidence, and not real-robot validation."
    )
    vae_video = summary["level_b_tracking"]["official_csv_loop_vae_closed_loop_rollout_video_asset"]
    vae_capture = summary["level_b_tracking"]["official_csv_loop_vae_closed_loop_rollout_capture"]
    lines.append(
        f"- Official csv-loop local VAE action-reconstruction rollout video: `{vae_capture['status']}`; "
        f"claim level `{vae_video['claim_level']}`; metrics "
        f"`{json.dumps(vae_video['metrics'], sort_keys=True)}`; assets "
        f"`{json.dumps(vae_video['assets'], sort_keys=True)}`. This captures a 299-frame single-env rollout in "
        "which the local PPO teacher action is encoded and decoded by the local conditional action VAE before "
        "stepping IsaacLab. It gives the English report and PPT a concrete robot motion video for the VAE "
        "closed-loop gate, but it is not the unreleased official BeyondMimic VAE checkpoint, not autonomous VAE "
        "control, not receding-horizon guided diffusion, not Fig. 5/Fig. 6 reproduction, and not real-robot evidence."
    )
    importer_vae_video = summary["level_b_tracking"][
        "official_importer_export_full_bundle_vae_closed_loop_rollout_video_asset"
    ]
    importer_vae_capture = summary["level_b_tracking"][
        "official_importer_export_full_bundle_vae_closed_loop_rollout_capture"
    ]
    lines.append(
        f"- Official-importer-export full-bundle VAE action-reconstruction rollout video: "
        f"`{importer_vae_capture['status']}`; claim level `{importer_vae_video['claim_level']}`; metrics "
        f"`{json.dumps(importer_vae_video['metrics'], sort_keys=True)}`; assets "
        f"`{json.dumps(importer_vae_video['assets'], sort_keys=True)}`. This captures a 299-frame single-env "
        "rollout using the current official-importer-export G1 USDA, local 40-motion PPO teacher, and local "
        "full-bundle conditional action VAE. It is the report video counterpart to the two-GPU importer VAE "
        "closed-loop metric gate. It remains local qualitative evidence: not the official BeyondMimic VAE "
        "checkpoint, not autonomous VAE control, not receding-horizon guided diffusion, not Fig. 5/Fig. 6 "
        "reproduction, and not real-robot evidence."
    )
    teacher_rollout_config = summary["level_b_tracking"][
        "tracking_g1_resource_adjusted_teacher_rollout_dataset_config"
    ]
    teacher_rollout_aggregate = summary["level_b_tracking"][
        "tracking_g1_resource_adjusted_teacher_rollout_dataset_aggregate"
    ]
    teacher_rollout_summary = {
        "selected_physical_gpus": teacher_rollout_config["selected_physical_gpus"],
        "world_size": teacher_rollout_config["world_size"],
        "num_envs_per_rank": teacher_rollout_config["num_envs_per_rank"],
        "rollout_steps": teacher_rollout_config["rollout_steps"],
        "total_env_steps": teacher_rollout_aggregate["total_env_steps"],
        "shard_count": teacher_rollout_aggregate["shard_count"],
        "dataset_npz_total_size_bytes": teacher_rollout_aggregate["dataset_npz_total_size_bytes"],
        "reward_mean_by_rank": teacher_rollout_aggregate["reward_mean_by_rank"],
        "duration_seconds": summary["level_b_tracking"][
            "tracking_g1_resource_adjusted_teacher_rollout_dataset_duration_seconds"
        ],
        "gpu_metrics_summary": summary["level_b_tracking"][
            "tracking_g1_resource_adjusted_teacher_rollout_dataset_gpu_metrics"
        ],
    }
    lines.append(
        f"- Level B resource-adjusted teacher rollout dataset gate: "
        f"`{summary['level_b_tracking']['tracking_g1_resource_adjusted_teacher_rollout_dataset_status']}`; "
        f"summary `{json.dumps(teacher_rollout_summary, sort_keys=True)}`. "
        "The run used fixed physical GPUs 4 and 7, collected two raw `.npz` shards under ignored `res/runs`, and "
        "records policy observations, critic observations, actions, rewards, dones, timeouts, and motion timesteps "
        "from the local `model_99.pt` resource-adjusted teacher. This is suitable as a local downstream dataset "
        "candidate for VAE/state-latent experiments, but it is not the official BeyondMimic DAgger rollout log and "
        "does not validate paper-level Fig. 5/Fig. 6 closed-loop diffusion results."
    )
    csv_loop_teacher_config = summary["level_b_tracking"][
        "tracking_g1_official_csv_loop_teacher_rollout_dataset_config"
    ]
    csv_loop_teacher_aggregate = summary["level_b_tracking"][
        "tracking_g1_official_csv_loop_teacher_rollout_dataset_aggregate"
    ]
    csv_loop_teacher_summary = {
        "selected_physical_gpus": csv_loop_teacher_config["selected_physical_gpus"],
        "world_size": csv_loop_teacher_config["world_size"],
        "num_envs_per_rank": csv_loop_teacher_config["num_envs_per_rank"],
        "rollout_steps": csv_loop_teacher_config["rollout_steps"],
        "total_env_steps": csv_loop_teacher_aggregate["total_env_steps"],
        "shard_count": csv_loop_teacher_aggregate["shard_count"],
        "dataset_npz_total_size_bytes": csv_loop_teacher_aggregate["dataset_npz_total_size_bytes"],
        "reward_mean_by_rank": csv_loop_teacher_aggregate["reward_mean_by_rank"],
        "duration_seconds": summary["level_b_tracking"][
            "tracking_g1_official_csv_loop_teacher_rollout_dataset_duration_seconds"
        ],
        "gpu_metrics_summary": summary["level_b_tracking"][
            "tracking_g1_official_csv_loop_teacher_rollout_dataset_gpu_metrics"
        ],
    }
    lines.append(
        f"- Level B official csv-loop motion teacher rollout dataset gate: "
        f"`{summary['level_b_tracking']['tracking_g1_official_csv_loop_teacher_rollout_dataset_status']}`; "
        f"summary `{json.dumps(csv_loop_teacher_summary, sort_keys=True)}`. "
        "This uses the local iteration-299 checkpoint trained on official-loop motion and collects two raw `.npz` "
        "shards on GPUs 4 and 7. It records the same observation/action/reward/done/timeout/motion-timestep fields "
        "needed by the downstream VAE/state-latent pipeline, and is the strongest current local teacher-rollout "
        "dataset candidate. It is still not the paper's official DAgger dataset, because the source checkpoint and "
        "motion chain depend on the enriched-USD runtime patch and a 300-iteration local teacher."
    )
    teacher_assets = summary["level_b_tracking"]["official_csv_loop_teacher_rollout_report_assets"]
    lines.append(
        f"- Official csv-loop teacher rollout report assets: `{teacher_assets['status']}`; "
        f"metrics `{json.dumps(teacher_assets['metrics'], sort_keys=True)}`; "
        f"assets `{json.dumps(teacher_assets['assets'], sort_keys=True)}`. These provide report-ready reward/done, "
        "action-distribution, and motion-step coverage plots for the full 306,176-step local virtual teacher "
        "rollout dataset. They remain local virtual evidence, not official DAgger logs, not closed-loop guided "
        "diffusion, and not real-robot validation."
    )
    full_bundle_teacher_config = summary["level_b_tracking"][
        "tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset_config"
    ]
    full_bundle_teacher_aggregate = summary["level_b_tracking"][
        "tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset_aggregate"
    ]
    full_bundle_teacher_summary = {
        "selected_physical_gpus": full_bundle_teacher_config["selected_physical_gpus"],
        "world_size": full_bundle_teacher_config["world_size"],
        "num_envs_per_rank": full_bundle_teacher_config["num_envs_per_rank"],
        "rollout_steps": full_bundle_teacher_config["rollout_steps"],
        "total_env_steps": full_bundle_teacher_aggregate["total_env_steps"],
        "motion_count": full_bundle_teacher_aggregate["motion_count"],
        "total_motion_frames": full_bundle_teacher_aggregate["total_motion_frames"],
        "shard_count": full_bundle_teacher_aggregate["shard_count"],
        "dataset_npz_total_size_bytes": full_bundle_teacher_aggregate["dataset_npz_total_size_bytes"],
        "reward_mean_by_rank": full_bundle_teacher_aggregate["reward_mean_by_rank"],
        "done_count_total": full_bundle_teacher_aggregate["done_count_total"],
        "duration_seconds": summary["level_b_tracking"][
            "tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset_duration_seconds"
        ],
        "gpu_metrics_summary": summary["level_b_tracking"][
            "tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset_gpu_metrics"
        ],
    }
    lines.append(
        f"- Level B official csv-loop full-bundle teacher rollout dataset gate: "
        f"`{summary['level_b_tracking']['tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset_status']}`; "
        f"summary `{json.dumps(full_bundle_teacher_summary, sort_keys=True)}`. "
        "This collects two raw ignored `.npz` shards from the local iteration-299 PPO checkpoint trained on the "
        "40-motion public official-loop bundle. It is the strongest current local virtual teacher-rollout dataset "
        "candidate for downstream VAE/state-latent experiments, but it still depends on the enriched-USD runtime "
        "patch, an audited status shim for the shared rollout harness, and artificial bundle boundaries. It is not "
        "the official BeyondMimic DAgger dataset and does not validate Fig. 5/Fig. 6 closed-loop diffusion."
    )
    full_bundle_teacher_assets = summary["level_b_tracking"][
        "official_csv_loop_full_bundle_teacher_rollout_report_assets"
    ]
    lines.append(
        f"- Official csv-loop full-bundle teacher rollout report assets: "
        f"`{full_bundle_teacher_assets['status']}`; metrics "
        f"`{json.dumps(full_bundle_teacher_assets['metrics'], sort_keys=True)}`; assets "
        f"`{json.dumps(full_bundle_teacher_assets['assets'], sort_keys=True)}`. "
        "These add report-ready reward/done, action-distribution, and full-bundle motion-step coverage plots for "
        "the local 306,176-step teacher rollout dataset while preserving the non-official, non-real-robot claim level."
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
        f"- Level B USD API variant probe: "
        f"`{summary['level_b_tracking']['tracking_usd_api_variant_probe_status']}`; "
        f"current blocker `{summary['level_b_tracking']['tracking_usd_api_variant_probe_current_blocker']}`; "
        f"successful write APIs "
        f"`{json.dumps(summary['level_b_tracking']['tracking_usd_api_variant_probe_successful_attempt_labels'])}`. "
        "`layer.Save()` remains blocked by `permissionToSave=False`, but direct `Usd.Stage.Export(...)` paths write "
        "non-empty local USD files. This is a concrete next-step workaround for conversion plumbing, not official "
        "replay success."
    )
    lines.append(
        f"- Level B G1 URDF Stage.Export workaround probe: "
        f"`{summary['level_b_tracking']['tracking_g1_urdf_stage_export_probe_status']}`; "
        f"current blocker `{summary['level_b_tracking']['tracking_g1_urdf_stage_export_probe_current_blocker']}`; "
        f"parse result `{summary['level_b_tracking']['tracking_g1_urdf_stage_export_probe_parse_result']}`; "
        f"patch events `{json.dumps(summary['level_b_tracking']['tracking_g1_urdf_stage_export_probe_patch_events'], sort_keys=True)}`. "
        "The importer's initial `Stage.Save()` was routed to `Stage.Export()`, but the generated G1 destination and "
        "current stages still contain no robot prims because deeper base/physics/sensor layer saves remain blocked. "
        "This is a narrower blocker classification, not official replay success."
    )
    lines.append(
        f"- Level B G1 URDF Sdf.Layer.Save workaround probe: "
        f"`{summary['level_b_tracking']['tracking_g1_urdf_layer_save_probe_status']}`; "
        f"current blocker `{summary['level_b_tracking']['tracking_g1_urdf_layer_save_probe_current_blocker']}`; "
        f"parse result `{summary['level_b_tracking']['tracking_g1_urdf_layer_save_probe_parse_result']}`; "
        f"layer-save patch exception "
        f"`{summary['level_b_tracking']['tracking_g1_urdf_layer_save_probe_layer_save_exception']}`; "
        f"checks `{json.dumps(summary['level_b_tracking']['tracking_g1_urdf_layer_save_probe_checks'], sort_keys=True)}`. "
        "This probes the deeper Python-visible layer-save boundary for the importer configuration layers. It is not "
        "official replay success and produces no motion.npz unless the resulting USD contains a valid robot stage."
    )
    lines.append(
        f"- Level B G1 URDF in-memory import probe: "
        f"`{summary['level_b_tracking']['tracking_g1_urdf_in_memory_probe_status']}`; "
        f"current blocker `{summary['level_b_tracking']['tracking_g1_urdf_in_memory_probe_current_blocker']}`; "
        f"parse result `{summary['level_b_tracking']['tracking_g1_urdf_in_memory_probe_parse_result']}`; "
        f"markers `{json.dumps(summary['level_b_tracking']['tracking_g1_urdf_in_memory_probe_markers'], sort_keys=True)}`. "
        "This tries `dest_path=\"\"` so the URDF importer uses the current in-memory Kit stage instead of layered "
        "file output. It currently records Vulkan device loss before an exported robot stage can be captured, so it is "
        "not official replay success."
    )
    lines.append(
        f"- Level B G1 URDF SimulationApp in-memory import probe: "
        f"`{summary['level_b_tracking']['tracking_g1_urdf_simulationapp_in_memory_probe_status']}`; "
        f"current blocker "
        f"`{summary['level_b_tracking']['tracking_g1_urdf_simulationapp_in_memory_probe_current_blocker']}`; "
        f"return code `{summary['level_b_tracking']['tracking_g1_urdf_simulationapp_in_memory_probe_returncode']}`; "
        f"markers "
        f"`{json.dumps(summary['level_b_tracking']['tracking_g1_urdf_simulationapp_in_memory_probe_markers'], sort_keys=True)}`. "
        "This repeats the `dest_path=\"\"` URDF importer test under raw `SimulationApp` with the IsaacLab headless "
        "experience. It reaches the in-memory importer branch but crashes with Vulkan device loss before payload, so "
        "the blocker is now localized below the AppLauncher wrapper and remains a Kit/GPU runtime gate."
    )
    lines.append(
        f"- Level B G1 URDF in-memory variant matrix: "
        f"`{summary['level_b_tracking']['tracking_g1_urdf_in_memory_variant_matrix_probe_status']}`; "
        f"current blocker "
        f"`{summary['level_b_tracking']['tracking_g1_urdf_in_memory_variant_matrix_probe_current_blocker']}`; "
        f"cases "
        f"`{json.dumps(summary['level_b_tracking']['tracking_g1_urdf_in_memory_variant_matrix_probe_cases'], sort_keys=True)}`. "
        "This tests GPU 5, GPU 6, waitIdle/low-RTX settings, and the IsaacLab headless-rendering experience. It "
        "produces no valid G1 USD, so the official replay gate remains blocked and no motion.npz/replay result is "
        "claimed."
    )
    lines.append(
        f"- Level B G1 URDF in-memory GPU4 probe: "
        f"`{summary['level_b_tracking']['tracking_g1_urdf_in_memory_gpu4_probe_status']}`; "
        f"return code `{summary['level_b_tracking']['tracking_g1_urdf_in_memory_gpu4_probe_returncode']}`; "
        f"duration `{summary['level_b_tracking']['tracking_g1_urdf_in_memory_gpu4_probe_duration_seconds']}` seconds; "
        f"latest blocker `{summary['level_b_tracking']['tracking_g1_urdf_in_memory_gpu4_probe_latest_blocker']}`; "
        f"checks "
        f"`{json.dumps(summary['level_b_tracking']['tracking_g1_urdf_in_memory_gpu4_probe_checks'], sort_keys=True)}`. "
        "This repeats the official G1 URDF importer `dest_path=\"\"` path on the current GPU4 headless setup. It "
        "reaches AppLauncher, returns from in-memory URDF parsing, and writes a local USDA export, but Vulkan "
        "`ERROR_DEVICE_LOST` occurs before payload/clean close. It is therefore blocker evidence only, not official "
        "replay or paper-level tracking."
    )
    lines.append(
        f"- Level B G1 URDF in-memory GPU4 export structure audit: "
        f"`{summary['level_b_tracking']['tracking_g1_urdf_in_memory_gpu4_export_structure_status']}`; "
        f"latest blocker "
        f"`{summary['level_b_tracking']['tracking_g1_urdf_in_memory_gpu4_export_structure_latest_blocker']}`; "
        f"export `{json.dumps(summary['level_b_tracking']['tracking_g1_urdf_in_memory_gpu4_export_structure_export'], sort_keys=True)}`; "
        f"checks "
        f"`{json.dumps(summary['level_b_tracking']['tracking_g1_urdf_in_memory_gpu4_export_structure_checks'], sort_keys=True)}`. "
        "The local official-importer USDA contains a G1 default prim, 40 rigid-body API rows, one articulation root, "
        "29 revolute joints, 29 joint-state/drive rows, all 29 action joints, and checked target bodies. The large "
        "USDA remains ignored locally; no official csv_to_npz/replay_npz, PPO, DAgger, VAE/diffusion, TensorRT, "
        "Fig. 5/Fig. 6, or robot result is claimed from it."
    )
    lines.append(
        f"- Level B G1 preconverted asset audit: "
        f"`{summary['level_b_tracking']['tracking_g1_preconverted_asset_audit_status']}`; "
        f"counts "
        f"`{json.dumps(summary['level_b_tracking']['tracking_g1_preconverted_asset_audit_counts'], sort_keys=True)}`; "
        f"validated reference USD "
        f"`{json.dumps(summary['level_b_tracking']['tracking_g1_preconverted_asset_audit_reference_usd'], sort_keys=True)}`. "
        "The official whole_body_tracking work copy contains mesh-level USD files but no official full-robot "
        "preconverted G1 USD. A reference-code ASAP G1 USD opens as a robot-like stage in Kit, but it is explicitly "
        "not an official BeyondMimic replay asset and can only be used, if at all, as a clearly labeled "
        "resource-adjusted workaround."
    )
    lines.append(
        f"- Level B G1 reference USD compatibility audit: "
        f"`{summary['level_b_tracking']['tracking_g1_reference_usd_compatibility_audit_status']}`; "
        f"compatible for resource-adjusted replay "
        f"`{summary['level_b_tracking']['tracking_g1_reference_usd_compatibility_audit_compatible']}`; "
        f"official contract "
        f"`{json.dumps(summary['level_b_tracking']['tracking_g1_reference_usd_compatibility_audit_official_contract'], sort_keys=True)}`; "
        f"reference contract "
        f"`{json.dumps(summary['level_b_tracking']['tracking_g1_reference_usd_compatibility_audit_reference_contract'], sort_keys=True)}`; "
        f"missing action joints "
        f"`{summary['level_b_tracking']['tracking_g1_reference_usd_compatibility_audit_missing_action_joints']}`. "
        "All official target bodies are present, but the six wrist action joints are fixed rather than revolute in the "
        "reference USD, so it is not a drop-in 29-DoF BeyondMimic replay asset."
    )
    lines.append(
        f"- Level B official-URDF minimal skeleton USD audit: "
        f"`{summary['level_b_tracking']['tracking_g1_official_urdf_skeleton_usd_audit_status']}`; "
        f"contract ok `{summary['level_b_tracking']['tracking_g1_official_urdf_skeleton_usd_contract_ok']}`; "
        f"official contract "
        f"`{json.dumps(summary['level_b_tracking']['tracking_g1_official_urdf_skeleton_usd_official_contract'], sort_keys=True)}`; "
        f"skeleton contract "
        f"`{json.dumps(summary['level_b_tracking']['tracking_g1_official_urdf_skeleton_usd_skeleton_contract'], sort_keys=True)}`. "
        "This local USD preserves the official 40-link/29-revolute-joint/14-target-body naming contract and is "
        "validated by a read-only Kit probe, but it is a placeholder structure asset without official converter "
        "success, meshes, collisions, inertias, drives, motion.npz, replay, or training evidence."
    )
    lines.append(
        f"- Level B G1 URDF physical asset contract audit: "
        f"`{summary['level_b_tracking']['tracking_g1_urdf_physical_asset_contract_status']}`; "
        f"metrics "
        f"`{json.dumps(summary['level_b_tracking']['tracking_g1_urdf_physical_asset_contract_metrics'], sort_keys=True)}`. "
        "The official URDF provides all 35 visual mesh references, 29 collision elements, and all 29 non-fixed "
        "joint axis/limit/action-drive rows needed for an offline USD converter scaffold. Three sensor/IMU links "
        "lack inertial tags, no target body lacks inertial data, and no physical USD, motion.npz, replay, or training "
        "success is claimed."
    )
    source_equivalence = summary["level_b_tracking"]["tracking_g1_urdf_source_equivalence_download_vs_wbt"]
    source_equivalence_summary = {
        "download_vs_wbt_link_diff": source_equivalence["link_set_diff"],
        "download_vs_wbt_joint_diff": source_equivalence["joint_set_diff"],
        "action_joint_summary": summary["level_b_tracking"][
            "tracking_g1_urdf_source_equivalence_action_joint_summary"
        ],
    }
    lines.append(
        f"- Level B G1 URDF source-equivalence audit: "
        f"`{summary['level_b_tracking']['tracking_g1_urdf_source_equivalence_status']}`; "
        f"summary `{json.dumps(source_equivalence_summary, sort_keys=True)}`. "
        "The downloaded official LAFAN G1 URDF and the reproduction-data copy are byte-identical and structurally "
        "identical. The official `whole_body_tracking` G1 URDF keeps the same 29 non-fixed/action joints, but differs "
        "in support links/joints (`d435_link/d435_joint` versus `LL_FOOT/LR_FOOT` foot frames) and physical bookkeeping. "
        "This improves source traceability for the offline scaffold while explicitly avoiding any claim of identical "
        "URDF sources, official converter success, motion.npz, replay, or paper-level tracking."
    )
    lines.append(
        f"- Level B resource-adjusted enriched G1 USD scaffold probe: "
        f"`{summary['level_b_tracking']['tracking_g1_resource_adjusted_enriched_usd_status']}`; "
        f"readback "
        f"`{json.dumps(summary['level_b_tracking']['tracking_g1_resource_adjusted_enriched_usd_readback'], sort_keys=True)}`. "
        "The generated scaffold authors public URDF mass/inertia metadata, visual mesh references, collision proxy "
        "geometry, joint limits, and drive metadata onto the 29-DoF skeleton. It is still explicitly not official "
        "URDF converter output and has not passed official csv_to_npz/replay validation."
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
    resource_adjusted_vae_worker = summary["level_c_diffusion"][
        "resource_adjusted_teacher_rollout_vae_training_worker"
    ]
    resource_adjusted_vae_summary = {
        "sample_count": resource_adjusted_vae_worker["dataset"]["sample_count"],
        "obs_dim": resource_adjusted_vae_worker["dataset"]["obs_dim"],
        "action_dim": resource_adjusted_vae_worker["dataset"]["action_dim"],
        "splits": resource_adjusted_vae_worker["splits"],
        "latent_dim": resource_adjusted_vae_worker["training"]["latent_dim"],
        "epochs": resource_adjusted_vae_worker["training"]["epochs"],
        "cuda_visible_devices": resource_adjusted_vae_worker["cuda_visible_devices"],
        "torch_cuda_device_count": resource_adjusted_vae_worker["torch_cuda_device_count"],
        "data_parallel_used": resource_adjusted_vae_worker["data_parallel_used"],
        "validation_action_mse": resource_adjusted_vae_worker["evaluation"]["validation"]["action_mse"],
        "test_action_mse": resource_adjusted_vae_worker["evaluation"]["test"]["action_mse"],
        "test_action_abs_error_mean": resource_adjusted_vae_worker["evaluation"]["test"][
            "action_abs_error_mean"
        ],
        "gpu_metrics_summary": summary["level_c_diffusion"][
            "resource_adjusted_teacher_rollout_vae_training_gpu_metrics"
        ],
    }
    lines.append(
        f"- Resource-adjusted full teacher-rollout conditional action VAE training: "
        f"`{summary['level_c_diffusion']['resource_adjusted_teacher_rollout_vae_training_status']}`; "
        f"summary `{json.dumps(resource_adjusted_vae_summary, sort_keys=True)}`. "
        "This run trains on all currently collected local resource-adjusted teacher rollout shards and writes its "
        "checkpoint only under ignored `res/runs`. It is stronger than a smoke test, but remains a resource-adjusted "
        "local VAE result rather than the official BeyondMimic DAgger/VAE checkpoint or a closed-loop diffusion result."
    )
    official_loop_vae_worker = summary["level_c_diffusion"][
        "official_csv_loop_teacher_rollout_vae_training_worker"
    ]
    official_loop_vae_summary = {
        "sample_count": official_loop_vae_worker["dataset"]["sample_count"],
        "splits": official_loop_vae_worker["splits"],
        "obs_dim": official_loop_vae_worker["dataset"]["obs_dim"],
        "action_dim": official_loop_vae_worker["dataset"]["action_dim"],
        "epochs": official_loop_vae_worker["training"]["epochs"],
        "latent_dim": official_loop_vae_worker["training"]["latent_dim"],
        "torch_cuda_device_count": official_loop_vae_worker["torch_cuda_device_count"],
        "data_parallel_used": official_loop_vae_worker["data_parallel_used"],
        "validation_action_mse": official_loop_vae_worker["evaluation"]["validation"]["action_mse"],
        "test_action_mse": official_loop_vae_worker["evaluation"]["test"]["action_mse"],
        "test_action_abs_error_mean": official_loop_vae_worker["evaluation"]["test"][
            "action_abs_error_mean"
        ],
        "gpu_metrics_summary": summary["level_c_diffusion"][
            "official_csv_loop_teacher_rollout_vae_training_gpu_metrics"
        ],
    }
    lines.append(
        f"- Official csv-loop teacher-rollout conditional action VAE training: "
        f"`{summary['level_c_diffusion']['official_csv_loop_teacher_rollout_vae_training_status']}`; "
        f"summary `{json.dumps(official_loop_vae_summary, sort_keys=True)}`. "
        "This trains on the two-shard dataset collected from the local iteration-299 official-loop-motion PPO "
        "checkpoint. It is now the strongest local VAE training evidence for the downstream state-latent chain, but "
        "it remains a local virtual artifact rather than the paper's official DAgger/VAE checkpoint or closed-loop "
        "VAE rollout result."
    )
    official_loop_state_latent_worker = summary["level_c_diffusion"][
        "official_csv_loop_teacher_rollout_state_latent_dataset_worker"
    ]
    official_loop_state_latent_summary = {
        "sample_count": official_loop_state_latent_worker["dataset"]["sample_count"],
        "window_count": official_loop_state_latent_worker["dataset"]["window_count"],
        "split_counts": official_loop_state_latent_worker["dataset"]["split_counts"],
        "sequence_length": official_loop_state_latent_worker["dataset"]["sequence_length"],
        "obs_dim": official_loop_state_latent_worker["dataset"]["obs_dim"],
        "latent_dim": official_loop_state_latent_worker["dataset"]["latent_dim"],
        "token_dim": official_loop_state_latent_worker["dataset"]["token_dim"],
        "weighted_posterior_reconstruction_mse": official_loop_state_latent_worker["dataset"][
            "weighted_posterior_reconstruction_mse"
        ],
        "gpu_metrics_summary": summary["level_c_diffusion"][
            "official_csv_loop_teacher_rollout_state_latent_dataset_gpu_metrics"
        ],
    }
    lines.append(
        f"- Official csv-loop teacher-rollout state-latent dataset: "
        f"`{summary['level_c_diffusion']['official_csv_loop_teacher_rollout_state_latent_dataset_status']}`; "
        f"summary `{json.dumps(official_loop_state_latent_summary, sort_keys=True)}`. "
        "This converts the official-loop teacher rollout plus local official-loop VAE posterior into full 21-step "
        "state/latent windows. It is a stronger local downstream dataset for diffusion training, but it is not the "
        "paper's unreleased official DAgger/state-latent dataset and does not evaluate guided control."
    )
    official_loop_diffusion_worker = summary["level_c_diffusion"][
        "official_csv_loop_state_latent_diffusion_training_worker"
    ]
    official_loop_diffusion_summary = {
        "window_count": official_loop_diffusion_worker["dataset"]["window_count"],
        "split_counts": official_loop_diffusion_worker["dataset"]["split_counts"],
        "epochs": official_loop_diffusion_worker["training"]["epochs"],
        "batch_windows": official_loop_diffusion_worker["training"]["batch_windows"],
        "data_parallel_used": official_loop_diffusion_worker["data_parallel_used"],
        "validation_pred_token_mse": official_loop_diffusion_worker["evaluation"]["validation"][
            "pred_token_mse"
        ],
        "test_pred_token_mse": official_loop_diffusion_worker["evaluation"]["test"]["pred_token_mse"],
        "test_noisy_token_mse": official_loop_diffusion_worker["evaluation"]["test"]["noisy_token_mse"],
        "test_denoising_improvement_ratio": official_loop_diffusion_worker["evaluation"]["test"][
            "denoising_improvement_ratio"
        ],
        "gpu_metrics_summary": summary["level_c_diffusion"][
            "official_csv_loop_state_latent_diffusion_training_gpu_metrics"
        ],
    }
    lines.append(
        f"- Official csv-loop full state-latent denoiser training: "
        f"`{summary['level_c_diffusion']['official_csv_loop_state_latent_diffusion_training_status']}`; "
        f"summary `{json.dumps(official_loop_diffusion_summary, sort_keys=True)}`. "
        "This trains a local denoiser on all official-loop state-latent windows and reports held-out denoising "
        "improvement. It is still not the official BeyondMimic diffusion checkpoint, TensorRT/asynchronous "
        "deployment, or closed-loop Fig. 5/Fig. 6 guidance evidence."
    )
    full_bundle_vae_worker = summary["level_c_diffusion"][
        "official_csv_loop_full_bundle_teacher_rollout_vae_training_worker"
    ]
    full_bundle_vae_summary = {
        "sample_count": full_bundle_vae_worker["dataset"]["sample_count"],
        "motion_time_step_max": full_bundle_vae_worker["dataset"]["motion_time_step_max"],
        "splits": full_bundle_vae_worker["splits"],
        "epochs": full_bundle_vae_worker["training"]["epochs"],
        "latent_dim": full_bundle_vae_worker["training"]["latent_dim"],
        "data_parallel_used": full_bundle_vae_worker["data_parallel_used"],
        "test_action_mse": full_bundle_vae_worker["evaluation"]["test"]["action_mse"],
        "test_action_abs_error_mean": full_bundle_vae_worker["evaluation"]["test"]["action_abs_error_mean"],
        "gpu_metrics_summary": summary["level_c_diffusion"][
            "official_csv_loop_full_bundle_teacher_rollout_vae_training_gpu_metrics"
        ],
    }
    lines.append(
        f"- Official csv-loop full-bundle teacher-rollout conditional action VAE training: "
        f"`{summary['level_c_diffusion']['official_csv_loop_full_bundle_teacher_rollout_vae_training_status']}`; "
        f"summary `{json.dumps(full_bundle_vae_summary, sort_keys=True)}`. "
        "This uses the 40-motion public-bundle teacher rollout source rather than the earlier single-motion "
        "official-loop dataset. It improves local data coverage for the downstream latent pipeline, but the "
        "checkpoint remains a local virtual artifact, not the official BeyondMimic VAE checkpoint or closed-loop "
        "VAE control result."
    )
    importer_vae_worker = summary["level_c_diffusion"][
        "official_importer_export_full_bundle_teacher_rollout_vae_training_worker"
    ]
    importer_vae_summary = {
        "sample_count": importer_vae_worker["dataset"]["sample_count"],
        "motion_time_step_max": importer_vae_worker["dataset"]["motion_time_step_max"],
        "splits": importer_vae_worker["splits"],
        "epochs": importer_vae_worker["training"]["epochs"],
        "latent_dim": importer_vae_worker["training"]["latent_dim"],
        "data_parallel_used": importer_vae_worker["data_parallel_used"],
        "test_action_mse": importer_vae_worker["evaluation"]["test"]["action_mse"],
        "test_action_abs_error_mean": importer_vae_worker["evaluation"]["test"]["action_abs_error_mean"],
        "gpu_metrics_summary": summary["level_c_diffusion"][
            "official_importer_export_full_bundle_teacher_rollout_vae_training_gpu_metrics"
        ],
    }
    lines.append(
        f"- Official-importer-export full-bundle teacher-rollout conditional action VAE training: "
        f"`{summary['level_c_diffusion']['official_importer_export_full_bundle_teacher_rollout_vae_training_status']}`; "
        f"summary `{json.dumps(importer_vae_summary, sort_keys=True)}`. "
        "This trains on the teacher rollout shards collected from the local official-importer-export G1 USDA PPO "
        "checkpoint and the 40-motion public bundle. It is the strongest current local VAE training source on the "
        "more official robot-asset path, but it remains local virtual evidence from a short PPO teacher, not the "
        "official BeyondMimic DAgger/VAE checkpoint and not closed-loop Fig. 5/Fig. 6 guidance."
    )
    importer_vae_assets = summary["level_c_diffusion"]["official_importer_export_full_bundle_vae_assets"]
    lines.append(
        f"- Official-importer-export full-bundle VAE report assets: "
        f"`{importer_vae_assets['status']}`; metrics "
        f"`{json.dumps(importer_vae_assets['metrics'], sort_keys=True)}`; assets "
        f"`{json.dumps(importer_vae_assets['assets'], sort_keys=True)}`. "
        "These provide a report-ready VAE training curve plus split/epoch CSVs while preserving the non-official, "
        "non-real-robot claim level."
    )
    full_bundle_state_worker = summary["level_c_diffusion"][
        "official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset_worker"
    ]
    full_bundle_state_summary = {
        "sample_count": full_bundle_state_worker["dataset"]["sample_count"],
        "window_count": full_bundle_state_worker["dataset"]["window_count"],
        "split_counts": full_bundle_state_worker["dataset"]["split_counts"],
        "sequence_length": full_bundle_state_worker["dataset"]["sequence_length"],
        "obs_dim": full_bundle_state_worker["dataset"]["obs_dim"],
        "latent_dim": full_bundle_state_worker["dataset"]["latent_dim"],
        "token_dim": full_bundle_state_worker["dataset"]["token_dim"],
        "weighted_posterior_reconstruction_mse": full_bundle_state_worker["dataset"][
            "weighted_posterior_reconstruction_mse"
        ],
    }
    lines.append(
        f"- Official csv-loop full-bundle state-latent dataset: "
        f"`{summary['level_c_diffusion']['official_csv_loop_full_bundle_teacher_rollout_state_latent_dataset_status']}`; "
        f"summary `{json.dumps(full_bundle_state_summary, sort_keys=True)}`. "
        "This creates full-bundle 21-step state/action-latent windows from the local VAE posterior. It is a stronger "
        "diffusion-training input than the single-motion chain, but it remains non-official local virtual data."
    )
    full_bundle_diffusion_worker = summary["level_c_diffusion"][
        "official_csv_loop_full_bundle_state_latent_diffusion_training_worker"
    ]
    full_bundle_diffusion_summary = {
        "window_count": full_bundle_diffusion_worker["dataset"]["window_count"],
        "split_counts": full_bundle_diffusion_worker["dataset"]["split_counts"],
        "epochs": full_bundle_diffusion_worker["training"]["epochs"],
        "batch_windows": full_bundle_diffusion_worker["training"]["batch_windows"],
        "data_parallel_used": full_bundle_diffusion_worker["data_parallel_used"],
        "validation_pred_token_mse": full_bundle_diffusion_worker["evaluation"]["validation"]["pred_token_mse"],
        "test_pred_token_mse": full_bundle_diffusion_worker["evaluation"]["test"]["pred_token_mse"],
        "test_noisy_token_mse": full_bundle_diffusion_worker["evaluation"]["test"]["noisy_token_mse"],
        "test_denoising_improvement_ratio": full_bundle_diffusion_worker["evaluation"]["test"][
            "denoising_improvement_ratio"
        ],
        "gpu_metrics_summary": summary["level_c_diffusion"][
            "official_csv_loop_full_bundle_state_latent_diffusion_training_gpu_metrics"
        ],
    }
    lines.append(
        f"- Official csv-loop full-bundle state-latent denoiser training: "
        f"`{summary['level_c_diffusion']['official_csv_loop_full_bundle_state_latent_diffusion_training_status']}`; "
        f"summary `{json.dumps(full_bundle_diffusion_summary, sort_keys=True)}`. "
        "The denoiser now trains on the broader 40-motion state-latent window set and improves over the noisy input "
        "on held-out data. This is an important local pipeline milestone, but still not official diffusion, "
        "TensorRT deployment, closed-loop Fig. 5/Fig. 6 guidance, or real-robot evidence."
    )
    full_bundle_downstream_assets = summary["level_c_diffusion"][
        "official_csv_loop_full_bundle_downstream_assets"
    ]
    lines.append(
        f"- Official csv-loop full-bundle downstream report assets: "
        f"`{full_bundle_downstream_assets['status']}`; metrics "
        f"`{json.dumps(full_bundle_downstream_assets['metrics'], sort_keys=True)}`; assets "
        f"`{json.dumps(full_bundle_downstream_assets['assets'], sort_keys=True)}`. These add report-ready VAE and "
        "diffusion training curves plus split/stage metric tables for the English report and PPT while preserving "
        "the local-virtual, non-paper-level claim boundary."
    )
    importer_state_worker = summary["level_c_diffusion"][
        "official_importer_export_full_bundle_teacher_rollout_state_latent_dataset_worker"
    ]
    importer_state_summary = {
        "sample_count": importer_state_worker["dataset"]["sample_count"],
        "window_count": importer_state_worker["dataset"]["window_count"],
        "split_counts": importer_state_worker["dataset"]["split_counts"],
        "sequence_length": importer_state_worker["dataset"]["sequence_length"],
        "obs_dim": importer_state_worker["dataset"]["obs_dim"],
        "latent_dim": importer_state_worker["dataset"]["latent_dim"],
        "token_dim": importer_state_worker["dataset"]["token_dim"],
        "weighted_posterior_reconstruction_mse": importer_state_worker["dataset"][
            "weighted_posterior_reconstruction_mse"
        ],
        "gpu_metrics_summary": summary["level_c_diffusion"][
            "official_importer_export_full_bundle_teacher_rollout_state_latent_dataset_gpu_metrics"
        ],
    }
    lines.append(
        f"- Official-importer-export full-bundle state-latent dataset: "
        f"`{summary['level_c_diffusion']['official_importer_export_full_bundle_teacher_rollout_state_latent_dataset_status']}`; "
        f"summary `{json.dumps(importer_state_summary, sort_keys=True)}`. "
        "This converts the local official-importer-export teacher rollout and VAE chain into 21-step state/action-latent "
        "windows on GPU 5/6. It is a stronger downstream input on the more official G1 USDA path, but it remains "
        "local virtual data rather than official DAgger/state-latent paper data."
    )
    importer_diffusion_worker = summary["level_c_diffusion"][
        "official_importer_export_full_bundle_state_latent_diffusion_training_worker"
    ]
    importer_diffusion_summary = {
        "window_count": importer_diffusion_worker["dataset"]["window_count"],
        "split_counts": importer_diffusion_worker["dataset"]["split_counts"],
        "epochs": importer_diffusion_worker["training"]["epochs"],
        "batch_windows": importer_diffusion_worker["training"]["batch_windows"],
        "cuda_visible_devices": importer_diffusion_worker["cuda_visible_devices"],
        "data_parallel_used": importer_diffusion_worker["data_parallel_used"],
        "validation_pred_token_mse": importer_diffusion_worker["evaluation"]["validation"]["pred_token_mse"],
        "test_pred_token_mse": importer_diffusion_worker["evaluation"]["test"]["pred_token_mse"],
        "test_noisy_token_mse": importer_diffusion_worker["evaluation"]["test"]["noisy_token_mse"],
        "test_denoising_improvement_ratio": importer_diffusion_worker["evaluation"]["test"][
            "denoising_improvement_ratio"
        ],
        "gpu_metrics_summary": summary["level_c_diffusion"][
            "official_importer_export_full_bundle_state_latent_diffusion_training_gpu_metrics"
        ],
    }
    lines.append(
        f"- Official-importer-export full-bundle state-latent denoiser training: "
        f"`{summary['level_c_diffusion']['official_importer_export_full_bundle_state_latent_diffusion_training_status']}`; "
        f"summary `{json.dumps(importer_diffusion_summary, sort_keys=True)}`. "
        "The denoiser trains for 30 epochs on all importer-export state-latent windows and improves over noisy held-out "
        "tokens. The run uses GPU 5/6 and records low memory use, so it is reported as a local downstream model-training "
        "gate rather than paper-level high-memory PPO or closed-loop guided diffusion evidence."
    )
    importer_downstream_assets = summary["level_c_diffusion"][
        "official_importer_export_full_bundle_downstream_assets"
    ]
    lines.append(
        f"- Official-importer-export full-bundle downstream report assets: "
        f"`{importer_downstream_assets['status']}`; metrics "
        f"`{json.dumps(importer_downstream_assets['metrics'], sort_keys=True)}`; assets "
        f"`{json.dumps(importer_downstream_assets['assets'], sort_keys=True)}`. These add report-ready VAE and "
        "diffusion training curves plus split/stage metric tables for the English report and PPT while preserving "
        "the local-virtual, non-paper-level claim boundary."
    )
    scaled_importer_vae_worker = summary["level_c_diffusion"][
        "official_importer_export_scaled_ppo_teacher_rollout_vae_training_worker"
    ]
    scaled_importer_vae_summary = {
        "sample_count": scaled_importer_vae_worker["dataset"]["sample_count"],
        "motion_time_step_max": scaled_importer_vae_worker["dataset"]["motion_time_step_max"],
        "splits": scaled_importer_vae_worker["splits"],
        "epochs": scaled_importer_vae_worker["training"]["epochs"],
        "latent_dim": scaled_importer_vae_worker["training"]["latent_dim"],
        "data_parallel_used": scaled_importer_vae_worker["data_parallel_used"],
        "test_action_mse": scaled_importer_vae_worker["evaluation"]["test"]["action_mse"],
        "test_action_abs_error_mean": scaled_importer_vae_worker["evaluation"]["test"][
            "action_abs_error_mean"
        ],
        "gpu_metrics_summary": summary["level_c_diffusion"][
            "official_importer_export_scaled_ppo_teacher_rollout_vae_training_gpu_metrics"
        ],
    }
    lines.append(
        f"- Official-importer-export scaled PPO teacher-rollout conditional action VAE training: "
        f"`{summary['level_c_diffusion']['official_importer_export_scaled_ppo_teacher_rollout_vae_training_status']}`; "
        f"summary `{json.dumps(scaled_importer_vae_summary, sort_keys=True)}`. "
        "This retrains the local action VAE from the iteration-999 scaled PPO teacher rollout dataset rather than "
        "the older iteration-299 teacher-rollout candidate. It is stronger local downstream evidence, but it remains "
        "a local virtual checkpoint, not official BeyondMimic DAgger/VAE evidence and not closed-loop guidance."
    )
    scaled_importer_state_worker = summary["level_c_diffusion"][
        "official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset_worker"
    ]
    scaled_importer_state_summary = {
        "sample_count": scaled_importer_state_worker["dataset"]["sample_count"],
        "window_count": scaled_importer_state_worker["dataset"]["window_count"],
        "split_counts": scaled_importer_state_worker["dataset"]["split_counts"],
        "sequence_length": scaled_importer_state_worker["dataset"]["sequence_length"],
        "obs_dim": scaled_importer_state_worker["dataset"]["obs_dim"],
        "latent_dim": scaled_importer_state_worker["dataset"]["latent_dim"],
        "token_dim": scaled_importer_state_worker["dataset"]["token_dim"],
        "weighted_posterior_reconstruction_mse": scaled_importer_state_worker["dataset"][
            "weighted_posterior_reconstruction_mse"
        ],
        "gpu_metrics_summary": summary["level_c_diffusion"][
            "official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset_gpu_metrics"
        ],
    }
    lines.append(
        f"- Official-importer-export scaled PPO state-latent dataset: "
        f"`{summary['level_c_diffusion']['official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset_status']}`; "
        f"summary `{json.dumps(scaled_importer_state_summary, sort_keys=True)}`. "
        "This converts 1,224,704 scaled-teacher samples into 1,142,784 21-step state/action-latent windows on "
        "GPU 4/7. It is now the largest local official-importer-export downstream dataset in this workspace, but "
        "it remains local virtual data rather than official DAgger/state-latent paper data."
    )
    scaled_importer_diffusion_worker = summary["level_c_diffusion"][
        "official_importer_export_scaled_ppo_state_latent_diffusion_training_worker"
    ]
    scaled_importer_diffusion_summary = {
        "window_count": scaled_importer_diffusion_worker["dataset"]["window_count"],
        "split_counts": scaled_importer_diffusion_worker["dataset"]["split_counts"],
        "epochs": scaled_importer_diffusion_worker["training"]["epochs"],
        "batch_windows": scaled_importer_diffusion_worker["training"]["batch_windows"],
        "cuda_visible_devices": scaled_importer_diffusion_worker["cuda_visible_devices"],
        "data_parallel_used": scaled_importer_diffusion_worker["data_parallel_used"],
        "validation_pred_token_mse": scaled_importer_diffusion_worker["evaluation"]["validation"][
            "pred_token_mse"
        ],
        "test_pred_token_mse": scaled_importer_diffusion_worker["evaluation"]["test"]["pred_token_mse"],
        "test_noisy_token_mse": scaled_importer_diffusion_worker["evaluation"]["test"]["noisy_token_mse"],
        "test_denoising_improvement_ratio": scaled_importer_diffusion_worker["evaluation"]["test"][
            "denoising_improvement_ratio"
        ],
        "gpu_metrics_summary": summary["level_c_diffusion"][
            "official_importer_export_scaled_ppo_state_latent_diffusion_training_gpu_metrics"
        ],
    }
    lines.append(
        f"- Official-importer-export scaled PPO state-latent denoiser training: "
        f"`{summary['level_c_diffusion']['official_importer_export_scaled_ppo_state_latent_diffusion_training_status']}`; "
        f"summary `{json.dumps(scaled_importer_diffusion_summary, sort_keys=True)}`. "
        "The denoiser trains for 30 epochs on the larger scaled-teacher state-latent window set and reaches a "
        "positive held-out denoising improvement over noisy tokens. This is the strongest current local downstream "
        "training result on the recovered official-importer-export asset path, but it is still not the official "
        "BeyondMimic diffusion checkpoint, not TensorRT deployment, not closed-loop Fig. 5/Fig. 6 guidance, and "
        "not real-robot evidence."
    )
    scaled_importer_downstream_assets = summary["level_c_diffusion"][
        "official_importer_export_scaled_ppo_downstream_assets"
    ]
    lines.append(
        f"- Official-importer-export scaled PPO downstream report assets: "
        f"`{scaled_importer_downstream_assets['status']}`; metrics "
        f"`{json.dumps(scaled_importer_downstream_assets['metrics'], sort_keys=True)}`; assets "
        f"`{json.dumps(scaled_importer_downstream_assets['assets'], sort_keys=True)}`. These add report-ready VAE and "
        "diffusion training curves plus split/stage metric tables for the English report and PPT while preserving "
        "the local-virtual, non-paper-level claim boundary."
    )
    scaled_importer_onnx_async_summary = {
        "status": summary["level_c_diffusion"][
            "official_importer_export_scaled_ppo_vae_denoiser_onnx_async_status"
        ],
        "providers": summary["level_c_diffusion"][
            "official_importer_export_scaled_ppo_vae_denoiser_onnx_async_settings"
        ]["onnxruntime_available_providers"],
        "providers_used": summary["level_c_diffusion"][
            "official_importer_export_scaled_ppo_vae_denoiser_onnx_async_settings"
        ]["onnxruntime_execution_providers_used"],
        "consistency": summary["level_c_diffusion"][
            "official_importer_export_scaled_ppo_vae_denoiser_onnx_async_consistency"
        ],
        "async_summary": summary["level_c_diffusion"][
            "official_importer_export_scaled_ppo_vae_denoiser_onnx_async_summary"
        ],
        "outputs": summary["level_c_diffusion"][
            "official_importer_export_scaled_ppo_vae_denoiser_onnx_async_outputs"
        ],
    }
    lines.append(
        f"- Official-importer-export scaled PPO VAE/denoiser ONNXRuntime async deployment-path audit: "
        f"`{summary['level_c_diffusion']['official_importer_export_scaled_ppo_vae_denoiser_onnx_async_status']}`; "
        f"summary `{json.dumps(scaled_importer_onnx_async_summary, sort_keys=True)}`. "
        "This repeats the local export/runtime audit on the larger iteration-999 scaled PPO downstream VAE and "
        "state-latent denoiser chain. ONNXRuntime CPU outputs match PyTorch within micro absolute error and the "
        "thread-pool async proxy records local throughput improvement. CUDAExecutionProvider and TensorRT remain "
        "unavailable in the local ORT build, so this is not paper Mini-PC TensorRT latency, not an official "
        "BeyondMimic checkpoint, not CppAD guidance, not live IsaacLab deployment, and not real-robot evidence."
    )
    official_loop_guidance_worker = summary["level_c_diffusion"][
        "official_csv_loop_state_latent_guidance_eval_worker"
    ]
    official_loop_guidance_summary = {
        "total_selected_windows": official_loop_guidance_worker["metrics"]["total_selected_windows"],
        "selected_split_counts": official_loop_guidance_worker["settings"]["selected_split_counts"],
        "row_count": official_loop_guidance_worker["metrics"]["row_count"],
        "tasks": official_loop_guidance_worker["settings"]["tasks"],
        "scales": official_loop_guidance_worker["settings"]["scales"],
        "tasks_with_all_best_costs_improve": official_loop_guidance_worker["metrics"][
            "tasks_with_all_best_costs_improve"
        ],
        "tasks_with_nonzero_best_gradients": official_loop_guidance_worker["metrics"][
            "tasks_with_nonzero_best_gradients"
        ],
        "task_mean_best_cost_delta": {
            task: task_summary["mean_best_cost_delta"]
            for task, task_summary in official_loop_guidance_worker["task_summaries"].items()
        },
        "gpu_metrics_summary": summary["level_c_diffusion"][
            "official_csv_loop_state_latent_guidance_eval_gpu_metrics"
        ],
    }
    lines.append(
        f"- Official csv-loop full-split offline state-latent guidance eval: "
        f"`{summary['level_c_diffusion']['official_csv_loop_state_latent_guidance_eval_status']}`; "
        f"summary `{json.dumps(official_loop_guidance_summary, sort_keys=True)}`. "
        "This evaluates guidance over all validation/test windows from the official-loop local denoiser and confirms "
        "positive best-scale cost deltas for all four offline tasks. It remains an offline surrogate, not an IsaacLab "
        "closed-loop rollout, TensorRT deployment, or Fig. 5/Fig. 6 paper-level result."
    )
    full_bundle_guidance_worker = summary["level_c_diffusion"][
        "official_csv_loop_full_bundle_state_latent_guidance_eval_worker"
    ]
    full_bundle_guidance_summary = {
        "total_selected_windows": full_bundle_guidance_worker["metrics"]["total_selected_windows"],
        "selected_split_counts": full_bundle_guidance_worker["settings"]["selected_split_counts"],
        "row_count": full_bundle_guidance_worker["metrics"]["row_count"],
        "tasks": full_bundle_guidance_worker["settings"]["tasks"],
        "scales": full_bundle_guidance_worker["settings"]["scales"],
        "tasks_with_all_best_costs_improve": full_bundle_guidance_worker["metrics"][
            "tasks_with_all_best_costs_improve"
        ],
        "tasks_with_nonzero_best_gradients": full_bundle_guidance_worker["metrics"][
            "tasks_with_nonzero_best_gradients"
        ],
        "task_mean_best_cost_delta": {
            task: task_summary["mean_best_cost_delta"]
            for task, task_summary in full_bundle_guidance_worker["task_summaries"].items()
        },
        "gpu_metrics_summary": summary["level_c_diffusion"][
            "official_csv_loop_full_bundle_state_latent_guidance_eval_gpu_metrics"
        ],
    }
    lines.append(
        f"- Official csv-loop full-bundle full-split offline state-latent guidance eval: "
        f"`{summary['level_c_diffusion']['official_csv_loop_full_bundle_state_latent_guidance_eval_status']}`; "
        f"summary `{json.dumps(full_bundle_guidance_summary, sort_keys=True)}`. "
        "This repeats offline guidance on the 40-motion full-bundle denoiser and evaluates every validation/test "
        "window. All four proxy tasks improve at their best guidance scale. It is an important local guidance "
        "milestone, but it is still offline and does not claim IsaacLab closed-loop Fig. 5/Fig. 6 reproduction."
    )
    full_bundle_guidance_assets = summary["level_c_diffusion"]["official_csv_loop_full_bundle_guidance_assets"]
    lines.append(
        f"- Official csv-loop full-bundle guidance report assets: "
        f"`{full_bundle_guidance_assets['status']}`; metrics "
        f"`{json.dumps(full_bundle_guidance_assets['metrics'], sort_keys=True)}`; assets "
        f"`{json.dumps(full_bundle_guidance_assets['assets'], sort_keys=True)}`. These provide report-ready "
        "best-cost-delta and scale-response figures plus CSV tables for the English report/PPT without promoting "
        "the result to closed-loop or paper-level guidance."
    )
    importer_guidance_worker = summary["level_c_diffusion"][
        "official_importer_export_full_bundle_state_latent_guidance_eval_worker"
    ]
    importer_guidance_summary = {
        "total_selected_windows": importer_guidance_worker["metrics"]["total_selected_windows"],
        "selected_split_counts": importer_guidance_worker["settings"]["selected_split_counts"],
        "row_count": importer_guidance_worker["metrics"]["row_count"],
        "tasks": importer_guidance_worker["settings"]["tasks"],
        "scales": importer_guidance_worker["settings"]["scales"],
        "tasks_with_all_best_costs_improve": importer_guidance_worker["metrics"][
            "tasks_with_all_best_costs_improve"
        ],
        "tasks_with_nonzero_best_gradients": importer_guidance_worker["metrics"][
            "tasks_with_nonzero_best_gradients"
        ],
        "task_mean_best_cost_delta": {
            task: task_summary["mean_best_cost_delta"]
            for task, task_summary in importer_guidance_worker["task_summaries"].items()
        },
        "gpu_metrics_summary": summary["level_c_diffusion"][
            "official_importer_export_full_bundle_state_latent_guidance_eval_gpu_metrics"
        ],
    }
    lines.append(
        f"- Official-importer-export full-bundle full-split offline state-latent guidance eval: "
        f"`{summary['level_c_diffusion']['official_importer_export_full_bundle_state_latent_guidance_eval_status']}`; "
        f"summary `{json.dumps(importer_guidance_summary, sort_keys=True)}`. "
        "This repeats the full validation/test offline guidance gate on the recovered official-importer-export G1 "
        "USDA downstream chain. It confirms positive best-scale proxy-cost deltas for all four offline tasks over "
        "the local 40-motion denoiser outputs, but it remains offline task-cost guidance rather than closed-loop "
        "IsaacLab control, paper Fig. 5/Fig. 6, TensorRT/asynchronous deployment, or real-robot evidence."
    )
    scaled_importer_guidance_worker = summary["level_c_diffusion"][
        "official_importer_export_scaled_ppo_state_latent_guidance_eval_worker"
    ]
    scaled_importer_guidance_summary = {
        "total_selected_windows": scaled_importer_guidance_worker["metrics"]["total_selected_windows"],
        "selected_split_counts": scaled_importer_guidance_worker["settings"]["selected_split_counts"],
        "row_count": scaled_importer_guidance_worker["metrics"]["row_count"],
        "tasks": scaled_importer_guidance_worker["settings"]["tasks"],
        "scales": scaled_importer_guidance_worker["settings"]["scales"],
        "tasks_with_all_best_costs_improve": scaled_importer_guidance_worker["metrics"][
            "tasks_with_all_best_costs_improve"
        ],
        "tasks_with_nonzero_best_gradients": scaled_importer_guidance_worker["metrics"][
            "tasks_with_nonzero_best_gradients"
        ],
        "task_mean_best_cost_delta": {
            task: task_summary["mean_best_cost_delta"]
            for task, task_summary in scaled_importer_guidance_worker["task_summaries"].items()
        },
        "assets": summary["level_c_diffusion"]["official_importer_export_scaled_ppo_guidance_assets"][
            "assets"
        ],
        "gpu_metrics_summary": summary["level_c_diffusion"][
            "official_importer_export_scaled_ppo_state_latent_guidance_eval_gpu_metrics"
        ],
    }
    lines.append(
        f"- Official-importer-export scaled PPO full-split offline state-latent guidance eval: "
        f"`{summary['level_c_diffusion']['official_importer_export_scaled_ppo_state_latent_guidance_eval_status']}`; "
        f"summary `{json.dumps(scaled_importer_guidance_summary, sort_keys=True)}`. "
        "This reruns the full validation/test offline guidance gate on the larger iteration-999 scaled PPO "
        "teacher-rollout downstream chain. It evaluates 228,557 windows and confirms positive best-scale proxy-cost "
        "deltas for all four offline tasks, with report-ready cost-delta and scale-response assets. It is a stronger "
        "offline prerequisite for the next closed-loop guidance round, but it is not itself closed-loop IsaacLab "
        "control, paper Fig. 5/Fig. 6, TensorRT/asynchronous deployment, or real-robot evidence."
    )
    guided_decode_worker = summary["level_c_diffusion"][
        "official_csv_loop_guidance_vae_action_decode_eval_worker"
    ]
    guided_decode_summary = {
        "total_windows": guided_decode_worker["metrics"]["total_windows"],
        "total_action_steps_per_task": guided_decode_worker["metrics"]["total_action_steps_per_task"],
        "tasks_with_finite_actions": guided_decode_worker["metrics"]["tasks_with_finite_actions"],
        "task_mean_guided_base_action_l2": {
            task: task_summary["mean_guided_base_action_l2"]
            for task, task_summary in guided_decode_worker["task_summaries"].items()
        },
        "task_mean_guided_minus_base_teacher_mse": {
            task: task_summary["mean_guided_minus_base_teacher_mse"]
            for task, task_summary in guided_decode_worker["task_summaries"].items()
        },
        "assets": summary["level_c_diffusion"]["official_csv_loop_guidance_vae_action_decode_assets"]["assets"],
        "gpu_metrics_summary": summary["level_c_diffusion"][
            "official_csv_loop_guidance_vae_action_decode_eval_gpu_metrics"
        ],
    }
    lines.append(
        f"- Official csv-loop guided latent to VAE action decode eval: "
        f"`{summary['level_c_diffusion']['official_csv_loop_guidance_vae_action_decode_eval_status']}`; "
        f"summary `{json.dumps(guided_decode_summary, sort_keys=True)}`. "
        "This decodes guided denoiser outputs into finite 29D actions over all validation/test windows and generates "
        "report-ready guided-vs-base action plots under `res/report_assets`. It is still offline action decoding, "
        "not closed-loop IsaacLab control."
    )
    guided_action_probe_metrics = summary["level_c_diffusion"][
        "official_csv_loop_guided_action_rollout_probe_metrics"
    ]
    guided_action_probe_summary = {
        "rollout_steps": guided_action_probe_metrics["rollout_steps"],
        "base_guided_max_abs_action_delta": guided_action_probe_metrics["base_guided_max_abs_action_delta"],
        "base_guided_l2_mean": guided_action_probe_metrics["base_guided_l2_mean"],
        "base_teacher_mse": guided_action_probe_metrics["base_teacher_mse"],
        "guided_teacher_mse": guided_action_probe_metrics["guided_teacher_mse"],
        "variant_metrics": guided_action_probe_metrics["variant_metrics"],
        "assets": summary["level_c_diffusion"][
            "official_csv_loop_guided_action_rollout_probe_assets"
        ]["assets"],
    }
    lines.append(
        f"- Official csv-loop decoded-action IsaacLab rollout probe: "
        f"`{summary['level_c_diffusion']['official_csv_loop_guided_action_rollout_probe_status']}`; "
        f"summary `{json.dumps(guided_action_probe_summary, sort_keys=True)}`. "
        "This executes one short 21-step base/guided/teacher decoded-action sample in the resource-adjusted "
        "Tracking-Flat-G1-v0 task and records reward/done/body-error traces. It validates the action-to-sim bridge "
        "but is not receding-horizon diffusion guidance, not Fig. 5/Fig. 6 reproduction, and the sampled base/guided "
        "actions are numerically identical, so it is a negative result for guided behavior change."
    )
    action_guidance_metrics = summary["level_c_diffusion"][
        "official_csv_loop_action_guidance_rollout_eval_metrics"
    ]
    action_guidance_asset = summary["level_c_diffusion"][
        "official_csv_loop_action_guidance_rollout_asset"
    ]
    action_guidance_summary = {
        "rollout_steps": action_guidance_metrics["rollout_steps"],
        "guidance": action_guidance_metrics["guidance"],
        "variant_metrics": action_guidance_metrics["variant_metrics"],
        "assets": action_guidance_asset["assets"],
    }
    lines.append(
        f"- Official csv-loop local action-guidance closed-loop rollout: "
        f"`{summary['level_c_diffusion']['official_csv_loop_action_guidance_rollout_eval_status']}`; "
        f"summary `{json.dumps(action_guidance_summary, sort_keys=True)}`. "
        "This runs 299-step teacher, VAE-base, and teacher-consistency action-guided variants in the local "
        "resource-adjusted Tracking-Flat-G1-v0 task and produces an MP4/keyframe/metric plot under "
        "`res/visualization`. It demonstrates a closed-loop action-guidance bridge with visible behavior traces, "
        "but it is not the paper receding-horizon latent diffusion controller, not the official BeyondMimic "
        "diffusion checkpoint, not Fig. 5/Fig. 6 reproduction, and not real-robot evidence."
    )
    receding_guidance_metrics = summary["level_c_diffusion"][
        "official_csv_loop_receding_latent_guidance_rollout_eval_metrics"
    ]
    receding_guidance_asset = summary["level_c_diffusion"][
        "official_csv_loop_receding_latent_guidance_rollout_asset"
    ]
    receding_guidance_summary = {
        "rollout_steps": receding_guidance_metrics["rollout_steps"],
        "guidance": receding_guidance_metrics["guidance"],
        "variant_metrics": receding_guidance_metrics["variant_metrics"],
        "assets": receding_guidance_asset["assets"],
    }
    lines.append(
        f"- Official csv-loop local receding-horizon latent-guidance closed-loop rollout: "
        f"`{summary['level_c_diffusion']['official_csv_loop_receding_latent_guidance_rollout_eval_status']}`; "
        f"summary `{json.dumps(receding_guidance_summary, sort_keys=True)}`. "
        "This runs 299-step teacher, VAE-base, denoised-latent, and guided-latent variants in the local "
        "resource-adjusted Tracking-Flat-G1-v0 task and produces an MP4/keyframe/metric plot under "
        "`res/visualization`. The guided-latent variant recomputes a 21-step state-latent horizon every control "
        "step, applies the local denoiser and one composed-cost guidance update, decodes the current latent through "
        "the local VAE, and executes the action. This is the strongest current local bridge toward paper guided "
        "diffusion, but it is still not the official BeyondMimic checkpoint, not paper Fig. 5/Fig. 6 task "
        "reproduction, not TensorRT/asynchronous deployment, and not real-robot evidence."
    )
    full_bundle_receding_metrics = summary["level_c_diffusion"][
        "official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval_metrics"
    ]
    full_bundle_receding_asset = summary["level_c_diffusion"][
        "official_csv_loop_full_bundle_receding_latent_guidance_rollout_asset"
    ]
    full_bundle_receding_summary = {
        "bundle": summary["level_c_diffusion"][
            "official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval_bundle"
        ],
        "rollout_steps": full_bundle_receding_metrics["rollout_steps"],
        "guidance": full_bundle_receding_metrics["guidance"],
        "variant_metrics": full_bundle_receding_metrics["variant_metrics"],
        "assets": full_bundle_receding_asset["assets"],
    }
    lines.append(
        f"- Official csv-loop full-bundle receding-horizon latent-guidance closed-loop rollout: "
        f"`{summary['level_c_diffusion']['official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval_status']}`; "
        f"summary `{json.dumps(full_bundle_receding_summary, sort_keys=True)}`. "
        "This reruns the receding-latent closed-loop bridge with the 40-motion public official-csv-loop bundle and "
        "the matching local full-bundle PPO/VAE/denoiser artifacts, producing MP4, keyframes, metrics plot, CSV, "
        "GPU telemetry, and JSON evidence. It is the strongest current simulation-side guidance video artifact, "
        "but it remains local virtual/resource-adjusted evidence rather than official Fig. 5/Fig. 6, TensorRT, or "
        "real-robot reproduction."
    )
    full_bundle_task_rows = summary["level_c_diffusion"][
        "official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval_rows"
    ]
    full_bundle_task_summary = {
        row["task"]: {
            "rollout_steps": row["rollout_steps"],
            "reward_mean": row["guided_reward_mean"],
            "target_body_error_mean": row["guided_target_body_error_mean"],
            "guidance_cost_delta_mean": row["guidance_cost_delta_mean"],
            "mp4": row["mp4"],
        }
        for row in full_bundle_task_rows
    }
    lines.append(
        f"- Official csv-loop full-bundle task-conditioned latent-guidance closed-loop rollouts: "
        f"`{summary['level_c_diffusion']['official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval_status']}`; "
        f"bundle `{json.dumps(summary['level_c_diffusion']['official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval_bundle'], sort_keys=True)}`; "
        f"summary `{json.dumps(full_bundle_task_summary, sort_keys=True)}`. "
        "This extends the full-bundle guidance bridge from one composed-cost rollout to joystick, waypoint, "
        "obstacle_avoidance, and composed proxy tasks. Each task runs 299 IsaacLab steps and saves JSON/TSV "
        "metrics plus MP4/keyframes/plot/CSV visual evidence. The boundary remains explicit: these are local "
        "virtual task proxies using local full-bundle checkpoints and an enriched USD scaffold, not official "
        "BeyondMimic Fig. 5/Fig. 6 success-rate reproduction, TensorRT deployment, or real-robot validation."
    )
    full_bundle_task_assets = summary["level_c_diffusion"][
        "official_csv_loop_full_bundle_task_conditioned_guidance_summary_assets"
    ]
    lines.append(
        f"- Official csv-loop full-bundle task-conditioned guidance report assets: "
        f"`{full_bundle_task_assets['status']}`; assets "
        f"`{json.dumps(full_bundle_task_assets['assets'], sort_keys=True)}`. "
        "These compact plots and CSVs make the new full-bundle four-task rollouts easier to cite in the English "
        "reading report while preserving the qualitative-only claim level."
    )
    full_bundle_task_multiseed_metrics = summary["level_c_diffusion"][
        "official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval_metrics"
    ]
    full_bundle_task_multiseed_checks = summary["level_c_diffusion"][
        "official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval_checks"
    ]
    full_bundle_task_multiseed_aggregate = {
        row["task"]: {
            "seed_count": row["seed_count"],
            "guided_reward_mean": row["guided_reward_mean_mean"],
            "guided_reward_std": row["guided_reward_mean_std"],
            "guided_target_body_error_mean": row["guided_target_body_error_mean_mean"],
            "guided_target_body_error_std": row["guided_target_body_error_mean_std"],
            "guided_done_count_total_mean": row["guided_done_count_total_mean"],
            "guidance_cost_delta_mean": row["guidance_cost_delta_mean_mean"],
        }
        for row in summary["level_c_diffusion"][
            "official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval_aggregate"
        ]
    }
    lines.append(
        f"- Official csv-loop full-bundle task-conditioned latent-guidance multi-seed closed-loop rollouts: "
        f"`{summary['level_c_diffusion']['official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval_status']}`; "
        f"bundle `{json.dumps(summary['level_c_diffusion']['official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval_bundle'], sort_keys=True)}`; "
        f"metrics `{json.dumps(full_bundle_task_multiseed_metrics, sort_keys=True)}`, "
        f"checks `{json.dumps(full_bundle_task_multiseed_checks, sort_keys=True)}`, "
        f"aggregate `{json.dumps(full_bundle_task_multiseed_aggregate, sort_keys=True)}`. "
        "This aggregates three seed groups over joystick, waypoint, obstacle_avoidance, and composed proxy tasks "
        "for 12 local closed-loop IsaacLab rollouts over the 40-motion public official-csv-loop bundle. It is a "
        "stronger paper-facing robustness bridge than the previous single full-bundle task-conditioned run, but it "
        "is still qualitative-only local virtual evidence using local checkpoints, proxy costs, and an enriched USD "
        "runtime scaffold, not official BeyondMimic Fig. 5/Fig. 6 success metrics, TensorRT deployment, or "
        "real-robot validation."
    )
    full_bundle_task_multiseed_assets = summary["level_c_diffusion"][
        "official_csv_loop_full_bundle_task_conditioned_guidance_multiseed_assets"
    ]
    lines.append(
        f"- Official csv-loop full-bundle task-conditioned guidance multi-seed report assets: "
        f"`{full_bundle_task_multiseed_assets['status']}`; assets "
        f"`{json.dumps(full_bundle_task_multiseed_assets['assets'], sort_keys=True)}`. "
        "These figures and CSVs summarize the full-bundle multi-seed reward/error/done-count/guidance-cost "
        "statistics for direct use in the English reading report and PPT without promoting the result to "
        "paper-level reproduction."
    )
    task_conditioned_rows = summary["level_c_diffusion"][
        "official_csv_loop_task_conditioned_latent_guidance_rollout_eval_rows"
    ]
    task_conditioned_summary = {
        row["task"]: {
            "reward_mean": row["guided_reward_mean"],
            "target_body_error_mean": row["guided_target_body_error_mean"],
            "guidance_cost_delta_mean": row["guidance_cost_delta_mean"],
            "mp4": row["mp4"],
        }
        for row in task_conditioned_rows
    }
    lines.append(
        f"- Official csv-loop local task-conditioned latent-guidance closed-loop rollouts: "
        f"`{summary['level_c_diffusion']['official_csv_loop_task_conditioned_latent_guidance_rollout_eval_status']}`; "
        f"summary `{json.dumps(task_conditioned_summary, sort_keys=True)}`. "
        "This runs joystick, waypoint, obstacle_avoidance, and composed proxy guidance tasks for 299 IsaacLab "
        "steps each, comparing teacher, VAE-base, denoised-latent, and receding guided-latent variants and saving "
        "MP4/keyframes/plots/CSV assets. It gives the English report visible task-conditioned guided-control "
        "evidence, but it uses local proxy costs and local checkpoints rather than official BeyondMimic Fig. 5/"
        "Fig. 6 evaluation, TensorRT/asynchronous deployment, or real robot evidence."
    )
    task_conditioned_assets = summary["level_c_diffusion"][
        "official_csv_loop_task_conditioned_guidance_summary_assets"
    ]
    lines.append(
        f"- Official csv-loop task-conditioned guidance report assets: `{task_conditioned_assets['status']}`; "
        f"assets `{json.dumps(task_conditioned_assets['assets'], sort_keys=True)}`. "
        "These aggregate the four task rollouts into an overview figure, a guidance-cost/tracking-error tradeoff "
        "figure, guided summary CSV, and full metrics CSV for direct use in the English report and PPT."
    )
    task_conditioned_multiseed_metrics = summary["level_c_diffusion"][
        "official_csv_loop_task_conditioned_latent_guidance_multiseed_eval_metrics"
    ]
    task_conditioned_multiseed_checks = summary["level_c_diffusion"][
        "official_csv_loop_task_conditioned_latent_guidance_multiseed_eval_checks"
    ]
    task_conditioned_multiseed_aggregate = {
        row["task"]: {
            "seed_count": row["seed_count"],
            "guided_reward_mean": row["guided_reward_mean_mean"],
            "guided_reward_std": row["guided_reward_mean_std"],
            "guided_target_body_error_mean": row["guided_target_body_error_mean_mean"],
            "guided_target_body_error_std": row["guided_target_body_error_mean_std"],
            "guided_done_count_total_mean": row["guided_done_count_total_mean"],
            "guidance_cost_delta_mean": row["guidance_cost_delta_mean_mean"],
        }
        for row in summary["level_c_diffusion"][
            "official_csv_loop_task_conditioned_latent_guidance_multiseed_eval_aggregate"
        ]
    }
    lines.append(
        f"- Official csv-loop local task-conditioned latent-guidance multi-seed closed-loop rollouts: "
        f"`{summary['level_c_diffusion']['official_csv_loop_task_conditioned_latent_guidance_multiseed_eval_status']}`; "
        f"metrics `{json.dumps(task_conditioned_multiseed_metrics, sort_keys=True)}`, "
        f"checks `{json.dumps(task_conditioned_multiseed_checks, sort_keys=True)}`, "
        f"aggregate `{json.dumps(task_conditioned_multiseed_aggregate, sort_keys=True)}`. "
        "This aggregates three seed groups over joystick, waypoint, obstacle_avoidance, and composed proxy tasks "
        "for 12 local closed-loop IsaacLab rollouts and 14352 variant control steps. It adds robustness and "
        "presentation-ready MP4/keyframe/plot evidence beyond the prior single-seed task-conditioned bridge. It is "
        "still local virtual evidence using local checkpoints, proxy costs, and the enriched USD scaffold, not "
        "official BeyondMimic Fig. 5/Fig. 6 reproduction, TensorRT/asynchronous deployment, or real-robot evidence."
    )
    task_conditioned_multiseed_assets = summary["level_c_diffusion"][
        "official_csv_loop_task_conditioned_guidance_multiseed_assets"
    ]
    lines.append(
        f"- Official csv-loop task-conditioned guidance multi-seed report assets: "
        f"`{task_conditioned_multiseed_assets['status']}`; assets "
        f"`{json.dumps(task_conditioned_multiseed_assets['assets'], sort_keys=True)}`. "
        "These figures summarize guided reward/error/done-count/guidance-cost statistics and per-seed "
        "reward/error scatter for the English report and PPT."
    )
    vae_closed_loop_run = summary["level_c_diffusion"][
        "official_csv_loop_vae_closed_loop_rollout_eval_run"
    ]
    vae_closed_loop_aggregate = vae_closed_loop_run["aggregate_metrics"]
    vae_closed_loop_summary = {
        "total_num_envs": vae_closed_loop_aggregate["total_num_envs"],
        "rollout_steps": vae_closed_loop_aggregate["rollout_steps"],
        "total_env_steps": vae_closed_loop_aggregate["total_env_steps"],
        "duration_seconds": vae_closed_loop_run["duration_seconds"],
        "reward_mean": vae_closed_loop_aggregate["reward_mean"]["mean"],
        "done_count_total": vae_closed_loop_aggregate["done_count_total"],
        "timeout_count_total": vae_closed_loop_aggregate["timeout_count_total"],
        "teacher_vae_action_mse_mean": vae_closed_loop_aggregate["teacher_vae_action_mse"]["mean"],
        "teacher_vae_action_abs_error_mean": vae_closed_loop_aggregate[
            "teacher_vae_action_abs_error"
        ]["mean"],
        "gpu_metrics_summary": vae_closed_loop_run["gpu_metrics_summary"],
        "peak_memory_each_gpu_at_least_10gb": summary["level_c_diffusion"][
            "official_csv_loop_vae_closed_loop_rollout_eval_checks"
        ]["peak_memory_each_gpu_at_least_10gb"],
        "assets": summary["level_c_diffusion"]["official_csv_loop_vae_closed_loop_rollout_assets"]["assets"],
    }
    lines.append(
        f"- Official csv-loop VAE action-reconstruction closed-loop rollout eval: "
        f"`{summary['level_c_diffusion']['official_csv_loop_vae_closed_loop_rollout_eval_status']}`; "
        f"summary `{json.dumps(vae_closed_loop_summary, sort_keys=True)}`. "
        "This executes a full 299-step, two-rank IsaacLab rollout where the local PPO teacher action is encoded "
        "and decoded by the local official-csv-loop conditional action VAE before stepping the environment. It is "
        "stronger than the short decoded-action bridge because it covers 612352 simulated env steps. It remains a "
        "local virtual VAE action-reconstruction evaluation, not the official BeyondMimic VAE checkpoint, not an "
        "autonomous VAE rollout policy, not receding-horizon guided diffusion, not Fig. 5/Fig. 6 reproduction, and "
        "not real-robot evidence. GPU telemetry is kept honest: GPU4 exceeded 10GB peak memory, while GPU7 did not."
    )
    importer_vae_closed_loop_run = summary["level_c_diffusion"][
        "official_importer_export_full_bundle_vae_closed_loop_rollout_eval_run"
    ]
    importer_vae_closed_loop_aggregate = importer_vae_closed_loop_run["aggregate_metrics"]
    importer_vae_closed_loop_summary = {
        "total_num_envs": importer_vae_closed_loop_aggregate["total_num_envs"],
        "rollout_steps": importer_vae_closed_loop_aggregate["rollout_steps"],
        "total_env_steps": importer_vae_closed_loop_aggregate["total_env_steps"],
        "duration_seconds": importer_vae_closed_loop_run["duration_seconds"],
        "reward_mean": importer_vae_closed_loop_aggregate["reward_mean"]["mean"],
        "done_count_total": importer_vae_closed_loop_aggregate["done_count_total"],
        "timeout_count_total": importer_vae_closed_loop_aggregate["timeout_count_total"],
        "teacher_vae_action_mse_mean": importer_vae_closed_loop_aggregate["teacher_vae_action_mse"]["mean"],
        "teacher_vae_action_abs_error_mean": importer_vae_closed_loop_aggregate[
            "teacher_vae_action_abs_error"
        ]["mean"],
        "gpu_metrics_summary": importer_vae_closed_loop_run["gpu_metrics_summary"],
        "peak_memory_each_gpu_at_least_10gb": summary["level_c_diffusion"][
            "official_importer_export_full_bundle_vae_closed_loop_rollout_eval_checks"
        ]["peak_memory_each_gpu_at_least_10gb"],
        "assets": summary["level_c_diffusion"][
            "official_importer_export_full_bundle_vae_closed_loop_rollout_assets"
        ]["assets"],
    }
    lines.append(
        f"- Official-importer-export full-bundle VAE action-reconstruction closed-loop rollout eval: "
        f"`{summary['level_c_diffusion']['official_importer_export_full_bundle_vae_closed_loop_rollout_eval_status']}`; "
        f"summary `{json.dumps(importer_vae_closed_loop_summary, sort_keys=True)}`. "
        "This executes a full 299-step, two-rank IsaacLab rollout on the official-importer-export G1 USDA where the "
        "local 40-motion PPO teacher action is encoded and decoded by the local full-bundle conditional action VAE "
        "before stepping the environment. It covers 918528 simulated env steps and records report-ready PNG/CSV "
        "assets. It remains local virtual evidence: the source teacher is a short local PPO checkpoint, all env-step "
        "done counts are explicitly recorded, per-GPU memory stayed below 10GB, and it is not the official "
        "BeyondMimic VAE checkpoint, autonomous VAE policy, guided diffusion, Fig. 5/Fig. 6 reproduction, TensorRT, "
        "or real-robot evidence."
    )
    importer_task_rows = summary["level_c_diffusion"][
        "official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval_rows"
    ]
    importer_task_summary = {
        row["task"]: {
            "rollout_steps": row["rollout_steps"],
            "guided_reward_mean": row["guided_reward_mean"],
            "guided_target_body_error_mean": row["guided_target_body_error_mean"],
            "guided_teacher_action_mse_mean": row["guided_teacher_action_mse_mean"],
            "guidance_cost_delta_mean": row["guidance_cost_delta_mean"],
            "mp4": row["mp4"],
        }
        for row in importer_task_rows
    }
    lines.append(
        f"- Official-importer-export full-bundle task-conditioned latent-guidance closed-loop rollouts: "
        f"`{summary['level_c_diffusion']['official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval_status']}`; "
        f"bundle `{json.dumps(summary['level_c_diffusion']['official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval_bundle'], sort_keys=True)}`; "
        f"tasks `{json.dumps(summary['level_c_diffusion']['official_importer_export_full_bundle_task_conditioned_latent_guidance_rollout_eval_tasks'], sort_keys=True)}`; "
        f"summary `{json.dumps(importer_task_summary, sort_keys=True)}`. "
        "This executes joystick, waypoint, obstacle_avoidance, and composed local proxy guidance tasks on the "
        "official-importer-export G1 USDA path. Each task records teacher, VAE-base, denoised-latent, and "
        "receding-horizon guided-latent variants plus MP4/keyframes/metrics CSV/PNG evidence. It is now the "
        "strongest local closed-loop guided-control bridge on the recovered official-importer-export asset path, "
        "but it still uses local PPO/VAE/denoiser checkpoints and proxy costs rather than official BeyondMimic "
        "checkpoints, the paper Fig. 5/Fig. 6 task protocol, TensorRT/asynchronous deployment, or real robot."
    )
    scaled_importer_task_rows = summary["level_c_diffusion"][
        "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval_rows"
    ]
    scaled_importer_task_summary = {
        row["task"]: {
            "rollout_steps": row["rollout_steps"],
            "guided_reward_mean": row["guided_reward_mean"],
            "guided_target_body_error_mean": row["guided_target_body_error_mean"],
            "guided_teacher_action_mse_mean": row["guided_teacher_action_mse_mean"],
            "guidance_cost_delta_mean": row["guidance_cost_delta_mean"],
            "mp4": row["mp4"],
        }
        for row in scaled_importer_task_rows
    }
    scaled_importer_assets = summary["level_c_diffusion"][
        "official_importer_export_scaled_ppo_task_conditioned_guidance_summary_assets"
    ]
    lines.append(
        f"- Official-importer-export scaled-PPO task-conditioned latent-guidance closed-loop rollouts: "
        f"`{summary['level_c_diffusion']['official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval_status']}`; "
        f"bundle `{json.dumps(summary['level_c_diffusion']['official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval_bundle'], sort_keys=True)}`; "
        f"input statuses `{json.dumps(summary['level_c_diffusion']['official_importer_export_scaled_ppo_task_conditioned_latent_guidance_rollout_eval_inputs'], sort_keys=True)}`; "
        f"summary `{json.dumps(scaled_importer_task_summary, sort_keys=True)}`; report assets "
        f"`{json.dumps(scaled_importer_assets['assets'], sort_keys=True)}`. "
        "This repeats the importer-export closed-loop task-conditioned bridge with the iteration-999 scaled PPO "
        "teacher chain, the scaled VAE, scaled denoiser, and scaled offline-guidance summary. It is the current "
        "best local virtual closed-loop evidence that the stronger scaled-PPO downstream model can drive "
        "task-conditioned receding-latent guidance rollouts, but it still uses local proxy costs and local "
        "checkpoints. It is not official BeyondMimic Fig. 5/Fig. 6 success/failure evidence, not TensorRT/"
        "asynchronous deployment, and not real robot validation."
    )
    importer_task_multiseed_metrics = summary["level_c_diffusion"][
        "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval_metrics"
    ]
    importer_task_multiseed_checks = summary["level_c_diffusion"][
        "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval_checks"
    ]
    importer_task_multiseed_aggregate = {
        row["task"]: {
            "seed_count": row["seed_count"],
            "guided_reward_mean": row["guided_reward_mean_mean"],
            "guided_reward_std": row["guided_reward_mean_std"],
            "guided_target_body_error_mean": row["guided_target_body_error_mean_mean"],
            "guided_target_body_error_std": row["guided_target_body_error_mean_std"],
            "guided_done_count_total_mean": row["guided_done_count_total_mean"],
            "guidance_cost_delta_mean": row["guidance_cost_delta_mean_mean"],
        }
        for row in summary["level_c_diffusion"][
            "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval_aggregate"
        ]
    }
    lines.append(
        f"- Official-importer-export full-bundle task-conditioned latent-guidance multi-seed closed-loop rollouts: "
        f"`{summary['level_c_diffusion']['official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval_status']}`; "
        f"bundle `{json.dumps(summary['level_c_diffusion']['official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval_bundle'], sort_keys=True)}`; "
        f"metrics `{json.dumps(importer_task_multiseed_metrics, sort_keys=True)}`, "
        f"checks `{json.dumps(importer_task_multiseed_checks, sort_keys=True)}`, "
        f"aggregate `{json.dumps(importer_task_multiseed_aggregate, sort_keys=True)}`. "
        f"This aggregates {importer_task_multiseed_metrics['seed_group_count']} seed groups over joystick, waypoint, "
        f"obstacle_avoidance, and composed proxy tasks for {importer_task_multiseed_metrics['row_count']} local "
        "closed-loop IsaacLab rollouts on the official-importer-export G1 USDA path over the 40-motion public bundle, "
        f"covering {importer_task_multiseed_metrics['total_rollout_variant_steps']} recorded rollout-variant steps. "
        "It strengthens the single-seed importer-export guidance bridge with robustness and "
        "presentation-ready MP4/keyframe/plot evidence, but remains qualitative-only local virtual evidence using "
        "local checkpoints and proxy costs, not official BeyondMimic Fig. 5/Fig. 6 success metrics, TensorRT "
        "deployment, or real-robot validation."
    )
    importer_task_multiseed_assets = summary["level_c_diffusion"][
        "official_importer_export_full_bundle_task_conditioned_guidance_multiseed_assets"
    ]
    lines.append(
        f"- Official-importer-export task-conditioned guidance multi-seed report assets: "
        f"`{importer_task_multiseed_assets['status']}`; assets "
        f"`{json.dumps(importer_task_multiseed_assets['assets'], sort_keys=True)}`. "
        "These figures and CSVs summarize the importer-export multi-seed guided reward/error/done-count/"
        "guidance-cost statistics for the English report and PPT without upgrading the claim to paper-level "
        "reproduction."
    )
    scaled_importer_task_multiseed_metrics = summary["level_c_diffusion"][
        "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval_metrics"
    ]
    scaled_importer_task_multiseed_checks = summary["level_c_diffusion"][
        "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval_checks"
    ]
    scaled_importer_task_multiseed_aggregate = {
        row["task"]: {
            "seed_count": row["seed_count"],
            "guided_reward_mean": row["guided_reward_mean_mean"],
            "guided_reward_std": row["guided_reward_mean_std"],
            "guided_target_body_error_mean": row["guided_target_body_error_mean_mean"],
            "guided_target_body_error_std": row["guided_target_body_error_mean_std"],
            "guided_done_count_total_mean": row["guided_done_count_total_mean"],
            "guidance_cost_delta_mean": row["guidance_cost_delta_mean_mean"],
        }
        for row in summary["level_c_diffusion"][
            "official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval_aggregate"
        ]
    }
    lines.append(
        f"- Official-importer-export scaled-PPO task-conditioned latent-guidance multi-seed closed-loop rollouts: "
        f"`{summary['level_c_diffusion']['official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval_status']}`; "
        f"bundle `{json.dumps(summary['level_c_diffusion']['official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval_bundle'], sort_keys=True)}`; "
        f"input statuses `{json.dumps(summary['level_c_diffusion']['official_importer_export_scaled_ppo_task_conditioned_latent_guidance_multiseed_eval_inputs'], sort_keys=True)}`; "
        f"metrics `{json.dumps(scaled_importer_task_multiseed_metrics, sort_keys=True)}`, "
        f"checks `{json.dumps(scaled_importer_task_multiseed_checks, sort_keys=True)}`, "
        f"aggregate `{json.dumps(scaled_importer_task_multiseed_aggregate, sort_keys=True)}`. "
        f"This aggregates {scaled_importer_task_multiseed_metrics['seed_group_count']} seed groups over joystick, "
        f"waypoint, obstacle_avoidance, and composed proxy tasks for "
        f"{scaled_importer_task_multiseed_metrics['row_count']} local closed-loop IsaacLab rollouts from the "
        "iteration-999 scaled PPO teacher/VAE/denoiser chain on the official-importer-export G1 USDA path. It covers "
        f"{scaled_importer_task_multiseed_metrics['total_rollout_variant_steps']} recorded rollout-variant steps and "
        "20 local MP4 paths. This is stronger robustness evidence than the single-seed scaled bridge, but remains "
        "qualitative-only local virtual evidence: no official BeyondMimic VAE/diffusion checkpoint, no paper Fig. "
        "5/Fig. 6 success/failure protocol, no TensorRT deployment, and no real robot validation."
    )
    scaled_importer_task_multiseed_assets = summary["level_c_diffusion"][
        "official_importer_export_scaled_ppo_task_conditioned_guidance_multiseed_assets"
    ]
    lines.append(
        f"- Official-importer-export scaled-PPO task-conditioned guidance multi-seed report assets: "
        f"`{scaled_importer_task_multiseed_assets['status']}`; assets "
        f"`{json.dumps(scaled_importer_task_multiseed_assets['assets'], sort_keys=True)}`. "
        "These CSV/PNG assets summarize the scaled-PPO multi-seed closed-loop guidance rows for the English reading "
        "report while preserving the local-virtual, non-paper-level claim boundary."
    )
    importer_success_boundary = summary["level_c_diffusion"][
        "official_importer_export_full_bundle_task_conditioned_guidance_success_boundary"
    ]
    lines.append(
        f"- Official-importer-export task-conditioned guidance local proxy success boundary: "
        f"`{importer_success_boundary['status']}`; metrics "
        f"`{json.dumps(importer_success_boundary['metrics'], sort_keys=True)}`; aggregate "
        f"`{json.dumps(importer_success_boundary['aggregate'], sort_keys=True)}`; assets "
        f"`{json.dumps(importer_success_boundary['assets'], sort_keys=True)}`. "
        f"This turns the {importer_success_boundary['metrics']['row_count']} importer-export local closed-loop "
        f"guidance rollouts across {importer_success_boundary['metrics']['seed_group_count']} seed groups into explicit task-level proxy "
        "rates for 299-step completion, positive guidance signal, action change, reward improvement over the "
        "denoised baseline, tracking-error non-worsening, and a conservative local proxy pass flag. It is a "
        "report/PPT interpretation aid only, not an official BeyondMimic Fig. 5/Fig. 6 success protocol, not "
        "TensorRT deployment, and not real-robot validation."
    )
    importer_inpainting = summary["level_c_diffusion"][
        "official_importer_export_full_bundle_inpainting_guidance_rollout_eval"
    ]
    importer_inpainting_row = importer_inpainting["rows"][0]
    lines.append(
        f"- Official-importer-export Fig. 6A inpainting/keyframe diagnostic proxy: "
        f"`{importer_inpainting['status']}`; row "
        f"`{json.dumps(importer_inpainting_row, sort_keys=True)}`; checks "
        f"`{json.dumps(importer_inpainting['checks'], sort_keys=True)}`. "
        "This runs one 299-step local future-keyframe/root-path inpainting proxy on the recovered "
        "official-importer-export G1 USDA path and saves capture/video evidence. It is a useful virtual diagnostic "
        "for Fig. 6A, but it is not a success claim: the guided keyframe proxy error is larger than the denoised "
        "baseline for this seed. It is not the paper cartwheel keyframe protocol, not an official BeyondMimic "
        "checkpoint, not TensorRT deployment, and not real-robot validation."
    )
    fig56_proxy_matrix = summary["official_importer_export_fig5_fig6_proxy_protocol_matrix"]
    lines.append(
        f"- Official-importer-export Fig. 5/Fig. 6 proxy protocol matrix: "
        f"`{fig56_proxy_matrix['status']}`; metrics "
        f"`{json.dumps(fig56_proxy_matrix['metrics'], sort_keys=True)}`; assets "
        f"`{json.dumps({k: fig56_proxy_matrix[k] for k in ['json', 'csv', 'markdown', 'plot_png']}, sort_keys=True)}`. "
        "This maps the current importer-export local virtual evidence onto the six paper panels: joystick panels "
        "have supporting local closed-loop proxy evidence, waypoint/obstacle/composed evidence supports the Fig. 6 "
        "obstacle-navigation family only as a simulated proxy, and Fig. 6A now has one importer-export inpainting "
        "diagnostic proxy while still lacking the paper cartwheel/keyframe protocol. It is a planning and report "
        "aid, not a paper-level Fig. 5/Fig. 6 success, fall, collision, TensorRT, mocap, or real-robot protocol."
    )
    task_protocol_proxy = summary["official_importer_export_fig5_fig6_task_protocol_proxy"]
    lines.append(
        f"- Official-importer-export Fig. 5/Fig. 6 local task-protocol proxy metrics: "
        f"`{task_protocol_proxy['status']}`; metrics "
        f"`{json.dumps(task_protocol_proxy['metrics'], sort_keys=True)}`; thresholds "
        f"`{json.dumps(task_protocol_proxy['thresholds'], sort_keys=True)}`; aggregate "
        f"`{json.dumps(task_protocol_proxy['aggregate'], sort_keys=True)}`; assets "
        f"`{json.dumps({k: task_protocol_proxy[k] for k in ['json', 'rows_csv', 'aggregate_csv', 'markdown', 'rates_png', 'deltas_png']}, sort_keys=True)}`. "
        "This converts the 20 local closed-loop importer-export guidance traces across 5 seed groups into "
        "explicit report-facing proxy metrics: 299-step trace completion, endpoint/root-reference error, "
        "target-body tracking error, guidance-cost decrease, reward delta vs the denoised baseline, and "
        "tracking-error delta vs the denoised baseline. It strengthens the English reading report by replacing "
        "a vague statement of simulated guidance evidence with a thresholded local protocol table. The thresholds "
        "are local analysis thresholds, not BeyondMimic paper thresholds, so this remains qualitative-only local "
        "virtual evidence rather than official Fig. 5/Fig. 6 success, fall, collision, TensorRT, mocap, or "
        "real-robot validation."
    )
    scaled_task_protocol_proxy = summary["official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy"]
    lines.append(
        f"- Official-importer-export scaled-PPO Fig. 5/Fig. 6 local task-protocol proxy metrics: "
        f"`{scaled_task_protocol_proxy['status']}`; metrics "
        f"`{json.dumps(scaled_task_protocol_proxy['metrics'], sort_keys=True)}`; thresholds "
        f"`{json.dumps(scaled_task_protocol_proxy['thresholds'], sort_keys=True)}`; aggregate "
        f"`{json.dumps(scaled_task_protocol_proxy['aggregate'], sort_keys=True)}`; assets "
        f"`{json.dumps({k: scaled_task_protocol_proxy[k] for k in ['json', 'rows_csv', 'aggregate_csv', 'markdown', 'rates_png', 'deltas_png']}, sort_keys=True)}`. "
        f"This converts the {scaled_task_protocol_proxy['metrics']['row_count']} scaled-PPO importer-export local "
        f"closed-loop guidance traces across {scaled_task_protocol_proxy['metrics']['seed_group_count']} seed groups "
        "into the same stricter local protocol table. It records a higher local proxy pass rate than the earlier "
        "full-bundle chain, but the thresholds remain local analysis thresholds, not BeyondMimic Fig. 5/Fig. 6 "
        "paper success/fall/collision criteria; it is not TensorRT, mocap, or real robot evidence."
    )
    success_fall_collision_proxy = summary[
        "official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy"
    ]
    lines.append(
        f"- Official-importer-export scaled-PPO Fig. 5/Fig. 6 local success/fall/collision proxy metrics: "
        f"`{success_fall_collision_proxy['status']}`; metrics "
        f"`{json.dumps(success_fall_collision_proxy['metrics'], sort_keys=True)}`; thresholds "
        f"`{json.dumps(success_fall_collision_proxy['thresholds'], sort_keys=True)}`; aggregate "
        f"`{json.dumps(success_fall_collision_proxy['aggregate'], sort_keys=True)}`; assets "
        f"`{json.dumps({k: success_fall_collision_proxy[k] for k in ['json', 'rows_csv', 'aggregate_csv', 'markdown', 'rates_png']}, sort_keys=True)}`. "
        "This turns the same 20 scaled-PPO closed-loop traces into a stricter report-facing proxy for success, "
        "fall, and collision discussion: local success proxy rate, relative-root-height fall proxy rate, body-error "
        "spike anomaly rate, and explicit contact/collision-signal absence. It is useful for the English report "
        "because it states what the current virtual evidence can and cannot say. It is not an official paper "
        "success/fall/collision evaluation because the traces lack contact labels, official thresholds, the exact "
        "Fig. 5/Fig. 6 task protocol, TensorRT deployment evidence, mocap context, or real-robot validation."
    )
    transition = summary["level_c_diffusion"]["official_importer_export_full_bundle_transition_guidance_rollout_eval"]
    lines.append(
        f"- Official-importer-export Fig. 5B walk-to-run transition guidance proxy: "
        f"`{transition['status']}`; row `{json.dumps(transition['rows'][0], sort_keys=True)}`; checks "
        f"`{json.dumps(transition['checks'], sort_keys=True)}`. "
        "This runs one 299-step local closed-loop velocity-ramp transition diagnostic on GPU4 with the recovered "
        "official-importer-export G1 USDA path and local PPO/VAE/denoiser checkpoints, then records a local MP4 path "
        "and speed/path plots. It is useful report evidence because it moves the Fig. 5B/Fig. 5D discussion from "
        "latent visualization toward a real IsaacLab rollout. It is also a diagnostic rather than a success claim: "
        "the guided variant increases late-vs-early speed under the local proxy, but its speed-target correlation is "
        "weak and target-speed RMSE is high. It is not the paper walking-to-running transition protocol, not the "
        "paper Fig. 5D t-SNE panel, not an official BeyondMimic checkpoint result, not TensorRT deployment, and not "
        "real-robot validation."
    )
    latent_projection = summary["official_importer_export_full_bundle_latent_projection_report_assets"]
    lines.append(
        f"- Official-importer-export Fig. 5D latent PCA projection proxy: "
        f"`{latent_projection['status']}`; metrics "
        f"`{json.dumps(latent_projection['metrics'], sort_keys=True)}`; assets "
        f"`{json.dumps(latent_projection['assets'], sort_keys=True)}`. "
        "This projects local full-bundle VAE posterior means from 306176 official-importer-export teacher-rollout "
        "samples into a two-dimensional PCA space and records walk/run traces for report visualization. It is useful "
        "Fig. 5D-adjacent evidence for discussing latent organization, but it is not the paper t-SNE panel, not an "
        "official BeyondMimic checkpoint result, not a closed-loop walk-to-run transition protocol, not TensorRT "
        "deployment, and not real-robot validation."
    )
    importer_contact_sheet = summary["level_c_diffusion"][
        "official_importer_export_full_bundle_guidance_video_contact_sheet"
    ]
    lines.append(
        f"- Official-importer-export guidance video contact sheet: `{importer_contact_sheet['status']}`; "
        f"metrics `{json.dumps(importer_contact_sheet['metrics'], sort_keys=True)}`; assets "
        f"`{json.dumps(importer_contact_sheet['assets'], sort_keys=True)}`. "
        f"This indexes {importer_contact_sheet['metrics']['video_count']} local MP4 rollouts with SHA256 hashes and "
        "builds a compact keyframe contact sheet for "
        "the English reading report/PPT. The MP4 files remain local and are not committed to GitHub; the claim "
        "level remains local virtual official-importer-export evidence, not official Fig. 5/Fig. 6, TensorRT, "
        "or real-robot evidence."
    )
    onnx_async_summary = {
        "status": summary["level_c_diffusion"]["official_csv_loop_vae_denoiser_onnx_async_status"],
        "providers": summary["level_c_diffusion"]["official_csv_loop_vae_denoiser_onnx_async_settings"][
            "onnxruntime_available_providers"
        ],
        "providers_used": summary["level_c_diffusion"]["official_csv_loop_vae_denoiser_onnx_async_settings"][
            "onnxruntime_execution_providers_used"
        ],
        "consistency": summary["level_c_diffusion"]["official_csv_loop_vae_denoiser_onnx_async_consistency"],
        "async_summary": summary["level_c_diffusion"]["official_csv_loop_vae_denoiser_onnx_async_summary"],
        "outputs": summary["level_c_diffusion"]["official_csv_loop_vae_denoiser_onnx_async_outputs"],
    }
    lines.append(
        f"- Official csv-loop local VAE/denoiser ONNXRuntime async deployment-path audit: "
        f"`{summary['level_c_diffusion']['official_csv_loop_vae_denoiser_onnx_async_status']}`; "
        f"summary `{json.dumps(onnx_async_summary, sort_keys=True)}`. "
        "This exports the locally trained official-csv-loop VAE encoder/decoder and state-latent denoiser to ONNX, "
        "checks ONNXRuntime CPU outputs against PyTorch, and measures a sequential plus thread-pool async proxy. "
        "The local ORT build exposes CPU/Azure providers only, so CUDAExecutionProvider and TensorRT are explicitly "
        "recorded as unavailable. This is useful deployment-path evidence, not paper Mini-PC latency, not TensorRT, "
        "not CppAD guidance, not a live IsaacLab deployed controller, and not real-robot evidence."
    )
    full_bundle_onnx_async_summary = {
        "status": summary["level_c_diffusion"]["official_csv_loop_full_bundle_vae_denoiser_onnx_async_status"],
        "providers": summary["level_c_diffusion"]["official_csv_loop_full_bundle_vae_denoiser_onnx_async_settings"][
            "onnxruntime_available_providers"
        ],
        "providers_used": summary["level_c_diffusion"]["official_csv_loop_full_bundle_vae_denoiser_onnx_async_settings"][
            "onnxruntime_execution_providers_used"
        ],
        "consistency": summary["level_c_diffusion"]["official_csv_loop_full_bundle_vae_denoiser_onnx_async_consistency"],
        "async_summary": summary["level_c_diffusion"]["official_csv_loop_full_bundle_vae_denoiser_onnx_async_summary"],
        "outputs": summary["level_c_diffusion"]["official_csv_loop_full_bundle_vae_denoiser_onnx_async_outputs"],
    }
    lines.append(
        f"- Official csv-loop full-bundle VAE/denoiser ONNXRuntime async deployment-path audit: "
        f"`{summary['level_c_diffusion']['official_csv_loop_full_bundle_vae_denoiser_onnx_async_status']}`; "
        f"summary `{json.dumps(full_bundle_onnx_async_summary, sort_keys=True)}`. "
        "This repeats the deployment-path audit on the broader 40-motion full-bundle local VAE and denoiser. "
        "ONNXRuntime CPU outputs match PyTorch within sub-micro absolute error, and the thread-pool async proxy "
        "records local throughput improvement. CUDAExecutionProvider and TensorRT remain unavailable in the local "
        "ORT build, so this is still not the paper's TensorRT/Mini-PC deployment, not CppAD guidance, not a live "
        "controller integration, and not real-robot evidence."
    )
    importer_export_onnx_async_summary = {
        "status": summary["level_c_diffusion"][
            "official_importer_export_full_bundle_vae_denoiser_onnx_async_status"
        ],
        "providers": summary["level_c_diffusion"][
            "official_importer_export_full_bundle_vae_denoiser_onnx_async_settings"
        ]["onnxruntime_available_providers"],
        "providers_used": summary["level_c_diffusion"][
            "official_importer_export_full_bundle_vae_denoiser_onnx_async_settings"
        ]["onnxruntime_execution_providers_used"],
        "consistency": summary["level_c_diffusion"][
            "official_importer_export_full_bundle_vae_denoiser_onnx_async_consistency"
        ],
        "async_summary": summary["level_c_diffusion"][
            "official_importer_export_full_bundle_vae_denoiser_onnx_async_summary"
        ],
        "outputs": summary["level_c_diffusion"][
            "official_importer_export_full_bundle_vae_denoiser_onnx_async_outputs"
        ],
    }
    lines.append(
        f"- Official-importer-export full-bundle VAE/denoiser ONNXRuntime async deployment-path audit: "
        f"`{summary['level_c_diffusion']['official_importer_export_full_bundle_vae_denoiser_onnx_async_status']}`; "
        f"summary `{json.dumps(importer_export_onnx_async_summary, sort_keys=True)}`. "
        "This repeats the local export/runtime audit on the currently strongest official-importer-export VAE and "
        "state-latent denoiser chain. ONNXRuntime CPU outputs match PyTorch within sub-micro absolute error, and "
        "the thread-pool async proxy records local throughput improvement. The audit explicitly records that "
        "CUDAExecutionProvider and TensorRT are unavailable in the local ORT build, so this is not paper Mini-PC "
        "TensorRT latency, not an official BeyondMimic checkpoint, not a live IsaacLab deployment, and not "
        "real-robot evidence."
    )
    resource_adjusted_state_latent_worker = summary["level_c_diffusion"][
        "resource_adjusted_teacher_rollout_state_latent_dataset_worker"
    ]
    resource_adjusted_state_latent_summary = {
        "sample_count": resource_adjusted_state_latent_worker["dataset"]["sample_count"],
        "window_count": resource_adjusted_state_latent_worker["dataset"]["window_count"],
        "split_counts": resource_adjusted_state_latent_worker["dataset"]["split_counts"],
        "sequence_length": resource_adjusted_state_latent_worker["dataset"]["sequence_length"],
        "obs_dim": resource_adjusted_state_latent_worker["dataset"]["obs_dim"],
        "latent_dim": resource_adjusted_state_latent_worker["dataset"]["latent_dim"],
        "token_dim": resource_adjusted_state_latent_worker["dataset"]["token_dim"],
        "weighted_posterior_reconstruction_mse": resource_adjusted_state_latent_worker["dataset"][
            "weighted_posterior_reconstruction_mse"
        ],
    }
    lines.append(
        f"- Resource-adjusted full teacher-rollout state-latent dataset: "
        f"`{summary['level_c_diffusion']['resource_adjusted_teacher_rollout_state_latent_dataset_status']}`; "
        f"summary `{json.dumps(resource_adjusted_state_latent_summary, sort_keys=True)}`. "
        "This converts all current resource-adjusted teacher rollout shards into 21-step policy-observation plus VAE "
        "posterior-latent windows. It is a useful downstream diffusion dataset, but remains generated-resource local "
        "evidence rather than the official DAgger/state-latent dataset."
    )
    resource_adjusted_diffusion_worker = summary["level_c_diffusion"][
        "resource_adjusted_state_latent_diffusion_training_worker"
    ]
    resource_adjusted_diffusion_summary = {
        "window_count": resource_adjusted_diffusion_worker["dataset"]["window_count"],
        "split_counts": resource_adjusted_diffusion_worker["dataset"]["split_counts"],
        "epochs": resource_adjusted_diffusion_worker["training"]["epochs"],
        "batch_windows": resource_adjusted_diffusion_worker["training"]["batch_windows"],
        "data_parallel_used": resource_adjusted_diffusion_worker["data_parallel_used"],
        "validation_pred_token_mse": resource_adjusted_diffusion_worker["evaluation"]["validation"][
            "pred_token_mse"
        ],
        "test_pred_token_mse": resource_adjusted_diffusion_worker["evaluation"]["test"]["pred_token_mse"],
        "test_noisy_token_mse": resource_adjusted_diffusion_worker["evaluation"]["test"]["noisy_token_mse"],
        "test_denoising_improvement_ratio": resource_adjusted_diffusion_worker["evaluation"]["test"][
            "denoising_improvement_ratio"
        ],
        "gpu_metrics_summary": summary["level_c_diffusion"][
            "resource_adjusted_state_latent_diffusion_training_gpu_metrics"
        ],
    }
    lines.append(
        f"- Resource-adjusted full state-latent denoiser training: "
        f"`{summary['level_c_diffusion']['resource_adjusted_state_latent_diffusion_training_status']}`; "
        f"summary `{json.dumps(resource_adjusted_diffusion_summary, sort_keys=True)}`. "
        "This trains on all generated windows and shows held-out denoising improvement, but it is not the official "
        "BeyondMimic diffusion checkpoint, TensorRT engine, or Fig. 5/Fig. 6 closed-loop guidance result."
    )
    resource_adjusted_guidance_worker = summary["level_c_diffusion"][
        "resource_adjusted_state_latent_guidance_eval_worker"
    ]
    resource_adjusted_guidance_summary = {
        "total_selected_windows": resource_adjusted_guidance_worker["metrics"]["total_selected_windows"],
        "row_count": resource_adjusted_guidance_worker["metrics"]["row_count"],
        "tasks": resource_adjusted_guidance_worker["settings"]["tasks"],
        "scales": resource_adjusted_guidance_worker["settings"]["scales"],
        "tasks_with_all_best_costs_improve": resource_adjusted_guidance_worker["metrics"][
            "tasks_with_all_best_costs_improve"
        ],
        "tasks_with_nonzero_best_gradients": resource_adjusted_guidance_worker["metrics"][
            "tasks_with_nonzero_best_gradients"
        ],
        "task_summaries": {
            task: {
                "mean_best_cost_delta": value["mean_best_cost_delta"],
                "mean_positive_delta_fraction": value["mean_positive_delta_fraction"],
            }
            for task, value in resource_adjusted_guidance_worker["task_summaries"].items()
        },
        "gpu_metrics_summary": summary["level_c_diffusion"][
            "resource_adjusted_state_latent_guidance_eval_gpu_metrics"
        ],
    }
    lines.append(
        f"- Resource-adjusted offline state-latent guidance evaluation: "
        f"`{summary['level_c_diffusion']['resource_adjusted_state_latent_guidance_eval_status']}`; "
        f"summary `{json.dumps(resource_adjusted_guidance_summary, sort_keys=True)}`. "
        "This connects the local denoiser to task-cost guidance on validation/test windows. It is useful evidence for "
        "the reading-report reproduction section, but it is not a closed-loop IsaacLab rollout, not official "
        "Fig. 5/Fig. 6 evidence, and not a paper-level guidance reproduction."
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
        "res/tracking/official_replay_npz_entry_diagnostic/"
        "tracking_official_replay_npz_entry_diagnostic_audit.json",
        "res/tracking/official_replay_npz_entry_diagnostic/"
        "tracking_official_replay_npz_entry_diagnostic_probe.py",
        "res/failed_runs/tracking_official_replay_npz_entry_diagnostic/"
        "tracking_official_replay_npz_entry_diagnostic.log",
        "res/tracking/official_replay_npz_loop_with_enriched_usd/"
        "tracking_official_replay_npz_loop_with_enriched_usd_audit.json",
        "res/tracking/official_replay_npz_loop_with_enriched_usd/"
        "tracking_official_replay_npz_loop_with_enriched_usd_probe.py",
        "res/tracking/official_replay_npz_loop_full_dataset_with_enriched_usd/"
        "tracking_official_replay_npz_loop_full_dataset_with_enriched_usd_audit.json",
        "res/tracking/official_replay_npz_loop_full_dataset_with_enriched_usd/"
        "tracking_official_replay_npz_loop_full_dataset_with_enriched_usd_rows.csv",
        "res/tracking/official_replay_npz_loop_full_dataset_with_enriched_usd/"
        "tracking_official_replay_npz_loop_full_dataset_with_enriched_usd_rows.tsv",
        "res/tracking/official_replay_npz_loop_full_dataset_with_official_importer_export/"
        "tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_audit.json",
        "res/tracking/official_replay_npz_loop_full_dataset_with_official_importer_export/"
        "tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_rows.csv",
        "res/tracking/official_replay_npz_loop_full_dataset_with_official_importer_export/"
        "tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_rows.tsv",
        "res/tracking/official_csv_to_npz_loop_with_enriched_usd/"
        "tracking_official_csv_to_npz_loop_with_enriched_usd_audit.json",
        "res/tracking/official_csv_to_npz_loop_with_enriched_usd/"
        "tracking_official_csv_to_npz_loop_with_enriched_usd_metrics.json",
        "res/tracking/official_csv_to_npz_loop_with_enriched_usd/"
        "tracking_official_csv_to_npz_loop_with_enriched_usd_probe.py",
        "res/tracking/official_csv_to_npz_loop_full_dataset_with_enriched_usd/"
        "tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_audit.json",
        "res/tracking/official_csv_to_npz_loop_full_dataset_with_enriched_usd/"
        "tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_rows.csv",
        "res/tracking/official_csv_to_npz_loop_full_dataset_with_enriched_usd/"
        "tracking_official_csv_to_npz_loop_full_dataset_with_enriched_usd_rows.tsv",
        "res/tracking/official_csv_to_npz_loop_full_dataset_with_official_importer_export/"
        "tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_audit.json",
        "res/tracking/official_csv_to_npz_loop_full_dataset_with_official_importer_export/"
        "tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_rows.csv",
        "res/tracking/official_csv_to_npz_loop_full_dataset_with_official_importer_export/"
        "tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_rows.tsv",
        "reproduction/scripts/official_importer_export_full_dataset_reference_replay_video_asset.py",
        "res/visualization/official_importer_export_full_dataset_reference_replay/"
        "official_importer_export_full_dataset_reference_replay_video_asset.json",
        "res/visualization/official_importer_export_full_dataset_reference_replay/"
        "official_importer_export_full_dataset_reference_replay_keyframes.png",
        "res/visualization/official_importer_export_full_dataset_reference_replay/"
        "official_importer_export_full_dataset_reference_replay_summary.csv",
        "res/visualization/official_importer_export_full_dataset_reference_replay/README.md",
        "res/tracking/g1_official_csv_loop_full_dataset_task_eval/"
        "tracking_g1_official_csv_loop_full_dataset_task_eval.json",
        "res/tracking/g1_official_csv_loop_full_dataset_task_eval/"
        "tracking_g1_official_csv_loop_full_dataset_task_eval_rows.csv",
        "res/tracking/g1_official_csv_loop_full_dataset_task_eval/"
        "tracking_g1_official_csv_loop_full_dataset_task_eval_rows.tsv",
        "res/report_assets/official_csv_loop_full_dataset_task_eval/"
        "official_csv_loop_full_dataset_task_eval_assets.json",
        "res/report_assets/official_csv_loop_full_dataset_task_eval/"
        "official_csv_loop_full_dataset_task_eval_metrics.csv",
        "res/report_assets/official_csv_loop_full_dataset_task_eval/"
        "official_csv_loop_full_dataset_task_eval_completion_table.csv",
        "res/report_assets/official_csv_loop_full_dataset_task_eval/"
        "official_csv_loop_full_dataset_task_eval_reward_done.png",
        "res/report_assets/official_csv_loop_full_dataset_task_eval/"
        "official_csv_loop_full_dataset_task_eval_tracking_errors.png",
        "res/tracking/g1_official_importer_export_task_smoke/"
        "tracking_g1_official_importer_export_task_smoke.json",
        "res/tracking/g1_official_importer_export_task_smoke/"
        "tracking_g1_official_importer_export_task_smoke_metrics.json",
        "res/tracking/g1_official_importer_export_full_dataset_task_eval/"
        "tracking_g1_official_importer_export_full_dataset_task_eval.json",
        "res/tracking/g1_official_importer_export_full_dataset_task_eval/"
        "tracking_g1_official_importer_export_full_dataset_task_eval_rows.csv",
        "res/tracking/g1_official_importer_export_full_dataset_task_eval/"
        "tracking_g1_official_importer_export_full_dataset_task_eval_rows.tsv",
        "res/report_assets/official_importer_export_full_dataset_task_eval/"
        "official_importer_export_full_dataset_task_eval_assets.json",
        "res/report_assets/official_importer_export_full_dataset_task_eval/"
        "official_importer_export_full_dataset_task_eval_metrics.csv",
        "res/report_assets/official_importer_export_full_dataset_task_eval/"
        "official_importer_export_full_dataset_task_eval_completion_table.csv",
        "res/report_assets/official_importer_export_full_dataset_task_eval/"
        "official_importer_export_full_dataset_task_eval_reward_done.png",
        "res/report_assets/official_importer_export_full_dataset_task_eval/"
        "official_importer_export_full_dataset_task_eval_tracking_errors.png",
        "res/tracking/g1_urdf_import_config_variant_probe/"
        "tracking_g1_urdf_import_config_variant_probe.json",
        "res/tracking/g1_enriched_usd_replay_preflight/tracking_g1_enriched_usd_replay_preflight_audit.json",
        "res/tracking/g1_enriched_usd_bounded_replay_metrics/"
        "tracking_g1_enriched_usd_bounded_replay_metrics_audit.json",
        "res/tracking/g1_enriched_usd_bounded_replay_metrics/"
        "walk1_subject1_64step_resource_adjusted_replay_metrics.json",
        "res/tracking/g1_resource_adjusted_task_smoke/tracking_g1_resource_adjusted_task_smoke_audit.json",
        "res/tracking/g1_resource_adjusted_task_smoke/tracking_g1_resource_adjusted_task_smoke_metrics.json",
        "res/tracking/g1_resource_adjusted_multi_fixture_eval/"
        "tracking_g1_resource_adjusted_multi_fixture_eval_audit.json",
        "res/tracking/g1_resource_adjusted_multi_fixture_eval/"
        "tracking_g1_resource_adjusted_multi_fixture_eval_metrics.json",
        "res/tracking/g1_resource_adjusted_multi_fixture_eval/"
        "walk1_subject1_frames_1_180_debug_motion_task_eval_metrics.json",
        "res/tracking/g1_resource_adjusted_multi_fixture_eval/"
        "run2_subject1_frames_1_180_debug_motion_task_eval_metrics.json",
        "res/tracking/g1_resource_adjusted_multi_fixture_eval/"
        "jumps1_subject1_frames_1_180_debug_motion_task_eval_metrics.json",
        "res/tracking/g1_resource_adjusted_csv_conversion/"
        "tracking_g1_resource_adjusted_csv_conversion_audit.json",
        "res/tracking/g1_resource_adjusted_csv_conversion/"
        "tracking_g1_resource_adjusted_csv_conversion_metrics.json",
        "res/tracking/g1_resource_adjusted_csv_conversion/"
        "walk1_subject1_frames_1_180_resource_adjusted_motion_contract.json",
        "res/tracking/g1_resource_adjusted_csv_full_replay/"
        "tracking_g1_resource_adjusted_csv_full_replay_audit.json",
        "res/tracking/g1_resource_adjusted_csv_full_replay/"
        "walk1_subject1_frames_1_180_resource_adjusted_full_replay_metrics.json",
        "res/tracking/g1_resource_adjusted_csv_task_eval/"
        "tracking_g1_resource_adjusted_csv_task_eval_audit.json",
        "res/tracking/g1_resource_adjusted_csv_task_eval/"
        "tracking_g1_resource_adjusted_csv_task_eval_metrics.json",
        "res/tracking/g1_resource_adjusted_train_entry_diagnostic/"
        "tracking_g1_resource_adjusted_train_entry_diagnostic_audit.json",
        "res/tracking/g1_resource_adjusted_train_entry_diagnostic/"
        "tracking_g1_resource_adjusted_train_entry_diagnostic_metrics.json",
        "res/tracking/urdf_conversion_probe/tracking_urdf_conversion_probe.json",
        "res/tracking/urdf_path_tiny_probe/tracking_urdf_path_tiny_probe.json",
        "res/tracking/mjcf_stage_probe/tracking_mjcf_stage_probe.json",
        "res/tracking/usd_save_policy_probe/tracking_usd_save_policy_probe.json",
        "res/tracking/simulationapp_save_policy_probe/tracking_simulationapp_save_policy_probe.json",
        "res/tracking/usd_api_variant_probe/tracking_usd_api_variant_probe.json",
        "res/tracking/g1_urdf_stage_export_workaround/tracking_g1_urdf_stage_export_workaround_probe.json",
        "res/tracking/g1_urdf_layer_save_workaround/tracking_g1_urdf_layer_save_workaround_probe.json",
        "res/tracking/g1_urdf_in_memory_import/tracking_g1_urdf_in_memory_import_probe.json",
        "res/tracking/g1_urdf_simulationapp_in_memory_import/tracking_g1_urdf_simulationapp_in_memory_import_probe.json",
        "res/tracking/g1_urdf_in_memory_variant_matrix/tracking_g1_urdf_in_memory_variant_matrix_probe.json",
        "res/tracking/g1_preconverted_asset_audit/tracking_g1_preconverted_asset_audit.json",
        "res/tracking/g1_reference_usd_compatibility_audit/tracking_g1_reference_usd_compatibility_audit.json",
        "res/tracking/g1_official_urdf_skeleton_usd/tracking_g1_official_urdf_skeleton_usd_audit.json",
        "res/tracking/g1_official_urdf_skeleton_usd/g1_official_urdf_29dof_skeleton.usda",
        "res/tracking/g1_urdf_physical_asset_contract_audit/tracking_g1_urdf_physical_asset_contract_audit.json",
        "res/tracking/g1_urdf_physical_asset_contract_audit/tracking_g1_urdf_physical_asset_contract_audit.tsv",
        "res/tracking/g1_urdf_source_equivalence_audit/tracking_g1_urdf_source_equivalence_audit.json",
        "res/tracking/g1_urdf_source_equivalence_audit/tracking_g1_urdf_source_equivalence_audit.tsv",
        "res/tracking/g1_resource_adjusted_enriched_usd/tracking_g1_resource_adjusted_enriched_usd_probe.json",
        "res/tracking/g1_resource_adjusted_enriched_usd/g1_resource_adjusted_29dof_enriched_scaffold.usda",
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
        "res/level_c/resource_adjusted_teacher_rollout_vae_training/level_c_resource_adjusted_teacher_rollout_vae_training.json",
        "res/level_c/resource_adjusted_teacher_rollout_vae_training/level_c_resource_adjusted_teacher_rollout_vae_training.tsv",
        "res/level_c/resource_adjusted_teacher_rollout_state_latent_dataset/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.json",
        "res/level_c/resource_adjusted_teacher_rollout_state_latent_dataset/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.tsv",
        "res/level_c/resource_adjusted_state_latent_diffusion_training/level_c_resource_adjusted_state_latent_diffusion_training.json",
        "res/level_c/resource_adjusted_state_latent_diffusion_training/level_c_resource_adjusted_state_latent_diffusion_training.tsv",
        "res/level_c/resource_adjusted_state_latent_guidance_eval/level_c_resource_adjusted_state_latent_guidance_eval.json",
        "res/level_c/resource_adjusted_state_latent_guidance_eval/level_c_resource_adjusted_state_latent_guidance_eval.tsv",
        "res/level_c/official_csv_loop_guided_action_rollout_probe/"
        "tracking_g1_official_csv_loop_guided_action_rollout_probe.json",
        "res/level_c/official_csv_loop_guided_action_rollout_probe/"
        "official_csv_loop_guided_action_rollout_probe_assets.json",
        "res/level_c/official_csv_loop_guided_action_rollout_probe/"
        "official_csv_loop_guided_action_rollout_probe_timeseries.csv",
        "res/level_c/official_csv_loop_guided_action_rollout_probe/"
        "official_csv_loop_guided_action_rollout_probe_metrics.png",
        "res/level_c/official_csv_loop_action_guidance_rollout_eval/"
        "level_c_official_csv_loop_action_guidance_rollout_eval.json",
        "res/level_c/official_csv_loop_action_guidance_rollout_eval/"
        "level_c_official_csv_loop_action_guidance_rollout_eval.tsv",
        "res/visualization/official_csv_loop_action_guidance_rollout/"
        "official_csv_loop_action_guidance_rollout_asset.json",
        "res/visualization/official_csv_loop_action_guidance_rollout/"
        "official_csv_loop_action_guidance_rollout_metrics.csv",
        "res/visualization/official_csv_loop_action_guidance_rollout/"
        "official_csv_loop_action_guidance_rollout_metrics.png",
        "res/visualization/official_csv_loop_action_guidance_rollout/"
        "official_csv_loop_action_guidance_rollout_keyframes.png",
        "reproduction/scripts/tracking_g1_official_csv_loop_full_bundle_policy_rollout_video_capture.py",
        "res/visualization/official_csv_loop_full_bundle_policy_rollout/"
        "tracking_g1_official_csv_loop_policy_rollout_capture.json",
        "res/visualization/official_csv_loop_full_bundle_policy_rollout/"
        "official_csv_loop_policy_rollout_video_asset.json",
        "res/visualization/official_csv_loop_full_bundle_policy_rollout/"
        "official_csv_loop_policy_rollout_metrics.csv",
        "res/visualization/official_csv_loop_full_bundle_policy_rollout/"
        "official_csv_loop_policy_rollout_keyframes.png",
        "res/visualization/official_csv_loop_full_bundle_policy_rollout/README.md",
        "reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_policy_rollout_video_capture.py",
        "res/visualization/official_importer_export_full_bundle_scaled_ppo_policy_rollout/"
        "tracking_g1_official_importer_export_full_bundle_scaled_ppo_policy_rollout_capture.json",
        "res/visualization/official_importer_export_full_bundle_scaled_ppo_policy_rollout/"
        "official_importer_export_full_bundle_scaled_ppo_policy_rollout_video_asset.json",
        "res/visualization/official_importer_export_full_bundle_scaled_ppo_policy_rollout/"
        "tracking_g1_official_importer_export_full_bundle_scaled_ppo_policy_rollout_worker.py",
        "res/visualization/official_importer_export_full_bundle_scaled_ppo_policy_rollout/"
        "tracking_g1_official_importer_export_full_bundle_scaled_ppo_policy_rollout_render.py",
        "res/visualization/official_importer_export_full_bundle_scaled_ppo_policy_rollout/"
        "official_importer_export_full_bundle_scaled_ppo_policy_rollout_metrics.csv",
        "res/visualization/official_importer_export_full_bundle_scaled_ppo_policy_rollout/"
        "official_importer_export_full_bundle_scaled_ppo_policy_rollout_keyframes.png",
        "res/visualization/official_importer_export_full_bundle_scaled_ppo_policy_rollout/README.md",
        "res/level_c/official_csv_loop_receding_latent_guidance_rollout_eval/"
        "level_c_official_csv_loop_receding_latent_guidance_rollout_eval.json",
        "res/level_c/official_csv_loop_receding_latent_guidance_rollout_eval/"
        "level_c_official_csv_loop_receding_latent_guidance_rollout_eval.tsv",
        "res/visualization/official_csv_loop_receding_latent_guidance_rollout/"
        "official_csv_loop_receding_latent_guidance_rollout_asset.json",
        "res/visualization/official_csv_loop_receding_latent_guidance_rollout/"
        "official_csv_loop_receding_latent_guidance_rollout_metrics.csv",
        "res/visualization/official_csv_loop_receding_latent_guidance_rollout/"
        "official_csv_loop_receding_latent_guidance_rollout_metrics.png",
        "res/visualization/official_csv_loop_receding_latent_guidance_rollout/"
        "official_csv_loop_receding_latent_guidance_rollout_keyframes.png",
        "res/level_c/official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval/"
        "level_c_official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval.json",
        "res/level_c/official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval/"
        "level_c_official_csv_loop_full_bundle_receding_latent_guidance_rollout_eval.tsv",
        "res/visualization/official_csv_loop_full_bundle_receding_latent_guidance_rollout/"
        "official_csv_loop_receding_latent_guidance_rollout_asset.json",
        "res/visualization/official_csv_loop_full_bundle_receding_latent_guidance_rollout/"
        "official_csv_loop_receding_latent_guidance_rollout_metrics.csv",
        "res/visualization/official_csv_loop_full_bundle_receding_latent_guidance_rollout/"
        "official_csv_loop_receding_latent_guidance_rollout_metrics.png",
        "res/visualization/official_csv_loop_full_bundle_receding_latent_guidance_rollout/"
        "official_csv_loop_receding_latent_guidance_rollout_keyframes.png",
        "res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval/"
        "level_c_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval.json",
        "res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval/"
        "level_c_official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval.tsv",
        *[
            f"res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval/{task}/"
            f"{task}_task_conditioned_latent_guidance_rollout_eval.json"
            for task in ["joystick", "waypoint", "obstacle_avoidance", "composed"]
        ],
        *[
            f"res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout_eval/{task}/"
            f"{task}_task_conditioned_latent_guidance_rollout_eval.tsv"
            for task in ["joystick", "waypoint", "obstacle_avoidance", "composed"]
        ],
        *[
            "res/visualization/official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout/"
            f"{task}/official_csv_loop_receding_latent_guidance_rollout_asset.json"
            for task in ["joystick", "waypoint", "obstacle_avoidance", "composed"]
        ],
        *[
            "res/visualization/official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout/"
            f"{task}/official_csv_loop_task_conditioned_latent_guidance_rollout_metrics.csv"
            for task in ["joystick", "waypoint", "obstacle_avoidance", "composed"]
        ],
        *[
            "res/visualization/official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout/"
            f"{task}/official_csv_loop_task_conditioned_latent_guidance_rollout_metrics.png"
            for task in ["joystick", "waypoint", "obstacle_avoidance", "composed"]
        ],
        *[
            "res/visualization/official_csv_loop_full_bundle_task_conditioned_latent_guidance_rollout/"
            f"{task}/official_csv_loop_task_conditioned_latent_guidance_rollout_keyframes.png"
            for task in ["joystick", "waypoint", "obstacle_avoidance", "composed"]
        ],
        "res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_summary/"
        "official_csv_loop_task_conditioned_guidance_summary_assets.json",
        "res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_summary/"
        "task_conditioned_guidance_metrics.csv",
        "res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_summary/"
        "task_conditioned_guided_summary.csv",
        "res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_summary/"
        "task_conditioned_guidance_overview.png",
        "res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_summary/"
        "task_conditioned_guidance_tradeoff.png",
        "res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval/"
        "official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json",
        "res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval/"
        "official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_rows.csv",
        "res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval/"
        "official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_rows.tsv",
        "res/level_c/official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_eval/"
        "official_csv_loop_full_bundle_task_conditioned_latent_guidance_multiseed_aggregate.csv",
        "res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_multiseed/"
        "official_csv_loop_full_bundle_task_conditioned_guidance_multiseed_assets.json",
        "res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_multiseed/"
        "full_bundle_task_conditioned_guidance_multiseed_aggregate.csv",
        "res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_multiseed/"
        "full_bundle_task_conditioned_guidance_multiseed_bars.png",
        "res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_multiseed/"
        "full_bundle_task_conditioned_guidance_multiseed_seed_scatter.png",
        "res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_success_boundary/"
        "local_proxy_success_boundary.json",
        "res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_success_boundary/"
        "local_proxy_success_boundary_rows.csv",
        "res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_success_boundary/"
        "local_proxy_success_boundary_aggregate.csv",
        "res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_success_boundary/"
        "local_proxy_success_boundary_rates.png",
        "res/report_assets/official_csv_loop_full_bundle_task_conditioned_guidance_success_boundary/README.md",
        "res/report_assets/official_csv_loop_full_bundle_guidance_video_contact_sheet/"
        "full_bundle_guidance_video_index.json",
        "res/report_assets/official_csv_loop_full_bundle_guidance_video_contact_sheet/"
        "full_bundle_guidance_video_index.csv",
        "res/report_assets/official_csv_loop_full_bundle_guidance_video_contact_sheet/"
        "full_bundle_guidance_video_contact_sheet.png",
        "res/report_assets/official_csv_loop_full_bundle_guidance_video_contact_sheet/README.md",
        "res/level_c/official_csv_loop_task_conditioned_latent_guidance_rollout_eval/"
        "level_c_official_csv_loop_task_conditioned_latent_guidance_rollout_eval.json",
        "res/level_c/official_csv_loop_task_conditioned_latent_guidance_rollout_eval/"
        "level_c_official_csv_loop_task_conditioned_latent_guidance_rollout_eval.tsv",
        *[
            f"res/level_c/official_csv_loop_task_conditioned_latent_guidance_rollout_eval/{task}/"
            f"{task}_task_conditioned_latent_guidance_rollout_eval.json"
            for task in ["joystick", "waypoint", "obstacle_avoidance", "composed"]
        ],
        *[
            f"res/level_c/official_csv_loop_task_conditioned_latent_guidance_rollout_eval/{task}/"
            f"{task}_task_conditioned_latent_guidance_rollout_eval.tsv"
            for task in ["joystick", "waypoint", "obstacle_avoidance", "composed"]
        ],
        *[
            "res/visualization/official_csv_loop_task_conditioned_latent_guidance_rollout/"
            f"{task}/official_csv_loop_receding_latent_guidance_rollout_asset.json"
            for task in ["joystick", "waypoint", "obstacle_avoidance", "composed"]
        ],
        *[
            "res/visualization/official_csv_loop_task_conditioned_latent_guidance_rollout/"
            f"{task}/official_csv_loop_task_conditioned_latent_guidance_rollout_metrics.csv"
            for task in ["joystick", "waypoint", "obstacle_avoidance", "composed"]
        ],
        *[
            "res/visualization/official_csv_loop_task_conditioned_latent_guidance_rollout/"
            f"{task}/official_csv_loop_task_conditioned_latent_guidance_rollout_metrics.png"
            for task in ["joystick", "waypoint", "obstacle_avoidance", "composed"]
        ],
        *[
            "res/visualization/official_csv_loop_task_conditioned_latent_guidance_rollout/"
            f"{task}/official_csv_loop_task_conditioned_latent_guidance_rollout_keyframes.png"
            for task in ["joystick", "waypoint", "obstacle_avoidance", "composed"]
        ],
        "res/report_assets/official_csv_loop_task_conditioned_guidance_summary/"
        "official_csv_loop_task_conditioned_guidance_summary_assets.json",
        "res/report_assets/official_csv_loop_task_conditioned_guidance_summary/"
        "task_conditioned_guidance_metrics.csv",
        "res/report_assets/official_csv_loop_task_conditioned_guidance_summary/"
        "task_conditioned_guided_summary.csv",
        "res/report_assets/official_csv_loop_task_conditioned_guidance_summary/"
        "task_conditioned_guidance_overview.png",
        "res/report_assets/official_csv_loop_task_conditioned_guidance_summary/"
        "task_conditioned_guidance_tradeoff.png",
        "res/level_c/official_csv_loop_task_conditioned_latent_guidance_multiseed_eval/"
        "official_csv_loop_task_conditioned_latent_guidance_multiseed_eval.json",
        "res/level_c/official_csv_loop_task_conditioned_latent_guidance_multiseed_eval/"
        "official_csv_loop_task_conditioned_latent_guidance_multiseed_rows.csv",
        "res/level_c/official_csv_loop_task_conditioned_latent_guidance_multiseed_eval/"
        "official_csv_loop_task_conditioned_latent_guidance_multiseed_rows.tsv",
        "res/level_c/official_csv_loop_task_conditioned_latent_guidance_multiseed_eval/"
        "official_csv_loop_task_conditioned_latent_guidance_multiseed_aggregate.csv",
        "res/report_assets/official_csv_loop_task_conditioned_guidance_multiseed/"
        "official_csv_loop_task_conditioned_guidance_multiseed_assets.json",
        "res/report_assets/official_csv_loop_task_conditioned_guidance_multiseed/"
        "task_conditioned_guidance_multiseed_aggregate.csv",
        "res/report_assets/official_csv_loop_task_conditioned_guidance_multiseed/"
        "task_conditioned_guidance_multiseed_bars.png",
        "res/report_assets/official_csv_loop_task_conditioned_guidance_multiseed/"
        "task_conditioned_guidance_multiseed_seed_scatter.png",
        "res/level_c/official_csv_loop_vae_closed_loop_rollout_eval/"
        "tracking_g1_official_csv_loop_vae_closed_loop_rollout_eval.json",
        "res/level_c/official_csv_loop_vae_closed_loop_rollout_eval/"
        "tracking_g1_official_csv_loop_vae_closed_loop_rollout_worker.py",
        "res/report_assets/official_csv_loop_vae_closed_loop_rollout_eval/"
        "official_csv_loop_vae_closed_loop_rollout_assets.json",
        "res/report_assets/official_csv_loop_vae_closed_loop_rollout_eval/"
        "vae_closed_loop_reward_done_timeseries.png",
        "res/report_assets/official_csv_loop_vae_closed_loop_rollout_eval/"
        "vae_closed_loop_action_reconstruction_error.png",
        "res/report_assets/official_csv_loop_vae_closed_loop_rollout_eval/"
        "vae_closed_loop_action_magnitude.png",
        "res/report_assets/official_csv_loop_vae_closed_loop_rollout_eval/"
        "vae_closed_loop_tracking_errors.png",
        "res/report_assets/official_csv_loop_vae_closed_loop_rollout_eval/"
        "vae_closed_loop_gpu_memory.png",
        "res/report_assets/official_csv_loop_vae_closed_loop_rollout_eval/"
        "vae_closed_loop_shard_summary.csv",
        "res/report_assets/official_csv_loop_vae_closed_loop_rollout_eval/"
        "vae_closed_loop_gpu_summary.csv",
        "res/level_c/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/"
        "tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_eval.json",
        "res/level_c/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/"
        "tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_worker.py",
        "res/report_assets/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/"
        "official_importer_export_full_bundle_vae_closed_loop_rollout_assets.json",
        "res/report_assets/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/"
        "official_importer_vae_closed_loop_reward_done_timeseries.png",
        "res/report_assets/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/"
        "official_importer_vae_closed_loop_action_reconstruction_error.png",
        "res/report_assets/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/"
        "official_importer_vae_closed_loop_action_magnitude.png",
        "res/report_assets/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/"
        "official_importer_vae_closed_loop_tracking_errors.png",
        "res/report_assets/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/"
        "official_importer_vae_closed_loop_gpu_memory.png",
        "res/report_assets/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/"
        "official_importer_vae_closed_loop_shard_summary.csv",
        "res/report_assets/official_importer_export_full_bundle_vae_closed_loop_rollout_eval/"
        "official_importer_vae_closed_loop_gpu_summary.csv",
        "reproduction/scripts/tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_video_capture.py",
        "res/visualization/official_importer_export_full_bundle_vae_closed_loop_rollout/"
        "tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_capture.json",
        "res/visualization/official_importer_export_full_bundle_vae_closed_loop_rollout/"
        "official_importer_export_full_bundle_vae_closed_loop_rollout_video_asset.json",
        "res/visualization/official_importer_export_full_bundle_vae_closed_loop_rollout/"
        "tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_worker.py",
        "res/visualization/official_importer_export_full_bundle_vae_closed_loop_rollout/"
        "tracking_g1_official_importer_export_full_bundle_vae_closed_loop_rollout_render.py",
        "res/visualization/official_importer_export_full_bundle_vae_closed_loop_rollout/"
        "official_importer_export_full_bundle_vae_closed_loop_rollout_keyframes.png",
        "res/visualization/official_importer_export_full_bundle_vae_closed_loop_rollout/"
        "official_importer_export_full_bundle_vae_closed_loop_rollout_metrics.csv",
        "res/visualization/official_importer_export_full_bundle_vae_closed_loop_rollout/README.md",
        "reproduction/scripts/level_c_official_importer_export_full_bundle_teacher_rollout_state_latent_dataset.py",
        "res/level_c/official_importer_export_full_bundle_teacher_rollout_state_latent_dataset/"
        "level_c_official_importer_export_full_bundle_teacher_rollout_state_latent_dataset.json",
        "res/level_c/official_importer_export_full_bundle_teacher_rollout_state_latent_dataset/"
        "level_c_official_importer_export_full_bundle_teacher_rollout_state_latent_dataset.tsv",
        "reproduction/scripts/level_c_official_importer_export_full_bundle_state_latent_diffusion_training.py",
        "res/level_c/official_importer_export_full_bundle_state_latent_diffusion_training/"
        "level_c_official_importer_export_full_bundle_state_latent_diffusion_training.json",
        "res/level_c/official_importer_export_full_bundle_state_latent_diffusion_training/"
        "level_c_official_importer_export_full_bundle_state_latent_diffusion_training.tsv",
        "reproduction/scripts/official_importer_export_full_bundle_downstream_report_assets.py",
        "res/report_assets/official_importer_export_full_bundle_downstream/"
        "official_importer_export_full_bundle_downstream_report_assets.json",
        "res/report_assets/official_importer_export_full_bundle_downstream/"
        "official_importer_downstream_vae_training_curve.png",
        "res/report_assets/official_importer_export_full_bundle_downstream/"
        "official_importer_downstream_diffusion_training_curve.png",
        "res/report_assets/official_importer_export_full_bundle_downstream/"
        "official_importer_downstream_split_metrics.csv",
        "res/report_assets/official_importer_export_full_bundle_downstream/"
        "official_importer_downstream_stage_summary.csv",
        "res/report_assets/official_importer_export_full_bundle_downstream/README.md",
        "reproduction/scripts/level_c_official_importer_export_scaled_ppo_teacher_rollout_vae_training.py",
        "res/level_c/official_importer_export_scaled_ppo_teacher_rollout_vae_training/"
        "level_c_official_importer_export_scaled_ppo_teacher_rollout_vae_training.json",
        "res/level_c/official_importer_export_scaled_ppo_teacher_rollout_vae_training/"
        "level_c_official_importer_export_scaled_ppo_teacher_rollout_vae_training.tsv",
        "reproduction/scripts/level_c_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset.py",
        "res/level_c/official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset/"
        "level_c_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset.json",
        "res/level_c/official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset/"
        "level_c_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset.tsv",
        "reproduction/scripts/level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training.py",
        "res/level_c/official_importer_export_scaled_ppo_state_latent_diffusion_training/"
        "level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training.json",
        "res/level_c/official_importer_export_scaled_ppo_state_latent_diffusion_training/"
        "level_c_official_importer_export_scaled_ppo_state_latent_diffusion_training.tsv",
        "reproduction/scripts/official_importer_export_scaled_ppo_downstream_report_assets.py",
        "res/report_assets/official_importer_export_scaled_ppo_downstream/"
        "official_importer_export_full_bundle_downstream_report_assets.json",
        "res/report_assets/official_importer_export_scaled_ppo_downstream/"
        "official_importer_downstream_vae_training_curve.png",
        "res/report_assets/official_importer_export_scaled_ppo_downstream/"
        "official_importer_downstream_diffusion_training_curve.png",
        "res/report_assets/official_importer_export_scaled_ppo_downstream/"
        "official_importer_downstream_split_metrics.csv",
        "res/report_assets/official_importer_export_scaled_ppo_downstream/"
        "official_importer_downstream_stage_summary.csv",
        "res/report_assets/official_importer_export_scaled_ppo_downstream/README.md",
        "reproduction/scripts/tracking_g1_official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval.py",
        "res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval/"
        "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval.json",
        "res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval/"
        "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_rows.csv",
        "res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval/"
        "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_rows.tsv",
        "res/level_c/official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_eval/"
        "official_importer_export_full_bundle_task_conditioned_latent_guidance_multiseed_aggregate.csv",
        "reproduction/scripts/official_importer_export_full_bundle_task_conditioned_guidance_multiseed_report_assets.py",
        "res/report_assets/official_importer_export_full_bundle_task_conditioned_guidance_multiseed/"
        "official_importer_export_full_bundle_task_conditioned_guidance_multiseed_assets.json",
        "res/report_assets/official_importer_export_full_bundle_task_conditioned_guidance_multiseed/"
        "importer_export_task_conditioned_guidance_multiseed_aggregate.csv",
        "res/report_assets/official_importer_export_full_bundle_task_conditioned_guidance_multiseed/"
        "importer_export_task_conditioned_guidance_multiseed_bars.png",
        "res/report_assets/official_importer_export_full_bundle_task_conditioned_guidance_multiseed/"
        "importer_export_task_conditioned_guidance_multiseed_seed_scatter.png",
        "res/report_assets/official_importer_export_full_bundle_task_conditioned_guidance_multiseed/README.md",
        "reproduction/scripts/official_importer_export_full_bundle_task_conditioned_guidance_success_boundary.py",
        "res/report_assets/official_importer_export_full_bundle_task_conditioned_guidance_success_boundary/"
        "local_proxy_success_boundary.json",
        "res/report_assets/official_importer_export_full_bundle_task_conditioned_guidance_success_boundary/"
        "local_proxy_success_boundary_rows.csv",
        "res/report_assets/official_importer_export_full_bundle_task_conditioned_guidance_success_boundary/"
        "local_proxy_success_boundary_aggregate.csv",
        "res/report_assets/official_importer_export_full_bundle_task_conditioned_guidance_success_boundary/"
        "local_proxy_success_boundary_rates.png",
        "res/report_assets/official_importer_export_full_bundle_task_conditioned_guidance_success_boundary/README.md",
        "reproduction/scripts/tracking_g1_official_importer_export_full_bundle_inpainting_guidance_rollout_eval.py",
        "res/level_c/official_importer_export_full_bundle_inpainting_guidance_rollout_eval/"
        "level_c_official_importer_export_full_bundle_inpainting_guidance_rollout_eval.json",
        "res/level_c/official_importer_export_full_bundle_inpainting_guidance_rollout_eval/"
        "level_c_official_importer_export_full_bundle_inpainting_guidance_rollout_eval.tsv",
        "res/level_c/official_importer_export_full_bundle_inpainting_guidance_rollout_eval/"
        "underlying_task_conditioned_inpainting.json",
        "res/level_c/official_importer_export_full_bundle_inpainting_guidance_rollout_eval/"
        "underlying_task_conditioned_inpainting.tsv",
        "reproduction/scripts/tracking_g1_official_importer_export_full_bundle_transition_guidance_rollout_eval.py",
        "res/level_c/official_importer_export_full_bundle_transition_guidance_rollout_eval/"
        "level_c_official_importer_export_full_bundle_transition_guidance_rollout_eval.json",
        "res/level_c/official_importer_export_full_bundle_transition_guidance_rollout_eval/"
        "level_c_official_importer_export_full_bundle_transition_guidance_rollout_eval.tsv",
        "res/level_c/official_importer_export_full_bundle_transition_guidance_rollout_eval/"
        "underlying_transition_task.json",
        "res/report_assets/official_importer_export_full_bundle_transition_guidance/"
        "transition_guidance_report_assets.json",
        "res/report_assets/official_importer_export_full_bundle_transition_guidance/"
        "transition_speed_profile.png",
        "res/report_assets/official_importer_export_full_bundle_transition_guidance/"
        "transition_root_path.png",
        "res/report_assets/official_importer_export_full_bundle_transition_guidance/"
        "transition_metric_bars.png",
        "res/report_assets/official_importer_export_full_bundle_transition_guidance/README.md",
        "reproduction/scripts/official_importer_export_full_bundle_guidance_video_contact_sheet.py",
        "res/report_assets/official_importer_export_full_bundle_guidance_video_contact_sheet/"
        "importer_export_guidance_video_index.json",
        "res/report_assets/official_importer_export_full_bundle_guidance_video_contact_sheet/"
        "importer_export_guidance_video_index.csv",
        "res/report_assets/official_importer_export_full_bundle_guidance_video_contact_sheet/"
        "importer_export_guidance_video_contact_sheet.png",
        "res/report_assets/official_importer_export_full_bundle_guidance_video_contact_sheet/README.md",
        "reproduction/scripts/official_importer_export_fig5_fig6_proxy_protocol_matrix.py",
        "res/report_assets/official_importer_export_fig5_fig6_proxy_protocol_matrix/"
        "fig5_fig6_proxy_protocol_matrix.json",
        "res/report_assets/official_importer_export_fig5_fig6_proxy_protocol_matrix/"
        "fig5_fig6_proxy_protocol_matrix.csv",
        "res/report_assets/official_importer_export_fig5_fig6_proxy_protocol_matrix/"
        "fig5_fig6_proxy_protocol_matrix.md",
        "res/report_assets/official_importer_export_fig5_fig6_proxy_protocol_matrix/"
        "fig5_fig6_proxy_protocol_rates.png",
        "res/report_assets/official_importer_export_fig5_fig6_proxy_protocol_matrix/README.md",
        "reproduction/scripts/official_importer_export_fig5_fig6_task_protocol_proxy.py",
        "res/report_assets/official_importer_export_fig5_fig6_task_protocol_proxy/"
        "fig5_fig6_task_protocol_proxy.json",
        "res/report_assets/official_importer_export_fig5_fig6_task_protocol_proxy/"
        "fig5_fig6_task_protocol_proxy_rows.csv",
        "res/report_assets/official_importer_export_fig5_fig6_task_protocol_proxy/"
        "fig5_fig6_task_protocol_proxy_aggregate.csv",
        "res/report_assets/official_importer_export_fig5_fig6_task_protocol_proxy/"
        "fig5_fig6_task_protocol_proxy.md",
        "res/report_assets/official_importer_export_fig5_fig6_task_protocol_proxy/"
        "fig5_fig6_task_protocol_proxy_rates.png",
        "res/report_assets/official_importer_export_fig5_fig6_task_protocol_proxy/"
        "fig5_fig6_task_protocol_proxy_deltas.png",
        "res/report_assets/official_importer_export_fig5_fig6_task_protocol_proxy/README.md",
        "reproduction/scripts/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy.py",
        "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy/"
        "fig5_fig6_task_protocol_proxy.json",
        "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy/"
        "fig5_fig6_task_protocol_proxy_rows.csv",
        "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy/"
        "fig5_fig6_task_protocol_proxy_aggregate.csv",
        "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy/"
        "fig5_fig6_task_protocol_proxy.md",
        "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy/"
        "fig5_fig6_task_protocol_proxy_rates.png",
        "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy/"
        "fig5_fig6_task_protocol_proxy_deltas.png",
        "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_task_protocol_proxy/README.md",
        "reproduction/scripts/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy.py",
        "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy/"
        "success_fall_collision_proxy.json",
        "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy/"
        "success_fall_collision_proxy_rows.csv",
        "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy/"
        "success_fall_collision_proxy_aggregate.csv",
        "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy/"
        "success_fall_collision_proxy.md",
        "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy/"
        "success_fall_collision_proxy_rates.png",
        "res/report_assets/official_importer_export_scaled_ppo_fig5_fig6_success_fall_collision_proxy/README.md",
        "reproduction/scripts/official_importer_export_full_bundle_latent_projection_report_assets.py",
        "res/report_assets/official_importer_export_full_bundle_latent_projection/"
        "official_importer_export_full_bundle_latent_projection_assets.json",
        "res/report_assets/official_importer_export_full_bundle_latent_projection/"
        "latent_pca_projection_samples.csv",
        "res/report_assets/official_importer_export_full_bundle_latent_projection/"
        "latent_pca_family_summary.csv",
        "res/report_assets/official_importer_export_full_bundle_latent_projection/"
        "latent_pca_walk_run_trace.csv",
        "res/report_assets/official_importer_export_full_bundle_latent_projection/"
        "latent_pca_by_motion_family.png",
        "res/report_assets/official_importer_export_full_bundle_latent_projection/"
        "latent_pca_by_root_speed.png",
        "res/report_assets/official_importer_export_full_bundle_latent_projection/"
        "latent_pca_walk_run_trace.png",
        "res/report_assets/official_importer_export_full_bundle_latent_projection/README.md",
        "res/report_assets/guided_vs_unguided_closed_loop_matrix/"
        "guided_vs_unguided_closed_loop_matrix.json",
        "res/report_assets/guided_vs_unguided_closed_loop_matrix/"
        "guided_vs_unguided_closed_loop_matrix.csv",
        "res/report_assets/guided_vs_unguided_closed_loop_matrix/"
        "guided_vs_unguided_closed_loop_aggregate.csv",
        "res/report_assets/guided_vs_unguided_closed_loop_matrix/"
        "guided_vs_unguided_closed_loop_matrix.md",
        "res/report_assets/guided_vs_unguided_closed_loop_matrix/"
        "task_conditioned_multiseed_guided_deltas.png",
        "res/report_assets/guided_vs_unguided_closed_loop_matrix/"
        "task_conditioned_guidance_signal_strength.png",
        "res/level_c/official_csv_loop_vae_denoiser_onnx_async/"
        "level_c_official_csv_loop_vae_denoiser_onnx_async_audit.json",
        "res/level_c/official_csv_loop_vae_denoiser_onnx_async/"
        "level_c_official_csv_loop_vae_denoiser_onnx_async_audit.tsv",
        "res/level_c/official_csv_loop_vae_denoiser_onnx_async/"
        "level_c_official_csv_loop_vae_denoiser_onnx_async_latency.csv",
        "res/level_c/official_csv_loop_full_bundle_vae_denoiser_onnx_async/"
        "level_c_official_csv_loop_full_bundle_vae_denoiser_onnx_async_audit.json",
        "res/level_c/official_csv_loop_full_bundle_vae_denoiser_onnx_async/"
        "level_c_official_csv_loop_full_bundle_vae_denoiser_onnx_async_audit.tsv",
        "res/level_c/official_csv_loop_full_bundle_vae_denoiser_onnx_async/"
        "level_c_official_csv_loop_full_bundle_vae_denoiser_onnx_async_latency.csv",
        "reproduction/scripts/level_c_official_importer_export_full_bundle_vae_denoiser_onnx_async_audit.py",
        "res/level_c/official_importer_export_full_bundle_vae_denoiser_onnx_async/"
        "level_c_official_importer_export_full_bundle_vae_denoiser_onnx_async_audit.json",
        "res/level_c/official_importer_export_full_bundle_vae_denoiser_onnx_async/"
        "level_c_official_importer_export_full_bundle_vae_denoiser_onnx_async_audit.tsv",
        "res/level_c/official_importer_export_full_bundle_vae_denoiser_onnx_async/"
        "level_c_official_importer_export_full_bundle_vae_denoiser_onnx_async_latency.csv",
        "reproduction/scripts/level_c_official_importer_export_scaled_ppo_vae_denoiser_onnx_async_audit.py",
        "res/level_c/official_importer_export_scaled_ppo_vae_denoiser_onnx_async/"
        "level_c_official_importer_export_scaled_ppo_vae_denoiser_onnx_async_audit.json",
        "res/level_c/official_importer_export_scaled_ppo_vae_denoiser_onnx_async/"
        "level_c_official_importer_export_scaled_ppo_vae_denoiser_onnx_async_audit.tsv",
        "res/level_c/official_importer_export_scaled_ppo_vae_denoiser_onnx_async/"
        "level_c_official_importer_export_scaled_ppo_vae_denoiser_onnx_async_latency.csv",
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
