# Final Reproduction Evidence Report

This report consolidates the current BeyondMimic reproduction evidence. It is generated from machine-readable audits and does not mark the full goal complete.

## Current Status
- Master audit: `232/232` artifacts pass, failures `0`.
- Completion matrix counts: `{"blocked": 3, "complete": 73, "out_of_scope": 1, "partial": 85}`.
- Goal complete: `False`.
- Why not complete: The evidence set is internally consistent, but completion matrix still contains partial/blocked/out_of_scope items for live Kit tracking, teacher rollouts, true DAgger, trained Level C checkpoints, Fig. 5/6 paper reproduction, and real robot deployment.

## Level Summary
- Takeover audit: `ok_with_runtime_warnings`; checks `{"download_present": true, "download_treated_read_only": true, "json_artifacts_readable": true, "key_files_present": true, "old_root_text_paths_absent": true, "other_backup_present": true, "smoke_scripts_compile": true, "training_started": false, "workspace_promoted": true}`; runtime warnings `[{"name": "nvcc_version", "returncode": 1}]`.
- Environments: bm_analysis `ok`, bm_tracking `partial_blocked_for_kit`, bm_diffusion `ok` with bm_diffusion checks `{"base_numpy_scipy_yaml_tqdm_smoke_passes": true, "does_not_claim_training_or_paper_results": true, "lock_files_exist": true, "prefix_exists": true, "torch_cuda_smoke_passes": true, "torch_install_log_incomplete_recorded": true, "training_environment_smoke_ready": true}`.
- Environment import probe: `ok_with_runtime_warning`; checks `{"analysis_imports_ok": true, "diffusion_torch_cuda_visible_devices_5_6_ok": true, "isaaclab_import_ok": true, "isaaclab_live_headless_gate_ok": true, "isaacsim_import_ok": true, "tracking_basic_imports_ok": true, "tracking_pip_check_ok": true, "training_started": false}`.
- IsaacLab live gate probe: `ok_with_runtime_warning`; current blocker `none`; checks `{"app_launcher_reached_success_sentinel": true, "cuda_p2p_iommu_runtime_warning_retained": true, "cuda_visible_devices_single_gpu_not_viable": true, "current_inotify_limits_meet_targets": true, "does_not_claim_tracking_reproduction_complete": true, "fast_shutdown_false_candidate_recorded": true, "fast_shutdown_semantics_recorded": true, "no_training_started": true, "package_import_probe_ok": true, "project_egl_icd_exists": true, "project_egl_icd_removes_vulkan_error": true, "single_gpu_renderer_limits_active_gpu": true, "tracking_python_exists": true}`.
- Current IsaacLab headless AppLauncher gate: `ok`; config `{"candidate_physical_gpus": [4, 7], "cuda_visible_devices": "", "device": "cuda:4", "max_busy_util_percent": 50, "min_free_mb_required": 20000, "selected_physical_gpu": 4, "timeout_seconds": 240}`; checks `{"app_launcher_headless_success_sentinel": true, "does_not_claim_tracking_reproduction_complete": true, "no_fatal_runtime_error": true, "no_training_started": true, "payload_is_running": true, "sentinel_after_app": true, "sentinel_payload": true}`. This confirms the current headless startup sentinel on physical GPU 4 without claiming official replay, PPO, DAgger, Fig. 5/Fig. 6, or real-robot completion.
- Vulkan runtime probe: `ok`; checks `{"does_not_claim_isaaclab_gate_passed": true, "does_not_launch_kit_or_training": true, "isaac_bundled_loader_create_instance_ok": true, "libglx_nvidia_resolves": true, "nvidia_icd_json_exists": true, "nvidia_icd_mentions_libglx_nvidia": true, "project_egl_icd_written": true, "system_loader_create_instance_ok": false}`.
- CUDA P2P runtime probe: `ok`; checks `{"does_not_claim_isaaclab_gate_passed": true, "does_not_launch_kit_or_training": true, "has_peer_access_already_enabled_signature": false, "nvidia_smi_ok": true, "records_peer_access_results": true}`.
- IsaacLab GPU foundation settings audit: `ok`; checks `{"app_launcher_gate_clear_or_warning": true, "cpu_device_attempt_recorded": true, "cuda_p2p_iommu_runtime_warning_retained": true, "cuda_visible_devices_single_gpu_not_viable": true, "does_not_claim_tracking_reproduction_complete": true, "does_not_launch_kit_or_training": true, "project_egl_icd_removes_vulkan_error": true, "settings_surface_search_ran": true, "simapp_fast_shutdown_false_attempt_recorded": true, "simapp_multi_gpu_false_attempt_recorded": true, "single_gpu_renderer_limits_active_gpu": true}`.
- GPU resource monitoring: `ok`; `24` snapshot rows over `8` GPUs, nontrivial existing memory `{"6": 3988}`.
- Run management schema: `ok`; diagnostic run `setup_run_management_diagnostic_static_000_20260617_050000` has `48` GPU metric rows and required run files/directories.
- Checkpoint/resume smoke: `ok`; run `setup_checkpoint_resume_smoke_static_000_20260617_061500` resumes with max abs error `0.0` and writes `/mnt/infini-data/test/BeyondMimic/res/runs/setup_checkpoint_resume_smoke_static_000_20260617_061500/checkpoint/step_0005.npz`.
- Full run deliverable gap audit: `ok`; metrics `{"diagnostic_or_debug_run_count": 4, "diagnostic_or_debug_run_with_nonempty_videos_count": 1, "failed_check_count": 0, "run_directory_count": 4, "run_with_nonempty_checkpoint_dir_count": 3, "run_with_nonempty_figures_dir_count": 2, "run_with_nonempty_videos_dir_count": 1, "run_with_training_endpoint_metrics_count": 2, "schema_complete_run_count": 4, "valid_training_run_count": 0, "valid_training_run_with_nonempty_videos_count": 0}`.
- Failed-run retention: `ok`; failed run `phase1_isaaclab_headless_smoke_g1_inotify_0_20260616_200654` preserved with `48` GPU status rows.
- Official train-entry failed-run retention: `ok`; failed run `phase1_official_train_entry_retry_inotify_0_20260617_174742` preserved with `48` GPU status rows.
- Patch inventory audit: `ok`; metrics `{"official_repo_count": 3, "patch_file_count": 2, "row_count": 5, "status_timeout_repo_count": 1, "tracked_change_repo_count": 2, "tracked_modified_file_count": 20}`, status counts `{"present": 2, "status_timeout": 1, "tracked_changes": 2}`.
- Patch snapshot audit: `ok`; metrics `{"patch_file_count": 2, "semantic_empty_patch_count": 2, "snapshot_row_count": 2, "total_patch_size_bytes": 78164, "tracked_modified_file_count": 20}`, patch directory `/mnt/infini-data/test/BeyondMimic/reproduction/patches/official_worktree_snapshots`.
- Reimplementation package: `ok`; `18` Python files and `29` checked formula API symbols under `/mnt/infini-data/test/BeyondMimic/reproduction/src/beyondmimic_reimpl`.
- Reimplementation runtime integration audit: `ok`; metrics `{"current_downstream_action_mse": 0.007938830058739797, "dagger_teacher_query_count": 30.0, "decoded_teacher_action_mse": 0.00019921838212813084, "diffusion_mse_after": 0.2814410265203397, "diffusion_mse_before": 0.5864538340928773, "predicted_action_smoothness_penalty": 0.0320482910674662, "split_counts": {"test": 28, "train": 28, "validation": 28}, "survival_rate": 0.75, "token_shape": [84, 21, 131], "tracking_mean_error": 0.017320508075688783, "window_count": 84}`.
- Coding requirements audit: `ok`; `13` goal.md coding rows, failed `0`, public function rows `28`.
- Reimplementation package API tests: `ok`; `8` rows, failed `0`, covered items `["api_surface", "dagger", "diffusion", "evaluation", "finite_guards", "fixed_seed", "geometry", "goal_metrics", "guidance", "mask_shape", "package_exports", "sampling", "shape_errors", "state", "trajectory", "vae"]`.
- Reimplementation test suite: `ok`; `5/5` pure-Python code/test/audit steps passed, metrics `{"api_row_count": 8, "core_math_row_count": 23, "coverage_required_count": 20, "package_symbol_count": 29, "runtime_token_shape": [84, 21, 131], "runtime_window_count": 84}`.
- Resolved config manifest: `ok`; tracking `50.0` Hz, PPO max iterations `30000`, VAE latent `32`, diffusion batch `512`, denoising steps `20`.
- Artifact manifest: `ok`; `324` hashed key artifacts, missing `0`.
- Completion matrix status audit: `ok`; `162` rows, invalid statuses `0`, status counts `{"blocked": 3, "complete": 73, "out_of_scope": 1, "partial": 85}`.
- Download source integrity audit: `ok`; `6391` manifest rows, total bytes `6577530557`, required hashes `17`, reference hashes `8`.
- Run/log/config catalog: `ok`; metrics `{"config_file_count": 7, "file_count": 97, "invalid_or_debug_run_count": 5, "log_file_count": 62, "run_directory_count": 6, "valid_training_run_count": 0}`.
- Experiment protocol: `ok`; `19` required protocol patterns, missing `0`.
- Top-level README: `ok`; `20` required entry-point patterns, missing `0`.
- Final deliverables audit: `ok`; `38` deliverable rows, status counts `{"blocked_or_missing": 2, "complete": 18, "complete_for_core_math": 1, "complete_for_current_failures": 1, "complete_for_local_copies": 1, "complete_for_released_and_debug": 3, "complete_for_released_data": 1, "partial": 11}`, missing evidence rows `0`.
- Visual media inventory: `ok`; `113` media files, kind counts `{"gif": 4, "pdf": 30, "png": 49, "svg": 30}`, category counts `{"debug_augmentation_visual": 6, "debug_checkpoint_guidance_visual": 16, "debug_guidance_visual": 4, "debug_run_figure": 17, "debug_tiny_diffusion_preview": 4, "debug_tracking_visual": 3, "other_visual_media": 3, "released_data_figure": 60}`; paper-required rollout/robot videos remain absent.
- Verification command coverage audit: `ok`; `193` final-report commands categorized, lightweight smoke pass `10/10`.
- Verification command syntax audit: `ok`; `185` unique Python command scripts compiled, failed `0`.
- Verification command script manifest: `ok`; `185` unique Python command scripts hashed with SHA256.
- Required artifact absence audit: `ok`; `20` trained/deployment artifact rows, status counts `{"debug_only_not_required_artifact": 2, "missing_required_artifact": 12, "present_but_not_required_artifact": 6}`, local model files `7`, local videos `0`.
- Evaluation metrics coverage audit: `ok`; `44` `goal.md` Section 12 metrics, status counts `{"blocked_or_missing": 7, "debug_only": 1, "debug_or_released": 3, "formula_api_only": 5, "partial": 2, "public_data_checkpoint": 18, "released_data": 8}`, missing evidence rows `0`.
- Trial/failure accounting audit: `ok`; metrics `{"debug_seed_run_total": 15, "missing_paper_rollout_trial_rows": 1, "released_metric_row_total": 53, "retained_failed_run_count": 1, "row_count": 14, "source_table_real_segments": 24, "source_table_trial_rows": 36, "valid_training_run_count": 0}`, status counts `{"claim_accounting_rows": 1, "debug_seed_runs": 5, "failed_run_retained": 1, "missing_paper_rollout_trials": 1, "released_data_metric_rows": 3, "run_catalog_count": 1, "source_table_count_only": 2}`.
- Metrics catalog: `ok`; metrics `{"blocked_boundary_source_count": 2, "comparison_source_count": 1, "coverage_audit_source_count": 3, "debug_only_source_count": 11, "formula_api_source_count": 2, "released_data_source_count": 4, "source_count": 23, "total_indexed_rows": 456}`, level counts `{"blocked_boundary": 2, "comparison": 1, "coverage_audit": 3, "debug_only": 11, "formula_api": 2, "released_data": 4}`.
- Ablation coverage audit: `ok`; `15` Phase 9 items, group counts `{"diffusion": 9, "motion_tracking": 6}`, status counts `{"debug_ablation_only": 1, "debug_mechanics_only": 1, "debug_or_config_only": 3, "debug_or_formula_only": 1, "debug_or_protocol_only": 1, "public_data_checkpoint_reverse_and_offline_sweep": 1, "public_data_trained_comparison": 1, "released_and_code_audited": 1, "released_data_reproduced": 5}`.
- Public LAFAN1 symmetry training comparison: `ok`; metrics `{"base_checkpoint_size_bytes": 302997114, "base_elapsed_seconds": 148.18651772290468, "base_projection_sha256": "e18fd911ece39bd8b4f880a8fa12c36e0627b6b3191a0ea8f1fa73fff6f59eca", "base_token_count": 46200, "base_window_count": 2200, "metric_summary": {"final_test_decoded_action_mse": {"base": 0.032622650265693665, "delta": -0.006219921633601189, "lower_is_better_improved": true, "relative_delta": -0.19066267096460052, "symmetry_augmented": 0.026402728632092476}, "final_test_pred_tau_mse": {"base": 0.007946033962070942, "delta": 0.0008346522226929665, "lower_is_better_improved": false, "relative_delta": 0.10504010260678957, "symmetry_augmented": 0.008780686184763908}, "final_validation_decoded_action_mse": {"base": 0.02712077833712101, "delta": -0.005032284185290337, "lower_is_better_improved": true, "relative_delta": -0.18555087625942138, "symmetry_augmented": 0.022088494151830673}, "final_validation_pred_tau_mse": {"base": 0.0076224529184401035, "delta": 0.0009422223083674908, "lower_is_better_improved": false, "relative_delta": 0.12361143039507443, "symmetry_augmented": 0.008564675226807594}}, "symmetry_augmented_checkpoint_size_bytes": 302997370, "symmetry_augmented_elapsed_seconds": 172.26538935303688, "symmetry_augmented_projection_sha256": "e18fd911ece39bd8b4f880a8fa12c36e0627b6b3191a0ea8f1fa73fff6f59eca", "symmetry_augmented_token_count": 92400, "symmetry_augmented_window_count": 4400, "token_count_ratio": 2.0, "window_count_ratio": 2.0}`.
- Guidance task coverage audit: `ok`; `30` Phase 8 task-requirement rows, task counts `{"composed_objectives": 5, "inpainting": 5, "joystick": 5, "obstacle_avoidance": 5, "unconditional_rollout": 5, "waypoint": 5}`, status counts `{"blocked_missing_videos": 6, "debug_metric_only": 1, "debug_oracle_reverse_only": 1, "not_applicable_for_unconditional": 2, "public_data_reverse_guidance": 5, "public_data_reverse_guidance_baseline": 5, "public_data_reverse_guidance_metrics": 5, "public_data_reverse_guidance_scale_sweep": 5}`.
- Progress report audit: `ok`; `21` required fields, `17` key progress markers, missing `0`.
- Project boundary audit: `ok`; `8` path/download/cache checks, failures `0`.
- Core test coverage audit: `ok`; `20` explicit `goal.md` checklist items, missing `0`, core-test failures `0`.
- Level A released data: `complete_for_released_dataset_scope` with `13` released-figure rows and `21` panel-map rows.
- Released panel mapping audit: `ok`; metrics `{"expected_released_figure_id_count": 13, "mapped_released_figure_id_count": 13, "paper_panel_map_rows": 21, "released_figure_id_count": 13, "released_panel_fail_count": 0, "released_panel_pass_count": 15, "released_panel_rows": 15, "released_summary_rows": 13, "zip_member_count": 471}`.
- Released-data metrics summary: `ok`; metrics `{"ablation_row_count": 30, "best_global_position_ablation": {"baseline_experiment": "origin", "baseline_mean": 0.2350160799679482, "best_experiment": "quat", "best_mean": 0.2161546877936451, "figure_id": "ablation_orientation_representation"}, "grf_row_count": 12, "imu_duration_s": 6.318044000072405, "imu_row_count": 10, "peak_vertical_grf_abs": 2.316238655046181, "source_csv_count": 10}`.
- Released-data statistical audit: `ok`; metrics `{"ablation_comparison_rows": 30, "best_relative_ablation_improvement": {"baseline_experiment": "origin", "best_experiment": "wn25", "figure_id": "ablation_pd_gain", "metric": "ori_err", "range_based_effect_size": 6.661494579260983, "relative_improvement": 0.12405441378347314, "scope": "local"}, "grf_ci_rows": 12, "imu_ci_rows": 11, "imu_paper_claim_comparison": {"duration_s": 6.318044000072405, "mean_ang_abs_error": 2.1557431036475476, "paper_mean_ang_norm_rad_s": 7.01, "paper_peak_acc_norm_m_s2": 31.0, "paper_peak_ang_norm_rad_s": 20.0, "peak_acc_abs_error": 5.186530611276872, "peak_ang_abs_error": 3.235054179413506, "released_mean_ang_norm_rad_s": 4.854256896352452, "released_peak_acc_norm_m_s2": 36.18653061127687, "released_peak_ang_norm_rad_s": 16.764945820586494}, "source_csv_count": 10}`.
- Level A released-data/table suite: `ok`; `6/6` released-data/table steps passed, metrics `{"ablation_row_count": 30, "grf_ci_rows": 12, "imu_ci_rows": 11, "lafan_rows": 29, "paper_table_mismatch_rows": 0, "paper_table_rows": 58, "released_panel_fail_count": 0, "released_panel_rows": 15, "released_source_csv_count": 10}`.
- Level B tracking/deployment: `partial_blocked_for_live_kit_and_deployment`; live Kit and ROS/deployment gates remain listed below.
- Level B tracking smoke rerun audit: `ok`; metrics `{"inotify_max_user_instances": 128, "inotify_max_user_watches": 8192, "kit_retry_change_watch_failure_count": 17, "kit_retry_errno28_count": 17, "nonkit_log_size_bytes": 1199, "sysctl_output": "fs.inotify.max_user_watches = 8192\nfs.inotify.max_user_instances = 128", "sysctl_return_code": 0, "ulimit_n": "65535", "ulimit_n_return_code": 0}`.
- Level B Kit/inotify budget audit: `ok`; metrics `{"change_watch_failure_count": 17, "df_available_bytes": 1227936489472, "df_output": "Filesystem              1-blocks            Used     Available Capacity Mounted on\nyrfs_nodev:/wwcq 268280837177344 267052900687872 1227936489472     100% /mnt/infini-data", "df_return_code": 0, "directory_count_lower_bound": 522, "elapsed_seconds": 0.13333175517618656, "errno28_count": 17, "instance_limit_target": 1024, "max_user_instances": 10240, "max_user_watches": 1048576, "missing_roots": ["/mnt/infini-data/test/BeyondMimic/envs/isaacsim-4.5.0"], "stop_after": 1048576, "sysctl_output": "fs.inotify.max_user_watches = 1048576\nfs.inotify.max_user_instances = 10240", "sysctl_return_code": 0, "truncated": false, "unique_failed_watch_path_count": 17, "visited_roots": ["/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/IsaacLab-v2.1.0"], "watch_limit_target": 524288}`.
- Level B live inotify usage audit: `ok`; metrics `{"instance_headroom": 10233, "max_watch_process": {"command": "/root/.cursor-server/bin/linux-x64/5702c9cfca656d8710fad58402fe37f14345e3a0/node /root/.cursor-server/bin/linux-x64/5702c9cfca656d8710fad58402fe37f14345e3a0/out/bootstrap-fork --type=fileWatcher", "inotify_fd_count": 1, "inotify_watch_count": 1033585, "pid": 556912, "sample_fdinfo": ["fd=23 watches=1033585 sample=inotify wd:fc572 ino:205989ddfca900cf sdev:13463ba mask:4000fc6 ignored_mask:0 fhandle-bytes:20 fhandle-type:f3 f_handle:1904000034af336abd3400000daf336a650000002e0100000000000003000101"]}, "process_count_with_inotify": 7, "total_inotify_fd_count": 7, "total_inotify_watch_count": 1033626, "watch_headroom": 14950}`.
- VS Code watcher exclude audit: `ok`; snapshot `{"live_still_saturated_after_settings_write": false, "max_watch_process_command": "/root/.cursor-server/bin/linux-x64/5702c9cfca656d8710fad58402fe37f14345e3a0/node /root/.cursor-server/bin/linux-x64/5702c9cfca656d8710fad58402fe37f14345e3a0/out/bootstrap-fork --type=fileWatcher", "max_watch_process_pid": 556912, "max_watch_process_watches": 1033585, "total_inotify_watch_count": 1033626, "watch_headroom": 14950}`.
- Level B Kit watcher config surface audit: `ok`; metrics `{"disable_candidate_count": 0, "failed_paths_overlapping_app_extension_roots": 11, "failed_watch_path_count": 17, "fswatcher_extension_config_count": 12, "fswatcher_paths_config_count": 2, "fswatcher_patterns_config_count": 12, "omni_kit_watched_config_mentions": 24, "python_app_extension_folder_count": 5, "watch_root_token_count": 11}`.
- Level B tracking import gate audit: `ok`; metrics `{"expected_kit_namespace_failure_count": 6, "import_fail_count": 6, "import_ok_count": 1, "module_count": 7, "tracking_python_file_count": 27}`.
- Level B tracking extension namespace probe: `ok`; metrics `{"core_error_changed_from_missing_namespace": true, "core_namespace_path_count": 9, "import_fail_count": 8, "import_ok_count": 0, "isaacsim_path_len_after_append": 10, "kit_runtime_dependency_seen": true, "module_count": 8, "return_code": 0}`.
- Level B official source contract audit: `ok`; metrics `{"critic_term_count": 10, "event_term_count": 4, "policy_term_count": 8, "ppo_max_iterations": 30000, "reward_term_count": 9, "target_body_count": 14, "termination_term_count": 4, "urdf_non_fixed_joint_count": 29, "urdf_uncovered_non_fixed_joint_count": 0}`.
- Level B G1 action-scale audit: `ok`; metrics `{"action_scale_max": 0.5475464652142303, "action_scale_mean": 0.3950213343337363, "action_scale_min": 0.07450087032950714, "actuator_group_count": 5, "armature_max": 0.025101925, "armature_min": 0.003609725, "group_counts": {"arms": 14, "feet": 4, "legs": 8, "waist": 2, "waist_yaw": 1}, "joint_count": 29, "row_count": 29, "stiffness_max": 99.09842777666113, "stiffness_min": 14.25062309787429}`.
- Level B tracking reward formula audit: `ok`; metrics `{"motion_exp_reward_term_count": 6, "numeric_row_count": 30, "regularizer_term_count": 3, "reward_term_count": 9, "scan_distance_count": 5, "term_summary_count": 6}`.
- Level B tracking observation/action schema audit: `ok`; metrics `{"action_dimension": 29, "critic_dimension": 286, "critic_term_count": 10, "fixture_count": 3, "joint_count": 29, "policy_dimension": 160, "policy_term_count": 8, "target_body_count": 14}`.
- Level B tracking randomization/termination audit: `ok`; metrics `{"event_range_row_count": 13, "event_term_count": 4, "interval_event_count": 1, "startup_event_count": 3, "termination_term_count": 4}`.
- Level B non-Kit tracking suite: `ok`; `13/13` steps passed, metrics `{"adaptive_sampling_l1_difference": 1.0730253353204173, "critic_dimension": 286, "debug_onnx_inference_max_abs_error": 0.0, "debug_onnx_sha256": "d7ea68b5dab0d2d667b2ab5f3e4bc2b518ff776ce8f5441764ce4e70c80f45fe", "debug_onnx_size_bytes": 23238, "fixture_count": 3, "g1_action_scale_rows": 29, "local_preflight_steps": 6, "official_target_body_count": 14, "policy_dimension": 160, "randomization_event_terms": 4, "reward_motion_terms": 6}`.
- Level B ONNX export contract: `ok`; metrics `{"exported_but_unused_motion_output_count": 2, "exporter_missing_required_count": 0, "reference_onnx_missing_required_count": 11, "required_input_count": 2, "required_metadata_count": 11, "required_output_count": 7}`.
- Level B motion-policy ONNX contract fixture: `ok`; metrics `{"action_dim": 29, "consumer_unused_output_count": 2, "failed_check_count": 0, "input_count": 2, "joint_count": 29, "metadata_count": 11, "npz_sha256": "a271b026cf85196258ea0bc368288a3ad80a68ef5457020ff74919fc508c2fa9", "npz_size_bytes": 2814, "obs_dim": 160, "output_count": 7, "target_body_count": 14}`; NPZ `/mnt/infini-data/test/BeyondMimic/res/tracking/motion_policy_onnx_contract_fixture/debug_motion_policy_onnx_contract_fixture.npz` is debug-only, not a real trained ONNX.
- Level B debug motion-policy ONNX export: `ok`; ONNX `/mnt/infini-data/test/BeyondMimic/res/tracking/debug_motion_policy_onnx_export/debug_motion_policy_contract.onnx` (23238 bytes, sha256 `d7ea68b5dab0d2d667b2ab5f3e4bc2b518ff776ce8f5441764ce4e70c80f45fe`) matches the contract but is not trained.
- Level B debug motion-policy ONNX inference: `ok`; reference-evaluator metrics `{"max_abs_error": 0.0, "output_count": 7, "reference_evaluator": "onnx.reference.ReferenceEvaluator"}`. This proves graph load/inference for the debug contract only, not a trained policy.
- Level B adaptive sampling discrepancy audit: `ok`; metrics `{"code_argmax": 3, "code_kernel_size": 1, "code_pre_failure_mass_bins_1_2": 0.02272727272727273, "l1_difference": 1.0730253353204173, "override_candidate_count": 0, "paper_argmax": 3, "paper_kernel_size": 3, "paper_pre_failure_mass_bins_1_2": 0.5592399403874814}`.
- Level B motion preprocessing contract audit: `ok`; metrics `{"consumer_key_count": 7, "expected_csv_columns": 36, "g1_csv_count": 40, "g1_joint_count": 29, "max_csv_rows": 8194, "max_quat_norm_abs_error_from_1": 8.877706059173818e-07, "min_csv_rows": 3066, "producer_key_count": 7}`.
- Level B debug motion.npz fixture: `ok`; metrics `{"fixture_count": 3, "joint_count": 29, "max_body_quat_norm_abs_error_from_1": 2.220446049250313e-16, "max_time_steps": 299, "min_time_steps": 299, "total_time_steps": 897, "tracking_body_count": 14, "urdf_body_count": 40}`.
- Level B official replay preflight: `ok`; checks `{"does_not_claim_tracking_reproduction_complete": true, "does_not_execute_csv_to_npz_or_replay": true, "does_not_start_training": true, "live_gate_allows_replay_preflight": true, "local_replay_patch_exists": true, "motion_csv_exists": true, "motion_csv_finite": true, "motion_csv_has_36_columns": true, "motion_csv_has_enough_frames": true, "official_csv_to_npz_exists": true, "official_replay_npz_exists": true, "project_egl_icd_exists": true}`. This plans official conversion/replay commands only; it does not execute rendered replay or PPO.
- Level B official replay conversion attempt: `ok_with_blocked_conversion`; latest blocker `minimal_skeleton_usd_lacks_physical_fidelity_for_replay`; checks `{"app_launcher_layers_permission_to_save_false_recorded": true, "attempt_logs_present": true, "basic_usd_stage_save_failure_recorded": true, "does_not_claim_replay_success": true, "does_not_start_training": true, "g1_in_memory_import_vulkan_blocker_recorded": true, "g1_in_memory_variant_matrix_records_gpu5_gpu6_and_no_valid_usd": true, "g1_layer_save_patch_boundary_recorded": true, "g1_official_urdf_skeleton_usd_audit_recorded": true, "g1_official_urdf_skeleton_usd_contract_ok_recorded": true, "g1_preconverted_asset_audit_recorded": true, "g1_preconverted_asset_audit_separates_mesh_and_reference_usd": true, "g1_reference_usd_compatibility_audit_recorded": true, "g1_reference_usd_compatibility_blocks_drop_in_29dof_replay": true, "g1_simulationapp_in_memory_import_vulkan_blocker_recorded": true, "g1_stage_export_patch_empty_output_recorded": true, "g1_urdf_in_memory_import_probe_recorded": true, "g1_urdf_in_memory_variant_matrix_probe_recorded": true, "g1_urdf_layer_save_workaround_probe_recorded": true, "g1_urdf_simulationapp_in_memory_import_probe_recorded": true, "g1_urdf_stage_export_workaround_probe_recorded": true, "local_csv_to_npz_script_exists": true, "mjcf_bypass_blocked_recorded": true, "mjcf_stage_probe_recorded": true, "motion_npz_contract_written": false, "motion_npz_written": false, "rsl_rl_env_import_ok": true, "rsl_rl_isaaclab_expected_version_installed": true, "rsl_rl_missing_was_repaired": true, "simulationapp_and_applauncher_save_policy_match_recorded": true, "simulationapp_save_policy_probe_recorded": true, "system_libglu_available": true, "tracking_pip_check_ok": true, "urdf_conversion_probe_recorded": true, "urdf_converter_empty_usd_recorded": true, "urdf_path_tiny_probe_recorded": true, "urdf_save_forbidden_and_vulkan_device_lost_recorded": true, "usd_api_variant_probe_recorded": true, "usd_save_blocker_recorded": true, "usd_save_policy_probe_recorded": true, "usd_stage_export_workaround_recorded": true}`. RSL-RL and libGLU environment issues were repaired, but no valid official motion.npz was produced.
- Level B official `replay_npz.py` entry diagnostic: `ok_with_official_replay_npz_entry_blocker`; latest blocker `official_urdf_converter_layer_save_blocked`; summary `{"app_launcher_constructed": true, "blocked_before_artifact_download": true, "empty_robot_after_converter": true, "failed_to_save_layer": true}`. This runs the official replay entrypoint with a local fake-WandB artifact and bounded AppLauncher wrapper without modifying the official worktree. It reaches AppLauncher but blocks in the official URDF converter layer-save path before artifact download or replay-loop execution, leaving an empty robot prim. This is retained failure evidence, not official replay success or paper-level tracking.
- Level B official `csv_to_npz.py` loop with enriched-USD runtime patch: `ok_official_csv_to_npz_loop_with_enriched_usd_patch`; latest blocker `none_official_csv_to_npz_loop_completed_with_enriched_usd_patch`; summary `{"app_launcher_constructed": true, "body_pos_w_shape": [299, 40, 3], "fake_wandb_log_artifact_seen": true, "g1_cfg_patched_to_enriched_usd": true, "joint_pos_shape": [299, 29], "motion_loaded": true, "np_savez_redirect_seen": true, "official_loop_call_299_seen": true, "simulation_app_close_called": true}`. This executes the official csv_to_npz loop body to the 299-step bound, redirects the script's hard-coded `/tmp/motion.npz` output into the project result directory, and replaces wandb with a local fake registry. It remains resource-adjusted because the G1 config is patched in memory to use the validated enriched USD; therefore it is not unpatched official converter output and not paper-level replay/evaluation.
- Level B official `replay_npz.py` loop with enriched-USD runtime patch: `ok_official_replay_loop_with_enriched_usd_patch`; latest blocker `none_official_replay_loop_completed_with_enriched_usd_patch`; summary `{"app_launcher_constructed": true, "fake_wandb_download_seen": true, "g1_cfg_patched_to_enriched_usd": true, "official_loop_call_299_seen": true, "official_loop_complete_seen": true, "simulation_app_close_called": true}`. This executes the official replay loop body to the 299-step bound after patching runtime dependencies only: the G1 robot config uses the validated resource-adjusted enriched USD and a local fake-WandB artifact points to the official-CSV-derived motion. This is stronger than the copied local replay script, but it remains resource-adjusted because the official URDF converter and official `csv_to_npz.py` output are still not validated.
- Level B G1 URDF ImportConfig surface probe: `ok_with_import_config_surface_recorded_and_variants_blocked`; current blocker `official_urdf_converter_layer_save_or_vulkan_device_lost_after_import_config_variants`; summary `{"baseline_joint_count": 0, "baseline_prim_count": 0, "baseline_rigid_body_like_count": 0, "baseline_stage_open_ok": true, "has_set_instanceable_usd_path": false, "has_set_make_instanceable": false}`. In Isaac Sim 4.5 the official URDF import config exposed drive/default-prim setters but no `set_make_instanceable` or instanceable USD path setter, so the attempted Python-level instanceable patch surface is not available. The baseline official G1 URDF conversion produced an openable but empty USD (zero prims, joints, or rigid bodies). This closes one converter-debug path and points the next reproduction work back to runnable replay/task evaluation routes rather than more ImportConfig patching.
- Level B resource-adjusted enriched USD replay preflight: `ok_resource_adjusted_step_gate_passed_with_explicit_exit`; latest blocker `none_resource_adjusted_step_gate_passed_with_explicit_process_exit`; checks `{"bounded_command_executed": true, "clean_kit_shutdown_verified": false, "does_not_claim_official_replay_success": true, "does_not_claim_paper_level_rollout": true, "does_not_start_training": true, "enriched_usd_exists": true, "enriched_usd_readback_ok": true, "entrypoint_exists": true, "explicit_exit_after_success": true, "kit_reached_after_app": true, "kit_shutdown_timeout_after_step_gate": false, "motion_fixture_exists": true, "project_egl_icd_exists": true, "render_step_reached": true, "resource_adjusted_preflight_clean_exit": true, "resource_adjusted_step_gate_passed": true, "robot_contract_reached": true, "scene_creation_reached": true, "sim_context_reached": true, "tracking_python_exists": true}`. This bounded gate directly loads the generated enriched USD through IsaacLab `UsdFileCfg`, reaches `num_joints=29` and `num_bodies=40`, renders four fixture steps on `cuda:6`, and now returns from the bounded gate via an explicit success-after-sentinel process exit. Clean Kit shutdown is still not verified. It is a resource-adjusted environment/articulation gate only, not official csv_to_npz, official motion replay, PPO, DAgger, or paper-level closed-loop evidence.
- Level B resource-adjusted enriched USD bounded replay metrics: `ok_resource_adjusted_64step_metrics_gate`; metrics `{"body_pos_w_shape": [299, 40, 3], "device": "cuda:6", "executed_steps": 64, "fps": 50.0, "joint_pos_shape": [299, 29], "joint_vel_shape": [299, 29], "max_joint_pos_abs_error": 0.0, "max_joint_vel_abs_error": 0.0, "max_root_pos_abs_error": 5.820766091346741e-11, "max_root_quat_abs_error": 0.0, "motion_file": "/mnt/infini-data/test/BeyondMimic/reproduction/data/tracking_motion_npz_fixtures/walk1_subject1_frames_1_180_debug_motion.npz", "motion_total_steps": 299, "official_csv_to_npz_output": false, "paper_level_rollout": false, "requested_steps": 64, "robot_num_bodies": 40, "robot_num_joints": 29, "root_height_max": 0.7965530157089233, "root_height_min": 0.7963541746139526, "usd_path": "/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_enriched_usd/g1_resource_adjusted_29dof_enriched_scaffold.usda", "uses_resource_adjusted_usd": true}`. This extends the resource-adjusted gate to 64 debug-fixture steps, writes root and joint state, and records joint/root consistency metrics on `cuda:6`. It still uses a generated scaffold and debug fixture, so it is not official `csv_to_npz.py` output, official replay/evaluation, PPO, DAgger, or paper-level evidence.
- Level B resource-adjusted official tracking task smoke: `ok_resource_adjusted_tracking_task_smoke`; metrics `{"action_dim": 29, "action_terms": ["joint_pos"], "command_metrics": {"error_anchor_ang_vel": 0.6556865572929382, "error_anchor_lin_vel": 0.6241366863250732, "error_anchor_pos": 0.10726527869701385, "error_anchor_rot": 0.18256878852844238, "error_body_ang_vel": 0.7854078412055969, "error_body_lin_vel": 0.5988677740097046, "error_body_pos": 0.2957192361354828, "error_body_rot": 0.784701943397522, "error_joint_pos": 0.8580849170684814, "error_joint_vel": 0.0, "sampling_entropy": 0.9984743595123291, "sampling_top1_bin": 0.1666666716337204, "sampling_top1_prob": 0.19305363297462463}, "command_terms": ["motion"], "critic_observation_dim": 286, "device": "cuda:6", "event_modes": ["startup", "interval"], "motion_file": "/mnt/infini-data/test/BeyondMimic/reproduction/data/tracking_motion_npz_fixtures/walk1_subject1_frames_1_180_debug_motion.npz", "num_envs": 1, "observation_shapes": {"critic": [1, 286], "policy": [1, 160]}, "observation_terms": {"critic": ["command", "motion_anchor_pos_b", "motion_anchor_ori_b", "body_pos", "body_ori", "base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel", "actions"], "policy": ["command", "motion_anchor_pos_b", "motion_anchor_ori_b", "base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel", "actions"]}, "official_csv_to_npz_output": false, "paper_level_rollout": false, "policy_observation_dim": 160, "ppo_training": false, "reward_max": 0.009331542998552322, "reward_mean": -0.0034458234440535307, "reward_min": -0.025617677718400955, "reward_terms": ["motion_global_anchor_pos", "motion_global_anchor_ori", "motion_body_pos", "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel", "action_rate_l2", "joint_limit", "undesired_contacts"], "robot_num_bodies": 40, "robot_num_joints": 29, "single_action_space": "Box(-inf, inf, (29,), float32)", "single_observation_space": "Dict('policy': Box(-inf, inf, (160,), float32), 'critic': Box(-inf, inf, (286,), float32))", "step_count": 8, "task": "Tracking-Flat-G1-v0", "terminated_total": 8, "termination_terms": ["time_out", "anchor_pos", "anchor_ori", "ee_body_pos"], "truncated_total": 0, "usd_path": "/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_enriched_usd/g1_resource_adjusted_29dof_enriched_scaffold.usda", "uses_resource_adjusted_usd": true}`. This instantiates the official `Tracking-Flat-G1-v0` ManagerBasedRLEnv stack with the generated enriched USD and debug fixture, reaches reset, performs eight zero-action steps, and verifies action dimension 29, policy observation dimension 160, critic observation dimension 286, nine reward terms, and four termination terms. It is a resource-adjusted task smoke/eval gate, not official replay/evaluation, PPO, DAgger, or paper-level tracking evidence.
- Level B resource-adjusted official tracking task full fixture eval: `ok_resource_adjusted_multi_fixture_task_eval`; metrics `{"action_dim_all_29": true, "critic_observation_dim_all_286": true, "fixture_count": 3, "policy_observation_dim_all_160": true, "reward_terms_all_9": true, "robot_num_bodies_all_40": true, "robot_num_joints_all_29": true, "termination_terms_all_4": true, "total_steps": 897}`. This runs the official `Tracking-Flat-G1-v0` manager stack for all available steps in the three local debug fixtures (`walk`, `run`, `jump`), using isolated Kit processes to avoid observed teardown/recreate hangs, and records per-fixture rewards and termination counts. It is stronger resource-adjusted task-contract evidence than the eight-step smoke, but it is still not official `csv_to_npz.py` conversion, official replay/evaluation, PPO training, DAgger rollout data, or paper-level closed-loop tracking performance.
- Level B resource-adjusted official-CSV conversion gate: `ok_resource_adjusted_csv_conversion`; metrics `{"body_pos_w_shape": [299, 40, 3], "joint_pos_shape": [299, 29], "npz_size_bytes": 693078, "root_height_max": 0.7965530157089233, "root_height_min": 0.7617475986480713}`. This converts the downloaded official G1 LAFAN `walk1_subject1.csv` frame range 1-180 into a 299-step `motion.npz` using the official interpolation/logging schema plus the generated enriched G1 USD. It narrows the replay blocker to the official URDF/USD conversion path, but the resulting `motion.npz` is explicitly resource-adjusted and must not be reported as official `csv_to_npz.py` output.
- Level B resource-adjusted official-CSV full replay gate: `ok_resource_adjusted_csv_full_replay`; metrics `{"body_pos_w_shape": [299, 40, 3], "executed_steps": 299, "joint_pos_shape": [299, 29], "max_joint_pos_abs_error": 0.0, "max_root_pos_abs_error": 0.0, "motion_total_steps": 299, "root_height_max": 0.7965530157089233, "root_height_min": 0.7617475986480713}`. This replays the official-CSV-derived resource-adjusted motion for all 299 steps through the enriched USD replay surface and records zero joint/root write-read errors. It is still not official replay/evaluation, PPO, DAgger, or paper-level tracking performance.
- Level B resource-adjusted official-CSV tracking task eval: `ok_resource_adjusted_csv_task_eval`; metrics `{"action_dim": 29, "critic_observation_dim": 286, "policy_observation_dim": 160, "reward_max": 0.052452173084020615, "reward_mean": 0.02670689582525687, "reward_min": -0.010335305705666542, "robot_num_bodies": 40, "robot_num_joints": 29, "step_count": 299, "terminated_total": 26, "truncated_total": 12}`. This feeds the official-CSV-derived resource-adjusted `motion.npz` into the official `Tracking-Flat-G1-v0` ManagerBasedRLEnv stack for all 299 available steps and verifies action, observation, reward, termination, and robot-contract dimensions. It uses zero diagnostic actions and a generated enriched USD, so termination counts are not policy-quality evidence and the result is not official replay/evaluation or PPO.
- Level B resource-adjusted RSL-RL train-entry diagnostic: `ok_resource_adjusted_train_entry_diagnostic`; metrics `{"checkpoint_written": false, "configured_num_steps_per_env": 4, "num_actions": 29, "num_envs": 1, "num_obs": 160, "num_privileged_obs": 286, "requested_learning_iterations": 1, "runner_class": "MotionOnPolicyRunner", "runner_training_type": "rl"}`. This constructs the official `Tracking-Flat-G1-v0` env, wraps it with `RslRlVecEnvWrapper`, instantiates the official custom `MotionOnPolicyRunner`, and executes one tiny PPO learning iteration with four rollout steps. It verifies train-entry wiring only: no checkpoint is written, it is not formal PPO training, and it is not paper-level tracking performance. Runtime warning: The probe log contains PhysX GPU convex narrowphase kernel launch errors before the success sentinel.
- Level B resource-adjusted PPO training run: `ok_resource_adjusted_ppo_training_completed`; summary `{"attempted_training": true, "candidate_physical_gpus": [4, 5, 6, 7], "checkpoint_count": 3, "max_iterations": 100, "num_steps_per_env": 24, "resource_ready": true, "selected_physical_gpus": [4, 7], "total_num_envs": 1024, "world_size": 2}`. The harness selects available GPUs from physical GPUs 4-7 and launches `torch.distributed` with the official `Tracking-Flat-G1-v0` manager stack, official PPO rollout length, GPU telemetry, checkpoints, and run metadata. The current run completed 100 resource-adjusted iterations on GPUs selected by preflight. The asset/motion path remains resource-adjusted, so this is evidence of virtual training execution and not official paper-level PPO training or a validated BeyondMimic teacher.
- Level B resource-adjusted PPO checkpoint evaluation: `ok_resource_adjusted_ppo_checkpoint_eval_completed`; summary `{"duration_seconds": 1193.852, "error_anchor_pos_mean": 0.10595783163921091, "error_body_pos_mean": 0.18350737062859096, "error_joint_pos_mean": 1.2326450995776965, "eval_steps": 299, "loaded_iteration": 99, "num_envs": 512, "reward_mean": 0.025898515209431035, "selected_physical_gpus": [4, 7], "total_env_steps": 153088}`. The evaluator loads `model_99.pt` with the official RSL-RL `OnPolicyRunner` inference API and runs `Tracking-Flat-G1-v0` for 512 environments x 299 steps while recording reward, termination, action, GPU, and motion-command tracking metrics. This is useful virtual policy-evaluation evidence, but it remains resource-adjusted and below official paper-level tracking evaluation.
- Level B resource-adjusted teacher rollout dataset gate: `ok_resource_adjusted_teacher_rollout_dataset_completed`; summary `{"dataset_npz_total_size_bytes": 514367800, "duration_seconds": 650.478, "gpu_metrics_summary": {"exists": true, "per_gpu": {"4": {"mean_utilization_gpu_percent": 91.84615384615384, "memory_total_mb": 97871.0, "peak_memory_used_mb": 6775.0, "samples": 130}, "7": {"mean_utilization_gpu_percent": 93.23846153846154, "memory_total_mb": 97871.0, "peak_memory_used_mb": 6765.0, "samples": 130}}, "row_count": 260}, "num_envs_per_rank": 512, "reward_mean_by_rank": [0.02579653076827526, 0.025758858770132065], "rollout_steps": 299, "selected_physical_gpus": [4, 7], "shard_count": 2, "total_env_steps": 306176, "world_size": 2}`. The run used fixed physical GPUs 4 and 7, collected two raw `.npz` shards under ignored `res/runs`, and records policy observations, critic observations, actions, rewards, dones, timeouts, and motion timesteps from the local `model_99.pt` resource-adjusted teacher. This is suitable as a local downstream dataset candidate for VAE/state-latent experiments, but it is not the official BeyondMimic DAgger rollout log and does not validate paper-level Fig. 5/Fig. 6 closed-loop diffusion results.
- Level B G1 URDF conversion probe: `ok_with_urdf_usd_blocker`; payload `{"converter_usd_exists": true, "converter_usd_mode": "0o567", "converter_usd_path": "/mnt/infini-data/test/BeyondMimic/tmp/isaaclab_urdf_probe/g1/g1_probe.usd", "converter_usd_size": 492, "default_prim_path": null, "default_prim_valid": false, "prim_count": 0, "rigid_body_like_count": 0, "stage_open_ok": true, "stage_save_ok_after_open": true, "urdf": "/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf", "usd_dir": "/mnt/infini-data/test/BeyondMimic/tmp/isaaclab_urdf_probe/g1"}`. The isolated converter opens a tiny USD but records zero traversed prims and no valid default prim.
- Level B URDF path/tiny contrast probe: `ok_with_blocker_classified`; current blocker `usd_layer_save_forbidden_and_vulkan_device_lost_before_payload`; markers `{"libglu_missing": false, "p2p_iommu_warning": true, "segmentation_fault": false, "sentinel_after_app": true, "sentinel_after_close": false, "sentinel_payload": false, "timed_out": true, "traceback": false, "usd_save_not_allowed": true, "vulkan_device_lost": true}`. This shows the official replay blocker is now localized to Isaac Sim URDF USD write/runtime behavior, not to missing local G1 mesh files.
- Level B MJCF/stage bypass probe: `ok_with_blocker_classified`; current blocker `mjcf_or_stage_usd_save_forbidden_and_vulkan_device_lost`; checks `{"all_g1_mjcf_mesh_refs_resolve_statically": true, "app_launcher_closed_or_timeout_recorded": true, "app_launcher_payload_or_blocker_recorded": true, "app_launcher_reached_after_app": true, "does_not_claim_motion_npz": true, "does_not_start_replay_or_training": true, "g1_mjcf_conversion_success": false, "g1_mjcf_exists": true, "libglu_missing_absent": true, "minimal_stage_save_success": false, "project_egl_icd_exists": true, "tiny_mjcf_conversion_success": false, "tracking_python_exists": true}`. The minimal USD stage save itself fails with a save-forbidden error, and both tiny MJCF and official G1 MJCF conversion produce empty USD layers, so the blocker is below the URDF/MJCF asset-format choice.
- Level B USD save-policy probe: `ok_with_blocker_classified`; current blocker `app_launcher_layers_permission_to_save_false`; counts `{"export_ok_count": 0, "force_save_ok_count": 0, "permission_false_count": 18, "save_ok_count": 0}`. Plain `bm_tracking` Python cannot import `pxr`, while AppLauncher can import `pxr` but creates all tested local layers with `permissionToSave=False`; Save, Export, and SetPermissionToSave(True) attempts all fail.
- Level B SimulationApp/AppLauncher save-policy comparison: `ok_with_blocker_classified`; current blocker `isaaclab_headless_experience_layers_permission_to_save_false_with_isaacsim_base_vulkan_crash`; cases `[{"force_after_false_count": 0, "markers": {"p2p_iommu_warning": true, "sentinel_after_app": false, "sentinel_after_close": false, "sentinel_payload": false, "timed_out": false, "traceback": false, "usd_save_not_allowed": false, "vulkan_device_lost": true}, "name": "simulationapp_isaacsim_base_python", "permission_false_count": 0, "returncode": -11, "save_ok_count": 0}, {"force_after_false_count": 1, "markers": {"p2p_iommu_warning": true, "sentinel_after_app": true, "sentinel_after_close": true, "sentinel_payload": true, "timed_out": false, "traceback": false, "usd_save_not_allowed": true, "vulkan_device_lost": false}, "name": "simulationapp_isaaclab_headless", "permission_false_count": 3, "returncode": 0, "save_ok_count": 0}, {"force_after_false_count": 1, "markers": {"p2p_iommu_warning": true, "sentinel_after_app": true, "sentinel_after_close": true, "sentinel_payload": true, "timed_out": false, "traceback": false, "usd_save_not_allowed": true, "vulkan_device_lost": false}, "name": "applauncher_isaaclab_headless", "permission_false_count": 3, "returncode": 0, "save_ok_count": 0}]`. Raw SimulationApp with the IsaacLab headless experience reaches payload and shows the same local USD permissionToSave=False behavior as AppLauncher; the Isaac Sim base python experience records a Vulkan device-lost crash before payload. This keeps the official replay gate blocked and does not produce motion.npz.
- Level B USD API variant probe: `ok_with_stage_export_workaround`; current blocker `layer_save_blocked_but_stage_export_succeeds`; successful write APIs `["create_new_stage_export", "create_in_memory_stage_export", "sdf_layer_create_anonymous_export"]`. `layer.Save()` remains blocked by `permissionToSave=False`, but direct `Usd.Stage.Export(...)` paths write non-empty local USD files. This is a concrete next-step workaround for conversion plumbing, not official replay success.
- Level B G1 URDF Stage.Export workaround probe: `ok_with_importer_still_empty_after_stage_export_patch`; current blocker `stage_export_patch_applied_but_importer_output_empty`; parse result `(True, '/g1/pelvis')`; patch events `[{"method": "CreateNew", "path": "/mnt/infini-data/test/BeyondMimic/tmp/g1_urdf_stage_export_workaround/g1_parse_and_import_stage_export.usd"}, {"method": "Save", "path": "/mnt/infini-data/test/BeyondMimic/tmp/g1_urdf_stage_export_workaround/g1_parse_and_import_stage_export.usd", "routed_to": "Usd.Stage.Export"}]`. The importer's initial `Stage.Save()` was routed to `Stage.Export()`, but the generated G1 destination and current stages still contain no robot prims because deeper base/physics/sensor layer saves remain blocked. This is a narrower blocker classification, not official replay success.
- Level B G1 URDF Sdf.Layer.Save workaround probe: `ok_with_cpp_importer_save_path_not_intercepted`; current blocker `sdf_layer_save_patch_applied_but_cpp_importer_save_path_not_intercepted`; parse result `(True, '/g1/pelvis')`; layer-save patch exception `None`; checks `{"app_reached_after_app": true, "configuration_layer_count": 3, "configuration_layer_robotish_count": 0, "current_stage_export_has_robot": false, "current_stage_has_robot": false, "dest_stage_has_robot": false, "direct_layer_save_patch_test_opened": true, "does_not_claim_motion_npz": true, "does_not_start_replay_or_training": true, "g1_urdf_exists": true, "importer_configuration_layer_save_intercepted": false, "layer_save_events_recorded": true, "payload_recorded": true, "project_egl_icd_exists": true, "sdf_layer_save_patch_assignment_ok": true, "stage_create_new_patch_assignment_ok": true, "tracking_python_exists": true, "urdf_extension_enabled": true}`. This probes the deeper Python-visible layer-save boundary for the importer configuration layers. It is not official replay success and produces no motion.npz unless the resulting USD contains a valid robot stage.
- Level B G1 URDF in-memory import probe: `ok_with_vulkan_device_lost_before_payload`; current blocker `in_memory_import_vulkan_device_lost_before_payload`; parse result `None`; markers `{"sentinel_after_app": true, "sentinel_after_close": false, "sentinel_payload": false, "timed_out": true, "traceback": false, "usd_save_not_allowed": false, "vulkan_device_lost": true}`. This tries `dest_path=""` so the URDF importer uses the current in-memory Kit stage instead of layered file output. It currently records Vulkan device loss before an exported robot stage can be captured, so it is not official replay success.
- Level B G1 URDF SimulationApp in-memory import probe: `ok_with_vulkan_device_lost_before_payload`; current blocker `simulationapp_in_memory_import_vulkan_device_lost_before_payload`; return code `-11`; markers `{"in_memory_stage_branch": true, "p2p_iommu_warning": true, "sentinel_after_app": true, "sentinel_after_close": false, "sentinel_payload": false, "timed_out": false, "traceback": false, "usd_save_not_allowed": false, "vulkan_device_lost": true}`. This repeats the `dest_path=""` URDF importer test under raw `SimulationApp` with the IsaacLab headless experience. It reaches the in-memory importer branch but crashes with Vulkan device loss before payload, so the blocker is now localized below the AppLauncher wrapper and remains a Kit/GPU runtime gate.
- Level B G1 URDF in-memory variant matrix: `ok_with_no_valid_g1_usd`; current blocker `variant_matrix_no_valid_g1_usd`; cases `[{"current_blocker": "variant_vulkan_device_lost_before_payload", "gpu": 6, "log": "/mnt/infini-data/test/BeyondMimic/logs/tracking_g1_urdf_in_memory_variant_matrix/gpu6_headless_single_gpu.log", "markers": {"in_memory_stage_branch": true, "p2p_iommu_warning": true, "sentinel_after_app": true, "sentinel_after_close": false, "sentinel_payload": false, "timed_out": false, "traceback": false, "usd_save_not_allowed": false, "vulkan_device_lost": true}, "name": "gpu6_headless_single_gpu", "returncode": -11, "status": "ok_with_vulkan_device_lost_before_payload"}, {"current_blocker": "variant_vulkan_device_lost_before_payload", "gpu": 5, "log": "/mnt/infini-data/test/BeyondMimic/logs/tracking_g1_urdf_in_memory_variant_matrix/gpu5_headless_single_gpu.log", "markers": {"in_memory_stage_branch": true, "p2p_iommu_warning": true, "sentinel_after_app": true, "sentinel_after_close": false, "sentinel_payload": false, "timed_out": false, "traceback": false, "usd_save_not_allowed": false, "vulkan_device_lost": true}, "name": "gpu5_headless_single_gpu", "returncode": -11, "status": "ok_with_vulkan_device_lost_before_payload"}, {"current_blocker": "variant_vulkan_device_lost_before_payload", "gpu": 6, "log": "/mnt/infini-data/test/BeyondMimic/logs/tracking_g1_urdf_in_memory_variant_matrix/gpu6_headless_wait_idle_low_rtx.log", "markers": {"in_memory_stage_branch": true, "p2p_iommu_warning": true, "sentinel_after_app": true, "sentinel_after_close": false, "sentinel_payload": false, "timed_out": false, "traceback": false, "usd_save_not_allowed": false, "vulkan_device_lost": true}, "name": "gpu6_headless_wait_idle_low_rtx", "returncode": -11, "status": "ok_with_vulkan_device_lost_before_payload"}, {"current_blocker": "variant_failed_before_classification", "gpu": 6, "log": "/mnt/infini-data/test/BeyondMimic/logs/tracking_g1_urdf_in_memory_variant_matrix/gpu6_headless_rendering_experience.log", "markers": {"in_memory_stage_branch": false, "p2p_iommu_warning": false, "sentinel_after_app": false, "sentinel_after_close": false, "sentinel_payload": false, "timed_out": false, "traceback": false, "usd_save_not_allowed": false, "vulkan_device_lost": true}, "name": "gpu6_headless_rendering_experience", "returncode": -11, "status": "failed_before_classification"}]`. This tests GPU 5, GPU 6, waitIdle/low-RTX settings, and the IsaacLab headless-rendering experience. It produces no valid G1 USD, so the official replay gate remains blocked and no motion.npz/replay result is claimed.
- Level B G1 URDF in-memory GPU4 probe: `ok_with_vulkan_device_lost_blocker`; return code `-9`; duration `35.04` seconds; checks `{"app_launcher_reached": true, "does_not_claim_motion_npz": true, "does_not_claim_paper_level_replay": true, "does_not_start_replay_or_training": true, "export_exists": false, "export_has_joints": false, "export_has_rigid_bodies": false, "g1_urdf_exists": true, "in_memory_import_returned": false, "payload_recorded": false, "project_egl_icd_exists": true, "tracking_python_exists": true}`. This repeats the official G1 URDF importer `dest_path=""` path on the current GPU4 headless setup. It reaches AppLauncher and begins URDF parsing, but Vulkan `ERROR_DEVICE_LOST` kills the process before import return or stage export. It is therefore blocker evidence only, not official replay or paper-level tracking.
- Level B G1 preconverted asset audit: `ok_with_reference_usd_candidate`; counts `{"candidate_count": 65, "official_full_robot_preconverted_g1_usd_count": 0, "official_mesh_usd_count": 35, "reference_g1_usd_count": 1, "usd_candidate_count": 36, "validated_reference_robotish_usd_count": 1}`; validated reference USD `[{"has_robotish_stage": true, "payload_summary": {"articulation_api_count": 1, "default_prim_path": "/g1_29dof", "joint_count": 38, "pelvis_path_present": true, "prim_count": 161, "rigid_body_like_count": 39, "stage_open_ok": true, "torso_path_present": true}, "relative_path": "download/reference_code/ASAP/humanoidverse/data/robots/g1/g1_29dof_anneal_23dof.usd", "usable_as_official_beyondmimic_asset": false}]`. The official whole_body_tracking work copy contains mesh-level USD files but no official full-robot preconverted G1 USD. A reference-code ASAP G1 USD opens as a robot-like stage in Kit, but it is explicitly not an official BeyondMimic replay asset and can only be used, if at all, as a clearly labeled resource-adjusted workaround.
- Level B G1 reference USD compatibility audit: `ok_with_reference_usd_incompatible_or_partial`; compatible for resource-adjusted replay `False`; official contract `{"action_joint_count": 29, "anchor_body": "torso_link", "link_count": 40, "non_fixed_joint_count": 29, "target_bodies": ["pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link", "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link", "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link", "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link"]}`; reference contract `{"articulation_api_count": 1, "default_prim_path": "/g1_29dof", "joint_count": 38, "link_count": 39, "revolute_joint_count": 23, "rigid_body_count": 39, "stage_open_ok": true}`; missing action joints `['left_wrist_pitch_joint', 'left_wrist_roll_joint', 'left_wrist_yaw_joint', 'right_wrist_pitch_joint', 'right_wrist_roll_joint', 'right_wrist_yaw_joint']`. All official target bodies are present, but the six wrist action joints are fixed rather than revolute in the reference USD, so it is not a drop-in 29-DoF BeyondMimic replay asset.
- Level B official-URDF minimal skeleton USD audit: `ok_with_minimal_29dof_skeleton_usd`; contract ok `True`; official contract `{"action_joint_count": 29, "fixed_joint_count": 10, "link_count": 40, "non_fixed_joint_count": 29, "target_body_count": 14}`; skeleton contract `{"articulation_api_count": 1, "default_prim_path": "/g1_29dof_skeleton", "fixed_joint_count": 10, "link_count": 40, "revolute_joint_count": 29, "rigid_body_count": 40, "usd_size_bytes": 18441}`. This local USD preserves the official 40-link/29-revolute-joint/14-target-body naming contract and is validated by a read-only Kit probe, but it is a placeholder structure asset without official converter success, meshes, collisions, inertias, drives, motion.npz, replay, or training evidence.
- Level B G1 URDF physical asset contract audit: `ok_with_physical_contract_ready_for_converter_scaffold`; metrics `{"collision_element_count": 29, "collision_link_count": 14, "collision_type_counts": {"cylinder": 28, "sphere": 1}, "fixed_joint_count": 10, "inertial_link_count": 37, "joint_count": 39, "link_count": 40, "missing_inertial_link_count": 3, "missing_mesh_reference_count": 0, "nonfixed_joint_count": 29, "target_body_count": 14, "target_body_missing_inertial_count": 0, "visual_mesh_reference_count": 35}`. The official URDF provides all 35 visual mesh references, 29 collision elements, and all 29 non-fixed joint axis/limit/action-drive rows needed for an offline USD converter scaffold. Three sensor/IMU links lack inertial tags, no target body lacks inertial data, and no physical USD, motion.npz, replay, or training success is claimed.
- Level B G1 URDF source-equivalence audit: `ok_with_source_differences_recorded`; summary `{"action_joint_summary": {"action_scale_joint_count": 29, "action_scale_vs_wbt_nonfixed_diff": {"extra_in_right": [], "missing_from_right": []}, "download_nonfixed_joint_count": 29, "download_vs_wbt_nonfixed_diff": {"extra_in_right": [], "missing_from_right": []}, "whole_body_tracking_nonfixed_joint_count": 29}, "download_vs_wbt_joint_diff": {"extra_in_right": ["LL_FOOT_frame", "LR_FOOT_frame"], "missing_from_right": ["d435_joint"]}, "download_vs_wbt_link_diff": {"extra_in_right": ["LL_FOOT", "LR_FOOT"], "missing_from_right": ["d435_link"]}}`. The downloaded official LAFAN G1 URDF and the reproduction-data copy are byte-identical and structurally identical. The official `whole_body_tracking` G1 URDF keeps the same 29 non-fixed/action joints, but differs in support links/joints (`d435_link/d435_joint` versus `LL_FOOT/LR_FOOT` foot frames) and physical bookkeeping. This improves source traceability for the offline scaffold while explicitly avoiding any claim of identical URDF sources, official converter success, motion.npz, replay, or paper-level tracking.
- Level B resource-adjusted enriched G1 USD scaffold probe: `ok_with_resource_adjusted_enriched_usd_scaffold`; readback `{"articulation_api_count": 1, "collision_api_count": 29, "collision_proxy_count": 29, "joint_drive_metadata_count": 29, "joint_limit_count": 29, "joint_origin_metadata_count": 39, "link_count": 40, "mass_api_count": 37, "prim_count": 285, "revolute_joint_count": 29, "visual_mesh_reference_count": 35, "visual_proxy_count": 35}`. The generated scaffold authors public URDF mass/inertia metadata, visual mesh references, collision proxy geometry, joint limits, and drive metadata onto the 29-DoF skeleton. It is still explicitly not official URDF converter output and has not passed official csv_to_npz/replay validation.
- Level B local tracking smoke preflight: `ok`; `6/6` non-Kit steps passed.
- Level B official train entry retry: `ok`; classification `{"has_inotify_watch_failure": true, "has_no_space_left_on_device": true, "module_missing": false, "reason": "The retry reproduced Kit/inotify watcher-budget failure signatures.", "retry_result": "blocked_inotify", "returncode": 124, "timed_out": true}`.
- Level B MuJoCo/ROS launch contract: `ok`; metrics `{"controller_type_count": 3, "declared_dependency_count": 10, "manager_update_rate_hz": 500, "mujoco_launch_argument_count": 5, "readme_ros2_launch_command_count": 4, "real_launch_argument_count": 6, "standby_joint_count": 29, "walking_update_rate_hz": 50}`; host `{"colcon_path": null, "python_version": "3.8.10", "ros2_path": null, "rosdep_path": null, "ubuntu_codename": "focal", "ubuntu_version_id": "20.04"}`.
- Level B deployment controller semantics: `ok`; metrics `{"audited_source_file_count": 14, "controller_type_count": 3, "declared_dependency_count": 10, "failed_check_count": 0, "manager_update_rate_hz": 500, "motion_command_dim_for_29_joints": 58, "motion_observation_dim_for_14_target_bodies": 135, "motion_observation_dim_formula_for_n_bodies": "9 + 9 * body_names_count", "standby_joint_count": 29, "walking_update_rate_hz": 50}`; host `{"colcon_path": null, "python_version": "3.8.10", "ros2_path": null, "rosdep_path": null, "ubuntu_codename": "focal", "ubuntu_version_id": "20.04"}`.
- Level C VAE/diffusion: `debug_mechanics_and_audits_only`; official Level C code found `False`, checkpoint/engine found `False`.
- Level C debug suite: `ok`; `10/10` lightweight VAE/diffusion/guidance/debug-action steps passed, metrics `{"decoder_current_index": 4, "guided_cost_improvement_vs_unguided_final": 0.003971231909915218, "guided_final_mse": 2.377871825333812e-09, "paper_state_reverse_final_mse": 5.232890366443532e-10, "reverse_final_mse": 5.253923735449295e-10, "reverse_initial_mse": 0.11735366650108521, "test_predicted_current_action_mse": 0.0039371201747736865, "vae_accumulation_optimizer_steps": 1, "validation_predicted_current_action_mse": 0.00911132119759807}`.
- Diffusion equation audit: `ok`; exact coefficient schedule missing from public source `True`.
- Trajectory inverse transform audit: `ok`; root/body round-trip checks `{"all_windows_checked": true, "existing_debug_fixture_not_full_paper_root_window_state": true, "paper_formula_body_inverse_roundtrip": true, "paper_formula_root_inverse_roundtrip": true, "paper_formula_root_rotation_roundtrip": true}`.
- Emphasis projection audit: `ok`; metrics `{"gaussian_a_mean": -0.06525527906019948, "gaussian_a_std": 1.0014479557360592, "input_sample_count": 1764, "max_roundtrip_abs_error": 2.4077961846558082e-15, "mean_body_token_norm": 1.750899663031407, "mean_projected_extra_norm": 72.70233551700606, "mean_projected_identity_norm": 2.3083107744575857, "mean_root_token_norm": 1.493740775819515, "mean_roundtrip_abs_error": 1.2137705621338285e-16, "pinv_projection_identity_max_error": 1.713694935129439e-15, "projection_rank": 99, "projection_shape": [163, 99], "state_dim": 99}`.
- State representation source audit: `ok`; metrics `{"max_body_position_local_error": 1.3877787807814457e-17, "max_body_velocity_local_error": 1.8108774263211476, "max_root_current_frame_feature_error": 1.1687931299897365}`.
- Dataset collection protocol audit: `ok`; metrics `{"debug_motion_count": 3, "debug_sample_count": 84, "missing_paper_requirement_count": 8, "paper_state_token_count": 1764, "paper_state_window_count": 84, "rollout_manifest_episode_rows": 15000, "rollout_manifest_recorded_coverage_min_nonzero": 100, "rollout_manifest_valid_start_count": 150}`.
- Rollout rejection manifest probe: `ok`; metrics `{"debug_accept_count": 15000, "debug_reject_count": 0, "episode_manifest_rows": 15000, "motion_count": 3, "preview_ou_lag1_autocorr_min": 0.19084534673349865, "recorded_coverage_central_min": 100, "recorded_coverage_min_nonzero": 100, "recorded_coverage_target": 100, "valid_start_count_total": 150}`.
- State-latent schema audit: `ok`; `84` windows, split counts `{"test": 28, "train": 28, "validation": 28}`, token shapes `{"[21, 131]": 84}`.
- DAgger schema audit: `ok`; `30` synthetic teacher-query samples, split counts `{"test": 5, "train": 20, "validation": 5}`, metrics `{"accepted_count": 30.0, "action_mse": 0.0012922259414464622, "action_rmse": 0.03594754430342165, "evaluation_action_mse": 0.001292225941446462, "max_abs_action_error": 0.11045633198206498, "sample_count": 30.0, "teacher_query_count": 30.0}`.
- DAgger iteration smoke: `ok`; metrics `{"accepted_count": 288.0, "final_aggregate_action_mse": 1.0974932295569283e-12, "final_heldout_action_mse": 6.111142407404412e-12, "final_max_abs_action_error": 4.603872716724133e-06, "heldout_mse_reduction_ratio": 0.9999999999807272, "initial_heldout_action_mse": 0.3170884047315886, "total_samples": 288.0, "total_teacher_queries": 288.0}`.
- Paper-formula state windows: `ok`; counts `{"motion_count": 3, "sample_count": 1764, "window_count": 84}`.
- State-latent dataset consistency audit: `ok`; metrics `{"action_dim": 29, "latent_abs_mean": 0.1456600978669161, "latent_dim": 32, "max_state_abs_error_between_paper_windows_and_vae_npz": 1.1409430689113265e-07, "max_target_action_abs_error_between_action_npz_and_decoded_action": 0.0, "per_split_counts": {"test": 28, "train": 28, "validation": 28}, "row_count": 84, "state_dim": 99, "token_dim": 131}`.
- State-latent training dataset contract audit: `ok`; metrics `{"consistency_row_count": 84, "contract_row_count": 12, "dagger_schema_sample_count": 30.0, "debug_latent_abs_mean": 0.1456600978669161, "debug_latent_shape_counts": {"21x32": 84}, "debug_npz_key_count": 423, "debug_npz_sample_count": 84, "debug_state_shape_counts": {"21x99": 84}, "failed_check_count": 0, "missing_or_debug_only_count": 9, "paper_trainable_satisfied_count": 3, "rollout_manifest_episode_rows": 15000}`.
- Paper-state overfit gate: `ok`; metrics `{"all_paper_state_windows_baseline_loss": 0.06703983482819055, "all_paper_state_windows_loss_reduction_ratio": 1.0, "all_paper_state_windows_overfit_loss": 1.830447187210791e-18, "paper_state_dim": 99, "token_dim": 131}`.
- Debug-VAE-latent diffusion overfit gate: `ok`; metrics `{"all_debug_vae_latent_windows_baseline_loss": 0.06669130873528159, "all_debug_vae_latent_windows_loss_reduction_ratio": 1.0, "all_debug_vae_latent_windows_overfit_loss": 3.341915582546504e-19, "latent_source": "debug_tiny_vae_mu_nonzero_synthetic_teacher", "token_dim": 131}`.
- Paper-state held-out eval: `ok`; metrics `{"paper_state_dim": 99, "test_loss_reduction_ratio": 0.48728032634615426, "test_prediction_loss": 0.03338779974210115, "token_dim": 131, "train_prediction_loss": 0.013992778144063207, "validation_loss_reduction_ratio": 0.5431669271426506, "validation_prediction_loss": 0.030197348642920434}`.
- Debug-VAE-latent held-out eval: `ok`; metrics `{"latent_source": "debug_tiny_vae_mu_nonzero_synthetic_teacher", "test_loss_reduction_ratio": 0.8929658771029971, "test_prediction_loss": 0.007374265934471514, "token_dim": 131, "train_prediction_loss": 0.002362316228775186, "validation_loss_reduction_ratio": 0.8892813944733895, "validation_prediction_loss": 0.007872506845332104}`.
- Paper-state held-out multi-seed audit: `ok`; statistics `{"test_loss_reduction_ratio": {"max": 0.49844224740020926, "mean": 0.48924922500559204, "min": 0.4820251012704126, "std": 0.008383799449541141}, "test_prediction_loss": {"max": 0.034748334464957, "mean": 0.03381880860919664, "min": 0.03332029162053179, "std": 0.0008057003624268288}, "validation_loss_reduction_ratio": {"max": 0.5586566906336525, "mean": 0.5475202271651117, "min": 0.5407370637190321, "std": 0.009720682743715466}, "validation_prediction_loss": {"max": 0.030445831099325714, "mean": 0.030242310606480482, "min": 0.030083752077195297, "std": 0.00018517961062738787}}`.
- Debug-VAE-latent held-out multi-seed audit: `ok`; statistics `{"test_loss_reduction_ratio": {"max": 0.8929658771029971, "mean": 0.8925315111873378, "min": 0.8922359390329153, "std": 0.0003842527523094619}, "test_prediction_loss": {"max": 0.007374265934471514, "mean": 0.007036712230498105, "min": 0.00677377954138323, "std": 0.00030711919341671327}, "validation_loss_reduction_ratio": {"max": 0.8892813944733895, "mean": 0.8860020143594977, "min": 0.8817359731575921, "std": 0.0038682645702139994}, "validation_prediction_loss": {"max": 0.007872506845332104, "mean": 0.007586554173806038, "min": 0.0073889897571831455, "std": 0.00025358737416644636}}`.
- Level C extended debug suite: `ok`; `10/10` steps passed, metrics `{"action_dim": 29, "action_multiseed_test_current_mse_mean": 0.003998468510847306, "action_smoothness_test_penalty": 0.01579856790234264, "action_test_current_mse": 0.0039371201747736865, "latent_dim": 32, "small_dataset_heldout_test_reduction_mean": 0.4483171668248292, "state_dim": 99, "state_latent_rows": 84, "token_dim": 131, "vae_debug_latent_abs_mean": 0.1456600978669161, "vae_heldout_test_action_mse": 0.008726497973125745, "vae_latent_multiseed_test_reduction_mean": 0.8925315111873378, "vae_latent_overfit_loss": 3.341915582546504e-19, "vae_latent_test_prediction_loss": 0.007374265934471514}`.
- Paper-state Transformer architecture probe: `ok`; settings `{"attention_heads": 8, "denoising_steps": 20, "embedding_dim": 512, "state_dim": 99, "token_dim": 131, "transformer_layers": 6}`; metrics `{"clean_trajectory_mse": 1.708655595779419, "cuda_peak_memory_mb": 386.9970703125, "parameter_count": 19080323, "total_grad_norm": 5.51327657699585}`.
- Debug-VAE-latent Transformer architecture probe: `ok`; settings `{"attention_heads": 8, "denoising_steps": 20, "embedding_dim": 512, "latent_source": "debug_tiny_vae_mu_nonzero_synthetic_teacher", "state_dim": 99, "token_dim": 131, "transformer_layers": 6}`; metrics `{"clean_trajectory_mse": 1.5802749395370483, "cuda_peak_memory_mb": null, "parameter_count": 19080323, "total_grad_norm": 6.366533279418945}`.
- Transformer parameter-count audit: `ok`; metrics `{"expected_delta_from_input_output_projection": 84050, "fixture_181d_parameter_count": 19164373, "fixture_relative_delta_vs_paper": -0.032102373737373735, "paper_reference_count": 19800000, "paper_state_99d_parameter_count": 19080323, "paper_state_relative_delta_vs_paper": -0.03634732323232323, "parameter_count_delta_between_local_variants": 84050, "token_dim_delta": 82, "variant_count": 2}`.
- Transformer state-dict manifest: `ok`; metrics `{"different_seed_state_dict_sha256": "df4f05217c16a0cb9983be0a9a99eba3f1f9c3261b34b264413d2c6f2e7b9fa3", "overall_state_dict_sha256": "ff12bb75791998a31e227c11eda0f7a2721b3d25d2b2e3ede1c9b2e4903ce756", "parameter_count": 19080323, "state_dict_numel": 19080323, "state_dict_tensor_count": 79}`.
- Transformer EMA smoke: `ok`; metrics `{"ema_vs_model_l2": 6.1767519987654396e-09, "ema_vs_model_max_abs": 2.384185791015625e-07, "final_ema_sha256": "e9c07309a8abacd414af2413834653ed27e85fae3a4fd0eb6eff2dbb81353ca0", "final_loss_after": 1.5447429418563843, "final_loss_before": 1.5448564291000366, "final_model_sha256": "f55e883a05267d6b3e63de3b27923f04c4595fd7d9e6afab4db6a384d1f554fb", "initial_model_sha256": "bcb732a9d96a2c2f1a85d3aa2505a49edbcc362d68314c6ec64d34b92f35dc94", "loss_after_min": 1.5447429418563843, "model_vs_initial_l2": 2.299133328165226e-08, "parameter_count": 19080323, "step_count": 2}`.
- Debug-VAE-latent Transformer EMA smoke: `ok`; metrics `{"ema_vs_model_l2": 6.398524821094043e-09, "ema_vs_model_max_abs": 2.384185791015625e-07, "final_ema_sha256": "7602bb8a37f23eec5b0779151656b0f3a72b2581339d3adc788b484e8ef80db4", "final_loss_after": 1.5761398077011108, "final_loss_before": 1.5763062238693237, "final_model_sha256": "767acd563fcd601386c85ddc32e7bda1827501e48111d090c26972f50bcba885", "initial_model_sha256": "bcb732a9d96a2c2f1a85d3aa2505a49edbcc362d68314c6ec64d34b92f35dc94", "input_motion_count": 3, "input_motion_ids_first_batch": [0], "input_window_count": 84, "latent_abs_max": 1.548389196395874, "latent_abs_mean": 0.1456600978669161, "loss_after_min": 1.466834306716919, "model_vs_initial_l2": 2.3621366196380222e-08, "parameter_count": 19080323, "step_count": 2}`.
- Diffusion checkpoint smoke: `ok`; metrics `{"checkpoint_size_bytes": 305417863, "ema_l2_after_resume": 0.0, "eval_max_abs_error_after_resume": 0.0, "final_resumed_loss_after": 1.7795637845993042, "final_uninterrupted_loss_after": 1.7795637845993042, "model_l2_after_resume": 0.0}`; checkpoint `/mnt/infini-data/test/BeyondMimic/res/level_c/diffusion_checkpoint_smoke/debug_diffusion_transformer_checkpoint_smoke.pt` is debug-only.
- Bounded debug diffusion training run: `ok`; metrics `{"checkpoint_size_bytes": 305413425, "debug_step_count": 3, "debug_token_dim": 131, "final_loss_after": 1.3734568357467651, "initial_loss_before": 1.351046085357666, "is_training_run": false, "loss_figure_size_bytes": 3239, "paper_level": false, "parameter_count": 19080323}`; run dir `/mnt/infini-data/test/BeyondMimic/res/runs/level_c_bounded_debug_diffusion_static_000_20260617_083000`.
- Bounded debug diffusion checkpoint eval: `ok`; rows `[{"initial_model_mse": 1.438082218170166, "motion_names": "walk1_subject1_frames_1_180_state_fixture", "noisy_identity_mse": 0.06486856192350388, "split": "train", "token_count": 588, "trained_checkpoint_mse": 1.4378801584243774, "trained_vs_initial_delta": 0.00020205974578857422, "trained_vs_noisy_delta": -1.3730115965008736, "window_count": 28}, {"initial_model_mse": 1.4264111518859863, "motion_names": "run2_subject1_frames_1_180_state_fixture", "noisy_identity_mse": 0.06678884476423264, "split": "validation", "token_count": 588, "trained_checkpoint_mse": 1.4262079000473022, "trained_vs_initial_delta": 0.00020325183868408203, "trained_vs_noisy_delta": -1.3594190552830696, "window_count": 28}, {"initial_model_mse": 1.4604772329330444, "motion_names": "jumps1_subject1_frames_1_180_state_fixture", "noisy_identity_mse": 0.06540505588054657, "split": "test", "token_count": 588, "trained_checkpoint_mse": 1.4602681398391724, "trained_vs_initial_delta": 0.0002090930938720703, "trained_vs_noisy_delta": -1.3948630839586258, "window_count": 28}]`.
- Bounded debug diffusion action eval: `ok`; metrics `{"test_checkpoint_current_action_mse": 0.489066042088447, "test_checkpoint_full_action_mse": 0.4699513496897675, "validation_checkpoint_current_action_mse": 0.45596258052197547, "validation_checkpoint_full_action_mse": 0.456777148814784}`.
- Resource-adjusted tiny diffusion training run: `ok`; metrics `{"checkpoint_size_bytes": 1730542, "epochs": 180, "parameter_count": 143491, "test_pred_current_action_mse": 0.008886663625808453, "test_pred_token_mse": 0.006055245326838027, "train_pred_token_mse": 0.0007233637408933863, "validation_pred_current_action_mse": 0.009416264484913103, "validation_pred_token_mse": 0.007219147213891228}`; run dir `/mnt/infini-data/test/BeyondMimic/res/runs/level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500`.
- Resource-adjusted full teacher-rollout conditional action VAE training: `ok`; summary `{"action_dim": 29, "cuda_visible_devices": "4,7", "data_parallel_used": true, "epochs": 40, "gpu_metrics_summary": {"exists": true, "per_gpu": {"4": {"mean_power_w": 127.17666666666668, "mean_utilization_gpu_percent": 1.7333333333333334, "memory_total_mb": 97871, "peak_memory_used_mb": 1720.0, "samples": 15}, "7": {"mean_power_w": 123.32199999999999, "mean_utilization_gpu_percent": 1.1333333333333333, "memory_total_mb": 97871, "peak_memory_used_mb": 1670.0, "samples": 15}}, "row_count": 30}, "latent_dim": 32, "obs_dim": 160, "sample_count": 306176, "splits": {"test": 30618, "train": 244940, "validation": 30618}, "test_action_abs_error_mean": 0.04116625525057316, "test_action_mse": 0.002976319403387606, "torch_cuda_device_count": 2, "validation_action_mse": 0.0029512199107557535}`. This run trains on all currently collected local resource-adjusted teacher rollout shards and writes its checkpoint only under ignored `res/runs`. It is stronger than a smoke test, but remains a resource-adjusted local VAE result rather than the official BeyondMimic DAgger/VAE checkpoint or a closed-loop diffusion result.
- Resource-adjusted full teacher-rollout state-latent dataset: `ok`; summary `{"latent_dim": 32, "obs_dim": 160, "sample_count": 306176, "sequence_length": 21, "split_counts": {"test": 28569, "train": 228558, "validation": 28569}, "token_dim": 192, "weighted_posterior_reconstruction_mse": 0.002923722844570875, "window_count": 285696}`. This converts all current resource-adjusted teacher rollout shards into 21-step policy-observation plus VAE posterior-latent windows. It is a useful downstream diffusion dataset, but remains generated-resource local evidence rather than the official DAgger/state-latent dataset.
- Resource-adjusted full state-latent denoiser training: `ok`; summary `{"batch_windows": 2048, "data_parallel_used": true, "epochs": 30, "gpu_metrics_summary": {"exists": true, "per_gpu": {"4": {"mean_power_w": 152.41, "mean_utilization_gpu_percent": 9.043478260869565, "memory_total_mb": 97871, "peak_memory_used_mb": 2216.0, "samples": 23}, "7": {"mean_power_w": 144.72826086956525, "mean_utilization_gpu_percent": 4.391304347826087, "memory_total_mb": 97871, "peak_memory_used_mb": 1806.0, "samples": 23}}, "row_count": 46}, "split_counts": {"test": 28569, "train": 228558, "validation": 28569}, "test_denoising_improvement_ratio": 0.5491175139992032, "test_noisy_token_mse": 0.08264570789677757, "test_pred_token_mse": 0.03726350223379476, "validation_pred_token_mse": 0.037304522203547616, "window_count": 285696}`. This trains on all generated windows and shows held-out denoising improvement, but it is not the official BeyondMimic diffusion checkpoint, TensorRT engine, or Fig. 5/Fig. 6 closed-loop guidance result.
- Resource-adjusted offline state-latent guidance evaluation: `ok`; summary `{"gpu_metrics_summary": {"exists": true, "per_gpu": {"4": {"mean_power_w": 95.82499999999999, "mean_utilization_gpu_percent": 0.0, "memory_total_mb": 97871, "peak_memory_used_mb": 328.0, "samples": 2}, "7": {"mean_power_w": 72.955, "mean_utilization_gpu_percent": 0.0, "memory_total_mb": 97871, "peak_memory_used_mb": 4.0, "samples": 2}}, "row_count": 4}, "row_count": 48, "scales": [0.0, 0.0005, 0.001, 0.002, 0.005, 0.01], "task_summaries": {"composed": {"mean_best_cost_delta": 1.86315446626395e-07, "mean_positive_delta_fraction": 0.91748046875}, "latent_magnitude": {"mean_best_cost_delta": 1.5347613953053951e-06, "mean_positive_delta_fraction": 0.9642333984375}, "latent_smoothness": {"mean_best_cost_delta": 8.558126864954829e-07, "mean_positive_delta_fraction": 1.0}, "velocity_command": {"mean_best_cost_delta": 1.7268666852032766e-07, "mean_positive_delta_fraction": 1.0}}, "tasks": ["velocity_command", "latent_smoothness", "latent_magnitude", "composed"], "tasks_with_all_best_costs_improve": 4, "tasks_with_nonzero_best_gradients": 4, "total_selected_windows": 8192}`. This connects the local denoiser to task-cost guidance on validation/test windows. It is useful evidence for the reading-report reproduction section, but it is not a closed-loop IsaacLab rollout, not official Fig. 5/Fig. 6 evidence, and not a paper-level guidance reproduction.
- Public LAFAN1 paper-architecture VAE/diffusion training: `ok`; metrics `{"checkpoint_size_bytes": 302997114, "data_parallel": true, "diffusion_parameter_count": 19191023, "final_test_decoded_action_mse": 0.032622650265693665, "final_test_pred_tau_mse": 0.007946033962070942, "final_validation_decoded_action_mse": 0.02712077833712101, "final_validation_pred_tau_mse": 0.0076224529184401035, "gpu_device_ids": [0, 1, 2, 3, 4, 5, 6, 7], "public_lafan1_motion_count": 40, "token_count": 46200, "vae_parameter_count": 6047354, "window_count": 2200}`; outputs `{"checkpoint": "/mnt/infini-data/test/BeyondMimic/res/runs/level_c_lafan1_paper_arch_vae_diffusion_static_000_20260617_203000/checkpoint/lafan1_paper_arch_vae_diffusion.pt", "dataset_npz": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_vae_diffusion_training/lafan1_paper_arch_training_dataset.npz", "diffusion_figure": "/mnt/infini-data/test/BeyondMimic/res/runs/level_c_lafan1_paper_arch_vae_diffusion_static_000_20260617_203000/figures/diffusion_tau_mse.png", "diffusion_tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_vae_diffusion_training/lafan1_paper_arch_diffusion_rows.tsv", "json": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_vae_diffusion_training/lafan1_paper_arch_vae_diffusion_training.json", "run_dir": "/mnt/infini-data/test/BeyondMimic/res/runs/level_c_lafan1_paper_arch_vae_diffusion_static_000_20260617_203000", "vae_figure": "/mnt/infini-data/test/BeyondMimic/res/runs/level_c_lafan1_paper_arch_vae_diffusion_static_000_20260617_203000/figures/vae_action_mse.png", "vae_tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_vae_diffusion_training/lafan1_paper_arch_vae_rows.tsv"}`.
- Public LAFAN1 paper-architecture 3-seed statistics: `ok`; statistics `{"checkpoint_size_bytes": {"max": 302997178.0, "mean": 302997156.6666667, "min": 302997114.0, "std": 36.950417228136054}, "cuda_peak_memory_mb": {"max": 868.31298828125, "mean": 868.31298828125, "min": 868.31298828125, "std": 0.0}, "diffusion_parameter_count": {"max": 19191023.0, "mean": 19191023.0, "min": 19191023.0, "std": 0.0}, "elapsed_seconds": {"max": 154.67890293896198, "mean": 150.40162793919444, "min": 148.18651772290468, "std": 3.7050181030707297}, "final_test_decoded_action_mse": {"max": 0.03800620883703232, "mean": 0.03449550146857897, "min": 0.032622650265693665, "std": 0.003042631317967293}, "final_test_pred_tau_mse": {"max": 0.009039249271154404, "mean": 0.008332139501969019, "min": 0.007946033962070942, "std": 0.0006132395245080551}, "final_validation_decoded_action_mse": {"max": 0.03171243146061897, "mean": 0.02867438333729903, "min": 0.02712077833712101, "std": 0.002631254100427145}, "final_validation_pred_tau_mse": {"max": 0.008891399949789047, "mean": 0.008103635782996813, "min": 0.0076224529184401035, "std": 0.0006877868225348945}, "public_lafan1_motion_count": {"max": 40.0, "mean": 40.0, "min": 40.0, "std": 0.0}, "token_count": {"max": 46200.0, "mean": 46200.0, "min": 46200.0, "std": 0.0}, "vae_parameter_count": {"max": 6047354.0, "mean": 6047354.0, "min": 6047354.0, "std": 0.0}, "window_count": {"max": 2200.0, "mean": 2200.0, "min": 2200.0, "std": 0.0}}`; outputs `{"json": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_multiseed_audit/level_c_lafan1_paper_arch_multiseed_audit.json", "npz": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_multiseed_audit/level_c_lafan1_paper_arch_multiseed_metrics.npz", "tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_multiseed_audit/level_c_lafan1_paper_arch_multiseed_rows.tsv"}`.
- Public LAFAN1 paper-architecture symmetry-augmented 3-seed statistics: `ok`; statistics `{"augmented_motion_label_count": {"max": 80.0, "mean": 80.0, "min": 80.0, "std": 0.0}, "checkpoint_size_bytes": {"max": 302997370.0, "mean": 302997370.0, "min": 302997370.0, "std": 0.0}, "cuda_peak_memory_mb": {"max": 1646.95166015625, "mean": 1646.95166015625, "min": 1646.95166015625, "std": 0.0}, "diffusion_parameter_count": {"max": 19191023.0, "mean": 19191023.0, "min": 19191023.0, "std": 0.0}, "elapsed_seconds": {"max": 172.26538935303688, "mean": 169.5210083487133, "min": 164.9745894111693, "std": 3.965406219122857}, "final_test_decoded_action_mse": {"max": 0.030853088945150375, "mean": 0.028371505439281464, "min": 0.026402728632092476, "std": 0.002269064732806989}, "final_test_pred_tau_mse": {"max": 0.008780686184763908, "mean": 0.008661631805201372, "min": 0.00857826042920351, "std": 0.00010582534755953794}, "final_validation_decoded_action_mse": {"max": 0.026281803846359253, "mean": 0.023891310517986614, "min": 0.022088494151830673, "std": 0.0021575413570768523}, "final_validation_pred_tau_mse": {"max": 0.008564675226807594, "mean": 0.008435842270652453, "min": 0.008320493623614311, "std": 0.00012264800764293966}, "public_lafan1_unique_motion_label_count": {"max": 40.0, "mean": 40.0, "min": 40.0, "std": 0.0}, "token_count": {"max": 92400.0, "mean": 92400.0, "min": 92400.0, "std": 0.0}, "vae_parameter_count": {"max": 6047354.0, "mean": 6047354.0, "min": 6047354.0, "std": 0.0}, "window_count": {"max": 4400.0, "mean": 4400.0, "min": 4400.0, "std": 0.0}}`; outputs `{"json": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_multiseed_audit/level_c_lafan1_paper_arch_symmetry_augmented_multiseed_audit.json", "npz": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_multiseed_audit/level_c_lafan1_paper_arch_symmetry_augmented_multiseed_metrics.npz", "tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_multiseed_audit/level_c_lafan1_paper_arch_symmetry_augmented_multiseed_rows.tsv"}`.
- Public LAFAN1 paper-architecture 8-GPU high-memory batch audit: `ok`; metrics `{"forward_backward_seconds": 2.3310411646962166, "loss": 0.010348043404519558, "max_after_reserve_used_mb": 20001, "max_batch_peak_allocated_mb": 5597.81884765625, "min_after_reserve_used_mb": 20000, "min_batch_peak_allocated_mb": 4692.10009765625, "total_reserved_tensor_mb": 112354.0}`; outputs `{"json": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_high_memory_batch_audit/level_c_lafan1_paper_arch_high_memory_batch_audit.json", "npz": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_high_memory_batch_audit/level_c_lafan1_paper_arch_high_memory_batch_fixture.npz", "tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_high_memory_batch_audit/level_c_lafan1_paper_arch_high_memory_batch_rows.tsv"}`.
- Public LAFAN1 paper-architecture symmetry-augmented dataset: `ok`; metrics `{"augmented_action_abs_max": 2.8519175052642822, "augmented_motion_label_count": 80, "augmented_token_count": 92400, "augmented_window_count": 4400, "double_mirror_action_max_abs_error": 0.0, "double_mirror_state_max_abs_error": 5.960464477539063e-08, "mirror_window_count": 2200, "mirrored_projection_recompute_max_abs_error": 0.0, "mirrored_state_abs_max": 6.997476577758789, "public_lafan1_motion_count": 40, "source_projection_recompute_max_abs_error": 0.0, "source_state_abs_max": 6.997476577758789, "source_token_count": 46200, "source_window_count": 2200}`; outputs `{"json": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_dataset/lafan1_paper_arch_symmetry_dataset_audit.json", "npz": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_dataset/lafan1_paper_arch_symmetry_augmented_dataset.npz", "tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_dataset/level_c_lafan1_paper_arch_symmetry_dataset_splits.tsv"}`.
- Public LAFAN1 paper-architecture symmetry-augmented VAE/diffusion training: `ok`; metrics `{"augmented_motion_label_count": 80, "checkpoint_size_bytes": 302997370, "data_parallel": true, "diffusion_parameter_count": 19191023, "elapsed_seconds": 172.26538935303688, "final_test_decoded_action_mse": 0.026402728632092476, "final_test_pred_tau_mse": 0.008780686184763908, "final_validation_decoded_action_mse": 0.022088494151830673, "final_validation_pred_tau_mse": 0.008564675226807594, "gpu_device_ids": [0, 1, 2, 3, 4, 5, 6, 7], "public_lafan1_unique_motion_label_count": 40, "token_count": 92400, "vae_parameter_count": 6047354, "window_count": 4400}`; outputs `{"checkpoint": "/mnt/infini-data/test/BeyondMimic/res/runs/level_c_lafan1_paper_arch_symmetry_augmented_static_000_20260617_215500/checkpoint/lafan1_paper_arch_vae_diffusion.pt", "dataset_npz": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_training/lafan1_paper_arch_training_dataset.npz", "diffusion_figure": "/mnt/infini-data/test/BeyondMimic/res/runs/level_c_lafan1_paper_arch_symmetry_augmented_static_000_20260617_215500/figures/diffusion_tau_mse.png", "diffusion_tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_training/lafan1_paper_arch_diffusion_rows.tsv", "json": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_training/lafan1_paper_arch_vae_diffusion_training.json", "run_dir": "/mnt/infini-data/test/BeyondMimic/res/runs/level_c_lafan1_paper_arch_symmetry_augmented_static_000_20260617_215500", "source_dataset_npz": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_dataset/lafan1_paper_arch_symmetry_augmented_dataset.npz", "vae_figure": "/mnt/infini-data/test/BeyondMimic/res/runs/level_c_lafan1_paper_arch_symmetry_augmented_static_000_20260617_215500/figures/vae_action_mse.png", "vae_tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_training/lafan1_paper_arch_vae_rows.tsv"}`.
- Public LAFAN1 paper-architecture ONNX/latency audit: `ok`; metrics `{"diffusion_denoiser_onnx_reference_cpu_p95_ms": 56.45706132054329, "diffusion_denoiser_torch_cpu_p95_ms": 11.761460453271866, "diffusion_max_abs_onnx_vs_torch": 2.251937985420227e-06, "diffusion_onnx_sha256": "66a04543b4a6042bd2a981ebe8f5848c17e481c0972fd1ec74566c3fc09d1d8a", "diffusion_onnx_size_bytes": 76924041, "diffusion_parameter_count": 19191023, "diffusion_single_window_tau_mse": 0.002763733034953475, "vae_decoder_current_action_mse": 0.08320638537406921, "vae_decoder_onnx_reference_cpu_p95_ms": 0.3209322690963745, "vae_decoder_torch_cpu_p95_ms": 0.3471106290817261, "vae_max_abs_onnx_vs_torch": 1.1920928955078125e-07, "vae_onnx_sha256": "bb0b37621b7231457d74c030bbcf06747af8e4a4673c00783198280711026c6d", "vae_onnx_size_bytes": 11634669, "vae_parameter_count": 6047354}`; outputs `{"component_tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_onnx_latency/level_c_lafan1_paper_arch_onnx_latency_components.tsv", "diffusion_denoiser_onnx": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_onnx_latency/lafan1_paper_arch_diffusion_denoiser.onnx", "json": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_onnx_latency/level_c_lafan1_paper_arch_onnx_latency_audit.json", "latency_tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_onnx_latency/level_c_lafan1_paper_arch_onnx_latency_rows.tsv", "npz": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_onnx_latency/lafan1_paper_arch_onnx_io_fixture.npz", "vae_decoder_onnx": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_onnx_latency/lafan1_paper_arch_vae_decoder.onnx"}`.
- Public LAFAN1 paper-architecture symmetry-augmented ONNX/latency audit: `ok`; metrics `{"diffusion_denoiser_onnx_reference_cpu_p95_ms": 58.858951181173325, "diffusion_denoiser_torch_cpu_p95_ms": 12.31020763516426, "diffusion_max_abs_onnx_vs_torch": 2.771615982055664e-06, "diffusion_onnx_sha256": "fae3011d4090588215019cdf397d189e576ee42d79f878ec2b82c0b4b89211cb", "diffusion_onnx_size_bytes": 76924206, "diffusion_parameter_count": 19191023, "diffusion_single_window_tau_mse": 0.0033518108539283276, "vae_decoder_current_action_mse": 0.052795473486185074, "vae_decoder_onnx_reference_cpu_p95_ms": 0.3968238830566406, "vae_decoder_torch_cpu_p95_ms": 0.35236701369285583, "vae_max_abs_onnx_vs_torch": 2.384185791015625e-07, "vae_onnx_sha256": "a4d6ba8ab119a4d06fc2cc272fea9fda065012a3c909d1abde35b023e447cdc5", "vae_onnx_size_bytes": 11634834, "vae_parameter_count": 6047354}`; outputs `{"component_tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/level_c_lafan1_paper_arch_onnx_latency_components.tsv", "diffusion_denoiser_onnx": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/lafan1_paper_arch_diffusion_denoiser.onnx", "json": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/level_c_lafan1_paper_arch_onnx_latency_audit.json", "latency_tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/level_c_lafan1_paper_arch_onnx_latency_rows.tsv", "npz": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/lafan1_paper_arch_onnx_io_fixture.npz", "vae_decoder_onnx": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/lafan1_paper_arch_vae_decoder.onnx"}`.
- Public LAFAN1 paper-architecture offline metrics audit: `ok`; metrics `{"diffusion_denoiser_onnx_reference_cpu_p95_ms": 56.45706132054329, "diffusion_denoiser_torch_cpu_p95_ms": 11.761460453271866, "test_action_second_difference_mean_norm": 0.2847289443016052, "test_decoded_pred_current_action_mse": 0.03400135412812233, "test_diffusion_pred_tau_mse": 0.007869400084018707, "test_vae_kl_mean": 0.5659266114234924, "vae_decoder_torch_cpu_p95_ms": 0.3471106290817261, "validation_action_second_difference_mean_norm": 0.2830184996128082, "validation_decoded_pred_current_action_mse": 0.029158219695091248, "validation_diffusion_pred_tau_mse": 0.007752980571240187, "validation_vae_kl_mean": 0.4725886881351471}`; outputs `{"json": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_audit.json", "npz": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_offline_metrics/level_c_lafan1_paper_arch_offline_metrics.npz", "tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_rows.tsv"}`.
- Public LAFAN1 paper-architecture symmetry-augmented offline metrics audit: `ok`; metrics `{"diffusion_denoiser_onnx_reference_cpu_p95_ms": 56.45706132054329, "diffusion_denoiser_torch_cpu_p95_ms": 11.761460453271866, "test_action_second_difference_mean_norm": 0.3209989368915558, "test_decoded_pred_current_action_mse": 0.02813071385025978, "test_diffusion_pred_tau_mse": 0.00877812597900629, "test_vae_kl_mean": 0.4931441843509674, "vae_decoder_torch_cpu_p95_ms": 0.3471106290817261, "validation_action_second_difference_mean_norm": 0.31488358974456787, "validation_decoded_pred_current_action_mse": 0.02398289181292057, "validation_diffusion_pred_tau_mse": 0.008585984818637371, "validation_vae_kl_mean": 0.4297948181629181}`; outputs `{"json": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_audit.json", "npz": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_offline_metrics/level_c_lafan1_paper_arch_offline_metrics.npz", "tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_rows.tsv"}`.
- Public LAFAN1 paper-architecture offline guidance eval: `ok`; task summaries `{"composed_objectives": {"all_best_costs_improve": true, "best_rows_primary_improved_count": 5, "mean_base_cost": 1648.51044921875, "mean_best_cost": 1432.5726318359375, "mean_cost_delta": 215.9378173828125, "mean_gradient_norm": 476.65789794921875, "scale_count": 7, "window_count": 5}, "inpainting": {"all_best_costs_improve": true, "best_rows_primary_improved_count": 5, "mean_base_cost": 0.0010598529945127666, "mean_best_cost": 0.0010598326800391078, "mean_cost_delta": 2.0314473658800125e-08, "mean_gradient_norm": 0.0044587456155568365, "scale_count": 7, "window_count": 5}, "joystick": {"all_best_costs_improve": true, "best_rows_primary_improved_count": 5, "mean_base_cost": 1.299945306777954, "mean_best_cost": 1.2973467350006103, "mean_cost_delta": 0.00259857177734375, "mean_gradient_norm": 1.6112431049346925, "scale_count": 7, "window_count": 5}, "obstacle_avoidance": {"all_best_costs_improve": true, "best_rows_primary_improved_count": 5, "mean_base_cost": 1647.0359130859374, "mean_best_cost": 1431.1017333984375, "mean_cost_delta": 215.9341796875, "mean_gradient_norm": 476.6541259765625, "scale_count": 7, "window_count": 5}, "waypoint": {"all_best_costs_improve": true, "best_rows_primary_improved_count": 4, "mean_base_cost": 0.17457784116268157, "mean_best_cost": 0.17408451586961746, "mean_cost_delta": 0.0004933252930641174, "mean_gradient_norm": 0.6938012123107911, "scale_count": 7, "window_count": 5}}`; outputs `{"json": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.json", "npz": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.npz", "tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.tsv"}`.
- Public LAFAN1 paper-architecture symmetry-augmented offline guidance eval: `ok`; task summaries `{"composed_objectives": {"all_best_costs_improve": true, "best_rows_primary_improved_count": 5, "mean_base_cost": 1691.76162109375, "mean_best_cost": 1467.8094482421875, "mean_cost_delta": 223.9521728515625, "mean_gradient_norm": 485.4724853515625, "scale_count": 7, "window_count": 5}, "inpainting": {"all_best_costs_improve": true, "best_rows_primary_improved_count": 5, "mean_base_cost": 0.0008777789073064924, "mean_best_cost": 0.0008777620969340205, "mean_cost_delta": 1.6810372471809388e-08, "mean_gradient_norm": 0.004086224269121886, "scale_count": 7, "window_count": 5}, "joystick": {"all_best_costs_improve": true, "best_rows_primary_improved_count": 5, "mean_base_cost": 1.1632789611816405, "mean_best_cost": 1.1609535932540893, "mean_cost_delta": 0.0023253679275512694, "mean_gradient_norm": 1.5245662927627563, "scale_count": 7, "window_count": 5}, "obstacle_avoidance": {"all_best_costs_improve": true, "best_rows_primary_improved_count": 5, "mean_base_cost": 1690.4029296875, "mean_best_cost": 1466.453662109375, "mean_cost_delta": 223.949267578125, "mean_gradient_norm": 485.469482421875, "scale_count": 7, "window_count": 5}, "waypoint": {"all_best_costs_improve": true, "best_rows_primary_improved_count": 0, "mean_base_cost": 0.19541964530944825, "mean_best_cost": 0.1948183536529541, "mean_cost_delta": 0.0006012916564941406, "mean_gradient_norm": 0.7737938165664673, "scale_count": 7, "window_count": 5}}`; outputs `{"json": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.json", "npz": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.npz", "tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.tsv"}`.
- Public LAFAN1 paper-architecture symmetry-augmented full-split offline guidance eval: `ok`; rows `46200`; settings `{"max_windows_per_split": -1, "split_window_counts": {"test": 660, "validation": 660}, "splits": ["validation", "test"]}`; task summaries `{"composed_objectives": {"all_best_costs_improve": true, "best_rows_primary_improved_count": 1320, "mean_base_cost": 1902.6390316125119, "mean_best_cost": 1641.6832794189454, "mean_cost_delta": 260.9557521935665, "mean_gradient_norm": 523.7033147176106, "scale_count": 7, "window_count": 1320}, "inpainting": {"all_best_costs_improve": true, "best_rows_primary_improved_count": 1320, "mean_base_cost": 0.000930919799421562, "mean_best_cost": 0.0009309019227354535, "mean_cost_delta": 1.7876686108524374e-08, "mean_gradient_norm": 0.0041256765752085344, "scale_count": 7, "window_count": 1320}, "joystick": {"all_best_costs_improve": true, "best_rows_primary_improved_count": 1320, "mean_base_cost": 1.3631208822131158, "mean_best_cost": 1.3603960051681057, "mean_cost_delta": 0.002724877045010075, "mean_gradient_norm": 1.5989806913968287, "scale_count": 7, "window_count": 1320}, "obstacle_avoidance": {"all_best_costs_improve": true, "best_rows_primary_improved_count": 1320, "mean_base_cost": 1900.6584928570371, "mean_best_cost": 1639.7083324085581, "mean_cost_delta": 260.9501604484789, "mean_gradient_norm": 523.6979736559318, "scale_count": 7, "window_count": 1320}, "waypoint": {"all_best_costs_improve": true, "best_rows_primary_improved_count": 765, "mean_base_cost": 0.6174192042682659, "mean_best_cost": 0.615660644429877, "mean_cost_delta": 0.0017585598383889054, "mean_gradient_norm": 1.2189383715165385, "scale_count": 7, "window_count": 1320}}`; outputs `{"json": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/level_c_lafan1_paper_arch_guidance_eval.json", "npz": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/level_c_lafan1_paper_arch_guidance_eval.npz", "tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/level_c_lafan1_paper_arch_guidance_eval.tsv"}`.
- Public LAFAN1 paper-architecture reverse-denoising guidance audit: `ok`; improvement summary `{"tasks_with_all_best_costs_improved": 5, "tasks_with_some_primary_metric_improvement": 5, "total_tasks": 5}`; outputs `{"json": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_reverse_guidance/level_c_lafan1_paper_arch_reverse_guidance_audit.json", "npz": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_reverse_guidance/level_c_lafan1_paper_arch_reverse_guidance.npz", "tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_reverse_guidance/level_c_lafan1_paper_arch_reverse_guidance_rows.tsv"}`.
- Public LAFAN1 paper-architecture symmetry-augmented reverse-denoising guidance audit: `ok`; improvement summary `{"tasks_with_all_best_costs_improved": 4, "tasks_with_some_primary_metric_improvement": 5, "total_tasks": 5}`; outputs `{"json": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance/level_c_lafan1_paper_arch_reverse_guidance_audit.json", "npz": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance/level_c_lafan1_paper_arch_reverse_guidance.npz", "tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance/level_c_lafan1_paper_arch_reverse_guidance_rows.tsv"}`.
- Public LAFAN1 paper-architecture symmetry-augmented full-split reverse-denoising guidance audit: `ok`; metrics `{"min_after_reserve_used_mb": 17198, "min_reverse_peak_allocated_mb": 15399.4931640625, "row_count": 33000, "total_batches": 2, "total_reverse_forwards": 1000}`; settings `{"batch_size": 660, "scales": [0.0, 2e-05, 5e-05, 0.0001, 0.0002], "selected_window_count": 1320, "split_window_counts": {"test": 660, "validation": 660}, "splits": ["validation", "test"]}`; improvement summary `{"tasks_with_all_best_costs_improved": 2, "tasks_with_some_primary_metric_improvement": 5, "total_tasks": 5}`; outputs `{"gpu_tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/level_c_lafan1_paper_arch_reverse_guidance_full_split_gpu_rows.tsv", "json": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.json", "npz": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/level_c_lafan1_paper_arch_reverse_guidance_full_split.npz", "tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/level_c_lafan1_paper_arch_reverse_guidance_full_split_rows.tsv"}`.
- Resource-adjusted tiny diffusion suite: `ok`; `6/6` steps passed, metrics `{"checkpoint_max_token_delta": 0.0, "epochs": 180, "multiseed_test_token_mse_mean": 0.0064945946151577505, "multiseed_validation_token_mse_mean": 0.00685544651330446, "onnx_max_abs_vs_torch": 1.7881393432617188e-07, "onnx_reference_cpu_p95_ms": 0.5516432225704193, "onnx_size_bytes": 576637, "parameter_count": 143491, "test_pred_current_action_mse": 0.008886663625808453, "test_pred_token_mse": 0.006055245326838027, "torch_cpu_p95_ms": 0.16651395708322525, "validation_pred_current_action_mse": 0.009416264484913103, "validation_pred_token_mse": 0.007219147213891228, "video_preview_count": 2}`.
- Resource-adjusted tiny diffusion multi-seed audit: `ok`; statistics `{"test_pred_current_action_mse": {"max": 0.010096621160777279, "mean": 0.009321537230231582, "min": 0.008706551708632307, "std": 0.0005786748631468493}, "test_pred_token_mse": {"max": 0.006637011833851514, "mean": 0.0064945946151577505, "min": 0.006363330598278428, "std": 0.00011200788695217837}, "test_token_reduction_vs_noisy": {"max": 0.9068459933215657, "mean": 0.9032752972188615, "min": 0.9000731927930505, "std": 0.002777241031187853}, "validation_pred_current_action_mse": {"max": 0.008869804319224685, "mean": 0.008245099355464781, "min": 0.007198158165429193, "std": 0.0007448994513621566}, "validation_pred_token_mse": {"max": 0.006927318367506198, "mean": 0.00685544651330446, "min": 0.006775715079675593, "std": 6.214080227912959e-05}, "validation_token_reduction_vs_noisy": {"max": 0.9001934776089964, "mean": 0.8978161670947907, "min": 0.8950632312287335, "std": 0.002111188888480484}}`.
- Resource-adjusted tiny diffusion checkpoint eval: `ok`; metrics `{"max_abs_pred_current_action_mse_delta_vs_source": 0.0, "max_abs_pred_token_mse_delta_vs_source": 0.0, "parameter_count": 143491}`.
- Resource-adjusted tiny diffusion ONNX export/inference: `ok`; metrics `{"max_abs_onnx_vs_torch": 1.7881393432617188e-07, "onnx_mse_to_clean": 0.004402129910886288, "onnx_sha256": "e628e7f44518637be5bdb0de59c5d69aadebce72e876b04499ace2c689eb1f40", "onnx_size_bytes": 576637, "parameter_count": 143491, "sequence_length": 21, "token_dim": 131, "torch_mse_to_clean": 0.004402129910886288}`; ONNX `/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_onnx_export_inference/resource_adjusted_tiny_denoiser_debug.onnx`.
- Resource-adjusted tiny diffusion latency audit: `ok`; metrics `{"max_abs_onnx_vs_torch": 1.7881393432617188e-07, "onnx_reference_cpu_median_ms": 0.5066245794296265, "onnx_reference_cpu_p95_ms": 0.5516432225704193, "onnx_reference_p95_fraction_of_control_40ms": 0.013791080564260483, "onnx_reference_p95_fraction_of_paper_20ms": 0.027582161128520966, "torch_cpu_median_ms": 0.14808028936386108, "torch_cpu_p95_ms": 0.16651395708322525, "torch_p95_fraction_of_control_40ms": 0.004162848927080631, "torch_p95_fraction_of_paper_20ms": 0.008325697854161263}`.
- Resource-adjusted tiny diffusion video preview: `ok`; rows `[{"clean_vs_noisy_mse": 0.0790655230247486, "clean_vs_pred_mse": 0.004402129718253229, "frame_count": 21, "gif_path": "/mnt/infini-data/test/BeyondMimic/res/runs/level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500/videos/tiny_diffusion_validation_debug_preview.gif", "poster_path": "/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_video_preview/tiny_diffusion_validation_debug_preview_poster.png", "pred_reduction_vs_noisy": 0.9443230177978422, "sample_index": 28, "split": "validation"}, {"clean_vs_noisy_mse": 0.053575515496471236, "clean_vs_pred_mse": 0.002229067405986748, "frame_count": 21, "gif_path": "/mnt/infini-data/test/BeyondMimic/res/runs/level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500/videos/tiny_diffusion_test_debug_preview.gif", "poster_path": "/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_video_preview/tiny_diffusion_test_debug_preview_poster.png", "pred_reduction_vs_noisy": 0.9583939158525957, "sample_index": 0, "split": "test"}]`.
- Single-batch overfit gate: `ok`; metrics `{"final_overfit_loss": 4.557876969502246e-16, "initial_noisy_identity_loss": 0.06463638649755082, "loss_reduction_ratio": 0.9999999999999929}`.
- Single-motion overfit gate: `ok`; metrics `{"all_motion_baseline_loss": 0.06594478121987789, "all_motion_loss_reduction_ratio": 1.0, "all_motion_overfit_loss": 7.383799715088834e-19}`.
- Small-dataset overfit gate: `ok`; metrics `{"all_small_dataset_baseline_loss": 0.06878844770150597, "all_small_dataset_loss_reduction_ratio": 1.0, "all_small_dataset_overfit_loss": 1.1634015638380633e-18, "motion_count": 3, "window_count": 84}`.
- Small-dataset split manifest: `ok`; counts `{"motion_split_counts": {"test": 1, "train": 1, "validation": 1}, "motions_total": 3, "samples_total": 84, "split_counts": {"test": 28, "train": 28, "validation": 28}}`.
- Small-dataset multi-seed audit: `ok`; statistics `{"baseline_loss": {"max": 0.06878844770150597, "mean": 0.06836546494206706, "min": 0.06796966533282742, "std": 0.00041006747561319755}, "final_overfit_loss": {"max": 1.195713782610965e-18, "mean": 1.1825253467675343e-18, "min": 1.1634015638380633e-18, "std": 1.6954088866134628e-20}, "loss_reduction_ratio": {"max": 1.0, "mean": 1.0, "min": 1.0, "std": 0.0}}`.
- Small-dataset held-out eval: `ok`; metrics `{"test_loss_reduction_ratio": 0.4437436957932067, "test_prediction_loss": 0.0379766888712374, "train_prediction_loss": 0.00752064193304276, "validation_loss_reduction_ratio": 0.503227761999192, "validation_prediction_loss": 0.03443011209674649}`.
- Small-dataset held-out multi-seed audit: `ok`; statistics `{"test_loss_reduction_ratio": {"max": 0.4748072030115858, "mean": 0.4483171668248292, "min": 0.42640060166969507, "std": 0.024525236825074737}, "test_prediction_loss": {"max": 0.03843609754522177, "mean": 0.03790456355501306, "min": 0.03730090424858001, "std": 0.0005710232054088926}, "validation_loss_reduction_ratio": {"max": 0.5101561948309653, "mean": 0.5027735618904363, "min": 0.49493672884115175, "std": 0.007619892358679882}, "validation_prediction_loss": {"max": 0.03443011209674649, "mean": 0.03320381725788946, "min": 0.03172476500975526, "std": 0.0013702702092037793}}`.
- VAE checkpoint smoke: `ok`; metrics `{"checkpoint_size_bytes": 68387282, "eval_batch_size": 4, "grad_norm_before_optimizer_step": 0.09604582190513611, "max_abs_loaded_eval_action_error": 0.0, "mean_training_loss": 0.017978136924405894, "parameter_update_norm": 1.1801819801330566}`; checkpoint `/mnt/infini-data/test/BeyondMimic/res/level_c/vae_checkpoint_smoke/debug_conditional_vae_checkpoint_smoke.pt` is debug-only.
- VAE debug overfit latent artifact: `ok`; metrics `{"final_kl": 0.2548500895500183, "final_reconstruction_mse": 0.000199218382476829, "final_total_loss": 0.00023517623776569963, "initial_kl": 0.004473662003874779, "initial_reconstruction_mse": 0.08905366063117981, "latent_abs_mean_mean": 0.1456600978669161, "latent_abs_mean_min": 0.12255401622747913, "latent_mu_abs_mean": 0.14566008746623993, "latent_mu_std": 0.19919876754283905, "latent_std_mean": 0.19387336550589412, "latent_std_min": 0.1636773095897998, "reconstruction_loss_reduction_ratio": 0.9977629399952249, "row_count": 84, "split_counts": {"test": 28, "train": 28, "validation": 28}, "token_count": 1764}`.
- VAE motion-split held-out eval: `ok`; metrics `{"final_validation_kl_mean_sum_over_latent": 6.383434062754112, "test_action_mse_reduction_ratio": 0.9204363722178501, "test_final_action_mse": 0.008726497973125745, "token_counts": {"test": 588, "train": 588, "validation": 588}, "train_final_action_mse": 0.0006174284580543962, "validation_action_mse_reduction_ratio": 0.955827922139106, "validation_final_action_mse": 0.004535351618306475}`.
- VAE receding-horizon rollout smoke: `ok`; metrics `{"current_action_mse_max": 0.0008571597683028913, "current_action_mse_mean": 0.0001923113866260608, "full_window_action_mse_mean": 0.00019921838212813082, "mean_action_delta_mean": 0.061739505488998744, "next_latent_action_delta_mean": 0.06758337096089406, "row_count": 84}`.
- Diffusion-to-VAE action smoke: `ok`; metrics `{"test_current_mse_reduction_vs_noisy": 0.9435052428100379, "test_predicted_current_action_mse": 0.0039371201747736865, "test_predicted_full_action_mse": 0.003823876019275019, "validation_current_mse_reduction_vs_noisy": 0.8536799905511543, "validation_predicted_current_action_mse": 0.00911132119759807, "validation_predicted_full_action_mse": 0.011267543688377044}`.
- Diffusion-to-VAE action multi-seed audit: `ok`; statistics `{"test_current_mse_reduction_vs_noisy": {"max": 0.9435052428100379, "mean": 0.9298716385610665, "min": 0.9155199397934589, "std": 0.014006464258895155}, "test_predicted_current_action_mse": {"max": 0.0043505882574483255, "mean": 0.003998468510847306, "min": 0.0037076971003199063, "std": 0.00032580665084176963}, "validation_current_mse_reduction_vs_noisy": {"max": 0.8605450706437925, "mean": 0.8466026289638074, "min": 0.8255828256964751, "std": 0.018524481860975106}, "validation_predicted_current_action_mse": {"max": 0.01020560752960457, "mean": 0.009255110754950491, "min": 0.008448403537648832, "std": 0.0008873826936215285}}`.
- Diffusion-to-VAE action smoothness audit: `ok`; metrics `{"test_predicted_action_acceleration_mean_norm_at_25hz": 401.59270034038593, "test_predicted_action_rate_mean_norm_at_25hz": 9.46269689007263, "test_predicted_smoothness_penalty": 0.01579856790234264, "test_predicted_smoothness_reduction_vs_noisy": 0.9548695448720198, "validation_predicted_action_acceleration_mean_norm_at_25hz": 620.1071585816564, "validation_predicted_action_rate_mean_norm_at_25hz": 14.520518537468638, "validation_predicted_smoothness_penalty": 0.038083277980476095, "validation_predicted_smoothness_reduction_vs_noisy": 0.8984023373821292}`.
- Direct-vs-latent action ablation audit: `ok`; metrics `{"test_direct_current_action_mse": 0.006074739836526304, "test_latent_current_action_mse": 0.003937120162097986, "test_latent_vs_direct_current_mse_ratio": 0.6481133790166287, "validation_direct_current_action_mse": 0.009613393533001568, "validation_latent_current_action_mse": 0.009111321118349816, "validation_latent_vs_direct_current_mse_ratio": 0.9477736542326911}`.
- VAE contract audit: `ok`; metrics `{"effective_batch_size": 30, "failed_row_count": 0, "latent_probe_seed_count": 3, "row_count": 35, "teacher_parameter_count": 251933, "vae_parameter_count": 5697117, "vae_table_value_rows": 8}`.
- DAgger-to-VAE debug pipeline audit: `ok`; metrics `{"action_dim": 29, "dagger_heldout_reduction_ratio": 0.9999999999807272, "dagger_teacher_queries": 288, "dagger_total_samples": 288, "failed_check_count": 0, "latent_dim": 32, "ok_stage_count": 9, "stage_count": 9, "state_dim": 99, "state_latent_rows": 84, "token_dim": 131, "vae_checkpoint_size_bytes": 68387282, "vae_contract_rows": 35, "vae_debug_latent_abs_mean": 0.1456600978669161, "vae_effective_batch_size": 30, "vae_motion_split_test_action_mse": 0.008726497973125745, "vae_parameter_count": 5697117, "vae_receding_current_action_mse_max": 0.0008571597683028913, "vae_receding_current_action_mse_mean": 0.0001923113866260608}`.
- VAE latent probe: `ok`; statistics `{"curvature_mean": {"max": 1.2926116141898092e-05, "mean": 1.0459635935452146e-05, "min": 6.4400774135719985e-06, "std": 3.5108708017727515e-06}, "kl_loss": {"max": 0.18512773513793945, "mean": 0.16656828920046488, "min": 0.15711632370948792, "std": 0.01607387453652661}, "mean_latent_std": {"max": 1.0026994943618774, "mean": 1.0006367762883503, "min": 0.9986274242401123, "std": 0.0020365595298528437}, "mean_neighbor_action_delta": {"max": 0.003107238095253706, "mean": 0.0027207291374603906, "min": 0.002491074614226818, "std": 0.00033669993179177426}}`.
- Symmetry mapping audit: `ok`; metrics `{"center_joint_count": 3, "controller_missing_in_urdf_count": 0, "covered_joint_count": 29, "double_mirrored_fixture_shape": [299, 29], "extra_urdf_actuated_like_count": 0, "joint_count": 29, "mirrored_fixture_shape": [299, 29], "pair_count": 13, "pitch_like_sign_positive_count": 7, "roll_yaw_like_sign_negative_count": 9, "urdf_joint_count": 39}`.
- Guidance task scale sweep: `ok`; `40` rows; task summaries `{"composed_objectives": {"all_finite": true, "best_cost_after": 990.277256800157, "best_cost_delta": 1.2331578715416072, "best_improves_over_zero": true, "best_scale": 1e-05, "gradient_norm": 351.2508472856662, "initial_cost": 991.5104146716986}, "inpainting": {"all_finite": true, "best_cost_after": 0.0008714206132178352, "best_cost_delta": 1.6731278190164345e-10, "best_improves_over_zero": true, "best_scale": 1e-05, "gradient_norm": 0.004090388610656431, "initial_cost": 0.0008714207805306171}, "joystick": {"all_finite": true, "best_cost_after": 1.0462815105867935, "best_cost_delta": 2.0925944100191884e-05, "best_improves_over_zero": true, "best_scale": 1e-05, "gradient_norm": 1.4465838631278132, "initial_cost": 1.0463024365308937}, "obstacle_avoidance": {"all_finite": true, "best_cost_after": 989.1986149317418, "best_cost_delta": 1.2331344537504947, "best_improves_over_zero": true, "best_scale": 1e-05, "gradient_norm": 351.24751373773285, "initial_cost": 990.4317493854923}, "waypoint": {"all_finite": true, "best_cost_after": 0.0323624391585919, "best_cost_delta": 4.105169410864584e-07, "best_improves_over_zero": true, "best_scale": 1e-05, "gradient_norm": 0.20261307809142703, "initial_cost": 0.03236284967553299}}`.
- Guidance debug visualization: `ok`; primary metrics `{"composed": {"after": -0.06329020028220764, "before": -0.1598505830727198, "delta": 0.09656038279051216, "primary_metric": "min_obstacle_clearance"}, "inpainting": {"after": 0.013521512305151584, "before": 0.04599400237472454, "delta": 0.03247249006957296, "primary_metric": "keyframe_error"}, "joystick": {"after": 0.05254195714655683, "before": 0.12435979371656938, "delta": 0.07181783657001256, "primary_metric": "velocity_command_mse"}, "obstacle_avoidance": {"after": 0.03950747084636402, "before": -0.1598505830727198, "delta": 0.19935805391908382, "primary_metric": "min_obstacle_clearance"}, "waypoint": {"after": 0.06804232128257154, "before": 0.09053375319717637, "delta": 0.02249143191460483, "primary_metric": "terminal_goal_distance"}}`; outputs `{"gif": "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_debug_visualization/level_c_guidance_debug_visualization.gif", "json": "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_debug_visualization/level_c_guidance_debug_visualization.json", "npz": "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_debug_visualization/level_c_guidance_debug_visualization.npz", "pdf": "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_debug_visualization/level_c_guidance_debug_visualization.pdf", "png": "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_debug_visualization/level_c_guidance_debug_visualization.png", "svg": "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_debug_visualization/level_c_guidance_debug_visualization.svg", "tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_debug_visualization/level_c_guidance_debug_visualization.tsv"}`.
- Guidance task metric audit: `ok`; metrics `{"improved_task_count": 5, "mean_offline_full_split_best_cost_delta": 104.38207921936109, "mean_primary_delta": 0.08454003905275727, "mean_reverse_full_split_best_cost_delta": 31.105264450339263, "min_best_cost_delta": 1.6731278190164345e-10, "min_gradient_norm": 0.004090388610656431, "offline_full_split_row_count": 46200, "reverse_full_split_row_count": 33000, "row_count": 5, "scale_sweep_task_count": 5}`; primary metrics `{"composed": {"after": -0.06329020028220764, "before": -0.1598505830727198, "best_cost_delta": 1.2331578715416072, "best_scale": 1e-05, "delta": 0.09656038279051216, "direction": "higher_is_better", "improved": true, "primary_metric": "min_obstacle_clearance", "scale_sweep_task": "composed_objectives"}, "inpainting": {"after": 0.013521512305151584, "before": 0.04599400237472454, "best_cost_delta": 1.6731278190164345e-10, "best_scale": 1e-05, "delta": 0.03247249006957296, "direction": "lower_is_better", "improved": true, "primary_metric": "keyframe_error", "scale_sweep_task": "inpainting"}, "joystick": {"after": 0.05254195714655683, "before": 0.12435979371656938, "best_cost_delta": 2.0925944100191884e-05, "best_scale": 1e-05, "delta": 0.07181783657001256, "direction": "lower_is_better", "improved": true, "primary_metric": "velocity_command_mse", "scale_sweep_task": "joystick"}, "obstacle_avoidance": {"after": 0.03950747084636402, "before": -0.1598505830727198, "best_cost_delta": 1.2331344537504947, "best_scale": 1e-05, "delta": 0.19935805391908382, "direction": "higher_is_better", "improved": true, "primary_metric": "min_obstacle_clearance", "scale_sweep_task": "obstacle_avoidance"}, "waypoint": {"after": 0.06804232128257154, "before": 0.09053375319717637, "best_cost_delta": 4.105169410864584e-07, "best_scale": 1e-05, "delta": 0.02249143191460483, "direction": "lower_is_better", "improved": true, "primary_metric": "terminal_goal_distance", "scale_sweep_task": "waypoint"}}`; full-split linkage included.
- Guidance full-split result table: `ok`; metrics `{"mode_count": 2, "offline_source_rows": 46200, "reverse_min_after_reserve_used_mb": 17198.0, "reverse_source_rows": 33000, "row_count": 10, "task_count": 5}`; mode summary `{"offline": {"mean_best_cost_delta_by_task": {"composed_objectives": 260.9557521935665, "inpainting": 1.7876686108524374e-08, "joystick": 0.002724877045010075, "obstacle_avoidance": 260.9501604484789, "waypoint": 0.0017585598383889054}, "mean_positive_best_cost_delta_fraction": 1.0, "task_count": 5, "tasks_with_all_best_costs_improve": 5, "total_rows": 46200}, "reverse": {"mean_best_cost_delta_by_task": {"composed_objectives": 77.77570736046994, "inpainting": 0.00020350172636262077, "joystick": -0.0008358747867698019, "obstacle_avoidance": 77.59966291947798, "waypoint": 0.15158434480880245}, "mean_positive_best_cost_delta_fraction": 0.885909090909091, "task_count": 5, "tasks_with_all_best_costs_improve": 2, "total_rows": 33000}}`; outputs `{"csv": "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.csv", "figures": ["/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_cost_delta.pdf", "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_cost_delta.svg", "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_cost_delta.png"], "json": "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.json", "tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.tsv"}`.
- Guidance checkpoint visualization: `ok`; metrics `{"mode_count": 4, "offline_source_rows": 46200, "representative_window_index": 1540, "reverse_source_rows": 33000, "row_count": 20, "task_count": 5, "visual_file_count": 16}`; outputs `{"by_task": {"composed_objectives": ["/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_composed_objectives.pdf", "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_composed_objectives.svg", "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_composed_objectives.png"], "inpainting": ["/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_inpainting.pdf", "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_inpainting.svg", "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_inpainting.png"], "joystick": ["/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_joystick.pdf", "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_joystick.svg", "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_joystick.png"], "obstacle_avoidance": ["/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_obstacle_avoidance.pdf", "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_obstacle_avoidance.svg", "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_obstacle_avoidance.png"], "waypoint": ["/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_waypoint.pdf", "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_waypoint.svg", "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_waypoint.png"]}, "gif": "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_composed_preview.gif", "json": "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/level_c_guidance_checkpoint_visualization.json", "tsv": "/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/level_c_guidance_checkpoint_visualization.tsv"}`.
- Guidance visual deliverables audit: `ok`; metrics `{"blocked_fig_panel_count": 6, "blocked_video_requirement_count": 6, "failed_check_count": 0, "fig5_fig6_panel_count": 6, "guidance_task_count": 5, "improved_task_count": 5, "mean_primary_delta": 0.08454003905275727, "scale_sweep_row_count": 40, "visual_file_count": 6, "visual_total_size_bytes": 239631}`.
- Guidance cost coverage audit: `ok`; metrics `{"failed_row_count": 0, "formula_missing_row_count": 1, "full_split_source_row_count": 79200, "full_split_task_linked_row_count": 8, "guided_reverse_final_max_step": 0, "paper_explicit_row_count": 5, "row_count": 8, "scale_sweep_rows": 7, "selected_best_scale": 0.02}`.
- Core math unit tests: `ok`; metrics `{"covered_goal_item_count": 38, "failed_row_count": 0, "row_count": 23}`.
- Timestep/mask source coverage audit: `ok`; metrics `{"failed_row_count": 0, "guided_final_max_step": 0, "latent_dim": 32, "paper_explicit_or_high_level_rows": 4, "paper_state_debug_artifact_row_count": 1, "paper_state_mask_reverse_final_mse": 5.232890366443532e-10, "paper_state_mask_state_dim": 99, "paper_state_mask_tau_dim": 131, "policy_unspecified_row_count": 1, "reverse_final_max_step": 0, "row_count": 8, "sequence_length": 21, "state_dim": 181, "tau_dim": 213}`.
- Paper-state mask/reverse probe: `ok`; settings `{"denoising_steps": 20, "history": 4, "horizon": 16, "latent_dim": 32, "motion_key": "walk1_subject1_frames_1_180_state_fixture_paper_state_windows", "paper_state_dim": 99, "reverse_schedule": "future_keyframe_inpainting", "seed": 20260902, "sequence_length": 21, "tau_dim": 131, "window_index": 0}`; metrics `{"reverse_final_max_step": 0, "reverse_final_mse": 5.232890366443532e-10, "reverse_initial_mse": 0.11580444302027998, "reverse_observed_clamp_max_abs_error": 0.0}`.
- Smoothness/latency audit: `ok`; metrics `{"guidance_cost_reduction": 3.1454889029143187, "guided_final_latent_second_difference_mean_norm": 13.538484354740202, "guided_final_state_second_difference_mean_norm": 0.027660824669436108, "paper_denoising_fraction_of_control_period": 0.5, "schema_action_delta_current_vs_next_latent": 0.36904564596667855}`.
- Deployment protocol audit: `ok`; metrics `{"control_period_ms": 40.0, "denoising_fraction_of_control_period": 0.5, "deployment_boundary_row_count": 5, "failed_row_count": 0, "implemented_debug_row_count": 3, "paper_denoising_steps": 20, "paper_diffusion_latency_ms": 20.0, "paper_explicit_row_count": 9, "row_count": 9}`.

## Paper Coverage
- Download source integrity audit: `ok`; category counts `{"dependency_source": 1509, "download_doc": 1, "download_log": 2, "download_manifest": 5, "official_archive": 2, "official_code_or_dataset": 298, "paper_source": 2, "reference_code": 4572}`.
- LaTeX/source coverage audit: `ok`; counts `{"method_claims": 9, "missing_evidence_rows": 0, "missing_expected_labels": 0, "parsed_figures": 10, "parsed_tables": 8, "total_rows": 27, "unmapped_rows": 0}`.
- Coverage buckets: `{"blocked_or_unreproduced": 5, "debug_only": 10, "indexed": 3, "partial": 4, "strong": 5}`.
- LaTeX inventory audit: `ok`; counts `{"equation_count": 8, "expected_setting_count": 14, "experiment_setting_count": 14, "figure_count": 10, "missing_expected_setting_count": 0, "section_count": 51, "table_count": 8, "tex_file_count": 5}`; equation topics `{"joystick_cost": 1, "ou_perturbation": 1, "sdf_barrier_function": 1, "sdf_obstacle_cost": 1, "state_representation": 3, "waypoint_cost": 1}`.
- Paper formula/code trace audit: `ok`; `11` rows, missing evidence `0`, source counts `{"api_test_row_count": 8, "core_math_test_row_count": 23, "core_test_required_count": 20, "latex_equation_count": 8, "latex_experiment_setting_count": 14, "paper_table_value_mismatch_rows": 0, "paper_table_value_rows": 58, "reimpl_symbol_row_count": 29}`.
- PDF/source consistency audit: `ok`; metrics `{"pdf_anchor_count": 20, "pdf_anchor_present_count": 20, "pdf_page_count": 59, "pdf_text_page_count": 59, "source_tar_expected_member_count": 19, "source_tar_extracted_member_count": 19, "source_tar_present_member_count": 19, "unexpected_tar_file_member_count": 0}`.
- Table value audit: `ok`; counts `{"mismatch_rows": 0, "statuses": {"debug_match": 22, "match": 14, "source_value_present": 22}, "tables": {"tab:diffusion_hyperparameters": 14, "tab:domain_rand": 13, "tab:ppo_hyperparameters": 14, "tab:rewardterms": 9, "tab:vae_hyperparameters": 8}, "total_rows": 58}`.
- Skill-success table data audit: `ok`; metrics `{"dash_real_rows": 15, "extra_local_g1_csv_count": 13, "full_real_rows": 8, "lafan_rows": 29, "local_g1_csv_count": 40, "missing_lafan_csv_count": 1, "non_36_column_row_count": 0, "non_finite_csv_row_count": 0, "real_segment_count": 24, "real_segment_rows": 13, "segment_out_of_range_row_count": 2, "short_sequence_rows": 7, "total_rows_parsed": 36}`; missing LAFAN CSVs `["run1_subject4"]`.
- Paper-vs-reproduction comparison: `ok`; `136` rows, type counts `{"approximately_comparable": 19, "exactly_comparable": 58, "not_publicly_reproducible": 10, "qualitative_only": 46, "requires_real_robot": 3}`, missing goal checkpoint rows `0`.
- Results claims audit: `ok`; metrics `{"blocked_or_unreproduced_rows": 4, "debug_only_rows": 5, "failed_row_count": 0, "formula_api_linked_paper_metric_claim_rows": 2, "paper_metric_claim_rows_still_unreproduced": 2, "released_fig3b_max_angular_velocity_norm": 16.764945820586494, "released_fig3b_max_linear_acceleration_norm": 36.18653061127687, "released_fig3b_mean_angular_velocity_norm": 4.854256896352434, "released_fig3b_valid_angular_velocity_samples": 3150.0, "released_or_partial_reproduced_rows": 2, "row_count": 14}`; status counts `{"blocked_closed_loop_required": 1, "blocking_boundary_recorded": 1, "debug_latent_probe_only": 1, "debug_mechanics_only": 1, "formula_debug_only": 1, "formula_debug_only_requires_mocap": 1, "mask_reverse_debug_only": 1, "not_publicly_reproducible_currently": 2, "paper_only_unreproduced": 1, "partial_released_data_reproduced": 2, "requires_real_robot": 1, "source_table_data_audit_only": 1}`.
- Goal traceability audit: `ok`; `25` trace rows over `80` headings, status counts `{"blocked": 1, "covered": 7, "out_of_scope": 1, "partial": 16}`, missing evidence rows `0`.
- goal.md directive index: `ok`; `258` directive rows over `1951` lines and `80` headings, tag counts `{"boundary": 29, "deliverable": 72, "execution": 131, "mandatory": 41, "prohibition": 42}`.
- Goal requirement matrix: `ok`; `28` requirement rows over `1951` goal.md lines, status counts `{"blocked": 1, "complete": 11, "out_of_scope": 1, "partial": 15}`, missing evidence rows `0`.

## Blocked Gates
- Gate status counts: `{"blocked": 4, "clear": 1, "clear_with_historical_failure": 1, "clear_with_runtime_warning": 1, "out_of_scope": 1}`.
- `isaaclab_kit_inotify`: `clear_with_historical_failure`; blocks .
- `isaaclab_kit_vulkan_cuda_runtime`: `clear_with_runtime_warning`; blocks IsaacLab AppLauncher success sentinel, official whole_body_tracking replay_npz.py live replay, tracking task smoke/evaluation inside Kit, PPO motion-tracking training/evaluation, closed-loop VAE/diffusion rollout evaluation.
- `official_g1_usd_conversion_replay`: `blocked`; blocks official csv_to_npz.py motion preprocessing success, official replay_npz.py reference replay, official G1 USD/URDF converter output, paper-level tracking replay/evaluation, formal PPO tracking training on official assets.
- `ros2_jazzy_noble_controller`: `blocked`; blocks MuJoCo sim-to-sim launch from motion_tracking_controller, real.launch.py deployment path, ROS bag recording/evaluation through official deployment package.
- `unitree_g1_hardware`: `out_of_scope`; blocks real robot deployment and hardware robustness claims.
- `official_level_c_artifacts`: `blocked`; blocks paper-level VAE training reproduction, paper-level diffusion training reproduction, trained checkpoint evaluation, TensorRT/deployment reproduction for Level C.
- `fig5_fig6_paper_results`: `blocked`; blocks Figure 5 paper reproduction, Figure 6 paper reproduction, joystick/inpainting/SDF/latent result claims beyond debug mechanics.
- `long_training_safety_gate`: `clear`; blocks unexpected long PPO/VAE/diffusion training before smoke gates.

## Goal.md Final Report Requirements
- Official code used: Level B tracking/configuration audits use the downloaded official `whole_body_tracking` and `motion_tracking_controller` trees where accessible; live IsaacLab/Kit and ROS deployment execution remain blocked.
- Paper-faithful reimplementation: Level C VAE, diffusion, trajectory transforms, masks, and guidance mechanics are reimplemented from paper formulas and local source audits as debug/package evidence, not as unpublished official checkpoints.
- Released-data reproduction: Level A uses public released data for directly redrawable panels and released-figure summaries.
- Retrained results: no paper-scale PPO, VAE, diffusion, TensorRT, or real-robot result has been retrained to completion; only debug overfit, held-out, and multi-seed smoke probes were run.
- Qualitative-only comparison: qualitative/debug-only rows are separated in `res/comparison/paper_vs_reproduction.csv` and the Results claims audit.
- Not publicly reproducible: missing official Level C code/checkpoints, Fig. 5/Fig. 6 rollout data, TensorRT engine, and Unitree G1 hardware evidence are recorded as blocked or out of scope.
- Result differences: paper-vs-reproduction comparison rows record exact, approximate, qualitative-only, not-publicly-reproducible, and real-robot-required comparison types.
- Difference sources: blocked gates, adaptive-sampling discrepancy, missing checkpoints, missing deployment stack, and hardware absence are recorded as likely sources.
- Current reproduction credibility: strong for inventory, paper/source value audits, released-data panels, and unit-tested formula mechanics; partial or debug-only for tracking and Level C; blocked for paper-level deployment and Fig. 5/Fig. 6.
- Completed and incomplete scope: completion matrix, goal traceability audit, and blocked-gate audit separate complete, partial, blocked, and out-of-scope items.
- Hardware cost and training time: current evidence records non-training GPU resource snapshots, diagnostic/debug runtime artifacts, and failed-run retention; full training hardware cost and wall-clock time are missing because no long paper-scale training reached SUCCESS.
- One-command rerun path: the Verification Commands section and `reproduction/RUNBOOK.md` provide the current rerun sequence for all audited artifacts.

## Key Evidence
- [reproduction_master_audit.json](/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json)
- [blocked_gate_audit.json](/mnt/infini-data/test/BeyondMimic/res/blocked_gates/blocked_gate_audit.json)
- [tracking_smoke_rerun_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/tracking_smoke_rerun_audit/tracking_smoke_rerun_audit.json)
- [tracking_official_train_entry_retry_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/official_train_entry_retry_audit/tracking_official_train_entry_retry_audit.json)
- [official_train_entry_retry.log](/mnt/infini-data/test/BeyondMimic/logs/tracking_official_train_entry_retry/official_train_entry_retry.log)
- [official_train_entry_retry_tail.log](/mnt/infini-data/test/BeyondMimic/logs/tracking_official_train_entry_retry/official_train_entry_retry_tail.log)
- [kit_inotify_budget_audit.json](/mnt/infini-data/test/BeyondMimic/res/setup/kit_inotify_budget_audit/kit_inotify_budget_audit.json)
- [inotify_live_usage_audit.json](/mnt/infini-data/test/BeyondMimic/res/setup/inotify_live_usage_audit/inotify_live_usage_audit.json)
- [vscode_watcher_exclude_audit.json](/mnt/infini-data/test/BeyondMimic/res/setup/vscode_watcher_exclude_audit/vscode_watcher_exclude_audit.json)
- [settings.json](/mnt/infini-data/test/BeyondMimic/.vscode/settings.json)
- [kit_watcher_config_surface_audit.json](/mnt/infini-data/test/BeyondMimic/res/setup/kit_watcher_config_surface_audit/kit_watcher_config_surface_audit.json)
- [tracking_import_gate_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/tracking_import_gate_audit/tracking_import_gate_audit.json)
- [tracking_extension_namespace_probe.json](/mnt/infini-data/test/BeyondMimic/res/tracking/tracking_extension_namespace_probe/tracking_extension_namespace_probe.json)
- [tracking_official_source_contract_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/official_source_contract_audit/tracking_official_source_contract_audit.json)
- [tracking_g1_action_scale_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json)
- [tracking_reward_formula_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/reward_formula_audit/tracking_reward_formula_audit.json)
- [tracking_observation_action_schema_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/observation_action_schema_audit/tracking_observation_action_schema_audit.json)
- [tracking_randomization_termination_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/randomization_termination_audit/tracking_randomization_termination_audit.json)
- [level_b_tracking_nonkit_suite.json](/mnt/infini-data/test/BeyondMimic/res/tracking/nonkit_suite/level_b_tracking_nonkit_suite.json)
- [adaptive_sampling_discrepancy_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/adaptive_sampling_discrepancy_audit/adaptive_sampling_discrepancy_audit.json)
- [motion_preprocessing_contract_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/motion_preprocessing_contract_audit/motion_preprocessing_contract_audit.json)
- [tracking_local_smoke_preflight.json](/mnt/infini-data/test/BeyondMimic/res/tracking/local_smoke_preflight/tracking_local_smoke_preflight.json)
- [tracking_motion_npz_fixture.json](/mnt/infini-data/test/BeyondMimic/res/tracking/motion_npz_fixture/tracking_motion_npz_fixture.json)
- [tracking_official_replay_preflight.json](/mnt/infini-data/test/BeyondMimic/res/tracking/official_replay_preflight/tracking_official_replay_preflight.json)
- [tracking_official_replay_conversion_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/official_replay_conversion/tracking_official_replay_conversion_audit.json)
- [tracking_official_replay_npz_entry_diagnostic_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/official_replay_npz_entry_diagnostic/tracking_official_replay_npz_entry_diagnostic_audit.json)
- [tracking_official_replay_npz_entry_diagnostic_probe.py](/mnt/infini-data/test/BeyondMimic/res/tracking/official_replay_npz_entry_diagnostic/tracking_official_replay_npz_entry_diagnostic_probe.py)
- [tracking_official_replay_npz_entry_diagnostic.log](/mnt/infini-data/test/BeyondMimic/res/failed_runs/tracking_official_replay_npz_entry_diagnostic/tracking_official_replay_npz_entry_diagnostic.log)
- [tracking_official_replay_npz_loop_with_enriched_usd_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/official_replay_npz_loop_with_enriched_usd/tracking_official_replay_npz_loop_with_enriched_usd_audit.json)
- [tracking_official_replay_npz_loop_with_enriched_usd_probe.py](/mnt/infini-data/test/BeyondMimic/res/tracking/official_replay_npz_loop_with_enriched_usd/tracking_official_replay_npz_loop_with_enriched_usd_probe.py)
- [tracking_official_csv_to_npz_loop_with_enriched_usd_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/official_csv_to_npz_loop_with_enriched_usd/tracking_official_csv_to_npz_loop_with_enriched_usd_audit.json)
- [tracking_official_csv_to_npz_loop_with_enriched_usd_metrics.json](/mnt/infini-data/test/BeyondMimic/res/tracking/official_csv_to_npz_loop_with_enriched_usd/tracking_official_csv_to_npz_loop_with_enriched_usd_metrics.json)
- [tracking_official_csv_to_npz_loop_with_enriched_usd_probe.py](/mnt/infini-data/test/BeyondMimic/res/tracking/official_csv_to_npz_loop_with_enriched_usd/tracking_official_csv_to_npz_loop_with_enriched_usd_probe.py)
- [tracking_g1_urdf_import_config_variant_probe.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_urdf_import_config_variant_probe/tracking_g1_urdf_import_config_variant_probe.json)
- [tracking_g1_enriched_usd_replay_preflight_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_enriched_usd_replay_preflight/tracking_g1_enriched_usd_replay_preflight_audit.json)
- [tracking_g1_enriched_usd_bounded_replay_metrics_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_enriched_usd_bounded_replay_metrics/tracking_g1_enriched_usd_bounded_replay_metrics_audit.json)
- [walk1_subject1_64step_resource_adjusted_replay_metrics.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_enriched_usd_bounded_replay_metrics/walk1_subject1_64step_resource_adjusted_replay_metrics.json)
- [tracking_g1_resource_adjusted_task_smoke_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_task_smoke/tracking_g1_resource_adjusted_task_smoke_audit.json)
- [tracking_g1_resource_adjusted_task_smoke_metrics.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_task_smoke/tracking_g1_resource_adjusted_task_smoke_metrics.json)
- [tracking_g1_resource_adjusted_multi_fixture_eval_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_multi_fixture_eval/tracking_g1_resource_adjusted_multi_fixture_eval_audit.json)
- [tracking_g1_resource_adjusted_multi_fixture_eval_metrics.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_multi_fixture_eval/tracking_g1_resource_adjusted_multi_fixture_eval_metrics.json)
- [walk1_subject1_frames_1_180_debug_motion_task_eval_metrics.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_multi_fixture_eval/walk1_subject1_frames_1_180_debug_motion_task_eval_metrics.json)
- [run2_subject1_frames_1_180_debug_motion_task_eval_metrics.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_multi_fixture_eval/run2_subject1_frames_1_180_debug_motion_task_eval_metrics.json)
- [jumps1_subject1_frames_1_180_debug_motion_task_eval_metrics.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_multi_fixture_eval/jumps1_subject1_frames_1_180_debug_motion_task_eval_metrics.json)
- [tracking_g1_resource_adjusted_csv_conversion_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_csv_conversion/tracking_g1_resource_adjusted_csv_conversion_audit.json)
- [tracking_g1_resource_adjusted_csv_conversion_metrics.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_csv_conversion/tracking_g1_resource_adjusted_csv_conversion_metrics.json)
- [walk1_subject1_frames_1_180_resource_adjusted_motion_contract.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_csv_conversion/walk1_subject1_frames_1_180_resource_adjusted_motion_contract.json)
- [tracking_g1_resource_adjusted_csv_full_replay_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_csv_full_replay/tracking_g1_resource_adjusted_csv_full_replay_audit.json)
- [walk1_subject1_frames_1_180_resource_adjusted_full_replay_metrics.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_csv_full_replay/walk1_subject1_frames_1_180_resource_adjusted_full_replay_metrics.json)
- [tracking_g1_resource_adjusted_csv_task_eval_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_csv_task_eval/tracking_g1_resource_adjusted_csv_task_eval_audit.json)
- [tracking_g1_resource_adjusted_csv_task_eval_metrics.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_csv_task_eval/tracking_g1_resource_adjusted_csv_task_eval_metrics.json)
- [tracking_g1_resource_adjusted_train_entry_diagnostic_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_train_entry_diagnostic/tracking_g1_resource_adjusted_train_entry_diagnostic_audit.json)
- [tracking_g1_resource_adjusted_train_entry_diagnostic_metrics.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_train_entry_diagnostic/tracking_g1_resource_adjusted_train_entry_diagnostic_metrics.json)
- [tracking_urdf_conversion_probe.json](/mnt/infini-data/test/BeyondMimic/res/tracking/urdf_conversion_probe/tracking_urdf_conversion_probe.json)
- [tracking_urdf_path_tiny_probe.json](/mnt/infini-data/test/BeyondMimic/res/tracking/urdf_path_tiny_probe/tracking_urdf_path_tiny_probe.json)
- [tracking_mjcf_stage_probe.json](/mnt/infini-data/test/BeyondMimic/res/tracking/mjcf_stage_probe/tracking_mjcf_stage_probe.json)
- [tracking_usd_save_policy_probe.json](/mnt/infini-data/test/BeyondMimic/res/tracking/usd_save_policy_probe/tracking_usd_save_policy_probe.json)
- [tracking_simulationapp_save_policy_probe.json](/mnt/infini-data/test/BeyondMimic/res/tracking/simulationapp_save_policy_probe/tracking_simulationapp_save_policy_probe.json)
- [tracking_usd_api_variant_probe.json](/mnt/infini-data/test/BeyondMimic/res/tracking/usd_api_variant_probe/tracking_usd_api_variant_probe.json)
- [tracking_g1_urdf_stage_export_workaround_probe.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_urdf_stage_export_workaround/tracking_g1_urdf_stage_export_workaround_probe.json)
- [tracking_g1_urdf_layer_save_workaround_probe.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_urdf_layer_save_workaround/tracking_g1_urdf_layer_save_workaround_probe.json)
- [tracking_g1_urdf_in_memory_import_probe.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_urdf_in_memory_import/tracking_g1_urdf_in_memory_import_probe.json)
- [tracking_g1_urdf_simulationapp_in_memory_import_probe.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_urdf_simulationapp_in_memory_import/tracking_g1_urdf_simulationapp_in_memory_import_probe.json)
- [tracking_g1_urdf_in_memory_variant_matrix_probe.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_urdf_in_memory_variant_matrix/tracking_g1_urdf_in_memory_variant_matrix_probe.json)
- [tracking_g1_preconverted_asset_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_preconverted_asset_audit/tracking_g1_preconverted_asset_audit.json)
- [tracking_g1_reference_usd_compatibility_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_reference_usd_compatibility_audit/tracking_g1_reference_usd_compatibility_audit.json)
- [tracking_g1_official_urdf_skeleton_usd_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_urdf_skeleton_usd/tracking_g1_official_urdf_skeleton_usd_audit.json)
- [g1_official_urdf_29dof_skeleton.usda](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_urdf_skeleton_usd/g1_official_urdf_29dof_skeleton.usda)
- [tracking_g1_urdf_physical_asset_contract_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_urdf_physical_asset_contract_audit/tracking_g1_urdf_physical_asset_contract_audit.json)
- [tracking_g1_urdf_physical_asset_contract_audit.tsv](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_urdf_physical_asset_contract_audit/tracking_g1_urdf_physical_asset_contract_audit.tsv)
- [tracking_g1_urdf_source_equivalence_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_urdf_source_equivalence_audit/tracking_g1_urdf_source_equivalence_audit.json)
- [tracking_g1_urdf_source_equivalence_audit.tsv](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_urdf_source_equivalence_audit/tracking_g1_urdf_source_equivalence_audit.tsv)
- [tracking_g1_resource_adjusted_enriched_usd_probe.json](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_enriched_usd/tracking_g1_resource_adjusted_enriched_usd_probe.json)
- [g1_resource_adjusted_29dof_enriched_scaffold.usda](/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_enriched_usd/g1_resource_adjusted_29dof_enriched_scaffold.usda)
- [mujoco_ros_launch_contract_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/mujoco_ros_launch_contract_audit/mujoco_ros_launch_contract_audit.json)
- [tracking_deployment_controller_semantics_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/deployment_controller_semantics_audit/tracking_deployment_controller_semantics_audit.json)
- [tracking_onnx_export_contract_audit.json](/mnt/infini-data/test/BeyondMimic/res/tracking/onnx_export_contract_audit/tracking_onnx_export_contract_audit.json)
- [tracking_motion_policy_onnx_contract_fixture.json](/mnt/infini-data/test/BeyondMimic/res/tracking/motion_policy_onnx_contract_fixture/tracking_motion_policy_onnx_contract_fixture.json)
- [tracking_debug_motion_policy_onnx_export.json](/mnt/infini-data/test/BeyondMimic/res/tracking/debug_motion_policy_onnx_export/tracking_debug_motion_policy_onnx_export.json)
- [debug_motion_policy_contract.onnx](/mnt/infini-data/test/BeyondMimic/res/tracking/debug_motion_policy_onnx_export/debug_motion_policy_contract.onnx)
- [takeover_audit.json](/mnt/infini-data/test/BeyondMimic/res/takeover_audit/takeover_audit.json)
- [env_import_probe.json](/mnt/infini-data/test/BeyondMimic/res/setup/env_probe/env_import_probe.json)
- [bm_diffusion_env_audit.json](/mnt/infini-data/test/BeyondMimic/res/setup/bm_diffusion_env_audit/bm_diffusion_env_audit.json)
- [gpu_resource_audit.json](/mnt/infini-data/test/BeyondMimic/res/setup/gpu_resource_audit/gpu_resource_audit.json)
- [gpu_metrics.csv](/mnt/infini-data/test/BeyondMimic/logs/gpu/gpu_metrics.csv)
- [run_management_audit.json](/mnt/infini-data/test/BeyondMimic/res/run_management_audit/run_management_audit.json)
- [status.json](/mnt/infini-data/test/BeyondMimic/res/runs/setup_run_management_diagnostic_static_000_20260617_050000/status.json)
- [checkpoint_resume_smoke.json](/mnt/infini-data/test/BeyondMimic/res/run_management_audit/checkpoint_resume_smoke/checkpoint_resume_smoke.json)
- [status.json](/mnt/infini-data/test/BeyondMimic/res/runs/setup_checkpoint_resume_smoke_static_000_20260617_061500/status.json)
- [full_run_deliverable_gap_audit.json](/mnt/infini-data/test/BeyondMimic/res/run_management_audit/full_run_deliverable_gap_audit/full_run_deliverable_gap_audit.json)
- [failed_run_audit.json](/mnt/infini-data/test/BeyondMimic/res/failed_runs/failed_run_audit/failed_run_audit.json)
- [status.json](/mnt/infini-data/test/BeyondMimic/res/failed_runs/phase1_isaaclab_headless_smoke_g1_inotify_0_20260616_200654/status.json)
- [official_train_entry_failed_run_audit.json](/mnt/infini-data/test/BeyondMimic/res/failed_runs/official_train_entry_failed_run_audit/official_train_entry_failed_run_audit.json)
- [status.json](/mnt/infini-data/test/BeyondMimic/res/failed_runs/phase1_official_train_entry_retry_inotify_0_20260617_174742/status.json)
- [patch_inventory_audit.json](/mnt/infini-data/test/BeyondMimic/res/code/patch_inventory_audit/patch_inventory_audit.json)
- [patch_snapshot_audit.json](/mnt/infini-data/test/BeyondMimic/res/code/patch_snapshot_audit/patch_snapshot_audit.json)
- [reimpl_package_audit.json](/mnt/infini-data/test/BeyondMimic/res/code/reimpl_package_audit/reimpl_package_audit.json)
- [reimpl_runtime_integration_audit.json](/mnt/infini-data/test/BeyondMimic/res/code/reimpl_runtime_integration_audit/reimpl_runtime_integration_audit.json)
- [coding_requirements_audit.json](/mnt/infini-data/test/BeyondMimic/res/code/coding_requirements_audit/coding_requirements_audit.json)
- [reimpl_package_api_tests.json](/mnt/infini-data/test/BeyondMimic/res/tests/reimpl_package_api_tests/reimpl_package_api_tests.json)
- [reimpl_test_suite.json](/mnt/infini-data/test/BeyondMimic/res/tests/reimpl_test_suite/reimpl_test_suite.json)
- [__init__.py](/mnt/infini-data/test/BeyondMimic/reproduction/src/beyondmimic_reimpl/__init__.py)
- [resolved_reproduction_config.json](/mnt/infini-data/test/BeyondMimic/res/config/resolved_reproduction_config.json)
- [resolved_reproduction_config.yaml](/mnt/infini-data/test/BeyondMimic/res/config/resolved_reproduction_config.yaml)
- [artifact_manifest.json](/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json)
- [artifact_manifest.tsv](/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.tsv)
- [download_source_integrity_audit.json](/mnt/infini-data/test/BeyondMimic/res/source_integrity/download_source_integrity/download_source_integrity_audit.json)
- [download_source_integrity_manifest.tsv](/mnt/infini-data/test/BeyondMimic/res/source_integrity/download_source_integrity/download_source_integrity_manifest.tsv)
- [download_source_integrity_required.tsv](/mnt/infini-data/test/BeyondMimic/res/source_integrity/download_source_integrity/download_source_integrity_required.tsv)
- [run_log_config_catalog.json](/mnt/infini-data/test/BeyondMimic/res/run_log_config_catalog/run_log_config_catalog.json)
- [run_log_config_catalog.csv](/mnt/infini-data/test/BeyondMimic/res/run_log_config_catalog/run_log_config_catalog.csv)
- [README.md](/mnt/infini-data/test/BeyondMimic/README.md)
- [reproduction_report.md](/mnt/infini-data/test/BeyondMimic/res/final_report/reproduction_report.md)
- [final_report_requirement_audit.json](/mnt/infini-data/test/BeyondMimic/res/final_report/final_report_requirement_audit/final_report_requirement_audit.json)
- [final_deliverables_audit.json](/mnt/infini-data/test/BeyondMimic/res/final_deliverables_audit/final_deliverables_audit.json)
- [visual_media_inventory_audit.json](/mnt/infini-data/test/BeyondMimic/res/visual_media_inventory/visual_media_inventory_audit.json)
- [visual_media_inventory_audit.tsv](/mnt/infini-data/test/BeyondMimic/res/visual_media_inventory/visual_media_inventory_audit.tsv)
- [verification_command_coverage_audit.json](/mnt/infini-data/test/BeyondMimic/res/verification_command_coverage/verification_command_coverage_audit.json)
- [verification_command_syntax_audit.json](/mnt/infini-data/test/BeyondMimic/res/verification_command_syntax/verification_command_syntax_audit.json)
- [verification_command_script_manifest.json](/mnt/infini-data/test/BeyondMimic/res/verification_command_script_manifest/verification_command_script_manifest.json)
- [required_artifact_absence_audit.json](/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json)
- [trial_failure_accounting_audit.json](/mnt/infini-data/test/BeyondMimic/res/evaluation/trial_failure_accounting_audit/trial_failure_accounting_audit.json)
- [evaluation_metrics_coverage_audit.json](/mnt/infini-data/test/BeyondMimic/res/evaluation_metrics_coverage/evaluation_metrics_coverage_audit.json)
- [ablation_coverage_audit.json](/mnt/infini-data/test/BeyondMimic/res/ablation_coverage/ablation_coverage_audit.json)
- [metrics_catalog.json](/mnt/infini-data/test/BeyondMimic/res/metrics/metrics_catalog/metrics_catalog.json)
- [metrics_catalog.csv](/mnt/infini-data/test/BeyondMimic/res/metrics/metrics_catalog/metrics_catalog.csv)
- [released_data_metrics_summary.json](/mnt/infini-data/test/BeyondMimic/res/tables/released_data_metrics_summary/released_data_metrics_summary.json)
- [released_tracking_ablation_metrics.csv](/mnt/infini-data/test/BeyondMimic/res/tables/released_data_metrics_summary/released_tracking_ablation_metrics.csv)
- [released_grf_metrics.csv](/mnt/infini-data/test/BeyondMimic/res/tables/released_data_metrics_summary/released_grf_metrics.csv)
- [released_imu_metrics.csv](/mnt/infini-data/test/BeyondMimic/res/tables/released_data_metrics_summary/released_imu_metrics.csv)
- [released_data_statistical_audit.json](/mnt/infini-data/test/BeyondMimic/res/tables/released_data_statistical_audit/released_data_statistical_audit.json)
- [released_ablation_effect_sizes.csv](/mnt/infini-data/test/BeyondMimic/res/tables/released_data_statistical_audit/released_ablation_effect_sizes.csv)
- [released_grf_confidence_intervals.csv](/mnt/infini-data/test/BeyondMimic/res/tables/released_data_statistical_audit/released_grf_confidence_intervals.csv)
- [released_imu_confidence_intervals.csv](/mnt/infini-data/test/BeyondMimic/res/tables/released_data_statistical_audit/released_imu_confidence_intervals.csv)
- [level_a_released_data_suite.json](/mnt/infini-data/test/BeyondMimic/res/level_a/released_data_suite/level_a_released_data_suite.json)
- [guidance_task_coverage_audit.json](/mnt/infini-data/test/BeyondMimic/res/guidance_task_coverage/guidance_task_coverage_audit.json)
- [progress_report_audit.json](/mnt/infini-data/test/BeyondMimic/res/progress_report_audit/progress_report_audit.json)
- [completion_matrix_status_audit.json](/mnt/infini-data/test/BeyondMimic/res/docs/completion_matrix_status_audit/completion_matrix_status_audit.json)
- [project_boundary_audit.json](/mnt/infini-data/test/BeyondMimic/res/project_boundary_audit/project_boundary_audit.json)
- [experiment_protocol.md](/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md)
- [experiment_protocol_audit.json](/mnt/infini-data/test/BeyondMimic/res/docs/experiment_protocol_audit/experiment_protocol_audit.json)
- [readme_audit.json](/mnt/infini-data/test/BeyondMimic/res/docs/readme_audit/readme_audit.json)
- [paper_source_coverage_audit.json](/mnt/infini-data/test/BeyondMimic/res/paper_source_coverage/paper_source_coverage_audit.json)
- [paper_latex_inventory_audit.json](/mnt/infini-data/test/BeyondMimic/res/paper_latex_inventory/paper_latex_inventory_audit.json)
- [paper_latex_equations.tsv](/mnt/infini-data/test/BeyondMimic/res/paper_latex_inventory/paper_latex_equations.tsv)
- [paper_latex_experiment_settings.tsv](/mnt/infini-data/test/BeyondMimic/res/paper_latex_inventory/paper_latex_experiment_settings.tsv)
- [paper_formula_code_trace_audit.json](/mnt/infini-data/test/BeyondMimic/res/paper_formula_code_trace/paper_formula_code_trace_audit.json)
- [paper_formula_code_trace_audit.tsv](/mnt/infini-data/test/BeyondMimic/res/paper_formula_code_trace/paper_formula_code_trace_audit.tsv)
- [paper_pdf_source_consistency_audit.json](/mnt/infini-data/test/BeyondMimic/res/paper_pdf_source_consistency/paper_pdf_source_consistency_audit.json)
- [paper_pdf_anchor_audit.tsv](/mnt/infini-data/test/BeyondMimic/res/paper_pdf_source_consistency/paper_pdf_anchor_audit.tsv)
- [paper_source_tar_audit.tsv](/mnt/infini-data/test/BeyondMimic/res/paper_pdf_source_consistency/paper_source_tar_audit.tsv)
- [paper_table_value_audit.json](/mnt/infini-data/test/BeyondMimic/res/paper_table_values/paper_table_value_audit.json)
- [skill_success_table_data_audit.json](/mnt/infini-data/test/BeyondMimic/res/paper_skill_success_table_audit/skill_success_table_data_audit.json)
- [released_panel_mapping_audit.json](/mnt/infini-data/test/BeyondMimic/res/released_panel_mapping_audit/released_panel_mapping_audit.json)
- [paper_vs_reproduction.csv](/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.csv)
- [paper_vs_reproduction.md](/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.md)
- [level_c_diffusion_equation_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/diffusion_equation_audit/level_c_diffusion_equation_audit.json)
- [results_claims_audit.json](/mnt/infini-data/test/BeyondMimic/res/results_claims_audit/results_claims_audit.json)
- [level_c_trajectory_inverse_transform_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/trajectory_inverse_transform_audit/level_c_trajectory_inverse_transform_audit.json)
- [level_c_emphasis_projection_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/emphasis_projection_audit/level_c_emphasis_projection_audit.json)
- [level_c_state_representation_source_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/state_representation_source_audit/level_c_state_representation_source_audit.json)
- [level_c_dataset_collection_protocol_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/dataset_collection_protocol_audit/level_c_dataset_collection_protocol_audit.json)
- [level_c_rollout_rejection_manifest_probe.json](/mnt/infini-data/test/BeyondMimic/res/level_c/rollout_rejection_manifest_probe/level_c_rollout_rejection_manifest_probe.json)
- [state_latent_schema_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/state_latent_schema_audit/state_latent_schema_audit.json)
- [dagger_schema_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/dagger_schema_audit/dagger_schema_audit.json)
- [level_c_dagger_iteration_smoke.json](/mnt/infini-data/test/BeyondMimic/res/level_c/dagger_iteration_smoke/level_c_dagger_iteration_smoke.json)
- [level_c_paper_state_windows.json](/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_windows/level_c_paper_state_windows.json)
- [level_c_state_latent_dataset_consistency_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/state_latent_dataset_consistency_audit/level_c_state_latent_dataset_consistency_audit.json)
- [level_c_state_latent_training_dataset_contract_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/state_latent_training_dataset_contract_audit/level_c_state_latent_training_dataset_contract_audit.json)
- [level_c_paper_state_overfit_probe.json](/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_overfit_probe/level_c_paper_state_overfit_probe.json)
- [level_c_vae_latent_diffusion_overfit_probe.json](/mnt/infini-data/test/BeyondMimic/res/level_c/vae_latent_diffusion_overfit_probe/level_c_vae_latent_diffusion_overfit_probe.json)
- [level_c_paper_state_heldout_eval.json](/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_heldout_eval/level_c_paper_state_heldout_eval.json)
- [level_c_vae_latent_heldout_eval.json](/mnt/infini-data/test/BeyondMimic/res/level_c/vae_latent_heldout_eval/level_c_vae_latent_heldout_eval.json)
- [level_c_paper_state_heldout_multiseed_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_heldout_multiseed_audit/level_c_paper_state_heldout_multiseed_audit.json)
- [level_c_vae_latent_heldout_multiseed_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/vae_latent_heldout_multiseed_audit/level_c_vae_latent_heldout_multiseed_audit.json)
- [level_c_paper_state_transformer_arch_probe.json](/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_transformer_arch_probe/level_c_paper_state_transformer_arch_probe.json)
- [level_c_vae_latent_transformer_arch_probe.json](/mnt/infini-data/test/BeyondMimic/res/level_c/vae_latent_transformer_arch_probe/level_c_vae_latent_transformer_arch_probe.json)
- [level_c_transformer_parameter_count_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/transformer_parameter_count_audit/level_c_transformer_parameter_count_audit.json)
- [level_c_transformer_state_dict_manifest.json](/mnt/infini-data/test/BeyondMimic/res/level_c/transformer_state_dict_manifest/level_c_transformer_state_dict_manifest.json)
- [level_c_transformer_ema_smoke.json](/mnt/infini-data/test/BeyondMimic/res/level_c/transformer_ema_smoke/level_c_transformer_ema_smoke.json)
- [level_c_vae_latent_transformer_ema_smoke.json](/mnt/infini-data/test/BeyondMimic/res/level_c/vae_latent_transformer_ema_smoke/level_c_vae_latent_transformer_ema_smoke.json)
- [level_c_diffusion_checkpoint_smoke.json](/mnt/infini-data/test/BeyondMimic/res/level_c/diffusion_checkpoint_smoke/level_c_diffusion_checkpoint_smoke.json)
- [level_c_bounded_debug_diffusion_training_run.json](/mnt/infini-data/test/BeyondMimic/res/level_c/bounded_debug_diffusion_training_run/level_c_bounded_debug_diffusion_training_run.json)
- [status.json](/mnt/infini-data/test/BeyondMimic/res/runs/level_c_bounded_debug_diffusion_static_000_20260617_083000/status.json)
- [metrics.json](/mnt/infini-data/test/BeyondMimic/res/runs/level_c_bounded_debug_diffusion_static_000_20260617_083000/metrics.json)
- [debug_training_loss.png](/mnt/infini-data/test/BeyondMimic/res/runs/level_c_bounded_debug_diffusion_static_000_20260617_083000/figures/debug_training_loss.png)
- [level_c_bounded_debug_diffusion_checkpoint_eval.json](/mnt/infini-data/test/BeyondMimic/res/level_c/bounded_debug_diffusion_checkpoint_eval/level_c_bounded_debug_diffusion_checkpoint_eval.json)
- [level_c_bounded_debug_diffusion_action_eval.json](/mnt/infini-data/test/BeyondMimic/res/level_c/bounded_debug_diffusion_action_eval/level_c_bounded_debug_diffusion_action_eval.json)
- [level_c_resource_adjusted_tiny_diffusion_training_run.json](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_training_run/level_c_resource_adjusted_tiny_diffusion_training_run.json)
- [level_c_resource_adjusted_tiny_diffusion_suite.json](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_suite/level_c_resource_adjusted_tiny_diffusion_suite.json)
- [level_c_resource_adjusted_tiny_diffusion_suite.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_suite/level_c_resource_adjusted_tiny_diffusion_suite.tsv)
- [level_c_resource_adjusted_tiny_diffusion_multiseed_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_multiseed_audit/level_c_resource_adjusted_tiny_diffusion_multiseed_audit.json)
- [level_c_resource_adjusted_tiny_diffusion_multiseed_audit.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_multiseed_audit/level_c_resource_adjusted_tiny_diffusion_multiseed_audit.tsv)
- [level_c_resource_adjusted_tiny_diffusion_multiseed_audit.npz](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_multiseed_audit/level_c_resource_adjusted_tiny_diffusion_multiseed_audit.npz)
- [level_c_resource_adjusted_tiny_diffusion_checkpoint_eval.json](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_checkpoint_eval/level_c_resource_adjusted_tiny_diffusion_checkpoint_eval.json)
- [level_c_resource_adjusted_tiny_diffusion_checkpoint_eval.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_checkpoint_eval/level_c_resource_adjusted_tiny_diffusion_checkpoint_eval.tsv)
- [level_c_resource_adjusted_tiny_diffusion_checkpoint_eval.npz](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_checkpoint_eval/level_c_resource_adjusted_tiny_diffusion_checkpoint_eval.npz)
- [level_c_resource_adjusted_tiny_diffusion_onnx_export_inference.json](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_onnx_export_inference/level_c_resource_adjusted_tiny_diffusion_onnx_export_inference.json)
- [level_c_resource_adjusted_tiny_diffusion_onnx_export_inference.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_onnx_export_inference/level_c_resource_adjusted_tiny_diffusion_onnx_export_inference.tsv)
- [resource_adjusted_tiny_denoiser_onnx_debug_io.npz](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_onnx_export_inference/resource_adjusted_tiny_denoiser_onnx_debug_io.npz)
- [resource_adjusted_tiny_denoiser_debug.onnx](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_onnx_export_inference/resource_adjusted_tiny_denoiser_debug.onnx)
- [level_c_resource_adjusted_tiny_diffusion_latency_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_latency_audit/level_c_resource_adjusted_tiny_diffusion_latency_audit.json)
- [level_c_resource_adjusted_tiny_diffusion_latency_audit.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_latency_audit/level_c_resource_adjusted_tiny_diffusion_latency_audit.tsv)
- [level_c_resource_adjusted_tiny_diffusion_latency_audit.npz](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_latency_audit/level_c_resource_adjusted_tiny_diffusion_latency_audit.npz)
- [level_c_resource_adjusted_tiny_diffusion_video_preview.json](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_video_preview/level_c_resource_adjusted_tiny_diffusion_video_preview.json)
- [level_c_resource_adjusted_tiny_diffusion_video_preview.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_video_preview/level_c_resource_adjusted_tiny_diffusion_video_preview.tsv)
- [tiny_diffusion_validation_debug_preview_poster.png](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_video_preview/tiny_diffusion_validation_debug_preview_poster.png)
- [tiny_diffusion_test_debug_preview_poster.png](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_tiny_diffusion_video_preview/tiny_diffusion_test_debug_preview_poster.png)
- [level_c_resource_adjusted_teacher_rollout_vae_training.json](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_teacher_rollout_vae_training/level_c_resource_adjusted_teacher_rollout_vae_training.json)
- [level_c_resource_adjusted_teacher_rollout_vae_training.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_teacher_rollout_vae_training/level_c_resource_adjusted_teacher_rollout_vae_training.tsv)
- [level_c_resource_adjusted_teacher_rollout_state_latent_dataset.json](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_teacher_rollout_state_latent_dataset/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.json)
- [level_c_resource_adjusted_teacher_rollout_state_latent_dataset.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_teacher_rollout_state_latent_dataset/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.tsv)
- [level_c_resource_adjusted_state_latent_diffusion_training.json](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_state_latent_diffusion_training/level_c_resource_adjusted_state_latent_diffusion_training.json)
- [level_c_resource_adjusted_state_latent_diffusion_training.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_state_latent_diffusion_training/level_c_resource_adjusted_state_latent_diffusion_training.tsv)
- [level_c_resource_adjusted_state_latent_guidance_eval.json](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_state_latent_guidance_eval/level_c_resource_adjusted_state_latent_guidance_eval.json)
- [level_c_resource_adjusted_state_latent_guidance_eval.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/resource_adjusted_state_latent_guidance_eval/level_c_resource_adjusted_state_latent_guidance_eval.tsv)
- [tiny_diffusion_validation_debug_preview.gif](/mnt/infini-data/test/BeyondMimic/res/runs/level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500/videos/tiny_diffusion_validation_debug_preview.gif)
- [tiny_diffusion_test_debug_preview.gif](/mnt/infini-data/test/BeyondMimic/res/runs/level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500/videos/tiny_diffusion_test_debug_preview.gif)
- [status.json](/mnt/infini-data/test/BeyondMimic/res/runs/level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500/status.json)
- [metrics.json](/mnt/infini-data/test/BeyondMimic/res/runs/level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500/metrics.json)
- [tiny_denoiser_loss.png](/mnt/infini-data/test/BeyondMimic/res/runs/level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500/figures/tiny_denoiser_loss.png)
- [tiny_denoiser_eval_mse.png](/mnt/infini-data/test/BeyondMimic/res/runs/level_c_resource_adjusted_tiny_diffusion_static_000_20260617_091500/figures/tiny_denoiser_eval_mse.png)
- [level_c_lafan1_paper_arch_multiseed_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_multiseed_audit/level_c_lafan1_paper_arch_multiseed_audit.json)
- [level_c_lafan1_paper_arch_multiseed_rows.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_multiseed_audit/level_c_lafan1_paper_arch_multiseed_rows.tsv)
- [level_c_lafan1_paper_arch_multiseed_metrics.npz](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_multiseed_audit/level_c_lafan1_paper_arch_multiseed_metrics.npz)
- [level_c_lafan1_paper_arch_symmetry_augmented_multiseed_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_multiseed_audit/level_c_lafan1_paper_arch_symmetry_augmented_multiseed_audit.json)
- [level_c_lafan1_paper_arch_symmetry_augmented_multiseed_rows.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_multiseed_audit/level_c_lafan1_paper_arch_symmetry_augmented_multiseed_rows.tsv)
- [level_c_lafan1_paper_arch_symmetry_augmented_multiseed_metrics.npz](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_multiseed_audit/level_c_lafan1_paper_arch_symmetry_augmented_multiseed_metrics.npz)
- [level_c_lafan1_paper_arch_high_memory_batch_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_high_memory_batch_audit/level_c_lafan1_paper_arch_high_memory_batch_audit.json)
- [level_c_lafan1_paper_arch_high_memory_batch_rows.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_high_memory_batch_audit/level_c_lafan1_paper_arch_high_memory_batch_rows.tsv)
- [level_c_lafan1_paper_arch_high_memory_batch_fixture.npz](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_high_memory_batch_audit/level_c_lafan1_paper_arch_high_memory_batch_fixture.npz)
- [lafan1_paper_arch_symmetry_dataset_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_dataset/lafan1_paper_arch_symmetry_dataset_audit.json)
- [level_c_lafan1_paper_arch_symmetry_dataset_splits.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_dataset/level_c_lafan1_paper_arch_symmetry_dataset_splits.tsv)
- [lafan1_paper_arch_symmetry_augmented_dataset.npz](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_dataset/lafan1_paper_arch_symmetry_augmented_dataset.npz)
- [lafan1_paper_arch_vae_diffusion_training.json](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_training/lafan1_paper_arch_vae_diffusion_training.json)
- [lafan1_paper_arch_training_dataset.npz](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_training/lafan1_paper_arch_training_dataset.npz)
- [lafan1_paper_arch_vae_diffusion.pt](/mnt/infini-data/test/BeyondMimic/res/runs/level_c_lafan1_paper_arch_symmetry_augmented_static_000_20260617_215500/checkpoint/lafan1_paper_arch_vae_diffusion.pt)
- [metrics.json](/mnt/infini-data/test/BeyondMimic/res/runs/level_c_lafan1_paper_arch_symmetry_augmented_static_000_20260617_215500/metrics.json)
- [level_c_lafan1_paper_arch_symmetry_training_comparison_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_training_comparison/level_c_lafan1_paper_arch_symmetry_training_comparison_audit.json)
- [level_c_lafan1_paper_arch_symmetry_training_comparison.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_training_comparison/level_c_lafan1_paper_arch_symmetry_training_comparison.tsv)
- [level_c_lafan1_paper_arch_symmetry_training_comparison_metrics.npz](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_training_comparison/level_c_lafan1_paper_arch_symmetry_training_comparison_metrics.npz)
- [level_c_lafan1_paper_arch_onnx_latency_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_onnx_latency/level_c_lafan1_paper_arch_onnx_latency_audit.json)
- [level_c_lafan1_paper_arch_onnx_latency_components.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_onnx_latency/level_c_lafan1_paper_arch_onnx_latency_components.tsv)
- [level_c_lafan1_paper_arch_onnx_latency_rows.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_onnx_latency/level_c_lafan1_paper_arch_onnx_latency_rows.tsv)
- [lafan1_paper_arch_onnx_io_fixture.npz](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_onnx_latency/lafan1_paper_arch_onnx_io_fixture.npz)
- [lafan1_paper_arch_vae_decoder.onnx](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_onnx_latency/lafan1_paper_arch_vae_decoder.onnx)
- [lafan1_paper_arch_diffusion_denoiser.onnx](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_onnx_latency/lafan1_paper_arch_diffusion_denoiser.onnx)
- [level_c_lafan1_paper_arch_onnx_latency_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/level_c_lafan1_paper_arch_onnx_latency_audit.json)
- [level_c_lafan1_paper_arch_onnx_latency_components.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/level_c_lafan1_paper_arch_onnx_latency_components.tsv)
- [level_c_lafan1_paper_arch_onnx_latency_rows.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/level_c_lafan1_paper_arch_onnx_latency_rows.tsv)
- [lafan1_paper_arch_onnx_io_fixture.npz](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/lafan1_paper_arch_onnx_io_fixture.npz)
- [lafan1_paper_arch_vae_decoder.onnx](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/lafan1_paper_arch_vae_decoder.onnx)
- [lafan1_paper_arch_diffusion_denoiser.onnx](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency/lafan1_paper_arch_diffusion_denoiser.onnx)
- [level_c_lafan1_paper_arch_offline_metrics_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_audit.json)
- [level_c_lafan1_paper_arch_offline_metrics_rows.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_rows.tsv)
- [level_c_lafan1_paper_arch_offline_metrics.npz](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_offline_metrics/level_c_lafan1_paper_arch_offline_metrics.npz)
- [level_c_lafan1_paper_arch_offline_metrics_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_audit.json)
- [level_c_lafan1_paper_arch_offline_metrics_rows.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_rows.tsv)
- [level_c_lafan1_paper_arch_offline_metrics.npz](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_offline_metrics/level_c_lafan1_paper_arch_offline_metrics.npz)
- [level_c_lafan1_paper_arch_guidance_eval.json](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.json)
- [level_c_lafan1_paper_arch_guidance_eval.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.tsv)
- [level_c_lafan1_paper_arch_guidance_eval.npz](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.npz)
- [level_c_lafan1_paper_arch_guidance_eval.json](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.json)
- [level_c_lafan1_paper_arch_guidance_eval.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.tsv)
- [level_c_lafan1_paper_arch_guidance_eval.npz](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.npz)
- [level_c_lafan1_paper_arch_guidance_eval.json](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/level_c_lafan1_paper_arch_guidance_eval.json)
- [level_c_lafan1_paper_arch_guidance_eval.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/level_c_lafan1_paper_arch_guidance_eval.tsv)
- [level_c_lafan1_paper_arch_guidance_eval.npz](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split/level_c_lafan1_paper_arch_guidance_eval.npz)
- [level_c_lafan1_paper_arch_reverse_guidance_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_reverse_guidance/level_c_lafan1_paper_arch_reverse_guidance_audit.json)
- [level_c_lafan1_paper_arch_reverse_guidance_rows.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_reverse_guidance/level_c_lafan1_paper_arch_reverse_guidance_rows.tsv)
- [level_c_lafan1_paper_arch_reverse_guidance.npz](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_reverse_guidance/level_c_lafan1_paper_arch_reverse_guidance.npz)
- [level_c_lafan1_paper_arch_reverse_guidance_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance/level_c_lafan1_paper_arch_reverse_guidance_audit.json)
- [level_c_lafan1_paper_arch_reverse_guidance_rows.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance/level_c_lafan1_paper_arch_reverse_guidance_rows.tsv)
- [level_c_lafan1_paper_arch_reverse_guidance.npz](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance/level_c_lafan1_paper_arch_reverse_guidance.npz)
- [level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.json)
- [level_c_lafan1_paper_arch_reverse_guidance_full_split_rows.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/level_c_lafan1_paper_arch_reverse_guidance_full_split_rows.tsv)
- [level_c_lafan1_paper_arch_reverse_guidance_full_split_gpu_rows.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/level_c_lafan1_paper_arch_reverse_guidance_full_split_gpu_rows.tsv)
- [level_c_lafan1_paper_arch_reverse_guidance_full_split.npz](/mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split/level_c_lafan1_paper_arch_reverse_guidance_full_split.npz)
- [level_c_debug_suite.json](/mnt/infini-data/test/BeyondMimic/res/level_c/debug_suite/level_c_debug_suite.json)
- [level_c_extended_debug_suite.json](/mnt/infini-data/test/BeyondMimic/res/level_c/extended_debug_suite/level_c_extended_debug_suite.json)
- [level_c_single_batch_overfit_probe.json](/mnt/infini-data/test/BeyondMimic/res/level_c/single_batch_overfit_probe/level_c_single_batch_overfit_probe.json)
- [level_c_single_motion_overfit_probe.json](/mnt/infini-data/test/BeyondMimic/res/level_c/single_motion_overfit_probe/level_c_single_motion_overfit_probe.json)
- [level_c_small_dataset_overfit_probe.json](/mnt/infini-data/test/BeyondMimic/res/level_c/small_dataset_overfit_probe/level_c_small_dataset_overfit_probe.json)
- [small_dataset_split_manifest_summary.json](/mnt/infini-data/test/BeyondMimic/res/level_c/small_dataset_split_manifest/small_dataset_split_manifest_summary.json)
- [level_c_small_dataset_multiseed_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/small_dataset_multiseed_audit/level_c_small_dataset_multiseed_audit.json)
- [level_c_small_dataset_heldout_eval.json](/mnt/infini-data/test/BeyondMimic/res/level_c/small_dataset_heldout_eval/level_c_small_dataset_heldout_eval.json)
- [level_c_small_dataset_heldout_multiseed_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/small_dataset_heldout_multiseed_audit/level_c_small_dataset_heldout_multiseed_audit.json)
- [level_c_vae_checkpoint_smoke.json](/mnt/infini-data/test/BeyondMimic/res/level_c/vae_checkpoint_smoke/level_c_vae_checkpoint_smoke.json)
- [level_c_vae_debug_overfit_latent_artifact.json](/mnt/infini-data/test/BeyondMimic/res/level_c/vae_debug_overfit_latent_artifact/level_c_vae_debug_overfit_latent_artifact.json)
- [level_c_vae_motion_split_heldout_eval.json](/mnt/infini-data/test/BeyondMimic/res/level_c/vae_motion_split_heldout_eval/level_c_vae_motion_split_heldout_eval.json)
- [level_c_vae_receding_horizon_rollout_smoke.json](/mnt/infini-data/test/BeyondMimic/res/level_c/vae_receding_horizon_rollout_smoke/level_c_vae_receding_horizon_rollout_smoke.json)
- [level_c_diffusion_to_vae_action_smoke.json](/mnt/infini-data/test/BeyondMimic/res/level_c/diffusion_to_vae_action_smoke/level_c_diffusion_to_vae_action_smoke.json)
- [level_c_diffusion_to_vae_action_multiseed_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/diffusion_to_vae_action_multiseed_audit/level_c_diffusion_to_vae_action_multiseed_audit.json)
- [level_c_diffusion_to_vae_action_smoothness_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/diffusion_to_vae_action_smoothness_audit/level_c_diffusion_to_vae_action_smoothness_audit.json)
- [level_c_direct_vs_latent_action_ablation_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/direct_vs_latent_action_ablation_audit/level_c_direct_vs_latent_action_ablation_audit.json)
- [level_c_vae_contract_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/vae_contract_audit/level_c_vae_contract_audit.json)
- [level_c_dagger_vae_pipeline_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/dagger_vae_pipeline_audit/level_c_dagger_vae_pipeline_audit.json)
- [level_c_vae_latent_probe.json](/mnt/infini-data/test/BeyondMimic/res/level_c/vae_latent_probe/level_c_vae_latent_probe.json)
- [level_c_symmetry_mapping_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/symmetry_mapping_audit/level_c_symmetry_mapping_audit.json)
- [level_c_guidance_task_scale_sweep.json](/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_task_scale_sweep/level_c_guidance_task_scale_sweep.json)
- [level_c_guidance_debug_visualization.json](/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_debug_visualization/level_c_guidance_debug_visualization.json)
- [level_c_guidance_task_metric_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_task_metric_audit/level_c_guidance_task_metric_audit.json)
- [level_c_guidance_full_split_result_table.json](/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.json)
- [level_c_guidance_full_split_result_table.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.tsv)
- [level_c_guidance_full_split_result_table.csv](/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_result_table.csv)
- [level_c_guidance_full_split_cost_delta.pdf](/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_cost_delta.pdf)
- [level_c_guidance_full_split_cost_delta.svg](/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_cost_delta.svg)
- [level_c_guidance_full_split_cost_delta.png](/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_full_split_result_table/level_c_guidance_full_split_cost_delta.png)
- [level_c_guidance_checkpoint_visualization.json](/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/level_c_guidance_checkpoint_visualization.json)
- [level_c_guidance_checkpoint_visualization.tsv](/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/level_c_guidance_checkpoint_visualization.tsv)
- [checkpoint_guidance_joystick.png](/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_joystick.png)
- [checkpoint_guidance_waypoint.png](/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_waypoint.png)
- [checkpoint_guidance_obstacle_avoidance.png](/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_obstacle_avoidance.png)
- [checkpoint_guidance_inpainting.png](/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_inpainting.png)
- [checkpoint_guidance_composed_objectives.png](/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_composed_objectives.png)
- [checkpoint_guidance_composed_preview.gif](/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_checkpoint_visualization/checkpoint_guidance_composed_preview.gif)
- [level_c_guidance_visual_deliverables_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_visual_deliverables_audit/level_c_guidance_visual_deliverables_audit.json)
- [level_c_guidance_cost_coverage_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/guidance_cost_coverage_audit/level_c_guidance_cost_coverage_audit.json)
- [core_math_unit_tests.json](/mnt/infini-data/test/BeyondMimic/res/tests/core_math_unit_tests/core_math_unit_tests.json)
- [core_test_coverage_audit.json](/mnt/infini-data/test/BeyondMimic/res/tests/core_test_coverage_audit/core_test_coverage_audit.json)
- [level_c_timestep_mask_coverage_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/timestep_mask_coverage_audit/level_c_timestep_mask_coverage_audit.json)
- [level_c_paper_state_mask_reverse_probe.json](/mnt/infini-data/test/BeyondMimic/res/level_c/paper_state_mask_reverse_probe/level_c_paper_state_mask_reverse_probe.json)
- [level_c_smoothness_latency_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/smoothness_latency_audit/level_c_smoothness_latency_audit.json)
- [level_c_deployment_protocol_audit.json](/mnt/infini-data/test/BeyondMimic/res/level_c/deployment_protocol_audit/level_c_deployment_protocol_audit.json)
- [goal_traceability_audit.json](/mnt/infini-data/test/BeyondMimic/res/goal_traceability/goal_traceability_audit.json)
- [goal_directive_index_audit.json](/mnt/infini-data/test/BeyondMimic/res/goal_directive_index/goal_directive_index_audit.json)
- [goal_directive_rows.tsv](/mnt/infini-data/test/BeyondMimic/res/goal_directive_index/goal_directive_rows.tsv)
- [goal_heading_rows.tsv](/mnt/infini-data/test/BeyondMimic/res/goal_directive_index/goal_heading_rows.tsv)
- [goal_requirement_matrix_audit.json](/mnt/infini-data/test/BeyondMimic/res/goal_requirement_matrix/goal_requirement_matrix_audit.json)
- [released_figure_summary.tsv](/mnt/infini-data/test/BeyondMimic/res/released_figures/released_figure_summary.tsv)
- [completion_matrix.md](/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md)
- [level_c_diffusion_plan.md](/mnt/infini-data/test/BeyondMimic/reproduction/docs/level_c_diffusion_plan.md)
- [unresolved_details.md](/mnt/infini-data/test/BeyondMimic/reproduction/docs/unresolved_details.md)

## Verification Commands
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/blocked_gate_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_source_coverage_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_latex_inventory_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_formula_code_trace_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_pdf_source_consistency_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_table_value_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/skill_success_table_data_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/released_panel_mapping_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/results_claims_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/bm_diffusion_env_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/gpu_resource_audit.py --samples 3 --interval-sec 1
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/create_run_management_skeleton.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/checkpoint_resume_smoke.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/full_run_deliverable_gap_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/record_failed_run.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/record_official_train_entry_failed_run.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/resolved_reproduction_config.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/download_source_integrity_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_log_config_catalog.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/experiment_protocol_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/readme_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/patch_inventory_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/patch_snapshot_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_report_requirement_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_deliverables_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/visual_media_inventory_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/verification_command_coverage_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/verification_command_syntax_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/verification_command_script_manifest.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/required_artifact_absence_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/trial_failure_accounting_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/evaluation_metrics_coverage_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/ablation_coverage_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/metrics_catalog.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/released_data_metrics_summary.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/released_data_statistical_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_level_a_released_data_suite.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/guidance_task_coverage_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/progress_report_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/completion_matrix_status_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/project_boundary_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/core_test_coverage_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/coding_requirements_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/tests/test_reimpl_package_api.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_reimpl_test_suite.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_diffusion_equation_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_trajectory_inverse_transform_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_emphasis_projection_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_state_representation_source_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_dataset_collection_protocol_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_rollout_rejection_manifest_probe.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/state_latent_schema_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/dagger_schema_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_dagger_iteration_smoke.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/build_level_c_paper_state_windows.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_state_latent_dataset_consistency_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_state_latent_training_dataset_contract_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_state_overfit_probe.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_latent_diffusion_overfit_probe.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_state_heldout_eval.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_latent_heldout_eval.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_state_heldout_multiseed_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_latent_heldout_multiseed_audit.py
```
```bash
BM_LEVEL_C_TORCH_THREADS=2 /mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_state_transformer_arch_probe.py --device cuda:0 --batch-size 1
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_latent_transformer_arch_probe.py --device cpu --batch-size 1
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_transformer_parameter_count_audit.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_transformer_state_dict_manifest.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_transformer_ema_smoke.py --device cpu --steps 2
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_latent_transformer_ema_smoke.py --device cpu --steps 2
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_diffusion_checkpoint_smoke.py --device cpu
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_bounded_debug_diffusion_training_run.py --device cpu --steps 3 --batch-size 1 --torch-threads 2
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_bounded_debug_diffusion_checkpoint_eval.py --device cpu --torch-threads 2
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_bounded_debug_diffusion_action_eval.py --device cpu --torch-threads 2
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_level_c_resource_adjusted_tiny_diffusion_suite.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_training_run.py --device cpu --torch-threads 2 --epochs 180
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_multiseed_audit.py --device cpu --torch-threads 2 --epochs 80
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_checkpoint_eval.py --device cpu --torch-threads 2
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_onnx_export_inference.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_latency_audit.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_tiny_diffusion_video_preview.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_state_latent_guidance_eval.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/train_lafan1_paper_level_vae_diffusion.py --device cuda:0 --max-motions 40 --max-frames-per-motion 420 --vae-epochs 24 --diffusion-epochs 1000 --diffusion-batch-size 512 --data-parallel
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_lafan1_paper_arch_multiseed_audit.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_lafan1_paper_arch_symmetry_dataset_audit.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/train_lafan1_paper_level_vae_diffusion.py --device cuda:0 --seed 20260621 --projection-seed 20260617 --run-id level_c_lafan1_paper_arch_symmetry_augmented_static_000_20260617_215500 --output-dir /mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_training --dataset-npz /mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_dataset/lafan1_paper_arch_symmetry_augmented_dataset.npz --dataset-source-label public_lafan1_symmetry_augmented_dataset --max-motions 40 --max-frames-per-motion 420 --vae-epochs 24 --diffusion-epochs 1000 --diffusion-batch-size 512 --data-parallel
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/train_lafan1_paper_level_vae_diffusion.py --device cuda:0 --seed 20260622 --projection-seed 20260617 --run-id level_c_lafan1_paper_arch_symmetry_augmented_seed_20260622_static_000_20260617_215500 --output-dir /mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_training_seed_20260622 --dataset-npz /mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_dataset/lafan1_paper_arch_symmetry_augmented_dataset.npz --dataset-source-label public_lafan1_symmetry_augmented_dataset --max-motions 40 --max-frames-per-motion 420 --vae-epochs 24 --diffusion-epochs 1000 --diffusion-batch-size 512 --data-parallel
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/train_lafan1_paper_level_vae_diffusion.py --device cuda:0 --seed 20260623 --projection-seed 20260617 --run-id level_c_lafan1_paper_arch_symmetry_augmented_seed_20260623_static_000_20260617_215500 --output-dir /mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_training_seed_20260623 --dataset-npz /mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_dataset/lafan1_paper_arch_symmetry_augmented_dataset.npz --dataset-source-label public_lafan1_symmetry_augmented_dataset --max-motions 40 --max-frames-per-motion 420 --vae-epochs 24 --diffusion-epochs 1000 --diffusion-batch-size 512 --data-parallel
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_lafan1_paper_arch_symmetry_multiseed_audit.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_lafan1_paper_arch_high_memory_batch_audit.py --initial-batch-size 8192 --max-batch-size 32768 --target-memory-mb 20000
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_lafan1_symmetry_training_comparison_audit.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_lafan1_paper_arch_onnx_latency_audit.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_lafan1_paper_arch_onnx_latency_audit.py --training-json /mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_training/lafan1_paper_arch_vae_diffusion_training.json --output-dir /mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_onnx_latency
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_lafan1_paper_arch_offline_metrics_audit.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_lafan1_paper_arch_offline_metrics_audit.py --training-json /mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_training/lafan1_paper_arch_vae_diffusion_training.json --output-dir /mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_offline_metrics
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_lafan1_paper_arch_guidance_eval.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_lafan1_paper_arch_guidance_eval.py --training-json /mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_training/lafan1_paper_arch_vae_diffusion_training.json --offline-json /mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_audit.json --output-dir /mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_lafan1_paper_arch_guidance_eval.py --training-json /mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_training/lafan1_paper_arch_vae_diffusion_training.json --offline-json /mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_offline_metrics/level_c_lafan1_paper_arch_offline_metrics_audit.json --output-dir /mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval_full_split --splits validation,test --max-windows-per-split -1
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_lafan1_paper_arch_reverse_guidance_audit.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_lafan1_paper_arch_reverse_guidance_audit.py --training-json /mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_training/lafan1_paper_arch_vae_diffusion_training.json --offline-guidance-json /mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_guidance_eval/level_c_lafan1_paper_arch_guidance_eval.json --output-dir /mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_lafan1_paper_arch_reverse_guidance_full_split_audit.py --output-dir /mnt/infini-data/test/BeyondMimic/res/level_c/lafan1_paper_arch_symmetry_augmented_reverse_guidance_full_split --splits validation,test --max-windows-per-split -1 --batch-size 660 --target-memory-mb 10000
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_single_batch_overfit_probe.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_single_motion_overfit_probe.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_small_dataset_overfit_probe.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/build_level_c_small_dataset_split_manifest.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_small_dataset_multiseed_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_small_dataset_heldout_eval.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_small_dataset_heldout_multiseed_audit.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_checkpoint_smoke.py --device cpu
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_debug_overfit_latent_artifact.py --device cpu
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_motion_split_heldout_eval.py --device cpu
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_receding_horizon_rollout_smoke.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_diffusion_to_vae_action_smoke.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_diffusion_to_vae_action_multiseed_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_diffusion_to_vae_action_smoothness_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_direct_vs_latent_action_ablation_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_contract_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_level_c_dagger_vae_pipeline_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_vae_latent_probe.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_symmetry_mapping_audit.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_guidance_task_scale_sweep.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_guidance_debug_visualization.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_guidance_task_metric_audit.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_guidance_full_split_result_table.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_guidance_checkpoint_visualization.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_guidance_visual_deliverables_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_guidance_cost_coverage_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_level_c_debug_suite.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_level_c_extended_debug_suite.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/tests/test_core_math.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/reimpl_package_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/reimpl_runtime_integration_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_timestep_mask_coverage_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_state_mask_reverse_probe.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_smoothness_latency_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_deployment_protocol_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_smoke_rerun_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_official_train_entry_retry_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/kit_inotify_budget_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/inotify_live_usage_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/vulkan_runtime_probe.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/cuda_p2p_runtime_probe.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/isaaclab_live_gate_probe.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/isaaclab_gpu_foundation_settings_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/vscode_watcher_exclude_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/kit_watcher_config_surface_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_import_gate_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_extension_namespace_probe.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_official_source_contract_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_action_scale_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_reward_formula_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_observation_action_schema_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_randomization_termination_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/run_level_b_tracking_nonkit_suite.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/adaptive_sampling_discrepancy_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/motion_preprocessing_contract_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_local_smoke_preflight.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/build_tracking_motion_npz_fixture.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_official_replay_conversion_audit.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_official_replay_npz_entry_diagnostic_audit.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_official_csv_to_npz_loop_with_enriched_usd_audit.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_official_replay_npz_loop_with_enriched_usd_audit.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_urdf_conversion_probe.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_urdf_path_tiny_probe.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_mjcf_stage_probe.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_usd_save_policy_probe.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_simulationapp_save_policy_probe.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_usd_api_variant_probe.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_urdf_stage_export_workaround_probe.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_urdf_layer_save_workaround_probe.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_urdf_in_memory_import_probe.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_urdf_simulationapp_in_memory_import_probe.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_urdf_in_memory_variant_matrix_probe.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_preconverted_asset_audit.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_reference_usd_compatibility_audit.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_urdf_skeleton_usd_audit.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_urdf_physical_asset_contract_audit.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_enriched_usd_probe.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/mujoco_ros_launch_contract_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_deployment_controller_semantics_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_onnx_export_contract_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_motion_policy_onnx_contract_fixture.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_debug_motion_policy_onnx_export.py
```
```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_diffusion/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_debug_motion_policy_onnx_inference_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/goal_traceability_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/goal_directive_index_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/goal_requirement_matrix_audit.py
```
```bash
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py
```

## Boundary
The report is a consolidation artifact. It does not replace live IsaacLab rollouts, true DAgger data collection, trained VAE/diffusion checkpoints, TensorRT deployment, Fig. 5/Fig. 6 paper reproduction, or real Unitree G1 execution.
