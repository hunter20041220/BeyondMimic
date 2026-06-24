# BeyondMimic 训练前硬门控审计

- 状态：`blocked_pretraining_hard_gate_requires_teacher_and_adapter_fixes`
- 行数：`9`
- 阻塞项：`7`
- claim level：`audit_and_gate_only; no new training; no success video claim`

## 结论

当前只能把 Stage-1 teacher 重新训练/评估作为纠错方向；不能从当前 weak teacher 继续训练 VAE/diffusion，也不能把现有 MuJoCo 视频作为成功视频。

当前不得声称完整复现 BeyondMimic，除非所有 master audit 和 required paper-level gates 都通过。

## Permission

- `start_stage1_teacher_retraining`: `conditional_only_as_corrective_work_on_official_whole_body_tracking_route`
- `start_downstream_vae_training`: `False`
- `start_state_latent_diffusion_training`: `False`
- `start_guided_closed_loop_video_generation`: `False`
- `create_final_singleleg_success_folder`: `False`

## Gate Rows

### appendix_parameter_matrix_contract

- 结果：`BLOCK` / `blocked_appendix_matrix_has_required_fixes`
- 决策：`block_long_training_until_appendix_matrix_passes`
- 原因：The new appendix matrix keeps paper/appendix parameters, official public-code differences, and native MuJoCo deployment gates in one machine-readable pre-training checklist.
- 修复要求：Resolve or explicitly waive every appendix-matrix blocker before starting long downstream training or creating final success videos.
- 声明边界：This is an audit-only gate; it does not claim a trained policy, VAE, diffusion model, or video succeeded.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/res/audits/appendix_parameter_matrix/beyondmimic_appendix_parameter_matrix_audit.json`
  - `appendix_status=blocked_appendix_parameter_matrix_has_required_fixes`
  - `appendix_blocking_count=10`
  - `appendix_blocking_items=['Stage-1 RL / Termination thresholds', 'Stage-1 RL / PD gains, armature, and action scale', 'Stage-1 RL / Domain randomization table S2', 'Stage-1 RL / Adaptive sampling', 'VAE / Conditional VAE input and architecture table S5', 'Diffusion / State-latent representation and data collection']`

### paper_and_official_stage1_formula_contract

- 结果：`PASS` / `pass_for_formula_audit_but_not_teacher_quality`
- 决策：`allow_stage1_teacher_retraining_only_as_corrective_work`
- 原因：Official whole_body_tracking remains the right Stage-1 route and the main observation/action/reward/PD/PPO contracts are traced. This does not mean the current checkpoint is good.
- 修复要求：Use the official Stage-1 path for the next teacher run; do not change rewards or action semantics just to get better videos.
- 声明边界：This gate permits only corrective teacher training/evaluation, not downstream VAE/diffusion success claims.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/res/audits/formula_parameter_trace_audit/beyondmimic_formula_parameter_trace_audit.json`
  - `/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_tracking_parameter_contract_audit/stage1_tracking_parameter_contract_audit.json`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:110`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py:184`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py:8`

### teacher_quality_for_downstream_rollout_dataset

- 结果：`BLOCK` / `blocked_teacher_quality_failed`
- 决策：`block_downstream_dataset_vae_diffusion_video`
- 原因：The current teachers are weak/reset-heavy; this directly explains front-leaning or generic posture outputs.
- 修复要求：Select/train a teacher with low non-timeout done rate, continuous motion-time, meaningful reward, and target pose/body error gates before collecting VAE data.
- 声明边界：Current teacher/VAE/diffusion videos remain diagnostic only.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/res/tracking/hub_singleleg_paper_contract_ppo_checkpoint_eval/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.json`
  - `/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.json`
  - `singleleg_quality_gate=False`
  - `singleleg_done_rate=0.2793295359531773`
  - `singleleg_reward_mean=0.04114155067647979`
  - `multisource_done_rate=0.19413670568561872`
  - `multisource_reward_mean=0.024131401152315747`

### conditional_vae_paper_contract

- 结果：`BLOCK` / `blocked_until_teacher_quality_passes`
- 决策：`block_vae_long_training_from_current_teacher`
- 原因：The corrected VAE interface exists, but using a failed/reset-heavy teacher would train the VAE to imitate poor control. The old obs+action encoder route is explicitly disallowed for success videos.
- 修复要求：Retrain the paper-contract VAE only after the accepted teacher rollout dataset exists.
- 声明边界：VAE MSE alone is not enough for paper-level or video success claims.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_contract_teacher_rollout_vae_training/level_c_paper_contract_teacher_rollout_vae_training.json`
  - `paper_contract_vae_interface_ok=True`
  - `source_teacher_done_rate_low_enough=False`

### state_latent_diffusion_paper_contract

- 结果：`BLOCK` / `blocked_transformer_full_training_not_done`
- 决策：`block_diffusion_long_training_from_current_dataset`
- 原因：A paper-style Transformer code contract is present, but the full trained Transformer over an accepted teacher/VAE dataset is not available. MLP/resource-adjusted denoisers remain debug baselines.
- 修复要求：Use the paper-contract Transformer only after teacher and VAE gates pass; then run full train/eval with held-out splits.
- 声明边界：Current denoising MSE improvements are local diagnostic evidence, not proof of successful control.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_contract_transformer_state_latent_diffusion_training/paper_contract_transformer_state_latent_diffusion_training.json`
  - `transformer_code_contract_ok=True`
  - `dry_run=True`

### classifier_guidance_closed_loop_contract

- 结果：`BLOCK` / `blocked_offline_only`
- 决策：`block_guided_success_videos`
- 原因：Guidance has offline gradient/cost evidence, but no validated receding-horizon native closed-loop control path.
- 修复要求：After a trained paper-contract diffusion model exists, run receding-horizon closed-loop guidance with task metrics.
- 声明边界：Current guided videos are not Fig.5/Fig.6-level guidance evidence.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_paper_contract_state_latent_guidance_eval/level_c_official_importer_export_paper_contract_state_latent_guidance_eval.json`
  - `offline_guidance_ok=True`
  - `native_obs_adapter_ready=False`

### mujoco_native_action_adapter

- 结果：`PASS` / `partial_formula_ready_but_rollout_not_ready`
- 决策：`allow_formula_fixture_only`
- 原因：The theta0 + alpha * action fixture is available, and the no-action-clipping MuJoCo ctrlrange patch covers unit setpoints.
- 修复要求：Use the patched/generated no-clipping actuator range in future no-root-assist videos; this still does not validate observations or physics success.
- 声明边界：Formula fixture success is not a physics rollout success.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json`
  - `formula_adapter_ready=True`
  - `unit_targets_inside_ctrlrange=True`

### mujoco_native_observation_adapter

- 结果：`BLOCK` / `blocked_not_validated_against_isaaclab_or_deployment`
- 决策：`block_native_ppo_vae_diffusion_rollout`
- 原因：A dimension-correct 160-D vector can still be semantically wrong, which can produce leaning or collapsed postures.
- 修复要求：Numerically validate the MuJoCo observation builder against IsaacLab observation_manager and motion_tracking_controller frame semantics.
- 声明边界：No native MuJoCo PPO/VAE/diffusion rollout claim is allowed until this passes.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json`
  - `native_adapter_validated_against_isaaclab=False`
  - `native_adapter_validated_against_deployment=False`

### mujoco_physics_video_success_gate

- 结果：`BLOCK` / `blocked_root_assist_blending_and_material_gap`
- 决策：`block_final_success_folder`
- 原因：Readable videos currently rely on diagnostic aids or weak/local chains. Root assist, blending, absolute targets, and material mismatch prevent final success claims.
- 修复要求：Generate the final folder only from no-root-assist native action control with continuous motion-time and low fall metrics.
- 声明边界：Existing clean-walk/single-leg videos should be treated as failed/diagnostic unless later superseded.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_control_contract_audit/mujoco_control_contract_audit.json`
  - `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_control_suite_sweep/clean_walk_model_weight_sweep_summary.json`
  - `/mnt/infini-data/test/BeyondMimic/res/visualization/final_clean_walk_six_mujoco_videos_scaled_ppo_pure/clean_walk_mujoco_control_suite_summary.json`
  - `pure_model_weight_1p0_ok=False`
  - `no_root_assist_native_video_ok=False`
  - `final_walk_has_root_assist_boundary=True`
