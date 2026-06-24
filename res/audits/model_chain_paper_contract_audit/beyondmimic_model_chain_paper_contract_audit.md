# BeyondMimic 模型链论文合同审计

- 状态：`blocked_model_chain_not_paper_contract_and_teacher_quality_not_ready`
- 生成时间：`2026-06-24T05:14:33.083950+00:00`
- 结论：当前 teacher/VAE/diffusion 视频链不能声称已经学会单脚站立或正常走路。
- 当前不得声称完整复现 BeyondMimic，也不得把现有前倾站姿视频作为成功结果。
- 旧 resource-adjusted VAE/diffusion 链条必须继续标记为 diagnostic；新的 paper-contract VAE 链条可作为后续候选，但仍被 teacher quality、Transformer diffusion、closed-loop guidance gate 阻塞。

## 失败检查

- `mujoco_control_contract_native_ready`
- `mujoco_native_observation_adapter_ready`
- `public_lafan1_arch_full_vae_contract`
- `video_chain_success_claim_allowed`
- `singleleg_teacher_quality_gate_passed`
- `current_gpu_memory_target_reached_80gb_each`

## 模块结论

### stage1_tracking_parameter_contract_gate
- 状态：`blocked_stage1_teacher_contract_has_required_followups`
- 是否满足论文合同：`True`
- 说明：Stage-1 formula/parameter contract is now separately audited. The public/official code mostly matches paper contracts, but the current teacher quality gate remains blocked, so downstream VAE/diffusion cannot be treated as final.

### paper_contract_tracking_observation_action_reward
- 状态：`reference_contract_available`
- 是否满足论文合同：`True`
- 说明：The paper defines motion phase, anchor error, IMU/root twist, joint state, and last action; actions are normalized PD setpoints; rewards are exponential body tracking terms plus three regularizers.

### official_whole_body_tracking_stage1
- 状态：`mostly_matches_available_official_stage1_code`
- 是否满足论文合同：`True`
- 说明：Stage 1 should continue to use the official whole_body_tracking IsaacLab/RSL-RL task. The weak local videos are not evidence that the official Stage-1 formulation is wrong; they show the local teacher/checkpoint/data quality is not yet good enough.

### resource_adjusted_teacher_rollout_vae
- 状态：`fails_paper_vae_contract`
- 是否满足论文合同：`False`
- 说明：Local VAE encoder is obs+action -> latent and decoder is obs+latent -> action. The paper VAE encoder should encode reference motion intent E(psi, e_anchor), while the decoder combines z with proprioception. This mismatch can collapse learned outputs toward a generic posture.

### paper_contract_teacher_rollout_vae
- 状态：`ok_paper_contract_teacher_rollout_vae_training_completed`
- 是否满足论文合同：`True`
- 说明：This is the preferred local VAE route because it repairs the major formula-level interface: encoder uses reference intent terms and decoder uses proprioception plus latent. It still cannot be called paper-level because the source teacher rollout is local, not official DAgger, and the source teacher quality gate remains weak.

### resource_adjusted_state_latent_diffusion
- 状态：`fails_paper_diffusion_contract`
- 是否满足论文合同：`False`
- 说明：This path trains an MLP denoiser over policy_obs+latent windows. The paper uses a state-latent trajectory with hybrid character-yaw state, emphasis projection, individual state/latent denoising steps, and a Transformer denoiser. It is useful as a local diagnostic, not as the success video chain.

### paper_contract_state_latent_dataset
- 状态：`ok_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset`
- 是否满足论文合同：`True`
- 说明：This dataset is the correct local route after paper-contract VAE, but its state source is still policy_obs rather than the full hybrid character-yaw state described in the paper, and it uses local teacher rollouts with many done events. It is a local contract candidate, not paper-level data.

### paper_contract_state_latent_diffusion
- 状态：`ok_official_importer_export_paper_contract_state_latent_diffusion_training`
- 是否满足论文合同：`True`
- 说明：This is the current preferred local denoiser output because it improves denoising on a full paper-contract local window dataset. However the base implementation still uses the resource-adjusted MLP denoiser path, not the paper's 6-layer Transformer with 512-d embeddings and per-state/latent denoising embeddings. It therefore remains local diagnostic evidence.

### paper_contract_transformer_state_latent_diffusion_code_contract
- 状态：`ok_paper_contract_transformer_diffusion_dry_run`
- 是否满足论文合同：`True`
- 说明：This is the corrected local code-contract route for the paper-style state-latent diffusion model: 6-layer Transformer, 512-d embeddings, 8 attention heads, 20 denoising steps, separate state/latent denoising-step embeddings, and clean-trajectory prediction. It has only been dry-run tested on a tiny local subset, so it proves architecture/gradient viability but not full training quality or closed-loop control.

### paper_contract_offline_guidance
- 状态：`ok_official_importer_export_paper_contract_state_latent_guidance_eval`
- 是否满足论文合同：`True`
- 说明：The paper-contract guidance audit verifies differentiable cost gradients and cost reduction offline. It does not perform receding-horizon closed-loop MuJoCo/Isaac control, and its cost reductions are very small, so it cannot support Fig.5/Fig.6-style claims yet.

### mujoco_control_contract_gate
- 状态：`blocked_mujoco_control_semantics_not_native_policy_control`
- 是否满足论文合同：`False`
- 说明：The local MuJoCo videos have useful official G1 PD/action-scale numbers, but the current video adapter still uses absolute/IK joint targets, default root assist, and material/friction differences. It is therefore a diagnostic visualization route, not the native paper control path.

### mujoco_native_action_adapter_formula_gate
- 状态：`ok_native_action_adapter_formula_and_no_clip_ctrlrange_patch_ready`
- 是否满足论文合同：`True`
- 说明：The normalized-action-to-PD-setpoint formula gate is now available: theta_sp = theta_default + action_scale * clipped_action, with official joint order, deployment default pose, and action-scale rows. This is only a formula/order fixture; it does not prove native observations, physics stability, or video success.

### mujoco_native_observation_adapter_gate
- 状态：`blocked_native_mujoco_observation_adapter_not_validated`
- 是否满足论文合同：`False`
- 说明：The exact 160-D observation layout, empirical-normalizer requirement, and official deployment frame semantics are now enumerated. This gate remains blocked because the MuJoCo builder has not been numerically validated against IsaacLab observation_manager or motion_tracking_controller worldToInit_/Pinocchio local-frame semantics, and no no-root-assist native rollout has passed.

### lafan1_paper_arch_training
- 状态：`paper_architecture_public_data_approximation_not_full_paper_contract`
- 是否满足论文合同：`False`
- 说明：This path correctly uses a Transformer diffusion backbone, individual state/latent denoising steps, state projection, 20 diffusion steps, and paper-scale VAE/Transformer dimensions. However its VAE still encodes state+action rather than E(psi, e_anchor), and its dataset is public retargeted LAFAN1 reference data rather than teacher DAgger/VAE rollout trajectories.

### clean_walk_and_singleleg_video_chain
- 状态：`fails_success_claim_video_contract`
- 是否满足论文合同：`False`
- 说明：The video chain uses reference anchoring/model-target blending, one-step latent denoise, latent interpolation as 'guidance', and MuJoCo root assist in several scripts. These are valid diagnostics only. They do not prove teacher/VAE/diffusion learned single-leg or walking control.

## 路由决定

### disable_for_success_claims
- resource_adjusted_teacher_rollout_vae
- resource_adjusted_state_latent_diffusion
- clean_walk_and_singleleg_video_chain

### preferred_local_diagnostic_route
- stage1_tracking_parameter_contract_gate
- paper_contract_teacher_rollout_vae
- paper_contract_state_latent_dataset
- paper_contract_transformer_state_latent_diffusion_code_contract
- paper_contract_state_latent_diffusion
- paper_contract_offline_guidance

### why_still_blocked
- Stage-1 teacher quality gate has not passed.
- The state-latent dataset still uses local policy_obs rather than the full paper hybrid state.
- The paper-style Transformer denoiser has only passed a tiny dry-run code-contract gate; it has not been fully trained or evaluated.
- Guidance is offline cost-gradient evaluation, not receding-horizon closed-loop MuJoCo/Isaac control.
- MuJoCo video/control adapter uses absolute joint targets, IK traces, root assist, and material differences; it is not yet native normalized-action control.
- The native action formula adapter is ready as a fixture, but native observation reconstruction and no-root-assist physics rollout are still missing.
- The native 160-D observation adapter remains blocked until it is numerically validated against IsaacLab observation_manager output and motion_tracking_controller frame-alignment semantics.
- Official G1 PPO uses empirical observation normalization; native MuJoCo inference must preserve the exported normalizer or checkpoint obs_norm_state_dict.
- Existing videos use blending/root assist or weak teacher actions and cannot be the final single-leg success folder.

## 当前训练状态

- 最新训练日志指标：`{'log_exists': True, 'iteration': 372, 'max_iterations': 3000, 'mean_reward': 0.36, 'mean_episode_length': 13.01, 'error_anchor_pos': 0.2134, 'error_body_pos': 0.243, 'error_joint_pos': 3.012, 'termination_anchor_pos': 111.0417, 'termination_ee_body_pos': 262.875, 'eta': '01:10:54'}`
- GPU 5/6 快照：`[{'index': 5, 'name': 'NVIDIA H20', 'memory_used_mb': 1, 'memory_total_mb': 97871, 'utilization_gpu_percent': 0, 'power_draw_w': 72.84}, {'index': 6, 'name': 'NVIDIA H20', 'memory_used_mb': 1, 'memory_total_mb': 97871, 'utilization_gpu_percent': 0, 'power_draw_w': 73.78}]`
- 是否达到 80GB/卡目标：`False`

## 下一步

- Do not use current clean_walk/hub_singleleg learned videos as success evidence.
- Use the paper-contract VAE route for future diagnostics, not the legacy obs+action resource VAE.
- Run full training/evaluation of the paper-contract Transformer diffusion route only after teacher quality improves.
- Implement or verify the native MuJoCo/Isaac action adapter before producing final videos: obs -> model -> normalized action -> theta0 + alpha * action -> PD -> physics step.
- Implement and validate the native 160-D MuJoCo observation adapter before any direct PPO/VAE/diffusion actor-video claim.
- Use the new action adapter formula fixture, but keep logging raw setpoints versus MuJoCo ctrlrange-clipped setpoints for ankle-roll joints.
- Train/evaluate a high-throughput Stage-1 teacher with official whole_body_tracking until done rate and posture metrics pass.
- Only after the teacher quality gate passes, collect continuous rollouts, train the corrected VAE/diffusion chain, then render one final success folder.
