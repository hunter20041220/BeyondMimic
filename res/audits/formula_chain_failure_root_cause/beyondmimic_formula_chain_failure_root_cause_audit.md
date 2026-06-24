# BeyondMimic 公式链路失败根因审计

- status: `blocked_formula_chain_has_required_fixes_before_training_or_success_video`
- claim level: `audit_only; no new training; no success video claim`
- downstream training allowed: `False`
- success video generation allowed: `False`
- stage1 corrective teacher training allowed: `True`

## 当前根因排序

1. **MuJoCo native observation adapter 的 anchor pose 项还没有与 IsaacLab 数值等价。**
   - 影响：policy 会看到假的位置/姿态误差，容易输出恢复姿态或前倾站姿，而不是抬腿/迈步。
2. **当前 teacher checkpoint non-timeout done rate 高、reward 低。**
   - 影响：VAE/diffusion 学到的是弱 teacher 行为，而不是完整走路或单脚站立姿态。
3. **VAE 训练数据来自弱 teacher rollout，且包含 reset/done 样本。**
   - 影响：低 reconstruction MSE 只证明拟合了弱 action，不证明学会了 reference motion。
4. **当前 state-latent artifact 使用 160-D policy_obs，而不是论文 hybrid state。**
   - 影响：diffusion 训练在错误 state 表示上，还可能把 reference tracking cue 泄漏进模型。
5. **当前 denoiser/guidance 证据不是论文 Transformer + closed-loop guidance 链。**
   - 影响：offline MSE/cost 改善不能证明 MuJoCo action-control 成功。

## 逐项门禁

### Stage-1 teacher/RL formula - `pass`

- 论文/官方要求：论文 Stage-1 使用官方 motion tracking MDP: 160-D policy observation, normalized PD action theta_sp=theta0+alpha*a, reward table S1, domain randomization table S2, PPO 30k iterations.
- 本地证据：官方 whole_body_tracking 源码中 observation/order、reward、domain randomization、PPO config 和 G1 action scale/armature 都能定位到。
- 诊断：公式/参数源头不是主要问题；第一阶段应该继续以官方 whole_body_tracking 为准。
- 必须修复：任何新 teacher 训练都必须沿用官方 MDP/reward/action scale，偏离项要单独记录为 ablation。
- blocks training: `False`
- blocks success video: `False`
- evidence: /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:66; /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:78; /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:631; /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:772; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:110; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py:184; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py:8

### Stage-1 teacher quality - `blocked`

- 论文/官方要求：后续 VAE/diffusion 只能使用高质量 motion-tracking teacher 的闭环 rollout；teacher 应能连续跟踪 motion，而不是频繁 reset 或只维持前倾站姿。
- 本地证据：multi-source best: reward_mean=0.024131401152315747, done_rate=19.41%, body_pos_err=1.0095036663737982, joint_pos_err=1.6739522380175; singleleg gate passed=False, singleleg_reward=0.04114155067647979, singleleg_done_rate=27.93%.
- 诊断：当前 teacher 质量不足是“VAE/diffusion 学成前倾站姿”的直接上游原因。如果 teacher 的动作本身没有稳定学到抬腿/走路，VAE 只能拟合弱 teacher 动作。
- 必须修复：先重新筛选/训练 Stage-1 teacher，要求 non-timeout done rate < 5%, reward_mean > 0.1, body/joint error 达到本地质量门槛，并保存连续 motion_time_steps 的 rollout。
- blocks training: `True`
- blocks success video: `True`
- evidence: /mnt/infini-data/test/BeyondMimic/res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/stage1_multisource_best_teacher.json; /mnt/infini-data/test/BeyondMimic/res/tracking/hub_singleleg_paper_contract_ppo_checkpoint_eval/tracking_hub_singleleg_paper_contract_ppo_checkpoint_eval.json

### MuJoCo observation adapter - `blocked`

- 论文/官方要求：IsaacLab PPO checkpoint 只能在完全一致的 observation semantics 下部署；尤其是 command、anchor pose error、base velocities、joint state、last_action 的 frame/order 必须逐项数值一致。
- 本地证据：native obs status=blocked_native_mujoco_observation_adapter_not_validated; walk runtime status=blocked_mujoco_injected_state_observation_runtime_parity_mismatch; cross-sample torso status=blocked_fixed_torso_offset_not_stable_across_walk_and_dance_samples.
- 诊断：walk 样本中 command/base/joint/action 切片已基本通过，但 anchor pos/orientation 仍不匹配；单样本 torso offset 可以拟合，但 walk+dance 跨样本 offset 不稳定。这会让 policy 以为自己一直有错误姿态，从而输出恢复/前倾动作。
- 必须修复：在 walk、dance、singleleg 至少三个 non-terminated IsaacLab 样本上，把 MuJoCo 160-D obs 每个 slice 与 IsaacLab observation_manager 对齐到容差内；不要用单个固定 offset hack。
- blocks training: `True`
- blocks success video: `True`
- evidence: /mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json; /mnt/infini-data/test/BeyondMimic/res/audits/mujoco_observation_runtime_parity_walk_sample/mujoco_observation_runtime_parity_audit.json; /mnt/infini-data/test/BeyondMimic/res/audits/mujoco_torso_frame_offset_cross_sample/mujoco_torso_frame_offset_cross_sample_audit.json

### MuJoCo action adapter / PD scale - `pass`

- 论文/官方要求：动作必须按论文 theta_sp=theta0+alpha*a 转换为 PD setpoint；alpha 来自官方 G1 effort/stiffness，且 MuJoCo actuator ctrlrange 不能静默裁剪 normalized action。
- 本地证据：native action status=ok_native_action_adapter_formula_and_no_clip_ctrlrange_patch_ready; checks={'controller_default_pose_available': True, 'deployment_no_action_clip_xml_written': True, 'does_not_claim_rollout_or_success': True, 'isaaclab_affine_joint_action_semantics_available': True, 'large_action_clips_to_unit_scale': True, 'mujoco_mapping_order_matches_action_rows': True, 'negative_unit_action_delta_matches_action_scale': True, 'official_action_scale_rows_29': True, 'paper_formula_available': True, 'patched_pd_actuator_order_matches_action_rows': True, 'pd_actuator_order_matches_action_rows': True, 'unit_action_delta_matches_action_scale': True, 'unit_targets_inside_mujoco_ctrlrange': True, 'zero_action_returns_default_pose': True, 'zero_default_fallback_not_used': True}
- 诊断：action adapter 当前不是主要 blocker；但任何视频仍必须同时通过 observation adapter。
- 必须修复：保留 no-clip ctrlrange XML/adapter；不要为了视觉效果改 action scale 或 PD 公式。
- blocks training: `False`
- blocks success video: `False`
- evidence: /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:78; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py:184; /mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json

### VAE formula and data quality - `blocked`

- 论文/官方要求：论文 VAE encoder 只吃 reference intent E(psi,e_anchor)，decoder 吃 latent+proprioception 并用 DAgger/teacher actions 训练；但训练数据必须来自高质量连续 teacher rollout。
- 本地证据：VAE status=ok_official_importer_export_paper_contract_teacher_rollout_vae_training; test_action_mse=0.008244317956268787; dataset_done_count=47200/306176 (15.42%). 当前 VAE loader 直接展平 shards，没有在 VAE 训练阶段过滤 done/reset 样本。
- 诊断：VAE 接口公式大体已经改成 paper-contract，但源 teacher rollout 弱且包含大量 done/reset；低 action MSE 只说明它能拟合弱 teacher action，不能说明学到了单脚站立或走路姿态。
- 必须修复：先修 teacher 和 MuJoCo obs adapter；重新采集连续、低 done-rate 的 teacher rollout；VAE 数据加载必须过滤 done、timeout、motion_time_steps 跳变，并按 motion/episode 切分。
- blocks training: `True`
- blocks success video: `True`
- evidence: /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:109; /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:160; /mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_paper_contract_teacher_rollout_vae_training/level_c_official_importer_export_paper_contract_teacher_rollout_vae_training.json; /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_contract_teacher_rollout_vae_training.py:162; /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_contract_teacher_rollout_vae_training.py:186

### State-latent diffusion dataset - `blocked`

- 论文/官方要求：论文 diffusion 数据是 state-latent trajectory: hybrid character/yaw-centric state + VAE latent；不应把 tracking policy 的 160-D observation 当作 state，也不应包含 reference command。
- 本地证据：state_source='policy_obs in local paper-contract best-teacher rollout shards'; token_dim=192; window_count=285696; expected_window_count=285696; split_counts={'test': 28569, 'train': 228557, 'validation': 28570}.
- 诊断：当前官方 importer/export paper-contract state-latent artifact 仍显示 state_source 是 policy_obs。这与论文第 S2/S3 的 hybrid state 设计不一致，而且 policy_obs 含 reference command/anchor error，会把 tracking cue 泄漏进 diffusion，而不是学习 task-agnostic human-like behavior distribution。
- 必须修复：重新生成 state-latent dataset：使用 raw world root/body state 构造 paper hybrid state；启用 OU noise rollout、5 秒稳定性验证、done/timeout/不连续窗口 rejection；输出 state_source 必须是 paper_hybrid 或 paper_projected，不得是 policy_obs。
- blocks training: `True`
- blocks success video: `True`
- evidence: /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:139; /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:534; /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:539; /mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_paper_contract_teacher_rollout_state_latent_dataset/level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.json; /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py:76; /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py:67

### Diffusion architecture/training - `blocked`

- 论文/官方要求：论文 diffusion 使用 Transformer Encoder，horizon=16, history=4, embed=512, heads=8, layers=6, 20 denoising steps, batch=512, 1000 epochs, cosine/warmup/EMA。
- 本地证据：current denoiser status=ok_official_importer_export_paper_contract_state_latent_diffusion_training; token_dim=192; test_pred_token_mse=0.037122479772993495; denoising_improvement=0.5111648986394071. 当前 official_importer_export result 包装的是 resource_adjusted MLP StateLatentDenoiser。
- 诊断：denoising MSE 改善是有价值的 debug 结果，但它是在错误/弱数据和 MLP denoiser 上得到的，不能证明论文 Transformer state-latent diffusion 复现成功。
- 必须修复：在通过 teacher+state-latent dataset gate 后，改用 paper Transformer 架构和训练 schedule；输出必须记录 512/8/6/20、batch/epoch/scheduler/EMA，并在验证集上评估。
- blocks training: `True`
- blocks success video: `True`
- evidence: /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:126; /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:830; /mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_paper_contract_state_latent_diffusion_training/level_c_official_importer_export_paper_contract_state_latent_diffusion_training.json; /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py:108; /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_contract_transformer_state_latent_diffusion_training.py:166

### Guidance and closed-loop control - `blocked`

- 论文/官方要求：论文 guidance 是 receding-horizon classifier guidance：每帧对 state-latent trajectory cost 求梯度，取当前 latent，经 VAE decoder 输出 action，再由物理仿真反馈下一帧。
- 本地证据：guidance status=ok_official_importer_export_paper_contract_state_latent_guidance_eval; checks={'all_best_costs_improve': True, 'all_best_guidance_gradients_nonzero': True, 'all_rows_finite': True, 'all_tasks_evaluated': True, 'does_not_claim_closed_loop_rollout': True, 'does_not_claim_fig5_fig6_reproduction': True, 'does_not_claim_paper_level_guidance': True, 'row_count_matches_tasks_splits_scales': True, 'scale_grid_includes_unguided': True, 'uses_resource_adjusted_diffusion_checkpoint': False, 'uses_resource_adjusted_state_latent_dataset': False, 'validation_and_test_splits_evaluated': True}; scope=Full-split offline task-cost guidance over the local paper-contract denoiser. This is not closed-loop IsaacLab/MuJoCo guided control..
- 诊断：当前 guidance 是 offline proxy over denoiser outputs，不是 MuJoCo/IsaacLab 闭环；没有证明 joystick/waypoint/inpainting/obstacle cost 能驱动物理机器人。
- 必须修复：等 teacher、VAE、paper hybrid diffusion 和 MuJoCo obs/action adapter 通过后，再实现 receding-horizon closed-loop guidance，保存每步 action、state、cost、fall/done。
- blocks training: `True`
- blocks success video: `True`
- evidence: /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:6; /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:548; /mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_paper_contract_state_latent_guidance_eval/level_c_official_importer_export_paper_contract_state_latent_guidance_eval.json

### Existing MuJoCo videos - `blocked`

- 论文/官方要求：成功视频必须是 controller 输出 action -> PD torque/setpoint -> MuJoCo physics step -> next state feedback，且不能靠 root assist/reference blend 伪装。
- 本地证据：walk claim=Clean 15 s local MuJoCo walk suite with pure local model targets. Root assist is still enabled, so this is not paper-level BeyondMimic evidence.; singleleg claim=Clean 15 s local MuJoCo walk suite with pure local model targets. Root assist is still enabled, so this is not paper-level BeyondMimic evidence.; walk checks={'all_mp4_exist': True, 'all_primary_variants_fall_proxy_zero': True, 'does_not_claim_paper_level': True, 'does_not_claim_paper_level_pure_policy_success': True, 'uses_reference_anchor_blend_for_learned_variants': False, 'video_duration_at_least_10s': True}; singleleg checks={'all_mp4_exist': True, 'all_primary_variants_fall_proxy_zero': True, 'does_not_claim_paper_level': True, 'does_not_claim_paper_level_pure_policy_success': True, 'uses_reference_anchor_blend_for_learned_variants': False, 'video_duration_at_least_10s': True}.
- 诊断：现有视频可作为 diagnostic/presentation asset，但不是 paper-level 或可信闭环模型链证据；当前不能把它们作为最终成功文件夹。
- 必须修复：保留失败记录和 summary，不再从这些目录挑成功；只有 adapter/teacher/model-chain 全部通过后，才生成新的 single-leg/walk 成功目录，并清理旧失败视频到 failed_runs 或 archive。
- blocks training: `False`
- blocks success video: `True`
- evidence: /mnt/infini-data/test/BeyondMimic/res/visualization/final_clean_walk_six_mujoco_videos_scaled_ppo_pure/clean_walk_mujoco_control_suite_summary.json; /mnt/infini-data/test/BeyondMimic/res/visualization/hub_singleleg_full_chain_scaled_ppo_pure/clean_walk_mujoco_control_suite_summary.json
