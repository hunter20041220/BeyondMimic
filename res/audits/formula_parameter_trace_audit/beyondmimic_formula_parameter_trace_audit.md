# BeyondMimic 公式与参数复现审计

- 状态：`blocked_formula_parameter_trace_has_required_fixes_before_training`
- 行数：`16`
- goal_complete：`False`

## 结论

官方 Stage 1 motion tracking 代码整体符合论文 tracking 公式和主要参数；但是当前本地 VAE/diffusion/guidance/MuJoCo 成功视频链并没有完成 paper-contract 对齐。新的长训练或成功视频应当等 teacher 质量 gate 和模型链 gate 修复后再启动。

当前不得声称完整复现 BeyondMimic，也不得把现有 teacher/VAE/diffusion/MuJoCo 视频写成 paper-level 成功结果。

## 阻塞项

- Stage 1 adaptive sampling and reset
- Old resource-adjusted VAE
- New paper-contract VAE script
- Old resource-adjusted diffusion
- Public LAFAN1 paper-architecture diffusion
- State projection, OU noise, symmetry augmentation
- Classifier guidance and task costs
- MuJoCo action-control video adapter
- Current single-leg teacher quality

## 对照矩阵

### Stage 1 observation and normalized PD action

- 论文要求：o=[psi,e_anchor,V_imu,theta-theta0,theta_dot,a_last]; theta_sp=theta0+alpha*a.
- 本地状态：`aligned_to_official_stage1_code`
- 训练 gate：`pass_for_formula_audit`
- 对当前坏视频的影响：This part is unlikely to be the source of the generic leaning-pose failure if official obs order and exported normalizer are preserved.
- 成功声明前必须修复：Keep using official observation order/export metadata. Do not hand-build a 160-D MuJoCo obs without metadata validation.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:66`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:78`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:118`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:129`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py:14`

### Stage 1 anchor-relative tracking transform

- 论文要求：Non-anchor desired body pose is yaw-aligned, height-preserving, and translated under current anchor.
- 本地状态：`aligned_to_official_stage1_code`
- 训练 gate：`pass_for_formula_audit`
- 对当前坏视频的影响：A broken anchor transform would cause drifting/jumping references; official code implements the paper transform.
- 成功声明前必须修复：For MuJoCo replay/video, use the same anchor convention or label as diagnostic only.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:376`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:290`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:291`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:294`

### Stage 1 reward formulation

- 论文要求：Gaussian body pos/orientation/linear velocity/angular velocity rewards with std 0.3/0.4/1.0/3.14 plus action-rate, joint-limit, contact penalties.
- 本地状态：`aligned_to_official_stage1_code`
- 训练 gate：`pass_for_formula_audit`
- 对当前坏视频的影响：Weak teacher reward is more likely undertraining/data/asset/runtime quality than a different reward formula.
- 成功声明前必须修复：Use official rewards for next teacher; compare reward components and termination counts per checkpoint before downstream VAE.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:642`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:212`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:227`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:232`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py:32`

### Stage 1 termination

- 论文要求：Terminate when anchor or EE z-position error >0.25 m, or anchor orientation error >0.8 rad.
- 本地状态：`mostly_aligned_with_one_public_code_detail`
- 训练 gate：`pass_with_caution`
- 对当前坏视频的影响：High termination counts explain bad teacher rollouts and VAE collapse. The public code checks projected gravity for anchor orientation, not a full log-map norm.
- 成功声明前必须修复：Gate teacher checkpoints by done rate/episode length/motion-time continuity before collecting VAE data.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:444`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:258`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:266`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py:28`
- 备注：Public code detail should be reported instead of silently treated as exact supplementary-text implementation.

### Stage 1 PD, action scale, and armature

- 论文要求：kp=I*w^2, kd=2*I*zeta*w, w=10 Hz, zeta=2, alpha=0.25*tau_max/kp; dual ankles/waist use doubled armature.
- 本地状态：`aligned_to_official_stage1_code`
- 训练 gate：`pass_for_formula_audit`
- 对当前坏视频的影响：If MuJoCo PD uses different kp/kd/action scale/joint order, learned actions can look like leaning or tiny steps even with a good policy.
- 成功声明前必须修复：Before final MuJoCo videos, export or derive exact joint order, kp, kd, armature, default pose, and action scale from official metadata/checkpoint.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:406`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:409`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py:12`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py:13`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py:184`

### Stage 1 domain randomization

- 论文要求：Friction/restitution, default joint offsets, torso COM, and velocity perturbations; text mentions larger ankle offset.
- 本地状态：`partial_public_code_difference`
- 训练 gate：`pass_with_caution`
- 对当前坏视频的影响：This is not the main reason the current videos are generic, but it matters for final paper-contract training.
- 成功声明前必须修复：Decide explicitly whether to follow public code exactly or patch ankle default offsets to the supplementary text, then document the choice.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:415`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:420`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:163`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:175`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:180`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:190`

### Stage 1 adaptive sampling and reset

- 论文要求：Failure-rate bins, EMA alpha=0.001, uniform floor 0.1/S, non-causal kernel rho=0.8 over u={0,1,2}, reset to reference state with perturbations.
- 本地状态：`partial_public_code_difference`
- 训练 gate：`block_for_paper_exact_claim_only`
- 对当前坏视频的影响：Poor sampling can slow learning of hard segments such as leg lift/single-leg balance.
- 成功声明前必须修复：For hard single-leg training, either patch kernel size to 3 per paper or record that public code uses kernel size 1.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:448`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:451`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:371`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:368`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:268`

### PPO hyperparameters

- 论文要求：Actor/critic [512,256,128], ELU, 24 steps/env, 30000 iterations, lr 1e-3, clip 0.2, entropy 0.005, gamma 0.99, GAE 0.95, desired KL 0.01.
- 本地状态：`aligned_to_official_stage1_code`
- 训练 gate：`pass_for_formula_audit`
- 对当前坏视频的影响：Earlier short/low-memory candidates are not enough to learn hard motions; bad videos should not be read as final PPO failure.
- 成功声明前必须修复：Train only after formula gates pass; for full runs use checkpoint/eval gates, not last checkpoint by default.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:780`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py:7`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py:8`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py:14`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py:25`

### Old resource-adjusted VAE

- 论文要求：Paper VAE encoder is E(psi,e_anchor); decoder is D(z,[g,V_imu,theta,theta_dot,a_last]); modified ELBO beta=0.01 with DAgger.
- 本地状态：`mismatch`
- 训练 gate：`block_training_chain`
- 对当前坏视频的影响：This is a direct mechanism for generic/averaged action outputs and leaning posture.
- 成功声明前必须修复：Do not use old resource-adjusted VAE for success videos. Use/retrain paper-contract VAE only on high-quality continuous teacher rollouts.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:109`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:160`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py:100`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py:106`

### New paper-contract VAE script

- 论文要求：Formula-level split E(reference intent) and D(latent, proprioception), latent dim 32, hidden [2048,1024,512], lr 5e-4, beta 0.01.
- 本地状态：`formula_input_contract_repaired_but_data_gate_failed`
- 训练 gate：`block_downstream_until_teacher_quality_passes`
- 对当前坏视频的影响：Even a corrected VAE will reproduce a weak/reset-heavy teacher and can still lean instead of lifting the leg.
- 成功声明前必须修复：Collect continuous, low-done-rate teacher rollout data before retraining this VAE for videos.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_contract_teacher_rollout_vae_training.py`
  - `/mnt/infini-data/test/BeyondMimic/res/level_c/paper_contract_teacher_rollout_vae_training/level_c_paper_contract_teacher_rollout_vae_training.json`
  - `source_teacher_done_rate_low_enough_for_downstream=False`

### Old resource-adjusted diffusion

- 论文要求：Paper diffusion models tau=[s,z,...] with hybrid yaw-centric state, individual state/latent denoising steps, Transformer denoiser.
- 本地状态：`mismatch`
- 训练 gate：`block_training_chain`
- 对当前坏视频的影响：MLP over policy_obs+latent cannot be claimed as the paper state-latent diffusion and will not reliably produce leg-lift semantics.
- 成功声明前必须修复：Retire this path from success videos; keep it only as a debug baseline.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:171`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py:144`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py:103`

### Public LAFAN1 paper-architecture diffusion

- 论文要求：Transformer encoder, 512 dim, 8 heads, 6 layers, horizon 16, history 4, 20 denoising steps, batch 512, epochs 1000, weight decay 0.001, EMA.
- 本地状态：`architecture_partial_data_mismatch`
- 训练 gate：`block_paper_level_claim_but_useful_for_code_probes`
- 对当前坏视频的影响：The architecture probes are useful, but the dataset and VAE encoder still differ from the paper's teacher-rollout pipeline.
- 成功声明前必须修复：Rewire diffusion training to consume paper-contract VAE latents from good teacher rollouts, then use receding-horizon closed-loop inference.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:840`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/train_lafan1_paper_level_vae_diffusion.py:121`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/train_lafan1_paper_level_vae_diffusion.py:126`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/train_lafan1_paper_level_vae_diffusion.py:127`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/train_lafan1_paper_level_vae_diffusion.py:48`

### State projection, OU noise, symmetry augmentation

- 论文要求：Use yaw-centric state, emphasis projection c=6, OU action noise sigma=0.1/theta=0.8, and sagittal symmetry augmentation.
- 本地状态：`partially_implemented_as_helpers_and_public_data_audits`
- 训练 gate：`block_final_chain_until_integrated`
- 对当前坏视频的影响：Without the VAE rollout error band and symmetry in the actual chain, diffusion may overfit/static-average.
- 成功声明前必须修复：Integrate these helpers into the teacher-rollout to VAE/diffusion pipeline, not only standalone audits.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:527`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:544`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:545`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/src/beyondmimic_reimpl/state.py:10`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/src/beyondmimic_reimpl/sampling.py:41`

### Classifier guidance and task costs

- 论文要求：Guidance must apply -grad_tau G(tau) inside reverse denoising for joystick, waypoint, SDF obstacle, inpainting.
- 本地状态：`offline_audits_exist_but_current_video_chain_mismatch`
- 训练 gate：`block_success_video_claim`
- 对当前坏视频的影响：The visible guided videos were driven by latent interpolation/blending, not paper classifier guidance; they cannot prove task control.
- 成功声明前必须修复：Implement closed-loop receding-horizon guidance with differentiable task costs and record gradient norms/cost decrease.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:224`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:552`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:558`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:566`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/render_clean_walk_mujoco_control_suite.py:335`

### MuJoCo action-control video adapter

- 论文要求：Final video should execute policy/VAE/diffusion action -> theta_sp -> PD torque -> MuJoCo step, with no direct pose playback or root assist for success claims.
- 本地状态：`diagnostic_only_currently`
- 训练 gate：`block_success_video_claim`
- 对当前坏视频的影响：Root assist and reference-anchor blending can make videos readable while hiding policy failure; pure learned variants still look poor.
- 成功声明前必须修复：For final success folder, disable root assist/blending, center camera only visually, and require continuous motion-time and low fall metrics.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_pd_control_video.py:266`
  - `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_pd_control_video.py:440`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/render_clean_walk_mujoco_control_suite.py:14`
  - `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/render_clean_walk_mujoco_control_suite.py:492`

### Current single-leg teacher quality

- 论文要求：A teacher used for VAE/diffusion must track the target motion continuously, with low done rate and meaningful leg-lift/body errors.
- 本地状态：`failed_or_unproven`
- 训练 gate：`block_downstream_until_retrained_or_good_checkpoint_found`
- 对当前坏视频的影响：This directly explains why teacher/VAE/diffusion did not learn single-leg posture and instead leaned.
- 成功声明前必须修复：Find or train a teacher that passes checkpoint eval before collecting VAE/diffusion data.
- 证据：
  - `/mnt/infini-data/test/BeyondMimic/res/failed_runs/hub_singleleg_low_memory_candidate_stop_audit/hub_singleleg_low_memory_candidate_stop_audit.json`
  - `singleleg_stop_status=stopped_low_memory_candidate_not_downstream_source`
  - `model_chain_status=blocked_model_chain_not_paper_contract_and_teacher_quality_not_ready`
