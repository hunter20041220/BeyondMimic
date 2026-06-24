# BeyondMimic Appendix Parameter Matrix Audit

- Status: `blocked_appendix_parameter_matrix_has_required_fixes`
- Claim level: `audit_only; no training; no video success claim`
- Rows: `14`
- Status counts: `{"blocked": 8, "partial": 3, "pass": 3}`

## 结论

当前不允许从弱 teacher 继续做下游 VAE/diffusion/guidance 长训练，也不允许把现有 MuJoCo 视频当成功视频。
下一步应先修复/验证 native observation/action adapter、teacher quality gate、hybrid state-latent 数据生成，以及 appendix 中公开代码差异项。

## Permission

- `start_new_long_stage1_training`: `False`
- `start_downstream_vae_training`: `False`
- `start_diffusion_training`: `False`
- `start_guided_closed_loop_video_generation`: `False`
- `create_final_success_video_folder`: `False`
- `allowed_next_work`: `['numeric IsaacLab-vs-MuJoCo observation parity probe', 'MuJoCo action ctrlrange/action-scale repair or documented deployment-compatible model', 'teacher checkpoint quality selection with done/error/continuity gates', 'paper-contract hybrid state-latent dataset regeneration after teacher gate']`

## Blocking Rows

### Stage-1 RL / Normalized PD action and no kinematic clipping
- Status: `blocked`
- Expected: theta_sp=theta0+alpha*a; setpoints intentionally not clipped by joint kinematic limits.
- Observed: official_action_scale_formula=True; native_formula_ready=True; native_ctrlrange_allows_unit_targets=False.
- Impact: Ctrlrange clipping can shrink or distort ankle/leg actions, producing tiny steps or leaning poses.
- Required fix: Resolve MuJoCo ctrlrange/action-scale compatibility before trusting PPO/VAE/diffusion action videos.
- Claim boundary: Approximate or clipped action videos remain diagnostic.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:80; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py:184; /mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json`

### Stage-1 RL / Termination thresholds
- Status: `partial`
- Expected: Anchor/EE z error threshold 0.25 m and anchor orientation threshold 0.8 rad.
- Observed: Official public code uses z-only anchor/EE checks and projected-gravity anchor orientation threshold.
- Impact: High non-timeout done rates directly poison teacher rollout, VAE, and diffusion training.
- Required fix: Gate every teacher checkpoint by done rate, episode length, body error, joint error, and continuous motion_time_steps.
- Claim boundary: Public-code detail is acceptable if documented, but current reset-heavy rollouts cannot seed downstream training.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:444; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:258; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:266; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py:34`

### Stage-1 RL / PD gains, armature, and action scale
- Status: `partial`
- Expected: omega=10 Hz, zeta=2, kp=I*omega^2, kd=2*I*zeta*omega, alpha=0.25*tau_max/kp, dual ankles/waist doubled.
- Observed: Official G1 code implements armature constants, stiffness, damping, and action scale; MuJoCo final video gate is still not passed.
- Impact: A mismatch in armature/kp/kd/action scale can convert real actions into unstable or underpowered motions.
- Required fix: Export exact joint order, default pose, armature, kp, kd, action scale, ctrlrange and material into the MuJoCo runner manifest.
- Claim boundary: Official source pass is not the same as native MuJoCo deployment pass.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:406; /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:407; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py:12; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py:105; /mnt/infini-data/test/BeyondMimic/res/audits/mujoco_control_contract_audit/mujoco_control_contract_audit.json`

### Stage-1 RL / Domain randomization table S2
- Status: `partial`
- Expected: Static friction U[0.3,1.6], dynamic U[0.3,1.2], restitution U[0,0.5], joint default offsets, torso COM, random root velocity pushes.
- Observed: Official code matches friction/restitution/COM/push and general joint offset; supplementary text also mentions larger ankle offsets not visible in the public config.
- Impact: This affects robustness, but it does not excuse treating failed teacher/video outputs as success.
- Required fix: Record whether the next teacher follows public code exactly or patches ankle offsets to the supplementary text.
- Claim boundary: A public-code run can be described as public-code-faithful, not necessarily exact supplementary-private-config reproduction.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:701; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:163; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:175; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:180; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:190`

### Stage-1 RL / Adaptive sampling
- Status: `blocked`
- Expected: EMA alpha 0.001, uniform floor 0.1/S, rho=0.8, non-causal look-back u={0,1,2}.
- Observed: Official public MotionCommandCfg has alpha/rho/floor, but default adaptive_kernel_size is 1 unless explicitly overridden.
- Impact: Hard segments such as single-leg stance can remain under-sampled if pre-failure bins are not emphasized.
- Required fix: For paper-exact hard-motion training, set/verify kernel size 3 or explicitly label the run public-default.
- Claim boundary: Do not silently call kernel_size=1 an exact appendix adaptive-sampling reproduction.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:451; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:371; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:368`

### VAE / Conditional VAE input and architecture table S5
- Status: `blocked`
- Expected: Encoder E(psi,e_anchor), decoder D(z,[g,V_imu,theta,theta_dot,a_last]), latent dim 32, hidden [2048,1024,512], lr 5e-4, grad accum 15, KL 0.01.
- Observed: paper_contract_vae_interface=True; source_teacher_quality_ok=False.
- Impact: The old obs+action VAE can learn an averaged standing action. The corrected VAE still fails if trained on weak teacher data.
- Required fix: Only retrain paper-contract VAE after a continuous, low-fall teacher rollout dataset passes quality gates.
- Claim boundary: VAE MSE alone is not closed-loop success.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:153; /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:804; /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_contract_teacher_rollout_vae_training.py:162; /mnt/infini-data/test/BeyondMimic/res/level_c/paper_contract_teacher_rollout_vae_training/level_c_paper_contract_teacher_rollout_vae_training.json`

### Diffusion / State-latent representation and data collection
- Status: `blocked`
- Expected: Hybrid character-yaw-centric state + latent, individual state/latent denoising steps, VAE rollouts with OU action noise, 5s rejection, sagittal symmetry augmentation.
- Observed: hybrid_builder=True; existing_dataset_fresh=False; teacher_raw_state=False; window_filter=False; ou_noise_code=False; five_sec_rejection=False; symmetry_aug=False.
- Impact: Training diffusion on old policy_obs/latent windows or reset-heavy data can produce plausible MSE but poor actions.
- Required fix: Regenerate teacher rollout shards with raw state, then rebuild reset-safe hybrid state-latent windows with OU and symmetry manifests.
- Claim boundary: Current denoising MSE is diagnostic, not a paper-level control result.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:139; /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:539; /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:591; /mnt/infini-data/test/BeyondMimic/res/audits/state_latent_dataset_source_contract/beyondmimic_state_latent_dataset_source_contract_audit.json`

### Diffusion / Transformer architecture table S6
- Status: `blocked`
- Expected: Horizon 16, history 4, embed 512, heads 8, layers 6, denoising steps 20, batch 512, epochs 1000, lr 1e-4, weight decay 0.001, cosine, warmup 10000, EMA 0.75/0.9999.
- Observed: transformer_arch_code_present=True; full_training_over_accepted_dataset=False.
- Impact: A dry-run or MLP denoiser cannot explain or reproduce the paper's guided closed-loop behavior.
- Required fix: After teacher/VAE/data gates pass, train the paper-contract Transformer on held-out splits and record full metrics.
- Claim boundary: Architecture code presence is not trained-model evidence.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:830; /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_contract_transformer_state_latent_diffusion_training.py:166; /mnt/infini-data/test/BeyondMimic/res/level_c/paper_contract_transformer_state_latent_diffusion_training/paper_contract_transformer_state_latent_diffusion_training.json`

### Guidance / Classifier guidance and task costs
- Status: `blocked`
- Expected: Use -grad G(tau) during denoising in a receding-horizon closed-loop controller; SDF relaxed barrier for obstacle tasks.
- Observed: offline_guidance_gradients_ok=True; native_obs_ready=False; sdf_barrier_formula=True.
- Impact: Offline guidance can improve a token cost while the robot still leans or falls in physics.
- Required fix: Implement and validate receding-horizon MuJoCo/Isaac closed-loop guidance only after diffusion and native adapter gates pass.
- Claim boundary: Offline guidance is not Fig.5/Fig.6 reproduction.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:219; /mnt/infini-data/test/BeyondMimic/reproduction/src/beyondmimic_reimpl/guidance/costs.py:35; /mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_paper_contract_state_latent_guidance_eval/level_c_official_importer_export_paper_contract_state_latent_guidance_eval.json; /mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json`

### Deployment / Native MuJoCo observation/action/material gate
- Status: `blocked`
- Expected: Closed-loop video must use controller actions through PD physics, no root assist/blending, valid material/contact and verified observation semantics.
- Observed: native_obs_ready=False; native_action_range_ready=False; no_root_assist=False; material_ok=False.
- Impact: This is the most direct explanation for why existing MuJoCo action-control videos are not trustworthy.
- Required fix: Do not create the final success folder until this native deployment gate passes without root assist.
- Claim boundary: Reference replay can be useful visualization; it is not policy/diffusion control.
- Evidence: `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json; /mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json; /mnt/infini-data/test/BeyondMimic/res/audits/mujoco_control_contract_audit/mujoco_control_contract_audit.json`

### Training permission / No long downstream training from weak teacher
- Status: `blocked`
- Expected: Teacher quality and native adapter gates must pass before VAE/diffusion/guidance long training or success videos.
- Observed: teacher_quality_failed=True.
- Impact: Starting downstream training now would spend GPU time learning failed teacher/controller behavior.
- Required fix: Allowed next work: code/parameter fixes, numeric adapter validation, and corrective Stage-1 teacher evaluation/training under recorded contracts.
- Claim boundary: Current project remains incomplete.
- Evidence: `/mnt/infini-data/test/BeyondMimic/res/audits/pretraining_hard_gate/beyondmimic_pretraining_hard_gate_audit.json; /mnt/infini-data/test/BeyondMimic/res/audits/model_chain_paper_contract_audit/beyondmimic_model_chain_paper_contract_audit.json`

## Full Matrix

### Stage-1 RL / Policy observation vector
- Status: `pass`
- Expected: Policy input o=[psi,e_anchor,V_imu,theta-theta0,theta_dot,a_last], no temporal stacking.
- Observed: Official tracking config preserves command, anchor pos/orientation, base velocities, joint state, and last action.
- Required fix: Before MuJoCo PPO/video claims, compare each MuJoCo observation term numerically with IsaacLab observation_manager.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:66; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:114; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py:60`

### Stage-1 RL / Normalized PD action and no kinematic clipping
- Status: `blocked`
- Expected: theta_sp=theta0+alpha*a; setpoints intentionally not clipped by joint kinematic limits.
- Observed: official_action_scale_formula=True; native_formula_ready=True; native_ctrlrange_allows_unit_targets=False.
- Required fix: Resolve MuJoCo ctrlrange/action-scale compatibility before trusting PPO/VAE/diffusion action videos.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:80; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py:184; /mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json`

### Stage-1 RL / Reward table S1
- Status: `pass`
- Expected: Body pos/orientation/linear velocity/angular velocity stds 0.3/0.4/1.0/3.14; weights 1; optional anchor weight 0.5; action/joint/contact penalties -0.1/-10/-0.1.
- Observed: Official tracking config contains these reward terms, stds, and weights.
- Required fix: Keep reward components unchanged for paper-contract teacher training unless a row records an ablation.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:631; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:199; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py:32`

### Stage-1 RL / Termination thresholds
- Status: `partial`
- Expected: Anchor/EE z error threshold 0.25 m and anchor orientation threshold 0.8 rad.
- Observed: Official public code uses z-only anchor/EE checks and projected-gravity anchor orientation threshold.
- Required fix: Gate every teacher checkpoint by done rate, episode length, body error, joint error, and continuous motion_time_steps.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:444; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:258; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:266; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py:34`

### Stage-1 RL / PD gains, armature, and action scale
- Status: `partial`
- Expected: omega=10 Hz, zeta=2, kp=I*omega^2, kd=2*I*zeta*omega, alpha=0.25*tau_max/kp, dual ankles/waist doubled.
- Observed: Official G1 code implements armature constants, stiffness, damping, and action scale; MuJoCo final video gate is still not passed.
- Required fix: Export exact joint order, default pose, armature, kp, kd, action scale, ctrlrange and material into the MuJoCo runner manifest.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:406; /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:407; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py:12; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py:105; /mnt/infini-data/test/BeyondMimic/res/audits/mujoco_control_contract_audit/mujoco_control_contract_audit.json`

### Stage-1 RL / Domain randomization table S2
- Status: `partial`
- Expected: Static friction U[0.3,1.6], dynamic U[0.3,1.2], restitution U[0,0.5], joint default offsets, torso COM, random root velocity pushes.
- Observed: Official code matches friction/restitution/COM/push and general joint offset; supplementary text also mentions larger ankle offsets not visible in the public config.
- Required fix: Record whether the next teacher follows public code exactly or patches ankle offsets to the supplementary text.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:701; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:163; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:175; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:180; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:190`

### Stage-1 RL / Adaptive sampling
- Status: `blocked`
- Expected: EMA alpha 0.001, uniform floor 0.1/S, rho=0.8, non-causal look-back u={0,1,2}.
- Observed: Official public MotionCommandCfg has alpha/rho/floor, but default adaptive_kernel_size is 1 unless explicitly overridden.
- Required fix: For paper-exact hard-motion training, set/verify kernel size 3 or explicitly label the run public-default.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:451; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:371; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:368`

### Stage-1 RL / PPO hyperparameters table S4
- Status: `pass`
- Expected: Actor/critic [512,256,128], ELU, 24 steps/env, 30000 iterations, lr 1e-3, clip 0.2, entropy 0.005, gamma 0.99, GAE 0.95, KL 0.01, epochs 5, minibatches 4.
- Observed: Official RSL-RL config matches table S4.
- Required fix: If batch/env count is increased for GPU utilization, record the exact deviation and preserve all other paper constants.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:772; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py:7; /mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py:8`

### VAE / Conditional VAE input and architecture table S5
- Status: `blocked`
- Expected: Encoder E(psi,e_anchor), decoder D(z,[g,V_imu,theta,theta_dot,a_last]), latent dim 32, hidden [2048,1024,512], lr 5e-4, grad accum 15, KL 0.01.
- Observed: paper_contract_vae_interface=True; source_teacher_quality_ok=False.
- Required fix: Only retrain paper-contract VAE after a continuous, low-fall teacher rollout dataset passes quality gates.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:153; /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:804; /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_contract_teacher_rollout_vae_training.py:162; /mnt/infini-data/test/BeyondMimic/res/level_c/paper_contract_teacher_rollout_vae_training/level_c_paper_contract_teacher_rollout_vae_training.json`

### Diffusion / State-latent representation and data collection
- Status: `blocked`
- Expected: Hybrid character-yaw-centric state + latent, individual state/latent denoising steps, VAE rollouts with OU action noise, 5s rejection, sagittal symmetry augmentation.
- Observed: hybrid_builder=True; existing_dataset_fresh=False; teacher_raw_state=False; window_filter=False; ou_noise_code=False; five_sec_rejection=False; symmetry_aug=False.
- Required fix: Regenerate teacher rollout shards with raw state, then rebuild reset-safe hybrid state-latent windows with OU and symmetry manifests.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:139; /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:539; /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:591; /mnt/infini-data/test/BeyondMimic/res/audits/state_latent_dataset_source_contract/beyondmimic_state_latent_dataset_source_contract_audit.json`

### Diffusion / Transformer architecture table S6
- Status: `blocked`
- Expected: Horizon 16, history 4, embed 512, heads 8, layers 6, denoising steps 20, batch 512, epochs 1000, lr 1e-4, weight decay 0.001, cosine, warmup 10000, EMA 0.75/0.9999.
- Observed: transformer_arch_code_present=True; full_training_over_accepted_dataset=False.
- Required fix: After teacher/VAE/data gates pass, train the paper-contract Transformer on held-out splits and record full metrics.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:830; /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_contract_transformer_state_latent_diffusion_training.py:166; /mnt/infini-data/test/BeyondMimic/res/level_c/paper_contract_transformer_state_latent_diffusion_training/paper_contract_transformer_state_latent_diffusion_training.json`

### Guidance / Classifier guidance and task costs
- Status: `blocked`
- Expected: Use -grad G(tau) during denoising in a receding-horizon closed-loop controller; SDF relaxed barrier for obstacle tasks.
- Observed: offline_guidance_gradients_ok=True; native_obs_ready=False; sdf_barrier_formula=True.
- Required fix: Implement and validate receding-horizon MuJoCo/Isaac closed-loop guidance only after diffusion and native adapter gates pass.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:219; /mnt/infini-data/test/BeyondMimic/reproduction/src/beyondmimic_reimpl/guidance/costs.py:35; /mnt/infini-data/test/BeyondMimic/res/level_c/official_importer_export_paper_contract_state_latent_guidance_eval/level_c_official_importer_export_paper_contract_state_latent_guidance_eval.json; /mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json`

### Deployment / Native MuJoCo observation/action/material gate
- Status: `blocked`
- Expected: Closed-loop video must use controller actions through PD physics, no root assist/blending, valid material/contact and verified observation semantics.
- Observed: native_obs_ready=False; native_action_range_ready=False; no_root_assist=False; material_ok=False.
- Required fix: Do not create the final success folder until this native deployment gate passes without root assist.
- Evidence: `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json; /mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json; /mnt/infini-data/test/BeyondMimic/res/audits/mujoco_control_contract_audit/mujoco_control_contract_audit.json`

### Training permission / No long downstream training from weak teacher
- Status: `blocked`
- Expected: Teacher quality and native adapter gates must pass before VAE/diffusion/guidance long training or success videos.
- Observed: teacher_quality_failed=True.
- Required fix: Allowed next work: code/parameter fixes, numeric adapter validation, and corrective Stage-1 teacher evaluation/training under recorded contracts.
- Evidence: `/mnt/infini-data/test/BeyondMimic/res/audits/pretraining_hard_gate/beyondmimic_pretraining_hard_gate_audit.json; /mnt/infini-data/test/BeyondMimic/res/audits/model_chain_paper_contract_audit/beyondmimic_model_chain_paper_contract_audit.json`
