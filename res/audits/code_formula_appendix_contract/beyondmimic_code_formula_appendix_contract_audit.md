# BeyondMimic Code/Formula/Appendix Contract Audit

- Status: `blocked_code_formula_appendix_contract_has_required_fixes_before_training`
- Row count: `17`
- Status counts: `{"blocked": 5, "mismatch": 1, "partial": 1, "pass": 10}`
- Training permission: `{"allowed_next_work": ["regenerate teacher rollout shards with raw root/body state", "rebuild hybrid state-latent dataset with reset-safe windows", "validate MuJoCo native observation/action adapters without root assist", "run short code-level probes after fixes before long training"], "create_final_singleleg_success_folder": false, "start_downstream_vae_training": false, "start_guided_closed_loop_video_generation": false, "start_new_long_stage1_teacher_training": false, "start_state_latent_diffusion_training": false}`

## Required Fixes Before Long Training
- Before long teacher training, either set/verify adaptive_kernel_size=3 for paper-faithful runs or record the official default as a source-code discrepancy.
- Collect new teacher rollout shards with raw world-state fields, then rebuild hybrid state-latent windows with done/reset rejection.
- Add OU noise collection, 5 s stability rejection, and symmetry augmentation manifests before long diffusion training.
- Implement receding-horizon MuJoCo control where diffusion generates latent, VAE decodes action, and physics feeds back state.
- Validate MuJoCo obs term-by-term against IsaacLab/motion_tracking_controller before long policy/video claims.
- Fix floor/contact/material and produce no-root-assist native videos before success-folder cleanup.
- Do not start long downstream runs until the mismatches above are fixed and teacher quality is re-evaluated.

## Rows

### Stage-1 teacher/RL / Observation and PD-action contract
- Status: `pass`
- Expected: o=[psi,e_anchor,V_imu,theta-theta0,theta_dot,a_last], theta_sp=theta0+alpha*a.
- Observed: Official whole_body_tracking code exposes generated command, anchor error, base velocities, joints, and last action; G1 action scale uses 0.25*tau_max/kp.
- Required fix: Keep using official whole_body_tracking for Stage-1; do not train a custom 160-D obs policy unless it matches this contract.
- Claim boundary: Pass here only proves the official source contract exists, not that the current teacher checkpoint is good.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:not_found:\theta^{\text{sp}}; /mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py:not_found:generated_commands; /mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py:184`

### Stage-1 teacher/RL / Reward terms and weights
- Status: `pass`
- Expected: Gaussian body pose/orientation/velocity rewards plus anchor/regularization terms from appendix table.
- Observed: Official tracking config/reward code contains body_pos, body_orientation, body_lin_vel, body_ang_vel and Gaussian exp(-mean(square(error))/sigma^2) terms.
- Required fix: Train only with the official reward config unless a row records an intentional ablation.
- Claim boundary: Reward-code alignment does not make released-data or MuJoCo videos paper-level.
- Evidence: `/mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py:140; /mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py:not_found:def body_pos`

### Stage-1 teacher/RL / PPO hyperparameters
- Status: `pass`
- Expected: 50 Hz policy, 24 steps/env, 30000 iterations, actor/critic [512,256,128], ELU, empirical normalization.
- Observed: Official RSL-RL config matches these table parameters.
- Required fix: Use the official PPO config for any new teacher run; if GPU batch is increased, record the deviation explicitly.
- Claim boundary: PPO config matching is not teacher quality evidence.
- Evidence: `/mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py:7; /mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py:8`

### PD/action scale/armature/material / PD gains, action scale, armature
- Status: `pass`
- Expected: omega=10 Hz, damping ratio 2, kp=I*omega^2, kd=2*zeta*I*omega, alpha=0.25*tau_max/kp.
- Observed: Official G1 code implements natural frequency, damping ratio, armature-derived stiffness/damping, and action scale.
- Required fix: Propagate these constants into MuJoCo XML/adapters before judging control quality.
- Claim boundary: Matching constants in source do not prove the local MuJoCo XML/material/contact stack is identical.
- Evidence: `/mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py:12; /mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py:13`

### Stage-1 teacher/RL / Adaptive sampling kernel
- Status: `partial`
- Expected: Paper text states pre-failure look-back u={0,1,2} with rho^u weighting.
- Observed: Official MotionCommandCfg default currently exposes adaptive_kernel_size=1 unless an external config overrides it.
- Required fix: Before long teacher training, either set/verify adaptive_kernel_size=3 for paper-faithful runs or record the official default as a source-code discrepancy.
- Claim boundary: Do not silently call a kernel_size=1 run an exact appendix reproduction of adaptive sampling.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/root.tex:451; /mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py:84`

### VAE / Encoder/decoder input contract
- Status: `pass`
- Expected: Encoder E(psi,e_anchor), decoder D(z, proprioception), latent dim 32, KL 0.01, lr 5e-4.
- Observed: Local paper-contract VAE script uses command+anchor error encoder, proprio+latent decoder, latent dim 32, KL 0.01, lr 5e-4.
- Required fix: Keep this interface; do not train the older obs+action VAE as if it were paper-faithful.
- Claim boundary: Interface alignment does not mean the VAE is trained on official DAgger rollouts.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:153; /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_contract_teacher_rollout_vae_training.py:162`

### VAE / Network architecture and gradient accumulation
- Status: `pass`
- Expected: Appendix table gives encoder/decoder hidden dims [2048,1024,512] and accumulated gradient steps 15.
- Observed: Local paper-contract VAE now exposes HIDDEN_DIMS=[2048,1024,512] and GRAD_ACCUM_STEPS=15.
- Required fix: Keep this as a regression gate; do not reuse older obs+action VAE checkpoints for paper-facing videos.
- Claim boundary: Architecture alignment does not prove official DAgger data or paper-level rollout quality.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_contract_teacher_rollout_vae_training.py:47; /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_contract_teacher_rollout_vae_training.py:306`

### Diffusion / Transformer denoiser architecture
- Status: `pass`
- Expected: Horizon 16, history 4, sequence length 21, embed 512, heads 8, layers 6, 20 denoising steps.
- Observed: Local paper-contract diffusion script instantiates a Transformer encoder with configurable embed/heads/layers/steps and records dry-run/full-train mode.
- Required fix: Use this route instead of the older MLP denoiser for paper-facing runs.
- Claim boundary: Dry-run architecture pass does not prove full diffusion training or rollout quality.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_contract_transformer_state_latent_diffusion_training.py:155; /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_paper_contract_transformer_state_latent_diffusion_training.py:74`

### Diffusion / State-latent trajectory representation
- Status: `pass`
- Expected: Paper uses hybrid character-yaw-centric state plus latent, not raw 160-D policy observations.
- Observed: Code path now supports paper_hybrid state windows and the paper-contract wrapper requires it, but the existing generated dataset is still old policy_obs/latent data and must be rebuilt.
- Required fix: Regenerate the dataset from teacher rollout shards containing raw root/body state; do not train from the old policy_obs dataset.
- Claim boundary: A code-path pass is not a data-product pass; existing policy_obs+latent artifacts remain blocked.
- Evidence: `/mnt/infini-data/test/BeyondMimic/res/audits/state_latent_dataset_source_contract/beyondmimic_state_latent_dataset_source_contract_audit.json; /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py:76; /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:not_found:hybrid state representation`

### Diffusion / Trainable state-latent dataset freshness
- Status: `blocked`
- Expected: Generated training shards must contain corrected hybrid state windows, raw rollout state, and reset-safe windows.
- Observed: existing_dataset_corrected=False, teacher_shards_have_raw_state=False, window_filter_ready=False.
- Required fix: Collect new teacher rollout shards with raw world-state fields, then rebuild hybrid state-latent windows with done/reset rejection.
- Claim boundary: Until regenerated, downstream VAE/diffusion/guidance training from old shards is prohibited.
- Evidence: `/mnt/infini-data/test/BeyondMimic/res/audits/state_latent_dataset_source_contract/beyondmimic_state_latent_dataset_source_contract_audit.json`

### Diffusion / VAE rollout collection, OU noise, rejection, symmetry
- Status: `mismatch`
- Expected: Paper collects VAE rollouts with OU action noise, rejects failures before 5 s, and applies sagittal symmetry augmentation.
- Observed: Current local state-latent dataset scripts do not prove OU rollout/rejection/symmetry augmentation in the paper-contract path.
- Required fix: Add OU noise collection, 5 s stability rejection, and symmetry augmentation manifests before long diffusion training.
- Claim boundary: Current denoising MSE improvements are useful local probes, not paper-level data collection.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:not_found:Ornstein; /mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:not_found:symmetry`

### Guidance / Classifier guidance and task costs
- Status: `blocked`
- Expected: Guidance must optimize future states in a receding-horizon closed-loop denoising/control loop.
- Observed: Local guidance evaluations are still mostly offline/proxy; they do not validate MuJoCo closed-loop joystick/waypoint/inpainting/obstacle control.
- Required fix: Implement receding-horizon MuJoCo control where diffusion generates latent, VAE decodes action, and physics feeds back state.
- Claim boundary: Offline guidance cost improvement is not a successful task rollout video.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/paper/source/tex/method.tex:6; /mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_state_latent_guidance_eval.py:not_found:offline`

### Guidance / SDF relaxed barrier formula
- Status: `pass`
- Expected: B(x,delta)=-ln(x) for x>=delta, otherwise -ln(delta)+0.5*((x-2delta)/delta)^2-0.5.
- Observed: Local sdf_barrier has been repaired to the paper piecewise formula and is covered by core math tests.
- Required fix: Keep the unit test as a regression guard for obstacle guidance.
- Claim boundary: This formula fix alone does not make obstacle avoidance closed-loop successful.
- Evidence: `/mnt/infini-data/test/BeyondMimic/reproduction/src/beyondmimic_reimpl/guidance/costs.py:35`

### MuJoCo deployment adapter / Native action adapter
- Status: `pass`
- Expected: Policy output action must map to theta_sp=theta0+alpha*a and MuJoCo actuator targets without semantic clipping errors.
- Observed: Formula adapter is recorded, but unit targets inside MuJoCo ctrlrange are not fully validated.
- Required fix: Resolve ctrlrange/action-scale compatibility before judging PPO teacher rollout quality in MuJoCo.
- Claim boundary: Approximate action adapter videos cannot be used as official deployment evidence.
- Evidence: `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_action_adapter_contract/mujoco_native_action_adapter_contract.json`

### MuJoCo deployment adapter / Native observation adapter
- Status: `blocked`
- Expected: MuJoCo state must reproduce IsaacLab/deployment observation terms without reference-frame or last-action bugs.
- Observed: Native observation adapter audit is still blocked; previous videos used approximate obs/root assist and showed forward-leaning stance.
- Required fix: Validate MuJoCo obs term-by-term against IsaacLab/motion_tracking_controller before long policy/video claims.
- Claim boundary: A dimension-correct 160-D vector is insufficient evidence.
- Evidence: `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_native_observation_adapter_contract/mujoco_native_observation_adapter_contract.json`

### MuJoCo deployment adapter / Material/contact and no-root-assist video gate
- Status: `blocked`
- Expected: Paper-facing simulation should use valid contact/material semantics and no external root assist for success videos.
- Observed: Current MuJoCo contract audit still records floor material mismatch and no-root-assist/native video gate failure.
- Required fix: Fix floor/contact/material and produce no-root-assist native videos before success-folder cleanup.
- Claim boundary: Root-assisted videos are diagnostic/report visuals only.
- Evidence: `/mnt/infini-data/test/BeyondMimic/res/audits/mujoco_control_contract_audit/mujoco_control_contract_audit.json`

### Training permission / Current teacher/downstream readiness
- Status: `blocked`
- Expected: Teacher quality and adapter gates must pass before downstream VAE/diffusion/guidance long training.
- Observed: Pretraining hard gate still blocks downstream training from the current teacher chain.
- Required fix: Do not start long downstream runs until the mismatches above are fixed and teacher quality is re-evaluated.
- Claim boundary: The project remains goal_complete=false and cannot claim full BeyondMimic reproduction.
- Evidence: `/mnt/infini-data/test/BeyondMimic/res/audits/pretraining_hard_gate/beyondmimic_pretraining_hard_gate_audit.json`

