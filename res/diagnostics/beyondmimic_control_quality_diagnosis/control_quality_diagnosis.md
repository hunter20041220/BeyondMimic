# BeyondMimic Control Quality Diagnosis

Generated: `2026-06-22T08:30:08.843520+00:00`

## Executive Conclusion

The current poor MuJoCo movement-control videos are not just a rendering problem. They expose two upstream issues:

- The best local tracking teachers are still weak local virtual PPO checkpoints, with low reward, high done rates, and large body/joint errors.
- The current MuJoCo control videos do not run a faithful native PPO/VAE/guided controller. They use target joint/IK sequences, MuJoCo PD actuators, and root assist, so they cannot prove BeyondMimic-style learned control.

This project still must not claim full paper-level BeyondMimic reproduction.

## Paper And Official-Code Contract

- Policy observation: 160-D concatenation of generated motion command, anchor position/orientation error, base linear/angular velocity, relative joint position, relative joint velocity, and previous action.
- Action: 29-D normalized joint position setpoint command, converted with per-joint action scale and executed by low-level PD.
- Robot target bodies: 14 G1 bodies with torso as anchor.
- Tracking rewards: anchor global pose plus relative body position/orientation/linear-velocity/angular-velocity terms with paper tolerances around position 0.3 m, orientation 0.4 rad, linear velocity 1.0, angular velocity 3.14.
- Official training scale in source config: 4096 envs and 30000 PPO iterations; local runs are resource-adjusted and shorter.
- Important termination terms: anchor z, anchor orientation, and end-effector z-only body-position errors.

## Current Tracking Teacher Metrics

| name | reward_mean | done_rate | ee_term_rate | anchor_pos_term_rate | body_pos_err | joint_pos_err | action_abs_mean | claim |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| scaled_ppo_iter999 | 0.0242308 | 0.998841 | 0.998841 | 0.000349472 | 0.68763 | 0.894446 | 0.0322014 | local_virtual_tracking_eval |
| scaled_ppo_best_iter300 | 0.0237094 | 0.998587 | 0.998587 | 0.000413161 | 0.712698 | 0.918122 | 0.0275359 | local_virtual_tracking_eval |
| fk_repaired_robot_order_iter999 | 0.0207338 | 0.17828 | 0.154752 | 0.0310736 | 0.483168 | 1.84876 | 0.558096 | local_virtual_tracking_eval |
| endpoint_threshold_candidate_iter999 | 0.00550017 | 0.094075 | 0.0317922 | 0.0698373 | 0.318498 | 3.23716 | 0.834412 | tracking_endpoint_threshold_candidate_full_ppo_eval |

## Current MuJoCo Video Evidence

| video | duration | mj_step | root_assist | native_adapter | joint_error | root_error | claim |
| --- | ---: | --- | --- | --- | ---: | ---: | --- |
| denoised_latent_control | 15 | True | True | False | 0.177254 | 0.106536 | MuJoCo PD closed-loop tracking of IK targets fitted from existing local IsaacLab denoised-latent body trace; not native MuJoCo denoiser controller |
| guided_latent_control | 15 | True | True | False | 0.212337 | 0.121924 | MuJoCo PD closed-loop tracking of IK targets fitted from existing local IsaacLab guided-latent body trace; not native MuJoCo guided controller |
| guided_vs_unguided_control | 15 | False | False | False |  |  | Side-by-side MuJoCo PD closed-loop tracking-control videos: VAE-base target tracking on the left, guided-latent target tracking on the right; not native MuJoCo guidance |
| ppo_policy_control | 15 | True | True | False | 0.209981 | 0.124581 | MuJoCo PD closed-loop tracking of IK targets fitted from existing local IsaacLab PPO body trace; not native MuJoCo PPO policy |
| reference_control | 15 | True | True | False | 0.0826065 | 0.0364011 | MuJoCo PD closed-loop tracking of FK-repaired reference joint targets |
| vae_base_control | 15 | True | True | False | 0.196705 | 0.0758609 | MuJoCo PD closed-loop tracking of IK targets fitted from existing local IsaacLab VAE-base body trace; not native MuJoCo VAE controller |

## Diagnosis Findings

- **weak_tracking_teacher**: The current strongest local PPO teachers still have low reward and high done/termination rates; downstream VAE/diffusion can only imitate this weak behavior. Evidence: scaled_ppo_iter999 reward_mean=0.0242308 done_rate=0.998841; scaled PPO termination diagnostic dominant component=ee_body_pos fraction=0.998841
- **endpoint_body_semantics**: Endpoint z-only body-position termination dominates or remains a major failure source. Relaxing the endpoint threshold improves done rate but changes the evaluator and does not repair paper-level tracking. Evidence: threshold_sweep best_threshold=0.5, best_done_rate=0.0890762, why_not_paper_level=Changing the termination threshold changes the evaluator, so these rows are diagnostics only and cannot be reported as official BeyondMimic tracking scores.
- **mujoco_native_adapter_missing**: The native MuJoCo 160-D IsaacLab-compatible observation/action adapter is not complete, so the IsaacLab PPO checkpoint cannot be honestly claimed as a MuJoCo closed-loop policy. Evidence: adapter_gap checks native_mujoco_adapter_complete=False; actor input/output dimensions are 160/29.
- **pd_root_assist_video_boundary**: The current MuJoCo control videos use PD target tracking plus root-assist external forces; they are useful diagnostics but not learned unassisted humanoid control. Evidence: all current control rows mark native_mujoco_ppo_obs_adapter=False; most controller rows use_root_assist_controller=True.
- **missing_official_level_c_artifacts**: Official BeyondMimic VAE/diffusion checkpoints, true DAgger rollout logs, paper-level Fig.5/Fig.6 videos, TensorRT engine, and real-robot evidence remain absent. Evidence: The workspace contains reference-project ONNX/PT/GIF assets and many debug NPZ artifacts, but it does not contain the required official/teacher-rollout BeyondMimic tracking/VAE/diffusion checkpoints, TensorRT engine, closed-loop rollout logs, Fig.5/Fig.6 artifacts, or reproduced success/failure videos. It does contain one public-LAFAN1 paper-architecture VAE/diffusion checkpoint, which is recorded separately and must not be counted as an official DAgger/closed-loop paper checkpoint. It also contains resource-adjusted G1 PPO checkpoints, which prove local virtual training execution but are separately excluded from official paper-level tracking artifacts. The resource-adjusted teacher-rollout and official-importer-export PPO checkpoints are also separately excluded from official paper-level tracking artifacts. The resource-adjusted teacher-rollout VAE checkpoint and state-latent denoiser checkpoint are also separately classified as local evidence rather than official DAgger/VAE/diffusion artifacts, including the newer scaled PPO official-importer-export VAE and denoiser checkpoints. A local kinematic reference MP4 is present for reporting, but it is excluded from paper-level closed-loop/video evidence.

## Recommended Repair Order

1. **Stop regenerating paper-claim videos from the current weak teacher**: Longer MP4s will not fix the controller if the source PPO reward/done/body-error metrics remain poor.
2. **Repair tracking target semantics before training more**: Compare FK-repaired motion body_pos_w/body_quat_w, G1 target-body order, wrist endpoint z, reset pose, and target refresh against the official MotionCommand contract.
3. **Train/evaluate a stronger IsaacLab tracking teacher first**: Use the official 160-D observation and 29-D action path in IsaacLab non-render mode; track reward, done breakdown, body/joint errors, action distribution, and multi-seed variance.
4. **Only then rebuild teacher rollout, VAE, state-latent dataset, diffusion, and guidance**: Low reconstruction or denoising MSE against a weak teacher does not imply high-quality humanoid control.
5. **Implement a MuJoCo adapter as a separate validation project**: A valid adapter must mirror generated_commands, anchor-frame errors, base velocities, joint relative states, last action, action scale, empirical normalization, reset/termination semantics, and actuator limits before loading IsaacLab PPO weights.
6. **If MuJoCo is the final simulator, train a native MuJoCo tracking policy**: Directly transferring an IsaacLab PPO checkpoint into MuJoCo without matching physics, observation manager, normalization, and actuator semantics is not reliable.

## Claim Boundary

Current claim level: local virtual partial reproduction and diagnostics only. The MuJoCo videos can be used as diagnostic/report media, but not as official BeyondMimic PPO/VAE/guided control evidence. Real-robot deployment remains unavailable.
