# BeyondMimic Reproduction Technical Report

Generated at: `2026-06-23T07:25:13.455354+00:00`

## 0. Executive Summary

This project currently has a substantial, auditable reproduction codebase for BeyondMimic, but it does **not** fully reproduce BeyondMimic at paper level. The latest GPUs 5/6 multi-source Stage 1 teacher training completed and downstream VAE/state-latent/diffusion/guidance artifacts were generated. However, the teacher remains weak, and the newest MuJoCo action-control videos still do not show stable paper-quality humanoid motion.

Most important current finding: the diffusion denoiser reduces token MSE from `0.072816` to `0.043221` (`40.6%` improvement), but token-level denoising success does not imply closed-loop humanoid control success.

## 1. Paper Method Overview

BeyondMimic can be understood as a four-stage pipeline:

```text
Human motions
    -> RL motion tracking teacher policies
    -> DAgger / conditional VAE latent action policy
    -> state-latent diffusion model
    -> test-time guidance for new tasks
```

The key control equation used by the local MuJoCo diagnostics is:

$$\theta^{sp} = \theta^0 + \alpha \odot a$$

where the policy or VAE decoder produces action `a`, `alpha` is action scale, and the simulator executes a PD-like position target. The project currently has local approximations for this path, but official BeyondMimic VAE/diffusion checkpoints and real robot logs are not public.

## 2. Current Project Inventory

- Paper PDF: `download/papers/BeyondMimic_2508.08241.pdf` (FOUND)
- Paper source tar: `download/papers/BeyondMimic_2508.08241_source.tar` (FOUND)
- Official Stage 1 code: `download/official/whole_body_tracking` (FOUND)
- Official controller reference: `download/official/motion_tracking_controller` (FOUND)
- IsaacLab: `download/dependencies/IsaacLab-v2.1.0` (FOUND)
- MuJoCo experiment package: `mujoco_mp4` (FOUND)
- Report file inventory: `report/file_inventory.txt`

## 3. Data Sources and Preprocessing

The latest local Stage 1 bundle contains `49` motions and `2.491` hours. Source counts are:

```json
{
  "BeyondMimic Zenodo ablation reference CSV": 1,
  "HuB supplemental 29-DoF pkl": 8,
  "Unitree-retargeted LAFAN1": 40
}
```

This is close in duration to the paper's reported 2.5h, but it is **not** guaranteed to be the authors' exact private curated set. See `report/data_report.md` and `report/tables/dataset_inventory.csv`.

## 4. Module 1: Motion Tracking Teacher

Input: processed reference motion bundle, Unitree G1 model, IsaacLab task, reward/termination/PPO config.

Processing: PPO policy maps observation to 29-D action, action becomes PD target, environment returns tracking rewards and done signals.

Current result: checkpoint sweep selected iteration `29999` with reward mean `0.024131401152315747`, body-position error mean `1.0095036663737982`, and joint-position error mean `1.6739522380175`.

Gap to paper: the teacher is weak and cannot be treated as robust BeyondMimic motion tracking.

## 5. Module 2: Teacher Rollout

The selected teacher produced `612352` rollout samples across `2` shards. Done count is `118220`. These samples are useful local training data, but not official DAgger rollouts.

## 6. Module 3: Conditional VAE and DAgger

The local VAE uses teacher rollout obs/action pairs. Current test action MSE is `0.003289680986199528`. This is offline reconstruction evidence, not full DAgger closed-loop reproduction.

Formula:

$$\mathcal{L}_{VAE} = \|a - D(o,z)\|_2^2 + \beta D_{KL}(q(z|o,a)\|N(0,I))$$

## 7. Module 4: State-Latent Trajectory Diffusion

The local state-latent dataset contains `571392` windows with token dimension `192`.

Denoising result:

- Noisy token MSE: `0.072816`
- Test pred token MSE: `0.043221`
- Relative improvement: `40.6%`

This is a meaningful token-level result, but it does not prove physically stable humanoid control.

## 8. Module 5: Test-Time Guidance

Current guidance status: `ok_stage1_multisource_state_latent_guidance_eval`. The local run evaluates `8192` windows and records nonzero gradients / improving best costs for proxy tasks. This is offline evidence, not paper Fig. 5/Fig. 6 closed-loop evaluation.

## 9. Module 6: MuJoCo / Isaac Rendering

H20 true Isaac rendered MP4 remains blocked by the Isaac Sim Kit/Hydra/Vulkan rendering stack. MuJoCo rendering works and generated six continuous videos under:

`/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_continuous_mujoco_action_control_videos`

These videos use continuous motion-time steps and no reset stitching, but they remain failure/diagnostic videos because the current teacher and action-control chain are unstable.

## 10. Current Quantitative Results

- **Data collection and motion bundle**: `PARTIAL`; 49 motions, 2.491 h; evidence `res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle.json`.
- **PPO motion tracking teacher**: `FAILED/PARTIAL`; best iteration 29999, reward 0.024131401152315747, body error 1.0095036663737982; evidence `res/tracking/stage1_multisource_paper_contract_ppo_checkpoint_sweep/tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.json`.
- **Teacher rollout dataset**: `PARTIAL`; 612352 env steps, done_count=118220; evidence `res/tracking/stage1_multisource_best_teacher_rollout_dataset/tracking_stage1_multisource_best_teacher_rollout_dataset.json`.
- **Conditional VAE**: `PARTIAL`; test action MSE 0.003289680986199528; evidence `res/level_c/stage1_multisource_teacher_rollout_vae_training/level_c_stage1_multisource_teacher_rollout_vae_training.json`.
- **State-latent dataset**: `PARTIAL`; 571392 windows, token_dim=192; evidence `res/level_c/stage1_multisource_teacher_rollout_state_latent_dataset/level_c_stage1_multisource_teacher_rollout_state_latent_dataset.json`.
- **Diffusion denoiser**: `PARTIAL`; test pred token MSE 0.04322136765612023, noisy token MSE 0.07281625297452722; evidence `res/level_c/stage1_multisource_state_latent_diffusion_training/level_c_stage1_multisource_state_latent_diffusion_training.json`.
- **Classifier/task guidance**: `PARTIAL`; 8192 offline windows, 4 tasks improve in offline proxy; evidence `res/level_c/stage1_multisource_state_latent_guidance_eval/level_c_stage1_multisource_state_latent_guidance_eval.json`.
- **MuJoCo action-control videos**: `FAILED/PARTIAL`; 6 videos, checks={'all_continuous_primary_time_steps': True, 'all_mp4_exist': True, 'all_primary_metrics_csv_exist': True, 'does_not_claim_complete_beyondmimic_reproduction': True, 'does_not_claim_real_robot': True, 'selected_segment_single_source_motion': True}; evidence `res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/stage1_multisource_continuous_video_suite_summary.json`.

Detailed metrics are in:

- `report/tables/metrics_summary.csv`
- `report/experiment_results.md`
- `report/figures/denoising_mse_improvement.png`

## 11. Current Qualitative Video Results

Video index: `report/videos/video_index.md`

Failure montage: `report/figures/failure_montage.png`

The latest six videos are:

```json
[
  "diffusion_denoised_latent_action_control",
  "guided_latent_action_control",
  "guided_vs_unguided_action_control",
  "reference_action_control",
  "teacher_policy_action_control",
  "vae_reconstructed_action_control"
]
```

## 12. Failure Analysis

The most likely high-level cause is not one isolated rendering bug. The current teacher is weak, so VAE/diffusion/guidance inherit weak action distributions. Deployment mismatch may amplify the weakness. See `report/failure_analysis.md`.

## 13. Paper-vs-Project Alignment

See `report/paper_vs_project.md` and `report/tables/paper_project_comparison.csv`.

Summary: data/preprocessing and local downstream code are partially reproduced; paper-level teacher quality, official DAgger, official VAE/diffusion checkpoints, TensorRT/asynchronous deployment, Fig. 5/Fig. 6 videos, and real robot results are not reproduced.

## 14. Next Debugging Priorities

1. Fix Stage 1 teacher on a single clean motion before more multi-source training.
2. Audit reward/termination/reset/action-scale/PD gain contract.
3. Recollect rollout data only after the teacher has stable closed-loop tracking.
4. Retrain VAE/diffusion and rerun receding-horizon MuJoCo videos.

## 15. Links to Code Snippets and Pseudocode

- `report/code_snippets.md`
- `report/pseudocode.md`
- `report/code_review/key_code_index.md`
- `report/appendix/equations.md`

## Non-Claim Boundary

This project does not fully reproduce BeyondMimic at paper-level. Current MP4s are local MuJoCo virtual diagnostics, not real robot results and not official Isaac rendered paper videos.
