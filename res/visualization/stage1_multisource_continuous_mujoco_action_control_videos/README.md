# Stage-1 Multi-Source Continuous MuJoCo Action-Control Videos

This directory contains the corrected continuous video suite for the GPUs 5/6 multi-source teacher chain.

## Continuity Gate

- Shard: `/mnt/infini-data/test/BeyondMimic/res/runs/stage1_multisource_best_teacher_rollout_dataset/resource_adjusted_teacher_rollout_20260623_055602_seed20260854/rank_1/teacher_rollout_shard.npz`
- Rank/env: `1/606`
- Source frames: `1:299`
- Rendered frames: `298`
- Motion time steps: `418177..418474`
- Done count: `0`
- Source motion: `lafan1_walk3_subject4`
- Source family: `Unitree-retargeted LAFAN1`

## Videos

- `reference_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/reference_action_control/reference_action_control.mp4`
- `teacher_policy_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/teacher_policy_action_control/teacher_policy_action_control.mp4`
- `vae_reconstructed_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/vae_reconstructed_action_control/vae_reconstructed_action_control.mp4`
- `diffusion_denoised_latent_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/diffusion_denoised_latent_action_control/diffusion_denoised_latent_action_control.mp4`
- `guided_latent_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/guided_latent_action_control/guided_latent_action_control.mp4`
- `guided_vs_unguided_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/guided_vs_unguided_action_control/guided_vs_unguided_action_control.mp4`

## Claim Boundary

These are local MuJoCo diagnostics. The reference video is continuous pose replay; the other videos use MuJoCo `mj_step`, 29 position actuators, and root assist. They are not official BeyondMimic paper-level results.
