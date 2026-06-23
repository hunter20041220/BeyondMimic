# Stage-1 Multi-Source Quality-Gated MuJoCo Action-Control Videos

This directory contains the corrected quality-gated short video suite for the GPUs 5/6 multi-source teacher chain.

## Quality Gate

- Target frames: `30`
- Minimum root z mean: `0.45` m
- Minimum root z min: `0.3` m
- Maximum root z range: `0.18` m
- Minimum reward mean: `0.0`

## Selected Segment

- Shard: `/mnt/infini-data/test/BeyondMimic/res/runs/stage1_multisource_best_teacher_rollout_dataset/resource_adjusted_teacher_rollout_20260623_055602_seed20260854/rank_1/teacher_rollout_shard.npz`
- Rank/env: `1/601`
- Source frames: `70:100`
- Rendered frames: `30`
- Motion time steps: `286550..286579`
- Reward mean: `0.05464879038433234`
- Root z mean: `0.7893778244654338`
- Source motion: `lafan1_sprint1_subject4`

## Videos

- `reference_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/reference_action_control/reference_action_control.mp4`
- `teacher_policy_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/teacher_policy_action_control/teacher_policy_action_control.mp4`
- `vae_reconstructed_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/vae_reconstructed_action_control/vae_reconstructed_action_control.mp4`
- `diffusion_denoised_latent_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/diffusion_denoised_latent_action_control/diffusion_denoised_latent_action_control.mp4`
- `guided_latent_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/guided_latent_action_control/guided_latent_action_control.mp4`
- `guided_vs_unguided_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_quality_gated_mujoco_action_control_videos/guided_vs_unguided_action_control/guided_vs_unguided_action_control.mp4`

## Claim Boundary

These are local MuJoCo diagnostics. The reference video is continuous pose replay; the other videos use MuJoCo `mj_step`, 29 position actuators, and root assist. They are not official BeyondMimic paper-level results.
