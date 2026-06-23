# LAFAN1 Continuous MuJoCo Action-Control Videos

This directory is the corrected replacement for the old reset-spliced LAFAN1 action-control video suite.

## Continuity Gate

- Shard: `/mnt/infini-data/test/BeyondMimic/res/runs/tracking_g1_official_importer_export_paper_contract_best_teacher_rollout_dataset/resource_adjusted_teacher_rollout_20260622_222256_seed20260804/rank_1/teacher_rollout_shard.npz`
- Rank/env: `1/419`
- Source frames: `79:156`
- Rendered frames: `77`
- Motion time steps: `2314..2390`
- Done count: `0`

## Videos

- `reference_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_continuous_mujoco_action_control_videos/reference_action_control/reference_action_control.mp4`
- `teacher_policy_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_continuous_mujoco_action_control_videos/teacher_policy_action_control/teacher_policy_action_control.mp4`
- `vae_reconstructed_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_continuous_mujoco_action_control_videos/vae_reconstructed_action_control/vae_reconstructed_action_control.mp4`
- `diffusion_denoised_latent_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_continuous_mujoco_action_control_videos/diffusion_denoised_latent_action_control/diffusion_denoised_latent_action_control.mp4`
- `guided_latent_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_continuous_mujoco_action_control_videos/guided_latent_action_control/guided_latent_action_control.mp4`
- `guided_vs_unguided_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_continuous_mujoco_action_control_videos/guided_vs_unguided_action_control/guided_vs_unguided_action_control.mp4`

## Claim Boundary

The reference video is continuous pose replay. The other videos use MuJoCo `mj_step`, 29 position actuators, and root assist. They are local diagnostics from a weak teacher chain, not official BeyondMimic paper-level results.

Old failed suite audit: `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_paper_contract_videos/failed_discontinuous_action_control_audit.json`
