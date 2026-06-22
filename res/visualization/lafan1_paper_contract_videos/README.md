# LAFAN1 Paper-Contract MuJoCo Action-Control Videos

These MP4s are local MuJoCo action-to-PD control visualizations generated from the current paper-contract Stage-1 teacher, VAE, diffusion, and offline guidance artifacts.

They are not Isaac rendered MP4s, not official BeyondMimic checkpoints, not real robot results, and not paper-level Fig.5/Fig.6 closed-loop reproduction.

## Selected Teacher

- Shard: `/mnt/infini-data/test/BeyondMimic/res/runs/tracking_g1_official_importer_export_paper_contract_best_teacher_rollout_dataset/resource_adjusted_teacher_rollout_20260622_222256_seed20260804/rank_1/teacher_rollout_shard.npz`
- Rank/env: `1/462`
- First done frame: `0`
- Mean reward: `0.030472`

## Videos

- `reference_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_paper_contract_videos/reference_action_control/reference_action_control.mp4`
- `teacher_policy_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_paper_contract_videos/teacher_policy_action_control/teacher_policy_action_control.mp4`
- `vae_reconstructed_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_paper_contract_videos/vae_reconstructed_action_control/vae_reconstructed_action_control.mp4`
- `diffusion_denoised_latent_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_paper_contract_videos/diffusion_denoised_latent_action_control/diffusion_denoised_latent_action_control.mp4`
- `guided_latent_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_paper_contract_videos/guided_latent_action_control/guided_latent_action_control.mp4`
- `guided_vs_unguided_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_paper_contract_videos/guided_vs_unguided_action_control/guided_vs_unguided_action_control.mp4`

## Claim Boundary

The suite uses MuJoCo `mj_step` and 29 position actuators, but also uses a root-assist stabilizer for report-ready visualization. It should be described as local virtual simulation evidence, not full BeyondMimic reproduction.
