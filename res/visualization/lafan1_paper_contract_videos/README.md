# LAFAN1 Paper-Contract MuJoCo Action-Control Videos

These MP4s separate continuous reference visualization from MuJoCo action-to-PD control diagnostics generated from the current paper-contract Stage-1 teacher, VAE, diffusion, and offline guidance artifacts.

They are not Isaac rendered MP4s, not official BeyondMimic checkpoints, not real robot results, and not paper-level Fig.5/Fig.6 closed-loop reproduction.

## Reference Semantics

- `reference_pose_replay` is the clean continuous LAFAN1 reference visualization. It writes root pose and 29 joint positions frame-by-frame with `mj_forward`; use this to show what the source motion looks like.
- `reference_action_control` is a PD tracking diagnostic. It uses discontinuous teacher-rollout time steps and MuJoCo `mj_step`; do not treat it as the original dataset motion replay.

## Selected Teacher

- Shard: `/mnt/infini-data/test/BeyondMimic/res/runs/tracking_g1_official_importer_export_paper_contract_best_teacher_rollout_dataset/resource_adjusted_teacher_rollout_20260622_222256_seed20260804/rank_1/teacher_rollout_shard.npz`
- Rank/env: `1/462`
- First done frame: `0`
- Mean reward: `0.030472`
- Teacher motion-time-step non-+1 jumps: `35`

## Videos

- `reference_pose_replay`: `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_paper_contract_videos/reference_pose_replay/reference_pose_replay.mp4`
- `reference_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_paper_contract_videos/reference_action_control/reference_action_control.mp4`
- `teacher_policy_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_paper_contract_videos/teacher_policy_action_control/teacher_policy_action_control.mp4`
- `vae_reconstructed_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_paper_contract_videos/vae_reconstructed_action_control/vae_reconstructed_action_control.mp4`
- `diffusion_denoised_latent_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_paper_contract_videos/diffusion_denoised_latent_action_control/diffusion_denoised_latent_action_control.mp4`
- `guided_latent_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_paper_contract_videos/guided_latent_action_control/guided_latent_action_control.mp4`
- `guided_vs_unguided_action_control`: `/mnt/infini-data/test/BeyondMimic/res/visualization/lafan1_paper_contract_videos/guided_vs_unguided_action_control/guided_vs_unguided_action_control.mp4`

## Claim Boundary

The action-control videos use MuJoCo `mj_step` and 29 position actuators, but also use a root-assist stabilizer for report-ready visualization. The reference-pose replay writes qpos frame-by-frame. The suite should be described as local virtual simulation evidence, not full BeyondMimic reproduction.
