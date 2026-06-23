# Key Code Index

| stage | functionality | file | line range | origin | status | description |
|---|---|---|---|---|---|---|
| data preprocessing | csv_to_npz | `download/official/whole_body_tracking/scripts/csv_to_npz.py` | 1-36 | official | FOUND | CSV generalized coordinates to motion NPZ / registry workflow |
| tracking | MotionCommand | `download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py` | 53-91 | official | FOUND | Reference-motion command and tracking target computation |
| tracking | reward terms | `download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py` | 8-46 | official | FOUND | DeepMimic-style tracking rewards and smoothing terms |
| tracking | observations | `download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py` | 1-38 | official | FOUND | Policy observation construction |
| tracking | terminations | `download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py` | 1-38 | official | FOUND | Early termination conditions |
| tracking | PPO train entry | `download/official/whole_body_tracking/scripts/rsl_rl/train.py` | 1-38 | official | FOUND | RSL-RL PPO training entry |
| teacher rollout | rollout collection | `reproduction/scripts/tracking_g1_resource_adjusted_teacher_rollout_dataset.py` | 1-32 | custom wrapper | FOUND | Collect state/action/reward/done rollout shards |
| teacher rollout | 5/6 wrapper | `reproduction/scripts/tracking_stage1_multisource_best_teacher_rollout_dataset.py` | 1-38 | custom wrapper | FOUND | Bind best multi-source checkpoint to rollout collector |
| VAE | ConditionalActionVAE | `reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py` | 71-109 | paper-faithful local | FOUND | Encoder/decoder action VAE |
| state-latent | state-latent dataset | `reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py` | 548-586 | paper-faithful local | FOUND | Build token windows from obs and VAE latents |
| diffusion | StateLatentDenoiser | `reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py` | 95-133 | paper-faithful local | FOUND | Noising and denoising training loop |
| guidance | offline guidance | `reproduction/scripts/level_c_resource_adjusted_state_latent_guidance_eval.py` | 1-38 | paper-faithful local | FOUND | Task-cost guidance proxy evaluation |
| MuJoCo | PD video rendering | `mujoco_mp4/scripts/mujoco_pd_control_video.py` | 1-32 | custom MuJoCo | FOUND | Action-to-PD control video rendering |
| MuJoCo | continuous video suite | `reproduction/scripts/stage1_multisource_continuous_mujoco_action_control_videos.py` | 1-32 | custom wrapper | FOUND | Fresh 5/6 continuous MuJoCo video suite |
