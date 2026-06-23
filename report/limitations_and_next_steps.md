# Next Steps

## Highest Priority

1. Single-motion teacher sanity retraining: pick one clean LAFAN1 walking/running motion and make PPO visibly track it before multi-source training.
2. Reward/termination audit: dump reward components, done causes, body/joint errors, and reset phases for the weak 5/6 checkpoint.
3. MuJoCo/Isaac action contract audit: verify joint order, action scale, default pose, PD gains, armature, and control frequency with one-joint tests.

## After Teacher Repair

1. Recollect teacher rollout shards from stable policy.
2. Retrain VAE and check closed-loop VAE rollout.
3. Rebuild state-latent dataset and retrain denoiser.
4. Implement receding-horizon guided MuJoCo closed-loop tasks.
5. Run RTX Isaac rendered MP4 if true Isaac visuals are still needed.

## Files Most Likely To Inspect Next

- `/mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py`
- `/mnt/infini-data/test/BeyondMimic/download/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_pd_control_video.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_stage1_multisource_paper_contract_ppo_checkpoint_eval.py`
