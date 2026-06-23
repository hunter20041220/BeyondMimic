# Command History and Reproduction Commands

The following commands are reconstructed from current project files and this report-generation round. Items marked inferred should be treated as commands-to-run, not guaranteed original shell history.

## Stage1 multi-source checkpoint sweep

```bash
CUDA_VISIBLE_DEVICES=5,6 BM_STAGE1_MULTISOURCE_CHECKPOINT_SWEEP_TARGET_GPUS=5,6 BM_STAGE1_MULTISOURCE_CHECKPOINT_SWEEP_NUM_ENVS=256 BM_STAGE1_MULTISOURCE_CHECKPOINT_SWEEP_EVAL_STEPS=299 BM_STAGE1_MULTISOURCE_CHECKPOINT_SWEEP_STRIDE=2500 python3 reproduction/scripts/tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.py
```

## Teacher rollout

```bash
CUDA_VISIBLE_DEVICES=5,6 BM_STAGE1_MULTISOURCE_TEACHER_ROLLOUT_NUM_ENVS_PER_RANK=1024 BM_TEACHER_ROLLOUT_STEPS=299 python3 reproduction/scripts/tracking_stage1_multisource_best_teacher_rollout_dataset.py
```

## VAE / diffusion / guidance

```bash
CUDA_VISIBLE_DEVICES=5,6 python3 reproduction/scripts/level_c_stage1_multisource_teacher_rollout_vae_training.py
CUDA_VISIBLE_DEVICES=5,6 python3 reproduction/scripts/level_c_stage1_multisource_teacher_rollout_state_latent_dataset.py
CUDA_VISIBLE_DEVICES=5,6 python3 reproduction/scripts/level_c_stage1_multisource_state_latent_diffusion_training.py
CUDA_VISIBLE_DEVICES=5,6 python3 reproduction/scripts/level_c_stage1_multisource_state_latent_guidance_eval.py
```

## MuJoCo video suite

```bash
MUJOCO_GL=egl mujoco_mp4/.venv/bin/python reproduction/scripts/stage1_multisource_continuous_mujoco_action_control_videos.py
```
