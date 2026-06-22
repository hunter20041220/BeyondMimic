# Progress Update

## Goal

Prepare the post-training pipeline for the 4/7 LAFAN1 paper-contract PPO teacher run without interrupting the active training jobs. Once the 4/7 training finishes, the next stage should select the best teacher checkpoint by evaluation, collect teacher rollout state-action data, train the local VAE/diffusion chain, and produce downstream LAFAN1 virtual-control/video evidence.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/watch_stage1_tracking_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_checkpoint_sweep.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_teacher_rollout_dataset.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_teacher_rollout_state_latent_dataset.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_state_latent_diffusion_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_state_latent_guidance_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_csv_loop_policy_rollout_video_capture.py`
- `/mnt/infini-data/test/BeyondMimic/mujoco_mp4/scripts/mujoco_pd_control_video.py`

## Files Modified

New post-training scripts:

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_sweep.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_paper_contract_best_teacher_rollout_dataset.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_official_importer_export_paper_contract_teacher_rollout_vae_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_official_importer_export_paper_contract_state_latent_diffusion_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_official_importer_export_paper_contract_state_latent_guidance_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260623_042217_paper_contract_post_training_pipeline.md`

Existing dirty files under `isaac_mp4_need/` and generated training worker files were not modified.

## Commands Run

Monitoring and status:

```bash
python3 reproduction/scripts/watch_stage1_tracking_training.py --once --no-clear
ps -eo pid,etimes,cmd | rg 'tracking_g1|rsl_rl|resource_adjusted_ppo|stage1|paper_contract' | rg -v 'rg '
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader -i 4,5,6,7
find res/runs/tracking_g1_official_importer_export_paper_contract_ppo_training/resource_adjusted_ppo_20260622_084243_seed20260801/rank_0 -maxdepth 1 -type f -name 'model_*.pt' -printf '%f %s %TY-%Tm-%Td %TH:%TM:%TS\n' | sort -V | tail -20
```

Script verification:

```bash
envs/bm_analysis/bin/python -m py_compile \
  reproduction/scripts/tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_sweep.py \
  reproduction/scripts/tracking_g1_official_importer_export_paper_contract_best_teacher_rollout_dataset.py \
  reproduction/scripts/level_c_official_importer_export_paper_contract_teacher_rollout_vae_training.py \
  reproduction/scripts/level_c_official_importer_export_paper_contract_teacher_rollout_state_latent_dataset.py \
  reproduction/scripts/level_c_official_importer_export_paper_contract_state_latent_diffusion_training.py \
  reproduction/scripts/level_c_official_importer_export_paper_contract_state_latent_guidance_eval.py
```

## Results

4/7 LAFAN1 paper-contract PPO teacher remains alive and was not interrupted.

Latest monitored state at 2026-06-23 04:21:51 CST:

- Iteration: `27049/30000` (`90.16%`)
- ETA: about `01:15:02`
- Latest checkpoint: `/mnt/infini-data/test/BeyondMimic/res/runs/tracking_g1_official_importer_export_paper_contract_ppo_training/resource_adjusted_ppo_20260622_084243_seed20260801/rank_0/model_27000.pt`
- Reward: about `0.07`
- Episode length: about `3.09`
- Body position error: about `0.4178`
- Joint position error: about `1.1883`
- Joint velocity error: about `4.7695`
- Main blocker signal remains high `ee_body_pos` termination, about `655.5`

5/6 multi-source PPO teacher also remains alive and was not interrupted:

- Iteration: `15325/30000` (`51.08%`)
- ETA: about `08:11:30`
- Latest checkpoint: `/mnt/infini-data/test/BeyondMimic/res/runs/stage1_multisource_paper_contract_ppo_training/resource_adjusted_ppo_20260622_114146_seed20260851/rank_0/model_15000.pt`

## Verification

The six new post-training scripts passed `py_compile`.

No checkpoint sweep, teacher rollout collection, VAE training, diffusion training, or guidance evaluation was started yet, because the 4/7 teacher training is still running.

## Failed / Blocked Items

- The 4/7 teacher run has not finished yet, so no best checkpoint is selected yet.
- No new LAFAN1 downstream rollout/VAE/diffusion/video results are produced yet.
- Current training metrics still indicate a weak teacher candidate; best selection must be based on post-training checkpoint sweep, not the last checkpoint alone.

## Effect on English Reading Report

This adds a reproducible bridge from Stage 1 training to the later reproduction sections:

1. local checkpoint screening,
2. best teacher rollout dataset collection,
3. conditional action VAE training,
4. state-latent diffusion training,
5. offline guidance evaluation.

The report should still state that these are local public-resource reproduction artifacts, not official BeyondMimic teacher/DAgger/VAE/diffusion checkpoints and not real-robot evidence.

## Next Step

Continue monitoring the 4/7 training until completion. After it finishes:

```bash
BM_PAPER_CONTRACT_CHECKPOINT_SWEEP_ITERATIONS=0,500,1000,1500,2000,5000,10000,15000,20000,24000,26000,27000,28000,29000,30000 \
BM_PAPER_CONTRACT_CHECKPOINT_SWEEP_NUM_ENVS=256 \
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_paper_contract_ppo_checkpoint_sweep.py
```

Then run the best-teacher rollout and downstream VAE/diffusion chain if the selected teacher is credible enough.

## Git Commit

Pending. Commit after verification/audit refresh for this script-only pipeline update.
