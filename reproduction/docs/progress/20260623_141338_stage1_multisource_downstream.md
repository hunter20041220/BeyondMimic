# Progress Update

## Goal

Continue after the GPUs 5/6 Stage-1 multi-source PPO teacher run finished: evaluate checkpoints, select the best teacher, collect teacher rollout state-action shards, train local VAE/state-latent diffusion/guidance artifacts, and regenerate continuous MuJoCo action-control videos without reset-spliced motion-time jumps.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_multisource_motion_bundle/tracking_stage1_multisource_motion_bundle.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/stage1_multisource_paper_contract_ppo_training_run/tracking_stage1_multisource_paper_contract_ppo_training_run.json`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_teacher_rollout_dataset.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_teacher_rollout_vae_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_resource_adjusted_state_latent_guidance_eval.py`

## Files Added

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_stage1_multisource_paper_contract_ppo_checkpoint_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_stage1_multisource_best_teacher_rollout_dataset.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_stage1_multisource_teacher_rollout_vae_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_stage1_multisource_teacher_rollout_state_latent_dataset.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_stage1_multisource_state_latent_diffusion_training.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/level_c_stage1_multisource_state_latent_guidance_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/stage1_multisource_continuous_mujoco_action_control_videos.py`

## Commands Run

```bash
python -m py_compile reproduction/scripts/tracking_stage1_multisource_paper_contract_ppo_checkpoint_eval.py reproduction/scripts/tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.py reproduction/scripts/tracking_stage1_multisource_best_teacher_rollout_dataset.py reproduction/scripts/level_c_stage1_multisource_teacher_rollout_vae_training.py reproduction/scripts/level_c_stage1_multisource_teacher_rollout_state_latent_dataset.py reproduction/scripts/level_c_stage1_multisource_state_latent_diffusion_training.py reproduction/scripts/level_c_stage1_multisource_state_latent_guidance_eval.py reproduction/scripts/stage1_multisource_continuous_mujoco_action_control_videos.py
CUDA_VISIBLE_DEVICES=5,6 BM_STAGE1_MULTISOURCE_CHECKPOINT_SWEEP_TARGET_GPUS=5,6 BM_STAGE1_MULTISOURCE_CHECKPOINT_SWEEP_NUM_ENVS=256 BM_STAGE1_MULTISOURCE_CHECKPOINT_SWEEP_EVAL_STEPS=299 BM_STAGE1_MULTISOURCE_CHECKPOINT_SWEEP_STRIDE=2500 python3 reproduction/scripts/tracking_stage1_multisource_paper_contract_ppo_checkpoint_sweep.py
CUDA_VISIBLE_DEVICES=5,6 BM_STAGE1_MULTISOURCE_TEACHER_ROLLOUT_NUM_ENVS_PER_RANK=1024 BM_TEACHER_ROLLOUT_STEPS=299 python3 reproduction/scripts/tracking_stage1_multisource_best_teacher_rollout_dataset.py
CUDA_VISIBLE_DEVICES=5,6 python3 reproduction/scripts/level_c_stage1_multisource_teacher_rollout_vae_training.py
CUDA_VISIBLE_DEVICES=5,6 python3 reproduction/scripts/level_c_stage1_multisource_teacher_rollout_state_latent_dataset.py
CUDA_VISIBLE_DEVICES=5,6 python3 reproduction/scripts/level_c_stage1_multisource_state_latent_diffusion_training.py
CUDA_VISIBLE_DEVICES=5,6 python3 reproduction/scripts/level_c_stage1_multisource_state_latent_guidance_eval.py
MUJOCO_GL=egl mujoco_mp4/.venv/bin/python reproduction/scripts/stage1_multisource_continuous_mujoco_action_control_videos.py
```

## Results

Checkpoint sweep completed for 13 representative checkpoints. The best checkpoint is `/mnt/infini-data/test/BeyondMimic/res/runs/stage1_multisource_paper_contract_ppo_training/resource_adjusted_ppo_20260622_114146_seed20260851/rank_0/model_29999.pt`.

Teacher rollout completed with 2 shards and `612352` total env steps. VAE training completed on `612352` samples with test action MSE `0.003289680986199528`. State-latent dataset generation completed with `571392` windows. Denoiser training completed with test pred token MSE `0.04322136765612023` and denoising improvement ratio `0.40643241185123813`. Offline guidance completed over `8192` validation/test windows.

The MuJoCo continuous video suite generated six MP4s under `/mnt/infini-data/test/BeyondMimic/res/visualization/stage1_multisource_continuous_mujoco_action_control_videos/`. The selected segment is continuous within one source motion, `lafan1_walk3_subject4`, with `298` frames / `9.933333333333334 s`.

## Verification

Immediate checks passed for script syntax, checkpoint evaluation status, teacher rollout status, VAE status, state-latent dataset status, denoiser status, offline guidance wrapper status, and video-suite summary checks.

The full project verification suite is run after this progress note so artifact manifest, comparison, final report, completion matrix, verification command audits, and master audit can include the new small audit artifacts.

## Failed / Blocked Items

The multi-source teacher is still weak. Best checkpoint evaluation reward mean is only `0.024131401152315747`, non-timeout done rate is `0.19413670568561872`, body-position error mean is `1.0095036663737982`, and joint-position error mean is `1.6739522380175`. The new action-control videos are continuous and generated through MuJoCo control, but fall proxy remains high and MuJoCo reported instability warnings. Therefore these videos are pipeline diagnostics, not high-quality motion control results.

The official BeyondMimic teacher checkpoint, official DAgger rollout logs, official VAE/diffusion checkpoints, paper-level Fig.5/Fig.6 closed-loop videos, TensorRT/asynchronous deployment, and real robot evidence remain unavailable or incomplete.

## Effect on English Reading Report

This round provides a coherent code-reproduction story for the report: Stage 1 multi-source teacher training finished; checkpoint screening selected a best local teacher; the downstream VAE/state-latent/diffusion/guidance pipeline ran end-to-end; and continuous MuJoCo videos were generated from one non-reset-spliced source motion. It also provides an honest negative finding: simply expanding the available 2.49-hour motion bundle did not yield paper-quality control on this local chain.

## Next Step

Investigate why the Stage-1 teacher remains weak despite long training and paper-contract settings. Priority checks are reward scale/termination thresholds, motion sampling/reset behavior, action scale/PD gain consistency between IsaacLab and MuJoCo, and whether the multi-source motion bundle contains unstable root heights or impossible segments. Do not claim full BeyondMimic reproduction.

## Git Commit

Pending after full verification and careful staging of code/progress/audit files only. Large MP4s, checkpoints, rollout shards, and run directories should remain local and ignored.
