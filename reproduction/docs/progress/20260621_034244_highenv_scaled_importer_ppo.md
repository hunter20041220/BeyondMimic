# Progress Update

## Goal

Advance the official-importer-export tracking path beyond the previous scaled PPO run by rerunning the same audited path with 4096 environments per rank on GPUs 4 and 7, refreshing the checkpoint evaluation, regenerating report plots, and capturing a current policy-vs-reference rollout video.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_full_bundle_ppo_eval_report_assets.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_full_bundle_scaled_ppo_eval_report_assets.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_policy_rollout_video_capture.py`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260621_034244_highenv_scaled_importer_ppo.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_full_bundle_ppo_eval_report_assets.py`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_training_run/tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/`
- `/mnt/infini-data/test/BeyondMimic/res/visualization/official_importer_export_full_bundle_scaled_ppo_policy_rollout/`

## Commands Run

```bash
nvidia-smi
python3 reproduction/scripts/gpu_wangjc_process_guard.py
BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_NUM_ENVS_PER_RANK=4096 BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_MAX_ITERATIONS=1000 BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_SEED=20260696 envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run.py
BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_EVAL_SEED=20260697 BM_OFFICIAL_IMPORTER_EXPORT_SCALED_PPO_EVAL_NUM_ENVS=2048 envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.py
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_full_bundle_scaled_ppo_eval_report_assets.py
BM_IMPORTER_SCALED_PPO_POLICY_VIDEO_SEED=20260698 envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_policy_rollout_video_capture.py
```

## Results

- The higher-env scaled PPO training completed successfully on physical GPUs 4 and 7.
- Training config: world size `2`, `4096` envs/rank, `8192` total envs, `24` steps/env, `1000` PPO iterations, seed `20260696`.
- Training output: rank0 iteration `999`, `21` local checkpoints, `196608000` rank0/global timesteps, duration `3242.741` seconds.
- Training telemetry: GPU4 peak `7771` MiB, GPU7 peak `7767` MiB; mean utilization `56.14%` and `50.51%`.
- Checkpoint evaluation completed with `2048` envs x `299` steps, seed `20260697`, total env steps `612352`.
- Eval metrics: reward mean `0.02423080788881683`, done count total `611642`, anchor/body/joint position error means `0.05960297264333154`, `0.6893615395727763`, and `0.8996927592666651`.
- The current single-env video asset completed from the iteration-999 checkpoint with `299` frames, reward mean `0.024693377315998077`, and target-body error mean/max `0.3432866036891937` / `0.3649991452693939`.

## Verification

Verification passed after this progress note:

- `python3 reproduction/scripts/required_artifact_absence_audit.py`: ok, 29 rows.
- `python3 reproduction/scripts/artifact_manifest.py`: ok, 1145 artifacts.
- `python3 reproduction/scripts/paper_vs_reproduction_comparison.py`: ok, 191 rows.
- `python3 reproduction/scripts/final_reproduction_report.py`: ok.
- `python3 reproduction/scripts/completion_matrix_status_audit.py`: ok, 182 rows, 0 invalid statuses.
- `python3 reproduction/scripts/verification_command_syntax_audit.py`: ok.
- `python3 reproduction/scripts/verification_command_script_manifest.py`: ok.
- `python3 reproduction/scripts/verification_command_coverage_audit.py`: ok.
- `python3 reproduction/scripts/reproduction_master_audit.py`: ok, `goal_complete=false`.

## Failed / Blocked Items

- The run did not satisfy the formal 10GB/card memory target; peak training memory was about `7.77` GB/card even at `8192` total environments.
- The policy quality remains weak: reward is low and done counts are very high, so the checkpoint is not a mature tracking teacher.
- The produced checkpoints and videos are local virtual reproduction artifacts, not official BeyondMimic weights, not Fig. 5/Fig. 6 guided diffusion evidence, not TensorRT deployment, and not real-robot evidence.

## Effect on English Reading Report

This gives the English reading report a stronger and more current tracking-side evidence block: the official-importer-export path can run a longer two-GPU PPO job, evaluate the latest checkpoint, and produce visible robot-motion media. The report also now has a clearer limitation statement that the larger run still does not reach paper-level teacher quality or the 10GB/card formal GPU threshold.

## Next Step

Refresh all audit/report outputs, inspect the resulting diff for large artifacts, commit the code/report/audit updates, and attempt a GitHub push.

## Git Commit

Pending at the time this progress note was created.
