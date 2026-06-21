# Progress Update

## Goal

Move the tracking reproduction back toward the mainline by screening all saved scaled PPO checkpoints on the official-importer-export G1 path, instead of assuming the final checkpoint is the best local teacher candidate.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_checkpoint_multiseed_eval.py`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_training_run/tracking_g1_official_importer_export_full_bundle_scaled_ppo_training_run.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/.gitignore`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_checkpoint_sweep.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`

## Commands Run

```bash
BM_SCALED_PPO_CHECKPOINT_SWEEP_INCLUDE_FINAL_ONLY=1 BM_SCALED_PPO_CHECKPOINT_SWEEP_NUM_ENVS=64 /mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_checkpoint_sweep.py
BM_SCALED_PPO_CHECKPOINT_SWEEP_NUM_ENVS=256 /mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_checkpoint_sweep.py
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/completion_matrix_status_audit.py
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/verification_command_syntax_audit.py
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/verification_command_script_manifest.py
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/verification_command_coverage_audit.py
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/required_artifact_absence_audit.py
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/progress_report_audit.py
python3 /mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py
git status --short
git diff --stat
```

## Results

The new checkpoint sweep evaluated all `21` saved rank-0 checkpoints from the scaled PPO training run on the official-importer-export G1 USDA and the full 40-motion public bundle.

```text
checkpoint_count: 21
ok_checkpoint_count: 21
num_envs_per_checkpoint: 256
eval_steps_per_checkpoint: 299
total_env_steps: 1607424
best_iteration: 300
best_reward_mean: 0.02327705469343774
best_error_body_pos_mean: 0.6452447620522617
best_local_non_timeout_done_rate: 1.0
```

The sweep suggests that the final iteration-999 checkpoint is not clearly the best local teacher candidate. However, even the best screened checkpoint still has a non-timeout done rate of `1.0`, so the result points to teacher/training diagnosis rather than paper-level tracking success.

## Verification

The sweep is now included in artifact manifest, paper-vs-reproduction comparison, completion matrix, final report generation, English reading report, and master audit. The master audit passes and verifies all 21 checkpoint evals completed, the full public motion bundle was used, and no paper-level claim is made.

## Failed / Blocked Items

- This is local virtual checkpoint screening, not official BeyondMimic tracking evaluation.
- The screened checkpoints are local PPO checkpoints, not official released BeyondMimic teacher checkpoints.
- The result does not provide paper success/fall/collision metrics, DAgger logs, Fig. 5/Fig. 6 guided diffusion metrics, TensorRT deployment, or real robot evidence.
- `goal_complete` remains false.

## Effect on English Reading Report

This gives the reading report a stronger independent reproduction narrative: the recovered IsaacLab path can evaluate every saved checkpoint, and the local training dynamics suggest more iterations alone did not solve teacher quality. That is more informative than only reporting the final weak checkpoint.

## Next Step

Use the sweep to choose between two mainline options: run a 2048-env confirmation eval for iteration 300, or diagnose the PPO/tracking setup before spending more time on longer training.

## Git Commit

Pending at the time this progress note was written.
