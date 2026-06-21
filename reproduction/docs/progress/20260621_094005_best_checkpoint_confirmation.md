# Progress Update

## Goal

Confirm whether the checkpoint selected by the scaled PPO sweep is actually better than the final iteration-999 checkpoint when evaluated at the full 2048-env scale.

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
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_sweep/tracking_g1_official_importer_export_scaled_ppo_checkpoint_sweep.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/.gitignore`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_best_checkpoint_confirmation_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`

## Commands Run

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_best_checkpoint_confirmation_eval.py
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

The full-size confirmation eval ran iteration `300`, selected by the screening sweep, at the same `2048` env x `299` step scale as the final iteration-999 checkpoint.

```text
best_iteration: 300
best_reward_mean: 0.023709370602034405
final_iteration_999_reward_mean: 0.02423080788881683
reward_delta_best_minus_final: -0.0005214372867824238
body_error_delta_best_minus_final: 0.025006858002780685
joint_error_delta_best_minus_final: 0.024168011336821005
best_local_non_timeout_done_rate: 0.9985874137750836
final_local_non_timeout_done_rate: 0.9988405361622074
```

The confirmation result reverses the simple screening interpretation: iteration 300 does not beat the final checkpoint when both are evaluated at full scale.

## Verification

The result is included in artifact manifest, paper-vs-reproduction comparison, completion matrix, final report generation, English reading report, and master audit. The master audit verifies the 2048-env shape, official-importer-export asset path, full 40-motion bundle, iteration 300 vs 999 comparison, and no paper-level claim.

## Failed / Blocked Items

- This is local virtual checkpoint confirmation, not official BeyondMimic tracking evaluation.
- The checkpoint is a local PPO checkpoint, not an official released BeyondMimic teacher checkpoint.
- The result does not provide paper success/fall/collision metrics, DAgger logs, Fig. 5/Fig. 6 guided diffusion metrics, TensorRT deployment, or real robot evidence.
- `goal_complete` remains false.

## Effect on English Reading Report

This strengthens the report's discussion of reproduction methodology: a small screening sweep alone was not enough, and full-scale confirmation showed that the final checkpoint remains at least as good as the sweep-selected checkpoint. It supports a mature negative finding rather than overclaiming a better teacher.

## Next Step

Diagnose the PPO teacher quality itself: inspect reward/termination components and training configuration, then decide whether a tuned PPO rerun is more valuable than continuing downstream from the weak local teacher.

## Git Commit

Pending at the time this progress note was written.
