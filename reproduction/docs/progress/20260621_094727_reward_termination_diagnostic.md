# Progress Update

## Goal

Diagnose why the local scaled PPO tracking teacher remains weak after checkpoint sweep and full-size best-checkpoint confirmation, focusing on reward and termination components rather than only aggregate reward.

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
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_best_checkpoint_confirmation_eval/tracking_g1_official_importer_export_scaled_ppo_best_checkpoint_confirmation_eval.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_scaled_ppo_reward_termination_diagnostic.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`

## Commands Run

```bash
/mnt/infini-data/test/BeyondMimic/envs/bm_analysis/bin/python /mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_scaled_ppo_reward_termination_diagnostic.py
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

The diagnostic compares the full-size iteration-300 and iteration-999 scaled PPO evals and extracts logged reward, termination, and motion-error components.

```text
reward_component_rows: 18
termination_component_rows: 8
motion_metric_rows: 26
dominant termination, iteration 300: ee_body_pos
dominant termination fraction, iteration 300: 0.9985874137750836
dominant termination, iteration 999: ee_body_pos
dominant termination fraction, iteration 999: 0.9988405361622074
```

The key finding is that `ee_body_pos` dominates non-timeout termination for both local checkpoints. This explains why checkpoint selection did not resolve teacher weakness.

## Verification

The diagnostic is included in artifact manifest, paper-vs-reproduction comparison, completion matrix, final report generation, English reading report, and master audit. Master audit checks the component counts, dominant termination component, high termination fractions, output files, PNG assets, and no paper-level claim.

## Failed / Blocked Items

- This is local virtual diagnostic evidence, not official BeyondMimic tracking evaluation.
- It does not provide official success/fall/collision metrics, official DAgger logs, Fig. 5/Fig. 6 guided diffusion metrics, TensorRT deployment, or real robot evidence.
- `goal_complete` remains false.

## Effect on English Reading Report

The reading report can now explain the failure mode concretely: the recovered pipeline runs, but the local public-data PPO teacher is terminated almost entirely by an end-effector/body-position tracking condition. This supports a thoughtful reproducibility discussion rather than a simple pass/fail summary.

## Next Step

Inspect body index mapping, retargeted target-body trajectories, and termination thresholds for `ee_body_pos`; then decide whether to patch/tune tracking eval or rerun PPO with adjusted termination/reward settings.

## Git Commit

Pending at the time this progress note was written.
