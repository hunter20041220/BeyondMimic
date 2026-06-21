# Progress Update

## Goal

Move the reproduction back toward the main tracking bottleneck by diagnosing why the current robot-order FK PPO eval is still weak, and free safe disk space without deleting active evidence.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/current_project_reproduction_state_20260621.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/chinese_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/chinese_project_report.md`
- Robot-order FK PPO single and multi-seed eval JSON/CSV artifacts.
- Existing storage cleanup audit and large-run directory references.

## Files Modified

- Added `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/robot_order_fk_ppo_tracking_quality_diagnostic.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/cleanup_failed_large_artifacts.py`.
- Updated artifact manifest, paper-vs-reproduction comparison, final report generator, master audit, completion matrix, and reading/project reports.

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/robot_order_fk_ppo_tracking_quality_diagnostic.py
python3 reproduction/scripts/robot_order_fk_ppo_tracking_quality_diagnostic.py
python3 -m py_compile reproduction/scripts/cleanup_failed_large_artifacts.py
python3 reproduction/scripts/cleanup_failed_large_artifacts.py
```

Full verification was run after this progress file was written.

## Results

The tracking diagnostic found that the current robot-order FK PPO eval has a deterministic step-0 reset/bootstrap artifact: all three multi-seed runs report `2048/2048` done at step 0 and body-position error around `43.29` m. Removing step 0 lowers body-position error mean from about `0.360` to about `0.216`, but post-step0 done rate is still about `0.176`.

Storage cleanup deleted two superseded same-seed duplicate directories while retaining current active runs:

- `res/runs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset/resource_adjusted_teacher_rollout_20260620_195754_seed20260700`
- `res/runs/level_c_official_importer_export_scaled_ppo_teacher_rollout_state_latent_dataset/resource_adjusted_state_latent_dataset_20260621_042551_seed20260702`

Recorded freed candidate bytes: `2368915263`.

## Verification

Passed before commit:

```bash
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
python3 reproduction/scripts/required_artifact_absence_audit.py
```

Verification output summary:

- Tracking-quality diagnostic: `ok`, `4` rows.
- Cleanup audit: `ok`; after the first deletion run, repeated verification is idempotent and records `2` previously deleted superseded directories, `2368915263` known previously freed bytes.
- Artifact manifest: `ok`.
- Paper-vs-reproduction comparison: `ok`.
- Final reproduction report: `ok`.
- Completion matrix status audit: `ok`.
- Verification command syntax/script manifest/coverage audits: `ok`.
- Master audit: `ok`.
- Required artifact absence audit: `ok`, `32` rows.

## Failed / Blocked Items

- The current tracking teacher remains below paper-level quality.
- The immediate next technical blocker is reset/target alignment plus `ee_body_pos` termination, not downstream VAE/diffusion.
- No official BeyondMimic checkpoint, true DAgger logs, Fig. 5/Fig. 6 paper-level result, TensorRT result, or real robot result was produced.

## Effect on English Reading Report

The reading report can now explain why the current tracking result is incomplete in a sharper way: the local pipeline runs, but a reset/bootstrap spike and persistent post-step0 termination make the teacher unsuitable for final downstream reproduction.

## Next Step

Run a controlled reset/alignment probe before policy actions, including motion IDs, first target `body_pos_w`, runtime body names, endpoint body mapping, and `ee_body_pos` termination source.

## Git Commit

Planned commit message: `fix: add tracking quality diagnostic and cleanup`.
