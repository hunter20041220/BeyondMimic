# Progress Update

## Goal

Collect and audit a stronger local teacher rollout dataset from the latest official-importer-export scaled PPO checkpoint, then wire the result into reports, comparison tables, and master verification without claiming official BeyondMimic DAgger reproduction.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_full_bundle_scaled_ppo_teacher_rollout_report_assets.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_full_bundle_teacher_rollout_report_assets.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/required_artifact_absence_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`

## Commands Run

```bash
BM_OFFICIAL_IMPORTER_EXPORT_SCALED_TEACHER_ROLLOUT_NUM_ENVS_PER_RANK=2048 \
BM_OFFICIAL_IMPORTER_EXPORT_SCALED_TEACHER_ROLLOUT_SEED=20260700 \
envs/bm_analysis/bin/python \
  reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset.py

envs/bm_analysis/bin/python \
  reproduction/scripts/official_importer_export_full_bundle_scaled_ppo_teacher_rollout_report_assets.py
```

## Results

- Dataset status: `ok_official_importer_export_full_bundle_scaled_ppo_teacher_rollout_dataset_completed`
- GPUs: physical `[4,7]`
- Checkpoint source: local official-importer-export scaled PPO iteration `999`
- Total envs: `4096`
- Rollout steps: `299`
- Total virtual env steps: `1224704`
- Raw shards: `2`
- Motion count: `40`
- Total source motion frames: `11960`
- Raw shard bytes: `1919836221`
- Reward mean by rank: `[0.024104224517941475, 0.02374308556318283]`
- Report-asset reward mean over steps: `0.02392365585575037`
- Done count total: `1223466`
- Peak GPU memory: GPU4 `4847` MiB, GPU7 `4839` MiB

## Verification

Full verification passed in this work round:

```bash
python3 reproduction/scripts/progress_report_audit.py
python3 reproduction/scripts/required_artifact_absence_audit.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

Observed final statuses:

- `progress_report_audit.py`: `ok`, `38` rows
- `required_artifact_absence_audit.py`: `ok`, `30` rows
- `artifact_manifest.py`: `ok`, `1157` artifacts
- `paper_vs_reproduction_comparison.py`: `ok`
- `final_reproduction_report.py`: `ok`
- `completion_matrix_status_audit.py`: `ok`, `183` rows
- `verification_command_syntax_audit.py`: `ok`, `189` scripts
- `verification_command_script_manifest.py`: `ok`, `189` scripts
- `verification_command_coverage_audit.py`: `ok`, `197` commands
- `reproduction_master_audit.py`: `ok`, `goal_complete=false`

## Failed / Blocked Items

- This is not an official BeyondMimic DAgger rollout dataset.
- The source checkpoint is a weak local PPO checkpoint, not the official tracking teacher.
- Peak GPU memory stayed below the requested 10GB/card formal threshold.
- Downstream VAE/state-latent/diffusion artifacts have not yet been retrained from this larger dataset.
- Fig. 5/Fig. 6 paper-level closed-loop guidance, TensorRT deployment, and real-robot results remain blocked or unavailable.

## Effect on English Reading Report

The English report can now describe a larger and more recent local teacher-data candidate on the official-importer-export asset path. It should present this as stronger engineering evidence for the reproduction pipeline while explicitly separating it from official DAgger logs and from the older downstream VAE/diffusion metrics.

## Next Step

Run the full audit suite, refresh generated reports, commit the result, attempt GitHub push, and then consider retraining the VAE/state-latent/diffusion chain from this scaled teacher rollout dataset.

## Git Commit

Prepared for commit `report: add scaled teacher rollout dataset`; the final response records the actual commit hash.
