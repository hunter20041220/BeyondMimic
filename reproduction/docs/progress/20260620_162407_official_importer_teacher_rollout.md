# Progress Update

## Goal

Add and audit a local teacher rollout dataset collected from the official-importer-export full-bundle PPO checkpoint, then connect the result to the reproducibility evidence chain and English reading report without claiming paper-level DAgger or Fig. 5/Fig. 6 reproduction.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `reproduction/scripts/tracking_g1_resource_adjusted_teacher_rollout_dataset.py`
- `reproduction/scripts/tracking_g1_official_importer_export_full_bundle_ppo_training_run.py`
- `reproduction/scripts/tracking_g1_official_importer_export_full_bundle_ppo_checkpoint_eval.py`

## Files Modified

- `reproduction/PROGRESS.md`
- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `res/final_report/english_reading_report.md`

## Files Added

- `reproduction/scripts/tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset.py`
- `reproduction/scripts/official_importer_export_full_bundle_teacher_rollout_report_assets.py`
- `res/tracking/g1_official_importer_export_full_bundle_teacher_rollout_dataset/tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset.json`
- `res/tracking/g1_official_importer_export_full_bundle_teacher_rollout_dataset/base_compatible_official_importer_export_training_run_for_teacher_rollout.json`
- `res/tracking/g1_official_importer_export_full_bundle_teacher_rollout_dataset/tracking_g1_resource_adjusted_teacher_rollout_worker.py`
- `res/report_assets/official_importer_export_full_bundle_teacher_rollout_dataset/README.md`
- `res/report_assets/official_importer_export_full_bundle_teacher_rollout_dataset/official_importer_export_full_bundle_teacher_rollout_report_assets.json`
- `res/report_assets/official_importer_export_full_bundle_teacher_rollout_dataset/teacher_rollout_reward_done_timeseries.png`
- `res/report_assets/official_importer_export_full_bundle_teacher_rollout_dataset/teacher_rollout_action_distribution.png`
- `res/report_assets/official_importer_export_full_bundle_teacher_rollout_dataset/teacher_rollout_motion_step_coverage.png`
- `res/report_assets/official_importer_export_full_bundle_teacher_rollout_dataset/teacher_rollout_shard_summary.csv`
- `res/report_assets/official_importer_export_full_bundle_teacher_rollout_dataset/teacher_rollout_action_summary.csv`

Large raw `.npz` shards were intentionally kept under ignored `res/runs/tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset/` and are not intended for GitHub.

## Commands Run

```bash
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset.py
CUDA_VISIBLE_DEVICES=4,7 envs/bm_tracking/bin/python reproduction/scripts/tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset.py
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/official_importer_export_full_bundle_teacher_rollout_report_assets.py
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_full_bundle_teacher_rollout_report_assets.py
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/artifact_manifest.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/final_reproduction_report.py reproduction/scripts/reproduction_master_audit.py reproduction/scripts/official_importer_export_full_bundle_teacher_rollout_report_assets.py reproduction/scripts/tracking_g1_official_importer_export_full_bundle_teacher_rollout_dataset.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

## Results

- Teacher rollout status: `ok_official_importer_export_full_bundle_teacher_rollout_dataset_completed`.
- GPUs: physical GPUs `4` and `7`, `CUDA_VISIBLE_DEVICES=4,7`.
- Scope: `1024` total environments, `299` rollout steps, `306176` total virtual environment steps.
- Dataset: `2` raw `.npz` shards, `479719377` compressed bytes under ignored `res/runs`.
- Motion bundle: `40` public motions, `11960` source motion frames.
- Reward mean by rank: `[0.023514889180660248, 0.02330510877072811]`.
- Done count: `305635`; timeout count: `0`.
- GPU telemetry: peak memory around `3785` MiB on GPU4 and `3777` MiB on GPU7, so this is documented as rollout-data collection rather than a high-memory formal training run.
- Report assets added: reward/done trace, action distribution plot, motion-step coverage plot, shard summary CSV, action summary CSV.

## Verification

All required verification commands passed.

- `artifact_manifest.py`: `ok`, `825` artifacts.
- `paper_vs_reproduction_comparison.py`: `ok`, `172` rows after the new qualitative-only teacher-data row.
- `final_reproduction_report.py`: `ok`.
- `completion_matrix_status_audit.py`: `ok`, `170` rows, `0` invalid statuses.
- `verification_command_syntax_audit.py`: `ok`, `186` scripts.
- `verification_command_script_manifest.py`: `ok`, `186` scripts.
- `verification_command_coverage_audit.py`: `ok`, `194` commands.
- `reproduction_master_audit.py`: `ok`.

## Failed / Blocked Items

- This is not the official BeyondMimic DAgger dataset.
- This is not a paper-scale tracking teacher checkpoint.
- The high done count indicates weak short local episodes and should not be read as paper success.
- Official VAE/diffusion checkpoints, true DAgger logs, Fig. 5/Fig. 6 closed-loop videos/metrics, TensorRT/asynchronous deployment, and real robot evidence remain incomplete.
- `goal_complete` remains `false`.

## Effect on English Reading Report

The report can now describe a stronger local teacher-data bridge on the official-importer-export robot asset path: official-importer-export G1 USDA -> 40-motion public bundle -> 300-iteration local PPO checkpoint -> two-shard local virtual teacher rollout dataset. This helps explain the paper's DAgger/teacher-trajectory prerequisite while preserving the boundary between local qualitative evidence and official paper-level results.

## Next Step

Use this more official local teacher dataset as the candidate source for downstream local VAE/state-latent/diffusion/guidance checks only after explicitly deciding whether to replace or compare against the existing enriched-USD/official-loop downstream chain. Continue to avoid any real-robot claim unless hardware is explicitly confirmed.

## Git Commit

Pending at the time this progress note was written.
