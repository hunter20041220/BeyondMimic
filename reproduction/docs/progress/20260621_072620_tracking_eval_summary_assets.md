# Progress Update

## Goal

Create a compact, auditable report asset that links the current official-importer-export tracking evidence: full-dataset task diagnostic, scaled PPO checkpoint evaluation, and scaled local policy rollout video.

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
- `res/tracking/g1_official_importer_export_full_dataset_task_eval/tracking_g1_official_importer_export_full_dataset_task_eval.json`
- `res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json`
- `res/report_assets/official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/official_importer_export_full_bundle_scaled_ppo_checkpoint_eval_assets.json`
- `res/visualization/official_importer_export_full_bundle_scaled_ppo_policy_rollout/official_importer_export_full_bundle_scaled_ppo_policy_rollout_video_asset.json`

## Files Modified

- `reproduction/scripts/official_importer_export_tracking_eval_summary_assets.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/docs/completion_matrix.md`
- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/progress/20260621_072620_tracking_eval_summary_assets.md`

## Commands Run

- `jq '{status, checks, assets, metrics}' res/report_assets/official_importer_export_tracking_eval_summary/official_importer_export_tracking_eval_summary_assets.json`
- `envs/bm_analysis/bin/python -m py_compile reproduction/scripts/official_importer_export_tracking_eval_summary_assets.py`
- `envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_tracking_eval_summary_assets.py`
- `envs/bm_analysis/bin/python reproduction/scripts/progress_report_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/artifact_manifest.py`
- `envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py`
- `envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py`
- `envs/bm_analysis/bin/python reproduction/scripts/completion_matrix_status_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_syntax_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_script_manifest.py`
- `envs/bm_analysis/bin/python reproduction/scripts/verification_command_coverage_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/required_artifact_absence_audit.py`
- `envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py`

## Results

Generated a new report bundle under:

```text
res/report_assets/official_importer_export_tracking_eval_summary/
```

The summary links:

- 40/40 official-importer-export full-dataset task diagnostic with reward mean `0.023772245751250764`.
- Scaled iteration-999 PPO checkpoint evaluation with reward mean `0.02423080788881683`, body position error mean `0.6893615395727763`, and done count total `611642`.
- 299-frame scaled PPO policy-vs-reference video with reward mean `0.024693377315998077`.

## Verification

The new summary JSON reports `ok_official_importer_export_tracking_eval_summary_assets`. It checks that all report assets exist, the policy video exists, the scaled PPO eval uses the official-importer-export USD, the full-dataset task diagnostic passed 40/40 motions, and the asset does not claim paper-level tracking, Fig.5/Fig.6, or real-robot evidence.

One first verification pass failed in `paper_vs_reproduction_comparison.py` because I initially read `summary["claim_level"]` from the new JSON while the schema stores it under `summary["interpretation"]["claim_level"]`. I fixed the comparison and final-report readers, reran the verification chain, and the required scripts passed.

Final refreshed status:

- `artifact_manifest.py`: `ok`, `artifact_count=1250`, `missing_count=0`.
- `paper_vs_reproduction_comparison.py`: `ok`, `total_rows=198`; comparison types are `58 exactly_comparable`, `19 approximately_comparable`, `108 qualitative_only`, `10 not_publicly_reproducible`, and `3 requires_real_robot`.
- `final_reproduction_report.py`: `ok`.
- `completion_matrix_status_audit.py`: `ok`, `188` rows, `0` invalid statuses.
- `verification_command_syntax_audit.py`: `ok`, `189` scripts, `0` failed command parses.
- `verification_command_script_manifest.py`: `ok`, `189` scripts.
- `verification_command_coverage_audit.py`: `ok`, `197` commands, `10` smoke-pass entries.
- `required_artifact_absence_audit.py`: `ok`, `32` rows.
- `reproduction_master_audit.py`: `ok`, `328/328` artifacts passed.

## Failed / Blocked Items

No new execution failure occurred in this reporting step.

The remaining paper-level blockers are unchanged: no official BeyondMimic VAE/diffusion checkpoints, no real DAgger rollout logs, no real Fig.5/Fig.6 guided-diffusion rollout protocol/video, no TensorRT/asynchronous deployment result, and no real Unitree G1 hardware result. The current local tracking checkpoint is weak and cannot be treated as the paper tracking teacher.

## Effect on English Reading Report

The English report now has a concise tracking evidence bridge that explains what the virtual tracking pipeline can currently support and why it still falls short of paper-level reproduction.

## Next Step

Run the full verification chain, refresh generated audit/report artifacts, commit the changes, then continue toward stronger IsaacLab tracking gates or downstream report drafting.

## Git Commit

Pending at the time this progress note is written.
