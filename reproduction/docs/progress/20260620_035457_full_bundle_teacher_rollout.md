# Progress Update

## Goal

Extend the local virtual tracking evidence from full-public-motion PPO checkpoint evaluation to a full-bundle teacher rollout dataset that can support downstream VAE/state-latent experiments and the English reading report, while preserving the non-official/non-paper-level claim boundary.

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
- Existing teacher rollout and full-bundle PPO scripts/audits under `/mnt/infini-data/test/BeyondMimic/reproduction/scripts` and `/mnt/infini-data/test/BeyondMimic/res/tracking`.

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_csv_loop_full_bundle_teacher_rollout_report_assets.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/final_report/english_reading_report.md`
- Regenerated audit/report outputs under `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest`, `/mnt/infini-data/test/BeyondMimic/res/comparison`, `/mnt/infini-data/test/BeyondMimic/res/final_report`, `/mnt/infini-data/test/BeyondMimic/res/docs`, `/mnt/infini-data/test/BeyondMimic/res/verification_command_*`, `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence`, `/mnt/infini-data/test/BeyondMimic/res/final_deliverables_audit`, and `/mnt/infini-data/test/BeyondMimic/res/master_audit`.

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/official_csv_loop_full_bundle_teacher_rollout_report_assets.py
BM_OFFICIAL_CSV_LOOP_FULL_BUNDLE_TEACHER_ROLLOUT_SEED=20260672 python3 reproduction/scripts/tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset.py
envs/bm_analysis/bin/python reproduction/scripts/official_csv_loop_full_bundle_teacher_rollout_report_assets.py
python3 -m py_compile reproduction/scripts/artifact_manifest.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/final_reproduction_report.py reproduction/scripts/reproduction_master_audit.py reproduction/scripts/tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset.py reproduction/scripts/official_csv_loop_full_bundle_teacher_rollout_report_assets.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/required_artifact_absence_audit.py
python3 reproduction/scripts/final_deliverables_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

## Results

- Full-bundle teacher rollout status: `ok_official_csv_loop_full_bundle_teacher_rollout_dataset_completed`.
- GPUs: physical GPUs `[4, 7]`, `CUDA_VISIBLE_DEVICES=4,7`, world size `2`.
- Rollout scope: `1024` total environments, `299` steps, `306176` total virtual env steps.
- Motion source: full public official-loop bundle with `40` motions and `11960` frames.
- Raw rollout shards: `2`, retained under ignored `/mnt/infini-data/test/BeyondMimic/res/runs/tracking_g1_official_csv_loop_full_bundle_teacher_rollout_dataset/`.
- Raw compressed dataset size: `531492516` bytes, intentionally not committed.
- Aggregate metrics: done count `26743`, timeout count `0`, reward mean by rank `[0.023176534101366997, 0.022934164851903915]`.
- GPU telemetry: mean utilization about `92.5%` on GPU 4 and `92.1%` on GPU 7; peak memory about `7243 MB` and `7235 MB`.
- Report assets created under `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_full_bundle_teacher_rollout_dataset/`.
- Updated audit counts: artifact manifest `602` artifacts; paper-vs-reproduction `159` rows; master audit `268/268` passed.

## Verification

All required verification commands passed in this round:

- `artifact_manifest.py`: `ok`, 602 artifacts.
- `paper_vs_reproduction_comparison.py`: `ok`, 159 rows.
- `final_reproduction_report.py`: `ok`.
- `completion_matrix_status_audit.py`: `ok`, 170 rows.
- `verification_command_syntax_audit.py`: `ok`, 185 scripts.
- `verification_command_script_manifest.py`: `ok`, 185 scripts.
- `verification_command_coverage_audit.py`: `ok`, 193 commands.
- `required_artifact_absence_audit.py`: `ok`, 26 rows.
- `final_deliverables_audit.py`: `ok`, 38 rows.
- `reproduction_master_audit.py`: `ok`, 268/268 artifacts passed.

## Failed / Blocked Items

No new command failed in this round. The full-bundle teacher rollout remains local virtual evidence only. It is not the official BeyondMimic DAgger rollout dataset, not an official paper-scale teacher checkpoint, not Fig. 5/Fig. 6 closed-loop guided diffusion, and not real-robot evidence. The raw rollout shards are intentionally ignored as large local artifacts.

## Effect on English Reading Report

The English report can now describe a stronger teacher-data bridge: the project moved from single-motion teacher rollout evidence to a 40-motion public-bundle rollout dataset with audited GPU use, shard count, motion coverage, and report-ready plots. This improves the code-reproduction section while preserving the key statement that the project does not fully reproduce BeyondMimic at paper level.

## Next Step

Use this full-bundle teacher rollout dataset to build a full-bundle state-latent dataset, retrain/evaluate VAE or diffusion components on the broader local trajectory source, and then compare against the existing single-motion downstream pipeline. Continue to avoid real-robot claims unless Unitree G1 hardware is explicitly confirmed.

## Git Commit

Pending at the time this progress note was written.
