# Progress Update

## Goal

Move the official-importer-export `Tracking-Flat-G1-v0` full public-motion task diagnostic onto the stronger full official-importer-export `csv_to_npz.py` / `replay_npz.py` loop outputs, then refresh report assets and audits. This is a full 40-motion task diagnostic, not a smoke test.

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
- `/mnt/infini-data/test/BeyondMimic/res/tracking/official_csv_to_npz_loop_full_dataset_with_official_importer_export/tracking_official_csv_to_npz_loop_full_dataset_with_official_importer_export_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/official_replay_npz_loop_full_dataset_with_official_importer_export/tracking_official_replay_npz_loop_full_dataset_with_official_importer_export_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_full_dataset_task_eval/tracking_g1_official_importer_export_full_dataset_task_eval.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_full_dataset_task_eval.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/official_importer_export_tracking_eval_summary_assets.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_full_dataset_task_eval/tracking_g1_official_importer_export_full_dataset_task_eval.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_full_dataset_task_eval/tracking_g1_official_importer_export_full_dataset_task_eval_rows.csv`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_full_dataset_task_eval/tracking_g1_official_importer_export_full_dataset_task_eval_rows.tsv`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_full_dataset_task_eval/*`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_tracking_eval_summary/*`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.*`
- `/mnt/infini-data/test/BeyondMimic/res/final_report/*`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/*`

## Commands Run

```bash
envs/bm_analysis/bin/python -m py_compile reproduction/scripts/tracking_g1_official_importer_export_full_dataset_task_eval.py
BM_OFFICIAL_IMPORTER_EXPORT_FULL_TASK_GPU=4 BM_OFFICIAL_IMPORTER_EXPORT_FULL_TASK_LIMIT=0 BM_OFFICIAL_IMPORTER_EXPORT_FULL_TASK_STALL_SECONDS=900 envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_full_dataset_task_eval.py
envs/bm_analysis/bin/python reproduction/scripts/official_importer_export_tracking_eval_summary_assets.py
envs/bm_analysis/bin/python reproduction/scripts/paper_vs_reproduction_comparison.py
envs/bm_analysis/bin/python reproduction/scripts/final_reproduction_report.py
envs/bm_analysis/bin/python reproduction/scripts/reproduction_master_audit.py
```

Full required verification was then rerun separately after this progress note.

## Results

The task diagnostic now consumes:

```text
res/tracking/official_csv_to_npz_loop_full_dataset_with_official_importer_export/
res/tracking/official_replay_npz_loop_full_dataset_with_official_importer_export/
```

instead of the older enriched-USD NPZ set.

Key full-dataset task-eval metrics:

```text
status: ok_official_importer_export_full_dataset_task_eval
motions: 40/40
failed_count: 0
total task steps: 11960
action dim: 29
policy obs dim: 160
critic obs dim: 286
reward terms: 9
termination terms: 4
robot contract: 29 joints / 40 bodies
reward mean: 0.060974017945530964
anchor-position error mean: 0.07482020696625113
body-position error mean: 0.11435768743976951
joint-position error mean: 0.7957518182694912
```

Report assets refreshed:

```text
res/report_assets/official_importer_export_full_dataset_task_eval/
res/report_assets/official_importer_export_tracking_eval_summary/
```

## Verification

The quick chain passed before the full required verification:

```text
official_importer_export_tracking_eval_summary_assets: ok
paper_vs_reproduction_comparison: ok
final_reproduction_report: ok
reproduction_master_audit: ok
```

## Failed / Blocked Items

- This still uses a captured official-importer-export USDA and generated official-importer-export loop NPZs, not unmodified live official converter-entry success.
- The diagnostic uses zero actions, so it is not a trained PPO teacher evaluation.
- It is not DAgger, not VAE/diffusion, not TensorRT deployment, not Fig. 5/Fig. 6 paper-level evidence, and not a real-robot result.
- `goal_complete` remains false.

## Effect on English Reading Report

The report can now state that the official-importer-export tracking task diagnostic is internally aligned with the official-importer-export full preprocessing/replay loop rather than mixing that asset path with older enriched-USD NPZ inputs. This makes the tracking setup section cleaner and gives a stronger virtual evidence chain for the reading report and PPT.

## Next Step

Use the now-aligned official-importer-export task diagnostic as the gate for further tracking work: either rerun scaled PPO checkpoint evaluation against this refreshed input chain if needed, or move to more paper-facing policy evaluation/video evidence while preserving the no-overclaim boundary.

## Git Commit

Pending at report creation time.
