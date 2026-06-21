# Progress Update

## Goal

After the current headless gate and full replay refresh passed, re-run the official-importer-export full-dataset `Tracking-Flat-G1-v0` task diagnostic over all 40 public motion NPZs. The purpose is to verify task construction, reset, stepping, action/observation dimensions, reward terms, termination terms, robot contract, and report-ready tracking diagnostic plots.

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
- `reproduction/scripts/tracking_g1_official_importer_export_task_smoke.py`
- `reproduction/scripts/tracking_g1_official_importer_export_full_dataset_task_eval.py`

## Files Modified

- `res/tracking/g1_official_importer_export_full_dataset_task_eval/tracking_g1_official_importer_export_full_dataset_task_eval.json`
- `res/tracking/g1_official_importer_export_full_dataset_task_eval/tracking_g1_official_importer_export_full_dataset_task_eval_rows.csv`
- `res/tracking/g1_official_importer_export_full_dataset_task_eval/tracking_g1_official_importer_export_full_dataset_task_eval_rows.tsv`
- `res/tracking/g1_official_importer_export_full_dataset_task_eval/motions/`
- `res/report_assets/official_importer_export_full_dataset_task_eval/`
- `res/visual_media_inventory/`
- `res/report_assets/visual_evidence_index/`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/docs/progress/20260621_135758_full_task_eval_refresh.md`

## Commands Run

```bash
git status --short
git log -3 --oneline
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits -i 4,7
python3 reproduction/scripts/tracking_g1_official_importer_export_full_dataset_task_eval.py
envs/bm_analysis/bin/python reproduction/scripts/visual_media_inventory_audit.py
envs/bm_analysis/bin/python reproduction/scripts/visual_evidence_index.py
```

## Results

- Re-ran the full 40-motion official-importer-export task diagnostic on GPU 4.
- Status: `ok_official_importer_export_full_dataset_task_eval`.
- `row_count=40`, `ok_count=40`, `failed_count=0`.
- `total_steps=11960`.
- `total_done_count=1255`.
- Mean reward: `0.060974017945530964`.
- Mean final command errors:
  - anchor position: `0.07482020696625113`
  - body position: `0.11435768743976951`
  - joint position: `0.7957518182694912`
- Contract checks passed:
  - action dimension `29`
  - policy observation dimension `160`
  - critic observation dimension `286`
  - reward terms `9`
  - termination terms `4`
  - robot joints/bodies `29/40`
  - all rows use the official-importer-export USDA

Report assets refreshed:

- `res/report_assets/official_importer_export_full_dataset_task_eval/official_importer_export_full_dataset_task_eval_reward_done.png`
- `res/report_assets/official_importer_export_full_dataset_task_eval/official_importer_export_full_dataset_task_eval_tracking_errors.png`
- `res/report_assets/official_importer_export_full_dataset_task_eval/official_importer_export_full_dataset_task_eval_metrics.csv`
- `res/report_assets/official_importer_export_full_dataset_task_eval/official_importer_export_full_dataset_task_eval_completion_table.csv`

## Verification

Pre-final verification in this round passed:

```bash
envs/bm_analysis/bin/python reproduction/scripts/visual_media_inventory_audit.py
envs/bm_analysis/bin/python reproduction/scripts/visual_evidence_index.py
```

Results:

- `visual_media_inventory_audit.py`: `status=ok`, `471` visual media rows, `85` local video rows.
- `visual_evidence_index.py`: `status=ok`, `31` report-ready MP4 rows indexed.

The full repository verification chain passed after this file was written:

```bash
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

Key full-chain results:

- `artifact_manifest.py`: `status=ok`, `1364` artifacts.
- `paper_vs_reproduction_comparison.py`: `status=ok`.
- `final_reproduction_report.py`: `status=ok`.
- `completion_matrix_status_audit.py`: `status=ok`, `199` rows, `0` invalid statuses.
- `verification_command_syntax_audit.py`: `status=ok`, `199` scripts, `0` failures.
- `verification_command_script_manifest.py`: `status=ok`.
- `verification_command_coverage_audit.py`: `status=ok`, `207` commands categorized, `10` smoke commands passed.
- `reproduction_master_audit.py`: `status=ok`.

## Failed / Blocked Items

- No failure occurred in the full task diagnostic.
- This is still a zero-action diagnostic, not a trained PPO policy evaluation.
- It does not claim paper-level tracking performance, Fig. 5/Fig. 6 guided diffusion, TensorRT deployment, DAgger rollout data, or real-robot validation.

## Effect on English Reading Report

This gives the reading report a stronger Level-B tracking section: the current server can instantiate the official-importer-export G1 task, reset and step it for all 40 public motions, verify the task contract, and produce reward/done plus tracking-error plots for presentation.

## Next Step

Proceed to PPO tracking training/evaluation or scaled PPO checkpoint evaluation. Formal PPO GPU experiments should use GPU 4 and GPU 7, record seed/config/runtime/GPU memory, and only kill external `wangjc` GPU processes after command-line verification.

## Git Commit

Pending at file creation time; see Git history for the commit containing this progress update.
