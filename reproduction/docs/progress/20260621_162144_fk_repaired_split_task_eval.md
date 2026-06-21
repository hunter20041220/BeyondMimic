# Progress Update

## Goal

Move the tracking mainline beyond the current IsaacLab headless gate by rerunning the AppLauncher sentinel, then executing a full 40-motion `Tracking-Flat-G1-v0` task diagnostic on the FK-repaired per-motion public bundle using the captured official-importer-export G1 USDA. Also perform conservative storage cleanup for failed/superseded bulky artifacts.

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
- `reproduction/scripts/isaaclab_current_headless_gate.py`
- `reproduction/scripts/gpu_wangjc_process_guard.py`
- `reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_split_task_eval.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`

## Files Modified

- `reproduction/scripts/cleanup_failed_large_artifacts.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/progress/20260621_162144_fk_repaired_split_task_eval.md`
- `res/setup/isaaclab_current_headless_gate/isaaclab_current_headless_gate.json`
- `res/tracking/g1_official_importer_export_fk_repaired_split_task_eval/*`
- `res/report_assets/official_importer_export_fk_repaired_split_task_eval/*`
- `res/storage_cleanup/cleanup_failed_large_artifacts.json`
- refreshed comparison, final report, artifact manifest, verification, final-deliverables, and master-audit outputs.

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/gpu_wangjc_process_guard.py
BM_HEADLESS_GATE_GPU=4 envs/bm_analysis/bin/python reproduction/scripts/isaaclab_current_headless_gate.py
BM_FK_REPAIRED_SPLIT_TASK_GPU=4 BM_FK_REPAIRED_SPLIT_TASK_MAX_STEPS=299 BM_FK_REPAIRED_SPLIT_TASK_LIMIT=0 envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_split_task_eval.py
python3 reproduction/scripts/cleanup_failed_large_artifacts.py
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

- GPU guard dry-run found `0` wangjc guard targets before the run.
- Current IsaacLab headless gate passed with `gate_ok=true` on physical GPU 4.
- Full FK-repaired split task eval status: `ok_official_importer_export_fk_repaired_split_task_eval`.
- Motions evaluated: `40/40`.
- Failed rows: `0`.
- Total task steps: `11960`.
- Per-motion step bound: `299/299` for all rows.
- Robot/task contract checks passed for every row: action dim `29`, policy obs dim `160`, critic obs dim `286`, reward terms `9`, termination terms `4`, robot contract `29` joints / `40` bodies.
- Aggregate reward mean: `0.009514830887201243`.
- Aggregate final errors: anchor position mean `0.49441826045513154`, body position mean `0.5161251448094845`, joint position mean `0.9255241870880127`.
- Report assets generated:
  - `res/report_assets/official_importer_export_fk_repaired_split_task_eval/fk_repaired_split_task_eval_metrics.csv`
  - `res/report_assets/official_importer_export_fk_repaired_split_task_eval/fk_repaired_split_task_eval_completion_table.csv`
  - `res/report_assets/official_importer_export_fk_repaired_split_task_eval/fk_repaired_split_task_eval_reward_done.png`
  - `res/report_assets/official_importer_export_fk_repaired_split_task_eval/fk_repaired_split_task_eval_tracking_errors.png`
- Storage cleanup deleted the failed/superseded `res/tracking/g1_official_importer_export_fk_repaired_full_bundle_task_eval` working directory and retained large teacher/state-latent shards that are still referenced by current or historical downstream audit chains.

## Verification

The full verification chain passed:

- `artifact_manifest.py`: ok, 1381 artifacts.
- `paper_vs_reproduction_comparison.py`: ok.
- `final_reproduction_report.py`: ok.
- `completion_matrix_status_audit.py`: ok, 199 rows, 0 invalid statuses.
- `verification_command_syntax_audit.py`: ok, 199 scripts, 0 failed syntax checks.
- `verification_command_script_manifest.py`: ok, 199 scripts.
- `verification_command_coverage_audit.py`: ok, 207 commands.
- `reproduction_master_audit.py`: ok.

## Failed / Blocked Items

- This result is a zero-action task-contract diagnostic, not trained PPO tracking performance.
- The motion inputs are local FK-repaired per-motion NPZ candidates, not unmodified official `csv_to_npz.py` outputs.
- No real robot was used.
- Paper-level blockers remain: official BeyondMimic checkpoints, true DAgger logs, official Fig. 5/Fig. 6 rollout videos/metrics, TensorRT/asynchronous deployment evidence, and real-robot validation.
- Large teacher rollout/state-latent shard directories were not blindly deleted because active summaries and historical audits still reference them.

## Effect on English Reading Report

This round gives the report a stronger simulation-side tracking section: after proving the current IsaacLab headless gate, the project now has 40/40 all-motion official-importer-export task-contract evidence and report-ready reward/done plus tracking-error plots. It also documents a conservative storage hygiene policy for failed/superseded artifacts.

## Next Step

Use the 40/40 task-contract result as the gate for the next full tracking mainline: official-importer-export PPO tracking training/evaluation on GPUs 4 and 7 with GPU telemetry, checkpoint policy, and later policy rollout video capture.

## Git Commit

Pending at time of writing.
