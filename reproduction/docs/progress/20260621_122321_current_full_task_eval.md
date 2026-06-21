# Progress Update

## Goal

Move from the repaired current IsaacLab/G1 task-construction gate into the main tracking reproduction path by rerunning the full public-motion `Tracking-Flat-G1-v0` task diagnostic on the official-importer G1 USDA path.

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
- `reproduction/scripts/tracking_g1_official_importer_export_full_dataset_task_eval.py`
- `reproduction/scripts/tracking_g1_official_importer_export_task_smoke.py`
- `reproduction/scripts/artifact_manifest.py`

## Files Modified

- `reproduction/scripts/artifact_manifest.py`
- `reproduction/docs/progress/20260621_122321_current_full_task_eval.md`
- `res/tracking/g1_official_importer_export_full_dataset_task_eval/tracking_g1_official_importer_export_full_dataset_task_eval.json`
- `res/tracking/g1_official_importer_export_full_dataset_task_eval/tracking_g1_official_importer_export_full_dataset_task_eval_rows.csv`
- `res/tracking/g1_official_importer_export_full_dataset_task_eval/tracking_g1_official_importer_export_full_dataset_task_eval_rows.tsv`
- `res/report_assets/official_importer_export_full_dataset_task_eval/official_importer_export_full_dataset_task_eval_metrics.csv`
- `res/report_assets/official_importer_export_full_dataset_task_eval/official_importer_export_full_dataset_task_eval_plot_rows.json`

## Commands Run

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -i 4,7
BM_OFFICIAL_IMPORTER_EXPORT_FULL_TASK_GPU=4 envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_full_dataset_task_eval.py
```

## Results

- The current full public-motion task diagnostic completed successfully on physical GPU 4.
- Summary JSON: `res/tracking/g1_official_importer_export_full_dataset_task_eval/tracking_g1_official_importer_export_full_dataset_task_eval.json`
- Status: `ok_official_importer_export_full_dataset_task_eval`
- Motions: `40/40` ok.
- Total steps: `11960`.
- Per-motion step count: all `299`.
- Contracts verified across all rows: action dim `29`, policy observation dim `160`, critic observation dim `286`, reward terms `9`, termination terms `4`, robot joints `29`, robot bodies `40`.
- Aggregate reward mean: `0.060974017945530964`.
- Aggregate final anchor/body/joint position errors: anchor mean `0.07482020696625113`, body mean `0.11435768743976951`, joint mean `0.7957518182694912`.
- GPU guard record: `res/gpu_guard/20260621_115519_gpu47_wangjc_official_csv_task_eval_guard.json`; no target GPU 4/7 wangjc process was killed in this run.

## Verification

Final verification chain for this round passed:

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

Observed results:

- `artifact_manifest.py`: `ok`, `1361` artifacts.
- `paper_vs_reproduction_comparison.py`: `ok`.
- `final_reproduction_report.py`: `ok`.
- `completion_matrix_status_audit.py`: `ok`, `199` rows, `0` invalid statuses.
- `verification_command_syntax_audit.py`: `ok`, `199` scripts, `0` failures.
- `verification_command_script_manifest.py`: `ok`, `199` scripts.
- `verification_command_coverage_audit.py`: `ok`, `207` commands, `10` smoke pass references.
- `reproduction_master_audit.py`: `ok`.

## Failed / Blocked Items

- No failed rows were produced by the 40-motion task eval.
- This is still a zero-action diagnostic, not trained PPO policy evaluation, not DAgger rollout collection, not VAE/diffusion closed-loop control, not Fig. 5/Fig. 6 paper-level reproduction, not TensorRT/asynchronous deployment, and not real-robot evidence.
- No new MP4 was generated in this round because the script produces task metrics and report plots. The next policy/replay visualization round should generate a robot-motion MP4 after a meaningful rollout or reference replay gate.

## Effect on English Reading Report

This round strengthens the code reproduction section substantially: after repairing the live gate, the current environment now executes all 40 public motions through the official `Tracking-Flat-G1-v0` task stack on the official-importer G1 USDA path and generates report-ready CSV/PNG assets. It gives the report a concrete full-data virtual tracking diagnostic while keeping the claim boundary honest.

## Next Step

Use the current full-task gate as the prerequisite for the next mainline step: run policy/replay visualization or current-GPU PPO/checkpoint evaluation on GPU 4/7, and generate a report-ready MP4 plus tracking-error plot.

## Git Commit

Pending.
