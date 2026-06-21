# Progress Update

## Goal

Promote the latest robot-order FK tracking data-quality repair into the comparison/report evidence, update the English and Chinese report narratives for defense, and perform conservative storage hygiene without deleting experiment evidence.

## Files Read

- `prompt06211658.txt`
- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/chinese_reading_report.md`
- `reproduction/docs/chinese_project_report.md`
- `reproduction/docs/final_reproduction_report.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `res/tracking/fk_repaired_data_quality_gate/fk_repaired_data_quality_gate.json`
- `res/tracking/fk_repaired_body_order_runtime_probe/fk_repaired_body_order_runtime_probe.json`
- `res/tracking/g1_official_importer_export_fk_repaired_robot_order_split_task_eval/tracking_g1_official_importer_export_fk_repaired_robot_order_split_task_eval.json`

## Files Modified

- `reproduction/scripts/tracking_fk_repaired_data_quality_gate.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/chinese_reading_report.md`
- `reproduction/docs/chinese_project_report.md`
- `res/tracking/fk_repaired_data_quality_gate/fk_repaired_data_quality_gate.json`
- `res/tracking/fk_repaired_data_quality_gate/fk_repaired_data_quality_gate.md`
- `res/tracking/fk_repaired_data_quality_gate/fk_repaired_data_quality_gate.tsv`

## New Files

- `reproduction/scripts/tracking_fk_repaired_body_order_runtime_probe.py`
- `reproduction/scripts/tracking_g1_official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz.py`
- `reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_split_task_eval.py`
- `res/tracking/fk_repaired_body_order_runtime_probe/fk_repaired_body_order_runtime_probe.json`
- `res/tracking/fk_repaired_body_order_runtime_probe/fk_repaired_body_order_runtime_probe_rows.tsv`
- `res/tracking/fk_repaired_body_order_runtime_probe/fk_repaired_body_order_runtime_probe.md`
- `res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/tracking_g1_official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz.json`
- `res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/README.md`
- `res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/fk_repaired_robot_order_alignment.tsv`
- `res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/fk_repaired_robot_order_motion_rows.csv`
- `res/tracking/official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz/fk_repaired_robot_order_motion_rows.tsv`
- `res/tracking/g1_official_importer_export_fk_repaired_robot_order_split_task_eval/tracking_g1_official_importer_export_fk_repaired_robot_order_split_task_eval.json`
- `res/tracking/g1_official_importer_export_fk_repaired_robot_order_split_task_eval/tracking_g1_official_importer_export_fk_repaired_robot_order_split_task_eval_rows.csv`
- `res/tracking/g1_official_importer_export_fk_repaired_robot_order_split_task_eval/tracking_g1_official_importer_export_fk_repaired_robot_order_split_task_eval_rows.tsv`
- `res/report_assets/official_importer_export_fk_repaired_robot_order_split_task_eval/`
- `res/storage_cleanup/storage_cleanup_20260621_robot_order_fk.md`

Large `.npz` files were generated for the robot-order bundle but are intentionally not intended for GitHub.

## Commands Run

```bash
python3 -m py_compile reproduction/scripts/tracking_fk_repaired_body_order_runtime_probe.py reproduction/scripts/tracking_g1_official_csv_loop_full_bundle_fk_repaired_robot_order_motion_npz.py reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_split_task_eval.py reproduction/scripts/tracking_fk_repaired_data_quality_gate.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/tracking_fk_repaired_data_quality_gate.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
du -h -d 2 .
find . -path './download' -prune -o -path './other' -prune -o -path './envs' -prune -o -type f -size +500M -printf '%s\t%p\n'
find res logs tmp cache reproduction -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' \) -prune -exec rm -rf {} +
```

## Results

The live body-order probe showed that the prior FK-repaired bundle used URDF body order while the official IsaacLab `MotionLoader` indexes `body_pos_w` by runtime robot body order. The robot-order FK bundle is the current best tracking-data input candidate.

Full 40-motion zero-action split diagnostic:

- Old FK split done/termination: `11958/11960`.
- Robot-order FK split done/termination: `2166/11960`.
- Mean anchor error improved from about `0.494` to `0.084`.
- Mean body-position error improved from about `0.516` to `0.214`.
- Mean reward improved from about `0.0095` to `0.0161`.
- 40/40 motion rows completed without script failure.

The reading reports now describe this as a mainline tracking data-quality repair, while still stating that it is not paper-level teacher performance.

## Verification

Completed so far:

- Script compile check passed.
- `tracking_fk_repaired_data_quality_gate.py` passed with `ok_fk_repaired_data_quality_gate`.
- `paper_vs_reproduction_comparison.py` passed with `ok`.
- `final_reproduction_report.py` passed with `ok`.

The full required verification chain is run after this progress note is written.

## Failed / Blocked Items

- No new PPO run was launched in this round.
- The old FK-repaired PPO checkpoint remains a weak teacher and should not feed downstream DAgger/VAE/diffusion.
- Robot-order FK repair is a data-quality/task diagnostic; it is not a trained policy result.
- Official VAE/diffusion checkpoints, true DAgger logs, Fig. 5/Fig. 6 paper rollouts, TensorRT engine, and real robot evidence remain absent.

## Effect on English Reading Report

The English and Chinese reading reports now frame the project as a public-resource, auditable partial reproduction with a local virtual BeyondMimic-like pipeline. They emphasize the new body-order root cause, the improved robot-order FK task diagnostic, and the next step of training a stronger PPO teacher from that repaired input.

## Next Step

Run the full verification chain, commit and push this report/evidence update, then start the next mainline experiment: robot-order FK full PPO on GPU 4/7 with resource logging, checkpoint eval, curves, and policy video.

## Git Commit

Pending at the time this note was created.
