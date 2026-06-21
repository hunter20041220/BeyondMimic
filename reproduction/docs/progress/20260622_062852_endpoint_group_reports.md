# Progress Update

## Goal

Refresh the course-facing English/Chinese reports, keep storage claims current, and move tracking data-quality work forward with a full endpoint-group termination ablation rather than another PPO rerun.

## Files Read

- `prompt06211658.txt`
- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/chinese_reading_report.md`
- `reproduction/docs/chinese_project_report.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_ee_termination_ablation/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_ee_termination_ablation.json`

## Files Modified

- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/chinese_reading_report.md`
- `reproduction/docs/chinese_project_report.md`
- `res/final_report/english_reading_report.md`
- `res/final_report/chinese_reading_report.md`
- `res/final_report/chinese_project_report.md`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`

## Commands Run

- `df -h .`
- `du -h -d 2 res logs reproduction tmp cache`
- `python3 -m py_compile reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_group_ablation.py`
- `BM_ENDPOINT_GROUP_ABLATION_NUM_ENVS=2048 BM_ENDPOINT_GROUP_ABLATION_SEED=20260721 envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_group_ablation.py`
- `python3 reproduction/scripts/paper_vs_reproduction_comparison.py`
- `python3 reproduction/scripts/artifact_manifest.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- `python3 reproduction/scripts/completion_matrix_status_audit.py`
- `python3 reproduction/scripts/verification_command_syntax_audit.py`
- `python3 reproduction/scripts/verification_command_script_manifest.py`
- `python3 reproduction/scripts/verification_command_coverage_audit.py`
- `python3 reproduction/scripts/required_artifact_absence_audit.py`
- `python3 reproduction/scripts/final_deliverables_audit.py`
- `python3 reproduction/scripts/progress_report_audit.py`
- `python3 reproduction/scripts/reproduction_master_audit.py`

## Results

The endpoint-group ablation completed all three 2048-env x 299-step variants with the same seed and checkpoint as the target-refresh/termination diagnostics.

- target-refresh done rate: `0.22340745192307693`
- ankles-only done rate: `0.1132420568561873`
- wrists-only done rate: `0.18382727581521738`
- all-endpoint relaxed done rate: `0.07152912050585285`
- ankles-only post-step0 active endpoint rate: `0.044885827390939596`
- wrists-only post-step0 active endpoint rate: `0.16343494389681207`
- dominant endpoint group: `wrists`

This makes the next tracking repair target more concrete: inspect wrist target body order, FK height, wrist/hand link naming, observation alignment, and endpoint termination semantics before another PPO run.

## Verification

Verification passed after integration:

- `paper_vs_reproduction`: `232` rows
- `artifact_manifest`: `1532` artifacts
- `reproduction_master_audit`: `ok`, `384/384` artifacts passed
- comparison counts: exactly comparable `58`, approximately comparable `19`, qualitative-only `142`, not publicly reproducible `10`, requires real robot `3`

## Failed / Blocked Items

No command failed in this round. Disk remains tight at about `51` GiB free on `/mnt/infini-data`. I did not delete current large LAFAN1 checkpoints, scaled teacher rollout shards, state-latent datasets, or current PPO checkpoints because they are still active local evidence, not failed artifacts. Failed-run evidence itself is small.

The endpoint-group ablation is diagnostic only. It removes official endpoint termination bodies per variant, so it is not a paper-level tracking score, not DAgger/VAE/diffusion evidence, and not real robot evidence.

## Effect on English Reading Report

The English and Chinese reading reports now state the current audit counts and include the wrist-dominant endpoint-gate diagnosis. The Chinese project report also better explains the project path from paper reading, formula-to-code decomposition, environment recovery, data substitution, tracking diagnostics, local downstream pipeline, storage management, and defense narrative.

## Next Step

Repair the wrist endpoint target/termination path before rerunning PPO. A good next gate is a live wrist-endpoint FK/body-order/height probe that compares motion target wrist positions, simulator wrist body positions, `MotionLoader` body indexes, and `ee_body_pos` z-threshold violations immediately after reset and after one policy step.

## Git Commit

Pending.
