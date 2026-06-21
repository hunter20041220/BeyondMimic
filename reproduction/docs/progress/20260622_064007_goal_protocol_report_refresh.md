# Progress Update

## Goal

Refresh the current project objective and report materials around the real next blocker: tracking teacher quality, especially the wrist-dominant `ee_body_pos` endpoint termination issue. Keep the course-facing reports honest: strong public-resource partial reproduction, not full BeyondMimic paper-level reproduction.

## Files Read

- `prompt06211658.txt`
- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/current_project_reproduction_state_20260622.md`
- `reproduction/docs/current_project_reproduction_summary_20260622.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `res/report_assets/unified_local_task_protocol/unified_local_task_protocol.json`
- `res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_group_ablation/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_endpoint_group_ablation.json`

## Files Modified

- `goal.md`
- `README.md`
- `reproduction/scripts/unified_local_task_protocol_table.py`
- `reproduction/scripts/update_course_reports.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/chinese_reading_report.md`
- `reproduction/docs/chinese_project_report.md`
- `res/final_report/english_reading_report.md`
- `res/final_report/chinese_reading_report.md`
- `res/final_report/chinese_project_report.md`
- `res/report_assets/unified_local_task_protocol/unified_local_task_protocol.json`
- `res/report_assets/unified_local_task_protocol/unified_local_task_protocol.csv`
- `res/report_assets/unified_local_task_protocol/unified_local_task_protocol.md`
- `res/storage_cleanup/cleanup_failed_large_artifacts.json`

## Commands Run

- `git status --short`
- `sed` and `rg` inspections over reports, scripts, audits, and tracking diagnostics
- `df -h /mnt/infini-data/test/BeyondMimic`
- `du -sh download other reproduction res logs envs cache tmp`
- `python3 -m py_compile reproduction/scripts/unified_local_task_protocol_table.py reproduction/scripts/update_course_reports.py`
- `envs/bm_analysis/bin/python reproduction/scripts/unified_local_task_protocol_table.py`
- `envs/bm_analysis/bin/python reproduction/scripts/update_course_reports.py`
- `envs/bm_analysis/bin/python reproduction/scripts/cleanup_failed_large_artifacts.py`

## Results

- Updated the top-level goal to the current 2026-06-22 baseline: master audit `385/385`, artifact manifest `1533`, comparison rows `232`, completion counts `74 complete / 132 partial / 2 blocked / 1 out_of_scope`, and `goal_complete=false`.
- Added the wrist-dominant endpoint-group ablation as the current tracking repair target: target-refresh done rate `0.22340745192307693`, ankles-only `0.1132420568561873`, wrists-only `0.18382727581521738`, all-endpoint-relaxed `0.07152912050585285`.
- Updated README status so the active blocker is no longer described as IsaacLab import/headless setup, but as tracking teacher quality and wrist endpoint termination semantics.
- Updated the unified local task protocol so joystick, waypoint, obstacle, composed, transition, and inpainting proxy rows are explicitly conditioned on the tracking-teacher repair gate.
- Refreshed English reading report, Chinese reading report, and Chinese project/defense report with the wrist endpoint blocker and the updated report boundary.
- Ran conservative storage cleanup; no new files were deleted in this pass, and previously deleted/superseded candidates remain recorded.

## Verification

Initial syntax and generator checks passed:

- `python3 -m py_compile reproduction/scripts/unified_local_task_protocol_table.py reproduction/scripts/update_course_reports.py`
- `envs/bm_analysis/bin/python reproduction/scripts/unified_local_task_protocol_table.py`
- `envs/bm_analysis/bin/python reproduction/scripts/update_course_reports.py`
- `envs/bm_analysis/bin/python reproduction/scripts/cleanup_failed_large_artifacts.py`

Full audit refresh is run after this progress file is added to `artifact_manifest.py` and `reproduction_master_audit.py`.

## Failed / Blocked Items

- No new full PPO or downstream training was launched in this round because the latest live probes and endpoint-group ablation do not yet provide a safe repair candidate that improves both done rate and joint/action transients.
- Current paper-level blockers remain: high-quality official-equivalent tracking teacher, true DAgger rollout data, official VAE/diffusion checkpoints, strict Fig.5/Fig.6 simulation protocol metrics/videos, TensorRT/asynchronous deployment evidence, and real robot validation.

## Effect on English Reading Report

The report now has a stronger scientific narrative: the project is not just listing failures, it has narrowed the next tracking repair target to wrist endpoint semantics. This supports the course report's independent-thinking requirement while preserving the claim boundary that the current local virtual pipeline is not full paper-level BeyondMimic reproduction.

## Next Step

Run the full verification suite, commit, push, then implement a wrist-endpoint tracking probe that compares motion target z/xyz, runtime wrist body positions, FK target positions, `body_pos_relative_w`, `ee_body_pos` trigger counts, and reset-time velocities before launching another full PPO run.

## Git Commit

Pending.
