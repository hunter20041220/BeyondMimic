# Progress Update

## Goal

Free storage conservatively, integrate the no-advance reset-target refresh diagnostic into the report/audit chain, and keep the current tracking mainline focused on data quality before another PPO/downstream rerun.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/current_project_reproduction_state_20260622.md`
- `reproduction/docs/completion_matrix.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `res/tracking/robot_order_fk_reset_target_refresh_no_advance_live_probe/robot_order_fk_reset_target_refresh_no_advance_live_probe.json`
- `res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance.json`
- `res/storage_cleanup/cleanup_failed_large_artifacts.json`

## Files Modified

- `reproduction/scripts/cleanup_failed_large_artifacts.py`
- `reproduction/scripts/robot_order_fk_reset_target_refresh_no_advance_live_probe.py`
- `reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/paper_vs_reproduction_comparison.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/update_course_reports.py`
- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/chinese_reading_report.md`
- `reproduction/docs/chinese_project_report.md`
- `reproduction/docs/completion_matrix.md`
- `reproduction/docs/current_project_reproduction_state_20260622.md`
- `res/final_report/english_reading_report.md`
- `res/final_report/chinese_reading_report.md`
- `res/final_report/chinese_project_report.md`
- `res/storage_cleanup/cleanup_failed_large_artifacts.json`

## Commands Run

- `python3 reproduction/scripts/cleanup_failed_large_artifacts.py`
- `python3 -m py_compile reproduction/scripts/update_course_reports.py reproduction/scripts/final_reproduction_report.py reproduction/scripts/artifact_manifest.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/reproduction_master_audit.py reproduction/scripts/cleanup_failed_large_artifacts.py reproduction/scripts/robot_order_fk_reset_target_refresh_no_advance_live_probe.py reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance.py`
- `envs/bm_analysis/bin/python reproduction/scripts/update_course_reports.py`
- Verification commands listed below are part of this round's required rerun set.

## Results

- Storage cleanup deleted 8 old bulky/superseded raw-array candidates in this round, recorded 2 previously deleted candidates, and accounts for `4853459410` managed bytes removed or already absent while retaining the current scaled teacher rollout, current scaled state-latent run, current robot-order PPO training run, and later importer-export state-latent candidate.
- The no-advance reset-target live probe completed with status `ok_robot_order_fk_reset_target_refresh_no_advance_live_probe`. Endpoint-z done rate improved from `1.0` to `0.2734375`, endpoint-z error mean improved from `0.5298784375190735` m to `0.104344442486763` m, and `time_steps_unchanged_by_refresh=true`.
- The same-seed 2048-env x 299-step no-advance reset-target full eval completed with status `ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_target_refresh_no_advance_completed`. It improved step-0 done count by `-1453`, but total done rate still worsened from `0.1782798129180602` to `0.22340745192307693`, and post-step0 done-rate delta was `+0.047659854760906034`.
- A wrapper injection failure before the successful eval is retained under `res/failed_runs/tracking_g1_robot_order_target_refresh_no_advance_wrapper_injection_error/status.json` with a resolved follow-up pointer.

## Verification

This progress file is created before the final verification rerun so it can be included in the manifest and master audit. The intended verification set is:

- `python3 reproduction/scripts/artifact_manifest.py`
- `python3 reproduction/scripts/paper_vs_reproduction_comparison.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- `python3 reproduction/scripts/completion_matrix_status_audit.py`
- `python3 reproduction/scripts/verification_command_syntax_audit.py`
- `python3 reproduction/scripts/verification_command_script_manifest.py`
- `python3 reproduction/scripts/verification_command_coverage_audit.py`
- `python3 reproduction/scripts/reproduction_master_audit.py`

## Failed / Blocked Items

- The no-advance refresh fixes the stale reset-target spike but does not fix teacher quality. The local robot-order PPO checkpoint remains weak and must not be used as a paper-level teacher.
- The next tracking blocker is reset state/action distribution, initial joint velocity mismatch, endpoint thresholds, and `ee_body_pos` termination rather than IsaacLab import or headless startup.
- Official BeyondMimic teacher checkpoints, true DAgger logs, official VAE/diffusion checkpoints, paper Fig. 5/Fig. 6 rollout metrics/videos, TensorRT deployment, and real-robot results remain missing.

## Effect on English Reading Report

The English reading report, Chinese reading report, and Chinese project report now explain the no-advance reset-target diagnostic as mainline negative evidence: stale reset targets are real, but refreshing them is not enough to make the local PPO checkpoint a credible teacher. This improves the defense narrative by showing why the next work should repair tracking quality before rerunning downstream VAE/diffusion/guidance.

## Next Step

Run the required verification chain, fix any audit failures, then commit and push this report/audit refresh. The next experimental step after this round should inspect reset state/action distribution and `ee_body_pos` termination before launching another full PPO run.

## Git Commit

Planned commit message: `fix: integrate reset target refresh diagnostic`
