# Progress Update

## Goal

Refresh the course-facing English reading report, Chinese reading report, and Chinese project/defense report from current machine-readable evidence. Keep the reports useful for class presentation while preserving the claim boundary that the project is a public-resource partial reproduction, not a complete paper-level BeyondMimic reproduction.

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
- `/mnt/infini-data/test/BeyondMimic/res/tracking/fk_repaired_data_quality_gate/fk_repaired_data_quality_gate.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_multiseed_eval/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_multiseed_eval.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_ppo_tracking_quality_diagnostic/robot_order_fk_ppo_tracking_quality_diagnostic.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/robot_order_fk_reset_command_warmup_live_probe/robot_order_fk_reset_command_warmup_live_probe.json`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/unified_local_task_protocol/unified_local_task_protocol.json`
- `/mnt/infini-data/test/BeyondMimic/res/storage_cleanup/cleanup_failed_large_artifacts.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/update_course_reports.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/english_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/chinese_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/chinese_project_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/final_report/english_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/final_report/chinese_reading_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/final_report/chinese_project_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/final_report/final_reproduction_report.json`
- `/mnt/infini-data/test/BeyondMimic/res/final_report/reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.csv`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.md`
- `/mnt/infini-data/test/BeyondMimic/res/docs/completion_matrix_status_audit/completion_matrix_status_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/verification_command_syntax/verification_command_syntax_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/verification_command_script_manifest/verification_command_script_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/verification_command_coverage/verification_command_coverage_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/storage_cleanup/cleanup_failed_large_artifacts.json`
- `/mnt/infini-data/test/BeyondMimic/res/progress_report_audit/progress_report_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.tsv`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260622_004850_course_report_refresh.md`

## Commands Run

- `git status --short && git rev-parse --short HEAD && git log --oneline -5`
- `du -h --max-depth=2 res logs envs cache tmp`
- `python3 -m py_compile reproduction/scripts/update_course_reports.py`
- `python3 reproduction/scripts/update_course_reports.py`
- `python3 reproduction/scripts/cleanup_failed_large_artifacts.py`
- `python3 reproduction/scripts/artifact_manifest.py`
- `python3 reproduction/scripts/paper_vs_reproduction_comparison.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- `python3 reproduction/scripts/completion_matrix_status_audit.py`
- `python3 reproduction/scripts/verification_command_syntax_audit.py`
- `python3 reproduction/scripts/verification_command_script_manifest.py`
- `python3 reproduction/scripts/verification_command_coverage_audit.py`
- `python3 reproduction/scripts/required_artifact_absence_audit.py`
- `python3 reproduction/scripts/progress_report_audit.py`
- `python3 reproduction/scripts/reproduction_master_audit.py`

## Results

- Course-facing reports now use the current machine-readable baseline: master audit `356/356`, artifact manifest `1436`, paper-vs-reproduction rows `223`, comparison type counts `58/19/133/10/3`, and completion matrix counts `74/126/2/1`.
- The English reading report now has a clearer progress estimate: `85-90%` course-report readiness, `75-80%` public-resource engineering coverage, and `40-50%` strict simulation-side paper-level reproduction excluding real robot deployment.
- The reports now describe the current tracking mainline: robot-order FK-repaired bundle, robot-order PPO checkpoint eval, three-seed eval, step-0 reset spike, post-step0 done rate, and reset-command warmup as a partial fix.
- The Chinese project report now has a stronger defense narrative: how the project started from paper reading, how the paper was split into modules, how formulas were turned into code contracts, what data substitutes were used, what failed, and what should be done next.
- Storage hygiene remains conservative: no new large artifact deletion this round, `2` deleted-or-previously-deleted bulky candidates retained in the cleanup audit, and current active run directories are preserved.

## Verification

- `artifact_manifest.py`: `ok`, `1436` artifacts.
- `paper_vs_reproduction_comparison.py`: `ok`, refreshed CSV/JSON/MD.
- `final_reproduction_report.py`: `ok`.
- `completion_matrix_status_audit.py`: `ok`, `203` rows, `0` invalid statuses.
- `verification_command_syntax_audit.py`: `ok`, `199` scripts, `0` failures.
- `verification_command_script_manifest.py`: `ok`, `199` scripts hashed.
- `verification_command_coverage_audit.py`: `ok`, `207` commands, lightweight smoke pass `10/10`.
- `required_artifact_absence_audit.py`: `ok`, `32` rows.
- `progress_report_audit.py`: `ok` before this new progress file is incorporated into the next rerun.
- `reproduction_master_audit.py`: `ok`, `356/356` artifacts passing.

## Failed / Blocked Items

- No verification command failed in this round.
- The project still must not claim full paper-level reproduction. The current blocker is tracking quality: step-0 reset/bootstrap body-position spike, persistent post-step0 done rate, endpoint/`ee_body_pos` termination, and a weak local teacher.
- Official BeyondMimic tracking teacher checkpoint, true DAgger rollout logs, official VAE/diffusion checkpoint, paper Fig.5/Fig.6 closed-loop metrics/videos, TensorRT/Mini-PC deployment evidence, and real robot evidence remain absent.
- One untracked warmup full-eval script remains in the worktree and should be completed or cleaned in the next tracking round; it was not included in this report-only commit unless explicitly staged later.

## Effect on English Reading Report

This round directly improves the course deliverable. The English report now reads less like a path-indexed audit dump and more like a paper reading report with background, method understanding, reproduction setup, evidence, limitations, and personal reflection. It still states clearly: this project does not fully reproduce BeyondMimic at paper level.

## Next Step

Finish or discard the untracked robot-order warmup full-eval script. Then run the warmup full checkpoint eval on GPU 4/7. If it improves step-0 and done-count behavior, integrate it into the tracking gate and use it as the basis for a stronger PPO retraining plan.

## Git Commit

Pending.
