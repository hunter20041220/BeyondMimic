# Progress Update

## Goal

Move the tracking reproduction back toward the paper mainline by resolving whether reset-command warmup is a real teacher-quality improvement or only a reset/phase diagnostic. The work uses a full seed-matched evaluation, not another smoke test.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/prompt06211658.txt`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/current_project_reproduction_state_20260622.md`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup.json`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_seed_matched.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/robot_order_fk_warmup_seed_matched_phase_diagnostic.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/paper_vs_reproduction_comparison.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/update_course_reports.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/progress_report_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/completion_matrix.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/current_project_reproduction_state_20260622.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/progress/20260622_013910_seed_matched_warmup_phase.md`

## Commands Run

- `git status --short`
- `python3 -m py_compile reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_seed_matched.py reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup.py`
- `nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits -i 4,7`
- `python3 reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_seed_matched.py`
- `python3 -m py_compile reproduction/scripts/robot_order_fk_warmup_seed_matched_phase_diagnostic.py`
- `python3 reproduction/scripts/robot_order_fk_warmup_seed_matched_phase_diagnostic.py`
- `python3 -m py_compile reproduction/scripts/artifact_manifest.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/final_reproduction_report.py reproduction/scripts/reproduction_master_audit.py reproduction/scripts/update_course_reports.py reproduction/scripts/progress_report_audit.py reproduction/scripts/robot_order_fk_warmup_seed_matched_phase_diagnostic.py reproduction/scripts/tracking_g1_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_seed_matched.py`
- `python3 reproduction/scripts/artifact_manifest.py`
- `python3 reproduction/scripts/paper_vs_reproduction_comparison.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- `envs/bm_analysis/bin/python reproduction/scripts/update_course_reports.py`
- `python3 reproduction/scripts/completion_matrix_status_audit.py`
- `python3 reproduction/scripts/verification_command_syntax_audit.py`
- `python3 reproduction/scripts/verification_command_script_manifest.py`
- `python3 reproduction/scripts/verification_command_coverage_audit.py`
- `python3 reproduction/scripts/required_artifact_absence_audit.py`
- `python3 reproduction/scripts/progress_report_audit.py`
- `python3 reproduction/scripts/reproduction_master_audit.py`
- `python3 reproduction/scripts/artifact_manifest.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- `envs/bm_analysis/bin/python reproduction/scripts/update_course_reports.py`
- `python3 reproduction/scripts/reproduction_master_audit.py`
- `git diff --stat`

## Results

The seed-matched full warmup evaluation completed:

- status: `ok_official_importer_export_fk_repaired_robot_order_full_bundle_ppo_checkpoint_eval_warmup_seed_matched_completed`
- seed: `20260721`, matching the non-warmup baseline
- scope: `2048` envs x `299` steps = `612352` env steps
- step-0 done count: `2048 -> 596`
- step-0 body-position error: `43.294166564941406 -> 0.26491862535476685` m
- total done rate: `0.1782798129180602 -> 0.22153761235367894`

The phase diagnostic completed:

- status: `ok_robot_order_fk_warmup_seed_matched_phase_diagnostic`
- same-seed total done-rate delta: `0.04325779943561875`
- same-seed post-step0 done-rate delta: `0.04578210203439598`
- same-seed `ee_body_pos` termination fraction delta: `0.04554896530100333`
- same-seed sampling top-bin post-step0 delta: `0.0`

Interpretation: reset-command warmup removes the stale step-0 target artifact but is not a teacher-quality fix. It increases post-step0 `ee_body_pos` termination under the same seed, so the next mainline target is command/observation phase consistency rather than another blind PPO run.

## Verification

Final full verification passed after report/audit refresh:

- `artifact_manifest.py`: `ok`, `1465` artifacts, `0` missing
- `paper_vs_reproduction_comparison.py`: `ok`, `225` rows
- `final_reproduction_report.py`: `ok`
- `update_course_reports.py`: `ok`, final report copies refreshed with `1465` artifacts and `225` comparison rows
- `completion_matrix_status_audit.py`: `ok`, `205` rows, status counts `complete=74`, `partial=128`, `blocked=2`, `out_of_scope=1`
- `verification_command_syntax_audit.py`: `ok`, `199` scripts, `0` failed syntax checks
- `verification_command_script_manifest.py`: `ok`, `199` scripts
- `verification_command_coverage_audit.py`: `ok`, `207` commands, `10` smoke-pass commands
- `required_artifact_absence_audit.py`: `ok`, `32` rows
- `progress_report_audit.py`: `ok`, `199` rows, `161` per-round progress Markdown files audited
- `reproduction_master_audit.py`: `ok`, `364/364` artifacts passed

## Failed / Blocked Items

No new failed run in this round. The current blocker remains tracking teacher quality: the local checkpoint is runnable but not a paper-level teacher because endpoint termination remains too high.

## Effect on English Reading Report

This gives the report a stronger independent-analysis point: the project does not merely say the tracking teacher is weak; it identifies a concrete reset/phase mechanism, verifies it with a full same-seed eval, and explains why a stronger teacher requires fixing the phase/termination gate before DAgger/VAE/diffusion reruns.

## Next Step

Run a targeted reset-target refresh variant that recomputes motion targets after reset without advancing `MotionCommand.time_steps`, then only start another full PPO run if the full tracking eval termination gate improves.

## Git Commit

Planned commit message: `fix: diagnose robot-order warmup phase gate`
