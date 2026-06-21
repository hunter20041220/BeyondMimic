# Progress Update

## Goal

Refresh the course-facing BeyondMimic reading/project reports, reduce path-heavy audit prose, synchronize final-report copies, and inspect storage pressure without deleting current evidence-chain artifacts.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/chinese_reading_report.md`
- `reproduction/docs/chinese_project_report.md`
- `res/master_audit/reproduction_master_audit.json`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `res/tracking/fk_repaired_data_quality_gate/fk_repaired_data_quality_gate.json`
- `res/report_assets/unified_local_task_protocol/unified_local_task_protocol.json`
- `prompt06211658.txt`

## Files Modified

- `reproduction/scripts/update_course_reports.py`
- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/chinese_reading_report.md`
- `reproduction/docs/chinese_project_report.md`
- `res/final_report/english_reading_report.md`
- `res/final_report/chinese_reading_report.md`
- `res/final_report/chinese_project_report.md`
- regenerated audit/report files under `res/artifact_manifest/`, `res/final_report/`, `res/master_audit/`, `res/required_artifact_absence/`, and `res/verification_command_coverage/`.

## Commands Run

```bash
python3 reproduction/scripts/update_course_reports.py
python3 reproduction/scripts/cleanup_failed_large_artifacts.py
du -sh --apparent-size * .git
du -h --max-depth=2 res logs tmp cache envs reproduction
find /mnt/infini-data/test/BeyondMimic -xdev -type f -size +100M -printf '%s %p\n'
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/required_artifact_absence_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

## Results

- Replaced the path-heavy reading-report draft with a course-facing English report that keeps the key audit numbers current: `1403` manifest artifacts, `216` comparison rows, `343/343` master artifacts passing, and `goal_complete=false`.
- Added synchronized Chinese reading and Chinese project/defense reports with the same claim boundary.
- Added `update_course_reports.py` so future report refreshes read current JSON audits instead of hard-coding stale numbers.
- Storage inspection found that the largest safe-looking items are mostly required preserved archives, `other/` backup copies, project-local environments, public dataset copies, and currently referenced teacher/state-latent run directories. The conservative cleanup script found no additional safe large deletion in this round.

## Verification

- `artifact_manifest.py`: passed, `1403` artifacts, missing `0`.
- `paper_vs_reproduction_comparison.py`: passed.
- `final_reproduction_report.py`: passed.
- `completion_matrix_status_audit.py`: passed, `199` rows, invalid status count `0`.
- `verification_command_syntax_audit.py`: passed, `199` scripts, failed `0`.
- `verification_command_script_manifest.py`: passed, `199` scripts.
- `verification_command_coverage_audit.py`: passed, `207` commands, `10/10` lightweight smoke passes.
- `required_artifact_absence_audit.py`: passed, `32` rows.
- `reproduction_master_audit.py`: passed after refreshing the English report boundary text, final report, artifact manifest, and absence audit.

## Failed / Blocked Items

- The first report-generation attempt failed because the unified protocol JSON filename was `unified_local_task_protocol.json`, not `unified_local_task_protocol_table.json`; the script now supports both names.
- The first master-audit rerun failed because the shortened English report was below the expected word count and missed exact boundary strings used by the final-report audit; the report now includes the required paper-level boundary and official-loop virtual-chain wording.
- Tracking remains the main non-robot blocker: the FK-repaired PPO eval still has reward mean about `0.01129` and done count `612350 / 612352`, so it is not a trustworthy paper-level teacher.

## Effect on English Reading Report

The English report now reads as a course submission rather than a raw audit log. It explains the method, summarizes the public-resource reproduction, gives current evidence counts, states that this project does not fully reproduce BeyondMimic at paper-level, and identifies the next scientific step: repair tracking quality before rerunning downstream VAE/diffusion/guidance.

## Next Step

Return to the tracking mainline: inspect FK-repaired `body_pos_w`, endpoint z errors, reset alignment, and termination terms, then run a full task/eval protocol only after the data-quality issue has a concrete fix.

## Git Commit

Pending at the time this progress note was written.
