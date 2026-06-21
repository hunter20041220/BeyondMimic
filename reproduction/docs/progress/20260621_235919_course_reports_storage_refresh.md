# Progress Update

## Goal

Refresh the course-facing BeyondMimic reading/project reports with the latest audit baseline, keep the report text focused on paper understanding and defense narrative rather than path-heavy artifact listings, and run conservative storage hygiene without deleting active evidence-chain artifacts.

## Files Read

- `prompt06211658.txt`
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
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `res/report_assets/unified_local_task_protocol/unified_local_task_protocol.md`
- `reproduction/scripts/cleanup_failed_large_artifacts.py`

## Files Modified

- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/chinese_reading_report.md`
- `reproduction/docs/chinese_project_report.md`
- `res/final_report/english_reading_report.md`
- `res/final_report/chinese_reading_report.md`
- `res/final_report/chinese_project_report.md`
- `res/storage_cleanup/cleanup_failed_large_artifacts.json`
- Refreshed generated audit/report outputs under `res/artifact_manifest/`, `res/final_report/`, `res/master_audit/`, and `res/verification_command_coverage/`.

## Commands Run

```bash
git status --short
git rev-parse --short HEAD
rg --files reproduction/docs res/final_report res/report_assets
find res/runs res/failed_runs res/videos res/checkpoints -maxdepth 3 -type f
du -sh res/runs res/failed_runs res/visualization res/level_c res/tracking logs
python3 reproduction/scripts/cleanup_failed_large_artifacts.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
python3 reproduction/scripts/required_artifact_absence_audit.py
```

## Results

- Updated the English reading report with current audit numbers: master audit `352/352`, artifact manifest `1427`, paper-vs-reproduction `222`, and completion matrix `74 complete / 125 partial / 2 blocked / 1 out_of_scope`.
- Added a clearer "From Paper Equations To Code" section explaining how tracking rewards/terminations, VAE, state-latent diffusion, and guidance costs map into implementation contracts.
- Updated the Chinese reading report with the same current baseline, a formula-to-code explanation, and a compact task-protocol table.
- Updated the Chinese project report for defense use with the current baseline, current progress estimates, and a clearer Fig.5/Fig.6 proxy-protocol claim boundary.
- Synchronized all three reports into `res/final_report/`.
- Ran conservative storage cleanup. No new files were deleted in this round; the audit records two previously deleted superseded same-seed large directories and confirms that current active teacher-rollout/state-latent directories are retained.

## Verification

All required verification commands passed:

- `artifact_manifest.py`: `ok`, `1427` artifacts before this progress file.
- `paper_vs_reproduction_comparison.py`: `ok`, `222` rows.
- `final_reproduction_report.py`: `ok`.
- `completion_matrix_status_audit.py`: `ok`, `202` rows.
- `verification_command_syntax_audit.py`: `ok`, `199` scripts.
- `verification_command_script_manifest.py`: `ok`, `199` scripts.
- `verification_command_coverage_audit.py`: `ok`, `207` commands.
- `reproduction_master_audit.py`: `ok`.
- `required_artifact_absence_audit.py`: `ok`, `32` rows.

## Failed / Blocked Items

- No verification command failed in this report/storage round.
- No new training or live tracking experiment was run.
- Strict paper-level items remain incomplete: official BeyondMimic tracking teacher checkpoint, true DAgger rollout logs, official VAE/diffusion checkpoints, Fig.5/Fig.6 paper-level videos/metrics, TensorRT deployment, MuJoCo/ROS sim-to-sim logs, and real Unitree G1 evidence.
- Current tracking mainline remains blocked by reset/target alignment and `ee_body_pos` termination quality, especially the deterministic step-0 done spike in robot-order FK PPO eval.

## Effect on English Reading Report

The English report now reads more like a course reading report rather than a raw audit dump. It explains the paper's motivation, method decomposition, formula-to-code mapping, public-resource reproduction boundary, local virtual results, major difficulties, and personal reflection. It still explicitly states that the project does not fully reproduce BeyondMimic at paper level.

## Next Step

Return to the tracking mainline: implement and run the reset command-warmup live probe for the robot-order FK bundle, verify whether the step-0 termination spike is caused by stale/zero command targets, then patch local eval/train wrappers only if the live probe supports that fix. After the tracking-quality gate improves, run a stronger GPU 4/7 PPO full run and regenerate downstream teacher/VAE/diffusion/guidance evidence from the improved teacher.

## Git Commit

Pending at the time this progress note is written. Planned commit message: `docs: refresh course reports and storage audit`.
