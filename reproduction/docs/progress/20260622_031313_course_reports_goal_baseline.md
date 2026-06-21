# Progress Update

## Goal

Update the course-facing BeyondMimic reports and create a current goal baseline that reflects the real state after the IsaacLab headless gate was cleared. This round prioritizes English/Chinese reading-report material, Chinese project-defense narrative, storage-pressure visibility, and the next mainline reproduction target.

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
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `res/docs/completion_matrix_status_audit/completion_matrix_status_audit.json`

## Files Modified

- `reproduction/scripts/update_course_reports.py`
- `reproduction/docs/english_reading_report.md`
- `reproduction/docs/chinese_reading_report.md`
- `reproduction/docs/chinese_project_report.md`
- `reproduction/docs/current_goal_baseline_20260622.md`
- `res/final_report/english_reading_report.md`
- `res/final_report/chinese_reading_report.md`
- `res/final_report/chinese_project_report.md`
- refreshed audit/report outputs under `res/artifact_manifest`, `res/final_report`, `res/master_audit`, and `res/verification_command_coverage`

## Commands Run

```bash
git status --short
git log -1 --oneline
df -h /mnt/infini-data/test/BeyondMimic
du -sh reproduction res logs envs cache tmp
find res/runs -mindepth 1 -maxdepth 1 -type d -exec du -sh {} \; | sort -hr | head -n 40
envs/bm_analysis/bin/python reproduction/scripts/update_course_reports.py
python3 -m py_compile reproduction/scripts/update_course_reports.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/progress_report_audit.py
python3 reproduction/scripts/required_artifact_absence_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

## Results

- Updated English reading report with the current audit state, revised progress estimates, tracking-quality diagnosis, and storage table.
- Updated Chinese reading report with a clearer claim boundary and revised completion estimates.
- Updated Chinese project report with a defense-oriented project narrative: where the work started, how the paper was modularized, what data substitutes were used, what was implemented, what failed, and what remains.
- Added `current_goal_baseline_20260622.md` to replace the stale headless-gate-first objective with the current mainline: fix tracking data quality, then rerun stronger PPO, then redo downstream VAE/diffusion/guidance.
- Added storage-pressure visibility to `update_course_reports.py`; it now scans top `res/runs` directories and writes a compact storage table into course-facing reports.
- No active large artifact was deleted in this round. The active scaled-PPO teacher rollout shards and scaled-PPO state-latent dataset are still retained because they are the strongest currently available local downstream chain.

## Verification

All required verification commands passed:

- `artifact_manifest.py`: `ok`, `1485` artifacts, `0` missing
- `paper_vs_reproduction_comparison.py`: `ok`, `227` rows
- `final_reproduction_report.py`: `ok`
- `completion_matrix_status_audit.py`: `ok`, `207` rows
- `verification_command_syntax_audit.py`: `ok`, `199` scripts, `0` failed
- `verification_command_script_manifest.py`: `ok`, `199` scripts
- `verification_command_coverage_audit.py`: `ok`, `207` commands
- `progress_report_audit.py`: `ok` before this new progress file was added
- `required_artifact_absence_audit.py`: `ok`, `32` rows
- `reproduction_master_audit.py`: `ok`, `370/370` artifacts passed

## Failed / Blocked Items

- No verification command failed in this round.
- No new training was started.
- No large active downstream artifact was deleted because the dependency boundary is still useful for the next reproduction step.
- The core technical blocker remains tracking quality: reset target refresh fixes stale step-0 body target but exposes or creates initial joint-velocity/action transient and worsens post-step0 done rate.

## Effect on English Reading Report

The English report now better supports the course requirement: it emphasizes understanding, auditability, public-resource reproduction, local virtual pipeline evidence, and honest paper-level limitations. It explicitly avoids claiming a complete BeyondMimic reproduction.

## Next Step

Proceed to the tracking mainline rather than more report-only work:

1. implement a reset state/action consistency live probe,
2. test whether rewriting/resetting initial joint velocity and last-action observation reduces post-step0 done rate,
3. only then run the next full PPO training/evaluation on GPU 4/7,
4. if teacher quality improves, regenerate teacher rollout -> VAE -> state-latent -> denoiser -> guidance.

## Git Commit

Pending at the time this progress file is written.
