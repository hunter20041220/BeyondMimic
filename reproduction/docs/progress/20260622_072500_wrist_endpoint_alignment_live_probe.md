# Progress Update

## Goal

Move the tracking repair mainline from broad reset/termination diagnostics to a concrete wrist endpoint data-quality target before launching another full PPO run.

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
- Existing robot-order FK reset/action, target-refresh, endpoint-group, and deterministic-reset probe scripts/results.

## Files Modified

- Added `reproduction/scripts/robot_order_fk_wrist_endpoint_alignment_live_probe.py`.
- Updated `reproduction/scripts/artifact_manifest.py`.
- Updated `reproduction/scripts/paper_vs_reproduction_comparison.py`.
- Updated `reproduction/scripts/reproduction_master_audit.py`.
- Updated `reproduction/scripts/update_course_reports.py`.
- Regenerated `reproduction/docs/english_reading_report.md`.
- Regenerated `reproduction/docs/chinese_reading_report.md`.
- Regenerated `reproduction/docs/chinese_project_report.md`.
- Regenerated final report copies under `res/final_report/`.

## Commands Run

- `python3 -m py_compile reproduction/scripts/robot_order_fk_wrist_endpoint_alignment_live_probe.py`
- `python3 reproduction/scripts/robot_order_fk_wrist_endpoint_alignment_live_probe.py`
- `python3 -m py_compile reproduction/scripts/artifact_manifest.py reproduction/scripts/reproduction_master_audit.py reproduction/scripts/paper_vs_reproduction_comparison.py reproduction/scripts/update_course_reports.py reproduction/scripts/robot_order_fk_wrist_endpoint_alignment_live_probe.py`
- `python3 reproduction/scripts/update_course_reports.py`

## Results

The new live IsaacLab probe loads the current robot-order FK-repaired PPO checkpoint and records `body_pos_w`, `body_pos_relative_w`, and `robot_body_pos_w` for ankle and wrist endpoint groups before/after no-advance target refresh and after one zero/policy step.

Key metrics:

- Status: `ok_robot_order_fk_wrist_endpoint_alignment_live_probe`
- Refresh wrist z-error mean: `0.12762290239334106` m
- Refresh ankle z-error mean: `0.06544189155101776` m
- Refresh wrist done rate: `0.15234375`
- Refresh ankle done rate: `0.09765625`
- Policy-step wrist done rate: `0.09375`
- Policy-step ankle done rate: `0.06640625`
- Diagnosis: `wrist_endpoint_target_or_body_semantics_remain_primary_done_source`

This confirms the endpoint-group ablation with direct tensor evidence: target refresh helps, but wrists remain the larger endpoint data-quality/termination source.

## Verification

Full audit refresh is run after this progress note so the new probe enters the manifest, comparison table, final reports, completion audits, and master audit.

## Failed / Blocked Items

The first worker run wrote metrics but stayed in Isaac/Kit shutdown. It was our own probe process, so it was terminated after confirming the metrics file and then the script was fixed to call `os._exit(0)` after writing the payload. The successful rerun exited cleanly.

The result is not a paper metric and not a PPO improvement. It is a tracking data-quality gate that tells the next repair to inspect wrist body target generation, wrist FK height/body order, and `ee_body_pos` body semantics before a new full PPO/downstream chain.

## Effect on English Reading Report

The English reading report, Chinese reading report, and Chinese project report now describe this as the latest tracking bottleneck evidence. It strengthens the course narrative because it shows a concrete progression from paper understanding to live IsaacLab tensor-level debugging.

## Next Step

Repair or ablate wrist endpoint target/body semantics in the motion bundle or termination configuration, then run a full same-seed tracking eval. Only if done/endpoint metrics improve should the project launch a new full PPO run on GPUs 4/7.

## Git Commit

Pending.
