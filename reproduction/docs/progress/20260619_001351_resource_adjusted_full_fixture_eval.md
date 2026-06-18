# Progress Update

## Goal

Extend the successful resource-adjusted `Tracking-Flat-G1-v0` smoke into a full available fixture diagnostic, while preserving the boundary that this is not official BeyondMimic replay, PPO, DAgger, or paper-level rollout evidence.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/completion_matrix.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- Prior tracking task and enriched USD audit outputs under `res/tracking/`.

## Files Modified

- `reproduction/scripts/tracking_g1_resource_adjusted_multi_fixture_eval_audit.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/completion_matrix.md`
- `reproduction/PROGRESS.md`
- `reproduction/docs/progress/20260619_001351_resource_adjusted_full_fixture_eval.md`

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_multi_fixture_eval_audit.py
```

The first shared-Kit attempt reached the full walk fixture but timed out while setting up the run fixture. The script was revised to execute each fixture in an isolated Kit process with log-progress stall detection, then rerun successfully.

## Results

The resource-adjusted full fixture task eval passed with status `ok_resource_adjusted_multi_fixture_task_eval`.

Key metrics:

- `fixture_count=3`
- `total_steps=897`
- `walk/run/jump` each reached `299/299`
- action dimension `29`
- policy observation dimension `160`
- critic observation dimension `286`
- reward terms `9`
- termination terms `4`
- robot joints `29`
- robot bodies `40`

Failed shared-Kit evidence was retained under `res/failed_runs/tracking_g1_resource_adjusted_multi_fixture_eval_20260618T160520Z_shared_kit_timeout`.

## Verification

Full verification bundle is run after this progress file is written:

```bash
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/progress_report_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

## Failed / Blocked Items

This is still not official `csv_to_npz.py` conversion, official `replay_npz.py` evaluation, PPO training, DAgger rollout data, a teacher rollout dataset, TensorRT deployment, Fig. 5/Fig. 6 video evidence, or robot evidence.

The isolated-fixture runner exists because a single shared Kit process completed walk `299/299` but timed out during the next fixture setup. That failure is retained as an engineering blocker record, not hidden.

## Effect on English Reading Report

This adds a concrete, auditable paragraph for the code reproduction section: the project now demonstrates that the official `Tracking-Flat-G1-v0` ManagerBasedRLEnv surface can run full local debug fixtures under IsaacLab with the expected G1 task contracts, while clearly separating that evidence from paper-level tracking performance.

## Next Step

Use this validated task-contract gate as a baseline while retrying the official motion conversion/replay path. If the official asset/conversion blocker is resolved, proceed to bounded tracking train/eval before any larger PPO run.

## Git Commit

Pending.
