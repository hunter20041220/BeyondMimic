# Progress Update

## Goal

Feed the official-CSV-derived resource-adjusted `motion.npz` into the official `Tracking-Flat-G1-v0` task manager and run the full available 299-step diagnostic, rather than stopping at replay-only evidence.

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
- `reproduction/scripts/tracking_g1_resource_adjusted_multi_fixture_eval_audit.py`
- CSV conversion/replay audit outputs under `res/tracking/g1_resource_adjusted_csv_*`.

## Files Modified

- `reproduction/scripts/tracking_g1_resource_adjusted_csv_task_eval_audit.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/completion_matrix.md`
- `reproduction/PROGRESS.md`
- `reproduction/docs/progress/20260619_004056_resource_adjusted_csv_task_eval.md`

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_csv_task_eval_audit.py
```

## Results

The official-CSV-derived resource-adjusted motion was evaluated in the official `Tracking-Flat-G1-v0` ManagerBasedRLEnv stack for all 299 available motion steps.

Key metrics:

- status: `ok_resource_adjusted_csv_task_eval`
- step count: `299`
- action dimension: `29`
- policy observation dimension: `160`
- critic observation dimension: `286`
- reward terms: `9`
- termination terms: `4`
- robot joints/bodies: `29` / `40`
- reward mean/min/max: `0.02670689582525687` / `-0.010335305705666542` / `0.052452173084020615`
- terminated/truncated totals: `26` / `12`

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

This remains resource-adjusted diagnostic evidence. The motion source is an official downloaded CSV, but the robot asset uses the generated enriched USD and the actions are zero diagnostic actions. Termination/truncation counts are not policy-quality or paper success/failure metrics.

Still missing: official `csv_to_npz.py`, official `replay_npz.py`, PPO training/evaluation, DAgger rollout data, teacher rollout dataset, trained tracking checkpoint, TensorRT deployment, Fig. 5/Fig. 6 videos, and real robot execution.

## Effect on English Reading Report

This supports a stronger reproduction narrative: official-source motion data can now pass through conversion, replay, and official task-manager surfaces locally, while the report can clearly explain why this remains below paper-level closed-loop reproduction.

## Next Step

Use this as the strongest current tracking-side gate before either retrying a bounded train entry or continuing to isolate the official URDF/USD converter blocker.

## Git Commit

Pending.
