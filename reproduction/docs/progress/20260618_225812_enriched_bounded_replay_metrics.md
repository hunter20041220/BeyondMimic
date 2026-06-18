# Progress Update

## Goal

Move beyond the 4-step resource-adjusted articulation gate by running a longer bounded replay-like metrics diagnostic on
the enriched G1 USD scaffold, while preserving the boundary that this is not official BeyondMimic replay.

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
- `res/tracking/g1_enriched_usd_replay_preflight/tracking_g1_enriched_usd_replay_preflight_audit.json`
- `res/tracking/motion_npz_fixture/tracking_motion_npz_fixture.json`
- `reproduction/generated/whole_body_tracking_local/replay_npz_enriched_usd_preflight.py`
- `reproduction/scripts/tracking_g1_enriched_usd_replay_preflight_audit.py`

## Files Modified

- `reproduction/generated/whole_body_tracking_local/replay_npz_enriched_usd_preflight.py`
- `reproduction/scripts/tracking_g1_enriched_usd_bounded_replay_metrics_audit.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- refreshed small audit/report outputs under `res/artifact_manifest`, `res/comparison`, `res/final_report`,
  `res/docs`, `res/verification_command_*`, `res/progress_report_audit`, `res/master_audit`, and
  `res/tracking/g1_enriched_usd_bounded_replay_metrics`

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_enriched_usd_bounded_replay_metrics_audit.py
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

## Results

- Added optional metrics output to the enriched USD replay preflight entrypoint.
- Added `tracking_g1_enriched_usd_bounded_replay_metrics_audit.py`.
- Ran a 64-step bounded diagnostic on `cuda:6` using the debug walk fixture and generated enriched G1 USD scaffold.
- The run reached `step=64`, wrote metrics JSON, and returned code `0`.
- Metrics:
  - `executed_steps=64`
  - `robot_num_joints=29`
  - `robot_num_bodies=40`
  - `max_joint_pos_abs_error=0.0`
  - `max_joint_vel_abs_error=0.0`
  - `max_root_pos_abs_error=5.820766091346741e-11`
  - `max_root_quat_abs_error=0.0`
  - `root_height_min=0.7963541746139526`
  - `root_height_max=0.7965530157089233`

## Verification

- `artifact_manifest.py`: passed, `256` artifacts, missing `0`.
- `paper_vs_reproduction_comparison.py`: passed.
- `final_reproduction_report.py`: passed.
- `completion_matrix_status_audit.py`: passed, `161` rows, invalid status count `0`.
- `verification_command_syntax_audit.py`: passed, `178` scripts.
- `verification_command_script_manifest.py`: passed, `178` scripts.
- `verification_command_coverage_audit.py`: passed, `186` commands.
- `progress_report_audit.py`: passed.
- `reproduction_master_audit.py`: passed.

## Failed / Blocked Items

- This gate uses a generated resource-adjusted USD scaffold and debug fixture; it is not official `csv_to_npz.py`
  conversion, official replay/evaluation, PPO, DAgger, or paper-level closed-loop evidence.
- Clean Kit shutdown remains unverified because the deterministic gate exits after the success sentinel.
- Official motion conversion/replay, tracking task eval, PPO training/evaluation, teacher rollout data,
  VAE/diffusion closed-loop rollout, Fig. 5/Fig. 6 videos, TensorRT deployment, and real-robot results remain
  incomplete or blocked.

## Effect on English Reading Report

This gives the reading report a stronger reproduction-engineering result: the local virtual stack can now run a
64-step bounded G1 replay-like diagnostic and produce quantitative consistency metrics. It should be framed as evidence
of environment/asset-gate recovery, not as paper-level BeyondMimic reproduction.

## Next Step

Use the deterministic bounded gate to attempt an official-task environment smoke/eval with the resource-adjusted asset
or continue repairing the official `csv_to_npz.py` conversion path.

## Git Commit

Pending at progress-file creation time.
