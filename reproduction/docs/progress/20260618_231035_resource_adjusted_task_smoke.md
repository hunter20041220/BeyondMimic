# Progress Update

## Goal

Advance from resource-adjusted replay-like stepping to an official tracking task manager smoke/eval gate that verifies
reset, stepping, observation, action, reward, and termination surfaces without starting PPO training or claiming official
paper-level replay.

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
- `res/tracking/g1_enriched_usd_bounded_replay_metrics/tracking_g1_enriched_usd_bounded_replay_metrics_audit.json`
- `res/tracking/observation_action_schema_audit/tracking_observation_action_schema_audit.json`
- `res/tracking/randomization_termination_audit/tracking_randomization_termination_audit.json`
- `res/tracking/reward_formula_audit/tracking_reward_formula_audit.json`
- official `Tracking-Flat-G1-v0` config and `MotionCommand` source

## Files Modified

- `reproduction/scripts/tracking_g1_resource_adjusted_task_smoke_audit.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- refreshed small audit/report outputs under `res/artifact_manifest`, `res/comparison`, `res/final_report`,
  `res/docs`, `res/verification_command_*`, `res/progress_report_audit`, `res/master_audit`, and
  `res/tracking/g1_resource_adjusted_task_smoke`

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_task_smoke_audit.py
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

- Added a resource-adjusted official task smoke probe for `Tracking-Flat-G1-v0`.
- The probe substitutes the generated enriched G1 USD and debug motion fixture into the official ManagerBasedRLEnv
  config, then runs `num_envs=1`, reset, and eight zero-action steps on `cuda:6`.
- The probe returned code `0` with `status=ok_resource_adjusted_tracking_task_smoke`.
- Verified surfaces:
  - `action_dim=29`
  - `policy_observation_dim=160`
  - `critic_observation_dim=286`
  - `reward_terms=9`
  - `termination_terms=4`
  - `robot_num_joints=29`
  - `robot_num_bodies=40`
  - reset reached and eight environment steps completed
- Recorded reward range and command tracking metrics for audit, including anchor/body/joint error metrics.

## Verification

- `artifact_manifest.py`: passed, `258` artifacts, missing `0`.
- `paper_vs_reproduction_comparison.py`: passed.
- `final_reproduction_report.py`: passed.
- `completion_matrix_status_audit.py`: passed, `161` rows, invalid status count `0`.
- `verification_command_syntax_audit.py`: passed, `178` scripts.
- `verification_command_script_manifest.py`: passed, `178` scripts.
- `verification_command_coverage_audit.py`: passed, `186` commands.
- `progress_report_audit.py`: passed.
- `reproduction_master_audit.py`: passed.

## Failed / Blocked Items

- This is a resource-adjusted task smoke gate. It substitutes a generated enriched USD and debug fixture; it is not
  official `csv_to_npz.py` output, official replay/evaluation, PPO training, DAgger data, or paper-level evidence.
- `terminated_total=8` under zero-action smoke confirms the task termination surface is active, but it is not a success
  metric or trained policy result.
- Official motion conversion/replay, longer task evaluation, PPO training/evaluation, teacher rollout data,
  VAE/diffusion closed-loop rollout, Fig. 5/Fig. 6 videos, TensorRT deployment, and real-robot results remain
  incomplete or blocked.

## Effect on English Reading Report

This is useful for the reading report because it demonstrates that the official task manager stack can now be exercised
locally through reset and step surfaces, with exact action/observation/reward/termination dimensions matching the static
audits. It should be framed as resource-adjusted simulation smoke/eval evidence, not as BeyondMimic paper-level
tracking reproduction.

## Next Step

Use the passing task smoke as a base for either a longer resource-adjusted tracking evaluation or another attempt to
repair the official `csv_to_npz.py` conversion/replay path.

## Git Commit

Pending at progress-file creation time.
