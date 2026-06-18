# Progress Update

## Goal

Advance the IsaacLab tracking gate beyond static USD readback by testing whether the resource-adjusted enriched G1 USD
scaffold can be loaded as an IsaacLab articulation and stepped through a bounded replay-like debug fixture without
claiming official BeyondMimic replay success.

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
- `reproduction/generated/whole_body_tracking_local/csv_to_npz_local.py`
- `reproduction/generated/whole_body_tracking_local/replay_npz_local.py`
- `reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py`
- `res/tracking/g1_resource_adjusted_enriched_usd/tracking_g1_resource_adjusted_enriched_usd_probe.json`
- `res/tracking/official_replay_conversion/tracking_official_replay_conversion_audit.json`

## Files Modified

- `reproduction/generated/whole_body_tracking_local/replay_npz_enriched_usd_preflight.py`
- `reproduction/scripts/tracking_g1_enriched_usd_replay_preflight_audit.py`
- `reproduction/scripts/tracking_g1_resource_adjusted_enriched_usd_probe.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `res/tracking/g1_resource_adjusted_enriched_usd/g1_resource_adjusted_29dof_enriched_scaffold.usda`
- `res/tracking/g1_resource_adjusted_enriched_usd/tracking_g1_resource_adjusted_enriched_usd_probe.json`
- refreshed small audit/report outputs under `res/artifact_manifest`, `res/comparison`, `res/final_report`,
  `res/docs`, `res/verification_command_*`, `res/progress_report_audit`, and `res/master_audit`

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_enriched_usd_replay_preflight_audit.py
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_enriched_usd_probe.py
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

- Added a generated bounded replay preflight entrypoint that loads the resource-adjusted enriched G1 USD through
  IsaacLab `UsdFileCfg`.
- Fixed the enriched USD scaffold to author USD Physics revolute joint limits in degrees while preserving URDF radian
  limits in custom metadata.
- The bounded replay preflight reaches `sim_created`, `scene_created`, `sim_reset`,
  `robot_contract=num_joints=29,num_bodies=40,device=cuda:6`, and four render steps.
- The preflight records `status=ok_with_resource_adjusted_step_gate_passed_shutdown_timeout`: the step gate passes, but
  Kit shutdown still times out under the bounded command.
- This is resource-adjusted gate evidence only. It is not official `csv_to_npz.py` conversion, official replay, PPO
  training, DAgger data, or paper-level closed-loop BeyondMimic evidence.

## Verification

- `artifact_manifest.py`: passed, `254` artifacts, missing `0`.
- `paper_vs_reproduction_comparison.py`: passed.
- `final_reproduction_report.py`: passed.
- `completion_matrix_status_audit.py`: passed, `161` rows, invalid status count `0`.
- `verification_command_syntax_audit.py`: passed, `178` scripts.
- `verification_command_script_manifest.py`: passed, `178` scripts.
- `verification_command_coverage_audit.py`: passed, `186` commands.
- `progress_report_audit.py`: passed.
- `reproduction_master_audit.py`: passed.

## Failed / Blocked Items

- Official `csv_to_npz.py` still has not produced an official `motion.npz`.
- Official replay and paper-level tracking evaluation remain incomplete.
- The new resource-adjusted gate still times out during Kit shutdown after the success sentinel.
- Sensor/IMU links without inertial tags still trigger PhysX fallback warnings.
- No teacher rollout, DAgger dataset, VAE/diffusion closed-loop rollout, Fig. 5/Fig. 6 video, TensorRT deployment, or
  real-robot result was produced.

## Effect on English Reading Report

This provides a precise reproduction narrative for the report: the project progressed from static official-code and USD
contract audits to a bounded virtual articulation gate that actually instantiates the 29-DoF/40-body G1 scaffold and
steps it in IsaacLab. It should be described as engineering evidence for environment recovery and asset compatibility,
not as a reproduced BeyondMimic tracking result.

## Next Step

Investigate the remaining Kit shutdown timeout and then attempt a cleaner resource-adjusted replay path or official
motion replay conversion path. The next paper-facing milestone remains official replay/evaluation, not PPO training yet.

## Git Commit

Pending at progress-file creation time.
