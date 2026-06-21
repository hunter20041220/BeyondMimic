# Progress Update

## Goal

Move the scaled PPO tracking diagnosis back toward the main reproduction path by measuring the actual wrist/ankle z-errors that trigger the official `ee_body_pos` termination gate during a full-size checkpoint evaluation.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/PROGRESS.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/final_reproduction_report.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/english_reading_report.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `reproduction/scripts/tracking_g1_resource_adjusted_ppo_checkpoint_eval.py`
- `reproduction/scripts/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.py`
- `res/report_assets/official_importer_export_scaled_ppo_ee_body_pos_termination_source_audit/ee_body_pos_termination_source_audit.json`
- `res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval/tracking_g1_official_importer_export_full_bundle_scaled_ppo_checkpoint_eval.json`

## Files Modified

- Added `reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_endpoint_z_error_trace.py`
- Updated `reproduction/scripts/artifact_manifest.py`
- Updated `reproduction/scripts/paper_vs_reproduction_comparison.py`
- Updated `reproduction/scripts/final_reproduction_report.py`
- Updated `reproduction/scripts/reproduction_master_audit.py`
- Updated `reproduction/docs/completion_matrix.md`
- Updated `reproduction/docs/english_reading_report.md`
- Updated `res/final_report/english_reading_report.md`
- Added retained failed-run status `res/failed_runs/tracking_endpoint_z_error_trace_eula_env_20260621_020610/status.json`
- Regenerated comparison, manifest, final-report, verification, absence, and master-audit outputs.

## Commands Run

- `python3 -m py_compile reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_endpoint_z_error_trace.py`
- `envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_importer_export_scaled_ppo_endpoint_z_error_trace.py`
- `python3 reproduction/scripts/artifact_manifest.py`
- `python3 reproduction/scripts/paper_vs_reproduction_comparison.py`
- `python3 reproduction/scripts/final_reproduction_report.py`
- `python3 reproduction/scripts/completion_matrix_status_audit.py`
- `python3 reproduction/scripts/verification_command_syntax_audit.py`
- `python3 reproduction/scripts/verification_command_script_manifest.py`
- `python3 reproduction/scripts/verification_command_coverage_audit.py`
- `python3 reproduction/scripts/required_artifact_absence_audit.py`
- `python3 reproduction/scripts/progress_report_audit.py`
- `python3 reproduction/scripts/reproduction_master_audit.py`

## Results

New full-size endpoint trace:

- JSON: `res/tracking/g1_official_importer_export_full_bundle_scaled_ppo_endpoint_z_error_trace/tracking_g1_official_importer_export_scaled_ppo_endpoint_z_error_trace.json`
- Run dir: `res/runs/tracking_g1_official_importer_export_full_bundle_scaled_ppo_endpoint_z_error_trace/endpoint_z_error_trace_20260621_020704_seed20260801/`
- Report assets: `res/report_assets/official_importer_export_scaled_ppo_endpoint_z_error_trace/`

The run used the official-importer-export G1 USDA, full 40-motion public motion bundle, and local iteration-999 scaled PPO checkpoint for `2048` envs x `299` steps (`612352` env steps).

Key numbers:

- Aggregate z-threshold exceed-rate mean: `0.9986282399665551`
- Left ankle mean absolute z-error: `0.7105472759658278` m
- Right ankle mean absolute z-error: `0.72380428669046` m
- Left ankle mean exceed rate: `0.9983293922449832`
- Right ankle mean exceed rate: `0.9949914428302676`
- Left wrist mean exceed rate: `0.4216545385660535`
- Right wrist mean exceed rate: `0.5952279081312709`

This shows the current local scaled PPO teacher is mainly failing the official endpoint gate through ankle height, not merely through a vague aggregate low reward.

## Verification

All required verification commands passed after integration:

- `artifact_manifest.py`: `ok`, `1328` artifacts, `0` missing
- `paper_vs_reproduction_comparison.py`: `ok`, `208` rows
- `final_reproduction_report.py`: `ok`
- `completion_matrix_status_audit.py`: `ok`, `198` rows, `0` invalid statuses
- `verification_command_syntax_audit.py`: `ok`, `198` scripts, `0` failures
- `verification_command_script_manifest.py`: `ok`, `198` scripts
- `verification_command_coverage_audit.py`: `ok`, `206` commands
- `required_artifact_absence_audit.py`: `ok`, `32` rows
- `progress_report_audit.py`: `ok`
- `reproduction_master_audit.py`: `ok`, `339/339` artifacts passed

## Failed / Blocked Items

- The first endpoint-trace attempt failed before IsaacLab startup because the new wrapper did not pass the Isaac Sim EULA/cache/Vulkan environment variables. The retained status JSON records this under `res/failed_runs/tracking_endpoint_z_error_trace_eula_env_20260621_020610/status.json`.
- The wrapper was fixed by adding `OMNI_KIT_ACCEPT_EULA=YES`, `ACCEPT_EULA=Y`, Omniverse cache dirs, `VK_ICD_FILENAMES`, `ISAAC_PATH`, and GPU foundation `LD_LIBRARY_PATH`.
- The rerun completed successfully.
- This remains a local checkpoint diagnostic, not a paper-level tracking teacher, official BeyondMimic checkpoint, DAgger rollout, Fig. 5/Fig. 6 result, TensorRT deployment, or real robot result.

## Effect on English Reading Report

The report now has a concrete engineering lesson: the reproduction did not only fail to match paper-level tracking; it identified a specific endpoint-height failure mode. This supports a stronger discussion of how missing teacher checkpoints, retargeting details, and curriculum choices affect reproducibility in humanoid control papers.

## Next Step

Inspect retargeted ankle height and body-index consistency directly. The next mainline experiment should compare target versus robot ankle z trajectories at reset/early rollout, then test whether a termination warm-up or ankle-height retargeting correction reduces the >99% endpoint-threshold violation rate.

## Git Commit

Pending at the time this progress file was written.
