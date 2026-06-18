# Progress Update

## Goal

Test a second recovery route for the official G1 URDF conversion blocker by omitting `dest_path` and importing the robot into the current in-memory Kit stage before attempting `Stage.Export()`.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/final_reproduction_report.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_urdf_layer_save_workaround/tracking_g1_urdf_layer_save_workaround_probe.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/official_replay_conversion/tracking_official_replay_conversion_audit.json`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_urdf_in_memory_import_probe.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_official_replay_conversion_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- Refreshed audit/report outputs under `/mnt/infini-data/test/BeyondMimic/res`.

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_urdf_in_memory_import_probe.py
envs/bm_analysis/bin/python reproduction/scripts/tracking_official_replay_conversion_audit.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

## Results

- Added a bounded in-memory import probe for the official G1 URDF.
- First version with explicit `omni.usd.get_context().new_stage()` hit Vulkan `ERROR_DEVICE_LOST` before payload.
- Revised version avoids explicit `new_stage()` and uses `dest_path=""`.
- The log records Isaac's importer warning: `Creating Asset in an in-memory stage, will not create layered structure`, which confirms the attempted route is meaningfully different from the layered `dest_path` conversion path.
- The run still hits Vulkan `ERROR_DEVICE_LOST` before the probe can capture a payload or exported robot stage.
- Official replay conversion audit now reports latest blocker `in_memory_import_vulkan_device_lost_before_payload`.
- No valid G1 USD, official `motion.npz`, replay video, PPO run, checkpoint, VAE/diffusion rollout, or robot result was produced or claimed.

## Verification

- `artifact_manifest.py`: passed, artifact count `241`, missing `0`.
- `paper_vs_reproduction_comparison.py`: passed, `122` rows.
- `final_reproduction_report.py`: passed.
- `completion_matrix_status_audit.py`: passed, `161` rows, invalid statuses `0`.
- `verification_command_syntax_audit.py`: passed, failed `0`.
- `verification_command_script_manifest.py`: passed.
- `verification_command_coverage_audit.py`: passed, `169` commands, lightweight smoke `10/10`.
- `reproduction_master_audit.py`: passed.

## Failed / Blocked Items

- In-memory import currently cannot be used as the official replay conversion workaround because Vulkan device loss occurs before a current-stage payload/export can be captured.
- The previous layered-output path remains blocked by `permissionToSave=False`, empty destination/current stages, and a C++/Kit save path that is not intercepted by Python `Sdf.Layer.Save`.
- Paper-level tracking replay, PPO training/evaluation, teacher rollout dataset, closed-loop VAE rollout, closed-loop diffusion guidance evaluation, Fig. 5/Fig. 6 videos/metrics, TensorRT/asynchronous deployment, and real robot results remain incomplete.

## Effect on English Reading Report

This gives the reading report a sharper engineering narrative: the reproduction did not stop at package installation. It tested both file-backed/layered and in-memory URDF conversion routes and found distinct blockers. That supports a credible discussion of why official-code tracking replay is not yet paper-level reproduced.

## Next Step

Focus on Vulkan/runtime stability for the in-memory importer branch or locate a supported headless conversion experience/settings combination that can complete current-stage import and export. If a valid exported G1 USD is obtained, rerun official `csv_to_npz` and then `replay_npz` before any PPO or closed-loop claims.

## Git Commit

Pending at time of writing; this progress update should be committed with the in-memory import probe and refreshed audit outputs.
