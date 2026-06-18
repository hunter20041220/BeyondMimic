# Progress Update

## Goal

Narrow the official replay blocker below URDF/MJCF conversion by testing whether USD/PXR layers can be saved from plain Python and from IsaacLab `AppLauncher(headless=True)`.

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
- `res/tracking/mjcf_stage_probe/tracking_mjcf_stage_probe.json`
- `res/tracking/official_replay_conversion/tracking_official_replay_conversion_audit.json`
- `.gitignore`

## Files Modified

- `.gitignore`
- `reproduction/scripts/tracking_usd_save_policy_probe.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/tracking_official_replay_conversion_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/final_reproduction_report.md`
- `res/tracking/usd_save_policy_probe/tracking_usd_save_policy_probe.json`
- `res/tracking/official_replay_conversion/tracking_official_replay_conversion_audit.json`
- Refreshed manifest, master audit, final report, final deliverables audit, and verification JSON/TSV outputs

## Commands Run

```bash
git status --short
git log --oneline --decorate -5
envs/bm_analysis/bin/python reproduction/scripts/tracking_usd_save_policy_probe.py
envs/bm_analysis/bin/python reproduction/scripts/tracking_official_replay_conversion_audit.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
git diff --stat
git check-ignore -v res/tracking/usd_save_policy_probe/app_launcher_stage/plain.usd res/tracking/usd_save_policy_probe/app_launcher_stage/plain.usda
```

## Results

- Added a minimal USD save-policy probe.
- Plain `bm_tracking` Python cannot import `pxr`, so standalone PXR/USD saving is unavailable outside Kit in this environment.
- Inside `AppLauncher(headless=True)`, `pxr` imports successfully and AppLauncher reaches the success sentinel.
- The probe tried `tmp`, `cache`, and `res` paths; `.usd` and `.usda`; plain `Save`, `Export`, and `SetPermissionToSave(True)` followed by `Save`.
- All `18` AppLauncher layer-save attempts failed.
- Every AppLauncher-created layer reported `permissionToSave=False`.
- Calling `SetPermissionToSave(True)` left `permissionToSave` false.
- The current official replay blocker is now refined to `app_launcher_layers_permission_to_save_false`.
- `.gitignore` now excludes `*.usd` and `*.usda` to prevent accidental upload of generated simulator assets or failed empty USD layers.

## Verification

All required verification commands passed.

- `artifact_manifest.py`: `ok`, `236` artifacts.
- `paper_vs_reproduction_comparison.py`: `ok`.
- `final_reproduction_report.py`: `ok`.
- `completion_matrix_status_audit.py`: `ok`, `161` rows.
- `verification_command_syntax_audit.py`: `ok`, `161` scripts, `0` failed.
- `verification_command_script_manifest.py`: `ok`, `161` scripts.
- `verification_command_coverage_audit.py`: `ok`, `169` commands, `10` smoke pass.
- `reproduction_master_audit.py`: `ok`, `198/198` master artifacts passed.

## Failed / Blocked Items

- Official replay remains blocked.
- The blocker is not currently explained by missing G1 meshes, URDF vs MJCF format, or plain filesystem paths.
- The immediate technical blocker is that local Sdf layers created inside AppLauncher are not saveable.
- No valid G1 USD, `motion.npz`, replay video, tracking task smoke metric, PPO checkpoint, teacher rollout dataset, VAE/diffusion closed-loop evidence, TensorRT deployment evidence, or real robot result was produced.

## Effect on English Reading Report

This gives the reading report a precise, evidence-backed limitation: the reproduction pipeline reaches IsaacLab headless startup but fails before official replay because Kit-created local USD layers have `permissionToSave=False`. This is more informative than a generic "IsaacLab failed" statement and supports a discussion of simulator state, USD layer policy, and hidden infrastructure requirements in robotics reproducibility.

## Next Step

Investigate why AppLauncher creates non-saveable local Sdf layers. Candidate probes: compare Isaac Sim's own standalone `SimulationApp` without IsaacLab `AppLauncher`, inspect loaded Kit/user config for read-only USD policy, try a different experience file, and search for settings or extensions that set layer permission. A valid minimal USD save is required before retrying G1 conversion or official `csv_to_npz`.

## Git Commit

Pending at the time this progress note is written.
