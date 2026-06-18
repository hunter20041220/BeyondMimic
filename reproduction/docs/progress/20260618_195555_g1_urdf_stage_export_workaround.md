# Progress Update

## Goal

Test whether the previously discovered `Usd.Stage.Export(...)` workaround can be applied to the real official G1 URDF import path used before `csv_to_npz.py` / replay. This round focuses on the IsaacLab/Isaac Sim conversion gate only; it does not run official replay or training.

## Files Read

- `goal.md`
- `README.md`
- `reproduction/RUNBOOK.md`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/final_reproduction_report.md`
- `res/comparison/paper_vs_reproduction.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/master_audit/reproduction_master_audit.json`
- `res/required_artifact_absence/required_artifact_absence_audit.json`
- `res/tracking/usd_api_variant_probe/tracking_usd_api_variant_probe.json`
- `res/tracking/official_replay_conversion/tracking_official_replay_conversion_audit.json`
- Isaac Sim URDF importer command source under `envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/isaacsim.asset.importer.urdf-*/`
- IsaacLab `UrdfConverter`, `MjcfConverter`, and converter base sources
- Existing local `csv_to_npz_local.py` generated runner

## Files Modified

- `reproduction/scripts/tracking_g1_urdf_stage_export_workaround_probe.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/tracking_official_replay_conversion_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/experiment_protocol.md`
- `reproduction/docs/final_reproduction_report.md`
- `res/tracking/g1_urdf_stage_export_workaround/tracking_g1_urdf_stage_export_workaround_probe.json`
- refreshed manifest, master audit, final report, final deliverables, and verification-command audits under `res/`

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_urdf_stage_export_workaround_probe.py
envs/bm_analysis/bin/python reproduction/scripts/tracking_official_replay_conversion_audit.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
git diff --check
```

## Results

- Added a G1-specific URDF importer workaround probe.
- The probe monkeypatches the importer's initial `Usd.Stage.CreateNew(...).Save()` call so the save is routed through `Usd.Stage.Export(...)`.
- The patch is applied and recorded, and the official importer returns `(True, '/g1/pelvis')`.
- However, the destination USD, current stage, and exported current stage remain empty of robot prims.
- Logs show deeper generated layers such as `configuration/*_base.usd`, `configuration/*_physics.usd`, and `configuration/*_sensor.usd` still fail with `permissionToSave=False` save errors.
- Latest official replay conversion blocker is now classified as `stage_export_patch_applied_but_importer_output_empty`.
- Artifact manifest now tracks `239` artifacts.
- Master audit now tracks `201` artifacts and remains `ok`.
- Paper-vs-reproduction comparison remains `122` rows.

## Verification

All required verification commands passed:

- `artifact_manifest.py`: `ok`, 239 artifacts
- `paper_vs_reproduction_comparison.py`: `ok`, 122 rows
- `final_reproduction_report.py`: `ok`
- `completion_matrix_status_audit.py`: `ok`, 161 rows, 0 invalid statuses
- `verification_command_syntax_audit.py`: `ok`, 0 failed scripts
- `verification_command_script_manifest.py`: `ok`, 161 scripts
- `verification_command_coverage_audit.py`: `ok`, 169 commands
- `reproduction_master_audit.py`: `ok`, 201 artifacts passed
- `git diff --check`: clean

## Failed / Blocked Items

- Official replay remains blocked.
- No valid official G1 USD was produced.
- No official `motion.npz`, replay video, tracking task smoke, PPO checkpoint, teacher rollout dataset, or closed-loop evaluation was produced.
- The next blocker is deeper than the initial destination-stage save: Isaac Sim's URDF importer attempts to save generated base/physics/sensor configuration layers, and those layer saves remain blocked by `permissionToSave=False`.

## Effect on English Reading Report

This adds a useful negative engineering result: the reproduction effort found a plausible USD write workaround, tested it on the actual G1 importer path, and showed why it is insufficient. The reading report can use this as concrete evidence that the remaining simulation blocker is a layered Isaac Sim importer/save-policy issue, not a vague environment failure.

## Next Step

Probe a broader monkeypatch around `Sdf.Layer.Save` / generated configuration-layer save calls, or locate a valid preconverted official G1 USD. If a valid G1 USD is generated, validate default prim, joint/body counts, articulation schema, then retry official `csv_to_npz.py` and `replay_npz.py`.

## Git Commit

Pending at time of writing this progress update.

Current status remains: this project must not claim complete BeyondMimic reproduction unless all master audit and required paper-level gates pass.
