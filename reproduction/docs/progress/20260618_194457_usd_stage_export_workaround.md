# Progress Update

## Goal

Continue resolving the IsaacLab live/replay gate by narrowing the USD write blocker discovered in the official whole-body-tracking conversion path. This round specifically tested whether any direct USD API can write a local stage inside the same IsaacLab headless Kit session where `layer.Save()` is blocked.

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
- IsaacLab converter sources under `reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab/isaaclab/sim/converters/`
- IsaacLab and Isaac Sim Kit experience files under `reproduction/third_party/official/IsaacLab-v2.1.0/apps/` and `envs/bm_tracking/lib/python3.10/site-packages/isaacsim/apps/`

## Files Modified

- `reproduction/scripts/tracking_usd_api_variant_probe.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/tracking_official_replay_conversion_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/known_limitations.md`
- `reproduction/docs/final_reproduction_report.md`
- `res/tracking/usd_api_variant_probe/tracking_usd_api_variant_probe.json`
- `res/tracking/official_replay_conversion/tracking_official_replay_conversion_audit.json`
- refreshed manifest, master-audit, final-report, final-deliverables, and verification-command audit outputs under `res/`

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/tracking_usd_api_variant_probe.py
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

- Added a bounded IsaacLab headless USD API variant probe.
- Confirmed `layer.Save()` / `Sdf.Layer.CreateNew(...).Export(...)` are still blocked by `permissionToSave=False`.
- Found a concrete local USD write workaround inside the same headless Kit session: `Usd.Stage.Export(...)` writes non-empty USD files.
- Successful API labels: `create_new_stage_export`, `create_in_memory_stage_export`, `sdf_layer_create_anonymous_export`.
- Official replay conversion audit now records latest blocker/action as `layer_save_blocked_but_stage_export_succeeds`.
- Artifact manifest now tracks `238` artifacts.
- Master audit now tracks `200` artifacts and remains `ok`.
- Paper-vs-reproduction comparison remains `122` rows.

## Verification

All required verification commands passed:

- `artifact_manifest.py`: `ok`, 238 artifacts
- `paper_vs_reproduction_comparison.py`: `ok`, 122 rows
- `final_reproduction_report.py`: `ok`
- `completion_matrix_status_audit.py`: `ok`, 161 rows, 0 invalid statuses
- `verification_command_syntax_audit.py`: `ok`, 0 failed scripts
- `verification_command_script_manifest.py`: `ok`, 161 scripts
- `verification_command_coverage_audit.py`: `ok`, 169 commands
- `reproduction_master_audit.py`: `ok`, 200 artifacts passed
- `git diff --check`: clean

## Failed / Blocked Items

- Official replay is still blocked. This round did not produce a valid official G1 USD, official `motion.npz`, replay video, tracking task smoke, PPO checkpoint, teacher rollout dataset, or closed-loop evaluation.
- The importer/converter save path still needs to be adapted or bypassed. The new evidence only shows that a direct `Usd.Stage.Export(...)` path can write local USD files.
- This is an environment/conversion-gate audit, not a formal GPU training experiment.

## Effect on English Reading Report

This improves the reproducibility narrative: the report can now say the project did not merely fail at IsaacLab, but isolated a specific conversion/write-path issue and found a plausible API-level workaround. It is useful evidence for a careful limitations section and for explaining the next engineering step before official replay or PPO training.

## Next Step

Patch or wrap the official G1 URDF/MJCF conversion path to use a `Usd.Stage.Export(...)`-compatible flow, or supply a preconverted G1 USD, then retry official `csv_to_npz.py` / replay. If a valid G1 USD is generated, immediately validate prim count, default prim, articulation/body schema, then rerun official replay conversion.

## Git Commit

Pending at time of writing this progress update.

Current status remains: this project must not claim complete BeyondMimic reproduction unless all master audit and required paper-level gates pass.
