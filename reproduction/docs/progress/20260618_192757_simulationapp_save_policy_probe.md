# Progress Update

## Goal

Strengthen the IsaacLab live/replay blocker audit by comparing raw Isaac Sim `SimulationApp` and IsaacLab `AppLauncher` USD save behavior. This round does not run replay, PPO training, DAgger, VAE rollout, diffusion closed-loop evaluation, or robot execution.

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
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/tracking_official_replay_conversion_audit.py`
- `reproduction/scripts/final_reproduction_report.py`

## Files Modified

- `reproduction/scripts/tracking_simulationapp_save_policy_probe.py`
- `reproduction/scripts/artifact_manifest.py`
- `reproduction/scripts/reproduction_master_audit.py`
- `reproduction/scripts/tracking_official_replay_conversion_audit.py`
- `reproduction/scripts/final_reproduction_report.py`
- `reproduction/docs/final_reproduction_report.md`
- `res/tracking/simulationapp_save_policy_probe/tracking_simulationapp_save_policy_probe.json`
- `res/tracking/official_replay_conversion/tracking_official_replay_conversion_audit.json`
- `res/artifact_manifest/artifact_manifest.json`
- `res/artifact_manifest/artifact_manifest.tsv`
- `res/final_report/final_reproduction_report.json`
- `res/final_report/reproduction_report.md`
- `res/master_audit/reproduction_master_audit.json`
- `res/master_audit/reproduction_master_audit.tsv`
- verification audit JSON/TSV outputs refreshed by the standard verification bundle

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/tracking_simulationapp_save_policy_probe.py
envs/bm_analysis/bin/python reproduction/scripts/tracking_official_replay_conversion_audit.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
git status --short
git diff --stat
```

## Results

- Added a SimulationApp/AppLauncher USD save policy comparison probe.
- Raw `SimulationApp` with the IsaacLab headless experience reaches payload and records the same `permissionToSave=False` save blocker as `AppLauncher`.
- Raw `SimulationApp` with the Isaac Sim base python experience records a Vulkan device-lost crash before payload.
- Official replay conversion audit now records latest blocker: `isaaclab_headless_experience_layers_permission_to_save_false_with_isaacsim_base_vulkan_crash`.
- Artifact manifest now tracks `237` artifacts.
- Master audit now tracks `199` artifacts and remains `ok`.
- Paper-vs-reproduction comparison remains `122` rows.

## Verification

All required verification commands passed:

- `artifact_manifest.py`: `ok`, 237 artifacts
- `paper_vs_reproduction_comparison.py`: `ok`, 122 rows
- `final_reproduction_report.py`: `ok`
- `completion_matrix_status_audit.py`: `ok`, 161 rows, 0 invalid statuses
- `verification_command_syntax_audit.py`: `ok`, 0 failed scripts
- `verification_command_script_manifest.py`: `ok`, 161 scripts
- `verification_command_coverage_audit.py`: `ok`, 169 commands
- `reproduction_master_audit.py`: `ok`, 199 artifacts passed

## Failed / Blocked Items

- Official motion replay is still blocked. No valid official `motion.npz`, replay video, PPO checkpoint, teacher rollout dataset, or closed-loop tracking evaluation was produced.
- The current localized blocker is the IsaacLab headless Kit USD layer save policy: local layers are created with `permissionToSave=False`, and force-save/export attempts fail.
- The Isaac Sim base python experience also records a Vulkan device-lost crash on this host.
- This round is an environment/replay-gate audit, not a formal GPU experiment and not a paper-level result.

## Effect on English Reading Report

This strengthens the reproducibility discussion for the report: the project can now explain that IsaacLab/Isaac Sim package imports and AppLauncher startup are available, but official replay remains blocked by a lower-level Kit/USD save-policy issue rather than by missing Python dependencies alone. It supports a precise limitation statement without overstating reproduction success.

## Next Step

Investigate ways around the USD save-policy blocker: use a different Kit experience or launch flag, locate an official/preconverted G1 USD asset, or test a minimal USD write path outside the IsaacLab headless experience before retrying official `csv_to_npz.py` / replay.

## Git Commit

Pending at time of writing this progress update.

Current status remains: this project must not claim complete BeyondMimic reproduction unless all master audit and required paper-level gates pass.
