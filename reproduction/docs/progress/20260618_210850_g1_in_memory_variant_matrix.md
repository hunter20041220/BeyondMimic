# Progress Update

## Goal

Continue the official whole_body_tracking replay gate recovery after the live IsaacLab/AppLauncher sentinel had already
passed. This round tests whether the G1 URDF in-memory import crash is specific to GPU 6, AppLauncher defaults, or
ordinary RTX/waitIdle settings.

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
- `res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json`
- `res/tracking/g1_urdf_simulationapp_in_memory_import/tracking_g1_urdf_simulationapp_in_memory_import_probe.json`
- `res/tracking/official_replay_conversion/tracking_official_replay_conversion_audit.json`
- IsaacLab and Isaac Sim `.kit` app files under `reproduction/third_party/official/IsaacLab-v2.1.0/apps` and
  `envs/bm_tracking/lib/python3.10/site-packages/isaacsim/apps`

## Files Modified

- Added `reproduction/scripts/tracking_g1_urdf_in_memory_variant_matrix_probe.py`
- Updated `reproduction/scripts/artifact_manifest.py`
- Updated `reproduction/scripts/tracking_official_replay_conversion_audit.py`
- Updated `reproduction/scripts/reproduction_master_audit.py`
- Updated `reproduction/scripts/final_reproduction_report.py`
- Updated `reproduction/docs/known_limitations.md`
- Updated `reproduction/docs/experiment_protocol.md`
- Regenerated `reproduction/docs/final_reproduction_report.md`
- Regenerated small audit outputs under `res/artifact_manifest`, `res/final_report`, `res/master_audit`,
  `res/final_deliverables_audit`, `res/verification_command_script_manifest`,
  `res/verification_command_coverage`, and `res/tracking/official_replay_conversion`

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_urdf_in_memory_variant_matrix_probe.py
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

- New probe JSON:
  `res/tracking/g1_urdf_in_memory_variant_matrix/tracking_g1_urdf_in_memory_variant_matrix_probe.json`
- Matrix status: `ok_with_no_valid_g1_usd`
- Current blocker: `variant_matrix_no_valid_g1_usd`
- `gpu6_headless_single_gpu`: reached the in-memory importer branch, then Vulkan device loss before payload.
- `gpu5_headless_single_gpu`: reached the in-memory importer branch, then Vulkan device loss before payload.
- `gpu6_headless_wait_idle_low_rtx`: reached the in-memory importer branch, then Vulkan device loss before payload.
- `gpu6_headless_rendering_experience`: crashed before the app sentinel, during viewport/Hydra startup.
- No case produced a valid current-stage or exported G1 USD.
- No official `motion.npz`, replay video, PPO training, policy evaluation, DAgger rollout, VAE/diffusion closed-loop run,
  checkpoint, or robot result is claimed.

## Verification

- `artifact_manifest.py`: passed, 243 artifacts, missing count 0
- `paper_vs_reproduction_comparison.py`: passed, 122 comparison rows retained
- `final_reproduction_report.py`: passed
- `completion_matrix_status_audit.py`: passed, 161 rows, 0 invalid statuses
- `verification_command_syntax_audit.py`: passed, 161 scripts, 0 failures
- `verification_command_script_manifest.py`: passed, 161 scripts
- `verification_command_coverage_audit.py`: passed, 169 commands, 10 smoke-pass commands
- `reproduction_master_audit.py`: passed

## Failed / Blocked Items

- The official tracking replay gate remains blocked.
- The matrix makes simple GPU switching and basic RTX/waitIdle downgrades unlikely to solve the G1 URDF in-memory import
  crash on this host.
- The next recovery path should prioritize a trusted preconverted G1 USD or a lower-level/offline URDF conversion route
  before retrying official `csv_to_npz.py` / `replay_npz.py`.

## Effect on English Reading Report

This improves the reading report's reproduction narrative by showing a concrete negative result: after restoring the
IsaacLab live-app gate, the next blocker was investigated across multiple runtime variants and localized to the G1 URDF
conversion/runtime path. This is stronger evidence than a single failed run and supports a careful limitation statement.

## Next Step

Search the local official/downloaded assets for any already-converted G1 USD that can be trusted, or build an offline
conversion route that avoids the current in-memory Kit/Vulkan crash. Only after a valid G1 USD exists should the
official `csv_to_npz.py` / `replay_npz.py` gate be retried.

## Git Commit

Pending at the time this progress note is written.
