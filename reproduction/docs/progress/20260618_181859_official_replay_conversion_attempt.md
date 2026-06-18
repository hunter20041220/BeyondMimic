# Progress Update

## Goal

Advance from the IsaacLab live headless gate to a bounded official `whole_body_tracking` motion conversion attempt, while preserving failed-run evidence and avoiding any false claim of replay/training success.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/res/setup/isaaclab_live_gate_probe/isaaclab_live_gate_probe.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/official_replay_preflight/tracking_official_replay_preflight.json`
- `/mnt/infini-data/test/BeyondMimic/reproduction/generated/whole_body_tracking_local/csv_to_npz_local.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/validate_motion_npz_contract.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/robots/g1.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab/isaaclab/sim/converters/asset_converter_base.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab/isaaclab/sim/converters/urdf_converter.py`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/generated/whole_body_tracking_local/csv_to_npz_local.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_official_replay_conversion_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- Refreshed final report, artifact manifest, final deliverables audit, verification manifests, and master audit under `/mnt/infini-data/test/BeyondMimic/res`

## Commands Run

```bash
envs/bm_tracking/bin/python -m pip install --no-deps rsl-rl-lib==2.3.1
apt-get update && apt-get install -y libglu1-mesa
python3 reproduction/scripts/tracking_official_replay_conversion_audit.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/reproduction_master_audit.py
```

Several bounded `csv_to_npz_local.py` attempts were run and retained under:

`/mnt/infini-data/test/BeyondMimic/logs/tracking_official_replay_conversion`

## Results

- Restored IsaacLab-compatible RSL-RL:
  - `rsl-rl-lib==2.3.1`
  - `torch==2.5.1+cu121` retained
  - `pip check` passes
  - `import rsl_rl.env` passes
- Installed missing host runtime library:
  - `libGLU.so.1` via `libglu1-mesa`
- Removed the previous CUDA ordinal mistake from the conversion command:
  - Do not set `CUDA_VISIBLE_DEVICES=5,6` while passing Isaac/Omniverse physical `cuda:6`
- Added a local generated-script adaptation so URDF USD output uses:
  - `/mnt/infini-data/test/BeyondMimic/tmp/isaaclab_usd/g1_cylinder`
  - This avoids the hard-coded `/tmp/IsaacLab` default and keeps generated assets under project storage.
- Added machine-readable conversion audit:
  - `/mnt/infini-data/test/BeyondMimic/res/tracking/official_replay_conversion/tracking_official_replay_conversion_audit.json`
- Current conversion status:
  - `ok_with_blocked_conversion`
  - latest blocker: `urdf_usd_save_not_allowed`

## Verification

All required verification commands passed. Latest verification log:

`/mnt/infini-data/test/BeyondMimic/logs/setup/verification_20260618_181815_official_replay_conversion.log`

Current audit summary:

- artifact manifest: `232` artifacts
- master audit: `194/194` passed
- paper-vs-reproduction comparison: `122` rows
- conversion audit: `ok_with_blocked_conversion`

## Failed / Blocked Items

- No valid official `motion.npz` was produced.
- No replay video was produced.
- No tracking task smoke/eval was run.
- No PPO training was started.
- Isaac/Kit conversion still fails after URDF importer stage with:
  - USD save policy error: `saving not allowed`
  - unresolved robot reference prim
  - no contact sensors / no rigid bodies under `/World/envs/env_0/Robot`
- CUDA P2P/IOMMU warnings remain runtime warnings, not paper-level blockers by themselves.

## Effect on English Reading Report

This round adds concrete reproduction-process evidence: environment recovery was not only import-level. The report can honestly state that official tracking replay was attempted, that several environment blockers were repaired, and that the current remaining blocker is inside Isaac Sim URDF-to-USD conversion rather than missing Python packages. It also reinforces the limitation section: live gate success does not imply official replay or paper-level tracking metrics.

## Next Step

Investigate the Isaac Sim URDF importer USD save policy for generated files. Possible next probes are a minimal URDF conversion-only script, a pre-generated G1 USD asset under project cache, or a conversion variant that disables post-import USD save paths while preserving the official robot config boundary. Do not start PPO until a valid `motion.npz` conversion and replay smoke pass.

## Git Commit

Pending at time of writing.
