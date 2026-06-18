# Progress Update

## Goal

Continue the IsaacLab / whole_body_tracking recovery path by probing the deeper G1 URDF USD conversion blocker after the previous `Stage.Save()` to `Stage.Export()` workaround still produced empty destination/current stages.

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
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_urdf_stage_export_workaround/tracking_g1_urdf_stage_export_workaround_probe.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/official_replay_conversion/tracking_official_replay_conversion_audit.json`
- `/mnt/infini-data/test/BeyondMimic/envs/bm_tracking/lib/python3.10/site-packages/isaacsim/extscache/isaacsim.asset.importer.urdf-2.3.10+106.4.0.lx64.r.cp310/isaacsim/asset/importer/urdf/scripts/commands.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/IsaacLab-v2.1.0/source/isaaclab/isaaclab/sim/converters/urdf_converter.py`

## Files Modified

- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_urdf_layer_save_workaround_probe.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_official_replay_conversion_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- Generated/refreshed audit and report outputs under `/mnt/infini-data/test/BeyondMimic/res`.

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_urdf_layer_save_workaround_probe.py
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
git diff --check
```

## Results

- Added a bounded G1 URDF layer-save workaround probe.
- `Sdf.Layer.Save` can be monkeypatched in Python for a direct test layer, and the direct test layer writes a valid one-prim USD despite `permissionToSave=False`.
- The G1 importer still returns `(True, '/g1/pelvis')`, but the destination stage, current stage, and exported current stage contain no robot prims.
- Three generated configuration layers are present: base, physics, and sensor. Each is readable but empty from a robot-structure perspective.
- The probe records that the C++/Kit importer configuration-layer save path is not intercepted by the Python `Sdf.Layer.Save` monkeypatch.
- Latest official replay conversion blocker is now `sdf_layer_save_patch_applied_but_cpp_importer_save_path_not_intercepted`.
- No official `motion.npz`, replay video, PPO training, DAgger rollout, VAE/diffusion rollout, checkpoint, or robot result was produced or claimed.

## Verification

- `artifact_manifest.py`: passed, artifact count increased to `240`, missing `0`.
- `paper_vs_reproduction_comparison.py`: passed, `122` rows.
- `final_reproduction_report.py`: passed.
- `completion_matrix_status_audit.py`: passed, `161` rows, invalid statuses `0`.
- `verification_command_syntax_audit.py`: passed, failed `0`.
- `verification_command_script_manifest.py`: passed.
- `verification_command_coverage_audit.py`: passed, `169` commands, lightweight smoke `10/10`.
- `reproduction_master_audit.py`: passed.

## Failed / Blocked Items

- Official whole_body_tracking replay remains blocked.
- The Python `Sdf.Layer.Save` workaround is insufficient because the importer configuration-layer save path appears to execute below the Python monkeypatch boundary.
- A Vulkan `ERROR_DEVICE_LOST` message is still present during/after Kit shutdown in the probe log, but the payload was captured before shutdown and the audit records the blocker without treating it as replay success.
- Paper-level tracking replay, PPO training/evaluation, teacher rollout dataset, closed-loop VAE rollout, closed-loop diffusion guidance evaluation, Fig. 5/Fig. 6 videos/metrics, TensorRT/asynchronous deployment, and real robot results remain incomplete.

## Effect on English Reading Report

This adds a concrete, auditable example for the reproduction section: IsaacLab/AppLauncher can start and the official URDF importer can parse G1, but paper-level tracking replay is still blocked by an Isaac Sim USD conversion boundary. The report can now distinguish package/import readiness from true simulation/replay readiness with specific evidence instead of a generic environment failure.

## Next Step

Investigate importer-side alternatives that do not rely on Python `Sdf.Layer.Save`: supported Isaac/Kit conversion settings, exporter/importer command variants, a clean Isaac Sim conversion experience, or a valid preconverted G1 USD that can be traced to the official assets. Only after a valid G1 USD and official `motion.npz` exist should replay and tracking task smoke proceed.

## Git Commit

Pending at time of writing; this progress file should be committed with the layer-save workaround probe and refreshed audit outputs.
