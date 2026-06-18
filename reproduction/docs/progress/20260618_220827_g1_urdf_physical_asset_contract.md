# Progress Update

## Goal

Move the official tracking replay gate one step closer by auditing whether the public official G1 URDF contains the
physical asset fields needed to enrich the previously generated 29-DoF skeleton USD.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/PROGRESS.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/RUNBOOK.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_urdf_skeleton_usd/tracking_g1_official_urdf_skeleton_usd_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_action_scale_audit/tracking_g1_action_scale_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/official_source_contract_audit/tracking_official_source_contract_audit.json`
- `/mnt/infini-data/test/BeyondMimic/reproduction/third_party/official/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf`

## Files Modified

- Added `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_urdf_physical_asset_contract_audit.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`.
- Regenerated final-report, verification-command, artifact-manifest, and master-audit outputs under `res/`.

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_urdf_physical_asset_contract_audit.py
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

- New audit status: `ok_with_physical_contract_ready_for_converter_scaffold`.
- Official G1 URDF contract: `40` links, `39` joints, `29` non-fixed joints, `10` fixed joints.
- Visual mesh references: `35`; missing mesh files: `0`.
- Collision elements: `29` total, across `14` links; collision types are `28` cylinders and `1` sphere.
- Inertial links: `37`; missing inertial tags are `imu_in_pelvis`, `imu_in_torso`, and `mid360_link`.
- Target bodies with missing inertial tags: `0`.
- All `29` non-fixed joints have axes, limits, and local action-drive rows from the action-scale audit.
- This is a converter contract audit only. It does not generate a physically faithful USD, `motion.npz`, replay, PPO
  training, policy evaluation, video, or robot result.

## Verification

- `artifact_manifest.py`: passed, `249` artifacts, missing `0`.
- `paper_vs_reproduction_comparison.py`: passed.
- `final_reproduction_report.py`: passed.
- `completion_matrix_status_audit.py`: passed, `161` rows, invalid statuses `0`.
- `verification_command_syntax_audit.py`: passed, `177` scripts, failed `0`.
- `verification_command_script_manifest.py`: passed, `177` scripts.
- `verification_command_coverage_audit.py`: passed, `185` commands, smoke pass `10`.
- `reproduction_master_audit.py`: passed.

## Failed / Blocked Items

- No physically faithful full-robot USD was generated.
- No official `motion.npz`, `replay_npz.py` success, PPO training, policy evaluation, DAgger rollout, VAE/diffusion
  closed-loop evaluation, TensorRT deployment, Fig. 5/Fig. 6 video, or real-robot result was produced.
- Current replay blocker remains the absence of a validated physical USD/replay pipeline, not missing URDF source
  metadata.
- Current full-paper state remains incomplete; `goal_complete=false`.

## Effect on English Reading Report

This update sharpens the reproduction narrative: the public official assets contain enough structured URDF information
to build an offline G1 converter scaffold, but the Isaac/Kit converter path and replay validation remain unresolved.
That distinction is useful for explaining why robotics reproduction can fail at the simulator asset boundary even when
code, meshes, and kinematic contracts are locally available.

## Next Step

Implement a bounded offline USD enrichment pass that writes mesh references, collision primitives, inertial tensors,
joint origins/axes/limits, and drive metadata into the skeleton USD, then validate whether official `csv_to_npz.py` can
consume it. Keep it labeled as resource-adjusted until replay succeeds.

## Git Commit

Pending at time of writing this progress file.
