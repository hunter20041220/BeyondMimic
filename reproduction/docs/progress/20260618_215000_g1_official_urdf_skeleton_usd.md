# Progress Update

## Goal

Advance the blocked official G1 replay gate without claiming replay success by creating an auditable, minimal
official-URDF-derived 29-DoF skeleton USD scaffold and wiring it into the reproduction audit chain.

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
- `/mnt/infini-data/test/BeyondMimic/res/tracking/official_replay_conversion/tracking_official_replay_conversion_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_reference_usd_compatibility_audit/tracking_g1_reference_usd_compatibility_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/usd_api_variant_probe/tracking_usd_api_variant_probe.json`

## Files Modified

- Added `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_official_urdf_skeleton_usd_audit.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_official_replay_conversion_audit.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`.
- Regenerated audited JSON/TSV/Markdown outputs under `res/artifact_manifest`, `res/master_audit`,
  `res/final_report`, `res/verification_command_*`, and `res/tracking/official_replay_conversion`.

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_official_urdf_skeleton_usd_audit.py
envs/bm_analysis/bin/python reproduction/scripts/tracking_official_replay_conversion_audit.py
python3 reproduction/scripts/paper_vs_reproduction_comparison.py
python3 reproduction/scripts/completion_matrix_status_audit.py
python3 reproduction/scripts/verification_command_syntax_audit.py
python3 reproduction/scripts/verification_command_script_manifest.py
python3 reproduction/scripts/verification_command_coverage_audit.py
python3 reproduction/scripts/final_reproduction_report.py
python3 reproduction/scripts/artifact_manifest.py
python3 reproduction/scripts/reproduction_master_audit.py
```

## Results

- Created `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_urdf_skeleton_usd/g1_official_urdf_29dof_skeleton.usda`.
- Created `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_urdf_skeleton_usd/tracking_g1_official_urdf_skeleton_usd_audit.json`.
- Skeleton audit status: `ok_with_minimal_29dof_skeleton_usd`.
- Skeleton contract: `40` links, `29` revolute joints, `10` fixed joints, `40` rigid bodies, `1` articulation root.
- Official action-joint diff: no missing action joints.
- Official target-body diff: no missing target bodies.
- The generated USD is a placeholder structure scaffold only. It does not include physical mesh/collision/inertia/drive
  fidelity and is not official IsaacLab URDF converter success.
- Official replay conversion remains `ok_with_blocked_conversion`; latest blocker is now
  `minimal_skeleton_usd_lacks_physical_fidelity_for_replay`.

## Verification

- `artifact_manifest.py`: passed, `247` artifacts, missing `0`.
- `paper_vs_reproduction_comparison.py`: passed.
- `final_reproduction_report.py`: passed.
- `completion_matrix_status_audit.py`: passed, `161` rows, invalid statuses `0`.
- `verification_command_syntax_audit.py`: passed, `176` scripts, failed `0`.
- `verification_command_script_manifest.py`: passed, `176` scripts.
- `verification_command_coverage_audit.py`: passed, `184` commands, smoke pass `10`.
- `reproduction_master_audit.py`: passed, `209/209` artifacts, failures `0`.

## Failed / Blocked Items

- No official `motion.npz` was produced.
- No official replay, PPO training, policy evaluation, DAgger rollout, VAE closed-loop rollout, diffusion closed-loop
  evaluation, Fig. 5/Fig. 6 rollout video, TensorRT deployment, or real-robot result was produced.
- The minimal skeleton USD still needs official mesh, collision, inertia, and actuator-drive fidelity before it can be
  tested as a real replay asset.
- Current full-paper state remains incomplete; `goal_complete=false`.

## Effect on English Reading Report

This update gives the reading report a concrete reproducibility case study: the public code and assets expose the
official 29-DoF G1 kinematic naming contract, but the live Isaac/Kit converter path remains fragile. The skeleton USD
evidence can be described as a cautious engineering scaffold, not as paper-level tracking reproduction.

## Next Step

Use the skeleton USD as an offline converter scaffold: add official visual/collision meshes, inertial values, joint
limits, and drive metadata from the official URDF/assets, then rerun `csv_to_npz.py` and `replay_npz.py` as a clearly
bounded replay gate. Long tracking training should still wait until replay is valid.

## Git Commit

Pending at time of writing this progress file.
