# Progress Update

## Goal

Move the official tracking replay gate closer by turning the audited G1 URDF physical contract into a concrete,
resource-adjusted enriched USD scaffold that can be read back in Kit.

## Files Read

- `/mnt/infini-data/test/BeyondMimic/goal.md`
- `/mnt/infini-data/test/BeyondMimic/README.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`
- `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`
- `/mnt/infini-data/test/BeyondMimic/res/comparison/paper_vs_reproduction.json`
- `/mnt/infini-data/test/BeyondMimic/res/artifact_manifest/artifact_manifest.json`
- `/mnt/infini-data/test/BeyondMimic/res/master_audit/reproduction_master_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/required_artifact_absence/required_artifact_absence_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_urdf_physical_asset_contract_audit/tracking_g1_urdf_physical_asset_contract_audit.json`
- `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_official_urdf_skeleton_usd/g1_official_urdf_29dof_skeleton.usda`

## Files Modified

- Added `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/tracking_g1_resource_adjusted_enriched_usd_probe.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/artifact_manifest.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/reproduction_master_audit.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/scripts/final_reproduction_report.py`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/docs/known_limitations.md`.
- Updated `/mnt/infini-data/test/BeyondMimic/reproduction/docs/experiment_protocol.md`.
- Regenerated final-report, verification-command, artifact-manifest, and master-audit outputs under `res/`.

## Commands Run

```bash
envs/bm_analysis/bin/python reproduction/scripts/tracking_g1_resource_adjusted_enriched_usd_probe.py
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

- New probe status: `ok_with_resource_adjusted_enriched_usd_scaffold`.
- Generated enriched scaffold USD:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_enriched_usd/g1_resource_adjusted_29dof_enriched_scaffold.usda`.
- Generated enrichment contract:
  `/mnt/infini-data/test/BeyondMimic/res/tracking/g1_resource_adjusted_enriched_usd/g1_resource_adjusted_enrichment_contract.json`.
- Kit readback counts: `40` links, `37` MassAPI links, `35` visual proxies, `35` mesh references, `29` collision
  proxies, `29` CollisionAPI prims, `29` revolute joints, `29` joint-limit rows, `29` drive metadata rows, `1`
  articulation root.
- The scaffold authors public URDF mass/inertia metadata, visual mesh references, collision proxy geometry, joint
  limits, and drive metadata onto the previous 29-DoF skeleton USD.

## Verification

- `artifact_manifest.py`: passed, `252` artifacts, missing `0`.
- `paper_vs_reproduction_comparison.py`: passed.
- `final_reproduction_report.py`: passed.
- `completion_matrix_status_audit.py`: passed, `161` rows, invalid statuses `0`.
- `verification_command_syntax_audit.py`: passed, `178` scripts, failed `0`.
- `verification_command_script_manifest.py`: passed, `178` scripts.
- `verification_command_coverage_audit.py`: passed, `186` commands, smoke pass `10`.
- `reproduction_master_audit.py`: passed.

## Failed / Blocked Items

- The enriched scaffold is not official IsaacLab URDF converter output.
- It has not passed official `csv_to_npz.py` or `replay_npz.py`.
- No official `motion.npz`, replay video, PPO tracking training/evaluation, DAgger rollout, VAE/diffusion closed-loop
  evaluation, TensorRT deployment, Fig. 5/Fig. 6 result, or real-robot result was produced.
- Current full-paper state remains incomplete; `goal_complete=false`.

## Effect on English Reading Report

This update gives the reading report a concrete reproduction engineering story: the project moved from a minimal
29-DoF kinematic skeleton to a read-back-validated resource-adjusted USD scaffold, while still preserving the honest
boundary that paper-level replay has not been achieved.

## Next Step

Run a bounded official `csv_to_npz.py` / `replay_npz.py` preflight against the enriched scaffold or refine the authored
USD until the official preprocessing path accepts it. Keep all results labeled resource-adjusted unless official replay
validation succeeds.

## Git Commit

Pending at time of writing this progress file.
