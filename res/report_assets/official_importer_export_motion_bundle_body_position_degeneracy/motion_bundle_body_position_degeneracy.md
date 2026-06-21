# Motion Bundle Body-Position Degeneracy Audit

## Finding

The current full public-motion official-loop bundle has a valid outer schema but degenerate body_pos_w: every body is effectively at the root-like position. The URDF-FK candidate from the same public CSV and G1 URDF separates pelvis, ankles, torso, and wrists into plausible heights. The current bundle should therefore not be used as trusted target-body position evidence for teacher-quality PPO, DAgger, VAE, diffusion, or paper-level closed-loop evaluation until the body position generation path is repaired and replayed.

## Key Numbers

- Bundle shape: `[11960, 40, 3]`.
- Bundle max body-minus-root position spread: `7.153e-07` m.
- Bundle max z spread: `4.768e-07` m.
- FK candidate mean z spread: `1.238` m.
- Endpoint ankle exceed-rate mean values: `[0.9983293922449833, 0.9949914428302675]`.

## Claim Boundary

This is local diagnostic evidence only. It is not official BeyondMimic preprocessing, not a trained teacher, not DAgger, not Fig. 5/Fig. 6, and not real-robot evidence.

## Outputs

- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_motion_bundle_body_position_degeneracy/body_z_summary.csv`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_motion_bundle_body_position_degeneracy/body_position_spread_summary.csv`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_motion_bundle_body_position_degeneracy/target_body_height_contrast.csv`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_motion_bundle_body_position_degeneracy/body_z_mean_contrast.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_motion_bundle_body_position_degeneracy/body_z_spread_timeseries.png`
- `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_importer_export_motion_bundle_body_position_degeneracy/motion_bundle_body_position_degeneracy_audit.json`
