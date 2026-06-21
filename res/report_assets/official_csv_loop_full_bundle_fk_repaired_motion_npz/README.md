# FK-Repaired Full Public-Motion Bundle

This directory summarizes a non-Kit URDF-FK repaired candidate for the full local G1 public-motion bundle.

## Claim Boundary

This is a local repair candidate for debugging. It is not official Isaac/Kit csv_to_npz output, not paper-level tracking, not DAgger, not VAE/diffusion evidence, and not real-robot evidence.

## Key Metrics

- Motion count: `40`
- Total frames: `11960`
- Mean z spread: `1.2178876399993896` m
- Left/right ankle mean z: `0.05472046881914139` / `0.05676112323999405` m

## Outputs

- Candidate NPZ: `/mnt/infini-data/test/BeyondMimic/res/tracking/official_csv_loop_full_bundle_fk_repaired_motion_npz/official_csv_loop_full_public_motion_bundle_fk_repaired.npz`
- Summary JSON: `/mnt/infini-data/test/BeyondMimic/res/tracking/official_csv_loop_full_bundle_fk_repaired_motion_npz/tracking_g1_official_csv_loop_full_bundle_fk_repaired_motion_npz.json`
- Target height plot: `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_full_bundle_fk_repaired_motion_npz/fk_repaired_target_body_heights.png`
- Per-motion spread plot: `/mnt/infini-data/test/BeyondMimic/res/report_assets/official_csv_loop_full_bundle_fk_repaired_motion_npz/fk_repaired_per_motion_spread.png`
