# MuJoCo Observation Math Parity Audit

- Status: `blocked_observation_runtime_parity_missing_but_math_fixtures_pass`
- Generated: `2026-06-24T03:38:42.922560+00:00`
- Scope: formula fixture only; no IsaacLab task rollout, no MuJoCo physics claim.
- 当前不得声称完整复现 BeyondMimic；本审计只证明部分 observation 数学公式与 IsaacLab helper 对齐。

## Max Absolute Errors

- `base_velocity_body_frame`: `8.881784197001252e-16`
- `matrix_from_quat`: `4.440892098500626e-16`
- `motion_command_delta_yaw_sign_invariant`: `3.885780586188048e-16`
- `rot6_flatten_order`: `6.661338147750939e-16`
- `subtract_frame_transforms_position`: `9.992007221626409e-16`
- `subtract_frame_transforms_quat_sign_invariant`: `2.220446049250313e-16`
- `yaw_quat_sign_invariant`: `2.498001805406602e-16`

## Failed / Blocking Checks

- `native_probe_anchor_alignment_proven_equivalent_to_observation_manager`
- `runtime_observation_manager_sample_available`
- `native_obs_runtime_parity_ready`

## Interpretation

- Quaternion, Rot6D, yaw-only quaternion, frame subtraction, and body-frame velocity formulas match IsaacLab math fixtures.
- The current native probe still uses a local `world_to_init` anchor alignment for centered MuJoCo visualization.
- That alignment has not been proven numerically equivalent to the official IsaacLab observation_manager output.
- Therefore native MuJoCo PPO/VAE/diffusion videos remain blocked until a real observation_manager parity sample is captured.
