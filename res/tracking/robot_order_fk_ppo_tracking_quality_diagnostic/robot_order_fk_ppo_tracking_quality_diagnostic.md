# Robot-Order FK PPO Tracking Quality Diagnostic

This is a post-hoc diagnostic for the current local virtual PPO checkpoint. It is not a paper-level BeyondMimic tracking result.

## Key Findings

- Multi-seed row count: `3`.
- Step-0 done-rate mean: `1.0`.
- Step-0 body-position error mean: `43.29219436645508`.
- All-step body-position error mean: `0.3597400628005382`.
- Post-step0 body-position error mean: `0.2156714241976706`.
- All-step done-rate mean: `0.1785340240036232`.
- Post-step0 done-rate mean: `0.175777426768736`.

## Interpretation

The current eval has a deterministic reset/bootstrap spike: every multi-seed run reports 2048/2048 done at step 0 and body-position error around 43m. Removing step 0 lowers the body-position mean substantially, but the post-step0 done rate remains high, so the next fix should target reset/target alignment and ee_body_pos termination rather than downstream VAE/diffusion reruns.

## Next Actions

- Inspect why the first eval step reports 100% done and roughly 43m body-position error despite the robot-order FK bundle.
- Run a controlled reset/alignment probe before policy actions: env.reset/get_observations/zero-action first step with motion IDs and body_pos_w diagnostics.
- Check whether the eval loop should discard the reset/bootstrap step or whether the environment reset target state is misaligned.
- Inspect ee_body_pos termination thresholds and endpoint body mapping because post-step0 done rate remains around 0.176 even after removing the reset spike.
- Do not collect final DAgger/VAE/diffusion data from this teacher until step-0 alignment and ee-body termination are understood.
