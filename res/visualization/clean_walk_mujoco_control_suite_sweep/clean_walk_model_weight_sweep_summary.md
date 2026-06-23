# Clean Walk Model-Target-Weight Sweep

## 结论

The readable clean-walk videos are reference-anchored diagnostics. Pure or high-weight model-target control is not yet a credible Stage-1/VAE/diffusion success signal; the dominant blockers remain weak teacher quality and IsaacLab-to-MuJoCo obs/action contract fidelity.

## Claim Boundary

这是本地 MuJoCo 稳定性诊断。它不证明 BeyondMimic paper-level closed-loop rollout，也不是真实机器人结果。

## Sweep Results

| weight | status | primary zero fall | root height > 0.45 | unstable variants | output |
|---:|---|---:|---:|---|---|
| 0.20 | ok | True | True | none | `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_control_suite_sweep/w020` |
| 0.40 | ok | True | True | none | `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_control_suite_sweep/w040` |
| 0.60 | ok | True | True | none | `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_control_suite_sweep/w060` |
| 0.80 | ok | True | True | none | `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_control_suite_sweep/w080` |
| 1.00 | failed_unstable_variant | False | False | diffusion_denoised_latent_action_control, guided_latent_action_control, teacher_policy_action_control | `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_control_suite_sweep/w100` |

## Files

- JSON: `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_control_suite_sweep/clean_walk_model_weight_sweep_summary.json`
- CSV: `/mnt/infini-data/test/BeyondMimic/res/visualization/clean_walk_mujoco_control_suite_sweep/clean_walk_model_weight_sweep_summary.csv`
- Logs: `/mnt/infini-data/test/BeyondMimic/logs/mujoco/clean_walk_control_suite_sweep`
