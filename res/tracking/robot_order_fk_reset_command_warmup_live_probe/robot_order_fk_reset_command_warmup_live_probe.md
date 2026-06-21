# Robot-Order FK Reset Command-Warmup Live Probe

Generated: 2026-06-21T16:22:21.742263+00:00

## Goal

Diagnose whether the robot-order FK tracking step-0 done spike is caused by stale/zero motion command targets immediately after reset.

## Result

- Status: `ok_robot_order_fk_reset_command_warmup_live_probe`
- Worker status: `ok_robot_order_fk_reset_command_warmup_live_probe`
- Diagnosis: `command_warmup_partially_reduces_reset_endpoint_z_spike`
- GPU: `4`
- Num envs: `256`
- Log: `/mnt/infini-data/test/BeyondMimic/logs/tracking_robot_order_fk_reset_command_warmup_live_probe/robot_order_fk_reset_command_warmup_live_probe.log`

## Snapshot Metrics

| Stage | Endpoint done rate | Endpoint z mean (m) | Endpoint z max (m) | Body error mean (m) | Body target abs max |
|---|---:|---:|---:|---:|---:|
| after_reset_before_command_warmup | 1.0 | 0.5298784375190735 | 1.194818377494812 | 15.289739608764648 | 0.0 |
| after_manual_command_manager_compute | 0.2734375 | 0.10452914237976074 | 0.7732915282249451 | 0.15273417532444 | 19.84893798828125 |
| after_zero_action_step_following_warmup | 0.06640625 | 0.0917641893029213 | 0.8092472553253174 | 0.1462995857000351 | 19.45673370361328 |

## Checks

- `does_not_claim_goal_complete`: `True`
- `does_not_claim_paper_level_tracking`: `True`
- `does_not_claim_real_robot`: `True`
- `endpoint_names_found`: `True`
- `num_envs_positive`: `True`
- `post_warmup_manual_endpoint_done_rate_low`: `False`
- `pre_warmup_manual_endpoint_done_rate_high`: `True`
- `uses_official_importer_export_usd`: `True`
- `uses_robot_order_fk_repaired_bundle`: `True`
- `warmup_reduces_endpoint_z_error_mean`: `True`
- `warmup_reduces_manual_endpoint_done_rate`: `True`
- `zero_action_step_after_warmup_not_all_done`: `True`

## Interpretation

Patch local tracking train/eval wrappers to warm command targets immediately after reset, then rerun full robot-order FK tracking eval/PPO.

This is a live diagnostic, not paper-level tracking reproduction, not PPO training, not DAgger, not VAE/diffusion guidance, and not a real-robot result.
