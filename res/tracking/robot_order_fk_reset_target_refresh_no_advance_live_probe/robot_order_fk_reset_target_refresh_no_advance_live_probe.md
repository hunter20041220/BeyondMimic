# Robot-Order FK Reset Target Refresh No-Advance Probe

Generated: 2026-06-21T18:03:35.045645+00:00

## Result

- Status: `ok_robot_order_fk_reset_target_refresh_no_advance_live_probe`
- Worker status: `ok_robot_order_fk_reset_target_refresh_no_advance_live_probe`
- Diagnosis: `no_advance_target_refresh_partially_reduces_reset_endpoint_z_spike`
- Time steps unchanged by refresh: `True`
- Endpoint done-rate delta: `-0.7265625`
- Endpoint z mean delta: `-0.4255339950323105`

## Snapshot Metrics

| Stage | Endpoint done rate | Endpoint z mean (m) | Body error mean (m) | Time-step mean |
|---|---:|---:|---:|---:|
| after_reset_before_target_refresh | 1.0 | 0.5298784375190735 | 15.289739608764648 | 6196.125 |
| after_no_advance_target_refresh | 0.2734375 | 0.104344442486763 | 0.1527552455663681 | 6196.125 |
| after_zero_action_step_following_warmup | 0.06640625 | 0.08938737213611603 | 0.14660584926605225 | 6291.77734375 |

## Checks

- `worker_returned_zero`: `True`
- `worker_status_ok`: `True`
- `endpoint_names_found`: `True`
- `pre_refresh_manual_endpoint_done_rate_high`: `True`
- `refresh_reduces_endpoint_z_error_mean`: `True`
- `refresh_reduces_manual_endpoint_done_rate`: `True`
- `time_steps_unchanged_by_refresh`: `True`
- `zero_action_step_after_refresh_not_all_done`: `True`
- `does_not_claim_paper_level_tracking`: `True`
- `does_not_claim_goal_complete`: `True`
- `does_not_claim_real_robot`: `True`

This is a local live tracking data-quality gate. It is not PPO training, not DAgger, not paper-level closed-loop reproduction, and not a real-robot result.
