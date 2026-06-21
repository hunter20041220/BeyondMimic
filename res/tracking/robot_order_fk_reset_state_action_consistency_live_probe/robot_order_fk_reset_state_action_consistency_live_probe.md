# Robot-Order FK Reset State/Action Consistency Live Probe

Generated: 2026-06-21T19:48:40.904765+00:00

## Result

- Status: `ok_robot_order_fk_reset_state_action_consistency_live_probe`
- Worker status: `ok_robot_order_fk_reset_state_action_consistency_live_probe`
- Diagnosis: `action_offset_alignment_does_not_improve_target_refresh`
- Recommended full-eval variant: ``
- GPU: `4`
- Num envs: `256`

## Policy-Step Comparison

| variant | pre endpoint done | pre body error | offset error | policy action mean | step done | post joint vel |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 0.546875 | 0.5228188037872314 | 2.6713693141937256 | 0.15179121494293213 | 0.55078125 | 9.915872573852539 |
| target_refresh | 0.28125 | 0.15145054459571838 | 2.7248573303222656 | 0.14669468998908997 | 0.28125 | 14.182840347290039 |
| target_refresh_action_reset | 0.48046875 | 0.1636754721403122 | 2.7884442806243896 | 0.13491879403591156 | 0.4765625 | 10.899185180664062 |
| target_refresh_action_offset | 0.4921875 | 0.16717492043972015 | 0.0 | 0.1302420198917389 | 0.49609375 | 10.263128280639648 |
| target_refresh_rewrite_motion_state | 0.55078125 | 0.16651389002799988 | 3.0701146125793457 | 0.15233783423900604 | 0.5546875 | 15.3351411819458 |
| target_refresh_rewrite_motion_state_action_reset | 0.66015625 | 0.17185541987419128 | 3.121330976486206 | 0.15074209868907928 | 0.6640625 | 11.64767837524414 |
| target_refresh_rewrite_motion_state_action_offset | 0.72265625 | 0.17640076577663422 | 0.0 | 0.1444307416677475 | 0.73828125 | 8.305423736572266 |

## Checks

- `worker_returned_zero`: `True`
- `worker_status_ok`: `True`
- `uses_official_importer_export_usd`: `True`
- `uses_robot_order_fk_repaired_bundle`: `True`
- `checkpoint_loaded`: `True`
- `all_variants_policy_and_zero_action_tested`: `True`
- `any_variant_improves_done_and_joint_velocity`: `False`
- `candidate_improves_done_rate`: `False`
- `candidate_improves_joint_velocity`: `True`
- `does_not_claim_paper_level_tracking`: `True`
- `does_not_claim_goal_complete`: `True`
- `does_not_claim_real_robot`: `True`
- `does_not_train`: `True`

This is a live local tracking diagnostic. It does not train PPO, does not claim paper-level tracking, and does not use real robot hardware.
