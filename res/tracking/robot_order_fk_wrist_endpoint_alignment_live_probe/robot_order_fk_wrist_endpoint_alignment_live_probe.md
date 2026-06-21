# Robot-Order FK Wrist Endpoint Alignment Live Probe

Generated: 2026-06-21T23:24:14.214109+00:00

## Result

- Status: `ok_robot_order_fk_wrist_endpoint_alignment_live_probe`
- Worker status: `ok_robot_order_fk_wrist_endpoint_alignment_live_probe`
- Diagnosis: `wrist_endpoint_target_or_body_semantics_remain_primary_done_source`
- GPU: `4`
- Num envs: `256`
- Threshold: `0.25` m

## Key Metrics

- Refresh wrist done rate: `0.15234375`
- Refresh ankle done rate: `0.09765625`
- Refresh wrist minus ankle done rate: `0.0546875`
- Policy-step wrist done rate: `0.09375`
- Policy-step ankle done rate: `0.06640625`
- Policy-step done rate: `0.30078125`

## Snapshot Table

| label | group | rel z mean | rel z max | raw z mean | target rel z mean | robot z mean | rel done rate | step done |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| zero_branch_after_reset_before_target_refresh | ankles | 0.10488246381282806 | 0.7843928933143616 | 0.06544189155101776 | 0.0 | 0.10336752235889435 | 0.16796875 | None |
| zero_branch_after_reset_before_target_refresh | wrists | 0.9518524408340454 | 1.1886333227157593 | 0.12762290239334106 | 0.0 | 0.9518524408340454 | 1.0 | None |
| zero_branch_after_reset_before_target_refresh | all_endpoints | 0.5283674597740173 | 1.1886333227157593 | 0.09653239697217941 | 0.0 | 0.5276099443435669 | 1.0 | None |
| zero_branch_after_target_refresh_no_advance | ankles | 0.06544189155101776 | 0.7387728691101074 | 0.06544189155101776 | 0.052611783146858215 | 0.10336752235889435 | 0.09765625 | None |
| zero_branch_after_target_refresh_no_advance | wrists | 0.12762290239334106 | 0.3902810215950012 | 0.12762290239334106 | 0.9263699054718018 | 0.9518524408340454 | 0.15234375 | None |
| zero_branch_after_target_refresh_no_advance | all_endpoints | 0.09653239697217941 | 0.7387728691101074 | 0.09653239697217941 | 0.4894908368587494 | 0.5276099443435669 | 0.22265625 | None |
| zero_branch_after_one_zero_action_step | ankles | 0.04942924529314041 | 0.4917140007019043 | 0.04942924529314041 | 0.05029856786131859 | 0.08992604166269302 | 0.03125 | 0.23046875 |
| zero_branch_after_one_zero_action_step | wrists | 0.11983097344636917 | 0.2887447476387024 | 0.11983097344636917 | 0.9431684017181396 | 0.9597175121307373 | 0.03515625 | 0.23046875 |
| zero_branch_after_one_zero_action_step | all_endpoints | 0.0846301019191742 | 0.4917140007019043 | 0.0846301019191742 | 0.49673348665237427 | 0.5248216986656189 | 0.0625 | 0.23046875 |
| policy_branch_after_reset_before_target_refresh | ankles | 0.0863366574048996 | 0.8546786904335022 | 0.0795983225107193 | 0.05029856786131859 | 0.1117391288280487 | 0.17578125 | None |
| policy_branch_after_reset_before_target_refresh | wrists | 0.17959390580654144 | 0.7379451394081116 | 0.1333570033311844 | 0.9431684017181396 | 0.9337573051452637 | 0.35546875 | None |
| policy_branch_after_reset_before_target_refresh | all_endpoints | 0.1329652965068817 | 0.8546786904335022 | 0.10647766292095184 | 0.49673348665237427 | 0.5227482318878174 | 0.453125 | None |
| policy_branch_after_target_refresh_no_advance | ankles | 0.0795983225107193 | 0.7429066300392151 | 0.0795983225107193 | 0.055326998233795166 | 0.1117391288280487 | 0.14453125 | None |
| policy_branch_after_target_refresh_no_advance | wrists | 0.1333570033311844 | 0.4283693730831146 | 0.1333570033311844 | 0.9001705646514893 | 0.9337573051452637 | 0.19140625 | None |
| policy_branch_after_target_refresh_no_advance | all_endpoints | 0.10647766292095184 | 0.7429066300392151 | 0.10647766292095184 | 0.4777488112449646 | 0.5227482318878174 | 0.2890625 | None |
| policy_branch_after_one_policy_step | ankles | 0.0617649219930172 | 0.6918848752975464 | 0.0617649219930172 | 0.05248325690627098 | 0.10256412625312805 | 0.06640625 | 0.30078125 |
| policy_branch_after_one_policy_step | wrists | 0.1263536810874939 | 0.4036562442779541 | 0.1263536810874939 | 0.9254447221755981 | 0.9504228830337524 | 0.09375 | 0.30078125 |
| policy_branch_after_one_policy_step | all_endpoints | 0.0940592885017395 | 0.6918848752975464 | 0.0940592885017395 | 0.4889640212059021 | 0.526493489742279 | 0.12109375 | 0.30078125 |

## Checks

- `worker_returned_zero`: `True`
- `worker_status_ok`: `True`
- `uses_official_importer_export_usd`: `True`
- `uses_robot_order_fk_repaired_bundle`: `True`
- `checkpoint_loaded`: `True`
- `ankle_names_found`: `True`
- `wrist_names_found`: `True`
- `time_steps_unchanged_by_refresh`: `True`
- `records_body_pos_w`: `True`
- `records_body_pos_relative_w`: `True`
- `records_robot_body_pos_w`: `True`
- `records_ankle_and_wrist_groups`: `True`
- `wrist_dominance_classified`: `True`
- `does_not_claim_paper_level_tracking`: `True`
- `does_not_claim_goal_complete`: `True`
- `does_not_claim_real_robot`: `True`
- `does_not_train`: `True`

This is a live local tracking data-quality diagnostic. It does not train PPO, does not claim paper-level tracking, and does not use real robot hardware.
