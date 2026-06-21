# Robot-Order FK Wrist Endpoint Source Full Diagnostic

Generated: 2026-06-21T23:48:14.115896+00:00

## Result

- Status: `ok_robot_order_fk_wrist_endpoint_source_full_diagnostic`
- Worker status: `ok_robot_order_fk_wrist_endpoint_source_full_diagnostic_worker`
- Scope: `2048` envs x `299` steps
- Done rate: `0.21957958821070234`
- ee_body_pos rate: `0.19802009301839466`
- Mean pre wrist done rate: `0.06626907399665552`
- Mean pre ankle done rate: `0.057275880539297656`
- Mean post wrist done rate: `0.06591470265468227`
- Mean post ankle done rate: `0.05699989548494983`

## Top Wrist Motions

| motion | sample count | done rate | ee rate | wrist pre exceed | ankle pre exceed | delta |
|---|---:|---:|---:|---:|---:|---:|
| fallAndGetUp1_subject4 | 11210 | 0.41882247992863514 | 0.39232827832292594 | 0.32212310437109726 | 0.004727921498661909 | 0.31739518287243534 |
| dance1_subject2 | 13494 | 0.4705795168222914 | 0.46020453534904404 | 0.3167333629761375 | 0.0 | 0.3167333629761375 |
| walk3_subject5 | 10083 | 0.4991569969255182 | 0.4948923931369632 | 0.2724387583060597 | 0.3700287612813647 | -0.09759000297530501 |
| walk2_subject4 | 13648 | 0.3733880422039859 | 0.34950175849941384 | 0.2627491207502931 | 0.0004396248534583822 | 0.2623094958968347 |
| fallAndGetUp2_subject3 | 12614 | 0.3386713175836372 | 0.3289995243380371 | 0.22641509433962265 | 0.14737593150467734 | 0.0790391628349453 |
| dance2_subject4 | 12943 | 0.435602256045739 | 0.435602256045739 | 0.20057173761879007 | 0.0 | 0.20057173761879007 |
| fightAndSports1_subject4 | 15670 | 0.2632418634333121 | 0.24097000638162094 | 0.13446075303126995 | 0.00835992342054882 | 0.12610082961072114 |
| walk3_subject4 | 15996 | 0.2105526381595399 | 0.18079519879969994 | 0.1159039759939985 | 0.00918979744936234 | 0.10671417854463616 |

## Checks

- `worker_returned_zero`: `True`
- `worker_status_ok`: `True`
- `uses_official_importer_export_usd`: `True`
- `uses_robot_order_fk_repaired_bundle`: `True`
- `checkpoint_loaded`: `True`
- `same_full_eval_scope_2048x299`: `True`
- `motion_count_40`: `True`
- `records_step_motion_phase_and_body_sources`: `True`
- `time_steps_unchanged_by_initial_refresh`: `True`
- `wrist_pre_exceed_rate_recorded`: `True`
- `wrist_post_exceed_rate_recorded`: `True`
- `wrist_pre_exceed_rate_exceeds_ankle`: `True`
- `wrist_post_exceed_rate_exceeds_ankle`: `True`
- `does_not_claim_paper_level_tracking`: `True`
- `does_not_claim_goal_complete`: `True`
- `does_not_claim_real_robot`: `True`
- `does_not_train`: `True`

This is a full-size local diagnostic eval. It does not train PPO, does not claim paper-level tracking, and does not use real robot hardware.
