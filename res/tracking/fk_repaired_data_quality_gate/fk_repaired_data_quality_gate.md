# FK-Repaired Tracking Data Quality Gate

Status: `ok_fk_repaired_data_quality_gate`

This gate records why the old scaled-PPO chain is diagnostic-only and why the next PPO attempt should use the FK-repaired full public-motion bundle.

- Ready for FK-repaired full PPO attempt: `True`
- Paper-level tracking ready: `False`
- Old scaled chain trust level: `diagnostic_only_due_to_old_body_pos_w_degeneracy_and_endpoint_z_errors`

## Rows

| Item | Status | Reason |
| --- | --- | --- |
| `old_full_bundle` | `do_not_use_for_new_teacher_training` | body_pos_w and z-spread are degenerate/root-like |
| `fk_repaired_full_bundle` | `candidate_for_next_ppo` | 40 motions, 11960 frames, non-degenerate body_pos_w spread, ankles near ground |
| `fk_repaired_split_task_eval` | `passed_task_contract_but_runtime_order_misaligned` | 40/40 rows ok, total_done_count=11958, reward_mean=0.009514830887201243 |
| `fk_repaired_body_order_runtime_probe` | `root_cause_identified` | live MotionLoader indexing uses IsaacLab robot body order, not URDF order; max named-vs-loader z delta=0.9731605760753155 m |
| `fk_repaired_robot_order_bundle` | `candidate_for_next_ppo` | same FK targets reordered to IsaacLab runtime robot body order for official MotionLoader |
| `fk_repaired_robot_order_split_task_eval` | `passed_full_split_task_eval_with_lower_done_rate` | 40/40 rows ok, total_done_count=2166, done_rate=0.1811036789297659, reward_mean=0.016136537102128173, anchor/body error mean=0.08431245510000736/0.2135774338617921 |
| `old_scaled_ppo_chain` | `diagnostic_only` | endpoint z-error/termination diagnostics are retained, but the teacher chain should be rerun |
| `fk_repaired_full_bundle_ppo_training` | `completed_local_virtual_training` | 1000-iteration PPO completed on GPUs 4/7 with 21 checkpoints |
| `fk_repaired_full_bundle_ppo_eval` | `completed_but_not_downstream_ready` | done_count_total=612350, total_env_steps=612352, done_rate=0.9999967339046822, reward_mean=0.011290603055677884 |

## Next Action

Use the robot-order FK-repaired full bundle for the next PPO attempt. The runtime probe confirmed that the previous FK-repaired arrays were written in URDF order while MotionLoader indexes them in IsaacLab robot body order; the robot-order split eval substantially reduces zero-action done rate and body/anchor errors. Do not collect teacher rollouts from the older FK PPO checkpoint.
