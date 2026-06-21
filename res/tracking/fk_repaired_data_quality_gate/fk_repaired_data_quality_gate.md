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
| `fk_repaired_split_task_eval` | `passed_task_contract_but_zero_action` | 40/40 rows ok, total_done_count=11958, reward_mean=0.009514830887201243 |
| `old_scaled_ppo_chain` | `diagnostic_only` | endpoint z-error/termination diagnostics are retained, but the teacher chain should be rerun |
| `fk_repaired_full_bundle_ppo_training` | `completed_local_virtual_training` | 1000-iteration PPO completed on GPUs 4/7 with 21 checkpoints |
| `fk_repaired_full_bundle_ppo_eval` | `completed_but_not_downstream_ready` | done_count_total=612350, total_env_steps=612352, done_rate=0.9999967339046822, reward_mean=0.011290603055677884 |

## Next Action

Do not start teacher rollout/VAE/diffusion from the FK-repaired checkpoint yet. The full PPO/eval path now runs, but eval done/termination remains near one done per env-step. Next repair target is termination/reset/anchor alignment rather than more downstream training.
