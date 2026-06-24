# Paper-Contract Transformer State-Latent Diffusion

Status: `blocked_paper_contract_transformer_diffusion_training_hard_gate`

Full training is blocked because the current state-latent data does not yet prove the paper hybrid-state contract.

## Blocking Reasons

- `pretraining_permission_start_state_latent_diffusion_training_is_False`
- `pretraining_hard_gate_status=blocked_pretraining_hard_gate_requires_teacher_and_adapter_fixes`
- `source_contract_disallows_existing_policy_obs_state_latent_dataset`
- `state_latent_source_contract_status=blocked_state_latent_dataset_source_uses_policy_obs_and_missing_rollout_state`
- `state_latent_dataset_uses_policy_obs:policy_obs in local paper-contract best-teacher rollout shards`
- `state_source_is_not_raw_hybrid_or_projected`
- `state_dim_not_paper_contract:160`
- `window_filter_does_not_prove_done_reset_rejection`
- `state_dim_not_99_or_163:160`
- `token_dim_not_state_plus_32_latent:192`
